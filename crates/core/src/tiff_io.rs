//! 16-bit TIFF input and output.
//!
//! Input is read lazily, one strip at a time, and converted to linear-light `f32`
//! on the way out — a full 50 MP RGB frame is 601 MB as `f32`, so nothing here ever
//! materializes one. Output re-encodes to sRGB before quantizing back to 16 bits.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{TiffEncoder, colortype};

use crate::image::{FrameInfo, linear_to_srgb, srgb_to_linear};

/// Rows buffered per output strip. Small enough to stay cheap, large enough that
/// strip overhead is negligible.
const OUT_ROWS_PER_STRIP: u32 = 64;

/// Roughly how many input rows the strip cache is allowed to hold.
const CACHE_ROWS: u32 = 256;

/// Read a frame's geometry without decoding any pixels.
pub fn probe(path: &Path) -> Result<FrameInfo> {
    Ok(FrameReader::open(path)?.info())
}

/// Lazy row reader over one 16-bit RGB TIFF.
///
/// Rows are served from a bounded FIFO cache of decoded strips, so a sequential band
/// walk decodes each strip once. The real stacks are uncompressed with one row per
/// strip (a row is a direct seek); the synthetic stack is Deflate with 36 rows per
/// strip, where the cache is what stops overlapping bands re-inflating the same data.
pub struct FrameReader {
    decoder: Decoder<BufReader<File>>,
    info: FrameInfo,
    rows_per_strip: u32,
    cache: VecDeque<(u32, Vec<f32>)>,
    cache_cap: usize,
}

impl FrameReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut decoder = Decoder::new(BufReader::new(file))
            .with_context(|| format!("reading TIFF header of {}", path.display()))?;

        let (width, height) = decoder.dimensions()?;
        let color = decoder.colortype()?;
        let ColorType::RGB(16) = color else {
            bail!(
                "{}: expected 16-bit RGB, found {color:?} — input must be 16-bit TIFF \
                 developed outside the app",
                path.display()
            );
        };

        let rows_per_strip = decoder.chunk_dimensions().1;
        ensure!(rows_per_strip > 0, "{}: zero-height strips", path.display());

        Ok(Self {
            decoder,
            info: FrameInfo {
                width,
                height,
                samples: 3,
                bits_per_sample: 16,
            },
            rows_per_strip,
            cache: VecDeque::new(),
            cache_cap: (CACHE_ROWS / rows_per_strip).max(1) as usize,
        })
    }

    pub fn info(&self) -> FrameInfo {
        self.info
    }

    /// Fill `out` with `count` rows of linear-light RGB starting at row `y0`.
    ///
    /// `out` must be exactly `count * info.row_len()` samples long.
    pub fn read_rows(&mut self, y0: u32, count: u32, out: &mut [f32]) -> Result<()> {
        let row_len = self.info.row_len();
        ensure!(
            out.len() == count as usize * row_len,
            "output buffer is {} samples, expected {}",
            out.len(),
            count as usize * row_len
        );
        ensure!(
            y0.checked_add(count).is_some_and(|e| e <= self.info.height),
            "rows {y0}..{} out of bounds for frame of height {}",
            y0 as u64 + count as u64,
            self.info.height
        );

        for i in 0..count {
            let y = y0 + i;
            let offset = (y % self.rows_per_strip) as usize * row_len;
            let strip = self.strip(y / self.rows_per_strip)?;
            out[i as usize * row_len..][..row_len].copy_from_slice(&strip[offset..][..row_len]);
        }
        Ok(())
    }

    /// Decoded strip `index` as linear-light `f32`, decoding it if it isn't cached.
    fn strip(&mut self, index: u32) -> Result<&[f32]> {
        let pos = match self.cache.iter().position(|(i, _)| *i == index) {
            Some(pos) => pos,
            None => {
                let DecodingResult::U16(raw) = self.decoder.read_chunk(index)? else {
                    bail!("expected 16-bit samples in strip {index}");
                };
                let samples = raw
                    .iter()
                    .map(|&s| srgb_to_linear(s as f32 / u16::MAX as f32))
                    .collect();
                if self.cache.len() == self.cache_cap {
                    self.cache.pop_front();
                }
                self.cache.push_back((index, samples));
                self.cache.len() - 1
            }
        };
        Ok(&self.cache[pos].1)
    }
}

/// Write a 16-bit RGB TIFF, pulling one row of linear-light RGB at a time.
///
/// The pull shape keeps the whole image from ever being resident: the caller hands
/// back rows as they are produced. Each row is re-encoded to sRGB and quantized.
pub fn write_rgb16_srgb(
    path: &Path,
    info: FrameInfo,
    mut fill_row: impl FnMut(u32, &mut [f32]) -> Result<()>,
) -> Result<()> {
    ensure!(
        info.samples == 3,
        "output must be RGB, got {} samples",
        info.samples
    );

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file))?;
    let mut image = encoder.new_image::<colortype::RGB16>(info.width, info.height)?;
    image.rows_per_strip(OUT_ROWS_PER_STRIP)?;

    let row_len = info.row_len();
    let mut row = vec![0f32; row_len];
    let mut strip: Vec<u16> = Vec::new();
    let mut y = 0;

    while y < info.height {
        let wanted = image.next_strip_sample_count() as usize;
        ensure!(
            wanted > 0 && wanted.is_multiple_of(row_len),
            "strip of {wanted} samples is not a whole number of {row_len}-sample rows"
        );
        strip.clear();
        strip.reserve(wanted);
        while strip.len() < wanted {
            fill_row(y, &mut row)?;
            strip.extend(row.iter().map(|&v| quantize(v)));
            y += 1;
        }
        image.write_strip(&strip)?;
    }

    image.finish()?;
    Ok(())
}

fn quantize(linear: f32) -> u16 {
    (linear_to_srgb(linear).clamp(0.0, 1.0) * u16::MAX as f32).round() as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gradient that exercises shadows, midtones and highlights.
    fn sample(x: u32, y: u32, c: usize) -> f32 {
        let v = (x as f32 * 0.05 + y as f32 * 0.11 + c as f32 * 0.3) % 1.0;
        srgb_to_linear(v)
    }

    fn write_fixture(path: &Path, info: FrameInfo) {
        write_rgb16_srgb(path, info, |y, row| {
            for x in 0..info.width {
                for c in 0..3 {
                    row[x as usize * 3 + c] = sample(x, y, c);
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        // Height deliberately not a multiple of OUT_ROWS_PER_STRIP.
        let info = FrameInfo {
            width: 7,
            height: 100,
            samples: 3,
            bits_per_sample: 16,
        };
        write_fixture(&path, info);

        assert_eq!(probe(&path).unwrap(), info);

        let mut reader = FrameReader::open(&path).unwrap();
        let mut got = vec![0f32; info.row_len() * info.height as usize];
        reader.read_rows(0, info.height, &mut got).unwrap();

        for y in 0..info.height {
            for x in 0..info.width {
                for c in 0..3 {
                    let want = sample(x, y, c);
                    let have = got[y as usize * info.row_len() + x as usize * 3 + c];
                    assert!(
                        (have - want).abs() < 1e-3,
                        "at {x},{y},{c}: {have} != {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn reads_bands_out_of_order_consistently() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        let info = FrameInfo {
            width: 7,
            height: 100,
            samples: 3,
            bits_per_sample: 16,
        };
        write_fixture(&path, info);

        let mut reader = FrameReader::open(&path).unwrap();
        let mut whole = vec![0f32; info.row_len() * info.height as usize];
        reader.read_rows(0, info.height, &mut whole).unwrap();

        // Overlapping, backwards bands must agree with the single full read — this is
        // the access pattern banded fusion actually uses.
        for &(y0, n) in &[(90u32, 10u32), (40, 25), (55, 20), (0, 1)] {
            let mut band = vec![0f32; info.row_len() * n as usize];
            reader.read_rows(y0, n, &mut band).unwrap();
            assert_eq!(band, whole[y0 as usize * info.row_len()..][..band.len()]);
        }
    }

    /// Exercises both real strip layouts: uncompressed 1-row strips (ruler/blossom)
    /// and Deflate 36-row strips (synthetic_50). Ignored by default because
    /// `test-data/` is gitignored; run with `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires test-data/"]
    fn reads_real_stacks() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data");
        for (stack, frame, w, h) in [
            ("ruler", "A1_00001_01.tif", 8664, 5784),
            ("blossom", "A1_00001.tif", 8664, 5784),
            ("synthetic_50", "frame_001.tiff", 1200, 900),
        ] {
            let path = root.join(stack).join(frame);
            assert!(path.is_file(), "missing {}", path.display());

            let info = probe(&path).unwrap();
            assert_eq!((info.width, info.height), (w, h), "{stack}");

            let mut reader = FrameReader::open(&path).unwrap();
            let mut band = vec![0f32; info.row_len() * 8];
            reader.read_rows(h / 2, 8, &mut band).unwrap();
            assert!(
                band.iter().all(|v| (0.0..=1.0).contains(v)),
                "{stack}: linear samples outside [0,1]"
            );
            assert!(band.iter().any(|&v| v > 0.0), "{stack}: band is all zero");
        }
    }

    #[test]
    fn rejects_reads_past_the_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        let info = FrameInfo {
            width: 7,
            height: 20,
            samples: 3,
            bits_per_sample: 16,
        };
        write_fixture(&path, info);

        let mut reader = FrameReader::open(&path).unwrap();
        let mut buf = vec![0f32; info.row_len() * 5];
        assert!(reader.read_rows(18, 5, &mut buf).is_err());
        assert!(
            reader.read_rows(0, 4, &mut buf).is_err(),
            "buffer length mismatch"
        );
    }
}
