//! 16-bit TIFF input and output.
//!
//! Input is read lazily, one strip at a time, and converted to linear-light `f32`
//! on the way out — a full 50 MP RGB frame is 601 MB as `f32`, so nothing here ever
//! materializes one. Cached strips stay in their on-disk `u16`, which halves what a
//! reader holds; the conversion happens as rows are copied out, through [`srgb_lut`].
//! Output re-encodes to sRGB before quantizing back to 16 bits.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{TiffEncoder, colortype};

use crate::error::{Error, Result};
use crate::image::{FrameInfo, linear_to_srgb, srgb_to_linear};

/// Rows buffered per output strip. Small enough to stay cheap, large enough that
/// strip overhead is negligible.
const OUT_ROWS_PER_STRIP: u32 = 64;

/// Roughly how many input rows the strip cache is allowed to hold.
const CACHE_ROWS: u32 = 256;

/// Linear light for every 16-bit sample value, indexed by the sample itself.
///
/// **Both halves of this matter.** Cached strips hold raw `u16`, so the transfer function
/// is applied once per sample *copied out* rather than once per sample *decoded* — which
/// on its own would be a regression, because `srgb_to_linear` calls `powf`, and
/// `decode_cost.rs` measures that conversion as the CPU-bound part of decoding. A table
/// removes the call entirely: the domain is 65536 values wide, so it is enumerable, and
/// 256 KB is nothing beside the strips it lets us halve.
///
/// Exact, not approximate. Every entry is `srgb_to_linear` evaluated at the same argument
/// the old per-sample path used, so output is bit-identical — asserted over the whole
/// domain by the `the_lut_is_exact_over_every_sample_value` test below.
fn srgb_lut() -> &'static [f32; 1 << 16] {
    static LUT: OnceLock<Box<[f32; 1 << 16]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = Box::new([0f32; 1 << 16]);
        for (v, slot) in lut.iter_mut().enumerate() {
            *slot = srgb_to_linear(v as f32 / u16::MAX as f32);
        }
        lut
    })
}

/// The largest single chunk the decoder is allowed to allocate for, in bytes.
///
/// A 350 MP RGB16 frame in one strip, well past anything this is built for, and small
/// enough that a corrupt header claiming absurd dimensions is refused rather than
/// turned into an allocation.
const MAX_CHUNK_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Decoder limits for a file whose largest chunk is `chunk_bytes`.
///
/// **This exists because the crate's defaults reject legitimate files.** `tiff`'s
/// `Limits::default()` caps a chunk at 256 MB, which is fine for the stacks here — they
/// are one row per strip — and wrong for any exporter that writes the image as a single
/// strip, which is common. A 48 MP RGB16 frame is 288 MB in one strip, so every frame
/// fails with "decoder limits exceeded" before a pixel is read. Reported from a Windows
/// build, reproduced immediately on macOS: nothing about it is platform-specific.
///
/// Sized to the file rather than `Limits::unlimited()`, so a corrupt header still cannot
/// ask for an arbitrary allocation. The slack is because the crate compares an element
/// *count* against these byte budgets after dividing by the sample size, and an exact
/// fit lands on the boundary.
fn chunk_limits(chunk_bytes: u64) -> Option<tiff::decoder::Limits> {
    if chunk_bytes > MAX_CHUNK_BYTES {
        return None;
    }
    // Never *below* the crate's own defaults: this is only ever meant to raise the
    // ceiling for a large chunk, not to tighten it for a small one.
    let mut limits = tiff::decoder::Limits::default();
    let budget = (chunk_bytes + (1 << 20)).max(limits.decoding_buffer_size as u64);
    // Field-by-field because `Limits` is `#[non_exhaustive]`, so a struct expression
    // does not compile from outside the crate. `ifd_value_size` keeps its default.
    limits.decoding_buffer_size = budget as usize;
    limits.intermediate_buffer_size = budget as usize;
    Some(limits)
}

/// Read a frame's geometry without decoding any pixels.
pub fn probe(path: &Path) -> Result<FrameInfo> {
    Ok(FrameReader::open(path)?.info())
}

/// What one reader over this file costs while it is in flight, in bytes.
///
/// Reads the header only. Taken by path because the stage runners that size their
/// concurrency from it ([`crate::registration::register_stack`],
/// [`crate::focus::evaluate_stack`]) hold paths, not open frames — the whole point is to
/// decide how many readers to have before opening any.
pub fn cache_bytes_max(path: &Path) -> Result<u64> {
    Ok(FrameReader::open(path)?.cache_bytes_max())
}

/// Lazy row reader over one 16-bit RGB TIFF.
///
/// Rows are served from a bounded FIFO cache of decoded strips, so a sequential band
/// walk decodes each strip once. The real stacks are uncompressed with one row per
/// strip (a row is a direct seek); the synthetic stack is Deflate with 36 rows per
/// strip, where the cache is what stops overlapping bands re-inflating the same data.
///
/// **The cache holds raw `u16`, not linear `f32`, and that is a memory decision.** A
/// strip is the decoder's atomic unit, so a frame written as a *single* strip — common
/// from exporters, and the shape behind the v1.0.2 bug — is entirely resident while its
/// rows are read. At 8664x5784 RGB16 that is 300 MB stored as `u16` against 601 MB
/// stored as `f32`, and the pipeline holds several such readers at once. The transfer
/// function moves to the copy-out in [`Self::read_rows`], where [`srgb_lut`] makes it a
/// table lookup rather than the `powf` it used to be.
pub struct FrameReader {
    path: PathBuf,
    decoder: Decoder<BufReader<File>>,
    info: FrameInfo,
    rows_per_strip: u32,
    cache: VecDeque<(u32, Vec<u16>)>,
    cache_cap: usize,
}

impl FrameReader {
    pub fn open(path: &Path) -> Result<Self> {
        let decode = |source| Error::Decode {
            path: path.to_path_buf(),
            source,
        };

        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        let mut decoder = Decoder::new(BufReader::new(file)).map_err(decode)?;

        let (width, height) = decoder.dimensions().map_err(decode)?;
        let color = decoder.colortype().map_err(decode)?;
        let ColorType::RGB(16) = color else {
            return Err(Error::UnsupportedFormat {
                path: path.to_path_buf(),
                found: format!("{color:?}"),
            });
        };

        let rows_per_strip = decoder.chunk_dimensions().1;
        if rows_per_strip == 0 {
            return Err(Error::UnsupportedFormat {
                path: path.to_path_buf(),
                found: "zero-height strips".into(),
            });
        }

        // After the geometry is known, because the budget is derived from it, and before
        // any pixels are read, because that is what it governs.
        let chunk_bytes = width as u64 * rows_per_strip as u64 * 3 * 2;
        let Some(limits) = chunk_limits(chunk_bytes) else {
            return Err(Error::UnsupportedFormat {
                path: path.to_path_buf(),
                found: format!("{} MB in a single chunk", chunk_bytes / (1 << 20)),
            });
        };
        let decoder = decoder.with_limits(limits);

        Ok(Self {
            path: path.to_path_buf(),
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

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Fill `out` with `count` rows of linear-light RGB starting at row `y0`.
    ///
    /// `out` must be exactly `count * info.row_len()` samples long.
    pub fn read_rows(&mut self, y0: u32, count: u32, out: &mut [f32]) -> Result<()> {
        let row_len = self.info.row_len();
        let want = count as usize * row_len;
        if out.len() != want {
            return Err(Error::BufferSize {
                got: out.len(),
                want,
            });
        }
        let end = y0 as u64 + count as u64;
        if end > self.info.height as u64 {
            return Err(Error::Bounds {
                start: y0 as u64,
                end,
                height: self.info.height,
            });
        }

        let lut = srgb_lut();
        for i in 0..count {
            let y = y0 + i;
            let offset = (y % self.rows_per_strip) as usize * row_len;
            let strip = self.strip(y / self.rows_per_strip)?;
            let dst = &mut out[i as usize * row_len..][..row_len];
            for (slot, &sample) in dst.iter_mut().zip(&strip[offset..][..row_len]) {
                *slot = lut[sample as usize];
            }
        }
        Ok(())
    }

    /// The most the strip cache can ever hold, in bytes.
    ///
    /// `cache_cap` strips of `rows_per_strip` rows. For the striped stacks this is about
    /// [`CACHE_ROWS`] rows — 13 MB on a 50 MP frame — and for a single-strip frame it is
    /// the whole frame, 300 MB. That gap is the entire reason
    /// [`crate::budget`] exists: it is what one reader costs while it is in flight.
    pub fn cache_bytes_max(&self) -> u64 {
        self.cache_cap as u64 * self.rows_per_strip as u64 * self.info.row_len() as u64 * 2
    }

    /// Bytes the strip cache is currently holding.
    ///
    /// Exposed so callers can assert what they hold rather than infer it from a
    /// process-level measurement — see [`crate::pipeline::Image::cache_bytes`].
    pub fn cache_bytes(&self) -> usize {
        self.cache.iter().map(|(_, s)| size_of_val(&s[..])).sum()
    }

    /// Drop every cached strip.
    ///
    /// For a caller that knows a frame is finished with. Correctness never depends on
    /// this — a released strip is decoded again on the next read — only memory does.
    pub fn release_cache(&mut self) {
        self.cache.clear();
        self.cache.shrink_to_fit();
    }

    /// Decoded strip `index` as raw `u16` samples, decoding it if it isn't cached.
    fn strip(&mut self, index: u32) -> Result<&[u16]> {
        let pos = match self.cache.iter().position(|(i, _)| *i == index) {
            Some(pos) => pos,
            None => {
                let chunk = self
                    .decoder
                    .read_chunk(index)
                    .map_err(|source| Error::Decode {
                        path: self.path.clone(),
                        source,
                    })?;
                let DecodingResult::U16(raw) = chunk else {
                    return Err(Error::UnsupportedFormat {
                        path: self.path.clone(),
                        found: "non-16-bit samples".into(),
                    });
                };
                if self.cache.len() == self.cache_cap {
                    self.cache.pop_front();
                }
                self.cache.push_back((index, raw));
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
    if info.samples != 3 {
        return Err(Error::UnsupportedFormat {
            path: path.to_path_buf(),
            found: format!("{} samples per pixel on output", info.samples),
        });
    }
    let encode = |source| Error::Encode {
        path: path.to_path_buf(),
        source,
    };

    let file = File::create(path).map_err(|e| Error::io(path, e))?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file)).map_err(encode)?;
    let mut image = encoder
        .new_image::<colortype::RGB16>(info.width, info.height)
        .map_err(encode)?;
    image.rows_per_strip(OUT_ROWS_PER_STRIP).map_err(encode)?;

    let row_len = info.row_len();
    let mut row = vec![0f32; row_len];
    let mut strip: Vec<u16> = Vec::new();
    let mut y = 0;

    while y < info.height {
        let wanted = image.next_strip_sample_count() as usize;
        if wanted == 0 || !wanted.is_multiple_of(row_len) {
            return Err(Error::BufferSize {
                got: wanted,
                want: row_len,
            });
        }
        strip.clear();
        strip.reserve(wanted);
        while strip.len() < wanted {
            fill_row(y, &mut row)?;
            strip.extend(row.iter().map(|&v| quantize(v)));
            y += 1;
        }
        image.write_strip(&strip).map_err(encode)?;
    }

    image.finish().map_err(encode)?;
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

    /// Write a TIFF whose every row lives in one strip, the way many exporters do.
    ///
    /// `write_rgb16_srgb` cannot produce this — it strips at [`OUT_ROWS_PER_STRIP`] —
    /// and the shape is the whole point of these tests, so the encoder is driven here
    /// directly.
    fn write_single_strip(path: &Path, width: u32, height: u32, data: &[u16]) {
        let file = File::create(path).unwrap();
        let mut encoder = TiffEncoder::new(BufWriter::new(file)).unwrap();
        let mut image = encoder
            .new_image::<colortype::RGB16>(width, height)
            .unwrap();
        image.rows_per_strip(height).unwrap();
        image.write_data(data).unwrap();
    }

    fn synthetic_samples(width: u32, height: u32) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| (0..3).map(move |c| ((x + y + c) % 65536) as u16))
            })
            .collect()
    }

    /// A frame in one strip decodes. Regression: it used to fail outright once the
    /// strip passed 256 MB, which is a 48 MP frame — reported from the wild.
    ///
    /// Small here on purpose. The size that actually tripped the limit cannot go in a
    /// test that runs on every push, so the arithmetic that decides the budget is
    /// tested separately below, and this covers the decode path itself.
    #[test]
    fn reads_a_single_strip_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("single.tif");
        write_single_strip(&path, 64, 48, &synthetic_samples(64, 48));

        let mut reader = FrameReader::open(&path).unwrap();
        assert_eq!(reader.info().height, 48);
        assert_eq!(reader.rows_per_strip, 48, "fixture is not a single strip");

        let mut got = vec![0f32; reader.info().row_len() * 48];
        reader.read_rows(0, 48, &mut got).unwrap();
        assert!(got.iter().all(|v| v.is_finite()));
    }

    /// The budget clears what the crate's own default would have rejected.
    ///
    /// 8000x6000 RGB16 in one strip is 288 MB, against a 256 MB default — the exact
    /// shape of the reported failure, asserted without allocating 288 MB.
    #[test]
    fn a_large_single_chunk_is_allowed_where_the_default_refuses_it() {
        let chunk = 8000u64 * 6000 * 3 * 2;
        let default = tiff::decoder::Limits::default().decoding_buffer_size as u64;
        assert!(chunk > default, "fixture no longer exceeds the default");

        let limits = chunk_limits(chunk).expect("288 MB is within the cap");
        assert!(limits.decoding_buffer_size as u64 > chunk);
        assert!(limits.intermediate_buffer_size as u64 > chunk);
    }

    /// A small file does not get a *smaller* budget than the crate's default.
    #[test]
    fn a_small_chunk_keeps_the_default_budget() {
        let default = tiff::decoder::Limits::default();
        let limits = chunk_limits(4096).unwrap();
        assert_eq!(limits.decoding_buffer_size, default.decoding_buffer_size);
    }

    /// Sized to the file, so a corrupt header cannot ask for an arbitrary allocation.
    #[test]
    fn an_absurd_chunk_is_refused() {
        assert!(chunk_limits(MAX_CHUNK_BYTES).is_some());
        assert!(chunk_limits(MAX_CHUNK_BYTES + 1).is_none());
    }

    /// The table is the conversion, not an approximation of it.
    ///
    /// Every entry, over the whole 16-bit domain, compared bit-for-bit against the
    /// expression the per-sample path used to evaluate. This is what lets the pinned
    /// output hash stand across the switch to a `u16` cache: if these ever disagree, the
    /// fused image changes, and it should fail here rather than in a rating.
    #[test]
    fn the_lut_is_exact_over_every_sample_value() {
        let lut = srgb_lut();
        for v in 0..=u16::MAX {
            let want = srgb_to_linear(v as f32 / u16::MAX as f32);
            assert_eq!(lut[v as usize].to_bits(), want.to_bits(), "at {v}");
        }
    }

    /// Releasing is a memory operation, not a correctness one: the same rows come back.
    #[test]
    fn releasing_the_cache_frees_it_and_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        let info = FrameInfo {
            width: 7,
            height: 40,
            samples: 3,
            bits_per_sample: 16,
        };
        write_fixture(&path, info);

        let mut reader = FrameReader::open(&path).unwrap();
        let mut first = vec![0f32; info.row_len() * 8];
        reader.read_rows(16, 8, &mut first).unwrap();
        assert!(reader.cache_bytes() > 0);
        assert!(reader.cache_bytes() as u64 <= reader.cache_bytes_max());

        reader.release_cache();
        assert_eq!(reader.cache_bytes(), 0);

        let mut again = vec![0f32; info.row_len() * 8];
        reader.read_rows(16, 8, &mut again).unwrap();
        assert_eq!(first, again);
    }

    /// A single-strip frame charges the whole frame; a striped one charges a band.
    ///
    /// This ratio is the only thing [`crate::budget`] acts on, so it is asserted rather
    /// than assumed. Striped is bounded by [`CACHE_ROWS`] rows however tall the frame
    /// gets; single-strip grows with the frame, which is what makes it the case worth
    /// bounding — at the real 5784 rows it is ~23x the striped charge, against the 1.17x
    /// visible at this deliberately tiny fixture size.
    #[test]
    fn a_single_strip_frame_costs_a_whole_frame_to_hold() {
        let dir = tempfile::tempdir().unwrap();
        let info = FrameInfo {
            width: 64,
            height: 300,
            samples: 3,
            bits_per_sample: 16,
        };
        let row_bytes = info.row_len() as u64 * 2;

        let single = dir.path().join("single.tif");
        write_single_strip(
            &single,
            info.width,
            info.height,
            &synthetic_samples(info.width, info.height),
        );
        assert_eq!(
            cache_bytes_max(&single).unwrap(),
            row_bytes * info.height as u64
        );

        // `write_fixture` strips at `OUT_ROWS_PER_STRIP`, so the cache holds
        // `CACHE_ROWS / OUT_ROWS_PER_STRIP` of them and stops there.
        let striped = dir.path().join("striped.tif");
        write_fixture(&striped, info);
        assert_eq!(
            cache_bytes_max(&striped).unwrap(),
            row_bytes * (CACHE_ROWS - CACHE_ROWS % OUT_ROWS_PER_STRIP) as u64
        );
        assert!(cache_bytes_max(&single).unwrap() > cache_bytes_max(&striped).unwrap());
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

    /// Rebuild `test-data/fixtures/blossom_single_strip/` — 33 blossom frames, same
    /// pixels, one strip each.
    ///
    /// ```text
    /// cargo test --release -p stackaroni-core tiff_io -- --ignored builds_the_single_strip_fixture
    /// ```
    ///
    /// **This is a fixture builder, not an assertion.** The reported memory failure only
    /// appears on single-strip input, no exporter here produces it, and the stack that
    /// exposed it belongs to the reporter. Deriving it from blossom means the striped and
    /// single-strip measurements differ in strip layout and *nothing else* — the same
    /// pixels, so any gap between them is the layout and not the content.
    ///
    /// **Under `fixtures/`, one level below where the eval set is scanned, and that is
    /// load-bearing.** `discover_test_set` turns every directory holding TIFFs directly
    /// under `test-data/` into a stack, with no name filtering — so a fixture placed
    /// beside `blossom` silently joins the fixed comparison set that `CLAUDE.md` says
    /// must not change. That is not hypothetical: it happened, and `fuse_all_stacks`
    /// fused all 33 frames and left a `stackaroni_fused.tif` in the fixture directory.
    /// `fixtures/` itself holds no TIFFs, so the scan skips it.
    ///
    /// Idempotent: a frame already written is left alone, so an interrupted run resumes.
    /// The output is ~9.6 GB and `test-data/` is gitignored.
    #[test]
    #[ignore = "requires test-data/blossom; writes ~9.6 GB"]
    fn builds_the_single_strip_fixture() {
        const FRAMES: usize = 33;

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data");
        let src_dir = root.join("blossom");
        assert!(src_dir.is_dir(), "missing {}", src_dir.display());
        let dst_dir = root.join("fixtures").join("blossom_single_strip");
        std::fs::create_dir_all(&dst_dir).unwrap();

        let mut sources: Vec<PathBuf> = std::fs::read_dir(&src_dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("tif")))
            .collect();
        sources.sort();
        assert!(sources.len() >= FRAMES, "blossom has fewer than {FRAMES}");

        for src in &sources[..FRAMES] {
            let dst = dst_dir.join(src.file_name().unwrap());
            if dst.is_file() {
                continue;
            }
            let info = probe(src).unwrap();

            // Read every sample as it sits on disk. Not through `FrameReader`: that
            // converts to linear light, and quantizing back would put an sRGB round trip
            // between the two fixtures — a difference in *pixels*, which is exactly what
            // this must not introduce.
            let file = File::open(src).unwrap();
            let whole = info.width as u64 * info.height as u64 * 3 * 2;
            let mut decoder = Decoder::new(BufReader::new(file))
                .unwrap()
                .with_limits(chunk_limits(whole).expect("frame fits the chunk cap"));
            let DecodingResult::U16(data) = decoder.read_image().unwrap() else {
                panic!("{}: not 16-bit", src.display());
            };

            // Written beside the target and renamed, so an interrupted run cannot leave a
            // truncated frame that the skip above would then treat as done.
            let partial = dst.with_extension("partial");
            write_single_strip(&partial, info.width, info.height, &data);
            std::fs::rename(&partial, &dst).unwrap();

            let check = FrameReader::open(&dst).unwrap();
            assert_eq!(
                check.rows_per_strip,
                info.height,
                "{} is striped",
                dst.display()
            );
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
        assert!(matches!(
            reader.read_rows(18, 5, &mut buf),
            Err(Error::Bounds { .. })
        ));
        assert!(matches!(
            reader.read_rows(0, 4, &mut buf),
            Err(Error::BufferSize { .. })
        ));
    }

    #[test]
    fn rejects_non_tiff_input() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not.tif");
        std::fs::write(&path, b"definitely not a tiff").unwrap();
        assert!(matches!(probe(&path), Err(Error::Decode { .. })));
    }
}
