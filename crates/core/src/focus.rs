//! Windowed-Laplacian focus measurement.
//!
//! Implements the windowed Laplacian energy of `docs/algorithms.md` §3:
//!
//! ```text
//! F(x,y) = sum over window W of  ( laplacian(I)(i,j) )^2
//! ```
//!
//! The window is the point. A bare second derivative is extremely noise-sensitive,
//! which undercuts the robustness the measure exists to provide, so standard practice
//! aggregates over a small neighbourhood (Pertuz, Puig & Garcia, *Pattern Recognition*
//! 46(5), 2013, 1415-1432).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::filter::box_sum;
use crate::image::{FrameInfo, ScratchPlane};
use crate::pipeline::{FocusMap, FocusMetric, Image, Transform};

/// Rec. 709 luma coefficients, applied to linear-light RGB.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Output rows computed per pass. Keeps the working buffers to a few MB on a 50 MP
/// frame while amortizing the halo re-read.
const BAND_ROWS: u32 = 256;

/// Windowed Laplacian energy over a `(2*radius+1)` square window.
///
/// Frames and their transforms are supplied at construction: [`FocusMetric::evaluate`]
/// receives only an `&Image`, so the transform for that frame is looked up by path.
/// Same constructor-injection pattern as the fusion output path.
pub struct WindowedLaplacian {
    radius: u32,
    scratch: PathBuf,
    transforms: HashMap<PathBuf, Transform>,
}

impl WindowedLaplacian {
    pub fn new(radius: u32, scratch: &Path, transforms: HashMap<PathBuf, Transform>) -> Self {
        Self {
            radius,
            scratch: scratch.to_path_buf(),
            transforms,
        }
    }

    /// Focus energy in the frame's own coordinates, before alignment.
    fn measure_native(&self, image: &Image, path: &Path) -> Result<ScratchPlane> {
        let info = image.info();
        let mut plane = ScratchPlane::create(path, info.width, info.height)?;
        let halo = self.radius + 1;

        let mut y0 = 0;
        while y0 < info.height {
            let rows = BAND_ROWS.min(info.height - y0);
            // Read the band plus its halo, clamped to the frame.
            let read_from = y0.saturating_sub(halo);
            let read_to = (y0 + rows + halo).min(info.height);
            let read_rows = read_to - read_from;

            let mut band = vec![0f32; info.row_len() * read_rows as usize];
            image.read_rows(read_from, read_rows, &mut band)?;

            let luma = to_luma(&band, info, read_rows);
            let energy = laplacian_energy(&luma, info.width, read_rows);
            let summed = box_sum(&energy, info.width, read_rows, self.radius);

            let out = plane.rows_mut(y0, rows)?;
            for r in 0..rows {
                // Index of this output row inside the halo-extended band.
                let src = (y0 + r - read_from) as usize;
                let dst = r as usize * info.width as usize;
                out[dst..dst + info.width as usize]
                    .copy_from_slice(&summed[src * info.width as usize..][..info.width as usize]);
            }
            y0 += rows;
        }
        Ok(plane)
    }
}

impl FocusMetric for WindowedLaplacian {
    /// Measure focus, then warp the *map* into anchor coordinates.
    ///
    /// Deliberately not "warp the frame, then measure". Resampling an image is a
    /// low-pass operation whose severity depends on the sub-pixel phase of that
    /// frame's shift, so measuring afterwards would report frames as sharper or
    /// softer according to their registration offset rather than their actual focus
    /// — a bias that varies frame to frame and would steer frame selection directly.
    /// Measuring first keeps the pristine samples; the map is then a smooth,
    /// slowly-varying field where interpolation is cheap and unbiased.
    fn evaluate(&self, image: &Image) -> Result<FocusMap> {
        let info = image.info();
        let stem = file_stem(image.path());
        let native_path = self.scratch.join(format!("{stem}.native.f32"));
        let native = self.measure_native(image, &native_path)?;

        let transform = self
            .transforms
            .get(image.path())
            .copied()
            .unwrap_or(Transform::IDENTITY);

        let mut aligned = ScratchPlane::create(
            &self.scratch.join(format!("{stem}.focus.f32")),
            info.width,
            info.height,
        )?;
        warp_plane(&native, &mut aligned, transform, info)?;

        drop(native);
        let _ = std::fs::remove_file(&native_path);
        Ok(aligned)
    }
}

/// Resample `src` into `dst` under `transform`, which maps anchor coordinates onto
/// the frame's own — so each destination pixel reads straight through it.
fn warp_plane(
    src: &ScratchPlane,
    dst: &mut ScratchPlane,
    transform: Transform,
    info: FrameInfo,
) -> Result<()> {
    let (cx, cy) = (info.width as f32 / 2.0, info.height as f32 / 2.0);
    let mut y0 = 0;
    while y0 < info.height {
        let rows = BAND_ROWS.min(info.height - y0);
        let out = dst.rows_mut(y0, rows)?;
        for r in 0..rows {
            for x in 0..info.width {
                let (sx, sy) = transform.apply(x as f32 - cx, (y0 + r) as f32 - cy);
                out[r as usize * info.width as usize + x as usize] = src.sample(sx + cx, sy + cy);
            }
        }
        y0 += rows;
    }
    Ok(())
}

fn to_luma(band: &[f32], info: FrameInfo, rows: u32) -> Vec<f32> {
    let mut luma = vec![0f32; info.width as usize * rows as usize];
    for (i, out) in luma.iter_mut().enumerate() {
        let p = i * 3;
        *out = LUMA[0] * band[p] + LUMA[1] * band[p + 1] + LUMA[2] * band[p + 2];
    }
    luma
}

/// Squared 4-neighbour Laplacian, edges clamped.
fn laplacian_energy(luma: &[f32], width: u32, rows: u32) -> Vec<f32> {
    let (w, h) = (width as i64, rows as i64);
    let at = |x: i64, y: i64| luma[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize];

    let mut out = vec![0f32; luma.len()];
    for y in 0..h {
        for x in 0..w {
            let lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            out[(y * w + x) as usize] = lap * lap;
        }
    }
    out
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "frame".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiff_io::write_rgb16_srgb;

    fn write_frame(path: &Path, width: u32, height: u32, f: impl Fn(u32, u32) -> f32) {
        let info = FrameInfo {
            width,
            height,
            samples: 3,
            bits_per_sample: 16,
        };
        write_rgb16_srgb(path, info, |y, row| {
            for x in 0..width {
                row[x as usize * 3..][..3].fill(f(x, y));
            }
            Ok(())
        })
        .unwrap();
    }

    fn evaluate(path: &Path, scratch: &Path, radius: u32) -> FocusMap {
        let metric = WindowedLaplacian::new(radius, scratch, HashMap::new());
        metric.evaluate(&Image::open(path).unwrap()).unwrap()
    }

    #[test]
    fn flat_regions_measure_zero_focus() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flat.tif");
        write_frame(&path, 64, 48, |_, _| 0.4);

        let map = evaluate(&path, dir.path(), 3);
        for &v in map.rows(0, 48).unwrap() {
            assert!(
                v.abs() < 1e-6,
                "flat region should have no focus energy: {v}"
            );
        }
    }

    #[test]
    fn sharp_edges_measure_more_than_soft_ones() {
        let dir = tempfile::tempdir().unwrap();
        let sharp = dir.path().join("sharp.tif");
        let soft = dir.path().join("soft.tif");
        // A hard step versus a gradual ramp across the same span.
        write_frame(&sharp, 64, 48, |x, _| if x < 32 { 0.1 } else { 0.8 });
        write_frame(&soft, 64, 48, |x, _| {
            (0.1 + 0.7 * (x as f32 / 63.0)).clamp(0.0, 1.0)
        });

        let sharp_dir = dir.path().join("s1");
        let soft_dir = dir.path().join("s2");
        std::fs::create_dir_all(&sharp_dir).unwrap();
        std::fs::create_dir_all(&soft_dir).unwrap();

        let total = |m: &FocusMap| m.rows(0, 48).unwrap().iter().sum::<f32>();
        let sharp_total = total(&evaluate(&sharp, &sharp_dir, 3));
        let soft_total = total(&evaluate(&soft, &soft_dir, 3));

        assert!(
            sharp_total > soft_total * 10.0,
            "sharp {sharp_total:.4} should dominate soft {soft_total:.4}"
        );
    }

    #[test]
    fn focus_map_follows_the_transform() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edge.tif");
        // Vertical edge at x = 20.
        write_frame(&path, 96, 32, |x, _| if x < 20 { 0.1 } else { 0.9 });

        let peak_column = |map: &FocusMap| {
            let row = map.rows(16, 1).unwrap();
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };

        let native = evaluate(&path, dir.path(), 2);
        let unshifted = peak_column(&native);

        // The transform maps anchor coordinates onto the frame, so a +8 px transform
        // means anchor pixel p reads frame pixel p+8 — the feature lands 8 px
        // *earlier* in the aligned map.
        let shifted_dir = dir.path().join("shifted");
        std::fs::create_dir_all(&shifted_dir).unwrap();
        let mut transforms = HashMap::new();
        transforms.insert(path.clone(), Transform::translation(8.0, 0.0));
        let metric = WindowedLaplacian::new(2, &shifted_dir, transforms);
        let moved = metric.evaluate(&Image::open(&path).unwrap()).unwrap();

        let diff = peak_column(&moved) as i64 - unshifted as i64;
        assert!((diff + 8).abs() <= 1, "expected -8 px, got {diff}");
    }
}
