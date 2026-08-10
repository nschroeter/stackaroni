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
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::filter::box_sum;
use crate::fusion::{Bitmap, expand, reduce};
use crate::image::{FrameInfo, ScratchPlane};
use crate::pipeline::{FocusMap, FocusMetric, Image, RunControl, Stage, Transform};

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

/// Measure every frame in a stack, concurrently.
///
/// The per-frame loop lived in each caller — the CLI, the app, several tests — so
/// parallelising it there would have meant doing so once per caller that remembered.
/// It belongs beside [`crate::registration::register_stack`], which set the precedent
/// for the same reason.
///
/// Frames are independent: each reads its own file and writes its own scratch plane. The
/// returned order is the input order regardless of which thread finishes first, because
/// everything downstream indexes focus maps by frame.
pub fn evaluate_stack(
    metric: &dyn FocusMetric,
    frames: &[PathBuf],
    run: &dyn RunControl,
) -> Result<Vec<FocusMap>> {
    let total = frames.len();
    let done = AtomicUsize::new(0);

    let mut indexed: Vec<(usize, FocusMap)> = frames
        .par_iter()
        .enumerate()
        .map(|(index, path)| {
            if run.cancelled() {
                return Err(Error::Cancelled);
            }
            let map = metric.evaluate(&Image::open(path)?, run)?;
            run.progress(
                Stage::Focus,
                done.fetch_add(1, Ordering::Relaxed) + 1,
                total,
            );
            Ok((index, map))
        })
        .collect::<Result<Vec<_>>>()?;

    indexed.sort_unstable_by_key(|(index, _)| *index);
    Ok(indexed.into_iter().map(|(_, map)| map).collect())
}

/// Windowed Laplacian energy summed across a Gaussian pyramid — `docs/algorithms.md` §4.
///
/// ```text
/// F = F_0 + sum over k=1..scales-1  decay^k * expand^k(F_k)
/// ```
///
/// where `F_k` is [`WindowedLaplacian`]'s measure evaluated on level `k` of the frame's
/// Gaussian pyramid. The radius is the same at every level, so the window covers `2^k`
/// times the area as levels coarsen; that is where the multi-scale behaviour comes from.
///
/// `scales = 1` is the single-scale metric *exactly* — level 0 is accumulated unscaled and
/// the loop does not run — which `multi_scale_reduces_to_single_scale` pins bit-for-bit.
///
/// Combination is weighted-sum only; see §4 for why max-across-scales was scoped out of v1
/// rather than left as an unexposed branch.
pub struct MultiScaleLaplacian {
    radius: u32,
    scales: u32,
    decay: f32,
    scratch: PathBuf,
    transforms: HashMap<PathBuf, Transform>,
}

impl MultiScaleLaplacian {
    /// `scales` is clamped to at least 1: zero scales would measure nothing at all, and a
    /// focus map of zeros is the kind of input that produces a plausible-looking fused
    /// image rather than an error.
    pub fn new(
        radius: u32,
        scales: u32,
        decay: f32,
        scratch: &Path,
        transforms: HashMap<PathBuf, Transform>,
    ) -> Self {
        Self {
            radius,
            scales: scales.max(1),
            decay,
            scratch: scratch.to_path_buf(),
            transforms,
        }
    }

    fn measure_native(&self, image: &Image, path: &Path) -> Result<ScratchPlane> {
        let info = image.info();
        let mut plane = ScratchPlane::create(path, info.width, info.height)?;

        // Rows are read in bands, so each band builds its own pyramid. Two things have to
        // hold for that to give the same answer as pyramiding the whole frame.
        //
        // The halo must cover the coarsest level's reach: a level-k window spans
        // `(radius + 1) * 2^k` full-resolution rows, and building level k costs another
        // `2 * (2^k - 1)` for the binomial kernel's support at each halving.
        //
        // And the band must *start* on a multiple of `2^(scales-1)`, because `reduce` maps
        // source row `2i` to destination row `i` — an odd starting row shifts the whole
        // pyramid grid half a pixel relative to its neighbours, which would put a seam at
        // every band boundary. Aligning the read costs at most `2^(scales-1)` extra rows.
        let step = 1u32 << (self.scales - 1);
        let halo = (self.radius + 1) * step + 2 * (step - 1);

        let mut y0 = 0;
        while y0 < info.height {
            let rows = BAND_ROWS.min(info.height - y0);
            let read_from = (y0.saturating_sub(halo) / step) * step;
            let read_to = (y0 + rows + halo).min(info.height);
            let read_rows = read_to - read_from;

            let mut band = vec![0f32; info.row_len() * read_rows as usize];
            image.read_rows(read_from, read_rows, &mut band)?;

            let summed =
                self.pyramid_energy(to_luma(&band, info, read_rows), info.width, read_rows);

            let out = plane.rows_mut(y0, rows)?;
            for r in 0..rows {
                let src = (y0 + r - read_from) as usize;
                let dst = r as usize * info.width as usize;
                out[dst..dst + info.width as usize]
                    .copy_from_slice(&summed[src * info.width as usize..][..info.width as usize]);
            }
            y0 += rows;
        }
        Ok(plane)
    }

    /// Level 0's energy, plus each coarser level's expanded back up and decayed.
    fn pyramid_energy(&self, luma: Vec<f32>, width: u32, height: u32) -> Vec<f32> {
        let energy = laplacian_energy(&luma, width, height);
        let mut total = box_sum(&energy, width, height, self.radius);

        let mut level = Bitmap {
            width,
            height,
            channels: 1,
            data: luma,
        };
        // Dimensions of every level built so far, so a coarse map can be expanded back
        // through exactly the sizes it came down through — `reduce` rounds up, so the
        // sizes are not recoverable by halving.
        let mut sizes = vec![(width, height)];
        let mut weight = 1.0f32;

        for _ in 1..self.scales {
            level = reduce(&level);
            weight *= self.decay;

            let energy = laplacian_energy(&level.data, level.width, level.height);
            let mut up = Bitmap {
                width: level.width,
                height: level.height,
                channels: 1,
                data: box_sum(&energy, level.width, level.height, self.radius),
            };
            for &(w, h) in sizes.iter().rev() {
                up = expand(&up, w, h);
            }
            for (acc, add) in total.iter_mut().zip(up.data) {
                *acc += weight * add;
            }
            sizes.push((level.width, level.height));

            // Nothing below a few pixels carries usable structure, and `reduce` would
            // keep halving to a single pixel.
            if level.width <= 4 || level.height <= 4 {
                break;
            }
        }
        total
    }
}

impl FocusMetric for MultiScaleLaplacian {
    fn evaluate(&self, image: &Image, run: &dyn RunControl) -> Result<FocusMap> {
        let _ = run;
        let stem = file_stem(image.path());
        let native_path = self.scratch.join(format!("{stem}.native.f32"));
        let native = self.measure_native(image, &native_path)?;
        align_into_anchor(
            &self.scratch,
            &self.transforms,
            image,
            native,
            &native_path,
            &stem,
        )
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
    /// `run` is accepted and not polled: one call is one frame, ~1.3 s, and the loop
    /// over frames belongs to the caller, which checks between them.
    fn evaluate(&self, image: &Image, run: &dyn RunControl) -> Result<FocusMap> {
        let _ = run;
        let stem = file_stem(image.path());
        let native_path = self.scratch.join(format!("{stem}.native.f32"));
        let native = self.measure_native(image, &native_path)?;
        align_into_anchor(
            &self.scratch,
            &self.transforms,
            image,
            native,
            &native_path,
            &stem,
        )
    }
}

/// Warp a native-coordinate focus plane into anchor coordinates and drop the native one.
///
/// Shared by both metrics: they differ only in how the energy is measured, and the
/// alignment half — including the reasoning above about measuring before warping — is
/// identical. Keeping one copy means a second metric cannot quietly get this wrong.
fn align_into_anchor(
    scratch: &Path,
    transforms: &HashMap<PathBuf, Transform>,
    image: &Image,
    native: ScratchPlane,
    native_path: &Path,
    stem: &str,
) -> Result<FocusMap> {
    let info = image.info();
    let transform = transforms
        .get(image.path())
        .copied()
        .unwrap_or(Transform::IDENTITY);

    let mut aligned = ScratchPlane::create(
        &scratch.join(format!("{stem}.focus.f32")),
        info.width,
        info.height,
    )?;
    warp_plane(&native, &mut aligned, transform, info)?;

    drop(native);
    let _ = std::fs::remove_file(native_path);
    Ok(aligned)
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
        metric.evaluate(&Image::open(path).unwrap(), &()).unwrap()
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

    /// One scale must *be* the single-scale metric, bit for bit.
    ///
    /// Not "close to": if this is only approximately true then the two metrics differ by
    /// something nobody chose, and every later comparison between them silently includes
    /// that difference. Exactness is also what lets the multi-scale path be described as a
    /// generalization of §3 rather than a second metric that resembles it.
    ///
    /// Bits, not epsilon — the same instrument as `output_is_stable`, for the same reason:
    /// a one-ULP drift is invisible to any tolerance loose enough to be useful.
    #[test]
    fn multi_scale_reduces_to_single_scale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("texture.tif");
        // Structure at several scales, so the two paths have something to disagree about.
        write_frame(&path, 128, 96, |x, y| {
            let fine = ((x * 7 + y * 3) % 11) as f32 / 11.0;
            let coarse = ((x / 16 + y / 16) % 2) as f32;
            0.2 + 0.5 * fine * 0.3 + 0.4 * coarse
        });

        let (a_dir, b_dir) = (dir.path().join("a"), dir.path().join("b"));
        std::fs::create_dir_all(&a_dir).unwrap();
        std::fs::create_dir_all(&b_dir).unwrap();

        let single = WindowedLaplacian::new(4, &a_dir, HashMap::new())
            .evaluate(&Image::open(&path).unwrap(), &())
            .unwrap();
        let multi = MultiScaleLaplacian::new(4, 1, 1.0, &b_dir, HashMap::new())
            .evaluate(&Image::open(&path).unwrap(), &())
            .unwrap();

        let (a, b) = (single.rows(0, 96).unwrap(), multi.rows(0, 96).unwrap());
        let differing = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(differing, 0, "{differing} of {} samples differ", a.len());
    }

    #[test]
    fn more_scales_add_coarse_structure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("coarse.tif");
        // Broad blocks: little level-0 energy away from the block edges, plenty once the
        // pyramid brings the structure into the window.
        write_frame(&path, 128, 96, |x, y| {
            if (x / 24 + y / 24) % 2 == 0 { 0.2 } else { 0.7 }
        });

        let total = |scales: u32, sub: &str| {
            let scratch = dir.path().join(sub);
            std::fs::create_dir_all(&scratch).unwrap();
            let map = MultiScaleLaplacian::new(2, scales, 0.5, &scratch, HashMap::new())
                .evaluate(&Image::open(&path).unwrap(), &())
                .unwrap();
            map.rows(0, 96).unwrap().iter().sum::<f32>()
        };

        let one = total(1, "s1");
        let three = total(3, "s3");
        assert!(
            three > one,
            "coarse levels should add energy: {three} vs {one}"
        );
    }

    /// Bands must not show at their boundaries.
    ///
    /// Each band builds its own pyramid, and `reduce` maps source row `2i` to row `i`, so
    /// a band starting on an odd row would sample the pyramid on a different grid than its
    /// neighbour — a seam every `BAND_ROWS` rows. The frame here is taller than one band
    /// and constant down each column, so every interior row must measure identically; a
    /// misaligned band shows up as one row that does not.
    #[test]
    fn band_boundaries_leave_no_seam() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stripes.tif");
        let height = BAND_ROWS * 2 + 64;
        write_frame(&path, 64, height, |x, _| {
            if (x / 3) % 2 == 0 { 0.25 } else { 0.75 }
        });

        let map = MultiScaleLaplacian::new(3, 4, 0.5, dir.path(), HashMap::new())
            .evaluate(&Image::open(&path).unwrap(), &())
            .unwrap();

        // Skip the top and bottom, where clamping legitimately differs.
        let rows = map.rows(64, height - 128).unwrap();
        let width = 64usize;
        let reference = &rows[..width];
        for (r, row) in rows.chunks(width).enumerate() {
            for (x, (&got, &want)) in row.iter().zip(reference).enumerate() {
                assert!(
                    (got - want).abs() <= want.abs() * 1e-5,
                    "row {} column {x} differs: {got} vs {want}",
                    64 + r
                );
            }
        }
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
        let moved = metric.evaluate(&Image::open(&path).unwrap(), &()).unwrap();

        let diff = peak_column(&moved) as i64 - unshifted as i64;
        assert!((diff + 8).abs() <= 1, "expected -8 px, got {diff}");
    }
}
