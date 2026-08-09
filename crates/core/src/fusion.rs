//! Laplacian-pyramid fusion.
//!
//! Burt & Adelson, *IEEE Trans. Communications* 31(4), 1983, 532-540 for the pyramid
//! itself; Burt & Kolczynski, *ICCV* 1993, 173-182 for using it to fuse. Instead of
//! selecting a source frame per pixel, each spatial-frequency band is blended
//! separately under a weight map smoothed to that band's scale, which is what stops
//! the transitions between frames showing as seams.
//!
//! # Memory
//!
//! The blend is linear in the frames — `fused[l] = sum_k w_k[l] * L_k[l]` — so frames
//! accumulate one at a time and the frame count drops out of the budget entirely.
//! Peak is one warped frame plus three pyramids, about 2.5 GB on a 50 MP stack
//! whatever its depth. Row-banding would only be needed to remove a `x frame_count`
//! term that accumulation already removes, and it would introduce band boundaries to
//! blend across; there is nothing to gain here until a frame no longer fits.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::filter::box_sum;
use crate::image::{FrameInfo, ScratchPlane};
use crate::pipeline::{Image, ImageFusion, RunControl, Stage, Transform, WeightMaps};
use crate::tiff_io::write_rgb16_srgb;

/// Burt & Adelson's binomial kernel, the separable `a=0.4` case of their generating
/// kernel.
const KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Output rows warped per read pass.
const WARP_CHUNK: u32 = 256;

/// An interleaved float image.
#[derive(Clone)]
pub struct Bitmap {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Bitmap {
    pub fn new(width: u32, height: u32, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![0.0; width as usize * height as usize * channels],
        }
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y as usize * self.width as usize + x as usize) * self.channels
    }
}

/// How many pyramid levels an image of this size supports before the coarsest drops
/// below `floor`.
///
/// Derived rather than fixed, because the test stacks span 900 to 5784 rows: a fixed
/// level count would make the coarsest level represent a different physical scale on
/// each stack, and cross-stack comparisons in the eval log would then be conflating
/// algorithm behaviour with resolution.
pub fn level_count(width: u32, height: u32, floor: u32) -> usize {
    let mut levels = 1;
    let (mut w, mut h) = (width, height);
    while w.min(h) / 2 >= floor.max(1) {
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        levels += 1;
    }
    levels
}

/// Blur with the binomial kernel and drop every other sample.
pub fn reduce(src: &Bitmap) -> Bitmap {
    let blurred = blur(src);
    let mut out = Bitmap::new(src.width.div_ceil(2), src.height.div_ceil(2), src.channels);
    for y in 0..out.height {
        for x in 0..out.width {
            let (si, di) = (blurred.index(2 * x, 2 * y), out.index(x, y));
            out.data[di..di + out.channels].copy_from_slice(&blurred.data[si..si + out.channels]);
        }
    }
    out
}

/// Upsample to the given size.
///
/// Rather than inserting zeros and blurring, this applies the two phases of the
/// binomial kernel directly to the source: even outputs take `[1,6,1]/8`, odd
/// outputs `[1,1]/2`. Both sets sum to one, so a constant expands to that constant —
/// including at the borders, which zero-insertion gets wrong, because clamping then
/// replicates an inserted zero instead of a real sample and brightens the edge by
/// half.
pub fn expand(src: &Bitmap, width: u32, height: u32) -> Bitmap {
    let c = src.channels;

    let mut wide = Bitmap::new(width, src.height, c);
    for y in 0..src.height {
        for x in 0..width {
            let di = wide.index(x, y);
            let i = (x / 2) as i64;
            let tap = |o: i64, ch: usize| {
                let sx = o.clamp(0, src.width as i64 - 1) as u32;
                src.data[src.index(sx, y) + ch]
            };
            for ch in 0..c {
                wide.data[di + ch] = if x % 2 == 0 {
                    (tap(i - 1, ch) + 6.0 * tap(i, ch) + tap(i + 1, ch)) / 8.0
                } else {
                    (tap(i, ch) + tap(i + 1, ch)) / 2.0
                };
            }
        }
    }

    let mut out = Bitmap::new(width, height, c);
    for y in 0..height {
        let j = (y / 2) as i64;
        for x in 0..width {
            let di = out.index(x, y);
            let tap = |o: i64, ch: usize| {
                let sy = o.clamp(0, wide.height as i64 - 1) as u32;
                wide.data[wide.index(x, sy) + ch]
            };
            for ch in 0..c {
                out.data[di + ch] = if y % 2 == 0 {
                    (tap(j - 1, ch) + 6.0 * tap(j, ch) + tap(j + 1, ch)) / 8.0
                } else {
                    (tap(j, ch) + tap(j + 1, ch)) / 2.0
                };
            }
        }
    }
    out
}

/// Separable binomial blur, edges replicated.
fn blur(src: &Bitmap) -> Bitmap {
    let c = src.channels;
    let (w, h) = (src.width as i64, src.height as i64);
    let mut horizontal = Bitmap::new(src.width, src.height, c);

    for y in 0..h {
        for x in 0..w {
            let di = horizontal.index(x as u32, y as u32);
            for (k, weight) in KERNEL.iter().enumerate() {
                let sx = (x + k as i64 - 2).clamp(0, w - 1);
                let si = src.index(sx as u32, y as u32);
                for ch in 0..c {
                    horizontal.data[di + ch] += weight * src.data[si + ch];
                }
            }
        }
    }

    let mut out = Bitmap::new(src.width, src.height, c);
    for y in 0..h {
        for x in 0..w {
            let di = out.index(x as u32, y as u32);
            for (k, weight) in KERNEL.iter().enumerate() {
                let sy = (y + k as i64 - 2).clamp(0, h - 1);
                let si = horizontal.index(x as u32, sy as u32);
                for ch in 0..c {
                    out.data[di + ch] += weight * horizontal.data[si + ch];
                }
            }
        }
    }
    out
}

/// Successive reductions, finest first.
pub fn gaussian_pyramid(base: &Bitmap, levels: usize) -> Vec<Bitmap> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(base.clone());
    for i in 1..levels {
        pyramid.push(reduce(&pyramid[i - 1]));
    }
    pyramid
}

/// Band-pass levels, with the coarsest Gaussian residual kept last so the pyramid
/// reconstructs exactly.
pub fn laplacian_pyramid(base: &Bitmap, levels: usize) -> Vec<Bitmap> {
    let gaussian = gaussian_pyramid(base, levels);
    let mut pyramid = Vec::with_capacity(levels);
    for i in 0..levels - 1 {
        let up = expand(&gaussian[i + 1], gaussian[i].width, gaussian[i].height);
        let mut band = gaussian[i].clone();
        for (v, u) in band.data.iter_mut().zip(&up.data) {
            *v -= u;
        }
        pyramid.push(band);
    }
    pyramid.push(gaussian[levels - 1].clone());
    pyramid
}

/// Collapse a Laplacian pyramid back to an image.
pub fn reconstruct(pyramid: &[Bitmap]) -> Bitmap {
    let mut current = pyramid[pyramid.len() - 1].clone();
    for level in (0..pyramid.len() - 1).rev() {
        let target = &pyramid[level];
        let mut up = expand(&current, target.width, target.height);
        for (v, b) in up.data.iter_mut().zip(&target.data) {
            *v += b;
        }
        current = up;
    }
    current
}

/// Multi-scale blend of a whole stack under per-frame weight maps.
///
/// Output path and transforms are injected here: [`ImageFusion::fuse`] receives only
/// images and weights, and the result is a new file rather than one of the inputs.
pub struct LaplacianPyramidFusion {
    output: PathBuf,
    transforms: HashMap<PathBuf, Transform>,
    floor: u32,
}

impl LaplacianPyramidFusion {
    pub fn new(output: &Path, transforms: HashMap<PathBuf, Transform>, floor: u32) -> Self {
        Self {
            output: output.to_path_buf(),
            transforms,
            floor,
        }
    }
}

impl ImageFusion for LaplacianPyramidFusion {
    fn fuse(&self, images: &[Image], weights: &WeightMaps, run: &dyn RunControl) -> Result<Image> {
        assert_eq!(
            images.len(),
            weights.len(),
            "one weight plane per image required"
        );
        let info = images[0].info();
        let levels = level_count(info.width, info.height, self.floor);

        // Accumulator, one band-pass level at a time; frames add into it in turn.
        let mut accumulator: Vec<Bitmap> = {
            let seed = Bitmap::new(info.width, info.height, 3);
            gaussian_pyramid(&seed, levels)
        };

        // Per frame. This is the stage that makes cancellation worth having: ~6.2 s a
        // frame and ~10 minutes total on a 100-frame stack, against ~1.3 s a frame
        // everywhere else. Sub-frame checks were considered and deliberately left out —
        // see the run-control design notes in `CLAUDE.md`.
        for (index, (image, weight)) in images.iter().zip(weights).enumerate() {
            if run.cancelled() {
                return Err(Error::Cancelled);
            }
            let transform = self
                .transforms
                .get(image.path())
                .copied()
                .unwrap_or(Transform::IDENTITY);

            let warped = warp_frame(image, transform, info)?;
            let bands = laplacian_pyramid(&warped, levels);
            drop(warped);

            // The weight map is already in anchor coordinates, so it is not warped
            // again here. Smoothing it down the pyramid is what makes each frequency
            // band blend at its own scale rather than all of them at pixel scale.
            let weight_bitmap = plane_to_bitmap(weight)?;
            let weight_levels = gaussian_pyramid(&weight_bitmap, levels);

            for level in 0..levels {
                let (dst, src, w) = (
                    &mut accumulator[level],
                    &bands[level],
                    &weight_levels[level],
                );
                for i in 0..w.data.len() {
                    for ch in 0..3 {
                        dst.data[i * 3 + ch] += w.data[i] * src.data[i * 3 + ch];
                    }
                }
            }
            run.progress(Stage::Fuse, index + 1, images.len());
        }

        let fused = reconstruct(&accumulator);
        write_rgb16_srgb(&self.output, info, |y, row| {
            let start = y as usize * info.width as usize * 3;
            row.copy_from_slice(&fused.data[start..start + row.len()]);
            Ok(())
        })?;
        Image::open(&self.output)
    }
}

/// Per-level selection fusion: the PMax-shaped rule from `docs/algorithms.md` §6b.
///
/// Burt & Kolczynski, *ICCV* 1993, 173-182. The band-pass levels take a fresh decision
/// at every level and position from the pyramid coefficients themselves, instead of
/// inheriting one decision taken once at a single window scale. The coarsest (base)
/// level has no contrast to select on and keeps [`LaplacianPyramidFusion`]'s weighted
/// blend, so the weight maps are still required.
///
/// # Two deliberate deviations from the paper, both recorded here rather than silently
///
/// **The match/average branch is omitted.** B&K select where the sources disagree and
/// average where they agree, which needs every source's coefficients at a level
/// simultaneously — 100 frames of full-resolution pyramid, far past any memory budget
/// this pipeline can hold. Selection alone streams: one running best-salience plane per
/// level, frames folded in one at a time, the frame count out of the budget exactly as
/// in [`LaplacianPyramidFusion`]. If selection alone shows switching artifacts in
/// smoothly varying regions, that is the evidence that the match term is worth the
/// memory, and the place to look is background bokeh.
///
/// **Salience is joint across channels, not per channel.** Selecting a different frame
/// for red than for green at the same position would read as colour fringing on exactly
/// the high-contrast edges this rule exists to improve.
pub struct SelectionFusion {
    output: PathBuf,
    transforms: HashMap<PathBuf, Transform>,
    floor: u32,
    salience_radius: u32,
}

impl SelectionFusion {
    pub fn new(
        output: &Path,
        transforms: HashMap<PathBuf, Transform>,
        floor: u32,
        salience_radius: u32,
    ) -> Self {
        Self {
            output: output.to_path_buf(),
            transforms,
            floor,
            salience_radius,
        }
    }
}

impl ImageFusion for SelectionFusion {
    fn fuse(&self, images: &[Image], weights: &WeightMaps, run: &dyn RunControl) -> Result<Image> {
        assert_eq!(
            images.len(),
            weights.len(),
            "one weight plane per image required"
        );
        let info = images[0].info();
        let levels = level_count(info.width, info.height, self.floor);

        let mut result: Vec<Bitmap> = {
            let seed = Bitmap::new(info.width, info.height, 3);
            gaussian_pyramid(&seed, levels)
        };
        // Running best windowed salience per band-pass level. Negative so that the
        // first frame wins everywhere regardless of how flat it is.
        let mut best: Vec<Vec<f32>> = result[..levels - 1]
            .iter()
            .map(|b| vec![-1.0f32; (b.width as usize) * (b.height as usize)])
            .collect();

        // Per frame. This is the stage that makes cancellation worth having: ~6.2 s a
        // frame and ~10 minutes total on a 100-frame stack, against ~1.3 s a frame
        // everywhere else. Sub-frame checks were considered and deliberately left out —
        // see the run-control design notes in `CLAUDE.md`.
        for (index, (image, weight)) in images.iter().zip(weights).enumerate() {
            if run.cancelled() {
                return Err(Error::Cancelled);
            }
            let transform = self
                .transforms
                .get(image.path())
                .copied()
                .unwrap_or(Transform::IDENTITY);

            let warped = warp_frame(image, transform, info)?;
            let bands = laplacian_pyramid(&warped, levels);
            drop(warped);

            for level in 0..levels - 1 {
                select_more_salient(
                    &mut result[level],
                    &mut best[level],
                    &bands[level],
                    self.salience_radius,
                );
            }

            // Base level: the weight map reduced all the way down, blended as before.
            // Only the coarsest level is needed, so the intermediate levels are not kept.
            let mut w = plane_to_bitmap(weight)?;
            for _ in 1..levels {
                w = reduce(&w);
            }
            let (dst, src) = (&mut result[levels - 1], &bands[levels - 1]);
            for i in 0..w.data.len() {
                for ch in 0..3 {
                    dst.data[i * 3 + ch] += w.data[i] * src.data[i * 3 + ch];
                }
            }
            run.progress(Stage::Fuse, index + 1, images.len());
        }

        let fused = reconstruct(&result);
        write_rgb16_srgb(&self.output, info, |y, row| {
            let start = y as usize * info.width as usize * 3;
            row.copy_from_slice(&fused.data[start..start + row.len()]);
            Ok(())
        })?;
        Image::open(&self.output)
    }
}

/// Overwrite `dst` wherever `src` carries more windowed salience, updating `best`.
///
/// Salience is local energy — the sum of squared coefficients over a
/// `(2*radius+1)` square window — not the coefficient magnitude at the pixel itself.
/// The distinction is the whole point: per-pixel argmax over Laplacian coefficients
/// draws neighbouring pixels from inconsistent sources and, on ISO-1600 frames, would
/// routinely select noise (Wang et al., *PLOS ONE* 13(5), 2018, e0191085). The window
/// makes an isolated spike lose to genuine surrounding structure.
fn select_more_salient(dst: &mut Bitmap, best: &mut [f32], src: &Bitmap, radius: u32) {
    let n = best.len();
    let energy: Vec<f32> = (0..n)
        .map(|i| {
            let p = &src.data[i * 3..i * 3 + 3];
            p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
        })
        .collect();
    let salience = box_sum(&energy, src.width, src.height, radius);

    for i in 0..n {
        if salience[i] > best[i] {
            best[i] = salience[i];
            dst.data[i * 3..i * 3 + 3].copy_from_slice(&src.data[i * 3..i * 3 + 3]);
        }
    }
}

fn plane_to_bitmap(plane: &ScratchPlane) -> Result<Bitmap> {
    let mut out = Bitmap::new(plane.width(), plane.height(), 1);
    out.data.copy_from_slice(plane.rows(0, plane.height())?);
    Ok(out)
}

/// Resample a frame into anchor coordinates.
///
/// **Convention, and it matters.** `Transform` scales about the *image centre*, not
/// the origin — focus breathing zooms about the optical axis, and T5b's per-region
/// diagnostic confirmed the distortion is radially symmetric about the frame centre.
/// So every coordinate is made centre-relative before [`Transform::apply`] and shifted
/// back afterwards. Applying the same numbers about the origin would still produce a
/// plausible-looking image; it would fail as a soft radial doubling that grows toward
/// the corners, which is far harder to attribute than a hard seam.
///
/// Because scale is generally not 1, the source rows a given output row needs are not
/// a fixed offset — the range stretches as well as shifts, so it is recomputed per
/// chunk rather than assumed.
fn warp_frame(image: &Image, transform: Transform, info: FrameInfo) -> Result<Bitmap> {
    let (cx, cy) = (info.width as f32 / 2.0, info.height as f32 / 2.0);
    let mut out = Bitmap::new(info.width, info.height, 3);
    let source_y = |y: u32| transform.apply(0.0, y as f32 - cy).1 + cy;

    let mut y0 = 0;
    while y0 < info.height {
        let rows = WARP_CHUNK.min(info.height - y0);

        // Both ends, because a negative scale-and-shift can invert the ordering.
        let (a, b) = (source_y(y0), source_y(y0 + rows - 1));
        let from = (a.min(b).floor() as i64 - 1).clamp(0, info.height as i64 - 1) as u32;
        let to = (a.max(b).ceil() as i64 + 2).clamp(1, info.height as i64) as u32;
        let span = to - from;

        let mut band = vec![0f32; info.row_len() * span as usize];
        image.read_rows(from, span, &mut band)?;

        let sample = |x: f32, y: f32, ch: usize| -> f32 {
            let (x0, y0f) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0f);
            let get = |ix: i64, iy: i64| -> f32 {
                let ix = ix.clamp(0, info.width as i64 - 1) as usize;
                let iy = (iy - from as i64).clamp(0, span as i64 - 1) as usize;
                band[iy * info.row_len() + ix * 3 + ch]
            };
            let (ix, iy) = (x0 as i64, y0f as i64);
            let top = get(ix, iy) * (1.0 - fx) + get(ix + 1, iy) * fx;
            let bottom = get(ix, iy + 1) * (1.0 - fx) + get(ix + 1, iy + 1) * fx;
            top * (1.0 - fy) + bottom * fy
        };

        for r in 0..rows {
            for x in 0..info.width {
                let (sx, sy) = transform.apply(x as f32 - cx, (y0 + r) as f32 - cy);
                let di = out.index(x, y0 + r);
                for ch in 0..3 {
                    out.data[di + ch] = sample(sx + cx, sy + cy, ch);
                }
            }
        }
        y0 += rows;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;

    fn textured(width: u32, height: u32, channels: usize) -> Bitmap {
        let mut b = Bitmap::new(width, height, channels);
        for y in 0..height {
            for x in 0..width {
                let i = b.index(x, y);
                for ch in 0..channels {
                    b.data[i + ch] =
                        ((x as f32 * 0.31).sin() * (y as f32 * 0.17).cos() + 0.3 * ch as f32).abs();
                }
            }
        }
        b
    }

    #[test]
    fn levels_derive_from_the_smaller_edge() {
        // 900 -> 450 -> 225 -> 112 -> 56 -> 28, so it stops with 56 above the floor.
        assert_eq!(level_count(1200, 900, 32), 5);
        assert_eq!(level_count(8664, 5784, 32), 8);
        // A tiny image supports exactly one level.
        assert_eq!(level_count(40, 40, 32), 1);
    }

    #[test]
    fn expanding_a_constant_preserves_it() {
        let mut flat = Bitmap::new(16, 16, 1);
        flat.data.fill(0.7);
        for v in expand(&flat, 32, 32).data {
            assert!((v - 0.7).abs() < 1e-4, "{v}");
        }
    }

    #[test]
    fn pyramid_reconstructs_the_original() {
        // The property the whole stage rests on: if this is lossy, every fused
        // output carries the error whatever the weights say.
        let base = textured(64, 48, 3);
        let levels = level_count(64, 48, 8);
        let back = reconstruct(&laplacian_pyramid(&base, levels));

        assert_eq!(back.data.len(), base.data.len());
        let worst = base
            .data
            .iter()
            .zip(&back.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "reconstruction error {worst}");
    }

    #[test]
    fn odd_sizes_reconstruct_too() {
        let base = textured(37, 29, 3);
        let levels = level_count(37, 29, 8);
        let back = reconstruct(&laplacian_pyramid(&base, levels));
        let worst = base
            .data
            .iter()
            .zip(&back.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "reconstruction error {worst}");
    }

    #[test]
    fn all_weight_on_one_frame_returns_that_frame() {
        let a = textured(32, 32, 3);
        let levels = level_count(32, 32, 8);

        let mut ones = Bitmap::new(32, 32, 1);
        ones.data.fill(1.0);
        let weight_levels = gaussian_pyramid(&ones, levels);
        let bands = laplacian_pyramid(&a, levels);

        let mut acc = gaussian_pyramid(&Bitmap::new(32, 32, 3), levels);
        for level in 0..levels {
            for i in 0..weight_levels[level].data.len() {
                for ch in 0..3 {
                    acc[level].data[i * 3 + ch] +=
                        weight_levels[level].data[i] * bands[level].data[i * 3 + ch];
                }
            }
        }

        let worst = a
            .data
            .iter()
            .zip(&reconstruct(&acc).data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "should return frame a unchanged: {worst}");
    }

    /// A band with checkerboard detail in one half and nothing in the other.
    fn detail_in(half: Range<u32>, width: u32, height: u32) -> Bitmap {
        let mut b = Bitmap::new(width, height, 3);
        for y in 0..height {
            for x in half.clone() {
                let i = b.index(x, y);
                let v = if (x + y) % 2 == 0 { 0.5 } else { -0.5 };
                b.data[i..i + 3].fill(v);
            }
        }
        b
    }

    #[test]
    fn selection_takes_the_more_salient_source_at_each_position() {
        let (w, h) = (32u32, 16u32);
        let left = detail_in(0..16, w, h);
        let right = detail_in(16..32, w, h);

        let mut dst = Bitmap::new(w, h, 3);
        let mut best = vec![-1.0f32; (w * h) as usize];
        select_more_salient(&mut dst, &mut best, &left, 1);
        select_more_salient(&mut dst, &mut best, &right, 1);

        // Away from the seam, each half must come from whichever source has the
        // detail there — not from an average of the two, which would halve it.
        for y in 2..h - 2 {
            for x in [4u32, 27] {
                let i = dst.index(x, y);
                let want = if x < 16 { &left } else { &right };
                assert_eq!(dst.data[i], want.data[i], "at ({x},{y})");
                assert_eq!(dst.data[i].abs(), 0.5, "at ({x},{y})");
            }
        }
    }

    #[test]
    fn selection_is_joint_across_channels_never_per_channel() {
        // Load-bearing for colour, not a style preference. If R could come from one
        // frame while G and B came from another, each frame's independent colour noise
        // would combine into drift that no single source has — manufactured chroma
        // noise. Sources are built so the per-channel winner differs from the joint
        // winner: `blue` has the larger coefficient in one channel, `broad` the larger
        // total. Joint selection must take `broad` whole.
        let (w, h) = (16u32, 16u32);
        let mut broad = Bitmap::new(w, h, 3);
        let mut blue = Bitmap::new(w, h, 3);
        for i in 0..(w * h) as usize {
            broad.data[i * 3..i * 3 + 3].copy_from_slice(&[0.4, 0.4, 0.4]);
            blue.data[i * 3..i * 3 + 3].copy_from_slice(&[0.0, 0.0, 0.6]);
        }

        let mut dst = Bitmap::new(w, h, 3);
        let mut best = vec![-1.0f32; (w * h) as usize];
        select_more_salient(&mut dst, &mut best, &blue, 1);
        select_more_salient(&mut dst, &mut best, &broad, 1);

        // 0.48 total beats 0.36, so `broad` wins — and wins in *every* channel,
        // including blue where it is individually the weaker source.
        for i in 0..(w * h) as usize {
            assert_eq!(&dst.data[i * 3..i * 3 + 3], &[0.4, 0.4, 0.4], "at {i}");
        }
    }

    #[test]
    fn the_salience_window_loses_to_structure_against_an_isolated_spike() {
        // This is the §6b rationale under test: per-pixel argmax would take the
        // spike, because 2.0 > 0.5 at that one position.
        let (w, h) = (32u32, 16u32);
        let structure = detail_in(0..32, w, h);
        let mut spike = Bitmap::new(w, h, 3);
        let si = spike.index(10, 8);
        spike.data[si..si + 3].fill(2.0);

        let mut dst = Bitmap::new(w, h, 3);
        let mut best = vec![-1.0f32; (w * h) as usize];
        select_more_salient(&mut dst, &mut best, &structure, 2);
        select_more_salient(&mut dst, &mut best, &spike, 2);

        assert_eq!(
            dst.data[si], structure.data[si],
            "an isolated spike should lose to surrounding structure"
        );
    }

    #[test]
    fn selection_reconstructs_a_frame_that_dominates_every_level() {
        // Selection is only meaningful if it is exact where it selects: a source that
        // wins at every level and position must come back unchanged, the same
        // guarantee `all_weight_on_one_frame_returns_that_frame` gives the blend.
        let sharp = textured(32, 32, 3);
        // The same texture at a tenth the contrast, so its salience is exactly 0.01x
        // sharp's everywhere — strictly smaller wherever sharp has any, and equal only
        // where both coefficients are zero and the choice cannot matter.
        let mut dim = sharp.clone();
        for v in &mut dim.data {
            *v *= 0.1;
        }
        let levels = level_count(32, 32, 8);

        let mut result = gaussian_pyramid(&Bitmap::new(32, 32, 3), levels);
        let mut best: Vec<Vec<f32>> = result[..levels - 1]
            .iter()
            .map(|b| vec![-1.0f32; (b.width * b.height) as usize])
            .collect();
        for source in [&dim, &sharp] {
            let bands = laplacian_pyramid(source, levels);
            for level in 0..levels - 1 {
                select_more_salient(&mut result[level], &mut best[level], &bands[level], 2);
            }
            // All the base-level weight on the sharp frame, as the blend would give it.
            result[levels - 1] = bands[levels - 1].clone();
        }

        let worst = sharp
            .data
            .iter()
            .zip(&reconstruct(&result).data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "should return the sharp frame unchanged: {worst}"
        );
    }
}
