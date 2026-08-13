//! Edge-aware weight-map estimation by guided filtering.
//!
//! `docs/algorithms.md` §7 calls this the critical stage — more important than the
//! blend that follows. A raw winner-takes-all focus decision lets neighbouring
//! pixels choose unrelated source frames; smoothing it with an ordinary Gaussian
//! would fix the incoherence but bleed weight straight across true depth
//! boundaries, which is where the antennae and leg edges in the quality checklist
//! live.
//!
//! The guided filter (He, Sun & Tang, *ECCV* 2010; extended *TPAMI* 35(6), 2013)
//! smooths a map while respecting edges present in a separate guide image. Applying
//! it to fusion weights follows Li, Kang & Hu, *IEEE TIP* 22(7), 2013, 2864-2875.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::filter::{box_mean, mul};
use crate::image::{BAND_ROWS, ScratchPlane, linear_to_srgb, warp_plane};
use crate::pipeline::{FocusMap, Image, RunControl, Stage, Transform, WeightEstimator, WeightMaps};

/// Rec. 709 luma coefficients, applied to linear-light RGB.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Which tone space the guide image is measured in.
///
/// The guided filter decides how much edge to preserve from the guide's local
/// variance, so this changes which edges get protected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuideSpace {
    /// Guide in linear light, as decoded.
    Linear,
    /// Guide re-encoded to sRGB before measuring variance.
    ///
    /// Expected to be the better choice, and the reason is about which edges the
    /// filter protects. Linear radiance spans a far wider numeric range than
    /// gamma-compressed values, so a dark antenna against a mid-tone background can
    /// have a small linear-domain edge magnitude even though it is obvious to the
    /// eye, while a minor tonal step inside a bright highlight has a large one.
    /// Measuring variance in linear light therefore preferentially protects
    /// highlight boundaries and under-protects exactly the thin dark structures the
    /// checklist prioritizes. Gamma encoding is roughly where human contrast
    /// sensitivity is uniform, which is nearer to "where a person would say an edge
    /// is" — which is what a guide is for.
    Perceptual,
}

/// Argmax focus selection, refined by guided filtering against each frame's own
/// aligned image.
///
/// Frames and transforms are injected here because [`WeightEstimator::weights`]
/// receives only focus maps, and the guide for frame `k` must be frame `k`'s own
/// aligned content — a shared reference guide would be protecting edges that are not
/// present in the map being filtered.
pub struct GuidedWeights {
    frames: Vec<PathBuf>,
    transforms: Vec<Transform>,
    radius: u32,
    epsilon: f32,
    guide_space: GuideSpace,
    scratch: PathBuf,
}

impl GuidedWeights {
    pub fn new(
        frames: Vec<PathBuf>,
        transforms: Vec<Transform>,
        radius: u32,
        epsilon: f32,
        guide_space: GuideSpace,
        scratch: &Path,
    ) -> Self {
        assert_eq!(
            frames.len(),
            transforms.len(),
            "one transform per frame required"
        );
        Self {
            frames,
            transforms,
            radius,
            epsilon,
            guide_space,
            scratch: scratch.to_path_buf(),
        }
    }

    /// Per-pixel index of the sharpest frame, as a plane so it can be dumped for
    /// debugging and re-read band by band without rescanning every focus map.
    pub fn labels(&self, focus_maps: &[FocusMap], run: &dyn RunControl) -> Result<ScratchPlane> {
        let (width, height) = (focus_maps[0].width(), focus_maps[0].height());
        let mut labels = ScratchPlane::create(&self.scratch.join("labels.f32"), width, height)?;

        let mut y0 = 0;
        while y0 < height {
            // Per band rather than per frame: this loop is banded over rows with every
            // frame read inside each band, so the band is the only unit available.
            if run.cancelled() {
                return Err(Error::Cancelled);
            }
            let rows = BAND_ROWS.min(height - y0);
            let len = rows as usize * width as usize;
            let mut best = vec![f32::NEG_INFINITY; len];
            let mut winner = vec![0f32; len];

            for (k, map) in focus_maps.iter().enumerate() {
                let band = map.rows(y0, rows)?;
                for i in 0..len {
                    if band[i] > best[i] {
                        best[i] = band[i];
                        winner[i] = k as f32;
                    }
                }
            }
            labels.rows_mut(y0, rows)?.copy_from_slice(&winner);
            y0 += rows;
        }
        Ok(labels)
    }

    /// Frame `k`'s luma, aligned into anchor coordinates, in the configured space.
    fn guide(&self, index: usize) -> Result<ScratchPlane> {
        let image = Image::open(&self.frames[index])?;
        let info = image.info();
        let transform = self.transforms[index];

        // Native-coordinate luma first, then warp — same reasoning as the focus
        // metric: keep the resampling off the measurement, not just off the pixels.
        let native_path = self.scratch.join(format!("guide{index}.native.f32"));
        let mut native = ScratchPlane::create(&native_path, info.width, info.height)?;

        let mut y0 = 0;
        while y0 < info.height {
            let rows = BAND_ROWS.min(info.height - y0);
            let mut band = vec![0f32; info.row_len() * rows as usize];
            image.read_rows(y0, rows, &mut band)?;

            let out = native.rows_mut(y0, rows)?;
            for (i, slot) in out.iter_mut().enumerate() {
                let p = i * 3;
                let luma = LUMA[0] * band[p] + LUMA[1] * band[p + 1] + LUMA[2] * band[p + 2];
                *slot = match self.guide_space {
                    GuideSpace::Linear => luma,
                    GuideSpace::Perceptual => linear_to_srgb(luma.max(0.0)),
                };
            }
            y0 += rows;
        }

        let aligned_path = self.scratch.join(format!("guide{index}.f32"));
        let mut aligned = ScratchPlane::create(&aligned_path, info.width, info.height)?;
        warp_plane(&native, &mut aligned, transform, info)?;

        drop(native);
        let _ = std::fs::remove_file(&native_path);
        Ok(aligned)
    }

    /// Guided-filter frame `k`'s one-hot selection mask against its own guide.
    fn refine(
        &self,
        index: usize,
        labels: &ScratchPlane,
        guide: &ScratchPlane,
    ) -> Result<ScratchPlane> {
        let (width, height) = (labels.width(), labels.height());
        let mut out = ScratchPlane::create(
            &self.scratch.join(format!("weight{index}.f32")),
            width,
            height,
        )?;
        let halo = 2 * self.radius;

        let mut y0 = 0;
        while y0 < height {
            let rows = BAND_ROWS.min(height - y0);
            // Two box passes each reach `radius`, so a `2*radius` halo keeps the
            // band's own edge clamping from reaching the rows we keep.
            let from = y0.saturating_sub(halo);
            let to = (y0 + rows + halo).min(height);
            let span = to - from;

            let g: Vec<f32> = guide.rows(from, span)?.to_vec();
            let p: Vec<f32> = labels
                .rows(from, span)?
                .iter()
                .map(|&l| if l as usize == index { 1.0 } else { 0.0 })
                .collect();

            let q = guided_filter(&g, &p, width, span, self.radius, self.epsilon);

            let band = out.rows_mut(y0, rows)?;
            let offset = (y0 - from) as usize * width as usize;
            for (i, slot) in band.iter_mut().enumerate() {
                // The filter can undershoot slightly; weights must stay non-negative.
                *slot = q[offset + i].max(0.0);
            }
            y0 += rows;
        }
        Ok(out)
    }
}

impl WeightEstimator for GuidedWeights {
    fn weights(&self, focus_maps: &[FocusMap], run: &dyn RunControl) -> Result<WeightMaps> {
        assert_eq!(
            focus_maps.len(),
            self.frames.len(),
            "one focus map per frame required"
        );
        // Three checkpoints, not one: `labels` and `normalize` are full banded passes
        // over every frame's plane, before and after the per-frame loop, so a check
        // placed only in the loop would leave both ends unstoppable.
        let labels = self.labels(focus_maps, run)?;

        // Per frame and independent: each builds its own guide from its own frame and
        // writes its own plane, so the only shared input is the read-only label field.
        let total = self.frames.len();
        let done = AtomicUsize::new(0);
        let mut indexed: Vec<(usize, ScratchPlane)> = (0..total)
            .into_par_iter()
            .map(|index| {
                if run.cancelled() {
                    return Err(Error::Cancelled);
                }
                let guide = self.guide(index)?;
                let plane = self.refine(index, &labels, &guide)?;
                drop(guide);
                let _ = std::fs::remove_file(self.scratch.join(format!("guide{index}.f32")));
                run.progress(
                    Stage::Weights,
                    done.fetch_add(1, Ordering::Relaxed) + 1,
                    total,
                );
                Ok((index, plane))
            })
            .collect::<Result<Vec<_>>>()?;

        // Restored to frame order before normalising: `normalize` sums across planes, and
        // float addition is not associative, so a completion-order shuffle would perturb
        // the output — silently, and differently on every run.
        indexed.sort_unstable_by_key(|(index, _)| *index);
        let mut planes: Vec<ScratchPlane> = indexed.into_iter().map(|(_, p)| p).collect();

        normalize(&mut planes, run)?;
        Ok(planes)
    }
}

/// He, Sun & Tang's guided filter: the output is a locally linear function of the
/// guide, `q = a*I + b`, with the coefficients averaged over overlapping windows.
///
/// `epsilon` sets how much guide variance counts as an edge worth preserving rather
/// than noise worth smoothing.
fn guided_filter(
    guide: &[f32],
    input: &[f32],
    width: u32,
    height: u32,
    radius: u32,
    epsilon: f32,
) -> Vec<f32> {
    let mean_i = box_mean(guide, width, height, radius);
    let mean_p = box_mean(input, width, height, radius);
    let corr_i = box_mean(&mul(guide, guide), width, height, radius);
    let corr_ip = box_mean(&mul(guide, input), width, height, radius);

    let mut a = vec![0f32; guide.len()];
    let mut b = vec![0f32; guide.len()];
    for i in 0..guide.len() {
        let var_i = corr_i[i] - mean_i[i] * mean_i[i];
        let cov_ip = corr_ip[i] - mean_i[i] * mean_p[i];
        a[i] = cov_ip / (var_i + epsilon);
        b[i] = mean_p[i] - a[i] * mean_i[i];
    }

    let mean_a = box_mean(&a, width, height, radius);
    let mean_b = box_mean(&b, width, height, radius);
    (0..guide.len())
        .map(|i| mean_a[i] * guide[i] + mean_b[i])
        .collect()
}

/// Scale each pixel's weights across frames to sum to one.
fn normalize(planes: &mut [ScratchPlane], run: &dyn RunControl) -> Result<()> {
    let (width, height) = (planes[0].width(), planes[0].height());

    let mut y0 = 0;
    while y0 < height {
        if run.cancelled() {
            return Err(Error::Cancelled);
        }
        let rows = BAND_ROWS.min(height - y0);
        let len = rows as usize * width as usize;

        let mut totals = vec![0f32; len];
        for plane in planes.iter() {
            for (i, &v) in plane.rows(y0, rows)?.iter().enumerate() {
                totals[i] += v;
            }
        }

        let count = planes.len() as f32;
        for plane in planes.iter_mut() {
            let band = plane.rows_mut(y0, rows)?;
            for (i, slot) in band.iter_mut().enumerate() {
                // Nothing in focus anywhere: fall back to an even blend rather than
                // leaving a hole in the output.
                *slot = if totals[i] > 1e-8 {
                    *slot / totals[i]
                } else {
                    1.0 / count
                };
            }
        }
        y0 += rows;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guided_filter_of_a_constant_input_is_that_constant() {
        let guide: Vec<f32> = (0..400).map(|i| (i % 17) as f32 / 17.0).collect();
        let input = vec![0.3f32; 400];
        for v in guided_filter(&guide, &input, 20, 20, 3, 1e-3) {
            assert!((v - 0.3).abs() < 1e-4, "{v}");
        }
    }

    #[test]
    fn guided_filter_keeps_an_edge_the_guide_shares() {
        // Guide and input both step at x = 10; the step must survive.
        let (w, h) = (20u32, 20u32);
        let mut guide = vec![0f32; 400];
        let mut input = vec![0f32; 400];
        for y in 0..h {
            for x in 0..w {
                let step = if x < 10 { 0.0 } else { 1.0 };
                guide[(y * w + x) as usize] = step;
                input[(y * w + x) as usize] = step;
            }
        }

        let q = guided_filter(&guide, &input, w, h, 3, 1e-6);
        let left = q[(10 * w + 6) as usize];
        let right = q[(10 * w + 13) as usize];
        assert!(left < 0.05, "left of edge should stay low: {left}");
        assert!(right > 0.95, "right of edge should stay high: {right}");
    }

    #[test]
    fn guided_filter_smooths_where_the_guide_is_featureless() {
        // Flat guide, noisy input: with no edges to preserve, this reduces to a
        // box mean, which is exactly the incoherence-removal we want off-edge.
        let (w, h) = (20u32, 20u32);
        let guide = vec![0.5f32; 400];
        let input: Vec<f32> = (0..400)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();

        let q = guided_filter(&guide, &input, w, h, 3, 1e-3);
        let interior = q[(10 * w + 10) as usize];
        assert!(
            (interior - 0.5).abs() < 0.1,
            "should average toward 0.5, got {interior}"
        );
    }

    #[test]
    fn perceptual_guide_lifts_dark_edges_relative_to_highlights() {
        // The claim behind GuideSpace::Perceptual: a dark-on-mid edge and a
        // highlight-region edge that look comparable to the eye do not have
        // comparable linear-domain magnitudes.
        let dark_edge = (0.02f32, 0.10);
        let bright_edge = (0.70f32, 0.90);

        let linear_dark = dark_edge.1 - dark_edge.0;
        let linear_bright = bright_edge.1 - bright_edge.0;
        let perceptual_dark = linear_to_srgb(dark_edge.1) - linear_to_srgb(dark_edge.0);
        let perceptual_bright = linear_to_srgb(bright_edge.1) - linear_to_srgb(bright_edge.0);

        assert!(
            linear_dark < linear_bright,
            "linear light understates the dark edge"
        );
        assert!(
            perceptual_dark > perceptual_bright,
            "gamma encoding should restore its precedence: {perceptual_dark} vs {perceptual_bright}"
        );
    }
}
