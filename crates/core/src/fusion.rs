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
//!
//! **That claim was false in one respect until T18, and the exception is worth
//! knowing.** It describes what this module *allocates*, and said nothing about what the
//! `&[Image]` handed to it holds: each reader keeps its decoded strips for its own
//! lifetime, the caller owns every reader for the whole stage, and on single-strip input
//! one strip is one whole frame. So the frame count was back in the budget by the side
//! door — 25 GB on 33 frames. The per-frame `release_cache` in both `fuse` loops is what
//! makes the paragraph above true of the stage rather than only of its own buffers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::filter::box_sum;
use crate::image::{Coverage, FrameInfo, ScratchPlane};
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
    /// Only the tests address pixels this way now — the hot paths all walk rows as
    /// contiguous slices so they can be split across threads.
    #[cfg(test)]
    fn index(&self, x: u32, y: u32) -> usize {
        (y as usize * self.width as usize + x as usize) * self.channels
    }

    pub fn new(width: u32, height: u32, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![0.0; width as usize * height as usize * channels],
        }
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
///
/// The blur is evaluated *only* at the samples that survive decimation. Blurring every
/// pixel and then keeping one in four — which is what this did — computes four times the
/// taps it needs, at every pyramid level of every frame, in the pipeline's most expensive
/// stage.
///
/// This is arithmetically identical rather than merely equivalent: the same kernel taps
/// are summed in the same order over the same clamped source samples, so it is the same
/// floating-point expression, not a rearrangement of one. The byte-identical output gate
/// is what holds that claim to account.
pub fn reduce(src: &Bitmap) -> Bitmap {
    reduce_data(&src.data, src.width, src.height, src.channels)
}

/// [`reduce`] over borrowed samples, so a source that is not a [`Bitmap`] does not have
/// to become one first.
///
/// The weight planes are the case this exists for: they are mmapped scratch, and copying
/// one into an owned `Bitmap` purely to reduce it cost an allocation and a 200 MB memcpy
/// per frame at 50 MP — 28% of the fuse stage's wall time, second only to the `box_sum`
/// that T20 parallelized. Reducing straight from the mapped rows skips both; the pages
/// still fault in, which is the part that is real work.
pub fn reduce_data(data: &[f32], width: u32, height: u32, channels: usize) -> Bitmap {
    let c = channels;
    let (w, h) = (width as i64, height as i64);
    let (out_w, out_h) = (width.div_ceil(2), height.div_ceil(2));
    let (src_row, out_row) = (width as usize * c, out_w as usize * c);

    // Rows are independent in both passes, and within a row the kernel taps are still
    // summed in ascending `k` order — the same additions in the same sequence, just
    // spread across cores. Float addition is not associative, so "same order" is doing
    // real work here, not being pedantic.
    let mut horizontal = Bitmap::new(out_w, height, c);
    horizontal
        .data
        .par_chunks_mut(out_row)
        .enumerate()
        .for_each(|(y, dst)| {
            let row = &data[y * src_row..][..src_row];
            for ox in 0..out_w as usize {
                let x = 2 * ox as i64;
                for (k, weight) in KERNEL.iter().enumerate() {
                    let sx = (x + k as i64 - 2).clamp(0, w - 1) as usize;
                    for ch in 0..c {
                        dst[ox * c + ch] += weight * row[sx * c + ch];
                    }
                }
            }
        });

    let mut out = Bitmap::new(out_w, out_h, c);
    out.data
        .par_chunks_mut(out_row)
        .enumerate()
        .for_each(|(oy, dst)| {
            let y = 2 * oy as i64;
            for (k, weight) in KERNEL.iter().enumerate() {
                let sy = (y + k as i64 - 2).clamp(0, h - 1) as usize;
                let row = &horizontal.data[sy * out_row..][..out_row];
                for (slot, &v) in dst.iter_mut().zip(row) {
                    *slot += weight * v;
                }
            }
        });
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
    let mut out = Bitmap::new(width, height, src.channels);
    expand_into(src, &mut out, |slot, value| *slot = value);
    out
}

/// Expand `src` to `dst`'s size, combining each expanded sample into `dst` in place.
///
/// **This exists so a band-pass level never has to be materialised.** Building one costs
/// `band - expand(coarser)`, and writing that expansion to its own image first spends a
/// full-resolution allocation, a pass to fill it and a pass to read it back — ~1.2 GB of
/// traffic per frame at 50 MP for a value used exactly once. Handing the subtraction in
/// means the expanded sample is consumed where it is computed.
///
/// `combine` receives the destination slot and the expanded sample, in that order, so the
/// caller decides the operation *and its operand order* — `*slot -= value` is not
/// `value - *slot`, and float subtraction does not forgive the difference.
fn expand_into(src: &Bitmap, dst: &mut Bitmap, combine: impl Fn(&mut f32, f32) + Sync) {
    let c = src.channels;
    let (width, height) = (dst.width, dst.height);
    debug_assert_eq!(c, dst.channels, "expand cannot change the channel count");
    let wide_row = width as usize * c;

    let mut wide = Bitmap::new(width, src.height, c);
    wide.data
        .par_chunks_mut(wide_row)
        .enumerate()
        .for_each(|(y, dst)| {
            for x in 0..width as usize {
                let i = (x / 2) as i64;
                let tap = |o: i64, ch: usize| {
                    let sx = o.clamp(0, src.width as i64 - 1) as usize;
                    src.data[(y * src.width as usize + sx) * c + ch]
                };
                expand_phase(&mut dst[x * c..][..c], x % 2 == 0, i, tap);
            }
        });

    debug_assert_eq!(dst.data.len(), wide_row * height as usize);
    debug_assert!(c <= 4, "channels beyond RGBA are not expected");
    dst.data
        .par_chunks_mut(wide_row)
        .enumerate()
        .for_each(|(y, dst)| {
            let j = (y / 2) as i64;
            // One pixel of scratch, so the expanded sample never needs an image of its own.
            let mut sample = [0f32; 4];
            for x in 0..width as usize {
                let tap = |o: i64, ch: usize| {
                    let sy = o.clamp(0, wide.height as i64 - 1) as usize;
                    wide.data[(sy * width as usize + x) * c + ch]
                };
                expand_phase(&mut sample[..c], y % 2 == 0, j, tap);
                for (slot, &value) in dst[x * c..][..c].iter_mut().zip(&sample[..c]) {
                    combine(slot, value);
                }
            }
        });
}

/// One output pixel of `expand`, all channels: the even phase takes `[1,6,1]/8`
/// centred on source index `i`, the odd phase `[1,1]/2` spanning `i` and `i+1`.
fn expand_phase(dst: &mut [f32], even: bool, i: i64, tap: impl Fn(i64, usize) -> f32) {
    for (ch, slot) in dst.iter_mut().enumerate() {
        *slot = if even {
            (tap(i - 1, ch) + 6.0 * tap(i, ch) + tap(i + 1, ch)) / 8.0
        } else {
            (tap(i, ch) + tap(i + 1, ch)) / 2.0
        };
    }
}

/// Successive reductions, finest first.
///
/// Takes the base by value: it becomes level 0, so a caller that still needs it must
/// clone deliberately. It used to clone unconditionally, and every caller in the pipeline
/// dropped its copy immediately afterwards — 600 MB of memcpy per frame at 50 MP, spent
/// so that the one caller who might have cared could avoid a clone it never wanted.
pub fn gaussian_pyramid(base: Bitmap, levels: usize) -> Vec<Bitmap> {
    let mut pyramid = Vec::with_capacity(levels);
    pyramid.push(base);
    for i in 1..levels {
        pyramid.push(reduce(&pyramid[i - 1]));
    }
    pyramid
}

/// Band-pass levels, with the coarsest Gaussian residual kept last so the pyramid
/// reconstructs exactly.
///
/// Each Gaussian level is *moved* into the band it becomes, rather than copied and then
/// dropped: a level is only ever read by the band below it, and by the time that band is
/// built its own coarser neighbour has already been expanded. Walking finest to coarsest
/// with one level of lookahead is what makes that ordering work — the reverse walk would
/// need a level after it had been consumed.
pub fn laplacian_pyramid(base: Bitmap, levels: usize) -> Vec<Bitmap> {
    let mut gaussian = gaussian_pyramid(base, levels).into_iter().peekable();
    let mut pyramid = Vec::with_capacity(levels);
    while let Some(mut band) = gaussian.next() {
        // The coarsest level has nothing below it and stays a Gaussian residual.
        let Some(coarser) = gaussian.peek() else {
            pyramid.push(band);
            break;
        };
        expand_into(coarser, &mut band, |slot, value| *slot -= value);
        pyramid.push(band);
    }
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
/// Which fusion rule to run, and every string that names it to a user.
///
/// One type because there were three: a `FusionArg` in the CLI, a `FusionRule` in the app,
/// and a `select_fusion: bool` in the app's `Settings`. They agreed, but only because a
/// later change made all three read the same default — before that they were independent
/// literals, and nothing checked them against each other.
///
/// The three strings are deliberately different from one another and all live here:
///
/// - [`FusionKind::token`] — `select` / `blend`, the CLI's spelling. **Frozen**:
///   `docs/eval-log.md` cites `--fusion select` in rows that must stay reproducible.
/// - [`FusionKind::label`] — what a photographer reads in the UI. Names the effect, not
///   the implementation; both rules are pyramid methods and saying so helps nobody
///   choosing between them.
/// - [`FusionKind::summary`] — the trade-off, in one sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionKind {
    /// Per-level selection by windowed salience (§6b). Each point is decided on its own
    /// local evidence, which is what `salience_radius` sizes — so the parameter travels
    /// with the variant that reads it rather than beside a rule that may ignore it.
    Select { salience_radius: u32 },
    /// Weighted blend of every level under the refined weight maps (§6). Takes no
    /// parameters of its own.
    Blend,
}

impl FusionKind {
    pub const ALL: [Self; 2] = [
        Self::Select {
            salience_radius: crate::defaults::SALIENCE_RADIUS,
        },
        Self::Blend,
    ];

    /// Accepted CLI spellings, in [`Self::ALL`] order, so clap's possible-values list and
    /// the parser cannot drift apart.
    pub const TOKENS: [&'static str; 2] = ["select", "blend"];

    pub const fn token(self) -> &'static str {
        match self {
            Self::Select { .. } => Self::TOKENS[0],
            Self::Blend => Self::TOKENS[1],
        }
    }

    /// Parses the rule alone. Any parameters it carries arrive separately — clap sees
    /// `--fusion` and `--salience-radius` as unrelated arguments — so this yields the
    /// default radius and [`Self::with_salience_radius`] applies the real one afterwards.
    pub fn from_token(token: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.token() == token)
    }

    /// Replaces the radius on a rule that has one, and is a no-op on one that does not.
    ///
    /// This is where "blend ignores `--salience-radius`" is enforced, rather than by a
    /// constructor politely dropping an argument it was handed.
    pub const fn with_salience_radius(self, radius: u32) -> Self {
        match self {
            Self::Select { .. } => Self::Select {
                salience_radius: radius,
            },
            Self::Blend => Self::Blend,
        }
    }

    /// Named for the *mechanism*, because that is what stays true.
    ///
    /// "Local" against a global alternative: the rule decides each point from its own
    /// neighbourhood rather than committing every point to one decision made once. It
    /// promises no outcome, and it does not collide with "select", which in this app
    /// already means choosing frames in the filmstrip.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Select { .. } => "Local",
            Self::Blend => "Blend",
        }
    }

    /// One sentence on what the rule does to the photograph, shown next to the choice.
    pub const fn summary(self) -> &'static str {
        match self {
            Self::Select { .. } => {
                "Decides each point from its own neighbourhood, taking the sharpest source \
                 there. Preserves fine, thin detail, with slightly more grain in defocused \
                 background."
            }
            Self::Blend => {
                "Averages the sources together. Smoother and quieter in the background, \
                 softer on fine detail — noticeably so on hair and antennae."
            }
        }
    }

    pub const fn is_select(self) -> bool {
        matches!(self, Self::Select { .. })
    }

    /// The one place either rule is constructed.
    ///
    /// Takes only what *every* rule needs. Per-rule parameters ride inside the variant, so
    /// adding a third rule with its own knobs does not widen this signature with arguments
    /// the other two ignore.
    pub fn build(
        self,
        output: &Path,
        transforms: HashMap<PathBuf, Transform>,
        floor: u32,
    ) -> Box<dyn ImageFusion> {
        match self {
            Self::Select { salience_radius } => Box::new(SelectionFusion::new(
                output,
                transforms,
                floor,
                salience_radius,
            )),
            Self::Blend => Box::new(LaplacianPyramidFusion::new(output, transforms, floor)),
        }
    }
}

impl std::fmt::Display for FusionKind {
    /// The CLI token, so `default_value_t` prints what the flag accepts.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.token())
    }
}

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
            gaussian_pyramid(seed, levels)
        };
        // Weight actually applied per level, so margin pixels — where some frames were
        // skipped and the weights no longer sum to one — can be rescaled afterwards.
        let mut applied: Vec<Vec<f32>> = accumulator
            .iter()
            .map(|b| vec![0f32; b.width as usize * b.height as usize])
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
            // Every frame is read exactly once, here, and never revisited — so its
            // cached strips are dead the moment the warp returns. Holding them is what
            // made peak memory scale with the *stack*: the caller owns all `images` for
            // the whole loop, and on single-strip input one cached strip is one whole
            // frame. Releasing here keeps this stage at one frame regardless of count.
            image.release_cache();
            let bands = laplacian_pyramid(warped, levels);

            // The weight map is already in anchor coordinates, so it is not warped
            // again here. Smoothing it down the pyramid is what makes each frequency
            // band blend at its own scale rather than all of them at pixel scale.
            let weight_bitmap = plane_to_bitmap(weight)?;
            let weight_levels = gaussian_pyramid(weight_bitmap, levels);

            let mut covered = Coverage::of(transform, info);
            for level in 0..levels {
                let (dst, src, w) = (
                    &mut accumulator[level],
                    &bands[level],
                    &weight_levels[level],
                );
                accumulate_weighted(dst, &mut applied[level], w, src, covered);
                covered = covered.reduced(src.width, src.height);
            }
            run.progress(Stage::Fuse, index + 1, images.len());
        }

        let mut every = every_frame_covers(&self.transforms, images, info, 0);
        for level in 0..levels {
            let (w, h) = (accumulator[level].width, accumulator[level].height);
            normalize_uncovered(&mut accumulator[level], &applied[level], every);
            every = every.reduced(w, h);
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
            gaussian_pyramid(seed, levels)
        };
        // Running best windowed salience per band-pass level. Negative so that the
        // first frame wins everywhere regardless of how flat it is.
        let mut best: Vec<Vec<f32>> = result[..levels - 1]
            .iter()
            .map(|b| vec![-1.0f32; (b.width as usize) * (b.height as usize)])
            .collect();
        // Weight actually applied at the base level, so margin pixels — where some
        // frames were skipped and the weights no longer sum to one — can be rescaled.
        let mut base_weight = {
            let base = &result[levels - 1];
            vec![0f32; base.width as usize * base.height as usize]
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
            // Read once, never revisited — see the same call in `LaplacianPyramidFusion`.
            image.release_cache();
            let bands = laplacian_pyramid(warped, levels);

            let mut covered = Coverage::of(transform, info);
            for level in 0..levels - 1 {
                select_more_salient(
                    &mut result[level],
                    &mut best[level],
                    &bands[level],
                    self.salience_radius,
                    covered,
                );
                covered = covered.reduced(bands[level].width, bands[level].height);
            }

            // Base level: the weight map reduced all the way down, blended as before.
            // Only the coarsest level is needed, so the intermediate levels are not kept.
            //
            // The first reduction reads the mapped plane directly. Materializing it as a
            // full-resolution `Bitmap` first was pure overhead — an allocation and a
            // 200 MB copy per frame at 50 MP — since nothing but this chain ever reads it.
            // A one-level pyramid has nothing to reduce, so it still needs the copy.
            let mut w = if levels > 1 {
                reduce_data(
                    weight.rows(0, weight.height())?,
                    weight.width(),
                    weight.height(),
                    1,
                )
            } else {
                plane_to_bitmap(weight)?
            };
            for _ in 2..levels {
                w = reduce(&w);
            }
            let (dst, src) = (&mut result[levels - 1], &bands[levels - 1]);
            accumulate_weighted(dst, &mut base_weight, &w, src, covered);
            run.progress(Stage::Fuse, index + 1, images.len());
        }

        normalize_uncovered(
            &mut result[levels - 1],
            &base_weight,
            every_frame_covers(&self.transforms, images, info, levels - 1),
        );
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
/// `covered` restricts the frame to the region it could fill without sampling outside
/// itself. Outside that region `warp_frame` has border-replicated, and a replicated
/// strip is constant along one axis while still carrying the border's texture along the
/// other — so it carries real salience and would win wherever the true content is
/// smooth. That is the frame-margin streaking of T15, and skipping those positions is
/// the whole fix: they are not compared, so they cannot win.
fn select_more_salient(
    dst: &mut Bitmap,
    best: &mut [f32],
    src: &Bitmap,
    radius: u32,
    covered: Coverage,
) {
    let n = best.len();
    let energy: Vec<f32> = (0..n)
        .map(|i| {
            let p = &src.data[i * 3..i * 3 + 3];
            p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
        })
        .collect();
    let salience = box_sum(&energy, src.width, src.height, radius);
    let full = covered.is_full(src.width, src.height);
    let width = src.width as usize;

    // Each position decides for itself, so splitting them changes nothing.
    dst.data
        .par_chunks_mut(3)
        .zip(best.par_iter_mut())
        .zip(salience.par_iter())
        .zip(src.data.par_chunks(3))
        .enumerate()
        .for_each(|(i, (((slot, best), &salience), source))| {
            let inside = full || covered.contains((i % width) as u32, (i / width) as u32);
            if inside && salience > *best {
                *best = salience;
                slot.copy_from_slice(source);
            }
        });
}

/// Add `weight * src` into `dst` over the covered region, tallying the weight applied.
///
/// Skipping uncovered positions is what keeps border-replicated pixels out of the
/// result; the tally is what lets the ones that were skipped be corrected afterwards.
fn accumulate_weighted(
    dst: &mut Bitmap,
    applied: &mut [f32],
    weight: &Bitmap,
    src: &Bitmap,
    covered: Coverage,
) {
    let full = covered.is_full(dst.width, dst.height);
    let width = dst.width as usize;
    for (i, applied) in applied.iter_mut().enumerate() {
        if !full && !covered.contains((i % width) as u32, (i / width) as u32) {
            continue;
        }
        let w = weight.data[i];
        *applied += w;
        for ch in 0..3 {
            dst.data[i * 3 + ch] += w * src.data[i * 3 + ch];
        }
    }
}

/// Rescale the pixels that did not receive every frame's weight, leaving the rest alone.
///
/// **`every` is not an optimisation, it is the correctness condition.** Inside it all
/// frames contributed and the weights already sum to one — but only to within the
/// rounding of a hundred additions, so dividing there would perturb pixels that no
/// frame was ever skipped for. Confining the division to the margin is what lets an
/// interior pixel come out bit-for-bit identical to a run without coverage tracking,
/// and therefore what lets every rating in `docs/eval-log.md` survive this change.
fn normalize_uncovered(dst: &mut Bitmap, applied: &[f32], every: Coverage) {
    let width = dst.width as usize;
    for (i, &applied) in applied.iter().enumerate() {
        if every.contains((i % width) as u32, (i / width) as u32) {
            continue;
        }
        // A pixel no frame covered cannot be recovered; leaving it is better than
        // dividing by zero, and `Coverage::of` guarantees the anchor covers everything.
        if applied <= 0.0 {
            continue;
        }
        for ch in 0..3 {
            dst.data[i * 3 + ch] /= applied;
        }
    }
}

/// The region every frame in the stack covers, at pyramid level `level`.
fn every_frame_covers(
    transforms: &HashMap<PathBuf, Transform>,
    images: &[Image],
    info: FrameInfo,
    level: usize,
) -> Coverage {
    let mut every = Coverage::full(info);
    for image in images {
        let transform = transforms
            .get(image.path())
            .copied()
            .unwrap_or(Transform::IDENTITY);
        every = every.intersect(Coverage::of(transform, info));
    }
    let (mut w, mut h) = (info.width, info.height);
    for _ in 0..level {
        every = every.reduced(w, h);
        (w, h) = (w.div_ceil(2), h.div_ceil(2));
    }
    every
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
pub(crate) fn warp_frame(image: &Image, transform: Transform, info: FrameInfo) -> Result<Bitmap> {
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

        let row_len = info.row_len();
        out.data[y0 as usize * row_len..][..rows as usize * row_len]
            .par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(r, dst)| {
                let y = (y0 + r as u32) as f32 - cy;
                for x in 0..info.width as usize {
                    let (sx, sy) = transform.apply(x as f32 - cx, y);
                    for ch in 0..3 {
                        dst[x * 3 + ch] = sample(sx + cx, sy + cy, ch);
                    }
                }
            });
        y0 += rows;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;

    /// Every token round-trips, and the sets cannot drift apart.
    ///
    /// `TOKENS` feeds clap's possible-values list while `from_token` does the parsing, so
    /// a variant added to one and not the other would give an accepted value that fails
    /// to parse — a runtime error clap's own type system cannot catch here.
    #[test]
    fn fusion_tokens_round_trip() {
        assert_eq!(FusionKind::ALL.len(), FusionKind::TOKENS.len());
        for kind in FusionKind::ALL {
            assert_eq!(FusionKind::from_token(kind.token()), Some(kind));
            assert!(FusionKind::TOKENS.contains(&kind.token()));
            assert_eq!(kind.to_string(), kind.token());
        }
        assert_eq!(FusionKind::from_token("Selection"), None);
        assert_eq!(FusionKind::from_token(""), None);
    }

    /// The strings a user sees are distinct from the ones a user types.
    ///
    /// The CLI tokens are frozen by `docs/eval-log.md`'s reproducibility, and the labels
    /// exist to say what changes in the photograph. Collapsing them to one string is the
    /// obvious "simplification" that would silently break either the log or the UI.
    #[test]
    fn labels_and_tokens_are_separate_vocabularies() {
        let select = FusionKind::Select { salience_radius: 2 };
        assert_eq!(select.token(), "select");
        assert_eq!(FusionKind::Blend.token(), "blend");
        assert_eq!(select.label(), "Local");
        assert_eq!(FusionKind::Blend.label(), "Blend");
        for kind in FusionKind::ALL {
            assert!(!kind.summary().is_empty());
        }
    }

    /// A rule's parameters belong to the rule.
    ///
    /// `with_salience_radius` is where "blend ignores --salience-radius" is enforced. If
    /// it ever silently constructed a `Select`, passing `--fusion blend --salience-radius
    /// 4` would quietly change the fusion rule rather than the parameter.
    #[test]
    fn only_the_rule_that_reads_a_radius_carries_one() {
        let tuned = FusionKind::from_token("select")
            .unwrap()
            .with_salience_radius(4);
        assert_eq!(tuned, FusionKind::Select { salience_radius: 4 });

        let blend = FusionKind::from_token("blend")
            .unwrap()
            .with_salience_radius(4);
        assert_eq!(blend, FusionKind::Blend, "blend must not gain a parameter");
    }

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
        let back = reconstruct(&laplacian_pyramid(base.clone(), levels));

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
        let back = reconstruct(&laplacian_pyramid(base.clone(), levels));
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
        let weight_levels = gaussian_pyramid(ones, levels);
        let bands = laplacian_pyramid(a.clone(), levels);

        let mut acc = gaussian_pyramid(Bitmap::new(32, 32, 3), levels);
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

    /// Coverage over the whole of a band — what a frame that never leaves itself has.
    /// These tests are about salience, so they hold coverage constant.
    fn full_cover(b: &Bitmap) -> Coverage {
        Coverage {
            x0: 0,
            y0: 0,
            x1: b.width,
            y1: b.height,
        }
    }

    /// The T15 fix, pinned at the level it acts on.
    ///
    /// A frame that had to be sampled outside itself carries border-replicated content
    /// in the margin. That content is not flat — replication is constant along one axis
    /// but still carries the border's texture along the other — so it wins on salience
    /// against genuinely smooth content, which is exactly what put coloured stripes down
    /// blossom's margins under both fusion rules. Restricting the frame to its covered
    /// region has to stop that, and stop it *only* there: the covering frame must still
    /// win everywhere it legitimately does.
    #[test]
    fn a_frame_cannot_win_outside_the_region_it_covers() {
        let (w, h) = (32u32, 16u32);
        let smooth = Bitmap::new(w, h, 3); // all zeros: a covering frame, no detail
        let replicated = detail_in(0..w, w, h); // high salience everywhere

        let mut dst = Bitmap::new(w, h, 3);
        let mut best = vec![-1.0f32; (w * h) as usize];
        select_more_salient(&mut dst, &mut best, &smooth, 1, full_cover(&smooth));
        // This frame reaches only the left half — the right half is its replicated margin.
        select_more_salient(
            &mut dst,
            &mut best,
            &replicated,
            1,
            Coverage {
                x0: 0,
                y0: 0,
                x1: 16,
                y1: h,
            },
        );

        for y in 2..h - 2 {
            let inside = dst.index(8, y);
            assert_eq!(
                dst.data[inside], replicated.data[inside],
                "inside its coverage the frame must still win, at (8,{y})"
            );
            let outside = dst.index(24, y);
            assert_eq!(
                dst.data[outside], 0.0,
                "outside its coverage the replicated margin must not win, at (24,{y})"
            );
        }
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
        let cover = full_cover(&dst);
        select_more_salient(&mut dst, &mut best, &left, 1, cover);
        select_more_salient(&mut dst, &mut best, &right, 1, cover);

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
        let cover = full_cover(&dst);
        select_more_salient(&mut dst, &mut best, &blue, 1, cover);
        select_more_salient(&mut dst, &mut best, &broad, 1, cover);

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
        let cover = full_cover(&dst);
        select_more_salient(&mut dst, &mut best, &structure, 2, cover);
        select_more_salient(&mut dst, &mut best, &spike, 2, cover);

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

        let mut result = gaussian_pyramid(Bitmap::new(32, 32, 3), levels);
        let mut best: Vec<Vec<f32>> = result[..levels - 1]
            .iter()
            .map(|b| vec![-1.0f32; (b.width * b.height) as usize])
            .collect();
        for source in [&dim, &sharp] {
            let bands = laplacian_pyramid(source.clone(), levels);
            for level in 0..levels - 1 {
                let cover = full_cover(&bands[level]);
                select_more_salient(
                    &mut result[level],
                    &mut best[level],
                    &bands[level],
                    2,
                    cover,
                );
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
