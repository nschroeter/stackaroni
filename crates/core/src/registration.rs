//! Phase-correlation registration (Kuglin & Hines, 1975).
//!
//! Estimates pure translation between two frames from the phase of their
//! cross-power spectrum. `docs/algorithms.md` §10 places this first in the
//! translation → affine progression; ECC or feature-based affine registration would
//! be a second implementation of [`Registration`], deliberately not built yet.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

use crate::budget;
use crate::error::{Error, Result};
use crate::grid::Grid;
use crate::pipeline::{Image, Registration, RunControl, Stage, Transform};
use crate::tiff_io::cache_bytes_max;

/// Whitening regularizer, as a fraction of the strongest cross-spectrum bin.
const WHITENING_EPS: f32 = 1e-3;

/// Log-polar resampling grid. `RHO` bins span `R_MIN..r_max` logarithmically, so
/// scale resolution is `(r_max/R_MIN)^(1/RHO)` per bin — with the defaults below
/// that is ~0.5% per bin before sub-pixel refinement, against the ~0.145% per-frame
/// magnification measured on `ruler`.
const LOG_POLAR_RHO: u32 = 1024;
const LOG_POLAR_THETA: u32 = 512;
const LOG_POLAR_R_MIN: f32 = 4.0;

/// Phase correlation at a fixed pyramid level.
///
/// `level` trades resolution for robustness: adjacent frames in a focus bracket do
/// not share their high-frequency content — what is sharp in one is defocused in the
/// next — and the whitened spectrum weights every frequency equally, so that
/// mismatched detail is noise in the correlation. Downsampling suppresses exactly
/// those frequencies and leaves the coarse structure the frames do share to drive
/// the peak. See the `registration_accuracy` benchmark for the measured tradeoff.
pub struct PhaseCorrelation {
    level: u32,
}

impl PhaseCorrelation {
    pub fn new(level: u32) -> Self {
        Self { level }
    }

    pub fn level(&self) -> u32 {
        self.level
    }
}

impl Registration for PhaseCorrelation {
    /// `run` is accepted and not polled: one call is one frame pair, so the loop that
    /// could be stopped lives in [`register_stack`], which does check. An `align` that
    /// iterates internally — ECC affine, `docs/algorithms.md` §10 — would poll it.
    fn align(&self, reference: &Image, target: &Image, run: &dyn RunControl) -> Result<Transform> {
        let _ = run;
        let a = Grid::from_image(reference, self.level)?;
        let b = Grid::from_image(target, self.level)?;
        let est = correlate_similarity(&a, &b);
        Ok(scale_up(est.transform, self.level))
    }
}

/// Translations are in level pixels; convert to full resolution. Scale is a ratio,
/// so it carries across levels unchanged.
fn scale_up(t: Transform, level: u32) -> Transform {
    let s = (1u32 << level) as f32;
    Transform {
        scale: t.scale,
        dx: t.dx * s,
        dy: t.dy * s,
    }
}

/// Translation `d` such that `b(x) ≈ a(x - d)`, refined to sub-pixel precision.
///
/// Both grids must be the same size.
pub fn correlate(a: &Grid, b: &Grid) -> (f32, f32) {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "phase correlation needs equally sized grids"
    );

    // Zero-pad to powers of two: keeps the FFT on its fast path, and the padding
    // also stops the circular correlation from wrapping real content around.
    let fw = (a.width as usize + a.width as usize / 2).next_power_of_two();
    let fh = (a.height as usize + a.height as usize / 2).next_power_of_two();

    let mut planner = FftPlanner::<f32>::new();
    let mut spec_a = prepare(a, fw, fh);
    let mut spec_b = prepare(b, fw, fh);
    fft2(&mut spec_a, fw, fh, &mut planner, false);
    fft2(&mut spec_b, fw, fh, &mut planner, false);

    // Cross-power spectrum, whitened: conj(A)·B / (|conj(A)·B| + eps).
    //
    // The eps matters. Plain whitening normalizes every bin to unit magnitude,
    // including bins carrying no signal at all, so empty parts of the spectrum
    // contribute to the impulse as strongly as real structure. Regularizing by a
    // fraction of the strongest bin leaves well-supported frequencies untouched
    // while letting empty ones fall away — the phase-transform (GCC-PHAT)
    // regularization.
    let products: Vec<Complex32> = spec_a
        .iter()
        .zip(&spec_b)
        .map(|(&av, &bv)| av.conj() * bv)
        .collect();
    let eps = WHITENING_EPS * products.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
    let mut cross: Vec<Complex32> = products.iter().map(|&c| c / (c.norm() + eps)).collect();
    fft2(&mut cross, fw, fh, &mut planner, true);

    // The impulse sits at the displacement.
    let mut peak = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, c) in cross.iter().enumerate() {
        let m = c.norm();
        if m > best {
            best = m;
            peak = i;
        }
    }
    let px = peak % fw;
    let py = peak / fw;

    let mag = |x: usize, y: usize| cross[(y % fh) * fw + (x % fw)].norm();
    let dx = px as f32 + parabolic(best, mag((px + fw - 1) % fw, py), mag((px + 1) % fw, py));
    let dy = py as f32 + parabolic(best, mag(px, (py + fh - 1) % fh), mag(px, (py + 1) % fh));

    (wrap(dx, fw), wrap(dy, fh))
}

/// Sub-pixel offset of a peak from a parabola through it and its two neighbours.
///
/// An integer-only argmax carries up to half a pixel of error, which is the scale
/// that shows as a seam on a 1-3 px antenna line.
fn parabolic(mid: f32, left: f32, right: f32) -> f32 {
    let denom = left - 2.0 * mid + right;
    if denom.abs() < 1e-20 {
        0.0
    } else {
        0.5 * (left - right) / denom
    }
}

/// Map a periodic coordinate into `[-n/2, n/2)`, so shifts read as signed.
fn wrap(v: f32, n: usize) -> f32 {
    if v > n as f32 / 2.0 { v - n as f32 } else { v }
}

/// Mean-subtract, Hann-window and zero-pad a grid into an FFT buffer.
///
/// The window suppresses the spectral leakage a non-periodic frame edge would
/// otherwise inject into every frequency bin.
fn prepare(grid: &Grid, fw: usize, fh: usize) -> Vec<Complex32> {
    let mean = grid.mean();
    let wx = hann(grid.width as usize);
    let wy = hann(grid.height as usize);

    let mut buf = vec![Complex32::default(); fw * fh];
    for y in 0..grid.height as usize {
        for x in 0..grid.width as usize {
            let v = (grid.at(x as u32, y as u32) - mean) * wx[x] * wy[y];
            buf[y * fw + x] = Complex32::new(v, 0.0);
        }
    }
    buf
}

fn hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| {
            let t = 2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32;
            0.5 - 0.5 * t.cos()
        })
        .collect()
}

/// In-place 2D FFT: rows, then columns.
fn fft2(buf: &mut [Complex32], w: usize, h: usize, planner: &mut FftPlanner<f32>, inverse: bool) {
    let rows = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    for y in 0..h {
        rows.process(&mut buf[y * w..(y + 1) * w]);
    }

    let cols = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut col = vec![Complex32::default(); h];
    for x in 0..w {
        for (y, slot) in col.iter_mut().enumerate() {
            *slot = buf[y * w + x];
        }
        cols.process(&mut col);
        for (y, &v) in col.iter().enumerate() {
            buf[y * w + x] = v;
        }
    }
}

/// Register a whole stack against its middle frame.
///
/// The middle is the anchor and the chain runs outward in both directions: focus
/// breathing drifts monotonically across a bracket, so anchoring at one end would
/// give the far end both the largest cumulative transform and the most accumulated
/// chaining error — the frames most likely to show ghosting.
///
/// Returns one [`Transform`] per frame, relative to the anchor.
pub fn register_stack(
    registration: &dyn Registration,
    frames: &[PathBuf],
    run: &dyn RunControl,
) -> Result<Vec<Transform>> {
    let n = frames.len();
    let mut transforms = vec![Transform::IDENTITY; n];
    if n < 2 {
        return Ok(transforms);
    }
    let anchor = n / 2;

    // Split into the part that can go wide and the part that cannot.
    //
    // Each *pairwise* alignment is independent — it reads two frames and returns one
    // transform — so all `n - 1` of them run concurrently. Only the *chaining* is
    // sequential, because `transforms[i]` composes onto `transforms[i-1]`, and that is
    // a handful of multiplies over the whole stack rather than the ~1.6 s per pair the
    // alignment costs. Composing afterwards in index order also keeps the result
    // independent of the order the threads happen to finish in.
    let done = AtomicUsize::new(0);
    // Two frames per task: a pair alignment holds its reference and its target at the
    // same time. On striped input this charges ~26 MB and the cap does not bind; on
    // single-strip input it is 600 MB, and 14 concurrent pairs would be 8.4 GB.
    let per_task = 2 * cache_bytes_max(&frames[0])?;
    let steps: Vec<Transform> = budget::run_bounded(budget::concurrency_for(per_task), || {
        (0..n)
            .into_par_iter()
            .filter(|&i| i != anchor)
            .map(|i| {
                if run.cancelled() {
                    return Err(Error::Cancelled);
                }
                // Outward from the anchor: frames after it align against their predecessor,
                // frames before it against their successor.
                let reference = if i > anchor { i - 1 } else { i + 1 };
                let reference = Image::open(&frames[reference])?;
                let curr = Image::open(&frames[i])?;
                let step = registration.align(&reference, &curr, run)?;
                run.progress(
                    Stage::Register,
                    done.fetch_add(1, Ordering::Relaxed) + 1,
                    n - 1,
                );
                Ok((i, step))
            })
            .collect::<Result<Vec<_>>>()
            .map(|mut pairs| {
                pairs.sort_unstable_by_key(|(i, _)| *i);
                let mut steps = vec![Transform::IDENTITY; n];
                for (i, step) in pairs {
                    steps[i] = step;
                }
                steps
            })
    })?;

    for i in anchor + 1..n {
        transforms[i] = transforms[i - 1].then(steps[i]);
    }
    for i in (0..anchor).rev() {
        transforms[i] = transforms[i + 1].then(steps[i]);
    }

    Ok(transforms)
}

/// A similarity estimate plus the rotation that was measured but not modelled.
#[derive(Debug, Clone, Copy)]
pub struct SimilarityEstimate {
    pub transform: Transform,
    /// Rotation in degrees, from the same log-polar correlation that gives scale.
    ///
    /// Not part of [`Transform`]: no rotation showed in the per-region diagnostics
    /// on `ruler`, so it is reported as evidence rather than applied. If this starts
    /// coming back non-zero, that is the signal to escalate to ECC affine.
    pub rotation_degrees: f32,
}

/// Estimate a similarity transform: uniform scale then translation.
///
/// Reddy & Chatterji (IEEE TIP 5(8), 1996, 1266-1271). The magnitude of an image's
/// Fourier transform is invariant to translation, so scale can be recovered from it
/// alone: resampled into log-polar coordinates, a uniform magnification becomes a
/// pure shift along the log-radius axis, which the translation correlation already
/// solves. Undo the scale, then correlate normally for the translation.
pub fn correlate_similarity(a: &Grid, b: &Grid) -> SimilarityEstimate {
    let (la, log_base) = log_polar(&magnitude_spectrum(a));
    let (lb, _) = log_polar(&magnitude_spectrum(b));

    let (d_rho, d_theta) = correlate(&la, &lb);
    let scale = log_base.powf(-d_rho);
    let rotation_degrees = -d_theta * 180.0 / LOG_POLAR_THETA as f32;

    // Remove the magnification, then the residual is pure translation.
    let unscaled = b.warped(Transform {
        scale: 1.0 / scale,
        dx: 0.0,
        dy: 0.0,
    });
    let (dx, dy) = correlate(a, &unscaled);

    SimilarityEstimate {
        transform: Transform { scale, dx, dy },
        rotation_degrees,
    }
}

/// Translation-invariant magnitude spectrum, centred and high-pass filtered.
///
/// The high-pass is Reddy & Chatterji's `(1-X)(2-X)` with `X = cos(pi u)cos(pi v)`:
/// without it the log-polar correlation is dominated by the low-frequency bulk that
/// every photograph shares, which carries no scale information.
fn magnitude_spectrum(grid: &Grid) -> Grid {
    let fw = (grid.width as usize).next_power_of_two();
    let fh = (grid.height as usize).next_power_of_two();

    let mut planner = FftPlanner::<f32>::new();
    let mut buf = prepare(grid, fw, fh);
    fft2(&mut buf, fw, fh, &mut planner, false);

    let mut out = Grid::new(fw as u32, fh as u32);
    for y in 0..fh {
        for x in 0..fw {
            // Shift the origin to the centre so log-polar can sample radially.
            let sx = (x + fw / 2) % fw;
            let sy = (y + fh / 2) % fh;
            let u = x as f32 / fw as f32 - 0.5;
            let v = y as f32 / fh as f32 - 0.5;
            let cross = (std::f32::consts::PI * u).cos() * (std::f32::consts::PI * v).cos();
            let highpass = (1.0 - cross) * (2.0 - cross);
            // log1p keeps the huge dynamic range of a magnitude spectrum usable.
            out.data[sy * fw + sx] = (buf[y * fw + x].norm() * highpass).ln_1p();
        }
    }
    out
}

/// Resample a centred spectrum into log-polar coordinates.
///
/// Rows are angle over `[0, pi)` — a magnitude spectrum is centrally symmetric, so
/// half a turn covers it. Columns are log-radius. Returns the grid and the log base,
/// which converts a shift along rho back into a scale factor.
fn log_polar(spectrum: &Grid) -> (Grid, f32) {
    let (cx, cy) = (spectrum.width as f32 / 2.0, spectrum.height as f32 / 2.0);
    let r_max = cx.min(cy);
    let log_base = (r_max / LOG_POLAR_R_MIN).powf(1.0 / LOG_POLAR_RHO as f32);

    let mut out = Grid::new(LOG_POLAR_RHO, LOG_POLAR_THETA);
    for t in 0..LOG_POLAR_THETA {
        let theta = std::f32::consts::PI * t as f32 / LOG_POLAR_THETA as f32;
        let (sin, cos) = theta.sin_cos();
        for r in 0..LOG_POLAR_RHO {
            let radius = LOG_POLAR_R_MIN * log_base.powi(r as i32);
            out.data[t as usize * LOG_POLAR_RHO as usize + r as usize] =
                spectrum.sample(cx + radius * cos, cy + radius * sin);
        }
    }
    (out, log_base)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Broadband texture, defined analytically so a shifted copy can be generated
    /// without the zero-filled edge `Grid::shifted` leaves behind — that strip
    /// self-correlates into a spurious peak at the origin and would be testing the
    /// fixture rather than the estimator.
    ///
    /// Broadband matters: a handful of sinusoids leaves almost every FFT bin empty,
    /// which whitening then amplifies into noise that buries the impulse.
    /// Photographs have energy across the spectrum, so a sparse fixture would be
    /// testing a case the pipeline never sees.
    fn texture_at(x: f32, y: f32) -> f32 {
        let mut seed = 0x9e3779b9u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32
        };
        (0..48)
            .map(|_| {
                let fx = (next() - 0.5) * 2.4;
                let fy = (next() - 0.5) * 2.4;
                let phase = next() * std::f32::consts::TAU;
                (fx * x + fy * y + phase).sin()
            })
            .sum::<f32>()
            / 48.0
    }

    /// Grid of `texture_at`, displaced by `(dx, dy)`.
    fn textured(width: u32, height: u32, dx: f32, dy: f32) -> Grid {
        let mut g = Grid::new(width, height);
        for y in 0..height {
            for x in 0..width {
                g.data[y as usize * width as usize + x as usize] =
                    texture_at(x as f32 - dx, y as f32 - dy);
            }
        }
        g
    }

    #[test]
    fn recovers_whole_pixel_shifts() {
        let a = textured(128, 96, 0.0, 0.0);
        for &(dx, dy) in &[(0.0, 0.0), (5.0, 0.0), (0.0, -4.0), (-7.0, 3.0)] {
            let b = textured(128, 96, dx, dy);
            let (gx, gy) = correlate(&a, &b);
            assert!(
                (gx - dx).abs() < 0.1 && (gy - dy).abs() < 0.1,
                "want ({dx},{dy}) got ({gx:.3},{gy:.3})"
            );
        }
    }

    #[test]
    fn recovers_sub_pixel_shifts() {
        let a = textured(128, 96, 0.0, 0.0);
        for &(dx, dy) in &[(2.5, 0.0), (0.0, 1.25), (-3.4, 2.6), (0.75, -0.25)] {
            let b = textured(128, 96, dx, dy);
            let (gx, gy) = correlate(&a, &b);
            // An integer-only argmax would be off by up to 0.5 on every one of these.
            assert!(
                (gx - dx).abs() < 0.2 && (gy - dy).abs() < 0.2,
                "want ({dx},{dy}) got ({gx:.3},{gy:.3})"
            );
        }
    }

    #[test]
    fn sub_pixel_refinement_beats_integer_argmax() {
        let a = textured(128, 96, 0.0, 0.0);
        let (dx, dy) = (3.4, -2.6);
        let b = textured(128, 96, dx, dy);
        let (gx, gy) = correlate(&a, &b);
        let integer_err = (gx.round() - dx).abs().max((gy.round() - dy).abs());
        let refined_err = (gx - dx).abs().max((gy - dy).abs());
        assert!(
            refined_err < integer_err,
            "refined {refined_err:.3} should beat integer {integer_err:.3}"
        );
    }

    #[test]
    fn correlation_is_antisymmetric() {
        let a = textured(128, 96, 0.0, 0.0);
        let b = textured(128, 96, 3.5, -2.25);
        let (fx, fy) = correlate(&a, &b);
        let (rx, ry) = correlate(&b, &a);
        assert!(
            (fx + rx).abs() < 0.1 && (fy + ry).abs() < 0.1,
            "forward ({fx:.3},{fy:.3}) reverse ({rx:.3},{ry:.3})"
        );
    }

    #[test]
    fn recovers_uniform_scale() {
        let a = textured(256, 256, 0.0, 0.0);
        for want in [1.02f32, 0.97, 1.005] {
            let b = a.warped(Transform {
                scale: want,
                dx: 0.0,
                dy: 0.0,
            });
            let got = correlate_similarity(&a, &b).transform.scale;
            assert!((got - want).abs() < 0.004, "want scale {want} got {got:.5}");
        }
    }

    #[test]
    fn recovers_scale_and_translation_together() {
        let a = textured(256, 256, 0.0, 0.0);
        let want = Transform {
            scale: 1.03,
            dx: 4.0,
            dy: -3.0,
        };
        let b = a.warped(want);
        let got = correlate_similarity(&a, &b).transform;
        assert!((got.scale - want.scale).abs() < 0.005, "scale {got:?}");
        assert!((got.dx - want.dx).abs() < 1.0, "dx {got:?}");
        assert!((got.dy - want.dy).abs() < 1.0, "dy {got:?}");
    }

    #[test]
    fn reports_no_rotation_when_there_is_none() {
        let a = textured(256, 256, 0.0, 0.0);
        let b = a.warped(Transform {
            scale: 1.02,
            dx: 2.0,
            dy: 1.0,
        });
        let rot = correlate_similarity(&a, &b).rotation_degrees;
        assert!(rot.abs() < 1.0, "unexpected rotation {rot:.3} deg");
    }

    #[test]
    fn composition_matches_applying_transforms_in_turn() {
        let first = Transform {
            scale: 1.1,
            dx: 3.0,
            dy: -2.0,
        };
        let second = Transform {
            scale: 0.95,
            dx: -1.0,
            dy: 4.0,
        };
        let (x, y) = (7.0f32, -5.0f32);
        let (ax, ay) = first.apply(x, y);
        let (bx, by) = second.apply(ax, ay);
        let (cx, cy) = first.then(second).apply(x, y);
        assert!((bx - cx).abs() < 1e-4 && (by - cy).abs() < 1e-4);
    }

    #[test]
    fn inverse_undoes_the_transform() {
        let t = Transform {
            scale: 1.07,
            dx: 5.0,
            dy: -3.0,
        };
        let (x, y) = t.then(t.inverse()).apply(9.0, -4.0);
        assert!((x - 9.0).abs() < 1e-4 && (y + 4.0).abs() < 1e-4);
    }

    #[test]
    fn parabolic_offset_is_zero_at_a_symmetric_peak() {
        assert!(parabolic(1.0, 0.5, 0.5).abs() < 1e-9);
        assert!(parabolic(1.0, 0.5, 0.9) > 0.0, "peak leans right");
        assert!(parabolic(1.0, 0.9, 0.5) < 0.0, "peak leans left");
    }
}
