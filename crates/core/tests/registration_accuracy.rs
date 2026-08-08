//! Registration accuracy versus pyramid level, measured on a real stack.
//!
//! Run with:
//! ```text
//! cargo test --release -p stackaroni-core --test registration_accuracy -- --ignored --nocapture
//! ```
//!
//! There is no ground-truth transform for the real stacks, so accuracy is measured
//! two ways:
//!
//! 1. **Injected shift** — a real frame against a known-shifted copy of itself.
//!    Same content in both, so this isolates the estimator and the sub-pixel fit.
//!    Finer levels should win; if they don't, the interpolation is broken.
//! 2. **Defocus robustness** — real frame pairs at increasing index separation,
//!    which stands in for increasing defocus mismatch. Scored by self-consistency:
//!    antisymmetry (`align(a,b) == -align(b,a)`) and chain-versus-direct (summed
//!    single steps against one direct correlation). Both are zero for a perfect
//!    estimator and degrade when non-shared high-frequency content drives the peak.
//!
//! `ruler` is the instrument: a flat target, so a genuine mis-registration cannot be
//! confused with parallax.

use std::path::{Path, PathBuf};
use std::time::Instant;

use stackaroni_core::discovery::discover_stack;
use stackaroni_core::grid::Grid;
use stackaroni_core::pipeline::{FocusMetric, Image, Transform};
use stackaroni_core::registration::{correlate, correlate_similarity};

const LEVELS: [u32; 4] = [1, 2, 3, 4];
const INJECTED: [(f32, f32); 3] = [(3.37, -7.62), (0.5, 0.5), (-11.25, 4.8)];
const SEPARATIONS: [usize; 3] = [1, 5, 20];
const CHAIN: usize = 8;

fn frames() -> Option<Vec<PathBuf>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/ruler");
    discover_stack(&dir).ok().map(|s| s.frames)
}

fn grid(frames: &[PathBuf], index: usize, level: u32) -> Grid {
    Grid::from_image(&Image::open(&frames[index]).unwrap(), level).unwrap()
}

#[test]
#[ignore = "requires test-data/, run with --release"]
fn registration_accuracy() {
    let Some(frames) = frames() else {
        eprintln!("skipping: test-data/ruler not present");
        return;
    };
    let mid = frames.len() / 2;

    println!("\n=== 1. injected shift (same content, isolates the estimator) ===");
    println!(
        "{:>5}  {:>11}  {:>13}  {:>12}",
        "level", "grid", "mean err (px)", "max err (px)"
    );
    for level in LEVELS {
        let base = grid(&frames, mid, level);
        let margin = 24;
        let (cw, ch) = (base.width - 2 * margin, base.height - 2 * margin);
        let reference = base.crop(margin, margin, cw, ch);

        let mut errs = Vec::new();
        for (dx, dy) in INJECTED {
            // Crop away the zero-filled border the shift leaves behind.
            let moved = base.shifted(dx, dy).crop(margin, margin, cw, ch);
            let (gx, gy) = correlate(&reference, &moved);
            // Scale to full-resolution pixels, or the levels aren't comparable:
            // an error of 0.25 at level 4 is 4 full-res pixels.
            let scale = (1u32 << level) as f32;
            errs.push(((gx - dx).powi(2) + (gy - dy).powi(2)).sqrt() * scale);
        }
        let mean = errs.iter().sum::<f32>() / errs.len() as f32;
        let max = errs.iter().cloned().fold(0.0f32, f32::max);
        println!(
            "{level:>5}  {:>5}x{:<5}  {mean:>13.4}  {max:>12.4}",
            base.width, base.height
        );
    }

    println!("\n=== 2. defocus robustness (real pairs, self-consistency) ===");
    println!(
        "{:>5}  {:>4}  {:>10}  {:>10}  {:>9}",
        "level", "sep", "shift px", "antisym", "secs"
    );
    for level in LEVELS {
        for sep in SEPARATIONS {
            if mid + sep >= frames.len() {
                continue;
            }
            let start = Instant::now();
            let a = grid(&frames, mid, level);
            let b = grid(&frames, mid + sep, level);
            let (fx, fy) = correlate(&a, &b);
            let (rx, ry) = correlate(&b, &a);
            let scale = (1u32 << level) as f32;
            let antisym = ((fx + rx).powi(2) + (fy + ry).powi(2)).sqrt() * scale;
            let shift = (fx * fx + fy * fy).sqrt() * scale;
            println!(
                "{level:>5}  {sep:>4}  {shift:>10.3}  {antisym:>10.4}  {:>9.1}",
                start.elapsed().as_secs_f32()
            );
        }
    }

    println!("\n=== 3. chain vs direct ({CHAIN} steps from the middle) ===");
    println!(
        "{:>5}  {:>12}  {:>12}  {:>10}",
        "level", "chained px", "direct px", "disagree"
    );
    for level in LEVELS {
        let scale = (1u32 << level) as f32;
        let (mut cx, mut cy) = (0.0f32, 0.0f32);
        for i in 0..CHAIN {
            let a = grid(&frames, mid + i, level);
            let b = grid(&frames, mid + i + 1, level);
            let (sx, sy) = correlate(&a, &b);
            cx += sx;
            cy += sy;
        }
        let a = grid(&frames, mid, level);
        let b = grid(&frames, mid + CHAIN, level);
        let (dx, dy) = correlate(&a, &b);

        let disagree = ((cx - dx).powi(2) + (cy - dy).powi(2)).sqrt() * scale;
        println!(
            "{level:>5}  {:>12.3}  {:>12.3}  {disagree:>10.4}",
            (cx * cx + cy * cy).sqrt() * scale,
            (dx * dx + dy * dy).sqrt() * scale
        );
    }

    println!("\n=== 5. T5b: similarity correction (Reddy & Chatterji 1996) ===");
    println!(
        "{:>5}  {:>4}  {:>9}  {:>8}  {:>13}  {:>13}",
        "level", "sep", "scale", "rot deg", "spread before", "spread after"
    );
    for level in [2u32, 3] {
        for sep in [1usize, 5, 20] {
            let a = grid(&frames, mid, level);
            let b = grid(&frames, mid + sep, level);
            let scale_px = (1u32 << level) as f32;

            let est = correlate_similarity(&a, &b);
            // Bring the target back onto the reference and re-measure.
            let corrected = b.warped(est.transform.inverse());

            let before = region_spread(&a, &b) * scale_px;
            let after = region_spread(&a, &corrected) * scale_px;
            println!(
                "{level:>5}  {sep:>4}  {:>9.5}  {:>8.3}  {before:>13.3}  {after:>13.3}",
                est.transform.scale, est.rotation_degrees
            );
        }
    }

    // If the frames differ by pure translation, every region agrees on the shift.
    // If focus breathing is changing magnification, opposite regions disagree
    // symmetrically — which a translation-only Transform cannot represent.
    println!("\n=== 4. is it actually pure translation? (per-quadrant shift) ===");
    println!(
        "{:>5}  {:>4}  {:>16}  {:>16}  {:>10}",
        "level", "sep", "left dx / right dx", "top dy / bottom dy", "spread"
    );
    for level in [2u32, 3] {
        for sep in [1usize, 20] {
            let a = grid(&frames, mid, level);
            let b = grid(&frames, mid + sep, level);
            let scale = (1u32 << level) as f32;
            let (hw, hh) = (a.width / 2, a.height / 2);

            let half = |g: &Grid, x0, y0| g.crop(x0, y0, hw, hh);
            let (lx, _) = correlate(&half(&a, 0, hh / 2), &half(&b, 0, hh / 2));
            let (rx, _) = correlate(&half(&a, hw, hh / 2), &half(&b, hw, hh / 2));
            let (_, ty) = correlate(&half(&a, hw / 2, 0), &half(&b, hw / 2, 0));
            let (_, by) = correlate(&half(&a, hw / 2, hh), &half(&b, hw / 2, hh));

            let spread = ((lx - rx).abs()).max((ty - by).abs()) * scale;
            println!(
                "{level:>5}  {sep:>4}  {:>7.3} /{:>7.3}  {:>7.3} /{:>7.3}  {spread:>10.3}",
                lx * scale,
                rx * scale,
                ty * scale,
                by * scale
            );
        }
    }
    println!();
}

/// Largest disagreement between opposite halves, in grid pixels. Zero if the two
/// frames really do differ by a pure translation.
fn region_spread(a: &Grid, b: &Grid) -> f32 {
    let (hw, hh) = (a.width / 2, a.height / 2);
    let half = |g: &Grid, x0, y0| g.crop(x0, y0, hw, hh);
    let (lx, _) = correlate(&half(a, 0, hh / 2), &half(b, 0, hh / 2));
    let (rx, _) = correlate(&half(a, hw, hh / 2), &half(b, hw, hh / 2));
    let (_, ty) = correlate(&half(a, hw / 2, 0), &half(b, hw / 2, 0));
    let (_, by) = correlate(&half(a, hw / 2, hh), &half(b, hw / 2, hh));
    (lx - rx).abs().max((ty - by).abs())
}

#[test]
#[ignore = "requires test-data/, run with --release"]
fn write_alignment_overlays() {
    let Some(frames) = frames() else { return };
    let mid = frames.len() / 2;
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out");
    std::fs::create_dir_all(&out).unwrap();

    for sep in [1usize, 20] {
        let a = grid(&frames, mid, 3);
        let b = grid(&frames, mid + sep, 3);
        let (dx, dy) = correlate(&a, &b);
        stackaroni_core::debug::write_alignment_overlay(
            &out.join(format!("align_sep{sep:02}_translation.png")),
            &a,
            &b,
            Transform::translation(dx, dy),
        )
        .unwrap();

        let est = correlate_similarity(&a, &b);
        stackaroni_core::debug::write_alignment_overlay(
            &out.join(format!("align_sep{sep:02}_similarity.png")),
            &a,
            &b,
            est.transform,
        )
        .unwrap();
        println!(
            "sep {sep}: translation ({dx:.3},{dy:.3}) vs similarity {:?}",
            est.transform
        );
    }
    println!("wrote {}", out.display());
}

/// Noise floor of the `region_spread` metric and of the estimator on *shared*
/// content: warp one real frame by a known similarity, estimate it, correct, and
/// measure. Whatever residual survives here is measurement error, not misalignment,
/// and sets the bar the real-pair numbers should be read against.
#[test]
#[ignore = "requires test-data/, run with --release"]
fn similarity_noise_floor() {
    let Some(frames) = frames() else { return };
    let mid = frames.len() / 2;

    println!("\n=== control: known similarity on shared content ===");
    println!(
        "{:>5}  {:>9}  {:>9}  {:>12}  {:>12}",
        "level", "true", "estimated", "spread before", "spread after"
    );
    for level in [2u32, 3] {
        let a = grid(&frames, mid, level);
        for truth in [0.999f32, 0.994, 0.958] {
            let t = Transform {
                scale: truth,
                dx: 0.0,
                dy: 0.0,
            };
            let b = a.warped(t);
            let est = correlate_similarity(&a, &b);
            let corrected = b.warped(est.transform.inverse());
            let px = (1u32 << level) as f32;
            println!(
                "{level:>5}  {truth:>9.5}  {:>9.5}  {:>12.3}  {:>12.3}",
                est.transform.scale,
                region_spread(&a, &b) * px,
                region_spread(&a, &corrected) * px
            );
        }
    }
    println!();
}

/// Focus-map heatmaps for visual inspection, per CLAUDE.md's debug-output rule.
#[test]
#[ignore = "requires test-data/, run with --release"]
fn write_focus_heatmaps() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/synthetic_50");
    let Ok(stack) = discover_stack(&dir) else {
        return;
    };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out");
    let scratch = out.join("scratch");
    std::fs::create_dir_all(&scratch).unwrap();

    // Frame 1 is focused at the front, frame 25 mid-stack, frame 50 at the back.
    for index in [0usize, 24, 49] {
        let metric = stackaroni_core::focus::WindowedLaplacian::new(
            4,
            &scratch,
            std::collections::HashMap::new(),
        );
        let map = metric
            .evaluate(&Image::open(&stack.frames[index]).unwrap())
            .unwrap();
        stackaroni_core::debug::write_plane(&out.join(format!("focus_{:03}.png", index + 1)), &map)
            .unwrap();
    }
    println!("wrote focus heatmaps to {}", out.display());
}

/// Cost of one focus map on a full 50 MP frame — the number that decides whether
/// T9 can afford this per frame across a 100-frame stack.
#[test]
#[ignore = "requires test-data/, run with --release"]
fn focus_metric_cost_on_a_real_frame() {
    let Some(frames) = frames() else { return };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/scratch");
    std::fs::create_dir_all(&out).unwrap();

    let image = Image::open(&frames[frames.len() / 2]).unwrap();
    let info = image.info();
    let metric =
        stackaroni_core::focus::WindowedLaplacian::new(4, &out, std::collections::HashMap::new());

    let start = Instant::now();
    let map = metric.evaluate(&image).unwrap();
    let secs = start.elapsed().as_secs_f32();
    let mb = info.width as f64 * info.height as f64 * 4.0 / 1e6;

    println!(
        "{}x{}: {secs:.1}s, plane {mb:.0} MB => 100 frames = {:.0} min, {:.0} GB scratch",
        info.width,
        info.height,
        secs * 100.0 / 60.0,
        mb * 100.0 / 1000.0
    );
    drop(map);
}
