//! Does the fused image exceed what any weighting of the *aligned* sources could produce?
//!
//! ```text
//! cargo test --release -p stackaroni-core --test halo_warped_envelope -- --ignored --nocapture
//! ```
//!
//! A same-x comparison against unwarped frames put the fused above the source
//! maximum at 45/300 positions near a subject/background edge — the signature of
//! multi-scale bleed. But that test is confounded: the fused is in anchor
//! coordinates while the sources are not, and with scale spanning 0.9495..1.0569 a
//! point 115 px from centre displaces up to ~7 px between frames, which against a
//! steep edge could manufacture the same signature on its own.
//!
//! This removes the confound by warping every source frame by its own `Transform`
//! before building the per-position envelope, so fused and sources are compared in
//! the same coordinate system. If the fused still exceeds the max, fusion is
//! injecting energy; if not, the earlier signal was displacement.

use std::path::{Path, PathBuf};

use stackaroni_core::discovery::discover_stack;
use stackaroni_core::grid::Grid;
use stackaroni_core::pipeline::Image;
use stackaroni_core::registration::{PhaseCorrelation, register_stack};

/// Same strip as the confounded profile, in anchor coordinates.
const X0: u32 = 4150;
const WIDTH: u32 = 300;
const Y0: u32 = 3196;
const ROWS: u32 = 9;

#[test]
#[ignore = "requires test-data/, run with --release"]
fn fused_stays_within_the_warped_source_envelope() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/blossom");
    let fused_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/t10/blossom.tif");
    let Ok(stack) = discover_stack(&dir) else {
        eprintln!("skipping: test-data/blossom not present");
        return;
    };
    if !fused_path.is_file() {
        eprintln!("skipping: {} not present", fused_path.display());
        return;
    }

    let registration = PhaseCorrelation::new(3);
    let transforms = register_stack(&registration, &stack.frames, |d, t| {
        if d % 25 == 0 {
            println!("  register {d}/{t}");
        }
    })
    .unwrap();

    let fused = Grid::from_image(&Image::open(&fused_path).unwrap(), 0).unwrap();
    let (cx, cy) = (fused.width as f32 / 2.0, fused.height as f32 / 2.0);

    // Vertically averaged fused profile, matching the earlier measurement.
    let profile = |g: &Grid, sample: &dyn Fn(&Grid, f32, f32) -> f32| -> Vec<f32> {
        (0..WIDTH)
            .map(|i| {
                (0..ROWS)
                    .map(|r| sample(g, (X0 + i) as f32, (Y0 + r) as f32))
                    .sum::<f32>()
                    / ROWS as f32
            })
            .collect()
    };
    let direct = |g: &Grid, x: f32, y: f32| g.sample(x, y);
    let fused_profile = profile(&fused, &direct);

    let mut lo = vec![f32::MAX; WIDTH as usize];
    let mut hi = vec![f32::MIN; WIDTH as usize];

    for (path, transform) in stack.frames.iter().zip(&transforms) {
        let g = Grid::from_image(&Image::open(path).unwrap(), 0).unwrap();
        for i in 0..WIDTH as usize {
            let mut acc = 0.0;
            for r in 0..ROWS {
                // Anchor coordinates through this frame's transform, exactly as the
                // pipeline resolves them.
                let (sx, sy) = transform.apply((X0 + i as u32) as f32 - cx, (Y0 + r) as f32 - cy);
                acc += g.sample(sx + cx, sy + cy);
            }
            let v = acc / ROWS as f32;
            lo[i] = lo[i].min(v);
            hi[i] = hi[i].max(v);
        }
    }

    let above: Vec<usize> = (0..WIDTH as usize)
        .filter(|&i| fused_profile[i] > hi[i])
        .collect();
    let below = (0..WIDTH as usize)
        .filter(|&i| fused_profile[i] < lo[i])
        .count();

    println!(
        "\n=== fused vs WARPED source envelope, {} frames ===",
        stack.frames.len()
    );
    println!("above source max: {}/{WIDTH}", above.len());
    println!("below source min: {below}/{WIDTH}");

    if let Some(&worst) = above.iter().max_by(|&&a, &&b| {
        (fused_profile[a] - hi[a])
            .partial_cmp(&(fused_profile[b] - hi[b]))
            .unwrap()
    }) {
        let excess = fused_profile[worst] - hi[worst];
        println!(
            "largest overshoot: x={}  fused {:.5}  max {:.5}  excess {:.5} ({:.1}% of max)",
            X0 as usize + worst,
            fused_profile[worst],
            hi[worst],
            excess,
            100.0 * excess / hi[worst]
        );
    }

    println!(
        "\n{:>6}  {:>9}  {:>9}  {:>9}  {}",
        "x", "fused", "src min", "src max", "verdict"
    );
    for i in (0..WIDTH as usize).step_by(12) {
        let v = if fused_profile[i] > hi[i] {
            "ABOVE"
        } else if fused_profile[i] < lo[i] {
            "below"
        } else {
            "in range"
        };
        println!(
            "{:>6}  {:>9.5}  {:>9.5}  {:>9.5}  {v}",
            X0 as usize + i,
            fused_profile[i],
            lo[i],
            hi[i]
        );
    }
    println!();
}
