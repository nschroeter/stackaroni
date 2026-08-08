//! Does the guided filter's guide belong in linear light or gamma-encoded space?
//!
//! Run with:
//! ```text
//! cargo test --release -p stackaroni-core --test weight_guide_comparison -- --ignored --nocapture
//! ```
//!
//! `synthetic_50` ships a ground-truth all-in-focus render, so this can be scored
//! objectively rather than by impression. The scoring is deliberately split: an
//! overall error, and an error restricted to the highest-gradient 2% of the ground
//! truth — the thin antenna and leg structures the quality checklist prioritizes. A
//! guide can plausibly clean up background mottling while doing *worse* on those
//! edges, and a single averaged number would hide exactly that trade.
//!
//! The blend here is a plain weighted average, purely as a diagnostic. It is not the
//! `ImageFusion` implementation; Laplacian-pyramid fusion is T8.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use stackaroni_core::debug;
use stackaroni_core::discovery::discover_stack;
use stackaroni_core::focus::WindowedLaplacian;
use stackaroni_core::grid::Grid;
use stackaroni_core::image::ScratchPlane;
use stackaroni_core::pipeline::{FocusMap, FocusMetric, Image, Transform, WeightEstimator};
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

const FOCUS_RADIUS: u32 = 4;
const GUIDE_RADIUS: u32 = 8;
const EPSILONS: [f32; 5] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

fn stack_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/synthetic_50")
}

fn out_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/t7")
}

/// Weighted average of the frames' luma under `weights`.
fn blend_luma(frames: &[PathBuf], weights: &[ScratchPlane], width: u32, height: u32) -> Grid {
    let mut out = Grid::new(width, height);
    for (path, weight) in frames.iter().zip(weights) {
        let luma = Grid::from_image(&Image::open(path).unwrap(), 0).unwrap();
        let w = weight.rows(0, height).unwrap();
        for (i, slot) in out.data.iter_mut().enumerate() {
            *slot += luma.data[i] * w[i];
        }
    }
    out
}

/// RMSE overall, and over the sharpest `fraction` of ground-truth gradient.
fn scores(truth: &Grid, got: &Grid, fraction: f32) -> (f32, f32) {
    let (w, h) = (truth.width as i64, truth.height as i64);
    let mut gradient = Vec::with_capacity(truth.data.len());
    for y in 0..h {
        for x in 0..w {
            let at =
                |xx: i64, yy: i64| truth.at(xx.clamp(0, w - 1) as u32, yy.clamp(0, h - 1) as u32);
            let gx = at(x + 1, y) - at(x - 1, y);
            let gy = at(x, y + 1) - at(x, y - 1);
            gradient.push((gx * gx + gy * gy).sqrt());
        }
    }

    let mut sorted = gradient.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let threshold = sorted[((1.0 - fraction) * sorted.len() as f32) as usize];

    let (mut all, mut edge, mut edge_n) = (0.0f64, 0.0f64, 0usize);
    for (i, &g) in gradient.iter().enumerate() {
        let e = (truth.data[i] - got.data[i]) as f64;
        all += e * e;
        if g >= threshold {
            edge += e * e;
            edge_n += 1;
        }
    }
    (
        (all / truth.data.len() as f64).sqrt() as f32,
        (edge / edge_n.max(1) as f64).sqrt() as f32,
    )
}

/// Unfiltered one-hot selection, as the no-filter control.
fn one_hot_weights(labels: &ScratchPlane, count: usize, scratch: &Path) -> Vec<ScratchPlane> {
    let (w, h) = (labels.width(), labels.height());
    (0..count)
        .map(|k| {
            let mut plane =
                ScratchPlane::create(&scratch.join(format!("onehot{k}.f32")), w, h).unwrap();
            let src: Vec<f32> = labels
                .rows(0, h)
                .unwrap()
                .iter()
                .map(|&l| if l as usize == k { 1.0 } else { 0.0 })
                .collect();
            plane.rows_mut(0, h).unwrap().copy_from_slice(&src);
            plane
        })
        .collect()
}

#[test]
#[ignore = "requires test-data/, run with --release"]
fn guide_space_comparison() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let out = out_dir();
    std::fs::create_dir_all(&out).unwrap();

    let truth = Grid::from_image(
        &Image::open(&stack_dir().join("ground_truth_all_in_focus.tiff")).unwrap(),
        0,
    )
    .unwrap();
    let (width, height) = (truth.width, truth.height);

    // Focus maps once; both guide spaces score against the same input.
    let focus_dir = out.join("focus");
    std::fs::create_dir_all(&focus_dir).unwrap();
    let metric = WindowedLaplacian::new(FOCUS_RADIUS, &focus_dir, HashMap::new());
    let focus_maps: Vec<FocusMap> = stack
        .frames
        .iter()
        .map(|p| metric.evaluate(&Image::open(p).unwrap()).unwrap())
        .collect();

    let transforms = vec![Transform::IDENTITY; stack.frames.len()];

    // Control: raw argmax with no filtering at all. If the guided filter is not
    // moving these numbers, the guide-space question is moot — so establish the
    // no-filter baseline before comparing anything against it.
    let scratch = out.join("labels");
    std::fs::create_dir_all(&scratch).unwrap();
    let estimator = GuidedWeights::new(
        stack.frames.clone(),
        transforms.clone(),
        GUIDE_RADIUS,
        EPSILONS[0],
        GuideSpace::Perceptual,
        &scratch,
    );
    let labels = estimator.labels(&focus_maps).unwrap();
    debug::write_plane(&out.join("labels_argmax.png"), &labels).unwrap();

    let onehot = one_hot_weights(&labels, stack.frames.len(), &scratch);
    let raw = blend_luma(&stack.frames, &onehot, width, height);
    let (raw_all, raw_edges) = scores(&truth, &raw, 0.02);
    debug::write_grid(&out.join("blend_argmax.png"), &raw).unwrap();

    println!("\n=== overall vs thin-structure error ===");
    println!(
        "{:>12}  {:>9}  {:>12}  {:>18}",
        "guide", "epsilon", "RMSE all", "RMSE top-2% edges"
    );
    println!(
        "{:>12}  {:>9}  {raw_all:>12.5}  {raw_edges:>18.5}",
        "none", "-"
    );

    // Epsilon is not scale-invariant: sRGB values are numerically larger than
    // linear ones, so the same epsilon smooths less in perceptual space. Sweeping
    // it is what separates the guide-space question from smoothing strength.
    for (name, space) in [
        ("linear", GuideSpace::Linear),
        ("perceptual", GuideSpace::Perceptual),
    ] {
        for epsilon in EPSILONS {
            let scratch = out.join(format!("{name}_{epsilon:e}"));
            std::fs::create_dir_all(&scratch).unwrap();

            let estimator = GuidedWeights::new(
                stack.frames.clone(),
                transforms.clone(),
                GUIDE_RADIUS,
                epsilon,
                space,
                &scratch,
            );
            let weights = estimator.weights(&focus_maps).unwrap();
            let blended = blend_luma(&stack.frames, &weights, width, height);
            let (all, edges) = scores(&truth, &blended, 0.02);
            println!("{name:>12}  {epsilon:>9.0e}  {all:>12.5}  {edges:>18.5}");

            debug::write_grid(&out.join(format!("blend_{name}_{epsilon:e}.png")), &blended)
                .unwrap();
            let _ = std::fs::remove_dir_all(&scratch);
        }
    }

    println!("\nwrote debug output to {}\n", out.display());
}
