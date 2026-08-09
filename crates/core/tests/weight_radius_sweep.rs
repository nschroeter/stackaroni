//! Does a guided-filter radius exist that satisfies both failure modes at once?
//!
//! ```text
//! cargo test --release -p stackaroni-core --test weight_radius_sweep -- --ignored --nocapture
//! ```
//!
//! The radius is doing two opposing jobs. Large enough, it suppresses the
//! salt-and-pepper argmax mottling across defocused background (the T7 problem).
//! Small enough, it resolves a 1-3 px antenna instead of averaging frames across it
//! (the T8/`51d6833` problem). Shrinking it to fix sharpness can reintroduce
//! mottling, so both are measured on every configuration.
//!
//! Ground-truth RMSE cannot arbitrate this — it has already pointed the wrong way
//! twice, in opposite directions. So the two failure modes get purpose-built
//! measures instead, both scored against `ground_truth_all_in_focus.tiff` and both
//! reading 1.00 when the result matches the truth:
//!
//! - **detail**: Laplacian energy over the sharpest 2% of ground-truth gradient
//!   (thin structures). Below 1.0 means detail lost to over-averaging.
//! - **bokeh**: Laplacian energy over the flattest 50% (smooth background).
//!   Above 1.0 means mottling — texture present that the truth does not have.
//!
//! **"No radius satisfies both" is a real possible outcome, not a failed
//! experiment.** It is the evidence that would justify escalating to graph-cut /
//! MRF weight refinement, which `docs/algorithms.md` §8 names for exactly this.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use stackaroni_core::discovery::discover_stack;
use stackaroni_core::focus::WindowedLaplacian;
use stackaroni_core::fusion::LaplacianPyramidFusion;
use stackaroni_core::grid::Grid;
use stackaroni_core::pipeline::{
    FocusMap, FocusMetric, Image, ImageFusion, Transform, WeightEstimator,
};
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

const FOCUS_RADIUS: u32 = 4;
const PYRAMID_FLOOR: u32 = 32;
const RADII: [u32; 5] = [1, 2, 4, 8, 16];
const EPSILONS: [f32; 3] = [1e-4, 1e-3, 1e-2];

fn stack_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/synthetic_50")
}

/// Per-pixel squared Laplacian, edges clamped.
fn laplacian_energy(g: &Grid) -> Vec<f32> {
    let (w, h) = (g.width as i64, g.height as i64);
    let at = |x: i64, y: i64| g.at(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32);
    let mut out = vec![0f32; g.data.len()];
    for y in 0..h {
        for x in 0..w {
            let l = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            out[(y * w + x) as usize] = l * l;
        }
    }
    out
}

/// Mean energy over the pixels a mask selects.
fn masked_mean(energy: &[f32], mask: &[bool]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for (e, &m) in energy.iter().zip(mask) {
        if m {
            sum += *e as f64;
            n += 1;
        }
    }
    sum / n.max(1) as f64
}

/// Masks of the sharpest `top` fraction and flattest `bottom` fraction of the
/// ground truth's gradient.
fn masks(truth: &Grid, top: f32, bottom: f32) -> (Vec<bool>, Vec<bool>) {
    let (w, h) = (truth.width as i64, truth.height as i64);
    let at = |x: i64, y: i64| truth.at(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32);
    let mut gradient = Vec::with_capacity(truth.data.len());
    for y in 0..h {
        for x in 0..w {
            let gx = at(x + 1, y) - at(x - 1, y);
            let gy = at(x, y + 1) - at(x, y - 1);
            gradient.push((gx * gx + gy * gy).sqrt());
        }
    }
    let mut sorted = gradient.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hi = sorted[((1.0 - top) * sorted.len() as f32) as usize];
    let lo = sorted[(bottom * sorted.len() as f32) as usize];
    (
        gradient.iter().map(|&g| g >= hi).collect(),
        gradient.iter().map(|&g| g <= lo).collect(),
    )
}

#[test]
#[ignore = "requires test-data/, run with --release"]
fn guide_radius_sweep() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/sweep");
    std::fs::create_dir_all(&out).unwrap();

    let truth = Grid::from_image(
        &Image::open(&stack_dir().join("ground_truth_all_in_focus.tiff")).unwrap(),
        0,
    )
    .unwrap();
    let (structure, smooth) = masks(&truth, 0.02, 0.50);
    let truth_energy = laplacian_energy(&truth);
    let truth_detail = masked_mean(&truth_energy, &structure);
    let truth_bokeh = masked_mean(&truth_energy, &smooth);

    // Identity transforms isolate the weight stage; registration is exercised
    // elsewhere and would otherwise confound the comparison against an unwarped
    // ground truth.
    let transforms = vec![Transform::IDENTITY; stack.frames.len()];
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let focus_dir = out.join("focus");
    std::fs::create_dir_all(&focus_dir).unwrap();
    let metric = WindowedLaplacian::new(FOCUS_RADIUS, &focus_dir, by_path.clone());
    let focus_maps: Vec<FocusMap> = stack
        .frames
        .iter()
        .map(|p| metric.evaluate(&Image::open(p).unwrap(), &()).unwrap())
        .collect();

    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();

    println!("\n=== guide radius sweep (1.00 = matches ground truth) ===");
    println!("detail <1 means over-averaged; bokeh >1 means mottling\n");
    println!(
        "{:>7}  {:>8}  {:>8}  {:>8}",
        "radius", "epsilon", "detail", "bokeh"
    );

    for radius in RADII {
        for epsilon in EPSILONS {
            let scratch = out.join(format!("r{radius}_e{epsilon:e}"));
            std::fs::create_dir_all(&scratch).unwrap();

            let estimator = GuidedWeights::new(
                stack.frames.clone(),
                transforms.clone(),
                radius,
                epsilon,
                GuideSpace::Perceptual,
                &scratch,
            );
            let weights = estimator.weights(&focus_maps, &()).unwrap();

            let path = scratch.join("fused.tif");
            let fusion = LaplacianPyramidFusion::new(&path, by_path.clone(), PYRAMID_FLOOR);
            let fused = fusion.fuse(&images, &weights, &()).unwrap();
            let got = Grid::from_image(&fused, 0).unwrap();

            let energy = laplacian_energy(&got);
            let detail = masked_mean(&energy, &structure) / truth_detail;
            let bokeh = masked_mean(&energy, &smooth) / truth_bokeh;
            println!("{radius:>7}  {epsilon:>8.0e}  {detail:>8.3}  {bokeh:>8.3}");

            drop(weights);
            drop(fused);
            let _ = std::fs::remove_dir_all(&scratch);
        }
    }
    let _ = std::fs::remove_dir_all(&focus_dir);
    println!();
}

/// Ceiling for the `detail` metric: how much of the ground truth's thin-structure
/// energy any single source frame actually carries.
///
/// The ground truth is a synthetic all-in-focus render, so it may simply be sharper
/// than any real frame in the stack — in which case `detail` can never reach 1.00 and
/// a number below 1 is not evidence of a defect. Without this the sweep is
/// uninterpretable.
#[test]
#[ignore = "requires test-data/, run with --release"]
fn detail_ceiling_of_the_source_frames() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        return;
    };
    let truth = Grid::from_image(
        &Image::open(&stack_dir().join("ground_truth_all_in_focus.tiff")).unwrap(),
        0,
    )
    .unwrap();
    let (structure, smooth) = masks(&truth, 0.02, 0.50);
    let truth_energy = laplacian_energy(&truth);
    let truth_detail = masked_mean(&truth_energy, &structure);
    let truth_bokeh = masked_mean(&truth_energy, &smooth);

    let mut best_frame = 0.0f64;
    let mut per_pixel_max = vec![0f32; truth.data.len()];
    for path in &stack.frames {
        let g = Grid::from_image(&Image::open(path).unwrap(), 0).unwrap();
        let e = laplacian_energy(&g);
        best_frame = best_frame.max(masked_mean(&e, &structure) / truth_detail);
        for (m, v) in per_pixel_max.iter_mut().zip(&e) {
            *m = m.max(*v);
        }
    }

    println!("\n=== detail metric ceiling ===");
    println!("best single frame:        {best_frame:.3}");
    println!(
        "per-pixel oracle (upper): {:.3}",
        masked_mean(&per_pixel_max, &structure) / truth_detail
    );
    println!(
        "ground-truth bokeh energy is the 1.00 reference ({:.3e})\n",
        truth_bokeh
    );
}

/// Render the two candidate configurations for a visual call on checklist item 3.
///
/// The numbers cannot settle r4 versus r8: the difference is whether 30% excess
/// bokeh energy reads as patchy over-sharpening to a human. Both are written at full
/// resolution for side-by-side inspection.
#[test]
#[ignore = "requires test-data/, run with --release"]
fn render_candidate_configs() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        return;
    };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/candidates");
    std::fs::create_dir_all(&out).unwrap();

    let transforms = vec![Transform::IDENTITY; stack.frames.len()];
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let focus_dir = out.join("focus");
    std::fs::create_dir_all(&focus_dir).unwrap();
    let metric = WindowedLaplacian::new(FOCUS_RADIUS, &focus_dir, by_path.clone());
    let focus_maps: Vec<FocusMap> = stack
        .frames
        .iter()
        .map(|p| metric.evaluate(&Image::open(p).unwrap(), &()).unwrap())
        .collect();
    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();

    for (radius, epsilon) in [(4u32, 1e-4f32), (8, 1e-4)] {
        let scratch = out.join(format!("s{radius}"));
        std::fs::create_dir_all(&scratch).unwrap();
        let estimator = GuidedWeights::new(
            stack.frames.clone(),
            transforms.clone(),
            radius,
            epsilon,
            GuideSpace::Perceptual,
            &scratch,
        );
        let weights = estimator.weights(&focus_maps, &()).unwrap();
        let path = out.join(format!("r{radius}.tif"));
        LaplacianPyramidFusion::new(&path, by_path.clone(), PYRAMID_FLOOR)
            .fuse(&images, &weights, &())
            .unwrap();
        drop(weights);
        let _ = std::fs::remove_dir_all(&scratch);
        println!("wrote {}", path.display());
    }
    let _ = std::fs::remove_dir_all(&focus_dir);
}
