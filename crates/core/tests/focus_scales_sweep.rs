//! How many scales does the multi-scale focus measure actually want?
//!
//! ```text
//! cargo test --release -p stackaroni-core --test focus_scales_sweep -- --ignored --nocapture
//! ```
//!
//! Same instrument as the guide-radius sweep that settled `guide_radius`, and the same
//! reason for reusing it rather than inventing a metric: this project has twice had a
//! plausible new measurement point the wrong way, and the detail/bokeh pair is the one
//! that has held up against Niels's ratings.
//!
//! - **detail**: Laplacian energy over the sharpest 2% of ground-truth gradient.
//!   Below 1.0 means thin structure lost.
//! - **bokeh**: the same over the flattest 50%. Above 1.0 means texture the truth
//!   does not have — the mottling failure.
//!
//! `scales = 1` is the shipped single-scale metric exactly (see
//! `focus::tests::multi_scale_reduces_to_single_scale`), so the first row is the control
//! and every later row is measured against it on identical inputs.
//!
//! **What would justify a default above 1** is detail climbing while bokeh stays flat.
//! Both climbing together means the extra scales are adding background energy along with
//! signal, which is the trigger §4 records for trying a max rule instead of the sum.
//! Neither moving means scale count does not matter on this stack and the default should
//! stay at 1 — a real outcome, not a failed experiment.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use stackaroni_core::defaults;
use stackaroni_core::discovery::discover_stack;
use stackaroni_core::focus::MultiScaleLaplacian;
use stackaroni_core::fusion::SelectionFusion;
use stackaroni_core::grid::Grid;
use stackaroni_core::pipeline::{
    FocusMap, FocusMetric, Image, ImageFusion, Transform, WeightEstimator,
};
use stackaroni_core::weights::GuidedWeights;

/// Fixed while scales vary, so one thing moves at a time. Halving each octave is the
/// conventional pyramid weighting; whether it is the right value is a separate question
/// from how many levels to keep.
const DECAY: f32 = 0.5;
const SCALES: [u32; 5] = [1, 2, 3, 4, 5];

fn stack_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/synthetic_50")
}

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

/// Is a flat scales sweep really "scales do not matter", or just "decay 0.5 suppressed
/// them"? At `decay = 1.0` every level counts as much as level 0, which is the strongest
/// influence the coarse levels can have under a sum rule. If the numbers are still flat
/// there, the flatness is a property of the measure on this stack rather than of the
/// weighting.
#[test]
#[ignore = "requires test-data/synthetic_50, run with --release"]
fn focus_decay_sweep() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/decay");
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

    let transforms = vec![Transform::IDENTITY; stack.frames.len()];
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();
    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();

    println!("\n=== focus decay sweep, scales 3 ===");
    println!("{:>7}  {:>8}  {:>8}", "decay", "detail", "bokeh");

    for decay in [0.25f32, 0.5, 1.0, 2.0] {
        let scratch = out.join(format!("d{decay}"));
        std::fs::create_dir_all(&scratch).unwrap();

        let metric =
            MultiScaleLaplacian::new(defaults::FOCUS_RADIUS, 3, decay, &scratch, by_path.clone());
        let focus_maps: Vec<FocusMap> = stack
            .frames
            .iter()
            .map(|p| metric.evaluate(&Image::open(p).unwrap(), &()).unwrap())
            .collect();

        let weights = GuidedWeights::new(
            stack.frames.clone(),
            transforms.clone(),
            defaults::GUIDE_RADIUS,
            defaults::GUIDE_EPSILON,
            defaults::GUIDE_SPACE,
            &scratch,
        )
        .weights(&focus_maps, &())
        .unwrap();

        let path = scratch.join("fused.tif");
        let fused = SelectionFusion::new(
            &path,
            by_path.clone(),
            defaults::PYRAMID_FLOOR,
            defaults::SALIENCE_RADIUS,
        )
        .fuse(&images, &weights, &())
        .unwrap();

        let energy = laplacian_energy(&Grid::from_image(&fused, 0).unwrap());
        let detail = masked_mean(&energy, &structure) / truth_detail;
        let bokeh = masked_mean(&energy, &smooth) / truth_bokeh;
        println!("{decay:>7}  {detail:>8.3}  {bokeh:>8.3}");

        drop(focus_maps);
        drop(weights);
        drop(fused);
        let _ = std::fs::remove_dir_all(&scratch);
    }
    println!();
}

#[test]
#[ignore = "requires test-data/synthetic_50, run with --release"]
fn focus_scales_sweep() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/scales");
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

    // Identity transforms, matching the guide-radius sweep: registration would otherwise
    // put the fused image in the anchor frame's coordinates while the masks come from the
    // unwarped truth, which depresses `detail` for every row and confounds the comparison.
    let transforms = vec![Transform::IDENTITY; stack.frames.len()];
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();

    println!("\n=== focus scales sweep, decay {DECAY} (1.00 = matches ground truth) ===");
    println!(
        "radius {}, all other params shipped defaults",
        defaults::FOCUS_RADIUS
    );
    println!("scales 1 is the shipped single-scale metric exactly\n");
    println!("{:>7}  {:>8}  {:>8}", "scales", "detail", "bokeh");

    for scales in SCALES {
        let scratch = out.join(format!("s{scales}"));
        std::fs::create_dir_all(&scratch).unwrap();

        let metric = MultiScaleLaplacian::new(
            defaults::FOCUS_RADIUS,
            scales,
            DECAY,
            &scratch,
            by_path.clone(),
        );
        let focus_maps: Vec<FocusMap> = stack
            .frames
            .iter()
            .map(|p| metric.evaluate(&Image::open(p).unwrap(), &()).unwrap())
            .collect();

        let weights = GuidedWeights::new(
            stack.frames.clone(),
            transforms.clone(),
            defaults::GUIDE_RADIUS,
            defaults::GUIDE_EPSILON,
            defaults::GUIDE_SPACE,
            &scratch,
        )
        .weights(&focus_maps, &())
        .unwrap();

        let path = scratch.join("fused.tif");
        let fused = SelectionFusion::new(
            &path,
            by_path.clone(),
            defaults::PYRAMID_FLOOR,
            defaults::SALIENCE_RADIUS,
        )
        .fuse(&images, &weights, &())
        .unwrap();

        let energy = laplacian_energy(&Grid::from_image(&fused, 0).unwrap());
        let detail = masked_mean(&energy, &structure) / truth_detail;
        let bokeh = masked_mean(&energy, &smooth) / truth_bokeh;
        println!("{scales:>7}  {detail:>8.3}  {bokeh:>8.3}");

        drop(focus_maps);
        drop(weights);
        drop(fused);
        let _ = std::fs::remove_dir_all(&scratch);
    }
    println!();
}

/// Why is the sweep flat? Print what each level actually contributes.
///
/// A sum over scales is only meaningful if the terms are commensurable. The discrete
/// Laplacian is not scale-invariant — its response depends on the grid spacing — so
/// `F_k` computed on a half-size level is not on the same footing as `F_0`, and scale-space
/// theory says derivatives need normalizing by the scale before they can be compared or
/// combined (Lindeberg, *IJCV* 30(2), 1998). This prints the ratio so the question is
/// settled by a number rather than by argument.
#[test]
#[ignore = "requires test-data/synthetic_50, run with --release"]
fn per_level_contribution() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    let frame = &stack.frames[stack.frames.len() / 2];

    println!("\n=== mean focus energy vs scale count, one mid-stack frame ===");
    let mut base = 0.0f64;
    for scales in [1u32, 2, 3, 4, 5] {
        let dir = scratch.path().join(format!("s{scales}"));
        std::fs::create_dir_all(&dir).unwrap();
        let map =
            MultiScaleLaplacian::new(defaults::FOCUS_RADIUS, scales, 1.0, &dir, HashMap::new())
                .evaluate(&Image::open(frame).unwrap(), &())
                .unwrap();
        let info = Image::open(frame).unwrap().info();
        let rows = map.rows(0, info.height).unwrap();
        let mean = rows.iter().map(|&v| v as f64).sum::<f64>() / rows.len() as f64;
        if scales == 1 {
            base = mean;
        }
        println!(
            "scales {scales}: mean {mean:.6e}   {:.4}x level 0",
            mean / base
        );
    }
    println!();
}

/// Does coarse-scale energy actually distinguish a focused frame from a defocused one?
///
/// The contribution test shows coarse levels dominating the magnitude (48x level 0 at five
/// scales) while the sweep shows the fused output unmoved. Both can only be true if what
/// the coarse levels add is nearly the *same in every frame* — and a term common to all
/// frames cancels in the per-pixel argmax that the weight stage runs.
///
/// The mechanism would be defocus itself: blur is a low-pass, so it removes fine detail and
/// leaves coarse structure intact. Downsampling then makes it worse, because a blur of
/// sigma pixels at level 0 is sigma/2^k pixels at level k — coarser levels see a *less*
/// blurred image relative to their own grid, so they discriminate less.
///
/// Measured as between-frame contrast: mean|A-B| / mean((A+B)/2) for a sharp frame and a
/// defocused one. Falling with scale count means the discriminating signal is being diluted
/// by a shared pedestal.
#[test]
#[ignore = "requires test-data/synthetic_50, run with --release"]
fn coarse_levels_do_not_discriminate() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let scratch = tempfile::tempdir().unwrap();
    // Ends of the stack: whatever is sharp in one is thoroughly defocused in the other.
    let (a, b) = (&stack.frames[0], &stack.frames[stack.frames.len() - 1]);
    let info = Image::open(a).unwrap().info();

    println!("\n=== between-frame contrast vs scale count ===");
    println!("first and last frame; higher means the measure separates them better\n");
    for scales in [1u32, 2, 3, 4, 5] {
        let mut mean_diff = 0.0f64;
        let mut mean_avg = 0.0f64;
        let mut maps = Vec::new();
        for (i, frame) in [a, b].iter().enumerate() {
            let dir = scratch.path().join(format!("s{scales}_{i}"));
            std::fs::create_dir_all(&dir).unwrap();
            maps.push(
                MultiScaleLaplacian::new(defaults::FOCUS_RADIUS, scales, 1.0, &dir, HashMap::new())
                    .evaluate(&Image::open(frame).unwrap(), &())
                    .unwrap(),
            );
        }
        let (ra, rb) = (
            maps[0].rows(0, info.height).unwrap(),
            maps[1].rows(0, info.height).unwrap(),
        );
        for (&x, &y) in ra.iter().zip(rb) {
            mean_diff += (x - y).abs() as f64;
            mean_avg += ((x + y) / 2.0) as f64;
        }
        println!(
            "scales {scales}: contrast {:.4}",
            mean_diff / mean_avg.max(f64::MIN_POSITIVE)
        );
    }
    println!();
}

/// Does the fused image change at all between one scale and five?
///
/// The sweep reports identical `detail` to three decimals while the focus maps demonstrably
/// differ, and there are two very different explanations: the pipeline absorbs the change,
/// or the sweep is not varying what it thinks it is. This separates them by comparing the
/// output pixels directly instead of a summary statistic.
#[test]
#[ignore = "requires test-data/synthetic_50, run with --release"]
fn does_the_output_move_at_all() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/moves");
    std::fs::create_dir_all(&out).unwrap();

    let transforms = vec![Transform::IDENTITY; stack.frames.len()];
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();
    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();

    let mut results = Vec::new();
    for scales in [1u32, 5] {
        let scratch = out.join(format!("s{scales}"));
        std::fs::create_dir_all(&scratch).unwrap();
        let metric = MultiScaleLaplacian::new(
            defaults::FOCUS_RADIUS,
            scales,
            1.0,
            &scratch,
            by_path.clone(),
        );
        let focus_maps: Vec<FocusMap> = stack
            .frames
            .iter()
            .map(|p| metric.evaluate(&Image::open(p).unwrap(), &()).unwrap())
            .collect();
        let weights = GuidedWeights::new(
            stack.frames.clone(),
            transforms.clone(),
            defaults::GUIDE_RADIUS,
            defaults::GUIDE_EPSILON,
            defaults::GUIDE_SPACE,
            &scratch,
        )
        .weights(&focus_maps, &())
        .unwrap();
        let fused = SelectionFusion::new(
            &scratch.join("fused.tif"),
            by_path.clone(),
            defaults::PYRAMID_FLOOR,
            defaults::SALIENCE_RADIUS,
        )
        .fuse(&images, &weights, &())
        .unwrap();
        results.push(Grid::from_image(&fused, 0).unwrap());
    }

    let (a, b) = (&results[0], &results[1]);
    let mut differing = 0usize;
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for (&x, &y) in a.data.iter().zip(&b.data) {
        let d = (x - y).abs();
        if d > 0.0 {
            differing += 1;
        }
        sum += d as f64;
        max = max.max(d);
    }
    println!("\n=== scales 1 vs scales 5, fused pixels ===");
    println!(
        "differing {:.2}%   mean |diff| {:.3e}   max {:.3e}",
        100.0 * differing as f64 / a.data.len() as f64,
        sum / a.data.len() as f64,
        max
    );
    println!();
}
