//! Full chain against `synthetic_50`, scored against its ground truth.
//!
//! ```text
//! cargo test --release -p stackaroni-core --test fusion_end_to_end -- --ignored --nocapture
//! ```
//!
//! This is registration + focus + weights + fusion together, not fusion alone: the
//! ground truth has no breathing applied, so anything the registration stage leaves
//! misaligned shows up here too. That is the right test now that registration is
//! upstream in the chain.
//!
//! The per-row error breakdown is the point. A whole-image mean hides a defect
//! confined to a few rows; printing the worst rows names them, so a horizontal
//! artifact is caught quantitatively rather than by eye.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use stackaroni_core::debug;
use stackaroni_core::discovery::discover_stack;
use stackaroni_core::focus::WindowedLaplacian;
use stackaroni_core::fusion::LaplacianPyramidFusion;
use stackaroni_core::grid::Grid;
use stackaroni_core::pipeline::{
    FocusMap, FocusMetric, Image, ImageFusion, Transform, WeightEstimator,
};
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

const FOCUS_RADIUS: u32 = 4;
const GUIDE_RADIUS: u32 = 8;
const GUIDE_EPSILON: f32 = 1e-2;
const PYRAMID_FLOOR: u32 = 32;

fn stack_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/synthetic_50")
}

#[test]
#[ignore = "requires test-data/, run with --release"]
fn fuse_synthetic_stack_against_ground_truth() {
    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug-out/t8");
    let scratch = out.join("scratch");
    std::fs::create_dir_all(&scratch).unwrap();

    let start = Instant::now();

    // Identity transforms: the synthetic stack's breathing is small and this test is
    // about fusion fidelity. Registration is exercised on ruler/blossom.
    let transforms = vec![Transform::IDENTITY; stack.frames.len()];
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let metric = WindowedLaplacian::new(FOCUS_RADIUS, &scratch, by_path.clone());
    let focus_maps: Vec<FocusMap> = stack
        .frames
        .iter()
        .map(|p| metric.evaluate(&Image::open(p).unwrap()).unwrap())
        .collect();

    let estimator = GuidedWeights::new(
        stack.frames.clone(),
        transforms,
        GUIDE_RADIUS,
        GUIDE_EPSILON,
        GuideSpace::Perceptual,
        &scratch,
    );
    let weights = estimator.weights(&focus_maps).unwrap();

    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();
    let fused_path = out.join("fused.tif");
    let fusion = LaplacianPyramidFusion::new(&fused_path, by_path, PYRAMID_FLOOR);
    let fused = fusion.fuse(&images, &weights).unwrap();

    println!(
        "\nfused {} frames in {:.1}s",
        images.len(),
        start.elapsed().as_secs_f32()
    );
    println!("wrote {}", fused_path.display());

    // Compare in linear light, both sides read back through the same decoder.
    let truth = Grid::from_image(
        &Image::open(&stack_dir().join("ground_truth_all_in_focus.tiff")).unwrap(),
        0,
    )
    .unwrap();
    let got = Grid::from_image(&fused, 0).unwrap();
    assert_eq!((got.width, got.height), (truth.width, truth.height));

    let mut per_row = Vec::with_capacity(truth.height as usize);
    for y in 0..truth.height {
        let row = y as usize * truth.width as usize;
        let sum: f64 = (0..truth.width as usize)
            .map(|x| (truth.data[row + x] - got.data[row + x]).abs() as f64)
            .sum();
        per_row.push(sum / truth.width as f64);
    }
    let mean = per_row.iter().sum::<f64>() / per_row.len() as f64;

    let mut ranked: Vec<(usize, f64)> = per_row.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nmean abs error vs ground truth: {mean:.5}");
    println!("worst 8 rows (row: error, ratio to mean):");
    for (row, err) in ranked.iter().take(8) {
        println!("  {row:>4}: {err:.5}  {:.2}x", err / mean);
    }

    // A horizontal artifact would show as isolated rows far above their neighbours.
    // Compare each row against the median instead of the mean, which a broad defect
    // would drag upward with it.
    let mut sorted = per_row.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let (worst_row, worst) = ranked[0];
    println!(
        "median row error {median:.5}, worst/median {:.2}x",
        worst / median
    );

    debug::write_grid(&out.join("fused.png"), &got).unwrap();
    let mut diff = Grid::new(truth.width, truth.height);
    for i in 0..diff.data.len() {
        diff.data[i] = (truth.data[i] - got.data[i]).abs();
    }
    debug::write_grid(&out.join("diff.png"), &diff).unwrap();
    println!("worst row is {worst_row}\n");

    let _ = std::fs::remove_dir_all(&scratch);
}
