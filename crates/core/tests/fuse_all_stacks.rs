//! Full pipeline over every stack in `test-data/`, writing each result next to the
//! frames it came from for review.
//!
//! ```text
//! cargo test --release -p stackaroni-core --test fuse_all_stacks -- --ignored --nocapture
//! ```
//!
//! Output lands at `test-data/<stack>/stackaroni_fused.tif`. That stem is excluded
//! from frame discovery, so a result sitting beside its own inputs is not picked up
//! as an extra frame on the next run.
//!
//! Scratch is ~40 GB per stack (focus maps plus weight planes at 50 MP), so it is
//! removed between stacks rather than accumulated.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use stackaroni_core::discovery::discover_test_set;
use stackaroni_core::focus::WindowedLaplacian;
use stackaroni_core::fusion::LaplacianPyramidFusion;
use stackaroni_core::pipeline::{
    FocusMap, FocusMetric, Image, ImageFusion, Registration, Transform, WeightEstimator,
};
use stackaroni_core::registration::{PhaseCorrelation, register_stack};
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

const REGISTRATION_LEVEL: u32 = 3;
const FOCUS_RADIUS: u32 = 4;
const GUIDE_RADIUS: u32 = 8;
const GUIDE_EPSILON: f32 = 1e-2;
const PYRAMID_FLOOR: u32 = 32;

#[test]
#[ignore = "requires test-data/, run with --release"]
fn fuse_every_stack() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data");
    let Ok(stacks) = discover_test_set(&root) else {
        eprintln!("skipping: test-data/ not present");
        return;
    };

    for stack in &stacks {
        let started = Instant::now();
        let info = stack.probe().unwrap().info;
        println!(
            "\n=== {} : {} frames, {}x{} ===",
            stack.name,
            stack.frames.len(),
            info.width,
            info.height
        );

        let scratch = stack.dir.join(".stackaroni-scratch");
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();

        // Registration: similarity, chained outward from the middle anchor.
        let t0 = Instant::now();
        let registration: Box<dyn Registration> =
            Box::new(PhaseCorrelation::new(REGISTRATION_LEVEL));
        let transforms = register_stack(registration.as_ref(), &stack.frames, |done, total| {
            if done % 20 == 0 || done == total {
                println!("  register {done}/{total}");
            }
        })
        .unwrap();
        let span = transforms.iter().fold((f32::MAX, f32::MIN), |(lo, hi), t| {
            (lo.min(t.scale), hi.max(t.scale))
        });
        println!(
            "  registered in {:.0}s, scale range {:.4}..{:.4}",
            t0.elapsed().as_secs_f32(),
            span.0,
            span.1
        );

        let by_path: HashMap<PathBuf, Transform> = stack
            .frames
            .iter()
            .cloned()
            .zip(transforms.iter().copied())
            .collect();

        let t0 = Instant::now();
        let metric = WindowedLaplacian::new(FOCUS_RADIUS, &scratch, by_path.clone());
        let focus_maps: Vec<FocusMap> = stack
            .frames
            .iter()
            .map(|p| metric.evaluate(&Image::open(p).unwrap()).unwrap())
            .collect();
        println!("  focus maps in {:.0}s", t0.elapsed().as_secs_f32());

        let t0 = Instant::now();
        let estimator = GuidedWeights::new(
            stack.frames.clone(),
            transforms,
            GUIDE_RADIUS,
            GUIDE_EPSILON,
            GuideSpace::Perceptual,
            &scratch,
        );
        let weights = estimator.weights(&focus_maps).unwrap();
        println!("  weights in {:.0}s", t0.elapsed().as_secs_f32());

        let t0 = Instant::now();
        let images: Vec<Image> = stack
            .frames
            .iter()
            .map(|p| Image::open(p).unwrap())
            .collect();
        let output = stack.dir.join("stackaroni_fused.tif");
        let fusion = LaplacianPyramidFusion::new(&output, by_path, PYRAMID_FLOOR);
        fusion.fuse(&images, &weights).unwrap();
        println!("  fused in {:.0}s", t0.elapsed().as_secs_f32());

        drop(weights);
        drop(focus_maps);
        let _ = std::fs::remove_dir_all(&scratch);

        println!(
            "  -> {} ({:.0}s total)",
            output.display(),
            started.elapsed().as_secs_f32()
        );
    }
    println!();
}
