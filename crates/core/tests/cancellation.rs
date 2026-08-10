//! Does cancellation actually reach *inside* a stage?
//!
//! ```text
//! cargo test -p stackaroni-core --test cancellation
//! ```
//!
//! The whole reason `RunControl` is threaded into the stage methods rather than checked
//! between them is that the stages are long: on a 100-frame stack fusion alone runs ~10
//! minutes, so a caller polling only between stages could not stop anything. A test that
//! merely proved "the parameter compiles" would pass just as happily against an
//! implementation that ignored it entirely — which is exactly the regression worth
//! guarding, since two of the four impls legitimately *do* ignore it.
//!
//! So these run on small synthetic frames and assert on *where* the run stopped: the
//! frame counter recorded by the control has to show it gave up partway, not at the end.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use stackaroni_core::error::Error;
use stackaroni_core::focus::WindowedLaplacian;
use stackaroni_core::fusion::{LaplacianPyramidFusion, SelectionFusion};
use stackaroni_core::image::FrameInfo;
use stackaroni_core::pipeline::{
    FocusMap, FocusMetric, Image, ImageFusion, RunControl, Stage, Transform, WeightEstimator,
};
use stackaroni_core::tiff_io::write_rgb16_srgb;
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

/// Cancels once it has seen `after` progress reports, and records what it saw.
struct CancelAfter {
    after: usize,
    seen: AtomicUsize,
    cancelled: AtomicBool,
    stages: Mutex<Vec<(Stage, usize, usize)>>,
}

impl CancelAfter {
    fn new(after: usize) -> Self {
        Self {
            after,
            seen: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
            stages: Mutex::new(Vec::new()),
        }
    }

    fn reports(&self) -> Vec<(Stage, usize, usize)> {
        self.stages.lock().unwrap().clone()
    }
}

impl RunControl for CancelAfter {
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn progress(&self, stage: Stage, done: usize, total: usize) {
        self.stages.lock().unwrap().push((stage, done, total));
        if self.seen.fetch_add(1, Ordering::SeqCst) + 1 >= self.after {
            self.cancelled.store(true, Ordering::SeqCst);
        }
    }
}

fn write_frame(path: &Path, seed: f32) {
    let info = FrameInfo {
        width: 64,
        height: 48,
        samples: 3,
        bits_per_sample: 16,
    };
    write_rgb16_srgb(path, info, |y, row| {
        for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
            let v = ((x as f32 * 0.3 + y as f32 * 0.2 + seed).sin() * 0.4 + 0.5).clamp(0.0, 1.0);
            pixel.copy_from_slice(&[v, v, v]);
        }
        Ok(())
    })
    .unwrap();
}

/// Six frames, their focus maps, and the weights over them.
fn fixture(dir: &Path) -> (Vec<PathBuf>, Vec<FocusMap>, HashMap<PathBuf, Transform>) {
    let paths: Vec<PathBuf> = (0..6)
        .map(|i| {
            let p = dir.join(format!("f{i}.tif"));
            write_frame(&p, i as f32);
            p
        })
        .collect();

    let by_path: HashMap<PathBuf, Transform> = paths
        .iter()
        .cloned()
        .map(|p| (p, Transform::IDENTITY))
        .collect();

    let metric = WindowedLaplacian::new(2, dir, by_path.clone());
    let maps = paths
        .iter()
        .map(|p| metric.evaluate(&Image::open(p).unwrap(), &()).unwrap())
        .collect();

    (paths, maps, by_path)
}

#[test]
fn fusion_stops_partway_through_the_stack() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, maps, by_path) = fixture(dir.path());

    let weights = GuidedWeights::new(
        paths.clone(),
        vec![Transform::IDENTITY; paths.len()],
        2,
        1e-4,
        GuideSpace::Perceptual,
        dir.path(),
    )
    .weights(&maps, &())
    .unwrap();

    let images: Vec<Image> = paths.iter().map(|p| Image::open(p).unwrap()).collect();

    // Cancel after two frames have been folded in. The run must not reach the sixth.
    let control = CancelAfter::new(2);
    let fusion = SelectionFusion::new(&dir.path().join("out.tif"), by_path, 8, 2);
    // `unwrap_err` needs the Ok type to be Debug and `Image` is not, so match instead.
    let Err(error) = fusion.fuse(&images, &weights, &control) else {
        panic!("expected the run to be cancelled");
    };
    assert!(
        matches!(error, Error::Cancelled),
        "expected Cancelled, got {error}"
    );

    let reports = control.reports();
    assert_eq!(reports.len(), 2, "should have stopped after two frames");
    assert!(reports.iter().all(|(stage, ..)| *stage == Stage::Fuse));
    assert_eq!(reports[1], (Stage::Fuse, 2, 6));

    // The output is the real check: fusion writes only after the accumulate loop, so a
    // cancelled run must leave no file at all rather than a plausible-looking partial.
    assert!(
        !dir.path().join("out.tif").exists(),
        "a cancelled fusion must not leave an output file"
    );
}

#[test]
fn the_blend_rule_stops_too() {
    // Both `ImageFusion` impls carry their own copy of the accumulate loop, so the
    // check has to exist in both. Testing one would leave the other free to ignore it.
    let dir = tempfile::tempdir().unwrap();
    let (paths, maps, by_path) = fixture(dir.path());

    let weights = GuidedWeights::new(
        paths.clone(),
        vec![Transform::IDENTITY; paths.len()],
        2,
        1e-4,
        GuideSpace::Perceptual,
        dir.path(),
    )
    .weights(&maps, &())
    .unwrap();
    let images: Vec<Image> = paths.iter().map(|p| Image::open(p).unwrap()).collect();

    let control = CancelAfter::new(1);
    let fusion = LaplacianPyramidFusion::new(&dir.path().join("blend.tif"), by_path, 8);
    let Err(error) = fusion.fuse(&images, &weights, &control) else {
        panic!("expected the run to be cancelled");
    };
    assert!(matches!(error, Error::Cancelled), "got {error}");
    assert_eq!(control.reports().len(), 1);
}

/// Weights honours cancellation, but *not* at an exact frame — and that is correct.
///
/// The per-frame loop runs across threads, so by the time the flag is set every frame
/// still in flight finishes; on a machine with more cores than this fixture has frames,
/// that is all of them, and the run then stops at `normalize`. Asserting "stopped after
/// exactly three" encoded a sequential assumption and broke the moment the loop went
/// wide.
///
/// What remains guaranteed, and is what this checks: the call fails with `Cancelled`
/// rather than returning weights. An implementation that ignored the flag would return
/// `Ok` and fail here, which is the regression worth catching.
///
/// Fusion is different and its tests still pin an exact frame: its per-frame loop is
/// deliberately sequential, because float addition into the accumulator is not
/// associative and going wide there would change the output.
#[test]
fn weights_honours_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let (paths, maps, _) = fixture(dir.path());

    let control = CancelAfter::new(3);
    let estimator = GuidedWeights::new(
        paths.clone(),
        vec![Transform::IDENTITY; paths.len()],
        2,
        1e-4,
        GuideSpace::Perceptual,
        dir.path(),
    );
    let Err(error) = estimator.weights(&maps, &control) else {
        panic!("expected the run to be cancelled");
    };
    assert!(matches!(error, Error::Cancelled), "got {error}");

    let reports = control.reports();
    assert!(
        reports.iter().all(|(stage, ..)| *stage == Stage::Weights),
        "weights should only report its own stage"
    );
    assert!(
        !reports.is_empty() && reports.len() <= paths.len(),
        "expected between 1 and {} reports, got {}",
        paths.len(),
        reports.len()
    );
}

#[test]
fn a_control_that_never_cancels_runs_to_completion() {
    // The counterpart that stops these tests from passing trivially: if a stage returned
    // `Cancelled` unconditionally, every assertion above would still hold.
    let dir = tempfile::tempdir().unwrap();
    let (paths, maps, by_path) = fixture(dir.path());

    let weights = GuidedWeights::new(
        paths.clone(),
        vec![Transform::IDENTITY; paths.len()],
        2,
        1e-4,
        GuideSpace::Perceptual,
        dir.path(),
    )
    .weights(&maps, &())
    .unwrap();
    let images: Vec<Image> = paths.iter().map(|p| Image::open(p).unwrap()).collect();

    let out = dir.path().join("full.tif");
    SelectionFusion::new(&out, by_path, 8, 2)
        .fuse(&images, &weights, &())
        .unwrap();
    assert!(out.exists(), "an uncancelled run must produce its output");
}
