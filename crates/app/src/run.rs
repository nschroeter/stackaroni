//! Running the pipeline on a background thread, with progress and cancellation.
//!
//! # Two hazards this type exists to handle
//!
//! **The run is over when the thread has exited, not when it has sent its result.** A
//! worker that has posted its outcome is still unwinding: dropping mmapped scratch
//! planes, deleting its scratch directory. Re-enabling "Run stack" at the message would
//! let a second run start into that cleanup. So [`Run::poll`] reports completion from
//! [`std::thread::JoinHandle::is_finished`], and each run gets its own scratch directory
//! so overlapping cleanup cannot collide even if that reasoning is ever wrong.
//!
//! **egui is immediate-mode, so a background thread changing shared state redraws
//! nothing.** The worker holds a clone of the [`egui::Context`] and calls
//! `request_repaint` whenever progress moves; without it the progress bar sits frozen
//! until unrelated input happens to trigger a pass, which looks exactly like a hang.
//!
//! # Disk accounting
//!
//! Both halves are in place, and they were built together on purpose: a free-space
//! check that still lets forgotten debris eat the headroom it is protecting is only
//! half a fix.
//!
//! *Before a run:* [`Run::start`] refuses to begin unless the temp volume has room for
//! one focus map and one weight plane per frame, both f32 and full resolution — the
//! same arithmetic the CLI uses, ~38 GB for a 100-frame 50 MP stack.
//!
//! *At startup:* [`crate::reap`] removes `stackaroni-*` temp entries whose owning
//! process has exited. That is also what covers a result the user never exported, which
//! [`Export`] alone cannot: export cleans up after itself, but only once it runs.
//!
//! One thing neither counts: the fused output itself, ~300 MB beside the scratch
//! directory. The CLI does not count it either, and against a 38 GB scratch requirement
//! it is noise — but it is an omission rather than a decision, and a stack small enough
//! for that to matter is a stack with room to spare anyway.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::thread::JoinHandle;

use eframe::egui;
use fs4::available_space;
use stackaroni_core::error::{Error, Result};
use stackaroni_core::focus::{WindowedLaplacian, evaluate_stack};
use stackaroni_core::fusion::{LaplacianPyramidFusion, SelectionFusion};
use stackaroni_core::image::FrameInfo;
use stackaroni_core::pipeline::{
    Image, ImageFusion, RunControl, Stage, Transform, WeightEstimator,
};
use stackaroni_core::registration::{PhaseCorrelation, register_stack};
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

/// Everything the pipeline needs that the UI owns.
#[derive(Clone)]
pub struct Settings {
    pub registration_level: u32,
    pub focus_radius: u32,
    pub guide_radius: u32,
    pub guide_epsilon: f32,
    pub guide_space: GuideSpace,
    pub select_fusion: bool,
    pub salience_radius: u32,
    pub pyramid_floor: u32,
}

/// Shared between the UI thread and the worker. `RunControl` is implemented on this.
pub struct Shared {
    cancel: AtomicBool,
    stage: AtomicUsize,
    done: AtomicUsize,
    total: AtomicUsize,
    ctx: egui::Context,
}

const STAGES: [Stage; 4] = [Stage::Register, Stage::Focus, Stage::Weights, Stage::Fuse];

impl Shared {
    fn new(ctx: egui::Context) -> Self {
        Self {
            cancel: AtomicBool::new(false),
            stage: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            ctx,
        }
    }

    /// Current stage, frames done, frames total.
    pub fn snapshot(&self) -> (Stage, usize, usize) {
        (
            STAGES[self.stage.load(Ordering::Relaxed).min(3)],
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

impl RunControl for Shared {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    fn progress(&self, stage: Stage, done: usize, total: usize) {
        let index = STAGES.iter().position(|s| *s == stage).unwrap_or(0);
        self.stage.store(index, Ordering::Relaxed);
        self.done.store(done, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
        // The only thing that makes any of this visible.
        self.ctx.request_repaint();
    }
}

/// How a run ended.
pub enum Outcome {
    Done(PathBuf),
    Cancelled,
    Failed(String),
}

pub struct Run {
    pub shared: Arc<Shared>,
    /// Where this run's working files and result live. Unique per run.
    ///
    /// Read only by the cancellation test today, which asserts both are gone
    /// afterwards; `Outcome::Done` already carries the output path for the UI. Kept as
    /// fields because export needs somewhere to copy *from*, and because a run that
    /// cannot say where it put things is hard to diagnose.
    #[allow(dead_code)]
    pub scratch: PathBuf,
    #[allow(dead_code)]
    pub output: PathBuf,
    handle: Option<JoinHandle<()>>,
    receiver: Receiver<Outcome>,
    outcome: Option<Outcome>,
}

impl Run {
    /// Spawn a run over `frames`, writing the fused result beside the scratch directory.
    pub fn start(
        frames: Vec<PathBuf>,
        info: FrameInfo,
        settings: Settings,
        ctx: egui::Context,
        sequence: u64,
    ) -> std::result::Result<Self, String> {
        let shared = Arc::new(Shared::new(ctx));
        let (sender, receiver) = channel();

        // Unique per run, so a previous run still tearing down cannot delete this one's
        // working files. The output sits outside it, because scratch is deleted first.
        let root = std::env::temp_dir();

        // Before anything else. Scratch holds one focus map and one weight plane per
        // frame, both f32 and full resolution — the same arithmetic the CLI uses, which
        // is ~38 GB for a 100-frame 50 MP stack. Discovering that fifteen minutes in,
        // with the failure landing somewhere inside the weights stage, is the outcome
        // this exists to prevent.
        let needed = 2 * frames.len() as u64 * info.width as u64 * info.height as u64 * 4;
        let available = available_space(&root)
            .map_err(|e| format!("checking free space on {}: {e}", root.display()))?;
        if available < needed {
            return Err(format!(
                "not enough room to run: {:.1} GB needed for scratch, {:.1} GB free on {}",
                needed as f64 / 1e9,
                available as f64 / 1e9,
                root.display()
            ));
        }
        let scratch = root.join(format!("stackaroni-run-{}-{sequence}", std::process::id()));
        let output = root.join(format!(
            "stackaroni-run-{}-{sequence}.tif",
            std::process::id()
        ));
        std::fs::create_dir_all(&scratch)
            .map_err(|e| format!("creating {}: {e}", scratch.display()))?;
        let (scratch_path, output_path) = (scratch.clone(), output.clone());

        let worker = Arc::clone(&shared);
        let handle = std::thread::spawn(move || {
            let result = pipeline(&frames, &settings, &scratch, &output, &worker);

            // Success and cancellation both clean up; a genuine failure keeps its
            // scratch, and says where, because that is the case worth inspecting.
            let outcome = match result {
                Ok(path) => {
                    let _ = std::fs::remove_dir_all(&scratch);
                    Outcome::Done(path)
                }
                Err(Error::Cancelled) => {
                    let _ = std::fs::remove_dir_all(&scratch);
                    let _ = std::fs::remove_file(&output);
                    Outcome::Cancelled
                }
                Err(e) => Outcome::Failed(format!("{e}\nscratch kept at {}", scratch.display())),
            };
            let _ = sender.send(outcome);
            worker.ctx.request_repaint();
        });

        Ok(Self {
            shared,
            scratch: scratch_path,
            output: output_path,
            handle: Some(handle),
            receiver,
            outcome: None,
        })
    }

    /// Take the outcome, but only once the worker thread has actually exited.
    ///
    /// Returning at the message instead would hand back control while the thread is
    /// still unwinding — see the note at the top of this module.
    pub fn poll(&mut self) -> Option<Outcome> {
        if let Ok(outcome) = self.receiver.try_recv() {
            self.outcome = Some(outcome);
        }
        let finished = self.handle.as_ref().is_some_and(|h| h.is_finished());
        if !finished {
            return None;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.outcome.take().or(Some(Outcome::Cancelled))
    }
}

fn pipeline(
    frames: &[PathBuf],
    settings: &Settings,
    scratch: &Path,
    output: &Path,
    run: &Shared,
) -> Result<PathBuf> {
    let registration = PhaseCorrelation::new(settings.registration_level);
    let transforms = register_stack(&registration, frames, run)?;
    let by_path: std::collections::HashMap<PathBuf, Transform> = frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let metric = WindowedLaplacian::new(settings.focus_radius, scratch, by_path.clone());
    let focus_maps = evaluate_stack(&metric, frames, run)?;

    let weights = GuidedWeights::new(
        frames.to_vec(),
        transforms,
        settings.guide_radius,
        settings.guide_epsilon,
        settings.guide_space,
        scratch,
    )
    .weights(&focus_maps, run)?;

    let images: Vec<Image> = frames
        .iter()
        .map(|p| Image::open(p))
        .collect::<Result<_>>()?;

    let fusion: Box<dyn ImageFusion> = if settings.select_fusion {
        Box::new(SelectionFusion::new(
            output,
            by_path,
            settings.pyramid_floor,
            settings.salience_radius,
        ))
    } else {
        Box::new(LaplacianPyramidFusion::new(
            output,
            by_path,
            settings.pyramid_floor,
        ))
    };
    fusion.fuse(&images, &weights, run)?;
    Ok(output.to_path_buf())
}

/// The frames a run would use, in order.
pub fn included_paths(stack: &crate::stack::Stack) -> Vec<PathBuf> {
    stack
        .frames
        .iter()
        .filter(|f| f.included)
        .map(|f| f.path.clone())
        .collect()
}

/// Copying a finished result to wherever the user chose.
///
/// # Why this is threaded, which was not obvious
///
/// Measured on a 301 MB result, which is a normal size here:
///
/// | destination | time |
/// |---|---|
/// | same APFS volume | 465 µs |
/// | across volumes | 337 ms |
///
/// The first number is misleading and nearly led to doing this inline: APFS
/// copy-on-write means `fs::copy` *clones* rather than copies when source and
/// destination share a volume, so it looks free. It is only free in that one case. The
/// cross-volume figure is a real byte copy — and 893 MB/s is a fast local disk image,
/// far better than the destinations that actually matter. An external USB drive at
/// ~100 MB/s is ~3 s for the same file, a network share slower still, and "save the
/// finished stack to my photo drive" is exactly the case that will be cross-volume.
/// Anything over ~100 ms is a visible freeze, so it goes on a worker.
///
/// `rename` is not a shortcut either: across volumes it fails outright with
/// `CrossesDevices`, so the copy path has to exist regardless.
pub struct Export {
    handle: Option<JoinHandle<()>>,
    receiver: Receiver<std::result::Result<PathBuf, String>>,
    outcome: Option<std::result::Result<PathBuf, String>>,
}

impl Export {
    pub fn start(source: PathBuf, destination: PathBuf, ctx: egui::Context) -> Self {
        let (sender, receiver) = channel();
        let handle = std::thread::spawn(move || {
            // Try the cheap path first: a same-volume rename is metadata only, and on
            // APFS a same-volume copy is a clone. Both fall back to a real copy.
            let result = match std::fs::rename(&source, &destination) {
                Ok(()) => Ok(destination.clone()),
                Err(_) => std::fs::copy(&source, &destination)
                    .map(|_| {
                        // The temp original has served its purpose; leaving it is how
                        // the orphaned-result gap accumulates.
                        let _ = std::fs::remove_file(&source);
                        destination.clone()
                    })
                    .map_err(|e| format!("{}: {e}", destination.display())),
            };
            let _ = sender.send(result);
            ctx.request_repaint();
        });
        Self {
            handle: Some(handle),
            receiver,
            outcome: None,
        }
    }

    /// The result, once the worker has actually exited — same discipline as [`Run`].
    pub fn poll(&mut self) -> Option<std::result::Result<PathBuf, String>> {
        if let Ok(outcome) = self.receiver.try_recv() {
            self.outcome = Some(outcome);
        }
        if !self.handle.as_ref().is_some_and(|h| h.is_finished()) {
            return None;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.outcome
            .take()
            .or_else(|| Some(Err("the export thread ended without a result".into())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Cancel a real 100-frame run partway through fusion.
    ///
    /// ```text
    /// cargo test --release -p stackaroni-app -- --ignored --nocapture cancelling_a_real_run
    /// ```
    ///
    /// The unit tests in `core` cancel a six-frame stack in milliseconds, which proves
    /// the checks are wired but says nothing about the case that matters: a run that has
    /// already been going for ten minutes, holding a hundred mmapped scratch planes, with
    /// tens of GB on disk to unwind. This drives the identical `Run` the UI drives, and
    /// measures the thing the design traded away — how long the stop actually takes.
    ///
    /// It cannot tell you whether pressing the button *feels* right; that needs a human
    /// and a mouse.
    #[test]
    #[ignore = "requires test-data/blossom, takes ~12 minutes, run with --release"]
    fn cancelling_a_real_run_mid_fusion() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/blossom");
        let Ok(stack) = stackaroni_core::discovery::discover_stack(&dir) else {
            eprintln!("skipping: test-data/blossom not present");
            return;
        };
        let frames = stack.frames.clone();
        println!("starting a run over {} frames", frames.len());

        let settings = Settings {
            registration_level: 3,
            focus_radius: 4,
            guide_radius: 4,
            guide_epsilon: 1e-4,
            guide_space: GuideSpace::Perceptual,
            select_fusion: true,
            salience_radius: 2,
            pyramid_floor: 32,
        };
        let started = Instant::now();
        let info = stack.probe().unwrap().info;
        let mut run = Run::start(frames, info, settings, egui::Context::default(), 9001).unwrap();
        let (scratch, output) = (run.scratch.clone(), run.output.clone());

        // Wait for fusion to be a few frames in — the longest stage, and the one whose
        // per-frame checkpoint sets the worst-case latency.
        let mut last = String::new();
        loop {
            let (stage, done, total) = run.shared.snapshot();
            let line = format!("{} {done}/{total}", stage.label());
            if line != last {
                println!("  {:>6.0}s  {line}", started.elapsed().as_secs_f32());
                last = line;
            }
            if stage == Stage::Fuse && done >= 3 {
                break;
            }
            assert!(
                run.poll().is_none(),
                "the run finished before it could be cancelled"
            );
            std::thread::sleep(Duration::from_millis(200));
        }

        let at = run.shared.snapshot();
        println!("\ncancelling during {} {}/{}", at.0.label(), at.1, at.2);
        let pressed = Instant::now();
        run.shared.cancel();

        // Exactly what the UI does: poll until the worker has *exited*, not until it
        // has sent its outcome.
        let outcome = loop {
            if let Some(outcome) = run.poll() {
                break outcome;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let stop = pressed.elapsed();

        println!("stopped after {:.2}s", stop.as_secs_f32());
        assert!(
            matches!(outcome, Outcome::Cancelled),
            "expected Cancelled after a cancel"
        );
        assert!(
            !scratch.exists(),
            "a cancelled run must clean up its scratch: {}",
            scratch.display()
        );
        assert!(
            !output.exists(),
            "a cancelled run must leave no output: {}",
            output.display()
        );
        println!("scratch and output both cleaned up");
    }
}
