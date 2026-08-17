//! What a run will cost in memory, and how much of it we allow.
//!
//! **Why this exists.** A strip is the TIFF decoder's atomic unit, so a frame written as
//! one strip is entirely resident while its rows are read — 300 MB on a 50 MP frame,
//! against ~13 MB for the same pixels in 64-row strips. Multiply that by the frames a
//! stage reads at once and a 33-frame run went from 6.3 GB to 25 GB before T18. The
//! per-reader cost is unavoidable; how many readers exist at once is not.
//!
//! **Two jobs, one model.** [`estimate`] predicts peak memory for a given parallelism,
//! and [`fit`] picks the largest parallelism whose prediction stays under [`limit_bytes`].
//! A run predicted to exceed the limit even single-threaded is reported to the caller,
//! which warns and lets the user proceed anyway — the CLI with `--ignore-memory-limit`,
//! the app with a button. **Nothing here refuses on its own**: it returns an [`Estimate`]
//! and the decision belongs to the caller, the same way [`crate::image::Coverage`] reports
//! geometry rather than acting on it.
//!
//! **Concurrency is capped rather than caches evicted, and the distinction is the whole
//! design.** An LRU budget shared across readers was the obvious shape and is the wrong
//! one: a budget below one strip would make every row read re-decode 300 MB, turning a
//! memory problem into a catastrophic time one. A worker genuinely needs its strip while
//! it works. So the budget decides *how many workers* run, never *what they may keep*.
//!
//! **The cap does not bind on the normal path.** Striped stacks charge ~13 MB a reader,
//! so [`run_bounded`] hands the work straight to the global pool and scheduling is
//! exactly as it was. It binds on single-strip input, which is where the 25 GB came from.

use std::path::Path;
use std::sync::OnceLock;

use crate::error::Result;
use crate::image::{BAND_ROWS, FrameInfo};
use crate::tiff_io::cache_bytes_max;

/// Floor for the memory limit, whatever the machine.
///
/// Below this a 50 MP stack cannot run at all — fusion alone needs ~3 GB and the stage
/// working set another ~3 GB — so a smaller limit would refuse every real run rather than
/// prevent anything.
const LIMIT_FLOOR: u64 = 16 << 30;

/// Share of physical RAM the limit takes when that exceeds [`LIMIT_FLOOR`].
const LIMIT_SHARE: f64 = 0.25;

/// Ceiling as a share of physical RAM.
///
/// **Without this the floor defeats the feature exactly where it matters.** On an 8 GB
/// machine `max(16 GB, 25%)` is 16 GB — twice the RAM — so nothing would ever warn on the
/// machine most likely to thrash. Clamping keeps the limit meaningful there while leaving
/// it untouched on anything from 18 GB up.
const LIMIT_CEILING: f64 = 0.90;

/// Full-resolution RGB `f32` planes fusion holds at once.
///
/// The warped frame, its Laplacian bands, the accumulator pyramid, the applied-weight
/// planes and the reconstructed result. Independent of stack depth, because frames
/// accumulate one at a time — see the memory note in [`crate::fusion`].
const FUSION_PLANES: u64 = 5;

/// Banded working buffers the guided filter holds per thread.
///
/// Fitted, not derived: the filter allocates a shifting number of temporaries per band,
/// and this is the count that reproduces the measurements in the tests below. The *shape*
/// is structural — a band plus its halo, at full width — so it scales correctly with frame
/// size and radius even though the multiplier is calibration.
///
/// **It is large because this stage alone has to reach the measured peak.** Under the `max`
/// in [`estimate`] a stage is not topped up by the others, so the striped measurements —
/// where weights is the driver — pin this on its own. Fitting it as though stages summed
/// gave 20 and under-predicted every striped run by nearly half.
const WEIGHT_BAND_BUFFERS: u64 = 44;

/// Grid-sized buffers registration holds per thread, at its working pyramid level.
///
/// Fitted the same way as [`WEIGHT_BAND_BUFFERS`]: phase correlation holds several real
/// and complex planes per pair, and `rustfft` allocates scratch of its own.
const REGISTRATION_GRID_BUFFERS: u64 = 39;

/// Scratch-plane pages still dirty per frame when a stage peaks.
///
/// Focus maps and weight planes are mmapped files, so most of their pages are written back
/// and evicted rather than held — but not instantly, and the residue scales with depth.
/// Fitted: it is what separates the 8-frame and 33-frame measurements on striped input,
/// where nothing else changes.
const SCRATCH_RESIDUE_PER_FRAME: u64 = 15 << 20;

/// Physical RAM, cached. Falls back to 1 GB if the machine reports nothing sensible.
fn total_ram() -> u64 {
    static RAM: OnceLock<u64> = OnceLock::new();
    *RAM.get_or_init(|| {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        system.total_memory().max(1 << 30)
    })
}

/// How much memory a run may be predicted to need before the user is warned.
pub fn limit_bytes() -> u64 {
    limit_for(total_ram())
}

/// [`limit_bytes`] with the machine passed in, so every branch is testable.
fn limit_for(ram: u64) -> u64 {
    let share = (ram as f64 * LIMIT_SHARE) as u64;
    share
        .max(LIMIT_FLOOR)
        .min((ram as f64 * LIMIT_CEILING) as u64)
}

/// Everything the estimate depends on, read from headers rather than pixels.
#[derive(Debug, Clone, Copy)]
pub struct Workload {
    pub frames: usize,
    pub info: FrameInfo,
    /// What one open reader holds — [`crate::tiff_io::cache_bytes_max`].
    pub cache_bytes_per_frame: u64,
    pub guide_radius: u32,
    pub registration_level: u32,
}

impl Workload {
    /// Probe the first frame and describe the run. Header reads only.
    pub fn probe(
        first_frame: &Path,
        frames: usize,
        info: FrameInfo,
        guide_radius: u32,
        registration_level: u32,
    ) -> Result<Self> {
        Ok(Self {
            frames,
            info,
            cache_bytes_per_frame: cache_bytes_max(first_frame)?,
            guide_radius,
            registration_level,
        })
    }

    fn plane_bytes(&self) -> u64 {
        self.info.width as u64 * self.info.height as u64 * 4
    }

    /// One band plus the halo the guided filter reads around it, at full width.
    fn band_bytes(&self) -> u64 {
        let rows = BAND_ROWS as u64 + 4 * self.guide_radius as u64;
        rows * self.info.width as u64 * 4
    }

    /// One registration grid, at the level phase correlation works on.
    fn grid_bytes(&self) -> u64 {
        let shift = self.registration_level;
        let w = (self.info.width >> shift).max(1) as u64;
        let h = (self.info.height >> shift).max(1) as u64;
        w * h * 4
    }
}

/// A prediction, and what it was judged against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    /// Predicted peak, in bytes.
    pub peak_bytes: u64,
    /// Frames read at once that this prediction assumes.
    pub concurrency: usize,
    /// The limit it was compared against.
    pub limit_bytes: u64,
    /// Whether parallelism had to be reduced to reach this prediction.
    pub reduced: bool,
}

impl Estimate {
    pub fn fits(&self) -> bool {
        self.peak_bytes <= self.limit_bytes
    }
}

/// Predicted peak for a workload run with `concurrency` frames in flight on `threads`.
///
/// **The `max` is the whole model, and the measurements forced it.** Stages run one after
/// another, so the peak is the worst stage, not the sum. Eight single-strip frames measured
/// the *same* as eight striped ones — 5.72 against 5.88 GB — because registration's seven
/// concurrent pairs cost 4.2 GB there, which is under what weights and fusion cost anyway
/// and therefore never visible. At 33 frames the same term is 8.4 GB and becomes the peak,
/// which is the whole 10.10 GB measurement. A model that summed stages would have put the
/// 8-frame case at nearly twice its true cost, and warned on runs that were fine.
pub fn estimate(workload: &Workload, concurrency: usize, threads: usize) -> u64 {
    let (c, t) = (concurrency as u64, threads as u64);
    let cache = workload.cache_bytes_per_frame;
    let scratch = workload.frames as u64 * SCRATCH_RESIDUE_PER_FRAME;

    // Two readers a task: a pair alignment holds reference and target at once.
    let registration = c * 2 * cache + t * REGISTRATION_GRID_BUFFERS * workload.grid_bytes();
    // One reader a task, plus the banded luma and energy buffers.
    let focus = c * cache + t * 4 * workload.band_bytes();
    // No frame readers: the guided filter works from scratch planes.
    let weights = t * WEIGHT_BAND_BUFFERS * workload.band_bytes();
    // Sequential, one frame at a time, so depth does not enter.
    let fusion = FUSION_PLANES * 3 * workload.plane_bytes() + cache;

    scratch + registration.max(focus).max(weights).max(fusion)
}

/// The largest parallelism whose prediction fits the limit, and that prediction.
///
/// Never returns zero concurrency: a workload too large even single-threaded is reported as
/// not fitting, for the caller to warn about, rather than made unrunnable. Reducing
/// parallelism is silent when it succeeds — the run simply takes longer, which is the
/// outcome the user asked for over failing.
pub fn fit(workload: &Workload) -> Estimate {
    let threads = rayon::current_num_threads().max(1);
    let limit = limit_bytes();

    for concurrency in (1..=threads).rev() {
        let peak = estimate(workload, concurrency, threads);
        if peak <= limit {
            return Estimate {
                peak_bytes: peak,
                concurrency,
                limit_bytes: limit,
                reduced: concurrency < threads,
            };
        }
    }
    Estimate {
        peak_bytes: estimate(workload, 1, threads),
        concurrency: 1,
        limit_bytes: limit,
        reduced: threads > 1,
    }
}

/// How many tasks each holding `per_task_bytes` of frame cache may run at once.
///
/// **The per-stage guard, as distinct from [`fit`].** `fit` predicts the whole run so the
/// caller can warn about it; this keeps one stage's readers inside the limit whether or not
/// anybody consulted the estimate — a library caller that never calls `fit` still gets the
/// bound. The two agree by construction, because both compare against [`limit_bytes`].
///
/// Never zero and never above the pool's width: a task larger than the whole limit is
/// admitted alone rather than refused, because refusing would mean a machine with too
/// little RAM cannot open a frame at all, and one frame at a time is exactly the streaming
/// behaviour the pipeline should degrade to.
pub fn concurrency_for(per_task_bytes: u64) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let fits = limit_bytes() / per_task_bytes.max(1);
    (fits as usize).clamp(1, threads)
}

/// Run `work` with rayon parallelism capped to `concurrency`.
///
/// Takes the number rather than computing it, so the figure the user was shown is the
/// figure the run uses. A caller with no estimate passes `rayon::current_num_threads()`.
pub fn run_bounded<R: Send>(concurrency: usize, work: impl FnOnce() -> R + Send) -> R {
    if concurrency >= rayon::current_num_threads() {
        // The common case, and deliberately not routed through a private pool: where the
        // cap is not binding the work should schedule exactly as it did before this module
        // existed.
        return work();
    }
    match rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency.max(1))
        .build()
    {
        Ok(pool) => pool.install(work),
        // Pool construction fails when the OS will not give us threads. Losing the bound is
        // better than losing the run: the global pool still computes the right answer.
        Err(_) => work(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    /// 8664x5784 RGB16 in one strip, and the same file in 1-row strips at CACHE_ROWS.
    const SINGLE_STRIP: u64 = 8664 * 5784 * 3 * 2;
    const STRIPED: u64 = 8664 * 3 * 2 * 256;

    fn blossom(cache_bytes_per_frame: u64, frames: usize) -> Workload {
        Workload {
            frames,
            info: FrameInfo {
                width: 8664,
                height: 5784,
                samples: 3,
                bits_per_sample: 16,
            },
            cache_bytes_per_frame,
            guide_radius: crate::defaults::GUIDE_RADIUS,
            registration_level: crate::defaults::REGISTRATION_LEVEL,
        }
    }

    /// **The four measurements this model was fitted to** (CLI, macOS, 14 threads,
    /// `/usr/bin/time -l` peak footprint, 2026-08-16). They are why the estimator is a
    /// `max` over stages rather than a sum, and why the fitted constants have the values
    /// they do.
    ///
    /// The estimate must never come in *under* a measurement — a warning that cannot fire
    /// when memory is genuinely short is worse than no warning — and must not exceed it by
    /// more than 35%, or it warns on runs that would have been fine and gets ignored.
    #[test]
    fn the_estimate_brackets_every_measured_run() {
        let measured: [(&str, u64, usize, f64); 4] = [
            ("striped 8", STRIPED, 8, 5.880e9),
            ("striped 33", STRIPED, 33, 6.255e9),
            ("single-strip 8", SINGLE_STRIP, 8, 5.716e9),
            ("single-strip 33", SINGLE_STRIP, 33, 10.096e9),
        ];
        for (name, cache, frames, actual) in measured {
            let workload = blossom(cache, frames);
            // Concurrency is bounded by the work available: 8 frames is 7 pairs.
            let concurrency = (frames - 1).min(14);
            let predicted = estimate(&workload, concurrency, 14) as f64;
            assert!(
                predicted >= actual,
                "{name}: predicted {:.3} GB is under the measured {:.3} GB",
                predicted / 1e9,
                actual / 1e9
            );
            assert!(
                predicted <= actual * 1.35,
                "{name}: predicted {:.3} GB exceeds the measured {:.3} GB by over 35%",
                predicted / 1e9,
                actual / 1e9
            );
        }
    }

    /// The limit on the machines this has run on, and on the ones it has not.
    #[test]
    fn the_limit_has_a_floor_and_a_ceiling() {
        assert_eq!(limit_for(36 * GIB), 16 * GIB, "this Mac");
        assert_eq!(limit_for(64 * GIB), 16 * GIB, "the Windows reporter's");
        assert_eq!(
            limit_for(128 * GIB),
            32 * GIB,
            "the share takes over above 64 GB"
        );
        // The clamp that stops the floor exceeding a small machine's own RAM.
        assert!(limit_for(8 * GIB) < 8 * GIB);
    }

    /// Striped input never has to give up parallelism.
    #[test]
    fn a_striped_stack_fits_at_full_width() {
        let fitted = fit(&blossom(STRIPED, 100));
        assert!(fitted.fits());
        assert!(!fitted.reduced);
        assert_eq!(fitted.concurrency, rayon::current_num_threads());
    }

    /// Single-strip input is where the cap earns its place: readers alone would take
    /// 8.4 GB at full width on this machine.
    #[test]
    fn a_single_strip_stack_gives_up_parallelism_rather_than_the_run() {
        let workload = blossom(SINGLE_STRIP, 100);
        let threads = rayon::current_num_threads();
        let fitted = fit(&workload);
        assert!(fitted.concurrency >= 1);
        assert!(fitted.peak_bytes <= estimate(&workload, threads, threads));
        if fitted.reduced {
            assert!(
                fitted.fits(),
                "reducing parallelism should have been enough"
            );
        }
    }

    /// A workload too large even single-threaded is reported, not made unrunnable — the
    /// caller warns and the user may still proceed.
    #[test]
    fn an_impossible_workload_is_reported_rather_than_refused() {
        let mut workload = blossom(SINGLE_STRIP, 10);
        workload.info.width = 60_000;
        workload.info.height = 40_000;
        let fitted = fit(&workload);
        assert_eq!(fitted.concurrency, 1);
        assert!(!fitted.fits(), "a 2.4 gigapixel frame should not fit 16 GB");
    }

    #[test]
    fn bounded_work_still_runs_and_returns() {
        use rayon::prelude::*;
        let sum: u64 = run_bounded(1, || (0u64..1000).into_par_iter().sum());
        assert_eq!(sum, 499_500);
    }
}
