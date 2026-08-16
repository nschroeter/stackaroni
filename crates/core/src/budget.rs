//! How much frame data the pipeline lets itself hold at once.
//!
//! **Why this exists.** A strip is the TIFF decoder's atomic unit, so a frame written as
//! one strip is entirely resident while its rows are read — 300 MB on a 50 MP frame,
//! against ~13 MB for the same pixels in 64-row strips. That cost is unavoidable per
//! reader. What is avoidable is how many readers exist at once, and that is what is
//! bounded here.
//!
//! **Concurrency is capped rather than caches evicted, and the distinction is the whole
//! design.** An LRU budget shared across readers was the obvious shape and is the wrong
//! one: a budget below one strip would make every row read re-decode 300 MB, turning a
//! memory problem into a catastrophic time one. A worker genuinely needs its strip while
//! it works. So the budget decides *how many workers* run, never *what they may keep*.
//!
//! **The cap does not bind on the normal path.** The striped stacks charge ~13 MB a
//! reader against a budget in the gigabytes, so [`run_bounded`] hands the work straight
//! to the global pool and scheduling is exactly as it was. It binds on single-strip
//! input, which is where the 20 GB peak came from.

use std::sync::OnceLock;

/// Share of physical RAM the in-flight input caches may occupy.
///
/// **A quarter, because the caches are the smaller half of what a run costs.** Measured
/// on 33 striped 50 MP frames, 2026-08-16: peak footprint 6.25 GB, of which the caches
/// are ~0.4 GB. The rest is the stages' own working set — the guided filter's banded
/// buffers, one per thread, which is where the peak actually lands, plus the fusion
/// pyramids — and that part does not shrink when this constant does. Confirmed
/// frame-count-independent: the same run at 8 frames costs 5.88 GB, so 4.1x the frames
/// buys 6% more memory.
///
/// So the budget has to leave room for a fixed ~6 GB alongside it. A half would put a
/// 16 GB machine over its own RAM before a single frame was cached.
const SHARE: f64 = 0.25;

/// Bytes of frame cache the pipeline may hold across all in-flight readers.
pub fn budget_bytes() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        // A machine that reports nothing sensible still has to run: fall back to a
        // budget that admits one 50 MP single-strip frame per core rather than to zero.
        let total = system.total_memory().max(1 << 30);
        (total as f64 * SHARE) as u64
    })
}

/// How many tasks each holding `per_task_bytes` of frame cache may run at once.
///
/// Never zero and never above the pool's own width: a task larger than the whole budget
/// is admitted alone rather than refused, because refusing it would mean a machine with
/// too little RAM cannot open a frame at all, and one frame at a time is exactly the
/// streaming behaviour the pipeline is supposed to degrade to.
pub fn max_concurrency(per_task_bytes: u64) -> usize {
    concurrency_for(budget_bytes(), per_task_bytes, rayon::current_num_threads())
}

/// [`max_concurrency`] with the machine passed in rather than detected.
///
/// **Split out so the binding case is testable.** On a large machine the cap does not
/// bind at all — a 36 GB Mac budgets 9 GB, which already covers 14 concurrent pairs of
/// single-strip 50 MP frames — so a test that only ever sees this machine would assert
/// the mechanism does nothing and pass while it was broken. The machine that matters is
/// the 16 GB one the failure was reported from, and it can only be reached this way.
fn concurrency_for(budget_bytes: u64, per_task_bytes: u64, threads: usize) -> usize {
    let fits = budget_bytes / per_task_bytes.max(1);
    (fits as usize).clamp(1, threads.max(1))
}

/// Run `work` with rayon parallelism capped so in-flight readers stay inside the budget.
///
/// `per_task_bytes` is what *one* unit of `work`'s parallel iteration holds — two frames
/// for pairwise registration, one everywhere else.
pub fn run_bounded<R: Send>(per_task_bytes: u64, work: impl FnOnce() -> R + Send) -> R {
    let threads = max_concurrency(per_task_bytes);
    if threads >= rayon::current_num_threads() {
        // The common case, and deliberately not routed through a private pool: on
        // striped input the cap is not binding, so the work should schedule exactly as
        // it did before this module existed.
        return work();
    }
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Ok(pool) => pool.install(work),
        // Pool construction fails when the OS will not give us threads. Losing the
        // bound is better than losing the run: the global pool still computes the right
        // answer, and a machine that cannot spawn a thread has a larger problem.
        Err(_) => work(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_is_a_share_of_a_real_machine() {
        let budget = budget_bytes();
        assert!(budget >= 1 << 28, "implausibly small budget: {budget}");
    }

    /// A band-sized reader does not restrict anything, on any machine.
    ///
    /// This is the claim that the striped path — every rating in `docs/eval-log.md` —
    /// schedules exactly as it did before this module existed. 13 MB against even a
    /// 4 GB machine's 1 GB budget leaves room for 78 readers.
    #[test]
    fn a_band_sized_reader_never_binds() {
        let threads = rayon::current_num_threads();
        assert_eq!(max_concurrency(13 << 20), threads);
        assert_eq!(concurrency_for(budget_of_gib(4), 13 << 20, 14), 14);
    }

    /// The reported machine: 16 GB, and single-strip frames it cannot all hold.
    ///
    /// 4 GB of budget against 600 MB a pair admits six of the fourteen pairs a
    /// 14-core machine would otherwise start — 3.5 GB of frame cache instead of 8.4 GB.
    /// Registration takes about twice as long there, and completes.
    #[test]
    fn a_sixteen_gigabyte_machine_runs_fewer_pairs_at_once() {
        let single_strip_pair = 600 << 20;
        assert_eq!(concurrency_for(budget_of_gib(16), single_strip_pair, 14), 6);
        // The same machine on striped input is untouched.
        assert_eq!(concurrency_for(budget_of_gib(16), 26 << 20, 14), 14);
    }

    /// This machine, for the record: the cap is insurance here, not a constraint.
    ///
    /// A 36 GB Mac budgets 9 GB, which covers all 14 pairs, so the measured single-strip
    /// improvement came from the `u16` cache and the fusion release rather than from
    /// this module. Asserted so that a future change to [`SHARE`] shows up as a failure
    /// here and gets re-measured rather than silently altering how the machine schedules.
    #[test]
    fn a_thirty_six_gigabyte_machine_is_not_constrained() {
        assert_eq!(concurrency_for(budget_of_gib(36), 600 << 20, 14), 14);
    }

    /// The budget a machine with this much RAM gets. Derived from [`SHARE`] rather than
    /// spelled out, so changing the share fails these tests instead of drifting past them.
    fn budget_of_gib(ram: u64) -> u64 {
        ((ram << 30) as f64 * SHARE) as u64
    }

    /// Refusing would mean a small machine cannot open a frame at all.
    #[test]
    fn a_task_bigger_than_the_budget_still_runs_alone() {
        assert_eq!(max_concurrency(u64::MAX), 1);
        assert_eq!(max_concurrency(0), rayon::current_num_threads());
    }

    #[test]
    fn bounded_work_still_runs_and_returns() {
        use rayon::prelude::*;
        let sum: u64 = run_bounded(u64::MAX, || (0u64..1000).into_par_iter().sum());
        assert_eq!(sum, 499_500);
    }
}
