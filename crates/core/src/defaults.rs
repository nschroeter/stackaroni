//! The parameter values the pipeline is tuned to, in one place.
//!
//! These are not arbitrary starting points: they are the settings every row in
//! `docs/eval-log.md` was scored under (blossom 5/5, ruler 5/5, synthetic_50 4/5). A run
//! that touches none of them reproduces those results, so changing a value here
//! invalidates the ratings and needs a fresh eval, not just a passing build.
//!
//! They live in `core` rather than in either front-end because both `cli` and `app` expose
//! the same knobs, and previously each spelled the numbers out itself. Two independent
//! literal lists that happen to agree are indistinguishable from one source of truth right
//! up until someone edits one of them.

use crate::weights::GuideSpace;

/// Pyramid level for phase correlation.
pub const REGISTRATION_LEVEL: u32 = 3;

/// Window radius for the Laplacian focus measure, at every scale it is evaluated on.
pub const FOCUS_RADIUS: u32 = 4;

/// Whether focus is measured across a pyramid (§4) rather than at one scale (§3).
pub const MULTI_SCALE_FOCUS: bool = false;

/// Pyramid levels the multi-scale measure sums over. Read only when
/// [`MULTI_SCALE_FOCUS`] is set; `1` is the single-scale measure exactly.
///
/// **Not a tuned value, unlike everything else in this file.** The sweep that settled
/// `GUIDE_RADIUS` was run over scales 1-5 on synthetic_50 and came back flat — detail
/// 0.330 at every setting — so there is no measured basis for preferring any of them. 3 is
/// the midpoint of the exposed range. See `docs/eval-log.md`; do not cite this as tuned.
pub const FOCUS_SCALES: u32 = 3;

/// Per-octave weight decay for the multi-scale measure. Conventional octave halving; the
/// sweep found the output equally insensitive to this between 0.25 and 2.0.
pub const FOCUS_DECAY: f32 = 0.5;

/// Guided-filter radius for weight refinement.
pub const GUIDE_RADIUS: u32 = 4;

/// Guided-filter regularization.
pub const GUIDE_EPSILON: f32 = 1e-4;

/// Tone space the guided filter's guide image is measured in.
pub const GUIDE_SPACE: GuideSpace = GuideSpace::Perceptual;

/// Whether pyramid levels are combined by windowed selection rather than weighted blend.
///
/// Flipped to selection in T11; see `docs/algorithms.md` §6b.
pub const SELECT_FUSION: bool = true;

/// Salience window radius, read only when [`SELECT_FUSION`] is set.
pub const SALIENCE_RADIUS: u32 = 2;

/// Size at which the pyramid stops halving.
pub const PYRAMID_FLOOR: u32 = 32;
