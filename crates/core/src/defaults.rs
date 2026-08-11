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

use crate::fusion::FusionKind;
use crate::weights::GuideSpace;

/// Pyramid level for phase correlation.
pub const REGISTRATION_LEVEL: u32 = 3;

/// Window radius for the windowed-Laplacian focus measure.
pub const FOCUS_RADIUS: u32 = 4;

/// Guided-filter radius for weight refinement.
pub const GUIDE_RADIUS: u32 = 4;

/// Guided-filter regularization.
pub const GUIDE_EPSILON: f32 = 1e-4;

/// Tone space the guided filter's guide image is measured in.
pub const GUIDE_SPACE: GuideSpace = GuideSpace::Perceptual;

/// How pyramid levels are combined.
///
/// Flipped to selection in T11; see `docs/algorithms.md` §6b.
pub const FUSION: FusionKind = FusionKind::Select {
    salience_radius: SALIENCE_RADIUS,
};

/// Salience window radius. Carried by [`FusionKind::Select`], which is the only rule
/// that reads it.
pub const SALIENCE_RADIUS: u32 = 2;

/// Size at which the pyramid stops halving.
pub const PYRAMID_FLOOR: u32 = 32;
