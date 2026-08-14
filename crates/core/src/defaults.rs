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
use crate::pipeline::Method;
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
///
/// Shared by both methods: it is the depth of the multi-scale decomposition, and the
/// wavelet path reads it too so the two are comparable at matched depth rather than at
/// accidentally different ones.
pub const PYRAMID_FLOOR: u32 = 32;

/// Which pipeline shape runs by default.
///
/// Stays [`Method::Local`]: every rating in `docs/eval-log.md` was given to its output,
/// and `tests/output_is_stable.rs` hashes it.
pub const METHOD: Method = Method::Local { fusion: FUSION };

/// How many of a coefficient's 8 neighbours must agree before consistency verification
/// overrides its selected frame.
///
/// **A documented extension of the published rule, not the rule itself.** Li, Manjunath
/// & Mitra fuse two images, where "the majority of the neighbourhood" is always defined
/// — 5 of 8 settles it. With 100 frames, 8 neighbours can hold 8 different labels and a
/// strict majority usually does not exist, a case the paper has no reason to address.
/// So the filter takes the *plurality* and applies it only when it reaches this count.
/// At 2 frames a plurality of 5 or more is a majority, so the published behaviour is
/// recovered exactly as a special case rather than replaced.
pub const CONSISTENCY_THRESHOLD: u32 = 4;
