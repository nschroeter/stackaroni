//! The output must not change. This is the gate every optimisation passes through.
//!
//! ```text
//! cargo test --release -p stackaroni-core --test output_is_stable -- --ignored --nocapture
//! ```
//!
//! A speedup that alters a single pixel is not a speedup, it is an unreviewed algorithm
//! change wearing one's clothes. Re-rating the output by eye would be the wrong instrument
//! here — this log already records metrics pointing the wrong way twice, and a human cannot
//! see a one-ULP difference anyway. A hash can, and it cannot be argued with.
//!
//! **If this fails, the correct response is not to update the constant.** It is to find what
//! changed and decide, deliberately, whether the new output is better — which means an
//! eval-log row and a rating, not a new hash. The constant moves only after that.
//!
//! Deliberately runs the *whole* pipeline rather than one stage: registration feeds focus
//! feeds weights feeds fusion, and an optimisation in any of them lands here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use stackaroni_core::defaults;
use stackaroni_core::discovery::discover_stack;
use stackaroni_core::focus::{WindowedLaplacian, evaluate_stack};
use stackaroni_core::fusion::SelectionFusion;
use stackaroni_core::pipeline::{Image, ImageFusion, Transform, WeightEstimator};
use stackaroni_core::registration::{PhaseCorrelation, register_stack};
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

/// The live configuration, as recorded in `docs/eval-log.md`.
///
/// Spelled out rather than read from `core::defaults`, because [`EXPECTED`] is only valid
/// for these exact values — sourcing them would let a defaults edit change what is being
/// hashed. The coupling is checked instead, by the test below.
const REGISTRATION_LEVEL: u32 = 3;
const FOCUS_RADIUS: u32 = 4;
const GUIDE_RADIUS: u32 = 4;
const GUIDE_EPSILON: f32 = 1e-4;
const SALIENCE_RADIUS: u32 = 2;
const PYRAMID_FLOOR: u32 = 32;
const GUIDE_SPACE: GuideSpace = GuideSpace::Perceptual;

/// Hash of the fused output for `synthetic_50` at the configuration above.
///
/// Established on 2026-08-10 from the build at `0a94bd7`, before any optimisation work.
const EXPECTED: u64 = 0x0045_5c66_dd1e_4c95;

/// Is the pinned configuration still the one that ships?
///
/// The gate itself needs `test-data/` and is therefore `#[ignore]`d, so it never runs in
/// CI and cannot notice this. This one needs nothing and always runs: if a default moves,
/// the hash above stops describing the shipped pipeline, and the failure says so at the
/// moment of the edit rather than at the next manual eval.
#[test]
fn the_pinned_configuration_is_still_the_shipped_one() {
    let pinned = [
        (
            "registration_level",
            REGISTRATION_LEVEL,
            defaults::REGISTRATION_LEVEL,
        ),
        ("focus_radius", FOCUS_RADIUS, defaults::FOCUS_RADIUS),
        ("guide_radius", GUIDE_RADIUS, defaults::GUIDE_RADIUS),
        (
            "salience_radius",
            SALIENCE_RADIUS,
            defaults::SALIENCE_RADIUS,
        ),
        ("pyramid_floor", PYRAMID_FLOOR, defaults::PYRAMID_FLOOR),
    ];
    for (name, pinned, shipped) in pinned {
        assert_eq!(
            pinned, shipped,
            "{name} default changed to {shipped}; the hash gate still pins {pinned}. \
             Changing a default invalidates the eval-log ratings: re-run the eval, \
             re-establish EXPECTED, and update this file's constants together."
        );
    }
    assert_eq!(
        GUIDE_EPSILON,
        defaults::GUIDE_EPSILON,
        "guide_epsilon default changed"
    );
    assert_eq!(
        GUIDE_SPACE,
        defaults::GUIDE_SPACE,
        "guide_space default changed"
    );
    // A compile-time failure rather than a runtime one, since it is knowable then: the
    // gate calls `SelectionFusion` directly, so flipping this default would leave the
    // shipped fusion rule untested rather than merely differently configured.
    const { assert!(defaults::SELECT_FUSION, "the gate fuses by selection") }
}

/// FNV-1a, written out rather than pulled in: one dependency for sixteen bytes of state is
/// not a trade worth making, and nothing here needs collision resistance against an
/// adversary — only against accident.
fn hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

#[test]
#[ignore = "requires test-data/synthetic_50, run with --release"]
fn the_fused_output_is_byte_identical() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-data/synthetic_50");
    let Ok(stack) = discover_stack(&dir) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };

    let scratch = tempfile::tempdir().unwrap();
    let output = scratch.path().join("fused.tif");
    let started = Instant::now();

    let registration = PhaseCorrelation::new(REGISTRATION_LEVEL);
    let transforms = register_stack(&registration, &stack.frames, &()).unwrap();
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let metric = WindowedLaplacian::new(FOCUS_RADIUS, scratch.path(), by_path.clone());
    let focus_maps = evaluate_stack(&metric, &stack.frames, &()).unwrap();

    let weights = GuidedWeights::new(
        stack.frames.clone(),
        transforms,
        GUIDE_RADIUS,
        GUIDE_EPSILON,
        GUIDE_SPACE,
        scratch.path(),
    )
    .weights(&focus_maps, &())
    .unwrap();

    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();
    SelectionFusion::new(&output, by_path, PYRAMID_FLOOR, SALIENCE_RADIUS)
        .fuse(&images, &weights, &())
        .unwrap();

    let elapsed = started.elapsed();
    let bytes = std::fs::read(&output).unwrap();
    let got = hash(&bytes);

    println!("\n=== synthetic_50, full pipeline ===");
    println!("{:.1?}  {} bytes  hash {got:#018x}", elapsed, bytes.len());

    if EXPECTED == 0 {
        println!("\nNo baseline recorded yet. Set EXPECTED to the hash above.\n");
        return;
    }
    assert_eq!(
        got, EXPECTED,
        "\nthe fused output changed.\n\
         Do not simply update EXPECTED. Find what changed, decide whether the new output is \
         better, and record that decision in docs/eval-log.md with a rating — then move the \
         constant.\n"
    );
}
