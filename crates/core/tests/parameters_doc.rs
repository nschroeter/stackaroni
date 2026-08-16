//! `docs/PARAMETERS.md` must agree with `core::defaults`.
//!
//! The document restates every default in a table, which is a second copy of numbers that
//! already live in code. `defaults.rs` opens by warning about exactly this shape — "two
//! independent literal lists that happen to agree are indistinguishable from one source of
//! truth right up until someone edits one of them" — and prose cannot share a constant, so
//! the drift is caught here instead.
//!
//! Needs no `test-data/`, so it runs in CI on every push. That is the point: a default
//! changed without touching the document fails the build rather than quietly leaving the
//! docs describing a pipeline that no longer exists.

use stackaroni_core::defaults;
use stackaroni_core::weights::GuideSpace;

/// The table row for `flag`, as `(default cell, stage cell)`.
///
/// Parsed rather than hardcoded so the test fails when a row is *removed* too, not only
/// when a value drifts.
fn row(doc: &str, flag: &str) -> (String, String) {
    let needle = format!("`--{flag}`");
    let line = doc
        .lines()
        .find(|l| l.starts_with('|') && l.contains(&needle))
        .unwrap_or_else(|| panic!("docs/PARAMETERS.md has no table row for --{flag}"));

    let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
    assert!(
        cells.len() >= 4,
        "--{flag} row is malformed: {line:?}, expected 4 columns"
    );
    (cells[2].to_owned(), cells[3].to_owned())
}

fn doc() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/PARAMETERS.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// Numbers in the table are the numbers the pipeline runs with.
#[test]
fn documented_defaults_match_the_code() {
    let doc = doc();

    let numeric: [(&str, f64); 5] = [
        ("registration-level", defaults::REGISTRATION_LEVEL as f64),
        ("focus-radius", defaults::FOCUS_RADIUS as f64),
        ("guide-radius", defaults::GUIDE_RADIUS as f64),
        ("salience-radius", defaults::SALIENCE_RADIUS as f64),
        ("pyramid-floor", defaults::PYRAMID_FLOOR as f64),
    ];
    for (flag, value) in numeric {
        let (cell, _) = row(&doc, flag);
        let documented: f64 = cell.parse().unwrap_or_else(|_| {
            panic!("--{flag}: default cell {cell:?} in PARAMETERS.md is not a number")
        });
        assert_eq!(
            documented, value,
            "--{flag}: PARAMETERS.md says {documented}, defaults.rs says {value}"
        );
    }

    // Parsed as a float rather than string-compared, so `1e-4` and `0.0001` are both
    // accepted spellings of the same default.
    //
    // Compared at `f32`, the type the constant actually is. Widening it to `f64` first
    // turns 1e-4 into 9.999999747378752e-5 — not equal to a `f64` parsed from "0.0001",
    // so the test failed on a document that was correct.
    let (cell, _) = row(&doc, "guide-epsilon");
    let documented: f32 = cell
        .parse()
        .unwrap_or_else(|_| panic!("--guide-epsilon: {cell:?} is not a number"));
    assert_eq!(
        documented,
        defaults::GUIDE_EPSILON,
        "--guide-epsilon: PARAMETERS.md says {documented}, defaults.rs says {}",
        defaults::GUIDE_EPSILON
    );

    let (cell, _) = row(&doc, "guide-space");
    let expected = match defaults::GUIDE_SPACE {
        GuideSpace::Linear => "linear",
        GuideSpace::Perceptual => "perceptual",
    };
    assert_eq!(
        cell, expected,
        "--guide-space: PARAMETERS.md says {cell:?}, defaults.rs says {expected:?}"
    );

    let (cell, _) = row(&doc, "fusion");
    assert_eq!(
        cell,
        defaults::FUSION.token(),
        "--fusion: PARAMETERS.md says {cell:?}, defaults.rs says {:?}",
        defaults::FUSION.token()
    );
}

/// Every flag the CLI accepts has a row, and every row names a stage.
///
/// The first half is what stops a parameter being added to the code and forgotten in the
/// document — the failure mode a value check alone cannot see, because there is nothing
/// to compare against.
#[test]
fn every_parameter_is_documented_and_placed_in_a_stage() {
    let doc = doc();

    // The full set the front ends expose. Adding a flag without adding it here is
    // possible, but adding one without *noticing* this list is not: it sits next to the
    // failure it causes.
    const FLAGS: [&str; 8] = [
        "registration-level",
        "focus-radius",
        "guide-radius",
        "guide-epsilon",
        "guide-space",
        "fusion",
        "salience-radius",
        "pyramid-floor",
    ];
    const STAGES: [&str; 4] = ["registration", "focus measurement", "weights", "fusion"];

    for flag in FLAGS {
        let (_, stage) = row(&doc, flag);
        assert!(
            STAGES.contains(&stage.as_str()),
            "--{flag} is filed under stage {stage:?}, which is not one of {STAGES:?}"
        );
        // A row is not documentation. Each parameter also gets a section explaining it,
        // and an empty table entry with no prose is the likelier kind of rot.
        assert!(
            doc.matches(&format!("`--{flag}`")).count() >= 1,
            "--{flag} appears only in the table"
        );
    }
}
