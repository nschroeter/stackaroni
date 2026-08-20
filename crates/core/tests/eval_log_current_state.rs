//! The "Current state" block in `docs/eval-log.md` must agree with the code.
//!
//! That block is the answer to "what is true now", and the log's own header says to keep it
//! in sync with the rows. Nothing enforced that, and on 2026-08-20 an audit found nine stale
//! claims in it — including a paragraph describing `core::pipeline::Method` and a fifth
//! `StackFusion` trait as live, both deleted in T17, sitting below newer paragraphs
//! describing the same code correctly. The block had been appended to rather than pruned.
//!
//! **What this can check, and it is a narrow slice.** Two things in the block are also
//! written down in code, so the two copies can be compared: the live parameter set, and the
//! pinned output hash. Everything else there is prose about ratings, run times and removed
//! features, and no test can tell whether prose is true.
//!
//! **What it deliberately does not check.** Not the ratings — a human assigns those and
//! nothing in the tree knows them. Not the run times — 4.4 minutes went stale into 2 without
//! any constant changing. And not a deny-list of removed identifiers: the block now names
//! `Method` and `StackFusion` *on purpose*, in markers recording that they were removed, so
//! a test forbidding the words would fight the fix. Those failures need a reader.
//!
//! Needs no `test-data/`, so it runs on every push — the same reasoning as
//! `parameters_doc.rs`, whose shape this follows.

use stackaroni_core::defaults;
use stackaroni_core::weights::GuideSpace;

fn doc() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/eval-log.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// The "Current state" block: everything between its heading and the first table row.
///
/// Scoped rather than searched whole, because the rows below are append-only history and are
/// *supposed* to quote superseded values. A check that could not tell the two apart would
/// fail on correct history.
fn current_state(doc: &str) -> String {
    let start = doc
        .find("## Current state")
        .expect("docs/eval-log.md has no `## Current state` heading");
    let rest = &doc[start..];
    let end = rest
        .find("| Date | Commit |")
        .expect("docs/eval-log.md has no results table after `## Current state`");
    rest[..end].to_owned()
}

/// One paragraph of `block`, starting at `label`, with its line breaks flattened.
///
/// Flattened because the document is hard-wrapped and the labels do not respect it — the
/// live configuration ends "pyramid\nfloor 32 px", so matching "pyramid floor" against the
/// raw text finds nothing.
fn paragraph(block: &str, label: &str) -> String {
    let at = block
        .find(label)
        .unwrap_or_else(|| panic!("the Current state block has no {label:?} paragraph"));
    let rest = &block[at..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    rest[..end].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The word after `label`, stripped of the punctuation and markup around it.
///
/// Keeps `.` and `-` so `1e-4` survives; drops backticks, commas and semicolons so
/// "`select`," and "perceptual;" do not.
fn value_after(text: &str, label: &str) -> String {
    let at = text
        .find(label)
        .unwrap_or_else(|| panic!("live configuration does not mention {label:?}: {text:?}"));
    let rest = &text[at + label.len()..];
    let word = rest
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("nothing follows {label:?} in the live configuration"));
    word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.' && c != '-')
        .to_owned()
}

/// The parameters the block calls live are the parameters the pipeline runs with.
///
/// The block states them as prose rather than a table, so this reads them by label. That is
/// looser than `parameters_doc.rs`'s column parse and it is the right trade here: the
/// alternative is a table nobody would read in a document that is otherwise argument.
#[test]
fn the_live_configuration_is_the_shipped_one() {
    let doc = doc();
    let block = current_state(&doc);
    let live = paragraph(&block, "**Live configuration**");

    let numeric: [(&str, &str, u32); 5] = [
        (
            "salience radius",
            "SALIENCE_RADIUS",
            defaults::SALIENCE_RADIUS,
        ),
        (
            "guided filter radius",
            "GUIDE_RADIUS",
            defaults::GUIDE_RADIUS,
        ),
        ("focus radius", "FOCUS_RADIUS", defaults::FOCUS_RADIUS),
        (
            "at level",
            "REGISTRATION_LEVEL",
            defaults::REGISTRATION_LEVEL,
        ),
        ("pyramid floor", "PYRAMID_FLOOR", defaults::PYRAMID_FLOOR),
    ];
    for (label, konst, value) in numeric {
        let cell = value_after(&live, label);
        let documented: u32 = cell
            .parse()
            .unwrap_or_else(|_| panic!("eval-log says {label} {cell:?}, which is not a number"));
        assert_eq!(
            documented, value,
            "eval-log's Current state says {label} {documented}, defaults::{konst} is {value}"
        );
    }

    // Parsed at `f32`, the type the constant is: widening 1e-4 to `f64` first makes it
    // 9.999999747378752e-5, which is not equal to a `f64` parsed from "1e-4". The same trap
    // `parameters_doc.rs` documents, and the same fix.
    let cell = value_after(&live, "epsilon");
    let documented: f32 = cell
        .parse()
        .unwrap_or_else(|_| panic!("eval-log says epsilon {cell:?}, which is not a number"));
    assert_eq!(
        documented,
        defaults::GUIDE_EPSILON,
        "eval-log's Current state says epsilon {documented}, defaults::GUIDE_EPSILON is {}",
        defaults::GUIDE_EPSILON
    );

    let expected_space = match defaults::GUIDE_SPACE {
        GuideSpace::Linear => "linear",
        GuideSpace::Perceptual => "perceptual",
    };
    assert_eq!(
        value_after(&live, "guide space"),
        expected_space,
        "eval-log's Current state disagrees with defaults::GUIDE_SPACE"
    );

    assert_eq!(
        value_after(&live, "fusion rule"),
        defaults::FUSION.token(),
        "eval-log's Current state disagrees with defaults::FUSION"
    );
}

/// The hash the block quotes is the hash the gate pins.
///
/// This is the drift that actually happened: the block quoted `0x00455c66dd1e4c95` while the
/// gate had moved twice, to `0xee22e1aa7efbdf2f` and then to `0xc380280b4ed2051b`. The
/// constant lives in a sibling integration test, which cannot be imported, so it is read out
/// of the source. Parsing Rust with `find` is crude, but the failure is a loud panic rather
/// than a silent pass, which is the property that matters for a guard.
#[test]
fn the_quoted_output_hash_is_the_pinned_one() {
    let doc = doc();
    let block = current_state(&doc);

    let quoted = value_after(&block, "`output_is_stable` pins");
    let quoted = u64::from_str_radix(quoted.trim_start_matches("0x"), 16)
        .unwrap_or_else(|_| panic!("the hash quoted in Current state is not hex: {quoted:?}"));

    let gate = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/output_is_stable.rs");
    let source = std::fs::read_to_string(&gate)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", gate.display()));
    const DECL: &str = "const EXPECTED: u64 = ";
    let at = source
        .find(DECL)
        .expect("output_is_stable.rs no longer declares `const EXPECTED: u64 = `");
    let rest = &source[at + DECL.len()..];
    let literal: String = rest[..rest.find(';').expect("EXPECTED declaration has no `;`")]
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    let pinned = u64::from_str_radix(literal.trim_start_matches("0x"), 16)
        .unwrap_or_else(|_| panic!("cannot parse EXPECTED literal {literal:?}"));

    assert_eq!(
        quoted, pinned,
        "eval-log's Current state quotes {quoted:#018x}, output_is_stable.rs pins {pinned:#018x}"
    );
}
