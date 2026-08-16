//! `CHANGELOG.md` must have a section for the version being built.
//!
//! The release workflow turns that section into the GitHub release notes and fails the
//! run if it is missing — but that failure arrives at `git push origin v1.2.3`, with the
//! tag already public and the archives already built. Catching it in `cargo test` moves
//! the same failure to the commit that raised the version, which is where it can still be
//! fixed by editing a file rather than by deleting a tag.
//!
//! Same shape and same reasoning as `parameters_doc.rs`: prose cannot share a constant
//! with code, so the drift is caught here instead.
//!
//! Needs no `test-data/`, so it runs in CI on every push.

/// The heading the release workflow's `awk` looks for, with the same spelling.
fn heading(version: &str) -> String {
    format!("## [{version}]")
}

fn changelog() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../CHANGELOG.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// The workspace version has a section, and it is not empty.
#[test]
fn the_current_version_has_a_changelog_section() {
    let version = env!("CARGO_PKG_VERSION");
    let changelog = changelog();
    let heading = heading(version);

    let start = changelog.find(&heading).unwrap_or_else(|| {
        panic!(
            "CHANGELOG.md has no `{heading}` section.\n\
             The version in Cargo.toml is {version}; add a section for it before releasing."
        )
    });

    // A heading with nothing under it would satisfy a `contains` check and then produce
    // empty release notes, which is the failure this test exists to prevent.
    let body: String = changelog[start + heading.len()..]
        .lines()
        .skip(1)
        .take_while(|line| !line.starts_with("## ["))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !body.trim().is_empty(),
        "`{heading}` in CHANGELOG.md has no content under it"
    );
}

/// Versions appear once. Two sections for one version means the workflow's `awk` takes
/// the first and silently drops whatever was written in the second.
#[test]
fn no_version_is_sectioned_twice() {
    let changelog = changelog();
    let mut headings: Vec<&str> = changelog
        .lines()
        .filter(|line| line.starts_with("## ["))
        .map(|line| line.split_whitespace().nth(1).unwrap_or(line))
        .collect();

    let before = headings.len();
    headings.sort_unstable();
    headings.dedup();
    assert_eq!(
        before,
        headings.len(),
        "CHANGELOG.md has more than one section for the same version"
    );
}
