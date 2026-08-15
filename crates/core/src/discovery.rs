//! Finding the frames of a stack on disk, and checking they agree on geometry.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::image::FrameInfo;
use crate::tiff_io::probe;

/// TIFFs that live alongside a stack's frames but are not frames.
///
/// `test-data/synthetic_50/` ships its ground truth in the same directory as the
/// 50 frames; without this it would be stacked as a 51st frame.
/// `stackaroni_fused` is our own output: writing a result next to the frames it was
/// made from would otherwise feed it back in as an extra frame on the next run.
///
/// **This list is a backstop, not the defence.** It only recognises names it already
/// knows, and the case that actually happened went straight past it: fused results saved
/// as `blossom_stacked_local.tif` and `ruler_stacked_local.tif` into their own stack
/// directories, silently stacked as 101st frames on both real stacks, with the only
/// symptom a frame count in one line of CLI output. Enumerating more names does not fix
/// that — [`ensure_output_outside_stack`] does, by refusing to create the file. Extend
/// this list only for files a *third party* leaves beside the frames, the way
/// `reference_pmax` arrives.
const NON_FRAME_STEMS: &[&str] = &[
    "ground_truth_all_in_focus",
    "depth_map",
    "stackaroni_fused",
    "reference_pmax",
];

/// Refuse to write a fused result into the directory it was stacked from.
///
/// A fused frame has the same geometry and bit depth as its sources, so nothing
/// downstream can tell it apart: [`discover_stack`]'s geometry check accepts it, it
/// registers, it fuses, and the corruption is invisible in the output image. That makes
/// it exactly the kind of mistake that survives a rating and silently invalidates an
/// entry in `docs/eval-log.md`.
///
/// Compares canonical paths rather than strings, so `../blossom/out.tif`, a trailing
/// slash and a symlinked directory are all caught. An output whose parent cannot be
/// canonicalized does not exist yet and therefore is not the stack directory, which
/// does — so that case passes.
pub fn ensure_output_outside_stack(output: &Path, stack_dir: &Path) -> Result<()> {
    let parent = match output.parent() {
        // A bare filename means the current directory.
        Some(p) if p.as_os_str().is_empty() => Path::new("."),
        Some(p) => p,
        None => return Ok(()),
    };
    let (Ok(parent), Ok(dir)) = (parent.canonicalize(), stack_dir.canonicalize()) else {
        return Ok(());
    };
    if parent == dir {
        return Err(Error::OutputInsideStack {
            output: output.to_path_buf(),
            dir: stack_dir.to_path_buf(),
        });
    }
    Ok(())
}

/// Compare two filenames treating runs of digits as numbers.
///
/// `frame_2` before `frame_10`, and `frame_002` before `frame_010`, from the same rule.
/// Non-digit stretches compare byte-wise, which is what the previous plain sort did for
/// the whole name.
///
/// Leading zeros are stripped before comparing, then longer digit runs are the larger
/// number and equal-length runs compare byte-wise. That avoids parsing into an integer,
/// so a filename carrying a forty-digit run cannot overflow anything — it just sorts.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (mut i, mut j) = (0usize, 0usize);

    while i < a.len() && j < b.len() {
        if a[i].is_ascii_digit() && b[j].is_ascii_digit() {
            let run = |s: &[u8], from: usize| {
                let mut to = from;
                while to < s.len() && s[to].is_ascii_digit() {
                    to += 1;
                }
                to
            };
            let (ai, bj) = (run(a, i), run(b, j));
            fn strip(s: &[u8]) -> &[u8] {
                let mut k = 0;
                while k + 1 < s.len() && s[k] == b'0' {
                    k += 1;
                }
                &s[k..]
            }
            let (na, nb) = (strip(&a[i..ai]), strip(&b[j..bj]));

            let ordering = na.len().cmp(&nb.len()).then_with(|| na.cmp(nb));
            if ordering != Ordering::Equal {
                return ordering;
            }
            (i, j) = (ai, bj);
        } else {
            let ordering = a[i].cmp(&b[j]);
            if ordering != Ordering::Equal {
                return ordering;
            }
            i += 1;
            j += 1;
        }
    }
    // One is a prefix of the other, or they matched apart from zero padding.
    (a.len() - i).cmp(&(b.len() - j))
}

/// A directory of frames, in stacking order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stack {
    pub name: String,
    pub dir: PathBuf,
    pub frames: Vec<PathBuf>,
}

/// Per-frame geometry plus the geometry they all share.
#[derive(Debug, Clone)]
pub struct StackProbe {
    pub frames: Vec<(PathBuf, FrameInfo)>,
    pub info: FrameInfo,
}

/// Collect the frames in one stack directory.
///
/// Sorted in *natural* order — digit runs compared as numbers, everything else by
/// bytes — so `frame_2` precedes `frame_10` whether or not the numbering is padded.
/// Naming is otherwise not assumed to be consistent between stacks: `ruler/` uses
/// `A1_00001_01.tif` where `blossom/` uses `A1_00001.tif`.
///
/// **This order is the focus order**, which is why it is worth more than tidiness.
/// Registration chains outward from the middle frame on the assumption that adjacent
/// files are adjacent focus positions; a stack in the wrong order does not fail, it
/// produces a badly aligned image. Plain lexicographic sorting gave that outcome for any
/// unpadded stack — `frame_10` before `frame_2` — silently.
pub fn discover_stack(dir: &Path) -> Result<Stack> {
    let mut frames = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;

    for entry in entries {
        let path = entry.map_err(|e| Error::io(dir, e))?.path();
        if is_frame(&path) {
            frames.push(path);
        }
    }
    frames.sort_by(|a, b| {
        let key = |p: &Path| {
            p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_owned()
        };
        // The full path breaks ties, so `frame_1` and `frame_001` — numerically equal —
        // still land in a stable, repeatable order rather than whichever the filesystem
        // happened to hand back.
        natural_cmp(&key(a), &key(b)).then_with(|| a.cmp(b))
    });

    if frames.is_empty() {
        return Err(Error::NoFrames {
            dir: dir.to_path_buf(),
        });
    }

    Ok(Stack {
        name: dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.display().to_string()),
        dir: dir.to_path_buf(),
        frames,
    })
}

/// Collect every stack directory under `root`, sorted by name.
pub fn discover_test_set(root: &Path) -> Result<Vec<Stack>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| Error::io(root, e))?
        .map(|e| e.map(|e| e.path()).map_err(|e| Error::io(root, e)))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();

    // A directory with no TIFFs isn't a stack; skip it rather than failing the run.
    let stacks: Vec<Stack> = dirs.iter().filter_map(|d| discover_stack(d).ok()).collect();

    if stacks.is_empty() {
        return Err(Error::NoStacks {
            root: root.to_path_buf(),
        });
    }
    Ok(stacks)
}

impl Stack {
    /// Read every frame's geometry and check the stack is internally consistent.
    ///
    /// Header reads only — no pixel data is decoded.
    pub fn probe(&self) -> Result<StackProbe> {
        let mut frames = Vec::with_capacity(self.frames.len());
        for path in &self.frames {
            let info = probe(path)?;
            frames.push((path.clone(), info));
        }

        let info = frames[0].1;
        for (path, other) in &frames[1..] {
            if *other != info {
                return Err(Error::Geometry {
                    path: path.clone(),
                    width: other.width,
                    height: other.height,
                    bits: other.bits_per_sample,
                    reference: frames[0].0.clone(),
                    ref_width: info.width,
                    ref_height: info.height,
                    ref_bits: info.bits_per_sample,
                });
            }
        }

        Ok(StackProbe { frames, info })
    }
}

fn is_frame(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    // Skips dotfiles and macOS `._` AppleDouble sidecars.
    if name.starts_with('.') {
        return false;
    }
    let is_tiff = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();

    is_tiff && !NON_FRAME_STEMS.contains(&stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"").unwrap();
    }

    #[test]
    fn finds_and_sorts_frames() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["f_003.tiff", "f_001.tiff", "f_002.tif"] {
            touch(dir.path(), name);
        }

        let stack = discover_stack(dir.path()).unwrap();
        let names: Vec<_> = stack
            .frames
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["f_001.tiff", "f_002.tif", "f_003.tiff"]);
    }

    /// The exact mistake this guard exists for, pinned: a fused result saved beside the
    /// frames it came from. Named after the file that actually did it.
    #[test]
    fn refuses_an_output_inside_the_stack_directory() {
        let dir = tempfile::tempdir().unwrap();
        let stack = dir.path();
        let output = stack.join("blossom_stacked_local.tif");
        assert!(matches!(
            ensure_output_outside_stack(&output, stack),
            Err(Error::OutputInsideStack { .. })
        ));
    }

    /// String comparison would miss both of these; canonicalization catches them.
    #[test]
    fn refuses_an_output_reaching_the_stack_by_a_roundabout_path() {
        let dir = tempfile::tempdir().unwrap();
        let stack = dir.path().join("blossom");
        std::fs::create_dir(&stack).unwrap();

        for output in [
            stack.join("../blossom/out.tif"),
            dir.path().join("./blossom/./out.tif"),
        ] {
            assert!(
                matches!(
                    ensure_output_outside_stack(&output, &stack),
                    Err(Error::OutputInsideStack { .. })
                ),
                "should have been refused: {}",
                output.display()
            );
        }
    }

    #[test]
    fn allows_an_output_elsewhere() {
        let dir = tempfile::tempdir().unwrap();
        let stack = dir.path().join("blossom");
        std::fs::create_dir(&stack).unwrap();

        // A sibling directory, a parent, and a path whose directory does not exist yet —
        // the last cannot be the stack directory, because that one does exist.
        for output in [
            dir.path().join("out/blossom.tif"),
            dir.path().join("blossom.tif"),
            dir.path().join("not/created/yet/blossom.tif"),
        ] {
            assert!(
                ensure_output_outside_stack(&output, &stack).is_ok(),
                "should have been allowed: {}",
                output.display()
            );
        }
    }

    /// The case the old sort got wrong: unpadded numbering.
    #[test]
    fn natural_order_puts_frame_2_before_frame_10() {
        let mut names = vec![
            "frame_10.tif",
            "frame_1.tif",
            "frame_100.tif",
            "frame_2.tif",
        ];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            names,
            [
                "frame_1.tif",
                "frame_2.tif",
                "frame_10.tif",
                "frame_100.tif"
            ]
        );
        // The plain sort this replaced would have produced the wrong answer, which is
        // what makes the change worth having.
        let mut lexicographic = names.clone();
        lexicographic.sort();
        assert_ne!(lexicographic, names);
    }

    /// **Padded stacks must sort exactly as before.** Every rating in `docs/eval-log.md`
    /// was given to output produced from these orderings, and `output_is_stable` hashes
    /// one of them — so a natural sort that reordered a padded stack would silently
    /// invalidate the lot.
    #[test]
    fn padded_stacks_sort_identically_to_the_old_lexicographic_order() {
        for pattern in ["A1_{:05}.tif", "A1_{:05}_01.tif", "frame_{:03}.tiff"] {
            let names: Vec<String> = (1..=120)
                .map(|i| {
                    pattern
                        .replace("{:05}", &format!("{i:05}"))
                        .replace("{:03}", &format!("{i:03}"))
                })
                .collect();

            let mut natural = names.clone();
            natural.sort_by(|a, b| natural_cmp(a, b));
            let mut lexicographic = names.clone();
            lexicographic.sort();

            assert_eq!(natural, lexicographic, "pattern {pattern}");
            assert_eq!(
                natural, names,
                "pattern {pattern} should already be in order"
            );
        }
    }

    /// Mixed padding is numerically ambiguous; it must still be a total order rather
    /// than whatever the filesystem returned, or two runs of the same folder could
    /// stack in different orders.
    #[test]
    fn equal_numbers_with_different_padding_order_deterministically() {
        use std::cmp::Ordering;
        assert_eq!(natural_cmp("frame_1.tif", "frame_001.tif"), Ordering::Equal);

        let dir = tempfile::tempdir().unwrap();
        for name in ["f_2.tif", "f_002.tif", "f_10.tif"] {
            write_tiff(&dir.path().join(name), 4, 4);
        }
        let first = discover_stack(dir.path()).unwrap().frames;
        let second = discover_stack(dir.path()).unwrap().frames;
        assert_eq!(first, second, "repeated discovery must agree");
        assert!(
            first.last().unwrap().ends_with("f_10.tif"),
            "10 sorts last whatever the padding: {first:?}"
        );
    }

    /// Names with no digits at all fall back to the byte comparison the old sort used.
    #[test]
    fn names_without_digits_keep_byte_order() {
        let mut names = vec!["zulu.tif", "alpha.tif", "Mike.tif"];
        names.sort_by(|a, b| natural_cmp(a, b));
        let mut expected = names.clone();
        expected.sort();
        assert_eq!(names, expected);
    }

    /// A digit run longer than any integer type must sort rather than overflow.
    #[test]
    fn absurdly_long_digit_runs_do_not_overflow() {
        use std::cmp::Ordering;
        let long = format!("f_{}.tif", "9".repeat(40));
        let longer = format!("f_{}.tif", "9".repeat(41));
        assert_eq!(natural_cmp(&long, &longer), Ordering::Less);
    }

    #[test]
    fn excludes_non_frames() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "frame_001.tiff",
            "ground_truth_all_in_focus.tiff",
            "depth_map.tiff",
            "_contact_sheet_preview.png",
            "README.md",
            ".DS_Store",
            "._frame_001.tiff",
            // Our own output and a third-party reference render, both of which
            // legitimately live beside the frames they were made from.
            "stackaroni_fused.tif",
            "reference_pmax.tif",
        ] {
            touch(dir.path(), name);
        }

        let stack = discover_stack(dir.path()).unwrap();
        assert_eq!(stack.frames.len(), 1);
        assert!(stack.frames[0].ends_with("frame_001.tiff"));
    }

    fn write_tiff(path: &Path, width: u32, height: u32) {
        let info = FrameInfo {
            width,
            height,
            samples: 3,
            bits_per_sample: 16,
        };
        crate::tiff_io::write_rgb16_srgb(path, info, |_, row| {
            row.fill(0.5);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn probe_accepts_consistent_frames() {
        let dir = tempfile::tempdir().unwrap();
        write_tiff(&dir.path().join("f_001.tif"), 8, 4);
        write_tiff(&dir.path().join("f_002.tif"), 8, 4);

        let probe = discover_stack(dir.path()).unwrap().probe().unwrap();
        assert_eq!(probe.frames.len(), 2);
        assert_eq!((probe.info.width, probe.info.height), (8, 4));
    }

    #[test]
    fn probe_rejects_mismatched_frames() {
        let dir = tempfile::tempdir().unwrap();
        write_tiff(&dir.path().join("f_001.tif"), 8, 4);
        write_tiff(&dir.path().join("f_002.tif"), 8, 5);

        let err = discover_stack(dir.path())
            .unwrap()
            .probe()
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("f_002.tif"),
            "should name the odd frame: {err}"
        );
    }

    #[test]
    fn errors_on_directory_without_frames() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "notes.txt");
        assert!(discover_stack(dir.path()).is_err());
    }

    #[test]
    fn test_set_skips_non_stack_directories() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("stack_b")).unwrap();
        std::fs::create_dir(root.path().join("stack_a")).unwrap();
        std::fs::create_dir(root.path().join("empty")).unwrap();
        touch(&root.path().join("stack_a"), "f1.tif");
        touch(&root.path().join("stack_b"), "f1.tif");
        touch(root.path(), "README.md");

        let stacks = discover_test_set(root.path()).unwrap();
        let names: Vec<_> = stacks.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["stack_a", "stack_b"]);
    }
}
