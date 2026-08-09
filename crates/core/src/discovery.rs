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
const NON_FRAME_STEMS: &[&str] = &[
    "ground_truth_all_in_focus",
    "depth_map",
    "stackaroni_fused",
    "reference_pmax",
];

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
/// Frames are sorted lexicographically, which is also numeric order here because
/// every stack uses zero-padded indices. Naming is otherwise not assumed to be
/// consistent between stacks — `ruler/` uses `A1_00001_01.tif` where `blossom/`
/// uses `A1_00001.tif`.
pub fn discover_stack(dir: &Path) -> Result<Stack> {
    let mut frames = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;

    for entry in entries {
        let path = entry.map_err(|e| Error::io(dir, e))?.path();
        if is_frame(&path) {
            frames.push(path);
        }
    }
    frames.sort();

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
