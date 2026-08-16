//! Fusion must not accumulate frame caches across the stack.
//!
//! **The regression this exists for cost 15 GB and no wrong pixels.** Fusion's caller
//! owns every `Image` for the whole stage, the per-frame loop reads each one exactly
//! once, and a `FrameReader` keeps its decoded strips for its own lifetime — so the
//! caches of every frame already fused stayed resident until the stage ended. On
//! single-strip input one cached strip is one whole frame, which made peak memory scale
//! with stack depth: 33 frames peaked at 25.0 GB of footprint against 10.1 GB once the
//! loop released each frame as it finished.
//!
//! Nothing about the output changes when this breaks, so no rating and no hash gate can
//! catch it. It has to be asserted directly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use stackaroni_core::fusion::{LaplacianPyramidFusion, SelectionFusion};
use stackaroni_core::image::{FrameInfo, ScratchPlane};
use stackaroni_core::pipeline::{Image, ImageFusion, Transform, WeightMaps};
use stackaroni_core::tiff_io::write_rgb16_srgb;

const FRAMES: usize = 5;
const PYRAMID_FLOOR: u32 = 8;
const SALIENCE_RADIUS: u32 = 2;

/// Frames are 40 rows, under `write_rgb16_srgb`'s 64-row strip size, so each lands in a
/// single strip — the layout that makes a cached strip a whole frame.
const INFO: FrameInfo = FrameInfo {
    width: 48,
    height: 40,
    samples: 3,
    bits_per_sample: 16,
};

fn write_stack(dir: &Path) -> Vec<PathBuf> {
    (0..FRAMES)
        .map(|k| {
            let path = dir.join(format!("frame{k}.tif"));
            write_rgb16_srgb(&path, INFO, |y, row| {
                for (i, slot) in row.iter_mut().enumerate() {
                    let x = i / 3;
                    // Something with structure, so the salience rule has a decision to
                    // make rather than a flat field.
                    *slot = ((x + y as usize + k * 7) % 16) as f32 / 16.0;
                }
                Ok(())
            })
            .unwrap();
            path
        })
        .collect()
}

fn flat_weights(dir: &Path) -> WeightMaps {
    (0..FRAMES)
        .map(|k| {
            let mut plane =
                ScratchPlane::create(&dir.join(format!("w{k}.f32")), INFO.width, INFO.height)
                    .unwrap();
            plane
                .rows_mut(0, INFO.height)
                .unwrap()
                .fill(1.0 / FRAMES as f32);
            plane
        })
        .collect()
}

fn held_bytes(images: &[Image]) -> usize {
    images.iter().map(|i| i.cache_bytes()).sum()
}

/// Both fusion rules leave nothing cached, and the check is not vacuous.
///
/// The `assert!(> 0)` before each run matters as much as the zero after it: without it
/// the test would keep passing if `cache_bytes` ever started reporting nothing, or if
/// the fusion loop stopped reading the frames at all.
#[test]
fn fusing_leaves_no_frame_cached() {
    let dir = tempfile::tempdir().unwrap();
    let frames = write_stack(dir.path());
    let transforms: HashMap<PathBuf, Transform> = frames
        .iter()
        .cloned()
        .map(|p| (p, Transform::IDENTITY))
        .collect();

    let rules: Vec<(&str, Box<dyn ImageFusion>)> = vec![
        (
            "blend",
            Box::new(LaplacianPyramidFusion::new(
                &dir.path().join("blend.tif"),
                transforms.clone(),
                PYRAMID_FLOOR,
            )),
        ),
        (
            "select",
            Box::new(SelectionFusion::new(
                &dir.path().join("select.tif"),
                transforms.clone(),
                PYRAMID_FLOOR,
                SALIENCE_RADIUS,
            )),
        ),
    ];

    for (name, fusion) in rules {
        let images: Vec<Image> = frames.iter().map(|p| Image::open(p).unwrap()).collect();
        let weights = flat_weights(dir.path());

        // Warm every reader, so "nothing is held" cannot pass by nothing being read.
        let mut band = vec![0f32; INFO.row_len()];
        for image in &images {
            image.read_rows(0, 1, &mut band).unwrap();
        }
        let warm = held_bytes(&images);
        assert!(warm > 0, "{name}: readers hold nothing before fusing");

        fusion.fuse(&images, &weights, &()).unwrap();

        assert_eq!(
            held_bytes(&images),
            0,
            "{name}: fusion left frame caches resident — peak memory now scales with \
             stack depth, which is exactly the 15 GB regression this guards"
        );
    }
}

/// Releasing is a memory operation: the same rows come back afterwards.
#[test]
fn a_released_frame_still_reads_the_same_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let frames = write_stack(dir.path());
    let image = Image::open(&frames[0]).unwrap();

    let mut before = vec![0f32; INFO.row_len() * 4];
    image.read_rows(8, 4, &mut before).unwrap();
    assert!(image.cache_bytes() > 0);

    image.release_cache();
    assert_eq!(image.cache_bytes(), 0);

    let mut after = vec![0f32; INFO.row_len() * 4];
    image.read_rows(8, 4, &mut after).unwrap();
    assert_eq!(before, after);
}
