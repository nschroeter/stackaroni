//! The four replaceable pipeline stages and the types flowing between them.
//!
//! Trait signatures are as sketched in `CLAUDE.md` and `docs/algorithms.md` §14.
//! None of the types owns pixel data: `Image` reads rows from its TIFF on demand and
//! `FocusMap`/`WeightMaps` are mmapped scratch planes, so `&[Image]` over a 100-frame
//! stack costs handles, not 60 GB. Streaming lives inside each implementation.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::Result;
use crate::image::{FrameInfo, ScratchPlane};
use crate::tiff_io::FrameReader;

/// Per-pixel focus quality for one frame, single-channel.
pub type FocusMap = ScratchPlane;

/// One weight plane per frame; index `k` is frame `k`'s contribution.
pub type WeightMaps = Vec<ScratchPlane>;

/// Geometric correction mapping a frame onto the reference frame.
///
/// A similarity: uniform scale about the image centre, then translation.
/// Coordinates are centre-relative, so `frame_point = scale * anchor_point + (dx, dy)`.
///
/// Scale is here because translation alone measurably could not represent this data.
/// The `registration_accuracy` benchmark found opposite halves of a single adjacent
/// `ruler` pair reporting +3.3 px and −2.98 px — uniform magnification of ~0.1% per
/// frame, as `docs/algorithms.md` §10 predicts for focus breathing. Estimating scale
/// by log-polar phase correlation (Reddy & Chatterji, 1996) drops that per-region
/// spread from 6.30 px to 0.92 px, which is the estimator's own noise floor.
///
/// Rotation is deliberately absent. The same log-polar correlation measures it, and
/// it comes back under 0.13° on every real pair tested — so it is reported as
/// evidence in [`crate::registration::SimilarityEstimate`] rather than modelled. If
/// it ever stops being ~0, that is the signal to escalate to ECC affine
/// (Evangelidis & Psarakis, 2008), not before.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Uniform magnification, reference to frame. 1.0 is no change.
    pub scale: f32,
    /// Horizontal shift in pixels, reference to frame.
    pub dx: f32,
    /// Vertical shift in pixels, reference to frame.
    pub dy: f32,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        dx: 0.0,
        dy: 0.0,
    };

    pub fn translation(dx: f32, dy: f32) -> Self {
        Self { scale: 1.0, dx, dy }
    }

    /// Map a point given in centre-relative coordinates.
    pub fn apply(self, x: f32, y: f32) -> (f32, f32) {
        (self.scale * x + self.dx, self.scale * y + self.dy)
    }

    /// This transform followed by `next`.
    ///
    /// `next(self(p)) = s_next·(s_self·p + t_self) + t_next`, so the scales multiply
    /// and the leading translation picks up the trailing scale. This is what chains
    /// single-step alignments outward from the anchor.
    pub fn then(self, next: Self) -> Self {
        Self {
            scale: self.scale * next.scale,
            dx: next.scale * self.dx + next.dx,
            dy: next.scale * self.dy + next.dy,
        }
    }

    /// The transform undoing this one.
    pub fn inverse(self) -> Self {
        let inv = 1.0 / self.scale;
        Self {
            scale: inv,
            dx: -self.dx * inv,
            dy: -self.dy * inv,
        }
    }
}

/// A frame on disk, read a band at a time.
///
/// Rows are pulled through `&self` because the stage traits take `&Image`; the
/// decoder and its strip cache sit behind a `Mutex` so handles stay `Sync` for the
/// parallelism `docs/algorithms.md` §15 calls for.
pub struct Image {
    path: PathBuf,
    info: FrameInfo,
    reader: Mutex<FrameReader>,
}

impl Image {
    pub fn open(path: &Path) -> Result<Self> {
        let reader = FrameReader::open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            info: reader.info(),
            reader: Mutex::new(reader),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn info(&self) -> FrameInfo {
        self.info
    }

    /// Fill `out` with `count` rows of linear-light RGB starting at row `y0`.
    pub fn read_rows(&self, y0: u32, count: u32, out: &mut [f32]) -> Result<()> {
        let mut reader = self.reader.lock().expect("frame reader mutex poisoned");
        reader.read_rows(y0, count, out)
    }
}

// Every stage below touches the disk — frames are read a band at a time and stage
// outputs are mmapped scratch planes — so all four return `Result`, unlike the
// original sketch in `docs/algorithms.md` §14. A mid-run failure on frame 47 of 100
// has to name the frame, not abort the process.

/// Estimate the geometric correction aligning `target` onto `reference`.
pub trait Registration {
    fn align(&self, reference: &Image, target: &Image) -> Result<Transform>;
}

/// Measure per-pixel focus quality across one frame.
pub trait FocusMetric {
    fn evaluate(&self, image: &Image) -> Result<FocusMap>;
}

/// Turn per-frame focus maps into per-frame blending weights.
pub trait WeightEstimator {
    fn weights(&self, focus_maps: &[FocusMap]) -> Result<WeightMaps>;
}

/// Blend the frames together under the given weights.
///
/// The output path is supplied when the implementation is constructed rather than
/// through this method, the same constructor-injection pattern the guided-filter
/// weight estimator uses for its guide images.
pub trait ImageFusion {
    fn fuse(&self, images: &[Image], weights: &WeightMaps) -> Result<Image>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiff_io::write_rgb16_srgb;

    #[test]
    fn image_reads_rows_through_shared_reference() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.tif");
        let info = FrameInfo {
            width: 4,
            height: 8,
            samples: 3,
            bits_per_sample: 16,
        };
        write_rgb16_srgb(&path, info, |y, row| {
            row.fill(y as f32 / 100.0);
            Ok(())
        })
        .unwrap();

        let image = Image::open(&path).unwrap();
        assert_eq!(image.info(), info);
        assert_eq!(image.path(), path);

        // `&self`, not `&mut self` — this is what the stage traits require.
        let read_band = |img: &Image, y0| {
            let mut band = vec![0f32; info.row_len()];
            img.read_rows(y0, 1, &mut band).unwrap();
            band[0]
        };
        assert!((read_band(&image, 3) - 0.03).abs() < 1e-3);
        assert!((read_band(&image, 7) - 0.07).abs() < 1e-3);
    }

    #[test]
    fn identity_transform_is_zero() {
        assert_eq!(Transform::IDENTITY, Transform::default());
    }
}
