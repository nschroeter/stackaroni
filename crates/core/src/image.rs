//! Frame geometry, sRGB transfer functions, and the disk-backed plane that stage
//! outputs are stored in.

use std::fs::OpenOptions;
use std::path::Path;

use memmap2::MmapMut;

use crate::error::{Error, Result};
use crate::pipeline::Transform;

/// Output rows computed per banded pass. Keeps the working buffers to a few MB on a
/// 50 MP frame while amortizing any halo re-read.
pub const BAND_ROWS: u32 = 256;

/// Sampling margin, in pixels, that bilinear interpolation needs at the far edge.
///
/// [`warp_frame`](crate::fusion::warp_frame) reads `get(ix + 1, iy + 1)`, and at
/// `size - 1` that second tap clamps back onto the first — but it is also weighted by
/// `fx`, which is exactly zero there, so the sample is still correct. The bound is
/// therefore `size - 1` and not `size - 2`. The tighter-looking value would be wrong in
/// a way that hides: it would mark the last row and column uncovered for *every* frame
/// including the identity anchor, and a pixel no frame covers fuses to black.
const BILINEAR_MARGIN: f32 = 1.0;

/// Pixels trimmed from a covered region, once, at full resolution.
///
/// The pyramid's 5-tap binomial filter reaches two samples, so a coefficient right on
/// the boundary has already mixed in neighbours from outside it.
///
/// **Applied once here rather than per level, and that distinction is worth the
/// comment.** Eroding two coefficients at every level sounds equivalent and is not: two
/// coefficients at level `L` are `2 * 2^L` full-resolution pixels, so the trim compounds
/// geometrically and reaches ~128 px by level five. Measured on synthetic_50 — whose
/// registered range is 0.9994..1.0063, a true shortfall of about four pixels — that cost
/// a 120 px band of valid data to avoid contamination that is both attenuated by every
/// blur it passes through and, at coarse scales, barely different from the content it
/// would replace. Trimming once keeps the excluded band the size of the actual shortfall.
const EDGE_TRIM: u32 = 2;

/// The region of anchor coordinates a warped frame can fill without sampling outside
/// itself.
///
/// **Why a rectangle is enough.** [`Transform`] is a similarity — uniform scale plus
/// translation, no rotation — so the set of anchor points that map inside the frame is
/// an axis-aligned rectangle. That is what makes covered-region tracking cheap: a frame
/// needs four numbers, not a mask plane the size of the image.
///
/// Bounds are inclusive of `x0`/`y0` and exclusive of `x1`/`y1`, like a slice range. An
/// empty rectangle is represented by `x1 <= x0`, which callers get for free by iterating
/// the range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coverage {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

impl Coverage {
    /// The whole of a frame this size — what an untransformed frame covers.
    pub fn full(info: FrameInfo) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: info.width,
            y1: info.height,
        }
    }

    /// Where `transform` can be sampled from without leaving the frame.
    ///
    /// A source coordinate is `scale * (p - centre) + shift + centre`; requiring that to
    /// land in `0 ..= size - BILINEAR_MARGIN` and solving for `p` gives the interval
    /// below. Scale is positive by construction — it is a magnification — so the
    /// inequality does not flip. Rounding goes inward on both ends, so a partially
    /// covered pixel counts as uncovered.
    pub fn of(transform: Transform, info: FrameInfo) -> Self {
        let axis = |shift: f32, size: u32| {
            let size = size as f32;
            let centre = size / 2.0;
            let lo = centre + (-shift - centre) / transform.scale;
            let hi = centre + (size - BILINEAR_MARGIN - shift - centre) / transform.scale;
            let lo = lo.ceil().clamp(0.0, size) as u32;
            let hi = (hi.floor() + 1.0).clamp(0.0, size) as u32;
            (lo, hi.max(lo))
        };
        let (x0, x1) = axis(transform.dx, info.width);
        let (y0, y1) = axis(transform.dy, info.height);

        // Trimmed only where the frame actually falls short. A frame that reaches the
        // canvas edge is not contaminated there — `warp_frame` never had to clamp — so
        // trimming it would discard good pixels and, at the extreme, leave the identity
        // anchor not covering its own borders.
        let trim = |lo: u32, hi: u32, size: u32| {
            let lo = if lo > 0 { lo + EDGE_TRIM } else { 0 };
            let hi = if hi < size {
                hi.saturating_sub(EDGE_TRIM)
            } else {
                size
            };
            (lo, hi.max(lo))
        };
        let (x0, x1) = trim(x0, x1, info.width);
        let (y0, y1) = trim(y0, y1, info.height);
        Self { x0, y0, x1, y1 }
    }

    /// This region one pyramid level down, given the size of the level it is leaving.
    ///
    /// Rounds inward, so a partially covered edge coefficient counts as uncovered — with
    /// one exception: a region that reached the edge still reaches it after halving.
    /// Without that, `x1 = width` on an odd width would floor to one short of the next
    /// level's width and a fully covering frame would quietly stop counting as full,
    /// losing its last column at every level.
    ///
    /// No further trimming happens here — see [`EDGE_TRIM`], which is applied once at
    /// full resolution precisely so this does not compound.
    pub fn reduced(self, width: u32, height: u32) -> Self {
        let axis = |lo: u32, hi: u32, size: u32| {
            let lo = lo.div_ceil(2);
            let hi = if hi >= size { size.div_ceil(2) } else { hi / 2 };
            (lo, hi.max(lo))
        };
        let (x0, x1) = axis(self.x0, self.x1, width);
        let (y0, y1) = axis(self.y0, self.y1, height);
        Self { x0, y0, x1, y1 }
    }

    pub fn contains(self, x: u32, y: u32) -> bool {
        (self.x0..self.x1).contains(&x) && (self.y0..self.y1).contains(&y)
    }

    /// The region both cover. Over a whole stack this is where every frame contributed,
    /// which is the region that needs no renormalization.
    pub fn intersect(self, other: Self) -> Self {
        let (x0, y0) = (self.x0.max(other.x0), self.y0.max(other.y0));
        let (x1, y1) = (self.x1.min(other.x1), self.y1.min(other.y1));
        Self {
            x0,
            y0,
            x1: x1.max(x0),
            y1: y1.max(y0),
        }
    }

    /// Does this cover every pixel of a plane `width` by `height`?
    ///
    /// The fast path worth having: a frame that covers everything needs no per-pixel
    /// test at all, and in a well-registered stack most frames do.
    pub fn is_full(self, width: u32, height: u32) -> bool {
        self.x0 == 0 && self.y0 == 0 && self.x1 >= width && self.y1 >= height
    }
}

/// Shape and sample layout of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub samples: u16,
    pub bits_per_sample: u8,
}

impl FrameInfo {
    /// Samples in one row, i.e. `width * samples`.
    pub fn row_len(&self) -> usize {
        self.width as usize * self.samples as usize
    }
}

/// sRGB EOTF: encoded value in `[0,1]` to linear light.
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB EOTF: linear light to encoded value in `[0,1]`.
///
/// Applied before quantizing back to 16 bits — writing linear light into the TIFF
/// would make the file read far too dark in any normal viewer.
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Single-channel `f32` plane backed by a file on disk.
///
/// Backs `FocusMap` and `WeightMaps`. The OS pages rows in on demand, so holding a
/// slice over the whole plane does not mean the whole plane is resident — which is
/// what lets `&[FocusMap]` cover a 100-frame stack without 20 GB of RAM.
pub struct ScratchPlane {
    map: MmapMut,
    width: u32,
    height: u32,
}

impl ScratchPlane {
    /// Create (or truncate) a plane of `width * height` samples at `path`.
    pub fn create(path: &Path, width: u32, height: u32) -> Result<Self> {
        let len = width as u64 * height as u64 * size_of::<f32>() as u64;
        let scratch = |source| Error::Scratch {
            path: path.to_path_buf(),
            source,
        };

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(scratch)?;
        file.set_len(len).map_err(scratch)?;
        // SAFETY: we own the file for the lifetime of the map and no other process
        // writes it; the scratch directory is per-run.
        let map = unsafe { MmapMut::map_mut(&file) }.map_err(scratch)?;

        Ok(Self { map, width, height })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Read-only view of `count` rows starting at `y0`.
    pub fn rows(&self, y0: u32, count: u32) -> Result<&[f32]> {
        let (start, len) = self.span(y0, count)?;
        Ok(&bytemuck::cast_slice(&self.map)[start..start + len])
    }

    /// Writable view of `count` rows starting at `y0`.
    pub fn rows_mut(&mut self, y0: u32, count: u32) -> Result<&mut [f32]> {
        let (start, len) = self.span(y0, count)?;
        Ok(&mut bytemuck::cast_slice_mut(&mut self.map)[start..start + len])
    }

    /// Single sample, clamped to the plane's edges.
    pub fn at(&self, x: i64, y: i64) -> f32 {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        bytemuck::cast_slice::<u8, f32>(&self.map)[y * self.width as usize + x]
    }

    /// Bilinear sample, clamped at the edges.
    ///
    /// Edge clamping rather than zero-fill: a focus map warped with zero borders
    /// would read as "nothing in focus" along the frame edge and pull the weight
    /// map with it.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (ix, iy) = (x0 as i64, y0 as i64);

        let top = self.at(ix, iy) * (1.0 - fx) + self.at(ix + 1, iy) * fx;
        let bottom = self.at(ix, iy + 1) * (1.0 - fx) + self.at(ix + 1, iy + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    fn span(&self, y0: u32, count: u32) -> Result<(usize, usize)> {
        let end = y0 as u64 + count as u64;
        if end > self.height as u64 {
            return Err(Error::Bounds {
                start: y0 as u64,
                end,
                height: self.height,
            });
        }
        let w = self.width as usize;
        Ok((y0 as usize * w, count as usize * w))
    }
}

/// Resample `src` into `dst` under `transform`, which maps anchor coordinates onto
/// the frame's own — so each destination pixel reads straight through it.
pub fn warp_plane(
    src: &ScratchPlane,
    dst: &mut ScratchPlane,
    transform: Transform,
    info: FrameInfo,
) -> Result<()> {
    let (cx, cy) = (info.width as f32 / 2.0, info.height as f32 / 2.0);
    let mut y0 = 0;
    while y0 < info.height {
        let rows = BAND_ROWS.min(info.height - y0);
        let out = dst.rows_mut(y0, rows)?;
        for r in 0..rows {
            for x in 0..info.width {
                let (sx, sy) = transform.apply(x as f32 - cx, (y0 + r) as f32 - cy);
                out[r as usize * info.width as usize + x as usize] = src.sample(sx + cx, sy + cy);
            }
        }
        y0 += rows;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(width: u32, height: u32) -> FrameInfo {
        FrameInfo {
            width,
            height,
            samples: 3,
            bits_per_sample: 16,
        }
    }

    /// The anchor must cover every pixel, or fusion has positions no frame can fill.
    #[test]
    fn the_identity_covers_the_whole_frame() {
        let info = info(8664, 5784);
        let covered = Coverage::of(Transform::IDENTITY, info);
        assert_eq!(covered, Coverage::full(info));
        assert!(covered.is_full(info.width, info.height));
    }

    /// The bug this exists for: blossom's widest frame cannot fill the canvas.
    #[test]
    fn a_magnified_frame_covers_less_than_the_canvas() {
        let info = info(8664, 5784);
        let covered = Coverage::of(
            Transform {
                scale: 1.0569,
                dx: 0.0,
                dy: 0.0,
            },
            info,
        );
        assert!(!covered.is_full(info.width, info.height));
        // ~233 px in at each side, ~156 at top and bottom — the measured extent of the
        // streaking on blossom.
        assert!((230..=236).contains(&covered.x0), "{covered:?}");
        assert!((8428..=8434).contains(&covered.x1), "{covered:?}");
        assert!((153..=159).contains(&covered.y0), "{covered:?}");
    }

    /// A frame smaller than the canvas maps entirely inside it, so it covers fully.
    #[test]
    fn a_shrunk_frame_still_covers_everything() {
        let info = info(4000, 3000);
        let covered = Coverage::of(
            Transform {
                scale: 0.95,
                dx: 0.0,
                dy: 0.0,
            },
            info,
        );
        assert!(covered.is_full(info.width, info.height));
    }

    /// A shift leaves the trailing edge short by the shift, less the edge trim. The
    /// leading edge is untrimmed: the frame reaches it, so nothing was clamped there.
    #[test]
    fn translation_moves_the_covered_region() {
        let info = info(4000, 3000);
        let covered = Coverage::of(Transform::translation(100.0, 0.0), info);
        assert_eq!(covered.x0, 0);
        assert_eq!(covered.x1, info.width - 100 - EDGE_TRIM);
        assert!(covered.is_full(0, info.height), "untouched axis stays full");
    }

    #[test]
    fn reducing_halves_and_erodes() {
        let c = Coverage {
            x0: 200,
            y0: 100,
            x1: 8000,
            y1: 5000,
        };
        let r = c.reduced(8664, 5784);
        assert_eq!(r.x0, 100);
        assert_eq!(r.x1, 4000);
        assert_eq!(r.y0, 50);
    }

    /// The regression that made this constant worth measuring: an inset must stay the
    /// size of the real shortfall, not grow with depth. Trimming per level instead of
    /// once cost a ~120 px band on synthetic_50, whose actual shortfall is ~4 px.
    #[test]
    fn reducing_repeatedly_does_not_compound_the_inset() {
        let (mut w, mut h) = (8664u32, 5784u32);
        let mut c = Coverage {
            x0: 64,
            y0: 64,
            x1: w - 64,
            y1: h - 64,
        };
        for _ in 0..8 {
            c = c.reduced(w, h);
            (w, h) = (w.div_ceil(2), h.div_ceil(2));
        }
        // 64 >> 8 is 0, so a faithful halving lands at 0 — anything above a pixel or
        // two means the trim is accumulating.
        assert!(c.x0 <= 1, "inset grew down the pyramid: {c:?}");
        assert!(c.x1 >= w - 1, "far edge pulled in: {c:?} against width {w}");
    }

    /// A frame that covers everything must keep covering everything at every level,
    /// odd dimensions included — otherwise the anchor loses its last column down the
    /// pyramid and the margin it cannot fill grows out of nothing.
    #[test]
    fn full_coverage_survives_reduction_at_odd_sizes() {
        let (mut w, mut h) = (1201u32, 901u32);
        let mut c = Coverage {
            x0: 0,
            y0: 0,
            x1: w,
            y1: h,
        };
        for _ in 0..5 {
            c = c.reduced(w, h);
            (w, h) = (w.div_ceil(2), h.div_ceil(2));
            assert!(c.is_full(w, h), "lost fullness at {w}x{h}: {c:?}");
        }
    }

    #[test]
    fn intersecting_takes_the_tighter_bound() {
        let a = Coverage {
            x0: 10,
            y0: 0,
            x1: 100,
            y1: 100,
        };
        let b = Coverage {
            x0: 0,
            y0: 20,
            x1: 90,
            y1: 100,
        };
        assert_eq!(
            a.intersect(b),
            Coverage {
                x0: 10,
                y0: 20,
                x1: 90,
                y1: 100
            }
        );
    }

    #[test]
    fn a_disjoint_intersection_is_empty_not_inverted() {
        let a = Coverage {
            x0: 0,
            y0: 0,
            x1: 10,
            y1: 10,
        };
        let b = Coverage {
            x0: 50,
            y0: 50,
            x1: 60,
            y1: 60,
        };
        let i = a.intersect(b);
        assert!(i.x1 <= i.x0 && !i.contains(0, 0) && !i.contains(55, 55));
    }

    #[test]
    fn srgb_round_trips() {
        for &v in &[0.0, 0.002, 0.04045, 0.5, 1.0] {
            let back = linear_to_srgb(srgb_to_linear(v));
            assert!((back - v).abs() < 1e-6, "{v} -> {back}");
        }
    }

    #[test]
    fn srgb_to_linear_darkens_midtones() {
        // The whole point of the conversion: 0.5 encoded is ~0.214 linear, not 0.5.
        assert!((srgb_to_linear(0.5) - 0.2140).abs() < 1e-3);
    }

    #[test]
    fn scratch_plane_round_trips_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plane.f32");
        let mut plane = ScratchPlane::create(&path, 4, 3).unwrap();

        plane
            .rows_mut(1, 1)
            .unwrap()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(plane.rows(1, 1).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(plane.rows(0, 1).unwrap(), &[0.0; 4]);
        assert_eq!(plane.rows(0, 3).unwrap().len(), 12);
    }

    #[test]
    fn scratch_plane_rejects_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let mut plane = ScratchPlane::create(&dir.path().join("p.f32"), 4, 3).unwrap();
        assert!(matches!(plane.rows(2, 2), Err(Error::Bounds { .. })));
        assert!(matches!(plane.rows_mut(3, 1), Err(Error::Bounds { .. })));
        assert!(matches!(plane.rows(0, u32::MAX), Err(Error::Bounds { .. })));
    }
}
