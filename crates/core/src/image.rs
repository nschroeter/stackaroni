//! Frame geometry, sRGB transfer functions, and the disk-backed plane that stage
//! outputs are stored in.

use std::fs::OpenOptions;
use std::path::Path;

use memmap2::MmapMut;

use crate::error::{Error, Result};

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

#[cfg(test)]
mod tests {
    use super::*;

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
