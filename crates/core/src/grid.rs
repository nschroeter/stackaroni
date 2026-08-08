//! Single-channel `f32` images held in memory, used for registration and debug output.
//!
//! Unlike [`crate::pipeline::Image`], a `Grid` owns its pixels — so it is only ever
//! built at a resolution that fits comfortably. A full-resolution luma grid of a
//! 50 MP frame is 200 MB; downsampled levels are 4^level smaller.

use crate::error::Result;
use crate::pipeline::Image;

/// Rec. 709 luma coefficients, applied to linear-light RGB.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl Grid {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; (width as usize) * (height as usize)],
        }
    }

    pub fn at(&self, x: u32, y: u32) -> f32 {
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Build a luma grid from a frame, downsampled by `2^level`.
    ///
    /// Downsampling is area averaging over each `2^level` square block, which both
    /// decimates and low-pass filters in one pass — the anti-aliasing that makes a
    /// coarse level meaningful rather than aliased. (The Laplacian pyramid in the
    /// fusion stage uses the Burt & Adelson binomial kernel instead; this is the
    /// cheaper filter that suffices for correlation.)
    ///
    /// Only `2^level` source rows are resident at a time, so the frame itself is
    /// never fully materialized.
    pub fn from_image(image: &Image, level: u32) -> Result<Self> {
        let info = image.info();
        let scale = 1u32 << level;
        let (width, height) = (info.width / scale, info.height / scale);
        assert!(
            width > 0 && height > 0,
            "level {level} is coarser than the frame"
        );

        let row_len = info.row_len();
        let mut grid = Grid::new(width, height);
        let mut band = vec![0f32; row_len * scale as usize];
        let inv = 1.0 / (scale as f32 * scale as f32);

        for oy in 0..height {
            image.read_rows(oy * scale, scale, &mut band)?;
            let out = &mut grid.data[oy as usize * width as usize..][..width as usize];

            for sy in 0..scale as usize {
                let row = &band[sy * row_len..][..row_len];
                for (ox, slot) in out.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for sx in 0..scale as usize {
                        let p = (ox * scale as usize + sx) * 3;
                        acc += LUMA[0] * row[p] + LUMA[1] * row[p + 1] + LUMA[2] * row[p + 2];
                    }
                    *slot += acc;
                }
            }
            for v in out.iter_mut() {
                *v *= inv;
            }
        }
        Ok(grid)
    }

    /// A copy displaced by `(dx, dy)` pixels, sampled bilinearly.
    ///
    /// Used to synthesise known-shift test pairs and to render alignment overlays.
    /// Samples falling outside the source read as zero.
    pub fn shifted(&self, dx: f32, dy: f32) -> Grid {
        let mut out = Grid::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let sx = x as f32 - dx;
                let sy = y as f32 - dy;
                out.data[y as usize * self.width as usize + x as usize] = self.sample(sx, sy);
            }
        }
        out
    }

    /// Bilinear sample; zero outside the grid.
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;

        let get = |ix: f32, iy: f32| -> f32 {
            if ix < 0.0 || iy < 0.0 || ix >= self.width as f32 || iy >= self.height as f32 {
                0.0
            } else {
                self.at(ix as u32, iy as u32)
            }
        };

        let a = get(x0, y0) * (1.0 - fx) + get(x0 + 1.0, y0) * fx;
        let b = get(x0, y0 + 1.0) * (1.0 - fx) + get(x0 + 1.0, y0 + 1.0) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Sub-rectangle copy.
    ///
    /// Used to cut the zero-filled border off a [`Grid::shifted`] result, so a
    /// synthetic-shift test measures the estimator rather than the fill.
    pub fn crop(&self, x0: u32, y0: u32, width: u32, height: u32) -> Grid {
        assert!(
            x0 + width <= self.width && y0 + height <= self.height,
            "crop out of bounds"
        );
        let mut out = Grid::new(width, height);
        for y in 0..height {
            let src = (y0 + y) as usize * self.width as usize + x0 as usize;
            let dst = y as usize * width as usize;
            out.data[dst..dst + width as usize]
                .copy_from_slice(&self.data[src..src + width as usize]);
        }
        out
    }

    /// Mean sample value.
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::FrameInfo;
    use crate::tiff_io::write_rgb16_srgb;
    use std::path::Path;

    /// Writes a frame whose luma equals `f(x, y)` by putting the value in all three
    /// channels — the Rec. 709 weights sum to 1, so grey in means that value out.
    fn write_grey(path: &Path, width: u32, height: u32, f: impl Fn(u32, u32) -> f32) {
        let info = FrameInfo {
            width,
            height,
            samples: 3,
            bits_per_sample: 16,
        };
        write_rgb16_srgb(path, info, |y, row| {
            for x in 0..width {
                let v = f(x, y);
                row[x as usize * 3..][..3].fill(v);
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn luma_of_grey_is_the_grey_value() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.tif");
        write_grey(&path, 8, 4, |_, _| 0.25);

        let grid = Grid::from_image(&Image::open(&path).unwrap(), 0).unwrap();
        assert_eq!((grid.width, grid.height), (8, 4));
        for v in &grid.data {
            assert!((v - 0.25).abs() < 1e-3, "{v}");
        }
    }

    #[test]
    fn downsampling_averages_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.tif");
        // Column index as the value, so a 2x2 block average is predictable.
        write_grey(&path, 8, 4, |x, _| x as f32 / 16.0);

        let grid = Grid::from_image(&Image::open(&path).unwrap(), 1).unwrap();
        assert_eq!((grid.width, grid.height), (4, 2));
        for ox in 0..4u32 {
            let want = ((2 * ox) as f32 / 16.0 + (2 * ox + 1) as f32 / 16.0) / 2.0;
            assert!((grid.at(ox, 0) - want).abs() < 1e-3, "{ox}");
        }
    }

    #[test]
    fn shifting_by_whole_pixels_moves_content() {
        let mut g = Grid::new(4, 3);
        g.data[4 + 2] = 1.0;

        let s = g.shifted(1.0, 0.0);
        assert!((s.at(3, 1) - 1.0).abs() < 1e-6);
        assert!(s.at(2, 1).abs() < 1e-6);
    }

    #[test]
    fn shifting_by_half_a_pixel_splits_between_neighbours() {
        let mut g = Grid::new(4, 3);
        g.data[4 + 1] = 1.0;

        let s = g.shifted(0.5, 0.0);
        assert!((s.at(1, 1) - 0.5).abs() < 1e-6);
        assert!((s.at(2, 1) - 0.5).abs() < 1e-6);
    }
}
