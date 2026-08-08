//! Per-stage debug output, written as 8-bit PNG.
//!
//! These are diagnostic images meant to be opened and looked at — precision beyond
//! 8 bits is irrelevant for spotting a misalignment or a noisy focus map, and PNGs
//! stay small enough to flick through. The pipeline's own intermediates stay in
//! `f32` scratch planes; nothing here feeds back into the pipeline.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use crate::error::{Error, Result};
use crate::grid::Grid;
use crate::pipeline::Transform;

/// Longest edge of a debug image. Full-resolution debug output would be 100 MB a
/// frame and slow to open.
pub const MAX_EDGE: u32 = 1400;

/// Red/green overlay of a reference frame and a target aligned onto it.
///
/// Residual misalignment shows as colour fringing: where the two disagree, one
/// channel is bright and the other dark. A correct alignment reads as neutral
/// yellow-grey everywhere the frames share content.
pub fn write_alignment_overlay(
    path: &Path,
    reference: &Grid,
    target: &Grid,
    transform: Transform,
) -> Result<()> {
    // The transform maps reference coordinates onto the target, so invert it to
    // bring the target back onto the reference.
    let aligned = target.warped(transform.inverse());

    let r = normalize(&downsample_for_view(reference));
    let g = normalize(&downsample_for_view(&aligned));
    let (w, h) = (r.width, r.height);

    let mut rgb = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for i in 0..(w as usize) * (h as usize) {
        rgb.push(to_u8(r.data[i]));
        rgb.push(to_u8(g.data[i]));
        rgb.push(0);
    }
    write_png(path, w, h, &rgb, png::ColorType::Rgb)
}

/// Single-channel grid as a greyscale PNG.
pub fn write_grid(path: &Path, grid: &Grid) -> Result<()> {
    let view = normalize(&downsample_for_view(grid));
    let bytes: Vec<u8> = view.data.iter().map(|&v| to_u8(v)).collect();
    write_png(
        path,
        view.width,
        view.height,
        &bytes,
        png::ColorType::Grayscale,
    )
}

fn write_png(path: &Path, w: u32, h: u32, data: &[u8], color: png::ColorType) -> Result<()> {
    let file = File::create(path).map_err(|e| Error::io(path, e))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), w, h);
    encoder.set_color(color);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder
        .write_header()
        .map_err(|e| Error::io(path, std::io::Error::other(e)))?;
    writer
        .write_image_data(data)
        .map_err(|e| Error::io(path, std::io::Error::other(e)))?;
    Ok(())
}

/// Box-average down until the longest edge fits [`MAX_EDGE`].
fn downsample_for_view(grid: &Grid) -> Grid {
    let longest = grid.width.max(grid.height);
    let factor = longest.div_ceil(MAX_EDGE).max(1);
    if factor == 1 {
        return grid.clone();
    }

    let (w, h) = (grid.width / factor, grid.height / factor);
    let mut out = Grid::new(w.max(1), h.max(1));
    let inv = 1.0 / (factor * factor) as f32;
    for y in 0..out.height {
        for x in 0..out.width {
            let mut acc = 0.0;
            for sy in 0..factor {
                for sx in 0..factor {
                    acc += grid.at(x * factor + sx, y * factor + sy);
                }
            }
            out.data[y as usize * out.width as usize + x as usize] = acc * inv;
        }
    }
    out
}

/// Rescale to `[0,1]` so the image is actually visible whatever the input range.
fn normalize(grid: &Grid) -> Grid {
    let (lo, hi) = grid
        .data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let span = hi - lo;
    let mut out = grid.clone();
    if span > 1e-12 {
        for v in &mut out.data {
            *v = (*v - lo) / span;
        }
    }
    out
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_a_readable_overlay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overlay.png");
        let mut a = Grid::new(32, 24);
        for (i, v) in a.data.iter_mut().enumerate() {
            *v = (i % 7) as f32;
        }
        let b = a.shifted(2.0, 1.0);

        write_alignment_overlay(&path, &a, &b, Transform::translation(2.0, 1.0)).unwrap();

        let decoder = png::Decoder::new(std::io::BufReader::new(File::open(&path).unwrap()));
        let reader = decoder.read_info().unwrap();
        assert_eq!((reader.info().width, reader.info().height), (32, 24));
    }

    #[test]
    fn downsamples_oversized_grids_for_viewing() {
        let grid = Grid::new(MAX_EDGE * 3, MAX_EDGE);
        let view = downsample_for_view(&grid);
        assert!(view.width <= MAX_EDGE, "{}", view.width);
    }
}
