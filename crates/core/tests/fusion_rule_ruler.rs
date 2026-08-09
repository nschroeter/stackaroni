//! Does the selection rule move ruler's lifted black point?
//!
//! ```text
//! cargo test --release -p stackaroni-core --test fusion_rule_ruler -- --ignored --nocapture
//! ```
//!
//! ruler scored 3 under the blend, held back specifically by *low contrast*: the tick
//! marks read grey rather than black while their edges stay crisp. That was logged as a
//! separate thread from the blossom softness, on the reasoning that crisp edges rule out
//! a sharpness problem and point at the output transfer function.
//!
//! There is a second explanation the blend makes available, and it is not the transfer
//! function. Averaging a hundred frames per pyramid level mixes the surroundings of a
//! thin dark tick into the tick itself at every coarse level, which raises its floor
//! while leaving the fine level — and so the edge — intact. That predicts exactly
//! "grey but crisp". If it is right, selection should darken the ticks without touching
//! the edges, because it takes one frame's coefficients instead of averaging.
//!
//! Measured as percentiles rather than min/max: a single hot pixel or dust speck would
//! otherwise define the black point.

use std::path::{Path, PathBuf};

use stackaroni_core::image::{FrameInfo, linear_to_srgb};
use stackaroni_core::pipeline::Image;
use stackaroni_core::tiff_io::write_rgb16_srgb;

/// Dense tick marks against the yellow body, away from the frame edges.
const REGION: (u32, u32, u32, u32) = (3000, 2900, 600, 400);
const GUTTER: u32 = 24;

const CANDIDATES: [(&str, &str); 2] = [
    ("blend", "target/debug-out/t10/ruler.tif"),
    ("select", "target/debug-out/t11/ruler_select.tif"),
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_rect(image: &Image, x0: u32, y0: u32, w: u32, h: u32) -> Vec<f32> {
    let info = image.info();
    let mut band = vec![0f32; info.row_len() * h as usize];
    image.read_rows(y0, h, &mut band).unwrap();
    let mut out = vec![0f32; (w * h) as usize * 3];
    for y in 0..h as usize {
        let src = &band[y * info.row_len() + x0 as usize * 3..][..w as usize * 3];
        out[y * w as usize * 3..][..w as usize * 3].copy_from_slice(src);
    }
    out
}

#[test]
#[ignore = "requires test-data/ and both fused outputs, run with --release"]
fn black_point_and_contrast_on_the_tick_marks() {
    let (x0, y0, w, h) = REGION;
    let mut open = Vec::new();
    for (label, rel) in CANDIDATES {
        match Image::open(&root().join(rel)) {
            Ok(image) => open.push((label, image)),
            Err(e) => println!("{label}: skipped ({e})"),
        }
    }
    if open.len() < 2 {
        println!("nothing to compare");
        return;
    }

    println!("\n=== ruler tick contrast, {w}x{h} at ({x0},{y0}) ===");
    println!("sRGB-encoded luma percentiles. p1 is tick ink, p99 the yellow body.\n");
    println!(
        "{:>8}  {:>7}  {:>7}  {:>7}  {:>9}",
        "rule", "p1", "p50", "p99", "p99-p1"
    );

    for (label, image) in &open {
        let rect = read_rect(image, x0, y0, w, h);
        let mut luma: Vec<f32> = rect
            .chunks_exact(3)
            .map(|p| {
                0.299 * linear_to_srgb(p[0])
                    + 0.587 * linear_to_srgb(p[1])
                    + 0.114 * linear_to_srgb(p[2])
            })
            .collect();
        luma.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |q: f32| luma[((q * luma.len() as f32) as usize).min(luma.len() - 1)];
        let (p1, p50, p99) = (at(0.01), at(0.50), at(0.99));
        println!(
            "{label:>8}  {p1:>7.4}  {p50:>7.4}  {p99:>7.4}  {:>9.4}",
            p99 - p1
        );
    }
    println!("\nlower p1 at similar p99 = the ticks got darker, not the image contrastier\n");

    // The crop itself, because "reads grey" is a visual complaint and the percentiles
    // cannot show whether the edges stayed crisp while the floor dropped.
    let out = root().join("target/debug-out/t11/crops-ruler");
    std::fs::create_dir_all(&out).unwrap();
    let total = w * open.len() as u32 + GUTTER * (open.len() as u32 - 1);
    let mut stitched = vec![0f32; (total * h) as usize * 3];
    for (slot, (_, image)) in open.iter().enumerate() {
        let crop = read_rect(image, x0, y0, w, h);
        let x_off = slot as u32 * (w + GUTTER);
        for y in 0..h as usize {
            let dst = (y * total as usize + x_off as usize) * 3;
            stitched[dst..dst + w as usize * 3]
                .copy_from_slice(&crop[y * w as usize * 3..][..w as usize * 3]);
        }
    }
    let path = out.join("ticks.tif");
    let info = FrameInfo {
        width: total,
        height: h,
        samples: 3,
        bits_per_sample: 16,
    };
    write_rgb16_srgb(&path, info, |y, row| {
        let start = y as usize * total as usize * 3;
        row.copy_from_slice(&stitched[start..start + row.len()]);
        Ok(())
    })
    .unwrap();
    let order: Vec<&str> = open.iter().map(|(l, _)| *l).collect();
    println!("wrote {}  [{}]", path.display(), order.join(" | "));
}
