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
use stackaroni_core::pipeline::{Image, Registration, Transform};
use stackaroni_core::registration::PhaseCorrelation;
use stackaroni_core::tiff_io::write_rgb16_srgb;

/// Dense tick marks against the yellow body, away from the frame edges.
const REGION: (u32, u32, u32, u32) = (3000, 2900, 600, 400);
const GUTTER: u32 = 24;

const CANDIDATES: [(&str, &str); 3] = [
    ("blend", "target/debug-out/t10/ruler.tif"),
    ("select", "target/debug-out/t11/ruler_select.tif"),
    ("pmax", "test-data/ruler/reference_pmax.tif"),
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Read a rectangle of `image` **through a transform**, so the result lands in the
/// caller's coordinate system rather than the image's own.
///
/// Needed because Zerene resolves onto its own reference frame: at this crop, PMax is 78 px
/// away from where our output puts the same ticks. Reading the same (x,y) from both would
/// compare different ticks, and on a ruler — a repetitive pattern where a wrong crop still
/// *looks* plausible — that error is invisible rather than obvious.
///
/// Centre-relative, matching `fusion::warp_frame`'s convention, because `Transform` scales
/// about the image centre and not the origin.
fn read_rect_through(
    image: &Image,
    t: Transform,
    region: (u32, u32, u32, u32),
    centre: (f32, f32),
) -> Vec<f32> {
    let (x0, y0, w, h) = region;
    let (cx, cy) = centre;
    let info = image.info();
    let source_y = |y: u32| t.apply(0.0, y as f32 - cy).1 + cy;
    let (a, b) = (source_y(y0), source_y(y0 + h - 1));
    let from = (a.min(b).floor() as i64 - 1).clamp(0, info.height as i64 - 1) as u32;
    let to = (a.max(b).ceil() as i64 + 2).clamp(1, info.height as i64) as u32;
    let span = to - from;

    let mut band = vec![0f32; info.row_len() * span as usize];
    image.read_rows(from, span, &mut band).unwrap();

    let sample = |x: f32, y: f32, ch: usize| -> f32 {
        let (fx0, fy0) = (x.floor(), y.floor());
        let (fx, fy) = (x - fx0, y - fy0);
        let get = |ix: i64, iy: i64| -> f32 {
            let ix = ix.clamp(0, info.width as i64 - 1) as usize;
            let iy = (iy - from as i64).clamp(0, span as i64 - 1) as usize;
            band[iy * info.row_len() + ix * 3 + ch]
        };
        let (ix, iy) = (fx0 as i64, fy0 as i64);
        let top = get(ix, iy) * (1.0 - fx) + get(ix + 1, iy) * fx;
        let bottom = get(ix, iy + 1) * (1.0 - fx) + get(ix + 1, iy + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    };

    let mut out = vec![0f32; (w * h) as usize * 3];
    for j in 0..h {
        for i in 0..w {
            let (sx, sy) = t.apply((x0 + i) as f32 - cx, (y0 + j) as f32 - cy);
            let d = (j as usize * w as usize + i as usize) * 3;
            for ch in 0..3 {
                out[d + ch] = sample(sx + cx, sy + cy, ch);
            }
        }
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

    // Put PMax in our coordinates before anything is measured or cropped. Both stackaroni
    // outputs already share a frame; only the third party needs resolving.
    let info = open[0].1.info();
    let (cx, cy) = (info.width as f32 / 2.0, info.height as f32 / 2.0);
    let to_ours: Vec<Transform> = open
        .iter()
        .map(|(label, image)| {
            if *label != "pmax" {
                return Transform::IDENTITY;
            }
            let ours = &open.iter().find(|(l, _)| *l == "select").expect("select").1;
            let t = PhaseCorrelation::new(3).align(ours, image).unwrap();
            let shift = (t.dx * t.dx + t.dy * t.dy).sqrt();
            let (rx, ry) = (
                x0 as f32 + w as f32 / 2.0 - cx,
                y0 as f32 + h as f32 / 2.0 - cy,
            );
            println!(
                "\npmax sits at scale {:.5}, dx {:+.1}, dy {:+.1} against our output",
                t.scale, t.dx, t.dy
            );
            println!(
                "that is {:.0} px at this crop and {:.0} px at a corner — resampling it into",
                (t.scale - 1.0).abs() * (rx * rx + ry * ry).sqrt() + shift,
                (t.scale - 1.0).abs() * (cx * cx + cy * cy).sqrt() + shift
            );
            println!("our frame so all three panels show the same ticks");
            t
        })
        .collect();

    println!("\n=== ruler tick contrast, {w}x{h} at ({x0},{y0}) ===");
    println!("sRGB-encoded luma percentiles. p1 is tick ink, p99 the yellow body.\n");
    println!(
        "{:>8}  {:>7}  {:>7}  {:>7}  {:>9}",
        "rule", "p1", "p50", "p99", "p99-p1"
    );

    for ((label, image), &t) in open.iter().zip(&to_ours) {
        let rect = read_rect_through(image, t, REGION, (cx, cy));
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
    for (slot, ((_, image), &t)) in open.iter().zip(&to_ours).enumerate() {
        let crop = read_rect_through(image, t, REGION, (cx, cy));
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
