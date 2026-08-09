//! Where do the two fusion rules diverge from each other, and from Zerene's PMax?
//!
//! ```text
//! cargo test --release -p stackaroni-core --test fusion_rule_crops -- --ignored --nocapture
//! ```
//!
//! Deliberately not a score. `test-data/README.md` is explicit that
//! `reference_pmax.tif` is a qualitative reference and not an automated scoring
//! target — an aggregate error against a reference has already pointed the wrong way
//! twice in `docs/eval-log.md`. So this writes matched crops for looking at, and the
//! only numbers it prints are the ones needed to trust the crops themselves: whether
//! the three images share a coordinate system at all.

use std::path::{Path, PathBuf};

use stackaroni_core::debug;
use stackaroni_core::discovery::discover_stack;
use stackaroni_core::grid::Grid;
use stackaroni_core::image::{FrameInfo, linear_to_srgb};
use stackaroni_core::pipeline::{Image, Registration};
use stackaroni_core::registration::PhaseCorrelation;
use stackaroni_core::tiff_io::write_rgb16_srgb;

/// Label and path, relative to the repository root.
const CANDIDATES: [(&str, &str); 3] = [
    ("blend", "target/debug-out/t10/blossom.tif"),
    ("select", "target/debug-out/t11/blossom_select.tif"),
    ("pmax", "test-data/blossom/reference_pmax.tif"),
];

/// Name, x0, y0, width, height. Crops are square-ish and small enough to read at 1:1.
/// Coordinates read off the `overview_*.png` dumps, which are downsampled 7x.
const REGIONS: [(&str, u32, u32, u32, u32); 4] = [
    // The dense bud cluster at the tip — the densest fine detail in the frame.
    ("florets", 4100, 3600, 500, 400),
    // Crisscrossing stalks. PMax is documented as strong exactly here, so this is the
    // most diagnostic region for the rule change.
    ("stalks", 6050, 1060, 500, 400),
    // The silhouette against the background, where the blend overview shows a dark
    // fringe. Comparable to the halo-envelope profile.
    ("rim", 3850, 3750, 500, 400),
    // Away from the subject: the place hard selection is most likely to go wrong,
    // because with no frame actually in focus it has only noise to choose between.
    ("bokeh", 1600, 1950, 500, 400),
];

/// Gutter between stitched crops, in pixels.
const GUTTER: u32 = 24;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Read one rectangle as interleaved linear RGB.
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
#[ignore = "requires test-data/ and a fused blossom, run with --release"]
fn matched_crops_of_both_rules_and_pmax() {
    let out = root().join("target/debug-out/t11/crops");
    std::fs::create_dir_all(&out).unwrap();

    let mut open = Vec::new();
    for (label, rel) in CANDIDATES {
        let path = root().join(rel);
        match Image::open(&path) {
            Ok(image) => {
                let i = image.info();
                println!("{label:>7}: {}x{}  {}", i.width, i.height, rel);
                open.push((label, image));
            }
            Err(e) => println!("{label:>7}: skipped ({e})"),
        }
    }
    if open.len() < 2 {
        println!("\nnothing to compare");
        return;
    }

    // A shared coordinate system is what makes a crop comparison mean anything. Zerene
    // may trim the frame; if it has, the same (x,y) is not the same feature and the
    // crops are not evidence.
    let base = open[0].1.info();
    for (label, image) in &open[1..] {
        let i = image.info();
        if (i.width, i.height) != (base.width, base.height) {
            println!(
                "\n!! {label} is {}x{} against {}x{} — crops are NOT feature-matched",
                i.width, i.height, base.width, base.height
            );
        }
    }

    for (name, x0, y0, w, h) in REGIONS {
        let total = w * open.len() as u32 + GUTTER * (open.len() as u32 - 1);
        let mut stitched = vec![0f32; (total * h) as usize * 3];

        for (slot, (_, image)) in open.iter().enumerate() {
            let i = image.info();
            if x0 + w > i.width || y0 + h > i.height {
                continue;
            }
            let crop = read_rect(image, x0, y0, w, h);
            let x_off = slot as u32 * (w + GUTTER);
            for y in 0..h as usize {
                let dst = (y * total as usize + x_off as usize) * 3;
                stitched[dst..dst + w as usize * 3]
                    .copy_from_slice(&crop[y * w as usize * 3..][..w as usize * 3]);
            }
        }

        let path = out.join(format!("{name}.tif"));
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

    // "Selection looks noisier" is a visual claim, and the log carries a standing
    // caution about those in both directions. The background region is uniform in the
    // subject, so its spread is noise and nothing else — a noise floor, not an
    // aggregate error against the reference.
    let (_, x0, y0, w, h) = REGIONS[3];
    println!("\nbokeh noise, SD of linear luma over {w}x{h} at ({x0},{y0}):");
    for (label, image) in &open {
        let rect = read_rect(image, x0, y0, w, h);
        let luma: Vec<f64> = rect
            .chunks_exact(3)
            .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
            .collect();
        let mean = luma.iter().sum::<f64>() / luma.len() as f64;
        let sd = (luma.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / luma.len() as f64).sqrt();
        println!(
            "  {label:>7}: mean {mean:.4}  sd {sd:.5}  ({:.2}% of mean)",
            100.0 * sd / mean
        );
    }
    println!();
}

/// Is the colour noise in `select` manufactured by fusion, or inherited from the frames?
///
/// The luma SD already logged cannot answer this — colour noise lives in the chroma
/// difference channels, and a result can be quiet in Y while drifting in Cb/Cr.
///
/// **The control is the point.** Selection takes one frame's coefficients instead of
/// averaging a hundred, so it necessarily passes through roughly one frame's chroma
/// noise where the blend suppressed it by sqrt(N). That is arithmetic, not a defect. So
/// single source frames are measured alongside: if `select` lands near them, the noise is
/// inherited and the only lever is denoising; if it lands *above* the noisiest frame,
/// fusion is manufacturing colour that no source has, and that is a bug with a cause
/// worth finding.
///
/// Measured in sRGB-encoded space, because chroma noise is a perceptual complaint and
/// linear light does not weight it the way the eye does. The patch is flat background, so
/// the source frames are read unwarped — a few pixels of registration shift cannot change
/// the spread of a uniform region.
#[test]
#[ignore = "requires test-data/ and a fused blossom, run with --release"]
fn chroma_noise_against_a_single_frame_control() {
    let (_, x0, y0, w, h) = REGIONS[3];

    // BT.601 chroma difference channels.
    let stats = |image: &Image| -> (f64, f64, f64) {
        let rect = read_rect(image, x0, y0, w, h);
        let (mut cb, mut cr) = (Vec::new(), Vec::new());
        let mut y_sum = 0.0f64;
        for p in rect.chunks_exact(3) {
            let (r, g, b) = (
                linear_to_srgb(p[0]) as f64,
                linear_to_srgb(p[1]) as f64,
                linear_to_srgb(p[2]) as f64,
            );
            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            y_sum += luma;
            cb.push(0.564 * (b - luma));
            cr.push(0.713 * (r - luma));
        }
        let sd = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        (y_sum / cb.len() as f64, sd(&cb), sd(&cr))
    };

    println!("\n=== chroma noise, {w}x{h} background patch at ({x0},{y0}) ===");
    println!(
        "sRGB-encoded BT.601. {:>10}  {:>9}  {:>9}",
        "mean Y", "sd Cb", "sd Cr"
    );
    let mut candidates = Vec::new();
    for (label, rel) in CANDIDATES {
        let Ok(image) = Image::open(&root().join(rel)) else {
            continue;
        };
        let s = stats(&image);
        println!("{label:>21}  {:>10.4}  {:>9.5}  {:>9.5}", s.0, s.1, s.2);
        candidates.push((label, s));
    }

    // The control: individual frames, spread across the stack.
    let Ok(stack) = discover_stack(&root().join("test-data/blossom")) else {
        println!("\nno source frames to control against");
        return;
    };
    // Ten frames, not three. Chroma noise varies frame to frame by more than the margin
    // being judged, so a max over three samples is not a ceiling — `select` sitting a few
    // percent above it would be indistinguishable from sampling luck.
    let n = stack.frames.len();
    let mut frames = Vec::new();
    println!("{:->21}", "");
    for k in 0..10 {
        let i = k * (n - 1) / 9;
        let image = Image::open(&stack.frames[i]).unwrap();
        let s = stats(&image);
        let name = stack.frames[i]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!("{name:>21}  {:>10.4}  {:>9.5}  {:>9.5}", s.0, s.1, s.2);
        frames.push((name, s));
    }

    // Brightness-matched, and this is not a refinement — it is the whole comparison.
    // Chroma SD falls monotonically with mean Y across these frames, so comparing two
    // images at different brightness measures the tone difference as much as the noise.
    // `docs/eval-log.md` already records one conclusion wrecked by exactly this: a noise
    // floor measured in a bright region and subtracted from a dark one.
    println!("\nagainst the source frame nearest each candidate's own brightness:");
    for (label, (y, cb, cr)) in &candidates {
        let (name, (fy, fcb, fcr)) = frames
            .iter()
            .min_by(|a, b| (a.1.0 - y).abs().partial_cmp(&(b.1.0 - y).abs()).unwrap())
            .unwrap();
        println!(
            "  {label:>7} (Y {y:.4}) vs {name} (Y {fy:.4}):  Cb {:+.1}%  Cr {:+.1}%",
            100.0 * (cb / fcb - 1.0),
            100.0 * (cr / fcr - 1.0)
        );
    }
    println!("\nnear 0% => one frame's noise passed through, which is what selection must do.");
    println!("far above => fusion manufactures colour; far below => something denoises.\n");
}

/// Does Zerene's output actually sit in the same coordinate system as ours?
///
/// Matching canvas sizes are not enough. Zerene runs its own alignment and may resolve
/// the stack onto a different reference frame; with scale spanning ~11% across this
/// stack, a different anchor displaces features far from the centre by hundreds of
/// pixels while leaving the centre almost unmoved. That would make a far-from-centre
/// crop pair look like a fusion difference when it is a registration-reference
/// difference, so measure it rather than assume it.
#[test]
#[ignore = "requires test-data/ and a fused blossom, run with --release"]
fn pmax_versus_ours_is_a_similarity_not_an_identity() {
    let ours = root().join("target/debug-out/t11/blossom_select.tif");
    let theirs = root().join("test-data/blossom/reference_pmax.tif");
    let (Ok(ours), Ok(theirs)) = (Image::open(&ours), Image::open(&theirs)) else {
        println!("skipping: need both a select fusion and reference_pmax.tif");
        return;
    };

    let t = PhaseCorrelation::new(3).align(&theirs, &ours, &()).unwrap();
    let info = ours.info();
    let (cx, cy) = (info.width as f32 / 2.0, info.height as f32 / 2.0);
    let corner = (cx * cx + cy * cy).sqrt();

    println!("\n=== ours vs reference_pmax ===");
    println!("scale {:.5}  dx {:+.1}  dy {:+.1}", t.scale, t.dx, t.dy);
    println!(
        "displacement at the centre {:.0} px, at a corner {:.0} px",
        (t.dx * t.dx + t.dy * t.dy).sqrt(),
        ((t.scale - 1.0) * corner).abs() + (t.dx * t.dx + t.dy * t.dy).sqrt()
    );
    println!("crops are feature-matched only where that displacement is small\n");
}

/// Downsampled luma of each candidate, for picking crop coordinates.
#[test]
#[ignore = "requires test-data/ and a fused blossom, run with --release"]
fn overviews_for_choosing_regions() {
    let out = root().join("target/debug-out/t11/crops");
    std::fs::create_dir_all(&out).unwrap();

    for (label, rel) in CANDIDATES {
        let Ok(image) = Image::open(&root().join(rel)) else {
            continue;
        };
        let grid = Grid::from_image(&image, 0).unwrap();
        let path = out.join(format!("overview_{label}.png"));
        debug::write_grid(&path, &grid).unwrap();
        println!("wrote {}", path.display());
    }
}
