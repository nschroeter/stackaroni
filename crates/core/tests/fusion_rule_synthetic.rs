//! Does the selection rule hold up on the one stack with ground truth?
//!
//! ```text
//! cargo test --release -p stackaroni-core --test fusion_rule_synthetic -- --ignored --nocapture
//! ```
//!
//! synthetic_50 is the stack most likely to disagree with blossom about the fusion
//! rule: it scored 5 under the blend, and its bokeh is where hard selection is exposed,
//! because with nothing in focus there selection has only noise to choose between.
//!
//! The two measures are the ones `weight_radius_sweep` established, kept identical so
//! the numbers are comparable across the log rather than a fresh scale:
//!
//! - **detail**: Laplacian energy over the sharpest 2% of ground-truth gradient.
//!   Below 1.0 means detail lost. The ground truth is a synthetic all-in-focus render,
//!   sharper than any real frame, so 1.0 is unreachable — the **per-pixel oracle** is
//!   the real ceiling and is printed alongside. Without it the number is uninterpretable;
//!   the log records that omission as measurement error #1.
//! - **bokeh**: same over the flattest 50%. **Above 1.0 means mottling** — texture the
//!   truth does not have. This is checklist item 3, and the number to watch here.

use std::path::{Path, PathBuf};

use stackaroni_core::discovery::discover_stack;
use stackaroni_core::grid::Grid;
use stackaroni_core::pipeline::Image;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn stack_dir() -> PathBuf {
    root().join("test-data/synthetic_50")
}

/// Per-pixel squared Laplacian, edges clamped.
fn laplacian_energy(g: &Grid) -> Vec<f32> {
    let (w, h) = (g.width as i64, g.height as i64);
    let at = |x: i64, y: i64| g.at(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32);
    let mut out = vec![0f32; g.data.len()];
    for y in 0..h {
        for x in 0..w {
            let l = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            out[(y * w + x) as usize] = l * l;
        }
    }
    out
}

fn masked_mean(energy: &[f32], mask: &[bool]) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for (e, &m) in energy.iter().zip(mask) {
        if m {
            sum += *e as f64;
            n += 1;
        }
    }
    sum / n.max(1) as f64
}

fn masks(truth: &Grid, top: f32, bottom: f32) -> (Vec<bool>, Vec<bool>) {
    let (w, h) = (truth.width as i64, truth.height as i64);
    let at = |x: i64, y: i64| truth.at(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32);
    let mut gradient = Vec::with_capacity(truth.data.len());
    for y in 0..h {
        for x in 0..w {
            let gx = at(x + 1, y) - at(x - 1, y);
            let gy = at(x, y + 1) - at(x, y - 1);
            gradient.push((gx * gx + gy * gy).sqrt());
        }
    }
    let mut sorted = gradient.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hi = sorted[((1.0 - top) * sorted.len() as f32) as usize];
    let lo = sorted[(bottom * sorted.len() as f32) as usize];
    (
        gradient.iter().map(|&g| g >= hi).collect(),
        gradient.iter().map(|&g| g <= lo).collect(),
    )
}

#[test]
#[ignore = "requires test-data/ and both fused outputs, run with --release"]
fn selection_versus_blend_against_ground_truth() {
    let Ok(truth) = Image::open(&stack_dir().join("ground_truth_all_in_focus.tiff")) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let truth = Grid::from_image(&truth, 0).unwrap();
    let (structure, smooth) = masks(&truth, 0.02, 0.50);
    let truth_energy = laplacian_energy(&truth);
    let truth_detail = masked_mean(&truth_energy, &structure);
    let truth_bokeh = masked_mean(&truth_energy, &smooth);

    println!("\n=== synthetic_50, 1.00 = matches ground truth ===");
    println!("detail <1 means detail lost; bokeh >1 means mottling (checklist item 3)\n");
    println!("{:>10}  {:>8}  {:>8}", "rule", "detail", "bokeh");

    // The radius sweep tests one mechanism for the antenna softness Human User scored down:
    // the salience window is square and `r2` spans 5 px, wider than a 1-3 px antenna, so
    // at the finest level a neighbouring frame's broader defocused energy can win the
    // window even where the line is sharper at the exact pixel. If that is the cause,
    // detail should rise as the radius shrinks. `r0` is per-pixel argmax — the rule §6b
    // rejects as noise-sensitive — included as the endpoint that shows what the window
    // is buying.
    for (label, rel) in [
        ("blend", "target/debug-out/t10/synthetic_50.tif"),
        ("select r0", "target/debug-out/t11/synthetic_50_sel_r0.tif"),
        ("select r1", "target/debug-out/t11/synthetic_50_sel_r1.tif"),
        ("select r2", "target/debug-out/t11/synthetic_50_select.tif"),
        ("select r3", "target/debug-out/t11/synthetic_50_sel_r3.tif"),
    ] {
        let Ok(image) = Image::open(&root().join(rel)) else {
            println!("{label:>10}  (missing {rel})");
            continue;
        };
        let energy = laplacian_energy(&Grid::from_image(&image, 0).unwrap());
        println!(
            "{label:>10}  {:>8.3}  {:>8.3}",
            masked_mean(&energy, &structure) / truth_detail,
            masked_mean(&energy, &smooth) / truth_bokeh
        );
    }

    // The ceiling, without which `detail` cannot be read.
    let stack = discover_stack(&stack_dir()).unwrap();
    let mut per_pixel_max = vec![0f32; truth.data.len()];
    let mut best_frame = 0.0f64;
    for path in &stack.frames {
        let e = laplacian_energy(&Grid::from_image(&Image::open(path).unwrap(), 0).unwrap());
        best_frame = best_frame.max(masked_mean(&e, &structure) / truth_detail);
        for (m, v) in per_pixel_max.iter_mut().zip(&e) {
            *m = m.max(*v);
        }
    }
    println!(
        "\nceiling: best single frame {best_frame:.3}, per-pixel oracle {:.3} (unreachable)",
        masked_mean(&per_pixel_max, &structure) / truth_detail
    );
    println!("read detail as a fraction of the oracle, not of 1.00\n");
}

/// Is the antenna softness the fusion rule's, or the resampling the warp does?
///
/// The salience-radius sweep refuted the window-too-wide explanation: detail is flat
/// across r0..r3 and if anything *rises* with a wider window. That leaves a candidate the
/// fusion rule cannot be blamed for. Every selected coefficient is read out of a
/// **bilinearly resampled** frame, because fusion warps each frame into anchor
/// coordinates before building its pyramid — and bilinear interpolation is a low-pass
/// filter, which costs most exactly on 1-3 px structures like an antenna.
///
/// This runs the identical fusion with `Transform::IDENTITY`, so nothing is resampled.
/// The gap between the two is the resampling cost, isolated.
///
/// **The answer generalizes in the direction that matters.** If resampling is the cause,
/// real stacks are affected *more*, not less: synthetic_50's scale spans ~0.7% while
/// blossom and ruler span ~11%, so their frames are stretched far harder and interpolated
/// at correspondingly more non-integer positions.
#[test]
#[ignore = "requires test-data/, run with --release"]
fn how_much_detail_does_the_warp_itself_cost() {
    use stackaroni_core::focus::WindowedLaplacian;
    use stackaroni_core::fusion::SelectionFusion;
    use stackaroni_core::pipeline::{
        FocusMap, FocusMetric, ImageFusion, Registration, Transform, WeightEstimator,
    };
    use stackaroni_core::weights::{GuideSpace, GuidedWeights};
    use std::collections::HashMap;

    let Ok(stack) = discover_stack(&stack_dir()) else {
        eprintln!("skipping: test-data/synthetic_50 not present");
        return;
    };
    let truth = Grid::from_image(
        &Image::open(&stack_dir().join("ground_truth_all_in_focus.tiff")).unwrap(),
        0,
    )
    .unwrap();
    let (structure, smooth) = masks(&truth, 0.02, 0.50);
    let truth_energy = laplacian_energy(&truth);
    let truth_detail = masked_mean(&truth_energy, &structure);
    let truth_bokeh = masked_mean(&truth_energy, &smooth);

    let out = root().join("target/debug-out/t11/identity");
    std::fs::create_dir_all(&out).unwrap();

    let transforms = vec![Transform::IDENTITY; stack.frames.len()];
    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let metric = WindowedLaplacian::new(4, &out, by_path.clone());
    let focus_maps: Vec<FocusMap> = stack
        .frames
        .iter()
        .map(|p| metric.evaluate(&Image::open(p).unwrap(), &()).unwrap())
        .collect();
    let weights = GuidedWeights::new(
        stack.frames.clone(),
        transforms,
        4,
        1e-4,
        GuideSpace::Perceptual,
        &out,
    )
    .weights(&focus_maps, &())
    .unwrap();
    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p).unwrap())
        .collect();

    let path = out.join("identity_select.tif");
    let fused = SelectionFusion::new(&path, by_path, 32, 2)
        .fuse(&images, &weights, &())
        .unwrap();
    let energy = laplacian_energy(&Grid::from_image(&fused, 0).unwrap());

    println!("\n=== select r2, identity transforms (nothing resampled) ===");
    println!(
        "detail {:.3}   bokeh {:.3}",
        masked_mean(&energy, &structure) / truth_detail,
        masked_mean(&energy, &smooth) / truth_bokeh
    );
    println!("compare against select r2 with real registration: detail 0.134");

    // Do NOT attribute that gap to blur without this. The masks come from the unwarped
    // truth, and the registered output sits in the *anchor frame's* coordinates. If those
    // differ, the "sharpest 2% of truth gradient" mask no longer lands on the fused
    // image's edges, and measured energy collapses with no blurring involved at all — on a
    // 1-3 px antenna a 1 px offset is enough. Phase-correlate to find out which it is.
    let registered = root().join("target/debug-out/t11/synthetic_50_select.tif");
    if let Ok(registered) = Image::open(&registered) {
        let truth_image = Image::open(&stack_dir().join("ground_truth_all_in_focus.tiff")).unwrap();
        let t = stackaroni_core::registration::PhaseCorrelation::new(1)
            .align(&truth_image, &registered, &())
            .unwrap();
        let (cx, cy) = (truth.width as f32 / 2.0, truth.height as f32 / 2.0);
        println!(
            "\nregistered output vs truth: scale {:.5}  dx {:+.2}  dy {:+.2}",
            t.scale, t.dx, t.dy
        );
        println!(
            "worst-corner displacement {:.2} px",
            ((t.scale - 1.0) * (cx * cx + cy * cy).sqrt()).abs()
                + (t.dx * t.dx + t.dy * t.dy).sqrt()
        );
        println!("under ~1 px => masks still land, gap is real blur; over => gap is confounded");
    }
    println!();
}

/// Stitched `blend | select | ground truth` crops, because the numbers above do not
/// settle it.
///
/// `bokeh` above 1.0 is the mottling *threshold*, not a defect on its own — the r4
/// configuration shipped at 1.28-1.32 and rated 5 with no visible mottling. Whether
/// excess background energy reads as patchy sharpening to a human is a visual call, and
/// the log has twice recorded a numeric arbiter pointing the wrong way here.
#[test]
#[ignore = "requires test-data/ and both fused outputs, run with --release"]
fn synthetic_crops_for_a_visual_call() {
    // 1200x900. Regions chosen for the two checks that disagree above.
    const REGIONS: [(&str, u32, u32, u32, u32); 3] = [
        // Thin radiating antenna lines — where `detail` is measured.
        ("antennae", 380, 180, 400, 300),
        // Pure background bokeh, top-left corner — where `bokeh` is measured and the
        // item-3 risk lives. Must contain no subject at all: a crop that clips the body
        // edge measures the edge, not the background.
        ("bokeh", 60, 60, 400, 300),
        // Subject edge against background: halo and mottling meet here.
        ("edge", 560, 420, 400, 300),
    ];
    const GUTTER: u32 = 16;

    let sources = [
        (
            "blend",
            root().join("target/debug-out/t10/synthetic_50.tif"),
        ),
        (
            "select",
            root().join("target/debug-out/t11/synthetic_50_select.tif"),
        ),
        ("truth", stack_dir().join("ground_truth_all_in_focus.tiff")),
    ];
    let open: Vec<(&str, Image)> = sources
        .iter()
        .filter_map(|(l, p)| Image::open(p).ok().map(|i| (*l, i)))
        .collect();
    if open.len() < 2 {
        println!("skipping: need at least two of blend/select/truth");
        return;
    }

    let out = root().join("target/debug-out/t11/crops-synthetic");
    std::fs::create_dir_all(&out).unwrap();

    for (name, x0, y0, w, h) in REGIONS {
        let total = w * open.len() as u32 + GUTTER * (open.len() as u32 - 1);
        let mut stitched = vec![0f32; (total * h) as usize * 3];

        for (slot, (_, image)) in open.iter().enumerate() {
            let info = image.info();
            let mut band = vec![0f32; info.row_len() * h as usize];
            image.read_rows(y0, h, &mut band).unwrap();
            let x_off = slot as u32 * (w + GUTTER);
            for y in 0..h as usize {
                let src = &band[y * info.row_len() + x0 as usize * 3..][..w as usize * 3];
                let dst = (y * total as usize + x_off as usize) * 3;
                stitched[dst..dst + w as usize * 3].copy_from_slice(src);
            }
        }

        let path = out.join(format!("{name}.tif"));
        let info = stackaroni_core::image::FrameInfo {
            width: total,
            height: h,
            samples: 3,
            bits_per_sample: 16,
        };
        stackaroni_core::tiff_io::write_rgb16_srgb(&path, info, |y, row| {
            let start = y as usize * total as usize * 3;
            row.copy_from_slice(&stitched[start..start + row.len()]);
            Ok(())
        })
        .unwrap();
        let order: Vec<&str> = open.iter().map(|(l, _)| *l).collect();
        println!("wrote {}  [{}]", path.display(), order.join(" | "));
    }
}
