//! Headless runner used for evaluation. See the "Evaluation workflow" section of
//! `CLAUDE.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::builder::{PossibleValuesParser, TypedValueParser};
use clap::{Parser, ValueEnum};
use fs4::available_space;
use stackaroni_core::debug;
use stackaroni_core::defaults;
use stackaroni_core::discovery::{
    Stack, discover_stack, discover_test_set, ensure_output_outside_stack,
};
use stackaroni_core::focus::{WindowedLaplacian, evaluate_stack};
use stackaroni_core::fusion::FusionKind;
use stackaroni_core::grid::Grid;
use stackaroni_core::pipeline::{FocusMap, Image, RunControl, Stage, Transform, WeightEstimator};
use stackaroni_core::registration::{PhaseCorrelation, register_stack};
use stackaroni_core::weights::{GuideSpace, GuidedWeights};

#[derive(Parser)]
#[command(about = "Focus stacking for insect macro photography", version)]
struct Cli {
    /// Directory holding one stack's frames.
    #[arg(long, value_name = "DIR", required_unless_present = "test_set")]
    input: Option<PathBuf>,

    /// Directory holding several stack directories; every stack is processed.
    #[arg(long, value_name = "DIR", conflicts_with = "input")]
    test_set: Option<PathBuf>,

    /// Output TIFF for --input, or output directory for --test-set.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Write per-stage debug images here (alignment overlay, focus heatmap,
    /// label field, weight maps, fused preview).
    #[arg(long, value_name = "DIR")]
    debug_out: Option<PathBuf>,

    /// Parent for the per-run scratch directory. Defaults to the system temp dir.
    #[arg(long, value_name = "DIR")]
    scratch: Option<PathBuf>,

    /// Pyramid level for phase correlation. Higher is coarser and faster.
    #[arg(long, default_value_t = defaults::REGISTRATION_LEVEL)]
    registration_level: u32,

    /// Window radius for the windowed-Laplacian focus measure.
    #[arg(long, default_value_t = defaults::FOCUS_RADIUS)]
    focus_radius: u32,

    /// Guided-filter radius for weight refinement.
    ///
    /// The single most consequential parameter: too large averages many frames per
    /// pixel and destroys thin-structure contrast, too small lets argmax mottling
    /// through in defocused background. The best value is data-dependent — a noisy
    /// stack spreads weight further at the same radius — which is why it is exposed
    /// rather than fixed.
    #[arg(long, default_value_t = defaults::GUIDE_RADIUS)]
    guide_radius: u32,

    /// Guided-filter regularization. Larger smooths more.
    #[arg(long, default_value_t = defaults::GUIDE_EPSILON)]
    guide_epsilon: f32,

    /// Stop halving the pyramid once the coarsest level reaches this size.
    #[arg(long, default_value_t = defaults::PYRAMID_FLOOR)]
    pyramid_floor: u32,

    /// Tone space the guided filter's guide image is measured in.
    #[arg(long, value_enum, default_value_t = GuideSpaceArg::DEFAULT)]
    guide_space: GuideSpaceArg,

    /// How the pyramid levels are combined. See `docs/algorithms.md` §6b.
    ///
    /// `select` is the default as of T11, on ratings across all three stacks:
    /// blossom 1 -> 5, ruler 3 -> 5, synthetic_50 5 -> 4. Reproducing an eval-log row
    /// from before the flip needs an explicit `--fusion blend`.
    #[arg(
        long,
        value_name = "RULE",
        default_value_t = defaults::FUSION,
        value_parser = PossibleValuesParser::new(FusionKind::TOKENS)
            .try_map(|s| FusionKind::from_token(&s).ok_or_else(|| format!("unknown fusion rule {s}"))),
    )]
    fusion: FusionKind,

    /// Salience window radius for `--fusion select`. Ignored by `blend`.
    #[arg(long, default_value_t = defaults::SALIENCE_RADIUS)]
    salience_radius: u32,

    /// Report per-frame progress.
    #[arg(long, short)]
    verbose: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum GuideSpaceArg {
    Linear,
    Perceptual,
}

impl GuideSpaceArg {
    const DEFAULT: Self = match defaults::GUIDE_SPACE {
        GuideSpace::Linear => Self::Linear,
        GuideSpace::Perceptual => Self::Perceptual,
    };
}

impl From<GuideSpaceArg> for GuideSpace {
    fn from(value: GuideSpaceArg) -> Self {
        match value {
            GuideSpaceArg::Linear => GuideSpace::Linear,
            GuideSpaceArg::Perceptual => GuideSpace::Perceptual,
        }
    }
}

/// The CLI's `RunControl`: prints progress when `-v`, never cancels.
///
/// Cancellation is the app's concern — a headless batch run has nobody to press a
/// button — but progress reporting is shared, which is why one trait carries both.
struct Progress {
    verbose: bool,
}

impl RunControl for Progress {
    fn progress(&self, stage: Stage, done: usize, total: usize) {
        // Every tenth frame and the last one: a 100-frame stack would otherwise emit
        // 400 lines, and the timings printed per stage are the useful record.
        if self.verbose && (done.is_multiple_of(10) || done == total) {
            println!("  {} {done}/{total}", stage.label());
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let stacks = match (&cli.input, &cli.test_set) {
        (Some(dir), _) => vec![discover_stack(dir)?],
        (_, Some(root)) => discover_test_set(root)?,
        _ => unreachable!("clap requires one of --input or --test-set"),
    };

    // Every stack is checked before any of them runs. Failing on stack three of five
    // after two results are already on disk would leave the user to work out which are
    // trustworthy; refusing up front leaves nothing to untangle.
    let outputs: Vec<PathBuf> = stacks
        .iter()
        .map(|stack| {
            let output = output_path(&cli, stack, stacks.len())?;
            ensure_output_outside_stack(&output, &stack.dir)?;
            Ok(output)
        })
        .collect::<Result<_>>()?;

    for (stack, output) in stacks.iter().zip(&outputs) {
        run(&cli, stack, output)?;
    }
    Ok(())
}

/// Where this stack's result goes.
fn output_path(cli: &Cli, stack: &Stack, count: usize) -> Result<PathBuf> {
    Ok(match (&cli.output, count) {
        // A single stack with an explicit path uses it verbatim — but its directory has
        // to exist, and finding that out at the write means finding out after the whole
        // pipeline has run. That cost a full 100-frame blossom run once: the failure came
        // three minutes in, with 37 GB of scratch already written.
        (Some(path), 1) if cli.input.is_some() => {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating output directory {}", parent.display()))?;
            }
            path.clone()
        }
        (Some(dir), _) => {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating output directory {}", dir.display()))?;
            dir.join(format!("{}.tif", stack.name))
        }
        (None, _) => PathBuf::from(format!("{}_fused.tif", stack.name)),
    })
}

fn run(cli: &Cli, stack: &Stack, output: &Path) -> Result<()> {
    let started = Instant::now();
    let info = stack.probe()?.info;
    let frames = stack.frames.len();
    println!(
        "{}: {frames} frames, {}x{}, 16-bit RGB",
        stack.name, info.width, info.height
    );

    // Scratch holds one focus map and one weight plane per frame, both f32 and
    // full resolution. Check before starting rather than failing an hour in.
    let scratch_root = cli.scratch.clone().unwrap_or_else(std::env::temp_dir);
    let needed = 2 * frames as u64 * info.width as u64 * info.height as u64 * 4;
    let available = available_space(&scratch_root)
        .with_context(|| format!("checking free space on {}", scratch_root.display()))?;
    if available < needed {
        bail!(
            "{} needs {:.1} GB of scratch but {} has {:.1} GB free",
            stack.name,
            needed as f64 / 1e9,
            scratch_root.display(),
            available as f64 / 1e9
        );
    }

    let scratch = scratch_root.join(format!("stackaroni-{}-{}", stack.name, std::process::id()));
    std::fs::create_dir_all(&scratch)
        .with_context(|| format!("creating scratch {}", scratch.display()))?;
    let debug_dir = match &cli.debug_out {
        Some(dir) => {
            let d = dir.join(&stack.name);
            std::fs::create_dir_all(&d)?;
            Some(d)
        }
        None => None,
    };

    let result = pipeline(cli, stack, output, &scratch, debug_dir.as_deref());

    // A failed run's scratch is worth keeping to inspect. A *cancelled* one is not —
    // it is a deliberate stop with nothing to debug — so it cleans up like a success.
    // The CLI cannot currently cancel, but the branch belongs with the rule rather than
    // with the caller that first exercises it.
    let cancelled = matches!(
        result.as_ref().err().and_then(|e| e.downcast_ref()),
        Some(stackaroni_core::error::Error::Cancelled)
    );
    if result.is_ok() || cancelled {
        let _ = std::fs::remove_dir_all(&scratch);
    } else {
        eprintln!("scratch kept for inspection at {}", scratch.display());
    }
    result?;

    println!(
        "  wrote {} in {:.0}s",
        output.display(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

fn pipeline(
    cli: &Cli,
    stack: &Stack,
    output: &Path,
    scratch: &Path,
    debug_dir: Option<&Path>,
) -> Result<()> {
    let verbose = cli.verbose;

    // The CLI never cancels; `Progress` only prints. `()` would also be a complete
    // implementation, and is what the tests pass.
    let run = Progress { verbose };

    let step = Instant::now();
    let registration = PhaseCorrelation::new(cli.registration_level);
    let transforms = register_stack(&registration, &stack.frames, &run)?;
    let (lo, hi) = transforms.iter().fold((f32::MAX, f32::MIN), |(lo, hi), t| {
        (lo.min(t.scale), hi.max(t.scale))
    });
    println!(
        "  register  {:>5.0}s   scale {lo:.4}..{hi:.4}",
        step.elapsed().as_secs_f32()
    );

    let by_path: HashMap<PathBuf, Transform> = stack
        .frames
        .iter()
        .cloned()
        .zip(transforms.iter().copied())
        .collect();

    let images: Vec<Image> = stack
        .frames
        .iter()
        .map(|p| Image::open(p))
        .collect::<stackaroni_core::error::Result<_>>()?;

    let fused = {
        let step = Instant::now();
        let metric = WindowedLaplacian::new(cli.focus_radius, scratch, by_path.clone());
        let focus_maps = evaluate_stack(&metric, &stack.frames, &run)?;
        println!("  focus     {:>5.0}s", step.elapsed().as_secs_f32());

        let step = Instant::now();
        let estimator = GuidedWeights::new(
            stack.frames.clone(),
            transforms.clone(),
            cli.guide_radius,
            cli.guide_epsilon,
            cli.guide_space.into(),
            scratch,
        );
        if let Some(dir) = debug_dir {
            debug::write_plane(
                &dir.join("labels_argmax.png"),
                &estimator.labels(&focus_maps, &run)?,
            )?;
        }
        let weights = estimator.weights(&focus_maps, &run)?;
        println!("  weights   {:>5.0}s", step.elapsed().as_secs_f32());

        if let Some(dir) = debug_dir {
            write_debug(dir, stack, &transforms, &focus_maps, &weights)?;
        }

        let step = Instant::now();
        // `--salience-radius` is a separate argument, so the rule is parsed first and
        // its parameter folded in here. `Blend` drops it, which is what the flag has
        // always documented — now enforced by the type rather than by a constructor
        // ignoring it.
        let fusion = cli.fusion.with_salience_radius(cli.salience_radius).build(
            output,
            by_path,
            cli.pyramid_floor,
        );
        let fused = fusion.fuse(&images, &weights, &run)?;
        println!("  fuse      {:>5.0}s", step.elapsed().as_secs_f32());
        fused
    };

    if let Some(dir) = debug_dir {
        debug::write_grid(&dir.join("fused.png"), &Grid::from_image(&fused, 0)?)?;
        println!("  debug output in {}", dir.display());
    }
    Ok(())
}

/// Per-stage debug images for a sample of frames.
///
/// Sampled rather than exhaustive: a 100-frame stack would otherwise write 300
/// images. First, middle and last cover the range, and the last frame carries the
/// largest accumulated transform, which is where alignment problems show first.
fn write_debug(
    dir: &Path,
    stack: &Stack,
    transforms: &[Transform],
    focus_maps: &[FocusMap],
    weights: &[stackaroni_core::image::ScratchPlane],
) -> Result<()> {
    let n = stack.frames.len();
    let anchor = n / 2;
    let samples = [0, anchor, n - 1];

    for &i in samples.iter().filter(|&&i| i < n) {
        debug::write_plane(&dir.join(format!("focus_{i:03}.png")), &focus_maps[i])?;
        debug::write_plane(&dir.join(format!("weight_{i:03}.png")), &weights[i])?;
    }

    // Alignment overlay against the anchor, drawn at the registration level.
    // Frame 0 carries the largest accumulated transform from a middle anchor.
    let far = 0;
    // Chosen for legibility, not tied to the registration level: at level 3 a
    // 1200 px stack yields a 150 px overlay, too small to see a misalignment in —
    // which defeats the point of dumping it.
    let info = stack.probe()?.info;
    let mut overlay_level = 0;
    while (info.width.max(info.height) >> overlay_level) > debug::MAX_EDGE {
        overlay_level += 1;
    }
    let reference = Grid::from_image(&Image::open(&stack.frames[anchor])?, overlay_level)?;
    let target = Grid::from_image(&Image::open(&stack.frames[far])?, overlay_level)?;

    // Transforms are in full-resolution pixels; the grids are downsampled by
    // 2^level, so the translation has to come down with them. Scale is a ratio and
    // is level-invariant. Getting this wrong warps by 8x at the default level and
    // reads as a wildly misaligned overlay rather than an obvious unit error.
    let shrink = (1u32 << overlay_level) as f32;
    let at_level = Transform {
        scale: transforms[far].scale,
        dx: transforms[far].dx / shrink,
        dy: transforms[far].dy / shrink,
    };
    debug::write_alignment_overlay(
        &dir.join(format!("align_{far:03}_vs_{anchor:03}.png")),
        &reference,
        &target,
        at_level,
    )?;
    Ok(())
}
