//! Headless runner used for evaluation. See the "Evaluation workflow" section of
//! `CLAUDE.md`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use stackaroni_core::discovery::{Stack, discover_stack, discover_test_set};

#[derive(Parser)]
#[command(about = "Focus stacking for insect macro photography", version)]
struct Cli {
    /// Directory holding one stack's frames.
    #[arg(long, value_name = "DIR", required_unless_present = "test_set")]
    input: Option<PathBuf>,

    /// Directory holding several stack directories; every stack is processed.
    #[arg(long, value_name = "DIR", conflicts_with = "input")]
    test_set: Option<PathBuf>,

    /// List every frame instead of just the per-stack summary.
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let stacks = match (&cli.input, &cli.test_set) {
        (Some(dir), _) => vec![discover_stack(dir)?],
        (_, Some(root)) => discover_test_set(root)?,
        _ => unreachable!("clap requires one of --input or --test-set"),
    };

    for stack in &stacks {
        report(stack, cli.verbose)?;
    }
    Ok(())
}

fn report(stack: &Stack, verbose: bool) -> Result<()> {
    let probe = stack.probe()?;

    if verbose {
        for (path, info) in &probe.frames {
            println!(
                "  {:<28} {}x{}  {}-bit  {}ch",
                file_name(path),
                info.width,
                info.height,
                info.bits_per_sample,
                info.samples
            );
        }
    }

    let info = probe.info;
    println!(
        "{:<14} {:>3} frames  {}x{}  {}-bit RGB",
        stack.name,
        probe.frames.len(),
        info.width,
        info.height,
        info.bits_per_sample
    );
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
