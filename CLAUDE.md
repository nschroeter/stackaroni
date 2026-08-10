# Project overview

A focus-stacking application for insect macro photography. Rust + egui/eframe, cross-platform
(macOS primary, Windows, Linux), single-binary distribution. I (the human) rate output image
quality; there is no automated ground truth, so treat my ratings and the checklist below as the
source of truth for "correct."

# Hard constraints

- Language: Rust.
- UI: egui/eframe. No web/Electron/other UI stack.
- Platforms: macOS (primary), Windows, Linux — avoid platform-specific dependencies unless
  unavoidable; flag them explicitly if introduced.
- Cargo workspace with three crates, under `crates/`:
  - `stackaroni-core` — pipeline/algorithm logic only, no UI or CLI dependencies.
  - `stackaroni-cli` — headless batch runner used for evaluation (see Evaluation workflow below).
  - `stackaroni-app` — egui/eframe UI, depends on `stackaroni-core`.

  Packages are prefixed because a package named literally `core` shadows Rust's built-in `core`
  crate at every `use` site in the other crates.

# Input assumptions

- Input is 16-bit TIFF, already denoised and developed by the user outside the app (from RAW).
  Do not build RAW decoding or denoising into this app.
- Decode 16-bit integer samples to `f32` immediately and do all pyramid/blend math in float,
  converting to linear light on decode. On output, re-encode to sRGB *before* quantizing back to
  16-bit — writing linear light into the TIFF makes the file look far too dark in any normal
  viewer, which would silently corrupt the visual rating loop rather than just look off.
- Stacks can be large (24-60MP x 30-100+ frames, 16-bit). Do not assume a full stack fits in
  memory at once — design for streaming/tiled processing rather than loading everything up front.

# Algorithmic policy — read this before implementing any pipeline stage

Implement named, published algorithms. Do not invent registration or fusion math from scratch,
and do not guess at an approach — research first, then implement, and say what you're
implementing and why (cite the technique/paper/existing implementation).

- Full rationale, formulas, and citations: `docs/algorithms.md`. Read it before touching
  registration, focus measurement, weight-map estimation, or fusion code.
- Recommended pipeline (see `docs/algorithms.md` for detail): phase-correlation translation →
  ECC or ORB+RANSAC affine registration → multi-scale focus measurement (windowed Laplacian /
  Scharr / Tenengrad) → edge-aware (later graph-cut) weight-map refinement → Laplacian-pyramid
  fusion → 16-bit TIFF output.
- Existing prior art worth reading before implementing the equivalent piece yourself:
  `eadf/libstacker.rs` (Rust, OpenCV-backed ECC/keypoint alignment), `PetteriAimonen/focus-stack`
  (C++, full pipeline reference), `kornia-rs` (pure-Rust CV primitives).
- Deep-learning approaches are out of scope for now — classical deterministic pipeline first.

# Pipeline architecture

Keep each stage independently replaceable via traits, so different algorithms can be
benchmarked against each other without rewriting the app:

```rust
trait Registration {
    fn align(&self, reference: &Image, target: &Image, run: &dyn RunControl) -> Result<Transform>;
}
trait FocusMetric {
    fn evaluate(&self, image: &Image, run: &dyn RunControl) -> Result<FocusMap>;
}
trait WeightEstimator {
    fn weights(&self, focus_maps: &[FocusMap], run: &dyn RunControl) -> Result<WeightMaps>;
}
trait ImageFusion {
    fn fuse(&self, images: &[Image], weights: &WeightMaps, run: &dyn RunControl) -> Result<Image>;
}
```

`RunControl` carries cancellation and progress together — `cancelled() -> bool` and
`progress(stage, done, total)`, both defaulted, so `()` is a complete implementation for
callers that do neither (the CLI passes a printing impl; tests pass `()`). One trait, not
two, because both need the same checkpoints: a stage that can report "frame 47 of 100" is
exactly a stage that can be stopped at frame 47.

It is a *method* parameter, not constructor-injected like an output path or guide images,
because it is a property of the call rather than of the stage, and because these traits
are a multiply-implemented public surface — an implementer who never sees it in the
signature has nothing telling them cancellation is expected. That is also why it is on
all four even though `align` and `evaluate` do not poll it today: each handles a single
frame, so their loops live in the caller, but ECC affine registration (§10 of
`docs/algorithms.md`) iterates *inside* `align`, and adding the parameter then would be a
breaking change.

Checks go at per-frame granularity inside `weights` and `fuse` — plus `labels` and
`normalize`, which are full banded passes over every plane before and after the per-frame
loop. Worst-case stop latency is therefore about one frame of fusion, ~6 s on a 50 MP
stack. **Sub-frame checks were considered and deliberately left out:** responsiveness is
dominated by whether the UI acknowledges the click immediately, not by true stop latency,
and a 6 s tail on a 20-minute operation reads as normal. Revisit only if it actually feels
slow. Never check inside the final `write_rgb16_srgb` — a truncated TIFF that looks like a
real output is worse than finishing the write.

`Error::Cancelled` is the one error variant that is not a fault. Callers should clean up
scratch on it rather than keeping it for inspection, and treat the run as never having
happened — there is no partial output to preserve, precisely because the write is never
interrupted.

`Result` is `stackaroni_core::error::Result`, over a typed `Error` enum rather than
`anyhow`. `core` is a library boundary consumed by both `cli` and `app`, and callers need
to tell failure kinds apart — "frame 47 failed to decode" (`Error::Decode`), "ran out of
scratch disk" (`Error::Scratch`) and "you pressed cancel" (`Error::Cancelled`) are
different situations for the user, and the app already branches on the last of those to
decide whether to keep the scratch directory. `cli` and `app` still use `anyhow` for their own internal
error handling; `core::Error` converts into it via `?`.

None of these types owns pixel data. `Image` reads bands from its TIFF on demand;
`FocusMap` and `WeightMaps` are mmapped scratch planes. So `&[Image]` over a
100-frame stack costs handles, not 60 GB — the streaming strategy lives inside each
implementation, never in these signatures. Extra state a stage needs (a fusion output
path, the guide images for edge-aware weighting) goes into the implementing type via
its constructor, not into the trait method.

Every stage must be able to dump a debug-visualizable intermediate output when run via the CLI:
alignment overlay/diff, focus-measure heatmap, weight map, and the final fused image. This is
how I diagnose problems without algorithm expertise — if something looks wrong, the debug output
tells us which stage to look at.

# UI design

Same principle as the algorithmic policy: don't invent UI/UX from scratch, and don't chase
visual polish over function. I'm not a designer any more than an image-processing researcher —
judge this the same way, by outcomes I can actually assess, not by aesthetic guesswork.

- Reference layout: filmstrip of frame thumbnails + large preview pane + parameter panel on the
  side. This is the established pattern for exactly this category of tool (Lightroom Develop
  module, darktable, Zerene Stacker, Helicon Focus) — follow it rather than designing a new
  layout from scratch.
- Required views:
  1. **Main stacking view** — load a folder of TIFFs, see thumbnails, pan and zoom them to
     judge sharpness, adjust the handful of exposed parameters, run, export.
- **Deliberately not built. Do not rebuild these from this file.** Both were specified here,
  reconsidered against something running, and dropped on purpose. Recording that is the point:
  a later reading of this document would otherwise find them missing and treat it as an
  omission — which is exactly how a checklist that is *supposed* to catch silent gaps turns
  into one that manufactures phantom work.
  - **A debug/diagnostic view.** The pipeline already writes per-stage output (alignment
    overlay, focus-measure heatmap, weight map) to `target/debug-out/`, and opening those files
    directly is no worse than a viewer inside the app would be. Dropped 2026-08-10.
  - **Previewing registration/focus-map output on a crop**, once part of view 1. Built, run,
    and removed the same day: a scale factor and an offset with no baseline to judge them
    against is information rather than an answer. Replaced by pan and zoom on the preview pane,
    which serves the underlying need — zoom into an antenna, step through frames, see which one
    resolves it. Full reasoning in `crates/app/src/main.rs`'s module docs.
- Use existing egui crates rather than hand-rolling: `egui_extras` (image display, tables), `rfd`
  (native file/folder dialogs). Only reach for `egui_dock` (dockable/movable panels) if a fixed
  layout turns out to be genuinely limiting in practice — start with a fixed layout, it's one
  less unknown.
- Keep UI code in the `stackaroni-app` crate only; it depends on `stackaroni-core`, never the
  reverse. UI iteration
  should never risk touching pipeline logic.
- After any layout change, take a screenshot of the running app and attach it back into the
  session so it can actually be seen and reviewed, the same way pipeline debug output gets
  reviewed. Don't reason about widget code blindly — this is a visual medium, treat it like one.
- Default to functional clarity over visual polish. egui looks clean and usable by default; a
  fully custom polished look (fonts, theming, spacing) is extra, deliberate work — not something
  to chase incidentally while building features, and not a priority for this project.

# Quality checklist — what "good" means

Rate every candidate output against this list (this is my rubric, not a proxy metric):

- No visible seams or halos around high-contrast edges (antennae, legs, hair boundaries).
- No ghosting from misalignment.
- Background bokeh outside the stacked region stays smooth — not patchily sharpened by
  focus-measure noise.
- Consistent color and exposure across the fused image.

# Evaluation workflow

- A fixed set of 3-5 representative test stacks lives at `test-data/` (gitignored; see
  `test-data/README.md`). Do not change this set when comparing algorithm versions — swap it out
  deliberately, not casually.
- Run the pipeline against the test set headlessly:
  `cargo run -p stackaroni-cli -- --test-set test-data --debug-out <path>`
  or against one stack: `cargo run -p stackaroni-cli -- --input test-data/<stack> --output <file>`
- After any change that could affect output quality, run the eval, look at the debug output and
  final images, and log the result in `docs/eval-log.md`: git commit hash, what changed, my
  score (1-5) against the checklist above, and notes. Don't skip logging — this is what lets us
  avoid re-litigating the same failed approach across sessions.
- `docs/eval-log.md` keeps a "Current state" block at the top — update it in the
  same commit as any row that changes what's current, not as a follow-up. Past
  ~15 rows, an append-only log with no current-state summary becomes unreadable.
- If an algorithm choice needs validating before a full Rust implementation, it's fine to
  prototype it quickly in Python/OpenCV against the test set first, confirm it scores well, and
  then port the validated approach into Rust — don't debug algorithm correctness and Rust
  ownership/type issues at the same time.

# Docs

- `docs/algorithms.md` — reviewed algorithm overview, formulas, and citations. Read before
  implementing or changing registration, focus measurement, or fusion.
- `docs/eval-log.md` — running log of experiments and scores. Create it if it doesn't exist yet;
  append to it, don't overwrite history.

# Build & test

- `cargo build`, `cargo test`, `cargo run -p stackaroni-app` (GUI),
  `cargo run -p stackaroni-cli -- ...` (headless eval).
- Run `cargo fmt` and `cargo clippy` before considering a change finished.

# Verifying changes

Use `Edit` for source changes, not scripted find-replace — `Edit` fails loudly when its
pattern doesn't match, scripted replaces fail silently. `cargo fmt` reformats code between
sessions, so a pattern that matched yesterday may not today. After changing behaviour,
verify with a freshly built binary; a stale `target/release` binary will happily reproduce
the old behaviour. Add a test covering the new case in the same change.

# Out of scope for now

- RAW decoding and denoising — handled by the user upstream (Lightroom/RawTherapee/DxO/etc.),
  the app only ever sees 16-bit TIFF input.
- Deep-learning fusion — revisit only after the classical pipeline is solid; see
  `docs/algorithms.md` §12 for context if this comes up later.
- Subagents, MCP servers, and hooks — not needed at this project's current scale.

