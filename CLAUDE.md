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
    fn align(&self, reference: &Image, target: &Image) -> Transform;
}
trait FocusMetric {
    fn evaluate(&self, image: &Image) -> FocusMap;
}
trait WeightEstimator {
    fn weights(&self, focus_maps: &[FocusMap]) -> WeightMaps;
}
trait ImageFusion {
    fn fuse(&self, images: &[Image], weights: &WeightMaps) -> Image;
}
```

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
  1. **Main stacking view** — load a folder of TIFFs, see thumbnails, preview
     registration/focus-map output on a crop before running the full stack, adjust the handful
     of exposed parameters, run, export.
  2. **Debug/diagnostic view** — a place to inspect the per-stage debug output the pipeline
     already produces (alignment overlay, focus-measure heatmap, weight map), not just the
     final fused image. This is the visual equivalent of the "Pipeline architecture" debug-output
     requirement above — it needs a home in the UI, not just on disk.
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

# Out of scope for now

- RAW decoding and denoising — handled by the user upstream (Lightroom/RawTherapee/DxO/etc.),
  the app only ever sees 16-bit TIFF input.
- Deep-learning fusion — revisit only after the classical pipeline is solid; see
  `docs/algorithms.md` §12 for context if this comes up later.
- Subagents, MCP servers, and hooks — not needed at this project's current scale.

