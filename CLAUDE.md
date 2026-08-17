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

**Four stages, and choosing an algorithm means choosing one of them.** Registration, focus
measurement, weight estimation and fusion are separate, sequential slots. A "fusion rule" is
one implementation of the fourth slot; picking `select` over `blend` changes nothing about
how frames were aligned, how sharpness was measured, or how weights were refined — those
three run identically either way, and the same holds in the other direction for any future
registration or focus-measure alternative.

This matters for two things that keep coming up. **In the UI**, an algorithm chooser belongs
to a stage, not to the pipeline: a dropdown offering "Pyramid" as if it were a whole-pipeline
method misdescribes what changes when you use it. **In `docs/eval-log.md`**, a row that
swaps one stage's implementation leaves the other three stages' findings intact — which is
why the T11 fusion rows did not invalidate the T7 guided-filter measurements, and why a
future registration change will not invalidate any fusion rating.

**The stage-3/stage-4 boundary leaks, in the shipped default — know this before relying on
the independence claim above.** `SelectionFusion::fuse` (`crates/core/src/fusion.rs:491`)
consumes the `weights` argument for the *coarsest* pyramid level only. Every band-pass level
is decided inside `fuse` by `select_more_salient`, which computes its own windowed salience
over the Laplacian coefficients — so under `select`, the weight maps `GuidedWeights`
produced barely reach the output, and fusion re-does part of stage 2's job on its own
measure. This is not an accident to repair: local salience *is* the mechanism that took
blossom from 1 to 5 in T11, and `select`'s decision has to be made per pyramid level, where
a single per-frame weight plane cannot express it.

Two consequences, both live. **The "swapping one stage leaves the others' findings intact"
claim is weaker between stages 3 and 4 than it reads.** A guided-filter measurement taken
under `blend` describes a stage whose output `select` mostly discards; T7's numbers survive
as measurements of the weight stage, not as statements about the current fused image.
**And a wavelet-domain or guided-filter-as-fusion method (`docs/algorithms.md` §5, §9)
collapses stages 2-4 entirely** — there is no per-frame `FocusMap` to hand a
`WeightEstimator`. The four traits describe the classical focus-map pipeline they were drawn
from, and `select` is already the first implementation straining that shape. If a third
fusion rule strains it again, the question to settle is whether the contract is "weights
drive fusion" or "fusion may re-measure" — decide it deliberately, don't let a third
implementation settle it by accident.

**This was demonstrated and then removed, which is worth knowing before rebuilding it.**
T14 built the wavelet method and it needed a fifth trait, `StackFusion`, for exactly the
reason above. T17 deleted the method — it rated 2/4/2 against `local`'s 5/5/5 — and
`StackFusion` and the `Method` chooser went with it, so **the demonstration now survives
only as this paragraph and the eval-log rows.** The finding stands; the code that proved it
does not. If a future method collapses the stages again, expect to reintroduce a trait like
`StackFusion` rather than to force it into the four, and read the T14, T16 and T17 rows
first — the wavelet path was rated, diagnosed (defocus spread effect), had a published fix
implemented and measured inert, and was then removed. That is a closed line of work, not an
open one.

Keep each stage independently replaceable via traits, so different algorithms can be
benchmarked against each other without rewriting the app:

```rust
trait Registration: Sync {
    fn align(&self, reference: &Image, target: &Image, run: &dyn RunControl) -> Result<Transform>;
}
trait FocusMetric: Sync {
    fn evaluate(&self, image: &Image, run: &dyn RunControl) -> Result<FocusMap>;
}
trait WeightEstimator: Sync {
    fn weights(&self, focus_maps: &[FocusMap], run: &dyn RunControl) -> Result<WeightMaps>;
}
trait ImageFusion: Sync {
    fn fuse(&self, images: &[Image], weights: &WeightMaps, run: &dyn RunControl) -> Result<Image>;
}
```

`Sync` is a requirement, not decoration. Stages run across threads: `register_stack` aligns
every pair concurrently and `evaluate_stack` measures every frame concurrently, both in
`core` rather than in each caller. An implementation with thread-unsafe interior mutability
is not a valid stage. `Image` already holds its decoder behind a `Mutex` so handles stay
`Sync` through it.

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
stack, measured at 6.82 s on blossom.

**Parallelism changed what cancellation guarantees, in the stages that went wide.** Where a
loop runs across threads, every frame already in flight finishes before the flag is seen, so
the guarantee is "no *new* work starts", not "stops at frame N". Fusion is the exception and
deliberately so: its per-frame loop stays sequential, because float addition into the
accumulator is not associative and going wide there would change the output. That is also
why fusion's cancellation tests can still pin an exact frame and the weights test cannot. **Sub-frame checks were considered and deliberately left out:** responsiveness is
dominated by whether the UI acknowledges the click immediately, not by true stop latency,
and a 6 s tail on a 20-minute operation reads as normal. Revisit only if it actually feels
slow. Never check inside the final `write_rgb16_srgb` — a truncated TIFF that looks like a
real output is worse than finishing the write.

**How wide those stages go is a memory decision, not a core count** (`crates/core/src/budget.rs`,
T18-T19). A strip is the TIFF decoder's atomic unit, so a frame written as a *single* strip — common
from exporters — is entirely resident while its rows are read: 300 MB against ~13 MB for the same
pixels in 64-row strips. Registration holds two such readers per task, so 14 cores would start
8.4 GB of frame cache. The three parallel stages therefore run inside a rayon pool sized to a
share of physical RAM. **Capping concurrency, never evicting caches** — a shared LRU below one
strip would re-decode 300 MB per row read, turning a memory problem into a far worse time one.
On striped input the cap does not bind and scheduling is unchanged, which is what keeps every
rating in `docs/eval-log.md` valid. Fusion is exempt: it is sequential and releases each frame's
cache as it finishes with it, so it is O(one frame) regardless of stack depth.

**T19 turned that into a predicted budget the user can see.** `budget::estimate` predicts peak
memory from headers alone — frame size, strip layout, core count, pyramid depth — and `fit` picks
the widest parallelism that stays under `max(16 GB, 25% of RAM)`, clamped to 90% of RAM. Two things
about it are load-bearing. **It is a `max` over stages, never a sum**, because stages run in
sequence and the measurements prove a summing model over-predicts by nearly 2x. And **it warns
rather than refuses** — the CLI takes `--ignore-memory-limit`, the app offers "Run anyway" — because
three of its constants are fitted to four measured runs rather than derived, so it can be wrong in
either direction. The gate that keeps it honest is `the_estimate_brackets_every_measured_run`: clear the *highest*
measurement of each configuration, stay within 35% of the *lowest*. Adding a stage, or changing what
one allocates per thread, means re-fitting against that test rather than reasoning about it.

**Peak memory is not repeatable to better than ~10%, and the test lists several runs per
configuration because of it.** The same binary on the same 33 single-strip frames measured
10.096 GB one day and 11.329 GB the next. Peak footprint counts dirty mmapped scratch pages, and
how many are resident when a stage peaks depends on the machine's memory pressure at the time — an
input to every measurement here that cannot be controlled for. A model fitted to one day's number
under-predicted the next day's by 0.7 GB, which is why `DRIFT_MARGIN` exists. **Never compare a
memory measurement against one taken on another day and conclude anything from a difference under
about a gigabyte** — that mistake produced a "the app costs 0.68 GB more than the CLI" finding that
did not survive re-measurement.

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
- Use existing egui crates rather than hand-rolling: `rfd` for native file/folder dialogs, in
  use. Only reach for `egui_dock` (dockable/movable panels) if a fixed layout turns out to be
  genuinely limiting in practice — start with a fixed layout, it's one less unknown.
- **`egui_extras` was recommended here and deliberately never added.** Not a gap to close, and
  adding it now would mean an unused dependency. Its image loaders decode *encoded* bytes
  (PNG/JPEG) behind a URI, whereas frames are 16-bit TIFF decoded through `core` into raw
  pixels and uploaded with `ctx.load_texture` — a path that has nothing to hand a URI-based
  loader. Its other draw, `TableBuilder`, has no use in the current UI. The case for it will
  return if something ever needs to display encoded images or a real table; until then it
  would be weight for nothing.
- **The menu bar is native where the OS has one, in-window where it does not** — `muda` builds
  an `NSMenu` on macOS and an `HMENU` on Windows; Linux draws the same two items as an egui
  menu in the toolbar. That is not an unfinished third platform: `muda`'s Linux backend is
  gtk-only and eframe creates X11/Wayland surfaces directly, so there is no gtk window to
  attach to, and Linux desktops expect an in-window menu anyway. The toolbar menu doubles as
  the fallback when a native install fails, so it stays compiled off macOS.
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

