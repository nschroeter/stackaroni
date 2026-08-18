# Changelog

Hand-written, one section per release. The section whose heading matches the workspace
version becomes the GitHub release notes — `.github/workflows/release.yml` extracts it and
appends the download and installation block from `.github/release-notes.md`.

**Hand-written on purpose.** Generating this from commit subjects was considered and
rejected: this repository has one author and no pull requests, so a generator would
produce a list of subject lines under headings, and the part worth reading — what the
output looks like now, and what a change cost — lives in commit bodies and in
`docs/eval-log.md` where no generator can reach it.

Sections in a release, all optional except the first line: 🚀 Features, 🩹 Fixes,
🔬 Quality, 🧹 Removed, 📖 Docs. Scope in bold, then what changed.

`crates/core/tests/changelog.rs` fails the build if the current version has no section
here, for the same reason `parameters_doc.rs` exists.

`## [Unreleased]` collects what has landed on `main` since the last release. The workflow's
`awk` matches the version literally, so the heading is inert until it is renamed to the
version being cut — which is the one step that must not be forgotten when raising the
version in `Cargo.toml`.

## [Unreleased]

### 🔬 Quality

**Fusion is a further 10% faster and 0.5 GB smaller, output unchanged.** Building a
band-pass pyramid level expanded the level below it into a full-resolution image, then
subtracted that image and dropped it — about 1.2 GB of traffic per frame at 50 MP for a
value used exactly once. The expansion now hands its samples straight to the subtraction.
Measured on blossom, 100 frames, alternating against the previous build: **fuse 49-52 s →
44-46 s, peak 3.8-4.3 GB → 3.4-3.6 GB**, with no run of one overlapping the other. Pixels
identical, confirmed with `tiffcmp`.

**The memory estimate is re-fitted to what the pipeline now costs.** Fusion stopped copying
planes in 1.2.0, so striped stacks peak at 4.4-4.7 GB where the model still expected
6.2-6.3 GB, and it over-predicted them by around 40%. That direction is the safe one — the
estimate exists to warn before a run thrashes, and it never under-predicted — but a model
that over-warns eventually gets ignored. All four calibration configurations were
re-measured and the two fitted constants moved with them.

Single-strip stacks are unchanged, as expected: their peak is registration holding decoded
frames, which that work never touched.

### 🩹 Fixes

**Output TIFFs now carry an sRGB ICC profile.** They were written as sRGB and said nothing
about it, so a viewer had to guess. Most guess sRGB and land on the right answer, but a
colour-managed application assumes its own working space instead — which is how a correct
file ends up displayed wrong, and it made the ratings this project runs on partly a
property of the viewer.

The profile is generated in code rather than shipped as a binary asset: an OS profile is
not ours to redistribute, and a downloaded blob is not reviewable in a diff. It is a
display-class matrix/TRC profile with the published sRGB primaries and white point, and a
tone curve sampled from the same transfer function the decoder applies — so the file cannot
describe a curve the pipeline does not use.

**No pixel changes**, confirmed with `tiffcmp` on synthetic_50 and on a 100-frame blossom
run. The regression gate now hashes decoded pixels rather than file bytes, because adding
metadata moved a hash whose whole purpose is to catch pipeline changes; it separately
asserts the profile is present.

## [1.2.0] — 2026-08-18

### 🚀 Features

**Windows gets a real menu bar** — the same Help menu macOS has beside the Apple logo,
drawn by the OS under the titlebar rather than as a button in the toolbar. `muda` was
already in the tree for macOS and ships the win32 backend; it attaches an `HMENU` to the
window eframe created and reports clicks on the channel the app already drains.

**Linux keeps the in-window menu, deliberately.** `muda`'s Linux backend is gtk-only and
eframe creates X11/Wayland surfaces directly, so there is no gtk window to attach to — and
Linux desktops expect the menu drawn in-window anyway. The toolbar menu is also the
fallback on Windows if the install fails, so no platform can end up with no way to reach
About or the parameter reference.

### 🔬 Quality

**Fusion is 2.4x faster and uses a quarter less memory, and every pixel it produces is
unchanged.** Measured on blossom, 100 frames of 8664x5784 16-bit RGB: **fuse 114 s → 47 s,
whole run 205 s → 109 s, peak memory 5.8-5.9 GB → 4.2-4.4 GB.** Ruler produced its result
in 133 s against 197 s. All three test stacks were re-run and verified byte-identical to
1.1.1 output, then re-rated 5 / 5 / 5 against the quality checklist. Three changes, all
removing overhead rather than approximating anything:

- **`box_sum`, the windowed sums behind salience, runs across threads.** Profiled, it was
  **48.7% of the fuse stage on one thread while thirteen others waited**, and its vertical
  pass walked columns at stride `width`, pulling a cache line to use four bytes of it. Rows
  are now summed in parallel and columns in blocks of 256 that sweep downwards with one
  accumulator each, so source and destination are both read forwards.
- **The weight plane is reduced straight from its mapped rows.** Selection fusion needs
  each frame's weights at the coarsest pyramid level only, and it was copying the
  full-resolution plane into an owned image first — an allocation and a 200 MB memcpy per
  frame that nothing else ever read.
- **Pyramid levels are moved rather than copied.** Building one Laplacian pyramid copied
  the full-resolution frame twice — as the base of the Gaussian pyramid, and as the image
  each band is subtracted from — about 1.2 GB per frame at 50 MP, both copies dropped
  moments later.

**Why "byte-identical" is the headline rather than a footnote.** Float addition is not
associative, so the obvious faster shapes — reassociating a running sum, splitting a
column's accumulator across threads — would each have changed the image. Every column and
every kernel tap is still summed in the order it was, which is what lets the ratings above
carry over rather than needing to be earned again.

**Stacks whose peak is registration rather than fusion — single-strip input — are
unchanged**, in both time and memory, as expected.

### 📖 Docs

**Rejected experiments are kept as annotated tags, not branches** — the branch list stays
readable, the code stays reachable via `git show <tag>`. One such tag exists:
`t12-multiscale-focus`, the multi-scale focus measure built and measured inert in T12
(`docs/algorithms.md` §4, `docs/eval-log.md`). Every other tag here is a `vX.Y.Z` release.

**Memory limit** — `CHANGELOG.md` and `docs/PARAMETERS.md` now say where the 90%-of-RAM
clamp on the limit actually binds: below ~18 GB of RAM, where the 16 GB floor would
otherwise exceed the machine's own memory and no run could ever trip the warning.

## [1.1.1] — 2026-08-17

### 🩹 Fixes

**Memory warning could stay silent when it should not.** The peak-memory estimate shipped in
1.1.0 was fitted to measurements taken on a single day, and peak memory is not repeatable to
better than ~10% — the same binary on the same 33 frames measured 10.1 GB one day and 11.3 GB
the next, with no code change in between. The estimate therefore under-predicted by about 7%,
which is exactly the direction that matters: a warning that cannot fire when memory is short
is worse than no warning.

The estimate now carries 15% headroom, and its calibration is held against every measured
run of each configuration rather than one apiece.

No pipeline change: fused output is identical to 1.1.0.

## [1.1.0] — 2026-08-17

### 🚀 Features

**Memory limit with an override.** Before a run starts, the frame headers are read and peak
memory is predicted from frame size, strip layout, core count and pyramid depth. If the
prediction exceeds the limit, parallelism drops until it fits — the run takes longer and
says nothing. Only when even one frame at a time will not fit does anything appear: the app
offers **Run anyway**, and the CLI refuses unless given `--ignore-memory-limit`.

The limit is `max(16 GB, 25% of RAM)`, then clamped to 90% of RAM. The clamp only matters
below ~18 GB of RAM, where the 16 GB floor would otherwise exceed the machine's own memory
and no run could ever trip the warning — on an 8 GB machine the limit becomes 7.2 GB. Above
that it never binds. Single-strip frames dominate the estimate — one is fully resident while its
rows are read — so re-exporting with strips is the cheapest fix when the warning appears.

**It is a warning, not a refusal, because the estimate is a model**, fitted to four measured
runs and required never to under-predict them. It can be wrong, and you know your machine.

## [1.0.3] — 2026-08-16

### 🔬 Quality

**tiff, fusion** — single-strip frames no longer make memory grow with the size of the
stack, and everything got faster. Measured on 33 frames of 8664x5784 16-bit RGB written
as one strip each: **102.8 s → 52.0 s, peak memory 25.0 GB → 10.1 GB.** The striped
layout the app was developed against improved too, 75.5 s → 60.0 s.

Three changes. Decoded strips are kept as 16-bit samples rather than expanded to float,
which halves what one frame in flight costs and — via a lookup table for the sRGB
transfer function — removes the per-sample `powf` that dominated decoding. Fusion now
releases a frame's decoded data as soon as it has used it, instead of holding all of them
until the stage ends. And the stages that work on several frames at once size their
parallelism to the machine's RAM, so a 16 GB machine runs fewer frames concurrently
rather than running out.

**Fused output is byte-identical**: the pinned regression hash did not move, and the
lookup table is asserted exact against the function it replaces for all 65536 possible
sample values.

**This supersedes the 1.0.2 note below.** Re-exporting with strips is no longer the
remedy for a large single-strip stack, though striped input is still cheaper to read.

## [1.0.2] — 2026-08-16

### 🩹 Fixes

**tiff** — frames written as a *single strip* failed to load at all, with "decoding
failed: decoder limits exceeded" on every frame. The `tiff` crate caps one chunk at
256 MB by default, which a 48 MP 16-bit RGB frame exceeds the moment an exporter writes
the image in one piece rather than in strips. The stacks this was developed against are
one row per strip, so nothing here ever hit it. Reported against the Windows build and
reproduced immediately on macOS — it was never platform-specific.

The budget is now sized to the file rather than removed, so a corrupt header still
cannot ask for an arbitrary allocation.

**Single-strip frames cost more memory**, unavoidably: a strip is the decoder's unit, so
the whole frame is resident while its rows are read. Measured on eight 48 MP frames,
peak RSS 14.4 GB against 8.5 GB for the same pixels in 64-row strips. Re-exporting with
strips is worth it if memory is tight.

## [1.0.1] — 2026-08-16

An icon, and nothing else. **No pipeline change: fused output is identical to 1.0.0**, so
there is no reason to re-run anything you have already stacked.

### 🚀 Features

**icon** — an app icon: a mantis over three stacked planes. `AppIcon.icns` in the macOS
bundle, built from one PNG at release time; embedded as the window and taskbar icon on
Windows and Linux; and compiled into `stackaroni-app.exe` as a resource so Explorer draws
it for the executable too. `packaging/icon/README.md` has the provenance.

## [1.0.0] — 2026-08-16

First release. A complete classical focus-stacking pipeline with a GUI, a headless runner,
and a record of every measurement behind it.

### 🚀 Features

**pipeline** — registration → focus measurement → weight estimation → fusion, each stage
a swappable trait. Phase correlation with a log-polar scale estimate (Kuglin & Hines 1975;
Reddy & Chatterji 1996), windowed-Laplacian focus measurement, guided-filter weight
refinement (He, Sun & Tang), and Laplacian-pyramid fusion (Burt & Adelson 1983) combined
per level by the selection rule of Burt & Kolczynski (1993).

**app** — filmstrip, pan-and-zoom preview for judging sharpness frame by frame, the eight
exposed parameters, run and export. Native macOS menu bar with the parameter reference and
an About dialog.

**cli** — headless runs over one stack or a whole test set, with `--debug-out` writing
per-stage diagnostics: alignment overlay, focus heatmaps, the argmax label field, weight
maps and a fused preview. That is how a problem gets localised to a stage.

**streaming** — a 100-frame 50 MP stack never sits in memory at once. Frames are read from
their TIFFs on demand and intermediates are mmapped scratch planes, so a stack costs file
handles rather than 60 GB. A full run at that size takes about 4½ minutes.

**cancellation** — every stage takes a `RunControl` carrying progress and a cancel flag,
checked at per-frame granularity. The final TIFF write is never interrupted: a truncated
file that looks like a real output is worse than finishing.

### 🔬 Quality

Rated by eye against the four-point checklist in `CLAUDE.md`, on the fixed test stacks.
There is no automated ground truth for a photograph, so these are human scores, and
`docs/eval-log.md` carries the reasoning behind each one.

| Stack | Score |
|---|---|
| synthetic_50 | 5 / 5 |
| ruler | 5 / 5 |
| blossom | 5 / 5 |

**Output is never cropped.** The full field of view survives the pipeline; frames compete
only inside the region they can actually fill, so border-replicated pixels cannot be
selected without costing framing. The remaining softness in blossom's upper-right corner
is the capture — no input frame resolves it.

A hash gate pins one stack's fused output byte-for-byte, so a change claiming to be a pure
speedup can be checked rather than believed.

### 📖 Docs

**eval-log** — every experiment, its score and its reasoning, including the failures:
wavelet-domain stacking built and removed, a published fix for its defect measured inert
and discarded, depth-map weights rated and dropped, multi-scale focus measurement shown to
change nothing. The negative results are most of the value — they are what stops the same
idea being rebuilt.

**PARAMETERS.md** — every exposed parameter, which stage owns it, and what moving it
costs, with measured effects where they were measured.

**algorithms.md** — the reviewed algorithm overview, formulas and citations each stage
implements.
