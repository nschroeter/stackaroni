# Stackaroni

Focus stacking for insect macro photography. Rust, `egui`, single binary, no runtime
dependencies.

Takes a folder of 16-bit TIFF frames shot at stepped focus distances and produces one
image with everything sharp — aligning the frames, measuring where each is in focus, and
combining them without seams, halos or ghosting.

A single source frame — only the near end of the ruler is in focus:

![The Stackaroni window showing one source frame of a 100-frame stack, sharp at the near
end and blurred along the rest of the ruler](docs/images/screenshot.jpg)

The fused result from all 100 frames — sharp end to end:

![The same window showing the fused result, with the whole ruler in focus from the metal
tip to the far markings](docs/images/screenshot-result.jpg)

> **Status: it works and is in use, but it is one person's tool.** Output is judged by eye
> against a fixed checklist on a fixed set of test stacks; the ratings and the reasoning
> behind every change are in [`docs/eval-log.md`](docs/eval-log.md).

## Download

[**Latest release**](https://github.com/nschroeter/stackaroni/releases/latest) — macOS
(Apple Silicon), Linux x86-64, Windows x86-64. Each archive holds the GUI, the
`stackaroni-cli` headless runner and the docs, and `SHA256SUMS` covers all three.

**The builds are not code-signed**, so both systems object the first time. On macOS,
right-click `Stackaroni.app` → **Open** → **Open** (or
`xattr -d com.apple.quarantine Stackaroni.app`). On Windows, SmartScreen's **More info**
→ **Run anyway**.

Intel Macs and older Linux distributions are not covered — the macOS build is arm64 only
and the Linux binary needs glibc 2.39 or newer (Ubuntu 24.04 and up). Build from source
for those.

## What it does

```
registration  →  focus measurement  →  weight estimation  →  fusion
```

- **Registration** — phase correlation with a log-polar scale estimate (Kuglin & Hines
  1975; Reddy & Chatterji 1996). Focus breathing changes magnification by ~0.1% per frame,
  so translation alone is not enough; scale is estimated and corrected.
- **Focus measurement** — windowed Laplacian energy per frame.
- **Weight estimation** — edge-aware refinement with a guided filter (He, Sun & Tang).
- **Fusion** — Laplacian pyramid (Burt & Adelson 1983), combined per level by the
  selection rule of Burt & Kolczynski (1993).

Frames are streamed rather than loaded: a 100-frame 50 MP stack never sits in memory at
once. A full run on that size takes about 4½ minutes.

## Building from source

Needs a recent Rust toolchain — the workspace is on edition 2024.

```sh
cargo build --release
```

Two binaries:

```sh
cargo run --release -p stackaroni-app          # the GUI
cargo run --release -p stackaroni-cli -- ...   # headless, for batch runs and evaluation
```

Linux additionally needs the usual windowing and GL headers for `eframe`
(`libxkbcommon-dev`, `libwayland-dev`, `libx11-dev`, `libxcursor-dev`, `libxi-dev`,
`libxrandr-dev`, `libxcb-*-dev`, `libgl1-mesa-dev`). See `.github/workflows/ci.yml`, which
builds for Linux, Windows and macOS.

## Using it

**The app**: open a folder of frames, page through the filmstrip, zoom into an antenna to
see which frame resolves it, adjust parameters, run, export.

**The CLI**, for one stack or a whole test set:

```sh
stackaroni-cli --input frames/ --output stacked.tif
stackaroni-cli --test-set test-data --output out/ --debug-out debug/
```

`--debug-out` writes per-stage diagnostics — alignment overlay, focus heatmaps, the argmax
label field, weight maps and a fused preview. That is how problems get localised to a
stage without reading the code.

Parameters are exposed rather than hidden, because the useful values are data-dependent.
**[`docs/PARAMETERS.md`](docs/PARAMETERS.md) describes each one** — what it does, which
stage it belongs to, and what moving it costs, with the measured effects where they were
measured.

## Input assumptions

16-bit TIFF, already developed and denoised from RAW in whatever you normally use. RAW
decoding and denoising are deliberately out of scope. Samples are converted to linear
light on decode and re-encoded to sRGB on write.

**One folder per stack, and the filenames set the order.** Every frame of a stack lives in
its own directory, and frames are sorted in *natural* order — runs of digits compared as
numbers, everything else by bytes. That order is taken to be the focus order.

So `frame_1.tif … frame_100.tif` and `frame_001.tif … frame_100.tif` both work: `frame_2`
comes before `frame_10` either way. Padding is not required, and mixed padding still gives
a stable, repeatable order.

**What is *not* inferred:** nothing reads EXIF or focus distance. If the filenames do not
run in focus order, neither will the stack, and it will not complain — registration chains
outward from the middle frame assuming adjacent files are adjacent focus positions, so a
scrambled order produces a badly aligned image rather than an error.

**Keep other TIFFs out of the folder.** Any `.tif`/`.tiff` in the directory is treated as a
frame, apart from a short list of known non-frames. Saving a fused result next to its own
source frames would feed it back in as an extra frame on the next run — so the app and the
CLI both refuse to write output into the directory they are stacking from.

## How quality is judged

There is no automated ground truth for a photograph, so **a human rates every candidate
output** against a fixed checklist:

- no visible seams or halos around high-contrast edges — antennae, legs, hair boundaries
- no ghosting from misalignment
- background bokeh stays smooth, not patchily sharpened by focus-measure noise
- consistent colour and exposure across the fused image

A fixed set of test stacks lives in `test-data/` (gitignored — they are hundreds of MB;
`test-data/README.md` describes them). Every change that could affect output quality is
run against all of them, rated, and recorded in [`docs/eval-log.md`](docs/eval-log.md)
with the commit hash.

A hash gate (`crates/core/tests/output_is_stable.rs`) pins the fused output of one stack
byte-for-byte, so a change claiming to be a pure speedup can be checked rather than
believed.

## The eval log is the interesting part

[`docs/eval-log.md`](docs/eval-log.md) is a running record of what was tried, what it
scored, and why. It is unusually candid about failure, and deliberately so: **most of its
value is the negative results**, which are what stop the same idea being rebuilt.

Things in there that were built, measured, and removed or abandoned:

- **Wavelet-domain stacking** (Li, Manjunath & Mitra 1995) — implemented in full, rated
  2/4/2 against the Laplacian path's 5/5/5, removed. Its architectural finding survives:
  it needed a fifth pipeline stage, so the four-stage decomposition is not universal.
- **A published fix for its defect** — the defocus spread effect, corrected per Kim (2023).
  Implemented faithfully, measured inert, discarded.
- **Depth-map weights** (cost-volume filtering, Rhemann & Hosni et al. 2011/2013) — the
  second family of focus stacking, rated 2/3/1, discarded.
- **Multi-scale focus measurement** — built, measured to change nothing on this data.

Also recorded: the measurement mistakes. Reading a Laplacian standard deviation as
sharpness when it was counting noise; describing a change as a strict improvement when it
was slightly worse on one stack. The log corrects itself in place and says so.

`docs/algorithms.md` is the reviewed algorithm overview with the formulas and citations
each stage implements.

## Why "Stackaroni"

A nod to [TC Zwag](https://www.youtube.com/@Zwag/).

## Design notes

`CLAUDE.md` at the repository root is the project's standing brief — architecture,
constraints, and a list of things deliberately *not* built with the reasoning for each. It
is written for whoever works on this next, human or otherwise.

## Licence

MIT. See [LICENSE](LICENSE).
