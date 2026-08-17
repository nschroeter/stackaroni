# Parameters

Every parameter the app and the CLI expose, what it does, and what moving it costs.

The defaults are not starting points — they are the values every rating in
[`eval-log.md`](eval-log.md) was given to. A run that touches none of them reproduces
those results. **Changing one invalidates the ratings**, which is a reason to experiment
freely on your own stacks and a reason not to change the defaults in the repository
without a fresh evaluation.

They live in one place, `crates/core/src/defaults.rs`, because the app and the CLI expose
the same knobs and two independent lists of numbers that happen to agree are
indistinguishable from one source of truth right up until someone edits one of them.

| Parameter | CLI flag | Default | Stage |
|---|---|---|---|
| Registration level | `--registration-level` | 3 | registration |
| Focus radius | `--focus-radius` | 4 | focus measurement |
| Guide radius | `--guide-radius` | 4 | weights |
| Guide epsilon | `--guide-epsilon` | 1e-4 | weights |
| Guide space | `--guide-space` | perceptual | weights |
| Fusion rule | `--fusion` | select | fusion |
| Salience radius | `--salience-radius` | 2 | fusion |
| Decomposition floor | `--pyramid-floor` | 32 | fusion |

---

## Registration level

**What it is.** Which pyramid level phase correlation runs at. Level 0 is full
resolution; each step halves both dimensions, so level 3 aligns on an image one eighth the
size in each direction.

**Why it is not 0.** Adjacent frames in a focus bracket do not share their
high-frequency content — what is sharp in one is defocused in the next — and phase
correlation whitens the spectrum, weighting every frequency equally. That mismatched
detail is *noise* in the correlation. Downsampling suppresses exactly those frequencies
and leaves the coarse structure the frames genuinely share to drive the peak.

**Raise it** if registration is unstable on very noisy or very shallow-DoF stacks.
**Lower it** for more positional precision, at the risk of the peak being driven by
detail the two frames do not share. Sub-pixel refinement means level 3 is not as coarse as
it sounds: translations are fitted to a parabola through the correlation peak.

## Focus radius

**What it is.** The window radius of the windowed-Laplacian focus measure — how much
neighbourhood is summed to decide how sharp a pixel is.

**The trade.** A bare per-pixel second derivative is extremely noise-sensitive, which
undercuts the robustness the measure exists for. Summing over a window fixes that, but a
window that straddles a depth discontinuity inherits energy from the sharp side, which is
the known cause of a ring of wrong labels at subject boundaries (recorded in the eval log
as an open item).

**Larger** is steadier on noisy stacks and blunter at edges. **Smaller** resolves fine
structure better and picks up more noise.

## Guide radius

**The single most consequential parameter.** The guided filter's window radius when
refining the weight maps.

**Too large** averages many frames into every pixel and destroys thin-structure contrast.
**Too small** lets argmax mottling through in defocused background. The best value is
data-dependent — a noisier stack spreads weight further at the same radius — which is why
it is exposed rather than fixed.

**Measured**, going from 8 to 4 on `blossom`: noise-floor-subtracted signal ratio
**0.32 → 0.45**, effective frames averaged per pixel **25 → 7.6**, and on `synthetic_50`
detail **0.195 → 0.320** against a per-pixel oracle of 0.353 — from 55% to 91% of the
physical maximum.

Effective frames is the clearest single readout of how selective the weights are: noise
falls as the square root of the number of frames averaged, so the drop in background noise
says how many frames each pixel is really made of. The ideal is 1–3, so 7.6 still leaves
headroom.

## Guide epsilon

**What it is.** The guided filter's regularisation — how much variance in the guide image
counts as an edge worth preserving rather than noise worth smoothing.

**Larger** smooths more, treating more of the guide's structure as noise. **Smaller**
preserves more edges, including ones that are only noise.

## Guide space

**What it is.** The tone space the guide image is measured in: `linear` or `perceptual`.

**Why perceptual is the default.** The pipeline works in linear light, where a
dark-on-mid edge has a far smaller numerical difference than a bright-on-mid edge of the
same visual contrast. A guided filter reading linear values therefore under-weights edges
in shadow — exactly where an insect's legs and antennae usually are, against a dark
background. Encoding the guide to sRGB before measuring restores their precedence.

## Fusion rule

**`select`** decides at every pyramid level and position which frame's coefficient to
take, from the coefficients themselves. **`blend`** blends every level under the weight
maps.

`select` is the default as of T11, on ratings across all three stacks: blossom **1 → 5**,
ruler **3 → 5**, synthetic_50 **5 → 4**. One change to the fusion rule, no parameter
retuning. `blend` remains available for reproducing eval-log rows from before that flip,
and is CLI-only.

**A consequence worth knowing:** under `select` the weight maps only reach the *coarsest*
pyramid level, because every band-pass level is re-decided from the coefficients. So the
guide parameters above have far less influence under `select` than under `blend` — the
eval log measures a whole weight-family swap moving 931 pixels out of 1.08M.

## Salience radius

**What it is.** The window radius over which `select` measures salience — local energy,
summed over a square window, rather than the bare coefficient magnitude at a pixel.

**Why a window at all.** A per-pixel argmax over Laplacian coefficients draws neighbouring
pixels from inconsistent sources and, on ISO-1600 frames, routinely selects noise. The
window makes an isolated spike lose to genuine surrounding structure.

Ignored by `blend`, which has no selection to make.

## Decomposition floor

**What it is.** The size at which the pyramid stops halving, which sets how many levels it
has.

**Why a size rather than a level count.** The test stacks span 900 to 5784 rows. A fixed
level count would make the coarsest level represent a different physical scale on each
stack, and cross-stack comparisons would then be conflating algorithm behaviour with
resolution.

**Lower** means more levels and a coarsest level carrying broader structure. **Higher**
means fewer.

---

## Memory limit

Not a pipeline parameter — it changes nothing about the output — but it is the one thing
that can stop a run before it starts, so it belongs here.

Before a run, the frame headers are read and a peak-memory estimate is computed from frame
size, strip layout, core count and pyramid depth. If it exceeds the limit, parallelism is
lowered until it fits; the run simply takes longer, silently. Only when even one frame at a
time exceeds the limit does anything appear: the app shows a dialog with **Run anyway**, and
the CLI refuses unless given `--ignore-memory-limit`.

The limit is `max(16 GB, 25% of physical RAM)`, capped at 90% of RAM so it cannot exceed the
machine it is protecting. On a 36 GB or 64 GB machine that is 16 GB; on 128 GB it is 32 GB.

**Strip layout dominates the estimate.** A frame written as one strip is fully resident
while its rows are read — 300 MB at 50 MP against ~13 MB for the same pixels in 64-row
strips — so re-exporting with strips is the cheapest fix when the warning appears.

**The estimate is a model, not a measurement**, fitted to four measured runs and required to
never under-predict them. It can be wrong, which is why the override exists rather than a
refusal.

## Changing them

Every parameter is exposed in both front ends because the useful values are
data-dependent. Nothing here is a secret setting for experts — the defaults are simply
what scored best on one particular set of stacks.

If you find better values for your own subjects, that is the system working as intended.
If you find better values on *these* test stacks, that is an eval-log row: see
[`eval-log.md`](eval-log.md) for the format and the rating checklist.
