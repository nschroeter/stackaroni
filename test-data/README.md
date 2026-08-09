# Test data

Fixed set of test stacks used for evaluation (see `docs/eval-log.md` and the
"Evaluation workflow" section of `CLAUDE.md`). Not tracked in git — see
`.gitignore`. Don't change or replace a stack casually once it's in use for
comparisons; note it explicitly in `docs/eval-log.md` if you do.

## ruler

- **Subject:** Yellow rules on white table
- **Type:** Real photo, flat target
- **Frame count:** 100
- **Resolution / bit depth:** 8664x5784, 16-bit RGB TIFF
- **Camera / lens / magnification:** Sony Alpha 1 Mark 1, Sony FE 2.8/100mm Macro, 1.4x
- **Why it's a test case:** Isolates registration accuracy without occlusion/depth confounds, ISO 100 for least amount of noise
- **Known quirks / limitations:** flat 2D target with no real depth-based occlusion, so it's good for isolating registration/scale-correction
  accuracy but doesn't exercise halo/occlusion handling the way blossom or the synthetic stack do.

## blossom

- **Subject:** Green/violet blossom on a white table
- **Type:** Real photo
- **Frame count:** 100
- **Resolution / bit depth:** 8664x5784, 16-bit RGB TIFF
- **Camera / lens / magnification:** Sony Alpha 1 Mark 1, Sony FE 2.8/100mm Macro, 1.4x
- **Why it's a test case:** real depth structure, real ISO 1600 noise (post-denoise), real color/exposure consistency, full realistic pipeline test.
- **Known quirks / limitations:** The original images where taken with ISO 1600, before converting them to
  16-bit TIFF, a standard denoising profile was applied to all images in the stacks via Darktable .

## synthetic_50

- **Subject:** Procedurally generated scene (textured body, thin radiating
  antenna/leg lines, near foreground edge, background bokeh)
- **Type:** Synthetic
- **Frame count:** 50
- **Resolution / bit depth:** 1200x900, 16-bit RGB TIFF
- **Camera / lens / magnification:** N/A (rendered, not captured)
- **Why it's a test case:** Thin-structure halo/ghosting stress test, has
  ground truth (`ground_truth_all_in_focus.tiff`) for objective scoring in
  addition to visual rating, includes simulated focus breathing and sensor
  noise.
- **Known quirks / limitations:** Faint seam visible at the body's depth-band
  boundaries even when in focus — cosmetic compositing artifact. Note that
  `ground_truth_all_in_focus.tiff` sits in the same directory as the 50
  `frame_*.tiff` files; frame discovery must exclude it or it gets stacked as
  a 51st frame. Frames here are Deflate-compressed with 36 rows/strip, unlike
  the uncompressed 1-row/strip real stacks.

<!--
Add a new section above for each additional stack. Keep the field list
consistent so entries stay comparable at a glance.
-->




## Third-party reference renders

`<stack>/reference_pmax.tif` — the same source frames stacked by **Zerene Stacker**
using its **PMax** method, supplied by Niels. Present for `blossom` and `ruler` as of
2026-08-09.

**What it is for:** answering "is this achievable on this data at all?" and "where
specifically do we diverge?". PMax takes an independent highest-local-contrast decision at
each pyramid level, so a divergence localizes the problem rather than just scoring it.

**What changed as of T11:** these files were added when stackaroni's fusion rule was
architecturally *different* from PMax — one weight map propagated to every pyramid level,
the shape of Zerene's DMap. That gap is what the references were bought to diagnose, and
it is now closed: `--fusion select` takes an independent per-level decision by default
(`docs/algorithms.md` §6b), and on blossom it was rated **above** this reference. So their
job has shifted from "is this achievable?" to "where does a mature implementation of the
same family still differ from ours?" — a narrower question, and the honest framing now
that we are not chasing them.

**Coordinate systems do not match, and matching canvas sizes will not tell you so.**
Zerene runs its own alignment and resolves onto a different reference frame. Measured on
blossom: **scale 0.930** against our output, ~16 px displacement at the centre but ~380 px
at a corner, despite both being exactly 8664x5784. A crop pair near the centre is
feature-matched; one near a corner shows entirely different content and would read as a
fusion difference when it is a registration-anchor difference. Measure the similarity
before trusting any crop comparison —
`crates/core/tests/fusion_rule_crops.rs::pmax_versus_ours_is_a_similarity_not_an_identity`
does exactly this.

**Tone differs too, and it confounds noise comparisons.** blossom's PMax background sits
6.4% brighter than ours, and chroma noise falls monotonically with brightness on that
stack — so an unmatched Cb/Cr comparison overstated the gap by roughly half. Compare
against a source frame at the *same* brightness, not raw.

**What it is NOT for: an automated scoring target.** Do not compute RMSE or any aggregate
error against it and treat the number as quality. That mistake is already logged twice in
`docs/eval-log.md` — ground-truth RMSE rewarded noise reduction over sharpness, and
separately under-weighted background coherence, pointing the wrong way in opposite
directions. This file is a qualitative reference for looking at, and for locating *where*
to look. Ratings still come from Niels.

Excluded from frame discovery via `NON_FRAME_STEMS`, alongside `stackaroni_fused`.
