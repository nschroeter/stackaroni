# Evaluation log

Running record of experiments against the fixed test stacks in `test-data/`.
Append a new row after any change that could affect output quality — don't
edit or delete old rows, even if the approach was later abandoned; the
history of what didn't work is as useful as what did.

Score is 1-5 against the quality checklist in `CLAUDE.md`:

1. No visible seams/halos around high-contrast edges (antennae, legs, hair boundaries)
2. No ghosting from misalignment
3. Background bokeh stays smooth, not patchily sharpened
4. Consistent color/exposure across the fused image

Log one row per stack tested if scores differ between stacks (they often
will — e.g. the synthetic stack's thin antenna lines tend to expose halo
problems that the blossom stack won't).

| Date | Commit | Stack | Change | Score (1-5) | Notes |
|---|---|---|---|---|---|
| 2026-08-08 | `e30dba2` | ruler | T5: phase-correlation registration (Kuglin & Hines 1975), sub-pixel parabolic peak. | n/a — no fused output yet | **Translation is the wrong model for this data.** Per-region correlation on one adjacent pair: left half +3.32 px, right half −2.98 px; top +0.99, bottom −1.24. At sep=20: +75.5 / −57.8 and +29.6 / −47.7. Opposite-signed and near-symmetric about the centre ⇒ uniform magnification, not displacement. Implied scale change ~0.145%/frame, consistent between sep=1 (6.3 px over 4332 px) and sep=20 (133 px over 4332 px) ⇒ ~14% across 100 frames. Alignment overlay shows red/green doubling growing radially toward the corners. A single `(dx,dy)` averages a field spanning ±3 px, so every adjacent pair keeps ~±3 px residual, ±60 px at the stack ends — ghosting on high-contrast edges (checklist item 2). Predicted by `docs/algorithms.md` §10. Not wired in as the active `Registration` impl. Fix = T5b, log-polar phase correlation (Reddy & Chatterji, *IEEE TIP* 5(8), 1996, 1266–1271) for translation + uniform scale. No rotation or shear in the per-region data, so general affine/ECC is not yet warranted. |
| 2026-08-08 | `e30dba2` | ruler | Benchmark design: antisymmetry as a registration-accuracy measure. | n/a — measure discarded | **Didn't work, don't retry.** Scored `align(a,b)` against `−align(b,a)`, expecting it to expose defocus-driven correlation error. Measured ~1e-4 px at every level and separation — useless. Reason: phase correlation is antisymmetric *by construction*; swapping arguments conjugates the cross-power spectrum, so the peak negates exactly regardless of how wrong it is. It measures the algebra, not the data. Chain-vs-direct (3.5–5.4 px disagreement over 8 frames) and per-region spread did the real work. |
| 2026-08-08 | `e30dba2` | ruler | Registration accuracy vs pyramid level, injected known shift. | n/a — estimator check | Mean error in full-res px: level 1 → 0.38, level 2 → 0.92, level 3 → 2.03, level 4 → 4.42. On *shared* content finer is strictly better, so the hypothesis that downsampling improves accuracy by suppressing defocus-mismatched high frequencies is **not** supported for level choice on its own — the dominant error source turned out to be the wrong motion model, not frequency mismatch. Revisit level choice after T5b. Timing per pair: level 1 ≈ 7.1 s, level 3 ≈ 1.6 s, level 4 ≈ 1.3 s (mostly frame decode, which is level-independent). |

<!--
Tip: if a change scores worse on one stack but better on another, don't
average it away — note the tradeoff explicitly so it doesn't get relitigated
later without remembering why.
-->

