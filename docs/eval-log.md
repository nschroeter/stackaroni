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
| 2026-08-08 | `0000000` | — | Example row — delete once real entries exist. | — | Format reference: commit is the short hash of the change being evaluated, stack is `ruler` / `blossom` / `synthetic_50`, notes should call out *which* checklist item failed and where in the debug output (e.g. "halo on left antenna, frame index ~30, weight map shows hard edge — try edge-aware refinement before pyramid fusion"). |

<!--
Tip: if a change scores worse on one stack but better on another, don't
average it away — note the tradeoff explicitly so it doesn't get relitigated
later without remembering why.
-->

