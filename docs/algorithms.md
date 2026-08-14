# Focus Stacking Algorithms for Insect Macro Photography

*Algorithm overview and recommended architecture for a Rust + egui cross-platform application — reviewed edition*

## Review notes

This is a technical review of the original document. The named algorithms and formulas were checked against the published literature; no outright factual errors were found, but several places were imprecise or internally inconsistent. Changes made in this edition:

- Clarified the executive-summary table, which mixed two different axes — the focus-measurement/decomposition domain and the weight-refinement strategy — as if they were mutually exclusive whole-pipeline choices, and resolved the overlap between the "baseline" and "full pipeline" Laplacian-pyramid rows.
- Refined the Laplacian focus-measure formula to reflect standard practice: an energy/variance computed over a local window, not a single noise-sensitive pixel value (Section 3).
- Expanded Section 10 (Registration) with the specific, named techniques used in practice — phase correlation, ECC, and feature-based (ORB + RANSAC) alignment — rather than leaving registration unspecified.
- Added a `Registration` trait to the Rust architecture sketch (Section 14): Section 10 argues registration deserves equal attention to fusion, but the original trait sketch only covered focus measurement, weighting and fusion.
- Fixed a table-rendering overlap in the CNN / deep learning row of the executive summary.
- Added a References section with verified academic citations plus practical/engineering references (existing open-source tools and Rust crates).

---

## 1. Executive summary

For insect macro photography, focus stacking is best treated as two related problems: **registration** of the input frames and **fusion** of their in-focus regions. A naïve per-pixel maximum focus measure is fast but tends to produce noisy, spatially inconsistent decisions. A strong classical architecture is: **affine registration → multi-scale focus measurement → spatially coherent weight-map estimation → Laplacian-pyramid fusion**.

> **Note on this comparison.** The rows below actually span two different design axes that are often combined rather than chosen between: the *decomposition domain* used for multi-scale analysis (gradient/Laplacian pyramid vs. wavelet), and the *weight-refinement strategy* used to decide, region by region, which frame contributes (direct selection, pyramid blending, edge-aware filtering, or graph-cut/MRF optimization). Graph-cut and guided/edge-aware refinement are not alternatives to Laplacian-pyramid fusion — they can be layered on top of it to produce the weight map that then feeds the pyramid blend.

| Approach | Quality | Speed | Robustness | Macro suitability |
|---|---|---|---|---|
| Single-scale gradient/Laplacian measure + direct pyramid blending (baseline) | ★★★★☆ | ★★★★★ | ★★★☆☆ | Good baseline |
| Wavelet-domain stacking | ★★★★☆ | ★★★★☆ | ★★★★☆ | Very good |
| Multi-scale Laplacian-pyramid fusion (integrated measure + blend, full pipeline) | ★★★★★ | ★★★★☆ | ★★★★☆ | Excellent |
| Graph-cut / MRF weight refinement | ★★★★★ | ★★☆☆☆ | ★★★★★ | Excellent; complex |
| Guided / edge-aware weight refinement | ★★★★★ | ★★★☆☆ | ★★★★★ | Excellent |
| CNN / deep learning (end-to-end) | Potentially ★★★★★ | ★★☆☆☆ | Variable | Interesting later |
| Maximum-gradient selection (no blending) | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | Prototype only |

## 2. What the algorithm must solve

A focus stack is not simply a collection of pixels from different images. Insect macro stacks commonly contain very shallow depth of field, fine hairs and antennae, specular highlights, translucent structures, repetitive texture, diffraction, focus breathing and sometimes subject movement. These properties make both geometric registration and focus-map construction important.

**Registration** aligns the frames. **Fusion** decides which frame contributes each region and combines those regions without seams, halos or inconsistent texture.

## 3. Classical focus measures

### Gradient-based measures

A basic sharpness measure is gradient magnitude. A common form is the gradient energy:

```
F(x,y) = Ix(x,y)^2 + Iy(x,y)^2
```

Sobel or, preferably, Scharr derivatives are inexpensive and often effective. The weakness is that noise and strong texture can look like focus — an important issue for insect hairs and other high-frequency structures.

### Laplacian measures

A classical focus measure is based on the magnitude or local energy of the Laplacian. The pointwise definition is:

```
F(x,y) = |∇²I(x,y)|      (∇² = the Laplacian operator, i.e. Ixx + Iyy)
```

**Correction:** in practice this is almost never evaluated as a single pixel value — a bare second derivative is extremely noise-sensitive, which undercuts the very robustness the measure is meant to provide. The standard formulations instead aggregate over a small window *W*, e.g. the Sum-Modified-Laplacian (SML) or a windowed Laplacian energy:

```
F(x,y) = Σ(i,j) in W  ( ∇²I(i,j) )^2
```

This windowed form is fast, simple, well suited to 16-bit image processing, and is a good first implementation and benchmark — but it remains more sensitive to noise, halos and fine texture than multi-scale or edge-aware alternatives (Pertuz, Puig & Garcia, 2013).

### Tenengrad and local variance

Tenengrad is essentially gradient-energy focus measurement, typically summed (and sometimes thresholded) over a local window. Local variance can also be used because focused regions often contain greater local intensity variation. Both are useful comparison points in an experimental implementation, and both are covered in the same comparative survey referenced above.

## 4. Multi-scale focus measures

Rather than evaluating sharpness at only one spatial scale, a multi-scale method evaluates fine, medium and coarse structure. Gaussian and Laplacian pyramids (Burt & Adelson, 1983) are natural tools for this. Fine scales capture hairs and tiny details; coarser scales capture larger anatomical structures. This reduces the tendency of one tiny high-frequency feature to dictate the focus decision for a whole region.

### T12 — built, measured, and not shipped

**This section is a dead end, recorded so it is not walked twice.** The paragraph above is the argument that motivated it, and the argument still sounds right; the measurements say it does essentially nothing on this pipeline as built. The code exists in full — metric, CLI flag, UI entry, sweep test — on the unmerged branch `t12-multiscale-focus`, and is deliberately not on `main`.

The literature does not supply one canonical multi-scale focus measure. What is established is the pyramid (Burt & Adelson, 1983), the measure applied at each level (Nayar & Nakagawa's modified Laplacian and its windowed sum, 1990/1994; surveyed by Pertuz, Puig & Garcia, 2013), and the practice of combining per-scale sharpness (Zhang et al.'s sum-of-Gaussian-based modified Laplacian, *Digital Signal Processing* 2020; Li et al.'s region mosaicking on Laplacian pyramids, *PLOS ONE* 13(5), 2018). What was implemented is the composition of those pieces:

```text
G_0 = luma(I),  G_{k+1} = reduce(G_k)          (binomial [1,4,6,4,1]/16, Burt & Adelson)
F_k(x,y) = sum over window W of ( laplacian(G_k)(i,j) )^2       (the §3 measure, per level)
F(x,y)   = F_0(x,y) + sum over k=1..S-1  d^k * expand^k(F_k)(x,y)
```

`S` is the number of scales, `d` the per-octave decay, and the window radius is the same at every level — so a level-`k` window covers `2^k` times the image area. `S = 1` reduces to §3 exactly, not approximately, which a bit-identity test pinned.

**What the measurements showed** (full numbers in `docs/eval-log.md`): on `synthetic_50`, detail **0.330 at every scale count 1–5** and at every decay from 0.25 to 2.0; on `blossom`, **0.02% RMSE** against the single-scale result — 17 levels out of 65535.

**Why that is a real negative and not a broken experiment**, since a flat sweep usually means the latter. The coarse levels are not inert: they contribute **48× level 0's magnitude** at five scales, their between-frame contrast *rises* with scale count (0.32 → 1.44 between the first and last frame), and **97% of fused pixels move** between one scale and five. The measure changes, the change reaches the output, and the image looks the same regardless.

**The suspected cause, not established.** Levels are summed **unnormalized**, and the 48× figure is direct evidence that they are not commensurable — the discrete Laplacian's response depends on grid spacing, so `F_k` computed on a half-size level is not on the same footing as `F_0`. Scale-space theory requires normalizing derivatives by the scale before combining them (Lindeberg, *IJCV* 30(2), 1998).

**Trigger for revisiting: scale-normalized levels.** That is a *different algorithm*, not a parameter fix to this one, so it starts from this documented dead end on a new branch rather than by reviving `t12-multiscale-focus`. Two things to carry forward: the combination rule was weighted-sum only (max-across-scales was scoped out of v1 and never built), and the `S = 1` bit-identity requirement is what makes any such metric comparable to the shipped one on equal terms.

## 5. Wavelet-domain stacking

Wavelet transforms decompose an image into low-frequency structure and several high-frequency directional bands. Focus decisions can be made on wavelet coefficients and the result reconstructed (Li, Manjunath & Mitra, 1995). Wavelets offer good detail preservation and natural multi-scale behavior, but coefficient-selection rules and boundary handling introduce additional design choices.

### T14 — implemented as `core::wavelet`

**This is the first method in the codebase that does not fit the four-stage pipeline**, and that is why it was built: focus measurement, weight estimation and fusion collapse into one operation on transform coefficients. Registration still applies. It implements `StackFusion` rather than `ImageFusion`; see the stage-boundary note in `CLAUDE.md`.

**The transform.** CDF 5/3 (Cohen, Daubechies & Feauveau, 1992), by lifting (Sweldens, 1998) — the reversible pair JPEG2000 specifies (ISO/IEC 15444-1, Annex F). Two levels of choice were resolved against this project's quality checklist rather than by defaulting:

- **Over Haar**, whose two-tap support makes selection-based fusion show blocking artifacts — the seam class checklist item 1 scores hardest.
- **Over an undecimated (shift-invariant) transform**, which is 4x memory per level and is not what the paper specifies. The decimated transform is shift-variant, and consistency verification exists partly to mitigate that. **If seams appear at high-contrast edges, SWT is the documented next step** — a different transform on a new branch, not a retune of this one.

Boundary handling is whole-point symmetric extension, `x[-1] = x[1]`. Clamping instead would duplicate the edge sample and inject a step the transform reads as a genuine edge, ringing all four borders of every frame. Because lifting inverts its steps one at a time, perfect reconstruction holds for *any* boundary rule as long as the inverse matches — so the extension is a quality choice, not a correctness risk, and `analysis_and_synthesis_round_trip` pins it.

**Selection.** Detail coefficients are taken from whichever frame carries the most local activity, where activity is the windowed *energy* over a 3x3 area — the paper's area-based measure, not the bare coefficient magnitude. A per-coefficient argmax on ISO-1600 frames selects noise (Wang et al., 2018), the same finding that shaped §6b.

Two choices make the decision joint rather than independent, both for the reason §6b gives for keeping salience joint across channels — an inconsistent selection at one location reads as fringing:

- **On luminance, not per channel.** The transform is linear, so the luminance of the coefficients *is* the coefficient of the luminance; combining after the transform is exact and saves a fourth transform.
- **Across the three orientations, not per orientation**, giving one decision per position per level rather than three.

**Consistency verification, and a documented extension.** The paper filters the selection map so an isolated coefficient does not come from a different frame than its neighbourhood. It fuses **two** images, where "the majority of the 8 neighbours" is always well-defined. **With 100 frames a strict majority usually does not exist** — eight neighbours can hold eight different labels — a case the paper has no reason to address. So the filter takes the **plurality** and applies it only when it reaches `defaults::CONSISTENCY_THRESHOLD` (4 of 8). **At two frames a plurality of 5 or more is a majority, so the published rule is recovered exactly as a special case rather than replaced**, and `two_frames_reduce_to_the_published_majority` is the test holding that claim.

The filter reads from a snapshot, not in place: an in-place pass propagates its own corrections and smears one label across a row.

**Cost: the frames are decomposed twice.** Verification needs the whole label map before it can filter it, and the filtered labels then point at coefficients from frames already read and dropped. Keeping every frame's coefficients would be the stack itself in memory. So one pass decides and one gathers.

**The approximation band is averaged across all frames**, per the paper. Note this is an extrapolation of its rule: Li et al. average the LL band of *two* images. Averaging over 100 is the most likely source of a soft or washed result and is the first place to look if the output reads flat.

**Diagnostic output:** one greyscale PNG per level of the selection label map, written under `--debug-out`. A speckled map means selection is picking noise; a hard ring at the subject boundary means the opposite problem.

## 6. Laplacian-pyramid fusion

Laplacian-pyramid fusion is one of the strongest candidates for this application. Instead of selecting a source image independently for every pixel, a spatially varying mask is constructed and different spatial-frequency bands are blended separately. This substantially reduces seams and abrupt transitions. The technique traces to the original Laplacian pyramid representation (Burt & Adelson, 1983) and its extension to selective image fusion (Burt & Kolczynski, 1993).

```
  Images
    |
    +--> Laplacian pyramid A
    +--> Laplacian pyramid B
    |
    v
  spatial weight map
    |
    v
  weighted fusion (per pyramid level)
    |
    v
  reconstruction
```

## 6b. Selection versus averaging in the pyramid (added after the blossom rating)

§6 cites Burt & Kolczynski (1993) for "selective image fusion". The *selective* part is a
distinct fusion rule from the weighted average, and stackaroni initially implemented only
the average. This section records the difference, because it is the leading candidate for
fixing the blossom result (rated 1/5 at `e2e6b8a`).

**What is currently implemented (T8).** One weight map per frame, produced by T6 focus
measurement and T7 guided-filter refinement, is Gaussian-reduced to each pyramid level and
used to blend:

```
fused[level] = sum_k  W_k[level] * L_k[level]
```

Every level therefore inherits the *same* decision, taken once at a single window scale.
That decision must simultaneously be smooth enough to avoid background mottling and sharp
enough to preserve thin structure. This is architecturally the shape of Zerene Stacker's
DMap, and it carries DMap's documented weakness at fine detail.

**Burt & Kolczynski's rule.** The decision is taken independently at every level and
position, from the pyramid coefficients themselves:

```
S_k(l,p)  = salience, local energy of L_k over a window W around p
M(l,p)    = match, normalized correlation between the sources over W
if M < threshold:   select the coefficient with greatest salience
else:               average, weighted toward the more salient
```

A fresh contrast decision per frequency band, which is architecturally the shape of Zerene's
PMax — documented as strong on "overlapping structures like mats of hair and crisscrossing
bristles", i.e. exactly the subject stackaroni scores worst on.

**Why the salience window matters, rather than naive per-pixel maximum.** The simpler rule
— take the coefficient of largest absolute value at each level and position — is standard in
the multi-focus literature, but is noise-sensitive: reconstructed pixels are drawn from
different sources inconsistently across levels, "resulting in loss of original clear
information and introduction of distortion" (Wang et al., *PLOS ONE* 13(5), 2018, region
mosaicking on Laplacian pyramids). On ISO-1600 frames, per-pixel argmax over Laplacian
coefficients would routinely select noise. Use the windowed salience form; `filter::box_sum`
already provides the window.

**Scope.** This is a change to the fusion rule inside `fusion.rs`, not a new stage or
dependency, and it is plausibly *less* code than the current path. Note the consequence: for
the band-pass levels the decision comes from the pyramid itself, so T6/T7 are no longer on
the critical path for detail — they remain relevant for the base (coarsest) band, which has
no meaningful "contrast" to select on and still needs a weighted blend.

**Known simplification: the match/average branch is not implemented.** `SelectionFusion`
implements the selection half of the rule above and omits the `M >= threshold` branch, so it
always selects and never averages toward the more salient source. This is a **deliberate
consequence of the streaming architecture**, not an oversight: the match term is a
normalized correlation *between the sources* over a window, which requires every frame's
coefficients at a level simultaneously. Fusion instead folds frames in one at a time against
a running best-salience plane, which is what keeps the frame count out of the memory budget
— 100 full-resolution pyramids is not a budget that exists. Implementing the match term
means either a bounded working set of candidate sources per level or a second pass over
every frame, and both are real cost.

Tracked the same way §10's unmodelled rotation is tracked, and for the same reason — the
term is omitted because the data does not currently demand it, and the trigger for
revisiting is named in advance rather than left to judgement:

- **What would justify implementing it:** a stack where salience between frames is close to
  tied over a region, so the winner flips on noise. That shows as switching artifacts —
  patchy, blotchy background rather than the uniform grain of a single frame passed through.
- **Where it would show first:** smoothly varying regions with no frame genuinely in focus,
  i.e. background bokeh, which is checklist item 3.
- **What the current evidence says:** it has not shown up. On blossom the background grain
  is uniform, not patchy, and measures at one source frame's chroma noise rather than above
  it — which is what rules out selection combining inconsistent sources. On synthetic_50 the
  `bokeh` metric crosses 1.0 while the crops stay clean. So: **do not implement the match
  term speculatively.** Wait for a stack that actually shows the instability.

**References for this section**

- Burt, P. J., & Kolczynski, R. J. (1993). Enhanced image capture through fusion. *ICCV*, 173-182.
- Wang, J., et al. (2018). A multi-focus image fusion method via region mosaicking on
  Laplacian pyramids. *PLOS ONE*, 13(5), e0191085.

## 7. The focus map is the critical component

The quality of the focus map is often more important than the final blending operation. A raw winner-takes-all rule can make neighboring pixels choose unrelated source frames. The desired result is spatially coherent regions corresponding to meaningful focused structures.

This motivates spatial regularization, edge-aware filtering and optimization methods rather than purely local decisions.

## 8. MRF / graph-cut optimization

A more advanced approach treats each pixel as a labeling problem. Let `L(x,y)` denote the source frame selected at a pixel. An energy can combine a data term measuring focus quality with a smoothness term encouraging neighboring pixels to have consistent labels:

```
E(L) = Σ_p D_p(L_p) + λ Σ_(p,q) V_p,q(L_p, L_q)
```

Graph cuts, Markov random fields and related optimization techniques (Boykov, Veksler & Zabih, 2001) can produce very clean focus boundaries. The trade-off is considerably greater implementation and computational complexity.

## 9. Guided and edge-aware filtering

Raw focus maps benefit from smoothing, but ordinary Gaussian smoothing can blur true object boundaries. Guided filtering (He, Sun & Tang, 2010), bilateral filtering (Tomasi & Manduchi, 1998) and related edge-aware methods can smooth unreliable focus decisions while preserving boundaries such as insect hairs against the background or eye/head transitions. Guided filtering has also been applied directly as a multi-focus fusion framework, not just as a weight-map post-process (Li, Kang & Hu, 2013), and is a reasonable single reference implementation to prototype against.

## 10. Registration

Registration deserves equal attention to fusion. Focus breathing can change magnification as the focus position changes, so translation-only alignment may be insufficient. A sensible progression is translation → affine registration → optional local refinement.

**Addition — named techniques.** The original document left this progression unspecified; in practice each step corresponds to a well-established, named algorithm rather than something to design from scratch:

- **Phase correlation** (Kuglin & Hines, 1975) — a fast, Fourier-domain estimate of pure translation between two frames; a reasonable coarse first pass before refinement.
- **ECC — Enhanced Correlation Coefficient maximization** (Evangelidis & Psarakis, 2008) — an intensity-based optimizer that directly refines a translation, Euclidean, affine or homography transform without needing distinct keypoints; well suited to the small, smooth deformations typical between adjacent stack frames.
- **Feature-based matching** (e.g. ORB keypoints, Rublee et al., 2011, with RANSAC-based model fitting, Fischler & Bolles, 1981) — more robust to larger displacements or frames where intensity-based optimization struggles, at the cost of depending on enough distinctive keypoints being present.

For the affine step specifically, this matters because focus breathing changes apparent scale between frames, not just position — a similarity or affine model (which includes scale) is a more honest fit than translation alone. Existing prior art worth studying directly: the Rust crate `libstacker` implements both keypoint-based and ECC-based alignment on top of OpenCV, and is a useful reference for how these pieces fit together in practice.

Optical flow can handle local deformation but can also behave poorly around thin hairs, antennae, transparent wings and defocused backgrounds unless strongly constrained — treat it as a later refinement, not a first implementation.

## 11. Diffraction and focus measures

Stopping down increases depth of field but also increases diffraction and reduces high-frequency contrast. Because a normal focus stack keeps aperture constant, this is less problematic than comparing different apertures, but it reinforces the value of robust multi-scale focus measures rather than relying on a single high-frequency metric.

## 12. Deep learning

CNN- or transformer-based methods can learn focus probability maps and fusion behavior directly (e.g. Liu, Chen, Peng & Wang, 2017, for two-image multi-focus fusion; more recent work targets full focus stacks specifically, e.g. the 2023 preprint "Towards Real-World Focus Stacking with Deep Learning" on arXiv, not yet peer-reviewed at time of writing). These are potentially powerful, but they require suitable training data, an inference runtime and more complex cross-platform deployment. For a photography application where pixel fidelity matters, a classical deterministic pipeline is a better foundation. ML could be an optional future component.

## 13. Recommended architecture

```
  TIFF stack
    v
  16-bit linear image representation
    v
  Registration
    +-- translation (phase correlation)
    +-- affine (ECC / feature-based, later)
    v
  Multi-scale decomposition
    v
  Focus quality maps
    v
  Consistency / edge-aware refinement
    v
  Weight maps
    v
  Laplacian-pyramid fusion
    v
  16-bit reconstruction
    v
  16-bit TIFF output
```

## 14. Recommended Rust architecture

Keep the registration, focus metric, weight estimation and fusion stages independently replaceable. This makes it possible to benchmark different algorithms on the same image stack without rewriting the application.

**Addition:** the original trait sketch omitted registration despite Section 10 arguing it deserves equal attention to fusion. Added below.

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

## 15. Practical development priorities

- **Registration** — start with phase-correlation translation, then add ECC or feature-based affine registration.
- **Multi-scale focus measurement** — benchmark Scharr, windowed-Laplacian and Tenengrad variants.
- **Spatially coherent focus maps** — add edge-aware refinement and later evaluate MRF/graph-cut approaches.
- **Laplacian-pyramid fusion** — use multi-scale blending rather than direct pixel selection.
- **Artifact handling** — specifically test noise, halos, fine hairs, transparent structures and moving subjects.
- **Performance** — design the pipeline for multithreading and later GPU acceleration.
- **16-bit fidelity** — avoid unnecessary conversion through 8-bit and keep adequate precision internally (decode to f32 early, quantize back only on output).

## Conclusion

For the stated Rust + egui + Windows/Linux/macOS constraints, the most promising classical foundation is: phase-correlation / ECC-based affine registration → multi-scale gradient/Laplacian focus measure → edge-aware or optimization-based weight maps → multi-scale Laplacian fusion. It offers a strong quality/performance/complexity balance without requiring a neural-network runtime, and every stage corresponds to a named, published technique rather than something that has to be invented from first principles.

---

## References

### Academic

- Burt, P. J., & Adelson, E. H. (1983). The Laplacian pyramid as a compact image code. *IEEE Transactions on Communications*, 31(4), 532–540.
- Burt, P. J., & Kolczynski, R. J. (1993). Enhanced image capture through fusion. *Proceedings of the 4th International Conference on Computer Vision (ICCV)*, 173–182.
- Pertuz, S., Puig, D., & Garcia, M. A. (2013). Analysis of focus measure operators for shape-from-focus. *Pattern Recognition*, 46(5), 1415–1432.
- Li, H., Manjunath, B. S., & Mitra, S. K. (1995). Multisensor image fusion using the wavelet transform. *Graphical Models and Image Processing*, 57(3), 235–245.
- He, K., Sun, J., & Tang, X. (2010). Guided image filtering. *Proceedings of the European Conference on Computer Vision (ECCV)*. Extended version: *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(6), 1397–1409 (2013).
- Li, S., Kang, X., & Hu, J. (2013). Image fusion with guided filtering. *IEEE Transactions on Image Processing*, 22(7), 2864–2875.
- Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 839–846.
- Boykov, Y., Veksler, O., & Zabih, R. (2001). Fast approximate energy minimization via graph cuts. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 23(11), 1222–1239.
- Kuglin, C. D., & Hines, D. C. (1975). The phase correlation image alignment method. *Proceedings of the IEEE International Conference on Cybernetics and Society*, 163–165.
- Evangelidis, G. D., & Psarakis, E. Z. (2008). Parametric image alignment using enhanced correlation coefficient maximization. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(10), 1858–1865.
- Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). ORB: An efficient alternative to SIFT or SURF. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2564–2571.
- Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, 24(6), 381–395.
- Liu, Y., Chen, X., Peng, H., & Wang, Z. (2017). Multi-focus image fusion with a deep convolutional neural network. *Information Fusion*, 36, 191–207.

### Practical / engineering references

- OpenCV documentation — `cv::findTransformECC` and the Image Alignment (ECC) tutorial: a directly usable reference implementation of the algorithm above.
- `PetteriAimonen/focus-stack` (GitHub) — an open-source C++ focus-stacking tool; useful prior art for how the full pipeline is assembled end to end.
- `eadf/libstacker.rs` (GitHub) — a Rust crate providing OpenCV-backed keypoint and ECC image alignment plus stacking; the closest existing prior art in the target language.
- `kornia-rs` (GitHub) — an actively developed pure-Rust computer-vision library, relevant if the OpenCV C++ dependency is undesirable for cross-platform packaging.
- Zerene Stacker and Helicon Focus — commercial stacking tools widely used in macro photography; useful as a practical quality bar and for understanding expected user-facing behavior (e.g. both accept TIFF/JPEG input in addition to RAW).

