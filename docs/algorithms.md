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

**Addition — the implemented formulation (T12).** The paragraph above names the idea but not an algorithm, and the literature does not supply one canonical form: what is established is the pyramid (Burt & Adelson, 1983), the measure applied at each level (Nayar & Nakagawa's modified Laplacian and its windowed sum, 1990/1994; surveyed by Pertuz, Puig & Garcia, 2013), and the practice of combining per-scale sharpness rather than trusting one band (recent examples: Zhang et al.'s sum-of-Gaussian-based modified Laplacian, *Digital Signal Processing* 2020; Li et al.'s region mosaicking on Laplacian pyramids, *PLOS ONE* 13(5), 2018). What is implemented here is the composition of those pieces, written out so it is reproducible rather than implied:

```text
G_0 = luma(I),  G_{k+1} = reduce(G_k)          (binomial [1,4,6,4,1]/16, Burt & Adelson)
F_k(x,y) = sum over window W of ( laplacian(G_k)(i,j) )^2       (the §3 measure, per level)
F(x,y)   = F_0(x,y) + sum over k=1..S-1  d^k * expand^k(F_k)(x,y)
```

`S` is the number of scales and `d` the per-octave decay. The window radius is the same at every level, so a level-`k` window covers `2^k` times the image area — that, not a changing radius, is what makes it multi-scale.

Two properties are deliberate. **`S = 1` reduces to §3 exactly** — not approximately: level 0 is never scaled, so the single-scale metric is the `S = 1` case of this one and is verified as bit-identical by a test rather than by inspection. And **coarse levels are additive, not decisive**: they can raise a region's measured focus but cannot veto a fine-scale response, which is the intended asymmetry for hairs and antennae.

**Known simplification, scoped deliberately for v1: the combination rule is weighted-sum only.** Max-across-scales is the obvious alternative and is *not* implemented — no `rule` parameter exists, and the sum is not one option among several in the code. Two reasons, both revisable. A max rule is winner-takes-all across scales and inherits the spatial-incoherence failure §7 describes, which this project has already measured the cost of once (the argmax label field at `ced3a45`). And it would multiply the evaluation matrix by two before there is evidence that scale count matters at all on these stacks. **Trigger for revisiting:** if the sweep shows detail improving with `S` while background energy also climbs, a max rule is the next thing to try, because that pattern would mean the sum is accumulating background noise along with the signal it is meant to add.

## 5. Wavelet-domain stacking

Wavelet transforms decompose an image into low-frequency structure and several high-frequency directional bands. Focus decisions can be made on wavelet coefficients and the result reconstructed (Li, Manjunath & Mitra, 1995). Wavelets offer good detail preservation and natural multi-scale behavior, but coefficient-selection rules and boundary handling introduce additional design choices.

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

