//! Wavelet-domain focus stacking.
//!
//! Li, Manjunath & Mitra, *Graphical Models and Image Processing* 57(3), 1995,
//! 235-245, via `docs/algorithms.md` §5. Each registered frame is decomposed by a
//! discrete wavelet transform; detail coefficients are selected from whichever frame
//! carries the most local activity there, the selection map is cleaned up by a
//! majority filter, and the inverse transform reconstructs the fused frame.
//!
//! # Why this method does not fit the four-stage pipeline
//!
//! Registration still applies, but focus measurement, weight estimation and fusion
//! collapse into one operation: the activity measure *is* the focus measure, the
//! selection map *is* the weight map, and both exist only inside the transform's
//! coefficient domain, where no per-pixel `FocusMap` or `WeightMaps` can represent
//! them. That is why this implements [`StackFusion`] rather than [`ImageFusion`] —
//! see the stage-boundary note in `CLAUDE.md`.
//!
//! # The transform: CDF 5/3, by lifting
//!
//! Cohen, Daubechies & Feauveau, *Comm. Pure Appl. Math.* 45(5), 1992, 485-560; the
//! lifting factorization is Sweldens, *SIAM J. Math. Anal.* 29(2), 1998, 511-546, and
//! this is the reversible 5/3 pair specified by JPEG2000 (ISO/IEC 15444-1, Annex F).
//!
//! Chosen over Haar because Haar's two-tap support makes selection-based fusion show
//! blocking artifacts, which is exactly the seam class the quality checklist in
//! `CLAUDE.md` scores hardest. Chosen over an undecimated (shift-invariant) transform
//! because that is 4x the memory per level and is not the transform the paper
//! specifies; if the decimated transform's shift-variance shows as seams, the
//! undecimated variant is the documented next step, not a retune of this one.
//!
//! Lifting matters for correctness here, not just for speed: because the predict and
//! update steps are inverted one at a time, perfect reconstruction holds for *any*
//! boundary rule, as long as the inverse uses the same one. The symmetric extension
//! below is therefore a quality choice, not a correctness risk.
//!
//! [`StackFusion`]: crate::pipeline::StackFusion
//! [`ImageFusion`]: crate::pipeline::ImageFusion

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::fusion::{level_count, warp_frame};
use crate::grid::Grid;
use crate::image::FrameInfo;
use crate::pipeline::{Image, RunControl, StackFusion, Stage, Transform};
use crate::tiff_io::write_rgb16_srgb;

/// Mirror `i` back into `0..n` by whole-point symmetric extension.
///
/// `x[-1] = x[1]` and `x[n] = x[n-2]`, the extension JPEG2000 specifies for 5/3. The
/// alternative — clamping, so `x[-1] = x[0]` — duplicates the edge sample and injects
/// a step the transform reads as a genuine edge, putting a bright rim of detail
/// coefficients around all four borders of every frame.
fn mirror(i: i64, n: i64) -> usize {
    debug_assert!(n > 0);
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j > n - 1 {
        j = period - j;
    }
    j as usize
}

/// One level of the forward transform along a single signal.
///
/// Writes `[approx | detail]` into `out`: `ceil(n/2)` low-pass coefficients followed
/// by `floor(n/2)` high-pass ones. Splitting the output rather than working in place
/// keeps the deinterleave free — the caller wants the subbands contiguous anyway.
fn analyze_1d(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(out.len(), n);
    if n < 2 {
        out[..n].copy_from_slice(x);
        return;
    }
    let na = n.div_ceil(2);
    let nd = n / 2;
    let ni = n as i64;

    // Predict: each odd sample against the average of its two even neighbours.
    let (approx, detail) = out.split_at_mut(na);
    for (i, d) in detail.iter_mut().enumerate() {
        let c = 2 * i + 1;
        *d = x[c] - 0.5 * (x[mirror(c as i64 - 1, ni)] + x[mirror(c as i64 + 1, ni)]);
    }

    // Update: lift the even samples by the detail either side, which is what makes the
    // low-pass band a smoothed decimation rather than a bare subsample.
    for i in 0..na {
        let left = detail[(i as i64 - 1).clamp(0, nd as i64 - 1).max(0) as usize];
        let right = detail[i.min(nd.saturating_sub(1))];
        let lift = if nd == 0 { 0.0 } else { 0.25 * (left + right) };
        approx[i] = x[2 * i] + lift;
    }
}

/// Invert [`analyze_1d`], undoing the update step and then the predict step.
fn synthesize_1d(coeffs: &[f32], out: &mut [f32]) {
    let n = coeffs.len();
    debug_assert_eq!(out.len(), n);
    if n < 2 {
        out[..n].copy_from_slice(coeffs);
        return;
    }
    let na = n.div_ceil(2);
    let nd = n / 2;
    let ni = n as i64;
    let (approx, detail) = coeffs.split_at(na);

    // Even samples first: they are what the predict step referred to, so they have to
    // exist before it can be undone.
    for i in 0..na {
        let left = detail[(i as i64 - 1).clamp(0, nd as i64 - 1).max(0) as usize];
        let right = detail[i.min(nd.saturating_sub(1))];
        let lift = if nd == 0 { 0.0 } else { 0.25 * (left + right) };
        out[2 * i] = approx[i] - lift;
    }
    // `out` is written at odd positions while being read at even ones, which are all
    // already final — so this iterates `detail`, not `out`.
    for (i, &d) in detail.iter().enumerate() {
        let c = 2 * i + 1;
        out[c] = d + 0.5 * (out[mirror(c as i64 - 1, ni)] + out[mirror(c as i64 + 1, ni)]);
    }
}

/// Run `f` over every row of `src` independently, into a fresh plane of the same size.
///
/// Rows are the unit of parallelism throughout. The column direction is handled by
/// transposing and reusing this, rather than by a strided second kernel: a strided
/// in-place column pass cannot be split across threads without aliasing, and the two
/// transposes cost less than the sequential pass they replace.
fn row_map(src: &Plane, f: fn(&[f32], &mut [f32])) -> Plane {
    let w = src.width as usize;
    let mut out = Plane::new(src.width, src.height);
    out.data
        .par_chunks_mut(w)
        .zip(src.data.par_chunks(w))
        .for_each(|(dst, row)| f(row, dst));
    out
}

fn transpose(src: &Plane) -> Plane {
    let (w, h) = (src.width as usize, src.height as usize);
    let mut out = Plane::new(src.height, src.width);
    // Writes stay sequential per output row; the strided side is the read, which is
    // the cheaper one to make irregular.
    out.data.par_chunks_mut(h).enumerate().for_each(|(x, dst)| {
        for (y, cell) in dst.iter_mut().enumerate() {
            *cell = src.data[y * w + x];
        }
    });
    out
}

/// A single-channel float plane.
///
/// Separate from [`crate::fusion::Bitmap`] because every operation here is
/// single-channel by nature: the transform runs per colour channel, and activity,
/// labels and the selection map are all one value per coefficient. Carrying an
/// interleaved channel stride through all of it would be a stride that is always 1.
#[derive(Clone)]
pub struct Plane {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl Plane {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width as usize * height as usize],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// The three detail orientations at one decomposition level, each a quarter of the
/// level's input: horizontal (`hl`), vertical (`lh`) and diagonal (`hh`).
pub struct Details {
    pub hl: Plane,
    pub lh: Plane,
    pub hh: Plane,
}

/// A full multi-level decomposition: detail bands finest-first, plus what is left.
pub struct Decomposition {
    pub details: Vec<Details>,
    pub approx: Plane,
}

/// One 2D transform step: rows, then columns, then split into the four quadrants.
fn analyze_step(src: &Plane) -> (Plane, Details) {
    let rows = row_map(src, analyze_1d);
    let cols = transpose(&row_map(&transpose(&rows), analyze_1d));

    let (w, h) = (src.width, src.height);
    let (nax, nay) = (w.div_ceil(2), h.div_ceil(2));
    let quadrant = |x0: u32, y0: u32, qw: u32, qh: u32| {
        let mut q = Plane::new(qw, qh);
        for y in 0..qh as usize {
            let from = (y0 as usize + y) * w as usize + x0 as usize;
            q.data[y * qw as usize..][..qw as usize]
                .copy_from_slice(&cols.data[from..][..qw as usize]);
        }
        q
    };
    (
        quadrant(0, 0, nax, nay),
        Details {
            hl: quadrant(nax, 0, w - nax, nay),
            lh: quadrant(0, nay, nax, h - nay),
            hh: quadrant(nax, nay, w - nax, h - nay),
        },
    )
}

/// Invert [`analyze_step`]: reassemble the quadrants, then undo columns and rows.
fn synthesize_step(approx: &Plane, details: &Details) -> Plane {
    let w = approx.width + details.hl.width;
    let h = approx.height + details.lh.height;
    let mut joined = Plane::new(w, h);
    let mut place = |src: &Plane, x0: u32, y0: u32| {
        for y in 0..src.height as usize {
            let to = (y0 as usize + y) * w as usize + x0 as usize;
            joined.data[to..][..src.width as usize]
                .copy_from_slice(&src.data[y * src.width as usize..][..src.width as usize]);
        }
    };
    place(approx, 0, 0);
    place(&details.hl, approx.width, 0);
    place(&details.lh, 0, approx.height);
    place(&details.hh, approx.width, approx.height);

    let cols = transpose(&row_map(&transpose(&joined), synthesize_1d));
    row_map(&cols, synthesize_1d)
}

/// Decompose `plane` into `levels` levels of detail plus a final approximation.
pub fn forward(plane: &Plane, levels: usize) -> Decomposition {
    let mut approx = plane.clone();
    let mut details = Vec::with_capacity(levels);
    for _ in 0..levels {
        // A plane that cannot be halved again has nothing left to decompose; stopping
        // early is what lets `level_count`'s floor be a size rather than a promise about
        // how many levels every stack supports.
        if approx.width < 2 || approx.height < 2 {
            break;
        }
        let (next, d) = analyze_step(&approx);
        details.push(d);
        approx = next;
    }
    Decomposition { details, approx }
}

/// Rebuild the plane a [`Decomposition`] came from.
pub fn inverse(decomposition: &Decomposition) -> Plane {
    let mut plane = decomposition.approx.clone();
    for details in decomposition.details.iter().rev() {
        plane = synthesize_step(&plane, details);
    }
    plane
}

/// Rec. 709 luminance, the weights `tiff_io` already encodes with.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Radius of the activity window and of the consistency filter, both 3x3 as published.
const WINDOW: u32 = 1;

/// Local activity of the three detail orientations at one level, on luminance.
///
/// **Luminance, not per channel.** Selecting a different frame for red than for green
/// at one coefficient would read as colour fringing on exactly the high-contrast edges
/// this rule exists to sharpen — the same reason [`crate::fusion::SelectionFusion`]
/// keeps its salience joint across channels. The wavelet transform is linear, so the
/// luminance of the coefficients *is* the coefficient of the luminance; combining after
/// the transform is exact, not an approximation, and saves a fourth transform.
///
/// **Joint across the three orientations too**, giving one decision per position per
/// level rather than three. Independent orientation decisions would reconstruct a
/// position from up to three different frames, which is the same inconsistency in the
/// gradient domain.
///
/// Activity is the windowed *energy* — the paper's area-based measure — not the bare
/// coefficient magnitude. A per-coefficient argmax on ISO-1600 frames selects noise.
fn activity(details: [&Details; 3], grid: (u32, u32)) -> Vec<f32> {
    let (gw, gh) = grid;
    let mut energy = vec![0f32; gw as usize * gh as usize];
    for y in 0..gh {
        for x in 0..gw {
            let mut acc = 0.0;
            for (channel, weight) in LUMA.iter().enumerate() {
                let d = details[channel];
                // Each orientation is a separate band; their energies add.
                for band in [&d.hl, &d.lh, &d.hh] {
                    let c = band.data[at(band, x, y)];
                    acc += weight * c * c;
                }
            }
            energy[y as usize * gw as usize + x as usize] = acc;
        }
    }
    crate::filter::box_sum(&energy, gw, gh, WINDOW)
}

/// Address a subband at a position on the level's label grid.
///
/// The three orientations are not all the same size: at an odd dimension the
/// decimation splits `n` into `ceil(n/2)` low and `floor(n/2)` high coefficients, so
/// `hl` is a row taller than `hh` and `lh` a column wider. The label grid is the union
/// — the size of that level's approximation band — and the odd row or column is read
/// by clamping, which shares the edge label rather than leaving it unselected.
///
/// This costs one duplicated row or column per odd dimension per level. Levels below
/// the first are odd about half the time (8664 -> 4332 -> 2166 -> 1083), so it is not a
/// rare path and is not treated as one.
fn at(plane: &Plane, x: u32, y: u32) -> usize {
    let cx = x.min(plane.width.saturating_sub(1));
    let cy = y.min(plane.height.saturating_sub(1));
    cy as usize * plane.width as usize + cx as usize
}

/// The label-grid size at each level, following [`forward`]'s stopping rule exactly.
///
/// Computed up front so the accumulators can be allocated before the first frame is
/// read, rather than lazily on the first decomposition.
fn grids(width: u32, height: u32, levels: usize) -> Vec<(u32, u32)> {
    let (mut w, mut h) = (width, height);
    let mut out = Vec::with_capacity(levels);
    for _ in 0..levels {
        if w < 2 || h < 2 {
            break;
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        out.push((w, h));
    }
    out
}

/// Consistency verification: replace a label by the plurality of its 8 neighbours,
/// where that plurality is decisive enough.
///
/// See [`crate::defaults::CONSISTENCY_THRESHOLD`] for why this is a plurality rather
/// than the paper's majority, and why the two coincide at two frames.
///
/// Reads from a snapshot rather than in place, so the result does not depend on scan
/// order — an in-place filter propagates its own corrections rightward and downward and
/// would smear one label across a whole row.
fn verify_consistency(labels: &mut [u16], width: u32, height: u32, threshold: u32) -> usize {
    let source = labels.to_vec();
    let (w, h) = (width as i64, height as i64);
    let mut changed = 0;

    for y in 0..h {
        for x in 0..w {
            let centre = source[(y * w + x) as usize];
            // Small fixed tally: at most 8 distinct labels can appear among 8
            // neighbours, so this stays cheaper than a map.
            let mut seen: [(u16, u32); 8] = [(0, 0); 8];
            let mut kinds = 0;
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (cx, cy) = ((x + dx).clamp(0, w - 1), (y + dy).clamp(0, h - 1));
                    let label = source[(cy * w + cx) as usize];
                    match seen[..kinds].iter_mut().find(|(l, _)| *l == label) {
                        Some((_, count)) => *count += 1,
                        None => {
                            seen[kinds] = (label, 1);
                            kinds += 1;
                        }
                    }
                }
            }
            let (top, count) = seen[..kinds]
                .iter()
                .copied()
                .max_by_key(|(_, c)| *c)
                .unwrap();
            if count >= threshold && top != centre {
                labels[(y * w + x) as usize] = top;
                changed += 1;
            }
        }
    }
    changed
}

/// Wavelet-domain focus stacking over a registered stack.
///
/// # Two passes, and why
///
/// Consistency verification needs the whole label map before it can filter it, and the
/// filtered labels then point at coefficients from frames already read and dropped.
/// Keeping every frame's coefficients instead would be the stack itself in memory —
/// 60 GB on a 100-frame 50 MP stack — so the frames are decomposed twice: once to
/// decide, once to gather. That doubles this method's transform cost, which is the
/// price the published algorithm's consistency step carries at this stack size.
///
/// # Memory
///
/// One frame's decomposition at a time, plus accumulators that do not grow with the
/// frame count: selected coefficients (one stack-sized set), the running best activity
/// and the labels. The frame count stays out of the budget, as in
/// [`crate::fusion::SelectionFusion`].
pub struct WaveletStack {
    output: PathBuf,
    floor: u32,
    consistency_threshold: u32,
    debug_dir: Option<PathBuf>,
}

impl WaveletStack {
    pub fn new(
        output: &Path,
        floor: u32,
        consistency_threshold: u32,
        debug_dir: Option<&Path>,
    ) -> Self {
        Self {
            output: output.to_path_buf(),
            floor,
            consistency_threshold,
            debug_dir: debug_dir.map(Path::to_path_buf),
        }
    }

    /// Warp one frame into anchor coordinates and decompose each colour channel.
    fn decompose(
        &self,
        image: &Image,
        transforms: &HashMap<PathBuf, Transform>,
        info: FrameInfo,
        levels: usize,
    ) -> Result<[Decomposition; 3]> {
        let transform = transforms
            .get(image.path())
            .copied()
            .unwrap_or(Transform::IDENTITY);
        let warped = warp_frame(image, transform, info)?;

        let n = (info.width as usize) * (info.height as usize);
        let mut planes = [
            Plane::new(info.width, info.height),
            Plane::new(info.width, info.height),
            Plane::new(info.width, info.height),
        ];
        for i in 0..n {
            for (ch, plane) in planes.iter_mut().enumerate() {
                plane.data[i] = warped.data[i * 3 + ch];
            }
        }
        drop(warped);
        Ok(planes.map(|p| forward(&p, levels)))
    }
}

impl StackFusion for WaveletStack {
    fn stack(
        &self,
        images: &[Image],
        transforms: &HashMap<PathBuf, Transform>,
        run: &dyn RunControl,
    ) -> Result<Image> {
        let info = images[0].info();
        // One fewer than the pyramid's level count: that counts the base level too, and
        // here the base is the approximation band rather than a level of details.
        let levels = level_count(info.width, info.height, self.floor).saturating_sub(1);
        let grid = grids(info.width, info.height, levels);

        let mut best: Vec<Vec<f32>> = grid
            // Negative so the first frame wins everywhere regardless of how flat it is,
            // the same seeding as `SelectionFusion`.
            .iter()
            .map(|(w, h)| vec![-1.0f32; *w as usize * *h as usize])
            .collect();
        let mut labels: Vec<Vec<u16>> = grid
            .iter()
            .map(|(w, h)| vec![0u16; *w as usize * *h as usize])
            .collect();
        let mut approx_sum: Option<[Plane; 3]> = None;

        // --- pass 1: activity and selection --------------------------------------
        for (index, image) in images.iter().enumerate() {
            if run.cancelled() {
                return Err(Error::Cancelled);
            }
            let d = self.decompose(image, transforms, info, levels)?;

            for (level, dims) in grid.iter().enumerate() {
                let a = activity(
                    [
                        &d[0].details[level],
                        &d[1].details[level],
                        &d[2].details[level],
                    ],
                    *dims,
                );
                let (best, labels) = (&mut best[level], &mut labels[level]);
                for (i, energy) in a.iter().enumerate() {
                    if *energy > best[i] {
                        best[i] = *energy;
                        labels[i] = index as u16;
                    }
                }
            }

            // The approximation band is averaged, per the paper. Accumulated here so
            // pass 2 does not have to touch it again.
            let sums = approx_sum.get_or_insert_with(|| {
                let (w, h) = (d[0].approx.width, d[0].approx.height);
                [Plane::new(w, h), Plane::new(w, h), Plane::new(w, h)]
            });
            for ch in 0..3 {
                for (acc, v) in sums[ch].data.iter_mut().zip(&d[ch].approx.data) {
                    *acc += v;
                }
            }
            run.progress(Stage::Focus, index + 1, images.len());
        }

        // --- consistency verification --------------------------------------------
        for (level, (w, h)) in grid.iter().enumerate() {
            if run.cancelled() {
                return Err(Error::Cancelled);
            }
            verify_consistency(&mut labels[level], *w, *h, self.consistency_threshold);
            run.progress(Stage::Weights, level + 1, grid.len().max(1));
        }
        if let Some(dir) = &self.debug_dir {
            self.write_label_maps(dir, &labels, &grid, images.len())?;
        }

        // --- pass 2: gather the selected coefficients ----------------------------
        let mut out: Vec<[Details; 3]> = Vec::new();
        for (index, image) in images.iter().enumerate() {
            if run.cancelled() {
                return Err(Error::Cancelled);
            }
            let d = self.decompose(image, transforms, info, levels)?;
            if out.is_empty() {
                out = (0..grid.len())
                    .map(|level| {
                        std::array::from_fn(|ch| {
                            let s = &d[ch].details[level];
                            Details {
                                hl: Plane::new(s.hl.width, s.hl.height),
                                lh: Plane::new(s.lh.width, s.lh.height),
                                hh: Plane::new(s.hh.width, s.hh.height),
                            }
                        })
                    })
                    .collect();
            }

            for (level, (gw, gh)) in grid.iter().enumerate() {
                let labels = &labels[level];
                for y in 0..*gh {
                    for x in 0..*gw {
                        if labels[y as usize * *gw as usize + x as usize] != index as u16 {
                            continue;
                        }
                        for ch in 0..3 {
                            let src = &d[ch].details[level];
                            let dst = &mut out[level][ch];
                            // Clamped the same way on both sides, so the shared edge
                            // row or column is copied from the frame that owns it.
                            for (to, from) in [
                                (&mut dst.hl, &src.hl),
                                (&mut dst.lh, &src.lh),
                                (&mut dst.hh, &src.hh),
                            ] {
                                let i = at(to, x, y);
                                to.data[i] = from.data[at(from, x, y)];
                            }
                        }
                    }
                }
            }
            run.progress(Stage::Fuse, index + 1, images.len());
        }

        // --- reconstruct ----------------------------------------------------------
        let scale = 1.0 / images.len() as f32;
        let mut approx = approx_sum.expect("a stack has at least one frame");
        let mut channels: Vec<Plane> = Vec::with_capacity(3);
        for ch in (0..3).rev() {
            for v in approx[ch].data.iter_mut() {
                *v *= scale;
            }
            let details: Vec<Details> = out
                .iter_mut()
                .map(|level| std::mem::replace(&mut level[ch], EMPTY_DETAILS))
                .collect();
            channels.push(inverse(&Decomposition {
                details,
                approx: std::mem::replace(&mut approx[ch], Plane::new(0, 0)),
            }));
        }
        channels.reverse();

        // Never checked past here: a truncated TIFF that looks like a real output is
        // worse than finishing the write.
        write_rgb16_srgb(&self.output, info, |y, row| {
            let start = y as usize * info.width as usize;
            for x in 0..info.width as usize {
                for ch in 0..3 {
                    row[x * 3 + ch] = channels[ch].data[start + x];
                }
            }
            Ok(())
        })?;
        Image::open(&self.output)
    }
}

/// Placeholder left behind when a level's details are moved out for reconstruction.
const EMPTY_DETAILS: Details = Details {
    hl: Plane {
        width: 0,
        height: 0,
        data: Vec::new(),
    },
    lh: Plane {
        width: 0,
        height: 0,
        data: Vec::new(),
    },
    hh: Plane {
        width: 0,
        height: 0,
        data: Vec::new(),
    },
};

impl WaveletStack {
    /// One greyscale PNG per level: which frame each coefficient was taken from.
    ///
    /// This is the method's diagnostic intermediate, in the sense `CLAUDE.md` requires
    /// of every stage. It answers the question the fused image cannot: a speckled map
    /// means selection is picking noise, and a map with a hard ring at the subject
    /// boundary means the opposite problem.
    fn write_label_maps(
        &self,
        dir: &Path,
        labels: &[Vec<u16>],
        grid: &[(u32, u32)],
        frames: usize,
    ) -> Result<()> {
        let scale = 1.0 / (frames.saturating_sub(1).max(1)) as f32;
        for (level, (w, h)) in grid.iter().enumerate() {
            let mut g = Grid::new(*w, *h);
            for (dst, label) in g.data.iter_mut().zip(&labels[level]) {
                *dst = *label as f32 * scale;
            }
            crate::debug::write_grid(&dir.join(format!("wavelet_labels_{level}.png")), &g)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lifting guarantees this for any boundary rule; the test is here because a
    /// transposed index or a mismatched mirror between the two directions would break
    /// it, and nothing downstream would say so — a slightly-wrong inverse looks like a
    /// slightly-soft image, which is indistinguishable from an algorithm that simply
    /// performs badly.
    #[test]
    fn analysis_and_synthesis_round_trip() {
        for n in 1..40usize {
            let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 19) as f32) * 0.1 - 0.7).collect();
            let mut coeffs = vec![0.0; n];
            let mut back = vec![0.0; n];
            analyze_1d(&x, &mut coeffs);
            synthesize_1d(&coeffs, &mut back);
            for (a, b) in x.iter().zip(&back) {
                assert!((a - b).abs() < 1e-5, "n={n}: {a} != {b}");
            }
        }
    }

    /// A constant signal has nothing to predict, so every detail coefficient is zero
    /// and the approximation carries the constant through unchanged. This is what makes
    /// the activity measure meaningful: flat regions score zero, so a frame wins there
    /// only if something in it is genuinely not flat.
    #[test]
    fn constant_signal_has_no_detail() {
        let x = vec![0.25f32; 16];
        let mut coeffs = vec![0.0; 16];
        analyze_1d(&x, &mut coeffs);
        assert!(coeffs[8..].iter().all(|d| d.abs() < 1e-6), "{coeffs:?}");
        assert!(coeffs[..8].iter().all(|a| (a - 0.25).abs() < 1e-6));
    }

    fn ramp(width: u32, height: u32) -> Plane {
        let mut p = Plane::new(width, height);
        for (i, v) in p.data.iter_mut().enumerate() {
            *v = ((i * 31 % 97) as f32) * 0.01 - 0.4;
        }
        p
    }

    /// Odd sizes included deliberately: the split point differs between the two
    /// directions there, which is where a transposed `nax`/`nay` would hide.
    #[test]
    fn two_dimensional_round_trip() {
        for (w, h) in [(1, 1), (2, 3), (7, 5), (16, 16), (33, 17), (5, 40)] {
            let src = ramp(w, h);
            let back = inverse(&forward(&src, 4));
            assert_eq!((back.width, back.height), (w, h));
            for (i, (a, b)) in src.data.iter().zip(&back.data).enumerate() {
                assert!((a - b).abs() < 1e-4, "{w}x{h} at {i}: {a} != {b}");
            }
        }
    }

    /// The decomposition stops when a plane can no longer be halved, so asking for more
    /// levels than the size supports is not an error and not an infinite loop.
    #[test]
    fn depth_is_capped_by_size() {
        let d = forward(&ramp(4, 4), 10);
        assert_eq!(d.details.len(), 2);
        assert_eq!((d.approx.width, d.approx.height), (1, 1));
    }

    /// The label grids are allocated before any frame is decomposed, so a disagreement
    /// with `forward`'s own stopping rule would be an out-of-bounds panic on the first
    /// odd-sized level — or, worse, a silently skipped level.
    #[test]
    fn grid_sizes_agree_with_the_decomposition() {
        // Odd sizes and a size that runs out of levels early, which are the two ways
        // the two rules could disagree.
        for (w, h) in [(33, 17), (64, 64), (5, 40), (3, 3)] {
            let levels = 6;
            let d = forward(&ramp(w, h), levels);
            let expected = grids(w, h, levels);
            assert_eq!(d.details.len(), expected.len(), "{w}x{h}");
            for (level, (gw, gh)) in expected.iter().enumerate() {
                // The label grid is the union of the three orientations, which is
                // exactly the size of that level's approximation band.
                let hl = &d.details[level].hl;
                let lh = &d.details[level].lh;
                assert_eq!(*gw, lh.width.max(hl.width), "{w}x{h} level {level} width");
                assert_eq!(
                    *gh,
                    hl.height.max(lh.height),
                    "{w}x{h} level {level} height"
                );
            }
        }
    }

    /// At two frames a plurality of 5 or more *is* a majority, so the published rule is
    /// recovered exactly rather than approximated. This is the test that holds the
    /// documented extension to being an extension.
    #[test]
    fn two_frames_reduce_to_the_published_majority() {
        // Centre says frame 1; six of its eight neighbours say frame 0.
        let mut labels = vec![0u16, 0, 0, 0, 1, 0, 0, 1, 1];
        let changed = verify_consistency(&mut labels, 3, 3, 5);
        assert_eq!(changed, 1);
        assert_eq!(labels[4], 0, "a 6-of-8 majority should override the centre");

        // An even 4-4 split is not a majority, so at threshold 5 the centre stands.
        let mut split = vec![0u16, 0, 0, 0, 1, 1, 1, 1, 1];
        verify_consistency(&mut split, 3, 3, 5);
        assert_eq!(split[4], 1, "a 4-4 split must not override the centre");
    }

    /// With many frames a strict majority usually does not exist, which is the case the
    /// paper never has to address. The plurality rule still acts; a threshold above what
    /// any label reaches leaves everything alone.
    #[test]
    fn plurality_acts_where_no_majority_exists() {
        // Eight distinct neighbours: no label reaches even 2.
        let mut scattered: Vec<u16> = vec![1, 2, 3, 4, 99, 5, 6, 7, 8];
        assert_eq!(verify_consistency(&mut scattered, 3, 3, 4), 0);
        assert_eq!(scattered[4], 99);

        // Four agree, which is short of a majority but reaches the default threshold.
        let mut plurality: Vec<u16> = vec![7, 7, 7, 7, 99, 1, 2, 3, 4];
        assert_eq!(verify_consistency(&mut plurality, 3, 3, 4), 1);
        assert_eq!(plurality[4], 7);
    }

    /// Filtering in place would let a correction propagate into the neighbourhood of
    /// the next position and smear one label across the row.
    #[test]
    fn consistency_reads_from_a_snapshot() {
        let mut row = vec![5u16; 12];
        row[6] = 9;
        let mut once = row.clone();
        verify_consistency(&mut once, 12, 1, 4);
        // Only the one odd cell flips; its neighbours were already 5.
        assert_eq!(once.iter().filter(|&&l| l == 9).count(), 0);
        assert!(once.iter().all(|&l| l == 5));
    }

    #[test]
    fn mirror_is_whole_point_symmetric() {
        assert_eq!(mirror(-1, 5), 1);
        assert_eq!(mirror(5, 5), 3);
        assert_eq!(mirror(0, 1), 0);
    }
}
