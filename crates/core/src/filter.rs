//! Separable box filters.
//!
//! Both the windowed-Laplacian focus measure and the guided filter are built out of
//! sliding sums over a square window, so both run in O(1) per pixel regardless of
//! window size. That is what makes a large radius affordable on a 50 MP frame.
//!
//! **Both passes run across threads, and the vertical one is blocked by column.** This
//! was the largest single cost in fusion: profiled on blossom, `box_sum` was 48.7% of
//! the fuse stage's wall time on one thread while thirteen others waited. Two things
//! were wrong with it — it was sequential, and its vertical pass walked columns at
//! stride `width`, so every tap pulled a fresh cache line to use four bytes of it.
//!
//! **What the shape here preserves is the arithmetic.** Each column's running sum is
//! still accumulated in the same order, one subtraction and one addition per row, so
//! every output is the same float expression it was before — not a rearrangement of
//! one. Float addition is not associative, so a faster reduction over a different order
//! would change the fused output and move the hash `output_is_stable` pins. That is
//! also why the vertical pass cannot simply be split into row bands: restarting a
//! column's accumulator part-way down would sum the same taps in a different order.

use rayon::prelude::*;

/// Columns per task in the vertical pass.
///
/// Wide enough that each row touch is several whole cache lines and the inner loop over
/// accumulators vectorizes, narrow enough that a 50 MP frame still splits into dozens of
/// tasks. The block's accumulators are what make this cache-friendly: one sweep down the
/// rows updates all of them, so both source and destination are read forwards.
const COLUMN_BLOCK: usize = 256;

/// Sum over a `(2*radius+1)` square window, edges replicated.
///
/// Replication matters: out-of-range taps repeat the border sample rather than being
/// skipped. Skipping would silently shrink the window near the borders, which reads
/// as reduced energy along every frame edge.
pub fn box_sum(data: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let (w, h, r) = (width as i64, height as i64, radius as i64);
    debug_assert_eq!(data.len(), (w * h) as usize);
    let (w_us, h_us) = (width as usize, height as usize);

    // Rows are independent, and each writes only its own chunk.
    let mut horizontal = vec![0f32; data.len()];
    horizontal
        .par_chunks_mut(w_us)
        .enumerate()
        .for_each(|(y, dst)| {
            let row = &data[y * w_us..][..w_us];
            sliding_sum(w, r, |x| row[x.clamp(0, w - 1) as usize], dst);
        });

    // Columns are independent too, but a column is not a contiguous slice, so each task
    // writes its block to its own buffer and the merge below puts them back row by row.
    // The buffer is block-major — `[y * block + x]` — so the sweep writes forwards.
    let blocks: Vec<(usize, Vec<f32>)> = (0..w_us)
        .step_by(COLUMN_BLOCK)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|x0| {
            let block = COLUMN_BLOCK.min(w_us - x0);
            let row_at = |y: i64| {
                let y = y.clamp(0, h - 1) as usize;
                &horizontal[y * w_us + x0..][..block]
            };

            let mut acc = vec![0f32; block];
            for k in -r..=r {
                for (slot, &v) in acc.iter_mut().zip(row_at(k)) {
                    *slot += v;
                }
            }

            let mut out = vec![0f32; block * h_us];
            for y in 0..h {
                out[y as usize * block..][..block].copy_from_slice(&acc);
                for ((slot, &leaving), &entering) in
                    acc.iter_mut().zip(row_at(y - r)).zip(row_at(y + r + 1))
                {
                    *slot -= leaving;
                    *slot += entering;
                }
            }
            (x0, out)
        })
        .collect();

    // Freed before the result is allocated, so the peak is two planes as it was before
    // the blocking — the block buffers stand in for the old output plane, not beside it.
    drop(horizontal);

    let mut out = vec![0f32; data.len()];
    out.par_chunks_mut(w_us).enumerate().for_each(|(y, dst)| {
        for (x0, block_data) in &blocks {
            let block = COLUMN_BLOCK.min(w_us - x0);
            dst[*x0..*x0 + block].copy_from_slice(&block_data[y * block..][..block]);
        }
    });
    out
}

/// One 1-D sliding sum of radius `r`: `out.len()` outputs written in order. `tap`
/// clamps, so the window stays full width at the edges.
fn sliding_sum(n: i64, r: i64, tap: impl Fn(i64) -> f32, out: &mut [f32]) {
    let mut acc: f32 = (-r..=r).map(&tap).sum();
    for (i, slot) in out.iter_mut().enumerate().take(n as usize) {
        *slot = acc;
        acc -= tap(i as i64 - r);
        acc += tap(i as i64 + r + 1);
    }
}

/// Mean over a `(2*radius+1)` square window, edges replicated.
pub fn box_mean(data: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let count = (2 * radius + 1) as f32;
    let inv = 1.0 / (count * count);
    let mut out = box_sum(data, width, height, radius);
    for v in &mut out {
        *v *= inv;
    }
    out
}

/// Elementwise product, for the covariance terms the guided filter needs.
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_sum_matches_a_naive_sum() {
        let (w, h, r) = (9u32, 7u32, 2u32);
        let data: Vec<f32> = (0..w * h).map(|i| (i % 5) as f32).collect();
        let got = box_sum(&data, w, h, r);

        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let mut want = 0.0;
                for dy in -(r as i64)..=r as i64 {
                    for dx in -(r as i64)..=r as i64 {
                        let cy = (y + dy).clamp(0, h as i64 - 1);
                        let cx = (x + dx).clamp(0, w as i64 - 1);
                        want += data[(cy * w as i64 + cx) as usize];
                    }
                }
                let have = got[(y * w as i64 + x) as usize];
                assert!((have - want).abs() < 1e-3, "at {x},{y}: {have} != {want}");
            }
        }
    }

    /// The narrow fixture above fits in one column block, so it cannot see a seam between
    /// two of them. This one is wider than [`COLUMN_BLOCK`] and not a multiple of it, so
    /// the last block is short and both boundaries are exercised.
    #[test]
    fn blocks_join_without_a_seam() {
        let (w, h, r) = (COLUMN_BLOCK as u32 * 2 + 37, 9u32, 3u32);
        let data: Vec<f32> = (0..w * h).map(|i| ((i * 7) % 13) as f32).collect();
        let got = box_sum(&data, w, h, r);

        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let mut want = 0.0;
                for dy in -(r as i64)..=r as i64 {
                    for dx in -(r as i64)..=r as i64 {
                        let cy = (y + dy).clamp(0, h as i64 - 1);
                        let cx = (x + dx).clamp(0, w as i64 - 1);
                        want += data[(cy * w as i64 + cx) as usize];
                    }
                }
                let have = got[(y * w as i64 + x) as usize];
                assert!((have - want).abs() < 1e-2, "at {x},{y}: {have} != {want}");
            }
        }
    }

    #[test]
    fn box_mean_of_a_constant_is_that_constant() {
        let data = vec![0.75f32; 40];
        for v in box_mean(&data, 8, 5, 2) {
            assert!((v - 0.75).abs() < 1e-5, "{v}");
        }
    }
}
