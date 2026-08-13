//! Separable box filters.
//!
//! Both the windowed-Laplacian focus measure and the guided filter are built out of
//! sliding sums over a square window, so both run in O(1) per pixel regardless of
//! window size. That is what makes a large radius affordable on a 50 MP frame.

/// Sum over a `(2*radius+1)` square window, edges replicated.
///
/// Replication matters: out-of-range taps repeat the border sample rather than being
/// skipped. Skipping would silently shrink the window near the borders, which reads
/// as reduced energy along every frame edge.
pub fn box_sum(data: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let (w, h, r) = (width as i64, height as i64, radius as i64);
    debug_assert_eq!(data.len(), (w * h) as usize);

    let mut horizontal = vec![0f32; data.len()];
    for y in 0..h {
        let row = &data[(y * w) as usize..][..w as usize];
        sliding_sum(
            w,
            r,
            |x| row[x.clamp(0, w - 1) as usize],
            &mut horizontal,
            y * w,
            1,
        );
    }

    let mut out = vec![0f32; data.len()];
    for x in 0..w {
        sliding_sum(
            h,
            r,
            |y| horizontal[(y.clamp(0, h - 1) * w + x) as usize],
            &mut out,
            x,
            w,
        );
    }
    out
}

/// One 1-D sliding sum of radius `r`: `n` outputs from `tap`, written to
/// `out[start + i * stride]`. `tap` clamps, so the window stays full width at the edges.
fn sliding_sum(n: i64, r: i64, tap: impl Fn(i64) -> f32, out: &mut [f32], start: i64, stride: i64) {
    let mut acc: f32 = (-r..=r).map(&tap).sum();
    for i in 0..n {
        out[(start + i * stride) as usize] = acc;
        acc -= tap(i - r);
        acc += tap(i + r + 1);
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

    #[test]
    fn box_mean_of_a_constant_is_that_constant() {
        let data = vec![0.75f32; 40];
        for v in box_mean(&data, 8, 5, 2) {
            assert!((v - 0.75).abs() < 1e-5, "{v}");
        }
    }
}
