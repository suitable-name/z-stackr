use rayon::prelude::*;
use stacker_core::image::PlanarImage;

// ---------------------------------------------------------------------------
// Geometry helpers (module-level so they are not "items after statements")
// ---------------------------------------------------------------------------

/// Σ_{i=a}^{b} i  (closed form, exact in i64)
#[inline]
const fn sum_range(a: i64, b: i64) -> i64 {
    (a + b) * (b - a + 1) / 2
}

/// Σ_{i=a}^{b} i²  (closed form, exact in i64)
///
/// Uses: b(b+1)(2b+1)/6 − (a−1)a(2a−1)/6
#[inline]
const fn sum_sq_range(a: i64, b: i64) -> i64 {
    let hi = b * (b + 1) * (2 * b + 1) / 6;
    let lo = if a == 0 {
        0
    } else {
        (a - 1) * a * (2 * a - 1) / 6
    };
    hi - lo
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the detrended residual-variance focus measure (plane-fit `PMod`).
///
/// For each pixel `(cx, cy)`, a window of half-size `radius` is extracted.
/// A local plane `f(dx, dy) = slopeX*(dx - meanDx) + slopeY*(dy - meanDy)` is
/// fitted to the pixel values by least-squares (plane-fit residual-variance
/// focus measure).  The plane is then subtracted from each
/// window pixel, and the corrected sample variance of the residuals is returned
/// as the sharpness score.
///
/// **Why this is better than plain Modified Laplacian**: a linearly-tilted but
/// smooth region (e.g. a gradient ramp) has zero residual variance, so it is
/// correctly classified as out-of-focus.  The Modified Laplacian would score it
/// non-zero because it responds to the slope.
///
/// The output `PlanarImage` carries the variance map in its `luma` channel;
/// `chroma_a` and `chroma_b` are zero.
///
/// # Algorithm
///
/// Implemented as an exact O(N) summed-area-table (SAT) pass.  Four SATs are
/// built once: `Σv`, `Σv²`, `Σ(col·v)`, `Σ(row·v)`.  Each pixel then
/// resolves its window statistics with four O(1) 4-corner queries plus purely
/// closed-form geometry sums, then applies the algebraic reduction
///
/// ```text
/// var = ( Σv² − covarX²/varX − covarY²/varY − (Σv)²/n ) / (n − 1)
/// ```
///
/// The result is numerically equivalent to the naive O(N·r²) per-pixel window
/// loop (verified pixel-by-pixel in `test_sat_equivalence`).
///
/// # Panics
/// Never — all window bounds are clamped.
// SAT construction + pixel loop — splitting would hurt readability
// x/y and dx/dy pairs are intentionally symmetric
#[allow(clippy::too_many_lines)] // cohesive SAT construction + pixel loop; splitting would hurt readability
pub fn compute_sum_modified_laplacian(img: &PlanarImage<f32>, radius: usize) -> PlanarImage<f32> {
    let width = img.width;
    let height = img.height;
    let len = width * height;

    // ── Build four Summed-Area Tables (SATs) in f64 ──────────────────────────
    //
    // Each SAT has dimensions (width+1) × (height+1), stored row-major.
    // sat[r][c] = Σ_{row<r, col<c} value(row,col).
    //
    // Tables:
    //   sat_v   : Σ v
    //   sat_v2  : Σ v²
    //   sat_col_v : Σ col·v   (col = absolute column index, 0-based)
    //   sat_row_v : Σ row·v   (row = absolute row index, 0-based)
    let sat_w = width + 1;
    let sat_h = height + 1;
    let sat_len = sat_w * sat_h;

    let mut sat_v = vec![0.0_f64; sat_len];
    let mut sat_v2 = vec![0.0_f64; sat_len];
    let mut sat_col_v = vec![0.0_f64; sat_len];
    let mut sat_row_v = vec![0.0_f64; sat_len];

    // Pass 1: row prefix sums (left-to-right per row).
    for row in 0..height {
        let mut acc_v = 0.0_f64;
        let mut acc_v2 = 0.0_f64;
        let mut acc_col_v = 0.0_f64;
        let mut acc_row_v = 0.0_f64;
        for col in 0..width {
            let v = f64::from(img.luma[row * width + col]);
            acc_v += v;
            acc_v2 += v * v;
            acc_col_v += col as f64 * v;
            acc_row_v += row as f64 * v;
            // Offset by +1 in both dimensions for the zero-padded SAT border.
            let idx = (row + 1) * sat_w + (col + 1);
            sat_v[idx] = acc_v;
            sat_v2[idx] = acc_v2;
            sat_col_v[idx] = acc_col_v;
            sat_row_v[idx] = acc_row_v;
        }
    }
    // Pass 2: column prefix sums (top-to-bottom per column).
    for col in 1..sat_w {
        for row in 1..sat_h {
            let idx = row * sat_w + col;
            let above = (row - 1) * sat_w + col;
            sat_v[idx] += sat_v[above];
            sat_v2[idx] += sat_v2[above];
            sat_col_v[idx] += sat_col_v[above];
            sat_row_v[idx] += sat_row_v[above];
        }
    }

    // 4-corner SAT query for rectangle [x0..x1] × [y0..y1] (inclusive).
    // Returns the sum of the table over that rectangle in O(1).
    let query = |sat: &[f64], x0: usize, y0: usize, x1: usize, y1: usize| -> f64 {
        let r1 = y1 + 1;
        let c1 = x1 + 1;
        sat[r1 * sat_w + c1] - sat[y0 * sat_w + c1] - sat[r1 * sat_w + x0] + sat[y0 * sat_w + x0]
    };

    // ── Per-pixel plane-fit residual variance using SATs ─────────────────────
    let r = radius as isize;
    let w = width as isize;
    let h = height as isize;

    let mut variance_map = vec![0.0_f32; len];

    variance_map
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(cy, row_slice)| {
            let icy = cy as isize;
            let y0 = (icy - r).max(0) as usize;
            let y1 = (icy + r).min(h - 1) as usize;
            let ny = (y1 - y0 + 1) as i64;

            for (cx, out) in row_slice.iter_mut().enumerate() {
                let icx = cx as isize;
                let x0 = (icx - r).max(0) as usize;
                let x1 = (icx + r).min(w - 1) as usize;
                let nx = (x1 - x0 + 1) as i64;
                let n = nx * ny;

                if n < 2 {
                    // Degenerate: not enough pixels to compute a variance.
                    continue;
                }

                // ── Closed-form geometry sums ─────────────────────────────────
                //
                // Gx  = Σ_{col=x0}^{x1} col   (each repeated ny times over rows)
                // Gxx = Σ_{col=x0}^{x1} col²
                // Gy, Gyy: analogous over rows
                let ix0 = x0 as i64;
                let ix1 = x1 as i64;
                let iy0 = y0 as i64;
                let iy1 = y1 as i64;
                let cx_double_precision = cx as f64;
                let cy_fp64 = cy as f64;
                let fn64 = n as f64;
                let fnx = nx as f64;
                let fny = ny as f64;

                let gx = sum_range(ix0, ix1) as f64;
                let gxx = sum_sq_range(ix0, ix1) as f64;
                let gy = sum_range(iy0, iy1) as f64;
                let gyy = sum_sq_range(iy0, iy1) as f64;

                // Σdx over the 2D window  (dx = col − cx)
                //   = (Σ_{col} col − cx·nx) · ny
                // Σdx² over the 2D window
                //   = (Σ col² − 2cx·Σ col + cx²·nx) · ny
                let sum_delta_h = (gx - cx_double_precision * fnx) * fny;
                let sum_sq_delta_h = (gxx - 2.0 * cx_double_precision * gx
                    + cx_double_precision * cx_double_precision * fnx)
                    * fny;
                let sum_delta_v = (gy - cy_fp64 * fny) * fnx;
                let sum_vertical_delta_sq =
                    (gyy - 2.0 * cy_fp64 * gy + cy_fp64 * cy_fp64 * fny) * fnx;

                // Unnormalised variance of dx / dy over the 2D window:
                //   varX = Σdx² − (Σdx)²/n
                //   varY = Σdy² − (Σdy)²/n
                let var_h = sum_sq_delta_h - sum_delta_h * sum_delta_h / fn64;
                let var_v = sum_vertical_delta_sq - sum_delta_v * sum_delta_v / fn64;

                if var_h <= 0.0 || var_v <= 0.0 {
                    // Degenerate: single column or row — slopes undefined.
                    continue;
                }

                // ── O(1) SAT queries ──────────────────────────────────────────
                let sv = query(&sat_v, x0, y0, x1, y1); // Σ v
                let sv2 = query(&sat_v2, x0, y0, x1, y1); // Σ v²
                let sxv = query(&sat_col_v, x0, y0, x1, y1); // Σ col·v
                let syv = query(&sat_row_v, x0, y0, x1, y1); // Σ row·v

                // Mean of dx over the 2D window = Σdx / n
                let mean_delta_h = sum_delta_h / fn64;
                let mean_delta_v = sum_delta_v / fn64;

                // covarX = Σ(v·dx) − meanDx·Σv
                //        = (Σ col·v − cx·Σv) − meanDx·Σv
                // covarY: analogous with rows
                let covar_h = (sxv - cx_double_precision * sv) - mean_delta_h * sv;
                let covar_v = (syv - cy_fp64 * sv) - mean_delta_v * sv;

                // Algebraic reduction of residual variance:
                //   Σr² = Σv² − covarX²/varX − covarY²/varY
                //   var  = (Σr² − (Σv)²/n) / (n − 1)
                let sum_sq_res = sv2 - covar_h * covar_h / var_h - covar_v * covar_v / var_v;
                let variance = (sum_sq_res - sv * sv / fn64) / (fn64 - 1.0);

                // Clamp to ≥ 0 to absorb tiny floating-point negatives near zero.
                *out = variance.max(0.0) as f32;
            }
        });

    PlanarImage {
        width,
        height,
        luma: variance_map,
        chroma_a: vec![0.0; len],
        chroma_b: vec![0.0; len],
    }
}
