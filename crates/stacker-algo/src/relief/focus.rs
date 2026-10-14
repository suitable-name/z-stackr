//! The `Relief` focus measure: the residual variance of a local plane fit.
//!
//! [`compute_focus_map`] scores every pixel by how much of its window's
//! variation a plane cannot explain, so flat areas and smooth ramps score 0 and
//! texture scores high. It runs in O(N) from rolling summed-area tables of `f64`
//! ([`compute_focus_map_banded`] exposes the band height), and
//! [`compute_plane_fit_variance`] wraps the map in a [`PlanarImage`].

#![expect(
    clippy::cast_precision_loss,
    reason = "row/column indices and window sums stay far below 2^52, so the usize/i64 to f64 conversions are exact"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "image dimensions never exceed i64::MAX; the signed values only feed the closed-form window sums"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the f64 window variance is narrowed to the f32 focus map on purpose"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "the closed-form window algebra is evaluated as plain multiply-add, matching the naive reference in the tests"
)]

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
// Rolling summed-area tables
// ---------------------------------------------------------------------------

/// Output rows handled per band by [`compute_focus_map`].  The four `f64`
/// tables hold `band + 2·radius + 2` rows of `width + 1` entries at a time, i.e.
/// `32·(band + 2r + 2)·(width + 1)` bytes, instead of `32·(height + 1)·(width + 1)`
/// for the whole image.
const DEFAULT_BAND_ROWS: usize = 128;

/// The four summed-area tables (`Σv`, `Σv²`, `Σ col·v`, `Σ row·v`), of which only
/// a ring of the most recent `ring_len` table rows is kept.
///
/// Table row `ρ` (`0..=height`) holds, at column `c` (`0..=width`), the sum over
/// image rows `< ρ` and columns `< c`.  Row `ρ` depends only on row `ρ − 1` and
/// image row `ρ − 1`, so rows can be produced in order and old ones dropped once
/// no window reaches them.  Each entry is computed with the same additions in the
/// same order as the full-table construction (row prefix sum, plus the entry
/// above), so the values are bit-identical to it.
struct SatRing {
    sat_w: usize,
    ring_len: usize,
    v: Vec<f64>,
    v2: Vec<f64>,
    col_v: Vec<f64>,
    row_v: Vec<f64>,
}

impl SatRing {
    fn new(width: usize, rows_in_ring: usize) -> Self {
        let sat_w = width + 1;
        let len = sat_w * rows_in_ring;
        Self {
            sat_w,
            ring_len: rows_in_ring,
            v: vec![0.0; len],
            v2: vec![0.0; len],
            col_v: vec![0.0; len],
            row_v: vec![0.0; len],
        }
    }

    /// Offset of table row `rho` inside each ring buffer.
    #[inline]
    const fn slot(&self, rho: usize) -> usize {
        (rho % self.ring_len) * self.sat_w
    }

    /// Produce table row `rho ≥ 1` from image row `rho − 1` (`luma_row`).
    /// Table row `rho − 1` must still be in the ring.
    fn push_row(&mut self, rho: usize, luma_row: &[f32]) {
        let row = (rho - 1) as f64;
        let cur = self.slot(rho);
        let prev = self.slot(rho - 1);
        // Column 0 is the zero border; slots are reused, so reset it.
        self.v[cur] = 0.0;
        self.v2[cur] = 0.0;
        self.col_v[cur] = 0.0;
        self.row_v[cur] = 0.0;

        let mut acc_v = 0.0_f64;
        let mut acc_v2 = 0.0_f64;
        let mut acc_col_v = 0.0_f64;
        let mut acc_row_v = 0.0_f64;
        for (col, &px) in luma_row.iter().enumerate() {
            let v = f64::from(px);
            acc_v += v;
            acc_v2 += v * v;
            acc_col_v += col as f64 * v;
            acc_row_v += row * v;
            let i = col + 1;
            self.v[cur + i] = acc_v + self.v[prev + i];
            self.v2[cur + i] = acc_v2 + self.v2[prev + i];
            self.col_v[cur + i] = acc_col_v + self.col_v[prev + i];
            self.row_v[cur + i] = acc_row_v + self.row_v[prev + i];
        }
    }

    /// 4-corner query of one table over the inclusive rectangle
    /// `[x0..=x1] × [y0..=y1]`.
    #[inline]
    fn query(&self, table: &[f64], x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
        let top = self.slot(y0);
        let bottom = self.slot(y1 + 1);
        let c1 = x1 + 1;
        table[bottom + c1] - table[top + c1] - table[bottom + x0] + table[top + x0]
    }
}

// ---------------------------------------------------------------------------
// Per-pixel evaluation
// ---------------------------------------------------------------------------

/// Plane-fit residual variance of pixel `(cx, cy)` over the clamped window
/// `[x0..=x1] × [y0..=y1]`, from the tables and closed-form geometry sums.
/// Returns `0.0` for the degenerate windows (fewer than two pixels, or a single
/// row or column, where the slopes are undefined).
#[expect(
    clippy::similar_names,
    reason = "x/y and dx/dy pairs are intentionally symmetric"
)]
fn window_variance(
    sats: &SatRing,
    (cx, cy): (usize, usize),
    (x0, y0, x1, y1): (usize, usize, usize, usize),
) -> f32 {
    let nx = (x1 - x0 + 1) as i64;
    let ny = (y1 - y0 + 1) as i64;
    let n = nx * ny;

    if n < 2 {
        // Degenerate: not enough pixels to compute a variance.
        return 0.0;
    }

    // ── Closed-form geometry sums ─────────────────────────────────────────
    //
    // Gx  = Σ_{col=x0}^{x1} col   (each repeated ny times over rows)
    // Gxx = Σ_{col=x0}^{x1} col²
    // Gy, Gyy: analogous over rows
    let ix0 = x0 as i64;
    let ix1 = x1 as i64;
    let iy0 = y0 as i64;
    let iy1 = y1 as i64;
    let cx_f = cx as f64;
    let cy_f = cy as f64;
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
    let sum_dx = (gx - cx_f * fnx) * fny;
    let sum_dx2 = (gxx - 2.0 * cx_f * gx + cx_f * cx_f * fnx) * fny;
    let sum_dy = (gy - cy_f * fny) * fnx;
    let sum_dy2 = (gyy - 2.0 * cy_f * gy + cy_f * cy_f * fny) * fnx;

    // Unnormalised variance of dx / dy over the 2D window:
    //   varX = Σdx² − (Σdx)²/n
    //   varY = Σdy² − (Σdy)²/n
    let var_x = sum_dx2 - sum_dx * sum_dx / fn64;
    let var_y = sum_dy2 - sum_dy * sum_dy / fn64;

    if var_x <= 0.0 || var_y <= 0.0 {
        // Degenerate: single column or row — slopes undefined.
        return 0.0;
    }

    // ── O(1) SAT queries ──────────────────────────────────────────────────
    let sv = sats.query(&sats.v, x0, y0, x1, y1); // Σ v
    let sv2 = sats.query(&sats.v2, x0, y0, x1, y1); // Σ v²
    let sxv = sats.query(&sats.col_v, x0, y0, x1, y1); // Σ col·v
    let syv = sats.query(&sats.row_v, x0, y0, x1, y1); // Σ row·v

    // Mean of dx over the 2D window = Σdx / n
    let mean_dx = sum_dx / fn64;
    let mean_dy = sum_dy / fn64;

    // covarX = Σ(v·dx) − meanDx·Σv
    //        = (Σ col·v − cx·Σv) − meanDx·Σv
    // covarY: analogous with rows
    let covar_x = (sxv - cx_f * sv) - mean_dx * sv;
    let covar_y = (syv - cy_f * sv) - mean_dy * sv;

    // Algebraic reduction of residual variance:
    //   Σr² = Σv² − covarX²/varX − covarY²/varY
    //   score = (Σr² − (Σv)²/n) / (n − 1)   (see `compute_plane_fit_variance`
    //   for why the divisor is n − 1 rather than n − 3)
    let sum_sq_res = sv2 - covar_x * covar_x / var_x - covar_y * covar_y / var_y;
    let variance = (sum_sq_res - sv * sv / fn64) / (fn64 - 1.0);

    // Clamp to ≥ 0 to absorb tiny floating-point negatives near zero.
    variance.max(0.0) as f32
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the plane-fit residual-variance focus measure of the luma plane.
///
/// This is **not** a Sum-Modified Laplacian (SML); that is a different, well
/// known measure (a sum of absolute second differences) which this crate does
/// not implement.
///
/// For each pixel `(cx, cy)`, the window of half-size `radius` around it
/// (clamped to the image) is taken.  A plane `a + b·dx + c·dy` is fitted to the
/// luma values in it by least squares, and the sum of squared residuals of
/// that fit, divided by `n − 1` (`n` = pixels in the clamped window), is the
/// score.  Three parameters are fitted, so the unbiased estimator would divide
/// by `n − 3`; `n − 1` is kept because only the arg-max over frames matters
/// and the interior scale is uniform.
///
/// Properties that follow from this definition:
///
/// * A flat region and a linear ramp both score 0 up to rounding (the plane
///   fits them exactly), so a smooth gradient is not mistaken for detail.
///   (A Modified Laplacian also vanishes on a ramp, so this is not what
///   distinguishes the two measures.)
/// * Near the image border the window is clipped and `n` is smaller, so the
///   plane absorbs a larger share of the variation and border scores are
///   biased low relative to the interior (a corner window at `radius = 1` has
///   `n = 4` and one residual degree of freedom).  Windows with fewer than two
///   pixels, or a single row or column, score 0.
/// * An image with no pixels (`width == 0` or `height == 0`) returns an empty
///   map.
///
/// The output `PlanarImage` carries the map in its `luma` channel;
/// `chroma_a` and `chroma_b` are zero.  The two zero planes are 8 of the 12
/// bytes per pixel this return type costs; callers that only need the map should
/// use [`compute_focus_map`], which returns the single plane.
///
/// # Algorithm
///
/// See [`compute_focus_map`].
///
/// # Panics
/// Never in practice — all window bounds are clamped, an empty image returns an
/// empty map, and the map always holds `width * height` samples.
#[must_use]
pub fn compute_plane_fit_variance(img: &PlanarImage<f32>, radius: usize) -> PlanarImage<f32> {
    PlanarImage::with_luma(img.width, img.height, compute_focus_map(img, radius))
        .expect("the focus map holds width * height samples")
}

/// The plane-fit residual-variance focus measure of [`compute_plane_fit_variance`]
/// as a single `width × height` plane (`out[y * width + x]`, 4 bytes per pixel).
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
/// ## Memory
///
/// The tables are `f64` (an `f32` table loses the window sums on large
/// images), so a whole-image set costs 32 bytes per pixel, per frame in flight
/// under a parallel loop over frames.  Here only a ring of
/// `128 + 2·radius + 2` table rows is resident: the rows are produced in order
/// and the window of an output row reaches back at most `radius` rows.  The ring
/// holds exactly the values the whole-image tables would, so the output is
/// bit-identical to the whole-image construction (see
/// [`compute_focus_map_banded`]).
#[must_use]
pub fn compute_focus_map(img: &PlanarImage<f32>, radius: usize) -> Vec<f32> {
    compute_focus_map_banded(img, radius, DEFAULT_BAND_ROWS)
}

/// [`compute_focus_map`] with an explicit band height.
///
/// The band only sets how many output rows are evaluated in parallel between two
/// extensions of the table ring (larger bands expose more parallelism, smaller
/// ones use less memory); the output does not depend on it.  A `band_rows` of
/// `0` is treated as `1`.
///
/// # Panics
/// Never — all window bounds are clamped.
#[must_use]
pub fn compute_focus_map_banded(
    img: &PlanarImage<f32>,
    radius: usize,
    band_rows: usize,
) -> Vec<f32> {
    let width = img.width;
    let height = img.height;
    let len = width * height;
    let mut variance_map = vec![0.0_f32; len];
    if len == 0 {
        return variance_map;
    }
    let band_rows = band_rows.max(1);
    // A window larger than the image is the whole image; clamping keeps the
    // signed arithmetic below in range for any `radius`.
    let radius = radius.min(width.max(height));

    // Ring large enough that, when table row `b1 + radius` has just been
    // produced for the band `[b0, b1)`, table row `b0 − radius` (the oldest one a
    // window of that band can touch) is still resident.
    let ring_rows = (band_rows + 2 * radius + 2).min(height + 1);
    let mut sats = SatRing::new(width, ring_rows);

    let r = radius.cast_signed();
    let w = width.cast_signed();
    let h = height.cast_signed();

    // Highest table row produced so far (table row 0 is all zeros).
    let mut built = 0_usize;

    for b0 in (0..height).step_by(band_rows) {
        let b1 = (b0 + band_rows).min(height);

        // Windows of this band reach table row `b1 − 1 + radius + 1` at most.
        let need = (b1 + radius).min(height);
        while built < need {
            built += 1;
            sats.push_row(built, &img.luma[(built - 1) * width..built * width]);
        }

        let tables = &sats;
        variance_map[b0 * width..b1 * width]
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(i, row_slice)| {
                let cy = b0 + i;
                let icy = cy.cast_signed();
                let y0 = (icy - r).max(0).cast_unsigned();
                let y1 = (icy + r).min(h - 1).cast_unsigned();
                for (cx, out) in row_slice.iter_mut().enumerate() {
                    let icx = cx.cast_signed();
                    let x0 = (icx - r).max(0).cast_unsigned();
                    let x1 = (icx + r).min(w - 1).cast_unsigned();
                    *out = window_variance(tables, (cx, cy), (x0, y0, x1, y1));
                }
            });
    }

    variance_map
}
