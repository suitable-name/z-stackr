//! Warping a [`PlanarImage`] with a 3x3 matrix, and the coverage bookkeeping that
//! follows from it.
//!
//! * [`warp_image`]: Catmull-Rom bicubic, zero-filled border, `std::simd` inner
//!   loop with a scalar tail that performs the same operations in the same order;
//! * [`warp_image_clamped`]: the registration mode's four-tap C0 cubic sampler
//!   with edge-clamped borders;
//! * [`coverage_mask`], [`intersect_coverage`], [`largest_true_rectangle`]: the
//!   common valid rectangle of a stack of warped frames.
//!
//! [`warp_image`] takes the forward mapping source to destination. The
//! refinement in [`crate::refine`] returns the opposite direction (reference to
//! source), so a caller aligning a frame passes the inverse of the refined
//! matrix.
//!
//! # SIMD paths
//!
//! The vector width is chosen at compile time from the target features (see
//! `WARP_LANES`). Because the vector body and the scalar tail are
//! operation-for-operation the same, the result of a pixel does not depend on
//! which one computed it, nor on the width. [`simd_build_report`] says what the
//! build selected.

#![expect(
    clippy::cast_precision_loss,
    reason = "pixel coordinates, image dimensions and tap offsets are far below 2^24, so the usize/isize/i32 to f32 conversions are exact"
)]

use nalgebra::Matrix3;
use rayon::prelude::*;
use stacker_core::{
    error::StackerError,
    fma::{fma_f32, fma_simd},
    image::PlanarImage,
};
use std::simd::{StdFloat, prelude::*};

/// SIMD width (in `f32` lanes) of the warp inner loop, selected from the
/// compile-time target features: 16 (one `zmm`) with `avx512f`, 8 (one `ymm`)
/// with `avx2`, otherwise 4 (one `xmm`, the SSE2 baseline and NEON).
///
/// `std::simd` has no runtime dispatch: the vector type is resolved when the
/// crate is compiled, so the width follows `-C target-cpu` / `-C
/// target-feature`. `mul_add` additionally needs the `fma` feature, without
/// which it is an exactly rounded library call per lane (same bits, lower
/// speed). Every operation in the body is elementwise (no horizontal reduction),
/// so the width changes speed only: widths 4, 8 and 16 produce identical pixels,
/// and the scalar tail covers the last `width % WARP_LANES` columns with the same
/// arithmetic. See [`simd_build_report`] and the "CPU features" paragraph of the
/// root README for the `RUSTFLAGS` that select a CPU level.
const WARP_LANES: usize = if cfg!(target_feature = "avx512f") {
    16
} else if cfg!(target_feature = "avx2") {
    8
} else {
    4
};
type WarpSimd = Simd<f32, WARP_LANES>;

/// `true` when this build was compiled with hardware fused multiply-add
/// (`target_feature = "fma"`), i.e. [`StdFloat::mul_add`] lowers to an
/// instruction instead of a per-lane `fmaf` library call.
///
/// Warp and pyramid results do not depend on it (`mul_add` is exact either
/// way), only the speed does. On `wasm32` the code uses unfused `a * b + c`
/// instead (see [`stacker_core::fma`]), so results there may differ in the last
/// bits.
#[must_use]
pub const fn has_hardware_fma() -> bool {
    cfg!(target_feature = "fma")
}

/// One-line description of what the SIMD code was compiled for.
///
/// Meant for a start-up log line.  The vector width is fixed at compile time,
/// so this is a property of the binary, not of the CPU it happens to run on.
#[must_use]
pub const fn simd_build_report() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "avx512f: 16-lane f32 vectors are single zmm operations"
    } else if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
        "avx2+fma: 8-lane f32 vectors are single ymm operations, mul_add is an instruction"
    } else if cfg!(target_feature = "avx2") {
        "avx2 without fma: 8-lane f32 vectors, mul_add is a library call per lane; add target-feature=+fma"
    } else if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        "baseline x86 (SSE2): 4-lane f32 vectors are single xmm operations and mul_add is a \
         library call per lane; build with RUSTFLAGS=\"-C target-cpu=x86-64-v3\""
    } else {
        "non-x86 target: 4-lane f32 vectors lower to the target's native vector width"
    }
}

/// Vectorised Catmull-Rom cubic kernel — the lane-wise equivalent of [`cubic`],
/// using the identical `mul_add` expressions so each lane is bit-for-bit equal
/// to the scalar version.
#[inline]
fn cubic_simd(t: WarpSimd) -> WarpSimd {
    let abs = t.abs();
    let abs2 = abs * abs;
    let abs3 = abs2 * abs;
    // Inner kernel (|t| ≤ 1):  1.5t³ − 2.5t² + 1
    let inner = fma_simd(
        WarpSimd::splat(2.5),
        -abs2,
        fma_simd(WarpSimd::splat(1.5), abs3, WarpSimd::splat(1.0)),
    );
    // Outer kernel (1 < |t| < 2):  −0.5t³ + 2.5t² − 4t + 2
    let outer = fma_simd(
        WarpSimd::splat(4.0),
        -abs,
        fma_simd(
            WarpSimd::splat(2.5),
            abs2,
            fma_simd(WarpSimd::splat(-0.5), abs3, WarpSimd::splat(2.0)),
        ),
    );
    let le1 = abs.simd_le(WarpSimd::splat(1.0));
    let lt2 = abs.simd_lt(WarpSimd::splat(2.0));
    le1.select(inner, lt2.select(outer, WarpSimd::splat(0.0)))
}

/// Inverse-mapping coefficients for one output row.
///
/// For a fixed output row `y` every homogeneous component of `m_inv · (x, y, 1)`
/// is affine in `x`: `p = m*0 · x + (m*1 · y + m*2)`.  This is the single
/// definition of the output → source coordinate used by [`warp_image`] (SIMD
/// body and scalar tail), [`warp_image_clamped`] and [`coverage_mask`], so the
/// mask and the warp's zero-fill border agree on every pixel, including those
/// whose source coordinate lands within an ulp of the frame edge.
#[derive(Clone, Copy)]
struct RowMap {
    m00: f32,
    m10: f32,
    m20: f32,
    c0: f32,
    c1: f32,
    c2: f32,
}

// `suboptimal_flops`: the coordinates are intentionally computed with separate
// mul + add (not fused `mul_add`) so every consumer, including the SIMD lanes,
// produces bit-identical values.
#[expect(
    clippy::suboptimal_flops,
    reason = "unfused mul+add keeps the SIMD, scalar and mask paths bit-identical"
)]
impl RowMap {
    #[inline]
    fn new(m_inv: &Matrix3<f32>, y: usize) -> Self {
        let yf = y as f32;
        Self {
            m00: m_inv[(0, 0)],
            m10: m_inv[(1, 0)],
            m20: m_inv[(2, 0)],
            c0: m_inv[(0, 1)] * yf + m_inv[(0, 2)],
            c1: m_inv[(1, 1)] * yf + m_inv[(1, 2)],
            c2: m_inv[(2, 1)] * yf + m_inv[(2, 2)],
        }
    }

    /// Source coordinate `(sx, sy)` of output column `x` (homogeneous divide by
    /// reciprocal multiply, as in the SIMD body).  Non-finite when the divisor
    /// is zero.
    #[inline]
    fn source_at(&self, x: f32) -> (f32, f32) {
        let inv = 1.0 / (self.m20 * x + self.c2);
        (
            (self.m00 * x + self.c0) * inv,
            (self.m10 * x + self.c1) * inv,
        )
    }
}

/// `true` iff `(x, y)` is finite and inside `[0, width-1] × [0, height-1]`, the
/// region [`bicubic_interpolate_channel`] samples; anything else is zero-fill.
#[inline]
pub(crate) fn source_in_bounds(x: f32, y: f32, width: usize, height: usize) -> bool {
    x.is_finite()
        && y.is_finite()
        && x >= 0.0
        && y >= 0.0
        && x <= (width as f32 - 1.0)
        && y <= (height as f32 - 1.0)
}

fn validate_matrix(matrix: &Matrix3<f32>) -> Result<(), StackerError> {
    if !matrix.iter().all(|&v| v.is_finite()) {
        return Err(StackerError::MathError("non-finite matrix element".into()));
    }
    Ok(())
}

#[inline]
fn cubic(x: f32) -> f32 {
    let abs_x = x.abs();
    let abs_x2 = abs_x * abs_x;
    let abs_x3 = abs_x2 * abs_x;
    if abs_x <= 1.0 {
        // Catmull-Rom inner kernel: 1.5t³ − 2.5t² + 1
        fma_f32(2.5, -abs_x2, fma_f32(1.5, abs_x3, 1.0))
    } else if abs_x < 2.0 {
        // Catmull-Rom outer kernel: −0.5t³ + 2.5t² − 4t + 2
        fma_f32(
            4.0,
            -abs_x,
            fma_f32(2.5, abs_x2, fma_f32(-0.5, abs_x3, 2.0)),
        )
    } else {
        0.0
    }
}

/// Bicubically interpolate a single planar channel at `(x, y)`.
///
/// Returns `0.0` for out-of-range or non-finite coordinates instead of
/// panicking — this is the NaN safety boundary: a near-singular matrix can
/// produce huge / NaN projected coordinates after the homogeneous divide, and
/// we clamp them here rather than propagating them into the output.
///
/// The intensity-refinement module calls it directly so it need not warp all
/// three channels; it is public only so the integration tests can pin it.
#[doc(hidden)]
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the coordinate is range-checked by source_in_bounds, so floor() fits an isize"
)]
pub fn bicubic_interpolate_channel(
    plane: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> f32 {
    // Reject NaN, infinities, and out-of-bounds coords early.
    if !source_in_bounds(x, y, width, height) {
        return 0.0;
    }

    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let dx = x - x.floor();
    let dy = y - y.floor();

    // Width/height are pixel dimensions that cannot exceed isize::MAX on any
    // supported platform — a usize→isize cast wrapping would require an image
    // larger than the address space.
    #[expect(
        clippy::cast_possible_wrap,
        reason = "dimensions never exceed isize::MAX"
    )]
    let (iwidth, iheight) = (width as isize, height as isize);

    // x weights and clamped x taps do not depend on the tap row: compute the
    // four of each once, not once per tap row.
    let mut weights_x = [0.0_f32; 4];
    let mut taps_x = [0_usize; 4];
    for (k, i) in (-1_isize..=2).enumerate() {
        weights_x[k] = cubic(dx - i as f32);
        taps_x[k] = (x0 + i).clamp(0, iwidth - 1).cast_unsigned();
    }

    let mut sum = 0.0_f32;
    for j in -1_isize..=2 {
        let weight_y = cubic(dy - j as f32);
        let mut row_sum = 0.0_f32;
        let py = (y0 + j).clamp(0, iheight - 1).cast_unsigned();
        let row_offset = py * width;
        for (&weight_x, &px) in weights_x.iter().zip(taps_x.iter()) {
            row_sum = fma_f32(weight_x, plane[row_offset + px], row_sum);
        }
        sum = fma_f32(weight_y, row_sum, sum);
    }
    sum
}

/// Warp `src` by the forward mapping `matrix` (src → dst) using
/// inverse-mapping Catmull-Rom bicubic interpolation (4 × 4 taps).
///
/// All three planes (`luma`, `chroma_a`, `chroma_b`) are warped independently.
/// Output pixels whose inverse-mapped source coordinate falls outside
/// `[0, width-1] × [0, height-1]` are zero-filled; [`coverage_mask`] marks
/// exactly the pixels that are not.
///
/// To bring a frame into the reference frame with a matrix `M` from
/// [`crate::refine`] (which maps reference coordinates to source coordinates),
/// pass `M⁻¹` here.
///
/// # Errors
/// Returns [`StackerError::MathError`] when `matrix` is non-finite or
/// non-invertible.
// `suboptimal_flops`: the inverse-map coordinates are intentionally computed
// with separate mul + add (not fused `mul_add`) so the SIMD body, scalar tail,
// and the reference test stay bit-for-bit identical.
// Single fully-vectorised warp (SIMD body + scalar tail for all three planes)
// kept in one function so the documented bit-for-bit accumulation order holds.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "one fully vectorised warp (SIMD body + scalar tail) so the bit-for-bit accumulation order holds; image dimensions fit an i32 lane index; the a/b plane names are paired"
)]
pub fn warp_image(
    src: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> Result<PlanarImage<f32>, StackerError> {
    validate_matrix(matrix)?;

    let m_inv = matrix
        .try_inverse()
        .ok_or_else(|| StackerError::MathError("non-invertible warp matrix".into()))?;

    let width = src.width;
    let height = src.height;
    let mut dst = PlanarImage::new(width, height);

    let luma_src = &src.luma;
    let ca_src = &src.chroma_a;
    let cb_src = &src.chroma_b;

    // Parallel row-wise warp for all three channels simultaneously.
    dst.luma
        .par_chunks_mut(width)
        .zip(dst.chroma_a.par_chunks_mut(width))
        .zip(dst.chroma_b.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, ((row_l, row_a), row_b))| {
            // Inverse-map coefficients, shared with `coverage_mask`.
            let row_map = RowMap::new(&m_inv, y);
            let RowMap {
                m00,
                m10,
                m20,
                c0,
                c1,
                c2,
            } = row_map;

            let mut x = 0usize;

            // ── SIMD body: WARP_LANES output pixels per iteration ─────────────
            let lane_f = WarpSimd::from_array(std::array::from_fn(|k| k as f32));
            let zero = WarpSimd::splat(0.0);
            let wm1 = WarpSimd::splat(width as f32 - 1.0);
            let hm1 = WarpSimd::splat(height as f32 - 1.0);
            let wi1 = Simd::<i32, WARP_LANES>::splat(width as i32 - 1);
            let hi1 = Simd::<i32, WARP_LANES>::splat(height as i32 - 1);
            let zi = Simd::<i32, WARP_LANES>::splat(0);
            let wide = Simd::<usize, WARP_LANES>::splat(width);

            while x + WARP_LANES <= width {
                let xf = WarpSimd::splat(x as f32) + lane_f;
                let p0 = WarpSimd::splat(m00) * xf + WarpSimd::splat(c0);
                let p1 = WarpSimd::splat(m10) * xf + WarpSimd::splat(c1);
                let p2 = WarpSimd::splat(m20) * xf + WarpSimd::splat(c2);
                let inv = WarpSimd::splat(1.0) / p2;
                let sx = p0 * inv;
                let sy = p1 * inv;

                // Lanes whose source coordinate is non-finite or out of bounds
                // get zeroed — matching `bicubic_interpolate_channel`.
                let valid = sx.is_finite()
                    & sy.is_finite()
                    & sx.simd_ge(zero)
                    & sy.simd_ge(zero)
                    & sx.simd_le(wm1)
                    & sy.simd_le(hm1);

                let x0 = sx.floor();
                let y0 = sy.floor();
                let dx = sx - x0;
                let dy = sy - y0;
                let x0i = x0.cast::<i32>();
                let y0i = y0.cast::<i32>();

                // The four x weights and clamped x taps depend only on the
                // lane's fractional x, not on the tap row `j`: compute them
                // once per vector instead of once per (j, i) pair (4 instead of
                // 16 `cubic_simd` calls and clamps).  They are pure functions of
                // the lane's inputs, so each lane equals the scalar sampler.
                let mut wxs = [zero; 4];
                let mut pxs = [Simd::<usize, WARP_LANES>::splat(0); 4];
                for (k, i) in (-1..=2_i32).enumerate() {
                    wxs[k] = cubic_simd(dx - WarpSimd::splat(i as f32));
                    pxs[k] = (x0i + Simd::splat(i)).simd_clamp(zi, wi1).cast::<usize>();
                }

                let mut acc_l = zero;
                let mut acc_a = zero;
                let mut acc_b = zero;
                for j in -1..=2_i32 {
                    let wy = cubic_simd(dy - WarpSimd::splat(j as f32));
                    let py = (y0i + Simd::splat(j)).simd_clamp(zi, hi1).cast::<usize>();
                    let row_base = py * wide;
                    let mut row_l_acc = zero;
                    let mut row_a_acc = zero;
                    let mut row_b_acc = zero;
                    for (&wx, &px) in wxs.iter().zip(pxs.iter()) {
                        let idx = row_base + px;
                        let s_l = WarpSimd::gather_or(luma_src, idx, zero);
                        let s_a = WarpSimd::gather_or(ca_src, idx, zero);
                        let s_b = WarpSimd::gather_or(cb_src, idx, zero);
                        row_l_acc = fma_simd(wx, s_l, row_l_acc);
                        row_a_acc = fma_simd(wx, s_a, row_a_acc);
                        row_b_acc = fma_simd(wx, s_b, row_b_acc);
                    }
                    acc_l = fma_simd(wy, row_l_acc, acc_l);
                    acc_a = fma_simd(wy, row_a_acc, acc_a);
                    acc_b = fma_simd(wy, row_b_acc, acc_b);
                }

                valid
                    .select(acc_l, zero)
                    .copy_to_slice(&mut row_l[x..x + WARP_LANES]);
                valid
                    .select(acc_a, zero)
                    .copy_to_slice(&mut row_a[x..x + WARP_LANES]);
                valid
                    .select(acc_b, zero)
                    .copy_to_slice(&mut row_b[x..x + WARP_LANES]);
                x += WARP_LANES;
            }

            // ── Scalar tail (and the whole row for images narrower than a vector) ──
            while x < width {
                let (src_x, src_y) = row_map.source_at(x as f32);
                row_l[x] = bicubic_interpolate_channel(luma_src, width, height, src_x, src_y);
                row_a[x] = bicubic_interpolate_channel(ca_src, width, height, src_x, src_y);
                row_b[x] = bicubic_interpolate_channel(cb_src, width, height, src_x, src_y);
                x += 1;
            }
        });

    Ok(dst)
}

// ── Four-tap C0 cubic warp (edge-clamped border) ──────────────────────────────
//
// The registration alignment mode samples with a four-tap piecewise-cubic
// kernel and edge-clamped borders (instead of the zero-fill used by
// `warp_image`/`bicubic_interpolate_channel`). Edge clamping keeps the warped
// border from biasing the registration objective and avoids a hard black seam
// in the output.
//
// The kernel is **not** Catmull-Rom, and it is not claimed to reproduce any
// external engine's interpolation (that cannot be verified from this
// repository). Its properties, all pinned by the unit tests at the end of this
// file:
//
// - the four weights sum to 1 for every fractional offset (partition of unity),
//   so a constant plane stays constant;
// - it interpolates: at `t = 0` the weights are exactly `[0, 1, 0, 0]`;
// - it is continuous across knots (weights at `t -> 1` equal the shifted
//   weights at `t = 0`) but only C0: the one-sided slopes differ at the knots
//   (the centre tap leaves `t = 0` with slope -0.2 where Catmull-Rom has 0), so
//   the sampled image has a derivative jump at every integer coordinate and the
//   registration objective is kinked at integer shifts.
//
// The numerics are kept as they are because the registration path's recorded
// results depend on them; the (public) sampler keeps its name for the same
// reason.

/// Weights of the four-tap C0 cubic kernel for fractional offset `t` in
/// `[0, 1)`, for the taps at `floor - 1 ..= floor + 2`.
///
/// Not Catmull-Rom (see the section comment above); the weights are a partition
/// of unity and the kernel interpolates (`t = 0` gives `[0, 1, 0, 0]`).
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "the Horner form is evaluated unfused, as the registration path's recorded results require"
)]
fn c0_cubic4_weights(t: f32) -> [f32; 4] {
    [
        ((-0.333_333_34_f32 * t + 0.8) * t - 0.466_666_67) * t,
        ((t - 1.8) * t - 0.2) * t + 1.0,
        ((1.2 - t) * t + 0.8) * t,
        ((0.333_333_34_f32 * t - 0.2) * t - 0.133_333_34) * t,
    ]
}

/// Sample `plane` at `(x, y)` with the four-tap C0 cubic kernel (not
/// Catmull-Rom; see the section comment above) and edge-clamped borders.
///
/// Out-of-range taps are clamped to the nearest valid pixel, so no zero-fill
/// border is introduced.  The division by the weight sum on the border path is a
/// no-op up to rounding (clamping redirects taps, it does not drop weights, and
/// the 16 weights always sum to 1); it stays only because removing it would
/// change the last bits of the registration path's results.  Returns `0.0` for
/// non-finite coordinates.
///
/// Public only so the integration tests can pin the sampler; the callers are
/// [`warp_image_clamped`] and the registration objective.
#[doc(hidden)]
#[must_use]
pub fn spline4x4_sample_clamped(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    SplineTaps::new(x, y).map_or(0.0, |taps| taps.sample(plane, width, height))
}

/// Integer tap origin and the two sets of spline weights for one source
/// coordinate.  They depend only on `(x, y)`, so the three-plane warp computes
/// them once per pixel and samples all planes with them.
struct SplineTaps {
    idx_x: isize,
    idx_y: isize,
    wx: [f32; 4],
    wy: [f32; 4],
}

impl SplineTaps {
    /// `None` for a non-finite coordinate.
    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the tap origin is floor() of a source coordinate; an out-of-range one is clamped later"
    )]
    fn new(x: f32, y: f32) -> Option<Self> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        Some(Self {
            idx_x: x.floor() as isize,
            idx_y: y.floor() as isize,
            wx: c0_cubic4_weights(x - x.floor()),
            wy: c0_cubic4_weights(y - y.floor()),
        })
    }

    /// Weighted 4×4 sum over `plane` (fast in-range path, or the clamped and
    /// renormalised border path).
    #[inline]
    fn sample(&self, plane: &[f32], width: usize, height: usize) -> f32 {
        let Self {
            idx_x,
            idx_y,
            wx,
            wy,
        } = self;
        spline_sample(plane, width, height, *idx_x, *idx_y, wx, wy)
    }
}

#[expect(
    clippy::cast_possible_wrap,
    reason = "dimensions and tap offsets never exceed isize::MAX"
)]
fn spline_sample(
    plane: &[f32],
    width: usize,
    height: usize,
    idx_x: isize,
    idx_y: isize,
    wx: &[f32; 4],
    wy: &[f32; 4],
) -> f32 {
    let iw = width as isize;
    let ih = height as isize;

    // Fast path: all four taps in range (centre taps at idx-1..=idx+2).
    if idx_x >= 1 && idx_x <= iw - 3 && idx_y >= 1 && idx_y <= ih - 3 {
        let mut val = 0.0_f32;
        for (ky, &wyk) in wy.iter().enumerate() {
            let row = (idx_y - 1 + ky as isize).cast_unsigned() * width;
            let mut row_sum = 0.0_f32;
            for (kx, &wxk) in wx.iter().enumerate() {
                let px = (idx_x - 1 + kx as isize).cast_unsigned();
                row_sum = fma_f32(wxk, plane[row + px], row_sum);
            }
            val = fma_f32(wyk, row_sum, val);
        }
        return val;
    }

    // Border path: clamp tap indices to the edge and renormalise by weight sum.
    let mut sum_w = 0.0_f32;
    let mut acc = 0.0_f32;
    for (ky, &wyk) in wy.iter().enumerate() {
        let cy = (idx_y - 1 + ky as isize).clamp(0, ih - 1).cast_unsigned();
        for (kx, &wxk) in wx.iter().enumerate() {
            let cx = (idx_x - 1 + kx as isize).clamp(0, iw - 1).cast_unsigned();
            let w = wxk * wyk;
            sum_w += w;
            acc = fma_f32(w, plane[cy * width + cx], acc);
        }
    }
    if sum_w.abs() < 1.0e-12 {
        0.0
    } else {
        acc / sum_w
    }
}

/// Warp `src` by the forward mapping `matrix` (src → dst) using the four-tap C0
/// cubic kernel (not Catmull-Rom; see [`spline4x4_sample_clamped`]) with
/// edge-clamped borders.
///
/// All three planes are warped independently. This is the registration mode's
/// counterpart to [`warp_image`] (which uses Catmull-Rom bicubic interpolation
/// with a zero-fill border).
///
/// # Errors
/// Returns [`StackerError::MathError`] when `matrix` is non-finite or
/// non-invertible.
#[expect(
    clippy::similar_names,
    reason = "the a/b chroma plane names are paired"
)]
pub fn warp_image_clamped(
    src: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> Result<PlanarImage<f32>, StackerError> {
    validate_matrix(matrix)?;
    let m_inv = matrix
        .try_inverse()
        .ok_or_else(|| StackerError::MathError("non-invertible warp matrix".into()))?;

    let width = src.width;
    let height = src.height;
    let mut dst = PlanarImage::new(width, height);

    let luma_src = &src.luma;
    let ca_src = &src.chroma_a;
    let cb_src = &src.chroma_b;

    dst.luma
        .par_chunks_mut(width)
        .zip(dst.chroma_a.par_chunks_mut(width))
        .zip(dst.chroma_b.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, ((row_l, row_a), row_b))| {
            let row_map = RowMap::new(&m_inv, y);

            for x in 0..width {
                let (src_x, src_y) = row_map.source_at(x as f32);
                // Weights depend on the coordinate only: build them once and
                // sample all three planes with them, not once per plane.
                if let Some(taps) = SplineTaps::new(src_x, src_y) {
                    row_l[x] = taps.sample(luma_src, width, height);
                    row_a[x] = taps.sample(ca_src, width, height);
                    row_b[x] = taps.sample(cb_src, width, height);
                } else {
                    row_l[x] = 0.0;
                    row_a[x] = 0.0;
                    row_b[x] = 0.0;
                }
            }
        });

    Ok(dst)
}

// ── Common valid-area cropping ────────────────────────────────────────────────
//
// Warping a frame into the reference coordinate system zero-fills the pixels
// that map outside the source image (a black border whose thickness depends on
// the frame's shift / scale / rotation). Fusing the full frames therefore bleeds
// those black borders into the result — visible as dark "frame edges".
//
// The fix is to crop the output to the region covered by *every* frame: build a
// per-frame coverage mask, intersect them, and crop to the largest axis-aligned
// rectangle that is valid in all frames.

/// Per-pixel coverage mask for warping a frame by `matrix`.
///
/// `mask[y * width + x]` is `true` iff the inverse-mapped source coordinate for
/// output pixel `(x, y)` falls inside the source bounds `[0, width-1] ×
/// [0, height-1]` — i.e. that output pixel receives real data from
/// [`warp_image`] rather than the zero-fill border.  The mask uses the same
/// per-row inverse-mapping routine (`RowMap`) and the same in-range predicate
/// as [`warp_image`], so the two agree bit-for-bit on every pixel.
///
/// A non-invertible or non-finite `matrix` yields an all-`false` mask.
#[must_use]
pub fn coverage_mask(matrix: &Matrix3<f32>, width: usize, height: usize) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    if width == 0 || height == 0 || !matrix.iter().all(|v| v.is_finite()) {
        return mask;
    }
    let Some(m_inv) = matrix.try_inverse() else {
        return mask;
    };

    mask.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let row_map = RowMap::new(&m_inv, y);
        for (x, cell) in row.iter_mut().enumerate() {
            let (sx, sy) = row_map.source_at(x as f32);
            *cell = source_in_bounds(sx, sy, width, height);
        }
    });
    mask
}

/// Intersect (`AND`) `other` into `acc` in place.
///
/// Used to accumulate the common coverage of a whole stack: start from an
/// all-`true` mask (e.g. the reference frame) and fold each frame's
/// [`coverage_mask`] in.
///
/// # Panics
/// Panics if the two masks differ in length.
pub fn intersect_coverage(acc: &mut [bool], other: &[bool]) {
    assert_eq!(
        acc.len(),
        other.len(),
        "coverage masks must be the same size"
    );
    for (a, b) in acc.iter_mut().zip(other.iter()) {
        *a &= *b;
    }
}

/// Largest axis-aligned all-`true` rectangle in a boolean mask, returned as
/// `(x, y, w, h)`. Returns `None` if the mask is empty or fully `false`.
///
/// Runs in `O(width * height)` using the classic maximal-rectangle-in-histogram
/// algorithm (per-row bar heights + a monotonic stack), so it is cheap even at
/// full sensor resolution.
#[must_use]
pub fn largest_true_rectangle(
    mask: &[bool],
    width: usize,
    height: usize,
) -> Option<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 || mask.len() != width * height {
        return None;
    }

    let mut heights = vec![0usize; width];
    let mut best_area = 0usize;
    let mut best_rect: Option<(usize, usize, usize, usize)> = None;

    for y in 0..height {
        let row = &mask[y * width..(y + 1) * width];
        for (h, &valid) in heights.iter_mut().zip(row.iter()) {
            *h = if valid { *h + 1 } else { 0 };
        }

        // Largest rectangle in the `heights` histogram, tracking position.
        let mut stack: Vec<usize> = Vec::with_capacity(width + 1);
        let mut x = 0usize;
        while x <= width {
            let cur = if x == width { 0 } else { heights[x] };
            match stack.last().copied() {
                // Pop bars taller than the current one and score their rectangles.
                Some(top) if cur < heights[top] => {
                    stack.pop();
                    let bar_h = heights[top];
                    let left = stack.last().map_or(0, |&l| l + 1);
                    let rect_w = x - left;
                    let area = rect_w * bar_h;
                    if bar_h > 0 && area > best_area {
                        best_area = area;
                        best_rect = Some((left, y + 1 - bar_h, rect_w, bar_h));
                    }
                }
                _ => {
                    stack.push(x);
                    x += 1;
                }
            }
        }
    }

    best_rect
}

#[cfg(test)]
mod tests {
    use super::c0_cubic4_weights;

    /// ALN-11: the kernel is a partition of unity and interpolates at the knot.
    #[test]
    fn c0_kernel_is_a_partition_of_unity_and_interpolates() {
        for i in 0..=100_u8 {
            let t = f32::from(i) / 100.0;
            let sum: f32 = c0_cubic4_weights(t).iter().sum();
            assert!(
                (sum - 1.0).abs() < 1.0e-5,
                "weights at t = {t} sum to {sum}"
            );
        }
        assert_eq!(c0_cubic4_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
    }

    /// ALN-11: continuous across a knot (the weights at `t -> 1` are the weights
    /// at `t = 0` shifted by one tap) but only C0: the centre tap leaves `t = 0`
    /// with slope -0.2, where Catmull-Rom's slope is 0.
    #[test]
    fn c0_kernel_is_continuous_but_has_a_slope_jump_at_the_knot() {
        let end = c0_cubic4_weights(1.0);
        let start = c0_cubic4_weights(0.0);
        // The window moves one pixel right: tap k at t = 1 is tap k - 1 at t = 0,
        // and the tap that left the window carries no weight.
        for (k, (e, s)) in end[1..].iter().zip(&start[..3]).enumerate() {
            assert!(
                (e - s).abs() < 1.0e-5,
                "tap {}: {end:?} vs {start:?}",
                k + 1
            );
        }
        assert!(end[0].abs() < 1.0e-5, "tap leaving the window: {end:?}");

        let h = 1.0e-3_f32;
        let slope = (c0_cubic4_weights(h)[1] - c0_cubic4_weights(0.0)[1]) / h;
        assert!(
            (slope + 0.2).abs() < 5.0e-3,
            "centre-tap slope at 0+ is {slope}, expected -0.2"
        );
    }
}
