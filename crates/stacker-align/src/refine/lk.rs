//! Pyramid Lucas-Kanade / Gauss-Newton refinement: the analytic-gradient
//! alternative to the Nelder-Mead simplex.
//!
//! The objective is the one the simplex minimises (the mean-corrected RMS of
//! `source - warped reference` over the pixels that received a sample, see
//! [`LevelObjective`]), solved with Gauss-Newton steps instead of derivative-free
//! simplex moves.  It handles the whole [`AffineParams`] model (shift, scale,
//! rotation, aspect, shear) gated by the `allow_*` flags of
//! [`BoundedRefineOptions`].
//!
//! ## Conventions
//!
//! Exactly those of the rest of the crate.  The matrix `M` maps **reference
//! coordinates to source coordinates**; the warped reference is
//! `warped(x) = reference(M⁻¹ x)` and the residual is `e = warped - source`.
//! The luma planes are the linear-light ones the loader produces, pixels without
//! a reference sample (`NaN` from the Catmull-Rom sampler) take no part, and the
//! mean of the residual is removed (a global exposure offset is not a
//! misalignment).
//!
//! ## Jacobian
//!
//! Differentiating `warped(x; p) = reference(M(p)⁻¹ x)` with respect to a
//! parameter `p_k`, with `u = M⁻¹ x` the matching reference point and `g` the
//! gradient of the *warped* image:
//!
//! ```text
//! ∂warped/∂p_k = ∇reference(u) · ∂(M⁻¹ x)/∂p_k
//!              = ∇reference(u) · ( -M⁻¹ (∂M/∂p_k) u )
//!              = - g(x) · ( A_k u + t_k )
//! ```
//!
//! where `[A_k | t_k] = ∂M/∂p_k` is the 2x3 derivative of the affine matrix (a
//! central finite difference of the closed form of [`AffineParams`], exact for
//! the shifts and accurate to ~1e-10 for the others) and `g = M_linᵀ ∇reference`
//! was used to move the gradient to the warped image.  The mean correction is
//! applied exactly: both the residual and every Jacobian column are centred over
//! the pixels that take part, which is the Gauss-Newton step of the
//! mean-corrected objective.
//!
//! ## Step rule
//!
//! Levenberg-damped normal equations `(H + λ diag H) δ = -Jᵀ e`, the standard
//! trust-region schedule (λ / 3 after an accepted step, λ · 3 and a retry after a
//! rejected one, at most 8 retries), an additive update `p ← p + δ` and a clamp
//! to the same half-width intervals the bounded simplex searches
//! (`max_shift_*`, `max_scale`, `max_rotation_deg`, `max_aspect`, `max_shear`)
//! around the seed.  A step is accepted when the objective, evaluated by the very
//! code the simplex uses, does not get worse.
//!
//! ## Level schedule
//!
//! Coarse to fine on the [`LumaPyramid`], each level warm-started from the one
//! before (the parameters are resolution independent).  The schedule is that of
//! GitHub z-stackr: a *large* level (short side above 512 pixels) is not worth
//! the work, so among the finest two levels the first large one ends the loop
//! and any other large level runs only 3 iterations; the remaining levels run up
//! to 20.  For a typical 20-megapixel frame LK therefore stops at the level of
//! roughly 1000 pixels.  A small image (short side up to 512) is refined at full
//! resolution too (GitHub skipped that level; there it costs nothing and
//! recovers the last fraction of a pixel), and an image so small that the rule
//! would process nothing is refined at its only level.
//!
//! ## Determinism and platforms
//!
//! The normal equations are accumulated per row and the row partials are added
//! serially in row order, so the result does not depend on the thread count.  LK
//! has no GPU form: [`BoundedRefineOptions::device`] is ignored (the GPU
//! objective serves the simplex only), and it is not part of the ask/tell
//! machine of [`super::search`], which stays Nelder-Mead.

#![allow(
    clippy::cast_precision_loss,
    reason = "pixel indices and counts are far below 2^52, so the usize to f64 conversions are exact"
)]
#![allow(
    clippy::suboptimal_flops,
    reason = "plain multiply-add keeps the accumulation order explicit and platform independent"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "x/y, u/v and the matrix entries are the textbook names"
)]
#![allow(
    clippy::needless_range_loop,
    reason = "the normal equations index several small fixed-size arrays by the same DOF index"
)]

use nalgebra::Matrix3;
use rayon::prelude::*;
use stacker_core::{error::StackerError, image::PlanarImage};

use super::{
    AffineParams, BoundedRefineOptions, LevelObjective, LevelView, LumaPyramid, PENALTY,
    SERIAL_LEVEL_PIXELS, Warper, check_pyramid, seed_params,
};
use crate::device::ObjectiveSampler;

/// Parameter-step norm below which a level has converged.  The parameters mix
/// units (fractions of the image size, radians, dimensionless ratios) but all are
/// of order one, so one small threshold serves them.
const LK_STEP_EPS: f64 = 1.0e-8;

/// Relative objective improvement below which a level has converged.
const LK_REL_IMPROVEMENT: f64 = 1.0e-10;

/// Iterations of a level that is not *large*.
const LK_MAX_ITER_PER_LEVEL: usize = 20;

/// Short side (pixels) above which a level is *large*.
const LK_LARGE_SHORT_SIDE: usize = 512;

/// Iterations of a large level that is processed at all.
const LK_LARGE_LEVEL_ITERS: usize = 3;

/// Initial Levenberg damping.
const LM_LAMBDA_INIT: f64 = 1.0e-3;
/// Factor λ grows by after a rejected step.
const LM_LAMBDA_UP: f64 = 3.0;
/// Factor λ shrinks by after an accepted step.
const LM_LAMBDA_DOWN: f64 = 1.0 / 3.0;
/// Floor of λ.
const LM_LAMBDA_MIN: f64 = 1.0e-12;
/// Ceiling of λ.
const LM_LAMBDA_MAX: f64 = 1.0e6;
/// Retries (with a larger λ) of one outer iteration before the level stops.
const LM_MAX_REJECTIONS: usize = 8;

/// Step of the central difference that turns the closed-form matrix into its
/// derivative with respect to a parameter.
const FD_STEP: f64 = 1.0e-6;

/// Fewest pixels that must take part for a Gauss-Newton step to be meaningful.
const MIN_PIXELS: f64 = 16.0;

/// The result of a Lucas-Kanade refinement.
#[derive(Debug, Clone, Copy)]
pub struct LkResult {
    /// The refined matrix, mapping reference coordinates to source coordinates.
    pub matrix: Matrix3<f32>,
    /// The objective at the seed, evaluated at the *finest level the schedule
    /// processes* (never a large one, see the module documentation).
    ///
    /// GitHub measured this at the coarsest processed level and the final value
    /// at the finest, so its `final <= starting` test compared RMS values of
    /// different resolutions; both are measured at the same level here.
    pub starting_rms: f64,
    /// The objective at the result, at the same level as
    /// [`Self::starting_rms`].
    pub final_rms: f64,
}

/// The degrees of freedom, in the fixed column order of the normal equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LkDof {
    ShiftX,
    ShiftY,
    Scale,
    Rotate,
    Aspect,
    Shear,
}

impl LkDof {
    /// Add `d` to this DOF of `p`.
    fn add(self, p: &mut AffineParams, d: f64) {
        match self {
            Self::ShiftX => p.tx += d,
            Self::ShiftY => p.ty += d,
            Self::Scale => p.scale += d,
            Self::Rotate => p.rotate += d,
            Self::Aspect => p.aspect += d,
            Self::Shear => p.shear += d,
        }
    }
}

/// The DOFs `o` enables, in column order.
fn active_dofs(o: &BoundedRefineOptions) -> Vec<LkDof> {
    let mut active = Vec::with_capacity(6);
    if o.allow_shift_x {
        active.push(LkDof::ShiftX);
    }
    if o.allow_shift_y {
        active.push(LkDof::ShiftY);
    }
    if o.allow_scale {
        active.push(LkDof::Scale);
    }
    if o.allow_rotation {
        active.push(LkDof::Rotate);
    }
    if o.allow_aspect {
        active.push(LkDof::Aspect);
    }
    if o.allow_shear {
        active.push(LkDof::Shear);
    }
    active
}

/// Per-field clamp intervals around the seed.
struct DofBounds {
    lower: AffineParams,
    upper: AffineParams,
}

impl DofBounds {
    fn new(p0: &AffineParams, o: &BoundedRefineOptions) -> Self {
        let d_rot = o.max_rotation_deg.to_radians();
        let d_scale = o.max_scale / 100.0;
        let d_x = o.max_shift_x / 100.0;
        let d_y = o.max_shift_y / 100.0;
        let d_aspect = o.max_aspect / 100.0;
        let d_shear = o.max_shear / 100.0;
        Self {
            lower: AffineParams {
                tx: p0.tx - d_x,
                ty: p0.ty - d_y,
                scale: p0.scale - d_scale,
                rotate: p0.rotate - d_rot,
                aspect: p0.aspect - d_aspect,
                shear: p0.shear - d_shear,
            },
            upper: AffineParams {
                tx: p0.tx + d_x,
                ty: p0.ty + d_y,
                scale: p0.scale + d_scale,
                rotate: p0.rotate + d_rot,
                aspect: p0.aspect + d_aspect,
                shear: p0.shear + d_shear,
            },
        }
    }

    /// Clamp every field of `p` into its interval (`max`/`min` rather than
    /// `clamp`, which panics on an inverted or `NaN` interval).
    fn clamp(&self, p: &mut AffineParams) {
        let (lo, hi) = (self.lower.to_vec(), self.upper.to_vec());
        let mut v = p.to_vec();
        for i in 0..6 {
            v[i] = v[i].max(lo[i]).min(hi[i]);
        }
        *p = AffineParams::from_vec(&v);
    }
}

// ── the normal equations ──────────────────────────────────────────────────────

/// Raw moments of the residual `e` and the Jacobian columns `j` over the pixels
/// of one row (or, summed, of the level).
#[derive(Clone, Copy)]
struct Moments6 {
    n: f64,
    se: f64,
    sj: [f64; 6],
    sje: [f64; 6],
    sjj: [[f64; 6]; 6],
}

impl Moments6 {
    const ZERO: Self = Self {
        n: 0.0,
        se: 0.0,
        sj: [0.0; 6],
        sje: [0.0; 6],
        sjj: [[0.0; 6]; 6],
    };

    fn add(&mut self, other: &Self) {
        self.n += other.n;
        self.se += other.se;
        for i in 0..6 {
            self.sj[i] += other.sj[i];
            self.sje[i] += other.sje[i];
            for k in i..6 {
                self.sjj[i][k] += other.sjj[i][k];
            }
        }
    }
}

/// The inverse of the current matrix and the derivative matrices of the active
/// DOFs, all as the six free entries `[a00, a01, a02, a10, a11, a12]`.
struct Linearisation {
    /// `M⁻¹` as `[s00, s01, s02, s10, s11, s12]`.
    inverse: [f64; 6],
    /// `∂M/∂p_k` for every active DOF.
    derivs: Vec<[f64; 6]>,
}

impl Linearisation {
    fn new(p: &AffineParams, active: &[LkDof], width: usize, height: usize) -> Option<Self> {
        let m = p.entries(width, height);
        let det = m[0] * m[4] - m[1] * m[3];
        if !det.is_finite() || det.abs() < 1.0e-12 {
            return None;
        }
        let s00 = m[4] / det;
        let s01 = -m[1] / det;
        let s10 = -m[3] / det;
        let s11 = m[0] / det;
        let s02 = -(s00 * m[2] + s01 * m[5]);
        let s12 = -(s10 * m[2] + s11 * m[5]);
        let derivs = active
            .iter()
            .map(|&dof| {
                let mut hi = *p;
                dof.add(&mut hi, FD_STEP);
                let mut lo = *p;
                dof.add(&mut lo, -FD_STEP);
                let (a, b) = (hi.entries(width, height), lo.entries(width, height));
                std::array::from_fn(|i| (a[i] - b[i]) / (2.0 * FD_STEP))
            })
            .collect();
        let inverse = [s00, s01, s02, s10, s11, s12];
        inverse
            .iter()
            .all(|v| v.is_finite())
            .then_some(Self { inverse, derivs })
    }
}

/// Moments of one row `y` of the level: the residual `warped - source`, the
/// Jacobian columns and their products over the interior pixels whose warped
/// value and four neighbours all exist.
fn row_moments(
    y: usize,
    width: usize,
    height: usize,
    warped: &[f32],
    source: &[f32],
    lin: &Linearisation,
) -> Moments6 {
    let mut acc = Moments6::ZERO;
    if y == 0 || y + 1 >= height || width < 3 {
        return acc;
    }
    let s = &lin.inverse;
    let yf = y as f64;
    let row = y * width;
    for x in 1..width - 1 {
        let c = warped[row + x];
        let l = warped[row + x - 1];
        let r = warped[row + x + 1];
        let up = warped[row - width + x];
        let dn = warped[row + width + x];
        if c.is_nan() || l.is_nan() || r.is_nan() || up.is_nan() || dn.is_nan() {
            continue;
        }
        let e = f64::from(c) - f64::from(source[row + x]);
        if !e.is_finite() {
            continue;
        }
        let gx = 0.5 * f64::from(r - l);
        let gy = 0.5 * f64::from(dn - up);
        let xf = x as f64;
        let ux = s[0] * xf + s[1] * yf + s[2];
        let uy = s[3] * xf + s[4] * yf + s[5];
        let mut j = [0.0_f64; 6];
        for (jk, d) in j.iter_mut().zip(&lin.derivs) {
            let vx = d[0] * ux + d[1] * uy + d[2];
            let vy = d[3] * ux + d[4] * uy + d[5];
            *jk = -(gx * vx + gy * vy);
        }
        let n_act = lin.derivs.len();
        acc.n += 1.0;
        acc.se += e;
        for i in 0..n_act {
            acc.sj[i] += j[i];
            acc.sje[i] += j[i] * e;
            for k in i..n_act {
                acc.sjj[i][k] += j[i] * j[k];
            }
        }
    }
    acc
}

/// Mean-centred Gauss-Newton system `(H, b)` for the active DOFs, from the
/// summed moments: `H = Σ J Jᵀ - (Σ J)(Σ J)ᵀ / N`, `b = -(Σ J e - Σ J Σ e / N)`.
fn normal_equations(m: &Moments6, n_act: usize) -> Option<([[f64; 6]; 6], [f64; 6])> {
    if m.n < MIN_PIXELS {
        return None;
    }
    let mut h = [[0.0_f64; 6]; 6];
    let mut b = [0.0_f64; 6];
    for i in 0..n_act {
        b[i] = -(m.sje[i] - m.sj[i] * m.se / m.n);
        for k in i..n_act {
            let v = m.sjj[i][k] - m.sj[i] * m.sj[k] / m.n;
            h[i][k] = v;
            h[k][i] = v;
        }
    }
    let finite = h.iter().flatten().chain(b.iter()).all(|v| v.is_finite());
    finite.then_some((h, b))
}

/// Solve the `n x n` system `a x = b` by Gaussian elimination with partial
/// pivoting; `None` for a singular or non-finite system.
fn solve(a: &[[f64; 6]; 6], b: &[f64; 6], n: usize) -> Option<[f64; 6]> {
    let mut m = *a;
    let mut x = *b;
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        let pivot = m[piv][col];
        if !pivot.is_finite() || pivot.abs() < 1.0e-300 {
            return None;
        }
        m.swap(col, piv);
        x.swap(col, piv);
        for r in col + 1..n {
            let f = m[r][col] / m[col][col];
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
            x[r] -= f * x[col];
        }
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for c in i + 1..n {
            s -= m[i][c] * x[c];
        }
        x[i] = s / m[i][i];
    }
    x[..n].iter().all(|v| v.is_finite()).then_some(x)
}

/// Solve the Levenberg-damped system `(H + λ diag H) δ = b`.
fn solve_damped(h: &[[f64; 6]; 6], b: &[f64; 6], n: usize, lambda: f64) -> Option<[f64; 6]> {
    let mut d = *h;
    for i in 0..n {
        d[i][i] += lambda * d[i][i].abs().max(1.0e-12);
    }
    solve(&d, b, n)
}

// ── one level ─────────────────────────────────────────────────────────────────

/// Gauss-Newton at one pyramid level from `start`; never worse than `start`
/// (a step is only accepted when the objective does not grow).
#[expect(
    clippy::too_many_lines,
    reason = "one Gauss-Newton loop; splitting would thread a dozen locals through helpers"
)]
fn refine_level(
    start: &AffineParams,
    active: &[LkDof],
    bounds: &DofBounds,
    level: LevelView<'_>,
    source: &[f32],
    max_iter: usize,
) -> AffineParams {
    let (width, height) = (level.width, level.height);
    let n = width * height;
    if active.is_empty() || n == 0 || start.to_matrix(width, height).is_none() {
        return *start;
    }
    let objective = LevelObjective::new(
        level.luma,
        source,
        width,
        height,
        ObjectiveSampler::Bicubic,
        None,
    );
    let ref_sampler = Warper {
        ref_luma: level.luma,
        width,
        height,
        sampler: ObjectiveSampler::Bicubic,
    };
    let serial = n < SERIAL_LEVEL_PIXELS;

    let mut current = *start;
    let mut current_rms = objective.eval_affine(&current);
    let mut lambda = LM_LAMBDA_INIT;
    let mut warped = vec![0.0_f32; n];

    for _ in 0..max_iter {
        let Some(fwd) = current.to_matrix(width, height) else {
            break;
        };
        let Some(inv) = fwd.try_inverse() else {
            break;
        };
        if !inv.iter().all(|v| v.is_finite()) {
            break;
        }
        let Some(lin) = Linearisation::new(&current, active, width, height) else {
            break;
        };

        // The warped reference, pixels without a sample as NaN.
        if serial {
            warped
                .chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| ref_sampler.row(&inv, y, row));
        } else {
            warped
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| ref_sampler.row(&inv, y, row));
        }

        // Row partials, added serially in row order.
        let rows: Vec<Moments6> = if serial {
            (0..height)
                .map(|y| row_moments(y, width, height, &warped, source, &lin))
                .collect()
        } else {
            (0..height)
                .into_par_iter()
                .map(|y| row_moments(y, width, height, &warped, source, &lin))
                .collect()
        };
        let mut total = Moments6::ZERO;
        for r in &rows {
            total.add(r);
        }
        let Some((h, b)) = normal_equations(&total, active.len()) else {
            break;
        };

        let mut accepted = false;
        for _ in 0..LM_MAX_REJECTIONS {
            let Some(delta) = solve_damped(&h, &b, active.len(), lambda) else {
                lambda = (lambda * LM_LAMBDA_UP).min(LM_LAMBDA_MAX);
                continue;
            };
            let step_norm = delta[..active.len()]
                .iter()
                .map(|d| d * d)
                .sum::<f64>()
                .sqrt();
            let mut candidate = current;
            for (&dof, &d) in active.iter().zip(&delta) {
                dof.add(&mut candidate, d);
            }
            bounds.clamp(&mut candidate);
            if !candidate.is_finite() {
                lambda = (lambda * LM_LAMBDA_UP).min(LM_LAMBDA_MAX);
                continue;
            }
            let candidate_rms = objective.eval_affine(&candidate);
            if candidate_rms.is_finite() && candidate_rms <= current_rms {
                let improvement = current_rms - candidate_rms;
                let converged = step_norm < LK_STEP_EPS
                    || improvement <= LK_REL_IMPROVEMENT * current_rms.abs();
                current = candidate;
                current_rms = candidate_rms;
                lambda = (lambda * LM_LAMBDA_DOWN).max(LM_LAMBDA_MIN);
                if converged {
                    return current;
                }
                accepted = true;
                break;
            }
            lambda = (lambda * LM_LAMBDA_UP).min(LM_LAMBDA_MAX);
        }
        if !accepted {
            // No improving step even with heavy damping: converged or stuck.
            break;
        }
    }
    current
}

/// The levels the schedule processes, coarse to fine, with their iteration
/// budgets.  See the module documentation.
fn schedule(pyramid: &LumaPyramid, full: &[f32]) -> Vec<(usize, usize)> {
    let n_levels = pyramid.n_levels();
    let mut out = Vec::with_capacity(n_levels);
    for idx in 0..n_levels {
        let view = pyramid.level(idx, full);
        let large = view.width.min(view.height) > LK_LARGE_SHORT_SIDE;
        let finest_two = idx + 2 >= n_levels;
        if large && finest_two {
            break;
        }
        out.push((
            idx,
            if large {
                LK_LARGE_LEVEL_ITERS
            } else {
                LK_MAX_ITER_PER_LEVEL
            },
        ));
    }
    if out.is_empty() {
        out.push((n_levels - 1, LK_MAX_ITER_PER_LEVEL));
    }
    out
}

// ── public API ────────────────────────────────────────────────────────────────

/// Refine a coarse matrix with pyramid Lucas-Kanade / Gauss-Newton.
///
/// Builds both luma pyramids, then runs [`refine_alignment_lk_with_pyramid`].
/// The matrix maps reference coordinates to source coordinates, like every other
/// refiner of the crate; the degrees of freedom and their bounds come from
/// `opts` (`max_iterations` and `device` are ignored, see the module
/// documentation).
///
/// # Errors
/// - [`StackerError::MathError`] when `initial` is not finite.
/// - [`StackerError::AlignmentFailed`] when the image sizes differ, or the
///   result is not finite or not a valid matrix.
pub fn refine_alignment_lk(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<LkResult, StackerError> {
    refine_alignment_lk_with_pyramid(
        reference,
        &LumaPyramid::new(reference),
        source,
        initial,
        opts,
    )
}

/// [`refine_alignment_lk`] with the reference's luma pyramid supplied by the
/// caller (build it once per run, as for the simplex refiners).
///
/// With no enabled degree of freedom the seed is returned unchanged, and both
/// RMS values of the result are the objective of the seed.
///
/// # Errors
/// As [`refine_alignment_lk`], plus [`StackerError::AlignmentFailed`] when
/// `ref_pyramid` was built from an image of a different size.
pub fn refine_alignment_lk_with_pyramid(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<LkResult, StackerError> {
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(StackerError::MathError(
            "refine_alignment_lk: initial matrix contains non-finite values".into(),
        ));
    }
    if reference.width != source.width || reference.height != source.height {
        return Err(StackerError::AlignmentFailed(
            "refine_alignment_lk: reference and source dimensions do not match".into(),
        ));
    }
    check_pyramid("refine_alignment_lk", reference, ref_pyramid)?;

    let (w, h) = (source.width, source.height);
    let p0 = seed_params(initial, w, h, opts);
    let active = active_dofs(opts);
    let bounds = DofBounds::new(&p0, opts);

    let src_pyramid = LumaPyramid::new(source);
    let levels = schedule(ref_pyramid, &reference.luma);
    // The level both RMS values are measured at: the finest one processed.
    let (last_idx, _) = levels[levels.len() - 1];
    let rms_at_last = |p: &AffineParams| -> f64 {
        let r = ref_pyramid.level(last_idx, &reference.luma);
        let s = src_pyramid.level(last_idx, &source.luma);
        LevelObjective::new(
            r.luma,
            s.luma,
            r.width,
            r.height,
            ObjectiveSampler::Bicubic,
            None,
        )
        .eval_affine(p)
    };
    let starting_rms = rms_at_last(&p0);

    if active.is_empty() {
        return Ok(LkResult {
            matrix: *initial,
            starting_rms,
            final_rms: starting_rms,
        });
    }

    let mut current = p0;
    for &(idx, max_iter) in &levels {
        let r = ref_pyramid.level(idx, &reference.luma);
        let s = src_pyramid.level(idx, &source.luma);
        let refined = refine_level(&current, &active, &bounds, r, s.luma, max_iter);
        // The level's guard returns the last good parameters; anything else
        // that is not finite is dropped here.
        if refined.is_finite() {
            current = refined;
        }
    }

    if !current.is_finite() {
        return Err(StackerError::AlignmentFailed(
            "refine_alignment_lk: optimizer produced non-finite parameters".into(),
        ));
    }
    let matrix = current.to_matrix(w, h).ok_or_else(|| {
        StackerError::AlignmentFailed(
            "refine_alignment_lk: refined parameters produced a degenerate matrix".into(),
        )
    })?;
    let final_rms = rms_at_last(&current);
    Ok(LkResult {
        matrix,
        starting_rms,
        final_rms,
    })
}

/// The objective (mean-corrected RMS of `source - warped reference`) of `matrix`.
///
/// Evaluated at full resolution, with the Catmull-Rom sampler and the coverage
/// rule of the refiners; [`PENALTY`]-sized (`1e6`) for a matrix that is not
/// invertible or samples less than half of the frame.
#[must_use]
pub fn lk_rms(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> f64 {
    if reference.width != source.width || reference.height != source.height {
        return PENALTY;
    }
    LevelObjective::new(
        &reference.luma,
        &source.luma,
        reference.width,
        reference.height,
        ObjectiveSampler::Bicubic,
        None,
    )
    .eval_forward(matrix)
}
