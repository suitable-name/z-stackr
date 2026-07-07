//! Pyramid Lucas-Kanade / Gauss-Newton intensity refinement — an
//! analytic-gradient alternative to the blind-sampling Nelder-Mead simplex
//! in [`super::core::refine_alignment_registration`].
//!
//! ## Formulation
//!
//! This module implements a **forward-additive** Lucas-Kanade / Gauss-Newton
//! optimiser over the same 6-DOF [`RegistrationParams`] space `(tx, ty,
//! scale, rotate, aspect, shear)` used throughout `stacker_align::refine`,
//! parameterised so its per-DOF Jacobian columns are evaluated **at the
//! identity increment around the current estimate** — i.e. the warp
//! Jacobian's closed form is evaluated once per iteration at the DOF
//! generators' identity point (see [`warp_jacobian_column`]), the same
//! constant-coefficient generators an inverse-compositional formulation
//! would use, while the *image* gradient is resampled from the actual
//! warped moving image every iteration (rather than reusing a fixed
//! template gradient) so the linearisation stays valid as the estimate
//! moves away from the seed. This is the standard "forward-additive with
//! constant warp generators" middle ground: cheaper than full
//! forward-additive re-differentiation of the warp at the current
//! parameter values (the generators are the same closed-form constants at
//! every iteration — see [`warp_jacobian_column`]'s derivation), while
//! staying unambiguous about the composition rule (a plain vector add in
//! parameter space, `p ← p + Δp`, no matrix composition/inversion
//! required).
//!
//! Reusing this crate's existing warp convention (see
//! [`super::core::objective_registration`]): the *fixed* target is the
//! **source** luma plane, and the *moving* image is the **reference** luma
//! plane, sampled through the forward transform `W(x; p) =
//! params_to_matrix(p) * x`. We minimise `sum_x [ I(W(x; p)) - T(x) ]^2`
//! (`I` = reference, `T` = source) — exactly
//! [`super::core::objective_registration`]'s objective, just solved with
//! analytic gradients instead of a derivative-free simplex.
//!
//! Each iteration:
//! 1. Warp the reference by the current estimate's matrix, producing the
//!    error image `I(W(x;p)) - T(x)` (identical warp to
//!    [`super::core::objective_registration`]).
//! 2. Sample the **warped image's** gradient (central difference of the
//!    same warped-reference plane) at every pixel.
//! 3. Steepest-descent image per active DOF: `∇I(W(x;p)) · ∂W/∂Δp|_{Δp=0}`
//!    (the constant-coefficient generators from [`warp_jacobian_column`]).
//! 4. Accumulate the Gauss-Newton normal equations and solve for `Δp`
//!    (Levenberg-damped — see below).
//! 5. `p ← clamp(p + Δp)`, accept only if the resulting RMS does not
//!    regress (see the trust-region retry loop in [`refine_level`]).
//!
//! ## DOF gating
//!
//! Only the degrees of freedom the caller's [`BoundedRefineOptions`] enables
//! (`allow_shift_x/y, allow_scale, allow_rotation, allow_aspect,
//! allow_shear`) get a Jacobian column and therefore an entry in the 6×6 (or
//! smaller) Gauss-Newton normal-equations solve; the fixed-lattice mapping
//! from "active DOF index" to `RegistrationParams` field is `[ShiftX,
//! ShiftY, Scale, Rotate, Aspect, Shear]` restricted to the enabled subset,
//! mirroring the ordering `align_frame`'s DOF gating already produces.
//! Disabled DOFs are simply never included in the linear solve, so they
//! never move away from the current per-level estimate's value for that
//! field.
//!
//! ## Damping
//!
//! Pure Gauss-Newton (`H·Δp = b` with no damping) can overshoot or diverge
//! on a poorly-conditioned Hessian (e.g. very few active DOFs on a
//! low-texture patch). This implementation uses a **Levenberg-style damped
//! normal equation**: `(H + λ·diag(H))·Δp = b`, with `λ` adapted per
//! iteration (classic Levenberg-Marquardt trust-region schedule: shrink `λ`
//! by 3× after an RMS-improving step, grow it by 3× and reject the step
//! after a non-improving one). This was chosen over pure step-halving
//! because it directly conditions the normal-equations solve itself (guards
//! against a near-singular Hessian on flat/low-texture regions, not just
//! against an overly aggressive step length), and it degrades gracefully to
//! undamped Gauss-Newton (fast quadratic convergence) as `λ → 0` once the
//! estimate is already close to the optimum — which is the common case once
//! the pyramid has warm-started from a coarser level.
//!
//! ## Convergence / iteration budget
//!
//! Each pyramid level stops when the parameter-update step norm (in the
//! same units [`SearchBounds`]-style clamping uses: fractional pixels for
//! `tx`/`ty`, dimensionless ratios for `scale`/`aspect`/`shear`, radians for
//! `rotate`) drops below [`LK_STEP_EPS`], or after [`LK_MAX_ITER_PER_LEVEL`]
//! iterations — far fewer than Nelder-Mead's per-level budgets (50-200)
//! because each LK iteration uses the actual gradient direction instead of
//! blindly sampling the objective at simplex vertices.
//!
//! ## Coarse-biased level schedule (shared with `core.rs`)
//!
//! This module now runs the **same coarse-biased schedule**
//! [`super::core::refine_alignment_registration`] (Nelder-Mead) uses,
//! reusing its `pub(super)` constants ([`super::core::REF_SHORT_SIDE_THRESHOLD`],
//! [`super::core::REF_LARGE_LEVEL_ITERS`]) rather than duplicating the
//! numbers:
//!
//! - The full-resolution (finest) pyramid level is **never** refined.
//! - Once a "large" level (short side > [`super::core::REF_SHORT_SIDE_THRESHOLD`]
//!   pixels) is reached among the finest two levels, the level loop stops
//!   entirely rather than processing it.
//! - Any other large level gets a small iteration budget
//!   ([`super::core::REF_LARGE_LEVEL_ITERS`]) instead of the normal
//!   [`LK_MAX_ITER_PER_LEVEL`] budget.
//!
//! So the finest level this module actually *processes* is, in the common
//! (multi-megapixel) case, a coarser level than the sole full-resolution
//! level — the opposite of what an earlier version of this module did
//! (giving the full-resolution level a *larger* iteration budget than every
//! other level). This is intentional: on real (tens-of-megapixel)
//! frames the full-resolution level's warp + gradient + accumulation pass
//! dominates total runtime, and — exactly as for Nelder-Mead — is
//! deliberately under-refined to avoid chasing per-frame sensor noise at
//! full resolution. Because `tx`/`ty` are fractional and every other DOF is
//! resolution-independent, the coarser level's result is still a valid,
//! directly-usable transform at full resolution.
//!
//! For small (e.g. test-sized) images whose pyramid never grows past a
//! single coarser level plus the full-resolution level, this mirrors
//! `core.rs`'s exact behaviour: the loop always processes at least the one
//! coarser level before stopping at full resolution. The one case this
//! does *not* refine anything is a pyramid with a **single** level (full
//! resolution only, i.e. the source image's short side is already below
//! [`super::pyramid::build_luma_pyramid`]'s minimum level size) — `core.rs`
//! has the identical degenerate behaviour for images at or below its own
//! 64px pyramid floor, so this is consistency with the existing reference
//! schedule, not a new deviation.
//!
//! ## Bounds / NaN guard
//!
//! After every accepted update the full 6-DOF parameter vector is clamped to
//! the same half-width intervals `SearchBounds` would compute from
//! `BoundedRefineOptions` (`max_shift_x`/`max_shift_y`, `max_scale`,
//! `max_rotation_deg`, `max_aspect`, `max_shear`) around the seed `p0` —
//! reimplemented directly here
//! (simple per-DOF min/max) rather than depending on `core.rs`'s private
//! `SearchBounds`/`Dof`, which stay private to keep this module's coupling
//! to the simplex internals minimal. Any non-finite parameter, matrix, or
//! Hessian-solve failure aborts the *current pyramid level* and returns the
//! last known-good (pre-failure) parameter vector — this function never
//! panics and never propagates NaN upward, matching every other function in
//! `stacker_align::refine`.

use nalgebra::{Matrix3, SMatrix, SVector};
use rayon::prelude::*;
use stacker_core::{error::StackerError, image::PlanarImage};

use super::{
    BoundedRefineOptions,
    core::{REF_LARGE_LEVEL_ITERS, REF_SHORT_SIDE_THRESHOLD, objective_registration},
    params::{RegistrationParams, matrix_to_params, params_to_matrix},
    pyramid::build_luma_pyramid,
};

/// Parameter-update step-norm convergence threshold. Units are mixed
/// (fractional pixels for tx/ty, radians for rotate, dimensionless ratios
/// for scale/aspect/shear) but all are O(1) scale quantities in this
/// parameterisation, so a single small threshold is a reasonable
/// stopping rule across all of them (same spirit as `NM_TOL` being a
/// single scalar tolerance across all 6 DOFs in the simplex).
const LK_STEP_EPS: f64 = 1.0e-8;

/// Iteration budget for a normal-sized (non-"large") pyramid level. IC-LK
/// converges in far fewer iterations than Nelder-Mead per level because it
/// follows the analytic gradient instead of sampling; 20 is generous
/// headroom. "Large" levels instead use
/// [`super::core::REF_LARGE_LEVEL_ITERS`] — see the module doc's "coarse
/// biased level schedule" section.
const LK_MAX_ITER_PER_LEVEL: usize = 20;

/// Initial Levenberg damping factor.
const LM_LAMBDA_INIT: f64 = 1.0e-3;
/// Multiplicative factor lambda is scaled by after a rejected (non-improving) step.
const LM_LAMBDA_UP: f64 = 3.0;
/// Multiplicative factor lambda is scaled by after an accepted (improving) step.
const LM_LAMBDA_DOWN: f64 = 1.0 / 3.0;
/// Hard floor/ceiling on the damping factor to avoid it drifting to 0 or infinity.
const LM_LAMBDA_MIN: f64 = 1.0e-12;
const LM_LAMBDA_MAX: f64 = 1.0e6;

/// Maximum inner Levenberg trust-region retries (re-solving with a larger
/// lambda after a rejected step) before giving up on the current outer
/// iteration and treating it as converged.
const LM_MAX_REJECTIONS: usize = 8;

/// Result of a successful [`refine_alignment_lk`] call.
#[derive(Debug, Clone, Copy)]
pub struct LkResult {
    /// The refined forward (source → reference) transform.
    pub matrix: Matrix3<f32>,
    /// The RMS objective ([`objective_registration`]) evaluated at the
    /// **finest pyramid level this call actually processed** (never full
    /// resolution — see the module doc's "coarse-biased level schedule"
    /// section), computed once at that level's resolution *before* the
    /// first iteration touches it.
    ///
    /// This is a small-level warp+RMS, not a full-resolution one — it
    /// exists so `pipeline::align_frame`'s Auto-mode fallback decision can
    /// compare LK's starting vs. final RMS without a separate full-res
    /// [`lk_rms`] call per frame (see that function's doc comment for the
    /// two-full-res-calls cost this replaced).
    pub starting_rms: f64,
    /// The RMS objective ([`objective_registration`]), at the same level
    /// [`Self::starting_rms`] was computed at, after all processed pyramid
    /// levels have run.
    pub final_rms: f64,
}

/// The six [`RegistrationParams`] fields, in the fixed DOF-column order used
/// throughout this module: shift X, shift Y, scale, rotate, aspect, shear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dof {
    ShiftX,
    ShiftY,
    Scale,
    Rotate,
    Aspect,
    Shear,
}

/// Which DOFs are active, in a fixed canonical order, derived from
/// [`BoundedRefineOptions`]'s gating booleans.
fn active_dofs(o: &BoundedRefineOptions) -> Vec<Dof> {
    let mut active = Vec::with_capacity(6);
    if o.allow_shift_x {
        active.push(Dof::ShiftX);
    }
    if o.allow_shift_y {
        active.push(Dof::ShiftY);
    }
    if o.allow_scale {
        active.push(Dof::Scale);
    }
    if o.allow_rotation {
        active.push(Dof::Rotate);
    }
    if o.allow_aspect {
        active.push(Dof::Aspect);
    }
    if o.allow_shear {
        active.push(Dof::Shear);
    }
    active
}

/// Per-DOF clamp half-widths around the seed `p0`, in the same units
/// `SearchBounds` (private to `core.rs`) computes them in. Reimplemented
/// directly here (simple min/max) rather than depending on that private
/// type, as documented in the module doc comment.
struct DofBounds {
    lower: RegistrationParams,
    upper: RegistrationParams,
}

impl DofBounds {
    fn new(p0: &RegistrationParams, o: &BoundedRefineOptions) -> Self {
        let d_rot = o.max_rotation_deg * std::f64::consts::PI / 180.0;
        let d_scale = o.max_scale / 100.0;
        let d_shift_x = o.max_shift_x / 100.0;
        let d_shift_y = o.max_shift_y / 100.0;
        let d_aspect = o.max_aspect / 100.0;
        let d_shear = o.max_shear / 100.0;

        Self {
            lower: RegistrationParams {
                tx: p0.tx - d_shift_x,
                ty: p0.ty - d_shift_y,
                scale: p0.scale - d_scale,
                rotate: p0.rotate - d_rot,
                aspect: p0.aspect - d_aspect,
                shear: p0.shear - d_shear,
            },
            upper: RegistrationParams {
                tx: p0.tx + d_shift_x,
                ty: p0.ty + d_shift_y,
                scale: p0.scale + d_scale,
                rotate: p0.rotate + d_rot,
                aspect: p0.aspect + d_aspect,
                shear: p0.shear + d_shear,
            },
        }
    }

    /// Clamp `p` in place to `[lower, upper]` per-field.
    const fn clamp(&self, p: &mut RegistrationParams) {
        p.tx = p.tx.clamp(self.lower.tx, self.upper.tx);
        p.ty = p.ty.clamp(self.lower.ty, self.upper.ty);
        p.scale = p.scale.clamp(self.lower.scale, self.upper.scale);
        p.rotate = p.rotate.clamp(self.lower.rotate, self.upper.rotate);
        p.aspect = p.aspect.clamp(self.lower.aspect, self.upper.aspect);
        p.shear = p.shear.clamp(self.lower.shear, self.upper.shear);
    }
}

/// Central-difference luma gradient (∂I/∂x, ∂I/∂y), edge-clamped at the
/// borders. Recomputed every iteration on the current warped-reference
/// plane (see the module doc's "forward-additive with constant warp
/// generators" note on why this — rather than a one-shot template gradient
/// — is what keeps the linearisation valid as the estimate moves).
fn central_diff_gradient(luma: &[f32], width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0_f32; width * height];
    let mut gy = vec![0.0_f32; width * height];

    gx.par_chunks_mut(width)
        .zip(gy.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (gx_row, gy_row))| {
            let row = &luma[y * width..(y + 1) * width];
            let row_up = if y > 0 {
                &luma[(y - 1) * width..y * width]
            } else {
                row
            };
            let row_down = if y + 1 < height {
                &luma[(y + 1) * width..(y + 2) * width]
            } else {
                row
            };
            for x in 0..width {
                let left = if x > 0 { row[x - 1] } else { row[x] };
                let right = if x + 1 < width { row[x + 1] } else { row[x] };
                gx_row[x] = (right - left) * 0.5;
                gy_row[x] = (row_down[x] - row_up[x]) * 0.5;
            }
        });

    (gx, gy)
}

/// Warp Jacobian `∂(x_ref, y_ref)/∂dof` evaluated at the identity increment
/// (`Δp = 0`), at a pixel whose image-centre-relative coordinates are `(dx,
/// dy)`, for a level of size `width × height` (used only for the
/// translation columns, which scale by the level's absolute dimensions
/// since `tx`/`ty` are fractional).
///
/// Derived by differentiating `params_to_matrix`'s closed form
/// `(a00, a01, a02, a10, a11, a12)` with respect to each DOF at the
/// identity point (`scale=1, rotate=0, aspect=1, shear=0, tx=ty=0`); see the
/// module doc comment for the full derivation summary.
const fn warp_jacobian_column(dof: Dof, dx: f64, dy: f64, w: f64, h: f64) -> (f64, f64) {
    match dof {
        Dof::ShiftX => (w, 0.0),
        Dof::ShiftY => (0.0, h),
        Dof::Scale => (dx, dy),
        Dof::Rotate => (-dy, dx),
        Dof::Aspect => (0.0, dy),
        Dof::Shear => (dy, 0.0),
    }
}

/// Fixed-size (up to 6×6) Gauss-Newton normal equations accumulator +
/// Levenberg-damped solve for the currently active DOF subset.
type Hessian = SMatrix<f64, 6, 6>;
type Grad = SVector<f64, 6>;

/// Accumulate the Gauss-Newton normal equations `H = J^T J`, `b = -J^T
/// error` over every pixel, in parallel over row chunks.
///
/// `H`/`b` are independent of the Levenberg damping factor `lambda`, so this
/// is computed **once per outer iteration**; only the cheap ≤6×6 damped
/// solve ([`solve_damped`]) needs to be redone on each trust-region
/// rejection retry (see [`refine_level`]'s inner retry loop) — previously
/// this whole per-pixel accumulation was repeated on every retry (up to
/// [`LM_MAX_REJECTIONS`] times) even though only the diagonal damping term
/// changes.
///
/// Parallelised with rayon: each row-chunk fold produces a local `(Hessian,
/// Grad)` accumulator pair, reduced (summed) at the end. Reduction order is
/// not fixed, so the result may differ from a strictly sequential
/// accumulation in the last few ULPs of `f64` — immaterial here since all
/// callers compare against tolerances, never bit-exact values.
/// Pixel-chunk size for [`accumulate_normal_equations`]'s parallel fold —
/// large enough to amortise rayon's per-chunk overhead, small enough to
/// give the scheduler plenty of chunks to balance across threads.
const PIXELS_PER_CHUNK: usize = 4096;

fn accumulate_normal_equations(
    active: &[Dof],
    sd_images: &[Vec<f32>],
    error: &[f32],
) -> (Hessian, Grad) {
    let n = active.len();
    if n == 0 || error.is_empty() {
        return (Hessian::zeros(), Grad::zeros());
    }

    // Chunk directly over the flat pixel buffer (there's no row-alignment
    // requirement for a plain sum-reduction) so each parallel task
    // accumulates a local (H, b) pair in a scalar loop; chunk results are
    // summed (not row-mirrored) in `reduce`, with the upper-triangle mirror
    // applied once at the end below.
    let (mut h, b) = error
        .par_chunks(PIXELS_PER_CHUNK)
        .enumerate()
        .map(|(chunk_idx, error_chunk)| {
            let base = chunk_idx * PIXELS_PER_CHUNK;
            let mut h_local = Hessian::zeros();
            let mut b_local = Grad::zeros();
            for (offset, &err) in error_chunk.iter().enumerate() {
                let idx = base + offset;
                let e = -f64::from(err);
                for i in 0..n {
                    let sdi = f64::from(sd_images[i][idx]);
                    b_local[i] += sdi * e;
                    for j in i..n {
                        let sdj = f64::from(sd_images[j][idx]);
                        h_local[(i, j)] += sdi * sdj;
                    }
                }
            }
            (h_local, b_local)
        })
        .reduce(
            || (Hessian::zeros(), Grad::zeros()),
            |(h0, b0), (h1, b1)| (h0 + h1, b0 + b1),
        );

    // Mirror the upper triangle (symmetric Hessian).
    for i in 0..n {
        for j in 0..i {
            h[(i, j)] = h[(j, i)];
        }
    }

    (h, b)
}

/// Levenberg-damped solve of the (already-accumulated) normal equations for
/// the currently active DOF subset: `(H + lambda*diag(H)) delta = b`.
/// Returns `Some(delta)` (indexed by `active`, same order) on a successful
/// solve, `None` if the Hessian solve fails or produces non-finite output
/// (NaN guard — caller aborts the level on `None`).
///
/// Cheap relative to [`accumulate_normal_equations`] (a fixed ≤6×6 LU
/// solve), so this is safe to re-run on every Levenberg trust-region retry
/// without re-touching the per-pixel data.
fn solve_damped(h: &Hessian, b: &Grad, n: usize, lambda: f64) -> Option<Vec<f64>> {
    if n == 0 {
        return Some(Vec::new());
    }

    let mut h_damped = *h;
    // Levenberg damping: H + lambda * diag(H).
    for i in 0..n {
        let diag = h_damped[(i, i)];
        h_damped[(i, i)] = diag + lambda * diag.abs().max(1.0e-12);
    }

    // Solve the n x n system (submatrix of the fixed 6x6 storage).
    let h_n = h_damped.fixed_view::<6, 6>(0, 0).clone_owned();
    let b_n = *b;

    // Use nalgebra's generic LU solve on the actual active dimension by
    // copying into dynamically-sized matrices (n is small, <= 6).
    let h_dyn = nalgebra::DMatrix::from_fn(n, n, |r, c| h_n[(r, c)]);
    let b_dyn = nalgebra::DVector::from_fn(n, |r, _| b_n[r]);

    let decomp = h_dyn.lu();
    let delta = decomp.solve(&b_dyn)?;

    if !delta.iter().all(|v: &f64| v.is_finite()) {
        return None;
    }

    Some(delta.iter().copied().collect())
}

/// Apply a parameter delta (indexed by `active`, in the same order) to
/// `base` via a direct additive update in `RegistrationParams` space:
/// `tx += d`, `scale += d` (etc. — see the module doc's "forward-additive"
/// note for why a plain vector add, rather than matrix composition, is the
/// correct update rule for this module's Jacobian convention). Inactive
/// DOFs get an implicit delta of exactly 0.0.
fn apply_update(base: &RegistrationParams, active: &[Dof], delta: &[f64]) -> RegistrationParams {
    let mut p = *base;
    for (i, &dof) in active.iter().enumerate() {
        let d = delta[i];
        match dof {
            Dof::ShiftX => p.tx += d,
            Dof::ShiftY => p.ty += d,
            Dof::Scale => p.scale += d,
            Dof::Rotate => p.rotate += d,
            Dof::Aspect => p.aspect += d,
            Dof::Shear => p.shear += d,
        }
    }
    p
}

/// Compute the RMS objective ([`objective_registration`]) for `p` at a
/// single pyramid level.
fn level_rms(
    p: &RegistrationParams,
    ref_luma: &[f32],
    src_luma: &[f32],
    w: usize,
    h: usize,
) -> f64 {
    objective_registration(&p.to_vec(), ref_luma, src_luma, w, h)
}

/// Warp `ref_luma` by `fwd` into the `width`×`height` dst grid, matching
/// [`super::core::objective_registration`]'s exact warp convention
/// (edge-clamped 4-tap spline sampler). Returns `None` if any sampled pixel
/// is non-finite (NaN guard).
fn warp_reference(
    fwd: &Matrix3<f32>,
    ref_luma: &[f32],
    width: usize,
    height: usize,
) -> Option<Vec<f32>> {
    let mut warped = vec![0.0_f32; width * height];
    let mut ok = true;
    warped
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let p_dst = nalgebra::Vector3::new(x as f32, y as f32, 1.0_f32);
                let p_src = fwd * p_dst;
                let denom = p_src[2];
                *out = if denom.abs() < 1.0e-8 {
                    f32::NAN
                } else {
                    let sx = p_src[0] / denom;
                    let sy = p_src[1] / denom;
                    crate::transform::spline4x4_sample_clamped(ref_luma, width, height, sx, sy)
                };
            }
        });
    if warped.iter().any(|v| !v.is_finite()) {
        ok = false;
    }
    ok.then_some(warped)
}

/// Immutable per-level inputs bundled together so [`refine_level`] stays
/// under clippy's argument-count lint (this is a plain grouping of
/// borrowed slices/values, not a type with independent invariants).
struct LevelInputs<'a> {
    bounds: &'a DofBounds,
    ref_luma: &'a [f32],
    src_luma: &'a [f32],
    width: usize,
    height: usize,
    max_iter: usize,
}

/// Run forward-additive Gauss-Newton at a single pyramid level, starting
/// from `p_start`. Returns the best parameters found (never worse than
/// `p_start` by construction: any non-improving step is rejected via the
/// Levenberg trust-region retry loop, and the level stops rather than
/// applying it).
fn refine_level(
    p_start: &RegistrationParams,
    active: &[Dof],
    inputs: &LevelInputs<'_>,
) -> RegistrationParams {
    let LevelInputs {
        bounds,
        ref_luma,
        src_luma,
        width,
        height,
        max_iter,
    } = *inputs;

    if active.is_empty() {
        return *p_start;
    }

    let cx = width as f64 * 0.5;
    let cy = height as f64 * 0.5;
    let w = width as f64;
    let h = height as f64;
    let npix = width * height;

    // Validate `p_start` decomposes to a usable matrix before iterating;
    // if not, there's nothing safe to refine from at this level.
    if params_to_matrix(p_start, width, height).is_none() {
        return *p_start;
    }

    let mut current = *p_start;
    let mut current_rms = level_rms(&current, ref_luma, src_luma, width, height);
    let mut lambda = LM_LAMBDA_INIT;

    for _iter in 0..max_iter {
        let Some(fwd) = params_to_matrix(&current, width, height) else {
            break;
        };
        let Some(warped) = warp_reference(&fwd, ref_luma, width, height) else {
            break;
        };

        // Error image: I(W(x;p)) - T(x), matching
        // `objective_registration`'s warp convention exactly.
        let error: Vec<f32> = warped
            .iter()
            .zip(src_luma.iter())
            .map(|(&i, &t)| i - t)
            .collect();

        // Gradient of the warped (moving) image, resampled fresh this
        // iteration — see the module doc's rationale for why this (rather
        // than a one-shot template gradient) is what this module does.
        let (gx, gy) = central_diff_gradient(&warped, width, height);

        // Steepest-descent images: one per active DOF.
        let sd_images: Vec<Vec<f32>> = active
            .iter()
            .map(|&dof| {
                let mut sd = vec![0.0_f32; npix];
                sd.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                    let dy = y as f64 - cy;
                    for (x, out) in row.iter_mut().enumerate() {
                        let dx = x as f64 - cx;
                        let (jx, jy) = warp_jacobian_column(dof, dx, dy, w, h);
                        let idx = y * width + x;
                        let gxv = f64::from(gx[idx]);
                        let gyv = f64::from(gy[idx]);
                        *out = (gxv * jx + gyv * jy) as f32;
                    }
                });
                sd
            })
            .collect();

        // Accumulate H/b ONCE per outer iteration — they are independent of
        // the Levenberg damping factor `lambda`, so only the cheap damped
        // solve below needs to be redone on each trust-region rejection
        // retry (see `accumulate_normal_equations`'s doc comment).
        let (h, b) = accumulate_normal_equations(active, &sd_images, &error);

        // Try to solve, growing lambda on rejected steps (trust-region style).
        let mut accepted = false;
        for _rejection in 0..LM_MAX_REJECTIONS {
            let Some(delta) = solve_damped(&h, &b, active.len(), lambda) else {
                lambda = (lambda * LM_LAMBDA_UP).min(LM_LAMBDA_MAX);
                continue;
            };

            let step_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();

            let mut candidate = apply_update(&current, active, &delta);
            bounds.clamp(&mut candidate);
            if !candidate.is_finite() || params_to_matrix(&candidate, width, height).is_none() {
                lambda = (lambda * LM_LAMBDA_UP).min(LM_LAMBDA_MAX);
                continue;
            }

            let candidate_rms = level_rms(&candidate, ref_luma, src_luma, width, height);
            if candidate_rms.is_finite() && candidate_rms <= current_rms {
                current = candidate;
                current_rms = candidate_rms;
                lambda = (lambda * LM_LAMBDA_DOWN).max(LM_LAMBDA_MIN);
                accepted = true;
                if step_norm < LK_STEP_EPS {
                    return current;
                }
                break;
            }
            lambda = (lambda * LM_LAMBDA_UP).min(LM_LAMBDA_MAX);
        }

        if !accepted {
            // Could not find any improving step even after growing lambda
            // repeatedly — converged (or stuck); stop this level.
            break;
        }
    }

    current
}

/// Refine a coarse alignment matrix using pyramid forward-additive
/// Lucas-Kanade / Gauss-Newton optimisation.
///
/// Builds Gaussian luma pyramids for both `reference` and `source` (via
/// [`build_luma_pyramid`], shared with the rest of `stacker_align::refine`)
/// and runs from the coarsest level to the finest, warm-starting each level
/// from the previous one's result (valid because `tx`/`ty` are fractional
/// and every other DOF is resolution-independent — the same reasoning
/// `refine_alignment_registration` relies on).
///
/// DOF gating and bounds come from `opts`, exactly like
/// [`super::core::refine_alignment_registration`]'s `BoundedRefineOptions`.
///
/// # Errors
///
/// - [`StackerError::MathError`] — `initial` contains non-finite values.
/// - [`StackerError::AlignmentFailed`] — dimension mismatch, or the final
///   refined parameters are non-finite / produce a degenerate matrix (the
///   per-level NaN guard means this should be rare — it only surfaces if
///   even the seed itself cannot be decomposed/rebuilt).
pub fn refine_alignment_lk(
    reference: &PlanarImage<f32>,
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

    let w = source.width;
    let h = source.height;
    let p0 = matrix_to_params(initial, w, h);
    let active = active_dofs(opts);
    let bounds = DofBounds::new(&p0, opts);

    if active.is_empty() {
        // Pass-through: no active DOF, nothing was evaluated at any level.
        // `starting_rms`/`final_rms` both report the (identical) full-res
        // RMS of the unchanged matrix so callers relying on them (e.g. the
        // Auto-mode regression check) see a trivially-non-regressing result
        // rather than a fabricated/inconsistent value.
        let rms = lk_rms(reference, source, initial);
        return Ok(LkResult {
            matrix: *initial,
            starting_rms: rms,
            final_rms: rms,
        });
    }

    let ref_pyramid = build_luma_pyramid(reference);
    let src_pyramid = build_luma_pyramid(source);
    debug_assert_eq!(ref_pyramid.len(), src_pyramid.len());
    let n_levels = ref_pyramid.len();

    let mut current = p0;
    let mut starting_rms: Option<f64> = None;
    let mut final_rms = f64::MAX;

    for (lvl_idx, (ref_lvl, src_lvl)) in ref_pyramid.iter().zip(src_pyramid.iter()).enumerate() {
        // ── Coarse-biased stop / budget rules — shared with `core.rs`'s ──
        // Nelder-Mead schedule (see the module doc's "coarse-biased level
        // schedule" section). `lvl_idx` counts from the coarsest (0) to the
        // finest (`n_levels - 1`), matching `core.rs`'s convention exactly.
        let short_side = ref_lvl.width.min(ref_lvl.height);
        let is_large = short_side > REF_SHORT_SIDE_THRESHOLD;
        let is_full_res = lvl_idx == n_levels - 1;
        let is_finest_two = lvl_idx + 2 >= n_levels;

        // Never refine full resolution, and stop entirely once a
        // finest-two level is still large.
        if is_full_res || (is_large && is_finest_two) {
            break;
        }

        let max_iter = if is_large {
            REF_LARGE_LEVEL_ITERS
        } else {
            LK_MAX_ITER_PER_LEVEL
        };

        let inputs = LevelInputs {
            bounds: &bounds,
            ref_luma: &ref_lvl.luma,
            src_luma: &src_lvl.luma,
            width: ref_lvl.width,
            height: ref_lvl.height,
            max_iter,
        };

        // Record the starting RMS the first time we actually process a
        // level (i.e. at the coarsest processed level) and the final RMS
        // after every processed level — so by the time the loop exits,
        // `final_rms` holds the RMS at the finest level this call actually
        // touched (see `LkResult::starting_rms`/`final_rms`'s doc comments).
        // Both are small-level warps (never full-resolution), since this
        // loop never reaches the full-res level.
        if starting_rms.is_none() {
            starting_rms = Some(level_rms(
                &current,
                &ref_lvl.luma,
                &src_lvl.luma,
                ref_lvl.width,
                ref_lvl.height,
            ));
        }

        let refined = refine_level(&current, &active, &inputs);

        if refined.is_finite() {
            current = refined;
        }
        // else: keep `current` from the previous (coarser) level — the
        // NaN guard contract ("abort the current level, return last-good").

        final_rms = level_rms(
            &current,
            &ref_lvl.luma,
            &src_lvl.luma,
            ref_lvl.width,
            ref_lvl.height,
        );
    }

    if !current.is_finite() {
        return Err(StackerError::AlignmentFailed(
            "refine_alignment_lk: optimizer produced non-finite parameters".into(),
        ));
    }

    let matrix = params_to_matrix(&current, w, h).ok_or_else(|| {
        StackerError::AlignmentFailed(
            "refine_alignment_lk: refined parameters produced a degenerate matrix".into(),
        )
    })?;

    // If the schedule processed no level at all (single-level pyramid whose
    // sole level is full resolution — see the module doc's degenerate-case
    // note), fall back to reporting the (identical, unchanged) full-res RMS
    // for both fields, exactly like the no-active-DOF pass-through above.
    let (starting_rms, final_rms) = starting_rms.map_or_else(
        || {
            let rms = lk_rms(reference, source, &matrix);
            (rms, rms)
        },
        |start| (start, final_rms),
    );

    Ok(LkResult {
        matrix,
        starting_rms,
        final_rms,
    })
}

/// Evaluate the registration RMS for `matrix` at full resolution.
///
/// A thin convenience wrapper over [`objective_registration`] (a
/// full-resolution forward transform, compared directly against the
/// full-resolution `reference` / `source` luma planes) so callers — e.g.
/// `pipeline::align_frame`'s Auto-mode fallback decision — can compare LK's
/// starting RMS against its final RMS without reaching into `core.rs`
/// directly.
#[must_use]
pub fn lk_rms(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> f64 {
    let w = reference.width;
    let h = reference.height;
    let p = matrix_to_params(matrix, w, h);
    if !p.is_finite() {
        return f64::MAX;
    }
    objective_registration(&p.to_vec(), &reference.luma, &source.luma, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::warp_image_clamped;

    fn make_gradient_image(width: usize, height: usize) -> PlanarImage<f32> {
        let mut img = PlanarImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = ((x as f32 * 0.11).sin() + (y as f32 * 0.09).cos())
                    .mul_add(0.4, 0.5)
                    .clamp(0.0, 1.0);
                img.luma[y * width + x] = v;
                img.chroma_a[y * width + x] = v * 0.5;
                img.chroma_b[y * width + x] = v * 0.25;
            }
        }
        img
    }

    #[test]
    fn lk_recovers_small_translation() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);

        let known_p = RegistrationParams {
            tx: 0.02,
            ty: -0.015,
            scale: 1.0,
            rotate: 0.0,
            aspect: 1.0,
            shear: 0.0,
        };
        let known_matrix = params_to_matrix(&known_p, w, h).expect("valid matrix");
        // `warp_image_clamped(src, m)` samples `src` via `m`'s INVERSE (the
        // standard backward-mapping convention — see `transform::warp`'s
        // module docs), whereas `objective_registration`'s `fwd` convention
        // (which this module's `params_to_matrix(p)` / `refine_alignment_lk`
        // also follow) samples the reference DIRECTLY via `fwd`, no
        // inversion. So the transform this function should recover — the
        // one satisfying `warped_ref(x) = ref(recovered * x) ~= source(x)`
        // — is `known_matrix`'s inverse, not `known_matrix` itself.
        let source = warp_image_clamped(&reference, &known_matrix).expect("warp ok");
        let expected_p = matrix_to_params(
            &known_matrix.try_inverse().expect("known_matrix invertible"),
            w,
            h,
        );

        let opts = BoundedRefineOptions::default();
        let result = refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts)
            .expect("refine_alignment_lk should succeed");

        let recovered = matrix_to_params(&result.matrix, w, h);
        assert!(
            (recovered.tx - expected_p.tx).abs() < 0.01,
            "tx: expected {}, got {}",
            expected_p.tx,
            recovered.tx
        );
        assert!(
            (recovered.ty - expected_p.ty).abs() < 0.01,
            "ty: expected {}, got {}",
            expected_p.ty,
            recovered.ty
        );
    }
}
