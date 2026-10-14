//! Intensity-based subpixel alignment refinement.
//!
//! This module implements intensity-based alignment refinement: starting from a
//! coarse transform (e.g. AKAZE + RANSAC), it minimises the RMS intensity
//! difference between the warped reference luma plane and the source luma plane
//! using a Nelder-Mead simplex optimiser over a 4-DOF parameter space
//! `(tx, ty, scale, rotate)`.
//!
//! ## Motion models and optimisers
//!
//! The 4-DOF model is the default.  The bounded refiners also search the
//! six-parameter model of [`AffineParams`] (`+ aspect, shear`) when
//! [`BoundedRefineOptions::allow_aspect`] / [`allow_shear`](BoundedRefineOptions::allow_shear)
//! are set; without them every search runs the arithmetic below bit for bit.
//! [`motion`] maps a motion model ([`Motion`]) and an optimiser ([`Optimizer`]:
//! the simplex, or the analytic-gradient [`lk`] Lucas-Kanade) onto these
//! refiners, [`gate`] guards a refined matrix against the identity, and
//! [`search`] is the resumable form of the simplex (Lucas-Kanade has none).
//!
//! ## Coarse-to-fine pyramid
//!
//! To avoid warping the full-resolution reference on every Nelder-Mead
//! evaluation, we build a Gaussian luma pyramid of both the reference and source
//! planes.  The optimiser runs from the coarsest level towards the finest
//! under one **coarse-biased level schedule** ([`level_plan`], GitHub's rule),
//! shared by every simplex refiner ([`refine_alignment_with_pyramid`],
//! [`refine_alignment_bounded_with_pyramid`] and
//! [`refine_alignment_registration_with_pyramid`]) and mirrored by the resumable
//! [`search::AlignSearch`]:
//!
//! - At each level the objective warps only that level's (smaller) luma plane.
//! - The best parameter vector carries over unchanged to the next finer level
//!   because `(tx, ty)` are *fractional* (pixels / image dimension), so they
//!   are scale-invariant across pyramid levels.
//! - A level with a short side above 512 px is "large": it gets 3 iterations,
//!   every other level 150 (each clamped to the caller's `max_iterations`).
//!   Full resolution is never a regular level, and the walk stops at the first
//!   large level among the finest two.  At 24 MP that means the levels
//!   62 .. 500 px at 150 iterations, the 1000 px level at 3 and nothing above.
//! - **Then a short full-resolution polish** ([`RefineOptions::polish_iterations`],
//!   40 by default): one more simplex run at full resolution, seeded from the
//!   last level, with an initial simplex shrunk to the expected residual of that
//!   level (a quarter of its pixel).  `polish_iterations = 0` is GitHub's
//!   schedule exactly.  See [`level_plan`] for the budget's reasoning.
//! - Restarts happen at the coarsest level only, ten of them: the first start is
//!   the seed, the other nine are *deterministic* offsets.  [`refine_alignment`]
//!   (unbounded) offsets `tx` and `ty` only: for j = 1..=9, `tx` moves by
//!   `+0.001·j` for even `j` and `-0.001·j` for odd `j`, `ty` by `+0.001·j` when
//!   `j` is a multiple of 3 and `-0.001·j` otherwise (fractions of the image
//!   size).  The bounded refiners jitter every logit coordinate by
//!   `(u - 0.5) · 0.001` (±0.0005) with `u` from a fixed-seed LCG, so the
//!   sequence is the same on every run and every machine.  The best objective
//!   wins.
//!
//! ## NaN guard
//!
//! If any parameter or projected coordinate is non-finite, or the transform is
//! not invertible, the objective returns a large finite penalty (`PENALTY`)
//! instead of panicking.  There is no early return: the optimiser keeps
//! iterating (the penalty simply loses every comparison), and the only
//! non-finite check on the result is made once after the last level, which
//! fails with [`StackerError::AlignmentFailed`].

#![expect(
    clippy::suboptimal_flops,
    reason = "plain multiply-add (no mul_add) keeps the objective and the simplex arithmetic identical on targets with and without FMA and to the recorded goldens"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "pixel counts and image dimensions are far below 2^52 (f64) and 2^24 (f32), so the usize to float conversions are exact"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the f64 optimiser parameters are narrowed to the f32 warp matrix on purpose"
)]

use std::cell::RefCell;

use nalgebra::Matrix3;
use rayon::prelude::*;
use stacker_core::{error::StackerError, fma::fma_f32, image::PlanarImage};

use crate::{
    device::{DeviceHandle, LevelDevice, Moments, ObjectiveSampler as Sampler},
    transform::{bicubic_interpolate_channel, source_in_bounds, spline4x4_sample_clamped},
};

/// The six-parameter (shift, scale, rotation, aspect, shear) model.
mod affine;
/// The post-refinement identity gate.
pub mod gate;
/// Lucas-Kanade / Gauss-Newton refinement, the analytic-gradient alternative to
/// the simplex.
pub mod lk;
/// Motion models and optimiser selection: the one entry point the pipeline maps
/// its alignment settings onto.
pub mod motion;
/// The resumable (ask/tell) form of the refiners, for objectives that answer
/// asynchronously.
pub mod search;

pub use affine::{AffineParams, affine_params_to_matrix, matrix_to_affine_params};
pub use gate::{apply_identity_gate, refined_beats_identity};
pub use lk::{LkResult, lk_rms, refine_alignment_lk, refine_alignment_lk_with_pyramid};
pub use motion::{Engine, Motion, MotionRefine, Optimizer, Refined, refine_motion};

// ── constants ─────────────────────────────────────────────────────────────────

/// Large finite penalty returned when parameters or projected coords are
/// non-finite.  Must be larger than any realistic RMS value (pixel values
/// are in [0,1] so RMS ≤ 1).
const PENALTY: f64 = 1.0e6;

/// Maximum Nelder-Mead iterations before declaring convergence.
const NM_MAX_ITER: usize = 400;

/// Nelder-Mead convergence tolerance: stops when the simplex spread in
/// objective-function value falls below this threshold.
const NM_TOL: f64 = 1.0e-7;

/// Initial simplex step sizes for each DOF.
/// tx/ty are normalised by image width/height (fractional pixel offsets).
const NM_STEP_TX: f64 = 0.002;
const NM_STEP_TY: f64 = 0.002;
const NM_STEP_SCALE: f64 = 0.002;
const NM_STEP_ROTATE: f64 = 0.001_5; // ≈ ±0.086 deg

/// Minimum short-side size (pixels) for a pyramid level.
/// Levels whose short side would fall below this are not generated.
const PYRAMID_MIN_SIZE: usize = 32;

/// Short side (pixels) above which a pyramid level is "large" (GitHub's
/// `REF_SHORT_SIDE_THRESHOLD`).
const LARGE_SHORT_SIDE: usize = 512;

/// Iterations of a large level (GitHub's `REF_LARGE_LEVEL_ITERS`).
const LARGE_LEVEL_ITERS: usize = 3;

/// Iterations of every other level (GitHub's `REF_DEFAULT_LEVEL_ITERS`).
const DEFAULT_LEVEL_ITERS: usize = 150;

/// Starts of the coarsest level: the seed and nine deterministic offsets.
const RESTARTS: u32 = 10;

/// Default of [`RefineOptions::polish_iterations`] and
/// [`BoundedRefineOptions::polish_iterations`]: the Nelder-Mead iterations of the
/// full-resolution polish that follows the coarse-biased schedule.
///
/// See [`level_plan`] for why this number.
pub const DEFAULT_POLISH_ITERATIONS: usize = 40;

// ── the level schedule ───────────────────────────────────────────────────────

/// What a [`LevelStep`] of the schedule is.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// The first level processed: ten starts (the seed and nine deterministic
    /// offsets), the best wins.
    Coarsest,
    /// A further level: one run from the carried parameters.
    Refine,
    /// The full-resolution polish: one run from the carried parameters with an
    /// initial simplex scaled to the expected residual.
    Polish,
}

/// One entry of the level schedule: which pyramid level to process, with how
/// many Nelder-Mead iterations, and in what role.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelStep {
    /// Pyramid level, `0` = coarsest, the last one = full resolution.
    pub level: usize,
    /// Nelder-Mead iterations (a cap; the simplex also stops on `NM_TOL`).
    pub iters: usize,
    /// The role of the step.
    pub kind: StepKind,
}

/// The coarse-biased level schedule of the Nelder-Mead refiners: GitHub's rule
/// plus a full-resolution polish.  One function serves the synchronous
/// refiners and [`search::AlignSearch`], so both walk the same levels.
///
/// `short_sides` are the short sides of the pyramid levels, coarsest first, the
/// last one being full resolution (see [`LumaPyramid`]; its coarsest level is
/// 32 to 63 px, which is GitHub's "pyramid down to about 64 px").
///
/// ## The rule (GitHub `refine/core.rs`)
///
/// Walking from the coarsest level:
///
/// - the full-resolution level is never a regular level;
/// - a level is *large* when its short side exceeds 512 px; the walk stops at
///   the first large level among the finest two (it is not processed);
/// - a large level gets 3 iterations, every other level 150, each clamped to
///   `max_iterations`;
/// - the first level processed is [`StepKind::Coarsest`] (restarts).
///
/// Examples: a 6000 × 4000 image has levels of 62, 125, 250, 500, 1000, 2000 and
/// 4000 px; the schedule is 62 .. 500 at 150 iterations, 1000 at 3, then the
/// polish at 4000.  A 1600 × 1600 image has 50, 100, 200, 400, 800, 1600: the
/// 800 px level is large and among the finest two, so the walk ends after
/// 400 px.  A 160 × 128 image has 32, 64, 128: levels 32 and 64 at 150, polish
/// at 128.
///
/// **Deviation for images with no coarse level.**  When the rule leaves
/// nothing to process (a pyramid of one level: short side below 64 px), GitHub
/// returns the seed unrefined.  Here the full-resolution level is processed as
/// the coarsest, regular level (150 iterations, restarts) instead, so a small
/// image is still aligned; the polish does not apply (there is no level to seed
/// it from).
///
/// ## The polish
///
/// If at least one coarse level was processed and `polish_iterations > 0`, a
/// final [`StepKind::Polish`] step at full resolution is appended with
/// `polish_iterations` iterations (clamped to `max_iterations`).
///
/// The budget reasoning: after the last level, which ran at `1/s` of the full
/// resolution, the parameters are correct to a fraction of a pixel *of that
/// level*; the polish assumes a quarter, i.e. `s / 4` full-resolution pixels
/// (the "expected residual").  It starts a simplex of that size (shift steps of
/// `s / 4` pixels over the image size, scale and rotation steps of `s / 4`
/// pixels over the half diagonal) instead of the default one, which at 24 MP is
/// several pixels wide.  A simplex that starts too small only costs a few
/// expansions (each doubles it), one that starts too large costs contractions
/// (each halves it), so the start errs small.  A Nelder-Mead simplex in 4
/// dimensions shrinks by about a factor of two every 5 to 8 iterations near a
/// smooth minimum, so 40 iterations bring a start of 1 px (a 4 times reduced
/// level) to a few hundredths of a pixel, at roughly 5 + 40 · 1.7 ≈ 75
/// full-resolution objective evaluations; the 400 iterations the finest level
/// used to get cost around 600.  The number is the `polish_iterations` option,
/// not a constant, because it trades accuracy for time and only the owner's
/// measurements can settle it.
#[doc(hidden)]
#[must_use]
pub fn level_plan(
    short_sides: &[usize],
    max_iterations: usize,
    polish_iterations: usize,
) -> Vec<LevelStep> {
    let n = short_sides.len();
    let mut plan: Vec<LevelStep> = Vec::with_capacity(n);
    for (idx, &short) in short_sides.iter().enumerate() {
        let large = short > LARGE_SHORT_SIDE;
        let full = idx + 1 == n;
        let finest_two = idx + 2 >= n;
        if full || (large && finest_two) {
            break;
        }
        let level_iters = if large {
            LARGE_LEVEL_ITERS
        } else {
            DEFAULT_LEVEL_ITERS
        };
        plan.push(LevelStep {
            level: idx,
            iters: level_iters.min(max_iterations),
            kind: if plan.is_empty() {
                StepKind::Coarsest
            } else {
                StepKind::Refine
            },
        });
    }
    if plan.is_empty() {
        if n > 0 {
            plan.push(LevelStep {
                level: n - 1,
                iters: DEFAULT_LEVEL_ITERS.min(max_iterations),
                kind: StepKind::Coarsest,
            });
        }
    } else if polish_iterations > 0 {
        plan.push(LevelStep {
            level: n - 1,
            iters: polish_iterations.min(max_iterations),
            kind: StepKind::Polish,
        });
    }
    plan
}

/// The schedule of a pair of pyramids plus what the polish needs from it.
pub(crate) struct Schedule {
    /// The steps, in order.
    pub(crate) steps: Vec<LevelStep>,
    /// Expected residual, in full-resolution pixels, of the last level before the
    /// polish: a quarter of the factor between full resolution and that level.  `0`
    /// without a polish.
    pub(crate) residual_px: f64,
}

impl Schedule {
    /// The schedule for `pyramid` under the caller's caps.
    pub(crate) fn new(
        pyramid: &LumaPyramid,
        max_iterations: usize,
        polish_iterations: usize,
    ) -> Self {
        let short_sides = pyramid.short_sides();
        let steps = level_plan(&short_sides, max_iterations, polish_iterations);
        let residual_px = match steps.as_slice() {
            [.., last_coarse, polish] if polish.kind == StepKind::Polish => {
                let full = short_sides.last().copied().unwrap_or(1);
                let at = short_sides.get(last_coarse.level).copied().unwrap_or(full);
                0.25 * full as f64 / at.max(1) as f64
            }
            _ => 0.0,
        };
        Self { steps, residual_px }
    }
}

/// Initial simplex steps of the polish for the unbounded 4-DOF model: the
/// default step of each parameter, but no larger than the expected residual of
/// `residual_px` pixels expressed in that parameter (shift: pixels over the
/// image size; scale and rotation: pixels over the half diagonal, where their
/// effect is largest).  A zero step (a pinned parameter) stays zero.
pub(crate) fn polish_steps_4d(
    step: &[f64; 4],
    residual_px: f64,
    width: usize,
    height: usize,
) -> [f64; 4] {
    let (w, h) = (width as f64, height as f64);
    let half_diagonal = 0.5 * w.hypot(h);
    let wanted = [
        residual_px / w,
        residual_px / h,
        residual_px / half_diagonal,
        residual_px / half_diagonal,
    ];
    std::array::from_fn(|i| {
        if step[i] == 0.0 {
            0.0
        } else {
            step[i].min(wanted[i])
        }
    })
}

/// Deterministic jitter for the restarts of the bounded refiners: GitHub's LCG
/// (Knuth's multiplier and addend) with a fixed seed, so every run and every
/// machine draws the same sequence.
pub(crate) struct JitterRng(u64);

impl JitterRng {
    /// 2^-53, to scale the upper 53 bits of the state into `[0, 1)`.
    const TWO_POW_NEG_53: f64 = 1.0 / 9_007_199_254_740_992.0;

    /// The generator at its fixed seed.
    pub(crate) const fn new() -> Self {
        Self(0x2545_F491_4F6C_DD1D)
    }

    /// The next jitter, uniform in `[-0.0005, 0.0005)`.
    pub(crate) fn jitter(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 * Self::TWO_POW_NEG_53 - 0.5) * 0.001
    }
}

/// Which bounded refiner: the same pyramid search, differing in the sampler of
/// the objective and in the wording of the errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flavor {
    /// [`refine_alignment_bounded_with_pyramid`]: the Catmull-Rom sampler, pixels
    /// without a sample left out.
    Bounded,
    /// [`refine_alignment_registration_with_pyramid`]: the edge-clamped spline
    /// sampler.
    Registration,
}

impl Flavor {
    /// The function name used as the prefix of error messages.
    pub(crate) const fn who(self) -> &'static str {
        match self {
            Self::Bounded => "refine_alignment_bounded",
            Self::Registration => "refine_alignment_registration",
        }
    }

    /// The sampler of the objective.
    pub(crate) const fn sampler(self) -> Sampler {
        match self {
            Self::Bounded => Sampler::Bicubic,
            Self::Registration => Sampler::SplineClamped,
        }
    }
}

// ── parameter model ──────────────────────────────────────────────────────────

/// Four-DOF registration parameter model.
///
/// - `tx` / `ty` are **fractional** offsets (pixels / image dimension).
/// - `scale` is a uniform scale factor (1.0 = no change).
/// - `rotate` is rotation in radians (0.0 = no rotation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegistrationParams {
    /// Horizontal shift as a fraction of the image width.
    pub tx: f64,
    /// Vertical shift as a fraction of the image height.
    pub ty: f64,
    /// Uniform scale about the image centre (1.0 = unchanged).
    pub scale: f64,
    /// Rotation about the image centre in radians (0.0 = unchanged).
    pub rotate: f64,
}

impl RegistrationParams {
    /// Identity parameters: no translation, no scale change, no rotation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            scale: 1.0,
            rotate: 0.0,
        }
    }

    /// Returns `true` if all fields are finite.
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.tx.is_finite()
            && self.ty.is_finite()
            && self.scale.is_finite()
            && self.rotate.is_finite()
    }

    /// Pack into a `[f64; 4]` for the optimizer.
    #[must_use]
    pub const fn to_vec(self) -> [f64; 4] {
        [self.tx, self.ty, self.scale, self.rotate]
    }

    /// Unpack from a `[f64; 4]`.
    const fn from_vec(v: &[f64; 4]) -> Self {
        Self {
            tx: v[0],
            ty: v[1],
            scale: v[2],
            rotate: v[3],
        }
    }
}

// ── transform builder ─────────────────────────────────────────────────────────

/// Build a 3×3 homogeneous forward transform from `RegistrationParams`.
///
/// ```text
/// M = T(tx*W, ty*H) · T(cx, cy) · R(rotate) · S(scale) · T(-cx, -cy)
/// ```
///
/// where `cx = W/2`, `cy = H/2`.  The image centre is the pivot for rotation
/// and scale; fractional translation is applied after.
///
/// Returns `None` if any parameter is non-finite.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    reason = "w/h, cx/cy, s and m are the textbook names of the pivot-scale-rotate matrix"
)]
pub fn params_to_matrix(
    p: &RegistrationParams,
    width: usize,
    height: usize,
) -> Option<Matrix3<f32>> {
    if !p.is_finite() {
        return None;
    }
    let w = width as f64;
    let h = height as f64;
    let cx = w * 0.5;
    let cy = h * 0.5;

    let cos_r = p.rotate.cos();
    let sin_r = p.rotate.sin();
    let s = p.scale;
    let tx = p.tx * w;
    let ty = p.ty * h;

    // Expand:
    //   translate(tx,ty) * translate(cx,cy) * rotate(r) * scale(s) * translate(-cx,-cy)
    //
    // = [ s*cos_r   -s*sin_r   cx*(1 - s*cos_r) + cy*s*sin_r  + tx ]
    //   [ s*sin_r    s*cos_r   cy*(1 - s*cos_r) - cx*s*sin_r  + ty ]
    //   [ 0          0         1                                    ]
    let a00 = s * cos_r;
    let a01 = -s * sin_r;
    let a10 = s * sin_r;
    let a11 = s * cos_r;
    let a02 = cx * (1.0 - a00) + cy * s * sin_r + tx;
    let a12 = cy * (1.0 - a11) - cx * s * sin_r + ty;

    for v in [a00, a01, a02, a10, a11, a12] {
        if !v.is_finite() {
            return None;
        }
    }

    let mut m = Matrix3::<f32>::identity();
    m[(0, 0)] = a00 as f32;
    m[(0, 1)] = a01 as f32;
    m[(0, 2)] = a02 as f32;
    m[(1, 0)] = a10 as f32;
    m[(1, 1)] = a11 as f32;
    m[(1, 2)] = a12 as f32;
    Some(m)
}

/// Decompose a 3×3 affine matrix into `RegistrationParams`.
///
/// Uses the image centre as the rotation/scale pivot (same convention as
/// [`params_to_matrix`]). Returns `RegistrationParams::identity()` for a
/// non-finite or degenerate matrix so the optimiser always has a safe starting
/// point.
///
/// ## Reduction rule for general affine input
///
/// The four-parameter model can only express a similarity (uniform scale,
/// rotation, translation), so a matrix with shear or anisotropic scale is
/// reduced, explicitly and symmetrically in both columns, as follows.  With the
/// linear part `A = [[a00, a01], [a10, a11]]`:
///
/// - `scale·cos(rotate) = (a00 + a11) / 2` and `scale·sin(rotate) = (a10 - a01) / 2`.
///   This is the least-squares (Frobenius-nearest) similarity to `A`.
/// - The translation is chosen so that the image centre `c` maps to the same
///   point under the reduced similarity as under the full affine map
///   (`A·c + t`), i.e. the reduction is exact at the pivot.
///
/// For an exact similarity (`a00 == a11`, `a01 == -a10`) this reproduces the
/// inverse of [`params_to_matrix`] exactly.  Callers that need the shear and
/// anisotropy kept (the six-parameter affine model) must not pass the result
/// of this function through [`params_to_matrix`] and expect the original
/// matrix back; on a failed refinement they should fall back to the reduced
/// similarity, not the raw seed, so the applied model does not depend on which
/// branch ran.
#[must_use]
pub fn matrix_to_params(m: &Matrix3<f32>, width: usize, height: usize) -> RegistrationParams {
    if !m.iter().all(|v| v.is_finite()) {
        return RegistrationParams::identity();
    }
    let w = width as f64;
    let h = height as f64;
    let cx = w * 0.5;
    let cy = h * 0.5;

    let a00 = f64::from(m[(0, 0)]);
    let a01 = f64::from(m[(0, 1)]);
    let a10 = f64::from(m[(1, 0)]);
    let a11 = f64::from(m[(1, 1)]);
    let a02 = f64::from(m[(0, 2)]);
    let a12 = f64::from(m[(1, 2)]);

    // Nearest similarity: s·cos = (a00 + a11) / 2, s·sin = (a10 - a01) / 2.
    let sim_cos = f64::midpoint(a00, a11);
    let sim_sin = 0.5 * (a10 - a01);
    let scale = sim_cos.hypot(sim_sin);
    if scale < 1.0e-10 {
        return RegistrationParams::identity();
    }

    let rotate = sim_sin.atan2(sim_cos);

    // Absolute translation = displacement of the image centre under the full
    // affine map: (A·c + t) - c.  For a similarity built by `params_to_matrix`
    // this equals `a02 - cx*(1 - a00) - cy*a10` (resp. the y analogue).
    let (tx, ty) = {
        let abs_x = a00 * cx + a01 * cy + a02 - cx;
        let abs_y = a10 * cx + a11 * cy + a12 - cy;
        (abs_x / w, abs_y / h)
    };

    if !tx.is_finite() || !ty.is_finite() || !scale.is_finite() || !rotate.is_finite() {
        return RegistrationParams::identity();
    }

    RegistrationParams {
        tx,
        ty,
        scale,
        rotate,
    }
}

/// The similarity that [`refine_alignment`] and friends actually optimise from
/// `m`: `params_to_matrix(matrix_to_params(m))`, using the reduction rule
/// documented on [`matrix_to_params`].
///
/// Apply this (not the raw seed) when a refinement fails, so the model that
/// ends up in the output is the same similarity on both the success and the
/// failure branch.  Returns the identity for a non-finite or degenerate `m`.
#[must_use]
pub fn reduce_to_similarity(m: &Matrix3<f32>, width: usize, height: usize) -> Matrix3<f32> {
    params_to_matrix(&matrix_to_params(m, width, height), width, height)
        .unwrap_or_else(Matrix3::identity)
}

// ── Gaussian / box luma pyramid ───────────────────────────────────────────────

/// A single level of a luma pyramid: raw pixel data plus dimensions.
#[doc(hidden)]
pub struct PyramidLevel {
    /// Row-major luma samples, `width * height` of them.
    pub luma: Vec<f32>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

/// Downsample a luma plane by a factor of 2 using a separable 3-tap box
/// (1/4, 1/2, 1/4) blur + stride-2 decimation.
///
/// This is equivalent to one level of a Gaussian pyramid and is entirely
/// self-contained — no extra crate dependencies. Hidden from the rendered
/// documentation: the alignment code uses it through [`LumaPyramid`]; it is
/// public only so the integration tests can pin it.
#[doc(hidden)]
#[must_use]
pub fn downsample_luma(src: &[f32], src_w: usize, src_h: usize) -> (Vec<f32>, usize, usize) {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);

    // Horizontal pass: blur along X into a `src_h × dst_w` intermediate.
    let mut h_blur = vec![0.0_f32; src_h * dst_w];
    h_blur
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(y, out_row)| {
            let in_row = &src[y * src_w..(y + 1) * src_w];
            for (dx, out) in out_row.iter_mut().enumerate() {
                let sx = dx * 2;
                let left = if sx > 0 { in_row[sx - 1] } else { in_row[0] };
                let center = in_row[sx];
                let right = if sx + 1 < src_w {
                    in_row[sx + 1]
                } else {
                    in_row[src_w - 1]
                };
                *out = fma_f32(left, 0.25, fma_f32(center, 0.5, right * 0.25));
            }
        });

    // Vertical pass: blur along Y into the final `dst_h × dst_w` plane.
    let mut dst = vec![0.0_f32; dst_h * dst_w];
    dst.par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, out_row)| {
            let sy = dy * 2;
            let top = if sy > 0 { sy - 1 } else { 0 };
            let bot = if sy + 1 < src_h { sy + 1 } else { src_h - 1 };
            let row_top = &h_blur[top * dst_w..(top + 1) * dst_w];
            let row_mid = &h_blur[sy * dst_w..(sy + 1) * dst_w];
            let row_bot = &h_blur[bot * dst_w..(bot + 1) * dst_w];
            for x in 0..dst_w {
                out_row[x] = fma_f32(
                    row_top[x],
                    0.25,
                    fma_f32(row_mid[x], 0.5, row_bot[x] * 0.25),
                );
            }
        });

    (dst, dst_w, dst_h)
}

/// A luma pyramid that borrows its full-resolution level from the image it was
/// built from, so no full-size copy is made.
///
/// Level `0` is the coarsest, the last level is the image's own luma plane.
/// Build the reference's pyramid **once** per run and hand it to
/// [`refine_alignment_with_pyramid`] / [`refine_alignment_bounded_with_pyramid`]
/// for every frame; the plain [`refine_alignment`] / [`refine_alignment_bounded`]
/// entry points rebuild it per call.
pub struct LumaPyramid {
    /// Levels below full resolution, coarsest first.
    coarse: Vec<PyramidLevel>,
    width: usize,
    height: usize,
}

impl LumaPyramid {
    /// Build the pyramid of `img`'s luma plane.
    ///
    /// Halves until the short side would drop below [`PYRAMID_MIN_SIZE`]; very
    /// small images keep a single (full-resolution) level.
    #[must_use]
    pub fn new(img: &PlanarImage<f32>) -> Self {
        // Finest-first while building, reversed at the end.
        let mut coarse: Vec<PyramidLevel> = Vec::new();
        let (mut w, mut h) = (img.width, img.height);
        loop {
            // Stop if halving would go below the minimum.
            if w.min(h) / 2 < PYRAMID_MIN_SIZE {
                break;
            }
            let (luma, nw, nh) = coarse.last().map_or_else(
                || downsample_luma(&img.luma, w, h),
                |last| downsample_luma(&last.luma, w, h),
            );
            coarse.push(PyramidLevel {
                luma,
                width: nw,
                height: nh,
            });
            (w, h) = (nw, nh);
        }
        coarse.reverse();
        Self {
            coarse,
            width: img.width,
            height: img.height,
        }
    }

    /// Number of levels, the full-resolution one included.
    #[must_use]
    pub const fn n_levels(&self) -> usize {
        self.coarse.len() + 1
    }

    /// The short side of every level, coarsest first, full resolution last: the
    /// input of [`level_plan`].
    pub(crate) fn short_sides(&self) -> Vec<usize> {
        self.coarse
            .iter()
            .map(|l| l.width.min(l.height))
            .chain(std::iter::once(self.width.min(self.height)))
            .collect()
    }

    /// Level `idx` (0 = coarsest) as `(luma, width, height)`; the last level is
    /// `full`, the luma plane of the image this pyramid was built from. For the
    /// GPU objective, which uploads the levels, and for profiling.
    #[doc(hidden)]
    #[must_use]
    pub fn level_luma<'a>(&'a self, idx: usize, full: &'a [f32]) -> (&'a [f32], usize, usize) {
        let view = self.level(idx, full);
        (view.luma, view.width, view.height)
    }

    /// Level `idx` (0 = coarsest); the last level is `full`, the luma plane of
    /// the image this pyramid was built from.
    fn level<'a>(&'a self, idx: usize, full: &'a [f32]) -> LevelView<'a> {
        self.coarse.get(idx).map_or(
            LevelView {
                luma: full,
                width: self.width,
                height: self.height,
            },
            |l| LevelView {
                luma: &l.luma,
                width: l.width,
                height: l.height,
            },
        )
    }
}

/// Borrowed view of one pyramid level.
#[derive(Clone, Copy)]
struct LevelView<'a> {
    luma: &'a [f32],
    width: usize,
    height: usize,
}

/// Build a luma pyramid for `img`.
///
/// The first level is the original full-resolution luma (cloned).  Each
/// subsequent level is half the size.  Generation stops when the short side
/// would drop below [`PYRAMID_MIN_SIZE`].  For very small images (short side
/// already < `2 * PYRAMID_MIN_SIZE`) only a single level is returned so the
/// pyramid still degrades gracefully.  The alignment code uses the borrowing
/// [`LumaPyramid`] instead; this owned form is public only so the integration
/// tests can inspect the levels.
#[doc(hidden)]
#[must_use]
pub fn build_luma_pyramid(img: &PlanarImage<f32>) -> Vec<PyramidLevel> {
    let mut levels = LumaPyramid::new(img).coarse;
    // Reversed order: index 0 = coarsest, last = finest.
    levels.push(PyramidLevel {
        luma: img.luma.clone(),
        width: img.width,
        height: img.height,
    });
    levels
}

// ── objective function ────────────────────────────────────────────────────────

/// Compute mean-bias-corrected RMS difference between two luma planes.
///
/// ```text
/// diff_i = (src[i] − mean_src) − (warped_ref[i] − mean_ref)
/// RMS = sqrt( sum(diff_i²) / (N − 1) )
/// ```
///
/// Subtracting the per-image mean makes the objective invariant to global
/// brightness offset — useful when breathing correction changes apparent
/// exposure.  Parallelised with rayon over rows for planes of at least
/// [`SERIAL_LEVEL_PIXELS`] pixels; smaller planes (the coarse pyramid levels)
/// are summed on the calling thread, where a fork/join costs more than the work.
/// The per-row partial sums are always added serially in row order, so the
/// result is bit-identical for any thread count.
///
/// Public only so the integration tests can pin the formula; the refinement
/// itself evaluates `masked_rms`, which leaves out pixels without a sample.
#[doc(hidden)]
#[must_use]
pub fn rms_difference(src: &[f32], warped_ref: &[f32], width: usize, height: usize) -> f64 {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(warped_ref.len(), width * height);
    let n = width * height;
    if n == 0 {
        return 0.0;
    }
    let parallel = n >= SERIAL_LEVEL_PIXELS;
    let mean_src = plane_sum(src, width, parallel) / n as f64;
    let mean_ref = plane_sum(warped_ref, width, parallel) / n as f64;
    rms_from_means(src, warped_ref, width, mean_src, mean_ref)
}

/// Pixel count below which a pyramid level is evaluated on the calling thread.
/// A rayon fork/join costs a few microseconds; a 128 × 128 level takes about as
/// long to warp serially, and the coarsest level is evaluated up to
/// 10 restarts × 50 iterations × (n + 2) times.
const SERIAL_LEVEL_PIXELS: usize = 128 * 128;

/// Sum of a plane in `f64`.
///
/// The per-row partial sums may be computed in parallel, but they are collected
/// into a vector (rayon's indexed `collect` keeps row order) and added serially
/// in row order.  The result therefore does not depend on thread count or
/// work-stealing; rayon's own `sum`/`reduce` associate in a scheduling-dependent
/// tree, which made the objective's low bits (and with them the simplex path)
/// vary between runs.
fn plane_sum(plane: &[f32], width: usize, parallel: bool) -> f64 {
    let row_sum = |row: &[f32]| -> f64 { row.iter().map(|&v| f64::from(v)).sum() };
    let rows: Vec<f64> = if parallel {
        plane.par_chunks(width).map(row_sum).collect()
    } else {
        plane.chunks(width).map(row_sum).collect()
    };
    rows.iter().sum()
}

/// `sqrt(Σ((s − mean_s) − (r − mean_r))² / (N − 1))`, the second pass of
/// [`rms_difference`], given the two plane means.
fn rms_from_means(
    src: &[f32],
    warped_ref: &[f32],
    width: usize,
    mean_src: f64,
    mean_ref: f64,
) -> f64 {
    let n = src.len();
    let row_var = |(row_s, row_r): (&[f32], &[f32])| -> f64 {
        let mut acc = 0.0_f64;
        for (&vs, &vr) in row_s.iter().zip(row_r.iter()) {
            let d = (f64::from(vs) - mean_src) - (f64::from(vr) - mean_ref);
            acc += d * d;
        }
        acc
    };
    // Per-row partials, added serially in row order (see `plane_sum`).
    let rows: Vec<f64> = if n >= SERIAL_LEVEL_PIXELS {
        src.par_chunks(width)
            .zip(warped_ref.par_chunks(width))
            .map(row_var)
            .collect()
    } else {
        src.chunks(width)
            .zip(warped_ref.chunks(width))
            .map(row_var)
            .collect()
    };
    let var_sum: f64 = rows.iter().sum();
    (var_sum / (n as f64 - 1.0).max(1.0)).sqrt()
}

// ── objective wrapper ─────────────────────────────────────────────────────────

/// Evaluate the RMS objective for a given parameter vector at a single
/// pyramid level described by raw luma slices.
///
/// Returns `PENALTY` for non-finite parameters, a non-invertible transform, or
/// a transform under which fewer than half of the pixels can be sampled from
/// the reference. The optimiser is not stopped by it (there is no early
/// return); the penalty simply loses every comparison against a finite
/// objective.
///
/// Only pixels whose inverse-mapped position lies inside the reference take
/// part in the RMS and in the mean correction. Counting the zero-filled
/// out-of-frame pixels as data would bias the optimum towards the identity by an
/// amount proportional to the shift's share of the frame (2 % of all pixels for
/// a 3 px shift on a 128 px test image; negligible at 4000 px).
///
/// Public only so the integration tests can evaluate the objective directly.
#[doc(hidden)]
#[must_use]
pub fn objective_at_level(
    params_vec: &[f64; 4],
    ref_luma: &[f32],
    src_luma: &[f32],
    width: usize,
    height: usize,
) -> f64 {
    LevelObjective::new(ref_luma, src_luma, width, height, Sampler::Bicubic, None).eval(params_vec)
}

/// The RMS objective at one pyramid level, with everything that does not change
/// between evaluations hoisted out of the optimiser's inner loop:
///
/// - the warped-reference buffer is allocated **once** per level and reused by
///   every evaluation (a fresh `width × height` `Vec` per call would be up to
///   400 allocations of the full level per frame);
/// - the mean of the (fixed) source plane is computed once instead of once per
///   evaluation;
/// - the warp and the reference-plane sum share one pass, so an evaluation is
///   two passes over the level: warp + sum, then the centred squares.
///
/// The value is independent of thread count and scheduling: per-row partial sums
/// are added serially in row order (see [`masked_rms`]).
struct LevelObjective<'a> {
    warper: Warper<'a>,
    src_luma: &'a [f32],
    /// Warped reference, reused across evaluations.  Evaluations are strictly
    /// sequential (the optimiser is single-threaded; the parallelism is inside
    /// one evaluation), so the borrow can never overlap.
    scratch: RefCell<Vec<f32>>,
    /// The level prepared on a device, when the caller supplied one and it took
    /// the level; evaluations then run there (the CPU covers a failed one).
    device: Option<RefCell<Box<dyn LevelDevice>>>,
}

/// The `Sync` part of [`LevelObjective`]: everything a rayon worker needs to
/// warp one row.
#[derive(Clone, Copy)]
struct Warper<'a> {
    ref_luma: &'a [f32],
    width: usize,
    height: usize,
    sampler: Sampler,
}

impl Warper<'_> {
    /// Warp one output row of the reference. A pixel whose inverse-mapped
    /// position lies outside the reference is written as `NaN` so the objective
    /// can leave it out; the clamped spline sampler never produces one.
    fn row(&self, inv: &Matrix3<f32>, y: usize, row: &mut [f32]) {
        for (x, dst) in row.iter_mut().enumerate() {
            let p_dst = nalgebra::Vector3::new(x as f32, y as f32, 1.0_f32);
            let p_src = inv * p_dst;
            let denom = p_src[2];
            if denom.abs() < 1.0e-8 {
                *dst = f32::NAN;
                continue;
            }
            let sx = p_src[0] / denom;
            let sy = p_src[1] / denom;
            *dst = match self.sampler {
                Sampler::Bicubic => {
                    if source_in_bounds(sx, sy, self.width, self.height) {
                        bicubic_interpolate_channel(self.ref_luma, self.width, self.height, sx, sy)
                    } else {
                        f32::NAN
                    }
                }
                Sampler::SplineClamped => {
                    spline4x4_sample_clamped(self.ref_luma, self.width, self.height, sx, sy)
                }
            };
        }
    }
}

/// The RMS of [`masked_rms`] from a device's moments: `sqrt(m2 / (N - 1))`, and
/// `0` for fewer than two pixels.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a level's pixel count fits a usize"
)]
fn rms_of(moments: Moments) -> (f64, usize) {
    let count = moments.count as usize;
    if count < 2 {
        return (0.0, count);
    }
    ((moments.m2.max(0.0) / (count as f64 - 1.0)).sqrt(), count)
}

/// Mean-corrected RMS over the pixels where `warped_ref` holds a sample (not
/// `NaN`), computed as `sqrt((Σd² − (Σd)²/N) / (N − 1))` with `d = src − ref`,
/// which equals the centred form without a separate pass for the two means.
/// Returns the RMS and the number of pixels that took part.
fn masked_rms(src: &[f32], warped_ref: &[f32], width: usize) -> (f64, usize) {
    let row_sums = |(row_s, row_r): (&[f32], &[f32])| -> (f64, f64, usize) {
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut count = 0_usize;
        for (&vs, &vr) in row_s.iter().zip(row_r.iter()) {
            if vr.is_nan() {
                continue;
            }
            let d = f64::from(vs) - f64::from(vr);
            sum += d;
            sum_sq += d * d;
            count += 1;
        }
        (sum, sum_sq, count)
    };
    // Per-row partials, added serially in row order, so the value does not depend
    // on rayon's scheduling (see `plane_sum`).
    let rows: Vec<(f64, f64, usize)> = if src.len() >= SERIAL_LEVEL_PIXELS {
        src.par_chunks(width)
            .zip(warped_ref.par_chunks(width))
            .map(row_sums)
            .collect()
    } else {
        src.chunks(width)
            .zip(warped_ref.chunks(width))
            .map(row_sums)
            .collect()
    };
    let (sum, sum_sq, count) = rows.iter().fold((0.0_f64, 0.0_f64, 0_usize), |a, b| {
        (a.0 + b.0, a.1 + b.1, a.2 + b.2)
    });
    if count < 2 {
        return (0.0, count);
    }
    let n = count as f64;
    let var = (sum_sq - sum * sum / n).max(0.0) / (n - 1.0);
    (var.sqrt(), count)
}

impl<'a> LevelObjective<'a> {
    fn new(
        ref_luma: &'a [f32],
        src_luma: &'a [f32],
        width: usize,
        height: usize,
        sampler: Sampler,
        device: Option<&DeviceHandle>,
    ) -> Self {
        let n = width * height;
        debug_assert_eq!(ref_luma.len(), n);
        debug_assert_eq!(src_luma.len(), n);
        let device = device
            .and_then(|d| d.0.level(ref_luma, src_luma, width, height, sampler))
            .map(RefCell::new);
        Self {
            warper: Warper {
                ref_luma,
                width,
                height,
                sampler,
            },
            src_luma,
            // The CPU buffer is only needed without a device; one that fails later
            // allocates it on first use.
            scratch: RefCell::new(if device.is_some() {
                Vec::new()
            } else {
                vec![0.0_f32; n]
            }),
            device,
        }
    }

    /// Objective for `params_vec`; [`PENALTY`] for non-finite parameters or a
    /// non-invertible transform.
    fn eval(&self, params_vec: &[f64; 4]) -> f64 {
        let ctx = self.warper;
        let (width, height) = (ctx.width, ctx.height);
        let p = RegistrationParams::from_vec(params_vec);
        if !p.is_finite() {
            return PENALTY;
        }
        let Some(fwd) = params_to_matrix(&p, width, height) else {
            return PENALTY;
        };
        self.eval_forward(&fwd)
    }

    /// Objective for the six-parameter model; [`PENALTY`] for non-finite
    /// parameters or a non-invertible transform.  For `aspect == 1`,
    /// `shear == 0` it is [`Self::eval`], bit for bit.
    fn eval_affine(&self, p: &AffineParams) -> f64 {
        p.to_matrix(self.warper.width, self.warper.height)
            .map_or(PENALTY, |fwd| self.eval_forward(&fwd))
    }

    /// Objective for the forward matrix `fwd` (reference to source); the part of
    /// [`Self::eval`] after the matrix is built.
    fn eval_forward(&self, fwd: &Matrix3<f32>) -> f64 {
        let ctx = self.warper;
        let (width, height) = (ctx.width, ctx.height);
        let Some(inv) = fwd.try_inverse() else {
            return PENALTY;
        };
        if !inv.iter().all(|v| v.is_finite()) {
            return PENALTY;
        }

        let n = width * height;
        if n == 0 {
            return 0.0;
        }

        let (rms, covered) = self
            .device
            .as_ref()
            .and_then(|d| d.borrow_mut().moments(&inv))
            .map_or_else(|| self.cpu_rms(&inv), rms_of);
        // A transform that samples less than half of the frame is not a
        // candidate: the trivial "shift everything out" solution must lose.
        if covered * 2 < n {
            return PENALTY;
        }
        rms
    }

    /// Warp and RMS on the CPU.
    fn cpu_rms(&self, inv: &Matrix3<f32>) -> (f64, usize) {
        let ctx = self.warper;
        let (width, n) = (ctx.width, ctx.width * ctx.height);
        // Warp the reference into the source frame via the inverse mapping;
        // pixels without a source sample come back as NaN.
        let mut warped = self.scratch.borrow_mut();
        if warped.len() != n {
            warped.resize(n, 0.0);
        }
        if n >= SERIAL_LEVEL_PIXELS {
            warped
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| ctx.row(inv, y, row));
        } else {
            warped
                .chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| ctx.row(inv, y, row));
        }
        masked_rms(self.src_luma, &warped, width)
    }
}

// ── Nelder-Mead simplex ───────────────────────────────────────────────────────

// (The Nelder-Mead minimiser, `nelder_mead_generic`, lives below with the
// bounded search; the unbounded 4-DOF refinement uses it through
// `minimize_4d`.)

// ── public API ────────────────────────────────────────────────────────────────

/// Options controlling the intensity-based refinement pass.
#[derive(Debug, Clone)]
pub struct RefineOptions {
    /// Cap on the Nelder-Mead iterations of every level of the schedule
    /// ([`level_plan`]): the levels run 150 iterations (3 for a large level) and
    /// the polish [`Self::polish_iterations`], each clamped to this value, so no
    /// level exceeds it.  At the coarsest level the cap applies to each of the
    /// ten restarts separately.  Defaults to [`NM_MAX_ITER`] (400), which no
    /// level reaches; lower it to shorten every level.
    pub max_iterations: usize,
    /// Iterations of the full-resolution polish that follows the coarse levels
    /// (default [`DEFAULT_POLISH_ITERATIONS`]).  `0` switches the polish off,
    /// which is GitHub's schedule exactly: full resolution is never processed.
    pub polish_iterations: usize,
    /// If `true`, optimise scale (default: `true`).
    pub allow_scale: bool,
    /// If `true`, optimise rotation (default: `true`).
    pub allow_rotate: bool,
    /// Evaluate the objective on this device where it accepts a level (default:
    /// none, the CPU); see [`crate::device`].
    pub device: Option<DeviceHandle>,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            max_iterations: NM_MAX_ITER,
            polish_iterations: DEFAULT_POLISH_ITERATIONS,
            allow_scale: true,
            allow_rotate: true,
            device: None,
        }
    }
}

/// Refine a coarse alignment matrix using intensity-based Nelder-Mead optimisation.
///
/// ## Algorithm (coarse-to-fine pyramid)
///
/// 1. Decompose `initial` into `RegistrationParams (tx, ty, scale, rotate)`.
/// 2. Build Gaussian luma pyramids for both `reference` and `source`
///    (levels at full, 1/2, 1/4, … resolution; coarsest ≥ 32 px short side).
/// 3. Run Nelder-Mead from the *coarsest* level along the coarse-biased
///    schedule of [`level_plan`]:
///    - Each level's objective warps only that level's (smaller) luma.
///    - The best parameter vector transfers unchanged (params are normalised).
///    - 150 iterations per level (3 for a large one), ten restarts at the
///      coarsest; full resolution only gets the short polish
///      ([`RefineOptions::polish_iterations`]).
/// 4. Return the refined 3×3 matrix `M` mapping **reference coordinates to
///    source coordinates**: at the optimum `source(M·p) ≈ reference(p)`.
///
/// ## Direction convention
///
/// The objective warps the reference by `M⁻¹` and compares it with the source,
/// so `M` carries a reference pixel to where the same content sits in the
/// source.  To bring `source` into the reference frame, warp it with the
/// **inverse**: `transform::warp_image(source, &m.try_inverse()?)`.  Passing
/// `M` itself to `warp_image` displaces the frame by twice its offset.  The
/// RANSAC seed (`ransac::AlignmentEstimator`) fits the same direction.
///
/// ## Image roles
///
/// - `reference`: the fixed frame (frame 0 in the pipeline).
/// - `source`: the target frame being aligned **to** the reference.
/// - `initial`: coarse matrix in the same reference→source convention, from
///   AKAZE/RANSAC or identity.
///
/// ## Errors
///
/// - [`StackerError::MathError`] — `initial` contains non-finite values.
/// - [`StackerError::AlignmentFailed`] — dimension mismatch, or the optimiser
///   produced a non-finite parameter vector (checked once, after the last
///   pyramid level).
pub fn refine_alignment(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &RefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    refine_alignment_with_pyramid(
        reference,
        &LumaPyramid::new(reference),
        source,
        initial,
        opts,
    )
}

/// Check that a prebuilt reference pyramid belongs to `reference`.
fn check_pyramid(
    who: &str,
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
) -> Result<(), StackerError> {
    if ref_pyramid.width != reference.width || ref_pyramid.height != reference.height {
        return Err(StackerError::AlignmentFailed(format!(
            "{who}: reference pyramid is {}x{} but the reference image is {}x{}",
            ref_pyramid.width, ref_pyramid.height, reference.width, reference.height
        )));
    }
    Ok(())
}

/// [`refine_alignment`] with the reference's luma pyramid supplied by the
/// caller.
///
/// A stack aligns N−1 frames to one reference.  Build
/// `LumaPyramid::new(reference)` once and pass it for every frame: this skips
/// rebuilding the reference pyramid (about 5.3 bytes per pixel of allocation and
/// two passes per level) N−1 times.  `ref_pyramid` must have been built from
/// `reference`; the result is identical to [`refine_alignment`].
///
/// ## Errors
///
/// As [`refine_alignment`], plus [`StackerError::AlignmentFailed`] when
/// `ref_pyramid` was built from an image of a different size.
pub fn refine_alignment_with_pyramid(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &RefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(StackerError::MathError(
            "refine_alignment: initial matrix contains non-finite values".into(),
        ));
    }
    if reference.width != source.width || reference.height != source.height {
        return Err(StackerError::AlignmentFailed(
            "refine_alignment: reference and source dimensions do not match".into(),
        ));
    }
    check_pyramid("refine_alignment", reference, ref_pyramid)?;

    let w = source.width;
    let h = source.height;

    let p0 = matrix_to_params(initial, w, h);

    let step = [
        NM_STEP_TX,
        NM_STEP_TY,
        if opts.allow_scale { NM_STEP_SCALE } else { 0.0 },
        if opts.allow_rotate {
            NM_STEP_ROTATE
        } else {
            0.0
        },
    ];

    // Coarse-to-fine pyramid of the source luma (the reference's was supplied).
    let src_pyramid = LumaPyramid::new(source);

    // Pyramids must have the same depth (they were built from same-sized images).
    debug_assert_eq!(ref_pyramid.n_levels(), src_pyramid.n_levels());
    let schedule = Schedule::new(ref_pyramid, opts.max_iterations, opts.polish_iterations);

    let mut best = p0.to_vec();

    for planned in &schedule.steps {
        let ref_lvl = ref_pyramid.level(planned.level, &reference.luma);
        let src_lvl = src_pyramid.level(planned.level, &source.luma);

        // One objective per level: owns the warp buffer and the source mean.
        let objective = LevelObjective::new(
            ref_lvl.luma,
            src_lvl.luma,
            ref_lvl.width,
            ref_lvl.height,
            Sampler::Bicubic,
            opts.device.as_ref(),
        );

        // The polish starts a simplex sized to the expected residual.
        let level_step = if planned.kind == StepKind::Polish {
            polish_steps_4d(&step, schedule.residual_px, w, h)
        } else {
            step
        };
        let minimize = |x0: [f64; 4]| minimize_4d(&objective, x0, &level_step, planned.iters);

        if planned.kind == StepKind::Coarsest {
            let mut best_obj = f64::INFINITY;
            let mut best_params = best;
            for j in 0..RESTARTS {
                let mut start = best;
                if j > 0 {
                    let factor = f64::from(j) * 0.001;
                    start[0] += factor * if j % 2 == 0 { 1.0 } else { -1.0 };
                    start[1] += factor * if j % 3 == 0 { 1.0 } else { -1.0 };
                }
                let res = minimize(start);
                let obj = objective.eval(&res);
                if obj < best_obj {
                    best_obj = obj;
                    best_params = res;
                }
            }
            best = best_params;
        } else {
            best = minimize(best);
        }
    }

    let best_p = RegistrationParams::from_vec(&best);

    if !best_p.is_finite() {
        return Err(StackerError::AlignmentFailed(
            "refine_alignment: optimizer produced non-finite parameters (NaN encountered)".into(),
        ));
    }

    let refined = params_to_matrix(&best_p, w, h).ok_or_else(|| {
        StackerError::AlignmentFailed(
            "refine_alignment: refined parameters produced a degenerate matrix".into(),
        )
    })?;

    Ok(refined)
}

// ── Bounded registration ("Bounded" alignment mode) ──────────────────────────
//
// Each enabled degree of freedom is optimised inside an explicit `[lower, upper]`
// interval centred on the initial estimate, in a *logit-transformed* search
// space.  Because the inverse map is a sigmoid, the parameters can never leave
// their bounds, so a wandering simplex can no longer collapse the transform into
// a degenerate (near-singular) matrix — the key robustness win over the
// unconstrained `refine_alignment` path.

/// Half-width of the logit bracket the single-DOF golden-section search covers.
/// `sigmoid(±8)` is within 0.04 % of the band's edges, so the whole band is
/// reachable while every probe still sees a varying objective.
const GOLDEN_BRACKET: f64 = 8.0;
/// Interval tolerance (in logit space) of the single-DOF golden-section search.
const GOLDEN_TOL: f64 = 1.0e-6;
/// Initial simplex step in the (logit) search space for the multi-DOF case.
const BOUNDED_STEP: f64 = 0.5;

/// Per-DOF search configuration: which degrees of freedom are active and their
/// allowed half-width intervals (rotation, scale, shift X/Y).
///
/// A DOF whose `allow_*` flag is `false` is not searched: it is pinned to the
/// value decomposed from the seed matrix (see [`matrix_to_params`]), i.e. its
/// scale stays at the seed's scale and its rotation at the seed's rotation.
///
/// **`Translation` alignment mode must pass `allow_scale: false` and
/// `allow_rotation: false`.**  [`Default`] enables all four DOFs, so a caller
/// that builds the options with `BoundedRefineOptions::default()` gets a full
/// similarity fit (scale ±5 %, rotation ±5°) regardless of the requested mode.
/// The default shift interval is also ±5 % of the image size, which bounds the
/// shift that can be recovered around the seed.
///
/// ## The six-parameter model
///
/// `allow_aspect` and `allow_shear` (both off by default) add the two affine
/// degrees of freedom of [`AffineParams`].  As soon as either is set
/// ([`Self::uses_affine`]) the refiners decompose the seed exactly
/// ([`matrix_to_affine_params`]) instead of reducing it to a similarity, and
/// search the aspect inside `±max_aspect %` and the shear inside `±max_shear %`
/// of their seed values.  With both off every refiner runs the four-parameter
/// arithmetic it always did, bit for bit.
#[expect(
    clippy::struct_excessive_bools,
    reason = "six independent per-DOF switches that mirror the GUI toggles"
)]
#[derive(Debug, Clone)]
pub struct BoundedRefineOptions {
    /// Search the X shift (default `true`).
    pub allow_shift_x: bool,
    /// Search the Y shift (default `true`).
    pub allow_shift_y: bool,
    /// Search the uniform scale (default `true`); `false` pins it to the seed's.
    pub allow_scale: bool,
    /// Search the rotation (default `true`); `false` pins it to the seed's.
    pub allow_rotation: bool,
    /// Search the aspect ratio (Y scale relative to X scale); default `false`.
    /// Setting it selects the six-parameter model.
    pub allow_aspect: bool,
    /// Search the shear; default `false`.  Setting it selects the six-parameter
    /// model.
    pub allow_shear: bool,
    /// Half-width of the X-shift interval, in percent of image width.
    pub max_shift_x: f64,
    /// Half-width of the Y-shift interval, in percent of image height.
    pub max_shift_y: f64,
    /// Half-width of the scale interval, in percent.
    pub max_scale: f64,
    /// Half-width of the rotation interval, in degrees.
    pub max_rotation_deg: f64,
    /// Half-width of the aspect interval, in percent (default 10).
    pub max_aspect: f64,
    /// Half-width of the shear interval, in percent (default 10).
    pub max_shear: f64,
    /// Cap on the optimiser iterations of every level of the schedule
    /// ([`level_plan`]: 150 per level, 3 for a large one, plus the polish), each
    /// clamped to it.  Applies to the multi-DOF simplex only; the single-DOF
    /// golden-section search stops at `GOLDEN_TOL` instead.
    pub max_iterations: usize,
    /// Iterations of the full-resolution polish that follows the coarse levels
    /// (default [`DEFAULT_POLISH_ITERATIONS`]); `0` is GitHub's schedule exactly.
    /// The single-DOF golden-section search polishes inside a narrow bracket
    /// around the carried value instead.
    pub polish_iterations: usize,
    /// Evaluate the objective on this device where it accepts a level (default:
    /// none, the CPU); see [`crate::device`].
    pub device: Option<DeviceHandle>,
}

impl Default for BoundedRefineOptions {
    fn default() -> Self {
        Self {
            allow_shift_x: true,
            allow_shift_y: true,
            allow_scale: true,
            allow_rotation: true,
            allow_aspect: false,
            allow_shear: false,
            max_shift_x: 5.0,
            max_shift_y: 5.0,
            max_scale: 5.0,
            max_rotation_deg: 5.0,
            max_aspect: 10.0,
            max_shear: 10.0,
            max_iterations: NM_MAX_ITER,
            polish_iterations: DEFAULT_POLISH_ITERATIONS,
            device: None,
        }
    }
}

/// Half-width of the shift and scale windows (percent) of a search that is not
/// bounded: GitHub z-stackr's windows. A focus series with strong breathing
/// changes scale by well over the bounded search's 5 % between its first and
/// last frame (13.5 % on the sample series), all of it relative to the
/// identity seed.
pub const WIDE_WINDOW_PERCENT: f64 = 20.0;

/// Half-width of the rotation window (degrees) of a search that is not bounded.
pub const WIDE_ROTATION_DEG: f64 = 20.0;

impl BoundedRefineOptions {
    /// The defaults with GitHub's wide windows: shift and scale
    /// ±[`WIDE_WINDOW_PERCENT`], rotation ±[`WIDE_ROTATION_DEG`]; aspect and
    /// shear keep their ±10 %. What Lucas-Kanade and the six-parameter simplex
    /// search when bounded search is off
    /// ([`MotionRefine::bounded_options`]).
    #[must_use]
    pub fn wide() -> Self {
        Self {
            max_shift_x: WIDE_WINDOW_PERCENT,
            max_shift_y: WIDE_WINDOW_PERCENT,
            max_scale: WIDE_WINDOW_PERCENT,
            max_rotation_deg: WIDE_ROTATION_DEG,
            ..Self::default()
        }
    }

    /// `true` when the aspect or the shear is searched, i.e. the six-parameter
    /// model applies.
    #[must_use]
    pub const fn uses_affine(&self) -> bool {
        self.allow_aspect || self.allow_shear
    }
}

/// The seed parameters of a bounded search: the exact six-parameter
/// decomposition of `initial` when `opts` searches the aspect or the shear,
/// otherwise the nearest similarity ([`matrix_to_params`]) as always.
fn seed_params(
    initial: &Matrix3<f32>,
    width: usize,
    height: usize,
    opts: &BoundedRefineOptions,
) -> AffineParams {
    if opts.uses_affine() {
        matrix_to_affine_params(initial, width, height)
    } else {
        AffineParams::from(matrix_to_params(initial, width, height))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dof {
    Rotation,
    Scale,
    ShiftX,
    ShiftY,
    Aspect,
    Shear,
}

/// Bounded search-space mapper: forward (parameter → logit) and inverse
/// (logit → parameter) transforms for the active DOFs.  Active DOFs are
/// ordered rotation, scale, shiftX, shiftY, aspect, shear; the first four keep
/// the order of the four-parameter model, so a search without aspect and shear
/// is unchanged.
struct SearchBounds {
    active: Vec<Dof>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    base: AffineParams,
}

impl SearchBounds {
    fn new(p0: &AffineParams, o: &BoundedRefineOptions) -> Self {
        let mut active = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        if o.allow_rotation {
            let d = o.max_rotation_deg * std::f64::consts::PI / 180.0;
            active.push(Dof::Rotation);
            lower.push(p0.rotate - d);
            upper.push(p0.rotate + d);
        }
        if o.allow_scale {
            let d = o.max_scale / 100.0;
            active.push(Dof::Scale);
            lower.push(p0.scale - d);
            upper.push(p0.scale + d);
        }
        if o.allow_shift_x {
            let d = o.max_shift_x / 100.0;
            active.push(Dof::ShiftX);
            lower.push(p0.tx - d);
            upper.push(p0.tx + d);
        }
        if o.allow_shift_y {
            let d = o.max_shift_y / 100.0;
            active.push(Dof::ShiftY);
            lower.push(p0.ty - d);
            upper.push(p0.ty + d);
        }
        if o.allow_aspect {
            let d = o.max_aspect / 100.0;
            active.push(Dof::Aspect);
            lower.push(p0.aspect - d);
            upper.push(p0.aspect + d);
        }
        if o.allow_shear {
            let d = o.max_shear / 100.0;
            active.push(Dof::Shear);
            lower.push(p0.shear - d);
            upper.push(p0.shear + d);
        }
        Self {
            active,
            lower,
            upper,
            base: *p0,
        }
    }

    const fn n(&self) -> usize {
        self.active.len()
    }

    const fn dof_value(p: &AffineParams, dof: Dof) -> f64 {
        match dof {
            Dof::Rotation => p.rotate,
            Dof::Scale => p.scale,
            Dof::ShiftX => p.tx,
            Dof::ShiftY => p.ty,
            Dof::Aspect => p.aspect,
            Dof::Shear => p.shear,
        }
    }

    /// Forward map params → unbounded logit search coordinates.
    fn to_search_space(&self, p: &AffineParams) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.n()];
        for (i, &dof) in self.active.iter().enumerate() {
            let v = Self::dof_value(p, dof);
            let (lo, hi) = (self.lower[i], self.upper[i]);
            out[i] = if v <= lo {
                -100.0
            } else if v >= hi {
                100.0
            } else {
                let n = (v - lo) / (hi - lo);
                (n / (1.0 - n)).ln()
            };
        }
        out
    }

    /// Inverse map logit search coordinates → params (inactive DOFs keep base).
    fn to_param_space(&self, coords: &[f64]) -> AffineParams {
        let mut p = self.base;
        for (i, &dof) in self.active.iter().enumerate() {
            let c = coords[i].clamp(-100.0, 100.0);
            let v = 1.0 / (1.0 + (-c).exp()) * (self.upper[i] - self.lower[i]) + self.lower[i];
            match dof {
                Dof::Rotation => p.rotate = v,
                Dof::Scale => p.scale = v,
                Dof::ShiftX => p.tx = v,
                Dof::ShiftY => p.ty = v,
                Dof::Aspect => p.aspect = v,
                Dof::Shear => p.shear = v,
            }
        }
        p
    }

    /// Initial simplex steps of the polish, in logit coordinates.
    ///
    /// Each active DOF gets the step that moves the transform by about
    /// `residual_px` full-resolution pixels (shifts: over the image size; the
    /// other DOFs: over the half diagonal), converted to logit space with the
    /// local slope of the sigmoid at `coords`, and capped at the default
    /// [`BOUNDED_STEP`] so the polish simplex is never larger than a regular one.
    pub(crate) fn polish_steps(
        &self,
        coords: &[f64],
        residual_px: f64,
        width: usize,
        height: usize,
    ) -> Vec<f64> {
        let (w, h) = (width as f64, height as f64);
        let half_diagonal = 0.5 * w.hypot(h);
        self.active
            .iter()
            .enumerate()
            .map(|(i, &dof)| {
                let wanted = match dof {
                    Dof::ShiftX => residual_px / w,
                    Dof::ShiftY => residual_px / h,
                    Dof::Rotation | Dof::Scale | Dof::Aspect | Dof::Shear => {
                        residual_px / half_diagonal
                    }
                };
                let n = 1.0 / (1.0 + (-coords[i].clamp(-100.0, 100.0)).exp());
                let slope = (self.upper[i] - self.lower[i]) * n * (1.0 - n);
                let step = wanted / slope;
                if step.is_finite() {
                    step.min(BOUNDED_STEP)
                } else {
                    BOUNDED_STEP
                }
            })
            .collect()
    }

    /// The bracket of the single-DOF polish: the carried coordinate plus or minus
    /// four polish steps (the expected residual with a margin of four), inside
    /// the full bracket.
    pub(crate) fn polish_bracket(
        &self,
        coords: &[f64],
        residual_px: f64,
        width: usize,
        height: usize,
    ) -> (f64, f64) {
        let step = self
            .polish_steps(coords, residual_px, width, height)
            .first()
            .copied()
            .unwrap_or(BOUNDED_STEP);
        let half = (4.0 * step).min(GOLDEN_BRACKET);
        let centre = coords
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(-GOLDEN_BRACKET, GOLDEN_BRACKET);
        (centre - half, centre + half)
    }
}

/// Objective in bounded search space: map coords → params → warp ref → RMS vs src.
fn objective_bounded(coords: &[f64], bounds: &SearchBounds, objective: &LevelObjective) -> f64 {
    if !coords.iter().all(|c| c.is_finite()) {
        return PENALTY;
    }
    objective.eval_affine(&bounds.to_param_space(coords))
}

/// The unbounded 4-DOF minimisation of one pyramid level, run on the shared
/// simplex ([`nelder_mead_generic`]).  A zero `step` pins that parameter.
fn minimize_4d(
    objective: &LevelObjective,
    x0: [f64; 4],
    step: &[f64; 4],
    max_iter: usize,
) -> [f64; 4] {
    let f = |v: &[f64]| objective.eval(&[v[0], v[1], v[2], v[3]]);
    let best = nelder_mead_generic(&f, &x0, step, max_iter, NM_TOL);
    [best[0], best[1], best[2], best[3]]
}

/// Nelder-Mead (downhill simplex) minimiser over an arbitrary number of
/// dimensions; the only simplex implementation in the crate (the unbounded 4-DOF
/// path and the bounded logit-space path both run on it).
///
/// Implements the classical Nelder & Mead (1965) algorithm with the standard
/// coefficients (reflection α=1, expansion γ=2, contraction ρ=0.5, shrink σ=0.5)
/// and the textbook acceptance rules (Lagarias et al. 1998): the reflected point
/// replaces the worst vertex when it beats the second-worst; an *outside*
/// contraction is accepted only when it is at least as good as the reflected
/// point, an *inside* contraction when it beats the worst vertex; otherwise the
/// simplex shrinks towards the best vertex.
///
/// The initial simplex is `x0` plus one vertex per axis offset by `step[i]`
/// (a zero step duplicates `x0` and so pins that axis).  Iterates until the
/// spread of the function values drops below `tol` or `max_iter` iterations ran,
/// and returns the best vertex.
fn nelder_mead_generic<F: Fn(&[f64]) -> f64>(
    f: &F,
    x0: &[f64],
    step: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = x0.len();
    debug_assert_eq!(step.len(), n, "one initial step per dimension");
    let (alpha, gamma, rho, sigma) = (1.0_f64, 2.0_f64, 0.5_f64, 0.5_f64);

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for (i, &s) in step.iter().enumerate() {
        let mut v = x0.to_vec();
        v[i] += s;
        simplex.push(v);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_unstable_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        let f_spread = fvals[n] - fvals[0];
        if f_spread < tol && f_spread >= 0.0 {
            break;
        }

        let mut centroid = vec![0.0_f64; n];
        for v in &simplex[..n] {
            for (cj, vj) in centroid.iter_mut().zip(v.iter()) {
                *cj += vj;
            }
        }
        for cj in &mut centroid {
            *cj /= n as f64;
        }

        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = f(&xr);

        if fr < fvals[0] {
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (xr[j] - centroid[j]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Outside contraction when the reflected point beats the worst
            // vertex (accepted if it is no worse than the reflected point),
            // inside contraction otherwise (accepted if it beats the worst).
            let outside = fr < fvals[n];
            let xc: Vec<f64> = if outside {
                (0..n)
                    .map(|j| centroid[j] + rho * (xr[j] - centroid[j]))
                    .collect()
            } else {
                (0..n)
                    .map(|j| centroid[j] - rho * (centroid[j] - simplex[n][j]))
                    .collect()
            };
            let fc = f(&xc);
            let accepted = if outside { fc <= fr } else { fc < fvals[n] };
            if accepted {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                let x0_best = simplex[0].clone();
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = x0_best[j] + sigma * (simplex[i][j] - x0_best[j]);
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_unstable_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    simplex[order[0]].clone()
}

/// Golden-section minimiser for the single-DOF bounded case.
///
/// A plain bracketing search (interval shrinks by 0.618 per evaluation, no
/// parabolic interpolation), so it is **not** Brent's method; it needs the
/// objective to be unimodal on `[a, b]` and returns when the interval is
/// narrower than `tol` or after 200 iterations.  The bounded refiners call it
/// with the logit bracket `[-GOLDEN_BRACKET, GOLDEN_BRACKET]` at every pyramid
/// level, ignoring the coordinate carried over from the coarser level.  The
/// bracket must stay inside the part of the logit axis where the sigmoid still
/// moves: with `[-100, 100]` both initial probes sat in the saturated tails,
/// evaluated to the same value, and the tie rule walked the interval to the
/// band edge, so a single-DOF fit always returned the bound (ALN-9).
fn golden_section_min<F: Fn(f64) -> f64>(mut a: f64, mut b: f64, tol: f64, f: &F) -> f64 {
    const R: f64 = 0.618_033_988_749_895;
    const C: f64 = 1.0 - R;
    let mut x1 = a + C * (b - a);
    let mut x2 = a + R * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..200 {
        if (b - a).abs() < tol {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + C * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + R * (b - a);
            f2 = f(x2);
        }
    }
    if f1 < f2 { x1 } else { x2 }
}

/// Objective for the faithful single-resolution registration path.
///
/// Warps the reference luma into the source frame via the inverse transform
/// using the edge-clamped 4-tap spline sampler, then returns the
/// mean-corrected RMS difference against the source luma. Unlike
/// [`objective_at_level`] this uses edge-clamped (not zero-fill) borders, so
/// the warped border does not bias the fit.
#[cfg(test)]
fn objective_registration(
    params_vec: &[f64; 4],
    ref_luma: &[f32],
    src_luma: &[f32],
    width: usize,
    height: usize,
) -> f64 {
    LevelObjective::new(
        ref_luma,
        src_luma,
        width,
        height,
        Sampler::SplineClamped,
        None,
    )
    .eval(params_vec)
}

/// Bounded intensity registration with the edge-clamped spline sampler.
///
/// The pyramid search of [`refine_alignment_bounded`] (same coarse-biased schedule,
/// restarts and polish, see [`level_plan`]) with a different objective sampler.
///
/// It used to run a single simplex of up to 400 iterations at full resolution;
/// it is now a pyramid user like the others.  This form builds the reference
/// pyramid itself; a caller that registers a stack should build
/// `LumaPyramid::new(reference)` once and use
/// [`refine_alignment_registration_with_pyramid`].
///
/// # Errors
/// Returns [`StackerError`] if `initial` is non-finite, the image dimensions
/// disagree, or the optimiser produces non-finite / degenerate parameters.
pub fn refine_alignment_registration(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    refine_alignment_registration_with_pyramid(
        reference,
        &LumaPyramid::new(reference),
        source,
        initial,
        opts,
    )
}

/// [`refine_alignment_registration`] with the reference's luma pyramid supplied
/// by the caller (build it once per stack); the result is identical.
///
/// # Errors
/// As [`refine_alignment_registration`], plus [`StackerError::AlignmentFailed`]
/// when `ref_pyramid` was built from an image of a different size.
pub fn refine_alignment_registration_with_pyramid(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    refine_bounded_levels(
        Flavor::Registration,
        reference,
        ref_pyramid,
        source,
        initial,
        opts,
    )
}

/// Multi-DOF multistart of the bounded refiners: `RESTARTS` runs of the simplex
/// from `start` (the first as it is, the others with every coordinate jittered by
/// `rng`), each followed by one objective evaluation to rank it.  The first best
/// wins; the incoming `start` is the fallback.
fn multistart<F: Fn(&[f64]) -> f64>(
    f: &F,
    start: &[f64],
    steps: &[f64],
    max_iter: usize,
    rng: &mut JitterRng,
) -> Vec<f64> {
    let mut best_obj = f64::INFINITY;
    let mut best_coords = start.to_vec();
    for attempt in 0..RESTARTS {
        let from: Vec<f64> = if attempt == 0 {
            start.to_vec()
        } else {
            start.iter().map(|&c| c + rng.jitter()).collect()
        };
        let result = nelder_mead_generic(f, &from, steps, max_iter, NM_TOL);
        let obj = f(&result);
        if obj < best_obj {
            best_obj = obj;
            best_coords = result;
        }
    }
    best_coords
}

/// The shared body of [`refine_alignment_bounded_with_pyramid`] and
/// [`refine_alignment_registration_with_pyramid`]: validation, the bounded
/// search space, and the walk along the level schedule.
fn refine_bounded_levels(
    flavor: Flavor,
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    let who = flavor.who();
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(StackerError::MathError(format!(
            "{who}: initial matrix contains non-finite values"
        )));
    }
    if reference.width != source.width || reference.height != source.height {
        return Err(StackerError::AlignmentFailed(format!(
            "{who}: reference and source dimensions do not match"
        )));
    }
    check_pyramid(who, reference, ref_pyramid)?;

    let w = source.width;
    let h = source.height;
    let p0 = seed_params(initial, w, h, opts);
    let bounds = SearchBounds::new(&p0, opts);

    // No active DOF → nothing to optimise, return the initial transform.
    if bounds.n() == 0 {
        return Ok(*initial);
    }

    let src_pyramid = LumaPyramid::new(source);
    debug_assert_eq!(ref_pyramid.n_levels(), src_pyramid.n_levels());
    let schedule = Schedule::new(ref_pyramid, opts.max_iterations, opts.polish_iterations);

    // Centre of the search space corresponds to the initial estimate.
    let mut coords = bounds.to_search_space(&p0);
    let mut rng = JitterRng::new();

    for planned in &schedule.steps {
        let ref_lvl = ref_pyramid.level(planned.level, &reference.luma);
        let src_lvl = src_pyramid.level(planned.level, &source.luma);

        // One objective per level: owns the warp buffer and the source mean.
        let objective = LevelObjective::new(
            ref_lvl.luma,
            src_lvl.luma,
            ref_lvl.width,
            ref_lvl.height,
            flavor.sampler(),
            opts.device.as_ref(),
        );
        let f = |c: &[f64]| objective_bounded(c, &bounds, &objective);
        let polish = planned.kind == StepKind::Polish;

        coords = if bounds.n() == 1 {
            // Golden section over the whole bracket, or around the carried
            // value for the polish.
            let (lo, hi) = if polish {
                bounds.polish_bracket(&coords, schedule.residual_px, w, h)
            } else {
                (-GOLDEN_BRACKET, GOLDEN_BRACKET)
            };
            vec![golden_section_min(lo, hi, GOLDEN_TOL, &|x| f(&[x]))]
        } else {
            let steps = if polish {
                bounds.polish_steps(&coords, schedule.residual_px, w, h)
            } else {
                vec![BOUNDED_STEP; coords.len()]
            };
            if planned.kind == StepKind::Coarsest {
                multistart(&f, &coords, &steps, planned.iters, &mut rng)
            } else {
                nelder_mead_generic(&f, &coords, &steps, planned.iters, NM_TOL)
            }
        };
    }

    let best_p = bounds.to_param_space(&coords);
    if !best_p.is_finite() {
        // The bounded refiner's message mentions the NaN guard; the
        // registration one never did.
        let note = if flavor == Flavor::Bounded {
            " (NaN encountered)"
        } else {
            ""
        };
        return Err(StackerError::AlignmentFailed(format!(
            "{who}: optimizer produced non-finite parameters{note}"
        )));
    }

    best_p.to_matrix(w, h).ok_or_else(|| {
        StackerError::AlignmentFailed(format!(
            "{who}: refined parameters produced a degenerate matrix"
        ))
    })
}

/// Refine a coarse alignment matrix using the bounded optimizer.
///
/// Builds the logit-bounded search space from the initial estimate, then runs a
/// coarse-to-fine pyramid optimisation in it (golden-section for a single
/// active DOF, Nelder-Mead otherwise) along the coarse-biased schedule of
/// [`level_plan`]: ten restarts at the coarsest level, 150 iterations per level,
/// then a short full-resolution polish ([`BoundedRefineOptions::polish_iterations`]).
/// Parameters are guaranteed to stay within `[lower, upper]`, preventing
/// degenerate transforms.
///
/// ## Errors
/// - [`StackerError::MathError`] — `initial` contains non-finite values.
/// - [`StackerError::AlignmentFailed`] — dimension mismatch, or the optimiser
///   produced non-finite parameters / a degenerate matrix.
pub fn refine_alignment_bounded(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    refine_alignment_bounded_with_pyramid(
        reference,
        &LumaPyramid::new(reference),
        source,
        initial,
        opts,
    )
}

/// [`refine_alignment_bounded`] with the reference's luma pyramid supplied by
/// the caller; see [`refine_alignment_with_pyramid`].  The result is identical
/// to [`refine_alignment_bounded`].
///
/// ## Errors
/// As [`refine_alignment_bounded`], plus [`StackerError::AlignmentFailed`] when
/// `ref_pyramid` was built from an image of a different size.
pub fn refine_alignment_bounded_with_pyramid(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    refine_bounded_levels(
        Flavor::Bounded,
        reference,
        ref_pyramid,
        source,
        initial,
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::warp_image_clamped;
    use nalgebra::Matrix3;

    /// Build a synthetic 64×64 `PlanarImage<f32>` whose luma carries a
    /// deterministic gradient pattern.
    fn make_gradient_image(width: usize, height: usize) -> PlanarImage<f32> {
        let mut img = PlanarImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v =
                    ((x as f32 / width as f32) * 0.7 + (y as f32 / height as f32) * 0.3).min(1.0);
                img.luma[y * width + x] = v;
                img.chroma_a[y * width + x] = v * 0.5;
                img.chroma_b[y * width + x] = v * 0.25;
            }
        }
        img
    }

    /// PERF-1: the warp buffer is reused across evaluations; a later evaluation
    /// must not see anything left behind by an earlier one.
    #[test]
    fn reused_scratch_buffer_does_not_leak_between_evaluations() {
        for &(w, h) in &[(48_usize, 40_usize), (160, 128)] {
            let reference = make_gradient_image(w, h);
            let mut source = make_gradient_image(w, h);
            for (i, v) in source.luma.iter_mut().enumerate() {
                *v = (*v + ((i * 13) % 11) as f32 * 0.01).min(1.0);
            }
            let objective =
                LevelObjective::new(&reference.luma, &source.luma, w, h, Sampler::Bicubic, None);
            let near = [0.001, -0.002, 1.0, 0.0];
            let far = [0.04, 0.03, 1.02, 0.01];
            let first = objective.eval(&near);
            let mid = objective.eval(&far);
            let again = objective.eval(&near);
            let fresh = objective_at_level(&near, &reference.luma, &source.luma, w, h);
            assert_ne!(
                first.to_bits(),
                mid.to_bits(),
                "{w}x{h}: probe points must differ"
            );
            // The row partials are added in a fixed order, so both the serial
            // and the parallel level are reproducible; the few-ulp slack on the
            // parallel level is kept from before the reduction was made ordered.
            let tol = if w * h < SERIAL_LEVEL_PIXELS {
                0.0
            } else {
                1.0e-12
            };
            assert!((first - again).abs() <= tol, "{w}x{h}: {first} vs {again}");
            assert!((first - fresh).abs() <= tol, "{w}x{h}: {first} vs {fresh}");
        }
    }

    /// PERF-1: a non-invertible transform returns the penalty and leaves the
    /// buffer usable for the next evaluation.
    #[test]
    fn penalty_evaluation_does_not_disturb_the_buffer() {
        let (w, h) = (40_usize, 40_usize);
        let reference = make_gradient_image(w, h);
        let source = make_gradient_image(w, h);
        let objective = LevelObjective::new(
            &reference.luma,
            &source.luma,
            w,
            h,
            Sampler::SplineClamped,
            None,
        );
        let ok = objective.eval(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(
            objective.eval(&[f64::NAN, 0.0, 1.0, 0.0]).to_bits(),
            PENALTY.to_bits()
        );
        assert_eq!(
            objective.eval(&[0.0, 0.0, 0.0, 0.0]).to_bits(),
            PENALTY.to_bits()
        );
        assert_eq!(
            objective.eval(&[0.0, 0.0, 1.0, 0.0]).to_bits(),
            ok.to_bits()
        );
    }

    /// The plan as `(level, iters, kind)` triples.
    fn triples(plan: &[LevelStep]) -> Vec<(usize, usize, StepKind)> {
        plan.iter().map(|s| (s.level, s.iters, s.kind)).collect()
    }

    /// ALN-3, now for the whole schedule: no level exceeds the caller's cap.
    #[test]
    fn level_plan_never_exceeds_the_cap() {
        for cap in [0_usize, 1, 3, 10, 150, 400] {
            for sides in [
                vec![62, 125, 250, 500, 1000, 2000, 4000],
                vec![32, 64, 128],
                vec![50, 100, 200, 400, 800, 1600],
                vec![40],
            ] {
                for polish in [0_usize, 5, 40, 1000] {
                    for step in level_plan(&sides, cap, polish) {
                        assert!(
                            step.iters <= cap,
                            "{sides:?} cap {cap} polish {polish}: {step:?}"
                        );
                    }
                }
            }
        }
    }

    /// The 24 MP rule: 62 .. 500 px at 150 iterations, the large 1000 px level
    /// at 3, the 2000 px level (large, among the finest two) not processed, full
    /// resolution only through the polish.
    #[test]
    fn level_plan_for_a_24_megapixel_pyramid() {
        let sides = [62, 125, 250, 500, 1000, 2000, 4000];
        assert_eq!(
            triples(&level_plan(&sides, 400, 0)),
            vec![
                (0, 150, StepKind::Coarsest),
                (1, 150, StepKind::Refine),
                (2, 150, StepKind::Refine),
                (3, 150, StepKind::Refine),
                (4, 3, StepKind::Refine),
            ],
            "polish off is GitHub's schedule"
        );
        let polished = level_plan(&sides, 400, 40);
        assert_eq!(polished.len(), 6);
        assert_eq!(
            polished.last().map(|s| (s.level, s.iters, s.kind)),
            Some((6, 40, StepKind::Polish))
        );
        assert_eq!(polished[..5], level_plan(&sides, 400, 0)[..]);
    }

    /// Mid-size and small pyramids: nothing is large below 512 px, and the full
    /// resolution level is never a regular level.
    #[test]
    fn level_plan_for_small_pyramids() {
        // 1600 px short side: the 800 px level is large and among the finest two.
        assert_eq!(
            triples(&level_plan(&[50, 100, 200, 400, 800, 1600], 400, 40)),
            vec![
                (0, 150, StepKind::Coarsest),
                (1, 150, StepKind::Refine),
                (2, 150, StepKind::Refine),
                (3, 150, StepKind::Refine),
                (5, 40, StepKind::Polish),
            ]
        );
        // 128 px short side: every coarse level is regular, full resolution is
        // only the polish.
        assert_eq!(
            triples(&level_plan(&[32, 64, 128], 400, 40)),
            vec![
                (0, 150, StepKind::Coarsest),
                (1, 150, StepKind::Refine),
                (2, 40, StepKind::Polish),
            ]
        );
        // 600 px: the 300 px level is not large; 600 is full resolution.
        assert_eq!(
            triples(&level_plan(&[37, 75, 150, 300, 600], 400, 0)),
            vec![
                (0, 150, StepKind::Coarsest),
                (1, 150, StepKind::Refine),
                (2, 150, StepKind::Refine),
                (3, 150, StepKind::Refine),
            ]
        );
        // A pyramid of one level (image below 64 px): GitHub would do nothing; here
        // the level is processed as the coarsest, with or without a polish.
        for polish in [0, 40] {
            assert_eq!(
                triples(&level_plan(&[40], 400, polish)),
                vec![(0, 150, StepKind::Coarsest)]
            );
        }
        assert_eq!(triples(&level_plan(&[], 400, 40)), vec![]);
    }

    /// The residual the polish is sized to is a quarter of the factor between
    /// full resolution and the last coarse level.
    #[test]
    fn polish_residual_is_a_quarter_of_the_last_level_factor() {
        let big = PlanarImage::<f32>::new(1600, 1600);
        let s = Schedule::new(&LumaPyramid::new(&big), 400, 40);
        // Last coarse level 400 px: factor 4, residual 1 px.
        assert!((s.residual_px - 1.0).abs() < 1.0e-12, "{}", s.residual_px);
        let none = Schedule::new(&LumaPyramid::new(&big), 400, 0);
        assert!(none.residual_px.abs() < f64::EPSILON);

        // Polish steps never exceed the default simplex and keep pinned axes pinned.
        let default = [NM_STEP_TX, NM_STEP_TY, NM_STEP_SCALE, 0.0];
        let polished = polish_steps_4d(&default, 1.0, 1600, 1600);
        for (p, d) in polished.iter().zip(default) {
            assert!(*p <= d, "{p} > {d}");
        }
        assert!((polished[0] - 1.0 / 1600.0).abs() < 1.0e-15);
        assert!(polished[3].abs() < f64::EPSILON, "pinned axis stays pinned");
    }

    /// The restart jitter is the same on every run and stays within ±0.0005.
    #[test]
    fn jitter_is_deterministic_and_bounded() {
        let draw = || {
            let mut rng = JitterRng::new();
            (0..1000).map(|_| rng.jitter()).collect::<Vec<_>>()
        };
        let a = draw();
        assert_eq!(a, draw());
        assert!(a.iter().all(|j| j.abs() <= 0.000_5));
        assert!(a.iter().any(|j| *j > 0.000_2) && a.iter().any(|j| *j < -0.000_2));
    }

    /// ALN-3: the simplex honours its iteration cap (evaluation count bounded
    /// by the initial simplex plus at most `n + 2` evaluations per iteration).
    #[test]
    fn nelder_mead_generic_stops_at_the_iteration_cap() {
        use std::cell::Cell;

        let rosenbrock = |v: &[f64]| {
            let (x, y) = (v[0], v[1]);
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };
        let count = |max_iter: usize| {
            let evals = Cell::new(0_usize);
            let f = |v: &[f64]| {
                evals.set(evals.get() + 1);
                rosenbrock(v)
            };
            let _ = nelder_mead_generic(&f, &[-1.2, 1.0], &[0.5, 0.5], max_iter, 0.0);
            evals.get()
        };
        let n = 2_usize;
        let capped = count(10);
        assert!(
            capped <= (n + 1) + 10 * (n + 2),
            "10 iterations used {capped} evaluations"
        );
        assert!(count(100) > capped, "a larger cap must allow more work");
    }

    /// ALN-10: the single simplex implementation finds the minimum of known
    /// functions, including the zero-step (pinned axis) form the 4-DOF path uses.
    #[test]
    fn nelder_mead_generic_finds_known_minima() {
        let bowl =
            |v: &[f64]| (v[0] - 3.0).powi(2) + 2.0 * (v[1] + 1.0).powi(2) + (v[2] - 0.5).powi(2);
        let best = nelder_mead_generic(&bowl, &[0.0, 0.0, 0.0], &[0.5, 0.5, 0.5], 600, 1.0e-14);
        for (got, want) in best.iter().zip([3.0, -1.0, 0.5]) {
            assert!((got - want).abs() < 1.0e-3, "bowl minimum {best:?}");
        }

        let rosenbrock = |v: &[f64]| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2);
        let best = nelder_mead_generic(&rosenbrock, &[-1.2, 1.0], &[0.5, 0.5], 800, 1.0e-14);
        assert!(rosenbrock(&best) < 1.0e-3, "Rosenbrock minimum {best:?}");

        // A zero step pins its axis at the start value.
        let best = nelder_mead_generic(&bowl, &[0.0, 0.0, 0.0], &[0.5, 0.0, 0.5], 600, 1.0e-14);
        assert!(best[1].abs() < 1.0e-12, "pinned axis moved: {best:?}");
        assert!((best[0] - 3.0).abs() < 1.0e-3, "free axis: {best:?}");
    }

    /// ALN-9: the single-DOF search is golden-section and finds the minimum of a
    /// unimodal function inside the full logit bracket.
    #[test]
    fn golden_section_min_finds_the_minimum_of_a_unimodal_function() {
        let x = golden_section_min(-100.0, 100.0, GOLDEN_TOL, &|x| (x - 1.3).powi(2));
        assert!((x - 1.3).abs() < 1.0e-4, "minimum at {x}");
        let x = golden_section_min(-100.0, 100.0, GOLDEN_TOL, &|x| (x + 42.0).abs());
        assert!((x + 42.0).abs() < 1.0e-4, "minimum at {x}");
    }

    /// `refine_alignment_registration` should return `Ok` when given a source
    /// that was produced by warping the reference with a small known translation.
    /// The recovered transform should reduce the RMS residual compared to the
    /// identity initialisation.
    #[test]
    fn registration_reduces_rms_after_small_shift() {
        let w = 64_usize;
        let h = 64_usize;
        let reference = make_gradient_image(w, h);

        // Produce source by applying a small known shift (tx=0.03, ty=0.02).
        let known_p = RegistrationParams {
            tx: 0.03,
            ty: 0.02,
            scale: 1.0,
            rotate: 0.0,
        };
        let known_matrix =
            params_to_matrix(&known_p, w, h).expect("known params should produce a valid matrix");
        let source = warp_image_clamped(&reference, &known_matrix)
            .expect("warp_image_clamped should succeed for a valid matrix");

        // Identity RMS (no alignment correction).
        let id_p = RegistrationParams::identity();
        let id_vec = id_p.to_vec();
        let identity_rms = objective_registration(&id_vec, &reference.luma, &source.luma, w, h);

        // Refine from identity.
        let opts = BoundedRefineOptions {
            max_iterations: 200,
            ..BoundedRefineOptions::default()
        };
        let refined =
            refine_alignment_registration(&reference, &source, &Matrix3::identity(), &opts)
                .expect("refine_alignment_registration should return Ok");

        // Compute RMS with the refined transform.
        let refined_p = matrix_to_params(&refined, w, h);
        let refined_rms =
            objective_registration(&refined_p.to_vec(), &reference.luma, &source.luma, w, h);

        assert!(
            refined_rms < identity_rms,
            "refined RMS ({refined_rms:.6}) should be less than identity RMS ({identity_rms:.6})"
        );
    }
}
