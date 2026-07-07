use crate::transform::spline4x4_sample_clamped;
use nalgebra::Matrix3;
use rayon::prelude::*;
use stacker_core::{error::StackerError, image::PlanarImage};

use super::{
    params::{RegistrationParams, matrix_to_params, params_to_matrix},
    pyramid::downsample_luma,
    simplex::{NM_MAX_ITER, NM_TOL, brent_min, nelder_mead_generic},
};

/// Large finite penalty returned when parameters or projected coords are
/// non-finite.  Must be larger than any realistic RMS value (pixel values
/// are in [0,1] so RMS ≤ 1).
const PENALTY: f64 = 1.0e6;

/// Short-side pixel threshold above which a pyramid level is considered
/// "large".
///
/// Shared with [`super::lucas_kanade`]: both optimisers use the same
/// coarse-biased schedule (never refine full resolution; stop early at a
/// large finest-two level; cap iterations on any other large level), so this
/// threshold is `pub(super)` rather than duplicated as a second magic number
/// in the sibling module.
pub(super) const REF_SHORT_SIDE_THRESHOLD: usize = 512;

/// Optimiser iterations for a "large" (short side > threshold) level.
///
/// Shared with [`super::lucas_kanade`] — see
/// [`REF_SHORT_SIDE_THRESHOLD`]'s doc comment.
pub(super) const REF_LARGE_LEVEL_ITERS: usize = 3;

/// Optimiser iterations for a normal-sized level.
const REF_DEFAULT_LEVEL_ITERS: usize = 150;

/// Compute mean-bias-corrected RMS difference between two luma planes.
///
/// ```text
/// diff_i = (src[i] − mean_src) − (warped_ref[i] − mean_ref)
/// RMS = sqrt( sum(diff_i²) / (N − 1) )
/// ```
///
/// Subtracting the per-image mean makes the objective invariant to global
/// brightness offset — useful when breathing correction changes apparent
/// exposure.  Parallelised with rayon over rows.
pub fn rms_difference(src: &[f32], warped_ref: &[f32], width: usize, height: usize) -> f64 {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(warped_ref.len(), width * height);
    let n = width * height;
    if n == 0 {
        return 0.0;
    }

    let (sum_src, sum_ref): (f64, f64) = src
        .par_chunks(width)
        .zip(warped_ref.par_chunks(width))
        .map(|(row_s, row_r)| {
            let s: f64 = row_s.iter().map(|&v| f64::from(v)).sum();
            let r: f64 = row_r.iter().map(|&v| f64::from(v)).sum();
            (s, r)
        })
        .reduce(|| (0.0, 0.0), |(a0, b0), (a1, b1)| (a0 + a1, b0 + b1));

    let mean_src = sum_src / n as f64;
    let mean_ref = sum_ref / n as f64;

    let var_sum: f64 = src
        .par_chunks(width)
        .zip(warped_ref.par_chunks(width))
        .map(|(row_s, row_r)| {
            let mut acc = 0.0_f64;
            for (&vs, &vr) in row_s.iter().zip(row_r.iter()) {
                let d = (f64::from(vs) - mean_src) - (f64::from(vr) - mean_ref);
                acc += d * d;
            }
            acc
        })
        .sum();

    (var_sum / (n as f64 - 1.0).max(1.0)).sqrt()
}

/// Brent convergence tolerance for the single-DOF case.
const BRENT_TOL: f64 = 1.0e-6;
/// Initial simplex step in the (logit) search space for the multi-DOF case.
const BOUNDED_STEP: f64 = 0.5;

/// Per-DOF search configuration: which degrees of freedom are active and their
/// allowed half-width intervals (rotation, scale, shift X/Y, aspect, shear).
#[derive(Debug, Clone)]
// Each bool independently toggles one DOF's participation in the search;
// this is a flat options struct, not state-machine-shaped, so splitting it
// into enums would be a signature/behaviour-affecting refactor, not a lint fix.
#[allow(clippy::struct_excessive_bools)]
pub struct BoundedRefineOptions {
    pub allow_shift_x: bool,
    pub allow_shift_y: bool,
    pub allow_scale: bool,
    pub allow_rotation: bool,
    /// If `true`, optimise the aspect DOF (ratio of Y-scale to X-scale).
    /// Default `false` so `Registration`/`Translation` stay a pure 4-DOF
    /// similarity model. Only `Affine` sets this.
    pub allow_aspect: bool,
    /// If `true`, optimise the shear DOF (X-shear factor). Default `false`
    /// — see [`Self::allow_aspect`].
    pub allow_shear: bool,
    /// Half-width of the X-shift interval, in percent of image width.
    pub max_shift_x: f64,
    /// Half-width of the Y-shift interval, in percent of image height.
    pub max_shift_y: f64,
    /// Half-width of the scale interval, in percent.
    pub max_scale: f64,
    /// Half-width of the rotation interval, in degrees.
    pub max_rotation_deg: f64,
    /// Half-width of the aspect interval, in percent (around the seed's
    /// aspect value). Default 10.0.
    pub max_aspect: f64,
    /// Half-width of the shear interval, in percent (around the seed's
    /// shear value). Default 10.0.
    pub max_shear: f64,
    /// Max optimiser iterations at the finest pyramid level.
    pub max_iterations: usize,
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
            max_shift_x: 20.0,
            max_shift_y: 20.0,
            max_scale: 20.0,
            max_rotation_deg: 20.0,
            max_aspect: 10.0,
            max_shear: 10.0,
            max_iterations: NM_MAX_ITER,
        }
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
/// ordered rotation, scale, shiftX, shiftY, aspect, shear.
struct SearchBounds {
    active: Vec<Dof>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    base: RegistrationParams,
}

impl SearchBounds {
    fn new(p0: &RegistrationParams, o: &BoundedRefineOptions) -> Self {
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

    const fn dof_value(p: &RegistrationParams, dof: Dof) -> f64 {
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
    fn to_search_space(&self, p: &RegistrationParams) -> Vec<f64> {
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
    fn to_param_space(&self, coords: &[f64]) -> RegistrationParams {
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
}

/// Objective for the faithful multi-scale registration path.
///
/// Warps the reference luma into the source frame via the inverse transform
/// using the edge-clamped 4-tap spline sampler, then returns the
/// mean-corrected RMS difference against the source luma. Uses edge-clamped
/// (not zero-fill) borders, so the warped border does not bias the fit.
///
/// `pub(super)` (rather than private) so the sibling `lucas_kanade` module
/// can reuse this exact warp+RMS objective for its own RMS-vs-starting-RMS
/// regression check, instead of forking a third copy of the warp/RMS logic.
pub(super) fn objective_registration(
    params_vec: &[f64; 6],
    ref_luma: &[f32],
    src_luma: &[f32],
    width: usize,
    height: usize,
) -> f64 {
    let p = RegistrationParams::from_vec(params_vec);
    if !p.is_finite() {
        return PENALTY;
    }
    let Some(fwd) = params_to_matrix(&p, width, height) else {
        return PENALTY;
    };
    let Some(inv) = fwd.try_inverse() else {
        return PENALTY;
    };
    if !inv.iter().all(|v| v.is_finite()) {
        return PENALTY;
    }

    let mut warped_luma = vec![0.0_f32; width * height];
    warped_luma
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, dst) in row.iter_mut().enumerate() {
                let p_dst = nalgebra::Vector3::new(x as f32, y as f32, 1.0_f32);
                let p_src = fwd * p_dst;
                let denom = p_src[2];
                if denom.abs() < 1.0e-8 {
                    *dst = 0.0;
                    continue;
                }
                let sx = p_src[0] / denom;
                let sy = p_src[1] / denom;
                *dst = spline4x4_sample_clamped(ref_luma, width, height, sx, sy);
            }
        });

    rms_difference(src_luma, &warped_luma, width, height)
}

/// [`objective_registration`] evaluated in the bounded logit search space.
fn objective_registration_bounded(
    coords: &[f64],
    bounds: &SearchBounds,
    ref_luma: &[f32],
    src_luma: &[f32],
    width: usize,
    height: usize,
) -> f64 {
    if !coords.iter().all(|c| c.is_finite()) {
        return PENALTY;
    }
    let p = bounds.to_param_space(coords);
    objective_registration(&p.to_vec(), ref_luma, src_luma, width, height)
}

/// Evaluate the [`objective_registration`] RMS for `matrix` at a downsampled
/// resolution, without needing a full-size warp.
///
/// `matrix` is a full-resolution forward transform (dimensions `full_width` ×
/// `full_height`). Because [`RegistrationParams`]' `tx`/`ty` are *fractional*
/// (pixels / image dimension) and `scale`/`rotate`/`aspect`/`shear` are all
/// resolution-independent (aspect and shear are dimensionless ratios, same
/// as scale), decomposing `matrix` at the full resolution and rebuilding it
/// at the small resolution reproduces the same effective transform at that
/// scale — the matrix itself is NOT simply reused, since its absolute-pixel
/// translation terms are resolution-dependent. This is exactly why
/// [`matrix_to_params`] must be a lossless round-trip for the full 6-DOF
/// model, not just the old 4-DOF similarity.
///
/// `small_ref_luma` / `small_src_luma` must already be at `small_width` ×
/// `small_height`.
///
/// This is the shared building block behind the `pipeline::align_frame`
/// post-refinement sanity gate, which calls it once for the refined matrix
/// and once for the identity matrix and rejects the refined result if it
/// scores worse.
pub fn registration_rms_at_dims(
    matrix: &Matrix3<f32>,
    full_width: usize,
    full_height: usize,
    small_ref_luma: &[f32],
    small_src_luma: &[f32],
    small_width: usize,
    small_height: usize,
) -> f64 {
    let params = matrix_to_params(matrix, full_width, full_height);
    if !params.is_finite() {
        return PENALTY;
    }
    // `objective_registration` rebuilds the matrix from `params_vec` at the
    // dimensions it is called with, so passing the small width/height here
    // is what actually re-scales the transform; no separate
    // `params_to_matrix` call is needed.
    objective_registration(
        &params.to_vec(),
        small_ref_luma,
        small_src_luma,
        small_width,
        small_height,
    )
}

/// Faithful multi-scale coarse-to-fine bounded intensity registration.
///
/// Builds Gaussian luma pyramids for both `reference` and `source`, then runs
/// a bounded optimiser (golden-section for a single active DOF, Nelder-Mead
/// otherwise) from the coarsest pyramid level to the finest, using the
/// edge-clamped spline warp (`objective_registration_bounded`).
///
/// ## Key design choices
///
/// - **Coarsest-level random restarts**: 10 independent starts are tried at the
///   coarsest level using a deterministic LCG; the best result is kept. This
///   widens the effective basin of attraction without paying full-resolution
///   warp costs.
/// - **Warm-start carry-down**: the best search-space coordinates from each
///   level are passed unchanged to the next finer level. This works because
///   `(tx, ty)` are *fractional* (pixels / image dimension), so the same
///   parameter values apply at every pyramid level — the bounds are built once
///   from the initial estimate and reused at all levels.
/// - **Wider default bounds**: `BoundedRefineOptions::default()` uses ±20 % for
///   shift/scale/rotation, giving the search room to handle larger misalignments.
///
/// # Errors
/// Returns [`StackerError`] if `initial` is non-finite, the image dimensions
/// disagree, or the optimiser produces non-finite / degenerate parameters.
///
/// # Panics
/// Will not panic in practice: the pyramid vector is always seeded with the
/// full-resolution level before the loop, so `.last().expect(…)` is infallible.
pub fn refine_alignment_registration(
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
) -> Result<Matrix3<f32>, StackerError> {
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(StackerError::MathError(
            "refine_alignment_registration: initial matrix contains non-finite values".into(),
        ));
    }
    if reference.width != source.width || reference.height != source.height {
        return Err(StackerError::AlignmentFailed(
            "refine_alignment_registration: reference and source dimensions do not match".into(),
        ));
    }

    let w = source.width;
    let h = source.height;
    let p0 = matrix_to_params(initial, w, h);
    let bounds = SearchBounds::new(&p0, opts);

    // No active DOF → nothing to optimise, return the initial transform.
    if bounds.n() == 0 {
        return Ok(*initial);
    }

    // ── Build coarse-to-fine luma pyramids (finest-first, then reversed). ──
    // We build by hand (rather than using build_luma_pyramid) so we can work
    // directly from slices and control the stop condition per the spec: keep
    // halving while width.min(height) > 64.
    let mut ref_pyramid: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    let mut src_pyramid: Vec<(Vec<f32>, usize, usize)> = Vec::new();

    // Level 0 = full resolution.
    ref_pyramid.push((reference.luma.clone(), w, h));
    src_pyramid.push((source.luma.clone(), w, h));

    loop {
        let &(_, lw, lh) = ref_pyramid.last().expect("pyramid is non-empty");
        if lw.min(lh) <= 64 {
            break;
        }
        let (ref_down, dw, dh) = downsample_luma(&ref_pyramid.last().expect("non-empty").0, lw, lh);
        let (src_down, _, _) = downsample_luma(&src_pyramid.last().expect("non-empty").0, lw, lh);
        ref_pyramid.push((ref_down, dw, dh));
        src_pyramid.push((src_down, dw, dh));
    }

    // Reverse so index 0 = coarsest, last = finest.
    ref_pyramid.reverse();
    src_pyramid.reverse();

    // ── Starting point in logit search space. ──
    let mut coords = bounds.to_search_space(&p0);

    // ── Deterministic LCG for coarsest-level restarts. ──
    // Multiplier and addend from Knuth (TAOCP vol. 2). Seed is a fixed constant.
    // Produces a u64; scale to [0, 1) as (state >> 11) * 2^{-53}.
    #[allow(clippy::unreadable_literal)]
    let mut lcg_state: u64 = 0x2545F491_4F6CDD1D_u64;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Upper 53 bits → uniform float in [0, 1).
        (*state >> 11) as f64 * f64::from_bits(0x3CA0_0000_0000_0000_u64) // 2^{-53}
    };

    let n_levels = ref_pyramid.len();

    for (lvl_idx, (ref_entry, src_entry)) in ref_pyramid.iter().zip(src_pyramid.iter()).enumerate()
    {
        let (ref_luma, lw, lh) = (ref_entry.0.as_slice(), ref_entry.1, ref_entry.2);
        let (src_luma, _, _) = (src_entry.0.as_slice(), src_entry.1, src_entry.2);

        // ── Coarse-biased stop / budget rules ─────────────────────────────────
        //
        // This loop runs from the coarsest level down to — but never
        // including — the full-resolution level, and additionally breaks out
        // of the two finest levels once they are "large" (> 512 px short
        // side). Here `lvl_idx` counts from the coarsest (0) to the finest
        // (`n_levels - 1`), so "the two finest levels" are the last two indices.
        let short_side = lw.min(lh);
        let is_large = short_side > REF_SHORT_SIDE_THRESHOLD;
        let is_full_res = lvl_idx == n_levels - 1;
        let is_finest_two = lvl_idx + 2 >= n_levels;

        // Never refine full resolution, and stop entirely once a finest-two
        // level is still large — see the schedule rationale above.
        if is_full_res || (is_large && is_finest_two) {
            break;
        }

        // Per-level iteration cap (150 by default, 3 when the level is
        // large). The coarsest level's restarts also use this cap.
        let level_iters = if is_large {
            REF_LARGE_LEVEL_ITERS
        } else {
            REF_DEFAULT_LEVEL_ITERS
        };

        let f = |c: &[f64]| objective_registration_bounded(c, &bounds, ref_luma, src_luma, lw, lh);

        if bounds.n() == 1 {
            // Brent is a global 1-D search on the interval; no restarts needed.
            coords = vec![brent_min(-100.0, 100.0, BRENT_TOL, &|x| f(&[x]))];
        } else if lvl_idx == 0 {
            // Coarsest level: 10 attempts (attempt 0 unperturbed, 1..=9 jittered).
            let mut best_obj = f64::INFINITY;
            let mut best_coords = coords.clone();
            for attempt in 0..10_usize {
                let start: Vec<f64> = if attempt == 0 {
                    coords.clone()
                } else {
                    coords
                        .iter()
                        .map(|&c| c + (lcg_next(&mut lcg_state) - 0.5) * 0.001)
                        .collect()
                };
                let result = nelder_mead_generic(&f, &start, BOUNDED_STEP, level_iters, NM_TOL);
                let obj = f(&result);
                if obj < best_obj {
                    best_obj = obj;
                    best_coords = result;
                }
            }
            coords = best_coords;
        } else {
            // Finer levels: warm-start from the carried-down coords, capped to
            // the reference per-level iteration budget.
            coords = nelder_mead_generic(&f, &coords, BOUNDED_STEP, level_iters, NM_TOL);
        }
    }

    let best_p = bounds.to_param_space(&coords);
    if !best_p.is_finite() {
        return Err(StackerError::AlignmentFailed(
            "refine_alignment_registration: optimizer produced non-finite parameters".into(),
        ));
    }

    params_to_matrix(&best_p, w, h).ok_or_else(|| {
        StackerError::AlignmentFailed(
            "refine_alignment_registration: refined parameters produced a degenerate matrix".into(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        super::params::{RegistrationParams, matrix_to_params, params_to_matrix},
        *,
    };
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

    /// `refine_alignment_registration` on a 160×120 image with a known small
    /// shift + scale should produce a refined transform whose RMS residual is
    /// lower than the identity baseline.  The image is large enough (short side
    /// 120 > 64) that the multi-scale path builds at least two pyramid levels,
    /// exercising the coarsest-level restart loop and the warm-start carry-down.
    #[test]
    fn registration_multiscale_reduces_rms_larger_image() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);

        // Known shift + tiny scale change.
        let known_p = RegistrationParams {
            tx: 0.025,
            ty: 0.015,
            scale: 1.005,
            rotate: 0.0,
            aspect: 1.0,
            shear: 0.0,
        };
        let known_matrix =
            params_to_matrix(&known_p, w, h).expect("known params should produce a valid matrix");
        let source = warp_image_clamped(&reference, &known_matrix)
            .expect("warp_image_clamped should succeed");

        // Identity RMS (no alignment correction).
        let id_p = RegistrationParams::identity();
        let id_vec = id_p.to_vec();
        let identity_rms = objective_registration(&id_vec, &reference.luma, &source.luma, w, h);

        // Refine from identity with default (wide) bounds.
        let opts = BoundedRefineOptions {
            max_iterations: 200,
            ..BoundedRefineOptions::default()
        };
        let refined =
            refine_alignment_registration(&reference, &source, &Matrix3::identity(), &opts)
                .expect("refine_alignment_registration should return Ok for 160×120 image");

        let refined_p = matrix_to_params(&refined, w, h);
        let refined_rms =
            objective_registration(&refined_p.to_vec(), &reference.luma, &source.luma, w, h);

        assert!(
            refined_rms < identity_rms,
            "refined RMS ({refined_rms:.6}) should be less than identity RMS ({identity_rms:.6}) \
             for 160×120 multi-scale path"
        );
    }

    /// `refine_alignment_registration` should return `Ok` when given a source
    /// that was produced by warping the reference with a small known translation.
    /// The recovered transform should reduce the RMS residual compared to the
    /// identity initialisation.
    #[test]
    fn registration_reduces_rms_after_small_shift() {
        // Use a size with at least one coarser pyramid level (short side > 64),
        // since the faithful reference schedule never refines the sole
        // full-resolution level — a 64×64 image has no coarser level to refine.
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);

        // Produce source by applying a small known shift (tx=0.03, ty=0.02).
        let known_p = RegistrationParams {
            tx: 0.03,
            ty: 0.02,
            scale: 1.0,
            rotate: 0.0,
            aspect: 1.0,
            shear: 0.0,
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

        // Verify that the refined matrix correctly maps source to reference.
        // known_matrix maps reference to source.
        // refined maps source to reference.
        // So refined * known_matrix should be close to identity.
    }
}
