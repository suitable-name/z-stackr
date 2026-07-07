use akaze::KeyPoint;
use nalgebra::Matrix3;
use stacker_core::error::StackerError;

use super::utils::extract_pairs;
use crate::akaze_match::Match;

/// Result of a breathing-scale estimation.
#[derive(Debug, Clone, Copy)]
pub struct BreathingEstimate {
    /// Recovered uniform scale factor relative to the reference frame.
    /// Values > 1 mean the target frame is magnified (zoomed in) relative
    /// to the reference; values < 1 mean it is zoomed out.
    pub scale: f32,
    /// Residual translation in X after removing the scale component (pixels).
    pub tx: f32,
    /// Residual translation in Y after removing the scale component (pixels).
    pub ty: f32,
}

/// Estimator for the focus-breathing similarity model.
///
/// ## Model
///
/// Focus breathing causes the field of view to change slightly as the focus
/// distance changes, producing a uniform magnification `s` about the optical
/// axis (approximated by the image centre `(cx, cy)`).  The full transform
/// from source to destination is:
///
/// ```text
/// dst_x = s * src_x + (1 - s) * cx + tx_residual
/// dst_y = s * src_y + (1 - s) * cy + ty_residual
/// ```
///
/// Or, in matrix form (centre-anchored similarity + translation):
///
/// ```text
/// M = T(cx, cy) · S(s) · T(-cx, -cy) · T(tx, ty)
///
///   = [ s  0  (1-s)*cx + tx ]
///     [ 0  s  (1-s)*cy + ty ]
///     [ 0  0       1        ]
/// ```
///
/// The 3 parameters `(s, tx_total_x, tx_total_y)` are estimated by
/// least-squares on the provided point correspondences (≥ 2 required).
///
/// ## Composition with the affine alignment step
///
/// The breathing correction is designed as a **pre-processing** step: after
/// applying it, the remaining inter-frame misalignment (rotation, shear, local
/// warp) is handled by the affine RANSAC stage.  The two steps compose as:
///
/// ```text
/// final_matrix = M_affine · M_breathing
/// ```
///
/// To avoid double-correcting, the breathing step should **replace** the
/// `TranslationAndScale` RANSAC mode, not run alongside it.  When `Affine`
/// mode is used as the second stage, any residual scale not captured by the
/// breathing estimator is absorbed by the affine matrix's diagonal.
pub struct BreathingCorrector;

// ── Breathing RANSAC constants ─────────────────────────────────────────────

/// Iterations for the breathing-corrector RANSAC loop.
const BREATHING_RANSAC_ITERS: usize = 500;
/// Reprojection threshold for breathing inliers: 3 pixels squared.
const BREATHING_REPROJ_THRESH_SQ: f32 = 9.0;
/// Minimal sample for a centre-anchored similarity (scale + 2 translation DOF).
const BREATHING_MIN_SAMPLE: usize = 2;

impl BreathingCorrector {
    /// Fit the centre-anchored similarity model `[s, bx, by]` to a **minimal
    /// sample** (exactly 2 point pairs) or to any larger inlier set using
    /// least-squares.
    ///
    /// Returns `None` when the 3×3 normal-equation system is degenerate.
    fn fit_breathing(pairs: &[(f32, f32, f32, f32)]) -> Option<BreathingEstimate> {
        if pairs.len() < BREATHING_MIN_SAMPLE {
            return None;
        }
        let n = pairs.len();
        let rows = 2 * n;
        let mut a = nalgebra::DMatrix::<f64>::zeros(rows, 3);
        let mut b_vec = nalgebra::DVector::<f64>::zeros(rows);
        for (i, &(sx, sy, dx, dy)) in pairs.iter().enumerate() {
            a[(2 * i, 0)] = f64::from(sx);
            a[(2 * i, 1)] = 1.0;
            b_vec[2 * i] = f64::from(dx);
            a[(2 * i + 1, 0)] = f64::from(sy);
            a[(2 * i + 1, 2)] = 1.0;
            b_vec[2 * i + 1] = f64::from(dy);
        }
        let ata = a.transpose() * &a;
        let atb = a.transpose() * &b_vec;
        let p = ata.lu().solve(&atb)?;
        Some(BreathingEstimate {
            scale: p[0] as f32,
            tx: p[1] as f32,
            ty: p[2] as f32,
        })
    }

    /// Squared reprojection error of a `BreathingEstimate` for a single pair,
    /// expressed as the centre-anchored forward map (before recovering `tx/ty`).
    ///
    /// The raw LSQ parameters are `[s, bx, by]` where
    /// `predicted_dx = s*sx + bx` and `predicted_dy = s*sy + by`.
    /// We reuse `build_matrix` to get the 3×3 form and delegate to
    /// [`reproj_err_sq`].
    fn breathing_reproj_sq(est_raw: &(f32, f32, f32), sx: f32, sy: f32, dx: f32, dy: f32) -> f32 {
        // est_raw = (s, bx, by) from the raw LSQ solve — NOT the final (scale, tx, ty).
        // predicted_dx = s*sx + bx,  predicted_dy = s*sy + by
        let (s, bx, by) = *est_raw;
        let pred_delta_x = s.mul_add(sx, bx);
        let pred_delta_y = s.mul_add(sy, by);
        (pred_delta_x - dx).mul_add(pred_delta_x - dx, (pred_delta_y - dy) * (pred_delta_y - dy))
    }

    /// Estimate focus-breathing scale and residual translation from point
    /// correspondences using a **RANSAC loop** to reject gross outliers.
    ///
    /// The image centre `(cx, cy)` defines the pivot about which breathing
    /// magnification occurs (typically `width/2`, `height/2`).
    ///
    /// ## RANSAC design
    ///
    /// | Parameter        | Value                                              |
    /// |------------------|----------------------------------------------------|
    /// | Minimal sample   | 2 correspondences (3 DOF: scale + 2 translation)  |
    /// | Iterations       | 500 (deterministic LCG — no `rand` crate)          |
    /// | Inlier threshold | 3 px reprojection error (squared = 9.0)            |
    /// | Final refit      | Least-squares over all inliers                     |
    ///
    /// ## Least-squares formulation
    ///
    /// Expanding the centre-anchored model:
    ///
    /// ```text
    /// dx = s * sx + bx   where bx = (1-s)*cx + tx_residual
    /// dy = s * sy + by   where by = (1-s)*cy + ty_residual
    /// ```
    ///
    /// Parameterised as `[s, bx, by]` over the 2N×3 system:
    ///
    /// ```text
    /// [ sx  1  0 ] [ s  ]   [ dx ]
    /// [ sy  0  1 ] [ bx ] = [ dy ]
    ///              [ by ]
    /// ```
    ///
    /// After RANSAC the inlier refit recovers `(s, bx, by)`.  The residual
    /// translation is extracted as `tx = bx − (1−s)·cx`, `ty = by − (1−s)·cy`.
    ///
    /// # Errors
    /// Returns [`StackerError::AlignmentFailed`] if fewer than 2
    /// correspondences are provided, every RANSAC hypothesis is degenerate, or
    /// the inlier refit is degenerate.
    pub fn estimate_from_pairs(
        pairs: &[(f32, f32, f32, f32)],
        cx: f32,
        cy: f32,
    ) -> Result<BreathingEstimate, StackerError> {
        if pairs.len() < BREATHING_MIN_SAMPLE {
            return Err(StackerError::AlignmentFailed(
                "breathing estimation needs at least 2 correspondences".into(),
            ));
        }

        let n = pairs.len();
        let mut best_count: usize = 0;
        let mut best_raw: Option<(f32, f32, f32)> = None;

        // Deterministic LCG — same constants as AlignmentEstimator::compute_matrix.
        let mut lcg: u64 = 0xdead_beef_cafe_f00d;
        let mut lcg_next = || -> u64 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lcg
        };

        for _ in 0..BREATHING_RANSAC_ITERS {
            // Draw 2 distinct indices.
            let mut sample_idx: Vec<usize> = Vec::with_capacity(BREATHING_MIN_SAMPLE);
            let mut tries = 0usize;
            while sample_idx.len() < BREATHING_MIN_SAMPLE && tries < BREATHING_MIN_SAMPLE * 20 {
                tries += 1;
                // High bits again — see the comment in
                // `AlignmentEstimator::compute_matrix` on why an LCG's low
                // bits must not be used for `% n` index sampling.
                let idx = ((lcg_next() >> 32) as usize) % n;
                if !sample_idx.contains(&idx) {
                    sample_idx.push(idx);
                }
            }
            if sample_idx.len() < BREATHING_MIN_SAMPLE {
                continue;
            }

            let sample: Vec<_> = sample_idx.iter().map(|&i| pairs[i]).collect();
            let Some(raw) = Self::fit_breathing(&sample) else {
                continue;
            };

            // Convert raw LSQ result to the (s, bx, by) triple used for reprojection.
            // fit_breathing returns BreathingEstimate{scale, tx=bx, ty=by} at this point
            // (the final tx/ty unwrapping happens after the centre is known).
            let raw_triple = (raw.scale, raw.tx, raw.ty);

            let inlier_count = (0..n)
                .filter(|&i| {
                    let (sx, sy, dx, dy) = pairs[i];
                    Self::breathing_reproj_sq(&raw_triple, sx, sy, dx, dy)
                        <= BREATHING_REPROJ_THRESH_SQ
                })
                .count();

            if inlier_count > best_count {
                best_count = inlier_count;
                best_raw = Some(raw_triple);
            }
        }

        // Same rationale as `AlignmentEstimator::compute_matrix`: a model
        // corroborated by only `BREATHING_MIN_SAMPLE` (2) points is not
        // trustworthy on noisy match sets. Require at least 20% of all
        // candidate matches, floored at `BREATHING_MIN_SAMPLE + 2` and capped
        // at 50. Callers treat `Err` as "no breathing correction available"
        // and skip the step, so being stricter here is safe.
        let min_inliers = ((n as f32 * 0.2).ceil() as usize)
            .clamp(BREATHING_MIN_SAMPLE + 2, 50)
            .min(n);
        if best_count < min_inliers {
            return Err(StackerError::AlignmentFailed(format!(
                "breathing RANSAC found only {best_count} inliers (need {min_inliers})"
            )));
        }

        let best_raw = best_raw.ok_or_else(|| {
            StackerError::AlignmentFailed("breathing RANSAC produced no valid model".into())
        })?;

        // Recompute the winning model's inlier set and refit for a numerically
        // stable final estimate.
        let inlier_pairs: Vec<_> = (0..n)
            .filter(|&i| {
                let (sx, sy, dx, dy) = pairs[i];
                Self::breathing_reproj_sq(&best_raw, sx, sy, dx, dy) <= BREATHING_REPROJ_THRESH_SQ
            })
            .map(|i| pairs[i])
            .collect();
        let raw = Self::fit_breathing(&inlier_pairs).ok_or_else(|| {
            StackerError::AlignmentFailed(
                "breathing RANSAC: degenerate inlier set in final refit".into(),
            )
        })?;

        // raw.tx = bx = (1-s)*cx + tx_residual  →  unwrap to residual translation.
        let scale = raw.scale;
        let tx = (1.0 - scale).mul_add(-cx, raw.tx);
        let ty = (1.0 - scale).mul_add(-cy, raw.ty);

        Ok(BreathingEstimate { scale, tx, ty })
    }

    /// Estimate from raw akaze match data (convenience wrapper).
    ///
    /// Internally uses the RANSAC-hardened estimator; same behaviour as
    /// [`Self::estimate_from_pairs`] but accepts akaze match structures.
    ///
    /// # Errors
    /// Same as [`Self::estimate_from_pairs`].
    pub fn estimate_from_matches(
        matches: &[Match],
        kps0: &[KeyPoint],
        kps1: &[KeyPoint],
        cx: f32,
        cy: f32,
    ) -> Result<BreathingEstimate, StackerError> {
        let pairs = extract_pairs(matches, kps0, kps1);
        Self::estimate_from_pairs(&pairs, cx, cy)
    }

    /// Build a centre-anchored similarity matrix from a `BreathingEstimate`.
    ///
    /// ## Matrix derivation
    ///
    /// Scaling by `s` about `(cx, cy)` with residual translation `(tx, ty)`:
    ///
    /// ```text
    /// M = T(cx+tx, cy+ty) · S(s) · T(-cx, -cy)
    ///   = [ s  0  (1-s)*cx + tx ]
    ///     [ 0  s  (1-s)*cy + ty ]
    ///     [ 0  0       1        ]
    /// ```
    ///
    /// A point `p` in the source maps to `M * p` in the destination, correctly
    /// keeping the image centre stationary when `tx = ty = 0`.
    #[must_use]
    pub fn build_matrix(est: &BreathingEstimate, cx: f32, cy: f32) -> Matrix3<f32> {
        let mut m = Matrix3::identity();
        m[(0, 0)] = est.scale;
        m[(1, 1)] = est.scale;
        m[(0, 2)] = (1.0 - est.scale).mul_add(cx, est.tx);
        m[(1, 2)] = (1.0 - est.scale).mul_add(cy, est.ty);
        m
    }
}
