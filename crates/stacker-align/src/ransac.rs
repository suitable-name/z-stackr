//! RANSAC estimation of a transform from AKAZE keypoint matches.
//!
//! The estimators fit the **reference to source** direction (the same one the
//! intensity refinement in [`crate::refine`] optimises), so their output is a
//! valid seed for it. Sampling is deterministic: a fixed-seed `SplitMix64`
//! generator drives every draw, so the same matches always give the same matrix.

use akaze::KeyPoint;
use nalgebra::Matrix3;
use stacker_core::{error::StackerError, fma::fma_f32};

use crate::akaze_match::Match;

/// The motion model [`AlignmentEstimator::compute_matrix`] fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignmentMode {
    /// Shift only (1 correspondence is the minimal sample).
    TranslationOnly,
    /// Shift and uniform scale (minimal sample: 2).
    TranslationAndScale,
    /// Full 6-parameter affine map (minimal sample: 3).
    Affine,
    /// Projective map with 8 degrees of freedom (minimal sample: 4).
    Homography,
}

/// RANSAC fitting of an [`AlignmentMode`] model to keypoint matches.
pub struct AlignmentEstimator;

// ── helpers ────────────────────────────────────────────────────────────────

/// Extract `(sx, sy, dx, dy)` tuples from akaze matches + `KeyPoint` lists.
fn extract_pairs(
    matches: &[Match],
    kps0: &[KeyPoint],
    kps1: &[KeyPoint],
) -> Vec<(f32, f32, f32, f32)> {
    matches
        .iter()
        .filter_map(|m| {
            let s = kps0.get(m.index_0)?;
            let d = kps1.get(m.index_1)?;

            // Use dot notation (.0, .1) for tuple access instead of indexing [0], [1]
            Some((s.point.0, s.point.1, d.point.0, d.point.1))
        })
        .collect()
}

/// Apply a 3×3 homogeneous matrix to a 2-D point.
/// Returns `None` when the homogeneous divisor is near-zero (degenerate).
///
/// Public only so the integration tests can project points through a fitted
/// matrix.
#[doc(hidden)]
#[must_use]
pub fn apply_h(m: &Matrix3<f32>, px: f32, py: f32) -> Option<(f32, f32)> {
    let w = fma_f32(m[(2, 1)], py, m[(2, 0)] * px) + m[(2, 2)];
    if w.abs() < 1e-8 {
        return None;
    }
    let x = (fma_f32(m[(0, 1)], py, m[(0, 0)] * px) + m[(0, 2)]) / w;
    let y = (fma_f32(m[(1, 1)], py, m[(1, 0)] * px) + m[(1, 2)]) / w;
    Some((x, y))
}

/// Squared reprojection error for one correspondence.
fn reproj_err_sq(m: &Matrix3<f32>, sx: f32, sy: f32, dx: f32, dy: f32) -> f32 {
    match apply_h(m, sx, sy) {
        Some((px, py)) => fma_f32(py - dy, py - dy, (px - dx) * (px - dx)),
        None => f32::MAX,
    }
}

// ── mode-specific estimators ───────────────────────────────────────────────

/// Translation-only: t = mean(dst − src).
#[expect(
    clippy::cast_precision_loss,
    reason = "the number of keypoint pairs is far below 2^24"
)]
fn fit_translation(pairs: &[(f32, f32, f32, f32)]) -> Option<Matrix3<f32>> {
    if pairs.is_empty() {
        return None;
    }
    let n = pairs.len() as f32;
    let (tx, ty) = pairs
        .iter()
        .fold((0.0_f32, 0.0_f32), |(ax, ay), &(sx, sy, dx, dy)| {
            (ax + (dx - sx), ay + (dy - sy))
        });
    let mut m = Matrix3::identity();
    m[(0, 2)] = tx / n;
    m[(1, 2)] = ty / n;
    Some(m)
}

/// Translation + uniform scale.  Model: `dx = s·sx + tx`, `dy = s·sy + ty`.
/// Solved via least-squares (2N × 3 system).
#[expect(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    reason = "a/b/p are the textbook names of the normal equations; the f64 solve is stored in the f32 matrix"
)]
fn fit_translation_scale(pairs: &[(f32, f32, f32, f32)]) -> Option<Matrix3<f32>> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len();
    let rows = 2 * n;
    let mut a = nalgebra::DMatrix::<f64>::zeros(rows, 3);
    let mut b = nalgebra::DVector::<f64>::zeros(rows);
    for (i, &(sx, sy, dx, dy)) in pairs.iter().enumerate() {
        a[(2 * i, 0)] = f64::from(sx);
        a[(2 * i, 1)] = 1.0;
        b[2 * i] = f64::from(dx);
        a[(2 * i + 1, 0)] = f64::from(sy);
        a[(2 * i + 1, 2)] = 1.0;
        b[2 * i + 1] = f64::from(dy);
    }
    let ata = a.transpose() * &a;
    let atb = a.transpose() * &b;
    let p = ata.lu().solve(&atb)?;
    let scale = p[0] as f32;
    let tx = p[1] as f32;
    let ty = p[2] as f32;
    let mut m = Matrix3::identity();
    m[(0, 0)] = scale;
    m[(1, 1)] = scale;
    m[(0, 2)] = tx;
    m[(1, 2)] = ty;
    Some(m)
}

/// Affine (6 DOF): least-squares via 2N × 6 system.
#[expect(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    reason = "a/b/p are the textbook names of the normal equations; the f64 solve is stored in the f32 matrix"
)]
fn fit_affine(pairs: &[(f32, f32, f32, f32)]) -> Option<Matrix3<f32>> {
    if pairs.len() < 3 {
        return None;
    }
    let n = pairs.len();
    let rows = 2 * n;
    let mut a = nalgebra::DMatrix::<f64>::zeros(rows, 6);
    let mut b = nalgebra::DVector::<f64>::zeros(rows);
    for (i, &(sx, sy, dx, dy)) in pairs.iter().enumerate() {
        a[(2 * i, 0)] = f64::from(sx);
        a[(2 * i, 1)] = f64::from(sy);
        a[(2 * i, 2)] = 1.0;
        b[2 * i] = f64::from(dx);
        a[(2 * i + 1, 3)] = f64::from(sx);
        a[(2 * i + 1, 4)] = f64::from(sy);
        a[(2 * i + 1, 5)] = 1.0;
        b[2 * i + 1] = f64::from(dy);
    }
    let ata = a.transpose() * &a;
    let atb = a.transpose() * &b;
    let p = ata.lu().solve(&atb)?;
    let mut m = Matrix3::identity();
    m[(0, 0)] = p[0] as f32;
    m[(0, 1)] = p[1] as f32;
    m[(0, 2)] = p[2] as f32;
    m[(1, 0)] = p[3] as f32;
    m[(1, 1)] = p[4] as f32;
    m[(1, 2)] = p[5] as f32;
    Some(m)
}

/// Homography (8 DOF): DLT with SVD — normalised null-space solution.
///
/// Returns `None` (which `compute_matrix` treats as a degenerate hypothesis)
/// for fewer than four pairs or when the null vector has a vanishing last
/// coordinate.
///
/// nalgebra's thin SVD of an `m × 9` matrix yields only `min(m, 9)` rows of
/// `Vᵀ`.  The minimal four-pair sample gives an 8 × 9 design matrix, whose
/// `Vᵀ` has no row 8, so the design matrix is padded with zero rows up to at
/// least 9 × 9.  Zero rows do not change the null space, and the singular
/// values are sorted in descending order, so the last row of `Vᵀ` is the null
/// (smallest-singular-value) direction in every case.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the f64 SVD solution is stored in the f32 matrix"
)]
fn fit_homography(pairs: &[(f32, f32, f32, f32)]) -> Option<Matrix3<f32>> {
    if pairs.len() < 4 {
        return None;
    }
    let n = pairs.len();
    let mut a = nalgebra::DMatrix::<f64>::zeros((2 * n).max(9), 9);
    for (i, &(sx, sy, dx, dy)) in pairs.iter().enumerate() {
        let (sx, sy, dx, dy) = (f64::from(sx), f64::from(sy), f64::from(dx), f64::from(dy));
        a[(2 * i, 0)] = -sx;
        a[(2 * i, 1)] = -sy;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = dx * sx;
        a[(2 * i, 7)] = dx * sy;
        a[(2 * i, 8)] = dx;
        a[(2 * i + 1, 3)] = -sx;
        a[(2 * i + 1, 4)] = -sy;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = dy * sx;
        a[(2 * i + 1, 7)] = dy * sy;
        a[(2 * i + 1, 8)] = dy;
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t.as_ref()?;
    if vt.nrows() < 9 {
        return None;
    }
    let h_vec = vt.row(8);
    let hscale = h_vec[8];
    if hscale.abs() < 1e-10 {
        return None;
    }
    let mut m = Matrix3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            m[(r, c)] = (h_vec[r * 3 + c] / hscale) as f32;
        }
    }
    Some(m)
}

// ── RANSAC loop ────────────────────────────────────────────────────────────

const RANSAC_ITERS: usize = 500;
/// 3-pixel reprojection threshold (squared).
const REPROJ_THRESH_SQ: f32 = 9.0;

/// Fixed seed of the RANSAC sampler: the same stack always draws the same
/// samples, so results are reproducible run to run.
const RANSAC_SEED: u64 = 0xdead_beef_cafe_f00d;

/// Small deterministic generator for RANSAC sample indices (`SplitMix64`,
/// Steele et al.), standing in for the `rand` crate, which this workspace does not
/// depend on.
///
/// It replaces a plain 64-bit LCG reduced with `% n`: the low bit of a
/// power-of-two LCG alternates on every draw, so for an even match count every
/// two-point sample paired one even with one odd index and half of all pairs
/// could never be drawn.  `SplitMix64` has no such low-bit structure, and
/// [`Self::below`] removes the modulo bias by rejection.
struct SampleRng(u64);

impl SampleRng {
    const fn new() -> Self {
        Self(RANSAC_SEED)
    }

    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform index in `0..n` (`n > 0`), by rejection sampling.
    fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "cannot draw an index below zero");
        let bound = u64::try_from(n).unwrap_or(u64::MAX);
        // 2^64 mod bound: draws below it would over-represent small residues.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return usize::try_from(x % bound).unwrap_or(0);
            }
        }
    }
}

/// Fraction of all candidate matches (as a divisor: one fifth) a model must
/// be corroborated by.
const MIN_INLIER_DIVISOR: usize = 5;
/// Upper bound of the inlier requirement, so huge match sets do not demand
/// thousands of inliers.
const MAX_REQUIRED_INLIERS: usize = 50;

/// The number of inliers a RANSAC model needs to be accepted:
/// `ceil(0.2 * matches)`, floored at `min_sample + 2` (a model backed by the
/// bare minimal sample is never trustworthy) and capped at 50, and never more
/// than the match count itself.
///
/// This is the rule of the original z-stackr implementation; `n.div_ceil(5)`
/// equals `ceil(0.2 * n)` exactly (no float rounding).
const fn required_inliers(matches: usize, min_sample: usize) -> usize {
    let wanted = matches.div_ceil(MIN_INLIER_DIVISOR);
    let floor = min_sample + 2;
    let clamped = if wanted < floor {
        floor
    } else if wanted > MAX_REQUIRED_INLIERS {
        MAX_REQUIRED_INLIERS
    } else {
        wanted
    };
    if clamped < matches { clamped } else { matches }
}

const fn min_sample_size(mode: AlignmentMode) -> usize {
    match mode {
        AlignmentMode::TranslationOnly => 1,
        AlignmentMode::TranslationAndScale => 2,
        AlignmentMode::Affine => 3,
        AlignmentMode::Homography => 4,
    }
}

fn fit_minimal(pairs: &[(f32, f32, f32, f32)], mode: AlignmentMode) -> Option<Matrix3<f32>> {
    match mode {
        AlignmentMode::TranslationOnly => fit_translation(pairs),
        AlignmentMode::TranslationAndScale => fit_translation_scale(pairs),
        AlignmentMode::Affine => fit_affine(pairs),
        AlignmentMode::Homography => fit_homography(pairs),
    }
}

impl AlignmentEstimator {
    /// Estimate a transformation matrix from akaze feature matches using RANSAC.
    ///
    /// # Errors
    /// Returns [`StackerError::AlignmentFailed`] when there are too few
    /// matches or every candidate model is degenerate.
    pub fn compute_matrix(
        matches: &[Match],
        kps0: &[KeyPoint],
        kps1: &[KeyPoint],
        mode: AlignmentMode,
    ) -> Result<Matrix3<f32>, StackerError> {
        let pairs = extract_pairs(matches, kps0, kps1);
        let min_pts = min_sample_size(mode);

        if pairs.len() < min_pts {
            return Err(StackerError::AlignmentFailed(format!(
                "need at least {min_pts} matches for {mode:?}, got {}",
                pairs.len()
            )));
        }

        let mut best_count: usize = 0;
        let mut best_model: Option<Matrix3<f32>> = None;
        let n = pairs.len();

        // Deterministic pseudo-random index selection (see `SampleRng`).
        let mut rng = SampleRng::new();

        for _ in 0..RANSAC_ITERS {
            // Draw `min_pts` distinct indices.
            let mut sample_idx: Vec<usize> = Vec::with_capacity(min_pts);
            let mut tries = 0usize;
            while sample_idx.len() < min_pts && tries < min_pts * 20 {
                tries += 1;
                let idx = rng.below(n);
                if !sample_idx.contains(&idx) {
                    sample_idx.push(idx);
                }
            }
            if sample_idx.len() < min_pts {
                continue;
            }

            let sample: Vec<_> = sample_idx.iter().map(|&i| pairs[i]).collect();
            let Some(m) = fit_minimal(&sample, mode) else {
                continue;
            };

            // Count inliers without allocating a vector each iteration; the
            // winning model's inlier set is recomputed once after the loop.
            let inlier_count = (0..n)
                .filter(|&i| {
                    let (sx, sy, dx, dy) = pairs[i];
                    reproj_err_sq(&m, sx, sy, dx, dy) <= REPROJ_THRESH_SQ
                })
                .count();

            if inlier_count > best_count {
                best_count = inlier_count;
                best_model = Some(m);
            }
        }

        let required = required_inliers(n, min_pts);
        if best_count < required {
            return Err(StackerError::AlignmentFailed(format!(
                "RANSAC found only {best_count} inliers (need {required})"
            )));
        }

        let best_m = best_model.ok_or_else(|| {
            StackerError::AlignmentFailed("RANSAC produced no valid model".into())
        })?;

        // Recompute the winning model's inlier set and refit for a better
        // final estimate.
        let inlier_pairs: Vec<_> = (0..n)
            .filter(|&i| {
                let (sx, sy, dx, dy) = pairs[i];
                reproj_err_sq(&best_m, sx, sy, dx, dy) <= REPROJ_THRESH_SQ
            })
            .map(|i| pairs[i])
            .collect();
        fit_minimal(&inlier_pairs, mode).ok_or_else(|| {
            StackerError::AlignmentFailed("degenerate inlier set in final refit".into())
        })
    }
}

// ── Focus-breathing correction ─────────────────────────────────────────────

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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the f64 least-squares solution is stored in the f32 estimate"
    )]
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
    #[expect(
        clippy::similar_names,
        reason = "pred_dx/pred_dy are the paired predictions"
    )]
    fn breathing_reproj_sq(est_raw: &(f32, f32, f32), sx: f32, sy: f32, dx: f32, dy: f32) -> f32 {
        // est_raw = (s, bx, by) from the raw LSQ solve — NOT the final (scale, tx, ty).
        // predicted_dx = s*sx + bx,  predicted_dy = s*sy + by
        let (s, bx, by) = *est_raw;
        let pred_dx = fma_f32(s, sx, bx);
        let pred_dy = fma_f32(s, sy, by);
        fma_f32(pred_dx - dx, pred_dx - dx, (pred_dy - dy) * (pred_dy - dy))
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
    /// | Iterations       | 500 (deterministic `SplitMix64`, no `rand` crate)   |
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

        // Same deterministic generator and seed as AlignmentEstimator::compute_matrix.
        let mut rng = SampleRng::new();

        for _ in 0..BREATHING_RANSAC_ITERS {
            // Draw 2 distinct indices.
            let mut sample_idx: Vec<usize> = Vec::with_capacity(BREATHING_MIN_SAMPLE);
            let mut tries = 0usize;
            while sample_idx.len() < BREATHING_MIN_SAMPLE && tries < BREATHING_MIN_SAMPLE * 20 {
                tries += 1;
                let idx = rng.below(n);
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

        let required = required_inliers(n, BREATHING_MIN_SAMPLE);
        if best_count < required {
            return Err(StackerError::AlignmentFailed(format!(
                "breathing RANSAC found only {best_count} inliers (need {required})"
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
        let tx = fma_f32(1.0 - scale, -cx, raw.tx);
        let ty = fma_f32(1.0 - scale, -cy, raw.ty);

        Ok(BreathingEstimate { scale, tx, ty })
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
        m[(0, 2)] = fma_f32(1.0 - est.scale, cx, est.tx);
        m[(1, 2)] = fma_f32(1.0 - est.scale, cy, est.ty);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::{SampleRng, required_inliers};

    /// The acceptance floor: 20 % of the matches, at least `min_sample + 2`,
    /// at most 50, never above the match count.
    #[test]
    fn required_inliers_follows_the_floor_and_cap() {
        assert_eq!(required_inliers(100, 1), 20);
        assert_eq!(required_inliers(101, 1), 21, "rounds up");
        assert_eq!(required_inliers(10, 1), 3, "floor min_sample + 2");
        assert_eq!(required_inliers(10, 4), 6);
        assert_eq!(required_inliers(1000, 1), 50, "cap");
        assert_eq!(required_inliers(2, 1), 2, "never above the match count");
        assert_eq!(required_inliers(0, 1), 0);
    }

    /// ALN-5: with an even index count the old LCG-modulo sampler paired an even
    /// with an odd index in every two-point sample.  Over 500 draws of `n = 8`
    /// the sampler must produce samples of two even indices and samples of two
    /// odd ones, and reach every index.
    #[test]
    fn sampler_draws_same_parity_pairs_for_even_n() {
        let n = 8;
        let mut rng = SampleRng::new();
        let (mut both_even, mut both_odd, mut mixed) = (0_usize, 0_usize, 0_usize);
        let mut seen = [false; 8];
        for _ in 0..500 {
            let a = rng.below(n);
            let mut b = rng.below(n);
            while b == a {
                b = rng.below(n);
            }
            seen[a] = true;
            seen[b] = true;
            match (a % 2, b % 2) {
                (0, 0) => both_even += 1,
                (1, 1) => both_odd += 1,
                _ => mixed += 1,
            }
        }
        assert!(both_even > 0, "no sample of two even indices in 500 draws");
        assert!(both_odd > 0, "no sample of two odd indices in 500 draws");
        assert!(mixed > 0, "no mixed-parity sample in 500 draws");
        assert!(seen.iter().all(|&s| s), "an index was never drawn");
    }

    /// The generator is a pure function of its fixed seed, and `below` stays in
    /// range for awkward bounds.
    #[test]
    fn sampler_is_deterministic_and_in_range() {
        let mut a = SampleRng::new();
        let mut b = SampleRng::new();
        for n in [1_usize, 2, 3, 7, 8, 1000, 65_537] {
            for _ in 0..200 {
                let (x, y) = (a.below(n), b.below(n));
                assert_eq!(x, y, "same seed, same sequence");
                assert!(x < n, "{x} is out of range for n = {n}");
            }
        }
    }
}
