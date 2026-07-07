use akaze::KeyPoint;
use nalgebra::Matrix3;
use stacker_core::error::StackerError;

use super::utils::{extract_pairs, reproj_err_sq};
use crate::akaze_match::Match;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignmentMode {
    TranslationOnly,
    TranslationAndScale,
    Affine,
    Homography,
}

pub struct AlignmentEstimator;

/// Translation-only: t = mean(dst − src).
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
fn fit_homography(pairs: &[(f32, f32, f32, f32)]) -> Option<Matrix3<f32>> {
    if pairs.len() < 4 {
        return None;
    }
    let n = pairs.len();
    let mut a = nalgebra::DMatrix::<f64>::zeros(2 * n, 9);
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

const RANSAC_ITERS: usize = 500;
/// 3-pixel reprojection threshold (squared).
const REPROJ_THRESH_SQ: f32 = 9.0;

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

        // Deterministic pseudo-random index selection via LCG
        // (avoids the `rand` crate which is not in the allowed list).
        let mut lcg: u64 = 0xdead_beef_cafe_f00d;
        let mut lcg_next = || -> u64 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            lcg
        };

        for _ in 0..RANSAC_ITERS {
            // Draw `min_pts` distinct indices.
            let mut sample_idx: Vec<usize> = Vec::with_capacity(min_pts);
            let mut tries = 0usize;
            while sample_idx.len() < min_pts && tries < min_pts * 20 {
                tries += 1;
                // Take the HIGH bits of the LCG state, not the low bits: with a
                // power-of-two modulus (the implicit `% 2^64` of wrapping
                // arithmetic), the low bits of a linear congruential generator
                // have very short periods (bit 0 alternates every step, bit 1
                // has period 4, …), which biases `% n` sampling badly whenever
                // `n` is even. The high bits do not suffer from this and are
                // the standard mitigation for LCGs used as index generators.
                let idx = ((lcg_next() >> 32) as usize) % n;
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

        // A model must be corroborated by substantially more matches than its
        // own minimal sample, or it is statistically meaningless: on
        // defocused / low-texture macro frames a large fraction of AKAZE
        // matches are wrong, so a model that only barely clears `min_pts`
        // (e.g. exactly 1 inlier for `TranslationOnly`, whose minimal sample
        // size IS 1) can be an arbitrary large, wrong translation that
        // happened to agree with a single spurious match. Require support
        // from at least 20% of all candidate matches, with a floor of
        // `min_pts + 2` (so tiny match sets still demand more than the bare
        // minimum) and a cap of 50 (so huge match sets do not demand
        // thousands of inliers). Callers treat `Err` here as "no usable
        // seed" and fall back to the previous frame's matrix or identity, so
        // being stricter is safe.
        let min_inliers = ((n as f32 * 0.2).ceil() as usize)
            .clamp(min_pts + 2, 50)
            .min(n);
        if best_count < min_inliers {
            return Err(StackerError::AlignmentFailed(format!(
                "RANSAC found only {best_count} inliers (need {min_inliers})"
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
