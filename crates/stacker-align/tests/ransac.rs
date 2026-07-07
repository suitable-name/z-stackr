#![cfg(feature = "akaze")]
#![allow(
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use akaze::KeyPoint;
use stacker_align::{akaze_match::Match, ransac::*, transform::warp_image_clamped};
use stacker_core::{error::StackerError, image::PlanarImage};

/// Build a synthetic `KeyPoint` at `(x, y)`.
fn kp(x: f32, y: f32) -> KeyPoint {
    KeyPoint {
        point: (x, y), // akaze 0.7.0 uses [f32; 2] instead of (f32, f32)
        response: 1.0,
        size: 1.0,
        angle: 0.0,
        octave: 0,
        class_id: 0,
    }
}

/// Build a `Match` between index `i` and index `j`.
fn mk_match(i: usize, j: usize) -> Match {
    Match {
        index_0: i,
        index_1: j,
        // distance field removed to match the updated local `Match` struct
    }
}

/// Build parallel `kps0`/`kps1`/`matches` slices from `(sx, sy, dx, dy)` pairs.
fn build_correspondences(
    pairs: &[(f32, f32, f32, f32)],
) -> (Vec<KeyPoint>, Vec<KeyPoint>, Vec<Match>) {
    let kps0: Vec<KeyPoint> = pairs.iter().map(|&(sx, sy, _, _)| kp(sx, sy)).collect();
    let kps1: Vec<KeyPoint> = pairs.iter().map(|&(_, _, dx, dy)| kp(dx, dy)).collect();
    let matches: Vec<Match> = (0..pairs.len()).map(|i| mk_match(i, i)).collect();
    (kps0, kps1, matches)
}

#[test]
fn test_too_few_matches_returns_error() {
    let result = AlignmentEstimator::compute_matrix(&[], &[], &[], AlignmentMode::TranslationOnly);
    assert!(matches!(result, Err(StackerError::AlignmentFailed(_))));
}

#[test]
fn test_translation_recovery() {
    let tx = 10.0_f32;
    let ty = 5.0_f32;
    let raw: Vec<(f32, f32, f32, f32)> = vec![
        (10.0, 20.0, 10.0 + tx, 20.0 + ty),
        (50.0, 80.0, 50.0 + tx, 80.0 + ty),
        (120.0, 60.0, 120.0 + tx, 60.0 + ty),
        (200.0, 150.0, 200.0 + tx, 150.0 + ty),
        (30.0, 90.0, 30.0 + tx, 90.0 + ty),
        (80.0, 40.0, 80.0 + tx, 40.0 + ty),
    ];
    let (kps0, kps1, matches) = build_correspondences(&raw);

    let m =
        AlignmentEstimator::compute_matrix(&matches, &kps0, &kps1, AlignmentMode::TranslationOnly)
            .expect("should converge on clean data");

    let err_x = (m[(0, 2)] - tx).abs();
    let err_y = (m[(1, 2)] - ty).abs();
    assert!(err_x < 0.5, "tx error {err_x} too large");
    assert!(err_y < 0.5, "ty error {err_y} too large");
}

#[test]
fn test_affine_recovery_with_outliers() {
    let angle = 5.0_f32.to_radians();
    let s = 1.2_f32;
    let cos_a = angle.cos() * s;
    let sin_a = angle.sin() * s;
    let tx = 8.0_f32;
    let ty = -3.0_f32;

    let mut raw: Vec<(f32, f32, f32, f32)> = (0..20)
        .map(|i| {
            let sx = i as f32 * 15.0;
            let sy = (i % 5) as f32 * 20.0;
            let dx = cos_a * sx - sin_a * sy + tx;
            let dy = sin_a * sx + cos_a * sy + ty;
            (sx, sy, dx, dy)
        })
        .collect();

    // Gross outliers
    for off in [500.0_f32, -400.0, 300.0, -200.0] {
        raw.push((10.0, 10.0, 10.0 + off, 10.0 + off));
    }

    let (kps0, kps1, matches) = build_correspondences(&raw);

    let m = AlignmentEstimator::compute_matrix(&matches, &kps0, &kps1, AlignmentMode::Affine)
        .expect("should converge with majority inliers");

    let test_sx = 30.0_f32;
    let test_sy = 40.0_f32;
    let expected_dx = cos_a * test_sx - sin_a * test_sy + tx;
    let expected_dy = sin_a * test_sx + cos_a * test_sy + ty;
    let (got_dx, got_dy) = apply_h(&m, test_sx, test_sy).expect("valid projection");
    let err = (got_dx - expected_dx).hypot(got_dy - expected_dy);
    assert!(err < 1.5, "reprojection error {err} too large");
}

// ── BreathingCorrector tests ───────────────────────────────────────────

/// Sanity-check: the `build_matrix` function maps the image centre to
/// itself (plus any residual translation).  With zero residual translation
/// the centre should map to itself exactly regardless of scale.
#[test]
fn test_build_matrix_centre_is_fixed_point() {
    let cx = 100.0_f32;
    let cy = 80.0_f32;
    let est = BreathingEstimate {
        scale: 1.05,
        tx: 0.0,
        ty: 0.0,
    };
    let m = BreathingCorrector::build_matrix(&est, cx, cy);

    let mapped_cx = m[(0, 0)] * cx + m[(0, 1)] * cy + m[(0, 2)];
    let mapped_cy = m[(1, 0)] * cx + m[(1, 1)] * cy + m[(1, 2)];
    assert!(
        (mapped_cx - cx).abs() < 1e-4,
        "centre x should be fixed: got {mapped_cx}, expected {cx}"
    );
    assert!(
        (mapped_cy - cy).abs() < 1e-4,
        "centre y should be fixed: got {mapped_cy}, expected {cy}"
    );
}

/// **Scale recovery test** — synthesize a breathing transform at a known
/// scale about a known centre, feed the exact correspondences to
/// `estimate_from_pairs`, and assert the recovered scale is within 0.1 %
/// of the ground truth.
#[test]
fn test_breathing_scale_recovery_exact_correspondences() {
    let cx = 200.0_f32;
    let cy = 150.0_f32;
    let known_scale = 1.03_f32; // 3 % breathing — typical for macro stacks
    let known_tx = 2.0_f32; // small residual translation
    let known_ty = -1.5_f32;

    // Build synthetic correspondences: dst = s*(src - center) + center + (tx,ty)
    let src_points: &[(f32, f32)] = &[
        (50.0, 60.0),
        (300.0, 200.0),
        (100.0, 250.0),
        (350.0, 80.0),
        (180.0, 170.0),
        (220.0, 130.0),
        (80.0, 120.0),
        (320.0, 280.0),
    ];
    let pairs: Vec<(f32, f32, f32, f32)> = src_points
        .iter()
        .map(|&(sx, sy)| {
            let dx = known_scale * (sx - cx) + cx + known_tx;
            let dy = known_scale * (sy - cy) + cy + known_ty;
            (sx, sy, dx, dy)
        })
        .collect();

    let est = BreathingCorrector::estimate_from_pairs(&pairs, cx, cy)
        .expect("should converge on exact correspondences");

    let scale_err = (est.scale - known_scale).abs();
    assert!(
        scale_err < 0.001,
        "scale error {scale_err:.6} exceeds 0.001 (0.1 %); recovered={:.5}, expected={known_scale}",
        est.scale
    );
    let tx_err = (est.tx - known_tx).abs();
    assert!(
        tx_err < 0.1,
        "tx residual error {tx_err:.4} exceeds 0.1 px; recovered={:.4}, expected={known_tx}",
        est.tx
    );
    let ty_err = (est.ty - known_ty).abs();
    assert!(
        ty_err < 0.1,
        "ty residual error {ty_err:.4} exceeds 0.1 px; recovered={:.4}, expected={known_ty}",
        est.ty
    );
}

/// **Warp-quality test (non-tautological)** — synthesize a reference
/// image and a target image that is the reference scaled by a *known*
/// breathing magnification about the image centre.  Measure pixel-level
/// misalignment (MSE against the reference) for two cases:
///
/// 1. **No correction** (use target as-is) — expected higher MSE.
/// 2. **Breathing correction applied** (warp target by recovered matrix)
///    — expected lower MSE, strictly less than case 1.
///
/// ## Correspondence convention
///
/// `estimate_from_pairs` takes `(src_x, src_y, dst_x, dst_y)` tuples and
/// estimates the forward map `src → dst`.  `warp_image(img, M)` applies
/// the *inverse* of `M` as the source-lookup (standard inverse-mapping).
///
/// So to correct `target` back toward `reference` we need:
///
/// - Pairs as `(target_pt, ref_pt)` — i.e. *src = target, dst = reference*.
/// - Estimator returns `M` mapping `target → reference` (scale ≈ 1/s).
/// - `warp_image(target, M)` inverts `M` (scale ≈ s), sampling
///   `target[s*(p − c) + c]` = `reference[p]`.  Correct.
///
/// This mirrors the pipeline: AKAZE gives `(ref_kp, tgt_kp)` pairs, so
/// `estimate_from_pairs` returns the forward `ref → tgt` map, and
/// `warp_image(target_frame, M_fwd)` applies `M_fwd⁻¹` to bring the
/// target into alignment with the reference.
///
/// This test is non-tautological because:
/// - The recovered matrix is derived from exact point correspondences,
///   not by sampling the pixel data.
/// - The warp uses bilinear interpolation (a real resampling step).
/// - The assertion is relative (corrected < uncorrected), not circular.
#[test]

fn test_breathing_warp_reduces_misalignment() {
    const W: usize = 64;
    const H: usize = 64;
    // Border margin for interior-only MSE comparison (see comment below).
    const MARGIN: usize = 4;
    let cx = W as f32 / 2.0;
    let cy = H as f32 / 2.0;
    let breathing_scale = 1.04_f32; // 4 % — clearly visible but realistic

    // Build reference image: smooth gradient that is NOT axis-symmetric
    // so a misaligned warp produces measurable MSE.
    let mut reference = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            reference.luma[i] = ((x as f32 * 0.12).sin() + (y as f32 * 0.09).cos()).abs() * 0.5;
        }
    }

    // Build the "breathing" target: reference scaled outward by
    // `breathing_scale` about the centre (zoom-in effect).
    // target[p] = reference[p_src]  where  p_src = (p - c)/s + c
    // i.e. inverse-map: each target pixel comes from a *closer-to-centre*
    // reference pixel.
    let mut target = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let i = y * W + x;
            let src_x = (x as f32 - cx) / breathing_scale + cx;
            let src_y = (y as f32 - cy) / breathing_scale + cy;
            let rx = src_x.round() as isize;
            let ry = src_y.round() as isize;
            if rx >= 0 && ry >= 0 && (rx as usize) < W && (ry as usize) < H {
                target.luma[i] = reference.luma[ry as usize * W + rx as usize];
            }
        }
    }

    // Build correspondences as (target_pt → ref_pt) so the estimator
    // returns the CORRECTION matrix (target → reference, scale ≈ 1/s).
    // Calling warp_image(target, M_corr) then inverts M_corr, effectively
    // applying scale ≈ s — expanding the target back to match the reference.
    //
    // target_pt = s*(ref_pt - c) + c  →  ref_pt = (target_pt - c)/s + c
    //
    // So for src=target, dst=ref:
    //   src_x = s*(ref_x - cx) + cx  →  ref_x = (src_x - cx)/s + cx
    let key_points: &[(f32, f32)] = &[
        (8.0, 8.0),
        (56.0, 8.0),
        (8.0, 56.0),
        (56.0, 56.0),
        (32.0, 16.0),
        (16.0, 48.0),
        (48.0, 32.0),
        (24.0, 24.0),
    ];
    let pairs: Vec<(f32, f32, f32, f32)> = key_points
        .iter()
        .map(|&(ref_x, ref_y)| {
            // target location of the reference KeyPoint
            let tgt_x = breathing_scale * (ref_x - cx) + cx;
            let tgt_y = breathing_scale * (ref_y - cy) + cy;
            // src = target_pt, dst = ref_pt
            (tgt_x, tgt_y, ref_x, ref_y)
        })
        .collect();

    // Estimate correction matrix: target → reference (scale ≈ 1/s).
    let est = BreathingCorrector::estimate_from_pairs(&pairs, cx, cy)
        .expect("estimation should succeed on exact correspondences");
    let m_corr = BreathingCorrector::build_matrix(&est, cx, cy);

    // Sanity: recovered scale should be ≈ 1/breathing_scale.
    let expected_inv_scale = 1.0 / breathing_scale;
    assert!(
        (est.scale - expected_inv_scale).abs() < 0.002,
        "correction scale {:.5} should be ≈ 1/s = {expected_inv_scale:.5}",
        est.scale
    );

    // Apply correction: warp_image inverts M_corr, so it samples
    // target at scale*s ≈ 1 — recovering the reference content.
    let corrected = warp_image_clamped(&target, &m_corr).expect("warp must succeed");

    // Compare only the interior region (MARGIN-pixel border excluded) that
    // is valid in both images.  `target` has zero-fill at the outer border
    // because the breathing pushes source coordinates out of bounds during
    // construction; `corrected` may sample those zeros through the warp.
    // With 4 % breathing on 64 px the affected zone is ≈ 2 px; MARGIN=4
    // is a conservative guard.
    let mut mse_uncorrected = 0.0_f32;
    let mut mse_corrected = 0.0_f32;
    let mut count = 0usize;
    for y in MARGIN..(H - MARGIN) {
        for x in MARGIN..(W - MARGIN) {
            let i = y * W + x;
            let r = reference.luma[i];
            let t = target.luma[i];
            let c = corrected.luma[i];
            mse_uncorrected += (r - t) * (r - t);
            mse_corrected += (r - c) * (r - c);
            count += 1;
        }
    }
    mse_uncorrected /= count as f32;
    mse_corrected /= count as f32;

    assert!(
        mse_corrected < mse_uncorrected,
        "breathing correction must reduce interior MSE: \
         corrected={mse_corrected:.6} uncorrected={mse_uncorrected:.6}"
    );
    // The correction should reduce error substantially — at least 50 %.
    assert!(
        mse_corrected < mse_uncorrected * 0.5,
        "breathing correction must reduce interior MSE by at least 50 %: \
         corrected={mse_corrected:.6} (>{:.6}), uncorrected={mse_uncorrected:.6}",
        mse_uncorrected * 0.5
    );
}

/// **Regression test: single-application alignment scale (Option B fix)**
///
/// Verifies that the pipeline's new affine-only alignment path applies the
/// geometric correction exactly ONCE — not twice as in the previous buggy
/// code that composed `affine_matrix * breathing_matrix` where BOTH matrices
/// were estimated from the same raw `KeyPoint` correspondences.
///
/// ## Setup
///
/// Synthesize a reference→target breathing transform at a known scale
/// `s = 1.05` about the image centre.  Build point correspondences as the
/// pipeline does: `(ref_kp, target_kp)` pairs, i.e. `src = ref, dst = target`.
///
/// ## What is asserted
///
/// 1. **Single-application (correct):** `AlignmentEstimator::compute_matrix`
///    on the raw correspondences returns a matrix whose effective scale
///    ≈ `s = 1.05` (the forward ref→target magnification).  Warping the
///    target by this matrix (which internally inverts it) applies scale `1/s`
///    — a single, correct correction.
///
/// 2. **Double-application (old bug):** composing
///    `affine_matrix * breathing_matrix`, where both were estimated from the
///    same raw pairs, yields a matrix whose effective scale ≈ `s²`.  The test
///    explicitly asserts this to document what the bug produced.
///
/// 3. The single-application scale `s` must NOT equal `s²` within the tight
///    tolerance, confirming that the two code paths produce detectably
///    different results.
#[test]

fn test_affine_only_alignment_applies_correction_once_not_twice() {
    let cx = 200.0_f32;
    let cy = 150.0_f32;
    // Known forward breathing scale: target is magnified by s about (cx, cy).
    let s = 1.05_f32;

    // Build correspondences: src = reference KeyPoint, dst = target KeyPoint.
    // dst_x = s*(src_x - cx) + cx,  dst_y = s*(src_y - cy) + cy
    let src_points: &[(f32, f32)] = &[
        (50.0, 60.0),
        (300.0, 200.0),
        (100.0, 250.0),
        (350.0, 80.0),
        (180.0, 170.0),
        (220.0, 130.0),
        (80.0, 120.0),
        (320.0, 280.0),
        (160.0, 90.0),
        (240.0, 210.0),
        (60.0, 200.0),
        (360.0, 100.0),
    ];
    let pairs: Vec<(f32, f32, f32, f32)> = src_points
        .iter()
        .map(|&(sx, sy)| {
            let dx = s * (sx - cx) + cx;
            let dy = s * (sy - cy) + cy;
            (sx, sy, dx, dy)
        })
        .collect();

    let (kps0, kps1, matches) = build_correspondences(&pairs);

    // ── Path 1 (Option B — the fix): affine RANSAC on raw correspondences ─
    let affine_matrix =
        AlignmentEstimator::compute_matrix(&matches, &kps0, &kps1, AlignmentMode::Affine)
            .expect("affine RANSAC must converge on exact correspondences");

    // The affine matrix maps ref→target with scale ≈ s.
    // Its diagonal entries m[0,0] and m[1,1] encode the effective scale
    // (for a pure centre-anchored similarity, both equal s).
    let affine_scale_x = affine_matrix[(0, 0)];
    let affine_scale_y = affine_matrix[(1, 1)];
    // Average diagonal as proxy for the uniform scale component.
    let affine_scale = (affine_scale_x + affine_scale_y) * 0.5;

    // Single application: recovered scale must be ≈ s (not s²).
    let single_err = (affine_scale - s).abs();
    assert!(
        single_err < 0.01,
        "affine-only scale {affine_scale:.5} should be ≈ s={s:.5} (single correction); \
         error={single_err:.5}"
    );

    // ── Path 2 (old bug): compose affine_matrix * breathing_matrix ────────
    // Both were estimated from the same raw pairs, so both approximate the
    // same full forward transform.  The composition approximates s².
    let breathing_est = BreathingCorrector::estimate_from_pairs(&pairs, cx, cy)
        .expect("breathing estimator must converge on exact correspondences");
    let breathing_matrix = BreathingCorrector::build_matrix(&breathing_est, cx, cy);

    let composed_buggy = affine_matrix * breathing_matrix;
    // For a pure-scale similarity, the composed scale ≈ s * s = s².
    let composed_scale_x = composed_buggy[(0, 0)];
    let composed_scale_y = composed_buggy[(1, 1)];
    let composed_scale = (composed_scale_x + composed_scale_y) * 0.5;

    let s_sq = s * s;
    let double_err = (composed_scale - s_sq).abs();
    assert!(
        double_err < 0.02,
        "composed (buggy) scale {composed_scale:.5} should be ≈ s²={s_sq:.5} \
         (double correction); error={double_err:.5}"
    );

    // ── The two paths must be detectably different ──────────────────────
    // s=1.05, s²=1.1025 — difference is 0.0525, well above any rounding.
    let path_diff = (affine_scale - composed_scale).abs();
    assert!(
        path_diff > 0.04,
        "single-correction scale ({affine_scale:.5}) and double-correction scale \
         ({composed_scale:.5}) must differ by > 0.04; got diff={path_diff:.5}"
    );

    // Explicit confirmation: the single path does NOT produce s².
    assert!(
        (affine_scale - s_sq).abs() > 0.04,
        "affine-only scale ({affine_scale:.5}) must NOT be ≈ s²={s_sq:.5} \
         (that would indicate the double-apply bug is still present)"
    );
}

/// **Regression test for the 1-inlier acceptance bug.** `TranslationOnly`'s
/// minimal RANSAC sample is a single correspondence, so before the
/// `min_inliers` support requirement was added, a model fitted to exactly
/// one spurious match could be accepted outright. With 20 pairs where 18
/// are consistent with a known translation and 2 are wild outliers, the
/// estimator must recover the true translation using the majority support.
#[test]
fn test_translation_only_recovers_with_majority_inliers_and_two_outliers() {
    let tx = 12.0_f32;
    let ty = -7.0_f32;

    let mut raw: Vec<(f32, f32, f32, f32)> = (0..18_usize)
        .map(|i| {
            let sx = 10.0 + i as f32 * 14.0;
            let sy = 20.0 + (i % 6) as f32 * 17.0;
            (sx, sy, sx + tx, sy + ty)
        })
        .collect();

    // 2 wild outliers, inconsistent with the translation model.
    raw.push((15.0, 15.0, 400.0, -300.0));
    raw.push((60.0, 200.0, -250.0, 500.0));

    let (kps0, kps1, matches) = build_correspondences(&raw);

    let m =
        AlignmentEstimator::compute_matrix(&matches, &kps0, &kps1, AlignmentMode::TranslationOnly)
            .expect("should recover the majority-supported translation despite 2 outliers");

    let err_x = (m[(0, 2)] - tx).abs();
    let err_y = (m[(1, 2)] - ty).abs();
    assert!(err_x < 0.5, "tx error {err_x} too large");
    assert!(err_y < 0.5, "ty error {err_y} too large");
}

/// **Regression test for the 1-inlier acceptance bug (negative case).**
/// With 20 pairs of pure random noise (no consistent translation model
/// underlies the data), `TranslationOnly` must return `Err` rather than
/// accepting whatever single-match translation happened to win the RANSAC
/// loop. This is the core regression test: before the `min_inliers` support
/// requirement, a `TranslationOnly` model needs support from only 1 match
/// (its own minimal sample) to be accepted, so pure noise would still
/// spuriously "succeed".
#[test]
fn test_translation_only_rejects_pure_noise() {
    // Deterministic pseudo-random destinations that share no consistent
    // translation model with the sources (each pair's implied translation
    // is wildly different from every other pair's).
    let raw: Vec<(f32, f32, f32, f32)> = (0..20_usize)
        .map(|i| {
            let fi = i as f32;
            let sx = 10.0 + fi * 13.7;
            let sy = 20.0 + (fi * 7.3) % 200.0;
            // Destination jitter is large and non-uniform across pairs, so
            // no single translation explains more than a handful of them.
            let dx = (fi * 97.0) % 500.0;
            let dy = (fi * 53.0) % 400.0;
            (sx, sy, dx, dy)
        })
        .collect();

    let (kps0, kps1, matches) = build_correspondences(&raw);

    let result =
        AlignmentEstimator::compute_matrix(&matches, &kps0, &kps1, AlignmentMode::TranslationOnly);
    assert!(
        matches!(result, Err(StackerError::AlignmentFailed(_))),
        "pure noise correspondences must not produce an accepted translation model, got {result:?}"
    );
}

/// Sanity check: fewer than 2 correspondences must return an error.
#[test]
fn test_breathing_too_few_pairs_errors() {
    let result = BreathingCorrector::estimate_from_pairs(&[], 100.0, 100.0);
    assert!(
        matches!(result, Err(StackerError::AlignmentFailed(_))),
        "zero pairs must return AlignmentFailed"
    );
    let one = vec![(10.0_f32, 20.0, 15.0, 25.0)];
    let result2 = BreathingCorrector::estimate_from_pairs(&one, 100.0, 100.0);
    assert!(
        matches!(result2, Err(StackerError::AlignmentFailed(_))),
        "one pair must return AlignmentFailed"
    );
}

/// **RANSAC robustness test** — inject 35 % gross outliers into the
/// correspondence set and verify that the RANSAC breathing estimator
/// still recovers the true scale within a tight tolerance, while also
/// demonstrating (by assertion) that plain least-squares over the full
/// contaminated set is pulled off.
///
/// ## Why this test is non-tautological
///
/// - The inlier set is generated from a *known* ground-truth transform; the
///   outliers are independently random.
/// - The RANSAC assertion is tight (±0.002) — far less than the bias
///   introduced by the outliers.
/// - The plain-LSQ assertion checks the *other direction*: it must deviate
///   by more than the RANSAC tolerance, confirming the outliers do pollute
///   the naïve solution and the RANSAC is doing real work.
#[test]

fn test_breathing_ransac_robust_to_outliers() {
    let cx = 200.0_f32;
    let cy = 150.0_f32;
    let known_scale = 1.04_f32; // 4 % breathing
    let known_tx = 3.0_f32;
    let known_ty = -2.0_f32;

    // Build 20 clean inlier correspondences spread across the image.
    let mut pairs: Vec<(f32, f32, f32, f32)> = (0..20_usize)
        .map(|k| {
            let sx = 20.0 + (k as f32) * 17.0;
            let sy = 30.0 + (k as f32) * 11.0;
            let dx = known_scale * (sx - cx) + cx + known_tx;
            let dy = known_scale * (sy - cy) + cy + known_ty;
            (sx, sy, dx, dy)
        })
        .collect();

    // Inject 11 gross outliers ≈ 35 % of total (11/31).
    // Outlier displacements are 80-200 px — far outside the 3-px inlier threshold.
    let outlier_offsets: &[(f32, f32)] = &[
        (120.0, 90.0),
        (-150.0, 80.0),
        (200.0, -110.0),
        (-80.0, 130.0),
        (170.0, -60.0),
        (-200.0, -90.0),
        (95.0, 185.0),
        (-130.0, -170.0),
        (85.0, 95.0),
        (-100.0, 75.0),
        (160.0, 140.0),
    ];
    for (i, &(ox, oy)) in outlier_offsets.iter().enumerate() {
        let sx = 40.0 + i as f32 * 13.0;
        let sy = 50.0 + i as f32 * 9.0;
        // Destination is wildly off — a gross outlier.
        pairs.push((sx, sy, sx + ox, sy + oy));
    }

    // ── RANSAC estimator ──────────────────────────────────────────────
    let ransac_est = BreathingCorrector::estimate_from_pairs(&pairs, cx, cy)
        .expect("RANSAC should converge with 65 % inliers");

    let ransac_scale_err = (ransac_est.scale - known_scale).abs();
    assert!(
        ransac_scale_err < 0.002,
        "RANSAC scale error {ransac_scale_err:.5} exceeds 0.002; \
         recovered={:.5}, truth={known_scale}",
        ransac_est.scale
    );

    // ── Plain least-squares over the contaminated full set ────────────
    // Build the raw (s, bx, by) directly without centre unwrapping to
    // test whether outliers actually bias the naïve fit.
    // We call fit_breathing (private) via the public API: call
    // estimate_from_pairs on the same contaminated set but with cx=cy=0
    // so bx=tx_residual, by=ty_residual, making scale directly comparable.
    //
    // Note: with cx=cy=0 the breathing formula collapses to a simple
    // affine scale+translate, so the `scale` field IS the LSQ scale
    // without any centre-anchor subtraction.
    let lsq_est_raw = {
        // Use cx=0, cy=0 — now bx=tx_total, by=ty_total.
        // We only care about the `scale` field being biased.
        // Manually replicate the plain LSQ formula over all pairs (no RANSAC).
        let n = pairs.len();
        let rows = 2 * n;
        let mut a = nalgebra::DMatrix::<f32>::zeros(rows, 3);
        let mut b_vec = nalgebra::DVector::<f32>::zeros(rows);
        for (i, &(sx, sy, dx, dy)) in pairs.iter().enumerate() {
            a[(2 * i, 0)] = sx;
            a[(2 * i, 1)] = 1.0;
            b_vec[2 * i] = dx;
            a[(2 * i + 1, 0)] = sy;
            a[(2 * i + 1, 2)] = 1.0;
            b_vec[2 * i + 1] = dy;
        }
        let ata = a.transpose() * &a;
        let atb = a.transpose() * &b_vec;
        ata.lu()
            .solve(&atb)
            .expect("plain LSQ must not be degenerate")
    };
    let lsq_scale = lsq_est_raw[0];
    let lsq_scale_err = (lsq_scale - known_scale).abs();

    // The plain LSQ scale MUST be more biased than the RANSAC result.
    assert!(
        lsq_scale_err > ransac_scale_err,
        "plain LSQ scale error ({lsq_scale_err:.5}) should exceed RANSAC error \
         ({ransac_scale_err:.5}), confirming outliers bias the naïve fit"
    );
    // Additionally, the LSQ error should be measurably larger than the
    // RANSAC tolerance — confirming the 35 % outlier rate actually contaminates it.
    assert!(
        lsq_scale_err > 0.003,
        "plain LSQ scale error ({lsq_scale_err:.5}) is suspiciously small; \
         outliers should pull it beyond 0.003"
    );
}
