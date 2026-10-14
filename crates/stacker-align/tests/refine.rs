#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "fixture sizes and shifts are tiny, so index to float conversions are exact and in range"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "the fixtures spell out the reference formulas as plain multiply-add"
)]
#![expect(
    clippy::similar_names,
    reason = "reference/source and x/y pairs are named in parallel"
)]

use nalgebra::Matrix3;
use stacker_align::refine::*;
use stacker_core::{error::StackerError, image::PlanarImage};

// ── helpers ──────────────────────────────────────────────────────────────

/// Smooth, non-symmetric gradient image in [0, 1].
fn make_test_image(width: usize, height: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            img.luma[i] = ((x as f32 * 0.11).sin() + (y as f32 * 0.09).cos())
                .mul_add(0.4, 0.5)
                .clamp(0.0, 1.0);
        }
    }
    img
}

/// Backward-compatible thin wrapper: evaluate objective at full resolution.
///
/// Used by the test suite.
fn objective(
    params_vec: &[f64; 4],
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
) -> f64 {
    objective_at_level(
        params_vec,
        &reference.luma,
        &source.luma,
        reference.width,
        reference.height,
    )
}

// ── params_to_matrix / matrix_to_params round-trip ───────────────────────

#[test]
fn test_params_to_matrix_identity() {
    let p = RegistrationParams::identity();
    let m = params_to_matrix(&p, 100, 100).expect("identity must produce a matrix");
    let diff = (m - Matrix3::<f32>::identity()).norm();
    assert!(
        diff < 1.0e-4,
        "identity params must yield the identity matrix; diff={diff:.6}"
    );
}

#[test]
fn test_params_round_trip() {
    let orig = RegistrationParams {
        tx: 0.05,
        ty: -0.03,
        scale: 1.02,
        rotate: 0.01,
    };
    let m = params_to_matrix(&orig, 200, 150).expect("must build matrix");
    let recovered = matrix_to_params(&m, 200, 150);
    let err_tx = (recovered.tx - orig.tx).abs();
    let err_ty = (recovered.ty - orig.ty).abs();
    let err_scale = (recovered.scale - orig.scale).abs();
    let err_rotate = (recovered.rotate - orig.rotate).abs();
    assert!(err_tx < 1.0e-5, "tx round-trip error {err_tx:.8}");
    assert!(err_ty < 1.0e-5, "ty round-trip error {err_ty:.8}");
    assert!(err_scale < 1.0e-5, "scale round-trip error {err_scale:.8}");
    assert!(
        err_rotate < 1.0e-5,
        "rotate round-trip error {err_rotate:.8}"
    );
}

#[test]
fn test_params_to_matrix_nan_returns_none() {
    let p = RegistrationParams {
        tx: f64::NAN,
        ty: 0.0,
        scale: 1.0,
        rotate: 0.0,
    };
    assert!(params_to_matrix(&p, 100, 100).is_none());
}

// ── rms_difference ───────────────────────────────────────────────────────

#[test]
fn test_rms_difference_identical_planes_is_zero() {
    let plane: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
    let rms = rms_difference(&plane, &plane, 8, 8);
    assert!(
        rms < 1.0e-8,
        "identical planes must have RMS ≈ 0; got {rms:.3e}"
    );
}

#[test]
fn test_rms_difference_global_offset_is_zero() {
    // A global brightness offset between the two planes must not affect
    // the mean-bias-corrected RMS (Java subtracts the mean before squaring).
    let plane: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
    let shifted: Vec<f32> = plane.iter().map(|&v| v + 0.2_f32).collect();
    let rms = rms_difference(&plane, &shifted, 8, 8);
    assert!(
        rms < 1.0e-6,
        "global brightness offset must not affect mean-bias-corrected RMS; got {rms:.3e}"
    );
}

// ── objective / warp ─────────────────────────────────────────────────────

#[test]
fn test_objective_identity_identical_images_is_zero() {
    let img = make_test_image(32, 32);
    let p = RegistrationParams::identity().to_vec();
    let rms = objective(&p, &img, &img);
    assert!(
        rms < 1.0e-5,
        "RMS of identity warp on identical images must be ≈ 0; got {rms:.3e}"
    );
}

// ── refine_alignment: guard tests ────────────────────────────────────────

#[test]
fn test_refine_dimension_mismatch_errors() {
    let ref_img = PlanarImage::<f32>::new(32, 32);
    let src_img = PlanarImage::<f32>::new(64, 64);
    let result = refine_alignment(
        &ref_img,
        &src_img,
        &Matrix3::identity(),
        &RefineOptions::default(),
    );
    assert!(
        matches!(result, Err(StackerError::AlignmentFailed(_))),
        "dimension mismatch must return AlignmentFailed"
    );
}

#[test]
fn test_refine_nan_initial_matrix_errors() {
    let img = PlanarImage::<f32>::new(32, 32);
    let mut bad = Matrix3::<f32>::identity();
    bad[(0, 0)] = f32::NAN;
    let result = refine_alignment(&img, &img, &bad, &RefineOptions::default());
    assert!(
        matches!(result, Err(StackerError::MathError(_))),
        "non-finite initial matrix must return MathError"
    );
}

// ── refine_alignment: convergence tests (non-tautological) ───────────────

/// Core convergence test: subpixel translation recovery.
///
/// ## Setup
/// 1. Build a reference image with smooth sinusoidal texture.
/// 2. Create a source that is the reference shifted by a known subpixel
///    translation `(true_tx, true_ty)`.
/// 3. Give `refine_alignment` an initial guess that is 0.3 px off the
///    true translation — a realistic coarse-pass residual.
///
/// ## Assertions
/// - The refined translation is within 0.15 px of the ground truth
///   (tighter than the 0.3 px initial error).
/// - The RMS of the refined warp is strictly less than the RMS of the
///   coarse initial warp, proving the pass did real work.
#[test]
fn test_refine_subpixel_translation_converges() {
    const W: usize = 64;
    const H: usize = 64;

    let reference = make_test_image(W, H);

    let true_tx = 0.7_f32;
    let true_ty = -0.5_f32;

    // Build source by genuine subpixel sampling with the same bicubic
    // kernel the optimiser uses. A nearest-neighbour `round()` shift cannot
    // represent a sub-pixel translation, which would make the < 0.15 px
    // assertion meaningless. `source[x,y] = reference(x - tx, y - ty)`
    // makes the forward translation (tx, ty) the exact RMS minimum.
    let mut source = PlanarImage::<f32>::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let sx = x as f32 - true_tx;
            let sy = y as f32 - true_ty;
            source.luma[y * W + x] = stacker_align::transform::bicubic_interpolate_channel(
                &reference.luma,
                W,
                H,
                sx,
                sy,
            );
        }
    }

    // Coarse initial guess: 0.3 px off in each axis.
    let coarse_tx = (true_tx - 0.3) / W as f32;
    let coarse_ty = (true_ty + 0.3) / H as f32;
    let p_coarse = RegistrationParams {
        tx: f64::from(coarse_tx),
        ty: f64::from(coarse_ty),
        scale: 1.0,
        rotate: 0.0,
    };
    let initial_matrix = params_to_matrix(&p_coarse, W, H).expect("coarse matrix must be valid");

    let opts = RefineOptions::default();
    let refined = refine_alignment(&reference, &source, &initial_matrix, &opts)
        .expect("refinement must succeed on well-conditioned data");

    let p_refined = matrix_to_params(&refined, W, H);
    let refined_tx_px = (p_refined.tx * W as f64) as f32;
    let refined_ty_px = (p_refined.ty * H as f64) as f32;

    let err_tx = (refined_tx_px - true_tx).abs();
    let err_ty = (refined_ty_px - true_ty).abs();

    assert!(
        err_tx < 0.15,
        "refined tx error {err_tx:.4} px should be < 0.15 px; \
         truth={true_tx}, got={refined_tx_px:.4}"
    );
    assert!(
        err_ty < 0.15,
        "refined ty error {err_ty:.4} px should be < 0.15 px; \
         truth={true_ty}, got={refined_ty_px:.4}"
    );

    // RMS must strictly decrease.
    let rms_coarse = objective(&p_coarse.to_vec(), &reference, &source);
    let rms_refined = objective(&p_refined.to_vec(), &reference, &source);
    assert!(
        rms_refined < rms_coarse,
        "refined RMS ({rms_refined:.6}) must be < coarse RMS ({rms_coarse:.6})"
    );
}

/// Translation-only mode (scale+rotate disabled): still converges.
#[test]
fn test_refine_translation_only_mode_converges() {
    const W: usize = 64;
    const H: usize = 64;

    let reference = make_test_image(W, H);

    let true_tx = -1.2_f32;
    let true_ty = 0.8_f32;

    // Subpixel-accurate source (see note in the previous test).
    let mut source = PlanarImage::<f32>::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let sx = x as f32 - true_tx;
            let sy = y as f32 - true_ty;
            source.luma[y * W + x] = stacker_align::transform::bicubic_interpolate_channel(
                &reference.luma,
                W,
                H,
                sx,
                sy,
            );
        }
    }

    let opts = RefineOptions {
        allow_scale: false,
        allow_rotate: false,
        ..Default::default()
    };
    let refined = refine_alignment(&reference, &source, &Matrix3::identity(), &opts)
        .expect("translation-only refinement must succeed");

    let p_refined = matrix_to_params(&refined, W, H);
    let refined_tx_px = (p_refined.tx * W as f64) as f32;
    let refined_ty_px = (p_refined.ty * H as f64) as f32;

    let err_tx = (refined_tx_px - true_tx).abs();
    let err_ty = (refined_ty_px - true_ty).abs();

    assert!(
        err_tx < 0.15,
        "translation-only: tx error {err_tx:.4} px; truth={true_tx}, got={refined_tx_px:.4}"
    );
    assert!(
        err_ty < 0.15,
        "translation-only: ty error {err_ty:.4} px; truth={true_ty}, got={refined_ty_px:.4}"
    );
}

// ── pyramid tests ─────────────────────────────────────────────────────────

/// Verify that the pyramid builder produces correct level counts and
/// dimensions for a typical image size.
#[test]
fn test_build_luma_pyramid_level_count_and_dims() {
    // 128×96: short side = 96.
    // Level 0 (full): 128×96  — short side 96 / 2 = 48 ≥ 32 → add level 1
    // Level 1 (½):     64×48  — short side 48 / 2 = 24 < 32 → stop
    // Pyramid (coarse→fine): [64×48, 128×96]  → 2 levels
    let img = make_test_image(128, 96);
    let pyramid = build_luma_pyramid(&img);
    assert_eq!(
        pyramid.len(),
        2,
        "128×96 image should yield 2 pyramid levels"
    );

    // Coarsest level (index 0) is the half-res level.
    assert_eq!(pyramid[0].width, 64);
    assert_eq!(pyramid[0].height, 48);
    assert_eq!(pyramid[0].luma.len(), 64 * 48);

    // Finest level (last) is full resolution.
    let finest = pyramid.last().expect("non-empty");
    assert_eq!(finest.width, 128);
    assert_eq!(finest.height, 96);
    assert_eq!(finest.luma.len(), 128 * 96);
}

/// Very small image: pyramid should degenerate to a single level.
#[test]
fn test_build_luma_pyramid_small_image_single_level() {
    // 32×32: short side 32.  32 / 2 = 16 < PYRAMID_MIN_SIZE(32) → stop immediately.
    let img = make_test_image(32, 32);
    let pyramid = build_luma_pyramid(&img);
    assert_eq!(
        pyramid.len(),
        1,
        "32×32 image must yield exactly 1 pyramid level"
    );
    assert_eq!(pyramid[0].width, 32);
    assert_eq!(pyramid[0].height, 32);
}

/// Pyramid with a larger image: verify a 3-level pyramid forms correctly.
#[test]
fn test_build_luma_pyramid_three_levels() {
    // 256×256: short side = 256.
    // 256/2 = 128 ≥ 32 → level 1 (128×128)
    // 128/2 =  64 ≥ 32 → level 2  (64×64)
    //  64/2 =  32 ≥ 32 → level 3  (32×32)
    //  32/2 =  16 < 32 → stop
    // Pyramid (coarse→fine): [32×32, 64×64, 128×128, 256×256]  → 4 levels
    let img = make_test_image(256, 256);
    let pyramid = build_luma_pyramid(&img);
    assert_eq!(
        pyramid.len(),
        4,
        "256×256 image should yield 4 pyramid levels"
    );
    assert_eq!(pyramid[0].width, 32);
    assert_eq!(pyramid[0].height, 32);
    assert_eq!(pyramid[1].width, 64);
    assert_eq!(pyramid[1].height, 64);
    assert_eq!(pyramid[2].width, 128);
    assert_eq!(pyramid[2].height, 128);
    assert_eq!(pyramid[3].width, 256);
    assert_eq!(pyramid[3].height, 256);
}

// ── refine_alignment_bounded: guard tests ────────────────────────────────

#[test]
fn test_refine_bounded_dimension_mismatch_errors() {
    let ref_img = PlanarImage::<f32>::new(32, 32);
    let src_img = PlanarImage::<f32>::new(64, 64);
    let result = stacker_align::refine_alignment_bounded(
        &ref_img,
        &src_img,
        &Matrix3::identity(),
        &stacker_align::BoundedRefineOptions::default(),
    );
    assert!(
        matches!(result, Err(StackerError::AlignmentFailed(_))),
        "bounded: dimension mismatch must return AlignmentFailed"
    );
}

#[test]
fn test_refine_bounded_nan_initial_matrix_errors() {
    let img = PlanarImage::<f32>::new(32, 32);
    let mut bad = Matrix3::<f32>::identity();
    bad[(0, 0)] = f32::NAN;
    let result = stacker_align::refine_alignment_bounded(
        &img,
        &img,
        &bad,
        &stacker_align::BoundedRefineOptions::default(),
    );
    assert!(
        matches!(result, Err(StackerError::MathError(_))),
        "bounded: non-finite initial matrix must return MathError"
    );
}

// ── refine_alignment_bounded: convergence tests ──────────────────────────

/// Bounded mode on identical images (identity initial): must return Ok with
/// a finite matrix close to identity (no transform needed).
#[test]
fn test_refine_bounded_identical_images_returns_ok_near_identity() {
    const W: usize = 64;
    const H: usize = 64;
    let img = make_test_image(W, H);

    let opts = stacker_align::BoundedRefineOptions::default();
    let result = stacker_align::refine_alignment_bounded(&img, &img, &Matrix3::identity(), &opts);
    let refined = result.expect("bounded refinement on identical images must succeed");

    // All entries must be finite.
    assert!(
        refined.iter().all(|v| v.is_finite()),
        "bounded: refined matrix must be fully finite"
    );

    // Must be close to identity (within ±0.05 on diagonal, ±2 px on translation).
    let diff = (refined - Matrix3::<f32>::identity()).norm();
    assert!(
        diff < 0.15,
        "bounded: identical images → refined matrix should be near identity; diff={diff:.4}"
    );
}

/// Bounded mode recovers a small known subpixel shift within the default ±5 % bounds.
#[test]
fn test_refine_bounded_subpixel_translation_converges() {
    const W: usize = 64;
    const H: usize = 64;

    let reference = make_test_image(W, H);

    let true_tx = 0.6_f32;
    let true_ty = -0.4_f32;

    // Build source via bicubic subpixel shift (same method as the unbounded test).
    let mut source = PlanarImage::<f32>::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let sx = x as f32 - true_tx;
            let sy = y as f32 - true_ty;
            source.luma[y * W + x] = stacker_align::transform::bicubic_interpolate_channel(
                &reference.luma,
                W,
                H,
                sx,
                sy,
            );
        }
    }

    let opts = stacker_align::BoundedRefineOptions::default();
    let refined =
        stacker_align::refine_alignment_bounded(&reference, &source, &Matrix3::identity(), &opts)
            .expect("bounded refinement must succeed on well-conditioned data");

    let p_refined = matrix_to_params(&refined, W, H);
    let refined_tx_px = (p_refined.tx * W as f64) as f32;
    let refined_ty_px = (p_refined.ty * H as f64) as f32;

    let err_tx = (refined_tx_px - true_tx).abs();
    let err_ty = (refined_ty_px - true_ty).abs();

    // RMS must also decrease vs identity.
    let rms_identity = objective(
        &RegistrationParams::identity().to_vec(),
        &reference,
        &source,
    );
    let rms_refined = objective(&p_refined.to_vec(), &reference, &source);
    assert!(
        rms_refined < rms_identity,
        "bounded: refined RMS ({rms_refined:.6}) must be < identity RMS ({rms_identity:.6})"
    );

    assert!(
        err_tx < 0.20,
        "bounded: tx error {err_tx:.4} px should be < 0.20 px; \
         truth={true_tx}, got={refined_tx_px:.4}"
    );
    assert!(
        err_ty < 0.20,
        "bounded: ty error {err_ty:.4} px should be < 0.20 px; \
         truth={true_ty}, got={refined_ty_px:.4}"
    );
}

/// Coarse-to-fine refinement reduces RMS vs a coarse initial guess on a
/// larger (128×128) image so the pyramid has multiple levels in play.
#[test]
fn test_pyramid_refinement_reduces_rms_vs_coarse() {
    const W: usize = 128;
    const H: usize = 128;

    let reference = make_test_image(W, H);

    let true_tx = 1.5_f32;
    let true_ty = -1.0_f32;

    let mut source = PlanarImage::<f32>::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let sx = x as f32 - true_tx;
            let sy = y as f32 - true_ty;
            source.luma[y * W + x] = stacker_align::transform::bicubic_interpolate_channel(
                &reference.luma,
                W,
                H,
                sx,
                sy,
            );
        }
    }

    // Coarse initial guess: 0.5 px off in each axis.
    let p_coarse = RegistrationParams {
        tx: f64::from((true_tx - 0.5) / W as f32),
        ty: f64::from((true_ty + 0.5) / H as f32),
        scale: 1.0,
        rotate: 0.0,
    };
    let initial_matrix = params_to_matrix(&p_coarse, W, H).expect("coarse matrix must be valid");

    let opts = RefineOptions::default();
    let refined = refine_alignment(&reference, &source, &initial_matrix, &opts)
        .expect("refinement must succeed");

    let p_refined = matrix_to_params(&refined, W, H);
    let rms_coarse = objective(&p_coarse.to_vec(), &reference, &source);
    let rms_refined = objective(&p_refined.to_vec(), &reference, &source);

    assert!(
        rms_refined < rms_coarse,
        "pyramid refinement must reduce RMS: coarse={rms_coarse:.6}, refined={rms_refined:.6}"
    );

    // Also assert sub-pixel accuracy.
    let err_tx = ((p_refined.tx * W as f64) as f32 - true_tx).abs();
    let err_ty = ((p_refined.ty * H as f64) as f32 - true_ty).abs();
    assert!(
        err_tx < 0.15,
        "pyramid: tx error {err_tx:.4} px; truth={true_tx}"
    );
    assert!(
        err_ty < 0.15,
        "pyramid: ty error {err_ty:.4} px; truth={true_ty}"
    );
}

// ── direction convention: the refined matrix must be inverted before warping ──

/// RMS luma difference over the interior (a `margin`-pixel border is skipped so
/// zero-filled out-of-frame samples do not dominate the figure).
fn interior_rms(first: &PlanarImage<f32>, second: &PlanarImage<f32>, margin: usize) -> f64 {
    let (width, height) = (first.width, first.height);
    let mut sum = 0.0_f64;
    let mut count = 0_u64;
    for row in margin..height - margin {
        for col in margin..width - margin {
            let idx = row * width + col;
            let diff = f64::from(first.luma[idx] - second.luma[idx]);
            sum += diff * diff;
            count += 1;
        }
    }
    (sum / count as f64).sqrt()
}

/// The library returns `M` with `source(M·p) ≈ reference(p)`; an application
/// that wants `source` registered onto `reference` must warp with `M⁻¹`.
/// Warping with `M` itself doubles the misalignment instead of removing it.
#[test]
fn test_warping_source_with_inverse_of_refined_matrix_aligns_it() {
    const W: usize = 128;
    const H: usize = 128;
    const MARGIN: usize = 8;

    let reference = make_test_image(W, H);

    // `warp_image(src, T)` yields `dst(p) = src(T⁻¹·p)`, so with a pure
    // translation `T(t)` the source is `reference(p - t)`: content moved by +t.
    let (tx, ty) = (3.0_f32, -2.0_f32);
    let mut shift = Matrix3::<f32>::identity();
    shift[(0, 2)] = tx;
    shift[(1, 2)] = ty;
    let source = stacker_align::transform::warp_image(&reference, &shift)
        .expect("building the shifted source must succeed");

    let refined = refine_alignment(
        &reference,
        &source,
        &Matrix3::<f32>::identity(),
        &RefineOptions {
            allow_scale: false,
            allow_rotate: false,
            ..RefineOptions::default()
        },
    )
    .expect("refinement must succeed on a pure translation");

    // The refiner recovers +t (its documented reference→source convention).
    assert!(
        (refined[(0, 2)] - tx).abs() < 0.25 && (refined[(1, 2)] - ty).abs() < 0.25,
        "refined translation ({}, {}) should be close to ({tx}, {ty})",
        refined[(0, 2)],
        refined[(1, 2)]
    );

    let unaligned = interior_rms(&reference, &source, MARGIN);

    let inverse = refined.try_inverse().expect("a translation is invertible");
    let aligned_right = stacker_align::transform::warp_image(&source, &inverse)
        .expect("warp with the inverse must succeed");
    let residual_right = interior_rms(&reference, &aligned_right, MARGIN);

    let aligned_wrong = stacker_align::transform::warp_image(&source, &refined)
        .expect("warp with the forward matrix must succeed");
    let residual_wrong = interior_rms(&reference, &aligned_wrong, MARGIN);

    assert!(
        residual_right < 0.5 * unaligned,
        "warping with M⁻¹ must at least halve the residual: aligned={residual_right:.5} unaligned={unaligned:.5}"
    );
    assert!(
        residual_wrong > residual_right,
        "warping with M itself must be worse than warping with M⁻¹: wrong={residual_wrong:.5} right={residual_right:.5}"
    );
}
