#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    clippy::similar_names
)]

use nalgebra::Matrix3;
use stacker_align::refine::*;
use stacker_core::image::PlanarImage;

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
        aspect: 1.0,
        shear: 0.0,
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
        aspect: 1.0,
        shear: 0.0,
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
    // the mean-bias-corrected RMS.
    let plane: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
    let shifted: Vec<f32> = plane.iter().map(|&v| v + 0.2_f32).collect();
    let rms = rms_difference(&plane, &shifted, 8, 8);
    assert!(
        rms < 1.0e-6,
        "global brightness offset must not affect mean-bias-corrected RMS; got {rms:.3e}"
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
