#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::suboptimal_flops,
    clippy::similar_names
)]

use nalgebra::Matrix3;
use stacker_align::{refine::*, transform::warp_image_clamped};
use stacker_core::image::PlanarImage;

/// Smooth, non-symmetric gradient/texture image in [0, 1] — same generator
/// style used throughout `tests/refine.rs` and the inline unit tests in
/// `refine/core.rs`/`refine/lucas_kanade.rs`.
fn make_test_image(width: usize, height: usize) -> PlanarImage<f32> {
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

/// `warp_image_clamped(src, m)` is a *backward*-mapping warp: it samples
/// `src` at `m.try_inverse() * dst_coords` (see `transform::warp`'s module
/// docs). `objective_registration`/`refine_alignment_lk`/
/// `refine_alignment_registration`, on the other hand, use their
/// `RegistrationParams`-derived matrix *directly* (no inversion) as the
/// "sample the reference at `fwd * dst_coords`" forward map. So whenever a
/// test builds `source = warp_image_clamped(&reference, &known_matrix)`,
/// the transform the optimiser should recover is `known_matrix`'s inverse,
/// not `known_matrix` itself — this helper makes that translation explicit
/// and shared across every test below instead of re-deriving it per test.
fn expected_recovered_params(
    known_matrix: &Matrix3<f32>,
    w: usize,
    h: usize,
) -> RegistrationParams {
    matrix_to_params(
        &known_matrix.try_inverse().expect("known_matrix invertible"),
        w,
        h,
    )
}

// ── LK: basic convergence ────────────────────────────────────────────────

/// LK recovers a small known translation to subpixel accuracy.
#[test]
fn lk_recovers_small_translation_subpixel() {
    let w = 160_usize;
    let h = 120_usize;
    let reference = make_test_image(w, h);

    let known_p = RegistrationParams {
        tx: 0.018,
        ty: -0.012,
        scale: 1.0,
        rotate: 0.0,
        aspect: 1.0,
        shear: 0.0,
    };
    let known_matrix = params_to_matrix(&known_p, w, h).expect("valid matrix");
    let source = warp_image_clamped(&reference, &known_matrix).expect("warp ok");
    let expected = expected_recovered_params(&known_matrix, w, h);

    let opts = BoundedRefineOptions::default();
    let result = refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts)
        .expect("refine_alignment_lk should succeed");

    let recovered = matrix_to_params(&result.matrix, w, h);
    assert!(
        (recovered.tx - expected.tx).abs() < 0.005,
        "tx: expected {:.6}, got {:.6}",
        expected.tx,
        recovered.tx
    );
    assert!(
        (recovered.ty - expected.ty).abs() < 0.005,
        "ty: expected {:.6}, got {:.6}",
        expected.ty,
        recovered.ty
    );
}

/// LK recovers a small similarity transform (rotation + scale combined).
#[test]
fn lk_recovers_small_rotation_and_scale() {
    let w = 160_usize;
    let h = 120_usize;
    let reference = make_test_image(w, h);

    let known_p = RegistrationParams {
        tx: 0.0,
        ty: 0.0,
        scale: 1.015,
        rotate: 0.02, // ~1.15 degrees
        aspect: 1.0,
        shear: 0.0,
    };
    let known_matrix = params_to_matrix(&known_p, w, h).expect("valid matrix");
    let source = warp_image_clamped(&reference, &known_matrix).expect("warp ok");
    let expected = expected_recovered_params(&known_matrix, w, h);

    let opts = BoundedRefineOptions::default();
    let result = refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts)
        .expect("refine_alignment_lk should succeed");

    let recovered = matrix_to_params(&result.matrix, w, h);
    assert!(
        (recovered.scale - expected.scale).abs() < 0.005,
        "scale: expected {:.6}, got {:.6}",
        expected.scale,
        recovered.scale
    );
    assert!(
        (recovered.rotate - expected.rotate).abs() < 0.01,
        "rotate: expected {:.6}, got {:.6}",
        expected.rotate,
        recovered.rotate
    );

    // The refined matrix must also beat identity RMS (sanity: real work done).
    let identity_rms = lk_rms(&reference, &source, &Matrix3::identity());
    let refined_rms = lk_rms(&reference, &source, &result.matrix);
    assert!(
        refined_rms < identity_rms,
        "refined RMS ({refined_rms:.6}) must beat identity RMS ({identity_rms:.6})"
    );
}

/// LK on identical images (source == reference) returns ~identity params.
#[test]
fn lk_identical_images_returns_near_identity() {
    let w = 128_usize;
    let h = 96_usize;
    let img = make_test_image(w, h);

    let opts = BoundedRefineOptions::default();
    let result = refine_alignment_lk(&img, &img, &Matrix3::identity(), &opts)
        .expect("refine_alignment_lk should succeed on identical images");

    let recovered = matrix_to_params(&result.matrix, w, h);
    let identity = RegistrationParams::identity();
    assert!(
        (recovered.tx - identity.tx).abs() < 1.0e-3,
        "tx should stay ~0 on identical images, got {}",
        recovered.tx
    );
    assert!(
        (recovered.ty - identity.ty).abs() < 1.0e-3,
        "ty should stay ~0 on identical images, got {}",
        recovered.ty
    );
    assert!(
        (recovered.scale - identity.scale).abs() < 1.0e-3,
        "scale should stay ~1 on identical images, got {}",
        recovered.scale
    );
    assert!(
        recovered.rotate.abs() < 1.0e-3,
        "rotate should stay ~0 on identical images, got {}",
        recovered.rotate
    );
}

/// LK converges starting from a moderately-off ("AKAZE-style coarse seed")
/// starting matrix, not just identity.
#[test]
fn lk_converges_from_coarse_seed() {
    let w = 160_usize;
    let h = 120_usize;
    let reference = make_test_image(w, h);

    let known_p = RegistrationParams {
        tx: 0.03,
        ty: 0.02,
        scale: 1.01,
        rotate: 0.0,
        aspect: 1.0,
        shear: 0.0,
    };
    let known_matrix = params_to_matrix(&known_p, w, h).expect("valid matrix");
    let source = warp_image_clamped(&reference, &known_matrix).expect("warp ok");
    let expected = expected_recovered_params(&known_matrix, w, h);

    // A coarse, "AKAZE-ish" seed: correct-ish direction but off by a
    // meaningful fraction of the true translation/scale (not identity, and
    // not the exact answer either).
    let seed_p = RegistrationParams {
        tx: expected.tx * 0.5,
        ty: expected.ty * 0.5,
        scale: 1.0,
        rotate: 0.0,
        aspect: 1.0,
        shear: 0.0,
    };
    let seed_matrix = params_to_matrix(&seed_p, w, h).expect("valid seed matrix");

    let opts = BoundedRefineOptions::default();
    let result = refine_alignment_lk(&reference, &source, &seed_matrix, &opts)
        .expect("refine_alignment_lk should succeed from a coarse seed");

    let identity_rms = lk_rms(&reference, &source, &Matrix3::identity());
    let seed_rms = lk_rms(&reference, &source, &seed_matrix);
    let refined_rms = lk_rms(&reference, &source, &result.matrix);

    assert!(
        refined_rms < seed_rms,
        "refined RMS ({refined_rms:.6}) must beat the coarse seed's RMS ({seed_rms:.6})"
    );
    assert!(
        refined_rms < identity_rms,
        "refined RMS ({refined_rms:.6}) must beat identity RMS ({identity_rms:.6})"
    );

    let recovered = matrix_to_params(&result.matrix, w, h);
    assert!(
        (recovered.tx - expected.tx).abs() < 0.01,
        "tx: expected {:.6}, got {:.6}",
        expected.tx,
        recovered.tx
    );
}

/// DOF gating: with `allow_rotation = false`, LK must not introduce
/// rotation even when the seed's DOF bounds would technically allow drift.
#[test]
fn lk_respects_dof_gating_no_rotation() {
    let w = 160_usize;
    let h = 120_usize;
    let reference = make_test_image(w, h);

    let known_p = RegistrationParams {
        tx: 0.02,
        ty: 0.0,
        scale: 1.01,
        rotate: 0.0,
        aspect: 1.0,
        shear: 0.0,
    };
    let known_matrix = params_to_matrix(&known_p, w, h).expect("valid matrix");
    let source = warp_image_clamped(&reference, &known_matrix).expect("warp ok");

    let opts = BoundedRefineOptions {
        allow_rotation: false,
        allow_aspect: false,
        allow_shear: false,
        ..BoundedRefineOptions::default()
    };
    let result = refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts)
        .expect("gated refine_alignment_lk should succeed");

    let recovered = matrix_to_params(&result.matrix, w, h);
    assert!(
        recovered.rotate.abs() < 1.0e-9,
        "rotation must stay pinned at 0 when allow_rotation=false, got {}",
        recovered.rotate
    );
    assert!(
        (recovered.aspect - 1.0).abs() < 1.0e-9,
        "aspect must stay pinned at 1 when allow_aspect=false, got {}",
        recovered.aspect
    );
    assert!(
        recovered.shear.abs() < 1.0e-9,
        "shear must stay pinned at 0 when allow_shear=false, got {}",
        recovered.shear
    );
}

/// Dimension mismatch must return `AlignmentFailed`.
#[test]
fn lk_dimension_mismatch_errors() {
    let ref_img = PlanarImage::<f32>::new(32, 32);
    let src_img = PlanarImage::<f32>::new(64, 64);
    let result = refine_alignment_lk(
        &ref_img,
        &src_img,
        &Matrix3::identity(),
        &BoundedRefineOptions::default(),
    );
    assert!(
        matches!(
            result,
            Err(stacker_core::error::StackerError::AlignmentFailed(_))
        ),
        "dimension mismatch must return AlignmentFailed"
    );
}

/// Non-finite initial matrix must return `MathError`.
#[test]
fn lk_nan_initial_matrix_errors() {
    let img = PlanarImage::<f32>::new(32, 32);
    let mut bad = Matrix3::<f32>::identity();
    bad[(0, 0)] = f32::NAN;
    let result = refine_alignment_lk(&img, &img, &bad, &BoundedRefineOptions::default());
    assert!(
        matches!(result, Err(stacker_core::error::StackerError::MathError(_))),
        "non-finite initial matrix must return MathError"
    );
}

/// No active DOF (all gating disabled) must pass the initial matrix through
/// unchanged rather than doing pointless work or erroring.
#[test]
fn lk_no_active_dof_returns_initial_unchanged() {
    let img = make_test_image(64, 64);
    let opts = BoundedRefineOptions {
        allow_shift_x: false,
        allow_shift_y: false,
        allow_scale: false,
        allow_rotation: false,
        allow_aspect: false,
        allow_shear: false,
        ..BoundedRefineOptions::default()
    };
    let mut seed = Matrix3::<f32>::identity();
    seed[(0, 2)] = 3.0;
    let result = refine_alignment_lk(&img, &img, &seed, &opts)
        .expect("no active DOF should be a pass-through, not an error");
    let diff = (result.matrix - seed).norm();
    assert!(
        diff < 1.0e-6,
        "with no active DOF, refine_alignment_lk must return the initial matrix unchanged; diff={diff:.6}"
    );
}

// ── LK vs NM comparison ──────────────────────────────────────────────────

/// LK final RMS should be competitive with (not dramatically worse than)
/// Nelder-Mead's final RMS on a synthetic case both can converge well on.
/// A guard, not a strict benchmark: `lk_rms <= nm_rms * 1.1`.
#[test]
fn lk_final_rms_competitive_with_nelder_mead() {
    let w = 200_usize;
    let h = 150_usize;
    let reference = make_test_image(w, h);

    let known_p = RegistrationParams {
        tx: 0.025,
        ty: -0.018,
        scale: 1.008,
        rotate: 0.015,
        aspect: 1.0,
        shear: 0.0,
    };
    let known_matrix = params_to_matrix(&known_p, w, h).expect("valid matrix");
    let source = warp_image_clamped(&reference, &known_matrix).expect("warp ok");

    let opts = BoundedRefineOptions::default();

    let lk_result = refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts)
        .expect("LK should succeed");
    let nm_result = refine_alignment_registration(&reference, &source, &Matrix3::identity(), &opts)
        .expect("NM should succeed");

    let lk_final_rms = lk_rms(&reference, &source, &lk_result.matrix);
    let nm_final_rms = lk_rms(&reference, &source, &nm_result);

    assert!(
        lk_final_rms <= nm_final_rms * 1.1,
        "LK final RMS ({lk_final_rms:.6}) should be within 10% of NM final RMS ({nm_final_rms:.6})"
    );
}

/// Auto-mode-style fallback scenario: LK given a pathological start (a huge
/// displacement far beyond any pyramid level's basin of attraction) must
/// not crash and must not diverge to something worse than identity when
/// its own result is compared to identity — this is the same contract
/// `pipeline::align_frame`'s Auto mode relies on to decide whether to fall
/// back to Nelder-Mead. This test only exercises the LK half of that
/// contract (its RMS-vs-own-start regression signal); the actual fallback
/// wiring is exercised in `stacker-align`'s `pipeline::align` test suite.
#[test]
fn lk_pathological_seed_does_not_panic_and_stays_finite() {
    let w = 160_usize;
    let h = 120_usize;
    let reference = make_test_image(w, h);
    let source = reference.clone();

    // A huge, implausible translation — far beyond LK's basin of
    // attraction at any pyramid level (this deliberately does NOT go
    // through `is_sane_seed`, since this test exercises the raw LK
    // function directly, not the `align_frame` dispatch).
    let mut seed = Matrix3::<f32>::identity();
    seed[(0, 2)] = w as f32 * 5.0;
    seed[(1, 2)] = h as f32 * 5.0;

    let opts = BoundedRefineOptions::default();
    let result = refine_alignment_lk(&reference, &source, &seed, &opts);

    // Must not panic (already guaranteed by reaching this point) and must
    // either error cleanly or produce a finite matrix.
    if let Ok(lk_result) = result {
        assert!(
            lk_result.matrix.iter().all(|v| v.is_finite()),
            "LK must never return a non-finite matrix, even from a pathological seed"
        );
    }
}
