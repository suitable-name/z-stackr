#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

//! Lucas-Kanade / Gauss-Newton refinement.
//!
//! Ported from GitHub z-stackr's `lucas_kanade_test.rs` and re-derived for this
//! repository's conventions:
//!
//! - the refiners return the matrix `M` that maps **reference coordinates to
//!   source coordinates** (`source(M p) = reference(p)`); GitHub's tests
//!   recovered the *inverse* of the warp they applied, so every "expected"
//!   below is the applied matrix itself;
//! - the fixtures are analytic (the source is the reference texture evaluated at
//!   the inverse-mapped coordinates), so the objective's minimum is the truth up
//!   to the interpolation error of the sampler, with no border artefact;
//! - tolerances are in pixels at the image corners, not in fractions of a
//!   parameter.

use nalgebra::{Matrix3, Vector3};
use stacker_align::{
    refine::{
        AffineParams, BoundedRefineOptions, LumaPyramid, lk_rms, matrix_to_affine_params,
        refine_alignment_bounded, refine_alignment_lk, refine_alignment_lk_with_pyramid,
    },
    transform::warp_image_clamped,
};
use stacker_core::{error::StackerError, image::PlanarImage};

const W: usize = 160;
const H: usize = 120;

// ── fixtures ──────────────────────────────────────────────────────────────────

/// A smooth multi-directional texture in [0.06, 0.94].
fn texture_at(x: f64, y: f64) -> f32 {
    (0.5 + 0.18 * (0.21 * x + 0.13 * y).sin()
        + 0.18 * (0.17 * x - 0.19 * y).cos()
        + 0.08 * (0.07 * x).sin() * (0.05 * y).cos()) as f32
}

fn image(w: usize, h: usize, f: impl Fn(f64, f64) -> f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.luma[y * w + x] = f(x as f64, y as f64);
        }
    }
    img
}

fn reference() -> PlanarImage<f32> {
    image(W, H, texture_at)
}

/// The reference content moved by `m` (reference to source), analytically.
fn moved(m: &Matrix3<f32>) -> PlanarImage<f32> {
    let inv = m.cast::<f64>().try_inverse().expect("invertible matrix");
    image(W, H, |x, y| {
        let p = inv * Vector3::new(x, y, 1.0);
        texture_at(p.x, p.y)
    })
}

fn matrix(tx: f64, ty: f64, scale: f64, rotate: f64, aspect: f64, shear: f64) -> Matrix3<f32> {
    AffineParams {
        tx,
        ty,
        scale,
        rotate,
        aspect,
        shear,
    }
    .to_matrix(W, H)
    .expect("finite parameters")
}

/// The largest displacement, over the corners and the centre, between the
/// points `a` and `b` map.
fn corner_error(a: &Matrix3<f32>, b: &Matrix3<f32>) -> f64 {
    let (a, b) = (a.cast::<f64>(), b.cast::<f64>());
    let (w, h) = (W as f64, H as f64);
    [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h), (w / 2.0, h / 2.0)]
        .iter()
        .map(|&(x, y)| {
            let v = Vector3::new(x, y, 1.0);
            let d = a * v - b * v;
            d.x.hypot(d.y)
        })
        .fold(0.0, f64::max)
}

fn bits(m: &Matrix3<f32>) -> [u32; 9] {
    std::array::from_fn(|i| m[(i / 3, i % 3)].to_bits())
}

fn no_aspect_shear() -> BoundedRefineOptions {
    BoundedRefineOptions::default()
}

// ── convergence ───────────────────────────────────────────────────────────────

/// A sub-pixel-scale shift (2.9 px, -1.4 px) is recovered to a few hundredths of
/// a pixel.
#[test]
fn recovers_a_small_translation() {
    let truth = matrix(0.018, -0.012, 1.0, 0.0, 1.0, 0.0);
    let result = refine_alignment_lk(
        &reference(),
        &moved(&truth),
        &Matrix3::identity(),
        &no_aspect_shear(),
    )
    .expect("refinement succeeds");
    let err = corner_error(&result.matrix, &truth);
    assert!(err < 0.05, "corner error {err} px");
}

/// Rotation and scale together.
#[test]
fn recovers_a_small_rotation_and_scale() {
    let truth = matrix(0.0, 0.0, 1.015, 0.02, 1.0, 0.0);
    let reference = reference();
    let source = moved(&truth);
    let result = refine_alignment_lk(
        &reference,
        &source,
        &Matrix3::identity(),
        &no_aspect_shear(),
    )
    .expect("refinement succeeds");
    let err = corner_error(&result.matrix, &truth);
    assert!(err < 0.05, "corner error {err} px");
    // Real work was done.
    let identity_rms = lk_rms(&reference, &source, &Matrix3::identity());
    let refined_rms = lk_rms(&reference, &source, &result.matrix);
    assert!(
        refined_rms < 0.1 * identity_rms,
        "refined RMS {refined_rms} vs identity RMS {identity_rms}"
    );
}

/// The headline requirement: a known six-parameter warp (shift, scale, rotation,
/// aspect and shear) of a textured synthetic is recovered within 0.05 px.
#[test]
fn recovers_a_known_six_parameter_warp_within_five_hundredths_of_a_pixel() {
    let truth = matrix(0.010, -0.008, 1.008, 0.010, 1.010, 0.010);
    let opts = BoundedRefineOptions {
        allow_aspect: true,
        allow_shear: true,
        ..BoundedRefineOptions::default()
    };
    let result = refine_alignment_lk(&reference(), &moved(&truth), &Matrix3::identity(), &opts)
        .expect("refinement succeeds");
    let err = corner_error(&result.matrix, &truth);
    assert!(err < 0.05, "corner error {err} px");
    // The six parameters themselves.
    let p = matrix_to_affine_params(&result.matrix, W, H);
    assert!((p.aspect - 1.010).abs() < 0.002, "aspect {}", p.aspect);
    assert!((p.shear - 0.010).abs() < 0.002, "shear {}", p.shear);
    assert!(result.final_rms < result.starting_rms * 0.1);
}

/// Seeded away from both identity and the truth, LK still gets there.
#[test]
fn converges_from_a_coarse_seed() {
    let truth = matrix(0.03, 0.02, 1.01, 0.0, 1.0, 0.0);
    let reference = reference();
    let source = moved(&truth);
    let seed = matrix(0.015, 0.01, 1.0, 0.0, 1.0, 0.0);
    let result = refine_alignment_lk(&reference, &source, &seed, &no_aspect_shear())
        .expect("refinement succeeds");
    let err = corner_error(&result.matrix, &truth);
    assert!(err < 0.05, "corner error {err} px");
    assert!(
        lk_rms(&reference, &source, &result.matrix) < lk_rms(&reference, &source, &seed),
        "the result must beat the seed"
    );
}

/// Identical frames stay at the identity.
#[test]
fn identical_images_stay_at_the_identity() {
    let img = reference();
    let result = refine_alignment_lk(&img, &img, &Matrix3::identity(), &no_aspect_shear())
        .expect("refinement succeeds");
    let err = corner_error(&result.matrix, &Matrix3::identity());
    assert!(err < 0.01, "corner error {err} px");
    assert!(result.final_rms <= result.starting_rms);
}

// ── DOF gating ────────────────────────────────────────────────────────────────

/// With rotation, aspect and shear disabled, rotation stays at 0, aspect at 1,
/// shear at 0, even though the source is scaled and shifted.
#[test]
fn disabled_dofs_stay_at_the_seed() {
    let truth = matrix(0.02, 0.0, 1.01, 0.0, 1.0, 0.0);
    let opts = BoundedRefineOptions {
        allow_rotation: false,
        allow_aspect: false,
        allow_shear: false,
        ..BoundedRefineOptions::default()
    };
    let result = refine_alignment_lk(&reference(), &moved(&truth), &Matrix3::identity(), &opts)
        .expect("refinement succeeds");
    let p = matrix_to_affine_params(&result.matrix, W, H);
    assert!(p.rotate.abs() < 1.0e-6, "rotation {}", p.rotate);
    assert!((p.aspect - 1.0).abs() < 1.0e-6, "aspect {}", p.aspect);
    assert!(p.shear.abs() < 1.0e-6, "shear {}", p.shear);
    assert!(corner_error(&result.matrix, &truth) < 0.05);
}

/// Shift only: scale stays at 1 exactly.
#[test]
fn shift_only_leaves_the_linear_part_untouched() {
    let truth = matrix(0.02, -0.01, 1.0, 0.0, 1.0, 0.0);
    let opts = BoundedRefineOptions {
        allow_scale: false,
        allow_rotation: false,
        ..BoundedRefineOptions::default()
    };
    let result = refine_alignment_lk(&reference(), &moved(&truth), &Matrix3::identity(), &opts)
        .expect("refinement succeeds");
    let m = result.matrix;
    assert!((m[(0, 0)] - 1.0).abs() < 1.0e-6 && (m[(1, 1)] - 1.0).abs() < 1.0e-6);
    assert!(m[(0, 1)].abs() < 1.0e-6 && m[(1, 0)].abs() < 1.0e-6);
    assert!(corner_error(&m, &truth) < 0.05);
}

/// A pinned DOF keeps the seed's value, not 1 / 0.
#[test]
fn a_pinned_dof_keeps_the_seed_value() {
    let truth = matrix(0.02, 0.0, 1.01, 0.01, 1.0, 0.0);
    let opts = BoundedRefineOptions {
        allow_scale: false,
        allow_rotation: false,
        ..BoundedRefineOptions::default()
    };
    let seed = matrix(0.0, 0.0, 1.01, 0.01, 1.0, 0.0);
    let result = refine_alignment_lk(&reference(), &moved(&truth), &seed, &opts)
        .expect("refinement succeeds");
    let p = matrix_to_affine_params(&result.matrix, W, H);
    assert!((p.scale - 1.01).abs() < 1.0e-6, "scale {}", p.scale);
    assert!((p.rotate - 0.01).abs() < 1.0e-6, "rotation {}", p.rotate);
}

/// No enabled DOF: the seed comes back unchanged, with equal RMS values.
#[test]
fn no_active_dof_returns_the_seed() {
    let img = reference();
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
    let result = refine_alignment_lk(&img, &img, &seed, &opts).expect("pass-through");
    assert_eq!(bits(&result.matrix), bits(&seed));
    assert_eq!(result.starting_rms.to_bits(), result.final_rms.to_bits());
}

// ── guards ────────────────────────────────────────────────────────────────────

#[test]
fn dimension_mismatch_is_an_alignment_error() {
    let a = PlanarImage::<f32>::new(32, 32);
    let b = PlanarImage::<f32>::new(64, 64);
    let result = refine_alignment_lk(&a, &b, &Matrix3::identity(), &no_aspect_shear());
    assert!(matches!(result, Err(StackerError::AlignmentFailed(_))));
}

#[test]
fn non_finite_seed_is_a_math_error() {
    let img = PlanarImage::<f32>::new(32, 32);
    let mut bad = Matrix3::<f32>::identity();
    bad[(0, 0)] = f32::NAN;
    let result = refine_alignment_lk(&img, &img, &bad, &no_aspect_shear());
    assert!(matches!(result, Err(StackerError::MathError(_))));
}

#[test]
fn a_pyramid_of_another_image_is_rejected() {
    let a = reference();
    let other = image(64, 64, texture_at);
    let pyramid = LumaPyramid::new(&other);
    let result = refine_alignment_lk_with_pyramid(
        &a,
        &pyramid,
        &a,
        &Matrix3::identity(),
        &no_aspect_shear(),
    );
    assert!(matches!(result, Err(StackerError::AlignmentFailed(_))));
}

/// A hopeless seed (shifted five frame widths away) neither panics nor produces a
/// non-finite matrix.
#[test]
fn pathological_seed_stays_finite() {
    let img = reference();
    let mut seed = Matrix3::<f32>::identity();
    seed[(0, 2)] = W as f32 * 5.0;
    seed[(1, 2)] = H as f32 * 5.0;
    if let Ok(result) = refine_alignment_lk(&img, &img, &seed, &no_aspect_shear()) {
        assert!(result.matrix.iter().all(|v| v.is_finite()));
    }
}

/// Textureless frames: nothing to follow, nothing explodes.
#[test]
fn flat_frames_stay_finite() {
    let flat = image(W, H, |_, _| 0.4);
    let result = refine_alignment_lk(&flat, &flat, &Matrix3::identity(), &no_aspect_shear())
        .expect("refinement succeeds");
    assert!(result.matrix.iter().all(|v| v.is_finite()));
    assert!(corner_error(&result.matrix, &Matrix3::identity()) < 1.0e-3);
}

// ── comparison and properties ─────────────────────────────────────────────────

/// LK and the simplex agree on a similarity: the same corner positions within a
/// fraction of a pixel, and LK's objective is competitive.
#[test]
fn agrees_with_the_simplex_on_a_similarity() {
    let truth = matrix(0.025, -0.018, 1.008, 0.015, 1.0, 0.0);
    let reference = reference();
    let source = moved(&truth);
    let opts = no_aspect_shear();
    let lk =
        refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts).expect("LK succeeds");
    let nm = refine_alignment_bounded(&reference, &source, &Matrix3::identity(), &opts)
        .expect("NM succeeds");
    assert!(
        corner_error(&lk.matrix, &nm) < 0.3,
        "LK and NM differ by {} px",
        corner_error(&lk.matrix, &nm)
    );
    assert!(corner_error(&lk.matrix, &truth) < 0.05);
    let lk_final = lk_rms(&reference, &source, &lk.matrix);
    let nm_final = lk_rms(&reference, &source, &nm);
    assert!(
        lk_final <= nm_final * 1.1 + 1.0e-3,
        "LK RMS {lk_final} vs NM RMS {nm_final}"
    );
}

/// The row partials are added in a fixed order, so the result is the same on
/// every run and with a prebuilt pyramid.
#[test]
fn is_deterministic_and_pyramid_independent() {
    let truth = matrix(0.012, 0.008, 1.004, 0.006, 1.0, 0.0);
    let reference = reference();
    let source = moved(&truth);
    let opts = no_aspect_shear();
    let a = refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts).expect("ok");
    let b = refine_alignment_lk(&reference, &source, &Matrix3::identity(), &opts).expect("ok");
    let pyramid = LumaPyramid::new(&reference);
    let c = refine_alignment_lk_with_pyramid(
        &reference,
        &pyramid,
        &source,
        &Matrix3::identity(),
        &opts,
    )
    .expect("ok");
    assert_eq!(bits(&a.matrix), bits(&b.matrix));
    assert_eq!(bits(&a.matrix), bits(&c.matrix));
    assert_eq!(a.final_rms.to_bits(), b.final_rms.to_bits());
}

/// The two RMS values of a result are measured at the same level, so a good
/// refinement shows `final <= starting` (the test `Optimizer::Auto` applies).
#[test]
fn rms_pair_shows_the_improvement() {
    let truth = matrix(0.02, -0.01, 1.005, 0.008, 1.0, 0.0);
    let result = refine_alignment_lk(
        &reference(),
        &moved(&truth),
        &Matrix3::identity(),
        &no_aspect_shear(),
    )
    .expect("refinement succeeds");
    assert!(result.starting_rms.is_finite() && result.final_rms.is_finite());
    assert!(
        result.final_rms < 0.1 * result.starting_rms,
        "starting {} final {}",
        result.starting_rms,
        result.final_rms
    );
}

/// Frames produced by the library's own clamped warp (the border is replicated,
/// not analytic) are still aligned to well under a pixel.
#[test]
fn works_on_frames_from_the_library_warp() {
    let truth = matrix(0.015, -0.01, 1.0, 0.01, 1.0, 0.0);
    let reference = reference();
    let source = warp_image_clamped(&reference, &truth).expect("invertible");
    let result = refine_alignment_lk(
        &reference,
        &source,
        &Matrix3::identity(),
        &no_aspect_shear(),
    )
    .expect("refinement succeeds");
    let err = corner_error(&result.matrix, &truth);
    assert!(err < 0.3, "corner error {err} px");
}
