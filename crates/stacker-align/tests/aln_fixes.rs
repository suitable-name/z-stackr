//! Library-side tests for review findings ALN-2 (pinned DOFs in the bounded
//! refiner), ALN-8 (thread-count-independent objective), ALN-9 (the single-DOF
//! bounded path) and ALN-13 (explicit reduction of a general affine seed).

#![expect(
    clippy::cast_precision_loss,
    reason = "fixture sizes are a few hundred pixels, so index to float conversions are exact"
)]

use nalgebra::Matrix3;
use stacker_align::{
    BoundedRefineOptions,
    refine::{
        RegistrationParams, matrix_to_params, objective_at_level, params_to_matrix,
        reduce_to_similarity, rms_difference,
    },
    refine_alignment_bounded, refine_alignment_registration,
    transform::bicubic_interpolate_channel,
};
use stacker_core::image::PlanarImage;

const W: usize = 80;
const H: usize = 80;

/// Smooth, non-symmetric luma pattern in [0, 1].
fn make_reference() -> PlanarImage<f32> {
    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            img.luma[y * W + x] = ((x as f32 * 0.11).sin() + (y as f32 * 0.09).cos())
                .mul_add(0.4, 0.5)
                .clamp(0.0, 1.0);
        }
    }
    img
}

/// Source whose content is the reference magnified by `scale` about the image
/// centre and shifted by `(tx, ty)` pixels (reference→source direction).
fn make_source(reference: &PlanarImage<f32>, scale: f32, tx: f32, ty: f32) -> PlanarImage<f32> {
    let (cx, cy) = (W as f32 * 0.5, H as f32 * 0.5);
    let mut source = PlanarImage::<f32>::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let sx = (x as f32 - cx - tx) / scale + cx;
            let sy = (y as f32 - cy - ty) / scale + cy;
            source.luma[y * W + x] = bicubic_interpolate_channel(&reference.luma, W, H, sx, sy);
        }
    }
    source
}

// ── ALN-2 ────────────────────────────────────────────────────────────────

/// `allow_scale: false` must leave the scale at the seed's value even when the
/// source really differs by 2 % in scale.
#[test]
fn bounded_refine_pins_scale_when_not_allowed() {
    let reference = make_reference();
    let source = make_source(&reference, 1.02, 0.0, 0.0);
    let opts = BoundedRefineOptions {
        allow_scale: false,
        ..BoundedRefineOptions::default()
    };
    let refined = refine_alignment_bounded(&reference, &source, &Matrix3::identity(), &opts)
        .expect("bounded refinement must succeed");
    let p = matrix_to_params(&refined, W, H);
    assert!(
        (p.scale - 1.0).abs() < 1.0e-6,
        "scale must stay at the seed's 1.0, got {}",
        p.scale
    );
}

/// `Translation` mode: both scale and rotation pinned, so the linear part of
/// the result is exactly the seed's (identity) whatever the source does.
#[test]
fn bounded_refine_translation_mode_keeps_linear_part_identity() {
    let reference = make_reference();
    let source = make_source(&reference, 1.02, 1.5, -1.0);
    let opts = BoundedRefineOptions {
        allow_scale: false,
        allow_rotation: false,
        ..BoundedRefineOptions::default()
    };
    let refined = refine_alignment_bounded(&reference, &source, &Matrix3::identity(), &opts)
        .expect("bounded refinement must succeed");
    assert!(
        (refined[(0, 0)] - 1.0).abs() < 1.0e-6 && (refined[(1, 1)] - 1.0).abs() < 1.0e-6,
        "diagonal must stay 1, got {refined}"
    );
    assert!(
        refined[(0, 1)].abs() < 1.0e-6 && refined[(1, 0)].abs() < 1.0e-6,
        "off-diagonal must stay 0, got {refined}"
    );
}

/// A pinned DOF keeps the *seed's* value, not 1.0 / 0.0.
#[test]
fn bounded_refine_pinned_dofs_keep_the_seed_values() {
    let reference = make_reference();
    let source = make_source(&reference, 1.0, 0.5, 0.0);
    let seed_params = RegistrationParams {
        tx: 0.0,
        ty: 0.0,
        scale: 1.01,
        rotate: 0.01,
    };
    let seed = params_to_matrix(&seed_params, W, H).expect("valid seed");
    let opts = BoundedRefineOptions {
        allow_scale: false,
        allow_rotation: false,
        ..BoundedRefineOptions::default()
    };
    let refined = refine_alignment_bounded(&reference, &source, &seed, &opts)
        .expect("bounded refinement must succeed");
    let p = matrix_to_params(&refined, W, H);
    assert!((p.scale - 1.01).abs() < 1.0e-6, "scale {}", p.scale);
    assert!((p.rotate - 0.01).abs() < 1.0e-6, "rotate {}", p.rotate);
}

// ── ALN-9 ────────────────────────────────────────────────────────────────

/// Options with exactly one active DOF (the X shift): the golden-section path.
fn x_shift_only() -> BoundedRefineOptions {
    BoundedRefineOptions {
        allow_shift_x: true,
        allow_shift_y: false,
        allow_scale: false,
        allow_rotation: false,
        ..BoundedRefineOptions::default()
    }
}

/// The single-DOF path of the pyramid-based bounded refiner recovers a 1 % x
/// shift (0.8 px on the 80-px fixture) and leaves everything else at the seed.
#[test]
fn bounded_refine_single_dof_recovers_an_x_shift() {
    let reference = make_reference();
    let source = make_source(&reference, 1.0, 0.8, 0.0);
    let refined =
        refine_alignment_bounded(&reference, &source, &Matrix3::identity(), &x_shift_only())
            .expect("single-DOF bounded refinement must succeed");
    let p = matrix_to_params(&refined, W, H);
    let shift_px = p.tx * W as f64;
    assert!(
        (shift_px - 0.8).abs() < 0.2,
        "recovered x shift {shift_px} px, expected 0.8"
    );
    assert!(
        p.ty.abs() < 1.0e-6,
        "ty must stay at the seed, got {}",
        p.ty
    );
    assert!((p.scale - 1.0).abs() < 1.0e-6, "scale {}", p.scale);
    assert!(p.rotate.abs() < 1.0e-6, "rotate {}", p.rotate);
}

/// The single-resolution registration path (spline sampler) takes the same
/// golden-section branch with one active DOF.
#[test]
fn registration_single_dof_recovers_an_x_shift() {
    let reference = make_reference();
    let source = make_source(&reference, 1.0, 0.8, 0.0);
    let refined =
        refine_alignment_registration(&reference, &source, &Matrix3::identity(), &x_shift_only())
            .expect("single-DOF registration must succeed");
    let p = matrix_to_params(&refined, W, H);
    let shift_px = p.tx * W as f64;
    assert!(
        (shift_px - 0.8).abs() < 0.25,
        "recovered x shift {shift_px} px, expected 0.8"
    );
}

// ── ALN-8 ────────────────────────────────────────────────────────────────

/// Deterministic pseudo-random plane in [0, 1).
fn noise_plane(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1_u64 << 24) as f32
        })
        .collect()
}

/// Run `f` on rayon pools of different sizes and return every result.
fn on_pools<T: Send>(f: impl Fn() -> T + Sync) -> Vec<T> {
    [1_usize, 2, 3, 8]
        .iter()
        .flat_map(|&threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            // Several runs per pool, so work-stealing gets a chance to vary.
            (0..4).map(|_| pool.install(&f)).collect::<Vec<_>>()
        })
        .collect()
}

/// The row partial sums are added in row order, so the RMS of a plane above the
/// serial threshold is bit-identical for every thread count and every run.
#[test]
fn rms_difference_is_bit_identical_across_thread_counts() {
    let (w, h) = (192_usize, 160_usize);
    assert!(w * h >= 128 * 128, "the plane must take the parallel path");
    let a = noise_plane(w * h, 1);
    let b = noise_plane(w * h, 2);
    let values = on_pools(|| rms_difference(&a, &b, w, h));
    assert!(values[0].is_finite() && values[0] > 0.0);
    for v in &values {
        assert_eq!(v.to_bits(), values[0].to_bits(), "{v} vs {}", values[0]);
    }
}

/// Same for the masked objective the optimiser actually evaluates.
#[test]
fn objective_at_level_is_bit_identical_across_thread_counts() {
    let (w, h) = (192_usize, 160_usize);
    let reference = noise_plane(w * h, 3);
    let source = noise_plane(w * h, 4);
    let params = [0.011, -0.008, 1.002, 0.003];
    let values = on_pools(|| objective_at_level(&params, &reference, &source, w, h));
    assert!(values[0].is_finite() && values[0] > 0.0 && values[0] < 1.0e5);
    for v in &values {
        assert_eq!(v.to_bits(), values[0].to_bits(), "{v} vs {}", values[0]);
    }
}

// ── ALN-13 ───────────────────────────────────────────────────────────────

/// A matrix with shear and anisotropic scale is reduced by the documented
/// rule: the nearest similarity, exact at the image centre.
#[test]
fn matrix_to_params_reduces_general_affine_at_the_centre() {
    let mut m = Matrix3::<f32>::identity();
    m[(0, 0)] = 1.03;
    m[(0, 1)] = 0.04;
    m[(1, 0)] = -0.01;
    m[(1, 1)] = 0.97;
    m[(0, 2)] = 3.0;
    m[(1, 2)] = -2.0;

    let p = matrix_to_params(&m, W, H);
    let sim = params_to_matrix(&p, W, H).expect("finite params");

    // Reduction rule: s·cos = (a00 + a11) / 2, s·sin = (a10 - a01) / 2.
    let c = f64::from(f32::midpoint(1.03_f32, 0.97));
    let s = f64::from(0.5 * (-0.01_f32 - 0.04));
    assert!((p.scale - c.hypot(s)).abs() < 1.0e-6, "scale {}", p.scale);
    assert!(
        (p.rotate - s.atan2(c)).abs() < 1.0e-6,
        "rotate {}",
        p.rotate
    );

    // The reduced similarity sends the image centre exactly where the full
    // affine map does.
    let centre = nalgebra::Vector3::new(W as f32 * 0.5, H as f32 * 0.5, 1.0);
    let full = m * centre;
    let reduced = sim * centre;
    assert!(
        (full[0] - reduced[0]).abs() < 1.0e-3 && (full[1] - reduced[1]).abs() < 1.0e-3,
        "centre maps to {full:?} under the affine, {reduced:?} under the reduction"
    );

    // `reduce_to_similarity` is that same matrix.
    let reduced_m = reduce_to_similarity(&m, W, H);
    assert!((reduced_m - sim).norm() < 1.0e-6);
}

/// Pure anisotropic scale about the origin: no rotation, mean scale.
#[test]
fn matrix_to_params_anisotropic_scale_uses_the_mean_scale() {
    let mut m = Matrix3::<f32>::identity();
    m[(0, 0)] = 1.04;
    m[(1, 1)] = 0.96;
    let p = matrix_to_params(&m, W, H);
    assert!((p.scale - 1.0).abs() < 1.0e-6, "scale {}", p.scale);
    assert!(p.rotate.abs() < 1.0e-6, "rotate {}", p.rotate);
}

/// An exact similarity still round-trips through `params_to_matrix`.
#[test]
fn matrix_to_params_similarity_round_trip_is_unchanged() {
    let orig = RegistrationParams {
        tx: 0.05,
        ty: -0.03,
        scale: 1.02,
        rotate: 0.01,
    };
    let m = params_to_matrix(&orig, W, H).expect("valid params");
    let back = matrix_to_params(&m, W, H);
    assert!((back.tx - orig.tx).abs() < 1.0e-6);
    assert!((back.ty - orig.ty).abs() < 1.0e-6);
    assert!((back.scale - orig.scale).abs() < 1.0e-6);
    assert!((back.rotate - orig.rotate).abs() < 1.0e-6);
}
