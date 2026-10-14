//! PERF-1 / PERF-3 / PERF-4: the reworked alignment hot paths must not change
//! what they compute.
//!
//! - the objective (reused warp buffer, hoisted source mean, fused warp + sum,
//!   serial small levels) against an independent straightforward evaluation;
//! - the `*_with_pyramid` entry points against the plain ones (bit-identical:
//!   every level of the test images is below the serial-evaluation threshold, so
//!   there is no rayon reduction whose association could differ);
//! - the shared-weight spline warp against the per-plane scalar sampler;
//! - the build-report helpers against the compile-time configuration.
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture sizes are small, so index to float conversions are exact"
)]
#![expect(
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    reason = "the fixtures spell out the reference formulas as plain multiply-add with conventional names"
)]

use nalgebra::{Matrix3, Vector3};
use stacker_align::{
    refine::{
        BoundedRefineOptions, LumaPyramid, RefineOptions, RegistrationParams, objective_at_level,
        params_to_matrix, refine_alignment, refine_alignment_bounded,
        refine_alignment_bounded_with_pyramid, refine_alignment_with_pyramid,
    },
    transform::{
        bicubic_interpolate_channel, has_hardware_fma, simd_build_report, spline4x4_sample_clamped,
        warp_image, warp_image_clamped,
    },
};
use stacker_core::{error::StackerError, image::PlanarImage};

/// Textured, non-symmetric test image with all three planes populated.
fn textured(width: usize, height: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let v = ((x as f32 * 0.11).sin() + (y as f32 * 0.09).cos() + (x as f32 * 0.31).sin())
                .mul_add(0.25, 0.5);
            img.luma[i] = v.clamp(0.0, 1.0);
            img.chroma_a[i] = ((x * 7 + y * 3) % 19) as f32 / 19.0 - 0.5;
            img.chroma_b[i] = ((x * 2 + y * 5) % 23) as f32 / 23.0 - 0.5;
        }
    }
    img
}

fn translation(tx: f32, ty: f32) -> Matrix3<f32> {
    let mut m = Matrix3::identity();
    m[(0, 2)] = tx;
    m[(1, 2)] = ty;
    m
}

/// The objective written the obvious way: fresh buffer, two plain passes.
fn reference_objective(
    p: &[f64; 4],
    ref_luma: &[f32],
    src_luma: &[f32],
    w: usize,
    h: usize,
) -> f64 {
    let params = RegistrationParams {
        tx: p[0],
        ty: p[1],
        scale: p[2],
        rotate: p[3],
    };
    let fwd = params_to_matrix(&params, w, h).expect("finite params");
    let inv = fwd.try_inverse().expect("invertible");
    // Only pixels whose inverse-mapped position lies inside the reference take
    // part (the same rule as the bicubic sampler's zero-fill region); the
    // objective leaves the others out of both the means and the sum.
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let p_src = inv * Vector3::new(x as f32, y as f32, 1.0_f32);
            let denom = p_src[2];
            if denom.abs() < 1.0e-8 {
                continue;
            }
            let (sx, sy) = (p_src[0] / denom, p_src[1] / denom);
            let inside = sx.is_finite()
                && sy.is_finite()
                && sx >= 0.0
                && sy >= 0.0
                && sx <= (w as f32 - 1.0)
                && sy <= (h as f32 - 1.0);
            if !inside {
                continue;
            }
            let r = bicubic_interpolate_channel(ref_luma, w, h, sx, sy);
            pairs.push((f64::from(src_luma[y * w + x]), f64::from(r)));
        }
    }
    let n = pairs.len() as f64;
    let mean_src = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_ref = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let var: f64 = pairs
        .iter()
        .map(|&(s, r)| {
            let d = (s - mean_src) - (r - mean_ref);
            d * d
        })
        .sum();
    (var / (n - 1.0).max(1.0)).sqrt()
}

const PARAM_SETS: [[f64; 4]; 4] = [
    [0.0, 0.0, 1.0, 0.0],
    [0.013, -0.009, 1.0, 0.0],
    [-0.02, 0.011, 1.015, 0.004],
    [0.004, 0.002, 0.985, -0.012],
];

/// PERF-1: the objective equals the plain evaluation on a level small enough to
/// be summed serially and on one large enough for the parallel reduction.
#[test]
fn objective_matches_straightforward_evaluation() {
    // 96×80 = 7 680 px (serial) and 160×128 = 20 480 px (rayon).
    for &(w, h) in &[(96_usize, 80_usize), (160, 128)] {
        let reference = textured(w, h);
        let mut source = textured(w, h);
        // Make the source differ from the reference so the RMS is not ~0.
        for (i, v) in source.luma.iter_mut().enumerate() {
            *v = (*v + ((i * 31) % 17) as f32 * 0.004).min(1.0);
        }
        for p in &PARAM_SETS {
            let got = objective_at_level(p, &reference.luma, &source.luma, w, h);
            let want = reference_objective(p, &reference.luma, &source.luma, w, h);
            assert!(
                (got - want).abs() <= 1.0e-9 * want.abs().max(1.0e-3),
                "{w}x{h} params {p:?}: objective {got:.12e} vs reference {want:.12e}"
            );
        }
    }
}

/// PERF-1: non-finite parameters still give the penalty, not a panic or NaN.
#[test]
fn objective_penalty_for_non_finite_params_is_unchanged() {
    let img = textured(64, 48);
    let bad = [f64::NAN, 0.0, 1.0, 0.0];
    let v = objective_at_level(&bad, &img.luma, &img.luma, 64, 48);
    assert_eq!(v.to_bits(), 1.0e6_f64.to_bits());
}

fn shifted_pair(w: usize, h: usize) -> (PlanarImage<f32>, PlanarImage<f32>) {
    let reference = textured(w, h);
    let source = warp_image(&reference, &translation(1.7, -0.9)).expect("warp");
    (reference, source)
}

/// PERF-1: supplying the reference pyramid gives bit-identical matrices, and one
/// pyramid serves several sources.
#[test]
fn refine_with_prebuilt_reference_pyramid_is_identical() {
    // 128×96: levels of 64×48 and 128×96 px, both below the serial threshold.
    let (reference, source_a) = shifted_pair(128, 96);
    let source_b = warp_image(&reference, &translation(-2.2, 0.6)).expect("warp");
    let opts = RefineOptions {
        max_iterations: 60,
        ..RefineOptions::default()
    };
    let pyramid = LumaPyramid::new(&reference);
    assert_eq!(pyramid.n_levels(), 2);
    for source in [&source_a, &source_b] {
        let plain = refine_alignment(&reference, source, &Matrix3::identity(), &opts).expect("ok");
        let shared = refine_alignment_with_pyramid(
            &reference,
            &pyramid,
            source,
            &Matrix3::identity(),
            &opts,
        )
        .expect("ok");
        assert_eq!(plain, shared, "matrices must be bit-identical");
    }
}

#[test]
fn refine_bounded_with_prebuilt_reference_pyramid_is_identical() {
    let (reference, source) = shifted_pair(128, 96);
    let opts = BoundedRefineOptions {
        max_iterations: 60,
        ..BoundedRefineOptions::default()
    };
    let pyramid = LumaPyramid::new(&reference);
    let plain =
        refine_alignment_bounded(&reference, &source, &Matrix3::identity(), &opts).expect("ok");
    let shared = refine_alignment_bounded_with_pyramid(
        &reference,
        &pyramid,
        &source,
        &Matrix3::identity(),
        &opts,
    )
    .expect("ok");
    assert_eq!(plain, shared, "matrices must be bit-identical");
}

/// A pyramid built from a different-sized image is rejected, not silently used.
#[test]
fn refine_rejects_a_pyramid_of_another_size() {
    let (reference, source) = shifted_pair(128, 96);
    let wrong = LumaPyramid::new(&textured(96, 96));
    let r = refine_alignment_with_pyramid(
        &reference,
        &wrong,
        &source,
        &Matrix3::identity(),
        &RefineOptions::default(),
    );
    assert!(matches!(r, Err(StackerError::AlignmentFailed(_))), "{r:?}");
    let r = refine_alignment_bounded_with_pyramid(
        &reference,
        &wrong,
        &source,
        &Matrix3::identity(),
        &BoundedRefineOptions::default(),
    );
    assert!(matches!(r, Err(StackerError::AlignmentFailed(_))), "{r:?}");
}

/// PERF-4: the spline weights are now built once per pixel and shared by the
/// three planes; every plane still equals the per-plane scalar sampler.
#[test]
fn clamped_warp_equals_per_plane_scalar_sampler_bit_for_bit() {
    let (w, h) = (37_usize, 29_usize);
    let img = textured(w, h);
    let mut m = Matrix3::<f32>::identity();
    let ang = 3.0_f32.to_radians();
    m[(0, 0)] = ang.cos() * 1.03;
    m[(0, 1)] = -ang.sin() * 1.03;
    m[(1, 0)] = ang.sin() * 1.03;
    m[(1, 1)] = ang.cos() * 1.03;
    m[(0, 2)] = 2.4;
    m[(1, 2)] = -1.3;
    let warped = warp_image_clamped(&img, &m).expect("warp");

    // Same inverse-map operation order as the warp (see `RowMap`).
    let inv = m.try_inverse().expect("invertible");
    for y in 0..h {
        let yf = y as f32;
        let c0 = inv[(0, 1)] * yf + inv[(0, 2)];
        let c1 = inv[(1, 1)] * yf + inv[(1, 2)];
        let c2 = inv[(2, 1)] * yf + inv[(2, 2)];
        for x in 0..w {
            let xf = x as f32;
            let r = 1.0 / (inv[(2, 0)] * xf + c2);
            let sx = (inv[(0, 0)] * xf + c0) * r;
            let sy = (inv[(1, 0)] * xf + c1) * r;
            let i = y * w + x;
            for (got, plane, name) in [
                (warped.luma[i], &img.luma, "luma"),
                (warped.chroma_a[i], &img.chroma_a, "chroma_a"),
                (warped.chroma_b[i], &img.chroma_b, "chroma_b"),
            ] {
                let want = spline4x4_sample_clamped(plane, w, h, sx, sy);
                assert_eq!(got.to_bits(), want.to_bits(), "{name} at ({x},{y})");
            }
        }
    }
}

/// PERF-3: the report helpers describe the compile-time configuration.
#[test]
fn simd_build_report_matches_the_compiled_features() {
    let report = simd_build_report();
    assert_ne!(report, "");
    assert_eq!(has_hardware_fma(), cfg!(target_feature = "fma"));
    if cfg!(target_feature = "avx512f") {
        assert!(report.starts_with("avx512f"), "{report}");
    } else if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
        assert!(report.starts_with("avx2+fma"), "{report}");
    } else if cfg!(any(target_arch = "x86", target_arch = "x86_64"))
        && !cfg!(target_feature = "avx2")
    {
        assert!(report.starts_with("baseline x86"), "{report}");
    }
}
