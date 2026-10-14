//! The GPU registration objective and warp against the CPU code of
//! `stacker_align`.
//!
//! The objective is compared through what the refiner sees: the number of pixels
//! that take part and the mean-corrected RMS. The warp is compared pixel by pixel
//! and its coverage mask against `coverage_mask`. Both differ from the CPU by
//! rounding only (fused multiply-adds, the division); a pixel whose source
//! coordinate lies within an ulp of the frame edge can fall on the other side.
//! Without a GPU adapter the tests print `SKIPPED` and pass.

#![expect(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    reason = "test formulas; indices and counts are far below 2^24"
)]

use std::sync::Arc;

use stacker_align::{
    Matrix3,
    refine::{RegistrationParams, objective_at_level, params_to_matrix},
    transform::{
        bicubic_interpolate_channel, coverage_mask, spline4x4_sample_clamped, warp_image,
        warp_image_clamped,
    },
};
use stacker_core::image::PlanarImage;
use stacker_gpu::{
    Failure, GpuContext, GpuError,
    align::{Moments, ObjectivePlan, Sampler, WarpPlan},
};

fn context() -> Option<Arc<GpuContext>> {
    match GpuContext::new(None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            None
        }
    }
}

/// A textured frame, shifted by `(dx, dy)` pixels.
fn frame(w: usize, h: usize, dx: f32, dy: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 - dx, y as f32 - dy);
            let v = 0.4
                + 0.2 * (0.21 * fx).sin() * (0.17 * fy).cos()
                + 0.1 * (0.05 * fx + 0.03 * fy).sin();
            let i = y * w + x;
            img.luma[i] = v;
            img.chroma_a[i] = 0.05 * (0.09 * fx).cos();
            img.chroma_b[i] = 0.04 * (0.08 * fy).sin();
        }
    }
    img
}

fn rows(m: &Matrix3<f32>) -> [[f32; 3]; 3] {
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

/// The CPU objective's moments, computed the way `refine` warps: `inverse * p`
/// and a division, bicubic pixels outside the frame left out.
fn cpu_moments(
    reference: &[f32],
    source: &[f32],
    w: usize,
    h: usize,
    inverse: &Matrix3<f32>,
    sampler: Sampler,
) -> Moments {
    let mut values = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32, y as f32);
            let q: [f32; 3] = std::array::from_fn(|r| {
                inverse[(r, 0)] * fx + inverse[(r, 1)] * fy + inverse[(r, 2)]
            });
            if q[2].abs() < 1.0e-8 {
                continue;
            }
            let (sx, sy) = (q[0] / q[2], q[1] / q[2]);
            let warped = match sampler {
                Sampler::Bicubic => {
                    let inside = sx.is_finite()
                        && sy.is_finite()
                        && sx >= 0.0
                        && sy >= 0.0
                        && sx <= w as f32 - 1.0
                        && sy <= h as f32 - 1.0;
                    if !inside {
                        continue;
                    }
                    bicubic_interpolate_channel(reference, w, h, sx, sy)
                }
                Sampler::SplineClamped => spline4x4_sample_clamped(reference, w, h, sx, sy),
            };
            values.push(f64::from(source[y * w + x]) - f64::from(warped));
        }
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    Moments {
        count: values.len() as u64,
        mean,
        m2: values.iter().map(|d| (d - mean) * (d - mean)).sum(),
    }
}

fn rms(m: Moments) -> f64 {
    (m.m2.max(0.0) / (m.count as f64 - 1.0)).sqrt()
}

#[test]
fn objective_moments_match_the_cpu() {
    let Some(ctx) = context() else { return };
    for &(w, h) in &[(257_usize, 131_usize), (640, 480), (64, 300)] {
        let reference = frame(w, h, 0.0, 0.0);
        let source = frame(w, h, 2.3, -1.4);
        for sampler in [Sampler::Bicubic, Sampler::SplineClamped] {
            let mut plan = ObjectivePlan::new(&ctx, &reference.luma, &source.luma, w, h, sampler)
                .expect("plan");
            for params in [
                [0.0, 0.0, 1.0, 0.0],
                [0.004, -0.003, 1.0, 0.0],
                [-0.02, 0.011, 1.02, 0.01],
                [0.1, 0.1, 0.97, -0.03],
            ] {
                let p = RegistrationParams {
                    tx: params[0],
                    ty: params[1],
                    scale: params[2],
                    rotate: params[3],
                };
                let inverse = params_to_matrix(&p, w, h)
                    .and_then(Matrix3::try_inverse)
                    .expect("an invertible candidate");
                let gpu = plan.moments(&rows(&inverse)).expect("moments");
                let cpu = cpu_moments(&reference.luma, &source.luma, w, h, &inverse, sampler);
                let (g, c) = (rms(gpu), rms(cpu));
                eprintln!(
                    "{w}x{h} {sampler:?} {params:?}: count {} / {}, rms {g:.9} / {c:.9}",
                    gpu.count, cpu.count
                );
                assert!(
                    gpu.count.abs_diff(cpu.count) <= 2,
                    "{w}x{h} {sampler:?}: {} vs {} pixels",
                    gpu.count,
                    cpu.count
                );
                assert!(
                    (g - c).abs() <= 1e-5 * c.max(1e-3),
                    "{w}x{h} {sampler:?} {params:?}: rms {g} vs {c}"
                );
                if sampler == Sampler::Bicubic {
                    // The library's own objective agrees with the reference above.
                    let lib = objective_at_level(&params, &reference.luma, &source.luma, w, h);
                    if cpu.count * 2 >= (w * h) as u64 {
                        assert!((lib - c).abs() < 1e-9, "{lib} vs {c}");
                    }
                }
            }
        }
    }
}

#[test]
fn an_evaluation_repeats_itself_exactly() {
    let Some(ctx) = context() else { return };
    let (w, h) = (300, 200);
    let reference = frame(w, h, 0.0, 0.0);
    let source = frame(w, h, 1.5, 0.5);
    let mut plan = ObjectivePlan::new(&ctx, &reference.luma, &source.luma, w, h, Sampler::Bicubic)
        .expect("plan");
    let inverse = rows(&Matrix3::new(
        1.0, 0.001, 1.2, -0.001, 1.0, 0.4, 0.0, 0.0, 1.0,
    ));
    let first = plan.moments(&inverse).expect("moments");
    for _ in 0..5 {
        assert_eq!(plan.moments(&inverse).expect("moments"), first);
    }
}

#[test]
fn the_async_calls_equal_the_blocking_ones_bit_for_bit() {
    let Some(ctx) = context() else { return };
    let (w, h) = (300, 200);
    let reference = frame(w, h, 0.0, 0.0);
    let source = frame(w, h, 1.5, 0.5);
    for sampler in [Sampler::Bicubic, Sampler::SplineClamped] {
        let mut plan =
            ObjectivePlan::new(&ctx, &reference.luma, &source.luma, w, h, sampler).expect("plan");
        let inverse = rows(&Matrix3::new(
            1.0, 0.001, 1.2, -0.001, 1.0, 0.4, 0.0, 0.0, 1.0,
        ));
        let blocking = plan.moments(&inverse).expect("moments");
        let awaited = pollster::block_on(plan.moments_async(&inverse)).expect("moments");
        assert_eq!(blocking.count, awaited.count);
        assert_eq!(blocking.mean.to_bits(), awaited.mean.to_bits());
        assert_eq!(blocking.m2.to_bits(), awaited.m2.to_bits());

        let matrix = Matrix3::new(1.01, 0.004, -1.7, -0.004, 1.01, 2.2, 0.0, 0.0, 1.0);
        let inverse = rows(&matrix.try_inverse().expect("invertible"));
        let mut warp = WarpPlan::new(&ctx, w, h).expect("plan");
        let (image, coverage) = warp.warp(&reference, &inverse, sampler).expect("warp");
        let (image_async, coverage_async) =
            pollster::block_on(warp.warp_async(&reference, &inverse, sampler)).expect("warp");
        assert_eq!(coverage, coverage_async);
        for (a, b) in [
            (&image.luma, &image_async.luma),
            (&image.chroma_a, &image_async.chroma_a),
            (&image.chroma_b, &image_async.chroma_b),
        ] {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }
    // The async path reports the same injected failures.
    let mut plan = ObjectivePlan::new(&ctx, &reference.luma, &source.luma, w, h, Sampler::Bicubic)
        .expect("plan");
    let identity = rows(&Matrix3::identity());
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(
        pollster::block_on(plan.moments_async(&identity)),
        Err(GpuError::Injected(_))
    ));
    assert!(pollster::block_on(plan.moments_async(&identity)).is_ok());
}

#[test]
fn the_warp_matches_the_cpu_and_its_coverage_mask() {
    let Some(ctx) = context() else { return };
    for &(w, h) in &[(257_usize, 131_usize), (640, 480)] {
        let src = frame(w, h, 0.0, 0.0);
        let mut plan = WarpPlan::new(&ctx, w, h).expect("plan");
        for matrix in [
            Matrix3::new(1.0, 0.0, 3.25, 0.0, 1.0, -2.5, 0.0, 0.0, 1.0),
            Matrix3::new(1.01, 0.004, -1.7, -0.004, 1.01, 2.2, 0.0, 0.0, 1.0),
            Matrix3::identity(),
        ] {
            let inverse = rows(&matrix.try_inverse().expect("invertible"));
            for (sampler, cpu) in [
                (Sampler::Bicubic, warp_image(&src, &matrix).expect("warp")),
                (
                    Sampler::SplineClamped,
                    warp_image_clamped(&src, &matrix).expect("warp"),
                ),
            ] {
                let (gpu, coverage) = plan.warp(&src, &inverse, sampler).expect("warp");
                let mask = coverage_mask(&matrix, w, h);
                let mismatched = mask.iter().zip(&coverage).filter(|(a, b)| a != b).count();
                assert!(
                    mismatched <= 2,
                    "{w}x{h} {sampler:?}: {mismatched} mask pixels"
                );
                let mut max = 0.0_f32;
                for (a, b) in [
                    (&cpu.luma, &gpu.luma),
                    (&cpu.chroma_a, &gpu.chroma_a),
                    (&cpu.chroma_b, &gpu.chroma_b),
                ] {
                    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                        // A pixel on the other side of the edge is black on one
                        // side only; the mask comparison above bounds those.
                        if mask[i] == coverage[i] {
                            max = max.max((x - y).abs());
                        }
                    }
                }
                eprintln!("{w}x{h} {sampler:?}: max {max:.2e}, mask mismatches {mismatched}");
                assert!(max < 2e-5, "{w}x{h} {sampler:?}: max difference {max}");
                if matrix == Matrix3::identity() {
                    assert!(coverage.iter().all(|&c| c));
                }
            }
        }
    }
}

#[test]
fn injected_failures_and_wrong_sizes_surface_as_errors() {
    let Some(ctx) = context() else { return };
    let (w, h) = (96, 64);
    let reference = frame(w, h, 0.0, 0.0);
    ctx.inject_failure(Failure::Plan);
    assert!(matches!(
        ObjectivePlan::new(
            &ctx,
            &reference.luma,
            &reference.luma,
            w,
            h,
            Sampler::Bicubic
        ),
        Err(GpuError::Injected(_))
    ));
    assert!(matches!(
        ObjectivePlan::new(
            &ctx,
            &reference.luma,
            &reference.luma[1..],
            w,
            h,
            Sampler::Bicubic
        ),
        Err(GpuError::SizeMismatch { .. })
    ));
    let mut plan = ObjectivePlan::new(
        &ctx,
        &reference.luma,
        &reference.luma,
        w,
        h,
        Sampler::Bicubic,
    )
    .expect("plan");
    let identity = rows(&Matrix3::identity());
    for at in [Failure::Fold, Failure::ReadBack] {
        ctx.inject_failure(at);
        assert!(matches!(
            plan.moments(&identity),
            Err(GpuError::Injected(_))
        ));
    }
    let after = plan.moments(&identity).expect("works after the faults");
    assert_eq!(after.count, (w * h) as u64);
    assert!(after.m2.abs() < 1e-9, "identical planes: {after:?}");

    let mut warp = WarpPlan::new(&ctx, w, h).expect("plan");
    assert!(matches!(
        warp.warp(&frame(w + 1, h, 0.0, 0.0), &identity, Sampler::Bicubic),
        Err(GpuError::SizeMismatch { .. })
    ));
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(
        warp.warp(&reference, &identity, Sampler::Bicubic),
        Err(GpuError::Injected(_))
    ));
    let (back, _) = warp
        .warp(&reference, &identity, Sampler::Bicubic)
        .expect("works after the fault");
    assert_eq!(back.luma, reference.luma, "the identity warp is exact");
    assert!(ctx.take_error().is_none(), "no stray device error");
}
