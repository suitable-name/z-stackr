//! The GPU `Apex` against the CPU `Apex` on synthetic stacks.
//!
//! The kernels follow the CPU operation for operation, but a driver may fuse a
//! multiply and an add, so the comparison is a tolerance, not bit equality: the
//! mean difference must be tiny and only a handful of pixels may differ visibly
//! (a winner flipping between two frames whose energies differ by one rounding
//! step). Without a GPU adapter the tests print `SKIPPED` and pass.

#![expect(
    clippy::suboptimal_flops,
    reason = "the synthetic-image formulas read better without mul_add"
)]

use std::sync::Arc;

use stacker_algo::apex::fuse::build_and_fuse_pyramids;
use stacker_core::image::PlanarImage;
use stacker_gpu::{Failure, GpuContext, GpuError, apex::ApexPlan};

fn context() -> Option<Arc<GpuContext>> {
    match GpuContext::new(None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            None
        }
    }
}

/// A repeatable value in `[0, 1)` for a pixel and a seed.
#[expect(clippy::cast_precision_loss, reason = "a 24-bit value is exact in f32")]
fn hash(x: usize, y: usize, seed: u32) -> f32 {
    let mut h = u32::try_from(x)
        .unwrap_or(u32::MAX)
        .wrapping_mul(0x9E37_79B1)
        ^ u32::try_from(y)
            .unwrap_or(u32::MAX)
            .wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f32 / 16_777_216.0
}

/// Frame `k` of `n`: a smooth image whose fine detail is strongest where the
/// "depth" is near the frame's focus position, so different frames win different
/// regions.
#[expect(
    clippy::cast_precision_loss,
    reason = "indices and counts are far below 2^24"
)]
fn frame(k: usize, n: usize, w: usize, h: usize) -> PlanarImage<f32> {
    let mut luma = Vec::with_capacity(w * h);
    let mut ca = Vec::with_capacity(w * h);
    let mut cb = Vec::with_capacity(w * h);
    let focus = (k as f32 + 0.5) / n as f32;
    let spread = 2.0 * (n * n) as f32;
    for y in 0..h {
        for x in 0..w {
            let depth = 0.5 + 0.5 * (0.13 * x as f32).sin() * (0.11 * y as f32).cos();
            let sharp = (-(depth - focus) * (depth - focus) * spread).exp();
            let noise = hash(x, y, 1) - 0.5;
            luma.push(0.25 + 0.4 * depth + noise * (0.02 + 0.6 * sharp));
            ca.push(0.03 * (0.05 * x as f32).sin() + (hash(x, y, 2) - 0.5) * 0.2 * sharp);
            cb.push(0.03 * (0.07 * y as f32).cos() + (hash(x, y, 3) - 0.5) * 0.2 * sharp);
        }
    }
    PlanarImage::from_planes(w, h, luma, ca, cb).expect("planes have w * h samples")
}

fn stack(n: usize, w: usize, h: usize) -> Vec<PlanarImage<f32>> {
    (0..n).map(|k| frame(k, n, w, h)).collect()
}

struct Difference {
    max: f32,
    mean: f64,
    visible: usize,
    samples: usize,
}

fn compare(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> Difference {
    let mut max = 0.0_f32;
    let mut sum = 0.0_f64;
    let mut visible = 0;
    let mut samples = 0;
    for (pa, pb) in [
        (&a.luma, &b.luma),
        (&a.chroma_a, &b.chroma_a),
        (&a.chroma_b, &b.chroma_b),
    ] {
        for (x, y) in pa.iter().zip(pb) {
            let d = (x - y).abs();
            max = max.max(d);
            sum += f64::from(d);
            visible += usize::from(d > 1e-3);
            samples += 1;
        }
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample count is far below 2^52"
    )]
    let mean = sum / samples as f64;
    Difference {
        max,
        mean,
        visible,
        samples,
    }
}

fn run_gpu(
    ctx: &Arc<GpuContext>,
    frames: &[PlanarImage<f32>],
    levels: usize,
    use_color: bool,
    grit: bool,
) -> PlanarImage<f32> {
    let (w, h) = (frames[0].width, frames[0].height);
    let mut plan = ApexPlan::new(ctx, w, h, levels, use_color, grit).expect("plan");
    plan.begin(u32::try_from(frames.len()).expect("few frames"));
    for f in frames {
        plan.fold(f).expect("fold");
    }
    plan.finish().expect("finish")
}

#[test]
fn matches_the_cpu_for_every_setting_combination_and_odd_sizes() {
    let Some(ctx) = context() else { return };
    for &(w, h, levels) in &[(67, 45, 4), (128, 96, 6), (257, 131, 5), (33, 200, 8)] {
        let frames = stack(5, w, h);
        for use_color in [false, true] {
            for grit in [false, true] {
                let cpu = build_and_fuse_pyramids(&frames, levels, use_color, grit).reconstruct();
                let gpu = run_gpu(&ctx, &frames, levels, use_color, grit);
                let d = compare(&cpu, &gpu);
                eprintln!(
                    "{w}x{h} L{levels} color={use_color} grit={grit}: max {:.2e} mean {:.2e} visible {}/{}",
                    d.max, d.mean, d.visible, d.samples
                );
                assert!(
                    d.mean < 1e-5,
                    "{w}x{h} L{levels}: mean difference {:.2e}",
                    d.mean
                );
                assert!(
                    d.visible * 500 <= d.samples,
                    "{w}x{h} L{levels} color={use_color} grit={grit}: {} of {} samples differ visibly",
                    d.visible,
                    d.samples
                );
            }
        }
    }
}

#[test]
fn identical_frames_reproduce_the_frame() {
    // Every energy ties, so the first frame wins everywhere; the averaged base of
    // identical frames is the frame's own base. The result is the frame back.
    let Some(ctx) = context() else { return };
    let one = frame(2, 5, 97, 61);
    let frames = vec![one.clone(), one.clone(), one.clone()];
    let gpu = run_gpu(&ctx, &frames, 5, true, true);
    let d = compare(&one, &gpu);
    assert!(d.max < 1e-4, "round trip differs by {:.2e}", d.max);
}

#[test]
fn a_plan_can_be_reused_and_repeats_itself_exactly() {
    let Some(ctx) = context() else { return };
    let frames = stack(4, 150, 90);
    let mut plan = ApexPlan::new(&ctx, 150, 90, 5, false, true).expect("plan");
    let mut results = Vec::new();
    for _ in 0..2 {
        plan.begin(4);
        for f in &frames {
            plan.fold(f).expect("fold");
        }
        results.push(plan.finish().expect("finish"));
    }
    assert_eq!(results[0].luma, results[1].luma);
    assert_eq!(results[0].chroma_a, results[1].chroma_a);
    assert_eq!(results[0].chroma_b, results[1].chroma_b);
}

#[test]
fn a_frame_of_the_wrong_size_is_refused() {
    let Some(ctx) = context() else { return };
    let mut plan = ApexPlan::new(&ctx, 64, 48, 4, false, false).expect("plan");
    plan.begin(1);
    let wrong = frame(0, 1, 65, 48);
    assert!(matches!(
        plan.fold(&wrong),
        Err(GpuError::SizeMismatch {
            expected: (64, 48),
            got: (65, 48)
        })
    ));
}

#[test]
fn an_image_the_device_cannot_hold_is_refused_not_crashed() {
    let Some(ctx) = context() else { return };
    // Far beyond any buffer limit: 3 planes of 2^20 x 2^20 f32.
    let result = ApexPlan::new(&ctx, 1 << 20, 1 << 20, 6, false, false);
    assert!(matches!(result, Err(GpuError::TooLarge { .. })));
}

#[test]
fn the_memory_estimate_matches_the_allocation() {
    let Some(ctx) = context() else { return };
    for &(w, h, levels) in &[(67, 45, 4), (256, 192, 8), (1000, 600, 6)] {
        let plan = ApexPlan::new(&ctx, w, h, levels, false, true).expect("plan");
        let image = (3 * w * h * 4) as u64;
        assert_eq!(
            plan.device_bytes() + 3 * image,
            stacker_gpu::apex::plan_bytes(w, h, levels),
            "{w}x{h} L{levels}: device bytes plus staging and read-back"
        );
    }
}

#[test]
fn injected_failures_surface_as_errors_and_the_plan_recovers() {
    let Some(ctx) = context() else { return };
    let frames = stack(3, 80, 60);

    // Building a plan fails once; the next attempt works.
    ctx.inject_failure(Failure::Plan);
    assert!(matches!(
        ApexPlan::new(&ctx, 80, 60, 4, false, true),
        Err(GpuError::Injected(_))
    ));
    let mut plan = ApexPlan::new(&ctx, 80, 60, 4, false, true).expect("plan after the fault");

    // Folding a frame fails once, and a whole stack still works afterwards.
    plan.begin(3);
    ctx.inject_failure(Failure::Fold);
    assert!(matches!(plan.fold(&frames[0]), Err(GpuError::Injected(_))));
    let expected = run_gpu(&ctx, &frames, 4, false, true);
    plan.begin(3);
    for f in &frames {
        plan.fold(f).expect("fold after the fault");
    }
    let again = plan.finish().expect("finish after the fault");
    assert_eq!(expected.luma, again.luma);

    // Reading the result back fails once.
    plan.begin(3);
    for f in &frames {
        plan.fold(f).expect("fold");
    }
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(plan.finish(), Err(GpuError::Injected(_))));
    assert!(
        ctx.take_error().is_none(),
        "no stray device error is left over"
    );
}
