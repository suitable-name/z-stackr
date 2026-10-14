//! The GPU `Strata` kernels against the CPU code of `stacker_algo`: the saliency
//! (`strata::saliency::compute_saliency`) and the guided-filter pair
//! (`relief::guided::guided_filter_pair_luma`).
//!
//! The saliency runs the CPU's taps in the CPU's order on `f32`, so the maps
//! agree to `1e-5` of their range. The guided filter sums its windows directly in
//! `f32` over shifted values where the CPU reads `f64` summed-area tables, so it
//! agrees to a stated tolerance (worst sample below `1e-3`, mean below `1e-5`, for
//! masks in `0..=1`). Without a GPU adapter the tests print `SKIPPED` and pass.

#![expect(
    clippy::cast_precision_loss,
    reason = "test formulas; counts are far below 2^24, and a 24-bit hash fits an f32"
)]

use std::sync::Arc;

use stacker_algo::{relief::guided::guided_filter_pair_luma, strata::saliency::compute_saliency};
use stacker_gpu::{Failure, GpuContext, GpuError, strata::StrataPlan};

const R_BIG: usize = 45;
const EPS_BIG: f32 = 0.3;
const R_SMALL: usize = 7;
const EPS_SMALL: f32 = 1e-6;

fn context() -> Option<Arc<GpuContext>> {
    match GpuContext::new(None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            None
        }
    }
}

/// A plane of blocks with edges and noise, in `0.05..0.55` (a dark linear base).
fn luma_plane(width: usize, height: usize, seed: u32) -> Vec<f32> {
    let mut state = 0x9E37_79B9_u32 ^ seed.wrapping_mul(0x85EB_CA77);
    (0..width * height)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state >> 8) as f32 / 16_777_216.0;
            let (x, y) = (i % width, i / width);
            let edge = if (x / 17 + y / 13).is_multiple_of(2) {
                0.3
            } else {
                0.0
            };
            0.2_f32.mul_add(noise, 0.05 + edge)
        })
        .collect()
}

/// A mask in `0..=1` like the one Strata refines.
fn mask_plane(width: usize, height: usize) -> Vec<f32> {
    luma_plane(width, height, 7)
        .iter()
        .map(|v| ((v - 0.05) * 2.0).clamp(0.0, 1.0))
        .collect()
}

fn worst_and_mean(a: &[f32], b: &[f32]) -> (f32, f64) {
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    let mean = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .sum::<f64>()
        / a.len().max(1) as f64;
    (worst, mean)
}

#[test]
fn the_saliency_matches_the_cpu() {
    let Some(ctx) = context() else { return };
    // Odd sizes, a width below the blur's reach, and single rows and columns.
    for &(w, h) in &[
        (130_usize, 90_usize),
        (61, 47),
        (200, 33),
        (16, 16),
        (7, 40),
        (40, 7),
        (2, 9),
        (9, 2),
        (1, 12),
        (12, 1),
        (1, 1),
    ] {
        let luma = luma_plane(w, h, 1);
        let mut plan = StrataPlan::new(&ctx, w, h).expect("plan");
        let gpu = plan.saliency(&luma).expect("saliency");
        let cpu = compute_saliency(&luma, w, h);
        let scale = cpu.iter().copied().fold(0.0_f32, f32::max);
        let (worst, _) = worst_and_mean(&gpu, &cpu);
        eprintln!("{w}x{h}: saliency worst {worst:.2e} of range {scale:.2e}");
        assert!(
            worst <= 1e-5 * scale.max(f32::MIN_POSITIVE),
            "{w}x{h}: saliency differs by {worst:.2e} of {scale:.2e}"
        );
    }
}

#[test]
fn the_guided_pair_matches_the_cpu() {
    let Some(ctx) = context() else { return };
    for &(w, h, ra, rb) in &[
        (130_usize, 90_usize, R_BIG, R_SMALL),
        (61, 47, R_BIG, R_SMALL),
        // Radii larger than the image are the whole image.
        (40, 30, 100, 3),
        (200, 33, 12, 1),
        (9, 1, 2, 0),
    ] {
        let guidance = luma_plane(w, h, 3);
        let mask = mask_plane(w, h);
        let mut plan = StrataPlan::new(&ctx, w, h).expect("plan");
        let (gpu_a, gpu_b) = plan
            .guided_pair(&guidance, &mask, (ra, EPS_BIG), (rb, EPS_SMALL))
            .expect("guided pair");
        let (cpu_a, cpu_b) =
            guided_filter_pair_luma(&guidance, &mask, w, h, ra, EPS_BIG, rb, EPS_SMALL);
        for (name, gpu, cpu) in [("first", &gpu_a, &cpu_a), ("second", &gpu_b, &cpu_b)] {
            let (worst, mean) = worst_and_mean(gpu, cpu);
            eprintln!("{w}x{h} r{ra}/{rb}: {name} worst {worst:.2e} mean {mean:.2e}");
            assert!(worst < 1e-3, "{w}x{h} {name}: worst {worst:.2e}");
            assert!(mean < 1e-5, "{w}x{h} {name}: mean {mean:.2e}");
        }
    }
}

#[test]
fn the_self_guided_case_matches_the_cpu() {
    // The same slice as guidance and source is the case the Relief smoothing runs.
    let Some(ctx) = context() else { return };
    let (w, h) = (97, 61);
    let plane = luma_plane(w, h, 5);
    let mut plan = StrataPlan::new(&ctx, w, h).expect("plan");
    let (gpu, _) = plan
        .guided_pair(&plane, &plane, (6, 0.01), (2, 0.01))
        .expect("guided pair");
    let (cpu, _) = guided_filter_pair_luma(&plane, &plane, w, h, 6, 0.01, 2, 0.01);
    let (worst, mean) = worst_and_mean(&gpu, &cpu);
    assert!(
        worst < 1e-3 && mean < 1e-5,
        "worst {worst:.2e} mean {mean:.2e}"
    );
}

#[test]
fn a_plan_repeats_itself_exactly_and_serves_many_frames() {
    let Some(ctx) = context() else { return };
    let (w, h) = (120, 80);
    let mut plan = StrataPlan::new(&ctx, w, h).expect("plan");
    let first = luma_plane(w, h, 1);
    let second = luma_plane(w, h, 2);
    let sal_first = plan.saliency(&first).expect("saliency");
    let _ = plan.saliency(&second).expect("saliency");
    let sal_again = plan.saliency(&first).expect("saliency");
    assert_eq!(
        sal_first, sal_again,
        "an earlier frame leaves nothing behind"
    );
    let mask = mask_plane(w, h);
    let pair_first = plan
        .guided_pair(&first, &mask, (R_BIG, EPS_BIG), (R_SMALL, EPS_SMALL))
        .expect("pair");
    let _ = plan
        .saliency(&second)
        .expect("saliency between the guided filters");
    let pair_again = plan
        .guided_pair(&first, &mask, (R_BIG, EPS_BIG), (R_SMALL, EPS_SMALL))
        .expect("pair");
    assert_eq!(pair_first, pair_again);
}

#[test]
fn faults_and_bad_input_surface_as_errors() {
    let Some(ctx) = context() else { return };
    let (w, h) = (64, 48);
    let luma = luma_plane(w, h, 1);
    let mask = mask_plane(w, h);
    ctx.inject_failure(Failure::Plan);
    assert!(matches!(
        StrataPlan::new(&ctx, w, h),
        Err(GpuError::Injected(_))
    ));
    let mut plan = StrataPlan::new(&ctx, w, h).expect("plan");
    assert!(matches!(
        plan.saliency(&luma[..10]),
        Err(GpuError::SizeMismatch { .. })
    ));
    assert!(matches!(
        plan.guided_pair(&luma, &mask[..10], (3, 0.1), (3, 0.1)),
        Err(GpuError::SizeMismatch { .. })
    ));
    ctx.inject_failure(Failure::Fold);
    assert!(matches!(plan.saliency(&luma), Err(GpuError::Injected(_))));
    ctx.inject_failure(Failure::Fold);
    assert!(matches!(
        plan.guided_pair(&luma, &mask, (3, 0.1), (3, 0.1)),
        Err(GpuError::Injected(_))
    ));
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(plan.saliency(&luma), Err(GpuError::Injected(_))));
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(
        plan.guided_pair(&luma, &mask, (3, 0.1), (3, 0.1)),
        Err(GpuError::Injected(_))
    ));
    // The plan works after the faults.
    let out = plan.saliency(&luma).expect("saliency after the faults");
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(ctx.take_error().is_none(), "no stray device error");
}
