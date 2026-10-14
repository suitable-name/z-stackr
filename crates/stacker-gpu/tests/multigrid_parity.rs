//! The GPU multigrid solver against `stacker_algo::relief::multigrid`: the padded
//! finest level after `solve()` must agree sample for sample (both run `f32`
//! Jacobi sweeps with the same operations in the same order; the only difference
//! a driver can introduce is a fused multiply-add). Without a GPU adapter the tests
//! print `SKIPPED` and pass.

use std::sync::Arc;

use stacker_algo::relief::multigrid::MultigridSolver;
use stacker_gpu::{
    Failure, GpuContext, GpuError,
    multigrid::{self, MultigridPlan},
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

/// Pins: about one pixel in `every` is pinned to a frame index below `frames`.
fn pins(width: usize, height: usize, every: u32, frames: u32) -> (Vec<u16>, Vec<f32>) {
    let mut state = 0x1357_9BDF_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let index = (0..width * height)
        .map(|_| u16::try_from(next() % frames).expect("small"))
        .collect();
    let weight = (0..width * height)
        .map(|_| {
            if next().is_multiple_of(every) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (index, weight)
}

/// The CPU's padded finest level after the solve.
fn cpu_padded(width: usize, height: usize, index: &[u16], weight: &[f32]) -> (Vec<f32>, usize) {
    let mut solver = MultigridSolver::from_index(width, height, index, weight);
    solver.solve();
    (solver.layers[0].a.clone(), solver.layers[0].width)
}

#[test]
fn the_solution_matches_the_cpu_for_several_shapes() {
    let Some(ctx) = context() else { return };
    // Single pixels and rows have no coarse level; 5x5 and 64x48 are solved by the
    // tail workgroup alone; the larger ones dispatch their finest levels and hand
    // the rest to the tail (the tail starts at 8192 samples).
    for &(w, h, every, frames) in &[
        (1_usize, 1_usize, 1_u32, 3_u32),
        (1, 9, 2, 5),
        (9, 1, 2, 5),
        (2, 2, 2, 4),
        (5, 5, 3, 6),
        (64, 48, 9, 40),
        (97, 61, 7, 40),
        (200, 150, 11, 99),
        (301, 77, 25, 12),
        (256, 256, 40, 99),
    ] {
        let (index, weight) = pins(w, h, every, frames);
        let (cpu, cpu_width) = cpu_padded(w, h, &index, &weight);
        let mut plan = MultigridPlan::new(&ctx, w, h, &index, &weight).expect("plan");
        let (gpu, gpu_width, gpu_height) = plan.solve_padded().expect("solve");
        assert_eq!((gpu_width, gpu_height * gpu_width), (cpu_width, cpu.len()));
        let worst = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let mean = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| f64::from((a - b).abs()))
            .sum::<f64>()
            / gpu.len() as f64;
        eprintln!(
            "{w}x{h}: worst {worst:.2e}, mean {mean:.2e}, {} dispatches",
            plan.dispatches()
        );
        assert!(worst < 1e-3, "{w}x{h}: worst {worst:.2e}");
        assert!(mean < 1e-4, "{w}x{h}: mean {mean:.2e}");
    }
}

#[test]
fn an_image_without_pins_is_all_zero() {
    let Some(ctx) = context() else { return };
    let (w, h) = (40, 30);
    let mut plan =
        MultigridPlan::new(&ctx, w, h, &vec![7_u16; w * h], &vec![0.0; w * h]).expect("plan");
    let (gpu, _, _) = plan.solve_padded().expect("solve");
    assert!(gpu.iter().all(|&v| v == 0.0));
}

#[test]
fn a_plan_solves_again_with_the_same_result() {
    let Some(ctx) = context() else { return };
    let (w, h) = (97, 61);
    let (index, weight) = pins(w, h, 7, 40);
    let mut plan = MultigridPlan::new(&ctx, w, h, &index, &weight).expect("plan");
    let (first, _, _) = plan.solve_padded().expect("solve");
    let (second, _, _) = plan.solve_padded().expect("solve again");
    assert_eq!(first, second);
}

#[test]
fn faults_and_bad_input_surface_as_errors() {
    let Some(ctx) = context() else { return };
    let (w, h) = (64, 48);
    let (index, weight) = pins(w, h, 9, 40);
    ctx.inject_failure(Failure::Plan);
    assert!(matches!(
        MultigridPlan::new(&ctx, w, h, &index, &weight),
        Err(GpuError::Injected(_))
    ));
    assert!(matches!(
        MultigridPlan::new(&ctx, w, h, &index[..10], &weight),
        Err(GpuError::SizeMismatch { .. })
    ));
    assert!(matches!(
        MultigridPlan::new(&ctx, w, h, &index, &weight[..10]),
        Err(GpuError::SizeMismatch { .. })
    ));
    assert!(matches!(
        MultigridPlan::new(&ctx, 0, h, &[], &[]),
        Err(GpuError::SizeMismatch { .. })
    ));
    let mut plan = MultigridPlan::new(&ctx, w, h, &index, &weight).expect("plan");
    ctx.inject_failure(Failure::Fold);
    assert!(matches!(plan.solve_padded(), Err(GpuError::Injected(_))));
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(plan.solve_padded(), Err(GpuError::Injected(_))));
    let (out, _, _) = plan.solve_padded().expect("solve after the faults");
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(ctx.take_error().is_none(), "no stray device error");
}

#[test]
fn the_memory_estimate_covers_the_hierarchy() {
    // 24 MP: well under the 3 GiB budget the pipeline allows.
    let bytes = multigrid::plan_bytes(6000, 4000);
    assert!(bytes > 16 * 6001 * 4001 && bytes < 1 << 31, "{bytes}");
}
