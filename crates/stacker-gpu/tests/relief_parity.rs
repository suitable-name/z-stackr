//! The GPU `Relief` against the CPU code of `stacker_algo::relief` (focus measure,
//! selection with the first frame winning ties, mean below the threshold,
//! self-guided filter).
//!
//! The CPU sums the focus windows in `f64`, the device in `f32` on shifted
//! values, so the focus maps agree to a relative tolerance and the winner can
//! flip where two frames are within rounding of each other; the fused image is
//! compared by the mean difference and the share of visibly different samples.
//! Without a GPU adapter the tests print `SKIPPED` and pass.

#![expect(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    reason = "test formulas; indices and counts are far below 2^24"
)]

use std::sync::Arc;

use stacker_algo::relief::{
    focus::compute_focus_map,
    guided::guided_filter_luma,
    threshold::{compute_relief_threshold, mask_above},
};
use stacker_core::image::PlanarImage;
use stacker_gpu::{Failure, GpuContext, GpuError, relief::ReliefPlan};

const EPS: f32 = 0.01;

fn context() -> Option<Arc<GpuContext>> {
    match GpuContext::new(None) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            None
        }
    }
}

fn hash(col: usize, row: usize, seed: u32) -> f32 {
    let index = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    let mut state = index(col).wrapping_mul(0x9E37_79B1)
        ^ index(row).wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    state ^= state >> 15;
    state = state.wrapping_mul(0x2C1B_3C6D);
    state ^= state >> 12;
    (state >> 8) as f32 / 16_777_216.0
}

/// Frame `k` of `n`: texture that is strongest where the depth is near the
/// frame's focus position, on a dark linear base (as decoded photographs are).
fn frame(index: usize, count: usize, width: usize, height: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    let focus = (index as f32 + 0.5) / count as f32;
    for row in 0..height {
        for col in 0..width {
            let depth = 0.5 + 0.5 * (0.05 * col as f32).sin() * (0.04 * row as f32).cos();
            let sharp = (-(depth - focus) * (depth - focus) * 30.0).exp();
            let i = row * width + col;
            img.luma[i] = 0.08 + 0.05 * depth + (hash(col, row, 1) - 0.5) * (0.005 + 0.1 * sharp);
            img.chroma_a[i] = 0.02 * (0.03 * col as f32).sin() + (hash(col, row, 2) - 0.5) * 0.02;
            img.chroma_b[i] = 0.02 * (0.02 * row as f32).cos() + (hash(col, row, 3) - 0.5) * 0.02;
        }
    }
    img
}

/// The CPU `Relief` with a given threshold, as `stacker_pipeline::fuse` runs it.
fn cpu_relief(
    frames: &[PlanarImage<f32>],
    radius: usize,
    smoothing: usize,
    pct: f32,
) -> (Vec<f32>, PlanarImage<f32>) {
    let (width, height) = (frames[0].width, frames[0].height);
    let pixels = width * height;
    let maps: Vec<Vec<f32>> = frames
        .iter()
        .map(|frame| compute_focus_map(frame, radius))
        .collect();
    let mut max = vec![f32::NEG_INFINITY; pixels];
    for map in &maps {
        for (best, &value) in max.iter_mut().zip(map) {
            if value > *best {
                *best = value;
            }
        }
    }
    let threshold = compute_relief_threshold(&max, pct, false).value;
    let mask = mask_above(&max, threshold);
    let count = frames.len() as f32;
    let mut out = PlanarImage::new(width, height);
    for i in 0..pixels {
        if mask[i] {
            let mut best = f32::NEG_INFINITY;
            for (frame, map) in frames.iter().zip(&maps) {
                if map[i] > best {
                    best = map[i];
                    out.luma[i] = frame.luma[i];
                    out.chroma_a[i] = frame.chroma_a[i];
                    out.chroma_b[i] = frame.chroma_b[i];
                }
            }
        } else {
            let mut sums = [0.0_f32; 3];
            for frame in frames {
                sums[0] += frame.luma[i];
                sums[1] += frame.chroma_a[i];
                sums[2] += frame.chroma_b[i];
            }
            out.luma[i] = sums[0] / count;
            out.chroma_a[i] = sums[1] / count;
            out.chroma_b[i] = sums[2] / count;
        }
    }
    out.luma = guided_filter_luma(&out.luma, &out.luma, width, height, smoothing, EPS);
    (max, out)
}

fn gpu_relief(
    ctx: &Arc<GpuContext>,
    frames: &[PlanarImage<f32>],
    radius: usize,
    smoothing: usize,
    pct: f32,
) -> (Vec<f32>, PlanarImage<f32>) {
    let (w, h) = (frames[0].width, frames[0].height);
    let mut plan = ReliefPlan::new(ctx, w, h).expect("plan");
    plan.begin(u32::try_from(frames.len()).expect("few"), radius)
        .expect("radius");
    for f in frames {
        plan.fold(f).expect("fold");
    }
    let max = plan.focus_max().expect("focus map");
    let threshold = compute_relief_threshold(&max, pct, false).value;
    let out = plan.finish(threshold, smoothing, EPS).expect("finish");
    (max, out)
}

#[test]
fn matches_the_cpu_for_several_radii_and_sizes() {
    let Some(ctx) = context() else { return };
    for &(w, h, radius, smoothing, pct) in &[
        (97_usize, 61_usize, 1_usize, 0_usize, 0.0_f32),
        (160, 120, 5, 2, 0.0),
        (257, 131, 5, 2, 0.3),
        (128, 200, 12, 6, 0.5),
        (64, 48, 40, 3, 0.1),
    ] {
        let frames: Vec<PlanarImage<f32>> = (0..5).map(|k| frame(k, 5, w, h)).collect();
        let (cpu_max, cpu) = cpu_relief(&frames, radius, smoothing, pct);
        let (gpu_max, gpu) = gpu_relief(&ctx, &frames, radius, smoothing, pct);

        let scale = cpu_max.iter().copied().fold(0.0_f32, f32::max);
        let focus_err = cpu_max
            .iter()
            .zip(&gpu_max)
            .map(|(c, g)| (c - g).abs())
            .fold(0.0_f32, f32::max)
            / scale;
        let mut sum = 0.0_f64;
        let mut visible = 0_usize;
        for (a, b) in [
            (&cpu.luma, &gpu.luma),
            (&cpu.chroma_a, &gpu.chroma_a),
            (&cpu.chroma_b, &gpu.chroma_b),
        ] {
            for (x, y) in a.iter().zip(b.iter()) {
                let d = (x - y).abs();
                sum += f64::from(d);
                visible += usize::from(d > 1e-3);
            }
        }
        let samples = 3 * w * h;
        let mean = sum / samples as f64;
        eprintln!(
            "{w}x{h} r{radius} s{smoothing} pct {pct}: focus max rel err {focus_err:.2e}, \
             mean {mean:.2e}, visible {visible}/{samples}"
        );
        assert!(
            focus_err < 1e-3,
            "focus map differs by {focus_err:.2e} of its range"
        );
        assert!(mean < 1e-5, "mean difference {mean:.2e}");
        assert!(
            visible * 500 <= samples,
            "{visible} of {samples} samples differ visibly"
        );
    }
}

#[test]
fn identical_frames_reproduce_the_frame_before_smoothing() {
    let Some(ctx) = context() else { return };
    let one = frame(1, 3, 90, 70);
    let frames = vec![one.clone(), one.clone(), one.clone()];
    // Smoothing radius 0: the guided filter returns its input.
    let (_, gpu) = gpu_relief(&ctx, &frames, 4, 0, 0.0);
    let max = one
        .luma
        .iter()
        .zip(&gpu.luma)
        .chain(one.chroma_a.iter().zip(&gpu.chroma_a))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max < 1e-6, "round trip differs by {max:.2e}");
}

#[test]
fn a_plan_repeats_itself_exactly() {
    let Some(ctx) = context() else { return };
    let frames: Vec<PlanarImage<f32>> = (0..4).map(|k| frame(k, 4, 120, 80)).collect();
    let (m1, a) = gpu_relief(&ctx, &frames, 5, 2, 0.2);
    let (m2, b) = gpu_relief(&ctx, &frames, 5, 2, 0.2);
    assert_eq!(m1, m2);
    assert_eq!(a.luma, b.luma);
    assert_eq!(a.chroma_b, b.chroma_b);
}

#[test]
fn faults_and_bad_input_surface_as_errors() {
    let Some(ctx) = context() else { return };
    let frames: Vec<PlanarImage<f32>> = (0..2).map(|k| frame(k, 2, 64, 48)).collect();
    ctx.inject_failure(Failure::Plan);
    assert!(matches!(
        ReliefPlan::new(&ctx, 64, 48),
        Err(GpuError::Injected(_))
    ));
    let mut plan = ReliefPlan::new(&ctx, 64, 48).expect("plan");
    assert!(matches!(plan.begin(2, 101), Err(GpuError::TooLarge { .. })));
    plan.begin(2, 3).expect("radius");
    assert!(matches!(
        plan.fold(&frame(0, 2, 65, 48)),
        Err(GpuError::SizeMismatch { .. })
    ));
    ctx.inject_failure(Failure::Fold);
    assert!(matches!(plan.fold(&frames[0]), Err(GpuError::Injected(_))));
    plan.begin(2, 3).expect("radius");
    for f in &frames {
        plan.fold(f).expect("fold");
    }
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(plan.focus_max(), Err(GpuError::Injected(_))));
    ctx.inject_failure(Failure::ReadBack);
    assert!(matches!(
        plan.finish(0.0, 2, EPS),
        Err(GpuError::Injected(_))
    ));
    let out = plan.finish(0.0, 2, EPS).expect("finish after the faults");
    assert!(out.luma.iter().all(|v| v.is_finite()));
    assert!(ctx.take_error().is_none(), "no stray device error");
}
