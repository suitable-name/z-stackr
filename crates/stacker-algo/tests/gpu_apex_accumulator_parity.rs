//! GPU-vs-CPU dispatch and parity tests for the GPU-resident incremental
//! `Apex` accumulator (`apex::gpu::accumulator::GpuApexAccumulator`), and its
//! integration into [`fuse_pyramids_incremental`]/
//! [`fuse_pyramids_incremental_with_progress`].
//!
//! This file is compiled unconditionally (including in default, `gpu`-off
//! builds run by `run_clippy --all-targets`), so every test guards its own
//! GPU-specific behaviour with `#[cfg(feature = "gpu")]` rather than gating
//! the whole file. See `tests/gpu_fuse_parity.rs`'s module docs for the
//! general pattern: runtime-gated parity assertions (skip-with-a-note when
//! no adapter), max-abs tolerance `< 1e-3`, and a width not divisible by 16
//! (33 px) to exercise the row-stride padding.

#[cfg(feature = "gpu")]
use stacker_algo::apex::fuse::ApexAccumulator;
use stacker_algo::apex::fuse::fuse_pyramids_incremental;
use stacker_core::image::PlanarImage;

/// A small deterministic stack of frames with complementary sharp/blurred
/// halves plus per-frame phase drift, so both the residual (average) level
/// and the argmax levels see genuinely varying, non-degenerate input.
fn make_frames(width: usize, height: usize, count: usize) -> Vec<PlanarImage<f32>> {
    (0..count)
        .map(|f| {
            let mut img = PlanarImage::new(width, height);
            for y in 0..height {
                for x in 0..width {
                    let i = y * width + x;

                    let phase = f as f32 * 0.9;

                    let v = ((x as f32).mul_add(0.29, phase).sin()
                        * (y as f32).mul_add(0.17, -phase).cos())
                    .mul_add(0.5, 0.5)
                    .clamp(0.0, 1.0);
                    img.luma[i] = v;
                    img.chroma_a[i] = (v - 0.5) * 0.4;
                    img.chroma_b[i] = (0.5 - v) * 0.25;
                }
            }
            img
        })
        .collect()
}

fn max_abs_diff(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> f32 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    let mut worst = 0.0_f32;
    for (x, y) in a.luma.iter().zip(b.luma.iter()) {
        worst = worst.max((x - y).abs());
    }
    for (x, y) in a.chroma_a.iter().zip(b.chroma_a.iter()) {
        worst = worst.max((x - y).abs());
    }
    for (x, y) in a.chroma_b.iter().zip(b.chroma_b.iter()) {
        worst = worst.max((x - y).abs());
    }
    worst
}

/// [`fuse_pyramids_incremental`] must be deterministic across repeated
/// calls — meaningful in every build (default build has no GPU call site;
/// `gpu` build with no adapter falls straight through to the CPU
/// accumulator).
#[test]
fn fuse_pyramids_incremental_is_deterministic() {
    for grit in [true, false] {
        for use_color in [true, false] {
            let frames = make_frames(24, 18, 4);
            let a = fuse_pyramids_incremental(&frames, use_color, grit);
            let b = fuse_pyramids_incremental(&frames, use_color, grit);
            let diff = max_abs_diff(&a, &b);
            assert!(
                diff < 1e-3,
                "fuse_pyramids_incremental is not repeatable within tolerance \
                 (use_color={use_color}, grit={grit}): {diff}"
            );
        }
    }
}

/// GPU-vs-CPU parity: [`GpuApexAccumulator`] blended across several frames
/// must match the plain CPU [`ApexAccumulator`] within tolerance. Only runs
/// its real assertion when an adapter is present; otherwise prints a skip
/// note and passes.
#[cfg(feature = "gpu")]
#[test]
fn gpu_accumulator_matches_cpu_within_tolerance_including_unaligned_width() {
    use stacker_algo::apex::gpu::accumulator::GpuApexAccumulator;

    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_accumulator_matches_cpu_within_tolerance_including_unaligned_width: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    // 33 px wide: not a multiple of 16, exercising the row-stride padding.
    for &(width, height) in &[(33, 20), (16, 16), (64, 40)] {
        for grit in [true, false] {
            for use_color in [true, false] {
                let frames = make_frames(width, height, 4);

                let mut gpu_acc =
                    GpuApexAccumulator::new(&frames[0], use_color, grit).expect(
                        "adapter reported available by context() but GpuApexAccumulator::new returned None",
                    );
                for frame in &frames[1..] {
                    assert!(
                        gpu_acc.accumulate(frame),
                        "GpuApexAccumulator::accumulate unexpectedly failed on a host with an available adapter"
                    );
                }
                let gpu_result = gpu_acc
                    .finish()
                    .expect("GpuApexAccumulator::finish unexpectedly failed after successful accumulate calls");

                let mut cpu_acc = ApexAccumulator::new(&frames[0], use_color, grit);
                for frame in &frames[1..] {
                    cpu_acc.blend(frame);
                }
                let cpu_result = cpu_acc.reconstruct();

                let diff = max_abs_diff(&gpu_result, &cpu_result);
                assert!(
                    diff < 1e-3,
                    "GPU/CPU incremental-accumulator parity failed at {width}x{height} \
                     (use_color={use_color}, grit={grit}): max abs diff {diff} >= 1e-3"
                );
            }
        }
    }
}

/// `GpuApexAccumulator::read_back` mid-stack must reflect exactly the
/// frames accumulated so far — i.e. matches a CPU accumulator fed the same
/// PREFIX of frames, not the full stack. This is the exact invariant the
/// mid-stack GPU-failure hand-off in `fuse_pyramids_incremental_with_progress`
/// depends on (see `apex::gpu::accumulator`'s module docs).
#[cfg(feature = "gpu")]
#[test]
fn gpu_accumulator_read_back_mid_stack_matches_cpu_prefix() {
    use stacker_algo::apex::gpu::accumulator::GpuApexAccumulator;

    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_accumulator_read_back_mid_stack_matches_cpu_prefix: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    let (width, height) = (33, 20);
    let frames = make_frames(width, height, 5);

    let mut gpu_acc = GpuApexAccumulator::new(&frames[0], false, true)
        .expect("adapter available but GpuApexAccumulator::new returned None");
    assert!(
        gpu_acc.accumulate(&frames[1]),
        "accumulate unexpectedly failed on a host with an available adapter"
    );
    assert!(
        gpu_acc.accumulate(&frames[2]),
        "accumulate unexpectedly failed on a host with an available adapter"
    );
    // Stop after 3 frames (indices 0..=2) and read back — this must match a
    // CPU accumulator fed exactly those same 3 frames, not all 5.
    let levels = gpu_acc
        .read_back()
        .expect("read_back unexpectedly failed on a host with an available adapter");
    let gpu_prefix_result = stacker_algo::apex::pyramid::LaplacianPyramid { levels }.reconstruct();

    let mut cpu_acc = ApexAccumulator::new(&frames[0], false, true);
    cpu_acc.blend(&frames[1]);
    cpu_acc.blend(&frames[2]);
    let cpu_prefix_result = cpu_acc.reconstruct();

    let diff = max_abs_diff(&gpu_prefix_result, &cpu_prefix_result);
    assert!(
        diff < 1e-3,
        "GPU accumulator's mid-stack read_back does not match the CPU accumulator's \
         3-frame prefix: max abs diff {diff} >= 1e-3"
    );
}

/// `stacker_core::gpu::set_enabled(false)` must force
/// `fuse_pyramids_incremental` onto the CPU accumulator path
/// (`GpuApexAccumulator::new` returns `None` because `stacker_core::gpu::context()`
/// does), producing output that is EXACTLY (bit-for-bit) equal to a direct
/// CPU-only computation.
#[cfg(feature = "gpu")]
#[test]
fn set_enabled_false_forces_exact_cpu_path() {
    let (width, height) = (33, 20);
    let frames = make_frames(width, height, 4);

    stacker_core::gpu::set_enabled(false);
    let disabled_result = fuse_pyramids_incremental(&frames, false, true);
    stacker_core::gpu::set_enabled(true);

    let mut cpu_acc = ApexAccumulator::new(&frames[0], false, true);
    for frame in &frames[1..] {
        cpu_acc.blend(frame);
    }
    let cpu_result = cpu_acc.reconstruct();

    assert_eq!(
        disabled_result.luma, cpu_result.luma,
        "set_enabled(false) must make fuse_pyramids_incremental bit-for-bit match the CPU accumulator"
    );
    assert_eq!(disabled_result.chroma_a, cpu_result.chroma_a);
    assert_eq!(disabled_result.chroma_b, cpu_result.chroma_b);
}
