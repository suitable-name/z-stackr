//! GPU-vs-CPU dispatch and parity tests for
//! [`stacker_algo::strata::saliency::compute_saliency`]'s GPU-accelerated
//! Laplacian-magnitude pass.
//!
//! This file is compiled unconditionally (including in default, `gpu`-off
//! builds run by `run_clippy --all-targets`), so every test guards its own
//! GPU-specific behaviour with `#[cfg(feature = "gpu")]` rather than gating
//! the whole file — a default build still exercises the pure-CPU
//! determinism check, which is the only path a non-`gpu` build can reach.
//!
//! See `tests/gpu_fuse_parity.rs`'s module docs for the general pattern this
//! file follows: runtime-gated parity assertions (skip-with-a-note when no
//! adapter), max-abs tolerance `< 1e-3`, and a width not divisible by 16
//! (33 px) to exercise the row-stride padding in `strata::gpu`.

use stacker_algo::strata::saliency::compute_saliency;

/// A deterministic synthetic luma buffer with non-trivial local structure
/// (a 2-D sinusoid), so the Laplacian-magnitude pass produces a genuinely
/// varying, non-degenerate response across the image.
fn make_luma(width: usize, height: usize) -> Vec<f32> {
    let mut luma = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let v = ((x as f32) * 0.31).sin() * ((y as f32) * 0.19).cos();
            luma[y * width + x] = v.mul_add(0.5, 0.5).clamp(0.0, 1.0);
        }
    }
    luma
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// [`compute_saliency`] must be deterministic across repeated calls with
/// identical input — meaningful in every build (in a default, non-`gpu`
/// build there is no GPU call site at all; in a `gpu` build with no
/// adapter, `laplacian_magnitude_gpu` returns `None` and the CPU body below
/// it runs unmodified).
#[test]
fn compute_saliency_is_deterministic() {
    let (width, height) = (33, 20);
    let luma = make_luma(width, height);
    let a = compute_saliency(&luma, width, height);
    let b = compute_saliency(&luma, width, height);
    let diff = max_abs_diff(&a, &b);
    assert!(
        diff < 1e-3,
        "compute_saliency is not repeatable within tolerance: {diff}"
    );
}

/// GPU-vs-CPU parity for the Laplacian-magnitude pass specifically (before
/// the five CPU-only blur passes are applied) — this isolates the exact
/// step `strata::gpu::laplacian_magnitude_gpu` accelerates, rather than the
/// whole (mostly CPU-blur-dominated) `compute_saliency` pipeline.
#[cfg(feature = "gpu")]
#[test]
fn gpu_laplacian_matches_cpu_within_tolerance_including_unaligned_width() {
    use stacker_algo::strata::gpu::laplacian_magnitude_gpu;

    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_laplacian_matches_cpu_within_tolerance_including_unaligned_width: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    // 33 px wide: not a multiple of 16, exercising the row-stride padding.
    for &(width, height) in &[(33, 20), (16, 16), (64, 40)] {
        let luma = make_luma(width, height);

        // Reference: the CPU kernel is private to `strata::saliency`, so
        // recompute the identical 4-neighbour Laplacian magnitude here
        // directly (mirrors the private `laplacian_magnitude_cpu` exactly).
        let mut cpu = vec![0.0_f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let up_row = y.saturating_sub(1);
                let down_row = (y + 1).min(height.saturating_sub(1));
                let left_col = x.saturating_sub(1);
                let right_col = (x + 1).min(width.saturating_sub(1));
                let center = luma[y * width + x];
                let up = luma[up_row * width + x];
                let down = luma[down_row * width + x];
                let left = luma[y * width + left_col];
                let right = luma[y * width + right_col];
                cpu[y * width + x] = 4.0f32.mul_add(-center, up + down + left + right).abs();
            }
        }

        let gpu = laplacian_magnitude_gpu(&luma, width, height).expect(
            "adapter reported available by context() but laplacian_magnitude_gpu returned None",
        );
        let diff = max_abs_diff(&cpu, &gpu);
        assert!(
            diff < 1e-3,
            "GPU/CPU Laplacian-magnitude parity failed at {width}x{height}: max abs diff {diff} >= 1e-3"
        );
    }
}

/// `stacker_core::gpu::set_enabled(false)` must force `compute_saliency`
/// onto the CPU path, producing output that is EXACTLY (bit-for-bit) equal
/// to a direct CPU computation — not merely tolerance-close — since with
/// the switch off, `laplacian_magnitude_gpu` never runs at all (the runtime
/// switch is checked before `context()` even attempts adapter acquisition).
#[cfg(feature = "gpu")]
#[test]
fn set_enabled_false_forces_exact_cpu_path() {
    let (width, height) = (33, 20);
    let luma = make_luma(width, height);

    stacker_core::gpu::set_enabled(false);
    let disabled = compute_saliency(&luma, width, height);
    stacker_core::gpu::set_enabled(true);

    // Independent CPU-only reference computation (same algorithm as
    // `laplacian_magnitude_cpu`, but recomputed here rather than calling
    // the private function directly).
    let mut cpu = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let up_row = y.saturating_sub(1);
            let down_row = (y + 1).min(height.saturating_sub(1));
            let left_col = x.saturating_sub(1);
            let right_col = (x + 1).min(width.saturating_sub(1));
            let center = luma[y * width + x];
            let up = luma[up_row * width + x];
            let down = luma[down_row * width + x];
            let left = luma[y * width + left_col];
            let right = luma[y * width + right_col];
            cpu[y * width + x] = 4.0f32.mul_add(-center, up + down + left + right).abs();
        }
    }
    let mut img = stacker_core::image::PlanarImage {
        width,
        height,
        luma: cpu,
        chroma_a: vec![0.0; width * height],
        chroma_b: vec![0.0; width * height],
    };
    // Mirror compute_saliency's 5 blur passes to get an exact reference.
    for _ in 0..5 {
        img = stacker_algo::apex::pyramid::apply_gaussian_blur(&img);
    }

    assert_eq!(
        disabled, img.luma,
        "set_enabled(false) must make compute_saliency bit-for-bit match the CPU-only path"
    );
}
