//! GPU-vs-CPU dispatch and parity tests for
//! [`stacker_align::transform::warp_image_clamped`].
//!
//! This file is compiled unconditionally (including in default, `gpu`-off
//! builds run by `run_clippy --all-targets`), so every test guards its own
//! GPU-specific behaviour with `#[cfg(feature = "gpu")]` rather than gating
//! the whole file.
//!
//! # What is tested here
//!
//! - **Dispatch/fallback** (`dispatch_matches_cpu_reference`): the public
//!   [`warp_image_clamped`] must always match its own pure-CPU reference
//!   implementation, [`warp_image_clamped_cpu`]. Without `gpu` compiled in,
//!   `warp_image_clamped` is a thin wrapper with no GPU call site, so this
//!   is an exact-equality regression check. With `gpu` compiled in, on a
//!   host with no `wgpu` adapter `warp_image_clamped_gpu` returns `Ok(None)`
//!   internally and `warp_image_clamped` falls through to
//!   `warp_image_clamped_cpu` — so the two must still match exactly.
//! - **Parity** (`gpu_result_matches_cpu_within_tolerance_including_unaligned_width`,
//!   `gpu`-feature-only): only runs its real assertions when
//!   `stacker_core::gpu::context()` returns `Some`; otherwise prints a skip
//!   note and passes. Compares
//!   [`stacker_align::transform::gpu::warp_image_clamped_gpu`]'s output
//!   against [`warp_image_clamped_cpu`] on synthetic images, including a
//!   width that is **not** a multiple of 16 (33 px), to exercise the `wgpu`
//!   `COPY_BYTES_PER_ROW_ALIGNMENT` row-stride padding/de-striding logic.
//!   Tolerance: max-abs diff `< 1e-3` (the epsilon documented on
//!   `transform::gpu`'s module docs).

use nalgebra::Matrix3;
use stacker_align::transform::{warp_image_clamped, warp_image_clamped_cpu};
use stacker_core::image::PlanarImage;

/// Build a small deterministic test image with non-trivial per-channel detail.
fn make_test_image(width: usize, height: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;

            let v = ((x as f32 * 0.27).sin() * (y as f32 * 0.21).cos())
                .mul_add(0.5, 0.5)
                .clamp(0.0, 1.0);
            img.luma[i] = v;
            img.chroma_a[i] = (v - 0.5) * 0.4;
            img.chroma_b[i] = (0.5 - v) * 0.25;
        }
    }
    img
}

/// A mild rotation + translation + slight scale, well-conditioned (finite,
/// invertible) for every test image size used here.
fn sample_matrix() -> Matrix3<f32> {
    let theta: f32 = 0.05;
    let (s, c) = theta.sin_cos();
    let scale = 1.02_f32;
    Matrix3::new(c * scale, -s, 1.5, s, c * scale, -0.75, 0.0, 0.0, 1.0)
}

/// Max absolute difference across all three channels.
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

/// Dispatch/fallback: [`warp_image_clamped`] must exactly match
/// [`warp_image_clamped_cpu`] on this (GPU-less, or `gpu`-not-compiled-in)
/// host. Meaningful in every build: in a default build `warp_image_clamped`
/// is a zero-overhead wrapper, so this is a plain regression test; in a
/// `gpu` build (see the crate's "gpu flip" test procedure) it proves the
/// fallback path is exact when no adapter is available.
#[test]
fn dispatch_matches_cpu_reference() {
    // With the `gpu` feature compiled in AND a wgpu adapter actually present
    // (e.g. a software rasteriser like llvmpipe on a headless CI host), the
    // public dispatch legitimately runs the GPU kernel, whose output is
    // tolerance-equal — not bit-equal — to the CPU reference (see
    // `transform::gpu`'s module docs). Exact equality is only the correct
    // expectation when the dispatch is guaranteed to take the CPU path.
    #[cfg(feature = "gpu")]
    let exact = stacker_core::gpu::context().is_none();
    #[cfg(not(feature = "gpu"))]
    let exact = true;

    for &(width, height) in &[(33, 20), (16, 16), (64, 40), (5, 7)] {
        let img = make_test_image(width, height);
        let matrix = sample_matrix();
        let dispatched = warp_image_clamped(&img, &matrix).expect("warp_image_clamped failed");
        let reference =
            warp_image_clamped_cpu(&img, &matrix).expect("warp_image_clamped_cpu failed");
        let diff = max_abs_diff(&dispatched, &reference);
        if exact {
            assert!(
                diff == 0.0,
                "warp_image_clamped diverged from warp_image_clamped_cpu at {width}x{height}: max abs diff {diff}"
            );
        } else {
            assert!(
                diff < 1e-3,
                "GPU-dispatched warp_image_clamped exceeded the documented 1e-3 epsilon vs \
                 warp_image_clamped_cpu at {width}x{height}: max abs diff {diff}"
            );
        }
    }
}

/// GPU-vs-CPU parity, only meaningful when the `gpu` feature is compiled in.
/// Skips its real assertions (prints a note and passes) when no `wgpu`
/// adapter is available on the host running the test.
#[cfg(feature = "gpu")]
#[test]
fn gpu_result_matches_cpu_within_tolerance_including_unaligned_width() {
    use stacker_align::transform::gpu::warp_image_clamped_gpu;

    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_result_matches_cpu_within_tolerance_including_unaligned_width: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    // 33 px wide: not a multiple of 16, exercising the row-stride
    // padding/de-striding logic in `transform::gpu`.
    for &(width, height) in &[(33, 20), (16, 16), (64, 40)] {
        let img = make_test_image(width, height);
        let matrix = sample_matrix();
        let cpu = warp_image_clamped_cpu(&img, &matrix).expect("warp_image_clamped_cpu failed");
        let gpu = warp_image_clamped_gpu(&img, &matrix)
            .expect("warp_image_clamped_gpu returned an Err")
            .expect(
                "adapter reported available by context() but warp_image_clamped_gpu returned None",
            );
        let diff = max_abs_diff(&cpu, &gpu);
        assert!(
            diff < 1e-3,
            "GPU/CPU parity failed at {width}x{height}: max abs diff {diff} >= 1e-3"
        );
    }
}
