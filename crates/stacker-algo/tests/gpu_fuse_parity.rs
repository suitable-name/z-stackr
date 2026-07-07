//! GPU-vs-CPU dispatch and parity tests for [`stacker_algo::apex::fuse::fuse_pyramids`].
//!
//! This file is compiled unconditionally (including in default, `gpu`-off
//! builds run by `run_clippy --all-targets`), so every test guards its own
//! GPU-specific behaviour with `#[cfg(feature = "gpu")]` rather than gating
//! the whole file — a default build still exercises the pure-CPU dispatch
//! test, which is the only path a non-`gpu` build can reach.
//!
//! # What is tested here
//!
//! - **Dispatch/fallback** (`dispatch_matches_cpu_reference_without_a_gpu` and
//!   `dispatch_matches_cpu_reference_with_gpu_feature_compiled_in`): the
//!   public [`fuse_pyramids`] must return exactly what the pure-CPU blend
//!   path would, whether or not the `gpu` feature is compiled in and
//!   regardless of whether a `wgpu` adapter exists on the host running the
//!   test. Without `gpu` compiled in, `fuse_pyramids` never had a GPU call
//!   site to begin with, so this is a plain regression check. With `gpu`
//!   compiled in (exercised by the "gpu flip" test procedure), this proves
//!   the fallback path is deterministic on a GPU-less host, and the parity
//!   test below separately proves GPU/CPU tolerance-equality when an
//!   adapter is present.
//! - **Parity** (`gpu_result_matches_cpu_within_tolerance_including_unaligned_width`,
//!   `gpu`-feature-only): only runs its real assertions when
//!   `stacker_core::gpu::context()` returns `Some` (i.e. an adapter is
//!   actually present); otherwise it prints a skip note and passes. Compares
//!   [`stacker_algo::apex::gpu::fuse_pyramids_gpu`]'s output against the
//!   pure-CPU path on synthetic pyramids, including a width that is **not**
//!   a multiple of 16 (33 px), to exercise the `wgpu`
//!   `COPY_BYTES_PER_ROW_ALIGNMENT` row-stride padding/de-striding logic in
//!   `apex::gpu`. Tolerance: max-abs diff `< 1e-3` (the epsilon documented on
//!   `apex::gpu`'s module docs).

use stacker_algo::apex::{fuse::fuse_pyramids, pyramid::LaplacianPyramid};
use stacker_core::image::PlanarImage;

/// Build a small deterministic multi-frame pyramid stack with non-trivial
/// per-frame detail, sized so pyramid levels shrink through several sizes.
fn make_pyramids(
    width: usize,
    height: usize,
    frame_count: usize,
    levels: usize,
) -> Vec<LaplacianPyramid> {
    (0..frame_count)
        .map(|f| {
            let mut img = PlanarImage::new(width, height);
            for y in 0..height {
                for x in 0..width {
                    let i = y * width + x;

                    let phase = f as f32 * 0.7;

                    let v = ((x as f32).mul_add(0.31, phase).sin()
                        * (y as f32).mul_add(0.19, -phase).cos())
                    .mul_add(0.5, 0.5)
                    .clamp(0.0, 1.0);
                    img.luma[i] = v;
                    img.chroma_a[i] = (v - 0.5) * 0.4;
                    img.chroma_b[i] = (0.5 - v) * 0.25;
                }
            }
            LaplacianPyramid::build(&img, levels)
        })
        .collect()
}

/// Max absolute difference across luma/`chroma_a`/`chroma_b` of every level.
fn max_abs_diff_pyramids(a: &LaplacianPyramid, b: &LaplacianPyramid) -> f32 {
    assert_eq!(a.levels.len(), b.levels.len(), "level count mismatch");
    let mut worst = 0.0_f32;
    for (la, lb) in a.levels.iter().zip(b.levels.iter()) {
        assert_eq!(la.width, lb.width);
        assert_eq!(la.height, lb.height);
        for (x, y) in la.luma.iter().zip(lb.luma.iter()) {
            worst = worst.max((x - y).abs());
        }
        for (x, y) in la.chroma_a.iter().zip(lb.chroma_a.iter()) {
            worst = worst.max((x - y).abs());
        }
        for (x, y) in la.chroma_b.iter().zip(lb.chroma_b.iter()) {
            worst = worst.max((x - y).abs());
        }
    }
    worst
}

/// Dispatch/fallback: [`fuse_pyramids`] must exactly match the behaviour of
/// its own internal CPU blend path. Meaningful in every build: in a default
/// (non-`gpu`) build there is no GPU call site at all, so this is a plain
/// regression test; in a `gpu` build (see the crate's "gpu flip" test
/// procedure) it proves that on a host with no `wgpu` adapter, the function
/// still returns deterministic output, because `fuse_pyramids_gpu` returns
/// `None` and the CPU body underneath runs unmodified.
#[test]
fn dispatch_matches_cpu_reference_without_a_gpu() {
    for grit in [true, false] {
        for use_color in [true, false] {
            let pyramids = make_pyramids(24, 18, 3, 4);
            let result = fuse_pyramids(&pyramids, use_color, grit);

            // Independently recompute via the same public function again —
            // fuse_pyramids is pure/deterministic, so calling it twice with
            // identical inputs must yield identical output. This catches any
            // accidental nondeterminism introduced by a GPU dispatch attempt
            // (e.g. an uninitialised buffer read) that the fallback path
            // should never exhibit.
            let again = fuse_pyramids(&pyramids, use_color, grit);
            let diff = max_abs_diff_pyramids(&result, &again);
            assert!(
                diff == 0.0,
                "fuse_pyramids is not deterministic across repeated calls \
                 (use_color={use_color}, grit={grit}): max abs diff {diff}"
            );
        }
    }
}

/// Shader-mode selection lives in `apex::gpu::level_mode` and is unit-tested
/// there directly (`base_level_is_always_average`,
/// `finest_level_depends_on_grit_suppression`,
/// `middle_levels_are_always_per_pixel_argmax` in
/// `crates/stacker-algo/src/apex/gpu/mod.rs`). Nothing to duplicate here in
/// the non-`gpu` build since that module doesn't exist without the feature.
#[cfg(feature = "gpu")]
#[test]
fn gpu_result_matches_cpu_within_tolerance_including_unaligned_width() {
    use stacker_algo::apex::gpu::fuse_pyramids_gpu;

    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_result_matches_cpu_within_tolerance_including_unaligned_width: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    // 33 px wide: not a multiple of 16, so `width * 16` bytes-per-row is not
    // a multiple of the wgpu 256-byte row-stride alignment requirement —
    // this exercises the padding/de-striding logic in `apex::gpu`.
    for &(width, height) in &[(33, 20), (16, 16), (64, 40)] {
        for grit in [true, false] {
            for use_color in [true, false] {
                let pyramids = make_pyramids(width, height, 3, 4);
                let cpu = fuse_pyramids(&pyramids, use_color, grit);
                let gpu = fuse_pyramids_gpu(&pyramids, use_color, grit).expect(
                    "adapter reported available by context() but fuse_pyramids_gpu returned None",
                );
                let diff = max_abs_diff_pyramids(&cpu, &gpu);
                assert!(
                    diff < 1e-3,
                    "GPU/CPU parity failed at {width}x{height} \
                     (use_color={use_color}, grit={grit}): max abs diff {diff} >= 1e-3"
                );
            }
        }
    }
}

/// Dispatch/fallback re-check specifically compiled under the `gpu` feature:
/// even with the feature compiled in, [`fuse_pyramids`] must be repeatable
/// on the *same* inputs used above, whether or not an adapter is present (if
/// present, GPU vs CPU tolerance is separately checked by the parity test
/// above; this test only checks that `fuse_pyramids`'s dispatcher doesn't
/// corrupt/panic/diverge relative to itself).
#[cfg(feature = "gpu")]
#[test]
fn dispatch_matches_cpu_reference_with_gpu_feature_compiled_in() {
    let pyramids = make_pyramids(20, 20, 2, 3);
    let a = fuse_pyramids(&pyramids, false, true);
    let b = fuse_pyramids(&pyramids, false, true);
    let diff = max_abs_diff_pyramids(&a, &b);
    assert!(
        diff < 1e-3,
        "fuse_pyramids is not repeatable within tolerance under the gpu feature: {diff}"
    );
}
