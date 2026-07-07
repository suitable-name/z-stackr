//! GPU-vs-CPU dispatch and parity tests for `relief::gpu`'s fused
//! guided-filter and Multigrid-relaxation compute shaders.
//!
//! This file is compiled unconditionally (including in default, `gpu`-off
//! builds run by `run_clippy --all-targets`), so every test guards its own
//! GPU-specific behaviour with `#[cfg(feature = "gpu")]` rather than gating
//! the whole file. See `tests/gpu_fuse_parity.rs`'s module docs for the
//! general pattern: runtime-gated parity assertions (skip-with-a-note when
//! no adapter), max-abs tolerance `< 1e-3`, and a width not divisible by 16
//! (33 px) to exercise the row-stride padding in `relief::gpu::guided_filter_gpu`
//! (`relax_gpu` uses plain storage buffers with no such requirement).
//!
//! # Why `guided_filter_gpu`, not a standalone `box_filter_gpu`
//!
//! `relief::guided::box_filter` used to try a per-call `wgpu` dispatch
//! (`relief::gpu::box_filter_gpu`), but `guided_filter` calls it six times,
//! each a *separate* two-pass GPU round trip — twelve serialized dispatches
//! per `guided_filter` call, holding the process-wide `dispatch_guard()`
//! mutex the whole time, which pegged one core and was slower than the
//! rayon-parallel CPU path it replaced. `box_filter` is now always CPU/SAT
//! (see that function's doc comment), and the entire guided-filter
//! pipeline — all six box-mean steps plus the elementwise algebra between
//! them — is GPU-accelerated as ONE fused dispatch sequence,
//! `relief::gpu::guided_filter_gpu`, tried by `guided_filter` before falling
//! back to its CPU body. `box_filter_gpu`/`box_filter_pass_gpu` are gone
//! (fully dead once nothing called them — `strata::mod`'s direct
//! `box_filter` call was always CPU-only and stays that way), so this
//! file's direct-dispatch guard now targets `guided_filter_gpu` instead.
//!
//! `box_filter` and `MultigridLayer::relax` are private/internal to
//! `stacker-algo`, so these tests exercise the GPU paths through the public
//! surfaces that call them: [`guided_filter`] (which tries
//! `guided_filter_gpu` internally) and [`MultigridSolver::solve`] (whose
//! V-cycle calls `relax` repeatedly at every level).

use stacker_algo::relief::{
    guided::{guided_filter, guided_filter_pair},
    multigrid::MultigridSolver,
};
use stacker_core::image::PlanarImage;

fn make_image(width: usize, height: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;

            let v = ((x as f32) * 0.27).sin() * ((y as f32) * 0.23).cos();
            img.luma[i] = v.mul_add(0.5, 0.5).clamp(0.0, 1.0);
        }
    }
    img
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Radii/eps used by [`gpu_guided_filter_pair_matches_cpu_within_tolerance_including_unaligned_width`],
/// matching Strata's `R_BIG`/`EPS_BIG`/`R_SMALL`/`EPS_SMALL` constants
/// (`strata::mod`) — hoisted to module scope (rather than declared inside
/// that test) purely to avoid `clippy::items_after_statements`; the values
/// and their only use site are unchanged.
#[cfg(feature = "gpu")]
const R_BIG: usize = 45;
#[cfg(feature = "gpu")]
const EPS_BIG: f32 = 0.3;
#[cfg(feature = "gpu")]
const R_SMALL: usize = 7;
#[cfg(feature = "gpu")]
const EPS_SMALL: f32 = 1e-6;

/// Brute-force `f64` reference guided filter, independent of both the CPU
/// SAT implementation and the GPU fused implementation: a direct
/// `O(radius^2)` clipped-window mean (out-of-bounds taps excluded, divisor
/// is the in-bounds tap count — exactly `box_filter`'s edge semantics) used
/// for every one of the six box-mean steps, with the same elementwise
/// algebra as `guided_filter`. Only used by the `gpu`-feature parity test
/// below (dead code in a default build otherwise).
#[cfg(feature = "gpu")]
#[allow(clippy::needless_range_loop)]
fn brute_force_guided_filter(
    guidance: &[f32],
    src: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    eps: f32,
) -> Vec<f32> {
    let r = radius.cast_signed();
    let box_mean = |data: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0_f32; width * height];
        for y in 0..height.cast_signed() {
            for x in 0..width.cast_signed() {
                let mut sum = 0.0_f64;
                let mut count = 0.0_f64;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = x + dx;
                        let sy = y + dy;
                        if sx >= 0
                            && sx < width.cast_signed()
                            && sy >= 0
                            && sy < height.cast_signed()
                        {
                            {
                                sum += f64::from(data[(sy as usize) * width + (sx as usize)]);
                            }
                            count += 1.0;
                        }
                    }
                }

                {
                    out[(y as usize) * width + (x as usize)] = (sum / count) as f32;
                }
            }
        }
        out
    };

    let mean_guidance = box_mean(guidance);
    let mean_src = box_mean(src);
    let i_p: Vec<f32> = guidance.iter().zip(src).map(|(&i, &p)| i * p).collect();
    let i_i: Vec<f32> = guidance.iter().map(|&i| i * i).collect();
    let mean_cross = box_mean(&i_p);
    let mean_sq = box_mean(&i_i);

    let mut a = vec![0.0_f32; width * height];
    let mut b = vec![0.0_f32; width * height];
    for idx in 0..width * height {
        let cov_ip = mean_guidance[idx].mul_add(-mean_src[idx], mean_cross[idx]);
        let var_i = mean_guidance[idx]
            .mul_add(-mean_guidance[idx], mean_sq[idx])
            .max(0.0);
        a[idx] = cov_ip / (var_i + eps);
        b[idx] = a[idx].mul_add(-mean_guidance[idx], mean_src[idx]);
    }

    let mean_a = box_mean(&a);
    let mean_b = box_mean(&b);

    (0..width * height)
        .map(|idx| mean_a[idx].mul_add(guidance[idx], mean_b[idx]))
        .collect()
}

/// [`guided_filter`] must be deterministic across repeated calls — the only
/// path reachable in a default (non-`gpu`) build, and also true in a `gpu`
/// build with no adapter (the GPU guided filter returns `None` and the
/// CPU/SAT path runs unmodified).
#[test]
fn guided_filter_is_deterministic() {
    let (width, height) = (33, 20);
    let img = make_image(width, height);
    let a = guided_filter(&img, &img, 6, 0.01);
    let b = guided_filter(&img, &img, 6, 0.01);
    let diff = max_abs_diff(&a.luma, &b.luma);
    assert!(
        diff < 1e-3,
        "guided_filter is not repeatable within tolerance: {diff}"
    );
}

/// [`guided_filter_pair`] must be deterministic across repeated calls, and
/// each of its two chains must match an independent [`guided_filter`] call
/// at the same radius/eps — the only path reachable in a default (non-`gpu`)
/// build, and also true in a `gpu` build with no adapter (the fused GPU pair
/// dispatch returns `None` and the CPU shared-SAT path runs unmodified).
#[test]
fn guided_filter_pair_matches_independent_calls_and_is_deterministic() {
    let (width, height) = (33, 20);
    let img = make_image(width, height);

    let (pair_first_a, pair_first_b) = guided_filter_pair(&img, &img, 45, 0.3, 7, 1e-6);
    let (pair_second_a, pair_second_b) = guided_filter_pair(&img, &img, 45, 0.3, 7, 1e-6);
    assert!(
        max_abs_diff(&pair_first_a.luma, &pair_second_a.luma) < 1e-3,
        "guided_filter_pair's radius_a chain is not repeatable within tolerance"
    );
    assert!(
        max_abs_diff(&pair_first_b.luma, &pair_second_b.luma) < 1e-3,
        "guided_filter_pair's radius_b chain is not repeatable within tolerance"
    );

    let independent_a = guided_filter(&img, &img, 45, 0.3);
    let independent_b = guided_filter(&img, &img, 7, 1e-6);
    assert!(
        max_abs_diff(&pair_first_a.luma, &independent_a.luma) < 1e-3,
        "guided_filter_pair's radius_a chain diverged from an independent guided_filter call"
    );
    assert!(
        max_abs_diff(&pair_first_b.luma, &independent_b.luma) < 1e-3,
        "guided_filter_pair's radius_b chain diverged from an independent guided_filter call"
    );
}

/// [`MultigridSolver::solve`] must be deterministic across repeated solves
/// of the same problem — same rationale as the guided-filter check above.
#[test]
fn multigrid_solve_is_deterministic() {
    let (width, height) = (33, 20);
    let mut target = vec![0.0_f32; width * height];
    let mut weight = vec![0.0_f32; width * height];
    for y in 0..height {
        target[y * width] = 0.0;
        weight[y * width] = 1.0;
        target[y * width + (width - 1)] = 1.0;
        weight[y * width + (width - 1)] = 1.0;
    }

    let mut solver_a = MultigridSolver::new(width, height, &target, &weight);
    solver_a.solve();
    let a = solver_a.get_solution();

    let mut solver_b = MultigridSolver::new(width, height, &target, &weight);
    solver_b.solve();
    let b = solver_b.get_solution();

    let diff = max_abs_diff(&a, &b);
    assert!(
        diff < 1e-3,
        "MultigridSolver::solve is not repeatable within tolerance: {diff}"
    );
}

/// GPU-vs-CPU parity for the fused guided filter, via `guided_filter`'s
/// public surface. Only runs its real assertion when an adapter is present;
/// otherwise prints a skip note and passes.
///
/// Covers both a tight radius (6, Relief's typical `smooth_radius`) and a
/// Strata-like wide radius (45, Strata's `R_BIG`), at `eps` values `0.01`
/// and `0.3` (Strata's `EPS_BIG`) — the four combinations that matter for
/// the `f32`-on-GPU-vs-`f64`-on-CPU box-mean tolerance argument in
/// `relief::gpu::guided_filter_gpu`'s doc comment.
#[cfg(feature = "gpu")]
#[test]
fn gpu_guided_filter_matches_cpu_within_tolerance_including_unaligned_width() {
    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_guided_filter_matches_cpu_within_tolerance_including_unaligned_width: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    for &(width, height) in &[(33, 20), (16, 16), (64, 40)] {
        let img = make_image(width, height);

        for &(radius, eps) in &[
            (6_usize, 0.01_f32),
            (45_usize, 0.01_f32),
            (45_usize, 0.3_f32),
        ] {
            // Direct dispatch check FIRST: `guided_filter_gpu` must return
            // `Some` on a host whose adapter `context()` just reported
            // available — a `None` here means the dispatch itself failed
            // (e.g. a WGSL compile error) and the `guided_filter` comparison
            // below would silently degrade into CPU-vs-CPU, passing
            // vacuously. That is exactly how the `fn get` reserved-keyword
            // bug in `box_filter.wgsl` originally slipped through this
            // test (see the old `box_filter_gpu` version of this guard).
            let gpu_direct = stacker_algo::relief::gpu::guided_filter_gpu(&img, &img, radius, eps)
                .expect(
                    "adapter reported available by context() but guided_filter_gpu returned None",
                );

            let reference =
                brute_force_guided_filter(&img.luma, &img.luma, width, height, radius, eps);
            let direct_diff = max_abs_diff(&gpu_direct, &reference);
            assert!(
                direct_diff < 1e-3,
                "GPU guided_filter_gpu vs brute-force reference failed at {width}x{height}, \
                 r={radius}, eps={eps}: max abs diff {direct_diff} >= 1e-3"
            );

            // GPU path (engaged automatically since an adapter is present).
            let gpu_result = guided_filter(&img, &img, radius, eps);

            // CPU-only reference: disable the runtime switch for the
            // duration of this call so `guided_filter` takes its CPU body
            // (six `box_filter` calls) unconditionally.
            stacker_core::gpu::set_enabled(false);
            let cpu_result = guided_filter(&img, &img, radius, eps);
            stacker_core::gpu::set_enabled(true);

            let diff = max_abs_diff(&gpu_result.luma, &cpu_result.luma);
            assert!(
                diff < 1e-3,
                "GPU/CPU guided_filter parity failed at {width}x{height}, r={radius}, eps={eps} \
                 (via guided_filter): max abs diff {diff} >= 1e-3"
            );
        }
    }
}

/// GPU-vs-CPU parity for [`guided_filter_pair`] (the shared-work dual-radius
/// call Strata's Pass 2 uses — see that function's doc comment), via its
/// public surface. Only runs its real assertion when an adapter is present;
/// otherwise prints a skip note and passes.
///
/// Covers both radii Strata actually pairs (`R_BIG = 45, EPS_BIG = 0.3` and
/// `R_SMALL = 7, EPS_SMALL = 1e-6`, matching `strata::mod`'s constants) at a
/// 33px width (not a multiple of 16, exercising the row-stride padding in
/// `relief::gpu::guided_filter_pair_gpu`), plus a direct-`Some` vacuous-
/// fallback guard mirroring `gpu_guided_filter_matches_cpu_within_tolerance_including_unaligned_width`'s
/// identical guard above (a `None` from the direct dispatch call would mean
/// the fused pair dispatch itself failed, and the `guided_filter_pair`
/// comparison below would otherwise silently degrade into CPU-vs-CPU,
/// passing vacuously).
#[cfg(feature = "gpu")]
#[test]
fn gpu_guided_filter_pair_matches_cpu_within_tolerance_including_unaligned_width() {
    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_guided_filter_pair_matches_cpu_within_tolerance_including_unaligned_width: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    for &(width, height) in &[(33, 20), (16, 16), (64, 40)] {
        let img = make_image(width, height);

        // Direct dispatch check FIRST — see this test's doc comment.
        let (gpu_direct_a, gpu_direct_b) = stacker_algo::relief::gpu::guided_filter_pair_gpu(
            &img, &img, R_BIG, EPS_BIG, R_SMALL, EPS_SMALL,
        )
        .expect("adapter reported available by context() but guided_filter_pair_gpu returned None");

        let reference_a =
            brute_force_guided_filter(&img.luma, &img.luma, width, height, R_BIG, EPS_BIG);
        let reference_b =
            brute_force_guided_filter(&img.luma, &img.luma, width, height, R_SMALL, EPS_SMALL);
        let direct_diff_a = max_abs_diff(&gpu_direct_a, &reference_a);
        let direct_diff_b = max_abs_diff(&gpu_direct_b, &reference_b);
        assert!(
            direct_diff_a < 1e-3,
            "GPU guided_filter_pair_gpu (radius_a) vs brute-force reference failed at \
             {width}x{height}: max abs diff {direct_diff_a} >= 1e-3"
        );
        assert!(
            direct_diff_b < 1e-3,
            "GPU guided_filter_pair_gpu (radius_b) vs brute-force reference failed at \
             {width}x{height}: max abs diff {direct_diff_b} >= 1e-3"
        );

        // Via the public `guided_filter_pair` surface (engaged automatically
        // since an adapter is present).
        let (gpu_a, gpu_b) = guided_filter_pair(&img, &img, R_BIG, EPS_BIG, R_SMALL, EPS_SMALL);

        // CPU-only reference: disable the runtime switch so
        // `guided_filter_pair` takes its CPU (shared-SAT) body
        // unconditionally.
        stacker_core::gpu::set_enabled(false);
        let (cpu_a, cpu_b) = guided_filter_pair(&img, &img, R_BIG, EPS_BIG, R_SMALL, EPS_SMALL);
        stacker_core::gpu::set_enabled(true);

        let diff_a = max_abs_diff(&gpu_a.luma, &cpu_a.luma);
        let diff_b = max_abs_diff(&gpu_b.luma, &cpu_b.luma);
        assert!(
            diff_a < 1e-3,
            "GPU/CPU guided_filter_pair parity failed at {width}x{height} (radius_a, via \
             guided_filter_pair): max abs diff {diff_a} >= 1e-3"
        );
        assert!(
            diff_b < 1e-3,
            "GPU/CPU guided_filter_pair parity failed at {width}x{height} (radius_b, via \
             guided_filter_pair): max abs diff {diff_b} >= 1e-3"
        );

        // `guided_filter_pair`'s two chains must also individually agree
        // (within tolerance) with plain independent `guided_filter` calls
        // at the same radius/eps — proves the shared-SAT/shared-upload
        // optimisation didn't change the actual result, only how it's
        // computed.
        let independent_a = guided_filter(&img, &img, R_BIG, EPS_BIG);
        let independent_b = guided_filter(&img, &img, R_SMALL, EPS_SMALL);
        assert!(
            max_abs_diff(&gpu_a.luma, &independent_a.luma) < 1e-3,
            "guided_filter_pair's radius_a chain diverged from an independent guided_filter call \
             at {width}x{height}"
        );
        assert!(
            max_abs_diff(&gpu_b.luma, &independent_b.luma) < 1e-3,
            "guided_filter_pair's radius_b chain diverged from an independent guided_filter call \
             at {width}x{height}"
        );
    }
}

/// GPU-vs-CPU parity for the Multigrid relaxation sweep, via
/// `MultigridSolver::solve`'s public surface (its V-cycle calls `relax`
/// repeatedly at every level). Only runs its real assertion when an adapter
/// is present; otherwise prints a skip note and passes.
#[cfg(feature = "gpu")]
#[test]
fn gpu_relax_matches_cpu_within_tolerance_including_unaligned_width() {
    if stacker_core::gpu::context().is_none() {
        eprintln!(
            "gpu_relax_matches_cpu_within_tolerance_including_unaligned_width: \
             skipped — no wgpu adapter available on this host"
        );
        return;
    }

    for &(width, height) in &[(33, 20), (16, 16), (64, 40)] {
        let mut target = vec![0.0_f32; width * height];
        let mut weight = vec![0.0_f32; width * height];
        for y in 0..height {
            target[y * width] = 0.0;
            weight[y * width] = 1.0;
            target[y * width + (width - 1)] = 1.0;
            weight[y * width + (width - 1)] = 1.0;
        }

        // Direct dispatch check FIRST: `relax_gpu` must return `Some` on a
        // host whose adapter `context()` just reported available — a `None`
        // means the dispatch itself failed (e.g. a WGSL compile error) and
        // the solver-vs-solver comparison below would silently degrade into
        // CPU-vs-CPU, passing vacuously (see the identical guard in the
        // guided-filter test above and the `fn get` bug it exists to catch).
        let zeros = vec![0.0_f32; width * height];
        let direct = stacker_algo::relief::gpu::relax_gpu(&zeros, &target, &weight, width, height)
            .expect("adapter reported available by context() but relax_gpu returned None");
        assert_eq!(direct.len(), width * height);
        assert!(
            direct.iter().all(|v| v.is_finite()),
            "relax_gpu produced non-finite values at {width}x{height}"
        );

        let mut gpu_solver = MultigridSolver::new(width, height, &target, &weight);
        gpu_solver.solve();
        let gpu_result = gpu_solver.get_solution();

        stacker_core::gpu::set_enabled(false);
        let mut cpu_solver = MultigridSolver::new(width, height, &target, &weight);
        cpu_solver.solve();
        let cpu_result = cpu_solver.get_solution();
        stacker_core::gpu::set_enabled(true);

        let diff = max_abs_diff(&gpu_result, &cpu_result);
        assert!(
            diff < 1e-3,
            "GPU/CPU relax parity failed at {width}x{height} (via MultigridSolver::solve): \
             max abs diff {diff} >= 1e-3"
        );
    }
}

/// `stacker_core::gpu::set_enabled(false)` must force both `guided_filter`
/// and `relax` onto their CPU paths, producing output that is EXACTLY
/// (bit-for-bit) equal across two separate disabled runs.
#[cfg(feature = "gpu")]
#[test]
fn set_enabled_false_forces_exact_cpu_path() {
    let (width, height) = (33, 20);
    let img = make_image(width, height);

    stacker_core::gpu::set_enabled(false);
    let a = guided_filter(&img, &img, 6, 0.01);
    let b = guided_filter(&img, &img, 6, 0.01);
    stacker_core::gpu::set_enabled(true);

    assert_eq!(
        a.luma, b.luma,
        "set_enabled(false) must make guided_filter bit-for-bit repeatable (pure CPU path)"
    );
}
