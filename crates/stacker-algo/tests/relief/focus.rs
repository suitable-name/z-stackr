#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops
)]

use rayon::prelude::*;
use stacker_algo::relief::focus::*;
use stacker_core::image::PlanarImage;

fn make_planar(width: usize, height: usize, luma: Vec<f32>) -> PlanarImage<f32> {
    let len = width * height;
    PlanarImage {
        width,
        height,
        luma,
        chroma_a: vec![0.0; len],
        chroma_b: vec![0.0; len],
    }
}

// ── Reference implementation (naive O(N·r²)) ────────────────────────────
//
// Kept here (cfg(test) only) so that `test_sat_equivalence` can compare
// the SAT production path against it pixel-by-pixel.
//
// This is the verbatim original loop; do NOT change it.

fn plane_fit_variance_naive(
    luma: &[f32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
) -> f32 {
    let r = radius as isize;
    let w = width as isize;
    let h = height as isize;
    let icx = cx as isize;
    let icy = cy as isize;

    let mut sum_dy: f64 = 0.0;
    let mut count_dy: i64 = 0;
    let mut sum_dx: f64 = 0.0;
    let mut count_dx: i64 = 0;
    let mut var_dy: f64 = 0.0;
    let mut var_dx: f64 = 0.0;
    let mut sum_pixel: f64 = 0.0;
    let mut count_pixel: i64 = 0;

    for dy in -r..=r {
        let ny = icy + dy;
        if ny >= 0 && ny < h {
            sum_dy += dy as f64;
            count_dy += 1;
        }
    }
    for dx in -r..=r {
        let nx = icx + dx;
        if nx >= 0 && nx < w {
            sum_dx += dx as f64;
            count_dx += 1;
        }
    }
    if count_dy == 0 || count_dx == 0 {
        return 0.0;
    }
    let mean_dy = (sum_dy / count_dy as f64) as f32;
    let mean_dx = (sum_dx / count_dx as f64) as f32;

    for dy in -r..=r {
        let ny = icy + dy;
        if ny < 0 || ny >= h {
            continue;
        }
        for dx in -r..=r {
            let nx = icx + dx;
            if nx < 0 || nx >= w {
                continue;
            }
            let fdy = dy as f32 - mean_dy;
            let fdx = dx as f32 - mean_dx;
            var_dy += f64::from(fdy * fdy);
            var_dx += f64::from(fdx * fdx);
            let pv = luma[ny as usize * width + nx as usize];
            sum_pixel += f64::from(pv);
            count_pixel += 1;
        }
    }

    if count_pixel < 2 || var_dy == 0.0 || var_dx == 0.0 {
        return 0.0;
    }
    let mean_pixel = (sum_pixel / count_pixel as f64) as f32;

    let mut cov_dy: f64 = 0.0;
    let mut cov_dx: f64 = 0.0;

    for dy in -r..=r {
        let ny = icy + dy;
        if ny < 0 || ny >= h {
            continue;
        }
        for dx in -r..=r {
            let nx = icx + dx;
            if nx < 0 || nx >= w {
                continue;
            }
            let pixel_diff = luma[ny as usize * width + nx as usize] - mean_pixel;
            let fdy = dy as f32 - mean_dy;
            let fdx = dx as f32 - mean_dx;
            cov_dy += f64::from(fdy * pixel_diff);
            cov_dx += f64::from(fdx * pixel_diff);
        }
    }

    let slope_y = (cov_dy / var_dy) as f32;
    let slope_x = (cov_dx / var_dx) as f32;

    let mut sum_res: f64 = 0.0;
    let mut sum_sq_res: f64 = 0.0;

    for dy in -r..=r {
        let ny = icy + dy;
        if ny < 0 || ny >= h {
            continue;
        }
        for dx in -r..=r {
            let nx = icx + dx;
            if nx < 0 || nx >= w {
                continue;
            }
            let pv = luma[ny as usize * width + nx as usize];
            let fdy = dy as f32 - mean_dy;
            let fdx = dx as f32 - mean_dx;
            let residual = pv - slope_y * fdy - slope_x * fdx;
            sum_res += f64::from(residual);
            sum_sq_res += f64::from(residual * residual);
        }
    }

    let n = count_pixel as f64;
    let variance = (sum_sq_res - sum_res * sum_res / n) / (n - 1.0);
    variance.max(0.0) as f32
}

// ── Equivalence test ────────────────────────────────────────────────────

/// Verify that the SAT-based `compute_sum_modified_laplacian` agrees with
/// the naive per-pixel `plane_fit_variance_naive` on every pixel, across
/// several image sizes, aspect ratios, and radii.
///
/// Tolerance: absolute < 1e-2 OR relative < 1e-3.  These are generous
/// enough to absorb the accumulated f32 ↔ f64 round-trips in the naive
/// path (which mixes f32 arithmetic with f64 accumulators, incurring
/// intermediate f32 rounding on slopes/residuals) vs the SAT path (which
/// accumulates entirely in f64 and converts once at the very end).
#[test]
fn test_sat_equivalence() {
    let cases: &[(usize, usize, usize)] = &[
        (16, 16, 1),
        (16, 16, 3),
        (16, 16, 8),
        (30, 20, 2),
        (20, 30, 2),
        (5, 5, 1),
        (5, 5, 3),
        (7, 3, 2),
        (3, 7, 2),
        (64, 48, 5),
        (64, 48, 8),
    ];

    for &(width, height, radius) in cases {
        // Sinusoidal texture with real contrast so variance is non-trivially
        // non-zero in most of the image.
        let luma: Vec<f32> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let v = (x as f32 * 0.4 + y as f32 * 0.3).sin() * 0.5
                        + (x as f32 * 0.7 - y as f32 * 0.5).cos() * 0.3
                        + 0.5;
                    v.clamp(0.0, 1.0)
                })
            })
            .collect();

        let img = make_planar(width, height, luma.clone());
        let sat_out = compute_sum_modified_laplacian(&img, radius);

        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;

        for y in 0..height {
            for x in 0..width {
                let naive = plane_fit_variance_naive(&luma, width, height, x, y, radius);
                let sat = sat_out.luma[y * width + x];

                let abs_err = (f64::from(naive) - f64::from(sat)).abs();
                let rel_err = if naive.abs() > 1e-9_f32 {
                    abs_err / f64::from(naive.abs())
                } else {
                    0.0
                };

                max_abs = max_abs.max(abs_err);
                max_rel = max_rel.max(rel_err);

                assert!(
                    abs_err < 1e-2 || rel_err < 1e-3,
                    "SAT vs naive mismatch at ({x},{y}) size={width}×{height} r={radius}: \
                     naive={naive:.6e} sat={sat:.6e} abs_err={abs_err:.2e} rel_err={rel_err:.2e}"
                );
            }
        }

        println!("  size={width}×{height} r={radius}: max_abs={max_abs:.2e} max_rel={max_rel:.2e}");
    }
}

// ── Correctness tests (retained from original) ──────────────────────────

/// A perfectly flat image (constant value) has zero residual variance
/// everywhere regardless of the window radius.
#[test]
fn test_flat_image_zero_variance() {
    let width = 15;
    let height = 15;
    let img = make_planar(width, height, vec![0.5_f32; width * height]);
    for radius in [1_usize, 2, 3, 5] {
        let out = compute_sum_modified_laplacian(&img, radius);
        for (i, &v) in out.luma.iter().enumerate() {
            assert!(
                v < 1e-6,
                "flat image: non-zero variance {v} at pixel {i} (radius {radius})"
            );
        }
    }
}

/// A linearly-tilted (ramp) image has zero residual variance everywhere —
/// the plane fit removes the tilt exactly.
#[test]
fn test_linear_ramp_zero_variance() {
    let width = 20;
    let height = 20;
    // luma(x,y) = 0.01*x + 0.02*y  — a pure linear plane
    let luma: Vec<f32> = (0..height)
        .flat_map(|y| (0..width).map(move |x| 0.01 * x as f32 + 0.02 * y as f32))
        .collect();
    let img = make_planar(width, height, luma);
    for radius in [1_usize, 2, 3] {
        let out = compute_sum_modified_laplacian(&img, radius);
        // Interior pixels should have near-zero variance; allow a small
        // tolerance for f32 arithmetic at the boundaries.
        for y in radius..height.saturating_sub(radius) {
            for x in radius..width.saturating_sub(radius) {
                let v = out.luma[y * width + x];
                assert!(
                    v < 1e-6,
                    "ramp image: non-zero variance {v} at ({x},{y}) (radius {radius})"
                );
            }
        }
    }
}

/// A step-edge image must produce large variance near the edge and
/// near-zero variance in the flat interior regions.
#[test]
fn test_step_edge_large_variance() {
    let width = 30;
    let height = 30;
    // Left half 0.0, right half 1.0 — hard vertical step at x = 15.
    let luma: Vec<f32> = (0..height)
        .flat_map(|_y| (0..width).map(move |x| if x >= 15 { 1.0_f32 } else { 0.0_f32 }))
        .collect();
    let img = make_planar(width, height, luma);
    let radius = 3;
    let out = compute_sum_modified_laplacian(&img, radius);

    // Pixels centred on the edge column should have large variance.
    let edge_val = out.luma[15 * width + 15];
    assert!(
        edge_val > 0.01,
        "step edge: expected large variance at edge, got {edge_val}"
    );

    // Pixels well inside the flat region should have near-zero variance.
    let flat_val = out.luma[15 * width + 3];
    assert!(
        flat_val < 1e-6,
        "step edge: expected near-zero variance in flat region, got {flat_val}"
    );
}

/// A sinusoidal texture must produce large variance (the plane fit cannot
/// remove oscillating detail).
#[test]
fn test_texture_large_variance() {
    let width = 20;
    let height = 20;
    let luma: Vec<f32> = (0..height)
        .flat_map(|_y| (0..width).map(move |x| (x as f32 * std::f32::consts::PI * 0.5).sin().abs()))
        .collect();
    let img = make_planar(width, height, luma);
    let out = compute_sum_modified_laplacian(&img, 2);

    // Interior texture pixels should have non-trivial variance.
    let interior_max = out
        .luma
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let x = i % width;
            let y = i / width;
            x >= 3 && x < width - 3 && y >= 3 && y < height - 3
        })
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        interior_max > 0.01,
        "texture: expected large variance, got max={interior_max}"
    );
}

// ── Timing harness ───────────────────────────────────────────────────────

/// Timing harness: naive O(N·r²) vs SAT O(N) on a 512×512 image at radius 8.
///
/// Run with:
/// ```text
/// cargo test -p stacker-algo `relief_sat_speedup` -- --nocapture --ignored
/// ```
#[test]
#[ignore = "slow timing harness: run with --ignored --nocapture to see speedup numbers"]

fn relief_sat_speedup() {
    use std::time::Instant;

    const W: usize = 512;
    const H: usize = 512;
    const RADIUS: usize = 8;
    const REPS: u32 = 3;

    // Build a realistic test image with real contrast.
    let luma: Vec<f32> = (0..H)
        .flat_map(|y| {
            (0..W).map(move |x| {
                let v = (x as f32 * 0.05 + 0.7).sin() * (y as f32 * 0.04 + 1.3).cos() * 0.5 + 0.5;
                v.clamp(0.0, 1.0)
            })
        })
        .collect();

    let img = make_planar(W, H, luma.clone());

    // ── Naive timing ──────────────────────────────────────────────────────
    let mut naive_total = std::time::Duration::ZERO;
    for _ in 0..REPS {
        let t = Instant::now();
        let mut out = vec![0.0_f32; W * H];
        out.par_chunks_mut(W).enumerate().for_each(|(cy, row)| {
            for (cx, v) in row.iter_mut().enumerate() {
                *v = plane_fit_variance_naive(&luma, W, H, cx, cy, RADIUS);
            }
        });
        naive_total += t.elapsed();
        let _ = out;
    }
    let naive_ms = naive_total.as_secs_f64() * 1000.0 / f64::from(REPS);

    // ── SAT timing ────────────────────────────────────────────────────────
    let mut sat_total = std::time::Duration::ZERO;
    for _ in 0..REPS {
        let t = Instant::now();
        let out = compute_sum_modified_laplacian(&img, RADIUS);
        sat_total += t.elapsed();
        let _ = out;
    }
    let sat_ms = sat_total.as_secs_f64() * 1000.0 / f64::from(REPS);

    let speedup = naive_ms / sat_ms;
    println!(
        "\n[relief timing] {W}×{H} image, radius={RADIUS}, {REPS} reps\n\
         naive   : {naive_ms:.1} ms/iter\n\
         SAT     : {sat_ms:.1} ms/iter\n\
         speedup : {speedup:.2}×\n"
    );

    // Sanity: SAT must not be more than 2× slower (regression guard only).
    assert!(
        sat_ms < naive_ms * 2.0,
        "SAT path ({sat_ms:.1} ms) is more than 2× slower than naive ({naive_ms:.1} ms) — likely a regression"
    );
}
