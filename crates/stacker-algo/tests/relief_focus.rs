#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test images are at most a few hundred pixels wide; the naive reference mirrors the original signed-offset loop"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "the references spell the formulas out as plain multiply-add"
)]

use rayon::prelude::*;
use stacker_algo::relief::focus::*;
use stacker_core::image::PlanarImage;

fn make_planar(width: usize, height: usize, luma: Vec<f32>) -> PlanarImage<f32> {
    let len = width * height;
    PlanarImage::from_planes(width, height, luma, vec![0.0; len], vec![0.0; len])
        .expect("the luma plane holds width * height samples")
}

// ── Reference implementation (naive O(N·r²)) ────────────────────────────
//
// The verbatim original loop; do NOT change it. It is the baseline of the
// (ignored) timing harness at the end of this file. It mixes `f32` and `f64`
// and is only an approximation of the exact measure, so the equivalence test
// uses `plane_fit_variance_f64` below instead.
#[expect(
    clippy::similar_names,
    reason = "x/y and dx/dy pairs are intentionally symmetric"
)]
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

/// Independent exact reference for the measure: the least-squares plane
/// `v ≈ a + b·x + c·y` of the clamped window, found by solving the 2×2 normal
/// equations of the mean-centred problem explicitly, and `Σr² / (n − 1)` of its
/// residuals. Everything is `f64`; there are no summed-area tables, no
/// closed-form geometry sums and no `f32` intermediates, so it shares no
/// arithmetic with the production path (only the documented `n − 1` divisor).
/// Windows with fewer than two rows or columns have undefined slopes and score 0.
fn plane_fit_variance_f64(
    luma: &[f32],
    width: usize,
    height: usize,
    (cx, cy): (usize, usize),
    radius: usize,
) -> f64 {
    let x0 = cx.saturating_sub(radius);
    let x1 = (cx + radius).min(width - 1);
    let y0 = cy.saturating_sub(radius);
    let y1 = (cy + radius).min(height - 1);
    let (nx, ny) = (x1 - x0 + 1, y1 - y0 + 1);
    if nx < 2 || ny < 2 {
        return 0.0;
    }
    let n = (nx * ny) as f64;
    let window = || {
        (y0..=y1).flat_map(move |y| {
            (x0..=x1).map(move |x| (x as f64, y as f64, f64::from(luma[y * width + x])))
        })
    };
    let (sum_x, sum_y, sum_v) =
        window().fold((0.0, 0.0, 0.0), |a, (x, y, v)| (a.0 + x, a.1 + y, a.2 + v));
    let (mean_x, mean_y, mean_v) = (sum_x / n, sum_y / n, sum_v / n);
    let (mut sxx, mut syy, mut sxy, mut sxv, mut syv, mut svv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y, v) in window() {
        let (dx, dy, dv) = (x - mean_x, y - mean_y, v - mean_v);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        sxv += dx * dv;
        syv += dy * dv;
        svv += dv * dv;
    }
    let det = sxy.mul_add(-sxy, sxx * syy);
    let slope_x = (sxv * syy - syv * sxy) / det;
    let slope_y = (syv * sxx - sxv * sxy) / det;
    let sum_sq_residual = svv - slope_x * sxv - slope_y * syv;
    sum_sq_residual.max(0.0) / (n - 1.0)
}

/// Verify that the SAT-based `compute_plane_fit_variance` agrees with the
/// exact `f64` plane-fit reference on every pixel, across several image sizes,
/// aspect ratios and radii.
///
/// The production map is an `f32` rounding of an `f64` computation, so it may
/// differ from the exact value by half an `f32` ulp (6e-8 relative) plus an
/// `f64` rounding error many orders of magnitude smaller. The bound is
/// therefore `2e-7 · reference + 1e-9`: tight enough that a map which is zero,
/// off by a window, or computed with a wrong divisor fails wherever the
/// reference exceeds about 1e-9, instead of passing below an absolute 1e-2.
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
        let sat_out = compute_plane_fit_variance(&img, radius);

        let mut max_abs = 0.0_f64;
        let mut max_ref = 0.0_f64;

        for y in 0..height {
            for x in 0..width {
                let exact = plane_fit_variance_f64(&luma, width, height, (x, y), radius);
                let sat = f64::from(sat_out.luma[y * width + x]);

                let abs_err = (exact - sat).abs();
                max_abs = max_abs.max(abs_err);
                max_ref = max_ref.max(exact);

                assert!(
                    abs_err <= 2e-7 * exact + 1e-9,
                    "SAT vs exact mismatch at ({x},{y}) size={width}×{height} r={radius}: \
                     exact={exact:.6e} sat={sat:.6e} abs_err={abs_err:.2e}"
                );
            }
        }

        // The texture must actually score, so the comparison is not 0 == 0.
        assert!(
            max_ref > 1e-6,
            "size={width}×{height} r={radius}: the reference never exceeds 1e-6 ({max_ref:.2e})"
        );
        println!("  size={width}×{height} r={radius}: max_abs={max_abs:.2e} max_ref={max_ref:.2e}");
    }
}

// ── Rolling-band tables are bit-identical to the whole-image tables ─────
//
// PERF-5: the production path keeps a ring of `band + 2r + 2` table rows. This
// is the previous whole-image construction (four full `(w+1)·(h+1)` tables,
// row prefix sums then a column pass), kept verbatim as the reference.

const fn ref_sum_range(a: i64, b: i64) -> i64 {
    (a + b) * (b - a + 1) / 2
}

const fn ref_sum_sq_range(a: i64, b: i64) -> i64 {
    let hi = b * (b + 1) * (2 * b + 1) / 6;
    let lo = if a == 0 {
        0
    } else {
        (a - 1) * a * (2 * a - 1) / 6
    };
    hi - lo
}

/// The four whole-image tables: `Σv`, `Σv²`, `Σ col·v`, `Σ row·v`.
type FullTables = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn full_tables(luma: &[f32], width: usize, height: usize) -> FullTables {
    let sat_w = width + 1;
    let sat_h = height + 1;
    let sat_len = sat_w * sat_h;
    let mut sat_v = vec![0.0_f64; sat_len];
    let mut sat_v2 = vec![0.0_f64; sat_len];
    let mut sat_col_v = vec![0.0_f64; sat_len];
    let mut sat_row_v = vec![0.0_f64; sat_len];
    for row in 0..height {
        let mut acc_v = 0.0_f64;
        let mut acc_v2 = 0.0_f64;
        let mut acc_col_v = 0.0_f64;
        let mut acc_row_v = 0.0_f64;
        for col in 0..width {
            let v = f64::from(luma[row * width + col]);
            acc_v += v;
            acc_v2 += v * v;
            acc_col_v += col as f64 * v;
            acc_row_v += row as f64 * v;
            let idx = (row + 1) * sat_w + (col + 1);
            sat_v[idx] = acc_v;
            sat_v2[idx] = acc_v2;
            sat_col_v[idx] = acc_col_v;
            sat_row_v[idx] = acc_row_v;
        }
    }
    for col in 1..sat_w {
        for row in 1..sat_h {
            let idx = row * sat_w + col;
            let above = (row - 1) * sat_w + col;
            sat_v[idx] += sat_v[above];
            sat_v2[idx] += sat_v2[above];
            sat_col_v[idx] += sat_col_v[above];
            sat_row_v[idx] += sat_row_v[above];
        }
    }
    (sat_v, sat_v2, sat_col_v, sat_row_v)
}

#[expect(
    clippy::similar_names,
    reason = "x/y and dx/dy pairs are intentionally symmetric"
)]
fn full_table_reference(luma: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let len = width * height;
    let sat_w = width + 1;
    let (sat_v, sat_v2, sat_col_v, sat_row_v) = full_tables(luma, width, height);
    let query = |sat: &[f64], x0: usize, y0: usize, x1: usize, y1: usize| -> f64 {
        let r1 = y1 + 1;
        let c1 = x1 + 1;
        sat[r1 * sat_w + c1] - sat[y0 * sat_w + c1] - sat[r1 * sat_w + x0] + sat[y0 * sat_w + x0]
    };
    let r = radius as isize;
    let w = width as isize;
    let h = height as isize;
    let mut out = vec![0.0_f32; len];
    for cy in 0..height {
        let icy = cy as isize;
        let y0 = (icy - r).max(0) as usize;
        let y1 = (icy + r).min(h - 1) as usize;
        let ny = (y1 - y0 + 1) as i64;
        for cx in 0..width {
            let icx = cx as isize;
            let x0 = (icx - r).max(0) as usize;
            let x1 = (icx + r).min(w - 1) as usize;
            let nx = (x1 - x0 + 1) as i64;
            let n = nx * ny;
            if n < 2 {
                continue;
            }
            let (ix0, ix1, iy0, iy1) = (x0 as i64, x1 as i64, y0 as i64, y1 as i64);
            let (cx_f, cy_f) = (cx as f64, cy as f64);
            let (fn64, fnx, fny) = (n as f64, nx as f64, ny as f64);
            let gx = ref_sum_range(ix0, ix1) as f64;
            let gxx = ref_sum_sq_range(ix0, ix1) as f64;
            let gy = ref_sum_range(iy0, iy1) as f64;
            let gyy = ref_sum_sq_range(iy0, iy1) as f64;
            let sum_dx = (gx - cx_f * fnx) * fny;
            let sum_dx2 = (gxx - 2.0 * cx_f * gx + cx_f * cx_f * fnx) * fny;
            let sum_dy = (gy - cy_f * fny) * fnx;
            let sum_dy2 = (gyy - 2.0 * cy_f * gy + cy_f * cy_f * fny) * fnx;
            let var_x = sum_dx2 - sum_dx * sum_dx / fn64;
            let var_y = sum_dy2 - sum_dy * sum_dy / fn64;
            if var_x <= 0.0 || var_y <= 0.0 {
                continue;
            }
            let sv = query(&sat_v, x0, y0, x1, y1);
            let sv2 = query(&sat_v2, x0, y0, x1, y1);
            let sxv = query(&sat_col_v, x0, y0, x1, y1);
            let syv = query(&sat_row_v, x0, y0, x1, y1);
            let mean_dx = sum_dx / fn64;
            let mean_dy = sum_dy / fn64;
            let covar_x = (sxv - cx_f * sv) - mean_dx * sv;
            let covar_y = (syv - cy_f * sv) - mean_dy * sv;
            let sum_sq_res = sv2 - covar_x * covar_x / var_x - covar_y * covar_y / var_y;
            let variance = (sum_sq_res - sv * sv / fn64) / (fn64 - 1.0);
            out[cy * width + cx] = variance.max(0.0) as f32;
        }
    }
    out
}

fn textured_luma(width: usize, height: usize) -> Vec<f32> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                ((x as f32 * 0.4 + y as f32 * 0.3).sin() * 0.5
                    + (x as f32 * 0.7 - y as f32 * 0.5).cos() * 0.3
                    + ((x * 31 + y * 17) % 13) as f32 * 0.02)
                    .clamp(0.0, 1.0)
            })
        })
        .collect()
}

/// The ring of table rows reproduces the whole-image tables bit for bit, for
/// every band height (including bands shorter than the window, a band of one
/// row, and a ring that wraps many times), image shape and radius.
#[test]
fn test_rolling_band_tables_are_bit_identical_to_full_tables() {
    let cases: &[(usize, usize, usize)] = &[
        (16, 16, 1),
        (16, 16, 3),
        (30, 20, 2),
        (20, 30, 2),
        (5, 5, 3),
        (7, 3, 2),
        (3, 7, 2),
        (1, 10, 2),
        (10, 1, 2),
        (1, 1, 1),
        (64, 48, 5),
        (50, 40, 100), // window larger than the image
        (37, 91, 4),
    ];
    for &(width, height, radius) in cases {
        let luma = textured_luma(width, height);
        let img = make_planar(width, height, luma.clone());
        let want = full_table_reference(&luma, width, height, radius);
        for band in [0_usize, 1, 2, 3, 7, 50, 128, 10_000] {
            let got = compute_focus_map_banded(&img, radius, band);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{width}x{height} r={radius} band={band}: pixel {i} is {g} vs {w}"
                );
            }
        }
        let default_band = compute_focus_map(&img, radius);
        assert!(
            default_band
                .iter()
                .zip(&want)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "{width}x{height} r={radius}: default band differs"
        );
    }
}

/// The single-plane map is the `luma` of the three-plane wrapper, whose chroma
/// planes stay zero (the wrapper keeps the old public contract).
#[test]
fn test_focus_map_matches_the_three_plane_wrapper() {
    let (width, height) = (23_usize, 19_usize);
    let img = make_planar(width, height, textured_luma(width, height));
    let map = compute_focus_map(&img, 2);
    let planar = compute_plane_fit_variance(&img, 2);
    assert_eq!(map.len(), width * height);
    assert_eq!(planar.luma, map);
    assert!(planar.chroma_a.iter().all(|&v| v.to_bits() == 0));
    assert!(planar.chroma_b.iter().all(|&v| v.to_bits() == 0));
    assert!(map.iter().any(|&v| v > 0.0), "texture must score");
}

/// Empty images give an empty map instead of panicking.
#[test]
fn test_focus_map_of_an_empty_image_is_empty() {
    assert_eq!(compute_focus_map(&make_planar(0, 0, vec![]), 3).len(), 0);
    assert_eq!(compute_focus_map(&make_planar(0, 5, vec![]), 3).len(), 0);
}

/// FUS-5: the three-plane wrapper also returns an empty map (and does not
/// panic) for a zero-width or zero-height image.
#[test]
fn test_plane_fit_variance_of_a_zero_width_image_is_empty() {
    let out = compute_plane_fit_variance(&PlanarImage::new(0, 4), 1);
    assert_eq!((out.width, out.height), (0, 4));
    assert_eq!(out.luma.len(), 0);
    let out = compute_plane_fit_variance(&PlanarImage::new(4, 0), 1);
    assert_eq!(out.luma.len(), 0);
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
        let out = compute_plane_fit_variance(&img, radius);
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
        let out = compute_plane_fit_variance(&img, radius);
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
    let out = compute_plane_fit_variance(&img, radius);

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
    let out = compute_plane_fit_variance(&img, 2);

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

/// Timing harness (report only, no assertion): naive O(N·r²) vs SAT O(N) on a
/// 512×512 image at radius 8; prints both times and their ratio.
///
/// Run with:
/// ```text
/// cargo test -p z-stackr-algo relief_sat_speedup -- --nocapture --ignored
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
        let out = compute_plane_fit_variance(&img, RADIUS);
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

    // Report-only: nothing about elapsed time is asserted (it depends on the
    // machine's load and thread count).  Correctness of the SAT path is
    // covered by `test_sat_equivalence`.
}
