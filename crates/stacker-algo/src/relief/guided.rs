//! The guided filter (He et al.) on `f32` planes.
//!
//! It is built on box filters that use `f64` summed-area tables:
//! [`guided_filter`] takes a [`PlanarImage`] and [`guided_filter_luma`] bare
//! planes. The `Relief` fusion uses it to smooth the selected image along its own
//! edges. [`guided_filter_pair_luma`] runs the filter twice, at two
//! `(radius, eps)` settings, on one guidance/source pair and shares the summed-area
//! tables of the four moments between the two runs (`Strata` needs exactly that).
//! [`box_filter`] is the underlying window mean.

#![expect(
    clippy::suboptimal_flops,
    reason = "the guided-filter algebra is evaluated as plain multiply-add, matching the published formulas and the recorded results"
)]
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "a window holds far fewer than 2^52 pixels, and the f64 window mean is narrowed to the f32 plane on purpose"
)]

use rayon::prelude::*;
use stacker_core::image::PlanarImage;

/// Build the `f64` summed-area table of a `width × height` plane.
///
/// The table has `(width + 1) × (height + 1)` entries; entry `(y, x)` holds the
/// sum of the samples of all rows `< y` and columns `< x`, and row 0 and column 0
/// are zero.
///
/// It is built in two passes. The first writes the horizontal prefix sum of every
/// row straight into its table row and runs over the rows in parallel. The second
/// adds each table row to the one above it. That pass is a sequential sweep over
/// whole rows (each step is one vectorised row addition), because a parallel
/// column split of a row-major table would only trade the sweep for strided
/// access; it is memory-bound and costs a small fraction of the first pass. Both
/// passes together need no memory besides the table itself (8 bytes per pixel).
///
/// Every entry is `row_prefix + entry_above`, evaluated in that order; the
/// result does not depend on the number of threads.
fn build_integral_image(data: &[f32], width: usize, height: usize) -> Vec<f64> {
    let stride = width + 1;
    let mut integral = vec![0.0_f64; stride * (height + 1)];
    if width == 0 || height == 0 {
        return integral;
    }

    // Pass A: the horizontal prefix sum of every image row, into table rows 1..=height.
    integral[stride..]
        .par_chunks_mut(stride)
        .zip(data.par_chunks(width))
        .for_each(|(table_row, image_row)| {
            let mut running = 0.0_f64;
            for (cell, &value) in table_row[1..].iter_mut().zip(image_row) {
                running += f64::from(value);
                *cell = running;
            }
        });

    // Pass B: add the table row above (column 0 stays zero).
    for y in 2..=height {
        let (above, rest) = integral.split_at_mut(y * stride);
        let previous = &above[(y - 1) * stride..];
        for (cell, &up) in rest[..stride].iter_mut().zip(previous) {
            *cell += up;
        }
    }
    integral
}

/// Read the mean over the `(2 * radius + 1)²` window (clamped at the borders)
/// around every pixel out of a summed-area table built by [`build_integral_image`].
fn box_filter_from_integral(
    integral: &[f64],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; width * height];
    if width == 0 || height == 0 {
        return out;
    }
    let stride = width + 1;
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let y_min = y.saturating_sub(radius);
        let y_max = (y + radius).min(height - 1);
        for (x, val) in row.iter_mut().enumerate() {
            let x_min = x.saturating_sub(radius);
            let x_max = (x + radius).min(width - 1);

            let sum = integral[(y_max + 1) * stride + (x_max + 1)]
                - integral[(y_max + 1) * stride + x_min]
                - integral[y_min * stride + (x_max + 1)]
                + integral[y_min * stride + x_min];

            let count = ((x_max - x_min + 1) * (y_max - y_min + 1)) as f64;
            *val = (sum / count) as f32;
        }
    });
    out
}

/// Mean over the `(2 * radius + 1)²` window around every pixel of a
/// `width × height` plane, with the window clipped at the borders (the divisor is
/// the number of pixels inside the image).
///
/// The summed-area table is accumulated in `f64`: the bottom-right entry of a
/// 24 MP image of values in `[0, 1]` is about `1e7`, where `f32` spacing is
/// `1.0`, so an `f32` table makes a window sum (a difference of four entries)
/// wrong by up to a few tenths.  `f64` keeps the error below `1e-8`.  The cost is
/// independent of `radius`. An empty plane gives an empty result.
///
/// # Panics
/// Panics when `data` holds fewer than `width * height` samples.
#[must_use]
pub fn box_filter(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    assert!(
        data.len() >= width * height,
        "the plane must hold width * height samples"
    );
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let integral = build_integral_image(data, width, height);
    box_filter_from_integral(&integral, width, height, radius)
}

/// The per-pixel product of two planes.
fn product(first: &[f32], second: &[f32], len: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; len];
    out.par_iter_mut().enumerate().for_each(|(i, value)| {
        *value = first[i] * second[i];
    });
    out
}

/// The linear coefficients `a` and `b` of the guided filter from the four window
/// means.
///
/// The variance is clamped at 0: a near-constant window can still leave a
/// fractionally negative cancellation residue even with the `f64` table, while the
/// variance is non-negative by definition (with a tiny `eps` a negative value
/// would blow `a` up).
fn coefficients(
    mean_i: &[f32],
    mean_p: &[f32],
    mean_i_p: &[f32],
    mean_i_i: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = mean_i.len();
    let mut a = vec![0.0_f32; len];
    let mut b = vec![0.0_f32; len];
    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .enumerate()
        .for_each(|(i, (a_out, b_out))| {
            let cov_i_p = mean_i_p[i] - mean_i[i] * mean_p[i];
            let var_i = (mean_i_i[i] - mean_i[i] * mean_i[i]).max(0.0);

            *a_out = cov_i_p / (var_i + eps);
            *b_out = mean_p[i] - *a_out * mean_i[i];
        });
    (a, b)
}

/// The filter output `mean(a) · guidance + mean(b)`.
fn blend(mean_a: &[f32], mean_b: &[f32], guidance: &[f32]) -> Vec<f32> {
    let mut q = vec![0.0_f32; mean_a.len()];
    q.par_iter_mut().enumerate().for_each(|(i, q_out)| {
        *q_out = mean_a[i] * guidance[i] + mean_b[i];
    });
    q
}

/// Guided filter of the luma of `src` with the luma of `guidance`; the chroma
/// planes of the result are zero. A thin wrapper over [`guided_filter_luma`].
///
/// # Panics
/// Panics when a luma plane does not hold `guidance.width * guidance.height`
/// samples.
#[must_use]
pub fn guided_filter(
    guidance: &PlanarImage<f32>,
    src: &PlanarImage<f32>,
    radius: usize,
    eps: f32,
) -> PlanarImage<f32> {
    let q = guided_filter_luma(
        &guidance.luma,
        &src.luma,
        guidance.width,
        guidance.height,
        radius,
        eps,
    );
    PlanarImage::with_luma(guidance.width, guidance.height, q)
        .expect("the filtered plane has one sample per pixel")
}

/// Guided filter on single `width × height` planes: `src` smoothed along the
/// edges of `guidance`. Both slices must hold `width * height` samples.
///
/// The filter runs six box filters (summed-area tables). When `guidance` and
/// `src` are the *same slice* (`std::ptr::eq`, the self-guided case the `Relief`
/// smoothing uses), `mean(src)` is `mean(guidance)` and `mean(guidance·src)` is
/// `mean(guidance²)` exactly, so two of the six are skipped; the result is
/// bit-identical to the general path. The window variance is clamped at 0 (see
/// [`guided_filter_pair_luma`]).
///
/// # Panics
/// Panics when a plane is shorter than `width * height`.
#[must_use]
pub fn guided_filter_luma(
    guidance: &[f32],
    src: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    eps: f32,
) -> Vec<f32> {
    let len = width * height;
    assert!(
        guidance.len() >= len && src.len() >= len,
        "the planes must hold width * height samples"
    );
    let self_guided = std::ptr::eq(guidance, src);

    let mean_i = box_filter(guidance, width, height, radius);
    let mean_p_owned = (!self_guided).then(|| box_filter(src, width, height, radius));
    let mean_p: &[f32] = mean_p_owned.as_deref().unwrap_or(&mean_i);

    let i_i = product(guidance, guidance, len);
    let mean_i_i = box_filter(&i_i, width, height, radius);
    drop(i_i);

    let mean_i_p_owned = (!self_guided).then(|| {
        let i_p = product(guidance, src, len);
        box_filter(&i_p, width, height, radius)
    });
    let mean_i_p: &[f32] = mean_i_p_owned.as_deref().unwrap_or(&mean_i_i);

    let (a, b) = coefficients(&mean_i, mean_p, mean_i_p, &mean_i_i, eps);
    let mean_a = box_filter(&a, width, height, radius);
    let mean_b = box_filter(&b, width, height, radius);
    blend(&mean_a, &mean_b, guidance)
}

/// Two guided filters of the same `guidance`/`src` pair, at the settings
/// `(radius_a, eps_a)` and `(radius_b, eps_b)`; returns `(result_a, result_b)`.
///
/// The result equals `(guided_filter_luma(guidance, src, width, height, radius_a,
/// eps_a), guided_filter_luma(guidance, src, width, height, radius_b, eps_b))`
/// (the two paths run the same arithmetic in the same order), but the summed-area
/// tables of `guidance`, `src`, `guidance·src` and `guidance²` do not depend on
/// the radius or `eps`, so they are built once and read at both radii. Only the
/// coefficient solve and the `mean(a)` / `mean(b)` steps run twice. When
/// `guidance` and `src` are the same slice, the `src` and `guidance·src` tables
/// are the `guidance` and `guidance²` tables and are not built.
///
/// While the pair runs it holds the four tables (32 bytes per pixel) next to the
/// planes of the running chain; use [`guided_filter_luma`] when only one setting
/// is needed.
///
/// Empty planes give two empty results.
///
/// # Panics
/// Panics when a plane is shorter than `width * height`.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "the signature is the contract with the Strata caller: a guidance/source pair and two (radius, eps) settings"
)]
pub fn guided_filter_pair_luma(
    guidance: &[f32],
    src: &[f32],
    width: usize,
    height: usize,
    radius_a: usize,
    eps_a: f32,
    radius_b: usize,
    eps_b: f32,
) -> (Vec<f32>, Vec<f32>) {
    let len = width * height;
    assert!(
        guidance.len() >= len && src.len() >= len,
        "the planes must hold width * height samples"
    );
    if len == 0 {
        return (Vec::new(), Vec::new());
    }
    let self_guided = std::ptr::eq(guidance, src);

    let integral_i = build_integral_image(guidance, width, height);
    let i_i = product(guidance, guidance, len);
    let integral_i_i = build_integral_image(&i_i, width, height);
    drop(i_i);
    let (integral_p, integral_i_p) = if self_guided {
        (None, None)
    } else {
        let integral_p = build_integral_image(src, width, height);
        let i_p = product(guidance, src, len);
        let integral_i_p = build_integral_image(&i_p, width, height);
        (Some(integral_p), Some(integral_i_p))
    };
    let table_p = integral_p.as_deref().unwrap_or(&integral_i);
    let table_i_p = integral_i_p.as_deref().unwrap_or(&integral_i_i);

    let run_chain = |radius: usize, eps: f32| -> Vec<f32> {
        let mean_i = box_filter_from_integral(&integral_i, width, height, radius);
        let mean_p = box_filter_from_integral(table_p, width, height, radius);
        let mean_i_p = box_filter_from_integral(table_i_p, width, height, radius);
        let mean_i_i = box_filter_from_integral(&integral_i_i, width, height, radius);
        let (a, b) = coefficients(&mean_i, &mean_p, &mean_i_p, &mean_i_i, eps);
        drop((mean_i, mean_p, mean_i_p, mean_i_i));

        let mean_a = box_filter(&a, width, height, radius);
        drop(a);
        let mean_b = box_filter(&b, width, height, radius);
        drop(b);
        blend(&mean_a, &mean_b, guidance)
    };

    (run_chain(radius_a, eps_a), run_chain(radius_b, eps_b))
}
