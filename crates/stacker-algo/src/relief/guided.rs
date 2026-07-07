use rayon::prelude::*;
use stacker_core::image::PlanarImage;

/// Box-filter (mean over a `(2*radius+1)²` window) via a summed-area table.
///
/// # Numerical precision
///
/// The integral image is accumulated in **`f64`**, mirroring the SAT
/// construction in `relief::focus` (see that module's doc comment for the
/// rationale). At a few megapixels with values in `0..1`, an `f32` running
/// sum reaches ~1e6-1e7, where the `f32` ULP is between 0.06 and 1.0 — i.e.
/// comparable to or larger than the small per-window sums (~10) recovered by
/// the 4-corner difference. That difference is then catastrophically
/// cancelled, turning `mean_i`/`mean_p`/`var_i` into noise that blows up
/// `a = cov / (var + eps)`. Accumulating in `f64` (ULP ~2e-10 at 1e7) keeps
/// the 4-corner difference accurate to many more significant digits than the
/// window sum needs, and the result is only narrowed to `f32` at the final
/// per-pixel division below.
/// `pub(crate)` (rather than private) so `strata::fuse_strata_with_progress`
/// can reuse this exact box filter for its base/detail decomposition
/// (`docs/strata-fusion-design.md` §2 Step 1) instead of duplicating it.
///
/// # Always CPU — no per-call GPU dispatch
///
/// This function is always the CPU/SAT implementation below, even in `gpu`
/// builds. It used to try a `wgpu` compute-shader dispatch on every call,
/// but [`guided_filter`] calls it six separate times: `mean_i`, `mean_p`,
/// `mean_i_p`, `mean_i_i`, `mean_a`, `mean_b`. Each of those six calls used
/// to be its own *separate* two-pass GPU round trip (horizontal + vertical,
/// each with its own texture upload, dispatch, and readback while holding
/// the process-wide `stacker_core::gpu::dispatch_guard()` mutex), for
/// twelve serialized full-image GPU round trips per `guided_filter` call —
/// slower than this rayon-parallel `f64` SAT path, and serialized in a way
/// that collapsed what should be parallel CPU work down to one core.
///
/// [`guided_filter`] now tries a single **fused** GPU pipeline
/// ([`crate::relief::gpu::guided_filter_gpu`]) that runs the entire guided
/// filter — including all six box-mean steps — as one held dispatch guard,
/// one upload, one readback; this per-call `box_filter` is the fallback
/// (and the only path at all in a non-`gpu` build) and stays CPU-only so it
/// composes safely with `strata::fuse_strata_with_progress`'s *direct* call
/// for the base/detail split (see that module's docs — that direct call is
/// deliberately not part of the fused guided-filter GPU path and remains
/// CPU-only).
pub(crate) fn box_filter(img_data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![0.0; width * height];
    }
    let integral = build_integral_image(img_data, width, height);
    box_filter_from_integral(&integral, width, height, radius)
}

/// Build the `f64`-accumulated summed-area table for `img_data` — the
/// radius-independent half of [`box_filter`], factored out so
/// [`guided_filter_pair`] can build it ONCE per input plane and reuse it for
/// every radius that plane's guided-filter chain needs, instead of
/// redundantly re-scanning the same data once per radius (the SAT
/// construction, not the windowed read-out, is the expensive part of a box
/// filter at these image sizes).
///
/// See [`box_filter`]'s doc comment for the parallelisation strategy this
/// mirrors exactly (same two rayon passes, same bit-for-bit result); this
/// function is that same body with the final windowed-readout pass
/// (which is what actually depends on `radius`) removed.
fn build_integral_image(img_data: &[f32], width: usize, height: usize) -> Vec<f64> {
    let stride = width + 1;
    let mut integral = vec![0.0_f64; stride * (height + 1)];

    // Pass A: horizontal prefix sum per row.
    let mut row_prefix = vec![0.0_f64; width * height];
    row_prefix
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let mut acc = 0.0_f64;
            for (x, out) in row.iter_mut().enumerate() {
                acc += f64::from(img_data[y * width + x]);
                *out = acc;
            }
        });

    // Pass B: vertical accumulation, parallelised across column chunks.
    // Chunk width chosen from the available thread count; each chunk owns
    // a contiguous `[x_lo, x_hi)` span of columns and fills its rows
    // sequentially into a small owned per-chunk buffer, then the results
    // are scattered back into `integral` — keeps the code free of
    // `unsafe`, still just two rayon passes total.
    let row_prefix_ref = &row_prefix;
    let num_chunks = rayon::current_num_threads().max(1);
    let chunk_width = width.div_ceil(num_chunks).max(1);
    let column_ranges: Vec<(usize, usize)> = (0..width)
        .step_by(chunk_width)
        .map(|x_lo| (x_lo, (x_lo + chunk_width).min(width)))
        .collect();

    let filled: Vec<Vec<f64>> = column_ranges
        .par_iter()
        .map(|&(x_lo, x_hi)| {
            let w = x_hi - x_lo;
            let mut band = vec![0.0_f64; w * height];
            for y in 0..height {
                for cx in 0..w {
                    let x = x_lo + cx;
                    let above = if y == 0 { 0.0 } else { band[(y - 1) * w + cx] };
                    band[y * w + cx] = row_prefix_ref[y * width + x] + above;
                }
            }
            band
        })
        .collect();

    for (&(x_lo, x_hi), band) in column_ranges.iter().zip(filled.iter()) {
        let w = x_hi - x_lo;
        for y in 0..height {
            for cx in 0..w {
                let x = x_lo + cx;
                integral[(y + 1) * stride + (x + 1)] = band[y * w + cx];
            }
        }
    }
    integral
}

/// Read a box-mean of radius `radius` out of a pre-built summed-area table
/// (see [`build_integral_image`]) — the radius-dependent half of
/// [`box_filter`], factored out so the same `integral` can be read at
/// multiple radii without rebuilding it.
fn box_filter_from_integral(
    integral: &[f64],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let y_min = y.saturating_sub(radius);
        let y_max = (y + radius).min(height.saturating_sub(1));
        for (x, val) in row.iter_mut().enumerate() {
            let x_min = x.saturating_sub(radius);
            let x_max = (x + radius).min(width.saturating_sub(1));

            let sum = integral[(y_max + 1) * (width + 1) + (x_max + 1)]
                - integral[(y_max + 1) * (width + 1) + x_min]
                - integral[y_min * (width + 1) + (x_max + 1)]
                + integral[y_min * (width + 1) + x_min];

            let count = ((x_max - x_min + 1) * (y_max - y_min + 1)) as f64;
            *val = (sum / count) as f32;
        }
    });
    out
}

pub fn guided_filter(
    guidance: &PlanarImage<f32>,
    src: &PlanarImage<f32>,
    radius: usize,
    eps: f32,
) -> PlanarImage<f32> {
    let width = guidance.width;
    let height = guidance.height;

    #[cfg(feature = "gpu")]
    if let Some(luma) = crate::relief::gpu::guided_filter_gpu(guidance, src, radius, eps) {
        return PlanarImage {
            width,
            height,
            luma,
            chroma_a: vec![0.0; width * height],
            chroma_b: vec![0.0; width * height],
        };
    }

    let mean_i = box_filter(&guidance.luma, width, height, radius);
    let mean_p = box_filter(&src.luma, width, height, radius);

    let mut i_p = vec![0.0; width * height];
    let mut i_i = vec![0.0; width * height];

    i_p.par_iter_mut()
        .zip(i_i.par_iter_mut())
        .enumerate()
        .for_each(|(i, (p_out, i_out))| {
            *p_out = guidance.luma[i] * src.luma[i];
            *i_out = guidance.luma[i] * guidance.luma[i];
        });

    let mean_i_p = box_filter(&i_p, width, height, radius);
    let mean_i_i = box_filter(&i_i, width, height, radius);

    let mut a = vec![0.0; width * height];
    let mut b = vec![0.0; width * height];

    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .enumerate()
        .for_each(|(i, (a_out, b_out))| {
            let cov_i_p = mean_i_p[i] - mean_i[i] * mean_p[i];
            // Clamp to >= 0: even with the f64 SAT above, subtractive
            // cancellation can still push this fractionally negative for a
            // near-constant window (mathematically var_i >= 0 always).
            let var_i = (mean_i_i[i] - mean_i[i] * mean_i[i]).max(0.0);

            *a_out = cov_i_p / (var_i + eps);
            *b_out = mean_p[i] - *a_out * mean_i[i];
        });

    let mean_a = box_filter(&a, width, height, radius);
    let mean_b = box_filter(&b, width, height, radius);

    let mut q = vec![0.0; width * height];
    q.par_iter_mut().enumerate().for_each(|(i, q_out)| {
        *q_out = mean_a[i] * guidance.luma[i] + mean_b[i];
    });

    PlanarImage {
        width,
        height,
        luma: q,
        chroma_a: vec![0.0; width * height],
        chroma_b: vec![0.0; width * height],
    }
}

/// Run [`guided_filter`] TWICE against the SAME `guidance`/`src` pair, at
/// two different `(radius, eps)` combinations, sharing work the two
/// independent calls would otherwise duplicate.
///
/// This is Strata's Pass 2 shape exactly (design doc §2 Step 4): the same
/// per-frame `(frame, p_image)` pair feeds both the wide/smooth
/// `w_base = guided_filter(frame, p_image, R_BIG, EPS_BIG)` and the
/// tight/crisp `w_detail = guided_filter(frame, p_image, R_SMALL,
/// EPS_SMALL)` — two full guided-filter passes over the *same inputs*,
/// previously computed by two entirely independent `guided_filter` calls
/// (two uploads of the same `guidance`/`src` on the GPU path, two lock
/// holds, two readbacks; two redundant SAT builds of `I`/`p`/`I*p`/`I*I` on
/// the CPU path, since none of those four SATs depend on radius or eps).
///
/// Returns `(result_for_radius_a, result_for_radius_b)`.
///
/// # GPU path
/// Tries [`crate::relief::gpu::guided_filter_pair_gpu`] first: one upload of
/// `I`/`p`, both radius chains dispatched under a single held
/// `dispatch_guard`, two readbacks batched at the end — see that function's
/// doc comment. Falls back to the CPU path below on any failure, exactly
/// like [`guided_filter`]'s own fallback contract.
///
/// # CPU path — shared SAT construction
/// `I` and `p`'s summed-area tables (via [`build_integral_image`]) and the
/// `I*p`/`I*I` elementwise products (which also do not depend on
/// radius/eps) are each built ONCE and reused for both radii's box-mean
/// reads ([`box_filter_from_integral`]) — the CPU-side equivalent of the
/// GPU path's single upload. Only the `a`/`b` coefficient solve and the
/// `mean_a`/`mean_b`/final-blend steps run twice (once per radius), since
/// those genuinely differ between the two chains (`eps` enters the
/// `a`/`b` solve, and `mean_a`/`mean_b` are box-filtered at the
/// chain-specific radius).
pub fn guided_filter_pair(
    guidance: &PlanarImage<f32>,
    src: &PlanarImage<f32>,
    radius_a: usize,
    eps_a: f32,
    radius_b: usize,
    eps_b: f32,
) -> (PlanarImage<f32>, PlanarImage<f32>) {
    let width = guidance.width;
    let height = guidance.height;

    #[cfg(feature = "gpu")]
    if let Some((luma_a, luma_b)) =
        crate::relief::gpu::guided_filter_pair_gpu(guidance, src, radius_a, eps_a, radius_b, eps_b)
    {
        let wrap = |luma: Vec<f32>| PlanarImage {
            width,
            height,
            luma,
            chroma_a: vec![0.0; width * height],
            chroma_b: vec![0.0; width * height],
        };
        return (wrap(luma_a), wrap(luma_b));
    }

    if width == 0 || height == 0 {
        let empty = || PlanarImage {
            width,
            height,
            luma: Vec::new(),
            chroma_a: Vec::new(),
            chroma_b: Vec::new(),
        };
        return (empty(), empty());
    }

    // Elementwise products: computed once, radius/eps-independent.
    let mut i_p = vec![0.0; width * height];
    let mut i_i = vec![0.0; width * height];
    i_p.par_iter_mut()
        .zip(i_i.par_iter_mut())
        .enumerate()
        .for_each(|(i, (p_out, i_out))| {
            *p_out = guidance.luma[i] * src.luma[i];
            *i_out = guidance.luma[i] * guidance.luma[i];
        });

    // Shared SATs: built once, read at each radius below via
    // `box_filter_from_integral` instead of rebuilding per radius.
    let integral_i = build_integral_image(&guidance.luma, width, height);
    let integral_p = build_integral_image(&src.luma, width, height);
    let integral_i_p = build_integral_image(&i_p, width, height);
    let integral_i_i = build_integral_image(&i_i, width, height);

    let run_chain = |radius: usize, eps: f32| -> PlanarImage<f32> {
        let mean_i = box_filter_from_integral(&integral_i, width, height, radius);
        let mean_p = box_filter_from_integral(&integral_p, width, height, radius);
        let mean_i_p = box_filter_from_integral(&integral_i_p, width, height, radius);
        let mean_i_i = box_filter_from_integral(&integral_i_i, width, height, radius);

        let mut a = vec![0.0; width * height];
        let mut b = vec![0.0; width * height];
        a.par_iter_mut()
            .zip(b.par_iter_mut())
            .enumerate()
            .for_each(|(i, (a_out, b_out))| {
                let cov_i_p = mean_i_p[i] - mean_i[i] * mean_p[i];
                let var_i = (mean_i_i[i] - mean_i[i] * mean_i[i]).max(0.0);
                *a_out = cov_i_p / (var_i + eps);
                *b_out = mean_p[i] - *a_out * mean_i[i];
            });

        let integral_a = build_integral_image(&a, width, height);
        let integral_b = build_integral_image(&b, width, height);
        let mean_a = box_filter_from_integral(&integral_a, width, height, radius);
        let mean_b = box_filter_from_integral(&integral_b, width, height, radius);

        let mut q = vec![0.0; width * height];
        q.par_iter_mut().enumerate().for_each(|(i, q_out)| {
            *q_out = mean_a[i] * guidance.luma[i] + mean_b[i];
        });

        PlanarImage {
            width,
            height,
            luma: q,
            chroma_a: vec![0.0; width * height],
            chroma_b: vec![0.0; width * height],
        }
    };

    (run_chain(radius_a, eps_a), run_chain(radius_b, eps_b))
}
