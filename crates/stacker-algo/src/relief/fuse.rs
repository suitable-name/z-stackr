use rayon::prelude::*;
use stacker_core::image::PlanarImage;

use crate::relief::{
    focus::compute_sum_modified_laplacian,
    guided::guided_filter,
    threshold::{ReliefSettings, generate_mask},
};

/// Compute the per-frame Sum Modified Laplacian and the pixel-wise maximum SML across all frames.
pub fn compute_relief_smls(
    images: &[PlanarImage<f32>],
    est_radius: usize,
) -> (Vec<PlanarImage<f32>>, PlanarImage<f32>) {
    compute_relief_smls_with_progress(images, est_radius, |_, _| {})
}

/// Like [`compute_relief_smls`], but invokes `on_frame_done(completed, total)`
/// each time one frame's SML map finishes.
///
/// The per-frame maps are computed in parallel (rayon), so `on_frame_done`
/// is called from worker threads in completion order — `completed` is a
/// monotonically increasing count, not a frame index. Callers using it to
/// drive UI progress must marshal to their event loop themselves.
pub fn compute_relief_smls_with_progress<F>(
    images: &[PlanarImage<f32>],
    est_radius: usize,
    on_frame_done: F,
) -> (Vec<PlanarImage<f32>>, PlanarImage<f32>)
where
    F: Fn(usize, usize) + Sync,
{
    let width = images[0].width;
    let height = images[0].height;
    let len = width * height;
    let total = images.len();
    let done = std::sync::atomic::AtomicUsize::new(0);

    let smls: Vec<PlanarImage<f32>> = images
        .par_iter()
        .map(|img| {
            let sml = compute_sum_modified_laplacian(img, est_radius);
            let completed = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            on_frame_done(completed, total);
            sml
        })
        .collect();

    let mut max_luma = vec![f32::NEG_INFINITY; len];
    for sml in &smls {
        for (m, &s) in max_luma.iter_mut().zip(sml.luma.iter()) {
            if s > *m {
                *m = s;
            }
        }
    }

    let max_sml = PlanarImage {
        width,
        height,
        luma: max_luma,
        chroma_a: vec![0.0; len],
        chroma_b: vec![0.0; len],
    };

    (smls, max_sml)
}

/// Computes an automatic contrast-percentile threshold using the Median
/// Absolute Deviation (MAD) of the non-zero focus-measure population.
///
/// The SML distribution is heavily skewed with a massive peak near zero (sensor noise
/// in flat/out-of-focus areas) and a long tail of strong edges. Otsu's method tends
/// to split the strong edges from everything else, resulting in excessively high
/// percentiles (e.g. 98%).
/// This uses a robust statistic (`median + 1.5 * MAD`) to identify the noise floor
/// instead, returning a much lower, more optimal percentile that preserves soft textures.
///
/// The returned fraction is a percentile **of the non-zero population**,
/// matching [`crate::relief::threshold::generate_mask`]'s interpretation of
/// `contrast_pct`, so the value can be fed straight into
/// [`crate::relief::threshold::ReliefSettings::contrast_pct`].
pub fn auto_contrast_threshold(img: &PlanarImage<f32>) -> f32 {
    if img.luma.is_empty() {
        return 0.0;
    }

    // Subsample for performance (we only need approximate percentiles)
    let step = (img.luma.len() / 100_000).max(1);
    let mut sample: Vec<f32> = img
        .luma
        .iter()
        .step_by(step)
        .copied()
        .filter(|&v| v > 0.0)
        .collect();

    if sample.is_empty() {
        return 0.0;
    }

    sample.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sample[sample.len() / 2];

    let mut abs_devs: Vec<f32> = sample.iter().map(|&v| (v - median).abs()).collect();
    abs_devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_devs[abs_devs.len() / 2];

    if mad < 1e-6 {
        return 0.0;
    }

    // threshold = median + 3 * MAD (typical robust noise threshold)
    // We use a slightly tighter bound (1.5 * MAD) to ensure the percentage stays lower,
    // as users prefer keeping more texture (Relief Multigrid handles noise well).
    let threshold_val = median + 1.5 * mad;

    let count = sample.iter().take_while(|&&v| v <= threshold_val).count();
    (count as f32 / sample.len() as f32).clamp(0.0, 1.0)
}

/// Fuses aligned images using their SMLs and a contrast mask, then applies guided filter.
pub fn fuse_relief_with_mask(
    images: &[PlanarImage<f32>],
    smls: &[PlanarImage<f32>],
    max_sml: &PlanarImage<f32>,
    settings: &ReliefSettings,
) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    let width = images[0].width;
    let height = images[0].height;
    let len = width * height;

    let mask = generate_mask(max_sml, settings);

    let mut out_luma = vec![0.0_f32; len];
    let mut out_chroma_a = vec![0.0_f32; len];
    let mut out_chroma_b = vec![0.0_f32; len];
    let n_frames = images.len() as f32;

    out_luma
        .par_iter_mut()
        .zip(out_chroma_a.par_iter_mut())
        .zip(out_chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((ol, oa), ob))| {
            if mask.get(i).copied().unwrap_or(true) {
                let mut best = f32::NEG_INFINITY;
                for (img, sml) in images.iter().zip(smls.iter()) {
                    if sml.luma[i] > best {
                        best = sml.luma[i];
                        *ol = img.luma[i];
                        *oa = img.chroma_a[i];
                        *ob = img.chroma_b[i];
                    }
                }
            } else {
                let mut sum_l = 0.0_f32;
                let mut sum_a = 0.0_f32;
                let mut sum_b = 0.0_f32;
                for img in images {
                    sum_l += img.luma[i];
                    sum_a += img.chroma_a[i];
                    sum_b += img.chroma_b[i];
                }
                *ol = sum_l / n_frames;
                *oa = sum_a / n_frames;
                *ob = sum_b / n_frames;
            }
        });

    let out = PlanarImage {
        width,
        height,
        luma: out_luma,
        chroma_a: out_chroma_a,
        chroma_b: out_chroma_b,
    };

    let smoothed = guided_filter(&out, &out, settings.smooth_radius, 0.01);
    PlanarImage {
        width,
        height,
        luma: smoothed.luma,
        chroma_a: out.chroma_a,
        chroma_b: out.chroma_b,
    }
}

/// Fuses aligned images using a confidence-weighted Multigrid solver for depth map interpolation.
pub fn fuse_relief_multigrid(
    images: &[PlanarImage<f32>],
    smls: &[PlanarImage<f32>],
    max_sml: &PlanarImage<f32>,
    settings: &ReliefSettings,
) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    let width = images[0].width;
    let height = images[0].height;
    let len = width * height;

    let mask = generate_mask(max_sml, settings);

    let mut target_index = vec![0.0_f32; len];
    let mut weight = vec![0.0_f32; len];

    for i in 0..len {
        if mask.get(i).copied().unwrap_or(true) {
            let mut best_val = f32::NEG_INFINITY;
            let mut best_idx = 0.0_f32;
            for (idx, sml) in smls.iter().enumerate() {
                if sml.luma[i] > best_val {
                    best_val = sml.luma[i];
                    best_idx = idx as f32;
                }
            }
            target_index[i] = best_idx;
            weight[i] = 1.0;
        } else {
            target_index[i] = 0.0;
            weight[i] = 0.0;
        }
    }

    let mut solver =
        crate::relief::multigrid::MultigridSolver::new(width, height, &target_index, &weight);
    solver.solve();
    // Post-solve smoothing: a Gaussian-pyramid reduce/expand chain — see
    // `crate::relief::pyramid` docs for why this (not a guided filter) is
    // the right smoothing pass here.
    let final_index = solver.get_smoothed_solution(settings.smooth_radius);

    let mut out_luma = vec![0.0_f32; len];
    let mut out_chroma_a = vec![0.0_f32; len];
    let mut out_chroma_b = vec![0.0_f32; len];
    let max_idx = (images.len() - 1) as f32;

    out_luma
        .par_iter_mut()
        .zip(out_chroma_a.par_iter_mut())
        .zip(out_chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((ol, oa), ob))| {
            let idx = final_index[i].clamp(0.0, max_idx);
            let idx0 = idx.floor() as usize;
            let idx1 = idx.ceil() as usize;
            let frac = idx - (idx0 as f32);

            let img0 = &images[idx0];
            let img1 = &images[idx1];

            *ol = img0.luma[i] * (1.0 - frac) + img1.luma[i] * frac;
            *oa = img0.chroma_a[i] * (1.0 - frac) + img1.chroma_a[i] * frac;
            *ob = img0.chroma_b[i] * (1.0 - frac) + img1.chroma_b[i] * frac;
        });

    PlanarImage {
        width,
        height,
        luma: out_luma,
        chroma_a: out_chroma_a,
        chroma_b: out_chroma_b,
    }
}
