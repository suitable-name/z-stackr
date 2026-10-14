//! Tensor helpers shared by the fusion losses and the photometric loss.

use burn::{
    nn::pool::{AvgPool2d, AvgPool2dConfig},
    prelude::*,
};

/// Compute finite-difference gradients of `img [N,C,H,W]`.
///
/// Returns `(grad_x, grad_y)` where:
/// - `grad_x = img[.., .., .., 1:] - img[.., .., .., :-1]`  shape `[N,C,H,W-1]`
/// - `grad_y = img[.., .., 1:, ..] - img[.., .., :-1, ..]`  shape `[N,C,H-1,W]`
#[must_use]
pub fn finite_diff<B: Backend>(img: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [nb, nc, nh, nw] = img.dims();
    // grad_x: difference along width dimension (dim 3).
    let grad_x = img
        .clone()
        .slice([0..nb, 0..nc, 0..nh, 1..nw])
        .sub(img.clone().slice([0..nb, 0..nc, 0..nh, 0..(nw - 1)]));
    // grad_y: difference along height dimension (dim 2).
    let grad_y = img
        .clone()
        .slice([0..nb, 0..nc, 1..nh, 0..nw])
        .sub(img.slice([0..nb, 0..nc, 0..(nh - 1), 0..nw]));
    (grad_x, grad_y)
}

/// Mean of the weight map over the two pixels each finite difference spans.
///
/// Mirrors [`finite_diff`]: returns `(w_horizontal, w_vertical)` with shapes
/// `[N,1,H,W-1]` and `[N,1,H-1,W]`.
#[must_use]
pub fn finite_pair_mean<B: Backend>(w: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [nb, nc, nh, nw] = w.dims();
    let horiz = w
        .clone()
        .slice([0..nb, 0..nc, 0..nh, 1..nw])
        .add(w.clone().slice([0..nb, 0..nc, 0..nh, 0..(nw - 1)]))
        .mul_scalar(0.5_f32);
    let vert = w
        .clone()
        .slice([0..nb, 0..nc, 1..nh, 0..nw])
        .add(w.slice([0..nb, 0..nc, 0..(nh - 1), 0..nw]))
        .mul_scalar(0.5_f32);
    (horiz, vert)
}

/// 3×3 average-pool with stride 1, pad 1 (output same size as input).
///
/// `count_include_pad = false` so border pixels use only actual neighbourhood
/// values — this gives a true local average. The resulting Laplacian
/// `x - laplacian_pool(x)` is therefore a genuine high-pass filter.
#[must_use]
pub fn laplacian_pool<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    // `AvgPool2d` carries no parameters; building it on the fly is zero-cost.
    let pool: AvgPool2d = AvgPool2dConfig::new([3, 3])
        .with_strides([1, 1])
        .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
        .with_count_include_pad(false)
        .init();
    pool.forward(x)
}

/// Downsample `x` by `2^scale` via average pooling.
///
/// Returns `None` if either spatial dimension would fall below 2 after pooling,
/// preventing zero-size slices in the gradient computation.
#[must_use]
pub fn avg_pool_scale<B: Backend>(x: Tensor<B, 4>, scale: usize) -> Option<Tensor<B, 4>> {
    if scale == 0 {
        return Some(x);
    }
    let kernel = 1_usize << scale; // 2^scale
    let [_nb, _nc, nh, nw] = x.dims();
    if nh / kernel < 2 || nw / kernel < 2 {
        return None;
    }
    let pool: AvgPool2d = AvgPool2dConfig::new([kernel, kernel])
        .with_strides([kernel, kernel])
        .with_padding(burn::nn::PaddingConfig2d::Valid)
        .init();
    Some(pool.forward(x))
}

/// Per-pixel occlusion (depth-edge) down-weight map `w = 1 - (1 - min) · occlusion`,
/// in `[min, 1]`: `1` away from edges, `occlusion_min_weight` at them.
#[must_use]
pub fn occlusion_weight_map<B: Backend>(
    occlusion: &Tensor<B, 4>,
    occlusion_min_weight: f32,
) -> Tensor<B, 4> {
    occlusion
        .clone()
        .mul_scalar(1.0_f32 - occlusion_min_weight)
        .neg()
        .add_scalar(1.0_f32)
}

/// Per-pixel sharpness of `img` (`[N,C,H,W]`): the absolute 3×3 Laplacian
/// high-pass `|x − avgpool3×3(x)|`, averaged over the colour channels —
/// `[N,1,H,W]`.
#[must_use]
pub fn sharpness_map<B: Backend>(img: Tensor<B, 4>) -> Tensor<B, 4> {
    let lap = img.clone().sub(laplacian_pool(img));
    let [nb, nc, nh, nw] = lap.dims();
    lap.abs()
        .reshape([nb, nc, nh * nw])
        .mean_dim(1) // mean over the colour channels → [N, 1, H*W]
        .reshape([nb, 1, nh, nw])
}

/// Multi-scale gradient L1 term between `pred` and `gt` (`[N,C,H,W]`).
///
/// Scale `s` downsamples by `2^s` (scale 0 is full resolution); scales whose
/// pooled size falls below 2 px are skipped, and the result is the mean over the
/// scales that were used (zero if none were).
///
/// With `weight = Some(w)` (`[N,1,H,W]`, e.g. [`occlusion_weight_map`]) the map is
/// average-pooled to each scale and each finite difference is weighted by the mean
/// of `w` at the two pixels it spans, so a gradient that straddles a down-weighted
/// pixel is down-weighted too.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "the number of gradient scales is a handful"
)]
pub fn multi_scale_gradient_l1<B: Backend>(
    pred: &Tensor<B, 4>,
    gt: &Tensor<B, 4>,
    weight: Option<&Tensor<B, 4>>,
    gradient_scales: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut grad_acc: Option<Tensor<B, 1>> = None;
    let mut grad_count = 0_usize;

    for scale in 0..gradient_scales {
        let pred_scaled = avg_pool_scale(pred.clone(), scale);
        let gt_scaled = avg_pool_scale(gt.clone(), scale);
        let w_scaled = weight.map(|w| avg_pool_scale(w.clone(), scale));

        let (Some(pred_s), Some(gt_s)) = (pred_scaled, gt_scaled) else {
            continue;
        };
        // Compute each axis independently to avoid paired bindings that
        // trigger `clippy::similar_names` (x vs y differ by one char).
        let (pred_grad_horiz, pred_grad_vert) = finite_diff(pred_s);
        let (gt_grad_horiz, gt_grad_vert) = finite_diff(gt_s);
        let mut diff_horiz = pred_grad_horiz.sub(gt_grad_horiz).abs();
        let mut diff_vert = pred_grad_vert.sub(gt_grad_vert).abs();

        match w_scaled {
            None => {}
            Some(Some(w_s)) => {
                // w_* are [N,1,..] and broadcast over the colour channels.
                let (w_horiz, w_vert) = finite_pair_mean(w_s);
                diff_horiz = diff_horiz.mul(w_horiz);
                diff_vert = diff_vert.mul(w_vert);
            }
            Some(None) => continue,
        }

        let scale_term = diff_horiz.mean().add(diff_vert.mean());
        grad_acc = Some(match grad_acc {
            None => scale_term,
            Some(acc) => acc.add(scale_term),
        });
        grad_count += 1;
    }

    // Average across valid scales; fall back to zero if none were valid.
    match grad_acc {
        Some(acc) if grad_count > 0 => acc.div_scalar(grad_count as f32),
        _ => Tensor::<B, 1>::zeros([1], device),
    }
}
