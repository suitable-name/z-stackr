use burn::{
    nn::pool::{AvgPool2d, AvgPool2dConfig},
    prelude::*,
};

/// Compute finite-difference gradients of `img [N,C,H,W]`.
pub(crate) fn finite_diff<B: Backend>(img: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [nb, nc, nh, nw] = img.dims();
    let grad_x = img
        .clone()
        .slice([0..nb, 0..nc, 0..nh, 1..nw])
        .sub(img.clone().slice([0..nb, 0..nc, 0..nh, 0..(nw - 1)]));
    let grad_y = img
        .clone()
        .slice([0..nb, 0..nc, 1..nh, 0..nw])
        .sub(img.slice([0..nb, 0..nc, 0..(nh - 1), 0..nw]));
    (grad_x, grad_y)
}

/// 3×3 average-pool with stride 1, pad 1 (output same size as input).
pub(crate) fn laplacian_pool<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let pool: AvgPool2d = AvgPool2dConfig::new([3, 3])
        .with_strides([1, 1])
        .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_count_include_pad(false)
        .init();
    pool.forward(x)
}

/// Downsample `x` by `2^scale` via average pooling.
pub(crate) fn avg_pool_scale<B: Backend>(x: Tensor<B, 4>, scale: usize) -> Option<Tensor<B, 4>> {
    if scale == 0 {
        return Some(x);
    }
    let kernel = 1_usize << scale;
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

/// Per-pixel occlusion (depth-edge) down-weight map.
pub(crate) fn occlusion_weight_map<B: Backend>(
    occlusion: &Tensor<B, 4>,
    occlusion_min_weight: f32,
) -> Tensor<B, 4> {
    occlusion
        .clone()
        .mul_scalar(1.0_f32 - occlusion_min_weight)
        .neg()
        .add_scalar(1.0_f32)
}

/// Per-pixel Laplacian-high-pass sharpness measure.
pub(crate) fn sharpness_map<B: Backend>(img: Tensor<B, 4>) -> Tensor<B, 4> {
    let [nb, _nc, nh, nw] = img.dims();
    let lap = img.clone().sub(laplacian_pool(img));
    lap.abs()
        .reshape([nb, 3, nh * nw])
        .mean_dim(1)
        .reshape([nb, 1, nh, nw])
}

/// Multi-scale gradient L1 term.
pub(crate) fn multi_scale_gradient_l1<B: Backend>(
    pred: &Tensor<B, 4>,
    gt: &Tensor<B, 4>,
    gradient_scales: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut grad_acc: Option<Tensor<B, 1>> = None;
    let mut grad_count = 0_usize;

    for scale in 0..gradient_scales {
        let pred_scaled = avg_pool_scale(pred.clone(), scale);
        let gt_scaled = avg_pool_scale(gt.clone(), scale);

        if let (Some(pred_s), Some(gt_s)) = (pred_scaled, gt_scaled) {
            let (pred_grad_horiz, pred_grad_vert) = finite_diff(pred_s);
            let (gt_grad_horiz, gt_grad_vert) = finite_diff(gt_s);

            let scale_term = pred_grad_horiz
                .sub(gt_grad_horiz)
                .abs()
                .mean()
                .add(pred_grad_vert.sub(gt_grad_vert).abs().mean());

            grad_acc = Some(match grad_acc {
                None => scale_term,
                Some(acc) => acc.add(scale_term),
            });
            grad_count += 1;
        }
    }

    match grad_acc {
        Some(acc) if grad_count > 0 => acc.div_scalar(grad_count as f32),
        _ => Tensor::<B, 1>::zeros([1], device),
    }
}
