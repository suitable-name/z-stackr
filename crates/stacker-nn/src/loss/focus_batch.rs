use crate::{
    loss::helpers::{multi_scale_gradient_l1, occlusion_weight_map, sharpness_map},
    traits::BatchFusionLoss,
};
use burn::prelude::*;

/// Configuration for [`FocusBatchLoss`] (the whole-stack batch-fusion loss).
///
/// Mirrors [`super::focus_fusion::FocusFusionLossConfig`]'s term weights —
/// see the [`crate::loss`] module docs' "Per-term reasoning" table for why
/// each term exists and how the batch adaptation differs from the pairwise
/// original (sharpness compares against the sharpest stack frame rather than
/// a single `source`; confidence supervision becomes gate supervision).
#[derive(Config, Debug)]
pub struct FocusBatchLossConfig {
    /// Weight of the robust (Charbonnier) RGB reconstruction term.
    #[config(default = 1.0)]
    pub w_charbonnier: f64,
    /// Weight of the multi-scale gradient L1 (high-frequency preservation) term.
    #[config(default = 1.0)]
    pub w_gradient: f64,
    /// Weight of the sharpness-retention term (penalises `merged` blurrier
    /// than the sharpest input frame, per pixel, over the stack dimension).
    #[config(default = 0.5)]
    pub w_sharpness: f64,
    /// Weight of the gate-supervision term (the batch analog of pairwise
    /// confidence supervision).
    #[config(default = 0.2)]
    pub w_gate: f64,
    /// Charbonnier smoothing epsilon (`sqrt(diff^2 + eps^2)`); keeps the loss
    /// differentiable at zero residual.
    #[config(default = 0.001)]
    pub charbonnier_eps: f64,
    /// Floor of the occlusion down-weighting map, in `0..1` (never fully zeroes a term).
    #[config(default = 0.25)]
    pub occlusion_min_weight: f64,
    #[config(default = 3)]
    pub gradient_scales: usize,
    /// Minimum `sum_S(masks)` for a pixel to be considered "some frame claims
    /// to be in focus here" and included in the gate-supervision term;
    /// elsewhere the correct gate is genuinely ambiguous and the term is
    /// masked out.
    #[config(default = 0.05)]
    pub gate_mask_threshold: f64,
}

/// Whole-stack batch-fusion training loss: Charbonnier, multi-scale
/// gradient, sharpness-retention, and gate-supervision terms, each
/// down-weighted near depth edges by the scene's occlusion mask — the batch
/// counterpart of [`super::focus_fusion::FocusFusionLoss`], brought up to
/// the same standard.
///
/// See the [`crate::loss`] module docs for the per-term rationale.
#[derive(Debug, Clone)]
pub struct FocusBatchLoss {
    w_charbonnier: f32,
    w_gradient: f32,
    w_sharpness: f32,
    w_gate: f32,
    charbonnier_eps: f32,
    occlusion_min_weight: f32,
    gradient_scales: usize,
    gate_mask_threshold: f32,
}

impl FocusBatchLossConfig {
    #[must_use]
    pub const fn init(&self) -> FocusBatchLoss {
        FocusBatchLoss {
            w_charbonnier: self.w_charbonnier as f32,
            w_gradient: self.w_gradient as f32,
            w_sharpness: self.w_sharpness as f32,
            w_gate: self.w_gate as f32,
            charbonnier_eps: self.charbonnier_eps as f32,
            occlusion_min_weight: self.occlusion_min_weight as f32,
            gradient_scales: self.gradient_scales,
            gate_mask_threshold: self.gate_mask_threshold as f32,
        }
    }
}

impl FocusBatchLoss {
    /// Full training loss (all four terms) — used by `train::batch_loss`,
    /// which has access to the model's raw stack/masks/gate output, unlike
    /// the trait-level [`BatchFusionLoss::forward`] (reconstruction-only,
    /// used where only `pred_merged`/`gt_merged` are available).
    ///
    /// `alpha` is the predicted softmax gate `[N,S,1,H,W]`
    /// (`BatchMergeNet::forward_with_alpha`'s second output); `masks` are the
    /// per-frame ground-truth in-focus coverage maps `[N,S,1,H,W]` used both
    /// to build the sharpness reference (max over frames) and the gate
    /// supervision target (`masks / sum_S(masks)`, i.e. a normalised
    /// "which frame should win here" distribution).
    #[must_use]
    pub fn forward_full<B: Backend>(
        &self,
        pred_merged: &Tensor<B, 4>,
        alpha: &Tensor<B, 5>,
        gt_merged: &Tensor<B, 4>,
        stack: &Tensor<B, 5>,
        masks: &Tensor<B, 5>,
        occlusion: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let device = &occlusion.device();

        let w = occlusion_weight_map(occlusion, self.occlusion_min_weight); // [N,1,H,W]

        let eps = self.charbonnier_eps;
        let diff = pred_merged.clone().sub(gt_merged.clone());
        let charb_map = diff.clone().mul(diff).add_scalar(eps * eps).sqrt(); // [N,3,H,W]
        let charbonnier = charb_map.mul(w.clone()).mean();

        let gradient =
            multi_scale_gradient_l1(pred_merged, gt_merged, self.gradient_scales, device);

        let [n, s, _c, h, w_dim] = stack.dims();
        let stack_flat = stack.clone().reshape([n * s, 3, h, w_dim]);
        let sharp_flat = sharpness_map(stack_flat); // [N*S, 1, H, W]
        let sharp_per_frame = sharp_flat.reshape([n, s, 1, h, w_dim]);
        let ref_sharp = sharp_per_frame.max_dim(1).squeeze_dim::<4>(1); // [N,1,H,W]

        let pred_sharp = sharpness_map(pred_merged.clone());
        let sharpness = ref_sharp
            .sub(pred_sharp)
            .clamp_min(0.0_f32)
            .mul(w.clone())
            .mean();

        // Gate supervision: build a target blending distribution from the
        // per-frame in-focus masks (normalised so it sums to 1 across the
        // stack dimension wherever at least one frame claims to be in
        // focus), and supervise the predicted softmax gate against it with
        // L1. Pixels where no frame claims to be in focus (`mask_sum` below
        // `gate_mask_threshold`) are excluded via `gate_active` — the
        // correct gate there is genuinely ambiguous, so supervising it would
        // fight the network rather than train it (mirrors the occlusion
        // down-weighting rationale, but as a hard mask rather than a floor).
        let mask_sum = masks.clone().sum_dim(1); // [N,1,1,H,W]
        let target_alpha = masks.clone().div(mask_sum.clone().add_scalar(1e-6_f32)); // [N,S,1,H,W]
        let gate_active = mask_sum
            .squeeze_dim::<4>(1) // [N,1,H,W]
            .greater_elem(self.gate_mask_threshold)
            .float(); // 1.0 where active, 0.0 elsewhere
        let gate_diff = alpha
            .clone()
            .sub(target_alpha)
            .abs()
            .sum_dim(1)
            .squeeze_dim::<4>(1); // [N,1,H,W]
        let gate_w = gate_active.mul(w);
        let gate_denom = gate_w.clone().sum().add_scalar(1e-6_f32);
        let gate = gate_diff.mul(gate_w).sum().div(gate_denom);

        charbonnier
            .mul_scalar(self.w_charbonnier)
            .add(gradient.mul_scalar(self.w_gradient))
            .add(sharpness.mul_scalar(self.w_sharpness))
            .add(gate.mul_scalar(self.w_gate))
    }
}

impl<B: Backend> BatchFusionLoss<B> for FocusBatchLoss {
    /// Reconstruction-only subset of [`Self::forward_full`] (Charbonnier +
    /// multi-scale gradient), for trait callers that only have
    /// `pred_merged`/`gt_merged` and not the full stack/masks/gate/occlusion
    /// context `forward_full` needs for the sharpness and gate terms.
    fn forward(&self, pred_merged: &Tensor<B, 4>, gt_merged: &Tensor<B, 4>) -> Tensor<B, 1> {
        let device = &pred_merged.device();

        let eps = self.charbonnier_eps;
        let diff = pred_merged.clone().sub(gt_merged.clone());
        let charb_map = diff.clone().mul(diff).add_scalar(eps * eps).sqrt();
        let charbonnier = charb_map.mean();

        let gradient =
            multi_scale_gradient_l1(pred_merged, gt_merged, self.gradient_scales, device);

        charbonnier
            .mul_scalar(self.w_charbonnier)
            .add(gradient.mul_scalar(self.w_gradient))
    }
}
