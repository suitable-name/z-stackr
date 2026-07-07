use crate::{
    loss::helpers::{multi_scale_gradient_l1, occlusion_weight_map, sharpness_map},
    traits::{MergeStep, PairwiseFusionLoss},
};
use burn::prelude::*;

/// Configuration for [`FocusFusionLoss`] (the pairwise recurrent-merge loss).
///
/// See the [`crate::loss`] module docs' "Per-term reasoning" table for why
/// each term exists; the weights below are that table's default balance.
#[derive(Config, Debug)]
pub struct FocusFusionLossConfig {
    /// Weight of the robust (Charbonnier) RGB reconstruction term.
    #[config(default = 1.0)]
    pub w_charbonnier: f64,
    /// Weight of the multi-scale gradient L1 (high-frequency preservation) term.
    #[config(default = 1.0)]
    pub w_gradient: f64,
    /// Weight of the sharpness-retention term (penalises `merged` blurrier than `source`).
    #[config(default = 0.5)]
    pub w_sharpness: f64,
    /// Weight of the confidence-supervision term.
    #[config(default = 0.2)]
    pub w_confidence: f64,
    /// Charbonnier smoothing epsilon (`sqrt(diff^2 + eps^2)`); keeps the loss
    /// differentiable at zero residual.
    #[config(default = 0.001)]
    pub charbonnier_eps: f64,
    /// Floor of the occlusion down-weighting map, in `0..1` (never fully zeroes a term).
    #[config(default = 0.25)]
    pub occlusion_min_weight: f64,
    #[config(default = 3)]
    pub gradient_scales: usize,
}

/// Pairwise recurrent-merge training loss: Charbonnier, multi-scale
/// gradient, sharpness-retention, and confidence-supervision terms, each
/// down-weighted near depth edges by the scene's occlusion mask.
///
/// See the [`crate::loss`] module docs for the per-term rationale.
#[derive(Debug, Clone)]
pub struct FocusFusionLoss {
    w_charbonnier: f32,
    w_gradient: f32,
    w_sharpness: f32,
    w_confidence: f32,
    charbonnier_eps: f32,
    occlusion_min_weight: f32,
    gradient_scales: usize,
}

impl FocusFusionLossConfig {
    #[must_use]
    pub const fn init(&self) -> FocusFusionLoss {
        FocusFusionLoss {
            w_charbonnier: self.w_charbonnier as f32,
            w_gradient: self.w_gradient as f32,
            w_sharpness: self.w_sharpness as f32,
            w_confidence: self.w_confidence as f32,
            charbonnier_eps: self.charbonnier_eps as f32,
            occlusion_min_weight: self.occlusion_min_weight as f32,
            gradient_scales: self.gradient_scales,
        }
    }
}

impl<B: Backend> PairwiseFusionLoss<B> for FocusFusionLoss {
    fn forward(
        &self,
        pred: &MergeStep<B>,
        gt_merged: &Tensor<B, 4>,
        gt_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
        occlusion: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let device = &occlusion.device();

        let w = occlusion_weight_map(occlusion, self.occlusion_min_weight); // [N,1,H,W]

        let eps = self.charbonnier_eps;
        let diff = pred.merged.clone().sub(gt_merged.clone());
        let charb_map = diff.clone().mul(diff).add_scalar(eps * eps).sqrt(); // [N,3,H,W]
        let charbonnier = charb_map.mul(w.clone()).mean();

        let gradient =
            multi_scale_gradient_l1(&pred.merged, gt_merged, self.gradient_scales, device);

        let sharp_pred = sharpness_map(pred.merged.clone());
        let sharp_src = sharpness_map(source);

        let sharpness = sharp_src
            .sub(sharp_pred)
            .clamp_min(0.0_f32)
            .mul(w.clone())
            .mean();

        let confidence = pred.conf.clone().sub(gt_conf).abs().mul(w).mean();

        charbonnier
            .mul_scalar(self.w_charbonnier)
            .add(gradient.mul_scalar(self.w_gradient))
            .add(sharpness.mul_scalar(self.w_sharpness))
            .add(confidence.mul_scalar(self.w_confidence))
    }
}
