//! The whole-stack focus-merge loss ([`FocusBatchLoss`]).
//!
//! The batch counterpart of [`crate::loss::FocusFusionLoss`], with the same review
//! fixes from the start: **every** term is weighted by the occlusion map, the
//! multi-scale gradient term weights each finite difference by the mean of the map
//! at the two pixels it spans, and the sharpness reference is the per-pixel
//! maximum over *all* frames of the stack.
//!
//! | Term | Weight | Description |
//! |---|---|---|
//! | Charbonnier | `w_charbonnier` | Robust RGB reconstruction against the ground truth. |
//! | Gradient L1 | `w_gradient` | Multi-scale finite-difference sharpness. |
//! | Sharpness | `w_sharpness` | Penalise output blurrier than the sharpest input frame (max over the stack). |
//! | Gate | `w_gate` | L1 of the softmax gate against `masks / Σ_S masks`, where `Σ_S masks` exceeds the threshold. |
//!
//! The gate term replaces the pairwise loss's confidence supervision: it
//! supervises the selection mechanism directly, but only where some frame claims
//! to be in focus; elsewhere the correct gate is genuinely ambiguous and
//! supervising it would fight the network.

use burn::prelude::*;

use crate::{
    loss::helpers::{multi_scale_gradient_l1, occlusion_weight_map, sharpness_map},
    model::blocks::max_over_stack,
    traits::BatchFusionLoss,
};

/// Configuration for [`FocusBatchLoss`].
#[derive(Config, Debug)]
pub struct FocusBatchLossConfig {
    /// Weight of the robust (Charbonnier) RGB reconstruction term.
    #[config(default = 1.0)]
    pub w_charbonnier: f64,
    /// Weight of the multi-scale gradient L1 (high-frequency preservation) term.
    #[config(default = 1.0)]
    pub w_gradient: f64,
    /// Weight of the sharpness-retention term (penalises `merged` blurrier than
    /// the sharpest input frame, per pixel, over the stack dimension).
    #[config(default = 0.5)]
    pub w_sharpness: f64,
    /// Weight of the gate-supervision term (the batch analogue of the pairwise
    /// confidence supervision).
    #[config(default = 0.2)]
    pub w_gate: f64,
    /// Charbonnier smoothing epsilon (`sqrt(diff^2 + eps^2)`).
    #[config(default = 0.001)]
    pub charbonnier_eps: f64,
    /// Floor of the occlusion down-weighting map, in `0..1`.
    #[config(default = 0.25)]
    pub occlusion_min_weight: f64,
    /// Number of pyramid scales of the gradient term (>= 1).
    #[config(default = 3)]
    pub gradient_scales: usize,
    /// Minimum `Σ_S masks` for a pixel to be included in the gate-supervision
    /// term.
    #[config(default = 0.05)]
    pub gate_mask_threshold: f64,
}

/// Resolved whole-stack loss: plain scalars, cheap to clone.
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
    /// Build a [`FocusBatchLoss`] from this config.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the f64 config values are narrowed to the f32 scalars the tensor ops use"
    )]
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

/// The individual terms of [`FocusBatchLoss::forward_terms`], each a scalar.
#[derive(Debug)]
pub struct BatchLossTerms<B: Backend> {
    /// Weighted sum of all terms.
    pub total: Tensor<B, 1>,
    /// Charbonnier reconstruction term.
    pub charbonnier: Tensor<B, 1>,
    /// Multi-scale gradient L1 term.
    pub gradient: Tensor<B, 1>,
    /// Sharpness-retention penalty.
    pub sharpness: Tensor<B, 1>,
    /// Gate-supervision term.
    pub gate: Tensor<B, 1>,
}

impl FocusBatchLoss {
    /// Full training loss (all four terms), returning the weighted total.
    ///
    /// See [`Self::forward_terms`] for the arguments.
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
        self.forward_terms(pred_merged, alpha, gt_merged, stack, masks, occlusion)
            .total
    }

    /// Full training loss (all four terms) with the individual terms for logging.
    ///
    /// * `pred_merged` – the network's composite `[N, 3, H, W]`.
    /// * `alpha`       – its softmax gate `[N, S, 1, H, W]`
    ///   ([`crate::model::BatchMergeNet::forward_with_alpha`]).
    /// * `gt_merged`   – ground-truth composite `[N, 3, H, W]`.
    /// * `stack`       – the input frames `[N, S, 3, H, W]` (sharpness reference).
    /// * `masks`       – per-frame ground-truth in-focus coverage `[N, S, 1, H, W]`;
    ///   `masks / Σ_S masks` is the gate target.
    /// * `occlusion`   – depth-edge mask `[N, 1, H, W]` in 0..1, 1 = edge.
    #[must_use]
    pub fn forward_terms<B: Backend>(
        &self,
        pred_merged: &Tensor<B, 4>,
        alpha: &Tensor<B, 5>,
        gt_merged: &Tensor<B, 4>,
        stack: &Tensor<B, 5>,
        masks: &Tensor<B, 5>,
        occlusion: &Tensor<B, 4>,
    ) -> BatchLossTerms<B> {
        let device = &occlusion.device();

        let w = occlusion_weight_map(occlusion, self.occlusion_min_weight); // [N,1,H,W]

        // 1. Charbonnier reconstruction.
        let eps = self.charbonnier_eps;
        let diff = pred_merged.clone().sub(gt_merged.clone());
        let charb_map = diff.clone().mul(diff).add_scalar(eps * eps).sqrt(); // [N,3,H,W]
        let charbonnier = charb_map.mul(w.clone()).mean();

        // 2. Multi-scale gradient L1, occlusion-weighted.
        let gradient = multi_scale_gradient_l1(
            pred_merged,
            gt_merged,
            Some(&w),
            self.gradient_scales,
            device,
        );

        // 3. Sharpness retention against the sharpest frame of the stack.
        let [n, s, _c, h, w_dim] = stack.dims();
        let stack_flat = stack.clone().reshape([n * s, 3, h, w_dim]);
        let sharp_per_frame = sharpness_map(stack_flat).reshape([n, s, 1, h, w_dim]);
        let ref_sharp = max_over_stack(&sharp_per_frame); // [N,1,H,W]
        let pred_sharp = sharpness_map(pred_merged.clone());
        let sharpness = ref_sharp
            .sub(pred_sharp)
            .clamp_min(0.0_f32)
            .mul(w.clone())
            .mean();

        // 4. Gate supervision against masks / Σ_S masks, where some frame claims
        //    to be in focus.
        let mask_sum = masks.clone().sum_dim(1); // [N,1,1,H,W]
        let target_alpha = masks.clone().div(mask_sum.clone().add_scalar(1e-6_f32)); // [N,S,1,H,W]
        let gate_active = mask_sum
            .squeeze::<4>(1) // [N,1,H,W]
            .greater_elem(self.gate_mask_threshold)
            .float(); // 1.0 where active, 0.0 elsewhere
        let gate_diff = alpha
            .clone()
            .sub(target_alpha)
            .abs()
            .sum_dim(1)
            .squeeze::<4>(1); // [N,1,H,W]
        let gate_w = gate_active.mul(w);
        let gate_denom = gate_w.clone().sum().add_scalar(1e-6_f32);
        let gate = gate_diff.mul(gate_w).sum().div(gate_denom);

        let total = charbonnier
            .clone()
            .mul_scalar(self.w_charbonnier)
            .add(gradient.clone().mul_scalar(self.w_gradient))
            .add(sharpness.clone().mul_scalar(self.w_sharpness))
            .add(gate.clone().mul_scalar(self.w_gate));

        BatchLossTerms {
            total,
            charbonnier,
            gradient,
            sharpness,
            gate,
        }
    }
}

impl<B: Backend> BatchFusionLoss<B> for FocusBatchLoss {
    /// Reconstruction-only subset of [`FocusBatchLoss::forward_full`] (Charbonnier
    /// plus the multi-scale gradient, both unweighted), for callers that only
    /// have the predicted and ground-truth composites.
    fn forward(&self, pred_merged: &Tensor<B, 4>, gt_merged: &Tensor<B, 4>) -> Tensor<B, 1> {
        let device = &pred_merged.device();

        let eps = self.charbonnier_eps;
        let diff = pred_merged.clone().sub(gt_merged.clone());
        let charbonnier = diff.clone().mul(diff).add_scalar(eps * eps).sqrt().mean();

        let gradient =
            multi_scale_gradient_l1(pred_merged, gt_merged, None, self.gradient_scales, device);

        charbonnier
            .mul_scalar(self.w_charbonnier)
            .add(gradient.mul_scalar(self.w_gradient))
    }
}
