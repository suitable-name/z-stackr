//! The pairwise focus-merge loss ([`FocusFusionLoss`]).
//!
//! ## Terms
//!
//! | Term | Weight | Description |
//! |---|---|---|
//! | Charbonnier | `w_charbonnier` | Robust RGB reconstruction (smooth L1). |
//! | Gradient L1 | `w_gradient` | Multi-scale finite-difference sharpness. |
//! | Sharpness | `w_sharpness` | Penalise output blurrier than the sharpest frame merged so far. |
//! | Confidence | `w_confidence` | L1 supervision of predicted confidence. |
//!
//! ## Occlusion weighting
//!
//! A per-pixel weight map `w = 1 - (1 - occlusion_min_weight) * occlusion`
//! down-weights depth-edge pixels where supervision is ambiguous. **All four
//! terms** are weighted by it. For the multi-scale gradient term the map is
//! average-pooled to each scale and each finite difference is weighted by the mean
//! of `w` at the two pixels it spans, so a gradient that straddles an occluded
//! pixel is down-weighted too.
//!
//! ## Sharpness reference
//!
//! The sharpness hinge is `relu(ref − sharp(pred))` where `ref` is the per-pixel
//! maximum of [`sharpness_map`] over *all frames merged so far* (the first frame
//! and every source up to and including the current one), not the current source
//! alone: an output may not be blurrier than the sharpest evidence available at
//! that pixel. [`FocusFusionLoss::forward_with_reference`] takes `ref` directly
//! (the rollout in [`crate::train`] maintains the running maximum);
//! [`FocusFusionLoss::forward`] keeps the single-source form by using
//! `sharpness_map(source)` as the reference.

use burn::prelude::*;

use crate::{
    loss::helpers::{multi_scale_gradient_l1, occlusion_weight_map, sharpness_map},
    model::MergeOutput,
    traits::{MergeStep, PairwiseFusionLoss},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`FocusFusionLoss`].
#[derive(Config, Debug)]
pub struct FocusFusionLossConfig {
    /// Weight for the Charbonnier (robust RGB reconstruction) term.
    #[config(default = 1.0)]
    pub w_charbonnier: f64,
    /// Weight for the multi-scale gradient L1 term.
    #[config(default = 1.0)]
    pub w_gradient: f64,
    /// Weight for the sharpness-retention term.
    #[config(default = 0.5)]
    pub w_sharpness: f64,
    /// Weight for the confidence supervision term.
    #[config(default = 0.2)]
    pub w_confidence: f64,
    /// Epsilon for Charbonnier smoothing (`sqrt(x^2 + eps^2)`).
    #[config(default = 0.001)]
    pub charbonnier_eps: f64,
    /// Loss weight applied AT occlusion-edge pixels (1.0 = no down-weighting).
    #[config(default = 0.25)]
    pub occlusion_min_weight: f64,
    /// Number of pyramid scales for the multi-scale gradient term (>= 1).
    /// Scale `s` downsamples by `2^s`; scale 0 is full resolution.
    #[config(default = 3)]
    pub gradient_scales: usize,
}

// ---------------------------------------------------------------------------
// Loss struct
// ---------------------------------------------------------------------------

/// Resolved, ready-to-use multi-term focus-merge loss.
///
/// This struct is `Clone` + `Debug` and holds only plain scalars — no tensors,
/// no device state — so it is cheap to clone and share across threads.
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
    /// Build a [`FocusFusionLoss`] from this config.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the f64 config values are narrowed to the f32 scalars the tensor ops use"
    )]
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

// ---------------------------------------------------------------------------
// Individual loss terms (returned for logging)
// ---------------------------------------------------------------------------

/// All individual loss terms plus the weighted total, each reduced to a scalar
/// `Tensor<B, 1>` for easy logging.
#[derive(Debug)]
pub struct LossTerms<B: Backend> {
    /// Weighted sum of all terms.
    pub total: Tensor<B, 1>,
    /// Charbonnier robust RGB reconstruction term.
    pub charbonnier: Tensor<B, 1>,
    /// Multi-scale gradient L1 term.
    pub gradient: Tensor<B, 1>,
    /// Sharpness-retention penalty.
    pub sharpness: Tensor<B, 1>,
    /// Confidence L1 supervision term.
    pub confidence: Tensor<B, 1>,
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

impl FocusFusionLoss {
    /// Compute all loss terms.
    ///
    /// # Arguments
    /// * `pred`       – network predictions (merged, conf, alpha).
    /// * `gt_merged`  – ground-truth composite `[N, 3, H, W]`.
    /// * `gt_conf`    – ground-truth confidence `[N, 1, H, W]`.
    /// * `source`     – source frame `[N, 3, H, W]`; its [`sharpness_map`] is the
    ///   sharpness reference. Use [`Self::forward_with_reference`] to pass the
    ///   running maximum over all merged frames instead.
    /// * `occlusion`  – depth-edge mask `[N, 1, H, W]` in 0..1, 1 = edge.
    pub fn forward<B: Backend>(
        &self,
        pred: &MergeOutput<B>,
        gt_merged: &Tensor<B, 4>,
        gt_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
        occlusion: &Tensor<B, 4>,
    ) -> LossTerms<B> {
        self.forward_with_reference(pred, gt_merged, gt_conf, sharpness_map(source), occlusion)
    }

    /// Compute all loss terms against an explicit per-pixel sharpness reference.
    ///
    /// `sharp_ref` (`[N, 1, H, W]`) is the per-pixel maximum of [`sharpness_map`]
    /// over the frames merged so far; the hinge penalises the prediction being
    /// less sharp than it. Other arguments as in [`Self::forward`].
    pub fn forward_with_reference<B: Backend>(
        &self,
        pred: &MergeOutput<B>,
        gt_merged: &Tensor<B, 4>,
        gt_conf: Tensor<B, 4>,
        sharp_ref: Tensor<B, 4>,
        occlusion: &Tensor<B, 4>,
    ) -> LossTerms<B> {
        self.terms(
            &pred.merged,
            &pred.conf,
            gt_merged,
            gt_conf,
            sharp_ref,
            occlusion,
        )
    }

    /// The shared body: the prediction as plain `merged` / `conf` tensors.
    fn terms<B: Backend>(
        &self,
        pred_merged: &Tensor<B, 4>,
        pred_conf: &Tensor<B, 4>,
        gt_merged: &Tensor<B, 4>,
        gt_conf: Tensor<B, 4>,
        sharp_ref: Tensor<B, 4>,
        occlusion: &Tensor<B, 4>,
    ) -> LossTerms<B> {
        let device = &occlusion.device();

        // ── Occlusion weight map ────────────────────────────────────────────
        // w ∈ [occlusion_min_weight, 1]:  1 at non-edges, min_weight at edges.
        let w = occlusion_weight_map(occlusion, self.occlusion_min_weight); // [N,1,H,W]

        // ── 1. Charbonnier reconstruction ───────────────────────────────────
        let eps = self.charbonnier_eps;
        let diff = pred_merged.clone().sub(gt_merged.clone());
        // c = sqrt(diff^2 + eps^2) — smooth L1.
        let charb_map = diff.clone().mul(diff).add_scalar(eps * eps).sqrt(); // [N,3,H,W]
        // w broadcasts [N,1,H,W] → [N,3,H,W].
        let charbonnier = charb_map.mul(w.clone()).mean();

        // ── 2. Multi-scale gradient L1 (occlusion-weighted) ─────────────────
        let gradient = multi_scale_gradient_l1(
            pred_merged,
            gt_merged,
            Some(&w),
            self.gradient_scales,
            device,
        );

        // ── 3. Sharpness retention ──────────────────────────────────────────
        // Laplacian high-pass magnitude of the prediction, [N, 1, H, W].
        let sharp_pred = sharpness_map(pred_merged.clone());

        // Penalise output being blurrier than the reference (the sharpest frame
        // merged so far): relu(sharp_ref - sharp_pred).
        let sharpness = sharp_ref
            .sub(sharp_pred)
            .clamp_min(0.0_f32)
            .mul(w.clone())
            .mean();

        // ── 4. Confidence supervision ───────────────────────────────────────
        let confidence = pred_conf.clone().sub(gt_conf).abs().mul(w).mean();

        // ── Total ───────────────────────────────────────────────────────────
        let total = charbonnier
            .clone()
            .mul_scalar(self.w_charbonnier)
            .add(gradient.clone().mul_scalar(self.w_gradient))
            .add(sharpness.clone().mul_scalar(self.w_sharpness))
            .add(confidence.clone().mul_scalar(self.w_confidence));

        LossTerms {
            total,
            charbonnier,
            gradient,
            sharpness,
            confidence,
        }
    }
}

impl<B: Backend> PairwiseFusionLoss<B> for FocusFusionLoss {
    /// The total of [`FocusFusionLoss::forward`] for a [`MergeStep`] prediction
    /// (the single-source sharpness reference).
    fn forward(
        &self,
        pred: &MergeStep<B>,
        gt_merged: &Tensor<B, 4>,
        gt_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
        occlusion: &Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        self.terms(
            &pred.merged,
            &pred.conf,
            gt_merged,
            gt_conf,
            sharpness_map(source),
            occlusion,
        )
        .total
    }
}
