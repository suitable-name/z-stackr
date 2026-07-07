//! Differentiable training primitives for all four of the crate's
//! strategies: [`rollout_loss`] (pairwise fusion), [`batch_loss`] (batch
//! fusion), [`align_loss`] (batch alignment), and [`fusion_align_loss`]
//! (pairwise alignment, training [`FusionAlignNet`] on one reference/frame
//! pair at a time — see `docs/fusionalign-design.md`).
//!
//! The training *driver* (optimiser loop, checkpointing, CLI) lives in the
//! `stacker-nn-train` binary (`src/bin/train.rs`), which requires the `train`
//! feature because `burn::optim` is only available there.
//!
//! The pieces in this module are pure forward + loss-graph construction plus
//! scalar schedule helpers, so they build under the default (`autodiff`)
//! feature set and are exercised by CI — this is where the tricky recurrent
//! rollout (the part that is easy to get subtly wrong) is verified.
//!
//! ## Recurrent rollout
//!
//! Training mirrors inference: the network is folded over a focus stack one
//! pairwise merge at a time. At step `k` the model receives a *target*
//! composite, its confidence, and a *source* frame, and predicts an updated
//! composite + confidence. Two ways to provide the target:
//!
//! * **Teacher forcing** — use the dataset's ground-truth prefix composite
//!   (`MergeSample::target` / `target_conf`).
//! * **Scheduled sampling** — feed the model's OWN previous prediction back in,
//!   to fight exposure bias (train/inference mismatch).
//!
//! The fed-back prediction is **detached** (truncated BPTT): every step still
//! contributes a supervised loss that back-propagates into the model, but
//! gradients do not flow across step boundaries, keeping memory bounded.

use burn::prelude::*;

use crate::{
    data::{AlignSample, BatchSample, MergeSample, PairAlignSample},
    loss::FocusBatchLoss,
    model::{BatchAlignNet, BatchMergeNet, FocusMergeNet, FusionAlignNet},
    traits::{BatchAlignmentLoss, MergeStep, PairAlignmentLoss, PairwiseFusionLoss},
};

// ---------------------------------------------------------------------------
// Rollout loss
// ---------------------------------------------------------------------------

/// Run the recurrent rollout over one scene's steps and return the mean loss
/// (averaged over steps).
///
/// * `steps` — the per-step samples for ONE scene, in merge order. Every sample
///   must share the same spatial size (use a consistent crop across the scene).
/// * `use_prediction` — per-step flag (same length as `steps`); when `true` and
///   `k > 0`, step `k` is fed the model's own (detached) previous prediction
///   instead of the ground-truth target. Element `0` is always ignored (the
///   first step seeds from `steps[0].target`).
///
/// All `MergeSample` tensors are rank-3 `[C,H,W]`; they are unsqueezed to a
/// batch of 1 internally.
///
/// # Panics
///
/// Panics if `steps` is empty or if `steps.len() != use_prediction.len()`.
#[must_use]
pub fn rollout_loss<B: Backend, L: PairwiseFusionLoss<B>>(
    model: &FocusMergeNet<B>,
    loss: &L,
    steps: &[MergeSample<B>],
    use_prediction: &[bool],
) -> Tensor<B, 1> {
    assert!(!steps.is_empty(), "rollout requires at least one step");
    assert_eq!(
        steps.len(),
        use_prediction.len(),
        "use_prediction must have one flag per step"
    );

    // Running (detached) prediction threaded across steps.
    let mut prev: Option<(Tensor<B, 4>, Tensor<B, 4>)> = None;
    let mut acc: Option<Tensor<B, 1>> = None;

    for (k, s) in steps.iter().enumerate() {
        let (target, target_conf) = match (prev.as_ref(), use_prediction[k]) {
            (Some((merged, conf)), true) if k > 0 => (merged.clone(), conf.clone()),
            _ => (
                s.target.clone().unsqueeze_dim::<4>(0),
                s.target_conf.clone().unsqueeze_dim::<4>(0),
            ),
        };

        let source = s.source.clone().unsqueeze_dim::<4>(0);
        let out = model.forward(target, target_conf, source.clone());

        let gt_merged = s.gt_merged.clone().unsqueeze_dim::<4>(0);
        let gt_conf = s.gt_conf.clone().unsqueeze_dim::<4>(0);
        let occlusion = s.occlusion.clone().unsqueeze_dim::<4>(0);

        let step = MergeStep {
            merged: out.merged.clone(),
            conf: out.conf.clone(),
        };
        let terms = loss.forward(&step, &gt_merged, gt_conf, source, &occlusion);

        // Detach prediction for the next step (truncated BPTT).
        prev = Some((out.merged.detach(), out.conf.detach()));

        acc = Some(match acc {
            None => terms,
            Some(a) => a.add(terms),
        });
    }

    // `acc` is always `Some` because `steps` is non-empty (asserted above).
    let summed = acc.expect("non-empty steps guarantees an accumulated loss");
    summed.div_scalar(steps.len() as f32)
}

// ---------------------------------------------------------------------------
// Batch loss
// ---------------------------------------------------------------------------

/// Run the batch fusion model on a full scene stack and return the loss.
///
/// Adds the batch dimension (`N = 1`) [`BatchSample`]'s tensors lack, runs
/// [`BatchMergeNet::forward_with_alpha`] (rather than going through the
/// [`crate::traits::FusionModel`] trait, which only returns the merged
/// image — the same pattern [`align_loss`] uses for
/// [`BatchAlignNet::forward_with_params`]), and scores the result against
/// `sample.gt_merged`/`sample.masks`/`sample.occlusion` with
/// [`FocusBatchLoss::forward_full`], which exercises all four of that loss's
/// terms (Charbonnier, multi-scale gradient, sharpness retention, gate
/// supervision — see [`crate::loss`]'s module docs).
#[must_use]
pub fn batch_loss<B: Backend>(
    model: &BatchMergeNet<B>,
    loss: &FocusBatchLoss,
    sample: &BatchSample<B>,
) -> Tensor<B, 1> {
    // Add batch dimension (N=1)
    let stack = sample.stack.clone().unsqueeze_dim::<5>(0); // [1, S, 3, H, W]
    let gt_merged = sample.gt_merged.clone().unsqueeze_dim::<4>(0); // [1, 3, H, W]
    let masks = sample.masks.clone().unsqueeze_dim::<5>(0); // [1, S, 1, H, W]
    let occlusion = sample.occlusion.clone().unsqueeze_dim::<4>(0); // [1, 1, H, W]

    let (pred_merged, alpha) = model.forward_with_alpha(stack.clone());
    loss.forward_full(&pred_merged, &alpha, &gt_merged, &stack, &masks, &occlusion)
}

// ---------------------------------------------------------------------------
// Alignment loss
// ---------------------------------------------------------------------------

/// Weight of the delta regulariser term added to the corner-alignment loss —
/// `docs/batchalign-v2-design.md` §5.1. Not tunable independently of the
/// training-data ranges in that document's §4.4; changing it is a training
/// hyperparameter decision, not a bug fix.
const DELTA_REG_WEIGHT: f32 = 1e-4;

/// Run the alignment model on a full scene stack and return the total loss:
/// <code>corner_loss + [DELTA_REG_WEIGHT] * mean(raw_params^2)</code>
/// (`docs/batchalign-v2-design.md` §5.1).
///
/// Adds the batch dimension (`N = 1`) [`AlignSample`]'s tensors lack, runs
/// [`BatchAlignNet::forward_with_params`] (rather than going through the
/// [`crate::traits::BatchAlignmentModel`] trait, which only returns the
/// matrices), and scores the predicted matrices against `sample.gt_matrices`
/// with `loss`
/// (both already in the crate-wide normalized `[-1, 1]²` convention — see
/// [`crate::bridge::align_planar`]'s docs — so no coordinate conversion
/// happens here). The delta regulariser penalises the six *raw* (pre-tanh)
/// head outputs, discouraging the head from saturating deep into the tanh
/// bounds (§3.4) where gradients vanish.
#[must_use]
pub fn align_loss<B: Backend, L: BatchAlignmentLoss<B>>(
    model: &BatchAlignNet<B>,
    loss: &L,
    sample: &AlignSample<B>,
) -> Tensor<B, 1> {
    let stack = sample.stack.clone().unsqueeze_dim::<5>(0); // [1, S, 3, H, W]
    let gt_matrices = sample.gt_matrices.clone().unsqueeze_dim::<4>(0); // [1, S, 3, 3]

    let (pred_matrices, raw_params) = model.forward_with_params(stack);
    let corner = loss.forward(pred_matrices, gt_matrices);
    let reg = raw_params.powf_scalar(2.0).mean();
    corner.add(reg.mul_scalar(DELTA_REG_WEIGHT))
}

/// Run the pairwise alignment model on one reference/frame pair and return
/// the total loss: <code>corner_loss + [DELTA_REG_WEIGHT] * mean(raw_params^2)</code>
/// — the exact same composition as [`align_loss`], just for
/// [`FusionAlignNet`]'s `[N,3,3]` (no `S`) output shape. See
/// `docs/fusionalign-design.md` for why the same regulariser weight is
/// reused unchanged rather than re-tuned: `FusionAlignNet`'s head is the same
/// bounded 6-number parameterisation (`bounded_affine_from_raw6`) as
/// `BatchAlignNet`'s, so the same tanh-saturation concern (and the same
/// fix) applies identically.
///
/// Adds the batch dimension (`N = 1`) [`PairAlignSample`]'s tensors lack, runs
/// [`FusionAlignNet::forward_with_params`] (rather than going through the
/// [`crate::traits::PairAlignmentModel`] trait, which only returns the
/// matrix), and scores the predicted matrix against `sample.gt_matrix` with
/// `loss` (already in the crate-wide normalized `[-1, 1]²` convention — see
/// [`crate::bridge::align_planar_pairwise`]'s docs — so no coordinate
/// conversion happens here).
#[must_use]
pub fn fusion_align_loss<B: Backend, L: PairAlignmentLoss<B>>(
    model: &FusionAlignNet<B>,
    loss: &L,
    sample: &PairAlignSample<B>,
) -> Tensor<B, 1> {
    let reference = sample.reference.clone().unsqueeze_dim::<4>(0); // [1, 3, H, W]
    let frame = sample.frame.clone().unsqueeze_dim::<4>(0); // [1, 3, H, W]
    let gt_matrix = sample.gt_matrix.clone().unsqueeze_dim::<3>(0); // [1, 3, 3]

    let (pred_matrix, raw_params) = model.forward_with_params(reference, frame);
    let corner = loss.forward(pred_matrix, gt_matrix);
    let reg = raw_params.powf_scalar(2.0).mean();
    corner.add(reg.mul_scalar(DELTA_REG_WEIGHT))
}

// ---------------------------------------------------------------------------
// Schedule helpers (pure scalar math — used by the training driver)
// ---------------------------------------------------------------------------

/// Cosine-decayed learning rate with linear warm-up.
///
/// `step` is the current global optimiser step in `0..total_steps`. The rate
/// ramps linearly from ~0 to `base_lr` over the first `warmup` steps, then
/// follows a half-cosine down to 0 at `total_steps`.
#[must_use]
pub fn cosine_lr(base_lr: f64, step: usize, total_steps: usize, warmup: usize) -> f64 {
    if total_steps == 0 {
        return base_lr;
    }
    if step < warmup {
        return base_lr * (step as f64 + 1.0) / (warmup.max(1) as f64);
    }
    let denom = (total_steps - warmup).max(1) as f64;
    let progress = ((step - warmup) as f64 / denom).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    base_lr * cosine
}

/// Probability of using the model's own prediction (scheduled sampling) at a
/// given `epoch`, ramping linearly from 0 (pure teacher forcing) to `max_prob`
/// at the final epoch.
#[must_use]
pub fn scheduled_sampling_prob(epoch: usize, total_epochs: usize, max_prob: f64) -> f64 {
    if total_epochs <= 1 {
        return 0.0;
    }
    let p = max_prob * (epoch as f64) / ((total_epochs - 1) as f64);
    p.clamp(0.0, max_prob)
}
