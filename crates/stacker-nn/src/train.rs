//! Differentiable training primitives for all four strategies of the trainer:
//! the recurrent pairwise merge ([`rollout_loss`] / [`rollout_train`]), the
//! whole-stack merge ([`batch_loss`]), whole-stack alignment ([`align_loss`]) and
//! streaming pairwise alignment ([`fusion_align_loss`]).
//!
//! The training *driver* (optimiser loop) lives in the `z-stackr-train` binary
//! (`src/bin/train.rs`). What it needs to be testable lives in the submodules:
//! [`args`] (the command line and [`args::TrainStrategy`]), [`rng`] (the seeded
//! random source and crop sampler) and [`strategy_checkpoint`] (checkpoints and
//! resume state of the non-pairwise strategies; the pairwise strategy keeps using
//! [`crate::checkpoint`]).
//!
//! The pieces in this module are forward + loss-graph construction, the
//! per-step-update rollout [`rollout_train`], and scalar schedule helpers, so they
//! build under the default (`autodiff`) feature set and are covered by
//! `tests/train.rs` and `tests/train_strategies.rs` — this is where the tricky
//! recurrent rollout (the part that is easy to get subtly wrong) is verified.
//!
//! ## Alignment losses
//!
//! [`align_loss`] and [`fusion_align_loss`] add a small regulariser,
//! [`DELTA_REG_WEIGHT`] times the mean squared **raw** (pre-`tanh`) head outputs,
//! to the corner loss. It keeps the head from saturating deep inside the `tanh`
//! bounds, where the gradient vanishes. The matrices they compare are in the
//! normalised convention of the crop the network is shown (see
//! [`crate::data::align`]).
//!
//! ## Recurrent rollout
//!
//! Training mirrors inference: the network is folded over a focus stack one
//! pairwise merge at a time. At step `k` the model receives a *target*
//! composite, its confidence, and a *source* frame, and predicts an updated
//! composite + confidence. Two ways to provide the target:
//!
//! * **Teacher forcing** — use the dataset's ground-truth prefix composite
//!   (`MergeSample::target` / `target_conf`). Step 0 is the raw first frame with
//!   confidence 0, exactly what inference starts from.
//! * **Scheduled sampling** — feed the model's OWN previous prediction back in,
//!   to fight exposure bias (train/inference mismatch).
//!
//! The fed-back prediction is **detached**, so gradients never cross step
//! boundaries (truncated BPTT with a horizon of one step).
//!
//! ## Memory: one graph at a time
//!
//! [`rollout_train`] runs `forward -> loss -> backward -> optimiser step` for each
//! step and drops that step's autodiff graph before the next step starts, so the
//! peak graph memory is that of **one** pairwise merge regardless of how many
//! planes the scene has. ([`rollout_loss`] returns a single differentiable loss
//! for the whole rollout and therefore keeps every step's graph alive until the
//! caller's `backward`; it is for evaluation and tests, not for training.)
//!
//! ## Sharpness reference
//!
//! The sharpness hinge compares the prediction with the per-pixel maximum of
//! [`crate::loss::sharpness_map`] over every frame merged so far (the raw first
//! frame and each source up to the current step), see [`crate::loss`].

#![expect(
    clippy::cast_precision_loss,
    reason = "step, epoch and scene counts are far below 2^24 (f32) and 2^52 (f64), so the usize to float conversions are exact"
)]

use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::{ElementConversion, backend::AutodiffBackend},
};

pub mod args;
pub mod rng;
pub mod strategy_checkpoint;

use crate::{
    data::{AlignSample, BatchSample, MergeSample, PairAlignSample},
    loss::{
        FocusBatchLoss, FocusFusionLoss, LossTerms, photometric::photometric_gradient_loss,
        sharpness_map,
    },
    model::{BatchAlignNet, BatchMergeNet, FocusMergeNet, FusionAlignNet, MergeOutput},
    traits::{BatchAlignmentLoss, PairAlignmentLoss},
};

// ---------------------------------------------------------------------------
// Shared per-step forward
// ---------------------------------------------------------------------------

/// Running state threaded through a rollout.
struct RolloutState<B: Backend> {
    /// Detached prediction of the previous step `(merged, conf)`.
    prev: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    /// Per-pixel max of `sharpness_map` over the frames merged so far.
    sharp_ref: Option<Tensor<B, 4>>,
}

impl<B: Backend> RolloutState<B> {
    const fn new() -> Self {
        Self {
            prev: None,
            sharp_ref: None,
        }
    }

    /// Forward + loss for step `k`; updates the running prediction and sharpness
    /// reference. Returns the loss terms (graph attached on autodiff backends).
    fn step(
        &mut self,
        model: &FocusMergeNet<B>,
        loss: &FocusFusionLoss,
        k: usize,
        s: &MergeSample<B>,
        use_prediction: bool,
    ) -> LossTerms<B> {
        let (target, target_conf) = match (self.prev.take(), use_prediction) {
            (Some((merged, conf)), true) if k > 0 => (merged, conf),
            _ => (
                s.target.clone().unsqueeze_dim::<4>(0),
                s.target_conf.clone().unsqueeze_dim::<4>(0),
            ),
        };

        let source = s.source.clone().unsqueeze_dim::<4>(0);

        // Sharpest evidence so far: the first frame (step 0's target is the raw
        // first frame) and every source merged up to and including this one.
        let source_sharp = sharpness_map(source.clone()).detach();
        let reference = match self.sharp_ref.take() {
            Some(r) => r.max_pair(source_sharp),
            None => sharpness_map(s.target.clone().unsqueeze_dim::<4>(0))
                .detach()
                .max_pair(source_sharp),
        };
        self.sharp_ref = Some(reference.clone());

        let out: MergeOutput<B> = model.forward(target, target_conf, source);

        let gt_merged = s.gt_merged.clone().unsqueeze_dim::<4>(0);
        let gt_conf = s.gt_conf.clone().unsqueeze_dim::<4>(0);
        let occlusion = s.occlusion.clone().unsqueeze_dim::<4>(0);

        let terms = loss.forward_with_reference(&out, &gt_merged, gt_conf, reference, &occlusion);

        // Detach the prediction for the next step (truncated BPTT).
        self.prev = Some((out.merged.detach(), out.conf.detach()));
        terms
    }
}

// ---------------------------------------------------------------------------
// Rollout loss (evaluation / tests)
// ---------------------------------------------------------------------------

/// Accumulate `a + b` term-wise.
fn add_terms<B: Backend>(a: LossTerms<B>, b: LossTerms<B>) -> LossTerms<B> {
    LossTerms {
        total: a.total.add(b.total),
        charbonnier: a.charbonnier.add(b.charbonnier),
        gradient: a.gradient.add(b.gradient),
        sharpness: a.sharpness.add(b.sharpness),
        confidence: a.confidence.add(b.confidence),
    }
}

/// Divide every term by `n`.
fn div_terms<B: Backend>(t: LossTerms<B>, n: f32) -> LossTerms<B> {
    LossTerms {
        total: t.total.div_scalar(n),
        charbonnier: t.charbonnier.div_scalar(n),
        gradient: t.gradient.div_scalar(n),
        sharpness: t.sharpness.div_scalar(n),
        confidence: t.confidence.div_scalar(n),
    }
}

/// Run the recurrent rollout over one scene's steps and return the mean loss
/// (averaged over steps) as differentiable [`LossTerms`].
///
/// All steps' graphs stay alive until the caller drops or back-propagates the
/// result, so this is for evaluation and tests; **training uses
/// [`rollout_train`]**, which frees each step's graph before the next.
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
pub fn rollout_loss<B: Backend>(
    model: &FocusMergeNet<B>,
    loss: &FocusFusionLoss,
    steps: &[MergeSample<B>],
    use_prediction: &[bool],
) -> LossTerms<B> {
    assert!(!steps.is_empty(), "rollout requires at least one step");
    assert_eq!(
        steps.len(),
        use_prediction.len(),
        "use_prediction must have one flag per step"
    );

    let mut state = RolloutState::new();
    let mut acc: Option<LossTerms<B>> = None;

    for (k, s) in steps.iter().enumerate() {
        let terms = state.step(model, loss, k, s, use_prediction[k]);
        acc = Some(match acc {
            None => terms,
            Some(a) => add_terms(a, terms),
        });
    }

    // `acc` is always `Some` because `steps` is non-empty (asserted above).
    let summed = acc.expect("non-empty steps guarantees an accumulated loss");
    div_terms(summed, steps.len() as f32)
}

// ---------------------------------------------------------------------------
// Training rollout (per-step backward + optimiser step)
// ---------------------------------------------------------------------------

/// Loss values of a [`rollout_train`] call, each the mean over the rollout's steps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RolloutStats {
    /// Weighted total.
    pub total: f64,
    /// Charbonnier term.
    pub charbonnier: f64,
    /// Occlusion-weighted multi-scale gradient term.
    pub gradient: f64,
    /// Sharpness-retention term.
    pub sharpness: f64,
    /// Confidence term.
    pub confidence: f64,
    /// Optimiser updates performed (one per step).
    pub updates: usize,
}

/// Train on one scene: for every step run `forward -> loss -> backward ->
/// optimiser step`, then drop that step's graph before starting the next.
///
/// This is truncated back-propagation through the fold with a horizon of one step:
/// the fed-back prediction is detached (as in [`rollout_loss`]), so a step's
/// gradient involves only that step's forward pass, and the autodiff graph held in
/// memory is that of a single pairwise merge no matter how many planes the scene
/// has. The model is updated after every step (`steps.len()` `AdamW` updates per
/// scene, all at learning rate `lr`), so later steps already see the improved
/// weights; the optimiser (and its moments) is shared across steps and scenes.
///
/// `steps` and `use_prediction` are as in [`rollout_loss`]. Returns the updated
/// model and the mean loss values over the steps.
///
/// # Panics
///
/// Panics if `steps` is empty or if `steps.len() != use_prediction.len()`.
pub fn rollout_train<B, O>(
    mut model: FocusMergeNet<B>,
    optim: &mut O,
    lr: f64,
    loss: &FocusFusionLoss,
    steps: &[MergeSample<B>],
    use_prediction: &[bool],
) -> (FocusMergeNet<B>, RolloutStats)
where
    B: AutodiffBackend,
    O: Optimizer<FocusMergeNet<B>, B>,
{
    assert!(!steps.is_empty(), "rollout requires at least one step");
    assert_eq!(
        steps.len(),
        use_prediction.len(),
        "use_prediction must have one flag per step"
    );

    let mut state = RolloutState::new();
    let mut sum = [0.0_f64; 5];

    for (k, s) in steps.iter().enumerate() {
        let LossTerms {
            total,
            charbonnier,
            gradient,
            sharpness,
            confidence,
        } = state.step(&model, loss, k, s, use_prediction[k]);

        let scalar = |t: &Tensor<B, 1>| -> f64 { t.clone().into_scalar().elem() };
        sum[0] += scalar(&total);
        sum[1] += scalar(&charbonnier);
        sum[2] += scalar(&gradient);
        sum[3] += scalar(&sharpness);
        sum[4] += scalar(&confidence);

        // Release the logging terms, back-propagate, then drop the total: this
        // step's graph is gone before the next step builds its own.
        drop((charbonnier, gradient, sharpness, confidence));
        let grads = GradientsParams::from_grads(total.backward(), &model);
        drop(total);
        model = optim.step(lr, model, grads);
    }

    let n = steps.len() as f64;
    (
        model,
        RolloutStats {
            total: sum[0] / n,
            charbonnier: sum[1] / n,
            gradient: sum[2] / n,
            sharpness: sum[3] / n,
            confidence: sum[4] / n,
            updates: steps.len(),
        },
    )
}

// ---------------------------------------------------------------------------
// Batch fusion loss
// ---------------------------------------------------------------------------

/// Run the batch fusion model on the whole stack of a scene and return the loss.
///
/// Adds the batch dimension (`N = 1`) that [`BatchSample`]'s tensors lack, runs
/// [`BatchMergeNet::forward_with_alpha`] (the merged image and the per-frame gate,
/// which the gate-supervision term needs) and scores it with
/// [`FocusBatchLoss::forward_full`]: Charbonnier, multi-scale gradient, sharpness
/// retention against the sharpest input and gate supervision, each weighted down
/// near depth edges.
#[must_use]
pub fn batch_loss<B: Backend>(
    model: &BatchMergeNet<B>,
    loss: &FocusBatchLoss,
    sample: &BatchSample<B>,
) -> Tensor<B, 1> {
    let stack = sample.stack.clone().unsqueeze_dim::<5>(0); // [1, S, 3, H, W]
    let gt_merged = sample.gt_merged.clone().unsqueeze_dim::<4>(0); // [1, 3, H, W]
    let masks = sample.masks.clone().unsqueeze_dim::<5>(0); // [1, S, 1, H, W]
    let occlusion = sample.occlusion.clone().unsqueeze_dim::<4>(0); // [1, 1, H, W]

    let (pred_merged, alpha) = model.forward_with_alpha(stack.clone());
    loss.forward_full(&pred_merged, &alpha, &gt_merged, &stack, &masks, &occlusion)
}

// ---------------------------------------------------------------------------
// Alignment losses
// ---------------------------------------------------------------------------

/// Weight of the raw-output regulariser added to the corner loss; a training
/// hyper-parameter tied to the ranges of the simulated misalignment.
pub const DELTA_REG_WEIGHT: f32 = 1e-4;

/// `corner + DELTA_REG_WEIGHT * mean(raw^2)`.
fn with_delta_regulariser<B: Backend>(corner: Tensor<B, 1>, raw: Tensor<B, 3>) -> Tensor<B, 1> {
    let reg = raw.clone().mul(raw).mean();
    corner.add(reg.mul_scalar(DELTA_REG_WEIGHT))
}

/// Run the whole-stack alignment model on a scene and return its loss:
/// `corner_loss + DELTA_REG_WEIGHT * mean(raw_params^2)`.
///
/// Adds the batch dimension (`N = 1`) that [`AlignSample`]'s tensors lack, runs
/// [`BatchAlignNet::forward_with_params`] and scores the predicted matrices against
/// `sample.gt_matrices` with `loss`. Both are in the normalised convention of the
/// crop the stack was cut to, so no coordinate conversion happens here.
#[must_use]
pub fn align_loss<B: Backend, L: BatchAlignmentLoss<B>>(
    model: &BatchAlignNet<B>,
    loss: &L,
    sample: &AlignSample<B>,
) -> Tensor<B, 1> {
    let stack = sample.stack.clone().unsqueeze_dim::<5>(0); // [1, S, 3, H, W]
    let gt_matrices = sample.gt_matrices.clone().unsqueeze_dim::<4>(0); // [1, S, 3, 3]

    let (pred_matrices, raw_params) = model.forward_with_params(stack);
    with_delta_regulariser(loss.forward(pred_matrices, gt_matrices), raw_params)
}

/// Run the pairwise alignment model on one reference/frame pair and return the
/// same composition as [`align_loss`] for its `[N, 3, 3]` output.
///
/// Adds the batch dimension (`N = 1`) that [`PairAlignSample`]'s tensors lack and
/// runs [`FusionAlignNet::forward_with_params`].
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
    // The raw outputs are [N, 6]; the regulariser only needs their mean square.
    let reg = raw_params.clone().mul(raw_params).mean();
    loss.forward(pred_matrix, gt_matrix)
        .add(reg.mul_scalar(DELTA_REG_WEIGHT))
}

/// The photometric fine-tune term for one real, unlabelled stack: the mean of
/// [`photometric_gradient_loss`] over the non-reference frames, each warped by the
/// matrix the model predicts for it.
///
/// `stack` is `[S, 3, H, W]` (frame 0 is the reference). Returns `None` for a
/// single frame, which has nothing to score. The trainer adds
/// `--photometric-weight` times this to the supervised loss in the last quarter of
/// the epochs.
///
/// **Unvalidated:** this phase has only been checked for shapes and finiteness; it
/// has never been shown to improve a trained model (that needs a training run on
/// real stacks, see the README), which is why it is off unless `--real-data` is
/// given.
#[must_use]
pub fn photometric_finetune_term<B: Backend>(
    model: &BatchAlignNet<B>,
    stack: &Tensor<B, 4>,
) -> Option<Tensor<B, 1>> {
    let [s, c, h, w] = stack.dims();
    if s < 2 {
        return None;
    }
    let (matrices, _raw) = model.forward_with_params(stack.clone().unsqueeze_dim::<5>(0));
    let matrices = matrices.squeeze::<3>(0); // [S, 3, 3]

    let reference = stack.clone().slice([0..1, 0..c, 0..h, 0..w]);
    let mut sum: Option<Tensor<B, 1>> = None;
    for i in 1..s {
        let frame = stack.clone().slice([i..i + 1, 0..c, 0..h, 0..w]);
        let matrix = matrices.clone().slice([i..i + 1, 0..3, 0..3]);
        let term = photometric_gradient_loss(frame, reference.clone(), matrix);
        sum = Some(match sum {
            None => term,
            Some(acc) => acc.add(term),
        });
    }
    sum.map(|total| total.div_scalar((s - 1) as f32))
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
    let cosine = 1.0_f64.midpoint((std::f64::consts::PI * progress).cos());
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
