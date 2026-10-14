//! The small trait layer the generic inference code is written against.
//!
//! * [`FusionModel`] — a focus-merge network, either streaming pairwise
//!   ([`FusionStrategy::Pairwise`], [`crate::model::FocusMergeNet`]) or whole-stack
//!   ([`FusionStrategy::Batch`], [`crate::model::BatchMergeNet`]).
//! * [`BatchAlignmentModel`] / [`PairAlignmentModel`] — an alignment network that
//!   registers a whole stack at once ([`crate::model::BatchAlignNet`]) or one
//!   frame against a reference per call ([`crate::model::FusionAlignNet`]).
//! * The matching loss traits ([`PairwiseFusionLoss`], [`BatchFusionLoss`],
//!   [`BatchAlignmentLoss`], [`PairAlignmentLoss`]).
//!
//! The three built-in network families are the only implementors the crate
//! itself uses; the runtime dispatches between them with the enums in
//! [`crate::runtime`] rather than with trait objects, so there is no dynamic
//! dispatch on the hot path. The traits exist so [`crate::infer`] and
//! [`crate::bridge`] are written once for both fusion strategies.
//!
//! ## Tensor conventions
//!
//! * Images are `[N, 3, H, W]` (or `[N, S, 3, H, W]` for a stack), f32,
//!   **linear-light RGB**, nominally `0..1`. Never gamma-encoded: the bridge does
//!   the `YCbCr` ⇄ RGB conversion on either side of the model boundary through
//!   `stacker_core::color`.
//! * Confidence is `[N, 1, H, W]` in `0..1` (`0` = no information yet).
//! * Alignment matrices are `[N, 3, 3]` / `[N, S, 3, 3]`, in the normalised
//!   `[-1, 1]²` coordinates of the image the network saw, and map **reference
//!   coordinates to source (frame) coordinates**, the convention of the whole
//!   pipeline (`p_frame = M · p_reference`). See [`crate::bridge`] for the exact
//!   normalisation and for converting to full-resolution pixel matrices.

use burn::prelude::*;

/// Output of one [`FusionModel::merge`] step: the updated composite and its
/// confidence channel, same shapes as the `target` / `target_conf` inputs.
///
/// Intentionally does not carry [`crate::model::MergeOutput`]'s `alpha` map: that
/// is an implementation detail of the pairwise network's gating head.
#[derive(Debug)]
pub struct MergeStep<B: Backend> {
    /// Updated merged RGB composite — `[N, 3, H, W]`, linear light.
    ///
    /// Values may transiently leave `0..1`; callers must not assume a clamp.
    pub merged: Tensor<B, 4>,
    /// Updated per-pixel confidence — `[N, 1, H, W]` in `0..1`.
    pub conf: Tensor<B, 4>,
}

/// Execution strategy of a [`FusionModel`], selecting which of
/// [`FusionModel::merge`] / [`FusionModel::fuse_batch`] the generic inference
/// machinery calls.
///
/// Both variants must be size-preserving and free of global pooling, so tiles
/// processed independently blend back together without seams;
/// [`FusionModel::receptive_field`] declares how far context travels, which sets
/// the tile overlap (`2·R`, see [`crate::infer::TileConfig::for_model`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Streaming pairwise accumulation, low memory: one `source` frame is folded
    /// into the running composite at a time via [`FusionModel::merge`].
    Pairwise,
    /// The entire stack at once via [`FusionModel::fuse_batch`]. Memory scales
    /// with `S · tile_h · tile_w`, because one tile of every frame is resident
    /// at a time.
    Batch,
}

/// A focus-merge model implementing one of the two [`FusionStrategy`] variants.
///
/// Implement only the method matching [`Self::strategy`]; the other keeps its
/// default body, which the generic inference code never calls because it
/// dispatches on [`Self::strategy`] first.
pub trait FusionModel<B: Backend> {
    /// Whether this model runs pairwise recursively or in a single batch.
    fn strategy(&self) -> FusionStrategy;

    /// Run one pairwise-merge step. Only called for [`FusionStrategy::Pairwise`]
    /// models.
    ///
    /// * `target`      – current composite `[N, 3, H, W]`, linear 0..1.
    /// * `target_conf` – per-pixel confidence of `target`, `[N, 1, H, W]`.
    /// * `source`      – new source frame `[N, 3, H, W]`, linear 0..1.
    ///
    /// # Panics
    ///
    /// The default implementation always panics; override it when
    /// [`Self::strategy`] returns [`FusionStrategy::Pairwise`].
    fn merge(
        &self,
        _target: Tensor<B, 4>,
        _target_conf: Tensor<B, 4>,
        _source: Tensor<B, 4>,
    ) -> MergeStep<B> {
        unimplemented!("this model does not support pairwise fusion")
    }

    /// Fuse an entire stack `[N, S, 3, H, W]` into `[N, 3, H, W]`. Only called for
    /// [`FusionStrategy::Batch`] models.
    ///
    /// # Panics
    ///
    /// The default implementation always panics; override it when
    /// [`Self::strategy`] returns [`FusionStrategy::Batch`].
    fn fuse_batch(&self, _stack: Tensor<B, 5>) -> Tensor<B, 4> {
        unimplemented!("this model does not support batch fusion")
    }

    /// Receptive-field radius in pixels: how far an output pixel's value can
    /// depend on an input pixel along each axis. A purely per-pixel model
    /// returns `0`.
    fn receptive_field(&self) -> usize;
}

/// A pairwise focus-merge loss.
pub trait PairwiseFusionLoss<B: Backend> {
    /// Compute the total scalar loss.
    ///
    /// * `pred`      – network predictions (merged, conf).
    /// * `gt_merged` – ground-truth composite `[N, 3, H, W]`.
    /// * `gt_conf`   – ground-truth confidence `[N, 1, H, W]`.
    /// * `source`    – source frame `[N, 3, H, W]` (sharpness reference).
    /// * `occlusion` – depth-edge mask `[N, 1, H, W]` in 0..1, 1 = edge.
    fn forward(
        &self,
        pred: &MergeStep<B>,
        gt_merged: &Tensor<B, 4>,
        gt_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
        occlusion: &Tensor<B, 4>,
    ) -> Tensor<B, 1>;
}

/// A batched (whole-stack) focus-merge loss.
pub trait BatchFusionLoss<B: Backend> {
    /// Reconstruction loss between the predicted and ground-truth composites,
    /// both `[N, 3, H, W]`. The sharpness and gate terms need the stack, masks
    /// and gate, which only the concrete loss type's `forward_full` takes.
    fn forward(&self, pred_merged: &Tensor<B, 4>, gt_merged: &Tensor<B, 4>) -> Tensor<B, 1>;
}

/// A whole-stack alignment model that predicts one affine matrix per frame.
pub trait BatchAlignmentModel<B: Backend> {
    /// Predict alignment matrices for a batch of stacks.
    ///
    /// * `stack` – `[N, S, 3, H, W]`.
    ///
    /// Returns the matrices `[N, S, 3, 3]` (reference → frame, normalised
    /// coordinates; frame 0 is exactly the identity).
    fn align_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4>;
}

/// A loss for [`BatchAlignmentModel`]s.
pub trait BatchAlignmentLoss<B: Backend> {
    /// Alignment loss between predicted and ground-truth matrices, both
    /// `[N, S, 3, 3]`.
    fn forward(&self, pred_matrices: Tensor<B, 4>, gt_matrices: Tensor<B, 4>) -> Tensor<B, 1>;
}

/// A pairwise alignment model: one affine matrix registering a `frame` against a
/// `reference`, called once per frame (streaming, O(1) memory in the stack size).
pub trait PairAlignmentModel<B: Backend> {
    /// Predict the matrix registering `frame` to `reference`, both
    /// `[N, 3, H, W]`. Returns `[N, 3, 3]` in the same convention as
    /// [`BatchAlignmentModel::align_batch`].
    fn align_pair(&self, reference: Tensor<B, 4>, frame: Tensor<B, 4>) -> Tensor<B, 3>;
}

/// A loss for [`PairAlignmentModel`]s.
pub trait PairAlignmentLoss<B: Backend> {
    /// Alignment loss between predicted and ground-truth matrices, both
    /// `[N, 3, 3]`.
    fn forward(&self, pred_matrix: Tensor<B, 3>, gt_matrix: Tensor<B, 3>) -> Tensor<B, 1>;
}
