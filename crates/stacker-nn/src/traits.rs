//! Extensibility traits: plug a third-party architecture into this crate's
//! tiled inference, feathered blending, planar bridge, and recurrent fold.
//!
//! This crate supports three fusion **strategies**, each with its own model
//! trait, loss trait, and (for the two fusion strategies) a shared
//! [`FusionModel`] surface used by the generic inference machinery:
//!
//! * **Pairwise** ([`FusionStrategy::Pairwise`]) — [`FusionModel::merge`]
//!   folds one `source` frame into a running composite per call. The
//!   built-in implementation is [`crate::model::FocusMergeNet`]; its loss
//!   trait is [`PairwiseFusionLoss`] (built-in: [`crate::loss::FocusFusionLoss`]).
//! * **Batch** ([`FusionStrategy::Batch`]) — [`FusionModel::fuse_batch`]
//!   ingests the entire stack at once. The built-in implementation is
//!   [`crate::model::BatchMergeNet`]; its loss trait is [`BatchFusionLoss`]
//!   (built-in: [`crate::loss::FocusBatchLoss`]).
//! * **Alignment (batch)** — a separate family, not a [`FusionModel`]
//!   variant, that predicts per-frame affine registration matrices instead of
//!   a fused image. Implement [`BatchAlignmentModel`] (built-in:
//!   [`crate::model::BatchAlignNet`]); its loss trait is
//!   [`BatchAlignmentLoss`] (built-in: [`crate::loss::CornerAlignmentLoss`]).
//! * **Alignment (pairwise)** — the streaming sibling of batch alignment:
//!   instead of ingesting the whole stack at once, [`PairAlignmentModel`]
//!   registers one `frame` against a `reference` per call, so the caller
//!   loops once per frame with O(1) memory in stack size rather than holding
//!   every frame's features resident simultaneously. The built-in
//!   implementation is [`crate::model::FusionAlignNet`]; its loss trait is
//!   [`PairAlignmentLoss`] (built-in:
//!   [`crate::loss::PairCornerAlignmentLoss`]). See
//!   `docs/fusionalign-design.md` for the motivation and how it relates to
//!   [`BatchAlignmentModel`].
//!
//! Nothing in [`crate::infer`] or [`crate::bridge`] is hard-wired to the
//! built-in types beyond their trait impls. A third party can define their
//! own `Module` type, implement the relevant trait(s) for it, and reuse:
//!
//! * [`crate::infer::fuse_stack`] / [`crate::infer::fuse_stack_with`] — the
//!   streaming recurrent fold (pairwise) or tiled whole-stack fusion (batch)
//!   over a focus stack, dispatching on [`FusionModel::strategy`].
//! * [`crate::bridge::fuse_planar`] / [`crate::bridge::fuse_planar_with_progress`]
//!   — the same, at the level of the pipeline's [`stacker_core::image::PlanarImage`]
//!   (handles the gamma ⇄ linear and `YCbCr` ⇄ `RGB` conversions).
//! * [`crate::bridge::align_planar`] — run a [`BatchAlignmentModel`] over a
//!   stack of [`stacker_core::image::PlanarImage`], returning pixel-space
//!   affine matrices (see that function's docs for the coordinate contract).
//! * [`crate::bridge::align_planar_pairwise`] — the same, for a
//!   [`PairAlignmentModel`]: registers each frame against the stack's
//!   reference frame in a streaming loop, sharing the exact same pixel-space
//!   coordinate contract as [`crate::bridge::align_planar`].
//! * [`crate::infer::TileConfig::for_model`] — derive a seam-safe tile
//!   overlap from the model's own [`FusionModel::receptive_field`].
//!
//! Weight loading, model discovery ([`crate::discovery`]) and the concrete
//! [`crate::runtime::LoadedModel`]/[`crate::runtime::LoadedAlignModel`]
//! dispatchers remain specific to the built-in architectures — see the
//! "Extending with your own architecture" section of the crate docs
//! ([`crate`]) for how a third party is expected to load their own weights
//! and drive inference directly.

use burn::prelude::*;

/// Output of one [`FusionModel::merge`] step: the updated composite
/// and its confidence channel, same shapes as the `target`/`target_conf`
/// inputs.
///
/// This is the public contract's output type. It intentionally does **not**
/// carry [`crate::model::MergeOutput`]'s `alpha` (soft selection map) field,
/// since that is an implementation detail of `FocusMergeNet`'s
/// gating-head architecture that a third-party model has no obligation to
/// produce. Implementations that *do* have an analogous internal map are
/// free to expose it their own way (e.g. a second method, or their own
/// richer output type wrapped behind [`FusionModel`]) — the trait
/// only requires what the tiled fold and blend actually consume.
#[derive(Debug)]
pub struct MergeStep<B: Backend> {
    /// Updated merged RGB composite — `[N, 3, H, W]`, f32, linear-light,
    /// nominally in `0..1`.
    ///
    /// Values may transiently leave that range before the next step's blend
    /// renormalises them; callers should not assume a hard clamp.
    pub merged: Tensor<B, 4>,
    /// Updated per-pixel confidence — `[N, 1, H, W]`, f32, in `0..1`. `0`
    /// means "no information yet" (e.g. the seed frame before any merge),
    /// `1` means fully confident.
    ///
    /// Confidence is what makes the recurrent accumulation order-robust: it
    /// is fed back as `target_conf` on the next step so the model can weigh
    /// a well-established composite against a fresh source frame.
    pub conf: Tensor<B, 4>,
}

/// Execution strategy of a [`FusionModel`], selecting which of
/// [`FusionModel::merge`] / [`FusionModel::fuse_batch`] the generic inference
/// machinery in [`crate::infer`] calls.
///
/// ## Shape contract (applies to both variants' methods)
///
/// * `target`, `merged` — `[N, 3, H, W]`, f32, **linear-light RGB**, nominally
///   `0..1`. Never gamma-encoded: the crate's [`crate::bridge`] module
///   performs the `YCbCr` ⇄ `RGB` and gamma ⇄ linear conversions on either
///   side of the model boundary, so an implementation only ever sees/returns
///   linear RGB tensors.
/// * `target_conf`, `conf` — `[N, 1, H, W]`, f32, in `0..1`. `0` = no
///   information (e.g. the seed frame), `1` = fully confident. The value is
///   threaded back in as `target_conf` on the following recurrent step, so
///   its scale must be self-consistent across steps.
/// * `source` — `[N, 3, H, W]`, f32, linear-light RGB, same convention as
///   `target`.
/// * All tensors in a single call share `N`, `H`, `W` (checked by the
///   caller before tiling); an implementation does not need to re-validate
///   shapes, but per [`crate::infer`]'s Same-padding requirement it **must**
///   preserve `H × W` — no down/up-sampling, no global pooling — so that
///   tiles processed independently blend back together without seams.
///
/// ## Receptive field and tiling
///
/// [`FusionModel::receptive_field`] declares the radius (in pixels)
/// beyond which a pixel's output cannot depend on the input, i.e. how far
/// context can propagate through the network's convolutions. The tiled
/// inference in [`crate::infer`] slices the image into overlapping tiles and
/// blends them back together with a feathered (raised-cosine) window; for
/// that blend to be seam-free the tile overlap **must** be at least the
/// receptive field, otherwise a pixel near a tile border is computed from
/// incomplete context and the blend becomes visible. Use
/// [`crate::infer::TileConfig::for_model`] to derive a safe overlap
/// automatically instead of hard-coding one.
///
/// The same applies to [`FusionModel::fuse_batch`] for [`FusionStrategy::Batch`]
/// models: [`crate::infer::fuse_batch_tiled`](crate::infer) (invoked internally
/// by [`crate::infer::fuse_stack`]) tiles the whole stack the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Streaming pairwise accumulation, low memory footprint: one `source`
    /// frame is folded into the running composite at a time via
    /// [`FusionModel::merge`].
    Pairwise,
    /// Processes the entire stack at once via [`FusionModel::fuse_batch`].
    /// High memory footprint — VRAM scales linearly with stack size, since a
    /// [`crate::infer::TileConfig`] tile's worth of every frame in the stack
    /// must be resident simultaneously (see that type's docs).
    Batch,
}

/// A focus-merge model implementing one of the two [`FusionStrategy`] variants.
///
/// Either streaming pairwise ([`Self::merge`]) or whole-stack batch
/// ([`Self::fuse_batch`]). Implement only the method matching the strategy
/// [`Self::strategy`] reports — the other keeps its default
/// `unimplemented!()` body, which the generic inference machinery in
/// [`crate::infer`] never calls because it dispatches on [`Self::strategy`]
/// first. See the module docs for the full extensibility story and the
/// built-in implementations ([`crate::model::FocusMergeNet`],
/// [`crate::model::BatchMergeNet`]).
pub trait FusionModel<B: Backend> {
    /// Indicates whether this model runs pairwise recursively or in a single
    /// batch.
    ///
    /// The generic inference entry points ([`crate::infer::fuse_stack`],
    /// [`crate::bridge::fuse_planar`]) read this to decide whether to call
    /// [`Self::merge`] in a streaming fold or [`Self::fuse_batch`] once per
    /// tile.
    fn strategy(&self) -> FusionStrategy;

    /// Run one pairwise-merge step. Only called for
    /// [`FusionStrategy::Pairwise`] models; the default body panics.
    ///
    /// * `target`      – current composite `[N, 3, H, W]`, f32, linear 0..1.
    /// * `target_conf` – per-pixel confidence of `target`, `[N, 1, H, W]`.
    /// * `source`      – new source frame `[N, 3, H, W]`, f32, linear 0..1.
    ///
    /// Returns the updated composite and confidence, same shapes as
    /// `target`/`target_conf`.
    ///
    /// # Panics
    ///
    /// The default implementation always panics with `unimplemented!()`;
    /// override it if [`Self::strategy`] returns [`FusionStrategy::Pairwise`].
    fn merge(
        &self,
        _target: Tensor<B, 4>,
        _target_conf: Tensor<B, 4>,
        _source: Tensor<B, 4>,
    ) -> MergeStep<B> {
        unimplemented!("This model does not support pairwise fusion")
    }

    /// Fuse an entire stack of frames at once. Only called for
    /// [`FusionStrategy::Batch`] models; the default body panics.
    ///
    /// * `stack` - stack of frames `[N, S, 3, H, W]`, f32, linear 0..1.
    ///
    /// Returns the merged composite `[N, 3, H, W]`.
    ///
    /// # Panics
    ///
    /// The default implementation always panics with `unimplemented!()`;
    /// override it if [`Self::strategy`] returns [`FusionStrategy::Batch`].
    fn fuse_batch(&self, _stack: Tensor<B, 5>) -> Tensor<B, 4> {
        unimplemented!("This model does not support batch fusion")
    }

    /// Receptive-field radius in pixels: the maximum distance a single
    /// output pixel's value can depend on an input pixel, summed across the
    /// network's convolutions.
    ///
    /// Drives the minimum tile overlap the tiled inference
    /// ([`crate::infer`]) must use to stay seam-free — see the trait's
    /// "Receptive field and tiling" docs above. A model with no spatial
    /// context mixing at all (e.g. a purely pointwise/per-pixel model) should
    /// return `0`.
    fn receptive_field(&self) -> usize;
}

/// A pairwise focus-merge loss function.
///
/// Implement this trait to provide a custom loss function for training
/// pairwise fusion models.
pub trait PairwiseFusionLoss<B: Backend> {
    /// Compute the total scalar loss.
    ///
    /// # Arguments
    /// * `pred`       – network predictions (merged, conf).
    /// * `gt_merged`  – ground-truth composite `[N, 3, H, W]`.
    /// * `gt_conf`    – ground-truth confidence `[N, 1, H, W]`.
    /// * `source`     – source frame `[N, 3, H, W]` (used for sharpness ref).
    /// * `occlusion`  – depth-edge mask `[N, 1, H, W]` in 0..1, 1 = edge.
    fn forward(
        &self,
        pred: &MergeStep<B>,
        gt_merged: &Tensor<B, 4>,
        gt_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
        occlusion: &Tensor<B, 4>,
    ) -> Tensor<B, 1>;
}

/// A batched focus-merge loss function.
///
/// Implement this trait to provide a custom loss function for training
/// batched fusion models.
pub trait BatchFusionLoss<B: Backend> {
    /// Compute the total scalar loss.
    ///
    /// # Arguments
    /// * `pred_merged` - predicted final composite `[N, 3, H, W]`.
    /// * `gt_merged`   - ground-truth final composite `[N, 3, H, W]`.
    fn forward(&self, pred_merged: &Tensor<B, 4>, gt_merged: &Tensor<B, 4>) -> Tensor<B, 1>;
}

/// An alignment model that predicts affine matrices.
pub trait BatchAlignmentModel<B: Backend> {
    /// Predict alignment matrices for a batch of sequences.
    ///
    /// * `stack` - `[N, S, C, H, W]`
    ///
    /// Returns the predicted affine matrices `[N, S, 3, 3]`.
    fn align_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4>;
}

/// A loss function for the alignment model.
pub trait BatchAlignmentLoss<B: Backend> {
    /// Computes the alignment loss (e.g. corner/grid loss).
    ///
    /// * `pred_matrices` - predicted affine matrices `[N, S, 3, 3]`.
    /// * `gt_matrices`   - ground-truth affine matrices `[N, S, 3, 3]`.
    fn forward(&self, pred_matrices: Tensor<B, 4>, gt_matrices: Tensor<B, 4>) -> Tensor<B, 1>;
}

/// A pairwise alignment model: predicts one affine matrix registering a
/// single moving `frame` against a `reference`, called once per frame
/// (streaming, O(1) memory in stack size) rather than batched over the whole
/// stack.
///
/// This is the streaming sibling of [`BatchAlignmentModel`] — see
/// `docs/fusionalign-design.md` for why both exist and how a caller decides
/// which one to use. The built-in implementation is
/// [`crate::model::FusionAlignNet`].
pub trait PairAlignmentModel<B: Backend> {
    /// Predict the alignment matrix registering `frame` to `reference`.
    ///
    /// * `reference` - `[N, C, H, W]`
    /// * `frame`      - `[N, C, H, W]`
    ///
    /// Returns the predicted affine matrices `[N, 3, 3]`, in the SAME
    /// normalized-coordinate convention as
    /// [`BatchAlignmentModel::align_batch`] (see that method's docs, and
    /// [`crate::bridge::align_planar`]'s docs for the pixel ⇄ normalized
    /// conjugation).
    fn align_pair(&self, reference: Tensor<B, 4>, frame: Tensor<B, 4>) -> Tensor<B, 3>;
}

/// A loss function for the pairwise alignment model.
///
/// Mirrors [`BatchAlignmentLoss`] but for a single `[N, 3, 3]` matrix pair —
/// no `S` (stack) dimension, since [`PairAlignmentModel::align_pair`] scores
/// one frame at a time.
pub trait PairAlignmentLoss<B: Backend> {
    /// Computes the alignment loss (e.g. corner/grid loss).
    ///
    /// * `pred_matrix` - predicted affine matrix `[N, 3, 3]`.
    /// * `gt_matrix`   - ground-truth affine matrix `[N, 3, 3]`.
    fn forward(&self, pred_matrix: Tensor<B, 3>, gt_matrix: Tensor<B, 3>) -> Tensor<B, 1>;
}
