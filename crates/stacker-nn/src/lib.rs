#![allow(clippy::similar_names)]
//! # stacker-nn — Neural-network focus-merge and alignment crate
//!
//! This crate implements the learning-based half of the focus-stacking
//! pipeline: FOUR training/inference strategies, each with a built-in
//! network architecture and a matching loss:
//!
//! | Strategy | Trait | Built-in model | Built-in loss | Manifest tag |
//! |---|---|---|---|---|
//! | Pairwise fusion | [`traits::FusionModel`] (`merge`) | [`model::FocusMergeNet`] | [`loss::FocusFusionLoss`] | `focusmerge-v1` |
//! | Batch fusion | [`traits::FusionModel`] (`fuse_batch`) | [`model::BatchMergeNet`] | [`loss::FocusBatchLoss`] | `batchmerge-v2` |
//! | Batch alignment | [`traits::BatchAlignmentModel`] | [`model::BatchAlignNet`] | [`loss::CornerAlignmentLoss`] | `batchalign-v2` |
//! | Pairwise alignment | [`traits::PairAlignmentModel`] | [`model::FusionAlignNet`] | [`loss::PairCornerAlignmentLoss`] | `fusionalign-v1` |
//!
//! ## Pairwise fusion
//!
//! A **recurrent pairwise-merge network** folds a focus stack one frame at a
//! time:
//!
//! ```text
//!   frame_0  ──► FocusMergeNet ──► target_1
//!   frame_1  ──►                       │
//!                                      ▼
//!                             FocusMergeNet ──► target_2
//!                   frame_2  ──►
//!                                      …
//! ```
//!
//! Each step takes a *current composite* (`target_k`, confidence `[1,H,W]`) and
//! a new *source frame* (`[3,H,W]` RGB) and predicts an updated composite and
//! confidence. Ground-truth supervision uses `prefix_composite`, which
//! composites frames `0..=k` deterministically from the training data. Memory
//! is bounded (one frame resident at a time); see [`model::FocusMergeNet`].
//!
//! ## Batch fusion
//!
//! [`model::BatchMergeNet`] instead ingests the ENTIRE focus stack at once and
//! predicts soft per-frame blending weights directly, with no recurrence.
//! Simpler gradient flow (no truncated BPTT), at the cost of VRAM scaling
//! linearly with stack size — [`infer::fuse_stack`]'s tiled path
//! ([`infer::fuse_batch_tiled`]) keeps this bounded to one tile's worth of the
//! stack at a time; see that function's docs and the README's memory note.
//!
//! ## Alignment
//!
//! [`model::BatchAlignNet`] predicts one affine registration matrix per frame
//! (not a fused image) — the network the pipeline runs BEFORE fusion when
//! frames are misaligned (focus breathing, slight camera shift). See
//! [`bridge::align_planar`]'s docs for the full pixel ⇄ normalized coordinate
//! contract, and `scripts/03_simulate_misalignment.py` for how training data
//! (with ground-truth matrices) is generated.
//!
//! [`model::FusionAlignNet`] is the streaming sibling: it registers one frame
//! against the reference per call (O(1) memory in stack size) instead of
//! ingesting the whole stack at once, via [`bridge::align_planar_pairwise`]
//! (same coordinate contract as `align_planar`). See
//! `docs/fusionalign-design.md` for the motivation and
//! [`runtime::LoadedAlignModel`]'s docs for how a caller picks between the
//! two alignment architectures transparently.
//!
//! ## Backend feature matrix
//!
//! | Feature flag | Backend                  | Use-case                         |
//! |---|---|---|
//! | `ndarray`    | `burn-ndarray` (CPU)     | CI, unit tests, pure-Rust default |
//! | `wgpu`       | `burn-wgpu` (Vulkan/MSL) | AMD / Intel / NVIDIA inference    |
//! | `cuda`       | `burn-cuda` (CUDA)       | NVIDIA A100 training              |
//! | `autodiff`   | + autodiff wrapper       | Any backend, enables backprop     |
//!
//! Only **one** backend feature should be active at a time.  The `autodiff`
//! feature (default) wraps the selected backend in `AutodiffBackend` for
//! gradient flow. The whole training stack — the differentiable losses
//! (`train::rollout_loss` / `train::batch_loss` / `train::align_loss`) AND the
//! optimiser-driven `stacker-nn-train` binary — builds under the default
//! features and is CI-verified: `burn::optim` (`AdamW`) is available once
//! `burn/std` is on, so no extra feature is needed.
//!
//! ## Extending with your own architecture
//!
//! The built-in models are **not** the only architectures this crate can run.
//! The tiled inference, feathered blending, planar bridge, and recurrent fold
//! are all generic over the [`traits::FusionModel`] trait (fusion) or
//! [`traits::BatchAlignmentModel`] (alignment), so a third party can plug in
//! a different network without touching any of this crate's internals:
//!
//! 1. **Implement the trait** for your own `burn::module::Module` type. For
//!    fusion: `fn strategy(&self) -> FusionStrategy`, plus EITHER
//!    `fn merge(&self, target, target_conf, source) -> MergeStep<B>`
//!    (pairwise) OR `fn fuse_batch(&self, stack) -> Tensor<B, 4>` (batch),
//!    and `fn receptive_field(&self) -> usize` (how far spatial context
//!    propagates through your convolutions — this sizes the tile overlap so
//!    tiled inference stays seam-free). For alignment: `fn align_batch(&self,
//!    stack) -> Tensor<B, 4>` (whole-stack) or `fn align_pair(&self,
//!    reference, frame) -> Tensor<B, 3>` (streaming, one frame at a time).
//!    See the traits' own docs for the full shape/value contracts, and the
//!    crate README for a ~15-line toy example.
//! 2. **Load your weights however you like.** Weight loading and model
//!    *discovery* ([`discovery`], [`runtime::LoadedModel`],
//!    [`runtime::LoadedAlignModel`]) are specific to the built-in
//!    architectures' manifest/checkpoint formats — a custom architecture is
//!    expected to bring its own config type and its own `burn::record` (or
//!    otherwise) deserialisation, then construct `Self` directly. There is no
//!    generic `LoadedModel<M>`; see [`runtime::LoadedModel`]'s docs for why.
//! 3. **Run inference** by calling [`bridge::fuse_planar`] (from
//!    `PlanarImage<f32>`, the pipeline's colour type) or
//!    [`infer::fuse_stack`] (from raw linear-RGB tensors) with `&your_model`
//!    — both are generic over `M: FusionModel<B>`, so they accept your type
//!    exactly as they accept `&FocusMergeNet<B>`/`&BatchMergeNet<B>` today.
//!    For alignment, call [`bridge::align_planar`] with `&your_model: &M
//!    where M: BatchAlignmentModel<B>`, or [`bridge::align_planar_pairwise`]
//!    with `&your_model: &M where M: PairAlignmentModel<B>`. Use
//!    [`infer::TileConfig::for_model`] to derive a seam-safe tile overlap
//!    from `your_model.receptive_field()` instead of hard-coding one.
//! 4. **Training integration is architecture-specific today.** The
//!    differentiable loss functions in [`train`] (`rollout_loss`,
//!    `batch_loss`, `align_loss`) are intentionally left concrete over the
//!    built-in model types rather than generalised over these traits — see
//!    that module's docs for why forcing them through the traits would be
//!    more invasive than useful. A third-party architecture trains with its
//!    own loop (the pattern in `train.rs`/`bin/train.rs` is a reasonable
//!    template to copy) and only needs to satisfy [`traits::FusionModel`] /
//!    [`traits::BatchAlignmentModel`] to reuse this crate's *inference*-side
//!    machinery afterwards.
//!
//! ## Workspace lint policy
//!
//! The workspace enables `clippy::pedantic` and `clippy::nursery`.  We
//! suppress a small set of lints that are genuinely noisy for ML tensor code:
//!
//! * `cast_precision_loss` / `cast_possible_truncation` / `cast_sign_loss` –
//!   unavoidable pixel ↔ f32 conversions throughout the data pipeline.
//! * `module_name_repetitions` – common in Rust domain crates (e.g. `DataError`).
//! * `missing_errors_doc` – internal API; doc coverage comes in a later polish pass.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    // `clippy::nursery` implies this lint, which reports exactly 2 spanless
    // warnings ("first doc comment paragraph is too long" with NO file:line
    // location at all) for this crate. An exhaustive manual review of every
    // `///`/`//!` doc block's first paragraph in every `src/*.rs` file
    // (confirmed via `run_clippy` scoped to the plain `lib` target, which
    // rules out `bin/train.rs`, `tests/*.rs`, and `#[cfg(test)]` modules as
    // the source) found and fixed several genuinely-too-long first
    // paragraphs, but the warning count stayed at exactly 2 throughout —
    // strongly suggesting a diagnostic-span bug in this lint (nursery-tier,
    // known to be less mature) rather than a remaining real offender.
    // Toggling this attribute on/off was used to confirm the 2 warnings are
    // indeed this lint (they vanish with it, confirmed via `run_clippy`).
    clippy::too_long_first_doc_paragraph
)]

// ---------------------------------------------------------------------------
// Backend type aliases
//
// `SelectedBackend` is the single "training/scalar" backend used by the train
// binary and `TrainBackend`. Several backend features may be enabled at once
// (e.g. the GUI compiles `ndarray` + `wgpu` to offer a CPU/GPU choice at
// runtime via the [`runtime`] module), so the alias resolves by PRIORITY —
// cuda > wgpu > ndarray — rather than requiring exactly one. Runtime inference
// uses concrete backend types directly, not this alias.
// ---------------------------------------------------------------------------

// --- cuda (NVIDIA A100 training) — highest priority ---
#[cfg(feature = "cuda")]
pub type SelectedBackend = burn::backend::cuda::Cuda;

// --- wgpu (Vulkan / Metal / DX12 GPU) ---
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type SelectedBackend = burn::backend::Wgpu;

// --- ndarray (default, pure-Rust CPU) — lowest priority ---
#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "cuda")))]
pub type SelectedBackend = burn::backend::NdArray;

// Autodiff (training) wrapper — active with the `autodiff` feature, which is in
// the default set so the backward pass is exercised by CI. It wraps whichever
// base backend is selected.
#[cfg(feature = "autodiff")]
pub type TrainBackend = burn::backend::Autodiff<SelectedBackend>;

// ---------------------------------------------------------------------------
// Public modules
// ---------------------------------------------------------------------------

/// Dataset loading, preprocessing, and sample construction for all three
/// strategies (pairwise/batch fusion and alignment).
pub mod data;

/// Built-in network architectures: [`model::FocusMergeNet`] (pairwise
/// fusion), [`model::BatchMergeNet`] (batch fusion), [`model::BatchAlignNet`]
/// (batch alignment), [`model::FusionAlignNet`] (pairwise alignment).
pub mod model;

/// The [`traits::FusionModel`] / [`traits::BatchAlignmentModel`] /
/// [`traits::PairAlignmentModel`] extensibility traits.
///
/// Implement one of these to plug a custom architecture into the crate's
/// tiled inference, planar bridge, and recurrent fold without touching the
/// built-in models. See the "Extending with your own architecture" section
/// above.
pub mod traits;

/// Loss functions for all four strategies: [`loss::FocusFusionLoss`]
/// (pairwise fusion), [`loss::FocusBatchLoss`] (batch fusion),
/// [`loss::CornerAlignmentLoss`] (batch alignment), and
/// [`loss::PairCornerAlignmentLoss`] (pairwise alignment).
pub mod loss;

/// Tiled inference for large (8K+) images, for both fusion strategies.
///
/// Apron-overlap tiling with a raised-cosine feathered blend, plus the
/// streaming `fuse_stack` fold over a focus stack (dispatching between the
/// pairwise recurrent fold and the batch tiled path on the model's
/// [`traits::FusionStrategy`]).
pub mod infer;

/// Bridge between the pipeline's `PlanarImage` and model tensors.
///
/// Provides `fuse_planar` (fusion), `align_planar` (batch alignment), and
/// `align_planar_pairwise` (pairwise alignment), the high-level entry points
/// the CLI/GUI use to run a trained model.
pub mod bridge;

/// Differentiable affine image warp (the §5.2 photometric fine-tuning
/// building block).
///
/// [`warp::warp_affine`] resamples an image through a predicted affine
/// matrix with full gradient flow back into the matrix entries; see the
/// module docs for the coordinate convention and the validity-mask contract.
pub mod warp;

/// Discovery of trained models (`.mpk` + `.json` manifest) from a directory,
/// across all four architecture tags.
pub mod discovery;

/// Runtime backend dispatch (the app-facing inference entry point).
///
/// Hides Burn from the CLI/GUI: pick an `InferDevice` and call `fuse_entry`
/// (fusion) or use [`runtime::LoadedAlignModel`] (either alignment
/// architecture, batch or pairwise). Available devices are gated by the
/// compiled-in backend features.
pub mod runtime;

/// Differentiable training primitives for all four strategies.
///
/// `rollout_loss` (pairwise fusion, with scheduled sampling), `batch_loss`
/// (batch fusion), `align_loss` (batch alignment), `fusion_align_loss`
/// (pairwise alignment), plus the LR and sampling schedules. Builds under the
/// default `autodiff` feature and is CI-tested; the optimiser loop itself
/// lives in the `stacker-nn-train` binary.
#[cfg(feature = "autodiff")]
pub mod train;

/// Convenience re-exports for the high-level fusion entry points and discovery.
pub use bridge::{align_planar, align_planar_pairwise, fuse_planar};
pub use discovery::{
    BATCHALIGN_V2, BATCHMERGE_V2, FOCUSMERGE_V1, FUSIONALIGN_V1, ModelEntry, ModelManifest,
    discover_default_models, discover_models,
};
pub use infer::{TileConfig, fuse_stack};
pub use model::ModelSize;
pub use runtime::{
    AlignNet, AlignNetKind, InferDevice, LoadedAlignModel, LoadedModel, RuntimeError,
    available_devices, fuse_entry, gpu_available,
};
pub use traits::{BatchAlignmentModel, FusionModel, FusionStrategy, MergeStep, PairAlignmentModel};
