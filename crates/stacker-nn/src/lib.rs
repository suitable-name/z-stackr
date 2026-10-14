//! # stacker-nn — Neural-network pairwise focus-merge crate
//!
//! This crate implements the **recurrent pairwise-merge network** that drives
//! the learning-based focus-stacking pipeline.  The design is:
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
//! confidence.  Ground-truth supervision uses `prefix_composite`, which
//! composites frames `0..=k` deterministically from the training data.
//!
//! ## Backend feature matrix
//!
//! | Feature flag | Backend                  | Use-case                         |
//! |---|---|---|
//! | `ndarray`    | `burn-ndarray` (CPU)     | unit tests, pure-Rust default     |
//! | `wgpu`       | `burn-wgpu` (Vulkan/MSL) | AMD / Intel / NVIDIA inference    |
//! | `autodiff`   | + autodiff wrapper       | Any backend, enables backprop     |
//!
//! At least one of `ndarray` / `wgpu` must be on (enforced below); both may be
//! on at once. The `autodiff` feature (default) wraps the selected backend in
//! `AutodiffBackend` for gradient flow. The whole training stack — the
//! differentiable rollout (`train::rollout_loss`), the checkpoint helpers
//! (`checkpoint`) AND the optimiser-driven `z-stackr-train` binary — builds under
//! the default features: `burn::optim` (`AdamW`) is available once `burn/std` is
//! on, so no extra feature is needed.
//!
//! ## Determinism
//!
//! Inference is a pure function of the weights, the frames and the tiling.
//! Training draws random numbers from two places, and the trainer binary seeds
//! both from `--seed`: its own xorshift generator (scene order, crops, scheduled
//! sampling; seed plus the global step) and Burn's global generator, which
//! initialises the weights. Two runs with the same seed start from the same
//! weights and see the same scenes and crops.
//!
//! ## Workspace lint policy
//!
//! The workspace enables `clippy::pedantic` and `clippy::nursery`.  There is no
//! crate-wide `allow`: the few suppressions that remain are `#[expect(..., reason
//! = "...")]` on the item (or the module) that needs them, mostly pixel ↔ f32
//! conversions in the data pipeline.

#![warn(missing_docs)]

// ---------------------------------------------------------------------------
// Backend type aliases
//
// `SelectedBackend` is the single "training/scalar" backend used by the train
// binary and `TrainBackend`. Several backend features may be enabled at once
// (e.g. the GUI compiles `ndarray` + `wgpu` to offer a CPU/GPU choice at
// runtime via the [`runtime`] module), so the alias resolves by PRIORITY —
// wgpu > ndarray — rather than requiring exactly one. Runtime inference
// uses concrete backend types directly, not this alias.
// ---------------------------------------------------------------------------

#[cfg(not(any(feature = "ndarray", feature = "wgpu")))]
compile_error!(
    "z-stackr-nn needs an inference backend: enable the `ndarray` and/or `wgpu` feature"
);

// --- wgpu (Vulkan / Metal / DX12 GPU) — highest priority ---
/// The backend the trainer and [`TrainBackend`] use: `burn-wgpu` when the `wgpu`
/// feature is on (it wins when both are).
#[cfg(feature = "wgpu")]
pub type SelectedBackend = burn::backend::Wgpu;

// --- ndarray (default, pure-Rust CPU) — lowest priority ---
/// The backend the trainer and [`TrainBackend`] use: the pure-Rust `burn-ndarray`
/// CPU backend (no `wgpu` feature in this build).
#[cfg(all(feature = "ndarray", not(feature = "wgpu")))]
pub type SelectedBackend = burn::backend::NdArray;

// Autodiff (training) wrapper — active with the `autodiff` feature, which is in
// the default set so the backward pass is reachable (and tested) under the
// default build. It wraps whichever base backend is selected.
/// [`SelectedBackend`] wrapped for automatic differentiation: the backend the
/// training loop runs on.
#[cfg(feature = "autodiff")]
pub type TrainBackend = burn::backend::Autodiff<SelectedBackend>;

// ---------------------------------------------------------------------------
// Public modules
// ---------------------------------------------------------------------------

/// Dataset loading, preprocessing, and rollout sequence construction.
pub mod data;

pub mod model;

pub mod traits;

/// Training losses.
///
/// `FocusFusionLoss` / `FocusBatchLoss` combine Charbonnier reconstruction, a
/// multi-scale gradient term, sharpness retention and confidence (or gate)
/// supervision with occlusion down-weighting; the corner losses supervise the
/// alignment networks; `photometric_gradient_loss` is the unsupervised fine-tune
/// loss.
pub mod loss;

pub mod warp;

/// Tiled inference for large (8K+) images.
///
/// Apron-overlap tiling with a raised-cosine feathered blend, the streaming
/// `fuse_stack` fold over a focus stack (pairwise models) and `fuse_batch_tiled`
/// (whole-stack models).
pub mod infer;

/// Bridge between the pipeline's `PlanarImage` and model tensors.
///
/// Provides `fuse_planar`, the high-level entry point the CLI/GUI use to run a
/// trained model as a stacking algorithm, the thumbnail-based alignment entry
/// points (`FrameThumbs`, `align_thumbs`) and the pixel / normalised matrix
/// conversions that define the alignment coordinate conventions.
pub mod bridge;

/// Discovery of trained models (`.mpk` + `.json` manifest) from a directory.
///
/// The manifest's `architecture` tag selects one of the four network families;
/// fusion and alignment models are told apart with `ModelEntry::is_fusion` /
/// `is_alignment`.
pub mod discovery;

/// Crash-safe training checkpoints (atomic writes, distinct model / optimiser
/// files, full-precision rolling recorder) and the resume state.
#[cfg(feature = "autodiff")]
pub mod checkpoint;

/// Runtime backend dispatch (the app-facing inference entry point).
///
/// Hides Burn from the CLI/GUI: pick an `InferDevice` and call `fuse_entry`
/// (fusion) or load a `LoadedAlignModel` (alignment). Available devices are gated
/// by the compiled-in backend features.
pub mod runtime;

/// Differentiable training primitives.
///
/// The recurrent rollout (`rollout_loss`) with scheduled sampling, plus the LR
/// and sampling schedules. Builds under the default `autodiff` feature; the
/// optimiser loop itself lives in the `z-stackr-train` binary.
#[cfg(feature = "autodiff")]
pub mod train;

/// Convenience re-exports for the high-level fusion entry points and discovery.
pub use bridge::fuse_planar;
pub use discovery::{
    Architecture, ModelEntry, ModelManifest, discover_default_alignment_models,
    discover_default_fusion_models, discover_default_models, discover_models,
};
pub use infer::{TileConfig, fuse_stack};
pub use model::ModelSize;
pub use runtime::{
    InferDevice, LoadedAlignModel, LoadedModel, RuntimeError, available_devices, fuse_entry,
    gpu_available,
};
