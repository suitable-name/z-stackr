//! Runtime backend dispatch for *inference* — the app-facing entry point.
//!
//! The CLI/GUI do not depend on Burn directly. They discover models via
//! [`crate::discovery`], pick an [`InferDevice`], and call [`fuse_entry`] (or
//! [`LoadedModel::load`] for repeated fusions), which monomorphises the
//! generic pipeline onto the concrete backend selected at runtime. For
//! alignment, [`LoadedAlignModel`] is the analogous wrapper around
//! [`crate::model::BatchAlignNet`]. Which devices exist is decided at
//! *compile time* by the backend features:
//!
//! * `ndarray` (default) → [`InferDevice::Cpu`]
//! * `wgpu` → [`InferDevice::Gpu`]
//!
//! So a CPU-only build exposes only `Cpu` (and the UI should not show a backend
//! chooser), while a `wgpu`-enabled build exposes both and the user can pick.

use stacker_core::image::PlanarImage;

use crate::{
    bridge::{BridgeError, fuse_planar},
    discovery::{DiscoveryError, ModelEntry},
    infer::TileConfig,
    model::{BatchAlignNet, BatchMergeNet, FocusMergeNet, FusionAlignNet},
    traits::{FusionModel, FusionStrategy, MergeStep},
};

/// Errors from a runtime fusion request.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Loading the model weights failed.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    /// The fusion itself failed.
    #[error(transparent)]
    Bridge(#[from] BridgeError),
    /// No inference backend was compiled in.
    #[error("no inference backend available (build with the `ndarray` and/or `wgpu` feature)")]
    NoBackend,
}

/// A compute device available for inference, gated by compiled-in backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferDevice {
    /// Pure-Rust CPU backend (`ndarray`).
    #[cfg(feature = "ndarray")]
    Cpu,
    /// GPU backend via wgpu/Vulkan (`wgpu`).
    #[cfg(feature = "wgpu")]
    Gpu,
}

impl InferDevice {
    /// Short label for UIs / logs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu => "CPU",
            #[cfg(feature = "wgpu")]
            Self::Gpu => "GPU",
        }
    }
}

/// Whether a GPU backend is compiled in.
#[must_use]
pub const fn gpu_available() -> bool {
    cfg!(feature = "wgpu")
}

/// All inference devices available in this build, GPU first when present.
///
/// Empty only if the crate was built with no backend feature at all. When this
/// has length ≤ 1 the UI should hide the backend selector.
// `mut`/push pattern is required because the variants are cfg-gated.
#[allow(unused_mut, clippy::vec_init_then_push)]
#[must_use]
pub fn available_devices() -> Vec<InferDevice> {
    let mut devices = Vec::new();
    #[cfg(feature = "wgpu")]
    devices.push(InferDevice::Gpu);
    #[cfg(feature = "ndarray")]
    devices.push(InferDevice::Cpu);
    devices
}

/// The preferred default device (GPU if available, else CPU).
#[must_use]
pub fn default_device() -> Option<InferDevice> {
    available_devices().into_iter().next()
}

/// Recommended tiling for app inference. The apron must exceed the model's
/// receptive field; the largest preset (`xxl`) needs roughly 90 px, so a 96 px
/// overlap is safe across all presets.
///
/// This is a `const fn` (so it works without a loaded model, e.g. for UI
/// defaults before a model is picked) and therefore can't call
/// [`crate::infer::TileConfig::for_model`], which needs a concrete
/// [`crate::traits::FusionModel`] instance to read
/// `receptive_field()` from. Once a [`LoadedModel`] is resident, prefer
/// `TileConfig::for_model(&model)` — it derives the same 96 px floor and
/// widens automatically for a larger custom architecture.
#[must_use]
pub const fn recommended_tile() -> TileConfig {
    TileConfig {
        tile: 512,
        overlap: 96,
    }
}

/// Either built-in fusion architecture, dispatched through one [`FusionModel`] impl.
///
/// Lets [`LoadedModel`] hold either without a generic parameter. Not part of
/// the curated public API surface beyond that impl — construct via
/// [`LoadedModel::load`], which picks the variant from the discovered entry's
/// [`crate::discovery::ModelManifest::architecture`] tag.
pub enum FusionNet<B: burn::prelude::Backend> {
    /// Wraps [`FocusMergeNet`] ([`FusionStrategy::Pairwise`]).
    ///
    /// Both variants are boxed (rather than only the larger one) so
    /// `FusionNet` stays a small, uniformly-sized enum regardless of which
    /// architecture is resident — the models themselves are heap-allocated
    /// parameter trees either way, so this indirection costs nothing beyond
    /// one pointer.
    Focus(Box<FocusMergeNet<B>>),
    /// Wraps [`BatchMergeNet`] ([`FusionStrategy::Batch`]).
    Batch(Box<BatchMergeNet<B>>),
}

impl<B: burn::prelude::Backend> FusionModel<B> for FusionNet<B> {
    fn strategy(&self) -> FusionStrategy {
        match self {
            Self::Focus(m) => m.strategy(),
            Self::Batch(m) => m.strategy(),
        }
    }

    fn merge(
        &self,
        target: burn::tensor::Tensor<B, 4>,
        target_conf: burn::tensor::Tensor<B, 4>,
        source: burn::tensor::Tensor<B, 4>,
    ) -> MergeStep<B> {
        match self {
            Self::Focus(m) => m.merge(target, target_conf, source),
            Self::Batch(m) => m.merge(target, target_conf, source),
        }
    }

    fn fuse_batch(&self, stack: burn::tensor::Tensor<B, 5>) -> burn::tensor::Tensor<B, 4> {
        match self {
            Self::Focus(m) => m.fuse_batch(stack),
            Self::Batch(m) => m.fuse_batch(stack),
        }
    }

    fn receptive_field(&self) -> usize {
        match self {
            Self::Focus(m) => m.receptive_field(),
            Self::Batch(m) => m.receptive_field(),
        }
    }
}

impl<B: burn::prelude::Backend> FusionNet<B> {
    /// Wrap a loaded [`FocusMergeNet`] as a [`FusionNet::Focus`], boxing it.
    fn focus(model: FocusMergeNet<B>) -> Self {
        Self::Focus(Box::new(model))
    }
}

/// A model loaded onto a concrete backend, ready to fuse many stacks.
///
/// Holds the weights resident so repeated fusions (e.g. the CLI's per-tile
/// loop) avoid re-reading them; use [`fuse_entry`] for a one-shot fusion.
///
/// `LoadedModel` wraps either built-in FUSION architecture —
/// [`crate::model::FocusMergeNet`] ([`FusionStrategy::Pairwise`]) or
/// [`crate::model::BatchMergeNet`] ([`FusionStrategy::Batch`]) — on one of
/// the compiled-in backends, chosen automatically by [`LoadedModel::load`]
/// from the discovered entry's
/// [`crate::discovery::ModelManifest::architecture`] tag (see
/// [`crate::discovery`]'s docs for the tag table). Both architectures
/// implement [`FusionModel`], so [`LoadedModel::fuse`] /
/// [`LoadedModel::fuse_with_progress`] work identically regardless of which
/// one is resident — [`crate::infer::fuse_stack`] dispatches on
/// [`FusionModel::strategy`] internally. It is the discovery/runtime
/// dispatch layer that lets the CLI/GUI stay Burn-agnostic; alignment models
/// go through the separate [`LoadedAlignModel`] instead (a different trait,
/// [`crate::traits::BatchAlignmentModel`], with a different output type).
///
/// **Third-party architectures bypass this type entirely.** There is no
/// generic `LoadedModel<M>` because weight *loading* (the `.mpk` record
/// format, the manifest-to-config reconstruction in
/// [`crate::discovery::ModelManifest::to_config`]) is inherently specific to
/// the built-in architectures' parameter layouts — a different architecture
/// will have its own checkpoint format. A custom model instead constructs
/// itself however it likes (its own config type, its own
/// `Recorder`/deserialisation) and calls [`crate::bridge::fuse_planar`] /
/// [`crate::infer::fuse_stack`] directly with `&self`; both are generic over
/// [`FusionModel`], not tied to `LoadedModel`. See the crate docs'
/// "Extending with your own architecture" section for a worked example.
pub enum LoadedModel {
    /// CPU (`ndarray`) backend.
    #[cfg(feature = "ndarray")]
    Cpu {
        /// The loaded network (either fusion architecture).
        model: FusionNet<burn::backend::NdArray>,
        /// Device handle.
        device: burn::prelude::Device<burn::backend::NdArray>,
    },
    /// GPU (`wgpu`) backend.
    #[cfg(feature = "wgpu")]
    Gpu {
        /// The loaded network (either fusion architecture).
        model: FusionNet<burn::backend::Wgpu>,
        /// Device handle.
        device: burn::prelude::Device<burn::backend::Wgpu>,
    },
}

/// Either built-in alignment architecture, dispatched by kind.
///
/// Unlike [`FusionNet`] (which unifies its two variants behind one
/// [`FusionModel`] impl because both fusion architectures share that trait),
/// [`crate::traits::BatchAlignmentModel`] and
/// [`crate::traits::PairAlignmentModel`] are genuinely different traits with
/// different method signatures (`align_batch(stack: Tensor<B,5>)` vs.
/// `align_pair(reference: Tensor<B,4>, frame: Tensor<B,4>)`) — there is no
/// single trait method to dispatch through. `AlignNet` instead exposes
/// [`AlignNet::kind`] so [`LoadedAlignModel::align`] can branch explicitly
/// between a whole-stack [`crate::bridge::align_planar`] call and a
/// streaming per-frame [`crate::bridge::align_planar_pairwise`] loop, while
/// still returning the identical `Vec<nalgebra::Matrix3<f32>>` type either
/// way.
///
/// Both variants are boxed for the same reason as [`FusionNet`]'s variants:
/// keeps `AlignNet` a small, uniformly-sized enum regardless of which
/// architecture is resident, at the cost of one pointer indirection.
pub enum AlignNet<B: burn::prelude::Backend> {
    /// Wraps [`BatchAlignNet`] ([`crate::traits::BatchAlignmentModel`]).
    Batch(Box<BatchAlignNet<B>>),
    /// Wraps [`FusionAlignNet`] ([`crate::traits::PairAlignmentModel`]).
    Fusion(Box<FusionAlignNet<B>>),
}

/// Which alignment architecture kind an [`AlignNet`] holds — used by
/// [`LoadedAlignModel::align`] to pick the whole-stack vs. per-frame bridge
/// entry point, and exposed for callers (e.g. the tiled pipeline) that need
/// to know up front which dispatch a loaded model will take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignNetKind {
    /// [`BatchAlignNet`] — registers the whole stack in one call.
    Batch,
    /// [`FusionAlignNet`] — registers one frame against the reference per
    /// call, streamed.
    Fusion,
}

impl<B: burn::prelude::Backend> AlignNet<B> {
    /// Which architecture kind this is — see [`AlignNetKind`].
    #[must_use]
    pub const fn kind(&self) -> AlignNetKind {
        match self {
            Self::Batch(_) => AlignNetKind::Batch,
            Self::Fusion(_) => AlignNetKind::Fusion,
        }
    }
}

/// An alignment model loaded onto a concrete backend, ready to register many
/// stacks.
///
/// The alignment counterpart to [`LoadedModel`]: wraps an [`AlignNet`] (either
/// [`crate::model::BatchAlignNet`] or [`crate::model::FusionAlignNet`]) on
/// one of the compiled-in backends. Construct via [`LoadedAlignModel::load`]
/// (which, like `LoadedModel::load`, validates the discovered entry's
/// architecture tag before deserialising — see
/// [`crate::discovery::ModelEntry::load_align`] /
/// [`crate::discovery::ModelEntry::load_fusion_align`]), then call
/// [`LoadedAlignModel::align`] to register a stack, returning one pixel-space
/// affine matrix per frame (see [`crate::bridge::align_planar`]'s /
/// [`crate::bridge::align_planar_pairwise`]'s docs for the exact,
/// shared coordinate contract) regardless of which architecture kind is
/// resident.
///
/// Kept as a separate type from [`LoadedModel`] rather than folded into it
/// because alignment is a different trait family
/// ([`crate::traits::BatchAlignmentModel`] /
/// [`crate::traits::PairAlignmentModel`]) with a different output type
/// (matrices, not an image) — there is no shared `strategy()`-style dispatch
/// that would make a single enum variant set meaningful across both fusion
/// and alignment.
pub enum LoadedAlignModel {
    /// CPU (`ndarray`) backend.
    #[cfg(feature = "ndarray")]
    Cpu {
        /// The loaded network (either alignment architecture).
        model: AlignNet<burn::backend::NdArray>,
        /// Device handle.
        device: burn::prelude::Device<burn::backend::NdArray>,
    },
    /// GPU (`wgpu`) backend.
    #[cfg(feature = "wgpu")]
    Gpu {
        /// The loaded network (either alignment architecture).
        model: AlignNet<burn::backend::Wgpu>,
        /// Device handle.
        device: burn::prelude::Device<burn::backend::Wgpu>,
    },
}

impl LoadedModel {
    /// A [`TileConfig`] sized for this specific resident model via
    /// [`TileConfig::for_model`] (tile 512, overlap = `receptive_field()`
    /// clamped to at least 96 px). Prefer this over the fixed
    /// [`recommended_tile`] once a model is loaded — it stays correct even if
    /// a future built-in preset grows past the current ~90 px receptive-field
    /// figure.
    #[must_use]
    pub fn recommended_tile(&self) -> TileConfig {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, .. } => TileConfig::for_model(model),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, .. } => TileConfig::for_model(model),
        }
    }

    /// Load `entry`'s weights onto `device`, keeping the model resident.
    ///
    /// Dispatches on `entry.manifest.architecture`
    /// ([`crate::discovery::FOCUSMERGE_V1`] or
    /// [`crate::discovery::BATCHMERGE_V2`]) to decide which built-in
    /// architecture to deserialise into.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Discovery`] (wrapping
    /// [`crate::discovery::DiscoveryError::ArchitectureMismatch`]) if
    /// `entry`'s manifest names neither fusion architecture (e.g. it is an
    /// alignment checkpoint — use [`LoadedAlignModel::load`] instead), or any
    /// other discovery/load error.
    pub fn load(entry: &ModelEntry, device: InferDevice) -> Result<Self, RuntimeError> {
        match device {
            #[cfg(feature = "ndarray")]
            InferDevice::Cpu => {
                use burn::{backend::NdArray, prelude::Device};
                let dev = Device::<NdArray>::default();
                let model = load_fusion_net::<NdArray>(entry, &dev)?;
                Ok(Self::Cpu { model, device: dev })
            }
            #[cfg(feature = "wgpu")]
            InferDevice::Gpu => {
                use burn::{backend::Wgpu, prelude::Device};
                let dev = Device::<Wgpu>::default();
                let model = load_fusion_net::<Wgpu>(entry, &dev)?;
                Ok(Self::Gpu { model, device: dev })
            }
        }
    }

    /// Fuse one focus stack with the resident model.
    pub fn fuse(
        &self,
        frames: &[PlanarImage<f32>],
        tile: TileConfig,
    ) -> Result<PlanarImage<f32>, RuntimeError> {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, device } => Ok(fuse_planar(model, frames, tile, device)?),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, device } => Ok(fuse_planar(model, frames, tile, device)?),
        }
    }

    /// Fuse one stack, invoking `on_step(&running_composite)` after each
    /// progress step — for a live preview that shows the result sharpening.
    /// See [`crate::infer::fuse_stack_with`]'s docs for how "progress step"
    /// differs between the pairwise and batch strategies.
    pub fn fuse_with_progress<F>(
        &self,
        frames: &[PlanarImage<f32>],
        tile: TileConfig,
        mut on_step: F,
    ) -> Result<PlanarImage<f32>, RuntimeError>
    where
        F: FnMut(&PlanarImage<f32>),
    {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, device } => Ok(crate::bridge::fuse_planar_with_progress(
                model,
                frames,
                tile,
                device,
                |_idx, img| on_step(img),
            )?),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, device } => Ok(crate::bridge::fuse_planar_with_progress(
                model,
                frames,
                tile,
                device,
                |_idx, img| on_step(img),
            )?),
        }
    }
}

/// Load `entry` into a [`FusionNet`] on backend `B`, dispatching on the
/// manifest's architecture tag. Shared by both [`LoadedModel::load`] arms.
fn load_fusion_net<B: burn::prelude::Backend>(
    entry: &ModelEntry,
    device: &burn::prelude::Device<B>,
) -> Result<FusionNet<B>, DiscoveryError> {
    if entry.manifest.architecture == crate::discovery::BATCHMERGE_V2 {
        Ok(FusionNet::Batch(Box::new(entry.load_batch::<B>(device)?)))
    } else {
        // `entry.load` itself validates the tag is FOCUSMERGE_V1 and returns
        // `DiscoveryError::ArchitectureMismatch` otherwise (e.g. an alignment
        // checkpoint), so no separate check is needed here.
        Ok(FusionNet::focus(entry.load::<B>(device)?))
    }
}

impl LoadedAlignModel {
    /// Load `entry`'s weights onto `device`, keeping the model resident.
    ///
    /// Dispatches on `entry.manifest.architecture`
    /// ([`crate::discovery::BATCHALIGN_V2`] or
    /// [`crate::discovery::FUSIONALIGN_V1`]) to decide which built-in
    /// alignment architecture to deserialise into — mirrors
    /// [`LoadedModel::load`]'s dispatch pattern for the fusion architectures.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::Discovery`] (wrapping
    /// [`crate::discovery::DiscoveryError::ArchitectureMismatch`]) if
    /// `entry`'s manifest names neither alignment architecture (e.g. it is a
    /// fusion checkpoint — use [`LoadedModel::load`] instead), or any other
    /// discovery/load error.
    pub fn load(entry: &ModelEntry, device: InferDevice) -> Result<Self, RuntimeError> {
        match device {
            #[cfg(feature = "ndarray")]
            InferDevice::Cpu => {
                use burn::{backend::NdArray, prelude::Device};
                let device = Device::<NdArray>::default();
                let model = load_align_net::<NdArray>(entry, &device)?;
                Ok(Self::Cpu { model, device })
            }
            #[cfg(feature = "wgpu")]
            InferDevice::Gpu => {
                use burn::{backend::Wgpu, prelude::Device};
                let device = Device::<Wgpu>::default();
                let model = load_align_net::<Wgpu>(entry, &device)?;
                Ok(Self::Gpu { model, device })
            }
        }
    }

    /// Which alignment architecture kind is resident — see [`AlignNetKind`].
    ///
    /// Useful for a caller (e.g. the tiled pipeline) that wants to know up
    /// front whether it should feed the whole stack at once or loop
    /// per-frame, without pattern-matching on [`AlignNet`] itself. In
    /// practice this is rarely needed: [`Self::align`] already picks the
    /// right dispatch internally and returns the same
    /// `Vec<nalgebra::Matrix3<f32>>` type regardless of kind.
    #[must_use]
    pub const fn kind(&self) -> AlignNetKind {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, .. } => model.kind(),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, .. } => model.kind(),
        }
    }

    /// Register `frames`, returning one pixel-space affine matrix per frame.
    ///
    /// Dispatches on which [`AlignNet`] variant is resident:
    /// [`AlignNetKind::Batch`] runs [`crate::bridge::align_planar`] (one
    /// whole-stack call); [`AlignNetKind::Fusion`] runs
    /// [`crate::bridge::align_planar_pairwise`] (one streaming call per
    /// frame against the stack's reference). Both bridge functions share the
    /// exact same pixel-space coordinate contract (see either function's
    /// docs), so the returned matrices are interchangeable regardless of
    /// which architecture produced them.
    pub fn align(
        &self,
        frames: &[PlanarImage<f32>],
    ) -> Result<Vec<nalgebra::Matrix3<f32>>, RuntimeError> {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, device } => Ok(align_with_net(model, frames, device)?),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, device } => Ok(align_with_net(model, frames, device)?),
        }
    }
}

/// Load `entry` into an [`AlignNet`] on backend `B`, dispatching on the
/// manifest's architecture tag. Shared by both [`LoadedAlignModel::load`]
/// arms — mirrors [`load_fusion_net`]'s pattern for the fusion side.
fn load_align_net<B: burn::prelude::Backend>(
    entry: &ModelEntry,
    device: &burn::prelude::Device<B>,
) -> Result<AlignNet<B>, DiscoveryError> {
    if entry.manifest.architecture == crate::discovery::FUSIONALIGN_V1 {
        Ok(AlignNet::Fusion(Box::new(
            entry.load_fusion_align::<B>(device)?,
        )))
    } else {
        // `entry.load_align` itself validates the tag is BATCHALIGN_V2 and
        // returns `DiscoveryError::ArchitectureMismatch` otherwise (e.g. a
        // fusion checkpoint), so no separate check is needed here.
        Ok(AlignNet::Batch(Box::new(entry.load_align::<B>(device)?)))
    }
}

/// Run [`LoadedAlignModel::align`]'s dispatch for one resident [`AlignNet`].
///
/// Factored out of the `Cpu`/`Gpu` match arms since the dispatch logic itself
/// (not just the backend selection) is identical for both.
fn align_with_net<B: burn::prelude::Backend>(
    model: &AlignNet<B>,
    frames: &[PlanarImage<f32>],
    device: &burn::prelude::Device<B>,
) -> Result<Vec<nalgebra::Matrix3<f32>>, BridgeError> {
    match model {
        AlignNet::Batch(m) => crate::bridge::align_planar(m.as_ref(), frames, device),
        AlignNet::Fusion(m) => crate::bridge::align_planar_pairwise(m.as_ref(), frames, device),
    }
}

/// Load `entry` on `device` and fuse `frames` into one all-in-focus image.
///
/// Convenience one-shot wrapper around [`LoadedModel`].
pub fn fuse_entry(
    entry: &ModelEntry,
    device: InferDevice,
    frames: &[PlanarImage<f32>],
    tile: TileConfig,
) -> Result<PlanarImage<f32>, RuntimeError> {
    LoadedModel::load(entry, device)?.fuse(frames, tile)
}
