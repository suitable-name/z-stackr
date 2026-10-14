//! Runtime backend dispatch for *inference* — the app-facing entry point.
//!
//! The CLI/GUI do not depend on Burn directly. They discover models via
//! [`crate::discovery`], pick an [`InferDevice`], and call [`fuse_entry`] (or
//! [`LoadedModel::load`] for repeated fusions), which monomorphises the generic
//! pipeline onto the concrete backend selected at runtime. For alignment,
//! [`LoadedAlignModel`] is the analogous wrapper around the two alignment
//! networks. Which devices exist is decided at *compile time* by the backend
//! features:
//!
//! * `ndarray` (default) → [`InferDevice::Cpu`]
//! * `wgpu` → [`InferDevice::Gpu`]
//!
//! So a CPU-only build exposes only `Cpu` (and the UI should not show a backend
//! chooser), while a `wgpu`-enabled build exposes both and the user can pick.
//!
//! Which network an entry holds is read from its manifest's architecture tag:
//! [`LoadedModel::load`] accepts the two fusion families and returns
//! [`DiscoveryError::ArchitectureMismatch`] (inside [`RuntimeError::Discovery`])
//! for an alignment entry; [`LoadedAlignModel::load`] is the mirror image.

use stacker_core::image::PlanarImage;

use crate::{
    bridge::{
        BridgeError, FrameThumbs, PixelMatrix, align_thumbs, align_thumbs_pairwise, fuse_planar,
    },
    discovery::{Architecture, DiscoveryError, ModelEntry},
    infer::TileConfig,
    model::{BatchAlignNet, BatchMergeNet, FocusMergeNet, FusionAlignNet, ModelSize},
    traits::{FusionModel, FusionStrategy, MergeStep},
};

/// Errors from a runtime request.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// Loading the model weights failed.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    /// The fusion or alignment itself failed.
    #[error(transparent)]
    Bridge(#[from] BridgeError),
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
// The variants are cfg-gated, so the list is an array whose elements are too.
#[must_use]
pub fn available_devices() -> Vec<InferDevice> {
    Vec::from([
        #[cfg(feature = "wgpu")]
        InferDevice::Gpu,
        #[cfg(feature = "ndarray")]
        InferDevice::Cpu,
    ])
}

/// The preferred default device (GPU if available, else CPU).
#[must_use]
pub fn default_device() -> Option<InferDevice> {
    available_devices().into_iter().next()
}

/// Recommended tiling for app inference when the model is not known.
///
/// The overlap is `2·R` for the largest preset (`xxl`, `R = 100` px, so 200 px),
/// which is safe for every preset. Computed from
/// [`crate::model::receptive_field_radius`] — the same function the README table
/// and [`TileConfig::default`] use.
///
/// Prefer [`recommended_tile_for`] when the [`ModelEntry`] is at hand: a small
/// model needs far less overlap (32 px for `xs`).
#[must_use]
pub const fn recommended_tile() -> TileConfig {
    TileConfig::for_size(ModelSize::Xxl)
}

/// Recommended tiling for `entry`.
///
/// The overlap is `2·R` where `R` is the receptive-field radius of the entry's
/// exact geometry (manifest width/depth, not just its preset); both fusion
/// families share the radius formula. An unusable manifest, or an alignment
/// entry (which is not tiled), falls back to [`recommended_tile`].
#[must_use]
pub fn recommended_tile_for(entry: &ModelEntry) -> TileConfig {
    let radius = match entry.architecture() {
        Some(Architecture::FocusMerge) => entry
            .manifest
            .to_config()
            .ok()
            .map(|cfg| cfg.receptive_field_radius()),
        Some(Architecture::BatchMerge) => entry
            .manifest
            .to_batch_config()
            .ok()
            .map(|cfg| cfg.receptive_field_radius()),
        _ => None,
    };
    radius.map_or_else(recommended_tile, TileConfig::for_radius)
}

/// Either built-in fusion architecture, dispatched through one [`FusionModel`]
/// impl.
///
/// Lets [`LoadedModel`] hold either without a generic parameter. Construct via
/// [`LoadedModel::load`], which picks the variant from the entry's architecture
/// tag. Both variants are boxed so the enum stays small whichever is resident.
pub enum FusionNet<B: burn::prelude::Backend> {
    /// Wraps [`FocusMergeNet`] ([`FusionStrategy::Pairwise`]).
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

/// A fusion model loaded onto a concrete backend, ready to fuse many stacks.
///
/// Holds the weights resident so repeated fusions (e.g. the CLI's per-tile loop)
/// avoid re-reading them; use [`fuse_entry`] for a one-shot fusion. It wraps
/// either [`FocusMergeNet`] (a streaming pairwise fold) or [`BatchMergeNet`] (one
/// call per tile over the whole stack), chosen by [`LoadedModel::load`] from the
/// entry's architecture tag; [`LoadedModel::fuse`] behaves identically for both.
/// Alignment models go through [`LoadedAlignModel`].
pub enum LoadedModel {
    /// CPU (`ndarray`) backend.
    #[cfg(feature = "ndarray")]
    Cpu {
        /// The loaded network (either fusion architecture).
        model: FusionNet<burn::backend::NdArray>,
        /// Device handle.
        device: <burn::backend::NdArray as burn::prelude::Backend>::Device,
    },
    /// GPU (`wgpu`) backend.
    #[cfg(feature = "wgpu")]
    Gpu {
        /// The loaded network (either fusion architecture).
        model: FusionNet<burn::backend::Wgpu>,
        /// Device handle.
        device: <burn::backend::Wgpu as burn::prelude::Backend>::Device,
    },
}

/// Load `entry` into a [`FusionNet`] on backend `B`, dispatching on the manifest's
/// architecture tag.
fn load_fusion_net<B: burn::prelude::Backend>(
    entry: &ModelEntry,
    device: &B::Device,
) -> Result<FusionNet<B>, DiscoveryError> {
    if entry.architecture() == Some(Architecture::BatchMerge) {
        Ok(FusionNet::Batch(Box::new(entry.load_batch::<B>(device)?)))
    } else {
        // `entry.load` itself validates the tag is focusmerge and returns
        // `ArchitectureMismatch` otherwise (e.g. an alignment checkpoint).
        Ok(FusionNet::Focus(Box::new(entry.load::<B>(device)?)))
    }
}

impl LoadedModel {
    /// Load `entry`'s weights onto `device`, keeping the model resident.
    ///
    /// # Errors
    /// [`RuntimeError::Discovery`] when the entry is not a fusion model
    /// ([`DiscoveryError::ArchitectureMismatch`]), the manifest is unusable or the
    /// weights cannot be loaded.
    pub fn load(entry: &ModelEntry, device: InferDevice) -> Result<Self, RuntimeError> {
        match device {
            #[cfg(feature = "ndarray")]
            InferDevice::Cpu => {
                use burn::{backend::NdArray, prelude::Backend};
                let device = <NdArray as Backend>::Device::default();
                let model = load_fusion_net::<NdArray>(entry, &device)?;
                Ok(Self::Cpu { model, device })
            }
            #[cfg(feature = "wgpu")]
            InferDevice::Gpu => {
                use burn::{backend::Wgpu, prelude::Backend};
                let device = <Wgpu as Backend>::Device::default();
                let model = load_fusion_net::<Wgpu>(entry, &device)?;
                Ok(Self::Gpu { model, device })
            }
        }
    }

    /// The execution strategy of the resident network.
    #[must_use]
    pub fn strategy(&self) -> FusionStrategy {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, .. } => model.strategy(),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, .. } => model.strategy(),
        }
    }

    /// A [`TileConfig`] sized for this specific resident model: overlap `2·R` for
    /// its exact receptive-field radius `R`.
    #[must_use]
    pub fn recommended_tile(&self) -> TileConfig {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, .. } => TileConfig::for_model(model),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, .. } => TileConfig::for_model(model),
        }
    }

    /// Fuse one focus stack with the resident model.
    ///
    /// # Errors
    /// [`RuntimeError::Bridge`] for an empty or mismatched stack or a failed
    /// inference.
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

    /// Fuse one stack, invoking `on_step(&running_composite)` after each progress
    /// step — for a live preview that shows the result sharpening. A pairwise
    /// model reports after every frame, a batch model after every completed tile
    /// row (see [`crate::infer::fuse_batch_tiled`]).
    ///
    /// # Errors
    /// As [`Self::fuse`].
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

/// Load `entry` on `device` and fuse `frames` into one all-in-focus image.
///
/// Convenience one-shot wrapper around [`LoadedModel`].
///
/// # Errors
/// Any error of [`LoadedModel::load`] or [`LoadedModel::fuse`].
pub fn fuse_entry(
    entry: &ModelEntry,
    device: InferDevice,
    frames: &[PlanarImage<f32>],
    tile: TileConfig,
) -> Result<PlanarImage<f32>, RuntimeError> {
    LoadedModel::load(entry, device)?.fuse(frames, tile)
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

/// Either built-in alignment architecture, dispatched by kind.
///
/// Unlike [`FusionNet`], the two alignment traits have different method shapes
/// (`align_batch(stack)` vs `align_pair(reference, frame)`), so there is no single
/// trait method to dispatch through; [`AlignNet::kind`] lets [`LoadedAlignModel`]
/// pick the whole-stack or the streaming bridge entry point while returning the
/// same matrix type either way. Both variants are boxed.
pub enum AlignNet<B: burn::prelude::Backend> {
    /// Wraps [`BatchAlignNet`] ([`crate::traits::BatchAlignmentModel`]).
    Batch(Box<BatchAlignNet<B>>),
    /// Wraps [`FusionAlignNet`] ([`crate::traits::PairAlignmentModel`]).
    Fusion(Box<FusionAlignNet<B>>),
}

/// Which alignment architecture kind an [`AlignNet`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignNetKind {
    /// [`BatchAlignNet`] — registers the whole stack in one call.
    Batch,
    /// [`FusionAlignNet`] — registers one frame against the reference per call,
    /// streamed.
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

    /// Register a stack of thumbnails, returning one full-resolution pixel
    /// matrix per frame (see [`crate::bridge`]).
    fn align_thumbs(
        &self,
        thumbs: &FrameThumbs,
        device: &B::Device,
    ) -> Result<Vec<PixelMatrix>, BridgeError> {
        match self {
            Self::Batch(m) => align_thumbs(m.as_ref(), thumbs, device),
            Self::Fusion(m) => align_thumbs_pairwise(m.as_ref(), thumbs, device),
        }
    }
}

/// Load `entry` into an [`AlignNet`] on backend `B`, dispatching on the manifest's
/// architecture tag.
fn load_align_net<B: burn::prelude::Backend>(
    entry: &ModelEntry,
    device: &B::Device,
) -> Result<AlignNet<B>, DiscoveryError> {
    if entry.architecture() == Some(Architecture::FusionAlign) {
        Ok(AlignNet::Fusion(Box::new(
            entry.load_fusion_align::<B>(device)?,
        )))
    } else {
        // `entry.load_align` itself validates the tag is batchalign and returns
        // `ArchitectureMismatch` otherwise (e.g. a fusion checkpoint).
        Ok(AlignNet::Batch(Box::new(entry.load_align::<B>(device)?)))
    }
}

/// An alignment model loaded onto a concrete backend, ready to register many
/// stacks.
///
/// The alignment counterpart to [`LoadedModel`]: wraps an [`AlignNet`] (either
/// [`BatchAlignNet`] or [`FusionAlignNet`]) on one of the compiled-in backends.
/// The networks run on **thumbnails** (longest side ≤ 512 px,
/// [`crate::bridge::ALIGN_DOWNSCALE_LONG_SIDE`]) and the returned matrices are in
/// **full-resolution pixel coordinates**, reference → frame, frame 0 being the
/// identity (exactly for a batch model, approximately for a streaming one); see
/// [`crate::bridge`] for the conventions. To keep the stack's memory bound, build a
/// [`FrameThumbs`] while streaming the frames and call
/// [`LoadedAlignModel::align_thumbs`]; [`LoadedAlignModel::align`] is the
/// convenience for frames already in memory.
pub enum LoadedAlignModel {
    /// CPU (`ndarray`) backend.
    #[cfg(feature = "ndarray")]
    Cpu {
        /// The loaded network (either alignment architecture).
        model: AlignNet<burn::backend::NdArray>,
        /// Device handle.
        device: <burn::backend::NdArray as burn::prelude::Backend>::Device,
    },
    /// GPU (`wgpu`) backend.
    #[cfg(feature = "wgpu")]
    Gpu {
        /// The loaded network (either alignment architecture).
        model: AlignNet<burn::backend::Wgpu>,
        /// Device handle.
        device: <burn::backend::Wgpu as burn::prelude::Backend>::Device,
    },
}

impl LoadedAlignModel {
    /// Load `entry`'s weights onto `device`, keeping the model resident.
    ///
    /// # Errors
    /// [`RuntimeError::Discovery`] when the entry is not an alignment model
    /// ([`DiscoveryError::ArchitectureMismatch`]; use [`LoadedModel::load`] for a
    /// fusion entry), the manifest is unusable or the weights cannot be loaded.
    pub fn load(entry: &ModelEntry, device: InferDevice) -> Result<Self, RuntimeError> {
        match device {
            #[cfg(feature = "ndarray")]
            InferDevice::Cpu => {
                use burn::{backend::NdArray, prelude::Backend};
                let device = <NdArray as Backend>::Device::default();
                let model = load_align_net::<NdArray>(entry, &device)?;
                Ok(Self::Cpu { model, device })
            }
            #[cfg(feature = "wgpu")]
            InferDevice::Gpu => {
                use burn::{backend::Wgpu, prelude::Backend};
                let device = <Wgpu as Backend>::Device::default();
                let model = load_align_net::<Wgpu>(entry, &device)?;
                Ok(Self::Gpu { model, device })
            }
        }
    }

    /// Which alignment architecture kind is resident.
    #[must_use]
    pub const fn kind(&self) -> AlignNetKind {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, .. } => model.kind(),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, .. } => model.kind(),
        }
    }

    /// Register a stack of thumbnails, returning one full-resolution pixel-space
    /// matrix per frame (reference → frame).
    ///
    /// # Errors
    /// [`RuntimeError::Bridge`] for an empty stack or a failed inference.
    pub fn align_thumbs(&self, thumbs: &FrameThumbs) -> Result<Vec<PixelMatrix>, RuntimeError> {
        match self {
            #[cfg(feature = "ndarray")]
            Self::Cpu { model, device } => Ok(model.align_thumbs(thumbs, device)?),
            #[cfg(feature = "wgpu")]
            Self::Gpu { model, device } => Ok(model.align_thumbs(thumbs, device)?),
        }
    }

    /// Register `frames` (all the same size), returning one full-resolution
    /// pixel-space matrix per frame. Thumbnails the frames first.
    ///
    /// # Errors
    /// [`RuntimeError::Bridge`] for an empty or mismatched stack or a failed
    /// inference.
    pub fn align(&self, frames: &[PlanarImage<f32>]) -> Result<Vec<PixelMatrix>, RuntimeError> {
        self.align_thumbs(&FrameThumbs::from_frames(frames).map_err(RuntimeError::from)?)
    }
}
