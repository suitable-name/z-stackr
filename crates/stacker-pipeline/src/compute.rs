//! Where the heavy stages run: the CPU, or the GPU with the CPU behind it.
//!
//! [`Compute`] is what both apps call for the stages that have GPU kernels:
//!
//! | Stage | `Compute` method | CPU reference |
//! |---|---|---|
//! | `Apex` fusion | [`Compute::apex`] | [`fuse::apex`] |
//! | `Relief` fusion | [`Compute::relief`], [`Compute::relief_tile`], [`Compute::relief_focus_max`] | [`fuse::focus_maps`], [`fuse::select_and_smooth`] |
//! | `Relief` multigrid depth | [`Compute::multigrid_depth`] | `stacker_algo::relief::multigrid::multigrid_depth` |
//! | `Strata` saliency and guided pair | [`Compute::strata_kernels`] (handed to `fuse_strata_on`) | `CpuStrata` |
//! | alignment objective | [`Compute::alignment_device`] (handed to the refiners) | `stacker_align::refine` |
//! | final alignment warp | [`Compute::warp`] | `warp_image`, `warp_image_clamped`, `coverage_mask` |
//!
//! Decoding, colour conversion, auto-cull, export and the AI mode (which has its
//! own `nn-gpu` path) stay on the CPU.
//!
//! With `compute_device = "cpu"` (the default), or in a build without the `gpu`
//! feature, every method is exactly its CPU reference. With
//! `compute_device = "gpu"` in a build that has GPU support each tries the device
//! first and takes care of everything that can go wrong, so a run never fails
//! because of the GPU:
//!
//! | What happens | What `Compute` does |
//! |---|---|
//! | no adapter, or the build has no GPU support | CPU from the start, one notice saying why |
//! | the call needs more device memory than the budget ([`GPU_BUDGET_BYTES`]), or a setting the kernels do not cover | CPU for that call only, the GPU stays on for other calls |
//! | the device reports an error, a call fails, the code panics, or the result has non-finite values | CPU for that call and for the rest of the run, one notice with the reason |
//!
//! The CPU result is always the reference; a GPU result agrees with it within a
//! small tolerance (see the `stacker-gpu` crate), not bit for bit.
//!
//! The backend is a trait ([`Backend`]) so the fallback logic is tested without
//! a GPU, with backends that fail on purpose.

use std::{
    collections::HashSet,
    panic::{AssertUnwindSafe, catch_unwind},
};

use rayon::prelude::*;
use stacker_algo::{
    relief::{multigrid::multigrid_depth as multigrid_depth_cpu, threshold::mask_above},
    strata::{CpuStrata, StrataKernels},
};
use stacker_align::{
    Matrix3,
    device::DeviceHandle,
    transform::{coverage_mask, warp_image, warp_image_clamped},
};
use stacker_core::{
    error::StackerError,
    image::PlanarImage,
    settings::{ComputeDevice, StackingSettings},
};

use crate::fuse;

/// The most device memory one call may need, in bytes.
///
/// wgpu cannot report the size of video memory, so this is a fixed, conservative
/// budget; a call that needs more runs on the CPU instead. A 24 MP `Apex` stack
/// needs about 2.1 GiB, a 24 MP `Relief` stack about 1.5 GiB.
pub const GPU_BUDGET_BYTES: u64 = 3 << 30;

/// Why a backend could not run a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The call is too large for the device, or uses something the kernels do
    /// not cover. Not a failure: other calls may still run there, so the backend
    /// stays in use.
    DoesNotFit(String),
    /// The device or the driver failed. The backend is not used again this run.
    Fault(String),
}

impl BackendError {
    /// [`Self::DoesNotFit`] for a stage this backend has no kernels for.
    #[must_use]
    pub fn unsupported(stage: &str) -> Self {
        Self::DoesNotFit(format!("{stage} is not supported by this backend"))
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DoesNotFit(why) | Self::Fault(why) => f.write_str(why),
        }
    }
}

/// Something other than the CPU code that can run the heavy stages. Every
/// method but [`Self::name`] and [`Self::apex`] has a default that declines, so
/// a backend implements only what it has kernels for.
pub trait Backend: Send {
    /// A short description for logs and the status line.
    fn name(&self) -> String;

    /// Fuse `images` as [`fuse::apex`] does.
    ///
    /// # Errors
    /// [`BackendError::DoesNotFit`] when the stack is too large for the device,
    /// [`BackendError::Fault`] for anything else that went wrong.
    fn apex(
        &mut self,
        images: &[PlanarImage<f32>],
        settings: &StackingSettings,
    ) -> Result<PlanarImage<f32>, BackendError>;

    /// Compute the `Relief` focus measure of every frame of `images` and fold it
    /// into the best frame per pixel; return the best focus per pixel (the
    /// maximum focus map, `width * height` values). The folded stack is kept for
    /// [`Self::relief_finish`].
    ///
    /// # Errors
    /// As [`Self::apex`].
    fn relief_fold(
        &mut self,
        images: &[PlanarImage<f32>],
        estimation_radius: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let _ = (images, estimation_radius);
        Err(BackendError::unsupported("Relief"))
    }

    /// Select and smooth the stack folded by the last [`Self::relief_fold`], as
    /// [`fuse::select_and_smooth`] does with the mask `max focus >= threshold`.
    ///
    /// # Errors
    /// As [`Self::apex`].
    fn relief_finish(
        &mut self,
        threshold: f32,
        smoothing_radius: usize,
    ) -> Result<PlanarImage<f32>, BackendError> {
        let _ = (threshold, smoothing_radius);
        Err(BackendError::unsupported("Relief"))
    }

    /// Warp `frame` by `inverse` (output pixel `p` samples the frame at
    /// `inverse * p`, so `inverse` is the inverse of the matrix `warp_image`
    /// takes) with the bicubic sampler and a black border, or with the clamped
    /// spline (`clamped`); return the frame and its coverage mask.
    ///
    /// # Errors
    /// As [`Self::apex`].
    fn warp(
        &mut self,
        frame: &PlanarImage<f32>,
        inverse: &Matrix3<f32>,
        clamped: bool,
    ) -> Result<(PlanarImage<f32>, Vec<bool>), BackendError> {
        let _ = (frame, inverse, clamped);
        Err(BackendError::unsupported("the warp"))
    }

    /// The kernels `Strata` runs its saliency and guided filters through, or
    /// `None` for the CPU. The returned object owns what it needs (it is `'static`
    /// and `Send`), builds its device plan lazily for the frame size it is first
    /// asked about, and declines (the caller then runs the CPU code) any call it
    /// cannot serve; after a fault it declines every call and reports the fault
    /// through [`Self::take_strata_fault`].
    fn strata_kernels(&mut self) -> Option<Box<dyn StrataKernels>> {
        None
    }

    /// The fault the `Strata` kernels ran into since the last call, if any. They
    /// carried on on the CPU; [`Compute`] gives the backend up.
    fn take_strata_fault(&mut self) -> Option<String> {
        None
    }

    /// The whole-image depth of the `Relief` multigrid solver, as
    /// `stacker_algo::relief::multigrid::multigrid_depth` returns it
    /// (`width * height` values).
    ///
    /// # Errors
    /// As [`Self::apex`].
    fn multigrid_depth(
        &mut self,
        width: usize,
        height: usize,
        index: &[u16],
        weight: &[f32],
        smoothing_radius: usize,
    ) -> Result<Vec<f32>, BackendError> {
        let _ = (width, height, index, weight, smoothing_radius);
        Err(BackendError::unsupported("the multigrid solver"))
    }

    /// The device the alignment refiners evaluate their objective on, if any.
    fn alignment_device(&mut self) -> Option<DeviceHandle> {
        None
    }

    /// The fault the alignment device ran into since the last call, if any. The
    /// refiners carried on on the CPU; [`Compute`] gives the backend up.
    fn take_alignment_fault(&mut self) -> Option<String> {
        None
    }

    /// Release everything after a fault, so a later run starts clean. The default
    /// does nothing.
    fn reset(&mut self) {}
}

/// The handle both apps run the heavy stages through.
pub struct Compute {
    backend: Option<Box<dyn Backend>>,
    /// A message for the user that has not been collected yet.
    notice: Option<String>,
    /// The stages for which a call that did not fit has been noted in the log.
    fit_logged: HashSet<&'static str>,
}

impl Compute {
    /// Always the CPU.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            backend: None,
            notice: None,
            fit_logged: HashSet::new(),
        }
    }

    /// The CPU, with a notice explaining why the GPU is not in use.
    #[must_use]
    pub fn cpu_because(why: &str) -> Self {
        tracing::warn!("{why}");
        Self {
            notice: Some(why.to_owned()),
            ..Self::cpu()
        }
    }

    /// `backend` first, the CPU behind it.
    #[must_use]
    pub fn with_backend(backend: Box<dyn Backend>) -> Self {
        Self {
            backend: Some(backend),
            ..Self::cpu()
        }
    }

    /// The compute for `settings.compute_device`. A request for the GPU that
    /// cannot be met (no GPU support in this build, no adapter) yields the CPU
    /// with a notice; it is never an error.
    #[must_use]
    pub fn for_settings(settings: &StackingSettings) -> Self {
        match settings.compute_device {
            ComputeDevice::Cpu => Self::cpu(),
            ComputeDevice::Gpu => Self::gpu_or_cpu(),
        }
    }

    #[cfg(feature = "gpu")]
    fn gpu_or_cpu() -> Self {
        match gpu::GpuBackend::open() {
            Ok(backend) => {
                tracing::info!("heavy stages on {}", backend.name());
                Self::with_backend(Box::new(backend))
            }
            Err(e) => Self::cpu_because(&format!("No usable GPU ({e}); using the CPU.")),
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn gpu_or_cpu() -> Self {
        Self::cpu_because("This build has no GPU support (the gpu feature); using the CPU.")
    }

    /// What the stages run on right now: `"CPU"` or the backend's name.
    #[must_use]
    pub fn describe(&self) -> String {
        self.backend
            .as_ref()
            .map_or_else(|| "CPU".to_owned(), |b| b.name())
    }

    /// `true` while a GPU backend is in use.
    #[must_use]
    pub fn uses_gpu(&self) -> bool {
        self.backend.is_some()
    }

    /// The message waiting for the user (why the GPU is not in use, or why it was
    /// given up), once; later calls return `None` until something new happens.
    pub const fn take_notice(&mut self) -> Option<String> {
        self.notice.take()
    }

    /// Run `call` on the backend. `None` when there is none or it could not
    /// deliver (a result that fails `trusted` counts as a fault); the caller then
    /// runs the CPU reference.
    fn attempt<T>(
        &mut self,
        stage: &'static str,
        trusted: impl FnOnce(&T) -> bool,
        call: impl FnOnce(&mut dyn Backend) -> Result<T, BackendError>,
    ) -> Option<T> {
        let backend = self.backend.as_mut()?;
        let attempt = catch_unwind(AssertUnwindSafe(|| call(backend.as_mut())));
        match attempt {
            Ok(Ok(value)) if trusted(&value) => Some(value),
            Ok(Ok(_)) => {
                self.give_up(&format!("the GPU returned non-finite values in {stage}"));
                None
            }
            Ok(Err(BackendError::DoesNotFit(why))) => {
                if self.fit_logged.insert(stage) {
                    tracing::info!("{why}; {stage} runs on the CPU for this call");
                }
                None
            }
            Ok(Err(BackendError::Fault(why))) => {
                self.give_up(&why);
                None
            }
            Err(payload) => {
                self.give_up(&format!(
                    "the GPU code panicked: {}",
                    panic_message(payload.as_ref())
                ));
                None
            }
        }
    }

    /// Drop the backend after a fault and leave a notice.
    fn give_up(&mut self, why: &str) {
        tracing::warn!("GPU failed ({why}); using the CPU for the rest of this run");
        if let Some(mut backend) = self.backend.take() {
            // A backend that panics while resetting must not take the run down.
            let _ = catch_unwind(AssertUnwindSafe(|| backend.reset()));
        }
        self.notice = Some(format!(
            "The GPU failed ({why}). The rest of this run uses the CPU."
        ));
    }

    /// `Apex` fusion of `images`: the GPU when this handle has a backend, the CPU
    /// otherwise and whenever the backend cannot deliver. Never fails.
    #[must_use]
    pub fn apex(
        &mut self,
        images: &[PlanarImage<f32>],
        settings: &StackingSettings,
    ) -> PlanarImage<f32> {
        if images.is_empty() {
            return fuse::apex(images, settings);
        }
        self.attempt("Apex", all_finite, |b| b.apex(images, settings))
            .unwrap_or_else(|| fuse::apex(images, settings))
    }

    /// The maximum `Relief` focus map of `images` (`width * height` values), as
    /// `fuse::focus_maps(images, estimation_radius).max.luma`. A tiled caller
    /// builds the whole-image map from these to compute the threshold.
    ///
    /// # Panics
    /// Panics when `images` is empty.
    #[must_use]
    pub fn relief_focus_max(
        &mut self,
        images: &[PlanarImage<f32>],
        estimation_radius: usize,
    ) -> Vec<f32> {
        self.attempt(
            "Relief",
            |m: &Vec<f32>| finite(m),
            |b| b.relief_fold(images, estimation_radius),
        )
        .unwrap_or_else(|| fuse::focus_max_map(images, estimation_radius))
    }

    /// `Relief` fusion of one padded tile against the whole-image `threshold`, as
    /// [`fuse::relief_tile`]. Never fails.
    #[must_use]
    pub fn relief_tile(
        &mut self,
        images: &[PlanarImage<f32>],
        settings: &StackingSettings,
        threshold: f32,
    ) -> PlanarImage<f32> {
        if images.is_empty() {
            return fuse::relief_tile(images, settings, threshold);
        }
        let (radius, smoothing) = relief_radii(settings);
        self.attempt("Relief", all_finite, |b| {
            b.relief_fold(images, radius)?;
            b.relief_finish(threshold, smoothing)
        })
        .unwrap_or_else(|| fuse::relief_tile(images, settings, threshold))
    }

    /// `Relief` fusion of a whole image whose threshold `resolve` decides from the
    /// maximum focus map (the map is in the `luma` plane). `resolve` is called
    /// exactly once; `None` from it (the user cancelled) is passed on.
    ///
    /// With `resolve` returning `fuse::relief_threshold(&map.luma, settings).value`
    /// this is [`fuse::relief`].
    ///
    /// # Panics
    /// Never in practice: the focus map always has one value per pixel.
    pub fn relief(
        &mut self,
        images: &[PlanarImage<f32>],
        settings: &StackingSettings,
        resolve: impl FnOnce(&PlanarImage<f32>) -> Option<f32>,
    ) -> Option<PlanarImage<f32>> {
        if images.is_empty() {
            return Some(PlanarImage::new(0, 0));
        }
        let (width, height) = (images[0].width, images[0].height);
        let (radius, smoothing) = relief_radii(settings);
        let folded = self.attempt(
            "Relief",
            |m: &Vec<f32>| finite(m),
            |b| b.relief_fold(images, radius),
        );
        if let Some(max) = folded {
            let map = PlanarImage::with_luma(width, height, max)
                .expect("the focus map has one value per pixel");
            let threshold = resolve(&map)?;
            return Some(
                self.attempt("Relief", all_finite, |b| {
                    b.relief_finish(threshold, smoothing)
                })
                .unwrap_or_else(|| fuse::relief_tile(images, settings, threshold)),
            );
        }
        if images.len() > fuse::MAX_MULTIGRID_FRAMES {
            // More frames than a `u16` winner index holds: one map per frame.
            let maps = fuse::focus_maps(images, radius);
            let threshold = resolve(&maps.max)?;
            let mask = mask_above(&maps.max.luma, threshold);
            return Some(fuse::select_and_smooth(
                images, &maps.smls, &mask, smoothing,
            ));
        }
        // The CPU streams the focus maps into the best value and the winner
        // (6 B/px) instead of keeping one map per frame.
        let (best, index) = fuse::focus_votes(images, radius).into_parts();
        let max = PlanarImage::with_luma(width, height, best)
            .expect("the focus map has one value per pixel");
        let threshold = resolve(&max)?;
        Some(fuse::select_from_votes(
            images, &max.luma, &index, threshold, smoothing,
        ))
    }

    /// Warp `frame` by `matrix` (as `warp_image(frame, matrix)`, or
    /// `warp_image_clamped` with `clamped`) and return it with its coverage mask
    /// (as `coverage_mask(matrix, ..)`).
    ///
    /// # Errors
    /// [`StackerError::MathError`] when `matrix` is non-finite or not invertible,
    /// as the CPU warp.
    pub fn warp(
        &mut self,
        frame: &PlanarImage<f32>,
        matrix: &Matrix3<f32>,
        clamped: bool,
    ) -> Result<(PlanarImage<f32>, Vec<bool>), StackerError> {
        let inverse = matrix
            .iter()
            .all(|v| v.is_finite())
            .then(|| matrix.try_inverse())
            .flatten();
        if let Some(inverse) = inverse
            && let Some(done) = self.attempt(
                "the warp",
                |(image, _): &(PlanarImage<f32>, Vec<bool>)| all_finite(image),
                |b| b.warp(frame, &inverse, clamped),
            )
        {
            return Ok(done);
        }
        let warped = if clamped {
            warp_image_clamped(frame, matrix)?
        } else {
            warp_image(frame, matrix)?
        };
        Ok((warped, coverage_mask(matrix, frame.width, frame.height)))
    }

    /// The whole-image `Relief` multigrid depth: the GPU solver when this handle
    /// has a backend that serves it, `stacker_algo::relief::multigrid::multigrid_depth`
    /// otherwise and whenever the backend cannot deliver. Never fails.
    ///
    /// `index` holds the winning frame of every pixel and `weight` the confidence
    /// (a positive weight pins the pixel to its frame); `smoothing_radius` is
    /// `relief_smoothing_radius` (the final `pyramid_smooth` runs on the CPU
    /// either way). The result has `width * height` fractional frame indices.
    ///
    /// # Panics
    /// Panics when `index` or `weight` holds fewer than `width * height` samples.
    #[must_use]
    pub fn multigrid_depth(
        &mut self,
        width: usize,
        height: usize,
        index: &[u16],
        weight: &[f32],
        smoothing_radius: usize,
    ) -> Vec<f32> {
        assert!(
            index.len() >= width * height && weight.len() >= width * height,
            "the index and the weight must hold width * height samples"
        );
        if width == 0 || height == 0 {
            return Vec::new();
        }
        self.attempt(
            "Multigrid",
            |depth: &Vec<f32>| finite(depth),
            |b| b.multigrid_depth(width, height, index, weight, smoothing_radius),
        )
        .unwrap_or_else(|| multigrid_depth_cpu(width, height, index, weight, smoothing_radius))
    }

    /// The kernels to run `Strata` through (`stacker_algo::strata::fuse_strata_on`):
    /// the device's saliency and guided-filter kernels when this handle has a
    /// backend that offers them, otherwise `CpuStrata`. Each kernel call that the
    /// device cannot serve (a frame size over the memory budget, a fault) is
    /// answered `None` and the caller runs the CPU code for that call.
    ///
    /// The returned kernels are independent of this handle (they can be moved to
    /// another thread). A fault inside them is reported to this handle only when
    /// [`Self::check_strata_kernels`] is called; call it after the run (or after
    /// each frame) so the GPU is given up and the user gets the notice.
    pub fn strata_kernels(&mut self) -> Box<dyn StrataKernels> {
        self.backend
            .as_mut()
            .and_then(|backend| backend.strata_kernels())
            .unwrap_or_else(|| Box::new(CpuStrata))
    }

    /// Give the backend up if the `Strata` kernels from [`Self::strata_kernels`]
    /// failed since the last check (they carried on on the CPU).
    pub fn check_strata_kernels(&mut self) {
        let fault = self.backend.as_mut().and_then(|b| b.take_strata_fault());
        if let Some(why) = fault {
            self.give_up(&why);
        }
    }

    /// The device the alignment refiners should evaluate their objective on, or
    /// `None` for the CPU.
    pub fn alignment_device(&mut self) -> Option<DeviceHandle> {
        self.backend.as_mut()?.alignment_device()
    }

    /// Give the backend up if the alignment device failed since the last check
    /// (the refiners carried on on the CPU). Call after each aligned frame.
    pub fn check_alignment_device(&mut self) {
        let fault = self.backend.as_mut().and_then(|b| b.take_alignment_fault());
        if let Some(why) = fault {
            self.give_up(&why);
        }
    }
}

/// `relief_estimation_radius` and `relief_smoothing_radius`.
const fn relief_radii(settings: &StackingSettings) -> (usize, usize) {
    (
        settings.relief_estimation_radius as usize,
        settings.relief_smoothing_radius as usize,
    )
}

/// `true` when every value is finite.
fn finite(values: &[f32]) -> bool {
    values.par_iter().all(|v| v.is_finite())
}

/// `true` when every sample of every plane is finite.
fn all_finite(image: &PlanarImage<f32>) -> bool {
    [&image.luma, &image.chroma_a, &image.chroma_b]
        .into_iter()
        .all(|plane| finite(plane))
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}

#[cfg(feature = "gpu")]
pub use gpu::shared_gpu_context;

/// The GPU backend (the `gpu` feature).
#[cfg(feature = "gpu")]
mod gpu {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{
            Arc, Mutex, PoisonError,
            atomic::{AtomicBool, Ordering},
        },
    };

    use stacker_algo::{
        relief::{multigrid::crop_plane, pyramid::pyramid_smooth_owned},
        strata::StrataKernels,
    };
    use stacker_align::{
        Matrix3,
        device::{DeviceHandle, LevelDevice, Moments, ObjectiveDevice, ObjectiveSampler},
    };
    use stacker_core::{image::PlanarImage, settings::StackingSettings};
    use stacker_gpu::{
        GpuContext, GpuError,
        align::{self, ObjectivePlan, Sampler, WarpPlan},
        apex::{ApexPlan, plan_bytes},
        multigrid::{self, MultigridPlan},
        relief::{self, ReliefPlan},
        strata::{self, StrataPlan},
    };

    use super::{Backend, BackendError, GPU_BUDGET_BYTES, finite, panic_message};

    /// The device, opened once per process and shared by every run. Cleared after
    /// a fault so the next run opens a fresh one.
    static CONTEXT: Mutex<Option<Arc<GpuContext>>> = Mutex::new(None);

    /// `Apex` plans kept for reuse: a tiled run fuses many tiles of the same size.
    const MAX_PLANS: usize = 2;

    /// The smallest pyramid level whose objective is evaluated on the device.
    /// Measured on an integrated GPU: a 480x270 level takes 1.5 ms on the CPU and
    /// 0.3 ms on the device, a 240x135 level 0.36 ms and 0.22 ms, so from about
    /// 64k pixels the device wins by more than its upload costs.
    pub const MIN_OBJECTIVE_PIXELS: usize = 1 << 16;

    /// The `Relief` regularisation, as `fuse` uses it.
    const GUIDED_EPS: f32 = 0.01;

    fn open_context() -> Result<Arc<GpuContext>, GpuError> {
        let mut slot = CONTEXT.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(ctx) = slot.as_ref() {
            return Ok(Arc::clone(ctx));
        }
        let ctx = GpuContext::new(None)?;
        *slot = Some(Arc::clone(&ctx));
        drop(slot);
        Ok(ctx)
    }

    fn clear_context() {
        *CONTEXT.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// The shared device, if one has been opened. For tests that inject failures.
    #[doc(hidden)]
    #[must_use]
    pub fn shared_gpu_context() -> Option<Arc<GpuContext>> {
        CONTEXT
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct PlanKey {
        width: usize,
        height: usize,
        levels: usize,
        use_color: bool,
        grit: bool,
    }

    fn classify(error: GpuError) -> BackendError {
        match error {
            GpuError::TooLarge { .. } => BackendError::DoesNotFit(error.to_string()),
            other => BackendError::Fault(other.to_string()),
        }
    }

    fn over_budget(what: &str, width: usize, height: usize, need: u64) -> Option<BackendError> {
        (need > GPU_BUDGET_BYTES).then(|| {
            BackendError::DoesNotFit(format!(
                "a {width}x{height} {what} needs {} MiB of device memory, the budget is {} MiB",
                need >> 20,
                GPU_BUDGET_BYTES >> 20
            ))
        })
    }

    fn same_size(images: &[PlanarImage<f32>]) -> Result<(usize, usize), BackendError> {
        let (width, height) = (images[0].width, images[0].height);
        if images
            .iter()
            .any(|i| (i.width, i.height) != (width, height))
        {
            return Err(BackendError::Fault("the frames differ in size".to_owned()));
        }
        Ok((width, height))
    }

    fn rows(m: &Matrix3<f32>) -> [[f32; 3]; 3] {
        std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))
    }

    /// The first fault the alignment device ran into, shared by the device and
    /// every level it prepared.
    #[derive(Default)]
    struct FaultSlot(Mutex<Option<String>>);

    impl FaultSlot {
        fn failed(&self) -> bool {
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .is_some()
        }

        fn record(&self, why: String) {
            tracing::warn!("GPU alignment objective failed ({why}); the CPU takes over");
            self.0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(why);
        }

        fn take(&self) -> Option<String> {
            self.0.lock().unwrap_or_else(PoisonError::into_inner).take()
        }
    }

    /// The registration objective on the device.
    struct GpuObjective {
        ctx: Arc<GpuContext>,
        fault: Arc<FaultSlot>,
    }

    impl ObjectiveDevice for GpuObjective {
        fn level(
            &self,
            reference: &[f32],
            source: &[f32],
            width: usize,
            height: usize,
            sampler: ObjectiveSampler,
        ) -> Option<Box<dyn LevelDevice>> {
            if self.fault.failed()
                || width * height < MIN_OBJECTIVE_PIXELS
                || align::objective_bytes(width, height) > GPU_BUDGET_BYTES
            {
                return None;
            }
            let sampler = match sampler {
                ObjectiveSampler::Bicubic => Sampler::Bicubic,
                ObjectiveSampler::SplineClamped => Sampler::SplineClamped,
            };
            let built = catch_unwind(AssertUnwindSafe(|| {
                ObjectivePlan::new(&self.ctx, reference, source, width, height, sampler)
            }));
            match built {
                Ok(Ok(plan)) => Some(Box::new(GpuLevel {
                    plan,
                    fault: Arc::clone(&self.fault),
                })),
                Ok(Err(GpuError::TooLarge { .. })) => None,
                Ok(Err(e)) => {
                    self.fault.record(e.to_string());
                    None
                }
                Err(_) => {
                    self.fault
                        .record("the GPU objective panicked while preparing a level".to_owned());
                    None
                }
            }
        }
    }

    struct GpuLevel {
        plan: ObjectivePlan,
        fault: Arc<FaultSlot>,
    }

    impl LevelDevice for GpuLevel {
        fn moments(&mut self, inverse: &Matrix3<f32>) -> Option<Moments> {
            if self.fault.failed() {
                return None;
            }
            let inverse = rows(inverse);
            let result = catch_unwind(AssertUnwindSafe(|| self.plan.moments(&inverse)));
            match result {
                Ok(Ok(m)) if m.mean.is_finite() && m.m2.is_finite() => Some(Moments {
                    count: m.count,
                    mean: m.mean,
                    m2: m.m2,
                }),
                Ok(Ok(_)) => {
                    self.fault
                        .record("the GPU objective returned non-finite moments".to_owned());
                    None
                }
                Ok(Err(e)) => {
                    self.fault.record(e.to_string());
                    None
                }
                Err(_) => {
                    self.fault.record("the GPU objective panicked".to_owned());
                    None
                }
            }
        }
    }

    /// What the backend and the `Strata` kernels it handed out share: whether the
    /// kernels may still use the device, and the first fault they ran into.
    #[derive(Default)]
    struct StrataState {
        /// Set by the first fault and when the backend is given up; never cleared.
        dead: AtomicBool,
        fault: Mutex<Option<String>>,
    }

    impl StrataState {
        fn is_dead(&self) -> bool {
            self.dead.load(Ordering::SeqCst)
        }

        /// The kernels cannot be trusted any more: remember why (the first time).
        fn kill(&self, why: String) {
            tracing::warn!("GPU Strata kernel failed ({why}); the CPU takes over");
            self.dead.store(true, Ordering::SeqCst);
            self.fault
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(why);
        }

        fn take(&self) -> Option<String> {
            self.fault
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
        }
    }

    /// The `Strata` saliency and guided-filter kernels on the device. It owns a
    /// plan for the frame size it was last asked about, so one `Strata` run builds
    /// one plan; every call it cannot serve answers `None`.
    struct GpuStrata {
        ctx: Arc<GpuContext>,
        plan: Option<StrataPlan>,
        /// A frame size that was declined (over the budget or the device's
        /// limits), so it is not tried again for every frame.
        declined: Option<(usize, usize)>,
        state: Arc<StrataState>,
    }

    impl GpuStrata {
        /// The plan for `width x height`, built on first use; `None` when the
        /// device cannot serve that size or has failed.
        fn plan_for(&mut self, width: usize, height: usize) -> Option<&mut StrataPlan> {
            if self.state.is_dead() || width * height == 0 || self.declined == Some((width, height))
            {
                return None;
            }
            if self
                .plan
                .as_ref()
                .is_none_or(|plan| plan.size() != (width, height))
            {
                self.plan = None;
                if strata::plan_bytes(width, height) > GPU_BUDGET_BYTES {
                    tracing::info!(
                        "a {width}x{height} Strata frame needs more device memory than the budget; it runs on the CPU"
                    );
                    self.declined = Some((width, height));
                    return None;
                }
                let built = catch_unwind(AssertUnwindSafe(|| {
                    StrataPlan::new(&self.ctx, width, height)
                }));
                match built {
                    Ok(Ok(plan)) => self.plan = Some(plan),
                    Ok(Err(GpuError::TooLarge { .. })) => {
                        self.declined = Some((width, height));
                        return None;
                    }
                    Ok(Err(e)) => {
                        self.state.kill(e.to_string());
                        return None;
                    }
                    Err(payload) => {
                        self.state.kill(format!(
                            "the GPU code panicked: {}",
                            panic_message(payload.as_ref())
                        ));
                        return None;
                    }
                }
            }
            self.plan.as_mut()
        }

        /// Run `call` on the plan for `width x height`. `None` when the plan is not
        /// available, or the call failed, panicked or returned values that are not
        /// finite (all of which kill the kernels, so the rest of the run is CPU).
        fn run<T>(
            &mut self,
            (width, height): (usize, usize),
            call: impl FnOnce(&mut StrataPlan) -> Result<T, GpuError>,
            trusted: impl FnOnce(&T) -> bool,
        ) -> Option<T> {
            let plan = self.plan_for(width, height)?;
            let result = catch_unwind(AssertUnwindSafe(|| call(plan)));
            match result {
                Ok(Ok(value)) if trusted(&value) => Some(value),
                Ok(Ok(_)) => {
                    self.state
                        .kill("the GPU returned non-finite values in Strata".to_owned());
                    None
                }
                Ok(Err(e)) => {
                    self.state.kill(e.to_string());
                    None
                }
                Err(payload) => {
                    self.state.kill(format!(
                        "the GPU code panicked: {}",
                        panic_message(payload.as_ref())
                    ));
                    None
                }
            }
        }
    }

    impl StrataKernels for GpuStrata {
        fn saliency(&mut self, luma: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
            if luma.len() != width * height {
                return None;
            }
            self.run(
                (width, height),
                |plan| plan.saliency(luma),
                |map: &Vec<f32>| finite(map),
            )
        }

        fn guided_pair(
            &mut self,
            guidance: &[f32],
            src: &[f32],
            width: usize,
            height: usize,
            radius_a: usize,
            eps_a: f32,
            radius_b: usize,
            eps_b: f32,
        ) -> Option<(Vec<f32>, Vec<f32>)> {
            if guidance.len() != width * height || src.len() != width * height {
                return None;
            }
            self.run(
                (width, height),
                |plan| plan.guided_pair(guidance, src, (radius_a, eps_a), (radius_b, eps_b)),
                |(first, second): &(Vec<f32>, Vec<f32>)| finite(first) && finite(second),
            )
        }
    }

    /// The heavy stages on the GPU through [`stacker_gpu`].
    pub struct GpuBackend {
        ctx: Arc<GpuContext>,
        apex_plans: Vec<(PlanKey, ApexPlan)>,
        relief_plan: Option<((usize, usize), ReliefPlan)>,
        warp_plan: Option<((usize, usize), WarpPlan)>,
        objective: Arc<GpuObjective>,
        strata: Arc<StrataState>,
    }

    impl GpuBackend {
        /// Open the best adapter (or reuse the one already open).
        ///
        /// # Errors
        /// [`GpuError::NoAdapter`] or [`GpuError::Device`] when there is none.
        pub fn open() -> Result<Self, GpuError> {
            let ctx = open_context()?;
            Ok(Self {
                objective: Arc::new(GpuObjective {
                    ctx: Arc::clone(&ctx),
                    fault: Arc::new(FaultSlot::default()),
                }),
                ctx,
                apex_plans: Vec::new(),
                relief_plan: None,
                warp_plan: None,
                strata: Arc::new(StrataState::default()),
            })
        }

        /// Free the plans of the other stages before allocating one for `keep`,
        /// so device memory holds one stage's plans at a time.
        fn make_room(&mut self, keep: &str) {
            if keep != "apex" {
                self.apex_plans.clear();
            }
            if keep != "relief" {
                self.relief_plan = None;
            }
            if keep != "warp" {
                self.warp_plan = None;
            }
        }
    }

    impl Backend for GpuBackend {
        fn name(&self) -> String {
            let info = self.ctx.info();
            format!("GPU ({}, {:?})", info.name, info.backend)
        }

        fn apex(
            &mut self,
            images: &[PlanarImage<f32>],
            settings: &StackingSettings,
        ) -> Result<PlanarImage<f32>, BackendError> {
            let (width, height) = same_size(images)?;
            let key = PlanKey {
                width,
                height,
                levels: (settings.pyramid_levels as usize).max(2),
                use_color: settings.use_all_color_channels,
                grit: settings.grit_suppression,
            };
            if let Some(e) = over_budget(
                "Apex stack",
                width,
                height,
                plan_bytes(width, height, key.levels),
            ) {
                return Err(e);
            }
            self.make_room("apex");
            let at = if let Some(at) = self.apex_plans.iter().position(|(k, _)| *k == key) {
                at
            } else {
                // Free the oldest plan before allocating the new one, so the peak
                // stays at one plan plus the new one.
                if self.apex_plans.len() >= MAX_PLANS {
                    self.apex_plans.remove(0);
                }
                let plan = ApexPlan::new(
                    &self.ctx,
                    width,
                    height,
                    key.levels,
                    key.use_color,
                    key.grit,
                )
                .map_err(classify)?;
                self.apex_plans.push((key, plan));
                self.apex_plans.len() - 1
            };
            let plan = &mut self.apex_plans[at].1;
            let frames = u32::try_from(images.len())
                .map_err(|_| BackendError::Fault("too many frames".to_owned()))?;
            plan.begin(frames);
            for frame in images {
                plan.fold(frame).map_err(classify)?;
            }
            plan.finish().map_err(classify)
        }

        fn relief_fold(
            &mut self,
            images: &[PlanarImage<f32>],
            estimation_radius: usize,
        ) -> Result<Vec<f32>, BackendError> {
            let (width, height) = same_size(images)?;
            if let Some(e) = over_budget(
                "Relief stack",
                width,
                height,
                relief::plan_bytes(width, height),
            ) {
                return Err(e);
            }
            if estimation_radius > relief::MAX_RADIUS {
                return Err(BackendError::DoesNotFit(format!(
                    "the GPU Relief covers focus radii up to {}, not {estimation_radius}",
                    relief::MAX_RADIUS
                )));
            }
            self.make_room("relief");
            if self
                .relief_plan
                .as_ref()
                .is_none_or(|(size, _)| *size != (width, height))
            {
                self.relief_plan = None;
                let plan = ReliefPlan::new(&self.ctx, width, height).map_err(classify)?;
                self.relief_plan = Some(((width, height), plan));
            }
            let plan = &mut self.relief_plan.as_mut().expect("created above").1;
            let frames = u32::try_from(images.len())
                .map_err(|_| BackendError::Fault("too many frames".to_owned()))?;
            plan.begin(frames, estimation_radius).map_err(classify)?;
            for frame in images {
                plan.fold(frame).map_err(classify)?;
            }
            plan.focus_max().map_err(classify)
        }

        fn relief_finish(
            &mut self,
            threshold: f32,
            smoothing_radius: usize,
        ) -> Result<PlanarImage<f32>, BackendError> {
            let (_, plan) = self
                .relief_plan
                .as_mut()
                .ok_or_else(|| BackendError::Fault("no Relief stack has been folded".to_owned()))?;
            plan.finish(threshold, smoothing_radius, GUIDED_EPS)
                .map_err(classify)
        }

        fn warp(
            &mut self,
            frame: &PlanarImage<f32>,
            inverse: &Matrix3<f32>,
            clamped: bool,
        ) -> Result<(PlanarImage<f32>, Vec<bool>), BackendError> {
            let (width, height) = (frame.width, frame.height);
            if let Some(e) = over_budget("warp", width, height, align::warp_bytes(width, height)) {
                return Err(e);
            }
            self.make_room("warp");
            if self
                .warp_plan
                .as_ref()
                .is_none_or(|(size, _)| *size != (width, height))
            {
                self.warp_plan = None;
                let plan = WarpPlan::new(&self.ctx, width, height).map_err(classify)?;
                self.warp_plan = Some(((width, height), plan));
            }
            let plan = &mut self.warp_plan.as_mut().expect("created above").1;
            let sampler = if clamped {
                Sampler::SplineClamped
            } else {
                Sampler::Bicubic
            };
            plan.warp(frame, &rows(inverse), sampler).map_err(classify)
        }

        fn strata_kernels(&mut self) -> Option<Box<dyn StrataKernels>> {
            if self.strata.is_dead() {
                return None;
            }
            Some(Box::new(GpuStrata {
                ctx: Arc::clone(&self.ctx),
                plan: None,
                declined: None,
                state: Arc::clone(&self.strata),
            }))
        }

        fn take_strata_fault(&mut self) -> Option<String> {
            self.strata.take()
        }

        fn multigrid_depth(
            &mut self,
            width: usize,
            height: usize,
            index: &[u16],
            weight: &[f32],
            smoothing_radius: usize,
        ) -> Result<Vec<f32>, BackendError> {
            if let Some(e) = over_budget(
                "multigrid solve",
                width,
                height,
                multigrid::plan_bytes(width, height),
            ) {
                return Err(e);
            }
            self.make_room("multigrid");
            // The pins go up once, the whole cycle runs on the device, the finest
            // level comes back once; the plan is not kept (the next image has other
            // pins). The final smoothing is the CPU's, on the padded plane, as in
            // `stacker_algo::relief::multigrid::multigrid_depth`.
            let mut plan =
                MultigridPlan::new(&self.ctx, width, height, index, weight).map_err(classify)?;
            let (padded, padded_width, padded_height) = plan.solve_padded().map_err(classify)?;
            drop(plan);
            let smoothed =
                pyramid_smooth_owned(padded, padded_width, padded_height, smoothing_radius);
            Ok(crop_plane(&smoothed, padded_width, (0, 0), (width, height)))
        }

        fn alignment_device(&mut self) -> Option<DeviceHandle> {
            if self.objective.fault.failed() {
                return None;
            }
            Some(DeviceHandle(
                Arc::clone(&self.objective) as Arc<dyn ObjectiveDevice>
            ))
        }

        fn take_alignment_fault(&mut self) -> Option<String> {
            self.objective.fault.take()
        }

        fn reset(&mut self) {
            self.apex_plans.clear();
            self.relief_plan = None;
            self.warp_plan = None;
            // The Strata kernels already handed out must not touch the device again.
            self.strata.dead.store(true, Ordering::SeqCst);
            clear_context();
        }
    }
}
