//! Adapter listing and device creation.

use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicU8, Ordering},
};

use crate::GpuError;

/// Where a test can make the next GPU call fail, to prove the caller's fallback.
///
/// Hidden from the documentation: it is a test hook, not part of the contract.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    /// Building the next plan fails.
    Plan,
    /// Folding the next frame (`Apex`, `Relief`) fails, or the next `Strata`
    /// kernel call or multigrid solve.
    Fold,
    /// Reading the result back fails.
    ReadBack,
}

impl Failure {
    const fn code(self) -> u8 {
        match self {
            Self::Plan => 1,
            Self::Fold => 2,
            Self::ReadBack => 3,
        }
    }
}

/// One adapter the system offers, as shown to the user.
#[derive(Debug, Clone)]
pub struct AdapterSummary {
    /// Position in the enumeration; pass it to [`GpuContext::new`] to pick it.
    pub index: usize,
    /// The driver's name for the adapter.
    pub name: String,
    /// The graphics API it is reached through (Vulkan, Dx12, ...).
    pub backend: String,
    /// Discrete, integrated, virtual, CPU or other.
    pub device_type: String,
}

/// A device and its queue, shared by every plan built on it.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    /// Device errors wgpu reported (validation, out of memory, device lost).
    /// wgpu panics on them by default; here they are collected so the caller can
    /// treat them as a failure of the GPU path and carry on with the CPU.
    errors: Arc<Mutex<Vec<String>>>,
    /// A [`Failure`] armed by a test, or 0.
    armed: AtomicU8,
}

/// Lower is better: a discrete GPU over an integrated one over anything else.
#[cfg(not(target_arch = "wasm32"))]
const fn device_rank(kind: wgpu::DeviceType) -> u8 {
    match kind {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::VirtualGpu => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

/// Lower is better: Vulkan, then Direct3D 12, then the rest.
#[cfg(not(target_arch = "wasm32"))]
const fn backend_rank(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Dx12 => 1,
        wgpu::Backend::Metal => 2,
        _ => 3,
    }
}

/// Debug and validation layers only on request (`WGPU_DEBUG=1`,
/// `WGPU_VALIDATION=1`). The build-config default turns them on in debug
/// builds, and the D3D12 layer then logs through `tracing` during process
/// exit, after its thread-locals are gone: one caught panic per message.
#[cfg(not(target_arch = "wasm32"))]
fn enumerate() -> Vec<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        flags: wgpu::InstanceFlags::empty().with_env(),
        ..wgpu::InstanceDescriptor::default()
    });
    instance.enumerate_adapters(wgpu::Backends::all())
}

/// The adapter at `adapter_index`, or the best one when `None`.
#[cfg(not(target_arch = "wasm32"))]
fn pick_adapter(adapter_index: Option<usize>) -> Result<wgpu::Adapter, GpuError> {
    let mut adapters = enumerate();
    Ok(match adapter_index {
        Some(index) if index < adapters.len() => adapters.swap_remove(index),
        Some(index) => {
            return Err(GpuError::NoAdapter(format!(
                "index {index} of {}",
                adapters.len()
            )));
        }
        // OpenGL adapters are never picked automatically: they lack the
        // compute features the kernels need. One can still be chosen by index.
        None => adapters
            .into_iter()
            .filter(|a| {
                matches!(
                    a.get_info().backend,
                    wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal
                )
            })
            .min_by_key(|a| {
                let info = a.get_info();
                (device_rank(info.device_type), backend_rank(info.backend))
            })
            .ok_or_else(|| GpuError::NoAdapter("none enumerated".to_owned()))?,
    })
}

impl GpuContext {
    /// Every adapter the system offers, in enumeration order.
    #[cfg(not(target_arch = "wasm32"))]
    #[must_use]
    pub fn adapters() -> Vec<AdapterSummary> {
        enumerate()
            .iter()
            .enumerate()
            .map(|(index, adapter)| {
                let info = adapter.get_info();
                AdapterSummary {
                    index,
                    name: info.name,
                    backend: format!("{:?}", info.backend),
                    device_type: format!("{:?}", info.device_type),
                }
            })
            .collect()
    }

    /// Open the adapter at `adapter_index`, or the best one when `None`: a
    /// discrete GPU before an integrated one, Vulkan before Direct3D 12, a software
    /// adapter only when nothing else exists, and never an OpenGL adapter.
    ///
    /// The device is created with the adapter's own buffer limits, because the
    /// defaults (128 MiB per storage binding) are far below one 24 MP frame.
    ///
    /// # Errors
    /// [`GpuError::NoAdapter`] when there is no such adapter, [`GpuError::Device`]
    /// when the adapter refuses to open.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(adapter_index: Option<usize>) -> Result<Arc<Self>, GpuError> {
        pollster::block_on(Self::open(pick_adapter(adapter_index)?))
    }

    /// Open the best adapter without blocking: the future can be driven by any
    /// executor, which is the only way to open a device in a browser.
    ///
    /// On `wasm32` this asks the browser for a high-performance WebGPU adapter; on
    /// other targets it picks the adapter [`GpuContext::new`] picks for `None`.
    ///
    /// # Errors
    /// [`GpuError::NoAdapter`] when there is no adapter (the browser offers no
    /// WebGPU), [`GpuError::Device`] when the adapter refuses to open.
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::future_not_send,
            reason = "WebGPU handles are not Send; the worker is single-threaded"
        )
    )]
    pub async fn new_async() -> Result<Arc<Self>, GpuError> {
        #[cfg(target_arch = "wasm32")]
        let adapter = {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                flags: wgpu::InstanceFlags::empty(),
                ..wgpu::InstanceDescriptor::default()
            });
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok_or_else(|| GpuError::NoAdapter("the browser offers no WebGPU".to_owned()))?
        };
        #[cfg(not(target_arch = "wasm32"))]
        let adapter = pick_adapter(None)?;
        Self::open(adapter).await
    }

    /// Open a device on `adapter` with the adapter's own buffer limits, collect
    /// its errors and report it lost.
    #[cfg_attr(
        target_arch = "wasm32",
        allow(
            clippy::arc_with_non_send_sync,
            reason = "a browser device lives on one thread; the Arc shares it between plans"
        )
    )]
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            clippy::future_not_send,
            reason = "WebGPU handles are not Send; the worker is single-threaded"
        )
    )]
    async fn open(adapter: wgpu::Adapter) -> Result<Arc<Self>, GpuError> {
        let info = adapter.get_info();
        let supported = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: supported.max_buffer_size,
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("z-stackr"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits.clone(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&errors);
        device.on_uncaptured_error(Box::new(move |error| {
            sink.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(error.to_string());
        }));
        let sink = Arc::clone(&errors);
        device.set_device_lost_callback(move |reason, message| {
            // Dropping the device on shutdown reports itself as lost; that is not
            // a failure.
            if !matches!(reason, wgpu::DeviceLostReason::Destroyed) {
                sink.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(format!("device lost ({reason:?}): {message}"));
            }
        });
        tracing::info!(
            adapter = info.name.as_str(),
            backend = ?info.backend,
            "GPU device opened"
        );
        Ok(Arc::new(Self {
            device,
            queue,
            info,
            limits,
            errors,
            armed: AtomicU8::new(0),
        }))
    }

    /// The device errors reported since the last call, joined into one message,
    /// or `None`. A plan checks this after building and after reading back; any
    /// error means the result cannot be trusted.
    #[must_use]
    pub fn take_error(&self) -> Option<String> {
        let mut errors = self.errors.lock().unwrap_or_else(PoisonError::into_inner);
        if errors.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut *errors).join("; "))
        }
    }

    /// Make the next GPU call at `at` fail once. For tests of the fallback.
    #[doc(hidden)]
    pub fn inject_failure(&self, at: Failure) {
        self.armed.store(at.code(), Ordering::SeqCst);
    }

    /// `true` once, when a failure armed for `at` is due; disarms it.
    pub(crate) fn injected(&self, at: Failure) -> bool {
        self.armed
            .compare_exchange(at.code(), 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    /// What the driver reports about the adapter in use.
    #[must_use]
    pub const fn info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// The limits the device was created with.
    #[must_use]
    pub const fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }
}
