//! Shared `wgpu` context acquisition, gated behind the `gpu` feature.
//!
//! `z-stackr-algo`'s and `z-stackr-align`'s own `gpu` features both enable
//! this crate's `gpu` feature and call [`context`] instead of each
//! maintaining their own copy of the adapter/device/queue acquisition
//! boilerplate — both crates want the exact same high-performance-preference,
//! no-compatible-surface adapter request, so a single shared definition
//! removes any risk of the two drifting apart.
//!
//! # Fallback contract
//!
//! [`context`] returns `None` when no adapter could be acquired (no GPU
//! present, no supported backend, driver issue, etc.) — it never panics.
//! Every GPU compute path in this workspace (`z-stackr-algo::apex::gpu`,
//! `z-stackr-align::transform::gpu`) is required to treat a `None` here as
//! "fall back to the CPU/SIMD implementation", not as an error to propagate,
//! per `docs/gpu_acceleration_summary.md`'s fallback rule.
//!
//! # Why a shared `&'static` pair, not `Clone`
//!
//! `wgpu::Device`/`wgpu::Queue` in the pinned `wgpu = "30.0.0"` are cheap
//! `Arc`-backed handles and (as of this version) do implement `Clone`, but
//! there is no need to clone them at all: the context lives in a
//! process-wide [`OnceLock`] for the process's whole lifetime, so every
//! caller can simply borrow `&'static Device`/`&'static Queue` from
//! [`context`]'s returned reference. This sidesteps the question entirely
//! (no clone, no refcount bump per call) rather than relying on `Device`/
//! `Queue` happening to be `Clone` in this particular `wgpu` release.
#[cfg(feature = "gpu")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "gpu")]
use std::sync::{Mutex, OnceLock};

/// Process-wide runtime GPU enable/disable switch, default **on**.
///
/// This is deliberately independent of the `gpu` Cargo feature: the feature
/// controls whether any `wgpu`/GPU code is *compiled in at all* (a default,
/// non-`gpu` build has zero `wgpu` dependency and this flag has nothing to
/// gate); this flag controls, **within a `gpu`-featured build**, whether
/// [`context`] is allowed to hand out the adapter it already has. Flipping
/// it to `false` makes every GPU dispatch site in the workspace
/// (`stacker_algo::apex::gpu`, `stacker_algo::strata::gpu`,
/// `stacker_algo::relief::gpu`, `stacker_align::transform::gpu`) fall back
/// to its CPU/rayon implementation on the very next call, with no call-site
/// changes needed — every one of them already treats `context() == None` as
/// "fall back to CPU", which is exactly what a disabled switch produces.
///
/// Not present at all in a non-`gpu` build (the whole point of the switch
/// is moot when there is no GPU code to disable), so
/// `StackingSettings::use_gpu` is harmlessly ignored by such a build — see
/// that field's docs.
#[cfg(feature = "gpu")]
static GPU_ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable the GPU path workspace-wide.
///
/// See [`GPU_ENABLED`]'s docs. Takes effect on the very next [`context`]
/// call — it does not tear down an already-acquired adapter/device, it only
/// hides it from callers while disabled, so re-enabling after a disable is
/// instant (no re-init, no re-log).
///
/// Callers (CLI/GUI/Python, all behind their own `#[cfg(feature = "gpu")]`)
/// call this once per run, before any stacking work starts, with
/// `StackingSettings::use_gpu`.
#[cfg(feature = "gpu")]
pub fn set_enabled(enabled: bool) {
    GPU_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns the current runtime GPU enable/disable switch state.
///
/// See [`set_enabled`]. Independent of, and checked *in addition to*,
/// actual adapter availability — both must hold for [`context`] to return
/// `Some`.
#[cfg(feature = "gpu")]
#[must_use]
pub fn enabled() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}

/// Lazily-initialised, process-wide `wgpu` adapter/device/queue pair.
///
/// `None` once initialisation has been attempted and failed (no adapter
/// available) — the failure is cached too, so a GPU-less host doesn't retry
/// (and re-log) adapter acquisition on every fusion/warp call.
#[cfg(feature = "gpu")]
static WGPU_CTX: OnceLock<Option<(wgpu::Device, wgpu::Queue)>> = OnceLock::new();

/// Acquire (or return the cached) process-wide `wgpu` context.
///
/// Requests a high-performance adapter with no compatible surface (this
/// workspace only ever does off-screen compute, never rendering to a
/// window), then a device with default limits/features. Returns `None` on
/// any failure — no adapter, no device, or the request otherwise erroring —
/// so every call site can implement a plain "try GPU, else CPU" branch
/// without matching on a specific error type. Also returns `None`
/// (regardless of adapter availability) whenever the runtime switch
/// ([`set_enabled`]) is off — checked first, before touching the `OnceLock`,
/// so disabling the switch never forces an eager adapter probe on a host
/// that hadn't done one yet.
///
/// The very first successful acquisition logs the selected backend once
/// (`tracing::info!`) so a run's logs make it obvious whether the GPU path
/// actually engaged; failures log once at `debug` level (expected and
/// silent-by-default on the common GPU-less CI/dev-container host).
#[cfg(feature = "gpu")]
#[must_use]
pub fn context() -> Option<&'static (wgpu::Device, wgpu::Queue)> {
    if !enabled() {
        return None;
    }
    WGPU_CTX
        .get_or_init(|| {
            pollster::block_on(async {
                let instance = wgpu::Instance::new(
                    wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
                );
                let adapter = match instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                        apply_limit_buckets: false,
                    })
                    .await
                {
                    Ok(adapter) => adapter,
                    Err(err) => {
                        tracing::debug!(error = %err, "gpu: no wgpu adapter available, falling back to CPU");
                        return None;
                    }
                };
                let info = adapter.get_info();
                match adapter
                    .request_device(&wgpu::DeviceDescriptor::default())
                    .await
                {
                    Ok((device, queue)) => {
                        // Downgrade uncaptured wgpu errors from the library
                        // default (panic) to a log line: every GPU entry
                        // point in this workspace wraps its work in a
                        // validation error scope and falls back to CPU, so
                        // anything still reaching the uncaptured handler is
                        // a stray late error that must not abort the
                        // process (the GUI runs fusion on worker threads
                        // whose panics would take the whole app down).
                        device.on_uncaptured_error(std::sync::Arc::new(|e| {
                            tracing::warn!(
                                error = %e,
                                "gpu: uncaptured wgpu error (CPU fallback paths handle recovery)"
                            );
                        }));
                        tracing::info!(
                            backend = ?info.backend,
                            adapter = %info.name,
                            "gpu: wgpu context acquired, GPU compute paths engaged"
                        );
                        Some((device, queue))
                    }
                    Err(err) => {
                        tracing::debug!(error = %err, "gpu: wgpu device request failed, falling back to CPU");
                        None
                    }
                }
            })
        })
        .as_ref()
}

/// Returns `true` if a `wgpu` context is available on this host.
///
/// Acquires and caches it if this is the first call. Convenience wrapper
/// around [`context`] for call sites that only need the yes/no answer (e.g.
/// tests that skip the GPU-vs-CPU parity comparison when there is no
/// adapter).
#[cfg(feature = "gpu")]
#[must_use]
pub fn is_available() -> bool {
    context().is_some()
}

/// Serializes every GPU dispatch's "push validation error scope → build
/// resources → submit → pop error scope (→ map/poll a readback buffer)"
/// critical section across threads.
///
/// [`context`] hands out the *same* process-wide `&'static Device` to every
/// caller (see this module's docs on why it is a shared singleton rather
/// than cloned per call site). `wgpu::Device::push_error_scope`/
/// `pop_error_scope` form a per-device STACK: if two threads each push a
/// scope, dispatch, and pop around their own (unrelated) work concurrently,
/// the pops can interleave with the pushes from a DIFFERENT thread's
/// in-flight dispatch — one thread's `pop_error_scope` can silently consume
/// another thread's pushed scope (or vice versa), so a real validation
/// error on one thread's dispatch can be reported as "no error" to another
/// thread's call (or a clean dispatch can spuriously report a stray error
/// that actually belonged to someone else's work). This is exactly the kind
/// of nondeterministic, hard-to-reproduce failure that surfaced when
/// multiple GPU-dispatching tests in the same test binary began running
/// concurrently (`cargo test`'s default per-binary thread pool): calls that
/// pass in isolation intermittently returned `None`/`false` from otherwise
/// correct dispatch code under concurrent load.
///
/// Every GPU dispatch call site (`stacker_algo::apex::gpu`,
/// `stacker_algo::apex::gpu::accumulator`, `stacker_algo::relief::gpu`,
/// `stacker_algo::strata::gpu`, `stacker_align::transform::gpu`) must hold
/// this lock for the full duration of its push/dispatch/submit/pop
/// sequence (and any synchronous readback that logically belongs to the
/// same dispatch), released once the result is fully read back to the CPU
/// (or the failure is fully handled) — never held across a fallback to
/// CPU-only work, which does not touch the device at all.
#[cfg(feature = "gpu")]
static GPU_DISPATCH_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the process-wide GPU dispatch lock.
///
/// See [`GPU_DISPATCH_LOCK`]'s docs for why every
/// push-error-scope/dispatch/pop-error-scope critical section in this
/// workspace must hold this guard for its full duration.
///
/// Returns the guard directly (not a `Result`) — a poisoned lock (a panic
/// while some other thread held it) is recovered via `into_inner`, since a
/// GPU dispatch failure must never cascade into poisoning every subsequent
/// caller's ability to try the GPU path at all; every call site already
/// treats any GPU failure as "fall back to CPU", so running with
/// potentially-torn (but still logically consistent, since `()` carries no
/// data) lock state after a recovered poison is safe.
#[cfg(feature = "gpu")]
pub fn dispatch_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_DISPATCH_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Pad a row's byte stride up to the next multiple of
/// `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256), as `copy_texture_to_buffer`
/// requires.
///
/// `unpadded_bytes_per_row` is the tight row size (e.g. `width * 16` for an
/// `Rgba32Float` row); the returned value is always `>= unpadded_bytes_per_row`
/// and a multiple of 256. Both GPU modules (`z-stackr-algo::apex::gpu`,
/// `z-stackr-align::transform::gpu`) use this instead of passing the
/// unpadded row size directly to `wgpu::TexelCopyBufferLayout::bytes_per_row`,
/// which panics/validation-errors at runtime for any width not divisible by
/// 16.
#[cfg(feature = "gpu")]
#[must_use]
pub const fn padded_bytes_per_row(unpadded_bytes_per_row: u32) -> u32 {
    const ALIGN: u32 = 256;
    unpadded_bytes_per_row.next_multiple_of(ALIGN)
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::padded_bytes_per_row;

    #[test]
    fn padded_bytes_per_row_rounds_up_to_256() {
        assert_eq!(padded_bytes_per_row(256), 256);
        assert_eq!(padded_bytes_per_row(257), 512);
        assert_eq!(padded_bytes_per_row(1), 256);
        assert_eq!(padded_bytes_per_row(0), 0);
        // width=100 * 16 bytes/px (Rgba32Float) = 1600, a width not itself
        // a multiple of 16 (100 % 16 != 0); the byte-stride padding must
        // still round up to a 256 multiple.
        assert_eq!(padded_bytes_per_row(100 * 16), 1792);
    }

    #[test]
    fn context_never_panics_on_a_gpu_less_host() {
        // The whole point of `context()` is that it is safe to call on a
        // host with no GPU at all — it must return `None`, not panic.
        let _ = super::context();
    }

    #[test]
    fn set_enabled_false_forces_context_to_none() {
        // Regardless of whether an adapter is actually available on this
        // host, disabling the runtime switch must make `context()` return
        // `None` — every GPU dispatch site's fallback contract depends on
        // this. Restore the default (`true`) afterwards so this test
        // doesn't leak state into other tests in the same process (the
        // switch is a process-wide `static`).
        super::set_enabled(false);
        assert!(!super::enabled());
        assert!(super::context().is_none());
        super::set_enabled(true);
        assert!(super::enabled());
    }
}
