//! The Worker's WebGPU: the device it opens at start-up and the alignment backend
//! built on it.
//!
//! In the browser alignment runs on the GPU only (the objective of every pyramid
//! level and the final warp); fusion, the cull and decoding stay on the CPU. A
//! browser cannot block on a GPU read-back, so the backend answers through the
//! asynchronous API of `z-stackr-gpu` and is driven by
//! `stacker_pipeline::session::align_frames_async`.

use std::{cell::RefCell, future::Future, sync::Arc};

use stacker_align::{
    Matrix3,
    device::{AsyncLevelDevice, Moments, ObjectiveSampler},
};
use stacker_core::image::PlanarImage;
use stacker_gpu::{
    GpuContext,
    align::{ObjectivePlan, Sampler, WarpPlan},
};
use stacker_pipeline::align::AsyncAlignBackend;
use wasm_bindgen::JsValue;

/// What a run that needs alignment says when the browser has no WebGPU.
pub const NO_WEBGPU: &str =
    "Alignment in the browser runs on the GPU (WebGPU), which this browser does not offer.";

thread_local! {
    /// The device, opened once and shared by every alignment. Cleared after a
    /// failure so that the next alignment opens a fresh one.
    static CONTEXT: RefCell<Option<Arc<GpuContext>>> = const { RefCell::new(None) };
}

thread_local! {
    /// The GPU's own words for the last thing it refused, so that the failure the
    /// page shows can carry them (the console is out of the user's sight).
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn log(message: &str) {
    web_sys::console::warn_1(&JsValue::from_str(&format!("z-stackr worker: {message}")));
}

/// Log a GPU refusal and remember it for [`take_last_error`].
fn record(message: String) {
    log(&message);
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(message));
}

/// The last GPU refusal since the previous call, if any.
pub fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|slot| slot.borrow_mut().take())
}

/// The open device, or a new one.
async fn context() -> Result<Arc<GpuContext>, String> {
    if let Some(ctx) = CONTEXT.with(|slot| slot.borrow().clone()) {
        return Ok(ctx);
    }
    let ctx = GpuContext::new_async()
        .await
        .map_err(|error| error.to_string())?;
    CONTEXT.with(|slot| *slot.borrow_mut() = Some(Arc::clone(&ctx)));
    Ok(ctx)
}

/// Open the device at start-up. Returns the adapter's name for the page's status
/// line, or `None` (the reason goes to the console) when there is no WebGPU.
pub async fn init() -> Option<String> {
    match context().await {
        Ok(ctx) => {
            let name = ctx.info().name.trim().to_owned();
            Some(if name.is_empty() {
                "WebGPU adapter".to_owned()
            } else {
                name
            })
        }
        Err(reason) => {
            log(&format!("no WebGPU: {reason}"));
            None
        }
    }
}

/// Drop the shared device after a failure, so that the next alignment opens
/// another one.
pub fn forget() {
    CONTEXT.with(|slot| *slot.borrow_mut() = None);
}

/// The alignment backend on the shared device.
///
/// # Errors
/// The [`NO_WEBGPU`] sentence with the reason, when the device cannot be opened.
pub async fn backend() -> Result<WebGpuBackend, String> {
    match context().await {
        Ok(ctx) => Ok(WebGpuBackend {
            ctx,
            objective: None,
            warp_plan: None,
        }),
        Err(reason) => Err(format!("{NO_WEBGPU} ({reason})")),
    }
}

/// A matrix as the row-major arrays the kernels take.
fn rows(m: &Matrix3<f32>) -> [[f32; 3]; 3] {
    std::array::from_fn(|r| std::array::from_fn(|c| m[(r, c)]))
}

/// The registration objective and the final warp on WebGPU.
///
/// Unlike the desktop backend this one takes every pyramid level, however small
/// (`MIN_OBJECTIVE_PIXELS` is a CPU-versus-GPU speed rule, and the browser has no
/// CPU alignment to hand a small level to). Any failure is reported as `false` /
/// `None`, which ends the alignment with the GPU-only error.
pub struct WebGpuBackend {
    ctx: Arc<GpuContext>,
    /// The level the search is on.
    objective: Option<ObjectivePlan>,
    /// The warp buffers, reused for frames of the same size.
    warp_plan: Option<((usize, usize), WarpPlan)>,
}

impl WebGpuBackend {
    fn open_level(
        &mut self,
        reference: &[f32],
        source: &[f32],
        width: usize,
        height: usize,
        sampler: ObjectiveSampler,
    ) -> bool {
        // The previous level's buffers go first: one level lives on the device.
        self.objective = None;
        let sampler = match sampler {
            ObjectiveSampler::Bicubic => Sampler::Bicubic,
            ObjectiveSampler::SplineClamped => Sampler::SplineClamped,
        };
        match ObjectivePlan::new(&self.ctx, reference, source, width, height, sampler) {
            Ok(plan) => {
                self.objective = Some(plan);
                true
            }
            Err(error) => {
                record(format!("the GPU refused a {width}x{height} level: {error}"));
                false
            }
        }
    }
}

impl AsyncLevelDevice for WebGpuBackend {
    fn level(
        &mut self,
        reference: &[f32],
        source: &[f32],
        width: usize,
        height: usize,
        sampler: ObjectiveSampler,
    ) -> impl Future<Output = bool> {
        std::future::ready(self.open_level(reference, source, width, height, sampler))
    }

    async fn moments(&mut self, inverse: &Matrix3<f32>) -> Option<Moments> {
        let plan = self.objective.as_mut()?;
        match plan.moments_async(&rows(inverse)).await {
            Ok(m) if m.mean.is_finite() && m.m2.is_finite() => Some(Moments {
                count: m.count,
                mean: m.mean,
                m2: m.m2,
            }),
            Ok(_) => {
                record("the GPU objective returned non-finite moments".to_owned());
                None
            }
            Err(error) => {
                record(format!("the GPU objective failed: {error}"));
                None
            }
        }
    }
}

impl AsyncAlignBackend for WebGpuBackend {
    async fn warp(
        &mut self,
        frame: &PlanarImage<f32>,
        apply: &Matrix3<f32>,
        edge_clamped: bool,
    ) -> Option<(PlanarImage<f32>, Vec<bool>)> {
        // The kernel samples the frame at `inverse * p`.
        let inverse = apply.try_inverse()?;
        let size = (frame.width, frame.height);
        // The last level is done with: free it before the warp's buffers.
        self.objective = None;
        if self
            .warp_plan
            .as_ref()
            .is_none_or(|(have, _)| *have != size)
        {
            self.warp_plan = None;
            match WarpPlan::new(&self.ctx, size.0, size.1) {
                Ok(plan) => self.warp_plan = Some((size, plan)),
                Err(error) => {
                    record(format!("the GPU refused the warp buffers: {error}"));
                    return None;
                }
            }
        }
        let (_, plan) = self.warp_plan.as_mut()?;
        let sampler = if edge_clamped {
            Sampler::SplineClamped
        } else {
            Sampler::Bicubic
        };
        match plan.warp_async(frame, &rows(&inverse), sampler).await {
            Ok(done) => Some(done),
            Err(error) => {
                record(format!("the GPU warp failed: {error}"));
                None
            }
        }
    }
}
