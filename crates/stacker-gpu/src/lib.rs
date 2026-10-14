//! GPU kernels for z-stackr, behind their own crate so the CPU crates never
//! depend on a GPU stack.
//!
//! The kernels behind the `gpu` feature of the pipeline, all through wgpu:
//!
//! - [`apex`]: the whole `Apex` fusion (Laplacian pyramid per frame, strongest
//!   coefficient per pixel and band, averaged base, reconstruction);
//! - [`relief`]: the `Relief` fusion (focus measure per frame, best frame per pixel,
//!   selection against a threshold, guided-filter smoothing);
//! - [`strata`]: the `Strata` saliency of a luma plane and the guided-filter pair
//!   that refines its layer weights;
//! - [`multigrid`]: the `Relief` multigrid depth solver, resident on the device
//!   from the upload of the pins to the read-back of the depth;
//! - [`align`]: the registration objective the alignment optimiser evaluates
//!   hundreds of times per frame, and the final warp of each frame.
//!
//! The kernels follow the CPU code (`stacker_algo`, `stacker_align`) operation for
//! operation and are checked against it by the tests; the `apex_bench` and
//! `gpu_bench` examples time both on the same frames.
//!
//! # Contract
//!
//! The CPU path stays the reference. A GPU result agrees with it within a stated
//! tolerance and is repeatable on one device and driver, but it is not bit
//! identical: drivers may contract a multiply and an add into one fused operation,
//! and that can move the last bit of a value or flip a winner between two frames
//! whose selection energies differ by about one rounding step.
//!
//! # Device use
//!
//! A frame is uploaded once, the whole pyramid is built and folded into the fused
//! pyramid on the device, and only the fused result is read back. Frames are folded
//! one at a time, so device memory does not grow with the number of frames
//! ([`apex::ApexPlan::device_bytes`] says how much one plan needs).
//!
//! # In the browser
//!
//! On `wasm32` the crate offers only the alignment kernels, through WebGPU and an
//! async API ([`GpuContext::new_async`], [`align::ObjectivePlan::moments_async`],
//! [`align::WarpPlan::warp_async`]); nothing blocks. Fusion and the synchronous
//! calls are native only.

#![warn(missing_docs)]

pub mod align;
mod common;
mod context;
// Fusion blocks on its read-backs; in the browser it stays on the CPU.
#[cfg(not(target_arch = "wasm32"))]
pub mod apex;
#[cfg(not(target_arch = "wasm32"))]
pub mod multigrid;
#[cfg(not(target_arch = "wasm32"))]
pub mod relief;
#[cfg(not(target_arch = "wasm32"))]
pub mod strata;

pub use context::{AdapterSummary, Failure, GpuContext};

/// What can go wrong when setting up or running the GPU path.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// No usable adapter was found (or the requested index does not exist).
    #[error("no usable GPU adapter ({0})")]
    NoAdapter(String),
    /// The adapter refused to create a device.
    #[error("could not create a GPU device: {0}")]
    Device(String),
    /// A buffer the plan needs is larger than the device allows.
    #[error(
        "the image needs a {needed}-byte buffer, the device allows {limit} bytes; use a smaller tile"
    )]
    TooLarge {
        /// The size of the largest buffer the plan needs, in bytes.
        needed: u64,
        /// The largest buffer (or storage binding) the device supports, in bytes.
        limit: u64,
    },
    /// A frame does not have the size the plan was built for.
    #[error("frame is {got:?} pixels, the plan was built for {expected:?}")]
    SizeMismatch {
        /// The size the plan expects, as (width, height).
        expected: (usize, usize),
        /// The size of the frame that was passed, as (width, height).
        got: (usize, usize),
    },
    /// Reading the result back from the device failed.
    #[error("could not read the result back from the GPU: {0}")]
    ReadBack(String),
    /// A failure armed by a test through [`GpuContext::inject_failure`].
    #[error("injected GPU failure while {0}")]
    Injected(&'static str),
}
