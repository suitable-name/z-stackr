//! Geometric image transforms: warping, resizing, and common-area cropping.
//!
//! * [`warp`] — forward-matrix affine/homographic warp via
//!   [`warp_image_clamped`] (edge-clamped 4-tap spline — the production path
//!   every active alignment mode warps with).
//! * [`resize`] — [`resize_planar_clamped`], a non-uniform (independent X/Y
//!   scale) resize using the same edge-clamped spline kernel as
//!   `warp_image_clamped`, used to optionally stretch a common-coverage-cropped
//!   stack back to the original canvas resolution.
//! * [`crop`] — [`coverage_mask`] / [`intersect_coverage`] /
//!   [`largest_true_rectangle`] / [`resolve_common_crop`]: computing the
//!   largest axis-aligned rectangle valid in every warped frame, so the fused
//!   output can be cropped to exclude the black/replicated border a warp
//!   introduces. See [`resolve_common_crop`]'s docs for the `None` semantics
//!   and the 25% rogue-frame guard.

pub mod crop;
/// GPU-accelerated production warp.
///
/// Internally engaged by [`warp::warp_image_clamped`] (public signature
/// unchanged — every call site benefits with zero churn). See the module's
/// own docs for the fallback contract and tolerance-equal (not bit-equal)
/// parity with the CPU/SIMD kernel.
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod resize;
pub mod warp;

pub use crop::{coverage_mask, intersect_coverage, largest_true_rectangle, resolve_common_crop};
pub use resize::resize_planar_clamped;
pub use warp::{spline4x4_sample_clamped, warp_image_clamped, warp_image_clamped_cpu};
