//! Chain (sequential) alignment primitives.
//!
//! Instead of aligning every frame to frame 0, a *chain* aligns frame `k` to its
//! neighbour `k - 1`, so a slow drift (focus breathing, a creeping rail) never
//! has to be bridged in one step.  The loop over the frames belongs to the
//! pipeline; this module has the two pieces of arithmetic that are easy to get
//! wrong: composing the per-step matrices, and the coverage of the result.
//!
//! ## Convention
//!
//! A matrix `M` from the refiners maps **reference coordinates to source
//! coordinates**: `source(M p) ≈ reference(p)`.  Write `C_k` for the cumulative
//! matrix of frame `k`, mapping **frame-0 coordinates to frame-`k` coordinates**
//! (`frame_k(C_k p) ≈ frame_0(p)`, `C_0 = I`).  `C_k` is what the stack needs: the
//! frame is brought into frame 0's geometry by warping it with `C_k⁻¹`
//! ([`crate::transform::warp_image`]), exactly as for a fixed reference.
//!
//! ## Two ways to run the chain
//!
//! **A. Raw previous frame as the reference (compose).**  Refine frame `k`
//! against the *unwarped* frame `k - 1`; the result `R_k` maps frame-`(k-1)`
//! coordinates to frame-`k` coordinates: `frame_k(R_k q) ≈ frame_{k-1}(q)`.  With
//! `q = C_{k-1} p`:
//!
//! ```text
//! frame_k(R_k C_{k-1} p) ≈ frame_{k-1}(C_{k-1} p) ≈ frame_0(p)
//!   =>  C_k = R_k · C_{k-1}
//! ```
//!
//! [`compose_chain`] is that product.  Warm start each step with the previous
//! step (`R_{k-1}`, steady drift) or the identity.  The reference has no black
//! border, because it is a raw frame.
//!
//! **B. Warped previous frame as the reference (no composition).**  Refine frame
//! `k` against `frame_{k-1}` *already warped into frame 0's geometry*.  That image
//! lives in frame-0 coordinates, so the refiner's result maps frame-0 coordinates
//! to frame-`k` coordinates: it **is** `C_k`, no product needed (composing it
//! again would apply the drift twice).  Warm start with `C_{k-1}`.  The warped
//! reference is zero outside the previous frame's coverage, so its border is an
//! artificial edge that the objective sees unless it is masked.
//!
//! Either way the coverage of frame `k` in frame-0 coordinates is a property of
//! the **cumulative** matrix, not of the step: [`chain_coverage`].  Using the
//! step's matrix (the GitHub z-stackr mask) under-estimates the shift and lets a
//! border of undefined pixels into the common area.

use nalgebra::Matrix3;

use crate::transform::coverage_mask;

/// The cumulative matrix of frame `k` from the one of frame `k - 1` and the
/// step between them (scheme A in the module documentation):
/// `C_k = step · C_{k-1}`.
///
/// - `prev_cumulative` maps frame-0 coordinates to frame-`(k-1)` coordinates
///   (the identity for frame 1's predecessor, frame 0).
/// - `step` maps frame-`(k-1)` coordinates to frame-`k` coordinates (what the
///   refiners return with frame `k - 1` as the reference and frame `k` as the
///   source).
///
/// The product is formed in `f64` and rounded once, so a long chain does not
/// accumulate `f32` rounding in the intermediate products.
#[must_use]
pub fn compose_chain(prev_cumulative: &Matrix3<f32>, step: &Matrix3<f32>) -> Matrix3<f32> {
    let step64: Matrix3<f64> = step.cast();
    let prev64: Matrix3<f64> = prev_cumulative.cast();
    (step64 * prev64).cast()
}

/// The coverage mask of frame `k` in frame-0 coordinates: `mask[y * width + x]`
/// is `true` iff the point `C_k · (x, y)` lies inside the frame, i.e. the output
/// pixel `(x, y)` of the aligned frame receives real data.
///
/// `cumulative` is `C_k` (frame-0 to frame-`k` coordinates).  This equals
/// [`coverage_mask`] of the matrix the frame is warped with, `C_k⁻¹`.  A
/// non-finite or singular `cumulative` gives an all-`false` mask.  Fold the masks
/// of all frames with [`crate::transform::intersect_coverage`] and crop to
/// [`crate::transform::largest_true_rectangle`] as for a fixed reference.
#[must_use]
pub fn chain_coverage(cumulative: &Matrix3<f32>, width: usize, height: usize) -> Vec<bool> {
    cumulative.try_inverse().map_or_else(
        || vec![false; width * height],
        |applied| coverage_mask(&applied, width, height),
    )
}
