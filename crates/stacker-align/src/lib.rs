//! Frame registration for focus stacks.
//!
//! Registration is intensity based: [`refine`] minimises the RMS difference
//! between the reference and the warped source luma over a motion model
//! ([`Motion`]: shift, shift and uniform scale, the four-parameter similarity,
//! or the six-parameter affine map), coarse to fine, with the Nelder-Mead simplex
//! or Lucas-Kanade / Gauss-Newton ([`Optimizer`]). [`refine::motion`] is the one
//! entry point the pipeline maps its settings onto, [`refine::gate`] the identity
//! gate that guards a refined matrix, and [`chain`] the arithmetic of sequential
//! (chain) alignment. With the `akaze` feature, [`akaze_match`] and [`ransac`] can
//! produce a feature-based seed for that search; without it the search starts
//! from the identity.
//!
//! [`transform`] applies a matrix to a [`PlanarImage`](stacker_core::image::PlanarImage)
//! (Catmull-Rom bicubic with a zero-filled border, or the edge-clamped
//! registration sampler) and derives the coverage mask and the largest common
//! rectangle of a stack of warped frames.
//!
//! Matrices follow one convention throughout: the matrix returned by the
//! refinement maps **reference coordinates to source coordinates**; to bring a
//! source frame into the reference frame, warp it with the *inverse* (see
//! [`transform::warp_image`]).
//!
//! The crate README follows; its usage example is compiled and run as a
//! doctest.
//!
//! ---
//!
#![doc = include_str!("../README.md")]
#![feature(portable_simd)]
#![warn(missing_docs)]

/// AKAZE keypoint extraction and descriptor matching (feature `akaze`).
#[cfg(feature = "akaze")]
pub mod akaze_match;
/// Per-frame brightness correction against the reference (gamma + scale fit in
/// linear light).
pub mod brightness;
/// Chain (sequential) alignment: composing the per-step matrices and the
/// coverage of the result.
pub mod chain;
/// The hook through which the registration objective runs on a GPU.
pub mod device;
/// RANSAC estimation of a transform from matched keypoints (feature `akaze`).
#[cfg(feature = "akaze")]
pub mod ransac;
/// Intensity-based subpixel refinement of an alignment matrix.
pub mod refine;
/// Warping with a matrix, coverage masks and the common valid rectangle.
pub mod transform;

pub use brightness::{
    BrightnessFit, BrightnessTarget, correct_brightness, correct_brightness_masked,
};

// Re-export the breathing types at the crate root for convenient use.
#[cfg(feature = "akaze")]
pub use ransac::{BreathingCorrector, BreathingEstimate};

// Re-export the intensity-based refinement entry points.
pub use refine::{
    AffineParams, BoundedRefineOptions, Motion, MotionRefine, Optimizer, RefineOptions, Refined,
    apply_identity_gate, refine_alignment, refine_alignment_bounded, refine_alignment_lk,
    refine_alignment_registration, refine_motion, refined_beats_identity,
};

/// Re-export `nalgebra::Matrix3` so callers (e.g. `stacker-cli`) can build
/// identity / composed matrices without adding a direct `nalgebra` dependency.
pub use nalgebra::Matrix3;
