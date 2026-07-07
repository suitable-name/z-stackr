//! Feature-match-based coarse alignment (AKAZE + RANSAC), feature-gated
//! behind `akaze`.
//!
//! * [`alignment`] — [`AlignmentEstimator`] fits a translation-only or affine
//!   `Matrix3` seed from AKAZE keypoint correspondences via RANSAC, with a
//!   strict min-inlier-support acceptance rule (see the module's docs for why
//!   a model corroborated only by its own minimal sample is rejected).
//! * [`breathing`] — [`BreathingCorrector`] fits the narrower centre-anchored
//!   similarity model (uniform scale + residual translation) that isolates
//!   focus-breathing magnification specifically; see its docs for the model
//!   derivation and composition with the affine alignment step.
//! * [`utils`] — shared helpers ([`apply_h`] projective point mapping,
//!   correspondence extraction) used by both estimators.
//!
//! This module is an *optional accelerator* for the seed the intensity-based
//! [`crate::pipeline`] refinement starts from — never a requirement; see the
//! crate README's "Alignment Pipeline" section for the full seeding story.

#![allow(clippy::many_single_char_names)]

pub mod alignment;
pub mod breathing;
pub mod utils;

pub use alignment::{AlignmentEstimator, AlignmentMode};
pub use breathing::{BreathingCorrector, BreathingEstimate};
pub use utils::apply_h;
