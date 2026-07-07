//! Intensity-based subpixel alignment refinement.
//!
//! This module implements intensity-based alignment refinement: starting from a
//! coarse transform (e.g. AKAZE + RANSAC), it minimises the RMS intensity
//! difference between the warped reference luma plane and the source luma plane
//! over a 6-DOF parameter space `(tx, ty, scale, rotate, aspect, shear)`. Not
//! every mode solves all six — see [`crate::pipeline::align_frame`] for the
//! per-mode DOF gating.
//!
//! Two independent optimisers solve this same objective:
//!
//! - [`lucas_kanade::refine_alignment_lk`] — a forward-additive Lucas-Kanade
//!   / Gauss-Newton optimiser using analytic image-gradient Jacobians. Far
//!   fewer iterations per pyramid level than the simplex below, since it
//!   follows the actual gradient instead of blind function sampling. See
//!   that module's doc comment for the full formulation, damping strategy,
//!   and convergence thresholds.
//! - [`refine_alignment_registration`] — the original Nelder-Mead (downhill
//!   simplex) optimiser, described in detail below.
//!
//! [`stacker_core::settings::OptimizerSetting`] selects which one
//! `pipeline::align_frame` uses (`Auto` tries Lucas-Kanade first and falls
//! back to Nelder-Mead on failure/regression).
//!
//! ## Coarse-to-fine pyramid (Nelder-Mead path)
//!
//! To avoid warping the full-resolution reference on every Nelder-Mead
//! evaluation (up to ~400 iterations), we build a Gaussian luma pyramid of
//! both the reference and source planes.  The optimiser runs from the coarsest
//! level to the finest:
//!
//! - At each level the objective warps only that level's (smaller) luma plane.
//! - The best parameter vector carries over unchanged to the next finer level
//!   because `(tx, ty)` are *fractional* (pixels / image dimension), so they
//!   are scale-invariant across pyramid levels.
//! - The coarsest levels use fewer iterations; a full polish runs at the finest
//!   level only, keeping total work far below the single-level baseline.
//!
//! ## NaN guard
//!
//! If any parameter or projected coordinate is non-finite the objective
//! returns a large finite penalty (`PENALTY`) instead of panicking, triggering
//! the NaN-guard early-return path (which aborts with the last-good params).
//! [`lucas_kanade`] follows the same never-panic, never-propagate-NaN
//! contract — see its own module doc for its specific guard mechanics.

#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::many_single_char_names
)]

mod core;
mod lucas_kanade;
mod params;
mod pyramid;
mod simplex;

pub use self::{core::*, lucas_kanade::*, params::*, pyramid::*, simplex::*};
