#![allow(
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

#[cfg(feature = "akaze")]
pub mod akaze_match;
pub mod brightness;
pub mod pipeline;
#[cfg(feature = "akaze")]
pub mod ransac;
pub mod refine;
pub mod transform;

// Re-export the breathing types at the crate root for convenient use.
#[cfg(feature = "akaze")]
pub use ransac::{BreathingCorrector, BreathingEstimate};

// Re-export the intensity-based refinement entry points.
//
// `refine_alignment_registration` (Nelder-Mead) and `refine_alignment_lk`
// (Lucas-Kanade) are both live: `pipeline::align_frame` selects between them
// (or tries both, in `OptimizerSetting::Auto`) based on the caller's
// resolved `stacker_core::settings::OptimizerSetting`.
pub use refine::{
    BoundedRefineOptions, LkResult, refine_alignment_lk, refine_alignment_registration,
};

// Re-export the single shared per-frame alignment entry point so both apps
// call exactly the same dispatch.
pub use brightness::{BrightnessTarget, apply_brightness_correction};
pub use pipeline::{align_frame, is_sane_seed};

/// Re-export `nalgebra::Matrix3` so callers (e.g. `stacker-cli`) can build
/// identity / composed matrices without adding a direct `nalgebra` dependency.
pub use nalgebra::Matrix3;
