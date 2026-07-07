//! Single source of truth for per-frame intensity-based alignment + warp.
//!
//! Both the CLI (`stacker-cli`) and GUI (`stacker-gui`) align each frame to a
//! reference using the same intensity-based refinement dispatch, living here
//! so the two apps can never silently diverge on alignment logic.
//!
//! AKAZE coarse-seed *computation* deliberately stays in each caller (it is
//! feature-gated and call-site specific); this module only consumes the
//! already-resolved `seed` transform.

mod align;
mod refinement;
mod seed;

pub use align::align_frame;
#[cfg(feature = "akaze")]
pub use refinement::akaze_mode_for_alignment;
pub use seed::is_sane_seed;
