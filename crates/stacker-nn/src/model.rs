//! The networks.
//!
//! | tag | type | strategy | purpose |
//! |---|---|---|---|
//! | `focusmerge-v1` | [`FocusMergeNet`] | pairwise fold | fuse a focus stack one frame at a time |
//! | `batchmerge-v2` | [`BatchMergeNet`] | whole stack | fuse a (tile of a) stack at once |
//! | `batchalign-v2` | [`BatchAlignNet`] | whole stack | predict one affine per frame |
//! | `fusionalign-v1` | [`FusionAlignNet`] | pairwise | predict one affine per frame, streaming |
//!
//! All four are built from the blocks in [`blocks`], and every normalisation layer
//! is the per-pixel [`ChannelNorm`], so a pixel's output never depends on the
//! extent of the tensor it is computed in (the property that makes tiled
//! inference and 256 px training crops consistent with whole-image runs).
//!
//! The architecture of [`FocusMergeNet`], including the receptive-field table the
//! tile overlap derives from, is documented in [`focus_merge`].

pub mod align_common;
pub mod batch_align;
pub mod batch_merge;
pub mod blocks;
pub mod focus_merge;
pub mod fusion_align;
pub mod size;

pub use batch_align::{BatchAlignNet, BatchAlignNetConfig};
pub use batch_merge::{BatchMergeNet, BatchMergeNetConfig};
pub use blocks::{ChannelNorm, fusion_radius_for_depth};
pub use focus_merge::{FocusMergeNet, FocusMergeNetConfig, MergeOutput, receptive_field_radius};
pub use fusion_align::{FusionAlignNet, FusionAlignNetConfig};
pub use size::ModelSize;
