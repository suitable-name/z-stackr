//! Loss functions for the crate's four training strategies.
//!
//! * [`FocusFusionLoss`] — pairwise recurrent-merge loss
//!   ([`crate::traits::PairwiseFusionLoss`]).
//! * [`FocusBatchLoss`] — whole-stack batch-fusion loss
//!   ([`crate::traits::BatchFusionLoss`]).
//! * [`CornerAlignmentLoss`] — batch affine-registration loss
//!   ([`crate::traits::BatchAlignmentLoss`]).
//! * [`PairCornerAlignmentLoss`] — pairwise affine-registration loss
//!   ([`crate::traits::PairAlignmentLoss`]), the `[N,3,3]` (no `S` dimension)
//!   analogue of [`CornerAlignmentLoss`] used to train
//!   [`crate::model::FusionAlignNet`].
//! * [`photometric_gradient_loss`] — the unsupervised fine-tuning loss on real
//!   stacks (a skeleton, see [`photometric`]).
//!
//! ## Per-term reasoning (fusion losses)
//!
//! Both [`FocusFusionLoss`] and [`FocusBatchLoss`] combine the same family of
//! masked terms, adapted from the pairwise `source`-relative form to the batch
//! whole-stack form:
//!
//! | Term | Pairwise | Batch | Why |
//! |---|---|---|---|
//! | Charbonnier | RGB vs. ground truth | same | Robust (non-quadratic) reconstruction: tolerates the occasional large residual at a hard depth edge. |
//! | Multi-scale gradient L1 | `pred` vs. `gt`, several scales | same | Preserves high-frequency detail — the point of focus stacking — at several scales so fine texture and larger soft-focus halos are both caught. |
//! | Sharpness retention | `merged` not blurrier than the sharpest frame merged so far | not blurrier than the sharpest frame of the stack | Stops the network averaging away sharpness: the result must be at least as sharp as the best evidence at each pixel. |
//! | Confidence / gate | predicted confidence vs. ground-truth coverage | softmax gate vs. `masks / Σ_S masks` where some frame claims focus | Directly supervises the selection mechanism instead of only shaping it through the reconstruction terms. |
//!
//! All terms are down-weighted near depth edges by the scene's occlusion mask
//! ([`helpers::occlusion_weight_map`]); occlusion boundaries are inherently
//! ambiguous (no single frame is "correct" there), so a hard target there would
//! fight the network. The gradient term weights each finite difference by the mean
//! of the map at the two pixels it spans.
//!
//! The alignment losses project the four normalised image corners through the
//! predicted and ground-truth matrices and take the MSE of the projected points,
//! which is scale-aware without per-parameter weights (see
//! [`corner_alignment`]).
//!
//! Everything works on linear-light tensors.

pub mod corner_alignment;
pub mod focus_batch;
pub mod focus_fusion;
pub mod helpers;
pub mod pair_corner_alignment;
pub mod photometric;

pub use corner_alignment::{BatchAlignmentLossConfig, CornerAlignmentLoss};
pub use focus_batch::{BatchLossTerms, FocusBatchLoss, FocusBatchLossConfig};
pub use focus_fusion::{FocusFusionLoss, FocusFusionLossConfig, LossTerms};
pub use helpers::sharpness_map;
pub use pair_corner_alignment::{PairAlignmentLossConfig, PairCornerAlignmentLoss};
pub use photometric::photometric_gradient_loss;
