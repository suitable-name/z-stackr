//! Loss functions for the crate's four training strategies.
//!
//! * [`FocusFusionLoss`] — pairwise recurrent-merge loss ([`PairwiseFusionLoss`]).
//! * [`FocusBatchLoss`] — whole-stack batch-fusion loss ([`BatchFusionLoss`]).
//! * [`CornerAlignmentLoss`] — batch affine-registration loss ([`BatchAlignmentLoss`]).
//! * [`PairCornerAlignmentLoss`] — pairwise affine-registration loss
//!   ([`PairAlignmentLoss`]), the `[N,3,3]` (no `S` dimension) analogue of
//!   [`CornerAlignmentLoss`] used to train [`crate::model::FusionAlignNet`].
//!
//! ## Per-term reasoning (fusion losses)
//!
//! Both [`FocusFusionLoss`] and [`FocusBatchLoss`] combine the same family of
//! masked terms, adapted from the pairwise `source`-relative form to the
//! batch whole-stack form:
//!
//! | Term | Pairwise (`FocusFusionLoss`) | Batch (`FocusBatchLoss`) | Why |
//! |---|---|---|---|
//! | Charbonnier | RGB vs. ground truth | RGB vs. ground truth | Robust (non-quadratic) reconstruction loss — tolerates the occasional large residual near a hard depth edge without letting it dominate the gradient the way an L2 term would. |
//! | Multi-scale gradient L1 | `pred` vs. `gt`, several downsample scales | same | Preserves high-frequency detail — the entire point of focus stacking — by penalising blurring directly in gradient space, at multiple scales so both fine texture and larger soft-focus halos are caught. |
//! | Sharpness retention | penalises `merged` blurrier than `source` | penalises `merged` blurrier than the sharpest input frame (max over the stack) | Stops the network from "averaging away" sharpness: the fused result must be at least as sharp as the best evidence available at each pixel. |
//! | Confidence / gate supervision | predicted confidence vs. ground-truth in-focus coverage | predicted softmax gate vs. a target blend built from per-frame in-focus masks, masked where no frame claims to be in focus | Directly supervises the selection mechanism (confidence channel / blending gate) so it learns to localise the in-focus frame instead of only being shaped indirectly through the reconstruction terms. |
//!
//! All terms in both losses are down-weighted near depth edges via the
//! scene's occlusion mask ([`helpers::occlusion_weight_map`]), using the same
//! weighting formula and default floor — occlusion boundaries are inherently
//! ambiguous (no single frame is "correct" there), so forcing a hard target
//! at those pixels would fight the network rather than train it.
//!
//! [`CornerAlignmentLoss`] instead projects the four normalized-space image
//! corners through both the predicted and ground-truth matrices and takes
//! the MSE between the projected points, rather than comparing matrix entries
//! directly — this makes the loss scale-aware (a rotation/shear error is
//! penalised in proportion to how far it actually displaces visible content,
//! not by an arbitrary weighting between translation/rotation/scale terms).
//! [`PairCornerAlignmentLoss`] uses the exact same corner-projection
//! technique, just over a single `[N,3,3]` matrix pair instead of a
//! `[N,S,3,3]` stack — see that type's docs.

pub mod corner_alignment;
pub mod focus_batch;
pub mod focus_fusion;
pub mod helpers;
pub mod pair_corner_alignment;
pub mod photometric;

pub use corner_alignment::{BatchAlignmentLossConfig, CornerAlignmentLoss};
pub use focus_batch::{FocusBatchLoss, FocusBatchLossConfig};
pub use focus_fusion::{FocusFusionLoss, FocusFusionLossConfig};
pub use pair_corner_alignment::{PairAlignmentLossConfig, PairCornerAlignmentLoss};
pub use photometric::photometric_gradient_loss;
