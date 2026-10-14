//! [`PairCornerAlignmentLoss`]: the corner-projection alignment loss for the
//! pairwise `[N, 3, 3]` matrices of `fusionalign-v1`.
//!
//! Exactly the technique and rationale of [`crate::loss::CornerAlignmentLoss`],
//! without the stack dimension: one reference/frame pair per matrix.

use burn::{prelude::*, tensor::TensorData};

use crate::{loss::corner_alignment::corner_data, traits::PairAlignmentLoss};

/// Configuration for [`PairCornerAlignmentLoss`]. Parameter-free.
#[derive(Config, Debug)]
pub struct PairAlignmentLossConfig {}

/// Affine-registration loss for `fusionalign-v1`: MSE of the projected corners of
/// `[N, 3, 3]` matrices.
#[derive(Debug, Clone)]
pub struct PairCornerAlignmentLoss {}

impl PairAlignmentLossConfig {
    /// Build the loss.
    #[must_use]
    pub const fn init(&self) -> PairCornerAlignmentLoss {
        PairCornerAlignmentLoss {}
    }
}

impl<B: Backend> PairAlignmentLoss<B> for PairCornerAlignmentLoss {
    fn forward(&self, pred_matrix: Tensor<B, 3>, gt_matrix: Tensor<B, 3>) -> Tensor<B, 1> {
        let device = &pred_matrix.device();
        let corners = Tensor::<B, 2>::from_data(TensorData::new(corner_data(), [3, 4]), device)
            .reshape([1, 3, 4]);

        // Broadcast the shared corner set over the [N,3,3] matrix batch.
        let pred_pts = pred_matrix.matmul(corners.clone());
        let gt_pts = gt_matrix.matmul(corners);

        pred_pts.sub(gt_pts).powf_scalar(2.0).mean()
    }
}
