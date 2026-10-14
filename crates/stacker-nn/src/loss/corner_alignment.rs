//! [`CornerAlignmentLoss`]: the corner-projection alignment loss for
//! `[N, S, 3, 3]` whole-stack matrices (`batchalign-v2`).
//!
//! It projects the four normalised-space (`[-1, 1]²`) image corners through the
//! predicted and the ground-truth matrix and takes the mean squared distance
//! between the projected points, rather than comparing matrix entries directly.
//! This is scale-aware by construction: translation, rotation and scale/shear
//! errors live on very different numeric scales as raw matrix entries, but each
//! displaces a projected corner by a physically comparable amount, so one
//! unweighted MSE penalises all of them in proportion to how much they move
//! visible content. Hence the config has no knobs. The pairwise `[N, 3, 3]`
//! analogue is [`crate::loss::PairCornerAlignmentLoss`].
//!
//! The matrices must be in the normalised coordinates of **the image the network
//! saw** (see [`crate::bridge`]); a ground truth normalised by a larger frame than
//! the crop the network was shown is a different transform and trains the head
//! towards the wrong answer.

use burn::{prelude::*, tensor::TensorData};

use crate::traits::BatchAlignmentLoss;

/// The four normalised image corners as homogeneous columns, row-major `3 × 4`:
/// `(-1,-1,1)`, `(1,-1,1)`, `(-1,1,1)`, `(1,1,1)`.
pub(crate) fn corner_data() -> Vec<f32> {
    vec![
        -1.0, 1.0, -1.0, 1.0, //
        -1.0, -1.0, 1.0, 1.0, //
        1.0, 1.0, 1.0, 1.0,
    ]
}

/// Configuration for [`CornerAlignmentLoss`]. Parameter-free; see the module docs.
#[derive(Config, Debug)]
pub struct BatchAlignmentLossConfig {}

/// Affine-registration loss for `batchalign-v2`: MSE of the projected corners of
/// `[N, S, 3, 3]` matrices. See the module docs.
#[derive(Debug, Clone)]
pub struct CornerAlignmentLoss {}

impl BatchAlignmentLossConfig {
    /// Build the loss.
    #[must_use]
    pub const fn init(&self) -> CornerAlignmentLoss {
        CornerAlignmentLoss {}
    }
}

impl<B: Backend> BatchAlignmentLoss<B> for CornerAlignmentLoss {
    fn forward(&self, pred_matrices: Tensor<B, 4>, gt_matrices: Tensor<B, 4>) -> Tensor<B, 1> {
        let device = &pred_matrices.device();
        let corners = Tensor::<B, 2>::from_data(TensorData::new(corner_data(), [3, 4]), device)
            .reshape([1, 1, 3, 4]);

        // Broadcast the shared corner set over the [N,S,3,3] matrix batch.
        let pred_pts = pred_matrices.matmul(corners.clone());
        let gt_pts = gt_matrices.matmul(corners);

        pred_pts.sub(gt_pts).powf_scalar(2.0).mean()
    }
}
