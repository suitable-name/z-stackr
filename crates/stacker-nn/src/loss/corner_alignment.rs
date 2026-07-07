use crate::traits::BatchAlignmentLoss;
use burn::prelude::*;

/// Configuration for [`CornerAlignmentLoss`]. Currently parameter-free — the
/// corner-projection formulation needs no tunable weights (see
/// [`CornerAlignmentLoss`]'s docs for why).
#[derive(Config, Debug)]
pub struct BatchAlignmentLossConfig {}

/// Affine-registration training loss for `batchalign-v2`.
///
/// Projects the four normalized-space (`[-1,1]²`) image corners through both
/// the predicted and ground-truth affine matrices and takes the mean squared
/// distance between the projected points, rather than comparing matrix
/// entries directly. This is deliberately scale-aware: translation, rotation,
/// and scale/shear errors live on very different numeric scales as raw matrix
/// entries, but every one of them displaces a projected corner by a
/// physically comparable amount, so a single unweighted MSE over corner
/// positions penalises all parameter errors in proportion to how much they
/// actually move visible content — no per-parameter weight tuning needed.
/// See [`super::BatchAlignmentLossConfig`] for why no config knobs exist.
#[derive(Debug, Clone)]
pub struct CornerAlignmentLoss {}

impl BatchAlignmentLossConfig {
    #[must_use]
    pub const fn init(&self) -> CornerAlignmentLoss {
        CornerAlignmentLoss {}
    }
}

impl<B: Backend> BatchAlignmentLoss<B> for CornerAlignmentLoss {
    fn forward(&self, pred_matrices: Tensor<B, 4>, gt_matrices: Tensor<B, 4>) -> Tensor<B, 1> {
        let device = &pred_matrices.device();

        // The four normalized-space image corners in homogeneous coordinates,
        // as columns: (-1,-1,1), (1,-1,1), (-1,1,1), (1,1,1).
        #[rustfmt::skip]
        let corners_data = [
            -1.0_f32,  1.0, -1.0, 1.0,
            -1.0,     -1.0,  1.0, 1.0,
             1.0,      1.0,  1.0, 1.0,
        ];

        let corners = Tensor::<B, 1>::from_floats(corners_data, device).reshape([1, 1, 3, 4]);

        // Project the corners through each matrix (broadcasting the shared
        // [1,1,3,4] corner set over the [N,S,3,3] matrix batch) and compare.
        let pred_pts = pred_matrices.matmul(corners.clone());
        let gt_pts = gt_matrices.matmul(corners);

        pred_pts.sub(gt_pts).powf_scalar(2.0).mean()
    }
}
