use crate::traits::PairAlignmentLoss;
use burn::prelude::*;

/// Configuration for [`PairCornerAlignmentLoss`]. Currently parameter-free —
/// mirrors [`super::BatchAlignmentLossConfig`]; see that type's docs for why
/// the corner-projection formulation needs no tunable weights.
#[derive(Config, Debug)]
pub struct PairAlignmentLossConfig {}

/// Affine-registration training loss for `FusionAlignNet` (pairwise
/// alignment) — the `[N,3,3]` analogue of [`super::CornerAlignmentLoss`],
/// with no `S` (stack) dimension since [`crate::traits::PairAlignmentModel`]
/// scores one reference/frame pair at a time.
///
/// Projects the four normalized-space (`[-1,1]²`) image corners through both
/// the predicted and ground-truth affine matrices and takes the mean squared
/// distance between the projected points, rather than comparing matrix
/// entries directly — identical technique and identical rationale to
/// [`super::CornerAlignmentLoss`] (translation/rotation/scale/shear errors
/// live on very different numeric scales as raw matrix entries, but every one
/// of them displaces a projected corner by a physically comparable amount, so
/// a single unweighted MSE over corner positions penalises all parameter
/// errors in proportion to how much they actually move visible content — no
/// per-parameter weight tuning needed). See [`PairAlignmentLossConfig`] for
/// why no config knobs exist.
#[derive(Debug, Clone)]
pub struct PairCornerAlignmentLoss {}

impl PairAlignmentLossConfig {
    #[must_use]
    pub const fn init(&self) -> PairCornerAlignmentLoss {
        PairCornerAlignmentLoss {}
    }
}

impl<B: Backend> PairAlignmentLoss<B> for PairCornerAlignmentLoss {
    fn forward(&self, pred_matrix: Tensor<B, 3>, gt_matrix: Tensor<B, 3>) -> Tensor<B, 1> {
        let device = &pred_matrix.device();

        // The four normalized-space image corners in homogeneous coordinates,
        // as columns: (-1,-1,1), (1,-1,1), (-1,1,1), (1,1,1). Same layout as
        // `CornerAlignmentLoss`, just broadcast over `[1,3,4]` (no `S` dim)
        // instead of `[1,1,3,4]`.
        #[rustfmt::skip]
        let corners_data = [
            -1.0_f32,  1.0, -1.0, 1.0,
            -1.0,     -1.0,  1.0, 1.0,
             1.0,      1.0,  1.0, 1.0,
        ];

        let corners = Tensor::<B, 1>::from_floats(corners_data, device).reshape([1, 3, 4]);

        // Project the corners through each matrix (broadcasting the shared
        // [1,3,4] corner set over the [N,3,3] matrix batch) and compare.
        let pred_pts = pred_matrix.matmul(corners.clone());
        let gt_pts = gt_matrix.matmul(corners);

        pred_pts.sub(gt_pts).powf_scalar(2.0).mean()
    }
}
