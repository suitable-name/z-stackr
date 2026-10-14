//! Unsupervised photometric fine-tuning loss for the alignment networks.
//!
//! **Skeleton.** The loss itself is complete and tested; the training phase that
//! uses it (the trainer's `--real-data` / `--photometric-weight` fine-tune on the
//! last quarter of the epochs) has no separate real-data batch, schedule or
//! logging and has not been validated on a GPU run. Nothing here is on by default
//! and nothing proves that fine-tuning improves a model; treat results as
//! unvalidated until measured against `scripts/04_eval_alignment.py`.

use burn::prelude::*;

use crate::{loss::helpers::finite_diff, warp::warp_affine};

/// Charbonnier smoothing epsilon of the gradient comparison.
pub const PHOTOMETRIC_CHARBONNIER_EPS: f32 = 0.001;

/// Warp `frame` by the predicted `matrix` and compare its **gradient magnitudes**
/// (not raw pixel values) against `reference`'s, masked by the warp's validity
/// mask.
///
/// Comparing gradients rather than intensities is what makes the loss robust to
/// genuine defocus differences between the frames of a real focus stack: two
/// frames are blurred differently by definition, so intensities differ even when
/// perfectly aligned, while coarse edge *positions* barely move with a blur
/// change. The loss can therefore be used on **unlabelled real stacks** (no
/// ground-truth matrix) to fine-tune a model trained on synthetic data, mixed with
/// the supervised [`crate::loss::CornerAlignmentLoss`] on synthetic batches.
///
/// `matrix` maps reference coordinates to frame coordinates in the normalised
/// convention of [`crate::warp::warp_affine`], so the matrix that registers the
/// frame to the reference minimises the loss. All tensors are linear light.
///
/// Returns a scalar `[1]` tensor.
///
/// # Panics
///
/// Panics if `frame` and `reference` do not share the same shape, or if either
/// spatial dimension is smaller than 2 (see [`crate::warp::warp_affine`]).
#[must_use]
pub fn photometric_gradient_loss<B: Backend>(
    frame: Tensor<B, 4>,
    reference: Tensor<B, 4>,
    matrix: Tensor<B, 3>,
) -> Tensor<B, 1> {
    assert_eq!(
        frame.dims(),
        reference.dims(),
        "photometric_gradient_loss: frame/reference shape mismatch"
    );

    let warped = warp_affine(frame, matrix);

    let (warped_grad_x, warped_grad_y) = finite_diff(warped.warped);
    let (ref_grad_x, ref_grad_y) = finite_diff(reference);

    let [vn, _vc, vh, vw] = warped.valid.dims();
    // A finite difference spans two pixels; it counts only when BOTH sampled
    // inside the source (a clamped out-of-bounds neighbour would fake a zero
    // gradient at the border).
    let valid = warped.valid;
    let mask_x = valid
        .clone()
        .slice([0..vn, 0..1, 0..vh, 0..(vw - 1)])
        .mul(valid.clone().slice([0..vn, 0..1, 0..vh, 1..vw]));
    let mask_y = valid
        .clone()
        .slice([0..vn, 0..1, 0..(vh - 1), 0..vw])
        .mul(valid.slice([0..vn, 0..1, 1..vh, 0..vw]));

    let eps = PHOTOMETRIC_CHARBONNIER_EPS;
    let diff_x = warped_grad_x.sub(ref_grad_x);
    let diff_y = warped_grad_y.sub(ref_grad_y);
    let charb_x = diff_x.clone().mul(diff_x).add_scalar(eps * eps).sqrt();
    let charb_y = diff_y.clone().mul(diff_y).add_scalar(eps * eps).sqrt();

    let masked_sum_x = charb_x.mul(mask_x.clone()).sum();
    let masked_sum_y = charb_y.mul(mask_y.clone()).sum();
    let denom = mask_x.sum().add(mask_y.sum()).add_scalar(1e-6);

    masked_sum_x.add(masked_sum_y).div(denom)
}
