use crate::{loss::helpers::finite_diff, warp::warp_affine};
use burn::prelude::*;

const PHOTOMETRIC_CHARBONNIER_EPS: f32 = 0.001;

/// Unsupervised photometric fine-tuning loss for the alignment model
/// (`docs/batchalign-v2-design.md` §5.2): warps `frame` by the predicted
/// `matrix` and compares its **gradient magnitudes** (not raw pixel values)
/// against `reference`'s, masked by the warp's validity mask.
///
/// Gradient-domain comparison — rather than a direct Charbonnier on
/// intensities, as [`super::focus_fusion::FocusFusionLoss`] uses — is what
/// makes this term robust to genuine defocus difference between frames: a
/// real focus stack's two frames are blurred differently by definition, so
/// intensities differ even when perfectly aligned, but coarse edge
/// *positions* barely move with a blur change. This lets the loss be used on
/// **unlabelled real stacks** (no ground-truth matrix needed) to fine-tune a
/// model trained on synthetic data, mixed with the supervised
/// [`super::corner_alignment::CornerAlignmentLoss`] on synthetic batches.
///
/// # Panics
///
/// Panics if `frame` and `reference` do not share the same shape, or if
/// either spatial dimension is smaller than 2 (see [`crate::warp::warp_affine`]).
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
    let mask_x = warped
        .valid
        .clone()
        .slice([0..vn, 0..1, 0..vh, 0..(vw - 1)]);
    let mask_y = warped.valid.slice([0..vn, 0..1, 0..(vh - 1), 0..vw]);

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

#[cfg(test)]
mod photometric_tests {
    use super::*;
    use burn::{backend::NdArray, tensor::TensorData};

    type B = NdArray;

    fn device() -> burn::prelude::Device<B> {
        burn::prelude::Device::<B>::default()
    }

    fn identity_matrix(dev: burn::prelude::Device<B>) -> Tensor<B, 3> {
        Tensor::<B, 1>::from_data(
            TensorData::new(vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [9]),
            &dev,
        )
        .reshape([1, 3, 3])
    }

    fn translation_matrix(
        dx_px: f32,
        dy_px: f32,
        w: usize,
        h: usize,
        dev: burn::prelude::Device<B>,
    ) -> Tensor<B, 3> {
        let tx = 2.0 * dx_px / (w as f32 - 1.0);
        let ty = 2.0 * dy_px / (h as f32 - 1.0);
        Tensor::<B, 1>::from_data(
            TensorData::new(vec![1.0_f32, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0], [9]),
            &dev,
        )
        .reshape([1, 3, 3])
    }

    #[test]
    fn photometric_loss_near_zero_for_identity_warp_of_identical_images() {
        let dev = device();
        let (h, w) = (12_usize, 12_usize);
        let mut data = vec![0f32; h * w];
        let mut t = 0.37_f32;
        for v in &mut data {
            t = t.mul_add(1.7, 0.11).fract();
            *v = t;
        }
        let img =
            Tensor::<B, 1>::from_data(TensorData::new(data, [h * w]), &dev).reshape([1, 1, h, w]);

        let loss = photometric_gradient_loss(img.clone(), img, identity_matrix(dev));
        let val: f32 = loss.into_data().iter::<f32>().next().unwrap();
        assert!(
            val < 2.0 * PHOTOMETRIC_CHARBONNIER_EPS,
            "expected near-zero (epsilon-floor) loss for identity warp of identical images, got {val}"
        );
    }

    #[test]
    fn photometric_loss_positive_for_shifted_pair() {
        let dev = device();
        let (h, w) = (16_usize, 16_usize);
        let mut reference = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                reference[y * w + x] = ((x + y) % 4) as f32 / 3.0;
            }
        }
        let ref_t = Tensor::<B, 1>::from_data(TensorData::new(reference, [h * w]), &dev)
            .reshape([1, 1, h, w]);

        let mut frame = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let sx = x + 3; // shift the pattern by 3 px in x
                frame[y * w + x] = ((sx + y) % 4) as f32 / 3.0;
            }
        }
        let frame_t =
            Tensor::<B, 1>::from_data(TensorData::new(frame, [h * w]), &dev).reshape([1, 1, h, w]);

        let loss = photometric_gradient_loss(frame_t, ref_t, identity_matrix(dev));
        let val: f32 = loss.into_data().iter::<f32>().next().unwrap();
        assert!(
            val > 1e-2,
            "expected clearly positive loss for a shifted pair, got {val}"
        );
    }

    #[test]
    fn photometric_loss_decreases_when_matrix_corrects_the_shift() {
        let dev = device();
        let (h, w) = (16_usize, 16_usize);
        let dx = 3.0_f32;

        let mut reference = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                reference[y * w + x] = ((x + y) % 4) as f32 / 3.0;
            }
        }
        let ref_t = Tensor::<B, 1>::from_data(TensorData::new(reference, [h * w]), &dev)
            .reshape([1, 1, h, w]);

        let mut frame = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let sx = x + dx as usize;
                frame[y * w + x] = ((sx + y) % 4) as f32 / 3.0;
            }
        }
        let frame_t =
            Tensor::<B, 1>::from_data(TensorData::new(frame, [h * w]), &dev).reshape([1, 1, h, w]);

        let uncorrected =
            photometric_gradient_loss(frame_t.clone(), ref_t.clone(), identity_matrix(dev));
        let corrected =
            photometric_gradient_loss(frame_t, ref_t, translation_matrix(dx, 0.0, w, h, dev));

        let uncorrected_val: f32 = uncorrected.into_data().iter::<f32>().next().unwrap();
        let corrected_val: f32 = corrected.into_data().iter::<f32>().next().unwrap();
        assert!(
            corrected_val < uncorrected_val,
            "expected corrected loss ({corrected_val}) < uncorrected loss ({uncorrected_val})"
        );
    }
}
