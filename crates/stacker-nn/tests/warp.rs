//! Tests for the differentiable affine warp and the photometric loss built on it.
//!
//! The direction test is the important one: matrices map **reference
//! coordinates to source coordinates** (the pipeline convention), so warping the
//! source with the matrix that registers it reproduces the reference. It is
//! checked end to end through the pixel ↔ normalised conversions of
//! [`stacker_nn::bridge`], with a known integer shift.
#![cfg(feature = "ndarray")]
#![allow(
    clippy::suboptimal_flops,
    reason = "plain fixture arithmetic for the analytic texture"
)]

use burn::{
    backend::{Autodiff, NdArray},
    prelude::{Backend, Tensor},
    tensor::TensorData,
};
use stacker_nn::{
    bridge::{PixelMatrix, pixel_matrix_to_normalized},
    loss::photometric::{PHOTOMETRIC_CHARBONNIER_EPS, photometric_gradient_loss},
    warp::warp_affine,
};

type B = NdArray;

fn dev() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

fn float(v: i32) -> f32 {
    f32::from(i16::try_from(v).unwrap())
}

/// A smooth analytic texture, defined for any integer position (also outside the
/// image), so a shifted copy can be generated exactly.
fn texture(x: i32, y: i32) -> f32 {
    let (x, y) = (float(x), float(y));
    (0.7 * x + 0.3 * y).sin() + 0.5 * (0.4 * x - 0.9 * y).cos()
}

/// `[1, 1, h, w]` image whose pixel `(x, y)` is `texture(x - dx, y - dy)`.
fn shifted_texture(w: i32, h: i32, dx: i32, dy: i32) -> Tensor<B, 4> {
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            data.push(texture(x - dx, y - dy));
        }
    }
    Tensor::<B, 4>::from_data(
        TensorData::new(
            data,
            [
                1,
                1,
                usize::try_from(h).unwrap(),
                usize::try_from(w).unwrap(),
            ],
        ),
        &dev(),
    )
}

const fn translation(dx: f32, dy: f32) -> PixelMatrix {
    [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]
}

fn normalized(m: PixelMatrix, w: usize, h: usize) -> Tensor<B, 3> {
    let n = pixel_matrix_to_normalized(m, w, h);
    let flat: Vec<f32> = n.iter().flatten().copied().collect();
    Tensor::<B, 3>::from_data(TensorData::new(flat, [1, 3, 3]), &dev())
}

fn values(t: Tensor<B, 4>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn identity_is_a_no_op_and_everything_is_valid() {
    let (w, h) = (9, 7);
    let image = shifted_texture(9, 7, 0, 0);
    let out = warp_affine(image.clone(), normalized(translation(0.0, 0.0), w, h));
    let (got, want) = (values(out.warped), values(image));
    for (g, v) in got.iter().zip(&want) {
        assert!((g - v).abs() < 1e-5, "got {g}, want {v}");
    }
    assert!(values(out.valid).iter().all(|v| (v - 1.0).abs() < 1e-6));
}

/// The direction test. The scene point seen at reference pixel `p` sits at source
/// pixel `p + d` (so the registering matrix is `T(+d)`); `warp_affine(source,
/// T(+d))` therefore returns the reference wherever `p + d` is inside the source.
#[test]
fn warping_the_source_by_the_registering_matrix_reproduces_the_reference() {
    let (w, h) = (14_usize, 11_usize);
    let (dx, dy) = (3_i32, -2_i32);
    let reference = shifted_texture(14, 11, 0, 0);
    let source = shifted_texture(14, 11, dx, dy); // source(q) = texture(q - d)

    let m = normalized(translation(3.0, -2.0), w, h);
    let out = warp_affine(source.clone(), m);
    let (warped, valid, want) = (values(out.warped), values(out.valid), values(reference));

    let mut checked = 0;
    for y in 0..h {
        for x in 0..w {
            let inside = (0..14).contains(&(i32::try_from(x).unwrap() + dx))
                && (0..11).contains(&(i32::try_from(y).unwrap() + dy));
            let i = y * w + x;
            assert_eq!(valid[i] > 0.5, inside, "validity of ({x}, {y})");
            if inside {
                checked += 1;
                assert!(
                    (warped[i] - want[i]).abs() < 1e-4,
                    "({x}, {y}): warped {}, reference {}",
                    warped[i],
                    want[i]
                );
            }
        }
    }
    assert_eq!(checked, (14 - 3) * (11 - 2));

    // The opposite sign is a different transform: it must NOT reproduce the
    // reference, or the test above could not tell the directions apart.
    let wrong = warp_affine(source, normalized(translation(-3.0, 2.0), w, h));
    let wrong = values(wrong.warped);
    let worst = wrong
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst > 0.1, "the reversed matrix also matches ({worst})");
}

#[test]
fn a_non_integer_shift_interpolates_bilinearly() {
    // A horizontal ramp is reproduced exactly by bilinear interpolation.
    let (w, h) = (10_usize, 6_usize);
    let mut data = Vec::new();
    for _ in 0..h {
        for x in 0..10 {
            data.push(float(x));
        }
    }
    let ramp = Tensor::<B, 4>::from_data(TensorData::new(data, [1, 1, h, w]), &dev());
    let out = warp_affine(ramp, normalized(translation(0.5, 0.0), w, h));
    let (warped, valid) = (values(out.warped), values(out.valid));
    for y in 0..h {
        for x in 0..w - 1 {
            let i = y * w + x;
            assert!(valid[i] > 0.5);
            assert!((warped[i] - (float(i32::try_from(x).unwrap()) + 0.5)).abs() < 1e-4);
        }
        assert!(
            valid[y * w + w - 1] < 0.5,
            "x + 0.5 is past the last column"
        );
    }
}

#[test]
fn gradients_flow_to_the_matrix() {
    type Ab = Autodiff<NdArray>;
    let dev = <Ab as Backend>::Device::default();
    let image = Tensor::<Ab, 4>::from_data(shifted_texture(12, 10, 0, 0).into_data(), &dev);
    let identity = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let matrix = Tensor::<Ab, 3>::from_data(TensorData::new(identity.to_vec(), [1, 3, 3]), &dev)
        .require_grad();
    let target = Tensor::<Ab, 4>::zeros([1, 1, 10, 12], &dev);
    let loss = warp_affine(image, matrix.clone())
        .warped
        .sub(target)
        .powf_scalar(2.0)
        .mean();
    let grads = loss.backward();
    let grad = matrix.grad(&grads).expect("the matrix has a gradient");
    let g = grad.into_data().to_vec::<f32>().unwrap();
    assert!(g.iter().all(|v| v.is_finite()));
    assert!(g.iter().any(|v| v.abs() > 1e-6), "all-zero gradient: {g:?}");
}

#[test]
#[should_panic(expected = "H and W must both be >= 2")]
fn a_single_row_image_is_rejected() {
    let image = Tensor::<B, 4>::zeros([1, 1, 1, 8], &dev());
    let _ = warp_affine(image, normalized(translation(0.0, 0.0), 8, 2));
}

// ---------------------------------------------------------------------------
// Photometric loss
// ---------------------------------------------------------------------------

fn loss_value(frame: &Tensor<B, 4>, reference: &Tensor<B, 4>, m: Tensor<B, 3>) -> f32 {
    photometric_gradient_loss(frame.clone(), reference.clone(), m)
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0]
}

#[test]
fn the_photometric_loss_sits_at_the_epsilon_floor_for_identical_aligned_images() {
    let image = shifted_texture(24, 20, 0, 0);
    let loss = loss_value(&image, &image, normalized(translation(0.0, 0.0), 24, 20));
    assert!(
        loss < 2.0 * PHOTOMETRIC_CHARBONNIER_EPS,
        "identity warp of identical images should be at the floor, got {loss}"
    );
}

/// The frame shows the reference content moved by `d`; the loss is smallest (at
/// the epsilon floor) for the matrix `T(+d)` that registers it (reference →
/// frame), and clearly larger for no correction and for the reversed shift.
#[test]
fn the_photometric_loss_prefers_the_registering_matrix() {
    let (w, h) = (24_usize, 20_usize);
    let reference = shifted_texture(24, 20, 0, 0);
    let frame = shifted_texture(24, 20, 2, 1);

    let right = loss_value(&frame, &reference, normalized(translation(2.0, 1.0), w, h));
    let none = loss_value(&frame, &reference, normalized(translation(0.0, 0.0), w, h));
    let wrong = loss_value(
        &frame,
        &reference,
        normalized(translation(-2.0, -1.0), w, h),
    );

    assert!(
        right < 1.5 * PHOTOMETRIC_CHARBONNIER_EPS,
        "registered: {right}"
    );
    assert!(
        none > 10.0 * right,
        "uncorrected {none} vs registered {right}"
    );
    assert!(
        wrong > 10.0 * right,
        "reversed {wrong} vs registered {right}"
    );
}
