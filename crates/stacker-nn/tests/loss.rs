//! Integration tests for [`stacker_nn::loss::FocusFusionLoss`].
#![cfg(all(feature = "ndarray", feature = "autodiff"))]

use burn::{
    backend::{Autodiff, NdArray},
    prelude::{Backend, Tensor},
    tensor::{Distribution, TensorData},
};
use stacker_nn::{
    loss::{FocusFusionLoss, FocusFusionLossConfig, sharpness_map},
    model::{FocusMergeNetConfig, MergeOutput},
};

type B = NdArray;

fn device() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

fn rand4(shape: [usize; 4]) -> Tensor<B, 4> {
    Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &device())
}

fn default_loss() -> FocusFusionLoss {
    FocusFusionLossConfig::new().init()
}

/// Build a hand-constructed `MergeOutput` from random tensors.
fn fake_pred(nb: usize, nh: usize, nw: usize) -> MergeOutput<B> {
    MergeOutput {
        merged: rand4([nb, 3, nh, nw]),
        conf: rand4([nb, 1, nh, nw]),
        alpha: rand4([nb, 1, nh, nw]),
    }
}

#[test]
fn loss_forward_finite_scalars() {
    let loss = default_loss();
    let pred = fake_pred(1, 16, 16);
    let gt_merged = rand4([1, 3, 16, 16]);
    let gt_conf = rand4([1, 1, 16, 16]);
    let source = rand4([1, 3, 16, 16]);
    let occlusion = rand4([1, 1, 16, 16]);

    let terms = loss.forward(&pred, &gt_merged, gt_conf, source, &occlusion);

    let check = |name: &str, val: f32| {
        assert!(val.is_finite(), "{name} is not finite: {val}");
        assert!(val >= 0.0, "{name} is negative: {val}");
    };
    check("total", terms.total.into_scalar());
    check("charbonnier", terms.charbonnier.into_scalar());
    check("gradient", terms.gradient.into_scalar());
    check("sharpness", terms.sharpness.into_scalar());
    check("confidence", terms.confidence.into_scalar());
}

#[test]
fn loss_forward_finite_non_square() {
    // Ensure the loss handles non-square inputs (important for tiling).
    let loss = default_loss();
    let pred = fake_pred(2, 15, 17);
    let gt_merged = rand4([2, 3, 15, 17]);
    let gt_conf = rand4([2, 1, 15, 17]);
    let source = rand4([2, 3, 15, 17]);
    let occlusion = rand4([2, 1, 15, 17]);

    let terms = loss.forward(&pred, &gt_merged, gt_conf, source, &occlusion);
    assert!(terms.total.into_scalar().is_finite());
}

/// `[1, c, 16, 16]` tensor from a per-pixel function `f(channel, y, x)`.
fn map4(c: usize, f: impl Fn(usize, usize, usize) -> f32) -> Tensor<B, 4> {
    let mut data = Vec::with_capacity(c * 16 * 16);
    for ch in 0..c {
        for y in 0..16 {
            for x in 0..16 {
                data.push(f(ch, y, x));
            }
        }
    }
    Tensor::<B, 4>::from_data(TensorData::new(data, [1, c, 16, 16]), &device())
}

/// A prediction that equals `gt` plus `error(y, x)` on all three colour channels.
fn pred_with_error(gt: &Tensor<B, 4>, error: impl Fn(usize, usize) -> f32) -> MergeOutput<B> {
    MergeOutput {
        merged: gt.clone().add(map4(3, |_, y, x| error(y, x))),
        conf: map4(1, |_, _, _| 0.5),
        alpha: map4(1, |_, _, _| 0.5),
    }
}

const fn checker(y: usize, x: usize) -> f32 {
    if (y + x).is_multiple_of(2) {
        0.25
    } else {
        -0.25
    }
}

#[test]
fn gradient_term_is_occlusion_weighted() {
    // NN-7: with occlusion = 1 everywhere and min weight 0, every pixel has zero
    // weight, so the gradient term (and every other masked term) is exactly 0.
    let loss = FocusFusionLossConfig::new()
        .with_occlusion_min_weight(0.0)
        .init();
    let gt = map4(3, |_, _, _| 0.5);
    let pred = pred_with_error(&gt, checker);
    let all_occluded = map4(1, |_, _, _| 1.0);
    let none_occluded = map4(1, |_, _, _| 0.0);
    let gt_conf = map4(1, |_, _, _| 0.0);

    let t = loss.forward(&pred, &gt, gt_conf.clone(), gt.clone(), &all_occluded);
    let gradient: f32 = t.gradient.into_scalar();
    assert!(
        gradient.abs() < 1e-9,
        "gradient not occlusion-weighted: {gradient}"
    );
    let charbonnier: f32 = t.charbonnier.into_scalar();
    assert!(
        charbonnier.abs() < 1e-9,
        "charbonnier not weighted: {charbonnier}"
    );

    let free = loss.forward(&pred, &gt, gt_conf, gt.clone(), &none_occluded);
    assert!(
        free.gradient.into_scalar() > 0.01,
        "the unweighted gradient error must be visible"
    );
}

#[test]
fn gradient_error_inside_the_occluded_region_is_down_weighted() {
    // The same checkerboard error, once confined to the occluded left half and once
    // to the free right half. Only the one finite difference that straddles the
    // boundary sees the occluded pixels at half weight, so the occluded case must
    // score far lower.
    let loss = FocusFusionLossConfig::new()
        .with_occlusion_min_weight(0.0)
        .init();
    let gt = map4(3, |_, _, _| 0.5);
    let occlusion = map4(1, |_, _, x| if x < 8 { 1.0 } else { 0.0 });
    let gt_conf = map4(1, |_, _, _| 0.0);

    let in_occluded = pred_with_error(&gt, |y, x| if x < 8 { checker(y, x) } else { 0.0 });
    let in_free = pred_with_error(&gt, |y, x| if x >= 8 { checker(y, x) } else { 0.0 });

    let low = loss
        .forward(&in_occluded, &gt, gt_conf.clone(), gt.clone(), &occlusion)
        .gradient
        .into_scalar();
    let high = loss
        .forward(&in_free, &gt, gt_conf, gt.clone(), &occlusion)
        .gradient
        .into_scalar();
    assert!(high > 0.01, "free-region error must register: {high}");
    assert!(
        low < 0.25 * high,
        "occluded error {low} should be well below free error {high}"
    );
}

#[test]
fn sharpness_hinge_is_taken_against_the_reference_not_the_current_source() {
    // NN-7: a flat prediction against a flat current source costs nothing, but when
    // an earlier frame was sharp at these pixels the output must not be blurrier
    // than that frame.
    let loss = FocusFusionLossConfig::new().init();
    let flat = map4(3, |_, _, _| 0.5);
    let sharp = map4(3, |_, y, x| 0.5 + checker(y, x));
    let pred = MergeOutput {
        merged: flat.clone(),
        conf: map4(1, |_, _, _| 0.5),
        alpha: map4(1, |_, _, _| 0.5),
    };
    let gt_conf = map4(1, |_, _, _| 0.0);
    let occlusion = map4(1, |_, _, _| 0.0);

    // Old behaviour (reference = the current, blurry source): no penalty.
    let vs_source = loss.forward(&pred, &flat, gt_conf.clone(), flat.clone(), &occlusion);
    assert!(vs_source.sharpness.into_scalar() < 1e-6);

    // Reference = max sharpness over the frames so far, including the sharp one.
    let reference = sharpness_map(flat.clone()).max_pair(sharpness_map(sharp));
    let vs_reference = loss.forward_with_reference(&pred, &flat, gt_conf, reference, &occlusion);
    assert!(
        vs_reference.sharpness.into_scalar() > 0.05,
        "a blurry output must be penalised against the sharpest earlier frame"
    );
}

#[test]
fn loss_backward_end_to_end() {
    // Full autodiff smoke test: forward + backward through model + loss.
    type Ab = Autodiff<NdArray>;

    let dev = <Ab as Backend>::Device::default();

    let rand4_ab =
        |shape: [usize; 4]| Tensor::<Ab, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &dev);

    let model = FocusMergeNetConfig::new().init::<Ab>(&dev);

    let target = rand4_ab([1, 3, 8, 8]);
    let target_conf = rand4_ab([1, 1, 8, 8]);
    let source = rand4_ab([1, 3, 8, 8]);

    let pred = model.forward(target, target_conf, source.clone());

    let loss = FocusFusionLossConfig::new().init();
    let gt_merged = rand4_ab([1, 3, 8, 8]);
    let gt_conf = rand4_ab([1, 1, 8, 8]);
    let occlusion = rand4_ab([1, 1, 8, 8]);

    let terms = loss.forward(&pred, &gt_merged, gt_conf, source, &occlusion);

    let scalar: f32 = terms.total.clone().into_scalar();
    assert!(scalar.is_finite(), "total loss {scalar} not finite");

    // backward() not panicking proves end-to-end differentiability.
    let _grads = terms.total.backward();
}
