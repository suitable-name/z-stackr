//! Integration tests for [`stacker_nn::model::FocusMergeNet`].
#![cfg(all(feature = "ndarray", feature = "autodiff"))]

use burn::{
    backend::{Autodiff, NdArray},
    prelude::{Backend, Tensor},
    tensor::Distribution,
};
use stacker_nn::model::{FocusMergeNet, FocusMergeNetConfig, ModelSize};

type B = NdArray;

fn device() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

/// Build a default model on `NdArray`.
fn default_model() -> FocusMergeNet<B> {
    FocusMergeNetConfig::new().init(&device())
}

/// Random rank-4 tensor on `NdArray`.
fn rand_tensor(shape: [usize; 4]) -> Tensor<B, 4> {
    Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &device())
}

#[test]
fn forward_shape_square() {
    let model = default_model();
    let target = rand_tensor([1, 3, 16, 16]);
    let target_conf = rand_tensor([1, 1, 16, 16]);
    let source = rand_tensor([1, 3, 16, 16]);

    let out = model.forward(target, target_conf, source);

    assert_eq!(out.merged.dims(), [1, 3, 16, 16]);
    assert_eq!(out.conf.dims(), [1, 1, 16, 16]);
    assert_eq!(out.alpha.dims(), [1, 1, 16, 16]);
}

#[test]
fn forward_shape_non_square_odd() {
    // Verify Same padding preserves non-square / odd spatial dimensions.
    let model = default_model();
    let target = rand_tensor([1, 3, 15, 17]);
    let target_conf = rand_tensor([1, 1, 15, 17]);
    let source = rand_tensor([1, 3, 15, 17]);

    let out = model.forward(target, target_conf, source);

    assert_eq!(out.merged.dims(), [1, 3, 15, 17]);
    assert_eq!(out.conf.dims(), [1, 1, 15, 17]);
    assert_eq!(out.alpha.dims(), [1, 1, 15, 17]);
}

/// The blend weight and the confidence are sigmoid outputs, so any value in
/// `[0, 1]` is trivially true. What distinguishes a working head from a dead one
/// is that, for a freshly initialised model on random input, the values stay
/// strictly inside the open interval (no saturated lane) and actually vary with
/// the input (not a constant).
#[test]
fn alpha_and_conf_are_interior_and_not_constant() {
    let model = default_model();
    let target = rand_tensor([1, 3, 16, 16]);
    let target_conf = rand_tensor([1, 1, 16, 16]);
    let source = rand_tensor([1, 3, 16, 16]);

    let out = model.forward(target, target_conf, source);

    for (name, t) in [("alpha", out.alpha), ("conf", out.conf)] {
        let lo: f32 = t.clone().min().into_scalar();
        let hi: f32 = t.max().into_scalar();
        assert!(lo > 0.0, "{name} min {lo} is not above 0 (saturated)");
        assert!(hi < 1.0, "{name} max {hi} is not below 1 (saturated)");
        assert!(
            hi - lo > 1e-4,
            "{name} is constant over the image (range {})",
            hi - lo
        );
    }
}

#[test]
fn backward_pass_no_panic() {
    // Use the autodiff backend to verify end-to-end differentiability.
    type Ab = Autodiff<NdArray>;

    let dev = <Ab as Backend>::Device::default();
    let model: FocusMergeNet<Ab> = FocusMergeNetConfig::new().init(&dev);

    let rand4 =
        |shape: [usize; 4]| Tensor::<Ab, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &dev);

    let target = rand4([1, 3, 8, 8]);
    let target_conf = rand4([1, 1, 8, 8]);
    let source = rand4([1, 3, 8, 8]);

    let out = model.forward(target, target_conf, source);
    let loss = out.merged.mean();
    let scalar: f32 = loss.clone().into_scalar();
    assert!(scalar.is_finite(), "loss {scalar} is not finite");

    // backward() not panicking proves the graph is differentiable.
    let _grads = loss.backward();
}

#[test]
fn all_presets_forward_shape() {
    // Every preset must initialise and run while preserving spatial dims.
    let dev = device();
    for size in ModelSize::ALL {
        let model: FocusMergeNet<B> = FocusMergeNetConfig::from_size(size).init(&dev);
        let target = rand_tensor([1, 3, 16, 16]);
        let target_conf = rand_tensor([1, 1, 16, 16]);
        let source = rand_tensor([1, 3, 16, 16]);

        let out = model.forward(target, target_conf, source);
        assert_eq!(
            out.merged.dims(),
            [1, 3, 16, 16],
            "preset {} changed spatial dims",
            size.as_str()
        );
        assert_eq!(out.conf.dims(), [1, 1, 16, 16]);
    }
}

#[test]
fn model_size_parse_roundtrips() {
    for size in ModelSize::ALL {
        assert_eq!(ModelSize::parse(size.as_str()), Some(size));
        assert_eq!(ModelSize::parse(&size.as_str().to_uppercase()), Some(size));
    }
    assert_eq!(ModelSize::parse("nonsense"), None);
}
