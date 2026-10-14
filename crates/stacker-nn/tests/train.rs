//! Integration tests for the rollout training primitives in
//! [`stacker_nn::train`].
#![cfg(all(feature = "ndarray", feature = "autodiff"))]

use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamWConfig,
    prelude::{Backend, Tensor},
    tensor::Distribution,
};
use stacker_nn::{
    data::MergeSample,
    loss::FocusFusionLossConfig,
    model::{FocusMergeNet, FocusMergeNetConfig, ModelSize},
    train::{cosine_lr, rollout_loss, rollout_train, scheduled_sampling_prob},
};

/// Build a random `MergeSample<Bk>` of spatial size `h x w`.
fn rand_sample<Bk: Backend>(h: usize, w: usize, device: &Bk::Device) -> MergeSample<Bk> {
    let rgb =
        |c: usize| Tensor::<Bk, 3>::random([c, h, w], Distribution::Uniform(0.0, 1.0), device);
    MergeSample {
        target: rgb(3),
        target_conf: rgb(1),
        source: rgb(3),
        gt_merged: rgb(3),
        gt_conf: rgb(1),
        occlusion: rgb(1),
    }
}

#[test]
fn rollout_loss_forward_finite() {
    type B = NdArray;
    let device = <B as Backend>::Device::default();

    let model = FocusMergeNetConfig::new().init::<B>(&device);
    let loss = FocusFusionLossConfig::new().init();

    let steps = vec![
        rand_sample::<B>(16, 16, &device),
        rand_sample::<B>(16, 16, &device),
        rand_sample::<B>(16, 16, &device),
    ];
    let use_prediction = [false, false, false];

    let terms = rollout_loss(&model, &loss, &steps, &use_prediction);
    assert!(terms.total.into_scalar().is_finite());
}

#[test]
fn rollout_loss_scheduled_sampling_backward() {
    // With scheduled sampling enabled (feed back predictions), the rollout must
    // still be differentiable end-to-end.
    type Ab = Autodiff<NdArray>;
    let device = <Ab as Backend>::Device::default();

    let model = FocusMergeNetConfig::new().init::<Ab>(&device);
    let loss = FocusFusionLossConfig::new().init();

    let steps = vec![
        rand_sample::<Ab>(12, 12, &device),
        rand_sample::<Ab>(12, 12, &device),
    ];
    // Step 1 feeds back the model's own prediction.
    let use_prediction = [false, true];

    let terms = rollout_loss(&model, &loss, &steps, &use_prediction);
    let scalar: f32 = terms.total.clone().into_scalar();
    assert!(scalar.is_finite(), "rollout total {scalar} not finite");

    let _grads = terms.total.backward();
}

#[test]
fn rollout_train_updates_once_per_step_with_finite_losses() {
    // NN-8: backward + optimiser step per rollout step (one graph at a time).
    type Ab = Autodiff<NdArray>;
    let device = <Ab as Backend>::Device::default();

    let model = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device);
    let loss = FocusFusionLossConfig::new().init();
    let mut optim = AdamWConfig::new().init::<Ab, FocusMergeNet<Ab>>();

    let steps = vec![
        rand_sample::<Ab>(12, 12, &device),
        rand_sample::<Ab>(12, 12, &device),
        rand_sample::<Ab>(12, 12, &device),
    ];
    let probe = |m: &FocusMergeNet<Ab>| -> f32 {
        let s = &steps[0];
        m.forward(
            s.target.clone().unsqueeze_dim::<4>(0),
            s.target_conf.clone().unsqueeze_dim::<4>(0),
            s.source.clone().unsqueeze_dim::<4>(0),
        )
        .merged
        .mean()
        .into_scalar()
    };
    let before = probe(&model);

    // Step 2 is fed the model's own (detached) prediction.
    let (model, stats) = rollout_train(
        model,
        &mut optim,
        1e-2,
        &loss,
        &steps,
        &[false, false, true],
    );

    assert_eq!(stats.updates, 3, "one optimiser update per rollout step");
    for (name, v) in [
        ("total", stats.total),
        ("charbonnier", stats.charbonnier),
        ("gradient", stats.gradient),
        ("sharpness", stats.sharpness),
        ("confidence", stats.confidence),
    ] {
        assert!(v.is_finite() && v >= 0.0, "{name} = {v}");
    }
    let after = probe(&model);
    assert!(
        (after - before).abs() > 1e-7,
        "three AdamW updates must change the model ({before} -> {after})"
    );
}

#[test]
fn cosine_lr_warmup_and_decay() {
    let base = 1.0_f64;
    let total = 100;
    let warmup = 10;

    // Warm-up: strictly increasing toward base, below base.
    let lr_early = cosine_lr(base, 0, total, warmup);
    let lr_mid_warmup = cosine_lr(base, 5, total, warmup);
    assert!(lr_early < lr_mid_warmup);
    assert!(lr_mid_warmup <= base);

    // Mid training is below base; end approaches 0.
    let lr_mid = cosine_lr(base, 55, total, warmup);
    let lr_end = cosine_lr(base, 100, total, warmup);
    assert!(lr_mid < base);
    assert!(lr_end < lr_mid);
    assert!(lr_end < 1e-9, "lr at end should approach 0, got {lr_end}");
}

#[test]
fn scheduled_sampling_ramps_from_zero_to_max() {
    let total = 10;
    let max = 0.8_f64;
    assert!((scheduled_sampling_prob(0, total, max) - 0.0).abs() < 1e-12);
    let last = scheduled_sampling_prob(total - 1, total, max);
    assert!(
        (last - max).abs() < 1e-12,
        "last epoch should reach max, got {last}"
    );
    // Monotonic non-decreasing.
    let a = scheduled_sampling_prob(3, total, max);
    let b = scheduled_sampling_prob(7, total, max);
    assert!(a <= b);
    // Degenerate single-epoch run → 0.
    assert!((scheduled_sampling_prob(0, 1, max) - 0.0).abs() < 1e-12);
}
