//! The losses of the non-pairwise strategies, the photometric fine-tune term and the
//! trainer's random source. Shapes and finiteness, the identity-at-initialisation
//! contract of the aligners, and that one optimiser step moves the weights; no
//! accuracy claims (no model is trained here).
#![cfg(all(feature = "ndarray", feature = "autodiff"))]

use burn::{
    backend::{Autodiff, NdArray},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::{Backend, Tensor},
    tensor::{Distribution, TensorData},
};
use stacker_nn::{
    data::{AlignSample, BatchSample, PairAlignSample},
    loss::{
        corner_alignment::BatchAlignmentLossConfig, focus_batch::FocusBatchLossConfig,
        pair_corner_alignment::PairAlignmentLossConfig,
    },
    model::{
        ModelSize, batch_align::BatchAlignNetConfig, batch_merge::BatchMergeNetConfig,
        fusion_align::FusionAlignNetConfig,
    },
    train::{
        DELTA_REG_WEIGHT, align_loss, batch_loss, fusion_align_loss, photometric_finetune_term,
        rng::{Rng, random_crop},
    },
};

type Ab = Autodiff<NdArray>;

fn device() -> <Ab as Backend>::Device {
    <Ab as Backend>::Device::default()
}

fn random<const D: usize>(shape: [usize; D]) -> Tensor<Ab, D> {
    Tensor::random(shape, Distribution::Uniform(0.0, 1.0), &device())
}

/// `n` stacked 3x3 matrices, each a translation by `(tx, ty)` in normalised units.
fn translations(n: usize, tx: f32, ty: f32) -> Vec<f32> {
    let one = [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0];
    (0..n).flat_map(|_| one).collect()
}

fn scalar(t: Tensor<Ab, 1>) -> f32 {
    assert_eq!(t.dims(), [1]);
    t.into_scalar()
}

// ---------------------------------------------------------------------------
// Batch fusion
// ---------------------------------------------------------------------------

fn batch_sample(s: usize, size: usize) -> BatchSample<Ab> {
    BatchSample {
        stack: random([s, 3, size, size]),
        gt_merged: random([3, size, size]),
        masks: random([s, 1, size, size]),
        occlusion: random([1, size, size]),
    }
}

#[test]
fn batch_loss_is_finite_and_differentiable() {
    let model = BatchMergeNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    let loss = FocusBatchLossConfig::new().init();
    let sample = batch_sample(3, 16);

    let terms = batch_loss(&model, &loss, &sample);
    let value = scalar(terms.clone());
    assert!(value.is_finite() && value >= 0.0, "loss {value}");
    let _grads = terms.backward();
}

#[test]
fn one_optimiser_step_moves_the_batch_model() {
    let device = device();
    let mut model = BatchMergeNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device);
    let loss = FocusBatchLossConfig::new().init();
    let sample = batch_sample(3, 16);
    let mut optim = AdamWConfig::new().init::<Ab, _>();

    let before = scalar(batch_loss(&model, &loss, &sample));
    let terms = batch_loss(&model, &loss, &sample);
    let grads = GradientsParams::from_grads(terms.backward(), &model);
    model = optim.step(1e-2, model, grads);
    let after = scalar(batch_loss(&model, &loss, &sample));

    assert!(after.is_finite());
    assert!(
        (after - before).abs() > 0.0,
        "the step changed nothing: {before} -> {after}"
    );
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

fn align_sample(s: usize, size: usize, tx: f32, ty: f32) -> AlignSample<Ab> {
    AlignSample {
        stack: random([s, 3, size, size]),
        gt_matrices: Tensor::from_data(
            TensorData::new(translations(s, tx, ty), [s, 3, 3]),
            &device(),
        ),
    }
}

#[test]
fn align_loss_is_zero_for_identity_ground_truth_at_initialisation() {
    // The head starts at zero, so every predicted matrix is the identity; against
    // an identity ground truth the corner loss and the regulariser both vanish.
    let model = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    let loss = BatchAlignmentLossConfig::new().init();
    let value = scalar(align_loss(&model, &loss, &align_sample(3, 32, 0.0, 0.0)));
    assert!((0.0..1e-4).contains(&value), "loss {value}");
}

#[test]
fn align_loss_grows_with_the_misalignment_and_is_differentiable() {
    let model = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    let loss = BatchAlignmentLossConfig::new().init();

    let small = scalar(align_loss(&model, &loss, &align_sample(3, 32, 0.05, 0.0)));
    let large = scalar(align_loss(&model, &loss, &align_sample(3, 32, 0.20, 0.0)));
    assert!(small > 0.0 && large > small, "{small} then {large}");

    let _grads = align_loss(&model, &loss, &align_sample(3, 32, 0.1, -0.1)).backward();
}

#[test]
fn one_optimiser_step_moves_the_align_model() {
    let device = device();
    let mut model = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device);
    let loss = BatchAlignmentLossConfig::new().init();
    let sample = align_sample(3, 32, 0.1, 0.05);
    let mut optim = AdamWConfig::new().init::<Ab, _>();

    let before = scalar(align_loss(&model, &loss, &sample));
    let grads = GradientsParams::from_grads(align_loss(&model, &loss, &sample).backward(), &model);
    model = optim.step(1e-2, model, grads);
    let after = scalar(align_loss(&model, &loss, &sample));

    assert!(after.is_finite());
    assert!(
        (after - before).abs() > 0.0,
        "the step changed nothing: {before} -> {after}"
    );
}

#[test]
fn corr_radius_changes_the_head_but_not_the_loss_contract() {
    for radius in [1, 2] {
        let config = BatchAlignNetConfig::from_size(ModelSize::Xs).with_corr_radius(radius);
        assert_eq!(config.corr_radius, radius);
        let model = config.init::<Ab>(&device());
        let loss = BatchAlignmentLossConfig::new().init();
        let value = scalar(align_loss(&model, &loss, &align_sample(2, 32, 0.0, 0.0)));
        assert!(value.is_finite());
    }
}

fn pair_sample(size: usize, tx: f32, ty: f32) -> PairAlignSample<Ab> {
    PairAlignSample {
        reference: random([3, size, size]),
        frame: random([3, size, size]),
        gt_matrix: Tensor::from_data(TensorData::new(translations(1, tx, ty), [3, 3]), &device()),
    }
}

#[test]
fn fusion_align_loss_is_zero_for_identity_ground_truth_at_initialisation() {
    let model = FusionAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    let loss = PairAlignmentLossConfig::new().init();
    let value = scalar(fusion_align_loss(&model, &loss, &pair_sample(32, 0.0, 0.0)));
    assert!((0.0..1e-4).contains(&value), "loss {value}");
}

#[test]
fn fusion_align_loss_is_positive_off_identity_and_differentiable() {
    let model = FusionAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    let loss = PairAlignmentLossConfig::new().init();
    let terms = fusion_align_loss(&model, &loss, &pair_sample(32, 0.15, -0.1));
    let value = scalar(terms.clone());
    assert!(value.is_finite() && value > 0.0, "loss {value}");
    let _grads = terms.backward();
}

#[test]
fn delta_regulariser_is_small_and_positive() {
    const { assert!(DELTA_REG_WEIGHT > 0.0 && DELTA_REG_WEIGHT <= 1e-3) };
}

// ---------------------------------------------------------------------------
// Photometric fine-tune
// ---------------------------------------------------------------------------

#[test]
fn photometric_term_is_a_finite_scalar_for_a_real_stack() {
    let model = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    let stack = random([3, 3, 32, 32]);

    let term = photometric_finetune_term(&model, &stack).expect("three frames give a term");
    let value = scalar(term.clone());
    assert!(value.is_finite() && value >= 0.0, "term {value}");
    let _grads = term.backward();
}

#[test]
fn photometric_term_needs_a_non_reference_frame() {
    let model = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    assert!(photometric_finetune_term(&model, &random([1, 3, 32, 32])).is_none());
}

// ---------------------------------------------------------------------------
// Random source
// ---------------------------------------------------------------------------

#[test]
fn rng_is_deterministic_and_shuffles_permute() {
    let mut a = Rng::new(7);
    let mut b = Rng::new(7);
    let from_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
    let from_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
    assert_eq!(from_a, from_b);
    assert_ne!(
        from_a,
        (0..8).map(|_| Rng::new(8).next_u64()).collect::<Vec<_>>()
    );

    let mut rng = Rng::new(1);
    assert_eq!(rng.below(0), 0);
    for _ in 0..100 {
        assert!(rng.below(5) < 5);
        let f = rng.next_f64();
        assert!((0.0..1.0).contains(&f));
    }
    let mut items: Vec<usize> = (0..20).collect();
    rng.shuffle(&mut items);
    let mut sorted = items.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    assert_ne!(
        items, sorted,
        "20 items stayed in order (astronomically unlikely)"
    );
}

#[test]
fn random_crop_fits_and_clamps() {
    let mut rng = Rng::new(1234);
    for _ in 0..50 {
        let crop = random_crop((80, 100), 32, &mut rng);
        assert_eq!(crop.size, 32);
        assert!(crop.top + crop.size <= 80 && crop.left + crop.size <= 100);
    }
    // Larger than the image: clamp to the smaller side, one valid position.
    let crop = random_crop((15, 20), 256, &mut rng);
    assert_eq!((crop.size, crop.top), (15, 0));
    assert!(crop.left + crop.size <= 20);
}
