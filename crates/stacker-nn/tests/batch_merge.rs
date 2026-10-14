//! Integration tests for [`stacker_nn::model::BatchMergeNet`], its loss and the
//! tiled whole-stack inference path.
//!
//! No trained weights exist, so these pin structural properties only: shapes,
//! softmax gating, order invariance, the receptive field, tile invariance and the
//! loss terms. There are no accuracy tests.
#![cfg(feature = "ndarray")]
#![allow(
    clippy::suboptimal_flops,
    reason = "plain tolerance arithmetic in assertions"
)]

use burn::{
    backend::{Autodiff, NdArray},
    prelude::{Backend, Tensor},
    tensor::{Distribution, TensorData},
};
use stacker_nn::{
    infer::{TileConfig, fuse_batch_tiled, fuse_stack},
    loss::{FocusBatchLossConfig, helpers::occlusion_weight_map},
    model::{BatchMergeNet, BatchMergeNetConfig, FocusMergeNetConfig, ModelSize},
    traits::{BatchFusionLoss, FusionModel, FusionStrategy},
};

type F32 = NdArray;
type F64 = NdArray<f64>;

fn rand<B: Backend, const D: usize>(shape: [usize; D], dev: &B::Device) -> Tensor<B, D> {
    Tensor::<B, D>::random(shape, Distribution::Uniform(0.0, 1.0), dev)
}

fn values<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().convert::<f64>().to_vec::<f64>().unwrap()
}

#[test]
fn every_preset_initialises_and_preserves_the_spatial_size() {
    let dev = <F32 as Backend>::Device::default();
    for size in ModelSize::ALL {
        let model: BatchMergeNet<F32> = BatchMergeNetConfig::from_size(size).init(&dev);
        let stack = rand::<F32, 5>([1, 3, 3, 12, 14], &dev);
        let (merged, alpha) = model.forward_with_alpha(stack.clone());
        assert_eq!(merged.dims(), [1, 3, 12, 14], "merged {}", size.as_str());
        assert_eq!(alpha.dims(), [1, 3, 1, 12, 14], "alpha {}", size.as_str());
        assert_eq!(model.forward(stack).dims(), [1, 3, 12, 14]);
    }
}

#[test]
fn presets_use_the_pairwise_geometry_and_the_same_radius() {
    for size in ModelSize::ALL {
        let batch = BatchMergeNetConfig::from_size(size);
        let pairwise = FocusMergeNetConfig::from_size(size);
        assert_eq!(batch.width, pairwise.width, "{}", size.as_str());
        assert_eq!(batch.depth, pairwise.depth, "{}", size.as_str());
        assert_eq!(batch.norm_groups, 1, "per-pixel norm over all channels");
        assert_eq!(
            batch.receptive_field_radius(),
            pairwise.receptive_field_radius(),
            "{}",
            size.as_str()
        );
    }
}

#[test]
fn the_gate_is_a_softmax_over_the_stack_axis() {
    let dev = <F32 as Backend>::Device::default();
    let model: BatchMergeNet<F32> = BatchMergeNetConfig::from_size(ModelSize::Xs).init(&dev);
    let (_merged, alpha) = model.forward_with_alpha(rand::<F32, 5>([2, 4, 3, 9, 7], &dev));
    let sums = values(alpha.clone().sum_dim(1));
    assert!(
        sums.iter().all(|s| (s - 1.0).abs() < 1e-5),
        "alpha does not sum to 1 over S: {sums:?}"
    );
    assert!(values(alpha).iter().all(|a| (0.0..=1.0).contains(a)));
}

#[test]
fn it_is_a_batch_strategy_fusion_model() {
    let dev = <F32 as Backend>::Device::default();
    let model: BatchMergeNet<F32> = BatchMergeNetConfig::from_size(ModelSize::M).init(&dev);
    assert_eq!(model.strategy(), FusionStrategy::Batch);
    assert_eq!(model.receptive_field(), 40);
    assert_eq!(TileConfig::for_model(&model).overlap, 80);
    let stack = rand::<F32, 5>([1, 2, 3, 8, 8], &dev);
    assert_eq!(model.fuse_batch(stack).dims(), [1, 3, 8, 8]);
}

/// The merged image is a function of the *set* of frames: there is no recurrence
/// and no positional input.
#[test]
fn permuting_the_frames_does_not_change_the_result() {
    let dev = <F64 as Backend>::Device::default();
    let model: BatchMergeNet<F64> = BatchMergeNetConfig::from_size(ModelSize::Xs).init(&dev);
    let stack = rand::<F64, 5>([1, 3, 3, 10, 10], &dev);
    let frame = |i: usize| stack.clone().slice([0..1, i..i + 1, 0..3, 0..10, 0..10]);
    let reversed = Tensor::cat(vec![frame(2), frame(1), frame(0)], 1);
    let a = values(model.forward(stack.clone()));
    let b = values(model.forward(reversed));
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(worst < 1e-9, "order changed the result by {worst}");
}

#[test]
fn the_backward_pass_runs() {
    type Ab = Autodiff<NdArray>;
    let dev = <Ab as Backend>::Device::default();
    let model: BatchMergeNet<Ab> = BatchMergeNetConfig::from_size(ModelSize::Xs).init(&dev);
    let loss = model.forward(rand::<Ab, 5>([1, 2, 3, 8, 8], &dev)).mean();
    assert!(loss.clone().into_scalar().is_finite());
    let _grads = loss.backward();
}

// ---------------------------------------------------------------------------
// Receptive field and tile invariance
// ---------------------------------------------------------------------------

#[test]
fn a_perturbation_never_reaches_beyond_the_stated_radius() {
    let dev = <F64 as Backend>::Device::default();
    let config = BatchMergeNetConfig::from_size(ModelSize::Xs);
    let radius = config.receptive_field_radius(); // dilations 1, 2 -> 16
    assert_eq!(radius, 16);
    let model: BatchMergeNet<F64> = config.init(&dev);

    let side = 65;
    let centre = side / 2;
    let stack = rand::<F64, 5>([1, 2, 3, side, side], &dev);
    // +1.0 on one red pixel of frame 1.
    let mut bump = vec![0.0_f64; 2 * 3 * side * side];
    bump[3 * side * side + centre * side + centre] = 1.0;
    let bump = Tensor::<F64, 5>::from_data(TensorData::new(bump, [1, 2, 3, side, side]), &dev);

    let base = values(model.forward(stack.clone()));
    let moved = values(model.forward(stack.add(bump)));

    let (mut reach, mut leak) = (0_usize, 0.0_f64);
    for ch in 0..3 {
        for row in 0..side {
            for col in 0..side {
                let idx = (ch * side + row) * side + col;
                let dist = row.abs_diff(centre).max(col.abs_diff(centre));
                let diff = (base[idx] - moved[idx]).abs();
                if diff > 1e-12 {
                    reach = reach.max(dist);
                }
                if dist > radius {
                    leak = leak.max(diff);
                }
            }
        }
    }
    assert!(leak <= 1e-12, "change of {leak} beyond the radius {radius}");
    assert!(reach <= radius, "reach {reach} > radius {radius}");
    assert!(
        reach >= radius / 4,
        "reach {reach} is under a quarter of the radius: the field is not used"
    );
}

#[test]
fn tiled_fusion_matches_the_whole_image_away_from_the_blend_band() {
    let dev = <F32 as Backend>::Device::default();
    let config = BatchMergeNetConfig::from_size(ModelSize::Xs);
    let model: BatchMergeNet<F32> = config.init(&dev);
    let radius = config.receptive_field_radius();

    let (h, w) = (24, 96);
    let frames: Vec<Tensor<F32, 3>> = (0..3).map(|_| rand::<F32, 3>([3, h, w], &dev)).collect();

    // One tile larger than the image: the whole-image fast path.
    let whole = fuse_stack(
        &model,
        &frames,
        TileConfig {
            tile: usize::MAX,
            overlap: 0,
        },
        &dev,
    )
    .unwrap();
    // 64-px tiles at x = 0 and x = 32 with the documented overlap 2R = 32.
    let tiled = fuse_stack(
        &model,
        &frames,
        TileConfig {
            tile: 64,
            overlap: 2 * radius,
        },
        &dev,
    )
    .unwrap();

    let (whole, tiled) = (values(whole), values(tiled));
    let mut worst = 0.0_f64;
    for ch in 0..3 {
        for y in 0..h {
            for x in (0..32).chain(64..96) {
                let i = (ch * h + y) * w + x;
                worst = worst.max((whole[i] - tiled[i]).abs());
            }
        }
    }
    assert!(
        worst < 1e-4,
        "tiled batch fusion differs from the whole image by {worst}"
    );
}

#[test]
fn progress_fires_once_per_tile_row_and_once_for_a_single_tile() {
    let dev = <F32 as Backend>::Device::default();
    let model: BatchMergeNet<F32> = BatchMergeNetConfig::from_size(ModelSize::Xs).init(&dev);

    // 70 x 30 with 40-px tiles and overlap 16: row offsets 0, 24, 30.
    let frames: Vec<Tensor<F32, 3>> = (0..2).map(|_| rand::<F32, 3>([3, 70, 30], &dev)).collect();
    let mut rows = Vec::new();
    let out = fuse_batch_tiled(
        &model,
        &frames,
        TileConfig {
            tile: 40,
            overlap: 16,
        },
        &dev,
        |row, preview| {
            assert_eq!(preview.dims(), [3, 70, 30]);
            rows.push(row);
        },
    )
    .unwrap();
    assert_eq!(out.dims(), [3, 70, 30]);
    assert_eq!(rows, [0, 1, 2]);
    assert!(values(out).iter().all(|v| v.is_finite()));

    let mut calls = Vec::new();
    let small: Vec<Tensor<F32, 3>> = (0..2).map(|_| rand::<F32, 3>([3, 16, 16], &dev)).collect();
    fuse_batch_tiled(&model, &small, TileConfig::default(), &dev, |row, _| {
        calls.push(row);
    })
    .unwrap();
    assert_eq!(calls, [0]);
}

#[test]
fn tiled_batch_fusion_validates_its_input() {
    let dev = <F32 as Backend>::Device::default();
    let model: BatchMergeNet<F32> = BatchMergeNetConfig::from_size(ModelSize::Xs).init(&dev);
    let none: Vec<Tensor<F32, 3>> = Vec::new();
    assert!(matches!(
        fuse_batch_tiled(&model, &none, TileConfig::default(), &dev, |_, _| {}),
        Err(stacker_nn::infer::InferError::EmptyStack)
    ));
    let mismatched = vec![
        rand::<F32, 3>([3, 8, 8], &dev),
        rand::<F32, 3>([3, 8, 9], &dev),
    ];
    assert!(matches!(
        fuse_batch_tiled(&model, &mismatched, TileConfig::default(), &dev, |_, _| {}),
        Err(stacker_nn::infer::InferError::ShapeMismatch { index: 1, .. })
    ));
}

// ---------------------------------------------------------------------------
// FocusBatchLoss
// ---------------------------------------------------------------------------

fn scalar(t: Tensor<F32, 1>) -> f64 {
    values(t)[0]
}

/// `[stack, masks]` for a 2-frame, 12 x 12 scene whose frame 0 is in focus on the
/// left half and frame 1 on the right half.
fn scene(dev: <F32 as Backend>::Device) -> [Tensor<F32, 5>; 2] {
    let (h, w) = (12, 12);
    let mut masks = vec![0.0_f32; 2 * h * w];
    for y in 0..h {
        for x in 0..w {
            if x < w / 2 {
                masks[y * w + x] = 1.0;
            } else {
                masks[h * w + y * w + x] = 1.0;
            }
        }
    }
    [
        rand::<F32, 5>([1, 2, 3, h, w], &dev),
        Tensor::<F32, 5>::from_data(TensorData::new(masks, [1, 2, 1, h, w]), &dev),
    ]
}

#[test]
fn a_perfect_prediction_has_a_near_zero_reconstruction_loss() {
    let dev = <F32 as Backend>::Device::default();
    let loss = FocusBatchLossConfig::new().init();
    let gt = rand::<F32, 4>([1, 3, 12, 12], &dev);
    let total = scalar(BatchFusionLoss::<F32>::forward(&loss, &gt, &gt));
    // Charbonnier floor sqrt(eps^2) = 1e-3 times its weight; the gradient term is 0.
    assert!(
        total < 2e-3,
        "reconstruction loss of a perfect prediction: {total}"
    );
    let worse = rand::<F32, 4>([1, 3, 12, 12], &dev);
    assert!(scalar(BatchFusionLoss::<F32>::forward(&loss, &worse, &gt)) > total);
}

#[test]
fn every_term_of_the_full_loss_responds() {
    let dev = <F32 as Backend>::Device::default();
    let loss = FocusBatchLossConfig::new().init();
    let [stack, masks] = scene(dev);
    let occlusion = Tensor::<F32, 4>::zeros([1, 1, 12, 12], &dev);
    let gt = rand::<F32, 4>([1, 3, 12, 12], &dev);
    let pred = rand::<F32, 4>([1, 3, 12, 12], &dev);

    // The target gate: frame 0 on the left, frame 1 on the right.
    let target = masks.clone();
    let wrong = Tensor::cat(
        vec![
            masks.clone().slice([0..1, 1..2, 0..1, 0..12, 0..12]),
            masks.clone().slice([0..1, 0..1, 0..1, 0..12, 0..12]),
        ],
        1,
    );

    let terms_good = loss.forward_terms(&pred, &target, &gt, &stack, &masks, &occlusion);
    let terms_bad = loss.forward_terms(&pred, &wrong, &gt, &stack, &masks, &occlusion);
    assert!(
        scalar(terms_good.gate.clone()) < 1e-4,
        "a correct gate costs ~nothing"
    );
    assert!(
        scalar(terms_bad.gate.clone()) > 0.5,
        "a swapped gate must be penalised"
    );
    assert!(scalar(terms_bad.total) > scalar(terms_good.total.clone()));
    for term in [
        &terms_good.charbonnier,
        &terms_good.gradient,
        &terms_good.sharpness,
    ] {
        assert!(scalar(term.clone()).is_finite());
    }
    let total = loss.forward_full(&pred, &target, &gt, &stack, &masks, &occlusion);
    assert!((scalar(total) - scalar(terms_good.total)).abs() < 1e-6);
}

#[test]
fn occlusion_down_weights_every_term() {
    let dev = <F32 as Backend>::Device::default();
    let loss = FocusBatchLossConfig::new().init();
    let [stack, masks] = scene(dev);
    let gt = rand::<F32, 4>([1, 3, 12, 12], &dev);
    let pred = rand::<F32, 4>([1, 3, 12, 12], &dev);
    let alpha = rand::<F32, 5>([1, 2, 1, 12, 12], &dev);

    let none = Tensor::<F32, 4>::zeros([1, 1, 12, 12], &dev);
    let all = Tensor::<F32, 4>::ones([1, 1, 12, 12], &dev);
    let open = loss.forward_terms(&pred, &alpha, &gt, &stack, &masks, &none);
    let edge = loss.forward_terms(&pred, &alpha, &gt, &stack, &masks, &all);

    // Every term is weighted by w = 1 - (1 - 0.25) * occlusion = 0.25 here.
    for (name, open, edge) in [
        ("charbonnier", open.charbonnier, edge.charbonnier),
        ("gradient", open.gradient, edge.gradient),
        ("sharpness", open.sharpness, edge.sharpness),
    ] {
        let (open, edge) = (scalar(open), scalar(edge));
        assert!(
            (edge - 0.25 * open).abs() < 1e-5 + 1e-4 * open,
            "{name}: {edge} is not 0.25 x {open}"
        );
    }
    // The gate term is a weighted mean, so a uniform weight cancels in it.
    assert!((scalar(open.gate) - scalar(edge.gate)).abs() < 1e-5);

    let w = occlusion_weight_map(&all, 0.25);
    assert!(values(w).iter().all(|v| (v - 0.25).abs() < 1e-6));
}
