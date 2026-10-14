//! Integration tests for the two alignment networks, [`BatchAlignNet`] and
//! [`FusionAlignNet`], and the corner losses.
//!
//! No trained weights exist, so these pin structure only: shapes for every preset,
//! the zero-initialised head (identity at init), the bounded affine
//! parameterisation (positive determinant, bounded translation), the frame-0
//! normalisation, the geometry checks and the losses. There are no accuracy tests.
#![cfg(feature = "ndarray")]
#![allow(
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "plain determinant arithmetic with the conventional matrix entry names"
)]

use burn::{
    backend::{Autodiff, NdArray},
    module::{Module, ModuleMapper, ParamId},
    prelude::{Backend, Tensor},
    tensor::Distribution,
};
use stacker_nn::{
    loss::{BatchAlignmentLossConfig, PairAlignmentLossConfig},
    model::{
        BatchAlignNet, BatchAlignNetConfig, FusionAlignNet, FusionAlignNetConfig, ModelSize,
        align_common::{MAX_CORR_RADIUS, check_align_geometry},
    },
    traits::{BatchAlignmentLoss, BatchAlignmentModel, PairAlignmentLoss, PairAlignmentModel},
};

type F32 = NdArray;
type F64 = NdArray<f64>;

fn rand<B: Backend, const D: usize>(shape: [usize; D], dev: &B::Device) -> Tensor<B, D> {
    Tensor::<B, D>::random(shape, Distribution::Uniform(0.0, 1.0), dev)
}

fn values<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().convert::<f64>().to_vec::<f64>().unwrap()
}

/// Adds Gaussian noise to the final linear layer (the `[width/2, 6]` weight and
/// the 6-element bias) only: turns the zero-initialised head into a non-trivial
/// one so the bounded parameterisation and the frame-0 normalisation are
/// exercised. Jittering the whole network would push every frame's raw output into
/// the same `tanh` saturation, making all frames predict one identical matrix.
struct Jitter(f64);

impl<B: Backend> ModuleMapper<B> for Jitter {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let dims = tensor.dims();
        let is_head = (D == 2 && dims[1] == 6) || (D == 1 && dims[0] == 6);
        if !is_head {
            return tensor;
        }
        let noise = Tensor::random(
            tensor.shape(),
            Distribution::Normal(0.0, self.0),
            &tensor.device(),
        );
        tensor.add(noise)
    }
}

const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}: element {i} is {g}, want {w}");
    }
}

#[test]
fn presets_have_the_documented_geometry() {
    let expected = [
        (ModelSize::Xs, 32, 1),
        (ModelSize::S, 48, 1),
        (ModelSize::M, 64, 2),
        (ModelSize::L, 96, 2),
        (ModelSize::Xl, 128, 3),
        (ModelSize::Xxl, 160, 3),
    ];
    for (size, width, depth) in expected {
        for (w, d, g, r) in [
            {
                let c = BatchAlignNetConfig::from_size(size);
                (c.width, c.depth, c.norm_groups, c.corr_radius)
            },
            {
                let c = FusionAlignNetConfig::from_size(size);
                (c.width, c.depth, c.norm_groups, c.corr_radius)
            },
        ] {
            assert_eq!((w, d), (width, depth), "{}", size.as_str());
            assert_eq!(g, 1, "per-pixel norm over all channels");
            assert_eq!(r, 4);
        }
    }
}

#[test]
fn batch_align_shapes_for_every_preset_and_odd_sizes() {
    let dev = <F32 as Backend>::Device::default();
    for size in ModelSize::ALL {
        let model: BatchAlignNet<F32> = BatchAlignNetConfig::from_size(size).init(&dev);
        // 13 x 21 is not a multiple of 8: the strided convolutions round up. Small
        // inputs keep the debug-build time of the big presets down.
        let (matrices, raw) = model.forward_with_params(rand::<F32, 5>([1, 2, 3, 13, 21], &dev));
        assert_eq!(matrices.dims(), [1, 2, 3, 3], "{}", size.as_str());
        assert_eq!(raw.dims(), [1, 2, 6], "{}", size.as_str());
    }
}

#[test]
fn fusion_align_shapes_for_every_preset_and_odd_sizes() {
    let dev = <F32 as Backend>::Device::default();
    for size in ModelSize::ALL {
        let model: FusionAlignNet<F32> = FusionAlignNetConfig::from_size(size).init(&dev);
        let (matrix, raw) = model.forward_with_params(
            rand::<F32, 4>([1, 3, 13, 21], &dev),
            rand::<F32, 4>([1, 3, 13, 21], &dev),
        );
        assert_eq!(matrix.dims(), [1, 3, 3], "{}", size.as_str());
        assert_eq!(raw.dims(), [1, 6], "{}", size.as_str());
    }
}

#[test]
fn an_untrained_network_predicts_the_identity() {
    let dev = <F64 as Backend>::Device::default();

    let batch: BatchAlignNet<F64> = BatchAlignNetConfig::from_size(ModelSize::Xs).init(&dev);
    let (matrices, raw) = batch.forward_with_params(rand::<F64, 5>([1, 3, 3, 24, 24], &dev));
    assert!(
        values(raw).iter().all(|v| v.abs() < 1e-12),
        "head is zero-initialised"
    );
    let want: Vec<f64> = IDENTITY.repeat(3);
    assert_close(&values(matrices), &want, 1e-9, "batch identity");

    let pair: FusionAlignNet<F64> = FusionAlignNetConfig::from_size(ModelSize::Xs).init(&dev);
    let m = pair.align_pair(
        rand::<F64, 4>([1, 3, 24, 24], &dev),
        rand::<F64, 4>([1, 3, 24, 24], &dev),
    );
    assert_close(&values(m), &IDENTITY, 1e-9, "pair identity");
}

#[test]
fn the_bounded_parameterisation_never_collapses_or_reflects() {
    let dev = <F64 as Backend>::Device::default();
    // Large jitter drives the raw outputs well into tanh saturation. The bounds are
    // applied to f32 constants, hence the 1e-6 slack on the f64 backend.
    let pair: FusionAlignNet<F64> = FusionAlignNetConfig::from_size(ModelSize::Xs)
        .init(&dev)
        .map(&mut Jitter(3.0));
    let m = values(pair.align_pair(
        rand::<F64, 4>([4, 3, 16, 16], &dev),
        rand::<F64, 4>([4, 3, 16, 16], &dev),
    ));
    let mut moved = false;
    for matrix in m.as_chunks::<9>().0 {
        let (a, b, tx) = (matrix[0], matrix[1], matrix[2]);
        let (c, d, ty) = (matrix[3], matrix[4], matrix[5]);
        assert_close(&matrix[6..], &[0.0, 0.0, 1.0], 1e-12, "last row");
        let det = a * d - b * c;
        // det = sx * sy with |ln s| <= 0.08.
        assert!(
            det >= (-0.16_f64).exp() - 1e-6 && det <= 0.16_f64.exp() + 1e-6,
            "determinant {det} outside the bounded range"
        );
        assert!(
            tx.abs() <= 0.30 + 1e-6 && ty.abs() <= 0.30 + 1e-6,
            "translation {tx}, {ty}"
        );
        moved |= (a - 1.0).abs() > 1e-3 || tx.abs() > 1e-3;
    }
    assert!(moved, "the jittered head still predicts the identity");
}

#[test]
fn frame_zero_is_exactly_the_identity_and_the_rest_are_not() {
    let dev = <F64 as Backend>::Device::default();
    let model: BatchAlignNet<F64> = BatchAlignNetConfig::from_size(ModelSize::Xs)
        .init(&dev)
        .map(&mut Jitter(0.3));
    let matrices = values(model.align_batch(rand::<F64, 5>([2, 3, 3, 16, 16], &dev)));
    // [N=2, S=3, 3, 3]: frame 0 of each stack is the identity.
    for n in 0..2 {
        let frame0 = &matrices[n * 27..n * 27 + 9];
        assert_close(frame0, &IDENTITY, 1e-9, "frame 0");
        for s in 1..3 {
            let m = &matrices[n * 27 + s * 9..n * 27 + (s + 1) * 9];
            assert!(m.iter().all(|v| v.is_finite()));
            assert!(m[0] * m[4] - m[1] * m[3] > 0.0, "reflection in frame {s}");
        }
    }
    let rest = &matrices[9..18];
    let off_identity = rest
        .iter()
        .zip(IDENTITY)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        off_identity > 1e-4,
        "frame 1 is still the identity: {off_identity}"
    );
}

#[test]
fn identical_inputs_give_identical_outputs() {
    let dev = <F32 as Backend>::Device::default();
    let model: BatchAlignNet<F32> = BatchAlignNetConfig::from_size(ModelSize::Xs)
        .init(&dev)
        .map(&mut Jitter(1.0));
    let stack = rand::<F32, 5>([1, 2, 3, 16, 16], &dev);
    let a = values(model.align_batch(stack.clone()));
    let b = values(model.align_batch(stack));
    assert_eq!(a, b);
}

#[test]
fn the_backward_pass_reaches_the_encoder() {
    type Ab = Autodiff<NdArray>;
    let dev = <Ab as Backend>::Device::default();
    let model: FusionAlignNet<Ab> = FusionAlignNetConfig::from_size(ModelSize::Xs)
        .init(&dev)
        .map(&mut Jitter(0.5));
    let (matrix, raw) = model.forward_with_params(
        rand::<Ab, 4>([1, 3, 16, 16], &dev),
        rand::<Ab, 4>([1, 3, 16, 16], &dev),
    );
    let target =
        Tensor::<Ab, 3>::from_data(burn::tensor::TensorData::new(I32.to_vec(), [1, 3, 3]), &dev);
    let loss = PairAlignmentLossConfig::new()
        .init()
        .forward(matrix, target)
        .add(raw.powf_scalar(2.0).mean().mul_scalar(1e-4_f32));
    assert!(loss.clone().into_scalar().is_finite());
    let _grads = loss.backward();
}

// ---------------------------------------------------------------------------
// Geometry checks
// ---------------------------------------------------------------------------

#[test]
fn geometry_checks_reject_what_the_network_cannot_be_built_with() {
    assert!(check_align_geometry(64, 1, 4).is_ok());
    assert!(check_align_geometry(64, 8, 4).is_ok());
    assert!(
        check_align_geometry(64, 1, 0).is_ok(),
        "radius 0 is a 1-channel volume"
    );
    assert!(check_align_geometry(0, 1, 4).is_err());
    assert!(check_align_geometry(64, 0, 4).is_err());
    assert!(check_align_geometry(63, 1, 4).is_err(), "odd width");
    assert!(
        check_align_geometry(36, 8, 4).is_err(),
        "groups must divide the width"
    );
    assert!(
        check_align_geometry(24, 8, 4).is_err(),
        "groups must divide width / 2"
    );
    assert!(check_align_geometry(64, 1, MAX_CORR_RADIUS).is_ok());
    assert!(check_align_geometry(64, 1, MAX_CORR_RADIUS + 1).is_err());
}

#[test]
#[should_panic(expected = "invalid alignment network geometry")]
fn an_odd_width_panics_at_init() {
    let dev = <F32 as Backend>::Device::default();
    let _ = BatchAlignNetConfig::new().with_width(33).init::<F32>(&dev);
}

#[test]
fn a_zero_correlation_radius_still_runs() {
    let dev = <F32 as Backend>::Device::default();
    let model: BatchAlignNet<F32> = BatchAlignNetConfig::from_size(ModelSize::Xs)
        .with_corr_radius(0)
        .init(&dev);
    let m = model.align_batch(rand::<F32, 5>([1, 2, 3, 16, 16], &dev));
    assert_eq!(m.dims(), [1, 2, 3, 3]);
}

// ---------------------------------------------------------------------------
// Corner losses
// ---------------------------------------------------------------------------

fn matrices(rows: &[[f32; 9]]) -> Tensor<F32, 3> {
    let dev = <F32 as Backend>::Device::default();
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Tensor::<F32, 3>::from_data(
        burn::tensor::TensorData::new(flat, [rows.len(), 3, 3]),
        &dev,
    )
}

fn scalar(t: Tensor<F32, 1>) -> f64 {
    values(t)[0]
}

const I32: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[test]
fn corner_loss_is_zero_for_equal_matrices_and_exact_for_a_translation() {
    let loss = PairAlignmentLossConfig::new().init();
    let same = matrices(&[I32]);
    assert!(scalar(loss.forward(same.clone(), same.clone())) < 1e-12);

    // A pure translation by (0.25, 0) moves every corner by exactly that, so the
    // MSE over all x/y/w components is 0.25^2 / 3 (x differs, y and w do not).
    let shifted = matrices(&[[1.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]);
    let got = scalar(loss.forward(shifted, same));
    assert!((got - 0.0625 / 3.0).abs() < 1e-7, "got {got}");
}

#[test]
fn the_batch_corner_loss_matches_the_pairwise_one_per_frame() {
    let pair = PairAlignmentLossConfig::new().init();
    let batch = BatchAlignmentLossConfig::new().init();
    let shifted = [1.0, 0.0, 0.1, 0.0, 1.0, -0.2, 0.0, 0.0, 1.0];
    let a = matrices(&[I32, shifted, I32, I32]);
    let b = matrices(&[I32, I32, I32, I32]);
    let stacked_a = a.clone().reshape([2, 2, 3, 3]);
    let stacked_b = b.clone().reshape([2, 2, 3, 3]);
    let want = scalar(pair.forward(a, b));
    let got = scalar(batch.forward(stacked_a, stacked_b));
    assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    assert!(got > 0.0);
}

#[test]
fn the_corner_loss_weighs_a_rotation_by_how_far_it_moves_the_corners() {
    let loss = PairAlignmentLossConfig::new().init();
    let id = matrices(&[I32]);
    // A 0.1 rad rotation moves a corner at radius sqrt(2) by about 0.14; a 0.01
    // translation moves it by 0.01. The loss must rank them accordingly.
    let (c, s) = (0.1_f32.cos(), 0.1_f32.sin());
    let rotated = matrices(&[[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]]);
    let nudged = matrices(&[[1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]);
    let rot = scalar(loss.forward(rotated, id.clone()));
    let nud = scalar(loss.forward(nudged, id));
    assert!(rot > 50.0 * nud, "rotation {rot} vs translation {nud}");
}
