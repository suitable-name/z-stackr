//! NN-13: seeding Burn's global generator makes weight initialisation
//! reproducible, which is what `z-stackr-train --seed N` relies on.
//!
//! This is the only test in this file on purpose: the `NdArray` generator is
//! process-global, so two tests initialising models in parallel threads would
//! interleave draws from it.
#![cfg(feature = "ndarray")]

use burn::{
    backend::NdArray,
    prelude::{Backend, Int, Tensor},
};
use stacker_nn::model::{FocusMergeNetConfig, ModelSize};

type B = NdArray;

/// Seed, initialise an `xs` model, and return its output on a fixed input.
fn init_and_probe(seed: u64) -> Vec<f32> {
    B::seed(seed);
    let dev = <B as Backend>::Device::default();
    let model = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);

    let ramp = |range: std::ops::Range<i64>, shape: [usize; 4]| {
        Tensor::<B, 1, Int>::arange(range, &dev)
            .float()
            .mul_scalar(0.05)
            .sin()
            .reshape(shape)
    };
    let out = model.forward(
        ramp(0..192, [1, 3, 8, 8]),
        ramp(0..64, [1, 1, 8, 8]).abs(),
        ramp(7..199, [1, 3, 8, 8]),
    );
    out.merged
        .into_data()
        .to_vec::<f32>()
        .expect("the output is f32")
}

#[test]
fn same_seed_gives_identical_weights_and_a_different_seed_does_not() {
    let a = init_and_probe(1);
    let b = init_and_probe(1);
    let c = init_and_probe(2);

    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "output {i} differs for seed 1");
    }
    assert!(
        a.iter().zip(&c).any(|(x, y)| x.to_bits() != y.to_bits()),
        "seeds 1 and 2 produced the same weights"
    );
}
