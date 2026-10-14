//! Integration tests for trained-model discovery (save / discover / load).
//!
//! Ordering and manifest-robustness tests live in `discovery_order.rs`.
#![cfg(feature = "ndarray")]

use burn::{
    backend::NdArray,
    module::Module,
    prelude::Backend,
    record::CompactRecorder,
    tensor::{Tensor, TensorData},
};
use stacker_nn::{
    discovery::discover_models,
    infer::load_weights,
    model::{FocusMergeNet, FocusMergeNetConfig, ModelSize},
};

type B = NdArray;

#[test]
fn discover_missing_dir_is_empty() {
    let dir = std::path::Path::new("/nonexistent/xyzzy/models");
    assert!(discover_models(dir).unwrap().is_empty());
}

#[test]
fn discover_and_load_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = <B as Backend>::Device::default();

    // Save a tiny model as "tiny.mpk" with a matching minimal manifest.
    let model = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    model
        .clone()
        .save_file(tmp.path().join("tiny"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(tmp.path().join("tiny.json"), r#"{"size":"xs"}"#).unwrap();

    // A bare .mpk without a manifest must be ignored.
    model
        .save_file(tmp.path().join("orphan"), &CompactRecorder::new())
        .unwrap();

    let found = discover_models(tmp.path()).unwrap();
    assert_eq!(
        found.len(),
        1,
        "orphan .mpk without manifest should be skipped"
    );
    assert_eq!(found[0].name, "tiny");
    assert_eq!(found[0].manifest.size, ModelSize::Xs);

    // The discovered entry loads the saved weights: its output equals that of a
    // model loaded straight from the same file, and differs from a freshly
    // initialised one (so `load` cannot silently return a random model).
    let loaded: FocusMergeNet<B> = found[0].load(&dev).unwrap();
    let config = FocusMergeNetConfig::from_size(ModelSize::Xs);
    let direct: FocusMergeNet<B> = load_weights(&config, &found[0].weights_path, &dev).unwrap();
    let fresh: FocusMergeNet<B> = config.init(&dev);
    let (loaded, direct, fresh) = (probe(&loaded), probe(&direct), probe(&fresh));
    assert!(
        loaded
            .iter()
            .zip(&direct)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the discovered model does not reproduce the weights in its file"
    );
    assert!(
        loaded
            .iter()
            .zip(&fresh)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the discovered model equals a freshly initialised one"
    );
}

/// Forward output on a fixed deterministic input, as raw `f32` values.
fn probe(model: &FocusMergeNet<B>) -> Vec<f32> {
    let dev = <B as Backend>::Device::default();
    let ramp = |len: usize, shape: [usize; 4], phase: f32| {
        let data: Vec<f32> = (0..len)
            .map(|i| {
                f32::from(u16::try_from(i).unwrap())
                    .mul_add(0.05, phase)
                    .sin()
                    .abs()
            })
            .collect();
        Tensor::<B, 4>::from_data(TensorData::new(data, shape), &dev)
    };
    let out = model.forward(
        ramp(192, [1, 3, 8, 8], 0.0),
        ramp(64, [1, 1, 8, 8], 0.5),
        ramp(192, [1, 3, 8, 8], 1.0),
    );
    out.merged.into_data().to_vec::<f32>().unwrap()
}
