//! Integration tests for the planar<->tensor fusion bridge.
#![cfg(feature = "ndarray")]

use burn::{backend::NdArray, prelude::Backend};
use stacker_core::image::PlanarImage;
use stacker_nn::{
    ModelSize,
    bridge::{BridgeError, fuse_planar},
    infer::TileConfig,
    model::FocusMergeNetConfig,
};

type B = NdArray;

/// Deterministic pseudo-random planar image (no integer→float casts, to stay
/// clippy-clean in the test crate).
fn planar(w: usize, h: usize, seed: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    let mut t = seed.fract().abs() + 0.1;
    for i in 0..w * h {
        t = t.mul_add(1.1, 0.3).fract();
        img.luma[i] = t;
        img.chroma_a[i] = (t - 0.5) * 0.2;
        img.chroma_b[i] = (0.5 - t) * 0.1;
    }
    img
}

fn xs_model(dev: <B as Backend>::Device) -> stacker_nn::model::FocusMergeNet<B> {
    FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev)
}

#[test]
fn fuse_planar_shape_and_finite() {
    let dev = <B as Backend>::Device::default();
    let model = xs_model(dev);
    let frames = vec![
        planar(24, 20, 0.0),
        planar(24, 20, 1.0),
        planar(24, 20, 2.0),
    ];
    let out = fuse_planar(
        &model,
        &frames,
        TileConfig {
            tile: 16,
            overlap: 4,
        },
        &dev,
    )
    .unwrap();

    assert_eq!((out.width, out.height), (24, 20));
    assert!(out.luma.iter().all(|v| v.is_finite()));
    assert!(out.chroma_a.iter().all(|v| v.is_finite()));
    assert!(out.chroma_b.iter().all(|v| v.is_finite()));
}

#[test]
fn fuse_planar_empty_is_error() {
    let dev = <B as Backend>::Device::default();
    let model = xs_model(dev);
    let frames: Vec<PlanarImage<f32>> = Vec::new();
    assert!(matches!(
        fuse_planar(&model, &frames, TileConfig::default(), &dev),
        Err(BridgeError::EmptyStack)
    ));
}

#[test]
fn fuse_planar_shape_mismatch_is_error() {
    let dev = <B as Backend>::Device::default();
    let model = xs_model(dev);
    let frames = vec![planar(16, 16, 0.0), planar(16, 12, 1.0)];
    assert!(matches!(
        fuse_planar(&model, &frames, TileConfig::default(), &dev),
        Err(BridgeError::ShapeMismatch { index: 1, .. })
    ));
}
