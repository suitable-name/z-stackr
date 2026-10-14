//! Integration tests for [`stacker_nn::infer`].
#![cfg(feature = "ndarray")]

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Distribution, Tensor},
};
use stacker_nn::{
    infer::{InferError, TileConfig, fuse_stack},
    model::FocusMergeNetConfig,
};

type B = NdArray;

fn device() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

fn default_model() -> stacker_nn::model::FocusMergeNet<B> {
    FocusMergeNetConfig::new().init(&device())
}

/// Random rank-3 tensor `[3, H, W]` on `NdArray`.
fn rand_frame(h: usize, w: usize) -> Tensor<B, 3> {
    Tensor::<B, 3>::random([3, h, w], Distribution::Uniform(0.0, 1.0), &device())
}

/// All values of a tensor as `f32`.
fn values(t: Tensor<B, 3>) -> Vec<f32> {
    t.into_data().iter::<f32>().collect()
}

// ---------------------------------------------------------------------------
// Test 1: single-frame stack — the fold never calls the model, so the output
// is the input frame itself, value for value.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_single_frame_is_the_frame_itself() {
    let model = default_model();
    let frame = rand_frame(12, 12);
    let cfg = TileConfig {
        tile: 32,
        overlap: 4,
    };
    let out = fuse_stack(&model, std::slice::from_ref(&frame), cfg, &device())
        .expect("fuse_stack failed");
    assert_eq!(out.dims(), [3, 12, 12], "output shape mismatch");

    let got = values(out);
    let want = values(frame);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "value {i}: {g} vs {w}");
    }
}

// ---------------------------------------------------------------------------
// Test 1b: with two different frames the model IS invoked, so the result is
// neither frame.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_two_frames_invokes_the_model() {
    let model = default_model();
    let first = rand_frame(12, 12);
    let second = rand_frame(12, 12);
    let cfg = TileConfig {
        tile: 32,
        overlap: 4,
    };
    let out = fuse_stack(&model, &[first.clone(), second.clone()], cfg, &device())
        .expect("fuse_stack failed");

    let merged = values(out);
    for (name, frame) in [("first", first), ("second", second)] {
        let reference = values(frame);
        let worst = merged
            .iter()
            .zip(&reference)
            .map(|(m, r)| (m - r).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst > 1e-4,
            "the merge of two different frames equals the {name} frame (max diff {worst})"
        );
    }
    assert!(merged.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// Test 2: small image stays in fast path (no tiling).
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_small_no_tiling() {
    let model = default_model();
    let frames: Vec<Tensor<B, 3>> = (0..3).map(|_| rand_frame(16, 16)).collect();
    // tile > image size → fast path
    let cfg = TileConfig {
        tile: 32,
        overlap: 8,
    };
    let out = fuse_stack(&model, &frames, cfg, &device()).expect("fuse_stack failed");
    assert_eq!(out.dims(), [3, 16, 16]);

    let data = out.into_data();
    for v in data.iter::<f32>() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}

// ---------------------------------------------------------------------------
// Test 3: tiled path exercised (image larger than tile).
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_tiled_path() {
    let model = default_model();
    let frames: Vec<Tensor<B, 3>> = (0..3).map(|_| rand_frame(40, 40)).collect();
    // tile=16, overlap=4 → step=12; forces multiple tiles + clamped border tile
    let cfg = TileConfig {
        tile: 16,
        overlap: 4,
    };
    let out = fuse_stack(&model, &frames, cfg, &device()).expect("fuse_stack failed");
    assert_eq!(out.dims(), [3, 40, 40]);

    let data = out.into_data();
    for v in data.iter::<f32>() {
        assert!(v.is_finite(), "output contains non-finite/NaN value: {v}");
    }
}

// ---------------------------------------------------------------------------
// Test 4: deterministic — same model + input produces identical output.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_deterministic() {
    let model = default_model();
    let frames: Vec<Tensor<B, 3>> = (0..2).map(|_| rand_frame(20, 20)).collect();
    let cfg = TileConfig {
        tile: 16,
        overlap: 4,
    };

    let out1 = fuse_stack(&model, &frames, cfg, &device()).expect("first run failed");
    let out2 = fuse_stack(&model, &frames, cfg, &device()).expect("second run failed");

    let d1 = out1.into_data();
    let d2 = out2.into_data();

    let vals1: Vec<f32> = d1.iter::<f32>().collect();
    let vals2: Vec<f32> = d2.iter::<f32>().collect();

    assert_eq!(vals1.len(), vals2.len());
    for (a, b) in vals1.iter().zip(vals2.iter()) {
        assert!(a.to_bits() == b.to_bits(), "outputs differ: {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// Test 5: empty stack returns EmptyStack error.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_empty_is_error() {
    let model = default_model();
    let cfg = TileConfig::default();
    let result = fuse_stack::<B>(&model, &[], cfg, &device());
    assert!(
        matches!(result, Err(InferError::EmptyStack)),
        "expected EmptyStack, got: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 6: mismatched frame shapes return ShapeMismatch error.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_shape_mismatch_is_error() {
    let model = default_model();
    let frame0 = rand_frame(16, 16);
    let frame1 = rand_frame(12, 16); // different H
    let cfg = TileConfig::default();
    let result = fuse_stack(&model, &[frame0, frame1], cfg, &device());
    assert!(
        matches!(result, Err(InferError::ShapeMismatch { index: 1, .. })),
        "expected ShapeMismatch, got: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 7 (NN-13): the streaming fold gives bit-identical output to the slice
// fold, and pulls each frame only when it is folded in (the bridge relies on
// this to convert one frame at a time instead of holding a tensor copy of the
// whole stack).
// ---------------------------------------------------------------------------
#[test]
fn fuse_stream_matches_fuse_stack_and_is_lazy() {
    let model = default_model();
    let frames: Vec<Tensor<B, 3>> = (0..3).map(|_| rand_frame(20, 20)).collect();
    let cfg = TileConfig {
        tile: 16,
        overlap: 4,
    };

    let want = values(fuse_stack(&model, &frames, cfg, &device()).expect("slice fold"));

    let produced = std::cell::Cell::new(0_usize);
    let pulled_at_step = std::cell::RefCell::new(Vec::new());
    let lazy = frames.iter().map(|f| {
        produced.set(produced.get() + 1);
        f.clone()
    });
    let got = values(
        stacker_nn::infer::fuse_stream_with(&model, lazy, cfg, &device(), |index, _running| {
            pulled_at_step.borrow_mut().push((index, produced.get()));
        })
        .expect("stream fold"),
    );

    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(&want) {
        assert_eq!(a.to_bits(), b.to_bits(), "stream and slice folds differ");
    }
    // Step k has seen exactly k + 1 frames: none is converted ahead of its fold.
    assert_eq!(*pulled_at_step.borrow(), vec![(0, 1), (1, 2), (2, 3)]);
}
