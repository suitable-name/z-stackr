//! Integration tests for [`stacker_nn::infer`].

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Distribution, Tensor},
};
use stacker_nn::{
    FusionModel, FusionStrategy,
    infer::{InferError, TileConfig, fuse_stack},
    model::FocusMergeNetConfig,
};

type B = NdArray;

fn device() -> burn::prelude::Device<B> {
    burn::prelude::Device::<B>::default()
}

fn default_model() -> stacker_nn::model::FocusMergeNet<B> {
    FocusMergeNetConfig::new().init(&device())
}

/// Random rank-3 tensor `[3, H, W]` on `NdArray`.
fn rand_frame(h: usize, w: usize) -> Tensor<B, 3> {
    Tensor::<B, 3>::random([3, h, w], Distribution::Uniform(0.0, 1.0), &device())
}

// ---------------------------------------------------------------------------
// Test 1: single-frame stack — output shape matches input.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_single_frame_is_identity_shape() {
    let model = default_model();
    let frame = rand_frame(12, 12);
    let cfg = TileConfig {
        tile: 32,
        overlap: 4,
    };
    let out = fuse_stack(&model, &[frame], cfg, &device()).expect("fuse_stack failed");
    assert_eq!(out.dims(), [3, 12, 12], "output shape mismatch");

    // Check all values are finite.
    let data = out.into_data();
    for v in data.iter::<f32>() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
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
// Test 4: repeatable — same model + input produces the same output within
// a tight tolerance. NOT bit-exact: the ndarray backend parallelises
// reductions (rayon), so f32 accumulation order — and therefore the last
// ULP of a sum — can differ between two runs of the identical computation.
// Repeatability within a small epsilon is the guarantee inference actually
// provides; bit-equality assertions here are flaky by construction.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_repeatable_within_tolerance() {
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
        assert!(
            (a - b).abs() < 1e-6,
            "outputs differ beyond tolerance: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: empty stack returns EmptyStack error.
// ---------------------------------------------------------------------------
#[test]
fn fuse_stack_empty_is_error() {
    let model = default_model();
    let cfg = TileConfig::default();
    let result = fuse_stack::<B, _>(&model, &[], cfg, &device());
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
// Test 7: fuse_batch_tiled equivalence (tiled vs untiled batch fusion).
// ---------------------------------------------------------------------------

/// Minimal test-only [`FusionModel`] implementing [`FusionStrategy::Batch`]:
/// averages the frames across the `S` dimension. No spatial mixing at all, so
/// `receptive_field() == 0` and tiling (with any overlap) must reproduce the
/// exact untiled result — a deterministic, per-pixel-predictable batch
/// reduction that lets the test assert tiled/untiled equivalence exactly.
struct AverageBatchMerge;

impl<B: Backend> FusionModel<B> for AverageBatchMerge {
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::Batch
    }

    fn fuse_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4> {
        stack.mean_dim(1).squeeze_dim(1)
    }

    fn receptive_field(&self) -> usize {
        0
    }
}

/// Tiled batch fusion ([`fuse_batch_tiled`](stacker_nn::infer), exercised
/// indirectly via [`fuse_stack`] dispatching on [`FusionStrategy::Batch`])
/// must reproduce the untiled (single-tile) result closely: run the same
/// stack once with a `TileConfig` that forces multiple tiles and once with a
/// `TileConfig` that fits the whole image in a single tile, and compare.
#[test]
fn fuse_batch_tiled_matches_untiled() {
    let model = AverageBatchMerge;
    let dev = device();
    let (h, w) = (24, 24);
    let frames: Vec<Tensor<B, 3>> = (0..4).map(|_| rand_frame(h, w)).collect();

    // Forces tiling: tile smaller than the image.
    let tiled_cfg = TileConfig {
        tile: 10,
        overlap: 3,
    };
    // Fits in a single tile: no tiling.
    let untiled_cfg = TileConfig {
        tile: 64,
        overlap: 3,
    };

    let out_tiled = fuse_stack(&model, &frames, tiled_cfg, &dev).expect("tiled fuse_stack failed");
    let out_untiled =
        fuse_stack(&model, &frames, untiled_cfg, &dev).expect("untiled fuse_stack failed");

    assert_eq!(out_tiled.dims(), [3, h, w]);
    assert_eq!(out_untiled.dims(), [3, h, w]);

    let tiled_data: Vec<f32> = out_tiled.into_data().iter::<f32>().collect();
    let untiled_data: Vec<f32> = out_untiled.into_data().iter::<f32>().collect();

    for (a, b) in tiled_data.iter().zip(untiled_data.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "tiled vs untiled batch fusion mismatch: {a} vs {b}"
        );
    }
}
