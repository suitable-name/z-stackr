//! Integration tests for [`stacker_nn::traits::FusionModel`] —
//! verifies that the crate's tiled inference ([`fuse_stack`]) and planar
//! bridge ([`fuse_planar`]) are genuinely generic over the trait, not
//! secretly tied to [`stacker_nn::model::FocusMergeNet`], by driving a
//! minimal test-only model through both.

use burn::{backend::NdArray, prelude::Backend, tensor::Tensor};
use stacker_core::image::PlanarImage;
use stacker_nn::{
    FusionModel, FusionStrategy,
    bridge::fuse_planar,
    infer::{TileConfig, fuse_stack},
    traits::MergeStep,
};

type B = NdArray;

fn device() -> burn::prelude::Device<B> {
    burn::prelude::Device::<B>::default()
}

/// A source-independent confidence floor `MeanMerge` reports whenever it
/// exceeds the incoming `target_conf`.
const CONF_FLOOR: f32 = 0.25;

/// Minimal test-only [`FusionModel`] implementation: no learnable
/// parameters, no spatial mixing at all (every output pixel depends only on
/// the co-located input pixels), so `receptive_field() == 0`.
///
/// * `merged = 0.5 * target + 0.5 * source` — a running average, so folding
///   it over a stack via [`fuse_stack`] produces a predictable weighted mean
///   we can assert against exactly.
/// * `conf = max(target_conf, CONF_FLOOR)` — a source-independent floor,
///   deliberately not derived from `source` at all, to prove the trait
///   doesn't require confidence to depend on every input.
struct MeanMerge;

impl<B: Backend> FusionModel<B> for MeanMerge {
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::Pairwise
    }

    fn merge(
        &self,
        target: Tensor<B, 4>,
        target_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
    ) -> MergeStep<B> {
        let merged = target.mul_scalar(0.5_f32).add(source.mul_scalar(0.5_f32));
        // max(target_conf, CONF_FLOOR): CONF_FLOOR is a plain scalar, so
        // `clamp_min` is exactly a source-independent confidence floor.
        let conf = target_conf.clamp_min(CONF_FLOOR);
        MergeStep { merged, conf }
    }

    fn receptive_field(&self) -> usize {
        0
    }
}

/// A constant-valued `[3, H, W]` frame (every pixel = `v`).
fn const_frame(h: usize, w: usize, v: f32) -> Tensor<B, 3> {
    Tensor::<B, 3>::ones([3, h, w], &device()).mul_scalar(v)
}

// ---------------------------------------------------------------------------
// fuse_stack over the toy model
// ---------------------------------------------------------------------------

/// Three constant-valued tiny frames folded through [`fuse_stack`] with
/// [`MeanMerge`] must produce the exact running-average value the recurrence
/// implies: `((f0 * 0.5 + f1 * 0.5) * 0.5 + f2 * 0.5)`.
#[test]
fn fuse_stack_with_mean_merge_matches_expected_running_average() {
    let model = MeanMerge;
    let (h, w) = (6, 6);
    let (v0, v1, v2) = (0.0_f32, 1.0_f32, 0.4_f32);
    let frames = vec![
        const_frame(h, w, v0),
        const_frame(h, w, v1),
        const_frame(h, w, v2),
    ];
    let cfg = TileConfig {
        tile: 4,
        overlap: 1,
    }; // forces the tiled path even though receptive_field() == 0
    let out = fuse_stack(&model, &frames, cfg, &device()).expect("fuse_stack failed");

    assert_eq!(out.dims(), [3, h, w], "output shape mismatch");

    let step1 = 0.5_f32.mul_add(v1, 0.5 * v0);
    let expected = 0.5_f32.mul_add(v2, 0.5 * step1);

    let data = out.into_data();
    for v in data.iter::<f32>() {
        assert!(
            (v - expected).abs() < 1e-4,
            "pixel {v} != expected running average {expected}"
        );
    }
}

/// Single-frame stack: the fold seeds `result = frames[0]` unconditionally
/// (no merge step runs), so the output must equal the input exactly,
/// regardless of the model.
#[test]
fn fuse_stack_with_mean_merge_single_frame_is_seed_frame() {
    let model = MeanMerge;
    let frame = const_frame(5, 7, 0.73);
    let cfg = TileConfig::default();
    let out = fuse_stack(&model, &[frame], cfg, &device()).expect("fuse_stack failed");
    assert_eq!(out.dims(), [3, 5, 7]);
    let data = out.into_data();
    for v in data.iter::<f32>() {
        assert!((v - 0.73).abs() < 1e-5, "expected 0.73, got {v}");
    }
}

// ---------------------------------------------------------------------------
// fuse_planar round trip over the toy model
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random planar image (mirrors `tests/bridge.rs`'s
/// helper of the same name/shape).
fn planar(w: usize, h: usize, seed: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    let mut t = seed.fract().abs() + 0.1;
    for i in 0..w * h {
        t = (t * 1.1 + 0.3).fract();
        img.luma[i] = t;
        img.chroma_a[i] = (t - 0.5) * 0.2;
        img.chroma_b[i] = (0.5 - t) * 0.1;
    }
    img
}

/// [`fuse_planar`] with the toy [`MeanMerge`] model must round-trip
/// dimensions and produce finite output — proves `bridge::fuse_planar` is
/// generic over [`FusionModel`], not hard-wired to `FocusMergeNet`.
#[test]
fn fuse_planar_with_mean_merge_preserves_dims() {
    let model = MeanMerge;
    let dev = device();
    let frames = vec![planar(10, 8, 0.0), planar(10, 8, 1.0), planar(10, 8, 2.0)];
    let cfg = TileConfig {
        tile: 6,
        overlap: 2,
    };
    let out = fuse_planar::<B, _>(&model, &frames, cfg, &dev).expect("fuse_planar failed");

    assert_eq!((out.width, out.height), (10, 8));
    assert!(out.luma.iter().all(|v| v.is_finite()));
    assert!(out.chroma_a.iter().all(|v| v.is_finite()));
    assert!(out.chroma_b.iter().all(|v| v.is_finite()));
}

/// `TileConfig::for_model` derives its overlap from `receptive_field()`
/// (clamped to the crate's historical 96 px floor); for a zero-receptive-field
/// model like `MeanMerge` it must fall back exactly to that floor.
#[test]
fn tile_config_for_model_uses_floor_for_zero_receptive_field() {
    let model = MeanMerge;
    let cfg = TileConfig::for_model::<B, _>(&model);
    assert_eq!(cfg.overlap, 96);
    assert_eq!(cfg.tile, TileConfig::default().tile);
}
