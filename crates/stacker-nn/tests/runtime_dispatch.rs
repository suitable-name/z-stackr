//! Runtime dispatch: `LoadedModel` picks the pairwise or the whole-stack fusion
//! network from the manifest tag, `LoadedAlignModel` the whole-stack or the
//! streaming aligner, and each refuses the other family's entries.
#![cfg(feature = "ndarray")]

use std::path::Path;

use burn::{backend::NdArray, module::Module, prelude::Backend, record::CompactRecorder};
use stacker_core::image::PlanarImage;
use stacker_nn::{
    InferDevice, LoadedAlignModel, LoadedModel, ModelEntry, ModelManifest, RuntimeError,
    bridge::FrameThumbs,
    discovery::{DiscoveryError, discover_models},
    infer::TileConfig,
    model::{
        BatchAlignNetConfig, BatchMergeNetConfig, FocusMergeNetConfig, FusionAlignNetConfig,
        ModelSize,
    },
    runtime::{AlignNetKind, recommended_tile, recommended_tile_for},
    traits::FusionStrategy,
};

type B = NdArray;

fn dev() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

fn install<M: Module<B>>(dir: &Path, stem: &str, model: M, manifest: &ModelManifest) {
    model
        .save_file(dir.join(stem), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        dir.join(format!("{stem}.json")),
        serde_json::to_string(manifest).unwrap(),
    )
    .unwrap();
}

/// A directory holding one untrained `xs` model of every family.
fn models() -> (tempfile::TempDir, Vec<ModelEntry>) {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    let size = ModelSize::Xs;
    install(
        d,
        "focus",
        FocusMergeNetConfig::from_size(size).init::<B>(&dev()),
        &ModelManifest::from_size(size),
    );
    install(
        d,
        "batch",
        BatchMergeNetConfig::from_size(size).init::<B>(&dev()),
        &ModelManifest::from_size_batch(size),
    );
    install(
        d,
        "align",
        BatchAlignNetConfig::from_size(size).init::<B>(&dev()),
        &ModelManifest::from_size_align(size),
    );
    install(
        d,
        "pairs",
        FusionAlignNetConfig::from_size(size).init::<B>(&dev()),
        &ModelManifest::from_size_fusion_align(size),
    );
    let entries = discover_models(d).unwrap();
    (tmp, entries)
}

fn entry<'a>(entries: &'a [ModelEntry], name: &str) -> &'a ModelEntry {
    entries.iter().find(|e| e.name == name).unwrap()
}

fn frame(w: usize, h: usize, seed: f32) -> PlanarImage<f32> {
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

#[test]
fn fusion_loads_dispatch_on_the_tag_and_both_fuse_a_stack() {
    let (_tmp, entries) = models();
    let frames = vec![frame(24, 20, 0.0), frame(24, 20, 1.0), frame(24, 20, 2.0)];

    let focus = LoadedModel::load(entry(&entries, "focus"), InferDevice::Cpu).unwrap();
    let batch = LoadedModel::load(entry(&entries, "batch"), InferDevice::Cpu).unwrap();
    assert_eq!(focus.strategy(), FusionStrategy::Pairwise);
    assert_eq!(batch.strategy(), FusionStrategy::Batch);
    // Same preset, same radius: overlap 2 * 16.
    assert_eq!(focus.recommended_tile().overlap, 32);
    assert_eq!(batch.recommended_tile().overlap, 32);

    for model in [&focus, &batch] {
        let out = model.fuse(&frames, TileConfig::default()).unwrap();
        assert_eq!((out.width, out.height), (24, 20));
        assert!(out.luma.iter().all(|v| v.is_finite()));
    }

    // A pairwise model reports after every frame, a batch model after every tile row.
    let mut steps = 0;
    focus
        .fuse_with_progress(&frames, TileConfig::default(), |_| steps += 1)
        .unwrap();
    assert_eq!(steps, 3);
    let mut steps = 0;
    batch
        .fuse_with_progress(&frames, TileConfig::default(), |_| steps += 1)
        .unwrap();
    assert_eq!(steps, 1);

    // Tiled batch fusion goes through the same entry point.
    let tiled = batch
        .fuse(
            &frames,
            TileConfig {
                tile: 16,
                overlap: 4,
            },
        )
        .unwrap();
    assert_eq!((tiled.width, tiled.height), (24, 20));
}

#[test]
fn each_loader_refuses_the_other_family() {
    let (_tmp, entries) = models();
    for name in ["align", "pairs"] {
        assert!(
            matches!(
                LoadedModel::load(entry(&entries, name), InferDevice::Cpu),
                Err(RuntimeError::Discovery(
                    DiscoveryError::ArchitectureMismatch { .. }
                ))
            ),
            "{name} is not a fusion model"
        );
    }
    for name in ["focus", "batch"] {
        assert!(
            matches!(
                LoadedAlignModel::load(entry(&entries, name), InferDevice::Cpu),
                Err(RuntimeError::Discovery(
                    DiscoveryError::ArchitectureMismatch { .. }
                ))
            ),
            "{name} is not an alignment model"
        );
    }
}

#[test]
fn alignment_loads_dispatch_on_the_tag() {
    let (_tmp, entries) = models();
    let batch = LoadedAlignModel::load(entry(&entries, "align"), InferDevice::Cpu).unwrap();
    let pairs = LoadedAlignModel::load(entry(&entries, "pairs"), InferDevice::Cpu).unwrap();
    assert_eq!(batch.kind(), AlignNetKind::Batch);
    assert_eq!(pairs.kind(), AlignNetKind::Fusion);

    // An untrained network is the identity for every frame, whichever kind.
    let frames = vec![frame(32, 24, 0.0), frame(32, 24, 1.0), frame(32, 24, 2.0)];
    for model in [&batch, &pairs] {
        let matrices = model.align(&frames).unwrap();
        assert_eq!(matrices.len(), 3);
        for m in &matrices {
            for (i, row) in m.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((v - want).abs() < 1e-4, "{m:?}");
                }
            }
        }
    }

    // The streaming entry point takes pre-made thumbnails.
    let thumbs = FrameThumbs::from_frames(&frames).unwrap();
    assert_eq!(batch.align_thumbs(&thumbs).unwrap().len(), 3);
    assert!(matches!(
        batch.align(&[]),
        Err(RuntimeError::Bridge(
            stacker_nn::bridge::BridgeError::EmptyStack
        ))
    ));
}

#[test]
fn the_recommended_tile_follows_the_entrys_geometry() {
    let (_tmp, entries) = models();
    assert_eq!(recommended_tile_for(entry(&entries, "focus")).overlap, 32);
    assert_eq!(recommended_tile_for(entry(&entries, "batch")).overlap, 32);
    // Alignment models are not tiled: the generic (largest-preset) tiling.
    for name in ["align", "pairs"] {
        assert_eq!(
            recommended_tile_for(entry(&entries, name)).overlap,
            recommended_tile().overlap,
            "{name}"
        );
    }
}
