//! Batch fusion data: one sample per scene holding the whole stack.
#![cfg(feature = "ndarray")]

use std::path::Path;

use burn::{backend::NdArray, prelude::Backend};
use image::{GrayImage, Luma, Rgb, RgbImage};
use stacker_nn::data::{
    ALLFOCUS_MASK_THRESHOLD, BatchStackDataset, CropParams, DataError, RolloutSequence, SceneCache,
    discover_rollout_scenes,
};

type B = NdArray;

fn device() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

fn values<const D: usize>(t: burn::prelude::Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// A 3-plane scene of `size` x `size` pixels. `merge_order` is `[2, 0, 1]`, so the
/// batch must come out reordered. Frame `i` is a flat colour with red = `10 + 40 i`;
/// mask `i` is the flat value `mask_levels[i]`; the all-focus image is green.
fn write_scene(dir: &Path, size: u32, mask_levels: [u8; 3]) {
    for i in 0..3u8 {
        RgbImage::from_fn(size, size, |_, _| Rgb([10 + 40 * i, 0, 0]))
            .save(dir.join(format!("frame_{i}.png")))
            .unwrap();
        GrayImage::from_fn(size, size, |_, _| Luma([mask_levels[usize::from(i)]]))
            .save(dir.join(format!("mask_{i}.png")))
            .unwrap();
    }
    RgbImage::from_fn(size, size, |_, _| Rgb([0, 255, 0]))
        .save(dir.join("allfocus.png"))
        .unwrap();
    GrayImage::from_fn(size, size, |x, _| Luma([if x == 0 { 255 } else { 0 }]))
        .save(dir.join("occlusion.png"))
        .unwrap();
    let meta = serde_json::json!({
        "n_planes": 3,
        "focus_fractions": [0.0, 0.5, 1.0],
        "stack": ["frame_0.png", "frame_1.png", "frame_2.png"],
        "masks": ["mask_0.png", "mask_1.png", "mask_2.png"],
        "allfocus": "allfocus.png",
        "occlusion": "occlusion.png",
        "merge_order": [2, 0, 1]
    });
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
}

fn scene(root: &Path, name: &str, size: u32, mask_levels: [u8; 3]) -> RolloutSequence {
    let dir = root.join(name);
    std::fs::create_dir(&dir).unwrap();
    write_scene(&dir, size, mask_levels);
    RolloutSequence::new(dir).unwrap()
}

/// sRGB code to linear light, to compare with what the loader produces.
fn linear(code: u8) -> f32 {
    stacker_nn::data::to_linear(f32::from(code) / 255.0)
}

#[test]
fn batch_holds_the_whole_stack_in_merge_order() {
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 8, [0, 0, 0]);

    let sample = seq.get_batch::<B>(&device(), None).unwrap();
    assert_eq!(sample.stack.shape().dims, [3, 3, 8, 8]);
    assert_eq!(sample.masks.shape().dims, [3, 1, 8, 8]);
    assert_eq!(sample.gt_merged.shape().dims, [3, 8, 8]);
    assert_eq!(sample.occlusion.shape().dims, [1, 8, 8]);

    // merge_order = [2, 0, 1]: slot 0 is frame 2 (red 90), then frame 0 (10), frame 1 (50).
    let stack = values(sample.stack);
    let plane = 8 * 8;
    for (slot, red) in [90u8, 10, 50].into_iter().enumerate() {
        let got = stack[slot * 3 * plane];
        assert!((got - linear(red)).abs() < 1e-6, "slot {slot}: {got}");
    }
}

#[test]
fn masks_follow_the_merge_order() {
    let tmp = tempfile::tempdir().unwrap();
    // mask levels by plane index: 0 -> 30, 1 -> 60, 2 -> 90.
    let seq = scene(tmp.path(), "s", 8, [30, 60, 90]);
    let masks = values(seq.get_batch::<B>(&device(), None).unwrap().masks);
    let plane = 8 * 8;
    for (slot, level) in [90u8, 30, 60].into_iter().enumerate() {
        assert!((masks[slot * plane] - f32::from(level) / 255.0).abs() < 1e-6);
    }
}

#[test]
fn target_is_the_full_stack_prefix_composite() {
    let tmp = tempfile::tempdir().unwrap();
    // No plane in focus anywhere: the target falls back to the plane with the
    // highest mask (plane 2, red 90), not to the all-focus green.
    let dim = scene(tmp.path(), "dim", 8, [10, 20, 30]);
    let gt = values(dim.get_batch::<B>(&device(), None).unwrap().gt_merged);
    assert!((gt[0] - linear(90)).abs() < 1e-6);
    assert!(gt[2 * 64].abs() < 1e-6, "no all-focus blue");

    // A plane above the threshold: the all-focus render (green) is the target.
    assert!(f32::from(200u8) / 255.0 > ALLFOCUS_MASK_THRESHOLD);
    let focused = scene(tmp.path(), "focused", 8, [0, 200, 0]);
    let gt = values(focused.get_batch::<B>(&device(), None).unwrap().gt_merged);
    assert!(gt[0].abs() < 1e-6);
    assert!((gt[64] - 1.0).abs() < 1e-6);
}

#[test]
fn crop_cuts_every_tensor_at_the_same_window() {
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 8, [0, 0, 0]);
    let crop = CropParams {
        top: 1,
        left: 0,
        size: 4,
    };
    let sample = seq.get_batch::<B>(&device(), Some(crop)).unwrap();
    assert_eq!(sample.stack.shape().dims, [3, 3, 4, 4]);
    assert_eq!(sample.masks.shape().dims, [3, 1, 4, 4]);
    assert_eq!(sample.gt_merged.shape().dims, [3, 4, 4]);
    // The occlusion edge is column 0: with left = 0 the crop's first column holds it.
    let occ = values(sample.occlusion);
    assert_eq!(occ.len(), 16);
    assert!((occ[0] - 1.0).abs() < 1e-6 && occ[1].abs() < 1e-6);

    let bad = CropParams {
        top: 6,
        left: 0,
        size: 4,
    };
    assert!(matches!(
        seq.get_batch::<B>(&device(), Some(bad)),
        Err(DataError::CropTooLarge { .. })
    ));
}

#[test]
fn cached_batches_equal_fresh_ones_and_decode_the_scene_once() {
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 8, [30, 60, 90]);
    let crop = Some(CropParams {
        top: 2,
        left: 2,
        size: 4,
    });

    let mut cache = SceneCache::new(1 << 20);
    let images = cache.get(&seq).unwrap();
    assert_eq!(cache.decoded_files(), 2 * 3 + 2);

    let fresh = seq.get_batch::<B>(&device(), crop).unwrap();
    let cached = seq
        .batch_from_images::<B>(&images, &device(), crop)
        .unwrap();
    assert_eq!(values(fresh.stack), values(cached.stack));
    assert_eq!(values(fresh.masks), values(cached.masks));
    assert_eq!(values(fresh.gt_merged), values(cached.gt_merged));
    assert_eq!(values(fresh.occlusion), values(cached.occlusion));

    for _ in 0..3 {
        cache.get(&seq).unwrap();
    }
    assert_eq!(cache.decoded_files(), 2 * 3 + 2, "hits decode nothing");
}

#[test]
fn dataset_yields_one_sample_per_scene() {
    let tmp = tempfile::tempdir().unwrap();
    scene(tmp.path(), "b", 8, [0, 0, 0]);
    scene(tmp.path(), "a", 8, [0, 0, 0]);
    std::fs::create_dir(tmp.path().join("other")).unwrap();

    let scenes = discover_rollout_scenes(tmp.path()).unwrap();
    assert_eq!(scenes.len(), 2);
    assert!(scenes[0].scene_dir.ends_with("a"));

    let dataset = BatchStackDataset::<B>::from_root(
        tmp.path(),
        device(),
        Some(CropParams {
            top: 0,
            left: 0,
            size: 4,
        }),
    )
    .unwrap();
    assert_eq!(dataset.len(), 2);
    assert!(!dataset.is_empty());
    assert_eq!(dataset.get(1).unwrap().stack.shape().dims, [3, 3, 4, 4]);

    let empty = tempfile::tempdir().unwrap();
    let none = BatchStackDataset::<B>::from_root(empty.path(), device(), None).unwrap();
    assert!(none.is_empty());
}
