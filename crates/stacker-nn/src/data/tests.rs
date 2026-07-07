//! Unit tests for the dataset / rollout pipeline.
//!
//! These live as a child module of `data` (rather than under `tests/`) because
//! they reach crate-internal items (`Path`, `RgbImage`, private crop helpers)
//! via `use super::*`, which only resolves for an in-crate child module.

use super::*;
use burn::backend::NdArray;

type B = NdArray;

// -----------------------------------------------------------------------
// Helpers: synthesise a minimal scene on disk
// -----------------------------------------------------------------------

/// Write a plain RGB PNG with all pixels set to `rgb` (sRGB u8).
fn write_rgb_png(path: &Path, w: u32, h: u32, rgb: [u8; 3]) {
    let img = RgbImage::from_fn(w, h, |_, _| image::Rgb(rgb));
    img.save(path).expect("write rgb png");
}

/// Write a plain grayscale PNG with all pixels set to `v`.
fn write_gray_png(path: &Path, w: u32, h: u32, v: u8) {
    let img = GrayImage::from_fn(w, h, |_, _| image::Luma([v]));
    img.save(path).expect("write gray png");
}

/// Build a 3-plane 8×8 scene under `dir`.
///
/// Mask layout:
/// * frame 0: mask = 0   (no focus anywhere)
/// * frame 1: mask = 200 (in focus — value > threshold*255)
/// * frame 2: mask = 0   (no focus anywhere)
///
/// `merge_order` = [0, 1, 2]
///
/// allfocus = pure green (0, 255, 0)
/// All frames = pure red (255, 0, 0)
fn write_scene(dir: &Path) {
    let (w, h) = (8u32, 8u32);

    for i in 0..3u32 {
        write_rgb_png(&dir.join(format!("frame_{i:03}.png")), w, h, [255, 0, 0]);
        let mask_val: u8 = if i == 1 { 200 } else { 0 };
        write_gray_png(&dir.join(format!("mask_{i:03}.png")), w, h, mask_val);
    }

    // allfocus = green
    write_rgb_png(&dir.join("allfocus.png"), w, h, [0, 255, 0]);
    // occlusion = 0 (no edges)
    write_gray_png(&dir.join("occlusion.png"), w, h, 0);

    let meta = serde_json::json!({
        "n_planes": 3,
        "focus_fractions": [0.0, 0.5, 1.0],
        "stack":  ["frame_000.png", "frame_001.png", "frame_002.png"],
        "masks":  ["mask_000.png",  "mask_001.png",  "mask_002.png"],
        "allfocus":  "allfocus.png",
        "occlusion": "occlusion.png",
        "merge_order": [0, 1, 2]
    });
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
}

// -----------------------------------------------------------------------
// Test 1: rollout step count
// -----------------------------------------------------------------------

#[test]
fn test_rollout_step_count() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);

    let seq = RolloutSequence::new(scene).unwrap();
    // 3 planes → 2 steps
    assert_eq!(seq.n_steps, 2);
}

// -----------------------------------------------------------------------
// Test 2: tensor shapes are correct (full image, no crop)
// -----------------------------------------------------------------------

#[test]
fn test_tensor_shapes_full_image() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);

    let seq = RolloutSequence::new(scene).unwrap();
    let device = burn::prelude::Device::<B>::default();
    let sample = seq.get_step::<B>(0, &device, None).unwrap();

    assert_eq!(sample.target.dims(), [3, 8, 8]);
    assert_eq!(sample.target_conf.dims(), [1, 8, 8]);
    assert_eq!(sample.source.dims(), [3, 8, 8]);
    assert_eq!(sample.gt_merged.dims(), [3, 8, 8]);
    assert_eq!(sample.gt_conf.dims(), [1, 8, 8]);
    assert_eq!(sample.occlusion.dims(), [1, 8, 8]);
}

// -----------------------------------------------------------------------
// Test 3: prefix_composite uses allfocus where mask > threshold
// -----------------------------------------------------------------------

#[test]
fn test_prefix_composite_logic() {
    // 1×1 images for easy reasoning.
    let frame_red = vec![1.0f32, 0.0, 0.0]; // [3, 1, 1] – red
    let frame_blue = vec![0.0f32, 0.0, 1.0]; // [3, 1, 1] – blue
    let allfocus = vec![0.0f32, 1.0, 0.0]; // [3, 1, 1] – green

    // mask_a = 0.5 > threshold(0.1) → should use allfocus
    let mask_high = vec![0.5f32];
    let result = prefix_composite(
        std::slice::from_ref(&frame_red),
        &[mask_high],
        &allfocus,
        1,
        1,
        1,
        0.1,
    );
    assert!(
        (result[0] - 0.0).abs() < 1e-6
            && (result[1] - 1.0).abs() < 1e-6
            && (result[2] - 0.0).abs() < 1e-6,
        "expected allfocus (green) when mask > threshold, got {result:?}"
    );

    // mask_a = 0.05 < threshold(0.1) → should use least-defocused frame
    // With two frames, mask_a=0.05 < mask_b=0.08 → use frame_blue
    let mask_low_a = vec![0.05f32];
    let mask_low_b = vec![0.08f32];
    let result2 = prefix_composite(
        &[frame_red, frame_blue],
        &[mask_low_a, mask_low_b],
        &allfocus,
        2,
        1,
        1,
        0.1,
    );
    // Neither mask > 0.1, best is frame_blue (higher mask)
    assert!(
        (result2[0] - 0.0).abs() < 1e-6
            && (result2[1] - 0.0).abs() < 1e-6
            && (result2[2] - 1.0).abs() < 1e-6,
        "expected blue frame (argmax mask) when all masks < threshold, got {result2:?}"
    );
}

// -----------------------------------------------------------------------
// Test 4: random crop produces correct patch size
// -----------------------------------------------------------------------

#[test]
fn test_crop_patch_size() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);

    let seq = RolloutSequence::new(scene).unwrap();
    let device = burn::prelude::Device::<B>::default();
    let crop = CropParams {
        top: 0,
        left: 0,
        size: 4,
    };
    let sample = seq.get_step::<B>(0, &device, Some(crop)).unwrap();

    assert_eq!(sample.target.dims(), [3, 4, 4]);
    assert_eq!(sample.target_conf.dims(), [1, 4, 4]);
    assert_eq!(sample.source.dims(), [3, 4, 4]);
    assert_eq!(sample.gt_merged.dims(), [3, 4, 4]);
    assert_eq!(sample.gt_conf.dims(), [1, 4, 4]);
    assert_eq!(sample.occlusion.dims(), [1, 4, 4]);
}

// -----------------------------------------------------------------------
// Test 5: FocusStackDataset scans root and returns correct item count
// -----------------------------------------------------------------------

#[test]
fn test_dataset_item_count() {
    let tmp = tempfile::tempdir().unwrap();
    // Two scenes × 2 steps each = 4 total items
    for i in 0..2 {
        let scene = tmp.path().join(format!("scene{i}"));
        std::fs::create_dir(&scene).unwrap();
        write_scene(&scene);
    }

    let ds =
        FocusStackDataset::<B>::from_root(tmp.path(), burn::prelude::Device::<B>::default(), None)
            .unwrap();

    assert_eq!(ds.len(), 4);
    assert!(!ds.is_empty());
}

// -----------------------------------------------------------------------
// Test 6: dataset.get() returns correct shapes
// -----------------------------------------------------------------------

#[test]
fn test_dataset_get_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);

    let ds =
        FocusStackDataset::<B>::from_root(tmp.path(), burn::prelude::Device::<B>::default(), None)
            .unwrap();

    let s = ds.get(0).unwrap();
    assert_eq!(s.target.dims(), [3, 8, 8]);
    assert_eq!(s.target_conf.dims(), [1, 8, 8]);
    assert_eq!(s.gt_conf.dims(), [1, 8, 8]);
}

// -----------------------------------------------------------------------
// Test 7: get_batch loads masks and occlusion (re-indexed/cropped like frames)
// -----------------------------------------------------------------------

/// [`RolloutSequence::get_batch`] must load `masks` (`[S, 1, H, W]`, one per
/// stack frame, re-indexed by `merge_order` and cropped identically to
/// `stack`) and `occlusion` (`[1, H, W]`), matching [`write_scene`]'s known
/// mask layout (frame 1's mask = 200/255 ≈ 0.784, frames 0/2 = 0) and
/// occlusion (uniformly 0, i.e. no depth edges).
#[test]
fn get_batch_loads_masks_and_occlusion() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);

    let seq = RolloutSequence::new(scene).unwrap();
    let device = burn::prelude::Device::<B>::default();
    let sample = seq.get_batch::<B>(&device, None).unwrap();

    assert_eq!(sample.stack.dims(), [3, 3, 8, 8]);
    assert_eq!(sample.masks.dims(), [3, 1, 8, 8]);
    assert_eq!(sample.occlusion.dims(), [1, 8, 8]);

    let masks_data: Vec<f32> = sample.masks.into_data().iter::<f32>().collect();
    let np = 8 * 8;
    // merge_order = [0, 1, 2]; frame 1's mask is 200/255, others are 0.
    let expected_1 = 200.0_f32 / 255.0;
    for px in 0..np {
        assert!(
            (masks_data[px] - 0.0).abs() < 1e-5,
            "frame 0 mask should be 0 at px {px}"
        );
        assert!(
            (masks_data[np + px] - expected_1).abs() < 1e-5,
            "frame 1 mask should be ~{expected_1} at px {px}, got {}",
            masks_data[np + px]
        );
        assert!(
            (masks_data[2 * np + px] - 0.0).abs() < 1e-5,
            "frame 2 mask should be 0 at px {px}"
        );
    }

    let occ_data: Vec<f32> = sample.occlusion.into_data().iter::<f32>().collect();
    for (px, &v) in occ_data.iter().enumerate() {
        assert!(
            (v - 0.0).abs() < 1e-5,
            "occlusion should be 0 at px {px}, got {v}"
        );
    }
}

/// A cropped [`RolloutSequence::get_batch`] call must crop `masks` and
/// `occlusion` identically to `stack`/`gt_merged`.
#[test]
fn get_batch_crops_masks_and_occlusion() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);

    let seq = RolloutSequence::new(scene).unwrap();
    let device = burn::prelude::Device::<B>::default();
    let crop = CropParams {
        top: 0,
        left: 0,
        size: 4,
    };
    let sample = seq.get_batch::<B>(&device, Some(crop)).unwrap();

    assert_eq!(sample.stack.dims(), [3, 3, 4, 4]);
    assert_eq!(sample.masks.dims(), [3, 1, 4, 4]);
    assert_eq!(sample.occlusion.dims(), [1, 4, 4]);
}
