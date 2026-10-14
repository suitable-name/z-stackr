//! Unit tests for the dataset / rollout pipeline.
//!
//! These live as a child module of `data` (rather than under `tests/`) because
//! they reach crate-internal items (`Path`, `RgbImage`, private crop helpers)
//! via `use super::*`, which only resolves for an in-crate child module.

use super::*;

// The tensor-producing tests need a CPU backend; `ndarray` is always the one used
// here (never `SelectedBackend`, which is wgpu when that feature is on). Without
// the `ndarray` feature this module compiles to nothing (see `data/mod.rs`).
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
    let device = <B as Backend>::Device::default();
    let sample = seq.get_step::<B>(0, &device, None).unwrap();

    assert_eq!(sample.target.shape().dims, [3, 8, 8]);
    assert_eq!(sample.target_conf.shape().dims, [1, 8, 8]);
    assert_eq!(sample.source.shape().dims, [3, 8, 8]);
    assert_eq!(sample.gt_merged.shape().dims, [3, 8, 8]);
    assert_eq!(sample.gt_conf.shape().dims, [1, 8, 8]);
    assert_eq!(sample.occlusion.shape().dims, [1, 8, 8]);
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
    let device = <B as Backend>::Device::default();
    let crop = CropParams {
        top: 0,
        left: 0,
        size: 4,
    };
    let sample = seq.get_step::<B>(0, &device, Some(crop)).unwrap();

    assert_eq!(sample.target.shape().dims, [3, 4, 4]);
    assert_eq!(sample.target_conf.shape().dims, [1, 4, 4]);
    assert_eq!(sample.source.shape().dims, [3, 4, 4]);
    assert_eq!(sample.gt_merged.shape().dims, [3, 4, 4]);
    assert_eq!(sample.gt_conf.shape().dims, [1, 4, 4]);
    assert_eq!(sample.occlusion.shape().dims, [1, 4, 4]);
}

// -----------------------------------------------------------------------
// NN-6: step 0 starts from the raw first frame, confidence 0
// -----------------------------------------------------------------------

/// Like [`write_scene`], but with explicit per-frame mask values (u8).
fn write_scene_masks(dir: &Path, masks: [u8; 3]) {
    write_scene(dir);
    for (i, v) in masks.iter().enumerate() {
        write_gray_png(&dir.join(format!("mask_{i:03}.png")), 8, 8, *v);
    }
}

fn values<B2: Backend, const D: usize>(t: Tensor<B2, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn step_zero_is_the_raw_first_frame_with_zero_confidence() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    // Frame 0 is "in focus" (mask 200/255 = 0.78 > threshold) everywhere: handing
    // the all-focus (green) composite and a confidence of 0.78 to the network at
    // step 0 would differ from inference, which starts from the raw red frame and
    // confidence 0.
    write_scene_masks(&scene, [200, 200, 0]);

    let seq = RolloutSequence::new(scene).unwrap();
    let device = <B as Backend>::Device::default();
    let np = 64;

    let s0 = seq.get_step::<B>(0, &device, None).unwrap();
    let target = values(s0.target);
    assert!(
        target[..np].iter().all(|v| (v - 1.0).abs() < 1e-6),
        "step 0 target must be the raw (red) first frame"
    );
    assert!(
        target[np..].iter().all(|v| v.abs() < 1e-6),
        "step 0 target must have no green/blue"
    );
    assert!(
        values(s0.target_conf).iter().all(|v| *v == 0.0),
        "step 0 target_conf must be 0, as at inference"
    );
    // The ground truth after step 0 is still the oracle prefix composite (green).
    let gt = values(s0.gt_merged);
    assert!(gt[np..2 * np].iter().all(|v| (v - 1.0).abs() < 1e-6));

    // Later steps are teacher-forced from the oracle composite and the mask.
    let s1 = seq.get_step::<B>(1, &device, None).unwrap();
    let t1 = values(s1.target);
    assert!(
        t1[..np].iter().all(|v| v.abs() < 1e-6),
        "step 1 target is green"
    );
    assert!(t1[np..2 * np].iter().all(|v| (v - 1.0).abs() < 1e-6));
    let c1 = values(s1.target_conf);
    assert!(c1.iter().all(|v| (v - 200.0 / 255.0).abs() < 1e-6));
}

#[test]
fn allfocus_threshold_is_the_one_sigma_contour() {
    assert!((ALLFOCUS_MASK_THRESHOLD - (-0.5_f32).exp()).abs() < 1e-6);
}

// -----------------------------------------------------------------------
// NN-8: decode each plane once per scene use; bounded cache
// -----------------------------------------------------------------------

#[test]
fn scene_images_decode_each_file_exactly_once() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);
    let seq = RolloutSequence::new(scene).unwrap();

    // 3 planes: 3 frames + 3 masks + all-focus + occlusion.
    let all = SceneImages::load(&seq, 3).unwrap();
    assert_eq!(all.planes(), 3);
    assert_eq!(all.decoded_files(), 2 * 3 + 2);
    assert_eq!(all.dims(), (8, 8));
    assert_eq!(all.byte_size(), (4 * 3 + 4) * 8 * 8);

    // A single step only needs k + 2 planes.
    let two = SceneImages::load(&seq, 2).unwrap();
    assert_eq!(two.decoded_files(), 2 * 2 + 2);
}

#[test]
fn cache_hit_decodes_nothing_and_all_steps_come_from_one_decode() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene(&scene);
    let seq = RolloutSequence::new(scene).unwrap();
    let device = <B as Backend>::Device::default();

    let mut cache = SceneCache::new(1 << 20);
    let images = cache.get(&seq).unwrap();
    let n_planes = seq.meta.n_planes;
    // At most 2·n_planes + 2 decodes for the whole scene, however many steps.
    assert_eq!(cache.decoded_files(), 2 * n_planes + 2);
    assert!(cache.decoded_files() <= 2 * n_planes + 2);

    for top in [0, 2, 4] {
        let crop = CropParams {
            top,
            left: 1,
            size: 4,
        };
        let steps = seq.crop_steps::<B>(&images, &device, Some(crop)).unwrap();
        assert_eq!(steps.len(), seq.n_steps);
        assert_eq!(steps[0].target.shape().dims, [3, 4, 4]);
        // A second optimiser step on the same scene is a cache hit.
        let again = cache.get(&seq).unwrap();
        assert!(Arc::ptr_eq(&images, &again));
    }
    assert_eq!(
        cache.decoded_files(),
        2 * n_planes + 2,
        "cache hits and crops must not decode anything"
    );
    assert_eq!(cache.cached_bytes(), images.byte_size());
}

#[test]
fn cache_never_retains_more_than_its_budget() {
    let tmp = tempfile::tempdir().unwrap();
    let mut seqs = Vec::new();
    for i in 0..2 {
        let scene = tmp.path().join(format!("scene{i}"));
        std::fs::create_dir(&scene).unwrap();
        write_scene(&scene);
        seqs.push(RolloutSequence::new(scene).unwrap());
    }
    let scene_bytes = (4 * 3 + 4) * 8 * 8; // 1024

    // Room for one scene, not two: LRU eviction keeps only the newest.
    let mut cache = SceneCache::new(scene_bytes + scene_bytes / 2);
    cache.get(&seqs[0]).unwrap();
    cache.get(&seqs[1]).unwrap();
    assert_eq!(cache.len(), 1);
    assert!(cache.cached_bytes() <= scene_bytes + scene_bytes / 2);
    // Scene 0 was evicted, so asking again decodes again.
    let before = cache.decoded_files();
    cache.get(&seqs[0]).unwrap();
    assert_eq!(cache.decoded_files(), before + 8);

    // Budget 0: nothing is retained, every use decodes once.
    let mut none = SceneCache::new(0);
    none.get(&seqs[0]).unwrap();
    none.get(&seqs[0]).unwrap();
    assert!(none.is_empty());
    assert_eq!(none.cached_bytes(), 0);
    assert_eq!(none.decoded_files(), 16);
}

#[test]
fn crop_steps_matches_get_step_and_rejects_bad_crops() {
    let tmp = tempfile::tempdir().unwrap();
    let scene = tmp.path().join("scene0");
    std::fs::create_dir(&scene).unwrap();
    write_scene_masks(&scene, [0, 200, 90]);
    let seq = RolloutSequence::new(scene).unwrap();
    let device = <B as Backend>::Device::default();
    let crop = CropParams {
        top: 2,
        left: 3,
        size: 4,
    };

    let images = SceneImages::load(&seq, seq.meta.n_planes).unwrap();
    let steps = seq.crop_steps::<B>(&images, &device, Some(crop)).unwrap();
    for (k, step) in steps.into_iter().enumerate() {
        let single = seq.get_step::<B>(k, &device, Some(crop)).unwrap();
        assert_eq!(values(step.target), values(single.target));
        assert_eq!(values(step.target_conf), values(single.target_conf));
        assert_eq!(values(step.source), values(single.source));
        assert_eq!(values(step.gt_merged), values(single.gt_merged));
        assert_eq!(values(step.gt_conf), values(single.gt_conf));
    }

    // A window that sticks out of the image is an error, not a panic.
    let bad = CropParams {
        top: 6,
        left: 0,
        size: 4,
    };
    assert!(matches!(
        seq.crop_steps::<B>(&images, &device, Some(bad)),
        Err(DataError::CropTooLarge { .. })
    ));
}
