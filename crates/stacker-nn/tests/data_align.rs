//! Alignment training data: the matrix conventions, the crop-correct ground truth,
//! the datasets and the caches.
#![cfg(feature = "ndarray")]
#![allow(
    clippy::suboptimal_flops,
    reason = "plain fixture arithmetic; fusing would change the expected rounding"
)]

use std::path::Path;

use burn::{backend::NdArray, prelude::Backend};
use image::{Rgb, RgbImage};
use stacker_nn::data::{
    AlignImages, AlignSceneCache, AlignSequence, AlignStackDataset, CropParams, DataError,
    MIN_ALIGN_CROP, RealStackScene, crop_normalized_matrix, crop_pixel_matrix,
    discover_align_scenes, discover_real_stacks, normalized_matrix_to_pixel,
    pixel_matrix_to_normalized,
};

type B = NdArray;

const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn device() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

/// Apply a 3x3 matrix to a point (no perspective division needed for affine ones).
fn apply(m: &[[f32; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let f = |r: usize| f64::from(m[r][0]) * x + f64::from(m[r][1]) * y + f64::from(m[r][2]);
    (f(0), f(1))
}

fn assert_matrix_close(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3], tol: f32) {
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (a[r][c] - b[r][c]).abs() <= tol,
                "element [{r}][{c}]: {} vs {} (tolerance {tol})\n{a:?}\n{b:?}",
                a[r][c],
                b[r][c]
            );
        }
    }
}

const fn translation(tx: f32, ty: f32) -> [[f32; 3]; 3] {
    [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]
}

/// Rotation by `deg` degrees about `(cx, cy)`, with a translation on top.
fn rotation_about(deg: f64, cx: f64, cy: f64, tx: f64, ty: f64) -> [[f32; 3]; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "test matrix, f32 is plenty"
    )]
    let m = |v: f64| v as f32;
    [
        [m(c), m(-s), m(cx - c * cx + s * cy + tx)],
        [m(s), m(c), m(cy - s * cx - c * cy + ty)],
        [0.0, 0.0, 1.0],
    ]
}

// ---------------------------------------------------------------------------
// Matrix conventions
// ---------------------------------------------------------------------------

#[test]
fn normalisation_maps_the_image_corners_to_plus_minus_one() {
    // A translation by t pixels is 2t/(w-1) normalised units: the whole image
    // spans [-1, 1], i.e. w-1 pixel steps.
    let n = pixel_matrix_to_normalized(translation(4.0, -6.0), 101, 61);
    assert_matrix_close(&n, &translation(0.08, -0.2), 1e-6);
}

#[test]
fn pixel_and_normalised_conversions_are_inverse() {
    let m = rotation_about(3.0, 50.0, 40.0, 5.0, -2.0);
    let back = normalized_matrix_to_pixel(pixel_matrix_to_normalized(m, 120, 90), 120, 90);
    assert_matrix_close(&back, &m, 2e-4);
    assert_matrix_close(
        &pixel_matrix_to_normalized(IDENTITY, 33, 17),
        &IDENTITY,
        1e-7,
    );
}

#[test]
fn a_translation_is_the_same_in_every_pixel_grid() {
    // T(-o) T(t) T(o) = T(t): cropping does not change a pure shift in pixels.
    let m = translation(4.0, -6.0);
    assert_matrix_close(&crop_pixel_matrix(m, 17, 9), &m, 1e-6);
}

#[test]
fn known_shift_is_normalised_by_the_crop_not_the_full_frame() {
    // A 4 px / -6 px shift of a 200 x 160 frame, seen through a 32 px crop.
    let m = translation(4.0, -6.0);
    let crop = CropParams {
        top: 20,
        left: 50,
        size: 32,
    };
    let label = crop_normalized_matrix(m, Some(crop), 200, 160);
    // Normalised by the crop: 2 * 4 / 31 and 2 * -6 / 31.
    assert_matrix_close(&label, &translation(8.0 / 31.0, -12.0 / 31.0), 1e-6);

    // The full-frame normalisation (the old, wrong label) is 6x smaller.
    let wrong = pixel_matrix_to_normalized(m, 200, 160);
    assert!((wrong[0][2] - 8.0 / 199.0).abs() < 1e-6);
    assert!((label[0][2] / wrong[0][2]).abs() > 5.0);

    // No crop: the full-frame normalisation.
    assert_matrix_close(&crop_normalized_matrix(m, None, 200, 160), &wrong, 1e-7);
}

#[test]
fn crop_label_maps_every_crop_point_to_its_crop_image() {
    // A rotation about the FULL-frame centre is a rotation about a different point
    // of the crop. The label has to move any crop point to where the pixel actually
    // goes, in the crop's own normalised coordinates.
    let (fw, fh) = (100usize, 80usize);
    let m = rotation_about(2.0, 49.5, 39.5, 3.0, -2.0);
    let crop = CropParams {
        top: 20,
        left: 30,
        size: 32,
    };
    let label = crop_normalized_matrix(m, Some(crop), fw, fh);
    let norm = |p: f64| 2.0 * p / 31.0 - 1.0;
    let denorm = |u: f64| (u + 1.0) * 31.0 / 2.0;

    let mut checked = 0;
    for cy in (2..30).step_by(5) {
        for cx in (2..30).step_by(5) {
            let (cxf, cyf) = (f64::from(cx), f64::from(cy));
            // Where the full-frame matrix sends the pixel, expressed in the crop grid.
            let (qx, qy) = apply(&m, cxf + 30.0, cyf + 20.0);
            let (qx, qy) = (qx - 30.0, qy - 20.0);
            // Where the label sends the same pixel.
            let (u, v) = apply(&label, norm(cxf), norm(cyf));
            assert!(
                (denorm(u) - qx).abs() < 1e-3 && (denorm(v) - qy).abs() < 1e-3,
                "crop pixel ({cx}, {cy}) -> expected ({qx}, {qy}), label gives ({}, {})",
                denorm(u),
                denorm(v)
            );
            checked += 1;
        }
    }
    assert!(checked > 20);
}

// ---------------------------------------------------------------------------
// Scenes on disk
// ---------------------------------------------------------------------------

/// A texture with no repeats in a small window.
fn texture(x: i64, y: i64) -> [u8; 3] {
    let v = |a: i64| u8::try_from(a.rem_euclid(256)).unwrap();
    [v(x * 7 + y * 13), v(x * 3 + y * 5 + 40), v(x * y + 90)]
}

/// Write a scene whose frame `k` is the reference shifted by `shifts[k]` pixels
/// (`frame_k(x + dx, y + dy) = ref(x, y)`), with the matching `alignment_gt`.
fn write_align_scene(dir: &Path, w: u32, h: u32, shifts: &[(i64, i64)]) {
    let mut stack = Vec::new();
    let mut gt = Vec::new();
    for (k, &(dx, dy)) in shifts.iter().enumerate() {
        let img = RgbImage::from_fn(w, h, |x, y| {
            let (sx, sy) = (i64::from(x) - dx, i64::from(y) - dy);
            let inside = (0..i64::from(w)).contains(&sx) && (0..i64::from(h)).contains(&sy);
            Rgb(if inside { texture(sx, sy) } else { [0, 0, 0] })
        });
        let name = format!("frame_{k:02}.png");
        img.save(dir.join(&name)).unwrap();
        stack.push(name);
        #[expect(clippy::cast_precision_loss, reason = "tiny test shifts")]
        gt.push(translation(dx as f32, dy as f32));
    }
    let meta = serde_json::json!({
        "n_planes": shifts.len(),
        "stack": stack,
        "alignment_gt": gt,
        "cropped_dims": [w, h],
    });
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
}

fn scene(root: &Path, name: &str, w: u32, h: u32, shifts: &[(i64, i64)]) -> AlignSequence {
    let dir = root.join(name);
    std::fs::create_dir(&dir).unwrap();
    write_align_scene(&dir, w, h, shifts);
    AlignSequence::new(dir).unwrap()
}

fn values<const D: usize>(t: burn::prelude::Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn full_frame_sample_has_the_frame_normalisation() {
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 40, 30, &[(0, 0), (5, -3)]);

    let sample = seq.get::<B>(&device(), None).unwrap();
    assert_eq!(sample.stack.shape().dims, [2, 3, 30, 40]);
    assert_eq!(sample.gt_matrices.shape().dims, [2, 3, 3]);

    let gt = values(sample.gt_matrices);
    let expected = pixel_matrix_to_normalized(translation(5.0, -3.0), 40, 30);
    for (r, row) in expected.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            assert!((gt[9 + 3 * r + c] - v).abs() < 1e-6);
            assert!(
                (gt[3 * r + c] - IDENTITY[r][c]).abs() < 1e-6,
                "frame 0 is the identity"
            );
        }
    }
}

#[test]
fn cropped_sample_label_agrees_with_the_cropped_pixels() {
    // Frame 1 really is frame 0 shifted by (5, -3). Take a 24 px crop and check the
    // label end to end: move each reference crop pixel with the normalised matrix,
    // convert back to crop pixels, and compare the two crops' pixel values.
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 48, 40, &[(0, 0), (5, -3)]);
    let crop = CropParams {
        top: 8,
        left: 10,
        size: 24,
    };

    let sample = seq.get::<B>(&device(), Some(crop)).unwrap();
    assert_eq!(sample.stack.shape().dims, [2, 3, 24, 24]);
    let pixels = values(sample.stack);
    let gt = values(sample.gt_matrices);
    let label: [[f32; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|c| gt[9 + 3 * r + c]));

    // 2 * 5 / 23 and 2 * -3 / 23: the crop's normalisation.
    assert_matrix_close(&label, &translation(10.0 / 23.0, -6.0 / 23.0), 1e-6);

    let plane = 24 * 24;
    let at = |frame: usize, ch: usize, x: usize, y: usize| {
        pixels[frame * 3 * plane + ch * plane + y * 24 + x]
    };
    let px = |p: usize| f64::from(u32::try_from(p).unwrap());
    let mut compared = 0;
    for cy in 0..24usize {
        for cx in 0..24usize {
            let norm = |p: usize| 2.0 * px(p) / 23.0 - 1.0;
            let (u, v) = apply(&label, norm(cx), norm(cy));
            let (mx, my) = ((u + 1.0) * 23.0 / 2.0, (v + 1.0) * 23.0 / 2.0);
            assert!((mx - (px(cx) + 5.0)).abs() < 1e-3);
            assert!((my - (px(cy) - 3.0)).abs() < 1e-3);
            let (fx, fy) = (cx + 5, cy.wrapping_sub(3));
            if fx < 24 && fy < 24 {
                for ch in 0..3 {
                    assert_eq!(at(1, ch, fx, fy), at(0, ch, cx, cy));
                }
                compared += 1;
            }
        }
    }
    assert!(compared > 200, "compared only {compared} pixels");
}

#[test]
fn pair_sample_matches_the_stack_sample() {
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 48, 40, &[(0, 0), (5, -3), (-2, 4)]);
    let crop = Some(CropParams {
        top: 6,
        left: 12,
        size: 20,
    });
    let stack = seq.get::<B>(&device(), crop).unwrap();
    let pair = seq.get_pair::<B>(2, &device(), crop).unwrap();

    assert_eq!(pair.reference.shape().dims, [3, 20, 20]);
    assert_eq!(pair.frame.shape().dims, [3, 20, 20]);
    assert_eq!(pair.gt_matrix.shape().dims, [3, 3]);

    let gt = values(stack.gt_matrices);
    assert_eq!(values(pair.gt_matrix), gt[18..27].to_vec());
    let pixels = values(stack.stack);
    let plane = 3 * 20 * 20;
    assert_eq!(values(pair.reference), pixels[..plane].to_vec());
    assert_eq!(values(pair.frame), pixels[2 * plane..3 * plane].to_vec());

    // The reference against itself is a valid sample with an identity label.
    let same = seq.get_pair::<B>(0, &device(), crop).unwrap();
    assert_eq!(values(same.reference), values(same.frame));
    let id = values(same.gt_matrix);
    for (i, v) in id.iter().enumerate() {
        assert!((v - IDENTITY[i / 3][i % 3]).abs() < 1e-6);
    }

    assert!(matches!(
        seq.get_pair::<B>(3, &device(), crop),
        Err(DataError::InvalidMetadata { .. })
    ));
}

#[test]
fn cached_samples_equal_fresh_ones_and_decode_each_frame_once() {
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 40, 30, &[(0, 0), (3, 2), (-4, 1)]);
    let crop = Some(CropParams {
        top: 2,
        left: 5,
        size: 20,
    });

    let mut cache = AlignSceneCache::new(1 << 20);
    let images = cache.get(&seq).unwrap();
    assert_eq!(images.planes(), 3);
    assert_eq!(images.dims(), (30, 40));
    assert_eq!(images.byte_size(), 3 * 3 * 30 * 40);
    assert_eq!(cache.decoded_files(), 3);

    let fresh = seq.get::<B>(&device(), crop).unwrap();
    let cached = seq
        .sample_from_images::<B>(&images, &device(), crop)
        .unwrap();
    assert_eq!(values(fresh.stack), values(cached.stack));
    assert_eq!(values(fresh.gt_matrices), values(cached.gt_matrices));

    for frame in 0..3 {
        let again = cache.get(&seq).unwrap();
        let pair = seq
            .pair_from_images::<B>(&again, frame, &device(), crop)
            .unwrap();
        assert_eq!(pair.frame.shape().dims, [3, 20, 20]);
    }
    assert_eq!(cache.decoded_files(), 3, "hits decode nothing");
    assert_eq!(cache.cached_bytes(), images.byte_size());
}

#[test]
fn cache_never_retains_more_than_its_budget() {
    let tmp = tempfile::tempdir().unwrap();
    let a = scene(tmp.path(), "a", 16, 16, &[(0, 0), (1, 1)]);
    let b = scene(tmp.path(), "b", 16, 16, &[(0, 0), (2, 2)]);
    let scene_bytes = 3 * 2 * 16 * 16;

    let mut cache = AlignSceneCache::new(scene_bytes + scene_bytes / 2);
    cache.get(&a).unwrap();
    cache.get(&b).unwrap();
    assert_eq!(cache.len(), 1, "room for one scene, the older is evicted");
    assert!(cache.cached_bytes() <= scene_bytes + scene_bytes / 2);
    let before = cache.decoded_files();
    cache.get(&a).unwrap();
    assert_eq!(cache.decoded_files(), before + 2);

    let mut none = AlignSceneCache::new(0);
    none.get(&a).unwrap();
    none.get(&a).unwrap();
    assert!(none.is_empty());
    assert_eq!(none.cached_bytes(), 0);
    assert_eq!(none.decoded_files(), 4);
}

#[test]
fn crop_errors_and_metadata_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let seq = scene(tmp.path(), "s", 40, 30, &[(0, 0), (1, 1)]);

    let too_big = CropParams {
        top: 20,
        left: 0,
        size: 20,
    };
    assert!(matches!(
        seq.get::<B>(&device(), Some(too_big)),
        Err(DataError::CropTooLarge { .. })
    ));
    let too_small = CropParams {
        top: 0,
        left: 0,
        size: MIN_ALIGN_CROP - 1,
    };
    assert!(matches!(
        seq.get::<B>(&device(), Some(too_small)),
        Err(DataError::CropTooSmall { .. })
    ));

    // A frame that is not cropped_dims in size would make the label wrong.
    let bad = tmp.path().join("bad");
    std::fs::create_dir(&bad).unwrap();
    write_align_scene(&bad, 40, 30, &[(0, 0), (1, 1)]);
    RgbImage::new(39, 30)
        .save(bad.join("frame_01.png"))
        .unwrap();
    let bad_seq = AlignSequence::new(bad).unwrap();
    assert!(matches!(
        AlignImages::load(&bad_seq),
        Err(DataError::ShapeMismatch { .. })
    ));

    // Lengths that disagree with n_planes are rejected at load time.
    let broken = tmp.path().join("broken");
    std::fs::create_dir(&broken).unwrap();
    std::fs::write(
        broken.join("metadata.json"),
        r#"{"n_planes": 2, "stack": ["a.png"], "alignment_gt": [], "cropped_dims": [8, 8]}"#,
    )
    .unwrap();
    assert!(matches!(
        AlignSequence::new(broken),
        Err(DataError::InvalidMetadata { .. })
    ));
}

#[test]
fn dataset_discovers_scenes_in_path_order() {
    let tmp = tempfile::tempdir().unwrap();
    scene(tmp.path(), "b_scene", 32, 32, &[(0, 0), (1, 0)]);
    scene(tmp.path(), "a_scene", 32, 32, &[(0, 0), (0, 1), (1, 1)]);
    std::fs::create_dir(tmp.path().join("not_a_scene")).unwrap();

    let found = discover_align_scenes(tmp.path()).unwrap();
    let names: Vec<_> = found
        .iter()
        .map(|s| {
            s.scene_dir
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(names, ["a_scene", "b_scene"]);

    let crop = Some(CropParams {
        top: 0,
        left: 0,
        size: 16,
    });
    let dataset = AlignStackDataset::<B>::from_root(tmp.path(), device(), crop).unwrap();
    assert_eq!(dataset.len(), 2);
    assert!(!dataset.is_empty());
    assert_eq!(dataset.get(0).unwrap().stack.shape().dims, [3, 3, 16, 16]);
    assert_eq!(dataset.get(1).unwrap().gt_matrices.shape().dims, [2, 3, 3]);
}

// ---------------------------------------------------------------------------
// Real stacks
// ---------------------------------------------------------------------------

#[test]
fn real_stacks_are_ordered_naturally_and_cropped_while_decoding() {
    let tmp = tempfile::tempdir().unwrap();
    let stack = tmp.path().join("stack");
    std::fs::create_dir(&stack).unwrap();
    // img2 < img10 naturally, but not as plain strings.
    for (name, level) in [("img10.png", 30u8), ("img2.png", 20), ("IMG1.PNG", 10)] {
        RgbImage::from_fn(24, 20, |_, _| Rgb([level, level, level]))
            .save(stack.join(name))
            .unwrap();
    }
    std::fs::write(stack.join("notes.txt"), "not an image").unwrap();
    std::fs::create_dir(tmp.path().join("empty")).unwrap();

    let scenes = discover_real_stacks(tmp.path()).unwrap();
    assert_eq!(scenes.len(), 1, "a directory without images is not a scene");
    let real: &RealStackScene = &scenes[0];
    assert_eq!(real.len(), 3);
    assert!(!real.is_empty());
    let order: Vec<_> = real
        .frames
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(order, ["IMG1.PNG", "img2.png", "img10.png"]);
    assert_eq!(real.frame_dims().unwrap(), (20, 24));

    let full = real.get::<B>(&device(), None).unwrap();
    assert_eq!(full.shape().dims, [3, 3, 20, 24]);
    let crop = CropParams {
        top: 2,
        left: 3,
        size: 8,
    };
    let cut = real.get::<B>(&device(), Some(crop)).unwrap();
    assert_eq!(cut.shape().dims, [3, 3, 8, 8]);
    // Frame 0 is the darkest (level 10), frame 2 the brightest: the order is real.
    let v = values(cut);
    assert!(v[0] < v[3 * 64] && v[3 * 64] < v[2 * 3 * 64]);

    assert!(matches!(
        real.get::<B>(
            &device(),
            Some(CropParams {
                top: 15,
                left: 0,
                size: 8
            })
        ),
        Err(DataError::CropTooLarge { .. })
    ));
}
