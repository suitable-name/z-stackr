//! Integration tests for the rollout training primitives in
//! [`stacker_nn::train`].

use burn::{
    backend::{Autodiff, NdArray},
    prelude::{Backend, Tensor},
    tensor::Distribution,
};
use image::RgbImage;
use stacker_nn::{
    data::{AlignSequence, MergeSample},
    loss::{BatchAlignmentLossConfig, FocusFusionLossConfig, PairAlignmentLossConfig},
    model::{BatchAlignNetConfig, FocusMergeNetConfig, FusionAlignNetConfig},
    train::{align_loss, cosine_lr, fusion_align_loss, rollout_loss, scheduled_sampling_prob},
};

/// Build a random `MergeSample<Bk>` of spatial size `h x w`.
fn rand_sample<Bk: Backend>(
    h: usize,
    w: usize,
    device: &burn::prelude::Device<Bk>,
) -> MergeSample<Bk> {
    let rgb =
        |c: usize| Tensor::<Bk, 3>::random([c, h, w], Distribution::Uniform(0.0, 1.0), device);
    MergeSample {
        target: rgb(3),
        target_conf: rgb(1),
        source: rgb(3),
        gt_merged: rgb(3),
        gt_conf: rgb(1),
        occlusion: rgb(1),
    }
}

#[test]
fn rollout_loss_forward_finite() {
    type B = NdArray;
    let device = burn::prelude::Device::<B>::default();

    let model = FocusMergeNetConfig::new().init::<B>(&device);
    let loss = FocusFusionLossConfig::new().init();

    let steps = vec![
        rand_sample::<B>(16, 16, &device),
        rand_sample::<B>(16, 16, &device),
        rand_sample::<B>(16, 16, &device),
    ];
    let use_prediction = [false, false, false];

    let total = rollout_loss(&model, &loss, &steps, &use_prediction);
    assert!(total.into_scalar().is_finite());
}

#[test]
fn rollout_loss_scheduled_sampling_backward() {
    // With scheduled sampling enabled (feed back predictions), the rollout must
    // still be differentiable end-to-end.
    type Ab = Autodiff<NdArray>;
    let device = burn::prelude::Device::<Ab>::default();

    let model = FocusMergeNetConfig::new().init::<Ab>(&device);
    let loss = FocusFusionLossConfig::new().init();

    let steps = vec![
        rand_sample::<Ab>(12, 12, &device),
        rand_sample::<Ab>(12, 12, &device),
    ];
    // Step 1 feeds back the model's own prediction.
    let use_prediction = [false, true];

    let total = rollout_loss(&model, &loss, &steps, &use_prediction);
    let scalar: f32 = total.clone().into_scalar();
    assert!(scalar.is_finite(), "rollout total {scalar} not finite");

    let _grads = total.backward();
}

#[test]
fn cosine_lr_warmup_and_decay() {
    let base = 1.0_f64;
    let total = 100;
    let warmup = 10;

    // Warm-up: strictly increasing toward base, below base.
    let lr_early = cosine_lr(base, 0, total, warmup);
    let lr_mid_warmup = cosine_lr(base, 5, total, warmup);
    assert!(lr_early < lr_mid_warmup);
    assert!(lr_mid_warmup <= base);

    // Mid training is below base; end approaches 0.
    let lr_mid = cosine_lr(base, 55, total, warmup);
    let lr_end = cosine_lr(base, 100, total, warmup);
    assert!(lr_mid < base);
    assert!(lr_end < lr_mid);
    assert!(lr_end < 1e-9, "lr at end should approach 0, got {lr_end}");
}

#[test]
fn scheduled_sampling_ramps_from_zero_to_max() {
    let total = 10;
    let max = 0.8_f64;
    assert!((scheduled_sampling_prob(0, total, max) - 0.0).abs() < 1e-12);
    let last = scheduled_sampling_prob(total - 1, total, max);
    assert!(
        (last - max).abs() < 1e-12,
        "last epoch should reach max, got {last}"
    );
    // Monotonic non-decreasing.
    let a = scheduled_sampling_prob(3, total, max);
    let b = scheduled_sampling_prob(7, total, max);
    assert!(a <= b);
    // Degenerate single-epoch run → 0.
    assert!((scheduled_sampling_prob(0, 1, max) - 0.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// align_loss + alignment dataset loader round trip
// ---------------------------------------------------------------------------

/// Write a plain RGB PNG with all pixels set to `rgb` (sRGB u8).
fn write_rgb_png(path: &std::path::Path, w: u32, h: u32, rgb: [u8; 3]) {
    let img = RgbImage::from_fn(w, h, |_, _| image::Rgb(rgb));
    img.save(path).expect("write rgb png");
}

/// Build a tiny 2-frame alignment scene on disk: two 8x8 PNGs plus a
/// `metadata.json` matching [`stacker_nn::data::AlignSceneMeta`]'s schema
/// (`n_planes`, `stack`, `alignment_gt`, `cropped_dims`). Ground-truth
/// matrices are both identity for simplicity.
fn write_align_scene(dir: &std::path::Path) {
    let (w, h) = (8u32, 8u32);
    write_rgb_png(&dir.join("frame_000.png"), w, h, [200, 100, 50]);
    write_rgb_png(&dir.join("frame_001.png"), w, h, [50, 100, 200]);

    #[rustfmt::skip]
    let identity = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let meta = serde_json::json!({
        "n_planes": 2,
        "stack": ["frame_000.png", "frame_001.png"],
        "alignment_gt": [identity, identity],
        "cropped_dims": [w, h],
    });
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
}

/// Full round trip: disk -> `AlignSceneMeta`/`AlignSequence` -> `AlignSample`
/// -> [`align_loss`]. Verifies the alignment dataset loader and `align_loss`
/// compose correctly end-to-end with a fresh (untrained) `BatchAlignNet`.
#[test]
fn align_loss_round_trips_from_disk_scene() {
    type B = NdArray;
    let device = burn::prelude::Device::<B>::default();

    let tmp = tempfile::tempdir().unwrap();
    let scene_dir = tmp.path().join("scene0");
    std::fs::create_dir(&scene_dir).unwrap();
    write_align_scene(&scene_dir);

    let seq = AlignSequence::new(scene_dir).unwrap();
    let sample = seq.get::<B>(&device, None).unwrap();

    assert_eq!(sample.stack.dims(), [2, 3, 8, 8]);
    assert_eq!(sample.gt_matrices.dims(), [2, 3, 3]);

    let model = BatchAlignNetConfig::new().init::<B>(&device);
    let loss = BatchAlignmentLossConfig::new().init();

    let total: f32 = align_loss(&model, &loss, &sample).into_scalar();
    assert!(total.is_finite(), "align_loss result {total} is not finite");
    assert!(total >= 0.0, "align_loss result {total} is negative");
}

// ---------------------------------------------------------------------------
// fusion_align_loss + pairwise-alignment dataset loader round trip
// ---------------------------------------------------------------------------

/// Full round trip: disk -> `AlignSceneMeta`/`AlignSequence::get_pair` ->
/// `PairAlignSample` -> [`fusion_align_loss`]. Reuses the SAME on-disk scene
/// format/fixture as `align_loss_round_trips_from_disk_scene` (see
/// `docs/fusionalign-design.md` for why `FusionAlignNet` trains from
/// identical scene data to `BatchAlignNet`, just sliced into pairs) —
/// verifies the pairwise dataset loader and `fusion_align_loss` compose
/// correctly end-to-end with a fresh (untrained) `FusionAlignNet`.
#[test]
fn fusion_align_loss_round_trips_from_disk_scene() {
    type B = NdArray;
    let device = burn::prelude::Device::<B>::default();

    let tmp = tempfile::tempdir().unwrap();
    let scene_dir = tmp.path().join("scene0");
    std::fs::create_dir(&scene_dir).unwrap();
    write_align_scene(&scene_dir);

    let seq = AlignSequence::new(scene_dir).unwrap();
    // frame_index = 1: register stack[1] against the reference stack[0].
    let sample = seq.get_pair::<B>(1, &device, None).unwrap();

    assert_eq!(sample.reference.dims(), [3, 8, 8]);
    assert_eq!(sample.frame.dims(), [3, 8, 8]);
    assert_eq!(sample.gt_matrix.dims(), [3, 3]);

    let model = FusionAlignNetConfig::new().init::<B>(&device);
    let loss = PairAlignmentLossConfig::new().init();

    let total: f32 = fusion_align_loss(&model, &loss, &sample).into_scalar();
    assert!(
        total.is_finite(),
        "fusion_align_loss result {total} is not finite"
    );
    assert!(total >= 0.0, "fusion_align_loss result {total} is negative");
}

/// `get_pair`'s `frame_index = 0` self-pair (reference registered against
/// itself) must also round-trip cleanly through `fusion_align_loss` — the
/// legitimate "identity" training example documented on
/// [`stacker_nn::data::AlignSequence::get_pair`].
#[test]
fn fusion_align_loss_round_trips_for_self_pair() {
    type B = NdArray;
    let device = burn::prelude::Device::<B>::default();

    let tmp = tempfile::tempdir().unwrap();
    let scene_dir = tmp.path().join("scene0");
    std::fs::create_dir(&scene_dir).unwrap();
    write_align_scene(&scene_dir);

    let seq = AlignSequence::new(scene_dir).unwrap();
    let sample = seq.get_pair::<B>(0, &device, None).unwrap();

    assert_eq!(sample.reference.dims(), [3, 8, 8]);
    assert_eq!(sample.frame.dims(), [3, 8, 8]);

    let model = FusionAlignNetConfig::new().init::<B>(&device);
    let loss = PairAlignmentLossConfig::new().init();

    let total: f32 = fusion_align_loss(&model, &loss, &sample).into_scalar();
    assert!(
        total.is_finite(),
        "fusion_align_loss result {total} is not finite"
    );
}

/// `get_pair` with an out-of-range `frame_index` must error rather than
/// panic (index into `meta.stack`/`meta.alignment_gt` is guarded up front).
#[test]
fn get_pair_out_of_range_frame_index_errors() {
    type B = NdArray;
    let device = burn::prelude::Device::<B>::default();

    let tmp = tempfile::tempdir().unwrap();
    let scene_dir = tmp.path().join("scene0");
    std::fs::create_dir(&scene_dir).unwrap();
    write_align_scene(&scene_dir); // n_planes = 2

    let seq = AlignSequence::new(scene_dir).unwrap();
    let err = seq.get_pair::<B>(5, &device, None).unwrap_err();
    assert!(
        matches!(err, stacker_nn::data::DataError::InvalidMetadata { .. }),
        "expected InvalidMetadata for out-of-range frame_index, got: {err:?}"
    );
}
