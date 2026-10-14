//! The streaming fusions and the tile loop's read-ahead: the streamed `Apex` fold
//! and the streamed `Relief` focus maps equal the batch paths bit for bit, the
//! prefetch does not change a result, the crop settings reach the session.

use std::{cell::RefCell, io::Cursor};

use image::{DynamicImage, ImageFormat, RgbImage};
use stacker_algo::{
    apex::fuse::build_and_fuse_pyramids, relief::threshold::mask_above, strata::NO_PREVIEW,
};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, PreprocessingSettings, StackingSettings},
};
use stacker_pipeline::{
    NeverCancel, NoProgress, PipelineError,
    compute::Compute,
    fuse,
    session::{
        BytesSource, PassKind, align_frames, crop_result, crop_result_for, estimate_peak_bytes,
        fuse_apex, fuse_apex_with_preview, fuse_strata_compute, load_frames,
    },
    tiled::{FusePlan, Hooks, MemoryFrames, Prefetch, TileFrames, fuse_tiles_prefetch, plan_run},
    tiling::{Method, TilePlan},
};

#[expect(
    clippy::cast_precision_loss,
    reason = "pixel coordinates and frame counts are small"
)]
fn stack(width: usize, height: usize, count: usize) -> Vec<PlanarImage<f32>> {
    (0..count)
        .map(|k| {
            let mut img = PlanarImage::new(width, height);
            let band = width / count;
            for row in 0..height {
                for col in 0..width {
                    let i = row * width + col;
                    let hash =
                        ((col * 73_856_093) ^ (row * 19_349_663) ^ ((k + 1) * 83_492_791)) % 1000;
                    let noise = hash as f32 / 1000.0 - 0.5;
                    let sharp = if col / band.max(1) == k {
                        0.3 * noise
                    } else {
                        0.0
                    };
                    let ramp = 0.4 + 0.2 * col as f32 / width as f32;
                    img.luma[i] = ramp + sharp;
                    img.chroma_a[i] = 0.05 * (k as f32 + sharp);
                    img.chroma_b[i] = -0.02 * sharp;
                }
            }
            img
        })
        .collect()
}

fn same(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.luma == b.luma
        && a.chroma_a == b.chroma_a
        && a.chroma_b == b.chroma_b
}

fn settings() -> StackingSettings {
    StackingSettings {
        pyramid_levels: 3,
        ..StackingSettings::default()
    }
}

/// The session's `Apex` (the streaming fold) is the batch fusion of all pyramids.
#[test]
fn the_streamed_apex_equals_the_batch_fusion() {
    let frames = stack(60, 48, 4);
    for (color, grit) in [(false, true), (true, false), (false, false)] {
        let s = StackingSettings {
            use_all_color_channels: color,
            grit_suppression: grit,
            ..settings()
        };
        let streamed = fuse_apex(&mut Compute::cpu(), &frames, &s);
        let batch = build_and_fuse_pyramids(&frames, 3, color, grit).reconstruct();
        assert!(same(&streamed, &batch), "color {color}, grit {grit}");
        assert!(same(&streamed, &fuse::apex(&frames, &s)));
    }
}

/// Previews come after every `preview_every` frames (not the last) and once with
/// the finished image, which is the plain fusion.
#[test]
fn the_apex_preview_hook_announces_partial_and_final_images() {
    let frames = stack(48, 40, 5);
    let s = settings();
    let mut previews = Vec::new();
    let fused = fuse_apex_with_preview(&mut Compute::cpu(), &frames, &s, &NoProgress, 2, |p| {
        previews.push(p.clone());
    });
    assert!(same(&fused, &fuse_apex(&mut Compute::cpu(), &frames, &s)));
    assert_eq!(previews.len(), 3, "after frames 2 and 4, then the result");
    assert!(same(previews.last().unwrap(), &fused));
    // The partial preview is the fusion of the frames so far.
    assert!(same(&previews[0], &fuse::apex(&frames[..2], &s)));

    let mut none = 0;
    let _ = fuse_apex_with_preview(
        &mut Compute::cpu(),
        &frames,
        &s,
        &NoProgress,
        NO_PREVIEW,
        |_| {
            none += 1;
        },
    );
    assert_eq!(none, 0);
}

/// The streamed `Relief` focus maps give the result of the per-frame maps.
#[test]
fn the_streamed_relief_equals_the_per_frame_map_path() {
    let frames = stack(64, 50, 4);
    for contrast in [0.0_f32, 30.0, 80.0] {
        let s = StackingSettings {
            relief_contrast_pct: contrast,
            relief_auto_detect: false,
            relief_smoothing_radius: 3,
            ..settings()
        };
        let radius = s.relief_estimation_radius as usize;
        let maps = fuse::focus_maps(&frames, radius);
        let threshold = fuse::relief_threshold(&maps.max.luma, &s);
        let mask = mask_above(&maps.max.luma, threshold.value);
        let reference = fuse::select_and_smooth(&frames, &maps.smls, &mask, 3);

        assert!(same(&fuse::relief(&frames, &s), &reference), "{contrast}");
        assert!(same(
            &fuse::relief_tile(&frames, &s, threshold.value),
            &reference
        ));
        assert_eq!(fuse::focus_max_map(&frames, radius), maps.max.luma);
        let mut compute = Compute::cpu();
        let through = compute
            .relief(&frames, &s, |_| Some(threshold.value))
            .unwrap();
        assert!(same(&through, &reference));
    }
}

/// A source that cannot be shared: the serial, unprefetched path.
struct Serial<'a>(MemoryFrames<'a>);

impl TileFrames for Serial<'_> {
    fn tile(&self, frame: usize, plan: &TilePlan) -> Result<PlanarImage<f32>, PipelineError> {
        self.0.tile(frame, plan)
    }
}

fn fuse_with(
    method: Method,
    frames: &dyn TileFrames,
    all: &[PlanarImage<f32>],
    prefetch: Prefetch,
) -> PlanarImage<f32> {
    let s = settings();
    let size = (all[0].width, all[0].height);
    let plan = plan_run(method, size, 32, &s);
    assert!(plan.tiles.len() > 1);
    let order: Vec<usize> = (0..all.len()).collect();
    let mut out = PlanarImage::new(size.0, size.1);
    fuse_tiles_prefetch(
        &FusePlan::new(method, &s, &plan, &order),
        frames,
        &mut Compute::cpu(),
        &mut out,
        &Hooks::silent(),
        prefetch,
    )
    .unwrap();
    out
}

/// Reading ahead and reading in parallel do not change a result.
#[test]
fn the_prefetch_does_not_change_the_result() {
    let frames = stack(96, 80, 4);
    for method in [Method::Apex, Method::Relief, Method::Strata] {
        let off = fuse_with(method, &MemoryFrames(&frames), &frames, Prefetch::Off);
        let auto = fuse_with(method, &MemoryFrames(&frames), &frames, Prefetch::Auto);
        let serial = fuse_with(
            method,
            &Serial(MemoryFrames(&frames)),
            &frames,
            Prefetch::Auto,
        );
        assert!(same(&off, &auto), "{method:?}: prefetch on and off");
        assert!(same(&off, &serial), "{method:?}: shared and serial reads");
    }
}

/// A tile that cannot be read ends the tile loop with that error, on every path.
#[test]
fn a_failing_read_ends_the_loop_with_its_error() {
    struct Failing<'a>(MemoryFrames<'a>, RefCell<usize>);
    impl TileFrames for Failing<'_> {
        fn tile(&self, frame: usize, plan: &TilePlan) -> Result<PlanarImage<f32>, PipelineError> {
            let mut calls = self.1.borrow_mut();
            *calls += 1;
            if *calls > 6 {
                return Err(PipelineError::invalid("read failed"));
            }
            self.0.tile(frame, plan)
        }
    }
    let frames = stack(96, 80, 3);
    for method in [Method::Apex, Method::Relief, Method::Strata] {
        let source = Failing(MemoryFrames(&frames), RefCell::new(0));
        let s = settings();
        let plan = plan_run(method, (96, 80), 32, &s);
        let order = [0, 1, 2];
        let mut out = PlanarImage::new(96, 80);
        let result = fuse_tiles_prefetch(
            &FusePlan::new(method, &s, &plan, &order),
            &source,
            &mut Compute::cpu(),
            &mut out,
            &Hooks::silent(),
            Prefetch::Auto,
        );
        assert!(
            matches!(result, Err(PipelineError::Invalid(ref m)) if m == "read failed"),
            "{method:?}: {result:?}"
        );
    }
}

/// `Strata` through the compute handle's kernels (the CPU's here) equals the
/// plain CPU fusion.
#[test]
fn strata_through_the_compute_handle_equals_the_cpu_fusion() {
    let frames = stack(48, 40, 3);
    let s = settings();
    let direct = fuse::strata(&frames, &s);
    let through = fuse_strata_compute(
        &mut Compute::cpu(),
        &frames,
        &s,
        &NoProgress,
        NO_PREVIEW,
        |_| {},
    );
    assert!(same(&direct, &through));
}

fn png(shift: u32) -> Vec<u8> {
    let img = RgbImage::from_fn(160, 120, |x, y| {
        let x = x + shift;
        let v = u8::try_from((x * 37 + y * 91 + (x * y) % 13 * 17) % 256).unwrap();
        let w = u8::try_from((x * 11 + y * 5) % 256).unwrap();
        image::Rgb([v, w, v / 2 + 40])
    });
    let mut out = Vec::new();
    DynamicImage::ImageRgb8(img)
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .unwrap();
    out
}

/// `crop_to_common_area` reaches the session's alignment; `resize_cropped_to_original`
/// stretches the cropped result back to the canvas.
#[test]
fn the_crop_settings_reach_the_session() {
    let mut source = BytesSource::new(
        [0_u32, 6, 12]
            .iter()
            .enumerate()
            .map(|(i, &s)| (format!("f{i}.png"), png(s)))
            .collect(),
    );
    let mut load = || {
        load_frames(
            &mut source,
            &PreprocessingSettings::default(),
            &NeverCancel,
            &NoProgress,
        )
        .unwrap()
        .frames
    };
    let mut s = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        ..settings()
    };
    let mut frames = load();
    let outcome = align_frames(
        &mut frames,
        &s,
        &mut Compute::cpu(),
        &NeverCancel,
        &mut |_, _, _| {},
    )
    .unwrap();
    let crop = outcome.crop.expect("shifted frames leave a border to crop");

    s.crop_to_common_area = false;
    let mut frames = load();
    let outcome = align_frames(
        &mut frames,
        &s,
        &mut Compute::cpu(),
        &NeverCancel,
        &mut |_, _, _| {},
    )
    .unwrap();
    assert_eq!(outcome.crop, None, "no crop when the setting is off");

    // The resize target is the canvas; off, the cropped size stays.
    let fused = PlanarImage::new(160, 120);
    let kept = crop_result_for(fused.clone(), Some(crop), &s);
    assert_eq!((kept.width, kept.height), (crop.2, crop.3));
    assert!(same(&kept, &crop_result(fused.clone(), Some(crop))));
    s.resize_cropped_to_original = true;
    let resized = crop_result_for(fused.clone(), Some(crop), &s);
    assert_eq!((resized.width, resized.height), (160, 120));
    let untouched = crop_result_for(fused, None, &s);
    assert_eq!((untouched.width, untouched.height), (160, 120));
}

/// The chain adds its previous frame to the align stage of the estimate.
#[test]
fn the_chain_is_in_the_estimate() {
    let mut s = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        auto_cull: false,
        ..settings()
    };
    // A deep stack of a pass whose own figure is small, so align dominates.
    let plain = estimate_peak_bytes(2, 4000, 3000, &s, &[]);
    s.chain_alignment = true;
    let chained = estimate_peak_bytes(2, 4000, 3000, &s, &[]);
    assert!(chained > plain, "{chained} vs {plain}");
    let _ = PassKind::Apex;
}
