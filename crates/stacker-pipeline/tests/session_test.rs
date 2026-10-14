//! `session`: the in-memory stack sequence.

use std::{
    future::Future,
    io::Cursor,
    pin::pin,
    task::{Context, Poll, Waker},
};

use image::{DynamicImage, ImageFormat, RgbImage};
use stacker_algo::{
    apex::fuse::apex_fold_bytes_per_pixel,
    relief::threshold::compute_relief_threshold,
    strata::{CpuStrata, NO_PREVIEW},
};
use stacker_align::{
    Matrix3,
    device::{AsyncLevelDevice, Moments, ObjectiveSampler},
};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, PreprocessingSettings, StackingSettings},
};
use stacker_pipeline::{
    Cancel, NeverCancel, NoProgress, PipelineError, Progress, Stage,
    align::AsyncAlignBackend,
    compute::Compute,
    fuse::{relief_multigrid, strata},
    session::{
        AlignOutcome, BytesSource, FallbackKind, FrameFallback, PassKind, THRESHOLD_TABLE_LEN,
        align_frames, align_frames_async, auto_contrast_pct, crop_result, cull, cull_frames,
        estimate_peak_bytes, fit_pre_resize, fuse_apex, fuse_relief, fuse_strata, fuse_strata_with,
        load_frames, relief_finish, relief_focus_max, reorder, run_sort_cull,
        sort_frames_by_sharpness, threshold_for_pct, threshold_from_settings, threshold_table,
    },
};

const W: u32 = 160;
const H: u32 = 120;

/// A textured frame, shifted `shift` pixels to the left.
fn png(shift: u32) -> Vec<u8> {
    let img = RgbImage::from_fn(W, H, |x, y| {
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

fn source(shifts: &[u32]) -> BytesSource {
    BytesSource::new(
        shifts
            .iter()
            .enumerate()
            .map(|(i, &s)| (format!("web:/frame_{i}.png"), png(s)))
            .collect(),
    )
}

fn settings() -> StackingSettings {
    StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        auto_cull: true,
        pyramid_levels: 3,
        ..StackingSettings::default()
    }
}

struct Always;

impl Cancel for Always {
    fn is_cancelled(&self) -> bool {
        true
    }
}

fn load(shifts: &[u32]) -> Vec<PlanarImage<f32>> {
    load_frames(
        &mut source(shifts),
        &PreprocessingSettings::default(),
        &NeverCancel,
        &NoProgress,
    )
    .unwrap()
    .frames
}

fn same_planes(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> bool {
    let planes = |i: &PlanarImage<f32>| {
        (
            i.width,
            i.height,
            [i.luma.clone(), i.chroma_a.clone(), i.chroma_b.clone()],
        )
    };
    planes(a) == planes(b)
}

#[test]
fn loading_skips_unreadable_frames_and_names_them() {
    let mut items = vec![
        ("web:/a.png".to_owned(), png(0)),
        ("web:/broken.jpg".to_owned(), b"not an image".to_vec()),
        ("web:/b.png".to_owned(), png(2)),
    ];
    let loaded = load_frames(
        &mut BytesSource::new(std::mem::take(&mut items)),
        &PreprocessingSettings::default(),
        &NeverCancel,
        &NoProgress,
    )
    .unwrap();
    assert_eq!(loaded.names, ["web:/a.png", "web:/b.png"]);
    assert_eq!(loaded.frames.len(), 2);
    assert_eq!(loaded.skipped.len(), 1);
    assert_eq!(loaded.skipped[0].0, "web:/broken.jpg");
    let note = loaded.skipped_note().unwrap();
    assert!(
        note.contains("1 unreadable frame(s) skipped: broken.jpg"),
        "{note}"
    );
    let warning = loaded.skipped_warning().unwrap();
    assert!(warning.contains("broken.jpg"), "{warning}");
    assert!(warning.contains("first error"), "{warning}");
}

#[test]
fn loading_rejects_empty_unreadable_and_mismatched_sets() {
    let pre = PreprocessingSettings::default();
    let none = load_frames(
        &mut BytesSource::new(Vec::new()),
        &pre,
        &NeverCancel,
        &NoProgress,
    );
    assert!(matches!(none, Err(PipelineError::Invalid(_))));

    let all_bad = load_frames(
        &mut BytesSource::new(vec![("x.jpg".to_owned(), b"nope".to_vec())]),
        &pre,
        &NeverCancel,
        &NoProgress,
    );
    let message = all_bad.unwrap_err().to_string();
    assert!(message.contains("None of the 1 image(s)"), "{message}");

    let small = {
        let mut out = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::new(8, 8))
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        out
    };
    let mismatched = load_frames(
        &mut BytesSource::new(vec![
            ("a.png".to_owned(), png(0)),
            ("small.png".to_owned(), small),
        ]),
        &pre,
        &NeverCancel,
        &NoProgress,
    );
    let message = mismatched.unwrap_err().to_string();
    assert!(message.contains("small.png is 8x8 px"), "{message}");

    let cancelled = load_frames(&mut source(&[0, 1]), &pre, &Always, &NoProgress);
    assert!(matches!(cancelled, Err(PipelineError::Cancelled)));
}

#[test]
fn load_align_cull_fuse_runs() {
    let settings = settings();
    let loaded = load_frames(
        &mut source(&[0, 2, 4]),
        &settings.preprocessing,
        &NeverCancel,
        &NoProgress,
    )
    .unwrap();
    let mut frames = loaded.frames;
    assert_eq!(frames.len(), 3);

    let mut announced = Vec::new();
    let outcome = align_frames(
        &mut frames,
        &settings,
        &mut Compute::cpu(),
        &NeverCancel,
        &mut |i, frame, mask| {
            assert_eq!(mask.len(), frame.width * frame.height);
            announced.push(i);
        },
    )
    .expect("not cancelled");
    assert_eq!(announced, [1, 2]);
    assert!(frames.iter().all(|f| (f.width, f.height) == (160, 120)));
    if let Some((x, y, w, h)) = outcome.crop {
        assert!(x + w <= 160 && y + h <= 120);
    }

    let culled = cull(&frames, &settings).unwrap();
    assert_ne!(culled.order.len(), 0, "the cull keeps at least one frame");
    let mut all: Vec<usize> = culled
        .order
        .iter()
        .chain(&culled.dropped)
        .copied()
        .collect();
    all.sort_unstable();
    assert_eq!(all, [0, 1, 2], "kept and dropped partition the frames");

    let ordered = reorder(frames, &culled.order);
    assert_eq!(ordered.len(), culled.order.len());
    let fused = fuse_apex(&mut Compute::cpu(), &ordered, &settings);
    assert_eq!((fused.width, fused.height), (160, 120));
    assert!(fused.luma.iter().all(|v| v.is_finite()));

    let cropped = crop_result(fused, Some((10, 8, 100, 80)));
    assert_eq!((cropped.width, cropped.height), (100, 80));
}

#[test]
fn aligning_can_be_cancelled() {
    let settings = settings();
    let mut frames = load(&[0, 3]);
    let outcome = align_frames(
        &mut frames,
        &settings,
        &mut Compute::cpu(),
        &Always,
        &mut |_, _, _| {},
    );
    assert!(outcome.is_none());
}

#[test]
fn alignment_off_or_one_frame_changes_nothing() {
    let off = StackingSettings {
        alignment_mode: AlignmentModeSetting::None,
        ..settings()
    };
    let mut frames = load(&[0, 3]);
    let before = frames.clone();
    let outcome = align_frames(
        &mut frames,
        &off,
        &mut Compute::cpu(),
        &NeverCancel,
        &mut |_, _, _| {},
    )
    .unwrap();
    assert_eq!(outcome.crop, None);
    assert!(frames.iter().zip(&before).all(|(a, b)| same_planes(a, b)));
}

/// A device that declines every level and every warp.
struct Refuser;

impl AsyncLevelDevice for Refuser {
    fn level(
        &mut self,
        _reference: &[f32],
        _source: &[f32],
        _width: usize,
        _height: usize,
        _sampler: ObjectiveSampler,
    ) -> impl Future<Output = bool> {
        std::future::ready(false)
    }

    fn moments(&mut self, _inverse: &Matrix3<f32>) -> impl Future<Output = Option<Moments>> {
        std::future::ready(None)
    }
}

impl AsyncAlignBackend for Refuser {
    fn warp(
        &mut self,
        _frame: &PlanarImage<f32>,
        _apply: &Matrix3<f32>,
        _edge_clamped: bool,
    ) -> impl Future<Output = Option<(PlanarImage<f32>, Vec<bool>)>> {
        std::future::ready(None)
    }
}

/// Poll a future that never returns `Pending` to its end.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

#[test]
fn async_alignment_without_a_working_device_fails_with_the_gpu_only_message() {
    let mut frames = load(&[0, 3]);
    let error = block_on(align_frames_async(
        &mut frames,
        &settings(),
        &mut Refuser,
        &NeverCancel,
        &mut |_, _, _| {},
    ))
    .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("The GPU stopped during alignment"), "{text}");
    assert!(
        text.contains("In the browser, alignment runs on the GPU only."),
        "{text}"
    );
}

#[test]
fn async_alignment_off_needs_no_device_and_can_be_cancelled() {
    let off = StackingSettings {
        alignment_mode: AlignmentModeSetting::None,
        ..settings()
    };
    let mut frames = load(&[0, 3]);
    let outcome = block_on(align_frames_async(
        &mut frames,
        &off,
        &mut Refuser,
        &NeverCancel,
        &mut |_, _, _| {},
    ))
    .unwrap()
    .unwrap();
    assert_eq!(outcome.crop, None);

    let cancelled = block_on(align_frames_async(
        &mut frames,
        &settings(),
        &mut Refuser,
        &Always,
        &mut |_, _, _| {},
    ))
    .unwrap();
    assert!(cancelled.is_none());
}

#[test]
fn without_auto_cull_every_frame_is_kept_in_order() {
    let settings = StackingSettings {
        auto_cull: false,
        ..settings()
    };
    let frames = load(&[0, 1, 2, 3]);
    let outcome = cull(&frames, &settings).unwrap();
    assert_eq!(outcome.order, [0, 1, 2, 3]);
    assert_eq!(outcome.dropped, Vec::<usize>::new());
    assert_eq!(outcome.warning(), None);
}

#[test]
fn reorder_moves_frames_and_ignores_bad_indices() {
    let frames = load(&[0, 1, 2]);
    let marker = frames[2].luma[100];
    let ordered = reorder(frames, &[2, 2, 9, 0]);
    assert_eq!(ordered.len(), 2);
    assert!((ordered[0].luma[100] - marker).abs() < f32::EPSILON);
}

#[test]
fn the_relief_split_equals_the_one_call_fusion() {
    let settings = settings();
    let frames = load(&[0, 1, 2]);

    let one_call = fuse_relief(&mut Compute::cpu(), &frames, &settings, |max| {
        Some(threshold_for_pct(max, 0.4))
    })
    .unwrap();

    let mut compute = Compute::cpu();
    let focus = relief_focus_max(&mut compute, &frames, &settings).unwrap();
    let threshold = threshold_for_pct(focus.max(), 0.4);
    let auto = auto_contrast_pct(focus.max());
    assert!((0.0..=1.0).contains(&auto), "{auto}");
    let split = relief_finish(&mut compute, &frames, &settings, focus, threshold);
    assert!(same_planes(&one_call, &split));

    // The settings' own rule gives the same answer through both routes.
    let by_settings = fuse_relief(&mut Compute::cpu(), &frames, &settings, |max| {
        Some(threshold_from_settings(max, &settings))
    })
    .unwrap();
    let focus = relief_focus_max(&mut Compute::cpu(), &frames, &settings).unwrap();
    let threshold = threshold_from_settings(focus.max(), &settings);
    let split = relief_finish(&mut Compute::cpu(), &frames, &settings, focus, threshold);
    assert!(same_planes(&by_settings, &split));
}

#[test]
fn a_cancelled_relief_resolve_ends_the_fusion() {
    let frames = load(&[0, 1]);
    let fused = fuse_relief(&mut Compute::cpu(), &frames, &settings(), |_| None);
    assert!(fused.is_none());
}

#[test]
fn the_fallback_warning_names_the_files_per_kind() {
    let names: Vec<String> = ["dir/a.jpg", "dir/b.jpg", "dir/c.jpg"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    assert_eq!(AlignOutcome::default().warning(&names), None);
    let outcome = AlignOutcome {
        crop: None,
        fallbacks: vec![
            FrameFallback {
                index: 1,
                kind: FallbackKind::Seed,
            },
            FrameFallback {
                index: 2,
                kind: FallbackKind::Unaligned,
            },
        ],
        device_notice: Some("The GPU failed (lost).".to_owned()),
    };
    let text = outcome.warning(&names).unwrap();
    assert!(text.contains("refined for b.jpg"), "{text}");
    assert!(text.contains("warp c.jpg"), "{text}");
    assert!(text.ends_with("The GPU failed (lost)."), "{text}");
}

#[test]
fn the_estimate_grows_with_frames_pixels_and_passes() {
    let s = settings();
    let both = [PassKind::Relief, PassKind::Apex];
    let few = estimate_peak_bytes(10, 4000, 3000, &s, &[PassKind::Apex]);
    let many = estimate_peak_bytes(20, 4000, 3000, &s, &[PassKind::Apex]);
    let large = estimate_peak_bytes(10, 6000, 4000, &s, &[PassKind::Apex]);
    let two_passes = estimate_peak_bytes(10, 4000, 3000, &s, &both);
    assert!(few < many && few < large);
    // The second pass fuses while the first pass's result is still held.
    assert!(two_passes >= few);
    assert_eq!(estimate_peak_bytes(0, 4000, 3000, &s, &both), 0);

    // Apex folds the frames into one accumulator: 12 B/px per frame plus a fixed
    // working set (it was 28 per frame plus 66 when every pyramid was kept).
    let four_k = estimate_peak_bytes(28, 3840, 2160, &s, &[PassKind::Apex]);
    let fold = u64::try_from(apex_fold_bytes_per_pixel(s.pyramid_levels as usize)).unwrap();
    assert!(four_k >= (28 * 12 + fold) * 3840 * 2160, "{four_k}");
    assert!(four_k > 2 << 30, "{four_k}");
    // u64 arithmetic: no overflow on a 32-bit target's limits.
    assert!(estimate_peak_bytes(100, 20_000, 20_000, &s, &both) > u64::from(u32::MAX));

    // A smaller Pre-resize is cheaper, and the estimate sees the resize.
    let mut half = s;
    half.preprocessing.pre_resize_percent = 50;
    assert!(estimate_peak_bytes(10, 4000, 3000, &half, &[PassKind::Apex]) < few);
}

#[test]
fn fit_pre_resize_returns_a_multiple_of_five_that_fits() {
    let s = settings();
    let passes = [PassKind::Apex];
    let (frames, w, h) = (20, 4000, 3000);
    let at = |percent: u32| {
        let mut t = s.clone();
        t.preprocessing.pre_resize_percent = percent;
        estimate_peak_bytes(frames, w, h, &t, &passes)
    };

    // Plenty of room: the setting stays.
    assert_eq!(
        fit_pre_resize(frames, w, h, &s, &passes, u64::MAX),
        Some(100)
    );

    let budget = at(60);
    let fitted = fit_pre_resize(frames, w, h, &s, &passes, budget).unwrap();
    assert_eq!(fitted % 5, 0);
    assert!(at(fitted) <= budget);
    assert!(fitted >= 60);
    assert!(
        fitted == 100 || at(fitted + 5) > budget,
        "the largest that fits"
    );

    // Not even the smallest size fits.
    assert_eq!(fit_pre_resize(frames, w, h, &s, &passes, 1), None);
}

#[test]
fn the_threshold_table_matches_the_shared_percentile_rule() {
    let mut map = PlanarImage::new(25, 20);
    map.luma = (0..500_u16)
        .map(|i| f32::from(i % 37).mul_add(0.25, f32::from(i % 5)))
        .collect();
    let table = threshold_table(&map.luma);
    assert_eq!(table.len(), THRESHOLD_TABLE_LEN);
    for step in [0_u16, 1, 137, 500, 733, 999, 1000] {
        let pct = f32::from(step) / 1000.0;
        assert_eq!(
            table[usize::from(step)].to_bits(),
            threshold_for_pct(&map, pct).to_bits(),
            "step {step}"
        );
    }
    assert_eq!(threshold_table(&[]), vec![0.0; THRESHOLD_TABLE_LEN]);
}

// ── Cull and sort (auto_cull_threshold_pct, sort_by_sharpness) ──────────────────

/// Frame `k` carries hash noise in the column band `bands[k]` (of `n_bands`) and a
/// smooth ramp elsewhere; `u16::MAX` is a frame with no texture at all.
fn banded(w: u16, h: u16, n_bands: u16, bands: &[u16]) -> Vec<PlanarImage<f32>> {
    let width = w / n_bands;
    bands
        .iter()
        .enumerate()
        .map(|(k, &band)| {
            let mut img = PlanarImage::new(usize::from(w), usize::from(h));
            let seed = u32::try_from(k + 1).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let hash = (u32::from(x).wrapping_mul(73_856_093)
                        ^ u32::from(y).wrapping_mul(19_349_663)
                        ^ seed.wrapping_mul(83_492_791))
                        % 1000;
                    let noise = f32::from(u16::try_from(hash).unwrap()) / 1000.0 - 0.5;
                    let ramp = 0.4 + 0.2 * f32::from(x) / f32::from(w);
                    let i = usize::from(y) * usize::from(w) + usize::from(x);
                    img.luma[i] = ramp + if x / width == band { 0.3 * noise } else { 0.0 };
                }
            }
            img
        })
        .collect()
}

/// Four frames that are each sharp in one quarter of the image, in a scrambled
/// order: frame 1 holds the leftmost quarter, then frame 3, 2 and 0.
fn scrambled() -> Vec<PlanarImage<f32>> {
    banded(96, 64, 4, &[3, 0, 2, 1])
}

fn cull_settings(auto_cull: bool, sort: bool, pct: f32) -> StackingSettings {
    StackingSettings {
        auto_cull,
        sort_by_sharpness: sort,
        auto_cull_threshold_pct: pct,
        ..StackingSettings::default()
    }
}

#[test]
fn sort_by_sharpness_picks_between_census_order_and_stack_order() {
    let frames = scrambled();

    // Culling alone keeps the survivors in stack order ...
    let stack_order = cull(&frames, &cull_settings(true, false, 0.0)).unwrap();
    assert_eq!(stack_order.order, [0, 1, 2, 3]);
    assert_eq!(stack_order.dropped, [] as [usize; 0]);

    // ... sorting asks for the census (nearest-neighbour) order, with or without
    // the cull, and a sort alone drops nothing.
    let census_order = cull(&frames, &cull_settings(true, true, 0.0)).unwrap();
    assert_eq!(census_order.order, [1, 3, 2, 0]);
    let sort_only = cull(&frames, &cull_settings(false, true, 0.0)).unwrap();
    assert_eq!(sort_only.order, [1, 3, 2, 0]);
    assert_eq!(sort_only.dropped, [] as [usize; 0]);

    // Neither: the list as it is.
    let neither = cull(&frames, &cull_settings(false, false, 0.0)).unwrap();
    assert_eq!(neither.order, [0, 1, 2, 3]);

    // The standalone actions do not read the toggles.
    let sorted = sort_frames_by_sharpness(&frames, &cull_settings(false, false, 0.0)).unwrap();
    assert_eq!(sorted.order, [1, 3, 2, 0]);
    let culled = cull_frames(&frames, &cull_settings(false, false, 0.0)).unwrap();
    assert_eq!(culled.order, [0, 1, 2, 3]);
    let both = run_sort_cull(&frames, true, true, &cull_settings(false, false, 0.0)).unwrap();
    assert_eq!(both.order, [1, 3, 2, 0]);
}

#[test]
fn auto_cull_threshold_pct_sets_a_fixed_threshold() {
    // Four sharp frames win about a quarter of the detail each; the fifth has no
    // texture and wins nothing.
    let frames = banded(96, 64, 4, &[3, 0, 2, 1, u16::MAX]);

    // 0: the frame-count-scaled rule, min(2, 50/5) = 2 %.
    let automatic = cull(&frames, &cull_settings(true, false, 0.0)).unwrap();
    assert_eq!(automatic.order, [0, 1, 2, 3]);
    assert_eq!(automatic.dropped, [4]);

    // A fixed 10 % gives the same here ...
    let ten = cull_frames(&frames, &cull_settings(false, false, 10.0)).unwrap();
    assert_eq!(ten.order, [0, 1, 2, 3]);
    assert_eq!(ten.dropped, [4]);

    // ... and a fixed 40 % would drop every frame, so all are kept with a warning.
    let forty = cull(&frames, &cull_settings(true, false, 40.0)).unwrap();
    assert_eq!(forty.order, [0, 1, 2, 3, 4]);
    assert_eq!(forty.dropped, [] as [usize; 0]);
    let warning = forty.warning().expect("a warning");
    assert!(
        warning.contains("would have dropped every frame"),
        "{warning}"
    );
}

#[test]
fn sorting_and_culling_need_frames() {
    let settings = cull_settings(true, true, 0.0);
    assert!(matches!(
        cull(&[], &settings),
        Err(PipelineError::Invalid(_))
    ));
    assert!(matches!(
        sort_frames_by_sharpness(&[], &settings),
        Err(PipelineError::Invalid(_))
    ));
    // Nothing requested, nothing to do: not an error.
    let none = cull(&[], &cull_settings(false, false, 0.0)).unwrap();
    assert_eq!(none.order, [] as [usize; 0]);
}

// ── Strata ──────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Recorder(std::cell::RefCell<Vec<(Stage, f32)>>);

impl Progress for Recorder {
    fn stage(&self, stage: Stage, fraction: f32, _label: &str) {
        self.0.borrow_mut().push((stage, fraction));
    }
}

#[test]
fn strata_reports_two_ticks_per_frame_and_previews() {
    let settings = StackingSettings::default();
    let frames = load(&[0, 1, 2]);
    let plain = strata(&frames, &settings);

    let recorder = Recorder::default();
    let mut previews = Vec::new();
    let fused = fuse_strata(&frames, &settings, &recorder, 1, |p| {
        previews.push(p.clone());
    });
    assert!(same_planes(&plain, &fused), "progress changes nothing");

    let seen = recorder.0.borrow();
    // One opening report and two ticks (saliency, accumulation) per frame.
    assert_eq!(seen.len(), 1 + 6);
    assert!(seen.iter().all(|(stage, _)| *stage == Stage::Fuse));
    assert!(seen.windows(2).all(|pair| pair[0].1 <= pair[1].1));
    assert!(seen[0].1.abs() < f32::EPSILON);
    assert!((seen[6].1 - 1.0).abs() < f32::EPSILON);

    // A snapshot after every frame but the last of the second pass, then the
    // finished image.
    assert_eq!(previews.len(), 3);
    assert!(same_planes(previews.last().unwrap(), &fused));

    // No previews, the CPU kernels spelled out: the same image.
    let mut count = 0;
    let again = fuse_strata_with(
        &mut CpuStrata,
        &frames,
        &settings,
        &NoProgress,
        NO_PREVIEW,
        |_| count += 1,
    );
    assert_eq!(count, 0);
    assert!(same_planes(&plain, &again));

    let empty = fuse_strata(&[], &settings, &NoProgress, NO_PREVIEW, |_| {});
    assert_eq!(empty.width, 0);
}

// ── Multigrid ───────────────────────────────────────────────────────────────────

#[test]
fn the_multigrid_relief_split_equals_the_one_call_fusion() {
    let settings = StackingSettings {
        relief_use_multigrid: true,
        relief_contrast_pct: 0.4,
        ..settings()
    };
    let frames = load(&[0, 1, 2]);

    let one_call = fuse_relief(&mut Compute::cpu(), &frames, &settings, |max| {
        Some(threshold_from_settings(max, &settings))
    })
    .unwrap();
    assert!(same_planes(
        &one_call,
        &relief_multigrid(&frames, &settings)
    ));

    let mut compute = Compute::cpu();
    let focus = relief_focus_max(&mut compute, &frames, &settings).unwrap();
    let threshold = threshold_from_settings(focus.max(), &settings);
    let split = relief_finish(&mut compute, &frames, &settings, focus, threshold);
    assert!(same_planes(&one_call, &split));

    // A different threshold moves the result: the decision still belongs to the
    // caller, as with the guided path.
    let focus = relief_focus_max(&mut compute, &frames, &settings).unwrap();
    let other = relief_finish(&mut compute, &frames, &settings, focus, f32::INFINITY);
    assert!(!same_planes(&one_call, &other));

    // The caller can still cancel.
    assert!(fuse_relief(&mut Compute::cpu(), &frames, &settings, |_| None).is_none());
}

/// The preview starts at the same fraction the run turns into its threshold.
#[test]
fn the_preview_start_value_is_the_auto_detected_fraction_of_the_run() {
    let frames = load(&[0, 1, 2]);
    let settings = settings();
    let focus = relief_focus_max(&mut Compute::cpu(), &frames, &settings).unwrap();
    let preview = auto_contrast_pct(focus.max());
    let run = compute_relief_threshold(&focus.max().luma, 0.0, true);
    assert!((preview - run.contrast_pct).abs() < f32::EPSILON);
    let auto_settings = StackingSettings {
        relief_auto_detect: true,
        ..settings
    };
    assert!(
        (threshold_from_settings(focus.max(), &auto_settings) - run.value).abs() < f32::EPSILON
    );
}

// ── Memory ──────────────────────────────────────────────────────────────────────

#[test]
fn strata_and_the_multigrid_are_in_the_estimate() {
    let s = settings();
    let pixels = 4000 * 3000_u64;

    // Ten frames: the Strata pass dominates the load, align and cull stages. A
    // frame is 12 bytes, the working set 150 per pixel whatever the frame count.
    let strata_peak = estimate_peak_bytes(10, 4000, 3000, &s, &[PassKind::Strata]);
    assert_eq!(strata_peak, (10 * 12 + 150) * pixels);
    // The working set does not grow with the frames, the frames do.
    let more = estimate_peak_bytes(20, 4000, 3000, &s, &[PassKind::Strata]);
    assert_eq!(more - strata_peak, 10 * 12 * pixels);
    // Both stream: Apex's fixed set (the fold) is below Strata's 150. The cull
    // stage (16 per frame and pixel) would dominate both with 40 frames, so it
    // is switched off for this comparison.
    let no_cull = StackingSettings {
        auto_cull: false,
        ..s.clone()
    };
    let deep_strata = estimate_peak_bytes(40, 4000, 3000, &no_cull, &[PassKind::Strata]);
    let deep_apex = estimate_peak_bytes(40, 4000, 3000, &no_cull, &[PassKind::Apex]);
    assert!(deep_apex < deep_strata);
    // After an earlier pass the result of it (12 bytes per pixel) is still held.
    let after = estimate_peak_bytes(10, 4000, 3000, &s, &[PassKind::Relief, PassKind::Strata]);
    assert_eq!(after, (10 * 12 + 150 + 12) * pixels);

    // The multigrid adds its solver (28), depth (4) and best-focus map (4).
    let guided = estimate_peak_bytes(10, 4000, 3000, &s, &[PassKind::Relief]);
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..s.clone()
    };
    let solved = estimate_peak_bytes(10, 4000, 3000, &multigrid, &[PassKind::Relief]);
    assert_eq!(solved - guided, 36 * pixels);

    // A Pre-resize that fits the budget exists for Strata too.
    let fitted = fit_pre_resize(10, 4000, 3000, &s, &[PassKind::Strata], strata_peak / 2);
    assert!(fitted.is_some_and(|percent| percent < 100));
}
