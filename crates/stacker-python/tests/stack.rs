//! The stacking flows behind `stack_arrays`, `stack_files` and `batch_stack`, tested without
//! Python on small synthetic stacks.

use std::{cell::RefCell, path::Path};

use image::{Rgb, RgbImage};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, ImageSavingSettings, OutputFormat, StackingSettings},
};
use stacker_pipeline::{
    Cancel, NoProgress, PipelineError, Progress, Stage,
    tiled::{Hooks, RunLog},
};
use stacker_python::{
    progress::ProgressForwarder,
    stack::{
        BatchEntry, CollectLog, FileJob, FuseMode, batch_output_path, batch_subfolders, read_rgb16,
        run_batch, run_files, select_indices, stack_in_ram,
    },
};

/// Settings that make a tiny synthetic stack deterministic: no alignment, no brightness
/// correction, no cull, a shallow pyramid and the smallest `Strata` base radius.
fn quiet_settings() -> StackingSettings {
    StackingSettings {
        alignment_mode: AlignmentModeSetting::None,
        correct_brightness: false,
        auto_cull: false,
        sort_by_sharpness: false,
        pyramid_levels: 3,
        strata_base_radius: 8,
        ..StackingSettings::default()
    }
}

/// A `w × h` linear-light frame that is a fine checkerboard inside the columns
/// `sharp.0..sharp.1` and flat gray elsewhere.
fn frame(w: usize, h: usize, sharp: (usize, usize)) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.luma[y * w + x] = if (sharp.0..sharp.1).contains(&x) {
                if (x + y) % 2 == 0 { 0.8 } else { 0.1 }
            } else {
                0.4
            };
        }
    }
    img
}

/// The variance of the `luma` plane over the columns `xs` and rows `ys`.
fn luma_variance(
    img: &PlanarImage<f32>,
    xs: std::ops::Range<usize>,
    ys: std::ops::Range<usize>,
) -> f32 {
    let mut values = Vec::new();
    for y in ys {
        for x in xs.clone() {
            values.push(img.luma[y * img.width + x]);
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "a handful of samples")]
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n
}

fn complementary_pair() -> Vec<PlanarImage<f32>> {
    vec![frame(64, 64, (0, 32)), frame(64, 64, (32, 64))]
}

// ── the method and the selection ────────────────────────────────────────────

#[test]
fn modes_parse_with_their_older_names() {
    assert_eq!(FuseMode::parse("apex").unwrap(), FuseMode::Apex);
    assert_eq!(FuseMode::parse("APEX").unwrap(), FuseMode::Apex);
    assert_eq!(FuseMode::parse("pmax").unwrap(), FuseMode::Apex);
    assert_eq!(FuseMode::parse("relief").unwrap(), FuseMode::Relief);
    assert_eq!(FuseMode::parse("dmap").unwrap(), FuseMode::Relief);
    assert_eq!(FuseMode::parse(" Strata ").unwrap(), FuseMode::Strata);
    for mode in [FuseMode::Apex, FuseMode::Relief, FuseMode::Strata] {
        assert_eq!(FuseMode::parse(mode.name()).unwrap(), mode);
    }
}

#[test]
fn an_unknown_mode_lists_the_valid_ones() {
    let err = FuseMode::parse("blend").unwrap_err();
    for needle in ["blend", "apex", "relief", "strata"] {
        assert!(err.message().contains(needle), "{err}");
    }
    let err = FuseMode::parse("ai").unwrap_err();
    assert!(err.message().contains("not available"), "{err}");
}

#[test]
fn frames_are_selected_by_reverse_and_every_nth() {
    assert_eq!(select_indices(5, false, 1), vec![0, 1, 2, 3, 4]);
    assert_eq!(select_indices(5, true, 1), vec![4, 3, 2, 1, 0]);
    assert_eq!(select_indices(7, false, 3), vec![0, 3, 6]);
    assert_eq!(select_indices(5, true, 2), vec![4, 2, 0]);
    assert_eq!(select_indices(4, false, 0), vec![0, 1, 2, 3]);
    assert_eq!(select_indices(0, false, 1), Vec::<usize>::new());
}

// ── frames in RAM ───────────────────────────────────────────────────────────

#[test]
fn apex_keeps_the_sharp_half_of_each_frame() {
    let fused = stack_in_ram(
        complementary_pair(),
        &quiet_settings(),
        FuseMode::Apex,
        &Hooks::silent(),
    )
    .unwrap();
    assert_eq!((fused.width, fused.height), (64, 64));
    assert!(fused.is_consistent());
    assert!(fused.luma.iter().all(|v| v.is_finite()));
    // The checkerboard of frame 0 (left) and of frame 1 (right) both survive.
    assert!(luma_variance(&fused, 4..24, 8..56) > 0.01);
    assert!(luma_variance(&fused, 40..60, 8..56) > 0.01);
}

#[test]
fn relief_and_strata_return_a_finite_image_of_the_input_size() {
    for mode in [FuseMode::Relief, FuseMode::Strata] {
        let fused = stack_in_ram(
            complementary_pair(),
            &quiet_settings(),
            mode,
            &Hooks::silent(),
        )
        .unwrap();
        assert_eq!((fused.width, fused.height), (64, 64), "{mode:?}");
        assert!(fused.is_consistent(), "{mode:?}");
        assert!(fused.luma.iter().all(|v| v.is_finite()), "{mode:?}");
        let first = fused.luma[0];
        assert!(
            fused.luma.iter().any(|&v| (v - first).abs() > 1.0e-3),
            "{mode:?} produced a flat image"
        );
    }
}

#[test]
fn a_single_frame_is_returned_as_it_is_in_size() {
    let fused = stack_in_ram(
        vec![frame(32, 24, (0, 32))],
        &quiet_settings(),
        FuseMode::Apex,
        &Hooks::silent(),
    )
    .unwrap();
    assert_eq!((fused.width, fused.height), (32, 24));
}

#[test]
fn an_empty_stack_is_an_invalid_request() {
    let err = stack_in_ram(
        Vec::new(),
        &quiet_settings(),
        FuseMode::Apex,
        &Hooks::silent(),
    )
    .unwrap_err();
    assert!(matches!(err, PipelineError::Invalid(_)), "{err}");
}

#[test]
fn frames_of_different_sizes_are_rejected_naming_the_frame() {
    let frames = vec![frame(16, 16, (0, 16)), frame(16, 15, (0, 16))];
    let err =
        stack_in_ram(frames, &quiet_settings(), FuseMode::Apex, &Hooks::silent()).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("frame 1") && message.contains("same size"),
        "{message}"
    );
}

struct AlwaysCancel;

impl Cancel for AlwaysCancel {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn a_cancelled_run_reports_cancelled() {
    let log = CollectLog::default();
    let hooks = Hooks {
        progress: &NoProgress,
        cancel: &AlwaysCancel,
        log: &log,
    };
    let err = stack_in_ram(
        complementary_pair(),
        &quiet_settings(),
        FuseMode::Apex,
        &hooks,
    )
    .unwrap_err();
    assert!(matches!(err, PipelineError::Cancelled), "{err}");
}

#[test]
fn progress_runs_through_the_stages_in_order_and_ends_at_the_total() {
    let calls = RefCell::new(Vec::new());
    let forwarder = ProgressForwarder::new(|stage, current, total| {
        calls.borrow_mut().push((stage, current, total));
    });
    let log = CollectLog::default();
    let hooks = Hooks {
        progress: &forwarder,
        cancel: &stacker_pipeline::NeverCancel,
        log: &log,
    };
    stack_in_ram(
        complementary_pair(),
        &quiet_settings(),
        FuseMode::Apex,
        &hooks,
    )
    .unwrap();
    let calls = calls.into_inner();
    let stages: Vec<&str> = calls.iter().map(|c| c.0).collect();
    let first_fuse = stages.iter().position(|&s| s == "fuse").unwrap();
    let last_align = stages.iter().rposition(|&s| s == "align").unwrap();
    assert!(last_align < first_fuse, "{stages:?}");
    for stage in ["align", "fuse"] {
        let last = calls.iter().rfind(|c| c.0 == stage).unwrap();
        assert_eq!((last.1, last.2), (1000, 1000), "{stage}");
        let values: Vec<u32> = calls.iter().filter(|c| c.0 == stage).map(|c| c.1).collect();
        assert!(
            values.windows(2).all(|w| w[0] <= w[1]),
            "{stage}: {values:?}"
        );
    }
}

#[test]
fn alignment_runs_and_keeps_the_size_of_a_shifted_stack() {
    // A smooth bump, then the same bump shifted by two pixels.
    let bump = |dx: f32| {
        let mut img = PlanarImage::new(64, 64);
        for y in 0..64_u8 {
            for x in 0..64_u8 {
                let (fx, fy) = (f32::from(x) - 32.0 - dx, f32::from(y) - 32.0);
                let radius_sq = fy.mul_add(fy, fx * fx);
                img.luma[usize::from(y) * 64 + usize::from(x)] =
                    0.6_f32.mul_add((-radius_sq / 200.0).exp(), 0.1);
            }
        }
        img
    };
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        correct_brightness: false,
        auto_cull: false,
        pyramid_levels: 3,
        ..StackingSettings::default()
    };
    let fused = stack_in_ram(
        vec![bump(0.0), bump(2.0)],
        &settings,
        FuseMode::Apex,
        &Hooks::silent(),
    )
    .unwrap();
    // The size is the crop to the common area: at most the input, and not collapsed.
    assert!(fused.width <= 64 && fused.height <= 64);
    assert!(fused.width >= 32 && fused.height >= 32);
    assert!(fused.luma.iter().all(|v| v.is_finite()));
}

// ── files ───────────────────────────────────────────────────────────────────

/// Write `count` 64 x 48 frames, frame `k` sharp in its own band of columns.
fn write_frames(dir: &Path, count: u8) -> Vec<std::path::PathBuf> {
    std::fs::create_dir_all(dir).unwrap();
    (0..count)
        .map(|k| {
            let band = u32::from(k) * 20..u32::from(k) * 20 + 24;
            let img = RgbImage::from_fn(64, 48, |x, y| {
                let value = if band.contains(&x) {
                    if (x + y) % 2 == 0 { 230 } else { 30 }
                } else {
                    110
                };
                Rgb([value, value, value])
            });
            let path = dir.join(format!("frame_{k}.png"));
            img.save(&path).unwrap();
            path
        })
        .collect()
}

#[test]
fn stack_files_out_of_core_writes_a_png() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_frames(&dir.path().join("in"), 3);
    let output = dir.path().join("out").join("stacked.png");
    run_files(
        &FileJob {
            paths: &paths,
            output_file: &output,
            mode: FuseMode::Apex,
            tile_size: 64,
        },
        &quiet_settings(),
        &Hooks::silent(),
    )
    .unwrap();
    let written = image::open(&output).unwrap();
    assert_eq!(image::GenericImageView::dimensions(&written), (64, 48));
}

#[test]
fn stack_files_in_ram_writes_the_format_of_the_extension() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_frames(&dir.path().join("in"), 2);
    for (name, mode) in [
        ("a.tif", FuseMode::Apex),
        ("b.jpg", FuseMode::Relief),
        ("c.png", FuseMode::Strata),
    ] {
        let output = dir.path().join(name);
        run_files(
            &FileJob {
                paths: &paths,
                output_file: &output,
                mode,
                tile_size: 0,
            },
            &quiet_settings(),
            &Hooks::silent(),
        )
        .unwrap();
        let written = image::open(&output).unwrap();
        assert_eq!(
            image::GenericImageView::dimensions(&written),
            (64, 48),
            "{name}"
        );
    }
}

#[test]
fn the_tile_size_of_the_settings_serves_when_the_argument_is_zero() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_frames(&dir.path().join("in"), 2);
    let output = dir.path().join("tiles.png");
    let settings = StackingSettings {
        tile_size: 64,
        ..quiet_settings()
    };
    run_files(
        &FileJob {
            paths: &paths,
            output_file: &output,
            mode: FuseMode::Apex,
            tile_size: 0,
        },
        &settings,
        &Hooks::silent(),
    )
    .unwrap();
    assert!(output.exists());
}

#[test]
fn a_tile_size_below_the_minimum_is_an_invalid_request() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_frames(&dir.path().join("in"), 2);
    let output = dir.path().join("never.png");
    let err = run_files(
        &FileJob {
            paths: &paths,
            output_file: &output,
            mode: FuseMode::Apex,
            tile_size: 10,
        },
        &quiet_settings(),
        &Hooks::silent(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, PipelineError::Invalid(m) if m.contains("64")),
        "{err}"
    );
    assert!(!output.exists());
}

#[test]
fn an_output_path_without_a_known_extension_is_an_invalid_request() {
    let paths = vec![std::path::PathBuf::from("does_not_exist.png")];
    for name in ["result.bmp", "result"] {
        let output = Path::new(name);
        let err = run_files(
            &FileJob {
                paths: &paths,
                output_file: output,
                mode: FuseMode::Apex,
                tile_size: 0,
            },
            &quiet_settings(),
            &Hooks::silent(),
        )
        .unwrap_err();
        assert!(
            matches!(&err, PipelineError::Invalid(m) if m.contains(".tif")),
            "{err}"
        );
    }
}

#[test]
fn a_missing_input_file_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let paths = vec![dir.path().join("missing.png")];
    let output = dir.path().join("out.png");
    assert!(
        run_files(
            &FileJob {
                paths: &paths,
                output_file: &output,
                mode: FuseMode::Apex,
                tile_size: 0,
            },
            &quiet_settings(),
            &Hooks::silent(),
        )
        .is_err()
    );
    assert!(!output.exists());
}

#[test]
fn the_output_file_is_never_read_as_an_input_frame() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_frames(&dir.path().join("in"), 2);
    // The output is `frame_0`, which is also listed as an input: only `frame_1` stays, and a
    // single frame is still a result.
    let output = dir.path().join("in").join("frame_0.png");
    run_files(
        &FileJob {
            paths: &paths,
            output_file: &output,
            mode: FuseMode::Apex,
            tile_size: 0,
        },
        &quiet_settings(),
        &Hooks::silent(),
    )
    .unwrap();
    let written = image::open(&output).unwrap();
    assert_eq!(image::GenericImageView::dimensions(&written), (64, 48));
}

#[test]
fn a_tiff_result_with_copy_metadata_warns_that_it_cannot_carry_exif() {
    let dir = tempfile::tempdir().unwrap();
    let paths = write_frames(&dir.path().join("in"), 2);
    let output = dir.path().join("out.tif");
    let settings = StackingSettings {
        image_saving: ImageSavingSettings {
            copy_metadata: true,
            ..ImageSavingSettings::default()
        },
        ..quiet_settings()
    };
    let log = CollectLog::default();
    let hooks = Hooks {
        progress: &NoProgress,
        cancel: &stacker_pipeline::NeverCancel,
        log: &log,
    };
    run_files(
        &FileJob {
            paths: &paths,
            output_file: &output,
            mode: FuseMode::Apex,
            tile_size: 0,
        },
        &settings,
        &hooks,
    )
    .unwrap();
    let warnings = log.into_warnings();
    assert!(warnings.iter().any(|w| w.contains("TIFF")), "{warnings:?}");
}

// ── batches ─────────────────────────────────────────────────────────────────

#[test]
fn the_batch_output_name_follows_the_template_and_the_format() {
    let saving = ImageSavingSettings {
        output_format: OutputFormat::Png,
        filename_template: "{name}_final".to_owned(),
        ..ImageSavingSettings::default()
    };
    assert_eq!(
        batch_output_path(Path::new("out"), "mite", &saving),
        Path::new("out").join("mite_final.png")
    );
    let default = ImageSavingSettings::default();
    assert_eq!(
        batch_output_path(Path::new("out"), "mite", &default),
        Path::new("out").join("mite_stacked.tif")
    );
}

#[test]
fn a_batch_stacks_every_folder_that_holds_frames() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in");
    write_frames(&input.join("beetle"), 2);
    write_frames(&input.join("ant"), 3);
    std::fs::create_dir_all(input.join("empty")).unwrap();
    std::fs::write(input.join("notes.txt"), "not a folder").unwrap();
    let output = dir.path().join("out");

    let folders = batch_subfolders(&input).unwrap();
    let names: Vec<String> = folders
        .iter()
        .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["ant", "beetle"]);

    let entries = run_batch(
        &input,
        &output,
        &quiet_settings(),
        FuseMode::Apex,
        0,
        &Hooks::silent(),
    )
    .unwrap();
    assert_eq!(entries.len(), 2);
    for (entry, name) in entries.iter().zip(["ant", "beetle"]) {
        assert_eq!(entry.name, name);
        let path = entry.result.as_ref().unwrap();
        assert_eq!(path, &output.join(format!("{name}_stacked.tif")));
        assert!(path.exists());
    }
}

#[test]
fn a_failing_folder_does_not_stop_the_batch() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in");
    write_frames(&input.join("good"), 2);
    // A folder whose only "frame" is not an image.
    std::fs::create_dir_all(input.join("bad")).unwrap();
    std::fs::write(input.join("bad").join("broken.png"), b"not a png").unwrap();
    let entries: Vec<BatchEntry> = run_batch(
        &input,
        &dir.path().join("out"),
        &quiet_settings(),
        FuseMode::Apex,
        0,
        &Hooks::silent(),
    )
    .unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].name, "bad");
    assert!(entries[0].result.is_err());
    assert_eq!(entries[1].name, "good");
    assert!(entries[1].result.is_ok());
}

// ── one frame ───────────────────────────────────────────────────────────────

#[test]
fn read_rgb16_scales_an_8_bit_file_to_the_16_bit_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two.png");
    let img = RgbImage::from_fn(3, 2, |x, y| {
        Rgb([
            u8::try_from(x * 100).unwrap(),
            255,
            u8::try_from(y).unwrap(),
        ])
    });
    img.save(&path).unwrap();
    let (width, height, samples) = read_rgb16(&path).unwrap();
    assert_eq!((width, height), (3, 2));
    assert_eq!(samples.len(), 3 * 2 * 3);
    // Pixel (x = 2, y = 1): [200, 255, 1] scaled by 257.
    let at = (3 + 2) * 3;
    assert_eq!(&samples[at..at + 3], &[200 * 257, 65_535, 257]);
}

#[test]
fn read_rgb16_reports_a_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    assert!(read_rgb16(&dir.path().join("nope.png")).is_err());
}

/// A [`RunLog`] and [`Progress`] used only to prove the trait objects of the hooks are
/// satisfied by plain types from this crate's callers.
struct Quiet;

impl RunLog for Quiet {
    fn info(&self, _message: &str) {}
    fn warn(&self, _message: &str) {}
}

impl Progress for Quiet {
    fn stage(&self, _stage: Stage, _fraction: f32, _label: &str) {}
}

#[test]
fn hooks_accept_any_log_and_progress() {
    let quiet = Quiet;
    let hooks = Hooks {
        progress: &quiet,
        cancel: &stacker_pipeline::NeverCancel,
        log: &quiet,
    };
    let fused = stack_in_ram(
        vec![frame(16, 16, (0, 16))],
        &quiet_settings(),
        FuseMode::Apex,
        &hooks,
    )
    .unwrap();
    assert_eq!(fused.width, 16);
}
