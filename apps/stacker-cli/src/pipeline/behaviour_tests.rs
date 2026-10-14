#![expect(
    clippy::suboptimal_flops,
    reason = "test fixtures and image metrics: readable formulas, not performance code"
)]

//! Behavioural tests of `run_pipeline_in`: validation order, scratch cleanup,
//! and the settings keys that shape the frame set and the output file.
//!
//! All runs use `AlignmentModeSetting::None` (no alignment cost) on tiny
//! synthetic frames; alignment itself is covered by the crate's other tests.
use super::*;
use image::{ColorType, GenericImageView, RgbImage};
use stacker_core::settings::{AlignmentModeSetting, OutputFormat};

use crate::args::{CliArgs, OptimizerArg};

/// Fresh, empty directory unique to this test and process.
fn test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stacker_cli_beh_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");
    dir
}

/// Deterministic noisy frame; `seed` varies the pattern between frames.
fn noisy_frame(w: u32, h: u32, seed: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let mut v = x
            .wrapping_mul(0x9E37_79B1)
            .wrapping_add(y.wrapping_mul(0x85EB_CA6B))
            .wrapping_add(seed.wrapping_mul(0xC2B2_AE35));
        v ^= v >> 15;
        v = v.wrapping_mul(0x2C1B_3C6D);
        v ^= v >> 12;
        let [b0, b1, b2, _] = v.to_le_bytes();
        image::Rgb([b0, b1, b2])
    })
}

fn write_stack(dir: &Path, n: u32, w: u32, h: u32) {
    for i in 0..n {
        noisy_frame(w, h, i)
            .save(dir.join(format!("frame_{i:02}.png")))
            .expect("save frame");
    }
}

fn args_for(input: &Path, output: &Path) -> CliArgs {
    CliArgs::for_test(input, output, "apex", 64)
}

/// Quick settings: no alignment, no cull.
fn fast_settings() -> StackingSettings {
    StackingSettings {
        alignment_mode: AlignmentModeSetting::None,
        auto_cull: false,
        ..StackingSettings::default()
    }
}

fn entries(dir: &Path) -> usize {
    std::fs::read_dir(dir).map_or(0, std::iter::Iterator::count)
}

#[test]
fn bad_output_extension_fails_before_any_frame_is_decoded() {
    let input = test_dir("badext_in");
    // An undecodable frame: reaching the decoder would give a "cannot decode" error.
    std::fs::write(input.join("frame_a.png"), b"definitely not a png").unwrap();
    std::fs::write(input.join("frame_b.png"), b"nor is this").unwrap();
    let scratch = test_dir("badext_scratch");

    let args = args_for(&input, &input.join("result.xyz"));
    let err = run_pipeline_in(&args, &fast_settings(), &scratch)
        .unwrap_err()
        .to_string();

    assert!(err.contains("unsupported output extension"), "{err}");
    assert!(
        !err.contains("decode"),
        "decoder must not have been reached: {err}"
    );
    assert_eq!(entries(&scratch), 0, "no scratch directory may be created");

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&scratch);
}

#[test]
fn output_below_a_regular_file_fails_before_any_frame_is_decoded() {
    let input = test_dir("fileparent_in");
    std::fs::write(input.join("frame_a.png"), b"not a png").unwrap();
    let blocker = input.join("blocker.txt");
    std::fs::write(&blocker, b"x").unwrap();
    let scratch = test_dir("fileparent_scratch");

    let args = args_for(&input, &blocker.join("result.png"));
    let err = run_pipeline_in(&args, &fast_settings(), &scratch)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("not a directory") || err.contains("cannot create"),
        "{err}"
    );
    assert!(!err.contains("decode"), "{err}");
    assert_eq!(entries(&scratch), 0);

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&scratch);
}

#[test]
fn tile_size_below_the_minimum_is_rejected() {
    let input = test_dir("tile0_in");
    write_stack(&input, 2, 32, 32);
    let scratch = test_dir("tile0_scratch");

    let mut args = args_for(&input, &input.join("out").join("result.png"));
    args.tile_size = 0;
    let err = run_pipeline_in(&args, &fast_settings(), &scratch)
        .unwrap_err()
        .to_string();
    assert!(err.contains("tile size"), "{err}");
    assert_eq!(entries(&scratch), 0);

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// The second frame is 10 px narrower: the run must fail naming that file, and
/// the tile directory — already created and holding the reference frame's tiles
/// at that point — must be gone afterwards.
#[test]
fn mismatched_frame_size_is_an_error_and_the_tile_directory_is_removed() {
    let input = test_dir("mismatch_in");
    noisy_frame(64, 48, 0)
        .save(input.join("frame_00.png"))
        .unwrap();
    noisy_frame(54, 48, 1)
        .save(input.join("frame_01.png"))
        .unwrap();
    let output = test_dir("mismatch_out").join("result.png");
    let scratch = test_dir("mismatch_scratch");

    let err = run_pipeline_in(&args_for(&input, &output), &fast_settings(), &scratch)
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("frame_01.png"),
        "error must name the file: {err}"
    );
    assert!(err.contains("54x48") && err.contains("64x48"), "{err}");
    assert_eq!(
        entries(&scratch),
        0,
        "the stacker_tiles_* directory must be removed after the error"
    );
    assert!(!output.exists(), "no output may be written");

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&scratch);
}

#[test]
fn tile_directory_is_removed_after_a_successful_run_too() {
    let input = test_dir("ok_in");
    write_stack(&input, 2, 40, 40);
    let output = test_dir("ok_out").join("result.png");
    let scratch = test_dir("ok_scratch");

    run_pipeline_in(&args_for(&input, &output), &fast_settings(), &scratch).expect("run");
    assert!(output.exists());
    assert_eq!(entries(&scratch), 0);

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(output.parent().unwrap());
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `preprocessing.*` reaches the frames: rotate 90 deg then resize to 50 %
/// turns 64x32 inputs into 16x32 outputs.
#[test]
fn preprocessing_keys_shape_the_output() {
    let input = test_dir("pre_in");
    write_stack(&input, 2, 64, 32);
    let out_dir = test_dir("pre_out");
    let scratch = test_dir("pre_scratch");

    let mut settings = fast_settings();
    settings.preprocessing.pre_rotation = 90;
    settings.preprocessing.pre_resize_percent = 50;

    let output = out_dir.join("result.png");
    run_pipeline_in(&args_for(&input, &output), &settings, &scratch).expect("run");
    assert_eq!(image::open(&output).unwrap().dimensions(), (16, 32));

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `image_saving.bit_depth` selects 8 or 16 bits for PNG output.
#[test]
fn bit_depth_key_selects_the_png_sample_size() {
    let input = test_dir("depth_in");
    write_stack(&input, 2, 40, 40);
    let out_dir = test_dir("depth_out");
    let scratch = test_dir("depth_scratch");

    let mut settings = fast_settings();
    let out16 = out_dir.join("d16.png");
    run_pipeline_in(&args_for(&input, &out16), &settings, &scratch).expect("16-bit run");
    assert_eq!(image::open(&out16).unwrap().color(), ColorType::Rgb16);

    settings.image_saving.bit_depth = 8;
    let out8 = out_dir.join("d8.png");
    run_pipeline_in(&args_for(&input, &out8), &settings, &scratch).expect("8-bit run");
    assert_eq!(image::open(&out8).unwrap().color(), ColorType::Rgb8);

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `image_saving.jpeg_quality` reaches the encoder: quality 30 yields a smaller
/// file than quality 95 for the same stack (both used to be written at 75).
#[test]
fn jpeg_quality_key_changes_the_file_size() {
    let input = test_dir("jpeg_in");
    write_stack(&input, 2, 64, 64);
    let out_dir = test_dir("jpeg_out");
    let scratch = test_dir("jpeg_scratch");

    let mut sizes = Vec::new();
    for quality in [30_u32, 95] {
        let mut settings = fast_settings();
        settings.image_saving.jpeg_quality = quality;
        let output = out_dir.join(format!("q{quality}.jpg"));
        run_pipeline_in(&args_for(&input, &output), &settings, &scratch).expect("jpeg run");
        sizes.push(std::fs::metadata(&output).unwrap().len());
    }
    assert!(
        sizes[0] < sizes[1],
        "quality 30 ({}) must be smaller than quality 95 ({})",
        sizes[0],
        sizes[1]
    );

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// The previous result in the input directory is not picked up as a frame:
/// two runs in a row over the same directory see the same two frames.
#[test]
fn output_inside_the_input_directory_is_not_stacked_again() {
    let input = test_dir("self_in");
    write_stack(&input, 2, 40, 40);
    let output = input.join("stacked.png");
    let scratch = test_dir("self_scratch");

    let args = args_for(&input, &output);
    run_pipeline_in(&args, &fast_settings(), &scratch).expect("first run");
    assert!(output.exists());
    let prepared = prepare_run(&args, &fast_settings()).expect("second prepare");
    assert_eq!(
        prepared.paths.len(),
        2,
        "stacked.png must not count as a frame"
    );

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// Auto-cull on a stack where every frame is the same image keeps working
/// (ties go to the first frame; the rest fall under the threshold) and still
/// produces an output.
#[test]
fn auto_cull_runs_end_to_end() {
    let input = test_dir("cull_in");
    let frame = noisy_frame(64, 64, 7);
    for i in 0..3 {
        frame.save(input.join(format!("f{i}.png"))).unwrap();
    }
    let out_dir = test_dir("cull_out");
    let scratch = test_dir("cull_scratch");

    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::None,
        auto_cull: true,
        ..StackingSettings::default()
    };
    let output = out_dir.join("result.png");
    run_pipeline_in(&args_for(&input, &output), &settings, &scratch).expect("run");
    assert!(output.exists());
    assert_eq!(entries(&scratch), 0);

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&scratch);
}

// ── Tiled output equals the single-tile output (TST-2, CORE-3, FUS-4) ─────────

/// Run the pipeline over `frames` and return the 8-bit output samples.
fn run_to_samples(
    frames: &[RgbImage],
    settings: &StackingSettings,
    mode: &str,
    tile_size: usize,
    tag: &str,
) -> (Vec<u8>, (u32, u32)) {
    let input = test_dir(&format!("{tag}_in"));
    for (i, frame) in frames.iter().enumerate() {
        frame.save(input.join(format!("f{i}.png"))).unwrap();
    }
    let out_dir = test_dir(&format!("{tag}_out"));
    let scratch = test_dir(&format!("{tag}_scratch"));
    let output = out_dir.join("result.png");
    let mut args = args_for(&input, &output);
    args.mode = mode.to_owned();
    args.tile_size = tile_size;
    run_pipeline_in(&args, settings, &scratch).expect("run");
    let img = image::open(&output).unwrap();
    let dims = img.dimensions();
    let samples = img.to_rgb8().into_raw();
    assert_eq!(entries(&scratch), 0);
    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&scratch);
    (samples, dims)
}

fn eight_bit() -> StackingSettings {
    let mut settings = fast_settings();
    settings.image_saving.bit_depth = 8;
    settings
}

fn max_sample_difference(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// `Apex` with 64-px tiles (a 32-px apron derived from 4 pyramid levels) on a
/// 192x192 stack: the interior tiles are fused from strict sub-regions, yet the
/// file equals the single-tile run.
#[test]
fn apex_output_does_not_depend_on_the_tile_size() {
    let frames: Vec<RgbImage> = (0..3).map(|i| noisy_frame(192, 192, i)).collect();
    let mut settings = eight_bit();
    settings.pyramid_levels = 4;

    let (tiled, dims_tiled) = run_to_samples(&frames, &settings, "apex", 64, "apex_tiled");
    let (single, dims_single) = run_to_samples(&frames, &settings, "apex", 256, "apex_single");
    assert_eq!(dims_tiled, (192, 192));
    assert_eq!(dims_tiled, dims_single);
    let worst = max_sample_difference(&tiled, &single);
    assert!(
        worst <= 1,
        "tiled Apex differs from the single-tile run by {worst} levels"
    );
}

/// `Relief` with a non-zero contrast threshold: the threshold is computed once
/// over the whole image, so a tiled run equals the single-tile run.
#[test]
fn relief_output_does_not_depend_on_the_tile_size() {
    let frames: Vec<RgbImage> = (0..3).map(|i| noisy_frame(160, 160, 10 + i)).collect();
    for (pct, auto) in [(0.3_f32, false), (0.0, true)] {
        let mut settings = eight_bit();
        settings.relief_contrast_pct = pct;
        settings.relief_auto_detect = auto;

        let (tiled, _) = run_to_samples(&frames, &settings, "relief", 64, "relief_tiled");
        let (single, _) = run_to_samples(&frames, &settings, "relief", 256, "relief_single");
        let worst = max_sample_difference(&tiled, &single);
        assert!(
            worst <= 1,
            "tiled Relief (pct {pct}, auto {auto}) differs from the single-tile run by {worst} levels"
        );
    }
}

/// A smooth texture, moved by a whole number of pixels per frame.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a rounded value in 0..=255 becomes a byte"
)]
fn shifted_frames() -> [RgbImage; 3] {
    let frame = |dx: f32, dy: f32| {
        RgbImage::from_fn(128, 128, |x, y| {
            let (fx, fy) = (
                f32::from(u8::try_from(x).unwrap()) - dx,
                f32::from(u8::try_from(y).unwrap()) - dy,
            );
            let v = 0.5
                + 0.2 * (0.23 * fx).sin() * (0.19 * fy).cos()
                + 0.15 * (0.07 * (fx + fy)).sin()
                + 0.1 * (0.31 * fx - 0.27 * fy).cos();
            let byte = u8::try_from((v.clamp(0.0, 1.0) * 255.0).round() as u32).unwrap();
            image::Rgb([byte, byte, byte])
        })
    };
    [frame(0.0, 0.0), frame(3.0, 2.0), frame(-2.0, 3.0)]
}

/// A real alignment run crops the output to the area every frame covers (the
/// crop branch of the pipeline), and the shifted content is brought back.
#[test]
fn alignment_crops_the_output_to_the_common_area() {
    let frames = shifted_frames();
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        auto_cull: false,
        ..eight_bit()
    };
    let (_, (w, h)) = run_to_samples(&frames, &settings, "apex", 128, "crop");
    assert!(
        w < 128 && h < 128,
        "{w}x{h}: the uncovered border is cropped away"
    );
    assert!(
        w * 4 >= 128 * 3 && h * 4 >= 128 * 3,
        "{w}x{h}: only a few pixels are lost"
    );
}

/// `resize_cropped_to_original = true` resamples the cropped result back to the
/// input frame size.
#[test]
fn resize_cropped_to_original_restores_the_frame_size() {
    let frames = shifted_frames();
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        auto_cull: false,
        resize_cropped_to_original: true,
        ..eight_bit()
    };
    let (_, dims) = run_to_samples(&frames, &settings, "apex", 128, "resize");
    assert_eq!(dims, (128, 128));
}

/// `crop_to_common_area = false` keeps the full frame.
#[test]
fn alignment_keeps_the_full_frame_without_the_crop() {
    let frames = shifted_frames();
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        auto_cull: false,
        crop_to_common_area: false,
        ..eight_bit()
    };
    let (_, dims) = run_to_samples(&frames, &settings, "apex", 128, "nocrop");
    assert_eq!(dims, (128, 128));
}

/// `--optimizer` reaches the alignment: every optimizer runs the shifted frames
/// through to a result that is at most the frame and loses only a few pixels.
#[test]
fn the_optimizer_flag_runs_every_optimizer() {
    let frames = shifted_frames();
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        auto_cull: false,
        ..eight_bit()
    };
    for (index, optimizer) in [OptimizerArg::Auto, OptimizerArg::Lk, OptimizerArg::Nm]
        .into_iter()
        .enumerate()
    {
        let input = test_dir(&format!("opt{index}_in"));
        for (i, frame) in frames.iter().enumerate() {
            frame.save(input.join(format!("f{i}.png"))).unwrap();
        }
        let out_dir = test_dir(&format!("opt{index}_out"));
        let scratch = test_dir(&format!("opt{index}_scratch"));
        let output = out_dir.join("result.png");
        let mut args = CliArgs::for_test(&input, &output, "apex", 128);
        args.optimizer = Some(optimizer);
        run_pipeline_in(&args, &settings, &scratch).expect("run");
        let (w, h) = image::open(&output).unwrap().dimensions();
        assert!(
            w <= 128 && h <= 128 && w * 4 >= 128 * 3 && h * 4 >= 128 * 3,
            "{optimizer:?}: {w}x{h}"
        );
        let _ = std::fs::remove_dir_all(&input);
        let _ = std::fs::remove_dir_all(&out_dir);
        let _ = std::fs::remove_dir_all(&scratch);
    }
}

// ── Algorithms: the new names, the old names, several at once ─────────────────

/// The samples of the PNG at `path`, 8 bits.
fn samples_of(path: &Path) -> (Vec<u8>, (u32, u32)) {
    let img = image::open(path).unwrap();
    (img.to_rgb8().into_raw(), img.dimensions())
}

#[test]
fn the_old_mode_names_give_the_same_files_as_the_new_ones() {
    let frames: Vec<RgbImage> = (0..3).map(|i| noisy_frame(64, 64, 20 + i)).collect();
    let settings = eight_bit();
    for (old, new) in [("pmax", "apex"), ("DMap", "relief")] {
        let (from_old, dims_old) = run_to_samples(&frames, &settings, old, 64, "alias_old");
        let (from_new, dims_new) = run_to_samples(&frames, &settings, new, 64, "alias_new");
        assert_eq!(dims_old, dims_new);
        assert_eq!(from_old, from_new, "--mode {old} must equal --mode {new}");
    }
}

#[test]
fn strata_mode_writes_a_stack() {
    let frames: Vec<RgbImage> = (0..3).map(|i| noisy_frame(64, 64, 30 + i)).collect();
    let (samples, dims) = run_to_samples(&frames, &eight_bit(), "Strata", 64, "strata");
    assert_eq!(dims, (64, 64));
    assert_eq!(samples.len(), 64 * 64 * 3);
}

/// `both` and `all` align once and write one file per algorithm, named
/// `<name>_<algorithm>.<ext>`; the file of an algorithm equals the single-mode
/// result (one tile here, so the larger shared apron changes nothing).
#[test]
fn several_algorithms_write_one_suffixed_file_each() {
    let frames: Vec<RgbImage> = (0..3).map(|i| noisy_frame(64, 64, 40 + i)).collect();
    let settings = eight_bit();
    for (mode, expected) in [
        ("both", &["relief", "apex"][..]),
        ("ALL", &["strata", "relief", "apex"][..]),
    ] {
        let input = test_dir(&format!("multi_{mode}_in"));
        for (i, frame) in frames.iter().enumerate() {
            frame.save(input.join(format!("f{i}.png"))).unwrap();
        }
        let out_dir = test_dir(&format!("multi_{mode}_out"));
        let scratch = test_dir(&format!("multi_{mode}_scratch"));
        let output = out_dir.join("result.png");
        let mut args = args_for(&input, &output);
        args.mode = mode.to_owned();

        let written = run_stack_in(&args, &settings, &scratch).expect("run");
        let names: Vec<String> = written
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        let want: Vec<String> = expected
            .iter()
            .map(|label| format!("result_{label}.png"))
            .collect();
        assert_eq!(names, want, "--mode {mode}");
        assert!(!output.exists(), "no file is written under the plain name");
        assert_eq!(entries(&out_dir), expected.len());
        assert_eq!(entries(&scratch), 0);

        for (label, path) in expected.iter().zip(&written) {
            let (single, dims_single) = run_to_samples(&frames, &settings, label, 64, "single");
            let (multi, dims_multi) = samples_of(path);
            assert_eq!(dims_single, dims_multi, "{label}");
            assert_eq!(
                single, multi,
                "{label} of --mode {mode} equals --mode {label}"
            );
        }
        let _ = std::fs::remove_dir_all(&input);
        let _ = std::fs::remove_dir_all(&out_dir);
        let _ = std::fs::remove_dir_all(&scratch);
    }
}

/// Results of a multi-algorithm run that sit in the input folder are not frames
/// of the next run.
#[test]
fn per_algorithm_outputs_in_the_input_directory_are_not_stacked_again() {
    let input = test_dir("multi_self_in");
    write_stack(&input, 2, 40, 40);
    let scratch = test_dir("multi_self_scratch");
    let mut args = args_for(&input, &input.join("stacked.png"));
    args.mode = "both".to_owned();

    run_pipeline_in(&args, &eight_bit(), &scratch).expect("first run");
    assert!(input.join("stacked_relief.png").exists());
    assert!(input.join("stacked_apex.png").exists());
    let prepared = prepare_run(&args, &eight_bit()).expect("second prepare");
    assert_eq!(prepared.paths.len(), 2, "the results are not frames");

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&scratch);
}

// ── Metadata ─────────────────────────────────────────────────────────────────

/// A bare little-endian TIFF block with `Make = "Test"` and `Orientation = 6`.
const EXIF_BLOCK: [u8; 43] = [
    0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // header, IFD0 at 8
    0x02, 0x00, // two entries
    0x0F, 0x01, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00, // Make
    0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, // Orientation
    0x00, 0x00, 0x00, 0x00, // no next IFD
    b'T', b'e', b's', b't', 0x00,
];

/// `image_saving.copy_metadata` copies the first frame's EXIF into a PNG or
/// JPEG result, with the orientation turned to 1 because the loader rotated the
/// pixels; a TIFF result cannot carry it and the run still succeeds.
#[test]
fn copy_metadata_carries_the_exif_of_the_first_frame() {
    use stacker_core::metadata::{exif_orientation, extract_exif, inject_exif};

    let input = test_dir("exif_in");
    for i in 0..2_u32 {
        let path = input.join(format!("frame_{i}.jpg"));
        noisy_frame(48, 32, i).save(&path).unwrap();
        inject_exif(&path, &EXIF_BLOCK).expect("inject the fixture block");
    }
    assert_eq!(
        extract_exif(&input.join("frame_0.jpg")).and_then(|b| exif_orientation(&b)),
        Some(6),
        "the fixture block reads back"
    );
    let out_dir = test_dir("exif_out");
    let scratch = test_dir("exif_scratch");

    let mut settings = eight_bit();
    settings.image_saving.copy_metadata = true;

    for name in ["with.png", "with.jpg"] {
        let output = out_dir.join(name);
        run_pipeline_in(&args_for(&input, &output), &settings, &scratch).expect("run");
        let blob = extract_exif(&output).unwrap_or_else(|| panic!("{name} carries the EXIF"));
        assert!(
            blob.windows(5).any(|w| w == b"Test\0"),
            "{name}: the Make survives"
        );
        assert_eq!(
            exif_orientation(&blob),
            Some(1),
            "{name}: pixels are upright"
        );
    }

    settings.image_saving.output_format = OutputFormat::Tiff;
    let tiff = out_dir.join("without.tif");
    run_pipeline_in(&args_for(&input, &tiff), &settings, &scratch).expect("TIFF run");
    assert!(tiff.exists());

    settings.image_saving.copy_metadata = false;
    let plain = out_dir.join("plain.png");
    run_pipeline_in(&args_for(&input, &plain), &settings, &scratch).expect("run");
    assert!(extract_exif(&plain).is_none(), "copy_metadata off: no EXIF");

    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&scratch);
}

// ── Neural alignment without the nn feature ──────────────────────────────────

#[cfg(not(feature = "nn"))]
#[test]
fn neural_alignment_is_refused_before_any_work_without_the_nn_feature() {
    let input = test_dir("neural_in");
    write_stack(&input, 2, 32, 32);
    let scratch = test_dir("neural_scratch");
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Neural,
        ..fast_settings()
    };
    let err = run_pipeline_in(
        &args_for(&input, &input.join("out").join("r.png")),
        &settings,
        &scratch,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("neural") && err.contains("nn"), "{err}");
    assert_eq!(entries(&scratch), 0);
    let _ = std::fs::remove_dir_all(&input);
    let _ = std::fs::remove_dir_all(&scratch);
}
