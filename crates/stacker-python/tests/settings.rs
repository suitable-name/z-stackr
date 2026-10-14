//! The settings mirror and the option strings, tested without Python.

use stacker_core::settings::{
    AlignmentModeSetting, ComputeDevice, ImageSavingSettings, OptimizerSetting, OutputFormat,
    PreprocessingSettings, StackingSettings,
};
use stacker_python::{
    PyImageSavingSettings, PyPipelineParams, PyPreprocessingSettings, PyStackingSettings,
    options::{
        alignment_mode_options, extension_of, output_format_for_path, parse_alignment_mode,
        parse_compute_device, parse_optimizer, parse_output_format, parse_relief_engine,
        relief_engine_str,
    },
};

/// Every field set to a value that differs from the default (and is inside its range), so a
/// round trip detects a field that is dropped or mixed up.
fn non_default_core() -> StackingSettings {
    StackingSettings {
        settings_version: 2,
        alignment_mode: AlignmentModeSetting::Affine,
        optimizer: OptimizerSetting::LucasKanade,
        bounded_search: true,
        akaze_seeding: true,
        correct_brightness: false,
        chain_alignment: true,
        neural_refine_classically: false,
        crop_to_common_area: false,
        resize_cropped_to_original: true,
        auto_cull: false,
        auto_cull_threshold_pct: 3.5,
        sort_by_sharpness: true,
        stack_every_nth: 3,
        tile_size: 777,
        pyramid_levels: 5,
        use_all_color_channels: true,
        grit_suppression: false,
        relief_estimation_radius: 9,
        relief_smoothing_radius: 7,
        relief_contrast_pct: 0.5,
        relief_show_preview: true,
        relief_auto_detect: true,
        relief_use_multigrid: true,
        strata_base_radius: 40,
        strata_detail_focus: 5,
        compute_device: ComputeDevice::Gpu,
        preprocessing: PreprocessingSettings {
            pre_rotation: 90,
            pre_crop_enabled: true,
            pre_crop_spec: "10,20,300,400".to_owned(),
            pre_resize_percent: 50,
            sort_reverse: true,
            ignore_exif_orientation: true,
        },
        image_saving: ImageSavingSettings {
            output_format: OutputFormat::Jpeg,
            bit_depth: 8,
            jpeg_quality: 42,
            filename_template: "{name}_custom".to_owned(),
            default_output_dir: "out".to_owned(),
            copy_metadata: true,
        },
    }
}

#[test]
fn the_fixture_really_differs_from_the_default_in_every_key() {
    let (a, b) = (non_default_core(), StackingSettings::default());
    assert_ne!(a.alignment_mode, b.alignment_mode);
    assert_ne!(a.optimizer, b.optimizer);
    assert_ne!(a.bounded_search, b.bounded_search);
    assert_ne!(a.akaze_seeding, b.akaze_seeding);
    assert_ne!(a.correct_brightness, b.correct_brightness);
    assert_ne!(a.chain_alignment, b.chain_alignment);
    assert_ne!(a.neural_refine_classically, b.neural_refine_classically);
    assert_ne!(a.crop_to_common_area, b.crop_to_common_area);
    assert_ne!(a.resize_cropped_to_original, b.resize_cropped_to_original);
    assert_ne!(a.auto_cull, b.auto_cull);
    assert_ne!(a.sort_by_sharpness, b.sort_by_sharpness);
    assert_ne!(a.stack_every_nth, b.stack_every_nth);
    assert_ne!(a.tile_size, b.tile_size);
    assert_ne!(a.pyramid_levels, b.pyramid_levels);
    assert_ne!(a.use_all_color_channels, b.use_all_color_channels);
    assert_ne!(a.grit_suppression, b.grit_suppression);
    assert_ne!(a.relief_estimation_radius, b.relief_estimation_radius);
    assert_ne!(a.relief_smoothing_radius, b.relief_smoothing_radius);
    assert_ne!(a.relief_show_preview, b.relief_show_preview);
    assert_ne!(a.relief_auto_detect, b.relief_auto_detect);
    assert_ne!(a.relief_use_multigrid, b.relief_use_multigrid);
    assert_ne!(a.strata_base_radius, b.strata_base_radius);
    assert_ne!(a.strata_detail_focus, b.strata_detail_focus);
    assert_ne!(a.compute_device, b.compute_device);
    assert_ne!(a.preprocessing, b.preprocessing);
    assert_ne!(a.image_saving, b.image_saving);
}

#[test]
fn every_key_survives_core_to_python_to_core() {
    let core = non_default_core();
    let mirror = PyStackingSettings::from_core(&core);
    assert_eq!(mirror.into_core().unwrap(), core);
}

#[test]
fn the_default_mirror_is_the_default_core() {
    let mirror = PyStackingSettings::default();
    assert_eq!(
        mirror,
        PyStackingSettings::from_core(&StackingSettings::default())
    );
    assert_eq!(mirror.alignment_mode, "Registration");
    assert_eq!(mirror.optimizer, "Auto");
    assert_eq!(mirror.compute_device, "CPU");
    assert_eq!(mirror.image_saving.output_format, "TIFF");
    assert!(!mirror.relief_use_multigrid);
    assert_eq!(mirror.into_core().unwrap(), StackingSettings::default());
}

#[test]
fn the_nested_mirrors_round_trip() {
    let pre = non_default_core().preprocessing;
    assert_eq!(PyPreprocessingSettings::from_core(&pre).into_core(), pre);
    assert_eq!(
        PyPreprocessingSettings::default().into_core(),
        PreprocessingSettings::default()
    );
    let saving = non_default_core().image_saving;
    assert_eq!(
        PyImageSavingSettings::from_core(&saving)
            .into_core()
            .unwrap(),
        saving
    );
}

#[test]
fn the_mirror_agrees_with_the_settings_file() {
    // core -> mirror -> core -> TOML text -> core: the path `save_config` and `load_config` take.
    let core = non_default_core();
    let text =
        toml::to_string_pretty(&PyStackingSettings::from_core(&core).into_core().unwrap()).unwrap();
    let (loaded, warnings) = StackingSettings::from_toml_str(&text).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(loaded, core);
}

#[test]
fn the_default_config_text_loads_into_the_default_mirror() {
    let (loaded, warnings) =
        StackingSettings::from_toml_str(stacker_core::settings::DEFAULT_CONFIG_TOML).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        PyStackingSettings::from_core(&loaded),
        PyStackingSettings::default()
    );
}

#[test]
fn enum_strings_are_case_insensitive_and_accept_aliases() {
    let mirror = PyStackingSettings {
        alignment_mode: "AFFINE".to_owned(),
        optimizer: "lucas_kanade".to_owned(),
        compute_device: "gpu".to_owned(),
        image_saving: PyImageSavingSettings {
            output_format: "jpg".to_owned(),
            ..PyImageSavingSettings::default()
        },
        ..PyStackingSettings::default()
    };
    let core = mirror.into_core().unwrap();
    assert_eq!(core.alignment_mode, AlignmentModeSetting::Affine);
    assert_eq!(core.optimizer, OptimizerSetting::LucasKanade);
    assert_eq!(core.compute_device, ComputeDevice::Gpu);
    assert_eq!(core.image_saving.output_format, OutputFormat::Jpeg);
}

#[test]
fn the_legacy_bounded_label_means_registration_with_a_bounded_search() {
    let mirror = PyStackingSettings {
        alignment_mode: "Bounded".to_owned(),
        ..PyStackingSettings::default()
    };
    let core = mirror.into_core().unwrap();
    assert_eq!(core.alignment_mode, AlignmentModeSetting::Registration);
    assert!(core.bounded_search);
    assert_eq!(
        parse_alignment_mode("bounded").unwrap(),
        (AlignmentModeSetting::Registration, true)
    );
}

#[test]
fn every_label_of_the_enums_parses_back_to_itself() {
    for label in alignment_mode_options() {
        let (mode, _) = parse_alignment_mode(label).unwrap();
        assert_eq!(mode.as_combo_str(), label);
    }
    for mode in [
        OptimizerSetting::Auto,
        OptimizerSetting::LucasKanade,
        OptimizerSetting::NelderMead,
    ] {
        assert_eq!(parse_optimizer(mode.as_combo_str()).unwrap(), mode);
    }
    for device in [ComputeDevice::Cpu, ComputeDevice::Gpu] {
        assert_eq!(parse_compute_device(device.as_combo_str()).unwrap(), device);
    }
    for format in [OutputFormat::Tiff, OutputFormat::Png, OutputFormat::Jpeg] {
        assert_eq!(parse_output_format(format.as_combo_str()).unwrap(), format);
    }
    assert_eq!(parse_optimizer("nm").unwrap(), OptimizerSetting::NelderMead);
    assert_eq!(
        parse_optimizer("LK").unwrap(),
        OptimizerSetting::LucasKanade
    );
}

#[test]
fn an_unknown_option_lists_the_valid_ones() {
    let err = PyStackingSettings {
        alignment_mode: "Sideways".to_owned(),
        ..PyStackingSettings::default()
    }
    .into_core()
    .unwrap_err();
    let message = err.to_string();
    for needle in [
        "Sideways",
        "Translation",
        "Registration",
        "Affine",
        "None",
        "Neural",
    ] {
        assert!(message.contains(needle), "{message}");
    }

    let err = parse_optimizer("fast").unwrap_err();
    for needle in ["fast", "Auto", "Lucas-Kanade", "Nelder-Mead"] {
        assert!(err.message().contains(needle), "{err}");
    }
    let err = parse_compute_device("tpu").unwrap_err();
    assert!(err.message().contains("CPU") && err.message().contains("GPU"));
    let err = parse_output_format("bmp").unwrap_err();
    for needle in ["bmp", "TIFF", "PNG", "JPEG"] {
        assert!(err.message().contains(needle), "{err}");
    }
    let err = parse_relief_engine("gradient").unwrap_err();
    assert!(err.message().contains("guided_filter") && err.message().contains("multigrid"));
}

#[test]
fn numbers_outside_their_range_are_rejected_with_the_key_named() {
    fn rejected(key: &str, mutate: impl FnOnce(&mut PyStackingSettings)) {
        let mut mirror = PyStackingSettings::default();
        mutate(&mut mirror);
        let err = mirror.into_core().unwrap_err();
        assert!(err.message().contains(key), "{key}: {err}");
    }

    rejected("pyramid_levels", |s| s.pyramid_levels = 13);
    rejected("strata_detail_focus", |s| s.strata_detail_focus = 0);
    rejected("strata_detail_focus", |s| s.strata_detail_focus = 6);
    rejected("strata_base_radius", |s| s.strata_base_radius = 4);
    rejected("tile_size", |s| s.tile_size = 10);
    rejected("stack_every_nth", |s| s.stack_every_nth = 0);
    rejected("relief_contrast_pct", |s| s.relief_contrast_pct = 1.5);
    rejected("auto_cull_threshold_pct", |s| {
        s.auto_cull_threshold_pct = -1.0;
    });
    rejected("pre_resize_percent", |s| {
        s.preprocessing.pre_resize_percent = 5;
    });
    rejected("bit_depth", |s| s.image_saving.bit_depth = 12);
    rejected("jpeg_quality", |s| s.image_saving.jpeg_quality = 101);
    rejected("pre_rotation", |s| s.preprocessing.pre_rotation = 45);
}

#[test]
fn the_bounds_of_the_ranges_are_valid() {
    for focus in 1..=5 {
        let mirror = PyStackingSettings {
            strata_detail_focus: focus,
            ..PyStackingSettings::default()
        };
        assert!(mirror.into_core().is_ok(), "strata_detail_focus = {focus}");
    }
    for tile_size in [0, 64, 8192] {
        let mirror = PyStackingSettings {
            tile_size,
            ..PyStackingSettings::default()
        };
        assert!(mirror.into_core().is_ok(), "tile_size = {tile_size}");
    }
}

#[test]
fn use_gpu_and_relief_engine_are_views_of_the_real_keys() {
    let mut mirror = PyStackingSettings::default();
    assert!(!mirror.uses_gpu());
    mirror.compute_device = PyStackingSettings::device_label(true).to_owned();
    assert!(mirror.uses_gpu());
    assert_eq!(mirror.compute_device, "GPU");
    assert_eq!(PyStackingSettings::device_label(false), "CPU");

    assert!(!parse_relief_engine("guided_filter").unwrap());
    assert!(!parse_relief_engine("Guided-Filter").unwrap());
    assert!(parse_relief_engine("MULTIGRID").unwrap());
    assert_eq!(relief_engine_str(false), "guided_filter");
    assert_eq!(relief_engine_str(true), "multigrid");
}

#[test]
fn the_output_format_follows_the_file_extension() {
    use std::path::Path;
    for (name, format) in [
        ("a.tif", Some(OutputFormat::Tiff)),
        ("a.TIFF", Some(OutputFormat::Tiff)),
        ("dir/a.png", Some(OutputFormat::Png)),
        ("a.jpg", Some(OutputFormat::Jpeg)),
        ("a.JPEG", Some(OutputFormat::Jpeg)),
        ("a.bmp", None),
        ("a", None),
    ] {
        assert_eq!(output_format_for_path(Path::new(name)), format, "{name}");
    }
    assert_eq!(extension_of(OutputFormat::Tiff), "tif");
    assert_eq!(extension_of(OutputFormat::Png), "png");
    assert_eq!(extension_of(OutputFormat::Jpeg), "jpg");
}

#[test]
fn pipeline_params_keep_their_fields() {
    let params = PyPipelineParams::new(
        vec!["a.png".to_owned(), "b.png".to_owned()],
        "out.tif".to_owned(),
        "relief".to_owned(),
        256,
    );
    assert_eq!(params.paths.len(), 2);
    assert_eq!(params.output_file, "out.tif");
    assert_eq!(params.mode, "relief");
    assert_eq!(params.tile_size, 256);
}
