//! `config`: the one settings loader.

use stacker_core::settings::{
    AlignmentModeSetting, ComputeDevice, DEFAULT_CONFIG_TOML, OptimizerSetting, OutputFormat,
    SETTINGS_VERSION, StackingSettings,
};
use stacker_pipeline::config::{Leniency, parse_settings};

/// Every one of the 39 keys, set to a non-default value.
const ALL_KEYS: &str = r#"
settings_version = 2
alignment_mode = "translation"
optimizer = "lucaskanade"
bounded_search = true
akaze_seeding = true
correct_brightness = false
chain_alignment = true
neural_refine_classically = false
crop_to_common_area = false
resize_cropped_to_original = true
auto_cull = false
auto_cull_threshold_pct = 4.0
sort_by_sharpness = true
stack_every_nth = 3
tile_size = 256
pyramid_levels = 6
use_all_color_channels = true
grit_suppression = false
relief_estimation_radius = 7
relief_smoothing_radius = 4
relief_contrast_pct = 0.25
relief_show_preview = true
relief_auto_detect = true
relief_use_multigrid = true
strata_base_radius = 24
strata_detail_focus = 4
compute_device = "gpu"

[preprocessing]
pre_rotation = 90
pre_crop_enabled = true
pre_crop_spec = "50,40"
pre_resize_percent = 60
sort_reverse = true
ignore_exif_orientation = true

[image_saving]
output_format = "jpeg"
bit_depth = 8
jpeg_quality = 30
filename_template = "{name}_x"
default_output_dir = "out"
copy_metadata = true
"#;

#[test]
fn all_39_keys_come_out_of_the_loader_unchanged_in_both_modes() {
    for leniency in [Leniency::Strict, Leniency::Lenient] {
        let loaded = parse_settings(ALL_KEYS, leniency).unwrap();
        assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
        let s = loaded.settings;
        assert_eq!(s.settings_version, SETTINGS_VERSION);
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Translation);
        assert_eq!(s.optimizer, OptimizerSetting::LucasKanade);
        assert!(s.bounded_search && s.akaze_seeding && !s.auto_cull);
        assert!(!s.correct_brightness && s.chain_alignment && !s.neural_refine_classically);
        assert!(!s.crop_to_common_area && s.resize_cropped_to_original);
        assert!((s.auto_cull_threshold_pct - 4.0).abs() < f32::EPSILON);
        assert!(s.sort_by_sharpness);
        assert_eq!(s.tile_size, 256);
        assert!(s.relief_use_multigrid);
        assert_eq!((s.strata_base_radius, s.strata_detail_focus), (24, 4));
        assert_eq!(s.stack_every_nth, 3);
        assert_eq!(s.pyramid_levels, 6);
        assert!(s.use_all_color_channels && !s.grit_suppression);
        assert_eq!(
            (s.relief_estimation_radius, s.relief_smoothing_radius),
            (7, 4)
        );
        assert!((s.relief_contrast_pct - 0.25).abs() < f32::EPSILON);
        assert!(s.relief_show_preview && s.relief_auto_detect);
        assert_eq!(s.compute_device, ComputeDevice::Gpu);
        assert_eq!(s.preprocessing.pre_rotation, 90);
        assert!(s.preprocessing.pre_crop_enabled);
        assert_eq!(s.preprocessing.pre_crop_spec, "50,40");
        assert_eq!(s.preprocessing.pre_resize_percent, 60);
        assert!(s.preprocessing.sort_reverse && s.preprocessing.ignore_exif_orientation);
        assert_eq!(s.image_saving.output_format, OutputFormat::Jpeg);
        assert_eq!(s.image_saving.bit_depth, 8);
        assert_eq!(s.image_saving.jpeg_quality, 30);
        assert_eq!(s.image_saving.filename_template, "{name}_x");
        assert_eq!(s.image_saving.default_output_dir, "out");
        assert!(s.image_saving.copy_metadata);
    }
}

#[test]
fn the_generated_default_file_parses_to_the_defaults() {
    for leniency in [Leniency::Strict, Leniency::Lenient] {
        let loaded = parse_settings(DEFAULT_CONFIG_TOML, leniency).unwrap();
        assert_eq!(loaded.settings, StackingSettings::default());
        assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
    }
}

#[test]
fn bounded_is_an_alias_for_registration_with_bounded_search() {
    for leniency in [Leniency::Strict, Leniency::Lenient] {
        let s = parse_settings("alignment_mode = \"Bounded\"", leniency)
            .unwrap()
            .settings;
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
        assert!(s.bounded_search);
    }
}

/// The legacy keys are accepted in both modes, silently, and the current name
/// wins when both are present.
#[test]
fn legacy_aliases_are_accepted_in_both_modes() {
    let text = "dmap_estimation_radius = 9\ndmap_smoothing_radius = 4\n\
                dmap_contrast_pct = 0.5\ndmap_show_preview = true\ndmap_auto_detect = true\n\
                use_gpu = true\nrelief_engine = \"multigrid\"\n";
    for leniency in [Leniency::Strict, Leniency::Lenient] {
        let loaded = parse_settings(text, leniency).unwrap();
        assert!(loaded.warnings.is_empty(), "{:?}", loaded.warnings);
        let s = loaded.settings;
        assert_eq!(s.relief_estimation_radius, 9);
        assert_eq!(s.relief_smoothing_radius, 4);
        assert!((s.relief_contrast_pct - 0.5).abs() < f32::EPSILON);
        assert!(s.relief_show_preview && s.relief_auto_detect);
        assert_eq!(s.compute_device, ComputeDevice::Gpu);
        assert!(s.relief_use_multigrid);
    }

    for leniency in [Leniency::Strict, Leniency::Lenient] {
        let s = parse_settings(
            "relief_estimation_radius = 11\ndmap_estimation_radius = 9\n\
             compute_device = \"cpu\"\nuse_gpu = true\n",
            leniency,
        )
        .unwrap()
        .settings;
        assert_eq!(s.relief_estimation_radius, 11);
        assert_eq!(s.compute_device, ComputeDevice::Cpu);
    }

    // A wrongly typed legacy value is still an error in strict mode.
    assert!(parse_settings("dmap_estimation_radius = \"wide\"", Leniency::Strict).is_err());
}

/// A file without `settings_version` is migrated in both modes; a versioned
/// one is read literally.
#[test]
fn legacy_alignment_modes_are_migrated_in_both_modes() {
    for leniency in [Leniency::Strict, Leniency::Lenient] {
        let load = |text: &str| parse_settings(text, leniency).unwrap().settings;

        let s = load("alignment_mode = \"affine\"");
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
        assert!(!s.bounded_search);

        let s = load("alignment_mode = \"registration\"");
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
        assert!(s.bounded_search);

        let s = load("settings_version = 2\nalignment_mode = \"affine\"");
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Affine);
        assert!(!s.bounded_search);
    }
}

#[test]
fn numbers_are_clamped_and_reported() {
    let loaded = parse_settings(
        "stack_every_nth = 0\npyramid_levels = 40\nrelief_contrast_pct = 3.0\n\
         [preprocessing]\npre_rotation = 45\npre_resize_percent = 3\n\
         [image_saving]\nbit_depth = 12\njpeg_quality = 500\n",
        Leniency::Strict,
    )
    .unwrap();
    let s = &loaded.settings;
    assert_eq!(s.stack_every_nth, 1);
    assert_eq!(s.pyramid_levels, 12);
    assert!((s.relief_contrast_pct - 1.0).abs() < f32::EPSILON);
    assert_eq!(s.preprocessing.pre_rotation, 0);
    assert_eq!(s.preprocessing.pre_resize_percent, 10);
    assert_eq!(s.image_saving.bit_depth, 16);
    assert_eq!(s.image_saving.jpeg_quality, 100);
    assert_eq!(loaded.warnings.len(), 7, "{:?}", loaded.warnings);
}

/// The two modes differ only in what a wrongly typed value does.
#[test]
fn strict_stops_on_a_bad_value_and_lenient_keeps_the_rest() {
    let text = "alignment_mode = \"sideways\"\npyramid_levels = 5\n";
    let err = parse_settings(text, Leniency::Strict).unwrap_err();
    assert!(err.contains("sideways"), "{err}");

    let lenient = parse_settings(text, Leniency::Lenient).unwrap();
    assert_eq!(
        lenient.settings.alignment_mode,
        AlignmentModeSetting::Registration
    );
    assert_eq!(
        lenient.settings.pyramid_levels, 5,
        "the rest of the file counts"
    );
    assert!(
        lenient
            .warnings
            .iter()
            .any(|w| w.contains("alignment_mode")),
        "{:?}",
        lenient.warnings
    );

    // Unknown keys are never fatal, in either mode.
    for leniency in [Leniency::Strict, Leniency::Lenient] {
        let loaded = parse_settings("pyramid_levl = 5\n", leniency).unwrap();
        assert!(loaded.warnings.iter().any(|w| w.contains("pyramid_levl")));
    }
    // Invalid TOML is an error in both.
    for leniency in [Leniency::Strict, Leniency::Lenient] {
        assert!(parse_settings("pyramid_levels = = 3", leniency).is_err());
    }
}
