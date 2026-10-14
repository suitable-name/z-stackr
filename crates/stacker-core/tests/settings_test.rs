use stacker_core::settings::*;

#[test]
fn test_alignment_mode_setting_combo_str() {
    assert_eq!(AlignmentModeSetting::Affine.as_combo_str(), "Affine");
    assert_eq!(
        AlignmentModeSetting::Translation.as_combo_str(),
        "Translation"
    );
    assert_eq!(AlignmentModeSetting::None.as_combo_str(), "None");

    assert_eq!(
        AlignmentModeSetting::from_combo_str("Affine"),
        AlignmentModeSetting::Affine
    );
    assert_eq!(
        AlignmentModeSetting::from_combo_str("Translation"),
        AlignmentModeSetting::Translation
    );
    assert_eq!(
        AlignmentModeSetting::from_combo_str("None"),
        AlignmentModeSetting::None
    );
    assert_eq!(
        AlignmentModeSetting::from_combo_str("Registration"),
        AlignmentModeSetting::Registration
    );
    assert_eq!(
        AlignmentModeSetting::from_combo_str("Neural"),
        AlignmentModeSetting::Neural
    );
    assert_eq!(
        AlignmentModeSetting::Registration.as_combo_str(),
        "Registration"
    );
    assert_eq!(AlignmentModeSetting::Neural.as_combo_str(), "Neural");
    assert_eq!(
        AlignmentModeSetting::from_combo_str("Unknown"),
        AlignmentModeSetting::Registration
    ); // default
    assert_eq!(
        AlignmentModeSetting::default(),
        AlignmentModeSetting::Registration
    );
}

#[test]
fn test_alignment_serde_round_trip() {
    // Alignment mode plus the bounded-search / AKAZE-seeding toggles must
    // round-trip through TOML. "translation" is the lowercase serde form
    // (serde rename_all = "lowercase").
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        bounded_search: true,
        akaze_seeding: true,
        ..Default::default()
    };
    let toml_str = toml::to_string(&settings).expect("StackingSettings must serialize");
    assert!(
        toml_str.contains("alignment_mode = \"translation\""),
        "TOML must contain lowercase 'translation'; got: {toml_str}"
    );
    assert!(
        toml_str.contains("bounded_search = true"),
        "TOML must contain bounded_search; got: {toml_str}"
    );
    assert!(
        toml_str.contains("akaze_seeding = true"),
        "TOML must contain akaze_seeding; got: {toml_str}"
    );
    let back: StackingSettings =
        toml::from_str(&toml_str).expect("StackingSettings must deserialize");
    assert_eq!(back.alignment_mode, AlignmentModeSetting::Translation);
    assert!(back.bounded_search);
    assert!(back.akaze_seeding);
}

/// CORE-4: `sanitized` clamps every documented numeric range and reports
/// exactly one warning per clamped field.
#[test]
fn test_sanitized_clamps_and_reports() {
    let mut s = StackingSettings {
        stack_every_nth: 0,
        pyramid_levels: 1,
        relief_estimation_radius: 500,
        relief_smoothing_radius: 99,
        relief_contrast_pct: 2.0,
        ..Default::default()
    };
    s.preprocessing.pre_rotation = 45;
    s.preprocessing.pre_resize_percent = 5;
    s.image_saving.bit_depth = 10;
    s.image_saving.jpeg_quality = 0;

    let (clean, warnings) = s.sanitized();
    assert_eq!(clean.stack_every_nth, 1);
    assert_eq!(clean.pyramid_levels, 2);
    assert_eq!(clean.relief_estimation_radius, 100);
    assert_eq!(clean.relief_smoothing_radius, 50);
    assert!((clean.relief_contrast_pct - 1.0).abs() < f32::EPSILON);
    assert_eq!(clean.preprocessing.pre_rotation, 0);
    assert_eq!(clean.preprocessing.pre_resize_percent, 10);
    assert_eq!(clean.image_saving.bit_depth, 16);
    assert_eq!(clean.image_saving.jpeg_quality, 1);
    assert_eq!(
        warnings.len(),
        9,
        "one warning per clamped field: {warnings:?}"
    );
    for name in [
        "stack_every_nth",
        "pyramid_levels",
        "relief_estimation_radius",
        "relief_smoothing_radius",
        "relief_contrast_pct",
        "pre_rotation",
        "pre_resize_percent",
        "bit_depth",
        "jpeg_quality",
    ] {
        assert!(
            warnings.iter().any(|w| w.contains(name)),
            "no warning names {name}: {warnings:?}"
        );
    }

    // Upper bound of pyramid_levels, NaN contrast, and an in-range value stay quiet.
    let hi = StackingSettings {
        pyramid_levels: 13,
        relief_contrast_pct: f32::NAN,
        ..Default::default()
    };
    let (clean, warnings) = hi.sanitized();
    assert_eq!(clean.pyramid_levels, 12);
    assert!(clean.relief_contrast_pct.abs() < f32::EPSILON);
    assert_eq!(warnings.len(), 2, "{warnings:?}");

    let (same, warnings) = StackingSettings::default().sanitized();
    assert_eq!(same, StackingSettings::default());
    assert!(
        warnings.is_empty(),
        "defaults need no clamping: {warnings:?}"
    );
}

/// CORE-4: the shipped template parses cleanly to the defaults.
#[test]
fn test_default_template_loads_without_warnings() {
    let (s, warnings) =
        StackingSettings::from_toml_str(DEFAULT_CONFIG_TOML).expect("template must parse");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(s, StackingSettings::default());
}

/// CORE-4: `pyramid_levels = 1` in a file is clamped to 2 and reported.
#[test]
fn test_loader_clamps_out_of_range_value() {
    let text = DEFAULT_CONFIG_TOML.replace("pyramid_levels = 8", "pyramid_levels = 1");
    let (s, warnings) = StackingSettings::from_toml_str(&text).expect("must parse");
    assert_eq!(s.pyramid_levels, 2);
    assert!(
        warnings.iter().any(|w| w.contains("pyramid_levels")),
        "{warnings:?}"
    );
}

/// CORE-4: one wrong-typed value, one bad enum string and one unknown key
/// must not discard the rest of the file.
#[test]
fn test_loader_keeps_good_keys_around_bad_ones() {
    let text = r#"
alignment_mode = "Translation"
pyramid_levels = "eight"
pyramid_levls = 8
auto_cull = false
stack_every_nth = 3
relief_contrast_pct = 1

[preprocessing]
pre_rotation = 90
mystery = true

[image_saving]
output_format = "PNG"
jpeg_quality = "high"
bit_depth = 8
"#;
    let (s, warnings) = StackingSettings::from_toml_str(text).expect("syntax is valid");

    // Good keys survive, including a capitalised enum string.
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Translation);
    assert!(!s.auto_cull);
    assert_eq!(s.stack_every_nth, 3);
    assert!((s.relief_contrast_pct - 1.0).abs() < f32::EPSILON);
    assert_eq!(s.preprocessing.pre_rotation, 90);
    assert_eq!(s.image_saving.output_format, OutputFormat::Png);
    assert_eq!(s.image_saving.bit_depth, 8);

    // Bad keys fall back to their defaults.
    let d = StackingSettings::default();
    assert_eq!(s.pyramid_levels, d.pyramid_levels);
    assert_eq!(s.image_saving.jpeg_quality, d.image_saving.jpeg_quality);

    // Each problem is reported by name.
    for name in [
        "pyramid_levels:",
        "pyramid_levls",
        "preprocessing.mystery",
        "image_saving.jpeg_quality",
    ] {
        assert!(
            warnings.iter().any(|w| w.contains(name)),
            "no warning names {name}: {warnings:?}"
        );
    }
    assert_eq!(warnings.len(), 4, "{warnings:?}");
}

/// CORE-4: only a TOML syntax error fails the whole load.
#[test]
fn test_loader_rejects_only_syntax_errors() {
    assert!(StackingSettings::from_toml_str("pyramid_levels = = 3").is_err());
    assert!(StackingSettings::from_toml_str("").is_ok());
    // A scalar where a table is expected keeps the defaults for that section.
    let (s, warnings) = StackingSettings::from_toml_str("preprocessing = 5").expect("valid TOML");
    assert_eq!(s.preprocessing, PreprocessingSettings::default());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
}

/// CORE-4: enum strings are case-insensitive through plain serde too, and the
/// combo-box spelling round-trips.
#[test]
fn test_enums_deserialize_case_insensitively() {
    let s: StackingSettings =
        toml::from_str("alignment_mode = \"AFFINE\"\n[image_saving]\noutput_format = \"Jpeg\"")
            .expect("capitalised enums must parse");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Affine);
    assert_eq!(s.image_saving.output_format, OutputFormat::Jpeg);
    for mode in [
        AlignmentModeSetting::Affine,
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Registration,
        AlignmentModeSetting::None,
        AlignmentModeSetting::Neural,
    ] {
        let text = format!("alignment_mode = \"{}\"", mode.as_combo_str());
        let s: StackingSettings = toml::from_str(&text).expect("combo string must parse");
        assert_eq!(s.alignment_mode, mode);
    }
    assert!(toml::from_str::<StackingSettings>("alignment_mode = \"sideways\"").is_err());
}

#[test]
fn test_output_format_combo_str() {
    assert_eq!(OutputFormat::Tiff.as_combo_str(), "TIFF");
    assert_eq!(OutputFormat::Png.as_combo_str(), "PNG");
    assert_eq!(OutputFormat::Jpeg.as_combo_str(), "JPEG");

    assert_eq!(OutputFormat::from_combo_str("TIFF"), OutputFormat::Tiff);
    assert_eq!(OutputFormat::from_combo_str("PNG"), OutputFormat::Png);
    assert_eq!(OutputFormat::from_combo_str("JPEG"), OutputFormat::Jpeg);
    assert_eq!(OutputFormat::from_combo_str("Unknown"), OutputFormat::Tiff); // default
}

#[test]
fn test_preprocessing_settings_clamp() {
    let mut settings = PreprocessingSettings {
        pre_rotation: 45,      // invalid
        pre_resize_percent: 5, // below min
        ..Default::default()
    };

    settings.clamp_valid();

    assert_eq!(settings.pre_rotation, 0); // clamped to valid
    assert_eq!(settings.pre_resize_percent, 10); // clamped to min

    let mut settings = PreprocessingSettings {
        pre_rotation: 90,        // valid
        pre_resize_percent: 150, // above max
        ..Default::default()
    };

    settings.clamp_valid();

    assert_eq!(settings.pre_rotation, 90);
    assert_eq!(settings.pre_resize_percent, 100); // clamped to max
}

#[test]
fn test_image_saving_settings_clamp() {
    let mut settings = ImageSavingSettings {
        bit_depth: 10,   // invalid
        jpeg_quality: 0, // below min
        ..Default::default()
    };

    settings.clamp_valid();

    assert_eq!(settings.bit_depth, 16); // clamped to valid
    assert_eq!(settings.jpeg_quality, 1); // clamped to min

    let mut settings = ImageSavingSettings {
        bit_depth: 8,      // valid
        jpeg_quality: 150, // above max
        ..Default::default()
    };

    settings.clamp_valid();

    assert_eq!(settings.bit_depth, 8);
    assert_eq!(settings.jpeg_quality, 100); // clamped to max
}

/// The compute device defaults to the CPU, reads any letter case, falls back on
/// a bad value with a warning, and survives the template and a serialise round
/// trip.
#[test]
fn test_compute_device_setting() {
    assert_eq!(
        StackingSettings::default().compute_device,
        ComputeDevice::Cpu
    );
    assert_eq!(ComputeDevice::default(), ComputeDevice::Cpu);

    for (text, expected) in [
        ("compute_device = \"gpu\"", ComputeDevice::Gpu),
        ("compute_device = \"GPU\"", ComputeDevice::Gpu),
        ("compute_device = \"Cpu\"", ComputeDevice::Cpu),
    ] {
        let (s, warnings) = StackingSettings::from_toml_str(text).expect("valid TOML");
        assert_eq!(s.compute_device, expected, "{text}");
        assert!(warnings.is_empty(), "{text}: {warnings:?}");
    }

    let (s, warnings) =
        StackingSettings::from_toml_str("compute_device = \"tpu\"").expect("valid TOML");
    assert_eq!(s.compute_device, ComputeDevice::Cpu);
    assert!(
        warnings.iter().any(|w| w.contains("compute_device")),
        "{warnings:?}"
    );

    // The annotated template documents the key and parses to the default.
    assert!(DEFAULT_CONFIG_TOML.contains("compute_device = \"cpu\""));
    let (from_template, warnings) =
        StackingSettings::from_toml_str(DEFAULT_CONFIG_TOML).expect("template parses");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(from_template, StackingSettings::default());

    // Serialisation is lowercase and reads back.
    let gpu = StackingSettings {
        compute_device: ComputeDevice::Gpu,
        ..StackingSettings::default()
    };
    let text = toml::to_string(&gpu).expect("serialises");
    assert!(text.contains("compute_device = \"gpu\""), "{text}");
    let (back, _) = StackingSettings::from_toml_str(&text).expect("reads back");
    assert_eq!(back, gpu);

    // The combo-box spelling round-trips; an unknown label is the default.
    for device in [ComputeDevice::Cpu, ComputeDevice::Gpu] {
        assert_eq!(ComputeDevice::from_combo_str(device.as_combo_str()), device);
    }
    assert_eq!(ComputeDevice::from_combo_str("Quantum"), ComputeDevice::Cpu);
}
