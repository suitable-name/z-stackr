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
        AlignmentModeSetting::from_combo_str("Unknown"),
        AlignmentModeSetting::Affine
    ); // default
}

#[test]
fn test_alignment_serde_round_trip() {
    // Alignment mode plus the AKAZE-seeding toggle must round-trip through
    // TOML. "translation" is the lowercase serde form (serde rename_all =
    // "lowercase").
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        akaze_seeding: true,
        ..Default::default()
    };
    let toml_str = toml::to_string(&settings).expect("StackingSettings must serialize");
    assert!(
        toml_str.contains("alignment_mode = \"translation\""),
        "TOML must contain lowercase 'translation'; got: {toml_str}"
    );
    assert!(
        toml_str.contains("akaze_seeding = true"),
        "TOML must contain akaze_seeding; got: {toml_str}"
    );
    let back: StackingSettings =
        toml::from_str(&toml_str).expect("StackingSettings must deserialize");
    assert_eq!(back.alignment_mode, AlignmentModeSetting::Translation);
    assert!(back.akaze_seeding);
}

#[cfg(feature = "nn")]
#[test]
fn test_alignment_neural_serde_round_trip() {
    // The `Neural` variant only exists under the `nn` feature. It must
    // round-trip through TOML exactly like every other variant ("neural" is
    // the lowercase serde form, same `rename_all = "lowercase"` convention),
    // and a `StackingSettings` with `alignment_mode = Neural` set must
    // deserialize back to an equal value.
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Neural,
        ..Default::default()
    };
    let toml_str = toml::to_string(&settings).expect("StackingSettings must serialize");
    assert!(
        toml_str.contains("alignment_mode = \"neural\""),
        "TOML must contain lowercase 'neural'; got: {toml_str}"
    );
    let back: StackingSettings =
        toml::from_str(&toml_str).expect("StackingSettings must deserialize");
    assert_eq!(back.alignment_mode, AlignmentModeSetting::Neural);
    assert_eq!(back, settings);
}

#[cfg(feature = "nn")]
#[test]
fn test_alignment_mode_setting_combo_str_neural() {
    assert_eq!(AlignmentModeSetting::Neural.as_combo_str(), "Neural");
    assert_eq!(
        AlignmentModeSetting::from_combo_str("Neural"),
        AlignmentModeSetting::Neural
    );
}

#[test]
fn test_old_config_without_neural_variant_still_deserializes() {
    // A config file written before the `Neural` variant existed (or built
    // without the `nn` feature) simply never mentions "neural" — it must
    // keep deserializing into one of the pre-existing variants exactly as
    // before. This is the "no breaking change to old configs" contract for
    // adding a new enum variant behind a feature flag.
    let toml_str = "alignment_mode = \"registration\"\n";
    let back: StackingSettings =
        toml::from_str(toml_str).expect("old-style config must still deserialize");
    assert_eq!(back.alignment_mode, AlignmentModeSetting::Registration);
}

#[test]
fn test_optimizer_setting_combo_str() {
    assert_eq!(OptimizerSetting::Auto.as_combo_str(), "Auto");
    assert_eq!(OptimizerSetting::LucasKanade.as_combo_str(), "Lucas-Kanade");
    assert_eq!(OptimizerSetting::NelderMead.as_combo_str(), "Nelder-Mead");

    assert_eq!(
        OptimizerSetting::from_combo_str("Auto"),
        OptimizerSetting::Auto
    );
    assert_eq!(
        OptimizerSetting::from_combo_str("Lucas-Kanade"),
        OptimizerSetting::LucasKanade
    );
    assert_eq!(
        OptimizerSetting::from_combo_str("Nelder-Mead"),
        OptimizerSetting::NelderMead
    );
    assert_eq!(
        OptimizerSetting::from_combo_str("Unknown"),
        OptimizerSetting::Auto
    ); // default
}

#[test]
fn test_optimizer_serde_round_trip() {
    let settings = StackingSettings {
        optimizer: OptimizerSetting::LucasKanade,
        ..Default::default()
    };
    let toml_str = toml::to_string(&settings).expect("StackingSettings must serialize");
    assert!(
        toml_str.contains("optimizer = \"lucaskanade\""),
        "TOML must contain lowercase 'lucaskanade'; got: {toml_str}"
    );
    let back: StackingSettings =
        toml::from_str(&toml_str).expect("StackingSettings must deserialize");
    assert_eq!(back.optimizer, OptimizerSetting::LucasKanade);
}

#[test]
fn test_old_config_without_optimizer_field_still_deserializes() {
    // A config file written before the `optimizer` field existed simply
    // never mentions it — it must still deserialize, falling back to
    // `OptimizerSetting::Auto` via `#[serde(default)]` on `StackingSettings`
    // plus `Default` on `OptimizerSetting`. Same "no breaking change to old
    // configs" contract as `test_old_config_without_neural_variant_still_deserializes`.
    let toml_str = "alignment_mode = \"registration\"\n";
    let back: StackingSettings =
        toml::from_str(toml_str).expect("old-style config must still deserialize");
    assert_eq!(back.optimizer, OptimizerSetting::Auto);
}

#[test]
fn test_default_config_toml_parses_optimizer_auto() {
    let parsed: StackingSettings =
        toml::from_str(DEFAULT_CONFIG_TOML).expect("DEFAULT_CONFIG_TOML must parse");
    assert_eq!(parsed.optimizer, StackingSettings::default().optimizer);
    assert_eq!(parsed.optimizer, OptimizerSetting::Auto);
}

#[test]
fn test_use_gpu_defaults_true_and_round_trips_through_toml() {
    let settings = StackingSettings::default();
    assert!(
        settings.use_gpu,
        "use_gpu must default to true (harmless no-op in non-gpu builds, opt-out via config/CLI/GUI/Python otherwise)"
    );

    let toml_str = toml::to_string(&settings).expect("StackingSettings must serialize");
    assert!(
        toml_str.contains("use_gpu = true"),
        "TOML must contain use_gpu = true; got: {toml_str}"
    );

    let disabled = StackingSettings {
        use_gpu: false,
        ..Default::default()
    };
    let toml_str = toml::to_string(&disabled).expect("StackingSettings must serialize");
    let back: StackingSettings =
        toml::from_str(&toml_str).expect("StackingSettings must deserialize");
    assert!(!back.use_gpu);
}

#[test]
fn test_default_config_toml_parses_use_gpu_true() {
    // The shipped DEFAULT_CONFIG_TOML must agree with StackingSettings's own
    // Default impl on every field, use_gpu included.
    let parsed: StackingSettings =
        toml::from_str(DEFAULT_CONFIG_TOML).expect("DEFAULT_CONFIG_TOML must parse");
    assert_eq!(parsed.use_gpu, StackingSettings::default().use_gpu);
    assert!(parsed.use_gpu);
}

#[test]
fn test_old_config_without_strata_detail_focus_field_still_deserializes() {
    // A config file written before the `strata_detail_focus` field existed
    // simply never mentions it — it must still deserialize, falling back to
    // the default `3` via `#[serde(default)]` on `StackingSettings`. Same
    // "no breaking change to old configs" contract as
    // `test_old_config_without_optimizer_field_still_deserializes`.
    let toml_str = "alignment_mode = \"registration\"\n";
    let back: StackingSettings =
        toml::from_str(toml_str).expect("old-style config must still deserialize");
    assert_eq!(back.strata_detail_focus, 3);
}

#[test]
fn test_strata_detail_focus_clamp() {
    let mut settings = StackingSettings {
        strata_detail_focus: 0, // below min
        ..Default::default()
    };
    settings.clamp_valid();
    assert_eq!(settings.strata_detail_focus, 1);

    let mut settings = StackingSettings {
        strata_detail_focus: 9, // above max
        ..Default::default()
    };
    settings.clamp_valid();
    assert_eq!(settings.strata_detail_focus, 5);

    let mut settings = StackingSettings {
        strata_detail_focus: 4, // valid, untouched
        ..Default::default()
    };
    settings.clamp_valid();
    assert_eq!(settings.strata_detail_focus, 4);
}

#[test]
fn test_default_config_toml_parses_strata_detail_focus() {
    let parsed: StackingSettings =
        toml::from_str(DEFAULT_CONFIG_TOML).expect("DEFAULT_CONFIG_TOML must parse");
    assert_eq!(
        parsed.strata_detail_focus,
        StackingSettings::default().strata_detail_focus
    );
    assert_eq!(parsed.strata_detail_focus, 3);
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
