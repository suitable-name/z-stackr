//! The settings keys added with the GitHub port, the legacy aliases and the
//! version-2 migration of the alignment modes.

use stacker_core::settings::*;

/// The new keys default to the documented values.
#[test]
fn new_key_defaults() {
    let d = StackingSettings::default();
    assert_eq!(d.settings_version, SETTINGS_VERSION);
    assert_eq!(SETTINGS_VERSION, 2);
    assert_eq!(d.alignment_mode, AlignmentModeSetting::Registration);
    assert_eq!(d.optimizer, OptimizerSetting::Auto);
    assert!(d.correct_brightness);
    assert!(!d.chain_alignment);
    assert!(d.neural_refine_classically);
    assert!(d.crop_to_common_area);
    assert!(!d.resize_cropped_to_original);
    assert!(!d.sort_by_sharpness);
    assert!(d.auto_cull_threshold_pct.abs() < f32::EPSILON);
    assert_eq!(d.tile_size, 0);
    assert!(!d.relief_use_multigrid);
    assert_eq!(d.strata_base_radius, 31);
    assert_eq!(d.strata_detail_focus, 3);
}

/// Every new key, set to a non-default value, survives the lenient loader and
/// a serialise / load round trip.
#[test]
fn new_keys_load_and_round_trip() {
    let text = r#"
settings_version = 2
alignment_mode = "affine"
optimizer = "neldermead"
correct_brightness = false
chain_alignment = true
neural_refine_classically = false
crop_to_common_area = false
resize_cropped_to_original = true
auto_cull_threshold_pct = 7.5
sort_by_sharpness = true
tile_size = 512
relief_use_multigrid = true
strata_base_radius = 20
strata_detail_focus = 5
"#;
    let (s, warnings) = StackingSettings::from_toml_str(text).expect("valid TOML");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Affine);
    assert_eq!(s.optimizer, OptimizerSetting::NelderMead);
    assert!(!s.correct_brightness);
    assert!(s.chain_alignment);
    assert!(!s.neural_refine_classically);
    assert!(!s.crop_to_common_area);
    assert!(s.resize_cropped_to_original);
    assert!((s.auto_cull_threshold_pct - 7.5).abs() < f32::EPSILON);
    assert!(s.sort_by_sharpness);
    assert_eq!(s.tile_size, 512);
    assert!(s.relief_use_multigrid);
    assert_eq!(s.strata_base_radius, 20);
    assert_eq!(s.strata_detail_focus, 5);

    let written = toml::to_string(&s).expect("serialises");
    let (back, warnings) = StackingSettings::from_toml_str(&written).expect("reads back");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(back, s);
    // Plain serde reads the same text to the same value.
    assert_eq!(toml::from_str::<StackingSettings>(&written).unwrap(), s);
}

/// A wrongly typed new key keeps its default and is reported by name.
#[test]
fn new_keys_are_lenient() {
    let text = "optimizer = \"gradient\"\nchain_alignment = \"yes\"\ntile_size = \"big\"\n";
    let (s, warnings) = StackingSettings::from_toml_str(text).expect("valid TOML");
    let d = StackingSettings::default();
    assert_eq!(s.optimizer, d.optimizer);
    assert_eq!(s.chain_alignment, d.chain_alignment);
    assert_eq!(s.tile_size, d.tile_size);
    for name in ["optimizer", "chain_alignment", "tile_size"] {
        assert!(
            warnings.iter().any(|w| w.contains(name)),
            "no warning names {name}: {warnings:?}"
        );
    }
}

/// `OptimizerSetting` reads any letter case and the short and hyphenated
/// spellings; the combo labels round-trip.
#[test]
fn optimizer_setting_spellings() {
    for (text, expected) in [
        ("auto", OptimizerSetting::Auto),
        ("AUTO", OptimizerSetting::Auto),
        ("lucaskanade", OptimizerSetting::LucasKanade),
        ("LucasKanade", OptimizerSetting::LucasKanade),
        ("lk", OptimizerSetting::LucasKanade),
        ("lucas-kanade", OptimizerSetting::LucasKanade),
        ("Lucas-Kanade", OptimizerSetting::LucasKanade),
        ("neldermead", OptimizerSetting::NelderMead),
        ("nm", OptimizerSetting::NelderMead),
        ("nelder-mead", OptimizerSetting::NelderMead),
        ("Nelder-Mead", OptimizerSetting::NelderMead),
    ] {
        let s: StackingSettings =
            toml::from_str(&format!("optimizer = \"{text}\"")).expect("optimizer parses");
        assert_eq!(s.optimizer, expected, "{text}");
    }
    assert!(toml::from_str::<StackingSettings>("optimizer = \"newton\"").is_err());

    for optimizer in [
        OptimizerSetting::Auto,
        OptimizerSetting::LucasKanade,
        OptimizerSetting::NelderMead,
    ] {
        assert_eq!(
            OptimizerSetting::from_combo_str(optimizer.as_combo_str()),
            optimizer
        );
    }
    assert_eq!(
        OptimizerSetting::from_combo_str("???"),
        OptimizerSetting::Auto
    );
    assert_eq!(OptimizerSetting::default(), OptimizerSetting::Auto);

    let written = toml::to_string(&StackingSettings {
        optimizer: OptimizerSetting::LucasKanade,
        ..StackingSettings::default()
    })
    .unwrap();
    assert!(written.contains("optimizer = \"lucaskanade\""), "{written}");
}

/// The new ranges clamp with one warning each.
#[test]
fn new_ranges_clamp_and_report() {
    let s = StackingSettings {
        auto_cull_threshold_pct: 80.0,
        tile_size: 10,
        strata_base_radius: 3,
        strata_detail_focus: 9,
        ..StackingSettings::default()
    };
    let (clean, warnings) = s.sanitized();
    assert!((clean.auto_cull_threshold_pct - 50.0).abs() < f32::EPSILON);
    assert_eq!(clean.tile_size, 64);
    assert_eq!(clean.strata_base_radius, 8);
    assert_eq!(clean.strata_detail_focus, 5);
    assert_eq!(warnings.len(), 4, "{warnings:?}");
    for name in [
        "auto_cull_threshold_pct",
        "tile_size",
        "strata_base_radius",
        "strata_detail_focus",
    ] {
        assert!(
            warnings.iter().any(|w| w.contains(name)),
            "{name}: {warnings:?}"
        );
    }

    let s = StackingSettings {
        auto_cull_threshold_pct: -1.0,
        tile_size: 100_000,
        strata_base_radius: 100,
        strata_detail_focus: 0,
        ..StackingSettings::default()
    };
    let (clean, _) = s.sanitized();
    assert!(clean.auto_cull_threshold_pct.abs() < f32::EPSILON);
    assert_eq!(clean.tile_size, 8192);
    assert_eq!(clean.strata_base_radius, 64);
    assert_eq!(clean.strata_detail_focus, 1);

    let nan = StackingSettings {
        auto_cull_threshold_pct: f32::NAN,
        ..StackingSettings::default()
    };
    let (clean, warnings) = nan.sanitized();
    assert!(clean.auto_cull_threshold_pct.abs() < f32::EPSILON);
    assert_eq!(warnings.len(), 1, "{warnings:?}");

    // 0 (in RAM), the bounds and in-range values stay quiet.
    for tile_size in [0, 64, 1024, 8192] {
        let (clean, warnings) = StackingSettings {
            tile_size,
            ..StackingSettings::default()
        }
        .sanitized();
        assert_eq!(clean.tile_size, tile_size);
        assert!(warnings.is_empty(), "{tile_size}: {warnings:?}");
    }
}

/// A newer `settings_version` is read as it is and reported once.
#[test]
fn newer_settings_version_warns() {
    let (s, warnings) =
        StackingSettings::from_toml_str("settings_version = 9\nalignment_mode = \"affine\"")
            .expect("valid TOML");
    assert_eq!(s.settings_version, 9);
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Affine);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("settings_version"));
}

/// The legacy `dmap_*` keys read as the `relief_*` keys, silently.
#[test]
fn legacy_dmap_keys_are_aliases() {
    let text = "dmap_estimation_radius = 9\ndmap_smoothing_radius = 4\n\
                dmap_contrast_pct = 0.5\ndmap_show_preview = true\ndmap_auto_detect = true\n";
    let (s, warnings) = StackingSettings::from_toml_str(text).expect("valid TOML");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(s.relief_estimation_radius, 9);
    assert_eq!(s.relief_smoothing_radius, 4);
    assert!((s.relief_contrast_pct - 0.5).abs() < f32::EPSILON);
    assert!(s.relief_show_preview);
    assert!(s.relief_auto_detect);

    // Plain serde (project files) accepts them too.
    let plain: StackingSettings = toml::from_str(text).expect("aliases through serde");
    assert_eq!(plain.relief_estimation_radius, 9);
    assert!(plain.relief_auto_detect);
}

/// The new name wins when both spellings are present, and the old one is not
/// reported as an unknown key.
#[test]
fn legacy_dmap_keys_lose_to_the_new_names() {
    let text = "relief_estimation_radius = 11\ndmap_estimation_radius = 9\n\
                dmap_show_preview = true\nrelief_show_preview = false\n";
    let (s, warnings) = StackingSettings::from_toml_str(text).expect("valid TOML");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(s.relief_estimation_radius, 11);
    assert!(!s.relief_show_preview);
}

/// A wrongly typed legacy value is reported like a wrongly typed new one.
#[test]
fn legacy_dmap_key_of_the_wrong_type_is_reported() {
    let (s, warnings) =
        StackingSettings::from_toml_str("dmap_estimation_radius = \"wide\"").expect("valid TOML");
    assert_eq!(
        s.relief_estimation_radius,
        StackingSettings::default().relief_estimation_radius
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("relief_estimation_radius")),
        "{warnings:?}"
    );
}

/// `use_gpu` maps onto `compute_device`, which wins when both are present.
#[test]
fn legacy_use_gpu_alias() {
    let (s, w) = StackingSettings::from_toml_str("use_gpu = true").unwrap();
    assert_eq!(s.compute_device, ComputeDevice::Gpu);
    assert!(w.is_empty(), "{w:?}");

    let (s, w) = StackingSettings::from_toml_str("use_gpu = false").unwrap();
    assert_eq!(s.compute_device, ComputeDevice::Cpu);
    assert!(w.is_empty(), "{w:?}");

    let (s, _) =
        StackingSettings::from_toml_str("use_gpu = true\ncompute_device = \"cpu\"").unwrap();
    assert_eq!(s.compute_device, ComputeDevice::Cpu);

    let (s, _) =
        StackingSettings::from_toml_str("compute_device = \"gpu\"\nuse_gpu = false").unwrap();
    assert_eq!(s.compute_device, ComputeDevice::Gpu);
}

/// The Python binding's `relief_engine` string maps onto `relief_use_multigrid`.
#[test]
fn legacy_relief_engine_alias() {
    let (s, w) = StackingSettings::from_toml_str("relief_engine = \"multigrid\"").unwrap();
    assert!(s.relief_use_multigrid);
    assert!(w.is_empty(), "{w:?}");

    let (s, _) = StackingSettings::from_toml_str("relief_engine = \"guided_filter\"").unwrap();
    assert!(!s.relief_use_multigrid);

    let (s, _) = StackingSettings::from_toml_str(
        "relief_engine = \"multigrid\"\nrelief_use_multigrid = false",
    )
    .unwrap();
    assert!(!s.relief_use_multigrid, "the new key wins");
}

/// Aliases are read-only: saving writes only the current names.
#[test]
fn saving_writes_only_new_names() {
    let (s, _) = StackingSettings::from_toml_str(
        "dmap_estimation_radius = 9\nuse_gpu = true\nrelief_engine = \"multigrid\"\n\
         alignment_mode = \"bounded\"\n",
    )
    .unwrap();
    let written = toml::to_string(&s).expect("serialises");
    for old in ["dmap_", "use_gpu", "relief_engine", "bounded\""] {
        assert!(!written.contains(old), "{old} in {written}");
    }
    assert!(
        written.contains("relief_estimation_radius = 9"),
        "{written}"
    );
    assert!(written.contains("compute_device = \"gpu\""), "{written}");
    assert!(written.contains("relief_use_multigrid = true"), "{written}");
    assert!(written.contains("settings_version = 2"), "{written}");
}

/// `"bounded"` is `registration` plus `bounded_search`, in any version.
#[test]
fn bounded_alias() {
    for prefix in ["", "settings_version = 2\n"] {
        let (s, w) =
            StackingSettings::from_toml_str(&format!("{prefix}alignment_mode = \"Bounded\""))
                .unwrap();
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
        assert!(s.bounded_search);
        assert!(w.is_empty(), "{w:?}");
    }
}

/// Legacy files (no `settings_version`) keep their behaviour through the new
/// alignment-mode meanings.
#[test]
fn legacy_alignment_migration() {
    let load = |text: &str| StackingSettings::from_toml_str(text).unwrap().0;

    // Old `affine` (4-DOF) is today's `registration`.
    let s = load("alignment_mode = \"affine\"");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
    assert!(!s.bounded_search);
    assert_eq!(s.settings_version, SETTINGS_VERSION);

    // Old `registration` (single-pass bounded) is registration + bounded_search.
    let s = load("alignment_mode = \"registration\"");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
    assert!(s.bounded_search);

    for (text, mode) in [
        ("translation", AlignmentModeSetting::Translation),
        ("none", AlignmentModeSetting::None),
    ] {
        let s = load(&format!("alignment_mode = \"{text}\""));
        assert_eq!(s.alignment_mode, mode);
        assert!(!s.bounded_search);
    }

    // Old default (no alignment_mode key) was `affine`.
    let s = load("pyramid_levels = 5");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
    assert!(!s.bounded_search);

    // The old affine + bounded_search file keeps its bounded flag.
    let s = load("alignment_mode = \"affine\"\nbounded_search = true");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
    assert!(s.bounded_search);

    // An explicit version below 2 is legacy as well.
    let s = load("settings_version = 1\nalignment_mode = \"affine\"");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
    assert_eq!(s.settings_version, SETTINGS_VERSION);
}

/// A versioned file is read literally.
#[test]
fn versioned_alignment_is_literal() {
    let load = |mode: &str| {
        StackingSettings::from_toml_str(&format!(
            "settings_version = 2\nalignment_mode = \"{mode}\""
        ))
        .unwrap()
        .0
    };
    let s = load("affine");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Affine);
    assert!(!s.bounded_search);
    let s = load("registration");
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
    assert!(!s.bounded_search);
    assert_eq!(load("neural").alignment_mode, AlignmentModeSetting::Neural);
}

/// A freshly saved default file carries the version and the new default mode,
/// and loading it changes nothing (no second migration).
#[test]
fn fresh_default_file_is_versioned() {
    let written = toml::to_string(&StackingSettings::default()).expect("serialises");
    assert!(written.contains("settings_version = 2"), "{written}");
    assert!(
        written.contains("alignment_mode = \"registration\""),
        "{written}"
    );
    let (back, warnings) = StackingSettings::from_toml_str(&written).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(back, StackingSettings::default());
    assert!(!back.bounded_search);

    assert!(DEFAULT_CONFIG_TOML.contains("settings_version = 2"));
    assert!(DEFAULT_CONFIG_TOML.contains("alignment_mode = \"registration\""));
}

/// `migrate_legacy` on a value deserialised with plain serde.
#[test]
fn migrate_legacy_on_a_serde_value() {
    let mut s: StackingSettings = toml::from_str("alignment_mode = \"affine\"").unwrap();
    assert_eq!(s.settings_version, 0, "a missing key reads as legacy");
    s.migrate_legacy();
    assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
    assert_eq!(s.settings_version, SETTINGS_VERSION);
    let before = s.clone();
    s.migrate_legacy();
    assert_eq!(s, before, "a current value is left alone");
}

/// Every new key is documented in the template.
#[test]
fn template_documents_every_new_key() {
    for key in [
        "settings_version",
        "optimizer",
        "correct_brightness",
        "chain_alignment",
        "neural_refine_classically",
        "crop_to_common_area",
        "resize_cropped_to_original",
        "auto_cull_threshold_pct",
        "sort_by_sharpness",
        "tile_size",
        "relief_use_multigrid",
        "strata_base_radius",
        "strata_detail_focus",
    ] {
        assert!(
            DEFAULT_CONFIG_TOML.contains(&format!("\n{key} = ")),
            "{key} missing from the template"
        );
    }
}
