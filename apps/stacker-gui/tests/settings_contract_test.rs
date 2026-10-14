//! The settings contract: the pure `StackingSettings` <-> dialog-values mapping,
//! and guards that keep the Slint tree free of defaults of its own and wire every
//! setting to a widget.

use std::fs;

use stacker_gui::{
    settings::{
        AlignmentModeSetting, ComputeDevice, ImageSavingSettings, OptimizerSetting, OutputFormat,
        PreprocessingSettings, SETTINGS_VERSION, StackingSettings,
    },
    settings_contract::{
        SettingsValues, alignment_options, compute_device_options, optimizer_options,
        output_format_options, tile_size_from_label, tile_size_label, tile_size_options,
    },
};

/// A settings value with every field away from its default and on a range edge.
fn edge_settings() -> StackingSettings {
    StackingSettings {
        settings_version: SETTINGS_VERSION,
        alignment_mode: AlignmentModeSetting::Affine,
        optimizer: OptimizerSetting::NelderMead,
        bounded_search: true,
        akaze_seeding: true,
        correct_brightness: false,
        chain_alignment: true,
        neural_refine_classically: false,
        crop_to_common_area: false,
        resize_cropped_to_original: true,
        auto_cull: false,
        auto_cull_threshold_pct: 50.0,
        sort_by_sharpness: true,
        stack_every_nth: 10,
        tile_size: 8192,
        pyramid_levels: 12,
        use_all_color_channels: true,
        grit_suppression: false,
        relief_estimation_radius: 100,
        relief_smoothing_radius: 50,
        relief_contrast_pct: 1.0,
        relief_show_preview: true,
        relief_auto_detect: true,
        relief_use_multigrid: true,
        strata_base_radius: 64,
        strata_detail_focus: 5,
        compute_device: ComputeDevice::Gpu,
        preprocessing: PreprocessingSettings {
            pre_rotation: 270,
            pre_crop_enabled: true,
            pre_crop_spec: "10,20,1920,1080".to_owned(),
            pre_resize_percent: 10,
            sort_reverse: true,
            ignore_exif_orientation: true,
        },
        image_saving: ImageSavingSettings {
            output_format: OutputFormat::Jpeg,
            bit_depth: 8,
            jpeg_quality: 1,
            filename_template: "{name}_ü_stacked".to_owned(),
            default_output_dir: "C:\\out dir\\ß".to_owned(),
            copy_metadata: true,
        },
    }
}

#[test]
fn default_settings_round_trip() {
    let settings = StackingSettings::default();
    let values = SettingsValues::from(&settings);
    assert_eq!(values.to_settings(), settings);
    assert_eq!(values.to_sanitized(), settings);
}

#[test]
fn edge_settings_round_trip() {
    let settings = edge_settings();
    let values = SettingsValues::from(&settings);
    assert_eq!(values.to_settings(), settings);
    assert_eq!(values.to_sanitized(), settings);
}

#[test]
fn every_alignment_mode_and_optimizer_survives_a_round_trip() {
    // The combo may not list a mode in this build (Neural), but a file that
    // names it must not be rewritten.
    for mode in [
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Registration,
        AlignmentModeSetting::Affine,
        AlignmentModeSetting::None,
        AlignmentModeSetting::Neural,
    ] {
        for optimizer in [
            OptimizerSetting::Auto,
            OptimizerSetting::LucasKanade,
            OptimizerSetting::NelderMead,
        ] {
            let settings = StackingSettings {
                alignment_mode: mode,
                optimizer,
                ..StackingSettings::default()
            };
            let values = SettingsValues::from(&settings);
            assert_eq!(values.to_settings(), settings, "{mode:?} / {optimizer:?}");
        }
    }
}

#[test]
fn registration_is_a_real_mode_and_the_default() {
    // D3: Registration is no longer hidden; it is what a fresh install shows.
    let reset = SettingsValues::dialog_reset();
    assert_eq!(reset.alignment, "Registration");
    assert!(alignment_options().contains(&"Registration"));
}

#[test]
fn alignment_options_follow_the_build() {
    let options = alignment_options();
    let neural_build = cfg!(all(feature = "nn", not(target_arch = "wasm32")));
    let expected: &[&str] = if neural_build {
        &["Translation", "Registration", "Affine", "None", "Neural"]
    } else {
        &["Translation", "Registration", "Affine", "None"]
    };
    assert_eq!(options, expected);
    assert_eq!(options.contains(&"Neural"), neural_build);
    // Every entry is something the core can read back.
    for option in options {
        assert_eq!(
            AlignmentModeSetting::from_combo_str(option).as_combo_str(),
            option
        );
    }
}

#[test]
fn optimizer_options_match_the_core() {
    let options = optimizer_options();
    assert_eq!(options, ["Auto", "Lucas-Kanade", "Nelder-Mead"]);
    for option in options {
        assert_eq!(
            OptimizerSetting::from_combo_str(option).as_combo_str(),
            option
        );
    }
}

#[test]
fn output_format_options_match_the_core() {
    let options = output_format_options();
    assert_eq!(options, ["TIFF", "PNG", "JPEG"]);
    for option in options {
        assert_eq!(OutputFormat::from_combo_str(option).as_combo_str(), option);
    }
}

#[test]
fn tile_size_labels_round_trip() {
    let options = tile_size_options();
    assert_eq!(options.first().map(String::as_str), Some("In Memory"));
    assert_eq!(options[1], "512 px");
    // Every offered label reads back to a tile size the core accepts as it is.
    for option in &options {
        let size = tile_size_from_label(option);
        assert!(size == 0 || (64..=8192).contains(&size), "{option}");
        assert_eq!(&tile_size_label(size), option);
    }
    // A value that is not on the list is shown as typed and survives.
    assert_eq!(tile_size_label(700), "700 px");
    assert_eq!(tile_size_from_label("700 px"), 700);
    // Junk reads as the safe default, in RAM.
    assert_eq!(tile_size_from_label(""), 0);
    assert_eq!(tile_size_from_label("lots"), 0);
    assert_eq!(tile_size_from_label("-5 px"), 0);
    // Through the contract: a small nonzero value is lifted by sanitized().
    let mut values = SettingsValues::dialog_reset();
    values.tile_size = "10 px".to_owned();
    assert_eq!(values.to_sanitized().tile_size, 64);
}

#[test]
fn unknown_combo_strings_map_to_the_defaults() {
    let mut values = SettingsValues::dialog_reset();
    values.alignment = "Bogus".to_owned();
    values.optimizer = "Bogus".to_owned();
    values.output_format = String::new();
    let settings = values.to_settings();
    assert_eq!(settings.alignment_mode, AlignmentModeSetting::Registration);
    assert_eq!(settings.optimizer, OptimizerSetting::Auto);
    assert_eq!(settings.image_saving.output_format, OutputFormat::Tiff);
}

#[test]
fn dialog_reset_is_the_core_default() {
    let reset = SettingsValues::dialog_reset();
    assert_eq!(reset.to_settings(), StackingSettings::default());
    assert_eq!(reset.to_sanitized(), StackingSettings::default());

    // The reset shows the core's 8 / 5 / 2 / 95, not values typed into the Slint
    // tree.
    assert_eq!(reset.pyramid_levels.to_bits(), 8.0_f32.to_bits());
    assert_eq!(reset.relief_estimation_radius.to_bits(), 5.0_f32.to_bits());
    assert_eq!(reset.relief_smoothing_radius.to_bits(), 2.0_f32.to_bits());
    assert_eq!(reset.jpeg_quality.to_bits(), 95.0_f32.to_bits());
    // ... and the new keys' defaults.
    assert!(reset.correct_brightness);
    assert!(!reset.chain_alignment);
    assert!(reset.neural_refine_classically);
    assert!(reset.crop_to_common_area);
    assert!(!reset.resize_cropped_to_original);
    assert_eq!(reset.auto_cull_threshold_pct.to_bits(), 0.0_f32.to_bits());
    assert_eq!(reset.tile_size, "In Memory");
    assert_eq!(reset.strata_base_radius.to_bits(), 31.0_f32.to_bits());
    assert_eq!(reset.strata_detail_focus.to_bits(), 3.0_f32.to_bits());
    assert_eq!(reset.optimizer, "Auto");
    assert!(!reset.relief_use_multigrid);
    assert!(!reset.sort_by_sharpness);
}

#[test]
fn out_of_range_values_are_clamped_on_the_way_back() {
    let wild = StackingSettings {
        stack_every_nth: 0,
        pyramid_levels: 99,
        relief_estimation_radius: 5_000,
        relief_smoothing_radius: 5_000,
        relief_contrast_pct: 7.5,
        auto_cull_threshold_pct: 99.0,
        strata_base_radius: 2,
        strata_detail_focus: 40,
        tile_size: 20_000,
        preprocessing: PreprocessingSettings {
            pre_rotation: 45,
            pre_resize_percent: 3,
            ..PreprocessingSettings::default()
        },
        image_saving: ImageSavingSettings {
            bit_depth: 12,
            jpeg_quality: 250,
            ..ImageSavingSettings::default()
        },
        ..StackingSettings::default()
    };
    let clamped = SettingsValues::from(&wild).to_sanitized();
    assert_eq!(clamped, wild.sanitized().0);
    assert_eq!(clamped.stack_every_nth, 1);
    assert_eq!(clamped.pyramid_levels, 12);
    assert_eq!(clamped.relief_estimation_radius, 100);
    assert_eq!(clamped.relief_smoothing_radius, 50);
    assert_eq!(clamped.relief_contrast_pct.to_bits(), 1.0_f32.to_bits());
    assert_eq!(
        clamped.auto_cull_threshold_pct.to_bits(),
        50.0_f32.to_bits()
    );
    assert_eq!(clamped.strata_base_radius, 8);
    assert_eq!(clamped.strata_detail_focus, 5);
    assert_eq!(clamped.tile_size, 8192);
    assert_eq!(clamped.preprocessing.pre_rotation, 0);
    assert_eq!(clamped.preprocessing.pre_resize_percent, 10);
    assert_eq!(clamped.image_saving.bit_depth, 16);
    assert_eq!(clamped.image_saving.jpeg_quality, 100);
}

#[test]
fn slider_floats_are_rounded_and_junk_is_survivable() {
    let mut values = SettingsValues::dialog_reset();
    values.pyramid_levels = 6.4;
    values.relief_estimation_radius = -3.0;
    values.relief_smoothing_radius = f32::NAN;
    values.relief_contrast_pct = f32::NAN;
    values.auto_cull_threshold_pct = f32::NAN;
    values.strata_base_radius = 31.6;
    values.strata_detail_focus = f32::NAN;
    values.pre_rotation = -90;
    values.bit_depth = -1;
    let settings = values.to_sanitized();
    assert_eq!(settings.pyramid_levels, 6);
    assert_eq!(settings.relief_estimation_radius, 0);
    assert_eq!(settings.relief_smoothing_radius, 0);
    assert_eq!(settings.relief_contrast_pct.to_bits(), 0.0_f32.to_bits());
    assert_eq!(
        settings.auto_cull_threshold_pct.to_bits(),
        0.0_f32.to_bits()
    );
    assert_eq!(settings.strata_base_radius, 32);
    assert_eq!(settings.strata_detail_focus, 1);
    assert_eq!(settings.preprocessing.pre_rotation, 0);
    assert_eq!(settings.image_saving.bit_depth, 16);
}

// ── The Slint tree carries no setting default of its own ─────────────────────

fn read_ui_file(relative: &str) -> String {
    let path = format!("{}/ui/{relative}", env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

/// The property declarations of `settings.slint`: (`in`/`in-out`, name, rest of line).
fn settings_model_properties() -> Vec<(String, String, String)> {
    read_ui_file("models/settings.slint")
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let (kind, rest) = if let Some(rest) = line.strip_prefix("in-out property") {
                ("in-out", rest)
            } else {
                ("in", line.strip_prefix("in property")?)
            };
            let after_type = rest.split_once('>')?.1.trim();
            let name_end = after_type.find([':', ';'])?;
            let (name, tail) = after_type.split_at(name_end);
            Some((kind.to_owned(), name.trim().to_owned(), tail.to_owned()))
        })
        .collect()
}

#[test]
fn settings_model_declares_no_defaults() {
    let properties = settings_model_properties();
    assert!(
        !properties.is_empty(),
        "no property found: parser out of date"
    );
    for (_, name, tail) in &properties {
        assert!(
            tail.starts_with(';'),
            "`{name}` has a literal default in models/settings.slint (`{tail}`); \
             defaults come from StackingSettings::default() pushed by Rust"
        );
    }
}

#[test]
fn settings_model_has_the_38_values_and_five_option_lists() {
    let properties = settings_model_properties();
    let values = properties
        .iter()
        .filter(|(kind, ..)| kind == "in-out")
        .count();
    let options: Vec<&str> = properties
        .iter()
        .filter(|(kind, ..)| kind == "in")
        .map(|(_, name, _)| name.as_str())
        .collect();
    // 39 settings keys, minus `settings_version`, which no widget edits.
    assert_eq!(values, 38);
    assert_eq!(
        options,
        [
            "alignment-options",
            "optimizer-options",
            "compute-device-options",
            "tile-size-options",
            "output-format-options"
        ]
    );
}

#[test]
fn every_settings_model_value_has_a_widget_in_the_dialog() {
    let dialog = read_ui_file("components/settings_dialog.slint");
    for (kind, name, _) in settings_model_properties() {
        if kind == "in-out" {
            assert!(
                dialog.contains(&format!("SettingsModel.{name}")),
                "`{name}` is a settings value but no control in the dialog uses it"
            );
        }
    }
    // The pushed combo lists are used as the model of a combo box.
    for list in [
        "alignment-options",
        "optimizer-options",
        "compute-device-options",
        "tile-size-options",
        "output-format-options",
    ] {
        assert!(
            dialog.contains(&format!("model: SettingsModel.{list};")),
            "`{list}` is not the model of a combo box"
        );
    }
}

#[test]
fn the_dialog_has_no_combo_list_of_its_own_for_the_pushed_lists() {
    let dialog = read_ui_file("components/settings_dialog.slint");
    // Registration is a real mode now: it comes from the pushed list, not from a
    // literal in the dialog (the tooltip may name it).
    assert!(!dialog.contains("model: [\"Translation\""));
    assert!(!dialog.contains("model: [\"Auto\""));
}

/// The line of the row labelled `label` and whether the line above it opens the
/// row with `if !AppModel.web-build:`.
fn is_hidden_on_web(dialog: &str, label: &str) -> bool {
    let needle = format!("label: \"{label}\";");
    let lines: Vec<&str> = dialog.lines().collect();
    let at = lines
        .iter()
        .position(|l| l.trim() == needle)
        .unwrap_or_else(|| panic!("no row labelled `{label}`"));
    lines[..at]
        .iter()
        .rev()
        .find(|l| l.contains("SettingRow {"))
        .is_some_and(|l| l.contains("if !AppModel.web-build:"))
}

#[test]
fn the_browser_build_hides_what_it_cannot_do() {
    let dialog = read_ui_file("components/settings_dialog.slint");
    for label in [
        "Optimizer",
        "Compute Device",
        "Tile Size",
        "Default output dir",
    ] {
        assert!(
            is_hidden_on_web(&dialog, label),
            "`{label}` must be hidden when AppModel.web-build is set"
        );
    }
    // What the browser does offer.
    for label in [
        "Alignment Mode",
        "Bounded Search",
        "Correct Brightness",
        "Chain Alignment",
        "Cull Threshold %",
        "Base Radius",
        "Multigrid Depth Solver",
        "Crop to Common Area",
    ] {
        assert!(
            !is_hidden_on_web(&dialog, label),
            "`{label}` must stay available in the browser build"
        );
    }
    // The Neural rows hang on the Neural mode, which the browser never lists.
    assert!(dialog.contains("if SettingsModel.alignment == \"Neural\" : SettingRow {"));
}

#[test]
fn the_tab_table_is_the_only_place_that_numbers_the_tabs() {
    let dialog = read_ui_file("components/settings_dialog.slint");
    assert!(
        !dialog.contains("settings-tab == 0")
            && !dialog.contains("settings-tab == 1")
            && !dialog.contains("settings-tab == 2")
            && !dialog.contains("settings-tab == 3")
            && !dialog.contains("settings-tab == 4")
            && !dialog.contains("settings-tab == 5"),
        "the dialog must use SettingsTabIndex, not a bare tab number"
    );
    let tabs = read_ui_file("components/settings_tabs.slint");
    assert!(tabs.contains(
        "labels: [\"General\", \"Apex\", \"Relief\", \"Strata\", \"Input\", \"Output\"];"
    ));
    for (name, index) in [
        ("general", 0),
        ("apex", 1),
        ("relief", 2),
        ("strata", 3),
        ("input", 4),
        ("output", 5),
    ] {
        assert!(
            tabs.contains(&format!("out property <int> {name}: {index};")),
            "SettingsTabIndex.{name} must be {index}"
        );
        assert!(
            dialog.contains(&format!("SettingsTabIndex.{name} :")),
            "no tab content for SettingsTabIndex.{name}"
        );
    }
}

#[test]
fn the_algorithm_list_names_the_three_engines() {
    let expected = [
        "\"Relief (Depth Map)\"",
        "\"Apex (Laplacian)\"",
        "\"Strata (Guided Fusion)\"",
        "\"Both (Relief & Apex)\"",
        "\"All Three (Strata, Relief & Apex)\"",
    ];
    for file in ["components/sidebar.slint", "models/app_model.slint"] {
        let text = read_ui_file(file);
        for label in expected {
            assert!(text.contains(label), "{file} lacks {label}");
        }
        for legacy in ["PMax", "DMap"] {
            assert!(!text.contains(legacy), "{file} still says {legacy}");
        }
    }
}

#[test]
fn the_compute_device_combo_offers_the_gpu_only_in_a_gpu_build() {
    let options = compute_device_options();
    assert_eq!(options.first(), Some(&"CPU"), "the CPU is always first");
    assert_eq!(
        options.contains(&"GPU"),
        cfg!(feature = "gpu"),
        "GPU is listed exactly when the build has GPU support: {options:?}"
    );
    // Whatever the combo lists, a file that says gpu survives the dialog.
    let settings = StackingSettings {
        compute_device: ComputeDevice::Gpu,
        ..StackingSettings::default()
    };
    assert_eq!(SettingsValues::from(&settings).to_settings(), settings);
    // An unknown combo label is the default, the CPU.
    let mut values = SettingsValues::dialog_reset();
    values.compute_device = "Quantum".to_owned();
    assert_eq!(values.to_settings().compute_device, ComputeDevice::Cpu);
}
