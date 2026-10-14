//! Loading of the TOML settings file passed with `--config`.
//!
//! The CLI honours every key of the file it writes (`DEFAULT_CONFIG_TOML`):
//! alignment and fusion keys reach the fusion code, `auto_cull`,
//! `stack_every_nth` and `preprocessing.*` shape the frame set, and
//! `image_saving.*` shape the output file (see `pipeline::run_pipeline`).
//! `relief_show_preview`, `ignore_exif_orientation` and `copy_metadata` are
//! accepted for compatibility with the GUI's file but do nothing there either.
//! `tile_size` is the GUI's out-of-core switch: the CLI tiles through its own
//! `--tile-size` flag, which always wins over the key.
use stacker_core::{
    error::StackerError,
    settings::{DEFAULT_CONFIG_TOML, StackingSettings},
};
use stacker_pipeline::config::{Leniency, parse_settings};
use std::path::Path;

const fn invalid(msg: String) -> StackerError {
    StackerError::Config(msg)
}

/// Load the settings for a run.
///
/// * No `--config`: the built-in defaults.
/// * A path that exists: parsed strictly; a read or parse error is an error
///   (the run does not silently continue with defaults).
/// * A path that does not exist: an error, unless `write_default` is set, in
///   which case the commented default file is created there and the defaults
///   are used.
///
/// # Errors
/// Returns an error naming the file when it is missing (without
/// `write_default`), unreadable, or not valid settings TOML.
pub fn load_config(
    path: Option<&Path>,
    write_default: bool,
) -> Result<StackingSettings, StackerError> {
    let Some(path) = path else {
        return Ok(StackingSettings::default());
    };

    if !path.exists() {
        if !write_default {
            return Err(invalid(format!(
                "config file '{}' does not exist (pass --write-default-config to create it)",
                path.display()
            )));
        }
        tracing::info!("creating default config at {}", path.display());
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, DEFAULT_CONFIG_TOML)?;
        return Ok(StackingSettings::default());
    }

    tracing::info!("loading config from {}", path.display());
    let content = std::fs::read_to_string(path)
        .map_err(|e| invalid(format!("cannot read config file '{}': {e}", path.display())))?;
    parse_config(&content)
        .map_err(|msg| invalid(format!("invalid config file '{}': {msg}", path.display())))
}

/// Parse the text of a settings file.
///
/// Accepted beyond the plain schema:
/// * `alignment_mode = "bounded"` (documented in the README) means the
///   `registration` mode with `bounded_search = true`;
/// * the legacy `dmap_*` keys (now `relief_*`), `use_gpu` (now
///   `compute_device`) and `relief_engine` (now `relief_use_multigrid`);
/// * a file without `settings_version` is a legacy file: its `affine` becomes
///   `registration` and its `registration` becomes `registration` with
///   `bounded_search = true`; a file with `settings_version = 2` is read
///   literally;
/// * `alignment_mode`, `optimizer` and `image_saving.output_format` are
///   case-insensitive.
///
/// Out-of-range numeric values are clamped to their documented ranges.
///
/// # Errors
/// Returns the parser's message for invalid TOML, a wrongly typed value or an
/// unknown enum value.
pub fn parse_config(content: &str) -> Result<StackingSettings, String> {
    // The one loader both apps use: `StackingSettings::from_toml_str`, which
    // ends in `sanitized()`. The CLI is strict about wrongly typed values.
    let loaded = parse_settings(content, Leniency::Strict)?;
    for warning in &loaded.warnings {
        tracing::warn!(target: "config", "{warning}");
    }
    Ok(loaded.settings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stacker_core::settings::{
        AlignmentModeSetting, ComputeDevice, OptimizerSetting, OutputFormat, SETTINGS_VERSION,
    };

    /// Every one of the 39 keys, set to a non-default value, must come out of
    /// the loader unchanged.
    const ALL_KEYS: &str = r#"
settings_version = 2
alignment_mode = "translation"
optimizer = "neldermead"
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
    fn all_keys_round_trip_into_the_settings_struct() {
        let s = parse_config(ALL_KEYS).expect("valid config");
        assert_eq!(s.settings_version, SETTINGS_VERSION);
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Translation);
        assert_eq!(s.optimizer, OptimizerSetting::NelderMead);
        assert!(s.bounded_search);
        assert!(s.akaze_seeding);
        assert!(!s.correct_brightness);
        assert!(s.chain_alignment);
        assert!(!s.neural_refine_classically);
        assert!(!s.crop_to_common_area);
        assert!(s.resize_cropped_to_original);
        assert!(!s.auto_cull);
        assert!((s.auto_cull_threshold_pct - 4.0).abs() < f32::EPSILON);
        assert!(s.sort_by_sharpness);
        assert_eq!(s.stack_every_nth, 3);
        assert_eq!(s.tile_size, 256);
        assert!(s.relief_use_multigrid);
        assert_eq!((s.strata_base_radius, s.strata_detail_focus), (24, 4));
        assert_eq!(s.pyramid_levels, 6);
        assert!(s.use_all_color_channels);
        assert!(!s.grit_suppression);
        assert_eq!(s.relief_estimation_radius, 7);
        assert_eq!(s.relief_smoothing_radius, 4);
        assert!((s.relief_contrast_pct - 0.25).abs() < f32::EPSILON);
        assert!(s.relief_show_preview);
        assert!(s.relief_auto_detect);
        assert_eq!(s.compute_device, ComputeDevice::Gpu);
        assert_eq!(s.preprocessing.pre_rotation, 90);
        assert!(s.preprocessing.pre_crop_enabled);
        assert_eq!(s.preprocessing.pre_crop_spec, "50,40");
        assert_eq!(s.preprocessing.pre_resize_percent, 60);
        assert!(s.preprocessing.sort_reverse);
        assert!(s.preprocessing.ignore_exif_orientation);
        assert_eq!(s.image_saving.output_format, OutputFormat::Jpeg);
        assert_eq!(s.image_saving.bit_depth, 8);
        assert_eq!(s.image_saving.jpeg_quality, 30);
        assert_eq!(s.image_saving.filename_template, "{name}_x");
        assert_eq!(s.image_saving.default_output_dir, "out");
        assert!(s.image_saving.copy_metadata);
    }

    #[test]
    fn the_generated_default_file_parses_to_the_defaults() {
        assert_eq!(
            parse_config(DEFAULT_CONFIG_TOML).expect("default file parses"),
            StackingSettings::default()
        );
    }

    #[test]
    fn bounded_is_an_alias_for_registration_with_bounded_search() {
        let s = parse_config("alignment_mode = \"Bounded\"").expect("bounded accepted");
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
        assert!(s.bounded_search);
    }

    #[test]
    fn legacy_keys_and_modes_are_accepted_by_the_strict_parser() {
        let s = parse_config(
            "alignment_mode = \"affine\"\ndmap_estimation_radius = 9\nuse_gpu = true\n\
             relief_engine = \"multigrid\"\n",
        )
        .expect("legacy file accepted");
        assert_eq!(s.alignment_mode, AlignmentModeSetting::Registration);
        assert!(!s.bounded_search);
        assert_eq!(s.relief_estimation_radius, 9);
        assert_eq!(s.compute_device, ComputeDevice::Gpu);
        assert!(s.relief_use_multigrid);

        let literal = parse_config("settings_version = 2\nalignment_mode = \"affine\"")
            .expect("versioned file accepted");
        assert_eq!(literal.alignment_mode, AlignmentModeSetting::Affine);
        assert!(parse_config("optimizer = \"newton\"").is_err());
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let s = parse_config(
            "stack_every_nth = 0\n[preprocessing]\npre_rotation = 45\npre_resize_percent = 3\n\
             [image_saving]\nbit_depth = 12\njpeg_quality = 500\n",
        )
        .expect("clampable");
        assert_eq!(s.stack_every_nth, 1);
        assert_eq!(s.preprocessing.pre_rotation, 0);
        assert_eq!(s.preprocessing.pre_resize_percent, 10);
        assert_eq!(s.image_saving.bit_depth, 16);
        assert_eq!(s.image_saving.jpeg_quality, 100);
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("stacker_cli_cfg_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn parse_errors_and_missing_files_are_errors_not_defaults() {
        let dir = scratch("errors");

        let bad = dir.join("bad.toml");
        std::fs::write(&bad, "alignment_mode = \"sideways\"").unwrap();
        let err = load_config(Some(&bad), false).unwrap_err().to_string();
        assert!(err.contains("bad.toml"), "{err}");

        let missing = dir.join("typo.toml");
        let err = load_config(Some(&missing), false).unwrap_err().to_string();
        assert!(err.contains("does not exist"), "{err}");
        assert!(
            !missing.exists(),
            "a missing path must not be created implicitly"
        );

        // With the explicit flag the default file is written and used.
        let loaded = load_config(Some(&missing), true).expect("write default");
        assert_eq!(loaded, StackingSettings::default());
        assert!(missing.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
