//! Moving values between the Rust state and the Slint models: the source and
//! result lists and the settings dialog.

use std::{path::PathBuf, rc::Rc};

use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use stacker_gui::{
    settings::StackingSettings,
    settings_contract::{
        SettingsValues, alignment_options, compute_device_options, optimizer_options,
        output_format_options, tile_size_options,
    },
};

use crate::{App, SettingsModel};

// ── Slint list helpers ────────────────────────────────────────────────────────

fn display_names(paths: &[PathBuf]) -> ModelRc<SharedString> {
    let names: Vec<SharedString> = paths
        .iter()
        .map(|f| SharedString::from(f.file_name().unwrap_or_default().to_string_lossy().as_ref()))
        .collect();
    ModelRc::from(std::rc::Rc::new(VecModel::from(names)))
}

/// Show `paths` in the sidebar's source list and keep `has-images` and
/// `frame-count` in step with it.
pub fn push_source_panel(app: &App, paths: &[PathBuf]) {
    app.set_loaded_files(display_names(paths));
    app.set_has_images(!paths.is_empty());
    app.set_frame_count(i32::try_from(paths.len()).unwrap_or(i32::MAX));
}

/// Show `paths` in the sidebar's result list.
pub fn update_result_list(app: &App, paths: &[PathBuf]) {
    app.set_result_files(display_names(paths));
}

// ── Settings ↔ SettingsModel ──────────────────────────────────────────────────
//
// The pure mapping lives in `stacker_gui::settings_contract`; these functions
// only move a `SettingsValues` between the Slint global and Rust, one property
// per field, in exactly one place.

/// A Slint string model holding `items`.
fn string_model(items: &[&str]) -> ModelRc<SharedString> {
    let items: Vec<SharedString> = items.iter().map(|s| SharedString::from(*s)).collect();
    ModelRc::from(Rc::new(VecModel::from(items)))
}

/// Fill the combo-box entry lists the first time; later pushes leave them alone
/// so an open combo box is not rebuilt under the user's cursor.
fn push_option_lists(model: &SettingsModel<'_>) {
    if model.get_alignment_options().row_count() == 0 {
        model.set_alignment_options(string_model(&alignment_options()));
    }
    if model.get_optimizer_options().row_count() == 0 {
        model.set_optimizer_options(string_model(&optimizer_options()));
    }
    if model.get_tile_size_options().row_count() == 0 {
        let labels = tile_size_options();
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        model.set_tile_size_options(string_model(&labels));
    }
    if model.get_compute_device_options().row_count() == 0 {
        model.set_compute_device_options(string_model(&compute_device_options()));
    }
    if model.get_output_format_options().row_count() == 0 {
        model.set_output_format_options(string_model(&output_format_options()));
    }
}

/// Write `values` into the `SettingsModel` properties (no `settings-edited` is
/// fired: this is the program changing the dialog, not the user).
pub fn push_settings_values(app: &App, values: &SettingsValues) {
    let model = app.global::<SettingsModel>();
    push_option_lists(&model);

    model.set_alignment(values.alignment.as_str().into());
    model.set_optimizer(values.optimizer.as_str().into());
    model.set_bounded_search(values.bounded_search);
    model.set_akaze_seeding(values.akaze_seeding);
    model.set_correct_brightness(values.correct_brightness);
    model.set_chain_alignment(values.chain_alignment);
    model.set_neural_refine_classically(values.neural_refine_classically);
    model.set_auto_cull(values.auto_cull);
    model.set_auto_cull_threshold_pct(values.auto_cull_threshold_pct);
    model.set_sort_by_sharpness(values.sort_by_sharpness);
    model.set_stack_every_nth(values.stack_every_nth);

    model.set_pyramid_levels(values.pyramid_levels);
    model.set_use_all_color_channels(values.use_all_color_channels);
    model.set_grit_suppression(values.grit_suppression);
    model.set_compute_device(values.compute_device.as_str().into());
    model.set_tile_size(values.tile_size.as_str().into());

    model.set_relief_estimation_radius(values.relief_estimation_radius);
    model.set_relief_smoothing_radius(values.relief_smoothing_radius);
    model.set_relief_contrast_pct(values.relief_contrast_pct);
    model.set_relief_show_preview(values.relief_show_preview);
    model.set_relief_auto_detect(values.relief_auto_detect);
    model.set_relief_use_multigrid(values.relief_use_multigrid);

    model.set_strata_base_radius(values.strata_base_radius);
    model.set_strata_detail_focus(values.strata_detail_focus);
    model.set_crop_to_common_area(values.crop_to_common_area);
    model.set_resize_cropped_to_original(values.resize_cropped_to_original);

    model.set_pre_rotation(values.pre_rotation);
    model.set_pre_crop_enabled(values.pre_crop_enabled);
    model.set_pre_crop_spec(values.pre_crop_spec.as_str().into());
    model.set_pre_resize_percent(values.pre_resize_percent);
    model.set_sort_reverse(values.sort_reverse);
    model.set_ignore_exif(values.ignore_exif);

    model.set_output_format(values.output_format.as_str().into());
    model.set_bit_depth(values.bit_depth);
    model.set_jpeg_quality(values.jpeg_quality);
    model.set_filename_template(values.filename_template.as_str().into());
    model.set_default_output_dir(values.default_output_dir.as_str().into());
    model.set_copy_metadata(values.copy_metadata);
}

/// Read every `SettingsModel` property.
#[must_use]
pub fn read_settings_values(app: &App) -> SettingsValues {
    let model = app.global::<SettingsModel>();
    SettingsValues {
        alignment: model.get_alignment().to_string(),
        optimizer: model.get_optimizer().to_string(),
        bounded_search: model.get_bounded_search(),
        akaze_seeding: model.get_akaze_seeding(),
        correct_brightness: model.get_correct_brightness(),
        chain_alignment: model.get_chain_alignment(),
        neural_refine_classically: model.get_neural_refine_classically(),
        auto_cull: model.get_auto_cull(),
        auto_cull_threshold_pct: model.get_auto_cull_threshold_pct(),
        sort_by_sharpness: model.get_sort_by_sharpness(),
        stack_every_nth: model.get_stack_every_nth(),
        pyramid_levels: model.get_pyramid_levels(),
        use_all_color_channels: model.get_use_all_color_channels(),
        grit_suppression: model.get_grit_suppression(),
        relief_estimation_radius: model.get_relief_estimation_radius(),
        relief_smoothing_radius: model.get_relief_smoothing_radius(),
        relief_contrast_pct: model.get_relief_contrast_pct(),
        relief_show_preview: model.get_relief_show_preview(),
        relief_auto_detect: model.get_relief_auto_detect(),
        relief_use_multigrid: model.get_relief_use_multigrid(),
        strata_base_radius: model.get_strata_base_radius(),
        strata_detail_focus: model.get_strata_detail_focus(),
        crop_to_common_area: model.get_crop_to_common_area(),
        resize_cropped_to_original: model.get_resize_cropped_to_original(),
        compute_device: model.get_compute_device().to_string(),
        tile_size: model.get_tile_size().to_string(),
        pre_rotation: model.get_pre_rotation(),
        pre_crop_enabled: model.get_pre_crop_enabled(),
        pre_crop_spec: model.get_pre_crop_spec().to_string(),
        pre_resize_percent: model.get_pre_resize_percent(),
        sort_reverse: model.get_sort_reverse(),
        ignore_exif: model.get_ignore_exif(),
        output_format: model.get_output_format().to_string(),
        bit_depth: model.get_bit_depth(),
        jpeg_quality: model.get_jpeg_quality(),
        filename_template: model.get_filename_template().to_string(),
        default_output_dir: model.get_default_output_dir().to_string(),
        copy_metadata: model.get_copy_metadata(),
    }
}

/// Show `s` in the settings dialog.
pub fn push_settings_to_ui(app: &App, s: &StackingSettings) {
    push_settings_values(app, &SettingsValues::from(s));
}

/// The dialog's values as clamped [`StackingSettings`] (`sanitized()` applied).
#[must_use]
pub fn pull_settings_from_ui(app: &App) -> StackingSettings {
    read_settings_values(app).to_sanitized()
}
