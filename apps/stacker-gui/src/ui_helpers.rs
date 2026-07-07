use std::path::PathBuf;

use slint::{ModelRc, SharedString, VecModel};

use crate::{
    App,
    mem_estimate::{estimate_peak_bytes, format_bytes, total_system_memory_bytes},
    settings::{
        AlignmentModeSetting, ImageSavingSettings, OptimizerSetting, OutputFormat,
        PreprocessingSettings, StackingSettings,
    },
};

// ── Slint list helpers ────────────────────────────────────────────────────────

pub fn update_source_list(app: &App, paths: &[PathBuf]) {
    let display_names: Vec<SharedString> = paths
        .iter()
        .map(|f| SharedString::from(f.file_name().unwrap_or_default().to_string_lossy().as_ref()))
        .collect();
    app.set_loaded_files(ModelRc::from(std::rc::Rc::new(VecModel::from(
        display_names,
    ))));
    // No culling applied here — reset the parallel flags model so a
    // previously-marked list (from a Cull run / cached Cull application)
    // doesn't leave stale "culled" marks on an unrelated file-list update
    // (e.g. Open Files, Remove, drag-reorder).
    app.set_loaded_files_culled(ModelRc::from(std::rc::Rc::new(VecModel::from(vec![
        false;
        paths
            .len(
            )
    ]))));
    update_memory_estimate(app, paths);
}

/// Recomputes the "Est. peak memory" line shown under the source-file list
/// and flags `memory-warning` when the estimate exceeds total system RAM.
///
/// This estimate is for the GUI's default in-RAM path (`tile_size == 0`).
/// When the user sets `tile_size > 0` in Settings, the Stack handler switches
/// to the shared out-of-core `stacker_pipeline::run_pipeline` engine instead
/// (see `main.rs::on_request_stack`), whose peak memory scales with tile
/// area rather than stack size — this estimate does not apply to that mode.
pub fn update_memory_estimate(app: &App, paths: &[PathBuf]) {
    let tile_size = app.get_set_tile_size();
    if tile_size > 0.0 {
        app.set_memory_estimate_text(SharedString::from(
            "Est. peak memory: bounded (tiled processing)",
        ));
        app.set_memory_warning(false);
        return;
    }

    // Cheap header-only dimension read of the first frame — good enough for
    // an order-of-magnitude estimate, since every frame in a real focus
    // stack comes from the same sensor and is the same size.
    let Some((w, h)) = paths.iter().find_map(|p| image::image_dimensions(p).ok()) else {
        app.set_memory_estimate_text(SharedString::new());
        app.set_memory_warning(false);
        return;
    };

    let akaze = app.get_set_akaze_seeding();
    let estimate = estimate_peak_bytes(paths.len(), w, h, akaze);
    let total = total_system_memory_bytes();
    let warn = total.is_some_and(|t| estimate > t);
    let text = total.map_or_else(
        || format!("Est. peak memory: ~{}", format_bytes(estimate)),
        |t| {
            let suffix = if warn {
                " — may exceed available RAM"
            } else {
                ""
            };
            format!(
                "Est. peak memory: ~{} (system RAM: {}){suffix}",
                format_bytes(estimate),
                format_bytes(t)
            )
        },
    );
    app.set_memory_warning(warn);
    app.set_memory_estimate_text(SharedString::from(text));
}

/// Like [`update_source_list`], but also populates the parallel
/// `loaded-files-culled` flags model.
///
/// Used after the Cull button runs, or after Stack applies a cached Cull
/// result. The Slint side (`Sidebar`'s file-list `for`) renders a `true`
/// entry dimmed with a "(culled)" suffix; this function only supplies plain
/// display names plus the flags. `culled` is compared by path against
/// `paths`, not by index, so it is robust to any prior manual reordering of
/// `paths` relative to when the culled set was computed.
pub fn update_source_list_marked(app: &App, paths: &[PathBuf], culled: &[PathBuf]) {
    let display_names: Vec<SharedString> = paths
        .iter()
        .map(|f| SharedString::from(f.file_name().unwrap_or_default().to_string_lossy().as_ref()))
        .collect();
    let culled_flags: Vec<bool> = paths.iter().map(|f| culled.contains(f)).collect();
    app.set_loaded_files(ModelRc::from(std::rc::Rc::new(VecModel::from(
        display_names,
    ))));
    app.set_loaded_files_culled(ModelRc::from(std::rc::Rc::new(VecModel::from(
        culled_flags,
    ))));
    update_memory_estimate(app, paths);
}

pub fn update_result_list(app: &App, paths: &[PathBuf]) {
    let display_names: Vec<SharedString> = paths
        .iter()
        .map(|f| SharedString::from(f.file_name().unwrap_or_default().to_string_lossy().as_ref()))
        .collect();
    app.set_result_files(ModelRc::from(std::rc::Rc::new(VecModel::from(
        display_names,
    ))));
}

// ── Settings → UI helpers ─────────────────────────────────────────────────────

/// Push every field of `s` into the App's `set-*` properties.
///
/// The slider-bound settings are stored as `float` in the Slint component.
/// All values fit in f32 exactly: the largest integer field is `relief_estimation_radius`
/// capped at 100, well within the 23-bit mantissa.
pub fn push_settings_to_ui(app: &App, s: &StackingSettings) {
    // ── Existing fields ───────────────────────────────────────────────────
    app.set_set_alignment(SharedString::from(s.alignment_mode.as_combo_str()));
    app.set_set_optimizer(SharedString::from(s.optimizer.as_combo_str()));
    app.set_set_akaze_seeding(s.akaze_seeding);
    app.set_set_neural_refine_classically(s.neural_refine_classically);
    app.set_set_correct_brightness(s.correct_brightness);
    app.set_set_auto_cull(s.auto_cull);
    app.set_set_sort_by_sharpness(s.sort_by_sharpness);
    app.set_set_auto_cull_threshold_pct(s.auto_cull_threshold_pct);
    app.set_set_crop_to_common_area(s.crop_to_common_area);
    app.set_set_resize_cropped_to_original(s.resize_cropped_to_original);
    app.set_set_use_gpu(s.use_gpu);
    app.set_set_stack_every_nth(s.stack_every_nth as f32);
    app.set_set_tile_size(s.tile_size as f32);
    app.set_set_pyramid_levels(s.pyramid_levels as f32);
    app.set_set_use_all_color_channels(s.use_all_color_channels);
    app.set_set_grit_suppression(s.grit_suppression);
    app.set_set_relief_estimation_radius(s.relief_estimation_radius as f32);
    app.set_set_relief_smoothing_radius(s.relief_smoothing_radius as f32);
    app.set_set_relief_contrast_pct(s.relief_contrast_pct);
    app.set_set_relief_show_preview(s.relief_show_preview);
    app.set_set_relief_auto_detect(s.relief_auto_detect);
    app.set_set_relief_use_multigrid(s.relief_use_multigrid);
    app.set_set_strata_base_radius(s.strata_base_radius as f32);
    app.set_set_strata_detail_focus(s.strata_detail_focus as f32);

    // ── Preprocessing ─────────────────────────────────────────────────────
    let pre = &s.preprocessing;

    app.set_set_pre_rotation(pre.pre_rotation as i32);
    app.set_set_pre_crop_enabled(pre.pre_crop_enabled);
    app.set_set_pre_crop_spec(SharedString::from(pre.pre_crop_spec.as_str()));
    app.set_set_pre_resize_percent(pre.pre_resize_percent as f32);
    app.set_set_sort_reverse(pre.sort_reverse);
    app.set_set_ignore_exif(pre.ignore_exif_orientation);

    // ── Image saving ──────────────────────────────────────────────────────
    let sav = &s.image_saving;
    app.set_set_output_format(SharedString::from(sav.output_format.as_combo_str()));

    app.set_set_bit_depth(sav.bit_depth as i32);
    app.set_set_jpeg_quality(sav.jpeg_quality as f32);
    app.set_set_filename_template(SharedString::from(sav.filename_template.as_str()));
    app.set_set_default_output_dir(SharedString::from(sav.default_output_dir.as_str()));
    app.set_set_copy_metadata(sav.copy_metadata);
}

/// Read the App's `set-*` properties back into a [`StackingSettings`].
/// Slider values are `float`; we round to the nearest integer where applicable.
#[must_use]
pub fn pull_settings_from_ui(app: &App) -> StackingSettings {
    let preprocessing = PreprocessingSettings {
        pre_rotation: app.get_set_pre_rotation().max(0) as u32,
        pre_crop_enabled: app.get_set_pre_crop_enabled(),
        pre_crop_spec: app.get_set_pre_crop_spec().to_string(),
        pre_resize_percent: (app.get_set_pre_resize_percent().round() as u32).clamp(10, 100),
        sort_reverse: app.get_set_sort_reverse(),
        ignore_exif_orientation: app.get_set_ignore_exif(),
    };

    let image_saving = ImageSavingSettings {
        output_format: OutputFormat::from_combo_str(app.get_set_output_format().as_str()),
        bit_depth: app.get_set_bit_depth().max(0) as u32,
        jpeg_quality: (app.get_set_jpeg_quality().round() as u32).clamp(1, 100),
        filename_template: app.get_set_filename_template().to_string(),
        default_output_dir: app.get_set_default_output_dir().to_string(),
        copy_metadata: app.get_set_copy_metadata(),
    };

    let mut s = StackingSettings {
        alignment_mode: AlignmentModeSetting::from_combo_str(app.get_set_alignment().as_str()),
        optimizer: OptimizerSetting::from_combo_str(app.get_set_optimizer().as_str()),
        akaze_seeding: app.get_set_akaze_seeding(),
        neural_refine_classically: app.get_set_neural_refine_classically(),
        correct_brightness: app.get_set_correct_brightness(),
        auto_cull: app.get_set_auto_cull(),
        sort_by_sharpness: app.get_set_sort_by_sharpness(),
        auto_cull_threshold_pct: ((app.get_set_auto_cull_threshold_pct() * 10.0).round() / 10.0)
            .clamp(0.1, 5.0),
        crop_to_common_area: app.get_set_crop_to_common_area(),
        resize_cropped_to_original: app.get_set_resize_cropped_to_original(),
        use_gpu: app.get_set_use_gpu(),
        tile_size: app.get_set_tile_size().round().max(0.0) as u32,
        stack_every_nth: (app.get_set_stack_every_nth().round() as u32).max(1),
        pyramid_levels: (app.get_set_pyramid_levels().round() as u32).max(2),
        use_all_color_channels: app.get_set_use_all_color_channels(),
        grit_suppression: app.get_set_grit_suppression(),
        relief_estimation_radius: app.get_set_relief_estimation_radius().round() as u32,
        relief_smoothing_radius: app.get_set_relief_smoothing_radius().round() as u32,
        relief_contrast_pct: app.get_set_relief_contrast_pct(),
        relief_show_preview: app.get_set_relief_show_preview(),
        relief_auto_detect: app.get_set_relief_auto_detect(),
        relief_use_multigrid: app.get_set_relief_use_multigrid(),
        strata_base_radius: app.get_set_strata_base_radius().round() as u32,
        strata_detail_focus: app.get_set_strata_detail_focus().round() as u32,
        preprocessing,
        image_saving,
    };
    s.clamp_valid();
    s.preprocessing.clamp_valid();
    s.image_saving.clamp_valid();
    s
}
