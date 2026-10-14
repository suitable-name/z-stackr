//! The settings contract between the dialog and the stacking settings.
//!
//! The dialog edits 38 flat values (`SettingsModel` in `ui/models/settings.slint`);
//! the 39th settings key, `settings_version`, is written by the core and is not
//! a dialog value. Every control writes its property and fires one
//! `settings-edited()`; the Rust side then reads all properties into a
//! [`SettingsValues`], turns it into a [`StackingSettings`], runs `sanitized()`,
//! pushes the clamped values back and hands the result to the persister.
//!
//! This module is the pure half of that loop: the mapping between
//! [`StackingSettings`] and the flat values, with no Slint in sight, so it can be
//! unit-tested headless. `ui_helpers.rs` moves a [`SettingsValues`] to and from
//! the `SettingsModel` global.
//!
//! The Slint tree carries no setting default of its own. Every value, the reset
//! included, comes from [`StackingSettings::default`] by way of
//! [`SettingsValues::dialog_reset`] or [`SettingsValues::from`].
//!
//! # What each build offers
//!
//! The combo lists are built here, so the dialog never offers a choice the build
//! cannot run:
//!
//! - `Neural` alignment: only with the `nn` feature and never in the browser
//!   build (`wasm32`).
//! - `GPU` compute device: only with the `gpu` feature.
//! - The optimizer and the tile size are listed everywhere; the dialog hides
//!   both when `AppModel.web-build` is set (the browser always searches with
//!   Nelder-Mead and has no scratch disk), while a value read from a file still
//!   round-trips.

use crate::settings::{
    AlignmentModeSetting, ComputeDevice, ImageSavingSettings, OptimizerSetting, OutputFormat,
    PreprocessingSettings, SETTINGS_VERSION, StackingSettings,
};

/// The combo text of tile size 0: the whole stack stays in RAM.
pub const TILE_SIZE_IN_MEMORY: &str = "In Memory";

/// The unit written after a tile edge in the tile-size combo.
const TILE_SIZE_UNIT: &str = " px";

/// The tile edges (in pixels) the tile-size combo offers besides "In Memory".
const TILE_SIZE_CHOICES: [u32; 4] = [512, 1024, 2048, 4096];

/// The entries of the alignment combo box, in display order.
///
/// `Neural` is listed only in a build with the `nn` feature that is not the
/// browser build. A settings file that says `neural` in any other build is still
/// read and written back unchanged; the run then explains that the mode is
/// unavailable.
#[must_use]
pub fn alignment_options() -> Vec<&'static str> {
    let mut modes = vec![
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Registration,
        AlignmentModeSetting::Affine,
        AlignmentModeSetting::None,
    ];
    if cfg!(all(feature = "nn", not(target_arch = "wasm32"))) {
        modes.push(AlignmentModeSetting::Neural);
    }
    modes
        .into_iter()
        .map(AlignmentModeSetting::as_combo_str)
        .collect()
}

/// The entries of the optimizer combo box, in display order.
///
/// The dialog hides the box in the browser build (where `LucasKanade` behaves
/// like `Auto`); the list itself is the same everywhere.
#[must_use]
pub fn optimizer_options() -> Vec<&'static str> {
    [
        OptimizerSetting::Auto,
        OptimizerSetting::LucasKanade,
        OptimizerSetting::NelderMead,
    ]
    .map(OptimizerSetting::as_combo_str)
    .to_vec()
}

/// The entries of the compute-device combo box, in display order.
///
/// `GPU` is offered only in a build that has GPU support (the `gpu` feature);
/// without it the box lists the CPU alone. A settings file that says `gpu` is
/// still read and written back unchanged, and the run explains that this build
/// uses the CPU.
#[must_use]
pub fn compute_device_options() -> Vec<&'static str> {
    if cfg!(feature = "gpu") {
        vec![
            ComputeDevice::Cpu.as_combo_str(),
            ComputeDevice::Gpu.as_combo_str(),
        ]
    } else {
        vec![ComputeDevice::Cpu.as_combo_str()]
    }
}

/// The entries of the tile-size combo box, in display order: "In Memory" first,
/// then the offered tile edges ("512 px" .. "4096 px").
///
/// A value read from a file that is not in this list (for example 700) is shown
/// as it is and round-trips.
#[must_use]
pub fn tile_size_options() -> Vec<String> {
    std::iter::once(TILE_SIZE_IN_MEMORY.to_owned())
        .chain(TILE_SIZE_CHOICES.map(tile_size_label))
        .collect()
}

/// The combo text for a tile size: "In Memory" for 0, otherwise `"<n> px"`.
#[must_use]
pub fn tile_size_label(tile_size: u32) -> String {
    if tile_size == 0 {
        TILE_SIZE_IN_MEMORY.to_owned()
    } else {
        format!("{tile_size}{TILE_SIZE_UNIT}")
    }
}

/// The tile size a combo text stands for: `"<n> px"` (or a bare `"<n>"`) is `n`;
/// "In Memory" and anything unreadable is 0 (in RAM, the safe reading).
/// `sanitized()` later lifts a nonzero value below 64 to 64.
#[must_use]
pub fn tile_size_from_label(label: &str) -> u32 {
    let trimmed = label.trim();
    trimmed
        .strip_suffix(TILE_SIZE_UNIT.trim())
        .unwrap_or(trimmed)
        .trim()
        .parse()
        .unwrap_or(0)
}

/// The entries of the output-format combo box, in display order.
#[must_use]
pub fn output_format_options() -> Vec<&'static str> {
    [OutputFormat::Tiff, OutputFormat::Png, OutputFormat::Jpeg]
        .map(OutputFormat::as_combo_str)
        .to_vec()
}

/// Exact for every value the dialog can hold (all far below 2^24).
#[expect(
    clippy::cast_precision_loss,
    reason = "dialog values are small integers, exact in f32"
)]
const fn to_f32(value: u32) -> f32 {
    value as f32
}

/// Round a slider value to the nearest whole number. A float-to-integer `as`
/// cast saturates (negative and NaN become 0), which `sanitized()` then clamps
/// into the field's range.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the saturating cast is the intent; sanitized() clamps the result"
)]
const fn round_to_u32(value: f32) -> u32 {
    value.round() as u32
}

fn to_i32(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

/// The 38 dialog values in the types Slint holds them in: combo boxes as
/// strings, sliders as `f32`, the two discrete integers as `i32`.
///
/// The field names are the `SettingsModel` property names with `-` as `_`.
#[expect(
    clippy::struct_excessive_bools,
    reason = "one field per switch in the dialog; mirrors StackingSettings"
)]
#[derive(Debug, Clone, PartialEq)]
pub struct SettingsValues {
    /// The alignment mode, as the combo box spells it.
    pub alignment: String,
    /// The alignment optimizer, as the combo box spells it.
    pub optimizer: String,
    /// Limit the alignment search to a window around the seed.
    pub bounded_search: bool,
    /// Seed the alignment with AKAZE feature matches.
    pub akaze_seeding: bool,
    /// Match each aligned frame's brightness to the reference.
    pub correct_brightness: bool,
    /// Align every frame to its aligned predecessor.
    pub chain_alignment: bool,
    /// Polish the neural alignment's transforms with the classical optimizer.
    pub neural_refine_classically: bool,
    /// Leave out frames that add almost nothing.
    pub auto_cull: bool,
    /// The cull threshold in percent, 0 to 50; 0 is the automatic rule.
    pub auto_cull_threshold_pct: f32,
    /// Order the frames sharpest-first before alignment.
    pub sort_by_sharpness: bool,
    /// Use every N-th frame.
    pub stack_every_nth: f32,
    /// Where the alignment and fusions run, as the combo box spells it
    /// (`CPU` or `GPU`).
    pub compute_device: String,
    /// The out-of-core tile size, as the combo box spells it ("In Memory" or
    /// `"<n> px"`, see [`tile_size_label`]).
    pub tile_size: String,
    /// Pyramid levels of the `Apex` fusion.
    pub pyramid_levels: f32,
    /// Fuse on all colour channels, not luma alone.
    pub use_all_color_channels: bool,
    /// Suppress isolated bright specks in the fusion.
    pub grit_suppression: bool,
    /// Radius of the `Relief` focus estimate.
    pub relief_estimation_radius: f32,
    /// Radius of the `Relief` smoothing.
    pub relief_smoothing_radius: f32,
    /// The rejected fraction of the `Relief` contrast threshold, 0 to 1.
    pub relief_contrast_pct: f32,
    /// Show the contrast preview before the `Relief` fusion.
    pub relief_show_preview: bool,
    /// Suggest the contrast threshold automatically.
    pub relief_auto_detect: bool,
    /// Solve the `Relief` depth map with the multigrid solver.
    pub relief_use_multigrid: bool,
    /// Box radius of the `Strata` base/detail split, 8 to 64.
    pub strata_base_radius: f32,
    /// How strongly the `Strata` detail follows the sharpest frame, 1 to 5.
    pub strata_detail_focus: f32,
    /// Degrees: 0, 90, 180 or 270.
    pub pre_rotation: i32,
    /// Crop every frame before stacking.
    pub pre_crop_enabled: bool,
    /// The crop rectangle as text.
    pub pre_crop_spec: String,
    /// Downscale every frame to this percentage.
    pub pre_resize_percent: f32,
    /// Reverse the frame order.
    pub sort_reverse: bool,
    /// Ignore the EXIF orientation.
    pub ignore_exif: bool,
    /// Crop the result to the area every aligned frame covers.
    pub crop_to_common_area: bool,
    /// Resample a cropped result back to the original canvas size.
    pub resize_cropped_to_original: bool,
    /// The output file format, as the combo box spells it.
    pub output_format: String,
    /// 8 or 16.
    pub bit_depth: i32,
    /// JPEG quality, 1 to 100.
    pub jpeg_quality: f32,
    /// The output file name template.
    pub filename_template: String,
    /// The folder Save writes into without asking; empty asks each time.
    pub default_output_dir: String,
    /// Copy the EXIF metadata of the first frame to the output.
    pub copy_metadata: bool,
}

impl From<&StackingSettings> for SettingsValues {
    fn from(s: &StackingSettings) -> Self {
        let pre = &s.preprocessing;
        let saving = &s.image_saving;
        Self {
            alignment: s.alignment_mode.as_combo_str().to_owned(),
            optimizer: s.optimizer.as_combo_str().to_owned(),
            bounded_search: s.bounded_search,
            akaze_seeding: s.akaze_seeding,
            correct_brightness: s.correct_brightness,
            chain_alignment: s.chain_alignment,
            neural_refine_classically: s.neural_refine_classically,
            auto_cull: s.auto_cull,
            auto_cull_threshold_pct: s.auto_cull_threshold_pct,
            sort_by_sharpness: s.sort_by_sharpness,
            stack_every_nth: to_f32(s.stack_every_nth),
            compute_device: s.compute_device.as_combo_str().to_owned(),
            tile_size: tile_size_label(s.tile_size),
            pyramid_levels: to_f32(s.pyramid_levels),
            use_all_color_channels: s.use_all_color_channels,
            grit_suppression: s.grit_suppression,
            relief_estimation_radius: to_f32(s.relief_estimation_radius),
            relief_smoothing_radius: to_f32(s.relief_smoothing_radius),
            relief_contrast_pct: s.relief_contrast_pct,
            relief_show_preview: s.relief_show_preview,
            relief_auto_detect: s.relief_auto_detect,
            relief_use_multigrid: s.relief_use_multigrid,
            strata_base_radius: to_f32(s.strata_base_radius),
            strata_detail_focus: to_f32(s.strata_detail_focus),
            pre_rotation: to_i32(pre.pre_rotation),
            pre_crop_enabled: pre.pre_crop_enabled,
            pre_crop_spec: pre.pre_crop_spec.clone(),
            pre_resize_percent: to_f32(pre.pre_resize_percent),
            sort_reverse: pre.sort_reverse,
            ignore_exif: pre.ignore_exif_orientation,
            crop_to_common_area: s.crop_to_common_area,
            resize_cropped_to_original: s.resize_cropped_to_original,
            output_format: saving.output_format.as_combo_str().to_owned(),
            bit_depth: to_i32(saving.bit_depth),
            jpeg_quality: to_f32(saving.jpeg_quality),
            filename_template: saving.filename_template.clone(),
            default_output_dir: saving.default_output_dir.clone(),
            copy_metadata: saving.copy_metadata,
        }
    }
}

impl SettingsValues {
    /// What the dialog shows after "Reset to Defaults" and at start-up on a
    /// fresh install: the core defaults, nothing else.
    #[must_use]
    pub fn dialog_reset() -> Self {
        Self::from(&StackingSettings::default())
    }

    /// The settings these values describe, converted but not yet clamped: call
    /// [`StackingSettings::sanitized`] (or [`Self::to_sanitized`]) before using
    /// the result. An unknown combo string maps to the field's default.
    ///
    /// `settings_version` is always the current version: the dialog edits the
    /// current schema, so a value read from it needs no migration.
    #[must_use]
    pub fn to_settings(&self) -> StackingSettings {
        let preprocessing = PreprocessingSettings {
            pre_rotation: u32::try_from(self.pre_rotation).unwrap_or(0),
            pre_crop_enabled: self.pre_crop_enabled,
            pre_crop_spec: self.pre_crop_spec.clone(),
            pre_resize_percent: round_to_u32(self.pre_resize_percent),
            sort_reverse: self.sort_reverse,
            ignore_exif_orientation: self.ignore_exif,
        };
        let image_saving = ImageSavingSettings {
            output_format: OutputFormat::from_combo_str(&self.output_format),
            bit_depth: u32::try_from(self.bit_depth).unwrap_or(0),
            jpeg_quality: round_to_u32(self.jpeg_quality),
            filename_template: self.filename_template.clone(),
            default_output_dir: self.default_output_dir.clone(),
            copy_metadata: self.copy_metadata,
        };
        StackingSettings {
            settings_version: SETTINGS_VERSION,
            alignment_mode: AlignmentModeSetting::from_combo_str(&self.alignment),
            optimizer: OptimizerSetting::from_combo_str(&self.optimizer),
            bounded_search: self.bounded_search,
            akaze_seeding: self.akaze_seeding,
            correct_brightness: self.correct_brightness,
            chain_alignment: self.chain_alignment,
            neural_refine_classically: self.neural_refine_classically,
            crop_to_common_area: self.crop_to_common_area,
            resize_cropped_to_original: self.resize_cropped_to_original,
            auto_cull: self.auto_cull,
            auto_cull_threshold_pct: self.auto_cull_threshold_pct,
            sort_by_sharpness: self.sort_by_sharpness,
            stack_every_nth: round_to_u32(self.stack_every_nth),
            tile_size: tile_size_from_label(&self.tile_size),
            pyramid_levels: round_to_u32(self.pyramid_levels),
            use_all_color_channels: self.use_all_color_channels,
            grit_suppression: self.grit_suppression,
            relief_estimation_radius: round_to_u32(self.relief_estimation_radius),
            relief_smoothing_radius: round_to_u32(self.relief_smoothing_radius),
            relief_contrast_pct: self.relief_contrast_pct,
            relief_show_preview: self.relief_show_preview,
            relief_auto_detect: self.relief_auto_detect,
            relief_use_multigrid: self.relief_use_multigrid,
            strata_base_radius: round_to_u32(self.strata_base_radius),
            strata_detail_focus: round_to_u32(self.strata_detail_focus),
            compute_device: ComputeDevice::from_combo_str(&self.compute_device),
            preprocessing,
            image_saving,
        }
    }

    /// [`Self::to_settings`] followed by `sanitized()`: every numeric field is
    /// inside its documented range.
    #[must_use]
    pub fn to_sanitized(&self) -> StackingSettings {
        self.to_settings().sanitized().0
    }
}
