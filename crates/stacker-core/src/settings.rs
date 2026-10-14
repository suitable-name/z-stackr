//! Every user-facing option of a stacking run, and the lenient TOML loader that
//! both applications share.
//!
//! [`StackingSettings`] is the root: it serialises to the settings file the
//! applications write ([`DEFAULT_CONFIG_TOML`] is the annotated template) and
//! loads through [`StackingSettings::from_toml_str`], which never fails on a bad
//! value (it keeps the default and reports a warning), and
//! [`StackingSettings::sanitized`], which clamps every numeric field into the
//! range the template documents.

use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned};

/// The settings-file schema version this build writes.
///
/// Version 2 is the first with the current meaning of [`AlignmentModeSetting`]
/// (`registration` = 4-DOF similarity search, `affine` = full 6-DOF). A file
/// without the key, or with a smaller number, is *legacy* and is migrated on
/// load by [`StackingSettings::migrate_legacy`].
pub const SETTINGS_VERSION: u32 = 2;

/// What `settings_version` reads as when the key is absent from a file: `0`,
/// "older than any versioned file", so the legacy migration runs.
const fn legacy_settings_version() -> u32 {
    0
}

/// How frames are registered to the reference before fusion.
///
/// The classical modes all minimise the RMS difference to the reference over a
/// coarse-to-fine luma pyramid; they differ in the degrees of freedom they
/// search. Before settings version 2 this enum had other meanings for
/// `affine` and `registration` (see [`StackingSettings::migrate_legacy`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AlignmentModeSetting {
    /// Shift plus one uniform scale (3 degrees of freedom, no rotation), so
    /// focus breathing is corrected.
    Translation,
    /// Shift, scale and rotation (a 4-DOF similarity transform), coarse to fine
    /// over a luma pyramid; the default.
    #[default]
    Registration,
    /// The full 6-DOF transform: shift, scale, rotation and, in addition, an
    /// anisotropic scale (aspect) and shear. Use it when the lens or a
    /// tilting rail changes the aspect between frames.
    Affine,
    /// Skip alignment: every frame is used as it is.
    None,
    /// Learned alignment matrices from a neural model. Only builds with the
    /// `nn` feature can run it; the variant exists in every build so a settings
    /// file naming it round-trips, and an application without `nn` rejects it
    /// with a clear message at run time.
    Neural,
}

/// Case-insensitive: `"registration"`, `"Registration"` and `"REGISTRATION"`
/// all parse, so a value copied from the GUI combo box (`as_combo_str`) is
/// accepted in a file. Serialisation stays lowercase.
impl<'de> Deserialize<'de> for AlignmentModeSetting {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "translation" => Ok(Self::Translation),
            "registration" => Ok(Self::Registration),
            "affine" => Ok(Self::Affine),
            "none" => Ok(Self::None),
            "neural" => Ok(Self::Neural),
            _ => Err(serde::de::Error::unknown_variant(
                &raw,
                &["translation", "registration", "affine", "none", "neural"],
            )),
        }
    }
}

impl AlignmentModeSetting {
    /// The capitalised label a GUI combo box shows for this mode.
    #[must_use]
    pub const fn as_combo_str(self) -> &'static str {
        match self {
            Self::Translation => "Translation",
            Self::Registration => "Registration",
            Self::Affine => "Affine",
            Self::None => "None",
            Self::Neural => "Neural",
        }
    }

    /// The mode for a combo-box label ([`Self::as_combo_str`]); an unknown label
    /// yields the default, [`Self::Registration`].
    #[must_use]
    pub fn from_combo_str(s: &str) -> Self {
        match s {
            "Translation" => Self::Translation,
            "Affine" => Self::Affine,
            "None" => Self::None,
            "Neural" => Self::Neural,
            _ => Self::Registration,
        }
    }
}

/// Which intensity optimiser the classical alignment modes use.
///
/// Lucas-Kanade is a CPU path. The browser build always searches with
/// Nelder-Mead (its search is driven step by step from the page), so there
/// `LucasKanade` behaves like `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerSetting {
    /// Lucas-Kanade first; when it fails or ends worse than it started, the
    /// bounded Nelder-Mead simplex from the same seed. The default.
    #[default]
    Auto,
    /// Gauss-Newton Lucas-Kanade only (fast, needs a good seed).
    #[serde(rename = "lucaskanade")]
    LucasKanade,
    /// The bounded Nelder-Mead simplex only (slower, robust).
    #[serde(rename = "neldermead")]
    NelderMead,
}

/// Case-insensitive; also accepts `"lk"`, `"lucas-kanade"`, `"lucas_kanade"`,
/// `"nm"`, `"nelder-mead"` and `"nelder_mead"`. Serialisation stays
/// `auto` / `lucaskanade` / `neldermead`.
impl<'de> Deserialize<'de> for OptimizerSetting {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "lucaskanade" | "lk" | "lucas-kanade" | "lucas_kanade" => Ok(Self::LucasKanade),
            "neldermead" | "nm" | "nelder-mead" | "nelder_mead" => Ok(Self::NelderMead),
            _ => Err(serde::de::Error::unknown_variant(
                &raw,
                &["auto", "lucaskanade", "neldermead"],
            )),
        }
    }
}

impl OptimizerSetting {
    /// The label a GUI combo box shows for this optimiser.
    #[must_use]
    pub const fn as_combo_str(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::LucasKanade => "Lucas-Kanade",
            Self::NelderMead => "Nelder-Mead",
        }
    }

    /// The optimiser for a combo-box label ([`Self::as_combo_str`]); an
    /// unknown label yields the default, [`Self::Auto`].
    #[must_use]
    pub fn from_combo_str(s: &str) -> Self {
        match s {
            "Lucas-Kanade" => Self::LucasKanade,
            "Nelder-Mead" => Self::NelderMead,
            _ => Self::Auto,
        }
    }
}

/// Where the `Apex` fusion runs.
///
/// The CPU is the reference. The GPU path (a build with the `gpu` feature) agrees
/// with it within a small tolerance and falls back to the CPU whenever the device
/// is missing, too small or fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    /// Always the CPU (rayon and `std::simd`); the default.
    #[default]
    Cpu,
    /// The GPU when this build has GPU support and a usable adapter exists;
    /// otherwise, or after any device failure, the CPU with a warning.
    Gpu,
}

/// Case-insensitive (`"cpu"`, `"CPU"`); serialisation stays lowercase.
impl<'de> Deserialize<'de> for ComputeDevice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => Err(serde::de::Error::unknown_variant(&raw, &["cpu", "gpu"])),
        }
    }
}

impl ComputeDevice {
    /// The upper-case label a GUI combo box shows for this device.
    #[must_use]
    pub const fn as_combo_str(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
        }
    }

    /// The device for a combo-box label ([`Self::as_combo_str`]); an unknown label
    /// yields the default, [`Self::Cpu`].
    #[must_use]
    pub fn from_combo_str(s: &str) -> Self {
        match s {
            "GPU" => Self::Gpu,
            _ => Self::Cpu,
        }
    }
}

/// File format of the saved result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// TIFF (8 or 16 bit); the default.
    #[default]
    Tiff,
    /// PNG (8 or 16 bit).
    Png,
    /// JPEG (8 bit, lossy, controlled by `jpeg_quality`).
    Jpeg,
}

/// Case-insensitive (`"tiff"`, `"TIFF"`, `"Tiff"`); `"tif"` and `"jpg"` are
/// accepted as aliases. Serialisation stays lowercase.
impl<'de> Deserialize<'de> for OutputFormat {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "tiff" | "tif" => Ok(Self::Tiff),
            "png" => Ok(Self::Png),
            "jpeg" | "jpg" => Ok(Self::Jpeg),
            _ => Err(serde::de::Error::unknown_variant(
                &raw,
                &["tiff", "png", "jpeg"],
            )),
        }
    }
}

impl OutputFormat {
    /// The upper-case label a GUI combo box shows for this format.
    #[must_use]
    pub const fn as_combo_str(self) -> &'static str {
        match self {
            Self::Tiff => "TIFF",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
        }
    }

    /// The format for a combo-box label ([`Self::as_combo_str`]); an unknown label
    /// yields the default, [`Self::Tiff`].
    #[must_use]
    pub fn from_combo_str(s: &str) -> Self {
        match s {
            "PNG" => Self::Png,
            "JPEG" => Self::Jpeg,
            _ => Self::Tiff,
        }
    }
}

/// What is done to every frame while it is loaded, before alignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PreprocessingSettings {
    /// Clockwise rotation in degrees: 0, 90, 180 or 270.
    pub pre_rotation: u32,
    /// Whether the crop in `pre_crop_spec` is applied.
    pub pre_crop_enabled: bool,
    /// Crop rectangle as `"w,h"` (centred) or `"x,y,w,h"`.
    pub pre_crop_spec: String,
    /// Resize every frame to this percentage of its size, `10..=100`.
    pub pre_resize_percent: u32,
    /// Process the frames in reverse file order.
    pub sort_reverse: bool,
    /// Ignore the EXIF orientation tag when decoding.
    pub ignore_exif_orientation: bool,
}

impl Default for PreprocessingSettings {
    fn default() -> Self {
        Self {
            pre_rotation: 0,
            pre_crop_enabled: false,
            pre_crop_spec: String::new(),
            pre_resize_percent: 100,
            sort_reverse: false,
            ignore_exif_orientation: false,
        }
    }
}

impl PreprocessingSettings {
    /// Reset an unsupported rotation to 0 and clamp the resize percentage to
    /// `10..=100`, silently (the warning-producing form is
    /// [`StackingSettings::sanitized`]).
    pub fn clamp_valid(&mut self) {
        if !matches!(self.pre_rotation, 0 | 90 | 180 | 270) {
            self.pre_rotation = 0;
        }
        self.pre_resize_percent = self.pre_resize_percent.clamp(10, 100);
    }
}

/// How the fused result is written to disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageSavingSettings {
    /// File format of the result.
    pub output_format: OutputFormat,
    /// Bits per sample: 8 or 16.
    pub bit_depth: u32,
    /// JPEG quality, `1..=100`; ignored for the other formats.
    pub jpeg_quality: u32,
    /// Output file name without extension; `{name}` stands for the base name of
    /// the first input frame.
    pub filename_template: String,
    /// Directory results go to; empty means the application chooses.
    pub default_output_dir: String,
    /// Reserved: copying EXIF/XMP metadata from the first frame is not
    /// implemented in either application (the CLI warns when it is set).
    pub copy_metadata: bool,
}

impl Default for ImageSavingSettings {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Tiff,
            bit_depth: 16,
            jpeg_quality: 95,
            filename_template: "{name}_stacked".to_owned(),
            default_output_dir: String::new(),
            copy_metadata: false,
        }
    }
}

impl ImageSavingSettings {
    /// Reset an unsupported bit depth to 16 and clamp the JPEG quality to
    /// `1..=100`, silently (the warning-producing form is
    /// [`StackingSettings::sanitized`]).
    pub fn clamp_valid(&mut self) {
        if !matches!(self.bit_depth, 8 | 16) {
            self.bit_depth = 16;
        }
        self.jpeg_quality = self.jpeg_quality.clamp(1, 100);
    }
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "independent user-facing toggles mirrored one-to-one in the settings file and the GUI"
)]
/// The complete option set of one stacking run.
///
/// Fields map one-to-one onto the keys of the settings file
/// ([`DEFAULT_CONFIG_TOML`] documents each with its range); the nested
/// `preprocessing` and `image_saving` tables are their own structs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StackingSettings {
    /// Schema version of the settings file ([`SETTINGS_VERSION`] when written).
    /// A file that lacks the key reads as `0` (legacy) and is migrated by
    /// [`Self::migrate_legacy`], after which this field holds
    /// [`SETTINGS_VERSION`].
    #[serde(default = "legacy_settings_version")]
    pub settings_version: u32,
    /// Registration model applied before fusion.
    pub alignment_mode: AlignmentModeSetting,
    /// Intensity optimiser of the classical alignment modes.
    pub optimizer: OptimizerSetting,
    /// Constrain the intensity optimiser to a bounded band around the initial
    /// estimate (a logit-space simplex over fixed bands) instead of an
    /// unbounded search. Guarantees a non-degenerate transform.
    pub bounded_search: bool,
    /// Seed the intensity optimiser with an AKAZE feature-match estimate before
    /// refinement. When `false` (default) registration is pure intensity-based.
    /// Requires the `akaze` build feature in the consuming application.
    pub akaze_seeding: bool,
    /// Match each aligned frame's brightness to the reference (computed in
    /// linear light) so exposure drift between frames does not leak into the
    /// fusion.
    pub correct_brightness: bool,
    /// Align every frame to its aligned predecessor (a rolling reference)
    /// instead of to the single reference frame. Helps long stacks whose end
    /// frames share little with the reference; costs accumulated drift.
    pub chain_alignment: bool,
    /// With `alignment_mode = "neural"`: use the network's matrices as seeds
    /// for the classical refinement (`true`) or apply them directly without
    /// refinement (`false`). No effect for any other mode.
    pub neural_refine_classically: bool,
    /// Crop the result to the largest rectangle that every aligned frame
    /// covers. Local builds crop whenever the frames were aligned; a crop that
    /// would keep less than half the canvas is refused as a rogue frame. With
    /// `false` the full canvas is kept and the border shows edge-clamped
    /// data. Default `true`, which is what earlier versions always did.
    pub crop_to_common_area: bool,
    /// After a common-area crop, resample the result back to the original
    /// canvas size. A slight non-uniform stretch; ignored when no crop was
    /// applied or `crop_to_common_area` is `false`. Default `false`.
    pub resize_cropped_to_original: bool,
    /// Drop frames whose focus census shows they never contribute, before
    /// fusing.
    pub auto_cull: bool,
    /// Percentage of detail pixels a frame must win to survive the cull,
    /// `0.0..=50.0`. `0.0` (default) uses the automatic rule that scales the
    /// threshold with the frame count.
    pub auto_cull_threshold_pct: f32,
    /// Order the frames sharpest-first (the order of the focus census) before
    /// alignment, so the sharpest frame becomes the reference.
    pub sort_by_sharpness: bool,
    /// Use only every N-th frame of the loaded set (`>= 1`; 1 keeps all).
    pub stack_every_nth: u32,
    /// Edge length in pixels of the out-of-core tiles of the desktop
    /// application: `0` (default) keeps the stack in RAM, otherwise `64..=8192`.
    /// The command line has its own `--tile-size` flag, which always wins over
    /// this key.
    pub tile_size: u32,
    /// Laplacian-pyramid depth of the `Apex` fusion, `2..=12`.
    pub pyramid_levels: u32,
    /// Include the chroma planes in the `Apex` selection energy instead of luma
    /// only. The colour of a pixel is still copied from one winning frame.
    pub use_all_color_channels: bool,
    /// At the finest `Apex` band, choose the winner by the energy summed over a
    /// 3x3 neighbourhood instead of the pixel alone.
    pub grit_suppression: bool,
    /// Window radius in pixels of the `Relief` plane-fit focus measure, `0..=100`.
    #[serde(alias = "dmap_estimation_radius")]
    pub relief_estimation_radius: u32,
    /// Guided-filter window radius in pixels for the `Relief` depth-map smoothing,
    /// `0..=50`.
    #[serde(alias = "dmap_smoothing_radius")]
    pub relief_smoothing_radius: u32,
    /// Percentile in `0.0..=1.0` below which the `Relief` focus measure is treated
    /// as no signal.
    #[serde(alias = "dmap_contrast_pct")]
    pub relief_contrast_pct: f32,
    /// Show a live preview of the `Relief` contrast mask (GUI).
    #[serde(alias = "dmap_show_preview")]
    pub relief_show_preview: bool,
    /// Choose the `Relief` contrast threshold automatically when previewing (GUI).
    #[serde(alias = "dmap_auto_detect")]
    pub relief_auto_detect: bool,
    /// Solve the `Relief` depth map with the multigrid solver instead of the
    /// guided-filter argmax. Default `false`.
    pub relief_use_multigrid: bool,
    /// Box radius in pixels that splits the base and detail layers of the
    /// `Strata` fusion, `8..=64`.
    pub strata_base_radius: u32,
    /// How strongly the `Strata` detail weights concentrate on the sharpest
    /// frame, `1..=5` (1 soft, 5 close to winner-take-all).
    pub strata_detail_focus: u32,
    /// Where the alignment and the `Apex` and `Relief` fusions run: the CPU
    /// (default) or, in a build with GPU support, the GPU with a CPU fallback.
    /// The AI mode has its own device choice.
    pub compute_device: ComputeDevice,
    /// Per-frame loading options.
    pub preprocessing: PreprocessingSettings,
    /// Output file options.
    pub image_saving: ImageSavingSettings,
}

impl Default for StackingSettings {
    fn default() -> Self {
        Self {
            settings_version: SETTINGS_VERSION,
            alignment_mode: AlignmentModeSetting::Registration,
            optimizer: OptimizerSetting::Auto,
            bounded_search: false,
            akaze_seeding: false,
            correct_brightness: true,
            chain_alignment: false,
            neural_refine_classically: true,
            crop_to_common_area: true,
            resize_cropped_to_original: false,
            auto_cull: true,
            auto_cull_threshold_pct: 0.0,
            sort_by_sharpness: false,
            stack_every_nth: 1,
            tile_size: 0,
            pyramid_levels: 8,
            use_all_color_channels: false,
            grit_suppression: true,
            relief_estimation_radius: 5,
            relief_smoothing_radius: 2,
            relief_contrast_pct: 0.0,
            relief_show_preview: false,
            relief_auto_detect: false,
            relief_use_multigrid: false,
            strata_base_radius: 31,
            strata_detail_focus: 3,
            compute_device: ComputeDevice::Cpu,
            preprocessing: PreprocessingSettings::default(),
            image_saving: ImageSavingSettings::default(),
        }
    }
}

/// Read `key` out of `table` into `dst`. A value of the wrong type or an
/// invalid enum string leaves `dst` at its current (default) value and records
/// a warning instead of failing the whole file.
fn take_field<T: DeserializeOwned>(
    table: &mut toml::Table,
    section: &str,
    key: &str,
    dst: &mut T,
    warnings: &mut Vec<String>,
) {
    if let Some(value) = table.remove(key) {
        let parsed: Result<T, toml::de::Error> = value.try_into();
        match parsed {
            Ok(v) => *dst = v,
            Err(e) => warnings.push(format!(
                "{section}{key}: {}; keeping the default",
                e.message()
            )),
        }
    }
}

/// One warning per key left in `table` after every known key was taken.
fn warn_unknown(table: &toml::Table, section: &str, warnings: &mut Vec<String>) {
    for key in table.keys() {
        warnings.push(format!("{section}{key}: unknown key, ignored"));
    }
}

/// Clamp `value` into `lo..=hi`; when it moved, record a warning naming `range`.
fn clamp_u32(
    name: &str,
    value: &mut u32,
    (lo, hi): (u32, u32),
    range: &str,
    warnings: &mut Vec<String>,
) {
    let clamped = (*value).clamp(lo, hi);
    if clamped != *value {
        warnings.push(format!(
            "{name} = {} is outside {range}; using {clamped}",
            *value
        ));
        *value = clamped;
    }
}

impl PreprocessingSettings {
    fn apply_table(&mut self, mut table: toml::Table, warnings: &mut Vec<String>) {
        let s = "preprocessing.";
        take_field(
            &mut table,
            s,
            "pre_rotation",
            &mut self.pre_rotation,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "pre_crop_enabled",
            &mut self.pre_crop_enabled,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "pre_crop_spec",
            &mut self.pre_crop_spec,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "pre_resize_percent",
            &mut self.pre_resize_percent,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "sort_reverse",
            &mut self.sort_reverse,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "ignore_exif_orientation",
            &mut self.ignore_exif_orientation,
            warnings,
        );
        warn_unknown(&table, s, warnings);
    }
}

impl ImageSavingSettings {
    fn apply_table(&mut self, mut table: toml::Table, warnings: &mut Vec<String>) {
        let s = "image_saving.";
        take_field(
            &mut table,
            s,
            "output_format",
            &mut self.output_format,
            warnings,
        );
        take_field(&mut table, s, "bit_depth", &mut self.bit_depth, warnings);
        take_field(
            &mut table,
            s,
            "jpeg_quality",
            &mut self.jpeg_quality,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "filename_template",
            &mut self.filename_template,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "default_output_dir",
            &mut self.default_output_dir,
            warnings,
        );
        take_field(
            &mut table,
            s,
            "copy_metadata",
            &mut self.copy_metadata,
            warnings,
        );
        warn_unknown(&table, s, warnings);
    }
}

impl StackingSettings {
    /// Parse a settings file leniently.
    ///
    /// Only a TOML *syntax* error fails the load (`Err` carries the parser's
    /// message). Every other problem is local to its key: a value of the wrong
    /// type or an unknown enum string keeps that key's default, an unknown key
    /// is ignored, and each case adds one human-readable line to the returned
    /// warnings so the caller can show it. Enum strings are case-insensitive.
    /// The legacy spellings of [`apply_legacy_aliases`] are accepted (the new
    /// name wins when both are present), and a file without `settings_version`
    /// is migrated by [`Self::migrate_legacy`]; neither produces a warning.
    /// The result is passed through [`Self::sanitized`], whose clamp warnings
    /// are appended.
    ///
    /// # Errors
    /// Returns the TOML parser's message when `text` is not valid TOML.
    pub fn from_toml_str(text: &str) -> Result<(Self, Vec<String>), String> {
        let mut table: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
        apply_legacy_aliases(&mut table);
        let mut warnings = Vec::new();
        let mut s = Self::default();
        if !table.contains_key("settings_version") {
            s.settings_version = legacy_settings_version();
            // The default of a legacy file was the 4-DOF `affine` (kept when
            // the key is absent or unreadable), which the migration maps to
            // today's default.
            s.alignment_mode = AlignmentModeSetting::Affine;
        }
        s.take_top_level_fields(&mut table, &mut warnings);

        match table.remove("preprocessing") {
            Some(toml::Value::Table(t)) => s.preprocessing.apply_table(t, &mut warnings),
            Some(other) => warnings.push(format!(
                "preprocessing: expected a table, found {}; keeping the defaults",
                other.type_str()
            )),
            None => {}
        }
        match table.remove("image_saving") {
            Some(toml::Value::Table(t)) => s.image_saving.apply_table(t, &mut warnings),
            Some(other) => warnings.push(format!(
                "image_saving: expected a table, found {}; keeping the defaults",
                other.type_str()
            )),
            None => {}
        }
        warn_unknown(&table, "", &mut warnings);

        let (clean, clamp_warnings) = s.sanitized();
        warnings.extend(clamp_warnings);
        Ok((clean, warnings))
    }

    /// Move every top-level scalar key of `table` into `self`, one warning per
    /// wrong-typed or unknown-enum value (the key then keeps its default).
    #[expect(
        clippy::too_many_lines,
        reason = "one flat `take_field` call per settings key; splitting would only hide the list"
    )]
    fn take_top_level_fields(&mut self, table: &mut toml::Table, warnings: &mut Vec<String>) {
        take_field(
            table,
            "",
            "settings_version",
            &mut self.settings_version,
            warnings,
        );
        take_field(
            table,
            "",
            "alignment_mode",
            &mut self.alignment_mode,
            warnings,
        );
        take_field(table, "", "optimizer", &mut self.optimizer, warnings);
        take_field(
            table,
            "",
            "bounded_search",
            &mut self.bounded_search,
            warnings,
        );
        take_field(
            table,
            "",
            "akaze_seeding",
            &mut self.akaze_seeding,
            warnings,
        );
        take_field(
            table,
            "",
            "correct_brightness",
            &mut self.correct_brightness,
            warnings,
        );
        take_field(
            table,
            "",
            "chain_alignment",
            &mut self.chain_alignment,
            warnings,
        );
        take_field(
            table,
            "",
            "neural_refine_classically",
            &mut self.neural_refine_classically,
            warnings,
        );
        take_field(
            table,
            "",
            "crop_to_common_area",
            &mut self.crop_to_common_area,
            warnings,
        );
        take_field(
            table,
            "",
            "resize_cropped_to_original",
            &mut self.resize_cropped_to_original,
            warnings,
        );
        take_field(table, "", "auto_cull", &mut self.auto_cull, warnings);
        take_field(
            table,
            "",
            "auto_cull_threshold_pct",
            &mut self.auto_cull_threshold_pct,
            warnings,
        );
        take_field(
            table,
            "",
            "sort_by_sharpness",
            &mut self.sort_by_sharpness,
            warnings,
        );
        take_field(
            table,
            "",
            "stack_every_nth",
            &mut self.stack_every_nth,
            warnings,
        );
        take_field(table, "", "tile_size", &mut self.tile_size, warnings);
        take_field(
            table,
            "",
            "pyramid_levels",
            &mut self.pyramid_levels,
            warnings,
        );
        take_field(
            table,
            "",
            "use_all_color_channels",
            &mut self.use_all_color_channels,
            warnings,
        );
        take_field(
            table,
            "",
            "grit_suppression",
            &mut self.grit_suppression,
            warnings,
        );
        take_field(
            table,
            "",
            "relief_estimation_radius",
            &mut self.relief_estimation_radius,
            warnings,
        );
        take_field(
            table,
            "",
            "relief_smoothing_radius",
            &mut self.relief_smoothing_radius,
            warnings,
        );
        take_field(
            table,
            "",
            "relief_contrast_pct",
            &mut self.relief_contrast_pct,
            warnings,
        );
        take_field(
            table,
            "",
            "relief_show_preview",
            &mut self.relief_show_preview,
            warnings,
        );
        take_field(
            table,
            "",
            "relief_auto_detect",
            &mut self.relief_auto_detect,
            warnings,
        );
        take_field(
            table,
            "",
            "relief_use_multigrid",
            &mut self.relief_use_multigrid,
            warnings,
        );
        take_field(
            table,
            "",
            "strata_base_radius",
            &mut self.strata_base_radius,
            warnings,
        );
        take_field(
            table,
            "",
            "strata_detail_focus",
            &mut self.strata_detail_focus,
            warnings,
        );
        take_field(
            table,
            "",
            "compute_device",
            &mut self.compute_device,
            warnings,
        );
    }

    /// Return a copy with every numeric field clamped to its documented range
    /// (the ranges written in [`DEFAULT_CONFIG_TOML`]), plus one warning per
    /// field that had to change. An unchanged value produces no warning.
    ///
    /// - `settings_version`: a legacy value (below [`SETTINGS_VERSION`]) is
    ///   migrated silently by [`Self::migrate_legacy`]; a newer one only warns
    /// - `stack_every_nth`: at least 1
    /// - `auto_cull_threshold_pct`: 0.0..=50.0 (NaN becomes 0.0)
    /// - `tile_size`: 0 or 64..=8192
    /// - `strata_base_radius`: 8..=64
    /// - `strata_detail_focus`: 1..=5
    /// - `pyramid_levels`: 2..=12
    /// - `relief_estimation_radius`: 0..=100
    /// - `relief_smoothing_radius`: 0..=50
    /// - `relief_contrast_pct`: 0.0..=1.0 (NaN becomes 0.0)
    /// - `preprocessing.pre_rotation`: one of 0, 90, 180, 270 (else 0)
    /// - `preprocessing.pre_resize_percent`: 10..=100
    /// - `image_saving.bit_depth`: 8 or 16 (else 16)
    /// - `image_saving.jpeg_quality`: 1..=100
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "one clamp per documented field; the list is the documentation"
    )]
    pub fn sanitized(&self) -> (Self, Vec<String>) {
        let mut out = self.clone();
        let mut w = Vec::new();

        out.migrate_legacy();
        if out.settings_version > SETTINGS_VERSION {
            w.push(format!(
                "settings_version = {} is newer than this build knows ({SETTINGS_VERSION}); \
                 reading the file as it is",
                out.settings_version
            ));
        }
        clamp_u32(
            "stack_every_nth",
            &mut out.stack_every_nth,
            (1, u32::MAX),
            ">= 1",
            &mut w,
        );
        if out.auto_cull_threshold_pct.is_nan() {
            w.push("auto_cull_threshold_pct = NaN is not a number; using 0".to_owned());
            out.auto_cull_threshold_pct = 0.0;
        } else if !(0.0..=50.0).contains(&out.auto_cull_threshold_pct) {
            let clamped = out.auto_cull_threshold_pct.clamp(0.0, 50.0);
            w.push(format!(
                "auto_cull_threshold_pct = {} is outside 0.0..=50.0; using {clamped}",
                out.auto_cull_threshold_pct
            ));
            out.auto_cull_threshold_pct = clamped;
        }
        if out.tile_size != 0 {
            clamp_u32(
                "tile_size",
                &mut out.tile_size,
                (64, 8192),
                "0 or 64..=8192",
                &mut w,
            );
        }
        clamp_u32(
            "strata_base_radius",
            &mut out.strata_base_radius,
            (8, 64),
            "8..=64",
            &mut w,
        );
        clamp_u32(
            "strata_detail_focus",
            &mut out.strata_detail_focus,
            (1, 5),
            "1..=5",
            &mut w,
        );
        clamp_u32(
            "pyramid_levels",
            &mut out.pyramid_levels,
            (2, 12),
            "2..=12",
            &mut w,
        );
        clamp_u32(
            "relief_estimation_radius",
            &mut out.relief_estimation_radius,
            (0, 100),
            "0..=100",
            &mut w,
        );
        clamp_u32(
            "relief_smoothing_radius",
            &mut out.relief_smoothing_radius,
            (0, 50),
            "0..=50",
            &mut w,
        );
        if out.relief_contrast_pct.is_nan() {
            w.push("relief_contrast_pct = NaN is not a number; using 0".to_owned());
            out.relief_contrast_pct = 0.0;
        } else if !(0.0..=1.0).contains(&out.relief_contrast_pct) {
            let clamped = out.relief_contrast_pct.clamp(0.0, 1.0);
            w.push(format!(
                "relief_contrast_pct = {} is outside 0.0..=1.0; using {clamped}",
                out.relief_contrast_pct
            ));
            out.relief_contrast_pct = clamped;
        }

        let p = &mut out.preprocessing;
        if !matches!(p.pre_rotation, 0 | 90 | 180 | 270) {
            w.push(format!(
                "preprocessing.pre_rotation = {} is not one of 0, 90, 180, 270; using 0",
                p.pre_rotation
            ));
            p.pre_rotation = 0;
        }
        clamp_u32(
            "preprocessing.pre_resize_percent",
            &mut p.pre_resize_percent,
            (10, 100),
            "10..=100",
            &mut w,
        );

        let i = &mut out.image_saving;
        if !matches!(i.bit_depth, 8 | 16) {
            w.push(format!(
                "image_saving.bit_depth = {} is not 8 or 16; using 16",
                i.bit_depth
            ));
            i.bit_depth = 16;
        }
        clamp_u32(
            "image_saving.jpeg_quality",
            &mut i.jpeg_quality,
            (1, 100),
            "1..=100",
            &mut w,
        );

        (out, w)
    }
}

impl StackingSettings {
    /// Bring a legacy settings value up to [`SETTINGS_VERSION`]; a no-op for a
    /// current one.
    ///
    /// Before version 2, `affine` meant the 4-DOF similarity search and
    /// `registration` meant a single-pass bounded search. Since version 2
    /// `registration` is the 4-DOF similarity search and `affine` is the full
    /// 6-DOF model, so a legacy file keeps its behaviour through this mapping:
    ///
    /// | legacy `alignment_mode` | becomes |
    /// |---|---|
    /// | `affine` | `registration` |
    /// | `registration` | `registration` with `bounded_search = true` |
    /// | `translation`, `none` | unchanged |
    ///
    /// A value with `settings_version >= 2` is read literally. The loaders
    /// ([`Self::from_toml_str`], [`Self::sanitized`]) call this themselves;
    /// call it directly on a value deserialised through plain `serde`, where a
    /// missing `settings_version` reads as `0`. (Plain `serde` cannot tell a
    /// legacy file that omitted `alignment_mode` from one that named
    /// `registration`; [`Self::from_toml_str`] can and is the exact path.)
    pub const fn migrate_legacy(&mut self) {
        if self.settings_version >= SETTINGS_VERSION {
            return;
        }
        match self.alignment_mode {
            AlignmentModeSetting::Affine => {
                self.alignment_mode = AlignmentModeSetting::Registration;
            }
            AlignmentModeSetting::Registration => self.bounded_search = true,
            AlignmentModeSetting::Translation
            | AlignmentModeSetting::None
            | AlignmentModeSetting::Neural => {}
        }
        self.settings_version = SETTINGS_VERSION;
    }
}

/// Move the value of `old` to `new` unless `new` is already present (the new
/// name wins); `old` is removed either way.
fn rename_key(table: &mut toml::Table, old: &str, new: &str) {
    if let Some(value) = table.remove(old)
        && !table.contains_key(new)
    {
        table.insert(new.to_owned(), value);
    }
}

/// Rewrite the legacy spellings a settings table may contain into the current
/// schema. Reading only; saving always writes the current names.
///
/// - `dmap_estimation_radius`, `dmap_smoothing_radius`, `dmap_contrast_pct`,
///   `dmap_show_preview`, `dmap_auto_detect` become the matching `relief_*` key;
/// - the older `use_gpu = true|false` becomes `compute_device = "gpu"|"cpu"`;
/// - `relief_engine = "multigrid"|"guided_filter"` becomes
///   `relief_use_multigrid = true|false`;
/// - `alignment_mode = "bounded"` becomes `alignment_mode = "registration"`
///   with `bounded_search = true`.
///
/// When the current key is also present it wins and the legacy key is dropped.
/// A legacy key of the wrong type is left in the table (it then shows up as an
/// unknown key). Calling this twice changes nothing. It does not touch
/// `alignment_mode` values other than `bounded`: the version-dependent
/// meaning is handled by [`StackingSettings::migrate_legacy`].
pub fn apply_legacy_aliases(table: &mut toml::Table) {
    for (old, new) in [
        ("dmap_estimation_radius", "relief_estimation_radius"),
        ("dmap_smoothing_radius", "relief_smoothing_radius"),
        ("dmap_contrast_pct", "relief_contrast_pct"),
        ("dmap_show_preview", "relief_show_preview"),
        ("dmap_auto_detect", "relief_auto_detect"),
    ] {
        rename_key(table, old, new);
    }

    if let Some(gpu) = table.get("use_gpu").and_then(toml::Value::as_bool) {
        table.remove("use_gpu");
        if !table.contains_key("compute_device") {
            table.insert(
                "compute_device".to_owned(),
                (if gpu { "gpu" } else { "cpu" }).into(),
            );
        }
    }

    let multigrid = table
        .get("relief_engine")
        .and_then(toml::Value::as_str)
        .and_then(|s| match s.trim().to_ascii_lowercase().as_str() {
            "multigrid" => Some(true),
            "guided_filter" | "guided-filter" | "guided" => Some(false),
            _ => None,
        });
    if let Some(multigrid) = multigrid {
        table.remove("relief_engine");
        if !table.contains_key("relief_use_multigrid") {
            table.insert("relief_use_multigrid".to_owned(), multigrid.into());
        }
    }

    let bounded = table
        .get("alignment_mode")
        .and_then(toml::Value::as_str)
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("bounded"));
    if bounded {
        table.insert("alignment_mode".to_owned(), "registration".into());
        table.insert("bounded_search".to_owned(), true.into());
    }
}

/// The annotated settings file the applications write when none exists.
///
/// Every key carries its valid range or values in a comment, and the text parses
/// with [`StackingSettings::from_toml_str`]. It holds 39 keys: 27 top-level
/// ones, 6 in `[preprocessing]` and 6 in `[image_saving]`.
pub const DEFAULT_CONFIG_TOML: &str = r#"# z-stackr - Configuration File
# This file is automatically generated. Any missing settings will use their defaults.

# Version of this file's schema. Written by the applications; a file without
# it is read as a legacy file (see "alignment_mode" below).
settings_version = 2

# ── Alignment ────────────────────────────────────────────────────────────
# Registration model applied before fusion. The classical modes minimise the
# RMS difference to the reference over a coarse-to-fine luma pyramid.
# Valid values:
#   "translation"  - shift + one uniform scale (no rotation); corrects focus
#                    breathing
#   "registration" - shift + scale + rotation (4 degrees of freedom); the default
#   "affine"       - full 6 degrees of freedom: shift, scale, rotation, plus
#                    aspect and shear
#   "none"         - skip alignment entirely
#   "neural"       - learned matrices from a neural model (builds with the `nn`
#                    feature only; other builds refuse it)
# Files written before settings_version 2 used "affine" for what is now
# "registration" and "registration" for a single-pass bounded search; such a
# file is migrated when it is read ("affine" -> "registration", old
# "registration" -> "registration" with bounded_search = true).
alignment_mode = "registration"

# The intensity optimiser of the classical modes. Valid values:
#   "auto"        - Lucas-Kanade first, Nelder-Mead when it fails or ends worse
#   "lucaskanade" - Lucas-Kanade only (CPU; the browser build uses Nelder-Mead)
#   "neldermead"  - bounded Nelder-Mead simplex only
optimizer = "auto"

# Constrain the optimiser to a bounded band around the initial estimate
# (a logit-space simplex over fixed bands). This guarantees a non-degenerate
# transform. Leave false for an unconstrained pyramid search.
bounded_search = false

# Seed the optimiser with an AKAZE feature-match estimate before intensity
# refinement. Helps on textured subjects with large frame-to-frame shifts.
# Leave false for pure intensity registration, which is more robust on glossy / featureless subjects. Has no effect unless
# the application was built with the `akaze` feature.
akaze_seeding = false

# Match every aligned frame's brightness to the reference (computed in linear
# light), so exposure drift between frames does not reach the fusion.
correct_brightness = true

# Align each frame to its aligned predecessor (a rolling reference) instead of
# the single reference frame. Helps long stacks whose end frames share little
# with the reference; drift can accumulate.
chain_alignment = false

# With alignment_mode = "neural": true uses the network's matrices as seeds for
# the classical refinement, false applies them directly. No effect otherwise.
neural_refine_classically = true

# Crop the result to the largest rectangle that every aligned frame covers.
# A crop that would keep less than half the canvas is refused as a rogue
# frame. false keeps the full canvas (the border shows edge-clamped data).
crop_to_common_area = true

# After a common-area crop, resample the result back to the original canvas
# size (a slight non-uniform stretch). Ignored when nothing was cropped.
resize_cropped_to_original = false

# ── Frame selection ───────────────────────────────────────────────────────
# Automatically cull frames with low focus scores before stacking.
auto_cull = true

# Percentage of detail pixels a frame must win to survive the cull.
# Range: 0.0..=50.0. 0.0 = the automatic rule that scales with the frame count.
auto_cull_threshold_pct = 0.0

# Order the frames sharpest-first before alignment, so the sharpest frame
# becomes the reference.
sort_by_sharpness = false

# Process only every N-th frame from the loaded set (>= 1; 1 = all frames).
stack_every_nth = 1

# Edge length in pixels of the desktop application's out-of-core tiles.
# 0 = keep the stack in RAM; otherwise 64..=8192. The command line has its own
# --tile-size flag, which wins over this key.
tile_size = 0

# ── Apex (Laplacian-pyramid) ──────────────────────────────────────────────
# Number of pyramid levels used. Range: 2..=12.
pyramid_levels = 8

# Include the chroma (colour) bands in the Apex selection energy (luma² + chroma_a² +
# chroma_b²) instead of luma² only. Channels are not blended independently: all three
# are copied from the single winning frame.
use_all_color_channels = false

# At the finest Apex band, pick each pixel's winning frame by the energy summed over the
# 3x3 neighbourhood instead of the pixel alone. Nothing is smoothed; an isolated noisy
# pixel is just less likely to win.
grit_suppression = true

# ── Relief (Depth-Map / plane-fit focus measure) ────────────────────────────────────────────────
# Radius (in pixels) of the window for the plane-fit residual-variance focus measure.
# Range: 0..=100.
relief_estimation_radius = 5

# Guided-filter window radius for edge-preserving depth-map smoothing. Range: 0..=50.
relief_smoothing_radius = 2

# Percentile threshold for the depth-map focus-measure mask (0.0 to 1.0).
relief_contrast_pct = 0.0

# Show a live preview of the contrast mask during Relief stacking.
relief_show_preview = false

# Automatically compute the optimal contrast threshold when previewing.
relief_auto_detect = false

# Solve the depth map with the multigrid solver instead of the guided-filter
# argmax.
relief_use_multigrid = false

# ── Strata (Guided Fusion) ────────────────────────────────────────────────
# Box radius in pixels that splits the base and detail layers. Range: 8..=64.
strata_base_radius = 31

# How strongly the detail weights concentrate on the sharpest frame.
# Range: 1..=5 (1 soft, 5 close to winner-take-all).
strata_detail_focus = 3

# ── Compute ───────────────────────────────────────────────────────────────
# Where the alignment and the Apex and Relief fusions run (the AI mode has its
# own device choice). Valid values:
#   "cpu" - always the processor (the reference; the default)
#   "gpu" - the graphics card when this build has GPU support (the `gpu` feature)
#           and a usable adapter exists. The result agrees with the CPU within a
#           small tolerance, not bit for bit. When there is no adapter, the frames
#           do not fit in the device, or the device fails, the run continues on
#           the CPU and a warning says why.
compute_device = "cpu"

# ── Preprocessing ─────────────────────────────────────────────────────────
[preprocessing]
# Clockwise rotation applied to every frame before stacking.
# Valid values: 0, 90, 180, 270.
pre_rotation = 0

# Enable pre-stacking center-crop.
pre_crop_enabled = false

# Crop specification string ("w,h" or "x,y,w,h").
pre_crop_spec = ""

# Resize all frames to this percentage of their original size before stacking.
# Range: 10..=100.
pre_resize_percent = 100

# Process frames in reverse file order before stacking.
sort_reverse = false

# Ignore EXIF orientation tags when loading frames.
ignore_exif_orientation = false

# ── Image Saving ──────────────────────────────────────────────────────────
[image_saving]
# Output image file format.
# Valid values: "tiff", "png", "jpeg"
output_format = "tiff"

# Bit depth used when writing the output file.
# Valid values: 8, 16.
bit_depth = 16

# Quality level for JPEG output. Range: 1..=100.
jpeg_quality = 95

# Template string for the output filename (without extension).
# '{name}' is replaced with the base name of the first input frame.
filename_template = "{name}_stacked"

# Default directory for saving results.
default_output_dir = ""

# Reserved: metadata is not copied yet by either application (the CLI warns
# when this is true).
copy_metadata = false
"#;
