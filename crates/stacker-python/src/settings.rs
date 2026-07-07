#![cfg(feature = "python")]
//! Python-visible mirrors of `stacker_core::settings` and the pipeline's
//! per-run parameters.
//!
//! # "Break on change" contract
//!
//! [`PyStackingSettings::into_core`] and [`PyStackingSettings::from_core`]
//! (and their `PyPreprocessingSettings` / `PyImageSavingSettings`
//! equivalents) destructure the core structs **exhaustively**, with no
//! `..Default::default()` and no `..rest` catch-all. This is deliberate: if
//! a future change to `stacker_core::settings::StackingSettings` (or a
//! sub-struct) adds a field, these conversions fail to COMPILE until a
//! maintainer deliberately decides how that field should be exposed to
//! Python (mirror it, or explicitly document why it's skipped). A silently
//! stale binding — one that compiles fine but quietly drops a new setting —
//! is a worse failure mode than a compile error, especially for a setting
//! that changes fusion output.
//!
//! Every field of `StackingSettings` is mirrored here. Nothing is skipped:
//! there are no GUI-only transient/preview fields on `StackingSettings`
//! itself (`relief_show_preview` IS part of the struct and IS mirrored below;
//! genuinely transient GUI state such as the live `Relief` preview popup's
//! in-memory image buffer lives entirely in `apps/stacker-gui` and was never
//! part of `StackingSettings` to begin with).

use pyo3::{exceptions::PyValueError, prelude::*};
use stacker_core::settings::{
    AlignmentModeSetting, ImageSavingSettings, OptimizerSetting, OutputFormat,
    PreprocessingSettings, StackingSettings,
};

// ── Enum string conversion helpers ──────────────────────────────────────────
//
// Python passes/receives these as plain strings (matching the TOML/GUI combo-
// box vocabulary from `as_combo_str`), but unlike `from_combo_str` (which
// silently falls back to a default on an unrecognised string — the right
// choice for a forgiving TOML loader) an invalid string from a Python caller
// must raise `ValueError` naming the valid options, so a typo is caught
// immediately instead of silently substituting a different alignment mode.

/// Every valid `AlignmentModeSetting` combo string, in declaration order.
/// Used both to validate a Python-supplied string and to build the
/// `ValueError` message listing the valid options.
fn alignment_mode_options() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut opts = vec![
        AlignmentModeSetting::Affine.as_combo_str(),
        AlignmentModeSetting::Translation.as_combo_str(),
        AlignmentModeSetting::Registration.as_combo_str(),
        AlignmentModeSetting::None.as_combo_str(),
    ];
    #[cfg(feature = "nn")]
    opts.push(AlignmentModeSetting::Neural.as_combo_str());
    opts
}

/// Parse a Python-supplied alignment-mode string, raising `ValueError`
/// (listing the valid options) instead of silently falling back to a
/// default on an unrecognised value.
fn parse_alignment_mode(s: &str) -> PyResult<AlignmentModeSetting> {
    let options = alignment_mode_options();
    if let Some(pos) = options.iter().position(|&o| o == s) {
        // Re-derive the matching variant from its own combo string rather
        // than trusting `from_combo_str`'s silent-fallback behaviour, so a
        // typo can never be misinterpreted as `Affine`.
        return Ok(match pos {
            0 => AlignmentModeSetting::Affine,
            1 => AlignmentModeSetting::Translation,
            2 => AlignmentModeSetting::Registration,
            3 => AlignmentModeSetting::None,
            #[cfg(feature = "nn")]
            4 => AlignmentModeSetting::Neural,
            _ => unreachable!("position bounded by `options.len()`"),
        });
    }
    Err(PyValueError::new_err(format!(
        "invalid alignment_mode {s:?}; valid options are: {}",
        options.join(", ")
    )))
}

/// Every valid `OptimizerSetting` combo string, in declaration order.
const fn optimizer_options() -> [&'static str; 3] {
    [
        OptimizerSetting::Auto.as_combo_str(),
        OptimizerSetting::LucasKanade.as_combo_str(),
        OptimizerSetting::NelderMead.as_combo_str(),
    ]
}

/// Parse a Python-supplied optimizer string, raising `ValueError` (listing
/// the valid options) instead of silently falling back to a default on an
/// unrecognised value — same pattern as [`parse_alignment_mode`].
fn parse_optimizer(s: &str) -> PyResult<OptimizerSetting> {
    let options = optimizer_options();
    match options.iter().position(|&o| o == s) {
        Some(0) => Ok(OptimizerSetting::Auto),
        Some(1) => Ok(OptimizerSetting::LucasKanade),
        Some(2) => Ok(OptimizerSetting::NelderMead),
        _ => Err(PyValueError::new_err(format!(
            "invalid optimizer {s:?}; valid options are: {}",
            options.join(", ")
        ))),
    }
}

/// Every valid `OutputFormat` combo string, in declaration order.
const fn output_format_options() -> [&'static str; 3] {
    [
        OutputFormat::Tiff.as_combo_str(),
        OutputFormat::Png.as_combo_str(),
        OutputFormat::Jpeg.as_combo_str(),
    ]
}

/// Parse a Python-supplied output-format string, raising `ValueError`
/// (listing the valid options) instead of silently falling back to a
/// default on an unrecognised value.
fn parse_output_format(s: &str) -> PyResult<OutputFormat> {
    let options = output_format_options();
    match options.iter().position(|&o| o == s) {
        Some(0) => Ok(OutputFormat::Tiff),
        Some(1) => Ok(OutputFormat::Png),
        Some(2) => Ok(OutputFormat::Jpeg),
        _ => Err(PyValueError::new_err(format!(
            "invalid output_format {s:?}; valid options are: {}",
            options.join(", ")
        ))),
    }
}

/// Every valid `relief_engine` string this binding accepts, mapped onto
/// `StackingSettings::relief_use_multigrid` (the core struct has no separate
/// `ReliefEngine` enum — it's a plain `bool` — so this binding gives Python a
/// symbolic two-option string instead of a bare boolean, matching how the
/// GUI's "`Relief` engine" dropdown presents the same choice).
const RELIEF_ENGINE_GUIDED: &str = "guided_filter";
const RELIEF_ENGINE_MULTIGRID: &str = "multigrid";

fn parse_relief_engine(s: &str) -> PyResult<bool> {
    match s {
        RELIEF_ENGINE_GUIDED => Ok(false),
        RELIEF_ENGINE_MULTIGRID => Ok(true),
        other => Err(PyValueError::new_err(format!(
            "invalid relief_engine {other:?}; valid options are: {RELIEF_ENGINE_GUIDED}, {RELIEF_ENGINE_MULTIGRID}"
        ))),
    }
}

const fn relief_engine_str(use_multigrid: bool) -> &'static str {
    if use_multigrid {
        RELIEF_ENGINE_MULTIGRID
    } else {
        RELIEF_ENGINE_GUIDED
    }
}

/// Valid range (inclusive) for `strata_detail_focus`, mirroring
/// `stacker_core::settings::StackingSettings::clamp_valid`'s `1..=5` clamp.
/// Duplicated here as literals (same reason `stacker-algo`'s
/// `MAX_BASE_RADIUS_SETTING` duplicates its own clamp bound) purely so this
/// binding's `ValueError` can quote the exact bounds without depending on
/// `stacker-algo`.
const STRATA_DETAIL_FOCUS_MIN: u32 = 1;
const STRATA_DETAIL_FOCUS_MAX: u32 = 5;

/// Validate a Python-supplied `strata_detail_focus`, raising `ValueError`
/// (naming the valid range) instead of silently clamping — unlike the
/// GUI/CLI/config-file paths (which clamp forgivingly via `clamp_valid`, the
/// right behaviour for a TOML file that might have been hand-edited or
/// carried over from an older version), a Python caller passing an
/// out-of-range value almost certainly made a mistake in code that should
/// fail loudly, matching this module's existing string-enum validation
/// pattern ([`parse_alignment_mode`], [`parse_optimizer`], etc.).
fn validate_strata_detail_focus(v: u32) -> PyResult<u32> {
    if (STRATA_DETAIL_FOCUS_MIN..=STRATA_DETAIL_FOCUS_MAX).contains(&v) {
        Ok(v)
    } else {
        Err(PyValueError::new_err(format!(
            "invalid strata_detail_focus {v}; must be in range {STRATA_DETAIL_FOCUS_MIN}..={STRATA_DETAIL_FOCUS_MAX}"
        )))
    }
}

// ── PipelineParams mirror ───────────────────────────────────────────────────

/// Python mirror of `stacker_pipeline::PipelineParams`.
///
/// Every field maps 1:1 onto the CLI's equivalent flag (see
/// `apps/stacker-cli/src/args.rs`): `mode` ↔ `--mode`, `tile_size` ↔
/// `--tile-size`, `model`/`device` ↔ `--model`/`--device` (AI fusion),
/// `align_model` ↔ `--align-model` (neural alignment). `model`, `device`,
/// and `align_model` require a build of this extension compiled with the
/// `nn` feature (`maturin build --features nn`) — see the crate README's
/// "AI models" section; without it they are accepted but ignored (the
/// underlying `z-stackr-pipeline` build simply has no neural code paths to
/// dispatch to).
// `from_py_object` is opted in explicitly (pyo3 0.29 deprecated the implicit
// `FromPyObject` derive on `Clone` pyclasses in favour of an explicit choice):
// this type needs to keep being extractable from a Python object, matching
// its pre-0.29 behaviour.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyPipelineParams {
    /// Ordered list of input frame file paths.
    #[pyo3(get, set)]
    pub paths: Vec<String>,
    /// Destination path for the stacked result; format is inferred from the
    /// extension (`.png`/`.tif`/`.tiff` support 16-bit output).
    #[pyo3(get, set)]
    pub output_file: String,
    /// Fusion mode: `"apex"`, `"relief"`, or (with the `nn` feature) `"ai"`.
    #[pyo3(get, set)]
    pub mode: String,
    /// Tile edge length in pixels for the out-of-core tiled fusion pass.
    #[pyo3(get, set)]
    pub tile_size: usize,
    /// AI mode: name (file stem) of the fusion model in the `models/`
    /// directory. `None` uses the first discovered model. Requires the `nn`
    /// feature.
    #[pyo3(get, set)]
    pub model: Option<String>,
    /// AI mode: inference device, `"cpu"` or `"gpu"`. `None` uses the best
    /// available backend. Requires the `nn` feature.
    #[pyo3(get, set)]
    pub device: Option<String>,
    /// Neural alignment mode (`settings.alignment_mode == "Neural"`): name
    /// (file stem) of the alignment model in the `models/` directory.
    /// `None` uses the first discovered alignment model. Requires the `nn`
    /// feature.
    #[pyo3(get, set)]
    pub align_model: Option<String>,
}

#[pymethods]
impl PyPipelineParams {
    /// Construct pipeline parameters for [`crate::file_api::stack_files`].
    #[new]
    #[pyo3(signature = (paths, output_file, mode="apex".to_string(), tile_size=512, model=None, device=None, align_model=None))]
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        paths: Vec<String>,
        output_file: String,
        mode: String,
        tile_size: usize,
        model: Option<String>,
        device: Option<String>,
        align_model: Option<String>,
    ) -> Self {
        Self {
            paths,
            output_file,
            mode,
            tile_size,
            model,
            device,
            align_model,
        }
    }
}

// ── PreprocessingSettings mirror ────────────────────────────────────────────

/// Python mirror of `stacker_core::settings::PreprocessingSettings`.
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyPreprocessingSettings {
    /// Clockwise rotation in degrees, one of `0`, `90`, `180`, `270`.
    #[pyo3(get, set)]
    pub pre_rotation: u32,
    /// Enable pre-stacking center-crop.
    #[pyo3(get, set)]
    pub pre_crop_enabled: bool,
    /// Crop spec: `"w,h"` (center crop) or `"x,y,w,h"` (absolute).
    #[pyo3(get, set)]
    pub pre_crop_spec: String,
    /// Resize percentage applied before stacking, `10..=100`.
    #[pyo3(get, set)]
    pub pre_resize_percent: u32,
    /// Process frames in reverse file order.
    #[pyo3(get, set)]
    pub sort_reverse: bool,
    /// Ignore EXIF orientation tags when loading frames.
    #[pyo3(get, set)]
    pub ignore_exif_orientation: bool,
}

#[pymethods]
impl PyPreprocessingSettings {
    #[new]
    #[pyo3(signature = (
        pre_rotation=0,
        pre_crop_enabled=false,
        pre_crop_spec=String::new(),
        pre_resize_percent=100,
        sort_reverse=false,
        ignore_exif_orientation=false,
    ))]
    #[must_use]
    pub const fn new(
        pre_rotation: u32,
        pre_crop_enabled: bool,
        pre_crop_spec: String,
        pre_resize_percent: u32,
        sort_reverse: bool,
        ignore_exif_orientation: bool,
    ) -> Self {
        Self {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl Default for PyPreprocessingSettings {
    fn default() -> Self {
        Self::from_core(&PreprocessingSettings::default())
    }
}

impl PyPreprocessingSettings {
    /// Convert to the core struct. Exhaustively destructured — see the
    /// module-level "break on change" doc comment.
    #[must_use]
    pub fn into_core(self) -> PreprocessingSettings {
        let Self {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        } = self;
        PreprocessingSettings {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        }
    }

    /// Convert from the core struct. Exhaustively destructured — see the
    /// module-level "break on change" doc comment.
    #[must_use]
    pub fn from_core(core: &PreprocessingSettings) -> Self {
        let PreprocessingSettings {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        } = core.clone();
        Self {
            pre_rotation,
            pre_crop_enabled,
            pre_crop_spec,
            pre_resize_percent,
            sort_reverse,
            ignore_exif_orientation,
        }
    }
}

// ── ImageSavingSettings mirror ──────────────────────────────────────────────

/// Python mirror of `stacker_core::settings::ImageSavingSettings`.
///
/// `output_format` is a string (`"TIFF"` / `"PNG"` / `"JPEG"`, matching
/// [`OutputFormat::as_combo_str`]); an unrecognised value raises
/// `ValueError` when converted via [`PyImageSavingSettings::into_core`].
#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyImageSavingSettings {
    /// `"TIFF"`, `"PNG"`, or `"JPEG"`.
    #[pyo3(get, set)]
    pub output_format: String,
    /// `8` or `16`.
    #[pyo3(get, set)]
    pub bit_depth: u32,
    /// JPEG quality, `1..=100`.
    #[pyo3(get, set)]
    pub jpeg_quality: u32,
    /// Output filename template; `{name}` is substituted with the input
    /// stem (single-stack) or subfolder name (batch mode).
    #[pyo3(get, set)]
    pub filename_template: String,
    /// Default output directory (GUI-only convenience; unused by
    /// [`crate::file_api::stack_files`], which always takes an explicit
    /// output path).
    #[pyo3(get, set)]
    pub default_output_dir: String,
    /// Copy EXIF/XMP metadata from the first input frame to the output
    /// (JPEG/PNG only).
    #[pyo3(get, set)]
    pub copy_metadata: bool,
}

#[pymethods]
impl PyImageSavingSettings {
    #[new]
    #[pyo3(signature = (
        output_format="TIFF".to_string(),
        bit_depth=16,
        jpeg_quality=95,
        filename_template="{name}_stacked".to_string(),
        default_output_dir=String::new(),
        copy_metadata=false,
    ))]
    #[must_use]
    pub const fn new(
        output_format: String,
        bit_depth: u32,
        jpeg_quality: u32,
        filename_template: String,
        default_output_dir: String,
        copy_metadata: bool,
    ) -> Self {
        Self {
            output_format,
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

impl Default for PyImageSavingSettings {
    fn default() -> Self {
        Self::from_core(&ImageSavingSettings::default())
    }
}

impl PyImageSavingSettings {
    /// Convert to the core struct, validating `output_format`.
    ///
    /// # Errors
    /// Returns `ValueError` if `output_format` is not one of the valid
    /// combo strings.
    pub fn into_core(self) -> PyResult<ImageSavingSettings> {
        let Self {
            output_format,
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        } = self;
        Ok(ImageSavingSettings {
            output_format: parse_output_format(&output_format)?,
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        })
    }

    /// Convert from the core struct. Exhaustively destructured — see the
    /// module-level "break on change" doc comment.
    #[must_use]
    pub fn from_core(core: &ImageSavingSettings) -> Self {
        let ImageSavingSettings {
            output_format,
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        } = core.clone();
        Self {
            output_format: output_format.as_combo_str().to_owned(),
            bit_depth,
            jpeg_quality,
            filename_template,
            default_output_dir,
            copy_metadata,
        }
    }
}

// ── StackingSettings mirror ─────────────────────────────────────────────────

/// Python mirror of `stacker_core::settings::StackingSettings`.
///
/// This is the same TOML-serialisable settings struct the GUI's config
/// dialog and the CLI's `--config` flag both read/write. Every field here
/// corresponds 1:1 to a field of the core struct (see the module-level
/// "break on change" doc comment); the field-by-field mapping to the GUI's
/// own labels is documented in the crate README's API reference.
///
/// `alignment_mode` and `relief_engine` are plain strings for Python
/// ergonomics (no need to import a Rust-side enum type); invalid values
/// raise `ValueError` naming the valid options rather than silently
/// substituting a default.
// This mirrors `StackingSettings`, which is itself a flat TOML-serialisable
// config struct with many independent boolean toggles (matching the GUI's
// checkbox-heavy settings panel) — splitting it into a state machine or
// two-variant enums here would only diverge further from the core struct
// it must stay a 1:1 mirror of.

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyStackingSettings {
    /// `"Affine"`, `"Translation"`, `"Registration"`, `"None"`, or (`nn`
    /// builds only) `"Neural"`.
    #[pyo3(get, set)]
    pub alignment_mode: String,
    /// Which intensity-based optimiser performs subpixel alignment
    /// refinement: `"Auto"` (default, tries Lucas-Kanade first and falls
    /// back to Nelder-Mead on failure/regression), `"Lucas-Kanade"` (force
    /// Lucas-Kanade only, no fallback), or `"Nelder-Mead"` (force the
    /// original bounded-simplex optimiser).
    #[pyo3(get, set)]
    pub optimizer: String,
    /// Seed the optimiser with an AKAZE feature-match estimate. Requires
    /// the `akaze` build feature in the consuming application (not this
    /// binding crate) to have any effect.
    #[pyo3(get, set)]
    pub akaze_seeding: bool,
    /// Hybrid neural alignment: use the network's matrix as a seed into
    /// classical refinement (`true`, default) instead of applying it
    /// directly (`false`). Only relevant when `alignment_mode == "Neural"`.
    #[pyo3(get, set)]
    pub neural_refine_classically: bool,
    /// Apply per-frame brightness/gamma correction after warping.
    #[pyo3(get, set)]
    pub correct_brightness: bool,
    /// Auto-cull low-contribution frames before stacking. Honoured by
    /// [`crate::array_api::stack_arrays`] (the in-RAM path); the tiled
    /// out-of-core file path (`stack_files`) ignores this field (and logs a
    /// warning), exactly like the CLI.
    #[pyo3(get, set)]
    pub auto_cull: bool,
    /// Sort frames by sharpness before stacking. Same
    /// in-RAM-path-only caveat as `auto_cull`.
    #[pyo3(get, set)]
    pub sort_by_sharpness: bool,
    /// Out-of-core tile edge length in pixels; GUI-only sentinel semantics
    /// (`0` = in-RAM path) are not meaningful from this binding, since
    /// [`crate::file_api::stack_files`] always tiles and
    /// [`crate::array_api::stack_arrays`] never does — this field is
    /// carried for config-file round-tripping only.
    #[pyo3(get, set)]
    pub tile_size: u32,
    /// Process only every N-th loaded frame (`>= 1`).
    #[pyo3(get, set)]
    pub stack_every_nth: u32,
    /// `Apex` pyramid depth, `2..=12`.
    #[pyo3(get, set)]
    pub pyramid_levels: u32,
    /// `Apex`: use all three color channels in the selection metric instead
    /// of luma-only.
    #[pyo3(get, set)]
    pub use_all_color_channels: bool,
    /// `Apex`: suppress single-pixel "grit" artefacts via 3×3-neighbourhood
    /// energy selection on the finest band.
    #[pyo3(get, set)]
    pub grit_suppression: bool,
    /// `Relief`: Sum-Modified-Laplacian box-filter radius, `0..=100`.
    #[pyo3(get, set)]
    pub relief_estimation_radius: u32,
    /// `Relief`: post-processing smoothing radius, `0..=50`.
    #[pyo3(get, set)]
    pub relief_smoothing_radius: u32,
    /// `Relief`: contrast-mask percentile threshold, `0.0..=1.0`.
    #[pyo3(get, set)]
    pub relief_contrast_pct: f32,
    /// `Relief`: show a live contrast-mask preview. GUI-only in effect; has
    /// no observable effect from this binding (there is no interactive
    /// preview loop here), but is carried for config-file round-tripping.
    #[pyo3(get, set)]
    pub relief_show_preview: bool,
    /// `Relief`: auto-detect the contrast threshold from the SML noise floor
    /// instead of using `relief_contrast_pct` directly.
    #[pyo3(get, set)]
    pub relief_auto_detect: bool,
    /// `Relief` engine selector: `"guided_filter"` (default, argmax + Guided
    /// Filter smoothing) or `"multigrid"` (confidence-weighted geometric
    /// multigrid depth-solver). Mirrors `StackingSettings::relief_use_multigrid`
    /// (a plain `bool` on the core struct) as a symbolic string, matching
    /// the GUI's "`Relief` engine" dropdown.
    #[pyo3(get, set)]
    pub relief_engine: String,
    /// `Strata`: box-filter radius (pixels) separating the low-frequency
    /// "base" layer from the high-frequency "detail" layer, `8..=64`. See
    /// `stacker_algo::strata::StrataParams::base_radius`.
    #[pyo3(get, set)]
    pub strata_base_radius: u32,
    /// `Strata`: "Detail focus" — the detail-layer weight re-concentration
    /// exponent, `1..=5` (default `3`). Higher = crisper depth edges and
    /// more detail retention on deep (dozens-of-frames) stacks; lower =
    /// smoother blending with fewer artifacts on flat/glossy subjects. `1`
    /// is the original soft Li-Kang-Hu behaviour, `5` is near
    /// winner-take-all. See `stacker_algo::strata::StrataParams::detail_focus`.
    #[pyo3(get, set)]
    pub strata_detail_focus: u32,
    /// Crop the stacked output to the largest rectangle covered by every
    /// aligned frame.
    #[pyo3(get, set)]
    pub crop_to_common_area: bool,
    /// When `crop_to_common_area` shrank the output, resample it back up to
    /// the original canvas resolution.
    #[pyo3(get, set)]
    pub resize_cropped_to_original: bool,
    /// Auto-Cull win-rate threshold percentage, `0.1..=5.0`.
    #[pyo3(get, set)]
    pub auto_cull_threshold_pct: f32,
    /// Allow the GPU-accelerated compute paths (Apex fusion, the production
    /// warp, Strata saliency, and the Relief guided-filter/multigrid
    /// engines) to engage. Only has an effect when this extension was built
    /// with the `gpu` Cargo feature (`maturin build --features gpu`);
    /// otherwise it round-trips through config-file (de)serialisation like
    /// any other setting but is inert, since the underlying
    /// `z-stackr-pipeline` build has no `wgpu` dependency compiled in.
    #[pyo3(get, set)]
    pub use_gpu: bool,
    /// Pre-stacking preprocessing (rotation/crop/resize/reverse/EXIF).
    #[pyo3(get, set)]
    pub preprocessing: PyPreprocessingSettings,
    /// Output encoding (format/bit-depth/JPEG quality/filename template/
    /// metadata copy).
    #[pyo3(get, set)]
    pub image_saving: PyImageSavingSettings,
}

#[pymethods]
impl PyStackingSettings {
    /// Construct settings with the same defaults as
    /// `StackingSettings::default()` (registration alignment, brightness
    /// correction on, auto-cull + sort-by-sharpness on, `Apex`-friendly
    /// defaults, guided-filter `Relief` engine, crop-to-common-area on).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    #[pyo3(signature = (
        alignment_mode=None,
        optimizer=None,
        akaze_seeding=None,
        neural_refine_classically=None,
        correct_brightness=None,
        auto_cull=None,
        sort_by_sharpness=None,
        tile_size=None,
        stack_every_nth=None,
        pyramid_levels=None,
        use_all_color_channels=None,
        grit_suppression=None,
        relief_estimation_radius=None,
        relief_smoothing_radius=None,
        relief_contrast_pct=None,
        relief_show_preview=None,
        relief_auto_detect=None,
        relief_engine=None,
        strata_base_radius=None,
        strata_detail_focus=None,
        crop_to_common_area=None,
        resize_cropped_to_original=None,
        auto_cull_threshold_pct=None,
        use_gpu=None,
        preprocessing=None,
        image_saving=None,
    ))]
    pub fn new(
        alignment_mode: Option<String>,
        optimizer: Option<String>,
        akaze_seeding: Option<bool>,
        neural_refine_classically: Option<bool>,
        correct_brightness: Option<bool>,
        auto_cull: Option<bool>,
        sort_by_sharpness: Option<bool>,
        tile_size: Option<u32>,
        stack_every_nth: Option<u32>,
        pyramid_levels: Option<u32>,
        use_all_color_channels: Option<bool>,
        grit_suppression: Option<bool>,
        relief_estimation_radius: Option<u32>,
        relief_smoothing_radius: Option<u32>,
        relief_contrast_pct: Option<f32>,
        relief_show_preview: Option<bool>,
        relief_auto_detect: Option<bool>,
        relief_engine: Option<String>,
        strata_base_radius: Option<u32>,
        strata_detail_focus: Option<u32>,
        crop_to_common_area: Option<bool>,
        resize_cropped_to_original: Option<bool>,
        auto_cull_threshold_pct: Option<f32>,
        use_gpu: Option<bool>,
        preprocessing: Option<PyPreprocessingSettings>,
        image_saving: Option<PyImageSavingSettings>,
    ) -> Self {
        let d = Self::default();
        Self {
            alignment_mode: alignment_mode.unwrap_or(d.alignment_mode),
            optimizer: optimizer.unwrap_or(d.optimizer),
            akaze_seeding: akaze_seeding.unwrap_or(d.akaze_seeding),
            neural_refine_classically: neural_refine_classically
                .unwrap_or(d.neural_refine_classically),
            correct_brightness: correct_brightness.unwrap_or(d.correct_brightness),
            auto_cull: auto_cull.unwrap_or(d.auto_cull),
            sort_by_sharpness: sort_by_sharpness.unwrap_or(d.sort_by_sharpness),
            tile_size: tile_size.unwrap_or(d.tile_size),
            stack_every_nth: stack_every_nth.unwrap_or(d.stack_every_nth),
            pyramid_levels: pyramid_levels.unwrap_or(d.pyramid_levels),
            use_all_color_channels: use_all_color_channels.unwrap_or(d.use_all_color_channels),
            grit_suppression: grit_suppression.unwrap_or(d.grit_suppression),
            relief_estimation_radius: relief_estimation_radius
                .unwrap_or(d.relief_estimation_radius),
            relief_smoothing_radius: relief_smoothing_radius.unwrap_or(d.relief_smoothing_radius),
            relief_contrast_pct: relief_contrast_pct.unwrap_or(d.relief_contrast_pct),
            relief_show_preview: relief_show_preview.unwrap_or(d.relief_show_preview),
            relief_auto_detect: relief_auto_detect.unwrap_or(d.relief_auto_detect),
            relief_engine: relief_engine.unwrap_or(d.relief_engine),
            strata_base_radius: strata_base_radius.unwrap_or(d.strata_base_radius),
            strata_detail_focus: strata_detail_focus.unwrap_or(d.strata_detail_focus),
            crop_to_common_area: crop_to_common_area.unwrap_or(d.crop_to_common_area),
            resize_cropped_to_original: resize_cropped_to_original
                .unwrap_or(d.resize_cropped_to_original),
            auto_cull_threshold_pct: auto_cull_threshold_pct.unwrap_or(d.auto_cull_threshold_pct),
            use_gpu: use_gpu.unwrap_or(d.use_gpu),
            preprocessing: preprocessing.unwrap_or(d.preprocessing),
            image_saving: image_saving.unwrap_or(d.image_saving),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        format!(
            "StackingSettings(alignment_mode={:?}, relief_engine={:?}, crop_to_common_area={})",
            self.alignment_mode, self.relief_engine, self.crop_to_common_area
        )
    }
}

impl Default for PyStackingSettings {
    fn default() -> Self {
        Self::from_core(&StackingSettings::default())
    }
}

impl PyStackingSettings {
    /// Convert to the core `StackingSettings`, validating `alignment_mode`
    /// and `relief_engine` (see [`parse_alignment_mode`] / [`parse_relief_engine`])
    /// and `image_saving.output_format`.
    ///
    /// Exhaustively destructured with no `..Default::default()` — see the
    /// module-level "break on change" doc comment: a new field added to
    /// `StackingSettings` in the future will fail to compile here until this
    /// function is deliberately updated to handle it.
    ///
    /// # Errors
    /// Returns `ValueError` if `alignment_mode`, `relief_engine`, or
    /// `image_saving.output_format` is not a recognised value.
    pub fn into_core(self) -> PyResult<StackingSettings> {
        let Self {
            alignment_mode,
            optimizer,
            akaze_seeding,
            neural_refine_classically,
            correct_brightness,
            auto_cull,
            sort_by_sharpness,
            tile_size,
            stack_every_nth,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_engine,
            strata_base_radius,
            strata_detail_focus,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull_threshold_pct,
            use_gpu,
            preprocessing,
            image_saving,
        } = self;

        Ok(StackingSettings {
            alignment_mode: parse_alignment_mode(&alignment_mode)?,
            optimizer: parse_optimizer(&optimizer)?,
            akaze_seeding,
            neural_refine_classically,
            correct_brightness,
            auto_cull,
            sort_by_sharpness,
            tile_size,
            stack_every_nth,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_use_multigrid: parse_relief_engine(&relief_engine)?,
            strata_base_radius,
            strata_detail_focus: validate_strata_detail_focus(strata_detail_focus)?,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull_threshold_pct,
            use_gpu,
            preprocessing: preprocessing.into_core(),
            image_saving: image_saving.into_core()?,
        })
    }

    /// Convert from a core `StackingSettings`.
    ///
    /// Exhaustively destructured with no `..Default::default()` — see the
    /// module-level "break on change" doc comment.
    #[must_use]
    pub fn from_core(core: &StackingSettings) -> Self {
        let StackingSettings {
            alignment_mode,
            optimizer,
            akaze_seeding,
            neural_refine_classically,
            correct_brightness,
            auto_cull,
            sort_by_sharpness,
            tile_size,
            stack_every_nth,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_use_multigrid,
            strata_base_radius,
            strata_detail_focus,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull_threshold_pct,
            use_gpu,
            preprocessing,
            image_saving,
        } = core.clone();

        Self {
            alignment_mode: alignment_mode.as_combo_str().to_owned(),
            optimizer: optimizer.as_combo_str().to_owned(),
            akaze_seeding,
            neural_refine_classically,
            correct_brightness,
            auto_cull,
            sort_by_sharpness,
            tile_size,
            stack_every_nth,
            pyramid_levels,
            use_all_color_channels,
            grit_suppression,
            relief_estimation_radius,
            relief_smoothing_radius,
            relief_contrast_pct,
            relief_show_preview,
            relief_auto_detect,
            relief_engine: relief_engine_str(relief_use_multigrid).to_owned(),
            strata_base_radius,
            strata_detail_focus,
            crop_to_common_area,
            resize_cropped_to_original,
            auto_cull_threshold_pct,
            use_gpu,
            preprocessing: PyPreprocessingSettings::from_core(&preprocessing),
            image_saving: PyImageSavingSettings::from_core(&image_saving),
        }
    }

    /// Alias of [`into_core`](Self::into_core). New code should prefer
    /// `into_core` directly.
    ///
    /// # Errors
    /// Same as [`into_core`](Self::into_core): returns `ValueError` if
    /// `alignment_mode`, `relief_engine`, or `image_saving.output_format` is
    /// not a recognised value.
    pub fn to_rust(self) -> PyResult<StackingSettings> {
        self.into_core()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `StackingSettings` with EVERY field set to a value that
    /// differs from `StackingSettings::default()`, so a round-trip test can
    /// actually detect a field that was dropped or mishandled instead of
    /// coincidentally matching because both sides used the default.
    fn non_default_core_settings() -> StackingSettings {
        StackingSettings {
            alignment_mode: AlignmentModeSetting::Affine, // default is Registration
            optimizer: OptimizerSetting::LucasKanade,     // default is Auto
            akaze_seeding: true,                          // default false
            neural_refine_classically: false,             // default true
            correct_brightness: false,                    // default true
            auto_cull: false,                             // default true
            sort_by_sharpness: false,                     // default true
            tile_size: 777,                               // default 0
            stack_every_nth: 3,                           // default 1
            pyramid_levels: 5,                            // default 8
            use_all_color_channels: true,                 // default false
            grit_suppression: false,                      // default true
            relief_estimation_radius: 9,                  // default 5
            relief_smoothing_radius: 7,                   // default 2
            relief_contrast_pct: 0.42,                    // default 0.0
            relief_show_preview: true,                    // default false
            relief_auto_detect: true,                     // default false
            relief_use_multigrid: true,                   // default false
            strata_base_radius: 40,                       // default 31
            strata_detail_focus: 5,                       // default 3
            crop_to_common_area: false,                   // default true
            resize_cropped_to_original: true,             // default false
            auto_cull_threshold_pct: 3.5,                 // default 2.0
            use_gpu: false,                               // default true
            preprocessing: PreprocessingSettings {
                pre_rotation: 90,
                pre_crop_enabled: true,
                pre_crop_spec: "10,20,300,400".to_owned(),
                pre_resize_percent: 50,
                sort_reverse: true,
                ignore_exif_orientation: true,
            },
            image_saving: ImageSavingSettings {
                output_format: OutputFormat::Jpeg,
                bit_depth: 8,
                jpeg_quality: 42,
                filename_template: "{name}_custom".to_owned(),
                default_output_dir: "/tmp/out".to_owned(),
                copy_metadata: true,
            },
        }
    }

    #[test]
    fn round_trip_every_field_non_default() {
        let core = non_default_core_settings();
        let py = PyStackingSettings::from_core(&core);
        let back = py
            .into_core()
            .expect("valid combo strings round-trip cleanly");
        assert_eq!(
            core, back,
            "core -> py -> core round trip must preserve every field"
        );
    }

    #[test]
    fn round_trip_default() {
        let core = StackingSettings::default();
        let py = PyStackingSettings::from_core(&core);
        let back = py.into_core().expect("default settings round-trip cleanly");
        assert_eq!(core, back);
    }

    #[test]
    fn invalid_alignment_mode_raises_value_error_with_options() {
        let py = PyStackingSettings {
            alignment_mode: "Sideways".to_owned(),
            ..PyStackingSettings::default()
        };
        let err = py.into_core().expect_err("bogus alignment_mode must fail");
        let msg = err.to_string();
        assert!(msg.contains("Sideways"));
        assert!(msg.contains("Affine"));
        assert!(msg.contains("Registration"));
    }

    #[test]
    fn invalid_optimizer_raises_value_error_with_options() {
        let py = PyStackingSettings {
            optimizer: "Sideways".to_owned(),
            ..PyStackingSettings::default()
        };
        let err = py.into_core().expect_err("bogus optimizer must fail");
        let msg = err.to_string();
        assert!(msg.contains("Sideways"));
        assert!(msg.contains("Auto"));
        assert!(msg.contains("Lucas-Kanade"));
        assert!(msg.contains("Nelder-Mead"));
    }

    #[test]
    fn optimizer_options_round_trip_via_parse_optimizer() {
        for opt in optimizer_options() {
            let parsed = parse_optimizer(opt).expect("valid combo string must parse");
            assert_eq!(parsed.as_combo_str(), opt);
        }
    }

    #[test]
    fn invalid_relief_engine_raises_value_error_with_options() {
        let py = PyStackingSettings {
            relief_engine: "bogus".to_owned(),
            ..PyStackingSettings::default()
        };
        let err = py.into_core().expect_err("bogus relief_engine must fail");
        let msg = err.to_string();
        assert!(msg.contains("bogus"));
        assert!(msg.contains(RELIEF_ENGINE_GUIDED));
        assert!(msg.contains(RELIEF_ENGINE_MULTIGRID));
    }

    #[test]
    fn invalid_strata_detail_focus_raises_value_error_with_range() {
        let py = PyStackingSettings {
            strata_detail_focus: 6,
            ..PyStackingSettings::default()
        };
        let err = py
            .into_core()
            .expect_err("out-of-range strata_detail_focus must fail");
        let msg = err.to_string();
        assert!(msg.contains('6'));
        assert!(msg.contains("1..=5") || (msg.contains('1') && msg.contains('5')));
    }

    #[test]
    fn strata_detail_focus_zero_raises_value_error() {
        let py = PyStackingSettings {
            strata_detail_focus: 0,
            ..PyStackingSettings::default()
        };
        let err = py
            .into_core()
            .expect_err("strata_detail_focus of 0 (below the 1..=5 range) must fail");
        assert!(err.to_string().contains('0'));
    }

    #[test]
    fn strata_detail_focus_bounds_are_valid() {
        for v in 1..=5 {
            let py = PyStackingSettings {
                strata_detail_focus: v,
                ..PyStackingSettings::default()
            };
            assert!(
                py.into_core().is_ok(),
                "strata_detail_focus={v} should be within the valid 1..=5 range"
            );
        }
    }

    #[test]
    fn invalid_output_format_raises_value_error_with_options() {
        let mut py = PyStackingSettings::default();
        py.image_saving.output_format = "BMP".to_owned();
        let err = py.into_core().expect_err("bogus output_format must fail");
        let msg = err.to_string();
        assert!(msg.contains("BMP"));
        assert!(msg.contains("TIFF"));
        assert!(msg.contains("PNG"));
        assert!(msg.contains("JPEG"));
    }

    #[test]
    fn relief_engine_str_round_trips_bool() {
        assert_eq!(relief_engine_str(false), RELIEF_ENGINE_GUIDED);
        assert_eq!(relief_engine_str(true), RELIEF_ENGINE_MULTIGRID);
        assert!(!parse_relief_engine(RELIEF_ENGINE_GUIDED).unwrap());
        assert!(parse_relief_engine(RELIEF_ENGINE_MULTIGRID).unwrap());
    }
}
