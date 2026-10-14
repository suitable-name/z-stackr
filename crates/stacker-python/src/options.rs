//! Enum options as strings, and the error type of every rejected argument.
//!
//! Python passes and receives the enums of the settings as plain strings (the labels of the
//! GUI combo boxes: `"Registration"`, `"Lucas-Kanade"`, `"TIFF"`, `"CPU"`). The parsers here
//! are case-insensitive and ignore `-`, `_` and spaces, so `"lucas_kanade"` and `"Lucas-Kanade"`
//! both work. Unlike `from_combo_str` of the core (which keeps the default for an unknown
//! label, the right choice for a forgiving file loader), an unknown string from a Python caller
//! is an error that lists the valid options, so a typo can never silently select another mode.
//!
//! Nothing here needs Python: the module builds and is tested without the `python` feature.

use std::{fmt, path::Path};

use stacker_core::settings::{AlignmentModeSetting, ComputeDevice, OptimizerSetting, OutputFormat};

/// A rejected argument: an unknown option string, a value out of range, an array of the wrong
/// shape. Python receives it as a `ValueError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgError {
    message: String,
}

impl ArgError {
    /// An error with the given message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ArgError {}

#[cfg(feature = "python")]
impl From<ArgError> for pyo3::PyErr {
    fn from(error: ArgError) -> Self {
        pyo3::exceptions::PyValueError::new_err(error.message)
    }
}

/// `s` lower-cased, without spaces, `-` and `_`.
fn normalise(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

/// The error for an unknown `value` of the option `name`.
fn invalid(name: &str, value: &str, options: &[&str]) -> ArgError {
    ArgError::new(format!(
        "invalid {name} {value:?}; valid options are: {}",
        options.join(", ")
    ))
}

// ── alignment_mode ──────────────────────────────────────────────────────────

/// The labels of `alignment_mode`, in declaration order.
///
/// `"Neural"` is accepted in every build; without a neural model the pipeline aligns with
/// `Registration` instead and says so in a warning.
#[must_use]
pub const fn alignment_mode_options() -> [&'static str; 5] {
    [
        AlignmentModeSetting::Translation.as_combo_str(),
        AlignmentModeSetting::Registration.as_combo_str(),
        AlignmentModeSetting::Affine.as_combo_str(),
        AlignmentModeSetting::None.as_combo_str(),
        AlignmentModeSetting::Neural.as_combo_str(),
    ]
}

/// Parse an `alignment_mode` string.
///
/// Returns the mode and whether the string also asks for `bounded_search`: the legacy label
/// `"Bounded"` means `Registration` with a bounded search, as in a settings file.
///
/// # Errors
/// [`ArgError`] listing the valid options when `s` is none of them.
pub fn parse_alignment_mode(s: &str) -> Result<(AlignmentModeSetting, bool), ArgError> {
    match normalise(s).as_str() {
        "translation" => Ok((AlignmentModeSetting::Translation, false)),
        "registration" => Ok((AlignmentModeSetting::Registration, false)),
        "affine" => Ok((AlignmentModeSetting::Affine, false)),
        "none" => Ok((AlignmentModeSetting::None, false)),
        "neural" => Ok((AlignmentModeSetting::Neural, false)),
        "bounded" => Ok((AlignmentModeSetting::Registration, true)),
        _ => Err(invalid("alignment_mode", s, &alignment_mode_options())),
    }
}

// ── optimizer ───────────────────────────────────────────────────────────────

/// The labels of `optimizer`, in declaration order.
#[must_use]
pub const fn optimizer_options() -> [&'static str; 3] {
    [
        OptimizerSetting::Auto.as_combo_str(),
        OptimizerSetting::LucasKanade.as_combo_str(),
        OptimizerSetting::NelderMead.as_combo_str(),
    ]
}

/// Parse an `optimizer` string (`"lk"` and `"nm"` are accepted as short forms).
///
/// # Errors
/// [`ArgError`] listing the valid options when `s` is none of them.
pub fn parse_optimizer(s: &str) -> Result<OptimizerSetting, ArgError> {
    match normalise(s).as_str() {
        "auto" => Ok(OptimizerSetting::Auto),
        "lucaskanade" | "lk" => Ok(OptimizerSetting::LucasKanade),
        "neldermead" | "nm" => Ok(OptimizerSetting::NelderMead),
        _ => Err(invalid("optimizer", s, &optimizer_options())),
    }
}

// ── compute_device ──────────────────────────────────────────────────────────

/// The labels of `compute_device`.
#[must_use]
pub const fn compute_device_options() -> [&'static str; 2] {
    [
        ComputeDevice::Cpu.as_combo_str(),
        ComputeDevice::Gpu.as_combo_str(),
    ]
}

/// Parse a `compute_device` string.
///
/// # Errors
/// [`ArgError`] listing the valid options when `s` is none of them.
pub fn parse_compute_device(s: &str) -> Result<ComputeDevice, ArgError> {
    match normalise(s).as_str() {
        "cpu" => Ok(ComputeDevice::Cpu),
        "gpu" => Ok(ComputeDevice::Gpu),
        _ => Err(invalid("compute_device", s, &compute_device_options())),
    }
}

// ── output_format ───────────────────────────────────────────────────────────

/// The labels of `output_format`, in declaration order.
#[must_use]
pub const fn output_format_options() -> [&'static str; 3] {
    [
        OutputFormat::Tiff.as_combo_str(),
        OutputFormat::Png.as_combo_str(),
        OutputFormat::Jpeg.as_combo_str(),
    ]
}

/// Parse an `output_format` string (`"tif"` and `"jpg"` are accepted as aliases).
///
/// # Errors
/// [`ArgError`] listing the valid options when `s` is none of them.
pub fn parse_output_format(s: &str) -> Result<OutputFormat, ArgError> {
    match normalise(s).as_str() {
        "tiff" | "tif" => Ok(OutputFormat::Tiff),
        "png" => Ok(OutputFormat::Png),
        "jpeg" | "jpg" => Ok(OutputFormat::Jpeg),
        _ => Err(invalid("output_format", s, &output_format_options())),
    }
}

/// The lower-case file extension of `format`: `tif`, `png` or `jpg`.
#[must_use]
pub const fn extension_of(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Tiff => "tif",
        OutputFormat::Png => "png",
        OutputFormat::Jpeg => "jpg",
    }
}

/// The output format a file name asks for, from its extension (`.tif`, `.tiff`, `.png`,
/// `.jpg`, `.jpeg`, in any letter case); `None` for anything else.
#[must_use]
pub fn output_format_for_path(path: &Path) -> Option<OutputFormat> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "tif" | "tiff" => Some(OutputFormat::Tiff),
        "png" => Some(OutputFormat::Png),
        "jpg" | "jpeg" => Some(OutputFormat::Jpeg),
        _ => None,
    }
}

// ── relief_engine ───────────────────────────────────────────────────────────

/// The label of the guided-filter `Relief` engine.
pub const RELIEF_ENGINE_GUIDED: &str = "guided_filter";
/// The label of the multigrid `Relief` engine.
pub const RELIEF_ENGINE_MULTIGRID: &str = "multigrid";

/// Parse a `relief_engine` string into `relief_use_multigrid`.
///
/// # Errors
/// [`ArgError`] listing the valid options when `s` is none of them.
pub fn parse_relief_engine(s: &str) -> Result<bool, ArgError> {
    match normalise(s).as_str() {
        "guidedfilter" | "guided" => Ok(false),
        "multigrid" => Ok(true),
        _ => Err(invalid(
            "relief_engine",
            s,
            &[RELIEF_ENGINE_GUIDED, RELIEF_ENGINE_MULTIGRID],
        )),
    }
}

/// The `relief_engine` label for `relief_use_multigrid`.
#[must_use]
pub const fn relief_engine_str(use_multigrid: bool) -> &'static str {
    if use_multigrid {
        RELIEF_ENGINE_MULTIGRID
    } else {
        RELIEF_ENGINE_GUIDED
    }
}
