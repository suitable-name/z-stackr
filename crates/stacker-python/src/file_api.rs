//! The file path: `stack_files`, `batch_stack`, `load_config`, `save_config`, `load_image`.
//!
//! `stack_files` and `batch_stack` run the same stages as the command line tool and the desktop
//! app (`stacker_pipeline::tiled::run_tiled` for an out-of-core run, the in-memory session
//! stages for `tile_size = 0`, see `crate::stack`). A stack that does not fit in RAM belongs
//! here; for frames that are already numpy arrays see `crate::array_api::stack_arrays`.

use std::path::{Path, PathBuf};

use numpy::PyArray3;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use stacker_core::settings::StackingSettings;
use stacker_pipeline::tiled::Hooks;

use crate::{
    bridge::{SignalCancel, emit_warnings, python_progress, to_py_err},
    converter::rgb_array,
    settings::{PyPipelineParams, PyStackingSettings},
    stack::{CollectLog, FileJob, FuseMode, read_rgb16, run_batch, run_files},
};

/// Stack the frames of `params.paths` into `params.output_file`.
///
/// The format of the result follows the extension of `output_file` (`.tif`, `.tiff`, `.png`,
/// `.jpg`, `.jpeg`); bit depth, JPEG quality and `copy_metadata` come from
/// `settings.image_saving`. `params.tile_size` is the edge of the out-of-core tiles: peak
/// memory then depends on it, not on the number of frames. With `0` (and `tile_size = 0` in
/// the settings) the whole stack is decoded into RAM instead.
///
/// All settings apply, including `auto_cull`, `auto_cull_threshold_pct` and
/// `sort_by_sharpness` (in the out-of-core run from the focus census of the tile store),
/// `stack_every_nth` and `preprocessing`. The out-of-core run keeps the cropped size when
/// `resize_cropped_to_original` is set. Notes (a frame that fell back during the alignment,
/// a refused crop, a GPU that was given up) arrive as `UserWarning`.
///
/// `progress`, if given, is called as `progress(stage, current, total)`; see the README. The
/// compute runs with the GIL released and can be interrupted with Ctrl-C
/// (`KeyboardInterrupt`).
///
/// # Errors
/// `ValueError` for invalid settings, an unknown `mode`, an output path whose extension names
/// no format, no input frame, a tile size below 64, frames of different sizes.
/// `RuntimeError` when a frame cannot be read or the result cannot be written.
#[pyfunction]
#[pyo3(signature = (params, settings, progress=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn stack_files(
    py: Python<'_>,
    params: &PyPipelineParams,
    settings: &PyStackingSettings,
    progress: Option<Py<PyAny>>,
) -> PyResult<()> {
    let core = settings.clone().into_core()?;
    let mode = FuseMode::parse(&params.mode)?;
    let paths: Vec<PathBuf> = params.paths.iter().map(PathBuf::from).collect();
    let output_file = PathBuf::from(&params.output_file);
    let tile_size = params.tile_size;

    let (result, warnings) = py.detach(|| {
        let log = CollectLog::default();
        let forwarder = python_progress(progress.as_ref());
        let cancel = SignalCancel::new();
        let hooks = Hooks {
            progress: &forwarder,
            cancel: &cancel,
            log: &log,
        };
        let job = FileJob {
            paths: &paths,
            output_file: &output_file,
            mode,
            tile_size,
        };
        let result = run_files(&job, &core, &hooks);
        (result, log.into_warnings())
    });
    emit_warnings(py, &warnings)?;
    result.map_err(to_py_err)
}

/// Stack every direct sub-folder of `input_dir` that holds frames into its own file in
/// `output_dir`, one after the other.
///
/// The file of a folder is `settings.image_saving.filename_template` with `{name}` replaced by
/// the folder name, plus the extension of `settings.image_saving.output_format`. The frames of
/// a folder are its images in natural file-name order. `mode` and `tile_size` are as in
/// `stack_files`.
///
/// Returns one `(folder_name, succeeded, path_or_message)` per folder, in processing order:
/// the result file on success, the error message otherwise. A folder that fails does not stop
/// the batch (Ctrl-C does).
///
/// Unlike the command line tool this walks one level of sub-folders only; it has no monitor
/// mode and no prompt.
///
/// # Errors
/// `ValueError` for invalid settings or an unknown `mode`. `RuntimeError` when `input_dir`
/// cannot be read or `output_dir` cannot be created.
#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, settings, mode="apex".to_string(), tile_size=512))]
#[allow(clippy::needless_pass_by_value)]
pub fn batch_stack(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    settings: &PyStackingSettings,
    mode: String,
    tile_size: usize,
) -> PyResult<Vec<(String, bool, String)>> {
    let core = settings.clone().into_core()?;
    let mode = FuseMode::parse(&mode)?;

    let (result, warnings) = py.detach(|| {
        let log = CollectLog::default();
        let cancel = SignalCancel::new();
        let hooks = Hooks {
            progress: &stacker_pipeline::NoProgress,
            cancel: &cancel,
            log: &log,
        };
        let result = run_batch(
            Path::new(input_dir),
            Path::new(output_dir),
            &core,
            mode,
            tile_size,
            &hooks,
        );
        (result, log.into_warnings())
    });
    emit_warnings(py, &warnings)?;
    let entries = result.map_err(to_py_err)?;
    Ok(entries
        .into_iter()
        .map(|entry| match entry.result {
            Ok(path) => (entry.name, true, path.display().to_string()),
            Err(message) => (entry.name, false, message),
        })
        .collect())
}

/// Read a settings file: the format of the GUI and of the CLI's `--config`.
///
/// The file is read like the applications read it: a key that is missing keeps its default, the
/// old spellings (`dmap_*`, `use_gpu`, `relief_engine`) are accepted, a file from before
/// settings version 2 is migrated. A value that cannot be used (wrong type, unknown option,
/// out of range) keeps the default or is clamped, and each such case is raised as a
/// `UserWarning` naming the key.
///
/// # Errors
/// `RuntimeError` when the file cannot be read; `ValueError` when it is not valid TOML.
#[pyfunction]
pub fn load_config(py: Python<'_>, path: &str) -> PyResult<PyStackingSettings> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to read config file '{path}': {e}"))
    })?;
    let (core, warnings) = StackingSettings::from_toml_str(&text)
        .map_err(|e| PyValueError::new_err(format!("failed to parse config file '{path}': {e}")))?;
    let warnings: Vec<String> = warnings
        .into_iter()
        .map(|warning| format!("{path}: {warning}"))
        .collect();
    emit_warnings(py, &warnings)?;
    Ok(PyStackingSettings::from_core(&core))
}

/// Write `settings` to a settings file the GUI and the CLI (`--config`) can read.
///
/// # Errors
/// `ValueError` for invalid settings; `RuntimeError` when the file cannot be written.
#[pyfunction]
pub fn save_config(path: &str, settings: &PyStackingSettings) -> PyResult<()> {
    let core = settings.clone().into_core()?;
    let text = toml::to_string_pretty(&core)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to serialise settings: {e}")))?;
    std::fs::write(path, text)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to write config file '{path}': {e}")))
}

/// Read one frame as a `[H, W, 3]` `numpy.uint16` array of sRGB codes.
///
/// An 8-bit file is scaled to the 16-bit range; the EXIF orientation of a JPEG is applied. A
/// camera RAW file is developed (camera white balance, colour matrix, sRGB curve) in a build
/// with the `raw` feature and is an error otherwise. The array is what `stack_arrays`
/// expects for `uint16` input.
///
/// # Errors
/// `RuntimeError` when the file cannot be read or decoded.
#[pyfunction]
pub fn load_image<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray3<u16>>> {
    let (width, height, samples) = py
        .detach(|| read_rgb16(Path::new(path)))
        .map_err(to_py_err)?;
    rgb_array(py, height, width, samples)
}
