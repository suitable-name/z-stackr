#![cfg(feature = "python")]
//! File / out-of-core API: `stack_files`, `batch_stack`, `load_config`,
//! `save_config`, `load_image`.
//!
//! Everything in this module routes through `stacker_pipeline::run_pipeline`
//! (the same tiled, out-of-core engine `z-stackr-cli` always uses) or the
//! identical TOML serde `stacker_core::settings::StackingSettings` derives
//! from — never a reimplementation. This is the "whole stack does not need
//! to fit in RAM" path; see `crate::array_api` for the in-RAM numpy
//! counterpart and the crate README's "The two paths" section for when to
//! use which.

use std::path::{Path, PathBuf};

use image::GenericImageView;
use numpy::{IntoPyArray, PyArray3, ndarray::Array3};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use stacker_core::settings::StackingSettings;
use stacker_pipeline::{PipelineParams, collect_image_paths, run_pipeline};

use crate::{
    progress::call_progress,
    settings::{PyPipelineParams, PyStackingSettings},
};

/// Run the out-of-core, tiled focus-stacking pipeline on a list of image
/// file paths, writing the fused result directly to `params.output_file`.
///
/// This is the same engine `z-stackr-cli` always uses
/// (`stacker_pipeline::run_pipeline`): peak memory scales with the tile
/// size, not the number or resolution of input frames, so this is the right
/// entry point for stacks too large to hold in RAM at once.
///
/// For a stack that already fits in memory (e.g. built from live-captured
/// numpy arrays) see [`crate::array_api::stack_arrays`], which also
/// supports `auto_cull` / `sort_by_sharpness` — this out-of-core path does
/// not (matching the CLI; see the field docs on
/// `PyStackingSettings::auto_cull`).
///
/// `params.model` / `params.device` / `params.align_model` (AI fusion +
/// neural alignment) only have an effect when this extension was built with
/// the `nn` feature (`maturin build --features nn`); otherwise they are
/// accepted but the underlying pipeline has no neural code paths compiled
/// in and `params.mode = "ai"` / `settings.alignment_mode = "Neural"` will
/// fail at run time with a clear error instead of silently ignoring the
/// request.
///
/// `progress`, if given, is called with `(stage: str, current: int, total:
/// int)` at each pipeline stage boundary — see `crate::progress`'s module
/// docs for the exact mapping. The GIL is released for the whole compute
/// (`Python::detach`) so other Python threads keep running during a long
/// stack; it is re-acquired only for the duration of each `progress` call.
/// An exception raised by `progress` is printed to stderr and ignored — it
/// never aborts the stack.
///
/// # Errors
/// Raises `ValueError` if `settings.alignment_mode` / `relief_engine` /
/// `image_saving.output_format` is invalid, or `RuntimeError` if the
/// pipeline itself fails (I/O error, unsupported mode string, alignment
/// failure).
// `progress` is taken by value because pyo3's `#[pyfunction]` FFI boundary
// requires an owned `Option<Py<PyAny>>` here (extracted from an optional
// Python callable argument) — `clippy::needless_pass_by_value` does not
// know about that constraint.

#[pyfunction]
#[pyo3(signature = (params, settings, progress=None))]
pub fn stack_files(
    py: Python<'_>,
    params: &PyPipelineParams,
    settings: &PyStackingSettings,
    progress: Option<Py<PyAny>>,
) -> PyResult<()> {
    let pipeline_params = PipelineParams {
        paths: params.paths.iter().map(PathBuf::from).collect(),
        output_file: PathBuf::from(&params.output_file),
        mode: params.mode.clone(),
        tile_size: params.tile_size,
        model: params.model.clone(),
        device: params.device.clone(),
        align_model: params.align_model.clone(),
    };
    let rust_settings = settings.clone().into_core()?;

    // Release the GIL for the whole compute-heavy pipeline call so other
    // Python threads are not frozen during a long stack; the progress shim
    // re-acquires it only for the duration of each individual callback
    // invocation (see `crate::progress`'s module docs).
    py.detach(|| {
        smol::block_on(run_pipeline(&pipeline_params, &rust_settings, |event| {
            call_progress(progress.as_ref(), event);
        }))
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Load a `StackingSettings` TOML config file — the exact same file format
/// `z-stackr-cli`'s `--config` flag and the GUI's config dialog read/write.
///
/// Uses `stacker_core::settings::StackingSettings`'s serde, via the `toml`
/// crate. Missing fields fall back to their documented defaults, exactly
/// like the CLI's loader.
///
/// This is the "configure visually in the GUI, run headlessly from Python"
/// workflow: point the GUI at a folder, dial in settings, save a config
/// file, then call this function to get an equivalent [`PyStackingSettings`]
/// in a Python script.
///
/// # Errors
/// Raises `RuntimeError` if the file cannot be read, or `ValueError` if it
/// is not valid TOML / fails to parse as `StackingSettings` (including an
/// unrecognised `alignment_mode`/`relief_engine`/`output_format` string).
#[pyfunction]
pub fn load_config(path: &str) -> PyResult<PyStackingSettings> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to read config file '{path}': {e}"))
    })?;
    let mut core: StackingSettings = toml::from_str(&content)
        .map_err(|e| PyValueError::new_err(format!("failed to parse config file '{path}': {e}")))?;
    // Mirror the CLI's loader exactly (`apps/stacker-cli/src/main.rs`):
    // clamp every sub-struct after deserialising.
    core.clamp_valid();
    core.preprocessing.clamp_valid();
    core.image_saving.clamp_valid();
    Ok(PyStackingSettings::from_core(&core))
}

/// Save a [`PyStackingSettings`] to a TOML config file.
///
/// Uses the exact same serde `z-stackr-cli`/`z-stackr-gui` use, so the
/// resulting file is a drop-in `--config` for the CLI or a loadable config
/// for the GUI.
///
/// # Errors
/// Raises `ValueError` if the settings contain an invalid
/// `alignment_mode`/`relief_engine`/`output_format` string, or `RuntimeError`
/// if serialisation or the file write fails.
#[pyfunction]
pub fn save_config(path: &str, settings: &PyStackingSettings) -> PyResult<()> {
    let core = settings.clone().into_core()?;
    let text = toml::to_string_pretty(&core)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to serialise settings: {e}")))?;
    std::fs::write(path, text)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to write config file '{path}': {e}")))
}

/// Load a single image frame from disk, routed through
/// `stacker_core::io::load_frame`.
///
/// Uses the same loader the pipeline itself uses, returning a `[H, W, 3]`
/// `numpy.uint16` array to preserve bit depth regardless of the source
/// file's own depth (8-bit sources are simply scaled up to the full `u16`
/// range).
///
/// Supports RAW input (see the crate README's "RAW support" section) only
/// when this extension was built with the `raw` feature; otherwise a RAW
/// path raises `RuntimeError` with a clear "rebuild with `--features raw`"
/// message (the same message `stacker_core::io::LoadError::RawSupportDisabled`
/// carries).
///
/// # Errors
/// Raises `RuntimeError` if the file cannot be found/decoded.
///
/// # Panics
/// Never in practice: the output buffer is always built with exactly
/// `height * width * 3` elements, matching `Array3::from_shape_vec`'s shape
/// requirement.
#[pyfunction]
pub fn load_image<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray3<u16>>> {
    let dyn_img = stacker_core::io::load_frame(Path::new(path))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (w, h) = dyn_img.dimensions();
    let rgb16 = dyn_img.to_rgb16();
    let mut out = vec![0u16; (w as usize) * (h as usize) * 3];
    for (i, px) in rgb16.pixels().enumerate() {
        out[i * 3] = px[0];
        out[i * 3 + 1] = px[1];
        out[i * 3 + 2] = px[2];
    }
    let arr = Array3::from_shape_vec((h as usize, w as usize, 3), out)
        .expect("row-major buffer always matches (height, width, 3)");
    Ok(arr.into_pyarray(py))
}

/// Batch-stack every image-bearing direct subfolder of `input_dir` into its
/// own output file inside `output_dir`.
///
/// Uses `stacker_pipeline::collect_image_paths` to discover each
/// subfolder's frames — the identical shared helper the CLI uses
/// (respecting the `raw` feature gate), so this never drifts onto its own
/// hardcoded extension list.
///
/// Each subfolder's output filename is derived exactly like the CLI's own
/// subfolder-batch mode (`apps/stacker-cli/src/batch.rs`'s
/// `resolve_batch_output_path`): `settings.image_saving.filename_template`
/// with `{name}` substituted by the subfolder's own directory name, plus
/// the extension for `settings.image_saving.output_format`.
///
/// # Honest differences from the CLI
///
/// This function does **not** reimplement the CLI's interactive/`--stacks`
/// disambiguation prompt, its mixed-direct-images-and-subfolders handling,
/// or its `--monitor` mode — it only walks `input_dir`'s direct
/// subfolders that themselves directly contain images (via
/// `collect_image_paths`, so a subfolder with none is silently skipped,
/// matching the CLI's own non-recursive-beyond-one-level discovery rule)
/// and stacks each one independently and sequentially. A subfolder whose
/// stack fails is recorded in the returned list rather than raising and
/// aborting the whole batch, mirroring the CLI's "a failing subfolder must
/// not abort the batch" behaviour.
///
/// Returns a list of `(subfolder_name, succeeded, output_path_or_message)`
/// triples, one per discovered subfolder, in the order they were processed:
/// when `succeeded` is `True`, the third element is the output file path;
/// when `False`, it is the error message.
///
/// # Errors
/// Raises `RuntimeError` if `input_dir` cannot be read or `output_dir`
/// cannot be created. Per-subfolder failures are reported in the returned
/// list, not raised.
// `mode` is taken by value for the same pyo3 `#[pyo3(signature = ...)]`
// owned-default reason documented on `stack_files` above.

#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, settings, mode="apex".to_string(), tile_size=512))]
pub fn batch_stack(
    input_dir: &str,
    output_dir: &str,
    settings: &PyStackingSettings,
    mode: String,
    tile_size: usize,
) -> PyResult<Vec<(String, bool, String)>> {
    let input_dir = Path::new(input_dir);
    let output_dir = Path::new(output_dir);
    let rust_settings = settings.clone().into_core()?;

    std::fs::create_dir_all(output_dir).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "failed to create output directory '{}': {e}",
            output_dir.display()
        ))
    })?;

    let subfolders: Vec<PathBuf> = std::fs::read_dir(input_dir)
        .map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to read input directory '{}': {e}",
                input_dir.display()
            ))
        })?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && collect_image_paths(p).is_ok())
        .collect();
    let mut subfolders = subfolders;
    subfolders.sort();

    let mut results = Vec::with_capacity(subfolders.len());
    for subfolder in subfolders {
        let name = subfolder
            .file_name()
            .map_or_else(|| "stack".to_owned(), |n| n.to_string_lossy().into_owned());

        let outcome = (|| -> Result<PathBuf, String> {
            let paths = collect_image_paths(&subfolder).map_err(|e| e.to_string())?;
            // `{name}` here is a literal template placeholder being
            // substituted via `str::replace`, not a `format!`-style
            // argument — the lint below is a false positive for this
            // non-formatting use of the string.
            #[allow(clippy::literal_string_with_formatting_args)]
            let stem = rust_settings
                .image_saving
                .filename_template
                .replace("{name}", &name);
            let output_file = output_dir.join(format!(
                "{stem}.{}",
                rust_settings.image_saving.output_format.extension()
            ));

            let params = PipelineParams {
                paths,
                output_file: output_file.clone(),
                mode: mode.clone(),
                tile_size,
                model: None,
                device: None,
                align_model: None,
            };
            smol::block_on(run_pipeline(&params, &rust_settings, |_| {}))
                .map_err(|e| e.to_string())?;
            Ok(output_file)
        })();

        match outcome {
            Ok(p) => results.push((name, true, p.display().to_string())),
            Err(msg) => results.push((name, false, msg)),
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "z_stackr_python_file_api_test_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn config_round_trip_save_then_load() {
        Python::initialize();
        Python::attach(|_py| {
            let dir = tmp_dir("config_roundtrip");
            let path = dir.join("config.toml");

            let mut settings = PyStackingSettings {
                pyramid_levels: 11,
                relief_engine: "multigrid".to_owned(),
                ..PyStackingSettings::default()
            };
            settings.image_saving.output_format = "JPEG".to_owned();

            save_config(path.to_str().unwrap(), &settings).expect("save_config should succeed");
            let loaded = load_config(path.to_str().unwrap()).expect("load_config should succeed");

            assert_eq!(loaded, settings);
            let _ = std::fs::remove_dir_all(&dir);
        });
    }

    #[test]
    fn load_config_accepts_gui_style_default_toml() {
        Python::initialize();
        Python::attach(|_py| {
            let dir = tmp_dir("default_toml");
            let path = dir.join("default.toml");
            std::fs::write(&path, stacker_core::settings::DEFAULT_CONFIG_TOML).unwrap();

            let loaded = load_config(path.to_str().unwrap())
                .expect("the shipped default TOML must always load");
            let expected = PyStackingSettings::from_core(&StackingSettings::default());
            assert_eq!(loaded, expected);

            let _ = std::fs::remove_dir_all(&dir);
        });
    }

    #[test]
    fn load_config_missing_file_raises_runtime_error() {
        Python::initialize();
        Python::attach(|_py| {
            let err = load_config("this_config_does_not_exist_z_stackr.toml").unwrap_err();
            assert!(err.to_string().to_lowercase().contains("failed to read"));
        });
    }
}
