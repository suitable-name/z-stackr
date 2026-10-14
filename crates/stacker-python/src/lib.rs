//! Python bindings for **z-stackr**, the focus-stacking engine.
//!
//! The crate is a thin layer over the shared crates the command line tool and the desktop app
//! use (`z-stackr-pipeline`, `z-stackr-align`, `z-stackr-algo`, `z-stackr-core`): nothing here
//! reimplements a stage that could drift away from them. The Python import name is `zstackr`
//! (the `PyPI` name is `z-stackr`).
//!
//! # Layout
//!
//! The first group builds and is tested **without Python**:
//!
//! | Module | Content |
//! |---|---|
//! | [`options`] | enum options as strings, [`ArgError`](options::ArgError) |
//! | [`settings`] | mirrors of the 39 settings keys and of the parameters of a file run |
//! | [`convert`] | `[H, W, 3]` sample buffers to and from linear-light planar images |
//! | [`progress`] | the `(stage, current, total)` progress contract |
//! | [`stack`] | the flows: frames in RAM, files out of core or in RAM, batches |
//!
//! The second group needs the `python` feature (maturin enables it):
//!
//! | Module | Content |
//! |---|---|
//! | `converter` | the numpy layer over [`convert`] |
//! | `array_api` | `stack_arrays` |
//! | `file_api` | `stack_files`, `batch_stack`, `load_config`, `save_config`, `load_image` |
//! | `bridge` | exceptions, the progress callback, Ctrl-C, warnings |
//!
//! # Colour
//!
//! Arrays and files are sRGB-encoded; the pipeline computes in linear light. The conversions
//! use the exact transfer function in both directions (see [`convert`]), so a stack comes back
//! in the dtype and encoding it went in.
//!
//! See `README.md` for the API reference, the build instructions and examples.

#![warn(missing_docs)]

pub mod convert;
pub mod options;
pub mod progress;
pub mod settings;
pub mod stack;

#[cfg(feature = "python")]
pub mod array_api;
#[cfg(feature = "python")]
mod bridge;
#[cfg(feature = "python")]
pub mod converter;
#[cfg(feature = "python")]
pub mod file_api;

pub use settings::{
    PyImageSavingSettings, PyPipelineParams, PyPreprocessingSettings, PyStackingSettings,
};

#[cfg(feature = "python")]
pub use array_api::stack_arrays;
#[cfg(feature = "python")]
pub use file_api::{batch_stack, load_config, load_image, save_config, stack_files};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// The `zstackr` extension module.
///
/// The import name is set here and by `module-name` in `pyproject.toml`; it differs on
/// purpose from the PyPI name `z-stackr` and from the Cargo package `z-stackr-python`.
#[cfg(feature = "python")]
#[pymodule]
fn zstackr(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Files.
    m.add_function(wrap_pyfunction!(stack_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_stack, m)?)?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
    m.add_function(wrap_pyfunction!(save_config, m)?)?;
    m.add_function(wrap_pyfunction!(load_image, m)?)?;

    // Arrays.
    m.add_function(wrap_pyfunction!(stack_arrays, m)?)?;

    // Settings and parameters.
    m.add_class::<PyPipelineParams>()?;
    m.add_class::<PyStackingSettings>()?;
    m.add_class::<PyPreprocessingSettings>()?;
    m.add_class::<PyImageSavingSettings>()?;

    Ok(())
}
