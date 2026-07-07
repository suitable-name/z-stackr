#![cfg(feature = "python")]
#![allow(clippy::needless_pass_by_value, clippy::struct_excessive_bools)]
//! `PyO3` Python bindings for **z-stackr**, a pure-Rust out-of-core
//! focus-stacking engine.
//!
//! This crate is a thin binding layer: every function here forwards
//! directly into the same shared Rust crates the CLI (`z-stackr-cli`) and
//! GUI (`z-stackr-gui`) call — `stacker_pipeline::run_pipeline` for the
//! out-of-core file path, and `stacker_align`/`stacker_algo` directly for
//! the in-RAM numpy path — so nothing here is a parallel reimplementation
//! that could silently drift from the two shipped applications.
//!
//! # Module layout
//!
//! - [`settings`] — Python mirrors of `stacker_core::settings` (the same
//!   TOML-serialisable settings struct the GUI/CLI use) and
//!   `stacker_pipeline::PipelineParams`.
//! - [`converter`] — bulk numpy `[H, W, 3]` <-> `PlanarImage<f32>`
//!   conversions, one variant per supported dtype (`u8`/`u16`/`f32`).
//! - [`progress`] — maps `stacker_pipeline::PipelineProgress` onto a stable
//!   `(stage, current, total)` Python-callable contract.
//! - [`file_api`] — the **file / out-of-core** path: `stack_files`,
//!   `batch_stack`, `load_config`, `save_config`, `load_image`. Memory
//!   scales with tile size, not stack size — use this for stacks too large
//!   to fit in RAM at once.
//! - [`array_api`] — the **numpy / in-RAM** path: `stack_arrays`. Requires
//!   the whole stack to already fit in memory (the caller already holds it
//!   as a list of numpy arrays), but in exchange supports `auto_cull` /
//!   `sort_by_sharpness`, which the tiled file path structurally cannot
//!   (see that module's docs).
//!
//! See the crate's own `README.md` for the full API reference, a
//! quickstart, the config-interop workflow, and feature-flag / installation
//! documentation (including the `z-stackr` `PyPI` name vs. `zstackr` import
//! name pairing).

// Module declarations (kept as a single block so the crate's public
// surface is discoverable from the top of this file).
pub mod array_api;
pub mod converter;
pub mod file_api;
pub mod progress;
pub mod settings;

use pyo3::prelude::*;

pub use array_api::stack_arrays;
pub use file_api::{batch_stack, load_config, load_image, save_config, stack_files};
pub use settings::{
    PyImageSavingSettings, PyPipelineParams, PyPreprocessingSettings, PyStackingSettings,
};

/// The `zstackr` Python extension module.
///
/// Import name is `zstackr` (see `pyproject.toml`'s `[tool.maturin]
/// module-name` and the crate README's "Installation" section for why this
/// differs from both the `PyPI` package name `z-stackr` and this crate's own
/// Cargo package name `z-stackr-python`).
#[pymodule]
fn zstackr(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // File / out-of-core API.
    m.add_function(wrap_pyfunction!(stack_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_stack, m)?)?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
    m.add_function(wrap_pyfunction!(save_config, m)?)?;
    m.add_function(wrap_pyfunction!(load_image, m)?)?;

    // Array / in-RAM API.
    m.add_function(wrap_pyfunction!(stack_arrays, m)?)?;

    // Settings / parameter classes.
    m.add_class::<PyPipelineParams>()?;
    m.add_class::<PyStackingSettings>()?;
    m.add_class::<PyPreprocessingSettings>()?;
    m.add_class::<PyImageSavingSettings>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `#[pymodule]` function must register every top-level name this
    /// crate advertises as its public Python API — introspected via the
    /// module's own `__dict__`/attribute list rather than hardcoding
    /// assumptions about `wrap_pyfunction!`/`add_class` internals.
    #[test]
    fn module_registers_all_expected_top_level_names() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "zstackr").expect("module creation should succeed");
            zstackr(py, &module).expect("module init function should succeed");

            let expected = [
                "stack_files",
                "batch_stack",
                "load_config",
                "save_config",
                "load_image",
                "stack_arrays",
                "PyPipelineParams",
                "PyStackingSettings",
                "PyPreprocessingSettings",
                "PyImageSavingSettings",
            ];
            for name in expected {
                assert!(
                    module.hasattr(name).unwrap_or(false),
                    "zstackr module is missing expected top-level attribute '{name}'"
                );
            }
        });
    }
}
