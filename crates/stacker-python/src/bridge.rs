//! The glue between the pipeline's hooks and Python: exceptions, the progress callback,
//! Ctrl-C and warnings.
//!
//! A stack runs with the GIL released (`Python::detach`), so other Python threads keep going.
//! The GIL is taken again only for the length of one progress call and one signal check.
//!
//! # Exceptions
//!
//! | Pipeline error | Python exception |
//! |---|---|
//! | `Cancelled` (Ctrl-C) | `KeyboardInterrupt` |
//! | `Invalid` (a request that cannot be honoured, see below) | `ValueError` |
//! | anything else (I/O, decoding, encoding, tile store) | `RuntimeError` |
//!
//! A request that cannot be honoured: no frames, frames of different sizes, a tile size below
//! 64, an output path without a known extension, a cull that leaves no frame.
//!
//! # The progress callback
//!
//! `progress(stage, current, total)`, see [`crate::progress`]. A callback that raises has its
//! exception printed to stderr and is otherwise ignored: a broken callback must not abort a
//! stack that has been running for minutes.

use std::cell::Cell;

use pyo3::{
    exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError},
    prelude::*,
};
use stacker_pipeline::{Cancel, PipelineError};

use crate::progress::ProgressForwarder;

/// The Python exception for a pipeline error (see the module documentation).
#[must_use]
pub fn to_py_err(error: PipelineError) -> PyErr {
    match error {
        PipelineError::Cancelled => PyKeyboardInterrupt::new_err("interrupted"),
        PipelineError::Invalid(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// A [`Cancel`] that fires when Python has a pending signal (Ctrl-C).
///
/// Polled between frames and tiles. `PyErr_CheckSignals` only reports on the main thread, so
/// a stack started from another thread cannot be interrupted this way.
#[derive(Debug, Default)]
pub struct SignalCancel {
    interrupted: Cell<bool>,
}

impl SignalCancel {
    /// A cancel that has not fired.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            interrupted: Cell::new(false),
        }
    }
}

impl Cancel for SignalCancel {
    fn is_cancelled(&self) -> bool {
        if self.interrupted.get() {
            return true;
        }
        let pending = Python::attach(|py| py.check_signals().is_err());
        self.interrupted.set(pending);
        pending
    }
}

/// A progress sink that calls `callback(stage, current, total)` (a no-op without a callback).
pub fn python_progress(
    callback: Option<&Py<PyAny>>,
) -> ProgressForwarder<impl Fn(&'static str, u32, u32) + '_> {
    ProgressForwarder::new(move |stage, current, total| {
        let Some(callback) = callback else { return };
        Python::attach(|py| {
            if let Err(error) = callback.call1(py, (stage, current, total)) {
                error.print(py);
            }
        });
    })
}

/// Raise every message of `warnings` as a Python `UserWarning` (`warnings.warn`).
///
/// # Errors
/// The exception `warnings.warn` raises, when the caller turned warnings into errors.
pub fn emit_warnings(py: Python<'_>, warnings: &[String]) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let module = py.import("warnings")?;
    for warning in warnings {
        module.call_method1("warn", (warning.as_str(),))?;
    }
    Ok(())
}
