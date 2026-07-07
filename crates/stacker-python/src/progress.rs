#![cfg(feature = "python")]
//! Progress-callback shim: maps `stacker_pipeline::PipelineProgress` onto a
//! stable Python-facing `(stage: str, current: int, total: int)` tuple.
//!
//! # The mapping
//!
//! | `PipelineProgress` variant | `stage` string | `current` | `total` |
//! |---|---|---|---|
//! | `DecodeRaw { current, total }` | `"decode_raw"` | frame index (1-based) | RAW frame count |
//! | `AlignStart { total }` | `"align_start"` | `0` | frame count |
//! | `AlignFrame { current, total }` | `"align_frame"` | frame index (1-based) | frame count |
//! | `AlignDone` | `"align_done"` | `0` | `0` |
//! | `FuseStart { total }` | `"fuse_start"` | `0` | tile count |
//! | `FuseTile { current, total }` | `"fuse_tile"` | tile index (1-based) | tile count |
//! | `FuseDone` | `"fuse_done"` | `0` | `0` |
//! | `Encoding` | `"encoding"` | `0` | `0` |
//!
//! This mapping is intentionally a stable, minimal, structural contract
//! (a string tag plus two integers) rather than exposing
//! `PipelineProgress` itself as a `#[pyclass]` enum — pyo3 enums are more
//! ceremony for callers than a `(str, int, int)` tuple, and the tag/shape
//! above is easy to keep stable across future `PipelineProgress` variants
//! (a new variant just needs a new tag; existing callers pattern-matching
//! on known tags are unaffected).
//!
//! # GIL handling
//!
//! The pipeline's compute (`smol::block_on(run_pipeline(...))`) always runs
//! with the GIL released (`Python::detach`), so other Python threads are
//! not frozen during a long stack. The progress callback shim re-acquires
//! the GIL (`Python::attach`) only for the duration of each individual
//! Python callable invocation.
//!
//! # Exceptions raised by the Python callback
//!
//! If the Python progress callable raises, the exception is printed to
//! stderr (`PyErr::print`, the same thing an uncaught exception at the top
//! level would do) and otherwise ignored — the stack continues running.
//! A misbehaving progress callback must never abort an otherwise-successful
//! multi-minute stack.

use pyo3::prelude::*;
use stacker_pipeline::PipelineProgress;

/// Map one [`PipelineProgress`] event onto the stable `(stage, current,
/// total)` tuple described in the module docs.
#[must_use]
pub const fn progress_to_tuple(event: PipelineProgress) -> (&'static str, usize, usize) {
    match event {
        PipelineProgress::DecodeRaw { current, total } => ("decode_raw", current, total),
        PipelineProgress::AlignStart { total } => ("align_start", 0, total),
        PipelineProgress::AlignFrame { current, total } => ("align_frame", current, total),
        PipelineProgress::AlignDone => ("align_done", 0, 0),
        PipelineProgress::FuseStart { total } => ("fuse_start", 0, total),
        PipelineProgress::FuseTile { current, total } => ("fuse_tile", current, total),
        PipelineProgress::FuseDone => ("fuse_done", 0, 0),
        PipelineProgress::Encoding => ("encoding", 0, 0),
    }
}

/// Invoke an optional Python progress callable for one [`PipelineProgress`]
/// event, acquiring the GIL only for the duration of the call.
///
/// A no-op when `callback` is `None`. If the callable raises, the exception
/// is printed to stderr and swallowed (see the module docs) so a single bad
/// callback invocation can never abort an in-progress stack.
pub fn call_progress(callback: Option<&Py<PyAny>>, event: PipelineProgress) {
    let Some(callback) = callback else { return };
    let (stage, current, total) = progress_to_tuple(event);
    Python::attach(|py| {
        // A plain Rust tuple implements the conversion traits `call1` needs
        // directly, so there is no need to build a `PyTuple` by hand.
        if let Err(err) = callback.call1(py, (stage, current, total)) {
            err.print(py);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyTuple;

    #[test]
    fn mapping_matches_documented_table() {
        assert_eq!(
            progress_to_tuple(PipelineProgress::DecodeRaw {
                current: 2,
                total: 5
            }),
            ("decode_raw", 2, 5)
        );
        assert_eq!(
            progress_to_tuple(PipelineProgress::AlignStart { total: 7 }),
            ("align_start", 0, 7)
        );
        assert_eq!(
            progress_to_tuple(PipelineProgress::AlignFrame {
                current: 3,
                total: 7
            }),
            ("align_frame", 3, 7)
        );
        assert_eq!(
            progress_to_tuple(PipelineProgress::AlignDone),
            ("align_done", 0, 0)
        );
        assert_eq!(
            progress_to_tuple(PipelineProgress::FuseStart { total: 12 }),
            ("fuse_start", 0, 12)
        );
        assert_eq!(
            progress_to_tuple(PipelineProgress::FuseTile {
                current: 4,
                total: 12
            }),
            ("fuse_tile", 4, 12)
        );
        assert_eq!(
            progress_to_tuple(PipelineProgress::FuseDone),
            ("fuse_done", 0, 0)
        );
        assert_eq!(
            progress_to_tuple(PipelineProgress::Encoding),
            ("encoding", 0, 0)
        );
    }

    #[test]
    fn call_progress_none_callback_is_noop() {
        // Must not panic or require an initialised interpreter.
        call_progress(None, PipelineProgress::Encoding);
    }

    #[test]
    fn call_progress_invokes_python_callable_with_expected_args() {
        Python::initialize();
        Python::attach(|py| {
            let calls =
                std::sync::Arc::new(std::sync::Mutex::new(Vec::<(String, usize, usize)>::new()));

            // Build a Python-callable wrapper around a Rust closure via a
            // tiny `types.SimpleNamespace`-free approach: use `PyCFunction`.
            let calls_capture = calls.clone();
            let f = pyo3::types::PyCFunction::new_closure(
                py,
                None,
                None,
                move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<()> {
                    let stage: String = args.get_item(0)?.extract()?;
                    let current: usize = args.get_item(1)?.extract()?;
                    let total: usize = args.get_item(2)?.extract()?;
                    calls_capture.lock().unwrap().push((stage, current, total));
                    Ok(())
                },
            )
            .unwrap();
            let callback: Py<PyAny> = f.into_any().unbind();

            call_progress(
                Some(&callback),
                PipelineProgress::AlignFrame {
                    current: 2,
                    total: 9,
                },
            );

            let recorded = calls.lock().unwrap();
            assert_eq!(recorded.len(), 1);
            assert_eq!(recorded[0], ("align_frame".to_owned(), 2, 9));
            drop(recorded);
        });
    }

    #[test]
    fn call_progress_swallows_python_exceptions() {
        Python::initialize();
        Python::attach(|py| {
            let code =
                std::ffi::CString::new("def raiser(*a):\n    raise ValueError('boom')\n").unwrap();
            let filename = std::ffi::CString::new("raiser.py").unwrap();
            let module_name = std::ffi::CString::new("raiser_mod").unwrap();
            let module =
                pyo3::types::PyModule::from_code(py, &code, &filename, &module_name).unwrap();
            let raiser: Py<PyAny> = module.getattr("raiser").unwrap().unbind();

            // Must not panic or propagate the Python exception.
            call_progress(Some(&raiser), PipelineProgress::FuseDone);
        });
    }
}
