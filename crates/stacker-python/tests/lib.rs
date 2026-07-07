#![cfg(feature = "python")]
use pyo3::prelude::*;
use stacker_python::{
    settings::{PyPipelineParams, PyStackingSettings},
    stack_files,
};

#[test]
fn test_stack_files_missing_input() {
    Python::initialize();
    Python::attach(|py| {
        let params = PyPipelineParams {
            paths: vec!["does_not_exist.jpg".to_string()],
            output_file: "out.jpg".to_string(),
            mode: "apex".to_string(),
            tile_size: 512,
            model: None,
            device: None,
            align_model: None,
        };
        let settings = PyStackingSettings::default();

        let result = stack_files(py, &params, &settings, None);
        assert!(
            result.is_err(),
            "stacking non-existent input files must fail"
        );
    });
}

#[test]
fn test_stack_files_reports_progress_and_still_errors_on_missing_input() {
    // Regression coverage for the progress-callback plumbing: even on the
    // early-failure path (missing input), passing a `progress` callable
    // must not itself panic or change the error outcome.
    Python::initialize();
    Python::attach(|py| {
        let params = PyPipelineParams {
            paths: vec!["also_does_not_exist.jpg".to_string()],
            output_file: "out2.jpg".to_string(),
            mode: "apex".to_string(),
            tile_size: 512,
            model: None,
            device: None,
            align_model: None,
        };
        let settings = PyStackingSettings::default();

        let code = std::ffi::CString::new("def noop(*a):\n    pass\n").unwrap();
        let filename = std::ffi::CString::new("noop.py").unwrap();
        let module_name = std::ffi::CString::new("noop_mod").unwrap();
        let module = PyModule::from_code(py, &code, &filename, &module_name).unwrap();
        let progress: Py<PyAny> = module.getattr("noop").unwrap().unbind();

        let result = stack_files(py, &params, &settings, Some(progress));
        assert!(result.is_err());
    });
}
