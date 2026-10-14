//! The Python glue, run against a linked libpython: `cargo test --features python` (without
//! `extension-module`). The flows themselves are tested without Python in `stack.rs`; this file
//! checks what the numpy layer adds: dtype dispatch, shapes, error types.
#![cfg(feature = "python")]

use numpy::{IntoPyArray, PyArray3, PyArrayMethods, ndarray::Array3};
use pyo3::prelude::*;
use stacker_python::{PyStackingSettings, stack_arrays};

/// A `[32, 48, 3]` frame whose left (or right) half is a checkerboard and the rest flat.
fn half_sharp(sharp_left: bool) -> Array3<u8> {
    Array3::from_shape_fn((32, 48, 3), |(y, x, _)| {
        let sharp = if sharp_left { x < 24 } else { x >= 24 };
        match (sharp, (x + y) % 2) {
            (true, 0) => 230,
            (true, _) => 30,
            (false, _) => 110,
        }
    })
}

fn quiet_settings() -> PyStackingSettings {
    PyStackingSettings {
        alignment_mode: "None".to_owned(),
        auto_cull: false,
        correct_brightness: false,
        pyramid_levels: 3,
        ..PyStackingSettings::default()
    }
}

#[test]
fn stack_arrays_returns_the_dtype_and_shape_of_the_input() {
    Python::initialize();
    Python::attach(|py| {
        let frames: Vec<Bound<'_, PyAny>> = vec![
            half_sharp(true).into_pyarray(py).into_any(),
            half_sharp(false).into_pyarray(py).into_any(),
        ];
        let result = stack_arrays(
            py,
            frames,
            &quiet_settings(),
            "apex".to_owned(),
            "srgb".to_owned(),
            None,
        )
        .expect("stack_arrays should succeed");
        let arr: &Bound<'_, PyArray3<u8>> = result.cast().expect("uint8 in, uint8 out");
        assert_eq!(arr.readonly().as_array().shape(), [32, 48, 3]);
    });
}

#[test]
fn stack_arrays_handles_uint16_and_float32() {
    Python::initialize();
    Python::attach(|py| {
        let make_u16 = |left: bool| half_sharp(left).mapv(|v| u16::from(v) * 257);
        let frames: Vec<Bound<'_, PyAny>> = vec![
            make_u16(true).into_pyarray(py).into_any(),
            make_u16(false).into_pyarray(py).into_any(),
        ];
        let result = stack_arrays(
            py,
            frames,
            &quiet_settings(),
            "relief".to_owned(),
            "srgb".to_owned(),
            None,
        )
        .expect("uint16 stack");
        let arr: &Bound<'_, PyArray3<u16>> = result.cast().expect("uint16 out");
        assert_eq!(arr.readonly().as_array().shape(), [32, 48, 3]);

        let make_f32 = |left: bool| half_sharp(left).mapv(|v| f32::from(v) / 255.0);
        let frames: Vec<Bound<'_, PyAny>> = vec![
            make_f32(true).into_pyarray(py).into_any(),
            make_f32(false).into_pyarray(py).into_any(),
        ];
        let result = stack_arrays(
            py,
            frames,
            &quiet_settings(),
            "strata".to_owned(),
            "linear".to_owned(),
            None,
        )
        .expect("float32 stack");
        let arr: &Bound<'_, PyArray3<f32>> = result.cast().expect("float32 out");
        assert_eq!(arr.readonly().as_array().shape(), [32, 48, 3]);
    });
}

#[test]
fn stack_arrays_rejects_bad_arguments_with_value_errors() {
    Python::initialize();
    Python::attach(|py| {
        let settings = quiet_settings();
        let one = || half_sharp(true).into_pyarray(py).into_any();

        let err = stack_arrays(
            py,
            vec![],
            &settings,
            "apex".to_owned(),
            "srgb".to_owned(),
            None,
        )
        .unwrap_err();
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));
        assert!(err.to_string().contains("must not be empty"));

        let err = stack_arrays(
            py,
            vec![one()],
            &settings,
            "ai".to_owned(),
            "srgb".to_owned(),
            None,
        )
        .unwrap_err();
        assert!(err.is_instance_of::<pyo3::exceptions::PyValueError>(py));

        let err = stack_arrays(
            py,
            vec![one()],
            &settings,
            "apex".to_owned(),
            "gamma".to_owned(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("encoding"));

        let float = half_sharp(true)
            .mapv(|v| f32::from(v) / 255.0)
            .into_pyarray(py)
            .into_any();
        let err = stack_arrays(
            py,
            vec![one(), float],
            &settings,
            "apex".to_owned(),
            "srgb".to_owned(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("dtype"));

        let bad = PyStackingSettings {
            pyramid_levels: 99,
            ..quiet_settings()
        };
        let err = stack_arrays(
            py,
            vec![one()],
            &bad,
            "apex".to_owned(),
            "srgb".to_owned(),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("pyramid_levels"));
    });
}
