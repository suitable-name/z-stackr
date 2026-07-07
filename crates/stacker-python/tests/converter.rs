#![cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods, ndarray::Array3};
use pyo3::prelude::*;
use stacker_python::converter::{
    numpy_f32_to_planar, numpy_rgb_to_planar, numpy_u16_to_planar, planar_to_numpy_f32,
    planar_to_numpy_rgb, planar_to_numpy_u16,
};

#[test]
fn test_numpy_to_planar_and_back() {
    Python::initialize();
    Python::attach(|py| {
        let mut arr = Array3::<u8>::zeros((2, 2, 3));
        *arr.get_mut((0, 0, 0)).unwrap() = 255;
        *arr.get_mut((1, 1, 2)).unwrap() = 255;

        let py_arr = arr.into_pyarray(py);

        let planar = numpy_rgb_to_planar(&py_arr.readonly());
        assert_eq!(planar.width, 2);
        assert_eq!(planar.height, 2);

        let back_py_arr = planar_to_numpy_rgb(py, &planar);
        let back_arr = back_py_arr.readonly();
        let back_arr = back_arr.as_array();

        assert!((i32::from(back_arr[(0, 0, 0)]) - 255).abs() <= 2);
        assert!((i32::from(back_arr[(1, 1, 2)]) - 255).abs() <= 2);
    });
}

/// Upgraded coverage: the same gradient round-trips through the `u16` and
/// `f32` dtype variants too (the original test above only exercised `u8`),
/// proving the new dtype-dispatched converter functions added for
/// `stack_arrays` actually work end-to-end via the crate's public API, not
/// just via `converter.rs`'s own internal unit tests.
#[test]
fn test_numpy_u16_and_f32_round_trip_via_public_api() {
    Python::initialize();
    Python::attach(|py| {
        // u16
        let mut arr16 = Array3::<u16>::zeros((3, 3, 3));
        arr16[[0, 0, 0]] = 65535;
        arr16[[2, 2, 1]] = 40000;
        let py_arr16 = arr16.clone().into_pyarray(py);
        let planar16 = numpy_u16_to_planar(&py_arr16.readonly()).unwrap();
        assert_eq!((planar16.width, planar16.height), (3, 3));
        let back16 = planar_to_numpy_u16(py, &planar16);
        let back16 = back16.readonly();
        let back16 = back16.as_array();
        assert!((i32::from(back16[[0, 0, 0]]) - 65535).abs() <= 300);

        // f32
        let mut arr32 = Array3::<f32>::zeros((3, 3, 3));
        arr32[[0, 0, 0]] = 1.0;
        arr32[[2, 2, 1]] = 0.5;
        let py_arr32 = arr32.clone().into_pyarray(py);
        let planar32 = numpy_f32_to_planar(&py_arr32.readonly()).unwrap();
        let back32 = planar_to_numpy_f32(py, &planar32);
        let back32 = back32.readonly();
        let back32 = back32.as_array();
        assert!((back32[[0, 0, 0]] - 1.0).abs() < 1.0e-4);
    });
}
