//! numpy `[H, W, 3]` arrays to and from the pipeline's planar images, one function per dtype.
//!
//! This is the thin numpy layer over [`crate::convert`], which holds the conversions and the
//! description of the value ranges and of the [`Encoding`](crate::convert::Encoding) (sRGB by
//! default, linear light on request). Arrays must be C-contiguous: a transposed or sliced view
//! raises `ValueError` (call `numpy.ascontiguousarray` first) instead of being copied behind
//! the caller's back.

use numpy::{
    Element, IntoPyArray, PyArray3, PyReadonlyArray3, PyUntypedArrayMethods, ndarray::Array3,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use stacker_core::image::PlanarImage;

use crate::convert::{
    Encoding, planar_from_f32, planar_from_u8, planar_from_u16, planar_to_f32, planar_to_u8,
    planar_to_u16,
};

/// The data of a `[H, W, 3]` array as a flat row-major slice, with its `(height, width)`.
///
/// # Errors
/// `ValueError` when the last dimension is not 3 or the array is not C-contiguous.
fn rgb_slice<'a, T: Element>(
    arr: &'a PyReadonlyArray3<'_, T>,
) -> PyResult<(&'a [T], usize, usize)> {
    let shape = arr.shape();
    let (height, width, channels) = (shape[0], shape[1], shape[2]);
    if channels != 3 {
        return Err(PyValueError::new_err(format!(
            "expected a [H, W, 3] RGB array, got last dimension {channels} (shape {shape:?})"
        )));
    }
    let slice = arr.as_slice().map_err(|_| {
        PyValueError::new_err(
            "input array must be C-contiguous (row-major); call `numpy.ascontiguousarray(arr)` \
             first if it came from a transpose, slice or view",
        )
    })?;
    Ok((slice, height, width))
}

/// Convert a `uint8` array.
///
/// # Errors
/// `ValueError` for a wrong shape or a non-contiguous array.
pub fn numpy_u8_to_planar(
    arr: &PyReadonlyArray3<'_, u8>,
    encoding: Encoding,
) -> PyResult<PlanarImage<f32>> {
    let (data, height, width) = rgb_slice(arr)?;
    Ok(planar_from_u8(data, width, height, encoding)?)
}

/// Convert a `uint16` array.
///
/// # Errors
/// `ValueError` for a wrong shape or a non-contiguous array.
pub fn numpy_u16_to_planar(
    arr: &PyReadonlyArray3<'_, u16>,
    encoding: Encoding,
) -> PyResult<PlanarImage<f32>> {
    let (data, height, width) = rgb_slice(arr)?;
    Ok(planar_from_u16(data, width, height, encoding)?)
}

/// Convert a `float32` array.
///
/// # Errors
/// `ValueError` for a wrong shape or a non-contiguous array.
pub fn numpy_f32_to_planar(
    arr: &PyReadonlyArray3<'_, f32>,
    encoding: Encoding,
) -> PyResult<PlanarImage<f32>> {
    let (data, height, width) = rgb_slice(arr)?;
    Ok(planar_from_f32(data, width, height, encoding)?)
}

/// A `[height, width, 3]` array from row-major samples.
///
/// # Errors
/// `RuntimeError` when `data` is not `height × width × 3` samples long.
pub fn rgb_array<T: Element>(
    py: Python<'_>,
    height: usize,
    width: usize,
    data: Vec<T>,
) -> PyResult<Bound<'_, PyArray3<T>>> {
    let array = Array3::from_shape_vec((height, width, 3), data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

/// Convert `img` to a `uint8` array (see [`crate::convert::planar_to_u8`]).
///
/// # Errors
/// `RuntimeError` when the image planes disagree with its size.
pub fn planar_to_numpy_u8<'py>(
    py: Python<'py>,
    img: &PlanarImage<f32>,
    encoding: Encoding,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    rgb_array(py, img.height, img.width, planar_to_u8(img, encoding))
}

/// Convert `img` to a `uint16` array (see [`crate::convert::planar_to_u16`]).
///
/// # Errors
/// `RuntimeError` when the image planes disagree with its size.
pub fn planar_to_numpy_u16<'py>(
    py: Python<'py>,
    img: &PlanarImage<f32>,
    encoding: Encoding,
) -> PyResult<Bound<'py, PyArray3<u16>>> {
    rgb_array(py, img.height, img.width, planar_to_u16(img, encoding))
}

/// Convert `img` to a `float32` array (see [`crate::convert::planar_to_f32`]).
///
/// # Errors
/// `RuntimeError` when the image planes disagree with its size.
pub fn planar_to_numpy_f32<'py>(
    py: Python<'py>,
    img: &PlanarImage<f32>,
    encoding: Encoding,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    rgb_array(py, img.height, img.width, planar_to_f32(img, encoding))
}
