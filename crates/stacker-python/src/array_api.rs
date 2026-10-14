//! The in-RAM path: [`stack_arrays`], the numpy counterpart of `stack_files`.
//!
//! The caller already holds the frames as numpy arrays, so nothing is read from disk; the
//! whole stack has to fit in memory. The flow is `stack::stack_in_ram` (the same
//! `stacker_pipeline::session` stages the desktop app runs): align every frame to the first,
//! crop to the area every frame covers, auto-cull and order, fuse, stretch back if asked.
//!
//! # Colour
//!
//! The pipeline computes in linear light. By default (`encoding="srgb"`) the arrays are read
//! as sRGB-encoded (the usual state of 8- and 16-bit image data and of float images in
//! `0.0..=1.0`), converted to linear light with the exact transfer function, and the result is
//! encoded back with the same function: the output has the dtype and the encoding of the
//! input. With `encoding="linear"` the samples are taken as linear light and no transfer
//! function is applied in either direction.

use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray3};
use pyo3::{exceptions::PyValueError, prelude::*};
use stacker_core::image::PlanarImage;
use stacker_pipeline::tiled::Hooks;

use crate::{
    bridge::{SignalCancel, emit_warnings, python_progress, to_py_err},
    convert::Encoding,
    converter::{
        numpy_f32_to_planar, numpy_u8_to_planar, numpy_u16_to_planar, planar_to_numpy_f32,
        planar_to_numpy_u8, planar_to_numpy_u16,
    },
    settings::PyStackingSettings,
    stack::{CollectLog, FuseMode, select_indices, stack_in_ram},
};

/// The input and output dtypes of [`stack_arrays`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArrayDtype {
    U8,
    U16,
    F32,
}

/// Convert one `[H, W, 3]` array of dtype `uint8`, `uint16` or `float32`, and say which.
///
/// # Errors
/// `ValueError` for any other dtype or rank, a wrong last dimension or a non-contiguous array.
fn frame_to_planar(
    obj: &Bound<'_, PyAny>,
    index: usize,
    encoding: Encoding,
) -> PyResult<(PlanarImage<f32>, ArrayDtype)> {
    let context = |e: PyErr| PyValueError::new_err(format!("stack_arrays: frame {index}: {e}"));
    if let Ok(arr) = obj.cast::<PyArray3<u8>>() {
        let readonly: PyReadonlyArray3<'_, u8> = arr.readonly();
        let planar = numpy_u8_to_planar(&readonly, encoding).map_err(context)?;
        return Ok((planar, ArrayDtype::U8));
    }
    if let Ok(arr) = obj.cast::<PyArray3<u16>>() {
        let readonly: PyReadonlyArray3<'_, u16> = arr.readonly();
        let planar = numpy_u16_to_planar(&readonly, encoding).map_err(context)?;
        return Ok((planar, ArrayDtype::U16));
    }
    if let Ok(arr) = obj.cast::<PyArray3<f32>>() {
        let readonly: PyReadonlyArray3<'_, f32> = arr.readonly();
        let planar = numpy_f32_to_planar(&readonly, encoding).map_err(context)?;
        return Ok((planar, ArrayDtype::F32));
    }
    Err(PyValueError::new_err(format!(
        "stack_arrays: frame {index} must be a [H, W, 3] numpy array of dtype uint8, uint16 or \
         float32"
    )))
}

/// Focus-stack a list of numpy `[H, W, 3]` arrays that are already in memory.
///
/// `frames` must share one shape and one dtype (`uint8`, `uint16` or `float32`, C-contiguous);
/// the first is the alignment reference. `mode` is `"apex"` (default), `"relief"` or
/// `"strata"`. `encoding` is `"srgb"` (default) or `"linear"`, see the module documentation.
/// Returns an array of the dtype of the input, `[H', W', 3]`; `H'` and `W'` are smaller than
/// the input when `settings.crop_to_common_area` cropped the result (and
/// `settings.resize_cropped_to_original` did not stretch it back).
///
/// The settings that apply: everything about alignment, `auto_cull`,
/// `auto_cull_threshold_pct`, `sort_by_sharpness`, `stack_every_nth` and
/// `preprocessing.sort_reverse` (which frames, in which order), the fusion settings of the
/// chosen mode, `crop_to_common_area`, `resize_cropped_to_original` and `compute_device`.
/// Rotation, crop and resize of `preprocessing` are not applied to arrays. A frame that fell
/// back during the alignment, a refused crop and similar notes arrive as `UserWarning`.
///
/// `progress`, if given, is called as `progress(stage, current, total)`; see the README.
///
/// The compute runs with the GIL released and can be interrupted with Ctrl-C
/// (`KeyboardInterrupt`).
///
/// # Errors
/// `ValueError` for an empty list, an unknown `mode` or `encoding`, invalid settings, arrays
/// that are not `[H, W, 3]` of a supported dtype or not contiguous, frames of different
/// dtypes or sizes, a cull that leaves no frame. `RuntimeError` for failures inside the
/// pipeline.
#[pyfunction]
#[pyo3(signature = (
    frames,
    settings,
    mode="apex".to_string(),
    encoding="srgb".to_string(),
    progress=None,
))]
#[allow(clippy::needless_pass_by_value)]
pub fn stack_arrays<'py>(
    py: Python<'py>,
    frames: Vec<Bound<'py, PyAny>>,
    settings: &PyStackingSettings,
    mode: String,
    encoding: String,
    progress: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if frames.is_empty() {
        return Err(PyValueError::new_err(
            "stack_arrays: `frames` must not be empty",
        ));
    }
    let fuse_mode = FuseMode::parse(&mode)?;
    let encoding = Encoding::parse(&encoding)?;
    let core = settings.clone().into_core()?;

    // Convert while the GIL is held: numpy arrays are Python objects.
    let selected = select_indices(
        frames.len(),
        core.preprocessing.sort_reverse,
        core.stack_every_nth,
    );
    let mut planars = Vec::with_capacity(selected.len());
    let mut dtype: Option<ArrayDtype> = None;
    for &index in &selected {
        let (planar, this_dtype) = frame_to_planar(&frames[index], index, encoding)?;
        match dtype {
            None => dtype = Some(this_dtype),
            Some(expected) if expected == this_dtype => {}
            Some(_) => {
                return Err(PyValueError::new_err(format!(
                    "stack_arrays: frame {index} has a different dtype than the first frame; \
                     all frames must share one dtype"
                )));
            }
        }
        planars.push(planar);
    }
    let dtype = dtype.ok_or_else(|| PyValueError::new_err("stack_arrays: no frame selected"))?;

    let (result, warnings) = py.detach(|| {
        let log = CollectLog::default();
        let forwarder = python_progress(progress.as_ref());
        let cancel = SignalCancel::new();
        let hooks = Hooks {
            progress: &forwarder,
            cancel: &cancel,
            log: &log,
        };
        let result = stack_in_ram(planars, &core, fuse_mode, &hooks);
        (result, log.into_warnings())
    });
    emit_warnings(py, &warnings)?;
    let fused = result.map_err(to_py_err)?;

    Ok(match dtype {
        ArrayDtype::U8 => planar_to_numpy_u8(py, &fused, encoding)?.into_any(),
        ArrayDtype::U16 => planar_to_numpy_u16(py, &fused, encoding)?.into_any(),
        ArrayDtype::F32 => planar_to_numpy_f32(py, &fused, encoding)?.into_any(),
    })
}
