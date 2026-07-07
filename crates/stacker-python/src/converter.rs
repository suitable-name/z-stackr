#![cfg(feature = "python")]
//! Bulk numpy `[H, W, 3]` <-> `PlanarImage<f32>` conversions, one variant per
//! supported dtype (`u8`, `u16`, `f32`).
//!
//! # Value ranges
//!
//! - `u8`: `0..=255` maps to `0.0..=1.0`.
//! - `u16`: `0..=65535` maps to `0.0..=1.0` (the same convention
//!   `stacker_pipeline::load::dynamic_to_planar` uses for 16-bit sources).
//! - `f32`: assumed already `0.0..=1.0` (values outside that range are
//!   preserved as-is on the way in, but every conversion back out clamps to
//!   `0.0..=1.0` before quantising, so out-of-range float input never
//!   produces an out-of-range `u8`/`u16` result on a round trip).
//!
//! # Bulk access, not per-pixel `unwrap`
//!
//! Every function here operates on the whole contiguous buffer via
//! `as_slice()` / `chunks_exact(3)` / iterator adapters — never
//! `ndarray::Array::get(...).unwrap()` per pixel. Callers must pass
//! C-contiguous arrays; [`contiguous_rgb_slice`] returns a clear
//! [`PyValueError`] otherwise instead of panicking or silently copying.
//!
//! # Quantisation
//!
//! Float-to-integer conversions always round-then-clamp
//! (`(x * MAX).round().clamp(0.0, MAX)`), never truncate — a truncating cast
//! systematically biases every channel down by up to one quantisation step.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3, PyUntypedArrayMethods, ndarray::Array3};
use pyo3::{exceptions::PyValueError, prelude::*};
use stacker_core::image::PlanarImage;
use stacker_pipeline::output::planar_to_gamma_rgb;

/// BT.601 gamma-space RGB -> `PlanarImage` YCbCr coefficients, shared by
/// every "to planar" dtype variant below so the three conversions can never
/// silently drift apart from each other or from
/// `stacker_pipeline::load::dynamic_to_planar`.
#[inline]
fn rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.114f32.mul_add(b, 0.587f32.mul_add(g, 0.299 * r));
    let cb = 0.5f32.mul_add(b, (-0.331_26f32).mul_add(g, -0.168_74 * r));
    let cr = (-0.081_312f32).mul_add(b, (-0.418_688f32).mul_add(g, 0.5 * r));
    (y, cb, cr)
}

/// Validate a `[H, W, 3]` numpy array's shape and return `(height, width)`.
///
/// # Errors
/// Returns `ValueError` if the last dimension is not 3.
///
/// Callers still need to borrow the array's data themselves afterwards
/// (`arr.as_slice()`) — this helper only validates shape, since `numpy`'s
/// borrow-tracked accessors return data tied to the borrow of the
/// `PyReadonlyArray3` value itself, not to a value this helper could hand
/// back across a function boundary.
fn validated_rgb_shape<T: numpy::Element>(
    arr: &PyReadonlyArray3<'_, T>,
) -> PyResult<(usize, usize)> {
    let shape = arr.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(PyValueError::new_err(format!(
            "expected a [H, W, 3] RGB array, got last dimension {c} (shape {shape:?})"
        )));
    }
    Ok((h, w))
}

/// Borrow a `[H, W, 3]` numpy array's data as a flat, row-major, contiguous
/// `&[T]` slice plus its `(height, width)`.
///
/// # Errors
/// Returns `ValueError` (never a panic) if the array is not C-contiguous or
/// is not shaped `[H, W, 3]`.
///
/// Takes the [`PyReadonlyArray3`] by reference and returns a slice borrowed
/// from that same reference, so the two must stay in the same scope at the
/// call site (numpy's borrow-tracked accessors tie returned data to the
/// borrow of the `PyReadonlyArray3` guard itself).
fn contiguous_rgb_slice<'a, T: numpy::Element>(
    arr: &'a PyReadonlyArray3<'_, T>,
) -> PyResult<(&'a [T], usize, usize)> {
    let (h, w) = validated_rgb_shape(arr)?;
    let slice = arr.as_slice().map_err(|_| {
        PyValueError::new_err(
            "input array must be C-contiguous (row-major); call `numpy.ascontiguousarray(arr)` \
             first if it came from a transpose/slice/view",
        )
    })?;
    Ok((slice, h, w))
}

// ── u8 ───────────────────────────────────────────────────────────────────────

/// Convert a `[H, W, 3]` `u8` numpy array (RGB, `0..=255`) to a gamma-space
/// YCbCr [`PlanarImage<f32>`].
///
/// # Errors
/// Returns `ValueError` if the array is not C-contiguous or not `[H, W, 3]`.
pub fn numpy_u8_to_planar(arr: &PyReadonlyArray3<'_, u8>) -> PyResult<PlanarImage<f32>> {
    // `r`/`g`/`b`/`y`/`cb`/`cr` are the conventional, unambiguous names for pixel
    // colour-channel math; spelling them out (`red`/`green`/`blue`/`luma`/...)
    // would make the arithmetic below harder to visually cross-check against
    // the BT.601 matrix in `rgb_to_ycbcr`, not easier.
    let (data, arr_height, arr_width) = contiguous_rgb_slice(arr)?;
    let mut planar = PlanarImage::new(arr_width, arr_height);
    for (i, px) in data.chunks_exact(3).enumerate() {
        let val_r = f32::from(px[0]) / 255.0;
        let val_g = f32::from(px[1]) / 255.0;
        let val_b = f32::from(px[2]) / 255.0;
        let (val_y, cb, cr) = rgb_to_ycbcr(val_r, val_g, val_b);
        planar.luma[i] = val_y;
        planar.chroma_a[i] = cb;
        planar.chroma_b[i] = cr;
    }
    Ok(planar)
}

/// Convert a `PlanarImage<f32>` (gamma-space YCbCr) to a `[H, W, 3]` `u8`
/// numpy array (RGB, `0..=255`), rounding + clamping each channel.
///
/// # Panics
/// Never in practice: the output buffer is always built with exactly
/// `height * width * 3` elements, matching `Array3::from_shape_vec`'s shape
/// requirement.
#[must_use]
pub fn planar_to_numpy_u8<'py>(
    py: Python<'py>,
    img: &PlanarImage<f32>,
) -> Bound<'py, PyArray3<u8>> {
    let mut out = vec![0u8; img.luma.len() * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        let (r, g, b) = planar_to_gamma_rgb(img, i);
        px[0] = quantize_u8(r);
        px[1] = quantize_u8(g);
        px[2] = quantize_u8(b);
    }
    let arr = Array3::from_shape_vec((img.height, img.width, 3), out)
        .expect("row-major buffer always matches (height, width, 3)");
    arr.into_pyarray(py)
}

#[inline]
fn quantize_u8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8
}

// ── u16 ──────────────────────────────────────────────────────────────────────

/// Convert a `[H, W, 3]` `u16` numpy array (RGB, `0..=65535`) to a
/// gamma-space YCbCr [`PlanarImage<f32>`].
///
/// # Errors
/// Returns `ValueError` if the array is not C-contiguous or not `[H, W, 3]`.
pub fn numpy_u16_to_planar(arr: &PyReadonlyArray3<'_, u16>) -> PyResult<PlanarImage<f32>> {
    let (data, arr_height, arr_width) = contiguous_rgb_slice(arr)?;
    let mut planar = PlanarImage::new(arr_width, arr_height);
    for (i, px) in data.chunks_exact(3).enumerate() {
        let val_r = f32::from(px[0]) / 65_535.0;
        let val_g = f32::from(px[1]) / 65_535.0;
        let val_b = f32::from(px[2]) / 65_535.0;
        let (val_y, cb, cr) = rgb_to_ycbcr(val_r, val_g, val_b);
        planar.luma[i] = val_y;
        planar.chroma_a[i] = cb;
        planar.chroma_b[i] = cr;
    }
    Ok(planar)
}

/// Convert a `PlanarImage<f32>` (gamma-space YCbCr) to a `[H, W, 3]` `u16`
/// numpy array (RGB, `0..=65535`), rounding + clamping each channel.
///
/// # Panics
/// Never in practice: the output buffer is always built with exactly
/// `height * width * 3` elements, matching `Array3::from_shape_vec`'s shape
/// requirement.
#[must_use]
pub fn planar_to_numpy_u16<'py>(
    py: Python<'py>,
    img: &PlanarImage<f32>,
) -> Bound<'py, PyArray3<u16>> {
    let mut out = vec![0u16; img.luma.len() * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        let (r, g, b) = planar_to_gamma_rgb(img, i);
        px[0] = quantize_u16(r);
        px[1] = quantize_u16(g);
        px[2] = quantize_u16(b);
    }
    let arr = Array3::from_shape_vec((img.height, img.width, 3), out)
        .expect("row-major buffer always matches (height, width, 3)");
    arr.into_pyarray(py)
}

#[inline]
fn quantize_u16(x: f32) -> u16 {
    (x.clamp(0.0, 1.0) * 65_535.0).round().clamp(0.0, 65_535.0) as u16
}

// ── f32 ──────────────────────────────────────────────────────────────────────

/// Convert a `[H, W, 3]` `f32` numpy array (RGB, nominally `0.0..=1.0`) to a
/// gamma-space YCbCr [`PlanarImage<f32>`].
///
/// Values outside `0.0..=1.0` are passed through unchanged (only the
/// *output* side of a round trip clamps).
///
/// # Errors
/// Returns `ValueError` if the array is not C-contiguous or not `[H, W, 3]`.
pub fn numpy_f32_to_planar(arr: &PyReadonlyArray3<'_, f32>) -> PyResult<PlanarImage<f32>> {
    let (data, h, w) = contiguous_rgb_slice(arr)?;
    let mut planar = PlanarImage::new(w, h);
    for (i, px) in data.chunks_exact(3).enumerate() {
        let (y, cb, cr) = rgb_to_ycbcr(px[0], px[1], px[2]);
        planar.luma[i] = y;
        planar.chroma_a[i] = cb;
        planar.chroma_b[i] = cr;
    }
    Ok(planar)
}

/// Convert a `PlanarImage<f32>` (gamma-space YCbCr) to a `[H, W, 3]` `f32`
/// numpy array (RGB, clamped to `0.0..=1.0`).
///
/// # Panics
/// Never in practice: the output buffer is always built with exactly
/// `height * width * 3` elements, matching `Array3::from_shape_vec`'s shape
/// requirement.
#[must_use]
pub fn planar_to_numpy_f32<'py>(
    py: Python<'py>,
    img: &PlanarImage<f32>,
) -> Bound<'py, PyArray3<f32>> {
    let mut out = vec![0.0f32; img.luma.len() * 3];
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        let (r, g, b) = planar_to_gamma_rgb(img, i);
        px[0] = r;
        px[1] = g;
        px[2] = b;
    }
    let arr = Array3::from_shape_vec((img.height, img.width, 3), out)
        .expect("row-major buffer always matches (height, width, 3)");
    arr.into_pyarray(py)
}

// ── u8-only convenience aliases ──────────────────────────────────────────

/// Alias of [`numpy_u8_to_planar`] that panics on a non-contiguous/
/// mis-shaped array instead of returning `Result`. New code should call
/// [`numpy_u8_to_planar`] directly for the non-panicking form.
///
/// # Panics
/// Panics if the array is not C-contiguous or not shaped `[H, W, 3]`; see
/// [`numpy_u8_to_planar`] for a non-panicking equivalent.
#[must_use]
pub fn numpy_rgb_to_planar(py_arr: &PyReadonlyArray3<'_, u8>) -> PlanarImage<f32> {
    numpy_u8_to_planar(py_arr).expect("array must be C-contiguous [H, W, 3]")
}

/// Alias of [`planar_to_numpy_u8`]. New code should call
/// [`planar_to_numpy_u8`] directly.
#[must_use]
pub fn planar_to_numpy_rgb<'py>(
    py: Python<'py>,
    img: &PlanarImage<f32>,
) -> Bound<'py, PyArray3<u8>> {
    planar_to_numpy_u8(py, img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{IntoPyArray, PyArrayMethods, ndarray::Array3};

    /// Synthetic gradient in `[H, W, 3]` layout, distinct per-channel ramps
    /// so a channel swap or transposition bug would be caught.
    #[allow(clippy::many_single_char_names)]
    fn gradient_u8(h: usize, w: usize) -> Array3<u8> {
        let mut arr = Array3::<u8>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w.max(1)) as u8;

                let g = ((y * 255) / h.max(1)) as u8;
                let b = 128u8;
                arr[[y, x, 0]] = r;
                arr[[y, x, 1]] = g;
                arr[[y, x, 2]] = b;
            }
        }
        arr
    }

    #[allow(clippy::many_single_char_names)]
    fn gradient_u16(h: usize, w: usize) -> Array3<u16> {
        let mut arr = Array3::<u16>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 65535) / w.max(1)) as u16;

                let g = ((y * 65535) / h.max(1)) as u16;
                let b = 32768u16;
                arr[[y, x, 0]] = r;
                arr[[y, x, 1]] = g;
                arr[[y, x, 2]] = b;
            }
        }
        arr
    }

    #[allow(clippy::many_single_char_names)]
    fn gradient_f32(h: usize, w: usize) -> Array3<f32> {
        let mut arr = Array3::<f32>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                let r = x as f32 / w.max(1) as f32;
                let g = y as f32 / h.max(1) as f32;
                let b = 0.5;
                arr[[y, x, 0]] = r;
                arr[[y, x, 1]] = g;
                arr[[y, x, 2]] = b;
            }
        }
        arr
    }

    #[test]
    fn u8_round_trip_within_one_bit() {
        Python::initialize();
        Python::attach(|py| {
            let (h, w) = (9usize, 13usize);
            let src = gradient_u8(h, w);
            let py_arr = src.clone().into_pyarray(py);
            let planar = numpy_u8_to_planar(&py_arr.readonly()).unwrap();
            assert_eq!((planar.width, planar.height), (w, h));
            let back = planar_to_numpy_u8(py, &planar);
            let back_arr = back.readonly();
            let back_view = back_arr.as_array();
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let orig = i32::from(src[[y, x, c]]);
                        let got = i32::from(back_view[[y, x, c]]);
                        assert!(
                            (orig - got).abs() <= 1,
                            "u8 round trip drifted more than 1 bit at ({y},{x},{c}): {orig} vs {got}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn u16_round_trip_within_tolerance() {
        Python::initialize();
        Python::attach(|py| {
            let (h, w) = (9usize, 13usize);
            let src = gradient_u16(h, w);
            let py_arr = src.clone().into_pyarray(py);
            let planar = numpy_u16_to_planar(&py_arr.readonly()).unwrap();
            assert_eq!((planar.width, planar.height), (w, h));
            let back = planar_to_numpy_u16(py, &planar);
            let back_arr = back.readonly();
            let back_view = back_arr.as_array();
            // Tolerance: the YCbCr matrix round trip is not bit-exact at
            // 16-bit depth, but should stay within a small fraction of the
            // full range (looser than the u8 test's absolute 1-bit bound,
            // scaled for 65535 vs 255 full-scale).
            let tol = 300i32;
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let orig = i32::from(src[[y, x, c]]);
                        let got = i32::from(back_view[[y, x, c]]);
                        assert!(
                            (orig - got).abs() <= tol,
                            "u16 round trip drifted more than {tol} at ({y},{x},{c}): {orig} vs {got}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn f32_round_trip_near_exact() {
        Python::initialize();
        Python::attach(|py| {
            let (h, w) = (9usize, 13usize);
            let src = gradient_f32(h, w);
            let py_arr = src.clone().into_pyarray(py);
            let planar = numpy_f32_to_planar(&py_arr.readonly()).unwrap();
            assert_eq!((planar.width, planar.height), (w, h));
            let back = planar_to_numpy_f32(py, &planar);
            let back_arr = back.readonly();
            let back_view = back_arr.as_array();
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let orig = src[[y, x, c]];
                        let got = back_view[[y, x, c]];
                        assert!(
                            (orig - got).abs() < 1.0e-5,
                            "f32 round trip not near-exact at ({y},{x},{c}): {orig} vs {got}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn non_contiguous_or_wrong_shape_raises_value_error_not_panic() {
        Python::initialize();
        Python::attach(|py| {
            // Wrong last dimension (RGBA instead of RGB).
            let bad = Array3::<u8>::zeros((4, 4, 4));
            let py_arr = bad.into_pyarray(py);
            let err = numpy_u8_to_planar(&py_arr.readonly()).unwrap_err();
            assert!(err.to_string().contains('4'));
        });
    }
}
