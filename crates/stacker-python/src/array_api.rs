#![cfg(feature = "python")]
//! In-RAM / array API: [`stack_arrays`], the numpy counterpart of
//! `crate::file_api::stack_files`.
//!
//! # The two paths
//!
//! The file API (`stack_files`) is out-of-core: it streams tiles through a
//! bounded-memory pipeline so a stack can exceed available RAM. This
//! module's [`stack_arrays`] is the opposite trade-off — it requires the
//! **entire** stack to already be resident in memory (the caller passed a
//! Python list of numpy arrays, which by definition already fit), and in
//! exchange it can do things the tiled pipeline structurally cannot:
//! `auto_cull` / `sort_by_sharpness` (both need the whole aligned stack at
//! once to compare frames — see `PyStackingSettings::auto_cull`'s doc
//! comment) actually take effect here, unlike in `stack_files`.
//!
//! Use `stack_arrays` for stacks that already fit comfortably in memory
//! (e.g. already-loaded microscopy captures, a live-acquisition buffer);
//! use `stack_files` for anything large enough that "decode every frame at
//! once" would risk an out-of-memory condition.
//!
//! # Flow (mirrors `apps/stacker-gui/src/main.rs`'s in-RAM Stack handler)
//!
//! 1. **Convert** every input array to `PlanarImage<f32>` (dtype-dispatched
//!    via `crate::converter`).
//! 2. **Align** sequentially: frame 0 is the reference; each subsequent
//!    frame is registered against the *previously warped* frame (a rolling
//!    reference), warm-started from the previous frame's solved matrix,
//!    via `stacker_align::align_frame`. Honors `settings.alignment_mode`.
//!    When `settings.correct_brightness` is set, a `BrightnessTarget` is
//!    built from frame 0 and applied to every warped frame.
//! 3. **Coverage + crop**: each frame's `coverage_mask` is intersected into
//!    a running common mask; when `settings.crop_to_common_area` is set,
//!    `resolve_common_crop` resolves a crop rectangle (with the same
//!    25%-of-canvas rogue-frame guard as the file pipeline) and every frame
//!    is cropped to it.
//! 4. **Resize**: when `settings.crop_to_common_area &&
//!    settings.resize_cropped_to_original` and a crop actually shrank the
//!    canvas, the fused result (step 6) is resampled back up to the
//!    original pre-crop dimensions via `resize_planar_clamped`.
//! 5. **Sort/cull**: when `settings.auto_cull` and/or
//!    `settings.sort_by_sharpness` are set, `stacker_algo::optimize::
//!    optimize_stack` (fed the aligned+cropped frame set) reorders and/or
//!    drops frames exactly like the GUI's Sort/Cull buttons.
//! 6. **Fuse**: `apex` (honouring `use_all_color_channels`, `grit_suppression`,
//!    `pyramid_levels`), `relief` (honouring `relief_engine`,
//!    `relief_estimation_radius`, `relief_smoothing_radius`, `relief_contrast_pct`,
//!    `relief_auto_detect`), or `strata` (honouring `strata_base_radius`,
//!    `strata_detail_focus`).
//! 7. **Convert back** to the same numpy dtype as the input.
//!
//! The whole compute-heavy body runs with the GIL released
//! (`Python::detach`).

use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray3};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use stacker_algo::{
    apex::fuse::build_and_fuse_pyramids,
    optimize::optimize_stack,
    relief::{
        fuse::{
            auto_contrast_threshold, compute_relief_smls, fuse_relief_multigrid,
            fuse_relief_with_mask,
        },
        threshold::ReliefSettings,
    },
    strata::{StrataParams, fuse_strata},
};
use stacker_align::{
    Matrix3, align_frame,
    brightness::BrightnessTarget,
    transform::{coverage_mask, intersect_coverage, resize_planar_clamped, resolve_common_crop},
};
use stacker_core::{image::PlanarImage, memory::extract_tile, settings::AlignmentModeSetting};

use crate::{
    converter::{
        numpy_f32_to_planar, numpy_u8_to_planar, numpy_u16_to_planar, planar_to_numpy_f32,
        planar_to_numpy_u8, planar_to_numpy_u16,
    },
    settings::PyStackingSettings,
};

/// The three supported input/output dtypes for [`stack_arrays`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArrayDtype {
    U8,
    U16,
    F32,
}

/// The three fusion engines [`stack_arrays`] supports (`"ai"` needs a loaded
/// model and tiled inference machinery this in-RAM path does not wire up —
/// use `stack_files`/`batch_stack` for that).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FuseMode {
    Apex,
    Relief,
    Strata,
}

/// Sniff a `[H, W, 3]` array's dtype by attempting a zero-copy downcast to
/// each supported `PyReadonlyArray3<T>` in turn, converting whichever one
/// matches straight to `PlanarImage<f32>` (bulk slice conversion — see
/// `crate::converter`'s module docs). Returns the recognised dtype
/// alongside the converted frame so the caller can convert the *output*
/// back to the same dtype.
///
/// # Errors
/// Raises `ValueError` if the array is none of `uint8`/`uint16`/`float32`,
/// or is not C-contiguous / not shaped `[H, W, 3]` (surfaced from the
/// underlying `crate::converter` function).
fn frame_to_planar(obj: &Bound<'_, PyAny>) -> PyResult<(PlanarImage<f32>, ArrayDtype)> {
    if let Ok(arr) = obj.cast::<PyArray3<u8>>() {
        let readonly: PyReadonlyArray3<'_, u8> = arr.readonly();
        return Ok((numpy_u8_to_planar(&readonly)?, ArrayDtype::U8));
    }
    if let Ok(arr) = obj.cast::<PyArray3<u16>>() {
        let readonly: PyReadonlyArray3<'_, u16> = arr.readonly();
        return Ok((numpy_u16_to_planar(&readonly)?, ArrayDtype::U16));
    }
    if let Ok(arr) = obj.cast::<PyArray3<f32>>() {
        let readonly: PyReadonlyArray3<'_, f32> = arr.readonly();
        return Ok((numpy_f32_to_planar(&readonly)?, ArrayDtype::F32));
    }
    Err(PyValueError::new_err(
        "unsupported array dtype: expected a [H, W, 3] numpy array of dtype uint8, uint16, or float32",
    ))
}

/// Focus-stack a list of already-in-RAM numpy `[H, W, 3]` arrays entirely in
/// memory, mirroring the GUI's in-RAM Stack pipeline.
///
/// See the module docs for the exact step-by-step flow and which `settings`
/// fields each step honours.
///
/// `frames` must all share the same shape and dtype (one of `uint8`
/// `0..=255`, `uint16` `0..=65535`, or `float32` nominally `0.0..=1.0`;
/// mixed dtypes across the list raise `ValueError`). `mode` selects the
/// fusion algorithm: `"apex"` (Laplacian-pyramid fusion; default),
/// `"relief"` (SML + guided-filter depth-map fusion), or `"strata"`
/// (guided-filter soft-blend fusion; see `docs/strata-fusion-design.md`) —
/// matching `PipelineParams.mode`/`--mode`'s vocabulary, minus `"ai"` (neural
/// fusion needs a loaded model and tiled inference machinery this in-RAM
/// path does not wire up; use `stack_files`/`batch_stack` for AI mode). When
/// `mode == "relief"`, `settings.relief_engine` picks which of the two
/// `Relief` engines runs; when `mode == "strata"`,
/// `settings.strata_base_radius` controls the base/detail split and
/// `settings.strata_detail_focus` controls the detail-weight
/// re-concentration exponent. Returns a
/// numpy array of the SAME dtype as the input, shape `[H',
/// W', 3]` — `H'`/`W'` may be smaller than the input when
/// `settings.crop_to_common_area` cropped the result and
/// `settings.resize_cropped_to_original` was not set to restretch it back.
///
/// This is the one path in this crate where `settings.auto_cull` and
/// `settings.sort_by_sharpness` actually take effect (see the module docs);
/// `stack_files`' tiled out-of-core path cannot support either.
///
/// The whole compute-heavy body runs with the GIL released so other Python
/// threads are not frozen during a long in-RAM stack.
///
/// # Errors
/// Raises `ValueError` if: `frames` is empty; `mode` is not `"apex"`/
/// `"relief"`/`"strata"`; any array's dtype is not
/// `uint8`/`uint16`/`float32`; arrays are
/// not C-contiguous or not `[H, W, 3]`; frames have mismatched
/// shapes/dtypes; or `settings.alignment_mode`/`relief_engine`/
/// `image_saving.output_format` is invalid. Raises `RuntimeError` if the
/// fused result is empty (should not happen for non-empty, equally-shaped
/// input).
///
/// # Panics
/// Never in practice: the dtype tracked while converting the (non-empty)
/// `frames` list is always set by the time it is unwrapped below.
// `frames`/`mode` are taken by value because pyo3's `#[pyfunction]` FFI
// boundary requires owned argument types here (a `#[pyo3(signature = ...)]`
// string default and a `Vec` extracted from a Python list are not
// expressible as borrows) — `clippy::needless_pass_by_value` does not know
// about that constraint.

#[pyfunction]
#[pyo3(signature = (frames, settings, mode="apex".to_string()))]
pub fn stack_arrays<'py>(
    py: Python<'py>,
    frames: Vec<Bound<'py, PyAny>>,
    settings: &PyStackingSettings,
    mode: String,
) -> PyResult<Bound<'py, PyAny>> {
    if frames.is_empty() {
        return Err(PyValueError::new_err(
            "stack_arrays: `frames` must not be empty",
        ));
    }
    let fuse_mode = match mode.as_str() {
        "apex" => FuseMode::Apex,
        "relief" => FuseMode::Relief,
        "strata" => FuseMode::Strata,
        other => {
            return Err(PyValueError::new_err(format!(
                "stack_arrays: invalid mode {other:?}; valid options are: apex, relief, strata \
                 (use stack_files/batch_stack for \"ai\")"
            )));
        }
    };

    // Convert every frame while still holding the GIL (numpy arrays are
    // Python objects and must not be touched after we detach it below).
    let mut planar_frames = Vec::with_capacity(frames.len());
    let mut dtype: Option<ArrayDtype> = None;
    for (idx, obj) in frames.iter().enumerate() {
        let (planar, this_dtype) = frame_to_planar(obj)?;
        match dtype {
            None => dtype = Some(this_dtype),
            Some(expected) if expected == this_dtype => {}
            Some(_) => {
                return Err(PyValueError::new_err(format!(
                    "stack_arrays: frame {idx} has a different dtype than frame 0 — all frames must share one dtype"
                )));
            }
        }
        planar_frames.push(planar);
    }
    for (idx, f) in planar_frames.iter().enumerate().skip(1) {
        if f.width != planar_frames[0].width || f.height != planar_frames[0].height {
            return Err(PyValueError::new_err(format!(
                "stack_arrays: frame {idx} has shape ({}, {}) but frame 0 has ({}, {}) — all frames must share one shape",
                f.height, f.width, planar_frames[0].height, planar_frames[0].width
            )));
        }
    }
    let dtype = dtype.expect("frames is non-empty, so the loop above always sets dtype");

    let rust_settings = settings.clone().into_core()?;

    // Release the GIL for the whole align/crop/resize/fuse compute — none
    // of it touches Python objects.
    let fused = py
        .detach(|| run_in_ram_stack(planar_frames, &rust_settings, fuse_mode))
        .map_err(PyRuntimeError::new_err)?;

    Ok(match dtype {
        ArrayDtype::U8 => planar_to_numpy_u8(py, &fused).into_any(),
        ArrayDtype::U16 => planar_to_numpy_u16(py, &fused).into_any(),
        ArrayDtype::F32 => planar_to_numpy_f32(py, &fused).into_any(),
    })
}

/// The pure-Rust compute core of [`stack_arrays`] — no Python objects
/// touched, so it can run with the GIL released. Returns a plain `String`
/// error (surfaced as `RuntimeError` by the caller) since none of these
/// failure modes are user-input-shaped `ValueError`s (those are all
/// rejected before this function is ever called).
fn align_frames_in_ram(
    frames: &mut [PlanarImage<f32>],
    settings: &stacker_core::settings::StackingSettings,
    img_w: usize,
    img_h: usize,
    brightness_target: Option<&BrightnessTarget>,
) -> Vec<bool> {
    let n_frames = frames.len();
    let mut acc_mask = vec![true; img_w * img_h];
    if n_frames > 1 && settings.alignment_mode != AlignmentModeSetting::None {
        let mut rolling_ref = frames[0].clone();
        let mut prev_matrix = Matrix3::<f32>::identity();

        for (i, frame_slot) in frames.iter_mut().enumerate().take(n_frames).skip(1) {
            let cur = frame_slot.clone();
            let (matrix, warped) = align_frame(
                cur,
                &rolling_ref,
                prev_matrix,
                settings.alignment_mode,
                settings.optimizer,
                true,
                i,
                brightness_target,
            );
            *frame_slot = warped;

            let frame_mask = coverage_mask(&matrix, img_w, img_h);
            intersect_coverage(&mut acc_mask, &frame_mask);

            prev_matrix = matrix;
            rolling_ref = frame_slot.clone();
        }
    } else if settings.correct_brightness {
        // No alignment ran (single frame, or alignment_mode == None), but
        // brightness correction is still meaningful per-frame — apply it
        // directly without a warp, matching the GUI's behaviour of
        // correcting brightness independent of whether alignment moved
        // anything.
        if let Some(target) = brightness_target {
            for frame in frames.iter_mut().skip(1) {
                stacker_align::brightness::apply_brightness_correction(frame, target);
            }
        }
    }
    acc_mask
}

fn run_in_ram_stack(
    mut frames: Vec<PlanarImage<f32>>,
    settings: &stacker_core::settings::StackingSettings,
    fuse_mode: FuseMode,
) -> Result<PlanarImage<f32>, String> {
    // This in-RAM path never routes through `stacker_pipeline::run_pipeline`
    // (which applies `settings.use_gpu` itself for `stack_files`/
    // `batch_stack`), so it must engage the shared runtime GPU switch here
    // — mirroring the GUI's in-RAM Stack handler, which does the same
    // before calling its own `fuse_apex`/`fuse_relief`/`fuse_strata`. No-op
    // in a default, non-`gpu` build.
    #[cfg(feature = "gpu")]
    stacker_core::gpu::set_enabled(settings.use_gpu);

    let n_frames = frames.len();
    let (img_w, img_h) = (frames[0].width, frames[0].height);

    // ── 1. Brightness target (from frame 0, before any warp) ────────────
    let brightness_target = if settings.correct_brightness {
        Some(BrightnessTarget::new(&frames[0]))
    } else {
        None
    };

    // ── 2. Sequential chain alignment ────────────────────────────────────
    let acc_mask = align_frames_in_ram(
        &mut frames,
        settings,
        img_w,
        img_h,
        brightness_target.as_ref(),
    );

    // ── 3. Common-coverage crop ──────────────────────────────────────────
    let orig_canvas = (img_w, img_h);
    let crop_rect = if settings.crop_to_common_area
        && n_frames > 1
        && settings.alignment_mode != AlignmentModeSetting::None
    {
        resolve_common_crop(&acc_mask, img_w, img_h)
    } else {
        None
    };
    if let Some((cx, cy, cw, ch)) = crop_rect {
        for frame in &mut frames {
            *frame = extract_tile(frame, cx, cy, cw, ch);
        }
    }

    // ── 4. stack_every_nth subsampling ───────────────────────────────────
    let frames: Vec<PlanarImage<f32>> = if settings.stack_every_nth > 1 {
        frames
            .into_iter()
            .step_by(settings.stack_every_nth as usize)
            .collect()
    } else {
        frames
    };
    if frames.is_empty() {
        return Err("stack_arrays: no frames left after stack_every_nth subsampling".to_owned());
    }

    // ── 5. Sort / cull (the one path where these actually run) ──────────
    let ordered: Vec<PlanarImage<f32>> = if settings.sort_by_sharpness || settings.auto_cull {
        let threshold = if settings.auto_cull {
            settings.auto_cull_threshold_pct
        } else {
            0.0
        };
        let opt = optimize_stack(&frames, threshold);
        let order = if settings.sort_by_sharpness {
            opt.recommended_order
        } else {
            opt.kept_indices
        };
        if order.is_empty() {
            return Err("stack_arrays: no frames left after auto-cull".to_owned());
        }
        order.into_iter().map(|idx| frames[idx].clone()).collect()
    } else {
        frames
    };

    // ── 6. Fuse ───────────────────────────────────────────────────────────
    let mut fused = match fuse_mode {
        FuseMode::Relief => fuse_relief(&ordered, settings),
        FuseMode::Strata => {
            let params = StrataParams {
                base_radius: settings.strata_base_radius as usize,
                detail_focus: settings.strata_detail_focus,
            };
            fuse_strata(&ordered, &params)
        }
        FuseMode::Apex => build_and_fuse_pyramids(
            &ordered,
            settings.pyramid_levels as usize,
            settings.use_all_color_channels,
            settings.grit_suppression,
        )
        .reconstruct(),
    };

    // ── 7. Restretch back to original canvas ─────────────────────────────
    if settings.crop_to_common_area
        && settings.resize_cropped_to_original
        && crop_rect.is_some()
        && (fused.width, fused.height) != orig_canvas
    {
        fused = resize_planar_clamped(&fused, orig_canvas.0, orig_canvas.1);
    }

    Ok(fused)
}

/// `Relief` fusion honouring `relief_engine`, matching `apps/stacker-gui`'s
/// `fuse_relief` dispatch (minus the interactive preview-popup machinery,
/// which has no meaning outside a GUI event loop).
fn fuse_relief(
    images: &[PlanarImage<f32>],
    settings: &stacker_core::settings::StackingSettings,
) -> PlanarImage<f32> {
    let (smls, max_sml) = compute_relief_smls(images, settings.relief_estimation_radius as usize);

    let contrast_pct = if settings.relief_auto_detect {
        auto_contrast_threshold(&max_sml)
    } else {
        settings.relief_contrast_pct
    };

    let relief_settings = ReliefSettings {
        est_radius: settings.relief_estimation_radius as usize,
        smooth_radius: settings.relief_smoothing_radius as usize,
        contrast_pct,
        absolute_threshold: None,
    };

    if settings.relief_use_multigrid {
        fuse_relief_multigrid(images, &smls, &max_sml, &relief_settings)
    } else {
        fuse_relief_with_mask(images, &smls, &max_sml, &relief_settings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{IntoPyArray, ndarray::Array3};

    /// A 32x32 synthetic frame: sharp (high-frequency) on one half, blurred
    /// (flat) on the other, so `Apex` fusion of two complementary frames
    /// should visibly favour each half from a different source.
    fn half_sharp_frame(w: usize, h: usize, sharp_on_left: bool) -> Array3<u8> {
        let mut arr = Array3::<u8>::zeros((h, w, 3));
        for y in 0..h {
            for x in 0..w {
                let in_sharp_region = if sharp_on_left { x < w / 2 } else { x >= w / 2 };

                let v = if in_sharp_region {
                    // High-frequency checkerboard.
                    if (x + y) % 2 == 0 { 240u8 } else { 20u8 }
                } else {
                    128u8
                };
                arr[[y, x, 0]] = v;
                arr[[y, x, 1]] = v;
                arr[[y, x, 2]] = v;
            }
        }
        arr
    }

    #[test]
    fn stack_arrays_apex_smoke_test_produces_finite_different_result() {
        Python::initialize();
        Python::attach(|py| {
            let (w, h) = (32usize, 32usize);
            let a = half_sharp_frame(w, h, true).into_pyarray(py);
            let b = half_sharp_frame(w, h, false).into_pyarray(py);

            let frames: Vec<Bound<'_, PyAny>> = vec![a.into_any(), b.into_any()];
            // Deterministic single-shot fusion: disable auto_cull/sort so
            // both frames definitely participate.
            let settings = PyStackingSettings {
                alignment_mode: "None".to_owned(),
                auto_cull: false,
                sort_by_sharpness: false,
                correct_brightness: false,
                ..PyStackingSettings::default()
            };

            let result = stack_arrays(py, frames, &settings, "apex".to_owned())
                .expect("stack_arrays should succeed");
            let arr: &Bound<'_, PyArray3<u8>> = result
                .cast()
                .expect("output dtype should match input (uint8)");
            let readonly = arr.readonly();
            let view = readonly.as_array();
            assert_eq!(view.shape(), [h, w, 3]);

            // Every value must be finite (trivially true for u8, but the
            // shape/dtype checks above are the real assertions) and the
            // result must differ from a flat image (proving actual fusion
            // happened, not just a pass-through of one frame).
            let all_same = view.iter().all(|&v| v == view[[0, 0, 0]]);
            assert!(!all_same, "fused output must not be a flat/constant image");
        });
    }

    #[test]
    fn stack_arrays_relief_smoke_test_produces_finite_different_result() {
        Python::initialize();
        Python::attach(|py| {
            let (w, h) = (32usize, 32usize);
            let a = half_sharp_frame(w, h, true).into_pyarray(py);
            let b = half_sharp_frame(w, h, false).into_pyarray(py);

            let frames: Vec<Bound<'_, PyAny>> = vec![a.into_any(), b.into_any()];
            let settings = PyStackingSettings {
                alignment_mode: "None".to_owned(),
                auto_cull: false,
                sort_by_sharpness: false,
                correct_brightness: false,
                ..PyStackingSettings::default()
            };

            let result = stack_arrays(py, frames, &settings, "relief".to_owned())
                .expect("stack_arrays relief mode should succeed");
            let arr: &Bound<'_, PyArray3<u8>> = result
                .cast()
                .expect("output dtype should match input (uint8)");
            let readonly = arr.readonly();
            let view = readonly.as_array();
            assert_eq!(view.shape(), [h, w, 3]);
            let all_same = view.iter().all(|&v| v == view[[0, 0, 0]]);
            assert!(
                !all_same,
                "fused Relief output must not be a flat/constant image"
            );
        });
    }

    #[test]
    fn stack_arrays_strata_smoke_test_produces_finite_different_result() {
        Python::initialize();
        Python::attach(|py| {
            let (w, h) = (32usize, 32usize);
            let a = half_sharp_frame(w, h, true).into_pyarray(py);
            let b = half_sharp_frame(w, h, false).into_pyarray(py);

            let frames: Vec<Bound<'_, PyAny>> = vec![a.into_any(), b.into_any()];
            let settings = PyStackingSettings {
                alignment_mode: "None".to_owned(),
                auto_cull: false,
                sort_by_sharpness: false,
                correct_brightness: false,
                ..PyStackingSettings::default()
            };

            let result = stack_arrays(py, frames, &settings, "strata".to_owned())
                .expect("stack_arrays strata mode should succeed");
            let arr: &Bound<'_, PyArray3<u8>> = result
                .cast()
                .expect("output dtype should match input (uint8)");
            let readonly = arr.readonly();
            let view = readonly.as_array();
            assert_eq!(view.shape(), [h, w, 3]);
            let all_same = view.iter().all(|&v| v == view[[0, 0, 0]]);
            assert!(
                !all_same,
                "fused Strata output must not be a flat/constant image"
            );
        });
    }

    #[test]
    fn stack_arrays_rejects_empty_list() {
        Python::initialize();
        Python::attach(|py| {
            let settings = PyStackingSettings::default();
            let err = stack_arrays(py, vec![], &settings, "apex".to_owned()).unwrap_err();
            assert!(err.to_string().contains("must not be empty"));
        });
    }

    #[test]
    fn stack_arrays_rejects_invalid_mode() {
        Python::initialize();
        Python::attach(|py| {
            let a = Array3::<u8>::zeros((4, 4, 3)).into_pyarray(py);
            let frames: Vec<Bound<'_, PyAny>> = vec![a.into_any()];
            let settings = PyStackingSettings::default();
            let err = stack_arrays(py, frames, &settings, "ai".to_owned()).unwrap_err();
            assert!(err.to_string().contains("mode"));
        });
    }

    #[test]
    fn stack_arrays_rejects_mismatched_dtypes() {
        Python::initialize();
        Python::attach(|py| {
            let a = Array3::<u8>::zeros((4, 4, 3)).into_pyarray(py);
            let b = Array3::<f32>::zeros((4, 4, 3)).into_pyarray(py);
            let frames: Vec<Bound<'_, PyAny>> = vec![a.into_any(), b.into_any()];
            let settings = PyStackingSettings::default();
            let err = stack_arrays(py, frames, &settings, "apex".to_owned()).unwrap_err();
            assert!(err.to_string().contains("dtype"));
        });
    }

    #[test]
    fn stack_arrays_preserves_u16_dtype() {
        Python::initialize();
        Python::attach(|py| {
            let (w, h) = (16usize, 16usize);
            let a = Array3::<u16>::from_elem((h, w, 3), 30000u16).into_pyarray(py);
            let b = Array3::<u16>::from_elem((h, w, 3), 31000u16).into_pyarray(py);
            let frames: Vec<Bound<'_, PyAny>> = vec![a.into_any(), b.into_any()];
            let settings = PyStackingSettings {
                alignment_mode: "None".to_owned(),
                auto_cull: false,
                sort_by_sharpness: false,
                correct_brightness: false,
                ..PyStackingSettings::default()
            };

            let result = stack_arrays(py, frames, &settings, "apex".to_owned()).unwrap();
            let arr: &Bound<'_, PyArray3<u16>> = result
                .cast()
                .expect("output dtype should match input (uint16)");
            assert_eq!(arr.readonly().as_array().shape(), [h, w, 3]);
        });
    }
}
