use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use slint::Image;

use stacker_core::{image::PlanarImage, memory::extract_tile};

use stacker_align::transform::{
    coverage_mask, intersect_coverage, largest_true_rectangle, resolve_common_crop,
};
#[cfg(feature = "akaze")]
use stacker_align::{
    akaze_match::{KeypointMatcher, extract_ref_features},
    ransac::AlignmentEstimator,
};

use rayon::prelude::*;

use crate::{
    App,
    image_utils::planar_to_rgba_buffer,
    settings::{AlignmentModeSetting, StackingSettings},
};

#[cfg(feature = "nn")]
use crate::nn_bridge::nn_align_planar;

// ── Aligned-frame cache ───────────────────────────────────────────────────────

/// Cached result of the standalone Align pass.
///
/// `frames[i]` is the warped (aligned) `PlanarImage` for the i-th *loaded*
/// frame (frame 0 = the reference, stored as a clone of the original).
/// `crop` is the `(x, y, w, h)` common-coverage rectangle computed by
/// intersecting every frame's coverage mask and resolving the largest
/// all-valid rectangle through [`stacker_align::transform::resolve_common_crop`]'s
/// guard rails.  When `crop` is `None` no crop is needed or safe to apply
/// (e.g. alignment was disabled, only one frame was loaded, the rectangle
/// already covers the whole canvas, or a rogue/misaligned frame collapsed it
/// below 25% of the canvas area).
pub struct AlignedCache {
    pub frames: Vec<PlanarImage<f32>>,
    pub crop: Option<(usize, usize, usize, usize)>,
    /// Fingerprint of the exact inputs and settings that produced `frames` —
    /// see [`compute_align_fingerprint`]. A later Align or Stack run whose
    /// fingerprint matches can reuse `frames` instead of re-aligning.
    pub fingerprint: u64,
}

/// Build the exact ordered list of frame paths that get **aligned**.
///
/// Excludes previously-saved result files, then applies reverse-sort.
/// Deliberately does *not* apply `stack_every_nth` subsampling — alignment
/// doesn't depend on that setting, and every source-file-list UI index
/// (thumbnail clicks, the "Show Aligned" toggle) indexes into this same full
/// list, so subsampling it here would desync those indices from
/// [`AlignedCache::frames`]. The Stack pass applies `stack_every_nth` as a
/// separate selection step *after* pulling from (or populating) the cache
/// built from this list — see `on_request_stack`.
///
/// Used identically by the Align-only pass and the Stack pass so both align
/// the same frame set and can share one [`AlignedCache`].
#[must_use]
pub fn build_frame_path_list(paths: &[PathBuf], settings: &StackingSettings) -> Vec<PathBuf> {
    let mut work: Vec<PathBuf> = paths
        .iter()
        .filter(|p| !p.to_string_lossy().contains("stacker_result_"))
        .cloned()
        .collect();
    if settings.preprocessing.sort_reverse {
        work.reverse();
    }
    work
}

/// Apply the common-area crop and `stack_every_nth` subsampling to an
/// already-aligned frame set.
///
/// Uses the exact order and logic the Stack handler applies before its own
/// "Auto-cull stage" (see `on_request_stack`'s "Resolve the crop actually
/// applied…" and "Stack-every-Nth subsampling" sections).
///
/// Shared by the Stack handler and the Sort/Cull button handlers so both
/// score `optimize_stack` against byte-for-byte the same frame set — the
/// crop must happen before subsampling (subsampling indexes into the
/// already-cropped frames) and both must happen before any sort/cull
/// scoring, exactly as documented on [`crate::sort_cull::run_sort_cull`].
///
/// `align_paths` and `aligned_frames` must be the same length (the full,
/// pre-subsampling aligned set — e.g. an [`AlignedCache`]'s `frames` keyed
/// to `align_paths`). Returns the post-crop, post-subsampling `(paths,
/// frames)` pair.
#[must_use]
pub fn prepare_frames_for_scoring(
    align_paths: &[PathBuf],
    aligned_frames: &[PlanarImage<f32>],
    crop: Option<(usize, usize, usize, usize)>,
    settings: &StackingSettings,
) -> (Vec<PathBuf>, Vec<PlanarImage<f32>>) {
    let crop_rect = if settings.crop_to_common_area {
        crop
    } else {
        None
    };

    let cropped: Vec<PlanarImage<f32>> = if let Some((cx, cy, cw, ch)) = crop_rect {
        aligned_frames
            .iter()
            .map(|frame| extract_tile(frame, cx, cy, cw, ch))
            .collect()
    } else {
        aligned_frames.to_vec()
    };

    if settings.stack_every_nth > 1 {
        let nth = settings.stack_every_nth as usize;
        (0..align_paths.len())
            .step_by(nth)
            .map(|i| (align_paths[i].clone(), cropped[i].clone()))
            .unzip()
    } else {
        (align_paths.to_vec(), cropped)
    }
}

/// Change-detection fingerprint for the alignment stage.
///
/// Combines every input frame's identity (path, on-disk size, and mtime — so
/// a file edited in place is detected even when the path list itself hasn't
/// changed) with every setting that affects the alignment result. If this
/// value is unchanged since an [`AlignedCache`] was built, the cached aligned
/// frames are known-equivalent to what a fresh alignment pass would produce,
/// so re-aligning is redundant work and can be skipped.
#[must_use]
pub fn compute_align_fingerprint(paths_work: &[PathBuf], settings: &StackingSettings) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    paths_work.len().hash(&mut hasher);
    for p in paths_work {
        p.hash(&mut hasher);
        if let Ok(meta) = std::fs::metadata(p) {
            meta.len().hash(&mut hasher);
            if let Ok(modified) = meta.modified()
                && let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                since_epoch.as_nanos().hash(&mut hasher);
            }
        }
    }
    settings.alignment_mode.as_combo_str().hash(&mut hasher);
    settings.akaze_seeding.hash(&mut hasher);
    settings.neural_refine_classically.hash(&mut hasher);
    settings.correct_brightness.hash(&mut hasher);
    let pre = &settings.preprocessing;
    pre.pre_rotation.hash(&mut hasher);
    pre.pre_crop_enabled.hash(&mut hasher);
    pre.pre_crop_spec.hash(&mut hasher);
    pre.pre_resize_percent.hash(&mut hasher);
    pre.ignore_exif_orientation.hash(&mut hasher);
    hasher.finish()
}

// Per-frame alignment lives in `stacker_align::align_frame` — the single shared
// dispatch used by both the CLI and the GUI. AKAZE seed *computation* stays in
// the loops below (call-site specific); the shared fn consumes the resolved
// seed and sanity-filters it internally.

/// Outcome of [`run_alignment_pass`].
///
/// Either a completed [`AlignedCache`] payload, or an indication that the
/// background thread should stop (cancel requested, or nothing could be
/// loaded) — the caller has already posted the appropriate status via
/// `finish_ui!` in either `Err` case and should simply `return` without
/// posting its own "success" status.
pub enum AlignPassOutcome {
    Done {
        aligned_frames: Vec<PlanarImage<f32>>,
        crop: Option<(usize, usize, usize, usize)>,
    },
    StopSilently,
}

/// Run the full per-frame alignment loop against `planar_imgs`.
///
/// `planar_imgs` must already be decoded + preprocessed, one entry per
/// working path in order; live per-frame progress/preview is reported
/// through `app_weak`. Shared by the Align button, the Stack handler, and
/// the [`crate::sort_cull::SortCullCache`]-populating button handlers (Sort,
/// Cull), so every caller aligns a frame set through the exact same code
/// path instead of risking drift from separate hand-written copies.
///
/// `index_map` maps a `planar_imgs` position to its row index in the
/// sidebar's (unfiltered, unreversed) file list, for the live row
/// highlight/auto-scroll. `progress_lo`/`progress_range` let a caller that
/// wants alignment to occupy a sub-span of its own overall progress bar
/// remap the `0.1..=0.99` fraction this function reports internally
/// (matching `on_align_clicked`'s own bar) into `progress_lo..=progress_lo +
/// progress_range`; pass `(0.0, 1.0)` to report the fractions unchanged.
///
/// # Panics
///
/// Will not panic in practice: each `.expect("just pushed")` call
/// immediately follows an unconditional `aligned_frames.push(...)` for that
/// same iteration, so `aligned_frames` is never empty at that point.
#[allow(clippy::too_many_arguments)]
// `too_many_lines`: this is the shared per-frame alignment loop (neural +
// classical branches, each with cancel/progress/preview plumbing) — the
// task's no-refactor mandate means it stays one function rather than being
// split and risking behavioural drift between the two paths.
#[allow(clippy::too_many_lines)]
pub fn run_alignment_pass(
    planar_imgs: &[PlanarImage<f32>],
    index_map: &[usize],
    run_settings: &StackingSettings,
    #[allow(unused_variables)] ai_model: &str,
    #[allow(unused_variables)] ai_device: &str,
    app_weak_t: &slint::Weak<App>,
    cancel_requested_t: &Arc<AtomicBool>,
    progress_lo: f32,
    progress_range: f32,
) -> AlignPassOutcome {
    macro_rules! set_ui {
        ($a:expr, $body:expr) => {
            let _ = slint::invoke_from_event_loop({
                let a = $a.clone();
                move || {
                    if let Some(app) = a.upgrade() {
                        $body(app);
                    }
                }
            });
        };
    }
    macro_rules! finish_ui {
        ($a:expr, $body:expr) => {
            let _ = slint::invoke_from_event_loop({
                let a = $a.clone();
                move || {
                    if let Some(app) = a.upgrade() {
                        app.set_is_processing(false);
                        $body(app);
                    }
                }
            });
        };
    }

    let align_mode = run_settings.alignment_mode;
    let n = planar_imgs.len();

    // Frame 0 is the reference — keep it as-is.
    // The coverage mask for the reference is all-true.
    let (w, h) = (planar_imgs[0].width, planar_imgs[0].height);
    let mut acc_mask: Vec<bool> = vec![true; w * h];
    let mut aligned_frames: Vec<PlanarImage<f32>> = Vec::with_capacity(n);
    aligned_frames.push(planar_imgs[0].clone());

    if align_mode != AlignmentModeSetting::None && n > 1 {
        #[cfg(feature = "nn")]
        let is_neural_mode = align_mode == AlignmentModeSetting::Neural;
        #[cfg(not(feature = "nn"))]
        let is_neural_mode = false;

        #[cfg(feature = "nn")]
        if is_neural_mode {
            let mut matrices = vec![nalgebra::Matrix3::identity(); n];
            let align_res = nn_align_planar(planar_imgs, ai_model, ai_device);
            if let Ok(res_matrices) = align_res {
                matrices = res_matrices;
            } else {
                tracing::warn!("Neural alignment failed: {:?}", align_res.err());
            }

            let mut rolling_ref = planar_imgs[0].clone();
            let brightness_target = if run_settings.correct_brightness {
                Some(stacker_align::brightness::BrightnessTarget::new(
                    &planar_imgs[0],
                ))
            } else {
                None
            };

            for (i, planar_frame) in planar_imgs.iter().enumerate().skip(1) {
                if cancel_requested_t.load(Ordering::SeqCst) {
                    let status = format!("Align cancelled after {}/{} frames.", i - 1, n - 1);
                    finish_ui!(app_weak_t, move |app: App| {
                        app.set_status(status.into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return AlignPassOutcome::StopSilently;
                }

                let matrix = matrices[i];
                let (matrix, warped) = if run_settings.neural_refine_classically {
                    let result = stacker_align::align_frame(
                        planar_frame.clone(),
                        &rolling_ref,
                        matrix,
                        AlignmentModeSetting::Registration,
                        run_settings.optimizer,
                        true,
                        i,
                        brightness_target.as_ref(),
                    );
                    rolling_ref = result.1.clone();
                    result
                } else {
                    match stacker_align::transform::warp_image_clamped(planar_frame, &matrix) {
                        Ok(warped) => (matrix, warped),
                        Err(err) => {
                            tracing::warn!(
                                frame = i,
                                error = %err,
                                "neural-alignment warp failed; keeping the unwarped frame"
                            );
                            (nalgebra::Matrix3::identity(), planar_frame.clone())
                        }
                    }
                };

                let frame_mask = coverage_mask(&matrix, w, h);
                intersect_coverage(&mut acc_mask, &frame_mask);

                aligned_frames.push(warped);

                let last = aligned_frames.last().expect("just pushed");
                let preview_planar = match largest_true_rectangle(&acc_mask, w, h) {
                    Some((cx, cy, cw, ch)) => extract_tile(last, cx, cy, cw, ch),
                    None => last.clone(),
                };
                let preview = crate::image_utils::planar_to_rgba_buffer(&preview_planar);
                let step_prog = (i as f32) / (n as f32);
                let prog =
                    progress_lo + progress_range * (0.1 + (i as f32) / (n as f32 * 1.1)).min(0.99);
                let row_idx = index_map.get(i).copied().unwrap_or(0) as i32;
                set_ui!(app_weak_t, move |app: App| {
                    app.set_displayed_image(slint::Image::from_rgba8(preview));
                    app.set_status(format!("Aligning… frame {i}/{}", n - 1).into());
                    app.set_progress(prog);
                    app.set_step_progress(step_prog);
                    app.set_current_frame_index(row_idx);
                });
            }
        }
        if !is_neural_mode {
            #[cfg(feature = "akaze")]
            let current_mode = stacker_align::pipeline::akaze_mode_for_alignment(align_mode);

            let mut rolling_ref = planar_imgs[0].clone();
            let mut prev_matrix = nalgebra::Matrix3::<f32>::identity();

            #[cfg(feature = "akaze")]
            let akaze_ref = run_settings
                .akaze_seeding
                .then(|| extract_ref_features(&planar_imgs[0]));

            #[cfg(feature = "akaze")]
            let coarse_hints: Vec<Option<nalgebra::Matrix3<f32>>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    if i == 0 {
                        return None;
                    }
                    akaze_ref.as_ref().and_then(|(ref_kps, ref_desc)| {
                        KeypointMatcher::match_target(ref_kps, ref_desc, &planar_imgs[i])
                            .ok()
                            .and_then(|mr| {
                                AlignmentEstimator::compute_matrix(
                                    &mr.matches,
                                    &mr.kps0,
                                    &mr.kps1,
                                    current_mode,
                                )
                                .ok()
                            })
                    })
                })
                .collect();

            let brightness_target = if run_settings.correct_brightness {
                Some(stacker_align::brightness::BrightnessTarget::new(
                    &planar_imgs[0],
                ))
            } else {
                None
            };

            for (i, planar_frame) in planar_imgs.iter().enumerate().skip(1) {
                if cancel_requested_t.load(Ordering::SeqCst) {
                    let status = format!("Align cancelled after {}/{} frames.", i - 1, n - 1);
                    finish_ui!(app_weak_t, move |app: App| {
                        app.set_status(status.into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return AlignPassOutcome::StopSilently;
                }

                #[cfg(feature = "akaze")]
                let coarse_hint = coarse_hints[i];
                #[cfg(not(feature = "akaze"))]
                let coarse_hint: Option<nalgebra::Matrix3<f32>> = None;

                // The shared fn sanity-filters the seed internally, so
                // the AKAZE hint no longer needs pre-filtering here;
                // `unwrap_or(prev_matrix)` gives the warm-start fallback.
                let seed = coarse_hint.unwrap_or(prev_matrix);
                let (matrix, warped) = stacker_align::align_frame(
                    planar_frame.clone(),
                    &rolling_ref,
                    seed,
                    align_mode,
                    run_settings.optimizer,
                    true,
                    i,
                    brightness_target.as_ref(),
                );

                let frame_mask = coverage_mask(&matrix, w, h);
                intersect_coverage(&mut acc_mask, &frame_mask);

                aligned_frames.push(warped);

                let last = aligned_frames.last().expect("just pushed");
                let preview_planar = match largest_true_rectangle(&acc_mask, w, h) {
                    Some((cx, cy, cw, ch)) => extract_tile(last, cx, cy, cw, ch),
                    None => last.clone(),
                };
                let preview = planar_to_rgba_buffer(&preview_planar);
                let step_prog = (i as f32) / (n as f32);
                let prog =
                    progress_lo + progress_range * (0.1 + (i as f32) / (n as f32 * 1.1)).min(0.99);
                let row_idx = index_map.get(i).copied().unwrap_or(0) as i32;
                set_ui!(app_weak_t, move |app: App| {
                    app.set_displayed_image(Image::from_rgba8(preview));
                    app.set_status(format!("Aligning… frame {i}/{}", n - 1).into());
                    app.set_progress(prog);
                    app.set_step_progress(step_prog);
                    app.set_current_frame_index(row_idx);
                });

                prev_matrix = matrix;
                rolling_ref = aligned_frames.last().expect("just pushed").clone();
            }
        }
    } else {
        // No alignment — frames are as-is, no crop needed.
        for frame in planar_imgs.iter().skip(1) {
            aligned_frames.push(frame.clone());
        }
    }

    // Compute common-area crop rect (guard-railed: `None` when there's
    // nothing to crop, or the rectangle covers less than 25% of the canvas
    // — see `resolve_common_crop`).
    let crop = if align_mode != AlignmentModeSetting::None && n > 1 {
        let resolved = resolve_common_crop(&acc_mask, w, h);
        if resolved.is_none() && largest_true_rectangle(&acc_mask, w, h).is_some() {
            tracing::warn!(
                "common-coverage crop rejected by the rogue-frame guard (covers \
                 < 25% of the canvas); falling back to full canvas"
            );
        }
        resolved
    } else {
        None
    };

    AlignPassOutcome::Done {
        aligned_frames,
        crop,
    }
}
