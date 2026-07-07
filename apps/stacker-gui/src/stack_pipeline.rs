//! The out-of-core tiled Stack path, and the in-RAM Stack path's alignment
//! stage.
//!
//! Extracted from `callbacks::stacking::wire`'s `on_request_stack` handler
//! so that file stays closer to the module-size target. See the doc
//! comments on [`run_tiled_stack`] and [`run_inram_alignment`] for the full
//! behavioural contracts — both are a pure relocation of their respective
//! original inline blocks, not a rewrite: every captured variable is passed
//! in explicitly instead of being closed over, and the status-posting
//! macros are reproduced verbatim in each (macros are expanded at compile
//! time, so duplicating them changes nothing about their behaviour).

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use rayon::prelude::*;
use slint::Image;

use stacker_algo::hybrid::retouch::{RetouchHistory, RetouchSession};
use stacker_core::{image::PlanarImage, memory::extract_tile};

use stacker_align::transform::{
    coverage_mask, intersect_coverage, largest_true_rectangle, resolve_common_crop,
};
#[cfg(feature = "akaze")]
use stacker_align::{
    akaze_match::{KeypointMatcher, extract_ref_features},
    ransac::AlignmentEstimator,
};

#[cfg(feature = "nn")]
use crate::nn_bridge::nn_align_planar;
use crate::{
    App,
    image_utils::{load_as_planar, planar_to_rgba_buffer},
    retouch::RetouchState,
    settings::{AlignmentModeSetting, StackingSettings},
    ui_helpers::update_result_list,
};

/// Runs the out-of-core tiled Stack path (`run_settings.tile_size > 0`) on
/// the calling (background) thread.
///
/// Posts status/progress/result updates back to the UI thread exactly like
/// the in-RAM path does.
///
/// When `tile_size > 0` the user has opted into the same out-of-core, tiled
/// `stacker_pipeline::run_pipeline` engine the CLI always uses (per the
/// shared-code-path design — there is no separate GUI reimplementation of
/// tiling). This bypasses the entire in-RAM alignment/fusion path: Auto-Cull
/// and full live per-frame preview are not available here (same limitation
/// the CLI has always had), since the tiled engine never holds a full
/// decoded frame stack, or the fused result, in memory *during fusion*.
/// On success the finished result is reloaded from disk once so retouch
/// (brush + undo/redo) still works for tiled results, self-blend-wired the
/// same way the in-RAM path wires its default (non-"All Three") session.
///
/// Always returns (there is no separate success/failure return value): every
/// outcome — unsupported "Both"/AI selection, cancellation, pipeline error,
/// or success — is reported to the UI via `app_weak_t` before returning.
///
/// # Panics
///
/// Calls `.lock().unwrap()` on the shared `Mutex`-guarded state
/// (`retouch_arc_t`, `results_arc_t`, `viewed_arc_t`); those panic only if
/// another thread already panicked while holding the same lock (mutex
/// poisoning), which does not happen in normal operation.
#[allow(clippy::too_many_arguments)]
// `too_many_lines`: one cohesive "validate selection -> build params ->
// dispatch `run_pipeline` -> report result" sequence for the tiled path,
// mirroring `run_inram_alignment`'s identical single-function shape.
#[allow(clippy::too_many_lines)]
pub fn run_tiled_stack(
    run_all_three: bool,
    algo_str: &str,
    app_weak_t: &slint::Weak<App>,
    ts_ms: u128,
    paths: &[PathBuf],
    run_settings: &StackingSettings,
    retouch_arc_t: &Arc<Mutex<RetouchState>>,
    results_arc_t: &Arc<Mutex<Vec<PathBuf>>>,
    viewed_arc_t: &Arc<Mutex<Option<PathBuf>>>,
    cancel_requested_t: &Arc<AtomicBool>,
    t_total: Instant,
) {
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
    // Like `set_ui!`, but also clears `is_processing` — used at
    // every point this thread's work ends (success, cancel, or
    // an early "nothing to do" return).
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

    if run_all_three {
        finish_ui!(app_weak_t, |app: App| {
            app.set_status(
                "Tiled (out-of-core) mode doesn't support \"All Three\" — pick Apex or Relief in Settings, or set Tile size to 0.".into(),
            );
            app.set_progress(1.0);
            app.set_current_frame_index(-1);
        });
        return;
    }
    let is_ai = algo_str.contains("AI") || algo_str.contains("Neural");
    if is_ai {
        finish_ui!(app_weak_t, |app: App| {
            app.set_status(
                "Tiled (out-of-core) mode doesn't support AI stacking yet — pick Apex or Relief, or set Tile size to 0.".into(),
            );
            app.set_progress(1.0);
            app.set_current_frame_index(-1);
        });
        return;
    }
    let mode_str = if algo_str.contains("Relief") {
        "relief"
    } else if algo_str.contains("Strata") {
        "strata"
    } else {
        "apex"
    };

    let temp_path =
        std::env::temp_dir().join(format!("stacker_result_tiled_{mode_str}_{ts_ms}.png"));
    let params = stacker_pipeline::PipelineParams {
        paths: paths.to_vec(),
        output_file: temp_path.clone(),
        mode: mode_str.to_owned(),
        tile_size: run_settings.tile_size as usize,
        model: None,
        device: None,
        // Tiled mode doesn't support AI fusion yet (see the
        // `is_ai` check above, which already returns before
        // this point for that case) — neural alignment is
        // likewise not offered here; `run_settings.tile_size
        // > 0` forces classical alignment regardless of
        // `alignment_mode` in this code path today.
        align_model: None,
    };

    set_ui!(app_weak_t, |app: App| {
        app.set_status(
            "Starting tiled (out-of-core) stack — Auto-Cull and full live preview are unavailable in this mode…".into(),
        );
        app.set_progress(0.0);
    });

    let app_weak_p = app_weak_t.clone();
    let on_progress = move |event: stacker_pipeline::PipelineProgress| {
        use stacker_pipeline::PipelineProgress;
        let (msg, progress): (String, f32) = match event {
            PipelineProgress::DecodeRaw { current, total } => {
                let frac = current as f32 / total as f32;
                (
                    format!("Tiled: decoding RAW frame {current}/{total}…"),
                    0.01 * frac,
                )
            }
            PipelineProgress::AlignStart { total } => {
                (format!("Tiled: aligning frame 1/{total}…"), 0.02)
            }
            PipelineProgress::AlignFrame { current, total } => {
                let frac = current as f32 / total as f32;
                (
                    format!("Tiled: aligning frame {current}/{total}…"),
                    0.05 + 0.45 * frac,
                )
            }
            PipelineProgress::AlignDone => ("Tiled: alignment complete.".into(), 0.5),
            PipelineProgress::FuseStart { total } => {
                (format!("Tiled: fusing tile 1/{total}…"), 0.52)
            }
            PipelineProgress::FuseTile { current, total } => {
                let frac = current as f32 / total as f32;
                (
                    format!("Tiled: fusing tile {current}/{total}…"),
                    0.5 + 0.45 * frac,
                )
            }
            PipelineProgress::FuseDone => ("Tiled: fusion complete.".into(), 0.95),
            PipelineProgress::Encoding => ("Tiled: encoding output…".into(), 0.97),
        };
        let _ = slint::invoke_from_event_loop({
            let a = app_weak_p.clone();
            move || {
                if let Some(app) = a.upgrade() {
                    app.set_status(msg.into());
                    app.set_progress(progress);
                }
            }
        });
    };

    let result = smol::block_on(stacker_pipeline::run_pipeline(
        &params,
        run_settings,
        on_progress,
    ));

    // `run_pipeline` has no internal cancellation checkpoints
    // (same as the CLI) — a Cancel click during a tiled run
    // cannot interrupt it early. This check discards the
    // result instead, so a cancelled run doesn't silently
    // land in the results list.
    if cancel_requested_t.load(Ordering::SeqCst) {
        finish_ui!(app_weak_t, |app: App| {
            app.set_status("Tiled stacking cancelled.".into());
            app.set_progress(1.0);
            app.set_current_frame_index(-1);
        });
        return;
    }

    match result {
        Ok(()) => {
            // The tiled path never keeps a full-resolution `PlanarImage`
            // around *during fusion*, but the finished result is written
            // out to `temp_path` on success exactly like the in-RAM path's
            // is — so reload it from disk here and wire up a retouch
            // session from it, the same self-blend way the in-RAM path
            // wires one outside "All Three" mode (base and src both point
            // at this same reloaded result; there's no second algorithm
            // pass here to blend against). Without this, retouch silently
            // did nothing for every tiled result: no visual change and no
            // undo/redo, since `apply_and_snapshot` no-ops whenever
            // `session` is `None`.
            let loaded_arc: Option<Arc<PlanarImage<f32>>> = image::open(&temp_path)
                .ok()
                .map(|img| Arc::new(load_as_planar(&img)));
            let displayed_buf = loaded_arc.as_ref().map(|arc| planar_to_rgba_buffer(arc));
            {
                let mut rs = retouch_arc_t.lock().unwrap();
                rs.session = loaded_arc.map(|arc| RetouchSession::new(Arc::clone(&arc), arc));
                rs.history = RetouchHistory::new();
                // Seed the history with the pristine (unpainted) alpha
                // mask so the very first brush stroke has a baseline to
                // undo back to — mirrors the in-RAM path's identical
                // seeding in `callbacks::stacking::wire`'s `do_stack`.
                let baseline = rs.session.as_ref().map(RetouchSession::snapshot_alpha);
                if let Some(snap) = baseline {
                    rs.history.push(snap);
                }
                rs.result_path = Some(temp_path.clone());
            }

            let elapsed_total_s = t_total.elapsed().as_secs();
            let elapsed_str = if elapsed_total_s < 60 {
                format!("{elapsed_total_s}s")
            } else {
                format!("{}m {}s", elapsed_total_s / 60, elapsed_total_s % 60)
            };
            let approx_n_frames = if run_settings.stack_every_nth > 1 {
                (paths.len() as u32).div_ceil(run_settings.stack_every_nth) as i32
            } else {
                paths.len() as i32
            };
            let status_msg = format!(
                "Tiled [{}] stack finished — {elapsed_str}",
                mode_str.to_uppercase()
            );

            let _ = slint::invoke_from_event_loop({
                let a = app_weak_t.clone();
                let rf = Arc::clone(results_arc_t);
                let vf = Arc::clone(viewed_arc_t);
                move || {
                    if let Some(app) = a.upgrade() {
                        let snapshot = {
                            let mut guard = rf.lock().unwrap();
                            guard.push(temp_path.clone());
                            guard.clone()
                        };
                        update_result_list(&app, &snapshot);
                        *vf.lock().unwrap() = Some(temp_path);
                        if let Some(buf) = displayed_buf {
                            app.set_displayed_image(Image::from_rgba8(buf));
                        }
                        app.set_status(status_msg.into());
                        app.set_progress(1.0);
                        app.set_elapsed_text(elapsed_str.into());
                        app.set_frame_count(approx_n_frames);
                        app.set_is_processing(false);
                        app.set_current_frame_index(-1);
                        // Undo/redo button state lives entirely in the
                        // `RetouchWindow` popup now — there is no
                        // `can-undo`/`can-redo` property on `App` to reset
                        // here any more (see `callbacks::retouch_window`).
                        // A freshly (re)loaded tiled result starts with a
                        // fresh, pristine session (seeded above), so the
                        // popup would show "nothing to undo" the next time
                        // it's opened regardless — same as the in-RAM path.
                    }
                }
            });
        }
        Err(e) => {
            finish_ui!(app_weak_t, move |app: App| {
                app.set_status(format!("Tiled stacking failed: {e}").into());
                app.set_progress(1.0);
                app.set_current_frame_index(-1);
            });
        }
    }
}

/// Outcome of [`run_inram_alignment`].
///
/// Either the completed alignment result (the common-coverage crop rect,
/// plus the original pre-crop canvas dimensions — `planar_imgs` itself is
/// mutated in place, warped frame by warped frame, rather than returned),
/// or an indication that the caller's background thread should stop
/// immediately (cancel requested mid-loop). The `Cancelled` case has
/// already posted its own status via `finish_ui!`, matching
/// `run_tiled_stack`'s and `align_cache::AlignPassOutcome`'s convention —
/// the caller must simply `return` without posting its own status.
pub enum AlignStageOutcome {
    Done {
        computed_crop: Option<(usize, usize, usize, usize)>,
        orig_canvas_w: usize,
        orig_canvas_h: usize,
    },
    Cancelled,
}

/// Runs the in-RAM Stack path's alignment stage.
///
/// Extracted verbatim from `callbacks::stacking::wire`'s `on_request_stack`
/// handler so that file stays closer to the module-size target, mirroring
/// `run_tiled_stack`'s established extraction pattern: every captured
/// variable is passed in explicitly instead of closed over, and the two
/// status-posting macros are reproduced verbatim (macros are expanded at
/// compile time, so duplicating them here changes nothing about their
/// behaviour). The only non-mechanical change from the original inline
/// code is that the two mid-loop cancellation checks now return
/// [`AlignStageOutcome::Cancelled`] instead of bare-returning from the
/// enclosing closure — the caller performs that outer `return` itself.
///
/// `planar_imgs` is warped in place (frame 0, the reference, is left
/// untouched); the returned `computed_crop` is the common-coverage crop
/// rect (guard-railed exactly like [`crate::align_cache::run_alignment_pass`]),
/// and `orig_canvas_w`/`orig_canvas_h` are `planar_imgs[0]`'s dimensions
/// captured immediately after alignment, before any crop is applied —
/// used by the caller to optionally restretch a cropped fused result back
/// to the original resolution.
///
/// # Panics
///
/// Will not panic in practice: `planar_imgs` is asserted non-empty by every
/// caller before this is invoked, and all internal indexing
/// (`planar_imgs[0]`, `matrices[i]`, etc.) stays within bounds established
/// earlier in the same function.
#[allow(clippy::too_many_arguments)]
// `too_many_lines`: one cohesive alignment-stage loop (neural + classical
// branches, cancellation checks, coverage-mask accumulation, crop
// resolution) — matches `align_cache::run_alignment_pass`'s identical
// justification for staying a single function.
#[allow(clippy::too_many_lines)]
pub fn run_inram_alignment(
    planar_imgs: &mut [PlanarImage<f32>],
    used_cache: bool,
    run_settings: &StackingSettings,
    #[allow(unused_variables)] alignment_model: &str,
    #[allow(unused_variables)] ai_device: &str,
    index_map: &[usize],
    app_weak_t: &slint::Weak<App>,
    cancel_requested_t: &Arc<AtomicBool>,
) -> AlignStageOutcome {
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

    let t_align = Instant::now();
    let align_mode = run_settings.alignment_mode;
    // Accumulated common-coverage mask, hoisted above the `if` so
    // it survives to the crop-resolution step below regardless of
    // whether this pass actually ran a fresh alignment. On a
    // cache hit (`used_cache`) this stays all-`true` and is
    // unused — `computed_crop` below is only read when
    // `!used_cache`, matching the cache-write guard.
    let mut acc_mask_final: Vec<bool> = vec![true; planar_imgs[0].width * planar_imgs[0].height];
    if !used_cache && align_mode != AlignmentModeSetting::None && planar_imgs.len() > 1 {
        let align_str = align_mode.as_combo_str();
        let msg = format!("Aligning frames ({align_str})…");
        set_ui!(app_weak_t, move |app: App| {
            app.set_status(msg.into());
        });

        let n_planar = planar_imgs.len();
        let (w, h) = (planar_imgs[0].width, planar_imgs[0].height);

        #[cfg(feature = "nn")]
        let is_neural_mode = align_mode == AlignmentModeSetting::Neural;
        #[cfg(not(feature = "nn"))]
        let is_neural_mode = false;

        #[cfg(feature = "nn")]
        if is_neural_mode {
            let mut matrices = vec![nalgebra::Matrix3::identity(); n_planar];
            let align_res = nn_align_planar(planar_imgs, alignment_model, ai_device);
            if let Ok(res_matrices) = align_res {
                matrices = res_matrices;
            } else {
                tracing::warn!("Neural alignment failed: {:?}", align_res.err());
            }

            // Hybrid refinement (§6, default true): the neural
            // matrix feeds the classical dispatch as a seed
            // against a rolling reference, exactly like an AKAZE
            // seed does — sanity-filtered and gated against
            // ending up worse than identity. `false` keeps the
            // pure direct-matrix behaviour for benchmarking the
            // network in isolation.
            let mut rolling_ref = planar_imgs[0].clone();
            let brightness_target = if run_settings.correct_brightness {
                Some(stacker_align::brightness::BrightnessTarget::new(
                    &planar_imgs[0],
                ))
            } else {
                None
            };

            for i in 1..n_planar {
                if cancel_requested_t.load(Ordering::SeqCst) {
                    let status = format!(
                        "Stacking cancelled while aligning frame {i}/{}.",
                        n_planar - 1
                    );
                    finish_ui!(app_weak_t, move |app: App| {
                        app.set_status(status.into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return AlignStageOutcome::Cancelled;
                }

                let matrix = matrices[i];
                let matrix = if run_settings.neural_refine_classically {
                    // `Registration` regardless of `align_mode`
                    // (which is `Neural` here) so this actually
                    // reaches `align_frame`'s classical-
                    // refinement branch instead of its `Neural`
                    // pass-through arm.
                    let (final_matrix, warped) = stacker_align::align_frame(
                        planar_imgs[i].clone(),
                        &rolling_ref,
                        matrix,
                        AlignmentModeSetting::Registration,
                        run_settings.optimizer,
                        true, // always-bounded regardless of caller
                        i,
                        brightness_target.as_ref(),
                    );
                    planar_imgs[i] = warped;
                    rolling_ref = planar_imgs[i].clone();
                    final_matrix
                } else {
                    // Same faithful edge-clamped spline warp the
                    // classical modes use. On a warp failure keep
                    // the unwarped frame and an identity matrix so
                    // the coverage mask matches what is actually
                    // stored — mirroring `align_frame`'s fallback.
                    match stacker_align::transform::warp_image_clamped(&planar_imgs[i], &matrix) {
                        Ok(warped) => {
                            planar_imgs[i] = warped;
                            matrix
                        }
                        Err(err) => {
                            tracing::warn!(
                                frame = i,
                                error = %err,
                                "neural-alignment warp failed; keeping the unwarped frame"
                            );
                            nalgebra::Matrix3::identity()
                        }
                    }
                };

                let frame_mask = coverage_mask(&matrix, w, h);
                intersect_coverage(&mut acc_mask_final, &frame_mask);

                // Frame counts are tiny; f32 precision loss is irrelevant.
                let frac = (i as f32) / (n_planar as f32);
                let prog = 0.15 + 0.35 * frac;
                let row_idx = index_map.get(i).copied().unwrap_or(0) as i32;
                set_ui!(app_weak_t, move |app: App| {
                    app.set_status(format!("Aligning… frame {i}/{}", n_planar - 1).into());
                    app.set_progress(prog);
                    app.set_current_frame_index(row_idx);
                });
            }
        }
        if !is_neural_mode {
            // AKAZE feature mode is only needed to compute the optional
            // seed; gated out entirely without the `akaze` feature.
            #[cfg(feature = "akaze")]
            let current_mode = stacker_align::pipeline::akaze_mode_for_alignment(align_mode);

            // Sequential chain: each frame is registered against the
            // previously aligned frame, warm-started from that frame's
            // solved transform.  This tracks gradual focus-breathing.
            let mut rolling_ref = planar_imgs[0].clone();
            let mut prev_matrix = nalgebra::Matrix3::<f32>::identity();

            // AKAZE features are an optional seed — only extracted when the
            // user enabled seeding (and the `akaze` feature is compiled in).
            #[cfg(feature = "akaze")]
            let akaze_ref = run_settings
                .akaze_seeding
                .then(|| extract_ref_features(&planar_imgs[0]));

            let brightness_target = if run_settings.correct_brightness {
                Some(stacker_align::brightness::BrightnessTarget::new(
                    &planar_imgs[0],
                ))
            } else {
                None
            };

            // Precompute AKAZE coarse-seed candidates for every frame in
            // parallel. Each frame's match+RANSAC only depends on that
            // frame's own pixels and the fixed reference features above —
            // never on the sequential warm-start chain below — so it can
            // all run across every core up front instead of being
            // interleaved one-frame-at-a-time into the sequential loop.
            #[cfg(feature = "akaze")]
            let coarse_hints: Vec<Option<nalgebra::Matrix3<f32>>> = (0..n_planar)
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

            for i in 1..n_planar {
                if cancel_requested_t.load(Ordering::SeqCst) {
                    let status = format!(
                        "Stacking cancelled while aligning frame {i}/{}.",
                        n_planar - 1
                    );
                    finish_ui!(app_weak_t, move |app: App| {
                        app.set_status(status.into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return AlignStageOutcome::Cancelled;
                }

                let cur_planar = planar_imgs[i].clone();
                #[cfg(feature = "akaze")]
                let coarse_hint = coarse_hints[i];
                #[cfg(not(feature = "akaze"))]
                let coarse_hint: Option<nalgebra::Matrix3<f32>> = None;

                // The shared fn sanity-filters the seed internally, so
                // the AKAZE hint no longer needs pre-filtering here;
                // `unwrap_or(prev_matrix)` gives the warm-start fallback.
                let seed = coarse_hint.unwrap_or(prev_matrix);
                let (matrix, warped) = stacker_align::align_frame(
                    cur_planar,
                    &rolling_ref,
                    seed,
                    align_mode,
                    run_settings.optimizer,
                    true, // always-bounded regardless of caller
                    i,
                    brightness_target.as_ref(),
                );
                let scale_x = matrix[(0, 0)];
                let scale_y = matrix[(1, 1)];
                let tx = matrix[(0, 2)];
                let ty = matrix[(1, 2)];
                tracing::info!(frame = i, scale_x, scale_y, tx, ty, "alignment succeeded");
                planar_imgs[i] = warped;

                // Accumulate coverage mask
                let frame_mask = coverage_mask(&matrix, w, h);
                intersect_coverage(&mut acc_mask_final, &frame_mask);

                // Roll the chain forward: next frame aligns to this warped
                // result, warm-started from this frame's solved transform.
                prev_matrix = matrix;
                rolling_ref = planar_imgs[i].clone();

                let preview_planar = match largest_true_rectangle(&acc_mask_final, w, h) {
                    Some((cx, cy, cw, ch)) => extract_tile(&planar_imgs[i], cx, cy, cw, ch),
                    None => planar_imgs[i].clone(),
                };
                let preview = planar_to_rgba_buffer(&preview_planar);

                let step_prog = (i as f32) / (n_planar as f32);
                let prog = (0.15 + 0.35 * step_prog).min(0.50);
                let align_status = format!("Aligning… frame {i}/{}", n_planar - 1);
                let row_idx = index_map.get(i).copied().unwrap_or(0) as i32;
                set_ui!(app_weak_t, move |app: App| {
                    app.set_status(align_status.into());
                    app.set_progress(prog);
                    app.set_step_progress(step_prog);
                    app.set_current_frame_index(row_idx);
                    app.set_displayed_image(Image::from_rgba8(preview));
                });
            }
        }
    }
    let elapsed_align = t_align.elapsed();
    tracing::info!(
        elapsed_ms = elapsed_align.as_millis(),
        "alignment stage complete"
    );

    // Original (pre-crop) canvas dimensions, captured before any
    // common-coverage crop is applied to `planar_imgs` by the caller —
    // used to optionally restretch the fused result back to this
    // resolution when `resize_cropped_to_original` is set (see the
    // `do_stack` closure in `callbacks::stacking::wire`).
    let (orig_canvas_w, orig_canvas_h) = (planar_imgs[0].width, planar_imgs[0].height);

    // Resolve the common-coverage crop rect for this alignment
    // result, guard-rail included (see `resolve_common_crop`:
    // `None` when there's nothing to crop, or when the rectangle
    // covers less than 25% of the canvas — a rogue/misaligned
    // frame guard). Computed regardless of `used_cache` (needed
    // for the cache write below) and regardless of
    // `crop_to_common_area` (the cache/preview crop is wanted
    // whenever available; the setting only gates whether fusion
    // below actually crops using it — see `crop_rect`).
    let computed_crop = if align_mode != AlignmentModeSetting::None && planar_imgs.len() > 1 {
        let (w, h) = (planar_imgs[0].width, planar_imgs[0].height);
        let resolved = resolve_common_crop(&acc_mask_final, w, h);
        if resolved.is_none() && largest_true_rectangle(&acc_mask_final, w, h).is_some() {
            tracing::warn!(
                "common-coverage crop rejected by the rogue-frame guard (covers \
                 < 25% of the canvas); falling back to full canvas"
            );
        }
        resolved
    } else {
        None
    };

    AlignStageOutcome::Done {
        computed_crop,
        orig_canvas_w,
        orig_canvas_h,
    }
}
