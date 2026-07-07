use std::{
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use rayon::prelude::*;
use slint::{ComponentHandle, Image, SharedString};

use stacker_core::{image::PlanarImage, memory::extract_tile, preprocessing::preprocess_frame};

use crate::{
    App,
    align_cache::{
        AlignPassOutcome, AlignedCache, build_frame_path_list, compute_align_fingerprint,
        prepare_frames_for_scoring, run_alignment_pass,
    },
    callbacks::retouch_window::{RetouchWindowSlot, refresh_if_visible},
    image_utils::{load_as_planar, planar_to_rgba_buffer},
    retouch::RetouchState,
    settings::StackingSettings,
    sort_cull::{
        SortCullCache, SortCullCacheOps, SortCullCacheState, SortCullOpPressed,
        SortCullPromptContext, button_run_flags, compute_sort_cull_fingerprint, run_sort_cull,
    },
    ui_helpers::update_source_list_marked,
};

/// Wires the Align-only pass, the "Show Aligned" re-render toggle, the Sort
/// and Cull buttons, and the Sort/Cull stale-cache confirmation popup.
///
/// # Panics
///
/// The registered callbacks call `.lock().unwrap()` on the various shared
/// `Mutex`-guarded state (`file_paths`, `settings_arc`, `aligned_cache`,
/// etc.); those panic only if another thread already panicked while holding
/// the same lock (mutex poisoning), which does not happen in normal
/// operation.
#[allow(clippy::too_many_arguments)]
// `too_many_lines`: registers the Align/Sort/Cull callbacks (each with its
// own background-thread pipeline) in one place, mirroring `files::wire`'s
// and `stacking::wire`'s identical structure; splitting would only relocate
// code between files, not shorten any one callback's logic.
#[allow(clippy::too_many_lines)]
pub fn wire(
    app: &App,
    file_paths: &Arc<Mutex<Vec<PathBuf>>>,
    settings_arc: &Arc<Mutex<StackingSettings>>,
    aligned_cache: &Arc<Mutex<Option<AlignedCache>>>,
    sort_cull_cache: &Arc<Mutex<Option<SortCullCache>>>,
    sort_cull_prompt_ctx: &Arc<Mutex<Option<SortCullPromptContext>>>,
    current_view_idx: &Arc<Mutex<Option<usize>>>,
    cancel_requested: &Arc<AtomicBool>,
    retouch_state: &Arc<Mutex<RetouchState>>,
    retouch_popup_slot: &RetouchWindowSlot,
) {
    let app_weak = app.as_weak();

    // ── Sort/Cull stale-cache confirmation popup ────────────────────────────
    // Mirrors the Relief preview popup's take-the-context-out-of-the-slot-and-
    // reply pattern exactly (see `SortCullPromptContext`'s doc comment).
    {
        let app_weak_again = app_weak.clone();
        let ctx_arc_again = Arc::clone(sort_cull_prompt_ctx);
        app.on_sort_cull_stale_run_again(move || {
            let value = ctx_arc_again.lock().unwrap().take();
            if let Some(ctx) = value
                && let Some(app) = app_weak_again.upgrade()
            {
                app.set_sort_cull_stale_open(false);
                let _ = ctx.reply_tx.send(true);
            }
        });

        let app_weak_keep = app_weak.clone();
        let ctx_arc_keep = Arc::clone(sort_cull_prompt_ctx);
        app.on_sort_cull_stale_keep_as_is(move || {
            let value = ctx_arc_keep.lock().unwrap().take();
            if let Some(ctx) = value
                && let Some(app) = app_weak_keep.upgrade()
            {
                app.set_sort_cull_stale_open(false);
                let _ = ctx.reply_tx.send(false);
            }
        });
    }

    // ── Show-Aligned toggle: re-render the current frame in place ──────────
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        let aligned_cache_ref = Arc::clone(aligned_cache);
        let view_idx_ref = Arc::clone(current_view_idx);
        let retouch_arc = Arc::clone(retouch_state);
        let retouch_popup_slot = Rc::clone(retouch_popup_slot);
        app.on_refresh_view(move || {
            // Donor selection ("Show Aligned" toggle) changed in the main
            // window — keep the retouch popup's displayed composite (if any
            // session is active and the popup is visible) in sync too; see
            // `refresh_if_visible`'s doc comment.
            refresh_if_visible(&retouch_popup_slot, &retouch_arc);

            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let Some(idx) = *view_idx_ref.lock().unwrap() else {
                return;
            };
            let Some(path) = paths.lock().unwrap().get(idx).cloned() else {
                return;
            };

            // When Show-Aligned is on and we have a cached aligned frame for this
            // index, show the aligned (warped + common-area-cropped) version.
            if app.get_show_aligned() {
                let cache = aligned_cache_ref.lock().unwrap();
                if let Some(ref ac) = *cache
                    && idx < ac.frames.len()
                {
                    let frame = &ac.frames[idx];
                    let cropped = if let Some((cx, cy, cw, ch)) = ac.crop {
                        extract_tile(frame, cx, cy, cw, ch)
                    } else {
                        frame.clone()
                    };
                    app.set_displayed_image(Image::from_rgba8(planar_to_rgba_buffer(&cropped)));
                    return;
                }
            }
            // Otherwise show the original source frame from disk.
            if let Ok(img) = Image::load_from_path(&path) {
                app.set_displayed_image(img);
            }
        });
    }

    // ── Align-only pass ──────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let paths_arc = Arc::clone(file_paths);
        let settings_arc_a = Arc::clone(settings_arc);
        let aligned_cache_a = Arc::clone(aligned_cache);
        let cancel_requested_a = Arc::clone(cancel_requested);

        app.on_align_clicked(move |ai_model, ai_device| {
            let paths = paths_arc.lock().unwrap().clone();
            if paths.is_empty() {
                return;
            }

            #[allow(unused_variables)]
            let ai_model = ai_model.to_string();
            #[allow(unused_variables)]
            let ai_device = ai_device.to_string();

            let run_settings = settings_arc_a.lock().unwrap().clone();
            let app_weak_t = app_weak.clone();
            let aligned_cache_t = Arc::clone(&aligned_cache_a);
            let cancel_requested_t = Arc::clone(&cancel_requested_a);
            cancel_requested_t.store(false, Ordering::SeqCst);
            if let Some(app) = app_weak.upgrade() {
                app.set_is_processing(true);
            }

            std::thread::spawn(move || {
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

                set_ui!(app_weak_t, |app: App| {
                    app.set_status("Align: loading frames…".into());
                    app.set_progress(0.0);
                });

                // Build the working path list — identical normalisation to the
                // Stack handler (exclude prior results, reverse-sort, then
                // Nth-frame subsampling) so the two share one frame set and one
                // alignment cache.
                let paths_work = build_frame_path_list(&paths, &run_settings);

                // Maps a `paths_work` position to its row index in the sidebar's
                // `loaded-files` list (which mirrors the unfiltered, unreversed
                // `paths`), so the per-frame progress loop below can highlight
                // and auto-scroll to the right row while it runs.
                let index_map: Vec<usize> = paths_work
                    .iter()
                    .map(|p| paths.iter().position(|op| op == p).unwrap_or(0))
                    .collect();

                let pre = run_settings.preprocessing.clone();
                let total = paths_work.len();

                // Decode + preprocess all frames. Routed through the shared
                // `stacker_core::io::load_frame` so RAW inputs (`raw`
                // feature builds) are decoded identically to the CLI and the
                // tiled pipeline, instead of this call site's own
                // `image::open` silently failing on a RAW extension.
                let decode_results: Vec<(usize, Result<image::DynamicImage, stacker_core::io::LoadError>)> =
                    paths_work
                        .iter()
                        .enumerate()
                        .map(|(i, path)| (i, stacker_core::io::load_frame(path)))
                        .collect();

                let mut planar_imgs: Vec<PlanarImage<f32>> = Vec::with_capacity(total);
                for (i, result) in decode_results {
                    match result {
                        Ok(img) => {
                            let img = preprocess_frame(img, &pre);
                            planar_imgs.push(load_as_planar(&img));
                        }
                        Err(e) => {
                            tracing::warn!(
                                frame = i,
                                path = %paths_work[i].display(),
                                "align: failed to load frame: {e}"
                            );
                        }
                    }
                }

                if planar_imgs.is_empty() {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Align: no frames could be loaded.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }

                let n = planar_imgs.len();
                let (aligned_frames, crop) = match run_alignment_pass(
                    &planar_imgs,
                    &index_map,
                    &run_settings,
                    &ai_model,
                    &ai_device,
                    &app_weak_t,
                    &cancel_requested_t,
                    0.0,
                    1.0,
                ) {
                    AlignPassOutcome::Done {
                        aligned_frames,
                        crop,
                    } => (aligned_frames, crop),
                    AlignPassOutcome::StopSilently => return,
                };

                // Store in cache, keyed by a fingerprint of the exact inputs
                // and settings that determined this result — a later Align or
                // Stack click with an unchanged fingerprint reuses this
                // instead of re-aligning from scratch.
                let fingerprint = compute_align_fingerprint(&paths_work, &run_settings);
                *aligned_cache_t.lock().unwrap() = Some(AlignedCache {
                    frames: aligned_frames,
                    crop,
                    fingerprint,
                });

                let crop_msg = crop.map_or_else(
                    || "no crop".to_owned(),
                    |(cx, cy, cw, ch)| format!("crop {cw}×{ch} @ ({cx},{cy})"),
                );
                let status = format!(
                    "Align complete: {n} frames aligned, {crop_msg}. Toggle 'Show Aligned' to preview."
                );
                finish_ui!(app_weak_t, move |app: App| {
                    app.set_status(status.into());
                    app.set_progress(1.0);
                    app.set_current_frame_index(-1);
                });
            });
        });
    }

    // ── Sort / Cull buttons ──────────────────────────────────────────────
    // Both run the exact same standalone pipeline: build the working path
    // list, reuse a matching AlignedCache or align from scratch (via the
    // shared `run_alignment_pass` — the same code Align and Stack use), crop
    // + subsample identically to Stack's own pre-cull steps
    // (`prepare_frames_for_scoring`), then call `run_sort_cull` — the same
    // function Stack's own Auto-cull stage calls — so a button press and a
    // subsequent Stack run always agree.
    //
    // Sort and Cull are fully independent commands: neither reads
    // `settings.sort_by_sharpness`/`auto_cull` — pressing Sort always sorts,
    // pressing Cull always culls (at the threshold slider's current value),
    // regardless of how those Settings toggles are set. `button_run_flags`
    // decides the *other* op purely from whether a still-fresh cache already
    // covers it, so re-pressing one button after the other (with nothing
    // else changed) preserves and stays consistent with the other's result
    // instead of clobbering it — see `button_run_flags`'s doc comment.
    for op_pressed in [SortCullOpPressed::Sort, SortCullOpPressed::Cull] {
        let app_weak = app_weak.clone();
        let paths_arc = Arc::clone(file_paths);
        let settings_arc_b = Arc::clone(settings_arc);
        let aligned_cache_b = Arc::clone(aligned_cache);
        let sort_cull_cache_b = Arc::clone(sort_cull_cache);
        let cancel_requested_b = Arc::clone(cancel_requested);

        let button_label = match op_pressed {
            SortCullOpPressed::Sort => "Sort",
            SortCullOpPressed::Cull => "Cull",
        };

        let run_button = move |ai_model: SharedString, ai_device: SharedString| {
            let paths = paths_arc.lock().unwrap().clone();
            if paths.is_empty() {
                return;
            }
            let ai_model = ai_model.to_string();
            let ai_device = ai_device.to_string();

            let run_settings = settings_arc_b.lock().unwrap().clone();
            let app_weak_t = app_weak.clone();
            let aligned_cache_t = Arc::clone(&aligned_cache_b);
            let sort_cull_cache_t = Arc::clone(&sort_cull_cache_b);
            let cancel_requested_t = Arc::clone(&cancel_requested_b);
            cancel_requested_t.store(false, Ordering::SeqCst);
            if let Some(app) = app_weak.upgrade() {
                app.set_is_processing(true);
            }

            std::thread::spawn(move || {
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

                let _ = slint::invoke_from_event_loop({
                    let a = app_weak_t.clone();
                    let label = button_label;
                    move || {
                        if let Some(app) = a.upgrade() {
                            app.set_status(format!("{label}: loading frames…").into());
                            app.set_progress(0.0);
                        }
                    }
                });

                // Same working-list normalisation Align and Stack use.
                let align_paths = build_frame_path_list(&paths, &run_settings);
                let align_total = align_paths.len();
                if align_total == 0 {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Nothing to process.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }
                let index_map: Vec<usize> = align_paths
                    .iter()
                    .map(|p| paths.iter().position(|op| op == p).unwrap_or(0))
                    .collect();

                let align_fingerprint = compute_align_fingerprint(&align_paths, &run_settings);
                let cached_frames: Option<Vec<PlanarImage<f32>>> = {
                    let cache = aligned_cache_t.lock().unwrap();
                    cache
                        .as_ref()
                        .filter(|c| {
                            c.fingerprint == align_fingerprint && c.frames.len() == align_total
                        })
                        .map(|c| c.frames.clone())
                };

                let (aligned_frames, crop) = if let Some(frames) = cached_frames {
                    let cache_crop = aligned_cache_t
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|c| c.crop);
                    (frames, cache_crop)
                } else {
                    let pre = run_settings.preprocessing.clone();
                    let decode_results: Vec<(
                        usize,
                        Result<image::DynamicImage, stacker_core::io::LoadError>,
                    )> = align_paths
                        .par_iter()
                        .enumerate()
                        .map(|(i, path)| (i, stacker_core::io::load_frame(path)))
                        .collect();

                    let mut planar_imgs: Vec<PlanarImage<f32>> = Vec::with_capacity(align_total);
                    for (i, result) in decode_results {
                        match result {
                            Ok(img) => {
                                let img = preprocess_frame(img, &pre);
                                planar_imgs.push(load_as_planar(&img));
                            }
                            Err(e) => {
                                tracing::warn!(
                                    frame = i,
                                    path = %align_paths[i].display(),
                                    "{button_label}: failed to load frame: {e}"
                                );
                            }
                        }
                    }
                    if planar_imgs.is_empty() {
                        finish_ui!(app_weak_t, |app: App| {
                            app.set_status("No frames could be loaded.".into());
                            app.set_progress(1.0);
                            app.set_current_frame_index(-1);
                        });
                        return;
                    }

                    match run_alignment_pass(
                        &planar_imgs,
                        &index_map,
                        &run_settings,
                        &ai_model,
                        &ai_device,
                        &app_weak_t,
                        &cancel_requested_t,
                        0.0,
                        0.7,
                    ) {
                        AlignPassOutcome::Done {
                            aligned_frames,
                            crop,
                        } => {
                            *aligned_cache_t.lock().unwrap() = Some(AlignedCache {
                                frames: aligned_frames.clone(),
                                crop,
                                fingerprint: align_fingerprint,
                            });
                            (aligned_frames, crop)
                        }
                        AlignPassOutcome::StopSilently => return,
                    }
                };

                if cancel_requested_t.load(Ordering::SeqCst) {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status(format!("{button_label} cancelled.").into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }

                let _ = slint::invoke_from_event_loop({
                    let a = app_weak_t.clone();
                    let label = button_label;
                    move || {
                        if let Some(app) = a.upgrade() {
                            app.set_status(format!("{label}: scoring frames…").into());
                            app.set_progress(0.8);
                        }
                    }
                });

                let (scoring_paths, scoring_frames) =
                    prepare_frames_for_scoring(&align_paths, &aligned_frames, crop, &run_settings);
                if scoring_paths.is_empty() {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Nothing left to score after crop/subsampling.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }

                // Independence: never consult `run_settings.sort_by_sharpness`/
                // `auto_cull` here. The pressed op always runs; the *other*
                // op is preserved only if a still-fresh cache already ran it
                // (see `button_run_flags`), so Sort-then-Cull-then-Sort stays
                // internally consistent without either button ever reading
                // the Stack toggles.
                let sort_cull_fingerprint =
                    compute_sort_cull_fingerprint(&align_paths, &run_settings);
                let cache_state = {
                    let cache = sort_cull_cache_t.lock().unwrap();
                    match cache.as_ref() {
                        None => SortCullCacheState::Absent,
                        Some(c) if c.fingerprint == sort_cull_fingerprint => {
                            SortCullCacheState::Fresh(SortCullCacheOps {
                                ran_sort: c.ran_sort,
                                ran_cull: c.ran_cull,
                            })
                        }
                        Some(_) => SortCullCacheState::Stale,
                    }
                };
                let (want_sort, want_cull) = button_run_flags(op_pressed, cache_state);
                let outcome = run_sort_cull(
                    &scoring_frames,
                    want_sort,
                    want_cull,
                    run_settings.auto_cull_threshold_pct,
                );

                let order: Vec<PathBuf> = outcome
                    .order
                    .iter()
                    .map(|&idx| scoring_paths[idx].clone())
                    .collect();
                let culled: Vec<PathBuf> = outcome
                    .culled
                    .iter()
                    .map(|&idx| scoring_paths[idx].clone())
                    .collect();

                // Store the UNION of what actually ran this press (a stale
                // or absent cache is simply replaced by this single-op
                // result, since `want_sort`/`want_cull` above already fold
                // in whatever the previous fresh cache covered).
                *sort_cull_cache_t.lock().unwrap() = Some(SortCullCache {
                    fingerprint: sort_cull_fingerprint,
                    order: order.clone(),
                    culled: culled.clone(),
                    ran_sort: want_sort,
                    ran_cull: want_cull,
                });

                let n_kept = order.len();
                let n_culled = culled.len();
                let status = if want_cull {
                    format!("{button_label} complete: {n_kept} kept, {n_culled} culled.")
                } else {
                    format!("{button_label} complete: {n_kept} frames reordered by sharpness.")
                };

                finish_ui!(app_weak_t, move |app: App| {
                    // The remaining source paths not touched by this pass
                    // (e.g. files excluded from `align_paths` such as prior
                    // stack results) are appended after the ordered/kept set
                    // so no file silently disappears from the list.
                    let mut new_list = order.clone();
                    for p in &paths {
                        if !new_list.contains(p) && !culled.contains(p) {
                            new_list.push(p.clone());
                        }
                    }
                    for p in &culled {
                        if !new_list.contains(p) {
                            new_list.push(p.clone());
                        }
                    }
                    update_source_list_marked(&app, &new_list, &culled);
                    app.set_status(status.into());
                    app.set_progress(1.0);
                    app.set_current_frame_index(-1);
                });
            });
        };

        match op_pressed {
            SortCullOpPressed::Sort => app.on_request_sort(run_button),
            SortCullOpPressed::Cull => app.on_request_cull(run_button),
        }
    }
}
