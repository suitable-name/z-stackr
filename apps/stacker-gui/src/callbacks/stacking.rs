use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use rayon::prelude::*;
use slint::{ComponentHandle, Image};

use stacker_algo::hybrid::retouch::{RetouchHistory, RetouchSession};
use stacker_core::{image::PlanarImage, memory::extract_tile, preprocessing::preprocess_frame};

use stacker_align::transform::resize_planar_clamped;

use crate::{
    App,
    align_cache::{AlignedCache, build_frame_path_list, compute_align_fingerprint},
    image_utils::{load_as_planar, planar_to_rgb_image, planar_to_rgba_buffer},
    nn_bridge::nn_fuse_planar,
    retouch::RetouchState,
    settings::StackingSettings,
    sort_cull::{
        SortCullCache, SortCullCacheOps, SortCullCacheState, SortCullDecision,
        SortCullPromptContext, SortCullWants, compute_sort_cull_fingerprint, decide_sort_cull,
        run_sort_cull,
    },
    stacking::{
        ReliefPreviewContext, fuse_apex, fuse_relief, fuse_strata, generate_relief_preview,
    },
    ui_helpers::{update_result_list, update_source_list, update_source_list_marked},
};

/// Wires the Relief-preview popup callbacks, the Cancel button, and the
/// main Stack pipeline.
///
/// `on_request_stack` covers alignment, auto-cull, and fusion
/// (Apex/Relief/Strata/AI), including the out-of-core tiled path.
///
/// # Panics
///
/// The registered callbacks call `.lock().unwrap()` on the various shared
/// `Mutex`-guarded state (`file_paths`, `settings_arc`, `retouch_state`,
/// etc.); those panic only if another thread already panicked while
/// holding the same lock (mutex poisoning), which does not happen in
/// normal operation.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn wire(
    app: &App,
    file_paths: &Arc<Mutex<Vec<PathBuf>>>,
    result_paths: &Arc<Mutex<Vec<PathBuf>>>,
    currently_viewed_path: &Arc<Mutex<Option<PathBuf>>>,
    retouch_state: &Arc<Mutex<RetouchState>>,
    settings_arc: &Arc<Mutex<StackingSettings>>,
    relief_preview_ctx: &Arc<Mutex<Option<ReliefPreviewContext>>>,
    aligned_cache: &Arc<Mutex<Option<AlignedCache>>>,
    sort_cull_cache: &Arc<Mutex<Option<SortCullCache>>>,
    sort_cull_prompt_ctx: &Arc<Mutex<Option<SortCullPromptContext>>>,
    cancel_requested: &Arc<AtomicBool>,
) {
    let app_weak = app.as_weak();

    // ── Relief Preview Callbacks ───────────────────────────────────────────
    {
        let app_weak_apply = app_weak.clone();
        let ctx_arc_apply = Arc::clone(relief_preview_ctx);
        app.on_relief_preview_apply(move || {
            let value = ctx_arc_apply.lock().unwrap().take();
            if let Some(ctx) = value
                && let Some(app) = app_weak_apply.upgrade()
            {
                app.set_relief_preview_open(false);
                let _ = ctx.reply_tx.send(Some(app.get_set_relief_contrast_pct()));
            }
        });

        let app_weak_c = app_weak.clone();
        let ctx_arc_c = Arc::clone(relief_preview_ctx);
        app.on_relief_preview_cancel(move || {
            let value = ctx_arc_c.lock().unwrap().take();
            if let Some(ctx) = value
                && let Some(app) = app_weak_c.upgrade()
            {
                app.set_relief_preview_open(false);
                let _ = ctx.reply_tx.send(None);
            }
        });

        let app_weak_a = app_weak.clone();
        let ctx_arc_a = Arc::clone(relief_preview_ctx);
        app.on_relief_preview_auto_detect(move || {
            let value = ctx_arc_a.lock().unwrap().clone();
            if let Some(ctx) = value {
                let threshold =
                    stacker_algo::relief::fuse::auto_contrast_threshold(&ctx.state.max_sml);
                if let Some(app) = app_weak_a.upgrade() {
                    app.set_set_relief_contrast_pct(threshold);
                    let img = generate_relief_preview(&ctx.state, threshold);
                    app.set_relief_preview_image(slint::Image::from_rgba8(img));
                }
            }
        });

        let app_weak_cc = app_weak.clone();
        let ctx_arc_cc = Arc::clone(relief_preview_ctx);
        app.on_relief_preview_contrast_changed(move || {
            let value = ctx_arc_cc.lock().unwrap().clone();
            if let Some(ctx) = value
                && let Some(app) = app_weak_cc.upgrade()
            {
                let pct = app.get_set_relief_contrast_pct();
                let img = generate_relief_preview(&ctx.state, pct);
                app.set_relief_preview_image(slint::Image::from_rgba8(img));
            }
        });
    }

    // ── Cancel the in-flight Align/Stack run ───────────────────────────────
    {
        let cancel_requested = Arc::clone(cancel_requested);
        app.on_cancel_processing(move || {
            cancel_requested.store(true, Ordering::SeqCst);
        });
    }

    // ── Stacking ─────────────────────────────────────────────────────────
    {
        let paths_arc = Arc::clone(file_paths);
        let results_arc = Arc::clone(result_paths);
        let viewed_arc = Arc::clone(currently_viewed_path);
        let retouch_arc = Arc::clone(retouch_state);
        let settings_arc = Arc::clone(settings_arc);
        let relief_preview_ctx_stack = Arc::clone(relief_preview_ctx);
        let aligned_cache_stack = Arc::clone(aligned_cache);
        let sort_cull_cache_stack = Arc::clone(sort_cull_cache);
        let sort_cull_prompt_ctx_stack = Arc::clone(sort_cull_prompt_ctx);
        let cancel_requested_stack = Arc::clone(cancel_requested);

        app.on_request_stack(move |algo, ai_model, ai_device, alignment_model| {
            let paths = paths_arc.lock().unwrap().clone();
            if paths.is_empty() { return; }

            // Snapshot settings at the time the button is pressed.
            let run_settings = settings_arc.lock().unwrap().clone();

            // Engage/disengage the shared runtime GPU switch once per Stack
            // run, covering both the in-RAM path below and the tiled
            // `stacker_pipeline::run_pipeline` path (which also calls this
            // itself, redundantly but harmlessly, since it's the single
            // shared entry point the CLI/Python use too). No-op in a
            // default, non-`gpu` build.
            #[cfg(feature = "gpu")]
            stacker_core::gpu::set_enabled(run_settings.use_gpu);

            let algo_str = algo.to_string();
            let ai_model = ai_model.to_string();
            let ai_device = ai_device.to_string();
            // The fusion model (`ai_model`, filtered to `ModelEntry::is_fusion`)
            // and the alignment model (filtered to `ModelEntry::is_alignment`)
            // are never interchangeable checkpoints, so the neural-alignment
            // pre-pass below must use its own picker value rather than
            // reusing `ai_model` (see `alignment-model-value` in app.slint).
            // Only read inside `#[cfg(feature = "nn")]` blocks below, hence
            // the `unused_variables` allow for non-`nn` builds (same pattern
            // as `on_align_clicked`'s `ai_model`/`ai_device`).
            #[allow(unused_variables)]
            let alignment_model = alignment_model.to_string();
            let run_all_three = algo_str.contains("All Three");

            let app_weak_t    = app_weak.clone();
            let paths_arc_t   = Arc::clone(&paths_arc);
            let results_arc_t = Arc::clone(&results_arc);
            let viewed_arc_t  = Arc::clone(&viewed_arc);
            let retouch_arc_t = Arc::clone(&retouch_arc);
            let relief_preview_ctx_t = Arc::clone(&relief_preview_ctx_stack);
            let aligned_cache_t2 = Arc::clone(&aligned_cache_stack);
            let sort_cull_cache_t = Arc::clone(&sort_cull_cache_stack);
            let sort_cull_prompt_ctx_t = Arc::clone(&sort_cull_prompt_ctx_stack);
            let cancel_requested_t = Arc::clone(&cancel_requested_stack);
            cancel_requested_t.store(false, Ordering::SeqCst);
            if let Some(app) = app_weak.upgrade() {
                app.set_is_processing(true);
            }

            std::thread::spawn(move || {
                macro_rules! set_ui {
                    ($a:expr, $body:expr) => {
                        let _ = slint::invoke_from_event_loop({ let a = $a.clone(); move || { if let Some(app) = a.upgrade() { $body(app); } } });
                    };
                }
                // Like `set_ui!`, but also clears `is_processing` — used at
                // every point this thread's work ends (success, cancel, or
                // an early "nothing to do" return).
                macro_rules! finish_ui {
                    ($a:expr, $body:expr) => {
                        let _ = slint::invoke_from_event_loop({ let a = $a.clone(); move || { if let Some(app) = a.upgrade() { app.set_is_processing(false); $body(app); } } });
                    };
                }

                let ts_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                let log_path = std::env::current_dir()
                    .unwrap_or_else(|_| std::path::PathBuf::from("."))
                    .join("stacker.log");
                let log_path_str = log_path.display().to_string();

                let t_total = Instant::now();

                // ── Out-of-core tiled path ───────────────────────────────────────
                // See `stack_pipeline::run_tiled_stack`'s doc comment for the full
                // behavioural contract; extracted verbatim so this file stays
                // under the module-size target.
                if run_settings.tile_size > 0 {
                    crate::stack_pipeline::run_tiled_stack(
                        run_all_three,
                        &algo_str,
                        &app_weak_t,
                        ts_ms,
                        &paths,
                        &run_settings,
                        &retouch_arc_t,
                        &results_arc_t,
                        &viewed_arc_t,
                        &cancel_requested_t,
                        t_total,
                    );
                    return;
                }

                set_ui!(app_weak_t, |app: App| {
                    app.set_status("Loading frames...".into());
                    app.set_progress(0.0);
                });

                // The full retained + reverse-sorted set that gets *aligned*.
                // Deliberately not yet subsampled by `stack_every_nth` — see
                // `build_frame_path_list`'s doc comment for why that happens
                // as a separate step after alignment instead.
                let align_paths = build_frame_path_list(&paths, &run_settings);
                let align_total = align_paths.len();
                if align_total == 0 {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("No frames to stack.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }

                // Maps an `align_paths` position to its row index in the
                // sidebar's `loaded-files` list (which mirrors the unfiltered,
                // unreversed `paths`), so the per-frame alignment loop below
                // can highlight and auto-scroll to the right row while it runs.
                let index_map: Vec<usize> = align_paths
                    .iter()
                    .map(|p| paths.iter().position(|op| op == p).unwrap_or(0))
                    .collect();

                // Fingerprint of the exact inputs + alignment-affecting
                // settings. A match against a cached `AlignedCache` (from a
                // prior Align-only pass, or a prior Stack run with the same
                // frames/settings) means the cached aligned frames are
                // known-equivalent to what a fresh pass would produce, so the
                // decode + alignment stages below are skipped entirely rather
                // than redone for no reason.
                let fingerprint = compute_align_fingerprint(&align_paths, &run_settings);
                let cached_frames: Option<Vec<PlanarImage<f32>>> = {
                    let cache = aligned_cache_t2.lock().unwrap();
                    cache
                        .as_ref()
                        .filter(|c| c.fingerprint == fingerprint && c.frames.len() == align_total)
                        .map(|c| c.frames.clone())
                };
                let used_cache = cached_frames.is_some();

                let t_load = Instant::now();

                // `loaded_img0` (the reference frame's un-warped RGBA) is the
                // only decoded-RGBA frame actually consumed later — it's the
                // background image for the Relief contrast-preview popup. Every
                // other frame's RGBA was previously decoded here purely to sit
                // unused after alignment, so on a cache hit there is nothing to
                // decode but frame 0.
                let (mut planar_imgs, loaded_img0): (Vec<PlanarImage<f32>>, image::RgbaImage) =
                    if let Some(frames) = cached_frames {
                        set_ui!(app_weak_t, |app: App| {
                            app.set_status("Using cached alignment (unchanged since last Align/Stack)…".into());
                            app.set_progress(0.33);
                        });
                        let img0 = stacker_core::io::load_frame(&align_paths[0]).map_or_else(
                            |e| {
                                tracing::warn!(path = %align_paths[0].display(), "failed to reload reference frame for preview: {e}");
                                image::RgbaImage::new(1, 1)
                            },
                            |d| preprocess_frame(d, &run_settings.preprocessing).to_rgba8(),
                        );
                        (frames, img0)
                    } else {
                        // Decode in parallel, then apply preprocessing sequentially
                        // (preprocessing uses image crate which is not Send-safe for
                        // all pixel types, but par_iter is fine since each frame is
                        // independent).
                        //
                        // NOTE: ignore_exif_orientation — the `image` crate's `open`
                        // does NOT auto-apply EXIF orientation (it reads raw pixels
                        // only), so the existing pipeline already ignores EXIF.
                        // `true` (ignore) = current/default behaviour = no-op.
                        // `false` (apply) would require `ImageReader::with_guessed_format`
                        // + orientation extraction, which is not exposed as a simple
                        // API in this version of `image`.  The field is stored and
                        // round-trips through config but has no effect on the loaded
                        // pixels in either case until a new `image` version exposes it.
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
                        let mut loaded_img0: Option<image::RgbaImage> = None;
                        let mut first_dims: Option<(u32, u32)> = None;

                        for (i, result) in decode_results {
                            match result {
                                Ok(img) => {
                                    // Apply pre-processing (rotation → crop → resize).
                                    // All transforms are no-ops at default settings, so
                                    // the pipeline output is unchanged when nothing is
                                    // configured.
                                    let img = preprocess_frame(img, &pre);
                                    let (iw, ih) = image::GenericImageView::dimensions(&img);
                                    if first_dims.is_none() {
                                        first_dims = Some((iw, ih));
                                        tracing::info!(
                                            frame = i, width = iw, height = ih,
                                            n_frames = align_total, "first frame dimensions"
                                        );
                                    }
                                    if i == 0 {
                                        loaded_img0 = Some(img.to_rgba8());
                                    }
                                    planar_imgs.push(load_as_planar(&img));
                                }
                                Err(e) => {
                                    tracing::warn!(frame = i, path = %align_paths[i].display(), "failed to load frame: {e}");
                                }
                            }
                        }

                        // Total-progress model (monotonic across the whole run):
                        //   load 0.00–0.15 · align 0.15–0.50 · fuse 0.50–0.95 · save 0.95–1.0
                        set_ui!(app_weak_t, |app: App| { app.set_progress(0.15); });
                        let elapsed_load = t_load.elapsed();
                        tracing::info!(
                            n_loaded = planar_imgs.len(),
                            elapsed_ms = elapsed_load.as_millis(), "load stage complete"
                        );

                        let Some(img0) = loaded_img0 else {
                            finish_ui!(app_weak_t, |app: App| {
                                app.set_status("Stack: reference frame failed to load.".into());
                                app.set_progress(1.0);
                                app.set_current_frame_index(-1);
                            });
                            return;
                        };
                        (planar_imgs, img0)
                    };

                if planar_imgs.is_empty() {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Stack: no frames could be loaded.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }

                // ── Alignment stage ─────────────────────────────────────────
                // Extracted to `stack_pipeline::run_inram_alignment` — see its
                // doc comment for the full behavioural contract; extracted
                // verbatim (plus one mechanical `return` → `Cancelled` variant
                // translation for the two mid-loop cancel checks) so this file
                // stays closer to the module-size target.
                let align_outcome = crate::stack_pipeline::run_inram_alignment(
                    &mut planar_imgs,
                    used_cache,
                    &run_settings,
                    &alignment_model,
                    &ai_device,
                    &index_map,
                    &app_weak_t,
                    &cancel_requested_t,
                );
                let (computed_crop, orig_canvas_w, orig_canvas_h) = match align_outcome {
                    crate::stack_pipeline::AlignStageOutcome::Done {
                        computed_crop,
                        orig_canvas_w,
                        orig_canvas_h,
                    } => (computed_crop, orig_canvas_w, orig_canvas_h),
                    crate::stack_pipeline::AlignStageOutcome::Cancelled => return,
                };

                // Publish this alignment result to the shared cache so a later
                // Align-only pass or Stack run with the same fingerprint can
                // reuse it. A cache hit above already holds the correct result
                // (and its own crop, if any) — don't overwrite it with a
                // stale entry.
                if !used_cache {
                    *aligned_cache_t2.lock().unwrap() = Some(AlignedCache {
                        frames: planar_imgs.clone(),
                        crop: computed_crop,
                        fingerprint,
                    });
                }

                // ── Resolve the crop actually applied to this run's frames
                // before fusion, gated by `crop_to_common_area` ──
                // On a cache hit, reuse the cache's own crop (computed either
                // here on a prior run, or by the Align-only handler) instead
                // of recomputing from a mask this pass never built.
                let crop_rect: Option<(usize, usize, usize, usize)> = if run_settings.crop_to_common_area {
                    if used_cache {
                        aligned_cache_t2.lock().unwrap().as_ref().and_then(|c| c.crop)
                    } else {
                        computed_crop
                    }
                } else {
                    None
                };

                // Crop every aligned frame (and the Relief-preview backdrop
                // image) IMMEDIATELY, before stack-every-nth subsampling,
                // auto-cull scoring, and fusion — so none of those
                // downstream stages ever see the smeared edge-replication
                // band from focus breathing.
                let loaded_img0 = if let Some((cx, cy, cw, ch)) = crop_rect {
                    for frame in &mut planar_imgs {
                        *frame = extract_tile(frame, cx, cy, cw, ch);
                    }
                    image::imageops::crop_imm(&loaded_img0, cx as u32, cy as u32, cw as u32, ch as u32)
                        .to_image()
                } else {
                    loaded_img0
                };

                // Computed here (before `align_paths` is consumed by the
                // subsampling step below) so the Sort/Cull fingerprint is
                // taken against the same full, pre-subsampling aligned set
                // `compute_align_fingerprint` uses — matching the Sort/Cull
                // button handlers' own convention.
                let sort_cull_fingerprint = compute_sort_cull_fingerprint(&align_paths, &run_settings);

                // ── Stack-every-Nth subsampling ─────────────────────────────
                // Applied *after* alignment (and after the result above is
                // cached) rather than before it: alignment doesn't depend on
                // this setting, so aligning the full set once and selecting a
                // subset afterward means the cache stays valid — and gets
                // reused — even when this value changes between Stack runs.
                let (paths_work, planar_imgs): (Vec<PathBuf>, Vec<PlanarImage<f32>>) =
                    if run_settings.stack_every_nth > 1 {
                        let nth = run_settings.stack_every_nth as usize;
                        (0..align_paths.len())
                            .step_by(nth)
                            .map(|i| (align_paths[i].clone(), planar_imgs[i].clone()))
                            .unzip()
                    } else {
                        (align_paths, planar_imgs)
                    };
                let total = paths_work.len();
                if total == 0 { return; }

                if cancel_requested_t.load(Ordering::SeqCst) {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Stacking cancelled after alignment.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }

                // ── Auto-cull stage ─────────────────────────────────────────
                // Sort/Cull fingerprint is computed against `align_paths`
                // (the full, pre-subsampling aligned set), matching
                // `compute_align_fingerprint`'s own convention — the same
                // value the Sort/Cull button handlers compute.
                let t_cull = Instant::now();
                let wants = SortCullWants {
                    sort: run_settings.sort_by_sharpness,
                    cull: run_settings.auto_cull,
                };
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
                let mut decision = decide_sort_cull(cache_state, wants);
                // Set only by the "Keep as is" popup reply: forces the
                // `Compute` branch below into a pure identity/no-cull pass
                // (proceed with the current list order/selection as-is)
                // instead of running Sort/Cull with this run's real toggles.
                let mut keep_as_is = false;

                // On a stale cache, pause and ask the user — mirrors
                // `ReliefPreviewContext`'s park-on-`reply_rx.recv()` pattern
                // exactly (see `SortCullPromptContext`'s doc comment).
                if decision == SortCullDecision::Ask {
                    let (tx, rx) = std::sync::mpsc::channel::<bool>();
                    *sort_cull_prompt_ctx_t.lock().unwrap() = Some(SortCullPromptContext { reply_tx: tx });
                    let _ = slint::invoke_from_event_loop({
                        let a = app_weak_t.clone();
                        move || {
                            if let Some(app) = a.upgrade() {
                                app.set_sort_cull_stale_reason(
                                    "source files or relevant settings changed".into(),
                                );
                                app.set_sort_cull_stale_open(true);
                            }
                        }
                    });
                    if rx.recv() == Ok(true) {
                        decision = SortCullDecision::Compute;
                    } else {
                        // "Keep as is" (or the reply channel was dropped,
                        // treated the same way): proceed with the current
                        // list order/selection unchanged, and discard the
                        // stale cache entirely (it must not be applied
                        // later against a still-mismatching fingerprint).
                        *sort_cull_cache_t.lock().unwrap() = None;
                        decision = SortCullDecision::Compute;
                        keep_as_is = true;
                    }
                    if cancel_requested_t.load(Ordering::SeqCst) {
                        finish_ui!(app_weak_t, |app: App| {
                            app.set_status("Stacking cancelled.".into());
                            app.set_progress(1.0);
                            app.set_current_frame_index(-1);
                        });
                        return;
                    }
                }

                let effective_wants = if keep_as_is {
                    SortCullWants { sort: false, cull: false }
                } else {
                    wants
                };

                let order: Vec<usize> = match decision {
                    SortCullDecision::Skip => {
                        let cache = sort_cull_cache_t.lock().unwrap();
                        let cached = cache.as_ref().expect("Skip implies a fresh cache");
                        let idx_order: Vec<usize> = cached
                            .order
                            .iter()
                            .filter_map(|p| paths_work.iter().position(|w| w == p))
                            .collect();
                        drop(cache);
                        set_ui!(app_weak_t, |app: App| {
                            app.set_status("Using cached Sort/Cull result (unchanged since last run)…".into());
                        });
                        idx_order
                    }
                    SortCullDecision::Compute => {
                        if effective_wants.sort || effective_wants.cull {
                            set_ui!(app_weak_t, |app: App| { app.set_status("Optimising & culling…".into()); });
                        }
                        let outcome = run_sort_cull(
                            &planar_imgs,
                            effective_wants.sort,
                            effective_wants.cull,
                            run_settings.auto_cull_threshold_pct,
                        );
                        // Refresh the cache with this run's own result so a
                        // later Stack press (or Sort/Cull button press) with
                        // an unchanged fingerprint can reuse it too.
                        let order_paths: Vec<PathBuf> =
                            outcome.order.iter().map(|&i| paths_work[i].clone()).collect();
                        let culled_paths: Vec<PathBuf> =
                            outcome.culled.iter().map(|&i| paths_work[i].clone()).collect();
                        *sort_cull_cache_t.lock().unwrap() = Some(SortCullCache {
                            fingerprint: sort_cull_fingerprint,
                            order: order_paths,
                            culled: culled_paths,
                            ran_sort: effective_wants.sort,
                            ran_cull: effective_wants.cull,
                        });
                        outcome.order
                    }
                    SortCullDecision::Ask => unreachable!("Ask is resolved to Compute above"),
                };

                let kept_paths: Vec<PathBuf> = order.iter().map(|&idx| paths_work[idx].clone()).collect();
                let culled_paths_for_ui: Vec<PathBuf> = (0..total)
                    .filter(|i| !order.contains(i))
                    .map(|i| paths_work[i].clone())
                    .collect();
                let _ = slint::invoke_from_event_loop({
                    let a = app_weak_t.clone();
                    let pt = Arc::clone(&paths_arc_t);
                    move || {
                        if let Some(app) = a.upgrade() {
                            let snapshot = {
                                let mut p = pt.lock().unwrap();
                                p.clone_from(&kept_paths);
                                p.clone()
                            };
                            if culled_paths_for_ui.is_empty() {
                                update_source_list(&app, &snapshot);
                            } else {
                                update_source_list_marked(&app, &snapshot, &culled_paths_for_ui);
                            }
                        }
                    }
                });
                let elapsed_cull = t_cull.elapsed();
                tracing::info!(elapsed_ms = elapsed_cull.as_millis(), "cull stage complete");

                if order.is_empty() {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Stack: no frames left after culling.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }
                let stack_total  = order.len();
                let culled_count = total - stack_total;

                let ordered_planar: Vec<PlanarImage<f32>> = {
                    let is_identity = order.len() == planar_imgs.len()
                        && order.iter().enumerate().all(|(pos, &idx)| pos == idx);
                    if is_identity {
                        planar_imgs
                    } else {
                        order.iter().map(|&idx| planar_imgs[idx].clone()).collect()
                    }
                };

                // ── Inner closure: fuse one pass ──────────────────────────
                // Returns the fused result on success, `None` on AI failure or
                // a Relief-preview cancel — the dispatch below uses `None` to
                // stop instead of running further passes or overwriting the
                // failure/cancel status with "Finished!", and uses `Some` in
                // "Both" mode to wire the two passes' results into the
                // retouch brush (see below).
                let do_stack = |prefix: &str| -> Option<Arc<PlanarImage<f32>>> {
                    let msg = format!("Fusing [{prefix}] {stack_total} frames…");
                    let _ = slint::invoke_from_event_loop({
                        let a = app_weak_t.clone();
                        move || { if let Some(app) = a.upgrade() { app.set_status(msg.into()); } }
                    });

                    let t_fuse = Instant::now();
                    let fused = if prefix == "AI" {
                        // Live preview: update the canvas as each frame is folded
                        // into the running composite (watch it sharpen).
                        let app_weak_p = app_weak_t.clone();
                        let on_progress = move |img: &stacker_core::image::PlanarImage<f32>| {
                            let buf = planar_to_rgba_buffer(img);
                            let _ = slint::invoke_from_event_loop({
                                let a = app_weak_p.clone();
                                move || {
                                    if let Some(app) = a.upgrade() {
                                        app.set_displayed_image(Image::from_rgba8(buf));
                                    }
                                }
                            });
                        };
                        match nn_fuse_planar(&ordered_planar, &ai_model, &ai_device, on_progress) {
                            Ok(img) => img,
                            Err(msg) => {
                                let _ = slint::invoke_from_event_loop({
                                    let a = app_weak_t.clone();
                                    let m = format!("AI stacking failed: {msg}");
                                    move || {
                                        if let Some(app) = a.upgrade() {
                                            app.set_status(m.into());
                                            app.set_progress(1.0);
                                        }
                                    }
                                });
                                return None;
                            }
                        }
                    } else if prefix == "Apex" {
                        // Live progress: advance the status/progress bar every
                        // frame, and refresh the displayed image on throttled
                        // checkpoints so the result visibly sharpens instead of
                        // the GUI appearing stuck until the whole stack is fused.
                        let app_weak_p = app_weak_t.clone();
                        let on_progress = move |completed: usize,
                                                 total: usize,
                                                 preview: Option<PlanarImage<f32>>| {
                            let msg = format!("Fusing [Apex] frame {completed}/{total}…");
                            let frac = completed as f32 / total as f32;
                            // Fusion owns the 0.50..0.95 span of the total bar
                            // (alignment ends at 0.50 — keep this monotonic).
                            let progress = 0.50 + 0.45 * frac;
                            let _ = slint::invoke_from_event_loop({
                                let a = app_weak_p.clone();
                                move || {
                                    if let Some(app) = a.upgrade() {
                                        app.set_status(msg.into());
                                        app.set_progress(progress);
                                        app.set_step_progress(frac);
                                        if let Some(img) = preview {
                                            app.set_displayed_image(Image::from_rgba8(
                                                planar_to_rgba_buffer(&img),
                                            ));
                                        }
                                    }
                                }
                            });
                        };
                        fuse_apex(&ordered_planar, &run_settings, on_progress)
                    } else if prefix == "Strata" {
                        // Live progress: mirrors the Apex branch above, but
                        // Strata's ticks span two internal passes (saliency,
                        // then accumulation) rather than one frame-at-a-time
                        // accumulator, so there is no per-tick preview image.
                        let app_weak_p = app_weak_t.clone();
                        let on_progress = move |completed: usize,
                                                 total: usize,
                                                 preview: Option<PlanarImage<f32>>| {
                            let msg = format!("Fusing [Strata] step {completed}/{total}…");
                            let frac = completed as f32 / total as f32;
                            let progress = 0.50 + 0.45 * frac;
                            let _ = slint::invoke_from_event_loop({
                                let a = app_weak_p.clone();
                                move || {
                                    if let Some(app) = a.upgrade() {
                                        app.set_status(msg.into());
                                        app.set_progress(progress);
                                        app.set_step_progress(frac);
                                        if let Some(img) = preview {
                                            app.set_displayed_image(Image::from_rgba8(
                                                planar_to_rgba_buffer(&img),
                                            ));
                                        }
                                    }
                                }
                            });
                        };
                        fuse_strata(&ordered_planar, &run_settings, on_progress)
                    } else if let Some(res) = fuse_relief(
                        &ordered_planar,
                        &run_settings,
                        Some(&loaded_img0),
                        Some(&app_weak_t),
                        Some(&relief_preview_ctx_t)
                    ) {
                        res
                    } else {
                        let _ = slint::invoke_from_event_loop({
                            let a = app_weak_t.clone();
                            move || { if let Some(app) = a.upgrade() { app.set_status("Relief preview cancelled.".into()); app.set_progress(1.0); } }
                        });
                        return None;
                    };
                    let elapsed_fuse = t_fuse.elapsed();
                    tracing::info!(prefix, elapsed_ms = elapsed_fuse.as_millis(), "fuse stage complete");

                    // The crop (if any) was already applied to every input
                    // frame before fusion (see `crop_rect` above) — `fused`
                    // is therefore already at the cropped dimensions and
                    // needs no further cropping here.
                    //
                    // Optionally stretch the cropped result back up to the
                    // original (pre-crop) canvas resolution. Both passes in
                    // "Both" mode flow through this same closure, so the
                    // retouch base/src wiring below always sees consistently
                    // (re)sized frames either way.
                    let fused = if run_settings.crop_to_common_area
                        && run_settings.resize_cropped_to_original
                        && crop_rect.is_some()
                        && (fused.width != orig_canvas_w || fused.height != orig_canvas_h)
                    {
                        tracing::info!(
                            prefix,
                            crop_w = fused.width, crop_h = fused.height,
                            orig_w = orig_canvas_w, orig_h = orig_canvas_h,
                            "resizing cropped result back to original canvas resolution"
                        );
                        resize_planar_clamped(&fused, orig_canvas_w, orig_canvas_h)
                    } else {
                        fused
                    };
                    let fused_arc = Arc::new(fused);

                    // Default retouch wiring: base and src both point at this
                    // pass's own result, so the brush is a (harmless) no-op
                    // outside "Both" mode — there's no second source to paint
                    // in. When both Relief and Apex actually run, the dispatch
                    // below overwrites this with the *other* pass's result as
                    // `src`, so the brush blends between the two algorithms.
                    {
                        let mut rs = retouch_arc_t.lock().unwrap();
                        rs.session = Some(RetouchSession::new(
                            Arc::clone(&fused_arc),
                            Arc::clone(&fused_arc),
                        ));
                        rs.history = RetouchHistory::new();
                        // Seed the history with the pristine (unpainted) alpha
                        // mask so the very first brush stroke has a baseline to
                        // undo back to — without this, `push`ing only after a
                        // stroke leaves the cursor at index 0 with nothing
                        // earlier to restore.
                        let baseline = rs.session.as_ref().map(RetouchSession::snapshot_alpha);
                        if let Some(snap) = baseline {
                            rs.history.push(snap);
                        }
                    }

                    let slint_buffer = planar_to_rgba_buffer(&fused_arc);

                    let t_encode = Instant::now();
                    let composite = planar_to_rgb_image(&fused_arc);

                    let temp_path = std::env::temp_dir()
                        .join(format!("stacker_result_{prefix}_{ts_ms}.png"));
                    let _ = composite.save(&temp_path);
                    let elapsed_encode = t_encode.elapsed();
                    tracing::info!(
                        prefix, output_path = %temp_path.display(),
                        format = "PNG", elapsed_ms = elapsed_encode.as_millis(),
                        "encode stage complete"
                    );

                    retouch_arc_t.lock().unwrap().result_path = Some(temp_path.clone());

                    let elapsed_total_s = t_total.elapsed().as_secs();
                    let elapsed_str = if elapsed_total_s < 60 {
                        format!("{elapsed_total_s}s")
                    } else {
                        format!("{}m {}s", elapsed_total_s / 60, elapsed_total_s % 60)
                    };

                    let status = if run_settings.auto_cull {
                        format!(
                            "Merged [{prefix}] {stack_total}/{total} ({culled_count} dropped) — log: {log_path_str}"
                        )
                    } else {
                        format!(
                            "Merged [{prefix}] {stack_total}/{total} — log: {log_path_str}"
                        )
                    };

                    let _ = slint::invoke_from_event_loop({
                        let a  = app_weak_t.clone();
                        let rf = Arc::clone(&results_arc_t);
                        let vf = Arc::clone(&viewed_arc_t);
                        let n_frames = stack_total as i32;
                        move || {
                            if let Some(app) = a.upgrade() {
                                let snapshot = {
                                    let mut guard = rf.lock().unwrap();
                                    guard.push(temp_path.clone());
                                    guard.clone()
                                };
                                update_result_list(&app, &snapshot);
                                *vf.lock().unwrap() = Some(temp_path);
                                app.set_displayed_image(Image::from_rgba8(slint_buffer));
                                app.set_progress(1.0);
                                app.set_status(status.into());
                                app.set_frame_count(n_frames);
                                app.set_elapsed_text(elapsed_str.into());
                                // Undo/redo button state lives entirely in the
                                // `RetouchWindow` popup now — there is no
                                // `can-undo`/`can-redo` property on `App` to
                                // reset here any more (see
                                // `callbacks::retouch_window`). A freshly
                                // stacked result starts with a fresh,
                                // pristine session anyway (seeded above), so
                                // the popup would show "nothing to undo" the
                                // next time it's opened regardless.
                            }
                        }
                    });
                    Some(fused_arc)
                };

                if cancel_requested_t.load(Ordering::SeqCst) {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Stacking cancelled before fusion.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                    return;
                }

                // Fusion itself (Apex/Relief/AI) runs as a single non-interruptible
                // call per pass — `do_stack` has no internal cancellation points.
                // Apex and Relief now report interim status/progress during the
                // pass (Apex also periodically refreshes the displayed preview
                // as frames are folded in, same as AI already did), but the
                // pass still cannot be interrupted mid-flight — cancellation is
                // only checked at the pass boundaries below. In "Both" mode the
                // boundary between the two passes is still a checkpoint, so a
                // cancel during the Relief pass skips Apex instead of running it
                // to completion first. `do_stack` returns `None` on AI failure
                // or a Relief-preview cancel, in which case it has already set
                // its own status message — the dispatch below must not
                // overwrite that with "Finished!".
                let last_result = if algo_str.contains("AI") || algo_str.contains("Neural") {
                    do_stack("AI")
                } else if run_all_three {
                    do_stack("Strata").and_then(|_strata_arc| {
                        if cancel_requested_t.load(Ordering::SeqCst) {
                            finish_ui!(app_weak_t, |app: App| {
                                app.set_status("Stacking cancelled after Strata pass.".into());
                                app.set_progress(1.0);
                                app.set_current_frame_index(-1);
                            });
                            return None;
                        }
                        do_stack("Relief").and_then(|relief_arc| {
                            if cancel_requested_t.load(Ordering::SeqCst) {
                                finish_ui!(app_weak_t, |app: App| {
                                    app.set_status("Stacking cancelled after Relief pass.".into());
                                    app.set_progress(1.0);
                                    app.set_current_frame_index(-1);
                                });
                                return None;
                            }
                            do_stack("Apex").inspect(|apex_arc| {
                                // All three passes ran: rewire the retouch session so
                                // the brush blends between two algorithms'
                                // results instead of do_stack's default (inert)
                                // base==src wiring from the Apex pass above —
                                // base is Apex (the primary/last-shown result),
                                // src is Relief (what the brush paints in).
                                let mut rs = retouch_arc_t.lock().unwrap();
                                rs.session = Some(RetouchSession::new(
                                    Arc::clone(apex_arc),
                                    Arc::clone(&relief_arc),
                                ));
                                rs.history = RetouchHistory::new();
                                // Same baseline-seeding as the single-pass wiring
                                // above: without this the first stroke after
                                // rewiring the session has no earlier
                                // snapshot to undo back to.
                                let baseline = rs.session.as_ref().map(RetouchSession::snapshot_alpha);
                                if let Some(snap) = baseline {
                                    rs.history.push(snap);
                                }
                            })
                        })
                    })
                } else if algo_str.contains("Relief") {
                    do_stack("Relief")
                } else if algo_str.contains("Strata") {
                    do_stack("Strata")
                } else {
                    do_stack("Apex")
                };

                if last_result.is_some() {
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_status("Finished! All results available below.".into());
                        app.set_progress(1.0);
                        app.set_current_frame_index(-1);
                    });
                } else {
                    // `do_stack` already reported its own failure/cancellation
                    // status — just release the processing flag (and clear the
                    // row highlight) without overwriting that message with
                    // "Finished!".
                    finish_ui!(app_weak_t, |app: App| {
                        app.set_current_frame_index(-1);
                    });
                }
            });
        });
    }
}
