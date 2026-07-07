use std::{
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
    time::Duration,
};

use notify::{EventKind, Watcher};
use slint::{ComponentHandle, Image, SharedString};
use stacker_align::transform::resize_planar_clamped;

use crate::{
    App,
    align_cache::AlignedCache,
    callbacks::retouch_window::{RetouchWindowSlot, refresh_if_visible},
    image_utils::{load_as_planar, planar_to_rgba_buffer},
    sort_cull::{SortCullCache, SortCullPromptContext},
    ui_helpers::{self, update_result_list, update_source_list},
};

/// Wires the file-list, drag&drop/monitor, and zoom callbacks.
///
/// Covers loading, clicking, removing, reordering, clearing source files;
/// clicking/removing results; the folder-monitor start/stop; and the zoom
/// in/out/100% controls.
///
/// # Panics
///
/// The registered callbacks call `.lock().unwrap()` on the various shared
/// `Mutex`-guarded state (`file_paths`, `result_paths`, `retouch_state`,
/// etc.); those panic only if another thread already panicked while holding
/// the same lock (mutex poisoning), which does not happen in normal
/// operation.
#[allow(clippy::too_many_arguments)]
// `too_many_lines`: this registers every file-list/monitor/zoom callback in
// one place so `App::run`'s wiring stays in one readable call site; each
// `on_*` closure below is independent and short, splitting the function
// would only relocate code, not shorten any one callback's logic.
#[allow(clippy::too_many_lines)]
pub fn wire(
    app: &App,
    file_paths: &Arc<Mutex<Vec<PathBuf>>>,
    result_paths: &Arc<Mutex<Vec<PathBuf>>>,
    currently_viewed_path: &Arc<Mutex<Option<PathBuf>>>,
    aligned_cache: &Arc<Mutex<Option<AlignedCache>>>,
    current_view_idx: &Arc<Mutex<Option<usize>>>,
    retouch_state: &Arc<Mutex<crate::retouch::RetouchState>>,
    monitor_stop_tx: &Arc<Mutex<Option<std::sync::mpsc::Sender<()>>>>,
    retouch_popup_slot: &RetouchWindowSlot,
    sort_cull_cache: &Arc<Mutex<Option<SortCullCache>>>,
    sort_cull_prompt_ctx: &Arc<Mutex<Option<SortCullPromptContext>>>,
    cancel_requested: &Arc<std::sync::atomic::AtomicBool>,
) {
    let app_weak = app.as_weak();

    // ── Open files ───────────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        let viewed = Arc::clone(currently_viewed_path);
        app.on_request_open_files(move || {
            // RAW extensions are only offered in `raw`-feature builds: without
            // the feature, `stacker_core::io::load_frame` would just reject
            // them with a clear "rebuild with --features raw" error, so
            // there is no point letting the file picker select them.
            let mut dialog = rfd::FileDialog::new();
            dialog = if cfg!(feature = "raw") {
                let mut exts: Vec<&str> = vec!["png", "jpg", "jpeg", "tif", "tiff"];
                exts.extend_from_slice(stacker_core::io::RAW_EXTENSIONS);
                dialog.add_filter("Images", &exts)
            } else {
                dialog.add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff"])
            };
            if let Some(files) = dialog.pick_files()
                && let Some(app) = app_weak.upgrade()
            {
                let mut p = paths.lock().unwrap();
                for f in files {
                    if !p.contains(&f) {
                        p.push(f);
                    }
                }
                update_source_list(&app, &p);
                app.set_has_images(!p.is_empty());

                {
                    app.set_frame_count(p.len() as i32);
                }
                if let Some(first) = p.first() {
                    *viewed.lock().unwrap() = Some(first.clone());
                    if let Ok(img) = Image::load_from_path(first) {
                        app.set_displayed_image(img);
                    }
                }
            }
        });
    }

    // ── Source-file click ────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        let viewed = Arc::clone(currently_viewed_path);
        let aligned_cache_click = Arc::clone(aligned_cache);
        let view_idx_click = Arc::clone(current_view_idx);
        let retouch_arc_click = Arc::clone(retouch_state);
        let retouch_popup_slot_click = Rc::clone(retouch_popup_slot);
        app.on_file_clicked(move |idx| {
            let idx = idx as usize;
            *view_idx_click.lock().unwrap() = Some(idx);
            // A different source frame was picked as the donor preview — keep
            // the retouch popup's displayed composite (if any session is
            // active and the popup is visible) in sync too; see
            // `refresh_if_visible`'s doc comment.
            refresh_if_visible(&retouch_popup_slot_click, &retouch_arc_click);
            if let Some(app) = app_weak.upgrade() {
                let p = paths.lock().unwrap();
                if idx < p.len() {
                    let path = p[idx].clone();
                    drop(p);
                    *viewed.lock().unwrap() = Some(path.clone());

                    // Deliberately does NOT touch `retouch_state` here any
                    // more. The retouch popup is a separate, independent
                    // window now — its session must survive the user simply
                    // browsing source frames in the main window. This used
                    // to unconditionally clear `rs.session`, which silently
                    // killed the popup's brush/undo the moment ANY source
                    // frame was clicked, even while the popup was open and
                    // actively being retouched (this was the actual cause
                    // of "brush/undo do nothing in the popup" reports, not
                    // an input-routing bug). `perform_save_current_image`
                    // (see `save.rs`) independently guards Save against
                    // applying a stale session's composite to the wrong
                    // target by checking `rs.result_path` against the
                    // viewed path itself, so nothing here needs to
                    // defensively wipe the session to stay safe.

                    // When "Show Aligned" is on and the cache is populated for
                    // this index, display the aligned (warped + cropped) frame.
                    if app.get_show_aligned() {
                        let cache = aligned_cache_click.lock().unwrap();
                        if let Some(ref ac) = *cache
                            && idx < ac.frames.len()
                        {
                            let frame = &ac.frames[idx];
                            let cropped = if let Some((cx, cy, cw, ch)) = ac.crop {
                                stacker_core::memory::extract_tile(frame, cx, cy, cw, ch)
                            } else {
                                frame.clone()
                            };
                            let buf = planar_to_rgba_buffer(&cropped);
                            app.set_displayed_image(Image::from_rgba8(buf));
                            return;
                        }
                    }

                    // Fallback: load from disk as usual.
                    if let Ok(img) = Image::load_from_path(&path) {
                        app.set_displayed_image(img);
                    }
                }
            }
        });
    }

    // ── Result-file click ────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let results = Arc::clone(result_paths);
        let viewed = Arc::clone(currently_viewed_path);
        app.on_result_clicked(move |idx| {
            let idx = idx as usize;
            if let Some(app) = app_weak.upgrade() {
                let r = results.lock().unwrap();
                if idx < r.len() {
                    let path = r[idx].clone();
                    drop(r);
                    *viewed.lock().unwrap() = Some(path.clone());

                    // Deliberately does NOT touch `retouch_state` here any
                    // more: the retouch popup is a separate, independent
                    // window, and its session must survive simply browsing
                    // between result thumbnails in the main window's own
                    // preview. `perform_save_current_image` (see `save.rs`)
                    // independently guards Save against applying a stale
                    // session's composite to the wrong target by checking
                    // `rs.result_path` against the viewed path itself.

                    if let Ok(img) = Image::load_from_path(&path) {
                        app.set_displayed_image(img);
                    }
                }
            }
        });
    }

    // ── Use aligned source frame as retouch donor ───────────────────────
    // Right-click → "Use as Retouch Source (aligned)" on a Source Files row.
    // Lets the brush heal from a raw, pre-fusion frame instead of only ever
    // being able to heal from the other fusion algorithm's result — see
    // `RetouchSession::set_src`'s doc comment. No-op (silently) if there's
    // no active retouch session (nothing to swap the donor of yet), or if
    // this source frame was never aligned (e.g. alignment hasn't run).
    {
        let aligned_cache_donor = Arc::clone(aligned_cache);
        let retouch_arc_donor = Arc::clone(retouch_state);
        let retouch_popup_slot_donor = Rc::clone(retouch_popup_slot);
        app.on_use_aligned_as_retouch_source(move |idx| {
            let idx = idx as usize;
            // Computed as a single expression (rather than a `let`-`else`
            // early-return sequence against a named `cache` binding) so the
            // lock guard is dropped the instant this block ends, not held
            // across separate statements (clippy::significant_drop_tightening).
            let cropped = aligned_cache_donor.lock().unwrap().as_ref().and_then(|ac| {
                let frame = ac.frames.get(idx)?;
                Some(if let Some((cx, cy, cw, ch)) = ac.crop {
                    stacker_core::memory::extract_tile(frame, cx, cy, cw, ch)
                } else {
                    frame.clone()
                })
            });
            let Some(cropped) = cropped else {
                return;
            };

            let mut rs = retouch_arc_donor.lock().unwrap();
            let Some(session) = rs.session.as_mut() else {
                return;
            };
            let (bw, bh) = (session.base.width, session.base.height);
            let donor = if cropped.width == bw && cropped.height == bh {
                cropped
            } else {
                // The aligned cache holds pre-crop frames; `base` may be at
                // the cropped size, or restretched back to the original
                // canvas (see `resize_cropped_to_original` in
                // `callbacks::stacking`) — resize to whichever `base`
                // actually ended up at rather than assuming either case.
                resize_planar_clamped(&cropped, bw, bh)
            };
            session.set_src(Arc::new(donor));
            drop(rs);
            refresh_if_visible(&retouch_popup_slot_donor, &retouch_arc_donor);
        });
    }

    // ── Use a different result as retouch donor ─────────────────────────
    // Right-click → "Use as Retouch Source" on a Results row. Reloads that
    // result's saved PNG from disk and swaps it in as the donor — lets the
    // brush heal from e.g. Strata instead of whichever pass is currently
    // wired as `src`, without opening that result's own popup. Same no-op
    // conditions as above when there's no active session.
    {
        let results_donor = Arc::clone(result_paths);
        let retouch_arc_result_donor = Arc::clone(retouch_state);
        let retouch_popup_slot_result_donor = Rc::clone(retouch_popup_slot);
        app.on_use_result_as_retouch_source(move |idx| {
            let idx = idx as usize;
            let path = {
                let r = results_donor.lock().unwrap();
                r.get(idx).cloned()
            };
            let Some(path) = path else {
                return;
            };
            let Some(loaded) = image::open(&path).ok().map(|img| load_as_planar(&img)) else {
                return;
            };

            let mut rs = retouch_arc_result_donor.lock().unwrap();
            let Some(session) = rs.session.as_mut() else {
                return;
            };
            let (bw, bh) = (session.base.width, session.base.height);
            let donor = if loaded.width == bw && loaded.height == bh {
                loaded
            } else {
                resize_planar_clamped(&loaded, bw, bh)
            };
            session.set_src(Arc::new(donor));
            drop(rs);
            refresh_if_visible(&retouch_popup_slot_result_donor, &retouch_arc_result_donor);
        });
    }

    // ── Remove file ──────────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        app.on_remove_file(move |idx| {
            let idx = idx as usize;
            if let Some(app) = app_weak.upgrade() {
                let mut p = paths.lock().unwrap();
                if idx < p.len() {
                    p.remove(idx);
                    update_source_list(&app, &p);
                    if p.is_empty() {
                        app.set_has_images(false);
                    }

                    {
                        app.set_frame_count(p.len() as i32);
                    }
                }
            }
        });
    }

    // ── Recalc Memory ──────────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        app.on_recalc_memory(move || {
            if let Some(app) = app_weak.upgrade() {
                let p = paths.lock().unwrap();
                ui_helpers::update_memory_estimate(&app, &p);
            }
        });
    }

    // ── Move file up/down ──────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        app.on_move_file_up(move |idx| {
            let idx = idx as usize;
            if let Some(app) = app_weak.upgrade() {
                // Snapshot + drop the lock before touching the UI so the
                // mutex is never held across the (potentially slow) list
                // rebuild (clippy: significant_drop_tightening).
                let snapshot = {
                    let mut p = paths.lock().unwrap();
                    if idx > 0 && idx < p.len() {
                        p.swap(idx, idx - 1);
                        Some(p.clone())
                    } else {
                        None
                    }
                };
                if let Some(snapshot) = snapshot {
                    update_source_list(&app, &snapshot);
                }
            }
        });
    }
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        app.on_move_file_down(move |idx| {
            let idx = idx as usize;
            if let Some(app) = app_weak.upgrade() {
                // See on_move_file_up: snapshot + drop the lock before the
                // UI rebuild (clippy: significant_drop_tightening).
                let snapshot = {
                    let mut p = paths.lock().unwrap();
                    if idx + 1 < p.len() {
                        p.swap(idx, idx + 1);
                        Some(p.clone())
                    } else {
                        None
                    }
                };
                if let Some(snapshot) = snapshot {
                    update_source_list(&app, &snapshot);
                }
            }
        });
    }

    // ── Clear all files ──────────────────────────────────────────────────
    // Resets every piece of *processing* state that was derived from the
    // now-cleared source list — an `AlignedCache`/`SortCullCache` keyed to
    // files that no longer exist would otherwise still look "fresh" (by
    // fingerprint accident) the moment a brand-new, same-sized batch of
    // images is added, silently reusing stale alignment/sort/cull results
    // instead of recomputing for the new set. Deliberately does NOT touch
    // `result_paths`: those are on-disk output files that may not have been
    // saved yet, so Clear must never make them unreachable from the UI (see
    // the owner's "keep only the output files in case they weren't saved
    // yet" note).
    {
        let app_weak = app_weak.clone();
        let paths = Arc::clone(file_paths);
        let aligned_cache_clear = Arc::clone(aligned_cache);
        let view_idx_clear = Arc::clone(current_view_idx);
        let viewed_clear = Arc::clone(currently_viewed_path);
        let retouch_clear = Arc::clone(retouch_state);
        let sort_cull_cache_clear = Arc::clone(sort_cull_cache);
        let sort_cull_prompt_ctx_clear = Arc::clone(sort_cull_prompt_ctx);
        let cancel_requested_clear = Arc::clone(cancel_requested);
        let retouch_popup_slot_clear = Rc::clone(retouch_popup_slot);
        app.on_clear_all_files(move || {
            if let Some(app) = app_weak.upgrade() {
                let mut p = paths.lock().unwrap();
                p.clear();
                update_source_list(&app, &p);
                drop(p);
                *aligned_cache_clear.lock().unwrap() = None;
                *view_idx_clear.lock().unwrap() = None;
                *viewed_clear.lock().unwrap() = None;
                *sort_cull_cache_clear.lock().unwrap() = None;
                // Dropping any parked prompt context here is safe exactly
                // like `ReliefPreviewContext`'s established pattern: a Stack
                // thread currently blocked in `reply_rx.recv()` waiting on
                // this popup's answer will simply see the channel close and
                // get `Err`, which its existing fallback already treats the
                // same as "Keep as is" (see `on_request_stack`'s Sort/Cull
                // stale-prompt handling) — no new failure mode is introduced.
                *sort_cull_prompt_ctx_clear.lock().unwrap() = None;
                app.set_sort_cull_stale_open(false);
                // A source frame is never a retouch target and a fresh
                // source-file batch invalidates any brush session built
                // against the old files' donor frames — same rule
                // `on_file_clicked`/`on_result_clicked` already apply when
                // switching targets.
                {
                    let mut rs = retouch_clear.lock().unwrap();
                    rs.session = None;
                    rs.history = stacker_algo::hybrid::retouch::RetouchHistory::new();
                    rs.result_path = None;
                }
                // If the retouch popup is open, it has nothing left to show
                // or paint — refresh_if_visible only repaints an existing
                // session's composite (a no-op now that session is None),
                // so explicitly clear its image/undo-redo state too instead
                // of leaving stale pixels and enabled buttons on screen.
                if let Some(window) = retouch_popup_slot_clear
                    .borrow()
                    .as_ref()
                    .map(crate::RetouchWindow::clone_strong)
                {
                    window.set_has_images(false);
                    window.set_can_undo(false);
                    window.set_can_redo(false);
                }
                cancel_requested_clear.store(false, std::sync::atomic::Ordering::SeqCst);
                app.set_has_images(false);
                app.set_frame_count(0);
            }
        });
    }

    // ── Remove result ──────────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let results_arc = Arc::clone(result_paths);
        app.on_remove_result(move |idx| {
            let idx = idx as usize;
            if let Some(app) = app_weak.upgrade() {
                // Snapshot + drop the lock before the UI rebuild
                // (clippy: significant_drop_tightening).
                let snapshot = {
                    let mut p = results_arc.lock().unwrap();
                    if idx < p.len() {
                        p.remove(idx);
                        Some(p.clone())
                    } else {
                        None
                    }
                };
                if let Some(snapshot) = snapshot {
                    update_result_list(&app, &snapshot);
                }
            }
        });
    }

    // ── Monitor ──────────────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let paths_arc = Arc::clone(file_paths);
        let monitor_stop_tx = Arc::clone(monitor_stop_tx);

        app.on_start_monitor(move || {
            if let Some(folder) = rfd::FileDialog::new().pick_folder()
                && let Some(app) = app_weak.upgrade()
            {
                app.set_is_monitoring(true);

                let (stop_tx, stop_rx) = std::sync::mpsc::channel();
                *monitor_stop_tx.lock().unwrap() = Some(stop_tx);

                let app_weak_t = app_weak.clone();
                let paths_arc_t = Arc::clone(&paths_arc);

                std::thread::spawn(move || {
                    let (notify_tx, notify_rx) = std::sync::mpsc::channel();
                    let mut watcher =
                        notify::recommended_watcher(notify_tx).expect("Failed to create watcher");
                    watcher
                        .watch(&folder, notify::RecursiveMode::NonRecursive)
                        .expect("Failed to watch directory");

                    let run_pass = || {
                        let mut files: Vec<PathBuf> = std::fs::read_dir(&folder)
                            .into_iter()
                            .flatten()
                            .filter_map(Result::ok)
                            .map(|e| e.path())
                            .filter(|p| {
                                let ext = p
                                    .extension()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_lowercase();
                                matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff")
                                    || (cfg!(feature = "raw")
                                        && stacker_core::io::is_raw_extension(&ext))
                            })
                            .collect();
                        files.sort();

                        let _ = slint::invoke_from_event_loop({
                            let a = app_weak_t.clone();
                            let pt = Arc::clone(&paths_arc_t);
                            move || {
                                if let Some(app) = a.upgrade() {
                                    (*pt.lock().unwrap()).clone_from(&files);
                                    update_source_list(&app, &files);
                                    app.set_has_images(!files.is_empty());
                                    app.set_frame_count(files.len() as i32);
                                    if !files.is_empty() {
                                        app.invoke_request_stack(
                                            SharedString::from("Apex"),
                                            SharedString::new(),
                                            SharedString::new(),
                                            SharedString::new(),
                                        );
                                    }
                                }
                            }
                        });
                    };

                    // Initial pass
                    run_pass();

                    loop {
                        if stop_rx.try_recv().is_ok() {
                            break;
                        }
                        if let Ok(Ok(notify::Event {
                            kind: EventKind::Create(_) | EventKind::Modify(_),
                            ..
                        })) = notify_rx.recv_timeout(Duration::from_millis(500))
                        {
                            std::thread::sleep(Duration::from_millis(500));
                            while notify_rx.try_recv().is_ok() {}
                            run_pass();
                        }
                    }

                    let _ = slint::invoke_from_event_loop({
                        let a = app_weak_t.clone();
                        move || {
                            if let Some(app) = a.upgrade() {
                                app.set_is_monitoring(false);
                            }
                        }
                    });
                });
            }
        });
    }

    {
        let monitor_stop_tx = Arc::clone(monitor_stop_tx);
        app.on_stop_monitor(move || {
            let value = monitor_stop_tx.lock().unwrap().take();
            if let Some(tx) = value {
                let _ = tx.send(());
            }
        });
    }

    // ── Zoom controls ─────────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        app.on_zoom_in(move || {
            if let Some(app) = app_weak.upgrade() {
                let z = (app.get_zoom_factor() * 1.25).min(16.0);
                app.set_zoom_factor(z);
            }
        });
    }
    {
        let app_weak = app_weak.clone();
        app.on_zoom_out(move || {
            if let Some(app) = app_weak.upgrade() {
                // Low enough that "Fit" on a large source image in a small
                // canvas window (e.g. a 40+ MP frame at a modest window
                // size) still lands within the -/+ range, not clamped to it.
                let z = (app.get_zoom_factor() / 1.25).max(0.02);
                app.set_zoom_factor(z);
            }
        });
    }
    // "Fit" is computed directly in app.slint (it needs the canvas's on-screen
    // size and the loaded image's intrinsic pixel size, both naturally
    // available there); there is no `zoom-fit` Rust callback to wire.
    {
        app.on_zoom_100(move || {
            if let Some(app) = app_weak.upgrade() {
                app.set_zoom_factor(1.0);
            }
        });
    }
}
