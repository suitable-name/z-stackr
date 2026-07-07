//! Wires the "Save Project…" / "Load Project…" toolbar buttons to the
//! picture-set project format in `crate::project`.
//!
//! Clicking "Save Project…" first asks Rust whether the active retouch
//! session actually has any painted pixels (`on_request_save_project`),
//! since the optional "save rebrush history" checkbox on `SaveProjectDialog`
//! must not even appear otherwise. The dialog then reports the user's three
//! (or four) choices back via `on_confirm_save_project`, at which point this
//! module asks for a destination file and actually writes it. Load has no
//! options — it applies everything the project file has back onto the GUI's
//! own state, on a best-effort basis (a project referencing since-moved,
//! non-embedded files simply drops those specific entries rather than
//! failing the whole load).

use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use slint::{ComponentHandle, SharedString};
use stacker_algo::hybrid::retouch::{RetouchHistory, RetouchSession};

use crate::{
    App,
    image_utils::load_as_planar,
    project::{self, BlobRef, HistorySnapshot, LoadedProject, RetouchSnapshot, SaveOptions},
    retouch::RetouchState,
    settings::StackingSettings,
    ui_helpers::{push_settings_to_ui, update_result_list, update_source_list},
};

/// Wires `request-save-project`, `confirm-save-project` and
/// `load-project-requested`.
///
/// # Panics
///
/// The registered callbacks call `.lock().unwrap()` on the shared
/// `Mutex`-guarded state (`file_paths`, `result_paths`, `settings_arc`,
/// `retouch_state`); those panic only if another thread already panicked
/// while holding the same lock (mutex poisoning), which does not happen in
/// normal operation.
// `too_many_lines`: registers all three project-file callbacks
// (request/confirm-save, load) in one place, mirroring every other
// `callbacks::*::wire` function's established "one readable call site per
// domain" convention (see e.g. `callbacks::files::wire`'s identical allow).
#[allow(clippy::too_many_lines)]
pub fn wire(
    app: &App,
    file_paths: &Arc<Mutex<Vec<PathBuf>>>,
    result_paths: &Arc<Mutex<Vec<PathBuf>>>,
    settings_arc: &Arc<Mutex<StackingSettings>>,
    retouch_state: &Arc<Mutex<RetouchState>>,
) {
    let app_weak = app.as_weak();

    // ── Open the save dialog, with the rebrush-history checkbox gated on
    // whether there's actually anything painted to save history for ──────
    {
        let app_weak = app_weak.clone();
        let retouch_state = Arc::clone(retouch_state);
        app.on_request_save_project(move || {
            let has_brush_strokes = retouch_state
                .lock()
                .unwrap()
                .session
                .as_ref()
                .is_some_and(|session| session.alpha.luma.iter().any(|&a| a > 0.0));
            if let Some(app) = app_weak.upgrade() {
                app.set_can_save_rebrush_history(has_brush_strokes);
                app.set_save_project_open(true);
            }
        });
    }

    // ── Save ─────────────────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let file_paths = Arc::clone(file_paths);
        let result_paths = Arc::clone(result_paths);
        let settings_arc = Arc::clone(settings_arc);
        let retouch_state = Arc::clone(retouch_state);

        app.on_confirm_save_project(
            move |include_settings, embed_sources, embed_outputs, include_rebrush_history| {
                let Some(dest) = rfd::FileDialog::new()
                    .add_filter("z-stackr Project", &["zsproj"])
                    .set_file_name("picture_set.zsproj")
                    .save_file()
                else {
                    return;
                };

                let sources = file_paths.lock().unwrap().clone();
                let outputs: Vec<(PathBuf, String)> = result_paths
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|p| (p.clone(), infer_algorithm_label(p)))
                    .collect();
                let settings = settings_arc.lock().unwrap().clone();

                // The lock is only ever held long enough to clone out plain
                // data — `project::save_project` (file I/O + compression)
                // runs entirely after this block ends and the guard drops.
                let retouch_snapshot = {
                    let rs = retouch_state.lock().unwrap();
                    rs.session.as_ref().and_then(|session| {
                        let history = include_rebrush_history.then(|| {
                            let (stack, cursor) = rs.history.export_full();
                            HistorySnapshot { stack, cursor }
                        });
                        RetouchSnapshot::build(
                            (session.base.width, session.base.height),
                            session.snapshot_alpha(),
                            rs.result_path.as_deref(),
                            &outputs,
                            history,
                        )
                    })
                };

                let opts = SaveOptions {
                    include_settings,
                    embed_sources,
                    embed_outputs,
                    include_rebrush_history,
                };

                let result = project::save_project(
                    &dest,
                    &sources,
                    &outputs,
                    &settings,
                    retouch_snapshot,
                    opts,
                );

                if let Some(app) = app_weak.upgrade() {
                    match result {
                        Ok(()) => {
                            app.set_status(
                                format!("Saved picture set to {}", dest.display()).into(),
                            );
                        }
                        Err(e) => {
                            tracing::error!("failed to save project: {e}");
                            app.set_status(format!("Failed to save picture set: {e}").into());
                        }
                    }
                }
            },
        );
    }

    // ── Load ─────────────────────────────────────────────────────────────
    {
        // Last use of the outer `app_weak` — moved rather than cloned.
        let file_paths = Arc::clone(file_paths);
        let result_paths = Arc::clone(result_paths);
        let settings_arc = Arc::clone(settings_arc);
        let retouch_state = Arc::clone(retouch_state);

        app.on_load_project_requested(move || {
            let Some(src) = rfd::FileDialog::new()
                .add_filter("z-stackr Project", &["zsproj"])
                .pick_file()
            else {
                return;
            };

            let loaded = match project::load_project(&src) {
                Ok(loaded) => loaded,
                Err(e) => {
                    tracing::error!("failed to load project {}: {e}", src.display());
                    if let Some(app) = app_weak.upgrade() {
                        app.set_status(format!("Failed to load picture set: {e}").into());
                    }
                    return;
                }
            };

            apply_loaded_project(
                &loaded,
                &app_weak,
                &file_paths,
                &result_paths,
                &settings_arc,
                &retouch_state,
            );
        });
    }
}

/// Best-effort guess at which algorithm produced a result file, from its
/// filename — the result-path list has no separate parallel "algorithm"
/// field today, but every temp result filename already embeds this (see
/// `callbacks::stacking`/`stack_pipeline`'s `stacker_result_{prefix}_...`
/// naming), so a substring match is enough for a project file's
/// informational `OutputEntry::algorithm` label.
fn infer_algorithm_label(path: &Path) -> String {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    for label in ["Apex", "Relief", "Strata", "AI", "tiled"] {
        if name.contains(label) {
            return label.to_owned();
        }
    }
    "unknown".to_owned()
}

/// Resolve a saved entry back to a usable on-disk path: write embedded
/// bytes out to a temp file when present, otherwise fall back to the
/// original reference path if it still exists. Returns `None` if neither
/// is available (embed missing/failed to write, and the original path is
/// gone) — the caller drops that one entry rather than failing the load.
fn resolve_entry_path(
    original_path: Option<&Path>,
    embedded: Option<&BlobRef>,
    blob: &[u8],
    tmp_prefix: &str,
    idx: usize,
) -> Option<PathBuf> {
    if let Some(blob_ref) = embedded {
        let bytes = project::blob_slice(blob, blob_ref);
        let ext = if blob_ref.ext.is_empty() {
            "png"
        } else {
            blob_ref.ext.as_str()
        };
        let tmp_path = std::env::temp_dir().join(format!("zsproj_{tmp_prefix}_{idx}.{ext}"));
        if std::fs::write(&tmp_path, bytes).is_ok() {
            return Some(tmp_path);
        }
        tracing::warn!("failed to extract embedded {tmp_prefix} #{idx} to a temp file");
    }
    original_path.filter(|p| p.exists()).map(Path::to_path_buf)
}

/// Rebuild a `RetouchSession` + its history from a loaded
/// `project::RetouchSnapshot`, if its two donor outputs both resolved and
/// end up matching the snapshot's saved dimensions.
///
/// Returns `None` when the snapshot can't be honoured (missing/mismatched
/// donors) — the caller falls back to a fully-cleared retouch state.
fn rebuild_retouch_session(
    snap: &RetouchSnapshot,
    outputs: &[PathBuf],
) -> Option<(RetouchSession, RetouchHistory, PathBuf)> {
    let base_path = outputs.get(snap.base_output_idx)?;
    let src_path = outputs.get(snap.src_output_idx)?;
    let base_planar = load_as_planar(&image::open(base_path).ok()?);
    let src_planar = load_as_planar(&image::open(src_path).ok()?);
    if base_planar.width != snap.width
        || base_planar.height != snap.height
        || src_planar.width != snap.width
        || src_planar.height != snap.height
    {
        return None;
    }

    let mut session = RetouchSession::new(Arc::new(base_planar), Arc::new(src_planar));
    session.restore_alpha(snap.alpha.clone());

    let mut history = RetouchHistory::new();
    if let Some(h) = &snap.history {
        history.restore_full(h.stack.clone(), h.cursor);
    } else {
        // No saved history — only the loaded state itself is available, so
        // undo/redo works for new strokes made after loading but can't
        // reach back into whatever strokes existed before the save.
        history.push(snap.alpha.clone());
    }

    Some((session, history, base_path.clone()))
}

/// Applies a [`LoadedProject`]'s manifest back onto the GUI's own
/// `file_paths`/`result_paths`/`settings_arc`/`retouch_state` and refreshes
/// the corresponding UI lists — the shared tail end of
/// `on_load_project_requested`.
#[allow(clippy::too_many_lines)] // one cohesive "apply every manifest section" sequence
fn apply_loaded_project(
    loaded: &LoadedProject,
    app_weak: &slint::Weak<App>,
    file_paths: &Arc<Mutex<Vec<PathBuf>>>,
    result_paths: &Arc<Mutex<Vec<PathBuf>>>,
    settings_arc: &Arc<Mutex<StackingSettings>>,
    retouch_state: &Arc<Mutex<RetouchState>>,
) {
    let manifest = &loaded.manifest;

    if let Some(settings) = manifest.settings.clone() {
        *settings_arc.lock().unwrap() = settings.clone();
        if let Some(app) = app_weak.upgrade() {
            push_settings_to_ui(&app, &settings);
        }
    }

    let new_sources: Vec<PathBuf> = manifest
        .sources
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| {
            resolve_entry_path(
                Some(entry.original_path.as_path()),
                entry.embedded.as_ref(),
                &loaded.blob,
                "source",
                idx,
            )
        })
        .collect();
    (*file_paths.lock().unwrap()).clone_from(&new_sources);

    let new_outputs: Vec<PathBuf> = manifest
        .outputs
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| {
            resolve_entry_path(
                entry.original_path.as_deref(),
                entry.embedded.as_ref(),
                &loaded.blob,
                "output",
                idx,
            )
        })
        .collect();
    (*result_paths.lock().unwrap()).clone_from(&new_outputs);

    let dropped_sources = manifest.sources.len() - new_sources.len();
    let dropped_outputs = manifest.outputs.len() - new_outputs.len();

    let restored = manifest
        .retouch
        .as_ref()
        .and_then(|snap| rebuild_retouch_session(snap, &new_outputs));
    {
        let mut rs = retouch_state.lock().unwrap();
        if let Some((session, history, result_path)) = restored {
            rs.session = Some(session);
            rs.history = history;
            rs.result_path = Some(result_path);
        } else {
            rs.session = None;
            rs.history = RetouchHistory::new();
            rs.result_path = None;
        }
    }

    if let Some(app) = app_weak.upgrade() {
        update_source_list(&app, &new_sources);
        update_result_list(&app, &new_outputs);
        app.set_has_images(!new_sources.is_empty());
        app.set_frame_count(new_sources.len() as i32);

        let mut status = format!(
            "Loaded picture set: {} source(s), {} result(s)",
            new_sources.len(),
            new_outputs.len()
        );
        if dropped_sources > 0 || dropped_outputs > 0 {
            // `write!` into the existing `String` rather than
            // `push_str(&format!(...))`, which would allocate an
            // intermediate `String` just to copy it right back out.
            let _ = write!(
                status,
                " ({dropped_sources} source(s) and {dropped_outputs} result(s) could not be \
                 found and were skipped)"
            );
        }
        app.set_status(SharedString::from(status));
    }
}
