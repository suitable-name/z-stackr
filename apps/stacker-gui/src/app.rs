use std::{
    cell::RefCell,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex, atomic::AtomicBool},
};

use slint::ComponentHandle;
use stacker_core::telemetry::init_tracing_to_file;

use crate::{
    App,
    align_cache::AlignedCache,
    callbacks,
    retouch::RetouchState,
    settings::StackingSettings,
    sort_cull::{SortCullCache, SortCullPromptContext},
    stacking::ReliefPreviewContext,
    ui_helpers::push_settings_to_ui,
};

// ── main ──────────────────────────────────────────────────────────────────────

/// # Errors
///
/// Returns [`slint::PlatformError`] if the Slint `App` window fails to
/// initialise (`App::new()?`), or if running the Slint event loop
/// (`app.run()?`) fails.
pub fn run() -> Result<(), slint::PlatformError> {
    let log_path = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("stacker.log");
    if let Err(e) = init_tracing_to_file(&log_path, "info") {
        eprintln!("warning: file logging init failed ({e}); continuing without a log file");
    }
    eprintln!("stacker: logging to {}", log_path.display());

    let app = App::new()?;

    // ── AI model discovery, GPU availability ──────────────────────────────
    callbacks::models::wire(&app);

    // ── Load persisted settings ───────────────────────────────────────────
    let initial_settings = crate::settings::load();
    // On first run, drop a default (commented) config file so it exists, can be
    // hand-edited, and is usable with the CLI's `--config` switch.
    if let Ok(cfg_path) = crate::settings::get_config_path()
        && !cfg_path.exists()
    {
        if let Some(parent) = cfg_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match std::fs::write(&cfg_path, crate::settings::DEFAULT_CONFIG_TOML) {
            Ok(()) => tracing::info!("wrote default config to {}", cfg_path.display()),
            Err(e) => tracing::warn!("could not write default config: {e}"),
        }
    }
    push_settings_to_ui(&app, &initial_settings);
    // Authoritative copy shared across callbacks.
    let settings_arc: Arc<Mutex<StackingSettings>> = Arc::new(Mutex::new(initial_settings));

    let file_paths: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
    let result_paths: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
    let currently_viewed_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
    let retouch_state: Arc<Mutex<RetouchState>> = Arc::new(Mutex::new(RetouchState::new()));
    let monitor_stop_tx: Arc<Mutex<Option<std::sync::mpsc::Sender<()>>>> =
        Arc::new(Mutex::new(None));
    let relief_preview_ctx: Arc<Mutex<Option<ReliefPreviewContext>>> = Arc::new(Mutex::new(None));
    let aligned_cache: Arc<Mutex<Option<AlignedCache>>> = Arc::new(Mutex::new(None));
    // Cache of the last standalone Sort/Cull button press; consulted (and,
    // on a fingerprint match, applied instead of recomputed) by Stack — see
    // `decide_sort_cull`. Manual add/remove/reorder of `file_paths` does NOT
    // eagerly clear this: the fingerprint check at Stack time is the sole
    // invalidation mechanism (matching `aligned_cache`'s existing design).
    let sort_cull_cache: Arc<Mutex<Option<SortCullCache>>> = Arc::new(Mutex::new(None));
    // Cross-thread popup context for the "Sort/Cull cache is stale"
    // confirmation dialog — see `SortCullPromptContext`.
    let sort_cull_prompt_ctx: Arc<Mutex<Option<SortCullPromptContext>>> =
        Arc::new(Mutex::new(None));
    // Index of the source frame currently shown, so the Show-Aligned toggle can
    // re-render it on the fly.
    let current_view_idx: Arc<Mutex<Option<usize>>> = Arc::new(Mutex::new(None));
    // Set by the Cancel button, polled at checkpoints inside the Align-only
    // and Stack background threads (the per-frame alignment loop, and the
    // boundaries between coarse pipeline stages). Reset to `false` whenever
    // one of those threads starts a new run. A single flag is enough since
    // Align/Stack disable each other and themselves while one is in flight.
    let cancel_requested: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    // Shared handle to the single, lazily-created retouch popup instance —
    // see `callbacks::retouch_window::RetouchWindowSlot`'s doc comment. A
    // plain `Rc` (not `Arc`) is enough: every callback that touches it runs
    // on the Slint UI thread. Passed to `alignment::wire` and `files::wire`
    // too, so the "Show Aligned" toggle and source-file clicks (the two
    // donor-selection entry points in the main window) can refresh the
    // popup's displayed composite live via `retouch_window::refresh_if_visible`.
    let retouch_popup_slot: callbacks::retouch_window::RetouchWindowSlot =
        Rc::new(RefCell::new(None));

    callbacks::files::wire(
        &app,
        &file_paths,
        &result_paths,
        &currently_viewed_path,
        &aligned_cache,
        &current_view_idx,
        &retouch_state,
        &monitor_stop_tx,
        &retouch_popup_slot,
        &sort_cull_cache,
        &sort_cull_prompt_ctx,
        &cancel_requested,
    );

    callbacks::alignment::wire(
        &app,
        &file_paths,
        &settings_arc,
        &aligned_cache,
        &sort_cull_cache,
        &sort_cull_prompt_ctx,
        &current_view_idx,
        &cancel_requested,
        &retouch_state,
        &retouch_popup_slot,
    );

    callbacks::stacking::wire(
        &app,
        &file_paths,
        &result_paths,
        &currently_viewed_path,
        &retouch_state,
        &settings_arc,
        &relief_preview_ctx,
        &aligned_cache,
        &sort_cull_cache,
        &sort_cull_prompt_ctx,
        &cancel_requested,
    );

    // The retouch brush now lives exclusively in the always-on-top
    // `RetouchWindow` popup, opened by right-clicking a result entry — see
    // `callbacks::retouch_window::wire`'s doc comment for the full design
    // (single lazily-created instance, always-on-top via
    // `i-slint-backend-winit`, "leave retouch mode" semantics preserved on
    // close).
    callbacks::retouch_window::wire(&app, &retouch_state, &result_paths, &retouch_popup_slot);

    callbacks::settings_ui::wire(&app, &settings_arc);

    callbacks::saving::wire(
        &app,
        &result_paths,
        &file_paths,
        &currently_viewed_path,
        &settings_arc,
        &retouch_state,
    );

    callbacks::project::wire(
        &app,
        &file_paths,
        &result_paths,
        &settings_arc,
        &retouch_state,
    );

    app.run()?;
    Ok(())
}
