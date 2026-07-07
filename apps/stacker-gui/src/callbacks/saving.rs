use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use crate::{
    App, retouch::RetouchState, save::perform_save_current_image, settings::StackingSettings,
};

/// Wires the save-related callbacks.
///
/// Covers saving the currently-viewed image (with the configured default
/// format), saving it as an explicit format, and saving a specific
/// result-list entry as an explicit format (the context-menu shortcuts).
///
/// # Panics
///
/// The registered callbacks call `.lock().unwrap()` on the shared
/// `Mutex`-guarded state (`result_paths`, etc.); those panic only if
/// another thread already panicked while holding the same lock (mutex
/// poisoning), which does not happen in normal operation.
pub fn wire(
    app: &App,
    result_paths: &Arc<Mutex<Vec<PathBuf>>>,
    file_paths: &Arc<Mutex<Vec<PathBuf>>>,
    currently_viewed_path: &Arc<Mutex<Option<PathBuf>>>,
    settings_arc: &Arc<Mutex<StackingSettings>>,
    retouch_state: &Arc<Mutex<RetouchState>>,
) {
    // ── Save result as a specific format (context-menu shortcut) ─────────
    {
        let results_arc = Arc::clone(result_paths);
        let file_paths_s = Arc::clone(file_paths);
        let settings_arc_s = Arc::clone(settings_arc);
        let retouch_arc_s = Arc::clone(retouch_state);

        app.on_save_result_as(move |idx, fmt| {
            let idx = idx as usize;
            let result_path = {
                let p = results_arc.lock().unwrap();
                if idx < p.len() {
                    Some(p[idx].clone())
                } else {
                    None
                }
            };
            if let Some(path) = result_path {
                let forced = match fmt.as_str() {
                    "png" => crate::settings::OutputFormat::Png,
                    "jpg" | "jpeg" => crate::settings::OutputFormat::Jpeg,
                    "tiff" | "tif" => crate::settings::OutputFormat::Tiff,
                    other => {
                        tracing::warn!("unknown save-as format {other:?}; ignoring");
                        return;
                    }
                };
                let viewed = Arc::new(std::sync::Mutex::new(Some(path)));
                perform_save_current_image(
                    &viewed,
                    &file_paths_s,
                    &settings_arc_s,
                    &retouch_arc_s,
                    Some(forced),
                );
            }
        });
    }

    // ── Save viewed image ────────────────────────────────────────────────
    {
        let viewed = Arc::clone(currently_viewed_path);
        let file_paths_s = Arc::clone(file_paths);
        let settings_arc_s = Arc::clone(settings_arc);
        let retouch_arc_s = Arc::clone(retouch_state);

        app.on_save_current_image(move || {
            perform_save_current_image(
                &viewed,
                &file_paths_s,
                &settings_arc_s,
                &retouch_arc_s,
                None,
            );
        });
    }

    // ── Save viewed image as a specific format (context-menu shortcut) ────
    {
        let viewed = Arc::clone(currently_viewed_path);
        let file_paths_s = Arc::clone(file_paths);
        let settings_arc_s = Arc::clone(settings_arc);
        let retouch_arc_s = Arc::clone(retouch_state);

        app.on_save_current_image_as(move |fmt| {
            let forced = match fmt.as_str() {
                "png" => crate::settings::OutputFormat::Png,
                "jpg" | "jpeg" => crate::settings::OutputFormat::Jpeg,
                "tiff" | "tif" => crate::settings::OutputFormat::Tiff,
                other => {
                    tracing::warn!("unknown save-as format {other:?}; ignoring");
                    return;
                }
            };
            perform_save_current_image(
                &viewed,
                &file_paths_s,
                &settings_arc_s,
                &retouch_arc_s,
                Some(forced),
            );
        });
    }
}
