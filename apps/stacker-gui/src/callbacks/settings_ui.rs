use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, SharedString};

use crate::{
    App,
    settings::StackingSettings,
    ui_helpers::{pull_settings_from_ui, push_settings_to_ui},
};

/// Wires the Settings dialog callbacks: open (snapshot push), Apply, Save
/// Config / Load Config (file-based import/export), Cancel, Reset to
/// defaults, and the output-directory browse button.
///
/// # Panics
///
/// The registered callbacks call `.lock().unwrap()` on `settings_arc`;
/// this panics only if another thread already panicked while holding that
/// lock (mutex poisoning), which does not happen in normal operation.
// `too_many_lines`: registers every Settings-dialog callback (open, apply,
// save/load config, cancel, reset, browse) in one place so the dialog's
// full wiring is visible at one call site; each `on_*` closure is
// independent and short.
#[allow(clippy::too_many_lines)]
pub fn wire(app: &App, settings_arc: &Arc<Mutex<StackingSettings>>) {
    let app_weak = app.as_weak();

    // ── Settings: open (snapshot last-saved into dialog) ─────────────────
    {
        // open-settings is a no-op in Rust; the dialog visibility is toggled
        // entirely in Slint.  We keep the callback in case the caller needs
        // to push fresh values before the dialog renders.
        let app_weak = app_weak.clone();
        let settings_arc = Arc::clone(settings_arc);
        app.on_open_settings(move || {
            // Re-push the last-saved settings so Cancel always reverts to them.
            if let Some(app) = app_weak.upgrade() {
                let s = settings_arc.lock().unwrap().clone();
                push_settings_to_ui(&app, &s);
            }
        });
    }

    // ── Settings: Apply ───────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let settings_arc = Arc::clone(settings_arc);
        app.on_settings_apply(move || {
            if let Some(app) = app_weak.upgrade() {
                let new_settings = pull_settings_from_ui(&app);
                // Persist to disk (log on error, never panic).
                if let Err(e) = crate::settings::save(&new_settings) {
                    tracing::error!("settings save failed: {e}");
                }
                *settings_arc.lock().unwrap() = new_settings;
                tracing::info!("settings applied and saved");
            }
        });
    }

    // ── Settings: Save Config (export the dialog's values to a chosen file) ──
    {
        let app_weak = app_weak.clone();
        app.on_save_config(move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };
            let settings = pull_settings_from_ui(&app);

            // Default the dialog to the app's config directory (creating it
            // if necessary) so Save/Load Config start where `settings.toml`
            // already lives, instead of wherever the OS last remembered.
            let mut dialog = rfd::FileDialog::new()
                .add_filter("TOML config", &["toml"])
                .set_file_name("z-stackr-settings.toml");
            if let Ok(cfg_path) = crate::settings::get_config_path()
                && let Some(dir) = cfg_path.parent()
            {
                let _ = std::fs::create_dir_all(dir);
                if dir.is_dir() {
                    dialog = dialog.set_directory(dir);
                }
            }
            let Some(path) = dialog.save_file() else {
                return; // dialog cancelled
            };
            match toml::to_string_pretty(&settings) {
                Ok(content) => match std::fs::write(&path, content) {
                    Ok(()) => tracing::info!("config saved to {}", path.display()),
                    Err(e) => tracing::error!("save config to {}: {e}", path.display()),
                },
                Err(e) => tracing::error!("serialise config: {e}"),
            }
        });
    }

    // ── Settings: Load Config (import a config file into the dialog) ─────────
    {
        let app_weak = app_weak.clone();
        app.on_load_config(move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };

            // Default the dialog to the app's config directory (creating it
            // if necessary) so Save/Load Config start where `settings.toml`
            // already lives, instead of wherever the OS last remembered.
            let mut dialog = rfd::FileDialog::new().add_filter("TOML config", &["toml"]);
            if let Ok(cfg_path) = crate::settings::get_config_path()
                && let Some(dir) = cfg_path.parent()
            {
                let _ = std::fs::create_dir_all(dir);
                if dir.is_dir() {
                    dialog = dialog.set_directory(dir);
                }
            }
            let Some(path) = dialog.pick_file() else {
                return; // dialog cancelled
            };
            match std::fs::read_to_string(&path) {
                Ok(content) => match toml::from_str::<StackingSettings>(&content) {
                    Ok(mut s) => {
                        s.clamp_valid();
                        s.preprocessing.clamp_valid();
                        s.image_saving.clamp_valid();
                        // Populate the dialog; the user clicks Apply to keep it.
                        push_settings_to_ui(&app, &s);
                        tracing::info!(
                            "config loaded from {} (click Apply to keep it)",
                            path.display()
                        );
                    }
                    Err(e) => tracing::error!("parse config {}: {e}", path.display()),
                },
                Err(e) => tracing::error!("read config {}: {e}", path.display()),
            }
        });
    }

    // ── Settings: Cancel ─────────────────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        let settings_arc = Arc::clone(settings_arc);
        app.on_settings_cancel(move || {
            // Revert dialog properties to the last-saved authoritative copy.
            if let Some(app) = app_weak.upgrade() {
                let s = settings_arc.lock().unwrap().clone();
                push_settings_to_ui(&app, &s);
            }
        });
    }

    // ── Settings: Reset to defaults ──────────────────────────────────────
    {
        let app_weak = app_weak.clone();
        app.on_settings_reset_defaults(move || {
            // Push defaults into the dialog (do NOT persist until Apply).
            if let Some(app) = app_weak.upgrade() {
                push_settings_to_ui(&app, &StackingSettings::default());
            }
        });
    }

    // ── Settings: Browse output directory ────────────────────────────────
    {
        app.on_browse_output_dir(move || {
            // Open a native folder picker; write the chosen path into the
            // `set-default-output-dir` property.  Never panics — all failure
            // cases (dialog cancelled, no selection) are silently ignored.
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                let path_str = folder.to_string_lossy().into_owned();
                if let Some(app) = app_weak.upgrade() {
                    app.set_set_default_output_dir(SharedString::from(path_str.as_str()));
                }
            }
        });
    }
}
