//! The settings dialog: the settings contract, open, apply, cancel, reset,
//! config import/export and the output-folder browser.
//!
//! The contract: a control writes its `SettingsModel`
//! property and fires one `settings-edited()`; [`wire_edited`] reads all
//! properties into a `StackingSettings`, runs `sanitized()`, pushes the clamped
//! values back and hands the result to the persister. The authoritative copy
//! lives in the [`stacker_gui::settings::SettingsPersister`] held by [`Ctx`];
//! its debounced writer takes care of the file.
//!
//! Reset and Load Config change the dialog only (the program, not the user, wrote
//! the values, so no `settings-edited()` fires); they reach the persister when
//! Apply is pressed or the next control is edited. Cancel, Esc and the dialog's
//! cross go back to the settings as they were when the dialog opened.

use std::cell::RefCell;

use slint::ComponentHandle;
use stacker_gui::{
    pickers::{Picked, PickerKind, PickerRequest},
    settings::StackingSettings,
    settings_contract::SettingsValues,
    stall_guard::stall_guard,
};
use stacker_pipeline::config::{Leniency, parse_settings};

use super::{
    Ctx,
    feedback::{self, MessageKind},
    pickers,
};
use crate::{
    App, SettingsModel,
    ui_helpers::{pull_settings_from_ui, push_settings_to_ui, push_settings_values},
};

/// Show the persisted settings in the dialog and register its callbacks.
pub fn setup(ui: &App, ctx: &Ctx) {
    // The dialog carries no defaults of its own: show the persisted copy (and
    // the combo-box lists) before the window can be shown. `main` pushes the same
    // snapshot first; doing it here keeps this module correct on its own.
    push_settings_to_ui(ui, &ctx.settings.snapshot());
    wire_edited(ui, ctx);
    wire_revert(ui, ctx);
    wire_apply(ui, ctx);
    wire_reset_defaults(ui);
    wire_save_config(ui, ctx);
    wire_load_config(ui, ctx);
    wire_browse_output_dir(ui, ctx);
    ctx.start_persist_watch();
}

/// One control changed: read, sanitise, push the clamped values back, persist.
fn wire_edited(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.global::<SettingsModel>().on_settings_edited(move || {
        stall_guard("settings_io::settings-edited", || {
            let Some(app) = ctx.ui.upgrade() else {
                return;
            };
            let settings = pull_settings_from_ui(&app);
            push_settings_to_ui(&app, &settings);
            ctx.settings.update(|current| *current = settings);
            super::project::mark_dirty(&ctx);
        });
    });
}

thread_local! {
    /// The settings as they were when the dialog opened; `None` while it is closed
    /// or once Apply made the edits final.
    static OPENED_WITH: RefCell<Option<StackingSettings>> = const { RefCell::new(None) };
}

/// Opening the dialog remembers the settings; Cancel (also Esc and the cross)
/// goes back to exactly those. An edit takes effect at once (a running preview or
/// the next stack sees it), so Cancel has to undo the edits made since, not only
/// the dialog's display.
fn wire_revert(ui: &App, ctx: &Ctx) {
    OPENED_WITH.with(|slot| *slot.borrow_mut() = None);
    let ctx_open = ctx.clone();
    ui.on_open_settings(move || {
        stall_guard("settings_io::open-settings", || {
            let current = ctx_open.settings.snapshot();
            OPENED_WITH.with(|slot| *slot.borrow_mut() = Some(current.clone()));
            if let Some(app) = ctx_open.ui.upgrade() {
                push_settings_to_ui(&app, &current);
            }
        });
    });
    let ctx_cancel = ctx.clone();
    ui.on_settings_cancel(move || {
        stall_guard("settings_io::settings-cancel", || {
            let opened = OPENED_WITH.with(|slot| slot.borrow_mut().take());
            let restored = opened.unwrap_or_else(|| ctx_cancel.settings.snapshot());
            ctx_cancel
                .settings
                .update(|current| current.clone_from(&restored));
            if let Some(app) = ctx_cancel.ui.upgrade() {
                push_settings_to_ui(&app, &restored);
            }
            super::project::mark_dirty(&ctx_cancel);
        });
    });
}

fn wire_apply(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_settings_apply(move || {
        stall_guard("settings_io::settings-apply", || {
            let Some(app) = ctx.ui.upgrade() else {
                return;
            };
            let applied = pull_settings_from_ui(&app);
            push_settings_to_ui(&app, &applied);
            ctx.settings.update(|settings| *settings = applied);
            OPENED_WITH.with(|slot| *slot.borrow_mut() = None);
            super::project::mark_dirty(&ctx);
            ctx.settings.flush();
            if !ctx.settings.writes_enabled() {
                feedback::notify(
                    &ctx,
                    MessageKind::Warning,
                    "Settings applied for this session only: the settings file cannot be saved.",
                );
            }
            tracing::info!("settings applied");
        });
    });
}

/// Push the core defaults into the dialog (nothing is persisted until Apply).
fn wire_reset_defaults(ui: &App) {
    let weak = ui.as_weak();
    ui.on_settings_reset_defaults(move || {
        stall_guard("settings_io::settings-reset-defaults", || {
            if let Some(app) = weak.upgrade() {
                push_settings_values(&app, &SettingsValues::dialog_reset());
            }
        });
    });
}

/// Export the dialog's values to a chosen file.
fn wire_save_config(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_save_config(move || {
        stall_guard("settings_io::save-config", || {
            let Some(app) = ctx.ui.upgrade() else {
                return;
            };
            let settings = pull_settings_from_ui(&app);
            let request = PickerRequest::new(PickerKind::SaveFile)
                .titled("Save settings as")
                .filter("TOML config", &["toml"])
                .file_name("z-stackr-settings.toml");
            pickers::pick(&ctx, request, move |app, _ctx, picked| {
                let Some(path) = picked.into_iter().next() else {
                    return;
                };
                let platform = crate::platform::current();
                let shown = platform.display_path(&path);
                let (kind, message) = match toml::to_string_pretty(&settings)
                    .map_err(|e| e.to_string())
                    .and_then(|text| {
                        platform
                            .write_bytes(&path, text.as_bytes())
                            .map_err(|e| e.to_string())
                    }) {
                    Ok(()) => (MessageKind::Success, format!("Config saved to {shown}")),
                    Err(e) => (
                        MessageKind::Error,
                        format!("Could not save config to {shown}: {e}"),
                    ),
                };
                feedback::show(app, kind, &message);
            });
        });
    });
}

/// Import a config file into the dialog; the user clicks Apply to keep it.
fn wire_load_config(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_load_config(move || {
        stall_guard("settings_io::load-config", || {
            let request = PickerRequest::new(PickerKind::OpenFile)
                .titled("Load settings")
                .filter("TOML config", &["toml"]);
            pickers::pick(&ctx, request, |app, _ctx, picked| {
                load_picked_config(app, picked);
            });
        });
    });
}

fn load_picked_config(app: &App, picked: Picked) {
    let Some(path) = picked.into_iter().next() else {
        return;
    };
    let platform = crate::platform::current();
    let content = match platform.read_to_string(&path) {
        Ok(content) => content,
        Err(e) => {
            feedback::show(
                app,
                MessageKind::Error,
                &format!("Could not read {}: {e}", platform.display_path(&path)),
            );
            return;
        }
    };
    // The loader the CLI uses (aliases such as `alignment_mode = "bounded"`
    // included); a wrongly typed key keeps its default and is reported.
    match parse_settings(&content, Leniency::Lenient).map(|l| (l.settings, l.warnings)) {
        Ok((settings, warnings)) => {
            push_settings_to_ui(app, &settings);
            if warnings.is_empty() {
                feedback::show(
                    app,
                    MessageKind::Success,
                    "Config loaded (click Apply to keep it)",
                );
            } else {
                feedback::show(
                    app,
                    MessageKind::Warning,
                    &format!(
                        "Config {} loaded with {} warning(s) (click Apply to keep it): {}",
                        feedback::file_label(&path),
                        warnings.len(),
                        warnings.join("; ")
                    ),
                );
            }
        }
        Err(e) => feedback::show(
            app,
            MessageKind::Error,
            &format!("Could not parse {}: {e}", platform.display_path(&path)),
        ),
    }
}

/// Choose the default output directory; the path lands in the dialog field.
fn wire_browse_output_dir(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_browse_output_dir(move || {
        stall_guard("settings_io::browse-output-dir", || {
            let request =
                PickerRequest::new(PickerKind::PickFolder).titled("Default output folder");
            pickers::pick(&ctx, request, |app, _ctx, picked| {
                if let Some(folder) = picked.into_iter().next() {
                    let model = app.global::<SettingsModel>();
                    model.set_default_output_dir(folder.to_string_lossy().as_ref().into());
                    // The user picked it: same path as typing into the field.
                    model.invoke_settings_edited();
                }
            });
        });
    });
}
