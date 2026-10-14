//! The seams the slint-harness UI test (`tests/ui.rs`) needs from the app, compiled
//! only with the `ui-harness` feature.
//!
//! The test is an external crate and drives the real window on Slint's testing
//! backend inside a throw-away profile. This module gives it what `main` has:
//! [`launch`] builds the window and runs [`gui::bootstrap`], and the returned
//! [`Session`] keeps the wiring context alive with the window and stops the
//! settings writer when the window goes. The readers below expose the few
//! values a scenario checks (the lists, the status line, the toast).
//!
//! # Threads
//!
//! The harness restores the process environment when its profile is deleted, so
//! no app thread may be able to touch the profile by then:
//!
//! * the settings writer is flushed and joined when its [`Session`] is dropped;
//! * the run and fetch threads of the job runner are counted
//!   ([`live_workers`]); the teardown waits for zero, since each one owns
//!   scratch files in the profile's temp folder;
//! * the viewer decoders (the sidebar's and the runner's) are joined by
//!   [`stop_workers`];
//! * the folder monitor and the native dialog threads are never started: no
//!   scenario starts the monitor, and the picker answers from the harness
//!   queue (see `platform::native`).
//!
//! Idle threads that remain (the rayon pool) never read the environment or the
//! profile.

use slint::{ComponentHandle, Model as _};

use crate::{
    App, AppModel, DialogModel, ExplainerModel, ExportModel, FilesModel, ManualModel, ProjectModel,
    ShortcutsModel,
    gui::{self, Ctx},
};

/// What keeps a launched window's wiring alive.
///
/// Hand it to the harness with `Attached::keep`: the context (settings writer, state, timers) lives as long as
/// the window, and dropping it flushes and stops the writer thread.
pub struct Session {
    ctx: Ctx,
}

impl Session {
    /// The wiring context of the window.
    #[must_use]
    pub const fn ctx(&self) -> &Ctx {
        &self.ctx
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // The window is gone: write what is pending, then end the writer and the
        // viewer thread so neither can write or read after the profile is deleted.
        self.ctx.settings.stop();
        gui::files::stop_viewer();
    }
}

/// Build the window exactly as `desktop::run_desktop` does (without showing it or
/// running the event loop): a fresh [`App`] and [`gui::bootstrap`]. The harness
/// shows the window itself.
///
/// # Errors
///
/// Slint's platform error when the window cannot be created.
pub fn launch() -> Result<(App, Session), slint::PlatformError> {
    // A window starts without the aligned frames of the one before it.
    crate::runner::current().forget_aligned();
    let app = App::new()?;
    let ctx = gui::bootstrap(&app);
    Ok((app, Session { ctx }))
}

/// How many source frames the sidebar lists.
#[must_use]
pub fn source_count(app: &App) -> usize {
    app.global::<FilesModel>().get_loaded_files().row_count()
}

/// How many results the sidebar lists.
#[must_use]
pub fn result_count(app: &App) -> usize {
    app.global::<FilesModel>().get_result_files().row_count()
}

/// The names of the listed results, oldest first.
#[must_use]
pub fn result_names(app: &App) -> Vec<String> {
    app.global::<FilesModel>()
        .get_result_files()
        .iter()
        .map(|name| name.to_string())
        .collect()
}

/// The labels of the recent-project cards the empty canvas shows, newest first.
#[must_use]
pub fn recent_cards(app: &App) -> Vec<String> {
    app.global::<ProjectModel>()
        .get_recent_cards()
        .iter()
        .map(|label| label.to_string())
        .collect()
}

/// The sidebar's "Aligned frames (N)" line; empty while there is no Align cache.
#[must_use]
pub fn aligned_summary(app: &App) -> String {
    app.global::<FilesModel>().get_aligned_summary().to_string()
}

/// Whether a job (stack, align, sort, cull, monitor) is running.
#[must_use]
pub fn is_busy(app: &App) -> bool {
    app.global::<AppModel>().get_busy()
}

/// The status line.
#[must_use]
pub fn status(app: &App) -> String {
    app.global::<AppModel>().get_status().to_string()
}

/// The toast showing, as `(kind, text)`; the kind is `info`, `success`, `warning`
/// or `error` (see `gui::feedback::MessageKind::name`).
#[must_use]
pub fn notice(app: &App) -> Option<(String, String)> {
    let dialogs = app.global::<DialogModel>();
    dialogs.get_toast_visible().then(|| {
        (
            dialogs.get_toast_kind().to_string(),
            dialogs.get_toast_text().to_string(),
        )
    })
}

/// Hide the toast the way its close button does.
pub fn clear_notice(app: &App) {
    let dialogs = app.global::<DialogModel>();
    if dialogs.get_toast_visible() {
        dialogs.set_toast_visible(false);
        dialogs.invoke_toast_dismissed();
    }
}

/// The modal layers open right now, by name: `settings`, `confirm`, `shortcuts`,
/// `export`, `manual`, `about` and `explainer`.
#[must_use]
pub fn open_modals(app: &App) -> Vec<&'static str> {
    let manual = app.global::<ManualModel>();
    [
        ("settings", app.global::<AppModel>().get_settings_open()),
        ("confirm", app.global::<DialogModel>().get_confirm_open()),
        ("shortcuts", app.global::<ShortcutsModel>().get_open()),
        ("export", app.global::<ExportModel>().get_open()),
        ("manual", manual.get_open()),
        ("about", manual.get_about_open()),
        ("explainer", app.global::<ExplainerModel>().get_open()),
    ]
    .into_iter()
    .filter_map(|(name, open)| open.then_some(name))
    .collect()
}

/// How many run and fetch threads of the job runner are still alive.
#[must_use]
pub fn live_workers() -> usize {
    crate::runner::native::live_workers()
}

/// End the app's idle worker threads and forget the thread-local wiring context.
///
/// The viewer decoders are joined, the settings writer of the last window is
/// already gone with its [`Session`]. Call it after [`live_workers`] reached zero,
/// from `Harness::teardown`.
pub fn stop_workers() {
    gui::files::stop_viewer();
    crate::runner::current().stop_workers();
    Ctx::clear_current();
}
