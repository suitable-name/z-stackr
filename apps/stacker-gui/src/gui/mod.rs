//! The GUI wiring: one module per feature, each with a `setup(ui, ctx)` function
//! that registers that feature's callbacks.
//!
//! [`wire_all`] calls them in order; [`bootstrap`] is the whole start-up after the
//! window exists, shared by the desktop `main` and the browser entry point.
//!
//! What touches the host (dialogs, files, the config store, the log location) goes
//! through [`crate::platform`]; every heavy job (stack, align, the viewer's and
//! the retouch tool's frame fetches) goes through [`crate::runner`].

pub(crate) mod activity;
pub(crate) mod align;
pub mod confirm;
pub mod ctx;
pub mod dropped;
pub(crate) mod explainer;
pub(crate) mod export;
pub mod feedback;
pub(crate) mod files;
pub(crate) mod help;
pub(crate) mod jobs;
pub(crate) mod layout;
pub(crate) mod relief_preview;
// The folder monitor needs `notify` and a thread; the browser gets a stub.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod monitor;
#[cfg(target_arch = "wasm32")]
#[path = "monitor_web.rs"]
pub(crate) mod monitor;
pub(crate) mod pickers;
pub(crate) mod preview;
pub(crate) mod project;
pub(crate) mod retouch;
pub(crate) mod settings_io;
pub(crate) mod shortcuts;
pub(crate) mod sort_cull;
pub mod stack;
pub(crate) mod stale;
pub(crate) mod unsaved;
pub(crate) mod viewport;

use std::path::PathBuf;

use slint::ComponentHandle;

pub use ctx::Ctx;
pub use dropped::handle_paths;
pub use stack::populate_algorithms;

use crate::{App, AppModel, settings, ui_helpers::push_settings_to_ui};

/// Where the run log is written, when the host keeps one (`None` in the
/// browser).
#[must_use]
pub fn log_path() -> Option<PathBuf> {
    crate::platform::current().log_path()
}

/// Everything after `App::new()`: fill the algorithm list, load the persisted
/// settings, create the [`Ctx`] and wire every feature.
///
/// A settings file we cannot parse is moved aside and the user is told; nothing
/// overwrites it. The platform must be installed already (the desktop installs
/// its own on first use).
///
/// The caller then shows the window and runs the event loop, and calls
/// `ctx.settings.flush()` when the window goes away.
#[must_use]
pub fn bootstrap(app: &App) -> Ctx {
    stack::populate_algorithms(app);
    let opened = settings::open_default();
    push_settings_to_ui(app, &opened.persister.snapshot());
    let ctx = Ctx::new(app, opened.persister);
    settings::SettingsPersister::install_for_this_thread(&ctx.settings);
    if let Some(warning) = opened.outcome.user_message() {
        feedback::show(app, feedback::MessageKind::Warning, &warning);
    }
    wire_all(app, &ctx);
    ctx
}

/// Register every feature's callbacks on `app`.
pub fn wire_all(app: &App, ctx: &Ctx) {
    ctx.install_for_this_thread();
    // The browser build hides what a browser cannot do (see `ui/models/app_model.slint`).
    app.global::<AppModel>()
        .set_web_build(crate::platform::current().is_web());
    // Before every feature that can put pixels on the canvas.
    preview::setup(app, ctx);
    files::setup(app, ctx);
    // The registry must exist before any feature that starts a job or hooks it.
    activity::setup(app, ctx);
    align::setup(app, ctx);
    stack::setup(app, ctx);
    sort_cull::setup(app, ctx);
    relief_preview::setup(app, ctx);
    monitor::setup(app, ctx);
    retouch::setup(app, ctx);
    viewport::setup(app, ctx);
    layout::setup(app, ctx);
    settings_io::setup(app, ctx);
    export::setup(app, ctx);
    feedback::setup(app, ctx);
    confirm::setup(app, ctx);
    shortcuts::setup(app, ctx);
    stale::setup(app, ctx);
    help::setup(app, ctx);
    explainer::setup(app, ctx);
    dropped::setup(app, ctx);
    // Last: it reads the state and the settings every feature above has set up.
    project::setup(app, ctx);
}
