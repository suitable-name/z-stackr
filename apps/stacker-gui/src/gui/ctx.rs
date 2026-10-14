//! The shared context every callback holds, and the helpers workers use to
//! reach the UI thread.
//!
//! # Threading rules
//!
//! * [`Ctx`] is `!Send` (it holds an `Rc<RefCell<AppState>>`): it lives on the UI
//!   thread and callbacks clone it freely.
//! * Never hold a `RefCell` borrow across anything that can pump the event loop
//!   or call back into a handler (`pickers::pick`, `invoke_*` of a callback):
//!   take a short `borrow`/`borrow_mut`, copy what is needed out, drop the
//!   borrow, then push to the UI.
//! * Workers capture only `Send` data (a `Weak<App>`, paths, settings, finished
//!   images). They reach the UI thread with [`on_ui`], and the state with
//!   [`on_ui_ctx`], which looks the context up in a thread-local installed at
//!   start-up. They never touch the state themselves. The heavy jobs do not use
//!   these helpers: the [`crate::runner`] delivers their events on the UI thread.

use std::{cell::RefCell, rc::Rc, sync::Arc, time::Duration};

use slint::{ComponentHandle, Timer, TimerMode, Weak};
use stacker_gui::{settings::SettingsPersister, state::AppState};

use crate::App;

/// How often the UI checks whether the settings writer has failed.
const PERSIST_WATCH_INTERVAL: Duration = Duration::from_secs(2);

/// The timers the shell keeps alive.
#[derive(Default)]
pub struct Timers {
    /// Surfaces failed settings writes in the status line.
    pub persist_watch: Timer,
}

/// What every callback holds: the window (weakly), the state, the settings
/// writer, the timers.
#[derive(Clone)]
pub struct Ctx {
    /// The window; `upgrade()` fails only once it is going away.
    pub ui: Weak<App>,
    /// The UI-thread state.
    pub state: Rc<RefCell<AppState>>,
    /// Shared timers.
    pub timers: Rc<Timers>,
    /// The settings copy and its background writer.
    pub settings: Arc<SettingsPersister>,
}

thread_local! {
    static CURRENT: RefCell<Option<Ctx>> = const { RefCell::new(None) };
}

impl Ctx {
    /// A fresh context for `app` around the settings writer `settings`.
    #[must_use]
    pub fn new(app: &App, settings: Arc<SettingsPersister>) -> Self {
        Self {
            ui: app.as_weak(),
            state: Rc::new(RefCell::new(AppState::default())),
            timers: Rc::new(Timers::default()),
            settings,
        }
    }

    /// Register this context as the calling thread's ambient one, so
    /// [`on_ui_ctx`] can hand it to closures a worker posts to the event loop.
    pub fn install_for_this_thread(&self) {
        CURRENT.with(|slot| *slot.borrow_mut() = Some(self.clone()));
    }

    /// Forget the context [`Self::install_for_this_thread`] registered, so the
    /// settings writer it holds can go. For the UI harness's teardown.
    #[cfg(feature = "ui-harness")]
    pub fn clear_current() {
        CURRENT.with(|slot| *slot.borrow_mut() = None);
    }

    /// The context [`Self::install_for_this_thread`] registered on this thread.
    #[must_use]
    pub fn current() -> Option<Self> {
        CURRENT.with(|slot| slot.borrow().clone())
    }

    /// Start the timer that reports failed settings writes in the status line.
    /// It captures the writer and a weak window handle, never the context, so it
    /// forms no reference cycle.
    pub fn start_persist_watch(&self) {
        let settings = Arc::clone(&self.settings);
        let ui = self.ui.clone();
        self.timers
            .persist_watch
            .start(TimerMode::Repeated, PERSIST_WATCH_INTERVAL, move || {
                if let Some(message) = settings.take_error()
                    && let Some(app) = ui.upgrade()
                {
                    crate::gui::feedback::show(
                        &app,
                        crate::gui::feedback::MessageKind::Error,
                        &message,
                    );
                }
            });
    }
}

/// Run `f` on the UI thread with the window, from any thread. Silently does
/// nothing once the window is gone.
pub fn on_ui(ui: &Weak<App>, f: impl FnOnce(&App) + Send + 'static) {
    let _ = ui.upgrade_in_event_loop(move |app| f(&app));
}

/// Run `f` on the UI thread with the window and the installed [`Ctx`], from any
/// thread. Use this to store a finished result in the state.
pub fn on_ui_ctx(ui: &Weak<App>, f: impl FnOnce(&App, &Ctx) + Send + 'static) {
    let _ = ui.upgrade_in_event_loop(move |app| {
        if let Some(ctx) = Ctx::current() {
            f(&app, &ctx);
        }
    });
}
