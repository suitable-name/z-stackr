//! Window layout: the persisted sidebar width, the fit to the monitor, and the one
//! start-up notice that says where the run log is.
//!
//! The width lives in the Slint `ViewportModel` global. Rust pushes the limits and
//! the persisted width at start-up; `ViewportModel.layout-changed()` fires only
//! when the user drags the sidebar handle (a clamp caused by a window resize never
//! fires it), so it needs no "skip the first change" guard: the start-up push is
//! plain property writes. Each user change is saved after a short quiet period,
//! and marks the layout `touched`.

use std::{cell::Cell, rc::Rc, time::Duration};

#[cfg(not(target_arch = "wasm32"))]
use slint::winit_030::WinitWindowAccessor;
use slint::{ComponentHandle, Timer, TimerMode};
use stacker_gui::{
    settings::layout::{self as store, PanelLayout, SIDEBAR_MAX, SIDEBAR_MIN},
    stall_guard::stall_guard,
};

use super::{
    Ctx,
    feedback::{self, MessageKind},
};
use crate::{App, DialogModel, ViewportModel};

/// Quiet time after the last handle movement before `layout.toml` is written.
const SAVE_DEBOUNCE: Duration = Duration::from_millis(300);

/// How long after start-up the log-file notice appears: after the window is up,
/// so the toast is not spent on a window that is not on screen yet.
const LOG_NOTICE_DELAY: Duration = Duration::from_millis(700);

/// Push the layout limits and the saved width into the window and register the
/// save-on-drag callback.
pub fn setup(ui: &App, _ctx: &Ctx) {
    let model = ui.global::<ViewportModel>();
    model.set_sidebar_min(SIDEBAR_MIN);
    model.set_sidebar_max(SIDEBAR_MAX);

    let layout = store::load_stored();
    push(ui, layout);

    let state = Rc::new(Cell::new(layout));
    wire_save(ui, state);
    fit_to_monitor(ui);
    announce_log_file(ui);
}

/// Shrink the window to 94 % × 88 % of the monitor it opened on, never
/// enlarging it. The winit window exists only after the event loop has turned
/// once, hence the zero-length timer. The browser's canvas fills its page, so
/// there is nothing to fit there.
#[cfg(target_arch = "wasm32")]
const fn fit_to_monitor(_ui: &App) {}

/// Shrink the window to 94 % × 88 % of the monitor it opened on, never
/// enlarging it. The winit window exists only after the event loop has turned
/// once, hence the zero-length timer.
#[cfg(not(target_arch = "wasm32"))]
fn fit_to_monitor(ui: &App) {
    let weak = ui.as_weak();
    Timer::single_shot(Duration::ZERO, move || {
        let Some(app) = weak.upgrade() else {
            return;
        };
        let monitor = app
            .window()
            .with_winit_window(|w| {
                w.current_monitor()
                    .or_else(|| w.primary_monitor())
                    .map(|m| {
                        let (size, factor) = (m.size(), m.scale_factor());
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "monitor sizes in logical pixels fit f32 exactly"
                        )]
                        (
                            (f64::from(size.width) / factor) as f32,
                            (f64::from(size.height) / factor) as f32,
                        )
                    })
            })
            .flatten();
        let current = app.window().size().to_logical(app.window().scale_factor());
        if let Some(monitor) = monitor
            && let Some((w, h)) =
                store::fitted_window_size(monitor, (current.width, current.height), (0.94, 0.88))
        {
            app.window().set_size(slint::LogicalSize::new(w, h));
        }
    });
}

/// Write `layout` into the model.
fn push(ui: &App, layout: PanelLayout) {
    ui.global::<ViewportModel>()
        .set_sidebar_width(layout.sidebar_width);
}

/// Save on every user change, after the handle has been quiet for
/// [`SAVE_DEBOUNCE`].
fn wire_save(ui: &App, state: Rc<Cell<PanelLayout>>) {
    let weak = ui.as_weak();
    let timer = Timer::default();
    ui.global::<ViewportModel>().on_layout_changed(move || {
        stall_guard("layout::layout-changed", || {
            let Some(app) = weak.upgrade() else {
                return;
            };
            let model = app.global::<ViewportModel>();
            let layout = state.get().with_user_width(model.get_sidebar_width());
            state.set(layout);
            // Restarting the timer replaces the pending write, so a drag collapses
            // into one write with the final sizes.
            timer.start(TimerMode::SingleShot, SAVE_DEBOUNCE, move || {
                if let Err(e) = store::save_stored(layout) {
                    tracing::warn!("could not save the window layout: {e}");
                }
            });
        });
    });
}

/// Show where the run log is, once, as an Info toast. A toast that is already up
/// (a settings warning, say) is left alone.
fn announce_log_file(ui: &App) {
    let weak = ui.as_weak();
    Timer::single_shot(LOG_NOTICE_DELAY, move || {
        let Some(app) = weak.upgrade() else {
            return;
        };
        if app.global::<DialogModel>().get_toast_visible() {
            return;
        }
        // A host without a log file has nothing to announce.
        if let Some(log) = super::log_path() {
            feedback::show(
                &app,
                MessageKind::Info,
                &format!("Log file: {}", log.display()),
            );
        }
    });
}
