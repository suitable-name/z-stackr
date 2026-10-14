//! The folder monitor, browser build: there is none. A browser page cannot watch
//! a folder, so the menu entry is hidden (`AppModel.web-build`); this stub only
//! keeps the two callbacks answered and the rest of the wiring (`stack`) free of
//! target checks. The desktop implementation is `monitor.rs`.

use stacker_gui::stall_guard::stall_guard;

use super::{
    Ctx,
    feedback::{self, MessageKind},
};
use crate::App;

/// Answer Start Monitor with a notice and Stop Monitor with nothing.
pub fn setup(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_start_monitor(move || {
        stall_guard("monitor::start-monitor", || {
            if let Some(app) = ctx.ui.upgrade() {
                feedback::show(
                    &app,
                    MessageKind::Info,
                    "Live monitoring is not available in the browser: it cannot watch a folder.",
                );
            }
        });
    });
    ui.on_stop_monitor(|| {});
}

/// A stack run the monitor started has ended: the browser never starts one.
pub const fn run_finished(_ctx: &Ctx) {}
