//! The GUI side of the off-thread pickers: [`stacker_gui::pickers`] does the
//! dialog work; this wrapper hands the continuation the window and the context.

use stacker_gui::pickers::{self as dialogs, Picked, PickerRequest};

use super::Ctx;
use crate::App;

/// Show `request`'s dialog on a background thread and run `on_done` on the UI
/// thread with the window, the context and the answer (an empty list when the
/// dialog was cancelled). `on_done` is skipped when the window has closed in the
/// meantime.
///
/// Call this from a callback with no `RefCell` borrow outstanding: pick first,
/// borrow inside `on_done`.
pub fn pick(ctx: &Ctx, request: PickerRequest, on_done: impl FnOnce(&App, &Ctx, Picked) + 'static) {
    let ctx = ctx.clone();
    dialogs::pick(request, move |picked| {
        if let Some(app) = ctx.ui.upgrade() {
            on_done(&app, &ctx, picked);
        }
    });
}
