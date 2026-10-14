//! The unsaved-changes guard.
//!
//! Anything that would throw the open project away (New, Open, Open Recent,
//! closing the window) asks [`confirm_discard`] with the code that should run
//! afterwards. A clean project runs it at once. A dirty one shows the
//! Save / Discard / Cancel dialog and parks the continuation:
//!
//! * **Save** writes the project (asking for a path first when it has none) and
//!   runs the continuation only after the write succeeded; a cancelled picker or
//!   a failed write drops it, so nothing is lost by a half-finished save.
//! * **Discard** runs it without saving.
//! * **Cancel**, Escape and a backdrop click drop it.
//!
//! [`install_close_guard`] puts the same question in front of the window's close
//! button: the window stays up while the dialog is open and hides after Save or
//! Discard.

use std::{cell::RefCell, rc::Rc};

use slint::{CloseRequestResponse, ComponentHandle};
use stacker_gui::stall_guard::stall_guard;

use super::{
    Ctx,
    confirm::{self, Handler, Prompt},
    project,
};
use crate::App;

/// The id of the question, so a stale answer to a replaced dialog is ignored.
const ACTION: &str = "unsaved-changes";

/// Run `on_ok` once the open project may be replaced: at once when it has no
/// unsaved changes, else after the user chose Save (and it succeeded) or Discard.
pub fn confirm_discard(ctx: &Ctx, on_ok: impl FnOnce(&App, &Ctx) + 'static) {
    if !project::is_dirty(ctx) {
        if let Some(app) = ctx.ui.upgrade() {
            on_ok(&app, ctx);
        }
        return;
    }
    let message = format!(
        "\"{}\" has changes that are not saved. Save them before continuing?",
        project::display_name()
    );
    let prompt = Prompt {
        action: ACTION,
        heading: "Unsaved changes",
        message: &message,
        primary_label: "Save",
    };
    // Exactly one of the two buttons runs it; the other's handle stays empty.
    let handler: Handler = Box::new(on_ok);
    let parked = Rc::new(RefCell::new(Some(handler)));
    let for_save = Rc::clone(&parked);
    confirm::confirm_with_choice(
        ctx,
        &prompt,
        "Discard",
        move |_app, ctx| {
            let after = for_save.borrow_mut().take();
            project::save_then(ctx, after);
        },
        move |app, ctx| {
            let after = parked.borrow_mut().take();
            if let Some(after) = after {
                after(app, ctx);
            }
        },
    );
}

/// Ask before the window closes: with unsaved changes the close request is
/// refused, the dialog opens, and the window hides after Save or Discard.
pub fn install_close_guard(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.window().on_close_requested(move || {
        stall_guard("unsaved::close-requested", || {
            if !project::is_dirty(&ctx) {
                return CloseRequestResponse::HideWindow;
            }
            confirm_discard(&ctx, |app, _ctx| {
                if let Err(e) = app.hide() {
                    tracing::warn!("could not hide the window: {e}");
                }
            });
            CloseRequestResponse::KeepWindowShown
        })
    });
}
