//! Leaving the page: the unsaved-changes guard and the final settings flush.
//!
//! * `beforeunload` makes the browser ask before the tab closes or reloads while the
//!   project has unsaved changes (the window's own close guard never fires in a
//!   browser). The project's dirty flag is the one the title bar shows,
//!   `ProjectModel.dirty`.
//! * `pagehide` (and a tab that turns hidden, which on phones is the only warning
//!   before the page is discarded) writes the settings still inside the debounce
//!   window to `localStorage`.

use slint::ComponentHandle;
use stacker_gui::{ProjectModel, gui::Ctx};
use wasm_bindgen::{JsCast, closure::Closure};

use crate::diagnostics::console_warn;

/// Whether the open project has changes no save has captured.
fn has_unsaved_changes(ctx: &Ctx) -> bool {
    ctx.ui
        .upgrade()
        .is_some_and(|app| app.global::<ProjectModel>().get_dirty())
}

/// Install the `beforeunload` guard and the `pagehide` / `visibilitychange` flush,
/// once, for the page's lifetime (the closures are leaked with `forget()`).
pub fn install(ctx: &Ctx) {
    let Some(window) = web_sys::window() else {
        return;
    };

    let for_unload = ctx.clone();
    let on_unload = Closure::<dyn FnMut(web_sys::Event)>::new(move |event: web_sys::Event| {
        if !has_unsaved_changes(&for_unload) {
            return;
        }
        // Standard browsers ask on `preventDefault`; older ones need a `returnValue`.
        event.prevent_default();
        if let Some(event) = event.dyn_ref::<web_sys::BeforeUnloadEvent>() {
            event.set_return_value("This project has unsaved changes.");
        }
    });

    let for_hide = ctx.clone();
    let on_hide = Closure::<dyn FnMut(web_sys::Event)>::new(move |_event: web_sys::Event| {
        for_hide.settings.flush();
    });

    let for_visibility = ctx.clone();
    let on_visibility = Closure::<dyn FnMut(web_sys::Event)>::new(move |_event: web_sys::Event| {
        let hidden = web_sys::window()
            .and_then(|window| window.document())
            .is_some_and(|document| document.hidden());
        if hidden {
            for_visibility.settings.flush();
        }
    });

    let installed = window
        .add_event_listener_with_callback("beforeunload", on_unload.as_ref().unchecked_ref())
        .and_then(|()| {
            window.add_event_listener_with_callback("pagehide", on_hide.as_ref().unchecked_ref())
        })
        .and_then(|()| {
            window.document().map_or(Ok(()), |document| {
                document.add_event_listener_with_callback(
                    "visibilitychange",
                    on_visibility.as_ref().unchecked_ref(),
                )
            })
        });
    if installed.is_err() {
        console_warn("could not install the page-leave handlers");
    }
    on_unload.forget();
    on_hide.forget();
    on_visibility.forget();
}
