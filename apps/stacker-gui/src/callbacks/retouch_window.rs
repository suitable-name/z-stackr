use std::{
    cell::RefCell,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
};

use slint::{ComponentHandle, Image};
// Brings `WinitWindowAccessor::with_winit_window` into scope on
// `slint::Window` — the only way to reach the underlying
// `winit::window::Window` and its `set_window_level` call, since Slint's own
// `Window` API has no cross-platform "always on top" method. See the
// `i-slint-backend-winit` dependency comment in `Cargo.toml`.
use i_slint_backend_winit::WinitWindowAccessor;

use crate::{App, RetouchWindow, retouch::RetouchState};

/// Lazily-created, reused handle to the single retouch popup instance.
///
/// The popup is never constructed until the first right-click on a result
/// entry; after that, every subsequent "Retouch…" click reuses the same
/// window (moving it to front and refreshing its image) rather than
/// constructing a new one — a fresh `RetouchWindow` per click would leak
/// windows and desync which one is actually receiving brush strokes.
///
/// Shared (as a plain `Rc`, not `Arc`) with `callbacks::alignment::wire` and
/// `callbacks::files::wire` so that donor-selection changes made in the main
/// window (the "Show Aligned" toggle, and clicking a different source
/// frame) can refresh the popup's displayed composite live — see
/// [`refresh_if_visible`]. A plain `Rc` is safe here because every one of
/// these callbacks only ever runs on the Slint UI thread (event-loop
/// callbacks and `slint::invoke_from_event_loop` closures), never from a
/// background `std::thread::spawn` worker directly.
pub type RetouchWindowSlot = Rc<RefCell<Option<RetouchWindow>>>;

/// Refreshes the popup's displayed composite from the current
/// `RetouchState`, if the popup exists and a session is active.
///
/// A no-op when the popup has never been created, or when there is no
/// active retouch session (nothing to refresh). Called from
/// `callbacks::alignment::wire`'s "Show Aligned" toggle handler and
/// `callbacks::files::wire`'s source-file click handler — the two places
/// donor selection can change in the main window — so the popup's composite
/// stays in sync with the rest of the app instead of only updating on the
/// next brush stroke.
///
/// # Panics
///
/// Calls `.lock().unwrap()` on `retouch_state`; panics only if another
/// thread already panicked while holding that lock (mutex poisoning), which
/// does not happen in normal operation.
pub fn refresh_if_visible(
    popup_slot: &RetouchWindowSlot,
    retouch_state: &Arc<Mutex<RetouchState>>,
) {
    let Some(window) = popup_slot
        .borrow()
        .as_ref()
        .map(RetouchWindow::clone_strong)
    else {
        return;
    };
    let rs = retouch_state.lock().unwrap();
    let Some(session) = rs.session.as_ref() else {
        return;
    };
    let composite = session.render_composite();
    let alpha = rs.show_painted.then_some(session.alpha.luma.as_slice());
    let buf = crate::image_utils::planar_to_rgba_buffer_with_overlay(&composite, alpha);
    drop(rs);
    window.set_has_images(true);
    window.set_retouch_image(Image::from_rgba8(buf));
}

/// Wires the always-on-top retouch popup.
///
/// Covers opening it from a result's right-click context menu, its own
/// brush/undo/redo/show-painted callbacks (previously wired directly
/// against the main `App` — see the removed `callbacks::retouch::wire`),
/// and closing it.
///
/// ## Always-on-top
///
/// Slint 1.17's `Window` API has no cross-platform "always on top" property,
/// so it is applied imperatively right after the popup is (lazily)
/// constructed, via `i_slint_backend_winit::WinitWindowAccessor::
/// with_winit_window` + `winit::window::Window::set_window_level
/// (WindowLevel::AlwaysOnTop)`. This only takes effect once the winit
/// backend has actually created its native window, which is not guaranteed
/// to have happened yet immediately after `RetouchWindow::new()` returns —
/// so it is (re-)applied both right after construction and every time the
/// popup is shown, since `with_winit_window` silently no-ops (returns
/// `None`) if the native window doesn't exist yet.
///
/// ## Closing / "leave retouch mode"
///
/// Closing the popup (titlebar close button, Alt+F4, etc.) preserves the
/// previous "leave retouch mode" semantics: before this restructuring, the
/// brush simply stopped being reachable once the user navigated away (there
/// was no explicit session teardown tied to leaving retouch mode — the
/// brush's own `show-aligned`-gated `TouchArea` just stopped receiving
/// input). The popup's close handler preserves that exactly: it does not
/// clear `RetouchState.session`/`history` (a later right-click on the same
/// result must still see the same in-progress brush strokes), it only hides
/// the window instead of destroying it, so state and the single-instance
/// invariant both survive across close/reopen.
///
/// # Panics
///
/// The registered callback calls `.lock().unwrap()` on `retouch_state` and
/// `result_paths`; those panic only if another thread already panicked
/// while holding the same lock (mutex poisoning), which does not happen in
/// normal operation.
pub fn wire(
    app: &App,
    retouch_state: &Arc<Mutex<RetouchState>>,
    result_paths: &Arc<Mutex<Vec<PathBuf>>>,
    popup_slot: &RetouchWindowSlot,
) {
    // ── Right-click "Retouch…" → open (or refresh + raise) the popup ──────
    {
        let retouch_arc = Arc::clone(retouch_state);
        let results_arc = Arc::clone(result_paths);
        let popup_slot = Rc::clone(popup_slot);

        app.on_retouch_result(move |idx| {
            let idx = idx as usize;
            let Some(path) = results_arc.lock().unwrap().get(idx).cloned() else {
                return;
            };

            let window = get_or_create_popup(&popup_slot, &retouch_arc);

            // Render whatever composite is currently available: the live
            // session's composite when one exists for this exact result,
            // otherwise just the saved image on disk (browse-only — the
            // brush is a no-op with no active session, same as the old
            // in-window canvas was whenever `RetouchState.session` was
            // `None`).
            let (buf, can_undo, can_redo) = {
                let rs = retouch_arc.lock().unwrap();
                rs.session.as_ref().map_or((None, false, false), |session| {
                    let composite = session.render_composite();
                    let alpha = rs.show_painted.then_some(session.alpha.luma.as_slice());
                    let buf =
                        crate::image_utils::planar_to_rgba_buffer_with_overlay(&composite, alpha);
                    (Some(buf), rs.history.can_undo(), rs.history.can_redo())
                })
            };

            if let Some(buf) = buf {
                window.set_has_images(true);
                window.set_retouch_image(Image::from_rgba8(buf));
            } else if let Ok(img) = Image::load_from_path(&path) {
                window.set_has_images(true);
                window.set_retouch_image(img);
            }
            window.set_can_undo(can_undo);
            window.set_can_redo(can_redo);

            apply_always_on_top(&window);
            let _ = window.show();
        });
    }
}

/// Returns the existing popup instance, or constructs it (and wires its own
/// brush/undo/redo/show-painted/close callbacks) the first time it's
/// needed. See [`wire`]'s doc comment for why exactly one instance is kept
/// alive across every right-click rather than a fresh window per click.
// `too_many_lines`: constructs the popup once and registers all of its own
// callbacks (brush/undo/redo/show-painted/close) in that same construction
// step, so the "lazily create, then wire everything" invariant is visible
// in one place rather than split across helper functions.
#[allow(clippy::too_many_lines)]
fn get_or_create_popup(
    popup_slot: &RetouchWindowSlot,
    retouch_state: &Arc<Mutex<RetouchState>>,
) -> RetouchWindow {
    if let Some(existing) = popup_slot.borrow().as_ref() {
        return existing.clone_strong();
    }

    let window = RetouchWindow::new().expect("failed to construct RetouchWindow");

    // ── Brush stroke ─────────────────────────────────────────────────────
    {
        let window_weak = window.as_weak();
        let retouch_arc = Arc::clone(retouch_state);

        window.on_apply_brush(move |x, y, radius, opacity| {
            let px = x.max(0) as usize;
            let py = y.max(0) as usize;

            let Some((composite_buf, can_undo, can_redo)) = retouch_arc
                .lock()
                .unwrap()
                .apply_and_snapshot(px, py, radius, opacity)
            else {
                return;
            };

            let _ = slint::invoke_from_event_loop({
                let w = window_weak.clone();
                move || {
                    if let Some(window) = w.upgrade() {
                        window.set_retouch_image(Image::from_rgba8(composite_buf));
                        window.set_can_undo(can_undo);
                        window.set_can_redo(can_redo);
                    }
                }
            });
        });
    }

    // ── Undo ─────────────────────────────────────────────────────────────
    {
        let window_weak = window.as_weak();
        let retouch_arc = Arc::clone(retouch_state);

        window.on_undo_stroke(move || {
            let Some((composite_buf, can_undo, can_redo)) =
                retouch_arc.lock().unwrap().apply_undo()
            else {
                return;
            };

            let _ = slint::invoke_from_event_loop({
                let w = window_weak.clone();
                move || {
                    if let Some(window) = w.upgrade() {
                        window.set_retouch_image(Image::from_rgba8(composite_buf));
                        window.set_can_undo(can_undo);
                        window.set_can_redo(can_redo);
                    }
                }
            });
        });
    }

    // ── Redo ─────────────────────────────────────────────────────────────
    {
        let window_weak = window.as_weak();
        let retouch_arc = Arc::clone(retouch_state);

        window.on_redo_stroke(move || {
            let Some((composite_buf, can_undo, can_redo)) =
                retouch_arc.lock().unwrap().apply_redo()
            else {
                return;
            };

            let _ = slint::invoke_from_event_loop({
                let w = window_weak.clone();
                move || {
                    if let Some(window) = w.upgrade() {
                        window.set_retouch_image(Image::from_rgba8(composite_buf));
                        window.set_can_undo(can_undo);
                        window.set_can_redo(can_redo);
                    }
                }
            });
        });
    }

    // ── Show painted area toggle ──────────────────────────────────────────
    // Display-only, same contract as the old in-window toggle: never
    // touches the saved image (Save always re-renders straight from
    // `RetouchSession::render_composite`, which knows nothing about this
    // toggle — see `crate::save::perform_save_current_image`).
    {
        let window_weak = window.as_weak();
        let retouch_arc = Arc::clone(retouch_state);

        window.on_show_painted_toggled(move |show_painted| {
            let Some(composite_buf) = retouch_arc.lock().unwrap().set_show_painted(show_painted)
            else {
                return;
            };

            let _ = slint::invoke_from_event_loop({
                let w = window_weak.clone();
                move || {
                    if let Some(window) = w.upgrade() {
                        window.set_retouch_image(Image::from_rgba8(composite_buf));
                    }
                }
            });
        });
    }

    // ── Close → hide instead of destroy ───────────────────────────────────
    // Preserves the previous "leave retouch mode" semantics: no session
    // teardown, just stop showing the popup, so a later right-click on the
    // same (or a different) result reuses this exact instance instead of
    // constructing a new one. Slint's generated `Window::window()` exposes
    // the shared `slint::Window` handle `on_close_requested` is defined on.
    {
        let window_weak = window.as_weak();
        window.window().on_close_requested(move || {
            if let Some(window) = window_weak.upgrade() {
                window.invoke_closed();
                let _ = window.hide();
            }
            slint::CloseRequestResponse::KeepWindowShown
        });
    }

    apply_always_on_top(&window);

    *popup_slot.borrow_mut() = Some(window.clone_strong());
    window
}

/// Applies always-on-top to the popup via the winit backend. See [`wire`]'s
/// doc comment for why this can't be a declarative `.slint` property in
/// Slint 1.17, and why this is called both right after construction and
/// every time the popup is (re-)shown.
fn apply_always_on_top(window: &RetouchWindow) {
    window.window().with_winit_window(|winit_window| {
        winit_window
            .set_window_level(i_slint_backend_winit::winit::window::WindowLevel::AlwaysOnTop);
    });
}
