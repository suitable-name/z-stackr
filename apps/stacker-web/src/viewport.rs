//! Viewport-size tracking: the Slint window follows the browser viewport.
//!
//! Without it Slint's winit backend pins the canvas at the window's preferred size
//! and the responsive layout never reacts.

use std::time::Duration;

use slint::ComponentHandle;
use stacker_gui::App;
use wasm_bindgen::{JsCast, closure::Closure};

/// The smallest window the Slint content tree lays out in (logical px). The app's
/// minimum width is 1120; the height is the smallest the panels stay usable at.
/// Slint does not expose the content tree's minimum before the window exists, so these
/// are constants: a smaller browser window gets this size and the page scrolls.
const MIN_WINDOW_WIDTH: f32 = 1120.0;
const MIN_WINDOW_HEIGHT: f32 = 700.0;

/// The browser viewport in CSS pixels: `documentElement.client*` first (it excludes
/// scrollbar gutters), `window.inner*` second, 1100 x 700 when both read zero
/// (before the first layout, or in an offscreen embedding).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "CSS pixel counts are far below f32's exact range"
)]
fn viewport_logical_size() -> slint::LogicalSize {
    let (mut width, mut height) = (1100.0_f32, 700.0_f32);
    if let Some(window) = web_sys::window() {
        let from_document = window
            .document()
            .and_then(|document| document.document_element())
            .map(|element| {
                (
                    element.client_width() as f32,
                    element.client_height() as f32,
                )
            });
        let from_window = || {
            let inner_width = window.inner_width().ok().and_then(|v| v.as_f64())?;
            let inner_height = window.inner_height().ok().and_then(|v| v.as_f64())?;
            Some((inner_width as f32, inner_height as f32))
        };
        if let Some((w, h)) = from_document.filter(|(w, h)| *w > 0.0 && *h > 0.0) {
            (width, height) = (w, h);
        } else if let Some((w, h)) = from_window().filter(|(w, h)| *w > 0.0 && *h > 0.0) {
            (width, height) = (w, h);
        }
    }
    slint::LogicalSize::new(width.max(MIN_WINDOW_WIDTH), height.max(MIN_WINDOW_HEIGHT))
}

/// Size the Slint window to the browser viewport, now (deferred to the first
/// event-loop turn: the winit window does not exist before `run()`) and on every
/// browser `resize`. The `resize` closure lives for the page's lifetime, so it is
/// leaked with `forget()`.
pub fn track_browser_viewport(app: &App) {
    let initial = app.as_weak();
    slint::Timer::single_shot(Duration::ZERO, move || {
        if let Some(app) = initial.upgrade() {
            app.window().set_size(viewport_logical_size());
        }
    });

    let weak = app.as_weak();
    let on_resize = Closure::<dyn FnMut()>::new(move || {
        if let Some(app) = weak.upgrade() {
            app.window().set_size(viewport_logical_size());
        }
    });
    if let Some(window) = web_sys::window() {
        window.set_onresize(Some(on_resize.as_ref().unchecked_ref()));
    }
    on_resize.forget();
}
