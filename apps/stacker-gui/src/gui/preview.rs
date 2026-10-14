//! The one way pixels reach the canvas from anywhere but a click handler.
//!
//! Previews and repaints come from many threads at a rate the UI cannot follow:
//! the retouch preview worker, the live thumbnails of an Align or AI pass, and the
//! decoder that serves list clicks. They all go through [`submit_frame`]: a
//! [`RedrawGate`] keeps at most ONE closure in the event queue, and that closure
//! always shows the newest buffer submitted, so a burst costs one repaint and the
//! buffers it skipped are dropped by the producer.
//!
//! Producers that build frames repeatedly (the live previews) write them into a
//! [`stacker_gui::preview::FrameRing`] of rotating buffers rather than allocating
//! one per frame; [`LivePreview`](stacker_gui::preview::LivePreview) bundles the
//! throttle, the ring and the function below:
//!
//! ```ignore
//! let mut live = LivePreview::new(preview::submit_frame);
//! // in the per-frame callback of the pass:
//! live.offer(&aligned_frame);
//! ```
//!
//! # Viewport
//!
//! A frame of another size than the one shown is a different picture: the view
//! returns to Fit ([`viewport::image_shown`]). A same-sized frame (a retouch
//! stroke) keeps zoom and pan.

use std::sync::Mutex;

use slint::{ComponentHandle, Image, Rgba8Pixel, SharedPixelBuffer, Weak};
use stacker_gui::{redraw_gate::RedrawGate, sync::lock};

use super::{Ctx, viewport};
use crate::{App, ViewportModel};

/// A finished RGBA8 frame.
pub type Pixels = SharedPixelBuffer<Rgba8Pixel>;

/// The latest frame waiting to be shown; see the module docs.
static GATE: RedrawGate<Pixels> = RedrawGate::new();

/// The window the queued closure is posted to. Set by [`setup`]; frames
/// submitted before that are dropped.
static WINDOW: Mutex<Option<Weak<App>>> = Mutex::new(None);

/// Remember the window the queued repaint closure is posted to.
pub fn setup(ui: &App, _ctx: &Ctx) {
    *lock(&WINDOW) = Some(ui.as_weak());
}

/// Show `frame` on the canvas. Callable from any thread; it returns at once.
///
/// At most one closure is queued on the event loop however often this is called:
/// the closure takes the newest frame, so intermediate frames of a burst are
/// skipped (and dropped here, on the producer's side).
pub fn submit_frame(frame: Pixels) {
    if GATE.submit(frame).is_none() {
        // A queued closure will pick this frame up.
        return;
    }
    let window = lock(&WINDOW).clone();
    let posted = window.is_some_and(|ui| ui.upgrade_in_event_loop(|app| show_latest(&app)).is_ok());
    if !posted {
        // Nothing will run the closure: free the gate for the next submit.
        let _ = GATE.take();
    }
}

/// The queued closure: show the newest submitted frame. UI thread only.
fn show_latest(app: &App) {
    let Some(frame) = GATE.take() else {
        return;
    };
    let (width, height) = (frame.width(), frame.height());
    let model = app.global::<ViewportModel>();
    let shown = (model.get_image_width(), model.get_image_height());
    let resized = shown != (to_i32(width), to_i32(height));
    app.set_displayed_image(Image::from_rgba8(frame));
    if resized && let Some(ctx) = Ctx::current() {
        viewport::image_shown(&ctx, width, height);
    }
}

/// A raster dimension as a Slint `int`.
fn to_i32(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}
