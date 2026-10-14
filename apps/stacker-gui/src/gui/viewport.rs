//! The image viewport: wheel zoom about the pointer, pan, Fit and true 1:1, and
//! the placement of the displayed frame inside the canvas.
//!
//! The fit / zoom / pan rule lives once, in [`stacker_gui::viewport_map::Frame`].
//! The view state itself (`zoom`, `pan-x`, `pan-y`) and the canvas and raster
//! sizes live in the Slint `ViewportModel` global, so there is no second copy to
//! drift: every handler reads the model, computes the new view with the pure
//! functions and [`publish`]es it back together with the image placement the
//! canvas draws, and the toolbar's zoom read-out (`AppModel.zoom-factor`, in
//! physical pixels per raster pixel, so 1.0 really is "100 %").

use std::{cell::Cell, rc::Rc, time::Duration};

use slint::{ComponentHandle, Timer, TimerMode};
use stacker_gui::{
    stall_guard::stall_guard,
    viewport_map::{Frame, View, wheel_zoom},
};

use super::Ctx;
use crate::{App, ViewportModel};

/// A burst of canvas size changes (dragging a split handle or the window edge)
/// is recomputed at most once per this interval.
const RESIZE_COALESCE: Duration = Duration::from_millis(50);

/// A Slint `int` size as a raster dimension; negative counts as empty.
fn dimension(value: i32) -> u32 {
    u32::try_from(value).unwrap_or(0)
}

/// The current canvas and raster geometry.
fn frame(app: &App) -> Frame {
    let model = app.global::<ViewportModel>();
    Frame::new(
        model.get_canvas_width(),
        model.get_canvas_height(),
        dimension(model.get_image_width()),
        dimension(model.get_image_height()),
    )
}

/// The current zoom and pan.
fn view(app: &App) -> View {
    let model = app.global::<ViewportModel>();
    View {
        zoom: model.get_zoom(),
        pan_x: model.get_pan_x(),
        pan_y: model.get_pan_y(),
    }
}

/// `wanted` limited to the zoom range of `frame` on this display.
fn limited(app: &App, frame: Frame, wanted: f32) -> f32 {
    let (min, max) = frame.zoom_limits(app.window().scale_factor());
    if wanted.is_finite() {
        wanted.clamp(min, max)
    } else {
        1.0
    }
}

/// Store `wanted` (limited and pan-clamped) in the model together with the image
/// placement it implies, and update the toolbar read-out.
fn publish(app: &App, frame: Frame, wanted: View) {
    let view = frame.clamp_pan(View {
        zoom: limited(app, frame, wanted.zoom),
        ..wanted
    });
    let place = frame.placement(view);
    let model = app.global::<ViewportModel>();
    model.set_zoom(view.zoom);
    model.set_pan_x(view.pan_x);
    model.set_pan_y(view.pan_y);
    model.set_place_x(place.x);
    model.set_place_y(place.y);
    model.set_place_width(place.width);
    model.set_place_height(place.height);
    model.set_place_scale(place.scale);
    app.set_zoom_factor(frame.physical_scale(view, app.window().scale_factor()));
}

/// Recompute the placement for the current geometry and view (canvas resized,
/// image replaced).
fn refresh(app: &App) {
    publish(app, frame(app), view(app));
}

/// Whether the model's raster size differs from `width` x `height`.
fn raster_changed(app: &App, width: i32, height: i32) -> bool {
    let model = app.global::<ViewportModel>();
    (model.get_image_width(), model.get_image_height()) != (width, height)
}

/// Fit the whole image into the canvas.
fn reset_view(app: &App) {
    publish(app, frame(app), View::default());
}

/// Zoom to `zoom` about the centre of the canvas (buttons and shortcuts).
fn zoom_about_centre(app: &App, zoom: f32) {
    let frame = frame(app);
    let zoom = limited(app, frame, zoom);
    let view = frame.zoom_about(view(app), zoom, frame.canvas_w / 2.0, frame.canvas_h / 2.0);
    publish(app, frame, view);
}

/// Tell the viewport that a different image is on the canvas: the view returns
/// to Fit. Call it from the display path whenever the user selects another frame
/// or a result; same-sized pixel updates (a retouch stroke) must not call it.
pub fn image_shown(ctx: &Ctx, width: u32, height: u32) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    let model = app.global::<ViewportModel>();
    model.set_image_width(i32::try_from(width).unwrap_or(i32::MAX));
    model.set_image_height(i32::try_from(height).unwrap_or(i32::MAX));
    reset_view(&app);
}

/// Register the zoom, pan, canvas-size and image-change callbacks.
pub fn setup(ui: &App, ctx: &Ctx) {
    wire_wheel(ui);
    wire_pan(ui);
    wire_buttons(ui);
    wire_canvas_size(ui);
    wire_image_change(ui, ctx);
}

/// Wheel: zoom about the pointer, one notch is a factor of 1.1.
fn wire_wheel(ui: &App) {
    let weak = ui.as_weak();
    ui.global::<ViewportModel>().on_wheel(move |notches, x, y| {
        stall_guard("viewport::wheel", || {
            let Some(app) = weak.upgrade() else {
                return;
            };
            let frame = frame(&app);
            if frame.fit_scale() <= 0.0 {
                return;
            }
            let current = view(&app);
            let zoom = limited(&app, frame, wheel_zoom(current.zoom, notches));
            publish(&app, frame, frame.zoom_about(current, zoom, x, y));
        });
    });
}

/// Pan drag: incremental canvas-pixel deltas, clamped so the image stays in view.
fn wire_pan(ui: &App) {
    let weak = ui.as_weak();
    ui.global::<ViewportModel>().on_pan_by(move |dx, dy| {
        stall_guard("viewport::pan-by", || {
            let Some(app) = weak.upgrade() else {
                return;
            };
            let current = view(&app);
            publish(
                &app,
                frame(&app),
                View {
                    pan_x: current.pan_x + dx,
                    pan_y: current.pan_y + dy,
                    ..current
                },
            );
        });
    });
}

/// Fit, true 1:1 and the +/- steps.
fn wire_buttons(ui: &App) {
    let model = ui.global::<ViewportModel>();

    let weak = ui.as_weak();
    model.on_zoom_fit(move || {
        stall_guard("viewport::zoom-fit", || {
            if let Some(app) = weak.upgrade() {
                reset_view(&app);
            }
        });
    });

    let weak = ui.as_weak();
    model.on_zoom_one_to_one(move || {
        stall_guard("viewport::zoom-one-to-one", || {
            let Some(app) = weak.upgrade() else {
                return;
            };
            if let Some(zoom) = frame(&app).physical_one_to_one_zoom(app.window().scale_factor()) {
                zoom_about_centre(&app, zoom);
            }
        });
    });

    let weak = ui.as_weak();
    model.on_zoom_step(move |factor| {
        stall_guard("viewport::zoom-step", || {
            if let Some(app) = weak.upgrade() {
                zoom_about_centre(&app, view(&app).zoom * factor);
            }
        });
    });
}

/// Canvas size changes, coalesced: the first change of a burst is applied at
/// once, further changes within [`RESIZE_COALESCE`] are applied once when the
/// interval ends.
fn wire_canvas_size(ui: &App) {
    let weak = ui.as_weak();
    let timer = Timer::default();
    let dirty = Rc::new(Cell::new(false));
    ui.global::<ViewportModel>().on_canvas_resized(move || {
        stall_guard("viewport::canvas-resized", || {
            let Some(app) = weak.upgrade() else {
                return;
            };
            if timer.running() {
                dirty.set(true);
                return;
            }
            refresh(&app);
            let weak = weak.clone();
            let dirty = Rc::clone(&dirty);
            timer.start(TimerMode::SingleShot, RESIZE_COALESCE, move || {
                if dirty.replace(false)
                    && let Some(app) = weak.upgrade()
                {
                    refresh(&app);
                }
            });
        });
    });
}

/// The canvas reports the raster size of whatever it displays. A new size means a
/// different image ([`image_shown`]: back to Fit). The same size keeps zoom and
/// pan, which is what stepping through same-sized frames and retouch strokes need.
fn wire_image_change(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.global::<ViewportModel>()
        .on_image_changed(move |width, height| {
            stall_guard("viewport::image-changed", || {
                let Some(app) = ctx.ui.upgrade() else {
                    return;
                };
                if raster_changed(&app, width, height) {
                    image_shown(&ctx, dimension(width), dimension(height));
                } else {
                    refresh(&app);
                }
            });
        });
}
