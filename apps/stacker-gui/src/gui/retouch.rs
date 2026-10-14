//! The Retouch tool: brush strokes, undo and redo, and the source-frame picker
//! for the newest result.
//!
//! The tool is a state of the window (`RetouchModel.tool-on`, switched by the
//! toolbar toggle, the `R` key, the Edit menu, Esc and Done). While it is on the
//! options row is shown and a left-drag paints; while it is off a left-drag pans,
//! and the canvas emits no brush events at all. Rust hears about a change through
//! `RetouchModel.tool-toggled` ([`wire_tool`]).
//!
//! The canvas reports the pointer in logical canvas pixels through
//! `ViewportModel.brush-down/moved/up`. Each position goes through
//! [`stacker_gui::viewport_map::Frame`] (so zoom, pan and the letterbox are
//! honoured and a pointer outside the image paints nothing) and is appended to a
//! queue in [`RetouchState`](stacker_gui::retouch::RetouchState). A 16 ms timer
//! drains the queue (painting the alpha plane, which is proportional to the brush)
//! and repaints once per tick, however many `moved` events arrived. One drag is one
//! undo step.
//!
//! # The preview is composited off the UI thread
//!
//! The tick does not composite. [`present`] copies what the brush changed into the
//! state's [`PreviewFeed`] and wakes one [`LatestWorker`]; that thread composites the
//! changed rectangle into its 8-bit frame, and the frame reaches the canvas through
//! [`super::preview::submit_frame`] if the newest result is still the image on view.
//! Ticks that arrive while the worker is busy merge into one larger rectangle.
//!
//! # Hooks for the rest of the GUI
//!
//! [`sync_panel`] brings the Retouch options (source list, hint, undo/redo state)
//! in line with the application state and makes the retouch session match the
//! newest fused result. A slow timer calls it too, so the options are right even
//! if no caller does; calling it from the Stack pass's `publish_result`, the Align
//! pass's completion and the file-list and result-list clicks only makes them
//! update at once.

use std::{cell::RefCell, sync::Arc, time::Duration};

use slint::{ComponentHandle, Model, ModelRc, SharedString, Timer, TimerMode, VecModel, Weak};
use stacker_core::image::PlanarImage;
use stacker_gui::{
    latest_worker::LatestWorker,
    retouch::{SourceChoice, SourceRequest},
    retouch_preview::{Compositor, PreviewFeed},
    runner,
    stall_guard::stall_guard,
    viewport_map::View,
};

use super::{
    Ctx,
    ctx::on_ui_ctx,
    explainer,
    feedback::{self, MessageKind},
};
use crate::{App, RetouchModel, ViewportModel};

/// How often queued pointer events are painted (about 60 per second).
const DRAIN_INTERVAL: Duration = Duration::from_millis(16);
/// How often the panel re-checks the application state.
const PANEL_WATCH_INTERVAL: Duration = Duration::from_millis(400);

/// The thread that composites the retouch preview.
type CompositorThread = LatestWorker<Arc<PreviewFeed>>;

thread_local! {
    /// Paints the queued pointer events while a stroke is open.
    static DRAIN: Timer = Timer::default();
    /// Keeps the panel in step with the state.
    static PANEL_WATCH: Timer = Timer::default();
    /// The compositor of the window [`setup`] wired last.
    static COMPOSITOR: RefCell<Option<CompositorThread>> = const { RefCell::new(None) };
}

/// Register the tool, brush, undo, redo and source-frame callbacks, start the
/// preview worker and the options watcher.
pub fn setup(ui: &App, ctx: &Ctx) {
    install_compositor(ui.as_weak());
    wire_tool(ui, ctx);
    wire_brush(ui, ctx);
    wire_undo(ui, ctx);
    wire_redo(ui, ctx);
    wire_source(ui, ctx);
    start_panel_watch(ctx);
    sync_panel(ctx);
}

// ── The tool ─────────────────────────────────────────────────────────────────

/// The Retouch tool was switched on or off. Switching it on brings the options
/// up to date at once (so the first thing the row shows is right) and shows the
/// first-use explainer once; switching it off needs nothing, the canvas has
/// already stopped sending brush events.
fn wire_tool(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.global::<RetouchModel>().on_tool_toggled(move |on| {
        stall_guard("retouch::tool-toggled", || {
            if on {
                sync_panel(&ctx);
                explainer::show_once(
                    &ctx,
                    "retouch-sources",
                    "Retouch from a source frame",
                    "The brush paints pixels from the frame chosen under Paint from onto the \
                     result. Changing the frame later re-sources your strokes. Undo and Redo \
                     step through the strokes. A left-drag paints while the tool is on; press \
                     Esc or Done to leave it, and a left-drag pans again.",
                );
            }
        });
    });
}

/// Start the thread that turns the feed's work into frames and posts them to the
/// canvas.
fn install_compositor(ui: Weak<App>) {
    let mut compositor = Compositor::default();
    let worker = LatestWorker::spawn("z-stackr-retouch-preview", move |feed: Arc<PreviewFeed>| {
        if let Some(work) = feed.take()
            && let Some(pixels) = compositor.apply(work)
        {
            on_ui_ctx(&ui, move |_app, ctx| {
                // A frame that outlived the view it was made for is dropped.
                let viewing = ctx.state.borrow().viewing_retouchable();
                if viewing {
                    super::preview::submit_frame(pixels);
                }
            });
        }
    });
    COMPOSITOR.with(|cell| *cell.borrow_mut() = Some(worker));
}

/// Wake the preview worker for `feed`. Returns at once.
fn request_composite(feed: Arc<PreviewFeed>) {
    COMPOSITOR.with(|cell| {
        if let Some(worker) = cell.borrow().as_ref() {
            let _ = worker.submit(feed);
        }
    });
}

// ── Pointer events ───────────────────────────────────────────────────────────

/// What a pointer event needs from the window: the canvas geometry and the brush.
struct Pointer {
    canvas: (f32, f32),
    view: View,
    /// Brush radius in raster pixels (the slider's value).
    radius: f32,
    opacity: f32,
}

fn read_pointer(app: &App) -> Pointer {
    let viewport = app.global::<ViewportModel>();
    let retouch = app.global::<RetouchModel>();
    Pointer {
        canvas: (viewport.get_canvas_width(), viewport.get_canvas_height()),
        view: View {
            zoom: viewport.get_zoom(),
            pan_x: viewport.get_pan_x(),
            pan_y: viewport.get_pan_y(),
        },
        radius: retouch.get_brush_size(),
        opacity: retouch.get_brush_opacity(),
    }
}

fn wire_brush(ui: &App, ctx: &Ctx) {
    let viewport = ui.global::<ViewportModel>();
    let down = ctx.clone();
    viewport.on_brush_down(move |x, y| {
        stall_guard("retouch::brush-down", || brush_down(&down, x, y));
    });
    let moved = ctx.clone();
    viewport.on_brush_moved(move |x, y| {
        stall_guard("retouch::brush-moved", || brush_moved(&moved, x, y));
    });
    let up = ctx.clone();
    viewport.on_brush_up(move || stall_guard("retouch::brush-up", || brush_up(&up)));
}

/// Pointer press: open a stroke and paint at the pressed pixel.
fn brush_down(ctx: &Ctx, x: f32, y: f32) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    let blocker = ctx.state.borrow().retouch_blocker();
    if let Some(reason) = blocker {
        feedback::show_status(&app, reason);
        return;
    }
    let pointer = read_pointer(&app);
    {
        let mut state = ctx.state.borrow_mut();
        if !state.retouch.begin_stroke() {
            return;
        }
        let raster = state
            .retouch
            .map_pointer(pointer.canvas, pointer.view, x, y);
        state
            .retouch
            .queue_point(raster, pointer.radius, pointer.opacity);
    }
    DRAIN.with(|timer| {
        if !timer.running() {
            let ctx = ctx.clone();
            timer.start(TimerMode::Repeated, DRAIN_INTERVAL, move || tick(&ctx));
        }
    });
}

/// Pointer move: only queue it; the timer paints.
fn brush_moved(ctx: &Ctx, x: f32, y: f32) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    if !ctx.state.borrow().retouch.stroke_open() {
        return;
    }
    let pointer = read_pointer(&app);
    let mut state = ctx.state.borrow_mut();
    let raster = state
        .retouch
        .map_pointer(pointer.canvas, pointer.view, x, y);
    state
        .retouch
        .queue_point(raster, pointer.radius, pointer.opacity);
}

/// Pointer release: paint what is left and commit the drag as one undo step.
fn brush_up(ctx: &Ctx) {
    DRAIN.with(Timer::stop);
    let had_stroke = {
        let mut state = ctx.state.borrow_mut();
        let open = state.retouch.stroke_open();
        state.retouch.end_stroke();
        open
    };
    if had_stroke {
        present(ctx);
        super::project::mark_dirty(ctx);
    }
}

/// One timer tick: paint the queued events, repaint if anything changed, and
/// stop once the stroke is over.
fn tick(ctx: &Ctx) {
    let (painted, open) = {
        let mut state = ctx.state.borrow_mut();
        let painted = state.retouch.drain_pending();
        (painted, state.retouch.stroke_open())
    };
    if painted > 0 {
        present(ctx);
    }
    if !open {
        DRAIN.with(Timer::stop);
    }
}

// ── Showing the result ───────────────────────────────────────────────────────

/// Show the composite on the canvas (when the canvas shows the newest result) and
/// update the undo/redo buttons. The composite is made by the preview worker; this
/// only hands it what changed.
pub(super) fn present(ctx: &Ctx) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    // The feed is brought up to date even while another image is on view, so the
    // worker's next frame is right whenever the newest result is shown again.
    let (feed, viewing) = {
        let mut state = ctx.state.borrow_mut();
        let feed = state.retouch.flush_preview();
        (feed, state.viewing_retouchable())
    };
    if viewing {
        request_composite(feed);
    }
    push_history(ctx, &app);
}

/// Write the undo/redo availability and labels into `RetouchModel`.
fn push_history(ctx: &Ctx, app: &App) {
    let (can_undo, can_redo, undo, redo) = {
        let state = ctx.state.borrow();
        (
            state.retouch.can_undo(),
            state.retouch.can_redo(),
            state.retouch.undo_label(),
            state.retouch.redo_label(),
        )
    };
    let model = app.global::<RetouchModel>();
    model.set_can_undo(can_undo);
    model.set_can_redo(can_redo);
    model.set_undo_label(undo.into());
    model.set_redo_label(redo.into());
}

/// Bring the Retouch options in line with the state: make the session match the
/// newest result, refill the source list when it changed, and update the hint
/// and the undo/redo buttons. Idempotent and cheap; ignored while a stroke is
/// open.
pub fn sync_panel(ctx: &Ctx) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    let model = app.global::<RetouchModel>();
    let rows = model.get_source_names().row_count();
    let (labels, index, active, hint) = {
        let mut guard = ctx.state.borrow_mut();
        let state = &mut *guard;
        if state.retouch.stroke_open() {
            return;
        }
        state.ensure_retouch_session();
        let changed = state.refresh_retouch_sources();
        let labels = (changed || rows != state.retouch.sources().len())
            .then(|| state.retouch.source_labels());
        (
            labels,
            state
                .retouch
                .source_index()
                .map_or(-1, |i| i32::try_from(i).unwrap_or(-1)),
            state.viewing_retouchable(),
            state.retouch_blocker().unwrap_or_default(),
        )
    };
    if let Some(labels) = labels {
        let names: Vec<SharedString> = labels.into_iter().map(SharedString::from).collect();
        model.set_source_names(ModelRc::new(VecModel::from(names)));
    }
    model.set_source_index(index);
    model.set_active(active);
    model.set_hint(hint.into());
    push_history(ctx, &app);
}

fn start_panel_watch(ctx: &Ctx) {
    let ctx = ctx.clone();
    PANEL_WATCH.with(|timer| {
        timer.start(TimerMode::Repeated, PANEL_WATCH_INTERVAL, move || {
            sync_panel(&ctx);
        });
    });
}

// ── Undo and redo ────────────────────────────────────────────────────────────

fn wire_undo(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_undo_stroke(move || {
        stall_guard("retouch::undo-stroke", || step_history(&ctx, true));
    });
}

fn wire_redo(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_redo_stroke(move || {
        stall_guard("retouch::redo-stroke", || step_history(&ctx, false));
    });
}

/// Undo (`undo`) or redo one stroke, when the canvas shows the retouched result.
fn step_history(ctx: &Ctx, undo: bool) {
    let viewing = ctx.state.borrow().viewing_retouchable();
    if !viewing {
        if let Some(app) = ctx.ui.upgrade() {
            feedback::show_status(
                &app,
                "Select the newest result in the results list to undo or redo its strokes.",
            );
        }
        return;
    }
    let changed = {
        let mut state = ctx.state.borrow_mut();
        if undo {
            state.retouch.undo()
        } else {
            state.retouch.redo()
        }
    };
    if changed {
        present(ctx);
        super::project::mark_dirty(ctx);
    }
}

// ── Source frame ─────────────────────────────────────────────────────────────

fn wire_source(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.global::<RetouchModel>().on_source_chosen(move |index| {
        stall_guard("retouch::source-chosen", || choose_source(&ctx, index));
    });
}

/// The user picked entry `index` of the source list (`-1`: none).
fn choose_source(ctx: &Ctx, index: i32) {
    let wanted = usize::try_from(index).ok();
    let choice = ctx.state.borrow_mut().retouch.choose_source(wanted);
    match choice {
        SourceChoice::Unchanged => {}
        SourceChoice::Cleared => present(ctx),
        SourceChoice::Load(request) => load_source(ctx, request),
    }
}

/// Ask the job runner for the frame of `request`: the aligned frame from the
/// Align pass (cut to the common area) when it holds one, else the decoded file.
fn load_source(ctx: &Ctx, request: SourceRequest) {
    if let Some(app) = ctx.ui.upgrade() {
        feedback::show_status(
            &app,
            &format!("Loading {}…", feedback::file_label(&request.path)),
        );
    }
    let settings = ctx.settings.snapshot();
    let ctx = ctx.clone();
    let path = request.path.clone();
    runner::current().fetch_source(
        &path,
        &settings,
        Box::new(move |outcome| finish_source(&ctx, &request, outcome)),
    );
}

/// Install the fetched frame as the source layer, or report why it cannot be.
fn finish_source(ctx: &Ctx, request: &SourceRequest, outcome: Result<PlanarImage<f32>, String>) {
    let applied = match outcome {
        Ok(frame) => ctx
            .state
            .borrow_mut()
            .retouch
            .finish_source(request, frame)
            .map_err(|e| e.to_string()),
        Err(message) => Err(message),
    };
    match applied {
        Ok(true) => present(ctx),
        Ok(false) => {}
        Err(message) => {
            let reverted = ctx.state.borrow_mut().retouch.fail_source(request);
            feedback::notify(
                ctx,
                MessageKind::Error,
                &format!(
                    "Could not use {} as the retouch source: {message}",
                    feedback::file_label(&request.path)
                ),
            );
            if reverted {
                sync_panel(ctx);
            }
        }
    }
}
