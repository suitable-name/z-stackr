//! The `Relief` contrast-preview panel's callbacks.
//!
//! The job runner asks the question ([`crate::runner::ReliefAsk`]) when a `Relief`
//! pass has computed its focus map and the settings want the preview, and parks
//! the run until the GUI answers; the panel is an overlay card, not a modal
//! dialog, so the rest of the window stays usable while the run waits. Apply
//! answers with the chosen threshold, Cancel (and a cancel of the job) with
//! "cancelled".
//!
//! # Rendering
//!
//! The panel works on data already reduced to panel size: the first frame, the
//! focus map, and the full-size threshold of every slider position (1001 values,
//! entry `k` rejects the fraction `k / 1000`). A slider tick is therefore an
//! index lookup and a tint of a few hundred thousand pixels, done on the UI
//! thread into a rotating [`FrameRing`]; nothing is cloned or sorted.

use std::cell::RefCell;

use slint::{ComponentHandle, Image, Rgba8Pixel, SharedPixelBuffer};
use stacker_gui::{
    preview::{FrameRing, tint_rejected},
    stall_guard::stall_guard,
};

use super::{Ctx, explainer};
use crate::{
    App, SettingsModel,
    runner::{AskRelief, ReliefAsk},
};

/// The open panel: what it renders from and where its answer goes.
struct Panel {
    base: SharedPixelBuffer<Rgba8Pixel>,
    focus: Vec<f32>,
    thresholds: Vec<f32>,
    auto_pct: f32,
    ring: FrameRing,
    reply: Box<dyn FnOnce(Option<f32>)>,
}

thread_local! {
    /// The panel of the question being asked, if any (UI thread only).
    static OPEN: RefCell<Option<Panel>> = const { RefCell::new(None) };
}

/// Register the panel's callbacks.
pub fn setup(ui: &App, ctx: &Ctx) {
    // No question is waiting in a new window.
    OPEN.with(|open| *open.borrow_mut() = None);
    wire_apply(ui, ctx);
    wire_cancel(ui, ctx);
    wire_auto_detect(ui, ctx);
    wire_contrast_changed(ui, ctx);
}

/// The question handler a stack run is started with: it opens the panel and
/// answers the run when the user does.
pub fn asker(ctx: &Ctx) -> AskRelief {
    let ctx = ctx.clone();
    Box::new(move |ask, reply| open(&ctx, ask, reply))
}

/// Close the panel and answer its question with "cancelled", if one is open. The
/// run is being cancelled, so it must not wait for a click.
pub fn cancel_pending(ctx: &Ctx) {
    let panel = OPEN.with(|open| open.borrow_mut().take());
    if let Some(panel) = panel {
        if let Some(app) = ctx.ui.upgrade() {
            app.set_relief_preview_open(false);
        }
        (panel.reply)(None);
    }
}

/// Show the question: set the slider to the starting fraction, show the overlay
/// rendered for it and make the panel visible.
fn open(ctx: &Ctx, ask: ReliefAsk, reply: Box<dyn FnOnce(Option<f32>)>) {
    let Some(app) = ctx.ui.upgrade() else {
        reply(None);
        return;
    };
    // A question still open belongs to a run that is gone: end it.
    cancel_pending(ctx);
    let ReliefAsk {
        base,
        focus_max,
        thresholds,
        auto_pct,
        initial_pct,
    } = ask;
    let mut panel = Panel {
        base,
        focus: focus_max,
        thresholds,
        auto_pct,
        ring: FrameRing::new(),
        reply,
    };
    let initial = render(&mut panel, initial_pct);
    OPEN.with(|open| *open.borrow_mut() = Some(panel));
    app.global::<SettingsModel>()
        .set_relief_contrast_pct(initial_pct);
    app.set_relief_preview_image(Image::from_rgba8(initial));
    app.set_relief_preview_open(true);
    explainer::show_once(
        ctx,
        "relief-preview",
        "Depth-map preview",
        "Move the contrast threshold and watch the mask change, or press \
         Auto-detect. Apply continues the run with the chosen value; Cancel stops it.",
    );
}

// ── Rendering ────────────────────────────────────────────────────────────────

/// The position of the rejected fraction `contrast_pct` in a threshold table of
/// `len` entries (`0.0` is the first, `1.0` the last).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a fraction in 0..=1 scaled to a table index, rounded and clamped to the table"
)]
fn table_index(contrast_pct: f32, len: usize) -> usize {
    let last = len.saturating_sub(1);
    ((contrast_pct.clamp(0.0, 1.0) * last as f32).round() as usize).min(last)
}

/// The absolute threshold of the rejected fraction `contrast_pct`.
fn threshold_at(panel: &Panel, contrast_pct: f32) -> f32 {
    panel
        .thresholds
        .get(table_index(contrast_pct, panel.thresholds.len()))
        .copied()
        .unwrap_or(0.0)
}

/// The overlay for rejected fraction `contrast_pct`, written into the next buffer
/// of the panel's ring: the rejected pixels tinted red on the panel-size frame.
fn render(panel: &mut Panel, contrast_pct: f32) -> SharedPixelBuffer<Rgba8Pixel> {
    let threshold = threshold_at(panel, contrast_pct);
    let Panel {
        base, focus, ring, ..
    } = panel;
    ring.fill_with(base.width(), base.height(), |out| {
        tint_rejected(base.as_slice(), focus, threshold, out);
    })
}

/// Show the overlay for `contrast_pct` (no-op without an open panel).
fn show(app: &App, contrast_pct: f32) {
    let overlay = OPEN.with(|open| {
        open.borrow_mut()
            .as_mut()
            .map(|panel| render(panel, contrast_pct))
    });
    if let Some(overlay) = overlay {
        app.set_relief_preview_image(Image::from_rgba8(overlay));
    }
}

// ── Callbacks ────────────────────────────────────────────────────────────────

fn wire_apply(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_relief_preview_apply(move || {
        stall_guard("relief_preview::apply", || {
            let panel = OPEN.with(|open| open.borrow_mut().take());
            if let Some(panel) = panel
                && let Some(app) = ctx.ui.upgrade()
            {
                app.set_relief_preview_open(false);
                let chosen = app.global::<SettingsModel>().get_relief_contrast_pct();
                let threshold = threshold_at(&panel, chosen);
                (panel.reply)(Some(threshold));
            }
        });
    });
}

fn wire_cancel(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_relief_preview_cancel(move || {
        stall_guard("relief_preview::cancel", || {
            // The run returns `Cancelled` to the job, which ends the pass exactly
            // as the activity chip's Cancel does.
            cancel_pending(&ctx);
        });
    });
}

fn wire_auto_detect(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_relief_preview_auto_detect(move || {
        stall_guard("relief_preview::auto-detect", || {
            // The auto-detected (median + 1.5·MAD) fraction was computed once, when the question was asked.
            let auto = OPEN.with(|open| open.borrow().as_ref().map(|panel| panel.auto_pct));
            if let (Some(pct), Some(app)) = (auto, ctx.ui.upgrade()) {
                app.global::<SettingsModel>().set_relief_contrast_pct(pct);
                show(&app, pct);
            }
        });
    });
}

fn wire_contrast_changed(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_relief_preview_contrast_changed(move || {
        stall_guard("relief_preview::contrast-changed", || {
            let Some(app) = ctx.ui.upgrade() else {
                return;
            };
            let pct = app.global::<SettingsModel>().get_relief_contrast_pct();
            show(&app, pct);
        });
    });
}

#[cfg(test)]
mod tests {
    use slint::{Rgba8Pixel, SharedPixelBuffer};
    use stacker_gui::preview::FrameRing;

    use super::{Panel, render, table_index, threshold_at};

    /// An 8 x 1 panel: the left half has no focus, the right half has all; the
    /// threshold table is the identity on `k / 1000`.
    fn half_focused() -> Panel {
        let mut base = SharedPixelBuffer::<Rgba8Pixel>::new(8, 1);
        base.make_mut_slice()
            .fill(Rgba8Pixel::new(128, 128, 128, 255));
        let mut focus = vec![0.0_f32; 8];
        focus[4..].fill(1.0);
        Panel {
            base,
            focus,
            thresholds: (0..=1000_u16).map(|k| f32::from(k) / 1000.0).collect(),
            auto_pct: 0.5,
            ring: FrameRing::new(),
            reply: Box::new(|_| {}),
        }
    }

    #[test]
    fn the_slider_position_picks_its_table_entry() {
        assert_eq!(table_index(0.0, 1001), 0);
        assert_eq!(table_index(0.5, 1001), 500);
        assert_eq!(table_index(1.0, 1001), 1000);
        assert_eq!(table_index(7.0, 1001), 1000, "clamped to the table");
        assert_eq!(table_index(-1.0, 1001), 0);
        assert_eq!(table_index(0.5, 0), 0, "an empty table never indexes out");
        let panel = half_focused();
        assert!((threshold_at(&panel, 0.25) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn the_overlay_tints_the_unfocused_half_only() {
        let mut panel = half_focused();
        let overlay = render(&mut panel, 0.5);
        let px = overlay.as_slice();
        assert_eq!(px.len(), 8);
        assert!(px[0].r > px[7].r, "rejected pixels are redder");
        assert!(px[0].g < px[7].g, "and less green");
        assert_eq!(px[4], px[7], "focused pixels are the plain frame");
    }

    #[test]
    fn nothing_is_tinted_when_no_fraction_is_rejected() {
        let mut panel = half_focused();
        let overlay = render(&mut panel, 0.0);
        let px = overlay.as_slice();
        assert!(px.iter().all(|p| *p == px[0]));
    }
}
