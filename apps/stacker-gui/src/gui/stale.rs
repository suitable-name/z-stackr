//! Result currency in the sidebar: which of the aligned cache and the newest
//! result no longer match the files and settings they were built from.
//!
//! The pure derivation lives in [`stacker_gui::stale`] (and is unit-tested
//! there). This module moves its answer into `FilesModel`, where the sidebar
//! reads it: `aligned-summary` / `aligned-stale` / `aligned-stale-text` for the
//! Align cache, `result-stale` / `result-stale-text` for the newest result, and
//! `culled-files` for the auto-cull marks on the source rows.
//!
//! Two things trigger a push: a 400 ms timer (the file list and the settings are
//! edited from many places, so polling is cheaper than hooking each one) and the
//! passes that change the answer ([`refresh`] after an Align or a published
//! result). The timer sits out while a blocking job runs, since the sources are
//! being read then, and the passes push their own result when they finish. Only
//! a changed answer reaches Slint, so an idle sidebar is never rebuilt.
//!
//! The badge's Recompute starts the same Align / Stack pass as the toolbar, with
//! the sidebar's current algorithm choice for Stack.

use std::{cell::RefCell, time::Duration};

use slint::{ComponentHandle, ModelRc, Timer, TimerMode, VecModel};
use stacker_gui::{
    stale::{StaleReason, StaleView, stale_view},
    stall_guard::stall_guard,
};

use super::{Ctx, activity, align, stack};
use crate::{App, FilesModel};

/// How often the sidebar re-checks the currency of the aligned cache and result.
const POLL_INTERVAL: Duration = Duration::from_millis(400);

thread_local! {
    /// What the sidebar currently shows, so an unchanged answer pushes nothing.
    static PUSHED: RefCell<Option<StaleView>> = const { RefCell::new(None) };
}

/// Wire the badges' Recompute actions and start the polling timer.
pub fn setup(ui: &App, ctx: &Ctx) {
    // Nothing pushed to this window yet.
    PUSHED.with(|pushed| *pushed.borrow_mut() = None);
    let files = ui.global::<FilesModel>();
    files.on_recompute_aligned({
        let ctx = ctx.clone();
        move || stall_guard("stale::recompute-aligned", || align::start(&ctx))
    });
    files.on_recompute_result({
        let ctx = ctx.clone();
        move |algo, ai_model, ai_device| {
            stall_guard("stale::recompute-result", || {
                stack::start(
                    &ctx,
                    algo.as_str(),
                    ai_model.to_string(),
                    ai_device.to_string(),
                );
            });
        }
    });

    // The timer lives as long as the process; leaking it avoids a thread-local
    // destructor racing the event loop's teardown.
    let timer = Box::leak(Box::new(Timer::default()));
    let ctx = ctx.clone();
    timer.start(TimerMode::Repeated, POLL_INTERVAL, move || {
        if activity::current().is_some_and(|registry| registry.is_busy()) {
            return;
        }
        if let Some(app) = ctx.ui.upgrade() {
            refresh(&app, &ctx);
        }
    });
}

/// Recompute the currency of the aligned cache and the newest result and the
/// auto-cull marks, and push them to `FilesModel` when they changed.
pub fn refresh(app: &App, ctx: &Ctx) {
    // The model pick belongs to the alignment fingerprint.
    stack::sync_align_model(app, ctx);
    let settings = ctx.settings.snapshot();
    let view = {
        let state = ctx.state.borrow();
        stale_view(&state, &settings)
    };
    let changed = PUSHED.with(|slot| {
        let mut last = slot.borrow_mut();
        if last.as_ref() == Some(&view) {
            false
        } else {
            *last = Some(view.clone());
            true
        }
    });
    if changed {
        push(app, &view);
    }
}

fn badge_text(reason: Option<StaleReason>) -> &'static str {
    reason.map_or("", StaleReason::badge_text)
}

fn push(app: &App, view: &StaleView) {
    let files = app.global::<FilesModel>();
    files.set_aligned_summary(view.aligned_summary.as_str().into());
    files.set_aligned_stale(view.aligned_stale.is_some());
    files.set_aligned_stale_text(badge_text(view.aligned_stale).into());
    files.set_result_stale(view.result_stale.is_some());
    files.set_result_stale_text(badge_text(view.result_stale).into());
    files.set_culled_files(ModelRc::new(VecModel::from(view.culled.clone())));
}
