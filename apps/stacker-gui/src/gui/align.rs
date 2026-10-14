//! The Align-only pass.
//!
//! The alignment itself (seed, refinement per mode, warp, coverage) is the shared
//! `stacker_pipeline::align`, driven by the job runner, which also keeps the
//! aligned frames. This module is the UI side: the request, the progress, the
//! record of what was aligned ([`AlignedCache`]) and the user-facing messages.

use std::{path::PathBuf, rc::Rc};

use stacker_gui::{
    frames,
    jobs_core::{JobError, StagePlan},
    stale::{Fingerprint, fingerprint_alignment_with, fingerprint_paths},
    stall_guard::stall_guard,
    state::{AlignedCache, Crop},
};

use super::{
    Ctx,
    feedback::{self, MessageKind},
    jobs::{self, Job, JobSpec},
    preview, stack, stale,
};
use crate::{
    App,
    runner::{self, AlignRequest, RunEvent},
};

fn plan() -> StagePlan {
    StagePlan::new(&[("Loading", 0.25), ("Aligning", 0.75)])
}

/// Register the Align button's callback.
pub fn setup(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_align_clicked(move || stall_guard("align::align-clicked", || start(&ctx)));
}

/// What the UI side remembers of an Align run.
struct AlignUi {
    /// The files the run was asked to align.
    wanted: Vec<PathBuf>,
    /// The inputs the run was started with, stamped on the record it makes.
    fingerprint: Fingerprint,
    /// What the run reported once it finished the alignment.
    outcome: Outcome,
}

/// How far an Align run has got, as far as the UI side knows.
enum Outcome {
    /// The alignment has not reported yet.
    Pending,
    /// It aligned `frames` frames (the ones that decoded); the crop, if one is
    /// needed.
    Aligned { frames: usize, crop: Option<Crop> },
}

/// Run the Align pass over the current source list (the Align button and the
/// stale badge's Recompute both come here).
pub fn start(ctx: &Ctx) {
    if jobs::refuse_if_busy(ctx, "Align") {
        return;
    }
    let paths = ctx.state.borrow().file_paths.clone();
    if paths.is_empty() {
        feedback::notify(ctx, MessageKind::Warning, "Align: add images first.");
        return;
    }
    let settings = ctx.settings.snapshot();
    let align_model = ctx
        .ui
        .upgrade()
        .and_then(|app| stack::sync_align_model(&app, ctx));
    let model_name = align_model.as_ref().map_or("", |(name, _)| name.as_str());
    // Stamped when the pass starts: the record is current for the inputs it was
    // given, even if the list changes while it runs.
    let fingerprint = fingerprint_alignment_with(&fingerprint_paths(&paths), &settings, model_name);
    let wanted = frames::paths_for_run(&paths, &settings, false);
    let spec = JobSpec {
        kind: "align",
        label: "Align".to_owned(),
        cancellable: true,
        blocking: true,
        plan: plan(),
        on_cancel: None,
    };
    let Some(job) = jobs::begin(ctx, spec) else {
        return;
    };
    let request = AlignRequest {
        paths: wanted.clone(),
        settings,
        align_model,
    };
    let mut run = AlignUi {
        wanted,
        fingerprint,
        outcome: Outcome::Pending,
    };
    let events = {
        let (ctx, job) = (ctx.clone(), Rc::clone(&job));
        Box::new(move |event| on_event(&ctx, &job, &mut run, event))
    };
    let handle = runner::current().run_align(request, events);
    job.attach(handle);
}

/// One event of an Align run, on the UI thread. Nothing reaches the UI state
/// unless the whole pass finished, so a cancel leaves no half-aligned record
/// behind.
fn on_event(ctx: &Ctx, job: &Job, run: &mut AlignUi, event: RunEvent) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    match event {
        RunEvent::Progress {
            step,
            fraction,
            label,
        } => job.progress(&app, stack::stage_of(step), fraction, &label),
        RunEvent::Warning(text) => feedback::show(&app, MessageKind::Warning, &text),
        RunEvent::LiveFrame(buffer) => preview::submit_frame(buffer),
        RunEvent::AlignDone {
            paths,
            crop,
            warning,
        } => {
            if let Some(warning) = warning {
                feedback::show(&app, MessageKind::Warning, &warning);
            }
            let frames = paths.len();
            let cache = AlignedCache { paths, crop };
            ctx.state.borrow_mut().set_aligned(cache, run.fingerprint);
            stale::refresh(&app, ctx);
            run.outcome = Outcome::Aligned { frames, crop };
        }
        // Only a Stack run reports results, only a Sort / Cull run its census.
        RunEvent::Result(_) | RunEvent::SortCullDone { .. } => {}
        RunEvent::Finished(result) => {
            job.end(&app);
            finished(&app, result, run);
        }
    }
}

/// The UI-thread end of the run: report.
fn finished(app: &App, result: Result<Option<String>, JobError>, run: &AlignUi) {
    match result {
        Ok(note) => {
            let (frames, crop) = match run.outcome {
                Outcome::Aligned { frames, crop } => (frames, crop),
                Outcome::Pending => (run.wanted.len(), None),
            };
            let mut status = format!(
                "Align complete: {frames} frames aligned, {}. Toggle 'Show Aligned' to preview.",
                crop_text(crop)
            );
            if let Some(note) = note {
                status = format!("{status} ({note})");
            }
            feedback::show(app, MessageKind::Success, &status);
        }
        Err(error) => jobs::report_error(app, "Align", &error),
    }
}

/// `no crop` or `crop 640×480 @ (3,4)`.
fn crop_text(crop: Option<Crop>) -> String {
    crop.map_or_else(
        || "no crop".to_owned(),
        |(cx, cy, cw, ch)| format!("crop {cw}×{ch} @ ({cx},{cy})"),
    )
}

#[cfg(test)]
mod tests {
    use super::crop_text;

    #[test]
    fn the_status_names_the_crop() {
        assert_eq!(crop_text(None), "no crop");
        assert_eq!(crop_text(Some((3, 4, 640, 480))), "crop 640×480 @ (3,4)");
    }
}
