//! The Sort and Cull actions: look at the aligned frames, then reorder the source
//! list by sharpness (Sort) or mark the frames that add too little sharp detail
//! (Cull).
//!
//! Both are [`crate::runner`] runs: the runner aligns the files (or serves the
//! held aligned set when it is current), takes the focus census
//! (`session::sort_frames_by_sharpness` / `session::cull_frames`) and answers with
//! [`RunEvent::SortCullDone`]. A run that had to align leaves the aligned set
//! behind, exactly as the Align button does, so a Stack run after it starts from
//! aligned frames.
//!
//! * **Sort** rewrites [`stacker_gui::state::AppState::file_paths`] in the census
//!   order (the files the run did not look at keep their places). The aligned
//!   record then reads "Stale (files)": the frames were aligned to the old first
//!   frame, and a new Align uses the sharpest as the reference.
//! * **Cull** only sets the dimmed marks of the source rows
//!   ([`stacker_gui::state::AppState::set_culled`]); the list itself stays the
//!   user's, as with the auto-cull of a Stack run.
//!
//! Both are refused while another blocking job runs.

use std::rc::Rc;

use stacker_gui::{
    frames,
    jobs_core::{JobError, StagePlan},
    settings::StackingSettings,
    stale::{Fingerprint, fingerprint_alignment_with, fingerprint_paths},
    stall_guard::stall_guard,
    state::{AlignedCache, Crop},
};

use super::{
    Ctx,
    feedback::{self, MessageKind},
    jobs::{self, Job, JobSpec},
    preview, project, stack, stale,
};
use crate::{
    App,
    runner::{self, RunEvent, SortCullAction, SortCullRequest},
    ui_helpers::push_source_panel,
};

/// The stages of the job: load, align, then the census (the Cull share of the
/// bar).
fn plan() -> StagePlan {
    StagePlan::new(&[("Loading", 0.2), ("Aligning", 0.65), ("Counting", 0.15)])
}

/// Register the `sort-frames()` and `cull-frames()` callbacks.
pub fn setup(ui: &App, ctx: &Ctx) {
    ui.on_sort_frames({
        let ctx = ctx.clone();
        move || stall_guard("sort_cull::sort", || start(&ctx, SortCullAction::Sort))
    });
    ui.on_cull_frames({
        let ctx = ctx.clone();
        move || stall_guard("sort_cull::cull", || start(&ctx, SortCullAction::Cull))
    });
}

/// What the UI side remembers of a run.
struct RunUi {
    action: SortCullAction,
    settings: StackingSettings,
    /// The alignment inputs the run started with, for the record it may make.
    fingerprint: Fingerprint,
    /// The census answered (the run did not end before it).
    answered: bool,
}

/// Start `action` over the current source list.
pub fn start(ctx: &Ctx, action: SortCullAction) {
    let what = action.label();
    if jobs::refuse_if_busy(ctx, what) {
        return;
    }
    let paths = ctx.state.borrow().file_paths.clone();
    if paths.len() < 2 {
        feedback::notify(
            ctx,
            MessageKind::Warning,
            &format!("{what}: add at least two images first."),
        );
        return;
    }
    let settings = ctx.settings.snapshot();
    let align_model = ctx
        .ui
        .upgrade()
        .and_then(|app| stack::sync_align_model(&app, ctx));
    let model = align_model
        .as_ref()
        .map_or_else(String::new, |(name, _)| name.clone());
    let sources = fingerprint_paths(&paths);
    // The Align pass's list: every file, so its cache serves any later subsample.
    let wanted = frames::paths_for_run(&paths, &settings, false);
    let reuse_aligned = ctx
        .state
        .borrow()
        .reusable_frames(&sources, &settings, &wanted)
        .is_some();
    let fingerprint = fingerprint_alignment_with(&sources, &settings, &model);
    let spec = JobSpec {
        kind: if action == SortCullAction::Sort {
            "sort"
        } else {
            "cull"
        },
        label: what.to_owned(),
        cancellable: true,
        blocking: true,
        plan: plan(),
        on_cancel: None,
    };
    let Some(job) = jobs::begin(ctx, spec) else {
        return;
    };
    let request = SortCullRequest {
        action,
        paths: wanted,
        settings: settings.clone(),
        reuse_aligned,
        align_model,
    };
    let mut run = RunUi {
        action,
        settings,
        fingerprint,
        answered: false,
    };
    let events = {
        let (ctx, job) = (ctx.clone(), Rc::clone(&job));
        Box::new(move |event| on_event(&ctx, &job, &mut run, event))
    };
    let handle = runner::current().run_sort_cull(request, events);
    job.attach(handle);
}

/// One event of the run, on the UI thread.
fn on_event(ctx: &Ctx, job: &Job, run: &mut RunUi, event: RunEvent) {
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
        RunEvent::AlignDone { paths, crop, .. } => {
            record_alignment(ctx, run, paths, crop);
            stale::refresh(&app, ctx);
        }
        RunEvent::SortCullDone {
            action,
            order,
            dropped,
        } => {
            run.answered = true;
            apply(&app, ctx, run, action, &order, &dropped);
        }
        // A Stack run's result does not come here.
        RunEvent::Result(_) => {}
        RunEvent::Finished(result) => {
            job.end(&app);
            finished(&app, result, run);
        }
    }
}

/// The run aligned the files itself: record the aligned set like an Align pass.
fn record_alignment(ctx: &Ctx, run: &RunUi, paths: Vec<std::path::PathBuf>, crop: Option<Crop>) {
    ctx.state
        .borrow_mut()
        .set_aligned(AlignedCache { paths, crop }, run.fingerprint);
}

/// Put the answer into the state and the sidebar.
fn apply(
    app: &App,
    ctx: &Ctx,
    run: &RunUi,
    action: SortCullAction,
    order: &[std::path::PathBuf],
    dropped: &[std::path::PathBuf],
) {
    match action {
        SortCullAction::Cull => {
            ctx.state.borrow_mut().set_culled(dropped.to_vec());
            stale::refresh(app, ctx);
            project::mark_dirty(ctx);
        }
        SortCullAction::Sort => {
            // The stack run reverses the list first (`sort_reverse`), so the list
            // is stored the other way round to come out in census order.
            let wanted: Vec<_> = if run.settings.preprocessing.sort_reverse {
                order.iter().rev().cloned().collect()
            } else {
                order.to_vec()
            };
            let list = {
                let mut state = ctx.state.borrow_mut();
                let list = frames::apply_order(&state.file_paths, &wanted);
                if list != state.file_paths {
                    state.file_paths.clone_from(&list);
                    let viewed = state
                        .viewed_path
                        .as_ref()
                        .and_then(|viewed| list.iter().position(|p| p == viewed));
                    state.view_idx = viewed;
                }
                list
            };
            push_source_panel(app, &list);
            stale::refresh(app, ctx);
            project::note_sources_changed(ctx);
        }
    }
}

/// The status line for a finished run.
fn finished(app: &App, result: Result<Option<String>, JobError>, run: &RunUi) {
    let what = run.action.label();
    match result {
        Ok(note) if run.answered => {
            let mut text = match run.action {
                SortCullAction::Sort => {
                    "Sort complete: the list is now in order of sharpness.".to_owned()
                }
                SortCullAction::Cull => {
                    "Cull complete: frames with too little sharp detail are dimmed and marked \
                     (culled)."
                        .to_owned()
                }
            };
            if let Some(note) = note {
                text = format!("{text} ({note})");
            }
            feedback::show(app, MessageKind::Success, &text);
        }
        Ok(_) => feedback::show(
            app,
            MessageKind::Warning,
            &format!("{what} produced no answer."),
        ),
        Err(error) => jobs::report_error(app, what, &error),
    }
}
