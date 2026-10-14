//! Jobs: the UI-side bookkeeping of a run that the [`crate::runner`] executes.
//!
//! A job has
//!
//! * a [`JobHandle`] the cancel button reaches (it is attached once the runner
//!   has started the run),
//! * a progress report ([`Job::progress`]) that posts the overall fraction and a
//!   status text to the window and to the activity registry (which feeds the ETA
//!   estimator), mapped through a [`StagePlan`],
//! * a registry entry (see [`super::activity`]) that the status strip shows as a
//!   chip, and that disables Run / Align while a *blocking* job runs.
//!
//! Everything here runs on the UI thread: the runner delivers its events there.
//! Whoever starts a job calls [`begin`], starts the run, [`Job::attach`]es the
//! handle, reports each event, and calls [`Job::end`] with the run's last one.
//! The job handles live in the registry's cancel closures (keyed by activity id),
//! not in `AppState`.

use std::{cell::RefCell, rc::Rc};

use super::{
    Ctx, activity,
    feedback::{self, MessageKind},
};
use crate::{
    App,
    jobs_core::{JobError, StagePlan},
    runner::JobHandle,
};

/// The highest fraction a running job reports: only finishing reaches 1.0, so the
/// bar never reads "done" before the result is published.
const MAX_RUNNING: f32 = 0.99;

/// What to start: set on the UI thread, so it may hold `Rc`s.
pub struct JobSpec {
    /// The activity kind (`"align"`, `"stack"`, `"monitor"`).
    pub kind: &'static str,
    /// The chip text.
    pub label: String,
    /// Whether the chip offers a cancel button.
    pub cancellable: bool,
    /// Whether the job disables Run / Align while it runs.
    pub blocking: bool,
    /// The stages the job reports through.
    pub plan: StagePlan,
    /// Extra work for a cancel request, run on the UI thread after the run was
    /// told to stop (the stack job uses it to close a `Relief` preview that waits
    /// for an answer).
    pub on_cancel: Option<Rc<dyn Fn()>>,
}

/// The run being tracked: its chip and its way to be cancelled.
pub struct Job {
    id: u64,
    plan: StagePlan,
    handle: Rc<RefCell<Option<Rc<dyn JobHandle>>>>,
}

/// Register a job in the activity registry (its chip appears, Run / Align are
/// disabled when it is blocking). `None` when the registry is missing (the user
/// has been told).
///
/// The caller then starts the run and gives the job its handle with
/// [`Job::attach`]; the chip's cancel button reaches the run through it.
#[must_use]
pub fn begin(ctx: &Ctx, spec: JobSpec) -> Option<Rc<Job>> {
    let Some(registry) = activity::current() else {
        feedback::notify(
            ctx,
            MessageKind::Error,
            "Internal error: the activity registry is not installed.",
        );
        return None;
    };
    let handle: Rc<RefCell<Option<Rc<dyn JobHandle>>>> = Rc::new(RefCell::new(None));
    let cancel_closure: Option<Rc<dyn Fn()>> = spec.cancellable.then(|| {
        let slot = Rc::clone(&handle);
        let extra = spec.on_cancel.clone();
        Rc::new(move || {
            // Release the borrow before cancelling: a run that ends at once may
            // finish its own job from inside `cancel`.
            let running = slot.borrow().clone();
            if let Some(running) = running {
                running.cancel();
            }
            if let Some(extra) = &extra {
                extra();
            }
        }) as Rc<dyn Fn()>
    });
    let id = registry.start(spec.kind, &spec.label, cancel_closure, spec.blocking);
    Some(Rc::new(Job {
        id,
        plan: spec.plan,
        handle,
    }))
}

impl Job {
    /// Give the job the handle of its run, so the cancel button can stop it.
    pub fn attach(&self, handle: Box<dyn JobHandle>) {
        *self.handle.borrow_mut() = Some(Rc::from(handle));
    }

    /// Report that `stage` of the job's [`StagePlan`] is `within` (0 to 1) done,
    /// with a status `text` (skipped when empty).
    pub fn progress(&self, app: &App, stage: usize, within: f32, text: &str) {
        let fraction = self.plan.fraction(stage, within).clamp(0.0, MAX_RUNNING);
        app.set_progress(fraction);
        if !text.is_empty() {
            feedback::show_status(app, text);
        }
        if let Some(registry) = activity::current() {
            registry.progress(self.id, fraction);
        }
    }

    /// The run is over (success, failure or cancel): finish the activity and
    /// reset the progress bar. Call it for the run's last event, before
    /// reporting its outcome.
    pub fn end(&self, app: &App) {
        if let Some(registry) = activity::current() {
            registry.finish(self.id);
        }
        app.set_progress(1.0);
    }
}

/// Refuse a second blocking run: tells the user with an `Info` toast and returns
/// `true` when a blocking job is already running. The Run / Align buttons are
/// disabled too; this guards the keyboard shortcuts and the monitor.
#[must_use]
pub fn refuse_if_busy(ctx: &Ctx, what: &str) -> bool {
    let Some(registry) = activity::current() else {
        return false;
    };
    if !registry.is_busy() {
        return false;
    }
    let running = registry
        .busy_label()
        .unwrap_or_else(|| "another job".to_owned());
    feedback::notify(
        ctx,
        MessageKind::Info,
        &format!("{what}: \"{running}\" is still running. Cancel it or wait for it to finish."),
    );
    true
}

/// Report a job that did not produce a result: an `Info` toast for a cancel, an
/// `Error` for a failure or a panic.
pub fn report_error(app: &App, what: &str, error: &JobError) {
    let kind = if error.is_cancel() {
        MessageKind::Info
    } else {
        MessageKind::Error
    };
    feedback::show(app, kind, &error.message(what));
}
