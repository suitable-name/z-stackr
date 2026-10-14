//! The window-free half of the job machinery: everything `gui::jobs` and
//! `gui::activity` need that does not touch Slint, so it can be unit-tested.
//!
//! * [`CancelFlag`] and [`JobError`]: how a job learns it should stop and how it
//!   reports why it ended.
//! * [`run_job`] / [`run_guarded`]: the thread body. The body runs inside
//!   `catch_unwind`, and the `finish` closure runs on every exit path, including a
//!   panic, so an activity can never stay registered after its worker died.
//! * [`StagePlan`]: weighted stages (load, align, cull, fuse, save) mapped onto one
//!   overall fraction, which is what the progress bar and the ETA see.
//! * [`ActivityList`] and [`CancelMap`]: the bookkeeping behind the activity chips.
//! * [`RunQueue`]: the folder monitor's queue of pending stack runs.

use std::{
    any::Any,
    collections::{BTreeMap, VecDeque},
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use web_time::Instant;

use crate::progress_eta::{EtaEstimator, format_eta};

// ── Cancellation and errors ─────────────────────────────────────────────────

/// A cancel request shared between the UI thread (which sets it) and a worker
/// (which polls it between units of work: frames, stages).
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    /// Ask the job to stop at its next checkpoint. Idempotent.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// `true` once [`Self::cancel`] was called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// The checkpoint: `Err(Cancelled(at))` when a cancel was requested, so a
    /// body can write `cancel.check("after loading")?;`.
    ///
    /// # Errors
    ///
    /// [`JobError::Cancelled`] once the flag is set.
    pub fn check(&self, at: &str) -> Result<(), JobError> {
        if self.is_cancelled() {
            Err(JobError::Cancelled(at.to_owned()))
        } else {
            Ok(())
        }
    }
}

/// Why a job did not produce its result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobError {
    /// The user cancelled; carries where the job noticed ("while aligning").
    Cancelled(String),
    /// The job failed with a message meant for the user.
    Failed(String),
    /// The job body panicked; carries the panic text.
    Panicked(String),
}

impl JobError {
    /// `true` for a cancel, which is a choice and not a failure.
    #[must_use]
    pub const fn is_cancel(&self) -> bool {
        matches!(self, Self::Cancelled(_))
    }

    /// The sentence shown to the user for a job called `what` ("Stack").
    #[must_use]
    pub fn message(&self, what: &str) -> String {
        match self {
            Self::Cancelled(at) if at.is_empty() => format!("{what} cancelled."),
            Self::Cancelled(at) => format!("{what} cancelled {at}."),
            Self::Failed(message) => message.clone(),
            Self::Panicked(text) => {
                format!("{what} stopped unexpectedly: {text}. Nothing was changed; see the log.")
            }
        }
    }
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message("The job"))
    }
}

impl std::error::Error for JobError {}

/// The readable text of a panic payload.
#[must_use]
pub fn panic_text(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a payload that is not text")
}

/// Run `body`, turning a panic into [`JobError::Panicked`] (and a log line).
///
/// # Errors
///
/// Whatever `body` returns, or [`JobError::Panicked`].
pub fn run_guarded<T>(body: impl FnOnce() -> Result<T, JobError>) -> Result<T, JobError> {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(result) => result,
        Err(payload) => {
            let text = panic_text(payload.as_ref()).to_owned();
            tracing::error!("a job panicked: {text}");
            Err(JobError::Panicked(text))
        }
    }
}

/// The thread body of a job: run `body` under [`run_guarded`], then hand the
/// result to `finish`, which therefore runs on every exit path.
pub fn run_job<T>(
    cancel: &CancelFlag,
    body: impl FnOnce(&CancelFlag) -> Result<T, JobError>,
    finish: impl FnOnce(Result<T, JobError>),
) {
    let result = run_guarded(|| body(cancel));
    finish(result);
}

// ── Stage plans ─────────────────────────────────────────────────────────────

/// Weighted stages mapped onto one overall fraction in `[0, 1]`.
#[derive(Debug, Clone, PartialEq)]
pub struct StagePlan {
    /// `(label, start, end)` per stage, cumulative and normalised.
    stages: Vec<(String, f32, f32)>,
}

impl StagePlan {
    /// A plan from `(label, weight)` pairs. Weights are relative (a zero weight is
    /// a stage that takes no time, e.g. alignment switched off); when every weight
    /// is zero the stages share the range equally.
    #[must_use]
    pub fn new(stages: &[(&str, f32)]) -> Self {
        let weights: Vec<f32> = stages.iter().map(|&(_, w)| w.max(0.0)).collect();
        let sum: f32 = weights.iter().sum();
        let count = f32::from(u16::try_from(stages.len()).unwrap_or(u16::MAX)).max(1.0);
        let mut start = 0.0_f32;
        let mut planned = Vec::with_capacity(stages.len());
        for (&(label, _), &weight) in stages.iter().zip(&weights) {
            let share = if sum > 0.0 { weight / sum } else { 1.0 / count };
            let end = (start + share).min(1.0);
            planned.push((label.to_owned(), start, end));
            start = end;
        }
        Self { stages: planned }
    }

    /// Number of stages.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.stages.len()
    }

    /// `true` for a plan without stages.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// The label of `stage`.
    #[must_use]
    pub fn label(&self, stage: usize) -> Option<&str> {
        self.stages.get(stage).map(|(label, _, _)| label.as_str())
    }

    /// The overall fraction when `stage` is `within` (clamped to `[0, 1]`) done.
    /// An unknown stage counts as the end of the plan.
    #[must_use]
    pub fn fraction(&self, stage: usize, within: f32) -> f32 {
        self.stages.get(stage).map_or(1.0, |&(_, start, end)| {
            (end - start).mul_add(within.clamp(0.0, 1.0), start)
        })
    }
}

/// `done` of `total` as a fraction in `[0, 1]`; `1.0` for an empty total.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "frame and tile counts are far below 2^24"
)]
pub fn fraction_of(done: usize, total: usize) -> f32 {
    if total == 0 {
        1.0
    } else {
        (done as f32 / total as f32).clamp(0.0, 1.0)
    }
}

// ── Activities ──────────────────────────────────────────────────────────────

/// `progress` value meaning "no fraction known" (matches `activity.slint`).
pub const INDETERMINATE: f32 = -1.0;

/// One running activity: everything an `ActivityItem` needs, plus its id.
#[derive(Debug, Clone)]
pub struct ActivityEntry {
    /// The id the registry gave it.
    pub id: u64,
    /// What kind of activity it is (`stack`, `align`, `monitor`).
    pub kind: String,
    /// The chip text.
    pub label: String,
    /// `INDETERMINATE` or a fraction in `[0, 1]`.
    pub progress: f32,
    /// Whether the chip offers a cancel button.
    pub cancellable: bool,
    /// A blocking activity keeps the Run / Align buttons disabled; the folder
    /// monitor's watcher is not one (the Stop button must stay usable).
    pub blocking: bool,
    /// Milliseconds on the registry's epoch, not wall-clock time.
    pub started_ms: i64,
    eta: EtaEstimator,
}

/// The ordered list of running activities. Entries stay in start order, which is
/// the order the status strip shows them in; ids are never reused.
#[derive(Debug, Default)]
pub struct ActivityList {
    next_id: u64,
    entries: Vec<ActivityEntry>,
}

impl ActivityList {
    /// Register a new activity and return its id: always one more than the last
    /// id handed out, even across [`Self::finish`] calls.
    pub fn start(
        &mut self,
        kind: &str,
        label: &str,
        cancellable: bool,
        blocking: bool,
        started_ms: i64,
    ) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        self.entries.push(ActivityEntry {
            id,
            kind: kind.to_owned(),
            label: label.to_owned(),
            progress: INDETERMINATE,
            cancellable,
            blocking,
            started_ms,
            eta: EtaEstimator::default(),
        });
        id
    }

    /// Set `id`'s progress and feed its ETA estimator. `false` (a harmless no-op)
    /// when `id` already finished: a tick racing a completion is expected.
    pub fn set_progress(&mut self, id: u64, progress: f32, now: Instant) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) else {
            return false;
        };
        entry.progress = progress;
        if progress >= 0.0 {
            entry.eta.observe(now, f64::from(progress));
        }
        true
    }

    /// Replace `id`'s label. `false` when `id` is not registered.
    pub fn set_label(&mut self, id: u64, label: &str) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) else {
            return false;
        };
        label.clone_into(&mut entry.label);
        true
    }

    /// Mark `id` as cancelling: it keeps running until its worker notices, but
    /// the chip loses its cancel affordance and says so. `false` when unknown.
    pub fn mark_cancelling(&mut self, id: u64) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) else {
            return false;
        };
        entry.cancellable = false;
        entry.label = format!("{} (cancelling…)", entry.label);
        true
    }

    /// Remove `id`. `false` when it was already gone.
    pub fn finish(&mut self, id: u64) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != id);
        self.entries.len() != before
    }

    /// The entries in start order.
    #[must_use]
    pub fn entries(&self) -> &[ActivityEntry] {
        &self.entries
    }

    /// `true` when nothing is running.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many running activities block the Run / Align buttons.
    #[must_use]
    pub fn blocking_count(&self) -> usize {
        self.entries.iter().filter(|e| e.blocking).count()
    }

    /// The label of the first blocking activity, for the "already running" toast.
    #[must_use]
    pub fn first_blocking_label(&self) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.blocking)
            .map(|e| e.label.as_str())
    }

    /// `id`'s "about N left" text at `now`; empty when there is no estimate.
    #[must_use]
    pub fn eta_text(&self, id: u64, now: Instant) -> String {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .map_or_else(String::new, |e| format_eta(e.eta.eta(now)))
    }
}

/// Which closure cancelling activity N invokes.
///
/// A `BTreeMap`, so nothing depends on hash iteration order. Closures are `Rc`:
/// they live on the UI thread, and
/// [`Self::get`] hands out a clone so the caller can release its borrow of the
/// map before running one (a cancel closure may finish its own activity).
#[derive(Default)]
pub struct CancelMap {
    entries: BTreeMap<u64, Rc<dyn Fn()>>,
}

impl CancelMap {
    /// Register `cancel` for `id`, replacing any earlier one.
    pub fn insert(&mut self, id: u64, cancel: Rc<dyn Fn()>) {
        self.entries.insert(id, cancel);
    }

    /// Forget `id`'s closure, so a stray later cancel cannot reach a job that is
    /// gone.
    pub fn remove(&mut self, id: u64) {
        self.entries.remove(&id);
    }

    /// `id`'s closure, if registered.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<Rc<dyn Fn()>> {
        self.entries.get(&id).cloned()
    }

    /// Run `id`'s closure once; a no-op for an unknown id. `true` when one ran.
    #[must_use]
    pub fn invoke(&self, id: u64) -> bool {
        let Some(cancel) = self.get(id) else {
            return false;
        };
        cancel();
        true
    }
}

// ── The monitor's run queue ─────────────────────────────────────────────────

/// The label of the monitor's activity chip for `queued` pending runs and
/// whether a run is executing right now.
#[must_use]
pub fn monitor_label(queued: usize, running: bool) -> String {
    if queued > 0 {
        format!("Monitor: {queued} queued")
    } else if running {
        "Monitor: stacking".to_owned()
    } else {
        "Monitor: watching".to_owned()
    }
}

/// Pending stack runs of the folder monitor.
///
/// One entry per folder change, each holding the file list that change produced.
/// A scan identical to the newest pending one is dropped (a burst of events for
/// one write), and an empty scan is never queued.
#[derive(Debug, Default)]
pub struct RunQueue {
    pending: VecDeque<Vec<PathBuf>>,
}

impl RunQueue {
    /// Queue a run for `files`. `false` when it was empty or a duplicate of the
    /// newest pending entry.
    pub fn push(&mut self, files: Vec<PathBuf>) -> bool {
        if files.is_empty() || self.pending.back() == Some(&files) {
            return false;
        }
        self.pending.push_back(files);
        true
    }

    /// The oldest pending run.
    pub fn pop(&mut self) -> Option<Vec<PathBuf>> {
        self.pending.pop_front()
    }

    /// How many runs are waiting.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// `true` when nothing waits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Drop every pending run; returns how many there were.
    pub fn drain(&mut self) -> usize {
        let dropped = self.pending.len();
        self.pending.clear();
        dropped
    }
}
