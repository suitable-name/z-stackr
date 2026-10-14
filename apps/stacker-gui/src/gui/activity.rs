//! The activity registry: the one place every background operation registers
//! itself while it runs, instead of one busy flag per feature.
//!
//! The registry is the only writer of `ActivityModel`: it keeps the pure
//! bookkeeping ([`ActivityList`], [`CancelMap`], both in the library crate so they
//! are unit-tested), pushes the rows into `ActivityModel.activities` and ticks
//! `now_ms` once a second, but only while something is running.
//!
//! It lives in a UI-thread thread-local installed by [`setup`]; workers reach it
//! only through [`super::jobs`], which posts to the event loop.
//!
//! `started_ms` and `now_ms` are measured from a per-registry epoch (an `Instant`
//! taken at start-up), not wall-clock time: Slint `int` is 32-bit and milliseconds
//! since 1970 would overflow it.

use std::{
    cell::RefCell,
    rc::{Rc, Weak as RcWeak},
    time::Duration,
};

use slint::{ComponentHandle, ModelRc, Timer, TimerMode, VecModel, Weak};
use stacker_gui::{
    jobs_core::{ActivityList, CancelMap},
    stall_guard::stall_guard,
};
use web_time::Instant;

use super::Ctx;
use crate::{ActivityItem, ActivityModel, App};

/// How often the elapsed time and the ETA refresh while something runs.
const TICK_INTERVAL: Duration = Duration::from_secs(1);

thread_local! {
    static REGISTRY: RefCell<Option<Rc<ActivityRegistry>>> = const { RefCell::new(None) };
}

/// The registry [`setup`] installed on this (the UI) thread.
#[must_use]
pub fn current() -> Option<Rc<ActivityRegistry>> {
    REGISTRY.with(|slot| slot.borrow().clone())
}

/// Create the registry and wire `ActivityModel.cancel(id)` to it.
pub fn setup(ui: &App, _ctx: &Ctx) {
    let registry = ActivityRegistry::new(ui);
    ui.global::<ActivityModel>().on_cancel({
        let registry = Rc::clone(&registry);
        move |id| {
            stall_guard("activity::cancel", || {
                if let Ok(id) = u64::try_from(id) {
                    registry.request_cancel(id);
                }
            });
        }
    });
    REGISTRY.with(|slot| *slot.borrow_mut() = Some(registry));
}

/// The running activities and their cancel closures; see the module docs.
pub struct ActivityRegistry {
    list: RefCell<ActivityList>,
    cancels: RefCell<CancelMap>,
    ui: Weak<App>,
    epoch: Instant,
    ticker: Timer,
    /// Called (deferred, from a zero-length timer) when the last blocking
    /// activity finishes: the folder monitor starts its next queued run there.
    idle_hook: RefCell<Option<Rc<dyn Fn()>>>,
    /// The ticker closure holds this instead of an `Rc` to `self`, which would be
    /// a permanent cycle (the timer is a field of the registry).
    self_weak: RcWeak<Self>,
}

impl ActivityRegistry {
    fn new(ui: &App) -> Rc<Self> {
        Rc::new_cyclic(|weak_self| Self {
            list: RefCell::new(ActivityList::default()),
            cancels: RefCell::new(CancelMap::default()),
            ui: ui.as_weak(),
            epoch: Instant::now(),
            ticker: Timer::default(),
            idle_hook: RefCell::new(None),
            self_weak: weak_self.clone(),
        })
    }

    fn elapsed_ms(&self) -> i64 {
        i64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(i64::MAX)
    }

    /// Register an activity and return its id. `cancel` is `None` for an
    /// activity that cannot be cancelled (no cancel affordance on its chip).
    /// A `blocking` activity disables the Run / Align buttons while it runs.
    pub fn start(
        &self,
        kind: &str,
        label: &str,
        cancel: Option<Rc<dyn Fn()>>,
        blocking: bool,
    ) -> u64 {
        let started_ms = self.elapsed_ms();
        let was_empty = self.list.borrow().is_empty();
        let id = self
            .list
            .borrow_mut()
            .start(kind, label, cancel.is_some(), blocking, started_ms);
        if let Some(cancel) = cancel {
            self.cancels.borrow_mut().insert(id, cancel);
        }
        self.sync_model();
        if was_empty {
            self.start_ticker();
        }
        id
    }

    /// Set `id`'s progress in `[0, 1]` and feed its ETA estimator.
    pub fn progress(&self, id: u64, fraction: f32) {
        if self
            .list
            .borrow_mut()
            .set_progress(id, fraction, Instant::now())
        {
            self.sync_model();
        }
    }

    /// Replace the chip text of `id` (the monitor's "N queued").
    #[cfg_attr(
        target_arch = "wasm32",
        expect(
            dead_code,
            reason = "only the desktop's folder monitor relabels a chip"
        )
    )]
    pub fn set_label(&self, id: u64, label: &str) {
        if self.list.borrow_mut().set_label(id, label) {
            self.sync_model();
        }
    }

    /// Remove `id` and drop its cancel closure. Stops the ticker once nothing is
    /// left, and schedules the idle hook when the last blocking activity ended.
    pub fn finish(&self, id: u64) {
        self.cancels.borrow_mut().remove(id);
        let blocking_before = self.list.borrow().blocking_count();
        if self.list.borrow_mut().finish(id) {
            self.sync_model();
        }
        if self.list.borrow().is_empty() {
            self.ticker.stop();
        }
        if blocking_before > 0 && self.list.borrow().blocking_count() == 0 {
            let hook = self.idle_hook.borrow().clone();
            if let Some(hook) = hook {
                Timer::single_shot(Duration::ZERO, move || hook());
            }
        }
    }

    /// The user pressed the cancel affordance of `id`: run its closure (outside
    /// every borrow, the closure may finish its own activity) and show the chip as
    /// cancelling until the worker notices.
    pub fn request_cancel(&self, id: u64) {
        let cancel = self.cancels.borrow().get(id);
        let Some(cancel) = cancel else {
            return;
        };
        cancel();
        if self.list.borrow_mut().mark_cancelling(id) {
            self.sync_model();
        }
    }

    /// `true` while a blocking activity runs.
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.list.borrow().blocking_count() > 0
    }

    /// The label of the blocking activity that is running, if any.
    #[must_use]
    pub fn busy_label(&self) -> Option<String> {
        self.list.borrow().first_blocking_label().map(str::to_owned)
    }

    /// Install the callback run when the last blocking activity finishes.
    #[cfg_attr(
        target_arch = "wasm32",
        expect(dead_code, reason = "only the desktop's folder monitor hooks the idle")
    )]
    pub fn set_idle_hook(&self, hook: Rc<dyn Fn()>) {
        *self.idle_hook.borrow_mut() = Some(hook);
    }

    fn start_ticker(&self) {
        let registry = self.self_weak.clone();
        self.ticker
            .start(TimerMode::Repeated, TICK_INTERVAL, move || {
                let Some(registry) = registry.upgrade() else {
                    return;
                };
                if registry.list.borrow().is_empty() {
                    registry.ticker.stop();
                } else {
                    registry.sync_model();
                }
            });
    }

    /// Rebuild `ActivityModel` from the list. There are at most a handful of
    /// activities, so replacing the whole model on every change is cheap.
    fn sync_model(&self) {
        let Some(ui) = self.ui.upgrade() else {
            return;
        };
        let now = Instant::now();
        let list = self.list.borrow();
        let items: Vec<ActivityItem> = list
            .entries()
            .iter()
            .map(|entry| ActivityItem {
                id: i32::try_from(entry.id).unwrap_or(i32::MAX),
                kind: entry.kind.as_str().into(),
                label: entry.label.as_str().into(),
                progress: entry.progress,
                cancellable: entry.cancellable,
                started_ms: i32::try_from(entry.started_ms).unwrap_or(i32::MAX),
                eta_text: list.eta_text(entry.id, now).into(),
            })
            .collect();
        let blocking = i32::try_from(list.blocking_count()).unwrap_or(i32::MAX);
        drop(list);
        let model = ui.global::<ActivityModel>();
        model.set_activities(ModelRc::new(VecModel::from(items)));
        model.set_now_ms(i32::try_from(self.elapsed_ms()).unwrap_or(i32::MAX));
        model.set_blocking_count(blocking);
    }
}
