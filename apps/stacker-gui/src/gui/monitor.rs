//! Watch a folder: every change enqueues one stack run.
//!
//! The monitor is a queued job. Its watcher thread is registered as an activity
//! ("Monitor: watching", "Monitor: 2 queued") that is *not* blocking, so the Stop
//! button stays usable; each folder change posts the new file list to the UI
//! thread, which queues it in a [`RunQueue`]. One stack run at a time executes (a
//! normal blocking `stack` job, started by [`pump`] whenever the previous one
//! ended), so a burst of changes never starts overlapping runs.
//!
//! Cancelling the monitor's activity (or pressing Stop Monitor) stops the watcher
//! and drains the queue; a run already executing is not interrupted, it has its
//! own chip to cancel.
//!
//! The state is a UI-thread thread-local, not part of `AppState`.

use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    sync::mpsc,
    time::Duration,
};

use notify::{EventKind, Watcher};
use slint::Weak;
use stacker_gui::{
    jobs_core::{CancelFlag, JobError, RunQueue, monitor_label, run_guarded},
    pickers::{PickerKind, PickerRequest},
    stall_guard::stall_guard,
};
use stacker_pipeline::order::collect_image_paths;

use super::{
    Ctx,
    activity::{self, ActivityRegistry},
    ctx::on_ui_ctx,
    feedback::{self, MessageKind},
    jobs, pickers, stack,
};
use crate::{App, ui_helpers::push_source_panel};

/// How long the watcher waits for further events before polling the stop flag.
const POLL: Duration = Duration::from_millis(500);

/// The running monitor, if any.
struct MonitorState {
    /// The id of the monitor's activity; also identifies the watcher thread's
    /// posts, so a stale post from a replaced monitor is ignored.
    activity: u64,
    /// Stops the watcher thread.
    stop: CancelFlag,
    /// Folder changes waiting for their stack run.
    queue: RunQueue,
    /// A queued run is executing right now.
    run_active: bool,
}

thread_local! {
    static MONITOR: RefCell<Option<MonitorState>> = const { RefCell::new(None) };
}

/// Register the Start and Stop Monitor callbacks and the idle hook that starts
/// the next queued run.
pub fn setup(ui: &App, ctx: &Ctx) {
    // A new window watches nothing yet.
    MONITOR.with(|slot| *slot.borrow_mut() = None);
    wire_start(ui, ctx);
    wire_stop(ui, ctx);
    // Start the next queued run as soon as the last blocking job ended.
    if let Some(registry) = activity::current() {
        let ctx = ctx.clone();
        registry.set_idle_hook(Rc::new(move || pump(&ctx)));
    }
}

fn wire_start(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_start_monitor(move || {
        stall_guard("monitor::start-monitor", || {
            let request = PickerRequest::new(PickerKind::PickFolder).titled("Folder to watch");
            pickers::pick(&ctx, request, |app, ctx, picked| {
                if let Some(folder) = picked.into_iter().next() {
                    start(app, ctx, folder);
                }
            });
        });
    });
}

fn wire_stop(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_stop_monitor(move || {
        stall_guard("monitor::stop-monitor", || stop_with(&ctx, true));
    });
}

fn start(app: &App, ctx: &Ctx, folder: PathBuf) {
    if jobs::refuse_if_busy(ctx, "Monitor") {
        return;
    }
    // Starting a monitor replaces a running one.
    stop_with(ctx, false);
    let Some(registry) = activity::current() else {
        return;
    };
    let stop = CancelFlag::default();
    let cancel: Rc<dyn Fn()> = {
        let ctx = ctx.clone();
        Rc::new(move || stop_with(&ctx, true))
    };
    let id = registry.start("monitor", &monitor_label(0, false), Some(cancel), false);
    MONITOR.with(|slot| {
        *slot.borrow_mut() = Some(MonitorState {
            activity: id,
            stop: stop.clone(),
            queue: RunQueue::default(),
            run_active: false,
        });
    });
    app.set_is_monitoring(true);

    let ui = ctx.ui.clone();
    let spawned = std::thread::Builder::new()
        .name("z-stackr-monitor".to_owned())
        .spawn(move || {
            // A panic in the watcher is reported like any other failure.
            let outcome = run_guarded(|| watch_folder(&ui, id, &folder, &stop));
            if let Err(error) = outcome {
                on_ui_ctx(&ui, move |app, ctx| watcher_failed(app, ctx, id, &error));
            }
        });
    if let Err(e) = spawned {
        stop_with(ctx, false);
        feedback::notify(
            ctx,
            MessageKind::Error,
            &format!("Could not start the folder monitor: {e}"),
        );
    }
}

/// Stop the monitor: the watcher ends, the queue is drained, the chip goes away.
/// `announce` tells the user (and says how many queued runs were dropped).
fn stop_with(ctx: &Ctx, announce: bool) {
    let state = MONITOR.with(|slot| slot.borrow_mut().take());
    let Some(mut state) = state else {
        return;
    };
    state.stop.cancel();
    let dropped = state.queue.drain();
    if let Some(registry) = activity::current() {
        registry.finish(state.activity);
    }
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    app.set_is_monitoring(false);
    if announce {
        let text = if dropped > 0 {
            format!("Monitor stopped; {dropped} queued run(s) dropped.")
        } else {
            "Monitor stopped.".to_owned()
        };
        feedback::show(&app, MessageKind::Info, &text);
    }
}

/// `true` while the monitor with activity `id` is the running one.
fn is_current(id: u64) -> bool {
    MONITOR.with(|slot| slot.borrow().as_ref().is_some_and(|m| m.activity == id))
}

/// The watcher thread failed or panicked.
fn watcher_failed(app: &App, ctx: &Ctx, id: u64, error: &JobError) {
    if !is_current(id) {
        return;
    }
    stop_with(ctx, false);
    feedback::show(app, MessageKind::Error, &error.message("Monitor"));
}

/// A stack run the monitor started has ended (success, failure or cancel).
pub fn run_finished(ctx: &Ctx) {
    MONITOR.with(|slot| {
        if let Some(monitor) = slot.borrow_mut().as_mut() {
            monitor.run_active = false;
        }
    });
    if let Some(registry) = activity::current() {
        update_label(&registry);
    }
    pump(ctx);
}

/// Show the queue length on the monitor's chip.
fn update_label(registry: &ActivityRegistry) {
    let snapshot = MONITOR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|m| (m.activity, m.queue.len(), m.run_active))
    });
    if let Some((id, queued, running)) = snapshot {
        registry.set_label(id, &monitor_label(queued, running));
    }
}

/// Start the oldest queued run when nothing else is running.
fn pump(ctx: &Ctx) {
    let Some(registry) = activity::current() else {
        return;
    };
    if registry.is_busy() {
        return;
    }
    let next = MONITOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        let monitor = slot.as_mut()?;
        if monitor.run_active {
            return None;
        }
        let files = monitor.queue.pop()?;
        monitor.run_active = true;
        Some(files)
    });
    let Some(files) = next else {
        return;
    };
    update_label(&registry);
    ctx.state.borrow_mut().replace_paths(files.clone());
    if let Some(app) = ctx.ui.upgrade() {
        push_source_panel(&app, &files);
    }
    super::project::note_sources_changed(ctx);
    if !stack::start_monitor_run(ctx, &files) {
        // Not started (the user was told): do not wedge the queue.
        MONITOR.with(|slot| {
            if let Some(monitor) = slot.borrow_mut().as_mut() {
                monitor.run_active = false;
            }
        });
        update_label(&registry);
    }
}

/// The frames directly inside `folder` in natural file-name order, without the
/// app's own temporary result images (the pipeline's `collect_image_paths`, the
/// scan the CLI uses). An unreadable or image-less folder is an empty list: the
/// monitor keeps waiting.
fn scan_folder(folder: &Path) -> Vec<PathBuf> {
    collect_image_paths(folder, &[]).unwrap_or_default()
}

/// A folder scan reached the UI thread: queue a run for it (an empty folder just
/// clears the list and says so).
fn on_scan(app: &App, ctx: &Ctx, id: u64, files: Vec<PathBuf>) {
    if !is_current(id) {
        return;
    }
    if files.is_empty() {
        ctx.state.borrow_mut().replace_paths(Vec::new());
        push_source_panel(app, &[]);
        super::project::note_sources_changed(ctx);
        feedback::show(
            app,
            MessageKind::Info,
            "The watched folder has no images yet; waiting for new files.",
        );
        return;
    }
    MONITOR.with(|slot| {
        if let Some(monitor) = slot.borrow_mut().as_mut() {
            monitor.queue.push(files);
        }
    });
    if let Some(registry) = activity::current() {
        update_label(&registry);
    }
    pump(ctx);
}

/// Hand a folder scan to the UI thread.
fn post_scan(ui: &Weak<App>, id: u64, files: Vec<PathBuf>) {
    on_ui_ctx(ui, move |app, ctx| on_scan(app, ctx, id, files));
}

/// The watcher thread: scan once, then rescan whenever the folder changes, until
/// the stop flag is set (or a newer monitor replaces this one).
fn watch_folder(ui: &Weak<App>, id: u64, folder: &Path, stop: &CancelFlag) -> Result<(), JobError> {
    let (notify_tx, notify_rx) = mpsc::channel();
    let fail =
        |e: notify::Error| JobError::Failed(format!("Could not watch {}: {e}", folder.display()));
    let mut watcher = notify::recommended_watcher(notify_tx).map_err(fail)?;
    watcher
        .watch(folder, notify::RecursiveMode::NonRecursive)
        .map_err(fail)?;

    post_scan(ui, id, scan_folder(folder));
    while !stop.is_cancelled() {
        if let Ok(Ok(notify::Event {
            kind: EventKind::Create(_) | EventKind::Modify(_),
            ..
        })) = notify_rx.recv_timeout(POLL)
        {
            // Let a burst of writes settle, then rescan once.
            std::thread::sleep(POLL);
            while notify_rx.try_recv().is_ok() {}
            if stop.is_cancelled() {
                break;
            }
            post_scan(ui, id, scan_folder(folder));
        }
    }
    Ok(())
}
