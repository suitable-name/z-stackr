//! Debounced background settings writer.
//!
//! UI callbacks call [`SettingsPersister::update`]; the in-memory copy changes at
//! once and the disk write happens on a worker thread once the changes have been
//! quiet for the debounce interval, so a slider drag collapses into one write and
//! the UI thread never waits for the disk. [`SettingsPersister::flush`] writes any
//! pending change right now (called when the window closes).
//!
//! The in-memory copy is the only source of truth for what gets written: every
//! write of the settings file goes through this type, because a flush writes its
//! copy over whatever is on disk.
//!
//! A persister built with [`SettingsPersister::stored`] (the browser build, which
//! has no file and no thread) writes through the platform's key-value store
//! instead, on the UI thread, with the same debounce kept by a Slint timer.

use std::{
    cell::RefCell,
    path::PathBuf,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use slint::{Timer, TimerMode};
use stacker_core::settings::StackingSettings;
use web_time::Instant;

use super::{SETTINGS_KEY, store};
use crate::{platform, sync::lock};

/// Quiet time after the last change before the file is written.
pub const DEBOUNCE: Duration = Duration::from_millis(600);

/// How long [`SettingsPersister::flush`] waits for the worker. A settings file
/// is a few kilobytes, so this only matters when the disk is wedged; it bounds
/// how long closing the window can be held up.
const FLUSH_WAIT: Duration = Duration::from_secs(3);

enum Msg {
    Changed(Box<StackingSettings>),
    /// Write any pending change now, then signal on the sender.
    Flush(mpsc::Sender<()>),
    Shutdown,
}

thread_local! {
    /// The persister UI-thread code without a handle of its own reaches through
    /// [`SettingsPersister::installed_for_this_thread`]. Held weakly: the
    /// application's own `Arc` decides when the worker shuts down.
    static INSTALLED: RefCell<Weak<SettingsPersister>> = const { RefCell::new(Weak::new()) };
    /// The debounce timer of a persister built with [`SettingsPersister::stored`].
    static STORE_TIMER: Timer = Timer::default();
}

/// Handle to the settings copy and its background writer. Keep it in an `Arc`;
/// dropping the last handle asks the worker to write what is pending and stop.
pub struct SettingsPersister {
    /// `None` for a persister that never writes ([`Self::disabled`]) or that
    /// writes through the platform store ([`Self::stored`]).
    sender: Option<mpsc::Sender<Msg>>,
    /// `true` for a persister that writes through the platform store.
    stored: bool,
    /// A change the platform store has not seen yet.
    dirty: Arc<AtomicBool>,
    current: Arc<Mutex<StackingSettings>>,
    writes: Arc<AtomicUsize>,
    last_error: Arc<Mutex<Option<String>>>,
    /// The writer thread, kept so the UI harness can join it ([`Self::stop`]).
    #[cfg(feature = "ui-harness")]
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

/// Write the settings in `current` to the platform store, once.
fn write_stored(
    current: &Mutex<StackingSettings>,
    dirty: &AtomicBool,
    writes: &AtomicUsize,
    last_error: &Mutex<Option<String>>,
) {
    dirty.store(false, Ordering::Relaxed);
    let snapshot = lock(current).clone();
    let outcome = toml::to_string_pretty(&snapshot)
        .map_err(|e| e.to_string())
        .and_then(|text| platform::current().store_value(SETTINGS_KEY, &text));
    match outcome {
        Ok(()) => {
            writes.fetch_add(1, Ordering::Relaxed);
        }
        Err(e) => {
            tracing::warn!("could not store the settings: {e}");
            *lock(last_error) = Some(format!("Settings could not be saved: {e}"));
        }
    }
}

impl SettingsPersister {
    /// Start the worker for `path`, seeded with `initial`.
    #[must_use]
    pub fn spawn(path: PathBuf, initial: StackingSettings) -> Self {
        Self::spawn_with_debounce(path, initial, DEBOUNCE)
    }

    /// [`Self::spawn`] with an explicit debounce interval (tests use a short one).
    #[must_use]
    pub fn spawn_with_debounce(
        path: PathBuf,
        initial: StackingSettings,
        debounce: Duration,
    ) -> Self {
        let (sender, receiver) = mpsc::channel::<Msg>();
        let writes = Arc::new(AtomicUsize::new(0));
        let last_error = Arc::new(Mutex::new(None));
        let worker = Worker {
            path,
            debounce,
            writes: Arc::clone(&writes),
            last_error: Arc::clone(&last_error),
        };
        #[cfg(feature = "ui-harness")]
        let thread = Mutex::new(Some(thread::spawn(move || worker.run(&receiver))));
        #[cfg(not(feature = "ui-harness"))]
        thread::spawn(move || worker.run(&receiver));
        Self {
            sender: Some(sender),
            stored: false,
            dirty: Arc::new(AtomicBool::new(false)),
            current: Arc::new(Mutex::new(initial)),
            writes,
            last_error,
            #[cfg(feature = "ui-harness")]
            thread,
        }
    }

    /// A persister that keeps the settings in memory and never touches the disk:
    /// used while the settings file is one we could not understand and could not
    /// move aside, so nothing overwrites it.
    #[must_use]
    pub fn disabled(initial: StackingSettings) -> Self {
        Self {
            sender: None,
            stored: false,
            dirty: Arc::new(AtomicBool::new(false)),
            current: Arc::new(Mutex::new(initial)),
            writes: Arc::new(AtomicUsize::new(0)),
            last_error: Arc::new(Mutex::new(None)),
            #[cfg(feature = "ui-harness")]
            thread: Mutex::new(None),
        }
    }

    /// A persister that writes through the platform's key-value store
    /// ([`crate::platform::Platform::store_value`], key `settings.toml`) instead
    /// of a file. It starts no thread: [`Self::update`] and [`Self::flush`] must
    /// be called on the UI thread, and the debounce is a Slint timer.
    #[must_use]
    pub fn stored(initial: StackingSettings) -> Self {
        Self {
            sender: None,
            stored: true,
            dirty: Arc::new(AtomicBool::new(false)),
            current: Arc::new(Mutex::new(initial)),
            writes: Arc::new(AtomicUsize::new(0)),
            last_error: Arc::new(Mutex::new(None)),
            #[cfg(feature = "ui-harness")]
            thread: Mutex::new(None),
        }
    }

    /// `true` when changes are written to disk (or to the platform store).
    #[must_use]
    pub const fn writes_enabled(&self) -> bool {
        self.sender.is_some() || self.stored
    }

    /// Change the in-memory settings under `f` and schedule a debounced save.
    /// [`Self::snapshot`] reflects the change at once; only the write is deferred.
    pub fn update(&self, f: impl FnOnce(&mut StackingSettings)) {
        let snapshot = {
            let mut guard = lock(&self.current);
            f(&mut guard);
            guard.clone()
        };
        if let Some(sender) = &self.sender {
            let _ = sender.send(Msg::Changed(Box::new(snapshot)));
        } else if self.stored {
            self.schedule_stored_write();
        }
    }

    /// Mark the store stale and (re)start the debounce timer that writes it.
    fn schedule_stored_write(&self) {
        self.dirty.store(true, Ordering::Relaxed);
        let current = Arc::clone(&self.current);
        let dirty = Arc::clone(&self.dirty);
        let writes = Arc::clone(&self.writes);
        let last_error = Arc::clone(&self.last_error);
        STORE_TIMER.with(|timer| {
            // Restarting the timer replaces the pending write.
            timer.start(TimerMode::SingleShot, DEBOUNCE, move || {
                write_stored(&current, &dirty, &writes, &last_error);
            });
        });
    }

    /// A copy of the current settings: what the next write will contain.
    #[must_use]
    pub fn snapshot(&self) -> StackingSettings {
        lock(&self.current).clone()
    }

    /// Write any pending change now and return once the worker has done so (or
    /// after a bounded wait). Call when the window closes.
    pub fn flush(&self) {
        if self.stored {
            if self.dirty.load(Ordering::Relaxed) {
                STORE_TIMER.with(Timer::stop);
                write_stored(&self.current, &self.dirty, &self.writes, &self.last_error);
            }
            return;
        }
        let Some(sender) = &self.sender else {
            return;
        };
        let (ack_sender, ack_receiver) = mpsc::channel();
        if sender.send(Msg::Flush(ack_sender)).is_ok() {
            let _ = ack_receiver.recv_timeout(FLUSH_WAIT);
        }
    }

    /// Write any pending change, end the writer thread and wait for it. The
    /// persister keeps its in-memory copy; later changes are no longer written.
    /// For the UI harness's teardown, which must not leave a thread that can
    /// still write into a profile it is about to delete.
    #[cfg(feature = "ui-harness")]
    pub fn stop(&self) {
        self.flush();
        if let Some(sender) = &self.sender {
            let _ = sender.send(Msg::Shutdown);
        }
        let thread = lock(&self.thread).take();
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }

    /// How many times the settings file has been written.
    #[must_use]
    pub fn write_count(&self) -> usize {
        self.writes.load(Ordering::Relaxed)
    }

    /// The message of the last failed write, once (it is cleared by this call).
    #[must_use]
    pub fn take_error(&self) -> Option<String> {
        lock(&self.last_error).take()
    }

    /// Register `handle` as this thread's ambient persister, for UI-thread code
    /// without a handle of its own. Replaces any handle installed earlier.
    pub fn install_for_this_thread(handle: &Arc<Self>) {
        INSTALLED.with(|slot| *slot.borrow_mut() = Arc::downgrade(handle));
    }

    /// The handle [`Self::install_for_this_thread`] registered on this thread, if
    /// it is still alive.
    #[must_use]
    pub fn installed_for_this_thread() -> Option<Arc<Self>> {
        INSTALLED.with(|slot| slot.borrow().upgrade())
    }
}

impl Drop for SettingsPersister {
    fn drop(&mut self) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Msg::Shutdown);
        }
    }
}

/// The state of the writer thread.
struct Worker {
    path: PathBuf,
    debounce: Duration,
    writes: Arc<AtomicUsize>,
    last_error: Arc<Mutex<Option<String>>>,
}

impl Worker {
    fn run(&self, receiver: &mpsc::Receiver<Msg>) {
        let mut pending: Option<StackingSettings> = None;
        let mut deadline = Instant::now();
        loop {
            let message = if pending.is_some() {
                receiver.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            } else {
                receiver
                    .recv()
                    .map_err(|_| mpsc::RecvTimeoutError::Disconnected)
            };
            match message {
                Ok(Msg::Changed(settings)) => {
                    pending = Some(*settings);
                    deadline = Instant::now() + self.debounce;
                }
                Ok(Msg::Flush(ack)) => {
                    if let Some(settings) = pending.take() {
                        self.write(&settings);
                    }
                    let _ = ack.send(());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(settings) = pending.take() {
                        self.write(&settings);
                    }
                }
                Ok(Msg::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    if let Some(settings) = pending.take() {
                        self.write(&settings);
                    }
                    break;
                }
            }
        }
    }

    fn write(&self, settings: &StackingSettings) {
        match store::save_atomic(&self.path, settings) {
            Ok(()) => {
                self.writes.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                tracing::warn!("could not save settings to {}: {e}", self.path.display());
                *lock(&self.last_error) = Some(format!(
                    "Settings could not be saved to {}: {e}",
                    self.path.display()
                ));
            }
        }
    }
}
