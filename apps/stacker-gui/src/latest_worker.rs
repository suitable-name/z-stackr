//! A single background thread fed through a one-slot "latest request wins"
//! mailbox.
//!
//! A UI event that arrives faster than its work can run (a slider drag, a burst
//! of list clicks) must not queue one job per event: only the newest value
//! matters. [`LatestWorker::submit`] replaces whatever has not started yet and
//! wakes the thread, so the work done is bounded by the work that fits, not by
//! the number of events.
//!
//! # Hosts without threads
//!
//! On wasm32 there is no thread to start: [`LatestWorker::spawn`] creates none and
//! [`LatestWorker::submit`] runs the handler inline, on the calling thread, before
//! it returns (a request submitted from inside the handler runs right after it).
//! [`spawn_or_inline`] does the same for a one-off job.

use std::sync::{Arc, Mutex, PoisonError};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::mpsc,
};

#[cfg(not(target_arch = "wasm32"))]
use tracing::error;

/// Run `work` on its own thread named `name`; on a host without threads (wasm32)
/// run it right here instead. A thread that cannot be started is logged and the
/// work is dropped.
pub fn spawn_or_inline(name: &str, work: impl FnOnce() + Send + 'static) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Err(spawn_error) = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(work)
        {
            error!("Could not start the {name} thread: {spawn_error}");
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        work();
    }
}

/// A single background thread fed through a one-slot "latest request wins"
/// mailbox. Dropping it drops the wake-up sender, which ends the thread once it
/// finishes whatever it is processing.
///
/// A panic inside the handler is caught and logged; the thread keeps serving
/// later requests, so one bad job never silences the worker.
#[cfg(not(target_arch = "wasm32"))]
pub struct LatestWorker<R: Send + 'static> {
    slot: Arc<Mutex<Option<R>>>,
    wake: mpsc::Sender<()>,
    /// Kept so the UI harness can join the thread ([`Self::stop`]).
    #[cfg(feature = "ui-harness")]
    thread: Option<std::thread::JoinHandle<()>>,
}

/// The "latest request wins" mailbox of a host without threads: requests are
/// handled inline by [`Self::submit`].
#[cfg(target_arch = "wasm32")]
pub struct LatestWorker<R: Send + 'static> {
    slot: Arc<Mutex<Option<R>>>,
    handle: Mutex<Box<dyn FnMut(R) + Send>>,
}

/// The readable text of a panic payload.
#[cfg(not(target_arch = "wasm32"))]
fn panic_text(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a payload that is not text")
}

#[cfg(target_arch = "wasm32")]
impl<R: Send + 'static> LatestWorker<R> {
    /// Keep `handle`; it runs inline for each request. No thread is started.
    pub fn spawn(_name: &str, handle: impl FnMut(R) + Send + 'static) -> Self {
        Self {
            slot: Arc::new(Mutex::new(None)),
            handle: Mutex::new(Box::new(handle)),
        }
    }

    /// Keep `handle`; it runs inline for each request. `on_panic` is never
    /// called (a panic aborts the page).
    pub fn spawn_with_panic_hook(
        name: &str,
        handle: impl FnMut(R) + Send + 'static,
        _on_panic: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        Self::spawn(name, handle)
    }

    /// Handle `request` now, and any request the handler submits meanwhile.
    /// Always `true`.
    pub fn submit(&self, request: R) -> bool {
        *self.slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(request);
        // A submit from inside the handler finds it busy: the loop below picks
        // that request up when the handler returns.
        if let Ok(mut handle) = self.handle.try_lock() {
            loop {
                let next = self
                    .slot
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                let Some(next) = next else { break };
                (*handle)(next);
            }
        }
        true
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<R: Send + 'static> LatestWorker<R> {
    /// Spawn the thread, running `handle` for each request it picks up.
    pub fn spawn(name: &str, handle: impl FnMut(R) + Send + 'static) -> Self {
        Self::spawn_with_panic_hook(name, handle, || {})
    }

    /// Spawn the thread, running `handle` for each request it picks up. A panic in
    /// `handle` is logged and the thread carries on with the next request;
    /// `on_panic` runs on the worker thread after such a panic (the owner posts
    /// whatever it needs to the UI thread from there). A panic inside `on_panic`
    /// is caught and logged as well.
    pub fn spawn_with_panic_hook(
        name: &str,
        mut handle: impl FnMut(R) + Send + 'static,
        on_panic: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let slot: Arc<Mutex<Option<R>>> = Arc::new(Mutex::new(None));
        let (wake, rx) = mpsc::channel::<()>();
        let thread_slot = Arc::clone(&slot);
        let thread_name = name.to_owned();
        let spawned = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                for () in rx {
                    let request = thread_slot
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .take();
                    let Some(request) = request else { continue };
                    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| handle(request))) {
                        error!(
                            "The {thread_name} thread panicked on a request ({}); it keeps running.",
                            panic_text(payload.as_ref())
                        );
                        if let Err(hook_payload) = catch_unwind(AssertUnwindSafe(&on_panic)) {
                            error!(
                                "The {thread_name} panic hook panicked too ({}).",
                                panic_text(hook_payload.as_ref())
                            );
                        }
                    }
                }
            });
        if let Err(spawn_error) = &spawned {
            error!("Could not start the {name} thread: {spawn_error}");
        }
        Self {
            slot,
            wake,
            #[cfg(feature = "ui-harness")]
            thread: spawned.ok(),
        }
    }

    /// End the thread and wait for it: the request it is processing finishes, the
    /// rest is dropped. For the UI harness's teardown, which must not leave a
    /// thread reading files of a profile it is about to delete.
    #[cfg(feature = "ui-harness")]
    pub fn stop(self) {
        let Self { wake, thread, .. } = self;
        drop(wake);
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }

    /// Replace any not-yet-started request with `request` and wake the thread.
    /// Returns `false` when the thread is gone (it never started), so nothing will
    /// ever handle the request.
    pub fn submit(&self, request: R) -> bool {
        *self.slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(request);
        self.wake.send(()).is_ok()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::{
        sync::{Arc, Mutex, mpsc},
        time::Duration,
    };

    use super::LatestWorker;

    /// Long enough for a loaded machine, short enough that a hang fails the test.
    const WAIT: Duration = Duration::from_secs(10);

    #[test]
    fn a_panicking_request_does_not_stop_the_next_one() {
        let (seen_tx, seen_rx) = mpsc::channel::<u32>();
        let (hook_tx, hook_rx) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn_with_panic_hook(
            "test-panic",
            move |n: u32| {
                assert_ne!(n, 0, "the first request panics on purpose");
                let _ = seen_tx.send(n);
            },
            move || {
                let _ = hook_tx.send(());
            },
        );
        assert!(worker.submit(0));
        hook_rx
            .recv_timeout(WAIT)
            .expect("the panic hook runs after the handler panicked");
        assert!(worker.submit(1));
        assert_eq!(seen_rx.recv_timeout(WAIT), Ok(1));
    }

    #[test]
    fn only_the_latest_waiting_request_is_handled() {
        let (seen_tx, seen_rx) = mpsc::channel::<u32>();
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn("test-latest", move |n: u32| {
            let _ = seen_tx.send(n);
            if n == 0 {
                let _ = started_tx.send(());
                let _ = gate_rx.recv();
            }
        });
        assert!(worker.submit(0));
        started_rx
            .recv_timeout(WAIT)
            .expect("the first request started");
        // The handler is blocked inside request 0; these 100 replace one another
        // in the one-slot mailbox, so only the last of them is ever picked up.
        for n in 1..=100 {
            assert!(worker.submit(n));
        }
        gate_tx.send(()).expect("the handler is waiting");
        assert_eq!(seen_rx.recv_timeout(WAIT), Ok(0));
        assert_eq!(seen_rx.recv_timeout(WAIT), Ok(100));
        assert_eq!(
            seen_rx.recv_timeout(Duration::from_millis(200)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "nothing else was handled"
        );
    }

    #[test]
    fn dropping_the_worker_ends_its_thread() {
        let (alive_tx, alive_rx) = mpsc::channel::<()>();
        let worker = LatestWorker::spawn("test-drop", move |_: u32| {
            // Holding the sender in the handler ties its lifetime to the thread's.
            let _ = &alive_tx;
        });
        assert_eq!(
            alive_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "the thread is alive while the worker is"
        );
        drop(worker);
        assert_eq!(
            alive_rx.recv_timeout(WAIT),
            Err(mpsc::RecvTimeoutError::Disconnected),
            "the handler, and with it the sender, is gone once the thread ended"
        );
    }

    #[test]
    fn submitting_to_a_worker_without_a_thread_reports_it() {
        let (wake, rx) = mpsc::channel::<()>();
        drop(rx);
        let worker = LatestWorker::<u32> {
            slot: Arc::new(Mutex::new(None)),
            wake,
            #[cfg(feature = "ui-harness")]
            thread: None,
        };
        assert!(!worker.submit(1));
    }
}
