//! UI-thread stall detector.
//!
//! [`stall_guard`] wraps a callback body and logs a `tracing::warn!` naming the
//! callback and its elapsed time whenever it runs longer than
//! [`STALL_THRESHOLD`] (the frame budget a 60 Hz UI needs to stay responsive).
//! The run log shows the stall without any new UI surface.
//!
//! It is a plain wrapper function rather than a macro or an attribute: wrapping
//! the closure BODY in `stall_guard("name", || { ... })` needs no new syntax, and
//! the body keeps its own early returns exactly as before.
//!
//! A callback that hands its work to a background thread returns as soon as the
//! hand-off is queued; the guard measures exactly that return, never the later
//! completion closure (wrap that separately if it matters).

use std::time::Duration;

use web_time::Instant;

/// A callback body running longer than this on the event-loop thread is a
/// visible stall at 60 Hz (one frame is about 16.7 ms).
pub const STALL_THRESHOLD: Duration = Duration::from_millis(16);

/// Run `body`, logging a warning naming `name` and the elapsed time if it took
/// longer than [`STALL_THRESHOLD`]. Returns the body's own result unchanged.
pub fn stall_guard<R>(name: &str, body: impl FnOnce() -> R) -> R {
    let start = Instant::now();
    let result = body();
    let elapsed = start.elapsed();
    if is_stall(elapsed) {
        tracing::warn!(
            callback = name,
            elapsed_ms = elapsed.as_millis(),
            "UI-thread callback exceeded the 16ms stall threshold"
        );
    }
    result
}

/// The threshold check itself, so a test can drive it without sleeping.
#[must_use]
pub const fn is_stall(elapsed: Duration) -> bool {
    elapsed.as_millis() > STALL_THRESHOLD.as_millis()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{is_stall, stall_guard};

    #[test]
    fn returns_the_bodys_own_result() {
        assert_eq!(stall_guard("test", || 42), 42);
    }

    #[test]
    fn runs_the_body_exactly_once() {
        let mut calls = 0;
        stall_guard("test", || calls += 1);
        assert_eq!(calls, 1);
    }

    #[test]
    fn is_not_a_stall_at_or_under_the_threshold() {
        assert!(!is_stall(Duration::from_millis(0)));
        assert!(!is_stall(Duration::from_millis(16)));
    }

    #[test]
    fn is_a_stall_over_the_threshold() {
        assert!(is_stall(Duration::from_millis(17)));
        assert!(is_stall(Duration::from_secs(1)));
    }
}
