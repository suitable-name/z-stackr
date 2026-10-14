//! [`RedrawGate`]: coalesces a burst of redraw requests into one queued closure.
//!
//! `Weak::upgrade_in_event_loop` queues rather than runs, so a burst of requests
//! would otherwise queue one closure each. The gate keeps at most one pending
//! closure, which always acts on the LATEST submitted payload.
//!
//! Without it a producer that outpaces the UI (a retouch stroke, the live
//! previews of a stack pass) grows the event queue without bound, and every
//! queued closure carries its own full-size pixel buffer. With it the queue holds
//! one closure, and the buffers a burst skipped are dropped on the producer's
//! side.
//!
//! No Slint type appears in this module, so it is unit-testable as it stands.
//!
//! Two cooperating primitives:
//!
//! * `pending`: set by whichever [`RedrawGate::submit`] wins the compare-exchange,
//!   cleared by [`RedrawGate::take`], which the queued closure calls right before
//!   it does real work. Every other `submit` in between returns `None` and must
//!   not queue a second closure: the pending one reads the latest slot anyway.
//! * `generation`: bumped on every `submit` (won or not), so a caller or a test
//!   can confirm a burst was coalesced.

use std::sync::{
    Mutex, PoisonError,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

/// Coalesces redraw requests so that at most one is pending.
pub struct RedrawGate<T> {
    slot: Mutex<Option<T>>,
    pending: AtomicBool,
    generation: AtomicU64,
}

impl<T> Default for RedrawGate<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RedrawGate<T> {
    /// An empty gate.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
            pending: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        }
    }

    /// Record `value` as the latest payload (latest wins) and bump the
    /// generation. Returns `Some(generation)` exactly when THIS call must queue
    /// the closure (none is pending); `None` means a queued closure will pick the
    /// value up, so the caller must not queue another.
    #[must_use = "a `Some` means the caller must queue the closure that calls `take`"]
    pub fn submit(&self, value: T) -> Option<u64> {
        *self.slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(value);
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.pending
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
            .then_some(generation)
    }

    /// Called from inside the queued closure before real work: clears `pending`
    /// (so a later `submit` can queue the next closure) and takes the latest
    /// payload, which may be newer than the one that caused the closure to be
    /// queued. `None` only without a matching prior `submit`.
    pub fn take(&self) -> Option<T> {
        self.pending.store(false, Ordering::Release);
        self.slot
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// How many `submit` calls were made so far, coalesced ones included.
    #[must_use]
    pub fn current_generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Whether a closure is currently queued.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.pending.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::RedrawGate;

    #[test]
    fn first_submit_must_queue() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        assert_eq!(gate.submit(1), Some(1));
        assert!(gate.is_pending());
    }

    #[test]
    fn a_second_submit_before_take_must_not_queue_again() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        assert!(gate.submit(1).is_some());
        assert_eq!(gate.submit(2), None);
        assert_eq!(gate.submit(3), None);
    }

    #[test]
    fn every_submit_bumps_the_generation_even_when_coalesced() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        let _ = gate.submit(1);
        let _ = gate.submit(2);
        let _ = gate.submit(3);
        assert_eq!(gate.current_generation(), 3);
    }

    #[test]
    fn take_returns_the_latest_value_not_the_first_one() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        assert!(gate.submit(1).is_some());
        assert_eq!(gate.submit(2), None);
        assert_eq!(gate.submit(3), None);
        assert_eq!(gate.take(), Some(3));
    }

    #[test]
    fn take_clears_pending_so_a_later_submit_can_queue_again() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        assert!(gate.submit(1).is_some());
        let _ = gate.take();
        assert!(!gate.is_pending());
        assert!(gate.submit(2).is_some());
    }

    #[test]
    fn take_on_an_empty_gate_returns_none() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        assert_eq!(gate.take(), None);
    }

    #[test]
    fn a_burst_coalesces_to_exactly_one_queued_closure() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        let queued = (0..50).filter(|&i| gate.submit(i).is_some()).count();
        assert_eq!(queued, 1);
        assert_eq!(gate.take(), Some(49));
    }

    #[test]
    fn a_second_burst_after_take_queues_exactly_once_more() {
        let gate: RedrawGate<u32> = RedrawGate::new();
        for i in 0..10 {
            let _ = gate.submit(i);
        }
        let _ = gate.take();
        let queued = (10..20).filter(|&i| gate.submit(i).is_some()).count();
        assert_eq!(queued, 1);
        assert_eq!(gate.take(), Some(19));
    }

    #[test]
    fn works_as_a_pure_signal_with_a_unit_payload() {
        let gate: RedrawGate<()> = RedrawGate::new();
        assert!(gate.submit(()).is_some());
        assert_eq!(gate.submit(()), None);
        assert_eq!(gate.take(), Some(()));
        assert!(gate.submit(()).is_some());
    }
}
