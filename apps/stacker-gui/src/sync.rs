//! Poison-tolerant locking for the few mutexes that worker threads genuinely
//! share with the UI thread.
//!
//! A worker that panics while holding a guard poisons the mutex. Every later
//! `lock().unwrap()` on the UI thread would then panic too and take the whole
//! window down for a failure that is already reported elsewhere. The data these
//! mutexes guard is always replaced wholesale (`take`, `= Some(..)`), never left
//! half-updated, so recovering the inner guard is sound.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Lock `mutex`, recovering the guard when a panicking thread poisoned it.
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
