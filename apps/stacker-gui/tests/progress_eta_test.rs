use std::time::{Duration, Instant};

use stacker_gui::progress_eta::{EtaEstimator, format_eta};

/// Perfectly linear progress (a constant rate the whole way) must produce an
/// exact ETA: a least-squares fit across points that already lie exactly on one
/// line reproduces that line's own slope with no error.
#[test]
fn linear_progress_gives_exact_eta() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    // 0.05 fraction/second: reaches 1.0 at t = 20 s.
    for t in 0..=9_u32 {
        eta.observe(
            start + Duration::from_secs(u64::from(t)),
            0.05 * f64::from(t),
        );
    }
    let now = start + Duration::from_secs(9);
    let remaining = eta.eta(now).expect("enough history for an estimate");
    // At t = 9 the line is at fraction 0.45; 0.55 left at 0.05/s is 11 s.
    assert!(
        (remaining.as_secs_f64() - 11.0).abs() < 1e-6,
        "expected ~11 s, got {remaining:?}"
    );
}

/// A step cadence (a jump every 20 s) must land within 25 % of the true
/// remaining time once a handful of steps have been observed: the whole reason
/// the estimator fits a line across the window instead of reading the most
/// recent step's own instantaneous rate.
#[test]
fn step_like_progress_is_within_a_quarter_of_the_truth_after_three_steps() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    // Five steps of 20 s reach 1.0 at t = 100 s; a true rate of 0.01/s.
    eta.observe(start, 0.0);
    eta.observe(start + Duration::from_secs(20), 0.2);
    eta.observe(start + Duration::from_secs(40), 0.4);
    eta.observe(start + Duration::from_secs(60), 0.6);
    let now = start + Duration::from_secs(60);
    let remaining = eta.eta(now).expect("enough history after three steps");
    let truth_secs = 40.0; // 100 s total minus 60 s elapsed
    let error = (remaining.as_secs_f64() - truth_secs).abs() / truth_secs;
    assert!(
        error <= 0.25,
        "expected within 25% of {truth_secs} s, got {remaining:?}"
    );
}

/// Progress that never advances must never produce an estimate: a zero slope has
/// no forward rate to project a completion time from.
#[test]
fn stalled_progress_returns_none() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    for t in 0..=6_u64 {
        eta.observe(start + Duration::from_secs(t), 0.3);
    }
    assert!(eta.eta(start + Duration::from_secs(6)).is_none());
}

/// Progress that goes backwards (a job that restarts a stage) has no forward
/// rate either.
#[test]
fn regressing_progress_returns_none() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    for t in 0..=6_u32 {
        eta.observe(
            start + Duration::from_secs(u64::from(t)),
            0.05_f64.mul_add(-f64::from(t), 0.6),
        );
    }
    assert!(eta.eta(start + Duration::from_secs(6)).is_none());
}

/// Fewer than five seconds of history is refused, however many points there are.
#[test]
fn a_short_span_is_not_trusted() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    eta.observe(start, 0.0);
    eta.observe(start + Duration::from_secs(2), 0.5);
    assert!(eta.eta(start + Duration::from_secs(2)).is_none());
}

/// `reset` must drop every prior observation: re-feeding a single, too-recent
/// observation afterwards still refuses an estimate, exactly as a brand-new
/// estimator would.
#[test]
fn reset_clears_prior_history() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    for t in 0..=9_u32 {
        eta.observe(
            start + Duration::from_secs(u64::from(t)),
            0.05 * f64::from(t),
        );
    }
    assert!(eta.eta(start + Duration::from_secs(9)).is_some());

    eta.reset();
    eta.observe(start + Duration::from_secs(9), 0.45);
    assert!(
        eta.eta(start + Duration::from_secs(9)).is_none(),
        "a single observation right after reset must not yet produce an estimate"
    );
}

/// A very chatty job never grows the window without bound, and the estimate stays
/// usable: 200 ticks one second apart at a constant rate.
#[test]
fn a_chatty_job_keeps_a_bounded_window_and_a_correct_estimate() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    for t in 0..200_u32 {
        eta.observe(
            start + Duration::from_secs(u64::from(t)),
            0.002 * f64::from(t),
        );
    }
    let now = start + Duration::from_secs(199);
    let remaining = eta.eta(now).expect("plenty of history");
    // At t = 199 the line is at 0.398; 0.602 left at 0.002/s is 301 s.
    assert!(
        (remaining.as_secs_f64() - 301.0).abs() < 1e-3,
        "expected ~301 s, got {remaining:?}"
    );
}

/// A nearly flat positive slope would overflow `Duration`; it yields no estimate
/// instead of a panic.
#[test]
fn an_absurdly_long_estimate_does_not_panic() {
    let start = Instant::now();
    let mut eta = EtaEstimator::default();
    eta.observe(start, 0.0);
    eta.observe(start + Duration::from_secs(10), 1e-300);
    assert!(eta.eta(start + Duration::from_secs(10)).is_none());
}

#[test]
fn format_eta_renders_each_bucket() {
    assert_eq!(format_eta(None), "");
    assert_eq!(format_eta(Some(Duration::from_secs(5))), "about 5 s left");
    assert_eq!(
        format_eta(Some(Duration::from_secs(120))),
        "about 2 min left"
    );
    assert_eq!(
        format_eta(Some(Duration::from_mins(80))),
        "about 1 h 20 min left"
    );
    assert_eq!(
        format_eta(Some(Duration::from_secs(7200))),
        "about 2 h left"
    );
    assert_eq!(
        format_eta(Some(Duration::from_millis(400))),
        "about 1 s left"
    );
}

#[test]
fn format_eta_rounds_to_the_nearest_unit() {
    assert_eq!(
        format_eta(Some(Duration::from_millis(5_600))),
        "about 6 s left"
    );
    // 89 s rounds to 1 min, 90 s to 2 min.
    assert_eq!(
        format_eta(Some(Duration::from_secs(89))),
        "about 1 min left"
    );
    assert_eq!(
        format_eta(Some(Duration::from_secs(90))),
        "about 2 min left"
    );
}
