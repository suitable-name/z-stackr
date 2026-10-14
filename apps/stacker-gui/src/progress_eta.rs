//! A rolling, regression-based time-remaining estimator for a long-running job,
//! plus [`format_eta`] to turn its result into the short string the status strip
//! shows next to the percentage.
//!
//! # Why regression over a window, not "seconds since last tick / delta fraction"
//!
//! The progress a job reports is step-like, not smooth: the stack pass advances in
//! stage-sized chunks (a fusion can take tens of seconds between two reports) and
//! the alignment reports once per finished frame. An estimator built from the
//! single most recent pair of observations would swing wildly between updates:
//! near zero right after a chunk lands (the "instantaneous rate" looks huge), then
//! rising sharply as the next chunk takes its full time to arrive. Fitting a
//! least-squares line across a whole rolling window of `(time, fraction_done)`
//! observations instead smooths over that noise: a handful of irregular steps
//! still describe a fairly steady average rate, which is what a photographer
//! actually wants from a "how much longer" readout.

use std::{collections::VecDeque, time::Duration};

use web_time::Instant;

/// Observations older than this are dropped once the window also has more than
/// [`MAX_OBSERVATIONS`] entries; see [`EtaEstimator::observe`].
const MIN_WINDOW: Duration = Duration::from_secs(30);

/// Hard cap on how many observations [`EtaEstimator`] keeps once the 30-second
/// window ([`MIN_WINDOW`]) is already satisfied: it bounds the regression's own
/// cost and stops a very long, very chatty run from growing without limit.
const MAX_OBSERVATIONS: usize = 64;

/// [`EtaEstimator::eta`] refuses to guess from less than this much observed
/// history: two observations a fraction of a second apart would extrapolate a
/// noise-sized slope into a wildly wrong estimate.
const MIN_SPAN_FOR_ESTIMATE: Duration = Duration::from_secs(5);

/// A rolling window of `(time, fraction_done)` observations for one running job.
///
/// It is used to fit a least-squares line and read off "how much longer until
/// `fraction_done` reaches 1.0". See the module docs for why a regression over a
/// window rather than a single instantaneous rate.
///
/// Feed it with [`observe`](Self::observe) on every progress tick, read
/// [`eta`](Self::eta) whenever the UI needs a fresh estimate, and
/// [`reset`](Self::reset) it when a new, unrelated run starts: carrying a previous
/// run's observations into a new one would fit a line across two unrelated rates.
#[derive(Debug, Default, Clone)]
pub struct EtaEstimator {
    observations: VecDeque<(Instant, f64)>,
}

impl EtaEstimator {
    /// Records one `(now, fraction_done)` observation, trimming the window
    /// afterwards.
    ///
    /// The window keeps AT LEAST the last [`MIN_WINDOW`] (30 s) of history
    /// regardless of how many observations that takes, and separately caps at
    /// [`MAX_OBSERVATIONS`] (64) once that 30-second requirement is already met,
    /// so a job reporting many ticks per second still bounds this struct's memory
    /// while a job reporting only a handful of ticks keeps enough history to fit a
    /// meaningful line.
    pub fn observe(&mut self, now: Instant, fraction_done: f64) {
        self.observations
            .push_back((now, fraction_done.clamp(0.0, 1.0)));
        while self.observations.len() > MAX_OBSERVATIONS {
            let Some(&(oldest, _)) = self.observations.front() else {
                break;
            };
            if now.saturating_duration_since(oldest) <= MIN_WINDOW {
                // Still inside the required 30-second window: keep it even
                // though the count is already over the cap.
                break;
            }
            self.observations.pop_front();
        }
    }

    /// Fits a least-squares line across the current window and reads off the
    /// remaining time until it reaches `fraction_done == 1.0`, evaluated at `now`
    /// (not necessarily the time of the last observation: the caller may ask for
    /// an estimate between ticks).
    ///
    /// Returns `None` when there is not enough history to trust a fit yet (fewer
    /// than two observations, or a span under 5 s), when the window's own
    /// timestamps are all identical (a zero-variance fit has no slope to read), or
    /// when the fitted slope is zero or negative: a stalled or regressing job has
    /// no forward rate to project from, and no number is better than a wrong one.
    #[must_use]
    pub fn eta(&self, now: Instant) -> Option<Duration> {
        let &(first_time, _) = self.observations.front()?;
        let &(last_time, _) = self.observations.back()?;
        let span = last_time.duration_since(first_time);
        if self.observations.len() < 2 || span < MIN_SPAN_FOR_ESTIMATE {
            return None;
        }

        let points: Vec<(f64, f64)> = self
            .observations
            .iter()
            .map(|&(t, f)| (t.duration_since(first_time).as_secs_f64(), f))
            .collect();
        let count = f64::from(u32::try_from(points.len()).unwrap_or(u32::MAX));
        let mean_x = points.iter().map(|&(x, _)| x).sum::<f64>() / count;
        let mean_y = points.iter().map(|&(_, y)| y).sum::<f64>() / count;

        let mut covariance = 0.0;
        let mut variance = 0.0;
        for &(x, y) in &points {
            let dx = x - mean_x;
            covariance = dx.mul_add(y - mean_y, covariance);
            variance = dx.mul_add(dx, variance);
        }
        if variance <= f64::EPSILON {
            return None;
        }
        let slope = covariance / variance;
        if slope <= 0.0 {
            return None;
        }

        let intercept = slope.mul_add(-mean_x, mean_y);
        let now_x = now.saturating_duration_since(first_time).as_secs_f64();
        let fitted_now = slope.mul_add(now_x, intercept).max(0.0);
        let remaining_fraction = (1.0 - fitted_now).max(0.0);
        let remaining_secs = remaining_fraction / slope;
        // `try_from_secs_f64` also refuses an absurdly large value (a nearly flat
        // slope) instead of panicking.
        Duration::try_from_secs_f64(remaining_secs).ok()
    }

    /// Clears every recorded observation. Called when a new, unrelated run starts
    /// (or the previous one finished, failed or was cancelled), so its history
    /// never leaks into the next run's estimate.
    pub fn reset(&mut self) {
        self.observations.clear();
    }
}

/// Formats an [`EtaEstimator::eta`] result as the short string the strip shows.
///
/// `""` for `None` (hides the estimate entirely), otherwise `"about 5 s left"`,
/// `"about 2 min left"` or `"about 1 h 20 min left"`, rounded to the coarsest unit
/// that still reads as informative: never a sub-second reading, and never a bare
/// `"0 s"` for a duration that rounds down to nothing.
#[must_use]
pub fn format_eta(eta: Option<Duration>) -> String {
    let Some(remaining) = eta else {
        return String::new();
    };
    // Rounded to the nearest second, at least one.
    let total_secs =
        (remaining.as_secs() + u64::from(remaining.subsec_nanos() >= 500_000_000)).max(1);

    if total_secs < 60 {
        format!("about {total_secs} s left")
    } else if total_secs < 3600 {
        let minutes = (total_secs + 30) / 60;
        format!("about {minutes} min left")
    } else {
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        if minutes == 0 {
            format!("about {hours} h left")
        } else {
            format!("about {hours} h {minutes} min left")
        }
    }
}
