//! The progress contract of the Python API: `(stage, current, total)`.
//!
//! The pipeline reports `Progress::stage(stage, fraction, label)` with a fraction in
//! `0.0..=1.0`. Python callbacks get a stable tuple instead:
//!
//! | `stage` | meaning |
//! |---|---|
//! | `"load"` | decoding the frames (the in-RAM path of `stack_files` with `tile_size = 0`) |
//! | `"align"` | aligning the frames to the reference |
//! | `"cull"` | the focus census and the auto-cull |
//! | `"fuse"` | fusing (tile by tile on the out-of-core path; a two-pass engine counts both) |
//! | `"save"` | encoding and writing the result |
//!
//! `total` is always [`PROGRESS_TOTAL`] (1000) and `current` is the fraction in thousandths, so
//! `current / total` is the progress of the stage and `current == total` marks its end. A
//! stage is reported when `current` changes, so a long stage produces at most 1001 calls.
//!
//! The mapping is independent of Python: [`ProgressForwarder`] takes any closure. The Python
//! callback is wired in by `crate::bridge`.

use std::cell::Cell;

use stacker_pipeline::{Progress, Stage};

/// The `total` of every progress tuple: a fraction is reported in thousandths.
pub const PROGRESS_TOTAL: u32 = 1000;

/// The tag of `stage` in the Python callback.
#[must_use]
pub const fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Load => "load",
        Stage::Align => "align",
        Stage::Cull => "cull",
        Stage::Fuse => "fuse",
        Stage::Save => "save",
    }
}

/// The `(stage, current, total)` tuple of a pipeline report. A fraction outside `0.0..=1.0`
/// is clamped and NaN reads as 0.
#[must_use]
pub fn progress_tuple(stage: Stage, fraction: f32) -> (&'static str, u32, u32) {
    let fraction = if fraction.is_nan() {
        0.0
    } else {
        fraction.clamp(0.0, 1.0)
    };
    (stage_name(stage), thousandths(fraction), PROGRESS_TOTAL)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the fraction is in 0..=1, so the rounded product is in 0..=1000"
)]
fn thousandths(fraction: f32) -> u32 {
    (fraction * 1000.0).round() as u32
}

/// A [`Progress`] that turns reports into `(stage, current, total)` calls of a closure, and
/// skips a report that would repeat the previous tuple.
pub struct ProgressForwarder<F> {
    emit: F,
    last: Cell<Option<(Stage, u32)>>,
}

impl<F: Fn(&'static str, u32, u32)> ProgressForwarder<F> {
    /// A forwarder that calls `emit(stage, current, total)`.
    #[must_use]
    pub const fn new(emit: F) -> Self {
        Self {
            emit,
            last: Cell::new(None),
        }
    }
}

impl<F: Fn(&'static str, u32, u32)> Progress for ProgressForwarder<F> {
    fn stage(&self, stage: Stage, fraction: f32, _label: &str) {
        let (name, current, total) = progress_tuple(stage, fraction);
        if self.last.get() != Some((stage, current)) {
            self.last.set(Some((stage, current)));
            (self.emit)(name, current, total);
        }
    }
}
