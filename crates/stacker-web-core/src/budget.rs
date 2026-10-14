//! The memory budget of the browser worker.
//!
//! A browser tab fails on memory long before the 4 GiB a `wasm32` instance can
//! address, so the page checks a run against [`WORKER_BUDGET_BYTES`] before it
//! sends the files. [`plan_budget`] answers whether the run fits and, if not,
//! which Pre-resize percentage would.

use std::io::Cursor;

use stacker_core::settings::StackingSettings;
use stacker_pipeline::session::{self, PassKind as SessionPass};

use crate::protocol::PassKind;

/// What the worker may use at its peak: 2 GiB.
pub const WORKER_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Headroom for the allocator, the messages and the module itself: 64 MiB.
pub const MARGIN_BYTES: u64 = 64 * 1024 * 1024;

/// Whether a run fits the worker's memory, and what to do if it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// The estimate is within [`WORKER_BUDGET_BYTES`].
    pub fits: bool,
    /// The estimated peak in bytes: the run's stages, the encoded files the
    /// worker holds and [`MARGIN_BYTES`].
    pub estimate: u64,
    /// When the run does not fit: the largest Pre-resize percentage (a multiple
    /// of 5, below the setting) that does, or `None` when even the lowest
    /// percentage the settings allow does not fit. `None` when it fits.
    pub suggested_pre_resize: Option<u32>,
}

/// Check a run of `frames` files against the worker's budget.
///
/// `first_frame_dims` is the size of the first file (see [`dims_from_bytes`]);
/// `encoded_bytes` the summed size of all files, which the worker keeps next to
/// the decoded frames; `frames` counts the frames that enter the run (after
/// `stack_every_nth`).
#[must_use]
pub fn plan_budget(
    frames: usize,
    first_frame_dims: (u32, u32),
    encoded_bytes: u64,
    settings: &StackingSettings,
    passes: &[PassKind],
) -> Budget {
    let passes: Vec<SessionPass> = passes.iter().copied().map(SessionPass::from).collect();
    let (width, height) = first_frame_dims;
    let fixed = encoded_bytes.saturating_add(MARGIN_BYTES);
    let estimate = session::estimate_peak_bytes(frames, width, height, settings, &passes)
        .saturating_add(fixed);
    let fits = estimate <= WORKER_BUDGET_BYTES;
    let suggested_pre_resize = if fits {
        None
    } else {
        session::fit_pre_resize(
            frames,
            width,
            height,
            settings,
            &passes,
            WORKER_BUDGET_BYTES.saturating_sub(fixed),
        )
    };
    Budget {
        fits,
        estimate,
        suggested_pre_resize,
    }
}

/// The pixel size of an encoded image, read from its header (no decoding).
/// `None` when the format is not recognised or the header is damaged.
#[must_use]
pub fn dims_from_bytes(bytes: &[u8]) -> Option<(u32, u32)> {
    image::io::Reader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

#[cfg(test)]
mod tests {
    use stacker_core::settings::StackingSettings;

    use super::{WORKER_BUDGET_BYTES, dims_from_bytes, plan_budget};
    use crate::protocol::PassKind;

    #[test]
    fn a_small_run_fits() {
        let budget = plan_budget(
            3,
            (100, 100),
            30_000,
            &StackingSettings::default(),
            &[PassKind::Apex],
        );
        assert!(budget.fits);
        assert_eq!(budget.suggested_pre_resize, None);
        assert!(budget.estimate <= WORKER_BUDGET_BYTES);
    }

    #[test]
    fn a_large_run_is_reduced_to_a_percentage_that_fits() {
        let settings = StackingSettings::default();
        let budget = plan_budget(28, (6000, 4000), 280_000_000, &settings, &[PassKind::Apex]);
        assert!(!budget.fits);
        let percent = budget
            .suggested_pre_resize
            .expect("some percentage fits 28 frames");
        assert!(percent < settings.preprocessing.pre_resize_percent);
        assert_eq!(percent % 5, 0);

        let mut reduced = settings;
        reduced.preprocessing.pre_resize_percent = percent;
        let again = plan_budget(28, (6000, 4000), 280_000_000, &reduced, &[PassKind::Apex]);
        assert!(again.fits);
    }

    #[test]
    fn the_estimate_grows_with_frames_and_size() {
        let settings = StackingSettings::default();
        let estimate = |frames, dims| {
            plan_budget(
                frames,
                dims,
                0,
                &settings,
                &[PassKind::Relief, PassKind::Apex],
            )
            .estimate
        };
        assert!(estimate(10, (2000, 1500)) < estimate(20, (2000, 1500)));
        assert!(estimate(10, (2000, 1500)) < estimate(10, (4000, 3000)));
    }

    #[test]
    fn strata_grows_by_one_frame_per_image_not_by_a_pyramid() {
        // Strata keeps 12 B/px per frame (the budget uses the session estimate).
        let settings = StackingSettings::default();
        let estimate =
            |frames, pass| plan_budget(frames, (2000, 1500), 0, &settings, &[pass]).estimate;
        let per_frame = estimate(21, PassKind::Strata) - estimate(20, PassKind::Strata);
        assert_eq!(per_frame, 12 * 2000 * 1500);
        // ... which is less per frame than Apex's pyramid.
        assert!(per_frame < estimate(21, PassKind::Apex) - estimate(20, PassKind::Apex));
    }

    #[test]
    fn the_multigrid_depth_map_costs_extra_on_relief() {
        let plain = StackingSettings::default();
        let multigrid = StackingSettings {
            relief_use_multigrid: true,
            ..StackingSettings::default()
        };
        let relief = |settings: &StackingSettings| {
            plan_budget(8, (2000, 1500), 0, settings, &[PassKind::Relief]).estimate
        };
        assert!(relief(&multigrid) > relief(&plain));
    }

    #[test]
    fn the_encoded_files_count_against_the_budget() {
        let settings = StackingSettings::default();
        let plain = plan_budget(5, (1000, 1000), 0, &settings, &[PassKind::Apex]);
        let heavy = plan_budget(5, (1000, 1000), 500_000_000, &settings, &[PassKind::Apex]);
        assert_eq!(heavy.estimate - plain.estimate, 500_000_000);
    }

    #[test]
    fn dimensions_come_from_the_sample_header() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../sample/frame_0000.jpg");
        let Ok(bytes) = std::fs::read(path) else {
            return; // the sample images are not part of every checkout
        };
        let (width, height) = dims_from_bytes(&bytes).expect("a JPEG header");
        assert!(width > 0 && height > 0);
        assert_eq!(dims_from_bytes(b"not an image"), None);
    }
}
