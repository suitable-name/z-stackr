//! The `(stage, current, total)` progress contract, tested without Python.

use std::cell::RefCell;

use stacker_pipeline::{Progress, Stage};
use stacker_python::progress::{PROGRESS_TOTAL, ProgressForwarder, progress_tuple, stage_name};

#[test]
fn every_stage_has_its_documented_tag() {
    assert_eq!(stage_name(Stage::Load), "load");
    assert_eq!(stage_name(Stage::Align), "align");
    assert_eq!(stage_name(Stage::Cull), "cull");
    assert_eq!(stage_name(Stage::Fuse), "fuse");
    assert_eq!(stage_name(Stage::Save), "save");
}

#[test]
fn a_fraction_is_reported_in_thousandths() {
    assert_eq!(PROGRESS_TOTAL, 1000);
    assert_eq!(progress_tuple(Stage::Align, 0.0), ("align", 0, 1000));
    assert_eq!(progress_tuple(Stage::Align, 0.25), ("align", 250, 1000));
    assert_eq!(progress_tuple(Stage::Fuse, 1.0), ("fuse", 1000, 1000));
}

#[test]
fn a_fraction_outside_the_unit_interval_is_clamped() {
    assert_eq!(progress_tuple(Stage::Save, -0.5).1, 0);
    assert_eq!(progress_tuple(Stage::Save, 7.0).1, 1000);
    assert_eq!(progress_tuple(Stage::Save, f32::NAN).1, 0);
}

#[test]
fn the_forwarder_skips_a_report_that_repeats_the_tuple() {
    let calls = RefCell::new(Vec::new());
    let forwarder = ProgressForwarder::new(|stage, current, total| {
        calls.borrow_mut().push((stage, current, total));
    });
    forwarder.stage(Stage::Align, 0.0, "");
    forwarder.stage(Stage::Align, 0.0001, "");
    forwarder.stage(Stage::Align, 0.5, "frame");
    forwarder.stage(Stage::Align, 0.5, "");
    forwarder.stage(Stage::Fuse, 0.5, "");
    forwarder.stage(Stage::Align, 0.5, "");
    assert_eq!(
        calls.into_inner(),
        vec![
            ("align", 0, 1000),
            ("align", 500, 1000),
            ("fuse", 500, 1000),
            ("align", 500, 1000),
        ]
    );
}
