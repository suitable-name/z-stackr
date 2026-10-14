//! The retouch preview off the UI thread: the feed that collects what changed and
//! the compositor that turns it into frames, without a window.

use std::{path::PathBuf, sync::Arc};

use stacker_algo::hybrid::retouch::RetouchSession;
use stacker_core::image::PlanarImage;
use stacker_gui::{
    image_utils::planar_to_rgba_buffer,
    retouch::{RetouchState, SourceChoice},
    retouch_preview::{Compositor, PreviewFeed, Rect},
};

const W: usize = 48;
const H: usize = 36;

fn solid(level: f32) -> PlanarImage<f32> {
    let mut image = PlanarImage::new(W, H);
    image.luma.fill(level);
    image
}

/// Choose source 0 and install a solid frame of `level` for it.
fn use_source(state: &mut RetouchState, level: f32) {
    let SourceChoice::Load(request) = state.choose_source(Some(0)) else {
        panic!("choosing the first source should ask for a frame");
    };
    assert_eq!(state.finish_source(&request, solid(level)), Ok(true));
}

/// A state over a dark result with a bright source frame chosen.
fn state() -> RetouchState {
    let base = Arc::new(solid(0.1));
    let mut state = RetouchState::new();
    state.start_session(
        RetouchSession::new(Arc::clone(&base), base),
        PathBuf::from("stacker_result_Apex_1.png"),
    );
    state.refresh_sources(None, &[PathBuf::from("a.png")]);
    use_source(&mut state, 0.9);
    state
}

/// One complete drag through `points` at full opacity.
fn drag(state: &mut RetouchState, points: &[(u32, u32)], radius: f32) {
    assert!(state.begin_stroke());
    for &point in points {
        state.queue_point(Some(point), radius, 1.0);
        state.drain_pending();
    }
    assert!(state.end_stroke());
}

/// The 8-bit bytes of the composite made in one piece: what the preview must show.
fn expected(state: &RetouchState) -> Vec<u8> {
    let composite = state.composite_f32().expect("a session");
    planar_to_rgba_buffer(&composite).as_bytes().to_vec()
}

/// What the preview worker does for one wake-up: flush the state into the feed,
/// take the work and composite it. `None` when there was nothing to do.
fn pump(state: &mut RetouchState, compositor: &mut Compositor) -> Option<Vec<u8>> {
    let feed = state.flush_preview();
    let work = feed.take()?;
    compositor
        .apply(work)
        .map(|frame| frame.as_bytes().to_vec())
}

/// The patch and the full composite agree to within one level per channel (the
/// row encoder may round a narrower row differently in the last bit).
fn assert_close(actual: &[u8], wanted: &[u8]) {
    assert_eq!(actual.len(), wanted.len());
    let worst = actual
        .iter()
        .zip(wanted)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(worst <= 1, "the preview is off by {worst} levels");
}

// ── Rect ─────────────────────────────────────────────────────────────────────

#[test]
fn a_dab_dirties_its_radius_plus_a_margin() {
    let rect = Rect::around_segment(None, (10, 10), 2.0, (W, H)).expect("inside the frame");
    assert_eq!(
        rect,
        Rect {
            x0: 7,
            y0: 7,
            x1: 14,
            y1: 14
        }
    );
    assert_eq!((rect.width(), rect.height()), (7, 7));
}

#[test]
fn a_segment_dirties_the_box_around_both_ends() {
    let rect = Rect::around_segment(Some((4, 4)), (10, 8), 1.0, (W, H)).expect("inside");
    assert_eq!(
        rect,
        Rect {
            x0: 2,
            y0: 2,
            x1: 13,
            y1: 11
        }
    );
}

#[test]
fn the_dirty_area_is_cut_to_the_frame() {
    let corner = Rect::around_segment(None, (0, 0), 2.0, (W, H)).expect("inside");
    assert_eq!((corner.x0, corner.y0), (0, 0));
    let far = Rect::around_segment(None, (47, 35), 5.0, (W, H)).expect("inside");
    assert_eq!((far.x1, far.y1), (W, H));
    assert_eq!(Rect::around_segment(None, (500, 500), 2.0, (W, H)), None);
}

#[test]
fn a_huge_or_invalid_radius_does_not_overflow() {
    assert!(Rect::around_segment(None, (5, 5), f32::MAX, (W, H)).is_some());
    assert!(Rect::around_segment(None, (5, 5), f32::NAN, (W, H)).is_some());
    assert!(Rect::around_segment(None, (5, 5), -3.0, (W, H)).is_some());
}

#[test]
fn the_union_holds_both_rectangles() {
    let a = Rect {
        x0: 2,
        y0: 3,
        x1: 6,
        y1: 9,
    };
    let b = Rect {
        x0: 5,
        y0: 1,
        x1: 12,
        y1: 4,
    };
    assert_eq!(
        a.union(b),
        Rect {
            x0: 2,
            y0: 1,
            x1: 12,
            y1: 9
        }
    );
}

// ── Feed and compositor ──────────────────────────────────────────────────────

#[test]
fn a_feed_without_a_session_has_no_work() {
    let feed = PreviewFeed::new();
    assert!(feed.take().is_none());
    feed.reset(None);
    assert!(feed.take().is_none());
}

#[test]
fn the_first_work_is_the_whole_composite_and_then_there_is_nothing_to_do() {
    let mut state = state();
    let mut compositor = Compositor::default();
    let first = pump(&mut state, &mut compositor).expect("a full composite");
    assert_close(&first, &expected(&state));
    assert!(
        pump(&mut state, &mut compositor).is_none(),
        "nothing changed since"
    );
}

#[test]
fn a_stroke_becomes_a_patch_that_matches_the_full_composite() {
    let mut state = state();
    let mut compositor = Compositor::default();
    let before = pump(&mut state, &mut compositor).expect("a full composite");

    drag(&mut state, &[(10, 10), (20, 14), (30, 20)], 3.0);
    let after = pump(&mut state, &mut compositor).expect("a patch");
    assert_close(&after, &expected(&state));
    assert_ne!(after, before, "the stroke changed the picture");
}

#[test]
fn strokes_that_pile_up_while_the_worker_is_busy_merge_into_one_patch() {
    let mut state = state();
    let mut compositor = Compositor::default();
    pump(&mut state, &mut compositor).expect("a full composite");

    drag(&mut state, &[(6, 6), (9, 7)], 2.0);
    // The UI thread flushes after each tick; the worker has not taken the work yet.
    drop(state.flush_preview());
    drag(&mut state, &[(40, 30), (42, 31)], 2.0);

    let frame = pump(&mut state, &mut compositor).expect("one merged patch");
    assert_close(&frame, &expected(&state));
}

#[test]
fn undo_and_redo_replace_the_whole_mirror() {
    let mut state = state();
    let mut compositor = Compositor::default();
    let clean = pump(&mut state, &mut compositor).expect("a full composite");

    drag(&mut state, &[(12, 12), (25, 18)], 4.0);
    let painted = pump(&mut state, &mut compositor).expect("a patch");
    assert_close(&painted, &expected(&state));

    assert!(state.undo());
    let undone = pump(&mut state, &mut compositor).expect("a full composite after undo");
    assert_close(&undone, &clean);
    assert_close(&undone, &expected(&state));

    assert!(state.redo());
    let redone = pump(&mut state, &mut compositor).expect("a full composite after redo");
    assert_close(&redone, &painted);
}

#[test]
fn a_different_source_frame_replaces_the_whole_mirror() {
    let mut state = state();
    let mut compositor = Compositor::default();
    pump(&mut state, &mut compositor).expect("a full composite");
    drag(&mut state, &[(12, 12), (25, 18)], 4.0);
    pump(&mut state, &mut compositor).expect("a patch");

    // The same strokes over a darker frame: every painted pixel changes.
    state.choose_source(None);
    use_source(&mut state, 0.5);
    let reframed = pump(&mut state, &mut compositor).expect("a full composite");
    assert_close(&reframed, &expected(&state));

    // Clearing the layer shows the base again under the strokes.
    assert_eq!(state.choose_source(None), SourceChoice::Cleared);
    let cleared = pump(&mut state, &mut compositor).expect("a full composite");
    assert_close(&cleared, &expected(&state));
}

#[test]
fn a_new_session_starts_from_a_full_composite() {
    let mut state = state();
    let mut compositor = Compositor::default();
    pump(&mut state, &mut compositor).expect("a full composite");

    let base = Arc::new(solid(0.4));
    state.start_session(
        RetouchSession::new(Arc::clone(&base), base),
        PathBuf::from("stacker_result_Relief_2.png"),
    );
    let frame = pump(&mut state, &mut compositor).expect("a full composite");
    assert_close(&frame, &expected(&state));
}
