//! The retouch state without a window: strokes and undo steps, labels, pointer
//! mapping, the source layer and the composite.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use stacker_algo::hybrid::retouch::RetouchSession;
use stacker_core::image::PlanarImage;
use stacker_gui::{
    retouch::{
        BLOCKED_NO_RESULT, BLOCKED_NO_SOURCE, BLOCKED_NOT_NEWEST, RetouchState, SourceChoice,
        SourceError, SourceRequest,
    },
    state::{AppState, SaveTarget},
    viewport_map::View,
};

const W: usize = 40;
const H: usize = 30;
const EPS: f32 = 1e-6;

fn solid(w: usize, h: usize, level: f32) -> PlanarImage<f32> {
    let mut image = PlanarImage::new(w, h);
    image.luma.fill(level);
    image
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < EPS
}

fn result_path() -> PathBuf {
    PathBuf::from("stacker_result_Apex_1.png")
}

/// A state with a session over a dark result; the picker lists two frames.
fn state() -> RetouchState {
    let base = Arc::new(solid(W, H, 0.1));
    let mut state = RetouchState::new();
    state.start_session(RetouchSession::new(Arc::clone(&base), base), result_path());
    state.refresh_sources(None, &[PathBuf::from("a.png"), PathBuf::from("b.png")]);
    state
}

/// Choose source `index` and install a solid frame of `level` for it.
fn use_source(state: &mut RetouchState, index: usize, level: f32) {
    let SourceChoice::Load(request) = state.choose_source(Some(index)) else {
        panic!("choosing source {index} should ask for a frame");
    };
    assert_eq!(state.finish_source(&request, solid(W, H, level)), Ok(true));
}

fn alpha(state: &RetouchState) -> &[f32] {
    &state.session.as_ref().expect("a session").alpha.luma
}

fn alpha_at(state: &RetouchState, x: usize, y: usize) -> f32 {
    alpha(state)[y * W + x]
}

/// One complete drag through `points` at full opacity.
fn drag(state: &mut RetouchState, points: &[(u32, u32)], radius: f32) -> bool {
    assert!(state.begin_stroke());
    for &point in points {
        state.queue_point(Some(point), radius, 1.0);
        state.drain_pending();
    }
    state.end_stroke()
}

#[test]
fn a_drag_is_one_undo_step() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);

    // Many moves, drained at several different moments, still one step.
    assert!(state.begin_stroke());
    for x in 5..25_u32 {
        state.queue_point(Some((x, 10)), 2.0, 1.0);
        if x % 4 == 0 {
            state.drain_pending();
        }
    }
    assert!(!state.can_undo(), "no step is recorded while the drag runs");
    assert!(state.end_stroke());

    assert_eq!(state.history.len(), 1);
    assert!(state.can_undo());
    assert!(approx(alpha_at(&state, 15, 10), 1.0));

    assert!(state.undo());
    assert!(
        alpha(&state).iter().all(|&a| approx(a, 0.0)),
        "undo restores the pristine mask"
    );
    assert!(!state.can_undo());
    assert!(state.can_redo());

    assert!(state.redo());
    assert!(approx(alpha_at(&state, 15, 10), 1.0));
}

#[test]
fn two_drags_are_two_steps_and_a_new_drag_drops_the_redo_branch() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(drag(&mut state, &[(5, 5)], 2.0));
    assert!(drag(&mut state, &[(30, 20)], 2.0));
    assert_eq!(state.history.len(), 2);

    assert!(state.undo());
    assert!(state.can_redo());
    assert!(drag(&mut state, &[(20, 5)], 2.0));
    assert!(
        !state.can_redo(),
        "painting after an undo discards the redo branch"
    );
    assert_eq!(state.history.len(), 2);
}

#[test]
fn undo_and_redo_labels_follow_the_history() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert_eq!(state.undo_label(), "");
    assert_eq!(state.redo_label(), "");

    assert!(drag(&mut state, &[(10, 10)], 2.0));
    assert_eq!(state.undo_label(), "Undo: brush stroke");
    assert_eq!(state.redo_label(), "");

    assert!(state.undo());
    assert_eq!(state.undo_label(), "");
    assert_eq!(state.redo_label(), "Redo: brush stroke");
}

#[test]
fn history_stores_blocks_not_planes() {
    let base = Arc::new(solid(512, 512, 0.1));
    let mut state = RetouchState::new();
    state.start_session(RetouchSession::new(Arc::clone(&base), base), result_path());
    assert!(state.begin_stroke());
    state.queue_point(Some((256, 256)), 4.0, 1.0);
    assert!(state.end_stroke());
    let plane = 512 * 512;
    assert!(
        state.history.stored_values() < plane / 50,
        "a small dab must not store a plane ({} of {plane})",
        state.history.stored_values()
    );
}

#[test]
fn a_stroke_outside_the_image_is_ignored() {
    // A 40x30 image on a 400x100 canvas: scale 3.33, centred, letterboxed at the sides.
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    let canvas = (400.0, 100.0);
    let view = View::default();
    assert_eq!(
        state.map_pointer(canvas, view, 5.0, 50.0),
        None,
        "left letterbox"
    );
    assert_eq!(
        state.map_pointer(canvas, view, 395.0, 50.0),
        None,
        "right letterbox"
    );
    assert_eq!(state.map_pointer(canvas, view, 202.0, 52.0), Some((20, 15)));

    assert!(state.begin_stroke());
    let outside = state.map_pointer(canvas, view, 5.0, 50.0);
    state.queue_point(outside, 5.0, 1.0);
    assert_eq!(state.drain_pending(), 0);
    assert!(
        !state.end_stroke(),
        "a stroke that painted nothing is not a step"
    );
    assert!(!state.can_undo());
    assert!(alpha(&state).iter().all(|&a| approx(a, 0.0)));
}

#[test]
fn leaving_and_re_entering_the_image_does_not_draw_a_line_across_the_gap() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(state.begin_stroke());
    state.queue_point(Some((2, 15)), 1.0, 1.0);
    state.queue_point(None, 1.0, 1.0);
    state.queue_point(Some((37, 15)), 1.0, 1.0);
    state.drain_pending();
    assert!(state.end_stroke());
    assert!(approx(alpha_at(&state, 2, 15), 1.0));
    assert!(approx(alpha_at(&state, 37, 15), 1.0));
    assert!(
        approx(alpha_at(&state, 20, 15), 0.0),
        "the gap stays unpainted"
    );
}

#[test]
fn a_fast_drag_is_filled_in_between_the_reported_points() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(drag(&mut state, &[(3, 15), (35, 15)], 2.0));
    for x in 3..=35 {
        assert!(alpha_at(&state, x, 15) > 0.5, "gap at x = {x}");
    }
}

#[test]
fn repeated_pointer_positions_paint_once() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(state.begin_stroke());
    for _ in 0..5 {
        state.queue_point(Some((10, 10)), 3.0, 0.5);
    }
    assert_eq!(state.drain_pending(), 1);
    // Reporting the same pixel again in a later tick does not deepen the paint.
    state.queue_point(Some((10, 10)), 3.0, 0.5);
    assert_eq!(state.drain_pending(), 0);
    assert!(state.end_stroke());
    assert!(approx(alpha_at(&state, 10, 10), 0.5));
}

#[test]
fn moves_without_an_open_stroke_are_ignored() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    state.queue_point(Some((10, 10)), 3.0, 1.0);
    assert_eq!(state.drain_pending(), 0);
    assert!(alpha(&state).iter().all(|&a| approx(a, 0.0)));
}

#[test]
fn cancelling_a_stroke_restores_the_mask() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(state.begin_stroke());
    state.queue_point(Some((10, 10)), 3.0, 1.0);
    state.drain_pending();
    assert!(approx(alpha_at(&state, 10, 10), 1.0));
    state.cancel_stroke();
    assert!(approx(alpha_at(&state, 10, 10), 0.0));
    assert!(!state.stroke_open());
    assert!(!state.can_undo());
}

#[test]
fn switching_the_source_keeps_the_strokes_and_the_history() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(drag(&mut state, &[(10, 10)], 3.0));
    let painted = alpha(&state).to_vec();
    let bright = state.composite_f32().expect("a session");
    assert!(approx(bright.luma[10 * W + 10], 1.0));

    use_source(&mut state, 1, 0.5);
    assert_eq!(state.source_index(), Some(1));
    assert_eq!(
        state.history.len(),
        1,
        "the history survives a source switch"
    );
    assert!(state.can_undo());
    assert_eq!(alpha(&state), painted.as_slice(), "the mask is untouched");

    // The painted region now shows the new frame; the rest still shows the base.
    let dim = state.composite_f32().expect("a session");
    assert!(approx(dim.luma[10 * W + 10], 0.5));
    assert!(approx(dim.luma[0], 0.1));

    // Undo still works and reaches the pristine mask.
    assert!(state.undo());
    assert!(alpha(&state).iter().all(|&a| approx(a, 0.0)));
}

#[test]
fn clearing_the_source_returns_to_the_base() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(drag(&mut state, &[(10, 10)], 3.0));
    assert_eq!(state.choose_source(None), SourceChoice::Cleared);
    assert_eq!(state.source_index(), None);
    let image = state.composite_f32().expect("a session");
    assert!(approx(image.luma[10 * W + 10], 0.1));
    assert_eq!(state.history.len(), 1);
}

#[test]
fn a_late_frame_for_a_superseded_choice_is_dropped() {
    let mut state = state();
    let SourceChoice::Load(first) = state.choose_source(Some(0)) else {
        panic!("expected a load request");
    };
    let SourceChoice::Load(second) = state.choose_source(Some(1)) else {
        panic!("expected a load request");
    };
    assert_eq!(state.finish_source(&first, solid(W, H, 1.0)), Ok(false));
    assert!(
        !state.fail_source(&first),
        "a stale failure changes nothing"
    );
    assert_eq!(state.source_index(), Some(1));
    assert_eq!(state.finish_source(&second, solid(W, H, 0.7)), Ok(true));
}

#[test]
fn a_frame_of_the_wrong_size_is_rejected_and_the_selection_reverts() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    let SourceChoice::Load(request) = state.choose_source(Some(1)) else {
        panic!("expected a load request");
    };
    assert_eq!(request.previous, Some(0));
    let result = state.finish_source(&request, solid(W + 1, H, 0.5));
    assert_eq!(
        result,
        Err(SourceError::SizeMismatch {
            expected: (W, H),
            found: (W + 1, H)
        })
    );
    assert!(state.fail_source(&request));
    assert_eq!(state.source_index(), Some(0));
    // The layer is still the first source.
    let image = state.composite_f32().expect("a session");
    assert!(
        approx(image.luma[0], 0.1),
        "no stroke yet, so the base shows"
    );
}

#[test]
fn choosing_an_unknown_entry_or_the_same_one_changes_nothing() {
    let mut state = state();
    assert_eq!(state.choose_source(None), SourceChoice::Unchanged);
    assert_eq!(state.choose_source(Some(9)), SourceChoice::Unchanged);
    use_source(&mut state, 1, 0.5);
    assert_eq!(state.choose_source(Some(1)), SourceChoice::Unchanged);
}

#[test]
fn the_picker_lists_frames_without_result_files_and_follows_the_selection() {
    let mut state = state();
    use_source(&mut state, 1, 0.5);
    assert_eq!(state.source_labels(), vec!["1. a.png", "2. b.png"]);

    // The list is rebuilt with a result file in it and a new frame in front.
    let files = [
        PathBuf::from("new.png"),
        PathBuf::from("a.png"),
        PathBuf::from("b.png"),
        PathBuf::from("stacker_result_Relief_7.png"),
    ];
    assert!(state.refresh_sources(None, &files));
    assert_eq!(state.sources().len(), 3);
    assert_eq!(state.source_index(), Some(2), "the selection follows b.png");
    assert!(
        !state.refresh_sources(None, &files),
        "an unchanged list reports no change"
    );

    // The selected frame disappears: the layer is cleared.
    assert!(state.refresh_sources(None, &[PathBuf::from("a.png")]));
    assert_eq!(state.source_index(), None);
}

#[test]
fn the_blocker_names_what_is_missing() {
    let result = result_path();
    let result = result.as_path();
    let empty = RetouchState::new();
    assert_eq!(empty.blocker(Some(result)), Some(BLOCKED_NO_RESULT));

    let mut state = state();
    assert_eq!(
        state.blocker(Some(Path::new("a.png"))),
        Some(BLOCKED_NOT_NEWEST)
    );
    assert_eq!(state.blocker(None), Some(BLOCKED_NOT_NEWEST));
    assert_eq!(state.blocker(Some(result)), Some(BLOCKED_NO_SOURCE));
    use_source(&mut state, 0, 1.0);
    assert_eq!(state.blocker(Some(result)), None);
}

#[test]
fn a_new_session_forgets_the_history_and_the_source_choice() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(drag(&mut state, &[(10, 10)], 3.0));
    let base = Arc::new(solid(W, H, 0.2));
    state.start_session(RetouchSession::new(Arc::clone(&base), base), result_path());
    assert!(!state.can_undo());
    assert_eq!(state.source_index(), None);
    assert!(alpha(&state).iter().all(|&a| approx(a, 0.0)));
}

#[test]
fn the_composite_blends_base_and_source_by_the_mask() {
    let mut state = state();
    use_source(&mut state, 0, 1.0);
    assert!(state.begin_stroke());
    state.queue_point(Some((10, 10)), 3.0, 1.0);
    state.queue_point(None, 3.0, 1.0);
    state.queue_point(Some((30, 20)), 3.0, 0.25);
    assert!(state.end_stroke());

    let image = state.composite_f32().expect("a session");
    // Full-opacity dab: the source. Quarter-opacity dab: base + 0.25 (source - base).
    assert!(approx(image.luma[10 * W + 10], 1.0));
    assert!(approx(image.luma[20 * W + 30], 0.25_f32.mul_add(0.9, 0.1)));
    // Untouched pixels: the base, in every plane.
    assert!(approx(image.luma[0], 0.1));
    assert!(approx(image.chroma_a[0], 0.0));

    // What a worker composites from the snapshot is the same image.
    let copy = state
        .snapshot_session()
        .expect("a session")
        .render_composite();
    assert_eq!(copy.luma, image.luma);

    // The preview matches the composite's size and the buttons' state.
    let (buffer, can_undo, can_redo) = state.render_frame().expect("a session");
    assert_eq!((buffer.width() as usize, buffer.height() as usize), (W, H));
    assert!(can_undo);
    assert!(!can_redo);
}

#[test]
fn export_uses_the_session_only_for_the_result_it_belongs_to() {
    let state = state();
    let result = result_path();
    assert!(state.export_session(Some(result.as_path())).is_some());
    assert!(state.export_session(Some(Path::new("a.png"))).is_none());
    assert!(state.export_session(None).is_none());
}

// ── AppState glue ────────────────────────────────────────────────────────────

#[test]
fn the_session_follows_the_newest_fused_result() {
    let mut app = AppState::default();
    assert!(!app.ensure_retouch_session(), "nothing fused yet");

    let first = Arc::new(solid(W, H, 0.3));
    app.fused = Some(Arc::clone(&first));
    app.push_result(PathBuf::from("r1.png"));
    assert!(app.ensure_retouch_session());
    assert!(!app.ensure_retouch_session(), "already up to date");
    assert_eq!(app.retouch.result_path, Some(PathBuf::from("r1.png")));
    assert!(app.viewing_retouchable());

    let second = Arc::new(solid(W, H, 0.6));
    app.fused = Some(second);
    app.push_result(PathBuf::from("r2.png"));
    assert!(app.ensure_retouch_session());
    assert_eq!(app.retouch.result_path, Some(PathBuf::from("r2.png")));
}

#[test]
fn a_session_started_elsewhere_fills_the_fused_image() {
    let mut app = AppState::default();
    let base = Arc::new(solid(W, H, 0.3));
    app.retouch.start_session(
        RetouchSession::new(Arc::clone(&base), Arc::clone(&base)),
        PathBuf::from("r1.png"),
    );
    assert!(!app.ensure_retouch_session());
    assert!(app.fused.as_ref().is_some_and(|f| Arc::ptr_eq(f, &base)));
}

#[test]
fn save_encodes_the_result_only_when_the_result_is_on_the_canvas() {
    let mut app = AppState {
        fused: Some(Arc::new(solid(W, H, 0.3))),
        ..AppState::default()
    };
    app.push_result(PathBuf::from("r1.png"));
    app.ensure_retouch_session();
    assert!(matches!(app.save_target(), Some(SaveTarget::Result(_))));

    app.add_paths([PathBuf::from("frame.png")]);
    app.select_source(0);
    assert!(
        matches!(app.save_target(), Some(SaveTarget::File(path)) if path.as_path() == Path::new("frame.png"))
    );

    app.viewed_path = None;
    assert!(app.save_target().is_none());
}

#[test]
fn a_blocked_press_names_the_reason_through_the_app_state() {
    let mut app = AppState::default();
    assert_eq!(app.retouch_blocker(), Some(BLOCKED_NO_RESULT));
    app.fused = Some(Arc::new(solid(W, H, 0.3)));
    app.push_result(PathBuf::from("r1.png"));
    app.ensure_retouch_session();
    assert_eq!(app.retouch_blocker(), Some(BLOCKED_NO_SOURCE));
    app.add_paths([PathBuf::from("frame.png")]);
    app.select_source(0);
    assert_eq!(app.retouch_blocker(), Some(BLOCKED_NOT_NEWEST));
}

#[test]
fn a_request_carries_the_chosen_path() {
    let mut state = state();
    let SourceChoice::Load(request) = state.choose_source(Some(1)) else {
        panic!("expected a load request");
    };
    let expected = SourceRequest {
        generation: request.generation,
        index: 1,
        previous: None,
        path: PathBuf::from("b.png"),
    };
    assert_eq!(request, expected);
}
