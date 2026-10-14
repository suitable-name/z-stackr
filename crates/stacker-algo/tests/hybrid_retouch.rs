use stacker_algo::hybrid::retouch::*;
use stacker_core::image::PlanarImage;
use std::sync::Arc;

// ── helpers ───────────────────────────────────────────────────────────────

fn flat_image(val: f32) -> PlanarImage<f32> {
    PlanarImage::with_luma(10, 10, vec![val; 100]).expect("one sample per pixel")
}

fn make_session() -> RetouchSession {
    RetouchSession::new(Arc::new(flat_image(0.0)), Arc::new(flat_image(1.0)))
}

// ── RetouchSession ────────────────────────────────────────────────────────

#[test]
fn test_brush_paint_over_formula() {
    let mut s = make_session();
    // First stroke: opacity 0.5 → alpha = 0 + 0.5 * (1 − 0) = 0.5
    s.apply_brush(5, 5, 2.0, 0.5);
    let idx = 5 * 10 + 5;
    assert!((s.alpha.luma[idx] - 0.5).abs() < 1e-6);

    // Second stroke at same centre: 0.5 + 0.5 * (1 − 0.5) = 0.75
    s.apply_brush(5, 5, 2.0, 0.5);
    assert!((s.alpha.luma[idx] - 0.75).abs() < 1e-6);
}

#[test]
fn test_composite_alpha_one_yields_src() {
    // Paint alpha = 1 everywhere with a large brush.
    let mut s = make_session();
    s.apply_brush(5, 5, 20.0, 1.0); // radius larger than image → all pixels
    let comp = s.render_composite();
    // base = 0.0, src = 1.0 → all composited pixels should be 1.0.
    for v in &comp.luma {
        assert!((*v - 1.0).abs() < 1e-6, "expected 1.0 got {v}");
    }
}

#[test]
fn test_composite_alpha_zero_yields_base() {
    // Alpha is all-zero (no strokes) → composite should equal base.
    let s = make_session();
    let comp = s.render_composite();
    for v in &comp.luma {
        assert!(v.abs() < 1e-6, "expected 0.0 got {v}");
    }
}

#[test]
fn test_boundary_brush_does_not_panic() {
    let mut s = make_session();
    // Corner brushes and out-of-bounds centre must not panic.
    s.apply_brush(0, 0, 5.0, 1.0);
    s.apply_brush(9, 9, 5.0, 1.0);
    s.apply_brush(100, 100, 5.0, 1.0); // out-of-bounds — should be a no-op
    let comp = s.render_composite();
    assert_eq!(comp.luma[0], 1.0);
    assert_eq!(comp.luma[9 * 10 + 9], 1.0);
}

// ── RetouchHistory ────────────────────────────────────────────────────────

fn alpha(val: f32) -> Vec<f32> {
    vec![val; 100]
}

#[test]
fn test_push_and_undo_basic() {
    let mut h = RetouchHistory::new();
    assert!(!h.can_undo());
    assert!(!h.can_redo());

    h.push(alpha(0.1));
    h.push(alpha(0.2));
    h.push(alpha(0.3));
    assert_eq!(h.len(), 3);
    assert!(h.can_undo());
    assert!(!h.can_redo());

    // Undo to 0.2
    let snap = h.undo().expect("undo should succeed").clone();
    assert!((snap[0] - 0.2).abs() < 1e-6);

    // Undo to 0.1
    let snap = h.undo().expect("undo should succeed").clone();
    assert!((snap[0] - 0.1).abs() < 1e-6);

    // The pristine (all-zero) mask is on the stack: the first push is undoable.
    let snap = h.undo().expect("first push is undoable").clone();
    assert!(snap.iter().all(|&v| v == 0.0));

    // Cannot undo past the pristine state.
    assert!(h.undo().is_none());
    assert!(!h.can_undo());
}

#[test]
fn test_redo_after_undo() {
    let mut h = RetouchHistory::new();
    h.push(alpha(0.1));
    h.push(alpha(0.2));
    h.push(alpha(0.3));

    h.undo(); // at 0.2
    h.undo(); // at 0.1

    // Redo back to 0.2
    let snap = h.redo().expect("redo should succeed").clone();
    assert!((snap[0] - 0.2).abs() < 1e-6);

    // Redo back to 0.3
    let snap = h.redo().expect("redo should succeed").clone();
    assert!((snap[0] - 0.3).abs() < 1e-6);

    // Cannot redo past the end.
    assert!(h.redo().is_none());
    assert!(!h.can_redo());
}

#[test]
fn test_new_stroke_truncates_redo_branch() {
    let mut h = RetouchHistory::new();
    h.push(alpha(0.1)); // s0
    h.push(alpha(0.2)); // s1
    h.push(alpha(0.3)); // s2

    h.undo(); // cursor → s1
    h.undo(); // cursor → s0

    // New stroke: s1 and s2 must be discarded.
    h.push(alpha(0.9));
    assert_eq!(h.len(), 2, "s0 + new stroke = 2 entries");
    assert!(!h.can_redo(), "redo branch must be gone");

    let snap = h.undo().expect("undo should succeed").clone();
    assert!((snap[0] - 0.1).abs() < 1e-6, "s0 still intact");
}

/// FUS-8: the pristine state is recorded, so the first stroke can be undone
/// (this used to return `None`), and nothing lies before it.
#[test]
fn test_first_push_is_undoable_to_pristine() {
    let mut h = RetouchHistory::new();
    h.push(alpha(0.5));
    assert!(h.can_undo());
    let pristine = h.undo().expect("first stroke must be undoable").clone();
    assert!(pristine.iter().all(|&v| v == 0.0));
    assert!(h.undo().is_none());
    // Stack intact, len still 1, and the stroke can be redone.
    assert_eq!(h.len(), 1);
    let redone = h.redo().expect("redo of the first stroke").clone();
    assert!(redone.iter().all(|&v| (v - 0.5).abs() < 1e-6));
}

#[test]
fn test_redo_past_end_is_noop() {
    let mut h = RetouchHistory::new();
    h.push(alpha(0.5));
    assert!(h.redo().is_none());
    assert_eq!(h.len(), 1);
}

#[test]
fn test_memory_bound_evicts_oldest() {
    let mut h = RetouchHistory::new();
    // Push MAX_DEPTH + 5 entries; oldest 5 should be evicted.
    for i in 0..=RetouchHistory::MAX_DEPTH + 4 {
        h.push(alpha(
            f32::from(u8::try_from(i).expect("at most 54 pushes")) * 0.01,
        ));
    }
    assert_eq!(
        h.len(),
        RetouchHistory::MAX_DEPTH,
        "history must be capped at MAX_DEPTH"
    );
    // Every retained step is undoable, no more (the evicted steps are gone).
    for _ in 0..RetouchHistory::MAX_DEPTH {
        assert!(h.undo().is_some());
    }
    // Should be at the bottom with no more undos available.
    assert!(!h.can_undo());
}

#[test]
fn test_empty_history_undo_redo_safe() {
    let mut h = RetouchHistory::new();
    assert!(h.undo().is_none());
    assert!(h.redo().is_none());
    assert!(h.is_empty());
}

/// Integration: apply N strokes, undo K, verify alpha state at each step.
#[test]
fn test_stroke_undo_redo_alpha_state() {
    let mut session = make_session();
    let mut history = RetouchHistory::new();

    // Stroke 0: paint centre with opacity 1.0 → alpha[55] = 1.0
    session.apply_brush(5, 5, 1.0, 1.0);
    history.push(session.snapshot_alpha());
    let after_s0 = session.alpha.luma[55];

    // Stroke 1: paint a different pixel (0,0) with opacity 0.6
    session.apply_brush(0, 0, 0.5, 0.6);
    history.push(session.snapshot_alpha());
    let after_s1_at_0 = session.alpha.luma[0];

    // Stroke 2: paint (9,9)
    session.apply_brush(9, 9, 0.5, 0.8);
    history.push(session.snapshot_alpha());

    assert_eq!(history.len(), 3);

    // Undo stroke 2 → state should match after_s1
    let snap = history.undo().expect("undo s2").clone();
    session.restore_alpha(snap);
    assert!((session.alpha.luma[55] - after_s0).abs() < 1e-6);
    assert!((session.alpha.luma[0] - after_s1_at_0).abs() < 1e-6);
    // pixel (9,9) should be zero again (stroke 2 undone)
    assert!(session.alpha.luma[9 * 10 + 9] < 1e-6);

    // Undo stroke 1 → state should match after_s0
    let snap = history.undo().expect("undo s1").clone();
    session.restore_alpha(snap);
    assert!((session.alpha.luma[55] - after_s0).abs() < 1e-6);
    assert!(
        session.alpha.luma[0] < 1e-6,
        "pixel 0 should be unpainted after undo s1"
    );

    // Redo stroke 1
    let snap = history.redo().expect("redo s1").clone();
    session.restore_alpha(snap);
    assert!((session.alpha.luma[0] - after_s1_at_0).abs() < 1e-6);

    // New stroke after partial undo → truncates redo branch
    session.apply_brush(3, 3, 0.5, 1.0);
    history.push(session.snapshot_alpha());
    assert_eq!(history.len(), 3, "s0 + s1 + new = 3");
    assert!(!history.can_redo());
}

// ── Stroke API (FUS-8) ────────────────────────────────────────────────────

/// One drag: begin, one dab per point, end.
fn drag(
    h: &mut RetouchHistory,
    s: &mut RetouchSession,
    points: &[(usize, usize)],
    radius: f32,
    opacity: f32,
) -> bool {
    h.begin_stroke();
    for &(x, y) in points {
        h.extend_stroke(s, x, y, radius, opacity);
    }
    h.end_stroke(s)
}

/// A drag of many dabs is exactly one undo step, and undo/redo across several
/// strokes restores the exact alpha plane of each state, including the
/// pristine one before the first stroke.
#[test]
fn test_stroke_api_undo_redo_across_strokes() {
    let mut s = make_session();
    let mut h = RetouchHistory::new();
    let pristine = s.snapshot_alpha();

    assert!(drag(
        &mut h,
        &mut s,
        &[(2, 2), (3, 2), (4, 2), (5, 2)],
        1.5,
        0.7
    ));
    assert_eq!(h.len(), 1, "four dabs of one drag are one undo step");
    let after_a = s.snapshot_alpha();
    assert_ne!(after_a, pristine);

    assert!(drag(&mut h, &mut s, &[(7, 7), (7, 8)], 1.0, 1.0));
    assert_eq!(h.len(), 2);
    let after_b = s.snapshot_alpha();
    assert_ne!(after_b, after_a);

    // Undo B, then A: exact states, down to the pristine mask.
    assert!(h.undo_stroke(&mut s));
    assert_eq!(s.snapshot_alpha(), after_a);
    assert!(h.undo_stroke(&mut s));
    assert_eq!(
        s.snapshot_alpha(),
        pristine,
        "first stroke must be undoable"
    );
    assert!(!h.can_undo());
    assert!(!h.undo_stroke(&mut s));

    // Redo A, B: exact again.
    assert!(h.redo_stroke(&mut s));
    assert_eq!(s.snapshot_alpha(), after_a);
    assert!(h.redo_stroke(&mut s));
    assert_eq!(s.snapshot_alpha(), after_b);
    assert!(!h.can_redo());

    // Undo B, paint C: B's redo branch is gone.
    assert!(h.undo_stroke(&mut s));
    assert!(drag(&mut h, &mut s, &[(0, 9)], 1.0, 1.0));
    assert_eq!(h.len(), 2, "A + C");
    assert!(!h.can_redo());
    let after_c = s.snapshot_alpha();
    assert!(h.undo_stroke(&mut s));
    assert_eq!(s.snapshot_alpha(), after_a);
    assert!(h.redo_stroke(&mut s));
    assert_eq!(s.snapshot_alpha(), after_c);
}

/// A stroke that changes nothing records no step and keeps the redo branch.
#[test]
fn test_noop_stroke_records_nothing() {
    let mut s = make_session();
    let mut h = RetouchHistory::new();
    assert!(drag(&mut h, &mut s, &[(5, 5)], 2.0, 1.0));
    assert!(h.undo_stroke(&mut s));
    assert!(h.can_redo());

    // Opacity 0 and an out-of-bounds centre change nothing.
    assert!(!drag(&mut h, &mut s, &[(5, 5)], 2.0, 0.0));
    assert!(!drag(&mut h, &mut s, &[(500, 500)], 2.0, 1.0));
    assert!(
        h.can_redo(),
        "no-op strokes must not truncate the redo branch"
    );
    assert_eq!(h.len(), 1);
}

#[test]
fn test_cancel_stroke_restores_alpha() {
    let mut s = make_session();
    let mut h = RetouchHistory::new();
    let pristine = s.snapshot_alpha();
    h.begin_stroke();
    h.extend_stroke(&mut s, 5, 5, 3.0, 1.0);
    assert!(!h.can_undo(), "no undo while a stroke is open");
    h.cancel_stroke(&mut s);
    assert_eq!(s.snapshot_alpha(), pristine);
    assert!(h.is_empty());
}

/// More strokes than `MAX_DEPTH`: the oldest are evicted, the rest undo exactly.
#[test]
fn test_stroke_history_is_capped() {
    let mut s = make_session();
    let mut h = RetouchHistory::new();
    for i in 0..RetouchHistory::MAX_DEPTH + 7 {
        assert!(drag(&mut h, &mut s, &[(i % 10, (i / 10) % 10)], 0.5, 0.5));
    }
    assert_eq!(h.len(), RetouchHistory::MAX_DEPTH);
    for _ in 0..RetouchHistory::MAX_DEPTH {
        assert!(h.undo_stroke(&mut s));
    }
    assert!(!h.can_undo());
}

/// Memory: a stroke touching 100 x 100 pixels of a 2000 x 2000 plane stores a
/// small fraction of the plane, not a full snapshot.
#[test]
fn test_stroke_memory_is_proportional_to_the_touched_area() {
    let (w, h_px) = (2000, 2000);
    let image = Arc::new(PlanarImage::<f32>::new(w, h_px));
    let mut s = RetouchSession::new(Arc::clone(&image), image);
    let mut h = RetouchHistory::new();

    // A 100 x 100 px dab (radius 50) plus a few overlapping dabs in the same box.
    assert!(drag(
        &mut h,
        &mut s,
        &[(1000, 1000), (1010, 1000), (1000, 1010), (990, 990)],
        50.0,
        1.0
    ));
    let stored = h.stored_values();
    let plane = w * h_px;
    assert!(stored > 0);
    assert!(
        stored < plane / 50,
        "stored {stored} values for a ~110 x 110 px stroke on a {plane}-value plane"
    );

    // And it still undoes exactly.
    assert!(h.undo_stroke(&mut s));
    assert!(s.alpha.luma.iter().all(|&v| v == 0.0));
}
