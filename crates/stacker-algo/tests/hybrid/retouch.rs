#![allow(clippy::float_cmp)]

use stacker_algo::hybrid::retouch::*;
use stacker_core::image::PlanarImage;
use std::sync::Arc;

// ── helpers ───────────────────────────────────────────────────────────────

fn flat_image(val: f32) -> PlanarImage<f32> {
    PlanarImage {
        width: 10,
        height: 10,
        luma: vec![val; 100],
        chroma_a: vec![0.0; 100],
        chroma_b: vec![0.0; 100],
    }
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

    // Cannot undo past the beginning.
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

#[test]
fn test_undo_past_beginning_is_noop() {
    let mut h = RetouchHistory::new();
    h.push(alpha(0.5));
    assert!(h.undo().is_none()); // only 1 entry, cursor at 0 → no earlier state
    // Stack intact, len still 1.
    assert_eq!(h.len(), 1);
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
        
        h.push(alpha(i as f32 * 0.01));
    }
    assert_eq!(
        h.len(),
        RetouchHistory::MAX_DEPTH,
        "history must be capped at MAX_DEPTH"
    );
    // The oldest retained entry should be index 5 (val = 0.05).
    // After eviction cursor points at the newest (index MAX_DEPTH-1).
    // Undo all the way to the oldest retained.
    let target = RetouchHistory::MAX_DEPTH - 1;
    for _ in 0..target {
        h.undo();
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

/// Integration: reproduces the GUI's session-creation wiring (push the
/// pristine, all-zero alpha snapshot as the history baseline *before* any
/// stroke is applied — see `RetouchState`'s session-creation sites in
/// `apps/stacker-gui/src/main.rs`). Without that baseline push, the first
/// stroke's `history.push` would land at cursor 0 with nothing earlier in
/// the stack to undo back to, so "undo after the very first stroke" would
/// silently do nothing.
#[test]
fn test_baseline_snapshot_allows_undo_of_first_stroke() {
    let mut session = make_session();
    let mut history = RetouchHistory::new();

    // Mirrors the GUI: seed the history with the pristine mask immediately
    // after session creation, before any brush input.
    history.push(session.snapshot_alpha());
    assert!(!history.can_undo(), "baseline alone has nothing earlier");

    // First stroke.
    session.apply_brush(5, 5, 2.0, 1.0);
    let idx = 5 * 10 + 5;
    assert!((session.alpha.luma[idx] - 1.0).abs() < 1e-6);
    history.push(session.snapshot_alpha());
    assert!(
        history.can_undo(),
        "after baseline + one stroke, undo must be available"
    );

    // Undo the first stroke → must restore the pristine (all-zero) baseline.
    let snap = history.undo().expect("undo should restore baseline").clone();
    session.restore_alpha(snap);
    for v in &session.alpha.luma {
        assert!(v.abs() < 1e-6, "expected all-zero baseline, got {v}");
    }
    assert!(
        !history.can_undo(),
        "baseline is the oldest state — no further undo"
    );

    // Redo restores the stroke.
    let snap = history.redo().expect("redo should restore the stroke").clone();
    session.restore_alpha(snap);
    assert!((session.alpha.luma[idx] - 1.0).abs() < 1e-6);
    assert!(!history.can_redo(), "stroke was the newest state");
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
