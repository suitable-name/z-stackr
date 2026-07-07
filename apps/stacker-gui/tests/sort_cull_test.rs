use std::path::PathBuf;

use stacker_gui::{
    settings::StackingSettings,
    sort_cull::{
        SortCullCacheOps, SortCullCacheState, SortCullDecision, SortCullOpPressed, SortCullWants,
        button_run_flags, compute_sort_cull_fingerprint, decide_sort_cull,
    },
};

const fn wants(sort: bool, cull: bool) -> SortCullWants {
    SortCullWants { sort, cull }
}

// ── decide_sort_cull ────────────────────────────────────────────────────

#[test]
fn decide_sort_cull_no_cache_always_computes() {
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Absent, wants(true, true)),
        SortCullDecision::Compute
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Absent, wants(false, false)),
        SortCullDecision::Compute
    );
}

#[test]
fn decide_sort_cull_stale_cache_always_asks() {
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Stale, wants(true, false)),
        SortCullDecision::Ask
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Stale, wants(false, true)),
        SortCullDecision::Ask
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Stale, wants(true, true)),
        SortCullDecision::Ask
    );
}

#[test]
fn decide_sort_cull_fresh_full_cache_skips() {
    let ops = SortCullCacheOps {
        ran_sort: true,
        ran_cull: true,
    };
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(true, true)),
        SortCullDecision::Skip
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(true, false)),
        SortCullDecision::Skip
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(false, true)),
        SortCullDecision::Skip
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(false, false)),
        SortCullDecision::Skip
    );
}

#[test]
fn decide_sort_cull_sort_only_cache_covers_sort_but_not_cull() {
    let ops = SortCullCacheOps {
        ran_sort: true,
        ran_cull: false,
    };
    // Wants only what the cache ran -> skip.
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(true, false)),
        SortCullDecision::Skip
    );
    // Wants cull too, but the cache never ran it -> compute (partial-cache rule).
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(true, true)),
        SortCullDecision::Compute
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(false, true)),
        SortCullDecision::Compute
    );
}

#[test]
fn decide_sort_cull_cull_only_cache_covers_cull_but_not_sort() {
    let ops = SortCullCacheOps {
        ran_sort: false,
        ran_cull: true,
    };
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(false, true)),
        SortCullDecision::Skip
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(true, true)),
        SortCullDecision::Compute
    );
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(true, false)),
        SortCullDecision::Compute
    );
}

#[test]
fn decide_sort_cull_fresh_cache_wanting_nothing_skips() {
    let ops = SortCullCacheOps {
        ran_sort: false,
        ran_cull: false,
    };
    // Neither op is wanted, so "not covered" is vacuously satisfied.
    assert_eq!(
        decide_sort_cull(SortCullCacheState::Fresh(ops), wants(false, false)),
        SortCullDecision::Skip
    );
}

// ── compute_sort_cull_fingerprint ───────────────────────────────────────

fn temp_file(dir: &std::path::Path, name: &str, contents: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write temp file");
    path
}

#[test]
fn fingerprint_stable_for_identical_inputs() {
    let dir = std::env::temp_dir().join("sort_cull_fp_stable");
    std::fs::create_dir_all(&dir).unwrap();
    let a = temp_file(&dir, "a.png", b"aaa");
    let b = temp_file(&dir, "b.png", b"bbb");
    let paths = vec![a, b];
    let settings = StackingSettings::default();

    let fp1 = compute_sort_cull_fingerprint(&paths, &settings);
    let fp2 = compute_sort_cull_fingerprint(&paths, &settings);
    assert_eq!(fp1, fp2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fingerprint_flips_on_order_change() {
    let dir = std::env::temp_dir().join("sort_cull_fp_order");
    std::fs::create_dir_all(&dir).unwrap();
    let a = temp_file(&dir, "a.png", b"aaa");
    let b = temp_file(&dir, "b.png", b"bbb");
    let settings = StackingSettings::default();

    let fp_ab = compute_sort_cull_fingerprint(&[a.clone(), b.clone()], &settings);
    let fp_ba = compute_sort_cull_fingerprint(&[b, a], &settings);
    assert_ne!(fp_ab, fp_ba);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fingerprint_flips_on_sort_cull_specific_setting_change() {
    let dir = std::env::temp_dir().join("sort_cull_fp_settings");
    std::fs::create_dir_all(&dir).unwrap();
    let a = temp_file(&dir, "a.png", b"aaa");
    let b = temp_file(&dir, "b.png", b"bbb");
    let paths = vec![a, b];

    let base = StackingSettings::default();
    let fp_base = compute_sort_cull_fingerprint(&paths, &base);

    // stack_every_nth affects Sort/Cull scoring (post-subsampling frame
    // set) but is not part of compute_align_fingerprint's inputs, so this
    // also exercises the Sort/Cull-specific additions on top of the
    // align-fingerprint base.
    let mut nth_changed = base.clone();
    nth_changed.stack_every_nth = base.stack_every_nth + 1;
    assert_ne!(fp_base, compute_sort_cull_fingerprint(&paths, &nth_changed));

    let mut threshold_changed = base.clone();
    threshold_changed.auto_cull_threshold_pct = if base.auto_cull_threshold_pct < 4.0 {
        base.auto_cull_threshold_pct + 1.0
    } else {
        base.auto_cull_threshold_pct - 1.0
    };
    assert_ne!(
        fp_base,
        compute_sort_cull_fingerprint(&paths, &threshold_changed)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fingerprint_ignores_stack_toggle_changes() {
    // With independent Sort/Cull buttons, `auto_cull`/`sort_by_sharpness`
    // (the Stack-run toggles) must NOT be part of this fingerprint —
    // toggling a Stack setting must never invalidate a button-produced
    // cache the button never consulted in the first place. Op coverage
    // is handled separately by `decide_sort_cull`'s `Fresh(ops)` vs
    // `wants` comparison.
    let dir = std::env::temp_dir().join("sort_cull_fp_toggle_invariant");
    std::fs::create_dir_all(&dir).unwrap();
    let a = temp_file(&dir, "a.png", b"aaa");
    let b = temp_file(&dir, "b.png", b"bbb");
    let paths = vec![a, b];

    let base = StackingSettings::default();
    let fp_base = compute_sort_cull_fingerprint(&paths, &base);

    let mut cull_changed = base.clone();
    cull_changed.auto_cull = !base.auto_cull;
    assert_eq!(
        fp_base,
        compute_sort_cull_fingerprint(&paths, &cull_changed)
    );

    let mut sort_changed = base.clone();
    sort_changed.sort_by_sharpness = !base.sort_by_sharpness;
    assert_eq!(
        fp_base,
        compute_sort_cull_fingerprint(&paths, &sort_changed)
    );

    let mut both_changed = base.clone();
    both_changed.auto_cull = !base.auto_cull;
    both_changed.sort_by_sharpness = !base.sort_by_sharpness;
    assert_eq!(
        fp_base,
        compute_sort_cull_fingerprint(&paths, &both_changed)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fingerprint_flips_on_alignment_affecting_setting_change() {
    // Sanity check that the align-fingerprint base is actually folded
    // in: an alignment-only setting change (not one of the Sort/Cull-
    // specific additions above) must still flip the Sort/Cull fingerprint.
    let dir = std::env::temp_dir().join("sort_cull_fp_align_base");
    std::fs::create_dir_all(&dir).unwrap();
    let a = temp_file(&dir, "a.png", b"aaa");
    let paths = vec![a];

    let base = StackingSettings::default();
    let mut changed = base.clone();
    changed.akaze_seeding = !base.akaze_seeding;

    assert_ne!(
        compute_sort_cull_fingerprint(&paths, &base),
        compute_sort_cull_fingerprint(&paths, &changed)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── button_run_flags ─────────────────────────────────────────────────

#[test]
fn button_run_flags_sort_press_absent_cache() {
    assert_eq!(
        button_run_flags(SortCullOpPressed::Sort, SortCullCacheState::Absent),
        (true, false)
    );
}

#[test]
fn button_run_flags_sort_press_fresh_ran_cull_cache() {
    let ops = SortCullCacheOps {
        ran_sort: false,
        ran_cull: true,
    };
    assert_eq!(
        button_run_flags(SortCullOpPressed::Sort, SortCullCacheState::Fresh(ops)),
        (true, true)
    );
}

#[test]
fn button_run_flags_sort_press_stale_ran_cull_cache() {
    assert_eq!(
        button_run_flags(SortCullOpPressed::Sort, SortCullCacheState::Stale),
        (true, false)
    );
}

#[test]
fn button_run_flags_sort_press_fresh_cache_without_cull() {
    // A fresh cache that never ran Cull must not fabricate one.
    let ops = SortCullCacheOps {
        ran_sort: true,
        ran_cull: false,
    };
    assert_eq!(
        button_run_flags(SortCullOpPressed::Sort, SortCullCacheState::Fresh(ops)),
        (true, false)
    );
}

#[test]
fn button_run_flags_cull_press_absent_cache() {
    assert_eq!(
        button_run_flags(SortCullOpPressed::Cull, SortCullCacheState::Absent),
        (false, true)
    );
}

#[test]
fn button_run_flags_cull_press_fresh_ran_sort_cache() {
    let ops = SortCullCacheOps {
        ran_sort: true,
        ran_cull: false,
    };
    assert_eq!(
        button_run_flags(SortCullOpPressed::Cull, SortCullCacheState::Fresh(ops)),
        (true, true)
    );
}

#[test]
fn button_run_flags_cull_press_stale_ran_sort_cache() {
    assert_eq!(
        button_run_flags(SortCullOpPressed::Cull, SortCullCacheState::Stale),
        (false, true)
    );
}

#[test]
fn button_run_flags_cull_press_fresh_cache_without_sort() {
    // A fresh cache that never ran Sort must not fabricate one.
    let ops = SortCullCacheOps {
        ran_sort: false,
        ran_cull: true,
    };
    assert_eq!(
        button_run_flags(SortCullOpPressed::Cull, SortCullCacheState::Fresh(ops)),
        (false, true)
    );
}
