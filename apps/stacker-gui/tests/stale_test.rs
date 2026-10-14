//! Generation stamps: fingerprints of the inputs behind the aligned cache and the
//! stacked results, and how `AppState` answers "is this still current?".

use std::{fs, path::PathBuf, process};

use stacker_gui::{
    settings::{AlignmentModeSetting, ComputeDevice, OptimizerSetting, StackingSettings},
    stale::{
        Fingerprint, SourceStamp, StaleReason, fingerprint_alignment, fingerprint_alignment_with,
        fingerprint_paths, fingerprint_sources, fingerprint_stack, fingerprint_stack_full,
        stale_reason, stale_view, summarize_settings,
    },
    state::{AlignedCache, AppState, ResultMeta},
};

/// A scratch directory with `names` written into it (each file holds its own name
/// as content), removed on drop.
struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    fn new(tag: &str, names: &[&str]) -> Self {
        let dir = std::env::temp_dir().join(format!("z-stackr-stale-{tag}-{}", process::id()));
        fs::create_dir_all(&dir).expect("create the scratch directory");
        let scratch = Self { dir };
        for name in names {
            fs::write(scratch.path(name), name.as_bytes()).expect("write a scratch file");
        }
        scratch
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    fn paths(&self, names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(|name| self.path(name)).collect()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn sources_of(paths: &[PathBuf]) -> Fingerprint {
    fingerprint_paths(paths)
}

/// A state whose list is `paths` and whose aligned cache was built from exactly
/// that list with `settings`.
fn aligned_state(paths: Vec<PathBuf>, settings: &StackingSettings) -> AppState {
    let mut state = AppState::default();
    state.add_paths(paths.clone());
    let fingerprint = fingerprint_alignment(&sources_of(&state.file_paths), settings);
    state.set_aligned(
        AlignedCache {
            paths,
            crop: Some((0, 0, 1, 1)),
        },
        fingerprint,
    );
    state
}

// ── Fingerprints ─────────────────────────────────────────────────────────────

#[test]
fn the_same_inputs_give_the_same_fingerprint() {
    let scratch = Scratch::new("same", &["a.png", "b.png"]);
    let paths = scratch.paths(&["a.png", "b.png"]);
    assert_eq!(sources_of(&paths), sources_of(&paths));
    let settings = StackingSettings::default();
    let again = StackingSettings::default();
    let sources = sources_of(&paths);
    assert_eq!(
        fingerprint_alignment(&sources, &settings),
        fingerprint_alignment(&sources, &again)
    );
    assert_eq!(
        fingerprint_stack(&sources, &settings),
        fingerprint_stack(&sources, &again)
    );
}

#[test]
fn reordering_or_adding_a_file_changes_the_sources_fingerprint() {
    let scratch = Scratch::new("order", &["a.png", "b.png", "c.png"]);
    let base = sources_of(&scratch.paths(&["a.png", "b.png"]));
    assert_ne!(base, sources_of(&scratch.paths(&["b.png", "a.png"])));
    assert_ne!(
        base,
        sources_of(&scratch.paths(&["a.png", "b.png", "c.png"]))
    );
    assert_ne!(base, sources_of(&scratch.paths(&["a.png"])));
    assert_ne!(base, sources_of(&[]));
}

#[test]
fn modifying_or_deleting_a_file_changes_the_sources_fingerprint() {
    let scratch = Scratch::new("modify", &["a.png", "b.png"]);
    let paths = scratch.paths(&["a.png", "b.png"]);
    let before = sources_of(&paths);

    fs::write(scratch.path("b.png"), b"a much longer replacement body").expect("rewrite b.png");
    let after_edit = sources_of(&paths);
    assert_ne!(before, after_edit, "a different size must show");

    fs::remove_file(scratch.path("b.png")).expect("delete b.png");
    assert_ne!(after_edit, sources_of(&paths), "a missing file must show");
}

#[test]
fn a_missing_file_is_not_an_empty_one() {
    let scratch = Scratch::new("missing", &[]);
    fs::write(scratch.path("empty.png"), b"").expect("write an empty file");
    let empty = SourceStamp::read(&scratch.path("empty.png"));
    let missing = SourceStamp::read(&scratch.path("absent.png"));
    assert_eq!(empty.size, Some(0));
    assert_eq!(missing.size, None);
    assert_ne!(
        fingerprint_sources(&[empty]),
        fingerprint_sources(&[SourceStamp {
            path: scratch.path("empty.png"),
            ..missing
        }])
    );
}

#[test]
fn an_unrelated_setting_leaves_the_alignment_fingerprint_alone() {
    let sources = sources_of(&[PathBuf::from("x/a.png")]);
    let base = StackingSettings::default();
    let reference = fingerprint_alignment(&sources, &base);

    let mut jpeg = base.clone();
    jpeg.image_saving.jpeg_quality = 40;
    let mut pyramid = base.clone();
    pyramid.pyramid_levels = 4;
    let mut cull = base.clone();
    cull.auto_cull = !base.auto_cull;
    let mut nth = base.clone();
    nth.stack_every_nth = 3;
    // The fusion settings, the cull and sort rules, the tile size and the resize
    // of the cropped result do not touch the aligned frames.
    let mut multigrid = base.clone();
    multigrid.relief_use_multigrid = !base.relief_use_multigrid;
    let mut strata = base.clone();
    strata.strata_base_radius = 20;
    strata.strata_detail_focus = 5;
    let mut sort = base.clone();
    sort.sort_by_sharpness = !base.sort_by_sharpness;
    let mut threshold = base.clone();
    threshold.auto_cull_threshold_pct = 3.0;
    let mut tile = base.clone();
    tile.tile_size = 512;
    let mut resize = base;
    resize.resize_cropped_to_original = !resize.resize_cropped_to_original;
    for unrelated in [
        jpeg, pyramid, cull, nth, multigrid, strata, sort, threshold, tile, resize,
    ] {
        assert_eq!(reference, fingerprint_alignment(&sources, &unrelated));
    }
}

#[test]
fn the_new_alignment_settings_move_the_alignment_fingerprint() {
    let sources = sources_of(&[PathBuf::from("x/a.png")]);
    let base = StackingSettings::default();
    let reference = fingerprint_alignment(&sources, &base);

    let mut optimizer = base.clone();
    optimizer.optimizer = if base.optimizer == OptimizerSetting::NelderMead {
        OptimizerSetting::LucasKanade
    } else {
        OptimizerSetting::NelderMead
    };
    let mut brightness = base.clone();
    brightness.correct_brightness = !base.correct_brightness;
    let mut chain = base.clone();
    chain.chain_alignment = !base.chain_alignment;
    let mut crop = base.clone();
    crop.crop_to_common_area = !base.crop_to_common_area;
    let mut refine = base;
    refine.neural_refine_classically = !refine.neural_refine_classically;
    for changed in [optimizer, brightness, chain, crop, refine] {
        assert_ne!(reference, fingerprint_alignment(&sources, &changed));
    }
}

#[test]
fn the_neural_model_name_counts_only_in_the_neural_mode() {
    let sources = sources_of(&[PathBuf::from("x/a.png")]);
    let classic = StackingSettings::default();
    assert_eq!(
        fingerprint_alignment_with(&sources, &classic, "a"),
        fingerprint_alignment_with(&sources, &classic, "b"),
        "another mode ignores the pick"
    );
    let neural = StackingSettings {
        alignment_mode: AlignmentModeSetting::Neural,
        ..classic
    };
    assert_ne!(
        fingerprint_alignment_with(&sources, &neural, "a"),
        fingerprint_alignment_with(&sources, &neural, "b")
    );
    assert_eq!(
        fingerprint_alignment_with(&sources, &neural, "a"),
        fingerprint_alignment_with(&sources, &neural, "a")
    );
}

#[test]
fn a_changed_model_pick_makes_the_aligned_cache_stale() {
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Neural,
        ..StackingSettings::default()
    };
    let scratch = Scratch::new("model", &["a.png", "b.png"]);
    let paths = scratch.paths(&["a.png", "b.png"]);
    let mut state = AppState::default();
    state.add_paths(paths.clone());
    state.align_model = "net-a".to_owned();
    let built = fingerprint_alignment_with(&sources_of(&paths), &settings, "net-a");
    state.set_aligned(
        AlignedCache {
            paths: paths.clone(),
            crop: None,
        },
        built,
    );
    let sources = sources_of(&paths);
    assert_eq!(state.aligned_staleness(&sources, &settings), None);
    state.align_model = "net-b".to_owned();
    assert_eq!(
        state.aligned_staleness(&sources, &settings),
        Some(StaleReason::Settings)
    );
}

#[test]
fn every_alignment_setting_moves_the_alignment_fingerprint() {
    let sources = sources_of(&[PathBuf::from("x/a.png")]);
    let base = StackingSettings::default();
    let reference = fingerprint_alignment(&sources, &base);

    let mut mode = base.clone();
    mode.alignment_mode = AlignmentModeSetting::Translation;
    let mut bounded = base.clone();
    bounded.bounded_search = !base.bounded_search;
    let mut seeding = base.clone();
    seeding.akaze_seeding = !base.akaze_seeding;
    let mut rotation = base.clone();
    rotation.preprocessing.pre_rotation = 90;
    let mut reverse = base;
    reverse.preprocessing.sort_reverse = true;
    for changed in [mode, bounded, seeding, rotation, reverse] {
        assert_ne!(reference, fingerprint_alignment(&sources, &changed));
    }
}

#[test]
fn the_stack_fingerprint_follows_the_settings_that_shape_the_image() {
    let sources = sources_of(&[PathBuf::from("x/a.png")]);
    let base = StackingSettings::default();
    let reference = fingerprint_stack(&sources, &base);

    let mut pyramid = base.clone();
    pyramid.pyramid_levels = 4;
    let mut radius = base.clone();
    radius.relief_estimation_radius = 9;
    let mut mode = base.clone();
    mode.alignment_mode = AlignmentModeSetting::None;
    for changed in [pyramid, radius, mode] {
        assert_ne!(reference, fingerprint_stack(&sources, &changed));
    }

    // The saving block only shapes the file Save writes, not the fused image.
    let mut jpeg = base.clone();
    jpeg.image_saving.jpeg_quality = 40;
    assert_eq!(reference, fingerprint_stack(&sources, &jpeg));
    // The full variant does follow it.
    assert_ne!(
        fingerprint_stack_full(&sources, &base),
        fingerprint_stack_full(&sources, &jpeg)
    );
}

#[test]
fn the_compute_device_does_not_make_a_result_stale() {
    // The CPU and the GPU agree within a tolerance, so moving Apex between them
    // leaves a finished result current.
    let sources = sources_of(&[PathBuf::from("x/a.png")]);
    let base = StackingSettings::default();
    let gpu = StackingSettings {
        compute_device: ComputeDevice::Gpu,
        ..base.clone()
    };
    assert_eq!(
        fingerprint_stack(&sources, &base),
        fingerprint_stack(&sources, &gpu)
    );
    // The full variant (used for the project's unsaved check) does see it.
    assert_ne!(
        fingerprint_stack_full(&sources, &base),
        fingerprint_stack_full(&sources, &gpu)
    );
}

#[test]
fn the_alignment_and_stack_fingerprints_are_distinct_kinds() {
    let sources = sources_of(&[PathBuf::from("x/a.png")]);
    let settings = StackingSettings::default();
    assert_ne!(
        fingerprint_alignment(&sources, &settings),
        fingerprint_stack(&sources, &settings)
    );
    assert_ne!(
        fingerprint_stack(&sources, &settings),
        fingerprint_stack_full(&sources, &settings)
    );
}

#[test]
fn the_reason_names_the_half_that_moved() {
    let settings = StackingSettings::default();
    let a = sources_of(&[PathBuf::from("x/a.png")]);
    let b = sources_of(&[PathBuf::from("x/b.png")]);
    let mut other = settings.clone();
    other.pyramid_levels = 3;

    let built = fingerprint_stack(&a, &settings);
    assert_eq!(
        stale_reason(&built, &fingerprint_stack(&a, &settings)),
        None
    );
    assert_eq!(
        stale_reason(&built, &fingerprint_stack(&b, &settings)),
        Some(StaleReason::Files)
    );
    assert_eq!(
        stale_reason(&built, &fingerprint_stack(&a, &other)),
        Some(StaleReason::Settings)
    );
    assert_eq!(
        stale_reason(&built, &fingerprint_stack(&b, &other)),
        Some(StaleReason::Both)
    );
}

#[test]
fn the_settings_summary_is_one_readable_line() {
    let settings = StackingSettings {
        stack_every_nth: 2,
        ..StackingSettings::default()
    };
    let text = summarize_settings(&settings);
    assert!(text.contains("Registration alignment"), "{text}");
    assert!(text.contains("auto-cull on"), "{text}");
    assert!(text.contains("every 2th frame"), "{text}");
    assert!(!text.contains('\n'));
}

// ── The aligned cache ────────────────────────────────────────────────────────

#[test]
fn a_state_without_a_cache_is_never_current() {
    let state = AppState::default();
    assert!(!state.aligned_is_current(&StackingSettings::default()));
}

#[test]
fn the_cache_is_current_until_the_list_or_an_alignment_setting_changes() {
    let scratch = Scratch::new("cache", &["a.png", "b.png", "c.png"]);
    let settings = StackingSettings::default();
    let state = aligned_state(scratch.paths(&["a.png", "b.png"]), &settings);
    assert!(state.aligned_is_current(&settings));

    // A file added.
    let mut added = aligned_state(scratch.paths(&["a.png", "b.png"]), &settings);
    added.add_paths([scratch.path("c.png")]);
    assert!(!added.aligned_is_current(&settings));

    // A file removed.
    let mut removed = aligned_state(scratch.paths(&["a.png", "b.png"]), &settings);
    removed.remove_path(0);
    assert!(!removed.aligned_is_current(&settings));

    // The list reordered.
    let mut reordered = aligned_state(scratch.paths(&["a.png", "b.png"]), &settings);
    reordered.file_paths.reverse();
    assert!(!reordered.aligned_is_current(&settings));

    // An alignment setting changed.
    let mut other = settings.clone();
    other.alignment_mode = AlignmentModeSetting::Translation;
    assert!(!state.aligned_is_current(&other));

    // An unrelated setting changed.
    let mut unrelated = settings.clone();
    unrelated.image_saving.jpeg_quality = 30;
    unrelated.pyramid_levels = 5;
    unrelated.auto_cull = !settings.auto_cull;
    assert!(state.aligned_is_current(&unrelated));
}

#[test]
fn a_modified_source_file_makes_the_cache_stale() {
    let scratch = Scratch::new("edit", &["a.png", "b.png"]);
    let settings = StackingSettings::default();
    let state = aligned_state(scratch.paths(&["a.png", "b.png"]), &settings);
    assert!(state.aligned_is_current(&settings));
    fs::write(
        scratch.path("a.png"),
        b"changed on disk, longer than before",
    )
    .expect("rewrite");
    assert!(!state.aligned_is_current(&settings));
}

#[test]
fn a_cache_without_a_fingerprint_counts_as_stale() {
    let mut state = aligned_state(vec![PathBuf::from("x/a.png")], &StackingSettings::default());
    state.aligned_fp = None;
    assert!(!state.aligned_is_current(&StackingSettings::default()));
}

#[test]
fn only_a_current_cache_allows_reuse() {
    let scratch = Scratch::new("reuse", &["a.png", "b.png", "c.png"]);
    let settings = StackingSettings::default();
    let state = aligned_state(scratch.paths(&["a.png", "b.png", "c.png"]), &settings);
    let sources = sources_of(&state.file_paths);

    // The wanted order wins over the cache's, and only the wanted frames come out.
    let wanted = scratch.paths(&["c.png", "a.png"]);
    let reused = state
        .reusable_frames(&sources, &settings, &wanted)
        .expect("the cache is current");
    assert_eq!(reused.paths, wanted);
    assert_eq!(reused.crop, Some((0, 0, 1, 1)));

    // A file the cache never held: the run aligns afresh.
    let unknown = scratch.paths(&["a.png", "z.png"]);
    assert!(
        state
            .reusable_frames(&sources, &settings, &unknown)
            .is_none()
    );
    assert!(state.reusable_frames(&sources, &settings, &[]).is_none());

    // The files moved on: nothing is reused, and no crop leaks out.
    let mut moved = state;
    moved.remove_path(2);
    let moved_sources = sources_of(&moved.file_paths);
    assert!(
        moved
            .reusable_frames(&moved_sources, &settings, &wanted)
            .is_none()
    );

    // An alignment setting changed.
    let mut other = settings.clone();
    other.bounded_search = true;
    let fresh = aligned_state(scratch.paths(&["a.png", "b.png", "c.png"]), &settings);
    assert!(
        fresh
            .reusable_frames(&sources, &other, &scratch.paths(&["a.png"]))
            .is_none()
    );
}

// ── Results ──────────────────────────────────────────────────────────────────

fn stamped(state: &mut AppState, name: &str, settings: &StackingSettings) {
    let fingerprint = fingerprint_stack(&sources_of(&state.file_paths), settings);
    state.push_result_with(
        PathBuf::from(name),
        ResultMeta {
            fingerprint: Some(fingerprint),
            label: "Apex".to_owned(),
            settings_summary: summarize_settings(settings),
        },
    );
}

#[test]
fn a_result_is_current_until_its_inputs_change() {
    let scratch = Scratch::new("result", &["a.png", "b.png"]);
    let settings = StackingSettings::default();
    let mut state = AppState::default();
    state.add_paths(scratch.paths(&["a.png", "b.png"]));
    stamped(&mut state, "r1.png", &settings);
    assert!(state.result_is_current(0, &settings));
    assert!(!state.result_is_current(1, &settings), "no such result");

    // The export block is not an input of the fused image.
    let mut jpeg = settings.clone();
    jpeg.image_saving.jpeg_quality = 10;
    assert!(state.result_is_current(0, &jpeg));

    // A stacking setting is.
    let mut pyramid = settings.clone();
    pyramid.pyramid_levels = 3;
    assert!(!state.result_is_current(0, &pyramid));

    // So is the file list.
    state.remove_path(1);
    assert!(!state.result_is_current(0, &settings));
}

#[test]
fn a_result_with_no_record_is_never_flagged() {
    let mut state = AppState::default();
    state.push_result(PathBuf::from("legacy.png"));
    assert_eq!(state.result_meta.len(), 1);
    assert!(state.result_is_current(0, &StackingSettings::default()));
}

#[test]
fn the_result_records_stay_parallel_to_the_paths() {
    let settings = StackingSettings::default();
    let mut state = AppState::default();
    state.push_result(PathBuf::from("r0.png"));
    stamped(&mut state, "r1.png", &settings);
    state.result_paths.push(PathBuf::from("bare.png"));
    stamped(&mut state, "r3.png", &settings);
    assert_eq!(state.result_paths.len(), 4);
    assert_eq!(state.result_meta.len(), 4);
    assert_eq!(state.result_meta[3].label, "Apex");
    assert!(state.result_meta[2].fingerprint.is_none());
}

// ── Auto-cull marks ──────────────────────────────────────────────────────────

#[test]
fn auto_cull_marks_files_without_touching_the_list() {
    let mut state = AppState::default();
    let list = vec![
        PathBuf::from("a.png"),
        PathBuf::from("b.png"),
        PathBuf::from("c.png"),
    ];
    state.add_paths(list.clone());
    state.select_source(1);

    state.set_culled(vec![PathBuf::from("b.png")]);
    assert_eq!(state.file_paths, list, "the user's list stays as it is");
    assert_eq!(state.view_idx, Some(1));
    assert_eq!(state.culled_flags(), vec![false, true, false]);

    // A later run without culling clears the marks.
    state.set_culled(Vec::new());
    assert_eq!(state.culled_flags(), vec![false, false, false]);
}

#[test]
fn the_marks_follow_the_files_not_the_positions() {
    let mut state = AppState::default();
    state.add_paths(["a.png", "b.png", "c.png"].map(PathBuf::from));
    state.set_culled(vec![PathBuf::from("c.png")]);
    state.remove_path(0);
    assert_eq!(state.culled_flags(), vec![false, true]);
}

#[test]
fn a_new_or_empty_list_drops_the_marks() {
    let mut state = AppState::default();
    state.add_paths(["a.png", "b.png"].map(PathBuf::from));
    state.set_culled(vec![PathBuf::from("a.png")]);
    state.replace_paths(vec![PathBuf::from("a.png")]);
    assert_eq!(state.culled_flags(), vec![false]);

    state.set_culled(vec![PathBuf::from("a.png")]);
    state.clear_paths();
    assert_eq!(state.culled, Vec::<PathBuf>::new());
}

// ── The sidebar view ─────────────────────────────────────────────────────────

#[test]
fn the_view_reports_the_cache_the_newest_result_and_the_marks() {
    let scratch = Scratch::new("view", &["a.png", "b.png", "c.png"]);
    let settings = StackingSettings::default();
    let mut state = aligned_state(scratch.paths(&["a.png", "b.png"]), &settings);
    stamped(&mut state, "r1.png", &settings);
    state.set_culled(vec![scratch.path("b.png")]);

    let view = stale_view(&state, &settings);
    assert_eq!(view.aligned_summary, "Aligned frames (2)");
    assert_eq!(view.aligned_stale, None);
    assert_eq!(view.result_stale, None);
    assert_eq!(view.culled, vec![false, true]);

    // A file arrives: both the cache and the result are out of date for the files.
    state.add_paths([scratch.path("c.png")]);
    let view = stale_view(&state, &settings);
    assert_eq!(view.aligned_stale, Some(StaleReason::Files));
    assert_eq!(view.result_stale, Some(StaleReason::Files));

    // A stacking-only setting leaves the cache alone and flags the result.
    let mut pyramid = settings.clone();
    pyramid.pyramid_levels = 3;
    state.remove_path(2);
    let view = stale_view(&state, &pyramid);
    assert_eq!(view.aligned_stale, None);
    assert_eq!(view.result_stale, Some(StaleReason::Settings));
}

#[test]
fn an_empty_state_has_an_empty_view() {
    let view = stale_view(&AppState::default(), &StackingSettings::default());
    assert_eq!(view.aligned_summary, "");
    assert_eq!(view.aligned_stale, None);
    assert_eq!(view.result_stale, None);
    assert_eq!(view.culled, Vec::<bool>::new());
}

#[test]
fn only_the_newest_result_is_judged() {
    let scratch = Scratch::new("newest", &["a.png"]);
    let settings = StackingSettings::default();
    let mut other = settings.clone();
    other.pyramid_levels = 3;
    let mut state = AppState::default();
    state.add_paths(scratch.paths(&["a.png"]));
    stamped(&mut state, "old.png", &other);
    stamped(&mut state, "new.png", &settings);
    assert_eq!(stale_view(&state, &settings).result_stale, None);
    assert!(!state.result_is_current(0, &settings));
}
