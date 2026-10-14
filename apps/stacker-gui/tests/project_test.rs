//! The project file, the recent-projects list and the dirty rule, without a
//! window: round trips, path handling, tolerance, stroke replay, recents
//! ordering, the cap and the missing-file drop, and the fingerprint rule.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use stacker_algo::hybrid::retouch::RetouchSession;
use stacker_core::image::PlanarImage;
use stacker_gui::{
    project::{
        Baseline, CURRENT_VERSION, FILE_SUFFIX, Project, ProjectError, StrokeRecord, project_name,
        suggested_location, with_project_suffix,
    },
    retouch::{RetouchState, STROKE_BREAK, SourceChoice, Stroke},
    settings::{
        AlignmentModeSetting, StackingSettings,
        recent::{self as recent_projects, MAX_RECENT, RecentProjects},
    },
};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A fresh, empty directory under the system temp dir.
fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "z-stackr-project-test-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn touch(path: &Path) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, b"x").unwrap();
}

fn tweaked_settings() -> StackingSettings {
    StackingSettings {
        pyramid_levels: 7,
        auto_cull: true,
        ..StackingSettings::default()
    }
}

fn stroke_record(source_index: Option<usize>) -> StrokeRecord {
    StrokeRecord {
        source_index,
        points: vec![(5, 5), (20, 5), STROKE_BREAK, (30, 20)],
        radius: 2.5,
        opacity: 0.6,
    }
}

/// A project below `base` with one source outside it, an output name and two strokes.
fn sample_project(base: &Path) -> Project {
    let mut project = Project::new(
        vec![
            base.join("IMG_0001.jpg"),
            base.join("sub").join("IMG_0002.jpg"),
            std::env::temp_dir()
                .join("z-stackr-elsewhere")
                .join("IMG_0003.jpg"),
        ],
        tweaked_settings(),
    );
    project.output_name = Some("blossom".to_owned());
    project.strokes = vec![stroke_record(Some(1)), stroke_record(None)];
    project
}

#[test]
fn a_project_from_an_earlier_version_loads_with_the_current_meanings() {
    // No `settings_version`: the alignment mode names are the earlier ones.
    let affine = "version = 1\nsources = []\n[settings]\nalignment_mode = \"affine\"\n";
    let loaded = Project::from_toml(affine, None).expect("an old project loads");
    assert_eq!(
        loaded.project.settings.alignment_mode,
        AlignmentModeSetting::Registration,
        "the old `affine` is the current `registration`"
    );
    let bounded = "version = 1\nsources = []\n[settings]\nalignment_mode = \"bounded\"\n";
    let loaded = Project::from_toml(bounded, None).expect("an old project loads");
    assert_eq!(
        loaded.project.settings.alignment_mode,
        AlignmentModeSetting::Registration
    );
    assert!(loaded.project.settings.bounded_search);
}

#[test]
fn a_saved_project_keeps_its_current_alignment_mode() {
    let settings = StackingSettings {
        alignment_mode: AlignmentModeSetting::Affine,
        ..StackingSettings::default()
    };
    let project = Project::new(Vec::new(), settings);
    let text = project.to_toml(None).expect("encodes");
    let back = Project::from_toml(&text, None).expect("loads");
    assert_eq!(
        back.project.settings.alignment_mode,
        AlignmentModeSetting::Affine,
        "a file written now is not migrated again"
    );
}

// ── The file ─────────────────────────────────────────────────────────────────

#[test]
fn a_project_survives_save_and_load() {
    let dir = temp_dir("roundtrip");
    let file = dir.join("blossom.zstack.toml");
    let project = sample_project(&dir);

    project.save(&file).unwrap();
    let loaded = Project::load(&file).unwrap();

    assert_eq!(loaded.project, project);
    assert_eq!(loaded.project.version, CURRENT_VERSION);
    assert_eq!(loaded.project.fingerprint(), project.fingerprint());
}

#[test]
fn sources_below_the_file_are_stored_relative_and_others_absolute() {
    let dir = temp_dir("paths");
    let project = sample_project(&dir);

    let text = project.to_toml(Some(&dir)).unwrap();

    assert!(text.contains("\"IMG_0001.jpg\""), "{text}");
    assert!(text.contains("\"sub/IMG_0002.jpg\""), "{text}");
    let folder_name = dir.file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        !text.contains(&folder_name),
        "nothing below the file's folder is stored with its full path:\n{text}"
    );
    assert!(text.contains("IMG_0003.jpg"), "{text}");

    // Moving the folder moves the sources with it; the outside one stays put.
    let moved = temp_dir("paths-moved");
    let loaded = Project::from_toml(&text, Some(&moved)).unwrap().project;
    assert_eq!(loaded.sources[0], moved.join("IMG_0001.jpg"));
    assert_eq!(loaded.sources[1], moved.join("sub").join("IMG_0002.jpg"));
    assert_eq!(loaded.sources[2], project.sources[2]);
}

#[test]
fn without_a_base_folder_paths_are_stored_as_they_are() {
    let dir = temp_dir("nobase");
    let project = sample_project(&dir);

    let text = project.to_toml(None).unwrap();
    let loaded = Project::from_toml(&text, None).unwrap().project;

    assert_eq!(loaded.sources, project.sources);
}

#[test]
fn unknown_keys_and_newer_versions_are_tolerated() {
    let dir = temp_dir("lenient");
    let text = r#"
        version = 99
        future_key = "later"
        sources = ["a.jpg", "b.jpg"]

        [extra]
        anything = 1

        [settings]
        pyramid_levels = 7
        no_such_setting = true

        [[strokes]]
        radius = 3.0
        points = [1, 2, 3, 4]
        future_stroke_key = 5
    "#;

    let loaded = Project::from_toml(text, Some(&dir)).unwrap();

    assert_eq!(
        loaded.project.sources,
        vec![dir.join("a.jpg"), dir.join("b.jpg")]
    );
    assert_eq!(loaded.project.settings.pyramid_levels, 7);
    assert_eq!(loaded.project.strokes.len(), 1);
    assert_eq!(loaded.project.strokes[0].points, vec![(1, 2), (3, 4)]);
    assert!(
        loaded.warnings.iter().any(|w| w.contains("99")),
        "a newer version is noted: {:?}",
        loaded.warnings
    );
}

#[test]
fn a_malformed_stroke_is_skipped_and_reported() {
    let text = r"
        sources = []

        [[strokes]]
        radius = 3.0
        points = [1, 2, 3]

        [[strokes]]
        radius = 3.0
        points = [1, 2]
    ";

    let loaded = Project::from_toml(text, None).unwrap();

    assert_eq!(loaded.project.strokes.len(), 1);
    assert_eq!(loaded.warnings.len(), 1, "{:?}", loaded.warnings);
}

#[test]
fn text_that_is_not_a_project_is_an_error() {
    assert!(matches!(
        Project::from_toml("this is = = not toml", None),
        Err(ProjectError::Parse(_))
    ));
    assert!(matches!(
        Project::from_toml("sources = 3", None),
        Err(ProjectError::Parse(_))
    ));
    let dir = temp_dir("missing");
    assert!(matches!(
        Project::load(&dir.join("nope.zstack.toml")),
        Err(ProjectError::Io(_))
    ));
}

#[test]
fn file_names_follow_the_suffix_rule() {
    assert_eq!(project_name(Path::new("a/blossom.zstack.toml")), "blossom");
    assert_eq!(project_name(Path::new("a/Blossom.ZSTACK.TOML")), "Blossom");
    assert_eq!(project_name(Path::new("a/other.toml")), "other");
    assert_eq!(
        with_project_suffix(PathBuf::from("a/blossom")),
        PathBuf::from("a/blossom.zstack.toml")
    );
    assert_eq!(
        with_project_suffix(PathBuf::from("a/blossom.toml")),
        PathBuf::from("a/blossom.zstack.toml")
    );
    assert_eq!(
        with_project_suffix(PathBuf::from("a/blossom.zstack.toml")),
        PathBuf::from("a/blossom.zstack.toml")
    );

    let sources = [PathBuf::from("shots").join("rose").join("IMG_1.jpg")];
    let (folder, name) = suggested_location(&sources).unwrap();
    assert_eq!(folder, PathBuf::from("shots").join("rose"));
    assert_eq!(name, format!("rose{FILE_SUFFIX}"));
    assert!(suggested_location(&[]).is_none());
}

// ── Strokes ──────────────────────────────────────────────────────────────────

const W: usize = 40;
const H: usize = 30;

fn solid(level: f32) -> PlanarImage<f32> {
    let mut image = PlanarImage::new(W, H);
    image.luma.fill(level);
    image
}

/// A retouch state with a session over a dark result and two listed frames.
fn session() -> RetouchState {
    let base = Arc::new(solid(0.1));
    let mut state = RetouchState::new();
    state.start_session(
        RetouchSession::new(Arc::clone(&base), base),
        PathBuf::from("stacker_result_Apex_1.png"),
    );
    state.refresh_sources(None, &[PathBuf::from("a.png"), PathBuf::from("b.png")]);
    state
}

/// One drag through `points` (`None`: the pointer left the image), drained after
/// every point the way the 16 ms timer does.
fn drag(
    state: &mut RetouchState,
    points: &[Option<(u32, u32)>],
    radius: f32,
    opacity: f32,
) -> bool {
    assert!(state.begin_stroke());
    for &point in points {
        state.queue_point(point, radius, opacity);
        state.drain_pending();
    }
    state.end_stroke()
}

fn alpha(state: &RetouchState) -> &[f32] {
    &state.session.as_ref().expect("a session").alpha.luma
}

/// Two drags, the first one leaving the image once and painting from frame 2.
fn live_two_strokes() -> RetouchState {
    let mut state = session();
    let SourceChoice::Load(request) = state.choose_source(Some(1)) else {
        panic!("choosing a source asks for its frame");
    };
    assert_eq!(state.finish_source(&request, solid(1.0)), Ok(true));
    assert!(drag(
        &mut state,
        &[
            Some((5, 5)),
            Some((20, 5)),
            None,
            Some((30, 20)),
            Some((35, 25)),
        ],
        2.0,
        0.6,
    ));
    assert!(drag(
        &mut state,
        &[Some((10, 15)), Some((10, 25))],
        3.0,
        1.0
    ));
    state
}

#[test]
fn drags_are_logged_with_their_points_and_brush() {
    let state = live_two_strokes();
    let strokes = state.strokes();

    assert_eq!(strokes.len(), 2);
    assert_eq!(
        strokes[0].points,
        vec![(5, 5), (20, 5), STROKE_BREAK, (30, 20), (35, 25)]
    );
    assert!((strokes[0].radius - 2.0).abs() < f32::EPSILON);
    assert!((strokes[0].opacity - 0.6).abs() < f32::EPSILON);
    assert!(strokes[0].source.is_some(), "the chosen frame is recorded");
    assert_eq!(strokes[1].points, vec![(10, 15), (10, 25)]);
}

#[test]
fn replaying_the_log_gives_the_same_alpha_as_the_live_history() {
    let live = live_two_strokes();
    let strokes: Vec<Stroke> = live.strokes().to_vec();

    let mut fresh = session();
    let applied = fresh.replay(&strokes);

    assert_eq!(applied, 2);
    assert_eq!(alpha(&fresh), alpha(&live), "the mask is identical");
    assert_eq!(fresh.strokes(), strokes.as_slice());
    assert_eq!(fresh.history.len(), 2, "one undo step per replayed stroke");
}

#[test]
fn the_log_follows_undo_and_redo() {
    let mut state = live_two_strokes();
    let both: Vec<Stroke> = state.strokes().to_vec();

    assert!(state.undo());
    assert_eq!(state.strokes(), &both[..1]);
    assert!(state.undo());
    assert_eq!(state.strokes(), []);
    assert!(state.redo());
    assert!(state.redo());
    assert_eq!(state.strokes(), both.as_slice());

    // A new drag after an undo ends the redo branch in the log too.
    assert!(state.undo());
    assert!(drag(&mut state, &[Some((2, 2))], 2.0, 1.0));
    assert!(!state.redo());
    assert_eq!(state.strokes().len(), 2);
    assert_eq!(state.strokes()[0], both[0]);
}

#[test]
fn a_new_session_starts_with_an_empty_log() {
    let mut state = live_two_strokes();
    let base = Arc::new(solid(0.2));
    state.start_session(
        RetouchSession::new(Arc::clone(&base), base),
        PathBuf::from("stacker_result_Apex_2.png"),
    );
    assert_eq!(state.strokes(), []);
}

#[test]
fn stroke_records_name_the_source_by_position() {
    let state = live_two_strokes();
    let stroke = state.strokes()[0].clone();
    let painted_from = stroke.source.clone().expect("a source was chosen");
    let sources = vec![PathBuf::from("zz.png"), painted_from];

    let record = StrokeRecord::from_stroke(&stroke, &sources);

    assert_eq!(record.source_index, Some(1));
    assert_eq!(record.to_stroke(&sources), stroke);
    // A record whose frame is gone replays without one.
    assert_eq!(record.to_stroke(&[]).source, None);
}

#[test]
fn replaying_a_stored_project_reproduces_the_mask() {
    let live = live_two_strokes();
    let sources = vec![
        PathBuf::from("zz.png"),
        live.strokes()[0].source.clone().unwrap(),
    ];
    let mut project = Project::new(sources.clone(), StackingSettings::default());
    project.strokes = live
        .strokes()
        .iter()
        .map(|stroke| StrokeRecord::from_stroke(stroke, &sources))
        .collect();

    let dir = temp_dir("stroke-file");
    let file = dir.join("p.zstack.toml");
    project.save(&file).unwrap();
    let loaded = Project::load(&file).unwrap().project;

    let strokes: Vec<Stroke> = loaded
        .strokes
        .iter()
        .map(|record| record.to_stroke(&loaded.sources))
        .collect();
    let mut fresh = session();
    assert_eq!(fresh.replay(&strokes), 2);
    assert_eq!(alpha(&fresh), alpha(&live));
}

// ── Recents ──────────────────────────────────────────────────────────────────

fn paths(list: &RecentProjects) -> Vec<PathBuf> {
    list.entries().to_vec()
}

#[test]
fn recents_are_newest_first_and_each_project_appears_once() {
    let mut list = RecentProjects::new();
    list.record(Path::new("a.zstack.toml"));
    list.record(Path::new("b.zstack.toml"));
    list.record(Path::new("c.zstack.toml"));
    assert_eq!(
        paths(&list),
        ["c", "b", "a"].map(|n| PathBuf::from(format!("{n}.zstack.toml")))
    );

    list.record(Path::new("a.zstack.toml"));
    assert_eq!(
        paths(&list),
        ["a", "c", "b"].map(|n| PathBuf::from(format!("{n}.zstack.toml")))
    );

    list.forget(Path::new("c.zstack.toml"));
    assert_eq!(list.entries().len(), 2);
    list.clear();
    assert!(list.is_empty());
}

#[test]
fn recents_are_capped_and_drop_the_oldest() {
    let mut list = RecentProjects::new();
    for i in 0..(MAX_RECENT + 3) {
        list.record(Path::new(&format!("p{i}.zstack.toml")));
    }
    assert_eq!(list.entries().len(), MAX_RECENT);
    assert_eq!(
        list.entries()[0],
        PathBuf::from(format!("p{}.zstack.toml", MAX_RECENT + 2))
    );
    assert_eq!(
        list.entries()[MAX_RECENT - 1],
        PathBuf::from("p3.zstack.toml"),
        "p0 to p2 fell off"
    );
}

#[test]
fn recents_load_without_the_files_that_are_gone() {
    let dir = temp_dir("recent");
    let kept_old = dir.join("old.zstack.toml");
    let kept_new = dir.join("new.zstack.toml");
    let gone = dir.join("gone.zstack.toml");
    touch(&kept_old);
    touch(&kept_new);

    let mut list = RecentProjects::new();
    list.record(&kept_old);
    list.record(&gone);
    list.record(&kept_new);
    let file = recent_projects::path_beside(&dir.join("settings.toml"));
    assert_eq!(file, dir.join("recent.toml"));
    recent_projects::save_to(&file, &list).unwrap();

    let loaded = recent_projects::load_from(&file);

    assert_eq!(paths(&loaded), vec![kept_new, kept_old]);
}

#[test]
fn a_missing_or_garbled_recents_file_is_an_empty_list() {
    let dir = temp_dir("recent-bad");
    assert!(recent_projects::load_from(&dir.join("recent.toml")).is_empty());

    let garbled = dir.join("garbled.toml");
    fs::write(&garbled, "projects = [[[").unwrap();
    assert!(recent_projects::load_from(&garbled).is_empty());
    assert!(RecentProjects::from_toml("projects = 5").is_empty());
}

#[test]
fn the_recents_text_round_trips_and_labels_name_folder_and_project() {
    let mut list = RecentProjects::new();
    list.record(Path::new("one.zstack.toml"));
    list.record(&PathBuf::from("shots").join("rose.zstack.toml"));
    assert_eq!(RecentProjects::from_toml(&list.to_toml()), list);

    let label = recent_projects::display_label(&PathBuf::from("shots").join("rose.zstack.toml"));
    assert!(label.starts_with("rose \u{2014} "), "{label}");
    assert!(label.ends_with("shots"), "{label}");
    assert_eq!(
        recent_projects::display_label(Path::new("one.zstack.toml")),
        "one"
    );
}

// ── The dirty rule ───────────────────────────────────────────────────────────

#[test]
fn saving_the_same_content_twice_leaves_it_clean() {
    let dir = temp_dir("dirty-twice");
    let file = dir.join("p.zstack.toml");
    let project = sample_project(&dir);

    project.save(&file).unwrap();
    let baseline = Baseline::of(&project);
    project.save(&file).unwrap();

    assert!(!baseline.is_dirty(&project));
    // What was read back is the same content too.
    assert!(!baseline.is_dirty(&Project::load(&file).unwrap().project));
}

#[test]
fn a_reordered_source_list_is_dirty_and_putting_it_back_is_clean() {
    let dir = temp_dir("dirty-order");
    let project = sample_project(&dir);
    let baseline = Baseline::of(&project);

    let mut reordered = project;
    reordered.sources.swap(0, 1);
    assert!(baseline.is_dirty(&reordered));

    reordered.sources.swap(0, 1);
    assert!(
        !baseline.is_dirty(&reordered),
        "undoing the change cleans it"
    );
}

#[test]
fn settings_output_name_and_strokes_all_count() {
    let dir = temp_dir("dirty-fields");
    let project = sample_project(&dir);
    let baseline = Baseline::of(&project);

    let mut settings = project.clone();
    settings.settings.pyramid_levels += 1;
    assert!(baseline.is_dirty(&settings));

    let mut name = project.clone();
    name.output_name = None;
    assert!(baseline.is_dirty(&name));

    let mut stroke = project.clone();
    stroke.strokes.pop();
    assert!(baseline.is_dirty(&stroke));

    let mut point = project.clone();
    point.strokes[0].points.push((1, 1));
    assert!(baseline.is_dirty(&point));

    let mut added = project;
    added.sources.push(PathBuf::from("extra.jpg"));
    assert!(baseline.is_dirty(&added));
}

#[test]
fn the_fingerprint_does_not_depend_on_where_the_file_lives() {
    let dir = temp_dir("dirty-place");
    let project = sample_project(&dir);

    // Saved in two folders, read back with each folder as base: the sources that
    // lie below the file move with it, so only the content they name matters.
    let text = project.to_toml(Some(&dir)).unwrap();
    let again = Project::from_toml(&text, Some(&dir)).unwrap().project;

    assert_eq!(again.fingerprint(), project.fingerprint());
}
