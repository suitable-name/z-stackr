use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
};

use stacker_gui::{
    state::{AlignedCache, AppState},
    sync::lock,
};

fn path(name: &str) -> PathBuf {
    PathBuf::from(name)
}

#[test]
fn default_state_is_empty() {
    let state = AppState::default();
    assert_eq!(state.file_paths, Vec::<PathBuf>::new());
    assert_eq!(state.result_paths, Vec::<PathBuf>::new());
    assert!(state.viewed_path.is_none());
    assert!(state.view_idx.is_none());
    assert!(state.aligned.is_none());
    assert!(state.retouch.session.is_none());
    assert!(state.retouch.result_path.is_none());
    assert!(!state.retouch.history.can_undo());
    assert!(!state.retouch.history.can_redo());
}

#[test]
fn lock_recovers_from_a_poisoned_mutex() {
    let shared = Arc::new(Mutex::new(vec![1, 2, 3]));
    let worker = Arc::clone(&shared);
    let joined = thread::spawn(move || {
        let _guard = worker.lock().unwrap();
        panic!("a worker panicked while holding the lock");
    })
    .join();
    assert!(joined.is_err());
    assert!(shared.is_poisoned());

    let mut guard = lock(&shared);
    guard.push(4);
    assert_eq!(*guard, vec![1, 2, 3, 4]);
}

#[test]
fn add_paths_skips_duplicates_and_counts_new_files() {
    let mut state = AppState::default();
    assert_eq!(state.add_paths([path("a.png"), path("b.png")]), 2);
    assert_eq!(state.add_paths([path("b.png"), path("c.png")]), 1);
    assert_eq!(
        state.file_paths,
        vec![path("a.png"), path("b.png"), path("c.png")]
    );
}

#[test]
fn removing_a_source_keeps_the_viewed_index_on_the_same_file() {
    let mut state = AppState::default();
    state.add_paths([path("a.png"), path("b.png"), path("c.png")]);
    assert_eq!(state.select_source(2), Some(path("c.png")));

    assert_eq!(state.remove_path(0), Some(path("a.png")));
    assert_eq!(state.view_idx, Some(1));
    assert_eq!(state.file_paths[1], path("c.png"));

    assert_eq!(state.remove_path(1), Some(path("c.png")));
    assert_eq!(state.view_idx, None);
    assert_eq!(state.remove_path(7), None);
}

#[test]
fn selecting_out_of_range_records_the_index_but_not_a_path() {
    let mut state = AppState::default();
    state.add_paths([path("a.png")]);
    assert_eq!(state.select_source(5), None);
    assert_eq!(state.view_idx, Some(5));
    assert!(state.viewed_path.is_none());
    assert_eq!(state.select_result(0), None);
}

#[test]
fn a_pushed_result_becomes_the_viewed_file() {
    let mut state = AppState::default();
    state.push_result(path("r1.png"));
    state.push_result(path("r2.png"));
    assert_eq!(state.viewed_path, Some(path("r2.png")));
    assert_eq!(state.select_result(0), Some(path("r1.png")));
    assert_eq!(state.viewed_path, Some(path("r1.png")));
}

#[test]
fn the_aligned_record_covers_files_by_path_in_the_wanted_order() {
    // The cache was built from a reversed, filtered list: positions differ from
    // the sidebar's, paths do not.
    let cache = AlignedCache {
        paths: vec![path("c.png"), path("a.png")],
        crop: Some((1, 1, 3, 2)),
    };
    let reused = cache.covering(&[path("a.png"), path("c.png")]).unwrap();
    assert_eq!(reused.paths, vec![path("a.png"), path("c.png")]);
    assert_eq!(reused.crop, Some((1, 1, 3, 2)));
    assert!(cache.covering(&[path("a.png"), path("b.png")]).is_none());
    assert!(cache.covering(&[]).is_none());
}
