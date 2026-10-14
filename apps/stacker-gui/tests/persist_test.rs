use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
    thread,
    time::{Duration, Instant},
};

use stacker_gui::settings::{
    LoadOutcome, SettingsPersister, StackingSettings, open_at,
    store::{self, load_from},
};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A fresh, empty directory under the system temp dir.
fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "z-stackr-gui-test-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the writer"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn tmp_sibling(path: &std::path::Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

#[test]
fn a_burst_of_updates_is_written_once_with_the_final_value() {
    let dir = temp_dir("burst");
    let path = dir.join("settings.toml");
    let persister = SettingsPersister::spawn_with_debounce(
        path.clone(),
        StackingSettings::default(),
        Duration::from_millis(300),
    );

    for i in 0..20_u32 {
        persister.update(|s| s.pyramid_levels = 2 + (i % 10));
        thread::sleep(Duration::from_millis(5));
    }
    // The in-memory copy is current at once, before any write.
    assert_eq!(persister.snapshot().pyramid_levels, 11);

    wait_until(|| persister.write_count() >= 1);
    thread::sleep(Duration::from_millis(700));
    assert_eq!(
        persister.write_count(),
        1,
        "the burst collapsed into one write"
    );

    let (saved, _) = StackingSettings::from_toml_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(saved.pyramid_levels, 11);
    assert!(
        !tmp_sibling(&path).exists(),
        "the .tmp file was renamed away"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn flush_writes_a_pending_change_without_waiting_for_the_debounce() {
    let dir = temp_dir("flush");
    let path = dir.join("settings.toml");
    let persister = SettingsPersister::spawn_with_debounce(
        path.clone(),
        StackingSettings::default(),
        Duration::from_secs(60),
    );

    persister.flush();
    assert_eq!(
        persister.write_count(),
        0,
        "nothing pending, nothing written"
    );
    assert!(!path.exists());

    persister.update(|s| s.pyramid_levels = 7);
    persister.flush();
    assert_eq!(persister.write_count(), 1);
    let (saved, _) = StackingSettings::from_toml_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(saved.pyramid_levels, 7);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupt_file_is_renamed_aside_and_defaults_are_used() {
    let dir = temp_dir("corrupt");
    let path = dir.join("settings.toml");
    let garbage = "this is = = not [valid toml";
    fs::write(&path, garbage).unwrap();

    let loaded = load_from(&path);
    assert_eq!(loaded.settings, StackingSettings::default());
    assert!(loaded.outcome.allows_writes());
    assert!(loaded.outcome.user_message().is_some());

    let LoadOutcome::Corrupt {
        renamed_to: Some(moved),
        ..
    } = loaded.outcome
    else {
        panic!("expected a corrupt-file outcome with a rename");
    };
    assert!(
        !path.exists(),
        "the broken file is no longer at the settings path"
    );
    assert!(
        moved
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("settings.toml.corrupt-")
    );
    assert_eq!(fs::read_to_string(&moved).unwrap(), garbage);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn applying_after_a_corrupt_load_keeps_the_broken_file() {
    let dir = temp_dir("corrupt-apply");
    let path = dir.join("settings.toml");
    let garbage = "pyramid_levels = [[[";
    fs::write(&path, garbage).unwrap();

    let opened = open_at(Some(&path));
    let LoadOutcome::Corrupt {
        renamed_to: Some(moved),
        ..
    } = &opened.outcome
    else {
        panic!("expected a corrupt-file outcome");
    };
    assert!(opened.persister.writes_enabled());

    opened.persister.update(|s| s.pyramid_levels = 9);
    opened.persister.flush();

    let (saved, _) = StackingSettings::from_toml_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(
        saved.pyramid_levels, 9,
        "a fresh file holds the new settings"
    );
    assert_eq!(
        fs::read_to_string(moved).unwrap(),
        garbage,
        "the broken copy is untouched"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_file_that_could_not_be_moved_is_never_overwritten() {
    let not_movable = LoadOutcome::Corrupt {
        reason: "bad".to_owned(),
        renamed_to: None,
    };
    assert!(!not_movable.allows_writes());
    assert!(not_movable.user_message().is_some());
    assert!(
        !LoadOutcome::Unreadable {
            reason: "denied".to_owned()
        }
        .allows_writes()
    );

    let dir = temp_dir("disabled");
    let path = dir.join("settings.toml");
    let persister = SettingsPersister::disabled(StackingSettings::default());
    assert!(!persister.writes_enabled());
    persister.update(|s| s.pyramid_levels = 5);
    persister.flush();
    assert_eq!(persister.write_count(), 0);
    assert!(!path.exists());
    assert_eq!(
        persister.snapshot().pyramid_levels,
        5,
        "kept in memory for the session"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_first_run_writes_the_commented_default_config() {
    let dir = temp_dir("first-run");
    let path = dir.join("nested").join("settings.toml");

    let opened = open_at(Some(&path));
    assert_eq!(opened.outcome, LoadOutcome::Missing);
    assert!(opened.outcome.user_message().is_none());
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        stacker_core::settings::DEFAULT_CONFIG_TOML
    );
    assert_eq!(opened.persister.snapshot(), StackingSettings::default());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn load_from_reads_a_valid_file_and_reports_a_missing_one() {
    let dir = temp_dir("load");
    let missing = load_from(&dir.join("absent.toml"));
    assert_eq!(missing.outcome, LoadOutcome::Missing);
    assert_eq!(missing.settings, StackingSettings::default());

    let path = dir.join("settings.toml");
    fs::write(&path, "pyramid_levels = 4\n").unwrap();
    let loaded = load_from(&path);
    assert_eq!(loaded.settings.pyramid_levels, 4);
    assert!(matches!(loaded.outcome, LoadOutcome::Loaded { .. }));
    assert!(path.exists(), "a good file stays where it is");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn write_atomic_replaces_an_existing_file() {
    let dir = temp_dir("atomic");
    let path = dir.join("settings.toml");
    store::write_atomic(&path, "first = 1\n").unwrap();
    store::write_atomic(&path, "second = 2\n").unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), "second = 2\n");
    assert!(!tmp_sibling(&path).exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_installed_handle_is_per_thread_and_shares_the_state() {
    let persister = std::sync::Arc::new(SettingsPersister::disabled(StackingSettings::default()));
    SettingsPersister::install_for_this_thread(&persister);

    let installed = SettingsPersister::installed_for_this_thread().expect("installed");
    assert!(std::sync::Arc::ptr_eq(&installed, &persister));
    installed.update(|s| s.pyramid_levels = 3);
    assert_eq!(persister.snapshot().pyramid_levels, 3);

    let elsewhere = thread::spawn(|| SettingsPersister::installed_for_this_thread().is_none())
        .join()
        .unwrap();
    assert!(elsewhere, "another thread has its own, empty slot");
}
