//! `prefs.toml`: the suppressed-hints set survives a save and a load, and a bad
//! file never stops the app.

use std::path::PathBuf;

use stacker_gui::settings::prefs::{Prefs, load, prefs_path, save};

fn scratch_file(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("z-stackr-prefs-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir.join("prefs.toml")
}

fn cleanup(path: &std::path::Path) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[test]
fn a_saved_set_loads_back_unchanged() {
    let path = scratch_file("round-trip");
    let mut prefs = Prefs::default();
    assert!(prefs.suppress("auto-cull"));
    assert!(prefs.suppress("relief-preview"));
    save(&path, &prefs).expect("save the prefs");

    let loaded = load(&path);
    assert_eq!(loaded, prefs);
    assert!(loaded.is_suppressed("auto-cull"));
    assert!(loaded.is_suppressed("relief-preview"));
    assert!(!loaded.is_suppressed("retouch-sources"));
    cleanup(&path);
}

#[test]
fn the_text_round_trips_through_parse() {
    let mut prefs = Prefs::default();
    prefs.suppress("b");
    prefs.suppress("a");
    assert_eq!(Prefs::parse(&prefs.to_toml()), prefs);
}

#[test]
fn the_hint_id_of_the_earlier_name_is_read_as_the_current_one() {
    let prefs = Prefs::parse("suppressed_hints = [\"auto-cull\", \"dmap-preview\"]");
    assert!(prefs.is_suppressed("relief-preview"));
    assert!(!prefs.is_suppressed("dmap-preview"));
    assert!(prefs.is_suppressed("auto-cull"));
    // Writing uses the current name only.
    assert!(!prefs.to_toml().contains("dmap"));
}

#[test]
fn an_empty_set_round_trips() {
    let prefs = Prefs::default();
    assert_eq!(Prefs::parse(&prefs.to_toml()), prefs);
}

#[test]
fn suppressing_twice_reports_no_change_the_second_time() {
    let mut prefs = Prefs::default();
    assert!(prefs.suppress("x"));
    assert!(!prefs.suppress("x"));
}

#[test]
fn reset_forgets_every_hint() {
    let mut prefs = Prefs::default();
    prefs.suppress("x");
    prefs.suppress("y");
    assert!(prefs.reset_hints());
    assert!(!prefs.is_suppressed("x"));
    assert!(!prefs.reset_hints(), "nothing left to forget");
}

#[test]
fn a_missing_file_gives_the_defaults() {
    let path = scratch_file("missing");
    assert_eq!(load(&path), Prefs::default());
}

#[test]
fn text_that_is_not_toml_gives_the_defaults() {
    assert_eq!(Prefs::parse("{{ not toml"), Prefs::default());
}

#[test]
fn entries_that_are_not_strings_are_skipped_and_the_rest_kept() {
    let prefs = Prefs::parse("suppressed_hints = [\"keep\", 7, true, \"also\"]\n");
    assert!(prefs.is_suppressed("keep"));
    assert!(prefs.is_suppressed("also"));
    assert_eq!(prefs.suppressed_hints.len(), 2);
}

#[test]
fn a_wrongly_typed_key_gives_nothing_suppressed() {
    assert_eq!(
        Prefs::parse("suppressed_hints = \"auto-cull\"\n"),
        Prefs::default()
    );
}

#[test]
fn unknown_keys_are_ignored() {
    let prefs = Prefs::parse("future_option = 1\nsuppressed_hints = [\"a\"]\n");
    assert!(prefs.is_suppressed("a"));
}

#[test]
fn saving_creates_the_folder() {
    let path = scratch_file("creates-folder");
    assert!(!path.parent().is_some_and(std::path::Path::exists));
    save(&path, &Prefs::default()).expect("save into a folder that does not exist yet");
    assert!(path.is_file());
    cleanup(&path);
}

#[test]
fn the_file_lives_next_to_settings_toml() {
    // `None` only on a machine with no home directory; then there is nothing to check.
    if let Some(path) = prefs_path() {
        assert_eq!(
            path.file_name().and_then(|n| n.to_str()),
            Some("prefs.toml")
        );
    }
}
