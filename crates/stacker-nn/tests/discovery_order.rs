//! Discovery ordering and robustness (no tensor backend needed: the weights files
//! are placeholders because discovery only reads the manifests).

use std::path::Path;

use stacker_nn::discovery::{ModelManifest, discover_models, discover_models_report};

/// Write a placeholder `<stem>.mpk` and a manifest `<stem>.json`.
fn fake(dir: &Path, stem: &str, manifest: &str) {
    std::fs::write(dir.join(format!("{stem}.mpk")), b"placeholder").unwrap();
    std::fs::write(dir.join(format!("{stem}.json")), manifest).unwrap();
}

const XS: &str = r#"{"size":"xs"}"#;

fn names(entries: &[stacker_nn::discovery::ModelEntry]) -> Vec<&str> {
    entries.iter().map(|e| e.name.as_str()).collect()
}

#[test]
fn latest_is_the_default_not_the_first_epoch() {
    // The apps default to the first entry; lexicographic order made that
    // `focusmerge_epoch_000`, the least-trained snapshot.
    let tmp = tempfile::tempdir().unwrap();
    fake(tmp.path(), "focusmerge_epoch_000", XS);
    fake(tmp.path(), "focusmerge_epoch_001", XS);
    fake(tmp.path(), "focusmerge_latest.model", XS);

    let found = discover_models(tmp.path()).unwrap();
    assert_eq!(
        names(&found),
        [
            "focusmerge_latest",
            "focusmerge_epoch_001",
            "focusmerge_epoch_000"
        ]
    );
    assert_eq!(
        found[0].weights_path.file_name().unwrap(),
        "focusmerge_latest.model.mpk"
    );
}

#[test]
fn full_preference_order() {
    let tmp = tempfile::tempdir().unwrap();
    for stem in [
        "tiny",
        "focusmerge_epoch_002",
        "focusmerge_epoch_010",
        "focusmerge_final",
        "focusmerge_latest.model",
        "another",
    ] {
        fake(tmp.path(), stem, XS);
    }
    let found = discover_models(tmp.path()).unwrap();
    // latest, final, epochs newest first (numeric, not lexicographic), the rest A-Z.
    assert_eq!(
        names(&found),
        [
            "focusmerge_latest",
            "focusmerge_final",
            "focusmerge_epoch_010",
            "focusmerge_epoch_002",
            "another",
            "tiny"
        ]
    );
}

#[test]
fn bad_manifests_are_skipped_individually() {
    let tmp = tempfile::tempdir().unwrap();
    fake(tmp.path(), "focusmerge_epoch_000", XS);
    fake(tmp.path(), "focusmerge_final", XS);
    fake(tmp.path(), "corrupt", "{ not json");
    // 20 channels do not split into 8 groups: the manifest check rejects it.
    fake(
        tmp.path(),
        "bad_groups",
        r#"{"size":"xs","width":20,"norm_groups":8}"#,
    );
    fake(
        tmp.path(),
        "zero_groups",
        r#"{"size":"xs","norm_groups":0}"#,
    );

    let report = discover_models_report(tmp.path()).unwrap();
    assert_eq!(
        names(&report.models),
        ["focusmerge_final", "focusmerge_epoch_000"]
    );
    assert_eq!(report.skipped.len(), 3, "{:?}", report.skipped);
    assert!(
        report
            .skipped
            .iter()
            .any(|s| s.reason.contains("not divisible")),
        "{:?}",
        report.skipped
    );

    // The plain entry point agrees and does not abort on the bad files.
    let found = discover_models(tmp.path()).unwrap();
    assert_eq!(found.len(), 2);
}

#[test]
fn optimiser_state_and_legacy_latest_never_become_models() {
    let tmp = tempfile::tempdir().unwrap();
    // The optimiser record has no manifest, so it is not a model.
    std::fs::write(tmp.path().join("focusmerge_latest.optim.mpk"), b"adam").unwrap();
    // The pre-fix trainer wrote Adam moments over `focusmerge_latest.mpk`.
    fake(tmp.path(), "focusmerge_latest", XS);
    fake(tmp.path(), "focusmerge_latest.model", XS);

    let report = discover_models_report(tmp.path()).unwrap();
    assert_eq!(names(&report.models), ["focusmerge_latest"]);
    assert_eq!(
        report.models[0].weights_path.file_name().unwrap(),
        "focusmerge_latest.model.mpk"
    );
    assert_eq!(report.skipped.len(), 1);
    assert!(
        report.skipped[0].reason.contains("legacy"),
        "{:?}",
        report.skipped
    );
}

#[test]
fn manifest_geometry_is_validated() {
    let ok: ModelManifest = serde_json::from_str(r#"{"size":"m"}"#).unwrap();
    assert!(ok.to_config().is_ok());
    let bad: ModelManifest =
        serde_json::from_str(r#"{"size":"xs","width":20,"norm_groups":8}"#).unwrap();
    assert!(bad.to_config().is_err());
}
