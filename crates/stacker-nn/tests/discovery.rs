//! Integration tests for trained-model discovery.

use burn::{backend::NdArray, module::Module, record::CompactRecorder};
use stacker_nn::{
    ModelManifest,
    discovery::{
        BATCHALIGN_V2, BATCHMERGE_V2, DiscoveryError, FOCUSMERGE_V1, FUSIONALIGN_V1, ModelEntry,
        discover_models,
    },
    model::{
        BatchAlignNetConfig, BatchMergeNetConfig, FocusMergeNet, FocusMergeNetConfig,
        FusionAlignNetConfig, ModelSize,
    },
};

type B = NdArray;

#[test]
fn discover_missing_dir_is_empty() {
    let dir = std::path::Path::new("/nonexistent/xyzzy/models");
    assert!(discover_models(dir).unwrap().is_empty());
}

#[test]
fn discover_and_load_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();

    // Save a tiny model as "tiny.mpk" with a matching minimal manifest.
    let model = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    model
        .clone()
        .save_file(tmp.path().join("tiny"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(tmp.path().join("tiny.json"), r#"{"size":"xs"}"#).unwrap();

    // A bare .mpk without a manifest must be ignored.
    model
        .save_file(tmp.path().join("orphan"), &CompactRecorder::new())
        .unwrap();

    let found = discover_models(tmp.path()).unwrap();
    assert_eq!(
        found.len(),
        1,
        "orphan .mpk without manifest should be skipped"
    );
    assert_eq!(found[0].name, "tiny");
    assert_eq!(found[0].manifest.size, ModelSize::Xs);

    // The discovered entry loads into a usable model.
    let _loaded: FocusMergeNet<B> = found[0].load(&dev).unwrap();
}

/// A manifest JSON written before the `architecture` field existed (e.g. the
/// minimal `{"size":"xs"}` used by `discover_and_load_roundtrip` above) must
/// still deserialise, defaulting `architecture` to [`FOCUSMERGE_V1`].
#[test]
fn manifest_without_architecture_field_defaults_to_focusmerge_v1() {
    let manifest: ModelManifest = serde_json::from_str(r#"{"size":"m"}"#).unwrap();
    assert_eq!(manifest.architecture, FOCUSMERGE_V1);
    assert_eq!(manifest.size, ModelSize::M);
}

/// [`ModelManifest::from_size`] must stamp the built-in architecture tag.
#[test]
fn from_size_stamps_focusmerge_v1_architecture() {
    let manifest = ModelManifest::from_size(ModelSize::S);
    assert_eq!(manifest.architecture, FOCUSMERGE_V1);
}

/// A `.mpk` whose sidecar manifest declares a foreign `architecture` (e.g. a
/// third-party [`stacker_nn::traits::PairwiseFusionModel`] implementation's
/// own checkpoint format) must be skipped by [`discover_models`] rather than
/// misloaded as a `FocusMergeNet` — its parameter layout is unknown and
/// cannot be reconstructed safely.
#[test]
fn discover_models_skips_unknown_architecture() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();

    // A real FocusMergeNet checkpoint...
    let model = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    model
        .clone()
        .save_file(tmp.path().join("known"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("known.json"),
        r#"{"architecture":"focusmerge-v1","size":"xs"}"#,
    )
    .unwrap();

    // ...alongside a foreign-architecture entry (weights content is
    // irrelevant here since discovery must skip it before ever loading them).
    model
        .save_file(tmp.path().join("foreign"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("foreign.json"),
        r#"{"architecture":"other/v2","size":"xs"}"#,
    )
    .unwrap();

    let found = discover_models(tmp.path()).unwrap();
    assert_eq!(
        found.len(),
        1,
        "unknown-architecture manifest should be skipped, not loaded"
    );
    assert_eq!(found[0].name, "known");
    assert_eq!(found[0].manifest.architecture, FOCUSMERGE_V1);
}

// ---------------------------------------------------------------------------
// Architecture-tag validation across load / load_batch / load_align.
// ---------------------------------------------------------------------------

/// Write a `batchalign-v2`-tagged [`ModelEntry`] (weights content is
/// irrelevant to the mismatch checks — only the manifest tag is read before
/// any deserialisation is attempted).
fn write_batchalign_entry(dir: &std::path::Path) -> ModelEntry {
    let dev = burn::prelude::Device::<B>::default();
    let model = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    model
        .save_file(dir.join("align_entry"), &CompactRecorder::new())
        .unwrap();
    let manifest = ModelManifest::from_size_align(ModelSize::Xs);
    std::fs::write(
        dir.join("align_entry.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    ModelEntry {
        name: "align_entry".to_owned(),
        weights_path: dir.join("align_entry.mpk"),
        manifest,
    }
}

/// A `batchalign-v2` entry must be rejected by `.load()` and `.load_batch()`
/// with [`DiscoveryError::ArchitectureMismatch`], but accepted by
/// `.load_align()`.
#[test]
fn batchalign_entry_rejected_by_load_and_load_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();
    let entry = write_batchalign_entry(tmp.path());

    let err = entry.load::<B>(&dev).unwrap_err();
    assert!(
        matches!(
            err,
            DiscoveryError::ArchitectureMismatch {
                expected: FOCUSMERGE_V1,
                ..
            }
        ),
        "expected ArchitectureMismatch(expected=focusmerge-v1), got: {err:?}"
    );

    let err = entry.load_batch::<B>(&dev).unwrap_err();
    assert!(
        matches!(
            err,
            DiscoveryError::ArchitectureMismatch {
                expected: BATCHMERGE_V2,
                ..
            }
        ),
        "expected ArchitectureMismatch(expected=batchmerge-v2), got: {err:?}"
    );

    // Accepted by load_align.
    let _loaded = entry.load_align::<B>(&dev).unwrap();
}

/// A `focusmerge-v1` entry must be rejected by `.load_align()`.
#[test]
fn focusmerge_entry_rejected_by_load_align() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();

    let model = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    model
        .save_file(tmp.path().join("fm_entry"), &CompactRecorder::new())
        .unwrap();
    let manifest = ModelManifest::from_size(ModelSize::Xs);
    std::fs::write(
        tmp.path().join("fm_entry.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    let entry = ModelEntry {
        name: "fm_entry".to_owned(),
        weights_path: tmp.path().join("fm_entry.mpk"),
        manifest,
    };

    let err = entry.load_align::<B>(&dev).unwrap_err();
    assert!(
        matches!(
            err,
            DiscoveryError::ArchitectureMismatch {
                expected: BATCHALIGN_V2,
                ..
            }
        ),
        "expected ArchitectureMismatch(expected=batchalign-v2), got: {err:?}"
    );
}

/// A `batchmerge-v2` entry must round-trip correctly through `.load_batch()`.
#[test]
fn batchmerge_entry_round_trips_through_load_batch() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();

    let model = BatchMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    model
        .save_file(tmp.path().join("bm_entry"), &CompactRecorder::new())
        .unwrap();
    let manifest = ModelManifest::from_size_batch(ModelSize::Xs);
    std::fs::write(
        tmp.path().join("bm_entry.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    let entry = ModelEntry {
        name: "bm_entry".to_owned(),
        weights_path: tmp.path().join("bm_entry.mpk"),
        manifest,
    };

    let _loaded = entry.load_batch::<B>(&dev).unwrap();
}

/// [`discover_models`] must return entries for all three built-in
/// architecture tags in one directory, while still skipping a genuinely
/// unknown 4th tag (extends `discover_models_skips_unknown_architecture`
/// above, which only covers the two-tag case).
#[test]
fn discover_models_returns_all_three_known_architectures() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();

    let fm = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    fm.clone()
        .save_file(tmp.path().join("fm"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("fm.json"),
        serde_json::to_string(&ModelManifest::from_size(ModelSize::Xs)).unwrap(),
    )
    .unwrap();

    let bm = BatchMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    bm.save_file(tmp.path().join("bm"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("bm.json"),
        serde_json::to_string(&ModelManifest::from_size_batch(ModelSize::Xs)).unwrap(),
    )
    .unwrap();

    let ba = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    ba.save_file(tmp.path().join("ba"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("ba.json"),
        serde_json::to_string(&ModelManifest::from_size_align(ModelSize::Xs)).unwrap(),
    )
    .unwrap();

    // A genuinely unknown 4th tag must still be skipped.
    fm.save_file(tmp.path().join("unknown"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("unknown.json"),
        r#"{"architecture":"other/v2","size":"xs"}"#,
    )
    .unwrap();

    let found = discover_models(tmp.path()).unwrap();
    let tags: std::collections::HashSet<&str> = found
        .iter()
        .map(|e| e.manifest.architecture.as_str())
        .collect();

    assert_eq!(
        found.len(),
        3,
        "expected exactly the 3 known-architecture entries"
    );
    assert!(tags.contains(FOCUSMERGE_V1));
    assert!(tags.contains(BATCHMERGE_V2));
    assert!(tags.contains(BATCHALIGN_V2));
}

/// A `.mpk` whose sidecar manifest declares an architecture tag with no
/// corresponding built-in constant must be skipped by [`discover_models`],
/// exactly like any other unrecognised architecture — see that function's
/// docs and [`discover_models_skips_unknown_architecture`] above.
#[test]
fn discover_models_skips_unrecognised_batchmerge_style_tag() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();

    let bm = BatchMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    bm.save_file(tmp.path().join("old_batch"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("old_batch.json"),
        r#"{"architecture":"batchmerge-v1","size":"xs"}"#,
    )
    .unwrap();

    let found = discover_models(tmp.path()).unwrap();
    assert!(
        found.is_empty(),
        "a manifest with an unrecognised architecture tag should be skipped, found: {found:?}"
    );
}

/// [`ModelManifest::from_size_batch`] must stamp the current
/// [`BATCHMERGE_V2`] architecture tag.
#[test]
fn from_size_batch_stamps_batchmerge_v2_architecture() {
    let manifest = ModelManifest::from_size_batch(ModelSize::S);
    assert_eq!(manifest.architecture, BATCHMERGE_V2);
}

// ---------------------------------------------------------------------------
// FUSIONALIGN_V1 — the pairwise alignment architecture tag.
// ---------------------------------------------------------------------------

/// [`ModelManifest::from_size_fusion_align`] must stamp [`FUSIONALIGN_V1`].
#[test]
fn from_size_fusion_align_stamps_fusionalign_v1_architecture() {
    let manifest = ModelManifest::from_size_fusion_align(ModelSize::S);
    assert_eq!(manifest.architecture, FUSIONALIGN_V1);
}

/// Write a `fusionalign-v1`-tagged [`ModelEntry`] (weights content is
/// irrelevant to the mismatch checks — only the manifest tag is read before
/// any deserialisation is attempted) — mirrors `write_batchalign_entry`.
fn write_fusionalign_entry(dir: &std::path::Path) -> ModelEntry {
    let dev = burn::prelude::Device::<B>::default();
    let model = FusionAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    model
        .save_file(dir.join("fusion_align_entry"), &CompactRecorder::new())
        .unwrap();
    let manifest = ModelManifest::from_size_fusion_align(ModelSize::Xs);
    std::fs::write(
        dir.join("fusion_align_entry.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    ModelEntry {
        name: "fusion_align_entry".to_owned(),
        weights_path: dir.join("fusion_align_entry.mpk"),
        manifest,
    }
}

/// A `fusionalign-v1` entry must round-trip through `.load_fusion_align()`,
/// but be rejected by `.load()`, `.load_batch()`, and `.load_align()` with
/// [`DiscoveryError::ArchitectureMismatch`].
#[test]
fn fusionalign_entry_round_trips_and_is_rejected_by_other_loaders() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();
    let entry = write_fusionalign_entry(tmp.path());

    let _loaded = entry.load_fusion_align::<B>(&dev).unwrap();

    let err = entry.load::<B>(&dev).unwrap_err();
    assert!(
        matches!(
            err,
            DiscoveryError::ArchitectureMismatch {
                expected: FOCUSMERGE_V1,
                ..
            }
        ),
        "expected ArchitectureMismatch(expected=focusmerge-v1), got: {err:?}"
    );

    let err = entry.load_batch::<B>(&dev).unwrap_err();
    assert!(
        matches!(
            err,
            DiscoveryError::ArchitectureMismatch {
                expected: BATCHMERGE_V2,
                ..
            }
        ),
        "expected ArchitectureMismatch(expected=batchmerge-v2), got: {err:?}"
    );

    let err = entry.load_align::<B>(&dev).unwrap_err();
    assert!(
        matches!(
            err,
            DiscoveryError::ArchitectureMismatch {
                expected: BATCHALIGN_V2,
                ..
            }
        ),
        "expected ArchitectureMismatch(expected=batchalign-v2), got: {err:?}"
    );
}

/// A `batchalign-v2` entry must be rejected by `.load_fusion_align()`.
#[test]
fn batchalign_entry_rejected_by_load_fusion_align() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();
    let entry = write_batchalign_entry(tmp.path());

    let err = entry.load_fusion_align::<B>(&dev).unwrap_err();
    assert!(
        matches!(
            err,
            DiscoveryError::ArchitectureMismatch {
                expected: FUSIONALIGN_V1,
                ..
            }
        ),
        "expected ArchitectureMismatch(expected=fusionalign-v1), got: {err:?}"
    );
}

/// [`discover_models`] must return entries for all FOUR built-in
/// architecture tags in one directory, while still skipping a genuinely
/// unknown 5th tag — extends `discover_models_returns_all_three_known_architectures`
/// with the `fusionalign-v1` tag.
#[test]
fn discover_models_returns_all_four_known_architectures() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = burn::prelude::Device::<B>::default();

    let fm = FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    fm.clone()
        .save_file(tmp.path().join("fm"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("fm.json"),
        serde_json::to_string(&ModelManifest::from_size(ModelSize::Xs)).unwrap(),
    )
    .unwrap();

    let bm = BatchMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    bm.save_file(tmp.path().join("bm"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("bm.json"),
        serde_json::to_string(&ModelManifest::from_size_batch(ModelSize::Xs)).unwrap(),
    )
    .unwrap();

    let ba = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    ba.save_file(tmp.path().join("ba"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("ba.json"),
        serde_json::to_string(&ModelManifest::from_size_align(ModelSize::Xs)).unwrap(),
    )
    .unwrap();

    let fa = FusionAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev);
    fa.save_file(tmp.path().join("fa"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("fa.json"),
        serde_json::to_string(&ModelManifest::from_size_fusion_align(ModelSize::Xs)).unwrap(),
    )
    .unwrap();

    // A genuinely unknown 5th tag must still be skipped.
    fm.save_file(tmp.path().join("unknown"), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        tmp.path().join("unknown.json"),
        r#"{"architecture":"other/v2","size":"xs"}"#,
    )
    .unwrap();

    let found = discover_models(tmp.path()).unwrap();
    let tags: std::collections::HashSet<&str> = found
        .iter()
        .map(|e| e.manifest.architecture.as_str())
        .collect();

    assert_eq!(
        found.len(),
        4,
        "expected exactly the 4 known-architecture entries"
    );
    assert!(tags.contains(FOCUSMERGE_V1));
    assert!(tags.contains(BATCHMERGE_V2));
    assert!(tags.contains(BATCHALIGN_V2));
    assert!(tags.contains(FUSIONALIGN_V1));
}

/// [`ModelEntry::is_pairwise_alignment`]/[`ModelEntry::is_alignment`] must
/// both recognise a `fusionalign-v1` entry, while
/// [`ModelEntry::is_batch_alignment`]/[`ModelEntry::is_fusion`] must not.
#[test]
fn fusionalign_entry_is_pairwise_alignment_not_batch_or_fusion() {
    let tmp = tempfile::tempdir().unwrap();
    let entry = write_fusionalign_entry(tmp.path());

    assert!(entry.is_alignment());
    assert!(entry.is_pairwise_alignment());
    assert!(!entry.is_batch_alignment());
    assert!(!entry.is_fusion());
}
