//! Architecture tags in model discovery: the four network families share one
//! directory, old manifests (no tag) are still pairwise-fusion models, foreign tags
//! are skipped and reported, every `load_*` checks the tag, and each new network
//! survives a save / discover / load round trip.
#![cfg(feature = "ndarray")]

use std::path::Path;

use burn::{
    backend::NdArray,
    module::{Module, ModuleMapper, ParamId},
    prelude::{Backend, Tensor},
    record::CompactRecorder,
    tensor::Distribution,
};
use stacker_nn::{
    discovery::{
        Architecture, BATCHALIGN_V2, BATCHMERGE_V2, DiscoveryError, FOCUSMERGE_V1, FUSIONALIGN_V1,
        ModelManifest, discover_models, discover_models_report,
    },
    model::{
        BatchAlignNet, BatchAlignNetConfig, BatchMergeNet, BatchMergeNetConfig,
        FocusMergeNetConfig, FusionAlignNet, FusionAlignNetConfig, ModelSize,
    },
    traits::{BatchAlignmentModel, FusionModel, PairAlignmentModel},
};

type B = NdArray;

fn dev() -> <B as Backend>::Device {
    <B as Backend>::Device::default()
}

/// Adds Gaussian noise to the final linear layer (`[width/2, 6]` weight, 6-element
/// bias), so a saved aligner is distinguishable from a freshly initialised one (the
/// alignment heads are zero-initialised). Only the head is touched: jittering the
/// whole network saturates `tanh` and every frame then predicts the same matrix.
struct Jitter(f64);

impl<Bk: Backend> ModuleMapper<Bk> for Jitter {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<Bk, D>) -> Tensor<Bk, D> {
        let dims = tensor.dims();
        if !((D == 2 && dims[1] == 6) || (D == 1 && dims[0] == 6)) {
            return tensor;
        }
        let noise = Tensor::random(
            tensor.shape(),
            Distribution::Normal(0.0, self.0),
            &tensor.device(),
        );
        tensor.add(noise)
    }
}

fn rand<const D: usize>(shape: [usize; D]) -> Tensor<B, D> {
    Tensor::<B, D>::random(shape, Distribution::Uniform(0.0, 1.0), &dev())
}

fn values<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// Save `model` as `<stem>.mpk` with a manifest `<stem>.json`.
fn install<M: Module<B>>(dir: &Path, stem: &str, model: M, manifest: &ModelManifest) {
    model
        .save_file(dir.join(stem), &CompactRecorder::new())
        .unwrap();
    std::fs::write(
        dir.join(format!("{stem}.json")),
        serde_json::to_string(manifest).unwrap(),
    )
    .unwrap();
}

/// Write placeholder weights and a raw manifest (discovery only reads manifests).
fn fake(dir: &Path, stem: &str, manifest: &str) {
    std::fs::write(dir.join(format!("{stem}.mpk")), b"placeholder").unwrap();
    std::fs::write(dir.join(format!("{stem}.json")), manifest).unwrap();
}

fn names(entries: &[stacker_nn::ModelEntry]) -> Vec<&str> {
    entries.iter().map(|e| e.name.as_str()).collect()
}

// ---------------------------------------------------------------------------
// Tags and manifests
// ---------------------------------------------------------------------------

#[test]
fn a_manifest_without_an_architecture_is_a_pairwise_fusion_model() {
    let old: ModelManifest = serde_json::from_str(r#"{"size":"m","width":32}"#).unwrap();
    assert_eq!(old.architecture, FOCUSMERGE_V1);
    assert_eq!(old.architecture_kind(), Some(Architecture::FocusMerge));
    assert!(old.validate().is_ok());
    assert!(old.corr_radius.is_none());
}

#[test]
fn tags_and_bare_family_names_parse_case_insensitively() {
    for (arch, tag, family) in [
        (Architecture::FocusMerge, FOCUSMERGE_V1, "focusmerge"),
        (Architecture::BatchMerge, BATCHMERGE_V2, "batchmerge"),
        (Architecture::BatchAlign, BATCHALIGN_V2, "batchalign"),
        (Architecture::FusionAlign, FUSIONALIGN_V1, "fusionalign"),
    ] {
        assert_eq!(arch.tag(), tag);
        assert_eq!(arch.family(), family);
        assert_eq!(Architecture::parse(tag), Some(arch));
        assert_eq!(Architecture::parse(family), Some(arch));
        assert_eq!(Architecture::parse(&tag.to_uppercase()), Some(arch));
    }
    assert_eq!(Architecture::parse("focusmerge-v9"), None);
    assert_eq!(Architecture::parse(""), None);
    assert!(Architecture::BatchMerge.is_fusion() && !Architecture::BatchMerge.is_alignment());
    assert!(Architecture::FusionAlign.is_alignment() && !Architecture::FusionAlign.is_fusion());
}

#[test]
fn from_size_records_the_tag_and_the_exact_geometry_for_every_family() {
    for size in ModelSize::ALL {
        let focus = ModelManifest::from_size(size);
        assert_eq!(focus.architecture, FOCUSMERGE_V1);
        let want = FocusMergeNetConfig::from_size(size);
        let got = focus.to_config().unwrap();
        assert_eq!((got.width, got.depth), (want.width, want.depth));

        let batch = ModelManifest::from_size_batch(size);
        assert_eq!(batch.architecture, BATCHMERGE_V2);
        let want = BatchMergeNetConfig::from_size(size);
        let got = batch.to_batch_config().unwrap();
        assert_eq!((got.width, got.depth), (want.width, want.depth));
        assert!(batch.corr_radius.is_none());

        for (manifest, tag) in [
            (ModelManifest::from_size_align(size), BATCHALIGN_V2),
            (ModelManifest::from_size_fusion_align(size), FUSIONALIGN_V1),
        ] {
            assert_eq!(manifest.architecture, tag);
            assert!(
                manifest.refine_scale.is_none(),
                "aligners have no refinement head"
            );
            let want = BatchAlignNetConfig::from_size(size);
            let got = manifest.to_align_config().unwrap();
            assert_eq!(
                (got.width, got.depth, got.corr_radius),
                (want.width, want.depth, want.corr_radius)
            );
            assert!(manifest.to_fusion_align_config().is_ok());
        }
        assert_eq!(
            ModelManifest::from_size_for(Architecture::BatchAlign, size).architecture,
            BATCHALIGN_V2
        );
    }
}

#[test]
fn a_manifest_survives_a_json_round_trip() {
    let m = ModelManifest::from_size_align(ModelSize::L);
    let back: ModelManifest = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back.architecture, BATCHALIGN_V2);
    assert_eq!(back.corr_radius, Some(4));
    assert_eq!(back.width, m.width);
}

#[test]
fn alignment_geometry_is_validated_without_panicking() {
    let mut m = ModelManifest::from_size_align(ModelSize::Xs);
    m.width = Some(33);
    assert!(m.to_align_config().is_err(), "odd width");
    assert!(m.validate().is_err());
    let mut m = ModelManifest::from_size_fusion_align(ModelSize::Xs);
    m.corr_radius = Some(99);
    assert!(
        m.to_fusion_align_config().is_err(),
        "huge correlation radius"
    );
    let mut m = ModelManifest::from_size_batch(ModelSize::Xs);
    m.norm_groups = Some(0);
    assert!(m.to_batch_config().is_err());
}

// ---------------------------------------------------------------------------
// Scanning a directory
// ---------------------------------------------------------------------------

#[test]
fn all_four_families_are_listed_with_fusion_models_first() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fake(dir, "focusmerge_epoch_001", r#"{"size":"xs"}"#);
    fake(
        dir,
        "batchmerge_latest.model",
        r#"{"architecture":"batchmerge-v2","size":"xs"}"#,
    );
    fake(
        dir,
        "batchalign_latest.model",
        r#"{"architecture":"batchalign-v2","size":"xs"}"#,
    );
    fake(
        dir,
        "fusionalign_final",
        r#"{"architecture":"fusionalign","size":"s"}"#,
    );

    let report = discover_models_report(dir).unwrap();
    assert!(report.skipped.is_empty(), "{:?}", report.skipped);
    // Fusion before alignment; inside each, latest > final > epochs.
    assert_eq!(
        names(&report.models),
        [
            "batchmerge_latest",
            "focusmerge_epoch_001",
            "batchalign_latest",
            "fusionalign_final"
        ]
    );
    let kinds: Vec<_> = report
        .models
        .iter()
        .map(stacker_nn::ModelEntry::architecture)
        .collect();
    assert_eq!(
        kinds,
        [
            Some(Architecture::BatchMerge),
            Some(Architecture::FocusMerge),
            Some(Architecture::BatchAlign),
            Some(Architecture::FusionAlign)
        ]
    );

    let fusion: Vec<_> = report.models.iter().filter(|e| e.is_fusion()).collect();
    let align: Vec<_> = report.models.iter().filter(|e| e.is_alignment()).collect();
    assert_eq!((fusion.len(), align.len()), (2, 2));
    assert!(align[0].is_batch_alignment() && !align[0].is_pairwise_alignment());
    assert!(align[1].is_pairwise_alignment() && !align[1].is_batch_alignment());
    assert!(fusion.iter().all(|e| !e.is_alignment()));
}

#[test]
fn foreign_tags_and_bad_alignment_geometries_are_skipped_and_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fake(dir, "good", r#"{"size":"xs"}"#);
    fake(
        dir,
        "from_the_future",
        r#"{"architecture":"superfusion-v7","size":"xs"}"#,
    );
    fake(
        dir,
        "odd_aligner",
        r#"{"architecture":"batchalign-v2","size":"xs","width":33}"#,
    );

    let report = discover_models_report(dir).unwrap();
    assert_eq!(names(&report.models), ["good"]);
    assert_eq!(report.skipped.len(), 2, "{:?}", report.skipped);
    assert!(
        report
            .skipped
            .iter()
            .any(|s| s.reason.contains("superfusion-v7")),
        "the foreign tag is named in the reason: {:?}",
        report.skipped
    );
    assert!(report.skipped.iter().any(|s| s.reason.contains("even")));
    // The plain entry point does not abort either.
    assert_eq!(discover_models(dir).unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

#[test]
fn every_load_checks_the_architecture_tag() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    install(
        dir,
        "focus",
        FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev()),
        &ModelManifest::from_size(ModelSize::Xs),
    );
    install(
        dir,
        "aligner",
        BatchAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev()),
        &ModelManifest::from_size_align(ModelSize::Xs),
    );
    let entries = discover_models(dir).unwrap();
    let focus = entries.iter().find(|e| e.name == "focus").unwrap();
    let aligner = entries.iter().find(|e| e.name == "aligner").unwrap();

    assert!(focus.load::<B>(&dev()).is_ok());
    assert!(aligner.load_align::<B>(&dev()).is_ok());

    for result in [
        focus.load_batch::<B>(&dev()).map(|_| ()),
        focus.load_align::<B>(&dev()).map(|_| ()),
        focus.load_fusion_align::<B>(&dev()).map(|_| ()),
        aligner.load::<B>(&dev()).map(|_| ()),
        aligner.load_batch::<B>(&dev()).map(|_| ()),
        aligner.load_fusion_align::<B>(&dev()).map(|_| ()),
    ] {
        assert!(
            matches!(result, Err(DiscoveryError::ArchitectureMismatch { .. })),
            "{result:?}"
        );
    }
    match aligner.load::<B>(&dev()) {
        Err(DiscoveryError::ArchitectureMismatch {
            expected, found, ..
        }) => {
            assert_eq!(expected, FOCUSMERGE_V1);
            assert_eq!(found, BATCHALIGN_V2);
        }
        other => panic!("expected a mismatch, got {:?}", other.map(|_| ())),
    }
}

/// The two alignment families look alike (same knobs), but an entry loads only as
/// the family its tag names.
#[test]
fn an_alignment_entry_loads_only_as_its_own_family() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    install(
        dir,
        "pairs",
        FusionAlignNetConfig::from_size(ModelSize::Xs).init::<B>(&dev()),
        &ModelManifest::from_size_fusion_align(ModelSize::Xs),
    );
    let entry = discover_models(dir).unwrap().remove(0);
    assert!(entry.load_fusion_align::<B>(&dev()).is_ok());
    assert!(matches!(
        entry.load_align::<B>(&dev()),
        Err(DiscoveryError::ArchitectureMismatch { .. })
    ));
}

#[test]
fn batch_merge_round_trips_through_save_discover_load() {
    let tmp = tempfile::tempdir().unwrap();
    let model = BatchMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev());
    install(
        tmp.path(),
        "batchmerge_final",
        model,
        &ModelManifest::from_size_batch(ModelSize::Xs),
    );

    let entry = discover_models(tmp.path()).unwrap().remove(0);
    assert_eq!(entry.architecture(), Some(Architecture::BatchMerge));
    let loaded: BatchMergeNet<B> = entry.load_batch(&dev()).unwrap();
    let direct: BatchMergeNet<B> = stacker_nn::infer::load_weights_batch(
        &entry.manifest.to_batch_config().unwrap(),
        &entry.weights_path,
        &dev(),
    )
    .unwrap();
    let fresh: BatchMergeNet<B> = BatchMergeNetConfig::from_size(ModelSize::Xs).init(&dev());

    let stack = rand([1, 2, 3, 10, 10]);
    let (a, b, c) = (
        values(loaded.fuse_batch(stack.clone())),
        values(direct.fuse_batch(stack.clone())),
        values(fresh.fuse_batch(stack)),
    );
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(
        a.iter().zip(&c).any(|(x, y)| x.to_bits() != y.to_bits()),
        "the loaded model equals a freshly initialised one"
    );
}

#[test]
fn batch_align_round_trips_through_save_discover_load() {
    let tmp = tempfile::tempdir().unwrap();
    let model = BatchAlignNetConfig::from_size(ModelSize::Xs)
        .init::<B>(&dev())
        .map(&mut Jitter(0.3));
    install(
        tmp.path(),
        "batchalign_final",
        model,
        &ModelManifest::from_size_align(ModelSize::Xs),
    );

    let entry = discover_models(tmp.path()).unwrap().remove(0);
    let loaded: BatchAlignNet<B> = entry.load_align(&dev()).unwrap();
    let direct: BatchAlignNet<B> = stacker_nn::infer::load_weights_align(
        &entry.manifest.to_align_config().unwrap(),
        &entry.weights_path,
        &dev(),
    )
    .unwrap();
    let fresh: BatchAlignNet<B> = BatchAlignNetConfig::from_size(ModelSize::Xs).init(&dev());

    let stack = rand([1, 3, 3, 16, 16]);
    let (a, b, c) = (
        values(loaded.align_batch(stack.clone())),
        values(direct.align_batch(stack.clone())),
        values(fresh.align_batch(stack)),
    );
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(
        a.iter().zip(&c).any(|(x, y)| (x - y).abs() > 1e-4),
        "the loaded aligner still predicts the identity of an untrained one"
    );
}

#[test]
fn fusion_align_round_trips_through_save_discover_load() {
    let tmp = tempfile::tempdir().unwrap();
    let model = FusionAlignNetConfig::from_size(ModelSize::Xs)
        .init::<B>(&dev())
        .map(&mut Jitter(0.3));
    install(
        tmp.path(),
        "fusionalign_final",
        model,
        &ModelManifest::from_size_fusion_align(ModelSize::Xs),
    );

    let entry = discover_models(tmp.path()).unwrap().remove(0);
    let loaded: FusionAlignNet<B> = entry.load_fusion_align(&dev()).unwrap();
    let direct: FusionAlignNet<B> = stacker_nn::infer::load_weights_fusion_align(
        &entry.manifest.to_fusion_align_config().unwrap(),
        &entry.weights_path,
        &dev(),
    )
    .unwrap();
    let fresh: FusionAlignNet<B> = FusionAlignNetConfig::from_size(ModelSize::Xs).init(&dev());

    let (reference, frame) = (rand([1, 3, 16, 16]), rand([1, 3, 16, 16]));
    let (a, b, c) = (
        values(loaded.align_pair(reference.clone(), frame.clone())),
        values(direct.align_pair(reference.clone(), frame.clone())),
        values(fresh.align_pair(reference, frame)),
    );
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(
        a.iter().zip(&c).any(|(x, y)| (x - y).abs() > 1e-4),
        "the loaded aligner still predicts the identity of an untrained one"
    );
}
