//! Checkpoints of the non-pairwise strategies: per-strategy stems, atomic round
//! trip of weights and optimiser, resume checks and the manifest's architecture tag.
#![cfg(all(feature = "ndarray", feature = "autodiff"))]

use burn::{
    backend::{Autodiff, NdArray},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::{Backend, Tensor},
    tensor::{Distribution, TensorData},
};
use stacker_nn::{
    checkpoint::CheckpointError,
    data::AlignSample,
    discovery::{ModelManifest, discover_models},
    loss::corner_alignment::BatchAlignmentLossConfig,
    model::{
        ModelSize,
        batch_align::{BatchAlignNet, BatchAlignNetConfig},
    },
    train::{
        align_loss,
        args::TrainStrategy,
        strategy_checkpoint::{self as sc, StrategyCheckpointError, StrategyState},
    },
};

type Ab = Autodiff<NdArray>;

fn device() -> <Ab as Backend>::Device {
    <Ab as Backend>::Device::default()
}

fn state(strategy: TrainStrategy, epoch: usize, total_steps: usize) -> StrategyState {
    StrategyState {
        epoch,
        global_step: 12,
        size: ModelSize::Xs,
        strategy: strategy.as_str().to_owned(),
        total_steps: Some(total_steps),
    }
}

/// A model that has been through one optimiser step (so its moments are not empty)
/// together with that optimiser.
fn trained_once() -> (BatchAlignNet<Ab>, impl Optimizer<BatchAlignNet<Ab>, Ab>) {
    let device = device();
    let model = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device);
    let loss = BatchAlignmentLossConfig::new().init();
    let mut shifted = Vec::new();
    for _ in 0..2 {
        shifted.extend([1.0, 0.0, 0.1, 0.0, 1.0, -0.1, 0.0, 0.0, 1.0_f32]);
    }
    let sample = AlignSample {
        stack: Tensor::random([2, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &device),
        gt_matrices: Tensor::from_data(TensorData::new(shifted, [2, 3, 3]), &device),
    };
    let mut optim = AdamWConfig::new().init::<Ab, BatchAlignNet<Ab>>();
    let grads = GradientsParams::from_grads(align_loss(&model, &loss, &sample).backward(), &model);
    let model = optim.step(1e-2, model, grads);
    (model, optim)
}

/// The aligner's output on a fixed input, to compare two models by behaviour.
fn behaviour(model: &BatchAlignNet<Ab>) -> Vec<f32> {
    let pixels: Vec<f32> = (0..2 * 3 * 32 * 32_u32)
        .map(|i| f32::from(u8::try_from(i % 97).unwrap()) / 97.0)
        .collect();
    let stack = Tensor::<Ab, 5>::from_data(TensorData::new(pixels, [1, 2, 3, 32, 32]), &device());
    let (matrices, _) = model.forward_with_params(stack);
    matrices.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn strategies_never_share_a_file() {
    let dir = std::path::Path::new("out");
    let mut files = Vec::new();
    for strategy in TrainStrategy::ALL {
        files.push(sc::latest_model_path(dir, strategy));
        files.push(sc::latest_optim_path(dir, strategy));
        files.push(sc::state_path(dir, strategy));
        files.push(dir.join(format!("{}.mpk", sc::epoch_stem(strategy, 3))));
        files.push(dir.join(format!("{}.mpk", sc::final_stem(strategy))));
    }
    let count = files.len();
    files.sort();
    files.dedup();
    assert_eq!(files.len(), count);

    let name = |p: std::path::PathBuf| p.file_name().unwrap().to_string_lossy().into_owned();
    assert_eq!(
        name(sc::latest_model_path(dir, TrainStrategy::Align)),
        "batchalign_latest.model.mpk"
    );
    assert_eq!(
        name(sc::latest_optim_path(dir, TrainStrategy::FusionAlign)),
        "fusionalign_latest.optim.mpk"
    );
    assert_eq!(
        name(sc::state_path(dir, TrainStrategy::Batch)),
        "batchmerge_train_state.json"
    );
    assert_eq!(
        sc::epoch_stem(TrainStrategy::Align, 3),
        "batchalign_epoch_003"
    );
    assert_eq!(sc::final_stem(TrainStrategy::Batch), "batchmerge_final");
    // The pairwise trainer keeps its own files, none of which is one of these.
    assert_eq!(
        name(stacker_nn::checkpoint::latest_model_path(dir)),
        "focusmerge_latest.model.mpk"
    );
}

#[test]
fn rolling_save_round_trips_weights_optimiser_and_state() {
    let tmp = tempfile::tempdir().unwrap();
    let strategy = TrainStrategy::Align;
    let (model, optim) = trained_once();
    let manifest = ModelManifest::from_size_align(ModelSize::Xs);
    let saved = state(strategy, 3, 40);

    sc::save_rolling::<Ab, _, _>(&model, &optim, tmp.path(), strategy, &manifest, &saved).unwrap();
    for name in [
        "batchalign_latest.model.mpk",
        "batchalign_latest.model.json",
        "batchalign_latest.optim.mpk",
        "batchalign_train_state.json",
    ] {
        assert!(tmp.path().join(name).exists(), "{name} missing");
    }
    assert!(
        !tmp.path().join("train_state.json").exists(),
        "the pairwise state file is not touched"
    );

    assert_eq!(sc::read_state(tmp.path(), strategy).unwrap(), saved);
    assert_eq!(
        sc::plan_resume(tmp.path(), strategy, ModelSize::Xs, 40).unwrap(),
        saved
    );

    let skeleton = BatchAlignNetConfig::from_size(ModelSize::Xs).init::<Ab>(&device());
    assert_ne!(
        behaviour(&skeleton),
        behaviour(&model),
        "one step moved the weights"
    );
    let loaded = sc::load_latest_model::<Ab, _>(skeleton, tmp.path(), strategy, &device()).unwrap();
    assert_eq!(
        behaviour(&loaded),
        behaviour(&model),
        "f32 rolling weights are exact"
    );

    let fresh = AdamWConfig::new().init::<Ab, BatchAlignNet<Ab>>();
    sc::load_optim::<Ab, BatchAlignNet<Ab>, _>(fresh, tmp.path(), strategy, &device()).unwrap();

    // Overwriting in place leaves no temporary files behind.
    sc::save_rolling::<Ab, _, _>(&model, &optim, tmp.path(), strategy, &manifest, &saved).unwrap();
    let leftovers: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            n.starts_with("tmp_")
                || std::path::Path::new(n)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"))
        })
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn resume_refuses_a_different_run() {
    let tmp = tempfile::tempdir().unwrap();
    let strategy = TrainStrategy::Align;
    let (model, optim) = trained_once();
    let manifest = ModelManifest::from_size_align(ModelSize::Xs);
    sc::save_rolling::<Ab, _, _>(
        &model,
        &optim,
        tmp.path(),
        strategy,
        &manifest,
        &state(strategy, 3, 40),
    )
    .unwrap();

    assert!(matches!(
        sc::plan_resume(tmp.path(), strategy, ModelSize::S, 40),
        Err(StrategyCheckpointError::Checkpoint(
            CheckpointError::SizeMismatch { .. }
        ))
    ));
    assert!(matches!(
        sc::plan_resume(tmp.path(), strategy, ModelSize::Xs, 41),
        Err(StrategyCheckpointError::Checkpoint(
            CheckpointError::ScheduleMismatch {
                checkpoint: 40,
                requested: 41
            }
        ))
    ));

    // A state file that names another strategy (copied or renamed by hand).
    let foreign = StrategyState {
        strategy: TrainStrategy::Batch.as_str().to_owned(),
        ..state(strategy, 3, 40)
    };
    sc::save_state(tmp.path(), strategy, &foreign).unwrap();
    assert!(matches!(
        sc::plan_resume(tmp.path(), strategy, ModelSize::Xs, 40),
        Err(StrategyCheckpointError::StrategyMismatch {
            requested: "align",
            ..
        })
    ));
}

#[test]
fn resume_reports_missing_and_corrupt_files() {
    let tmp = tempfile::tempdir().unwrap();
    let strategy = TrainStrategy::FusionAlign;
    assert!(matches!(
        sc::plan_resume(tmp.path(), strategy, ModelSize::Xs, 10),
        Err(StrategyCheckpointError::Checkpoint(
            CheckpointError::MissingState(_)
        ))
    ));

    // A state without its weights is not a checkpoint.
    sc::save_state(tmp.path(), strategy, &state(strategy, 1, 10)).unwrap();
    assert!(matches!(
        sc::plan_resume(tmp.path(), strategy, ModelSize::Xs, 10),
        Err(StrategyCheckpointError::Checkpoint(
            CheckpointError::Missing(_)
        ))
    ));

    std::fs::write(sc::state_path(tmp.path(), strategy), "{ not json").unwrap();
    assert!(matches!(
        sc::read_state(tmp.path(), strategy),
        Err(CheckpointError::CorruptState { .. })
    ));
}

#[test]
fn exports_are_discoverable_and_carry_the_architecture_tag() {
    let tmp = tempfile::tempdir().unwrap();
    let strategy = TrainStrategy::Align;
    let (model, _optim) = trained_once();
    let manifest = ModelManifest::from_size_align(ModelSize::Xs);

    sc::save_export::<Ab, _>(&model, tmp.path(), &sc::final_stem(strategy), &manifest).unwrap();
    sc::save_export::<Ab, _>(&model, tmp.path(), &sc::epoch_stem(strategy, 0), &manifest).unwrap();

    let json = std::fs::read_to_string(tmp.path().join("batchalign_final.json")).unwrap();
    let written: ModelManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(written.architecture, manifest.architecture);
    assert_ne!(
        written.architecture,
        ModelManifest::from_size(ModelSize::Xs).architecture,
        "an alignment checkpoint must not look like a focus-merge one"
    );
    assert_eq!(written.corr_radius, manifest.corr_radius);

    let found = discover_models(tmp.path()).unwrap();
    let names: Vec<_> = found.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"batchalign_final"), "{names:?}");
    assert!(names.contains(&"batchalign_epoch_000"), "{names:?}");
    assert!(found.iter().all(stacker_nn::ModelEntry::is_alignment));
}
