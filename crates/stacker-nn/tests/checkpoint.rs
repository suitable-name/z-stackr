//! Tests for the crash-safe training checkpoints in [`stacker_nn::checkpoint`].
#![cfg(all(feature = "ndarray", feature = "autodiff"))]

use std::path::Path;

use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::{Backend, Int, Tensor},
    tensor::Distribution,
};
use stacker_nn::{
    checkpoint::{self, CheckpointError, TrainState},
    discovery::discover_models,
    infer::load_weights,
    model::{FocusMergeNet, FocusMergeNetConfig, ModelSize},
    train::cosine_lr,
};

type Ab = Autodiff<NdArray>;
type Cpu = NdArray;

/// A tiny model after one `AdamW` step, so both its weights and the optimiser
/// moments are non-trivial (non-zero, not representable in f16).
fn trained(
    dev: &<Ab as Backend>::Device,
) -> (FocusMergeNet<Ab>, impl Optimizer<FocusMergeNet<Ab>, Ab>) {
    let model: FocusMergeNet<Ab> = FocusMergeNetConfig::from_size(ModelSize::Xs).init(dev);
    let mut optim = AdamWConfig::new().init::<Ab, FocusMergeNet<Ab>>();
    let rand4 =
        |shape: [usize; 4]| Tensor::<Ab, 4>::random(shape, Distribution::Uniform(0.0, 1.0), dev);
    let out = model.forward(
        rand4([1, 3, 8, 8]),
        rand4([1, 1, 8, 8]),
        rand4([1, 3, 8, 8]),
    );
    let grads = GradientsParams::from_grads(out.merged.mean().backward(), &model);
    let model = optim.step(1e-2, model, grads);
    (model, optim)
}

/// Forward output of `model` on a fixed deterministic input, as raw f32 values.
fn probe(model: &FocusMergeNet<Cpu>, dev: <Cpu as Backend>::Device) -> Vec<f32> {
    let ramp = |range: std::ops::Range<i64>, shape: [usize; 4]| {
        Tensor::<Cpu, 1, Int>::arange(range, &dev)
            .float()
            .mul_scalar(0.05)
            .sin()
            .reshape(shape)
    };
    let out = model.forward(
        ramp(0..192, [1, 3, 8, 8]),
        ramp(0..64, [1, 1, 8, 8]).abs(),
        ramp(7..199, [1, 3, 8, 8]),
    );
    out.merged.into_data().to_vec::<f32>().unwrap()
}

fn file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------------------------
// NN-1: model and optimiser checkpoints must not share a file
// ---------------------------------------------------------------------------

#[test]
fn optimiser_checkpoint_does_not_overwrite_model_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = <Ab as Backend>::Device::default();
    let (model, optim) = trained(&dev);

    checkpoint::save_latest_model(&model, tmp.path(), ModelSize::Xs).unwrap();
    let model_file = checkpoint::latest_model_path(tmp.path());
    assert_eq!(
        model_file.file_name().unwrap(),
        "focusmerge_latest.model.mpk"
    );
    assert!(model_file.exists());
    let model_bytes = std::fs::read(&model_file).unwrap();

    checkpoint::save_optim::<Ab, _>(&optim, tmp.path()).unwrap();
    let optim_file = checkpoint::latest_optim_path(tmp.path());
    assert_eq!(
        optim_file.file_name().unwrap(),
        "focusmerge_latest.optim.mpk"
    );
    assert!(optim_file.exists());

    assert_eq!(
        std::fs::read(&model_file).unwrap(),
        model_bytes,
        "writing the optimiser state must not touch the model file"
    );
    let optim_bytes = std::fs::read(&optim_file).unwrap();
    assert_ne!(optim_bytes, model_bytes, "the two files must differ");
    assert!(
        !tmp.path().join("focusmerge_latest.mpk").exists(),
        "Burn's extension replacement must not produce the old shared name"
    );

    // The model file loads (it would fail or hold Adam moments if it had been
    // overwritten) and so does the optimiser state; restoring then re-saving the
    // optimiser reproduces the file byte for byte.
    let cpu = <Cpu as Backend>::Device::default();
    let cfg = FocusMergeNetConfig::from_size(ModelSize::Xs);
    let _loaded: FocusMergeNet<Cpu> = load_weights(&cfg, &model_file, &cpu).unwrap();

    let fresh = AdamWConfig::new().init::<Ab, FocusMergeNet<Ab>>();
    let restored = checkpoint::load_optim::<Ab, _>(fresh, tmp.path(), &dev).unwrap();
    // Burn serialises the optimiser record through maps whose entry order is not
    // stable across runs, so a byte comparison of two saves is not meaningful;
    // what matters is that the restored state saves and loads again.
    let again = tempfile::tempdir().unwrap();
    checkpoint::save_optim::<Ab, _>(&restored, again.path()).unwrap();
    let resaved = std::fs::read(checkpoint::latest_optim_path(again.path())).unwrap();
    assert_eq!(
        resaved.len(),
        optim_bytes.len(),
        "a re-saved optimiser state must have the same size as the original"
    );
    let fresh_again = AdamWConfig::new().init::<Ab, FocusMergeNet<Ab>>();
    checkpoint::load_optim::<Ab, _>(fresh_again, again.path(), &dev)
        .expect("the re-saved optimiser state must load again");
}

#[test]
fn loading_missing_optimiser_state_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = <Ab as Backend>::Device::default();
    let optim = AdamWConfig::new().init::<Ab, FocusMergeNet<Ab>>();
    let Err(err) = checkpoint::load_optim::<Ab, _>(optim, tmp.path(), &dev) else {
        panic!("no optimiser file: must not silently continue with cold moments");
    };
    assert!(matches!(err, CheckpointError::Missing(_)), "{err}");
}

#[test]
fn rolling_model_is_discoverable_and_loadable() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = <Ab as Backend>::Device::default();
    let (model, _optim) = trained(&dev);
    checkpoint::save_latest_model(&model, tmp.path(), ModelSize::Xs).unwrap();

    // The dotted stem must keep its manifest: `<stem>.json`, not `focusmerge_latest.json`.
    assert!(tmp.path().join("focusmerge_latest.model.json").exists());
    assert!(!tmp.path().join("focusmerge_latest.json").exists());

    let found = discover_models(tmp.path()).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].name, "focusmerge_latest");
    let cpu = <Cpu as Backend>::Device::default();
    let _loaded: FocusMergeNet<Cpu> = found[0].load(&cpu).unwrap();
}

// ---------------------------------------------------------------------------
// NN-3: rolling checkpoints are lossless, exports are half precision
// ---------------------------------------------------------------------------

#[test]
fn rolling_checkpoint_roundtrip_is_bit_identical() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = <Ab as Backend>::Device::default();
    let cpu = <Cpu as Backend>::Device::default();
    let (model, _optim) = trained(&dev);
    let cfg = FocusMergeNetConfig::from_size(ModelSize::Xs);

    let expected = probe(&model.valid(), cpu);

    checkpoint::save_latest_model(&model, tmp.path(), ModelSize::Xs).unwrap();
    let reloaded: FocusMergeNet<Cpu> =
        load_weights(&cfg, &checkpoint::latest_model_path(tmp.path()), &cpu).unwrap();
    assert_eq!(
        probe(&reloaded, cpu),
        expected,
        "full-precision rolling checkpoint must reload bit-identically"
    );

    // The half-precision export reads through the same reader, within f16 error.
    checkpoint::save_export(&model, tmp.path(), "focusmerge_final", ModelSize::Xs).unwrap();
    let exported: FocusMergeNet<Cpu> = load_weights(
        &cfg,
        &checkpoint::weights_path(tmp.path(), "focusmerge_final"),
        &cpu,
    )
    .unwrap();
    let max_diff = probe(&exported, cpu)
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 5e-2, "f16 export drifted by {max_diff}");
}

// ---------------------------------------------------------------------------
// NN-4: atomic writes; missing / corrupt state is an error
// ---------------------------------------------------------------------------

#[test]
fn rolling_save_is_complete_and_leaves_no_temporaries() {
    let tmp = tempfile::tempdir().unwrap();
    let dev = <Ab as Backend>::Device::default();
    let (model, optim) = trained(&dev);
    let mut state = TrainState {
        epoch: 0,
        global_step: 0,
        size: ModelSize::Xs,
        total_steps: Some(10),
    };

    checkpoint::save_rolling::<Ab, _>(&model, &optim, tmp.path(), &state).unwrap();
    // Overwriting existing checkpoints (rename over an existing file) works too.
    state.epoch = 1;
    state.global_step = 10;
    checkpoint::save_rolling::<Ab, _>(&model, &optim, tmp.path(), &state).unwrap();

    assert_eq!(
        file_names(tmp.path()),
        [
            "focusmerge_latest.model.json",
            "focusmerge_latest.model.mpk",
            "focusmerge_latest.optim.mpk",
            "train_state.json",
        ]
    );
    assert_eq!(checkpoint::read_state(tmp.path()).unwrap(), state);
}

#[test]
fn missing_corrupt_and_mismatched_state_are_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    assert!(matches!(
        checkpoint::read_state(dir),
        Err(CheckpointError::MissingState(_))
    ));
    assert!(matches!(
        checkpoint::plan_resume(dir, ModelSize::Xs, 10),
        Err(CheckpointError::MissingState(_))
    ));

    std::fs::write(dir.join("train_state.json"), b"{ \"epoch\": 3, \"glob").unwrap();
    assert!(matches!(
        checkpoint::read_state(dir),
        Err(CheckpointError::CorruptState { .. })
    ));
    assert!(matches!(
        checkpoint::plan_resume(dir, ModelSize::Xs, 10),
        Err(CheckpointError::CorruptState { .. })
    ));

    let state = TrainState {
        epoch: 2,
        global_step: 20,
        size: ModelSize::Xs,
        total_steps: Some(100),
    };
    checkpoint::save_state(dir, &state).unwrap();
    // A valid state without the model file is still not resumable.
    assert!(matches!(
        checkpoint::plan_resume(dir, ModelSize::Xs, 100),
        Err(CheckpointError::Missing(_))
    ));
    std::fs::write(checkpoint::latest_model_path(dir), b"x").unwrap();

    assert_eq!(
        checkpoint::plan_resume(dir, ModelSize::Xs, 100).unwrap(),
        state
    );
    assert!(matches!(
        checkpoint::plan_resume(dir, ModelSize::M, 100),
        Err(CheckpointError::SizeMismatch { .. })
    ));
    assert!(matches!(
        checkpoint::plan_resume(dir, ModelSize::Xs, 101),
        Err(CheckpointError::ScheduleMismatch {
            checkpoint: 100,
            requested: 101
        })
    ));
}

#[test]
fn state_files_from_before_total_steps_still_resume() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("train_state.json"),
        br#"{"epoch":4,"global_step":40,"size":"xs"}"#,
    )
    .unwrap();
    std::fs::write(checkpoint::latest_model_path(tmp.path()), b"x").unwrap();
    let state = checkpoint::plan_resume(tmp.path(), ModelSize::Xs, 123).unwrap();
    assert_eq!((state.epoch, state.global_step), (4, 40));
    assert_eq!(state.total_steps, None);
}

#[test]
fn mid_epoch_resume_rewinds_the_schedule_to_the_epoch_start() {
    // One epoch of 10 scenes; the run died after 3 of them. The trainer stores the
    // epoch-START step (0), so the resumed epoch replays all 10 steps and ends on
    // the schedule's last step instead of overshooting to learning rate 0.
    let (scenes, total, warmup, base) = (10_usize, 10_usize, 1_usize, 3e-4_f64);
    let tmp = tempfile::tempdir().unwrap();
    let saved = TrainState {
        epoch: 0,
        global_step: 0,
        size: ModelSize::Xs,
        total_steps: Some(total),
    };
    checkpoint::save_state(tmp.path(), &saved).unwrap();
    std::fs::write(checkpoint::latest_model_path(tmp.path()), b"x").unwrap();

    let resumed = checkpoint::plan_resume(tmp.path(), ModelSize::Xs, total).unwrap();
    let final_step = resumed.global_step + scenes - 1;
    assert_eq!(final_step, total - 1);
    let lr = cosine_lr(base, final_step, total, warmup);
    assert!(lr > 0.0 && lr < base, "final-step lr {lr}");

    // The pre-fix bookkeeping kept the 3 partial steps and reran the epoch.
    let overshoot = cosine_lr(base, 3 + scenes - 1, total, warmup);
    assert!(overshoot.abs() < 1e-12, "overshoot lr {overshoot}");
}
