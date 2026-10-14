//! Crash-safe checkpoints for the `batch`, `align` and `fusion-align` strategies.
//!
//! The pairwise strategy keeps using [`crate::checkpoint`] (its hard-wired
//! `focusmerge_latest.*` stems and `train_state.json`); this module is the same
//! design for any other network, with a file stem per strategy so strategies that
//! share an `--out` directory never touch each other's files:
//!
//! ```text
//!   <stem>_latest.model.mpk    rolling weights          (f32)
//!   <stem>_latest.model.json   its ModelManifest        (carries the architecture tag)
//!   <stem>_latest.optim.mpk    AdamW moments            (f32)
//!   <stem>_train_state.json    StrategyState            (the commit marker)
//!   <stem>_epoch_NNN.mpk/json  per-epoch export         (f16)
//!   <stem>_final.mpk/json      final export             (f16)
//! ```
//!
//! with `<stem>` = [`TrainStrategy::stem_prefix`]. The reasons for the dotted
//! `.model` / `.optim` stems, the f32 rolling and f16 export precisions and the
//! atomic temp-file-and-rename writes are those of [`crate::checkpoint`], whose
//! generic write helpers this module uses directly.
//!
//! A [`StrategyState`] records the strategy as well as the preset: resuming a
//! `batch` run from an `align` state file is refused with
//! [`StrategyCheckpointError::StrategyMismatch`].

use std::path::{Path, PathBuf};

use burn::{
    module::{AutodiffModule, Module},
    optim::Optimizer,
    tensor::backend::{AutodiffBackend, Backend},
};
use serde::{Deserialize, Serialize};

use super::args::TrainStrategy;
use crate::{
    checkpoint::{
        CheckpointError, ExportRecorder, RollingRecorder, load_optim_at, save_model_with,
        save_optim_at, weights_path, write_bytes_atomic,
    },
    discovery::ModelManifest,
    infer::WeightsReader,
    model::ModelSize,
};

/// Errors from the strategy checkpoints: those of [`crate::checkpoint`] plus the
/// strategy check.
#[derive(Debug, thiserror::Error)]
pub enum StrategyCheckpointError {
    /// Any error of the shared checkpoint machinery (I/O, record, state, size,
    /// schedule).
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    /// The state file belongs to a different strategy.
    #[error("resume strategy mismatch: checkpoint is '{checkpoint}', --strategy is '{requested}'")]
    StrategyMismatch {
        /// Strategy recorded in the state file.
        checkpoint: String,
        /// Strategy requested now.
        requested: &'static str,
    },
}

/// Resumable training state of a non-pairwise strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategyState {
    /// Epoch to (re)start from.
    pub epoch: usize,
    /// Global optimiser step at the START of `epoch` (a mid-epoch save stores the
    /// epoch-start step so the epoch reruns from its first example with the
    /// learning-rate schedule rewound to match).
    pub global_step: usize,
    /// Preset the run was started with.
    pub size: ModelSize,
    /// [`TrainStrategy::as_str`] of the run.
    pub strategy: String,
    /// Length of the learning-rate schedule; a resume with a different value is
    /// refused.
    #[serde(default)]
    pub total_steps: Option<usize>,
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Stem of the rolling model checkpoint, e.g. `batchalign_latest.model`.
#[must_use]
pub fn latest_model_stem(strategy: TrainStrategy) -> String {
    format!("{}_latest.model", strategy.stem_prefix())
}

/// Stem of the rolling optimiser checkpoint, e.g. `batchalign_latest.optim`.
#[must_use]
pub fn latest_optim_stem(strategy: TrainStrategy) -> String {
    format!("{}_latest.optim", strategy.stem_prefix())
}

/// Path of the rolling model checkpoint.
#[must_use]
pub fn latest_model_path(dir: &Path, strategy: TrainStrategy) -> PathBuf {
    weights_path(dir, &latest_model_stem(strategy))
}

/// Path of the rolling optimiser checkpoint.
#[must_use]
pub fn latest_optim_path(dir: &Path, strategy: TrainStrategy) -> PathBuf {
    weights_path(dir, &latest_optim_stem(strategy))
}

/// Path of the strategy's state file, e.g. `batchalign_train_state.json`.
#[must_use]
pub fn state_path(dir: &Path, strategy: TrainStrategy) -> PathBuf {
    dir.join(format!("{}_train_state.json", strategy.stem_prefix()))
}

/// Stem of the export of epoch `epoch`, e.g. `batchalign_epoch_003`.
#[must_use]
pub fn epoch_stem(strategy: TrainStrategy, epoch: usize) -> String {
    format!("{}_epoch_{epoch:03}", strategy.stem_prefix())
}

/// Stem of the final export, e.g. `batchalign_final`.
#[must_use]
pub fn final_stem(strategy: TrainStrategy) -> String {
    format!("{}_final", strategy.stem_prefix())
}

// ---------------------------------------------------------------------------
// Model checkpoints
// ---------------------------------------------------------------------------

/// Save the rolling model checkpoint of `strategy` (f32) with its manifest.
///
/// # Errors
/// [`CheckpointError::Io`], [`CheckpointError::Record`] or
/// [`CheckpointError::Serialize`] when a file cannot be written.
pub fn save_latest_model<B, M>(
    model: &M,
    dir: &Path,
    strategy: TrainStrategy,
    manifest: &ModelManifest,
) -> Result<(), CheckpointError>
where
    B: Backend,
    M: Module<B>,
{
    save_model_with::<B, M, _>(
        model,
        dir,
        &latest_model_stem(strategy),
        manifest,
        &RollingRecorder::new(),
    )
}

/// Save a per-epoch or final export `<stem>.mpk` (f16) with its manifest.
///
/// # Errors
/// As [`save_latest_model`].
pub fn save_export<B, M>(
    model: &M,
    dir: &Path,
    stem: &str,
    manifest: &ModelManifest,
) -> Result<(), CheckpointError>
where
    B: Backend,
    M: Module<B>,
{
    save_model_with::<B, M, _>(model, dir, stem, manifest, &ExportRecorder::new())
}

/// Load the rolling weights of `strategy` into `skeleton` (a freshly initialised
/// model of the right configuration).
///
/// # Errors
/// [`CheckpointError::Missing`] when the file does not exist and
/// [`CheckpointError::Record`] when it cannot be read or does not fit the model.
pub fn load_latest_model<B, M>(
    skeleton: M,
    dir: &Path,
    strategy: TrainStrategy,
    device: &B::Device,
) -> Result<M, CheckpointError>
where
    B: Backend,
    M: Module<B>,
{
    let path = latest_model_path(dir, strategy);
    if !path.exists() {
        return Err(CheckpointError::Missing(path));
    }
    skeleton
        .load_file(path.clone(), &WeightsReader::new(), device)
        .map_err(|source| CheckpointError::Record { path, source })
}

// ---------------------------------------------------------------------------
// Optimiser checkpoints
// ---------------------------------------------------------------------------

/// Persist the optimiser state (`AdamW` moments) of `strategy` (f32).
///
/// # Errors
/// [`CheckpointError::Io`] or [`CheckpointError::Record`] when the file cannot be
/// written.
pub fn save_optim<B, M, O>(
    optim: &O,
    dir: &Path,
    strategy: TrainStrategy,
) -> Result<(), CheckpointError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    save_optim_at::<B, M, O>(optim, dir, &latest_optim_path(dir, strategy))
}

/// Restore the optimiser state of `strategy`.
///
/// # Errors
/// [`CheckpointError::Missing`] when the file does not exist and
/// [`CheckpointError::Record`] when it cannot be read.
pub fn load_optim<B, M, O>(
    optim: O,
    dir: &Path,
    strategy: TrainStrategy,
    device: &B::Device,
) -> Result<O, CheckpointError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    load_optim_at::<B, M, O>(optim, latest_optim_path(dir, strategy), device)
}

// ---------------------------------------------------------------------------
// Resume state
// ---------------------------------------------------------------------------

/// Write the strategy's state file atomically.
///
/// # Errors
/// [`CheckpointError::Serialize`] or [`CheckpointError::Io`] when the file cannot
/// be produced.
pub fn save_state(
    dir: &Path,
    strategy: TrainStrategy,
    state: &StrategyState,
) -> Result<(), CheckpointError> {
    let path = state_path(dir, strategy);
    let json =
        serde_json::to_string_pretty(state).map_err(|source| CheckpointError::Serialize {
            path: path.clone(),
            source,
        })?;
    write_bytes_atomic(&path, json.as_bytes())
}

/// Read the strategy's state file.
///
/// # Errors
/// [`CheckpointError::MissingState`] when it does not exist,
/// [`CheckpointError::CorruptState`] when it cannot be parsed and
/// [`CheckpointError::Io`] for any other read failure.
pub fn read_state(dir: &Path, strategy: TrainStrategy) -> Result<StrategyState, CheckpointError> {
    let path = state_path(dir, strategy);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(CheckpointError::MissingState(path));
        }
        Err(source) => return Err(CheckpointError::Io { path, source }),
    };
    serde_json::from_slice(&bytes).map_err(|source| CheckpointError::CorruptState { path, source })
}

/// Everything a rolling save writes, in commit order: model and manifest, the
/// optimiser, then the state file (the commit marker).
///
/// # Errors
/// The first error of [`save_latest_model`], [`save_optim`] or [`save_state`].
pub fn save_rolling<B, M, O>(
    model: &M,
    optim: &O,
    dir: &Path,
    strategy: TrainStrategy,
    manifest: &ModelManifest,
    state: &StrategyState,
) -> Result<(), CheckpointError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    save_latest_model::<B, M>(model, dir, strategy, manifest)?;
    save_optim::<B, M, O>(optim, dir, strategy)?;
    save_state(dir, strategy, state)
}

/// Read the strategy's resume state and check it against the current run.
///
/// # Errors
/// Any error of [`read_state`]; [`StrategyCheckpointError::StrategyMismatch`] when
/// the state names another strategy; [`CheckpointError::SizeMismatch`] or
/// [`CheckpointError::ScheduleMismatch`] when it belongs to a different run;
/// [`CheckpointError::Missing`] when the rolling model file is absent.
pub fn plan_resume(
    dir: &Path,
    strategy: TrainStrategy,
    size: ModelSize,
    total_steps: usize,
) -> Result<StrategyState, StrategyCheckpointError> {
    let state = read_state(dir, strategy)?;
    if state.strategy != strategy.as_str() {
        return Err(StrategyCheckpointError::StrategyMismatch {
            checkpoint: state.strategy,
            requested: strategy.as_str(),
        });
    }
    if state.size != size {
        return Err(CheckpointError::SizeMismatch {
            checkpoint: state.size.as_str(),
            requested: size.as_str(),
        }
        .into());
    }
    if let Some(checkpoint) = state.total_steps
        && checkpoint != total_steps
    {
        return Err(CheckpointError::ScheduleMismatch {
            checkpoint,
            requested: total_steps,
        }
        .into());
    }
    let model = latest_model_path(dir, strategy);
    if !model.exists() {
        return Err(CheckpointError::Missing(model).into());
    }
    Ok(state)
}
