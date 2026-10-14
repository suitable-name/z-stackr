//! Crash-safe training checkpoints and the resume state.
//!
//! The trainer (`src/bin/train.rs`) keeps a *rolling* checkpoint that is refreshed
//! every `--ckpt-every` steps and at every epoch end, plus per-epoch and final
//! *exports*. Everything lives flat in the `--out` directory:
//!
//! ```text
//!   focusmerge_latest.model.mpk    rolling model weights        (f32, RollingRecorder)
//!   focusmerge_latest.model.json   its ModelManifest            (so discovery lists it)
//!   focusmerge_latest.optim.mpk    AdamW moments                (f32, RollingRecorder)
//!   train_state.json               TrainState (resume point)
//!   focusmerge_epoch_NNN.mpk/json  per-epoch export             (f16, ExportRecorder)
//!   focusmerge_final.mpk/json      final export                 (f16, ExportRecorder)
//! ```
//!
//! ## Why the stems contain a dot
//!
//! Burn's file recorders do `path.set_extension("mpk")`, which *replaces* the last
//! extension of whatever path they are given. The old optimiser stem
//! `focusmerge_latest.optim` therefore became `focusmerge_latest.mpk` — the
//! model's own file — and every rolling checkpoint was overwritten by Adam
//! moments. This module never hands a recorder a bare stem: it always passes the
//! complete `<stem>.mpk` path (see [`weights_path`]), on which `set_extension`
//! is a no-op, so `.model` and `.optim` survive and the two files stay distinct.
//!
//! ## Which recorder reads what
//!
//! | file | written by | precision | read by |
//! |---|---|---|---|
//! | `focusmerge_latest.model.mpk` | [`RollingRecorder`] | f32 | `infer::load_weights` |
//! | `focusmerge_latest.optim.mpk` | [`RollingRecorder`] | f32 | [`load_optim`] |
//! | `focusmerge_epoch_NNN.mpk`, `focusmerge_final.mpk` | [`ExportRecorder`] | f16 | `infer::load_weights` |
//!
//! Records are self-describing (each tensor carries its dtype) and the recorder
//! settings only pick the element type tensors are converted *to* on load, so one
//! reader (`infer::WeightsReader`, f32) loads both precisions: f16 files widen
//! exactly, f32 files are untouched. Reading an f32 file with the half-precision
//! `CompactRecorder` would silently round it, which is why that never happens.
//! Half precision is fine for the exports (smaller, inference-only) but not for
//! resuming: it rounds the weights and flushes most of Adam's second moments
//! (values below ~6e-5 are subnormal in f16), so the rolling files use f32.
//!
//! ## Atomicity
//!
//! Every file is written to a temporary name in the same directory and then
//! renamed over the target, so a crash or power loss leaves either the previous
//! complete file or the new complete file, never a torn one. A rolling save writes
//! model, manifest, optimiser and finally `train_state.json`; the state file is
//! the commit marker that [`plan_resume`] reads first.
//!
//! ## Resume point
//!
//! [`TrainState`] records the *epoch to (re)start* and the *global step at the
//! start of that epoch*. A mid-epoch rolling save stores the current epoch with
//! the epoch-start step, so on resume the epoch is rerun from its first scene with
//! the step counter (and therefore the cosine learning-rate schedule) rewound to
//! match; the few partial-epoch steps already in the weights are simply trained
//! again. Resuming never silently restarts: a missing or corrupt state file is an
//! error ([`CheckpointError::MissingState`] / [`CheckpointError::CorruptState`]),
//! and the caller decides (the trainer's `--fresh`) whether to start over.

use std::{
    ffi::OsString,
    fs::OpenOptions,
    path::{Path, PathBuf},
};

use burn::{
    module::{AutodiffModule, Module},
    optim::Optimizer,
    record::{
        CompactRecorder, FileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder,
        RecorderError,
    },
    tensor::backend::{AutodiffBackend, Backend},
};
use serde::{Deserialize, Serialize};

use crate::{
    discovery::{Architecture, ModelManifest},
    model::{FocusMergeNet, ModelSize},
};

/// Stem of the rolling model checkpoint (`focusmerge_latest.model.mpk`).
pub const LATEST_MODEL_STEM: &str = "focusmerge_latest.model";
/// Stem of the rolling optimiser checkpoint (`focusmerge_latest.optim.mpk`).
pub const LATEST_OPTIM_STEM: &str = "focusmerge_latest.optim";
/// File name of the resumable training-state sidecar.
pub const STATE_FILE: &str = "train_state.json";

/// Recorder for the rolling checkpoint (model and optimiser): f32, lossless.
pub type RollingRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;
/// Recorder for the per-epoch and final exports: f16, compact.
pub type ExportRecorder = CompactRecorder;

/// Errors from writing or reading checkpoints.
#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    /// Filesystem error.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// A recorder failed to write or read a record.
    #[error("checkpoint record error at {path}: {source}")]
    Record {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying recorder error.
        #[source]
        source: RecorderError,
    },
    /// Serialising a JSON sidecar failed.
    #[error("cannot serialise {path}: {source}")]
    Serialize {
        /// Path being written.
        path: PathBuf,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
    /// A file the resume needs is not there.
    #[error("nothing to resume from: {0} does not exist")]
    Missing(PathBuf),
    /// `train_state.json` is absent.
    #[error("cannot resume: {0} does not exist")]
    MissingState(PathBuf),
    /// `train_state.json` exists but cannot be parsed.
    #[error("cannot resume: {path} is corrupt: {source}")]
    CorruptState {
        /// The state file.
        path: PathBuf,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
    /// The checkpoint was trained at a different preset than requested.
    #[error("resume size mismatch: checkpoint is '{checkpoint}', --size is '{requested}'")]
    SizeMismatch {
        /// Preset recorded in the checkpoint.
        checkpoint: &'static str,
        /// Preset requested now.
        requested: &'static str,
    },
    /// The learning-rate schedule length differs from the checkpointed run.
    #[error(
        "resume schedule mismatch: checkpoint was started for {checkpoint} total steps, this run \
         has {requested} (same --epochs and scene count are required)"
    )]
    ScheduleMismatch {
        /// Total steps recorded in the checkpoint.
        checkpoint: usize,
        /// Total steps of the current run.
        requested: usize,
    },
}

/// Wrap an I/O error at `path` as a [`CheckpointError::Io`] (shared with the
/// strategy checkpoints).
pub(crate) fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> CheckpointError + '_ {
    move |source| CheckpointError::Io {
        path: path.to_path_buf(),
        source,
    }
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// `<dir>/<stem>.mpk` — the complete weights path for `stem`.
///
/// Built by appending, never with `with_extension`, so a dotted stem such as
/// `focusmerge_latest.model` keeps its `.model` marker.
#[must_use]
pub fn weights_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.mpk"))
}

/// `<dir>/<stem>.json` — the manifest path for `stem` (appended, see
/// [`weights_path`]).
#[must_use]
pub fn manifest_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.json"))
}

/// Path of the rolling model checkpoint.
#[must_use]
pub fn latest_model_path(dir: &Path) -> PathBuf {
    weights_path(dir, LATEST_MODEL_STEM)
}

/// Path of the rolling optimiser checkpoint.
#[must_use]
pub fn latest_optim_path(dir: &Path) -> PathBuf {
    weights_path(dir, LATEST_OPTIM_STEM)
}

/// Path of the resume-state sidecar.
#[must_use]
pub fn state_path(dir: &Path) -> PathBuf {
    dir.join(STATE_FILE)
}

// ---------------------------------------------------------------------------
// Atomic writes
// ---------------------------------------------------------------------------

/// Flush `path` to disk (best effort durability before the rename).
fn sync_file(path: &Path) -> Result<(), CheckpointError> {
    OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|f| f.sync_all())
        .map_err(io_err(path))
}

/// Write `bytes` to `target` atomically (temp file + rename).
pub(crate) fn write_bytes_atomic(target: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let mut tmp_name: OsString = target.as_os_str().to_owned();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);
    std::fs::write(&tmp, bytes).map_err(io_err(&tmp))?;
    sync_file(&tmp)?;
    rename_over(&tmp, target)
}

/// Rename `tmp` over `target`, removing `tmp` if that fails.
fn rename_over(tmp: &Path, target: &Path) -> Result<(), CheckpointError> {
    std::fs::rename(tmp, target).map_err(|source| {
        let _ = std::fs::remove_file(tmp);
        CheckpointError::Io {
            path: target.to_path_buf(),
            source,
        }
    })
}

/// Write a Burn record to `target` (a complete `….mpk` path) atomically.
///
/// `write` receives a temporary path that already ends in `.mpk` and contains no
/// other dot, so the recorder's `set_extension("mpk")` leaves it alone; the file
/// it produces is then renamed over `target`.
pub(crate) fn write_mpk_atomic<F>(target: &Path, write: F) -> Result<(), CheckpointError>
where
    F: FnOnce(PathBuf) -> Result<(), RecorderError>,
{
    let name = target.file_name().map_or_else(
        || "checkpoint".to_owned(),
        |n| n.to_string_lossy().into_owned(),
    );
    let tmp = target.with_file_name(format!("tmp_{}.mpk", name.replace('.', "_")));
    write(tmp.clone()).map_err(|source| CheckpointError::Record {
        path: target.to_path_buf(),
        source,
    })?;
    sync_file(&tmp)?;
    rename_over(&tmp, target)
}

// ---------------------------------------------------------------------------
// Model checkpoints
// ---------------------------------------------------------------------------

/// Save any network `model` as `<dir>/<stem>.mpk` plus `manifest` as
/// `<dir>/<stem>.json`, both atomically, weights first (a manifest therefore never
/// points at missing weights). Shared with the strategy checkpoints.
pub(crate) fn save_model_with<B, M, R>(
    model: &M,
    dir: &Path,
    stem: &str,
    manifest: &ModelManifest,
    recorder: &R,
) -> Result<(), CheckpointError>
where
    B: Backend,
    M: Module<B>,
    R: FileRecorder<B>,
{
    std::fs::create_dir_all(dir).map_err(io_err(dir))?;
    let weights = weights_path(dir, stem);
    write_mpk_atomic(&weights, |tmp| model.clone().save_file(tmp, recorder))?;

    let manifest_file = manifest_path(dir, stem);
    let json =
        serde_json::to_string_pretty(manifest).map_err(|source| CheckpointError::Serialize {
            path: manifest_file.clone(),
            source,
        })?;
    write_bytes_atomic(&manifest_file, json.as_bytes())
}

/// Persist the optimiser state of any network to `target` (a complete `….mpk`
/// path inside `dir`) in full precision, atomically. Shared with the strategy
/// checkpoints.
pub(crate) fn save_optim_at<B, M, O>(
    optim: &O,
    dir: &Path,
    target: &Path,
) -> Result<(), CheckpointError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    std::fs::create_dir_all(dir).map_err(io_err(dir))?;
    let record = optim.to_record();
    write_mpk_atomic(target, |tmp| {
        <RollingRecorder as Recorder<B>>::record(&RollingRecorder::new(), record, tmp)
    })
}

/// Restore the optimiser state of any network from `path`; a missing file is
/// [`CheckpointError::Missing`]. Shared with the strategy checkpoints.
pub(crate) fn load_optim_at<B, M, O>(
    optim: O,
    path: PathBuf,
    device: &B::Device,
) -> Result<O, CheckpointError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    if !path.exists() {
        return Err(CheckpointError::Missing(path));
    }
    let record: O::Record =
        <RollingRecorder as Recorder<B>>::load(&RollingRecorder::new(), path.clone(), device)
            .map_err(|source| CheckpointError::Record { path, source })?;
    Ok(optim.load_record(record))
}

/// Save the rolling model checkpoint (`focusmerge_latest.model.mpk` + manifest) in
/// full precision.
///
/// # Errors
/// [`CheckpointError::Io`], [`CheckpointError::Record`] or
/// [`CheckpointError::Serialize`] when a file cannot be written.
pub fn save_latest_model<B: Backend>(
    model: &FocusMergeNet<B>,
    dir: &Path,
    size: ModelSize,
) -> Result<(), CheckpointError> {
    save_model_with::<B, _, _>(
        model,
        dir,
        LATEST_MODEL_STEM,
        &ModelManifest::from_size_for(Architecture::FocusMerge, size),
        &RollingRecorder::new(),
    )
}

/// Save a per-epoch or final export (`<stem>.mpk` + manifest) in half precision.
///
/// # Errors
/// As [`save_latest_model`].
pub fn save_export<B: Backend>(
    model: &FocusMergeNet<B>,
    dir: &Path,
    stem: &str,
    size: ModelSize,
) -> Result<(), CheckpointError> {
    save_model_with::<B, _, _>(
        model,
        dir,
        stem,
        &ModelManifest::from_size_for(Architecture::FocusMerge, size),
        &ExportRecorder::new(),
    )
}

// ---------------------------------------------------------------------------
// Optimiser checkpoints
// ---------------------------------------------------------------------------

/// Persist the optimiser state (`AdamW` moments) to `focusmerge_latest.optim.mpk`.
///
/// Written in full precision so a resumed run continues with momentum and
/// variance intact rather than re-warming from zero.
///
/// # Errors
/// [`CheckpointError::Io`] or [`CheckpointError::Record`] when the file cannot be
/// written.
pub fn save_optim<B, O>(optim: &O, dir: &Path) -> Result<(), CheckpointError>
where
    B: AutodiffBackend,
    O: Optimizer<FocusMergeNet<B>, B>,
{
    save_optim_at::<B, FocusMergeNet<B>, O>(optim, dir, &latest_optim_path(dir))
}

/// Restore the optimiser state from `focusmerge_latest.optim.mpk`.
///
/// A missing file is an error ([`CheckpointError::Missing`]): resuming the weights
/// with cold Adam moments is a different run, and the caller must ask for that
/// explicitly.
///
/// # Errors
/// [`CheckpointError::Missing`] when the file does not exist and
/// [`CheckpointError::Record`] when it cannot be read.
pub fn load_optim<B, O>(optim: O, dir: &Path, device: &B::Device) -> Result<O, CheckpointError>
where
    B: AutodiffBackend,
    O: Optimizer<FocusMergeNet<B>, B>,
{
    load_optim_at::<B, FocusMergeNet<B>, O>(optim, latest_optim_path(dir), device)
}

// ---------------------------------------------------------------------------
// Resume state
// ---------------------------------------------------------------------------

/// Resumable training state, persisted next to the rolling checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainState {
    /// Epoch to (re)start from.
    pub epoch: usize,
    /// Global optimiser step at the START of `epoch` (not the step the weights
    /// have reached): the epoch is rerun from its first scene with the step counter
    /// rewound to here, so the cosine schedule ends exactly at `total_steps`.
    pub global_step: usize,
    /// Preset the run was started with (sanity check on resume).
    pub size: ModelSize,
    /// Length of the learning-rate schedule (`epochs * scenes`); a resume with a
    /// different value would warp the schedule and is refused. Absent in state
    /// files written before this field existed (then not checked).
    #[serde(default)]
    pub total_steps: Option<usize>,
}

/// Write `train_state.json` atomically.
///
/// # Errors
/// [`CheckpointError::Serialize`] or [`CheckpointError::Io`] when the file cannot
/// be produced.
pub fn save_state(dir: &Path, state: &TrainState) -> Result<(), CheckpointError> {
    let path = state_path(dir);
    let json =
        serde_json::to_string_pretty(state).map_err(|source| CheckpointError::Serialize {
            path: path.clone(),
            source,
        })?;
    write_bytes_atomic(&path, json.as_bytes())
}

/// Read `train_state.json`. Missing and corrupt are distinct errors; neither is
/// ever turned into "start from epoch 0".
///
/// # Errors
/// [`CheckpointError::MissingState`] when the file does not exist,
/// [`CheckpointError::CorruptState`] when it cannot be parsed, and
/// [`CheckpointError::Io`] for any other read failure.
pub fn read_state(dir: &Path) -> Result<TrainState, CheckpointError> {
    let path = state_path(dir);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(CheckpointError::MissingState(path));
        }
        Err(source) => return Err(CheckpointError::Io { path, source }),
    };
    serde_json::from_slice(&bytes).map_err(|source| CheckpointError::CorruptState { path, source })
}

/// Everything a rolling save writes, in commit order: model + manifest, then the
/// optimiser, then `train_state.json` (the commit marker).
///
/// # Errors
/// The first error of [`save_latest_model`], [`save_optim`] or [`save_state`].
pub fn save_rolling<B, O>(
    model: &FocusMergeNet<B>,
    optim: &O,
    dir: &Path,
    state: &TrainState,
) -> Result<(), CheckpointError>
where
    B: AutodiffBackend,
    O: Optimizer<FocusMergeNet<B>, B>,
{
    save_latest_model(model, dir, state.size)?;
    save_optim::<B, O>(optim, dir)?;
    save_state(dir, state)
}

/// Read the resume state and check it against the current run.
///
/// `size` is the requested `--size`, `total_steps` the current schedule length
/// (`epochs * scenes`). Also requires the rolling model file to exist.
///
/// # Errors
/// Any error of [`read_state`], [`CheckpointError::SizeMismatch`] or
/// [`CheckpointError::ScheduleMismatch`] when the checkpoint belongs to a
/// different run, and [`CheckpointError::Missing`] when the rolling model file is
/// absent.
pub fn plan_resume(
    dir: &Path,
    size: ModelSize,
    total_steps: usize,
) -> Result<TrainState, CheckpointError> {
    let state = read_state(dir)?;
    if state.size != size {
        return Err(CheckpointError::SizeMismatch {
            checkpoint: state.size.as_str(),
            requested: size.as_str(),
        });
    }
    if let Some(checkpoint) = state.total_steps
        && checkpoint != total_steps
    {
        return Err(CheckpointError::ScheduleMismatch {
            checkpoint,
            requested: total_steps,
        });
    }
    let model = latest_model_path(dir);
    if !model.exists() {
        return Err(CheckpointError::Missing(model));
    }
    Ok(state)
}
