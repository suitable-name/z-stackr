//! `stacker-nn-train` — training driver for [`FocusMergeNet`], [`BatchMergeNet`],
//! [`BatchAlignNet`], and [`FusionAlignNet`].
//!
//! This binary only requires the `autodiff` backend (in the default feature
//! set), so it builds with the default workspace build and is clippy-verified by
//! CI. `burn::optim` (`AdamW`) is available because `burn/std` is enabled by every
//! backend feature.
//!
//! The heavy differentiable logic (the recurrent rollout, loss aggregation,
//! scheduled sampling, LR schedule) lives in the library module
//! [`stacker_nn::train`], which is exercised by CI on the autodiff backend.
//! This file is only the optimiser loop, checkpointing, and CLI plumbing.
//!
//! ## Usage
//!
//! ```text
//! # Default CPU (ndarray) autodiff backend:
//! cargo run -p stacker-nn --release --bin stacker-nn-train -- \
//!     --data /path/to/scenes --out checkpoints --epochs 60 --lr 3e-4 --crop 256 --strategy pairwise
//!
//! # Batch fusion:
//! cargo run -p stacker-nn --release --bin stacker-nn-train -- \
//!     --data /path/to/scenes --out checkpoints --strategy batch
//!
//! # Alignment (data from scripts/03_simulate_misalignment.py):
//! cargo run -p stacker-nn --release --bin stacker-nn-train -- \
//!     --data /path/to/scenes_unaligned --out checkpoints --strategy align
//!
//! # Pairwise (streaming) alignment — SAME scene data as `--strategy align`,
//! # just trained one reference/frame pair at a time:
//! cargo run -p stacker-nn --release --bin stacker-nn-train -- \
//!     --data /path/to/scenes_unaligned --out checkpoints --strategy fusion-align
//!
//! # GPU (CUDA) backend:
//! cargo run -p stacker-nn --release --no-default-features \
//!     --features "cuda,autodiff" --bin stacker-nn-train -- --data /path/to/scenes
//! ```
//!
//! `--data` must point at a directory of scene sub-directories. For
//! `--strategy pairwise|batch` each scene contains the `metadata.json`
//! written by `02_blender_focus_stack.py`; for `--strategy align|fusion-align`
//! each scene contains the `metadata.json` written by
//! `03_simulate_misalignment.py` (which augments a stage-2 scene with
//! ground-truth alignment matrices) — `fusion-align` reuses the EXACT same
//! on-disk scenes as `align`, just extracting one reference/frame pair per
//! training step instead of the whole stack (see
//! [`stacker_nn::data::AlignSequence::get_pair`] and
//! `docs/fusionalign-design.md`).
//!
//! ## Checkpoint naming
//!
//! Each strategy writes to its own checkpoint stem so concurrent/successive
//! runs with different `--strategy` values in the same `--out` directory
//! never clobber each other:
//!
//! | Strategy | Rolling checkpoint | Per-epoch checkpoint | Final checkpoint |
//! |---|---|---|---|
//! | `pairwise` | `focusmerge_latest.mpk` | `focusmerge_epoch_NNN.mpk` | `focusmerge_final.mpk` |
//! | `batch` | `batchmerge_latest.mpk` | `batchmerge_epoch_NNN.mpk` | `batchmerge_final.mpk` |
//! | `align` | `batchalign_latest.mpk` | `batchalign_epoch_NNN.mpk` | `batchalign_final.mpk` |
//! | `fusion-align` | `fusionalign_latest.mpk` | `fusionalign_epoch_NNN.mpk` | `fusionalign_final.mpk` |
//!
//! (`pairwise` keeps its original `focusmerge_*` stems for backward
//! compatibility with existing checkpoints/scripts.) Every `.mpk` gets a
//! sibling `.json` manifest stamped with the matching architecture tag (see
//! [`stacker_nn::discovery`]) — this is what makes `--strategy batch`
//! checkpoints loadable via [`stacker_nn::discovery::ModelEntry::load_batch`]
//! instead of being misread as a `focusmerge-v1` checkpoint.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

use std::{
    error::Error,
    path::{Path, PathBuf},
};

use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{CompactRecorder, Recorder},
    tensor::ElementConversion,
};
use serde::{Deserialize, Serialize};

use stacker_nn::{
    TrainBackend,
    data::{
        AlignSequence, CropParams, MergeSample, RealStackScene, RolloutSequence,
        discover_real_stacks,
    },
    discovery::ModelManifest,
    loss::{
        BatchAlignmentLossConfig, FocusBatchLossConfig, FocusFusionLossConfig,
        PairAlignmentLossConfig, photometric_gradient_loss,
    },
    model::{
        BatchAlignNetConfig, BatchMergeNetConfig, FocusMergeNetConfig, FusionAlignNetConfig,
        ModelSize,
    },
    train::{
        align_loss, batch_loss, cosine_lr, fusion_align_loss, rollout_loss, scheduled_sampling_prob,
    },
};

/// File name (stem) of the resumable training-state sidecar.
const STATE_FILE: &str = "train_state.json";

/// Autodiff training backend (wraps whichever base backend feature is active).
type B = TrainBackend;

/// Which of the crate's four strategies this run trains. Distinct from
/// [`stacker_nn::traits::FusionStrategy`] (which only covers the two FUSION
/// strategies, `Pairwise`/`Batch`) because alignment is not a `FusionModel`
/// variant at all — it implements separate traits
/// ([`stacker_nn::traits::BatchAlignmentModel`] /
/// [`stacker_nn::traits::PairAlignmentModel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainStrategy {
    Pairwise,
    Batch,
    Align,
    FusionAlign,
}

impl TrainStrategy {
    /// Parse a `--strategy` value. Returns `None` if unrecognised.
    fn parse(s: &str) -> Option<Self> {
        match s {
            "pairwise" => Some(Self::Pairwise),
            "batch" => Some(Self::Batch),
            "align" => Some(Self::Align),
            "fusion-align" => Some(Self::FusionAlign),
            _ => None,
        }
    }

    /// Serialise for [`TrainState::strategy`] / resume-mismatch checks.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Pairwise => "pairwise",
            Self::Batch => "batch",
            Self::Align => "align",
            Self::FusionAlign => "fusion-align",
        }
    }

    /// Checkpoint stem prefix for this strategy — see the module docs'
    /// "Checkpoint naming" table. `pairwise` keeps the original `focusmerge`
    /// prefix for backward compatibility.
    const fn stem_prefix(self) -> &'static str {
        match self {
            Self::Pairwise => "focusmerge",
            Self::Batch => "batchmerge",
            Self::Align => "batchalign",
            Self::FusionAlign => "fusionalign",
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Parsed command-line arguments.
struct Args {
    data: PathBuf,
    out: PathBuf,
    size: ModelSize,
    strategy: TrainStrategy,
    epochs: usize,
    lr: f64,
    crop: usize,
    seed: u64,
    ss_max: f64,
    /// Save a rolling crash-recovery checkpoint every N optimiser steps.
    ckpt_every: usize,
    /// Resume from `<out>/<stem_prefix>_latest.mpk` + `train_state.json` if present.
    resume: bool,
    /// `--strategy align` only: correlation search radius passed to
    /// [`BatchAlignNetConfig::corr_radius`] (`docs/batchalign-v2-design.md`
    /// §3.1/§5.3). Defaults to the config's own default (4) when unset.
    corr_radius: Option<usize>,
    /// `--strategy align` only: optional directory of real, unlabelled focus
    /// stacks (plain image-folder scenes, no metadata — see
    /// [`stacker_nn::data::RealStackScene`]) enabling the §5.2 photometric
    /// fine-tuning phase for the last 25% of epochs. `None` disables the
    /// phase entirely (pure supervised training, the v2 baseline behaviour).
    real_data: Option<PathBuf>,
    /// Weight of the photometric fine-tune term relative to the supervised
    /// corner loss during the fine-tune phase (`docs/batchalign-v2-design.md`
    /// §5.2's `corner (synthetic) + 0.1 * photometric (real, unlabelled)`).
    photometric_weight: f64,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut data: Option<PathBuf> = None;
        let mut out = PathBuf::from("checkpoints");
        let mut size = ModelSize::M;
        let mut strategy = TrainStrategy::Pairwise;
        let mut epochs = 40usize;
        let mut lr = 3e-4_f64;
        let mut crop = 256usize;
        let mut seed = 42u64;
        let mut ss_max = 0.5_f64;
        let mut ckpt_every = 200usize;
        let mut resume = false;
        let mut corr_radius: Option<usize> = None;
        let mut real_data: Option<PathBuf> = None;
        let mut photometric_weight = 0.1_f64;

        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let flag = argv[i].clone();
            // `--resume` is a flag (no value); everything else consumes the next arg.
            if flag == "--resume" {
                resume = true;
                i += 1;
                continue;
            }
            if flag == "-h" || flag == "--help" {
                print_usage();
                std::process::exit(0);
            }
            match flag.as_str() {
                "--data" => data = Some(PathBuf::from(value_after(&argv, i, &flag)?)),
                "--out" => out = PathBuf::from(value_after(&argv, i, &flag)?),
                "--size" => {
                    let v = value_after(&argv, i, &flag)?;
                    size = ModelSize::parse(&v)
                        .ok_or_else(|| format!("invalid --size '{v}' (xs|s|m|l|xl|xxl)"))?;
                }
                "--strategy" => {
                    let v = value_after(&argv, i, &flag)?;
                    strategy = TrainStrategy::parse(&v).ok_or_else(|| {
                        format!("invalid --strategy '{v}' (pairwise|batch|align|fusion-align)")
                    })?;
                }
                "--epochs" => epochs = value_after(&argv, i, &flag)?.parse()?,
                "--lr" => lr = value_after(&argv, i, &flag)?.parse()?,
                "--crop" => crop = value_after(&argv, i, &flag)?.parse()?,
                "--seed" => seed = value_after(&argv, i, &flag)?.parse()?,
                "--ss-max" => ss_max = value_after(&argv, i, &flag)?.parse()?,
                "--ckpt-every" => ckpt_every = value_after(&argv, i, &flag)?.parse()?,
                "--corr-radius" => corr_radius = Some(value_after(&argv, i, &flag)?.parse()?),
                "--real-data" => real_data = Some(PathBuf::from(value_after(&argv, i, &flag)?)),
                "--photometric-weight" => {
                    photometric_weight = value_after(&argv, i, &flag)?.parse()?;
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
            i += 2;
        }

        let data = data.ok_or("--data <dir> is required")?;
        Ok(Self {
            data,
            out,
            size,
            strategy,
            epochs,
            lr,
            crop,
            seed,
            ss_max,
            ckpt_every: ckpt_every.max(1),
            resume,
            corr_radius,
            real_data,
            photometric_weight,
        })
    }
}

/// Return the argument following position `i` (the value for `flag`).
fn value_after(argv: &[String], i: usize, flag: &str) -> Result<String, Box<dyn Error>> {
    argv.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn print_usage() {
    eprintln!(
        "stacker-nn-train --data <scenes_dir> [--out <dir>] [--size xs|s|m|l|xl|xxl] \
         [--strategy pairwise|batch|align|fusion-align] [--epochs N] [--lr F] [--crop N] \
         [--seed N] [--ss-max F] [--ckpt-every N] [--resume] \
         [--corr-radius N] [--real-data <dir>] [--photometric-weight F]\n\
         \n\
         --corr-radius, --real-data, --photometric-weight only apply to \
         --strategy align (docs/batchalign-v2-design.md §5.2-5.3): \
         --corr-radius overrides BatchAlignNetConfig::corr_radius; \
         --real-data <dir> of plain image-folder scenes (no metadata) enables \
         the photometric fine-tune phase for the LAST 25% of epochs; \
         --photometric-weight sets that phase's `corner + w * photometric` mix \
         (default 0.1).\n\
         \n\
         --strategy fusion-align trains FusionAlignNet (streaming pairwise \
         alignment) on the SAME --data scenes as --strategy align, one \
         reference/frame pair per training step; --corr-radius applies to it \
         too (FusionAlignNetConfig::corr_radius), but --real-data / \
         --photometric-weight (the §5.2 fine-tune phase) do not — see \
         docs/fusionalign-design.md."
    );
}

// ---------------------------------------------------------------------------
// Checkpointing & resume
// ---------------------------------------------------------------------------

/// Resumable training state, persisted next to the rolling checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainState {
    /// Epoch to resume from (the next epoch to run).
    epoch: usize,
    /// Global optimiser step count reached so far.
    global_step: usize,
    /// Preset the in-progress run was started with (sanity check on resume).
    size: ModelSize,
    /// The strategy used for the training run (see [`TrainStrategy::as_str`]).
    strategy: String,
}

/// Save weights (`<stem>.mpk`) plus a sidecar `manifest` (`<stem>.json`) so
/// the architecture can be reconstructed by the discovery loader.
///
/// `manifest` is passed in by the caller rather than built internally via
/// [`ModelManifest::from_size`] — that constructor always stamps
/// [`stacker_nn::discovery::FOCUSMERGE_V1`], which would mislabel `batch`/
/// `align` checkpoints. Each training loop below builds the correct manifest
/// for its own strategy (`ModelManifest::from_size` / `from_size_batch` /
/// `from_size_align`).
fn save_checkpoint<M: Module<B>>(
    model: &M,
    dir: &Path,
    stem: &str,
    manifest: &ModelManifest,
) -> Result<(), Box<dyn Error>> {
    let base = dir.join(stem);
    model
        .clone()
        .save_file(base.clone(), &CompactRecorder::new())?;
    std::fs::write(
        base.with_extension("json"),
        serde_json::to_string_pretty(manifest)?,
    )?;
    Ok(())
}

/// Persist the rolling crash-recovery checkpoint plus the resumable state.
fn save_latest<M: Module<B>>(
    model: &M,
    dir: &Path,
    size: ModelSize,
    strategy: TrainStrategy,
    manifest: &ModelManifest,
    epoch: usize,
    global_step: usize,
) -> Result<(), Box<dyn Error>> {
    save_checkpoint(
        model,
        dir,
        &format!("{}_latest", strategy.stem_prefix()),
        manifest,
    )?;
    let state = TrainState {
        epoch,
        global_step,
        size,
        strategy: strategy.as_str().to_owned(),
    };
    std::fs::write(dir.join(STATE_FILE), serde_json::to_string_pretty(&state)?)?;
    Ok(())
}

/// Load resumable state from `<dir>/train_state.json`, if present.
fn load_state(dir: &Path) -> Option<TrainState> {
    let bytes = std::fs::read(dir.join(STATE_FILE)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Persist the optimiser state (`AdamW` moments) so a resumed run continues with
/// momentum/variance intact rather than re-warming from zero. Generic over the
/// optimiser type to avoid naming the concrete adaptor.
fn save_optim<M: AutodiffModule<B>, O: Optimizer<M, B>>(
    optim: &O,
    dir: &Path,
    strategy: TrainStrategy,
) -> Result<(), Box<dyn Error>> {
    CompactRecorder::new().record(
        optim.to_record(),
        dir.join(format!("{}_latest.optim", strategy.stem_prefix())),
    )?;
    Ok(())
}

/// Load optimiser state from the rolling checkpoint if it exists, returning the
/// (possibly restored) optimiser. A missing file leaves `optim` unchanged.
// `device` stays by-reference: GPU device handles (cuda/wgpu) are not `Copy`, so
// taking them by value would move the caller's device.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn load_optim<M: AutodiffModule<B>, O: Optimizer<M, B>>(
    optim: O,
    dir: &Path,
    device: &burn::prelude::Device<B>,
    strategy: TrainStrategy,
) -> Result<O, Box<dyn Error>> {
    let base = dir.join(format!("{}_latest.optim", strategy.stem_prefix()));
    if base.with_extension("mpk").exists() {
        let record = CompactRecorder::new().load(base, device)?;
        Ok(optim.load_record(record))
    } else {
        Ok(optim)
    }
}

// ---------------------------------------------------------------------------
// Tiny dependency-free RNG (xorshift64*) for shuffling / cropping / sampling.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        // Avoid the all-zero state, which is a fixed point of xorshift.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f64 in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `0..n` (returns 0 if `n == 0`).
    const fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Fisher–Yates shuffle.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.below(i + 1);
            slice.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Scene discovery
// ---------------------------------------------------------------------------

/// Scan `root` for sub-directories containing a `metadata.json`, as
/// [`RolloutSequence`]s (pairwise/batch fusion scenes).
fn discover_scenes(root: &Path) -> Result<Vec<RolloutSequence>, Box<dyn Error>> {
    let mut scenes = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() && path.join("metadata.json").exists() {
            scenes.push(RolloutSequence::new(path)?);
        }
    }
    scenes.sort_by(|a, b| a.scene_dir.cmp(&b.scene_dir));
    Ok(scenes)
}

/// Scan `root` for sub-directories containing a `metadata.json`, as
/// [`AlignSequence`]s (alignment scenes, i.e. `03_simulate_misalignment.py` output).
fn discover_align_scenes(root: &Path) -> Result<Vec<AlignSequence>, Box<dyn Error>> {
    let mut scenes = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() && path.join("metadata.json").exists() {
            scenes.push(AlignSequence::new(path)?);
        }
    }
    scenes.sort_by(|a, b| a.scene_dir.cmp(&b.scene_dir));
    Ok(scenes)
}

/// Choose a random square crop that fits inside `(w, h)`.
fn random_crop_dims(w: usize, h: usize, max_size: usize, rng: &mut Rng) -> CropParams {
    let size = max_size.min(w).min(h);
    let top = rng.below(h - size + 1);
    let left = rng.below(w - size + 1);
    CropParams { top, left, size }
}

/// Choose a random square crop that fits inside `scene`'s frames.
fn random_crop(
    scene: &RolloutSequence,
    max_size: usize,
    rng: &mut Rng,
) -> Result<CropParams, Box<dyn Error>> {
    let first = scene.scene_dir.join(&scene.meta.stack[0]);
    let (w, h) = image::image_dimensions(&first)?;
    Ok(random_crop_dims(w as usize, h as usize, max_size, rng))
}

/// Choose a random square crop that fits inside `scene`'s frames.
fn random_crop_align(
    scene: &AlignSequence,
    max_size: usize,
    rng: &mut Rng,
) -> Result<CropParams, Box<dyn Error>> {
    let first = scene.scene_dir.join(&scene.meta.stack[0]);
    let (w, h) = image::image_dimensions(&first)?;
    Ok(random_crop_dims(w as usize, h as usize, max_size, rng))
}

// ---------------------------------------------------------------------------
// Resume helpers (shared validation)
// ---------------------------------------------------------------------------

/// Validate a loaded [`TrainState`] against the current run's `--size` /
/// `--strategy`, returning `(start_epoch, global_step)`.
fn validate_resume_state(
    state: &TrainState,
    size: ModelSize,
    strategy: TrainStrategy,
) -> Result<(usize, usize), Box<dyn Error>> {
    if state.size != size {
        return Err(format!(
            "resume size mismatch: checkpoint is '{}', --size is '{}'",
            state.size.as_str(),
            size.as_str()
        )
        .into());
    }
    if state.strategy != strategy.as_str() {
        return Err(format!(
            "resume strategy mismatch: checkpoint is '{}', --strategy is '{}'",
            state.strategy,
            strategy.as_str()
        )
        .into());
    }
    Ok((state.epoch, state.global_step))
}

// ---------------------------------------------------------------------------
// Training loop (Pairwise)
// ---------------------------------------------------------------------------

// `device` stays by-reference: GPU device handles (cuda/wgpu) are not `Copy`,
// so taking them by value would move the caller's device (see `load_optim`'s
// identical justification above).
// `too_many_lines`: the recurrent rollout + loss aggregation + scheduled
// sampling + LR schedule + checkpointing sequence is cohesive top-level
// training-loop control flow; splitting it would scatter state that must be
// read/updated together across artificial function boundaries.
#[allow(clippy::trivially_copy_pass_by_ref, clippy::too_many_lines)]
fn train_pairwise(
    args: &Args,
    scenes: &[RolloutSequence],
    device: &burn::prelude::Device<B>,
) -> Result<(), Box<dyn Error>> {
    let strategy = TrainStrategy::Pairwise;
    let config = FocusMergeNetConfig::from_size(args.size);

    let mut start_epoch = 0usize;
    let mut global_step = 0usize;
    let latest_mpk = args
        .out
        .join(format!("{}_latest.mpk", strategy.stem_prefix()));
    let mut model = if args.resume && latest_mpk.exists() {
        if let Some(state) = load_state(&args.out) {
            (start_epoch, global_step) = validate_resume_state(&state, args.size, strategy)?;
        }
        println!(
            "Resuming from {} at epoch {start_epoch}, step {global_step}",
            latest_mpk.display()
        );
        let record = CompactRecorder::new().load(latest_mpk, device)?;
        let m = config.init::<B>(device);
        m.load_record(record)
    } else {
        config.init::<B>(device)
    };

    let loss = FocusFusionLossConfig::new().init();
    let mut optim = AdamWConfig::new().init();
    if args.resume {
        optim = load_optim(optim, &args.out, device, strategy)?;
    }

    let total_steps = args.epochs * scenes.len();
    let warmup = (total_steps / 20).max(1);
    let mut rng = Rng::new(args.seed.wrapping_add(global_step as u64));
    let manifest = ModelManifest::from_size(args.size);

    for epoch in start_epoch..args.epochs {
        let ss_prob = scheduled_sampling_prob(epoch, args.epochs, args.ss_max);
        let mut order: Vec<usize> = (0..scenes.len()).collect();
        rng.shuffle(&mut order);

        let mut epoch_loss = 0.0_f64;
        let mut counted = 0usize;

        for &si in &order {
            let crop = random_crop(&scenes[si], args.crop, &mut rng)?;
            let n_steps = scenes[si].n_steps;

            let mut steps: Vec<MergeSample<B>> = Vec::with_capacity(n_steps);
            for k in 0..n_steps {
                steps.push(scenes[si].get_step::<B>(k, device, Some(crop))?);
            }

            let use_prediction: Vec<bool> = (0..n_steps)
                .map(|k| k > 0 && rng.next_f64() < ss_prob)
                .collect();
            let terms = rollout_loss(&model, &loss, &steps, &use_prediction);
            let loss_val: f64 = terms.clone().into_scalar().elem();

            let lr = cosine_lr(args.lr, global_step, total_steps, warmup);
            let grads = terms.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            epoch_loss += loss_val;
            counted += 1;
            global_step += 1;

            if global_step.is_multiple_of(50) {
                println!(
                    "  step {global_step:>6}  epoch {epoch:>3}  lr {lr:.2e}  loss {loss_val:.5}  ss_p {ss_prob:.3}"
                );
            }

            if global_step.is_multiple_of(args.ckpt_every) {
                save_latest(
                    &model,
                    &args.out,
                    args.size,
                    strategy,
                    &manifest,
                    epoch,
                    global_step,
                )?;
                save_optim(&optim, &args.out, strategy)?;
            }
        }

        let mean = if counted == 0 {
            f64::NAN
        } else {
            epoch_loss / counted as f64
        };
        println!("epoch {epoch:>3}/{}  mean_loss {mean:.5}", args.epochs);

        save_checkpoint(
            &model,
            &args.out,
            &format!("{}_epoch_{epoch:03}", strategy.stem_prefix()),
            &manifest,
        )?;
        save_latest(
            &model,
            &args.out,
            args.size,
            strategy,
            &manifest,
            epoch + 1,
            global_step,
        )?;
        save_optim(&optim, &args.out, strategy)?;
    }

    let final_stem = format!("{}_final", strategy.stem_prefix());
    save_checkpoint(&model, &args.out, &final_stem, &manifest)?;
    println!(
        "Training complete. Final weights: {}.mpk",
        args.out.join(final_stem).display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Training loop (Batch)
// ---------------------------------------------------------------------------

// `device` stays by-reference: GPU device handles (cuda/wgpu) are not `Copy`,
// so taking them by value would move the caller's device (see `load_optim`'s
// identical justification above).
#[allow(clippy::trivially_copy_pass_by_ref)]
fn train_batch(
    args: &Args,
    scenes: &[RolloutSequence],
    device: &burn::prelude::Device<B>,
) -> Result<(), Box<dyn Error>> {
    let strategy = TrainStrategy::Batch;
    let config = BatchMergeNetConfig::from_size(args.size);

    let mut start_epoch = 0usize;
    let mut global_step = 0usize;
    let latest_mpk = args
        .out
        .join(format!("{}_latest.mpk", strategy.stem_prefix()));
    let mut model = if args.resume && latest_mpk.exists() {
        if let Some(state) = load_state(&args.out) {
            (start_epoch, global_step) = validate_resume_state(&state, args.size, strategy)?;
        }
        println!(
            "Resuming from {} at epoch {start_epoch}, step {global_step}",
            latest_mpk.display()
        );
        let record = CompactRecorder::new().load(latest_mpk, device)?;
        let m = config.init::<B>(device);
        m.load_record(record)
    } else {
        config.init::<B>(device)
    };

    let loss = FocusBatchLossConfig::new().init();
    let mut optim = AdamWConfig::new().init();
    if args.resume {
        optim = load_optim(optim, &args.out, device, strategy)?;
    }

    let total_steps = args.epochs * scenes.len();
    let warmup = (total_steps / 20).max(1);
    let mut rng = Rng::new(args.seed.wrapping_add(global_step as u64));
    let manifest = ModelManifest::from_size_batch(args.size);

    for epoch in start_epoch..args.epochs {
        let mut order: Vec<usize> = (0..scenes.len()).collect();
        rng.shuffle(&mut order);

        let mut epoch_loss = 0.0_f64;
        let mut counted = 0usize;

        for &si in &order {
            let crop = random_crop(&scenes[si], args.crop, &mut rng)?;
            let sample = scenes[si].get_batch::<B>(device, Some(crop))?;

            let terms = batch_loss(&model, &loss, &sample);
            let loss_val: f64 = terms.clone().into_scalar().elem();

            let lr = cosine_lr(args.lr, global_step, total_steps, warmup);
            let grads = terms.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            epoch_loss += loss_val;
            counted += 1;
            global_step += 1;

            if global_step.is_multiple_of(50) {
                println!(
                    "  step {global_step:>6}  epoch {epoch:>3}  lr {lr:.2e}  loss {loss_val:.5}"
                );
            }

            if global_step.is_multiple_of(args.ckpt_every) {
                save_latest(
                    &model,
                    &args.out,
                    args.size,
                    strategy,
                    &manifest,
                    epoch,
                    global_step,
                )?;
                save_optim(&optim, &args.out, strategy)?;
            }
        }

        let mean = if counted == 0 {
            f64::NAN
        } else {
            epoch_loss / counted as f64
        };
        println!("epoch {epoch:>3}/{}  mean_loss {mean:.5}", args.epochs);

        save_checkpoint(
            &model,
            &args.out,
            &format!("{}_epoch_{epoch:03}", strategy.stem_prefix()),
            &manifest,
        )?;
        save_latest(
            &model,
            &args.out,
            args.size,
            strategy,
            &manifest,
            epoch + 1,
            global_step,
        )?;
        save_optim(&optim, &args.out, strategy)?;
    }

    let final_stem = format!("{}_final", strategy.stem_prefix());
    save_checkpoint(&model, &args.out, &final_stem, &manifest)?;
    println!(
        "Training complete. Final weights: {}.mpk",
        args.out.join(final_stem).display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Training loop (Align)
// ---------------------------------------------------------------------------

/// Choose a random square crop that fits inside a [`RealStackScene`]'s frames.
fn random_crop_real(
    scene: &RealStackScene,
    max_size: usize,
    rng: &mut Rng,
) -> Result<CropParams, Box<dyn Error>> {
    let (w, h) = image::image_dimensions(&scene.frames[0])?;
    Ok(random_crop_dims(w as usize, h as usize, max_size, rng))
}

/// §5.2 photometric fine-tune step: pick a random real scene, run the model
/// on it, and add `photometric_weight * mean_over_non_reference_frames(
/// photometric_gradient_loss(frame_i, reference, matrix_i))` to `terms`.
///
/// Kept as a small standalone helper (rather than inlined into
/// `train_align`'s loop) so the "mix corner + photometric" composition in
/// §5.2 is legible as one unit, and so the skeletal nature of this phase
/// (see the module docs' "Fine-tune phase" note) is easy to audit.
///
/// # Panics
///
/// Panics if `scene` has fewer than 2 frames (the caller is expected to have
/// filtered these out via [`RealStackScene::len`]).
// `device` stays by-reference: GPU device handles (cuda/wgpu) are not `Copy`,
// so taking them by value would move the caller's device (see `load_optim`'s
// identical justification above) — `train_align`'s loop calls this function
// once per fine-tune step with the same shared `&<B as Backend>::Device`.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn add_photometric_term(
    terms: burn::tensor::Tensor<B, 1>,
    model: &stacker_nn::model::BatchAlignNet<B>,
    scene: &RealStackScene,
    args: &Args,
    device: &burn::prelude::Device<B>,
    rng: &mut Rng,
) -> Result<burn::tensor::Tensor<B, 1>, Box<dyn Error>> {
    assert!(scene.len() >= 2, "real-data scene must have >= 2 frames");

    let crop = random_crop_real(scene, args.crop, rng)?;
    let stack = scene.get::<B>(device, Some(crop))?; // [S, 3, H, W]
    let [s, c, h, w] = stack.dims();
    let stack_5d = stack.clone().unsqueeze_dim::<5>(0); // [1, S, 3, H, W]

    let (matrices, _raw) = model.forward_with_params(stack_5d); // [1, S, 3, 3]
    let matrices = matrices.squeeze_dim::<3>(0); // [S, 3, 3]

    let reference = stack
        .clone()
        .slice([0..1, 0..c, 0..h, 0..w])
        .reshape([1, c, h, w]);

    let mut photo_acc: Option<burn::tensor::Tensor<B, 1>> = None;
    for i in 1..s {
        let frame_i = stack
            .clone()
            .slice([i..i + 1, 0..c, 0..h, 0..w])
            .reshape([1, c, h, w]);
        let matrix_i = matrices.clone().slice([i..i + 1, 0..3, 0..3]);
        let term = photometric_gradient_loss(frame_i, reference.clone(), matrix_i);
        photo_acc = Some(match photo_acc {
            None => term,
            Some(acc) => acc.add(term),
        });
    }
    let Some(photo_sum) = photo_acc else {
        return Ok(terms); // s == 1: no non-reference frame to score (shouldn't happen, guarded above).
    };
    let photo_mean = photo_sum.div_scalar((s - 1) as f32);
    Ok(terms.add(photo_mean.mul_scalar(args.photometric_weight as f32)))
}

// `device` stays by-reference: GPU device handles (cuda/wgpu) are not `Copy`,
// so taking them by value would move the caller's device (see `load_optim`'s
// identical justification above).
//
// ## Fine-tune phase (§5.2) — SKELETAL, flagged explicitly
//
// The last 25% of epochs mix in [`photometric_gradient_loss`] on real,
// unlabelled scenes when `--real-data <dir>` is given, per
// `docs/batchalign-v2-design.md` §5.2/§5.3. This wiring is a CLEARLY
// STRUCTURED SKELETON in the sense the task's scope explicitly allows for
// this phase: it draws ONE random real scene per synthetic step during the
// fine-tune window and adds its photometric term to that step's loss —
// correct and functional, but unoptimised (no separate real-data batch size,
// no real-data-only epoch pass, no independent LR/schedule for the
// photometric term beyond the fixed `--photometric-weight` mix, and no
// eval-time reporting split between the two loss components). A full
// production fine-tuning setup (separate data loader threads, independent
// mixing schedules, per-term logging) is future work — see the design doc's
// work-plan step 8, explicitly listed as needing a GPU training run to
// validate, which is out of scope here.
//
// `device` stays by-reference: GPU device handles (cuda/wgpu) are not
// `Copy`, so taking them by value would move the caller's device (see
// `load_optim`'s identical justification above).
// `too_many_lines`: cohesive top-level training-loop control flow (rollout,
// loss aggregation, scheduled sampling, §5.2 photometric mixing, LR
// schedule, checkpointing) — splitting it would scatter state that must be
// read/updated together across artificial function boundaries.
#[allow(clippy::trivially_copy_pass_by_ref, clippy::too_many_lines)]
fn train_align(
    args: &Args,
    scenes: &[AlignSequence],
    device: &burn::prelude::Device<B>,
) -> Result<(), Box<dyn Error>> {
    let strategy = TrainStrategy::Align;
    let mut config = BatchAlignNetConfig::from_size(args.size);
    if let Some(r) = args.corr_radius {
        config = config.with_corr_radius(r);
    }

    let mut start_epoch = 0usize;
    let mut global_step = 0usize;
    let latest_mpk = args
        .out
        .join(format!("{}_latest.mpk", strategy.stem_prefix()));
    let mut model = if args.resume && latest_mpk.exists() {
        if let Some(state) = load_state(&args.out) {
            (start_epoch, global_step) = validate_resume_state(&state, args.size, strategy)?;
        }
        println!(
            "Resuming from {} at epoch {start_epoch}, step {global_step}",
            latest_mpk.display()
        );
        let record = CompactRecorder::new().load(latest_mpk, device)?;
        let m = config.init::<B>(device);
        m.load_record(record)
    } else {
        config.init::<B>(device)
    };

    let loss = BatchAlignmentLossConfig::new().init();
    let mut optim = AdamWConfig::new().init();
    if args.resume {
        optim = load_optim(optim, &args.out, device, strategy)?;
    }

    let total_steps = args.epochs * scenes.len();
    let warmup = (total_steps / 20).max(1);
    let mut rng = Rng::new(args.seed.wrapping_add(global_step as u64));
    let manifest = ModelManifest::from_size_align(args.size);

    // §5.2 fine-tune phase: only scenes with >= 2 frames are usable (a
    // photometric term needs at least one non-reference frame); the last
    // 25% of epochs mix it in when `--real-data` scenes are found.
    let real_scenes: Vec<RealStackScene> = match &args.real_data {
        Some(dir) => discover_real_stacks(dir)?
            .into_iter()
            .filter(|s| s.len() >= 2)
            .collect(),
        None => Vec::new(),
    };
    let finetune_start_epoch = args.epochs - args.epochs / 4; // last 25%
    if !real_scenes.is_empty() {
        println!(
            "Discovered {} real (unlabelled) scene(s) under {} — photometric \
             fine-tuning enabled for epochs {finetune_start_epoch}..{}",
            real_scenes.len(),
            args.real_data
                .as_ref()
                .expect("real_scenes non-empty implies Some")
                .display(),
            args.epochs
        );
    }

    for epoch in start_epoch..args.epochs {
        let mut order: Vec<usize> = (0..scenes.len()).collect();
        rng.shuffle(&mut order);
        let finetune_active = !real_scenes.is_empty() && epoch >= finetune_start_epoch;

        let mut epoch_loss = 0.0_f64;
        let mut counted = 0usize;

        for &si in &order {
            let crop = random_crop_align(&scenes[si], args.crop, &mut rng)?;
            let sample = scenes[si].get::<B>(device, Some(crop))?;

            let mut terms = align_loss(&model, &loss, &sample);
            if finetune_active {
                let ri = rng.below(real_scenes.len());
                terms =
                    add_photometric_term(terms, &model, &real_scenes[ri], args, device, &mut rng)?;
            }
            let loss_val: f64 = terms.clone().into_scalar().elem();

            let lr = cosine_lr(args.lr, global_step, total_steps, warmup);
            let grads = terms.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            epoch_loss += loss_val;
            counted += 1;
            global_step += 1;

            if global_step.is_multiple_of(50) {
                let tag = if finetune_active { " [finetune]" } else { "" };
                println!(
                    "  step {global_step:>6}  epoch {epoch:>3}  lr {lr:.2e}  loss {loss_val:.5}{tag}"
                );
            }

            if global_step.is_multiple_of(args.ckpt_every) {
                save_latest(
                    &model,
                    &args.out,
                    args.size,
                    strategy,
                    &manifest,
                    epoch,
                    global_step,
                )?;
                save_optim(&optim, &args.out, strategy)?;
            }
        }

        let mean = if counted == 0 {
            f64::NAN
        } else {
            epoch_loss / counted as f64
        };
        println!("epoch {epoch:>3}/{}  mean_loss {mean:.5}", args.epochs);

        save_checkpoint(
            &model,
            &args.out,
            &format!("{}_epoch_{epoch:03}", strategy.stem_prefix()),
            &manifest,
        )?;
        save_latest(
            &model,
            &args.out,
            args.size,
            strategy,
            &manifest,
            epoch + 1,
            global_step,
        )?;
        save_optim(&optim, &args.out, strategy)?;
    }

    let final_stem = format!("{}_final", strategy.stem_prefix());
    save_checkpoint(&model, &args.out, &final_stem, &manifest)?;
    println!(
        "Training complete. Final weights: {}.mpk",
        args.out.join(final_stem).display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Training loop (FusionAlign — streaming pairwise alignment)
// ---------------------------------------------------------------------------

/// One training example for `--strategy fusion-align`: which scene and which
/// `frame_index` within it (the reference is always that scene's frame 0 —
/// see [`AlignSequence::get_pair`]).
#[derive(Debug, Clone, Copy)]
struct PairExample {
    scene_idx: usize,
    frame_index: usize,
}

/// Enumerate every (scene, frame) pair across `scenes`, skipping
/// single-frame scenes (frame 0 registered against itself is a legitimate
/// sample per [`AlignSequence::get_pair`]'s docs, so single-plane scenes
/// still contribute exactly one example — the `frame_index = 0` self-pair —
/// rather than being dropped entirely).
fn enumerate_pairs(scenes: &[AlignSequence]) -> Vec<PairExample> {
    let mut examples = Vec::new();
    for (scene_idx, scene) in scenes.iter().enumerate() {
        for frame_index in 0..scene.meta.n_planes {
            examples.push(PairExample {
                scene_idx,
                frame_index,
            });
        }
    }
    examples
}

/// Trains [`stacker_nn::model::FusionAlignNet`] on the SAME scene data as
/// `train_align` ([`AlignSequence`]/`03_simulate_misalignment.py` output),
/// but each optimiser step is ONE reference/frame pair
/// ([`AlignSequence::get_pair`]) rather than a whole stack — see
/// `docs/fusionalign-design.md` for why. Unlike `train_align`, there is no
/// §5.2 photometric fine-tune phase here: that phase's rationale
/// (unsupervised refinement using the model's OWN predicted matrices warped
/// against real, unlabelled multi-frame stacks) is unchanged in principle
/// for the pairwise architecture, but wiring it up is left as future work —
/// see `docs/fusionalign-design.md`'s "future work" section — so
/// `--real-data`/`--photometric-weight` are silently ignored for this
/// strategy (documented in `print_usage`).
// `device` stays by-reference: GPU device handles (cuda/wgpu) are not `Copy`,
// so taking them by value would move the caller's device (see `load_optim`'s
// identical justification above).
// `too_many_lines`: cohesive top-level training-loop control flow, mirroring
// `train_align`'s identical justification above (pairwise variant, same
// rollout/loss/schedule/checkpointing sequence).
#[allow(clippy::trivially_copy_pass_by_ref, clippy::too_many_lines)]
fn train_fusion_align(
    args: &Args,
    scenes: &[AlignSequence],
    device: &burn::prelude::Device<B>,
) -> Result<(), Box<dyn Error>> {
    let strategy = TrainStrategy::FusionAlign;
    let mut config = FusionAlignNetConfig::from_size(args.size);
    if let Some(r) = args.corr_radius {
        config = config.with_corr_radius(r);
    }

    let mut start_epoch = 0usize;
    let mut global_step = 0usize;
    let latest_mpk = args
        .out
        .join(format!("{}_latest.mpk", strategy.stem_prefix()));
    let mut model = if args.resume && latest_mpk.exists() {
        if let Some(state) = load_state(&args.out) {
            (start_epoch, global_step) = validate_resume_state(&state, args.size, strategy)?;
        }
        println!(
            "Resuming from {} at epoch {start_epoch}, step {global_step}",
            latest_mpk.display()
        );
        let record = CompactRecorder::new().load(latest_mpk, device)?;
        let m = config.init::<B>(device);
        m.load_record(record)
    } else {
        config.init::<B>(device)
    };

    let loss = PairAlignmentLossConfig::new().init();
    let mut optim = AdamWConfig::new().init();
    if args.resume {
        optim = load_optim(optim, &args.out, device, strategy)?;
    }

    let examples = enumerate_pairs(scenes);
    if examples.is_empty() {
        return Err("no alignment scenes contain any frames to pair".into());
    }

    let total_steps = args.epochs * examples.len();
    let warmup = (total_steps / 20).max(1);
    let mut rng = Rng::new(args.seed.wrapping_add(global_step as u64));
    let manifest = ModelManifest::from_size_fusion_align(args.size);

    for epoch in start_epoch..args.epochs {
        let mut order: Vec<usize> = (0..examples.len()).collect();
        rng.shuffle(&mut order);

        let mut epoch_loss = 0.0_f64;
        let mut counted = 0usize;

        for &ei in &order {
            let example = examples[ei];
            let scene = &scenes[example.scene_idx];
            let crop = random_crop_align(scene, args.crop, &mut rng)?;
            let sample = scene.get_pair::<B>(example.frame_index, device, Some(crop))?;

            let terms = fusion_align_loss(&model, &loss, &sample);
            let loss_val: f64 = terms.clone().into_scalar().elem();

            let lr = cosine_lr(args.lr, global_step, total_steps, warmup);
            let grads = terms.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            epoch_loss += loss_val;
            counted += 1;
            global_step += 1;

            if global_step.is_multiple_of(50) {
                println!(
                    "  step {global_step:>6}  epoch {epoch:>3}  lr {lr:.2e}  loss {loss_val:.5}"
                );
            }

            if global_step.is_multiple_of(args.ckpt_every) {
                save_latest(
                    &model,
                    &args.out,
                    args.size,
                    strategy,
                    &manifest,
                    epoch,
                    global_step,
                )?;
                save_optim(&optim, &args.out, strategy)?;
            }
        }

        let mean = if counted == 0 {
            f64::NAN
        } else {
            epoch_loss / counted as f64
        };
        println!("epoch {epoch:>3}/{}  mean_loss {mean:.5}", args.epochs);

        save_checkpoint(
            &model,
            &args.out,
            &format!("{}_epoch_{epoch:03}", strategy.stem_prefix()),
            &manifest,
        )?;
        save_latest(
            &model,
            &args.out,
            args.size,
            strategy,
            &manifest,
            epoch + 1,
            global_step,
        )?;
        save_optim(&optim, &args.out, strategy)?;
    }

    let final_stem = format!("{}_final", strategy.stem_prefix());
    save_checkpoint(&model, &args.out, &final_stem, &manifest)?;
    println!(
        "Training complete. Final weights: {}.mpk",
        args.out.join(final_stem).display()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    std::fs::create_dir_all(&args.out)?;

    let device = burn::prelude::Device::<B>::default();

    match args.strategy {
        TrainStrategy::Pairwise | TrainStrategy::Batch => {
            let scenes = discover_scenes(&args.data)?;
            if scenes.is_empty() {
                return Err(format!(
                    "no scenes (dirs with metadata.json) found under {}",
                    args.data.display()
                )
                .into());
            }
            println!(
                "Discovered {} scene(s) under {}",
                scenes.len(),
                args.data.display()
            );
            match args.strategy {
                TrainStrategy::Pairwise => train_pairwise(&args, &scenes, &device)?,
                TrainStrategy::Batch => train_batch(&args, &scenes, &device)?,
                TrainStrategy::Align | TrainStrategy::FusionAlign => unreachable!(),
            }
        }
        TrainStrategy::Align | TrainStrategy::FusionAlign => {
            let scenes = discover_align_scenes(&args.data)?;
            if scenes.is_empty() {
                return Err(format!(
                    "no scenes (dirs with metadata.json) found under {}",
                    args.data.display()
                )
                .into());
            }
            println!(
                "Discovered {} alignment scene(s) under {}",
                scenes.len(),
                args.data.display()
            );
            match args.strategy {
                TrainStrategy::Align => train_align(&args, &scenes, &device)?,
                TrainStrategy::FusionAlign => train_fusion_align(&args, &scenes, &device)?,
                TrainStrategy::Pairwise | TrainStrategy::Batch => unreachable!(),
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests for pure helpers
// ---------------------------------------------------------------------------
//
// Only the pure, cheaply-testable helpers (`TrainStrategy` parsing/formatting
// and `random_crop_dims`'s bounds) are covered here. The checkpoint
// save/load/resume functions are file-IO-heavy and exercised indirectly by
// CI training smoke-runs, not unit tests in this module.
#[cfg(test)]
mod tests {
    use super::{Rng, TrainStrategy, enumerate_pairs, random_crop_dims};

    #[test]
    fn train_strategy_parse_round_trips() {
        for (s, expected) in [
            ("pairwise", TrainStrategy::Pairwise),
            ("batch", TrainStrategy::Batch),
            ("align", TrainStrategy::Align),
            ("fusion-align", TrainStrategy::FusionAlign),
        ] {
            let parsed = TrainStrategy::parse(s).unwrap_or_else(|| panic!("failed to parse {s}"));
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), s);
        }
        assert_eq!(TrainStrategy::parse("nonsense"), None);
    }

    #[test]
    fn train_strategy_stem_prefix_matches_checkpoint_naming_table() {
        // See the module docs' "Checkpoint naming" table.
        assert_eq!(TrainStrategy::Pairwise.stem_prefix(), "focusmerge");
        assert_eq!(TrainStrategy::Batch.stem_prefix(), "batchmerge");
        assert_eq!(TrainStrategy::Align.stem_prefix(), "batchalign");
        assert_eq!(TrainStrategy::FusionAlign.stem_prefix(), "fusionalign");
    }

    #[test]
    fn random_crop_dims_fits_within_bounds() {
        let mut rng = Rng::new(1234);
        for _ in 0..50 {
            let cp = random_crop_dims(100, 80, 32, &mut rng);
            assert_eq!(cp.size, 32, "crop size should equal max_size when it fits");
            assert!(cp.top + cp.size <= 80, "crop exceeds height: {cp:?}");
            assert!(cp.left + cp.size <= 100, "crop exceeds width: {cp:?}");
        }
    }

    #[test]
    fn random_crop_dims_clamps_to_smaller_dimension() {
        let mut rng = Rng::new(5678);
        // max_size larger than both w and h: clamps to the smaller dimension.
        let cp = random_crop_dims(20, 15, 256, &mut rng);
        assert_eq!(cp.size, 15, "crop should clamp to min(w, h, max_size)");
        assert_eq!(
            cp.top, 0,
            "single valid crop position (size == h) must be top=0"
        );
        assert!(cp.left + cp.size <= 20);
    }

    /// Write a minimal `AlignSequence`-compatible scene (see
    /// `stacker_nn::data::AlignSceneMeta`'s schema) with `n_planes` frames,
    /// all identity ground truth (content doesn't matter for
    /// `enumerate_pairs`, which only reads `meta.n_planes`).
    fn write_align_scene(dir: &std::path::Path, n_planes: usize) {
        use image::RgbImage;
        let (w, h) = (4u32, 4u32);
        let mut stack = Vec::with_capacity(n_planes);
        let mut alignment_gt = Vec::with_capacity(n_planes);
        #[rustfmt::skip]
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..n_planes {
            let name = format!("frame_{i:03}.png");
            let img = RgbImage::from_fn(w, h, |_, _| image::Rgb([100, 100, 100]));
            img.save(dir.join(&name)).expect("write png");
            stack.push(name);
            alignment_gt.push(identity);
        }
        let meta = serde_json::json!({
            "n_planes": n_planes,
            "stack": stack,
            "alignment_gt": alignment_gt,
            "cropped_dims": [w, h],
        });
        std::fs::write(
            dir.join("metadata.json"),
            serde_json::to_string_pretty(&meta).unwrap(),
        )
        .unwrap();
    }

    /// [`enumerate_pairs`] must produce exactly `n_planes` examples per
    /// scene (`frame_index` covering `0..n_planes`, reference always
    /// implicit as frame 0), including a single example for a one-frame
    /// scene (the legitimate self-pair — see [`enumerate_pairs`]'s docs).
    #[test]
    fn enumerate_pairs_covers_every_frame_per_scene() {
        let tmp = tempfile::tempdir().unwrap();
        let scene_a = tmp.path().join("scene_a");
        let scene_b = tmp.path().join("scene_b");
        std::fs::create_dir(&scene_a).unwrap();
        std::fs::create_dir(&scene_b).unwrap();
        write_align_scene(&scene_a, 3);
        write_align_scene(&scene_b, 1);

        let scenes = vec![
            stacker_nn::data::AlignSequence::new(scene_a).unwrap(),
            stacker_nn::data::AlignSequence::new(scene_b).unwrap(),
        ];

        let examples = enumerate_pairs(&scenes);
        assert_eq!(examples.len(), 4, "3 + 1 frames across the two scenes");

        let scene0_frames: Vec<usize> = examples
            .iter()
            .filter(|e| e.scene_idx == 0)
            .map(|e| e.frame_index)
            .collect();
        assert_eq!(scene0_frames, vec![0, 1, 2]);

        let scene1_frames: Vec<usize> = examples
            .iter()
            .filter(|e| e.scene_idx == 1)
            .map(|e| e.frame_index)
            .collect();
        assert_eq!(
            scene1_frames,
            vec![0],
            "single-frame scene contributes exactly the frame-0 self-pair"
        );
    }
}
