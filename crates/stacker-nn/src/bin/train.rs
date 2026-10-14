//! `z-stackr-train` — training driver for the networks of `stacker-nn`.
//!
//! This binary only requires the `autodiff` backend (in the default feature
//! set). `burn::optim` (`AdamW`) is available because `burn/std` is enabled by
//! every backend feature.
//!
//! The differentiable logic (the recurrent rollout, the batch, align and
//! fusion-align losses, scheduled sampling, LR schedule) lives in
//! [`stacker_nn::train`], the command line and [`TrainStrategy`] in
//! [`stacker_nn::train::args`], and checkpoint writing and the resume state in
//! [`stacker_nn::checkpoint`] (pairwise) and
//! [`stacker_nn::train::strategy_checkpoint`] (the other strategies): atomic
//! writes, distinct model and optimiser files, full-precision rolling recorder.
//! This file is only the optimiser loops and plumbing.
//!
//! ## Usage
//!
//! ```text
//! # Default CPU (ndarray) autodiff backend, pairwise merge:
//! cargo run -p z-stackr-nn --release --bin z-stackr-train -- \
//!     --data /path/to/scenes --out checkpoints --epochs 60 --lr 3e-4 --crop 256
//!
//! # The other strategies (see --help for the data each one reads):
//! ... --strategy batch --data /path/to/fusion_scenes
//! ... --strategy align --data /path/to/misaligned_scenes [--corr-radius 4]
//! ... --strategy fusion-align --data /path/to/misaligned_scenes
//!
//! # GPU (wgpu) backend:
//! cargo run -p z-stackr-nn --release --no-default-features \
//!     --features "wgpu,autodiff" --bin z-stackr-train -- --data /path/to/scenes
//!
//! # Continue an interrupted run (same --out, --strategy, --size, --epochs and scene set):
//! ... --out checkpoints --resume
//! ```
//!
//! Each scene is decoded once per optimiser step (or not at all on a cache hit):
//! `--cache-mb N` (default 1024) bounds the decoded scene pixels kept between
//! steps, `0` keeps nothing. The pairwise strategy does its own backward pass and
//! optimiser update per rollout step, so only one step's autodiff graph is alive at
//! a time; one "step" in the logs and in `--ckpt-every` is one scene (one scene for
//! `batch` and `align`, one reference/frame pair for `fusion-align`).
//!
//! `--resume` fails loudly when the strategy's state file or checkpoint files are
//! missing or corrupt, or belong to another strategy or preset. `--resume --fresh`
//! instead resumes when it can and starts from scratch when there is nothing
//! usable (handy for a restart-on-crash wrapper script).

#![allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "GPU device handles are not Copy, so the device is passed by reference on every backend"
)]

use std::error::Error;

use burn::{
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::{Backend, Tensor},
    tensor::ElementConversion,
};

use stacker_nn::{
    TrainBackend,
    checkpoint::{self, TrainState},
    data::{
        AlignSceneCache, RealStackScene, SceneCache, discover_align_scenes, discover_real_stacks,
        discover_rollout_scenes,
    },
    discovery::ModelManifest,
    infer::load_weights,
    loss::{
        FocusFusionLossConfig, corner_alignment::BatchAlignmentLossConfig,
        focus_batch::FocusBatchLossConfig, pair_corner_alignment::PairAlignmentLossConfig,
    },
    model::{
        FocusMergeNet, FocusMergeNetConfig, batch_align::BatchAlignNetConfig,
        batch_merge::BatchMergeNetConfig, fusion_align::FusionAlignNetConfig,
    },
    train::{
        align_loss,
        args::{Args, ArgsError, TrainStrategy, USAGE},
        batch_loss, cosine_lr, fusion_align_loss, photometric_finetune_term,
        rng::{Rng, random_crop},
        rollout_train, scheduled_sampling_prob,
        strategy_checkpoint::{self as sc, StrategyState},
    },
};

/// Autodiff training backend (wraps whichever base backend feature is active).
type B = TrainBackend;

/// The training device type of [`B`].
type Dev = <B as Backend>::Device;

/// What one step of a single-loss strategy yields: the loss and a short log tag.
type StepLoss = Result<(Tensor<B, 1>, &'static str), Box<dyn Error>>;

/// Bytes of the `--cache-mb` budget.
const fn cache_bytes(args: &Args) -> usize {
    args.cache_mb.saturating_mul(1024 * 1024)
}

/// Fail when `found == 0`: a data directory without scenes is a typing mistake.
fn require_scenes(found: usize, args: &Args) -> Result<(), Box<dyn Error>> {
    if found == 0 {
        return Err(format!(
            "no scenes (dirs with metadata.json) found under {}",
            args.data.display()
        )
        .into());
    }
    println!("Discovered {found} scene(s) under {}", args.data.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Pairwise (the original strategy: per-step updates, checkpoint.rs)
// ---------------------------------------------------------------------------

#[expect(
    clippy::too_many_lines,
    reason = "the training driver is one linear script: setup, resume, epoch loop, final export"
)]
fn train_pairwise(args: &Args, device: &Dev) -> Result<(), Box<dyn Error>> {
    let scenes = discover_rollout_scenes(&args.data)?;
    require_scenes(scenes.len(), args)?;

    let config = FocusMergeNetConfig::from_size(args.size);

    let total_steps = args.epochs * scenes.len();
    let warmup = (total_steps / 20).max(1);

    // Resume from the rolling checkpoint if requested. A missing or corrupt
    // checkpoint is an error unless `--fresh` explicitly asks to start over.
    let mut start_epoch = 0usize;
    let mut global_step = 0usize;
    let mut resumed = false;
    if args.resume {
        match checkpoint::plan_resume(&args.out, args.size, total_steps) {
            Ok(state) => {
                start_epoch = state.epoch;
                global_step = state.global_step;
                resumed = true;
            }
            Err(e) if args.fresh => {
                println!("--resume: {e}; --fresh given, starting from scratch");
            }
            Err(e) => return Err(e.into()),
        }
    }

    let mut model = if resumed {
        let latest = checkpoint::latest_model_path(&args.out);
        println!(
            "Resuming from {} at epoch {start_epoch}, step {global_step}",
            latest.display()
        );
        load_weights::<B>(&config, &latest, device)?
    } else {
        config.init::<B>(device)
    };

    let loss = FocusFusionLossConfig::new().init();
    // Decoded scenes kept between steps, bounded by --cache-mb (see SceneCache).
    let mut cache = SceneCache::new(cache_bytes(args));
    let mut optim = AdamWConfig::new().init::<B, FocusMergeNet<B>>();
    // Restore Adam moments on resume so training continues smoothly.
    if resumed {
        optim = checkpoint::load_optim::<B, _>(optim, &args.out, device)?;
    }

    // Advance the RNG deterministically so resumed runs don't repeat the exact
    // same crop/shuffle sequence from step 0.
    let mut rng = Rng::new(args.seed.wrapping_add(global_step as u64));

    for epoch in start_epoch..args.epochs {
        // Step counter at the start of this epoch: a mid-epoch checkpoint stores
        // this (not the current step) so a resume reruns the epoch from scene 0 with
        // the learning-rate schedule rewound to match.
        let epoch_start_step = global_step;
        let ss_prob = scheduled_sampling_prob(epoch, args.epochs, args.ss_max);

        let mut order: Vec<usize> = (0..scenes.len()).collect();
        rng.shuffle(&mut order);

        let mut epoch_loss = 0.0_f64;
        let mut counted = 0usize;

        for &si in &order {
            // Decode the scene once (or take it from the cache), pick ONE
            // consistent crop so the threaded composite stays spatially aligned
            // across steps, and cut every step from that crop.
            let images = cache.get(&scenes[si])?;
            let crop = random_crop(images.dims(), args.crop, &mut rng);
            let n_steps = scenes[si].n_steps;
            let steps = scenes[si].crop_steps::<B>(&images, device, Some(crop))?;
            drop(images);

            // Scheduled-sampling decisions: step 0 always teacher-forced.
            let use_prediction: Vec<bool> = (0..n_steps)
                .map(|k| k > 0 && rng.next_f64() < ss_prob)
                .collect();

            // One backward pass and one optimiser update per rollout step; each
            // step's graph is freed before the next (truncated BPTT).
            let lr = cosine_lr(args.lr, global_step, total_steps, warmup);
            let (updated, report) =
                rollout_train(model, &mut optim, lr, &loss, &steps, &use_prediction);
            model = updated;
            let loss_val = report.total;

            epoch_loss += loss_val;
            counted += 1;
            global_step += 1;

            if global_step.is_multiple_of(50) {
                println!(
                    "  step {global_step:>6}  epoch {epoch:>3}  lr {lr:.2e}  loss {loss_val:.5}  ss_p {ss_prob:.3}"
                );
            }

            // Rolling crash-recovery checkpoint. `epoch` (not epoch+1) and the
            // epoch-start step are stored so an interrupted epoch is restarted from
            // its beginning, with the schedule position it had then, on resume.
            if global_step.is_multiple_of(args.ckpt_every) {
                let state = TrainState {
                    epoch,
                    global_step: epoch_start_step,
                    size: args.size,
                    total_steps: Some(total_steps),
                };
                checkpoint::save_rolling::<B, _>(&model, &optim, &args.out, &state)?;
            }
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "the number of scenes per epoch is tiny"
        )]
        let mean = if counted == 0 {
            f64::NAN
        } else {
            epoch_loss / counted as f64
        };
        println!("epoch {epoch:>3}/{}  mean_loss {mean:.5}", args.epochs);

        // Per-epoch export (f16) + refreshed rolling checkpoint (f32). The state
        // records `epoch + 1` (the next epoch to run) and the step it starts at, so
        // a clean resume skips this epoch.
        checkpoint::save_export(
            &model,
            &args.out,
            &format!("focusmerge_epoch_{epoch:03}"),
            args.size,
        )?;
        let state = TrainState {
            epoch: epoch + 1,
            global_step,
            size: args.size,
            total_steps: Some(total_steps),
        };
        checkpoint::save_rolling::<B, _>(&model, &optim, &args.out, &state)?;
    }

    // Final export (+ manifest).
    checkpoint::save_export(&model, &args.out, "focusmerge_final", args.size)?;
    println!(
        "Training complete. Final weights: {}.mpk",
        args.out.join("focusmerge_final").display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// The other strategies: one loss and one optimiser update per example
// ---------------------------------------------------------------------------

/// The optimiser loop shared by `batch`, `align` and `fusion-align`.
///
/// `fresh_model` builds an untrained model; `step_loss(model, example, epoch, rng)`
/// builds the differentiable loss of one example (and a short log tag). Learning
/// rate schedule, resume, rolling and per-epoch checkpoints, and the final export
/// follow the pairwise driver, with the checkpoint stems of the strategy.
#[expect(
    clippy::too_many_lines,
    reason = "one linear script: setup, resume, epoch loop, final export"
)]
fn train_single_loss<M, F>(
    args: &Args,
    manifest: &ModelManifest,
    n_examples: usize,
    fresh_model: impl Fn() -> M,
    mut step_loss: F,
    device: &Dev,
) -> Result<(), Box<dyn Error>>
where
    M: AutodiffModule<B>,
    F: FnMut(&M, usize, usize, &mut Rng) -> StepLoss,
{
    let strategy = args.strategy;
    let total_steps = args.epochs * n_examples;
    let warmup = (total_steps / 20).max(1);

    let mut start_epoch = 0usize;
    let mut global_step = 0usize;
    let mut resumed = false;
    if args.resume {
        match sc::plan_resume(&args.out, strategy, args.size, total_steps) {
            Ok(state) => {
                start_epoch = state.epoch;
                global_step = state.global_step;
                resumed = true;
            }
            Err(e) if args.fresh => {
                println!("--resume: {e}; --fresh given, starting from scratch");
            }
            Err(e) => return Err(e.into()),
        }
    }

    let mut model = if resumed {
        println!(
            "Resuming from {} at epoch {start_epoch}, step {global_step}",
            sc::latest_model_path(&args.out, strategy).display()
        );
        sc::load_latest_model::<B, M>(fresh_model(), &args.out, strategy, device)?
    } else {
        fresh_model()
    };
    let mut optim = AdamWConfig::new().init::<B, M>();
    if resumed {
        optim = sc::load_optim::<B, M, _>(optim, &args.out, strategy, device)?;
    }

    let mut rng = Rng::new(args.seed.wrapping_add(global_step as u64));
    let state_at = |epoch: usize, global_step: usize| StrategyState {
        epoch,
        global_step,
        size: args.size,
        strategy: strategy.as_str().to_owned(),
        total_steps: Some(total_steps),
    };

    for epoch in start_epoch..args.epochs {
        let epoch_start_step = global_step;
        let mut order: Vec<usize> = (0..n_examples).collect();
        rng.shuffle(&mut order);

        let mut epoch_loss = 0.0_f64;
        let mut counted = 0usize;

        for &example in &order {
            let (terms, tag) = step_loss(&model, example, epoch, &mut rng)?;
            let loss_val: f64 = terms.clone().into_scalar().elem();

            let lr = cosine_lr(args.lr, global_step, total_steps, warmup);
            let grads = GradientsParams::from_grads(terms.backward(), &model);
            model = optim.step(lr, model, grads);

            epoch_loss += loss_val;
            counted += 1;
            global_step += 1;

            if global_step.is_multiple_of(50) {
                println!(
                    "  step {global_step:>6}  epoch {epoch:>3}  lr {lr:.2e}  loss {loss_val:.5}{tag}"
                );
            }

            // Mid-epoch saves store the epoch-start step, like the pairwise driver.
            if global_step.is_multiple_of(args.ckpt_every) {
                sc::save_rolling::<B, M, _>(
                    &model,
                    &optim,
                    &args.out,
                    strategy,
                    manifest,
                    &state_at(epoch, epoch_start_step),
                )?;
            }
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "the number of examples per epoch is tiny"
        )]
        let mean = if counted == 0 {
            f64::NAN
        } else {
            epoch_loss / counted as f64
        };
        println!("epoch {epoch:>3}/{}  mean_loss {mean:.5}", args.epochs);

        sc::save_export::<B, M>(
            &model,
            &args.out,
            &sc::epoch_stem(strategy, epoch),
            manifest,
        )?;
        sc::save_rolling::<B, M, _>(
            &model,
            &optim,
            &args.out,
            strategy,
            manifest,
            &state_at(epoch + 1, global_step),
        )?;
    }

    let final_stem = sc::final_stem(strategy);
    sc::save_export::<B, M>(&model, &args.out, &final_stem, manifest)?;
    println!(
        "Training complete. Final weights: {}.mpk",
        args.out.join(final_stem).display()
    );
    Ok(())
}

fn train_batch(args: &Args, device: &Dev) -> Result<(), Box<dyn Error>> {
    let scenes = discover_rollout_scenes(&args.data)?;
    require_scenes(scenes.len(), args)?;

    let config = BatchMergeNetConfig::from_size(args.size);
    let manifest = ModelManifest::from_size_batch(args.size);
    let loss = FocusBatchLossConfig::new().init();
    let mut cache = SceneCache::new(cache_bytes(args));

    train_single_loss(
        args,
        &manifest,
        scenes.len(),
        || config.init::<B>(device),
        |model, example, _epoch, rng| {
            let images = cache.get(&scenes[example])?;
            let crop = random_crop(images.dims(), args.crop, rng);
            let sample = scenes[example].batch_from_images::<B>(&images, device, Some(crop))?;
            Ok((batch_loss(model, &loss, &sample), ""))
        },
        device,
    )
}

fn train_align(args: &Args, device: &Dev) -> Result<(), Box<dyn Error>> {
    let scenes = discover_align_scenes(&args.data)?;
    require_scenes(scenes.len(), args)?;

    let mut config = BatchAlignNetConfig::from_size(args.size);
    if let Some(radius) = args.corr_radius {
        config = config.with_corr_radius(radius);
    }
    // The radius fixes the head's input width, so the manifest must carry the one
    // this run used (not the preset's).
    let mut manifest = ModelManifest::from_size_align(args.size);
    manifest.corr_radius = Some(config.corr_radius);
    let loss = BatchAlignmentLossConfig::new().init();
    let mut cache = AlignSceneCache::new(cache_bytes(args));

    // Photometric fine-tune: only stacks with a non-reference frame are usable, and
    // the last quarter of the epochs (rounded down) mixes the term in.
    let real_scenes: Vec<RealStackScene> = match &args.real_data {
        Some(dir) => discover_real_stacks(dir)?
            .into_iter()
            .filter(|s| s.len() >= 2)
            .collect(),
        None => Vec::new(),
    };
    let finetune_start = args.epochs - args.epochs / 4;
    if !real_scenes.is_empty() {
        println!(
            "Discovered {} real (unlabelled) stack(s): photometric fine-tuning for epochs \
             {finetune_start}..{}",
            real_scenes.len(),
            args.epochs
        );
    } else if args.real_data.is_some() {
        println!("note: --real-data holds no stack with two or more frames; fine-tune disabled");
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a loss weight; f32 precision is plenty"
    )]
    let photometric_weight = args.photometric_weight as f32;

    train_single_loss(
        args,
        &manifest,
        scenes.len(),
        || config.init::<B>(device),
        |model, example, epoch, rng| {
            let images = cache.get(&scenes[example])?;
            let crop = random_crop(images.dims(), args.crop, rng);
            let sample = scenes[example].sample_from_images::<B>(&images, device, Some(crop))?;
            let mut terms = align_loss(model, &loss, &sample);
            let mut tag = "";

            if !real_scenes.is_empty() && epoch >= finetune_start {
                let scene = &real_scenes[rng.below(real_scenes.len())];
                let crop = random_crop(scene.frame_dims()?, args.crop, rng);
                let stack = scene.get::<B>(device, Some(crop))?;
                if let Some(term) = photometric_finetune_term(model, &stack) {
                    terms = terms.add(term.mul_scalar(photometric_weight));
                    tag = " [finetune]";
                }
            }
            Ok((terms, tag))
        },
        device,
    )
}

fn train_fusion_align(args: &Args, device: &Dev) -> Result<(), Box<dyn Error>> {
    let scenes = discover_align_scenes(&args.data)?;
    require_scenes(scenes.len(), args)?;

    let mut config = FusionAlignNetConfig::from_size(args.size);
    if let Some(radius) = args.corr_radius {
        config = config.with_corr_radius(radius);
    }
    let mut manifest = ModelManifest::from_size_fusion_align(args.size);
    manifest.corr_radius = Some(config.corr_radius);
    let loss = PairAlignmentLossConfig::new().init();
    let mut cache = AlignSceneCache::new(cache_bytes(args));

    // One example per (scene, frame): frame 0 against itself is a valid sample.
    let examples: Vec<(usize, usize)> = scenes
        .iter()
        .enumerate()
        .flat_map(|(si, scene)| (0..scene.meta.n_planes).map(move |fi| (si, fi)))
        .collect();
    if examples.is_empty() {
        return Err("no alignment scene contains any frame to pair".into());
    }

    train_single_loss(
        args,
        &manifest,
        examples.len(),
        || config.init::<B>(device),
        |model, example, _epoch, rng| {
            let (si, frame) = examples[example];
            let images = cache.get(&scenes[si])?;
            let crop = random_crop(images.dims(), args.crop, rng);
            let sample = scenes[si].pair_from_images::<B>(&images, frame, device, Some(crop))?;
            Ok((fusion_align_loss(model, &loss, &sample), ""))
        },
        device,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli_words: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse_from(&cli_words) {
        Ok(parsed) => parsed,
        Err(ArgsError::Help) => {
            eprintln!("{USAGE}");
            return Ok(());
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    std::fs::create_dir_all(&args.out)?;
    for flag in args.ignored_flags() {
        println!(
            "note: {flag} has no effect with --strategy {}",
            args.strategy.as_str()
        );
    }

    let device = Dev::default();

    // Seed Burn's own RNG (weight initialisation, which draws from the backend's
    // global generator) as well as the xorshift generator (scene order, crops,
    // scheduled sampling), so two runs with the same --seed start from the same
    // weights. A resumed run overwrites the weights from the checkpoint.
    <B as Backend>::seed(args.seed);

    match args.strategy {
        TrainStrategy::Pairwise => train_pairwise(&args, &device),
        TrainStrategy::Batch => train_batch(&args, &device),
        TrainStrategy::Align => train_align(&args, &device),
        TrainStrategy::FusionAlign => train_fusion_align(&args, &device),
    }
}
