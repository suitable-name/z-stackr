#![allow(clippy::derive_partial_eq_without_eq)]

// Custom global allocator to reduce allocation churn. jemalloc's sys crate has
// no working build on Windows MSVC (autotools/sh based), so we use mimalloc
// there and jemalloc on every other platform (Linux/macOS).
#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(target_os = "windows")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use stacker_cli::{
    args::CliArgs,
    batch::{
        self, InputKind, StackOutcome, StacksDecision, classify_input_dir, decision_from_flag,
        ensure_batch_output_dir, looks_like_image_file, prompt_stacks_mode,
        resolve_batch_output_path, stdin_is_tty,
    },
    pipeline::{self, PipelineParams, PipelineProgress},
};
use stacker_core::{
    error::StackerError,
    settings::{DEFAULT_CONFIG_TOML, StackingSettings},
    telemetry::{init_tracing, init_tracing_to_file},
};
use std::{path::Path, sync::mpsc::channel, time::Duration};

/// Run a single stack (one `PipelineParams` worth of work) against a
/// resolved `paths` list and `output_file`, rendering the same two-bar
/// `indicatif` progress UX the CLI has always shown. Shared by both the
/// single-stack path and each iteration of the subfolder-batch loop.
fn run_stack(
    paths: Vec<std::path::PathBuf>,
    output_file: std::path::PathBuf,
    args: &CliArgs,
    settings: &StackingSettings,
) -> Result<(), StackerError> {
    let params = PipelineParams {
        paths,
        output_file,
        mode: args.mode.clone(),
        tile_size: args.tile_size,
        #[cfg(feature = "nn")]
        model: args.model.clone(),
        #[cfg(not(feature = "nn"))]
        model: None,
        #[cfg(feature = "nn")]
        device: args.device.clone(),
        #[cfg(not(feature = "nn"))]
        device: None,
        #[cfg(feature = "nn")]
        align_model: args.align_model.clone(),
        #[cfg(not(feature = "nn"))]
        align_model: None,
    };

    let mut decode_bar: Option<ProgressBar> = None;
    let mut align_bar: Option<ProgressBar> = None;
    let mut fuse_bar: Option<ProgressBar> = None;
    let on_progress = move |event: PipelineProgress| match event {
        PipelineProgress::DecodeRaw { current, total } => {
            let pb = decode_bar.get_or_insert_with(|| {
                let pb = ProgressBar::new(total as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} ({eta}) Decoding RAW Frames")
                        .expect("hardcoded progress-bar template is always valid")
                        .progress_chars("#>-"),
                );
                pb
            });
            pb.set_position(current as u64);
            if current == total {
                pb.finish_with_message("RAW decode complete");
            }
        }
        PipelineProgress::AlignStart { total } => {
            let pb = ProgressBar::new(total.saturating_sub(1) as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) Aligning Frames")
                    .expect("hardcoded progress-bar template is always valid")
                    .progress_chars("#>-"),
            );
            align_bar = Some(pb);
        }
        PipelineProgress::AlignFrame { current, .. } => {
            if let Some(pb) = &align_bar {
                pb.set_position(current.saturating_sub(1) as u64);
            }
        }
        PipelineProgress::AlignDone => {
            if let Some(pb) = align_bar.take() {
                pb.finish_with_message("Alignment complete");
            }
        }
        PipelineProgress::FuseStart { total } => {
            let pb = ProgressBar::new(total as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} ({eta}) Fusing Tiles")
                    .expect("hardcoded progress-bar template is always valid")
                    .progress_chars("#>-"),
            );
            fuse_bar = Some(pb);
        }
        PipelineProgress::FuseTile { current, .. } => {
            if let Some(pb) = &fuse_bar {
                pb.set_position(current as u64);
            }
        }
        PipelineProgress::FuseDone => {
            if let Some(pb) = fuse_bar.take() {
                pb.finish_with_message("Fusion complete");
            }
        }
        PipelineProgress::Encoding => {}
    };

    smol::block_on(pipeline::run_pipeline(&params, settings, on_progress))
}

/// Build the frame list for `args.input_dir` and run a single stack to
/// `args.output_file`, exactly as the CLI has always behaved for a plain
/// "one folder of images" `--input-dir`. Monitor mode calls this once per
/// detected filesystem change, so the frame list must be re-resolved every
/// time.
fn run_once(args: &CliArgs, settings: &StackingSettings) -> Result<(), StackerError> {
    let paths = pipeline::collect_image_paths(&args.input_dir)?;
    run_stack(paths, args.output_file.clone(), args, settings)
}

/// Run one subfolder's stack, prefixing every progress/status line with the
/// subfolder's name so a multi-stack batch run's output stays legible.
/// Errors are returned (never panics) so the caller can log-and-continue
/// per the "a failing subfolder must not abort the batch" requirement.
fn run_subfolder_stack(
    subfolder: &std::path::Path,
    output_dir: &std::path::Path,
    args: &CliArgs,
    settings: &StackingSettings,
) -> Result<std::path::PathBuf, StackerError> {
    let name = subfolder
        .file_name()
        .map_or_else(|| "stack".to_owned(), |n| n.to_string_lossy().into_owned());

    println!("=== [{name}] stacking '{}' ===", subfolder.display());
    tracing::info!(subfolder = %subfolder.display(), "batch: starting subfolder stack");

    let paths = pipeline::collect_image_paths(subfolder)?;
    let output_file = resolve_batch_output_path(output_dir, &name, &settings.image_saving);

    run_stack(paths, output_file.clone(), args, settings)?;
    println!("=== [{name}] done -> {} ===", output_file.display());
    Ok(output_file)
}

/// Run every image-bearing subfolder of `args.input_dir` as its own,
/// independent, sequential stack (see the module docs on `batch` for why
/// sequential — the pipeline is already internally parallel, and running
/// multiple stacks concurrently would multiply peak memory per stack). A
/// failing subfolder is logged and does not abort the batch; the exit code
/// reflects whether *any* subfolder failed.
///
/// `args.output_file` is reinterpreted as an output directory here (created
/// if missing) — the caller has already validated it doesn't look like an
/// image file.
fn run_batch(
    subfolders: &[std::path::PathBuf],
    args: &CliArgs,
    settings: &StackingSettings,
) -> Result<(), StackerError> {
    ensure_batch_output_dir(&args.output_file)?;

    let mut outcomes = Vec::with_capacity(subfolders.len());
    for subfolder in subfolders {
        let name = subfolder
            .file_name()
            .map_or_else(|| "stack".to_owned(), |n| n.to_string_lossy().into_owned());
        let result = run_subfolder_stack(subfolder, &args.output_file, args, settings).map_err(|e| {
            tracing::error!(subfolder = %subfolder.display(), error = %e, "batch: subfolder stack failed");
            e.to_string()
        });
        outcomes.push(StackOutcome { name, result });
    }

    let all_ok = batch::print_batch_summary(&outcomes);
    if all_ok {
        Ok(())
    } else {
        Err(StackerError::AlignmentFailed(
            "one or more subfolder stacks failed — see the summary above".to_owned(),
        ))
    }
}

/// Resolve the interactive/`--stacks`-flag decision for an `--input-dir`
/// that [`classify_input_dir`] reported as containing image-bearing
/// subfolders. Returns a hard error if neither a TTY nor `--stacks` is
/// available to answer the question.
fn resolve_stacks_decision(
    args: &CliArgs,
    subfolders: usize,
    direct_images: usize,
) -> Result<StacksDecision, StackerError> {
    if let Some(decision) = decision_from_flag(args.stacks) {
        return Ok(decision);
    }
    if stdin_is_tty() {
        return prompt_stacks_mode(subfolders, direct_images);
    }
    Err(StackerError::AlignmentFailed(format!(
        "'{}' contains {subfolders} subfolder(s) with images and {direct_images} image(s) directly in the \
         folder, and stdin is not an interactive terminal — pass `--stacks subfolders` or `--stacks single` \
         to say which one you mean",
        args.input_dir.display()
    )))
}

fn load_config_and_clamp(config_path: &Path) -> StackingSettings {
    let mut settings = StackingSettings::default();
    if config_path.exists() {
        tracing::info!("Loading config from {:?}", config_path);
        match std::fs::read_to_string(config_path) {
            Ok(content) => match toml::from_str::<StackingSettings>(&content) {
                Ok(mut s) => {
                    s.clamp_valid();
                    s.preprocessing.clamp_valid();
                    s.image_saving.clamp_valid();
                    settings = s;
                }
                Err(e) => tracing::error!("Failed to parse config file: {:?}", e),
            },
            Err(e) => tracing::error!("Failed to read config file: {:?}", e),
        }
    } else {
        tracing::info!(
            "Config file not found, creating default config at {:?}",
            config_path
        );
        if let Some(parent) = config_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) = std::fs::write(config_path, DEFAULT_CONFIG_TOML) {
            tracing::error!("Failed to create default config file: {:?}", e);
        }
    }
    settings
}

fn main() -> Result<(), StackerError> {
    let args = CliArgs::parse();

    // If --log-file was supplied write to file + stdout; otherwise stdout only.
    if let Some(ref log_path) = args.log_file {
        init_tracing_to_file(log_path, "info")?;
        tracing::info!(log_file = %log_path.display(), "file logging active");
    } else {
        init_tracing("info")?;
    }

    let mut settings = args
        .config
        .as_ref()
        .map_or_else(StackingSettings::default, |config_path| {
            load_config_and_clamp(config_path)
        });

    // `--no-gpu` overrides the config file's `use_gpu` for this run only
    // (never written back to disk). Apply before the runtime switch call
    // below so it is the effective value that reaches `set_enabled`.
    #[cfg(feature = "gpu")]
    if args.no_gpu {
        settings.use_gpu = false;
    }
    // `--optimizer` overrides the config file's `optimizer` for this run
    // only (never written back to disk), same pattern as `--no-gpu` above.
    if let Some(optimizer) = args.optimizer {
        settings.optimizer = optimizer.into();
    }
    // Engage/disengage the shared runtime GPU switch once, before any
    // stacking work starts. A no-op (compiles to nothing) in a default,
    // non-`gpu` build — see `stacker_core::gpu::set_enabled`'s docs.
    #[cfg(feature = "gpu")]
    stacker_core::gpu::set_enabled(settings.use_gpu);

    let input_kind = classify_input_dir(&args.input_dir)?;

    // Subfolder-batch mode: only reachable when `classify_input_dir` found
    // at least one image-bearing subfolder. `--monitor` and subfolder-batch
    // mode are mutually exclusive (watching N subfolders at once is future
    // work -- see the README) regardless of which way the subfolder/single
    // question is ultimately answered below; a folder that resolves to
    // `Single` (either by flag or by prompt) still runs fine under
    // `--monitor` -- only an actual `Subfolders` decision conflicts with it.
    if let InputKind::HasStackSubfolders {
        subfolders,
        direct_images,
    } = &input_kind
    {
        let decision = resolve_stacks_decision(&args, subfolders.len(), *direct_images)?;
        if decision == StacksDecision::Subfolders {
            if args.monitor {
                return Err(StackerError::AlignmentFailed(
                    "--monitor is incompatible with subfolder-batch mode (watching N subfolders at once is \
                     future work) -- run without --monitor, or pass --stacks single to stack only the images \
                     directly in --input-dir"
                        .to_owned(),
                ));
            }
            if looks_like_image_file(&args.output_file) {
                return Err(StackerError::AlignmentFailed(format!(
                    "in batch mode --output-file must be a directory, not an image file path (got '{}') -- \
                     each subfolder's output filename is derived from the configured image-saving settings",
                    args.output_file.display()
                )));
            }
            return run_batch(subfolders, &args, &settings);
        }
        // decision == Single: fall through to the unchanged single-stack /
        // monitor logic below, using only the direct images in input_dir
        // (collect_image_paths already ignores subdirectories).
    }

    if args.monitor {
        tracing::info!("Monitor mode enabled on {:?}", args.input_dir);

        let (tx, rx) = channel();
        let mut watcher = notify::recommended_watcher(tx).expect("Failed to create watcher");

        watcher
            .watch(&args.input_dir, RecursiveMode::NonRecursive)
            .expect("Failed to watch directory");

        // Canonicalise once up front so every event's paths can be compared
        // against it cheaply. `output_file` may not exist yet on the first
        // pass, so fall back to the raw (un-canonicalised) path in that case
        // — it still lets us filter out later self-triggered events once the
        // file exists and canonicalises successfully.
        let output_file_canon = args
            .output_file
            .canonicalize()
            .unwrap_or_else(|_| args.output_file.clone());

        // Run once initially
        tracing::info!("Initial stacking pass...");
        let _ = run_once(&args, &settings);

        loop {
            match rx.recv() {
                Ok(Ok(Event {
                    kind: EventKind::Create(_) | EventKind::Modify(_),
                    paths,
                    ..
                })) => {
                    // Skip events that only touch our own output file — writing
                    // the stacked result would otherwise re-trigger this branch
                    // and loop forever.
                    if is_output_file_only_event(&paths, &output_file_canon) {
                        continue;
                    }

                    // Debounce/wait slightly for file write to complete
                    std::thread::sleep(Duration::from_millis(500));
                    // Drain queue. The triggering event above already proved
                    // this batch contains a genuine (non-output-file) change,
                    // so we only need to drain — not re-evaluate — the rest.
                    while rx.try_recv().is_ok() {}

                    tracing::info!("Changes detected, re-running stacking pipeline...");
                    if let Err(e) = run_once(&args, &settings) {
                        tracing::error!("Error during live stack: {:?}", e);
                    }
                }
                Ok(Err(e)) => tracing::error!("Watch error: {:?}", e),
                Ok(_) => {} // ignore other events
                Err(e) => {
                    tracing::error!("Watch channel disconnected: {:?}", e);
                    break;
                }
            }
        }
        Ok(())
    } else {
        run_once(&args, &settings)
    }
}

/// Returns `true` if every path in `event_paths` refers to `output_file_canon`
/// (so the event should be skipped as self-triggered), and `false` if
/// `event_paths` is empty or contains at least one path that is a genuine
/// input change. Paths are canonicalised before comparison so symlinks/`..`
/// components don't cause false negatives; a path that fails to canonicalise
/// (e.g. it was already deleted) is treated as a real change rather than
/// silently ignored.
fn is_output_file_only_event(
    event_paths: &[std::path::PathBuf],
    output_file_canon: &std::path::Path,
) -> bool {
    if event_paths.is_empty() {
        return false;
    }
    event_paths.iter().all(|p| {
        p.canonicalize()
            .is_ok_and(|canon| canon == output_file_canon)
    })
}
