//! The `z-stackr` command-line entry point: parse the arguments, set up logging,
//! load the settings and run the pipeline once, once per subfolder (`--stacks`)
//! or in `--monitor` mode.

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
use notify::{RecursiveMode, Watcher};
use stacker_cli::{
    args::CliArgs,
    batch::{self, Plan},
    config::load_config,
    monitor::is_relevant_event,
    pipeline,
};
use stacker_core::{
    error::StackerError,
    settings::StackingSettings,
    telemetry::{init_tracing, init_tracing_to_file},
};
use std::{process::ExitCode, sync::mpsc::channel, time::Duration};

fn main() -> ExitCode {
    let args = CliArgs::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Display form: the variants carry the user-facing message.
            eprintln!("error: {e}");
            tracing::error!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &CliArgs) -> Result<(), StackerError> {
    // If --log-file was supplied write to file + stdout; otherwise stdout only.
    if let Some(ref log_path) = args.log_file {
        init_tracing_to_file(log_path, "info")?;
        tracing::info!(log_file = %log_path.display(), "file logging active");
    } else {
        init_tracing("info")?;
    }

    // A missing, unreadable or invalid config stops the run before any work.
    let settings = load_config(args.config.as_deref(), args.write_default_config)?;

    // One stack, or one per subfolder of the input folder (`--stacks`).
    match batch::plan(args)? {
        Plan::Batch(subfolders) => run_batch(args, &settings, &subfolders),
        Plan::Single if args.monitor => monitor(args, &settings),
        Plan::Single => pipeline::run_pipeline(args, &settings),
    }
}

/// Stack every subfolder on its own and print the summary; an error when any
/// stack failed.
fn run_batch(
    args: &CliArgs,
    settings: &StackingSettings,
    subfolders: &[std::path::PathBuf],
) -> Result<(), StackerError> {
    let outcomes = batch::run_batch(args, settings, subfolders)?;
    if batch::print_batch_summary(&outcomes) {
        Ok(())
    } else {
        Err(StackerError::Validation(
            "one or more stacks failed; see the summary above".to_owned(),
        ))
    }
}

fn watcher_error(what: &str, e: impl std::fmt::Display) -> StackerError {
    StackerError::Io(std::io::Error::other(format!("{what}: {e}")))
}

/// Watch the input directory and re-run the pipeline when a frame arrives.
fn monitor(args: &CliArgs, settings: &StackingSettings) -> Result<(), StackerError> {
    tracing::info!("Monitor mode enabled on {}", args.input_dir.display());

    let (tx, rx) = channel();
    let mut watcher = notify::recommended_watcher(tx)
        .map_err(|e| watcher_error("cannot create the file watcher", e))?;
    watcher
        .watch(&args.input_dir, RecursiveMode::NonRecursive)
        .map_err(|e| watcher_error(&format!("cannot watch '{}'", args.input_dir.display()), e))?;

    // Run once initially; a failure is reported but does not end the watch (the
    // directory may simply not hold enough frames yet).
    tracing::info!("Initial stacking pass...");
    if let Err(e) = pipeline::run_pipeline(args, settings) {
        tracing::error!("Initial stacking pass failed: {e}");
    }

    // The output image and the log file are never inputs, and their writes must
    // not retrigger a pass.
    let own_files = args.own_output_paths();
    loop {
        match rx.recv() {
            Ok(Ok(event)) => {
                if !is_relevant_event(&event, &own_files) {
                    continue;
                }
                // Debounce/wait slightly for file write to complete
                std::thread::sleep(Duration::from_millis(500));
                // Drain queue
                while rx.try_recv().is_ok() {}

                tracing::info!("Changes detected, re-running stacking pipeline...");
                if let Err(e) = pipeline::run_pipeline(args, settings) {
                    tracing::error!("Error during live stack: {e}");
                }
            }
            Ok(Err(e)) => tracing::error!("Watch error: {e}"),
            Err(e) => {
                tracing::error!("Watch channel disconnected: {e}");
                break;
            }
        }
    }
    Ok(())
}
