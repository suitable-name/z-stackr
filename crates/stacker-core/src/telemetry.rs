//! Process-wide `tracing` set-up for the applications: a stdout subscriber, with
//! or without an append-only log file.

use std::{fs::File, path::Path};

use crate::error::StackerError;
use tracing_subscriber::{EnvFilter, prelude::*};

/// Build the filter shared by both initialisers.
///
/// `filter` is always validated first, so a malformed argument is reported
/// even when `RUST_LOG` is set. The effective filter is then `RUST_LOG` when
/// that variable is set and parses (it overrides `filter`, so a user can raise
/// the level at runtime), otherwise `filter` itself. An unparsable `RUST_LOG`
/// is ignored rather than treated as an error.
fn build_filter(filter: &str) -> Result<EnvFilter, StackerError> {
    let from_arg = EnvFilter::try_new(filter)
        .map_err(|e| StackerError::MathError(format!("Invalid tracing filter `{filter}`: {e}")))?;
    Ok(EnvFilter::try_from_default_env().unwrap_or(from_arg))
}

/// Initialise `tracing` with a stdout subscriber only.
///
/// Idempotent: if a global subscriber is already registered (by an earlier
/// call or by anyone else) this returns `Ok(())` and leaves the existing
/// subscriber in place. The `filter` argument is still validated on every
/// call, and an invalid one is returned as an error. `RUST_LOG`, when set and
/// valid, overrides `filter`.
///
/// # Errors
/// [`StackerError::MathError`] when `filter` is not a valid tracing filter.
pub fn init_tracing(filter: &str) -> Result<(), StackerError> {
    let env_filter = build_filter(filter)?;

    if tracing::dispatcher::has_been_set() {
        return Ok(());
    }

    // `try_init` only fails when another thread registered a subscriber between
    // the check above and here; the contract is the same, so that is `Ok` too.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .try_init();

    Ok(())
}

/// Initialise `tracing` with **both** a stdout subscriber and a structured
/// append-only file subscriber writing to `path`.
///
/// The file is opened in append mode so that successive runs accumulate in the
/// same log file.  If the parent directory does not exist it is created.
///
/// The file writer is synchronous (`std::fs::File`) — no background thread is
/// spawned, keeping the implementation `tokio`-free.
///
/// # Errors
///
/// Unlike [`init_tracing`] this is not idempotent: the caller asked for a file
/// subscriber, and silently skipping it would lose the log.
///
/// Returns [`StackerError::MathError`] if `filter` is invalid or a global
/// subscriber is already set (checked before any file or directory is
/// created, so a failed call leaves nothing behind), or [`StackerError::Io`]
/// if the log file cannot be created/opened. `RUST_LOG`, when set and valid,
/// overrides `filter`.
pub fn init_tracing_to_file(path: &Path, filter: &str) -> Result<(), StackerError> {
    // Validate everything that can fail before touching the file system.
    let env_filter_stdout = build_filter(filter)?;
    let env_filter_file = build_filter(filter)?;
    if tracing::dispatcher::has_been_set() {
        return Err(StackerError::MathError(
            "Failed to initialize tracing: a global subscriber is already set".to_owned(),
        ));
    }

    // Ensure parent directory exists.
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let log_file = File::options().create(true).append(true).open(path)?;

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(env_filter_stdout);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::sync::Mutex::new(log_file))
        .with_ansi(false)
        .with_filter(env_filter_file);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .try_init()
        .map_err(|e| StackerError::MathError(format!("Failed to initialize tracing: {e}")))?;

    Ok(())
}
