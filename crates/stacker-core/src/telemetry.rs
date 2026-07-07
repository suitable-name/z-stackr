#![allow(clippy::missing_errors_doc, clippy::uninlined_format_args)]

use std::{fs::File, path::Path};

use crate::error::StackerError;
use tracing_subscriber::{EnvFilter, prelude::*};

/// Default per-crate directives appended to every caller-supplied filter (and
/// to the `"info"` fallback) by both [`init_tracing`] and
/// [`init_tracing_to_file`].
///
/// GPU builds otherwise spam `wgpu_hal`/`wgpu_core`/`naga` lines (Vulkan
/// loader messages, missing X11/Wayland surface extensions, validation-layer
/// probes, MESA driver enumeration) — plus `zbus` (a transitive dependency
/// of the Vulkan/D-Bus portal probing on Linux) — drowning out this crate's
/// own log lines. `wgpu_core`/`wgpu_hal` are quieted to `error` because
/// their headless-host probe chatter is emitted at WARN level and is
/// harmless (a missing windowing-surface extension is expected for
/// off-screen compute); real GPU problems still surface through this
/// workspace's own validation error scopes and log targets
/// (`stacker_core::gpu` and the per-crate `gpu` modules). `naga`/`zbus` are
/// quieted to `warn`, and `cubecl`/`burn` (the `nn-gpu` backend's equally
/// chatty internals) to `warn` as well. Application-level targets stay at
/// whatever level the caller asked for. Appended after the caller's filter,
/// so a caller directive for one of these same targets (e.g. an explicit
/// `wgpu_core=debug` in a supplied filter string) still loses to this
/// default — the noise suppression is unconditional whenever it's reached
/// via [`init_tracing`]/[`init_tracing_to_file`]'s `filter` parameter.
/// `RUST_LOG`, when set, bypasses this entirely (see those functions' docs).
const DEFAULT_NOISE_SUPPRESSION: &str =
    "wgpu_core=error,wgpu_hal=error,naga=warn,zbus=warn,cubecl=warn,burn=warn";

/// Resolve the effective [`EnvFilter`] for either logging entry point.
///
/// Precedence: `RUST_LOG`, when set, is used verbatim and takes full
/// override precedence — no noise-suppression directives are appended, so
/// it always shows exactly what the user asked for. Otherwise `filter` (or
/// `"info"` if `filter` itself fails to parse) has
/// [`DEFAULT_NOISE_SUPPRESSION`] appended automatically, so GPU builds don't
/// spam `wgpu_hal`/`wgpu_core`/`naga`/`zbus` INFO/WARN lines by default.
///
/// Shared by [`init_tracing`] and both layers of [`init_tracing_to_file`] so
/// the resolution logic (and the resulting filter) never drifts between
/// call sites.
///
/// # Errors
/// Returns a [`StackerError::MathError`] if neither the supplied `filter`
/// nor the `"info"` fallback parses as a valid [`EnvFilter`] directive
/// string.
fn resolve_env_filter(filter: &str) -> Result<EnvFilter, StackerError> {
    EnvFilter::try_from_default_env()
        .or_else(|_| {
            EnvFilter::try_new(format!("{filter},{DEFAULT_NOISE_SUPPRESSION}"))
                .or_else(|_| EnvFilter::try_new(format!("info,{DEFAULT_NOISE_SUPPRESSION}")))
        })
        .map_err(|e| StackerError::MathError(format!("Invalid tracing filter: {}", e)))
}

/// Initialise `tracing` with a stdout subscriber only.
///
/// Idempotent: if a global subscriber is already registered the error is
/// mapped to a [`StackerError::MathError`] (the subscriber type used for
/// general initialisation failures throughout the codebase).
///
/// # Filter precedence
///
/// `RUST_LOG`, when set, is used verbatim and takes full override
/// precedence — it is the escape hatch a developer uses to see everything
/// (including `wgpu`/`naga`/`zbus` internals) without editing code.
/// Otherwise the supplied `filter` (or `"info"` if `filter` itself fails to
/// parse) has [`DEFAULT_NOISE_SUPPRESSION`] appended automatically, so GPU
/// builds don't spam `wgpu_hal`/`wgpu_core`/`naga`/`zbus` INFO/WARN lines by
/// default.
pub fn init_tracing(filter: &str) -> Result<(), StackerError> {
    let env_filter = resolve_env_filter(filter)?;

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .try_init()
        .map_err(|e| StackerError::MathError(format!("Failed to initialize tracing: {}", e)))?;

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
/// # Filter precedence
///
/// `RUST_LOG`, when set, is used verbatim (identically for both the stdout
/// and file layers) and takes full override precedence — it is the escape
/// hatch a developer uses to see everything (including `wgpu`/`naga`/`zbus`
/// internals) without editing code. Otherwise the supplied `filter` (or
/// `"info"` if `filter` itself fails to parse) has
/// [`DEFAULT_NOISE_SUPPRESSION`] appended automatically, so GPU builds don't
/// spam `wgpu_hal`/`wgpu_core`/`naga`/`zbus` INFO/WARN lines by default.
///
/// # Errors
///
/// Returns [`StackerError::Io`] if the log file cannot be created/opened, or
/// [`StackerError::MathError`] if the global subscriber is already set.
pub fn init_tracing_to_file(path: &Path, filter: &str) -> Result<(), StackerError> {
    // Ensure parent directory exists.
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let log_file = File::options().create(true).append(true).open(path)?;

    // Both layers resolve the exact same filter (see resolve_env_filter's
    // docs) so stdout and the log file never disagree about what's being
    // emitted.
    let env_filter_stdout = resolve_env_filter(filter)?;
    let env_filter_file = resolve_env_filter(filter)?;

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
        .map_err(|e| StackerError::MathError(format!("Failed to initialize tracing: {}", e)))?;

    Ok(())
}
