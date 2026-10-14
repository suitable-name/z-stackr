//! The command-line focus stacker: argument parsing, settings loading, the
//! `--monitor` loop's event filter and the tiled, out-of-core run.
//!
//! The per-frame and per-tile stages, and the out-of-core tile loop itself
//! (`stacker_pipeline::tiled`), are shared with the desktop app; what lives here
//! is what only the command line has: the clap arguments, the algorithm and
//! batch (`--stacks`) handling, the progress bars and the output-path
//! resolution.

#![warn(missing_docs)]

/// Command-line arguments.
pub mod args;
/// Batch stacking: one stack per subfolder of the input folder.
pub mod batch;
/// Loading the settings file passed with `--config`.
pub mod config;
/// The decisions of the `--monitor` loop.
pub mod monitor;
/// The tiled, out-of-core run.
pub mod pipeline;
