//! The stages of the focus-stacking pipeline, shared by the command-line tool
//! and the desktop app.
//!
//! Nothing here needs a thread, a clock or (apart from `PathSource` and the file
//! writers) a file system, so the crate also builds for `wasm32-unknown-unknown`.
//!
//! Each step (decoding, frame selection, seeding and refining the alignment, the
//! `Relief` threshold, colour conversion back to integers, settings loading) lives
//! here exactly once, as a plain function over explicit inputs; the apps keep
//! only what is theirs: the CLI its progress bars and output-path rules (the tile
//! loop is `tiled`), the GUI its job plumbing and previews.
//!
//! | Module | Stage |
//! |---|---|
//! | [`config`] | settings text → [`StackingSettings`](stacker_core::settings::StackingSettings) |
//! | [`order`] | frame discovery, natural sort, reverse, every N-th |
//! | [`load`] | decode one frame: EXIF orientation, rotation, crop, resize, linear planar |
//! | [`align`] | seed → refine → invert → warp → coverage mask, and the frame-sequence driver |
//! | [`crop`] | the common valid rectangle of the aligned frames |
//! | [`cull`] | focus census, the cull rule (`auto_cull_threshold_pct`) and the decision |
//! | [`fuse`] | `Apex`, `Relief` (guided or multigrid) and `Strata` for whole images and tiles, the `Relief` threshold for tiled callers |
//! | [`compute`] | the CPU or, in a `gpu` build, the GPU for `Apex`, with a CPU fallback |
//! | [`tiling`] | tile grid and apron derived from the settings |
//! | `tiled` | the out-of-core tile loop as callbacks: align into a tile store, order, fuse tile by tile (not on `wasm32`) |
//! | [`output`] | planar linear light → 8/16-bit sRGB samples, as file bytes or files |
//! | [`session`] | the in-memory run (desktop app, browser worker): load, align, cull and sort, fuse, crop, memory estimate |
//!
//! Long-running stages report through [`Progress`] and poll [`Cancel`]; neither
//! trait knows anything about Slint, clap or indicatif.
//!
//! The per-pixel memory figures of every stage are in the crate README (section
//! "Memory and disk: per-pixel figures").
//!
//! # Determinism
//!
//! For the same frames, settings and build, the pipeline's output does not depend
//! on how many threads run or on how they are scheduled. What each source of
//! variation does in the workspace:
//!
//! * **Thread scheduling and thread count: no effect.** Every rayon loop either
//!   writes disjoint output (rows, pixels, whole frames), collects in input order
//!   (`par_iter().map().collect()` keeps the order: frame decoding, per-frame
//!   focus maps and pyramids, descriptor matching), or reduces with an operation
//!   whose result cannot depend on the grouping: `f32::max` over focus values,
//!   and integer sums (the 256-bin histogram of the AKAZE grey conversion). The
//!   one floating-point sum that feeds a decision, the alignment objective in
//!   `stacker_align::refine`, adds per-row partial sums serially in row order, so
//!   the simplex takes the same path for any thread count.
//! * **Iteration order of maps and sets: no effect on any output.** The tile
//!   store's `HashMap` is only looked up by key (draining it in `clear` can change
//!   which of several removal errors is reported first, never a result); model
//!   discovery de-duplicates through a `HashSet` and then sorts by a total key;
//!   directory listings are sorted (natural order, then plain path order) before
//!   use; the retouch history keeps its blocks in a `BTreeMap`.
//! * **Random numbers.** Alignment seeding has one source: the RANSAC sampler, a
//!   `SplitMix64` generator with the fixed seed `RANSAC_SEED` in
//!   `stacker_align::ransac`, created fresh for every estimate, so the same
//!   matches always give the same matrix. The remaining generators are in
//!   training only: the trainer's xorshift (seeded from `--seed` and the global
//!   step) and Burn's global generator (seeded from `--seed`).
//! * **Ties.** Wherever several frames score exactly the same (`Apex` selection,
//!   `Relief` selection, the auto-cull census) the frame that comes first in the
//!   stack wins.
//! * **Vector width and fused multiply-add.** The SIMD paths pick their width
//!   from the compile-time target features (4, 8 or 16 lanes), and every lane
//!   performs the same operations in the same order as the scalar tail, so the
//!   width changes speed only. `mul_add` is exactly rounded with or without
//!   hardware FMA, and plain `a * b + c` is never contracted by the compiler.
//! * **Wall-clock and process identifiers** appear only in names: the CLI tile
//!   directory `stacker_tiles_<pid>_<nanos>_<n>`, the desktop app's temporary
//!   result file `stacker_result_<label>_<ms>.png`, and the date fields of the
//!   output file-name template. They never reach pixel data.
//!
//! What is *not* promised: results across different target features or compiler
//! versions are not compared bit for bit, and the neural mode depends on the
//! backend (`ndarray` or `wgpu`) its weights are run on.

#![warn(missing_docs)]

pub mod align;
/// Where `Apex` runs: the CPU, or the GPU with the CPU behind it.
pub mod compute;
/// Settings text to [`StackingSettings`](stacker_core::settings::StackingSettings).
pub mod config;
/// The common valid rectangle of the aligned frames.
pub mod crop;
/// The focus census and the auto-cull decision.
pub mod cull;
mod error;
pub mod fuse;
/// Decoding one frame into the working representation.
pub mod load;
/// Frame discovery and ordering.
pub mod order;
/// Encoding the fused image, to bytes or to a file.
pub mod output;
/// The in-memory stack sequence: load, align, cull, fuse, crop, memory estimate.
pub mod session;
#[cfg(not(target_arch = "wasm32"))]
pub mod tiled;
/// The tile grid and the apron of an out-of-core run.
pub mod tiling;

pub use error::PipelineError;

/// The coarse stages a run reports progress for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Decoding the frames.
    Load,
    /// Aligning the frames to the reference.
    Align,
    /// The focus census and the auto-cull.
    Cull,
    /// Fusing the frames.
    Fuse,
    /// Encoding and writing the result.
    Save,
}

/// Receives progress from a long-running stage.
///
/// `fraction` is how far the stage is, `0.0` to `1.0`; `label` is a short status
/// text (empty when the stage has nothing new to say). Implementations must be
/// cheap and must not block. The stages call it from the thread that runs them
/// (never from a rayon worker), so it needs no `Sync` bound.
pub trait Progress {
    /// Report that `stage` is `fraction` done, with an optional status `label`.
    fn stage(&self, stage: Stage, fraction: f32, label: &str);
}

/// Polled by long-running stages between units of work (frames, tiles).
pub trait Cancel {
    /// `true` once the caller wants the stage to stop at the next opportunity.
    fn is_cancelled(&self) -> bool;
}

/// A [`Progress`] that discards everything.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProgress;

impl Progress for NoProgress {
    fn stage(&self, _stage: Stage, _fraction: f32, _label: &str) {}
}

/// A [`Cancel`] that never asks to stop.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeverCancel;

impl Cancel for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// `done / total` as a progress fraction (`1.0` when `total` is zero).
#[must_use]
pub fn fraction_of(done: usize, total: usize) -> f32 {
    if total == 0 {
        return 1.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "frame and tile counts are far below 2^24"
    )]
    let fraction = done as f32 / total as f32;
    fraction.clamp(0.0, 1.0)
}
