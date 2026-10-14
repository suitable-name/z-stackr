//! The out-of-core tile loop, as callbacks: align every frame into a tile store,
//! decide the fusion order, fuse tile by tile, hand the result to a sink.
//!
//! The command line, the desktop app's `tile_size` mode, the batch mode and the
//! Python bindings all run a stack that does not fit in memory the same way, so
//! the loop lives here once. What the callers keep is theirs: the CLI its progress
//! bars and output-path rules, the GUI its job plumbing, Python its GIL handling.
//! The loop knows them only through four small seams:
//!
//! | Seam | Trait or type | What it is |
//! |---|---|---|
//! | progress, cancel, log | [`Hooks`] over [`Progress`], [`Cancel`] and [`RunLog`] | polled between frames and tiles, never from a rayon worker |
//! | where frames come from | [`FrameSource`] (decoding) and [`TileFrames`] (padded tiles) | the run decodes with the first, fuses from the second ([`StoreFrames`] is the tile store, [`MemoryFrames`] a list in RAM) |
//! | where the result goes | [`TileSink`] | [`EncodedImage`] (8/16-bit samples) and `PlanarImage<f32>` implement it |
//! | which device | the [`Compute`] and the [`StrataKernels`] the caller passes | CPU by default |
//!
//! [`run_tiled`] is the whole run (decode the reference, align into the store,
//! cull and order, fuse). The pieces are public too ([`plan_run`],
//! [`align_into_store`], [`frame_order`], [`fuse_tiles`]) for a caller that
//! interleaves its own steps; [`fuse_tiles_by`] is the per-tile loop without an
//! engine, which the neural mode of the CLI uses.
//!
//! # Engines
//!
//! * `Apex`: the tile is fused by [`Compute::apex`]; the apron and the tile grid
//!   come from [`crate::tiling`]. On the CPU the tile's frames are folded into one
//!   accumulator pyramid, one frame at a time.
//! * `Relief` (guided): one pass over the tiles collects the whole-image maximum
//!   focus map, the threshold is decided once from it, a second pass fuses every
//!   tile against that value. The CPU folds each frame's focus map into the best
//!   value and the winner as it is computed (no map is kept per frame).
//! * `Relief` with `relief_use_multigrid`: the first pass folds the **interior**
//!   of every tile into whole-image best-focus and winner planes
//!   ([`fuse::DepthVotes`], 6 bytes per pixel), the depth map is solved once on them
//!   ([`Compute::multigrid_depth`]: the GPU solver with the CPU solver behind it;
//!   about 28 bytes per pixel at the peak, no frame in memory then), and the
//!   second pass composites every tile with its window of the depth. There are no
//!   seams. The focus maps are computed on the CPU, one frame at a time.
//! * `Strata`: the tile is fused by the [`StrataKernels`] the caller passes to
//!   [`fuse_tiles_with`], or by the kernels of the [`Compute`] ([`fuse_tiles`]:
//!   the device's when it has them, the CPU's otherwise), with the 101-pixel
//!   apron. The tile's frames are read on demand (two passes, two frames in
//!   memory at a time), not all at once.
//!
//! # Disk and memory
//!
//! The tile store takes `12 · ((T + 2A) / T)²` bytes per source pixel and frame
//! on disk ([`TiledPlan::scratch_bytes`] is the exact figure for a plan); the
//! scratch directory is removed on every exit path ([`ScratchDir`]). Fusing a
//! tile needs one padded tile of every frame in memory
//! ([`TiledPlan::fusion_ram_bytes`]; `Strata` only two), plus the sink's
//! full-image buffer.
//!
//! # Overlapped reading
//!
//! The N padded tiles of a tile are read in parallel (ordered collect), and
//! while tile `t` is fused a second thread reads the frames of tile `t + 1`
//! ([`Prefetch::Auto`], the default, whenever the [`TileFrames`] can be shared
//! between threads, see [`TileFrames::as_shared`], the run has more than one tile
//! and the caller is not a rayon worker). The result is identical to the serial
//! loop ([`Prefetch::Off`]); the price is one more set of tile frames in memory
//! ([`TiledPlan::prefetch_extra_bytes`]). The alignment can decode the next frame
//! on a second thread too: [`align_into_store_overlapped`] and
//! [`run_tiled_overlapped`] for a frame source that is `Send`.
//!
//! The module is native only: it creates files and directories.

use std::{
    borrow::Cow,
    cell::RefCell,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
};

use rayon::prelude::*;
use stacker_algo::{
    relief::{focus::compute_focus_map, multigrid::crop_plane},
    strata::{CpuStrata, FrameSource as StrataFrames, NO_PREVIEW, StrataKernels},
};
use stacker_core::{
    image::PlanarImage,
    memory::{MmapTileManager, TileProvider, extract_tile, paste_interior},
    settings::{PreprocessingSettings, StackingSettings},
};

use crate::{
    Cancel, NeverCancel, NoProgress, PipelineError, Progress, Stage,
    align::{
        AlignModel, AlignedFrame, AlignmentSettings, Fallback, FrameFlow, SequenceError,
        SequenceOutcome, align_sequence_on, align_sequence_overlapped,
    },
    compute::Compute,
    crop::{CommonCoverage, Rect},
    cull::{self, CullRule},
    fraction_of,
    fuse::{self, BoxedStrata},
    output::EncodedImage,
    session::{self, FrameSource},
    tiling::{Geometry, Method, TilePlan, commit_frame, multigrid_whole_image_bytes, plan_tiles},
};

/// The smallest tile edge a run accepts (the settings' own lower bound for
/// `tile_size`). [`plan_run`] and [`fuse_tiles`] take any size; [`run_tiled`]
/// refuses smaller ones.
pub const MIN_TILE_SIZE: usize = 64;

/// A padded tile grid larger than this factor over the image is reported: the
/// scratch disk and the fusion's working set grow with it.
pub const PADDING_WARN_FACTOR: f64 = 4.0;

// ── Hooks ───────────────────────────────────────────────────────────────────

/// Where a run writes its messages.
pub trait RunLog {
    /// Something worth a line in the log.
    fn info(&self, message: &str);

    /// Something the user should know: a fallback, a refused crop, a GPU that
    /// was given up.
    fn warn(&self, message: &str);
}

/// A [`RunLog`] that forwards to `tracing`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TracingLog;

impl RunLog for TracingLog {
    fn info(&self, message: &str) {
        tracing::info!("{message}");
    }

    fn warn(&self, message: &str) {
        tracing::warn!("{message}");
    }
}

/// A [`RunLog`] that discards everything.
#[derive(Debug, Clone, Copy, Default)]
pub struct SilentLog;

impl RunLog for SilentLog {
    fn info(&self, _message: &str) {}

    fn warn(&self, _message: &str) {}
}

/// The callbacks of a run: progress, cancellation and logging.
///
/// Progress is reported as [`Stage::Align`] (frames), [`Stage::Cull`] (the
/// census) and [`Stage::Fuse`] (tiles; a two-pass engine counts both passes).
/// `cancel` is polled between frames and between tiles; a cancelled run returns
/// [`PipelineError::Cancelled`] and leaves nothing behind.
#[derive(Clone, Copy)]
pub struct Hooks<'a> {
    /// Receives the stage fractions.
    pub progress: &'a dyn Progress,
    /// Polled between frames and tiles.
    pub cancel: &'a dyn Cancel,
    /// Receives the messages.
    pub log: &'a dyn RunLog,
}

impl Hooks<'static> {
    /// No progress, no cancellation, no log.
    #[must_use]
    pub const fn silent() -> Self {
        Self {
            progress: &NoProgress,
            cancel: &NeverCancel,
            log: &SilentLog,
        }
    }
}

impl Hooks<'_> {
    fn check_cancel(&self) -> Result<(), PipelineError> {
        if self.cancel.is_cancelled() {
            Err(PipelineError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Counts the units of one stage and reports them.
struct Ticks {
    done: usize,
    total: usize,
}

impl Ticks {
    const fn new(total: usize) -> Self {
        Self { done: 0, total }
    }

    fn tick(&mut self, hooks: &Hooks<'_>, label: &str) {
        self.done += 1;
        hooks
            .progress
            .stage(Stage::Fuse, fraction_of(self.done, self.total), label);
    }
}

// ── Scratch space ───────────────────────────────────────────────────────────

/// Distinguishes several runs started within one process (the monitor mode and the
/// batch mode re-run the pipeline; tests run several at once).
static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A freshly created tile directory that is removed again when the guard goes out
/// of scope.
///
/// The removal happens in `Drop`, so it covers every exit of a run: success,
/// any early `?` return and a panic that unwinds through it. (A process kill or an
/// abort-on-panic build cannot run destructors; the directory then stays.)
#[derive(Debug)]
pub struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    /// Create `stacker_tiles_<pid>_<nanos>_<n>` below `root`.
    ///
    /// # Errors
    /// The I/O error when the directory cannot be created.
    pub fn create(root: &Path) -> std::io::Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let run = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "stacker_tiles_{}_{nanos}_{run}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    /// The directory the tiles are written to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        match std::fs::remove_dir_all(&self.path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(
                dir = %self.path.display(),
                error = %e,
                "could not remove the temporary tile directory; delete it manually"
            ),
        }
    }
}

// ── Frames and sinks ────────────────────────────────────────────────────────

/// Where the fusion reads the padded tile of a frame from.
pub trait TileFrames {
    /// The padded tile `plan.padded` of the frame stored under index `frame`.
    ///
    /// # Errors
    /// The store's error when the tile cannot be read.
    fn tile(&self, frame: usize, plan: &TilePlan) -> Result<PlanarImage<f32>, PipelineError>;

    /// This source as one that may be read from several threads at once, when it
    /// can: the fusion then reads the `N` tiles of a tile in parallel and reads
    /// the next tile ahead (see [`Prefetch`]). The default is `None` (the source
    /// is read serially on the calling thread); a `Sync` implementation returns
    /// `Some(self)`.
    fn as_shared(&self) -> Option<&(dyn TileFrames + Sync)> {
        None
    }
}

/// Whether the fusion reads the frames of the next tile while it fuses the
/// current one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Prefetch {
    /// Read ahead whenever it can: the frames are shareable
    /// ([`TileFrames::as_shared`]), there is more than one tile, the method loads
    /// whole tiles (not `Strata`) and the caller is not a rayon worker thread.
    #[default]
    Auto,
    /// Never: read each tile's frames when its turn comes (in parallel when the
    /// frames are shareable). Uses the least memory.
    Off,
}

/// [`TileFrames`] over a tile store the frames were committed to.
#[derive(Clone, Copy)]
pub struct StoreFrames<'a>(pub &'a MmapTileManager);

impl std::fmt::Debug for StoreFrames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreFrames").finish_non_exhaustive()
    }
}

impl TileFrames for StoreFrames<'_> {
    fn tile(&self, frame: usize, plan: &TilePlan) -> Result<PlanarImage<f32>, PipelineError> {
        Ok(self.0.fetch_tile(frame, &plan.padded)?)
    }

    fn as_shared(&self) -> Option<&(dyn TileFrames + Sync)> {
        Some(self)
    }
}

/// [`TileFrames`] over aligned frames held in memory: the padded tile is cut out
/// of the frame on request.
#[derive(Debug, Clone, Copy)]
pub struct MemoryFrames<'a>(pub &'a [PlanarImage<f32>]);

impl TileFrames for MemoryFrames<'_> {
    fn tile(&self, frame: usize, plan: &TilePlan) -> Result<PlanarImage<f32>, PipelineError> {
        let image = self.0.get(frame).ok_or_else(|| {
            PipelineError::invalid(format!(
                "frame {frame} is not in the stack of {} frames",
                self.0.len()
            ))
        })?;
        let p = &plan.padded;
        Ok(extract_tile(image, p.start_x, p.start_y, p.width, p.height))
    }

    fn as_shared(&self) -> Option<&(dyn TileFrames + Sync)> {
        Some(self)
    }
}

/// Where the fused tiles go.
pub trait TileSink {
    /// Write the interior of the fused padded `tile` into the result.
    /// `interior_offset` is where the interior starts inside `tile`;
    /// `dest = (x, y, width, height)` is its rectangle in the result.
    fn paste(
        &mut self,
        tile: &PlanarImage<f32>,
        interior_offset: (usize, usize),
        dest: (usize, usize, usize, usize),
    );
}

impl TileSink for EncodedImage {
    fn paste(
        &mut self,
        tile: &PlanarImage<f32>,
        interior_offset: (usize, usize),
        dest: (usize, usize, usize, usize),
    ) {
        self.paste_tile(tile, interior_offset, dest);
    }
}

impl TileSink for PlanarImage<f32> {
    fn paste(
        &mut self,
        tile: &PlanarImage<f32>,
        interior_offset: (usize, usize),
        dest: (usize, usize, usize, usize),
    ) {
        let (x, y, width, height) = dest;
        paste_interior(
            self,
            tile,
            interior_offset.0,
            interior_offset.1,
            x,
            y,
            width,
            height,
        );
    }
}

// ── The plan ────────────────────────────────────────────────────────────────

/// The tile size to use: the explicit `requested` size if it is above 0, else
/// the `tile_size` setting. `0` means "no tiles requested" (the stack stays in
/// RAM); the caller decides what that means.
#[must_use]
pub const fn effective_tile_size(requested: usize, settings: &StackingSettings) -> usize {
    if requested > 0 {
        requested
    } else {
        settings.tile_size as usize
    }
}

/// The tile grid of a run: image size, the geometry (tile size and apron) and
/// every cell.
#[derive(Debug, Clone)]
pub struct TiledPlan {
    /// The image size `(width, height)`.
    pub size: (usize, usize),
    /// Tile size (already limited to the image) and apron.
    pub geometry: Geometry,
    /// The cells, row by row.
    pub tiles: Vec<TilePlan>,
}

/// Cut a `size` image into tiles for `method`: the apron comes from what is
/// fused (see [`crate::tiling`]), a tile larger than the image is one tile.
#[must_use]
pub fn plan_run(
    method: Method,
    size: (usize, usize),
    requested_tile: usize,
    settings: &StackingSettings,
) -> TiledPlan {
    let chosen = Geometry::choose(requested_tile, method, settings);
    let tile_size = chosen.tile_size.min(size.0.max(size.1)).max(1);
    let geometry = Geometry {
        tile_size,
        apron: chosen.apron,
    };
    TiledPlan {
        size,
        geometry,
        tiles: plan_tiles(size.0, size.1, tile_size, geometry.apron),
    }
}

fn to_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// Bytes per source pixel aligning into the store takes beyond the chain and the
/// look-ahead: the reference (12), the frame (12) and the refinement working set
/// (15).
const ALIGN_STORE_BYTES_PER_PIXEL: u64 = 39;

impl TiledPlan {
    /// Bytes the tile store takes on disk for `frames` frames: every padded
    /// tile, three `f32` planes.
    #[must_use]
    pub fn scratch_bytes(&self, frames: usize) -> u64 {
        let pixels: u64 = self
            .tiles
            .iter()
            .map(|t| to_u64(t.padded.width) * to_u64(t.padded.height))
            .sum();
        pixels
            .saturating_mul(to_u64(crate::tiling::SCRATCH_BYTES_PER_PIXEL))
            .saturating_mul(to_u64(frames))
    }

    /// Bytes of RAM fusing the largest tile of `frames` frames takes, plus the
    /// whole-image planes the multigrid `Relief` adds (see [`crate::tiling`]). The
    /// sink's full-image buffer is not included.
    #[must_use]
    pub fn fusion_ram_bytes(
        &self,
        method: Method,
        frames: usize,
        settings: &StackingSettings,
    ) -> u64 {
        let tile = to_u64(self.geometry.fusion_bytes(method, frames));
        let whole = if method == Method::Relief && settings.relief_use_multigrid {
            to_u64(multigrid_whole_image_bytes(self.size.0 * self.size.1))
        } else {
            0
        };
        tile.saturating_add(whole)
    }

    /// The RAM the read-ahead of the next tile adds to [`Self::fusion_ram_bytes`]
    /// when the fusion runs with [`Prefetch::Auto`] and the frames are shareable:
    /// the padded tile of every frame once more. Zero for `Strata`.
    #[must_use]
    pub fn prefetch_extra_bytes(&self, method: Method, frames: usize) -> u64 {
        if self.tiles.len() < 2 {
            return 0;
        }
        to_u64(self.geometry.prefetch_bytes(method, frames))
    }

    /// Bytes of RAM aligning into the store takes at its peak, whatever the
    /// frame count: the reference (12), the frame in flight (12) and the
    /// refinement working set (about 15: two luma pyramids and a warped plane),
    /// 39 in all per source pixel; `chain_alignment` adds the previous frame with
    /// its pyramid (18); a seed source (AKAZE, network) adds the look-ahead's
    /// frames, and `overlapped` ([`align_into_store_overlapped`]) the frame the
    /// decode thread holds (`align::lookahead_extra_bytes`).
    #[must_use]
    pub fn align_ram_bytes(&self, settings: &StackingSettings, overlapped: bool) -> u64 {
        let (width, height) = self.size;
        let pixels = to_u64(width).saturating_mul(to_u64(height));
        let mut per_pixel = ALIGN_STORE_BYTES_PER_PIXEL;
        if settings.chain_alignment {
            per_pixel += session::CHAIN_WORKING_BYTES;
        }
        pixels
            .saturating_mul(per_pixel)
            .saturating_add(session::lookahead_bytes(
                width, height, settings, overlapped,
            ))
    }

    /// A warning when the apron makes a stored tile several times its interior.
    #[must_use]
    pub fn padding_warning(&self, method: Method) -> Option<String> {
        (self.geometry.padding_factor() > PADDING_WARN_FACTOR).then(|| {
            format!(
                "each stored tile is {:.1}x its interior because the {method:?} apron ({} px) is \
                 large; a larger tile size reduces the scratch disk and memory overhead{}",
                self.geometry.padding_factor(),
                self.geometry.apron,
                if method == Method::Apex {
                    " or fewer pyramid levels"
                } else {
                    ""
                },
            )
        })
    }
}

// ── Alignment into the store ────────────────────────────────────────────────

/// Streams frames `1..N` from a [`FrameSource`] into the tile store through the
/// shared alignment driver: decode, size check, (the driver aligns), commit tile
/// by tile.
///
/// The warnings go to `warn`: the log itself for the serial driver, a list the
/// caller logs afterwards for the overlapped one (the log is not shareable
/// between threads, so it cannot be a field of a flow that must be `Send`).
struct StoreFlow<'a, S: ?Sized, W> {
    source: &'a mut S,
    preprocessing: &'a PreprocessingSettings,
    size: (usize, usize),
    store: &'a mut MmapTileManager,
    tiles: &'a [TilePlan],
    warn: W,
}

impl<S: FrameSource + ?Sized, W: FnMut(&str)> FrameFlow<PipelineError> for StoreFlow<'_, S, W> {
    fn load(&mut self, index: usize) -> Result<PlanarImage<f32>, PipelineError> {
        let frame = self.source.decode(index, self.preprocessing)?;
        let found = (frame.width, frame.height);
        if found != self.size {
            return Err(PipelineError::invalid(format!(
                "{} is {}x{} px but the reference frame {} is {}x{} px; all frames must have the \
                 same size (check the pre-processing settings).",
                self.source.name(index),
                found.0,
                found.1,
                self.source.name(0),
                self.size.0,
                self.size.1,
            )));
        }
        Ok(frame)
    }

    fn deliver(
        &mut self,
        index: usize,
        aligned: AlignedFrame,
        _common: &[bool],
    ) -> Result<(), PipelineError> {
        let name = self.source.name(index);
        match aligned.fallback {
            Some(Fallback::Seed) => (self.warn)(&format!(
                "{name}: alignment could not be refined; the starting estimate was used"
            )),
            Some(Fallback::Unaligned) => (self.warn)(&format!(
                "{name}: the frame could not be warped and is used unaligned"
            )),
            None => {}
        }
        if aligned.covered_fraction() < 0.5 {
            (self.warn)(&format!(
                "{name}: the frame covers less than half of the reference after alignment; its \
                 alignment probably failed"
            ));
        }
        Ok(commit_frame(self.store, index, &aligned.frame, self.tiles)?)
    }

    fn reload(&mut self, index: usize) -> Option<Result<PlanarImage<f32>, PipelineError>> {
        // The neural pre-pass reads every frame once more: decode it again.
        Some(self.load(index))
    }
}

/// Commit the `reference` (frame 0) to `store`, then decode, align and commit
/// frames `1..source.len()` one at a time, so a single aligned frame is in RAM
/// (about 39 bytes per pixel, whatever the frame count).
///
/// Returns the intersection of every frame's warp coverage.
///
/// # Errors
/// [`PipelineError::Cancelled`] when the hooks' cancel fires, the decoder's error
/// for a frame that cannot be read, [`PipelineError::Invalid`] for a frame of
/// another size than the reference, the store's error when a tile cannot be
/// written.
pub fn align_into_store<S: FrameSource + ?Sized>(
    source: &mut S,
    reference: &PlanarImage<f32>,
    settings: &StackingSettings,
    tiles: &[TilePlan],
    store: &mut MmapTileManager,
    compute: &mut Compute,
    hooks: &Hooks<'_>,
) -> Result<CommonCoverage, PipelineError> {
    align_into_store_with_alignment(
        source,
        reference,
        settings,
        &AlignmentSettings::from(settings),
        tiles,
        store,
        compute,
        hooks,
    )
}

/// [`align_into_store`] with a neural alignment `model` (for
/// `alignment_mode = "neural"`; `None` is the plain call). The store flow
/// re-decodes frames for the network's pre-pass (`FrameFlow::reload`).
///
/// # Errors
/// As [`align_into_store`].
#[expect(
    clippy::too_many_arguments,
    reason = "align_into_store's seven inputs and the model"
)]
pub fn align_into_store_with_model<S: FrameSource + ?Sized>(
    source: &mut S,
    reference: &PlanarImage<f32>,
    settings: &StackingSettings,
    model: Option<AlignModel>,
    tiles: &[TilePlan],
    store: &mut MmapTileManager,
    compute: &mut Compute,
    hooks: &Hooks<'_>,
) -> Result<CommonCoverage, PipelineError> {
    align_into_store_with_alignment(
        source,
        reference,
        settings,
        &AlignmentSettings::from(settings).with_model(model),
        tiles,
        store,
        compute,
        hooks,
    )
}

/// [`align_into_store`] with the alignment settings of the caller.
///
/// # Errors
/// As [`align_into_store`].
#[expect(
    clippy::too_many_arguments,
    reason = "align_into_store's seven inputs and the alignment settings"
)]
pub fn align_into_store_with_alignment<S: FrameSource + ?Sized>(
    source: &mut S,
    reference: &PlanarImage<f32>,
    settings: &StackingSettings,
    alignment: &AlignmentSettings,
    tiles: &[TilePlan],
    store: &mut MmapTileManager,
    compute: &mut Compute,
    hooks: &Hooks<'_>,
) -> Result<CommonCoverage, PipelineError> {
    commit_frame(store, 0, reference, tiles)?;
    let count = source.len();
    if count < 2 {
        return Ok(CommonCoverage::new(reference.width, reference.height));
    }
    hooks
        .log
        .info(&format!("alignment on {}", compute.describe()));
    let mut flow = StoreFlow {
        source,
        preprocessing: &settings.preprocessing,
        size: (reference.width, reference.height),
        store,
        tiles,
        warn: |message: &str| hooks.log.warn(message),
    };
    let outcome = align_sequence_on(
        reference,
        count,
        alignment,
        hooks.cancel,
        hooks.progress,
        &mut flow,
        compute,
    );
    store_alignment_result(outcome, compute, hooks)
}

/// [`align_into_store`] with the next frame decoded on a second thread.
///
/// The decode runs while the current frame is aligned
/// (`align::align_sequence_overlapped`): the same
/// coverage and the same tiles, bit for bit, for a frame source that is `Send`.
/// One more decoded frame is in memory ([`TiledPlan::align_ram_bytes`] with
/// `overlapped`). The per-frame warnings reach the log when the pass is over,
/// not while it runs.
///
/// # Errors
/// As [`align_into_store`].
pub fn align_into_store_overlapped<S: FrameSource + Send + ?Sized>(
    source: &mut S,
    reference: &PlanarImage<f32>,
    settings: &StackingSettings,
    tiles: &[TilePlan],
    store: &mut MmapTileManager,
    compute: &mut Compute,
    hooks: &Hooks<'_>,
) -> Result<CommonCoverage, PipelineError> {
    commit_frame(store, 0, reference, tiles)?;
    let count = source.len();
    if count < 2 {
        return Ok(CommonCoverage::new(reference.width, reference.height));
    }
    hooks.log.info(&format!(
        "alignment on {}, the next frame decoded ahead",
        compute.describe()
    ));
    let mut warnings: Vec<String> = Vec::new();
    let outcome = {
        let mut flow = StoreFlow {
            source,
            preprocessing: &settings.preprocessing,
            size: (reference.width, reference.height),
            store,
            tiles,
            warn: |message: &str| warnings.push(message.to_owned()),
        };
        align_sequence_overlapped(
            reference,
            count,
            &AlignmentSettings::from(settings),
            hooks.cancel,
            hooks.progress,
            &mut flow,
            compute,
        )
    };
    for warning in &warnings {
        hooks.log.warn(warning);
    }
    store_alignment_result(outcome, compute, hooks)
}

/// The coverage of a finished alignment pass, or its error; the device notice of
/// `compute`, if any, goes to the log first.
fn store_alignment_result(
    outcome: Result<SequenceOutcome, SequenceError<PipelineError>>,
    compute: &mut Compute,
    hooks: &Hooks<'_>,
) -> Result<CommonCoverage, PipelineError> {
    if let Some(notice) = compute.take_notice() {
        hooks.log.warn(&notice);
    }
    match outcome {
        Ok(outcome) => Ok(outcome.coverage),
        Err(SequenceError::Flow(error)) => Err(error),
        Err(SequenceError::Cancelled) => Err(PipelineError::Cancelled),
    }
}

// ── Cull and order ──────────────────────────────────────────────────────────

/// The frames to fuse, in fusion order, from the tile store.
///
/// Without `auto_cull` and `sort_by_sharpness`, or with fewer than two frames,
/// every frame in stack order. Otherwise the focus census is built tile by tile
/// and decided like [`crate::session::cull`] decides an in-memory stack:
/// `auto_cull` drops the frames that win too little of the detail (the threshold
/// is `auto_cull_threshold_pct`, or the frame-count-scaled rule when that is 0),
/// `sort_by_sharpness` fuses the survivors in census order and stack order
/// otherwise.
///
/// # Errors
/// The store's error when a tile cannot be read.
pub fn frame_order(
    settings: &StackingSettings,
    plan: &TiledPlan,
    store: &MmapTileManager,
    n_frames: usize,
    hooks: &Hooks<'_>,
) -> Result<Vec<usize>, PipelineError> {
    let (cull_on, sort) = (settings.auto_cull, settings.sort_by_sharpness);
    if (!cull_on && !sort) || n_frames < 2 {
        return Ok((0..n_frames).collect());
    }
    let (width, height) = plan.size;
    let census = cull::census_from_store(store, &plan.tiles, n_frames, width, height)?;
    let rule = if cull_on {
        CullRule::from_settings(settings)
    } else {
        CullRule::Fixed(0.0)
    };
    let decision = cull::plan_with(&census, width, height, rule);
    for warning in &decision.warnings {
        hooks.log.warn(&format!("auto-cull: {warning}"));
    }
    hooks.log.info(&format!(
        "auto-cull: kept {:?}, dropped {:?} (threshold {:.2} % of {} detail pixels)",
        decision.order, decision.dropped, decision.effective_pct, census.detail_pixels
    ));
    Ok(decision.fusion_order(sort))
}

// ── Fusion ──────────────────────────────────────────────────────────────────

/// Everything the tiled fusion needs to know about the stack.
#[derive(Debug, Clone, Copy)]
pub struct FusePlan<'a> {
    /// The engine; [`Method::Neural`] has no engine here (see [`fuse_tiles_by`]).
    pub method: Method,
    /// The settings the engine reads.
    pub settings: &'a StackingSettings,
    /// The image size `(width, height)`.
    pub size: (usize, usize),
    /// The tile grid.
    pub tiles: &'a [TilePlan],
    /// The frames to fuse, as indices into the frame store, in fusion order.
    pub order: &'a [usize],
}

impl<'a> FusePlan<'a> {
    /// The plan of a [`TiledPlan`] fused with `order`.
    #[must_use]
    pub fn new(
        method: Method,
        settings: &'a StackingSettings,
        plan: &'a TiledPlan,
        order: &'a [usize],
    ) -> Self {
        Self {
            method,
            settings,
            size: plan.size,
            tiles: &plan.tiles,
            order,
        }
    }
}

/// What fusing one tile returns: the fused padded tile.
type FusedTile = Result<PlanarImage<f32>, PipelineError>;

/// The per-tile loop without an engine: for every tile load the padded tile of
/// every frame in `plan.order`, hand them to `fuse`, and paste the interior of
/// what it returns into `sink`.
///
/// `fuse` receives the tile's cell and its frames by value (they are freed when
/// it returns, before the result is pasted). Progress is one [`Stage::Fuse`] step
/// per tile; `cancel` is polled before each.
///
/// The frames of a tile are read in parallel, and the next tile's are read ahead
/// while `fuse` runs, when `frames` is shareable ([`Prefetch::Auto`]; see
/// [`fuse_tiles_by_prefetch`]).
///
/// # Errors
/// [`PipelineError::Cancelled`], the frames' error, or whatever `fuse` returns.
pub fn fuse_tiles_by(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    fuse: impl FnMut(&TilePlan, Vec<PlanarImage<f32>>) -> FusedTile,
) -> Result<(), PipelineError> {
    fuse_tiles_by_prefetch(plan, frames, sink, hooks, Prefetch::Auto, fuse)
}

/// [`fuse_tiles_by`] with an explicit read-ahead policy. The result does not
/// depend on `prefetch`; only the memory (one more set of tile frames with
/// [`Prefetch::Auto`]) and the time do.
///
/// # Errors
/// As [`fuse_tiles_by`].
pub fn fuse_tiles_by_prefetch(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    prefetch: Prefetch,
    mut fuse: impl FnMut(&TilePlan, Vec<PlanarImage<f32>>) -> FusedTile,
) -> Result<(), PipelineError> {
    let mut ticks = Ticks::new(plan.tiles.len());
    for_each_tile(plan, frames, sink, hooks, &mut ticks, prefetch, &mut fuse)
}

/// The padded tile of every frame of `order`, in `order`, read one after the other.
fn load_tile_frames_serial(
    frames: &dyn TileFrames,
    order: &[usize],
    tile: &TilePlan,
) -> Result<Vec<PlanarImage<f32>>, PipelineError> {
    order
        .iter()
        .map(|&frame| frames.tile(frame, tile))
        .collect()
}

/// The padded tile of every frame of `order`, in `order`, read in parallel
/// (the collect keeps the order, so the result is that of the serial read).
fn load_tile_frames_parallel(
    frames: &(dyn TileFrames + Sync),
    order: &[usize],
    tile: &TilePlan,
) -> Result<Vec<PlanarImage<f32>>, PipelineError> {
    let loaded: Vec<Result<PlanarImage<f32>, PipelineError>> = order
        .par_iter()
        .map(|&frame| frames.tile(frame, tile))
        .collect();
    loaded.into_iter().collect()
}

/// The tile's frames, in parallel when `frames` can be shared.
fn load_tile_frames(
    frames: &dyn TileFrames,
    order: &[usize],
    tile: &TilePlan,
) -> Result<Vec<PlanarImage<f32>>, PipelineError> {
    frames.as_shared().map_or_else(
        || load_tile_frames_serial(frames, order, tile),
        |shared| load_tile_frames_parallel(shared, order, tile),
    )
}

/// Paste the interior of a fused padded tile into `sink`.
fn paste_tile(sink: &mut dyn TileSink, tile: &TilePlan, fused: &PlanarImage<f32>) {
    let c = &tile.interior;
    sink.paste(
        fused,
        tile.interior_offset,
        (c.start_x, c.start_y, c.width, c.height),
    );
}

/// For every tile in turn: poll the cancel, make the fused padded tile with
/// `produce`, paste its interior into `sink`, count one tick. For an engine that
/// reads its frames itself (`Strata`).
fn run_tiles(
    plan: &FusePlan<'_>,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    ticks: &mut Ticks,
    mut produce: impl FnMut(&TilePlan) -> FusedTile,
) -> Result<(), PipelineError> {
    for tile in plan.tiles {
        hooks.check_cancel()?;
        let fused = produce(tile)?;
        paste_tile(sink, tile, &fused);
        ticks.tick(hooks, "");
    }
    Ok(())
}

/// What a tile loop does with the frames of one tile.
type TileBody<'b> = dyn FnMut(&TilePlan, Vec<PlanarImage<f32>>) -> TileStep + 'b;

/// The outcome of one tile of a loop.
type TileStep = Result<(), PipelineError>;

/// For every tile in turn: poll the cancel, load the padded tile of every frame in
/// `plan.order` (in parallel, and ahead of time with [`Prefetch::Auto`] when the
/// frames can be shared) and hand them to `body`.
fn each_tile_frames(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    hooks: &Hooks<'_>,
    prefetch: Prefetch,
    body: &mut TileBody<'_>,
) -> Result<(), PipelineError> {
    // A caller that is itself a rayon worker must not block waiting for a reader
    // thread whose parallel reads need that worker (a one-thread pool would
    // deadlock): it reads serially in its own turn.
    if prefetch == Prefetch::Auto
        && plan.tiles.len() > 1
        && rayon::current_thread_index().is_none()
        && let Some(shared) = frames.as_shared()
    {
        return each_tile_frames_prefetched(plan, shared, hooks, body);
    }
    for tile in plan.tiles {
        hooks.check_cancel()?;
        let tile_frames = load_tile_frames(frames, plan.order, tile)?;
        body(tile, tile_frames)?;
    }
    Ok(())
}

/// [`each_tile_frames`] with a second thread that reads the frames of tile `t + 1`
/// while `body` works on tile `t`. The two threads meet at a rendezvous channel,
/// so at most one tile ahead is held. The tiles arrive in grid order, as the loop
/// consumes them, and the reads are the same ones the serial loop makes: the
/// result does not depend on the timing.
fn each_tile_frames_prefetched(
    plan: &FusePlan<'_>,
    frames: &(dyn TileFrames + Sync),
    hooks: &Hooks<'_>,
    body: &mut TileBody<'_>,
) -> Result<(), PipelineError> {
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::sync_channel::<LoadedTile>(0);
        let (tiles, order, stopper) = (plan.tiles, plan.order, &stop);
        scope.spawn(move || {
            for tile in tiles {
                if stopper.load(Ordering::Relaxed) {
                    break;
                }
                let loaded = load_tile_frames_parallel(frames, order, tile);
                let failed = loaded.is_err();
                if sender.send(loaded).is_err() || failed {
                    break;
                }
            }
        });
        let result = consume_tiles(plan, &receiver, hooks, body);
        stop.store(true, Ordering::Relaxed);
        // Closing the channel ends a reader that is blocked handing a tile over.
        drop(receiver);
        result
    })
}

/// What the reader thread sends: the frames of one tile, or why they could not be
/// read.
type LoadedTile = Result<Vec<PlanarImage<f32>>, PipelineError>;

/// The consuming half of [`each_tile_frames_prefetched`]: the tiles in grid order
/// from the reader, each handed to `body`.
fn consume_tiles(
    plan: &FusePlan<'_>,
    receiver: &mpsc::Receiver<LoadedTile>,
    hooks: &Hooks<'_>,
    body: &mut TileBody<'_>,
) -> TileStep {
    for tile in plan.tiles {
        hooks.check_cancel()?;
        let tile_frames = receiver.recv().map_err(|_| {
            PipelineError::invalid("the tile reader stopped before the last tile")
        })??;
        body(tile, tile_frames)?;
    }
    Ok(())
}

/// Fuse every tile: [`each_tile_frames`] with `fuse` making the padded tile and the
/// interior pasted into `sink`, one tick per tile.
fn for_each_tile(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    ticks: &mut Ticks,
    prefetch: Prefetch,
    fuse: &mut dyn FnMut(&TilePlan, Vec<PlanarImage<f32>>) -> FusedTile,
) -> Result<(), PipelineError> {
    each_tile_frames(plan, frames, hooks, prefetch, &mut |tile, tile_frames| {
        let fused = fuse(tile, tile_frames)?;
        paste_tile(sink, tile, &fused);
        ticks.tick(hooks, "");
        Ok(())
    })
}

/// Fuse the stack tile by tile with the kernels the `compute` handle offers.
///
/// For `Strata` (see [`fuse_tiles_with`]) these are the device's when the handle has them, the
/// CPU's otherwise. After a `Strata` run the handle checks the kernels
/// ([`Compute::check_strata_kernels`]), so a device fault is given up and shows in
/// `compute.take_notice()`; the callers read the notice themselves.
///
/// # Errors
/// As [`fuse_tiles_with`].
pub fn fuse_tiles(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    compute: &mut Compute,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
) -> Result<(), PipelineError> {
    fuse_tiles_prefetch(plan, frames, compute, sink, hooks, Prefetch::Auto)
}

/// [`fuse_tiles`] with an explicit read-ahead policy (see [`Prefetch`]); the
/// result does not depend on it.
///
/// # Errors
/// As [`fuse_tiles_with`].
pub fn fuse_tiles_prefetch(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    compute: &mut Compute,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    prefetch: Prefetch,
) -> Result<(), PipelineError> {
    if plan.method != Method::Strata {
        return fuse_tiles_with_prefetch(
            plan,
            frames,
            compute,
            &mut CpuStrata,
            sink,
            hooks,
            prefetch,
        );
    }
    let mut kernels = BoxedStrata(compute.strata_kernels());
    let result =
        fuse_tiles_with_prefetch(plan, frames, compute, &mut kernels, sink, hooks, prefetch);
    compute.check_strata_kernels();
    result
}

/// Fuse the stack tile by tile with the engine `plan.method` names.
///
/// The interiors are written into `sink`. `compute` serves `Apex`, the guided `Relief` and the
/// multigrid depth solve; `kernels` serves `Strata` (what it declines runs on the
/// CPU).
///
/// The result equals the whole-image fusion of the same frames: up to rounding for
/// `Strata` and the guided `Relief`, within the pyramid's tolerance for `Apex`,
/// exactly for the multigrid `Relief` (its depth is solved on whole-image planes).
///
/// # Errors
/// [`PipelineError::Cancelled`] when the hooks' cancel fires; the frames' error
/// when a tile cannot be read; [`PipelineError::Invalid`] for an empty `order`,
/// the neural method, or a multigrid run over more than 65 536 frames.
pub fn fuse_tiles_with<K: StrataKernels>(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    compute: &mut Compute,
    kernels: &mut K,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
) -> Result<(), PipelineError> {
    fuse_tiles_with_prefetch(plan, frames, compute, kernels, sink, hooks, Prefetch::Auto)
}

/// A frame source for `Strata`: the padded tile of the `index`-th frame of the
/// fusion order, read on demand. The algorithm asks for every frame twice, in
/// order, and holds at most two at a time, so a tile is fused with two tile
/// frames in memory, not `N`. A tile that cannot be read is recorded and
/// replaced by an all-zero tile so the call can finish; the error is returned
/// afterwards.
struct TileStream<'a> {
    frames: &'a dyn TileFrames,
    order: &'a [usize],
    tile: &'a TilePlan,
    failure: RefCell<Option<PipelineError>>,
}

impl StrataFrames for TileStream<'_> {
    fn len(&self) -> usize {
        self.order.len()
    }

    fn frame(&self, index: usize) -> Cow<'_, PlanarImage<f32>> {
        match self.frames.tile(self.order[index], self.tile) {
            Ok(image) => Cow::Owned(image),
            Err(error) => {
                self.failure.borrow_mut().get_or_insert(error);
                Cow::Owned(PlanarImage::new(
                    self.tile.padded.width,
                    self.tile.padded.height,
                ))
            }
        }
    }
}

/// [`fuse_tiles_with`] with an explicit read-ahead policy (see [`Prefetch`]); the
/// result does not depend on it. `Strata` reads on demand and never reads ahead.
///
/// # Errors
/// As [`fuse_tiles_with`].
pub fn fuse_tiles_with_prefetch<K: StrataKernels>(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    compute: &mut Compute,
    kernels: &mut K,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    prefetch: Prefetch,
) -> Result<(), PipelineError> {
    if plan.order.is_empty() {
        return Err(PipelineError::invalid("There are no frames to fuse."));
    }
    let settings = plan.settings;
    match plan.method {
        Method::Apex => {
            hooks
                .log
                .info(&format!("Apex fusion on {}", compute.describe()));
            let mut ticks = Ticks::new(plan.tiles.len());
            for_each_tile(
                plan,
                frames,
                sink,
                hooks,
                &mut ticks,
                prefetch,
                &mut |_, tile_frames| Ok(compute.apex(&tile_frames, settings)),
            )?;
        }
        Method::Relief if settings.relief_use_multigrid => {
            fuse_multigrid(plan, frames, compute, sink, hooks, prefetch)?;
        }
        Method::Relief => fuse_guided(plan, frames, compute, sink, hooks, prefetch)?,
        Method::Strata => {
            hooks.log.info("Strata fusion");
            let mut ticks = Ticks::new(plan.tiles.len());
            run_tiles(plan, sink, hooks, &mut ticks, |tile| {
                let stream = TileStream {
                    frames,
                    order: plan.order,
                    tile,
                    failure: RefCell::new(None),
                };
                let fused = fuse::strata_on(
                    &mut *kernels,
                    &stream,
                    settings,
                    || {},
                    NO_PREVIEW,
                    |_: &PlanarImage<f32>| {},
                );
                stream.failure.take().map_or(Ok(fused), Err)
            })?;
        }
        Method::Neural => {
            return Err(PipelineError::invalid(
                "The neural mode has its own tile loop: use fuse_tiles_by.",
            ));
        }
    }
    Ok(())
}

/// Copy the interior of a tile's single-plane `local` map (`local_width` wide)
/// into the whole-image `whole` map.
fn copy_interior(
    whole: &mut [f32],
    whole_width: usize,
    local: &[f32],
    local_width: usize,
    tile: &TilePlan,
) {
    let (ix, iy) = tile.interior_offset;
    let c = &tile.interior;
    for row in 0..c.height {
        let src = (iy + row) * local_width + ix;
        let dst = (c.start_y + row) * whole_width + c.start_x;
        whole[dst..dst + c.width].copy_from_slice(&local[src..src + c.width]);
    }
}

/// Guided `Relief`: the whole-image threshold from a pass over the tiles, then
/// every tile against that one value.
fn fuse_guided(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    compute: &mut Compute,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    prefetch: Prefetch,
) -> Result<(), PipelineError> {
    let settings = plan.settings;
    let (width, height) = plan.size;
    hooks
        .log
        .info(&format!("Relief fusion on {}", compute.describe()));
    let mut ticks = Ticks::new(2 * plan.tiles.len());

    let mut whole = vec![f32::NEG_INFINITY; width * height];
    let radius = settings.relief_estimation_radius as usize;
    each_tile_frames(plan, frames, hooks, prefetch, &mut |tile, tile_frames| {
        let max = compute.relief_focus_max(&tile_frames, radius);
        drop(tile_frames);
        copy_interior(&mut whole, width, &max, tile.padded.width, tile);
        ticks.tick(hooks, "");
        Ok(())
    })?;
    let threshold = fuse::relief_threshold(&whole, settings);
    drop(whole);
    hooks.log.info(&format!(
        "Relief threshold over the whole image: contrast_pct {:.4}, value {:.6e}, auto_detect {}",
        threshold.contrast_pct, threshold.value, settings.relief_auto_detect
    ));

    for_each_tile(
        plan,
        frames,
        sink,
        hooks,
        &mut ticks,
        prefetch,
        &mut |_, tile_frames| Ok(compute.relief_tile(&tile_frames, settings, threshold.value)),
    )
}

/// Multigrid `Relief`: winner planes from the tile interiors, one solve (on
/// `compute`, the GPU solver with the CPU solver behind it), one composite per
/// tile.
fn fuse_multigrid(
    plan: &FusePlan<'_>,
    frames: &dyn TileFrames,
    compute: &mut Compute,
    sink: &mut dyn TileSink,
    hooks: &Hooks<'_>,
    prefetch: Prefetch,
) -> Result<(), PipelineError> {
    let settings = plan.settings;
    let (width, height) = plan.size;
    if plan.order.len() > fuse::MAX_MULTIGRID_FRAMES {
        return Err(PipelineError::invalid(format!(
            "The multigrid depth map handles at most {} frames, but the stack has {}.",
            fuse::MAX_MULTIGRID_FRAMES,
            plan.order.len()
        )));
    }
    hooks
        .log
        .info("Relief fusion with the multigrid solver (focus maps on the CPU)");
    let mut ticks = Ticks::new(2 * plan.tiles.len());

    let radius = settings.relief_estimation_radius as usize;
    let mut votes = fuse::DepthVotes::new(width, height);
    for tile in plan.tiles {
        hooks.check_cancel()?;
        // One frame's padded tile at a time: the votes need only its focus map.
        let mut local = fuse::DepthVotes::new(tile.padded.width, tile.padded.height);
        for (position, &frame) in plan.order.iter().enumerate() {
            let padded = frames.tile(frame, tile)?;
            local.fold_frame(position, &compute_focus_map(&padded, radius));
        }
        let c = &tile.interior;
        votes.merge_window(
            &local,
            tile.interior_offset,
            (c.start_x, c.start_y, c.width, c.height),
        );
        ticks.tick(hooks, "");
    }

    hooks.check_cancel()?;
    let threshold = votes.threshold(settings);
    hooks.log.info(&format!(
        "Relief threshold over the whole image: contrast_pct {:.4}, value {:.6e}; solving the \
         depth map",
        threshold.contrast_pct, threshold.value
    ));
    let depth = votes.depth_on(compute, threshold.value, settings);
    drop(votes);

    for_each_tile(
        plan,
        frames,
        sink,
        hooks,
        &mut ticks,
        prefetch,
        &mut |tile, tile_frames| {
            let p = &tile.padded;
            let window = crop_plane(&depth, width, (p.start_x, p.start_y), (p.width, p.height));
            Ok(fuse::relief_multigrid_tile(&tile_frames, &window))
        },
    )
}

// ── The whole run ───────────────────────────────────────────────────────────

/// What a run is asked to do.
#[derive(Debug, Clone, Copy)]
pub struct TiledJob<'a> {
    /// The engine: `Apex`, `Relief` or `Strata`.
    pub method: Method,
    /// The settings every stage reads.
    pub settings: &'a StackingSettings,
    /// Interior size of a full tile, at least [`MIN_TILE_SIZE`] (see
    /// [`effective_tile_size`]).
    pub tile_size: usize,
    /// The directory the scratch directory is created in (the system temporary
    /// directory in production).
    pub scratch_root: &'a Path,
}

/// What a run produced.
#[derive(Debug)]
pub struct TiledOutput<K> {
    /// The sink, filled with the fused image (not yet cropped).
    pub sink: K,
    /// The image size `(width, height)`.
    pub size: (usize, usize),
    /// The frames that were fused, in fusion order (positions in the source).
    pub order: Vec<usize>,
    /// The crop to apply to the result: the common valid area of the aligned
    /// frames when `crop_to_common_area` is on and it removes something.
    pub crop: Option<Rect>,
    /// The coverage of every aligned frame, intersected.
    pub coverage: CommonCoverage,
}

/// The crop `settings` ask for: with `crop_to_common_area`, the common valid area
/// of the aligned frames (refused when it is below half the frame, see
/// [`CommonCoverage::crop`]); `None` when nothing is to be cut.
///
/// `resize_cropped_to_original` is applied after the crop: for a planar or an
/// encoded sink by [`TiledOutput::finish`], otherwise by the caller to the size
/// [`TiledOutput::resize_target`] names.
#[must_use]
pub fn crop_rect(
    coverage: &CommonCoverage,
    settings: &StackingSettings,
    size: (usize, usize),
    name_of: impl Fn(usize) -> String,
) -> Option<Rect> {
    coverage
        .crop_when(settings.crop_to_common_area, name_of)
        .needed(size.0, size.1)
}

impl<K> TiledOutput<K> {
    /// The size the cropped result is to be resampled to: the full canvas, when a
    /// crop removed something and `resize_cropped_to_original` is on; `None`
    /// otherwise.
    #[must_use]
    pub fn resize_target(&self, settings: &StackingSettings) -> Option<(usize, usize)> {
        (self.crop.is_some() && settings.resize_cropped_to_original).then_some(self.size)
    }
}

impl TiledOutput<PlanarImage<f32>> {
    /// The finished planar result: cropped, and resized back to the canvas with
    /// `resize_cropped_to_original` ([`session::crop_result_for`]).
    #[must_use]
    pub fn finish(self, settings: &StackingSettings) -> PlanarImage<f32> {
        session::crop_result_for(self.sink, self.crop, settings)
    }
}

impl TiledOutput<EncodedImage> {
    /// The finished encoded result: cropped, and resized back to the canvas with
    /// `resize_cropped_to_original` ([`EncodedImage::resize`], the encoded-sample
    /// counterpart of [`session::crop_result_for`]).
    #[must_use]
    pub fn finish(self, settings: &StackingSettings) -> EncodedImage {
        let target = self.resize_target(settings);
        let cropped = match self.crop {
            Some(rect) => self.sink.crop(rect),
            None => self.sink,
        };
        match target {
            Some((width, height)) => cropped.resize(width, height),
            None => cropped,
        }
    }
}

/// Run a whole stack out of core: decode the reference frame, align every frame
/// into a scratch tile store, decide the fusion order, fuse tile by tile into the
/// sink `make_sink(width, height)` returns.
///
/// The scratch directory is gone when this returns, whatever the outcome.
/// `compute` serves the alignment and the `Apex` and `Relief` fusions (the CPU
/// unless it has a device; a device that fails part-way is given up and the log
/// says why). `Strata` runs on the kernels `compute` offers
/// ([`Compute::strata_kernels`], checked afterwards); use the pieces of this
/// module ([`plan_run`], [`align_into_store`], [`frame_order`],
/// [`fuse_tiles_with`]) to pass other kernels. The fusion reads tiles in parallel
/// and ahead ([`Prefetch::Auto`]).
///
/// # Errors
/// [`PipelineError::Cancelled`]; [`PipelineError::Invalid`] for no frame, a tile
/// size below [`MIN_TILE_SIZE`], the neural method or a frame of another size than
/// the reference; the decoder's, the alignment's and the store's errors.
pub fn run_tiled<S, K, M>(
    source: &mut S,
    job: &TiledJob<'_>,
    compute: &mut Compute,
    make_sink: M,
    hooks: &Hooks<'_>,
) -> Result<TiledOutput<K>, PipelineError>
where
    S: FrameSource + ?Sized,
    K: TileSink,
    M: FnOnce(usize, usize) -> K,
{
    run_tiled_with(
        source,
        job,
        compute,
        make_sink,
        hooks,
        align_into_store::<S>,
    )
}

/// [`run_tiled`] with the next frame decoded on a second thread.
///
/// The decode runs while the current frame is aligned ([`align_into_store_overlapped`]), for a frame source
/// that is `Send`. The result is the same; one more decoded frame is in memory.
///
/// # Errors
/// As [`run_tiled`].
pub fn run_tiled_overlapped<S, K, M>(
    source: &mut S,
    job: &TiledJob<'_>,
    compute: &mut Compute,
    make_sink: M,
    hooks: &Hooks<'_>,
) -> Result<TiledOutput<K>, PipelineError>
where
    S: FrameSource + Send + ?Sized,
    K: TileSink,
    M: FnOnce(usize, usize) -> K,
{
    run_tiled_with(
        source,
        job,
        compute,
        make_sink,
        hooks,
        align_into_store_overlapped::<S>,
    )
}

/// [`run_tiled`] with a neural alignment `model` (see
/// [`align_into_store_with_model`]).
///
/// # Errors
/// As [`run_tiled`].
pub fn run_tiled_with_model<S, K, M>(
    source: &mut S,
    job: &TiledJob<'_>,
    model: Option<AlignModel>,
    compute: &mut Compute,
    make_sink: M,
    hooks: &Hooks<'_>,
) -> Result<TiledOutput<K>, PipelineError>
where
    S: FrameSource + ?Sized,
    K: TileSink,
    M: FnOnce(usize, usize) -> K,
{
    let align = move |source: &mut S,
                      reference: &PlanarImage<f32>,
                      settings: &StackingSettings,
                      tiles: &[TilePlan],
                      store: &mut MmapTileManager,
                      compute: &mut Compute,
                      hooks: &Hooks<'_>| {
        align_into_store_with_model(
            source, reference, settings, model, tiles, store, compute, hooks,
        )
    };
    run_tiled_with(source, job, compute, make_sink, hooks, align)
}

/// The shared body of the `run_tiled` family; `align` is the alignment pass.
fn run_tiled_with<S, K, M, A>(
    source: &mut S,
    job: &TiledJob<'_>,
    compute: &mut Compute,
    make_sink: M,
    hooks: &Hooks<'_>,
    align: A,
) -> Result<TiledOutput<K>, PipelineError>
where
    S: FrameSource + ?Sized,
    K: TileSink,
    M: FnOnce(usize, usize) -> K,
    A: FnOnce(
        &mut S,
        &PlanarImage<f32>,
        &StackingSettings,
        &[TilePlan],
        &mut MmapTileManager,
        &mut Compute,
        &Hooks<'_>,
    ) -> Result<CommonCoverage, PipelineError>,
{
    let settings = job.settings;
    if job.method == Method::Neural {
        return Err(PipelineError::invalid(
            "The neural mode has its own tile loop and is not run by run_tiled.",
        ));
    }
    if job.tile_size < MIN_TILE_SIZE {
        return Err(PipelineError::invalid(format!(
            "The tile size must be at least {MIN_TILE_SIZE} pixels (got {}).",
            job.tile_size
        )));
    }
    let n_frames = source.len();
    if n_frames == 0 {
        return Err(PipelineError::invalid(
            "No images to process: add some images first.",
        ));
    }
    hooks.check_cancel()?;

    let reference = source.decode(0, &settings.preprocessing)?;
    let size = (reference.width, reference.height);
    if size.0 == 0 || size.1 == 0 {
        return Err(PipelineError::invalid(format!(
            "The reference frame {} has unusable dimensions {}x{}.",
            source.name(0),
            size.0,
            size.1
        )));
    }
    let plan = plan_run(job.method, size, job.tile_size, settings);
    hooks.log.info(&format!(
        "image {}x{}, tile_size={}, apron={} px, {} tiles, scratch about {} bytes for {n_frames} \
         frames",
        size.0,
        size.1,
        plan.geometry.tile_size,
        plan.geometry.apron,
        plan.tiles.len(),
        plan.scratch_bytes(n_frames),
    ));
    if let Some(warning) = plan.padding_warning(job.method) {
        hooks.log.warn(&warning);
    }

    // The guard owns the scratch directory: whatever happens below (an error, a
    // panic, success) the tiles are removed when it drops.
    let scratch = ScratchDir::create(job.scratch_root)?;
    let mut store = MmapTileManager::new(scratch.path().to_path_buf());

    hooks
        .log
        .info(&format!("aligning and tiling {n_frames} frames"));
    let coverage = align(
        source,
        &reference,
        settings,
        &plan.tiles,
        &mut store,
        compute,
        hooks,
    )?;
    drop(reference);

    hooks.progress.stage(Stage::Cull, 0.0, "");
    let order = frame_order(settings, &plan, &store, n_frames, hooks)?;
    hooks.progress.stage(Stage::Cull, 1.0, "");

    hooks.log.info(&format!(
        "fusion pass: {} tiles x {} frames",
        plan.tiles.len(),
        order.len()
    ));
    let mut sink = make_sink(size.0, size.1);
    fuse_tiles(
        &FusePlan::new(job.method, settings, &plan, &order),
        &StoreFrames(&store),
        compute,
        &mut sink,
        hooks,
    )?;
    // The tiles are no longer needed; release the disk before the caller encodes.
    drop(store);
    drop(scratch);

    let crop = crop_rect(&coverage, settings, size, |i| source.name(i).to_owned());
    Ok(TiledOutput {
        sink,
        size,
        order,
        crop,
        coverage,
    })
}
