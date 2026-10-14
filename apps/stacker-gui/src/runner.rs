//! The seam between the GUI and whatever runs its heavy jobs.
//!
//! The wiring in `gui/` never decodes, aligns or fuses: it builds a request
//! ([`StackRequest`], [`AlignRequest`]), hands it to the [`JobRunner`] installed
//! for the UI thread and reacts to the [`RunEvent`]s that come back, always on the
//! UI thread and in order. The desktop app uses [`native::NativeRunner`] (worker
//! threads running `stacker_pipeline::session`); the browser build installs a
//! runner that talks to a Web Worker, where the aligned frames live in the
//! worker's memory.
//!
//! # What the runner holds
//!
//! The aligned frames of the last Align pass. The GUI keeps only the record
//! ([`crate::state::AlignedCache`]: the files and the crop) and decides whether a
//! run may reuse the frames ([`StackRequest::reuse_aligned`]); the runner serves
//! them to a stack run ([`JobRunner::run_stack`]), to the viewer's Show Aligned
//! and to the retouch source picker ([`JobRunner::fetch_source`]).
//!
//! # Threading
//!
//! A runner is `!Send` and every method is called on the UI thread. The closures
//! it is given ([`Events`], [`AskRelief`], the `then` continuations) are called on
//! the UI thread too. A runner on worker threads calls them later, from the event
//! loop; a runner on a host without threads may call one before the method that
//! received it has returned, so a caller must hold no borrow of its state across
//! such a call.
//!
//! The installed runner lives in a thread-local: a thread that never installed
//! one gets the native runner on the desktop builds (so tests need no set-up) and
//! panics on wasm32, where [`install`] must run first.

use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_core::{image::PlanarImage, settings::StackingSettings};

use crate::{gui::Ctx, jobs_core::JobError};

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

/// One fusion pass of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// The trained neural model (desktop builds with `--features nn`).
    Ai,
    /// Depth map.
    Relief,
    /// Laplacian-pyramid maximum contrast.
    Apex,
    /// Guided-filter soft blend.
    Strata,
}

impl Pass {
    /// The name the status line and the result record use.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ai => "AI",
            Self::Relief => "Relief",
            Self::Apex => "Apex",
            Self::Strata => "Strata",
        }
    }
}

/// What the Sort and Cull actions do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortCullAction {
    /// Reorder the list by the focus census, sharpest neighbours together.
    Sort,
    /// Mark the frames that add too little sharp detail.
    Cull,
}

impl SortCullAction {
    /// The word the status line and the job chip use.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sort => "Sort",
            Self::Cull => "Cull",
        }
    }
}

/// One step of a run, for progress (the GUI maps it onto its stage plan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunStep {
    /// Decoding the frames.
    Load,
    /// Aligning them.
    Align,
    /// The auto-cull.
    Cull,
    /// Fusing pass number `n` (0-based) of the run.
    Fuse(usize),
    /// Saving and publishing pass number `n` of the run.
    Save(usize),
}

/// What a run reports, on the UI thread, in order.
pub enum RunEvent {
    /// Progress of one step; `label` is the status text (empty: leave the status
    /// line alone).
    Progress {
        /// The step that reports.
        step: RunStep,
        /// How far that step is, `0.0..=1.0`.
        fraction: f32,
        /// The status text, or empty.
        label: String,
    },
    /// Something the user should read (skipped frames, a GPU fallback, the
    /// auto-cull's notes).
    Warning(String),
    /// A live canvas frame while aligning (RGBA8, already sized for display).
    LiveFrame(SharedPixelBuffer<Rgba8Pixel>),
    /// One finished pass (stack runs only).
    Result(StackResult),
    /// The aligned set is ready (align runs only; the runner now holds the
    /// frames).
    AlignDone {
        /// The files of the frames that actually aligned, in frame order (frames
        /// that failed to decode are not listed).
        paths: Vec<PathBuf>,
        /// The rectangle valid in every frame, `None` when no crop is needed.
        crop: Option<(usize, usize, usize, usize)>,
        /// The alignment's fallback warning, if any frame did not align properly.
        warning: Option<String>,
    },
    /// The answer of a Sort or Cull run (those runs only; an align that had to
    /// happen first has already reported [`Self::AlignDone`]).
    SortCullDone {
        /// What was asked.
        action: SortCullAction,
        /// The files of the run in their new order: for Sort all of them, for
        /// Cull the kept ones in stack order.
        order: Vec<PathBuf>,
        /// The files the Cull would leave out (empty for Sort).
        dropped: Vec<PathBuf>,
    },
    /// Last event of every run: `Ok` carries a note for the status line (skipped
    /// frames), `Err` why the run produced no (further) result.
    Finished(Result<Option<String>, JobError>),
}

/// One finished fusion pass.
pub struct StackResult {
    /// The pass that made it.
    pub pass: Pass,
    /// The fused image, cut to the common valid area.
    pub fused: Arc<PlanarImage<f32>>,
    /// Where the 8-bit preview of the result can be read back with
    /// `platform::current().read_bytes` (native: the temporary PNG; web: a
    /// `web:/results/...` path the web runner registered). It is the path the
    /// result list shows.
    pub preview_path: PathBuf,
    /// The source files the auto-cull dropped (first pass of a run only; empty for
    /// the others).
    pub culled: Vec<PathBuf>,
    /// Frames that reached the fusion (after the auto-cull).
    pub kept: usize,
    /// Frames that decoded.
    pub total: usize,
}

/// The `Relief` contrast question, asked when `relief_show_preview` is on.
pub struct ReliefAsk {
    /// The first frame at panel size.
    pub base: SharedPixelBuffer<Rgba8Pixel>,
    /// The focus map at panel size, row-major like `base`.
    pub focus_max: Vec<f32>,
    /// 1001 full-size thresholds: entry `k` rejects the fraction `k / 1000`
    /// (`stacker_pipeline::session::threshold_table`).
    pub thresholds: Vec<f32>,
    /// The auto-detected (median + 1.5·MAD) rejected fraction, `0.0..=1.0`.
    pub auto_pct: f32,
    /// The fraction the slider starts at (from the settings).
    pub initial_pct: f32,
}

/// What a stack run needs.
pub struct StackRequest {
    /// The files to stack, selected and ordered (`frames::paths_for_run`).
    pub paths: Vec<PathBuf>,
    /// The effective settings snapshot.
    pub settings: StackingSettings,
    /// The passes to run, in order.
    pub passes: Vec<Pass>,
    /// Use the frames of the last Align pass (the GUI checked they are current for
    /// `paths` and `settings`); the runner falls back to aligning afresh if it
    /// cannot serve them.
    pub reuse_aligned: bool,
    /// The AI model and device names (native `nn` builds).
    pub ai: Option<(String, String)>,
    /// The alignment model and device names for `alignment_mode = "neural"`
    /// (native `nn` builds; `None` otherwise).
    pub align_model: Option<(String, String)>,
}

/// What an align run needs.
pub struct AlignRequest {
    /// The files to align, selected and ordered (`frames::paths_for_run`).
    pub paths: Vec<PathBuf>,
    /// The effective settings snapshot.
    pub settings: StackingSettings,
    /// The alignment model and device names for `alignment_mode = "neural"`
    /// (native `nn` builds; `None` otherwise).
    pub align_model: Option<(String, String)>,
}

/// What a Sort or Cull run needs: the frames are aligned first (the held
/// aligned set is used when the GUI says it is current), then the census runs.
pub struct SortCullRequest {
    /// Sort or Cull.
    pub action: SortCullAction,
    /// The files, selected and ordered (`frames::paths_for_run` without
    /// subsampling, as the Align pass).
    pub paths: Vec<PathBuf>,
    /// The effective settings snapshot.
    pub settings: StackingSettings,
    /// Use the frames of the last Align pass (the GUI checked they are current).
    pub reuse_aligned: bool,
    /// The alignment model and device names (as [`AlignRequest::align_model`]).
    pub align_model: Option<(String, String)>,
}

/// Receives the [`RunEvent`]s of one run, on the UI thread.
pub type Events = Box<dyn FnMut(RunEvent)>;

/// Receives the `Relief` contrast question of one run, on the UI thread. The
/// second argument answers it, once: the absolute threshold, or `None` to cancel
/// the run.
pub type AskRelief = Box<dyn FnMut(ReliefAsk, Box<dyn FnOnce(Option<f32>)>)>;

/// The continuation of [`JobRunner::fetch_source`].
pub type SourceDone = Box<dyn FnOnce(Result<PlanarImage<f32>, String>)>;

/// The continuation of [`JobRunner::fetch_preview`].
pub type PreviewDone = Box<dyn FnOnce(Result<SharedPixelBuffer<Rgba8Pixel>, String>)>;

/// The continuation of [`JobRunner::preflight`]: the (possibly changed) request
/// and a note for the status line, or `None` when the user cancelled.
pub type PreflightDone = Box<dyn FnOnce(Option<(StackRequest, Option<String>)>)>;

/// A running job, kept to cancel it.
pub trait JobHandle {
    /// Ask the job to stop. It ends with a [`RunEvent::Finished`] carrying
    /// [`JobError::Cancelled`] (or its result, if it was already done).
    fn cancel(&self);
}

/// The handle of a run that ended before it began.
struct NoJob;

impl JobHandle for NoJob {
    fn cancel(&self) {}
}

/// Runs the GUI's heavy jobs. See the module docs.
pub trait JobRunner {
    /// Start a stack run. Every event arrives through `events`, the last one
    /// being [`RunEvent::Finished`]; the `Relief` contrast question goes to
    /// `ask_relief`.
    fn run_stack(
        &self,
        request: StackRequest,
        events: Events,
        ask_relief: AskRelief,
    ) -> Box<dyn JobHandle>;

    /// Start an align run: events as for [`Self::run_stack`], with
    /// [`RunEvent::AlignDone`] instead of results.
    fn run_align(&self, request: AlignRequest, events: Events) -> Box<dyn JobHandle>;

    /// Start a Sort or Cull run: align (or reuse the held aligned set), take the
    /// focus census, report [`RunEvent::SortCullDone`] and then
    /// [`RunEvent::Finished`]. A runner that cannot (the browser build) keeps the
    /// default, which finishes at once with an error.
    fn run_sort_cull(&self, request: SortCullRequest, events: Events) -> Box<dyn JobHandle> {
        let mut events = events;
        let what = request.action.label();
        events(RunEvent::Finished(Err(JobError::Failed(format!(
            "{what} is not available in this build."
        )))));
        Box::new(NoJob)
    }

    /// The aligned and cropped frame of `path` (from the aligned set the runner
    /// holds), else the file decoded the way a stack run decodes it.
    fn fetch_source(&self, path: &Path, settings: &StackingSettings, then: SourceDone);

    /// A display buffer of one source file at full size (the viewer). A newer
    /// call may supersede an older one whose `then` has not run; the viewer
    /// ignores superseded answers either way.
    fn fetch_preview(&self, path: &Path, then: PreviewDone);

    /// Drop the aligned set (New Project, Open Project).
    fn forget_aligned(&self);

    /// End the worker threads the runner keeps for itself (the viewer's decoder)
    /// and wait for them. For the UI harness's teardown; the runner starts them
    /// again on demand. The default has none.
    #[cfg(feature = "ui-harness")]
    fn stop_workers(&self) {}

    /// A run is about to start: the web runner checks its memory budget here and
    /// may ask the user (via `gui::confirm`) or change
    /// `request.settings.preprocessing.pre_resize_percent` for this run. The
    /// default goes on unchanged. `then(None)` = the user cancelled.
    fn preflight(&self, ctx: &Ctx, request: StackRequest, then: PreflightDone) {
        let _ = ctx;
        then(Some((request, None)));
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Rc<dyn JobRunner>>> = const { RefCell::new(None) };
}

/// Install `runner` as this thread's runner (the UI thread's), replacing any
/// installed earlier.
pub fn install(runner: Rc<dyn JobRunner>) {
    CURRENT.with(|slot| *slot.borrow_mut() = Some(runner));
}

#[cfg(not(target_arch = "wasm32"))]
fn default_runner() -> Rc<dyn JobRunner> {
    Rc::new(native::NativeRunner::new())
}

#[cfg(target_arch = "wasm32")]
fn default_runner() -> Rc<dyn JobRunner> {
    panic!("no job runner installed: call stacker_gui::runner::install before the GUI starts")
}

/// The runner of this thread: the one [`install`] registered, else the native one
/// on the desktop builds.
///
/// # Panics
///
/// On wasm32 when no runner was installed.
#[must_use]
pub fn current() -> Rc<dyn JobRunner> {
    CURRENT.with(|slot| Rc::clone(slot.borrow_mut().get_or_insert_with(default_runner)))
}
