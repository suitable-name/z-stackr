//! The desktop job runner: every job on worker threads, in this process.
//!
//! A stack or align run is one thread running the in-memory stages of
//! [`stacker_pipeline::session`] (the same functions the browser worker calls).
//! The worker never touches the UI state: it posts [`RunEvent`]s to the event
//! loop, where [`deliver`] hands each one to the run's sink, the closure the GUI
//! gave [`JobRunner::run_stack`]. The sinks live in a UI-thread thread-local keyed
//! by run id, because the closures are `!Send`; only the id and the event travel
//! through the event loop.
//!
//! # The `Relief` contrast question
//!
//! When the settings ask for the preview, the fusion worker posts a [`ReliefAsk`]
//! to the UI thread and blocks on a channel until the GUI answers through the
//! `reply` it was given. A cancel while the panel is open is answered by the GUI
//! (`None`), so a cancel never waits for a click.
//!
//! # The aligned set
//!
//! The frames of the last Align pass are held here, reference-counted, behind a
//! mutex shared with the run threads. A stack run that may reuse them takes
//! `Arc` clones when it starts and copies the pixels on its own thread, not on
//! the UI thread.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use stacker_align::transform::largest_true_rectangle;
use stacker_core::{
    image::PlanarImage,
    memory::{MmapTileManager, extract_tile},
    settings::{AlignmentModeSetting, ImageSavingSettings, OutputFormat, StackingSettings},
};
use stacker_pipeline::{
    Cancel, PipelineError, Progress, Stage,
    align::AlignModel,
    compute::Compute,
    output::encode,
    session::{self, FrameSource, PathSource},
    tiled::{self, FusePlan, Hooks, RunLog, ScratchDir, StoreFrames, TiledPlan},
    tiling::{Method, apron_for, plan_tiles},
};

use super::{
    AlignRequest, AskRelief, Events, JobHandle, JobRunner, Pass, PreviewDone, ReliefAsk, RunEvent,
    RunStep, SortCullAction, SortCullRequest, SourceDone, StackRequest, StackResult,
};
use crate::{
    frames::{self, FrameSet},
    gui::feedback,
    jobs_core::{CancelFlag, JobError, fraction_of, run_job},
    latest_worker::LatestWorker,
    preview::{LivePreview, decode_file, downsample_plane, downsample_to_long_edge},
    retouch::decode_source,
    state::Crop,
    sync::lock,
};

/// The long edge of the `Relief` preview panel's image, in pixels: the panel shows
/// it at about half this size on a normal display, and 640 keeps it sharp on a
/// high-DPI one while a render stays a few hundred thousand pixels.
const PANEL_LONG_EDGE: u32 = 640;

// ── The aligned set ─────────────────────────────────────────────────────────

/// The aligned frames of the last Align pass.
///
/// `frames[i]` is the aligned frame loaded from `paths[i]`; the two vectors are
/// parallel and lookups go through the path, so list edits, `sort_reverse`,
/// skipped frames and filtered result files can never shift which frame a click
/// shows. `crop` is the largest rectangle valid in every frame.
struct AlignedSet {
    paths: Vec<PathBuf>,
    frames: Vec<Arc<PlanarImage<f32>>>,
    crop: Option<Crop>,
}

/// One aligned frame together with the crop that still has to be applied to it.
///
/// Cutting the crop copies the frame, so the UI thread passes this on and the
/// thread that needs the pixels calls [`Self::cropped`].
struct AlignedView {
    frame: Arc<PlanarImage<f32>>,
    crop: Option<Crop>,
}

impl AlignedView {
    /// The frame cut to the common area (a copy), or a copy of the whole frame
    /// when there is no crop.
    fn cropped(&self) -> PlanarImage<f32> {
        match self.crop {
            Some((x, y, w, h)) => extract_tile(&self.frame, x, y, w, h),
            None => self.frame.as_ref().clone(),
        }
    }
}

/// Frames shared out of an [`AlignedSet`] for a stack run, in the run's order.
struct Reused {
    paths: Vec<PathBuf>,
    frames: Vec<Arc<PlanarImage<f32>>>,
    crop: Option<Crop>,
}

impl AlignedSet {
    /// The aligned frame that was computed for `path`.
    #[cfg(test)]
    fn frame_for(&self, path: &Path) -> Option<&PlanarImage<f32>> {
        let index = self.paths.iter().position(|p| p == path)?;
        self.frames.get(index).map(Arc::as_ref)
    }

    /// The aligned frame for `path` with the common crop still to be applied.
    /// Copies no pixels.
    fn view_for(&self, path: &Path) -> Option<AlignedView> {
        let index = self.paths.iter().position(|p| p == path)?;
        Some(AlignedView {
            frame: Arc::clone(self.frames.get(index)?),
            crop: self.crop,
        })
    }

    /// The held frames for `wanted`, in that order, with the set's crop. `None`
    /// when `wanted` is empty or the set lacks any of the files.
    fn frames_for(&self, wanted: &[PathBuf]) -> Option<Reused> {
        if wanted.is_empty() {
            return None;
        }
        let positions: Vec<usize> = wanted
            .iter()
            .map(|path| self.paths.iter().position(|p| p == path))
            .collect::<Option<_>>()?;
        let frames: Vec<Arc<PlanarImage<f32>>> = positions
            .into_iter()
            .map(|index| self.frames.get(index).map(Arc::clone))
            .collect::<Option<_>>()?;
        Some(Reused {
            paths: wanted.to_vec(),
            frames,
            crop: self.crop,
        })
    }
}

/// The frame itself when nothing else shares it, else a copy.
fn owned_frame(frame: Arc<PlanarImage<f32>>) -> PlanarImage<f32> {
    Arc::try_unwrap(frame).unwrap_or_else(|shared| shared.as_ref().clone())
}

/// The aligned set, shared between the UI thread and the run threads.
type Held = Arc<Mutex<Option<AlignedSet>>>;

// ── UI-thread delivery ──────────────────────────────────────────────────────

/// What the GUI gave a run: where its events and its `Relief` question go.
struct Sink {
    events: Events,
    ask: AskRelief,
}

type SharedSink = Rc<RefCell<Sink>>;

/// The source fetches in flight, by request id.
type SourceMap = BTreeMap<u64, SourceDone>;

thread_local! {
    /// The sinks of the runs in flight, by run id (UI thread only).
    static SINKS: RefCell<BTreeMap<u64, SharedSink>> = const { RefCell::new(BTreeMap::new()) };
    /// The viewer's pending request: only the newest is answered.
    static PREVIEW_WAITING: RefCell<Option<(u64, PreviewDone)>> = const { RefCell::new(None) };
    /// The continuations of the source fetches in flight, by request id.
    static SOURCE_WAITING: RefCell<SourceMap> = const { RefCell::new(BTreeMap::new()) };
}

/// Numbers runs and requests; shared by all, so ids never repeat.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// How many run and fetch threads are alive (the UI harness's teardown waits for
/// zero, because the profile it deletes holds their scratch files).
#[cfg(feature = "ui-harness")]
static LIVE_WORKERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// One live worker thread, counted in [`LIVE_WORKERS`] while it exists. Moved into
/// the thread's closure, so a closure that is dropped unrun is uncounted too.
#[cfg(feature = "ui-harness")]
struct LiveWorker;

#[cfg(feature = "ui-harness")]
impl LiveWorker {
    fn enter() -> Self {
        LIVE_WORKERS.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

#[cfg(feature = "ui-harness")]
impl Drop for LiveWorker {
    fn drop(&mut self) {
        LIVE_WORKERS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The run and fetch threads still alive. For the UI harness.
#[cfg(feature = "ui-harness")]
#[must_use]
pub fn live_workers() -> usize {
    LIVE_WORKERS.load(Ordering::SeqCst)
}

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Hand `event` to the sink of run `run`, on the UI thread. The sink is dropped
/// after the run's last event.
fn deliver(run: u64, event: RunEvent) {
    let sink = SINKS.with(|map| map.borrow().get(&run).cloned());
    let Some(sink) = sink else {
        return;
    };
    let last = matches!(event, RunEvent::Finished(_));
    if let Ok(mut sink) = sink.try_borrow_mut() {
        (sink.events)(event);
    }
    if last {
        SINKS.with(|map| map.borrow_mut().remove(&run));
    }
}

/// Put the `Relief` question of run `run` to its sink, on the UI thread. A run
/// that is gone (or a sink that is busy) gets the answer "cancel".
fn deliver_ask(run: u64, ask: ReliefAsk, reply: mpsc::Sender<Option<f32>>) {
    let sink = SINKS.with(|map| map.borrow().get(&run).cloned());
    let Some(sink) = sink else {
        let _ = reply.send(None);
        return;
    };
    if let Ok(mut sink) = sink.try_borrow_mut() {
        let answer: Box<dyn FnOnce(Option<f32>)> = Box::new(move |threshold| {
            let _ = reply.send(threshold);
        });
        (sink.ask)(ask, answer);
    } else {
        let _ = reply.send(None);
    }
}

/// The worker thread's way to the UI thread.
#[derive(Clone, Copy)]
struct Post {
    run: u64,
}

impl Post {
    fn event(self, event: RunEvent) {
        let run = self.run;
        let _ = slint::invoke_from_event_loop(move || deliver(run, event));
    }

    fn progress(self, step: RunStep, fraction: f32, label: &str) {
        self.event(RunEvent::Progress {
            step,
            fraction,
            label: label.to_owned(),
        });
    }

    fn warning(self, text: String) {
        self.event(RunEvent::Warning(text));
    }

    /// Put the `Relief` question to the UI and wait for the answer; `None` when
    /// the user cancelled (or the window is gone).
    fn ask(self, ask: ReliefAsk) -> Option<f32> {
        let (tx, rx) = mpsc::channel::<Option<f32>>();
        let run = self.run;
        let _ = slint::invoke_from_event_loop(move || deliver_ask(run, ask, tx));
        rx.recv().ok().flatten()
    }
}

// ── The runner ──────────────────────────────────────────────────────────────

/// What the viewer thread decodes: request id and file.
type PreviewJob = (u64, PathBuf);

/// Runs the GUI's jobs on worker threads. See the module docs.
#[derive(Default)]
pub struct NativeRunner {
    held: Held,
    previews: RefCell<Option<LatestWorker<PreviewJob>>>,
}

impl NativeRunner {
    /// A runner with an empty aligned set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Cancels a native run by raising its flag; the worker stops at its next
/// checkpoint.
struct NativeHandle {
    cancel: CancelFlag,
}

impl JobHandle for NativeHandle {
    fn cancel(&self) {
        self.cancel.cancel();
    }
}

/// Register the sinks of a new run and return its post.
fn register(events: Events, ask: AskRelief) -> Post {
    let run = next_id();
    SINKS.with(|map| {
        map.borrow_mut()
            .insert(run, Rc::new(RefCell::new(Sink { events, ask })));
    });
    Post { run }
}

/// Start `body` on a thread named after `kind`. A thread that cannot be started
/// ends the run at once.
fn spawn_worker(kind: &str, post: Post, body: impl FnOnce() + Send + 'static) {
    #[cfg(feature = "ui-harness")]
    let body = {
        let live = LiveWorker::enter();
        move || {
            let _live = live;
            body();
        }
    };
    let spawned = std::thread::Builder::new()
        .name(format!("z-stackr-{kind}"))
        .spawn(body);
    if let Err(error) = spawned {
        post.event(RunEvent::Finished(Err(JobError::Failed(format!(
            "Could not start the {kind} job: {error}"
        )))));
    }
}

impl JobRunner for NativeRunner {
    fn run_stack(
        &self,
        request: StackRequest,
        events: Events,
        ask_relief: AskRelief,
    ) -> Box<dyn JobHandle> {
        let post = register(events, ask_relief);
        let cancel = CancelFlag::default();
        let held = Arc::clone(&self.held);
        let flag = cancel.clone();
        spawn_worker("stack", post, move || {
            run_job(
                &flag,
                |flag| execute(post, flag, &request, &held),
                |result| post.event(RunEvent::Finished(result)),
            );
        });
        Box::new(NativeHandle { cancel })
    }

    fn run_align(&self, request: AlignRequest, events: Events) -> Box<dyn JobHandle> {
        // An align run never asks the `Relief` question.
        let never: AskRelief = Box::new(|_, reply| reply(None));
        let post = register(events, never);
        let cancel = CancelFlag::default();
        let held = Arc::clone(&self.held);
        let flag = cancel.clone();
        spawn_worker("align", post, move || {
            run_job(
                &flag,
                |flag| align(post, flag, &request, &held),
                |result| post.event(RunEvent::Finished(result)),
            );
        });
        Box::new(NativeHandle { cancel })
    }

    fn run_sort_cull(&self, request: SortCullRequest, events: Events) -> Box<dyn JobHandle> {
        // Sort and Cull never ask the `Relief` question.
        let never: AskRelief = Box::new(|_, reply| reply(None));
        let post = register(events, never);
        let cancel = CancelFlag::default();
        let held = Arc::clone(&self.held);
        let flag = cancel.clone();
        spawn_worker("sort-cull", post, move || {
            run_job(
                &flag,
                |flag| sort_cull(post, flag, &request, &held),
                |result| post.event(RunEvent::Finished(result)),
            );
        });
        Box::new(NativeHandle { cancel })
    }

    fn fetch_source(&self, path: &Path, settings: &StackingSettings, then: SourceDone) {
        let view = lock(&self.held).as_ref().and_then(|set| set.view_for(path));
        let pre = settings.preprocessing.clone();
        let path = path.to_path_buf();
        let id = next_id();
        SOURCE_WAITING.with(|map| map.borrow_mut().insert(id, then));
        #[cfg(feature = "ui-harness")]
        let live = LiveWorker::enter();
        std::thread::spawn(move || {
            #[cfg(feature = "ui-harness")]
            let _live = live;
            let outcome =
                view.map_or_else(|| decode_source(&path, &pre), |view| Ok(view.cropped()));
            let _ = slint::invoke_from_event_loop(move || source_arrived(id, outcome));
        });
    }

    fn fetch_preview(&self, path: &Path, then: PreviewDone) {
        let id = next_id();
        // A newer request replaces the waiting one: its answer would be dropped
        // by the viewer anyway.
        PREVIEW_WAITING.with(|slot| *slot.borrow_mut() = Some((id, then)));
        let submitted = self
            .previews
            .borrow_mut()
            .get_or_insert_with(|| {
                LatestWorker::spawn("z-stackr-viewer", |(id, path): PreviewJob| {
                    let outcome = decode_file(&path);
                    let _ = slint::invoke_from_event_loop(move || preview_arrived(id, outcome));
                })
            })
            .submit((id, path.to_path_buf()));
        if !submitted {
            preview_arrived(id, Err("the viewer thread is not running".to_owned()));
        }
    }

    fn forget_aligned(&self) {
        *lock(&self.held) = None;
    }

    #[cfg(feature = "ui-harness")]
    fn stop_workers(&self) {
        let previews = self.previews.borrow_mut().take();
        if let Some(worker) = previews {
            worker.stop();
        }
    }
}

/// A source fetch finished: run its continuation, on the UI thread.
fn source_arrived(id: u64, outcome: Result<PlanarImage<f32>, String>) {
    let then = SOURCE_WAITING.with(|map| map.borrow_mut().remove(&id));
    if let Some(then) = then {
        then(outcome);
    }
}

/// A preview decode finished: answer it if it is still the newest request.
fn preview_arrived(id: u64, outcome: Result<slint::SharedPixelBuffer<slint::Rgba8Pixel>, String>) {
    let then = PREVIEW_WAITING.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_some_and(|(waiting, _)| *waiting == id) {
            slot.take().map(|(_, then)| then)
        } else {
            None
        }
    });
    if let Some(then) = then {
        then(outcome);
    }
}

// ── The worker side ─────────────────────────────────────────────────────────

/// What a run thread reports through and checks for a cancel.
struct Run<'a> {
    post: Post,
    cancel: &'a CancelFlag,
}

impl Run<'_> {
    /// A checkpoint: `Err(Cancelled(at))` when the user asked to stop.
    fn check(&self, at: &str) -> Result<(), JobError> {
        self.cancel.check(at)
    }

    fn progress(&self, step: RunStep, fraction: f32, label: &str) {
        self.post.progress(step, fraction, label);
    }

    fn warn(&self, text: String) {
        self.post.warning(text);
    }
}

/// A [`CancelFlag`] as the shared stages' cancel check.
struct FlagCancel<'a>(&'a CancelFlag);

impl Cancel for FlagCancel<'_> {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

/// The files of `set` as the names the shared stages' messages use.
fn names_of(set: &FrameSet) -> Vec<String> {
    set.paths
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect()
}

/// What the status text of each finished pass needs.
struct RunSummary {
    /// Frames that decoded.
    total: usize,
    /// Frames that reach the fusion (after auto-cull).
    stack_total: usize,
    /// When the run started, for the temporary file name.
    ts_ms: u128,
}

/// What one fusion pass works on.
struct Fusion<'a> {
    ordered: &'a [PlanarImage<f32>],
    stack_crop: Option<Crop>,
    pass: Pass,
    /// Position of the pass in the run (picks its steps).
    index: usize,
}

/// The frames of the run, aligned, and the crop they share. Either the frames of
/// a current Align pass (taken from the held set), or decoded and aligned here.
/// The crop always belongs to THESE frames: a run that did not use the held set
/// never inherits its crop.
fn aligned_frames(
    run: &Run<'_>,
    request: &StackRequest,
    held: &Held,
) -> Result<(FrameSet, Option<Crop>), JobError> {
    if request.reuse_aligned {
        let reused = lock(held)
            .as_ref()
            .and_then(|set| set.frames_for(&request.paths));
        if let Some(reused) = reused {
            run.progress(RunStep::Load, 1.0, "Reusing the aligned frames…");
            run.progress(RunStep::Align, 1.0, "");
            // The held set keeps its frames, so these are copies; they are made
            // here, on the worker, not when the run was started.
            let set = FrameSet {
                paths: reused.paths,
                frames: reused.frames.into_iter().map(owned_frame).collect(),
                skipped: Vec::new(),
            };
            return Ok((set, reused.crop));
        }
    }

    let t_load = Instant::now();
    let mut set = frames::decode_frames(request.paths.clone(), &request.settings.preprocessing)
        .map_err(|e| JobError::Failed(e.to_string()))?;
    if let Some(warning) = feedback::skipped_frames_warning(&set.skipped) {
        run.warn(warning);
    }
    run.progress(RunStep::Load, 1.0, "");
    tracing::info!(
        n_loaded = set.len(),
        width = set.frames[0].width,
        height = set.frames[0].height,
        elapsed_ms = t_load.elapsed().as_millis(),
        "load stage complete"
    );
    run.check("after loading the frames")?;

    let t_align = Instant::now();
    let crop = align_stage(
        run,
        &request.settings,
        request.align_model.as_ref(),
        &mut set,
    )?;
    tracing::info!(
        elapsed_ms = t_align.elapsed().as_millis(),
        "alignment stage complete"
    );
    Ok((set, crop))
}

/// Align the frames to the first one; returns the common-area crop. `Cancelled`
/// when the user stopped it (the partly aligned frames are dropped with `set`).
fn align_stage(
    run: &Run<'_>,
    settings: &StackingSettings,
    model: Option<&(String, String)>,
    set: &mut FrameSet,
) -> Result<Option<Crop>, JobError> {
    let mode = settings.alignment_mode;
    if mode == AlignmentModeSetting::None || set.len() <= 1 {
        return Ok(None);
    }
    let total = set.len();
    let model = align_model_for(settings, model)?;
    run.progress(
        RunStep::Align,
        0.0,
        &format!("Aligning frames ({})…", mode.as_combo_str()),
    );
    // The objective and the warp run on the GPU when `compute_device` asks for it
    // and the build and the device allow; the CPU otherwise.
    let mut compute = Compute::for_settings(settings);
    let outcome = session::align_frames_with_model(
        &mut set.frames,
        settings,
        model,
        &mut compute,
        &FlagCancel(run.cancel),
        &mut |i, _, _| {
            run.progress(
                RunStep::Align,
                fraction_of(i, total - 1),
                &format!("Aligning… frame {i}/{}", total - 1),
            );
        },
    )
    .ok_or_else(|| JobError::Cancelled("while aligning".to_owned()))?;
    if let Some(warning) = outcome.warning(&names_of(set)) {
        run.warn(warning);
    }
    Ok(outcome.crop)
}

/// The neural model a run needs: `None` unless `alignment_mode = "neural"`.
/// Neural mode without a usable model is refused here, before any frame is
/// decoded, instead of silently aligning in another mode.
fn align_model_for(
    settings: &StackingSettings,
    chosen: Option<&(String, String)>,
) -> Result<Option<AlignModel>, JobError> {
    if settings.alignment_mode != AlignmentModeSetting::Neural {
        return Ok(None);
    }
    let (name, device) = chosen.map_or(("", ""), |(n, d)| (n.as_str(), d.as_str()));
    build_align_model(name, device)
        .map(Some)
        .map_err(|msg| JobError::Failed(format!("Neural alignment: {msg}")))
}

/// The order in which frames enter the fusion (positions in `set`; the kept order
/// is used for the fusion only) and the files the auto-cull dropped, which the
/// sidebar marks when the first result is published (not before: a cancelled run
/// must not leave marks). The user's source list is never rewritten.
fn cull_stage(
    run: &Run<'_>,
    settings: &StackingSettings,
    set: &FrameSet,
) -> Result<(Vec<usize>, Vec<PathBuf>), JobError> {
    if !settings.auto_cull && !settings.sort_by_sharpness {
        return Ok(((0..set.len()).collect(), Vec::new()));
    }
    run.progress(RunStep::Cull, 0.0, "Optimising & culling…");
    // The shared decision (the CLI applies the same census and threshold).
    let outcome =
        session::cull(&set.frames, settings).map_err(|e| JobError::Failed(e.to_string()))?;
    if let Some(warning) = outcome.warning() {
        run.warn(warning);
    }
    let culled: Vec<PathBuf> = outcome
        .dropped
        .iter()
        .filter_map(|&index| set.paths.get(index).cloned())
        .collect();
    run.progress(RunStep::Cull, 1.0, "");
    Ok((outcome.order, culled))
}

/// Load, align, cull and fuse. `Ok` carries a note for the status line: the
/// frames that were skipped.
fn execute(
    post: Post,
    cancel: &CancelFlag,
    request: &StackRequest,
    held: &Held,
) -> Result<Option<String>, JobError> {
    let run = Run { post, cancel };
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    if request.settings.tile_size > 0 {
        return execute_tiled(&run, request, ts_ms);
    }
    run.progress(RunStep::Load, 0.0, "Loading frames...");

    let (set, stack_crop) = aligned_frames(&run, request, held)?;
    run.check("after aligning")?;

    let t_cull = Instant::now();
    let (order, culled) = cull_stage(&run, &request.settings, &set)?;
    let mut culled = Some(culled);
    tracing::info!(
        elapsed_ms = t_cull.elapsed().as_millis(),
        "cull stage complete"
    );
    run.check("after culling")?;

    let summary = RunSummary {
        total: set.len(),
        stack_total: order.len(),
        ts_ms,
    };
    let note = set.skipped_note();
    let ordered = session::reorder(set.frames, &order);
    for (index, &pass) in request.passes.iter().enumerate() {
        fuse_pass(
            &run,
            request,
            &Fusion {
                ordered: &ordered,
                stack_crop,
                pass,
                index,
            },
            culled.take(),
            &summary,
        )?;
    }
    Ok(note)
}

/// Fuse one pass, write its result and publish it to the UI. `culled` is the
/// list of files the auto-cull dropped, published with the first result.
fn fuse_pass(
    run: &Run<'_>,
    request: &StackRequest,
    fusion: &Fusion<'_>,
    culled: Option<Vec<PathBuf>>,
    summary: &RunSummary,
) -> Result<(), JobError> {
    run.check("before fusing")?;
    let label = fusion.pass.label();
    run.progress(
        RunStep::Fuse(fusion.index),
        0.0,
        &format!("Fusing [{label}] {} frames…", fusion.ordered.len()),
    );

    let t_fuse = Instant::now();
    let fused = fuse(run, request, fusion)?;
    tracing::info!(
        prefix = label,
        elapsed_ms = t_fuse.elapsed().as_millis(),
        "fuse stage complete"
    );
    run.check("after fusing")?;

    // Crop the result to the common valid area to remove black warp borders.
    // `resize_cropped_to_original` stretches the cropped result back to the canvas.
    let fused = session::crop_result_for(fused, fusion.stack_crop, &request.settings);
    finish_pass(
        run,
        fusion.pass,
        fusion.index,
        Arc::new(fused),
        culled,
        summary,
    )
}

/// Write the result PNG of a fused pass and publish it to the UI.
fn finish_pass(
    run: &Run<'_>,
    pass: Pass,
    index: usize,
    fused: Arc<PlanarImage<f32>>,
    culled: Option<Vec<PathBuf>>,
    summary: &RunSummary,
) -> Result<(), JobError> {
    let label = pass.label();
    run.progress(RunStep::Save(index), 0.0, &format!("Saving [{label}]…"));
    let temp_path = write_result_png(&fused, label, summary.ts_ms).map_err(JobError::Failed)?;
    if let Err(cancelled) = run.check("while saving the result") {
        // Never leave a result nobody will list.
        let _ = std::fs::remove_file(&temp_path);
        return Err(cancelled);
    }
    run.post.event(RunEvent::Result(StackResult {
        pass,
        fused,
        preview_path: temp_path,
        culled: culled.unwrap_or_default(),
        kept: summary.stack_total,
        total: summary.total,
    }));
    run.progress(RunStep::Save(index), 1.0, "");
    Ok(())
}

// ── The out-of-core path (`tile_size > 0`) ──────────────────────────────────

/// Maps the stages of a tiled run onto the steps of the job. The load, align and
/// census stages happen once, before the first fusion; each fusion reports on the
/// step of the pass `pass` names.
struct TiledProgress {
    post: Post,
    /// The pass being fused (picks the `Fuse` step).
    pass: Cell<usize>,
}

impl Progress for TiledProgress {
    fn stage(&self, stage: Stage, fraction: f32, label: &str) {
        let step = match stage {
            Stage::Fuse => RunStep::Fuse(self.pass.get()),
            Stage::Load => RunStep::Load,
            Stage::Align => RunStep::Align,
            Stage::Cull => RunStep::Cull,
            Stage::Save => return,
        };
        self.post.progress(step, fraction, label);
    }
}

/// Forwards a tiled run's messages: warnings reach the UI, the rest the log.
struct PostLog {
    post: Post,
}

impl RunLog for PostLog {
    fn info(&self, message: &str) {
        tracing::info!("{message}");
    }

    fn warn(&self, message: &str) {
        tracing::warn!("{message}");
        self.post.warning(message.to_owned());
    }
}

/// The tile engine of a pass, `None` for the AI pass (which needs the whole
/// stack in memory).
const fn method_of(pass: Pass) -> Option<Method> {
    match pass {
        Pass::Apex => Some(Method::Apex),
        Pass::Relief => Some(Method::Relief),
        Pass::Strata => Some(Method::Strata),
        Pass::Ai => None,
    }
}

/// The tiles of a tiled run on disk, with the grid, the fusion order and the crop
/// the alignment decided. The store is closed before its directory is removed (a
/// mapped file cannot be deleted on Windows), which the field order guarantees.
struct TiledStore {
    store: MmapTileManager,
    plan: TiledPlan,
    order: Vec<usize>,
    crop: Option<Crop>,
    _scratch: ScratchDir,
}

/// The job error of a pipeline error: a cancel stays a cancel.
fn pipeline_failure(error: PipelineError) -> JobError {
    match error {
        PipelineError::Cancelled => JobError::Cancelled("while stacking".to_owned()),
        other => JobError::Failed(other.to_string()),
    }
}

/// The tile grid of a run over `size` pixels fusing with `methods`.
///
/// One engine gets its own geometry. Several share one tile store, so the grid is
/// the `Apex` one (the tile size rounded for the pyramid) with the largest apron
/// any of them needs. Returns the grid and the method it was planned for.
fn tiled_plan_for(
    methods: &[Method],
    size: (usize, usize),
    tile_size: usize,
    settings: &StackingSettings,
) -> (TiledPlan, Method) {
    if let [only] = methods {
        return (tiled::plan_run(*only, size, tile_size, settings), *only);
    }
    let mut plan = tiled::plan_run(Method::Apex, size, tile_size, settings);
    let apron = methods
        .iter()
        .map(|&method| apron_for(method, settings))
        .max()
        .unwrap_or(plan.geometry.apron);
    plan.geometry.apron = apron;
    plan.tiles = plan_tiles(size.0, size.1, plan.geometry.tile_size, apron);
    (plan, Method::Apex)
}

/// Decode the reference, align every frame into a scratch tile store (once, for
/// all the passes of the run) and decide the fusion order and the crop.
fn align_into_tiles(
    request: &StackRequest,
    methods: &[Method],
    model: Option<AlignModel>,
    compute: &mut Compute,
    hooks: &Hooks<'_>,
) -> Result<TiledStore, PipelineError> {
    let settings = &request.settings;
    let tile_size = tiled::effective_tile_size(0, settings);
    if tile_size < tiled::MIN_TILE_SIZE {
        return Err(PipelineError::invalid(format!(
            "The tile size must be at least {} pixels (got {tile_size}).",
            tiled::MIN_TILE_SIZE
        )));
    }
    let mut source = PathSource::new(request.paths.clone());
    let n_frames = source.len();
    if n_frames == 0 {
        return Err(PipelineError::invalid(
            "No images to process: add some images first.",
        ));
    }
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
    let (plan, plan_method) = tiled_plan_for(methods, size, tile_size, settings);
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
    if let Some(warning) = plan.padding_warning(plan_method) {
        hooks.log.warn(&warning);
    }

    // The guard owns the scratch directory: whatever happens below (an error, a
    // cancel, success) the tiles are removed when it drops.
    let scratch = ScratchDir::create(&std::env::temp_dir())?;
    let mut store = MmapTileManager::new(scratch.path().to_path_buf());
    hooks
        .log
        .info(&format!("aligning and tiling {n_frames} frames"));
    let coverage = tiled::align_into_store_with_model(
        &mut source,
        &reference,
        settings,
        model,
        &plan.tiles,
        &mut store,
        compute,
        hooks,
    )?;
    drop(reference);

    hooks.progress.stage(Stage::Cull, 0.0, "");
    let order = tiled::frame_order(settings, &plan, &store, n_frames, hooks)?;
    hooks.progress.stage(Stage::Cull, 1.0, "");
    let crop = tiled::crop_rect(&coverage, settings, size, |i| source.name(i).to_owned());
    Ok(TiledStore {
        store,
        plan,
        order,
        crop,
        _scratch: scratch,
    })
}

/// A stack run out of core: the frames are decoded and aligned ONCE into a
/// scratch tile store in the system temporary folder (with the neural model when
/// the mode asks for it), then culled, and every pass fuses tile by tile from that
/// store; the result is cropped (and, with `resize_cropped_to_original`, resized
/// back to the canvas). The held aligned set is neither used nor replaced. The
/// `Relief` contrast preview is not asked; the settings' value is used.
fn execute_tiled(
    run: &Run<'_>,
    request: &StackRequest,
    ts_ms: u128,
) -> Result<Option<String>, JobError> {
    let settings = &request.settings;
    let tile_size = tiled::effective_tile_size(0, settings);
    if settings.relief_show_preview && request.passes.contains(&Pass::Relief) {
        run.warn(
            "The Relief contrast preview is not shown with a tile size above 0; the setting's \
             value is used."
                .to_owned(),
        );
    }
    let methods = request
        .passes
        .iter()
        .map(|&pass| method_of(pass))
        .collect::<Option<Vec<Method>>>()
        .ok_or_else(|| {
            JobError::Failed(
                "The AI model needs the whole stack in memory: set Tile Size to 0.".to_owned(),
            )
        })?;
    // The alignment model is loaded (or the run fails) before any frame is decoded.
    let model = align_model_for(settings, request.align_model.as_ref())?;

    let progress = TiledProgress {
        post: run.post,
        pass: Cell::new(0),
    };
    let cancel = FlagCancel(run.cancel);
    let log = PostLog { post: run.post };
    let hooks = Hooks {
        progress: &progress,
        cancel: &cancel,
        log: &log,
    };
    // The objective, the warp and the fusion run on the device the settings ask for.
    let mut compute = Compute::for_settings(settings);
    run.check("before aligning")?;
    let tiles = align_into_tiles(request, &methods, model, &mut compute, &hooks)
        .map_err(pipeline_failure)?;
    if let Some(notice) = compute.take_notice() {
        run.warn(notice);
    }
    run.check("after aligning")?;

    let kept: std::collections::BTreeSet<usize> = tiles.order.iter().copied().collect();
    let mut culled = Some(
        request
            .paths
            .iter()
            .enumerate()
            .filter(|(i, _)| !kept.contains(i))
            .map(|(_, path)| path.clone())
            .collect::<Vec<PathBuf>>(),
    );
    let summary = RunSummary {
        total: request.paths.len(),
        stack_total: tiles.order.len(),
        ts_ms,
    };
    let size = tiles.plan.size;
    for (index, (&pass, &method)) in request.passes.iter().zip(&methods).enumerate() {
        run.check("before fusing")?;
        progress.pass.set(index);
        run.progress(
            RunStep::Fuse(index),
            0.0,
            &format!("Fusing [{}] in tiles of {tile_size} px…", pass.label()),
        );
        let mut sink = PlanarImage::<f32>::new(size.0, size.1);
        tiled::fuse_tiles(
            &FusePlan::new(method, settings, &tiles.plan, &tiles.order),
            &StoreFrames(&tiles.store),
            &mut compute,
            &mut sink,
            &hooks,
        )
        .map_err(pipeline_failure)?;
        if let Some(notice) = compute.take_notice() {
            run.warn(notice);
        }
        run.check("after fusing")?;
        let fused = session::crop_result_for(sink, tiles.crop, settings);
        finish_pass(run, pass, index, Arc::new(fused), culled.take(), &summary)?;
    }
    Ok(None)
}

/// Reports the `Fuse` stage of a library call as a step of the job.
struct StepProgress {
    post: Post,
    step: RunStep,
}

impl Progress for StepProgress {
    fn stage(&self, stage: Stage, fraction: f32, label: &str) {
        if matches!(stage, Stage::Fuse) {
            self.post.progress(self.step, fraction, label);
        }
    }
}

fn fuse(
    run: &Run<'_>,
    request: &StackRequest,
    fusion: &Fusion<'_>,
) -> Result<PlanarImage<f32>, JobError> {
    let settings = &request.settings;
    match fusion.pass {
        Pass::Ai => {
            // Live preview: update the canvas as each frame is folded into the
            // running composite (watch it sharpen), and advance the fuse step.
            let post = run.post;
            let step = RunStep::Fuse(fusion.index);
            let total = fusion.ordered.len();
            let mut done = 0_usize;
            let mut live = LivePreview::new(move |buffer| post.event(RunEvent::LiveFrame(buffer)));
            let on_progress = move |img: &PlanarImage<f32>| {
                done += 1;
                post.progress(step, fraction_of(done, total), "");
                live.offer(img);
            };
            let (model, device) = request.ai.as_ref().map_or(("", ""), |(model, device)| {
                (model.as_str(), device.as_str())
            });
            nn_fuse_planar(fusion.ordered, model, device, on_progress)
                .map_err(|msg| JobError::Failed(format!("AI stacking failed: {msg}")))
        }
        Pass::Apex => {
            let mut compute = Compute::for_settings(settings);
            run.progress(
                RunStep::Fuse(fusion.index),
                0.0,
                &format!(
                    "Fusing [Apex] {} frames on {}…",
                    fusion.ordered.len(),
                    compute.describe()
                ),
            );
            let fused = session::fuse_apex(&mut compute, fusion.ordered, settings);
            // Why the GPU was not used, or why it was given up part-way.
            if let Some(notice) = compute.take_notice() {
                run.warn(notice);
            }
            Ok(fused)
        }
        Pass::Strata => {
            let post = run.post;
            let step = RunStep::Fuse(fusion.index);
            run.progress(
                step,
                0.0,
                &format!("Fusing [Strata] {} frames…", fusion.ordered.len()),
            );
            // A snapshot of the composite costs one pass over the image: about six
            // per run keeps the canvas alive without doubling the work.
            let every = fusion.ordered.len().div_ceil(6).max(1);
            let mut live = LivePreview::new(move |buffer| post.event(RunEvent::LiveFrame(buffer)));
            let progress = StepProgress { post, step };
            Ok(session::fuse_strata(
                fusion.ordered,
                settings,
                &progress,
                every,
                |img| {
                    live.offer(img);
                },
            ))
        }
        Pass::Relief => {
            let mut compute = Compute::for_settings(settings);
            run.progress(
                RunStep::Fuse(fusion.index),
                0.0,
                &format!(
                    "Fusing [Relief] {} frames on {}…",
                    fusion.ordered.len(),
                    compute.describe()
                ),
            );
            // The threshold is evaluated once on the whole-image focus map, exactly
            // as the CLI does for its tiles; the preview, when on, is asked there.
            let fused = session::fuse_relief(&mut compute, fusion.ordered, settings, |max| {
                resolve_threshold(run, settings, max, fusion.ordered.first())
            });
            if let Some(notice) = compute.take_notice() {
                run.warn(notice);
            }
            fused.ok_or_else(|| JobError::Cancelled("at the Relief preview".to_owned()))
        }
    }
}

/// Decide the absolute contrast threshold on the whole-image focus map: from the
/// preview panel when it is enabled (the user's pick, a value of the threshold
/// table), else from the settings (`relief_contrast_pct`, or the median + 1.5·MAD
/// auto-detect with `relief_auto_detect`). `None` = cancelled.
fn resolve_threshold(
    run: &Run<'_>,
    settings: &StackingSettings,
    max: &PlanarImage<f32>,
    frame0: Option<&PlanarImage<f32>>,
) -> Option<f32> {
    if settings.relief_show_preview
        && let Some(frame0) = frame0
    {
        return run.post.ask(relief_question(max, frame0, settings));
    }
    Some(session::threshold_from_settings(max, settings))
}

/// What the preview panel renders from: the focus data reduced to panel size, the
/// full-size threshold of every slider position and the two starting values. The
/// panel never touches the full-size focus map.
fn relief_question(
    max: &PlanarImage<f32>,
    frame0: &PlanarImage<f32>,
    settings: &StackingSettings,
) -> ReliefAsk {
    let auto_pct = session::auto_contrast_pct(max);
    let (_, _, focus_max) = downsample_plane(&max.luma, max.width, max.height, PANEL_LONG_EDGE);
    ReliefAsk {
        base: downsample_to_long_edge(frame0, PANEL_LONG_EDGE),
        focus_max,
        thresholds: session::threshold_table(&max.luma),
        auto_pct,
        initial_pct: if settings.relief_auto_detect {
            auto_pct
        } else {
            settings.relief_contrast_pct
        },
    }
}

/// Write the result PNG that backs the results list and Save: the shared
/// 8-bit encoder (rounded sRGB, white is 255), the same one the CLI writes with.
fn write_result_png(fused: &PlanarImage<f32>, label: &str, ts_ms: u128) -> Result<PathBuf, String> {
    let t_encode = Instant::now();
    let saving = ImageSavingSettings {
        output_format: OutputFormat::Png,
        bit_depth: 8,
        ..ImageSavingSettings::default()
    };
    let composite = encode(fused, &saving);
    let temp_path = std::env::temp_dir().join(format!("stacker_result_{label}_{ts_ms}.png"));
    composite.save(&temp_path).map_err(|e| {
        // A failed write can leave a partial file behind.
        let _ = std::fs::remove_file(&temp_path);
        format!(
            "Could not write the result image {}: {e}",
            temp_path.display()
        )
    })?;
    tracing::info!(
        prefix = label,
        output_path = %temp_path.display(),
        format = "PNG",
        elapsed_ms = t_encode.elapsed().as_millis(),
        "encode stage complete"
    );
    Ok(temp_path)
}

// ── The Align pass ──────────────────────────────────────────────────────────

/// The Align worker: decode, align, and keep the aligned frames. Nothing becomes
/// the held set unless the whole pass finished, so a cancel leaves no
/// half-aligned set behind. `Ok` carries a note for the status line: the frames
/// that were skipped.
fn align(
    post: Post,
    cancel: &CancelFlag,
    request: &AlignRequest,
    held: &Held,
) -> Result<Option<String>, JobError> {
    let run = Run { post, cancel };
    let settings = &request.settings;
    run.progress(RunStep::Load, 0.0, "Align: loading frames…");

    let mut set = frames::decode_frames(request.paths.clone(), &settings.preprocessing)
        .map_err(|e| JobError::Failed(format!("Align: {e}")))?;
    if let Some(warning) = feedback::skipped_frames_warning(&set.skipped) {
        run.warn(warning);
    }
    run.check("after loading the frames")?;

    let n = set.len();
    let model = align_model_for(settings, request.align_model.as_ref())?;
    run.progress(RunStep::Align, 0.0, "Align: aligning frames…");
    // Throttled thumbnails while the pass runs; the last frame goes at once.
    let mut live = LivePreview::new(move |buffer| post.event(RunEvent::LiveFrame(buffer)));
    let mut compute = Compute::for_settings(settings);
    let outcome = session::align_frames_with_model(
        &mut set.frames,
        settings,
        model,
        &mut compute,
        &FlagCancel(cancel),
        &mut |i, frame, mask| {
            let region = largest_true_rectangle(mask, frame.width, frame.height);
            if i + 1 == n {
                live.offer_now(frame, region);
            } else {
                live.offer_region(frame, region);
            }
            run.progress(
                RunStep::Align,
                fraction_of(i, n.saturating_sub(1)),
                &format!("Aligning… frame {i}/{}", n - 1),
            );
        },
    )
    .ok_or_else(|| JobError::Cancelled("while aligning".to_owned()))?;
    run.check("after aligning")?;

    let warning = outcome.warning(&names_of(&set));
    let crop = outcome.crop;
    let note = set.skipped_note();
    let paths = set.paths.clone();
    *lock(held) = Some(AlignedSet {
        paths: set.paths,
        frames: set.frames.into_iter().map(Arc::new).collect(),
        crop,
    });
    post.event(RunEvent::AlignDone {
        paths,
        crop,
        warning,
    });
    Ok(note)
}

// ── Sort and Cull ───────────────────────────────────────────────────────────

/// The Sort / Cull worker: take the aligned frames (the held set when the GUI
/// says it is current, else decode and align, which then become the held set and
/// are announced as an Align result), count the sharp detail, and report the new
/// order or the frames to drop. Nothing is written to the held set unless the
/// whole run finished.
fn sort_cull(
    post: Post,
    cancel: &CancelFlag,
    request: &SortCullRequest,
    held: &Held,
) -> Result<Option<String>, JobError> {
    let run = Run { post, cancel };
    let settings = &request.settings;
    let what = request.action.label();
    run.progress(RunStep::Load, 0.0, &format!("{what}: loading frames…"));

    let reused = if request.reuse_aligned {
        lock(held)
            .as_ref()
            .and_then(|set| set.frames_for(&request.paths))
    } else {
        None
    };
    let mut note = None;
    // `fresh` carries the crop and the alignment warning of a run that aligned.
    let (paths, frames, fresh) = if let Some(reused) = reused {
        run.progress(RunStep::Load, 1.0, "Reusing the aligned frames…");
        run.progress(RunStep::Align, 1.0, "");
        let frames: Vec<PlanarImage<f32>> = reused.frames.into_iter().map(owned_frame).collect();
        (reused.paths, frames, None)
    } else {
        let mut set = frames::decode_frames(request.paths.clone(), &settings.preprocessing)
            .map_err(|e| JobError::Failed(format!("{what}: {e}")))?;
        if let Some(warning) = feedback::skipped_frames_warning(&set.skipped) {
            run.warn(warning);
        }
        run.progress(RunStep::Load, 1.0, "");
        run.check("after loading the frames")?;
        note = set.skipped_note();
        let crop = align_stage(&run, settings, request.align_model.as_ref(), &mut set)?;
        run.check("after aligning")?;
        // The warning of a fallback was shown by `align_stage`.
        (set.paths, set.frames, Some(crop))
    };

    run.progress(
        RunStep::Cull,
        0.0,
        &format!("{what}: counting sharp detail…"),
    );
    let outcome = match request.action {
        SortCullAction::Sort => session::sort_frames_by_sharpness(&frames, settings),
        SortCullAction::Cull => session::cull_frames(&frames, settings),
    }
    .map_err(|e| JobError::Failed(e.to_string()))?;
    if let Some(warning) = outcome.warning() {
        run.warn(warning);
    }
    run.check("after the census")?;
    let name = |indices: &[usize]| -> Vec<PathBuf> {
        indices
            .iter()
            .filter_map(|&index| paths.get(index).cloned())
            .collect()
    };
    let order = name(&outcome.order);
    let dropped = name(&outcome.dropped);

    if let Some(crop) = fresh {
        let aligned_paths = paths.clone();
        *lock(held) = Some(AlignedSet {
            paths,
            frames: frames.into_iter().map(Arc::new).collect(),
            crop,
        });
        post.event(RunEvent::AlignDone {
            paths: aligned_paths,
            crop,
            warning: None,
        });
    }
    run.progress(RunStep::Cull, 1.0, "");
    post.event(RunEvent::SortCullDone {
        action: request.action,
        order,
        dropped,
    });
    Ok(note)
}

// ── AI (neural) stacking bridge ─────────────────────────────────────────────

/// The alignment network `name` on the device labelled `device` (an unknown name
/// or label falls back to the first model / device). Native `nn` builds only.
#[cfg(feature = "nn")]
fn build_align_model(name: &str, device: &str) -> Result<AlignModel, String> {
    let models = stacker_nn::discover_default_alignment_models();
    let entry = models
        .iter()
        .find(|m| m.name == name)
        .or_else(|| models.first())
        .ok_or_else(|| "no alignment models found in the models/ folder".to_owned())?;
    let device = stacker_nn::available_devices()
        .into_iter()
        .find(|d| d.label().eq_ignore_ascii_case(device))
        .or_else(|| stacker_nn::available_devices().into_iter().next())
        .ok_or_else(|| "no inference backend available".to_owned())?;
    Ok(AlignModel::new(entry.clone(), device))
}

/// Stub for non-`nn` builds: the neural alignment mode is unavailable.
#[cfg(not(feature = "nn"))]
fn build_align_model(_name: &str, _device: &str) -> Result<AlignModel, String> {
    Err(
        "this build has no neural alignment (recompile with --features nn); choose another \
         alignment mode"
            .to_owned(),
    )
}

/// Fuse a stack with a trained neural model selected in the UI. Available only
/// in `nn` builds; the fallback below keeps the call site feature-agnostic.
#[cfg(feature = "nn")]
fn nn_fuse_planar<F>(
    frames: &[PlanarImage<f32>],
    model_name: &str,
    device_name: &str,
    on_progress: F,
) -> Result<PlanarImage<f32>, String>
where
    F: FnMut(&PlanarImage<f32>),
{
    let models = stacker_nn::discover_default_fusion_models();
    let entry = models
        .iter()
        .find(|m| m.name == model_name)
        .or_else(|| models.first())
        .ok_or_else(|| "no AI models found in the models/ folder".to_owned())?;
    let device = stacker_nn::available_devices()
        .into_iter()
        .find(|d| d.label().eq_ignore_ascii_case(device_name))
        .or_else(|| stacker_nn::available_devices().into_iter().next())
        .ok_or_else(|| "no inference backend available".to_owned())?;
    let model = stacker_nn::LoadedModel::load(entry, device).map_err(|e| e.to_string())?;
    // The overlap follows the selected model's receptive field (32 px for `xs`).
    let tile = stacker_nn::runtime::recommended_tile_for(entry);
    model
        .fuse_with_progress(frames, tile, on_progress)
        .map_err(|e| e.to_string())
}

/// Stub for non-`nn` builds: AI stacking is unavailable.
#[cfg(not(feature = "nn"))]
fn nn_fuse_planar<F>(
    _frames: &[PlanarImage<f32>],
    _model_name: &str,
    _device_name: &str,
    _on_progress: F,
) -> Result<PlanarImage<f32>, String>
where
    F: FnMut(&PlanarImage<f32>),
{
    Err("this build has no AI support (recompile with --features nn)".to_owned())
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use stacker_core::image::PlanarImage;

    use super::AlignedSet;

    fn path(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    fn marked_image(w: usize, h: usize, marker: f32) -> PlanarImage<f32> {
        let mut image = PlanarImage::new(w, h);
        image.luma[0] = marker;
        image
    }

    #[test]
    fn the_set_is_looked_up_by_path_not_by_position() {
        // The set was built from a reversed, filtered list: positions differ from
        // the sidebar's, paths do not.
        let set = AlignedSet {
            paths: vec![path("c.png"), path("a.png")],
            frames: vec![
                Arc::new(marked_image(4, 4, 3.0)),
                Arc::new(marked_image(4, 4, 1.0)),
            ],
            crop: None,
        };
        let a = set.frame_for(&path("a.png")).unwrap();
        assert_eq!(a.luma[0].to_bits(), 1.0_f32.to_bits());
        let c = set.frame_for(&path("c.png")).unwrap();
        assert_eq!(c.luma[0].to_bits(), 3.0_f32.to_bits());
        assert!(set.frame_for(&path("b.png")).is_none());
    }

    #[test]
    fn the_wanted_order_wins_and_a_missing_file_refuses_the_reuse() {
        let set = AlignedSet {
            paths: vec![path("a.png"), path("b.png"), path("c.png")],
            frames: vec![
                Arc::new(marked_image(2, 2, 0.0)),
                Arc::new(marked_image(2, 2, 1.0)),
                Arc::new(marked_image(2, 2, 2.0)),
            ],
            crop: Some((0, 0, 1, 1)),
        };
        let reused = set.frames_for(&[path("c.png"), path("a.png")]).unwrap();
        assert_eq!(reused.paths, vec![path("c.png"), path("a.png")]);
        assert!((reused.frames[0].luma[0] - 2.0).abs() < 1e-6);
        assert!((reused.frames[1].luma[0] - 0.0).abs() < 1e-6);
        assert_eq!(reused.crop, Some((0, 0, 1, 1)));
        assert!(set.frames_for(&[path("a.png"), path("z.png")]).is_none());
        assert!(set.frames_for(&[]).is_none());
    }

    #[test]
    fn the_view_applies_the_common_crop() {
        let set = AlignedSet {
            paths: vec![path("a.png")],
            frames: vec![Arc::new(marked_image(6, 4, 1.0))],
            crop: Some((1, 1, 3, 2)),
        };
        let view = set.view_for(&path("a.png")).unwrap();
        assert_eq!(
            (view.frame.width, view.frame.height),
            (6, 4),
            "the view shares the whole frame; no pixels are cut yet"
        );
        assert!(
            Arc::ptr_eq(&view.frame, &set.frames[0]),
            "the view shares the held frame instead of copying it"
        );
        let cut = view.cropped();
        assert_eq!((cut.width, cut.height), (3, 2));
        assert!(set.view_for(&path("b.png")).is_none());
    }

    #[test]
    fn an_uncropped_view_copies_the_whole_frame_on_demand() {
        let set = AlignedSet {
            paths: vec![path("a.png")],
            frames: vec![Arc::new(marked_image(5, 3, 7.0))],
            crop: None,
        };
        let copy = set.view_for(&path("a.png")).unwrap().cropped();
        assert_eq!((copy.width, copy.height), (5, 3));
        assert_eq!(copy.luma[0].to_bits(), 7.0_f32.to_bits());
    }
}
