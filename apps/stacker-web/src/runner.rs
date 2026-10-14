//! The browser's [`JobRunner`]: every heavy job goes to one Web Worker.
//!
//! A run sends the files (as named byte buffers) and the settings (as TOML) to the
//! worker and turns its replies into [`RunEvent`]s on the UI thread. The worker is
//! started lazily, and messages wait until it has announced itself with
//! [`FromWorker::Ready`]. It cannot read a message while it computes, so a cancel
//! terminates it and the next job starts a fresh one; its aligned set is lost with
//! it, and every run sends the files again, so the next run simply aligns again
//! (exactly as after a settings change).
//!
//! # WebGPU
//!
//! Alignment in the browser runs on the GPU only. `Ready` carries the adapter the
//! worker opened (or `None`); the runner remembers it. A stack that needs
//! alignment asks before it starts when there is no WebGPU ("Stack without
//! alignment?"), starting the worker first and waiting for `Ready` when the
//! answer is not known yet (the continuation parks in `Inner::waiters`).
//!
//! # Re-entrancy
//!
//! Nothing here calls user code (an events callback, a waiter) while the runner's
//! state is borrowed, and everything that can finish a job from inside a call that
//! started one (a failed post, a cancel) delivers it on the next turn of the event
//! loop instead.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    path::{Path, PathBuf},
    rc::{Rc, Weak},
    sync::Arc,
    time::Duration,
};

use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, ImageSavingSettings, OutputFormat, StackingSettings},
};
use stacker_gui::{
    gui::{
        Ctx,
        confirm::{self, Prompt},
        feedback::{self, MessageKind},
        stack::Pass,
    },
    jobs_core::JobError,
    runner::{
        AlignRequest, AskRelief, Events, JobHandle, JobRunner, ReliefAsk, RunEvent, RunStep,
        StackRequest, StackResult,
    },
};
use stacker_pipeline::{load::decode_frame_bytes, output};
use stacker_web_core::{
    Budget, FromWorker, NamedBytes, PROTOCOL_VERSION, PassKind as WirePass, PlanarF32, Stage,
    ToWorker, dims_from_bytes, plan_budget,
};

use crate::{vfs, worker::WorkerHandle};

/// The longest edge of a source-file viewer picture; `0` asks the worker for the
/// full size.
const FULL_SIZE: u32 = 0;

type SourceDone = Box<dyn FnOnce(Result<PlanarImage<f32>, String>)>;
type PreviewDone = Box<dyn FnOnce(Result<SharedPixelBuffer<Rgba8Pixel>, String>)>;
type Preflighted = Box<dyn FnOnce(Option<(StackRequest, Option<String>)>)>;
type Shared = Rc<RefCell<Inner>>;
/// Something that waits for the worker's `Ready`: `Ok` when it came, `Err` with
/// the reason when the worker stopped first.
type ReadyWaiter = Box<dyn FnOnce(Result<(), String>)>;

/// The status note of a stack that runs without alignment for want of WebGPU.
const NOT_ALIGNED_NOTE: &str = "not aligned (no WebGPU)";
/// The question when the browser has no WebGPU.
const NO_WEBGPU_QUESTION: &str = "Alignment in the browser runs on the GPU (WebGPU), which this \
                                  browser does not offer. Stack without alignment?";
/// The sentence for an Align that cannot run.
const NO_WEBGPU_SENTENCE: &str =
    "Alignment in the browser runs on the GPU (WebGPU), which this browser does not offer.";

/// The refusal of the Neural alignment mode, which only the desktop app offers.
const NEURAL_MESSAGE: &str =
    "Neural alignment is only in the desktop app. Choose Translation, Registration or Affine.";

/// Run `f` on the next turn of the event loop.
fn deliver_later(f: impl FnOnce() + 'static) {
    slint::Timer::single_shot(Duration::ZERO, f);
}

/// A run in flight (a Stack or an Align).
struct Job {
    events: RefCell<Events>,
    ask_relief: RefCell<Option<AskRelief>>,
    /// File name to virtual path, to turn the worker's names back into paths.
    names: HashMap<String, PathBuf>,
    passes: Vec<Pass>,
    /// `relief_auto_detect` and `relief_contrast_pct` of the run's settings: the
    /// preview's starting position.
    auto_detect: bool,
    contrast_pct: f32,
    /// The results received so far: the index of the pass in flight.
    results: Cell<usize>,
    /// The note of the last result.
    note: RefCell<Option<String>>,
    /// An Align run's own note (it was reduced to fit the browser's memory).
    align_note: Option<String>,
    /// The EXIF block of the first frame, fitted to the output, when
    /// `copy_metadata` is on and the frame has one: written into every result's
    /// file.
    exif: Option<Vec<u8>>,
}

impl Job {
    /// Hand `event` to the run's owner. An event that arrives while the owner's
    /// callback is itself running is dropped (it cannot be, see the module docs).
    fn emit(&self, event: RunEvent) {
        if let Ok(mut events) = self.events.try_borrow_mut() {
            (*events)(event);
        }
    }
}

/// A `fetch_source` waiting for the worker.
struct SourceWaiter {
    path: PathBuf,
    settings: StackingSettings,
    then: SourceDone,
}

impl SourceWaiter {
    /// Answer from the page: decode the file with the settings' preprocessing.
    fn fallback(self) {
        let result = decode_source(&self.path, &self.settings);
        (self.then)(result);
    }
}

/// A `fetch_preview` waiting for the worker.
struct PreviewWaiter {
    path: PathBuf,
    then: PreviewDone,
}

impl PreviewWaiter {
    /// Answer from the page: decode the file here.
    fn fallback(self) {
        let result = vfs::get(&self.path).map_or_else(
            || {
                Err(format!(
                    "{} is no longer loaded.",
                    vfs::file_name(&self.path)
                ))
            },
            |bytes| stacker_gui::preview::decode_bytes(&bytes),
        );
        (self.then)(result);
    }
}

/// What a stopped worker leaves waiting.
struct Orphans {
    jobs: Vec<Rc<Job>>,
    sources: Vec<SourceWaiter>,
    previews: Vec<PreviewWaiter>,
    ready: Vec<ReadyWaiter>,
}

impl Orphans {
    /// Tell every waiting job it failed with `message`, and answer the viewers
    /// from the page, on the next turn of the event loop.
    fn fail_later(self, message: String) {
        deliver_later(move || {
            for waiter in self.ready {
                waiter(Err(message.clone()));
            }
            for job in self.jobs {
                job.emit(RunEvent::Finished(Err(JobError::Failed(message.clone()))));
            }
            for waiter in self.sources {
                waiter.fallback();
            }
            for waiter in self.previews {
                waiter.fallback();
            }
        });
    }
}

/// What the worker has said about WebGPU.
enum GpuState {
    /// The worker has not announced itself yet.
    Unknown,
    /// The worker announced itself: the adapter name, or `None` for a browser
    /// without WebGPU.
    Reported(Option<String>),
}

/// The runner's state.
struct Inner {
    worker: Option<WorkerHandle>,
    /// The worker has announced itself.
    ready: bool,
    /// Messages posted before it did.
    queue: Vec<ToWorker>,
    next_id: u64,
    jobs: HashMap<u64, Rc<Job>>,
    sources: HashMap<u64, SourceWaiter>,
    previews: HashMap<u64, PreviewWaiter>,
    /// The Pre-resize percentage the memory budget chose for the next run.
    reduction: Option<u32>,
    /// What the worker said about WebGPU in `Ready`.
    gpu: GpuState,
    /// Continuations waiting for the worker's `Ready`.
    waiters: Vec<ReadyWaiter>,
}

impl Inner {
    fn new() -> Self {
        Self {
            worker: None,
            ready: false,
            queue: Vec::new(),
            next_id: 1,
            jobs: HashMap::new(),
            sources: HashMap::new(),
            previews: HashMap::new(),
            reduction: None,
            gpu: GpuState::Unknown,
            waiters: Vec::new(),
        }
    }

    /// `Some(true)` when the worker has WebGPU, `Some(false)` when it has none,
    /// `None` while the worker has not said.
    const fn has_webgpu(&self) -> Option<bool> {
        match self.gpu {
            GpuState::Unknown => None,
            GpuState::Reported(ref adapter) => Some(adapter.is_some()),
        }
    }

    const fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Stop the worker and take everything that was waiting for it.
    fn stop_worker(&mut self) -> Orphans {
        if let Some(worker) = self.worker.take() {
            worker.retire();
        }
        self.ready = false;
        self.queue.clear();
        Orphans {
            jobs: self.jobs.drain().map(|(_, job)| job).collect(),
            sources: self.sources.drain().map(|(_, waiter)| waiter).collect(),
            previews: self.previews.drain().map(|(_, waiter)| waiter).collect(),
            ready: std::mem::take(&mut self.waiters),
        }
    }
}

/// The runner that talks to the Web Worker.
pub struct WebRunner {
    shared: Shared,
}

impl WebRunner {
    /// A runner with no worker yet (it starts with the first job).
    #[must_use]
    pub fn new() -> Self {
        Self {
            shared: Rc::new(RefCell::new(Inner::new())),
        }
    }
}

impl Default for WebRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Install a [`WebRunner`] as the page's job runner.
pub fn install_runner() {
    stacker_gui::runner::install(Rc::new(WebRunner::new()));
}

// ── Talking to the worker ───────────────────────────────────────────────────

fn spawn_worker(shared: &Shared) -> Result<WorkerHandle, String> {
    let on_message = Rc::downgrade(shared);
    let on_error = Rc::downgrade(shared);
    WorkerHandle::spawn(
        move |message| {
            if let Some(shared) = on_message.upgrade() {
                dispatch(&shared, message);
            }
        },
        move |error| {
            if let Some(shared) = on_error.upgrade() {
                fail_all(
                    &shared,
                    format!(
                        "The browser worker stopped ({error}). Stacking large images in a browser \
                         can run out of memory: lower Pre-resize % and try again."
                    ),
                );
            }
        },
    )
}

/// Start the worker when there is none (it announces itself with `Ready`).
fn ensure_worker(shared: &Shared) -> Result<(), String> {
    let mut inner = shared.borrow_mut();
    if inner.worker.is_none() {
        inner.worker = Some(spawn_worker(shared)?);
        inner.ready = false;
        inner.queue.clear();
    }
    Ok(())
}

/// Post `message`, starting the worker first when there is none.
fn try_send(shared: &Shared, message: ToWorker) -> Result<(), String> {
    ensure_worker(shared)?;
    let mut inner = shared.borrow_mut();
    if inner.ready {
        inner
            .worker
            .as_ref()
            .map_or(Ok(()), |worker| worker.post(&message))
    } else {
        inner.queue.push(message);
        Ok(())
    }
}

/// Post `message`; when that fails, every waiting job fails.
fn send(shared: &Shared, message: ToWorker) {
    if let Err(error) = try_send(shared, message) {
        fail_all(shared, error);
    }
}

/// Stop the worker and fail everything that waits for it.
fn fail_all(shared: &Shared, message: String) {
    let orphans = shared.borrow_mut().stop_worker();
    orphans.fail_later(message);
}

/// The worker announced itself: check the protocol, remember its WebGPU, then
/// send the version and the messages that waited, and wake the continuations
/// that waited for the announcement.
fn on_ready(shared: &Shared, version: u32, gpu_adapter: Option<String>) {
    if version != PROTOCOL_VERSION {
        fail_all(
            shared,
            format!(
                "The page speaks protocol {PROTOCOL_VERSION} but the worker speaks {version}: \
                 reload the page to load matching files."
            ),
        );
        return;
    }
    let first = matches!(shared.borrow().gpu, GpuState::Unknown);
    if first {
        show_gpu_status(gpu_adapter.as_deref());
    }
    let (result, waiters) = {
        let mut inner = shared.borrow_mut();
        inner.ready = true;
        inner.gpu = GpuState::Reported(gpu_adapter);
        let queued = std::mem::take(&mut inner.queue);
        let result = inner.worker.as_ref().map_or(Ok(()), |worker| {
            worker.post(&ToWorker::Hello {
                version: PROTOCOL_VERSION,
            })?;
            queued.iter().try_for_each(|message| worker.post(message))
        });
        (result, std::mem::take(&mut inner.waiters))
    };
    if let Err(error) = result {
        fail_all(shared, error.clone());
        for waiter in waiters {
            waiter(Err(error.clone()));
        }
        return;
    }
    // No borrow is held: a waiter may start a run or ask a question.
    for waiter in waiters {
        waiter(Ok(()));
    }
}

/// Say once in the status line whether the browser has WebGPU.
fn show_gpu_status(adapter: Option<&str>) {
    let Some(ctx) = Ctx::current() else {
        return;
    };
    if let Some(app) = ctx.ui.upgrade() {
        let text = adapter.map_or_else(
            || "No WebGPU: alignment unavailable".to_owned(),
            |name| format!("GPU: {name}"),
        );
        feedback::show_status(&app, &text);
    }
}

fn find_job(shared: &Shared, job_id: u64) -> Option<Rc<Job>> {
    shared.borrow().jobs.get(&job_id).cloned()
}

/// End the run `job_id` with `result` (once: a run that is gone stays silent).
fn finish(shared: &Shared, job_id: u64, result: Result<Option<String>, JobError>) {
    let job = shared.borrow_mut().jobs.remove(&job_id);
    if let Some(job) = job {
        job.emit(RunEvent::Finished(result));
    }
}

/// A picture from the worker's RGBA bytes; `None` when the size does not match.
fn rgba_buffer(width: u32, height: u32, rgba: &[u8]) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
    let expected = u64::from(width) * u64::from(height) * 4;
    if u64::try_from(rgba.len()).ok()? != expected || width == 0 || height == 0 {
        return None;
    }
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    buffer.make_mut_bytes().copy_from_slice(rgba);
    Some(buffer)
}

const fn pass_from_wire(pass: WirePass) -> Pass {
    match pass {
        WirePass::Apex => Pass::Apex,
        WirePass::Relief => Pass::Relief,
        WirePass::Strata => Pass::Strata,
    }
}

const fn wire_pass(pass: Pass) -> Option<WirePass> {
    match pass {
        Pass::Apex => Some(WirePass::Apex),
        Pass::Relief => Some(WirePass::Relief),
        Pass::Strata => Some(WirePass::Strata),
        Pass::Ai => None,
    }
}

const fn pass_label(pass: Pass) -> &'static str {
    match pass {
        Pass::Apex => "Apex",
        Pass::Relief => "Relief",
        Pass::Strata => "Strata",
        Pass::Ai => "AI",
    }
}

/// The EXIF block to carry into the results: the first frame's, fitted to the
/// output (`None` when `copy_metadata` is off or the frame holds none).
fn first_frame_exif(paths: &[PathBuf], settings: &StackingSettings) -> Option<Vec<u8>> {
    if !settings.image_saving.copy_metadata {
        return None;
    }
    let bytes = vfs::get(paths.first()?)?;
    output::source_exif_from_bytes(&bytes, &settings.preprocessing, &settings.image_saving)
}

/// The 8-bit PNG that backs the results list, registered as a virtual file: the
/// shared encoder (rounded sRGB, white is 255) the CLI and the desktop write with.
/// `exif` is the source frame's EXIF block, written into the file when given.
fn preview_png(
    image: &PlanarImage<f32>,
    label: &str,
    exif: Option<Vec<u8>>,
) -> Result<PathBuf, String> {
    let saving = ImageSavingSettings {
        output_format: OutputFormat::Png,
        bit_depth: 8,
        ..ImageSavingSettings::default()
    };
    let bytes = output::encode(image, &saving)
        .with_exif(exif)
        .to_bytes()
        .map_err(|error| format!("Could not encode the result image: {error}"))?;
    Ok(vfs::register_result(label, bytes))
}

/// Handle one reply of the worker: its announcement and the reports of a run in
/// flight here, everything else in [`dispatch_data`].
fn dispatch(shared: &Shared, message: FromWorker) {
    match message {
        FromWorker::Ready {
            version,
            gpu_adapter,
        } => on_ready(shared, version, gpu_adapter),
        FromWorker::Progress {
            job_id,
            stage,
            fraction,
            label,
        } => on_progress(shared, job_id, stage, fraction, label),
        FromWorker::Warning { job_id, text } => {
            if let Some(job) = find_job(shared, job_id) {
                job.emit(RunEvent::Warning(text));
            }
        }
        FromWorker::AlignedFrame {
            job_id,
            preview_rgba,
            width,
            height,
            ..
        } => {
            if let (Some(job), Some(frame)) = (
                find_job(shared, job_id),
                rgba_buffer(width, height, &preview_rgba),
            ) {
                job.emit(RunEvent::LiveFrame(frame));
            }
        }
        other => dispatch_data(shared, other),
    }
}

/// The replies that carry data or end a run (see [`dispatch`]).
fn dispatch_data(shared: &Shared, message: FromWorker) {
    match message {
        FromWorker::ReliefPreview {
            job_id,
            width,
            height,
            focus_max,
            base_rgba,
            auto_pct,
            thresholds,
        } => on_relief_preview(
            shared,
            job_id,
            ReliefData {
                width,
                height,
                focus_max,
                base_rgba,
                auto_pct,
                thresholds,
            },
        ),
        FromWorker::Result {
            job_id,
            pass,
            planes,
            culled,
            kept,
            total,
            note,
        } => on_result(
            shared,
            job_id,
            pass,
            ResultData {
                planes,
                culled,
                kept,
                total,
                note,
            },
        ),
        FromWorker::AlignDone {
            job_id,
            crop,
            warning,
            names,
        } => on_align_done(shared, job_id, crop, warning, &names),
        FromWorker::Source {
            request_id, planes, ..
        } => {
            let waiter = shared.borrow_mut().sources.remove(&request_id);
            if let Some(waiter) = waiter {
                (waiter.then)(Ok(planes.into_planar()));
            }
        }
        FromWorker::SourceFailed { request_id, .. } => {
            let waiter = shared.borrow_mut().sources.remove(&request_id);
            if let Some(waiter) = waiter {
                waiter.fallback();
            }
        }
        FromWorker::Failed { job_id: 0, message } => fail_all(shared, message),
        FromWorker::Failed { job_id, message } => {
            finish(shared, job_id, Err(JobError::Failed(message)));
        }
        FromWorker::Done { job_id } => {
            let note = find_job(shared, job_id).and_then(|job| job.note.borrow_mut().take());
            finish(shared, job_id, Ok(note));
        }
        FromWorker::Preview {
            request_id,
            width,
            height,
            rgba,
            ..
        } => on_preview(shared, request_id, width, height, &rgba),
        FromWorker::Cancelled { job_id } => finish(
            shared,
            job_id,
            Err(JobError::Cancelled("at the Relief preview".to_owned())),
        ),
        FromWorker::PreviewFailed { request_id, .. } => {
            let waiter = shared.borrow_mut().previews.remove(&request_id);
            if let Some(waiter) = waiter {
                waiter.fallback();
            }
        }
        // Handled by `dispatch`.
        FromWorker::Ready { .. }
        | FromWorker::Progress { .. }
        | FromWorker::Warning { .. }
        | FromWorker::AlignedFrame { .. } => {}
    }
}

/// A progress report of the worker, as the run step it belongs to.
fn on_progress(shared: &Shared, job_id: u64, stage: Stage, fraction: f32, label: String) {
    let Some(job) = find_job(shared, job_id) else {
        return;
    };
    let pass = job.results.get().min(job.passes.len().saturating_sub(1));
    let step = match stage {
        Stage::Load => RunStep::Load,
        Stage::Align => RunStep::Align,
        Stage::Cull => RunStep::Cull,
        Stage::Fuse => RunStep::Fuse(pass),
        Stage::Save => RunStep::Save(pass),
    };
    job.emit(RunEvent::Progress {
        step,
        fraction,
        label,
    });
}

/// The fields of [`FromWorker::Result`] after the job id and the pass.
struct ResultData {
    planes: PlanarF32,
    culled: Vec<String>,
    kept: usize,
    total: usize,
    note: Option<String>,
}

/// One finished fusion pass: register its preview PNG and hand the result to the GUI.
fn on_result(shared: &Shared, job_id: u64, pass: WirePass, data: ResultData) {
    let Some(job) = find_job(shared, job_id) else {
        return;
    };
    let pass = pass_from_wire(pass);
    let fused = Arc::new(data.planes.into_planar());
    match preview_png(&fused, pass_label(pass), job.exif.clone()) {
        Ok(preview_path) => {
            job.results.set(job.results.get() + 1);
            if data.note.is_some() {
                *job.note.borrow_mut() = data.note;
            }
            let culled = data
                .culled
                .iter()
                .filter_map(|name| job.names.get(name).cloned())
                .collect();
            job.emit(RunEvent::Result(StackResult {
                pass,
                fused,
                preview_path,
                culled,
                kept: data.kept,
                total: data.total,
            }));
        }
        Err(message) => finish(shared, job_id, Err(JobError::Failed(message))),
    }
}

/// The Align run finished: report the aligned files and the crop, then end the job.
fn on_align_done(
    shared: &Shared,
    job_id: u64,
    crop: Option<stacker_web_core::Rect>,
    warning: Option<String>,
    names: &[String],
) {
    let Some(job) = find_job(shared, job_id) else {
        return;
    };
    let warning = match (warning, job.align_note.clone()) {
        (Some(warning), Some(note)) => Some(format!("{note} {warning}")),
        (warning, note) => warning.or(note),
    };
    let paths = names
        .iter()
        .filter_map(|name| job.names.get(name).cloned())
        .collect();
    job.emit(RunEvent::AlignDone {
        paths,
        crop: crop.map(stacker_web_core::Rect::into_crop),
        warning,
    });
    finish(shared, job_id, Ok(None));
}

/// The picture a preview request waited for: hand it over, or fall back.
fn on_preview(shared: &Shared, request_id: u64, width: u32, height: u32, rgba: &[u8]) {
    let waiter = shared.borrow_mut().previews.remove(&request_id);
    if let Some(waiter) = waiter {
        match rgba_buffer(width, height, rgba) {
            Some(picture) => (waiter.then)(Ok(picture)),
            None => waiter.fallback(),
        }
    }
}

/// The fields of [`FromWorker::ReliefPreview`] after the job id.
struct ReliefData {
    width: u32,
    height: u32,
    focus_max: Vec<f32>,
    base_rgba: Vec<u8>,
    auto_pct: f32,
    thresholds: Vec<f32>,
}

/// The worker waits at the contrast preview: ask the GUI, and post its answer.
fn on_relief_preview(shared: &Shared, job_id: u64, data: ReliefData) {
    let Some(job) = find_job(shared, job_id) else {
        return;
    };
    let Some(base) = rgba_buffer(data.width, data.height, &data.base_rgba) else {
        finish(
            shared,
            job_id,
            Err(JobError::Failed(
                "The worker sent a damaged contrast preview.".to_owned(),
            )),
        );
        return;
    };
    let initial_pct = if job.auto_detect {
        data.auto_pct
    } else {
        job.contrast_pct
    };
    let ask = ReliefAsk {
        base,
        focus_max: data.focus_max,
        thresholds: data.thresholds,
        auto_pct: data.auto_pct,
        initial_pct,
    };
    let weak = Rc::downgrade(shared);
    let answer: Box<dyn FnOnce(Option<f32>)> = Box::new(move |threshold| {
        if let Some(shared) = weak.upgrade() {
            // A run that was cancelled meanwhile has no worker to answer.
            if shared.borrow().jobs.contains_key(&job_id) {
                send(&shared, ToWorker::ReliefThreshold { job_id, threshold });
            }
        }
    });
    if let Ok(mut guard) = job.ask_relief.try_borrow_mut()
        && let Some(ask_relief) = guard.as_mut()
    {
        ask_relief(ask, answer);
    }
}

// ── Starting jobs ───────────────────────────────────────────────────────────

/// A handle for a run that was never registered (it failed before it started).
struct NoJob;

impl JobHandle for NoJob {
    fn cancel(&self) {}
}

/// Finish a run that cannot start: `Finished(Err(Failed(message)))` on the next
/// turn of the event loop.
fn refuse(mut events: Events, message: String) -> Box<dyn JobHandle> {
    deliver_later(move || events(RunEvent::Finished(Err(JobError::Failed(message)))));
    Box::new(NoJob)
}

/// The handle of a registered run.
struct WebJob {
    shared: Weak<RefCell<Inner>>,
    job_id: u64,
}

impl JobHandle for WebJob {
    /// Terminate the worker (it cannot read a message while it computes) and end
    /// the run as cancelled; the next job starts a fresh worker.
    fn cancel(&self) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        let (job, orphans) = {
            let mut inner = shared.borrow_mut();
            let Some(job) = inner.jobs.remove(&self.job_id) else {
                return;
            };
            (job, inner.stop_worker())
        };
        orphans.fail_later("The browser worker was restarted.".to_owned());
        deliver_later(move || {
            job.emit(RunEvent::Finished(Err(JobError::Cancelled(String::new()))));
        });
    }
}

/// The files of a run as named byte buffers, and the name-to-path map.
fn collect_frames(
    paths: &[PathBuf],
) -> Result<(Vec<NamedBytes>, HashMap<String, PathBuf>), String> {
    let mut frames = Vec::with_capacity(paths.len());
    let mut names = HashMap::with_capacity(paths.len());
    for path in paths {
        let name = vfs::file_name(path);
        let bytes = vfs::get(path).ok_or_else(|| format!("{name} is no longer loaded."))?;
        names.insert(name.clone(), path.clone());
        frames.push(NamedBytes {
            name,
            bytes: bytes.to_vec(),
        });
    }
    Ok((frames, names))
}

/// A size in GiB, for the messages.
#[expect(clippy::cast_precision_loss, reason = "a size shown with one decimal")]
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Whether a run of `paths` fits the worker's memory; `None` when the first file's
/// size cannot be read (the worker then reports the problem itself).
fn budget_for(paths: &[PathBuf], settings: &StackingSettings, passes: &[Pass]) -> Option<Budget> {
    let first = vfs::get(paths.first()?)?;
    let dims = dims_from_bytes(&first)?;
    let encoded = paths
        .iter()
        .filter_map(|path| vfs::get(path))
        .map(|bytes| u64::try_from(bytes.len()).unwrap_or(u64::MAX))
        .fold(0_u64, u64::saturating_add);
    // `stack_every_nth` thins the frames inside the worker.
    let nth = usize::try_from(settings.stack_every_nth.max(1)).unwrap_or(1);
    let frames = paths.len().div_ceil(nth);
    let wire: Vec<WirePass> = passes.iter().copied().filter_map(wire_pass).collect();
    Some(plan_budget(frames, dims, encoded, settings, &wire))
}

/// The answer of the memory question: a dropped one (Cancel, Escape, a replaced
/// dialog) tells the caller "no" on the next turn of the event loop.
struct Answer(Option<Preflighted>);

impl Answer {
    /// Take the continuation out: the caller answers it itself.
    fn into_inner(mut self) -> Option<Preflighted> {
        self.0.take()
    }
}

/// `true` when a run of `frames` files with `settings` aligns them.
fn needs_alignment(settings: &StackingSettings, frames: usize) -> bool {
    settings.alignment_mode != AlignmentModeSetting::None && frames > 1
}

/// `prior` and `note` as one status note.
fn join_notes(prior: Option<String>, note: String) -> String {
    match prior {
        Some(prior) => format!("{prior}; {note}"),
        None => note,
    }
}

/// The first question of a stack: GPU for the alignment. Goes on to
/// [`ask_memory`] when the run can start (or is changed to not align).
fn preflight_stack(shared: &Shared, ctx: &Ctx, request: StackRequest, then: Preflighted) {
    // Frames of the last Align are not aligned again.
    if !needs_alignment(&request.settings, request.paths.len()) || request.reuse_aligned {
        ask_memory(shared, ctx, request, None, then);
        return;
    }
    let has_gpu = shared.borrow().has_webgpu();
    match has_gpu {
        Some(true) => ask_memory(shared, ctx, request, None, then),
        Some(false) => ask_no_gpu(shared, ctx, request, then),
        None => wait_for_gpu_answer(shared, ctx, request, then),
    }
}

/// The worker has not said whether it has WebGPU: start it and decide when it
/// does (never block).
fn wait_for_gpu_answer(shared: &Shared, ctx: &Ctx, request: StackRequest, then: Preflighted) {
    if let Err(error) = ensure_worker(shared) {
        feedback::notify(ctx, MessageKind::Error, &error);
        deliver_later(move || then(None));
        return;
    }
    let (waiting_shared, waiting_ctx) = (Rc::clone(shared), ctx.clone());
    let answer = Answer(Some(then));
    shared
        .borrow_mut()
        .waiters
        .push(Box::new(move |ready| match ready {
            Ok(()) => {
                if let Some(then) = answer.into_inner() {
                    preflight_stack(&waiting_shared, &waiting_ctx, request, then);
                }
            }
            Err(message) => {
                feedback::notify(&waiting_ctx, MessageKind::Error, &message);
                drop(answer);
            }
        }));
}

/// A browser without WebGPU: offer to stack without alignment.
fn ask_no_gpu(shared: &Shared, ctx: &Ctx, mut request: StackRequest, then: Preflighted) {
    let prompt = Prompt {
        action: "browser-no-webgpu",
        heading: "No GPU for alignment",
        message: NO_WEBGPU_QUESTION,
        primary_label: "Stack without alignment",
    };
    let shared = Rc::clone(shared);
    let answer = Answer(Some(then));
    confirm::confirm(ctx, &prompt, move |_app, ctx| {
        request.settings.alignment_mode = AlignmentModeSetting::None;
        if let Some(then) = answer.into_inner() {
            ask_memory(
                &shared,
                ctx,
                request,
                Some(NOT_ALIGNED_NOTE.to_owned()),
                then,
            );
        }
    });
}

/// The memory question of a stack. `prior` is a note an earlier question left.
fn ask_memory(
    shared: &Shared,
    ctx: &Ctx,
    mut request: StackRequest,
    prior: Option<String>,
    then: Preflighted,
) {
    let Some(budget) = budget_for(&request.paths, &request.settings, &request.passes) else {
        then(Some((request, prior)));
        return;
    };
    if budget.fits {
        then(Some((request, prior)));
        return;
    }
    let Some(percent) = budget.suggested_pre_resize else {
        feedback::notify(
            ctx,
            MessageKind::Error,
            &format!(
                "This stack needs about {:.1} GB of memory and a browser tab can use about \
                 2 GB; it does not fit even at the smallest size. Use the desktop app for it.",
                gib(budget.estimate)
            ),
        );
        then(None);
        return;
    };
    let message = format!(
        "This stack needs about {:.1} GB of memory at full size; a browser tab can use about \
         2 GB. Stack it at {percent} % size instead? The desktop app stacks it at full size.",
        gib(budget.estimate)
    );
    let primary = format!("Stack at {percent} %");
    let prompt = Prompt {
        action: "browser-memory",
        heading: "Not enough browser memory",
        message: &message,
        primary_label: &primary,
    };
    let note = join_notes(
        prior,
        format!("stacked at {percent} % (browser memory limit)"),
    );
    let shared = Rc::clone(shared);
    let answer = Answer(Some(then));
    confirm::confirm(ctx, &prompt, move |_app, _ctx| {
        request.settings.preprocessing.pre_resize_percent = percent;
        shared.borrow_mut().reduction = Some(percent);
        if let Some(then) = answer.into_inner() {
            then(Some((request, Some(note))));
        }
    });
}

impl Drop for Answer {
    fn drop(&mut self) {
        if let Some(then) = self.0.take() {
            deliver_later(move || then(None));
        }
    }
}

impl JobRunner for WebRunner {
    fn run_stack(
        &self,
        request: StackRequest,
        events: Events,
        ask_relief: AskRelief,
    ) -> Box<dyn JobHandle> {
        if request.ai.is_some() || request.passes.contains(&Pass::Ai) {
            return refuse(events, "AI fusion is only in the desktop app.".to_owned());
        }
        if request.settings.alignment_mode == AlignmentModeSetting::Neural {
            return refuse(events, NEURAL_MESSAGE.to_owned());
        }
        let (frames, names) = match collect_frames(&request.paths) {
            Ok(collected) => collected,
            Err(message) => return refuse(events, message),
        };
        let settings_toml = match toml::to_string(&request.settings) {
            Ok(text) => text,
            Err(error) => {
                return refuse(events, format!("Could not hand the settings over: {error}"));
            }
        };
        let passes: Vec<WirePass> = request
            .passes
            .iter()
            .copied()
            .filter_map(wire_pass)
            .collect();
        let (job_id, pre_resize_override) = {
            let mut inner = self.shared.borrow_mut();
            (inner.take_id(), inner.reduction.take())
        };
        let exif = first_frame_exif(&request.paths, &request.settings);
        let job = Rc::new(Job {
            events: RefCell::new(events),
            ask_relief: RefCell::new(Some(ask_relief)),
            names,
            passes: request.passes,
            auto_detect: request.settings.relief_auto_detect,
            contrast_pct: request.settings.relief_contrast_pct,
            results: Cell::new(0),
            note: RefCell::new(None),
            align_note: None,
            exif,
        });
        self.shared.borrow_mut().jobs.insert(job_id, job);
        send(
            &self.shared,
            ToWorker::Run {
                job_id,
                settings_toml,
                passes,
                frames,
                reuse_aligned: request.reuse_aligned,
                pre_resize_override,
            },
        );
        Box::new(WebJob {
            shared: Rc::downgrade(&self.shared),
            job_id,
        })
    }

    fn run_align(&self, request: AlignRequest, events: Events) -> Box<dyn JobHandle> {
        if request.settings.alignment_mode == AlignmentModeSetting::Neural {
            return refuse(events, NEURAL_MESSAGE.to_owned());
        }
        let (frames, names) = match collect_frames(&request.paths) {
            Ok(collected) => collected,
            Err(message) => return refuse(events, message),
        };
        let settings_toml = match toml::to_string(&request.settings) {
            Ok(text) => text,
            Err(error) => {
                return refuse(events, format!("Could not hand the settings over: {error}"));
            }
        };
        // Alignment is GPU-only here. When the worker has not said yet, the job
        // goes out anyway and the worker, which knows, refuses it.
        if needs_alignment(&request.settings, request.paths.len())
            && self.shared.borrow().has_webgpu() == Some(false)
        {
            return refuse(events, NO_WEBGPU_SENTENCE.to_owned());
        }
        // Align has no question to ask: a stack that does not fit is aligned reduced.
        let mut pre_resize_override = None;
        let mut align_note = None;
        if let Some(budget) = budget_for(&request.paths, &request.settings, &[])
            && !budget.fits
        {
            match budget.suggested_pre_resize {
                Some(percent) => {
                    pre_resize_override = Some(percent);
                    align_note = Some(format!(
                        "Aligned at {percent} % size (browser memory limit)."
                    ));
                }
                None => {
                    return refuse(
                        events,
                        format!(
                            "These images need about {:.1} GB of memory to align; a browser tab \
                             can use about 2 GB. Use the desktop app for them.",
                            gib(budget.estimate)
                        ),
                    );
                }
            }
        }
        let job_id = self.shared.borrow_mut().take_id();
        let job = Rc::new(Job {
            events: RefCell::new(events),
            ask_relief: RefCell::new(None),
            names,
            passes: Vec::new(),
            auto_detect: false,
            contrast_pct: 0.0,
            results: Cell::new(0),
            note: RefCell::new(None),
            align_note,
            exif: None,
        });
        self.shared.borrow_mut().jobs.insert(job_id, job);
        send(
            &self.shared,
            ToWorker::Align {
                job_id,
                settings_toml,
                frames,
                pre_resize_override,
            },
        );
        Box::new(WebJob {
            shared: Rc::downgrade(&self.shared),
            job_id,
        })
    }

    fn fetch_source(&self, path: &Path, settings: &StackingSettings, then: SourceDone) {
        let waiter = SourceWaiter {
            path: path.to_path_buf(),
            settings: settings.clone(),
            then,
        };
        let request_id = {
            let mut inner = self.shared.borrow_mut();
            if inner.worker.is_none() {
                // No worker, no aligned set: the frame is decoded here.
                drop(inner);
                deliver_later(move || waiter.fallback());
                return;
            }
            let id = inner.take_id();
            inner.sources.insert(id, waiter);
            id
        };
        send(
            &self.shared,
            ToWorker::FetchSource {
                request_id,
                name: vfs::file_name(path),
            },
        );
    }

    fn fetch_preview(&self, path: &Path, then: PreviewDone) {
        let waiter = PreviewWaiter {
            path: path.to_path_buf(),
            then,
        };
        let request_id = {
            let mut inner = self.shared.borrow_mut();
            if inner.worker.is_none() {
                // A worker that was never started holds no files: decode here.
                drop(inner);
                deliver_later(move || waiter.fallback());
                return;
            }
            let id = inner.take_id();
            inner.previews.insert(id, waiter);
            id
        };
        send(
            &self.shared,
            ToWorker::FetchPreview {
                request_id,
                name: vfs::file_name(path),
                max_edge: FULL_SIZE,
            },
        );
    }

    fn forget_aligned(&self) {
        if self.shared.borrow().worker.is_some() {
            send(&self.shared, ToWorker::ForgetAligned);
        }
    }

    /// Two questions, in this order: is there a GPU for the alignment, then does
    /// the stack fit the browser's memory.
    fn preflight(&self, ctx: &Ctx, request: StackRequest, then: Preflighted) {
        self.shared.borrow_mut().reduction = None;
        preflight_stack(&self.shared, ctx, request, then);
    }
}

/// Decode the file at `path` on the page, with the settings' preprocessing.
fn decode_source(path: &Path, settings: &StackingSettings) -> Result<PlanarImage<f32>, String> {
    let name = vfs::file_name(path);
    let bytes = vfs::get(path).ok_or_else(|| format!("{name} is no longer loaded."))?;
    decode_frame_bytes(&bytes, &name, &settings.preprocessing).map_err(|error| error.to_string())
}
