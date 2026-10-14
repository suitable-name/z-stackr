//! What the Worker does with each message: the stack sequence of the desktop app
//! (`stacker_pipeline::session`), single-threaded, reporting through posted
//! messages. Alignment runs on WebGPU (awaited), everything else on the CPU.
//!
//! The Worker holds three things between messages (see [`State`]): the encoded
//! files of the last Run/Align (for the source-file viewer, and for a Run that
//! reuses the alignment), the aligned frames, and a `Relief` run parked at its
//! contrast preview. A message handler never holds the state borrowed while it
//! computes or awaits: it takes what it needs out and puts the result back.
//!
//! A Run or an Align is a task on the event loop (`spawn_local`) that owns a
//! [`Busy`] guard; while it is alive a second Run or Align is refused.

use std::{
    cell::{Cell, RefCell},
    fmt::Write as _,
    rc::Rc,
};

use stacker_algo::strata::NO_PREVIEW;
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, ComputeDevice, OptimizerSetting, StackingSettings},
};
use stacker_pipeline::{
    NeverCancel, Progress, Stage as PipelineStage,
    compute::Compute,
    config::{Leniency, parse_settings},
    crop::Rect as CropRect,
    fraction_of,
    load::decode_frame_bytes,
    session::{
        self, FrameSource, ReliefFocus, auto_contrast_pct, crop_result, fuse_apex, fuse_strata,
        load_frames, relief_finish, relief_focus_max, threshold_from_settings,
    },
};
use stacker_web_core::{
    FromWorker, NamedBytes, PROTOCOL_VERSION, PassKind, PlanarF32, Rect, Stage, ToWorker,
    downsample_plane, planar_to_rgba, threshold_table,
};
use wasm_bindgen_futures::spawn_local;

use crate::{
    gpu::{self, WebGpuBackend},
    host::{now_ms, post},
};

/// The longest edge of the live alignment preview, in pixels.
const LIVE_PREVIEW_EDGE: u32 = 960;
/// The shortest time between two live alignment previews, in milliseconds.
const LIVE_PREVIEW_INTERVAL_MS: f64 = 250.0;
/// The longest edge of the `Relief` preview panel's pictures, in pixels.
const PANEL_EDGE: u32 = 640;
/// The lowest Pre-resize percentage the settings allow.
const MIN_PRE_RESIZE: u32 = 10;

// ── State ───────────────────────────────────────────────────────────────────

/// The aligned frames of the last Align or Run.
struct Cache {
    /// Frame names, parallel to `frames`.
    names: Vec<String>,
    /// The aligned frames (after a Run: the kept frames in fusion order).
    frames: Vec<PlanarImage<f32>>,
    /// The common valid area of `frames`.
    crop: Option<CropRect>,
    /// The Pre-resize override the frames were made with.
    pre_resize: Option<u32>,
    /// The frames earlier runs left out through the auto-cull.
    culled: Vec<String>,
    /// Frames that decoded, before any cull.
    total: usize,
}

/// A run that stopped at the `Relief` contrast preview.
struct Parked {
    run: RunState,
    focus: ReliefFocus,
}

#[derive(Default)]
struct State {
    /// The encoded files of the last Run or Align, by name.
    bytes: Rc<Vec<NamedBytes>>,
    cache: Option<Cache>,
    parked: Option<Parked>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(|state| f(&mut state.borrow_mut()))
}

thread_local! {
    /// `true` while a Run or an Align task is in flight.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

/// Marks the Worker busy with a Run or an Align until it is dropped.
struct Busy;

impl Busy {
    /// `None` when another Run or Align is in flight.
    fn begin() -> Option<Self> {
        BUSY.with(|busy| (!busy.replace(true)).then_some(Self))
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        BUSY.with(|busy| busy.set(false));
    }
}

// ── Posting helpers ─────────────────────────────────────────────────────────

fn progress(job_id: u64, stage: Stage, fraction: f32, label: &str) {
    post(&FromWorker::Progress {
        job_id,
        stage,
        fraction,
        label: label.to_owned(),
    });
}

fn warn(job_id: u64, text: String) {
    post(&FromWorker::Warning { job_id, text });
}

/// Forwards the pipeline's stage reports as [`FromWorker::Progress`].
struct Reporter {
    job_id: u64,
}

impl Progress for Reporter {
    fn stage(&self, stage: PipelineStage, fraction: f32, label: &str) {
        progress(self.job_id, stage.into(), fraction, label);
    }
}

// ── Dispatch ────────────────────────────────────────────────────────────────

/// Handle one message from the page.
pub fn handle(message: ToWorker) {
    match message {
        ToWorker::Hello { version } => {
            if version != PROTOCOL_VERSION {
                post(&FromWorker::Failed {
                    job_id: 0,
                    message: format!(
                        "The page speaks protocol {version} but the worker speaks \
                         {PROTOCOL_VERSION}: reload the page to load matching files."
                    ),
                });
            }
        }
        ToWorker::Run {
            job_id,
            settings_toml,
            passes,
            frames,
            reuse_aligned,
            pre_resize_override,
        } => {
            if let Some(busy) = begin_job(job_id) {
                spawn_local(async move {
                    run_job(
                        job_id,
                        &settings_toml,
                        passes,
                        frames,
                        reuse_aligned,
                        pre_resize_override,
                    )
                    .await;
                    drop(busy);
                });
            }
        }
        ToWorker::Align {
            job_id,
            settings_toml,
            frames,
            pre_resize_override,
        } => {
            if let Some(busy) = begin_job(job_id) {
                spawn_local(async move {
                    align_job(job_id, &settings_toml, frames, pre_resize_override).await;
                    drop(busy);
                });
            }
        }
        ToWorker::ReliefThreshold { job_id, threshold } => resume_relief(job_id, threshold),
        ToWorker::FetchSource { request_id, name } => {
            spawn_local(async move { fetch_source(request_id, &name) });
        }
        ToWorker::ForgetAligned => with_state(|state| *state = State::default()),
        ToWorker::FetchPreview {
            request_id,
            name,
            max_edge,
        } => spawn_local(async move { fetch_preview(request_id, &name, max_edge) }),
    }
}

/// Claim the Worker for a Run or an Align; a second one is answered with
/// `Failed` (the page does not start one while another runs, so this is the
/// safety net).
fn begin_job(job_id: u64) -> Option<Busy> {
    let busy = Busy::begin();
    if busy.is_none() {
        post(&FromWorker::Failed {
            job_id,
            message: "busy: the worker is still stacking or aligning; wait for it to finish."
                .to_owned(),
        });
    }
    busy
}

// ── Settings ────────────────────────────────────────────────────────────────

/// The settings a run works with: the page's settings, the memory budget's
/// Pre-resize reduction, and the browser's limits (fusion on the CPU, alignment
/// on WebGPU through [`WebGpuBackend`], no AKAZE, no neural alignment, the
/// Nelder-Mead optimizer, everything in memory).
///
/// # Errors
/// The parse error, or the refusal of the `Neural` alignment mode (the browser
/// has no neural runtime; the page does not offer it, this is the safety net).
fn effective_settings(
    settings_toml: &str,
    pre_resize_override: Option<u32>,
) -> Result<StackingSettings, String> {
    let mut settings = parse_settings(settings_toml, Leniency::Lenient)?.settings;
    if settings.alignment_mode == AlignmentModeSetting::Neural {
        return Err(
            "Neural alignment is only in the desktop app: in the browser, alignment runs on the \
             GPU only. Choose Translation, Registration or Affine."
                .to_owned(),
        );
    }
    if let Some(percent) = pre_resize_override {
        settings.preprocessing.pre_resize_percent = percent.clamp(MIN_PRE_RESIZE, 100);
    }
    settings.compute_device = ComputeDevice::Cpu;
    settings.akaze_seeding = false;
    // The asynchronous GPU path refines with the resumable Nelder-Mead search
    // (`AlignSearch::for_motion`); Lucas-Kanade has no async form.
    settings.optimizer = OptimizerSetting::NelderMead;
    // The worker holds every frame in memory (the page budgets for it); there is
    // no scratch store to tile through.
    settings.tile_size = 0;
    Ok(settings)
}

// ── Loading and aligning ────────────────────────────────────────────────────

/// The files of a run, decoded one at a time on demand. The list is shared with
/// the state, which keeps the encoded files for the source-file viewer.
struct SharedBytes(Rc<Vec<NamedBytes>>);

impl FrameSource for SharedBytes {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn name(&self, index: usize) -> &str {
        &self.0[index].name
    }

    fn decode(
        &mut self,
        index: usize,
        pre: &stacker_core::settings::PreprocessingSettings,
    ) -> Result<PlanarImage<f32>, stacker_pipeline::PipelineError> {
        let file = &self.0[index];
        decode_frame_bytes(&file.bytes, &file.name, pre)
    }
}

/// Decoded and aligned frames.
struct Aligned {
    names: Vec<String>,
    frames: Vec<PlanarImage<f32>>,
    crop: Option<CropRect>,
    /// How many files decoded.
    total: usize,
    /// Which frames were left out because they could not be read.
    skipped_note: Option<String>,
    /// Which frames did not align properly.
    align_warning: Option<String>,
}

/// Decode every file and align the frames to the first (the first half of both a
/// Run and an Align). Posts the load and align progress and the live previews.
async fn load_and_align(
    job_id: u64,
    settings: &StackingSettings,
    bytes: &Rc<Vec<NamedBytes>>,
) -> Result<Aligned, String> {
    // The GPU is needed before anything is decoded: a browser without WebGPU
    // fails the run now, not after the decoding minutes.
    let backend = if settings.alignment_mode != AlignmentModeSetting::None && bytes.len() > 1 {
        Some(gpu::backend().await?)
    } else {
        None
    };
    progress(job_id, Stage::Load, 0.0, "Loading frames...");
    let mut source = SharedBytes(Rc::clone(bytes));
    let mut loaded = load_frames(
        &mut source,
        &settings.preprocessing,
        &NeverCancel,
        &Reporter { job_id },
    )
    .map_err(|error| error.to_string())?;
    let names = std::mem::take(&mut loaded.names);
    let mut frames = std::mem::take(&mut loaded.frames);
    if let Some(text) = loaded.skipped_warning() {
        warn(job_id, text);
    }
    let skipped_note = loaded.skipped_note();
    progress(job_id, Stage::Load, 1.0, "");

    let (crop, align_warning) = align_stage(job_id, settings, &names, &mut frames, backend).await?;
    let total = names.len();
    Ok(Aligned {
        names,
        frames,
        crop,
        total,
        skipped_note,
        align_warning,
    })
}

/// Align `frames` to the first one on `backend` (the GPU); returns the crop and
/// the warning text. `backend` is `None` exactly when nothing needs aligning.
async fn align_stage(
    job_id: u64,
    settings: &StackingSettings,
    names: &[String],
    frames: &mut [PlanarImage<f32>],
    backend: Option<WebGpuBackend>,
) -> Result<(Option<CropRect>, Option<String>), String> {
    let total = frames.len();
    let Some(mut backend) = backend.filter(|_| total > 1) else {
        progress(job_id, Stage::Align, 1.0, "");
        return Ok((None, None));
    };
    progress(
        job_id,
        Stage::Align,
        0.0,
        &format!(
            "Aligning frames ({}) on the GPU…",
            settings.alignment_mode.as_combo_str()
        ),
    );
    let mut last_preview = f64::NEG_INFINITY;
    let aligned = session::align_frames_async(
        frames,
        settings,
        &mut backend,
        &NeverCancel,
        &mut |index: usize, frame: &PlanarImage<f32>, _mask: &[bool]| {
            progress(
                job_id,
                Stage::Align,
                fraction_of(index, total - 1),
                &format!("Aligning… frame {index}/{}", total - 1),
            );
            let now = now_ms();
            if now - last_preview >= LIVE_PREVIEW_INTERVAL_MS {
                last_preview = now;
                let picture = planar_to_rgba(frame, LIVE_PREVIEW_EDGE);
                post(&FromWorker::AlignedFrame {
                    job_id,
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    total: u32::try_from(total).unwrap_or(u32::MAX),
                    preview_rgba: picture.rgba,
                    width: picture.width,
                    height: picture.height,
                });
            }
        },
    )
    .await;
    // The device buffers go before the CPU fusion allocates its own.
    drop(backend);
    let outcome = match aligned {
        Ok(Some(outcome)) => outcome,
        Ok(None) => return Err("Cancelled while aligning".to_owned()),
        Err(error) => {
            // The device may be lost: the next alignment opens a fresh one.
            gpu::forget();
            let mut message = error.to_string();
            if let Some(detail) = gpu::take_last_error() {
                let _ = write!(message, " GPU said: {detail}");
                warn(job_id, format!("GPU said: {detail}"));
            }
            return Err(message);
        }
    };
    let warning = outcome.warning(names);
    if let Some(text) = &warning {
        warn(job_id, text.clone());
    }
    progress(job_id, Stage::Align, 1.0, "");
    Ok((outcome.crop, warning))
}

/// The Align button: decode, align, keep the aligned frames.
async fn align_job(
    job_id: u64,
    settings_toml: &str,
    frames: Vec<NamedBytes>,
    pre_resize_override: Option<u32>,
) {
    if let Err(message) = align_inner(job_id, settings_toml, frames, pre_resize_override).await {
        post(&FromWorker::Failed { job_id, message });
    }
}

async fn align_inner(
    job_id: u64,
    settings_toml: &str,
    frames: Vec<NamedBytes>,
    pre_resize_override: Option<u32>,
) -> Result<(), String> {
    let settings = effective_settings(settings_toml, pre_resize_override)?;
    // The old frames go first: they would only add to the peak.
    let bytes = with_state(|state| {
        state.cache = None;
        state.parked = None;
        if !frames.is_empty() {
            state.bytes = Rc::new(frames);
        }
        Rc::clone(&state.bytes)
    });
    let aligned = load_and_align(job_id, &settings, &bytes).await?;
    let message = FromWorker::AlignDone {
        job_id,
        crop: aligned.crop.map(Rect::from_crop),
        warning: aligned.align_warning,
        names: aligned.names.clone(),
    };
    with_state(|state| {
        state.cache = Some(Cache {
            names: aligned.names,
            frames: aligned.frames,
            crop: aligned.crop,
            pre_resize: pre_resize_override,
            culled: Vec::new(),
            total: aligned.total,
        });
    });
    post(&message);
    Ok(())
}

// ── Run ─────────────────────────────────────────────────────────────────────

/// A run between its stages: everything the fusion passes need.
struct RunState {
    job_id: u64,
    settings: StackingSettings,
    passes: Vec<PassKind>,
    /// Position of the next pass to run.
    next_pass: usize,
    /// The names of the frames, parallel to `frames`.
    names: Vec<String>,
    /// The kept frames in fusion order.
    frames: Vec<PlanarImage<f32>>,
    crop: Option<CropRect>,
    pre_resize: Option<u32>,
    /// The culled frames, published with the first result.
    culled_to_publish: Option<Vec<String>>,
    /// Every frame culled so far (this run and the runs before), for the cache.
    culled_record: Vec<String>,
    total: usize,
    note: Option<String>,
    /// The `Relief` focus map and threshold the user has chosen, waiting for
    /// the pass to take them.
    pending_relief: Option<(ReliefFocus, f32)>,
}

enum Outcome {
    /// Every pass is published.
    Done,
    /// Waiting for the page's answer to the `Relief` preview.
    Parked,
}

async fn run_job(
    job_id: u64,
    settings_toml: &str,
    passes: Vec<PassKind>,
    frames: Vec<NamedBytes>,
    reuse_aligned: bool,
    pre_resize_override: Option<u32>,
) {
    match start_run(
        job_id,
        settings_toml,
        passes,
        frames,
        reuse_aligned,
        pre_resize_override,
    )
    .await
    {
        Ok(Outcome::Done) => post(&FromWorker::Done { job_id }),
        Ok(Outcome::Parked) => {}
        Err(message) => post(&FromWorker::Failed { job_id, message }),
    }
}

/// Load (or reuse), align, cull and order; then fuse.
async fn start_run(
    job_id: u64,
    settings_toml: &str,
    passes: Vec<PassKind>,
    frames: Vec<NamedBytes>,
    reuse_aligned: bool,
    pre_resize_override: Option<u32>,
) -> Result<Outcome, String> {
    let settings = effective_settings(settings_toml, pre_resize_override)?;
    if passes.is_empty() {
        return Err("No fusion pass was requested.".to_owned());
    }
    let (bytes, cache) = with_state(|state| {
        state.parked = None;
        if !frames.is_empty() {
            state.bytes = Rc::new(frames);
        }
        (Rc::clone(&state.bytes), state.cache.take())
    });

    let aligns = settings.alignment_mode != AlignmentModeSetting::None;
    let mut cache_note = None;
    // An unusable cache is dropped here, before the frames are decoded.
    let reusable = reuse_aligned
        .then_some(cache)
        .flatten()
        .filter(|cache| cache.pre_resize == pre_resize_override);
    let (names, frames, crop, total, earlier_culled, skipped_note) = match (reuse_aligned, reusable)
    {
        (true, Some(cache)) => {
            progress(job_id, Stage::Load, 1.0, "Reusing the aligned frames…");
            progress(job_id, Stage::Align, 1.0, "");
            if aligns {
                cache_note = Some("reused the frames of the last Align".to_owned());
            }
            (
                cache.names,
                cache.frames,
                cache.crop,
                cache.total,
                cache.culled,
                None,
            )
        }
        (reuse, _) => {
            if reuse && bytes.is_empty() {
                return Err(
                    "The aligned frames are no longer available: add the images again.".to_owned(),
                );
            }
            if reuse && aligns {
                cache_note = Some("aligned afresh: the last Align is out of date".to_owned());
            }
            // The alignment warning was posted by the align stage.
            let aligned = load_and_align(job_id, &settings, &bytes).await?;
            (
                aligned.names,
                aligned.frames,
                aligned.crop,
                aligned.total,
                Vec::new(),
                aligned.skipped_note,
            )
        }
    };

    // Cull, then move the kept frames into fusion order.
    if settings.auto_cull {
        progress(job_id, Stage::Cull, 0.0, "Optimising & culling…");
    }
    let cull = session::cull(&frames, &settings).map_err(|error| error.to_string())?;
    if let Some(text) = cull.warning() {
        warn(job_id, text);
    }
    progress(job_id, Stage::Cull, 1.0, "");
    let mut culled_record = earlier_culled;
    culled_record.extend(cull.dropped.iter().filter_map(|&i| names.get(i).cloned()));
    let kept_names: Vec<String> = cull
        .order
        .iter()
        .filter_map(|&i| names.get(i).cloned())
        .collect();
    let ordered = session::reorder(frames, &cull.order);

    let note = [skipped_note, cache_note]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("; ");
    let state = RunState {
        job_id,
        settings,
        passes,
        next_pass: 0,
        names: kept_names,
        frames: ordered,
        crop,
        pre_resize: pre_resize_override,
        culled_to_publish: Some(culled_record.clone()),
        culled_record,
        total,
        note: (!note.is_empty()).then_some(note),
        pending_relief: None,
    };
    continue_run(state)
}

/// The answer to the `Relief` preview: fuse with the chosen threshold, or end.
fn resume_relief(job_id: u64, threshold: Option<f32>) {
    let parked = with_state(|state| {
        if state
            .parked
            .as_ref()
            .is_some_and(|parked| parked.run.job_id == job_id)
        {
            state.parked.take()
        } else {
            None
        }
    });
    let Some(Parked { mut run, focus }) = parked else {
        post(&FromWorker::Failed {
            job_id,
            message: "No Relief preview is waiting for a contrast threshold.".to_owned(),
        });
        return;
    };
    let Some(threshold) = threshold else {
        store_cache(run);
        post(&FromWorker::Cancelled { job_id });
        return;
    };
    run.pending_relief = Some((focus, threshold));
    match continue_run(run) {
        Ok(Outcome::Done) => post(&FromWorker::Done { job_id }),
        Ok(Outcome::Parked) => {}
        Err(message) => post(&FromWorker::Failed { job_id, message }),
    }
}

/// Run the passes from `run.next_pass` on. Returns when all are published, or
/// when a `Relief` pass has parked the run at its preview.
fn continue_run(mut run: RunState) -> Result<Outcome, String> {
    let job_id = run.job_id;
    while let Some(&pass) = run.passes.get(run.next_pass) {
        let fused = match pass {
            PassKind::Apex => fuse_apex_pass(&run),
            PassKind::Relief => match fuse_relief_pass(&mut run)? {
                Some(fused) => fused,
                None => return Ok(Outcome::Parked),
            },
            PassKind::Strata => fuse_strata_pass(&run),
        };
        let cropped = crop_result(fused, run.crop);
        progress(
            job_id,
            Stage::Save,
            0.0,
            &format!("Saving [{}]…", pass_label(pass)),
        );
        post(&FromWorker::Result {
            job_id,
            pass,
            planes: PlanarF32::from(cropped),
            culled: run.culled_to_publish.take().unwrap_or_default(),
            kept: run.frames.len(),
            total: run.total,
            note: run.note.clone(),
        });
        progress(job_id, Stage::Save, 1.0, "");
        run.next_pass += 1;
    }
    store_cache(run);
    Ok(Outcome::Done)
}

const fn pass_label(pass: PassKind) -> &'static str {
    match pass {
        PassKind::Apex => "Apex",
        PassKind::Relief => "Relief",
        PassKind::Strata => "Strata",
    }
}

/// The `Strata` pass on the CPU. It reports `Stage::Fuse` progress itself (two
/// ticks per frame) and posts no live preview, like the other passes.
fn fuse_strata_pass(run: &RunState) -> PlanarImage<f32> {
    progress(
        run.job_id,
        Stage::Fuse,
        0.0,
        &format!("Fusing [Strata] {} frames on the CPU…", run.frames.len()),
    );
    let fused = fuse_strata(
        &run.frames,
        &run.settings,
        &Reporter { job_id: run.job_id },
        NO_PREVIEW,
        |_| {},
    );
    progress(run.job_id, Stage::Fuse, 1.0, "");
    fused
}

fn fuse_apex_pass(run: &RunState) -> PlanarImage<f32> {
    let mut compute = Compute::cpu();
    progress(
        run.job_id,
        Stage::Fuse,
        0.0,
        &format!(
            "Fusing [Apex] {} frames on {}…",
            run.frames.len(),
            compute.describe()
        ),
    );
    let fused = fuse_apex(&mut compute, &run.frames, &run.settings);
    if let Some(notice) = compute.take_notice() {
        warn(run.job_id, notice);
    }
    progress(run.job_id, Stage::Fuse, 1.0, "");
    fused
}

/// The `Relief` pass. `Ok(None)` when it parked the run at the contrast preview.
fn fuse_relief_pass(run: &mut RunState) -> Result<Option<PlanarImage<f32>>, String> {
    let job_id = run.job_id;
    let mut compute = Compute::cpu();
    let (focus, threshold) = if let Some(chosen) = run.pending_relief.take() {
        chosen
    } else {
        progress(
            job_id,
            Stage::Fuse,
            0.0,
            &format!(
                "Fusing [Relief] {} frames on {}…",
                run.frames.len(),
                compute.describe()
            ),
        );
        let focus = relief_focus_max(&mut compute, &run.frames, &run.settings)
            .map_err(|error| error.to_string())?;
        if run.settings.relief_show_preview {
            post_relief_preview(run, &focus);
            // Park the run: the Worker goes back to its event loop and waits for
            // `ReliefThreshold`.
            let parked = Parked {
                run: std::mem::replace(run, empty_run()),
                focus,
            };
            with_state(|state| state.parked = Some(parked));
            return Ok(None);
        }
        let threshold = threshold_from_settings(focus.max(), &run.settings);
        (focus, threshold)
    };
    let fused = relief_finish(&mut compute, &run.frames, &run.settings, focus, threshold);
    if let Some(notice) = compute.take_notice() {
        warn(job_id, notice);
    }
    progress(job_id, Stage::Fuse, 1.0, "");
    Ok(Some(fused))
}

/// A run without frames, left behind in the caller when its state moved into
/// [`Parked`]. `continue_run` returns right after, so nothing reads it.
fn empty_run() -> RunState {
    RunState {
        job_id: 0,
        settings: StackingSettings::default(),
        passes: Vec::new(),
        next_pass: 0,
        names: Vec::new(),
        frames: Vec::new(),
        crop: None,
        pre_resize: None,
        culled_to_publish: None,
        culled_record: Vec::new(),
        total: 0,
        note: None,
        pending_relief: None,
    }
}

/// Send the data the page's contrast panel renders from.
fn post_relief_preview(run: &RunState, focus: &ReliefFocus) {
    let max = focus.max();
    let (width, height, focus_max) = downsample_plane(&max.luma, max.width, max.height, PANEL_EDGE);
    let base = run
        .frames
        .first()
        .map(|frame| planar_to_rgba(frame, PANEL_EDGE))
        .unwrap_or_default();
    post(&FromWorker::ReliefPreview {
        job_id: run.job_id,
        width,
        height,
        focus_max,
        base_rgba: base.rgba,
        auto_pct: auto_contrast_pct(max),
        thresholds: threshold_table(&max.luma),
    });
}

/// Keep the run's frames (kept, in fusion order) for retouching and a later Run.
fn store_cache(run: RunState) {
    with_state(|state| {
        state.cache = Some(Cache {
            names: run.names,
            frames: run.frames,
            crop: run.crop,
            pre_resize: run.pre_resize,
            culled: run.culled_record,
            total: run.total,
        });
    });
}

// ── Source frames and previews ──────────────────────────────────────────────

/// The aligned, cropped frame called `name`, for retouching.
fn fetch_source(request_id: u64, name: &str) {
    let found = with_state(|state| {
        let cache = state.cache.as_ref()?;
        let index = cache.names.iter().position(|n| n == name)?;
        Some(crop_result(cache.frames[index].clone(), cache.crop))
    });
    match found {
        Some(frame) => post(&FromWorker::Source {
            request_id,
            name: name.to_owned(),
            planes: PlanarF32::from(frame),
        }),
        None => post(&FromWorker::SourceFailed {
            request_id,
            message: format!("The aligned frame '{name}' is not available: stack or align first."),
        }),
    }
}

/// A thumbnail of the file called `name`.
fn fetch_preview(request_id: u64, name: &str, max_edge: u32) {
    let bytes = with_state(|state| Rc::clone(&state.bytes));
    let Some(file) = bytes.iter().find(|file| file.name == name) else {
        post(&FromWorker::PreviewFailed {
            request_id,
            message: format!("The worker does not hold the file '{name}'."),
        });
        return;
    };
    let pre = StackingSettings::default().preprocessing;
    match decode_frame_bytes(&file.bytes, &file.name, &pre) {
        Ok(frame) => {
            let picture = planar_to_rgba(&frame, max_edge);
            post(&FromWorker::Preview {
                request_id,
                name: name.to_owned(),
                width: picture.width,
                height: picture.height,
                rgba: picture.rgba,
            });
        }
        Err(error) => post(&FromWorker::PreviewFailed {
            request_id,
            message: error.to_string(),
        }),
    }
}
