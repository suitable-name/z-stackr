//! The Stack pass: build the request, run it, publish what comes back.
//!
//! The pass is a [`crate::runner`] run. This module is its UI side: it snapshots
//! the inputs into a [`StackRequest`], lets the runner look at it
//! ([`JobRunner::preflight`]), starts the run and turns every [`RunEvent`] into a
//! UI update on the UI thread. Its frame list is a position list: every index the
//! runner reports, the auto-cull order included, is a position in the files of
//! the request.
//!
//! # Stages and cancellation
//!
//! The job reports through a [`StagePlan`] (load, align, cull, then a fuse and a
//! save stage per pass; "Both" is two passes of one job). The runner checks the
//! cancel flag between frames while aligning and between stages everywhere else,
//! so a cancel ends the job before anything half-written is published: the result
//! image is written completely and a cancel noticed afterwards deletes it again.
//!
//! # Memory
//!
//! A run holds the decoded frames as planar `f32` only: 12 bytes per pixel and
//! frame (`luma`, `chroma_a`, `chroma_b`). No frame has an RGBA copy; an 8-bit
//! display buffer (4 bytes per pixel) exists for the one image being shown at a
//! time (the `Relief` preview builds its own panel-sized picture). The auto-cull
//! reorder moves frames instead of cloning them. A run that reuses the Align
//! cache copies the held frames on the runner's thread, not on the UI thread.

use std::path::PathBuf;

use slint::{Image, SharedString};
use stacker_algo::hybrid::retouch::RetouchSession;
use stacker_gui::{
    frames,
    image_utils::planar_to_rgba_buffer,
    jobs_core::{JobError, StagePlan},
    settings::{AlignmentModeSetting, StackingSettings},
    stale::{Fingerprint, fingerprint_paths, fingerprint_stack, summarize_settings},
    stall_guard::stall_guard,
    state::ResultMeta,
};
use web_time::Instant;

use super::{
    Ctx, explainer,
    feedback::{self, MessageKind},
    jobs::{self, Job, JobSpec},
    log_path, monitor, preview, relief_preview, retouch, stale,
};
pub use crate::runner::Pass;
use crate::{
    App,
    runner::{self, RunEvent, RunStep, StackRequest, StackResult},
    ui_helpers::update_result_list,
};

/// The algorithm list the combo shows (the AI entry is appended in `nn` builds).
pub const ALGORITHMS: [&str; 5] = [
    "Relief (Depth Map)",
    "Apex (Laplacian)",
    "Strata (Guided Fusion)",
    "Both (Relief & Apex)",
    "All Three (Strata, Relief & Apex)",
];

/// Fill the algorithm list, and in `nn` builds the AI model, device and
/// alignment-model pickers.
pub fn populate_algorithms(app: &App) {
    use slint::{ModelRc, VecModel};

    let algos: Vec<SharedString> = ALGORITHMS.iter().map(|a| SharedString::from(*a)).collect();
    #[cfg(feature = "nn")]
    let algos = {
        let mut algos = algos;
        // The two pickers are filtered by architecture: a fusion network is no
        // aligner and the other way round.
        let models = stacker_nn::discover_default_fusion_models();
        let model_names: Vec<SharedString> = models
            .iter()
            .map(|m| SharedString::from(m.name.as_str()))
            .collect();
        let align_names: Vec<SharedString> = stacker_nn::discover_default_alignment_models()
            .iter()
            .map(|m| SharedString::from(m.name.as_str()))
            .collect();
        let devices = stacker_nn::available_devices();
        let device_names: Vec<SharedString> = devices
            .iter()
            .map(|d| SharedString::from(d.label()))
            .collect();

        algos.push("AI Model (Neural)".into());
        app.set_ai_models_available(!model_names.is_empty());
        app.set_ai_model_names(ModelRc::new(VecModel::from(model_names)));
        app.set_ai_device_names(ModelRc::new(VecModel::from(device_names)));
        app.set_ai_show_device(devices.len() > 1);
        app.set_align_models_available(!align_names.is_empty());
        app.set_align_model_names(ModelRc::new(VecModel::from(align_names)));
        algos
    };
    app.set_algo_options(ModelRc::new(VecModel::from(algos)));
}

/// Read the session's alignment-model pick from the UI and remember it in the state.
///
/// The pick belongs to the alignment fingerprint. It is returned as the request
/// carries it: the model name and the device label (empty: the first device).
/// `None` without a pick.
#[expect(
    clippy::must_use_candidate,
    reason = "also called for its effect on the state alone (the stale view)"
)]
pub fn sync_align_model(app: &App, ctx: &Ctx) -> Option<(String, String)> {
    use slint::ComponentHandle;

    let name = app.global::<crate::AppModel>().get_align_model_value();
    ctx.state.borrow_mut().align_model = name.to_string();
    (!name.is_empty()).then(|| (name.to_string(), String::new()))
}

// ── Wiring ──────────────────────────────────────────────────────────────────

/// Register the Stack button's callback.
pub fn setup(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_request_stack(move |algo, ai_model, ai_device| {
        stall_guard("stack::request-stack", || {
            start(
                &ctx,
                algo.as_str(),
                ai_model.to_string(),
                ai_device.to_string(),
            );
        });
    });
}

/// The passes the algorithm combo-box text asks for.
///
/// "All Three" and "Both" are tested before the single engines, because their
/// labels name them. The labels of earlier versions are read too: `DMap…` is
/// Relief, `PMax…` is Apex, `Both (DMap & PMax)` is Both.
fn passes_for(algo: &str) -> Vec<Pass> {
    if algo.contains("AI") || algo.contains("Neural") {
        vec![Pass::Ai]
    } else if algo.contains("All Three") {
        vec![Pass::Strata, Pass::Relief, Pass::Apex]
    } else if algo.contains("Both") {
        vec![Pass::Relief, Pass::Apex]
    } else if algo.contains("Strata") {
        vec![Pass::Strata]
    } else if algo.contains("Relief") || algo.contains("DMap") {
        vec![Pass::Relief]
    } else {
        vec![Pass::Apex]
    }
}

// ── Stages ──────────────────────────────────────────────────────────────────

const STAGE_LOAD: usize = 0;
const STAGE_ALIGN: usize = 1;
const STAGE_CULL: usize = 2;

/// The stage index of the fusion of pass number `pass` (0-based).
const fn stage_fuse(pass: usize) -> usize {
    3 + 2 * pass
}

/// The stage index of the save-and-publish of pass number `pass`.
const fn stage_save(pass: usize) -> usize {
    4 + 2 * pass
}

/// The stage of a run's [`StagePlan`] that `step` belongs to (the Align job's
/// plan shares the first two).
pub(super) const fn stage_of(step: RunStep) -> usize {
    match step {
        RunStep::Load => STAGE_LOAD,
        RunStep::Align => STAGE_ALIGN,
        RunStep::Cull => STAGE_CULL,
        RunStep::Fuse(pass) => stage_fuse(pass),
        RunStep::Save(pass) => stage_save(pass),
    }
}

/// The weighted stages of a run: load, align (weightless when alignment is off),
/// cull (weightless without auto-cull), then fuse and save for every pass.
fn plan_for(settings: &StackingSettings, passes: &[Pass]) -> StagePlan {
    let align = if settings.alignment_mode == AlignmentModeSetting::None {
        0.0
    } else {
        0.30
    };
    let cull = if settings.auto_cull || settings.sort_by_sharpness {
        0.05
    } else {
        0.0
    };
    let count = f32::from(u8::try_from(passes.len()).unwrap_or(u8::MAX)).max(1.0);
    let mut stages = vec![("Loading", 0.15), ("Aligning", align), ("Culling", cull)];
    for _ in passes {
        stages.push(("Fusing", 0.40 / count));
        stages.push(("Saving", 0.10 / count));
    }
    StagePlan::new(&stages)
}

/// Where a run came from: the monitor's runs tell the monitor when they end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Manual,
    Monitor,
}

/// What the UI side remembers of a run, snapshotted when it starts.
struct RunInfo {
    /// The inputs this run was started with, stamped on every result it makes.
    stamp: Fingerprint,
    /// The settings in one line, stamped on every result it makes.
    summary: String,
    /// Whether the auto-cull was on (the status text says what it dropped).
    auto_cull: bool,
    /// What the status says about the aligned cache (reused, or out of date) and
    /// about the memory budget.
    notes: Vec<String>,
}

/// The state of a run on the UI side: what the events are published with.
struct RunUi {
    info: RunInfo,
    started: Instant,
    /// How many results were published so far (the first one brings the culled
    /// marks).
    published: usize,
}

/// Run the Stack pass over the current source list (the toolbar button and the
/// stale badge's Recompute both come here).
pub fn start(ctx: &Ctx, algo: &str, ai_model: String, ai_device: String) {
    if jobs::refuse_if_busy(ctx, "Stack") {
        return;
    }
    if ctx.settings.snapshot().auto_cull {
        explainer::show_once(
            ctx,
            "auto-cull",
            "Auto-cull is on",
            "Frames that add almost nothing to the sharp detail are left out of this run. \
             They stay in the list, dimmed and marked (culled). If every frame would be \
             dropped, all are kept.",
        );
    }
    let paths = ctx.state.borrow().file_paths.clone();
    launch(ctx, &paths, algo, ai_model, ai_device, Origin::Manual);
}

/// A stack run for the folder monitor: `Apex` over `paths`. `false` when it could
/// not be started (the user has been told).
#[cfg_attr(
    target_arch = "wasm32",
    allow(dead_code, reason = "only the desktop's folder monitor starts runs")
)]
#[must_use]
pub fn start_monitor_run(ctx: &Ctx, paths: &[PathBuf]) -> bool {
    launch(
        ctx,
        paths,
        "Apex",
        String::new(),
        String::new(),
        Origin::Monitor,
    )
}

fn launch(
    ctx: &Ctx,
    paths: &[PathBuf],
    algo: &str,
    ai_model: String,
    ai_device: String,
    origin: Origin,
) -> bool {
    if paths.is_empty() {
        feedback::notify(ctx, MessageKind::Warning, "Add images before stacking.");
        return false;
    }
    let settings = ctx.settings.snapshot();
    let align_model = ctx.ui.upgrade().and_then(|app| sync_align_model(&app, ctx));
    let sources = fingerprint_paths(paths);
    let wanted = frames::paths_for_run(paths, &settings, true);
    let (reuse_aligned, cache_note) = plan_alignment(ctx, &sources, &wanted, &settings);
    let passes = passes_for(algo);
    let info = RunInfo {
        stamp: fingerprint_stack(&sources, &settings),
        summary: summarize_settings(&settings),
        auto_cull: settings.auto_cull,
        notes: cache_note.into_iter().collect(),
    };
    let ai = passes.contains(&Pass::Ai).then_some((ai_model, ai_device));
    let request = StackRequest {
        paths: wanted,
        settings,
        passes,
        reuse_aligned,
        ai,
        align_model,
    };
    // The runner may look at the run first (the browser checks its memory budget
    // and asks), so the run starts from its answer.
    let then_ctx = ctx.clone();
    runner::current().preflight(
        ctx,
        request,
        Box::new(move |answer| {
            if let Some((request, note)) = answer {
                let mut info = info;
                info.notes.extend(note);
                begin_run(&then_ctx, request, info, origin);
            } else {
                feedback::notify(&then_ctx, MessageKind::Info, "Stack cancelled.");
                end_origin(&then_ctx, origin);
            }
        }),
    );
    true
}

/// Decide whether the run takes the frames of the last Align pass: only when that
/// is current for these files and settings and holds every file the run works on.
/// The second value is the status note: the cache was reused, or it was out of
/// date and the run aligns afresh (`None` when there is nothing to say).
fn plan_alignment(
    ctx: &Ctx,
    sources: &Fingerprint,
    wanted: &[PathBuf],
    settings: &StackingSettings,
) -> (bool, Option<String>) {
    let aligns = settings.alignment_mode != AlignmentModeSetting::None;
    let state = ctx.state.borrow();
    if state.reusable_frames(sources, settings, wanted).is_some() {
        let note = aligns.then(|| "reused the frames of the last Align".to_owned());
        return (true, note);
    }
    let note = state
        .aligned_staleness(sources, settings)
        .filter(|_| aligns)
        .map(|reason| {
            format!(
                "aligned afresh: the last Align is out of date, {}",
                reason.words()
            )
        });
    (false, note)
}

/// What cancelling does beyond stopping the run: a `Relief` preview that waits for
/// an answer is answered with "cancelled" and closed, so a cancel never waits for
/// a click.
fn release_preview(ctx: &Ctx) -> std::rc::Rc<dyn Fn()> {
    let ctx = ctx.clone();
    std::rc::Rc::new(move || relief_preview::cancel_pending(&ctx))
}

/// Tell the monitor a run it started is over (nothing for a manual run).
fn end_origin(ctx: &Ctx, origin: Origin) {
    if origin == Origin::Monitor {
        monitor::run_finished(ctx);
    }
}

/// Start the run: register the job (its chip), hand the request to the runner and
/// give the job the handle that cancels it.
fn begin_run(ctx: &Ctx, request: StackRequest, info: RunInfo, origin: Origin) {
    // Something else may have started while the preflight asked.
    if jobs::refuse_if_busy(ctx, "Stack") {
        end_origin(ctx, origin);
        return;
    }
    let spec = JobSpec {
        kind: "stack",
        label: "Stack".to_owned(),
        cancellable: true,
        blocking: true,
        plan: plan_for(&request.settings, &request.passes),
        on_cancel: Some(release_preview(ctx)),
    };
    let Some(job) = jobs::begin(ctx, spec) else {
        end_origin(ctx, origin);
        return;
    };
    let mut run = RunUi {
        info,
        started: Instant::now(),
        published: 0,
    };
    let events = {
        let (ctx, job) = (ctx.clone(), std::rc::Rc::clone(&job));
        Box::new(move |event| on_event(&ctx, &job, &mut run, event, origin))
    };
    let handle = runner::current().run_stack(request, events, relief_preview::asker(ctx));
    job.attach(handle);
}

/// One event of a run, on the UI thread.
fn on_event(ctx: &Ctx, job: &Job, run: &mut RunUi, event: RunEvent, origin: Origin) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    match event {
        RunEvent::Progress {
            step,
            fraction,
            label,
        } => job.progress(&app, stage_of(step), fraction, &label),
        RunEvent::Warning(text) => feedback::show(&app, MessageKind::Warning, &text),
        RunEvent::LiveFrame(buffer) => preview::submit_frame(buffer),
        RunEvent::Result(result) => publish_result(&app, ctx, run, result),
        // Only an Align run reports this, only a Sort / Cull run the other.
        RunEvent::AlignDone { .. } | RunEvent::SortCullDone { .. } => {}
        RunEvent::Finished(result) => {
            job.end(&app);
            finished(&app, result, &run.info);
            end_origin(ctx, origin);
        }
    }
}

/// The UI-thread end of a run: report.
fn finished(app: &App, result: Result<Option<String>, JobError>, info: &RunInfo) {
    match result {
        Ok(runner_note) => {
            let finished = "Finished! All results available below.";
            let notes: Vec<String> = runner_note
                .into_iter()
                .chain(info.notes.iter().cloned())
                .collect();
            let text = if notes.is_empty() {
                finished.to_owned()
            } else {
                format!("{finished} ({})", notes.join("; "))
            };
            feedback::show(app, MessageKind::Success, &text);
        }
        Err(error) => jobs::report_error(app, "Stack", &error),
    }
}

// ── Publishing ──────────────────────────────────────────────────────────────

/// The status text of a finished pass.
fn merged_status(label: &str, kept: usize, total: usize, auto_cull: bool) -> String {
    // A host without a log file (the browser) leaves the pointer out.
    let log = log_path().map_or_else(String::new, |path| format!(" — log: {}", path.display()));
    if auto_cull {
        let dropped = total.saturating_sub(kept);
        format!("Merged [{label}] {kept}/{total} ({dropped} dropped){log}")
    } else {
        format!("Merged [{label}] {kept}/{total}{log}")
    }
}

/// How long the run took so far, as `42s` or `3m 7s`.
fn elapsed_text(started: Instant) -> String {
    let secs = started.elapsed().as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else {
        format!("{}m {}s", secs / 60, secs % 60)
    }
}

/// List a finished pass with the record of the inputs it was built from, show it,
/// start its retouch session, and (first pass of a run) mark the files the
/// auto-cull dropped. The source list itself is never rewritten.
fn publish_result(app: &App, ctx: &Ctx, run: &mut RunUi, result: StackResult) {
    let StackResult {
        pass,
        fused,
        preview_path,
        culled,
        kept,
        total,
    } = result;
    let label = pass.label();
    let buffer = planar_to_rgba_buffer(&fused);
    let status = merged_status(label, kept, total, run.info.auto_cull);
    let elapsed = elapsed_text(run.started);
    let n_frames = i32::try_from(kept).unwrap_or(i32::MAX);
    let meta = ResultMeta {
        fingerprint: Some(run.info.stamp),
        label: label.to_owned(),
        settings_summary: run.info.summary.clone(),
    };
    let first = run.published == 0;
    run.published += 1;
    let results = {
        let mut state = ctx.state.borrow_mut();
        if first {
            state.set_culled(culled);
        }
        state.fused = Some(std::sync::Arc::clone(&fused));
        let session = RetouchSession::new(std::sync::Arc::clone(&fused), fused);
        state.retouch.start_session(session, preview_path.clone());
        state.push_result_with(preview_path, meta);
        state.result_paths.clone()
    };
    update_result_list(app, &results);
    app.set_displayed_image(Image::from_rgba8(buffer));
    app.set_status(status.into());
    app.set_frame_count(n_frames);
    app.set_elapsed_text(elapsed.into());
    app.set_can_undo(false);
    app.set_can_redo(false);
    // The state's `fused` and the session are set: refresh the retouch options.
    retouch::sync_panel(ctx);
    // The culled marks and the newest result's currency changed.
    stale::refresh(app, ctx);
    // A new session starts with no strokes (and waiting ones paint onto it).
    super::project::mark_dirty(ctx);
}

#[cfg(test)]
mod tests {
    use stacker_gui::settings::{AlignmentModeSetting, StackingSettings};

    use super::{Pass, RunStep, passes_for, plan_for, stage_fuse, stage_of, stage_save};

    #[test]
    fn algorithm_text_selects_the_passes() {
        assert_eq!(passes_for("Relief (Depth Map)"), vec![Pass::Relief]);
        assert_eq!(passes_for("Apex (Laplacian)"), vec![Pass::Apex]);
        assert_eq!(
            passes_for("Both (Relief & Apex)"),
            vec![Pass::Relief, Pass::Apex]
        );
        assert_eq!(passes_for("AI Model (Neural)"), vec![Pass::Ai]);
        assert_eq!(passes_for("Strata (Guided Fusion)"), vec![Pass::Strata]);
        assert_eq!(
            passes_for("All Three (Strata, Relief & Apex)"),
            vec![Pass::Strata, Pass::Relief, Pass::Apex],
            "All Three names every engine and must win over them"
        );
    }

    #[test]
    fn every_listed_algorithm_selects_its_passes() {
        let expected = [
            vec![Pass::Relief],
            vec![Pass::Apex],
            vec![Pass::Strata],
            vec![Pass::Relief, Pass::Apex],
            vec![Pass::Strata, Pass::Relief, Pass::Apex],
        ];
        for (label, passes) in super::ALGORITHMS.iter().zip(expected) {
            assert_eq!(passes_for(label), passes, "{label}");
        }
    }

    #[test]
    fn the_labels_of_earlier_versions_are_still_read() {
        assert_eq!(passes_for("DMap (Depth Map)"), vec![Pass::Relief]);
        assert_eq!(passes_for("PMax (Laplacian)"), vec![Pass::Apex]);
        assert_eq!(
            passes_for("Both (DMap & PMax)"),
            vec![Pass::Relief, Pass::Apex]
        );
    }

    #[test]
    fn the_strata_pass_is_named_in_the_record() {
        assert_eq!(Pass::Strata.label(), "Strata");
    }

    #[test]
    fn the_sort_rule_gives_the_cull_stage_a_share_of_the_bar() {
        let settings = StackingSettings {
            auto_cull: false,
            sort_by_sharpness: true,
            ..StackingSettings::default()
        };
        let plan = plan_for(&settings, &[Pass::Apex]);
        assert!(plan.fraction(2, 1.0) > plan.fraction(1, 1.0));
    }

    #[test]
    fn both_mode_is_one_job_with_two_fuse_stages() {
        let settings = StackingSettings::default();
        let plan = plan_for(&settings, &[Pass::Relief, Pass::Apex]);
        // load, align, cull, then fuse + save per pass.
        assert_eq!(plan.len(), 7);
        assert_eq!(plan.label(stage_fuse(1)), Some("Fusing"));
        assert_eq!(plan.label(stage_save(1)), Some("Saving"));
        assert!((plan.fraction(stage_save(1), 1.0) - 1.0).abs() < 1e-5);
        // The first pass ends before the second begins.
        assert!(plan.fraction(stage_save(0), 1.0) < plan.fraction(stage_fuse(1), 0.5));
    }

    #[test]
    fn a_run_without_alignment_skips_that_weight() {
        let settings = StackingSettings {
            alignment_mode: AlignmentModeSetting::None,
            auto_cull: false,
            ..StackingSettings::default()
        };
        let plan = plan_for(&settings, &[Pass::Apex]);
        let after_load = plan.fraction(0, 1.0);
        assert!(
            (plan.fraction(2, 1.0) - after_load).abs() < 1e-6,
            "alignment and cull take no share of the bar when they are off"
        );
    }

    #[test]
    fn every_run_step_maps_onto_its_stage() {
        assert_eq!(stage_of(RunStep::Load), 0);
        assert_eq!(stage_of(RunStep::Align), 1);
        assert_eq!(stage_of(RunStep::Cull), 2);
        assert_eq!(stage_of(RunStep::Fuse(0)), stage_fuse(0));
        assert_eq!(stage_of(RunStep::Save(1)), stage_save(1));
        let plan = plan_for(&StackingSettings::default(), &[Pass::Relief, Pass::Apex]);
        assert!(plan.label(stage_of(RunStep::Save(1))).is_some());
        assert!(plan.label(stage_of(RunStep::Save(2))).is_none());
    }

    #[test]
    fn the_status_names_what_the_cull_dropped() {
        let culled = super::merged_status("Apex", 7, 10, true);
        assert!(
            culled.starts_with("Merged [Apex] 7/10 (3 dropped)"),
            "{culled}"
        );
        let plain = super::merged_status("Relief", 4, 4, false);
        assert!(plain.starts_with("Merged [Relief] 4/4"), "{plain}");
        assert!(!plain.contains("dropped"), "{plain}");
    }
}
