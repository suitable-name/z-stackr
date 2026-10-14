//! The project (session) document: New, Open, Open Recent, Save, Save As, the
//! recent list and the dirty rule.
//!
//! # What a project is
//!
//! The source list, the settings and the retouch strokes, saved as
//! `<name>.zstack.toml` next to the sources ([`stacker_gui::project`]). Opening
//! one loads the sources through the same list the Add button feeds (so decode
//! and the stale fingerprints behave as for hand-added files), applies the
//! settings through the persister, and queues the strokes: they are replayed
//! onto the next stack result, since a stroke only means something over the
//! image it was painted on.
//!
//! # Dirty
//!
//! There is no flag. [`is_dirty`] builds the project from the live state and
//! compares its fingerprint with the one taken at the last save or open
//! ([`Baseline`]), so reverting a change cleans the document again and a save
//! of unchanged content stays clean. The callbacks that change sources,
//! settings or strokes call [`mark_dirty`] / [`note_sources_changed`] so the
//! title and menu update at once; a 400 ms timer catches whatever else moves.
//!
//! # Unsaved changes
//!
//! New, Open and Open Recent go through [`unsaved::confirm_discard`]; closing
//! the window goes through [`unsaved::install_close_guard`].

use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    time::Duration,
};

use slint::{ComponentHandle, Image, ModelRc, SharedString, Timer, TimerMode, VecModel};
use stacker_gui::{
    pickers::{PickerKind, PickerRequest},
    project::{
        Baseline, FILE_SUFFIX, LoadedProject, Project, ProjectError, StrokeRecord, project_name,
        suggested_location, summary, with_project_suffix,
    },
    retouch::Stroke,
    settings::recent::{self as recent_projects, RecentProjects},
    stall_guard::stall_guard,
    state::AppState,
};

use super::{
    Ctx,
    confirm::Handler,
    feedback::{self, MessageKind},
    jobs, pickers, retouch, stale, unsaved,
};
use crate::{
    App, ProjectModel, RetouchModel,
    ui_helpers::{push_settings_to_ui, push_source_panel, update_result_list},
};

/// How often the title and menu re-check the dirty rule.
const WATCH_INTERVAL: Duration = Duration::from_millis(400);
/// How many recent projects the empty state shows as cards.
const CARDS: usize = 4;
/// The name of a project that was never saved.
const UNTITLED: &str = "Untitled";

/// What `ProjectModel` currently shows, so an unchanged answer pushes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pushed {
    name: String,
    path: String,
    dirty: bool,
}

/// The open document's bookkeeping (UI thread only).
#[derive(Default)]
struct Doc {
    /// Where the project lives; `None` until it is saved or opened.
    path: Option<PathBuf>,
    /// The content at the last save or open.
    baseline: Baseline,
    /// The export's result name carried through the file.
    output_name: Option<String>,
    /// Strokes loaded from a file and waiting for a result to paint on.
    pending: Option<Vec<StrokeRecord>>,
    /// The recent-projects list.
    recents: RecentProjects,
    pushed: Option<Pushed>,
}

thread_local! {
    static DOC: RefCell<Doc> = RefCell::new(Doc::default());
    static WATCH: Timer = Timer::default();
}

/// Load the recent list, take the clean baseline and register the project menu's
/// callbacks, the close guard and the dirty watcher.
pub fn setup(ui: &App, ctx: &Ctx) {
    // The browser build keeps no recent list: its project "files" are virtual and
    // gone with the page.
    let recents = if crate::platform::current().is_web() {
        RecentProjects::new()
    } else {
        recent_projects::load_stored()
    };
    // A fresh document: nothing of a window wired earlier on this thread (the UI
    // flow tests boot one per scenario) carries over.
    DOC.with(|doc| {
        *doc.borrow_mut() = Doc {
            recents,
            ..Doc::default()
        };
    });
    restart_baseline(ctx);
    wire(ui, ctx);
    unsaved::install_close_guard(ui, ctx);
    push_recents(ui);
    push_model(ctx);
    start_watch(ctx);
}

fn wire(ui: &App, ctx: &Ctx) {
    let model = ui.global::<ProjectModel>();
    model.on_new_project({
        let ctx = ctx.clone();
        move || stall_guard("project::new-project", || new_project(&ctx))
    });
    model.on_open_project({
        let ctx = ctx.clone();
        move || stall_guard("project::open-project", || open_project(&ctx))
    });
    model.on_open_path({
        let ctx = ctx.clone();
        move |path| {
            stall_guard("project::open-path", || {
                if let Some(path) = requested_path(path.as_str()) {
                    open_given(&ctx, path);
                }
            });
        }
    });
    model.on_open_recent({
        let ctx = ctx.clone();
        move |index| stall_guard("project::open-recent", || open_recent(&ctx, index))
    });
    model.on_save_project({
        let ctx = ctx.clone();
        move || stall_guard("project::save-project", || save_then(&ctx, None))
    });
    model.on_save_project_as({
        let ctx = ctx.clone();
        move || stall_guard("project::save-project-as", || save_as_then(&ctx, None))
    });
    model.on_clear_recent({
        let ctx = ctx.clone();
        move || stall_guard("project::clear-recent", || clear_recent(&ctx))
    });
}

fn start_watch(ctx: &Ctx) {
    let ctx = ctx.clone();
    WATCH.with(|timer| {
        timer.start(TimerMode::Repeated, WATCH_INTERVAL, move || refresh(&ctx));
    });
}

// ── The document ─────────────────────────────────────────────────────────────

/// The project as the live state describes it right now.
fn current(ctx: &Ctx) -> Project {
    let settings = ctx.settings.snapshot();
    let (sources, live) = {
        let state = ctx.state.borrow();
        let live: Vec<StrokeRecord> = state
            .retouch
            .strokes()
            .iter()
            .map(|stroke| StrokeRecord::from_stroke(stroke, &state.file_paths))
            .collect();
        (state.file_paths.clone(), live)
    };
    let (output_name, pending) = DOC.with(|doc| {
        let doc = doc.borrow();
        (doc.output_name.clone(), doc.pending.clone())
    });
    let mut project = Project::new(sources, settings);
    project.output_name = output_name;
    // Strokes still waiting for a result are part of the project too.
    project.strokes = pending.unwrap_or(live);
    project
}

/// `true` when the open project differs in content from what was last saved or
/// opened.
pub fn is_dirty(ctx: &Ctx) -> bool {
    let baseline = DOC.with(|doc| doc.borrow().baseline);
    baseline.is_dirty(&current(ctx))
}

/// The name the dialogs and the title use for the open project.
pub fn display_name() -> String {
    DOC.with(|doc| {
        doc.borrow()
            .path
            .as_deref()
            .map_or_else(|| UNTITLED.to_owned(), project_name)
    })
}

/// Sources, settings or strokes may have changed: re-check the dirty rule now.
pub fn mark_dirty(ctx: &Ctx) {
    refresh(ctx);
}

/// The source list changed (added, removed, cleared, replaced by the monitor).
pub fn note_sources_changed(ctx: &Ctx) {
    refresh(ctx);
}

/// Take the current content as the clean one.
fn restart_baseline(ctx: &Ctx) {
    let baseline = Baseline::of(&current(ctx));
    DOC.with(|doc| doc.borrow_mut().baseline = baseline);
}

/// One round of upkeep: paint waiting strokes when a result exists, then push
/// the name, path and dirty flag.
fn refresh(ctx: &Ctx) {
    replay_pending(ctx);
    push_model(ctx);
}

fn push_model(ctx: &Ctx) {
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    let dirty = is_dirty(ctx);
    let view = DOC.with(|doc| {
        let doc = doc.borrow();
        Pushed {
            name: doc
                .path
                .as_deref()
                .map_or_else(|| UNTITLED.to_owned(), project_name),
            path: doc
                .path
                .as_ref()
                .map_or_else(String::new, |p| p.display().to_string()),
            dirty,
        }
    });
    let changed = DOC.with(|doc| {
        let mut doc = doc.borrow_mut();
        if doc.pushed.as_ref() == Some(&view) {
            false
        } else {
            doc.pushed = Some(view.clone());
            true
        }
    });
    if changed {
        let model = app.global::<ProjectModel>();
        model.set_name(view.name.into());
        model.set_path(view.path.into());
        model.set_dirty(view.dirty);
    }
}

// ── Recents ──────────────────────────────────────────────────────────────────

fn push_recents(app: &App) {
    let (paths, cards) = DOC.with(|doc| {
        let doc = doc.borrow();
        let entries = doc.recents.entries();
        let paths: Vec<SharedString> = entries
            .iter()
            .map(|p| SharedString::from(p.display().to_string()))
            .collect();
        let cards: Vec<SharedString> = entries
            .iter()
            .take(CARDS)
            .map(|p| SharedString::from(recent_projects::display_label(p)))
            .collect();
        (paths, cards)
    });
    let model = app.global::<ProjectModel>();
    model.set_recent(ModelRc::new(VecModel::from(paths)));
    model.set_recent_cards(ModelRc::new(VecModel::from(cards)));
}

/// Write the list beside the settings file; a failure is logged, never shown
/// (the list is a convenience). The browser build keeps no list.
fn persist_recents() {
    if crate::platform::current().is_web() {
        return;
    }
    let list = DOC.with(|doc| doc.borrow().recents.clone());
    if let Err(e) = recent_projects::save_stored(&list) {
        tracing::warn!("could not save the recent projects: {e}");
    }
}

fn update_recents(ctx: &Ctx, change: impl FnOnce(&mut RecentProjects)) {
    DOC.with(|doc| change(&mut doc.borrow_mut().recents));
    persist_recents();
    if let Some(app) = ctx.ui.upgrade() {
        push_recents(&app);
    }
}

fn clear_recent(ctx: &Ctx) {
    update_recents(ctx, RecentProjects::clear);
}

// ── New and Open ─────────────────────────────────────────────────────────────

/// Drop the whole session: sources, results, aligned frames, retouch, canvas and
/// the document bookkeeping.
fn clear_session(app: &App, ctx: &Ctx) {
    if app.get_is_monitoring() {
        app.invoke_stop_monitor();
    }
    *ctx.state.borrow_mut() = AppState::default();
    // The state held only the record of the aligned set; the runner holds frames.
    crate::runner::current().forget_aligned();
    DOC.with(|doc| {
        let mut doc = doc.borrow_mut();
        doc.path = None;
        doc.output_name = None;
        doc.pending = None;
    });
    push_source_panel(app, &[]);
    update_result_list(app, &[]);
    app.set_displayed_image(Image::default());
    retouch::sync_panel(ctx);
    stale::refresh(app, ctx);
}

fn new_project(ctx: &Ctx) {
    if jobs::refuse_if_busy(ctx, "New project") {
        return;
    }
    unsaved::confirm_discard(ctx, |app, ctx| {
        if jobs::refuse_if_busy(ctx, "New project") {
            return;
        }
        clear_session(app, ctx);
        restart_baseline(ctx);
        push_model(ctx);
        feedback::show_status(app, "New project.");
    });
}

fn open_project(ctx: &Ctx) {
    if jobs::refuse_if_busy(ctx, "Open project") {
        return;
    }
    unsaved::confirm_discard(ctx, |_app, ctx| {
        let request = PickerRequest::new(PickerKind::OpenFile)
            .titled("Open project")
            .filter("z-stackr project", &["toml"]);
        pickers::pick(ctx, request, |app, ctx, picked| {
            if let Some(path) = picked.into_iter().next() {
                open_path(app, ctx, &path);
            }
        });
    });
}

fn open_recent(ctx: &Ctx, index: i32) {
    let path = usize::try_from(index)
        .ok()
        .and_then(|i| DOC.with(|doc| doc.borrow().recents.entries().get(i).cloned()));
    if let Some(path) = path {
        open_given(ctx, path);
    }
}

/// The path a `ProjectModel.open-path` call carries; `None` for an empty or
/// blank one, which opens nothing.
fn requested_path(text: &str) -> Option<PathBuf> {
    let text = text.trim();
    (!text.is_empty()).then(|| PathBuf::from(text))
}

/// Open the project at `path` the way Open Project does after its picker: refused
/// while a job runs, and through the unsaved-changes guard.
fn open_given(ctx: &Ctx, path: PathBuf) {
    if jobs::refuse_if_busy(ctx, "Open project") {
        return;
    }
    unsaved::confirm_discard(ctx, move |app, ctx| open_path(app, ctx, &path));
}

/// Load the project at `path` into the session.
fn open_path(app: &App, ctx: &Ctx, path: &Path) {
    if jobs::refuse_if_busy(ctx, "Open project") {
        return;
    }
    let platform = crate::platform::current();
    let LoadedProject {
        project,
        mut warnings,
    } = match Project::load(path) {
        Ok(loaded) => loaded,
        Err(error) => {
            if matches!(&error, ProjectError::Io(e) if e.kind() == std::io::ErrorKind::NotFound) {
                update_recents(ctx, |recents| recents.forget(path));
            }
            feedback::show(
                app,
                MessageKind::Error,
                &format!("Could not open {}: {error}", platform.display_path(path)),
            );
            return;
        }
    };
    let note = summary(&project);
    let Project {
        sources,
        settings,
        output_name,
        strokes,
        ..
    } = project;

    clear_session(app, ctx);
    ctx.settings.update(|stored| stored.clone_from(&settings));
    push_settings_to_ui(app, &settings);

    let missing: Vec<&Path> = sources
        .iter()
        .filter(|source| platform.file_stamp(source).is_none())
        .map(PathBuf::as_path)
        .collect();
    if !missing.is_empty() {
        warnings.push(format!(
            "{} source file(s) were not found: {}",
            missing.len(),
            feedback::list_files(missing.iter().copied())
        ));
    }

    let (list, first) = {
        let mut state = ctx.state.borrow_mut();
        state.add_paths(sources);
        let first = state.file_paths.first().cloned();
        state.viewed_path.clone_from(&first);
        (state.file_paths.clone(), first)
    };
    push_source_panel(app, &list);
    if first
        .as_deref()
        .is_some_and(|first| platform.file_stamp(first).is_some())
    {
        // The same click the sidebar row would make: records the selection and
        // decodes the frame off the UI thread.
        app.invoke_file_clicked(0);
    }

    DOC.with(|doc| {
        let mut doc = doc.borrow_mut();
        doc.path = Some(path.to_path_buf());
        doc.output_name = output_name;
        doc.pending = (!strokes.is_empty()).then_some(strokes);
    });
    restart_baseline(ctx);
    update_recents(ctx, |recents| recents.record(path));
    push_model(ctx);
    stale::refresh(app, ctx);

    let name = project_name(path);
    if warnings.is_empty() {
        feedback::show(
            app,
            MessageKind::Success,
            &format!("Opened {name} ({note})."),
        );
    } else {
        feedback::show(
            app,
            MessageKind::Warning,
            &format!("Opened {name} with notes: {}", warnings.join("; ")),
        );
    }
}

// ── Save ─────────────────────────────────────────────────────────────────────

/// Save the project, asking for a path first when it has none, and run `then`
/// after a successful write. A cancelled picker or a failed write drops `then`.
pub fn save_then(ctx: &Ctx, then: Option<Handler>) {
    let path = DOC.with(|doc| doc.borrow().path.clone());
    match path {
        Some(path) => finish_save(ctx, &path, then),
        None => save_as_then(ctx, then),
    }
}

/// Pick a path (next to the sources by default), save there, then run `then`.
fn save_as_then(ctx: &Ctx, then: Option<Handler>) {
    let (sources, current_path) = (
        ctx.state.borrow().file_paths.clone(),
        DOC.with(|doc| doc.borrow().path.clone()),
    );
    let mut request = PickerRequest::new(PickerKind::SaveFile)
        .titled("Save project as")
        .filter("z-stackr project", &["toml"]);
    if let Some(path) = &current_path {
        request = request
            .file_name(&format!("{}{FILE_SUFFIX}", project_name(path)))
            .starting_in(path.parent().map(Path::to_path_buf));
    } else if let Some((folder, name)) =
        suggested_location(&sources).filter(|_| !crate::platform::current().is_web())
    {
        // The browser's virtual sources have no folder worth naming the project after.
        request = request.file_name(&name).starting_in(Some(folder));
    } else {
        request = request.file_name(&format!("{UNTITLED}{FILE_SUFFIX}"));
    }
    pickers::pick(ctx, request, move |_app, ctx, picked| {
        let Some(path) = picked.into_iter().next() else {
            return;
        };
        finish_save(ctx, &with_project_suffix(path), then);
    });
}

fn finish_save(ctx: &Ctx, path: &Path, then: Option<Handler>) {
    let project = current(ctx);
    let platform = crate::platform::current();
    if let Err(error) = project.save(path) {
        feedback::notify(
            ctx,
            MessageKind::Error,
            &format!(
                "Could not save the project to {}: {error}",
                platform.display_path(path)
            ),
        );
        return;
    }
    DOC.with(|doc| {
        let mut doc = doc.borrow_mut();
        doc.path = Some(path.to_path_buf());
        doc.baseline = Baseline::of(&project);
    });
    update_recents(ctx, |recents| recents.record(path));
    push_model(ctx);
    feedback::notify(
        ctx,
        MessageKind::Success,
        &format!("Saved {}", platform.display_path(path)),
    );
    if let Some(then) = then
        && let Some(app) = ctx.ui.upgrade()
    {
        then(&app, ctx);
    }
}

// ── Strokes waiting for a result ─────────────────────────────────────────────

/// Paint the strokes a project brought along onto the first retouch session that
/// exists (the next stack result), once.
fn replay_pending(ctx: &Ctx) {
    let ready = {
        let state = ctx.state.borrow();
        state.retouch.session.is_some() && !state.retouch.stroke_open()
    };
    if !ready {
        return;
    }
    let Some(records) = DOC.with(|doc| doc.borrow_mut().pending.take()) else {
        return;
    };
    let (applied, source) = {
        let mut state = ctx.state.borrow_mut();
        let strokes: Vec<Stroke> = records
            .iter()
            .map(|record| record.to_stroke(&state.file_paths))
            .collect();
        let applied = state.retouch.replay(&strokes);
        let source = strokes
            .iter()
            .rev()
            .find_map(|stroke| stroke.source.clone());
        (applied, source)
    };
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    // The replayed strokes reach the canvas through the preview worker.
    retouch::present(ctx);
    retouch::sync_panel(ctx);
    if let Some(source) = source {
        let index = ctx
            .state
            .borrow()
            .retouch
            .sources()
            .iter()
            .position(|path| *path == source)
            .and_then(|i| i32::try_from(i).ok());
        if let Some(index) = index {
            app.global::<RetouchModel>().invoke_source_chosen(index);
        }
    }
    if applied > 0 {
        feedback::show(
            &app,
            MessageKind::Info,
            &format!("Restored {applied} retouch stroke(s) from the project."),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::requested_path;

    #[test]
    fn a_dropped_path_is_taken_as_it_is() {
        assert_eq!(
            requested_path("C:/shots/rose.zstack.toml"),
            Some(PathBuf::from("C:/shots/rose.zstack.toml"))
        );
    }

    #[test]
    fn surrounding_blanks_are_not_part_of_the_path() {
        assert_eq!(
            requested_path("  shots/rose.zstack.toml\n"),
            Some(PathBuf::from("shots/rose.zstack.toml"))
        );
    }

    #[test]
    fn an_empty_path_opens_nothing() {
        assert_eq!(requested_path(""), None);
        assert_eq!(requested_path("   \t"), None);
    }
}
