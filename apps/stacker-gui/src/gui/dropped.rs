//! Drag and drop: image files, a folder of images, or a project file dropped on
//! the window.
//!
//! # Where the drops come from
//!
//! Slint 1.18.1 has a `DropArea` element, but with the winit backend it only
//! takes drags that start in a Slint `DragArea`: the backend never turns the
//! operating system's file drops into drop events. winit itself reports them as
//! `WindowEvent::DroppedFile` (one event per file), and Slint hands every winit
//! window event to the filter registered with
//! `WinitWindowAccessor::on_winit_window_event`. This module registers that
//! filter (desktop builds). It is the window's only one: a second registration
//! would replace it.
//!
//! The browser build has no such filter: the page receives the drop itself, holds
//! the files' bytes under virtual paths and calls [`handle_paths`] with them.
//!
//! # What happens to a drop
//!
//! The files of one drop arrive as separate events, so they are collected for a
//! moment ([`COALESCE`]) and handled together:
//!
//! * a folder is replaced by the images directly inside it (not its subfolders);
//! * images go through the same add path as File > Add Images: the same
//!   duplicate check, source list, canvas and message;
//! * a project file (`.zstack.toml`) is opened instead of adding images;
//! * anything else is ignored and named in a message;
//! * while a dialog is open the drop is refused, so it cannot change the source
//!   list under a dialog that is about to act on it.

use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::{cell::RefCell, rc::Rc, time::Duration};

use slint::ComponentHandle;
#[cfg(not(target_arch = "wasm32"))]
use slint::{
    Timer, TimerMode,
    winit_030::{EventResult, WinitWindowAccessor, winit::event::WindowEvent},
};
use stacker_gui::stall_guard::stall_guard;
use stacker_pipeline::order::has_frame_extension;

use super::{
    Ctx,
    feedback::{self, MessageKind},
    files,
};
use crate::{
    App, DialogModel, ExplainerModel, ExportModel, ManualModel, ProjectModel, ShortcutsModel,
};

/// How long after the last dropped file the batch is handled: the files of one
/// drop arrive within a few milliseconds of each other.
#[cfg(not(target_arch = "wasm32"))]
const COALESCE: Duration = Duration::from_millis(100);

/// The suffix of a project file.
const PROJECT_SUFFIX: &str = ".zstack.toml";

/// Register the window's file-drop filter (the only winit event filter of the
/// window). The browser build registers nothing: its page calls
/// [`handle_paths`].
#[cfg(not(target_arch = "wasm32"))]
pub fn setup(ui: &App, ctx: &Ctx) {
    let pending: Rc<RefCell<Vec<PathBuf>>> = Rc::new(RefCell::new(Vec::new()));
    let timer = Rc::new(Timer::default());
    let ctx = ctx.clone();
    ui.window().on_winit_window_event(move |_window, event| {
        if let WindowEvent::DroppedFile(path) = event {
            pending.borrow_mut().push(path.clone());
            let (pending, ctx) = (Rc::clone(&pending), ctx.clone());
            // Restarting the timer postpones the handling until the batch is complete.
            timer.start(TimerMode::SingleShot, COALESCE, move || {
                stall_guard("dropped::handle", || {
                    handle(&ctx, std::mem::take(&mut *pending.borrow_mut()));
                });
            });
        }
        EventResult::Propagate
    });
}

/// Register the window's file-drop filter: nothing in the browser build (see the
/// module docs).
#[cfg(target_arch = "wasm32")]
pub const fn setup(_ui: &App, _ctx: &Ctx) {}

/// Handle one finished drop of `dropped` (image files, folders, a project file)
/// as if the window had received it.
///
/// The browser page calls this with the virtual paths of the files it received.
/// Must run on the UI thread.
pub fn handle_paths(ctx: &Ctx, dropped: Vec<PathBuf>) {
    stall_guard("dropped::handle", || handle(ctx, dropped));
}

// ── Sorting out what was dropped ─────────────────────────────────────────────

/// What a drop contains, after folders are opened.
#[derive(Debug, Default, PartialEq, Eq)]
struct Plan {
    /// Image files to add, sorted by name, each once.
    images: Vec<PathBuf>,
    /// The first project file, if any.
    project: Option<PathBuf>,
    /// Files that are neither.
    ignored: Vec<PathBuf>,
}

/// `true` for a file with one of the source-frame extensions.
fn is_image(path: &Path) -> bool {
    has_frame_extension(path)
        || (stacker_pipeline::load::RAW_SUPPORTED
            && stacker_pipeline::load::has_raw_extension(path))
}

/// `true` for a file named `*.zstack.toml`.
fn is_project(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(PROJECT_SUFFIX))
}

/// Sort `dropped` into a [`Plan`]. `list_dir` returns the entries of a folder, or
/// `None` for a path that is not a folder.
fn classify(dropped: Vec<PathBuf>, list_dir: impl Fn(&Path) -> Option<Vec<PathBuf>>) -> Plan {
    let mut plan = Plan::default();
    for path in dropped {
        if let Some(entries) = list_dir(&path) {
            // A folder offers its images only: no project, no deeper folders.
            plan.images
                .extend(entries.into_iter().filter(|entry| is_image(entry)));
        } else if is_project(&path) {
            if plan.project.is_none() {
                plan.project = Some(path);
            }
        } else if is_image(&path) {
            plan.images.push(path);
        } else {
            plan.ignored.push(path);
        }
    }
    plan.images
        .sort_by_cached_key(|path| path.to_string_lossy().to_lowercase());
    plan.images.dedup();
    plan
}

/// The entries of the folder `path`; `None` when `path` is not a folder (or cannot
/// be read: it is then ignored like any other unusable drop). The browser cannot
/// list folders, so every dropped path is a file there.
fn read_folder(path: &Path) -> Option<Vec<PathBuf>> {
    crate::platform::current().list_folder(path)
}

// ── Acting on it ─────────────────────────────────────────────────────────────

/// `true` while a modal layer is up.
fn modal_open(app: &App) -> bool {
    app.get_dialog_open()
        || app.global::<DialogModel>().get_confirm_open()
        || app.global::<ShortcutsModel>().get_open()
        || app.global::<ExportModel>().get_open()
        || app.global::<ManualModel>().get_open()
        || app.global::<ManualModel>().get_about_open()
        || app.global::<ExplainerModel>().get_open()
}

/// A finished drop of `dropped`, for the UI flow tests: the headless backend has
/// no window-system drops to deliver.
#[cfg(test)]
pub fn handle_for_tests(ctx: &Ctx, dropped: Vec<PathBuf>) {
    handle(ctx, dropped);
}

/// Handle one finished drop.
fn handle(ctx: &Ctx, dropped: Vec<PathBuf>) {
    if dropped.is_empty() {
        return;
    }
    let Some(app) = ctx.ui.upgrade() else {
        return;
    };
    if modal_open(&app) {
        feedback::show(
            &app,
            MessageKind::Info,
            "Close the open dialog first, then drop the files again.",
        );
        return;
    }
    let plan = classify(dropped, read_folder);
    if let Some(project) = plan.project {
        open_project(&app, &project);
        return;
    }
    if !plan.ignored.is_empty() {
        let kind = if plan.images.is_empty() {
            MessageKind::Warning
        } else {
            MessageKind::Info
        };
        feedback::show(
            &app,
            kind,
            &format!(
                "{} dropped file(s) are not images and were ignored: {}",
                plan.ignored.len(),
                feedback::list_files(plan.ignored.iter().map(PathBuf::as_path)),
            ),
        );
    }
    if !plan.images.is_empty() {
        add_images(&app, ctx, plan.images);
    }
}

/// Add `images` through the add path File > Add Images ends in (duplicate check,
/// source list, canvas, message), without the picker.
fn add_images(app: &App, ctx: &Ctx, images: Vec<PathBuf>) {
    files::add_picked_files(app, ctx, images);
}

/// Open the dropped project file through `ProjectModel.open-path`, which asks
/// before it discards unsaved changes.
fn open_project(app: &App, project: &Path) {
    app.global::<ProjectModel>()
        .invoke_open_path(project.display().to_string().into());
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{Plan, classify, is_image, is_project};

    fn p(text: &str) -> PathBuf {
        PathBuf::from(text)
    }

    fn no_folders(_: &Path) -> Option<Vec<PathBuf>> {
        None
    }

    #[test]
    fn image_extensions_are_matched_case_insensitively() {
        for name in ["a.jpg", "a.JPEG", "a.Png", "a.tif", "a.TIFF"] {
            assert!(is_image(Path::new(name)), "{name}");
        }
        for name in ["a.txt", "a", "a.jpg.bak", "a.zstack.toml"] {
            assert!(!is_image(Path::new(name)), "{name}");
        }
    }

    #[test]
    fn project_files_are_recognised_by_their_double_suffix() {
        assert!(is_project(Path::new("work/scene.zstack.toml")));
        assert!(is_project(Path::new("SCENE.ZSTACK.TOML")));
        assert!(!is_project(Path::new("settings.toml")));
        assert!(!is_project(Path::new("scene.zstack")));
    }

    #[test]
    fn images_are_sorted_by_name_and_listed_once() {
        let plan = classify(
            vec![
                p("d/IMG_3.jpg"),
                p("d/img_1.jpg"),
                p("d/IMG_3.jpg"),
                p("d/IMG_2.png"),
            ],
            no_folders,
        );
        assert_eq!(
            plan.images,
            vec![p("d/img_1.jpg"), p("d/IMG_2.png"), p("d/IMG_3.jpg")]
        );
        assert_eq!(plan.project, None);
        assert_eq!(plan.ignored, Vec::<PathBuf>::new());
    }

    #[test]
    fn a_folder_contributes_only_its_images() {
        let plan = classify(vec![p("shoot")], |dir| {
            (dir == Path::new("shoot")).then(|| {
                vec![
                    p("shoot/b.tif"),
                    p("shoot/notes.txt"),
                    p("shoot/a.jpg"),
                    p("shoot/scene.zstack.toml"),
                ]
            })
        });
        assert_eq!(plan.images, vec![p("shoot/a.jpg"), p("shoot/b.tif")]);
        assert_eq!(plan.project, None, "a folder never opens a project");
        assert_eq!(
            plan.ignored,
            Vec::<PathBuf>::new(),
            "its other files are skipped quietly"
        );
    }

    #[test]
    fn the_first_project_file_wins() {
        let plan = classify(vec![p("a.zstack.toml"), p("b.zstack.toml")], no_folders);
        assert_eq!(plan.project, Some(p("a.zstack.toml")));
    }

    #[test]
    fn unusable_files_are_reported_not_added() {
        let plan = classify(vec![p("x.pdf"), p("y.jpg"), p("noext")], no_folders);
        assert_eq!(plan.images, vec![p("y.jpg")]);
        assert_eq!(plan.ignored, vec![p("x.pdf"), p("noext")]);
    }

    #[test]
    fn an_empty_drop_gives_an_empty_plan() {
        assert_eq!(classify(Vec::new(), no_folders), Plan::default());
    }
}
