//! The action matrix: every toolbar button, key shortcut, menu item and sidebar
//! action, performed in every app state, each on a freshly booted app.
//!
//! Per cell it checks what must hold whatever the action was:
//!
//! * nothing panics (a panic fails the whole test);
//! * at most one modal layer is open, and the Retouch tool is only on with a
//!   result to paint on;
//! * no file dialog opens except where the action asks for one;
//! * while a modal layer is up, a shortcut or a menu item changes nothing;
//! * while a job runs, no second job starts;
//! * afterwards the app gets back to idle: the job ends, Esc closes every layer,
//!   and the keyboard still works (`?` opens the shortcut list).

use slint::{ComponentHandle, Model, platform::Key};
use stacker_gui::pickers::{self, PickerKind};

use super::harness::{FAST_SETTINGS, Outcome, Ui, sleep, wait_until};
use crate::{ActivityModel, AppModel, ProjectModel, RetouchModel, ViewportModel};

/// Where the app is when the action happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Empty,
    Files,
    Result,
    RetouchOn,
    /// A `Relief` stack waiting at its preview card: a job that stays running.
    JobPaused,
    SettingsOpen,
    ShortcutsOpen,
    ConfirmOpen,
    ExportOpen,
}

pub const STATES: &[State] = &[
    State::Empty,
    State::Files,
    State::Result,
    State::RetouchOn,
    State::JobPaused,
    State::SettingsOpen,
    State::ShortcutsOpen,
    State::ConfirmOpen,
    State::ExportOpen,
];

impl State {
    /// The modal layer this state holds open, if any.
    const fn modal(self) -> Option<&'static str> {
        match self {
            Self::SettingsOpen => Some("settings"),
            Self::ShortcutsOpen => Some("shortcuts"),
            Self::ConfirmOpen => Some("confirm"),
            Self::ExportOpen => Some("export"),
            _ => None,
        }
    }
}

/// How an action reaches the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Via {
    Click,
    Key,
    Menu,
}

/// One user action.
#[derive(Debug, Clone, Copy)]
pub struct Action {
    pub name: &'static str,
    via: Via,
    /// Click: the accessible label. Key: the key text. Menu: "Menu/Item".
    target: &'static str,
    ctrl: bool,
    shift: bool,
    /// The file dialogs this action may open.
    dialogs: &'static [PickerKind],
}

const fn click(name: &'static str, target: &'static str, dialogs: &'static [PickerKind]) -> Action {
    Action {
        name,
        via: Via::Click,
        target,
        ctrl: false,
        shift: false,
        dialogs,
    }
}
const fn key(
    name: &'static str,
    target: &'static str,
    ctrl: bool,
    shift: bool,
    dialogs: &'static [PickerKind],
) -> Action {
    Action {
        name,
        via: Via::Key,
        target,
        ctrl,
        shift,
        dialogs,
    }
}
const fn menu(name: &'static str, target: &'static str, dialogs: &'static [PickerKind]) -> Action {
    Action {
        name,
        via: Via::Menu,
        target,
        ctrl: false,
        shift: false,
        dialogs,
    }
}

const NONE: &[PickerKind] = &[];
const OPEN_FILES: &[PickerKind] = &[PickerKind::OpenFiles];
const OPEN_FILE: &[PickerKind] = &[PickerKind::OpenFile];
const SAVE_FILE: &[PickerKind] = &[PickerKind::SaveFile];
const FOLDER: &[PickerKind] = &[PickerKind::PickFolder];

pub const ACTIONS: &[Action] = &[
    // Toolbar.
    click("toolbar Open Files", "Open Files", OPEN_FILES),
    click("toolbar Settings", "Settings", NONE),
    click("toolbar Start Stacking", "Start Stacking", NONE),
    click("toolbar Live Monitor", "Live Monitor", FOLDER),
    click("toolbar Align", "Align", NONE),
    click("toolbar Show Aligned", "Show Aligned", NONE),
    click("toolbar Retouch", "Retouch", NONE),
    click("toolbar Export", "Export\u{2026}", NONE),
    click("toolbar ?", "?", NONE),
    // Below the canvas.
    click("zoom Fit", "Fit", NONE),
    click("zoom 100%", "100%", NONE),
    click("zoom -", "-", NONE),
    click("zoom +", "+", NONE),
    // Sidebar.
    click("sidebar Clear", "Clear all files", NONE),
    click("sidebar remove first", "Remove frame_00.png", NONE),
    click("sidebar show first", "Show frame_00.png", NONE),
    // Keys.
    key("Ctrl+O", "o", true, false, OPEN_FILES),
    key("Ctrl+S", "s", true, false, SAVE_FILE),
    key("Ctrl+Shift+S", "S", true, true, SAVE_FILE),
    key("Ctrl+N", "n", true, false, NONE),
    key("Ctrl+Shift+O", "O", true, true, OPEN_FILE),
    key("Ctrl+E", "e", true, false, NONE),
    key("Ctrl+,", ",", true, false, NONE),
    key("Ctrl+Z", "z", true, false, NONE),
    key("Ctrl+Y", "y", true, false, NONE),
    key("Ctrl+Shift+Z", "Z", true, true, NONE),
    key("F1", "F1", false, false, NONE),
    key("?", "?", false, true, NONE),
    key("+", "+", false, false, NONE),
    key("=", "=", false, false, NONE),
    key("-", "-", false, false, NONE),
    key("0", "0", false, false, NONE),
    key("1", "1", false, false, NONE),
    key("r", "r", false, false, NONE),
    key("R", "R", false, true, NONE),
    key("[", "[", false, false, NONE),
    key("]", "]", false, false, NONE),
    key("Space", " ", false, false, NONE),
    key("Esc", "Esc", false, false, NONE),
    // Return presses a dialog's default button; Export's asks for a file name.
    key("Return", "Return", false, false, SAVE_FILE),
    // Menus.
    menu("File > New Project", "File/New Project (Ctrl+N)", NONE),
    menu(
        "File > Open Project",
        "File/Open Project\u{2026} (Ctrl+Shift+O)",
        OPEN_FILE,
    ),
    menu(
        "File > Save Project",
        "File/Save Project (Ctrl+S)",
        SAVE_FILE,
    ),
    menu(
        "File > Save Project As",
        "File/Save Project As\u{2026} (Ctrl+Shift+S)",
        SAVE_FILE,
    ),
    menu(
        "File > Add Images",
        "File/Add Images\u{2026} (Ctrl+O)",
        OPEN_FILES,
    ),
    menu("File > Export", "File/Export\u{2026} (Ctrl+E)", NONE),
    menu("Edit > Undo", "Edit/Undo (Ctrl+Z)", NONE),
    menu("Edit > Redo", "Edit/Redo (Ctrl+Y)", NONE),
    menu("Edit > Retouch Brush", "Edit/Retouch Brush (R)", NONE),
    menu("Edit > Settings", "Edit/Settings\u{2026} (Ctrl+,)", NONE),
    menu("View > Fit", "View/Fit to Window (0)", NONE),
    menu("View > Actual Size", "View/Actual Size, 100 % (1)", NONE),
    menu("View > Zoom In", "View/Zoom In (+)", NONE),
    menu("View > Zoom Out", "View/Zoom Out (-)", NONE),
    menu("View > Show Aligned", "View/Show Aligned", NONE),
    menu("Help > Manual", "Help/Manual (F1)", NONE),
    menu(
        "Help > Keyboard Shortcuts",
        "Help/Keyboard Shortcuts (?)",
        NONE,
    ),
    menu("Help > About", "Help/About z-stackr", NONE),
];

/// What one cell compares before and after the action.
#[derive(Debug, Clone, PartialEq)]
#[expect(clippy::struct_excessive_bools, reason = "one field per observed flag")]
struct Snapshot {
    files: usize,
    results: usize,
    tool_on: bool,
    modals: Vec<&'static str>,
    activities: usize,
    zoom: i32,
    show_aligned: bool,
    brush: i32,
    dirty: bool,
    settings_open: bool,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "zoom and brush size rounded for comparison"
)]
fn snapshot(ui: &Ui) -> Snapshot {
    let app = &ui.app;
    Snapshot {
        files: ui.files(),
        results: ui.results(),
        tool_on: ui.tool_on(),
        modals: ui.modals(),
        activities: app.global::<ActivityModel>().get_activities().row_count(),
        zoom: (app.global::<ViewportModel>().get_zoom() * 1000.0).round() as i32,
        show_aligned: app.global::<AppModel>().get_show_aligned(),
        brush: app.global::<RetouchModel>().get_brush_size().round() as i32,
        dirty: app.global::<ProjectModel>().get_dirty(),
        settings_open: app.global::<AppModel>().get_settings_open(),
    }
}

/// Boot an app and bring it into `state`.
async fn enter(state: State, name: &str) -> Result<Ui, String> {
    let ui = if state == State::JobPaused {
        Ui::boot_with(
            name,
            &format!("{FAST_SETTINGS}relief_show_preview = true\n"),
            true,
        )
    } else {
        Ui::boot(name)
    };
    match state {
        State::Empty => {}
        State::Files => ui.load_frames(3).await?,
        State::Result => ui.with_result().await?,
        State::RetouchOn => {
            ui.with_result().await?;
            ui.click("Retouch").await?;
            if !ui.tool_on() {
                return Err("setup: the Retouch tool did not switch on".into());
            }
        }
        State::JobPaused => {
            ui.load_frames(3).await?;
            ui.click("Start Stacking").await?;
            wait_until("the depth-map preview to pause the job", 20_000, || {
                ui.app.global::<AppModel>().get_relief_preview_open()
            })
            .await?;
        }
        State::SettingsOpen => {
            ui.with_result().await?;
            ui.click("Settings").await?;
        }
        State::ShortcutsOpen => {
            ui.with_result().await?;
            ui.key_with("?", false, true).await;
        }
        State::ConfirmOpen => {
            ui.load_frames(3).await?;
            ui.click("Clear all files").await?;
        }
        State::ExportOpen => {
            ui.with_result().await?;
            ui.click("Export\u{2026}").await?;
        }
    }
    if let Some(modal) = state.modal()
        && ui.modals() != [modal]
    {
        return Err(format!(
            "setup: expected only {modal} open, got {:?}",
            ui.modals()
        ));
    }
    let _ = pickers::take_requests();
    Ok(ui)
}

/// Perform `action`. A control that is not on screen in this state is skipped
/// (`Ok(false)`): there is nothing a user could press.
async fn perform(ui: &Ui, action: &Action) -> Result<bool, String> {
    match action.via {
        Via::Click => {
            if ui.one(action.target).is_err() {
                return Ok(false);
            }
            ui.click(action.target).await?;
        }
        Via::Key => {
            let text: slint::SharedString = match action.target {
                "F1" => Key::F1.into(),
                "Esc" => Key::Escape.into(),
                "Return" => Key::Return.into(),
                other => other.into(),
            };
            ui.key_with(&text, action.ctrl, action.shift).await;
        }
        Via::Menu => {
            let (menu, item) = action.target.split_once('/').expect("Menu/Item");
            // A checked item carries a tick in its title.
            let ticked = format!("\u{2713} {item}");
            if ui.menu(menu, item).await.is_err() {
                // The menu is open from the failed attempt: close it first.
                ui.esc().await;
                if ui.menu(menu, &ticked).await.is_err() {
                    ui.esc().await;
                    return Err(format!("menu item {item:?} not found"));
                }
            }
        }
    }
    // Let a started job, a decode or a dialog answer arrive.
    sleep(150).await;
    Ok(true)
}

/// Bring the app back to idle and prove the keyboard still works.
async fn recover(ui: &Ui, quit: bool) -> Vec<String> {
    let mut problems = Vec::new();
    if quit && !ui.app.window().is_visible() {
        return problems;
    }
    // A click on a disabled menu item leaves its menu open, holding the keys.
    ui.esc().await;
    for _ in 0..8 {
        let app = ui.app.global::<AppModel>();
        if app.get_relief_preview_open() {
            ui.app.invoke_relief_preview_cancel();
            sleep(50).await;
            continue;
        }
        if ui.modals().is_empty() && !ui.tool_on() {
            break;
        }
        ui.esc().await;
    }
    if !ui.modals().is_empty() {
        problems.push(format!("Esc does not close {:?}", ui.modals()));
    }
    if wait_until("idle", 20_000, || !ui.busy()).await.is_err() {
        problems.push(format!("still busy 20 s later (status {:?})", ui.status()));
    }
    // Toasts close on Esc only when nothing else does; leave them.
    if ui.modals().is_empty() {
        ui.key_with("?", false, true).await;
        if ui.modals() == ["shortcuts"] {
            ui.esc().await;
        } else {
            problems.push(format!(
                "the keyboard no longer reaches the window: ? opened {:?}",
                ui.modals()
            ));
        }
    }
    problems
}

/// Run one cell; every broken expectation is one line of the result.
pub async fn cell(state: State, action: &Action) -> Vec<String> {
    let name = format!("{state:?}-{}", action.name)
        .replace([' ', '>', '/', '?', '+', ',', '[', ']', '='], "_");
    let ui = match enter(state, &name).await {
        Ok(ui) => ui,
        Err(e) => return vec![e],
    };
    let before = snapshot(&ui);
    let mut problems = Vec::new();
    match perform(&ui, action).await {
        Ok(true) => {}
        Ok(false) => return problems,
        Err(e) => {
            problems.push(e);
            return problems;
        }
    }
    let after = snapshot(&ui);
    let requests = pickers::take_requests();

    // Dialogs only where the action asks for one.
    for request in &requests {
        if !action.dialogs.contains(&request.kind) {
            problems.push(format!(
                "opened an unexpected {:?} dialog {:?}",
                request.kind, request.title
            ));
        }
    }
    // At most one modal layer.
    if after.modals.len() > 1 {
        problems.push(format!(
            "{} modal layers open at once: {:?}",
            after.modals.len(),
            after.modals
        ));
    }
    // The tool needs a result.
    if after.tool_on && after.results == 0 {
        problems.push("the Retouch tool is on without a result".into());
    }
    // While a modal layer is up, keys and menus do nothing (Esc and Return
    // belong to the layer itself).
    if let Some(modal) = state.modal()
        && matches!(action.via, Via::Key | Via::Menu)
        && !matches!(action.name, "Esc" | "Return")
    {
        if after != before {
            problems.push(format!(
                "with {modal} open it changed the app: {}",
                diff(&before, &after)
            ));
        }
        if !requests.is_empty() {
            problems.push(format!("with {modal} open it opened a file dialog"));
        }
    }
    // A click while a modal layer is up lands on the layer: at most it closes it.
    if let Some(modal) = state.modal()
        && action.via == Via::Click
    {
        let mut expected = before.clone();
        expected.modals.retain(|m| *m != modal);
        expected.settings_open = after.settings_open;
        if after != before && after != expected {
            problems.push(format!(
                "a click behind {modal} reached the app: {}",
                diff(&before, &after)
            ));
        }
    }
    // While a job runs no second one starts.
    if state == State::JobPaused && after.activities > 1 {
        problems.push(format!("{} jobs at once", after.activities));
    }

    problems.extend(recover(&ui, action.name.contains("Quit")).await);
    problems
}

fn diff(a: &Snapshot, b: &Snapshot) -> String {
    let mut out = Vec::new();
    macro_rules! field {
        ($f:ident) => {
            if a.$f != b.$f {
                out.push(format!("{} {:?} -> {:?}", stringify!($f), a.$f, b.$f));
            }
        };
    }
    field!(files);
    field!(results);
    field!(tool_on);
    field!(modals);
    field!(activities);
    field!(zoom);
    field!(show_aligned);
    field!(brush);
    field!(dirty);
    field!(settings_open);
    out.join(", ")
}

/// The whole matrix (about 520 cells, several minutes in a debug build). It runs
/// when `UI_FLOW_MATRIX=1` is set or a `UI_FLOW_STATE` / `UI_FLOW_ACTION` filter
/// narrows it; CI runs it as its own job.
pub async fn run() -> Outcome {
    let state_filter = std::env::var("UI_FLOW_STATE").ok();
    let action_filter = std::env::var("UI_FLOW_ACTION").ok();
    let wanted = std::env::var("UI_FLOW_MATRIX").is_ok_and(|v| v == "1");
    if !wanted && state_filter.is_none() && action_filter.is_none() {
        eprintln!("SKIPPED: the action matrix runs with UI_FLOW_MATRIX=1");
        return Ok(());
    }
    let mut failures = Vec::new();
    let mut cells = 0;
    for &state in STATES {
        if state_filter
            .as_deref()
            .is_some_and(|f| !format!("{state:?}").contains(f))
        {
            continue;
        }
        for action in ACTIONS {
            if action_filter
                .as_deref()
                .is_some_and(|f| !action.name.contains(f))
            {
                continue;
            }
            cells += 1;
            for problem in cell(state, action).await {
                eprintln!("  matrix {state:?} / {}: {problem}", action.name);
                failures.push(format!("{state:?} / {}: {problem}", action.name));
            }
        }
    }
    eprintln!("matrix: {cells} cells, {} problems", failures.len());
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("\n{}", failures.join("\n")))
    }
}
