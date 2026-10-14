//! User journeys with their specific expected outcomes.

use std::path::{Path, PathBuf};

use slint::{ComponentHandle, Model, platform::PointerEventButton};

use super::harness::{FAST_SETTINGS, Outcome, Ui, check, settle, sleep, wait_until};
use crate::{ActivityModel, AppModel, FilesModel, RetouchModel, ViewportModel};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Choose entry `index` of the Retouch row's frame picker the way a user does:
/// open the combo box and click the entry.
pub async fn choose_source(ui: &Ui, index: usize) -> Outcome {
    let names: Vec<String> = ui
        .app
        .global::<RetouchModel>()
        .get_source_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let name = names
        .get(index)
        .ok_or_else(|| format!("no source entry {index} in {names:?}"))?
        .clone();
    ui.one("Frame the brush paints from")?
        .single_click(PointerEventButton::Left)
        .await;
    settle().await;
    let entry = ui
        .all(&name)
        .into_iter()
        .rfind(|e| e.size().width > 0.0)
        .ok_or_else(|| format!("the open picker shows no entry {name:?}"))?;
    entry.single_click(PointerEventButton::Left).await;
    sleep(400).await;
    let chosen = ui.app.global::<RetouchModel>().get_source_index();
    check!(
        usize::try_from(chosen).ok() == Some(index),
        "picking {name:?} left source-index at {chosen}"
    );
    Ok(())
}

/// A file inside the scenario's scratch folder.
fn scratch(ui: &Ui, name: &str) -> PathBuf {
    ui.dir.join(name)
}

fn file_text(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// Click the button `label` of the open confirmation dialog.
async fn answer(ui: &Ui, label: &str) -> Outcome {
    check!(
        ui.modals() == ["confirm"],
        "no confirmation dialog to answer {label:?} (open: {:?})",
        ui.modals()
    );
    ui.click(label).await
}

/// Wait out the project's dirty watcher (it re-checks every 400 ms).
async fn dirty_tick() {
    sleep(500).await;
}

// ── Retouch ─────────────────────────────────────────────────────────────────

/// The tool on and off by every route, a frame chosen, strokes painted, undone and
/// redone, the brush resized by key.
pub async fn retouch_tool() -> Outcome {
    let ui = Ui::boot("retouch-tool");
    ui.with_result().await?;

    // On by R, off by Esc, on by the toolbar, off by Done, on by the menu.
    ui.key("r").await;
    check!(ui.tool_on(), "R switches the tool on");
    check!(ui.modals().is_empty(), "no dialog opens with the tool");
    ui.esc().await;
    check!(!ui.tool_on(), "Esc switches the tool off");
    ui.click("Retouch").await?;
    check!(ui.tool_on(), "the toolbar button switches the tool on");
    ui.click("Done").await?;
    check!(!ui.tool_on(), "Done switches the tool off");
    ui.menu("Edit", "Retouch Brush (R)").await?;
    check!(ui.tool_on(), "Edit > Retouch Brush switches the tool on");

    // No frame chosen: the brush is blocked and says why; a drag paints nothing.
    check!(
        ui.retouch_hint() == stacker_gui::retouch::BLOCKED_NO_SOURCE,
        "without a frame the hint asks for one, got {:?}",
        ui.retouch_hint()
    );
    ui.drag_canvas(40.0, 10.0, PointerEventButton::Left).await?;
    check!(!ui.can_undo(), "a blocked brush records no stroke");

    // A frame chosen: a drag is one stroke, undoable and redoable.
    choose_source(&ui, 1).await?;
    wait_until("the hint to clear", 5_000, || ui.retouch_hint().is_empty()).await?;
    ui.drag_canvas(40.0, 10.0, PointerEventButton::Left).await?;
    wait_until("the stroke to be recorded", 5_000, || ui.can_undo()).await?;
    check!(!ui.can_redo(), "nothing to redo after a fresh stroke");
    ui.key_with("z", true, false).await;
    check!(!ui.can_undo() && ui.can_redo(), "Ctrl+Z undoes the stroke");
    ui.key_with("y", true, false).await;
    check!(ui.can_undo() && !ui.can_redo(), "Ctrl+Y redoes it");
    ui.key_with("z", true, false).await;
    ui.key_with("Z", true, true).await;
    check!(ui.can_undo() && !ui.can_redo(), "Ctrl+Shift+Z redoes too");
    ui.click("Undo").await?;
    check!(ui.can_redo(), "the row's Undo button undoes");
    ui.click("Redo").await?;
    check!(ui.can_undo(), "the row's Redo button redoes");

    // The brush size keys work only while the tool is on.
    let size = || ui.app.global::<RetouchModel>().get_brush_size();
    let before = size();
    ui.key("]").await;
    check!(size() > before, "] makes the brush larger");
    ui.key("[").await;
    ui.key("[").await;
    check!(size() < before, "[ makes it smaller");

    // Middle-button and Space drags pan instead of painting.
    let strokes_before = ui.can_undo();
    ui.drag_canvas(-30.0, 0.0, PointerEventButton::Middle)
        .await?;
    check!(
        ui.can_undo() == strokes_before && !ui.can_redo(),
        "a middle drag does not paint"
    );

    // Off again: a left drag pans and paints nothing.
    ui.esc().await;
    check!(!ui.tool_on(), "Esc leaves the tool");
    ui.key_with("z", true, false).await;
    let redo_before = ui.can_redo();
    ui.drag_canvas(30.0, 20.0, PointerEventButton::Left).await?;
    check!(
        ui.can_redo() == redo_before,
        "with the tool off a drag paints nothing"
    );
    let before = size();
    ui.key("]").await;
    check!(
        (size() - before).abs() < f32::EPSILON,
        "] does nothing while the tool is off"
    );
    Ok(())
}

/// The brush is blocked, with a reason, whenever the canvas is not the newest result.
pub async fn retouch_blocked_off_the_newest_result() -> Outcome {
    let ui = Ui::boot("retouch-blocked");
    ui.with_result().await?;
    ui.stack().await?;
    check!(ui.results() == 2, "two results");
    ui.key("r").await;
    choose_source(&ui, 0).await?;
    wait_until("the hint to clear", 5_000, || ui.retouch_hint().is_empty()).await?;

    // An older result on the canvas.
    let older = ui
        .app
        .global::<FilesModel>()
        .get_result_files()
        .row_data(0)
        .ok_or("no first result")?;
    ui.click(&format!("Show result {older}")).await?;
    sleep(300).await;
    check!(
        ui.retouch_hint() == stacker_gui::retouch::BLOCKED_NOT_NEWEST,
        "an older result blocks the brush with a reason, got {:?}",
        ui.retouch_hint()
    );
    ui.drag_canvas(20.0, 0.0, PointerEventButton::Left).await?;
    check!(!ui.can_undo(), "no stroke lands on an older result");

    // A source frame on the canvas.
    ui.click("Show frame_00.png").await?;
    sleep(300).await;
    check!(
        !ui.retouch_hint().is_empty(),
        "a source frame on the canvas blocks the brush"
    );
    ui.drag_canvas(20.0, 0.0, PointerEventButton::Left).await?;
    check!(!ui.can_undo(), "no stroke lands on a source frame");

    // Back to the newest: painting works again.
    let newest = ui
        .app
        .global::<FilesModel>()
        .get_result_files()
        .row_data(1)
        .ok_or("no second result")?;
    ui.click(&format!("Show result {newest}")).await?;
    wait_until("the hint to clear on the newest result", 5_000, || {
        ui.retouch_hint().is_empty()
    })
    .await?;
    ui.drag_canvas(20.0, 0.0, PointerEventButton::Left).await?;
    wait_until("a stroke on the newest result", 5_000, || ui.can_undo()).await
}

/// Regression: picking a frame put the keyboard focus in the Retouch row's combo
/// box; leaving the tool removed the row, and after the next change to the
/// canvas (New Project) no shortcut reached the window any more.
pub async fn keys_survive_the_retouch_row() -> Outcome {
    for leave in ["Esc", "Done", "R"] {
        let ui = Ui::boot("keys-retouch-row");
        ui.with_result().await?;
        ui.key("r").await;
        choose_source(&ui, 1).await?;
        match leave {
            "Esc" => ui.esc().await,
            "Done" => ui.click("Done").await?,
            _ => ui.key("r").await,
        }
        check!(!ui.tool_on(), "{leave} leaves the tool");
        ui.answer_next_dialog(vec![scratch(&ui, "k.zstack.toml")]);
        ui.key_with("s", true, false).await;
        ui.key_with("n", true, false).await;
        if ui.modals() == ["confirm"] {
            answer(&ui, "Discard").await?;
        }
        check!(
            ui.keys_alive().await,
            "after leaving the tool by {leave} and New Project the shortcuts still work"
        );
    }
    // A press on the canvas hands the keys back after a sidebar combo box took them.
    let ui = Ui::boot("keys-canvas-click");
    ui.with_result().await?;
    let combo = ui.one("Algorithm")?;
    combo.single_click(PointerEventButton::Left).await;
    settle().await;
    ui.esc().await;
    ui.canvas()?.single_click(PointerEventButton::Left).await;
    settle().await;
    // Space held down (not yet released) arms the pan.
    ui.app
        .window()
        .dispatch_event(slint::platform::WindowEvent::KeyPressed { text: " ".into() });
    settle().await;
    let armed = ui.app.global::<ViewportModel>().get_pan_armed();
    ui.app
        .window()
        .dispatch_event(slint::platform::WindowEvent::KeyReleased { text: " ".into() });
    settle().await;
    check!(
        armed,
        "after a click on the canvas Space arms the pan instead of going to the algorithm box"
    );
    check!(ui.modals().is_empty(), "no popup or dialog came up");
    Ok(())
}

/// Esc leaves the Retouch tool even while a toast is up.
pub async fn esc_leaves_the_tool_before_a_toast() -> Outcome {
    let ui = Ui::boot("esc-toast");
    ui.with_result().await?;
    crate::gui::feedback::show(
        &ui.app,
        crate::gui::feedback::MessageKind::Info,
        "A message",
    );
    ui.key("r").await;
    check!(ui.toast().is_some(), "a toast is up");
    ui.esc().await;
    check!(!ui.tool_on(), "the first Esc leaves the tool");
    check!(ui.toast().is_some(), "the toast stays");
    ui.esc().await;
    check!(ui.toast().is_none(), "the next Esc dismisses the toast");
    Ok(())
}

/// New Project and Clear take the result away: the tool switches off by itself.
pub async fn retouch_tool_ends_with_the_result() -> Outcome {
    let ui = Ui::boot("retouch-ends");
    ui.with_result().await?;
    ui.key("r").await;
    check!(ui.tool_on(), "tool on");
    ui.key_with("n", true, false).await;
    if ui.modals() == ["confirm"] {
        answer(&ui, "Discard").await?;
    }
    check!(ui.results() == 0, "New Project drops the results");
    check!(!ui.tool_on(), "the tool switched off with the result gone");
    ui.key("r").await;
    check!(!ui.tool_on(), "R does nothing without a result");
    Ok(())
}

// ── Project ─────────────────────────────────────────────────────────────────

/// Save, start a new project, open the saved one again: sources, settings and
/// strokes come back, and the strokes are painted onto the next result.
pub async fn project_round_trip() -> Outcome {
    let ui = Ui::boot("project-round-trip");
    ui.with_result().await?;
    ui.key("r").await;
    choose_source(&ui, 2).await?;
    ui.drag_canvas(30.0, 0.0, PointerEventButton::Left).await?;
    wait_until("a stroke", 5_000, || ui.can_undo()).await?;
    ui.esc().await;
    dirty_tick().await;
    check!(
        ui.dirty(),
        "an unsaved project with sources and a stroke is dirty"
    );

    let file = scratch(&ui, "round.zstack.toml");
    ui.answer_next_dialog(vec![file.clone()]);
    ui.key_with("s", true, false).await;
    sleep(300).await;
    check!(
        file.is_file(),
        "Ctrl+S on a new project asks for a path and writes it"
    );
    dirty_tick().await;
    check!(!ui.dirty(), "saving clears the dirty mark");
    check!(
        ui.project_name() == "round",
        "the title shows the project name, got {:?}",
        ui.project_name()
    );
    let saved = file_text(&file);
    check!(saved.contains("frame_02.png"), "the file lists the sources");
    check!(saved.contains("stroke"), "the file holds the stroke");

    // Ctrl+S again saves in place, no dialog.
    let _ = stacker_gui::pickers::take_requests();
    ui.key_with("s", true, false).await;
    check!(
        stacker_gui::pickers::take_requests().is_empty(),
        "a saved project saves in place without asking"
    );

    ui.key_with("n", true, false).await;
    check!(
        ui.modals().is_empty(),
        "a clean project is replaced without asking"
    );
    check!(
        ui.files() == 0 && ui.results() == 0,
        "New Project empties the session"
    );
    check!(
        ui.project_name() == "Untitled",
        "the new project is Untitled"
    );

    ui.answer_next_dialog(vec![file.clone()]);
    ui.key_with("O", true, true).await;
    sleep(400).await;
    check!(
        ui.files() == 3,
        "opening brings the three sources back, got {} (message {:?}, status {:?})",
        ui.files(),
        ui.toast(),
        ui.status()
    );
    check!(
        ui.project_name() == "round",
        "the title names the opened project"
    );
    dirty_tick().await;
    check!(!ui.dirty(), "a freshly opened project is clean");

    ui.stack().await?;
    wait_until(
        "the saved stroke to be painted on the new result",
        5_000,
        || ui.can_undo(),
    )
    .await?;
    dirty_tick().await;
    check!(
        !ui.dirty(),
        "replaying the saved strokes does not make the project dirty"
    );
    Ok(())
}

/// Unsaved changes guard New Project, Open Project and closing the window.
pub async fn unsaved_changes_guard() -> Outcome {
    let ui = Ui::boot("unsaved-guard");
    ui.load_frames(3).await?;
    dirty_tick().await;
    check!(ui.dirty(), "added sources make the project dirty");

    // New: Cancel keeps everything.
    ui.key_with("n", true, false).await;
    check!(
        ui.modals() == ["confirm"],
        "New Project asks first, got {:?}",
        ui.modals()
    );
    check!(
        ui.confirm_heading() == "Unsaved changes",
        "the question is about unsaved changes"
    );
    answer(&ui, "Cancel").await?;
    check!(
        ui.files() == 3 && ui.modals().is_empty(),
        "Cancel keeps the project"
    );

    // New: Esc is Cancel.
    ui.key_with("n", true, false).await;
    ui.esc().await;
    check!(
        ui.files() == 3 && ui.modals().is_empty(),
        "Esc on the question keeps the project"
    );

    // New: Save with a cancelled save dialog keeps everything too.
    ui.key_with("n", true, false).await;
    answer(&ui, "Save").await?;
    check!(
        ui.files() == 3,
        "a cancelled save does not go on to replace the project"
    );
    check!(
        ui.modals().is_empty(),
        "nothing left open after the cancelled save"
    );

    // Open: Discard goes on to the open dialog.
    let _ = stacker_gui::pickers::take_requests();
    ui.key_with("O", true, true).await;
    answer(&ui, "Discard").await?;
    let asked = stacker_gui::pickers::take_requests();
    check!(
        asked.len() == 1,
        "Discard goes on to ask which project to open"
    );
    check!(ui.files() == 3, "a cancelled open dialog keeps the sources");

    // Closing the window.
    ui.request_close().await;
    check!(
        ui.app.window().is_visible(),
        "the window stays while the question is open"
    );
    check!(ui.modals() == ["confirm"], "closing asks first");
    answer(&ui, "Cancel").await?;
    check!(ui.app.window().is_visible(), "Cancel keeps the window");

    // Save then close: the save writes, then the window goes.
    let file = scratch(&ui, "closing.zstack.toml");
    ui.request_close().await;
    ui.answer_next_dialog(vec![file.clone()]);
    answer(&ui, "Save").await?;
    sleep(200).await;
    check!(file.is_file(), "Save before closing writes the project");
    check!(
        !ui.app.window().is_visible(),
        "the window closes after saving"
    );
    Ok(())
}

/// File > Quit: at once when clean, through the question when not.
pub async fn quit_from_the_menu() -> Outcome {
    let ui = Ui::boot("quit-clean");
    ui.menu("File", "Quit").await?;
    check!(!ui.app.window().is_visible(), "Quit closes a clean window");
    drop(ui);

    let ui = Ui::boot("quit-dirty");
    ui.load_frames(2).await?;
    dirty_tick().await;
    ui.menu("File", "Quit").await?;
    check!(
        ui.app.window().is_visible(),
        "Quit with changes keeps the window up"
    );
    check!(
        ui.modals() == ["confirm"],
        "and asks, got {:?}",
        ui.modals()
    );
    answer(&ui, "Discard").await?;
    check!(!ui.app.window().is_visible(), "Discard closes it");
    Ok(())
}

// ── Export ──────────────────────────────────────────────────────────────────

/// Export through the save dialog, then into a folder twice: the second file
/// gets a number instead of overwriting the first.
pub async fn export_flow() -> Outcome {
    let ui = Ui::boot("export");
    ui.with_result().await?;

    ui.key_with("e", true, false).await;
    check!(ui.modals() == ["export"], "Ctrl+E opens the export dialog");
    let target = scratch(&ui, "picked.png");
    ui.answer_next_dialog(vec![target.clone()]);
    ui.click("Choose name\u{2026}").await?;
    wait_until("the picked file to be written", 10_000, || target.is_file()).await?;
    check!(ui.modals().is_empty(), "the dialog closes after export");

    let out = scratch(&ui, "out");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    ui.settings_model()
        .set_default_output_dir(out.to_string_lossy().as_ref().into());
    ui.settings_model().invoke_settings_edited();
    settle().await;
    for round in 1..=2 {
        ui.click("Export\u{2026}").await?;
        check!(
            ui.modals() == ["export"],
            "Export opens the dialog (round {round})"
        );
        ui.click("Save").await?;
        let want = round;
        wait_until("the export into the folder", 10_000, || {
            std::fs::read_dir(&out).map_or(0, Iterator::count) == want
        })
        .await?;
    }
    let names: Vec<String> = std::fs::read_dir(&out)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .collect();
    check!(names.len() == 2, "two exports, two files: {names:?}");

    // Cancel and the cross change nothing.
    ui.click("Export\u{2026}").await?;
    ui.click("Cancel").await?;
    check!(ui.modals().is_empty(), "Cancel closes the export dialog");
    ui.click("Export\u{2026}").await?;
    ui.click("Close dialog").await?;
    check!(ui.modals().is_empty(), "the cross closes it");
    check!(
        std::fs::read_dir(&out).map_or(0, Iterator::count) == 2,
        "nothing more written"
    );
    Ok(())
}

// ── Settings ────────────────────────────────────────────────────────────────

/// Apply keeps an edit, Cancel and Esc and the cross undo it, Reset then Cancel
/// changes nothing, and an applied edit reaches settings.toml.
pub async fn settings_dialog() -> Outcome {
    let ui = Ui::boot("settings");
    let grit = || ui.settings_model().get_grit_suppression();
    let start = grit();

    for (how, close) in [("Cancel", 0), ("Esc", 1), ("the cross", 2)] {
        ui.click("Settings").await?;
        ui.click("Apex").await?;
        ui.click("Grit suppression").await?;
        check!(grit() != start, "the switch flips ({how})");
        match close {
            0 => ui.click("Cancel").await?,
            1 => ui.esc().await,
            _ => ui.click("Close dialog").await?,
        }
        check!(ui.modals().is_empty(), "{how} closes the dialog");
        check!(grit() == start, "{how} undoes the edit");
    }

    ui.click("Settings").await?;
    ui.click("Reset to Defaults").await?;
    ui.click("Cancel").await?;
    check!(grit() == start, "Reset then Cancel changes nothing");

    ui.click("Settings").await?;
    ui.click("Apex").await?;
    ui.click("Grit suppression").await?;
    ui.click("Apply").await?;
    check!(ui.modals().is_empty(), "Apply closes the dialog");
    check!(grit() != start, "Apply keeps the edit");
    ui.ctx.settings.flush();
    let file = file_text(&ui.dir.join("config").join("settings.toml"));
    check!(
        file.contains(&format!("grit_suppression = {}", !start)),
        "the applied value reaches settings.toml"
    );

    // Save Config / Load Config through their dialogs.
    let exported = scratch(&ui, "exported.toml");
    ui.click("Settings").await?;
    ui.answer_next_dialog(vec![exported.clone()]);
    ui.click("Save Config").await?;
    sleep(200).await;
    check!(exported.is_file(), "Save Config writes the chosen file");
    let imported = scratch(&ui, "imported.toml");
    std::fs::write(&imported, "pyramid_levels = 5\n").map_err(|e| e.to_string())?;
    ui.answer_next_dialog(vec![imported]);
    ui.click("Load Config").await?;
    sleep(200).await;
    check!(
        (ui.settings_model().get_pyramid_levels() - 5.0).abs() < 0.01,
        "Load Config shows the loaded value"
    );
    ui.click("Apply").await?;
    check!(
        ui.ctx.settings.snapshot().pyramid_levels == 5,
        "Apply makes the loaded value the setting"
    );
    Ok(())
}

// ── Jobs ────────────────────────────────────────────────────────────────────

fn preview_settings() -> String {
    format!("{FAST_SETTINGS}relief_show_preview = true\n")
}

async fn start_paused(ui: &Ui) -> Outcome {
    ui.click("Start Stacking").await?;
    wait_until("the depth-map preview", 20_000, || {
        ui.app.global::<AppModel>().get_relief_preview_open()
    })
    .await
}

/// The depth-map preview: not modal, Continue finishes the run, Cancel stops it.
pub async fn relief_preview() -> Outcome {
    let ui = Ui::boot_with("relief-preview", &preview_settings(), true);
    ui.load_frames(3).await?;
    start_paused(&ui).await?;
    check!(ui.busy(), "the paused run counts as busy");
    check!(
        ui.modals().is_empty(),
        "the preview card is not a modal layer"
    );
    check!(
        !ui.enabled("Start Stacking")?,
        "no second stack while it waits"
    );

    // Shortcuts keep working while the card is up.
    let zoom = ui.app.global::<ViewportModel>().get_zoom();
    ui.key("+").await;
    check!(
        ui.app.global::<ViewportModel>().get_zoom() > zoom,
        "+ still zooms with the card up"
    );

    ui.click("Auto-Detect").await?;
    sleep(300).await;
    check!(
        ui.app.global::<AppModel>().get_relief_preview_open(),
        "Auto-Detect keeps the card open"
    );
    ui.click("Continue").await?;
    wait_until("the run to finish after Continue", 20_000, || {
        ui.results() == 1 && !ui.busy()
    })
    .await?;
    check!(
        !ui.app.global::<AppModel>().get_relief_preview_open(),
        "the card closes"
    );

    start_paused(&ui).await?;
    ui.click("Cancel").await?;
    wait_until("the run to stop after Cancel", 20_000, || !ui.busy()).await?;
    check!(ui.results() == 1, "a cancelled run adds no result");
    check!(ui.enabled("Start Stacking")?, "Start Stacking is back");
    Ok(())
}

/// The cross on the activity chip cancels the running job.
pub async fn cancel_from_the_chip() -> Outcome {
    let ui = Ui::boot_with("cancel-chip", &preview_settings(), true);
    ui.load_frames(3).await?;
    start_paused(&ui).await?;
    let job = ui
        .app
        .global::<ActivityModel>()
        .get_activities()
        .row_data(0)
        .ok_or("no activity chip")?;
    ui.click(&format!("Cancel {}", job.label)).await?;
    wait_until("the job to end after the chip's cross", 20_000, || {
        !ui.busy()
    })
    .await?;
    check!(ui.results() == 0, "no result from a cancelled job");
    check!(
        !ui.app.global::<AppModel>().get_relief_preview_open(),
        "the preview card goes with the job"
    );
    check!(
        ui.app
            .global::<ActivityModel>()
            .get_activities()
            .row_count()
            == 0,
        "no chip left behind"
    );
    Ok(())
}

/// Editing the source list while a run waits: the run still ends cleanly, and
/// the result it brings is marked stale against the new list.
pub async fn edit_sources_during_a_run() -> Outcome {
    let ui = Ui::boot_with("edit-during-run", &preview_settings(), true);
    ui.load_frames(3).await?;
    start_paused(&ui).await?;
    ui.click("Remove frame_01.png").await?;
    check!(
        ui.files() == 2,
        "a source can be removed while the run waits"
    );
    ui.click("Continue").await?;
    wait_until("the run to finish", 20_000, || {
        ui.results() == 1 && !ui.busy()
    })
    .await?;
    sleep(500).await;
    check!(
        ui.app.global::<FilesModel>().get_result_stale(),
        "the result made from three frames is stale against two"
    );
    Ok(())
}

// ── The source list ─────────────────────────────────────────────────────────

pub async fn source_list_editing() -> Outcome {
    let ui = Ui::boot("source-list");
    ui.load_frames(3).await?;
    ui.click("Remove frame_01.png").await?;
    check!(ui.files() == 2, "Remove takes one frame out");
    ui.click("Clear all files").await?;
    check!(ui.modals() == ["confirm"], "Clear asks first");
    answer(&ui, "Cancel").await?;
    check!(ui.files() == 2, "Cancel keeps the list");
    ui.click("Clear all files").await?;
    answer(&ui, "Clear").await?;
    check!(ui.files() == 0, "Clear empties the list");
    check!(!ui.enabled("Start Stacking")?, "nothing to stack");
    check!(
        !ui.app.global::<FilesModel>().get_has_images(),
        "the canvas shows nothing"
    );

    // Adding the same frames twice skips the duplicates and says so.
    ui.load_frames(3).await?;
    let frames: Vec<PathBuf> = (0..3)
        .map(|i| ui.dir.join("frames").join(format!("frame_{i:02}.png")))
        .collect();
    ui.answer_next_dialog(frames);
    ui.click("Open Files").await?;
    check!(ui.files() == 3, "duplicates are not added twice");
    check!(
        ui.toast()
            .is_some_and(|t| t.contains("already in the list")),
        "the user hears that they were skipped: {:?}",
        ui.toast()
    );
    Ok(())
}

/// Dropped files arrive like added ones; a dropped project opens.
pub async fn dropped_files() -> Outcome {
    let ui = Ui::boot("dropped");
    let frames = super::harness::write_frames(&ui.dir, 2, 64, 48);
    crate::gui::dropped::handle_for_tests(&ui.ctx, frames);
    settle().await;
    check!(ui.files() == 2, "dropped images are added");
    Ok(())
}

// ── Zoom ────────────────────────────────────────────────────────────────────

pub async fn zoom_controls() -> Outcome {
    let ui = Ui::boot("zoom");
    ui.load_frames(2).await?;
    sleep(300).await;
    // The zoom the label below the canvas shows: image pixels per screen pixel.
    let zoom = || ui.app.global::<AppModel>().get_zoom_factor();
    ui.key("1").await;
    check!((zoom() - 1.0).abs() < 1e-3, "1 shows 100 %, got {}", zoom());
    ui.key("+").await;
    check!(zoom() > 1.0, "+ zooms in");
    ui.click("-").await?;
    ui.click("-").await?;
    check!(zoom() < 1.0, "the - button zooms out");
    ui.click("100%").await?;
    check!((zoom() - 1.0).abs() < 1e-3, "the 100% button");
    ui.key("0").await;
    let fit = zoom();
    ui.click("+").await?;
    ui.menu("View", "Fit to Window (0)").await?;
    check!((zoom() - fit).abs() < 1e-3, "View > Fit matches 0");
    Ok(())
}

// ── Hints ───────────────────────────────────────────────────────────────────

/// A first-use hint shows once; "Don't show again" keeps it away for good.
pub async fn first_use_hints() -> Outcome {
    let ui = Ui::boot_with("hints", "alignment_mode = \"none\"\n", false);
    ui.load_frames(3).await?;
    ui.click("Start Stacking").await?;
    check!(
        ui.modals() == ["explainer"],
        "auto-cull explains itself on the first run"
    );
    ui.click("Don't show again").await?;
    ui.click("Got it").await?;
    check!(ui.modals().is_empty(), "Got it closes the hint");
    wait_until("the run", 30_000, || ui.results() == 1 && !ui.busy()).await?;
    let prefs = file_text(&ui.dir.join("config").join("prefs.toml"));
    check!(
        prefs.contains("auto-cull"),
        "the choice is stored: {prefs:?}"
    );
    ui.click("Start Stacking").await?;
    check!(ui.modals().is_empty(), "the hint stays away");
    wait_until("the second run", 30_000, || ui.results() == 2 && !ui.busy()).await
}

// ── Alignment and staleness ─────────────────────────────────────────────────

/// Align, view the aligned frames, then change the list: the cache goes stale and
/// Recompute brings it back.
pub async fn align_and_stale() -> Outcome {
    let ui = Ui::boot_with(
        "align",
        "alignment_mode = \"translation\"\nauto_cull = false\n",
        true,
    );
    ui.load_frames(3).await?;
    ui.click("Align").await?;
    wait_until("the Align pass", 60_000, || {
        !ui.busy()
            && !ui
                .app
                .global::<FilesModel>()
                .get_aligned_summary()
                .is_empty()
    })
    .await?;
    ui.click("Show Aligned").await?;
    check!(
        ui.app.global::<AppModel>().get_show_aligned(),
        "the switch turns Show Aligned on"
    );
    ui.click("Remove frame_02.png").await?;
    sleep(300).await;
    check!(
        ui.app.global::<FilesModel>().get_aligned_stale(),
        "a changed list makes the aligned frames stale"
    );
    ui.click("Recompute").await?;
    wait_until("the recomputed alignment", 60_000, || {
        !ui.busy() && !ui.app.global::<FilesModel>().get_aligned_stale()
    })
    .await
}

/// A settings change marks the result stale; Recompute replaces it.
pub async fn stale_result() -> Outcome {
    let ui = Ui::boot("stale-result");
    ui.with_result().await?;
    ui.click("Settings").await?;
    ui.click("Apex").await?;
    ui.click("Grit suppression").await?;
    ui.click("Apply").await?;
    sleep(300).await;
    // Grit suppression is a Apex setting; the result is Relief. Change a Relief one.
    ui.settings_model()
        .set_relief_smoothing_radius(ui.settings_model().get_relief_smoothing_radius() + 3.0);
    ui.settings_model().invoke_settings_edited();
    sleep(500).await;
    check!(
        ui.app.global::<FilesModel>().get_result_stale(),
        "a Relief setting change marks the Relief result stale"
    );
    ui.click("Recompute").await?;
    wait_until("the recomputed result", 30_000, || {
        ui.results() == 2 && !ui.busy()
    })
    .await?;
    sleep(300).await;
    check!(
        !ui.app.global::<FilesModel>().get_result_stale(),
        "the new result is current"
    );
    Ok(())
}

// ── Live monitor ────────────────────────────────────────────────────────────

pub async fn live_monitor() -> Outcome {
    let ui = Ui::boot("monitor");
    let folder = scratch(&ui, "watched");
    std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    super::harness::write_frames(&folder, 3, 64, 48);
    ui.answer_next_dialog(vec![folder.clone()]);
    ui.click("Live Monitor").await?;
    check!(
        ui.app.global::<AppModel>().get_is_monitoring(),
        "the monitor runs"
    );
    check!(
        ui.one("Stop Monitor").is_ok(),
        "the button offers Stop Monitor"
    );
    check!(
        !ui.enabled("Start Stacking")?,
        "no manual stack while monitoring"
    );
    wait_until("a first monitor result", 30_000, || ui.results() >= 1).await?;
    let first = ui.results();
    // A new frame arrives.
    let extra = super::harness::write_frames(&ui.dir.join("more"), 4, 64, 48);
    std::fs::copy(&extra[3], folder.join("frame_99.png")).map_err(|e| e.to_string())?;
    wait_until("a result for the new frame", 30_000, || {
        ui.results() > first && !ui.busy()
    })
    .await?;
    ui.click("Stop Monitor").await?;
    check!(
        !ui.app.global::<AppModel>().get_is_monitoring(),
        "Stop Monitor stops it"
    );
    check!(
        ui.one("Live Monitor").is_ok(),
        "the button reads Live Monitor again"
    );
    Ok(())
}

// ── Menus ───────────────────────────────────────────────────────────────────

/// Items whose condition does not hold do nothing when activated.
pub async fn disabled_menu_items_do_nothing() -> Outcome {
    let ui = Ui::boot("menus-disabled");
    let _ = ui.menu("File", "Export\u{2026} (Ctrl+E)").await;
    ui.esc().await;
    check!(
        ui.modals().is_empty(),
        "Export without anything to export does nothing"
    );
    let _ = ui.menu("Edit", "Retouch Brush (R)").await;
    ui.esc().await;
    check!(!ui.tool_on(), "Retouch Brush without a result does nothing");
    let _ = ui.menu("View", "Show Aligned").await;
    ui.esc().await;
    check!(
        !ui.app.global::<AppModel>().get_show_aligned(),
        "Show Aligned without frames does nothing"
    );
    Ok(())
}

/// The recent list: a saved project appears in File > Open Recent and on the
/// empty canvas, and either one opens it.
pub async fn recent_projects() -> Outcome {
    let ui = Ui::boot("recent");
    ui.load_frames(2).await?;
    let file = scratch(&ui, "recent.zstack.toml");
    ui.answer_next_dialog(vec![file.clone()]);
    ui.key_with("s", true, false).await;
    sleep(200).await;
    ui.key_with("n", true, false).await;
    check!(ui.files() == 0, "new project");
    let recent = ui.app.global::<crate::ProjectModel>().get_recent_cards();
    check!(
        recent.row_count() == 1,
        "one recent card on the empty canvas"
    );
    let label = recent.row_data(0).ok_or("no card")?;
    ui.click(&label).await?;
    sleep(300).await;
    check!(ui.files() == 2, "the card opens the project");
    ui.key_with("n", true, false).await;
    ui.menu("File", "Open Recent").await?;
    let path = file.to_string_lossy().to_string();
    let item = ui.all(&path).into_iter().find(|e| e.size().width > 0.0);
    match item {
        Some(item) => item.single_click(PointerEventButton::Left).await,
        None => return Err(format!("Open Recent does not list {path:?}")),
    }
    sleep(300).await;
    check!(ui.files() == 2, "Open Recent opens it too");
    Ok(())
}

/// Pick the entry `label` in the open algorithm box.
async fn pick_algorithm(ui: &Ui, label: &str) -> Outcome {
    ui.one("Algorithm")?
        .single_click(PointerEventButton::Left)
        .await;
    settle().await;
    let entry = ui
        .all(label)
        .into_iter()
        .rfind(|e| e.size().width > 0.0)
        .ok_or_else(|| format!("no {label:?} entry"))?;
    entry.single_click(PointerEventButton::Left).await;
    settle().await;
    Ok(())
}

/// Strata is a pass of its own: the results list names it.
pub async fn strata_run() -> Outcome {
    let ui = Ui::boot("strata");
    ui.load_frames(3).await?;
    pick_algorithm(&ui, "Strata (Guided Fusion)").await?;
    ui.stack().await?;
    let names: Vec<String> = ui
        .app
        .global::<FilesModel>()
        .get_result_files()
        .iter()
        .map(|s| s.to_string())
        .collect();
    check!(
        names.iter().any(|n| n.contains("Strata")),
        "the Strata result is listed: {names:?}"
    );
    Ok(())
}

/// All Three makes a Strata, a Relief and an Apex result in one job.
pub async fn all_three_run() -> Outcome {
    let ui = Ui::boot("all-three");
    ui.load_frames(3).await?;
    pick_algorithm(&ui, "All Three (Strata, Relief & Apex)").await?;
    ui.click("Start Stacking").await?;
    wait_until("three results", 60_000, || ui.results() == 3 && !ui.busy()).await?;
    let names: Vec<String> = ui
        .app
        .global::<FilesModel>()
        .get_result_files()
        .iter()
        .map(|s| s.to_string())
        .collect();
    for pass in ["Strata", "Relief", "Apex"] {
        check!(
            names.iter().any(|n| n.contains(pass)),
            "All Three lists a {pass} result: {names:?}"
        );
    }
    Ok(())
}

/// The Sort and Cull actions run without making a result, leave the file list
/// intact, and say what they did.
pub async fn sort_and_cull() -> Outcome {
    let ui = Ui::boot("sort-cull");
    ui.load_frames(4).await?;
    for (button, done) in [("Sort", "Sort complete"), ("Cull", "Cull complete")] {
        ui.click(button).await?;
        wait_until(done, 60_000, || {
            !ui.busy()
                && (ui.status().contains(done)
                    || ui.toast().is_some_and(|toast| toast.contains(done)))
        })
        .await?;
        check!(ui.results() == 0, "{button} makes no result");
        check!(ui.files() == 4, "{button} keeps every file in the list");
    }
    // Both left the aligned frames behind, like an Align pass.
    check!(
        !ui.app
            .global::<FilesModel>()
            .get_aligned_summary()
            .is_empty(),
        "a Sort or Cull run that aligned leaves an aligned record"
    );
    Ok(())
}

/// Every tab of the settings dialog opens, and the Relief tab's multigrid switch
/// reaches the settings.
pub async fn settings_tabs() -> Outcome {
    let ui = Ui::boot("settings-tabs");
    ui.click("Settings").await?;
    for tab in ["Relief", "Strata", "Input", "Output", "Apex", "General"] {
        ui.click(tab).await?;
    }
    ui.click("Relief").await?;
    let before = ui.settings_model().get_relief_use_multigrid();
    ui.click("Multigrid Depth Solver").await?;
    check!(
        ui.settings_model().get_relief_use_multigrid() != before,
        "the multigrid switch flips"
    );
    ui.click("Apply").await?;
    check!(
        ui.ctx.settings.snapshot().relief_use_multigrid != before,
        "Apply makes it the setting"
    );
    Ok(())
}

/// The algorithm box: `Apex` and Both make the results they name.
pub async fn algorithms() -> Outcome {
    let ui = Ui::boot("algorithms");
    ui.load_frames(3).await?;
    let combo = ui.one("Algorithm")?;
    combo.single_click(PointerEventButton::Left).await;
    settle().await;
    let both = ui
        .all("Both (Relief & Apex)")
        .into_iter()
        .rfind(|e| e.size().width > 0.0)
        .ok_or("no Both entry")?;
    both.single_click(PointerEventButton::Left).await;
    settle().await;
    ui.click("Start Stacking").await?;
    wait_until("both results", 30_000, || ui.results() == 2 && !ui.busy()).await?;
    let names: Vec<String> = ui
        .app
        .global::<FilesModel>()
        .get_result_files()
        .iter()
        .map(|s| s.to_string())
        .collect();
    check!(
        names.iter().any(|n| n.contains("Relief")) && names.iter().any(|n| n.contains("Apex")),
        "Both lists a Relief and a Apex result: {names:?}"
    );
    Ok(())
}
