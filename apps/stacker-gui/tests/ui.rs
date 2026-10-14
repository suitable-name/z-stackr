//! The z-stackr GUI under the slint-harness: the real window and its Rust wiring on
//! Slint's testing backend, inside a throw-away profile (settings, layout, recent
//! projects, scratch files and the temp folder all live there). Every problem goes
//! into one report instead of stopping the run.
//!
//! Scenarios, in order:
//!
//! 1. `startup`: launch, sweep the window geometry at both sizes, wait for `Ready`.
//! 2. `open files`: a cancelled Add images dialog changes nothing; four synthetic
//!    frames come in through the dialog (one answer per click); a duplicate is
//!    skipped; the zoom buttons and a source row work.
//! 3. `stack each engine`: Relief, Apex, Strata and All Three each produce their
//!    results, driven through the Algorithm box and Start Stacking.
//! 4. `settings dialog`: all six tabs swept at both sizes, Save Config, Load
//!    Config, the output folder picker (cancelled), and every way out.
//! 5. `export dialog`: after a stack, the export dialog is swept and closed three
//!    ways, Save As... writes a file, and the Retouch tool toggles.
//! 6. `help dialogs`: manual, about and keyboard shortcuts, swept and closed.
//! 7. `first-use hint`: the auto-cull explainer shows once and stores "don't show
//!    again".
//! 8. `confirm dialog`: removing one frame, then Clear all with Cancel and with
//!    Clear.
//! 9. `align sort cull`: the three list actions run to completion.
//! 10. `project save and open`: Save Project (Ctrl+S) writes a file, New Project
//!     empties the session, Open Project (Ctrl+Shift+O) brings the sources back.
//! 11. `settings persist`: a switch changed in the dialog reaches `settings.toml`
//!     inside the profile after the debounce.
//! 12. `smoke`: the click-everything pass over the window with frames loaded.
//! 13. `smoke settings dialog`: the same pass with the settings dialog open, once per
//!     tab.
//!
//! File dialogs never open: a dialog a scenario did not queue an answer for is
//! answered "cancelled" and recorded by the harness, and the scenarios that care
//! read that record. In the smoke pass such a dialog is a note in the report (the
//! harness's own rule), not a finding.
//!
//! Real-time work (the stacking threads) runs while the harness pumps the event
//! loop, so the waits use generous real-time timeouts. Teardown waits for the run
//! threads, then joins the viewer decoders.
//!
//! The coverage gate scans `ui/` and expects every global callback and interactive
//! element to be reached or listed in `tests/coverage_allow.toml`. A scenario that
//! drove a global callback through the UI says so with `record_callback`.
//!
//! Run: `cargo test -p z-stackr-gui --features ui-harness --test ui`. The report is
//! `<target>/slint-harness/z-stackr-gui/report.md` (and `report.json`); exit code 0
//! means no findings, 1 findings, 2 an aborted run. `SLINT_HARNESS_FILTER=stack`
//! runs the scenarios whose name contains `stack`.
//!
//! The run is parallel (`Jobs::Auto`, at most four worker processes): it starts with
//! one worker and adds one every 5 s while at least 1.5 cores are idle. The report and
//! the console output stay in scenario order. `SLINT_HARNESS_JOBS=serial` runs
//! everything in one process.

use std::{
    cell::{Cell, RefCell},
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::Duration,
};

use slint::{ComponentHandle, Weak, platform::PointerEventButton};
use slint_harness::{coverage::record_activation, prelude::*};
use stacker_gui::{App, ManualModel, RetouchModel, SettingsModel, harness, settings};

/// Settings that keep a stack of tiny frames fast: no alignment, no auto-cull.
const FAST_SETTINGS: &str = "settings_version = 2\nalignment_mode = \"none\"\nauto_cull = false\n";
/// Alignment off but auto-cull left at its default, so the first-use hint shows.
const HINT_SETTINGS: &str = "settings_version = 2\nalignment_mode = \"none\"\n";
/// Every first-use hint marked as seen.
const QUIET_PREFS: &str =
    "suppressed_hints = [\"auto-cull\", \"relief-preview\", \"retouch-sources\"]\n";
/// The tabs of the settings dialog, in order.
const SETTINGS_TABS: [&str; 6] = ["General", "Apex", "Relief", "Strata", "Input", "Output"];
/// How long a stack run may take (real time).
const STACK_TIMEOUT: Duration = Duration::from_secs(180);
/// How long a small background step may take (real time).
const STEP_TIMEOUT: Duration = Duration::from_secs(15);

/// What the profile holds before a launch.
#[derive(Clone, Copy)]
enum Seed {
    /// Fast settings and every hint suppressed.
    Fast,
    /// Alignment off, auto-cull and the hints at their defaults.
    Hints,
}

/// The window the notice routines read: set by every launch, so a relaunch during
/// the smoke pass keeps the reader current.
type NoticeSlot = Rc<RefCell<Option<Weak<App>>>>;

/// Empties the profile's configuration folder and writes the files `seed` asks
/// for. The harness's data directory is where the app keeps `settings.toml` and
/// every file beside it.
fn seed_profile(seed: Seed) {
    let Some(dir) = slint_harness::hooks::paths::data_dir() else {
        return;
    };
    let _ = fs::create_dir_all(&dir);
    for name in [
        "settings.toml",
        settings::layout::FILE_NAME,
        settings::recent::FILE_NAME,
        settings::prefs::FILE_NAME,
    ] {
        let _ = fs::remove_file(dir.join(name));
    }
    let (settings_toml, quiet) = match seed {
        Seed::Fast => (FAST_SETTINGS, true),
        Seed::Hints => (HINT_SETTINGS, false),
    };
    let _ = fs::write(dir.join("settings.toml"), settings_toml);
    if quiet {
        let _ = fs::write(dir.join(settings::prefs::FILE_NAME), QUIET_PREFS);
    }
}

/// Registers the notice reader and the clear routine (both are per scenario) on
/// `slot`.
fn watch_notice(ctx: &mut Ctx<'_>, slot: &NoticeSlot) {
    let reader = Rc::clone(slot);
    ctx.on_error_notice("App", move || {
        let window = reader.borrow().as_ref()?.upgrade()?;
        harness::notice(&window)
    });
    let clearer = Rc::clone(slot);
    ctx.clear_notice_with(move || {
        if let Some(window) = clearer.borrow().as_ref().and_then(Weak::upgrade) {
            harness::clear_notice(&window);
        }
    });
}

/// Launches the app as `main` builds it and points `slot` at the new window.
fn launch(ctx: &mut Ctx<'_>, slot: &NoticeSlot) -> Result<Attached, Finding> {
    let mut session = None;
    let app = ctx.launch(|| {
        let (window, started) = harness::launch()?;
        session = Some(started);
        Ok::<App, slint::PlatformError>(window)
    })?;
    *slot.borrow_mut() = app.component::<App>().map(ComponentHandle::as_weak);
    Ok(match session {
        Some(session) => app.keep(session),
        None => app,
    })
}

/// Seeds the profile, watches the notice and launches: the start of a scenario.
fn boot(ctx: &mut Ctx<'_>, seed: Seed) -> Result<Attached, Finding> {
    seed_profile(seed);
    let slot = NoticeSlot::default();
    watch_notice(ctx, &slot);
    launch(ctx, &slot)
}

/// The `App` inside a launched window.
fn window(app: &Attached) -> Result<&App, Finding> {
    app.component::<App>()
        .ok_or_else(|| Finding::simple("launch", "the launched window is not an App"))
}

/// The single interactive control labelled `label`. A caption `Text` repeating a
/// control's label carries it too, so only non-text elements count.
fn control(app: &Attached, label: &str) -> Result<ElementHandle, Finding> {
    let view = app.view();
    let mut found: Vec<ElementHandle> = view
        .by_label(label)
        .into_iter()
        .filter(|element| element.accessible_role() != Some(AccessibleRole::Text))
        .collect();
    match found.len() {
        0 => view.find(label),
        1 => Ok(found.remove(0)),
        count => Err(Finding::simple(
            "ambiguous-label",
            format!("{count} controls are labelled {label:?}"),
        )),
    }
}

/// Activates the control labelled `label` as a screen reader would (and tells the
/// coverage gate), then lets the app react.
fn press(app: &Attached, label: &str) -> Result<(), Finding> {
    let element = control(app, label)?;
    record_activation(&element);
    element.invoke_accessible_default_action();
    pump(60);
    Ok(())
}

/// A real pointer press and release on `element` (a combo box popup needs the
/// pointer, not the accessible action). The click is an async task on Slint's
/// event loop that waits on mock time, so the harness pumps until it is done.
fn click_real(ctx: &mut Ctx<'_>, element: &ElementHandle) -> Result<(), Finding> {
    let done = Rc::new(Cell::new(false));
    let finished = Rc::clone(&done);
    let target = element.clone();
    let _task = slint::spawn_local(async move {
        target.single_click(PointerEventButton::Left).await;
        finished.set(true);
    })
    .map_err(|e| Finding::simple("launch", format!("cannot start a click task: {e}")))?;
    ctx.wait_for("the pointer click", Duration::from_secs(5), || done.get());
    pump(60);
    Ok(())
}

/// Picks `label` in the sidebar's Algorithm box.
fn pick_algorithm(ctx: &mut Ctx<'_>, app: &Attached, label: &str) -> Result<(), Finding> {
    let combo = app
        .view()
        .find_role(AccessibleRole::Combobox, "Algorithm")?;
    click_real(ctx, &combo)?;
    let entry = app
        .view()
        .by_label(label)
        .into_iter()
        .rfind(|element| element.size().width > 0.0)
        .ok_or_else(|| Finding::simple("not-found", format!("no {label:?} entry in the box")))?;
    click_real(ctx, &entry)
}

/// Writes `count` small textured PNG frames into `dir`, each sharp in a different
/// band so a stack has something to choose, and returns their paths.
fn write_frames(dir: &Path, count: u32, width: u32, height: u32) -> Vec<PathBuf> {
    let _ = fs::create_dir_all(dir);
    let band_height = height / count;
    (0..count)
        .map(|index| {
            let band = index * band_height;
            let image = image::RgbImage::from_fn(width, height, |col, row| {
                let sharp = row >= band && row < band + band_height;
                let checker = if ((col / 3) + (row / 3)) % 2 == 0 {
                    220_u8
                } else {
                    40_u8
                };
                let value = if sharp { checker } else { 130 };
                image::Rgb([value, value.saturating_sub(20), value.saturating_add(10)])
            });
            let path = dir.join(format!("frame_{index:02}.png"));
            image.save(&path).expect("a frame is written");
            path
        })
        .collect()
}

/// Adds `frames` through the Add images dialog (one queued answer per click, the
/// seam answers with a single path) and waits until the sidebar lists them.
fn add_frames(ctx: &mut Ctx<'_>, app: &Attached, frames: &[PathBuf]) -> Result<(), Finding> {
    let window = window(app)?;
    let want = harness::source_count(window) + frames.len();
    for frame in frames {
        ctx.hooks().queue_pick(frame.clone());
        press(app, "Open Files")?;
    }
    ctx.wait_for("the frames to be listed", STEP_TIMEOUT, || {
        harness::source_count(window) == want
    });
    Ok(())
}

/// Records a finding unless exactly the modal layers `want` are open.
fn expect_modals(ctx: &mut Ctx<'_>, window: &App, want: &[&str], when: &str) {
    let open = harness::open_modals(window);
    ctx.expect(
        open == want,
        "modal",
        format!("open layers {open:?} {when}, wanted {want:?}"),
    );
}

/// Runs one stack with `algorithm` and expects a new result for each name in
/// `passes`.
fn stack_with(
    ctx: &mut Ctx<'_>,
    app: &Attached,
    algorithm: &str,
    passes: &[&str],
) -> Result<(), Finding> {
    let window = window(app)?;
    pick_algorithm(ctx, app, algorithm)?;
    let before = harness::result_count(window);
    press(app, "Start Stacking")?;
    let want = before + passes.len();
    ctx.wait_for(&format!("{algorithm}: results"), STACK_TIMEOUT, || {
        harness::result_count(window) == want && !harness::is_busy(window)
    });
    let names = harness::result_names(window);
    let fresh = names.get(before..).unwrap_or_default();
    for pass in passes {
        ctx.expect(
            fresh.iter().any(|name| name.contains(pass)),
            "stack",
            format!("{algorithm}: no {pass} result among {fresh:?}"),
        );
    }
    Ok(())
}

fn startup(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    ctx.sweep_at_sizes(&app, "startup");
    let window = window(&app)?;
    ctx.wait_for("the status reads Ready", Duration::from_secs(2), || {
        harness::status(window) == "Ready"
    });
    // The empty canvas offers a New project card.
    press(&app, "New project")?;
    record_callback("ProjectModel.new_project");
    let failures = ctx.hooks().take_picker_failures();
    ctx.expect(
        failures.is_empty(),
        "picker",
        format!("unexpected file dialog(s) {failures:?}"),
    );
    ctx.shutdown(app);
    Ok(())
}

fn open_files(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;
    let frames = write_frames(&ctx.scratch_file("open-frames"), 4, 96, 72);

    // A cancelled dialog changes nothing, and a queued cancel is no failure.
    ctx.hooks().queue_cancel();
    press(&app, "Open Files")?;
    pump(30);
    let listed = harness::source_count(window);
    ctx.expect(
        listed == 0,
        "open files",
        format!("{listed} frame(s) listed after a cancelled dialog, wanted 0"),
    );
    let failures = ctx.hooks().take_picker_failures();
    ctx.expect(
        failures.is_empty(),
        "picker",
        format!("picker failures {failures:?}, wanted none"),
    );

    // Dialog with nothing queued: answered "cancelled" and reported back, naming it.
    press(&app, "Open Files")?;
    pump(30);
    let failures = ctx.hooks().take_picker_failures();
    ctx.expect(
        failures.len() == 1 && failures[0].contains("Add images"),
        "picker",
        format!("picker failures {failures:?}, wanted one naming `Add images`"),
    );

    add_frames(ctx, &app, &frames)?;
    let listed = harness::source_count(window);
    ctx.expect(
        listed == frames.len(),
        "open files",
        format!("{listed} frame(s) listed, wanted {}", frames.len()),
    );

    // The same file again is skipped, with an informational toast only.
    ctx.hooks().queue_pick(frames[0].clone());
    press(&app, "Open Files")?;
    pump(30);
    let listed = harness::source_count(window);
    ctx.expect(
        listed == frames.len(),
        "open files",
        format!(
            "{listed} frame(s) listed after a duplicate, wanted {}",
            frames.len()
        ),
    );
    // The informational toast closes by hand (the Esc key and its button share a path).
    press(&app, "Dismiss message")?;
    record_callback("DialogModel.toast_dismissed");

    ctx.sweep_at_sizes(&app, "four frames loaded");

    // A source row and the zoom buttons.
    press(&app, "Show frame_00.png")?;
    for (button, callback) in [
        ("Fit", "ViewportModel.zoom_fit"),
        ("100%", "ViewportModel.zoom_one_to_one"),
        ("-", "ViewportModel.zoom_step"),
        ("+", "ViewportModel.zoom_step"),
    ] {
        press(&app, button)?;
        record_callback(callback);
    }
    let failures = ctx.hooks().take_picker_failures();
    ctx.expect(
        failures.is_empty(),
        "picker",
        format!("unexpected file dialog(s) {failures:?}"),
    );
    ctx.shutdown(app);
    Ok(())
}

fn stack_each_engine(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;
    let frames = write_frames(&ctx.scratch_file("stack-frames"), 3, 96, 72);
    add_frames(ctx, &app, &frames)?;
    stack_with(ctx, &app, "Relief (Depth Map)", &["Relief"])?;
    stack_with(ctx, &app, "Apex (Laplacian)", &["Apex"])?;
    stack_with(ctx, &app, "Strata (Guided Fusion)", &["Strata"])?;
    stack_with(
        ctx,
        &app,
        "All Three (Strata, Relief & Apex)",
        &["Strata", "Relief", "Apex"],
    )?;
    let total = harness::result_count(window);
    ctx.expect(
        total == 6,
        "stack",
        format!("{total} result(s) in the list, wanted 6"),
    );
    ctx.sweep_at_sizes(&app, "six results");
    // A result row puts its image on the canvas.
    if let Some(first) = harness::result_names(window).first() {
        press(&app, &format!("Show result {first}"))?;
    }
    ctx.shutdown(app);
    Ok(())
}

fn settings_dialog(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;

    press(&app, "Settings")?;
    expect_modals(ctx, window, &["settings"], "after Settings");
    for tab in SETTINGS_TABS {
        press(&app, tab)?;
        ctx.sweep_at_sizes(&app, &format!("settings dialog, {tab} tab"));
    }

    // Save Config, then load that file back; the folder picker is cancelled.
    let exported = ctx.scratch_file("exported.toml");
    ctx.hooks().queue_pick(exported.clone());
    press(&app, "Save Config")?;
    // The answer reaches the app through the event loop, so wait for the file
    // rather than pumping a fixed time (a full run needs more than one pass).
    let written = exported.clone();
    ctx.wait_for("Save Config writes the file", STEP_TIMEOUT, move || {
        written.is_file()
    });
    ctx.hooks().queue_pick(exported);
    press(&app, "Load Config")?;
    pump(60);
    press(&app, "Output")?;
    ctx.hooks().queue_cancel();
    press(&app, "Browse\u{2026}")?;
    pump(60);
    press(&app, "Reset to Defaults")?;
    let failures = ctx.hooks().take_picker_failures();
    ctx.expect(
        failures.is_empty(),
        "picker",
        format!("unexpected file dialog(s) {failures:?}"),
    );

    // Every way out: Cancel, the cross and Esc.
    press(&app, "Cancel")?;
    expect_modals(ctx, window, &[], "after Cancel");
    press(&app, "Settings")?;
    press(&app, "Close dialog")?;
    expect_modals(ctx, window, &[], "after the cross");
    press(&app, "Settings")?;
    ctx.press(&app, "Escape");
    expect_modals(ctx, window, &[], "after Esc");
    ctx.shutdown(app);
    Ok(())
}

fn export_dialog(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;
    let frames = write_frames(&ctx.scratch_file("export-frames"), 3, 96, 72);
    add_frames(ctx, &app, &frames)?;
    stack_with(ctx, &app, "Apex (Laplacian)", &["Apex"])?;

    press(&app, "Export\u{2026}")?;
    record_callback("ExportModel.request_open");
    expect_modals(ctx, window, &["export"], "after Export");
    // The dialog's file name line and its placeholder note are bound to these.
    record_callback("ExportModel.preview_name");
    record_callback("ExportModel.unknown_placeholders");
    ctx.sweep_at_sizes(&app, "export dialog");
    press(&app, "Cancel")?;
    expect_modals(ctx, window, &[], "after Cancel");
    press(&app, "Export\u{2026}")?;
    press(&app, "Close dialog")?;
    expect_modals(ctx, window, &[], "after the cross");

    // With no output folder the main button asks for a name in a save dialog and
    // writes the image in the background; so does Save As...
    let chosen = ctx.scratch_file("chosen.png");
    press(&app, "Export\u{2026}")?;
    ctx.hooks().queue_pick(chosen.clone());
    press(&app, "Choose name\u{2026}")?;
    record_callback("ExportModel.save");
    ctx.wait_for("the chosen file", STEP_TIMEOUT, || chosen.is_file());
    expect_modals(ctx, window, &[], "after Choose name");
    let target = ctx.scratch_file("exported.png");
    press(&app, "Export\u{2026}")?;
    ctx.hooks().queue_pick(target.clone());
    press(&app, "Save As\u{2026}")?;
    ctx.wait_for("the exported file", STEP_TIMEOUT, || target.is_file());
    expect_modals(ctx, window, &[], "after Save As");
    let failures = ctx.hooks().take_picker_failures();
    ctx.expect(
        failures.is_empty(),
        "picker",
        format!("unexpected file dialog(s) {failures:?}"),
    );

    // The Retouch tool is offered once a result exists.
    press(&app, "Retouch")?;
    let on = window.global::<RetouchModel>().get_tool_on();
    ctx.expect(on, "retouch", "the Retouch tool did not switch on");
    ctx.sweep_at_sizes(&app, "retouch tool on");
    record_callback("RetouchModel.tool_toggled");
    press(&app, "Done")?;
    let on = window.global::<RetouchModel>().get_tool_on();
    ctx.expect(!on, "retouch", "Done did not switch the Retouch tool off");
    ctx.shutdown(app);
    Ok(())
}

fn help_dialogs(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;

    ctx.press(&app, "F1");
    expect_modals(ctx, window, &["manual"], "after F1");
    ctx.sweep_at_sizes(&app, "manual");
    press(&app, "Close")?;
    expect_modals(ctx, window, &[], "after closing the manual");

    // About has no key and lives in the Help menu: open it through its model.
    window.global::<ManualModel>().set_about_open(true);
    pump(60);
    expect_modals(ctx, window, &["about"], "after About");
    ctx.sweep_at_sizes(&app, "about");
    press(&app, "Close")?;
    expect_modals(ctx, window, &[], "after closing About");

    ctx.press(&app, "?");
    expect_modals(ctx, window, &["shortcuts"], "after ?");
    ctx.sweep_at_sizes(&app, "keyboard shortcuts");
    press(&app, "Close dialog")?;
    expect_modals(ctx, window, &[], "after closing the shortcuts");
    ctx.shutdown(app);
    Ok(())
}

fn first_use_hint(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Hints)?;
    let window = window(&app)?;
    let frames = write_frames(&ctx.scratch_file("hint-frames"), 3, 96, 72);
    add_frames(ctx, &app, &frames)?;
    press(&app, "Start Stacking")?;
    expect_modals(
        ctx,
        window,
        &["explainer"],
        "after the first Start Stacking",
    );
    ctx.sweep_at_sizes(&app, "first-use hint");
    press(&app, "Don't show again")?;
    press(&app, "Got it")?;
    record_callback("ExplainerModel.dismissed");
    expect_modals(ctx, window, &[], "after Got it");
    ctx.wait_for("the stack behind the hint", STACK_TIMEOUT, || {
        harness::result_count(window) == 1 && !harness::is_busy(window)
    });
    let prefs = slint_harness::hooks::paths::data_dir()
        .map(|dir| dir.join(settings::prefs::FILE_NAME))
        .and_then(|path| fs::read_to_string(path).ok())
        .unwrap_or_default();
    ctx.expect(
        prefs.contains("auto-cull"),
        "hint",
        format!("prefs.toml does not store the choice: {prefs:?}"),
    );
    ctx.shutdown(app);
    Ok(())
}

fn confirm_dialog(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;
    let frames = write_frames(&ctx.scratch_file("confirm-frames"), 3, 96, 72);
    add_frames(ctx, &app, &frames)?;

    press(&app, "Remove frame_01.png")?;
    let listed = harness::source_count(window);
    ctx.expect(
        listed == 2,
        "remove",
        format!("{listed} frame(s) listed after Remove, wanted 2"),
    );

    press(&app, "Clear all files")?;
    expect_modals(ctx, window, &["confirm"], "after Clear all");
    ctx.sweep_at_sizes(&app, "confirm dialog");
    press(&app, "Cancel")?;
    record_callback("DialogModel.confirm_cancel");
    let listed = harness::source_count(window);
    ctx.expect(
        listed == 2,
        "clear",
        format!("{listed} frame(s) listed after Cancel, wanted 2"),
    );
    press(&app, "Clear all files")?;
    press(&app, "Clear")?;
    record_callback("DialogModel.confirm_primary");
    let listed = harness::source_count(window);
    ctx.expect(
        listed == 0,
        "clear",
        format!("{listed} frame(s) listed after Clear, wanted 0"),
    );
    ctx.shutdown(app);
    Ok(())
}

fn align_sort_cull(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;
    let frames = write_frames(&ctx.scratch_file("list-frames"), 4, 96, 72);
    add_frames(ctx, &app, &frames)?;

    press(&app, "Align")?;
    ctx.wait_for("the Align pass", STACK_TIMEOUT, || {
        !harness::is_busy(window) && !harness::aligned_summary(window).is_empty()
    });
    for (button, done) in [("Sort", "Sort complete"), ("Cull", "Cull complete")] {
        press(&app, button)?;
        ctx.wait_for(done, STACK_TIMEOUT, || {
            !harness::is_busy(window)
                && (harness::status(window).contains(done)
                    || harness::notice(window).is_some_and(|(_, text)| text.contains(done)))
        });
    }
    let listed = harness::source_count(window);
    let results = harness::result_count(window);
    ctx.expect(
        listed == frames.len() && results == 0,
        "list actions",
        format!(
            "{listed} frame(s) and {results} result(s) after Align, Sort and Cull, wanted {} and 0",
            frames.len()
        ),
    );
    ctx.shutdown(app);
    Ok(())
}

fn project_save_and_open(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;
    let frames = write_frames(&ctx.scratch_file("project-frames"), 2, 96, 72);
    add_frames(ctx, &app, &frames)?;

    // Save Project asks for a path while the project has none.
    let project = ctx.scratch_file("demo.zstack.toml");
    ctx.hooks().queue_pick(project.clone());
    ctx.press(&app, "Ctrl+S");
    ctx.wait_for("the project file", STEP_TIMEOUT, || project.is_file());
    let inside = ctx.profile().contains(&project);
    ctx.expect(
        inside,
        "project",
        format!("{} is outside the profile", project.display()),
    );
    record_callback("ProjectModel.save_project");

    // New Project: the saved project is clean, so no question is asked.
    ctx.press(&app, "Ctrl+N");
    ctx.wait_for("an empty session", STEP_TIMEOUT, || {
        harness::source_count(window) == 0
    });
    record_callback("ProjectModel.new_project");

    // The saved project is the one recent card on the empty canvas.
    let want = frames.len();
    let cards = harness::recent_cards(window);
    ctx.expect(
        cards.len() == 1,
        "recent",
        format!("recent cards {cards:?}, wanted one"),
    );
    if let Some(card) = cards.first() {
        press(&app, card)?;
        ctx.wait_for("the recent card to open the project", STEP_TIMEOUT, || {
            harness::source_count(window) == want
        });
        record_callback("ProjectModel.open_recent");
        ctx.press(&app, "Ctrl+N");
        ctx.wait_for("an empty session again", STEP_TIMEOUT, || {
            harness::source_count(window) == 0
        });
    }

    // Open Project brings the sources back.
    ctx.hooks().queue_pick(project);
    ctx.press(&app, "Ctrl+Shift+O");
    ctx.wait_for("the sources to come back", STEP_TIMEOUT, || {
        harness::source_count(window) == want
    });
    record_callback("ProjectModel.open_project");
    let failures = ctx.hooks().take_picker_failures();
    ctx.expect(
        failures.is_empty(),
        "picker",
        format!("unexpected file dialog(s) {failures:?}"),
    );
    ctx.shutdown(app);
    Ok(())
}

fn settings_persist(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let app = boot(ctx, Seed::Fast)?;
    let window = window(&app)?;
    let path = settings::get_config_path()
        .map_err(|e| Finding::simple("settings", format!("no settings path: {e}")))?;
    let inside = ctx.profile().contains(&path);
    ctx.expect(
        inside,
        "settings",
        format!("{} is outside the profile", path.display()),
    );

    let before = window.global::<SettingsModel>().get_grit_suppression();
    press(&app, "Settings")?;
    press(&app, "Apex")?;
    press(&app, "Grit suppression")?;
    let flipped = window.global::<SettingsModel>().get_grit_suppression();
    ctx.expect(
        flipped != before,
        "settings",
        "the Grit suppression switch did not flip",
    );
    press(&app, "Apply")?;
    record_callback("SettingsModel.settings_edited");
    let wanted = format!("grit_suppression = {flipped}");
    ctx.wait_for("settings.toml after the debounce", STEP_TIMEOUT, || {
        fs::read_to_string(&path).is_ok_and(|text| text.contains(&wanted))
    });
    ctx.shutdown(app);
    Ok(())
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "every scenario has the same signature"
)]
fn smoke(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let slot = NoticeSlot::default();
    watch_notice(ctx, &slot);
    let frames = write_frames(&ctx.scratch_file("smoke-frames"), 3, 96, 72);
    // Denied for this pass: stacking and the folder monitor run for a long time (the
    // stack scenario drives them), and Align, Sort and Cull start passes that outlive
    // the relaunch after the click (their own scenario waits for them). Clear all,
    // Remove and Quit are on the harness's default list. A file dialog a click opens
    // is answered "cancelled" by the picker seam and shows up as a note.
    let denied = [
        DenyRule::label(DenyMatch::Exact, "Align", "starts a pass"),
        DenyRule::label(DenyMatch::Exact, "Sort", "starts a pass"),
        DenyRule::label(DenyMatch::Exact, "Cull", "starts a pass"),
    ];
    let smoke = Smoke::new("frames loaded")
        .deny(["start stacking", "monitor"])
        .deny_rules(&denied);
    let stats = ctx.smoke(smoke, move |ctx| {
        seed_profile(Seed::Fast);
        let app = launch(ctx, &slot)?;
        add_frames(ctx, &app, &frames)?;
        Ok(vec![app])
    });
    ctx.expect(
        stats.activated >= 10,
        "smoke",
        format!(
            "only {} control(s) activated, wanted at least 10",
            stats.activated
        ),
    );
    ctx.expect(
        stats.denied >= 3,
        "smoke",
        format!("only {} control(s) denied, wanted at least 3", stats.denied),
    );
    Ok(())
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "every scenario has the same signature"
)]
fn smoke_settings_dialog(ctx: &mut Ctx<'_>) -> Result<(), Finding> {
    let slot = NoticeSlot::default();
    watch_notice(ctx, &slot);
    // One pass per tab with the dialog open. The dialog's own buttons are fair game:
    // Apply and Cancel close it (the state is rebuilt), Reset to Defaults only edits
    // the dialog's copy, and the pickers behind Save Config, Load Config and Browse
    // are answered "cancelled" by the seam and shown as notes.
    for tab in SETTINGS_TABS {
        let slot = Rc::clone(&slot);
        let smoke = Smoke::new(format!("settings dialog, {tab} tab"));
        let stats = ctx.smoke(smoke, move |ctx| {
            seed_profile(Seed::Fast);
            let app = launch(ctx, &slot)?;
            press(&app, "Settings")?;
            press(&app, tab)?;
            Ok(vec![app])
        });
        ctx.expect(
            stats.activated >= 3,
            "smoke",
            format!(
                "only {} control(s) activated in the {tab} tab, wanted at least 3",
                stats.activated
            ),
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    Harness::new()
        // Most scenarios spend their time waiting on real-time work (stack runs, the
        // settings debounce): extra worker processes start while cores are idle, each
        // with its own profile. `SLINT_HARNESS_JOBS=serial` runs them one by one.
        .jobs(Jobs::Auto)
        .max_jobs(4)
        .sizes([(1400, 860), (1120, 700)])
        .profile(Profile::isolated().guard_paths(|| {
            settings::get_config_path()
                .ok()
                .into_iter()
                .chain([settings::layout::default_log_path()])
                .collect()
        }))
        .coverage(Coverage::scan("ui").allow_file("tests/coverage_allow.toml"))
        .teardown(|| {
            // Every run thread owns scratch files in the profile: wait for them, then
            // join the viewer decoders. The settings writers went with their windows.
            let drained = wait_until(Duration::from_secs(120), || harness::live_workers() == 0);
            if !drained {
                eprintln!("ui: run threads were still alive after 120 s");
            }
            harness::stop_workers();
        })
        .scenario("startup", startup)
        .scenario("open files", open_files)
        .scenario("stack each engine", stack_each_engine)
        .scenario("settings dialog", settings_dialog)
        .scenario("export dialog", export_dialog)
        .scenario("help dialogs", help_dialogs)
        .scenario("first-use hint", first_use_hint)
        .scenario("confirm dialog", confirm_dialog)
        .scenario("align sort cull", align_sort_cull)
        .scenario("project save and open", project_save_and_open)
        .scenario("settings persist", settings_persist)
        .scenario("smoke", smoke)
        .scenario("smoke settings dialog", smoke_settings_dialog)
        .run()
}
