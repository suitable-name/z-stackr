//! The harness the UI flow tests drive the real window with: the app wired
//! exactly as `main` wires it, on Slint's headless testing backend, with the
//! config directory moved to a scratch folder and native file dialogs stubbed.

use std::{
    cell::{Cell, RefCell},
    future::poll_fn,
    path::{Path, PathBuf},
    rc::Rc,
    task::{Poll, Waker},
    time::{Duration, Instant},
};

use i_slint_backend_testing::{AccessibleRole, ElementHandle};
use slint::{
    ComponentHandle, SharedString, Timer,
    platform::{Key, PointerEventButton, WindowEvent},
};
use stacker_gui::{pickers, settings};

use crate::{
    App, AppModel, DialogModel, ExplainerModel, ExportModel, FilesModel, ManualModel, RetouchModel,
    SettingsModel, ShortcutsModel,
    gui::{self, Ctx},
    ui_helpers::push_settings_to_ui,
};

/// What one scenario returns: `Err` carries every broken expectation's message.
pub type Outcome = Result<(), String>;

/// Fail the scenario with a message when `cond` does not hold.
macro_rules! check {
    ($cond:expr, $($msg:tt)+) => {
        let holds: bool = $cond;
        if !holds {
            return Err(format!("{} (line {})", format!($($msg)+), line!()));
        }
    };
}
pub(crate) use check;

/// Wait `ms` milliseconds while the event loop keeps running.
pub async fn sleep(ms: u64) {
    let done = Rc::new(Cell::new(false));
    let waker: Rc<RefCell<Option<Waker>>> = Rc::new(RefCell::new(None));
    {
        let done = Rc::clone(&done);
        let waker = Rc::clone(&waker);
        Timer::single_shot(Duration::from_millis(ms), move || {
            done.set(true);
            if let Some(w) = waker.borrow_mut().take() {
                w.wake();
            }
        });
    }
    poll_fn(|cx| {
        if done.get() {
            Poll::Ready(())
        } else {
            *waker.borrow_mut() = Some(cx.waker().clone());
            Poll::Pending
        }
    })
    .await;
}

/// Let queued callbacks and posted worker results run.
pub async fn settle() {
    sleep(30).await;
}

/// Wait until `cond` holds, polling every 20 ms, for at most `timeout_ms`.
pub async fn wait_until(what: &str, timeout_ms: u64, cond: impl Fn() -> bool) -> Outcome {
    let start = Instant::now();
    loop {
        if cond() {
            return Ok(());
        }
        if start.elapsed() > Duration::from_millis(timeout_ms) {
            return Err(format!(
                "timed out after {timeout_ms} ms waiting for: {what}"
            ));
        }
        sleep(20).await;
    }
}

/// A scratch folder that lives as long as the test process.
pub fn scratch_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("z-stackr-ui-flow-tests")
        .join(format!("{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Write `n` small, textured PNG frames into `dir`, each focused on a different
/// band so a stack has something to choose, and return their paths.
pub fn write_frames(dir: &Path, n: usize, w: u32, h: u32) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).expect("frames dir");
    let count = u32::try_from(n).expect("frame count");
    let band_height = h / count;
    (0..count)
        .map(|index| {
            let band = index * band_height;
            let img = image::RgbImage::from_fn(w, h, |col, row| {
                let sharp = row >= band && row < band + band_height;
                let checker = if ((col / 3) + (row / 3)) % 2 == 0 {
                    220u8
                } else {
                    40u8
                };
                let value = if sharp { checker } else { 130 };
                image::Rgb([value, value.saturating_sub(20), value.saturating_add(10)])
            });
            let path = dir.join(format!("frame_{index:02}.png"));
            img.save(&path).expect("write frame");
            path
        })
        .collect()
}

/// Settings that keep a stack of tiny frames fast in a debug build.
pub const FAST_SETTINGS: &str = "alignment_mode = \"none\"\nauto_cull = false\n";

/// The running app of one scenario.
pub struct Ui {
    pub app: App,
    pub ctx: Ctx,
    pub dir: PathBuf,
}

impl Ui {
    /// Start the app as `main` does, with its config in a fresh scratch folder,
    /// alignment and auto-cull off (fast stacks) and the first-use hints already
    /// dismissed. Scenarios about those use [`Self::boot_with`].
    pub fn boot(name: &str) -> Self {
        Self::boot_with(name, FAST_SETTINGS, true)
    }

    /// Start the app with `settings_toml` as its settings file (empty: none, the
    /// first-run defaults); `quiet` marks every first-use hint as seen.
    pub fn boot_with(name: &str, settings_toml: &str, quiet: bool) -> Self {
        pickers::set_headless(true);
        let _ = pickers::take_requests();
        let dir = scratch_dir(name);
        let config = dir.join("config");
        settings::override_config_dir(Some(config.clone()));
        std::fs::create_dir_all(&config).expect("config dir");
        if !settings_toml.is_empty() {
            std::fs::write(config.join("settings.toml"), settings_toml).expect("settings");
        }
        if quiet {
            std::fs::write(
                config.join("prefs.toml"),
                "suppressed_hints = [\"auto-cull\", \"relief-preview\", \"retouch-sources\"]\n",
            )
            .expect("prefs");
        }
        // A scenario starts without the aligned frames of the one before.
        stacker_gui::runner::current().forget_aligned();
        let app = App::new().expect("App::new");
        gui::stack::populate_algorithms(&app);
        let opened = settings::open_default();
        push_settings_to_ui(&app, &opened.persister.snapshot());
        let ctx = Ctx::new(&app, opened.persister);
        settings::SettingsPersister::install_for_this_thread(&ctx.settings);
        gui::wire_all(&app, &ctx);
        app.show().expect("show");
        app.window().dispatch_event(WindowEvent::Resized {
            size: slint::LogicalSize::new(1400.0, 860.0),
        });
        Self { app, ctx, dir }
    }

    // ── Looking things up ───────────────────────────────────────────────────

    /// Every element whose accessible label is `label`.
    pub fn all(&self, label: &str) -> Vec<ElementHandle> {
        ElementHandle::find_by_accessible_label(&self.app, label).collect()
    }

    /// The single visible element labelled `label`.
    pub fn one(&self, label: &str) -> Result<ElementHandle, String> {
        let mut found: Vec<_> = self
            .all(label)
            .into_iter()
            .filter(|e| e.size().width > 0.0 && e.size().height > 0.0)
            .collect();
        // A caption `Text` carries its text as an accessible label too; when a
        // control shares the label, the control is the one meant.
        if found.len() > 1 {
            found.retain(|e| e.accessible_role() != Some(AccessibleRole::Text));
        }
        match found.len() {
            1 => Ok(found.into_iter().next().unwrap()),
            0 => Err(format!("no visible element labelled {label:?}")),
            n => Err(format!("{n} visible elements labelled {label:?}")),
        }
    }

    /// Whether the element labelled `label` reports itself enabled.
    pub fn enabled(&self, label: &str) -> Result<bool, String> {
        Ok(self.one(label)?.accessible_enabled().unwrap_or(true))
    }

    // ── Acting ──────────────────────────────────────────────────────────────

    /// Click the element labelled `label` with the left button, as a user would
    /// (a disabled widget ignores it), then let the app react.
    pub async fn click(&self, label: &str) -> Outcome {
        let element = self.one(label)?;
        element.single_click(PointerEventButton::Left).await;
        settle().await;
        Ok(())
    }

    /// The menu-bar caption `menu`.
    fn menu_caption(&self, menu: &str) -> Result<ElementHandle, String> {
        ElementHandle::find_by_element_id(&self.app, "MenuBarItemBase::label")
            .find(|e| e.accessible_label().is_some_and(|l| l == menu))
            .ok_or_else(|| format!("no menu {menu:?}"))
    }

    /// Open the menu `menu` and return its visible item `item`.
    async fn open_menu_item(&self, menu: &str, item: &str) -> Result<ElementHandle, String> {
        self.menu_caption(menu)?
            .single_click(PointerEventButton::Left)
            .await;
        settle().await;
        self.all(item)
            .into_iter()
            .find(|e| e.size().width > 0.0)
            .ok_or_else(|| format!("no item {item:?} in menu {menu:?}"))
    }

    /// Open the menu `menu` in the menu bar and activate its item `item`.
    pub async fn menu(&self, menu: &str, item: &str) -> Outcome {
        let target = self.open_menu_item(menu, item).await?;
        target.single_click(PointerEventButton::Left).await;
        settle().await;
        Ok(())
    }

    /// Press and release one key, with the given modifiers held.
    pub async fn key_with(&self, text: &str, ctrl: bool, shift: bool) {
        let window = self.app.window();
        let modifiers: Vec<SharedString> = [(ctrl, Key::Control), (shift, Key::Shift)]
            .into_iter()
            .filter(|(on, _)| *on)
            .map(|(_, k)| SharedString::from(k))
            .collect();
        for m in &modifiers {
            window.dispatch_event(WindowEvent::KeyPressed { text: m.clone() });
        }
        let text = SharedString::from(text);
        window.dispatch_event(WindowEvent::KeyPressed { text: text.clone() });
        window.dispatch_event(WindowEvent::KeyReleased { text });
        for m in modifiers.iter().rev() {
            window.dispatch_event(WindowEvent::KeyReleased { text: m.clone() });
        }
        settle().await;
    }

    /// Press and release one plain key.
    pub async fn key(&self, text: &str) {
        self.key_with(text, false, false).await;
    }

    /// Press Escape.
    pub async fn esc(&self) {
        self.key(&SharedString::from(Key::Escape)).await;
    }

    /// Whether a key press still reaches the window's shortcut handler: `?`
    /// must open the shortcut list (it is closed again). Only meaningful with no
    /// modal layer open.
    pub async fn keys_alive(&self) -> bool {
        self.key_with("?", false, true).await;
        let alive = self.modals() == ["shortcuts"];
        if alive {
            self.esc().await;
        }
        alive
    }

    /// Answer the next file dialog with `paths` (empty: cancelled).
    #[expect(clippy::unused_self, reason = "reads as a step of the scenario")]
    pub fn answer_next_dialog(&self, paths: Vec<PathBuf>) {
        pickers::stub_next_answer(paths);
    }

    // ── Reading the state ───────────────────────────────────────────────────

    pub fn files(&self) -> usize {
        self.app
            .global::<FilesModel>()
            .get_loaded_files()
            .row_count()
    }

    pub fn results(&self) -> usize {
        self.app
            .global::<FilesModel>()
            .get_result_files()
            .row_count()
    }

    pub fn busy(&self) -> bool {
        self.app.global::<AppModel>().get_busy()
    }

    pub fn tool_on(&self) -> bool {
        self.app.global::<RetouchModel>().get_tool_on()
    }

    pub fn status(&self) -> String {
        self.app.global::<AppModel>().get_status().to_string()
    }

    pub fn toast(&self) -> Option<String> {
        let d = self.app.global::<DialogModel>();
        d.get_toast_visible()
            .then(|| d.get_toast_text().to_string())
    }

    /// The modal layers open right now, by name.
    pub fn modals(&self) -> Vec<&'static str> {
        let a = self.app.global::<AppModel>();
        let d = self.app.global::<DialogModel>();
        let m = self.app.global::<ManualModel>();
        [
            ("settings", a.get_settings_open()),
            ("confirm", d.get_confirm_open()),
            ("shortcuts", self.app.global::<ShortcutsModel>().get_open()),
            ("export", self.app.global::<ExportModel>().get_open()),
            ("manual", m.get_open()),
            ("about", m.get_about_open()),
            ("explainer", self.app.global::<ExplainerModel>().get_open()),
        ]
        .into_iter()
        .filter_map(|(name, open)| open.then_some(name))
        .collect()
    }

    /// The settings the dialog shows.
    pub fn settings_model(&self) -> SettingsModel<'_> {
        self.app.global::<SettingsModel>()
    }

    pub fn can_undo(&self) -> bool {
        self.app.global::<RetouchModel>().get_can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.app.global::<RetouchModel>().get_can_redo()
    }

    /// Why the brush cannot paint right now ("" when it can).
    pub fn retouch_hint(&self) -> String {
        self.app.global::<RetouchModel>().get_hint().to_string()
    }

    pub fn dirty(&self) -> bool {
        self.app.global::<crate::ProjectModel>().get_dirty()
    }

    /// The project name the title shows.
    pub fn project_name(&self) -> String {
        self.app
            .global::<crate::ProjectModel>()
            .get_name()
            .to_string()
    }

    pub fn confirm_heading(&self) -> String {
        self.app
            .global::<DialogModel>()
            .get_confirm_heading()
            .to_string()
    }

    /// The canvas's pointer surface.
    pub fn canvas(&self) -> Result<ElementHandle, String> {
        ElementHandle::find_by_element_id(&self.app, "ImageCanvas::touch")
            .next()
            .ok_or_else(|| "no canvas".to_owned())
    }

    /// Drag on the canvas from its centre by (`dx`, `dy`) with `button`.
    pub async fn drag_canvas(&self, dx: f32, dy: f32, button: PointerEventButton) -> Outcome {
        let canvas = self.canvas()?;
        let pos = canvas.absolute_position();
        let size = canvas.size();
        let target = slint::LogicalPosition::new(
            pos.x + size.width / 2.0 + dx,
            pos.y + size.height / 2.0 + dy,
        );
        canvas.drag(target, button).await;
        sleep(300).await;
        Ok(())
    }

    /// Ask the window to close, as its title-bar cross does.
    pub async fn request_close(&self) {
        self.app
            .window()
            .dispatch_event(WindowEvent::CloseRequested);
        settle().await;
    }

    // ── Getting to a state ──────────────────────────────────────────────────

    /// Load `n` synthetic frames through Open Files.
    pub async fn load_frames(&self, n: usize) -> Outcome {
        let frames = write_frames(&self.dir.join("frames"), n, 96, 72);
        self.answer_next_dialog(frames);
        self.click("Open Files").await?;
        wait_until("the frames to load", 10_000, || {
            self.files() == n && !self.busy()
        })
        .await
    }

    /// Stack the loaded frames with the algorithm on show and wait for the result.
    pub async fn stack(&self) -> Outcome {
        let before = self.results();
        self.click("Start Stacking").await?;
        wait_until("the stack to finish", 60_000, || {
            self.results() == before + 1 && !self.busy()
        })
        .await
    }

    /// Frames loaded and one stacked result on the canvas.
    pub async fn with_result(&self) -> Outcome {
        self.load_frames(3).await?;
        self.stack().await
    }
}

impl Drop for Ui {
    fn drop(&mut self) {
        let _ = self.app.hide();
        self.ctx.settings.flush();
    }
}

/// Use `Model::row_count` on the model handles above.
use slint::Model as _;
