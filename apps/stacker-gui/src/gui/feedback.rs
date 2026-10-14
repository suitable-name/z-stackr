//! User-visible feedback: the status line and the toast.
//!
//! Every outcome the user must learn about goes through [`show`] (or
//! [`show_from_worker`] / [`notify`]), never through a bare `set_status`:
//!
//! * the message lands in `AppModel.status` **and** in the toast;
//! * it is mirrored to `tracing` at the matching level, so the run log and the
//!   screen never disagree;
//! * `Info` and `Success` toasts dismiss themselves after [`AUTO_DISMISS`];
//!   `Warning` and `Error` stay until the user closes them, because they report
//!   something that must be read.
//!
//! There is one toast slot. A newer message replaces an older one, except that
//! a transient message (`Info`, `Success`) never replaces a `Warning` or
//! `Error` that is still on screen: it only updates the status line. A stale
//! auto-dismiss timer can never hide a newer toast: each toast bumps a
//! generation counter ([`ToastState`]) and a timer acts only while its own
//! generation is still the current one.
//!
//! Progress text ("Aligning… frame 3/10") is not feedback in this sense: it
//! keeps using [`show_status`], which touches only the status line.

use std::{cell::Cell, fmt::Write as _, path::Path, time::Duration};

use slint::{ComponentHandle, Timer, Weak};
use stacker_gui::{frames::SkippedFrame, stall_guard::stall_guard};

use super::{Ctx, ctx::on_ui};
use crate::{App, DialogModel};

/// How long an `Info` or `Success` toast stays up.
pub const AUTO_DISMISS: Duration = Duration::from_millis(3500);

/// How many file names a message lists before it says "and N more".
const MAX_LISTED: usize = 5;

/// How serious a message is; it picks the toast colour, the log level and
/// whether the toast dismisses itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    /// Neutral information; the toast dismisses itself.
    Info,
    /// A finished action; the toast dismisses itself.
    Success,
    /// Something went wrong but work continues; the toast stays.
    Warning,
    /// An action failed; the toast stays.
    Error,
}

impl MessageKind {
    /// The name `DialogModel.toast-kind` uses for this kind.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Success => "success",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }

    /// The kind a `DialogModel.toast-kind` value names.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "info" => Some(Self::Info),
            "success" => Some(Self::Success),
            "warning" => Some(Self::Warning),
            "error" => Some(Self::Error),
            _ => None,
        }
    }

    /// `true` for the kinds that stay on screen until dismissed.
    #[must_use]
    pub const fn is_sticky(self) -> bool {
        matches!(self, Self::Warning | Self::Error)
    }

    /// Whether a toast of this kind may replace the one on screen (`shown`, or
    /// `None` for an empty slot). A transient message never pushes a sticky one
    /// off the screen before the user has read it.
    #[must_use]
    pub const fn replaces(self, shown: Option<Self>) -> bool {
        match shown {
            Some(on_screen) if on_screen.is_sticky() => self.is_sticky(),
            _ => true,
        }
    }
}

/// The toast generation counter: the pure half of the auto-dismiss logic.
///
/// [`Self::begin`] is called for every toast shown and returns that toast's
/// token; a delayed dismiss acts only while [`Self::is_current`] still holds for
/// its token. [`Self::invalidate`] cancels every pending dismiss (the user closed
/// the toast by hand).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ToastState {
    generation: u64,
}

impl ToastState {
    /// Register a new toast and return its token.
    pub const fn begin(&mut self) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.generation
    }

    /// `true` while no toast has been shown (or invalidated) since `token`.
    #[must_use]
    pub const fn is_current(self, token: u64) -> bool {
        self.generation == token
    }

    /// Make every outstanding token stale.
    pub const fn invalidate(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }
}

thread_local! {
    /// The toast generation of the UI thread (toasts are only shown there).
    static TOAST: Cell<ToastState> = const { Cell::new(ToastState { generation: 0 }) };
}

/// Run `f` on the UI thread's [`ToastState`].
fn with_state<R>(f: impl FnOnce(&mut ToastState) -> R) -> R {
    TOAST.with(|cell| {
        let mut state = cell.get();
        let result = f(&mut state);
        cell.set(state);
        result
    })
}

/// Wire `DialogModel.toast-dismissed()`: a toast closed by hand must not be
/// "re-closed" by its own pending timer after a newer toast took its place.
pub fn setup(ui: &App, _ctx: &Ctx) {
    ui.global::<DialogModel>().on_toast_dismissed(|| {
        stall_guard("feedback::toast-dismissed", || {
            with_state(ToastState::invalidate);
        });
    });
}

/// Report `text` to the user: status line, toast and log. UI thread only.
pub fn show(ui: &App, kind: MessageKind, text: &str) {
    log(kind, text);
    show_status(ui, text);
    show_toast(ui, kind, text);
}

/// Set the status line only (progress text and other chatter that is not worth
/// a toast and not worth a log line).
pub fn show_status(ui: &App, text: &str) {
    ui.set_status(text.into());
}

/// [`show`] from a worker thread: the message reaches the UI thread through the
/// event loop, in order with the worker's other `on_ui` posts.
pub fn show_from_worker(ui: &Weak<App>, kind: MessageKind, text: String) {
    on_ui(ui, move |app| show(app, kind, &text));
}

/// [`show`] for a callback that holds a [`Ctx`] instead of the window.
pub fn notify(ctx: &Ctx, kind: MessageKind, text: &str) {
    if let Some(app) = ctx.ui.upgrade() {
        show(&app, kind, text);
    }
}

/// Mirror a message to the run log at the matching level.
fn log(kind: MessageKind, text: &str) {
    match kind {
        MessageKind::Info | MessageKind::Success => tracing::info!("{text}"),
        MessageKind::Warning => tracing::warn!("{text}"),
        MessageKind::Error => tracing::error!("{text}"),
    }
}

/// Put the toast up (or leave a sticky one alone, see the module docs) and arm
/// the auto-dismiss for transient kinds.
fn show_toast(ui: &App, kind: MessageKind, text: &str) {
    let dialogs = ui.global::<DialogModel>();
    let shown = dialogs
        .get_toast_visible()
        .then(|| MessageKind::from_name(dialogs.get_toast_kind().as_str()))
        .flatten();
    if !kind.replaces(shown) {
        return;
    }
    dialogs.set_toast_text(text.into());
    dialogs.set_toast_kind(kind.name().into());
    dialogs.set_toast_visible(true);

    let token = with_state(ToastState::begin);
    if kind.is_sticky() {
        return;
    }
    let weak = ui.as_weak();
    Timer::single_shot(AUTO_DISMISS, move || {
        if let Some(app) = weak.upgrade()
            && with_state(|state| state.is_current(token))
        {
            app.global::<DialogModel>().set_toast_visible(false);
        }
    });
}

/// The file name of `path`, or the whole path when it has none.
#[must_use]
pub fn file_label(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// `a.jpg, b.jpg, c.jpg and 4 more`: the file names of `paths`, at most
/// [`MAX_LISTED`] of them.
#[must_use]
pub fn list_files<'a>(paths: impl IntoIterator<Item = &'a Path>) -> String {
    let names: Vec<String> = paths.into_iter().map(file_label).collect();
    let mut text = names
        .iter()
        .take(MAX_LISTED)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > MAX_LISTED {
        let _ = write!(text, " and {} more", names.len() - MAX_LISTED);
    }
    text
}

/// The one warning that names every frame a decode pass left out, with the
/// first decoder message as the likely common cause. `None` when nothing was
/// skipped.
#[must_use]
pub fn skipped_frames_warning(skipped: &[SkippedFrame]) -> Option<String> {
    let first = skipped.first()?;
    Some(format!(
        "{} image(s) could not be read and were skipped: {} (first error: {})",
        skipped.len(),
        list_files(skipped.iter().map(|frame| frame.path.as_path())),
        first.reason,
    ))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use stacker_gui::frames::SkippedFrame;

    use super::{MessageKind, ToastState, list_files, skipped_frames_warning};

    #[test]
    fn a_stale_token_never_matches_a_newer_toast() {
        let mut state = ToastState::default();
        let first = state.begin();
        assert!(state.is_current(first));
        let second = state.begin();
        assert!(!state.is_current(first), "the first timer is stale now");
        assert!(state.is_current(second));
    }

    #[test]
    fn invalidate_cancels_every_pending_dismiss() {
        let mut state = ToastState::default();
        let token = state.begin();
        state.invalidate();
        assert!(!state.is_current(token));
    }

    #[test]
    fn the_counter_wraps_instead_of_panicking() {
        let mut state = ToastState {
            generation: u64::MAX,
        };
        let token = state.begin();
        assert_eq!(token, 0);
        assert!(state.is_current(0));
    }

    #[test]
    fn only_warnings_and_errors_are_sticky() {
        assert!(!MessageKind::Info.is_sticky());
        assert!(!MessageKind::Success.is_sticky());
        assert!(MessageKind::Warning.is_sticky());
        assert!(MessageKind::Error.is_sticky());
    }

    #[test]
    fn kind_names_round_trip() {
        for kind in [
            MessageKind::Info,
            MessageKind::Success,
            MessageKind::Warning,
            MessageKind::Error,
        ] {
            assert_eq!(MessageKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(MessageKind::from_name("loud"), None);
    }

    #[test]
    fn a_transient_toast_does_not_hide_a_sticky_one() {
        assert!(MessageKind::Success.replaces(None));
        assert!(MessageKind::Success.replaces(Some(MessageKind::Info)));
        assert!(!MessageKind::Success.replaces(Some(MessageKind::Warning)));
        assert!(!MessageKind::Info.replaces(Some(MessageKind::Error)));
        assert!(MessageKind::Error.replaces(Some(MessageKind::Warning)));
        assert!(MessageKind::Warning.replaces(Some(MessageKind::Error)));
    }

    #[test]
    fn long_file_lists_are_cut_with_a_count() {
        let paths: Vec<PathBuf> = (1..=8)
            .map(|i| PathBuf::from(format!("dir/f{i}.jpg")))
            .collect();
        let text = list_files(paths.iter().map(PathBuf::as_path));
        assert_eq!(text, "f1.jpg, f2.jpg, f3.jpg, f4.jpg, f5.jpg and 3 more");
        assert_eq!(list_files([Path::new("a/b.png")]), "b.png");
    }

    #[test]
    fn the_skipped_warning_lists_the_files_and_the_first_reason() {
        let skipped = vec![
            SkippedFrame {
                path: PathBuf::from("x/a.jpg"),
                reason: "truncated file".to_owned(),
            },
            SkippedFrame {
                path: PathBuf::from("x/b.jpg"),
                reason: "unsupported".to_owned(),
            },
        ];
        let text = skipped_frames_warning(&skipped).expect("two frames were skipped");
        assert!(text.starts_with("2 image(s)"), "{text}");
        assert!(text.contains("a.jpg, b.jpg"), "{text}");
        assert!(text.contains("truncated file"), "{text}");
        assert_eq!(skipped_frames_warning(&[]), None);
    }
}
