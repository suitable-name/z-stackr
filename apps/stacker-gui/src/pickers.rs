//! File and folder pickers that never block the UI thread.
//!
//! `rfd`'s synchronous dialogs pump their own message loop; opened from a Slint
//! callback they freeze the window (and every worker result waiting on the event
//! loop) until the dialog closes. [`pick`] hands the request to the installed
//! [`Platform`](crate::platform::Platform), which on the desktop shows the dialog
//! on a spawned thread and delivers the answer back on the UI thread.
//!
//! The continuation is usually not `Send` (it captures the `Rc`-based GUI
//! context), so it never crosses a thread boundary: it is parked in a
//! thread-local map under a plain `u64` key ([`park`]), and only that key and the
//! `Send`-safe answer travel through the spawned thread and the event loop
//! ([`deliver`]). The continuation runs on the UI thread, exactly once, or not at
//! all if the event loop has already ended.
//!
//! Callers must not hold a `RefCell` borrow across a call to [`pick`]: pick
//! first, borrow inside the continuation.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    path::PathBuf,
};

/// What kind of dialog to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PickerKind {
    /// Choose one existing file.
    OpenFile,
    /// Choose any number of existing files.
    OpenFiles,
    /// Choose a file name to write.
    SaveFile,
    /// Choose a directory.
    PickFolder,
}

/// One entry of the dialog's file-type drop-down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerFilter {
    /// The name shown in the drop-down.
    pub label: String,
    /// The lower-case extensions it admits, without the dot.
    pub extensions: Vec<String>,
}

impl PickerFilter {
    /// A filter such as `PickerFilter::new("PNG Image", &["png"])`.
    #[must_use]
    pub fn new(label: &str, extensions: &[&str]) -> Self {
        Self {
            label: label.to_owned(),
            extensions: extensions.iter().map(|e| (*e).to_owned()).collect(),
        }
    }
}

/// Everything one dialog needs, as plain data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerRequest {
    /// What the dialog asks for.
    pub kind: PickerKind,
    /// The dialog title; `None` keeps the OS default.
    pub title: Option<String>,
    /// File-type filters in order; the first one leads.
    pub filters: Vec<PickerFilter>,
    /// A save dialog's suggested file name.
    pub default_file_name: Option<String>,
    /// The directory the dialog starts in; `None` keeps the OS's last-used one.
    pub starting_dir: Option<PathBuf>,
}

impl PickerRequest {
    /// A request of `kind` with nothing else set.
    #[must_use]
    pub const fn new(kind: PickerKind) -> Self {
        Self {
            kind,
            title: None,
            filters: Vec::new(),
            default_file_name: None,
            starting_dir: None,
        }
    }

    /// Set the dialog title.
    #[must_use]
    pub fn titled(mut self, title: &str) -> Self {
        self.title = Some(title.to_owned());
        self
    }

    /// Append a file-type filter.
    #[must_use]
    pub fn filter(mut self, label: &str, extensions: &[&str]) -> Self {
        self.filters.push(PickerFilter::new(label, extensions));
        self
    }

    /// Suggest a file name in a save dialog.
    #[must_use]
    pub fn file_name(mut self, name: &str) -> Self {
        self.default_file_name = Some(name.to_owned());
        self
    }

    /// Start the dialog in `dir`.
    #[must_use]
    pub fn starting_in(mut self, dir: Option<PathBuf>) -> Self {
        self.starting_dir = dir;
        self
    }
}

/// The dialog's answer: the chosen paths, empty when it was cancelled. Single
/// pickers return at most one path.
pub type Picked = Vec<PathBuf>;

/// A parked continuation (named to keep the map type readable).
type Continuation = Box<dyn FnOnce(Picked)>;

thread_local! {
    /// Test hook: when `Some`, the NEXT [`pick`] on this thread resolves to this
    /// answer with no dialog and no spawned thread, consumed exactly once.
    static STUBBED_ANSWER: RefCell<Option<Picked>> = const { RefCell::new(None) };
    /// Continuations waiting for their dialog, by key (UI thread only).
    static PARKED: RefCell<HashMap<u64, Continuation>> = RefCell::new(HashMap::new());
    static NEXT_KEY: Cell<u64> = const { Cell::new(0) };
    /// Test hook: when set, [`pick`] never shows a real dialog on this thread.
    static HEADLESS: Cell<bool> = const { Cell::new(false) };
    /// The requests [`pick`] received while [`HEADLESS`] was set.
    static REQUESTS: RefCell<Vec<PickerRequest>> = const { RefCell::new(Vec::new()) };
}

/// Never show a real dialog on this thread: every [`pick`] answers with the
/// stubbed answer, or as cancelled, and is logged for [`take_requests`]. For
/// the UI flow tests.
#[doc(hidden)]
pub fn set_headless(on: bool) {
    HEADLESS.with(|cell| cell.set(on));
}

/// The dialogs requested since the last call, oldest first. For tests.
#[doc(hidden)]
#[must_use]
pub fn take_requests() -> Vec<PickerRequest> {
    REQUESTS.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

/// Stub the next [`pick`] on this thread: it will resolve to `answer` (an empty
/// list stands for a cancelled dialog) without showing anything. For tests.
#[doc(hidden)]
pub fn stub_next_answer(answer: Picked) {
    STUBBED_ANSWER.with(|cell| *cell.borrow_mut() = Some(answer));
}

/// `true` while a stubbed answer is waiting to be consumed. For tests.
#[doc(hidden)]
#[must_use]
pub fn has_stubbed_answer() -> bool {
    STUBBED_ANSWER.with(|cell| cell.borrow().is_some())
}

fn take_stubbed_answer() -> Option<Picked> {
    STUBBED_ANSWER.with(|cell| cell.borrow_mut().take())
}

/// Park `on_done` and return its key.
#[doc(hidden)]
#[must_use]
pub fn park(on_done: impl FnOnce(Picked) + 'static) -> u64 {
    let key = NEXT_KEY.with(|c| {
        let key = c.get();
        c.set(key + 1);
        key
    });
    PARKED.with(|map| map.borrow_mut().insert(key, Box::new(on_done)));
    key
}

/// Run the continuation parked under `key` with `picked`. Returns `false` when
/// there is none (already delivered), so a key is only ever honoured once.
#[doc(hidden)]
#[must_use]
pub fn deliver(key: u64, picked: Picked) -> bool {
    // Take it out of the map first: the continuation may itself call `pick`.
    let continuation = PARKED.with(|map| map.borrow_mut().remove(&key));
    continuation.is_some_and(|on_done| {
        on_done(picked);
        true
    })
}

/// Ask the installed [`Platform`](crate::platform::Platform) for `request`'s
/// answer and run `on_done` with it on the UI thread (an empty list when the
/// dialog was cancelled).
///
/// The desktop shows the dialog on a background thread (see
/// [`crate::platform::native`]); the browser answers from its own file picker, or
/// at once for a name to save under.
///
/// Must be called from the UI thread (the thread running the Slint event loop).
pub fn pick(request: PickerRequest, on_done: impl FnOnce(Picked) + 'static) {
    if HEADLESS.with(Cell::get) {
        // Tests: never a real dialog. The request is logged; the answer is the
        // stub, or "cancelled" when the test set none.
        let answer = take_stubbed_answer().unwrap_or_default();
        REQUESTS.with(|log| log.borrow_mut().push(request));
        on_done(answer);
        return;
    }
    if let Some(answer) = take_stubbed_answer() {
        on_done(answer);
        return;
    }
    crate::platform::current().pick(request, Box::new(on_done));
}
