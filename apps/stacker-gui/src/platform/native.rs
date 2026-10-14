//! The desktop [`Platform`]: native file dialogs, `std::fs` and the per-user
//! config folder.
//!
//! The dialogs are `rfd`'s synchronous ones, shown on a spawned thread so they
//! never block the UI thread (`rfd`'s dialogs pump their own message loop, which
//! would freeze the window and every worker result waiting on the event loop).
//! The continuation is usually not `Send` (it captures the `Rc`-based GUI
//! context), so it is parked in [`crate::pickers`] under a plain key and only the
//! key and the `Send` answer cross the thread boundary; the continuation runs on
//! the UI thread, exactly once, or not at all when the event loop has ended.
//!
//! With the `ui-harness` feature no dialog is ever built: the answer comes from
//! the slint-harness picker queue and is delivered through the same event-loop
//! route, without the thread.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use super::{FileStamp, PickDone, Platform};
use crate::{
    pickers::{self, Picked, PickerKind, PickerRequest},
    settings::{self, layout, store},
};

/// The platform of the desktop app.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativePlatform;

/// The file `key` names in the platform config folder (next to `settings.toml`);
/// `None` when the platform has no config folder.
fn config_file(key: &str) -> Option<PathBuf> {
    settings::get_config_path()
        .ok()
        .map(|settings| settings.with_file_name(key))
}

/// How the UI harness's picker queue names a request in its
/// `unexpected file dialog: ...` finding: the title, else the kind and the
/// suggested file name.
#[cfg(feature = "ui-harness")]
fn harness_detail(request: &PickerRequest) -> String {
    let kind = format!("{:?}", request.kind);
    match (&request.title, &request.default_file_name) {
        (Some(title), _) => format!("{title} ({kind})"),
        (None, Some(name)) => format!("{kind} {name}"),
        (None, None) => kind,
    }
}

/// Turn a request into a real `rfd::FileDialog` and show it. The only place in
/// the app that constructs one; always runs on a spawned thread. Not compiled
/// with the `ui-harness` feature, which answers from the harness's queue.
#[cfg_attr(
    feature = "ui-harness",
    expect(
        dead_code,
        reason = "the ui-harness build answers dialogs from the harness queue"
    )
)]
fn build_and_show(request: &PickerRequest) -> Picked {
    let mut dialog = rfd::FileDialog::new();
    if let Some(title) = &request.title {
        dialog = dialog.set_title(title);
    }
    for filter in &request.filters {
        let extensions: Vec<&str> = filter.extensions.iter().map(String::as_str).collect();
        dialog = dialog.add_filter(&filter.label, &extensions);
    }
    if let Some(name) = &request.default_file_name {
        dialog = dialog.set_file_name(name);
    }
    if let Some(dir) = &request.starting_dir {
        dialog = dialog.set_directory(dir);
    }
    match request.kind {
        PickerKind::OpenFile => dialog.pick_file().into_iter().collect(),
        PickerKind::OpenFiles => dialog.pick_files().unwrap_or_default(),
        PickerKind::SaveFile => dialog.save_file().into_iter().collect(),
        PickerKind::PickFolder => dialog.pick_folder().into_iter().collect(),
    }
}

impl Platform for NativePlatform {
    fn is_web(&self) -> bool {
        false
    }

    fn pick(&self, request: PickerRequest, on_done: PickDone) {
        // The UI harness answers from its queue: no dialog and no thread. The answer
        // is still delivered through the event loop, so the continuation never runs
        // inside the callback that asked.
        #[cfg(feature = "ui-harness")]
        {
            let picked: Picked = slint_harness_hooks::picker::next(&harness_detail(&request))
                .into_iter()
                .collect();
            let key = pickers::park(on_done);
            let queued = slint::invoke_from_event_loop(move || {
                let _ = pickers::deliver(key, picked);
            });
            if queued.is_err() {
                tracing::debug!("picker answer dropped: the event loop has ended");
            }
        }
        #[cfg(not(feature = "ui-harness"))]
        {
            let key = pickers::park(on_done);
            std::thread::spawn(move || {
                let picked = build_and_show(&request);
                let queued = slint::invoke_from_event_loop(move || {
                    let _ = pickers::deliver(key, picked);
                });
                if queued.is_err() {
                    tracing::debug!("picker answer dropped: the event loop has ended");
                }
            });
        }
    }

    fn read_bytes(&self, path: &Path) -> io::Result<Vec<u8>> {
        fs::read(path)
    }

    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        fs::read_to_string(path)
    }

    fn write_bytes(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        store::write_atomic(path, bytes)
    }

    fn load_value(&self, key: &str) -> Option<String> {
        let path = config_file(key)?;
        match fs::read_to_string(&path) {
            Ok(text) => Some(text),
            Err(e) => {
                if e.kind() != io::ErrorKind::NotFound {
                    tracing::warn!("could not read {}: {e}", path.display());
                }
                None
            }
        }
    }

    fn store_value(&self, key: &str, text: &str) -> Result<(), String> {
        let path = config_file(key)
            .ok_or_else(|| "no configuration directory on this system".to_owned())?;
        store::write_atomic(&path, text).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn list_folder(&self, folder: &Path) -> Option<Vec<PathBuf>> {
        if !folder.is_dir() {
            return None;
        }
        let entries = fs::read_dir(folder).ok()?;
        Some(
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|entry| entry.is_file())
                .collect(),
        )
    }

    fn file_stamp(&self, path: &Path) -> Option<FileStamp> {
        let metadata = fs::metadata(path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|elapsed| elapsed.as_nanos());
        Some(FileStamp {
            size: metadata.len(),
            modified_ns,
        })
    }

    fn log_path(&self) -> Option<PathBuf> {
        Some(layout::default_log_path())
    }

    fn image_dimensions(&self, path: &Path) -> Option<(u32, u32)> {
        image::image_dimensions(path).ok()
    }
}
