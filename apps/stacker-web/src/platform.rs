//! The browser's [`Platform`]: virtual files, `rfd`'s async picker, downloads and
//! `localStorage`.
//!
//! A path is `web:/<file name>` ([`crate::vfs`]). "Open" pickers read the chosen
//! files into the virtual file system and answer with their virtual paths; a "save"
//! needs no dialog (the browser asks where to put a download itself), so it answers
//! at once with `web:/<suggested name>`, and [`Platform::write_bytes`] turns the
//! write into a download.

use std::{
    io,
    path::{Path, PathBuf},
};

use stacker_gui::{
    gui::{
        Ctx,
        feedback::{self, MessageKind},
    },
    pickers::{PickerKind, PickerRequest},
    platform::{FileStamp, PickDone, Platform},
};
use stacker_web_core::dims_from_bytes;

use crate::{download, vfs};

/// The largest file the page reads: the browser holds the bytes in memory, twice
/// while a run sends them to the worker.
pub const MAX_FILE_BYTES: f64 = 512.0 * 1024.0 * 1024.0;

/// The prefix of every `localStorage` key this app writes.
const STORAGE_PREFIX: &str = "z-stackr.";

/// The message for a file over [`MAX_FILE_BYTES`], or `None` when it is small
/// enough to read.
#[must_use]
pub fn oversize_message(name: &str, size: f64) -> Option<String> {
    (size > MAX_FILE_BYTES).then(|| {
        format!(
            "\"{name}\" is larger than {} MB and was skipped: a browser tab cannot hold it. \
             Use the desktop app for files this large.",
            MAX_FILE_BYTES / (1024.0 * 1024.0)
        )
    })
}

/// Report `message` to the user, when the window exists.
pub fn report(kind: MessageKind, message: &str) {
    if let Some(ctx) = Ctx::current() {
        feedback::notify(&ctx, kind, message);
    }
}

/// The platform of the browser build.
#[derive(Debug, Default, Clone, Copy)]
pub struct WebPlatform;

impl WebPlatform {
    /// A new platform; it keeps no state of its own (the files are in [`crate::vfs`]).
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

/// The `localStorage`, `None` when the browser blocks it.
fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// The `localStorage` key of a stored value (`settings.toml` is `z-stackr.settings`).
fn storage_key(key: &str) -> String {
    format!(
        "{STORAGE_PREFIX}{}",
        key.strip_suffix(".toml").unwrap_or(key)
    )
}

/// Show the file picker for `request` and register what the user chose.
#[expect(
    clippy::future_not_send,
    reason = "wasm32 runs on one thread; the browser's file handles are not Send"
)]
async fn pick_files(request: PickerRequest) -> Vec<PathBuf> {
    let mut dialog = rfd::AsyncFileDialog::new();
    if let Some(title) = &request.title {
        dialog = dialog.set_title(title.as_str());
    }
    for filter in &request.filters {
        dialog = dialog.add_filter(filter.label.as_str(), filter.extensions.as_slice());
    }
    let handles = if request.kind == PickerKind::OpenFiles {
        dialog.pick_files().await.unwrap_or_default()
    } else {
        dialog.pick_file().await.into_iter().collect()
    };
    let mut paths = Vec::with_capacity(handles.len());
    for handle in handles {
        let name = handle.file_name();
        if let Some(message) = oversize_message(&name, handle.inner().size()) {
            report(MessageKind::Error, &message);
            continue;
        }
        paths.push(vfs::register(&name, handle.read().await));
    }
    paths
}

impl Platform for WebPlatform {
    fn is_web(&self) -> bool {
        true
    }

    fn pick(&self, request: PickerRequest, on_done: PickDone) {
        match request.kind {
            PickerKind::SaveFile => {
                let name = request
                    .default_file_name
                    .unwrap_or_else(|| "result".to_owned());
                on_done(vec![vfs::virtual_path(&name)]);
            }
            // The window hides everything that asks for a folder.
            PickerKind::PickFolder => on_done(Vec::new()),
            PickerKind::OpenFile | PickerKind::OpenFiles => {
                wasm_bindgen_futures::spawn_local(async move {
                    on_done(pick_files(request).await);
                });
            }
        }
    }

    fn read_bytes(&self, path: &Path) -> io::Result<Vec<u8>> {
        vfs::get(path).map(|bytes| bytes.to_vec()).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{} is not loaded", vfs::file_name(path)),
            )
        })
    }

    fn write_bytes(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        let name = vfs::file_name(path);
        download::download_bytes(&name, download::mime_for(&name), bytes)
            .map_err(io::Error::other)?;
        // Only text (a project, a config) is kept: opening a project just saved must
        // work, while a saved image would only hold memory.
        if name.to_ascii_lowercase().ends_with(".toml") {
            vfs::register_at(path.to_path_buf(), bytes.to_vec());
        }
        Ok(())
    }

    fn load_value(&self, key: &str) -> Option<String> {
        storage()?.get_item(&storage_key(key)).ok().flatten()
    }

    fn store_value(&self, key: &str, text: &str) -> Result<(), String> {
        let storage = storage().ok_or_else(|| {
            "The browser blocks local storage, so settings are not kept between visits.".to_owned()
        })?;
        storage
            .set_item(&storage_key(key), text)
            .map_err(|_| "The browser's local storage is full or blocked.".to_owned())
    }

    fn list_folder(&self, _folder: &Path) -> Option<Vec<PathBuf>> {
        None
    }

    fn file_stamp(&self, path: &Path) -> Option<FileStamp> {
        vfs::stamp(path)
    }

    fn log_path(&self) -> Option<PathBuf> {
        None
    }

    fn display_path(&self, path: &Path) -> String {
        vfs::file_name(path)
    }

    fn image_dimensions(&self, path: &Path) -> Option<(u32, u32)> {
        dims_from_bytes(&vfs::get(path)?)
    }
}
