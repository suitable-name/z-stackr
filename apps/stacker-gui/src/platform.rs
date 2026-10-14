//! The seam between the GUI and the host it runs on.
//!
//! The wiring in `gui/` and the headless pieces around it never touch the file
//! system, the native dialogs, the per-user config folder or the log location
//! directly: they ask the [`Platform`] installed for the UI thread. The desktop
//! app uses [`native::NativePlatform`] (today's behaviour: `rfd` dialogs, files
//! under the `directories` config folder, `std::fs`); the browser build installs
//! its own implementation, where "paths" are virtual `web:/<file name>` names
//! whose bytes the page holds, storage is `localStorage` and a "save" is a
//! download.
//!
//! # Threading
//!
//! A platform is `!Send` and every method is called on the UI thread. The
//! asynchronous ones ([`Platform::pick`]) take a continuation that the
//! implementation calls later, again on the UI thread. The installed platform
//! lives in a thread-local: a thread that never installed one gets the native
//! platform on the desktop builds (so tests and worker threads need no set-up)
//! and panics on wasm32, where [`install`] must run first.

use std::{
    cell::RefCell,
    io,
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::pickers::{Picked, PickerRequest};

#[cfg(not(target_arch = "wasm32"))]
pub mod native;

/// What a stale check needs to know about a file: its size and its modified
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStamp {
    /// The size in bytes.
    pub size: u64,
    /// The modified time in nanoseconds since the Unix epoch, when the host
    /// knows it.
    pub modified_ns: Option<u128>,
}

/// The continuation of a picker: called once, on the UI thread, with the answer
/// (empty when the dialog was cancelled).
pub type PickDone = Box<dyn FnOnce(Picked)>;

/// Everything the GUI needs from the host. See the module docs.
pub trait Platform {
    /// `true` for the browser build. The window hides what a browser cannot do
    /// (live folder monitor, quit, output folder, compute device, recent
    /// projects) and the shortcut list shows the browser's key variants.
    fn is_web(&self) -> bool;

    /// Ask the user for files, a folder or a file name to write, per `request`,
    /// and call `on_done` with the answer later (or at once, when the host needs
    /// no dialog). "Open" kinds answer with the paths [`Self::read_bytes`] can
    /// read; a "save" kind answers with the path to hand to
    /// [`Self::write_bytes`].
    fn pick(&self, request: PickerRequest, on_done: PickDone);

    /// The whole content of the file `path`.
    ///
    /// # Errors
    ///
    /// The host's I/O error ([`io::ErrorKind::NotFound`] for a missing file).
    fn read_bytes(&self, path: &Path) -> io::Result<Vec<u8>>;

    /// The text content of the file `path`.
    ///
    /// # Errors
    ///
    /// The read error, or [`io::ErrorKind::InvalidData`] when the file is not
    /// UTF-8 text.
    fn read_to_string(&self, path: &Path) -> io::Result<String> {
        String::from_utf8(self.read_bytes(path)?)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Write `bytes` as the file `path`: atomically (a temporary sibling and a
    /// rename) on the desktop, a download named after `path` in the browser.
    ///
    /// # Errors
    ///
    /// The host's I/O error.
    fn write_bytes(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;

    /// The stored text under `key` (`settings.toml`, `layout.toml`, `prefs.toml`,
    /// `recent.toml`); `None` when nothing is stored or it cannot be read.
    fn load_value(&self, key: &str) -> Option<String>;

    /// Store `text` under `key`.
    ///
    /// # Errors
    ///
    /// A message for the user when the host cannot store it.
    fn store_value(&self, key: &str, text: &str) -> Result<(), String>;

    /// The files directly inside the folder `folder`; `None` when `folder` is not
    /// a folder (or the host cannot list folders, as in the browser).
    fn list_folder(&self, folder: &Path) -> Option<Vec<PathBuf>>;

    /// The stamp of the file `path`; `None` when it does not exist (or is not a
    /// file).
    fn file_stamp(&self, path: &Path) -> Option<FileStamp>;

    /// Where the run log is written; `None` when the host keeps no log file.
    fn log_path(&self) -> Option<PathBuf>;

    /// How `path` is named in messages: the full path on the desktop, the file
    /// name in the browser.
    fn display_path(&self, path: &Path) -> String {
        path.display().to_string()
    }

    /// The pixel size of the image file `path`, read from its header.
    fn image_dimensions(&self, path: &Path) -> Option<(u32, u32)> {
        let bytes = self.read_bytes(path).ok()?;
        image::io::Reader::new(io::Cursor::new(bytes))
            .with_guessed_format()
            .ok()?
            .into_dimensions()
            .ok()
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Rc<dyn Platform>>> = const { RefCell::new(None) };
}

/// Install `platform` as this thread's platform (the UI thread's), replacing any
/// installed earlier.
pub fn install(platform: Rc<dyn Platform>) {
    CURRENT.with(|slot| *slot.borrow_mut() = Some(platform));
}

#[cfg(not(target_arch = "wasm32"))]
fn default_platform() -> Rc<dyn Platform> {
    Rc::new(native::NativePlatform)
}

#[cfg(target_arch = "wasm32")]
fn default_platform() -> Rc<dyn Platform> {
    panic!("no platform installed: call stacker_gui::platform::install before the GUI starts")
}

/// The platform of this thread: the one [`install`] registered, else the native
/// one on the desktop builds.
///
/// # Panics
///
/// On wasm32 when no platform was installed.
#[must_use]
pub fn current() -> Rc<dyn Platform> {
    CURRENT.with(|slot| Rc::clone(slot.borrow_mut().get_or_insert_with(default_platform)))
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        path::{Path, PathBuf},
        rc::Rc,
    };

    use super::{FileStamp, PickDone, Platform, current, install};
    use crate::pickers::PickerRequest;

    /// A host whose only file holds `bytes`.
    struct Memory {
        bytes: Vec<u8>,
    }

    impl Platform for Memory {
        fn is_web(&self) -> bool {
            true
        }

        fn pick(&self, _request: PickerRequest, on_done: PickDone) {
            on_done(vec![PathBuf::from("web:/a.txt")]);
        }

        fn read_bytes(&self, _path: &Path) -> io::Result<Vec<u8>> {
            Ok(self.bytes.clone())
        }

        fn write_bytes(&self, _path: &Path, _bytes: &[u8]) -> io::Result<()> {
            Ok(())
        }

        fn load_value(&self, _key: &str) -> Option<String> {
            None
        }

        fn store_value(&self, _key: &str, _text: &str) -> Result<(), String> {
            Ok(())
        }

        fn list_folder(&self, _folder: &Path) -> Option<Vec<PathBuf>> {
            None
        }

        fn file_stamp(&self, _path: &Path) -> Option<FileStamp> {
            None
        }

        fn log_path(&self) -> Option<PathBuf> {
            None
        }
    }

    #[test]
    fn the_installed_platform_is_the_one_current_returns() {
        install(Rc::new(Memory { bytes: Vec::new() }));
        assert!(current().is_web());
    }

    #[test]
    fn text_reads_default_to_utf8_over_the_byte_read() {
        let text = Memory {
            bytes: b"sources = []".to_vec(),
        };
        assert_eq!(
            text.read_to_string(Path::new("web:/p.zstack.toml"))
                .unwrap(),
            "sources = []"
        );
        let binary = Memory {
            bytes: vec![0xff, 0xfe],
        };
        let error = binary.read_to_string(Path::new("web:/x")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn a_web_path_is_named_by_its_full_text_by_default() {
        let host = Memory { bytes: Vec::new() };
        assert_eq!(host.display_path(Path::new("web:/a.jpg")), "web:/a.jpg");
    }
}
