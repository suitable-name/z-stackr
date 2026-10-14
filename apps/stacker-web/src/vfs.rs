//! The page's virtual file system.
//!
//! A browser gives the page the bytes of a file, never its path, so every file the
//! user picks or drops is registered here under the virtual path
//! `web:/<file name>` and the GUI's path-based logic (source lists, natural order,
//! stale fingerprints, project sources) works unchanged on those names. A second
//! file with the same name replaces the first: the same name is the same frame for
//! the user.
//!
//! Results live under `web:/results/<label>_<counter>.png`. Every entry carries a
//! stamp (its size and a registration counter standing in for the modified time),
//! so replacing a file is seen by the stale checks.
//!
//! The map is thread-local: the page has one UI thread and the platform and the
//! runner both live on it.

use std::{
    cell::RefCell,
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
};

use stacker_gui::platform::FileStamp;

/// The prefix of every virtual path.
const PREFIX: &str = "web:/";

/// One registered file.
struct Entry {
    bytes: Rc<[u8]>,
    stamp: FileStamp,
}

#[derive(Default)]
struct Vfs {
    files: HashMap<PathBuf, Entry>,
    /// Counts registrations; also numbers the result files.
    counter: u64,
}

thread_local! {
    static VFS: RefCell<Vfs> = RefCell::new(Vfs::default());
}

/// The virtual path of the file called `name`.
#[must_use]
pub fn virtual_path(name: &str) -> PathBuf {
    PathBuf::from(format!("{PREFIX}{name}"))
}

/// The file name of a virtual path (the whole text when it has none).
#[must_use]
pub fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Register `bytes` as the file called `name` and return its virtual path. An
/// earlier file of that name is replaced.
pub fn register(name: &str, bytes: Vec<u8>) -> PathBuf {
    let path = virtual_path(name);
    register_at(path.clone(), bytes);
    path
}

/// Register `bytes` under the virtual path `path`, replacing any earlier file there.
pub fn register_at(path: PathBuf, bytes: Vec<u8>) {
    VFS.with(|vfs| {
        let mut vfs = vfs.borrow_mut();
        vfs.counter += 1;
        let stamp = FileStamp {
            size: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            modified_ns: Some(u128::from(vfs.counter)),
        };
        vfs.files.insert(
            path,
            Entry {
                bytes: Rc::from(bytes),
                stamp,
            },
        );
    });
}

/// Register a result image and return its path
/// (`web:/results/<label>_<counter>.png`).
pub fn register_result(label: &str, bytes: Vec<u8>) -> PathBuf {
    let counter = VFS.with(|vfs| vfs.borrow().counter) + 1;
    let path = virtual_path(&format!("results/{label}_{counter}.png"));
    register_at(path.clone(), bytes);
    path
}

/// The bytes of the file at `path`, shared (no copy).
#[must_use]
pub fn get(path: &Path) -> Option<Rc<[u8]>> {
    VFS.with(|vfs| vfs.borrow().files.get(path).map(|e| Rc::clone(&e.bytes)))
}

/// The stamp of the file at `path`.
#[must_use]
pub fn stamp(path: &Path) -> Option<FileStamp> {
    VFS.with(|vfs| vfs.borrow().files.get(path).map(|e| e.stamp))
}
