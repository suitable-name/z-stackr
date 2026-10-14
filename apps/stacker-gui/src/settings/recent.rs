//! The recent-projects list: `recent.toml` next to `settings.toml`.
//!
//! Newest first, at most [`MAX_RECENT`] entries, each project once. A project
//! file that no longer exists is dropped when the list is loaded, so the menu
//! and the empty state never offer a dead link. The file is written atomically
//! through the settings store.

use std::{
    io,
    path::{Path, PathBuf},
};

use toml::{Table, Value};

#[cfg(not(target_arch = "wasm32"))]
use crate::settings::get_config_path;
use crate::{platform, project::project_name, settings::store};

/// The most projects the list remembers.
pub const MAX_RECENT: usize = 10;

/// The list's file name.
pub const FILE_NAME: &str = "recent.toml";

/// The recently used project files, newest first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecentProjects {
    entries: Vec<PathBuf>,
}

impl RecentProjects {
    /// An empty list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// A list of `entries` (newest first): repeats are dropped, the list is cut
    /// at [`MAX_RECENT`].
    #[must_use]
    pub fn from_entries(entries: impl IntoIterator<Item = PathBuf>) -> Self {
        let mut list = Self::new();
        for path in entries {
            if !list.entries.contains(&path) && list.entries.len() < MAX_RECENT {
                list.entries.push(path);
            }
        }
        list
    }

    /// The projects, newest first.
    #[must_use]
    pub fn entries(&self) -> &[PathBuf] {
        &self.entries
    }

    /// `true` when there are none.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Put `path` first (an earlier entry for it moves up); the oldest entry
    /// falls off past [`MAX_RECENT`].
    pub fn record(&mut self, path: &Path) {
        self.entries.retain(|p| p != path);
        self.entries.insert(0, path.to_path_buf());
        self.entries.truncate(MAX_RECENT);
    }

    /// Remove `path` (a project that turned out to be gone).
    pub fn forget(&mut self, path: &Path) {
        self.entries.retain(|p| p != path);
    }

    /// Empty the list.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Drop the entries whose file no longer exists; `true` when any went.
    pub fn retain_existing(&mut self) -> bool {
        let before = self.entries.len();
        let platform = platform::current();
        self.entries.retain(|p| platform.file_stamp(p).is_some());
        self.entries.len() != before
    }

    /// The list as TOML text.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let projects = self
            .entries
            .iter()
            .map(|p| Value::String(p.to_string_lossy().into_owned()))
            .collect();
        let mut table = Table::new();
        table.insert("projects".to_owned(), Value::Array(projects));
        toml::to_string(&table).unwrap_or_default()
    }

    /// Read a list from TOML text; text that is not a list gives an empty one.
    #[must_use]
    pub fn from_toml(text: &str) -> Self {
        let Ok(table) = toml::from_str::<Table>(text) else {
            return Self::new();
        };
        let Some(Value::Array(items)) = table.get("projects") else {
            return Self::new();
        };
        Self::from_entries(items.iter().filter_map(|item| match item {
            Value::String(path) if !path.is_empty() => Some(PathBuf::from(path)),
            _ => None,
        }))
    }
}

/// The list file that belongs to the settings file `settings_path`.
#[must_use]
pub fn path_beside(settings_path: &Path) -> PathBuf {
    settings_path.with_file_name(FILE_NAME)
}

/// The list file in the platform config directory, `None` without one. Desktop
/// builds only; see [`load_stored`] and [`save_stored`] for the platform store.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn default_path() -> Option<PathBuf> {
    get_config_path().ok().map(|path| path_beside(&path))
}

/// Load the list from the platform store (the list file on the desktop), without
/// the projects whose file is gone. Nothing stored is an empty list.
#[must_use]
pub fn load_stored() -> RecentProjects {
    let mut list = platform::current()
        .load_value(FILE_NAME)
        .map(|text| RecentProjects::from_toml(&text))
        .unwrap_or_default();
    list.retain_existing();
    list
}

/// Write the list to the platform store.
///
/// # Errors
///
/// A message when the platform cannot store it.
pub fn save_stored(list: &RecentProjects) -> Result<(), String> {
    platform::current().store_value(FILE_NAME, &list.to_toml())
}

/// Load the list at `path`, without the projects whose file is gone. A missing
/// or unreadable file is an empty list.
#[must_use]
pub fn load_from(path: &Path) -> RecentProjects {
    let mut list = std::fs::read_to_string(path)
        .map(|text| RecentProjects::from_toml(&text))
        .unwrap_or_default();
    list.retain_existing();
    list
}

/// Write the list to `path` atomically.
///
/// # Errors
///
/// The I/O error of the write.
pub fn save_to(path: &Path, list: &RecentProjects) -> io::Result<()> {
    store::write_atomic(path, list.to_toml())
}

/// `name — folder`, how the empty state's cards and the menu name a project.
#[must_use]
pub fn display_label(path: &Path) -> String {
    let folder = path
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    if folder.is_empty() {
        project_name(path)
    } else {
        format!("{} \u{2014} {folder}", project_name(path))
    }
}
