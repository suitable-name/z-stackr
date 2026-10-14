//! Loading and atomically writing the settings file.
//!
//! A settings file must never stop the app from starting, and it must never be
//! silently destroyed: a file that fails to parse is renamed to
//! `settings.toml.corrupt-<unix seconds>` before defaults are used, so the next
//! save cannot overwrite the only copy of what the user wrote.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use stacker_core::settings::StackingSettings;
use stacker_pipeline::config::{Leniency, parse_settings};
use web_time::{SystemTime, UNIX_EPOCH};

use super::SettingsError;

/// What happened while loading the settings file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadOutcome {
    /// There was no file (first run).
    Missing,
    /// The file parsed; `warnings` are its clamp and unknown-key notes.
    Loaded {
        /// The clamp and unknown-key notes.
        warnings: Vec<String>,
    },
    /// The file exists but is not valid settings TOML. It was moved to
    /// `renamed_to` (`None` when the rename itself failed and the broken file is
    /// still at its original path) and defaults are in use.
    Corrupt {
        /// Why the file was rejected.
        reason: String,
        /// Where the broken file was moved to.
        renamed_to: Option<PathBuf>,
    },
    /// The file could not be read at all (permissions, a device error).
    Unreadable {
        /// The I/O error message.
        reason: String,
    },
}

impl LoadOutcome {
    /// `true` when writing the settings file is safe. It is not while a file we
    /// could not understand is still sitting at the settings path.
    #[must_use]
    pub const fn allows_writes(&self) -> bool {
        match self {
            Self::Missing | Self::Loaded { .. } => true,
            Self::Corrupt { renamed_to, .. } => renamed_to.is_some(),
            Self::Unreadable { .. } => false,
        }
    }

    /// A line for the status bar, or `None` when there is nothing to tell.
    #[must_use]
    pub fn user_message(&self) -> Option<String> {
        match self {
            Self::Missing => None,
            Self::Loaded { warnings } if warnings.is_empty() => None,
            Self::Loaded { warnings } => Some(format!(
                "Settings file loaded with {} note(s); see the log for details.",
                warnings.len()
            )),
            Self::Corrupt {
                renamed_to: Some(moved),
                ..
            } => Some(format!(
                "Warning: the settings file could not be read. It was kept as {} and defaults are in use.",
                moved.file_name().map_or_else(
                    || moved.display().to_string(),
                    |n| n.to_string_lossy().into_owned()
                )
            )),
            Self::Corrupt {
                renamed_to: None, ..
            } => Some(
                "Warning: the settings file could not be read and could not be moved aside. \
                 Defaults are in use and changes will not be saved until it is fixed."
                    .to_owned(),
            ),
            Self::Unreadable { reason } => Some(format!(
                "Warning: the settings file could not be opened ({reason}). \
                 Defaults are in use and changes will not be saved."
            )),
        }
    }
}

/// The settings that were loaded and how the load went.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadResult {
    /// The settings in force: the file's, or the defaults when it was unusable.
    pub settings: StackingSettings,
    /// How the load went.
    pub outcome: LoadOutcome,
}

/// Rename `path` to `<path>.corrupt-<unix seconds>`; `None` (logged) on failure.
fn rename_aside(path: &Path) -> Option<PathBuf> {
    let unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let mut renamed = path.as_os_str().to_owned();
    renamed.push(format!(".corrupt-{unix_secs}"));
    let renamed = PathBuf::from(renamed);
    match fs::rename(path, &renamed) {
        Ok(()) => Some(renamed),
        Err(e) => {
            tracing::warn!(
                "could not rename the corrupt settings file {} to {}: {e}",
                path.display(),
                renamed.display()
            );
            None
        }
    }
}

/// Load the settings at `path`. Never fails: every problem becomes a
/// [`LoadOutcome`] and defaults.
#[must_use]
pub fn load_from(path: &Path) -> LoadResult {
    let defaults = |outcome| LoadResult {
        settings: StackingSettings::default(),
        outcome,
    };
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return defaults(LoadOutcome::Missing),
        Err(e) if e.kind() == io::ErrorKind::InvalidData => {
            let reason = e.to_string();
            let renamed_to = rename_aside(path);
            return defaults(LoadOutcome::Corrupt { reason, renamed_to });
        }
        Err(e) => {
            tracing::warn!("could not read {}: {e}", path.display());
            return defaults(LoadOutcome::Unreadable {
                reason: e.to_string(),
            });
        }
    };
    // The loader the CLI uses (aliases such as `alignment_mode = "bounded"`
    // included); a wrongly typed key keeps its default and is reported.
    match parse_settings(&content, Leniency::Lenient).map(|l| (l.settings, l.warnings)) {
        Ok((settings, warnings)) => {
            for w in &warnings {
                tracing::warn!("{}: {w}", path.display());
            }
            LoadResult {
                settings,
                outcome: LoadOutcome::Loaded { warnings },
            }
        }
        Err(reason) => {
            tracing::warn!("{} is not valid settings TOML: {reason}", path.display());
            let renamed_to = rename_aside(path);
            defaults(LoadOutcome::Corrupt { reason, renamed_to })
        }
    }
}

/// Write `contents` to `path` through a `.tmp` sibling and a rename, so a crash
/// mid-write leaves the previous file intact.
///
/// # Errors
///
/// Any I/O error from creating the directory, writing the temporary file or
/// renaming it over `path`.
pub fn write_atomic(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

/// Serialise `settings` and write them atomically to `path`.
///
/// # Errors
///
/// The TOML serialiser's error, or the I/O error from [`write_atomic`].
pub fn save_atomic(path: &Path, settings: &StackingSettings) -> Result<(), SettingsError> {
    let content = toml::to_string_pretty(settings)?;
    write_atomic(path, &content)?;
    Ok(())
}
