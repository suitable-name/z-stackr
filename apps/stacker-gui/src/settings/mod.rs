//! The GUI's settings: the shared [`StackingSettings`] model plus where it lives
//! on disk ([`store`]) and how changes reach the disk ([`persist`]).
//!
//! The per-user files beside `settings.toml` are submodules too: the window
//! layout ([`layout`]), the hint choices ([`prefs`]) and the recent-projects list
//! ([`recent`]).
//!
//! On the desktop every one of them is a file in the platform config folder. The
//! browser build keeps them in the platform's key-value store under the same
//! names ([`SETTINGS_KEY`], [`layout::FILE_NAME`], ...): see [`open_stored`].

pub mod layout;
pub mod persist;
pub mod prefs;
pub mod recent;
pub mod store;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, PoisonError};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub use persist::SettingsPersister;
pub use stacker_core::settings::*;
use stacker_pipeline::config::{Leniency, parse_settings};
pub use store::{LoadOutcome, LoadResult};
use thiserror::Error;

/// The key of the settings file in the platform's key-value store.
pub const SETTINGS_KEY: &str = "settings.toml";

/// Why the settings file could not be located, read or written.
#[derive(Debug, Error)]
pub enum SettingsError {
    /// The platform has no per-user configuration folder.
    #[error("could not determine config directory")]
    ConfigDirNotFound,
    /// Reading or writing the file failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not valid TOML.
    #[error("TOML parse error: {0}")]
    Parse(#[from] toml::de::Error),
    /// The settings could not be serialised.
    #[error("TOML serialise error: {0}")]
    Serialize(#[from] toml::ser::Error),
}

/// A folder that replaces the platform config directory, set by the UI flow
/// tests so they never read or write the user's real files.
#[cfg(not(target_arch = "wasm32"))]
static CONFIG_DIR_OVERRIDE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Use `dir` instead of the platform config directory for `settings.toml` and
/// every file beside it (layout, prefs, recent projects), process-wide. For
/// tests; `None` goes back to the platform directory.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub fn override_config_dir(dir: Option<PathBuf>) {
    *CONFIG_DIR_OVERRIDE
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = dir;
}

/// The path of the settings file in the platform config directory.
///
/// Desktop builds only: the browser has no such directory. With the `ui-harness`
/// feature the harness's data directory is used when it set one.
///
/// # Errors
///
/// [`SettingsError::ConfigDirNotFound`] when the platform has no home directory.
#[cfg(not(target_arch = "wasm32"))]
pub fn get_config_path() -> Result<PathBuf, SettingsError> {
    if let Some(dir) = CONFIG_DIR_OVERRIDE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
    {
        return Ok(dir.join("settings.toml"));
    }
    // The UI harness's throw-away profile: Windows Known Folders ignore the
    // environment redirect, so `directories` would reach the user's real folder.
    #[cfg(feature = "ui-harness")]
    if let Some(dir) = slint_harness_hooks::paths::data_dir() {
        return Ok(dir.join("settings.toml"));
    }
    let proj_dirs = directories::ProjectDirs::from("", "", "z-stackr")
        .ok_or(SettingsError::ConfigDirNotFound)?;
    Ok(proj_dirs.config_dir().join("settings.toml"))
}

/// The persister for the app's settings together with how the load went.
pub struct Opened {
    /// The settings copy and its background writer.
    pub persister: Arc<SettingsPersister>,
    /// How reading the file went (a first run, a damaged file, ...).
    pub outcome: LoadOutcome,
}

/// Load the settings at `path` and start a persister for them.
///
/// A first run drops the commented default config file so it exists and can be
/// hand-edited. A file that cannot be understood is renamed aside (see
/// [`store::load_from`]); when it could not even be moved, the persister is
/// [`SettingsPersister::disabled`] so the broken file is never overwritten.
/// `None` (no config directory) also yields a disabled persister.
#[must_use]
pub fn open_at(path: Option<&Path>) -> Opened {
    let Some(path) = path else {
        return Opened {
            persister: Arc::new(SettingsPersister::disabled(StackingSettings::default())),
            outcome: LoadOutcome::Unreadable {
                reason: "no configuration directory on this system".to_owned(),
            },
        };
    };
    let LoadResult { settings, outcome } = store::load_from(path);
    if outcome == LoadOutcome::Missing {
        match store::write_atomic(path, DEFAULT_CONFIG_TOML) {
            Ok(()) => tracing::info!("wrote default config to {}", path.display()),
            Err(e) => tracing::warn!("could not write default config: {e}"),
        }
    }
    let persister = if outcome.allows_writes() {
        SettingsPersister::spawn(path.to_path_buf(), settings)
    } else {
        SettingsPersister::disabled(settings)
    };
    Opened {
        persister: Arc::new(persister),
        outcome,
    }
}

/// Load the settings from the platform's key-value store (key
/// [`SETTINGS_KEY`]) and start a persister that writes back to it.
///
/// The same rules as [`open_at`]: a first run stores the commented default
/// config; text that cannot be understood is kept under `settings.toml.corrupt`
/// (never overwritten) and defaults are used; when even that copy cannot be
/// stored, the persister is [`SettingsPersister::disabled`].
#[must_use]
pub fn open_stored() -> Opened {
    let platform = crate::platform::current();
    let LoadResult { settings, outcome } = platform.load_value(SETTINGS_KEY).map_or_else(
        || LoadResult {
            settings: StackingSettings::default(),
            outcome: LoadOutcome::Missing,
        },
        |text| match parse_settings(&text, Leniency::Lenient) {
            Ok(loaded) => LoadResult {
                settings: loaded.settings,
                outcome: LoadOutcome::Loaded {
                    warnings: loaded.warnings,
                },
            },
            Err(reason) => {
                tracing::warn!("the stored settings are not valid settings TOML: {reason}");
                let aside = format!("{SETTINGS_KEY}.corrupt");
                let renamed_to = platform
                    .store_value(&aside, &text)
                    .is_ok()
                    .then(|| PathBuf::from(aside));
                LoadResult {
                    settings: StackingSettings::default(),
                    outcome: LoadOutcome::Corrupt { reason, renamed_to },
                }
            }
        },
    );
    if outcome == LoadOutcome::Missing
        && let Err(e) = platform.store_value(SETTINGS_KEY, DEFAULT_CONFIG_TOML)
    {
        tracing::warn!("could not store the default config: {e}");
    }
    let persister = if outcome.allows_writes() {
        SettingsPersister::stored(settings)
    } else {
        SettingsPersister::disabled(settings)
    };
    Opened {
        persister: Arc::new(persister),
        outcome,
    }
}

/// Load the settings of the running platform: the config file on the desktop
/// ([`open_at`] for the platform config path), the key-value store in the
/// browser ([`open_stored`]).
#[must_use]
pub fn open_default() -> Opened {
    #[cfg(not(target_arch = "wasm32"))]
    {
        open_at(get_config_path().ok().as_deref())
    }
    #[cfg(target_arch = "wasm32")]
    {
        open_stored()
    }
}
