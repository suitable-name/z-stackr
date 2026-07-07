pub use stacker_core::settings::*;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SettingsError {
    #[error("could not determine config directory")]
    ConfigDirNotFound,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("TOML serialise error: {0}")]
    Serialize(#[from] toml::ser::Error),
}

/// # Errors
///
/// Returns [`SettingsError::ConfigDirNotFound`] if the platform's config
/// directory cannot be determined (`directories::ProjectDirs::from` fails).
pub fn get_config_path() -> Result<PathBuf, SettingsError> {
    let proj_dirs = directories::ProjectDirs::from("", "", "z-stackr")
        .ok_or(SettingsError::ConfigDirNotFound)?;
    Ok(proj_dirs.config_dir().join("settings.toml"))
}

#[must_use]
pub fn load() -> StackingSettings {
    let path = match get_config_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warning: {e}");
            return StackingSettings::default();
        }
    };

    let Ok(content) = std::fs::read_to_string(&path) else {
        return StackingSettings::default();
    };

    match toml::from_str::<StackingSettings>(&content) {
        Ok(mut s) => {
            s.clamp_valid();
            s.preprocessing.clamp_valid();
            s.image_saving.clamp_valid();
            s
        }
        Err(e) => {
            eprintln!("warning: failed to parse {}: {e}", path.display());
            StackingSettings::default()
        }
    }
}

/// # Errors
///
/// Returns [`SettingsError::ConfigDirNotFound`] if the config path cannot be
/// resolved, [`SettingsError::Io`] if creating the config directory or
/// writing the file fails, or [`SettingsError::Serialize`] if `settings`
/// fails to serialise to TOML.
pub fn save(settings: &StackingSettings) -> Result<(), SettingsError> {
    let path = get_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = toml::to_string_pretty(settings)?;
    std::fs::write(path, content)?;
    Ok(())
}
