//! Settings text → [`StackingSettings`]: the one loader both apps use.
//!
//! The text is parsed by [`StackingSettings::from_toml_str`] and so passes
//! through [`StackingSettings::sanitized`]: every numeric key is clamped to its
//! documented range, enum strings are case-insensitive, and each problem is
//! returned as a warning line instead of being dropped.
//!
//! The legacy spellings of [`apply_legacy_aliases`] are accepted in both
//! modes: `alignment_mode = "bounded"` (the `registration` mode with
//! `bounded_search = true`), the `dmap_*` keys (now `relief_*`), the older
//! `use_gpu` (now `compute_device`) and `relief_engine` (now
//! `relief_use_multigrid`); the current name wins when both are present.
//! A file without `settings_version` is a legacy file whose
//! `alignment_mode` is migrated to the current meanings (see
//! [`StackingSettings::migrate_legacy`]). Reading only: nothing legacy is
//! ever written.
//!
//! [`Leniency`] decides what a wrongly typed value or an unknown enum string
//! does. The desktop app is [`Leniency::Lenient`]: that key keeps its default,
//! the rest of the file is honoured, and the warning is shown. The CLI is
//! [`Leniency::Strict`]: an unattended run must not silently continue with a
//! setting it could not read, so the run stops with the parser's message.

use stacker_core::settings::{StackingSettings, apply_legacy_aliases};

/// How to treat a value of the wrong type or an unknown enum string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leniency {
    /// Keep the key's default and report a warning.
    Lenient,
    /// Fail with the parser's message.
    Strict,
}

/// Settings plus the problems found while reading them.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedSettings {
    /// The parsed settings, aliases applied and every number clamped into range.
    pub settings: StackingSettings,
    /// One line per ignored key, defaulted value or clamped number.
    pub warnings: Vec<String>,
}

/// Parse the text of a settings file.
///
/// # Errors
/// Returns the parser's message for invalid TOML and, with
/// [`Leniency::Strict`], for a wrongly typed value or an unknown enum value.
pub fn parse_settings(text: &str, leniency: Leniency) -> Result<LoadedSettings, String> {
    if leniency == Leniency::Strict {
        // Only the *check* sees the rewritten table: the loader below applies
        // the same aliases itself and must see the original text, because the
        // legacy migration depends on whether `settings_version` is present.
        let mut table: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
        apply_legacy_aliases(&mut table);
        toml::Value::Table(table)
            .try_into::<StackingSettings>()
            .map_err(|e| e.to_string())?;
    }

    let (settings, warnings) = StackingSettings::from_toml_str(text)?;
    Ok(LoadedSettings { settings, warnings })
}
