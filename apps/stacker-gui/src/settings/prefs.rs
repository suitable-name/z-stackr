//! Small per-user choices that are not stacking settings, kept in `prefs.toml`.
//!
//! Today that is the set of one-time hints the user asked not to see again
//! ("Don't show again" on a first-run explainer). The file sits next to
//! `settings.toml` and `layout.toml`, is written through the settings store's
//! atomic writer, and is disposable like the layout file: a file that is not
//! valid TOML reads as "nothing suppressed" and is replaced by the next save.
//!
//! ```toml
//! suppressed_hints = ["auto-cull", "relief-preview"]
//! ```

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::{collections::BTreeSet, fs, io, path::Path};

#[cfg(not(target_arch = "wasm32"))]
use crate::settings::get_config_path;
use crate::settings::store;

/// The file name, next to `settings.toml`.
pub const FILE_NAME: &str = "prefs.toml";

/// The key of the suppressed-hints list in the file.
const HINTS_KEY: &str = "suppressed_hints";

/// The current key of a hint: the Relief preview hint was `dmap-preview` before
/// the algorithm was renamed, and a file written then must keep its choice.
/// Reading only; the file is written with the current key.
fn canonical_hint(key: &str) -> String {
    match key {
        "dmap-preview" => "relief-preview".to_owned(),
        other => other.to_owned(),
    }
}

/// The persisted per-user choices.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Prefs {
    /// The keys of the hints the user asked not to see again.
    pub suppressed_hints: BTreeSet<String>,
}

impl Prefs {
    /// `true` when the hint `key` must not be shown.
    #[must_use]
    pub fn is_suppressed(&self, key: &str) -> bool {
        self.suppressed_hints.contains(key)
    }

    /// Never show the hint `key` again; `true` when that changed anything.
    pub fn suppress(&mut self, key: &str) -> bool {
        self.suppressed_hints.insert(key.to_owned())
    }

    /// Show every hint again; `true` when there was anything to forget.
    pub fn reset_hints(&mut self) -> bool {
        let had_any = !self.suppressed_hints.is_empty();
        self.suppressed_hints.clear();
        had_any
    }

    /// Parse `prefs.toml` text. Never fails: text that is not TOML, a missing
    /// key, or entries that are not strings give nothing suppressed (a string
    /// entry next to a bad one is still kept).
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let Ok(table) = text.parse::<toml::Table>() else {
            return Self::default();
        };
        let suppressed_hints: BTreeSet<String> = table
            .get(HINTS_KEY)
            .and_then(toml::Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(toml::Value::as_str)
                    .map(canonical_hint)
                    .collect()
            })
            .unwrap_or_default();
        Self { suppressed_hints }
    }

    /// The `prefs.toml` text for these choices.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let hints: Vec<toml::Value> = self
            .suppressed_hints
            .iter()
            .cloned()
            .map(toml::Value::String)
            .collect();
        let mut table = toml::Table::new();
        table.insert(HINTS_KEY.to_owned(), toml::Value::Array(hints));
        let body = toml::to_string(&table).unwrap_or_default();
        format!("# z-stackr per-user choices (written by the app).\n{body}")
    }
}

/// Where `prefs.toml` lives: next to `settings.toml`.
///
/// `None` when the platform has no configuration directory. Desktop builds only;
/// the browser build keeps the choices in the platform store ([`load_stored`],
/// [`save_stored`]).
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn prefs_path() -> Option<PathBuf> {
    get_config_path()
        .ok()
        .map(|settings| settings.with_file_name(FILE_NAME))
}

/// Load the choices at `path`. Never fails: a missing or unreadable file gives
/// the defaults.
#[must_use]
pub fn load(path: &Path) -> Prefs {
    match fs::read_to_string(path) {
        Ok(text) => Prefs::parse(&text),
        Err(e) => {
            if e.kind() != io::ErrorKind::NotFound {
                tracing::warn!("could not read {}: {e}", path.display());
            }
            Prefs::default()
        }
    }
}

/// Write `prefs` to `path` through the settings store's atomic writer.
///
/// # Errors
///
/// Any I/O error from creating the directory or writing the file.
pub fn save(path: &Path, prefs: &Prefs) -> io::Result<()> {
    store::write_atomic(path, prefs.to_toml())
}

/// Load the choices from the platform store (the file next to `settings.toml` on
/// the desktop). Never fails: nothing stored gives the defaults.
#[must_use]
pub fn load_stored() -> Prefs {
    crate::platform::current()
        .load_value(FILE_NAME)
        .map_or_else(Prefs::default, |text| Prefs::parse(&text))
}

/// Write `prefs` to the platform store.
///
/// # Errors
///
/// A message when the platform cannot store it.
pub fn save_stored(prefs: &Prefs) -> Result<(), String> {
    crate::platform::current().store_value(FILE_NAME, &prefs.to_toml())
}
