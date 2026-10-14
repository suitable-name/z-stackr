//! The window layout the user chose, kept in a small `layout.toml`.
//!
//! The sidebar width lives next to `settings.toml`. This module is also home of
//! the other "where does a per-user file live" answer, the run log
//! ([`default_log_path`]).
//!
//! It is a separate file on purpose: [`stacker_core::settings::StackingSettings`]
//! is the shared pipeline model (the CLI reads the same file), and a panel width is
//! no business of the pipeline. A layout file is also disposable, so unlike the
//! settings file a broken one is simply replaced by the defaults on the next save.
//!
//! The `touched` flag records that the user has ever dragged the sidebar handle
//! themselves. Keys an earlier build wrote and this one no longer reads (the bottom
//! panel's height, from when the Retouch brush had a permanent panel) are ignored
//! and dropped on the next save.

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::{fs, io, path::Path};

#[cfg(not(target_arch = "wasm32"))]
use crate::settings::get_config_path;
use crate::settings::store;

/// Smallest sidebar width in logical pixels.
pub const SIDEBAR_MIN: f32 = 200.0;
/// Largest sidebar width in logical pixels.
pub const SIDEBAR_MAX: f32 = 520.0;
/// Sidebar width on a first run.
pub const SIDEBAR_DEFAULT: f32 = 280.0;

/// The file name, next to `settings.toml`.
pub const FILE_NAME: &str = "layout.toml";

/// The persisted panel sizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelLayout {
    /// Sidebar width in logical pixels.
    pub sidebar_width: f32,
    /// The user has dragged the sidebar handle at least once.
    pub touched: bool,
}

impl Default for PanelLayout {
    fn default() -> Self {
        Self {
            sidebar_width: SIDEBAR_DEFAULT,
            touched: false,
        }
    }
}

/// `value` limited to `min..=max`; a value that is not a finite number becomes
/// `default`.
const fn clamp_or(value: f32, min: f32, max: f32, default: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        default
    }
}

impl PanelLayout {
    /// The same layout with the width inside its limits.
    #[must_use]
    pub const fn clamped(self) -> Self {
        Self {
            sidebar_width: clamp_or(
                self.sidebar_width,
                SIDEBAR_MIN,
                SIDEBAR_MAX,
                SIDEBAR_DEFAULT,
            ),
            touched: self.touched,
        }
    }

    /// The layout after the user dragged the handle: the new width, clamped, and
    /// `touched` set for good.
    #[must_use]
    pub const fn with_user_width(self, sidebar_width: f32) -> Self {
        Self {
            sidebar_width,
            touched: true,
        }
        .clamped()
    }

    /// Parse `layout.toml` text. Never fails: a file that is not TOML gives the
    /// defaults, a missing or non-numeric key its default, an out-of-range number
    /// the nearest limit, and a key this build does not know is ignored.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let Ok(table) = text.parse::<toml::Table>() else {
            return Self::default();
        };
        let defaults = Self::default();
        let number =
            |key: &str, default: f32| table.get(key).and_then(number_of).unwrap_or(default);
        Self {
            sidebar_width: number("sidebar_width", defaults.sidebar_width),
            touched: table
                .get("touched")
                .and_then(toml::Value::as_bool)
                .unwrap_or(false),
        }
        .clamped()
    }

    /// The `layout.toml` text for this layout.
    #[must_use]
    pub fn to_toml(self) -> String {
        format!(
            "# z-stackr window layout (written by the app).\n\
             sidebar_width = {:.1}\n\
             touched = {}\n",
            self.sidebar_width, self.touched
        )
    }
}

/// A TOML number as `f32`; integers are accepted too (`sidebar_width = 300`).
#[expect(
    clippy::cast_possible_truncation,
    reason = "panel sizes are a few hundred pixels; f32 holds them exactly"
)]
fn number_of(value: &toml::Value) -> Option<f32> {
    match value {
        toml::Value::Float(f) => Some(*f as f32),
        toml::Value::Integer(i) => i32::try_from(*i).ok().map(|i| f64::from(i) as f32),
        _ => None,
    }
}

/// Where `layout.toml` lives: next to `settings.toml`.
///
/// `None` when the platform has no configuration directory. Desktop builds only;
/// the browser build keeps the layout in the platform store ([`load_stored`],
/// [`save_stored`]).
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn layout_path() -> Option<PathBuf> {
    get_config_path()
        .ok()
        .map(|settings| settings.with_file_name(FILE_NAME))
}

/// Where the run log is written: `stacker.log` in the platform data directory.
///
/// Not the working directory, which for a double-clicked app is anywhere. Falls
/// back to the temporary directory when the platform has no home directory.
/// Under the `ui-harness` feature the harness's data directory wins when it set
/// one. Desktop builds only.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn default_log_path() -> PathBuf {
    #[cfg(feature = "ui-harness")]
    if let Some(dir) = slint_harness_hooks::paths::data_dir() {
        return dir.join("stacker.log");
    }
    directories::ProjectDirs::from("", "", "z-stackr")
        .map_or_else(std::env::temp_dir, |dirs| dirs.data_dir().to_path_buf())
        .join("stacker.log")
}

/// Load the layout at `path`. Never fails: a missing or unreadable file gives
/// the defaults.
#[must_use]
pub fn load(path: &Path) -> PanelLayout {
    match fs::read_to_string(path) {
        Ok(text) => PanelLayout::parse(&text),
        Err(e) => {
            if e.kind() != io::ErrorKind::NotFound {
                tracing::warn!("could not read {}: {e}", path.display());
            }
            PanelLayout::default()
        }
    }
}

/// Write `layout` to `path` through the settings store's atomic writer.
///
/// # Errors
///
/// Any I/O error from creating the directory or writing the file.
pub fn save(path: &Path, layout: PanelLayout) -> io::Result<()> {
    store::write_atomic(path, layout.to_toml())
}

/// Load the layout from the platform store (the file next to `settings.toml` on
/// the desktop). Never fails: nothing stored gives the defaults.
#[must_use]
pub fn load_stored() -> PanelLayout {
    crate::platform::current()
        .load_value(FILE_NAME)
        .map_or_else(PanelLayout::default, |text| PanelLayout::parse(&text))
}

/// Write `layout` to the platform store.
///
/// # Errors
///
/// A message when the platform cannot store it.
pub fn save_stored(layout: PanelLayout) -> Result<(), String> {
    crate::platform::current().store_value(FILE_NAME, &layout.to_toml())
}

/// The size a freshly shown window of `current` logical size should take on a
/// monitor of `monitor` logical size.
///
/// At most `max_fraction` of the monitor on each axis, and never larger than it
/// already is (an OS-restored or already-clamped size is respected). `None` when
/// nothing needs to change or the monitor size is unusable.
#[must_use]
pub fn fitted_window_size(
    monitor: (f32, f32),
    current: (f32, f32),
    max_fraction: (f32, f32),
) -> Option<(f32, f32)> {
    let ((monitor_w, monitor_h), (current_w, current_h)) = (monitor, current);
    let usable = |v: f32| v.is_finite() && v > 0.0;
    if !usable(monitor_w) || !usable(monitor_h) {
        return None;
    }
    let width = current_w.min(monitor_w * max_fraction.0).floor();
    let height = current_h.min(monitor_h * max_fraction.1).floor();
    ((width - current_w).abs() > 0.5 || (height - current_h).abs() > 0.5).then_some((width, height))
}
