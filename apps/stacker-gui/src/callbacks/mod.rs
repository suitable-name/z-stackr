//! Slint callback wiring, grouped by domain. Each submodule exposes a
//! `wire(...)` function that registers its group of `app.on_*` callbacks
//! against the shared state handed to it from `crate::app::run`.

pub mod alignment;
pub mod files;
pub mod models;
pub mod project;
pub mod retouch_window;
pub mod saving;
pub mod settings_ui;
pub mod stacking;
