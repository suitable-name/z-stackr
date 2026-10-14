//! The z-stackr browser app.
//!
//! The page runs the desktop's own interface (`stacker_gui::App`) and its wiring on
//! a canvas, and supplies the two things a browser does differently:
//!
//! * [`platform`] implements `stacker_gui::platform::Platform`: files are virtual
//!   `web:/<file name>` paths whose bytes the page holds ([`vfs`]), pickers are
//!   `rfd`'s async ones, a "save" is a download, storage is `localStorage`.
//! * [`runner`] implements `stacker_gui::runner::JobRunner`: every heavy job goes to
//!   a Web Worker ([`worker`]) running the pipeline (`apps/stacker-web-worker`), and
//!   its replies come back as the runner's events.
//!
//! [`app`] is the entry point; [`viewport`], [`dropzone`], [`download`] and [`unload`]
//! are the small browser mechanics around it.
//!
//! Everything is `#[cfg(target_arch = "wasm32")]`, so a native build of the workspace
//! compiles this crate to nothing.

#[cfg(target_arch = "wasm32")]
mod app;
#[cfg(target_arch = "wasm32")]
mod diagnostics;
#[cfg(target_arch = "wasm32")]
mod download;
#[cfg(target_arch = "wasm32")]
mod dropzone;
#[cfg(target_arch = "wasm32")]
mod platform;
#[cfg(target_arch = "wasm32")]
mod runner;
#[cfg(target_arch = "wasm32")]
mod unload;
#[cfg(target_arch = "wasm32")]
mod vfs;
#[cfg(target_arch = "wasm32")]
mod viewport;
#[cfg(target_arch = "wasm32")]
mod worker;
