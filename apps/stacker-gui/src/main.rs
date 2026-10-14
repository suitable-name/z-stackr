//! The desktop app. The Slint window, every callback and the platform seam live in
//! the `stacker_gui` library (`gui/`, one module per feature); this file is the
//! allocator and the call into [`stacker_gui::desktop::run_desktop`].
#![windows_subsystem = "windows"]

// Custom global allocator to reduce allocation churn. jemalloc's sys crate has
// no working build on Windows MSVC (autotools/sh based), so we use mimalloc
// there and jemalloc on every other platform (Linux/macOS).
#[cfg(all(not(target_os = "windows"), not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(target_os = "windows")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), slint::PlatformError> {
    stacker_gui::desktop::run_desktop()
}

/// The browser build is `apps/stacker-web`; this binary has nothing to run there.
#[cfg(target_arch = "wasm32")]
fn main() {}
