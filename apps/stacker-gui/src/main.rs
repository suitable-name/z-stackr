#![windows_subsystem = "windows"]

// Custom global allocator to reduce allocation churn. jemalloc's sys crate has
// no working build on Windows MSVC (autotools/sh based), so we use mimalloc
// there and jemalloc on every other platform (Linux/macOS).
#[cfg(not(target_os = "windows"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(target_os = "windows")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> Result<(), slint::PlatformError> {
    stacker_gui::app::run()
}
