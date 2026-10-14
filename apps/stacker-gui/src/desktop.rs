//! The desktop entry point: logging to the run-log file, the window, the event
//! loop. `main.rs` only picks the allocator and calls [`run_desktop`].

use slint::ComponentHandle;
use stacker_core::telemetry::init_tracing_to_file;

use crate::{App, gui, settings::layout::default_log_path};

/// Start the desktop app and run it until its window is closed. The settings
/// written inside the debounce window are flushed on the way out.
///
/// # Errors
///
/// Slint's platform error when the window cannot be created or the event loop
/// fails.
pub fn run_desktop() -> Result<(), slint::PlatformError> {
    let log_path = default_log_path();
    if let Err(e) = init_tracing_to_file(&log_path, "info") {
        eprintln!("warning: file logging init failed ({e}); continuing without a log file");
    }
    eprintln!("stacker: logging to {}", log_path.display());

    let app = App::new()?;
    let ctx = gui::bootstrap(&app);
    app.run()?;

    // The window is closed: write any change still inside the debounce window.
    ctx.settings.flush();
    Ok(())
}
