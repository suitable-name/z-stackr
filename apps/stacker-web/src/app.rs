//! The wasm entry point.

use std::rc::Rc;

use slint::ComponentHandle;
use stacker_gui::{App, gui, platform};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use crate::{diagnostics, dropzone, platform::WebPlatform, runner, unload, viewport};

/// The entry point, run once when the module has instantiated: install the browser's
/// platform and job runner, create the desktop window on the canvas, wire it, hook
/// the page (viewport, drops, leaving) and start the event loop.
///
/// # Errors
///
/// The window could not be created or its event loop could not start (no canvas or
/// WebGL context); also shown on the page, and surfaced to `console.error` by
/// wasm-bindgen.
#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    diagnostics::install();
    platform::install(Rc::new(WebPlatform::new()));
    runner::install_runner();

    let app = App::new().map_err(|error| {
        diagnostics::show_fatal(&format!(
            "z-stackr could not start: {error}. It needs a browser with WebGL2 enabled \
             (hardware acceleration on); try a current Chrome, Edge, Firefox or Safari."
        ));
        JsValue::from_str(&error.to_string())
    })?;
    let ctx = gui::bootstrap(&app);

    // After `bootstrap`: sizing the window fires layout callbacks that must already
    // be registered.
    viewport::track_browser_viewport(&app);
    dropzone::install(&ctx);
    unload::install(&ctx);

    app.run().map_err(|error| {
        diagnostics::show_fatal(&format!("z-stackr stopped: {error}. Reload the page."));
        JsValue::from_str(&error.to_string())
    })?;
    // A closed window (never in a browser) still writes the last settings.
    ctx.settings.flush();
    Ok(())
}
