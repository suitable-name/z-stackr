//! Console diagnostics and the fatal-error screen.

/// Install the panic hook: a Rust panic as a readable `console.error` line
/// instead of the browser's opaque "unreachable executed" trap. Call first.
pub fn install() {
    console_error_panic_hook::set_once();
}

/// Replace the (blank) page with `message`, for a failure before the app's own
/// window exists (no WebGL, a missing canvas) where nothing else could tell the
/// reader why the page is empty. The message is plain text.
pub fn show_fatal(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let (Some(body), Ok(note)) = (document.body(), document.create_element("div")) else {
        return;
    };
    note.set_text_content(Some(message));
    let _ = note.set_attribute(
        "style",
        "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;\
         padding:24px;box-sizing:border-box;text-align:center;font:16px sans-serif;\
         color:#f1f5f9;background:#1b1b1f;",
    );
    let _ = body.append_child(&note);
}

/// One `z-stackr-web: <message>` line on `console.warn`.
pub fn console_warn(message: &str) {
    web_sys::console::warn_1(&format!("z-stackr-web: {message}").into());
}
