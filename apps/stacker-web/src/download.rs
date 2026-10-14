//! Browser downloads: a `Blob` over the bytes, an object URL, and a synthetic
//! `<a download>` click. The anchor is attached for the click (Firefox ignores a
//! click on a detached anchor) and removed at once; the object URL is revoked a
//! little later, once the browser has started reading it.

use std::time::Duration;

use wasm_bindgen::{JsCast, JsValue};

/// How long an object URL stays valid after its click.
const REVOKE_AFTER: Duration = Duration::from_secs(30);

/// The MIME type a download named `file_name` is offered with.
pub fn mime_for(file_name: &str) -> &'static str {
    let lower = file_name.to_ascii_lowercase();
    let extension = lower
        .rsplit_once('.')
        .map_or("", |(_, extension)| extension);
    match extension {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "tif" | "tiff" => "image/tiff",
        "toml" => "application/toml",
        "txt" | "log" => "text/plain;charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn js_error(context: &str, value: &JsValue) -> String {
    format!(
        "{context}: {}",
        value.as_string().unwrap_or_else(|| format!("{value:?}"))
    )
}

/// Offer `bytes` to the user as the download `file_name`.
///
/// # Errors
///
/// A readable message when the page has no document or body, or the browser
/// refuses to build the `Blob` or its URL.
pub fn download_bytes(file_name: &str, mime: &str, bytes: &[u8]) -> Result<(), String> {
    let window = web_sys::window().ok_or("No browser window to download from.")?;
    let document = window.document().ok_or("No document to download from.")?;
    let body = document
        .body()
        .ok_or("No document body to download from.")?;

    let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(bytes));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)
        .map_err(|e| js_error("Could not build the download", &e))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| js_error("Could not create the download URL", &e))?;

    let anchor: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .map_err(|e| js_error("Could not create the download link", &e))?
        .dyn_into()
        .map_err(|_| "The download link is not an anchor.".to_owned())?;
    anchor.set_href(&url);
    anchor.set_download(file_name);
    body.append_child(&anchor)
        .map_err(|e| js_error("Could not attach the download link", &e))?;
    anchor.click();
    let _ = body.remove_child(&anchor);

    slint::Timer::single_shot(REVOKE_AFTER, move || {
        let _ = web_sys::Url::revoke_object_url(&url);
    });
    Ok(())
}
