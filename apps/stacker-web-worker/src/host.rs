//! The Worker's view of the browser: the global scope, the clock, posting.

use js_sys::{Array, Uint8Array};
use stacker_web_core::{FromWorker, encode};
use wasm_bindgen::JsCast;
use web_sys::DedicatedWorkerGlobalScope;

/// The Worker's global scope.
pub fn scope() -> DedicatedWorkerGlobalScope {
    js_sys::global().unchecked_into()
}

/// Milliseconds on the Worker's own clock (`performance.now()`); `std::time`
/// panics on `wasm32-unknown-unknown`.
pub fn now_ms() -> f64 {
    scope().performance().map_or(0.0, |p| p.now())
}

/// Post `message` to the page as a transferred `ArrayBuffer`.
pub fn post(message: &FromWorker) {
    let bytes = encode(message);
    let buffer = Uint8Array::from(bytes.as_slice()).buffer();
    if let Err(error) = scope().post_message_with_transfer(&buffer, &Array::of1(&buffer)) {
        web_sys::console::error_2(&"z-stackr worker: postMessage failed".into(), &error);
    }
}
