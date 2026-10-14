//! One Web Worker: spawn it from [`LOADER_URL`], post postcard bytes in a
//! transferred `ArrayBuffer`, decode its replies.

use std::{cell::RefCell, rc::Rc, time::Duration};

use js_sys::{Array, ArrayBuffer, Uint8Array};
use stacker_web_core::{FromWorker, ToWorker, decode, encode};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{ErrorEvent, MessageEvent, Worker};

/// The loader Trunk writes for the worker package `z-stackr-web-worker`
/// (`data-loader-shim` in `index.html`), next to the page's own files.
pub const LOADER_URL: &str = "./z-stackr-web-worker_loader.js";

/// How long a terminated worker's callbacks stay alive before they are freed (a
/// handle must never be dropped from inside one of its own callbacks).
const RETIRE_AFTER: Duration = Duration::from_secs(1);

/// A live Worker and the two JS closures bound to it.
pub struct WorkerHandle {
    worker: Worker,
    _on_message: Closure<dyn FnMut(MessageEvent)>,
    _on_error: Closure<dyn FnMut(ErrorEvent)>,
}

impl WorkerHandle {
    /// Start a Worker. `on_message` gets every decoded reply; `on_error` gets a
    /// description of an undecodable reply or of the Worker's own `error` event (a
    /// script load failure or an uncaught panic).
    ///
    /// # Errors
    ///
    /// The browser's message when it refuses to start the Worker.
    pub fn spawn(
        mut on_message: impl FnMut(FromWorker) + 'static,
        on_error: impl FnMut(String) + 'static,
    ) -> Result<Self, String> {
        let worker = Worker::new(LOADER_URL).map_err(|e| {
            format!(
                "could not start the worker from {LOADER_URL}: {}",
                js_error_text(&e)
            )
        })?;
        let on_error = Rc::new(RefCell::new(on_error));
        let decode_error = Rc::clone(&on_error);
        let on_message_closure = Closure::<dyn FnMut(MessageEvent)>::new(
            move |event: MessageEvent| match decode_event(&event) {
                Ok(message) => on_message(message),
                Err(error) => (decode_error.borrow_mut())(error),
            },
        );
        worker.set_onmessage(Some(on_message_closure.as_ref().unchecked_ref()));
        let on_error_closure = Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
            event.prevent_default();
            (on_error.borrow_mut())(format!(
                "worker failed: {} ({}:{})",
                event.message(),
                event.filename(),
                event.lineno()
            ));
        });
        worker.set_onerror(Some(on_error_closure.as_ref().unchecked_ref()));
        Ok(Self {
            worker,
            _on_message: on_message_closure,
            _on_error: on_error_closure,
        })
    }

    /// Post `message`, transferring its buffer.
    ///
    /// # Errors
    ///
    /// The browser's message when it cannot post (a message beyond 4 GiB, a dead
    /// worker).
    pub fn post(&self, message: &ToWorker) -> Result<(), String> {
        let bytes = encode(message);
        let length = u32::try_from(bytes.len())
            .map_err(|_| "the message to the worker is larger than 4 GiB".to_owned())?;
        let array = Uint8Array::new_with_length(length);
        array.copy_from(&bytes);
        let buffer = array.buffer();
        self.worker
            .post_message_with_transfer(&buffer, &Array::of1(&buffer))
            .map_err(|e| format!("posting to the worker: {}", js_error_text(&e)))
    }

    /// Stop the Worker at once, unhook its callbacks and free the handle a little
    /// later (so this is safe to call from inside the Worker's own callbacks).
    pub fn retire(self) {
        self.worker.set_onmessage(None);
        self.worker.set_onerror(None);
        self.worker.terminate();
        slint::Timer::single_shot(RETIRE_AFTER, move || drop(self));
    }
}

fn decode_event(event: &MessageEvent) -> Result<FromWorker, String> {
    let buffer: ArrayBuffer = event
        .data()
        .dyn_into()
        .map_err(|_| "a worker reply was not an ArrayBuffer".to_owned())?;
    decode(&Uint8Array::new(&buffer).to_vec())
}

/// A readable text for a thrown JS value.
fn js_error_text(value: &JsValue) -> String {
    value.as_string().unwrap_or_else(|| {
        value
            .dyn_ref::<js_sys::Error>()
            .map_or_else(|| format!("{value:?}"), |e| String::from(e.message()))
    })
}
