//! The Worker's entry point: hook up `onmessage`, open WebGPU, announce readiness.

use js_sys::{ArrayBuffer, Uint8Array};
use stacker_web_core::{FromWorker, PROTOCOL_VERSION, ToWorker, decode};
use wasm_bindgen::{JsCast, closure::Closure, prelude::wasm_bindgen};
use wasm_bindgen_futures::spawn_local;
use web_sys::MessageEvent;

use crate::{
    gpu,
    host::{post, scope},
    run,
};

fn on_message(event: &MessageEvent) {
    match event.data().dyn_into::<ArrayBuffer>() {
        Ok(buffer) => match decode::<ToWorker>(&Uint8Array::new(&buffer).to_vec()) {
            Ok(message) => run::handle(message),
            Err(message) => post(&FromWorker::Failed { job_id: 0, message }),
        },
        Err(_) => post(&FromWorker::Failed {
            job_id: 0,
            message: "a worker message was not an ArrayBuffer".to_owned(),
        }),
    }
}

/// Runs when the module is instantiated: hook up `onmessage`, then say so.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let closure = Closure::<dyn FnMut(MessageEvent)>::new(|event: MessageEvent| {
        on_message(&event);
    });
    scope().set_onmessage(Some(closure.as_ref().unchecked_ref()));
    // The handler lives as long as the Worker.
    closure.forget();
    // Open WebGPU first (an async request), then announce readiness with the
    // adapter's name, or `None` when the browser offers none.
    spawn_local(async {
        let gpu_adapter = gpu::init().await;
        post(&FromWorker::Ready {
            version: PROTOCOL_VERSION,
            gpu_adapter,
        });
    });
}
