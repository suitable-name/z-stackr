//! Drag and drop onto the page's `<canvas id="canvas">` (the element Slint renders
//! into): `dragover` is accepted (`preventDefault`, or the browser navigates to the
//! file instead), and `drop` reads every dropped file with `File.arrayBuffer()` -- after
//! checking `File.size`, so an oversize file is never copied into memory --
//! registers them in the virtual file system and hands their virtual paths to
//! `stacker_gui::gui::handle_paths`, exactly like a window-system drop on the desktop.

use stacker_gui::gui::{Ctx, feedback::MessageKind, handle_paths};
use wasm_bindgen::{JsCast, closure::Closure};

use crate::{
    diagnostics::console_warn,
    platform::{oversize_message, report},
    vfs,
};

/// Read every file of a drop, then hand the batch to the GUI.
fn handle_drop(ctx: &Ctx, event: &web_sys::DragEvent) {
    event.prevent_default();
    let Some(list) = event.data_transfer().and_then(|transfer| transfer.files()) else {
        return;
    };
    let files: Vec<web_sys::File> = (0..list.length()).filter_map(|i| list.get(i)).collect();
    if files.is_empty() {
        return;
    }
    let ctx = ctx.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let mut paths = Vec::with_capacity(files.len());
        for file in files {
            let name = file.name();
            if let Some(message) = oversize_message(&name, file.size()) {
                report(MessageKind::Error, &message);
                continue;
            }
            match wasm_bindgen_futures::JsFuture::from(file.array_buffer()).await {
                Ok(buffer) => {
                    let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                    paths.push(vfs::register(&name, bytes));
                }
                Err(_) => report(
                    MessageKind::Error,
                    &format!(
                        "Could not read the dropped file \"{name}\" (a folder cannot be dropped)."
                    ),
                ),
            }
        }
        if !paths.is_empty() {
            handle_paths(&ctx, paths);
        }
    });
}

/// Install the `dragover`/`drop` listeners on `#canvas`. The closures live for the
/// page's lifetime, so they are leaked with `forget()`. A page without the canvas
/// (never the case with this app's `index.html`) just has no drop target.
pub fn install(ctx: &Ctx) {
    let Some(canvas) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("canvas"))
    else {
        console_warn("no #canvas element; drag and drop is unavailable");
        return;
    };

    let on_dragover = Closure::<dyn FnMut(web_sys::DragEvent)>::new(|event: web_sys::DragEvent| {
        event.prevent_default();
        if let Some(transfer) = event.data_transfer() {
            transfer.set_drop_effect("copy");
        }
    });
    let drop_ctx = ctx.clone();
    let on_drop =
        Closure::<dyn FnMut(web_sys::DragEvent)>::new(move |event: web_sys::DragEvent| {
            handle_drop(&drop_ctx, &event);
        });

    let target: &web_sys::EventTarget = canvas.as_ref();
    let added = target
        .add_event_listener_with_callback("dragover", on_dragover.as_ref().unchecked_ref())
        .and_then(|()| {
            target.add_event_listener_with_callback("drop", on_drop.as_ref().unchecked_ref())
        });
    if added.is_err() {
        console_warn("could not install the drag-and-drop listeners");
    }
    on_dragover.forget();
    on_drop.forget();
}
