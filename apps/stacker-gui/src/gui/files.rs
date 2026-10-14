//! The source and result lists: adding, removing, selecting and showing images.
//!
//! # Showing an image
//!
//! A click never decodes on the UI thread. It records which file is on view (in
//! the state, so the Show-Aligned toggle and Save know), hands a request to the
//! job runner (a source file, or its aligned frame) or to the viewer thread (a
//! result file, and the conversion of an aligned frame), and returns. A request
//! that was superseded while it ran is dropped (a [`Serial`] token per request),
//! and the viewer thread keeps only the newest waiting request
//! ([`LatestWorker`]), so a burst of clicks shows only the last file. The
//! frame reaches the canvas through [`preview::submit_frame`], and only while its
//! file is still the one on view: a stack result published in the meantime is not
//! overwritten by a late decode.

use std::{cell::RefCell, path::PathBuf, sync::Arc};

use slint::{ComponentHandle, Rgba8Pixel, SharedPixelBuffer, Weak};
use stacker_core::image::PlanarImage;
use stacker_gui::{
    image_utils::planar_to_rgba_buffer,
    latest_worker::LatestWorker,
    pickers::{Picked, PickerKind, PickerRequest},
    preview::{Serial, decode_bytes},
    runner,
    stall_guard::stall_guard,
};

use super::{
    Ctx,
    confirm::{self, Prompt},
    ctx::on_ui_ctx,
    feedback::{self, MessageKind},
    pickers, preview,
};
use crate::{App, ui_helpers::push_source_panel};

/// File extensions the app opens as source frames: the pipeline's own list, so
/// the picker filter, the drop filter and the stack run agree.
pub use stacker_pipeline::order::FRAME_EXTENSIONS as IMAGE_EXTENSIONS;

/// The picker's filter: [`IMAGE_EXTENSIONS`] plus the camera RAW extensions of a
/// build that decodes them (none otherwise).
fn picker_extensions() -> Vec<&'static str> {
    IMAGE_EXTENSIONS
        .iter()
        .chain(stacker_pipeline::load::RAW_FRAME_EXTENSIONS)
        .copied()
        .collect()
}

/// Register the source and result list callbacks and start the viewer thread.
pub fn setup(ui: &App, ctx: &Ctx) {
    install_viewer(ui.as_weak());
    wire_open_files(ui, ctx);
    wire_source_click(ui, ctx);
    wire_refresh_view(ui, ctx);
    wire_result_click(ui, ctx);
    wire_remove_file(ui, ctx);
    wire_clear_all(ui, ctx);
}

/// A Slint list index as `usize`; a negative index maps to `usize::MAX`, which
/// no list contains.
fn list_index(idx: i32) -> usize {
    usize::try_from(idx).unwrap_or(usize::MAX)
}

// ── The viewer: decode off the UI thread ─────────────────────────────────────

/// What the viewer thread does for one request.
enum Work {
    /// Read the result file `path` through the platform and decode it.
    Stored,
    /// Convert the aligned frame (already cropped) to a display buffer.
    Planar(PlanarImage<f32>),
}

/// One image to put on the canvas.
struct Request {
    /// Identifies the request among the newer ones (see [`Serial`]).
    token: u64,
    /// The file on view; the state's `viewed_path` must still equal it when the
    /// frame arrives.
    path: PathBuf,
    work: Work,
}

/// Where the pixels of a view come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewKind {
    /// A source file: the runner decodes it.
    Source,
    /// The aligned frame of a source (Show Aligned): the runner serves it.
    Aligned,
    /// A result file: read through the platform on the viewer thread.
    Result,
}

/// The decoder thread and the counter that supersedes its requests.
struct Viewer {
    worker: LatestWorker<Request>,
    serial: Arc<Serial>,
}

thread_local! {
    /// The viewer of the window [`setup`] wired last, on the UI thread.
    static VIEWER: RefCell<Option<Viewer>> = const { RefCell::new(None) };
}

fn install_viewer(ui: Weak<App>) {
    let serial = Arc::new(Serial::new());
    let decoder_serial = Arc::clone(&serial);
    let worker = LatestWorker::spawn("z-stackr-viewer", move |request: Request| {
        decode_and_publish(&ui, &decoder_serial, request);
    });
    // A second window on this thread (the UI flow tests boot one per scenario)
    // replaces the viewer; the old one's thread ends with it.
    VIEWER.with(|cell| *cell.borrow_mut() = Some(Viewer { worker, serial }));
}

/// End the viewer thread of the window wired last and wait for it. For the UI
/// harness, which must not leave a thread that can still read files of a profile
/// it is about to delete. A later [`setup`] starts a new viewer.
#[cfg(feature = "ui-harness")]
pub fn stop_viewer() {
    let viewer = VIEWER.with(|cell| cell.borrow_mut().take());
    if let Some(viewer) = viewer {
        viewer.worker.stop();
    }
}

/// Put `path` on the canvas, as `kind` says. Returns at once; UI thread only.
fn request_view(ctx: &Ctx, path: PathBuf, kind: ViewKind) {
    let Some((token, serial)) = VIEWER.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|viewer| (viewer.serial.begin(), Arc::clone(&viewer.serial)))
    }) else {
        return;
    };
    match kind {
        ViewKind::Result => submit_work(token, path, Work::Stored),
        ViewKind::Source => {
            let ui = ctx.ui.clone();
            let shown = path.clone();
            runner::current().fetch_preview(
                &path,
                Box::new(move |outcome| {
                    // A newer click arrived while this one decoded.
                    if serial.is_current(token) {
                        publish(&ui, &shown, outcome);
                    }
                }),
            );
        }
        ViewKind::Aligned => {
            let settings = ctx.settings.snapshot();
            let ui = ctx.ui.clone();
            let shown = path.clone();
            runner::current().fetch_source(
                &path,
                &settings,
                Box::new(move |outcome| {
                    if !serial.is_current(token) {
                        return;
                    }
                    match outcome {
                        Ok(frame) => submit_work(token, shown, Work::Planar(frame)),
                        Err(reason) => publish(&ui, &shown, Err(reason)),
                    }
                }),
            );
        }
    }
}

/// Hand `work` for `path` to the viewer thread.
fn submit_work(token: u64, path: PathBuf, work: Work) {
    VIEWER.with(|cell| {
        if let Some(viewer) = cell.borrow().as_ref() {
            let _ = viewer.worker.submit(Request { token, path, work });
        }
    });
}

/// The viewer thread's body for one request.
fn decode_and_publish(ui: &Weak<App>, serial: &Serial, request: Request) {
    if !serial.is_current(request.token) {
        return;
    }
    let Request { token, path, work } = request;
    let outcome = match work {
        Work::Planar(frame) => Ok(planar_to_rgba_buffer(&frame)),
        Work::Stored => crate::platform::current()
            .read_bytes(&path)
            .map_err(|e| e.to_string())
            .and_then(|bytes| decode_bytes(&bytes)),
    };
    if !serial.is_current(token) {
        // A newer click arrived while this one decoded.
        return;
    }
    let ui = ui.clone();
    // Back on the UI thread (inline on a host without threads).
    on_ui_ctx(&ui.clone(), move |_app, _ctx| publish(&ui, &path, outcome));
}

/// Show the decoded `outcome` for `path` on the canvas, if `path` is still the
/// file on view; tell the user when it could not be read. UI thread only.
fn publish(
    ui: &Weak<App>,
    path: &std::path::Path,
    outcome: Result<SharedPixelBuffer<Rgba8Pixel>, String>,
) {
    match outcome {
        Ok(pixels) => {
            let viewing = Ctx::current()
                .is_some_and(|ctx| ctx.state.borrow().viewed_path.as_deref() == Some(path));
            if viewing {
                preview::submit_frame(pixels);
            }
        }
        Err(reason) => {
            tracing::warn!("cannot display {}: {reason}", path.display());
            if let Some(app) = ui.upgrade() {
                feedback::show(
                    &app,
                    MessageKind::Warning,
                    &format!(
                        "Could not display {}: the file cannot be read as an image.",
                        feedback::file_label(path)
                    ),
                );
            }
        }
    }
}

// ── Callbacks ────────────────────────────────────────────────────────────────

fn wire_open_files(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_request_open_files(move || {
        stall_guard("files::request-open-files", || {
            let request = PickerRequest::new(PickerKind::OpenFiles)
                .titled("Add images")
                .filter("Images", &picker_extensions());
            pickers::pick(&ctx, request, add_picked_files);
        });
    });
}

/// Add `files` to the source list (duplicates skipped), show the first source and
/// tell the user how many were already there. File > Add Images and a drop of
/// images both end here.
pub(super) fn add_picked_files(app: &App, ctx: &Ctx, files: Picked) {
    stall_guard("files::add-picked-files", || {
        if files.is_empty() {
            return;
        }
        let offered = files.len();
        let (list, first, added) = {
            let mut state = ctx.state.borrow_mut();
            let added = state.add_paths(files);
            let first = state.file_paths.first().cloned();
            state.viewed_path.clone_from(&first);
            (state.file_paths.clone(), first, added)
        };
        push_source_panel(app, &list);
        super::project::note_sources_changed(ctx);
        if added < offered {
            feedback::show(
                app,
                MessageKind::Info,
                &format!(
                    "{} image(s) were already in the list and were skipped.",
                    offered - added
                ),
            );
        }
        if let Some(first) = first {
            request_view(ctx, first, ViewKind::Source);
        }
    });
}

/// Put the source frame `path` on the canvas: its aligned version when
/// "Show Aligned" is on and the Align pass produced one, else the file itself.
fn show_source(app: &App, ctx: &Ctx, path: PathBuf) {
    let aligned = app.get_show_aligned()
        && ctx
            .state
            .borrow()
            .aligned
            .as_ref()
            .is_some_and(|cache| cache.paths.contains(&path));
    let kind = if aligned {
        ViewKind::Aligned
    } else {
        ViewKind::Source
    };
    request_view(ctx, path, kind);
}

fn wire_source_click(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_file_clicked(move |idx| {
        stall_guard("files::file-clicked", || {
            let path = ctx.state.borrow_mut().select_source(list_index(idx));
            if let (Some(app), Some(path)) = (ctx.ui.upgrade(), path) {
                show_source(&app, &ctx, path);
            }
        });
    });
}

/// The Show-Aligned toggle: re-render the current frame in place.
fn wire_refresh_view(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_refresh_view(move || {
        stall_guard("files::refresh-view", || {
            let Some(app) = ctx.ui.upgrade() else {
                return;
            };
            let path = {
                let state = ctx.state.borrow();
                state
                    .view_idx
                    .and_then(|idx| state.file_paths.get(idx).cloned())
            };
            if let Some(path) = path {
                show_source(&app, &ctx, path);
            }
        });
    });
}

fn wire_result_click(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_result_clicked(move |idx| {
        stall_guard("files::result-clicked", || {
            let path = ctx.state.borrow_mut().select_result(list_index(idx));
            if let Some(path) = path {
                request_view(&ctx, path, ViewKind::Result);
            }
        });
    });
}

fn wire_remove_file(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_remove_file(move |idx| {
        stall_guard("files::remove-file", || {
            let list = {
                let mut state = ctx.state.borrow_mut();
                state
                    .remove_path(list_index(idx))
                    .map(|_| state.file_paths.clone())
            };
            if let (Some(app), Some(list)) = (ctx.ui.upgrade(), list) {
                push_source_panel(&app, &list);
                super::project::note_sources_changed(&ctx);
            }
        });
    });
}

fn wire_clear_all(ui: &App, ctx: &Ctx) {
    let ctx = ctx.clone();
    ui.on_clear_all_files(move || {
        stall_guard("files::clear-all-files", || {
            let count = ctx.state.borrow().file_paths.len();
            if count == 0 {
                return;
            }
            let message = format!(
                "Remove all {count} image(s) from the list? The files on disk are not touched."
            );
            let prompt = Prompt {
                action: "clear-sources",
                heading: "Clear the image list?",
                message: &message,
                primary_label: "Clear",
            };
            confirm::confirm(&ctx, &prompt, |app, ctx| {
                ctx.state.borrow_mut().clear_paths();
                push_source_panel(app, &[]);
                super::project::note_sources_changed(ctx);
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use stacker_pipeline::order::{FRAME_EXTENSIONS, has_frame_extension};

    use super::IMAGE_EXTENSIONS;

    #[test]
    fn the_picker_list_is_the_pipelines_list() {
        assert_eq!(IMAGE_EXTENSIONS, FRAME_EXTENSIONS);
        assert!(IMAGE_EXTENSIONS.contains(&"png") && IMAGE_EXTENSIONS.contains(&"tiff"));
    }

    #[test]
    fn every_picker_extension_is_accepted_by_the_pipeline_in_any_case() {
        for extension in IMAGE_EXTENSIONS {
            assert!(has_frame_extension(Path::new(&format!(
                "frame.{extension}"
            ))));
            let upper = extension.to_ascii_uppercase();
            assert!(has_frame_extension(Path::new(&format!("FRAME.{upper}"))));
        }
        assert!(!has_frame_extension(Path::new("notes.txt")));
    }
}
