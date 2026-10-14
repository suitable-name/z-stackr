//! The Web Worker of the z-stackr browser app.
//!
//! Trunk builds this crate as a worker (`data-type="worker"` in
//! `apps/stacker-web/index.html`): wasm-bindgen's `no-modules` output plus a
//! loader shim, which the page starts with `new Worker(...)`.
//!
//! On start the Worker installs its `onmessage` handler and posts
//! [`FromWorker::Ready`](stacker_web_core::FromWorker::Ready). Each message is a
//! postcard-encoded [`ToWorker`](stacker_web_core::ToWorker) in a transferred
//! `ArrayBuffer`; the replies go back the same way, as they happen (progress,
//! live previews, results). The Worker cannot read a message while it computes,
//! so the page cancels a run by terminating the Worker and starting a new one.
//!
//! The Worker is asynchronous: each Run, Align and fetch is a task on the
//! Worker's event loop (`spawn_local`), so the GPU's read-backs can be awaited.
//! A second Run or Align while one is in flight is refused with
//! [`FromWorker::Failed`](stacker_web_core::FromWorker::Failed). Alignment runs
//! on WebGPU only ([`gpu`]); the adapter's name (or its absence) travels in
//! `Ready`.
//!
//! Everything is `#[cfg(target_arch = "wasm32")]`, so a native build of the
//! workspace compiles this crate to nothing.

// WebGPU handles are not `Send`, and the Worker is single-threaded: none of the
// futures here ever leaves its thread.
#![cfg_attr(
    target_arch = "wasm32",
    expect(
        clippy::future_not_send,
        reason = "WebGPU handles are not Send; the worker is single-threaded"
    )
)]

#[cfg(target_arch = "wasm32")]
mod entry;
#[cfg(target_arch = "wasm32")]
mod gpu;
#[cfg(target_arch = "wasm32")]
mod host;
#[cfg(target_arch = "wasm32")]
mod run;
