pub mod args;
pub mod batch;
// The out-of-core tiled pipeline (alignment, fusion, tiling, encode) lives in
// the shared `z-stackr-pipeline` crate so the CLI and GUI call exactly the
// same code path instead of maintaining separate copies that can silently
// drift apart. Re-exporting it under the `pipeline` name keeps every
// `pipeline::...` call site in this crate working unchanged.
// `src/pipeline.rs` and `src/pipeline/*.rs` are not part of the module tree.
pub use stacker_pipeline as pipeline;
