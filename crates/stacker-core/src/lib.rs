//! Types and services shared by every other crate of the focus-stacking
//! workspace.
//!
//! * [`image`]: [`PlanarImage`](image::PlanarImage), three equally sized planes
//!   (luma and two chroma) of `f32` linear-light samples, the common currency of
//!   alignment, fusion and output.
//! * [`color`]: the sRGB transfer functions, the luma/chroma model, lookup-table
//!   decoders and the SIMD row encoders.
//! * [`memory`]: tile geometry (apron, interior, padded region) and the
//!   file-backed store that lets a stack larger than RAM be fused tile by tile.
//! * [`io`]: the camera RAW extensions and (feature `raw`, native only) the
//!   RAW decoder.
//! * [`metadata`]: EXIF extraction and injection for `copy_metadata`.
//! * [`settings`]: the user-facing option set and its lenient TOML loader.
//! * [`error`]: [`StackerError`](error::StackerError), the error type of the
//!   workspace.
//! * [`telemetry`]: `tracing` initialisation for the applications.
//!
//! The crate needs nightly Rust for `portable_simd` (the SIMD row encoder in
//! [`color`]).

#![feature(portable_simd)]
#![warn(missing_docs)]

/// sRGB transfer functions, the luma/chroma model and the SIMD row encoders.
pub mod color;
/// [`StackerError`](error::StackerError), the workspace error type.
pub mod error;
/// Fused multiply-add helpers that fall back to `a * b + c` on `wasm32`.
pub mod fma;
/// [`PlanarImage`](image::PlanarImage), the three-plane image type.
pub mod image;
/// Camera RAW input: recognised extensions and the `raw`-feature decoder.
pub mod io;
/// Tile geometry and the file-backed tile store for out-of-core fusion.
pub mod memory;
/// EXIF copy-through into JPEG and PNG outputs, and the orientation rewrite.
pub mod metadata;
/// User-facing settings and the lenient TOML loader.
pub mod settings;
/// `tracing` initialisation for stdout and for a log file.
pub mod telemetry;
