//! `stacker-gui` library crate root.
//!
//! This crate is a graphical, Slint-based alternative to the `z-stackr` CLI:
//! `src/main.rs` is now a thin binary shim (global allocator statics plus a
//! `stacker_gui::app::run()` call) — every other module lives here so the
//! generated Slint `App` type (via [`slint::include_modules!`], invoked
//! below) and all the supporting logic can be split across focused files
//! while still naming shared types through `crate::`.
#![allow(
    clippy::suboptimal_flops,           // colour-matrix coefficients; mul_add form harms readability
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

slint::include_modules!();

pub mod align_cache;
pub mod app;
pub mod callbacks;
pub mod image_utils;
pub mod mem_estimate;
pub mod nn_bridge;
pub mod project;
pub mod retouch;
pub mod save;
pub mod settings;
pub mod sort_cull;
pub mod stack_pipeline;
pub mod stacking;
pub mod ui_helpers;
