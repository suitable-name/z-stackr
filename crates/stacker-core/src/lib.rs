#![feature(portable_simd)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

pub mod color;
pub mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod image;
pub mod io;
pub mod memory;
pub mod metadata;
pub mod preprocessing;
pub mod settings;
pub mod telemetry;
