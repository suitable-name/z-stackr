//! Interleaved `[H, W, 3]` pixel buffers (what a numpy array holds) to and from the pipeline's
//! linear-light planar images.
//!
//! # Encodings
//!
//! The pipeline computes in **linear light**. Arrays that come from files, cameras or image
//! libraries are almost always sRGB-encoded, so the default [`Encoding::Srgb`] applies the
//! IEC 61966-2-1 transfer function exactly like the file loader does (the same lookup tables)
//! and encodes the result back with the exact rounding encoders the file writers use:
//!
//! | dtype | meaning of a sample | normalisation |
//! |---|---|---|
//! | `u8` | sRGB code | `0..=255` |
//! | `u16` | sRGB code | `0..=65535` |
//! | `f32` | sRGB-encoded value | nominally `0.0..=1.0` |
//!
//! With [`Encoding::Linear`] the samples are taken as linear light already (scientific data,
//! for example) and no transfer function is applied in either direction.
//!
//! On the way in nothing is clamped, but a non-finite `f32` sample reads as `0`. On the way out
//! every sample is clamped to `0.0..=1.0` and rounded (never truncated) to the integer grid, so
//! an out-of-range float input cannot produce an out-of-range integer.
//!
//! The functions take and return plain slices and `Vec`s; the numpy layer around them is
//! `crate::converter`. Nothing here needs Python.

use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use rayon::prelude::*;
use stacker_core::{
    color::{linear_rgb_to_planar, linear_to_srgb, planar_to_linear_rgb, srgb_to_linear},
    image::PlanarImage,
    settings::{ImageSavingSettings, OutputFormat},
};
use stacker_pipeline::{load::planar_from_dynamic, output::encode};

use crate::options::ArgError;

/// How the samples of an array relate to light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Encoding {
    /// sRGB-encoded samples (the default): the transfer function is applied on the way in and
    /// inverted on the way out.
    #[default]
    Srgb,
    /// Linear-light samples: no transfer function.
    Linear,
}

impl Encoding {
    /// The labels [`Self::parse`] accepts.
    pub const OPTIONS: [&'static str; 2] = ["srgb", "linear"];

    /// Parse `"srgb"` or `"linear"` (any letter case).
    ///
    /// # Errors
    /// [`ArgError`] listing the valid options for anything else.
    pub fn parse(s: &str) -> Result<Self, ArgError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "srgb" => Ok(Self::Srgb),
            "linear" => Ok(Self::Linear),
            _ => Err(ArgError::new(format!(
                "invalid encoding {s:?}; valid options are: {}",
                Self::OPTIONS.join(", ")
            ))),
        }
    }

    /// The label of this encoding.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Srgb => Self::OPTIONS[0],
            Self::Linear => Self::OPTIONS[1],
        }
    }
}

/// The checked sample count of a `width × height × 3` buffer.
fn sample_count(width: usize, height: usize) -> Result<usize, ArgError> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| ArgError::new(format!("image size {width}x{height} is too large")))
}

/// `data.len()` against `width × height × 3`.
fn check_len(len: usize, width: usize, height: usize) -> Result<(), ArgError> {
    let expected = sample_count(width, height)?;
    if len == expected {
        Ok(())
    } else {
        Err(ArgError::new(format!(
            "a {height}x{width}x3 array holds {expected} samples, got {len}"
        )))
    }
}

/// A `u32` image edge.
fn edge(n: usize) -> Result<u32, ArgError> {
    u32::try_from(n).map_err(|_| ArgError::new(format!("image edge {n} is too large")))
}

/// Build a planar image from a function that returns the linear RGB of pixel `i`.
fn build_planar(
    width: usize,
    height: usize,
    linear_rgb_at: impl Fn(usize) -> [f32; 3] + Sync,
) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    img.luma
        .par_iter_mut()
        .zip(img.chroma_a.par_iter_mut())
        .zip(img.chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((luma, chroma_a), chroma_b))| {
            [*luma, *chroma_a, *chroma_b] = linear_rgb_to_planar(linear_rgb_at(i));
        });
    img
}

/// An `f32` sample as linear light: non-finite reads as 0.
fn float_to_linear(v: f32, encoding: Encoding) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    match encoding {
        Encoding::Srgb => srgb_to_linear(v),
        Encoding::Linear => v,
    }
}

// ── in ──────────────────────────────────────────────────────────────────────

/// Convert interleaved 8-bit samples (`r, g, b, r, ...`, `width × height` pixels, row-major).
///
/// # Errors
/// [`ArgError`] when `data` does not hold exactly `width × height × 3` samples or an edge does
/// not fit in `u32`.
pub fn planar_from_u8(
    data: &[u8],
    width: usize,
    height: usize,
    encoding: Encoding,
) -> Result<PlanarImage<f32>, ArgError> {
    check_len(data.len(), width, height)?;
    match encoding {
        Encoding::Srgb => {
            let buffer = RgbImage::from_raw(edge(width)?, edge(height)?, data.to_vec())
                .ok_or_else(|| ArgError::new("the 8-bit buffer does not match its size"))?;
            Ok(planar_from_dynamic(&DynamicImage::ImageRgb8(buffer)))
        }
        Encoding::Linear => Ok(build_planar(width, height, |i| {
            let p = &data[i * 3..i * 3 + 3];
            [
                f32::from(p[0]) / 255.0,
                f32::from(p[1]) / 255.0,
                f32::from(p[2]) / 255.0,
            ]
        })),
    }
}

/// Convert interleaved 16-bit samples (`r, g, b, r, ...`, `width × height` pixels, row-major).
///
/// # Errors
/// [`ArgError`] when `data` does not hold exactly `width × height × 3` samples or an edge does
/// not fit in `u32`.
pub fn planar_from_u16(
    data: &[u16],
    width: usize,
    height: usize,
    encoding: Encoding,
) -> Result<PlanarImage<f32>, ArgError> {
    check_len(data.len(), width, height)?;
    match encoding {
        Encoding::Srgb => {
            let buffer: ImageBuffer<Rgb<u16>, Vec<u16>> =
                ImageBuffer::from_raw(edge(width)?, edge(height)?, data.to_vec())
                    .ok_or_else(|| ArgError::new("the 16-bit buffer does not match its size"))?;
            Ok(planar_from_dynamic(&DynamicImage::ImageRgb16(buffer)))
        }
        Encoding::Linear => Ok(build_planar(width, height, |i| {
            let p = &data[i * 3..i * 3 + 3];
            [
                f32::from(p[0]) / 65_535.0,
                f32::from(p[1]) / 65_535.0,
                f32::from(p[2]) / 65_535.0,
            ]
        })),
    }
}

/// Convert interleaved `f32` samples (`r, g, b, r, ...`, `width × height` pixels, row-major).
///
/// # Errors
/// [`ArgError`] when `data` does not hold exactly `width × height × 3` samples.
pub fn planar_from_f32(
    data: &[f32],
    width: usize,
    height: usize,
    encoding: Encoding,
) -> Result<PlanarImage<f32>, ArgError> {
    check_len(data.len(), width, height)?;
    Ok(build_planar(width, height, |i| {
        let p = &data[i * 3..i * 3 + 3];
        [
            float_to_linear(p[0], encoding),
            float_to_linear(p[1], encoding),
            float_to_linear(p[2], encoding),
        ]
    }))
}

// ── out ─────────────────────────────────────────────────────────────────────

/// `v` clamped to `0.0..=1.0`, NaN as 0.
const fn unit(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

/// The samples of `img` as linear RGB, each passed through `quantise`.
fn linear_samples<T: Copy + Default + Send>(
    img: &PlanarImage<f32>,
    quantise: impl Fn(f32) -> T + Sync,
) -> Vec<T> {
    let mut out = vec![T::default(); img.luma.len() * 3];
    out.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
        let rgb = planar_to_linear_rgb([img.luma[i], img.chroma_a[i], img.chroma_b[i]]);
        for (slot, value) in px.iter_mut().zip(rgb) {
            *slot = quantise(value);
        }
    });
    out
}

/// The `image_saving` settings that make [`encode`] produce a `bit_depth`-bit buffer.
fn raw_saving(bit_depth: u32) -> ImageSavingSettings {
    ImageSavingSettings {
        output_format: OutputFormat::Png,
        bit_depth,
        ..ImageSavingSettings::default()
    }
}

/// Convert `img` to interleaved 8-bit samples (row-major, `r, g, b, r, ...`).
///
/// With [`Encoding::Srgb`] this is the exact encoder of the file writers (linear white is
/// 255).
#[must_use]
pub fn planar_to_u8(img: &PlanarImage<f32>, encoding: Encoding) -> Vec<u8> {
    match encoding {
        Encoding::Srgb => encode(img, &raw_saving(8))
            .rgb8()
            .map_or_else(Vec::new, <[u8]>::to_vec),
        Encoding::Linear => linear_samples(img, |v| quantise_u8(unit(v))),
    }
}

/// Convert `img` to interleaved 16-bit samples (row-major, `r, g, b, r, ...`).
///
/// With [`Encoding::Srgb`] this is the exact encoder of the file writers (linear white is
/// 65 535).
#[must_use]
pub fn planar_to_u16(img: &PlanarImage<f32>, encoding: Encoding) -> Vec<u16> {
    match encoding {
        Encoding::Srgb => encode(img, &raw_saving(16))
            .rgb16()
            .map_or_else(Vec::new, <[u16]>::to_vec),
        Encoding::Linear => linear_samples(img, |v| quantise_u16(unit(v))),
    }
}

/// Convert `img` to interleaved `f32` samples in `0.0..=1.0` (row-major).
#[must_use]
pub fn planar_to_f32(img: &PlanarImage<f32>, encoding: Encoding) -> Vec<f32> {
    match encoding {
        Encoding::Srgb => linear_samples(img, |v| linear_to_srgb(unit(v))),
        Encoding::Linear => linear_samples(img, unit),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the argument is in 0..=1, so the rounded product is in 0..=255"
)]
fn quantise_u8(v: f32) -> u8 {
    (v * 255.0).round() as u8
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the argument is in 0..=1, so the rounded product is in 0..=65535"
)]
fn quantise_u16(v: f32) -> u16 {
    (v * 65_535.0).round() as u16
}
