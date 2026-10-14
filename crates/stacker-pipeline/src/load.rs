//! Decoding one frame: file → EXIF orientation → rotation → crop → resize →
//! linear-light planar `f32`.
//!
//! This is the only decoder. The command-line tool and the desktop app both call
//! [`decode_frame`], so the same settings file yields the same pixels in both.
//!
//! # Colour model
//!
//! All arithmetic downstream is done in **linear light** (not gamma-encoded
//! sRGB):
//!
//! 1. Samples are normalised to `[0, 1]` (÷255, or ÷65 535 for 16-bit sources;
//!    the precision of a 16-bit file is kept).
//! 2. The IEC 61966-2-1 sRGB decode is applied through the exact 256- or
//!    65 536-entry tables of `stacker_core::color`.
//! 3. The result is stored as `luma` / `chroma_a` / `chroma_b` with
//!    [`stacker_core::color::LUMA_CHROMA_FORWARD`] (BT.601 on linear RGB).
//!
//! Alpha is dropped and no ICC profile is applied.
//!
//! # Camera RAW
//!
//! With the `raw` feature on a native target, [`decode_frame`] also reads camera
//! RAW files ([`RAW_FRAME_EXTENSIONS`]): `stacker_core::io::decode_raw_file`
//! develops them (camera white balance, colour matrix, sRGB curve) into a 16-bit
//! sRGB image, which then follows the path of any 16-bit file above, so the
//! rest of the pipeline sees linear light. The EXIF orientation is not applied
//! to RAW pixels. Without the feature a RAW path is a decode error that says how
//! to enable RAW. RAW files cannot be decoded from memory
//! ([`decode_frame_bytes`]).
//!
//! # Preprocessing order
//!
//! EXIF orientation (unless `ignore_exif_orientation`), then `pre_rotation`,
//! then the crop, then the resize. Each is the identity at its default.

use std::{borrow::Cow, path::Path};

use image::{ColorType, DynamicImage, GenericImageView};
use rayon::prelude::*;
use stacker_core::{
    color::{linear_rgb_to_planar, srgb_decode_u8_table, srgb_decode_u16_table},
    image::PlanarImage,
    settings::PreprocessingSettings,
};

use crate::PipelineError;

pub use stacker_core::io::{RAW_EXTENSIONS, RAW_SUPPORTED, has_raw_extension, is_raw_extension};

/// The RAW extensions this build reads as frames: all of [`RAW_EXTENSIONS`] when
/// [`RAW_SUPPORTED`], none otherwise (so a directory scan of a build without RAW
/// does not list files it cannot decode).
pub const RAW_FRAME_EXTENSIONS: &[&str] = if RAW_SUPPORTED { RAW_EXTENSIONS } else { &[] };

/// Decode `path` and bring it into the pipeline's working representation.
///
/// A path with a RAW extension goes through the RAW decoder (see the module
/// documentation); every other file is decoded by the `image` crate.
///
/// # Errors
/// [`PipelineError::Decode`], naming `path`, when the file cannot be opened or
/// decoded, or is a RAW file and this build has no RAW support.
pub fn decode_frame(
    path: &Path,
    pre: &PreprocessingSettings,
) -> Result<PlanarImage<f32>, PipelineError> {
    if has_raw_extension(path) {
        let decoded =
            stacker_core::io::decode_raw_file(path).map_err(|e| PipelineError::decode(path, e))?;
        return Ok(planar_from_dynamic(&preprocess_frame(decoded, pre)));
    }
    let bytes = std::fs::read(path).map_err(|e| PipelineError::decode(path, e))?;
    decode_bytes(&bytes, path, pre)
}

/// Decode a frame from its encoded file contents.
///
/// `bytes` is a JPEG, PNG or TIFF file (the container is recognised from the
/// leading bytes, not from `name`). The steps and their order are those of
/// [`decode_frame`], which reads the file and calls this.
///
/// `name` is only used in error messages.
///
/// # Errors
/// [`PipelineError::Decode`], naming `name`, when the bytes cannot be decoded.
pub fn decode_frame_bytes(
    bytes: &[u8],
    name: &str,
    pre: &PreprocessingSettings,
) -> Result<PlanarImage<f32>, PipelineError> {
    decode_bytes(bytes, Path::new(name), pre)
}

/// The one decoding path behind [`decode_frame`] and [`decode_frame_bytes`].
fn decode_bytes(
    bytes: &[u8],
    label: &Path,
    pre: &PreprocessingSettings,
) -> Result<PlanarImage<f32>, PipelineError> {
    let decoded = image::load_from_memory(bytes).map_err(|e| PipelineError::decode(label, e))?;
    let oriented = if pre.ignore_exif_orientation {
        decoded
    } else {
        let head = &bytes[..bytes.len().min(EXIF_HEAD_BYTES)];
        match jpeg_exif_orientation(head) {
            Some(orientation) => apply_orientation(decoded, orientation),
            None => decoded,
        }
    };
    Ok(planar_from_dynamic(&preprocess_frame(oriented, pre)))
}

/// Decode `paths` in parallel. Result `i` belongs to `paths[i]`; a frame that
/// fails to decode is an `Err` in its own slot and does not stop the others.
#[must_use]
pub fn decode_frames(
    paths: &[std::path::PathBuf],
    pre: &PreprocessingSettings,
) -> Vec<Result<PlanarImage<f32>, PipelineError>> {
    paths.par_iter().map(|p| decode_frame(p, pre)).collect()
}

/// Convert a decoded image to the planar linear-light working representation
/// (see the module documentation).
#[must_use]
pub fn planar_from_dynamic(img: &DynamicImage) -> PlanarImage<f32> {
    let (w, h) = img.dimensions();
    let mut planar = PlanarImage::new(w as usize, h as usize);

    let is_16bit = matches!(
        img.color(),
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
    );

    if is_16bit {
        let dec = srgb_decode_u16_table();
        // Borrow the samples when the file already is RGB16 (no copy).
        let rgb16 = match img {
            DynamicImage::ImageRgb16(buf) => Cow::Borrowed(buf),
            other => Cow::Owned(other.to_rgb16()),
        };
        for (i, p) in rgb16.pixels().enumerate() {
            let rgb = [
                dec[usize::from(p[0])],
                dec[usize::from(p[1])],
                dec[usize::from(p[2])],
            ];
            [planar.luma[i], planar.chroma_a[i], planar.chroma_b[i]] = linear_rgb_to_planar(rgb);
        }
    } else {
        let dec = srgb_decode_u8_table();
        let rgb8 = match img {
            DynamicImage::ImageRgb8(buf) => Cow::Borrowed(buf),
            other => Cow::Owned(other.to_rgb8()),
        };
        for (i, p) in rgb8.pixels().enumerate() {
            let rgb = [
                dec[usize::from(p[0])],
                dec[usize::from(p[1])],
                dec[usize::from(p[2])],
            ];
            [planar.luma[i], planar.chroma_a[i], planar.chroma_b[i]] = linear_rgb_to_planar(rgb);
        }
    }

    planar
}

// ── Crop specification ────────────────────────────────────────────────────────

/// Parse a crop specification string into `(x, y, w, h)` clamped to image bounds.
///
/// Accepts two forms:
/// - `"w,h"` — centred crop of that size.
/// - `"x,y,w,h"` — explicit offset crop.
///
/// Returns `None` on any parse error, empty dimensions, or if the crop origin
/// lies outside the image (a warning is logged in the error cases).
#[must_use]
pub fn parse_crop_spec(spec: &str, img_w: u32, img_h: u32) -> Option<(u32, u32, u32, u32)> {
    let parts: Vec<&str> = spec.split(',').collect();
    let (cx, cy, cw, ch) = match parts.len() {
        2 => {
            let w: u32 = parts[0].trim().parse().ok()?;
            let h: u32 = parts[1].trim().parse().ok()?;
            let x = img_w.saturating_sub(w) / 2;
            let y = img_h.saturating_sub(h) / 2;
            (x, y, w, h)
        }
        4 => {
            let x: u32 = parts[0].trim().parse().ok()?;
            let y: u32 = parts[1].trim().parse().ok()?;
            let w: u32 = parts[2].trim().parse().ok()?;
            let h: u32 = parts[3].trim().parse().ok()?;
            (x, y, w, h)
        }
        _ => {
            tracing::warn!(spec, "pre_crop_spec: expected 'w,h' or 'x,y,w,h'");
            return None;
        }
    };

    if cw == 0 || ch == 0 {
        tracing::warn!(spec, "pre_crop_spec: zero dimension — skipping crop");
        return None;
    }
    if cx >= img_w || cy >= img_h {
        tracing::warn!(
            spec,
            cx,
            cy,
            img_w,
            img_h,
            "pre_crop_spec: origin outside image — skipping crop"
        );
        return None;
    }

    let cw = cw.min(img_w.saturating_sub(cx));
    let ch = ch.min(img_h.saturating_sub(cy));
    Some((cx, cy, cw, ch))
}

/// Apply the configured rotation, crop and resize to one decoded frame, in that
/// order. The defaults (`pre_rotation = 0`, crop disabled,
/// `pre_resize_percent = 100`) return the input unchanged.
#[must_use]
pub fn preprocess_frame(img: DynamicImage, pre: &PreprocessingSettings) -> DynamicImage {
    let img = match pre.pre_rotation {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img,
    };

    let img = if pre.pre_crop_enabled && !pre.pre_crop_spec.is_empty() {
        let (iw, ih) = img.dimensions();
        if let Some((x, y, w, h)) = parse_crop_spec(&pre.pre_crop_spec, iw, ih) {
            img.crop_imm(x, y, w, h)
        } else {
            tracing::warn!(
                spec = pre.pre_crop_spec.as_str(),
                "crop spec invalid or out-of-bounds — skipping crop"
            );
            img
        }
    } else {
        img
    };

    if pre.pre_resize_percent < 100 {
        let (iw, ih) = img.dimensions();
        let new_w = scaled_dimension(iw, pre.pre_resize_percent);
        let new_h = scaled_dimension(ih, pre.pre_resize_percent);
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    }
}

/// `dim * percent / 100`, rounded to the nearest pixel and at least 1.
#[must_use]
pub fn scaled_dimension(dim: u32, percent: u32) -> u32 {
    // Integer form of `round(dim * percent / 100)`; u64 keeps the product exact.
    let scaled = (u64::from(dim) * u64::from(percent) + 50) / 100;
    u32::try_from(scaled).unwrap_or(u32::MAX).max(1)
}

/// The size `(width, height)` a `width × height` source frame has after
/// [`preprocess_frame`]: rotation, then the crop, then the resize.
///
/// The EXIF orientation is not applied (it can only swap the two sides, which
/// matters to nothing but a centred crop of an odd shape); the estimate of the
/// memory a run needs uses this.
#[must_use]
pub fn preprocessed_dimensions(width: u32, height: u32, pre: &PreprocessingSettings) -> (u32, u32) {
    let (mut w, mut h) = if matches!(pre.pre_rotation, 90 | 270) {
        (height, width)
    } else {
        (width, height)
    };
    if pre.pre_crop_enabled
        && !pre.pre_crop_spec.is_empty()
        && let Some((_, _, cw, ch)) = parse_crop_spec(&pre.pre_crop_spec, w, h)
    {
        (w, h) = (cw, ch);
    }
    if pre.pre_resize_percent < 100 {
        w = scaled_dimension(w, pre.pre_resize_percent);
        h = scaled_dimension(h, pre.pre_resize_percent);
    }
    (w, h)
}

// ── EXIF orientation ──────────────────────────────────────────────────────────

/// Bytes of the file head searched for the EXIF block (an APP1 segment is at
/// most 64 KiB and sits right after the JPEG start marker).
pub const EXIF_HEAD_BYTES: usize = 70_000;

/// The EXIF orientation tag (values 1..=8) from the head of a JPEG file.
///
/// Walks the marker segments up to the first APP1 `Exif` block and reads tag
/// `0x0112` of IFD0. Returns `None` for anything that is not a well-formed
/// JPEG with that tag.
#[must_use]
pub fn jpeg_exif_orientation(head: &[u8]) -> Option<u8> {
    if !head.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut pos = 2;
    while pos + 4 <= head.len() {
        if head[pos] != 0xFF {
            return None;
        }
        let marker = head[pos + 1];
        match marker {
            // Fill byte before a marker.
            0xFF => {
                pos += 1;
                continue;
            }
            // Standalone markers without a length.
            0x01 | 0xD0..=0xD8 => {
                pos += 2;
                continue;
            }
            // Start of scan / end of image: no EXIF follows.
            0xDA | 0xD9 => return None,
            _ => {}
        }
        let len = usize::from(u16::from_be_bytes([head[pos + 2], head[pos + 3]]));
        if len < 2 {
            return None;
        }
        let segment = head.get(pos + 4..pos + 2 + len)?;
        if marker == 0xE1 && segment.starts_with(b"Exif\0\0") {
            return tiff_orientation(&segment[6..]);
        }
        pos += 2 + len;
    }
    None
}

/// Orientation tag of IFD0 in a TIFF block (the payload of an EXIF segment).
fn tiff_orientation(tiff: &[u8]) -> Option<u8> {
    let little = match tiff.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |o: usize| -> Option<u16> {
        let b = tiff.get(o..o + 2)?;
        Some(if little {
            u16::from_le_bytes([b[0], b[1]])
        } else {
            u16::from_be_bytes([b[0], b[1]])
        })
    };
    let u32_at = |o: usize| -> Option<u32> {
        let b = tiff.get(o..o + 4)?;
        Some(if little {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = usize::try_from(u32_at(4)?).ok()?;
    let entries = usize::from(u16_at(ifd)?);
    for i in 0..entries {
        let entry = ifd + 2 + 12 * i;
        if u16_at(entry)? == 0x0112 {
            let value = u16_at(entry + 8)?;
            return u8::try_from(value).ok().filter(|o| (1..=8).contains(o));
        }
    }
    None
}

/// Rotate / mirror `img` so that it is displayed upright for EXIF orientation
/// `orientation` (1 = already upright).
#[must_use]
pub fn apply_orientation(img: DynamicImage, orientation: u8) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate90().flipv(),
        8 => img.rotate270(),
        _ => img,
    }
}
