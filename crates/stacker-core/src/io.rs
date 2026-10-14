//! Camera RAW input.
//!
//! RAW files are recognised by their extension ([`is_raw_extension`], the list
//! [`RAW_EXTENSIONS`]) in every build, so file pickers and directory scans
//! agree whether or not RAW can actually be decoded. Decoding itself
//! ([`decode_raw_file`]) needs the `raw` cargo feature and a native target;
//! without either it returns [`RawError::Disabled`] with a message that says
//! how to get RAW support. [`RAW_SUPPORTED`] tells which of the two this build
//! is.
//!
//! # What the decoder does
//!
//! With the feature on, [`decode_raw_file`] runs `rawler` (the decoding engine
//! of dnglab): the container and the sensor data are parsed, then the default
//! `RawDevelop` chain develops them: rescale to `[0, 1]`, demosaic (PPG for a
//! Bayer sensor, bilinear for X-Trans), crop to the active area, apply the
//! camera white balance stored in the file, calibrate to sRGB primaries with
//! the file's colour matrix (D65, else the nearest illuminant adapted by
//! Bradford), crop to the recommended output area and apply the sRGB transfer
//! curve. The result is returned as a 16-bit sRGB `DynamicImage`, the same shape
//! a 16-bit TIFF has, so the pipeline's loader linearises it with the exact
//! IEC 61966-2-1 table and every later stage works in linear light like for
//! any other frame.
//!
//! # Limits
//!
//! * No lens corrections (distortion, vignetting, chromatic aberration), no
//!   highlight recovery, no white-balance choice: the white balance stored in
//!   the file is used as it is. For colour-critical work, develop the RAW files
//!   in a RAW converter and stack the TIFFs.
//! * The EXIF orientation is **not** applied to RAW pixels. (JPEG frames are
//!   rotated on load; RAW frames are not.) The copied EXIF keeps the source's
//!   orientation tag for them, so the output still displays upright.
//! * A monochrome sensor yields a 16-bit grey image, which the loader widens to
//!   RGB. A four-colour sensor whose colour matrix has not folded it to three
//!   channels is rejected.

use std::{fmt, path::Path};

use ::image::DynamicImage;

/// Lower-case, dot-less file extensions recognised as camera RAW.
///
/// The single list: [`is_raw_extension`], the pipeline's directory scan and the
/// apps' file pickers all read it. It is an allow-list of containers `rawler`
/// can parse, not a promise that every camera model of such a file decodes.
pub const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "nef", "arw", "dng", "raf", "orf", "rw2", "pef", "srw", "3fr", "erf", "kdc",
    "mrw", "nrw", "raw", "rwl", "sr2", "srf", "x3f",
];

/// `true` when this build can decode RAW files: the `raw` feature is on and the
/// target is not `wasm32`.
pub const RAW_SUPPORTED: bool = cfg!(all(feature = "raw", not(target_arch = "wasm32")));

/// `true` when `ext` (any letter case, no leading dot) is one of the
/// [`RAW_EXTENSIONS`].
#[must_use]
pub fn is_raw_extension(ext: &str) -> bool {
    RAW_EXTENSIONS
        .iter()
        .any(|known| known.eq_ignore_ascii_case(ext))
}

/// `true` when `path` has one of the [`RAW_EXTENSIONS`].
#[must_use]
pub fn has_raw_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(is_raw_extension)
}

/// Why a RAW file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawError {
    /// This build has no RAW support (`raw` feature off, or `wasm32`).
    Disabled {
        /// The file's extension, as given (without the dot).
        ext: String,
    },
    /// `rawler` could not decode the file.
    Decode {
        /// The decoder's message.
        message: String,
    },
}

impl fmt::Display for RawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled { ext } => write!(
                f,
                "'.{ext}' is a camera RAW format, but this build has no RAW support: rebuild \
                 with `--features raw` (desktop and command line only, not the browser), or \
                 convert the file to TIFF or DNG first"
            ),
            Self::Decode { message } => write!(f, "RAW decoding failed: {message}"),
        }
    }
}

impl std::error::Error for RawError {}

/// Decode the RAW file at `path` into a 16-bit sRGB image (see the module
/// documentation for the development chain).
///
/// # Errors
/// [`RawError::Decode`] when the file cannot be parsed or developed.
#[cfg(all(feature = "raw", not(target_arch = "wasm32")))]
pub fn decode_raw_file(path: &Path) -> Result<DynamicImage, RawError> {
    use rawler::imgop::develop::{Intermediate, RawDevelop};

    let decode_error = |e: &dyn fmt::Display| RawError::Decode {
        message: e.to_string(),
    };
    let rawimage = rawler::decode_file(path).map_err(|e| decode_error(&e))?;
    let developed = RawDevelop::default()
        .develop_intermediate(&rawimage)
        .map_err(|e| decode_error(&e))?;
    match developed {
        Intermediate::ThreeColor(px) => {
            let mut samples = Vec::with_capacity(px.data.len() * 3);
            for rgb in &px.data {
                samples.extend(rgb.iter().map(|&c| to_u16(c)));
            }
            sized_buffer(px.width, px.height, samples).map(DynamicImage::ImageRgb16)
        }
        Intermediate::Monochrome(px) => {
            let samples = px.data.iter().map(|&c| to_u16(c)).collect();
            sized_buffer(px.width, px.height, samples).map(DynamicImage::ImageLuma16)
        }
        Intermediate::FourColor(_) => Err(RawError::Decode {
            message: "the sensor has four colour channels that the colour matrix did not fold \
                      to three"
                .to_owned(),
        }),
    }
}

/// Without RAW support every decode fails with [`RawError::Disabled`].
///
/// # Errors
/// Always [`RawError::Disabled`].
#[cfg(not(all(feature = "raw", not(target_arch = "wasm32"))))]
pub fn decode_raw_file(path: &Path) -> Result<DynamicImage, RawError> {
    Err(RawError::Disabled {
        ext: path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_owned(),
    })
}

/// A `[0, 1]` sample as a rounded, clamped 16-bit code (NaN becomes 0).
#[cfg(all(feature = "raw", not(target_arch = "wasm32")))]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to 0..=65535 before the cast"
)]
fn to_u16(sample: f32) -> u16 {
    if sample.is_nan() {
        0
    } else {
        (sample.clamp(0.0, 1.0) * 65_535.0).round() as u16
    }
}

/// An image buffer of `width × height` from interleaved `samples`.
#[cfg(all(feature = "raw", not(target_arch = "wasm32")))]
fn sized_buffer<P>(
    width: usize,
    height: usize,
    samples: Vec<u16>,
) -> Result<::image::ImageBuffer<P, Vec<u16>>, RawError>
where
    P: ::image::Pixel<Subpixel = u16> + 'static,
{
    let mismatch = || RawError::Decode {
        message: "the decoded pixel buffer does not match its reported size".to_owned(),
    };
    let w = u32::try_from(width).map_err(|_| mismatch())?;
    let h = u32::try_from(height).map_err(|_| mismatch())?;
    ::image::ImageBuffer::from_raw(w, h, samples).ok_or_else(mismatch)
}
