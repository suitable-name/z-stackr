//! Encoding for Save: the `f32` result to an 8- or 16-bit file.
//!
//! Save encodes straight from the linear-light planar result (retouch strokes
//! included) with the core rounding encoders, not from the 8-bit preview PNG: a
//! 16-bit PNG or TIFF carries real 16-bit samples, and white is 65535 (8-bit: 255).
//!
//! Everything here is plain Rust so `tests/export_test.rs` can drive it without a
//! window; `gui/export.rs` adds the dialog and the worker thread.
//!
//! The encoders work in memory ([`encode_planar`], [`encode_image_bytes`]): the
//! desktop writes the bytes to the chosen file ([`save_planar`], [`save_file`]),
//! the browser hands them to a download.

use std::{io::Cursor, path::Path};

use image::{
    ColorType, DynamicImage, EncodableLayout, ImageBuffer, ImageEncoder, Rgb, RgbImage,
    codecs::{jpeg::JpegEncoder, png::PngEncoder, tiff::TiffEncoder},
};
use rayon::prelude::*;
use stacker_core::{
    color::{encode_row_u8, encode_row_u16, planar_to_linear_rgb},
    image::PlanarImage,
};

use crate::settings::{ImageSavingSettings, OutputFormat, PreprocessingSettings};

pub mod filename_template;

/// The format to write: the one the chosen extension names, else `configured`.
#[must_use]
pub fn effective_format(dest: &Path, configured: OutputFormat) -> OutputFormat {
    let ext = dest
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("png") => OutputFormat::Png,
        Some("jpg" | "jpeg") => OutputFormat::Jpeg,
        Some("tif" | "tiff") => OutputFormat::Tiff,
        _ => configured,
    }
}

/// `true` when a file of `format` is written with 16 bits per channel: PNG and
/// TIFF at the configured depth of 16; JPEG is always 8-bit.
#[must_use]
pub const fn writes_16_bit(format: OutputFormat, configured_depth: u32) -> bool {
    !matches!(format, OutputFormat::Jpeg) && configured_depth == 16
}

/// What a successful save wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SavedAs {
    /// The container format.
    pub format: OutputFormat,
    /// `true` for 16 bits per channel.
    pub bit16: bool,
}

impl SavedAs {
    /// `16-bit PNG`, `8-bit TIFF` or `JPEG`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match (self.format, self.bit16) {
            (OutputFormat::Png, true) => "16-bit PNG",
            (OutputFormat::Png, false) => "8-bit PNG",
            (OutputFormat::Tiff, true) => "16-bit TIFF",
            (OutputFormat::Tiff, false) => "8-bit TIFF",
            (OutputFormat::Jpeg, _) => "JPEG",
        }
    }
}

/// The raster size as `u32`, or why the image cannot be encoded.
fn raster_size(img: &PlanarImage<f32>) -> Result<(u32, u32), String> {
    if img.width == 0 || img.height == 0 {
        return Err("the image is empty".to_owned());
    }
    match (u32::try_from(img.width), u32::try_from(img.height)) {
        (Ok(w), Ok(h)) => Ok((w, h)),
        _ => Err("the image is too large to encode".to_owned()),
    }
}

/// Linear RGB of row `row` of `img` (`YCbCr` to RGB), three samples per pixel.
fn linear_row(img: &PlanarImage<f32>, row: usize, linear: &mut [f32]) {
    let base = row * img.width;
    for (x, px) in linear.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let i = base + x;
        *px = planar_to_linear_rgb([img.luma[i], img.chroma_a[i], img.chroma_b[i]]);
    }
}

/// The result as an 8-bit sRGB image (rounded: white is 255).
///
/// # Errors
///
/// An empty or oversized image.
pub fn planar_to_rgb8(img: &PlanarImage<f32>) -> Result<RgbImage, String> {
    let (w, h) = raster_size(img)?;
    let stride = img.width * 3;
    let mut raw = vec![0_u8; stride * img.height];
    raw.par_chunks_mut(stride).enumerate().for_each_init(
        || vec![0.0_f32; stride],
        |linear, (row, out)| {
            linear_row(img, row, linear);
            encode_row_u8(linear, out);
        },
    );
    RgbImage::from_raw(w, h, raw).ok_or_else(|| "internal error: buffer size mismatch".to_owned())
}

/// The result as a 16-bit sRGB image (rounded: white is 65535).
///
/// # Errors
///
/// An empty or oversized image.
pub fn planar_to_rgb16(img: &PlanarImage<f32>) -> Result<ImageBuffer<Rgb<u16>, Vec<u16>>, String> {
    let (w, h) = raster_size(img)?;
    let stride = img.width * 3;
    let mut raw = vec![0_u16; stride * img.height];
    raw.par_chunks_mut(stride).enumerate().for_each_init(
        || vec![0.0_f32; stride],
        |linear, (row, out)| {
            linear_row(img, row, linear);
            encode_row_u16(linear, out);
        },
    );
    ImageBuffer::from_raw(w, h, raw)
        .ok_or_else(|| "internal error: buffer size mismatch".to_owned())
}

/// Encode `img` into the bytes of a file named `dest` (only its extension
/// matters) in the effective format and the configured depth.
///
/// # Errors
///
/// The encoder error, as text.
pub fn encode_planar(
    img: &PlanarImage<f32>,
    dest: &Path,
    cfg: &ImageSavingSettings,
) -> Result<(Vec<u8>, SavedAs), String> {
    encode_planar_exif(img, dest, cfg, None)
}

/// [`encode_planar`] with an EXIF block (see [`exif_block`]) spliced into the
/// file. TIFF output carries none: the block is dropped there.
///
/// # Errors
///
/// The encoder error, as text.
pub fn encode_planar_exif(
    img: &PlanarImage<f32>,
    dest: &Path,
    cfg: &ImageSavingSettings,
    exif: Option<Vec<u8>>,
) -> Result<(Vec<u8>, SavedAs), String> {
    // The pipeline's encoder is the one the CLI writes with. An empty or oversized
    // image is refused here with a message instead of reaching it.
    raster_size(img)?;
    let format = effective_format(dest, cfg.output_format);
    let saving = ImageSavingSettings {
        output_format: format,
        ..cfg.clone()
    };
    let exif = exif.filter(|_| stacker_pipeline::output::supports_exif(format));
    let encoded = stacker_pipeline::output::encode(img, &saving).with_exif(exif);
    let bit16 = encoded.is_16_bit();
    let bytes = encoded.to_bytes().map_err(|e| e.to_string())?;
    Ok((bytes, SavedAs { format, bit16 }))
}

/// Re-encode the image file whose bytes are `source` into the bytes of a file
/// named `dest` (only its extension matters).
///
/// # Errors
///
/// The decoder or encoder error, as text.
pub fn encode_image_bytes(
    source: &[u8],
    dest: &Path,
    cfg: &ImageSavingSettings,
) -> Result<(Vec<u8>, SavedAs), String> {
    let img = image::load_from_memory(source)
        .map_err(|e| format!("could not open the image to export: {e}"))?;
    let format = effective_format(dest, cfg.output_format);
    let bit16 = writes_16_bit(format, cfg.bit_depth);
    let bytes = if bit16 {
        encode_rgb16(&widen_to_rgb16(&img), format)?
    } else {
        encode_rgb8(&img.to_rgb8(), format, jpeg_quality(cfg))?
    };
    Ok((bytes, SavedAs { format, bit16 }))
}

/// Encode `img` into `dest` in the effective format and the configured depth.
///
/// No metadata is copied; see [`save_planar_exif`].
///
/// # Errors
///
/// The encoder or I/O error, as text.
pub fn save_planar(
    img: &PlanarImage<f32>,
    dest: &Path,
    cfg: &ImageSavingSettings,
) -> Result<SavedAs, String> {
    save_planar_exif(img, dest, cfg, None)
}

/// [`save_planar`] with the EXIF block of the first source frame ([`exif_block`]).
///
/// # Errors
///
/// The encoder or I/O error, as text.
pub fn save_planar_exif(
    img: &PlanarImage<f32>,
    dest: &Path,
    cfg: &ImageSavingSettings,
    exif: Option<Vec<u8>>,
) -> Result<SavedAs, String> {
    let (bytes, saved) = encode_planar_exif(img, dest, cfg, exif)?;
    write_file(dest, &bytes)?;
    Ok(saved)
}

/// A platform's file reader: the bytes of the file at a path.
pub type ByteReader<'a> = &'a dyn Fn(&Path) -> std::io::Result<Vec<u8>>;

/// The EXIF block to write into the result (`copy_metadata`).
///
/// It is that of the first source frame `first`, orientation fitted to the pixels.
/// `None` when the setting is off, there is no source, or the file holds no EXIF.
///
/// The file is read through `read_bytes` (the platform's reader, so a browser
/// path `web:/...` works) when `from_bytes` is set, else straight from the path.
#[must_use]
pub fn exif_block(
    first: Option<&Path>,
    pre: &PreprocessingSettings,
    saving: &ImageSavingSettings,
    from_bytes: Option<ByteReader<'_>>,
) -> Option<Vec<u8>> {
    if !saving.copy_metadata {
        return None;
    }
    let first = first?;
    match from_bytes {
        Some(read) => {
            let bytes = read(first).ok()?;
            stacker_pipeline::output::source_exif_from_bytes(&bytes, pre, saving)
        }
        None => stacker_pipeline::output::source_exif(first, pre, saving),
    }
}

/// A note for the toast when `copy_metadata` is on but the file `dest` cannot
/// carry it (TIFF), else `None`.
#[must_use]
pub fn exif_note(dest: &Path, cfg: &ImageSavingSettings) -> Option<&'static str> {
    let format = effective_format(dest, cfg.output_format);
    (cfg.copy_metadata && !stacker_pipeline::output::supports_exif(format))
        .then_some("TIFF files carry no metadata: it was not copied")
}

/// Re-encode the image file `src` into `dest` (the path a source frame or an
/// older result takes: those exist only as files, so the file's own precision
/// is all there is).
///
/// # Errors
///
/// The decoder, encoder or I/O error, as text.
pub fn save_file(src: &Path, dest: &Path, cfg: &ImageSavingSettings) -> Result<SavedAs, String> {
    let source =
        std::fs::read(src).map_err(|e| format!("could not open the image to export: {e}"))?;
    let (bytes, saved) = encode_image_bytes(&source, dest, cfg)?;
    write_file(dest, &bytes)?;
    Ok(saved)
}

/// Write `bytes` to `dest`.
fn write_file(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(dest, bytes).map_err(|e| format!("could not write {}: {e}", dest.display()))
}

/// A 16-bit RGB copy of `img`. A 16-bit source is converted as is; an 8-bit
/// source is widened by exactly 257 per code so that 255 becomes 65535 (the
/// `image` crate's own conversion shifts by 8 and tops out at 65280).
fn widen_to_rgb16(img: &DynamicImage) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    match img {
        DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA16(_)
        | DynamicImage::ImageRgb16(_)
        | DynamicImage::ImageRgba16(_) => img.to_rgb16(),
        _ => {
            let rgb8 = img.to_rgb8();
            let (w, h) = rgb8.dimensions();
            let wide: Vec<u16> = rgb8.as_raw().iter().map(|&v| u16::from(v) * 257).collect();
            ImageBuffer::from_raw(w, h, wide).expect("same dimensions as the 8-bit source")
        }
    }
}

fn jpeg_quality(cfg: &ImageSavingSettings) -> u8 {
    u8::try_from(cfg.jpeg_quality.clamp(1, 100)).unwrap_or(95)
}

/// The file bytes of an 8-bit image in `format`.
fn encode_rgb8(rgb8: &RgbImage, format: OutputFormat, quality: u8) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    match format {
        OutputFormat::Jpeg => {
            JpegEncoder::new_with_quality(&mut out, quality)
                .encode_image(rgb8)
                .map_err(|e| format!("JPEG encoding failed: {e}"))?;
        }
        OutputFormat::Png => PngEncoder::new(&mut out)
            .write_image(rgb8.as_raw(), rgb8.width(), rgb8.height(), ColorType::Rgb8)
            .map_err(|e| e.to_string())?,
        OutputFormat::Tiff => TiffEncoder::new(Cursor::new(&mut out))
            .write_image(rgb8.as_raw(), rgb8.width(), rgb8.height(), ColorType::Rgb8)
            .map_err(|e| e.to_string())?,
    }
    Ok(out)
}

/// The file bytes of a 16-bit image in `format` (PNG or TIFF; JPEG has no
/// 16-bit form).
fn encode_rgb16(
    rgb16: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    format: OutputFormat,
) -> Result<Vec<u8>, String> {
    let bytes = rgb16.as_raw().as_bytes();
    let (width, height) = (rgb16.width(), rgb16.height());
    let mut out = Vec::new();
    match format {
        OutputFormat::Png => PngEncoder::new(&mut out)
            .write_image(bytes, width, height, ColorType::Rgb16)
            .map_err(|e| e.to_string())?,
        OutputFormat::Tiff => TiffEncoder::new(Cursor::new(&mut out))
            .write_image(bytes, width, height, ColorType::Rgb16)
            .map_err(|e| e.to_string())?,
        OutputFormat::Jpeg => return Err("JPEG is always written with 8 bits".to_owned()),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{SavedAs, effective_format, writes_16_bit};
    use crate::settings::OutputFormat;

    #[test]
    fn the_typed_extension_wins_over_the_configured_format() {
        assert_eq!(
            effective_format(Path::new("out.JPG"), OutputFormat::Tiff),
            OutputFormat::Jpeg
        );
        assert_eq!(
            effective_format(Path::new("out.xyz"), OutputFormat::Png),
            OutputFormat::Png
        );
    }

    #[test]
    fn jpeg_is_never_16_bit() {
        assert!(writes_16_bit(OutputFormat::Png, 16));
        assert!(writes_16_bit(OutputFormat::Tiff, 16));
        assert!(!writes_16_bit(OutputFormat::Png, 8));
        assert!(!writes_16_bit(OutputFormat::Jpeg, 16));
    }

    #[test]
    fn labels_name_the_depth() {
        let png16 = SavedAs {
            format: OutputFormat::Png,
            bit16: true,
        };
        assert_eq!(png16.label(), "16-bit PNG");
        let jpeg = SavedAs {
            format: OutputFormat::Jpeg,
            bit16: false,
        };
        assert_eq!(jpeg.label(), "JPEG");
    }
}
