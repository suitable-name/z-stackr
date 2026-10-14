//! Planar linear light → 8- or 16-bit sRGB samples, and the file writers.
//!
//! One path serves both apps: pixels go through
//! [`stacker_core::color::planar_to_linear_rgb`], then the core row encoders
//! (`encode_row_u8` / `encode_row_u16`), which **round** and clamp, so linear
//! white is exactly 255 / 65 535. The 16-bit precision of the fusion survives to
//! the file for PNG and TIFF; JPEG is always 8-bit.
//!
//! [`EncodedImage`] is a full-image buffer at its final bit depth. A tiled
//! caller creates it with [`EncodedImage::zeroed`] and pastes each fused tile's
//! interior with [`EncodedImage::paste_tile`] (the only full-image allocation
//! of a tiled run: 3 or 6 bytes per pixel); a caller with the whole fused image
//! calls [`encode`].
//!
//! # Metadata (`copy_metadata`)
//!
//! [`source_exif`] (a file) and [`source_exif_from_bytes`] (memory) return the
//! EXIF block of the first frame, already fitted to the output: `None` when
//! `image_saving.copy_metadata` is off or the frame has no EXIF, otherwise the
//! block with its orientation tag rewritten to 1 whenever the pixels were
//! already turned upright on load. [`EncodedImage::with_exif`] attaches it;
//! [`EncodedImage::save`] and [`EncodedImage::to_bytes`] then splice it into
//! the encoded file. JPEG and PNG carry it; **TIFF output cannot**, so a
//! TIFF result is written without metadata and a warning is logged (a
//! failure to copy never fails the write).

use std::{
    fs::File,
    io::{Cursor, Read, Write},
    path::Path,
};

use image::{
    ColorType, EncodableLayout, ImageBuffer, ImageEncoder, Rgb, RgbImage,
    codecs::{jpeg::JpegEncoder, png::PngEncoder, tiff::TiffEncoder},
};
use rayon::prelude::*;
use stacker_core::{
    color::{encode_row_u8, encode_row_u16, planar_to_linear_rgb},
    image::PlanarImage,
    metadata,
    settings::{ImageSavingSettings, OutputFormat, PreprocessingSettings},
};

use crate::{
    PipelineError,
    crop::{Rect, resize_plane},
    load::{EXIF_HEAD_BYTES, jpeg_exif_orientation},
};

type Rgb16Image = ImageBuffer<Rgb<u16>, Vec<u16>>;

/// The samples of an [`EncodedImage`].
#[derive(Debug, Clone)]
enum Samples {
    Rgb8(RgbImage),
    Rgb16(Rgb16Image),
}

/// A full-image sRGB buffer at its final bit depth, ready to be written.
#[derive(Debug, Clone)]
pub struct EncodedImage {
    format: OutputFormat,
    jpeg_quality: u8,
    samples: Samples,
    /// EXIF block (bare TIFF structure) to write into the file, if any.
    exif: Option<Vec<u8>>,
}

/// `true` when `format` at the configured `bit_depth` is written with 16 bits
/// per channel (PNG and TIFF only; JPEG is always 8-bit).
#[must_use]
pub const fn wants_16_bit(format: OutputFormat, bit_depth: u32) -> bool {
    !matches!(format, OutputFormat::Jpeg) && bit_depth == 16
}

/// JPEG quality from the configured value, clamped to `1..=100`.
#[must_use]
pub fn jpeg_quality(configured: u32) -> u8 {
    u8::try_from(configured.clamp(1, 100)).unwrap_or(95)
}

impl EncodedImage {
    /// An all-zero buffer of `width × height` pixels.
    ///
    /// # Panics
    /// Panics when a dimension exceeds `u32::MAX`.
    #[must_use]
    pub fn zeroed(
        width: usize,
        height: usize,
        format: OutputFormat,
        bit16: bool,
        jpeg_quality: u8,
    ) -> Self {
        let w = u32::try_from(width).expect("image width fits in u32");
        let h = u32::try_from(height).expect("image height fits in u32");
        let samples = if bit16 && !matches!(format, OutputFormat::Jpeg) {
            Samples::Rgb16(ImageBuffer::new(w, h))
        } else {
            Samples::Rgb8(RgbImage::new(w, h))
        };
        Self {
            format,
            jpeg_quality,
            samples,
            exif: None,
        }
    }

    /// Attach the EXIF block to write into the file ([`source_exif`] makes
    /// one); `None` removes it. Has no effect for a TIFF buffer, which cannot
    /// carry EXIF (a warning is logged when it is written).
    #[must_use]
    pub fn with_exif(mut self, exif: Option<Vec<u8>>) -> Self {
        self.exif = exif;
        self
    }

    /// The EXIF block that [`Self::save`] and [`Self::to_bytes`] write, if any.
    #[must_use]
    pub fn exif(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }

    /// Width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        match &self.samples {
            Samples::Rgb8(b) => b.width() as usize,
            Samples::Rgb16(b) => b.width() as usize,
        }
    }

    /// Height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        match &self.samples {
            Samples::Rgb8(b) => b.height() as usize,
            Samples::Rgb16(b) => b.height() as usize,
        }
    }

    /// `true` for a 16-bit buffer.
    #[must_use]
    pub const fn is_16_bit(&self) -> bool {
        matches!(self.samples, Samples::Rgb16(_))
    }

    /// The 8-bit samples (`r, g, b, r, …`), if this is an 8-bit buffer.
    #[must_use]
    pub fn rgb8(&self) -> Option<&[u8]> {
        match &self.samples {
            Samples::Rgb8(b) => Some(b.as_raw()),
            Samples::Rgb16(_) => None,
        }
    }

    /// The 16-bit samples (`r, g, b, r, …`), if this is a 16-bit buffer.
    #[must_use]
    pub fn rgb16(&self) -> Option<&[u16]> {
        match &self.samples {
            Samples::Rgb16(b) => Some(b.as_raw()),
            Samples::Rgb8(_) => None,
        }
    }

    /// Write the interior of a fused (padded) tile into the buffer.
    ///
    /// `pad_off` is the offset of the interior inside `tile`;
    /// `dest = (x, y, w, h)` is the destination rectangle (the interior).
    pub fn paste_tile(
        &mut self,
        tile: &PlanarImage<f32>,
        pad_off: (usize, usize),
        dest: (usize, usize, usize, usize),
    ) {
        let out_stride = self.width() * 3;
        match &mut self.samples {
            Samples::Rgb8(buf) => {
                let raw: &mut [u8] = buf;
                paste_rows(raw, out_stride, tile, pad_off, dest, encode_row_u8);
            }
            Samples::Rgb16(buf) => {
                let raw: &mut [u16] = buf;
                paste_rows(raw, out_stride, tile, pad_off, dest, encode_row_u16);
            }
        }
    }

    /// Cut the buffer down to `rect = (x, y, w, h)`. The full frame returns the
    /// buffer itself; anything else copies the rectangle once.
    ///
    /// # Panics
    /// Panics when `rect` reaches outside the buffer.
    #[must_use]
    pub fn crop(self, rect: Rect) -> Self {
        let (cx, cy, cw, ch) = rect;
        let (full_w, full_h) = (self.width(), self.height());
        if rect == (0, 0, full_w, full_h) {
            return self;
        }
        assert!(
            cx + cw <= full_w && cy + ch <= full_h,
            "crop {rect:?} reaches outside the {full_w}x{full_h} image"
        );
        let w = u32::try_from(cw).expect("crop width fits in u32");
        let h = u32::try_from(ch).expect("crop height fits in u32");
        let samples = match self.samples {
            Samples::Rgb8(buf) => Samples::Rgb8(
                RgbImage::from_raw(w, h, crop_interleaved(buf.as_raw(), full_w, 3, rect))
                    .expect("cropped buffer has w*h*3 samples"),
            ),
            Samples::Rgb16(buf) => Samples::Rgb16(
                Rgb16Image::from_raw(w, h, crop_interleaved(buf.as_raw(), full_w, 3, rect))
                    .expect("cropped buffer has w*h*3 samples"),
            ),
        };
        Self { samples, ..self }
    }

    /// Resample the buffer to `width × height` with the edge-clamped
    /// Catmull-Rom spline of [`crate::crop::resize_planar`] (pixel centres at
    /// `(i + ½)·scale − ½`), channel by channel on the encoded samples. This is
    /// `resize_cropped_to_original` for the out-of-core run, whose sink holds
    /// the encoded image. A buffer that already has the target size is returned
    /// as it is; an empty source or target gives an all-zero buffer of the
    /// target size. The attached EXIF block is kept.
    ///
    /// The spline acts on the encoded (gamma) samples here and on the linear
    /// planes in `resize_planar`: the two agree on flat and gently varying
    /// content (within one 8-bit code in the tests) and differ slightly across
    /// sharp edges.
    ///
    /// # Panics
    /// Panics when a dimension exceeds `u32::MAX`.
    #[must_use]
    pub fn resize(self, width: usize, height: usize) -> Self {
        let (sw, sh) = (self.width(), self.height());
        if (sw, sh) == (width, height) {
            return self;
        }
        let w = u32::try_from(width).expect("resized width fits in u32");
        let h = u32::try_from(height).expect("resized height fits in u32");
        let empty = sw == 0 || sh == 0 || width == 0 || height == 0;
        let samples = match self.samples {
            Samples::Rgb8(buf) => Samples::Rgb8(if empty {
                RgbImage::new(w, h)
            } else {
                RgbImage::from_raw(
                    w,
                    h,
                    resize_interleaved(
                        buf.as_raw(),
                        (sw, sh),
                        (width, height),
                        f32::from,
                        u8_from_f32,
                    ),
                )
                .expect("resized buffer has w*h*3 samples")
            }),
            Samples::Rgb16(buf) => Samples::Rgb16(if empty {
                Rgb16Image::new(w, h)
            } else {
                Rgb16Image::from_raw(
                    w,
                    h,
                    resize_interleaved(
                        buf.as_raw(),
                        (sw, sh),
                        (width, height),
                        f32::from,
                        u16_from_f32,
                    ),
                )
                .expect("resized buffer has w*h*3 samples")
            }),
        };
        Self { samples, ..self }
    }

    /// Write the image to `path`. The container follows the extension of
    /// `path` (`.png`, `.tif`/`.tiff`, `.jpg`/`.jpeg`), except that a JPEG
    /// buffer is always written as JPEG with the configured quality.
    ///
    /// # Errors
    /// Returns the encoder or I/O error. A failure to copy the attached EXIF
    /// block (see [`Self::with_exif`]) is logged, not returned: the image is
    /// already written by then.
    pub fn save(self, path: &Path) -> Result<(), PipelineError> {
        let exif = self.exif;
        match self.samples {
            Samples::Rgb16(buf) => {
                buf.save(path)?;
            }
            Samples::Rgb8(buf) => {
                if matches!(self.format, OutputFormat::Jpeg) {
                    save_jpeg(&buf, path, self.jpeg_quality)?;
                } else {
                    buf.save(path)?;
                }
            }
        }
        if let Some(blob) = exif
            && let Err(e) = metadata::inject_exif(path, &blob)
        {
            tracing::warn!(path = %path.display(), "{e}");
        }
        Ok(())
    }

    /// The file format this buffer is written in.
    #[must_use]
    pub const fn format(&self) -> OutputFormat {
        self.format
    }

    /// The (lower-case, dot-less) file extension of [`Self::format`]: `tif`,
    /// `png` or `jpg`.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self.format {
            OutputFormat::Tiff => "tif",
            OutputFormat::Png => "png",
            OutputFormat::Jpeg => "jpg",
        }
    }

    /// The encoded file contents in [`Self::format`], without touching the file
    /// system: the bytes [`Self::save`] writes to a path with the matching
    /// extension (same encoders, same sample order, same 16-bit byte order).
    ///
    /// # Errors
    /// Returns the encoder error.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PipelineError> {
        let mut out = Vec::new();
        match &self.samples {
            Samples::Rgb8(buf) => match self.format {
                OutputFormat::Jpeg => encode_jpeg(buf, &mut out, self.jpeg_quality)?,
                OutputFormat::Png => {
                    PngEncoder::new(&mut out).write_image(
                        buf.as_raw(),
                        buf.width(),
                        buf.height(),
                        ColorType::Rgb8,
                    )?;
                }
                OutputFormat::Tiff => {
                    TiffEncoder::new(Cursor::new(&mut out)).write_image(
                        buf.as_raw(),
                        buf.width(),
                        buf.height(),
                        ColorType::Rgb8,
                    )?;
                }
            },
            Samples::Rgb16(buf) => {
                let bytes = buf.as_raw().as_bytes();
                match self.format {
                    OutputFormat::Png => PngEncoder::new(&mut out).write_image(
                        bytes,
                        buf.width(),
                        buf.height(),
                        ColorType::Rgb16,
                    )?,
                    OutputFormat::Tiff => TiffEncoder::new(Cursor::new(&mut out)).write_image(
                        bytes,
                        buf.width(),
                        buf.height(),
                        ColorType::Rgb16,
                    )?,
                    OutputFormat::Jpeg => {
                        return Err(PipelineError::invalid("JPEG is always written with 8 bits"));
                    }
                }
            }
        }
        if let Some(blob) = &self.exif {
            match metadata::inject_exif_into_bytes(&out, blob) {
                Ok(with_exif) => out = with_exif,
                Err(e) => tracing::warn!("{e}"),
            }
        }
        Ok(out)
    }
}

/// Encode a fused image per the `image_saving` settings (format, bit depth,
/// JPEG quality) into a full-image buffer.
///
/// Linear white (`luma = 1`, no chroma) becomes 255 (8-bit) or 65 535 (16-bit).
#[must_use]
pub fn encode(planar: &PlanarImage<f32>, saving: &ImageSavingSettings) -> EncodedImage {
    let mut encoded = EncodedImage::zeroed(
        planar.width,
        planar.height,
        saving.output_format,
        wants_16_bit(saving.output_format, saving.bit_depth),
        jpeg_quality(saving.jpeg_quality),
    );
    encoded.paste_tile(planar, (0, 0), (0, 0, planar.width, planar.height));
    encoded
}

/// `true` when `format` can carry an EXIF block (JPEG and PNG; not TIFF).
#[must_use]
pub const fn supports_exif(format: OutputFormat) -> bool {
    !matches!(format, OutputFormat::Tiff)
}

/// The EXIF block of the source frame `source` (the first frame of the stack),
/// fitted to the output, or `None` when `saving.copy_metadata` is off, the
/// file cannot be read or it holds no EXIF.
///
/// The orientation tag is rewritten to 1 when the pixels no longer need it:
/// the loader already rotated them (a JPEG, with `ignore_exif_orientation` off)
/// or `pre.pre_rotation` turned them. Otherwise the source's tag is kept, so a
/// frame that was not rotated (a RAW or TIFF source, or an ignored orientation)
/// still displays as it did. Attach the result with
/// [`EncodedImage::with_exif`].
#[must_use]
pub fn source_exif(
    source: &Path,
    pre: &PreprocessingSettings,
    saving: &ImageSavingSettings,
) -> Option<Vec<u8>> {
    if !saving.copy_metadata {
        return None;
    }
    let mut blob = metadata::extract_exif(source)?;
    let mut head = Vec::new();
    // A read error leaves `head` short, which only means "orientation not applied".
    let _ = File::open(source).and_then(|f| f.take(EXIF_HEAD_BYTES as u64).read_to_end(&mut head));
    fit_orientation(&mut blob, &head, pre);
    Some(blob)
}

/// [`source_exif`] for a frame held in memory (the browser build): `bytes` is
/// the encoded source file.
#[must_use]
pub fn source_exif_from_bytes(
    bytes: &[u8],
    pre: &PreprocessingSettings,
    saving: &ImageSavingSettings,
) -> Option<Vec<u8>> {
    if !saving.copy_metadata {
        return None;
    }
    let mut blob = metadata::extract_exif_from_bytes(bytes)?;
    fit_orientation(&mut blob, &bytes[..bytes.len().min(EXIF_HEAD_BYTES)], pre);
    Some(blob)
}

/// Rewrite the orientation tag of `blob` to 1 when the loader turned the pixels
/// of the source (whose head is `head`) upright or `pre_rotation` turned them.
fn fit_orientation(blob: &mut [u8], head: &[u8], pre: &PreprocessingSettings) {
    let rotated_on_load = !pre.ignore_exif_orientation && jpeg_exif_orientation(head).is_some();
    if rotated_on_load || matches!(pre.pre_rotation, 90 | 180 | 270) {
        metadata::set_exif_orientation(blob, 1);
    }
}

/// Convert and encode the `dest` rectangle of `tile` into the interleaved
/// buffer `raw` (row stride `out_stride` samples), one output row per rayon
/// task. The rows are disjoint, so the result does not depend on the schedule.
fn paste_rows<T: Send>(
    raw: &mut [T],
    out_stride: usize,
    tile: &PlanarImage<f32>,
    (ix, iy): (usize, usize),
    (out_x, out_y, out_w, out_h): (usize, usize, usize, usize),
    encode_row: fn(&[f32], &mut [T]),
) {
    if out_w == 0 || out_h == 0 {
        return;
    }
    raw[out_y * out_stride..(out_y + out_h) * out_stride]
        .par_chunks_mut(out_stride)
        .enumerate()
        .for_each_init(
            || vec![0.0_f32; out_w * 3],
            |linear, (oy, row)| {
                let row_start = (iy + oy) * tile.width + ix;
                for (ox, px) in linear.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let i = row_start + ox;
                    *px = planar_to_linear_rgb([tile.luma[i], tile.chroma_a[i], tile.chroma_b[i]]);
                }
                encode_row(linear, &mut row[out_x * 3..(out_x + out_w) * 3]);
            },
        );
}

/// Encode `rgb` as a JPEG at `quality`.
fn save_jpeg(rgb: &RgbImage, path: &Path, quality: u8) -> Result<(), PipelineError> {
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    encode_jpeg(rgb, &mut writer, quality)?;
    writer.flush()?;
    Ok(())
}

/// Encode `rgb` as a JPEG at `quality` into `writer` (the encoder setup of
/// [`save_jpeg`] and of [`EncodedImage::to_bytes`]).
fn encode_jpeg<W: Write>(rgb: &RgbImage, writer: &mut W, quality: u8) -> Result<(), PipelineError> {
    let mut enc = JpegEncoder::new_with_quality(writer, quality);
    enc.encode_image(rgb)?;
    Ok(())
}

/// Round and clamp a resampled sample to the 8-bit range.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is rounded and clamped to 0..=255 first"
)]
const fn u8_from_f32(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Round and clamp a resampled sample to the 16-bit range.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is rounded and clamped to 0..=65535 first"
)]
const fn u16_from_f32(v: f32) -> u16 {
    v.round().clamp(0.0, 65_535.0) as u16
}

/// Resample an interleaved three-channel buffer `src = (w, h)` to `dst = (w, h)`
/// (both non-zero), each channel on its own with the Catmull-Rom kernel of
/// [`crate::crop::resize_planar`].
fn resize_interleaved<T: Copy + Default>(
    raw: &[T],
    src: (usize, usize),
    dst: (usize, usize),
    to_f32: impl Fn(T) -> f32,
    from_f32: impl Fn(f32) -> T,
) -> Vec<T> {
    let planes: Vec<Vec<f32>> = (0..3)
        .map(|channel| {
            let plane: Vec<f32> = raw
                .as_chunks::<3>()
                .0
                .iter()
                .map(|pixel| to_f32(pixel[channel]))
                .collect();
            resize_plane(&plane, src.0, src.1, dst.0, dst.1)
        })
        .collect();
    let mut out = vec![T::default(); dst.0 * dst.1 * 3];
    for (index, pixel) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        for (cell, plane) in pixel.iter_mut().zip(&planes) {
            *cell = from_f32(plane[index]);
        }
    }
    out
}

/// Crop an interleaved pixel buffer (stride = `full_w * channels`) to the
/// sub-rectangle `crop = (x, y, w, h)`.
fn crop_interleaved<T: Copy>(src: &[T], full_w: usize, channels: usize, crop: Rect) -> Vec<T> {
    let (cx, cy, cw, ch) = crop;
    let mut out = Vec::with_capacity(cw * ch * channels);
    for row in 0..ch {
        let src_start = ((cy + row) * full_w + cx) * channels;
        out.extend_from_slice(&src[src_start..src_start + cw * channels]);
    }
    out
}
