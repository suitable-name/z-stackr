//! Conversions between decoded images, the planar linear-light working format and
//! the 8-bit buffers Slint displays.

use image::DynamicImage;
use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_core::{
    color::{encode_row_u8, planar_to_linear_rgb},
    image::PlanarImage,
};

// ── Image conversion helpers ──────────────────────────────────────────────────

/// Convert a [`DynamicImage`] to a normalised `PlanarImage<f32>` in linear light.
///
/// The conversion (sRGB decode, then the `YCbCr`-style luma / `chroma_a` /
/// `chroma_b` matrix) is the pipeline's own,
/// [`stacker_pipeline::load::planar_from_dynamic`], so the desktop app and the
/// CLI cannot drift apart.
#[must_use]
pub fn load_as_planar(img: &DynamicImage) -> PlanarImage<f32> {
    stacker_pipeline::load::planar_from_dynamic(img)
}

/// Convert a linear-light `PlanarImage` to an RGBA byte buffer for Slint.
#[must_use]
pub fn planar_to_rgba_buffer(img: &PlanarImage<f32>) -> SharedPixelBuffer<Rgba8Pixel> {
    let w = u32::try_from(img.width).unwrap_or(u32::MAX);
    let h = u32::try_from(img.height).unwrap_or(u32::MAX);
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w, h);
    let mut linear = vec![0.0_f32; img.width * 3];
    let mut encoded = vec![0_u8; img.width * 3];
    for (row, px_row) in buf.make_mut_slice().chunks_exact_mut(img.width).enumerate() {
        let base = row * img.width;
        for (x, lin) in linear.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let i = base + x;
            // YCbCr → linear RGB; the row encoder does the rounded sRGB encode.
            *lin = planar_to_linear_rgb([img.luma[i], img.chroma_a[i], img.chroma_b[i]]);
        }
        encode_row_u8(&linear, &mut encoded);
        for (px, rgb) in px_row.iter_mut().zip(encoded.as_chunks::<3>().0) {
            *px = Rgba8Pixel::new(rgb[0], rgb[1], rgb[2], 255);
        }
    }
    buf
}
