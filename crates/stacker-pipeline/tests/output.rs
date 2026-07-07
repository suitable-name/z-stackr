use image::{ImageBuffer, Rgb, RgbImage};
use stacker_core::image::PlanarImage;
use stacker_pipeline::output::*;

// ── Test-only whole-image encoders ────────────────────────────────────────────

/// Convert a `PlanarImage<f32>` (gamma-space luma/chroma) back to an 8-bit
/// gamma [`RgbImage`].
///
/// Reconstruction order:
/// 1. Invert the YCbCr matrix to recover gamma-encoded RGB.
/// 2. Quantise to `u8` by multiplying by 255 (values already clamped to [0,1]).
///
/// Used by tests to generate a reference whole-image encoding.
#[must_use]
// w/h/r/g/b are the conventional, clearest names for width/height/red/green/blue
// here; renaming would hurt readability for a purely cosmetic pedantic lint.
#[allow(clippy::many_single_char_names)]
pub fn planar_to_rgb(img: &PlanarImage<f32>) -> RgbImage {
    let w = img.width as u32;
    let h = img.height as u32;
    let mut rgb = RgbImage::new(w, h);
    for (i, p) in rgb.pixels_mut().enumerate() {
        let (r, g, b) = planar_to_gamma_rgb(img, i);
        p[0] = (r * 255.0) as u8;
        p[1] = (g * 255.0) as u8;
        p[2] = (b * 255.0) as u8;
    }
    rgb
}

/// Convert a `PlanarImage<f32>` (gamma-space luma/chroma) back to a 16-bit
/// gamma [`ImageBuffer`].
///
/// Identical pipeline to [`planar_to_rgb`] but quantises to `u16` (×65 535),
/// preserving sub-8-bit tonal distinctions that would be lost at 8 bits.
///
/// Used by tests.
#[must_use]
// w/h/r/g/b are the conventional, clearest names for width/height/red/green/blue
// here; renaming would hurt readability for a purely cosmetic pedantic lint.
#[allow(clippy::many_single_char_names)]
pub fn planar_to_rgb16(img: &PlanarImage<f32>) -> ImageBuffer<Rgb<u16>, Vec<u16>> {
    let w = img.width as u32;
    let h = img.height as u32;
    let mut buf: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(w, h);
    for (i, p) in buf.pixels_mut().enumerate() {
        let (r, g, b) = planar_to_gamma_rgb(img, i);
        p[0] = (r * 65_535.0) as u16;
        p[1] = (g * 65_535.0) as u16;
        p[2] = (b * 65_535.0) as u16;
    }
    buf
}

/// A small greyscale ramp (zero chroma) — cheap to reason about exactly.
fn make_test_image(w: usize, h: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);

    let len = (w * h) as f32;
    for i in 0..w * h {
        let v = (i as f32 / len).clamp(0.0, 1.0);
        img.luma[i] = v;
        img.chroma_a[i] = 0.0;
        img.chroma_b[i] = 0.0;
    }
    img
}

#[test]
fn planar_to_rgb_preserves_dimensions_and_is_grayscale() {
    let img = make_test_image(4, 3);
    let rgb = planar_to_rgb(&img);
    assert_eq!(rgb.width(), 4);
    assert_eq!(rgb.height(), 3);
    // Zero chroma => R == G == B (grayscale) at every pixel.
    for p in rgb.pixels() {
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
    }
}

#[test]
fn planar_to_rgb16_agrees_with_8bit_within_rounding() {
    let img = make_test_image(4, 3);
    let rgb8 = planar_to_rgb(&img);
    let rgb16 = planar_to_rgb16(&img);
    assert_eq!(rgb16.width(), rgb8.width());
    assert_eq!(rgb16.height(), rgb8.height());
    for (p8, p16) in rgb8.pixels().zip(rgb16.pixels()) {
        // Same source value quantised at two different bit depths should
        // agree to within 8-bit rounding error once rescaled.

        let r8_from_16 = (f32::from(p16[0]) / 65_535.0 * 255.0).round() as u8;
        assert!((i16::from(r8_from_16) - i16::from(p8[0])).abs() <= 1);
    }
}
