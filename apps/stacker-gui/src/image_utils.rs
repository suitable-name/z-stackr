use image::{ColorType, DynamicImage, GenericImageView};
use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_core::image::PlanarImage;

// ── Image conversion helpers ──────────────────────────────────────────────────

/// Convert a [`DynamicImage`] to a normalised `PlanarImage<f32>` in gamma/sRGB space.
///
/// Pipeline:
/// 1. Normalise samples to `[0, 1]` using raw gamma-encoded values (÷255 or ÷65535 for 16-bit).
/// 2. Store as `YCbCr`-style luma / `chroma_a` / `chroma_b` using Rec.601 coefficients.
///    No transfer function is applied — all math operates directly on gamma-encoded values.
#[must_use]
pub fn load_as_planar(img: &DynamicImage) -> PlanarImage<f32> {
    let (w, h) = img.dimensions();
    let mut planar = PlanarImage::new(w as usize, h as usize);

    let is_16bit = matches!(
        img.color(),
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
    );

    if is_16bit {
        let rgb16 = img.to_rgb16();
        for (i, p) in rgb16.pixels().enumerate() {
            let r = f32::from(p[0]) / 65_535.0;
            let g = f32::from(p[1]) / 65_535.0;
            let b = f32::from(p[2]) / 65_535.0;
            planar.luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            planar.chroma_a[i] = -0.168_74 * r - 0.331_26 * g + 0.5 * b;
            planar.chroma_b[i] = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        }
    } else {
        let rgb8 = img.to_rgb8();
        for (i, p) in rgb8.pixels().enumerate() {
            let r = f32::from(p[0]) / 255.0;
            let g = f32::from(p[1]) / 255.0;
            let b = f32::from(p[2]) / 255.0;
            planar.luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
            planar.chroma_a[i] = -0.168_74 * r - 0.331_26 * g + 0.5 * b;
            planar.chroma_b[i] = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        }
    }

    planar
}

/// Convert a gamma-space `PlanarImage` to an 8-bit `image::RgbImage`.
///
/// Shared by the stack pipeline's temp-file encode and the save handler's
/// retouch-composite re-render, so the `YCbCr` → RGB math (identical to
/// [`planar_to_rgba_buffer`] below, minus the alpha channel) lives in one
/// place instead of being copy-pasted at every call site.
#[must_use]
pub fn planar_to_rgb_image(img: &PlanarImage<f32>) -> image::RgbImage {
    let w = img.width as u32;
    let h = img.height as u32;
    let mut out = image::RgbImage::new(w, h);
    for (i, p) in out.pixels_mut().enumerate() {
        let luma = img.luma[i];
        let cb = img.chroma_a[i];
        let cr = img.chroma_b[i];
        // YCbCr → gamma RGB: invert the matrix, clamp, quantise directly.
        let r = (luma + 1.402 * cr).clamp(0.0, 1.0);
        let g = (luma - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 1.0);
        let b = (luma + 1.772 * cb).clamp(0.0, 1.0);
        *p = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
    }
    out
}

/// Convert a gamma-space `PlanarImage` to an RGBA byte buffer for Slint.
#[must_use]
pub fn planar_to_rgba_buffer(img: &PlanarImage<f32>) -> SharedPixelBuffer<Rgba8Pixel> {
    let w = img.width as u32;
    let h = img.height as u32;
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w, h);
    for (i, px) in buf.make_mut_slice().iter_mut().enumerate() {
        let luma = img.luma[i];
        let cb = img.chroma_a[i];
        let cr = img.chroma_b[i];
        // YCbCr → gamma RGB: invert the YCbCr matrix, clamp, quantise directly.
        let r = (luma + 1.402 * cr).clamp(0.0, 1.0);
        let g = (luma - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 1.0);
        let b = (luma + 1.772 * cb).clamp(0.0, 1.0);
        *px = Rgba8Pixel::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255);
    }
    buf
}

/// Convert a gamma-space `PlanarImage` to an RGBA byte buffer for Slint, with
/// an optional display-only "painted area" tint.
///
/// `alpha`, when `Some`, is the retouch session's per-pixel mask (same
/// dimensions as `img`, one value per pixel in `[0, 1]`). Wherever `alpha` is
/// non-zero the resulting pixel is blended towards a fixed magenta highlight
/// by `0.5 * alpha`, so the user can see exactly where they have brushed.
/// This is purely a preview aid: it never touches `img` itself, so callers
/// that re-encode `img` for saving (via [`planar_to_rgb_image`]) are
/// unaffected regardless of whether this overlay is shown.
///
/// Passing `None` for `alpha` is identical to [`planar_to_rgba_buffer`].
#[must_use]
pub fn planar_to_rgba_buffer_with_overlay(
    img: &PlanarImage<f32>,
    alpha: Option<&[f32]>,
) -> SharedPixelBuffer<Rgba8Pixel> {
    // Highlight colour for the "show painted area" overlay: magenta reads
    // clearly against both dark and bright photographic content.
    const TINT_R: f32 = 1.0;
    const TINT_G: f32 = 0.0;
    const TINT_B: f32 = 1.0;

    let w = img.width as u32;
    let h = img.height as u32;
    let mut buf = SharedPixelBuffer::<Rgba8Pixel>::new(w, h);
    for (i, px) in buf.make_mut_slice().iter_mut().enumerate() {
        let luma = img.luma[i];
        let cb = img.chroma_a[i];
        let cr = img.chroma_b[i];
        // YCbCr → gamma RGB: invert the YCbCr matrix, clamp, quantise directly.
        let mut r = (luma + 1.402 * cr).clamp(0.0, 1.0);
        let mut g = (luma - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 1.0);
        let mut b = (luma + 1.772 * cb).clamp(0.0, 1.0);

        if let Some(alpha) = alpha {
            let a = alpha[i] * 0.5;
            if a > 0.0 {
                r += (TINT_R - r) * a;
                g += (TINT_G - g) * a;
                b += (TINT_B - b) * a;
            }
        }

        *px = Rgba8Pixel::new((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255);
    }
    buf
}
