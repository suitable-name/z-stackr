//! Small pictures made from planar images: the worker's thumbnails and the
//! `Relief` preview data.
//!
//! The same conversions the desktop app uses for its previews (a box average
//! in linear light, then the rounded sRGB encode), written over plain slices so
//! the worker and the tests share them.

use stacker_core::{
    color::{encode_row_u8, planar_to_linear_rgb},
    image::PlanarImage,
};

/// A picture of 8-bit RGBA pixels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rgba {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes, alpha 255.
    pub rgba: Vec<u8>,
}

/// The size of a thumbnail of a `width x height` picture whose longest edge is at
/// most `max_edge`. A picture that is small enough keeps its size; neither side
/// goes below one pixel.
#[must_use]
pub fn thumbnail_size(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    let long = width.max(height);
    if width == 0 || height == 0 || max_edge == 0 || long <= max_edge {
        return (width, height);
    }
    let scaled = |side: u32| {
        let value = u64::from(side) * u64::from(max_edge) / u64::from(long);
        u32::try_from(value).unwrap_or(max_edge).max(1)
    };
    (scaled(width), scaled(height))
}

/// The source range `[start, end)` along an axis of `source` pixels that output
/// pixel `index` of `target` covers (at least one pixel).
fn span(index: usize, source: usize, target: usize) -> (usize, usize) {
    let start = (index * source / target).min(source - 1);
    let end = ((index + 1) * source / target).clamp(start + 1, source);
    (start, end)
}

/// Box-average one plane of `width x height` samples down to `tw x th`.
fn box_average(plane: &[f32], width: usize, height: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(tw * th);
    for ty in 0..th {
        let (y0, y1) = span(ty, height, th);
        for tx in 0..tw {
            let (x0, x1) = span(tx, width, tw);
            let mut sum = 0.0_f64;
            for y in y0..y1 {
                let row = &plane[y * width + x0..y * width + x1];
                sum += row.iter().map(|&v| f64::from(v)).sum::<f64>();
            }
            let count = (y1 - y0) * (x1 - x0);
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                reason = "a box holds far fewer than 2^52 samples; the average is stored as f32"
            )]
            out.push((sum / count as f64) as f32);
        }
    }
    out
}

/// `plane` (`width x height`, row-major) reduced so its longest edge is at most
/// `max_edge`; returns the new size and the averaged samples. An empty or
/// inconsistent input gives `(0, 0, [])`.
#[must_use]
pub fn downsample_plane(
    plane: &[f32],
    width: usize,
    height: usize,
    max_edge: u32,
) -> (u32, u32, Vec<f32>) {
    let (Ok(w32), Ok(h32)) = (u32::try_from(width), u32::try_from(height)) else {
        return (0, 0, Vec::new());
    };
    if width == 0 || height == 0 || plane.len() < width * height {
        return (0, 0, Vec::new());
    }
    let (tw, th) = thumbnail_size(w32, h32, max_edge);
    let samples = box_average(plane, width, height, tw as usize, th as usize);
    (tw, th, samples)
}

/// `image` as an sRGB 8-bit RGBA picture whose longest edge is at most `max_edge`
/// (the box average is taken in linear light).
#[must_use]
pub fn planar_to_rgba(image: &PlanarImage<f32>, max_edge: u32) -> Rgba {
    let (Ok(w32), Ok(h32)) = (u32::try_from(image.width), u32::try_from(image.height)) else {
        return Rgba::default();
    };
    if !image.is_consistent() || image.width == 0 || image.height == 0 {
        return Rgba::default();
    }
    let (tw, th) = thumbnail_size(w32, h32, max_edge);
    let (cols, rows) = (tw as usize, th as usize);
    let luma = box_average(&image.luma, image.width, image.height, cols, rows);
    let chroma_a = box_average(&image.chroma_a, image.width, image.height, cols, rows);
    let chroma_b = box_average(&image.chroma_b, image.width, image.height, cols, rows);

    let mut rgba = vec![255_u8; cols * rows * 4];
    let mut linear = vec![0.0_f32; cols * 3];
    let mut encoded = vec![0_u8; cols * 3];
    for (row, out_row) in rgba.chunks_exact_mut(cols * 4).enumerate() {
        let base = row * cols;
        for (x, lin) in linear.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let i = base + x;
            *lin = planar_to_linear_rgb([luma[i], chroma_a[i], chroma_b[i]]);
        }
        encode_row_u8(&linear, &mut encoded);
        for (px, rgb) in out_row
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(encoded.as_chunks::<3>().0)
        {
            px[..3].copy_from_slice(rgb);
        }
    }
    Rgba {
        width: tw,
        height: th,
        rgba,
    }
}

#[cfg(test)]
mod tests {
    use super::{planar_to_rgba, thumbnail_size};
    use stacker_core::image::PlanarImage;

    #[test]
    fn a_small_picture_keeps_its_size() {
        assert_eq!(thumbnail_size(300, 200, 640), (300, 200));
        assert_eq!(thumbnail_size(0, 0, 640), (0, 0));
    }

    #[test]
    fn a_large_picture_shrinks_to_the_longest_edge() {
        assert_eq!(thumbnail_size(6000, 4000, 600), (600, 400));
        assert_eq!(thumbnail_size(4000, 6000, 600), (400, 600));
        assert_eq!(thumbnail_size(10_000, 1, 100), (100, 1));
    }

    #[test]
    fn white_stays_white_and_shrinks() {
        let mut image = PlanarImage::new(8, 4);
        image.luma.fill(1.0);
        let thumb = planar_to_rgba(&image, 4);
        assert_eq!((thumb.width, thumb.height), (4, 2));
        assert_eq!(thumb.rgba.len(), 4 * 2 * 4);
        assert!(thumb.rgba.chunks(4).all(|px| px == [255, 255, 255, 255]));
    }
}
