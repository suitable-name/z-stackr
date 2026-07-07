/// Output encoding: planar f32 → sRGB integer buffers, tile-direct writers.
///
/// # Memory profile (tiled path)
///
/// The output `ImageBuffer<Rgb<u8/u16>>` is the ONLY full-image allocation in
/// the tiled path:
///   - 8-bit:  W × H × 3 bytes        (e.g. ≈ 50 MB for 4096² image)
///   - 16-bit: W × H × 6 bytes        (e.g. ≈ 100 MB for 4096² image)
///
/// The per-tile f32 planar buffer (`fused_padded`) is `tile_area` × 3 × 4 bytes
/// (e.g. ≈ 3 MB for a 512 px tile) and is dropped after each tile — no
/// full-resolution `PlanarImage<f32>` accumulator is ever allocated in
/// addition to the output buffer.
///
/// Peak RAM during fusion scales with `TILE_AREA`, not `IMAGE_AREA`.
use image::{ImageBuffer, Rgb, RgbImage};
use stacker_core::image::PlanarImage;
use std::path::Path;

/// Returns `true` when the output path's extension implies a format that the
/// `image` crate can encode as a 16-bit RGB image.
pub fn supports_16bit_output(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("png" | "tif" | "tiff")
    )
}

/// Shared YCbCr-to-gamma-RGB reconstruction used by both output functions.
///
/// Inverts the YCbCr matrix to recover gamma-encoded RGB clamped to `[0.0, 1.0]`.
#[inline]
#[must_use]
pub fn planar_to_gamma_rgb(img: &PlanarImage<f32>, i: usize) -> (f32, f32, f32) {
    let luma = img.luma[i];
    let cb = img.chroma_a[i];
    let cr = img.chroma_b[i];
    let r = (luma + 1.402 * cr).clamp(0.0, 1.0);
    let g = (luma - 0.344_136 * cb - 0.714_136 * cr).clamp(0.0, 1.0);
    let b = (luma + 1.772 * cb).clamp(0.0, 1.0);
    (r, g, b)
}

// ── Tile-direct output writers ────────────────────────────────────────────────
//
// These functions write a fused (padded) tile's **interior** directly into a
// pre-allocated output `ImageBuffer`, performing the f32 planar → sRGB integer
// conversion per pixel as they go.

/// Write the interior of a fused (padded) tile into an 8-bit sRGB output buffer.
///
/// `pad_off` is `(ix_in_pad, iy_in_pad)` — the offset of the interior within
/// the padded tile (from [`TileCoordinate::interior_offset_in_padded`]).
/// `(out_x, out_y, out_w, out_h)` is the destination rectangle in `output`.
pub fn paste_tile_to_rgb8(
    output: &mut RgbImage,
    tile: &PlanarImage<f32>,
    pad_off: (usize, usize),
    out_x: usize,
    out_y: usize,
    out_w: usize,
    out_h: usize,
) {
    let (ix, iy) = pad_off;
    let src_stride = tile.width;
    for oy in 0..out_h {
        let sy = iy + oy;
        for ox in 0..out_w {
            let sx = ix + ox;
            let idx = sy * src_stride + sx;
            let (r, g, b) = planar_to_gamma_rgb(tile, idx);
            let px = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
            output.put_pixel((out_x + ox) as u32, (out_y + oy) as u32, px);
        }
    }
}

/// Write the interior of a fused (padded) tile into a 16-bit sRGB output buffer.
///
/// `pad_off` is `(ix_in_pad, iy_in_pad)`.  Identical to [`paste_tile_to_rgb8`]
/// but quantises to `u16`.
pub fn paste_tile_to_rgb16(
    output: &mut ImageBuffer<Rgb<u16>, Vec<u16>>,
    tile: &PlanarImage<f32>,
    pad_off: (usize, usize),
    out_x: usize,
    out_y: usize,
    out_w: usize,
    out_h: usize,
) {
    let (ix, iy) = pad_off;
    let src_stride = tile.width;
    for oy in 0..out_h {
        let sy = iy + oy;
        for ox in 0..out_w {
            let sx = ix + ox;
            let idx = sy * src_stride + sx;
            let (r, g, b) = planar_to_gamma_rgb(tile, idx);
            let px = image::Rgb([
                (r * 65_535.0) as u16,
                (g * 65_535.0) as u16,
                (b * 65_535.0) as u16,
            ]);
            output.put_pixel((out_x + ox) as u32, (out_y + oy) as u32, px);
        }
    }
}
