//! Tile-based out-of-core image storage.
//!
//! [`TileManager`] serialises tiles to temporary binary files and reads
//! them back on demand. Each plane is read/written as a single bulk byte copy
//! via [`bytemuck`] (no per-element conversion), in **native endianness** — the
//! tile files are ephemeral scratch for one run on one machine, so they are
//! never shared across architectures.
//!
//! `mmap(2)` is intentionally *not* used: the access pattern is
//! write-once / read-once and `PlanarImage` owns its buffers, so a memory map
//! would still be copied into the owned `Vec`s — capturing essentially none of
//! mmap's zero-copy/random-access benefit while adding `unsafe` and a mapping
//! lifetime to manage.
//!
//! ## Apron / overlap strategy
//!
//! Neighbourhood operations (Laplacian pyramid, SML, guided filter, and the
//! neural `FocusMergeNet`) need pixels outside the strict tile boundary to avoid
//! seams.  Callers should request tiles with a padding **apron** of at least
//! `APRON_PX` pixels on every side:
//!
//! ```text
//! apron = max(pyramid blur support, 2 * guided_radius, NN receptive field)
//!       = max(64, 16, ~100) = 128 pixels
//! ```
//!
//! After fusing the padded tile the caller crops the interior `(start_x, start_y,
//! width, height)` back out and splices it into the output buffer.
//! [`extract_tile`] handles the per-frame crop + apron clamping.
//! [`paste_interior`] splices a fused tile back into the output.

#![allow(clippy::missing_errors_doc)]

use std::{collections::HashMap, path::PathBuf};

use crate::{error::StackerError, image::PlanarImage};

/// Minimum apron (halo) in pixels on each side of a tile.
///
/// Derivation (the apron must cover the widest neighbourhood any algorithm
/// reads at a tile boundary):
/// - `Apex` Laplacian pyramid: deep levels have a blur support on the order of
///   64 px in the original image coordinate system.
/// - SML radius is typically 3 (support 7 px) — safely covered.
/// - Guided filter radius is typically 8 (support 17 px) — safely covered.
/// - The neural `FocusMergeNet` (largest `xxl` preset) has a receptive-field
///   radius of ~100 px (dilated-residual context stack `1→2→4→8`), so the AI
///   stacking mode needs the widest apron.
///
/// 128 px covers all of the above, ensuring no visible seams at tile boundaries
/// for any fusion mode, including AI at the largest preset.
pub const APRON_PX: usize = 128;

/// A tile coordinate identifying a rectangular region within an image.
///
/// The coordinate describes the **interior** region that will be written into
/// the output; the actual pixels fetched / fused include an `APRON_PX` halo.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileCoordinate {
    pub start_x: usize,
    pub start_y: usize,
    pub width: usize,
    pub height: usize,
}

impl TileCoordinate {
    /// Compute the padded region (with apron) clamped to `[0, img_w) × [0, img_h)`.
    ///
    /// Returns `(pad_x, pad_y, pad_w, pad_h)`.
    #[must_use]
    pub fn padded_region(&self, img_w: usize, img_h: usize) -> (usize, usize, usize, usize) {
        let pad_x = self.start_x.saturating_sub(APRON_PX);
        let pad_y = self.start_y.saturating_sub(APRON_PX);
        let right = (self.start_x + self.width + APRON_PX).min(img_w);
        let bottom = (self.start_y + self.height + APRON_PX).min(img_h);
        let pad_w = right.saturating_sub(pad_x);
        let pad_h = bottom.saturating_sub(pad_y);
        (pad_x, pad_y, pad_w, pad_h)
    }

    /// Offset of the interior within the padded tile (pixels from top-left of padded tile).
    ///
    /// Use this to crop the fused padded tile before pasting into the output.
    #[must_use]
    pub fn interior_offset_in_padded(&self, img_w: usize, img_h: usize) -> (usize, usize) {
        let (pad_x, pad_y, _, _) = self.padded_region(img_w, img_h);
        (self.start_x - pad_x, self.start_y - pad_y)
    }
}

/// Enumerate all non-overlapping tile coordinates for an image of size
/// `img_w × img_h` with a maximum tile size of `tile_size`.
///
/// Tiles at the right/bottom edge are smaller to exactly cover the image.
#[must_use]
pub fn enumerate_tiles(img_w: usize, img_h: usize, tile_size: usize) -> Vec<TileCoordinate> {
    let ts = tile_size.max(1);
    let mut tiles = Vec::new();
    let mut y = 0;
    while y < img_h {
        let h = (y + ts).min(img_h) - y;
        let mut x = 0;
        while x < img_w {
            let w = (x + ts).min(img_w) - x;
            tiles.push(TileCoordinate {
                start_x: x,
                start_y: y,
                width: w,
                height: h,
            });
            x += w;
        }
        y += h;
    }
    tiles
}

/// Extract a rectangular crop from `img` at `(crop_x, crop_y, crop_w, crop_h)`,
/// clamping coordinates to the image bounds (border pixels are replicated).
///
/// Used to slice a padded tile region out of an aligned full-frame image.
#[must_use]
pub fn extract_tile(
    img: &PlanarImage<f32>,
    crop_x: usize,
    crop_y: usize,
    crop_w: usize,
    crop_h: usize,
) -> PlanarImage<f32> {
    let src_w = img.width;
    let src_h = img.height;
    let mut out = PlanarImage::new(crop_w, crop_h);

    for oy in 0..crop_h {
        let sy = (crop_y + oy).min(src_h.saturating_sub(1));
        for ox in 0..crop_w {
            let sx = (crop_x + ox).min(src_w.saturating_sub(1));
            let src_idx = sy * src_w + sx;
            let dst_idx = oy * crop_w + ox;
            out.luma[dst_idx] = img.luma[src_idx];
            out.chroma_a[dst_idx] = img.chroma_a[src_idx];
            out.chroma_b[dst_idx] = img.chroma_b[src_idx];
        }
    }
    out
}

/// Paste the interior region of a fused (padded) tile into an output image.
///
/// `padded_tile` is the fused result including the apron.
/// `(ix_in_pad, iy_in_pad)` is the top-left of the interior within the padded tile
/// (from [`TileCoordinate::interior_offset_in_padded`]).
/// `(out_x, out_y, out_w, out_h)` is the destination rectangle in `output`.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn paste_interior(
    output: &mut PlanarImage<f32>,
    padded_tile: &PlanarImage<f32>,
    ix_in_pad: usize,
    iy_in_pad: usize,
    out_x: usize,
    out_y: usize,
    out_w: usize,
    out_h: usize,
) {
    let dst_stride = output.width;
    let src_stride = padded_tile.width;

    for oy in 0..out_h {
        let sy = iy_in_pad + oy;
        for ox in 0..out_w {
            let sx = ix_in_pad + ox;
            let src_idx = sy * src_stride + sx;
            let dst_idx = (out_y + oy) * dst_stride + (out_x + ox);
            output.luma[dst_idx] = padded_tile.luma[src_idx];
            output.chroma_a[dst_idx] = padded_tile.chroma_a[src_idx];
            output.chroma_b[dst_idx] = padded_tile.chroma_b[src_idx];
        }
    }
}

/// Trait for reading and writing image tiles.
pub trait TileProvider {
    fn fetch_tile(
        &self,
        img_idx: usize,
        coord: &TileCoordinate,
    ) -> Result<PlanarImage<f32>, StackerError>;
    fn commit_tile(
        &mut self,
        img_idx: usize,
        coord: &TileCoordinate,
        tile: PlanarImage<f32>,
    ) -> Result<(), StackerError>;
}

/// File-backed tile manager.
///
/// Each tile is stored as a flat binary file containing the three planar
/// channels (`luma`, `chroma_a`, `chroma_b`) concatenated in that order, each
/// written as a single bulk copy of its raw `f32` bytes in native endianness
/// (the files are ephemeral single-run scratch, never shared across machines).
///
/// ## I/O strategy
///
/// The `TileProvider` trait is synchronous (returns `Result`, not `Future`) so
/// callers need not be async and the trait stays object-safe without
/// `async_trait` or GATs.  Tiles are read and written with **plain blocking
/// `std::fs`** calls: the pipeline already drives its own async boundary via
/// a single outer `smol::block_on`, so routing individual tile reads/writes
/// through `smol::unblock` would only add a thread-pool round-trip (and a
/// nested `block_on`) per tile with no actual concurrency gained, since the
/// calling thread is parked either way.
///
/// To actually overlap disk I/O with fusion the right approach is to prefetch
/// the *next* tile's frames while the current tile is being fused (double
/// buffering).  That is a deliberate pipeline-level concern, not something this
/// leaf storage type should fake, so it is left as a future enhancement.
///
/// ## Why not `mmap`?
/// The access pattern is write-once / read-once and sequential, and
/// [`PlanarImage`] owns its channel `Vec`s — so a real memory map would still be
/// copied into those owned buffers, capturing almost none of mmap's benefit
/// while adding `unsafe` and a mapping lifetime. A bulk `bytemuck` byte-copy of
/// each plane is therefore used instead.
pub struct TileManager {
    pub temp_dir: PathBuf,
    pub tiles: HashMap<(usize, TileCoordinate), PathBuf>,
}

impl TileProvider for TileManager {
    fn fetch_tile(
        &self,
        img_idx: usize,
        coord: &TileCoordinate,
    ) -> Result<PlanarImage<f32>, StackerError> {
        let path = self
            .tiles
            .get(&(img_idx, coord.clone()))
            .ok_or_else(|| {
                StackerError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "tile not found",
                ))
            })?
            .clone();

        let data = std::fs::read(&path)?;

        let len = coord.width * coord.height;
        let expected = len * 4 * 3; // 3 channels × 4 bytes per f32
        if data.len() != expected {
            return Err(StackerError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("tile file has {} bytes, expected {}", data.len(), expected),
            )));
        }

        let mut img = PlanarImage::new(coord.width, coord.height);

        // Bulk byte-copy each plane into its `f32` buffer (native endianness).
        // `cast_slice_mut::<f32, u8>` is always alignment-safe (the destination
        // is `f32`-aligned), and the copy is a single `memcpy` per plane — no
        // per-element `from_le_bytes` decode.
        let load = |dst: &mut [f32], src: &[u8]| {
            bytemuck::cast_slice_mut::<f32, u8>(dst).copy_from_slice(src);
        };
        load(&mut img.luma, &data[0..len * 4]);
        load(&mut img.chroma_a, &data[len * 4..len * 8]);
        load(&mut img.chroma_b, &data[len * 8..len * 12]);

        Ok(img)
    }

    fn commit_tile(
        &mut self,
        img_idx: usize,
        coord: &TileCoordinate,
        tile: PlanarImage<f32>,
    ) -> Result<(), StackerError> {
        let filename = format!(
            "tile_{}_{}_{}_{}_{}.bin",
            img_idx, coord.start_x, coord.start_y, coord.width, coord.height
        );
        let path = self.temp_dir.join(filename);

        // Serialise the three planes into one flat byte buffer with a single
        // bulk `memcpy` per plane (`cast_slice::<f32, u8>` reinterprets the
        // owned `f32` data as bytes), then write it in one syscall — no
        // per-element `to_le_bytes` serialisation.
        let mut buf = Vec::with_capacity(tile.luma.len() * 4 * 3);
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&tile.luma));
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&tile.chroma_a));
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(&tile.chroma_b));

        std::fs::write(&path, &buf)?;

        self.tiles.insert((img_idx, coord.clone()), path);
        Ok(())
    }
}
