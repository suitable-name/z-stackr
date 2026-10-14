//! Tile-based out-of-core image storage.
//!
//! [`MmapTileManager`] serialises tiles to temporary binary files and reads
//! them back on demand. Each plane is read/written as a single bulk byte copy
//! via [`bytemuck`] (no per-element conversion), in **native endianness** — the
//! tile files are ephemeral scratch for one run on one machine, so they are
//! never shared across architectures.
//!
//! Real `mmap(2)` is intentionally *not* used: the access pattern is
//! write-once / read-once and `PlanarImage` owns its buffers, so a memory map
//! would still be copied into the owned `Vec`s — capturing essentially none of
//! mmap's zero-copy/random-access benefit while adding `unsafe` and a mapping
//! lifetime to manage. (The `Mmap` prefix in the type name is historical.)
//!
//! ## Apron / overlap strategy
//!
//! Neighbourhood operations (Laplacian pyramid, SML, guided filter, and the
//! neural `FocusMergeNet`) need pixels outside the strict tile boundary to avoid
//! seams.  The tile that is stored and fused is the interior grown by a padding
//! **apron** on every side, clamped to the image. How wide the apron must be
//! depends on the operation, and for the pyramid on its depth:
//!
//! ```text
//! Apex   apron = 4·(2^(L−1) − 1) + 1, rounded up to a multiple of 2^(L−1)
//!               (L pyramid levels: 128 for 6, 256 for 7, 512 for 8;
//!                stacker_algo::apex::fuse::apron_for_levels)
//! Relief   apron = estimation radius + 2 · smoothing radius + 1
//! AI     apron = APRON_PX (≈ 100 px receptive field, rounded up)
//! ```
//!
//! [`APRON_PX`] is the apron for the neural mode and the fallback for callers
//! that do not know better; the `_with` variants of the padding functions take
//! the apron explicitly.
//!
//! After fusing the padded tile the caller crops the interior `(start_x, start_y,
//! width, height)` back out and splices it into the output buffer.
//! [`extract_tile`] handles the per-frame crop + apron clamping.
//! [`paste_interior`] splices a fused tile back into the output.

use std::{collections::HashMap, path::PathBuf};

use crate::{error::StackerError, image::PlanarImage};

/// Apron (halo) in pixels on each side of a tile for callers that do not derive
/// their own, and the apron of the neural mode.
///
/// 100 px is the receptive-field radius of the neural `FocusMergeNet` at its
/// largest (`xxl`) preset (dilated-residual context stack `1→2→4→8`); 128 is
/// that figure rounded up to a power of two. It does **not** cover the `Apex`
/// pyramid at its default depth (that needs 512 px at 8 levels) or large `Relief`
/// radii: see the module documentation and
/// `stacker_algo::apex::fuse::apron_for_levels`.
pub const APRON_PX: usize = 128;

/// A rectangle of an image, used as the key of a stored tile.
///
/// The store does not interpret the rectangle: it is whatever region the caller
/// commits and later fetches, and the planes of a committed tile must match its
/// size. The CLI keys its tiles by the **padded** rectangle (interior plus
/// apron, clamped to the image) and keeps the interior rectangle separately for
/// writing the output; [`Self::padded_region_with`] and
/// [`Self::interior_offset_in_padded_with`] compute the former from the latter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileCoordinate {
    /// Left edge in pixels.
    pub start_x: usize,
    /// Top edge in pixels.
    pub start_y: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl TileCoordinate {
    /// The padded region with the default [`APRON_PX`] apron.
    ///
    /// Returns `(pad_x, pad_y, pad_w, pad_h)`.
    #[must_use]
    pub fn padded_region(&self, img_w: usize, img_h: usize) -> (usize, usize, usize, usize) {
        self.padded_region_with(img_w, img_h, APRON_PX)
    }

    /// The region of this tile grown by `apron` pixels on every side and
    /// clamped to `[0, img_w) × [0, img_h)`.
    ///
    /// Returns `(pad_x, pad_y, pad_w, pad_h)`.
    #[must_use]
    pub fn padded_region_with(
        &self,
        img_w: usize,
        img_h: usize,
        apron: usize,
    ) -> (usize, usize, usize, usize) {
        let pad_x = self.start_x.saturating_sub(apron);
        let pad_y = self.start_y.saturating_sub(apron);
        let right = (self.start_x + self.width + apron).min(img_w);
        let bottom = (self.start_y + self.height + apron).min(img_h);
        let pad_w = right.saturating_sub(pad_x);
        let pad_h = bottom.saturating_sub(pad_y);
        (pad_x, pad_y, pad_w, pad_h)
    }

    /// Offset of the interior within the padded tile (pixels from top-left of
    /// padded tile) for the default [`APRON_PX`] apron.
    ///
    /// Use this to crop the fused padded tile before pasting into the output.
    #[must_use]
    pub fn interior_offset_in_padded(&self, img_w: usize, img_h: usize) -> (usize, usize) {
        self.interior_offset_in_padded_with(img_w, img_h, APRON_PX)
    }

    /// [`Self::interior_offset_in_padded`] for an explicit `apron`.
    #[must_use]
    pub fn interior_offset_in_padded_with(
        &self,
        img_w: usize,
        img_h: usize,
        apron: usize,
    ) -> (usize, usize) {
        let (pad_x, pad_y, _, _) = self.padded_region_with(img_w, img_h, apron);
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
/// Used to slice a padded tile region out of an aligned full-frame image. Each
/// output row is one contiguous copy of the in-bounds run plus a fill of the
/// replicated border, not a per-pixel gather. An empty source or crop yields an
/// all-zero image of the requested size.
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
    if crop_w == 0 || crop_h == 0 || src_w == 0 || src_h == 0 {
        return out;
    }

    // Columns `crop_x ..` that exist in the source are one contiguous run per row;
    // everything right of it repeats the last source column (clamped replicate).
    let last_x = src_w - 1;
    let run = src_w.saturating_sub(crop_x).min(crop_w);
    for oy in 0..crop_h {
        let sy = (crop_y + oy).min(src_h - 1);
        let src_row = sy * src_w;
        let dst_row = oy * crop_w;
        let planes = [
            (&img.luma, &mut out.luma),
            (&img.chroma_a, &mut out.chroma_a),
            (&img.chroma_b, &mut out.chroma_b),
        ];
        for (src, dst) in planes {
            let dst = &mut dst[dst_row..dst_row + crop_w];
            if run > 0 {
                dst[..run].copy_from_slice(&src[src_row + crop_x..src_row + crop_x + run]);
            }
            dst[run..].fill(src[src_row + last_x]);
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
#[expect(
    clippy::too_many_arguments,
    clippy::similar_names,
    reason = "two (x, y) offsets plus a destination rectangle; a struct would only move the names"
)]
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

    if out_w == 0 {
        return;
    }

    // One contiguous copy per row and plane (the rows of both rectangles are
    // contiguous runs of `out_w` samples).
    for oy in 0..out_h {
        let src_start = (iy_in_pad + oy) * src_stride + ix_in_pad;
        let dst_start = (out_y + oy) * dst_stride + out_x;
        output.luma[dst_start..dst_start + out_w]
            .copy_from_slice(&padded_tile.luma[src_start..src_start + out_w]);
        output.chroma_a[dst_start..dst_start + out_w]
            .copy_from_slice(&padded_tile.chroma_a[src_start..src_start + out_w]);
        output.chroma_b[dst_start..dst_start + out_w]
            .copy_from_slice(&padded_tile.chroma_b[src_start..src_start + out_w]);
    }
}

/// Trait for reading and writing image tiles.
pub trait TileProvider {
    /// Read back the tile stored for frame `img_idx` at `coord`.
    ///
    /// # Errors
    /// [`StackerError::TileNotFound`] when no such tile was committed, or an
    /// I/O error when the backing store cannot be read.
    fn fetch_tile(
        &self,
        img_idx: usize,
        coord: &TileCoordinate,
    ) -> Result<PlanarImage<f32>, StackerError>;

    /// Store `tile` for frame `img_idx` at `coord`, replacing any earlier one.
    ///
    /// # Errors
    /// An I/O error when the backing store cannot be written.
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
/// `std::fs`** calls.
///
/// The calls block the calling thread; they are not wrapped in an executor
/// (`block_on` of an off-loaded task would park the caller anyway and only add a
/// thread-pool round-trip per tile).
///
/// Overlapping disk I/O with fusion would take a prefetch of the *next* tile's
/// frames while the current tile is being fused (double buffering). That belongs
/// to the pipeline that drives the loop, not to this leaf storage type, and is
/// not implemented.
///
/// ## Why not `mmap`?
/// The access pattern is write-once / read-once and sequential, and
/// [`PlanarImage`] owns its channel `Vec`s — so a real memory map would still be
/// copied into those owned buffers, capturing almost none of mmap's benefit
/// while adding `unsafe` and a mapping lifetime. A bulk `bytemuck` byte-copy of
/// each plane is therefore used instead. (The `Mmap` prefix is historical.)
///
/// ## Contract
///
/// * A tile is keyed by `(img_idx, coord)`; `coord` is just the rectangle the
///   caller stores (see [`TileCoordinate`]), and the committed planes must hold
///   exactly `coord.width * coord.height` samples each. [`commit_tile`] rejects
///   anything else, so a tile can never be read back misaligned. Committing a
///   key twice replaces the earlier tile.
/// * The store never deletes `temp_dir` by itself (a `Drop` that did so would
///   be a hazard for callers that point it at a shared directory). The owner
///   creates the directory and removes it; [`MmapTileManager::clear`] removes
///   just the tile files this manager wrote.
///
/// [`commit_tile`]: TileProvider::commit_tile
pub struct MmapTileManager {
    /// Directory the tile files are written into.
    pub temp_dir: PathBuf,
    /// Tile file of every committed `(frame index, rectangle)`. Only ever looked
    /// up by key or drained by [`MmapTileManager::clear`]; its iteration order
    /// never reaches an output.
    pub tiles: HashMap<(usize, TileCoordinate), PathBuf>,
}

impl MmapTileManager {
    /// A store that writes its tile files into `temp_dir` (which must exist).
    #[must_use]
    pub fn new(temp_dir: PathBuf) -> Self {
        Self {
            temp_dir,
            tiles: HashMap::new(),
        }
    }

    /// Number of tiles currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// `true` when no tile is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Delete every tile file this manager wrote and forget the tiles. The
    /// directory itself is left alone. Files that are already gone are fine.
    ///
    /// # Errors
    /// Returns the first I/O error other than "not found"; the remaining files
    /// are still attempted.
    pub fn clear(&mut self) -> Result<(), StackerError> {
        let mut first_error = None;
        for (_, path) in self.tiles.drain() {
            if let Err(e) = std::fs::remove_file(&path)
                && e.kind() != std::io::ErrorKind::NotFound
                && first_error.is_none()
            {
                first_error = Some(e);
            }
        }
        first_error.map_or(Ok(()), |e| Err(StackerError::Io(e)))
    }
}

impl TileProvider for MmapTileManager {
    fn fetch_tile(
        &self,
        img_idx: usize,
        coord: &TileCoordinate,
    ) -> Result<PlanarImage<f32>, StackerError> {
        let path = self
            .tiles
            .get(&(img_idx, coord.clone()))
            .ok_or(StackerError::TileNotFound(coord.start_x, coord.start_y))?
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
        let expected = coord.width * coord.height;
        for (name, plane) in [
            ("luma", tile.luma.len()),
            ("chroma_a", tile.chroma_a.len()),
            ("chroma_b", tile.chroma_b.len()),
        ] {
            if plane != expected {
                return Err(StackerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "tile {}x{} at ({}, {}) of frame {img_idx}: {name} plane has {plane} \
                         samples, expected {expected}",
                        coord.width, coord.height, coord.start_x, coord.start_y
                    ),
                )));
            }
        }

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
