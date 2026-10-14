//! The tile grid of an out-of-core run, with the apron derived from what is
//! being computed.
//!
//! A tile is stored and fused as its interior grown by an **apron** on every
//! side (clamped to the image). The interior of the fused padded tile equals the
//! whole-image result only when the apron covers everything an output pixel
//! depends on:
//!
//! | Method | Apron | Why |
//! |---|---|---|
//! | `Apex` | `apron_for_levels(pyramid_levels)` | reduce and expand chains of the Laplacian pyramid, plus the 3×3 grit window |
//! | `Relief` | `relief_estimation_radius + 2·relief_smoothing_radius + 1` | the focus window, then a guided filter (two box filters in sequence) |
//! | `Strata` | [`STRATA_APRON_PX`] (101) | saliency blur reach plus the two box filters of the wide guided filter, at the largest base radius |
//! | Neural | [`APRON_PX`] | receptive field of the network |
//!
//! `Relief` with `relief_use_multigrid` keeps the `Relief` apron (it is larger than
//! the focus window alone, which is all a tile computes): the depth is solved on
//! whole-image planes outside the tiles, see [`MULTIGRID_FINISH_BYTES_PER_PIXEL`].
//!
//! For `Apex` the tile size is additionally rounded up to a multiple of
//! `2^(L−1)` so that every padded tile starts on the decimation grid of the
//! whole-image pyramid.

use stacker_algo::{
    apex::fuse::{apex_fold_bytes_per_pixel, apron_for_levels, pyramid_grid},
    strata::{STRATA_APRON_PX, strata_bytes_per_pixel},
};
use stacker_core::{
    error::StackerError,
    image::PlanarImage,
    memory::{
        APRON_PX, MmapTileManager, TileCoordinate, TileProvider, enumerate_tiles, extract_tile,
    },
    settings::StackingSettings,
};

/// Bytes one pixel of one frame takes in the tile store (three `f32` planes).
pub const SCRATCH_BYTES_PER_PIXEL: usize = 12;

/// The deepest pyramid `pyramid_levels` allows (`2..=12`).
pub const MAX_PYRAMID_LEVELS: usize = 12;

/// Bytes per padded pixel the guided `Relief` keeps while it folds the focus maps
/// of a tile: the best focus (`f32`), the winning frame (`u16`) and the focus map
/// of the frame in flight (`f32`).
pub const RELIEF_VOTE_BYTES_PER_PIXEL: usize = 4 + 2 + 4;

/// Whole-image bytes per pixel the multigrid `Relief` keeps through both phases
/// of a tiled run: the best focus value (`f32`) and the winning frame (`u16`).
pub const MULTIGRID_VOTE_BYTES_PER_PIXEL: usize = 6;

/// Whole-image bytes per pixel at the peak of the multigrid solve.
///
/// They are the solver's
/// input planes, its four padded planes and the coarser levels; see
/// `stacker_algo::relief::multigrid`). Frames are not in memory then.
pub const MULTIGRID_FINISH_BYTES_PER_PIXEL: usize = 28;

/// Whole-image bytes per pixel of the depth map, kept while the tiles are
/// composited.
pub const MULTIGRID_DEPTH_BYTES_PER_PIXEL: usize = 4;

/// The whole-image bytes a multigrid `Relief` run needs at its peak on top of
/// the per-tile working set, for an image of `pixels` pixels.
///
/// That is the solve
/// ([`MULTIGRID_FINISH_BYTES_PER_PIXEL`]) while the best-focus map (4 bytes) is
/// still held for the threshold. The votes phase needs less (the votes plus the
/// threshold's sorted copy, 10 bytes).
#[must_use]
pub const fn multigrid_whole_image_bytes(pixels: usize) -> usize {
    (MULTIGRID_FINISH_BYTES_PER_PIXEL + 4) * pixels
}

/// One cell of the tile grid: the interior that is written to the output, the
/// padded region (interior plus apron) that is stored and fused, and the offset
/// of the interior inside the padded tile.
///
/// The tile store is keyed by [`Self::padded`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    /// The rectangle of the output this tile is responsible for.
    pub interior: TileCoordinate,
    /// The interior grown by the apron and clamped to the image; what the tile
    /// store holds.
    pub padded: TileCoordinate,
    /// `(x, y)` of the interior's top-left corner inside the padded tile.
    pub interior_offset: (usize, usize),
}

/// Cut an `img_w × img_h` image into tiles of at most `tile_size` pixels, each
/// with `apron` pixels of context on every side.
#[must_use]
pub fn plan_tiles(img_w: usize, img_h: usize, tile_size: usize, apron: usize) -> Vec<TilePlan> {
    enumerate_tiles(img_w, img_h, tile_size)
        .into_iter()
        .map(|interior| {
            let (px, py, pw, ph) = interior.padded_region_with(img_w, img_h, apron);
            TilePlan {
                interior_offset: interior.interior_offset_in_padded_with(img_w, img_h, apron),
                padded: TileCoordinate {
                    start_x: px,
                    start_y: py,
                    width: pw,
                    height: ph,
                },
                interior,
            }
        })
        .collect()
}

/// Write one frame's padded tiles into the store.
///
/// # Errors
/// Returns the store's error when a tile cannot be written.
pub fn commit_frame(
    store: &mut MmapTileManager,
    frame_idx: usize,
    frame: &PlanarImage<f32>,
    tiles: &[TilePlan],
) -> Result<(), StackerError> {
    for plan in tiles {
        let p = &plan.padded;
        let padded = extract_tile(frame, p.start_x, p.start_y, p.width, p.height);
        store.commit_tile(frame_idx, p, padded)?;
    }
    Ok(())
}

/// What is computed per tile; it decides the apron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Laplacian-pyramid fusion.
    Apex,
    /// Depth-map fusion (the guided filter, or the multigrid solver).
    Relief,
    /// Guided-filter soft blend.
    Strata,
    /// The trained network.
    Neural,
}

/// The apron `method` needs under `settings`, in pixels.
///
/// The focus census of the auto-cull reads a radius-1 window, which every apron
/// returned here covers.
#[must_use]
pub fn apron_for(method: Method, settings: &StackingSettings) -> usize {
    match method {
        Method::Apex => apron_for_levels(settings.pyramid_levels as usize),
        Method::Relief => (settings.relief_estimation_radius as usize
            + 2 * settings.relief_smoothing_radius as usize
            + 1)
        .max(2),
        Method::Strata => STRATA_APRON_PX,
        Method::Neural => APRON_PX,
    }
}

/// Tile size and apron of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    /// Interior size of a full tile.
    pub tile_size: usize,
    /// Context on every side of the interior.
    pub apron: usize,
}

impl Geometry {
    /// The geometry for a requested tile size: the apron from [`apron_for`],
    /// and for `Apex` the tile size rounded up to a multiple of the pyramid's
    /// coarsest spacing (see the module documentation).
    #[must_use]
    pub fn choose(requested_tile: usize, method: Method, settings: &StackingSettings) -> Self {
        let apron = apron_for(method, settings);
        let tile_size = match method {
            Method::Apex => {
                let grid = pyramid_grid(settings.pyramid_levels as usize);
                requested_tile.max(1).div_ceil(grid) * grid
            }
            Method::Relief | Method::Strata | Method::Neural => requested_tile.max(1),
        };
        Self { tile_size, apron }
    }

    /// Edge length of a full padded tile (an interior tile away from the image
    /// border).
    #[must_use]
    pub const fn padded_edge(self) -> usize {
        self.tile_size + 2 * self.apron
    }

    /// Pixels of a full padded tile.
    #[must_use]
    pub const fn padded_pixels(self) -> usize {
        self.padded_edge() * self.padded_edge()
    }

    /// Stored pixels per source pixel: `((T + 2A) / T)²`, the factor by which
    /// the tile store (and the fusion's working set) exceeds the image itself.
    #[must_use]
    pub fn padding_factor(self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "tile edges are far below 2^52")]
        let ratio = self.padded_edge() as f64 / self.tile_size as f64;
        ratio * ratio
    }

    /// Bytes of scratch disk per source pixel per frame (an upper bound: tiles
    /// at the image border are smaller).
    #[must_use]
    pub fn scratch_bytes_per_source_pixel(self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "a small constant")]
        let bytes = SCRATCH_BYTES_PER_PIXEL as f64;
        bytes * self.padding_factor()
    }

    /// Bytes of RAM one frame's padded tile occupies during the fusion of a
    /// tile: the tile itself (12 per padded pixel) for `Apex`, `Relief` and the
    /// neural mode. `Apex` folds the frames into one accumulator pyramid and
    /// `Relief` folds each focus map into the best value and the winner as it is
    /// computed, so neither keeps anything per frame beyond the tile (their
    /// working sets are [`Self::fusion_fixed_bytes`]). `Strata` reads its tile
    /// frames on demand, two at a time (see [`Self::fusion_fixed_bytes`]), so it
    /// holds nothing per frame.
    #[must_use]
    pub const fn fusion_bytes_per_frame(self, method: Method) -> usize {
        match method {
            Method::Apex | Method::Relief | Method::Neural => {
                SCRATCH_BYTES_PER_PIXEL * self.padded_pixels()
            }
            Method::Strata => 0,
        }
    }

    /// Bytes of RAM the fusion of one tile needs besides the per-frame figure of
    /// [`Self::fusion_bytes_per_frame`], whatever the number of frames:
    ///
    /// * `Apex`: the fold's accumulator pyramid, `best_energy` planes, grit scratch,
    ///   the incoming frame's pyramid and the build transients
    ///   (`apex_fold_bytes_per_pixel`, taken at the 12 levels the settings allow at
    ///   most, so it is an upper bound for any `pyramid_levels`: 96, against 84 at
    ///   the default 8 levels);
    /// * `Relief`: the best-focus plane (4), the winner index (2) and the focus map
    ///   in flight (4): 10;
    /// * `Strata`: `strata_bytes_per_pixel()` (150: the accumulators, winner map,
    ///   prepared planes, weights and filter scratch) plus the two tile frames it
    ///   holds at a time (24);
    /// * none for the neural mode.
    ///
    /// All per padded pixel.
    #[must_use]
    pub const fn fusion_fixed_bytes(self, method: Method) -> usize {
        let per_pixel = match method {
            Method::Apex => apex_fold_bytes_per_pixel(MAX_PYRAMID_LEVELS),
            Method::Relief => RELIEF_VOTE_BYTES_PER_PIXEL,
            Method::Strata => strata_bytes_per_pixel() + 2 * SCRATCH_BYTES_PER_PIXEL,
            Method::Neural => 0,
        };
        per_pixel * self.padded_pixels()
    }

    /// Bytes of RAM the read-ahead of the next tile holds on top of
    /// [`Self::fusion_bytes`]: the padded tile of every frame again (12 per frame
    /// and padded pixel). `Strata` reads on demand and never reads ahead.
    #[must_use]
    pub const fn prefetch_bytes(self, method: Method, frames: usize) -> usize {
        match method {
            Method::Apex | Method::Relief | Method::Neural => {
                SCRATCH_BYTES_PER_PIXEL * self.padded_pixels() * frames
            }
            Method::Strata => 0,
        }
    }

    /// RAM of the fusion of one tile of `frames` frames: the per-frame figure
    /// times `frames`, plus the fixed working set.
    #[must_use]
    pub const fn fusion_bytes(self, method: Method, frames: usize) -> usize {
        self.fusion_bytes_per_frame(method) * frames + self.fusion_fixed_bytes(method)
    }
}
