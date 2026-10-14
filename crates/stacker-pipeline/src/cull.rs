//! Auto-cull: the focus census and the decision built on it.
//!
//! The decision is the one `stacker_algo::optimize::optimize_stack` makes for a
//! stack that fits in memory (the desktop app):
//!
//! 1. per pixel, the frame with the highest focus measure (the plane-fit
//!    Laplacian, radius 1) wins; only pixels whose winning value exceeds 1 % of
//!    the global maximum (and `1e-6`) count as detail pixels;
//! 2. a frame survives when it wins at least a [`CullRule`] percent of the detail
//!    pixels: by default `min(2, 50 / N)` ([`stacker_algo::optimize::cull_threshold`]),
//!    or with `auto_cull_threshold_pct > 0` that fixed percentage;
//! 3. the survivors are ordered by a nearest-neighbour walk over the centroids
//!    of their winning pixels, starting from the leftmost centroid (the
//!    "census order"). A caller that does not want that order keeps the
//!    survivors in stack order ([`CullPlan::fusion_order`]; `sort_by_sharpness`
//!    asks for the census order).
//!
//! The CLI cannot hold every aligned frame, so it builds the same [`Census`]
//! tile by tile from its tile store ([`census_from_store`]) and applies
//! [`plan`]; the desktop app calls [`plan_frames`], which runs `optimize_stack`.
//! A test pins that both give the same order. The focus-measure window is 3×3, far inside
//! every apron of [`crate::tiling`], so per-tile focus values equal the
//! whole-image values (up to the rounding of the focus measure's summed-area
//! tables).

use rayon::prelude::*;
use stacker_algo::{
    optimize::{MAX_CULL_THRESHOLD_PCT, cull_threshold, optimize_stack},
    relief::focus::compute_focus_map,
};
use stacker_core::{
    error::StackerError,
    image::PlanarImage,
    memory::{MmapTileManager, TileProvider},
    settings::StackingSettings,
};

use crate::tiling::TilePlan;

/// How the cull threshold (percent of the detail pixels a frame must win) is
/// decided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CullRule {
    /// The value is a ceiling: the threshold is `min(value, 50 / N)` for a stack of
    /// `N` frames, so a deep stack keeps its frames (the default rule, with the
    /// ceiling [`MAX_CULL_THRESHOLD_PCT`]).
    Scaled(f32),
    /// The value is the threshold itself, whatever the frame count
    /// (`auto_cull_threshold_pct > 0`). `Fixed(0.0)` keeps every frame.
    Fixed(f32),
}

impl CullRule {
    /// The rule `settings` ask for: `auto_cull_threshold_pct` when it is above 0,
    /// else the frame-count-scaled rule with the ceiling 2 %.
    #[must_use]
    pub fn from_settings(settings: &StackingSettings) -> Self {
        if settings.auto_cull_threshold_pct > 0.0 {
            Self::Fixed(settings.auto_cull_threshold_pct)
        } else {
            Self::Scaled(MAX_CULL_THRESHOLD_PCT)
        }
    }

    /// The threshold in percent that applies to a stack of `n_frames` frames.
    #[must_use]
    pub const fn effective_pct(self, n_frames: usize) -> f32 {
        match self {
            Self::Scaled(ceiling) => ceiling.min(cull_threshold(n_frames)),
            Self::Fixed(pct) => pct,
        }
    }
}

/// Fraction of the global maximum below which a pixel is background
/// (mirrors `CONTRAST_FLOOR_FRACTION` in `stacker_algo::optimize`).
const CONTRAST_FLOOR_FRACTION: f32 = 0.01;
/// Absolute lower bound of the floor (mirrors `CONTRAST_FLOOR_ABSOLUTE`).
const CONTRAST_FLOOR_ABSOLUTE: f32 = 1e-6;

#[expect(
    clippy::cast_precision_loss,
    reason = "pixel counts and coordinates are far below 2^52"
)]
const fn to_f64(n: usize) -> f64 {
    n as f64
}

#[expect(
    clippy::cast_precision_loss,
    reason = "pixel counts are only used for a coarse percentage"
)]
const fn to_f32(n: usize) -> f32 {
    n as f32
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is a non-negative count produced by multiplying a count by a fraction"
)]
const fn floor_to_usize(x: f32) -> usize {
    x as usize
}

/// Per-frame outcome of the census.
#[derive(Debug, Clone, PartialEq)]
pub struct Census {
    /// Detail pixels each frame wins.
    pub win_counts: Vec<usize>,
    /// Sum of the `(x, y)` of the pixels each frame wins.
    pub centroids: Vec<(f64, f64)>,
    /// Pixels whose winning focus value is above the contrast floor.
    pub detail_pixels: usize,
}

/// The best focus value and its frame at every pixel of the image, filled tile
/// by tile.
#[derive(Debug, Clone)]
pub struct WinnerMap {
    width: usize,
    best_sml: Vec<f32>,
    best_idx: Vec<u32>,
}

impl WinnerMap {
    /// An empty map (`-1.0` marks a pixel nobody scored).
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            best_sml: vec![-1.0; width * height],
            best_idx: vec![0; width * height],
        }
    }

    /// Offer the single-plane focus map `sml` of the padded tile of `plan`
    /// (`plan.padded.width` samples per row) for frame `frame_id`. Only the
    /// interior of the tile is read; a frame replaces the current winner only
    /// with a strictly higher value, so ties go to the lower frame index when
    /// frames are offered in order.
    pub fn offer(&mut self, plan: &TilePlan, sml: &[f32], frame_id: u32) {
        let (ix, iy) = plan.interior_offset;
        let interior = &plan.interior;
        for row in 0..interior.height {
            let src = (iy + row) * plan.padded.width + ix;
            let dst = (interior.start_y + row) * self.width + interior.start_x;
            for col in 0..interior.width {
                let v = sml[src + col];
                if v > self.best_sml[dst + col] {
                    self.best_sml[dst + col] = v;
                    self.best_idx[dst + col] = frame_id;
                }
            }
        }
    }

    /// Offer the single-plane focus map `sml` of the **whole image** for frame
    /// `frame_id`, with the tie rule of [`Self::offer`].
    ///
    /// # Panics
    /// Panics when `sml` does not hold one value per pixel of the map.
    pub fn offer_whole(&mut self, sml: &[f32], frame_id: u32) {
        assert_eq!(sml.len(), self.best_sml.len(), "one value per pixel");
        for ((best, idx), &v) in self.best_sml.iter_mut().zip(&mut self.best_idx).zip(sml) {
            if v > *best {
                *best = v;
                *idx = frame_id;
            }
        }
    }

    /// Count the wins per frame.
    #[must_use]
    pub fn census(&self, n_frames: usize) -> Census {
        census_from_winners(&self.best_sml, &self.best_idx, self.width, n_frames)
    }
}

/// Count, per frame, the detail pixels it wins, from per-pixel winners.
///
/// `best_sml[i]` / `best_idx[i]` are the winning focus value and frame of pixel
/// `i` (row-major, `width` wide); `-1.0` marks a pixel nobody scored.
#[must_use]
pub fn census_from_winners(
    best_sml: &[f32],
    best_idx: &[u32],
    width: usize,
    n_frames: usize,
) -> Census {
    let global_max = best_sml.iter().copied().fold(0.0_f32, f32::max);
    let floor = (global_max * CONTRAST_FLOOR_FRACTION).max(CONTRAST_FLOOR_ABSOLUTE);

    let mut win_counts = vec![0_usize; n_frames];
    let mut centroids = vec![(0.0_f64, 0.0_f64); n_frames];
    let mut detail_pixels = 0_usize;
    for (i, (&sml, &idx)) in best_sml.iter().zip(best_idx).enumerate() {
        if sml > floor {
            let f = idx as usize;
            win_counts[f] += 1;
            detail_pixels += 1;
            centroids[f].0 += to_f64(i % width);
            centroids[f].1 += to_f64(i / width);
        }
    }
    Census {
        win_counts,
        centroids,
        detail_pixels,
    }
}

/// Which frames survive a threshold, and in which order they are fused.
///
/// Returns the kept frame indices in nearest-neighbour order. When no frame
/// reaches the threshold the returned order is empty and the caller decides
/// what to do ([`plan`] keeps every frame).
#[must_use]
#[expect(
    clippy::suboptimal_flops,
    reason = "the squared distance is written exactly as in optimize_stack, so both pick the same order"
)]
pub fn order_from_census(
    census: &Census,
    width: usize,
    height: usize,
    min_contribution_pct: f32,
) -> Vec<usize> {
    let min_detail = floor_to_usize(to_f32(census.detail_pixels) * (min_contribution_pct / 100.0));

    let mut kept = Vec::new();
    let mut kept_centroids = Vec::new();
    for (idx, &wins) in census.win_counts.iter().enumerate() {
        if wins >= min_detail {
            kept.push(idx);
            let count = to_f64(wins);
            kept_centroids.push(if count > 0.0 {
                (
                    census.centroids[idx].0 / count,
                    census.centroids[idx].1 / count,
                )
            } else {
                (to_f64(width) * 0.5, to_f64(height) * 0.5)
            });
        }
    }

    let mut order = Vec::with_capacity(kept.len());
    if kept.is_empty() {
        return order;
    }
    let mut unvisited: Vec<usize> = (0..kept.len()).collect();
    let mut start = 0;
    let mut min_x = f64::MAX;
    for (i, c) in kept_centroids.iter().enumerate() {
        if c.0 < min_x {
            min_x = c.0;
            start = i;
        }
    }
    let first = unvisited.iter().position(|&x| x == start).unwrap_or(0);
    let mut current = unvisited.remove(first);
    order.push(kept[current]);
    while !unvisited.is_empty() {
        let mut nearest = 0;
        let mut min_dist = f64::MAX;
        for (i, &cand) in unvisited.iter().enumerate() {
            let dx = kept_centroids[cand].0 - kept_centroids[current].0;
            let dy = kept_centroids[cand].1 - kept_centroids[current].1;
            let dist = dx * dx + dy * dy;
            if dist < min_dist {
                min_dist = dist;
                nearest = i;
            }
        }
        current = unvisited.remove(nearest);
        order.push(kept[current]);
    }
    order
}

/// The decision of the auto-cull.
#[derive(Debug, Clone, PartialEq)]
pub struct CullPlan {
    /// The kept frames in fusion order; never empty for a non-empty stack.
    pub order: Vec<usize>,
    /// The frames that were dropped, ascending.
    pub dropped: Vec<usize>,
    /// The threshold actually applied, in percent of the detail pixels.
    pub effective_pct: f32,
    /// Notes for the user: no detail at all, or a cull that would have emptied
    /// the stack.
    pub warnings: Vec<String>,
}

impl CullPlan {
    /// The kept frames in the order they are fused: the census (nearest-neighbour)
    /// order with `sort_by_sharpness`, else stack order (ascending positions).
    #[must_use]
    pub fn fusion_order(&self, sort_by_sharpness: bool) -> Vec<usize> {
        let mut order = self.order.clone();
        if !sort_by_sharpness {
            order.sort_unstable();
        }
        order
    }
}

fn dropped_of(order: &[usize], n_frames: usize) -> Vec<usize> {
    let mut kept = vec![false; n_frames];
    for &i in order {
        if let Some(slot) = kept.get_mut(i) {
            *slot = true;
        }
    }
    (0..n_frames).filter(|&i| !kept[i]).collect()
}

/// Decide from a census.
///
/// `requested_pct` is the caller's threshold (the apps pass `2.0`); it is capped
/// by `cull_threshold(N)` so deep stacks keep their frames. When the cull would
/// drop every frame, or there is no detail at all, all frames are kept (still in
/// nearest-neighbour order) and a warning says so.
#[must_use]
pub fn plan(census: &Census, width: usize, height: usize, requested_pct: f32) -> CullPlan {
    plan_with(census, width, height, CullRule::Scaled(requested_pct))
}

/// [`plan`] with an explicit [`CullRule`]: the scaled rule or a fixed percentage.
#[must_use]
pub fn plan_with(census: &Census, width: usize, height: usize, rule: CullRule) -> CullPlan {
    let n_frames = census.win_counts.len();
    let effective_pct = rule.effective_pct(n_frames);
    let mut warnings = Vec::new();
    if census.detail_pixels == 0 && n_frames > 0 {
        warnings.push(format!(
            "No frame shows any in-focus detail; keeping all {n_frames} frames."
        ));
    }
    let mut order = order_from_census(census, width, height, effective_pct);
    if order.is_empty() && n_frames > 0 {
        warnings.push(format!(
            "Auto-cull would have dropped every frame; keeping all {n_frames} frames."
        ));
        order = order_from_census(census, width, height, 0.0);
    }
    CullPlan {
        dropped: dropped_of(&order, n_frames),
        order,
        effective_pct,
        warnings,
    }
}

/// The decision for a stack held in memory: `optimize_stack`, mapped to a
/// [`CullPlan`].
#[must_use]
pub fn plan_frames(images: &[PlanarImage<f32>], requested_pct: f32) -> CullPlan {
    let result = optimize_stack(images, requested_pct);
    CullPlan {
        dropped: dropped_of(&result.recommended_order, images.len()),
        order: result.recommended_order,
        effective_pct: requested_pct.min(cull_threshold(images.len())),
        warnings: result.warnings,
    }
}

/// The census of a stack held in memory (the same winners as `optimize_stack`
/// finds: the plane-fit focus measure of radius 1, the first frame wins a tie).
#[must_use]
pub fn census_from_frames(images: &[PlanarImage<f32>]) -> Census {
    let Some(first) = images.first() else {
        return census_from_winners(&[], &[], 1, 0);
    };
    let maps: Vec<Vec<f32>> = images
        .par_iter()
        .map(|image| compute_focus_map(image, 1))
        .collect();
    let mut winners = WinnerMap::new(first.width, first.height);
    for (frame, sml) in maps.iter().enumerate() {
        winners.offer_whole(sml, u32::try_from(frame).unwrap_or(u32::MAX));
    }
    winners.census(images.len())
}

/// [`plan_frames`] with an explicit [`CullRule`].
///
/// The scaled rule is exactly [`plan_frames`] (`optimize_stack`); a fixed
/// percentage is decided on the same census without the frame-count cap.
#[must_use]
pub fn plan_frames_with(images: &[PlanarImage<f32>], rule: CullRule) -> CullPlan {
    match rule {
        CullRule::Scaled(ceiling) => plan_frames(images, ceiling),
        CullRule::Fixed(_) => {
            let (width, height) = images.first().map_or((0, 0), |f| (f.width, f.height));
            plan_with(&census_from_frames(images), width, height, rule)
        }
    }
}

/// Build the census of a stack that lives in a tile store: for every tile and
/// frame, fetch the padded tile, compute its focus map and offer the interior.
///
/// `tiles` describes the grid the store was filled with (frames `0..n_frames`).
///
/// # Errors
/// Returns the tile-store error when a tile cannot be read, and a
/// [`StackerError::MathError`] for more than `u32::MAX` frames.
pub fn census_from_store(
    store: &MmapTileManager,
    tiles: &[TilePlan],
    n_frames: usize,
    img_w: usize,
    img_h: usize,
) -> Result<Census, StackerError> {
    let mut winners = WinnerMap::new(img_w, img_h);
    for cell in tiles {
        for frame_idx in 0..n_frames {
            let frame_id = u32::try_from(frame_idx).map_err(|_| {
                StackerError::MathError("more than u32::MAX frames in a stack".to_owned())
            })?;
            let tile = store.fetch_tile(frame_idx, &cell.padded)?;
            let sml = compute_focus_map(&tile, 1);
            winners.offer(cell, &sml, frame_id);
        }
    }
    Ok(winners.census(n_frames))
}
