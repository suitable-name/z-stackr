//! The auto-cull: rank the frames of a stack by the in-focus detail they win,
//! drop the ones that add nothing, and order the survivors for fusion.
//!
//! [`optimize_stack`] runs the whole decision on frames held in memory;
//! [`cull_threshold`] is the threshold rule that scales with the frame count.

#![expect(
    clippy::cast_precision_loss,
    reason = "frame, pixel and win counts are exact in f64; in f32 they are exact up to 2^24 and only feed percentages"
)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the minimum detail pixel count is a non-negative fraction of the pixel count"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "the centroid distance is evaluated as plain multiply-add"
)]

use crate::relief::focus::compute_focus_map;
use rayon::prelude::*;
use stacker_core::image::PlanarImage;

/// Outcome of [`optimize_stack`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OptimizationResult {
    /// Indices (into the input slice) of the frames that survive the cull,
    /// in ascending order.  Never empty for a non-empty input.
    pub kept_indices: Vec<usize>,
    /// The kept frames in the recommended nearest-neighbour fusion order.
    pub recommended_order: Vec<usize>,
    /// Human-readable notes the caller can surface (a toast, a log line):
    /// for example that the cull would have dropped every frame and the full
    /// list was kept instead.  Empty when nothing unusual happened.
    pub warnings: Vec<String>,
}

/// Ceiling of the auto-cull threshold, in percent of the detail pixels.
pub const MAX_CULL_THRESHOLD_PCT: f32 = 2.0;

/// The auto-cull threshold for a stack of `frame_count` frames, in percent of
/// the detail pixels a frame must win to survive.
///
/// Formula: `min(2.0, 50.0 / frame_count)`.
///
/// With `N` frames that share the detail evenly each one wins `100 / N` %, so
/// `50 / N` is half of the average share: a frame is dropped only when it
/// contributes less than half of what an average frame does.  For stacks of up
/// to 25 frames this is the familiar absolute 2 %; beyond that it shrinks, so
/// a deep stack (a flat 2 % would empty any evenly spread stack of 51 or more
/// frames) keeps its frames.  A `frame_count` of 0 returns the 2 % ceiling.
#[must_use]
pub const fn cull_threshold(frame_count: usize) -> f32 {
    if frame_count == 0 {
        return MAX_CULL_THRESHOLD_PCT;
    }
    let scaled = 50.0 / frame_count as f32;
    if scaled < MAX_CULL_THRESHOLD_PCT {
        scaled
    } else {
        MAX_CULL_THRESHOLD_PCT
    }
}

/// Contrast floor as a fraction of the global maximum focus-measure value
/// (the plane-fit residual variance of [`compute_focus_map`]) across all
/// frames and all pixels.  Only pixels whose best-across-frames score exceeds
/// this floor are considered "detail pixels" and count toward a frame's win
/// total.  Flat / blown-out background pixels (score ≈ 0 everywhere) are
/// excluded from both numerator *and* denominator so the contribution
/// percentage is over meaningful content only.
///
/// 1 % of global max is deliberately conservative: a pixel must have at least
/// 1 % of the sharpest-measured contrast in the whole scene to be counted.
/// This eliminates smooth gradients and out-of-focus backgrounds while
/// retaining edge and texture pixels.
const CONTRAST_FLOOR_FRACTION: f32 = 0.01;

/// Minimum absolute focus-measure value applied as a secondary guard against
/// near-zero global-max images (e.g., a stack of completely white frames).
const CONTRAST_FLOOR_ABSOLUTE: f32 = 1e-6;

/// Rank the frames by the share of in-focus detail they win, drop the ones
/// that contribute (almost) nothing, and order the survivors for fusion.
///
/// `min_contribution_pct` is the caller's cull threshold in percent of the
/// detail pixels; it is capped by [`cull_threshold`] of the frame count, so the
/// effective threshold never exceeds half of the average per-frame share.  The
/// GUI passes `2.0`, which therefore means "2 % for small stacks, scaled down
/// for deep ones".
///
/// The result never has an empty `kept_indices` for a non-empty input: if the
/// cull would drop every frame (or there is no detail at all) the full list is
/// kept and a note is added to [`OptimizationResult::warnings`].
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the stack: focus maps, win census, cull and ordering share their intermediate vectors"
)]
pub fn optimize_stack(
    images: &[PlanarImage<f32>],
    min_contribution_pct: f32,
) -> OptimizationResult {
    if images.is_empty() {
        return OptimizationResult::default();
    }

    let width = images[0].width;
    let height = images[0].height;
    let num_pixels = width * height;

    // Focus maps are independent per frame — build them in parallel across the
    // stack (each `compute_focus_map` is itself internally parallel, but
    // mapping across frames keeps all cores busy on small images). One plane
    // per frame: the measure is a scalar map.
    let laplacians: Vec<Vec<f32>> = images
        .par_iter()
        .map(|img| compute_focus_map(img, 1))
        .collect();

    // ── Step 1: find the global maximum focus score across ALL frames and pixels ──
    // This is used to derive a scene-adaptive contrast floor so that flat /
    // blown-out backgrounds (score ≈ 0) are excluded from win counting.
    let global_max_sml: f32 = laplacians
        .iter()
        .flat_map(|lap| lap.iter().copied())
        .fold(0.0_f32, f32::max);

    // The floor is the larger of the fractional threshold and the absolute
    // minimum, so we are robust to near-zero global-max scenes.
    let contrast_floor = (global_max_sml * CONTRAST_FLOOR_FRACTION).max(CONTRAST_FLOOR_ABSOLUTE);

    tracing::debug!(
        global_max_sml,
        contrast_floor,
        CONTRAST_FLOOR_FRACTION,
        "contrast floor computed"
    );

    // ── Step 2: per-pixel winner selection (detail pixels only) ──────────
    let mut win_counts = vec![0usize; images.len()];
    let mut centroids = vec![(0.0_f64, 0.0_f64); images.len()];
    let mut detail_pixel_count = 0usize;

    for i in 0..num_pixels {
        // Find the best focus score across frames at this pixel.  Strict `>`:
        // on a tie the first frame wins (the tie rule shared with Relief and
        // Apex, see `fuse_pyramids`).
        let mut best_sml = -1.0_f32;
        let mut best_idx = 0_usize;

        for (idx, lap) in laplacians.iter().enumerate() {
            if lap[i] > best_sml {
                best_sml = lap[i];
                best_idx = idx;
            }
        }

        // Only count pixels that have genuine in-focus detail.
        // Pixels below the contrast floor are background / flat and are
        // excluded from both the win count AND the percentage denominator.
        if best_sml > contrast_floor {
            win_counts[best_idx] += 1;
            detail_pixel_count += 1;

            let x = (i % width) as f64;
            let y = (i / width) as f64;
            centroids[best_idx].0 += x;
            centroids[best_idx].1 += y;
        }
    }

    tracing::info!(
        n_frames = images.len(),
        detail_pixels = detail_pixel_count,
        total_pixels = num_pixels,
        contrast_floor,
        "detail pixel census complete"
    );

    // Log per-frame win counts before culling.
    for (idx, &wc) in win_counts.iter().enumerate() {
        let pct = if detail_pixel_count > 0 {
            wc as f32 / detail_pixel_count as f32 * 100.0
        } else {
            0.0
        };
        tracing::info!(
            frame = idx,
            win_count = wc,
            contribution_pct = pct,
            "frame win count"
        );
    }

    // ── Step 3: cull frames below the contribution threshold ─────────────
    // The threshold is expressed as a percentage of DETAIL pixels (not total
    // pixels), so it is insensitive to background area.
    // The caller's value is a ceiling; the frame-count-scaled threshold wins
    // for deep stacks (see `cull_threshold`).
    let effective_pct = min_contribution_pct.min(cull_threshold(images.len()));
    let min_detail_pixels = (detail_pixel_count as f32 * (effective_pct / 100.0)) as usize;

    let mut warnings = Vec::new();
    let mut keep: Vec<bool> = win_counts
        .iter()
        .map(|&wc| wc >= min_detail_pixels)
        .collect();
    if detail_pixel_count == 0 {
        warnings.push(format!(
            "No frame shows any in-focus detail; keeping all {} frames.",
            images.len()
        ));
    }
    if !keep.iter().any(|&k| k) {
        warnings.push(format!(
            "Auto-cull would have dropped every frame; keeping all {} frames.",
            images.len()
        ));
        keep.fill(true);
    }

    let mut kept_indices = Vec::new();
    let mut valid_centroids = Vec::new();

    for (idx, _) in keep.iter().enumerate().filter(|&(_, &k)| k) {
        kept_indices.push(idx);
        let count = win_counts[idx] as f64;
        let (cx, cy) = if count > 0.0 {
            (centroids[idx].0 / count, centroids[idx].1 / count)
        } else {
            // Frame won zero detail pixels — place centroid at image centre.
            (width as f64 * 0.5, height as f64 * 0.5)
        };
        valid_centroids.push((cx, cy));
    }

    // Log which frames were kept / dropped.
    {
        let all: Vec<usize> = (0..images.len()).collect();
        let dropped: Vec<usize> = all
            .iter()
            .copied()
            .filter(|i| !kept_indices.contains(i))
            .collect();
        tracing::info!(
            kept   = ?kept_indices,
            dropped = ?dropped,
            min_contribution_pct,
            effective_pct,
            min_detail_pixels,
            "frame culling result"
        );
    }

    // ── Step 4: nearest-neighbour ordering of kept frames ────────────────
    let mut recommended_order = Vec::new();
    if !kept_indices.is_empty() {
        let mut unvisited: Vec<usize> = (0..kept_indices.len()).collect();

        // Start from the frame whose centroid is leftmost (smallest x).
        let mut best_start = 0;
        let mut min_x = f64::MAX;
        for (i, c) in valid_centroids.iter().enumerate() {
            if c.0 < min_x {
                min_x = c.0;
                best_start = i;
            }
        }

        let start_pos = unvisited.iter().position(|&x| x == best_start).unwrap_or(0);
        let mut current = unvisited.remove(start_pos);
        recommended_order.push(kept_indices[current]);

        while !unvisited.is_empty() {
            let mut nearest_idx = 0;
            let mut min_dist = f64::MAX;

            for (i, &cand) in unvisited.iter().enumerate() {
                let dx = valid_centroids[cand].0 - valid_centroids[current].0;
                let dy = valid_centroids[cand].1 - valid_centroids[current].1;
                let dist = dx * dx + dy * dy;
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = i;
                }
            }

            current = unvisited.remove(nearest_idx);
            recommended_order.push(kept_indices[current]);
        }
    }

    OptimizationResult {
        kept_indices,
        recommended_order,
        warnings,
    }
}
