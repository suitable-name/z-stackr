use crate::relief::focus::compute_sum_modified_laplacian;
use rayon::prelude::*;
use stacker_core::image::PlanarImage;

pub struct OptimizationResult {
    pub kept_indices: Vec<usize>,
    pub recommended_order: Vec<usize>,
}

/// Contrast floor as a fraction of the global maximum SML value across all
/// frames and all pixels.  Only pixels whose best-across-frames SML exceeds
/// this floor are considered "detail pixels" and count toward a frame's win
/// total.  Flat / blown-out background pixels (SML ≈ 0 everywhere) are
/// excluded from both numerator *and* denominator so the contribution
/// percentage is over meaningful content only.
///
/// 1 % of global max is deliberately conservative: a pixel must have at least
/// 1 % of the sharpest-measured contrast in the whole scene to be counted.
/// This eliminates smooth gradients and out-of-focus backgrounds while
/// retaining edge and texture pixels.
const CONTRAST_FLOOR_FRACTION: f32 = 0.01;

/// Minimum absolute SML value applied as a secondary guard against
/// near-zero global-max images (e.g., a stack of completely white frames).
const CONTRAST_FLOOR_ABSOLUTE: f32 = 1e-6;

pub fn optimize_stack(
    images: &[PlanarImage<f32>],
    min_contribution_pct: f32,
) -> OptimizationResult {
    if images.is_empty() {
        return OptimizationResult {
            kept_indices: vec![],
            recommended_order: vec![],
        };
    }

    let width = images[0].width;
    let height = images[0].height;
    let num_pixels = width * height;

    // Focus maps are independent per frame — build them in parallel across the
    // stack (each `compute_sum_modified_laplacian` is itself internally
    // parallel, but mapping across frames keeps all cores busy on small images).
    let laplacians: Vec<PlanarImage<f32>> = images
        .par_iter()
        .map(|img| compute_sum_modified_laplacian(img, 1))
        .collect();

    // ── Step 1: find the global maximum SML across ALL frames and pixels ──
    // This is used to derive a scene-adaptive contrast floor so that flat /
    // blown-out backgrounds (SML ≈ 0) are excluded from win counting.
    let global_max_sml: f32 = laplacians
        .iter()
        .flat_map(|lap| lap.luma.iter().copied())
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
        // Find the best SML across frames at this pixel.
        let mut best_sml = -1.0_f32;
        let mut best_idx = 0_usize;

        for (idx, lap) in laplacians.iter().enumerate() {
            if lap.luma[i] > best_sml {
                best_sml = lap.luma[i];
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
    let min_detail_pixels = (detail_pixel_count as f32 * (min_contribution_pct / 100.0)) as usize;

    let mut kept_indices = Vec::new();
    let mut valid_centroids = Vec::new();

    for idx in 0..images.len() {
        if win_counts[idx] >= min_detail_pixels {
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
            min_detail_pixels,
            "frame culling result"
        );
    }

    // ── Step 4: nearest-neighbour ordering of kept frames ────────────────
    let recommended_order = compute_recommended_order(&kept_indices, &valid_centroids);

    OptimizationResult {
        kept_indices,
        recommended_order,
    }
}

fn compute_recommended_order(kept_indices: &[usize], centroids: &[(f64, f64)]) -> Vec<usize> {
    let mut recommended_order = Vec::new();
    if !kept_indices.is_empty() {
        let mut unvisited: Vec<usize> = (0..kept_indices.len()).collect();

        // Start from the frame whose centroid is leftmost (smallest x).
        let mut best_start = 0;
        let mut min_x = f64::MAX;
        for (i, c) in centroids.iter().enumerate() {
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
                let dx = centroids[cand].0 - centroids[current].0;
                let dy = centroids[cand].1 - centroids[current].1;
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
    recommended_order
}
