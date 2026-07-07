use nalgebra::Matrix3;
use rayon::prelude::*;

// ── Common valid-area cropping ────────────────────────────────────────────────
//
// Warping a frame into the reference coordinate system zero-fills the pixels
// that map outside the source image (a black border whose thickness depends on
// the frame's shift / scale / rotation). Fusing the full frames therefore bleeds
// those black borders into the result — visible as dark "frame edges".
//
// The fix is to crop the output to the region covered by *every* frame: build a
// per-frame coverage mask, intersect them, and crop to the largest axis-aligned
// rectangle that is valid in all frames.

/// Per-pixel coverage mask for warping a frame by `matrix`.
///
/// `mask[y * width + x]` is `true` iff the inverse-mapped source coordinate for
/// output pixel `(x, y)` falls inside the source bounds `[0, width-1] ×
/// [0, height-1]` — i.e. that output pixel receives real data rather than a
/// zero-fill/clamped border.
///
/// A non-invertible `matrix` yields an all-`false` mask.
#[must_use]
pub fn coverage_mask(matrix: &Matrix3<f32>, width: usize, height: usize) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    let Some(m_inv) = matrix.try_inverse() else {
        return mask;
    };
    let max_x = width as f32 - 1.0;
    let max_y = height as f32 - 1.0;

    mask.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, cell) in row.iter_mut().enumerate() {
            let p = m_inv * nalgebra::Vector3::new(x as f32, y as f32, 1.0);
            let denom = p[2];
            if denom.abs() < 1.0e-8 {
                continue;
            }
            let sx = p[0] / denom;
            let sy = p[1] / denom;
            *cell = sx.is_finite()
                && sy.is_finite()
                && sx >= 0.0
                && sy >= 0.0
                && sx <= max_x
                && sy <= max_y;
        }
    });
    mask
}

/// Intersect (`AND`) `other` into `acc` in place.
///
/// Used to accumulate the common coverage of a whole stack: start from an
/// all-`true` mask (e.g. the reference frame) and fold each frame's
/// [`coverage_mask`] in.
///
/// # Panics
/// Panics if the two masks differ in length.
pub fn intersect_coverage(acc: &mut [bool], other: &[bool]) {
    assert_eq!(
        acc.len(),
        other.len(),
        "coverage masks must be the same size"
    );
    for (a, b) in acc.iter_mut().zip(other.iter()) {
        *a &= *b;
    }
}

/// Largest axis-aligned all-`true` rectangle in a boolean mask, returned as
/// `(x, y, w, h)`. Returns `None` if the mask is empty or fully `false`.
///
/// Runs in `O(width * height)` using the classic maximal-rectangle-in-histogram
/// algorithm (per-row bar heights + a monotonic stack), so it is cheap even at
/// full sensor resolution.
#[must_use]
pub fn largest_true_rectangle(
    mask: &[bool],
    width: usize,
    height: usize,
) -> Option<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 || mask.len() != width * height {
        return None;
    }

    let mut heights = vec![0usize; width];
    let mut best_area = 0usize;
    let mut best_rect: Option<(usize, usize, usize, usize)> = None;

    for y in 0..height {
        let row = &mask[y * width..(y + 1) * width];
        for (h, &valid) in heights.iter_mut().zip(row.iter()) {
            *h = if valid { *h + 1 } else { 0 };
        }

        // Largest rectangle in the `heights` histogram, tracking position.
        let mut stack: Vec<usize> = Vec::with_capacity(width + 1);
        let mut x = 0usize;
        while x <= width {
            let cur = if x == width { 0 } else { heights[x] };
            match stack.last().copied() {
                // Pop bars taller than the current one and score their rectangles.
                Some(top) if cur < heights[top] => {
                    stack.pop();
                    let bar_h = heights[top];
                    let left = stack.last().map_or(0, |&l| l + 1);
                    let rect_w = x - left;
                    let area = rect_w * bar_h;
                    if bar_h > 0 && area > best_area {
                        best_area = area;
                        best_rect = Some((left, y + 1 - bar_h, rect_w, bar_h));
                    }
                }
                _ => {
                    stack.push(x);
                    x += 1;
                }
            }
        }
    }

    best_rect
}

/// Resolve the common-coverage crop rectangle for a stack, applying the
/// guard rails that make [`largest_true_rectangle`]'s raw result safe to use
/// as an automatic crop:
///
/// - Returns `None` when the mask has no all-`true` rectangle at all (mirrors
///   [`largest_true_rectangle`]).
/// - Returns `None` when the rectangle covers the *entire* canvas — there is
///   nothing to crop, so callers should treat this the same as "no crop".
/// - Returns `None` when the rectangle covers **less than 25 %** of the
///   canvas area. This is a rogue-frame guard: a single badly misaligned or
///   wildly zoomed frame can collapse the common-coverage intersection to a
///   sliver, and silently cropping the output down to that sliver would be a
///   much worse outcome than keeping the full canvas. Callers should log a
///   warning and fall back to the full canvas when this guard trips.
///
/// `width` / `height` must match the dimensions `mask` was built for (the
/// same values passed to [`coverage_mask`] / [`largest_true_rectangle`]).
#[must_use]
pub fn resolve_common_crop(
    mask: &[bool],
    width: usize,
    height: usize,
) -> Option<(usize, usize, usize, usize)> {
    let rect @ (_, _, rw, rh) = largest_true_rectangle(mask, width, height)?;

    let canvas_area = width * height;
    if canvas_area == 0 {
        return None;
    }
    let rect_area = rw * rh;

    // Nothing to crop: the rectangle already covers the whole canvas.
    if rect_area >= canvas_area {
        return None;
    }

    // Rogue-frame guard: refuse to crop down to less than a quarter of the
    // canvas. Compare via cross-multiplication to stay in exact integer
    // arithmetic (avoids float rounding at the 25% boundary).
    if rect_area * 4 < canvas_area {
        return None;
    }

    Some(rect)
}

#[cfg(test)]
mod resolve_common_crop_tests {
    use super::resolve_common_crop;

    #[test]
    fn all_true_mask_returns_none() {
        let mask = vec![true; 100 * 80];
        assert_eq!(resolve_common_crop(&mask, 100, 80), None);
    }

    #[test]
    fn border_band_returns_correct_inner_rect() {
        // 100×80 canvas with a 10%-border band of `false` on every side:
        // border thickness = 10 px (10% of 100) horizontally and 8 px (10%
        // of 80) vertically, so the true inner rectangle is
        // x in [10, 90), y in [8, 72) => (10, 8, 80, 64).
        let (w, h) = (100usize, 80usize);
        let mut mask = vec![false; w * h];
        for y in 8..72 {
            for x in 10..90 {
                mask[y * w + x] = true;
            }
        }
        assert_eq!(resolve_common_crop(&mask, w, h), Some((10, 8, 80, 64)));
    }

    #[test]
    fn mostly_false_mask_returns_none() {
        // Only a small 10×10 block valid inside a 100×80 canvas — far below
        // the 25% area guard (10*10=100 vs 100*80*0.25=2000).
        let (w, h) = (100usize, 80usize);
        let mut mask = vec![false; w * h];
        for y in 0..10 {
            for x in 0..10 {
                mask[y * w + x] = true;
            }
        }
        assert_eq!(resolve_common_crop(&mask, w, h), None);
    }
}
