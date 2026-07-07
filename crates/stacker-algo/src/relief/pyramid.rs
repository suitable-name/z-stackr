//! Post-solve pyramid smoothing for the `Relief` multigrid engine.
//!
//! After `MultigridSolver` diffuses a continuous depth-index field into the
//! low-confidence regions of a frame stack, that field is smoothed once more
//! before blending.
//!
//! `fuse_relief_multigrid` smooths the solved depth-index field with a
//! Gaussian image pyramid: built by repeated blur+decimate ([`reduce`]),
//! then collapsed from coarsest back to finest by repeated
//! blur+interpolate ([`expand_replace`]) — **replacing**, not adding back
//! to, each finer level.
//!
//! # Why `expand_replace` replaces rather than adds
//!
//! The collapse step needs to combine an upsampled coarser level into a
//! finer one. [`pyramid_smooth`] never computes or stores a residual (a
//! `reduce` result minus an `expand` result) while building its downsample
//! stack — it only ever calls the equivalent of `reduce`. An additive or
//! subtractive combine presupposes a residual was stashed in the
//! destination beforehand; here the destination already holds a *valid*
//! reduced value from the previous pass, so combining with add or subtract
//! would double-count or invert content rather than smooth it. Replace is
//! the only one of the three that produces a coherent smoothing result:
//! each finer level is unconditionally overwritten by the blurred-and-
//! upsampled coarser level, so the whole chain collapses to a pure
//! multi-octave lowpass reconstruction of the solved field — a real
//! Gaussian-pyramid blur, with no edge-awareness at all (unlike a guided
//! filter, which is edge-aware).
//!
//! If this reasoning is ever revisited, only [`expand_replace`]'s combine
//! mode needs to change — the kernel math, padding, and boundary handling
//! are independent of it.

/// Burt-Adelson 1-D kernel `[c, b, a, b, c]` with `a = 0.33`, `b = 0.25`,
/// `c = 0.25 - a / 2`. The 5x5 kernel used throughout is the separable
/// outer product of this with itself.
const REDUCTION_A: f32 = 0.33;
const REDUCTION_B: f32 = 0.25;
const REDUCTION_C: f32 = 0.25 - REDUCTION_A / 2.0;
const KERNEL_1D: [f32; 5] = [
    REDUCTION_C,
    REDUCTION_B,
    REDUCTION_A,
    REDUCTION_B,
    REDUCTION_C,
];

#[inline]
fn kernel_2d(ky: usize, kx: usize) -> f32 {
    KERNEL_1D[ky] * KERNEL_1D[kx]
}

/// Boundary-clipped kernel window for [`reduce`]: computes which taps of
/// the 5-wide kernel stay in bounds for destination coordinate `d`, given
/// the source dimension `src_len`.
const fn reduce_bounds(d: usize, src_len: usize) -> (usize, usize) {
    let min = if d == 0 { 2 } else { 0 };
    let d2 = 2 * d as isize;
    let src_len = src_len as isize;
    let max = if d2 == src_len - 1 {
        2
    } else if d2 == src_len - 2 {
        3
    } else {
        4
    };
    (min, max)
}

/// Boundary-clipped kernel window for [`expand_replace`]: computes which
/// taps of the 5-wide kernel stay in bounds for destination coordinate
/// `d`, given the destination dimension `dest_len`.
const fn expand_bounds(d: usize, dest_len: usize) -> (usize, usize) {
    let min = if d == 0 { 2 } else { (d == 1) as usize };
    let d = d as isize;
    let dest_len = dest_len as isize;
    let max = if d == dest_len - 1 {
        2
    } else if d == dest_len - 2 {
        3
    } else {
        4
    };
    (min, max)
}

/// Blur-and-decimate a scalar field by a factor of 2 in each dimension.
/// Destination size is `((w + 1) / 2, (h + 1) / 2)`. At the border, the
/// kernel window is clipped to stay in bounds and the result is
/// renormalised by the sum of the taps actually used (not zero-padded, not
/// edge-clamped).
fn reduce(src: &[f32], sw: usize, sh: usize) -> (Vec<f32>, usize, usize) {
    let dw = (sw + 1) / 2;
    let dh = (sh + 1) / 2;
    let mut dest = vec![0.0_f32; dw * dh];

    for dy in 0..dh {
        let (min_y, max_y) = reduce_bounds(dy, sh);
        for dx in 0..dw {
            let (min_x, max_x) = reduce_bounds(dx, sw);
            let mut weighted_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for ky in min_y..=max_y {
                let sy = dy * 2 + ky - 2;
                for kx in min_x..=max_x {
                    let sx = dx * 2 + kx - 2;
                    let w = kernel_2d(ky, kx);
                    weight_sum += w;
                    weighted_sum += w * src[sy * sw + sx];
                }
            }
            dest[dy * dw + dx] = weighted_sum / weight_sum;
        }
    }

    (dest, dw, dh)
}

/// Blur-and-interpolate a scalar field from `(cw, ch)` up to `(dw, dh)`
/// (`dw`/`dh` are always `2*cw`-ish, i.e. the size of the next-finer level
/// in the stack), **replacing** the destination outright rather than
/// adding to it — see the module docs for why replace is used here rather
/// than add/subtract. Only taps landing on an even `(dy, dx)` offset from
/// the coarse grid contribute; the boundary window is clipped and
/// renormalised exactly as in [`reduce`].
fn expand_replace(coarse: &[f32], cw: usize, ch: usize, dw: usize, dh: usize) -> Vec<f32> {
    let _ = ch; // kept for symmetry / documentation; not needed for indexing
    let mut dest = vec![0.0_f32; dw * dh];

    for dy in 0..dh {
        let (min_y, max_y) = expand_bounds(dy, dh);
        for dx in 0..dw {
            let (min_x, max_x) = expand_bounds(dx, dw);
            let mut weighted_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for ky in min_y..=max_y {
                let y_sum = dy + ky;
                if y_sum % 2 != 0 {
                    continue;
                }
                let sy = (y_sum - 2) / 2;
                for kx in min_x..=max_x {
                    let x_sum = dx + kx;
                    if x_sum % 2 != 0 {
                        continue;
                    }
                    let sx = (x_sum - 2) / 2;
                    let w = kernel_2d(ky, kx);
                    weight_sum += w;
                    weighted_sum += w * coarse[sy * cw + sx];
                }
            }
            dest[dy * dw + dx] = weighted_sum / weight_sum;
        }
    }

    dest
}

/// Smooths a scalar field via a Gaussian pyramid.
///
/// Smooths `field` (`width` x `height`) by building a Gaussian pyramid of
/// [`reduce`] levels until the accumulated scale reaches `max_scale`, then
/// collapsing that stack from coarsest back to finest with
/// [`expand_replace`].
///
/// `max_scale <= 1` is a no-op (the downsample-stack loop and the collapse
/// loop both degenerate to nothing), returning `field` unchanged.
#[must_use]
pub fn pyramid_smooth(field: &[f32], width: usize, height: usize, max_scale: usize) -> Vec<f32> {
    if max_scale <= 1 || width == 0 || height == 0 {
        return field.to_vec();
    }

    let mut levels: Vec<(Vec<f32>, usize, usize)> = vec![(field.to_vec(), width, height)];
    let mut current_scale = 1_usize;
    while current_scale < max_scale {
        let (data, w, h) = levels.last().expect("levels always has at least one entry");
        let (reduced, rw, rh) = reduce(data, *w, *h);
        levels.push((reduced, rw, rh));
        current_scale *= 2;
    }

    for i in (1..levels.len()).rev() {
        let (fine_part, coarse_part) = levels.split_at_mut(i);
        let (fine_data, fw, fh) = &mut fine_part[i - 1];
        let (coarse_data, cw, ch) = &coarse_part[0];
        *fine_data = expand_replace(coarse_data, *cw, *ch, *fw, *fh);
    }

    levels
        .into_iter()
        .next()
        .map_or_else(Vec::new, |(data, _, _)| data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_scale_le_1_is_identity() {
        let field: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let out0 = pyramid_smooth(&field, 8, 8, 0);
        let out1 = pyramid_smooth(&field, 8, 8, 1);
        assert_eq!(out0, field);
        assert_eq!(out1, field);
    }

    #[test]
    fn test_smooths_a_step_function() {
        // A hard step (left half 0.0, right half 1.0) on a moderately sized
        // field should come out with a graded transition band around the
        // boundary after smoothing, while the far interior of each side
        // stays close to its original flat value — much the same
        // qualitative signature a Gaussian blur would leave.
        let w = 32;
        let h = 32;
        let mut field = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                field[y * w + x] = if x < w / 2 { 0.0 } else { 1.0 };
            }
        }

        let smoothed = pyramid_smooth(&field, w, h, 4);

        // Interior pixels far from the boundary should be (almost)
        // untouched, since the smoothing radius is small relative to the
        // flat run length.
        let y = h / 2;
        assert!(
            smoothed[y * w + 2] < 0.1,
            "left interior should stay near 0"
        );
        assert!(
            smoothed[y * w + (w - 3)] > 0.9,
            "right interior should stay near 1"
        );

        // The boundary column should be a genuine blend, not a hard 0/1.
        let boundary = smoothed[y * w + w / 2];
        assert!(
            (0.05..0.95).contains(&boundary),
            "boundary should be smoothed, got {boundary}"
        );
    }

    #[test]
    fn test_uniform_field_is_unchanged() {
        // A perfectly flat field should be a fixed point of the whole
        // reduce/expand chain (every weighted average of a constant is
        // that same constant).
        let w = 16;
        let h = 16;
        let field = vec![0.42_f32; w * h];
        let smoothed = pyramid_smooth(&field, w, h, 4);
        for (i, &v) in smoothed.iter().enumerate() {
            assert!(
                (v - 0.42).abs() < 1e-4,
                "index {i}: expected ~0.42, got {v}"
            );
        }
    }
}
