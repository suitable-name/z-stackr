//! Post-solve pyramid smoothing for the `Relief` multigrid engine.
//!
//! After [`MultigridSolver`](super::multigrid::MultigridSolver) diffuses a
//! continuous depth-index field into the low-confidence regions of a frame stack,
//! that field is smoothed once more before the frames are blended.
//!
//! [`pyramid_smooth`] smooths the field with a Gaussian image pyramid: built by
//! repeated blur and decimation (`reduce`), then collapsed from the coarsest level
//! back to the finest by repeated blur and interpolation (`expand_replace`), which
//! **replaces** the finer level instead of adding a residual to it.
//!
//! # Why the collapse replaces rather than adds
//!
//! Building the stack never stores a residual (a `reduce` result minus an `expand`
//! result), so there is nothing to add back: every finer level already holds a
//! valid reduced value, and adding or subtracting the upsampled coarser level would
//! double-count or invert content. Replacing is the combine that gives a coherent
//! result: the whole chain collapses to a multi-octave lowpass reconstruction of the
//! solved field, a Gaussian-pyramid blur without any edge awareness (unlike a
//! guided filter).
//!
//! # Determinism
//!
//! Every output sample is computed from a fixed set of input samples in a fixed
//! order; rows are processed in parallel but each row is written by one task, so
//! the result does not depend on the thread count.

#![allow(
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "the tap sums are evaluated as plain multiply-add in a fixed order, which the GPU twin reproduces"
)]

use rayon::prelude::*;

/// Planes with fewer samples than this are processed on the calling thread.
const PARALLEL_MIN_SAMPLES: usize = 1 << 14;

/// The centre tap `a` of the 1-D kernel.
const KERNEL_A: f32 = 0.33;
/// The inner taps `b` of the 1-D kernel.
const KERNEL_B: f32 = 0.25;
/// The outer taps `c = 0.25 - a / 2` (0.085), spelled as the expression so that
/// the `f32` value is the one the Apex pyramid uses.
const KERNEL_C: f32 = 0.25 - KERNEL_A / 2.0;

/// The 1-D kernel `[c, b, a, b, c]` (the generating kernel of Burt and Adelson
/// with `a = 0.33`). The 5×5 kernel is the separable outer product of this with
/// itself.
const KERNEL_1D: [f32; 5] = [KERNEL_C, KERNEL_B, KERNEL_A, KERNEL_B, KERNEL_C];

/// The weight of kernel tap `(ky, kx)`.
const fn kernel_2d(ky: usize, kx: usize) -> f32 {
    KERNEL_1D[ky] * KERNEL_1D[kx]
}

/// The first and last tap (inclusive, in `0..=4`) of the 5-wide kernel that stay
/// inside the source for [`reduce`], for destination coordinate `d` and source
/// length `src_len`.
const fn reduce_bounds(d: usize, src_len: usize) -> (usize, usize) {
    let min = if d == 0 { 2 } else { 0 };
    let source_centre = 2 * d;
    let max = if source_centre + 1 == src_len {
        2
    } else if source_centre + 2 == src_len {
        3
    } else {
        4
    };
    (min, max)
}

/// The first and last tap (inclusive, in `0..=4`) of the 5-wide kernel that stay
/// inside the destination for [`expand_replace`], for destination coordinate `d`
/// and destination length `dest_len`.
const fn expand_bounds(d: usize, dest_len: usize) -> (usize, usize) {
    let min = if d == 0 {
        2
    } else if d == 1 {
        1
    } else {
        0
    };
    let max = if d + 1 == dest_len {
        2
    } else if d + 2 == dest_len {
        3
    } else {
        4
    };
    (min, max)
}

/// Run `row_fn(y, row)` for every `width`-wide row of `data`, in parallel when the
/// plane is large.
fn for_each_row<F>(data: &mut [f32], width: usize, row_fn: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    if data.len() >= PARALLEL_MIN_SAMPLES {
        data.par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| row_fn(y, row));
    } else {
        data.chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| row_fn(y, row));
    }
}

/// Blur and decimate a `sw × sh` field by a factor of 2 in each dimension.
///
/// The result is `((sw + 1) / 2) × ((sh + 1) / 2)`. At the border the kernel window
/// is clipped to stay inside the field and the weights of the taps actually used
/// are renormalised (no zero padding, no edge clamping).
fn reduce(src: &[f32], sw: usize, sh: usize) -> (Vec<f32>, usize, usize) {
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    let mut dest = vec![0.0_f32; dw * dh];

    for_each_row(&mut dest, dw, |dy, row| {
        let (min_y, max_y) = reduce_bounds(dy, sh);
        for (dx, out) in row.iter_mut().enumerate() {
            let (min_x, max_x) = reduce_bounds(dx, sw);
            let mut weighted_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for ky in min_y..=max_y {
                let sy = dy * 2 + ky - 2;
                for kx in min_x..=max_x {
                    let sx = dx * 2 + kx - 2;
                    let weight = kernel_2d(ky, kx);
                    weight_sum += weight;
                    weighted_sum += weight * src[sy * sw + sx];
                }
            }
            *out = weighted_sum / weight_sum;
        }
    });

    (dest, dw, dh)
}

/// Blur and interpolate a `cw`-wide coarse field up to `dw × dh` (the size of the
/// next finer level), **replacing** the destination outright (see the module
/// documentation).
///
/// Only the taps that land on a coarse sample (an even offset on both axes)
/// contribute; the border window is clipped and renormalised as in [`reduce`].
fn expand_replace(coarse: &[f32], cw: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut dest = vec![0.0_f32; dw * dh];

    for_each_row(&mut dest, dw, |dy, row| {
        let (min_y, max_y) = expand_bounds(dy, dh);
        for (dx, out) in row.iter_mut().enumerate() {
            let (min_x, max_x) = expand_bounds(dx, dw);
            let mut weighted_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for ky in min_y..=max_y {
                let y_sum = dy + ky;
                if !y_sum.is_multiple_of(2) {
                    continue;
                }
                let sy = (y_sum - 2) / 2;
                for kx in min_x..=max_x {
                    let x_sum = dx + kx;
                    if !x_sum.is_multiple_of(2) {
                        continue;
                    }
                    let sx = (x_sum - 2) / 2;
                    let weight = kernel_2d(ky, kx);
                    weight_sum += weight;
                    weighted_sum += weight * coarse[sy * cw + sx];
                }
            }
            *out = weighted_sum / weight_sum;
        }
    });

    dest
}

/// Smooth a scalar field with a Gaussian pyramid.
///
/// `field` is `width × height`. The pyramid is built by `reduce` (blur and
/// decimate by 2) while the accumulated scale, which starts at 1 and doubles with
/// every level, is below `max_scale`; it is then collapsed from the coarsest level
/// to the finest by `expand_replace`. `max_scale` is therefore an octave count in
/// disguise: 2 gives one `reduce`, 3 and 4 give two, 8 gives three. The `Relief`
/// multigrid engine passes `relief_smoothing_radius`.
///
/// `max_scale <= 1` (and an empty field) returns `field` unchanged. Every output
/// sample is a positive-weight average of input samples, so the result stays inside
/// the range of the input. Peak memory is about `4 · 1.33` bytes per pixel for the
/// level stack plus the output.
///
/// # Panics
/// Panics when `field` holds fewer than `width * height` samples.
#[must_use]
pub fn pyramid_smooth(field: &[f32], width: usize, height: usize, max_scale: usize) -> Vec<f32> {
    assert!(
        field.len() >= width * height,
        "the field must hold width * height samples"
    );
    pyramid_smooth_owned(field[..width * height].to_vec(), width, height, max_scale)
}

/// [`pyramid_smooth`] for a field the caller no longer needs: it takes the `Vec`
/// instead of copying the slice (the top level of the pyramid *is* the field).
///
/// # Panics
/// Panics when `field` does not hold exactly `width * height` samples.
#[must_use]
pub fn pyramid_smooth_owned(
    field: Vec<f32>,
    width: usize,
    height: usize,
    max_scale: usize,
) -> Vec<f32> {
    assert_eq!(
        field.len(),
        width * height,
        "the field must hold width * height samples"
    );
    if max_scale <= 1 || width == 0 || height == 0 {
        return field;
    }

    let mut levels: Vec<(Vec<f32>, usize, usize)> = vec![(field, width, height)];
    let mut current_scale = 1_usize;
    while current_scale < max_scale {
        let Some((data, level_width, level_height)) = levels.last() else {
            break;
        };
        let reduced = reduce(data, *level_width, *level_height);
        levels.push(reduced);
        current_scale *= 2;
    }

    // Collapse: replace every finer level by the expanded coarser one.
    while levels.len() > 1 {
        let Some((coarse, coarse_width, _)) = levels.pop() else {
            break;
        };
        if let Some((fine, fine_width, fine_height)) = levels.last_mut() {
            *fine = expand_replace(&coarse, coarse_width, *fine_width, *fine_height);
        }
    }

    levels
        .pop()
        .map_or_else(Vec::new, |(smoothed, _, _)| smoothed)
}
