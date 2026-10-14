//! The Strata saliency: where in a frame is it sharp.
//!
//! The metric is the magnitude of the 4-neighbour Laplacian of the luma plane,
//! smoothed by [`BLUR_PASSES`] applications of the pyramid's 5-tap blur. The
//! result is only ever compared between frames at the same pixel (a running
//! argmax), so its absolute scale does not matter.

use crate::apex::pyramid::blur_plane_into;
use rayon::prelude::*;
use stacker_core::{fma::fma_f32, image::PlanarImage};

/// Number of times the 5-tap separable blur
/// ([`apply_gaussian_blur`](crate::apex::pyramid::apply_gaussian_blur)) is
/// applied to the raw Laplacian magnitude.
///
/// One pass of the kernel `[0.085, 0.25, 0.33, 0.25, 0.085]` has a variance of
/// about `1.18` (`sigma` about `1.09`). Repeated convolution sums variances, so
/// `k` passes approximate `sigma = sqrt(k * 1.18)`; `k = 5` gives `sigma` about
/// `2.43`, a local average over a few pixels without a second blur kernel in
/// the code base.
pub const BLUR_PASSES: usize = 5;

/// How far (in pixels) a pixel of the saliency map reads from the luma plane:
/// one pixel for the Laplacian plus two per blur pass (the kernel has radius 2).
pub const SALIENCY_REACH_PX: usize = 1 + 2 * BLUR_PASSES;

/// The per-pixel 4-neighbour Laplacian magnitude of a luma plane,
/// `|up + down + left + right - 4 * center|`, with the border replicated.
///
/// This is the CPU implementation of the first saliency step; a device may
/// run the same computation through
/// [`StrataKernels::saliency`](super::StrataKernels::saliency).
///
/// # Panics
/// Panics when `luma` holds fewer than `width * height` samples.
#[must_use]
pub fn laplacian_magnitude(luma: &[f32], width: usize, height: usize) -> Vec<f32> {
    let len = width * height;
    let mut out = vec![0.0_f32; len];
    if len == 0 {
        return out;
    }
    assert!(
        luma.len() >= len,
        "luma plane is shorter than width * height"
    );
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let up_row = y.saturating_sub(1);
        let down_row = (y + 1).min(height - 1);
        for (x, cell) in row.iter_mut().enumerate() {
            let left_col = x.saturating_sub(1);
            let right_col = (x + 1).min(width - 1);
            let center = luma[y * width + x];
            let up = luma[up_row * width + x];
            let down = luma[down_row * width + x];
            let left = luma[y * width + left_col];
            let right = luma[y * width + right_col];
            *cell = fma_f32(-4.0, center, up + down + left + right).abs();
        }
    });
    out
}

/// Strata's saliency map of one luma plane: [`laplacian_magnitude`] smoothed by
/// [`BLUR_PASSES`] blur passes.
///
/// # Panics
/// Panics when `luma` holds fewer than `width * height` samples.
#[must_use]
pub fn compute_saliency(luma: &[f32], width: usize, height: usize) -> Vec<f32> {
    let magnitude = laplacian_magnitude(luma, width, height);
    smooth_saliency(magnitude, width, height)
}

/// Applies the [`BLUR_PASSES`] blur passes to a raw Laplacian magnitude plane.
///
/// Exposed so a device that computes only the Laplacian step can finish on the
/// CPU with the identical blur.
///
/// # Panics
/// Panics when `magnitude` does not hold exactly `width * height` samples.
#[must_use]
pub fn smooth_saliency(magnitude: Vec<f32>, width: usize, height: usize) -> Vec<f32> {
    let len = width * height;
    assert_eq!(
        magnitude.len(),
        len,
        "the magnitude plane holds width * height samples"
    );
    if len == 0 {
        return magnitude;
    }
    // One plane, three buffers for all passes: the input and output ping-pong,
    // and the horizontal pass writes into the shared scratch plane.
    let mut current = magnitude;
    let mut next = vec![0.0_f32; len];
    let mut scratch = vec![0.0_f32; len];
    for _ in 0..BLUR_PASSES {
        blur_plane_into(width, height, &current, &mut scratch, &mut next);
        std::mem::swap(&mut current, &mut next);
    }
    current
}

/// For every pixel the index of the frame with the largest saliency, and that
/// saliency. A strictly larger saliency is needed to replace the current
/// winner, so ties go to the frame that comes first.
///
/// # Panics
/// Panics when there are more than 65 536 frames or a frame is smaller than
/// `width * height`.
#[must_use]
pub fn compute_argmax(
    frames: &[PlanarImage<f32>],
    width: usize,
    height: usize,
) -> (Vec<u16>, Vec<f32>) {
    let len = width * height;
    let mut argmax_idx = vec![0_u16; len];
    let mut argmax_val = vec![f32::NEG_INFINITY; len];
    for (i, frame) in frames.iter().enumerate() {
        let saliency = compute_saliency(&frame.luma, width, height);
        update_argmax(&mut argmax_idx, &mut argmax_val, &saliency, i);
    }
    (argmax_idx, argmax_val)
}

/// Folds one frame's saliency into the running argmax: pixels where `saliency`
/// is strictly larger than the stored value take `frame_index` and the new
/// value.
///
/// # Panics
/// Panics when `frame_index` does not fit in a `u16`.
pub fn update_argmax(
    argmax_idx: &mut [u16],
    argmax_val: &mut [f32],
    saliency: &[f32],
    frame_index: usize,
) {
    let idx = u16::try_from(frame_index).expect("Strata supports at most 65 536 frames");
    argmax_idx
        .par_iter_mut()
        .zip(argmax_val.par_iter_mut())
        .zip(saliency.par_iter())
        .for_each(|((winner, best), &value)| {
            if value > *best {
                *best = value;
                *winner = idx;
            }
        });
}
