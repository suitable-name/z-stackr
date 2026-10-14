//! The post-refinement identity gate.
//!
//! A bounded search is centred on its seed.  When the seed is plausible but wrong
//! (a large spurious shift from a bad feature match, a frame that has nothing in
//! common with the reference) the optimiser can still converge on a transform
//! that fits *worse* than doing nothing.  The gate catches that cheaply: both
//! luma planes are halved until the short side is at most 256 pixels and the
//! objective of the refined matrix is compared with the objective of the
//! identity at that size.  If the identity is strictly better, the identity wins.
//!
//! The comparison uses the refiners' own objective (mean-corrected RMS over the
//! pixels that received a Catmull-Rom sample, with the half-frame coverage rule),
//! so a refined matrix that shifts most of the frame out of view scores a penalty
//! and loses.  The matrix is the refiners' reference-to-source matrix, *before*
//! it is inverted for warping.  It is rebuilt at the small size through the exact
//! six-parameter decomposition ([`matrix_to_affine_params`]), so shift, scale,
//! rotation, aspect and shear all carry over.
//!
//! There is no margin: equal scores keep the refined matrix, exactly as in GitHub
//! z-stackr.  The gate has no GPU form (the planes are at most 256 pixels on the
//! short side) and no state, so the synchronous and the ask/tell drivers call it
//! the same way after their search has finished.

use std::borrow::Cow;

use nalgebra::Matrix3;
use stacker_core::image::PlanarImage;

use super::{LevelObjective, downsample_luma, matrix_to_affine_params};
use crate::device::ObjectiveSampler;

/// Short side (pixels) the planes are halved down to.
const GATE_MAX_SHORT_SIDE: usize = 256;

/// Halve `luma` until its short side is at most `max_short_side` (or a side
/// would vanish); borrows the input when no halving is needed.
fn downsample_to_short_side(
    luma: &[f32],
    width: usize,
    height: usize,
    max_short_side: usize,
) -> (Cow<'_, [f32]>, usize, usize) {
    let mut current: Cow<'_, [f32]> = Cow::Borrowed(luma);
    let (mut w, mut h) = (width, height);
    while w.min(h) > max_short_side && w > 1 && h > 1 {
        let (next, nw, nh) = downsample_luma(&current, w, h);
        current = Cow::Owned(next);
        w = nw;
        h = nh;
    }
    (current, w, h)
}

/// `true` when `refined` should be kept: its objective at a reduced size is not
/// worse than the identity's.
///
/// `refined` maps reference coordinates to source coordinates (what every refiner
/// returns).  Returns `false` when the two frames differ in size, when `refined`
/// is not finite, and when the planes are empty.
#[must_use]
pub fn refined_beats_identity(
    reference: &PlanarImage<f32>,
    frame: &PlanarImage<f32>,
    refined: &Matrix3<f32>,
) -> bool {
    let (width, height) = (reference.width, reference.height);
    if frame.width != width
        || frame.height != height
        || width == 0
        || height == 0
        || !refined.iter().all(|v| v.is_finite())
    {
        return false;
    }
    let (ref_small, small_w, small_h) =
        downsample_to_short_side(&reference.luma, width, height, GATE_MAX_SHORT_SIDE);
    let (src_small, _, _) =
        downsample_to_short_side(&frame.luma, width, height, GATE_MAX_SHORT_SIDE);

    // Shifts are fractions of the image size and the other parameters are
    // dimensionless, so only the matrix has to be rebuilt at the small size.
    let params = matrix_to_affine_params(refined, width, height);
    let Some(small) = params.to_matrix(small_w, small_h) else {
        return false;
    };

    let objective = LevelObjective::new(
        &ref_small,
        &src_small,
        small_w,
        small_h,
        ObjectiveSampler::Bicubic,
        None,
    );
    let refined_rms = objective.eval_forward(&small);
    let identity_rms = objective.eval_forward(&Matrix3::identity());
    // Strictly worse than the identity is rejected; equal or better is kept.
    refined_rms <= identity_rms
}

/// `refined` if it passes [`refined_beats_identity`], otherwise the identity.
///
/// The matrix returned is in the same (reference-to-source) convention as the
/// one given, so the caller inverts it for warping as usual.
#[must_use]
pub fn apply_identity_gate(
    reference: &PlanarImage<f32>,
    frame: &PlanarImage<f32>,
    refined: &Matrix3<f32>,
) -> Matrix3<f32> {
    if refined_beats_identity(reference, frame, refined) {
        *refined
    } else {
        Matrix3::identity()
    }
}
