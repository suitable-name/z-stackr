use crate::{
    Matrix3,
    refine::{downsample_luma_to_max_side, registration_rms_at_dims},
};
use stacker_core::image::PlanarImage;
#[cfg(feature = "akaze")]
use stacker_core::settings::AlignmentModeSetting;

/// Map an [`AlignmentModeSetting`] to the AKAZE/RANSAC coarse-seed matcher's
/// degrees of freedom.
///
/// `Affine` and `Registration` both seed from a 6-DOF least-squares affine
/// estimate (`ransac::AlignmentMode::Affine`) — for `Registration` the
/// refinement stage still only *solves* the 4-DOF similarity subset (aspect
/// and shear stay pinned at identity), so the seed's extra DOFs are simply
/// ignored there. For `Affine`, the refinement stage now actually engages
/// all six DOFs, so the affine seed's anisotropic-scale/shear estimate feeds
/// directly into the matching refinement search space instead of being
/// discarded.
#[cfg(feature = "akaze")]
pub const fn akaze_mode_for_alignment(mode: AlignmentModeSetting) -> crate::ransac::AlignmentMode {
    match mode {
        AlignmentModeSetting::Affine | AlignmentModeSetting::Registration => {
            crate::ransac::AlignmentMode::Affine
        }
        // Translation refinement solves shift + breathing scale (see
        // `align_frame`), so its coarse seed should carry the same DOFs —
        // a translation-only seed would systematically underestimate the
        // scale component the optimiser then has to recover from a
        // seed-centred bounded window.
        //
        // `Neural` never reaches this function: callers dispatch neural
        // alignment before calling into this module's AKAZE/RANSAC/
        // intensity-refinement machinery (see `align_frame`'s docs). It is
        // matched here only so this function stays exhaustive when the
        // `stacker-core/nn` feature is unified on in the build graph (e.g.
        // a `--features nn` GUI/CLI build) and the `Neural` variant exists;
        // the arm is unreachable in practice.
        #[cfg(feature = "nn")]
        AlignmentModeSetting::Neural => crate::ransac::AlignmentMode::TranslationAndScale,
        AlignmentModeSetting::Translation | AlignmentModeSetting::None => {
            crate::ransac::AlignmentMode::TranslationAndScale
        }
    }
}

/// Cheap post-refinement sanity gate: downsample both frames and compare the
/// registration objective (RMS) of the refined matrix against identity.
///
/// `refine_alignment_registration` runs a *bounded* optimiser whose search
/// interval is centred on the seed. When the seed is "sane" per
/// [`is_sane_seed`] but still wrong (e.g. a large-but-plausible spurious
/// translation), the optimiser can converge on a transform that is actually
/// worse than doing nothing at all. This gate catches that case cheaply: it
/// downsamples both images (repeatedly halving until the short side is at
/// most 256 px) and compares the RMS objective of the refined matrix against
/// the RMS objective of the identity matrix at that same small resolution.
/// If identity is strictly better, the refined matrix is rejected in favour
/// of identity.
///
/// Returns `true` if the refined matrix should be kept (refined RMS is not
/// worse than identity RMS), `false` if the caller should fall back to
/// identity.
pub fn refined_beats_identity(
    reference: &PlanarImage<f32>,
    frame: &PlanarImage<f32>,
    refined: &Matrix3<f32>,
) -> bool {
    let full_w = reference.width;
    let full_h = reference.height;

    let (ref_small, small_w, small_h) =
        downsample_luma_to_max_side(&reference.luma, full_w, full_h, 256);
    let (src_small, _, _) = downsample_luma_to_max_side(&frame.luma, full_w, full_h, 256);

    // tx/ty in RegistrationParams are fractional (pixels / image dimension),
    // so they transfer across pyramid scales unchanged; only the matrix
    // itself must be rebuilt at the small dimensions.
    let refined_rms = registration_rms_at_dims(
        refined, full_w, full_h, &ref_small, &src_small, small_w, small_h,
    );
    let identity_rms = registration_rms_at_dims(
        &Matrix3::<f32>::identity(),
        full_w,
        full_h,
        &ref_small,
        &src_small,
        small_w,
        small_h,
    );

    // Strictly worse than identity => reject. Equal or better => keep.
    refined_rms <= identity_rms
}
