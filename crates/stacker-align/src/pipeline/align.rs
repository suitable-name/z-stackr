use crate::{
    Matrix3,
    refine::{BoundedRefineOptions, refine_alignment_lk, refine_alignment_registration},
    transform::warp_image_clamped,
};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, OptimizerSetting},
};

use super::{refinement::refined_beats_identity, seed::is_sane_seed};

/// Run the bounded intensity refinement for one frame using `opts` DOF
/// gating and the caller's resolved [`OptimizerSetting`], returning the
/// refined matrix. Never errors: on any optimiser failure it falls back to
/// `initial` (the sanity-filtered seed), logging a `tracing::warn!` — the
/// same graceful-degradation contract every optimiser in this crate has.
///
/// Dispatch:
/// - [`OptimizerSetting::NelderMead`] — call [`refine_alignment_registration`]
///   only, exactly the pre-Lucas-Kanade behaviour.
/// - [`OptimizerSetting::LucasKanade`] — call [`refine_alignment_lk`] only.
///   On `Err` there is **no** Nelder-Mead fallback (per spec) — only the
///   fallback to `initial` every optimiser has on error.
/// - [`OptimizerSetting::Auto`] — try Lucas-Kanade first; if it errors, OR
///   its final RMS does not improve on its own starting RMS (regression),
///   fall back to running Nelder-Mead from the same `initial` seed. The
///   starting/final RMS values used for this check come directly from
///   [`crate::refine::LkResult::starting_rms`]/
///   [`crate::refine::LkResult::final_rms`] — both computed by
///   `refine_alignment_lk` at the finest pyramid level it actually
///   processed (never full resolution) — rather than two separate
///   full-resolution [`crate::refine::lk_rms`] calls, which used to double
///   the full-resolution warp+RMS cost of every Auto-mode frame.
fn refine_matrix(
    reference: &PlanarImage<f32>,
    frame: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    opts: &BoundedRefineOptions,
    optimizer: OptimizerSetting,
    frame_idx: usize,
) -> Matrix3<f32> {
    let nelder_mead = || {
        refine_alignment_registration(reference, frame, initial, opts).unwrap_or_else(|err| {
            tracing::warn!(
                frame_idx,
                error = %err,
                "Nelder-Mead intensity-based refinement failed; falling back to the \
                 sanity-filtered initial matrix for this frame"
            );
            *initial
        })
    };

    match optimizer {
        OptimizerSetting::NelderMead => nelder_mead(),
        OptimizerSetting::LucasKanade => refine_alignment_lk(reference, frame, initial, opts)
            .map_or_else(
                |err| {
                    tracing::warn!(
                        frame_idx,
                        error = %err,
                        "Lucas-Kanade intensity-based refinement failed; falling back to the \
                         sanity-filtered initial matrix for this frame (no Nelder-Mead fallback \
                         in LucasKanade-only mode)"
                    );
                    *initial
                },
                |r| r.matrix,
            ),
        OptimizerSetting::Auto => match refine_alignment_lk(reference, frame, initial, opts) {
            Ok(lk_result) => {
                let starting_rms = lk_result.starting_rms;
                let final_rms = lk_result.final_rms;
                if final_rms.is_finite() && final_rms <= starting_rms {
                    lk_result.matrix
                } else {
                    tracing::warn!(
                        frame_idx,
                        starting_rms,
                        final_rms,
                        "Lucas-Kanade regressed vs its own starting RMS; falling back to \
                         Nelder-Mead for this frame"
                    );
                    nelder_mead()
                }
            }
            Err(err) => {
                tracing::warn!(
                    frame_idx,
                    error = %err,
                    "Lucas-Kanade intensity-based refinement errored; falling back to \
                     Nelder-Mead for this frame"
                );
                nelder_mead()
            }
        },
    }
}

/// Align a single frame to the reference using **intensity-based** refinement,
/// then warp it. This is the single shared dispatch used by both apps.
///
/// `seed` is the already-resolved initial transform — the AKAZE hint chosen by
/// the caller (when available), or the previous frame's matrix for the
/// sequential ("chain") alignment, or identity for the first frame. AKAZE
/// matching is not performed here. The seed is sanity-filtered internally via
/// [`is_sane_seed`]: an implausible seed is replaced by identity before
/// refinement, so callers do not need to pre-filter their hints.
///
/// `frame_idx` identifies the frame for the caller's own logging; it is also
/// used to identify the frame in the post-refinement gate's warning log when
/// a refined transform is rejected in favour of identity.
///
/// Dispatch — the alignment-mode ladder is `Translation ⊂ Registration ⊂
/// Affine`, each strictly adding degrees of freedom to the last:
/// * [`AlignmentModeSetting::None`] — skip refinement, warp with the seed.
/// * [`AlignmentModeSetting::Translation`] — shift X/Y **plus** centre-anchored
///   uniform scale (rotation fixed). Scale stays enabled so focus-breathing
///   magnification is corrected (a pure X/Y shift cannot represent it and
///   produces unusable stacks on real macro data).
/// * [`AlignmentModeSetting::Registration`] — shift X/Y + uniform scale +
///   rotation (a 4-DOF similarity transform). This is the **default** —
///   the best general-purpose choice, since it corrects the shift,
///   focus-breathing scale, and rotation present in most real stacks
///   without the extra anisotropic-scale/shear DOFs `Affine` adds.
/// * [`AlignmentModeSetting::Affine`] — everything `Registration` solves,
///   **plus** anisotropic (X/Y-independent) scale and shear: a true 6-DOF
///   affine solve. Use this when the source frames genuinely need
///   independent X/Y scaling or shear correction (e.g. sensor/lens
///   distortion that a similarity transform cannot represent); it is more
///   expensive to converge and more prone to overfitting sensor noise on
///   otherwise-similarity-only misalignment, which is why `Registration`
///   (not `Affine`) is the default.
///
/// All three active modes are refined by the same bounded DOF-gated
/// registration objective; only the bounded DOF gating differs between
/// modes. Which *optimiser* solves that objective is a separate axis,
/// selected by `optimizer` (see [`OptimizerSetting`]): [`OptimizerSetting::Auto`]
/// (the typical choice) tries the cheaper pyramid Lucas-Kanade / Gauss-Newton
/// optimiser (`refine_alignment_lk`) first and falls back to the original
/// Nelder-Mead coarse-to-fine bounded simplex (`refine_alignment_registration`)
/// on failure or RMS regression; [`OptimizerSetting::LucasKanade`] /
/// [`OptimizerSetting::NelderMead`] force one optimiser unconditionally (see
/// [`refine_matrix`]'s doc comment for the exact fallback ladder). These
/// modes are inherently bounded, so the `bounded` argument is ignored. After
/// refinement, a cheap downsampled RMS comparison against identity guards
/// against a bounded-but-wrong seed producing a transform that is worse than
/// doing nothing (see [`refined_beats_identity`]) — this gate runs
/// regardless of which optimiser was used.
///
/// All modes warp identically with the faithful edge-clamped spline kernel.
///
/// Graceful fallbacks: a refinement `Err` falls back to the (sanity-filtered)
/// initial matrix; a warp `Err` returns `(identity, frame)` so the caller
/// always has a usable, unwarped frame and a matrix it can detect as the
/// failure path.
///
/// Returns `(matrix, warped_image)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn align_frame(
    frame: PlanarImage<f32>,
    reference: &PlanarImage<f32>,
    seed: Matrix3<f32>,
    alignment_mode: AlignmentModeSetting,
    optimizer: OptimizerSetting,
    _bounded: bool,
    frame_idx: usize,
    brightness_target: Option<&crate::brightness::BrightnessTarget>,
) -> (Matrix3<f32>, PlanarImage<f32>) {
    // Sanity-filter the seed: an implausible feature-match estimate is replaced
    // by identity so the intensity optimiser starts from a safe point.
    let initial = if is_sane_seed(&seed, reference.width, reference.height) {
        seed
    } else {
        Matrix3::identity()
    };

    // DOF selection per mode — a strict ladder, each mode a superset of the
    // last:
    //   * `Translation` disables ROTATION (and aspect/shear stay off).
    //     Scale stays enabled because focus breathing (per-frame
    //     magnification as the focus distance changes) is present in
    //     essentially every macro focus stack — a pure X/Y shift cannot
    //     represent it, and the residual radial misalignment grows with
    //     distance from the image centre until the fused result is
    //     unusable. Solving shift + centre-anchored uniform scale (no
    //     rotation) is the minimal model that survives breathing; see
    //     `ransac::BreathingCorrector`'s module docs for the same model.
    //   * `Registration` enables rotation too (aspect/shear stay off): the
    //     full 4-DOF similarity transform. This is the **default**.
    //   * `Affine` enables rotation AND aspect AND shear: the full 6-DOF
    //     affine solve. This is the only mode that can recover anisotropic
    //     (independent X/Y) scale or shear.
    let allow_rotation = !matches!(alignment_mode, AlignmentModeSetting::Translation);
    let allow_aspect_shear = matches!(alignment_mode, AlignmentModeSetting::Affine);

    // ── Intensity-based subpixel refinement ─────────────────────────────────
    let matrix = match alignment_mode {
        // `Neural` never reaches this function in practice — the neural
        // alignment path is dispatched entirely by the caller (see
        // `nn_align_planar` in the GUI, and the pipeline's neural alignment
        // dispatch) before any per-frame call into this shared classical
        // refinement dispatch. Treated as a pass-through (no refinement,
        // warp with the already-resolved seed) purely so this match stays
        // exhaustive when the `stacker-core/nn` feature is unified on in the
        // build graph and the `Neural` variant exists.
        #[cfg(feature = "nn")]
        AlignmentModeSetting::Neural | AlignmentModeSetting::None => initial,
        #[cfg(not(feature = "nn"))]
        AlignmentModeSetting::None => initial,
        AlignmentModeSetting::Registration
        | AlignmentModeSetting::Affine
        | AlignmentModeSetting::Translation => {
            let opts = BoundedRefineOptions {
                allow_scale: true,
                allow_rotation,
                allow_aspect: allow_aspect_shear,
                allow_shear: allow_aspect_shear,
                ..BoundedRefineOptions::default()
            };
            let refined = refine_matrix(reference, &frame, &initial, &opts, optimizer, frame_idx);

            // Post-refinement gate: a bounded-but-garbage seed can make the
            // optimiser converge on something worse than no alignment at
            // all. Compare cheaply at a downsampled resolution and fall back
            // to identity if the refined transform loses to it.
            if refined_beats_identity(reference, &frame, &refined) {
                refined
            } else {
                tracing::warn!(
                    frame_idx,
                    "refined alignment matrix scored worse than identity at the post-refinement \
                     sanity gate; falling back to identity for this frame"
                );
                Matrix3::identity()
            }
        }
    };

    // All modes warp identically with the faithful edge-clamped spline kernel.
    let warp_result = warp_image_clamped(&frame, &matrix);

    // On a warp failure return identity (not the solved matrix) so callers can
    // detect the warp-failure path and skip dependent re-warps; the unwarped
    // frame is returned by value.
    warp_result.map_or_else(
        |err| {
            tracing::warn!(
                frame_idx,
                error = %err,
                "warp_image_clamped failed; returning identity matrix and the \
                 unwarped frame for this frame"
            );
            (Matrix3::<f32>::identity(), frame)
        },
        |mut warped| {
            if let Some(target) = brightness_target {
                crate::brightness::apply_brightness_correction(&mut warped, target);
            }
            (matrix, warped)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        refine::{RegistrationParams, params_to_matrix, registration_rms_at_dims},
        transform::warp_image_clamped as warp_clamped,
    };

    /// Build a synthetic gradient `PlanarImage<f32>` (non-symmetric so a
    /// misaligned warp is measurably different from a correctly aligned one).
    fn make_gradient_image(width: usize, height: usize) -> PlanarImage<f32> {
        let mut img = PlanarImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let v = ((x as f32 * 0.11).sin() + (y as f32 * 0.09).cos())
                    .mul_add(0.4, 0.5)
                    .clamp(0.0, 1.0);
                img.luma[y * width + x] = v;
                img.chroma_a[y * width + x] = v * 0.5;
                img.chroma_b[y * width + x] = v * 0.25;
            }
        }
        img
    }

    /// (a) Synthetic reference + source generated by warping with a known
    /// small translation: the post-refinement gate must keep the refined
    /// matrix because its RMS beats identity's.
    #[test]
    fn post_refinement_gate_keeps_good_refinement() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);

        let known_p = RegistrationParams {
            tx: 0.03,
            ty: 0.02,
            scale: 1.0,
            rotate: 0.0,
            aspect: 1.0,
            shear: 0.0,
        };
        let known_matrix =
            params_to_matrix(&known_p, w, h).expect("known params should produce a valid matrix");
        let source =
            warp_clamped(&reference, &known_matrix).expect("warp_image_clamped should succeed");

        let (matrix, _warped) = align_frame(
            source,
            &reference,
            Matrix3::identity(),
            AlignmentModeSetting::Registration,
            OptimizerSetting::NelderMead,
            true,
            0,
            None,
        );

        // The gate should have kept a non-identity refined matrix (it beats
        // identity RMS by construction: the source truly is shifted).
        let diff = (matrix - Matrix3::<f32>::identity()).norm();
        assert!(
            diff > 1.0e-3,
            "expected align_frame to keep a non-identity refined matrix for a genuinely \
             shifted source; diff from identity was {diff:.6}"
        );
    }

    /// Translation mode must recover focus-breathing magnification: a source
    /// generated by scaling the reference about its centre (no shift, no
    /// rotation — the classic breathing signature) must come back with a
    /// non-trivial scale in the solved matrix and an RMS no worse than
    /// identity. Regression test for the "Translation result unusable on
    /// breathing stacks" report: with the scale DOF disabled this recovers
    /// nothing and the fused stack shows radial misalignment.
    #[test]
    fn translation_mode_recovers_breathing_scale() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);

        // Centre-anchored 1.5% magnification — a typical breathing step.
        let known_p = RegistrationParams {
            tx: 0.0,
            ty: 0.0,
            scale: 1.015,
            rotate: 0.0,
            aspect: 1.0,
            shear: 0.0,
        };
        let known_matrix =
            params_to_matrix(&known_p, w, h).expect("known params should produce a valid matrix");
        let source =
            warp_clamped(&reference, &known_matrix).expect("warp_image_clamped should succeed");

        let (matrix, _warped) = align_frame(
            source.clone(),
            &reference,
            Matrix3::identity(),
            AlignmentModeSetting::Translation,
            OptimizerSetting::NelderMead,
            true,
            0,
            None,
        );

        // The solved matrix must carry a scale component (not pinned to 1.0)
        // and must not rotate.
        let sx = matrix[(0, 0)].hypot(matrix[(1, 0)]);
        assert!(
            (sx - 1.0).abs() > 1.0e-3,
            "Translation mode should have solved a breathing scale, got sx={sx:.5}"
        );
        assert!(
            matrix[(1, 0)].abs() < 1.0e-3,
            "Translation mode must not introduce rotation, got a10={}",
            matrix[(1, 0)]
        );

        // And the corrected result must beat doing nothing.
        let solved_rms =
            registration_rms_at_dims(&matrix, w, h, &reference.luma, &source.luma, w, h);
        let identity_rms = registration_rms_at_dims(
            &Matrix3::<f32>::identity(),
            w,
            h,
            &reference.luma,
            &source.luma,
            w,
            h,
        );
        assert!(
            solved_rms < identity_rms,
            "breathing-corrected RMS ({solved_rms:.6}) must beat identity ({identity_rms:.6})"
        );
    }

    /// (b) Source identical to reference but refinement seeded with a large
    /// "sane-looking" translation (20% of width, inside the `is_sane_seed`
    /// bound). After the gate, the final matrix must not make the result
    /// worse than unaligned: either the gate falls back to identity, or the
    /// refined matrix's RMS is no worse than identity's.
    #[test]
    fn post_refinement_gate_rejects_garbage_seed_on_identical_source() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);
        let source = reference.clone();

        // Seed with a large but "sane" translation: 20% of width.
        let mut seed = Matrix3::<f32>::identity();
        seed[(0, 2)] = w as f32 * 0.20;

        let (matrix, _warped) = align_frame(
            source,
            &reference,
            seed,
            AlignmentModeSetting::Registration,
            OptimizerSetting::NelderMead,
            true,
            0,
            None,
        );

        // Evaluate RMS of the final matrix vs identity at full resolution —
        // the gate must guarantee the chosen matrix is not worse than doing
        // nothing, i.e. the garbage seed cannot make the result worse than
        // unaligned.
        let full_w = reference.width;
        let full_h = reference.height;
        let final_rms = registration_rms_at_dims(
            &matrix,
            full_w,
            full_h,
            &reference.luma,
            &reference.luma,
            full_w,
            full_h,
        );
        let identity_rms = registration_rms_at_dims(
            &Matrix3::<f32>::identity(),
            full_w,
            full_h,
            &reference.luma,
            &reference.luma,
            full_w,
            full_h,
        );

        assert!(
            final_rms <= identity_rms + 1.0e-6,
            "garbage seed must not make the result worse than unaligned: \
             final_rms={final_rms:.6}, identity_rms={identity_rms:.6}"
        );
    }

    /// `Affine` mode must recover genuine anisotropy: a source warped with a
    /// known aspect (Y-scale != X-scale), shear, and small shift must align
    /// via `AlignmentModeSetting::Affine` to a matrix that both beats
    /// identity RMS and beats `AlignmentModeSetting::Registration` on the
    /// same input — proving the extra aspect/shear DOFs actually engage
    /// (Registration pins them at identity, so it can only chase the
    /// residual similarity component of the anisotropic warp).
    #[test]
    fn affine_mode_recovers_anisotropy_and_beats_registration() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);

        // Known anisotropic scale + shear + small shift — not representable
        // by any similarity (Registration) transform.
        let known_p = RegistrationParams {
            tx: 0.01,
            ty: -0.008,
            scale: 1.0,
            rotate: 0.0,
            aspect: 1.02,
            shear: 0.01,
        };
        let known_matrix =
            params_to_matrix(&known_p, w, h).expect("known params should produce a valid matrix");
        let source =
            warp_clamped(&reference, &known_matrix).expect("warp_image_clamped should succeed");

        let (affine_matrix, _warped) = align_frame(
            source.clone(),
            &reference,
            Matrix3::identity(),
            AlignmentModeSetting::Affine,
            OptimizerSetting::NelderMead,
            true,
            0,
            None,
        );
        let (registration_matrix, _warped2) = align_frame(
            source.clone(),
            &reference,
            Matrix3::identity(),
            AlignmentModeSetting::Registration,
            OptimizerSetting::NelderMead,
            true,
            0,
            None,
        );

        let identity_rms = registration_rms_at_dims(
            &Matrix3::<f32>::identity(),
            w,
            h,
            &reference.luma,
            &source.luma,
            w,
            h,
        );
        let affine_rms =
            registration_rms_at_dims(&affine_matrix, w, h, &reference.luma, &source.luma, w, h);
        let registration_rms = registration_rms_at_dims(
            &registration_matrix,
            w,
            h,
            &reference.luma,
            &source.luma,
            w,
            h,
        );

        assert!(
            affine_rms < identity_rms,
            "Affine mode must beat identity: affine_rms={affine_rms:.6}, identity_rms={identity_rms:.6}"
        );
        assert!(
            affine_rms < registration_rms,
            "Affine mode must beat Registration on a genuinely anisotropic/sheared source \
             (proving the extra aspect/shear DOFs engage): affine_rms={affine_rms:.6}, \
             registration_rms={registration_rms:.6}"
        );
    }

    /// `OptimizerSetting::Auto` must still produce a sane, non-worse-than-
    /// identity result even when the resolved seed is pathological for
    /// Lucas-Kanade (huge displacement, far beyond any pyramid level's
    /// basin of attraction) — the point of the Auto ladder is that a
    /// Lucas-Kanade failure/regression falls back to Nelder-Mead, and the
    /// existing post-refinement identity gate still applies on top of
    /// whichever optimiser wins.
    #[test]
    fn auto_mode_falls_back_and_stays_sane_on_pathological_seed() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);
        let source = reference.clone();

        // Large but still "sane" per `is_sane_seed` (< 25% of width/height),
        // which is the only way a pathological seed can reach the
        // optimiser at all through `align_frame`'s sanity filter.
        let mut seed = Matrix3::<f32>::identity();
        seed[(0, 2)] = w as f32 * 0.20;
        seed[(1, 2)] = h as f32 * 0.20;

        let (matrix, _warped) = align_frame(
            source,
            &reference,
            seed,
            AlignmentModeSetting::Registration,
            OptimizerSetting::Auto,
            true,
            0,
            None,
        );

        let final_rms =
            registration_rms_at_dims(&matrix, w, h, &reference.luma, &reference.luma, w, h);
        let identity_rms = registration_rms_at_dims(
            &Matrix3::<f32>::identity(),
            w,
            h,
            &reference.luma,
            &reference.luma,
            w,
            h,
        );

        assert!(
            final_rms <= identity_rms + 1.0e-6,
            "Auto mode with a pathological seed must not end up worse than identity: \
             final_rms={final_rms:.6}, identity_rms={identity_rms:.6}"
        );
    }

    /// `OptimizerSetting::LucasKanade` and `OptimizerSetting::NelderMead`
    /// both recover a genuine small shift at least as well as `Auto` on the
    /// same well-conditioned input — a basic interface-parity smoke test
    /// for the three-way dispatch (not a strict quality comparison).
    #[test]
    fn all_three_optimizer_settings_recover_a_genuine_shift() {
        let w = 160_usize;
        let h = 120_usize;
        let reference = make_gradient_image(w, h);

        let known_p = RegistrationParams {
            tx: 0.025,
            ty: 0.018,
            scale: 1.0,
            rotate: 0.0,
            aspect: 1.0,
            shear: 0.0,
        };
        let known_matrix =
            params_to_matrix(&known_p, w, h).expect("known params should produce a valid matrix");
        let source =
            warp_clamped(&reference, &known_matrix).expect("warp_image_clamped should succeed");

        let identity_rms = registration_rms_at_dims(
            &Matrix3::<f32>::identity(),
            w,
            h,
            &reference.luma,
            &source.luma,
            w,
            h,
        );

        for optimizer in [
            OptimizerSetting::Auto,
            OptimizerSetting::LucasKanade,
            OptimizerSetting::NelderMead,
        ] {
            let (matrix, _warped) = align_frame(
                source.clone(),
                &reference,
                Matrix3::identity(),
                AlignmentModeSetting::Registration,
                optimizer,
                true,
                0,
                None,
            );
            let rms = registration_rms_at_dims(&matrix, w, h, &reference.luma, &source.luma, w, h);
            assert!(
                rms < identity_rms,
                "{optimizer:?}: refined RMS ({rms:.6}) must beat identity RMS ({identity_rms:.6})"
            );
        }
    }
}
