/// Alignment helpers: AKAZE/RANSAC coarse seed + intensity-based subpixel refinement.
///
/// The two stages are deliberately **separate functions**:
///
/// * [`compute_akaze_coarse_seed`] only depends on one frame's own pixels and
///   the (fixed, precomputed-once) reference keypoints/descriptors — it has
///   no dependency on any other frame or on the sequential warm-start chain.
///   The caller runs this in parallel (via `rayon`) across every frame before
///   the sequential pass below.
/// * [`align_frame`] performs the intensity-based subpixel refinement + warp,
///   which genuinely must run sequentially: it aligns against a rolling
///   reference (the previously warped frame) and warm-starts from the already
///   -resolved coarse seed (from the parallel AKAZE pre-pass, or the previous
///   frame's solved matrix as fallback).
///
/// The caller (in `lib.rs`) handles tile-commit so that the `TileManager`
/// dependency stays in one place. Each aligned frame is warped and returned;
/// the caller commits it tile-by-tile and drops it before moving to the next
/// frame. Only **one** aligned frame lives in RAM at a time during the
/// sequential pass (≈ 3 × W × H × 4 bytes).
#[cfg(feature = "akaze")]
use stacker_align::akaze_match::{Descriptor, KeyPoint, KeypointMatcher};
#[cfg(feature = "akaze")]
use stacker_align::ransac::AlignmentEstimator;
use stacker_core::{
    error::StackerError,
    image::PlanarImage,
    settings::{AlignmentModeSetting, OptimizerSetting},
};

/// Compute a coarse alignment seed for `frame` via AKAZE feature matching +
/// RANSAC against the precomputed reference keypoints/descriptors.
///
/// This has no dependency on any other frame's result, so it is safe to call
/// from multiple threads concurrently (e.g. one call per frame via
/// `rayon::par_iter`) — unlike [`align_frame`], which depends on the
/// sequential warm-start chain.
///
/// Returns `None` if matching or RANSAC fails; the caller should fall back to
/// the previous frame's solved matrix (or identity for the first frame) in
/// that case.
#[cfg(feature = "akaze")]
#[must_use]
pub fn compute_akaze_coarse_seed(
    frame_idx: usize,
    frame: &PlanarImage<f32>,
    ref_kps: &[KeyPoint],
    ref_desc: &[Descriptor],
    alignment_mode: AlignmentModeSetting,
) -> Option<stacker_align::Matrix3<f32>> {
    let mr = KeypointMatcher::match_target(ref_kps, ref_desc, frame).ok()?;
    tracing::debug!(
        frame = frame_idx,
        target_kps = mr.kps1.len(),
        matched = mr.matches.len(),
        "AKAZE matching done"
    );
    let result = AlignmentEstimator::compute_matrix(
        &mr.matches,
        &mr.kps0,
        &mr.kps1,
        stacker_align::pipeline::akaze_mode_for_alignment(alignment_mode),
    )
    .ok();
    if result.is_none() {
        tracing::debug!(
            frame = frame_idx,
            "RANSAC failed — no coarse seed for this frame"
        );
    }
    result
}

/// Align `frame` against `ref_for_align` using the already-resolved `seed`
/// transform, then warp it.
///
/// `seed` is the coarse initial transform — the AKAZE hint computed by
/// [`compute_akaze_coarse_seed`] when available, or the previous frame's
/// solved matrix for the sequential ("chain") alignment, or identity for the
/// first frame. This function does not perform AKAZE matching itself.
///
/// On a refinement or warp failure the function falls back gracefully
/// (refinement `Err` → `seed`; warp `Err` → unwarped frame) rather than
/// propagating an error, consistent with the original pipeline behaviour.
// `unnecessary_wraps`: the shared dispatch this delegates to is infallible,
// but the `Result` return type is kept so the signature stays uniform with
// the rest of the pipeline's fallible stages (and so a future fallible step
// can be added here without an API break).
#[allow(clippy::too_many_arguments, clippy::unnecessary_wraps)]
pub fn align_frame(
    frame_idx: usize,
    frame: PlanarImage<f32>,
    ref_for_align: &PlanarImage<f32>,
    seed: stacker_align::Matrix3<f32>,
    alignment_mode: AlignmentModeSetting,
    optimizer: OptimizerSetting,
    bounded: bool,
    brightness_target: Option<&stacker_align::brightness::BrightnessTarget>,
) -> Result<(stacker_align::Matrix3<f32>, PlanarImage<f32>), StackerError> {
    // Delegate to the single shared dispatch in `stacker-align` so the CLI and
    // GUI can never drift apart. The mode selects the degrees of freedom
    // (Affine = shift+scale+rotation, Translation = shift only, None = skip);
    // `optimizer` selects which intensity optimiser solves that objective
    // (Auto/LucasKanade/NelderMead — see `stacker_align::pipeline::align_frame`'s
    // docs); `bounded` selects the constrained optimiser over the unbounded
    // one; the similarity DOFs are derived from the mode inside the shared
    // fn, which also sanity-filters the seed and applies the same graceful
    // fallbacks.
    let (final_matrix, warped) = stacker_align::align_frame(
        frame,
        ref_for_align,
        seed,
        alignment_mode,
        optimizer,
        bounded,
        frame_idx,
        brightness_target,
    );
    tracing::debug!(frame = frame_idx, "frame aligned");

    Ok((final_matrix, warped))
}

/// Compute one full-resolution PIXEL-space affine matrix per frame using a
/// trained neural alignment model (`AlignmentModeSetting::Neural`), for the
/// out-of-core tiled pipeline.
///
/// Unlike the classical [`compute_akaze_coarse_seed`]/[`align_frame`] split
/// (which loads and warps one frame at a time to stay memory-bounded), the
/// neural model needs the WHOLE stack resident at once. This function loads
/// every frame, decodes and preprocesses it, and hands the full
/// (full-resolution) stack to `stacker_nn::runtime::LoadedAlignModel::align`,
/// which internally calls `stacker_nn::bridge::align_planar` — that bridge
/// function downscales every frame to a 512px-longest-side copy before
/// running the batch alignment network (see its own docs for the exact
/// downscale/coordinate-conjugation contract). This function itself never
/// holds more than one full-resolution decoded stack in RAM at a time (the
/// same footprint the in-RAM GUI path already accepts for its own
/// live-preview alignment loop) — a deliberate trade-off against the tiled
/// pipeline's usual one-frame-at-a-time memory bound. See the `align_model`
/// field docs on [`crate::PipelineParams`].
///
/// Returns one matrix per `paths` entry, in order. With the v2 alignment
/// architecture the matrix at index `0` (the reference frame) IS guaranteed
/// to be exactly identity by construction (the model left-composes every
/// frame's matrix with the inverse of frame 0's before returning — see
/// `docs/batchalign-v2-design.md` §3.5), so callers may rely on this.
///
/// # Errors
///
/// Returns [`StackerError`] if a frame fails to load/decode, if the model
/// fails to load or is not an alignment model
/// (`stacker_nn::ModelEntry::is_alignment`), or if inference itself fails.
#[cfg(feature = "nn")]
pub fn compute_neural_alignment(
    paths: &[std::path::PathBuf],
    preprocessing: &stacker_core::settings::PreprocessingSettings,
    model_name: Option<&str>,
    device_name: Option<&str>,
    frame_cache: &crate::cache::FrameCache,
) -> Result<Vec<stacker_align::Matrix3<f32>>, StackerError> {
    use stacker_core::preprocessing::preprocess_frame;

    let mut frames = Vec::with_capacity(paths.len());
    for path in paths {
        // Routed through the shared decode-once cache (see `crate::cache`'s
        // module docs) so a RAW frame already decoded by the pre-pass isn't
        // demosaiced a second time here.
        let dyn_img = frame_cache.load(path)?;
        let dyn_img = preprocess_frame(dyn_img, preprocessing);
        frames.push(crate::load::dynamic_to_planar(&dyn_img));
    }

    // Filter to alignment-capable models only — `discover_default_models`
    // mixes fusion and alignment checkpoints together, and an alignment
    // picker must never select a fusion checkpoint (see
    // `stacker_nn::ModelEntry::is_alignment`'s docs).
    let models: Vec<_> = stacker_nn::discover_default_models()
        .into_iter()
        .filter(stacker_nn::ModelEntry::is_alignment)
        .collect();
    let entry = match model_name {
        Some(name) => models.iter().find(|m| m.name == *name).ok_or_else(|| {
            StackerError::MathError(format!("alignment AI model '{name}' not found"))
        })?,
        None => models.first().ok_or_else(|| {
            StackerError::MathError(
                "no alignment AI models found in the models/ directory".to_owned(),
            )
        })?,
    };

    let device = match device_name {
        None => stacker_nn::available_devices()
            .into_iter()
            .next()
            .ok_or_else(|| StackerError::MathError("no inference backend built".to_owned()))?,
        Some("cpu") => stacker_nn::available_devices()
            .into_iter()
            .find(|d| matches!(d, stacker_nn::InferDevice::Cpu))
            .ok_or_else(|| StackerError::MathError("CPU backend not built".to_owned()))?,
        #[cfg(feature = "nn-gpu")]
        Some("gpu") => stacker_nn::available_devices()
            .into_iter()
            .find(|d| matches!(d, stacker_nn::InferDevice::Gpu))
            .ok_or_else(|| StackerError::MathError("GPU backend not available".to_owned()))?,
        #[cfg(not(feature = "nn-gpu"))]
        Some("gpu") => {
            return Err(StackerError::MathError(
                "GPU requested but built without --features nn-gpu".to_owned(),
            ));
        }
        Some(other) => {
            return Err(StackerError::MathError(format!(
                "invalid alignment device '{other}' (cpu|gpu)"
            )));
        }
    };

    let model = stacker_nn::LoadedAlignModel::load(entry, device)
        .map_err(|e| StackerError::MathError(format!("failed to load alignment AI model: {e}")))?;
    model
        .align(&frames)
        .map_err(|e| StackerError::MathError(format!("neural alignment failed: {e}")))
}
