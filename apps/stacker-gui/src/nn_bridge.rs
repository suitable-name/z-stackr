// ── AI (neural) stacking bridge ─────────────────────────────────────────────

/// Fuse a stack with a trained neural model selected in the UI. Available only
/// in `nn` builds; the fallback below keeps the call site feature-agnostic.
///
/// # Errors
///
/// Returns `Err` if no matching (or fallback) fusion model can be found in
/// the models directory, if no inference backend matching `device_name` (or
/// any backend at all) is available, or if loading the model or running
/// fusion fails.
#[cfg(feature = "nn")]
pub fn nn_fuse_planar<F>(
    frames: &[stacker_core::image::PlanarImage<f32>],
    model_name: &str,
    device_name: &str,
    on_progress: F,
) -> Result<stacker_core::image::PlanarImage<f32>, String>
where
    F: FnMut(&stacker_core::image::PlanarImage<f32>),
{
    let models = stacker_nn::discover_default_models();
    let fusion_models: Vec<_> = models.iter().filter(|m| m.is_fusion()).collect();
    let entry = fusion_models
        .iter()
        .find(|m| m.name == model_name)
        .copied()
        .or_else(|| fusion_models.first().copied())
        .ok_or_else(|| "no fusion AI models found in the models/ folder".to_owned())?;
    let device = stacker_nn::available_devices()
        .into_iter()
        .find(|d| d.label().eq_ignore_ascii_case(device_name))
        .or_else(|| stacker_nn::available_devices().into_iter().next())
        .ok_or_else(|| "no inference backend available".to_owned())?;
    let model = stacker_nn::LoadedModel::load(entry, device).map_err(|e| e.to_string())?;
    model
        .fuse_with_progress(frames, stacker_nn::runtime::recommended_tile(), on_progress)
        .map_err(|e| e.to_string())
}

/// Compute a per-frame alignment matrix stack with a trained neural model
/// selected in the UI.
///
/// # Errors
///
/// Returns `Err` if no matching (or fallback) alignment model can be found
/// in the models directory, if no inference backend matching `device_name`
/// (or any backend at all) is available, or if loading the model or running
/// alignment fails.
#[cfg(feature = "nn")]
pub fn nn_align_planar(
    frames: &[stacker_core::image::PlanarImage<f32>],
    model_name: &str,
    device_name: &str,
) -> Result<Vec<nalgebra::Matrix3<f32>>, String> {
    let models = stacker_nn::discover_default_models();
    let alignment_models: Vec<_> = models.iter().filter(|m| m.is_alignment()).collect();
    let entry = alignment_models
        .iter()
        .find(|m| m.name == model_name)
        .copied()
        .or_else(|| alignment_models.first().copied())
        .ok_or_else(|| "no alignment AI models found in the models/ folder".to_owned())?;
    let device = stacker_nn::available_devices()
        .into_iter()
        .find(|d| d.label().eq_ignore_ascii_case(device_name))
        .or_else(|| stacker_nn::available_devices().into_iter().next())
        .ok_or_else(|| "no inference backend available".to_owned())?;
    let model = stacker_nn::LoadedAlignModel::load(entry, device).map_err(|e| e.to_string())?;
    model.align(frames).map_err(|e| e.to_string())
}

/// Stub for non-`nn` builds: AI stacking is unavailable.
///
/// # Errors
///
/// Always returns `Err` — this build was compiled without the `nn` feature,
/// so no fusion model can be loaded or run.
#[cfg(not(feature = "nn"))]
pub fn nn_fuse_planar<F>(
    _frames: &[stacker_core::image::PlanarImage<f32>],
    _model_name: &str,
    _device_name: &str,
    _on_progress: F,
) -> Result<stacker_core::image::PlanarImage<f32>, String>
where
    F: FnMut(&stacker_core::image::PlanarImage<f32>),
{
    Err("this build has no AI support (recompile with --features nn)".to_owned())
}
