//! Focus-stacking fusion algorithms: `Apex` (Laplacian pyramid) and `Relief` (SML + guided filter).
//!
//! Both `run_apex` and `run_relief` operate on a **slice of tile-sized
//! `PlanarImage<f32>`** rather than full-resolution frames.  The tiled
//! orchestration in `lib.rs` calls these once per tile, keeping the fusion
//! scratch memory proportional to tile area, not image area.

use stacker_algo::{
    apex::fuse::build_and_fuse_pyramids,
    relief::{
        fuse::{auto_contrast_threshold, compute_relief_smls, fuse_relief_with_mask},
        threshold::ReliefSettings,
    },
    strata::{StrataParams, fuse_strata},
};
use stacker_core::{error::StackerError, image::PlanarImage};

/// Supported fusion modes accepted by [`crate::PipelineParams::mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMode {
    Apex,
    Relief,
    /// Guided-filter soft-blend fusion (`stacker_algo::strata`) — see
    /// `docs/strata-fusion-design.md`.
    Strata,
    /// Learned neural pairwise-merge stacking (requires the `nn` feature).
    #[cfg(feature = "nn")]
    Ai,
}

impl std::str::FromStr for FusionMode {
    type Err = StackerError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "apex" => Ok(Self::Apex),
            "relief" => Ok(Self::Relief),
            "strata" => Ok(Self::Strata),
            #[cfg(feature = "nn")]
            "ai" | "nn" => Ok(Self::Ai),
            other => {
                #[cfg(feature = "nn")]
                let expected = "'apex', 'relief', 'strata', or 'ai'";
                #[cfg(not(feature = "nn"))]
                let expected = "'apex', 'relief', or 'strata'";
                Err(StackerError::AlignmentFailed(format!(
                    "unknown fusion mode '{other}'; expected {expected}"
                )))
            }
        }
    }
}

/// Single-pass neural fusion of one padded tile (all frames). The pipeline
/// already supplies apron context per tile, so the model's internal tiling
/// is disabled.
#[cfg(feature = "nn")]
pub fn nn_fuse_tile(
    model: &stacker_nn::LoadedModel,
    frame_tiles: &[PlanarImage<f32>],
) -> Result<PlanarImage<f32>, StackerError> {
    // A tile larger than any real crop forces the model's single-pass path.
    let cfg = stacker_nn::TileConfig {
        tile: usize::MAX,
        overlap: 0,
    };
    model
        .fuse(frame_tiles, cfg)
        .map_err(|e| StackerError::MathError(format!("AI fusion failed: {e}")))
}

/// Discover and load the AI model selected in `params` onto the chosen
/// device. Errors if no models are installed or the request can't be satisfied.
#[cfg(feature = "nn")]
pub fn load_nn_model(
    params: &crate::PipelineParams,
) -> Result<stacker_nn::LoadedModel, StackerError> {
    // `discover_default_models` returns every discovered kind mixed together
    // (fusion and alignment checkpoints share one models/ directory); filter
    // to fusion-capable entries only so an alignment-only (`BATCHALIGN_V2`)
    // checkpoint can never be selected here and fail later with a confusing
    // architecture-mismatch error from `LoadedModel::load`.
    let models: Vec<_> = stacker_nn::discover_default_models()
        .into_iter()
        .filter(stacker_nn::ModelEntry::is_fusion)
        .collect();
    if models.is_empty() {
        return Err(StackerError::MathError(
            "no fusion AI models found in the models/ directory".to_owned(),
        ));
    }
    let entry = match &params.model {
        Some(name) => models.iter().find(|m| m.name == *name).ok_or_else(|| {
            StackerError::MathError(format!("fusion AI model '{name}' not found"))
        })?,
        None => &models[0],
    };

    let device = select_device(params.device.as_deref())?;

    stacker_nn::LoadedModel::load(entry, device)
        .map_err(|e| StackerError::MathError(format!("failed to load AI model: {e}")))
}

/// Resolve the requested inference device against the compiled-in backends.
#[cfg(feature = "nn")]
fn select_device(request: Option<&str>) -> Result<stacker_nn::InferDevice, StackerError> {
    match request {
        None => stacker_nn::available_devices()
            .into_iter()
            .next()
            .ok_or_else(|| StackerError::MathError("no inference backend built".to_owned())),
        Some("cpu") => stacker_nn::available_devices()
            .into_iter()
            .find(|d| matches!(d, stacker_nn::InferDevice::Cpu))
            .ok_or_else(|| StackerError::MathError("CPU backend not built".to_owned())),
        #[cfg(feature = "nn-gpu")]
        Some("gpu") => stacker_nn::available_devices()
            .into_iter()
            .find(|d| matches!(d, stacker_nn::InferDevice::Gpu))
            .ok_or_else(|| StackerError::MathError("GPU backend not available".to_owned())),
        #[cfg(not(feature = "nn-gpu"))]
        Some("gpu") => Err(StackerError::MathError(
            "GPU requested but built without --features nn-gpu".to_owned(),
        )),
        Some(other) => Err(StackerError::MathError(format!(
            "invalid --device '{other}' (cpu|gpu)"
        ))),
    }
}

/// Run the `Apex` (Laplacian-pyramid) fusion pipeline on a tile slice.
///
/// `images` is a slice of padded tile images (all frames, same padded region).
/// Pyramid construction is parallelised across images with rayon.
pub fn run_apex(
    images: &[PlanarImage<f32>],
    settings: &stacker_core::settings::StackingSettings,
) -> PlanarImage<f32> {
    // Behavior-preserving defaults: luma-only metric, grit suppression enabled.
    // The GUI's in-RAM path passes real settings via its own call site.
    let fused = build_and_fuse_pyramids(
        images,
        settings.pyramid_levels as usize,
        settings.use_all_color_channels,
        settings.grit_suppression,
    );
    fused.reconstruct()
}

/// Run the `Relief` (SML + guided-filter) fusion pipeline on a tile slice.
///
/// `images` is a slice of padded tile images (all frames, same padded region).
///
/// # Global vs. per-tile threshold
///
/// When `abs_threshold` is `Some`, it was already resolved once from
/// **global** (whole-image, all-tile) SML statistics by the tiled pipeline's
/// pre-pass (see `run_pipeline`'s `Relief` pre-pass doc comment in
/// `stacker-pipeline`'s crate root) and is passed straight through as
/// [`ReliefSettings::absolute_threshold`], so this function's own
/// `otsu_threshold`/`contrast_pct` resolution is skipped entirely — every
/// tile ends up using the identical absolute SML value, eliminating
/// tile-to-tile threshold seams. When `abs_threshold` is `None` (non-tiled
/// callers, or callers that haven't run the pre-pass), the threshold is
/// resolved from this call's own tile-local data exactly as before.
pub fn run_relief(
    images: &[PlanarImage<f32>],
    settings: &stacker_core::settings::StackingSettings,
    abs_threshold: Option<f32>,
) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }

    let (smls, max_sml) = compute_relief_smls(images, settings.relief_estimation_radius as usize);

    let relief_settings = abs_threshold.map_or_else(
        || {
            let final_contrast_pct = if settings.relief_auto_detect {
                auto_contrast_threshold(&max_sml)
            } else {
                settings.relief_contrast_pct
            };

            ReliefSettings {
                est_radius: settings.relief_estimation_radius as usize,
                smooth_radius: settings.relief_smoothing_radius as usize,
                contrast_pct: final_contrast_pct,
                absolute_threshold: None,
            }
        },
        |t| ReliefSettings {
            est_radius: settings.relief_estimation_radius as usize,
            smooth_radius: settings.relief_smoothing_radius as usize,
            contrast_pct: settings.relief_contrast_pct,
            absolute_threshold: Some(t),
        },
    );

    if settings.relief_use_multigrid {
        stacker_algo::relief::fuse::fuse_relief_multigrid(images, &smls, &max_sml, &relief_settings)
    } else {
        fuse_relief_with_mask(images, &smls, &max_sml, &relief_settings)
    }
}

/// Run the `Strata` (guided-filter soft-blend) fusion pipeline on a tile
/// slice.
///
/// `images` is a slice of padded tile images (all frames, same padded
/// region) — Strata's own streaming discipline (see
/// `docs/strata-fusion-design.md` §2) applies within this call, on top of
/// the tiled pipeline's per-tile streaming.
pub fn run_strata(
    images: &[PlanarImage<f32>],
    settings: &stacker_core::settings::StackingSettings,
) -> PlanarImage<f32> {
    if images.is_empty() {
        return PlanarImage::new(0, 0);
    }
    let params = StrataParams {
        base_radius: settings.strata_base_radius as usize,
        detail_focus: settings.strata_detail_focus,
    };
    fuse_strata(images, &params)
}
