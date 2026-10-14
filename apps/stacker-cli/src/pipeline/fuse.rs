//! The fusion modes accepted on the command line, and the neural mode's model
//! loading.
//!
//! The engines themselves (`Apex`, `Relief`, `Strata`) are the shared
//! `stacker_pipeline::tiled` fusions; `pipeline/mod.rs` runs them once per
//! requested [`Engine`] over one tile store.
use stacker_core::error::StackerError;
#[cfg(feature = "nn")]
use stacker_core::image::PlanarImage;
use stacker_pipeline::tiling::Method;

#[cfg(feature = "nn")]
use crate::args::DeviceRequest;

/// One fusion algorithm, run over the aligned stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Guided-filter soft blend.
    Strata,
    /// Depth-map fusion.
    Relief,
    /// Laplacian-pyramid fusion.
    Apex,
    /// The trained network (requires the `nn` feature).
    #[cfg(feature = "nn")]
    Ai,
}

impl Engine {
    /// The tile geometry and the fusion the engine needs.
    #[must_use]
    pub const fn method(self) -> Method {
        match self {
            Self::Strata => Method::Strata,
            Self::Relief => Method::Relief,
            Self::Apex => Method::Apex,
            #[cfg(feature = "nn")]
            Self::Ai => Method::Neural,
        }
    }

    /// The lower-case name: the `--mode` word, and the suffix of an output file
    /// when a run writes several.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Strata => "strata",
            Self::Relief => "relief",
            Self::Apex => "apex",
            #[cfg(feature = "nn")]
            Self::Ai => "ai",
        }
    }
}

/// Supported fusion modes accepted on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMode {
    /// Laplacian-pyramid fusion.
    Apex,
    /// Depth-map fusion.
    Relief,
    /// Guided-filter soft blend.
    Strata,
    /// `Relief` and `Apex`, one output file each.
    Both,
    /// `Strata`, `Relief` and `Apex`, one output file each.
    All,
    /// Learned neural stacking (requires the `nn` feature).
    #[cfg(feature = "nn")]
    Ai,
}

impl FusionMode {
    /// The engines the mode runs, in output order. One entry writes the file the
    /// user named; several write `<name>_<engine>.<ext>` each.
    #[must_use]
    pub const fn engines(self) -> &'static [Engine] {
        match self {
            Self::Apex => &[Engine::Apex],
            Self::Relief => &[Engine::Relief],
            Self::Strata => &[Engine::Strata],
            Self::Both => &[Engine::Relief, Engine::Apex],
            Self::All => &[Engine::Strata, Engine::Relief, Engine::Apex],
            #[cfg(feature = "nn")]
            Self::Ai => &[Engine::Ai],
        }
    }
}

/// The current name of an algorithm that was renamed (`pmax` is `apex`, `dmap` is
/// `relief`), compared case-insensitively; `None` for any other word.
#[must_use]
pub fn renamed_mode(word: &str) -> Option<&'static str> {
    match word.trim().to_ascii_lowercase().as_str() {
        "pmax" => Some("apex"),
        "dmap" => Some("relief"),
        _ => None,
    }
}

impl std::str::FromStr for FusionMode {
    type Err = StackerError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "apex" | "pmax" => Ok(Self::Apex),
            "relief" | "dmap" => Ok(Self::Relief),
            "strata" => Ok(Self::Strata),
            "both" => Ok(Self::Both),
            "all" => Ok(Self::All),
            #[cfg(feature = "nn")]
            "ai" | "nn" => Ok(Self::Ai),
            other => Err(StackerError::Config(format!(
                "unknown fusion mode '{other}'; expected 'apex', 'relief', 'strata', 'both' or \
                 'all'{}",
                if cfg!(feature = "nn") { " or 'ai'" } else { "" }
            ))),
        }
    }
}

/// A loaded AI model together with the tiling its own receptive field asks for.
#[cfg(feature = "nn")]
pub struct NnModel {
    /// The network on its backend, weights resident.
    pub model: stacker_nn::LoadedModel,
    /// `stacker_nn::runtime::recommended_tile_for` of the model's manifest: the
    /// overlap is `2·R` of this model, not the largest preset's.
    pub tile: stacker_nn::TileConfig,
}

/// Single-pass neural fusion of one padded tile (all frames). The CLI already
/// supplies apron context per tile, so the model's internal tiling is disabled.
///
/// The network's normalisation is per pixel, so a single pass over the padded
/// tile is equivalent to tiling it, away from the cut edges (the apron covers
/// the receptive field there). Only `tile` is overridden; the overlap stays the
/// model's recommended one, so lowering `tile` later gives a correctly
/// overlapped fold.
///
/// # Errors
/// The network's failure, as a [`StackerError::MathError`].
#[cfg(feature = "nn")]
pub fn nn_fuse_tile(
    nn: &NnModel,
    frame_tiles: &[PlanarImage<f32>],
) -> Result<PlanarImage<f32>, StackerError> {
    // A tile larger than any real crop forces the model's single-pass path.
    let cfg = stacker_nn::TileConfig {
        tile: usize::MAX,
        ..nn.tile
    };
    nn.model
        .fuse(frame_tiles, cfg)
        .map_err(|e| StackerError::MathError(format!("AI fusion failed: {e}")))
}

/// Discover and load the AI fusion model selected by the CLI args onto the chosen
/// device. Errors if no fusion models are installed or the request can't be
/// satisfied. Alignment models in the same folder are never offered.
///
/// # Errors
/// [`StackerError::Config`] for no model, an unknown name, an unavailable device
/// or a model that fails to load.
#[cfg(feature = "nn")]
pub fn load_nn_model(args: &crate::args::CliArgs) -> Result<NnModel, StackerError> {
    let models = stacker_nn::discover_default_fusion_models();
    if models.is_empty() {
        return Err(StackerError::Config(
            "no AI fusion models found in the models/ directory".to_owned(),
        ));
    }
    let entry = match &args.model {
        Some(name) => models
            .iter()
            .find(|m| m.name == *name)
            .ok_or_else(|| StackerError::Config(format!("AI model '{name}' not found")))?,
        None => &models[0],
    };

    let device = select_device(args.inference_device()?)?;

    let model = stacker_nn::LoadedModel::load(entry, device)
        .map_err(|e| StackerError::Config(format!("failed to load AI model: {e}")))?;
    Ok(NnModel {
        model,
        tile: stacker_nn::runtime::recommended_tile_for(entry),
    })
}

/// Resolve the requested inference device against the compiled-in backends.
///
/// # Errors
/// [`StackerError::Config`] when the backend is not part of the build.
#[cfg(feature = "nn")]
pub fn select_device(
    request: Option<DeviceRequest>,
) -> Result<stacker_nn::InferDevice, StackerError> {
    match request {
        None => stacker_nn::available_devices()
            .into_iter()
            .next()
            .ok_or_else(|| StackerError::Config("no inference backend built".to_owned())),
        Some(DeviceRequest::Cpu) => stacker_nn::available_devices()
            .into_iter()
            .find(|d| matches!(d, stacker_nn::InferDevice::Cpu))
            .ok_or_else(|| StackerError::Config("CPU backend not built".to_owned())),
        #[cfg(feature = "nn-gpu")]
        Some(DeviceRequest::Gpu) => stacker_nn::available_devices()
            .into_iter()
            .find(|d| matches!(d, stacker_nn::InferDevice::Gpu))
            .ok_or_else(|| StackerError::Config("GPU backend not available".to_owned())),
        #[cfg(not(feature = "nn-gpu"))]
        Some(DeviceRequest::Gpu) => Err(StackerError::Config(
            "GPU requested but built without --features nn-gpu".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_mode_is_a_config_error_with_a_readable_message() {
        let err = "bogus".parse::<FusionMode>().unwrap_err();
        assert!(matches!(err, StackerError::Config(_)), "{err:?}");
        let text = err.to_string();
        assert!(text.contains("unknown fusion mode 'bogus'"), "{text}");
        assert!(
            text.contains("'strata'") && text.contains("'all'"),
            "{text}"
        );
    }

    #[test]
    fn modes_are_case_insensitive() {
        assert_eq!("Apex".parse::<FusionMode>().unwrap(), FusionMode::Apex);
        assert_eq!("RELIEF".parse::<FusionMode>().unwrap(), FusionMode::Relief);
        assert_eq!("Strata".parse::<FusionMode>().unwrap(), FusionMode::Strata);
        assert_eq!("BOTH".parse::<FusionMode>().unwrap(), FusionMode::Both);
        assert_eq!("All".parse::<FusionMode>().unwrap(), FusionMode::All);
    }

    #[test]
    fn the_old_names_are_aliases() {
        assert_eq!("pmax".parse::<FusionMode>().unwrap(), FusionMode::Apex);
        assert_eq!("PMax".parse::<FusionMode>().unwrap(), FusionMode::Apex);
        assert_eq!("dmap".parse::<FusionMode>().unwrap(), FusionMode::Relief);
        assert_eq!("DMAP".parse::<FusionMode>().unwrap(), FusionMode::Relief);
        assert_eq!(renamed_mode("PMax"), Some("apex"));
        assert_eq!(renamed_mode("dmap"), Some("relief"));
        assert_eq!(renamed_mode("apex"), None);
        assert_eq!(renamed_mode("both"), None);
    }

    #[test]
    fn modes_run_their_engines_in_output_order() {
        assert_eq!(FusionMode::Apex.engines(), &[Engine::Apex]);
        assert_eq!(FusionMode::Strata.engines(), &[Engine::Strata]);
        assert_eq!(FusionMode::Both.engines(), &[Engine::Relief, Engine::Apex]);
        assert_eq!(
            FusionMode::All.engines(),
            &[Engine::Strata, Engine::Relief, Engine::Apex]
        );
    }

    #[test]
    fn engines_name_their_method_and_their_file_suffix() {
        assert_eq!(Engine::Strata.method(), Method::Strata);
        assert_eq!(Engine::Relief.method(), Method::Relief);
        assert_eq!(Engine::Apex.method(), Method::Apex);
        assert_eq!(Engine::Strata.label(), "strata");
        assert_eq!(Engine::Relief.label(), "relief");
        assert_eq!(Engine::Apex.label(), "apex");
    }

    /// Without the `nn` feature `ai` is not a mode.
    #[cfg(not(feature = "nn"))]
    #[test]
    fn ai_is_unknown_without_the_nn_feature() {
        assert!("ai".parse::<FusionMode>().is_err());
        assert!("nn".parse::<FusionMode>().is_err());
    }

    /// `--device GPU` / `CPU` parse like `--mode Apex` does and reach the
    /// backend selection (the CPU request succeeds in every `nn` build).
    #[cfg(feature = "nn")]
    #[test]
    fn device_request_is_case_insensitive_end_to_end() {
        use clap::Parser;

        let parse = |device: &str| {
            crate::args::CliArgs::try_parse_from([
                "app",
                "--input-dir",
                "in",
                "--output-file",
                "out.png",
                "--mode",
                "ai",
                "--tile-size",
                "128",
                "--device",
                device,
            ])
            .expect("device spelled in any case")
        };
        assert_eq!(parse("GPU").device, Some(DeviceRequest::Gpu));
        let cpu = parse("CPU");
        assert_eq!(cpu.device, Some(DeviceRequest::Cpu));
        assert!(select_device(cpu.device).is_ok());
    }
}
