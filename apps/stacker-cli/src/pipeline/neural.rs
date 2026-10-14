//! Neural alignment for the out-of-core run (`alignment_mode = "neural"`, builds
//! with the `nn` feature).
//!
//! The alignment loop itself is the shared
//! `stacker_pipeline::tiled::align_into_store_with_model`; what is left here is
//! choosing and loading the model.
use stacker_core::{
    error::StackerError,
    settings::{AlignmentModeSetting, StackingSettings},
};
use stacker_pipeline::align::AlignModel;

use crate::{args::CliArgs, pipeline::fuse::select_device};

/// The alignment model a run needs: `Some` when the settings ask for the neural
/// alignment, loaded onto the device `--device`/`--no-gpu` name.
///
/// # Errors
/// [`StackerError::Config`] when there is no alignment model in the models folder,
/// `--align-model` names none that exists, or the device is not in the build.
pub fn align_model_for(
    args: &CliArgs,
    settings: &StackingSettings,
) -> Result<Option<AlignModel>, StackerError> {
    if settings.alignment_mode != AlignmentModeSetting::Neural {
        if args.align_model.is_some() {
            tracing::warn!("--align-model is ignored: alignment_mode is not \"neural\"");
        }
        return Ok(None);
    }
    let models = stacker_nn::discover_default_alignment_models();
    if models.is_empty() {
        return Err(StackerError::Config(
            "alignment_mode = \"neural\" needs an alignment model, but none was found in the \
             models/ directory"
                .to_owned(),
        ));
    }
    let entry = match &args.align_model {
        Some(name) => models
            .iter()
            .find(|m| m.name == *name)
            .ok_or_else(|| StackerError::Config(format!("alignment model '{name}' not found")))?,
        None => &models[0],
    };
    let device = select_device(args.inference_device()?)?;
    tracing::info!(model = %entry.name, device = device.label(), "neural alignment model");
    Ok(Some(AlignModel::new(entry.clone(), device)))
}
