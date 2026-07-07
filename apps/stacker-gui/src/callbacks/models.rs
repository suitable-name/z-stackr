use crate::App;

/// Populates the algorithm list plus (in `nn` builds) the AI fusion-model
/// picker, the neural alignment-model picker, and the inference-device
/// picker.
///
/// Also sets the GPU-availability flag in `gpu`-featured builds.
///
/// This isn't a Slint `.on_*` callback registration like the other
/// `callbacks::*::wire` functions — it's one-time UI population that must
/// run before the user can interact with the algorithm/model pickers — but
/// it lives here because it's the nn/gpu-gated model-discovery counterpart
/// to the rest of the callback wiring, and gating the whole file with
/// `#[cfg(feature = "nn")]`-style conditionals inside keeps all model/device
/// picker logic in one place.
pub fn wire(app: &App) {
    // ── AI model discovery (populates the algorithm list + model picker) ──────
    {
        use slint::{ModelRc, SharedString, VecModel};
        #[allow(unused_mut)]
        let mut algos: Vec<SharedString> = vec![
            "Relief (Depth Map)".into(),
            "Apex (Laplacian)".into(),
            "Strata (Guided Fusion)".into(),
            "All Three (Strata, Relief & Apex)".into(),
        ];
        #[cfg(feature = "nn")]
        {
            let models = stacker_nn::discover_default_models();
            // `discover_default_models` returns every discovered kind mixed
            // together (fusion and alignment checkpoints share one models/
            // folder) — each picker MUST filter to the kind it actually
            // loads, or a mismatched selection fails at load time with an
            // architecture-mismatch error. Fusion uses `LoadedModel` (via
            // `ModelEntry::is_fusion`); alignment uses `LoadedAlignModel`
            // (via `ModelEntry::is_alignment`).
            let fusion_model_names: Vec<SharedString> = models
                .iter()
                .filter(|m| m.is_fusion())
                .map(|m| SharedString::from(m.name.as_str()))
                .collect();
            let alignment_model_names: Vec<SharedString> = models
                .iter()
                .filter(|m| m.is_alignment())
                .map(|m| SharedString::from(m.name.as_str()))
                .collect();
            let devices = stacker_nn::available_devices();
            let device_names: Vec<SharedString> = devices
                .iter()
                .map(|d| SharedString::from(d.label()))
                .collect();

            algos.push("AI Model (Neural)".into());
            app.set_ai_models_available(!fusion_model_names.is_empty());
            app.set_ai_model_names(ModelRc::new(VecModel::from(fusion_model_names)));
            app.set_alignment_models_available(!alignment_model_names.is_empty());
            app.set_alignment_model_names(ModelRc::new(VecModel::from(alignment_model_names)));
            app.set_ai_device_names(ModelRc::new(VecModel::from(device_names)));
            app.set_ai_show_device(devices.len() > 1);

            let align_options: Vec<SharedString> = vec![
                "Affine".into(),
                "Translation".into(),
                "Registration".into(),
                "None".into(),
                "Neural".into(),
            ];
            app.set_align_options(ModelRc::new(VecModel::from(align_options)));

            // NN build capability: gates settings UI that only makes sense
            // when Neural alignment is selectable at all (currently the
            // "Refine Neural alignment classically" row). Non-`nn` builds
            // never set this, so it keeps its `false` `.slint` default and
            // the row stays hidden.
            app.set_nn_available(true);
        }
        app.set_algo_options(ModelRc::new(VecModel::from(algos)));
    }

    // ── GPU availability (only meaningful in `gpu`-featured builds) ───────
    // The "Use GPU acceleration" checkbox (`gpu-available`/`use-gpu` in
    // app.slint) is only shown when this is `true`. `stacker_core::gpu::
    // context()` performs the actual (possibly first-ever) adapter probe as
    // a side effect, mirroring the AI-model-discovery block above: cheap
    // enough to do once at startup, and it lets the checkbox reflect
    // whether a GPU is genuinely usable rather than just "this binary was
    // compiled with the feature". Non-`gpu` builds never set this property,
    // so it keeps its `false` `.slint` default and the checkbox stays
    // hidden.
    #[cfg(feature = "gpu")]
    app.set_gpu_available(stacker_core::gpu::context().is_some());
}
