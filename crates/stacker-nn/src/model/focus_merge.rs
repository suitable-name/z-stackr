use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::{sigmoid, tanh},
};

use crate::{
    model::{
        blocks::{ConvNormAct, DILATIONS, ResBlock, point_conv},
        size::ModelSize,
    },
    traits::{FusionModel, FusionStrategy, MergeStep},
};

/// Result of one pairwise-merge forward pass. All tensors `[N, C, H, W]`, f32,
/// linear light.
#[derive(Debug)]
pub struct MergeOutput<B: Backend> {
    /// Merged RGB image — `[N, 3, H, W]`.
    pub merged: Tensor<B, 4>,
    /// Per-pixel confidence — `[N, 1, H, W]` in 0..1.
    pub conf: Tensor<B, 4>,
    /// Soft blending map (alpha) — `[N, 1, H, W]` in 0..1. Exposed for
    /// inspection and auxiliary loss terms.
    pub alpha: Tensor<B, 4>,
}

/// Configuration for [`FocusMergeNet`].
///
/// Prefer [`FocusMergeNetConfig::from_size`] with a [`ModelSize`] preset; the
/// individual fields are exposed for fine-tuning and so a trained model's exact
/// geometry can be round-tripped through a sidecar manifest.
#[derive(Config, Debug)]
pub struct FocusMergeNetConfig {
    /// Feature-map width (channels). Must be divisible by `norm_groups`.
    #[config(default = 32)]
    pub width: usize,
    /// Number of dilated residual context blocks.
    #[config(default = 4)]
    pub depth: usize,
    /// `GroupNorm` group count. Must divide `width`.
    #[config(default = 8)]
    pub norm_groups: usize,
    /// Scale applied to the residual edge-refinement output (anti-halo/seam).
    #[config(default = 0.1)]
    pub refine_scale: f64,
}

impl FocusMergeNetConfig {
    /// Build the config for a named [`ModelSize`] preset.
    #[must_use]
    pub fn from_size(size: ModelSize) -> Self {
        let (width, depth) = size.dims();
        Self::new()
            .with_width(width)
            .with_depth(depth)
            .with_norm_groups(8)
    }
}

/// Fully-convolutional pairwise focus-merge network. See the module docs for
/// the architecture overview.
///
/// The `stem`, context stack, and sharpness branch are **weight-shared** between
/// the target and source frames (each module is applied to both), enforcing a
/// symmetric feature space in which the two frames are directly comparable.
#[derive(Module, Debug)]
pub struct FocusMergeNet<B: Backend> {
    /// Shared image encoder: 3 → width.
    stem: ConvNormAct<B>,
    /// Shared image encoder, second layer: width → width.
    stem2: ConvNormAct<B>,
    /// Shared dilated-residual context stack (applied per image).
    context: Vec<ResBlock<B>>,
    /// Shared sharpness branch body: width → width.
    sharp: ConvNormAct<B>,
    /// Shared sharpness branch head: width → 1.
    sharp_out: Conv2d<B>,
    /// Gating head body: (2·width + 3) → width.
    gate_in: ConvNormAct<B>,
    /// Gating head residual refinement.
    gate_res: ResBlock<B>,
    /// Gating head output: width → 1 (→ sigmoid = alpha).
    gate_out: Conv2d<B>,
    /// Refinement head body: 9 → width.
    refine_in: ConvNormAct<B>,
    /// Refinement head residual block.
    refine_res: ResBlock<B>,
    /// Refinement head output: width → 3 (→ tanh · scale).
    refine_out: Conv2d<B>,
    /// Confidence head body: 4 → width.
    conf_in: ConvNormAct<B>,
    /// Confidence head output: width → 1 (→ sigmoid).
    conf_out: Conv2d<B>,
    /// Residual-refinement scale.
    refine_scale: f64,
    /// Number of dilated context blocks (`context.len()`), cached at
    /// construction so [`FusionModel::receptive_field`] can compute
    /// its closed-form radius without walking `context`. Not a learnable
    /// parameter — `Module` derive treats it as a constant field.
    depth: usize,
}

impl FocusMergeNetConfig {
    /// Initialise a [`FocusMergeNet`] with this config on `device`.
    ///
    /// # Panics
    ///
    /// Panics (via `GroupNorm`) if `norm_groups` does not divide `width`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FocusMergeNet<B> {
        let w = self.width;
        let g = self.norm_groups;

        let context = (0..self.depth)
            .map(|i| ResBlock::new(w, g, DILATIONS[i & 3], device))
            .collect();

        FocusMergeNet {
            stem: ConvNormAct::new(3, w, g, device),
            stem2: ConvNormAct::new(w, w, g, device),
            context,
            sharp: ConvNormAct::new(w, w, g, device),
            sharp_out: point_conv(w, 1, device),
            gate_in: ConvNormAct::new(2 * w + 3, w, g, device),
            gate_res: ResBlock::new(w, g, 1, device),
            gate_out: point_conv(w, 1, device),
            refine_in: ConvNormAct::new(9, w, g, device),
            refine_res: ResBlock::new(w, g, 1, device),
            refine_out: Conv2dConfig::new([w, 3], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            conf_in: ConvNormAct::new(4, w, g, device),
            conf_out: point_conv(w, 1, device),
            refine_scale: self.refine_scale,
            depth: self.depth,
        }
    }
}

impl<B: Backend> FocusMergeNet<B> {
    /// Shared per-image encoder: stem + dilated context stack.
    fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = self.stem2.forward(self.stem.forward(x));
        for block in &self.context {
            f = block.forward(f);
        }
        f
    }

    /// Shared per-pixel sharpness estimate from encoded features.
    fn sharpness(&self, feat: Tensor<B, 4>) -> Tensor<B, 4> {
        self.sharp_out.forward(self.sharp.forward(feat))
    }

    /// Run one pairwise-merge forward pass.
    ///
    /// * `target`      – current composite `[N, 3, H, W]`, f32 linear 0..1.
    /// * `target_conf` – per-pixel confidence of `target`, `[N, 1, H, W]`.
    /// * `source`      – new source frame `[N, 3, H, W]`, f32 linear 0..1.
    pub fn forward(
        &self,
        target: Tensor<B, 4>,
        target_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
    ) -> MergeOutput<B> {
        // 1. Shared encoders (weight-shared across target & source).
        let feat_t = self.encode(target.clone());
        let feat_s = self.encode(source.clone());

        // 2. Shared sharpness estimates.
        let sharp_t = self.sharpness(feat_t.clone());
        let sharp_s = self.sharpness(feat_s.clone());

        // 3. Gating head → alpha.
        let gate = Tensor::cat(
            vec![
                feat_t,
                feat_s,
                sharp_t.clone(),
                sharp_s.clone(),
                target_conf.clone(),
            ],
            1,
        );
        let gate = self.gate_in.forward(gate);
        let gate = self.gate_res.forward(gate);
        let alpha = sigmoid(self.gate_out.forward(gate)); // [N,1,H,W]

        // 4. Pre-merge (alpha broadcasts over the 3 colour channels).
        let one_minus_alpha = alpha.clone().neg().add_scalar(1.0_f32);
        let merged_pre = alpha
            .clone()
            .mul(source.clone())
            .add(one_minus_alpha.mul(target.clone()));

        // 5. Residual edge-refinement (anti-halo / anti-seam).
        let refine = Tensor::cat(vec![merged_pre.clone(), target, source], 1); // [N,9,H,W]
        let refine = self.refine_in.forward(refine);
        let refine = self.refine_res.forward(refine);
        let refine = tanh(self.refine_out.forward(refine)).mul_scalar(self.refine_scale as f32);
        let merged = merged_pre.add(refine);

        // 6. Confidence head.
        let conf = Tensor::cat(vec![sharp_t, sharp_s, alpha.clone(), target_conf], 1); // [N,4,H,W]
        let conf = self.conf_in.forward(conf);
        let conf = sigmoid(self.conf_out.forward(conf)); // [N,1,H,W]

        MergeOutput {
            merged,
            conf,
            alpha,
        }
    }
}

impl<B: Backend> FusionModel<B> for FocusMergeNet<B> {
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::Pairwise
    }

    /// Delegates to [`FocusMergeNet::forward`], dropping the `alpha` soft
    /// selection map (an implementation detail of this architecture's gating
    /// head — see [`MergeStep`]'s docs) to produce the trait's minimal
    /// `merged` + `conf` contract.
    fn merge(
        &self,
        target: Tensor<B, 4>,
        target_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
    ) -> MergeStep<B> {
        let MergeOutput { merged, conf, .. } = self.forward(target, target_conf, source);
        MergeStep { merged, conf }
    }

    /// Closed-form receptive-field radius of the dilated-context
    /// architecture, in pixels.
    fn receptive_field(&self) -> usize {
        let dilation_sum: usize = (0..self.depth).map(|i| DILATIONS[i & 3]).sum();
        2 + 2 * dilation_sum
    }
}
