use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::tanh,
};

use crate::{
    model::{
        blocks::{ConvNormAct, DILATIONS, ResBlock, point_conv},
        size::ModelSize,
    },
    traits::{FusionModel, FusionStrategy},
};

/// Configuration for [`BatchMergeNet`].
#[derive(Config, Debug)]
pub struct BatchMergeNetConfig {
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

impl BatchMergeNetConfig {
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

/// Fully-convolutional batch focus-merge network.
///
/// Ingests an entire focus stack of `S` frames at once, applies a shared
/// encoder to extract per-frame features, a shared sharpness branch to
/// estimate per-frame per-pixel sharpness, a gating head + softmax over the
/// stack dimension to predict soft blending weights, and a max-pooled-feature
/// refinement head to composite and clean up the final image.
///
/// ### Sharpness branch — parity with `FocusMergeNet`
///
/// The gating head sees `[features ‖ sharpness]` per frame (rather than
/// features alone), mirroring [`FocusMergeNet`]'s design: raw encoder
/// features alone under-determine "which frame is in focus HERE" — two
/// frames can have similar feature-space activations near a depth
/// discontinuity while differing sharply in actual high-frequency content.
/// An explicit sharpness estimate gives the softmax gate a direct,
/// low-level signal to key on, which is exactly the signal
/// [`crate::loss::FocusBatchLoss`]'s sharpness-retention and gate-supervision
/// terms train against — without it, the gate has no cheap way to localise
/// the in-focus frame right at a depth edge, which manifests as softer
/// results and halo risk there (see the module docs' `BatchMergeNet`
/// section).
#[derive(Module, Debug)]
pub struct BatchMergeNet<B: Backend> {
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
    /// Gating head body: (width + 1) → width.
    gate_in: ConvNormAct<B>,
    /// Gating head residual refinement.
    gate_res: ResBlock<B>,
    /// Gating head output: width → 1 (→ softmax = alpha).
    gate_out: Conv2d<B>,
    /// Refinement head body: (3 + width) → width.
    refine_in: ConvNormAct<B>,
    /// Refinement head residual block.
    refine_res: ResBlock<B>,
    /// Refinement head output: width → 3 (→ tanh · scale).
    refine_out: Conv2d<B>,
    /// Residual-refinement scale.
    refine_scale: f64,
    /// Number of dilated context blocks, cached for `receptive_field`.
    depth: usize,
}

impl BatchMergeNetConfig {
    /// Initialise a [`BatchMergeNet`] with this config on `device`.
    ///
    /// # Panics
    ///
    /// Panics (via `GroupNorm`) if `norm_groups` does not divide `width`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BatchMergeNet<B> {
        let w = self.width;
        let g = self.norm_groups;

        let context = (0..self.depth)
            .map(|i| ResBlock::new(w, g, DILATIONS[i & 3], device))
            .collect();

        BatchMergeNet {
            stem: ConvNormAct::new(3, w, g, device),
            stem2: ConvNormAct::new(w, w, g, device),
            context,
            sharp: ConvNormAct::new(w, w, g, device),
            sharp_out: point_conv(w, 1, device),
            gate_in: ConvNormAct::new(w + 1, w, g, device),
            gate_res: ResBlock::new(w, g, 1, device),
            gate_out: point_conv(w, 1, device),
            refine_in: ConvNormAct::new(3 + w, w, g, device),
            refine_res: ResBlock::new(w, g, 1, device),
            refine_out: Conv2dConfig::new([w, 3], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            refine_scale: self.refine_scale,
            depth: self.depth,
        }
    }
}

impl<B: Backend> BatchMergeNet<B> {
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

    /// Fuse an entire stack of frames at once, also returning the softmax
    /// gate weights (`alpha`) used to combine them.
    ///
    /// * `stack` - `[N, S, 3, H, W]`, f32 linear 0..1.
    ///
    /// Returns `(merged [N, 3, H, W], alpha [N, S, 1, H, W])`. [`Self::forward`]
    /// and the [`FusionModel::fuse_batch`] trait method both delegate to this
    /// and drop `alpha` — an implementation detail of this architecture's
    /// softmax gating head, exposed here (like [`FocusMergeNet::forward`]'s
    /// `alpha` field and [`BatchAlignNet::forward_with_params`]'s raw
    /// parameters) purely so training-time auxiliary losses
    /// ([`crate::loss::FocusBatchLoss`]'s gate-supervision term) can
    /// supervise it directly without widening the trait's minimal contract.
    #[must_use]
    pub fn forward_with_alpha(&self, stack: Tensor<B, 5>) -> (Tensor<B, 4>, Tensor<B, 5>) {
        let [n, s, _c, h, w_dim] = stack.dims();
        let stack_flat = stack.clone().reshape([n * s, 3, h, w_dim]);

        let feat_flat = self.encode(stack_flat);
        let [_ns, feat_c, _h, _w] = feat_flat.dims();

        // Shared per-frame sharpness estimate, fed into the gate alongside
        // the raw features (see the type docs' "Sharpness branch" section).
        let sharp_flat = self.sharpness(feat_flat.clone());

        let gate_input = Tensor::cat(vec![feat_flat.clone(), sharp_flat], 1); // [N*S, width+1, H, W]
        let gate = self.gate_in.forward(gate_input);
        let gate = self.gate_res.forward(gate);
        let scores_flat = self.gate_out.forward(gate);

        let scores = scores_flat.reshape([n, s, 1, h, w_dim]);
        let alpha = burn::tensor::activation::softmax(scores, 1);

        let merged_pre_5d = alpha.clone().mul(stack).sum_dim(1);
        let merged_pre = merged_pre_5d.squeeze_dim(1);

        let feat = feat_flat.reshape([n, s, feat_c, h, w_dim]);
        let feat_pooled_5d = feat.max_dim(1);
        let feat_pooled = feat_pooled_5d.squeeze_dim(1);

        let refine_input = Tensor::cat(vec![merged_pre.clone(), feat_pooled], 1);
        let refine = self.refine_in.forward(refine_input);
        let refine = self.refine_res.forward(refine);
        let refine = tanh(self.refine_out.forward(refine)).mul_scalar(self.refine_scale as f32);

        (merged_pre.add(refine), alpha)
    }

    /// Fuse an entire stack of frames at once.
    ///
    /// * `stack` - `[N, S, 3, H, W]`, f32 linear 0..1.
    ///
    /// Delegates to [`Self::forward_with_alpha`], dropping the softmax gate
    /// weights — see that method's docs.
    #[must_use]
    pub fn forward(&self, stack: Tensor<B, 5>) -> Tensor<B, 4> {
        self.forward_with_alpha(stack).0
    }
}

impl<B: Backend> FusionModel<B> for BatchMergeNet<B> {
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::Batch
    }

    fn fuse_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4> {
        self.forward(stack)
    }

    fn receptive_field(&self) -> usize {
        let dilation_sum: usize = (0..self.depth).map(|i| DILATIONS[i & 3]).sum();
        2 + 2 * dilation_sum
    }
}
