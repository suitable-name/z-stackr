//! `BatchMergeNet` — a fully-convolutional whole-stack focus-merge network.
//!
//! Where [`crate::model::FocusMergeNet`] folds a stack one frame at a time, this
//! network ingests all `S` frames of a (tile of a) stack at once: a shared
//! encoder gives per-frame features, a shared sharpness branch gives a per-frame
//! per-pixel sharpness, a gating head scores every frame, a **softmax over the
//! stack axis** turns the scores into blend weights, and a refinement head cleans
//! up the weighted composite from the max-pooled features.
//!
//! ```text
//!   frames[S] ─ shared encode ─► feat[S] ─┬─► sharpness[S] ─┐
//!                                          │                 ▼
//!                                          └──────────► gate head ─► softmax_S ─► α[S]
//!   merged_pre = Σ_S α·frame
//!   merged     = merged_pre + refine(merged_pre, max_S feat) · refine_scale
//! ```
//!
//! ## Properties
//!
//! * **Order-invariant**: no recurrence and no positional input, so permuting the
//!   frames permutes `α` and leaves the merged image unchanged (up to the
//!   rounding of the sum).
//! * **Tile-invariant**: size-preserving, no pooling over space, per-pixel
//!   [`crate::model::ChannelNorm`] — the receptive-field radius is the same
//!   `2·Σ dilations + 10` as the pairwise network's, and tiles overlapping by
//!   `2·R` blend seam-free (see [`crate::infer::fuse_batch_tiled`]).
//! * **Memory** is `S · width · tile_h · tile_w` activations: use smaller tiles
//!   than the 512 default for the `xl` / `xxl` presets and long stacks.

use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::{softmax, tanh},
};

use crate::{
    model::{
        blocks::{
            ConvNormAct, DILATIONS, ResBlock, fusion_radius_for_depth, max_over_stack, point_conv,
        },
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
    /// Channel groups of the per-pixel [`crate::model::ChannelNorm`]. Must divide
    /// `width`; `1` is the default and what every preset uses.
    #[config(default = 1)]
    pub norm_groups: usize,
    /// Scale applied to the residual edge-refinement output (anti-halo/seam).
    #[config(default = 0.1)]
    pub refine_scale: f64,
}

impl BatchMergeNetConfig {
    /// Build the config for a named [`ModelSize`] preset (same table as
    /// [`crate::model::FocusMergeNetConfig::from_size`]).
    #[must_use]
    pub fn from_size(size: ModelSize) -> Self {
        let (width, depth) = size.dims();
        Self::new()
            .with_width(width)
            .with_depth(depth)
            .with_norm_groups(1)
    }

    /// Receptive-field radius, in pixels, of a network built from this config:
    /// `2·Σ dilations + 10`.
    #[must_use]
    pub const fn receptive_field_radius(&self) -> usize {
        fusion_radius_for_depth(self.depth)
    }
}

/// Fully-convolutional batch focus-merge network. See the module docs.
///
/// The gating head sees `[features ‖ sharpness]` per frame, as the pairwise
/// network's does: raw features alone under-determine "which frame is in focus
/// HERE" at a depth discontinuity, and an explicit sharpness estimate is the
/// direct signal [`crate::loss::FocusBatchLoss`]'s sharpness and gate terms train
/// against.
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
    /// Gating head output: width → 1 (→ softmax over the stack axis = alpha).
    gate_out: Conv2d<B>,
    /// Refinement head body: (3 + width) → width.
    refine_in: ConvNormAct<B>,
    /// Refinement head residual block.
    refine_res: ResBlock<B>,
    /// Refinement head output: width → 3 (→ tanh · scale).
    refine_out: Conv2d<B>,
    /// Residual-refinement scale.
    refine_scale: f64,
}

impl BatchMergeNetConfig {
    /// Initialise a [`BatchMergeNet`] with this config on `device`.
    ///
    /// # Panics
    ///
    /// Panics (via [`crate::model::ChannelNorm::new`]) if `norm_groups` is zero or
    /// does not divide `width`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BatchMergeNet<B> {
        let w = self.width;
        let g = self.norm_groups;

        let context = (0..self.depth)
            .map(|i| ResBlock::new(w, g, DILATIONS[i % DILATIONS.len()], device))
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

    /// Receptive-field radius (pixels) of this exact network.
    #[must_use]
    pub const fn receptive_field_radius(&self) -> usize {
        fusion_radius_for_depth(self.context.len())
    }

    /// Fuse an entire stack at once, also returning the softmax gate weights.
    ///
    /// * `stack` – `[N, S, 3, H, W]`, f32 linear 0..1.
    ///
    /// Returns `(merged [N, 3, H, W], alpha [N, S, 1, H, W])`; `alpha` sums to 1
    /// over `S` at every pixel. The gate is exposed so the gate-supervision term
    /// of [`crate::loss::FocusBatchLoss`] can train it directly.
    #[must_use]
    pub fn forward_with_alpha(&self, stack: Tensor<B, 5>) -> (Tensor<B, 4>, Tensor<B, 5>) {
        let [n, s, _c, h, w_dim] = stack.dims();
        let stack_flat = stack.clone().reshape([n * s, 3, h, w_dim]);

        let feat_flat = self.encode(stack_flat);
        let [_ns, feat_c, _h, _w] = feat_flat.dims();

        // Shared per-frame sharpness estimate, fed into the gate alongside the
        // raw features.
        let sharp_flat = self.sharpness(feat_flat.clone());

        // [N*S, width + 1, H, W]
        let gate_input = Tensor::cat(vec![feat_flat.clone(), sharp_flat], 1);
        let gate = self.gate_in.forward(gate_input);
        let gate = self.gate_res.forward(gate);
        let scores_flat = self.gate_out.forward(gate);

        let scores = scores_flat.reshape([n, s, 1, h, w_dim]);
        let alpha = softmax(scores, 1);

        // Σ_S α·frame — alpha broadcasts over the 3 colour channels.
        let merged_pre = alpha.clone().mul(stack).sum_dim(1).squeeze::<4>(1);

        let feat = feat_flat.reshape([n, s, feat_c, h, w_dim]);
        let feat_pooled = max_over_stack(&feat);

        let refine_input = Tensor::cat(vec![merged_pre.clone(), feat_pooled], 1);
        let refine = self.refine_in.forward(refine_input);
        let refine = self.refine_res.forward(refine);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the f64 config scale is narrowed to the f32 tensor scalar"
        )]
        let refine = tanh(self.refine_out.forward(refine)).mul_scalar(self.refine_scale as f32);

        (merged_pre.add(refine), alpha)
    }

    /// Fuse an entire stack at once — [`Self::forward_with_alpha`] without the gate.
    ///
    /// * `stack` – `[N, S, 3, H, W]`, f32 linear 0..1. Returns `[N, 3, H, W]`.
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
        self.receptive_field_radius()
    }
}
