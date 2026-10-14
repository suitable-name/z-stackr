//! `FocusMergeNet` — a specialized, fully-convolutional pairwise focus-merge
//! network.
//!
//! The network ingests a running composite (`target` RGB + per-pixel
//! confidence) and a new `source` frame, and predicts an updated composite,
//! confidence, and the soft selection map used to blend them.
//!
//! ## Design goals
//!
//! * **Fully convolutional & size-preserving** — one model serves every
//!   resolution; 8K is handled by tiled inference, not a bigger model. Every
//!   convolution preserves `H × W` (Same padding, or dilation-matched explicit
//!   padding), and there is **no down/up-sampling and no global pooling**, so
//!   tiles are independent and seams blend cleanly.
//! * **Bounded, controllable receptive field** — medium-range context (needed to
//!   disambiguate which frame is in focus) is gathered by a stack of *dilated
//!   residual blocks* whose dilations cycle `1 → 2 → 4 → 8`. This grows the
//!   receptive field geometrically without strided pooling. The exact radius of
//!   the merged output is `R = 2·Σ dilations + 10` pixels (see
//!   [`receptive_field_radius`]); it dictates the tiling overlap (`2·R`, see
//!   [`crate::infer::TileConfig::for_size`]).
//! * **Tile-invariant normalisation** — every normalisation layer computes its
//!   statistics *per pixel across channels* ([`crate::model::ChannelNorm`], a `LayerNorm` over the
//!   channel axis, optionally split into groups). No statistic ever involves a
//!   spatial extent, a batch or a tile, so a pixel's output depends only on the
//!   input pixels inside its receptive field. This is what makes tiled inference
//!   agree with whole-image inference away from the tile edges. (The original
//!   `GroupNorm` normalised over the whole spatial extent of the tensor it was
//!   given, so training crops, GUI tiles, CLI tiles and the whole-image fast path
//!   all saw different statistics.)
//!
//! ## Receptive field
//!
//! Radius of one forward pass, counted along each axis from an output pixel:
//! stem + stem2 (2) + context blocks (2 convs each, `2·d` per block) + sharpness
//! conv (1) + gate-in conv (1) + gate residual block (2) + refinement-in conv (1)
//! + refinement residual block (2) + refinement-out conv (1) = `2·Σd + 10`.
//!
//! | preset | depth | `Σ dilations` | `R` (px) | overlap `2·R` |
//! |---|---:|---:|---:|---:|
//! | `xs`  | 2  | 3  | 16  | 32  |
//! | `s`   | 3  | 7  | 24  | 48  |
//! | `m`   | 4  | 15 | 40  | 80  |
//! | `l`   | 6  | 18 | 46  | 92  |
//! | `xl`  | 9  | 31 | 72  | 144 |
//! | `xxl` | 12 | 45 | 100 | 200 |
//!
//! Tiling is applied per merge step (each step's blended full-size output is the
//! next step's target), so a pixel's value after `n` merges formally depends on a
//! neighbourhood of `n·R`. Away from the blend bands every tile sees the same
//! inputs as the whole-image run, so the deviation stays confined to the bands.
//! * **Scalable capacity** — width (channels) and depth (number of context
//!   blocks) are configurable, exposed as the [`ModelSize`] presets `xs…xxl`.
//!
//! ## Pipeline
//!
//! ```text
//!   target(3) ─┐                          ┌─► sharpness_t(1) ─┐
//!              ├─ shared encode (stem +    │                  │
//!   source(3) ─┘   dilated res-blocks) ──► ┼─► sharpness_s(1) ─┤
//!                                          │                  ▼
//!                       target_conf(1) ───────────────► gating head ─► α
//!                                          │                  │
//!     merged_pre = α·source + (1-α)·target ◄──────────────────┘
//!     merged     = merged_pre + refine(merged_pre, target, source)
//!     conf       = confidence_head(sharp_t, sharp_s, α, target_conf)
//! ```

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
        blocks::{ConvNormAct, DILATIONS, ResBlock, fusion_radius_for_depth, point_conv},
        size::ModelSize,
    },
    traits::{FusionModel, FusionStrategy, MergeStep},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

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
    /// Number of channel groups the per-pixel normalisation is split into
    /// ([`crate::model::ChannelNorm`]). Must divide `width`. `1` (the default and the value every
    /// preset uses) normalises over all channels of a pixel; larger values give
    /// groups of only `width / norm_groups` channels, which is degenerate for the
    /// small presets (2 channels normalise to ±1).
    #[config(default = 1)]
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
            .with_norm_groups(1)
    }

    /// Receptive-field radius, in pixels, of the merged output of a network built
    /// from this config: `2·Σ dilations + 10`. One forward pass reads at most this
    /// many pixels on each side of an output pixel, so tiles that overlap by
    /// `2·R` (see [`crate::infer::TileConfig::for_radius`]) give every pixel at
    /// least one tile with its full context.
    #[must_use]
    pub const fn receptive_field_radius(&self) -> usize {
        fusion_radius_for_depth(self.depth)
    }
}

/// Receptive-field radius (pixels) of a [`ModelSize`] preset.
///
/// The single source of truth for the tiling apron (`runtime::recommended_tile`,
/// the default [`crate::infer::TileConfig`] and the README table all derive from
/// it). `R = 2·Σ dilations + 10`; see the module docs for the layer-by-layer count.
/// `xs` 16, `s` 24, `m` 40, `l` 46, `xl` 72, `xxl` 100.
#[must_use]
pub const fn receptive_field_radius(size: ModelSize) -> usize {
    let (_, depth) = size.dims();
    fusion_radius_for_depth(depth)
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

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
}

impl FocusMergeNetConfig {
    /// Initialise a [`FocusMergeNet`] with this config on `device`.
    ///
    /// # Panics
    ///
    /// Panics (via [`crate::model::ChannelNorm::new`]) if `norm_groups` is zero or does not
    /// divide `width`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FocusMergeNet<B> {
        let w = self.width;
        let g = self.norm_groups;

        let context = (0..self.depth)
            .map(|i| ResBlock::new(w, g, DILATIONS[i % DILATIONS.len()], device))
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

    /// Receptive-field radius (pixels) of this exact network: `2·Σ dilations + 10`
    /// over its actual number of context blocks.
    #[must_use]
    pub const fn receptive_field_radius(&self) -> usize {
        fusion_radius_for_depth(self.context.len())
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
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the f64 config scale is narrowed to the f32 tensor scalar"
        )]
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

    fn merge(
        &self,
        target: Tensor<B, 4>,
        target_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
    ) -> MergeStep<B> {
        let out = self.forward(target, target_conf, source);
        MergeStep {
            merged: out.merged,
            conf: out.conf,
        }
    }

    fn receptive_field(&self) -> usize {
        self.receptive_field_radius()
    }
}
