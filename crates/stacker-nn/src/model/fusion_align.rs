//! `FusionAlignNet` — streaming pairwise alignment (architecture tag
//! `fusionalign-v1`).
//!
//! The two-input twin of [`crate::model::BatchAlignNet`]: the same shared-weight
//! strided encoder, L2-normalised local correlation and bounded geometric head
//! (see [`crate::model::align_common`]), but it registers **one frame against a
//! reference per call**, so a caller can stream a stack with O(1) memory in the
//! number of frames: the reference thumbnail plus the frame in flight.
//!
//! It deliberately has no frame-0 normalisation. `BatchAlignNet` forces frame 0 to
//! the exact identity because frame 0 *is* the reference living inside the same
//! batched call, so the extra left-composition is nearly free. Here the `reference`
//! argument already is the reference; calling `align_pair(reference, reference)` a
//! second time per frame just to force an exact identity would double the encoder
//! cost and defeat the architecture's purpose. Identity on `reference == frame`
//! is left to the zero-initialised head (identity at step 0) and to training data
//! that contains identity pairs. The matrix maps reference coordinates to the
//! frame's coordinates, in the normalised coordinates of the image the network saw.

use burn::prelude::*;

use crate::{
    model::{
        align_common::{AlignBackbone, bounded_affine_from_raw6},
        size::ModelSize,
    },
    traits::PairAlignmentModel,
};

/// Configuration for [`FusionAlignNet`].
///
/// The same knobs as [`crate::model::BatchAlignNetConfig`] — the two architectures
/// share one encoder/correlation/head shape. Kept as its own type so they can
/// diverge independently without leaking into each other's manifest plumbing.
#[derive(Config, Debug)]
pub struct FusionAlignNetConfig {
    /// Feature-map width (channels). Must be even, and divisible by
    /// `norm_groups` together with `width / 2`.
    #[config(default = 64)]
    pub width: usize,
    /// Residual blocks in each of the two encoder stages.
    #[config(default = 2)]
    pub depth: usize,
    /// Channel groups of the per-pixel [`crate::model::ChannelNorm`]; `1` is the
    /// default and what every preset uses.
    #[config(default = 1)]
    pub norm_groups: usize,
    /// Correlation search radius in feature cells at the `/8` encoder scale.
    #[config(default = 4)]
    pub corr_radius: usize,
}

impl FusionAlignNetConfig {
    /// Build the config for a named [`ModelSize`] preset. Uses the same
    /// `(width, depth)` table as [`crate::model::BatchAlignNetConfig::from_size`];
    /// change it independently if training results ever say the two should differ.
    #[must_use]
    pub fn from_size(size: ModelSize) -> Self {
        let (width, depth) = size.align_dims();
        Self::new()
            .with_width(width)
            .with_depth(depth)
            .with_norm_groups(1)
    }
}

/// Pairwise alignment network. See the module docs.
#[derive(Module, Debug)]
pub struct FusionAlignNet<B: Backend> {
    backbone: AlignBackbone<B>,
}

impl FusionAlignNetConfig {
    /// Initialise a [`FusionAlignNet`] with this config on `device`. The last
    /// linear layer is zero-initialised: an untrained network predicts the
    /// identity.
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`crate::model::BatchAlignNetConfig::init`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> FusionAlignNet<B> {
        FusionAlignNet {
            backbone: AlignBackbone::new(
                self.width,
                self.depth,
                self.norm_groups,
                self.corr_radius,
                device,
            ),
        }
    }
}

impl<B: Backend> FusionAlignNet<B> {
    /// Run the forward pass for one reference/frame pair, returning the composed
    /// `[N, 3, 3]` matrix (the [`PairAlignmentModel`] output) and the raw
    /// pre-`tanh` head outputs `[N, 6]` (for the trainer's delta regulariser).
    ///
    /// * `reference`, `frame` – `[N, 3, H, W]`, linear light, `H, W ≥ 8`, same size.
    ///
    /// Reference and frame are concatenated into one `2N` pseudo-batch and run
    /// through the shared encoder in a single call, then split.
    #[must_use]
    pub fn forward_with_params(
        &self,
        reference: Tensor<B, 4>,
        frame: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [n, _c, _h, _w] = reference.dims();
        let device = reference.device();

        let paired = Tensor::cat(vec![reference, frame], 0); // [2N, 3, H, W]
        let feat = self.backbone.encode(paired);
        let [_, cw, h8, w8] = feat.dims();

        let f_ref = feat.clone().slice([0..n, 0..cw, 0..h8, 0..w8]);
        let f_frm = feat.slice([n..2 * n, 0..cw, 0..h8, 0..w8]);

        let raw = self.backbone.regress(f_ref, f_frm, &device); // [N, 6]
        let m = bounded_affine_from_raw6::<B>(raw.clone(), &device); // [N, 3, 3]
        (m, raw)
    }
}

impl<B: Backend> PairAlignmentModel<B> for FusionAlignNet<B> {
    fn align_pair(&self, reference: Tensor<B, 4>, frame: Tensor<B, 4>) -> Tensor<B, 3> {
        self.forward_with_params(reference, frame).0
    }
}
