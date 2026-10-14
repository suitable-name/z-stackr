//! `BatchAlignNet` — whole-stack affine alignment (architecture tag
//! `batchalign-v2`).
//!
//! Predicts one `3 × 3` affine per frame of a stack, in the normalised
//! `[-1, 1]²` coordinates of the image the network saw, mapping **reference
//! coordinates to the frame's coordinates** (`p_frame = M · p_reference`). The
//! encoder, correlation volume and bounded head are shared with
//! [`crate::model::FusionAlignNet`] (see [`crate::model::align_common`]); this
//! network adds the whole-stack batching and the **frame-0 normalisation**: every
//! matrix is left-composed with the analytic inverse of matrix 0
//! (`M_i' = M_0⁻¹ · M_i`), so frame 0 is exactly the identity by construction and
//! the one global degree of freedom the corner loss cannot pin down (the whole
//! stack drifting together) is removed. The inverse is computed analytically, not
//! with a generic tensor inverse, so the whole operation stays differentiable
//! with no CPU round trip; `det M_0 = sx·sy ≥ e^-0.16 > 0`, so it never needs a
//! guard.
//!
//! Memory is `S` times the single-frame cost of the encoder (the frames are
//! flattened into one `N·S` batch).

use burn::prelude::*;

use crate::{
    model::{
        align_common::{AlignBackbone, bounded_affine_from_raw6},
        size::ModelSize,
    },
    traits::BatchAlignmentModel,
};

pub use crate::model::align_common::local_correlation;

/// Configuration for [`BatchAlignNet`].
#[derive(Config, Debug)]
pub struct BatchAlignNetConfig {
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
    /// Correlation search radius in feature cells at the `/8` encoder scale: the
    /// network can see displacements up to `radius · 8` thumbnail pixels
    /// directly (larger ones only through the bounded head's global pooling).
    /// The volume has `(2r + 1)²` channels.
    #[config(default = 4)]
    pub corr_radius: usize,
}

impl BatchAlignNetConfig {
    /// Build the config for a named [`ModelSize`] preset:
    /// `xs` (32, 1), `s` (48, 1), `m` (64, 2), `l` (96, 2), `xl` (128, 3),
    /// `xxl` (160, 3) as `(width, depth)`.
    #[must_use]
    pub fn from_size(size: ModelSize) -> Self {
        let (width, depth) = size.align_dims();
        Self::new()
            .with_width(width)
            .with_depth(depth)
            .with_norm_groups(1)
    }
}

/// Batch alignment network: a shared strided encoder, an explicit
/// local-correlation comparison against frame 0, a bounded geometric head and a
/// differentiable frame-0 normalisation. See the module docs.
#[derive(Module, Debug)]
pub struct BatchAlignNet<B: Backend> {
    backbone: AlignBackbone<B>,
}

impl BatchAlignNetConfig {
    /// Initialise a [`BatchAlignNet`] with this config on `device`. The last
    /// linear layer is zero-initialised: an untrained network predicts the
    /// identity.
    ///
    /// # Panics
    ///
    /// Panics if `width` is zero or odd, if `norm_groups` is zero or does not
    /// divide both `width` and `width / 2`, or if `corr_radius` exceeds
    /// [`crate::model::align_common::MAX_CORR_RADIUS`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> BatchAlignNet<B> {
        BatchAlignNet {
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

impl<B: Backend> BatchAlignNet<B> {
    /// Run the full forward pass, returning the composed `[N, S, 3, 3]` matrices
    /// (the [`BatchAlignmentModel`] output) and the raw pre-`tanh` head outputs
    /// `[N, S, 6]`.
    ///
    /// * `stack` – `[N, S, 3, H, W]`, linear light, `H, W ≥ 8`.
    ///
    /// The raw parameters are exposed so the trainer can apply the small delta
    /// regulariser (`1e-4 · mean(raw²)`) that keeps the head out of `tanh`
    /// saturation.
    #[must_use]
    pub fn forward_with_params(&self, stack: Tensor<B, 5>) -> (Tensor<B, 4>, Tensor<B, 3>) {
        let [batch_size, num_frames, _c, height, width] = stack.dims();
        let device = stack.device();
        let stack_flat = stack.reshape([batch_size * num_frames, 3, height, width]);

        let feat = self.backbone.encode(stack_flat);
        let [_, cw, h8, w8] = feat.dims();
        let feat_5d = feat.reshape([batch_size, num_frames, cw, h8, w8]);

        // Frame 0's features, repeated for every frame of the stack.
        let f_ref = feat_5d
            .clone()
            .slice([0..batch_size, 0..1, 0..cw, 0..h8, 0..w8])
            .repeat_dim(1, num_frames)
            .reshape([batch_size * num_frames, cw, h8, w8]);
        let f_frm = feat_5d.reshape([batch_size * num_frames, cw, h8, w8]);

        let raw = self.backbone.regress(f_ref, f_frm, &device); // [N*S, 6]
        let pre_tanh_params = raw.clone().reshape([batch_size, num_frames, 6]);

        let affine_matrices =
            bounded_affine_from_raw6::<B>(raw, &device).reshape([batch_size, num_frames, 3, 3]);

        // Reference normalisation: M_i' = M_0^-1 · M_i.
        let m0 = affine_matrices
            .clone()
            .slice([0..batch_size, 0..1, 0..3, 0..3]); // [N,1,3,3]
        let m0_inv = affine_inverse(&m0, &device); // [N,1,3,3]
        let m0_inv_bc = m0_inv.repeat_dim(1, num_frames); // [N,S,3,3]
        let m_normalised = m0_inv_bc.matmul(affine_matrices);

        (m_normalised, pre_tanh_params)
    }
}

/// Analytic inverse of a batch of bounded affine matrices `[N, 1, 3, 3]` (last row
/// `[0, 0, 1]`, positive determinant — see `bounded_affine_from_raw6`).
fn affine_inverse<B: Backend>(m: &Tensor<B, 4>, device: &B::Device) -> Tensor<B, 4> {
    let [n, _, _, _] = m.dims();
    let at = |r: usize, c: usize| m.clone().slice([0..n, 0..1, r..r + 1, c..c + 1]);
    let (m00, m01, m02) = (at(0, 0), at(0, 1), at(0, 2));
    let (m10, m11, m12) = (at(1, 0), at(1, 1), at(1, 2));

    // [N,1,1,1]
    let det = m00
        .clone()
        .mul(m11.clone())
        .sub(m01.clone().mul(m10.clone()));
    let inv_det = det.recip();

    let inv00 = m11.clone().mul(inv_det.clone());
    let inv01 = m01.clone().neg().mul(inv_det.clone());
    let inv02 = m01
        .mul(m12.clone())
        .sub(m02.clone().mul(m11))
        .mul(inv_det.clone());
    let inv10 = m10.clone().neg().mul(inv_det.clone());
    let inv11 = m00.clone().mul(inv_det.clone());
    let inv12 = m02.mul(m10).sub(m00.mul(m12)).mul(inv_det);

    let row1 = Tensor::cat(vec![inv00, inv01, inv02], 3);
    let row2 = Tensor::cat(vec![inv10, inv11, inv12], 3);
    let row3 = Tensor::cat(
        vec![
            Tensor::<B, 4>::zeros([n, 1, 1, 2], device),
            Tensor::<B, 4>::ones([n, 1, 1, 1], device),
        ],
        3,
    );
    Tensor::cat(vec![row1, row2, row3], 2) // [N,1,3,3]
}

impl<B: Backend> BatchAlignmentModel<B> for BatchAlignNet<B> {
    fn align_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4> {
        self.forward_with_params(stack).0
    }
}
