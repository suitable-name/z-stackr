//! Pieces shared by the two alignment networks.
//!
//! These are [`crate::model::BatchAlignNet`] and [`crate::model::FusionAlignNet`]:
//! the strided shared-weight encoder, the local correlation volume, the
//! regression head and the bounded affine parameterisation.
//!
//! ## Architecture
//!
//! ```text
//!   frame [3,H,W] ─ stride-2 conv ─ stride-2 conv ─ res-blocks ─ stride-2 conv ─ res-blocks
//!                                                                   │   features at /8
//!                                              L2-normalise over channels (cosine similarity)
//!   corr  = local correlation of reference features with frame features  ((2r+1)² channels)
//!   head  = conv(corr ‖ ref ‖ frame) ─ res ─ stride-2 conv ─ res ─ global mean ─ MLP ─ 6 numbers
//!   M     = bounded affine from the 6 numbers (tanh-scaled translation, rotation, log scales, shear)
//! ```
//!
//! Every normalisation is the per-pixel [`crate::model::ChannelNorm`], so a
//! pixel's features never depend on the extent of the image the network is run
//! on. (The original `GroupNorm` took statistics over the whole `H × W`, which
//! differs between the 256 px crops the aligner is trained on and the 512 px
//! thumbnails it is run on.) The global spatial mean in the head is the one
//! deliberate global operation: it is how a translation in feature cells becomes
//! a single transform.
//!
//! The last linear layer is zero-initialised, so an untrained network predicts
//! the identity for every input.

use burn::{
    nn::{Gelu, Initializer, Linear, LinearConfig},
    prelude::*,
    tensor::activation::tanh,
};

use crate::model::blocks::{ConvNormAct, ConvNormActS, ResBlock};

/// Largest correlation radius accepted. The volume has `(2r + 1)²` channels, so
/// the cost grows quadratically; `16` already means 1089 channels.
pub const MAX_CORR_RADIUS: usize = 16;

/// Check an alignment network geometry, returning a human-readable reason when it
/// cannot be built.
///
/// `width` must be even and positive (the first stem convolution outputs
/// `width / 2` channels), `norm_groups` must divide both `width` and `width / 2`,
/// and `corr_radius` must not exceed [`MAX_CORR_RADIUS`].
///
/// # Errors
///
/// A description of the first violated constraint.
pub fn check_align_geometry(
    width: usize,
    norm_groups: usize,
    corr_radius: usize,
) -> Result<(), String> {
    if width == 0 || norm_groups == 0 {
        return Err(format!(
            "width ({width}) and norm_groups ({norm_groups}) must be positive"
        ));
    }
    if !width.is_multiple_of(2) {
        return Err(format!("width {width} must be even"));
    }
    if !width.is_multiple_of(norm_groups) || !(width / 2).is_multiple_of(norm_groups) {
        return Err(format!(
            "norm_groups {norm_groups} must divide both width {width} and width / 2"
        ));
    }
    if corr_radius > MAX_CORR_RADIUS {
        return Err(format!(
            "corr_radius {corr_radius} exceeds the supported maximum {MAX_CORR_RADIUS}"
        ));
    }
    Ok(())
}

/// L2-normalise a feature map over the channel dimension: `[B, C, H, W]` → same
/// shape, each pixel's channel vector scaled to unit length (`+1e-6` against
/// division by zero).
///
/// Normalising first makes the downstream [`local_correlation`] a cosine
/// similarity in `[-1, 1]`: bounded activations, and brightness changes between
/// frames barely move it.
pub(crate) fn l2_normalize<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let norm = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1)
        .sqrt()
        .add_scalar(1e-6);
    x.div(norm) // norm is [B,1,H,W]; broadcasts over C
}

/// Local correlation volume between `f_ref` and `f_frm`, both `[B, C, H, W]` and
/// already L2-normalised. Returns `[B, (2r+1)², H, W]`: channel `dy·(2r+1) + dx`
/// holds the channel-mean product of the reference feature at `(y, x)` with the
/// frame feature at `(y + dy - r, x + dx - r)` (zero outside the map).
///
/// Implemented as `(2r+1)²` padded slice-multiply-mean operations: each makes one
/// full-size temporary, which is fine at /8 of a 512 px thumbnail.
#[doc(hidden)]
#[must_use]
pub fn local_correlation<B: Backend>(
    f_ref: &Tensor<B, 4>,
    f_frm: Tensor<B, 4>,
    radius: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let [batch_size, num_channels, height, width] = f_ref.dims();
    let mut padded = Tensor::<B, 4>::zeros(
        [
            batch_size,
            num_channels,
            height + 2 * radius,
            width + 2 * radius,
        ],
        device,
    );
    padded = padded.slice_assign(
        [
            0..batch_size,
            0..num_channels,
            radius..radius + height,
            radius..radius + width,
        ],
        f_frm,
    );

    let side = 2 * radius + 1;
    let mut channels = Vec::with_capacity(side * side);
    for dy in 0..side {
        for dx in 0..side {
            let win = padded.clone().slice([
                0..batch_size,
                0..num_channels,
                dy..dy + height,
                dx..dx + width,
            ]);
            channels.push(f_ref.clone().mul(win).mean_dim(1)); // [B,1,H,W]
        }
    }
    Tensor::cat(channels, 1) // [B, (2r+1)^2, H, W]
}

/// Compose a bounded `[M, 3, 3]` affine batch from `M` rows of 6 raw (pre-`tanh`)
/// head outputs `raw: [M, 6]`.
///
/// Each raw output is squashed through a scaled `tanh`, then composed:
///
/// | raw | meaning | bound |
/// |---|---|---|
/// | 0, 1 | translation `tx`, `ty` (normalised units) | ±0.30 |
/// | 2 | rotation `θ` | ±0.10 rad |
/// | 3, 4 | log-scales `ln sx`, `ln sy` | ±0.08 |
/// | 5 | shear `k` | ±0.05 |
///
/// ```text
/// A = [ sx·cosθ   sx·cosθ·k − sy·sinθ ]      M = [ A  t ]
///     [ sx·sinθ   sx·sinθ·k + sy·cosθ ]          [ 0  1 ]
/// ```
///
/// so `det A = sx·sy ≥ e^-0.16 > 0` and the matrix can never collapse or
/// reflect. The bounds cap how wrong an early prediction can be and give the
/// optimiser similarly scaled directions to move in. They are **not**
/// independent hyper-parameters: widen them only together with the ranges of
/// `scripts/03_simulate_misalignment.py`. Returns `[M, 3, 3]`.
pub(crate) fn bounded_affine_from_raw6<B: Backend>(
    raw: Tensor<B, 2>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [m, _six] = raw.dims();
    let p0 = raw.clone().slice([0..m, 0..1]);
    let p1 = raw.clone().slice([0..m, 1..2]);
    let p2 = raw.clone().slice([0..m, 2..3]);
    let p3 = raw.clone().slice([0..m, 3..4]);
    let p4 = raw.clone().slice([0..m, 4..5]);
    let p5 = raw.slice([0..m, 5..6]);

    let tx = tanh(p0).mul_scalar(0.30_f32);
    let ty = tanh(p1).mul_scalar(0.30_f32);
    let theta = tanh(p2).mul_scalar(0.10_f32);
    let lsx = tanh(p3).mul_scalar(0.08_f32);
    let lsy = tanh(p4).mul_scalar(0.08_f32);
    let k = tanh(p5).mul_scalar(0.05_f32);

    let sx = lsx.exp();
    let sy = lsy.exp();
    let cos_t = theta.clone().cos();
    let sin_t = theta.sin();

    // `stretch_*` carry the x scale, `squeeze_*` the y scale.
    let stretch_cos = sx.clone().mul(cos_t.clone());
    let stretch_sin = sx.mul(sin_t.clone());
    let squeeze_cos = sy.clone().mul(cos_t);
    let squeeze_sin = sy.mul(sin_t);

    let a00 = stretch_cos.clone();
    let a01 = stretch_cos.mul(k.clone()).sub(squeeze_sin);
    let a10 = stretch_sin.clone();
    let a11 = stretch_sin.mul(k).add(squeeze_cos);

    let row1 = Tensor::cat(vec![a00, a01, tx], 1).unsqueeze_dim::<3>(1);
    let row2 = Tensor::cat(vec![a10, a11, ty], 1).unsqueeze_dim::<3>(1);

    let zeros = Tensor::<B, 3>::zeros([m, 1, 2], device);
    let ones = Tensor::<B, 3>::ones([m, 1, 1], device);
    let row3 = Tensor::cat(vec![zeros, ones], 2);

    Tensor::cat(vec![row1, row2, row3], 1) // [M, 3, 3]
}

/// The shared-weight encoder and the regression head of both alignment
/// networks. See the module docs.
#[derive(Module, Debug)]
pub(crate) struct AlignBackbone<B: Backend> {
    stem1: ConvNormActS<B>,
    stem2: ConvNormActS<B>,
    stage1: Vec<ResBlock<B>>,
    down: ConvNormActS<B>,
    stage2: Vec<ResBlock<B>>,
    head_in: ConvNormAct<B>,
    head_res1: ResBlock<B>,
    head_down: ConvNormActS<B>,
    head_res2: ResBlock<B>,
    fc1: Linear<B>,
    fc_act: Gelu,
    fc2: Linear<B>,
    corr_radius: usize,
}

impl<B: Backend> AlignBackbone<B> {
    /// Build the backbone.
    ///
    /// # Panics
    ///
    /// Panics when [`check_align_geometry`] rejects the geometry.
    pub(crate) fn new(
        width: usize,
        depth: usize,
        groups: usize,
        corr_radius: usize,
        device: &B::Device,
    ) -> Self {
        if let Err(reason) = check_align_geometry(width, groups, corr_radius) {
            panic!("invalid alignment network geometry: {reason}");
        }
        let w = width;
        let g = groups;
        let corr_channels = (2 * corr_radius + 1) * (2 * corr_radius + 1);

        Self {
            stem1: ConvNormActS::new(3, w / 2, g, device),
            stem2: ConvNormActS::new(w / 2, w, g, device),
            stage1: (0..depth).map(|_| ResBlock::new(w, g, 1, device)).collect(),
            down: ConvNormActS::new(w, w, g, device),
            stage2: (0..depth).map(|_| ResBlock::new(w, g, 1, device)).collect(),
            head_in: ConvNormAct::new(corr_channels + 2 * w, w, g, device),
            head_res1: ResBlock::new(w, g, 1, device),
            head_down: ConvNormActS::new(w, w, g, device),
            head_res2: ResBlock::new(w, g, 1, device),
            fc1: LinearConfig::new(w, w / 2).init(device),
            fc_act: Gelu::new(),
            fc2: LinearConfig::new(w / 2, 6)
                .with_initializer(Initializer::Zeros)
                .init(device),
            corr_radius,
        }
    }

    /// Encode `[M, 3, H, W]` frames to L2-normalised `[M, w, ceil(H/8), ceil(W/8)]`
    /// features.
    pub(crate) fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = self.stem2.forward(self.stem1.forward(x));
        for block in &self.stage1 {
            f = block.forward(f);
        }
        f = self.down.forward(f);
        for block in &self.stage2 {
            f = block.forward(f);
        }
        l2_normalize(f)
    }

    /// Compare reference and frame features (`[M, w, h8, w8]` each, already
    /// encoded) and regress the 6 raw affine parameters: `[M, 6]`.
    pub(crate) fn regress(
        &self,
        f_ref: Tensor<B, 4>,
        f_frm: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let corr = local_correlation(&f_ref, f_frm.clone(), self.corr_radius, device);
        let x = Tensor::cat(vec![corr, f_ref, f_frm], 1); // [M, (2r+1)^2 + 2w, h8, w8]

        let x = self.head_in.forward(x);
        let x = self.head_res1.forward(x);
        let x = self.head_down.forward(x);
        let x = self.head_res2.forward(x);

        // Global spatial mean: [M, w, h, w] -> [M, w].
        let x = x.mean_dim(3).squeeze::<3>(3).mean_dim(2).squeeze::<2>(2);

        let x = self.fc_act.forward(self.fc1.forward(x));
        self.fc2.forward(x) // [M, 6]
    }
}
