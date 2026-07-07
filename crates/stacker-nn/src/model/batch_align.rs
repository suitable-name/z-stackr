use crate::model::{
    blocks::{ConvNormAct, ResBlock},
    size::ModelSize,
};
use burn::{
    nn::{
        Gelu, GroupNorm, GroupNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::tanh,
};

/// A private stride-2 counterpart to [`ConvNormAct`] (`Conv3×3(stride 2,
/// pad 1)` → `GroupNorm` → `GELU`), used only by the v2 alignment encoder's
/// downsampling stages. Kept separate from the shared `ConvNormAct` (rather
/// than adding a `stride` parameter to it) so the fusion networks' Same-
/// padding, size-preserving contract can never be accidentally perturbed by
/// an alignment-only change.
#[derive(Module, Debug)]
pub(crate) struct ConvNormActS<B: Backend> {
    conv: Conv2d<B>,
    norm: GroupNorm<B>,
    act: Gelu,
}

impl<B: Backend> ConvNormActS<B> {
    pub(crate) fn new(c_in: usize, c_out: usize, groups: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            norm: GroupNormConfig::new(groups, c_out).init(device),
            act: Gelu::new(),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        self.act.forward(x)
    }
}

/// L2-normalise a feature map over the channel dimension: `[B,C,H,W] →` same
/// shape, each spatial position's channel vector scaled to unit length
/// (`+1e-6` to avoid division by zero).
///
/// *Why (`docs/batchalign-v2-design.md` §3.3):* normalising first makes the
/// downstream [`local_correlation`] a cosine similarity in `[-1, 1]` —
/// bounded activations, no exploding values, and brightness changes between
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

/// Local correlation volume between `f_ref` and `f_frm`, both `[B, C, H, W]`
/// and already L2-normalised. Returns `[B, (2r+1)^2, H, W]`.
#[doc(hidden)]
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

    let mut channels = Vec::with_capacity((2 * radius + 1) * (2 * radius + 1));
    for dy in 0..=(2 * radius) {
        for dx in 0..=(2 * radius) {
            let win = padded.clone().slice([
                0..batch_size,
                0..num_channels,
                dy..dy + height,
                dx..dx + width,
            ]);
            let corr = f_ref.clone().mul(win).mean_dim(1); // [B,1,H,W]
            channels.push(corr);
        }
    }
    Tensor::cat(channels, 1) // [B, (2r+1)^2, H, W]
}

/// Compose a bounded `[M, 3, 3]` affine-matrix batch from `M` rows of 6 raw
/// (pre-`tanh`) head outputs `raw: [M, 6]` (`M` is any flattened batch size —
/// `N` for a single pair, `N*S` for a whole stack).
///
/// Shared by [`BatchAlignNet`] (which calls this after flattening its
/// `[N,S,6]` head output to `[N*S,6]`) and
/// [`crate::model::fusion_align::FusionAlignNet`] (which calls it directly on
/// its `[N,6]` head output) — factored out here rather than duplicated
/// because the bounded geometric parameterisation
/// (`docs/batchalign-v2-design.md` §3.4) is architecture-agnostic tensor math
/// once the input is flattened to `[M,6]`; only the caller's reshape
/// boundary differs.
///
/// See [`BatchAlignNet::forward_with_params`]'s inline comment for the full
/// rationale (bounded `tanh` scaling, why each constant is fixed, why the
/// determinant can never vanish). Returns `[M, 3, 3]`.
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

    // Bounded geometric parameterisation (`docs/batchalign-v2-design.md`
    // §3.4) — see `BatchAlignNet::forward_with_params`'s inline comment for
    // the full rationale; identical here, just operating on `[M, 1]` slices
    // instead of `[N, S, 1]`.
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

    let sx_cos = sx.clone().mul(cos_t.clone());
    let sx_sin = sx.mul(sin_t.clone());
    let sy_cos = sy.clone().mul(cos_t);
    let sy_sin = sy.mul(sin_t);

    let a00 = sx_cos.clone();
    let a01 = sx_cos.mul(k.clone()).sub(sy_sin);
    let a10 = sx_sin.clone();
    let a11 = sx_sin.mul(k).add(sy_cos);

    let row1 = Tensor::cat(vec![a00, a01, tx], 1).unsqueeze_dim::<3>(1);
    let row2 = Tensor::cat(vec![a10, a11, ty], 1).unsqueeze_dim::<3>(1);

    let zeros = Tensor::<B, 3>::zeros([m, 1, 2], device);
    let ones = Tensor::<B, 3>::ones([m, 1, 1], device);
    let row3 = Tensor::cat(vec![zeros, ones], 2);

    Tensor::cat(vec![row1, row2, row3], 1) // [M, 3, 3]
}

/// Configuration for [`BatchAlignNet`] (v2).
#[derive(Config, Debug)]
pub struct BatchAlignNetConfig {
    #[config(default = 64)]
    pub width: usize,
    #[config(default = 2)]
    pub depth: usize,
    #[config(default = 8)]
    pub norm_groups: usize,
    #[config(default = 4)]
    pub corr_radius: usize,
}

impl BatchAlignNetConfig {
    #[must_use]
    pub fn from_size(size: ModelSize) -> Self {
        let (width, depth) = match size {
            ModelSize::Xs => (32, 1),
            ModelSize::S => (48, 1),
            ModelSize::M => (64, 2),
            ModelSize::L => (96, 2),
            ModelSize::Xl => (128, 3),
            ModelSize::Xxl => (160, 3),
        };
        Self::new()
            .with_width(width)
            .with_depth(depth)
            .with_norm_groups(8)
    }
}

/// Batch alignment network (v2): a shared strided encoder, an explicit
/// local-correlation comparison against the reference frame, a geometric
/// bounded head, and a differentiable frame-0 normalisation.
#[derive(Module, Debug)]
pub struct BatchAlignNet<B: Backend> {
    stem1: ConvNormActS<B>,
    stem2: ConvNormActS<B>,
    stage1: Vec<ResBlock<B>>,
    down: ConvNormActS<B>,
    stage2: Vec<ResBlock<B>>,
    head_in: ConvNormAct<B>,
    head_res1: ResBlock<B>,
    head_down: ConvNormActS<B>,
    head_res2: ResBlock<B>,
    fc1: burn::nn::Linear<B>,
    fc_act: Gelu,
    fc2: burn::nn::Linear<B>,
    corr_radius: usize,
}

impl BatchAlignNetConfig {
    /// # Panics
    ///
    /// Panics if `width` is not divisible by 2, if `width / 2` is not
    /// divisible by `norm_groups`, or if `width` is not divisible by
    /// `norm_groups`
    pub fn init<B: Backend>(&self, device: &B::Device) -> BatchAlignNet<B> {
        let w = self.width;
        let g = self.norm_groups;
        let w_half = w / 2;
        assert!(
            w.is_multiple_of(2) && w_half.is_multiple_of(g) && w.is_multiple_of(g),
            "BatchAlignNetConfig: width ({w}) must be even, with both (width / 2) and width divisible by norm_groups ({g})"
        );

        let stage1 = (0..self.depth)
            .map(|_| ResBlock::new(w, g, 1, device))
            .collect();
        let stage2 = (0..self.depth)
            .map(|_| ResBlock::new(w, g, 1, device))
            .collect();

        let corr_channels = (2 * self.corr_radius + 1) * (2 * self.corr_radius + 1);

        BatchAlignNet {
            stem1: ConvNormActS::new(3, w_half, g, device),
            stem2: ConvNormActS::new(w_half, w, g, device),
            stage1,
            down: ConvNormActS::new(w, w, g, device),
            stage2,
            head_in: ConvNormAct::new(corr_channels + 2 * w, w, g, device),
            head_res1: ResBlock::new(w, g, 1, device),
            head_down: ConvNormActS::new(w, w, g, device),
            head_res2: ResBlock::new(w, g, 1, device),
            fc1: burn::nn::LinearConfig::new(w, w / 2).init(device),
            fc_act: Gelu::new(),
            fc2: burn::nn::LinearConfig::new(w / 2, 6)
                .with_initializer(burn::nn::Initializer::Zeros)
                .init(device),
            corr_radius: self.corr_radius,
        }
    }
}

impl<B: Backend> BatchAlignNet<B> {
    fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut f = self.stem2.forward(self.stem1.forward(x));
        for block in &self.stage1 {
            f = block.forward(f);
        }
        f = self.down.forward(f);
        for block in &self.stage2 {
            f = block.forward(f);
        }
        f
    }

    fn head(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.head_in.forward(x);
        let x = self.head_res1.forward(x);
        let x = self.head_down.forward(x);
        let x = self.head_res2.forward(x);

        let x = x
            .mean_dim(3)
            .squeeze_dim::<3>(3)
            .mean_dim(2)
            .squeeze_dim::<2>(2);

        let x = self.fc_act.forward(self.fc1.forward(x));
        self.fc2.forward(x) // [N*S, 6]
    }

    /// Run the full v2 alignment forward pass, returning both the composed
    /// `[N,S,3,3]` affine matrices (the [`crate::traits::BatchAlignmentModel`]
    /// trait's output) and the raw pre-`tanh` head outputs `[N,S,6]`.
    ///
    /// The raw params are exposed (rather than only through the trait) so
    /// `train::align_loss` can apply the delta regulariser
    /// (`docs/batchalign-v2-design.md` §5.1) on them directly; the trait impl
    /// below just discards them via [`Self::align_batch`].
    #[must_use]
    pub fn forward_with_params(&self, stack: Tensor<B, 5>) -> (Tensor<B, 4>, Tensor<B, 3>) {
        let [batch_size, num_frames, _c, feat_height, feat_width] = stack.dims();
        let device = stack.device();
        let stack_flat = stack.reshape([batch_size * num_frames, 3, feat_height, feat_width]);

        let feat = self.encode(stack_flat);
        let feat = l2_normalize(feat);
        let [_, cw, h8, w8] = feat.dims();
        let feat_5d = feat.reshape([batch_size, num_frames, cw, h8, w8]);

        let f_ref = feat_5d
            .clone()
            .slice([0..batch_size, 0..1, 0..cw, 0..h8, 0..w8])
            .repeat_dim(1, num_frames)
            .reshape([batch_size * num_frames, cw, h8, w8]);
        let f_frm = feat_5d.reshape([batch_size * num_frames, cw, h8, w8]);

        let corr = local_correlation(&f_ref, f_frm.clone(), self.corr_radius, &device);
        let input_tensor = Tensor::cat(vec![corr, f_ref, f_frm], 1); // [N*S, (2r+1)^2 + 2*cw, h8, w8]

        let raw = self.head(input_tensor); // [N*S, 6]
        let pre_tanh_params = raw.clone().reshape([batch_size, num_frames, 6]);

        // Bounded geometric parameterisation (`docs/batchalign-v2-design.md`
        // §3.4): each raw head output is squashed through a scaled `tanh`
        // rather than used as a direct matrix entry. This guarantees a
        // well-conditioned, non-degenerate matrix (determinant = sx*sy > 0,
        // no reflections, no collapse) at every training step, gives the
        // optimiser geometrically meaningful and similarly-scaled directions
        // to move in (translation/rotation/scale otherwise live on wildly
        // different numeric scales as raw matrix entries), and caps how
        // wrong an early-training prediction can be. These constants are not
        // tunable hyperparameters independent of the training-data transform
        // ranges — widen them only together with the data generator's ranges
        // (see the design doc §4.4). Factored into
        // [`bounded_affine_from_raw6`] (shared with
        // [`crate::model::fusion_align::FusionAlignNet`]) since this math is
        // identical once flattened to `[M, 6]`.
        let affine_matrices =
            bounded_affine_from_raw6::<B>(raw, &device).reshape([batch_size, num_frames, 3, 3]);

        // Reference normalisation (`docs/batchalign-v2-design.md` §3.5):
        // left-compose every frame's matrix with the analytic inverse of
        // frame 0's, forcing frame 0 to be EXACTLY identity by construction
        // (`M_i' = M_0^-1 . M_i`). This removes the one global degree of
        // freedom the corner loss cannot pin down on its own (the whole
        // stack could drift together and a loss on relative placement would
        // barely notice), and makes the pipeline's "reference frame is never
        // warped" guarantee hold exactly rather than approximately. Computed
        // analytically (not via a generic tensor inverse) so the whole
        // operation stays differentiable with no CPU round trip; `det` can
        // never be zero because the §3.4 bounds guarantee
        // `det(A) = sx*sy >= exp(-0.16) > 0`, so no runtime guard is needed.
        let m0 = affine_matrices
            .clone()
            .slice([0..batch_size, 0..1, 0..3, 0..3]); // [N,1,3,3]
        let m0_00 = m0.clone().slice([0..batch_size, 0..1, 0..1, 0..1]);
        let m0_01 = m0.clone().slice([0..batch_size, 0..1, 0..1, 1..2]);
        let m0_02 = m0.clone().slice([0..batch_size, 0..1, 0..1, 2..3]);
        let m0_10 = m0.clone().slice([0..batch_size, 0..1, 1..2, 0..1]);
        let m0_11 = m0.clone().slice([0..batch_size, 0..1, 1..2, 1..2]);
        let m0_12 = m0.slice([0..batch_size, 0..1, 1..2, 2..3]);

        let det = m0_00
            .clone()
            .mul(m0_11.clone())
            .sub(m0_01.clone().mul(m0_10.clone())); // [N,1,1,1]
        let inv_det = det.recip();

        let inv00 = m0_11.clone().mul(inv_det.clone());
        let inv01 = m0_01.clone().neg().mul(inv_det.clone());
        let inv02 = m0_01
            .mul(m0_12.clone())
            .sub(m0_02.clone().mul(m0_11))
            .mul(inv_det.clone());
        let inv10 = m0_10.clone().neg().mul(inv_det.clone());
        let inv11 = m0_00.clone().mul(inv_det.clone());
        let inv12 = m0_02.mul(m0_10).sub(m0_00.mul(m0_12)).mul(inv_det);

        let inv_row1 = Tensor::cat(vec![inv00, inv01, inv02], 3);
        let inv_row2 = Tensor::cat(vec![inv10, inv11, inv12], 3);
        let inv_row3_zeros = Tensor::<B, 4>::zeros([batch_size, 1, 1, 2], &device);
        let inv_row3_one = Tensor::<B, 4>::ones([batch_size, 1, 1, 1], &device);
        let inv_row3 = Tensor::cat(vec![inv_row3_zeros, inv_row3_one], 3);
        let m0_inv = Tensor::cat(vec![inv_row1, inv_row2, inv_row3], 2); // [N,1,3,3]

        let m0_inv_bc = m0_inv.repeat_dim(1, num_frames); // [N,S,3,3]
        let m_normalised = m0_inv_bc.matmul(affine_matrices);

        (m_normalised, pre_tanh_params)
    }
}

impl<B: Backend> crate::traits::BatchAlignmentModel<B> for BatchAlignNet<B> {
    fn align_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4> {
        self.forward_with_params(stack).0
    }
}
