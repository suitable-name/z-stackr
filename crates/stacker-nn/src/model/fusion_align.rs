use crate::model::{
    batch_align::{ConvNormActS, bounded_affine_from_raw6, l2_normalize, local_correlation},
    blocks::{ConvNormAct, ResBlock},
    size::ModelSize,
};
use burn::{nn::Gelu, prelude::*};

/// Configuration for [`FusionAlignNet`].
///
/// Same knobs as [`crate::model::BatchAlignNetConfig`] (`width`, `depth`,
/// `norm_groups`, `corr_radius`) — the two architectures share an identical
/// encoder/correlation/head shape (see [`FusionAlignNet`]'s docs), so there
/// is nothing pairwise-specific to configure differently. Kept as its own
/// `Config` type (rather than reusing `BatchAlignNetConfig` directly) so the
/// two architectures can diverge independently in the future without one's
/// config surface leaking into the other's manifest/discovery plumbing (see
/// `crate::discovery`'s `ModelManifest::from_size_fusion_align` /
/// `to_fusion_align_config`, which mirror the batch equivalents exactly).
#[derive(Config, Debug)]
pub struct FusionAlignNetConfig {
    #[config(default = 64)]
    pub width: usize,
    #[config(default = 2)]
    pub depth: usize,
    #[config(default = 8)]
    pub norm_groups: usize,
    #[config(default = 4)]
    pub corr_radius: usize,
}

impl FusionAlignNetConfig {
    /// `(width, depth)` per [`ModelSize`] preset.
    ///
    /// Reuses the EXACT same table as
    /// [`crate::model::BatchAlignNetConfig::from_size`] — a sane default
    /// since [`FusionAlignNet`] is structurally analogous (same strided
    /// encoder, same correlation head), just applied to a reference/frame
    /// pair instead of a whole stack. There is no reason the two
    /// architectures' capacity presets should diverge unless training
    /// results show otherwise; if they do, change this table independently
    /// of `BatchAlignNetConfig`'s.
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

/// Pairwise alignment network: the streaming sibling of
/// [`crate::model::BatchAlignNet`], registering one `frame` against a
/// `reference` per call instead of a whole stack at once.
///
/// See `docs/fusionalign-design.md` for the full motivation. Architecturally
/// this is a two-input twin of `BatchAlignNet` — same shared-weight strided
/// encoder, same L2-normalised local-correlation comparison, same bounded
/// geometric head (via the shared [`bounded_affine_from_raw6`] helper) — but
/// it deliberately has **no reference-normalisation step**
/// (`docs/batchalign-v2-design.md` §3.5's analytic-inverse trick):
/// `BatchAlignNet` forces frame 0 of a stack to exact identity because frame
/// 0 IS the reference living inside the same batched call, so an extra
/// left-composition is nearly free. `FusionAlignNet` has no such "frame 0 of
/// a stack" structure to exploit — its `reference` argument already IS the
/// reference — so calling `align_pair(reference, reference)` a second time
/// just to force exact identity would cost a whole extra encoder pass per
/// invocation, undermining the O(1)-per-frame memory/compute story that is
/// this architecture's entire reason to exist. Instead, exact-identity
/// behaviour on `reference == frame` inputs is left to the bounded
/// parameterisation (zero-init head ⇒ identity at step 0) and to training
/// data that includes reference/frame identity pairs, so the network learns
/// near-identity output for that case rather than having it guaranteed by
/// construction. See `docs/fusionalign-design.md` for the full tradeoff
/// discussion.
#[derive(Module, Debug)]
pub struct FusionAlignNet<B: Backend> {
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

impl FusionAlignNetConfig {
    /// # Panics
    ///
    /// Panics if `width` is not divisible by 2, if `width / 2` is not
    /// divisible by `norm_groups`, or if `width` is not divisible by
    /// `norm_groups` — identical constraint to
    /// [`crate::model::BatchAlignNetConfig::init`], for the same reason
    /// (`GroupNorm` needs the channel count to divide evenly into groups).
    pub fn init<B: Backend>(&self, device: &B::Device) -> FusionAlignNet<B> {
        let w = self.width;
        let g = self.norm_groups;
        let w_half = w / 2;
        assert!(
            w.is_multiple_of(2) && w_half.is_multiple_of(g) && w.is_multiple_of(g),
            "FusionAlignNetConfig: width ({w}) must be even, with both (width / 2) and width divisible by norm_groups ({g})"
        );

        let stage1 = (0..self.depth)
            .map(|_| ResBlock::new(w, g, 1, device))
            .collect();
        let stage2 = (0..self.depth)
            .map(|_| ResBlock::new(w, g, 1, device))
            .collect();

        let corr_channels = (2 * self.corr_radius + 1) * (2 * self.corr_radius + 1);

        FusionAlignNet {
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

impl<B: Backend> FusionAlignNet<B> {
    /// Identical to [`crate::model::BatchAlignNet`]'s private `encode` — a
    /// shared-weight strided encoder run on `[M, 3, H, W]` (`M` = however
    /// many frames are batched through in one call), producing `[M, w,
    /// H/8, W/8]` features. Not factored into a shared free function because
    /// each architecture's `encode` closes over its own field set (`self`),
    /// and the body is short enough that duplication costs less than the
    /// indirection of threading every field through a free function's
    /// parameter list.
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

    /// Identical to [`crate::model::BatchAlignNet`]'s private `head`.
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
        self.fc2.forward(x) // [N, 6]
    }

    /// Run the full forward pass for one reference/frame pair, returning
    /// both the composed `[N,3,3]` affine matrix (the
    /// [`crate::traits::PairAlignmentModel`] trait's output) and the raw
    /// pre-`tanh` head outputs `[N,6]`.
    ///
    /// The raw params are exposed (rather than only through the trait), same
    /// pattern as [`crate::model::BatchAlignNet::forward_with_params`], so a
    /// training loop can apply the same delta regulariser
    /// (`docs/batchalign-v2-design.md` §5.1) on them directly; the trait impl
    /// below just discards them via [`Self::align_pair`].
    ///
    /// Implementation note: `reference` and `frame` are concatenated into one
    /// size-2*N pseudo-batch and run through the shared-weight [`Self::encode`]
    /// in a SINGLE call, then split — mirroring how `BatchAlignNet` flattens
    /// `[N,S,...]` into `[N*S,...]` to share its encoder call across every
    /// frame in the stack. This costs nothing extra (same total amount of
    /// convolution work as two separate calls) and keeps the two
    /// architectures' encoder-batching style consistent.
    #[must_use]
    pub fn forward_with_params(
        &self,
        reference: Tensor<B, 4>,
        frame: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [n, _c, _h, _w_dim] = reference.dims();
        let device = reference.device();

        let paired = Tensor::cat(vec![reference, frame], 0); // [2N, 3, H, W]
        let feat = self.encode(paired);
        let feat = l2_normalize(feat);
        let [_, cw, h8, w8] = feat.dims();

        let f_ref = feat.clone().slice([0..n, 0..cw, 0..h8, 0..w8]);
        let f_frm = feat.slice([n..2 * n, 0..cw, 0..h8, 0..w8]);

        let corr = local_correlation(&f_ref, f_frm.clone(), self.corr_radius, &device);
        let x = Tensor::cat(vec![corr, f_ref, f_frm], 1); // [N, (2r+1)^2 + 2*cw, h8, w8]

        let raw = self.head(x); // [N, 6]
        let m = bounded_affine_from_raw6::<B>(raw.clone(), &device); // [N, 3, 3]

        // Deliberately NO reference-normalisation step here — see this
        // struct's doc comment and `docs/fusionalign-design.md` for why.
        (m, raw)
    }
}

impl<B: Backend> crate::traits::PairAlignmentModel<B> for FusionAlignNet<B> {
    fn align_pair(&self, reference: Tensor<B, 4>, frame: Tensor<B, 4>) -> Tensor<B, 3> {
        self.forward_with_params(reference, frame).0
    }
}
