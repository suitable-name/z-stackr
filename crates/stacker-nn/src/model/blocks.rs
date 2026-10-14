//! Building blocks shared by every network: the tile-invariant [`ChannelNorm`],
//! the size-preserving `Conv → norm → GELU` and dilated residual blocks, and the
//! strided variant the alignment encoders use.

use burn::{
    module::Param,
    nn::{
        Gelu, Initializer, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

/// Dilation cycle for the context stack. Block `i` uses `DILATIONS[i % 4]`,
/// growing the receptive field geometrically without any strided pooling.
pub(crate) const DILATIONS: [usize; 4] = [1, 2, 4, 8];

/// Radius contributed by everything outside the context stack of a fusion
/// network: stem + stem2 (2), sharpness conv (1), gate-in conv (1), gate residual
/// block (2), refinement-in conv (1), refinement residual block (2),
/// refinement-out conv (1).
pub(crate) const FIXED_RADIUS: usize = 10;

/// `2·Σ DILATIONS[i % 4] (i < depth) + FIXED_RADIUS` — the receptive-field radius
/// (pixels) of the merged output of a fusion network with `depth` context blocks.
///
/// Both [`crate::model::FocusMergeNet`] and [`crate::model::BatchMergeNet`] have
/// exactly this radius: the batch network's gate sees `2·Σd + 6` pixels and its
/// refinement head adds the same `4` the pairwise one does.
#[must_use]
pub const fn fusion_radius_for_depth(depth: usize) -> usize {
    let mut sum = 0;
    let mut i = 0;
    while i < depth {
        sum += DILATIONS[i % DILATIONS.len()];
        i += 1;
    }
    2 * sum + FIXED_RADIUS
}

/// Per-pixel channel normalisation — the tile-invariant replacement for
/// `GroupNorm`.
///
/// For every pixel independently, the `C` channel values are split into `groups`
/// groups of `C / groups` consecutive channels; each group is shifted to zero mean
/// and scaled to unit variance **over its channels at that pixel**, then a learned
/// per-channel affine `y = x̂·scale + shift` is applied. With `groups == 1` this is
/// a `LayerNorm` over the channel axis (the normalisation `ConvNeXt` applies at
/// every pixel).
///
/// No statistic ever involves a spatial position other than the pixel itself, a
/// batch entry or a tile, so the output at a pixel is a function of that pixel's
/// input alone. The layer therefore adds nothing to the receptive field and cannot
/// make a tile's output differ from the whole-image run, which `GroupNorm`
/// (statistics over the whole `H × W` extent) did.
///
/// The parameters are named `scale` / `shift` rather than `gamma` / `beta` on
/// purpose: a checkpoint of the old `GroupNorm` architecture then fails to load
/// (missing field) instead of loading silently into layers that mean something
/// else. Old weights must be retrained.
#[derive(Module, Debug)]
pub struct ChannelNorm<B: Backend> {
    /// Learned per-channel gain, initialised to ones.
    pub scale: Param<Tensor<B, 1>>,
    /// Learned per-channel bias, initialised to zeros.
    pub shift: Param<Tensor<B, 1>>,
    /// Number of channel groups statistics are computed over.
    pub groups: usize,
    /// Variance floor for numerical stability.
    pub epsilon: f64,
}

impl<B: Backend> ChannelNorm<B> {
    /// Build a norm over `channels` channels split into `groups` groups.
    ///
    /// # Panics
    ///
    /// Panics if `groups` is zero or does not divide `channels`.
    #[must_use]
    pub fn new(groups: usize, channels: usize, device: &B::Device) -> Self {
        assert!(
            groups > 0 && channels.is_multiple_of(groups),
            "the number of channels ({channels}) must be divisible by the number of groups ({groups})"
        );
        Self {
            scale: Initializer::Ones.init([channels], device),
            shift: Initializer::Zeros.init([channels], device),
            groups,
            epsilon: 1e-5,
        }
    }

    /// Normalise `x` (`[N, C, H, W]`), pixel by pixel.
    #[must_use]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let groups = self.groups;
        // [N, G, C/G, H, W]: the statistics run over axis 2 only.
        let grouped = x.reshape([batch, groups, channels / groups, height, width]);
        let centred = grouped.clone().sub(grouped.mean_dim(2));
        let variance = centred.clone().mul(centred.clone()).mean_dim(2);
        let normed = centred
            .div(variance.add_scalar(self.epsilon).sqrt())
            .reshape([batch, channels, height, width]);
        normed
            .mul(self.scale.val().reshape([1, channels, 1, 1]))
            .add(self.shift.val().reshape([1, channels, 1, 1]))
    }
}

/// `Conv3×3 (Same)` → [`ChannelNorm`] → `GELU`.
#[derive(Module, Debug)]
pub(crate) struct ConvNormAct<B: Backend> {
    conv: Conv2d<B>,
    norm: ChannelNorm<B>,
    act: Gelu,
}

impl<B: Backend> ConvNormAct<B> {
    pub(crate) fn new(c_in: usize, c_out: usize, groups: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            norm: ChannelNorm::new(groups, c_out, device),
            act: Gelu::new(),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        self.act.forward(x)
    }
}

/// Stride-2 counterpart of [`ConvNormAct`]: `Conv3×3 (stride 2, pad 1)` →
/// [`ChannelNorm`] → `GELU`. The output side is `ceil(side / 2)`.
///
/// Used only by the alignment encoders' downsampling stages, and kept apart from
/// [`ConvNormAct`] so the fusion networks' size-preserving contract can never be
/// perturbed by an alignment-only change.
#[derive(Module, Debug)]
pub(crate) struct ConvNormActS<B: Backend> {
    conv: Conv2d<B>,
    norm: ChannelNorm<B>,
    act: Gelu,
}

impl<B: Backend> ConvNormActS<B> {
    pub(crate) fn new(c_in: usize, c_out: usize, groups: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm: ChannelNorm::new(groups, c_out, device),
            act: Gelu::new(),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        self.act.forward(x)
    }
}

/// Pre-activation-style dilated residual block (size-preserving).
///
/// `y = N(conv_d(GELU(N(conv_d(x)))))` then `GELU(x + y)`, where `conv_d` is a
/// 3×3 convolution with dilation `d` and matching explicit padding `d`, and `N`
/// is the per-pixel [`ChannelNorm`].
#[derive(Module, Debug)]
pub(crate) struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    norm1: ChannelNorm<B>,
    conv2: Conv2d<B>,
    norm2: ChannelNorm<B>,
    act: Gelu,
}

impl<B: Backend> ResBlock<B> {
    pub(crate) fn new(width: usize, groups: usize, dilation: usize, device: &B::Device) -> Self {
        // For a 3×3 kernel, padding == dilation preserves H×W.
        let conv = || {
            Conv2dConfig::new([width, width], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(dilation, dilation))
                .with_dilation([dilation, dilation])
                .init(device)
        };
        Self {
            conv1: conv(),
            norm1: ChannelNorm::new(groups, width, device),
            conv2: conv(),
            norm2: ChannelNorm::new(groups, width, device),
            act: Gelu::new(),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.conv1.forward(x.clone());
        let y = self.act.forward(self.norm1.forward(y));
        let y = self.norm2.forward(self.conv2.forward(y));
        self.act.forward(x.add(y))
    }
}

/// Per-element maximum over the stack axis of `[N, S, C, H, W]` → `[N, C, H, W]`.
///
/// Built from elementwise `max_pair` instead of `max_dim`: the ndarray backend's
/// backward pass for `max_dim` gathers along a non-last dimension, which it does
/// not support.
pub(crate) fn max_over_stack<B: Backend>(stack: &Tensor<B, 5>) -> Tensor<B, 4> {
    let [batch, depth, chans, rows, cols] = stack.dims();
    let mut pooled = stack
        .clone()
        .slice([0..batch, 0..1, 0..chans, 0..rows, 0..cols]);
    for i in 1..depth {
        pooled =
            pooled.max_pair(
                stack
                    .clone()
                    .slice([0..batch, i..i + 1, 0..chans, 0..rows, 0..cols]),
            );
    }
    pooled.squeeze::<4>(1)
}

/// A pointwise 1×1 convolution to a small head output (no normalisation).
pub(crate) fn point_conv<B: Backend>(c_in: usize, c_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([c_in, c_out], [1, 1])
        .with_padding(PaddingConfig2d::Valid)
        .init(device)
}
