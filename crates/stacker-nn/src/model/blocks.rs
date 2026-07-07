use burn::{
    nn::{
        Gelu, GroupNorm, GroupNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

/// Dilation cycle for the context stack. Block `i` uses `DILATIONS[i & 3]`,
/// growing the receptive field geometrically without any strided pooling.
pub(crate) const DILATIONS: [usize; 4] = [1, 2, 4, 8];

/// `Conv3×3 (Same)` → `GroupNorm` → `GELU`.
#[derive(Module, Debug)]
pub(crate) struct ConvNormAct<B: Backend> {
    conv: Conv2d<B>,
    norm: GroupNorm<B>,
    act: Gelu,
}

impl<B: Backend> ConvNormAct<B> {
    pub(crate) fn new(c_in: usize, c_out: usize, groups: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [3, 3])
                .with_padding(PaddingConfig2d::Same)
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

/// Pre-activation-style dilated residual block (size-preserving).
///
/// `y = GN(conv_d(GELU(GN(conv_d(x)))))` then `GELU(x + y)`, where `conv_d` is a
/// 3×3 convolution with dilation `d` and matching explicit padding `d`.
#[derive(Module, Debug)]
pub(crate) struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    norm1: GroupNorm<B>,
    conv2: Conv2d<B>,
    norm2: GroupNorm<B>,
    act: Gelu,
}

impl<B: Backend> ResBlock<B> {
    pub(crate) fn new(width: usize, groups: usize, dilation: usize, device: &B::Device) -> Self {
        // For a 3×3 kernel, padding == dilation preserves H×W.
        let conv = || {
            Conv2dConfig::new([width, width], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(
                    dilation, dilation, dilation, dilation,
                ))
                .with_dilation([dilation, dilation])
                .init(device)
        };
        Self {
            conv1: conv(),
            norm1: GroupNormConfig::new(groups, width).init(device),
            conv2: conv(),
            norm2: GroupNormConfig::new(groups, width).init(device),
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

/// A pointwise 1×1 convolution to a small head output (no normalisation).
pub(crate) fn point_conv<B: Backend>(c_in: usize, c_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([c_in, c_out], [1, 1])
        .with_padding(PaddingConfig2d::Valid)
        .init(device)
}
