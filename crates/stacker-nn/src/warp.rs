//! Differentiable affine image warp — the spatial-transformer building block of
//! the photometric fine-tuning loss ([`crate::loss::photometric`]).
//!
//! [`warp_affine`] resamples an image `[N, C, H, W]` through a per-frame affine
//! matrix `[N, 3, 3]` in the crate-wide normalised `[-1, 1]²` convention (see
//! [`crate::bridge`]) and returns the warped image together with a validity mask
//! marking which output pixels sampled in-bounds source content. Gradients flow
//! through the bilinear blend weights (standard spatial-transformer behaviour),
//! which is what makes it usable as a training-time loss ingredient.
//!
//! ## Coordinate convention and direction
//!
//! `x_n = 2x/(W-1) - 1`, `y_n = 2y/(H-1) - 1`. A matrix `M` maps a **reference**
//! coordinate to the corresponding coordinate in the **source** (the frame being
//! warped): `p_source = M · p_reference`, the convention of every alignment
//! network in this crate and of the pipeline (the refined matrix maps reference
//! to source; the apps warp the source with the inverse of the *forward* map,
//! which is the same pull operation done here). So for every pixel of the
//! *reference grid* `warp_affine` evaluates `M · (x, y, 1)` and bilinearly samples
//! the source image there: warping a frame by the matrix that registers it to the
//! reference reproduces the reference.
//!
//! Example: a scene point at reference pixel `p` that sits at source pixel
//! `p + (3, 0)` is described by the translation `M = T(+3, 0)`, and
//! `warp_affine(source, M)` returns, at `p`, the source value at `p + (3, 0)`.
//!
//! No custom backward pass is needed: grid construction, matrix application,
//! floor/clamp, gather and the bilinear blend are standard differentiable tensor
//! ops, so Burn's autodiff traces through them whole. The gather index is built
//! in `f32`, so images above `2²⁴` pixels (4096 × 4096) are not supported — the
//! intended inputs are crops and ≤512 px thumbnails.

use burn::{
    prelude::*,
    tensor::{Int, TensorData},
};

/// Tolerance (pixels) of the validity test at the image border.
const VALID_EPS: f32 = 1e-3;

/// Result of [`warp_affine`]: the resampled image and its validity mask.
#[derive(Debug)]
pub struct WarpOutput<B: Backend> {
    /// Warped image, `[N, C, H, W]`, same shape as the input.
    ///
    /// Pixels sampled from outside the source image (see [`WarpOutput::valid`])
    /// hold a well-defined clamped-coordinate value; mask with
    /// [`WarpOutput::valid`] rather than relying on any out-of-bounds fill.
    pub warped: Tensor<B, 4>,
    /// Validity mask, `[N, 1, H, W]`: `1.0` where the **unclamped** sampling
    /// coordinate stayed within the source image, `0.0` otherwise. A hard 0/1
    /// indicator; no gradient flows through it.
    pub valid: Tensor<B, 4>,
}

/// Build a normalised `[-1, 1]` coordinate grid as two `[H*W]` tensors
/// `(xs, ys)`, row-major (`x` fastest), `x_n = 2x/(W-1) - 1`.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel indices are far below 2^24"
)]
fn normalized_grid<B: Backend>(
    h: usize,
    w: usize,
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let mut xs = vec![0f32; h * w];
    let mut ys = vec![0f32; h * w];
    let sx = 2.0_f32 / (w.max(2) as f32 - 1.0);
    let sy = 2.0_f32 / (h.max(2) as f32 - 1.0);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            xs[idx] = sx.mul_add(x as f32, -1.0);
            ys[idx] = sy.mul_add(y as f32, -1.0);
        }
    }
    let xs_t = Tensor::<B, 1>::from_data(TensorData::new(xs, [h * w]), device);
    let ys_t = Tensor::<B, 1>::from_data(TensorData::new(ys, [h * w]), device);
    (xs_t, ys_t)
}

/// Warp `image` (`[N, C, H, W]`) through the per-frame affine matrices `matrix`.
///
/// The matrices are `[N, 3, 3]` in the normalised convention, reference → source
/// (see the module docs); the output is `[N, C, H, W]` on the same `(H, W)` grid.
///
/// # Panics
///
/// Panics if `image`'s batch dimension does not match `matrix`'s, or if `H < 2`
/// or `W < 2` (the normalised grid is undefined for a single row or column).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel indices and sizes are far below 2^24"
)]
pub fn warp_affine<B: Backend>(image: Tensor<B, 4>, matrix: Tensor<B, 3>) -> WarpOutput<B> {
    let [n, c, h, w] = image.dims();
    let [mn, _, _] = matrix.dims();
    assert_eq!(n, mn, "warp_affine: image batch {n} != matrix batch {mn}");
    assert!(
        h >= 2 && w >= 2,
        "warp_affine: H and W must both be >= 2, got {h}x{w}"
    );

    let device = image.device();
    let (xs, ys) = normalized_grid::<B>(h, w, &device); // each [H*W]

    // Grid as [1, HW]; matrix entries as [N, 1] (broadcast over the grid).
    let xs_b = xs.reshape([1, h * w]);
    let ys_b = ys.reshape([1, h * w]);

    let m00 = matrix.clone().slice([0..n, 0..1, 0..1]).reshape([n, 1]);
    let m01 = matrix.clone().slice([0..n, 0..1, 1..2]).reshape([n, 1]);
    let m02 = matrix.clone().slice([0..n, 0..1, 2..3]).reshape([n, 1]);
    let m10 = matrix.clone().slice([0..n, 1..2, 0..1]).reshape([n, 1]);
    let m11 = matrix.clone().slice([0..n, 1..2, 1..2]).reshape([n, 1]);
    let m12 = matrix.slice([0..n, 1..2, 2..3]).reshape([n, 1]);

    // x' = m00*x + m01*y + m02 ; y' = m10*x + m11*y + m12  — both [N, HW].
    let x_src = m00.mul(xs_b.clone()).add(m01.mul(ys_b.clone())).add(m02);
    let y_src = m10.mul(xs_b).add(m11.mul(ys_b)).add(m12);

    // Normalised [-1,1] -> pixel coordinates in [0, W-1] / [0, H-1].
    let px = x_src.add_scalar(1.0_f32).mul_scalar((w as f32 - 1.0) / 2.0); // [N, HW]
    let py = y_src.add_scalar(1.0_f32).mul_scalar((h as f32 - 1.0) / 2.0); // [N, HW]

    // Validity: the unclamped pixel coordinate must stay in [0, W-1] / [0, H-1]
    // (within `VALID_EPS`, so a coordinate that is exactly on the border up to f32
    // rounding of the normalise/denormalise round trip still counts as inside).
    let valid_x = px.clone().greater_equal_elem(-VALID_EPS).float().mul(
        px.clone()
            .lower_equal_elem(w as f32 - 1.0 + VALID_EPS)
            .float(),
    );
    let valid_y = py.clone().greater_equal_elem(-VALID_EPS).float().mul(
        py.clone()
            .lower_equal_elem(h as f32 - 1.0 + VALID_EPS)
            .float(),
    );
    let valid = valid_x.mul(valid_y); // [N, HW]

    // Clamp for the actual sampling: out-of-bounds samples must still produce a
    // finite, in-range gather index.
    let col_pos = px.clamp(0.0, w as f32 - 1.0);
    let row_pos = py.clamp(0.0, h as f32 - 1.0);

    // The integer cell is a constant of the sampling (gradients flow through the
    // fractional weights only).
    let x0f = col_pos.clone().detach().floor();
    let y0f = row_pos.clone().detach().floor();
    let fx = col_pos.sub(x0f.clone()); // fractional bilinear weights
    let fy = row_pos.sub(y0f.clone());

    let x1f = x0f.clone().add_scalar(1.0_f32).clamp(0.0, w as f32 - 1.0);
    let y1f = y0f.clone().add_scalar(1.0_f32).clamp(0.0, h as f32 - 1.0);

    // Flatten the image to [N, C, H*W] for a dim-2 gather.
    let img_flat = image.reshape([n, c, h * w]);

    let idx00 = flat_index::<B>(&x0f, &y0f, w);
    let idx10 = flat_index::<B>(&x1f, &y0f, w);
    let idx01 = flat_index::<B>(&x0f, &y1f, w);
    let idx11 = flat_index::<B>(&x1f, &y1f, w);

    let gather3 = |idx: Tensor<B, 2, Int>| -> Tensor<B, 3> {
        let idx_c = idx.unsqueeze_dim::<3>(1).expand([n, c, h * w]);
        img_flat.clone().gather(2, idx_c)
    };
    let v00 = gather3(idx00); // [N, C, HW]
    let v10 = gather3(idx10);
    let v01 = gather3(idx01);
    let v11 = gather3(idx11);

    // Bilinear blend weights, broadcast [N, HW] -> [N, 1, HW] -> [N, C, HW].
    let fx3 = fx.unsqueeze_dim::<3>(1);
    let fy3 = fy.unsqueeze_dim::<3>(1);
    let left_weight = fx3.clone().neg().add_scalar(1.0_f32);
    let top_weight = fy3.clone().neg().add_scalar(1.0_f32);

    let top = v00.mul(left_weight.clone()).add(v10.mul(fx3.clone()));
    let bottom = v01.mul(left_weight).add(v11.mul(fx3));
    let blended = top.mul(top_weight).add(bottom.mul(fy3)); // [N, C, HW]

    WarpOutput {
        warped: blended.reshape([n, c, h, w]),
        valid: valid.reshape([n, 1, h, w]),
    }
}

/// Build the flat gather index `y * w + x` (an `Int` tensor `[N, HW]`) from
/// floor-valued pixel-coordinate float tensors, already clamped to the valid range.
#[expect(
    clippy::cast_precision_loss,
    reason = "the row stride is far below 2^24"
)]
fn flat_index<B: Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>, w: usize) -> Tensor<B, 2, Int> {
    y.clone().mul_scalar(w as f32).add(x.clone()).int()
}
