//! Differentiable affine image warp — the spatial-transformer building block
//! for the §5.2 photometric fine-tuning phase (`docs/batchalign-v2-design.md`).
//!
//! [`warp_affine`] resamples an image `[N, C, H, W]` through a per-frame
//! predicted affine matrix `[N, 3, 3]` (the crate-wide normalized `[-1, 1]²`
//! convention — see [`crate::bridge::align_planar`]'s docs), returning the
//! warped image AND a validity mask marking which output pixels sampled
//! in-bounds source content. Gradients flow through the bilinear blend
//! weights (standard spatial-transformer behaviour), which is what makes
//! this usable as a training-time loss ingredient rather than only an
//! inference-time convenience.
//!
//! ## Coordinate convention
//!
//! The same normalized `[-1, 1]²` convention as the rest of this crate:
//! `x_n = 2x/(W-1) - 1`, `y_n = 2y/(H-1) - 1`. A predicted matrix `M` maps
//! `[x_n, y_n, 1]^T` (a REFERENCE-frame coordinate) to `[x'_n, y'_n, 1]^T`
//! (the corresponding coordinate in the SOURCE frame) — the same direction
//! [`crate::model::BatchAlignNet::align_batch`] predicts. To resample the
//! source frame onto the reference grid, [`warp_affine`] evaluates, for every
//! reference-grid output pixel, `(x', y') = M . (x, y, 1)` and bilinearly
//! samples the source image at `(x', y')` — i.e. it performs the same
//! forward-mapping convention `warp_image_clamped` uses classically, just
//! differentiably and in normalized coordinates.
//!
//! No custom backward pass is needed: every operation (grid construction,
//! matrix application, floor/clamp, gather, bilinear blend) is a standard
//! differentiable tensor op, so Burn's autodiff traces through it whole.

use burn::{
    prelude::*,
    tensor::{Int, TensorData},
};

/// Result of [`warp_affine`]: the resampled image and its validity mask.
#[derive(Debug)]
pub struct WarpOutput<B: Backend> {
    /// Warped image, `[N, C, H, W]`, same shape as the input.
    ///
    /// Pixels sampled from outside the source image (see
    /// [`WarpOutput::valid`]) are still a well-defined (clamped-coordinate)
    /// value, not garbage — callers that care about validity should mask
    /// with [`WarpOutput::valid`] rather than relying on any particular
    /// out-of-bounds fill value.
    pub warped: Tensor<B, 4>,
    /// Validity mask, `[N, 1, H, W]`, `1.0` where the UNCLAMPED sampling
    /// coordinate stayed within the source image bounds, `0.0` otherwise.
    /// No gradient flows through this mask (and none is needed — it is a
    /// hard 0/1 indicator built from comparison ops).
    pub valid: Tensor<B, 4>,
}

/// Build a normalized `[-1, 1]` coordinate grid as two `[H*W]` tensors
/// `(xs, ys)`, row-major (`x` fastest), matching the crate-wide convention
/// `x_n = 2x/(W-1) - 1`, `y_n = 2y/(H-1) - 1`.
///
/// Built as a plain `Vec<f32>` filled by nested loops and uploaded via
/// `TensorData`, exactly like `infer::hann_1d` builds its window — cheap,
/// done once per warp call, and keeps the grid construction trivially
/// correct to read.
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

/// Warp `image` (`[N, C, H, W]`) through the per-frame affine matrices
/// `matrix` (`[N, 3, 3]`, normalized `[-1, 1]²` convention — see the module
/// docs), producing an `[N, C, H, W]` output on the SAME `(H, W)` grid.
///
/// # Panics
///
/// Panics if `image`'s batch dimension does not match `matrix`'s, or if
/// `H < 2` or `W < 2` (see [`normalized_grid`]'s convention, which is
/// undefined for a single-row/column image).
#[must_use]
pub fn warp_affine<B: Backend>(image: Tensor<B, 4>, matrix: Tensor<B, 3>) -> WarpOutput<B> {
    // `px_c`/`py_c` (clamped pixel coords) and `one_minus_fx`/`one_minus_fy`
    // (complementary bilinear weights) are paired x/y quantities from the same
    // formula (see the module docs' step 3-4 recipe) — the shared naming makes
    // that pairing legible rather than obscuring it.
    let [n, c, h, w] = image.dims();
    let [mn, _, _] = matrix.dims();
    assert_eq!(n, mn, "warp_affine: image batch {n} != matrix batch {mn}");
    assert!(
        h >= 2 && w >= 2,
        "warp_affine: H and W must both be >= 2, got {h}x{w}"
    );

    let device = image.device();
    let (xs, ys) = normalized_grid::<B>(h, w, &device); // each [H*W]

    // Broadcast the grid over the batch: [1, H*W] -> matrix entries are
    // [N, 1] sliced from `matrix`, broadcast-multiplied against [1, H*W].
    let xs_b = xs.reshape([1, h * w]); // [1, HW]
    let ys_b = ys.reshape([1, h * w]); // [1, HW]

    // Matrix entries as [N, 1] tensors (broadcast over the HW grid dim).
    let m00 = matrix.clone().slice([0..n, 0..1, 0..1]).reshape([n, 1]);
    let m01 = matrix.clone().slice([0..n, 0..1, 1..2]).reshape([n, 1]);
    let m02 = matrix.clone().slice([0..n, 0..1, 2..3]).reshape([n, 1]);
    let m10 = matrix.clone().slice([0..n, 1..2, 0..1]).reshape([n, 1]);
    let m11 = matrix.clone().slice([0..n, 1..2, 1..2]).reshape([n, 1]);
    let m12 = matrix.slice([0..n, 1..2, 2..3]).reshape([n, 1]);

    // x' = m00*x + m01*y + m02 ; y' = m10*x + m11*y + m12  — both [N, HW].
    let x_src = m00.mul(xs_b.clone()).add(m01.mul(ys_b.clone())).add(m02);
    let y_src = m10.mul(xs_b).add(m11.mul(ys_b)).add(m12);

    // Convert normalized [-1,1] -> pixel coordinates in [0, W-1] / [0, H-1].
    let px = x_src.add_scalar(1.0_f32).mul_scalar((w as f32 - 1.0) / 2.0); // [N, HW]
    let py = y_src.add_scalar(1.0_f32).mul_scalar((h as f32 - 1.0) / 2.0); // [N, HW]

    // Validity: unclamped pixel coordinate must stay in [0, W-1] / [0, H-1].
    // Built from comparisons — no gradient needed or wanted here.
    let valid_x = px
        .clone()
        .greater_equal_elem(0.0_f32)
        .float()
        .mul(px.clone().lower_equal_elem(w as f32 - 1.0).float());
    let valid_y = py
        .clone()
        .greater_equal_elem(0.0_f32)
        .float()
        .mul(py.clone().lower_equal_elem(h as f32 - 1.0).float());
    let valid = valid_x.mul(valid_y); // [N, HW]

    // Clamp for the actual sampling (out-of-bounds samples must still
    // produce a finite, in-range gather index).
    let px_c = px.clamp(0.0, w as f32 - 1.0);
    let py_c = py.clamp(0.0, h as f32 - 1.0);

    let x0f = px_c.clone().floor();
    let y0f = py_c.clone().floor();
    let fx = px_c.sub(x0f.clone()); // [N, HW] fractional weight
    let fy = py_c.sub(y0f.clone());

    let x1f = x0f.clone().add_scalar(1.0_f32).clamp(0.0, w as f32 - 1.0);
    let y1f = y0f.clone().add_scalar(1.0_f32).clamp(0.0, h as f32 - 1.0);

    // Flatten the image to [N, C, H*W] for a dim-2 gather.
    let img_flat = image.reshape([n, c, h * w]);

    // Build the four corner flat-index tensors [N, HW] (as Int), each
    // expanded to [N, C, HW] to gather every channel at once.
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
    let one_minus_fx = fx3.clone().neg().add_scalar(1.0_f32);
    let one_minus_fy = fy3.clone().neg().add_scalar(1.0_f32);

    let top = v00.mul(one_minus_fx.clone()).add(v10.mul(fx3.clone()));
    let bottom = v01.mul(one_minus_fx).add(v11.mul(fx3));
    let blended = top.mul(one_minus_fy).add(bottom.mul(fy3)); // [N, C, HW]

    let warped = blended.reshape([n, c, h, w]);
    let valid_map = valid.reshape([n, 1, h, w]);

    WarpOutput {
        warped,
        valid: valid_map,
    }
}

/// Build the flat gather index `y * w + x` (as an `Int` tensor `[N, HW]`)
/// from floor-valued pixel-coordinate float tensors `x`, `y` (already
/// clamped to valid range by the caller).
fn flat_index<B: Backend>(x: &Tensor<B, 2>, y: &Tensor<B, 2>, w: usize) -> Tensor<B, 2, Int> {
    y.clone().mul_scalar(w as f32).add(x.clone()).int()
}

// Test helpers/bodies pair up `dx_px`/`dy_px` and similar x/y quantities
// throughout — see the individual functions for why shared naming is
// clearer here than artificially distinct names.
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn device() -> burn::prelude::Device<B> {
        burn::prelude::Device::<B>::default()
    }

    /// Build the normalized-space translation matrix for a pure pixel-space
    /// shift `(dx_px, dy_px)` on an `h x w` image, matching the crate-wide
    /// `M_n = N . M_px . N^-1` conjugation (here inlined directly, since a
    /// pure translation's normalized-space form is simple to derive:
    /// `tx_n = 2*dx_px/(w-1)`, `ty_n = 2*dy_px/(h-1)`, scale/rotation
    /// entries stay identity).
    fn translation_matrix_normalized<B: Backend>(
        dx_px: f32,
        dy_px: f32,
        w: usize,
        h: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let tx = 2.0 * dx_px / (w as f32 - 1.0);
        let ty = 2.0 * dy_px / (h as f32 - 1.0);
        #[rustfmt::skip]
        let data = [
            1.0_f32, 0.0, tx,
            0.0, 1.0, ty,
            0.0, 0.0, 1.0,
        ];
        Tensor::<B, 1>::from_data(TensorData::new(data.to_vec(), [9]), device).reshape([1, 3, 3])
    }

    /// CPU reference bilinear sample of a `[H, W]` single-channel image
    /// (border-clamped), used to independently check [`warp_affine`]'s
    /// numeric output for a KNOWN translation, rather than testing the
    /// implementation against itself.
    fn cpu_bilinear_sample(img: &[f32], h: usize, w: usize, x: f32, y: f32) -> f32 {
        let xc = x.clamp(0.0, w as f32 - 1.0);
        let yc = y.clamp(0.0, h as f32 - 1.0);
        let x0 = xc.floor();
        let y0 = yc.floor();
        let x1 = (x0 + 1.0).min(w as f32 - 1.0);
        let y1 = (y0 + 1.0).min(h as f32 - 1.0);
        let fx = xc - x0;
        let fy = yc - y0;
        let at = |yy: f32, xx: f32| img[yy as usize * w + xx as usize];
        let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
        let bottom = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    /// [`warp_affine`] warped by a known translation matrix must match a
    /// CPU-computed bilinear reference at every pixel (interior + border).
    #[test]
    fn warp_affine_matches_cpu_reference_for_known_translation() {
        let dev = device();
        let (h, w) = (10_usize, 12_usize);
        let (dx_px, dy_px) = (2.0_f32, -1.5_f32);

        // Deterministic pseudo-random single-channel source image.
        let mut src = vec![0f32; h * w];
        let mut t = 0.21_f32;
        for v in &mut src {
            t = t.mul_add(1.1, 0.3).fract();
            *v = t;
        }
        let image = Tensor::<B, 1>::from_data(TensorData::new(src.clone(), [h * w]), &dev)
            .reshape([1, 1, h, w]);

        let matrix = translation_matrix_normalized::<B>(dx_px, dy_px, w, h, &dev);
        let out = warp_affine(image, matrix);
        assert_eq!(out.warped.dims(), [1, 1, h, w]);

        let warped_data: Vec<f32> = out.warped.into_data().iter::<f32>().collect();

        // `warp_affine` maps reference pixel (x,y) to source (x + dx_px,
        // y + dy_px) — matching the module docs' forward-mapping
        // convention (a translation matrix M with tx=dx_px in pixel space
        // sends [x,y,1] -> [x+dx_px, y+dy_px, 1]).
        for y in 0..h {
            for x in 0..w {
                let want = cpu_bilinear_sample(&src, h, w, x as f32 + dx_px, y as f32 + dy_px);
                let got = warped_data[y * w + x];
                assert!(
                    (got - want).abs() < 1e-4,
                    "pixel ({x},{y}): got {got}, expected {want}"
                );
            }
        }
    }

    /// The validity mask must be exactly 1.0 for output pixels whose
    /// unclamped source coordinate stayed in-bounds, and exactly 0.0 for
    /// those that fell outside — using a translation large enough to push
    /// one whole border band out of bounds, so both classes are exercised.
    #[test]
    fn warp_affine_validity_mask_matches_out_of_bounds_samples() {
        let dev = device();
        let (h, w) = (8_usize, 8_usize);
        let dx_px = 3.0_f32; // pushes the rightmost 3 output columns OOB
        let dy_px = 0.0_f32;

        let image = Tensor::<B, 1>::from_data(TensorData::new(vec![0.5f32; h * w], [h * w]), &dev)
            .reshape([1, 1, h, w]);
        let matrix = translation_matrix_normalized::<B>(dx_px, dy_px, w, h, &dev);

        let out = warp_affine(image, matrix);
        let valid_data: Vec<f32> = out.valid.into_data().iter::<f32>().collect();

        for y in 0..h {
            for x in 0..w {
                let src_x = x as f32 + dx_px;
                let expect_valid = (0.0..=(w as f32 - 1.0)).contains(&src_x);
                let got = valid_data[y * w + x];
                if expect_valid {
                    assert!(
                        (got - 1.0).abs() < 1e-6,
                        "pixel ({x},{y}): expected valid=1.0, got {got}"
                    );
                } else {
                    assert!(
                        got.abs() < 1e-6,
                        "pixel ({x},{y}): expected valid=0.0, got {got}"
                    );
                }
            }
        }
    }

    /// The identity matrix must warp an image to itself exactly (bilinear
    /// sampling at exact integer coordinates degenerates to a copy), and
    /// every output pixel must be valid.
    #[test]
    fn warp_affine_identity_is_no_op() {
        let dev = device();
        let (h, w) = (6_usize, 7_usize);
        let mut src = vec![0f32; h * w * 3];
        let mut t = 0.6_f32;
        for v in &mut src {
            t = t.mul_add(1.3, 0.2).fract();
            *v = t;
        }
        let image = Tensor::<B, 1>::from_data(TensorData::new(src.clone(), [h * w * 3]), &dev)
            .reshape([1, 3, h, w]);
        let identity = Tensor::<B, 1>::from_data(
            TensorData::new(vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [9]),
            &dev,
        )
        .reshape([1, 3, 3]);

        let out = warp_affine(image, identity);
        let warped_data: Vec<f32> = out.warped.into_data().iter::<f32>().collect();
        for (got, &want) in warped_data.iter().zip(src.iter()) {
            assert!((got - want).abs() < 1e-5, "got {got}, expected {want}");
        }

        let valid_data: Vec<f32> = out.valid.into_data().iter::<f32>().collect();
        assert!(valid_data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }
}
