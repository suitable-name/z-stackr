//! Integration tests for the planar<->tensor fusion bridge.

use burn::{
    backend::NdArray,
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use stacker_core::image::PlanarImage;
use stacker_nn::{
    ModelSize,
    bridge::{BridgeError, align_planar, align_planar_pairwise, fuse_planar},
    infer::TileConfig,
    model::FocusMergeNetConfig,
    traits::{BatchAlignmentModel, PairAlignmentModel},
};

type B = NdArray;

/// Deterministic pseudo-random planar image (no integer→float casts, to stay
/// clippy-clean in the test crate).
fn planar(w: usize, h: usize, seed: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    let mut t = seed.fract().abs() + 0.1;
    for i in 0..w * h {
        t = (t * 1.1 + 0.3).fract();
        img.luma[i] = t;
        img.chroma_a[i] = (t - 0.5) * 0.2;
        img.chroma_b[i] = (0.5 - t) * 0.1;
    }
    img
}

fn xs_model(dev: burn::prelude::Device<B>) -> stacker_nn::model::FocusMergeNet<B> {
    FocusMergeNetConfig::from_size(ModelSize::Xs).init::<B>(&dev)
}

#[test]
fn fuse_planar_shape_and_finite() {
    let dev = burn::prelude::Device::<B>::default();
    let model = xs_model(dev);
    let frames = vec![
        planar(24, 20, 0.0),
        planar(24, 20, 1.0),
        planar(24, 20, 2.0),
    ];
    let out = fuse_planar(
        &model,
        &frames,
        TileConfig {
            tile: 16,
            overlap: 4,
        },
        &dev,
    )
    .unwrap();

    assert_eq!((out.width, out.height), (24, 20));
    assert!(out.luma.iter().all(|v| v.is_finite()));
    assert!(out.chroma_a.iter().all(|v| v.is_finite()));
    assert!(out.chroma_b.iter().all(|v| v.is_finite()));
}

#[test]
fn fuse_planar_empty_is_error() {
    let dev = burn::prelude::Device::<B>::default();
    let model = xs_model(dev);
    let frames: Vec<PlanarImage<f32>> = Vec::new();
    assert!(matches!(
        fuse_planar(&model, &frames, TileConfig::default(), &dev),
        Err(BridgeError::EmptyStack)
    ));
}

#[test]
fn fuse_planar_shape_mismatch_is_error() {
    let dev = burn::prelude::Device::<B>::default();
    let model = xs_model(dev);
    let frames = vec![planar(16, 16, 0.0), planar(16, 12, 1.0)];
    assert!(matches!(
        fuse_planar(&model, &frames, TileConfig::default(), &dev),
        Err(BridgeError::ShapeMismatch { index: 1, .. })
    ));
}

// ---------------------------------------------------------------------------
// align_planar: normalized-space -> pixel-space matrix rescaling.
// ---------------------------------------------------------------------------

/// A fixed normalized-space affine (small rotation + translation), returned
/// unconditionally regardless of input, so the test can hand-compute the
/// expected pixel-space matrix independently of any learned behaviour.
#[rustfmt::skip]
const FIXED_NORMALIZED_MATRIX: [f32; 9] = [
    0.9,  -0.1,  0.05,
    0.1,   0.95, -0.02,
    0.0,   0.0,   1.0,
];

/// Test-only [`BatchAlignmentModel`] stub: ignores its input and always
/// returns [`FIXED_NORMALIZED_MATRIX`] for every frame.
struct FixedAlignModel;

impl<B: Backend> BatchAlignmentModel<B> for FixedAlignModel {
    fn align_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4> {
        let [n, s, _c, _h, _w] = stack.dims();
        let dev = stack.device();
        let base =
            Tensor::<B, 1>::from_data(TensorData::new(FIXED_NORMALIZED_MATRIX.to_vec(), [9]), &dev)
                .reshape([1, 1, 3, 3]);
        base.expand([n, s, 3, 3])
    }
}

/// Multiply two row-major 3x3 matrices (flat `[f32; 9]`).
// plain sum-of-products reads clearer than mul_add chains here
fn matmul3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0_f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            let mut acc = 0.0_f32;
            for k in 0..3 {
                acc = a[r * 3 + k].mul_add(b[k * 3 + c], acc);
            }
            out[r * 3 + c] = acc;
        }
    }
    out
}

/// Invert the specific upper-triangular-plus-translation-style 3x3
/// `normalize_matrix` form `[[sx,0,-1],[0,sy,-1],[0,0,1]]` analytically —
/// mirrors `bridge::normalize_matrix`'s construction so the test can compute
/// `N` and `N^-1` independently of the crate's (private) implementation.
// test-only image dims, always tiny (< 512 px)
fn normalize_matrix_and_inverse(w: usize, h: usize) -> ([f32; 9], [f32; 9]) {
    let sx = 2.0_f32 / (w as f32 - 1.0);
    let sy = 2.0_f32 / (h as f32 - 1.0);
    #[rustfmt::skip]
    let n = [
        sx,  0.0, -1.0,
        0.0, sy,  -1.0,
        0.0, 0.0,  1.0,
    ];
    let ix = (w as f32 - 1.0) / 2.0;
    let iy = (h as f32 - 1.0) / 2.0;
    #[rustfmt::skip]
    let n_inv = [
        ix,  0.0, ix,
        0.0, iy,  iy,
        0.0, 0.0, 1.0,
    ];
    (n, n_inv)
}

/// [`align_planar`] must convert the model's normalized-`[-1,1]^2`-space
/// output to pixel space via `M_px = N^-1 . M_n . N`, where `N` is built from
/// the FULL-resolution image dimensions. Uses a non-square image (so the
/// anisotropic `sx`/`sy` scale factors actually matter) small enough
/// (< `ALIGN_DOWNSCALE_LONG_SIDE` = 512) that `align_planar` performs no
/// downscaling, keeping the math exact against the full-resolution `N`.
#[test]
fn align_planar_rescales_normalized_matrix_to_pixel_space() {
    let dev = burn::prelude::Device::<B>::default();
    let model = FixedAlignModel;
    let (w, h) = (64, 40); // non-square, well under the 512px downscale threshold

    let frames = vec![planar(w, h, 0.0), planar(w, h, 1.0)];
    let matrices = align_planar::<B, _>(&model, &frames, &dev).expect("align_planar failed");
    assert_eq!(matrices.len(), 2);

    let (n, n_inv) = normalize_matrix_and_inverse(w, h);
    let expected_flat = matmul3(&matmul3(&n_inv, &FIXED_NORMALIZED_MATRIX), &n);

    for m_px in &matrices {
        for r in 0..3 {
            for c in 0..3 {
                let got = m_px[(r, c)];
                let want = expected_flat[r * 3 + c];
                assert!(
                    (got - want).abs() < 1e-3,
                    "matrix[{r}][{c}]: got {got}, expected {want}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// align_planar_pairwise: SAME coordinate contract as align_planar, but via
// the streaming PairAlignmentModel path.
// ---------------------------------------------------------------------------

/// Test-only [`PairAlignmentModel`] stub: ignores its inputs and always
/// returns [`FIXED_NORMALIZED_MATRIX`] for every reference/frame pair.
struct FixedPairAlignModel;

impl<B: Backend> PairAlignmentModel<B> for FixedPairAlignModel {
    fn align_pair(&self, reference: Tensor<B, 4>, _frame: Tensor<B, 4>) -> Tensor<B, 3> {
        let [n, _c, _h, _w] = reference.dims();
        let dev = reference.device();
        let base =
            Tensor::<B, 1>::from_data(TensorData::new(FIXED_NORMALIZED_MATRIX.to_vec(), [9]), &dev)
                .reshape([1, 3, 3]);
        base.expand([n, 3, 3])
    }
}

/// [`align_planar_pairwise`] must apply the IDENTICAL `M_px = N^-1 . M_n . N`
/// conjugation as [`align_planar`] (see that test's docs above) — same
/// non-square image, same fixed normalized-space matrix, same expected
/// pixel-space result, just driven through the streaming per-frame
/// [`PairAlignmentModel`] path instead of one whole-stack
/// [`BatchAlignmentModel`] call.
#[test]
fn align_planar_pairwise_rescales_normalized_matrix_to_pixel_space() {
    let dev = burn::prelude::Device::<B>::default();
    let model = FixedPairAlignModel;
    let (w, h) = (64, 40); // non-square, well under the 512px downscale threshold

    let frames = vec![planar(w, h, 0.0), planar(w, h, 1.0), planar(w, h, 2.0)];
    let matrices =
        align_planar_pairwise::<B, _>(&model, &frames, &dev).expect("align_planar_pairwise failed");
    assert_eq!(
        matrices.len(),
        3,
        "one matrix per frame, including the reference itself"
    );

    let (n, n_inv) = normalize_matrix_and_inverse(w, h);
    let expected_flat = matmul3(&matmul3(&n_inv, &FIXED_NORMALIZED_MATRIX), &n);

    for m_px in &matrices {
        for r in 0..3 {
            for c in 0..3 {
                let got = m_px[(r, c)];
                let want = expected_flat[r * 3 + c];
                assert!(
                    (got - want).abs() < 1e-3,
                    "matrix[{r}][{c}]: got {got}, expected {want}"
                );
            }
        }
    }
}

/// An empty stack must be rejected identically to [`align_planar`]'s
/// contract.
#[test]
fn align_planar_pairwise_empty_is_error() {
    let dev = burn::prelude::Device::<B>::default();
    let model = FixedPairAlignModel;
    let frames: Vec<PlanarImage<f32>> = Vec::new();
    assert!(matches!(
        align_planar_pairwise::<B, _>(&model, &frames, &dev),
        Err(BridgeError::EmptyStack)
    ));
}

/// A shape-mismatched frame must be rejected identically to
/// [`align_planar`]'s contract.
#[test]
fn align_planar_pairwise_shape_mismatch_is_error() {
    let dev = burn::prelude::Device::<B>::default();
    let model = FixedPairAlignModel;
    let frames = vec![planar(16, 16, 0.0), planar(16, 12, 1.0)];
    assert!(matches!(
        align_planar_pairwise::<B, _>(&model, &frames, &dev),
        Err(BridgeError::ShapeMismatch { index: 1, .. })
    ));
}
