//! Bridge between the pipeline's [`PlanarImage<f32>`] and the network's
//! linear-RGB tensors: [`fuse_planar`] (fusion strategies), [`align_planar`]
//! (batch alignment), and [`align_planar_pairwise`] (pairwise alignment) are
//! the high-level entry points the CLI/GUI call to run a trained model as a
//! stacking/alignment algorithm.
//!
//! The pipeline stores frames as **gamma-encoded (sRGB) BT.601 `Y`/`Cb`/`Cr`**
//! planes (`luma`/`chroma_a`/`chroma_b`, see [`PlanarImage`]'s docs), while
//! the network operates in **linear RGB** — it was trained on linearised
//! data (see [`crate::data::to_linear`] / the dataset loader's
//! `load_rgb_linear`). This bridge owns the conversion in both directions:
//! `planar_to_tensor` applies the YCbCr→RGB matrix and then linearises each
//! channel with [`crate::data::to_linear`]; `tensor_to_planar` reverses that
//! with [`crate::data::from_linear`] before the RGB→YCbCr matrix. The model
//! never sees gamma-encoded values at inference time.
//!
//! ## Alignment coordinate convention
//!
//! [`align_planar`] and [`align_planar_pairwise`] additionally bridge a
//! coordinate system, and share the exact same contract: the alignment
//! model (see [`crate::traits::BatchAlignmentModel`] /
//! [`crate::traits::PairAlignmentModel`]) is trained on small, downscaled
//! frames and predicts matrices in NORMALIZED `[-1, 1]²` coordinates
//! (matching [`crate::loss::CornerAlignmentLoss`]'s /
//! [`crate::loss::PairCornerAlignmentLoss`]'s convention), while every other
//! part of this crate — and the pipeline callers of either bridge function —
//! works in full-resolution PIXEL coordinates. See [`align_planar`]'s own
//! docs for the exact conjugation used to convert between the two;
//! [`align_planar_pairwise`] applies the identical conjugation and only
//! describes where it differs (the per-frame streaming loop in place of one
//! whole-stack batch call).

#![allow(clippy::suboptimal_flops, clippy::many_single_char_names)]

use burn::{
    prelude::Backend,
    tensor::{
        Tensor, TensorData,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use stacker_core::image::PlanarImage;

use crate::{
    data::{from_linear, to_linear},
    infer::{InferError, TileConfig, fuse_stack, fuse_stack_with},
    traits::FusionModel,
};

/// Target size (longest side, in pixels) the alignment model's input stack is
/// downscaled to by [`align_planar`] before inference. Matches the scale the
/// alignment training data is expected to run at (see the crate README) —
/// registration only needs coarse spatial context, and running the encoder at
/// full resolution would be needlessly expensive.
const ALIGN_DOWNSCALE_LONG_SIDE: usize = 512;

/// Errors from the planar fusion bridge.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// No frames were supplied.
    #[error("empty stack: need at least one frame")]
    EmptyStack,
    /// A frame's dimensions differ from frame 0.
    #[error("frame {index} is {got_w}x{got_h}, expected {exp_w}x{exp_h}")]
    ShapeMismatch {
        /// Offending frame index.
        index: usize,
        /// Expected width (frame 0).
        exp_w: usize,
        /// Expected height (frame 0).
        exp_h: usize,
        /// Actual width.
        got_w: usize,
        /// Actual height.
        got_h: usize,
    },
    /// The underlying tiled inference failed.
    #[error(transparent)]
    Infer(#[from] InferError),
}

// --- BT.601 (gamma-encoded) YCbCr <-> linear RGB --------------------------

/// Gamma-encoded `Y`/`Cb`/`Cr` → gamma-encoded `R`/`G`/`B` (BT.601).
#[inline]
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let r = y + 1.402 * cr;
    let g = y - 0.344_136 * cb - 0.714_136 * cr;
    let b = y + 1.772 * cb;
    (r, g, b)
}

/// Linear `R`/`G`/`B` → linear `Y`/`Cb`/`Cr` (BT.601), the inverse of
/// [`ycbcr_to_rgb`] (up to the transfer function applied by the caller).
#[inline]
fn rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    (y, cb, cr)
}

/// Convert a planar (gamma-encoded YCbCr) image to a `[3, H, W]` linear-RGB
/// tensor: BT.601 YCbCr→RGB matrix first, then [`to_linear`] per channel
/// (clamped to `[0, 1]`) to match the model's training data.
fn planar_to_tensor<B: Backend>(img: &PlanarImage<f32>, device: &B::Device) -> Tensor<B, 3> {
    let (w, h) = (img.width, img.height);
    let np = w * h;
    let mut data = vec![0f32; 3 * np];
    for i in 0..np {
        let (r, g, b) = ycbcr_to_rgb(img.luma[i], img.chroma_a[i], img.chroma_b[i]);
        data[i] = to_linear(r.clamp(0.0, 1.0));
        data[np + i] = to_linear(g.clamp(0.0, 1.0));
        data[2 * np + i] = to_linear(b.clamp(0.0, 1.0));
    }
    Tensor::from_data(TensorData::new(data, [3, h, w]), device)
}

/// Convert a `[3, H, W]` linear-RGB tensor back to a planar (gamma-encoded
/// YCbCr) image: [`from_linear`] per channel (clamped to `[0, 1]`) first,
/// then the BT.601 RGB→YCbCr matrix.
fn tensor_to_planar<B: Backend>(t: Tensor<B, 3>, w: usize, h: usize) -> PlanarImage<f32> {
    let np = w * h;
    let data: Vec<f32> = t
        .into_data()
        .to_vec::<f32>()
        .expect("tensor element type is f32");
    let mut img = PlanarImage::new(w, h);
    for i in 0..np {
        let r = from_linear(data[i].clamp(0.0, 1.0));
        let g = from_linear(data[np + i].clamp(0.0, 1.0));
        let b = from_linear(data[2 * np + i].clamp(0.0, 1.0));
        let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
        img.luma[i] = y;
        img.chroma_a[i] = cb;
        img.chroma_b[i] = cr;
    }
    img
}

/// Fuse a focus stack with a trained [`FusionModel`], returning a
/// single all-in-focus [`PlanarImage`] in the pipeline's colour space.
///
/// `model` may be the built-in [`crate::model::FocusMergeNet`] or a
/// third-party architecture. `frames` must be non-empty and all share frame
/// 0's dimensions. Inference is tiled per [`TileConfig`], so arbitrarily
/// large images are supported. Use [`TileConfig::for_model`] to size `cfg`
/// from the model's own [`FusionModel::receptive_field`].
pub fn fuse_planar<B: Backend, M: FusionModel<B>>(
    model: &M,
    frames: &[PlanarImage<f32>],
    cfg: TileConfig,
    device: &B::Device,
) -> Result<PlanarImage<f32>, BridgeError> {
    if frames.is_empty() {
        return Err(BridgeError::EmptyStack);
    }
    let (w, h) = (frames[0].width, frames[0].height);
    for (index, f) in frames.iter().enumerate().skip(1) {
        if f.width != w || f.height != h {
            return Err(BridgeError::ShapeMismatch {
                index,
                exp_w: w,
                exp_h: h,
                got_w: f.width,
                got_h: f.height,
            });
        }
    }

    let tensors: Vec<Tensor<B, 3>> = frames
        .iter()
        .map(|f| planar_to_tensor::<B>(f, device))
        .collect();

    let merged = fuse_stack(model, &tensors, cfg, device)?;
    Ok(tensor_to_planar::<B>(merged, w, h))
}

/// Like [`fuse_planar`], but calls `on_step(frame_index, &running_composite)`
/// after each frame is folded in — for a live "watch it sharpen" preview.
pub fn fuse_planar_with_progress<B: Backend, M: FusionModel<B>, F>(
    model: &M,
    frames: &[PlanarImage<f32>],
    cfg: TileConfig,
    device: &B::Device,
    mut on_step: F,
) -> Result<PlanarImage<f32>, BridgeError>
where
    F: FnMut(usize, &PlanarImage<f32>),
{
    if frames.is_empty() {
        return Err(BridgeError::EmptyStack);
    }
    let (w, h) = (frames[0].width, frames[0].height);
    for (index, f) in frames.iter().enumerate().skip(1) {
        if f.width != w || f.height != h {
            return Err(BridgeError::ShapeMismatch {
                index,
                exp_w: w,
                exp_h: h,
                got_w: f.width,
                got_h: f.height,
            });
        }
    }

    let tensors: Vec<Tensor<B, 3>> = frames
        .iter()
        .map(|f| planar_to_tensor::<B>(f, device))
        .collect();

    let merged = fuse_stack_with(model, &tensors, cfg, device, |index, running| {
        let planar = tensor_to_planar::<B>(running, w, h);
        on_step(index, &planar);
    })?;
    Ok(tensor_to_planar::<B>(merged, w, h))
}

/// Build the pixel-homogeneous → normalized-homogeneous conjugation matrix
/// `N` for a `w × h` image, using the crate-wide convention `x_n = 2x/(W-1) - 1`,
/// `y_n = 2y/(H-1) - 1` (pixel `(0,0) -> (-1,-1)`, pixel `(W-1,H-1) -> (1,1)`
/// — the same four corners [`crate::loss::CornerAlignmentLoss`] supervises).
///
/// `w`/`h` must be `>= 2`.
fn normalize_matrix(w: usize, h: usize) -> nalgebra::Matrix3<f32> {
    let sx = 2.0_f32 / (w as f32 - 1.0);
    let sy = 2.0_f32 / (h as f32 - 1.0);
    #[rustfmt::skip]
    let n = nalgebra::Matrix3::new(
        sx,  0.0, -1.0,
        0.0, sy,  -1.0,
        0.0, 0.0,  1.0,
    );
    n
}

/// Run a trained [`crate::traits::BatchAlignmentModel`] over a focus stack,
/// returning one PIXEL-space affine registration matrix per frame.
///
/// ## Pipeline
///
/// 1. Convert every frame to a linear-RGB tensor (same colour-space bridge as
///    [`fuse_planar`]).
/// 2. Downscale the whole stack so its longest side is
///    [`ALIGN_DOWNSCALE_LONG_SIDE`] (bilinear, aspect-preserving) — the
///    alignment model is trained at this scale; registration only needs
///    coarse structure, and this keeps `align_batch`'s memory/compute cost
///    small regardless of the input resolution.
/// 3. Run [`crate::traits::BatchAlignmentModel::align_batch`] on the
///    downscaled stack, producing matrices in NORMALIZED `[-1, 1]²`
///    coordinates (see the module docs' "Alignment coordinate convention").
/// 4. Convert each matrix back to PIXEL-space coordinates of the frame's
///    FULL `W × H` resolution (not the downscaled size) via the similarity
///    conjugation `M_px = N⁻¹ · M_n · N`, where `N` is [`normalize_matrix`]
///    for the full-resolution `(W, H)`:
///
///    ```text
///    N = [ 2/(W-1)    0      -1 ]      N⁻¹ = [ (W-1)/2    0     (W-1)/2 ]
///        [   0     2/(H-1)   -1 ]            [   0     (H-1)/2  (H-1)/2 ]
///        [   0        0       1 ]            [   0        0        1   ]
///    ```
///
///    Using the full-resolution `N` here (rather than the downscaled size)
///    is what makes the returned matrices directly usable against the
///    original, full-resolution frames — the model's normalized-space output
///    is resolution-independent by construction, so re-anchoring it to
///    whatever pixel grid the caller cares about is exactly this conjugation.
///
/// # Panics
///
/// Panics if `w < 2` or `h < 2` for any frame (a single-row/column image has
/// no well-defined normalized extent — see [`normalize_matrix`]).
pub fn align_planar<B: Backend, M: crate::traits::BatchAlignmentModel<B>>(
    model: &M,
    frames: &[PlanarImage<f32>],
    device: &B::Device,
) -> Result<Vec<nalgebra::Matrix3<f32>>, BridgeError> {
    if frames.is_empty() {
        return Err(BridgeError::EmptyStack);
    }
    let (w, h) = (frames[0].width, frames[0].height);
    assert!(
        w >= 2 && h >= 2,
        "align_planar: frames must be at least 2x2, got {w}x{h}"
    );
    for (index, f) in frames.iter().enumerate().skip(1) {
        if f.width != w || f.height != h {
            return Err(BridgeError::ShapeMismatch {
                index,
                exp_w: w,
                exp_h: h,
                got_w: f.width,
                got_h: f.height,
            });
        }
    }

    let s = frames.len();
    let mut stack = Vec::with_capacity(s);
    for f in frames {
        let t = planar_to_tensor::<B>(f, device);
        stack.push(t.unsqueeze_dim::<4>(0).unsqueeze_dim::<5>(0));
    }
    let stack_tensor = Tensor::cat(stack, 1); // [1, S, 3, H, W]

    // Downscale (aspect-preserving, longest side -> ALIGN_DOWNSCALE_LONG_SIDE)
    // before running the model — see this function's docs.
    let long_side = w.max(h);
    let (dl_h, dl_w) = if long_side <= ALIGN_DOWNSCALE_LONG_SIDE {
        (h, w)
    } else {
        let scale = ALIGN_DOWNSCALE_LONG_SIDE as f32 / long_side as f32;
        (
            ((h as f32 * scale).round() as usize).max(2),
            ((w as f32 * scale).round() as usize).max(2),
        )
    };
    let stack_small = if (dl_h, dl_w) == (h, w) {
        stack_tensor
    } else {
        let flat = stack_tensor.reshape([s, 3, h, w]);
        let resized = interpolate(
            flat,
            [dl_h, dl_w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        resized.reshape([1, s, 3, dl_h, dl_w])
    };

    let out = model.align_batch(stack_small); // [1, S, 3, 3], normalized-space
    let out_data: Vec<f32> = out
        .into_data()
        .to_vec::<f32>()
        .expect("tensor element type is f32");

    // Full-resolution normalize matrix and its inverse (constant across
    // frames since every frame shares (w, h)).
    let n = normalize_matrix(w, h);
    let n_inv = n
        .try_inverse()
        .expect("normalize_matrix is always invertible for w,h >= 2");

    let mut matrices = Vec::with_capacity(s);
    for i in 0..s {
        let base = i * 9;
        #[rustfmt::skip]
        let m_n = nalgebra::Matrix3::new(
            out_data[base],     out_data[base + 1], out_data[base + 2],
            out_data[base + 3], out_data[base + 4], out_data[base + 5],
            out_data[base + 6], out_data[base + 7], out_data[base + 8],
        );
        // M_px = N^-1 * M_n * N
        let m_px = n_inv * m_n * n;
        matrices.push(m_px);
    }

    Ok(matrices)
}

/// Run a trained [`crate::traits::PairAlignmentModel`] over a focus stack,
/// returning one PIXEL-space affine registration matrix per frame.
///
/// The streaming sibling of [`align_planar`] — SAME coordinate contract as
/// that function (see its docs for the full derivation): matrices come out
/// of the model in NORMALIZED `[-1, 1]²` coordinates and are converted to
/// PIXEL-space coordinates of the frame's FULL `W × H` resolution via the
/// identical similarity conjugation `M_px = N⁻¹ · M_n · N`, where `N` is
/// [`normalize_matrix`] for the full-resolution `(W, H)`.
///
/// ## Pipeline
///
/// 1. Convert frame 0 (the reference) to a linear-RGB tensor (same
///    colour-space bridge as [`fuse_planar`]/[`align_planar`]).
/// 2. Downscale the reference so its longest side is
///    [`ALIGN_DOWNSCALE_LONG_SIDE`] (bilinear, aspect-preserving) — same
///    target size [`align_planar`] uses, so a `FusionAlignNet` and a
///    `BatchAlignNet` trained at the same scale are interchangeable at this
///    bridge boundary.
/// 3. For EACH frame (including frame 0 itself, which is registered against
///    its own downscaled copy and is expected to come back near-identity —
///    see [`crate::model::FusionAlignNet`]'s docs for why this is not
///    guaranteed to be EXACT identity the way `align_planar`'s frame 0 is):
///    convert to a linear-RGB tensor, downscale it independently (the exact
///    same aspect-preserving resize as the reference), and call
///    [`crate::traits::PairAlignmentModel::align_pair`] with `(N=1)`
///    reference/frame tensors. This is a genuine per-frame streaming loop —
///    unlike [`align_planar`], no `[1, S, 3, H, W]` whole-stack tensor is
///    ever constructed, so memory stays O(1) in stack size `S` rather than
///    O(S) (see [`crate::traits::PairAlignmentModel`]'s docs). Downscaling
///    each frame independently inside the loop (rather than once up front)
///    produces IDENTICAL numeric results to downscaling the whole stack at
///    once, since resizing is a per-frame-independent operation — so nothing
///    is sacrificed for the streaming-memory property.
/// 4. Convert each returned `[1, 3, 3]` matrix back to PIXEL-space
///    coordinates of the FULL (not downscaled) resolution via the same
///    `M_px = N⁻¹ · M_n · N` conjugation [`align_planar`] uses, with `N`
///    built from the frame's full-resolution `(W, H)` (shared across all
///    frames, since every frame is validated to share frame 0's dimensions).
///
/// # Panics
///
/// Panics if `w < 2` or `h < 2` for any frame (a single-row/column image has
/// no well-defined normalized extent — see [`normalize_matrix`]), matching
/// [`align_planar`]'s panic contract.
pub fn align_planar_pairwise<B: Backend, M: crate::traits::PairAlignmentModel<B>>(
    model: &M,
    frames: &[PlanarImage<f32>],
    device: &B::Device,
) -> Result<Vec<nalgebra::Matrix3<f32>>, BridgeError> {
    if frames.is_empty() {
        return Err(BridgeError::EmptyStack);
    }
    let (w, h) = (frames[0].width, frames[0].height);
    assert!(
        w >= 2 && h >= 2,
        "align_planar_pairwise: frames must be at least 2x2, got {w}x{h}"
    );
    for (index, f) in frames.iter().enumerate().skip(1) {
        if f.width != w || f.height != h {
            return Err(BridgeError::ShapeMismatch {
                index,
                exp_w: w,
                exp_h: h,
                got_w: f.width,
                got_h: f.height,
            });
        }
    }

    // Aspect-preserving downscale target shared by every frame (and the
    // reference) — identical sizing rule to `align_planar`'s whole-stack
    // downscale, just applied per-frame here.
    let long_side = w.max(h);
    let (dl_h, dl_w) = if long_side <= ALIGN_DOWNSCALE_LONG_SIDE {
        (h, w)
    } else {
        let scale = ALIGN_DOWNSCALE_LONG_SIDE as f32 / long_side as f32;
        (
            ((h as f32 * scale).round() as usize).max(2),
            ((w as f32 * scale).round() as usize).max(2),
        )
    };

    let downscale_one = |img: &PlanarImage<f32>| -> Tensor<B, 4> {
        let t = planar_to_tensor::<B>(img, device).unsqueeze_dim::<4>(0); // [1,3,H,W]
        if (dl_h, dl_w) == (h, w) {
            t
        } else {
            interpolate(
                t,
                [dl_h, dl_w],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            )
        }
    };

    let reference_small = downscale_one(&frames[0]);

    // Full-resolution normalize matrix and its inverse (constant across
    // frames since every frame shares (w, h)) — identical to `align_planar`.
    let n = normalize_matrix(w, h);
    let n_inv = n
        .try_inverse()
        .expect("normalize_matrix is always invertible for w,h >= 2");

    let mut matrices = Vec::with_capacity(frames.len());
    for frame in frames {
        let frame_small = downscale_one(frame);
        let out = model.align_pair(reference_small.clone(), frame_small); // [1,3,3], normalized-space
        let out_data: Vec<f32> = out
            .into_data()
            .to_vec::<f32>()
            .expect("tensor element type is f32");

        #[rustfmt::skip]
        let m_n = nalgebra::Matrix3::new(
            out_data[0], out_data[1], out_data[2],
            out_data[3], out_data[4], out_data[5],
            out_data[6], out_data[7], out_data[8],
        );
        // M_px = N^-1 * M_n * N — same conjugation as `align_planar`.
        let m_px = n_inv * m_n * n;
        matrices.push(m_px);
    }

    Ok(matrices)
}

#[cfg(test)]
mod tests {
    use super::{planar_to_tensor, tensor_to_planar};
    use burn::backend::NdArray;
    use stacker_core::image::PlanarImage;

    type B = NdArray;

    /// A round trip planar → tensor → planar on synthetic gamma-encoded
    /// YCbCr data should be near-identity: the only lossy step is the
    /// `to_linear`/`from_linear` transfer-function pair, which round-trips
    /// to within float precision (not exactly, since it isn't a linear map).
    ///
    /// The synthetic samples are built from valid in-gamut RGB triples (not
    /// arbitrary `Y`/`Cb`/`Cr` combinations, which can encode out-of-`[0,1]`
    /// RGB and make the `clamp` in `planar_to_tensor`/`tensor_to_planar`
    /// legitimately lossy) so the round trip is expected to be near-exact.
    #[test]
    fn planar_tensor_round_trip_is_near_identity() {
        const TOL: f32 = 1e-3;
        let dev = burn::prelude::Device::<B>::default();
        let (w, h) = (6, 5);
        let mut img = PlanarImage::<f32>::new(w, h);
        for i in 0..w * h {
            let t = (i as f32 * 0.37).fract();
            // In-gamut gamma-encoded RGB triple, converted to Y/Cb/Cr with
            // the same BT.601 matrix `rgb_to_ycbcr` uses, so `ycbcr_to_rgb`
            // reconstructs an RGB triple that is already within [0, 1].
            let r = 0.1 + 0.8 * t;
            let g = 0.9 - 0.7 * t;
            let b = 0.2 + 0.5 * ((t * 3.1).fract());
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let cb = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
            let cr = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
            img.luma[i] = y;
            img.chroma_a[i] = cb;
            img.chroma_b[i] = cr;
        }

        let tensor = planar_to_tensor::<B>(&img, &dev);
        let round_tripped = tensor_to_planar::<B>(tensor, w, h);

        for i in 0..w * h {
            assert!(
                (round_tripped.luma[i] - img.luma[i]).abs() < TOL,
                "luma[{i}]: {} vs {}",
                round_tripped.luma[i],
                img.luma[i]
            );
            assert!(
                (round_tripped.chroma_a[i] - img.chroma_a[i]).abs() < TOL,
                "chroma_a[{i}]: {} vs {}",
                round_tripped.chroma_a[i],
                img.chroma_a[i]
            );
            assert!(
                (round_tripped.chroma_b[i] - img.chroma_b[i]).abs() < TOL,
                "chroma_b[{i}]: {} vs {}",
                round_tripped.chroma_b[i],
                img.chroma_b[i]
            );
        }
    }
}
