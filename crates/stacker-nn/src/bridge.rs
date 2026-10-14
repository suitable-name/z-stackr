//! Bridge between the pipeline's [`PlanarImage<f32>`] and the networks' linear-RGB
//! tensors: the high-level entry points the CLI/GUI call to run a trained model as
//! a fusion algorithm ([`fuse_planar`]) or as an aligner ([`align_thumbs`],
//! [`align_thumbs_pairwise`], [`align_planar`]).
//!
//! The pipeline stores frames as **linear-light BT.601 `Y`/`Cb`/`Cr`** planes
//! (`luma`/`chroma_a`/`chroma_b`), and the networks operate in **linear RGB**.
//! Both spaces are linear, so conversion here is a pure 3×3 colour-matrix round
//! trip through `stacker_core::color` with no gamma handling — the model never
//! sees sRGB at inference time.
//!
//! # Alignment coordinate conventions
//!
//! Every alignment matrix in this crate maps **reference coordinates to source
//! (frame) coordinates**: `p_frame = M · p_reference`. This is the convention of
//! the pipeline (a refined matrix maps reference to source; the apps warp with the
//! inverse), of [`crate::warp::warp_affine`] and of the training data, so a network
//! matrix is directly usable as a seed for the classical refiner.
//!
//! Three coordinate systems are involved, and getting the *image the matrix is
//! relative to* wrong is the classic bug (the original code normalised ground-truth
//! matrices by the full frame while the network saw 256 px crops):
//!
//! 1. **Pixel coordinates** `(x, y)` of some image: integer indices are pixel
//!    centres, the first pixel is `0` and the last is `W - 1` (the convention of
//!    `stacker_align`'s warp).
//! 2. **Normalised coordinates** of the image *the network is given*:
//!    `x_n = 2x/(W-1) - 1`, `y_n = 2y/(H-1) - 1`, i.e. the four corner pixel
//!    centres map to `(±1, ±1)`. [`pixel_matrix_to_normalized`] /
//!    [`normalized_matrix_to_pixel`] convert with `M_n = N · M · N⁻¹`.
//! 3. **Full-resolution pixel coordinates** of the frame on disk. The network is
//!    run on **thumbnails** (longest side ≤ [`ALIGN_DOWNSCALE_LONG_SIDE`] = 512 px,
//!    made by [`make_thumbnail`], an area resample), and a thumbnail pixel `i`
//!    corresponds to full-resolution pixel `(i + ½)·W/tw − ½`. [`FrameThumbs`]
//!    carries both sizes and converts in either direction.
//!
//! **Rule for training data.** The matrix a network is trained against must be
//! expressed relative to exactly the pixels it is shown: re-express a
//! full-frame pixel matrix `M` for a crop with top-left corner `o` (same crop of
//! reference and frame) as `T(-o) · M · T(o)` ([`crop_pixel_matrix`]), then
//! normalise with the *crop's* size ([`pixel_matrix_to_normalized`]). Normalising a
//! crop's ground truth by the full frame's size is a different transform and
//! trains the head towards the wrong answer. At inference the thumbnail is the
//! whole frame, so the normalised matrix is resolution independent and
//! [`FrameThumbs::normalized_to_full`] returns the full-resolution pixel matrix.

use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use stacker_core::{
    color::{linear_rgb_to_planar, planar_to_linear_rgb},
    image::PlanarImage,
};

use crate::{
    infer::{InferError, TileConfig, fuse_stream_with},
    traits::{BatchAlignmentModel, FusionModel, PairAlignmentModel},
};

/// Longest side, in pixels, of the thumbnails the alignment networks run on.
///
/// Registration only needs coarse structure, and this bounds both the compute and
/// the memory of the neural pre-pass regardless of the input resolution.
pub const ALIGN_DOWNSCALE_LONG_SIDE: usize = 512;

/// A `3 × 3` homogeneous matrix, row-major (`m[row][column]`), in `f32`.
pub type PixelMatrix = [[f32; 3]; 3];

/// Errors from the planar bridge.
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
    /// A frame is too small to have a normalised extent (alignment needs at least
    /// 2 × 2 pixels).
    #[error("frames must be at least 2x2 pixels for alignment, got {width}x{height}")]
    TooSmall {
        /// Frame width.
        width: usize,
        /// Frame height.
        height: usize,
    },
    /// The model returned data that is not a list of `f32` matrices.
    #[error("unexpected model output: {0}")]
    BadOutput(String),
    /// The underlying tiled inference failed.
    #[error(transparent)]
    Infer(#[from] InferError),
}

// ---------------------------------------------------------------------------
// Planar <-> tensor
// ---------------------------------------------------------------------------

// The BT.601 (linear) YCbCr <-> linear RGB matrices are the core's own
// (`stacker_core::color::{planar_to_linear_rgb, linear_rgb_to_planar}`), so the
// network sees exactly the colours the rest of the pipeline stores.

/// Convert a planar (linear YCbCr) image to a `[3, H, W]` linear-RGB tensor.
#[expect(
    clippy::many_single_char_names,
    reason = "w/h and r/g/b are the conventional names for size and colour channels"
)]
fn planar_to_tensor<B: Backend>(img: &PlanarImage<f32>, device: &B::Device) -> Tensor<B, 3> {
    let (w, h) = (img.width, img.height);
    let np = w * h;
    let mut data = vec![0f32; 3 * np];
    for i in 0..np {
        let [r, g, b] = planar_to_linear_rgb([img.luma[i], img.chroma_a[i], img.chroma_b[i]]);
        data[i] = r;
        data[np + i] = g;
        data[2 * np + i] = b;
    }
    Tensor::from_data(TensorData::new(data, [3, h, w]), device)
}

/// Convert a `[3, H, W]` linear-RGB tensor back to a planar (linear YCbCr) image.
fn tensor_to_planar<B: Backend>(t: Tensor<B, 3>, w: usize, h: usize) -> PlanarImage<f32> {
    let np = w * h;
    let data: Vec<f32> = t
        .into_data()
        .to_vec::<f32>()
        .expect("tensor element type is f32");
    let mut img = PlanarImage::new(w, h);
    for i in 0..np {
        let [y, cb, cr] = linear_rgb_to_planar([data[i], data[np + i], data[2 * np + i]]);
        img.luma[i] = y;
        img.chroma_a[i] = cb;
        img.chroma_b[i] = cr;
    }
    img
}

/// Check that `frames` is non-empty and every frame has frame 0's size; returns
/// that size as `(width, height)`.
fn check_frames(frames: &[PlanarImage<f32>]) -> Result<(usize, usize), BridgeError> {
    let first = frames.first().ok_or(BridgeError::EmptyStack)?;
    let (w, h) = (first.width, first.height);
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
    Ok((w, h))
}

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

/// Fuse a focus stack with a trained [`FusionModel`], returning a single
/// all-in-focus [`PlanarImage`] in the pipeline's colour space.
///
/// `frames` must be non-empty and all share frame 0's dimensions. Inference is
/// tiled per [`TileConfig`], so arbitrarily large images are supported; build `cfg`
/// with [`TileConfig::for_model`] or [`TileConfig::for_radius`].
///
/// A pairwise model ([`crate::traits::FusionStrategy::Pairwise`]) converts each
/// frame to a tensor only when it is folded into the running composite, so memory
/// holds the caller's planar frames plus the composite and one converted frame. A
/// batch model needs the whole stack as tensors (`S · 3 · H · W` f32) because each
/// tile crops every frame.
///
/// # Errors
/// [`BridgeError::EmptyStack`] for no frames, [`BridgeError::ShapeMismatch`] when
/// a frame differs in size from frame 0, and [`BridgeError::Infer`] when the
/// tiled inference fails.
pub fn fuse_planar<B: Backend>(
    model: &impl FusionModel<B>,
    frames: &[PlanarImage<f32>],
    cfg: TileConfig,
    device: &B::Device,
) -> Result<PlanarImage<f32>, BridgeError> {
    let (w, h) = check_frames(frames)?;

    // Converted lazily, one frame per fold step.
    let tensors = frames.iter().map(|f| planar_to_tensor::<B>(f, device));

    let merged = fuse_stream_with(model, tensors, cfg, device, |_, _| {})?;
    Ok(tensor_to_planar::<B>(merged, w, h))
}

/// Like [`fuse_planar`], but calls `on_step(index, &running_composite)` after each
/// progress step.
///
/// This feeds a live "watch it sharpen" preview. For a pairwise model the index
/// is the frame just folded in; for a batch model it is the completed tile row
/// (see [`crate::infer::fuse_batch_tiled`]).
///
/// # Errors
/// As [`fuse_planar`].
pub fn fuse_planar_with_progress<B: Backend, F>(
    model: &impl FusionModel<B>,
    frames: &[PlanarImage<f32>],
    cfg: TileConfig,
    device: &B::Device,
    mut on_step: F,
) -> Result<PlanarImage<f32>, BridgeError>
where
    F: FnMut(usize, &PlanarImage<f32>),
{
    let (w, h) = check_frames(frames)?;

    let tensors = frames.iter().map(|f| planar_to_tensor::<B>(f, device));

    let merged = fuse_stream_with(model, tensors, cfg, device, |index, running| {
        let planar = tensor_to_planar::<B>(running, w, h);
        on_step(index, &planar);
    })?;
    Ok(tensor_to_planar::<B>(merged, w, h))
}

// ---------------------------------------------------------------------------
// 3x3 matrix helpers (f64 internally)
// ---------------------------------------------------------------------------

type M3 = [[f64; 3]; 3];

#[expect(
    clippy::cast_precision_loss,
    reason = "image sizes are far below 2^52, so the usize to f64 conversion is exact"
)]
const fn dim_f64(n: usize) -> f64 {
    n as f64
}

fn widen(m: PixelMatrix) -> M3 {
    m.map(|row| row.map(f64::from))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "matrix entries are narrowed to the f32 the pipeline stores"
)]
fn narrow(m: M3) -> PixelMatrix {
    m.map(|row| row.map(|v| v as f32))
}

fn mul3(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0].mul_add(b[0][j], a[i][1].mul_add(b[1][j], a[i][2] * b[2][j]));
        }
    }
    out
}

/// `N` (pixel → normalised) and `N⁻¹` for a `w × h` image.
fn normalizer(w: usize, h: usize) -> (M3, M3) {
    let (wm, hm) = (dim_f64(w) - 1.0, dim_f64(h) - 1.0);
    (
        [
            [2.0 / wm, 0.0, -1.0],
            [0.0, 2.0 / hm, -1.0],
            [0.0, 0.0, 1.0],
        ],
        [
            [wm / 2.0, 0.0, wm / 2.0],
            [0.0, hm / 2.0, hm / 2.0],
            [0.0, 0.0, 1.0],
        ],
    )
}

/// Convert a pixel-space matrix of a `w × h` image to the normalised
/// `[-1, 1]²` convention of the network input: `M_n = N · M · N⁻¹`.
///
/// # Panics
///
/// Panics if `w < 2` or `h < 2` (a single row or column has no normalised extent).
#[must_use]
pub fn pixel_matrix_to_normalized(m: PixelMatrix, w: usize, h: usize) -> PixelMatrix {
    assert!(w >= 2 && h >= 2, "image must be at least 2x2, got {w}x{h}");
    let (n, n_inv) = normalizer(w, h);
    narrow(mul3(&mul3(&n, &widen(m)), &n_inv))
}

/// The inverse of [`pixel_matrix_to_normalized`]: `M = N⁻¹ · M_n · N`.
///
/// # Panics
///
/// Panics if `w < 2` or `h < 2`.
#[must_use]
pub fn normalized_matrix_to_pixel(m: PixelMatrix, w: usize, h: usize) -> PixelMatrix {
    assert!(w >= 2 && h >= 2, "image must be at least 2x2, got {w}x{h}");
    let (n, n_inv) = normalizer(w, h);
    narrow(mul3(&mul3(&n_inv, &widen(m)), &n))
}

/// Re-express a pixel-space matrix for a crop whose top-left pixel is `(ox, oy)`
/// of the frame the matrix is relative to (the same crop is taken of reference and
/// frame): `M_crop = T(-o) · M · T(o)`.
///
/// This is the first step of producing ground truth for a training crop; normalise
/// the result with the **crop's** size ([`pixel_matrix_to_normalized`]).
#[must_use]
pub fn crop_pixel_matrix(m: PixelMatrix, ox: f32, oy: f32) -> PixelMatrix {
    let (ox, oy) = (f64::from(ox), f64::from(oy));
    let to = [[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]];
    let from = [[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]];
    narrow(mul3(&mul3(&from, &widen(m)), &to))
}

// ---------------------------------------------------------------------------
// Thumbnails
// ---------------------------------------------------------------------------

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "callers pass non-negative values far below usize::MAX"
)]
const fn index_of(x: f64) -> usize {
    x as usize
}

/// Size `(width, height)` of the thumbnail of a `w × h` frame.
///
/// Unchanged when its longest side is at most [`ALIGN_DOWNSCALE_LONG_SIDE`],
/// otherwise scaled down aspect-preserving so the longest side is that long
/// (each side at least 2).
#[must_use]
pub fn thumbnail_size(w: usize, h: usize) -> (usize, usize) {
    let long = w.max(h);
    if long <= ALIGN_DOWNSCALE_LONG_SIDE {
        return (w, h);
    }
    let scale = dim_f64(ALIGN_DOWNSCALE_LONG_SIDE) / dim_f64(long);
    (
        index_of((dim_f64(w) * scale).round()).max(2),
        index_of((dim_f64(h) * scale).round()).max(2),
    )
}

/// Normalised box weights mapping `src` pixels to `dst` output pixels: for each
/// output pixel the first source index and the weights of the source pixels its
/// footprint `[o·src/dst, (o+1)·src/dst)` overlaps.
#[expect(
    clippy::cast_possible_truncation,
    reason = "normalised weights are narrowed to f32"
)]
fn axis_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let ratio = dim_f64(src) / dim_f64(dst);
    (0..dst)
        .map(|o| {
            let lo = dim_f64(o) * ratio;
            let hi = dim_f64(o + 1) * ratio;
            let first = index_of(lo.floor());
            let last = index_of(hi.ceil()).min(src);
            let overlaps: Vec<f64> = (first..last)
                .map(|i| (hi.min(dim_f64(i + 1)) - lo.max(dim_f64(i))).max(0.0))
                .collect();
            let total: f64 = overlaps.iter().sum();
            (first, overlaps.iter().map(|v| (v / total) as f32).collect())
        })
        .collect()
}

/// Area (box) resample of one `sw × sh` plane to `dw × dh`.
fn resample_plane(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let xs = axis_weights(sw, dw);
    let ys = axis_weights(sh, dh);

    // Horizontal pass: sh rows of dw samples.
    let mut tmp = vec![0f32; dw * sh];
    for (row_in, row_out) in src.chunks_exact(sw).zip(tmp.chunks_exact_mut(dw)) {
        for (out, (first, weights)) in row_out.iter_mut().zip(&xs) {
            *out = weights
                .iter()
                .zip(&row_in[*first..])
                .fold(0.0, |acc, (w, v)| v.mul_add(*w, acc));
        }
    }

    // Vertical pass.
    let mut out = vec![0f32; dw * dh];
    for (row_out, (first, weights)) in out.chunks_exact_mut(dw).zip(&ys) {
        for (x, cell) in row_out.iter_mut().enumerate() {
            *cell = weights.iter().enumerate().fold(0.0, |acc, (k, w)| {
                tmp[(first + k) * dw + x].mul_add(*w, acc)
            });
        }
    }
    out
}

/// Area-resample `img` to its [`thumbnail_size`]. Linear light, so averaging is
/// physically meaningful. A frame already within the limit is cloned.
///
/// The mapping between the two pixel grids is the standard area one: thumbnail
/// pixel `i` sits at full-resolution position `(i + ½)·W/tw − ½`
/// (see [`FrameThumbs::normalized_to_full`]).
#[must_use]
pub fn make_thumbnail(img: &PlanarImage<f32>) -> PlanarImage<f32> {
    let (tw, th) = thumbnail_size(img.width, img.height);
    if (tw, th) == (img.width, img.height) {
        return img.clone();
    }
    let (w, h) = (img.width, img.height);
    let mut out = PlanarImage::new(tw, th);
    out.luma = resample_plane(&img.luma, w, h, tw, th);
    out.chroma_a = resample_plane(&img.chroma_a, w, h, tw, th);
    out.chroma_b = resample_plane(&img.chroma_b, w, h, tw, th);
    out
}

/// The thumbnails of a stack, plus the geometry to map network output back to the
/// full-resolution frames.
///
/// Built one frame at a time ([`Self::push`]) while the caller streams the stack,
/// so only `3 · 4 · 512 · 384 · S` bytes of thumbnails (about 2.3 MB per frame) are
/// ever held, never the full-resolution stack.
#[derive(Debug, Clone)]
pub struct FrameThumbs {
    full_width: usize,
    full_height: usize,
    thumb_width: usize,
    thumb_height: usize,
    frames: Vec<PlanarImage<f32>>,
}

impl FrameThumbs {
    /// An empty collection for frames of `full_width × full_height`.
    ///
    /// # Errors
    /// [`BridgeError::TooSmall`] when either side is below 2 pixels.
    pub fn new(full_width: usize, full_height: usize) -> Result<Self, BridgeError> {
        if full_width < 2 || full_height < 2 {
            return Err(BridgeError::TooSmall {
                width: full_width,
                height: full_height,
            });
        }
        let (thumb_width, thumb_height) = thumbnail_size(full_width, full_height);
        Ok(Self {
            full_width,
            full_height,
            thumb_width,
            thumb_height,
            frames: Vec::new(),
        })
    }

    /// Thumbnail every frame of `frames` (all the same size as the first).
    ///
    /// # Errors
    /// [`BridgeError::EmptyStack`] for no frames, [`BridgeError::ShapeMismatch`] for
    /// a frame whose size differs from the first, [`BridgeError::TooSmall`] for a
    /// frame below 2 × 2.
    pub fn from_frames<'a>(
        frames: impl IntoIterator<Item = &'a PlanarImage<f32>>,
    ) -> Result<Self, BridgeError> {
        let mut frames = frames.into_iter();
        let first = frames.next().ok_or(BridgeError::EmptyStack)?;
        let mut thumbs = Self::new(first.width, first.height)?;
        thumbs.push(first)?;
        for frame in frames {
            thumbs.push(frame)?;
        }
        Ok(thumbs)
    }

    /// Thumbnail `frame` and append it.
    ///
    /// # Errors
    /// [`BridgeError::ShapeMismatch`] when `frame` is not `full_size()`.
    pub fn push(&mut self, frame: &PlanarImage<f32>) -> Result<(), BridgeError> {
        if frame.width != self.full_width || frame.height != self.full_height {
            return Err(BridgeError::ShapeMismatch {
                index: self.frames.len(),
                exp_w: self.full_width,
                exp_h: self.full_height,
                got_w: frame.width,
                got_h: frame.height,
            });
        }
        self.frames.push(make_thumbnail(frame));
        Ok(())
    }

    /// Number of frames collected.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frame has been pushed yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The thumbnails, in push order (frame 0 is the reference).
    #[must_use]
    pub fn frames(&self) -> &[PlanarImage<f32>] {
        &self.frames
    }

    /// Full-resolution size `(width, height)` the matrices are returned for.
    #[must_use]
    pub const fn full_size(&self) -> (usize, usize) {
        (self.full_width, self.full_height)
    }

    /// Thumbnail size `(width, height)` the network sees.
    #[must_use]
    pub const fn thumb_size(&self) -> (usize, usize) {
        (self.thumb_width, self.thumb_height)
    }

    /// `S`: thumbnail pixel coordinates → full-resolution pixel coordinates,
    /// `x_full = (x_t + ½)·sx − ½` with `sx = W / tw` (likewise for y), and its
    /// inverse.
    fn thumb_to_full(&self) -> (M3, M3) {
        let sx = dim_f64(self.full_width) / dim_f64(self.thumb_width);
        let sy = dim_f64(self.full_height) / dim_f64(self.thumb_height);
        // `0.5 * s` is exact, so the fused form rounds exactly like the plain one.
        let (ox, oy) = (0.5_f64.mul_add(sx, -0.5), 0.5_f64.mul_add(sy, -0.5));
        (
            [[sx, 0.0, ox], [0.0, sy, oy], [0.0, 0.0, 1.0]],
            [
                [1.0 / sx, 0.0, -ox / sx],
                [0.0, 1.0 / sy, -oy / sy],
                [0.0, 0.0, 1.0],
            ],
        )
    }

    /// Convert a matrix in the normalised coordinates of the **thumbnail** to the
    /// pixel coordinates of the **full-resolution** frames:
    /// `M_full = S · N_t⁻¹ · M_n · N_t · S⁻¹`.
    #[must_use]
    pub fn normalized_to_full(&self, m_n: PixelMatrix) -> PixelMatrix {
        let (n, n_inv) = normalizer(self.thumb_width, self.thumb_height);
        let (s, s_inv) = self.thumb_to_full();
        let m_thumb = mul3(&mul3(&n_inv, &widen(m_n)), &n);
        narrow(mul3(&mul3(&s, &m_thumb), &s_inv))
    }

    /// The inverse of [`Self::normalized_to_full`]: a full-resolution pixel matrix
    /// to the normalised coordinates of the thumbnail (what a network trained on
    /// thumbnails is supervised with).
    #[must_use]
    pub fn full_to_normalized(&self, m_full: PixelMatrix) -> PixelMatrix {
        let (n, n_inv) = normalizer(self.thumb_width, self.thumb_height);
        let (s, s_inv) = self.thumb_to_full();
        let m_thumb = mul3(&mul3(&s_inv, &widen(m_full)), &s);
        narrow(mul3(&mul3(&n, &m_thumb), &n_inv))
    }
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

/// Read `count` row-major `3 × 3` matrices out of a model output tensor's data.
fn read_matrices(data: &TensorData, count: usize) -> Result<Vec<PixelMatrix>, BridgeError> {
    let values = data
        .to_vec::<f32>()
        .map_err(|e| BridgeError::BadOutput(format!("{e:?}")))?;
    if values.len() != count * 9 {
        return Err(BridgeError::BadOutput(format!(
            "expected {count} matrices (= {} values), got {} values",
            count * 9,
            values.len()
        )));
    }
    // The length check above leaves no remainder.
    let (rows, _) = values.as_chunks::<9>();
    Ok(rows
        .iter()
        .map(|v| [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
        .collect())
}

/// Run a trained [`BatchAlignmentModel`] over a stack of thumbnails, returning one
/// **full-resolution pixel-space** matrix per frame (reference → frame; frame 0 is
/// the identity).
///
/// The whole stack goes through the network in one call (`S` times the
/// single-frame encoder cost; the thumbnails are ≤ 512 px so this stays small).
/// See the module docs for the coordinate conventions: the network output is in
/// the normalised coordinates of the thumbnails and [`FrameThumbs::normalized_to_full`]
/// maps it to the frames on disk.
///
/// # Errors
/// [`BridgeError::EmptyStack`] for no thumbnails and [`BridgeError::BadOutput`]
/// when the model output has the wrong size.
pub fn align_thumbs<B: Backend>(
    model: &impl BatchAlignmentModel<B>,
    thumbs: &FrameThumbs,
    device: &B::Device,
) -> Result<Vec<PixelMatrix>, BridgeError> {
    if thumbs.is_empty() {
        return Err(BridgeError::EmptyStack);
    }
    let s = thumbs.len();
    let stack = Tensor::cat(
        thumbs
            .frames()
            .iter()
            .map(|f| {
                planar_to_tensor::<B>(f, device)
                    .unsqueeze_dim::<4>(0)
                    .unsqueeze_dim::<5>(0)
            })
            .collect(),
        1,
    ); // [1, S, 3, th, tw]

    let out = model.align_batch(stack); // [1, S, 3, 3], normalised
    Ok(read_matrices(&out.into_data(), s)?
        .into_iter()
        .map(|m| thumbs.normalized_to_full(m))
        .collect())
}

/// Run a trained [`PairAlignmentModel`] over a stack of thumbnails, one frame at a
/// time against frame 0, returning one **full-resolution pixel-space** matrix per
/// frame. Same contract as [`align_thumbs`].
///
/// Frame 0 is registered against itself and comes back near (not exactly) the
/// identity; see [`crate::model::FusionAlignNet`]. Memory is the reference tensor
/// plus the frame in flight — never a stack tensor.
///
/// # Errors
/// As [`align_thumbs`].
pub fn align_thumbs_pairwise<B: Backend>(
    model: &impl PairAlignmentModel<B>,
    thumbs: &FrameThumbs,
    device: &B::Device,
) -> Result<Vec<PixelMatrix>, BridgeError> {
    let first = thumbs.frames().first().ok_or(BridgeError::EmptyStack)?;
    let reference = planar_to_tensor::<B>(first, device).unsqueeze_dim::<4>(0); // [1,3,th,tw]

    let mut matrices = Vec::with_capacity(thumbs.len());
    for frame in thumbs.frames() {
        let frame_t = planar_to_tensor::<B>(frame, device).unsqueeze_dim::<4>(0);
        let out = model.align_pair(reference.clone(), frame_t); // [1, 3, 3], normalised
        let m = read_matrices(&out.into_data(), 1)?[0];
        matrices.push(thumbs.normalized_to_full(m));
    }
    Ok(matrices)
}

/// Convenience over [`align_thumbs`] for frames already in memory: thumbnails them
/// (see [`FrameThumbs::from_frames`]), then aligns the stack.
///
/// Prefer building a [`FrameThumbs`] while streaming the stack when the full
/// stack should not be resident.
///
/// # Errors
/// Those of [`FrameThumbs::from_frames`] and [`align_thumbs`].
pub fn align_planar<B: Backend>(
    model: &impl BatchAlignmentModel<B>,
    frames: &[PlanarImage<f32>],
    device: &B::Device,
) -> Result<Vec<PixelMatrix>, BridgeError> {
    align_thumbs(model, &FrameThumbs::from_frames(frames)?, device)
}

/// Convenience over [`align_thumbs_pairwise`] for frames already in memory.
///
/// # Errors
/// Those of [`FrameThumbs::from_frames`] and [`align_thumbs_pairwise`].
pub fn align_planar_pairwise<B: Backend>(
    model: &impl PairAlignmentModel<B>,
    frames: &[PlanarImage<f32>],
    device: &B::Device,
) -> Result<Vec<PixelMatrix>, BridgeError> {
    align_thumbs_pairwise(model, &FrameThumbs::from_frames(frames)?, device)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type B = NdArray;

    fn device() -> <B as Backend>::Device {
        <B as Backend>::Device::default()
    }

    /// The bridge is a pure linear colour-matrix round trip: planar → RGB tensor →
    /// planar returns the input (no clamping, no transfer function), and a neutral
    /// grey of linear value 0.5 reaches the network as 0.5, not as its gamma
    /// decoding (~0.21).
    #[test]
    fn planar_tensor_round_trip_is_linear_and_exact() {
        let (w, h) = (6, 5);
        let mut img = PlanarImage::<f32>::new(w, h);
        let mut t = 0.37_f32;
        for i in 0..w * h {
            t = t.mul_add(1.7, 0.11).fract();
            img.luma[i] = t;
            img.chroma_a[i] = (t - 0.5) * 0.4;
            img.chroma_b[i] = (0.5 - t) * 0.3;
        }
        let tensor = planar_to_tensor::<B>(&img, &device());
        assert_eq!(tensor.dims(), [3, h, w]);
        let back = tensor_to_planar::<B>(tensor, w, h);
        for i in 0..w * h {
            assert!((back.luma[i] - img.luma[i]).abs() < 1e-5);
            assert!((back.chroma_a[i] - img.chroma_a[i]).abs() < 1e-5);
            assert!((back.chroma_b[i] - img.chroma_b[i]).abs() < 1e-5);
        }

        let mut grey = PlanarImage::<f32>::new(2, 2);
        grey.luma.fill(0.5);
        let data = planar_to_tensor::<B>(&grey, &device())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(data.iter().all(|v| (v - 0.5).abs() < 1e-5), "{data:?}");
    }
}
