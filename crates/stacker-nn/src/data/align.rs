//! Alignment training data: misaligned stacks with ground-truth matrices, and
//! unlabelled real stacks for the photometric fine-tune.
//!
//! ## On-disk layout (per scene directory)
//!
//! `scripts/03_simulate_misalignment.py` writes one directory per scene:
//!
//! ```text
//! scene_unaligned/
//!   metadata.json     n_planes, stack[], alignment_gt[][3][3], cropped_dims[w, h]
//!   <stack[i]>        misaligned frame i (sRGB PNG), every one of size cropped_dims
//! ```
//!
//! `alignment_gt[k]` is a **pixel-space** matrix in the shared `cropped_dims`
//! pixel grid that maps a reference (frame 0) pixel to the matching pixel of frame
//! `k`: `[x_k, y_k, 1]^T = M_k [x_ref, y_ref, 1]^T`. This is the same
//! reference-to-source direction the classical refiners return, so a trained
//! network's matrices can seed them directly.
//!
//! ## The normalised convention, and why the crop matters
//!
//! The networks work in normalised coordinates, where the pixel `(x, y)` of an
//! image of size `w x h` becomes `(2x/(w-1) - 1, 2y/(h-1) - 1)`, so the image
//! covers `[-1, 1]^2`:
//!
//! ```text
//! M_n = N(w, h) . M_px . N(w, h)^-1
//! ```
//!
//! **`(w, h)` is always the size of the image the network is shown.** At inference
//! that is the downscaled thumbnail, at training it is the training crop. A crop
//! is a different image with its own pixel grid (origin at the crop's top-left
//! corner) and its own normalisation, so the full-frame matrix has to be moved into
//! the crop's grid first:
//!
//! ```text
//! M_crop = T(-o) . M_full . T(o)        o = (left, top) of the crop window
//! M_n    = N(size, size) . M_crop . N(size, size)^-1
//! ```
//!
//! (The same window is cut from every frame, so reference and frame are shifted by
//! the same `o`.) Normalising the full-frame matrix by the full frame size while
//! training on a 256 px crop would label every sample with a translation that is
//! too small by `size / full_size` and a rotation or scale centre that is in the
//! wrong place; the network would learn a mapping that no inference-time image can
//! reproduce. [`crop_normalized_matrix`] does the correct conversion and is the one
//! the samples use.
//!
//! ## Decoding cost
//!
//! [`AlignImages`] decodes a scene's frames once (8-bit, in parallel);
//! [`AlignSceneCache`] keeps them between optimiser steps under a byte budget with
//! least-recently-used eviction, like [`super::SceneCache`] does for fusion scenes.
//! Only the crop is linearised.

use std::{
    cmp::Ordering,
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use image::RgbImage;
use rayon::prelude::*;
use serde::Deserialize;

use super::{
    ByteBudgetLru, CropParams, DataError, crop_rgb_linear, crop_window, open_rgb, srgb_lut,
};

/// Smallest crop side the alignment networks can be trained on: the encoder
/// downsamples by 8, and the head needs a feature map of at least 2 x 2.
pub const MIN_ALIGN_CROP: usize = 16;

// ---------------------------------------------------------------------------
// Matrix conversions
// ---------------------------------------------------------------------------

/// A 3x3 matrix in `f64`, row-major.
type Mat3 = [[f64; 3]; 3];

/// `a . b`.
fn mul3(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0_f64; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// Widen an `f32` matrix.
fn widen(m: [[f32; 3]; 3]) -> Mat3 {
    m.map(|row| row.map(f64::from))
}

/// Narrow an `f64` matrix.
#[expect(
    clippy::cast_possible_truncation,
    reason = "matrices are narrowed once, at the end of a conversion done in f64"
)]
fn narrow(m: &Mat3) -> [[f32; 3]; 3] {
    m.map(|row| row.map(|v| v as f32))
}

/// `N(w, h)` and its inverse: pixel `(x, y)` to normalised `(2x/(w-1) - 1, 2y/(h-1) - 1)`.
///
/// # Panics
/// Panics if `w < 2` or `h < 2` (a single pixel has no normalised extent).
#[expect(
    clippy::cast_precision_loss,
    reason = "image dimensions are far below 2^52, so the conversion to f64 is exact"
)]
fn normaliser(w: usize, h: usize) -> (Mat3, Mat3) {
    assert!(
        w >= 2 && h >= 2,
        "normalisation needs at least 2 x 2 pixels, got {w} x {h}"
    );
    let sx = 2.0 / (w as f64 - 1.0);
    let sy = 2.0 / (h as f64 - 1.0);
    (
        [[sx, 0.0, -1.0], [0.0, sy, -1.0], [0.0, 0.0, 1.0]],
        [
            [1.0 / sx, 0.0, 1.0 / sx],
            [0.0, 1.0 / sy, 1.0 / sy],
            [0.0, 0.0, 1.0],
        ],
    )
}

/// `T(dx, dy)`: translation by `(dx, dy)` pixels.
#[expect(
    clippy::cast_precision_loss,
    reason = "crop offsets are far below 2^52, so the conversion to f64 is exact"
)]
const fn translation(dx: usize, dy: usize) -> Mat3 {
    [
        [1.0, 0.0, dx as f64],
        [0.0, 1.0, dy as f64],
        [0.0, 0.0, 1.0],
    ]
}

/// Move a pixel-space matrix into the pixel grid of a crop whose top-left corner
/// is at `(left, top)`: `T(-o) . M . T(o)`, evaluated in `f64`.
fn crop_to_f64(m: &Mat3, left: usize, top: usize) -> Mat3 {
    // T(o) takes crop coordinates to full-frame coordinates, T(-o) back.
    let to_full = translation(left, top);
    let mut to_crop = to_full;
    to_crop[0][2] = -to_crop[0][2];
    to_crop[1][2] = -to_crop[1][2];
    mul3(&to_crop, &mul3(m, &to_full))
}

/// Convert a pixel-space matrix of a `w x h` image to the normalised `[-1, 1]^2`
/// convention: `N(w, h) . M . N(w, h)^-1`.
///
/// `(w, h)` must be the size of the image the network sees (see the module
/// documentation); for a training crop use [`crop_normalized_matrix`] instead.
///
/// # Panics
/// Panics if `w < 2` or `h < 2`.
#[must_use]
pub fn pixel_matrix_to_normalized(m: [[f32; 3]; 3], w: usize, h: usize) -> [[f32; 3]; 3] {
    let (n, n_inv) = normaliser(w, h);
    narrow(&mul3(&n, &mul3(&widen(m), &n_inv)))
}

/// The inverse of [`pixel_matrix_to_normalized`]: `N^-1 . M_n . N`.
///
/// # Panics
/// Panics if `w < 2` or `h < 2`.
#[must_use]
pub fn normalized_matrix_to_pixel(m_n: [[f32; 3]; 3], w: usize, h: usize) -> [[f32; 3]; 3] {
    let (n, n_inv) = normaliser(w, h);
    narrow(&mul3(&n_inv, &mul3(&widen(m_n), &n)))
}

/// Move a pixel-space matrix into the pixel grid of the crop window whose
/// top-left corner is `(left, top)`: `T(-o) . M . T(o)`.
///
/// A pure translation is unchanged by this (the shift is the same in every
/// grid); a rotation or scale about the frame centre becomes one about a different
/// point, with a different translation part.
#[must_use]
pub fn crop_pixel_matrix(m: [[f32; 3]; 3], left: usize, top: usize) -> [[f32; 3]; 3] {
    narrow(&crop_to_f64(&widen(m), left, top))
}

/// The training label for a full-frame pixel matrix `m` when the network is
/// shown the window `crop` of a `full_w x full_h` frame (`None`: the whole frame).
///
/// This is `N(size, size) . T(-o) . M . T(o) . N(size, size)^-1` for a crop, and
/// `N(full_w, full_h) . M . N^-1` for the whole frame, computed in `f64` and
/// narrowed once. See the module documentation for why the full-frame
/// normalisation is wrong for a crop.
///
/// # Panics
/// Panics if the normalising size is below 2 pixels per side.
#[must_use]
pub fn crop_normalized_matrix(
    m: [[f32; 3]; 3],
    crop: Option<CropParams>,
    full_w: usize,
    full_h: usize,
) -> [[f32; 3]; 3] {
    let m = widen(m);
    #[expect(
        clippy::option_if_let_else,
        reason = "both arms use the matrix `m`; the combinator form would move it twice"
    )]
    let (grid, w, h) = match crop {
        Some(cp) => (crop_to_f64(&m, cp.left, cp.top), cp.size, cp.size),
        None => (m, full_w, full_h),
    };
    let (n, n_inv) = normaliser(w, h);
    narrow(&mul3(&n, &mul3(&grid, &n_inv)))
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Metadata of an alignment-training scene, as `03_simulate_misalignment.py`
/// writes it.
///
/// Deserialises from the augmented `metadata.json`; the fields of the fusion
/// metadata that the alignment loader does not need (`masks`, `allfocus`,
/// `occlusion`, `merge_order`, `focus_fractions`, `crop_factor`, `original_dims`)
/// are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct AlignSceneMeta {
    /// Number of frames in the stack.
    pub n_planes: usize,
    /// PNG filenames of the misaligned frames, in stack order.
    pub stack: Vec<String>,
    /// Ground-truth matrices, one per `stack` entry, row-major, in the pixel grid
    /// of `cropped_dims`, mapping reference pixels to frame pixels.
    pub alignment_gt: Vec<[[f32; 3]; 3]>,
    /// `[width, height]` shared by every emitted image, i.e. the grid
    /// `alignment_gt` lives in.
    pub cropped_dims: [u32; 2],
}

impl AlignSceneMeta {
    /// Load and parse `metadata.json` from `scene_dir`.
    ///
    /// # Errors
    /// [`DataError::Io`] when the file cannot be read, [`DataError::MetadataParse`]
    /// when it is not valid JSON of this shape, and [`DataError::InvalidMetadata`]
    /// when the lengths disagree with `n_planes`, `cropped_dims` is below 2
    /// pixels per side, or a matrix holds a non-finite number.
    pub fn load(scene_dir: &Path) -> Result<Self, DataError> {
        let path = scene_dir.join("metadata.json");
        let bytes = std::fs::read(&path).map_err(|e| DataError::Io {
            path: path.clone(),
            source: e,
        })?;
        let meta: Self = serde_json::from_slice(&bytes).map_err(|e| DataError::MetadataParse {
            path: path.clone(),
            source: e,
        })?;
        meta.validate(&path)?;
        Ok(meta)
    }

    fn validate(&self, meta_path: &Path) -> Result<(), DataError> {
        let invalid = |reason: String| DataError::InvalidMetadata {
            path: meta_path.to_owned(),
            reason,
        };
        let n = self.n_planes;
        if self.stack.len() != n {
            return Err(invalid(format!(
                "`stack` has length {}, expected n_planes={n}",
                self.stack.len()
            )));
        }
        if self.alignment_gt.len() != n {
            return Err(invalid(format!(
                "`alignment_gt` has length {}, expected n_planes={n}",
                self.alignment_gt.len()
            )));
        }
        if n < 1 {
            return Err(invalid("n_planes must be >= 1".to_owned()));
        }
        let [w, h] = self.cropped_dims;
        if w < 2 || h < 2 {
            return Err(invalid(format!(
                "`cropped_dims` is {w} x {h}, expected at least 2 x 2"
            )));
        }
        if self
            .alignment_gt
            .iter()
            .any(|m| m.iter().flatten().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "`alignment_gt` holds a non-finite number".to_owned(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Samples
// ---------------------------------------------------------------------------

/// One alignment-training sample, all tensors on the chosen backend.
///
/// The matrices are in the normalised convention of the **crop the frames were cut
/// to** (see the module documentation).
#[derive(Debug)]
pub struct AlignSample<B: Backend> {
    /// The misaligned stack, `[S, 3, H, W]`, linear RGB.
    pub stack: Tensor<B, 4>,
    /// Ground-truth matrices (reference to frame `k`), normalised, `[S, 3, 3]`.
    pub gt_matrices: Tensor<B, 3>,
}

/// One pairwise-alignment sample: a reference, a frame and the matrix between them.
#[derive(Debug)]
pub struct PairAlignSample<B: Backend> {
    /// Reference frame `[3, H, W]`, linear RGB.
    pub reference: Tensor<B, 3>,
    /// Frame to register against the reference, `[3, H, W]`, linear RGB.
    pub frame: Tensor<B, 3>,
    /// Ground-truth matrix (reference to frame), normalised, `[3, 3]`.
    pub gt_matrix: Tensor<B, 2>,
}

// ---------------------------------------------------------------------------
// Decoded scene and cache
// ---------------------------------------------------------------------------

/// The decoded frames of one alignment scene, kept as 8-bit images.
#[derive(Debug)]
pub struct AlignImages {
    h: usize,
    w: usize,
    frames: Vec<RgbImage>,
}

impl AlignImages {
    /// Decode every frame of `seq` (in parallel), each exactly once.
    ///
    /// # Errors
    /// [`DataError::ImageDecode`] for a frame that is missing or cannot be decoded
    /// and [`DataError::ShapeMismatch`] for one that is not `cropped_dims` in size
    /// (the ground truth would then not describe it).
    pub fn load(seq: &AlignSequence) -> Result<Self, DataError> {
        let frames = seq
            .meta
            .stack
            .par_iter()
            .map(|name| seq.open_frame(name))
            .collect::<Result<Vec<_>, _>>()?;
        let [w, h] = seq.meta.cropped_dims;
        Ok(Self {
            h: h as usize,
            w: w as usize,
            frames,
        })
    }

    /// Image size `(height, width)` in pixels.
    #[must_use]
    pub const fn dims(&self) -> (usize, usize) {
        (self.h, self.w)
    }

    /// Number of frames held.
    #[must_use]
    pub const fn planes(&self) -> usize {
        self.frames.len()
    }

    /// Number of image files decoded to build this value (one per frame).
    #[must_use]
    pub const fn decoded_files(&self) -> usize {
        self.frames.len()
    }

    /// Bytes of decoded pixel data held: `3 . S . H . W`.
    #[must_use]
    pub const fn byte_size(&self) -> usize {
        3 * self.frames.len() * self.h * self.w
    }
}

/// Least-recently-used cache of decoded alignment scenes under a byte budget.
///
/// Same contract as [`super::SceneCache`]: the pixel data retained never exceeds
/// `budget_bytes`, a scene larger than the budget is returned but not retained, and
/// a budget of 0 keeps nothing (every use decodes the scene once).
#[derive(Debug)]
pub struct AlignSceneCache {
    store: ByteBudgetLru<AlignImages>,
    decoded_files: usize,
}

impl AlignSceneCache {
    /// A cache that retains at most `budget_bytes` of decoded pixel data.
    #[must_use]
    pub const fn new(budget_bytes: usize) -> Self {
        Self {
            store: ByteBudgetLru::new(budget_bytes),
            decoded_files: 0,
        }
    }

    /// The decoded scene for `seq`: a hit (no decoding) or a fresh
    /// [`AlignImages::load`], retained if it fits the budget.
    ///
    /// # Errors
    /// Any [`DataError`] that [`AlignImages::load`] reports.
    pub fn get(&mut self, seq: &AlignSequence) -> Result<Arc<AlignImages>, DataError> {
        if let Some(images) = self.store.lookup(&seq.scene_dir) {
            return Ok(images);
        }
        let images = Arc::new(AlignImages::load(seq)?);
        self.decoded_files += images.decoded_files();
        self.store
            .retain(seq.scene_dir.clone(), &images, images.byte_size());
        Ok(images)
    }

    /// Total image files this cache has decoded so far (hits add nothing).
    #[must_use]
    pub const fn decoded_files(&self) -> usize {
        self.decoded_files
    }

    /// Decoded pixel bytes currently retained (always `<=` the budget).
    #[must_use]
    pub const fn cached_bytes(&self) -> usize {
        self.store.used_bytes()
    }

    /// Number of scenes currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// `true` when no scene is retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Alignment sequence
// ---------------------------------------------------------------------------

/// A lazy alignment-training scene: one stack plus its ground-truth matrices.
#[derive(Debug, Clone)]
pub struct AlignSequence {
    /// Root directory of the scene.
    pub scene_dir: PathBuf,
    /// Parsed scene metadata.
    pub meta: AlignSceneMeta,
}

impl AlignSequence {
    /// Build a sequence by loading `metadata.json` from `scene_dir`.
    ///
    /// # Errors
    /// Any [`DataError`] that [`AlignSceneMeta::load`] reports.
    pub fn new(scene_dir: PathBuf) -> Result<Self, DataError> {
        let meta = AlignSceneMeta::load(&scene_dir)?;
        Ok(Self { scene_dir, meta })
    }

    /// Decode one frame by file name and check it is `cropped_dims` in size.
    fn open_frame(&self, name: &str) -> Result<RgbImage, DataError> {
        let img = open_rgb(&self.scene_dir.join(name))?;
        let [w, h] = self.meta.cropped_dims;
        if (img.width(), img.height()) != (w, h) {
            return Err(DataError::ShapeMismatch {
                expected: vec![3, h as usize, w as usize],
                got: vec![3, img.height() as usize, img.width() as usize],
            });
        }
        Ok(img)
    }

    /// The crop window for `crop` on this scene's frames, validated for alignment
    /// training (at least [`MIN_ALIGN_CROP`] pixels when a crop is requested).
    const fn window(
        &self,
        crop: Option<CropParams>,
    ) -> Result<(usize, usize, usize, usize), DataError> {
        if let Some(cp) = crop
            && cp.size < MIN_ALIGN_CROP
        {
            return Err(DataError::CropTooSmall {
                crop: cp.size,
                min: MIN_ALIGN_CROP,
            });
        }
        let [w, h] = self.meta.cropped_dims;
        crop_window(h as usize, w as usize, crop)
    }

    /// The normalised ground truth for frame `k` under `crop`.
    fn gt_normalized(&self, k: usize, crop: Option<CropParams>) -> [[f32; 3]; 3] {
        let [w, h] = self.meta.cropped_dims;
        crop_normalized_matrix(self.meta.alignment_gt[k], crop, w as usize, h as usize)
    }

    /// Decode the whole scene (see [`AlignImages`]).
    ///
    /// # Errors
    /// As [`AlignImages::load`].
    pub fn load_images(&self) -> Result<AlignImages, DataError> {
        AlignImages::load(self)
    }

    /// Materialise the whole scene as an [`AlignSample`], decoding it first.
    ///
    /// `crop = Some(..)` cuts the same square window from every frame and expresses
    /// the ground truth in that window's normalised grid ([`crop_normalized_matrix`]);
    /// `None` uses whole frames.
    ///
    /// # Errors
    /// Any [`DataError`] from decoding, [`DataError::CropTooLarge`] /
    /// [`DataError::CropTooSmall`] for a bad `crop`.
    pub fn get<B: Backend>(
        &self,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<AlignSample<B>, DataError> {
        let images = self.load_images()?;
        self.sample_from_images(&images, device, crop)
    }

    /// [`Self::get`] from frames that are already decoded (the cached path).
    ///
    /// # Errors
    /// [`DataError::InvalidMetadata`] when `images` holds a different number of
    /// frames than the scene, plus the crop errors of [`Self::get`].
    pub fn sample_from_images<B: Backend>(
        &self,
        images: &AlignImages,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<AlignSample<B>, DataError> {
        let n = self.meta.n_planes;
        if images.planes() != n {
            return Err(DataError::InvalidMetadata {
                path: self.scene_dir.join("metadata.json"),
                reason: format!(
                    "decoded scene holds {} frames, the scene has {n}",
                    images.planes()
                ),
            });
        }
        let window = self.window(crop)?;
        let (rows, cols) = (window.2, window.3);
        let lut = srgb_lut();

        let mut pixels = Vec::with_capacity(n * 3 * rows * cols);
        for frame in &images.frames {
            pixels.extend(crop_rgb_linear(frame, window, &lut));
        }
        let stack = Tensor::<B, 4>::from_data(TensorData::new(pixels, [n, 3, rows, cols]), device);

        let mut gt = Vec::with_capacity(n * 9);
        for k in 0..n {
            gt.extend(self.gt_normalized(k, crop).into_iter().flatten());
        }
        let gt_matrices = Tensor::<B, 3>::from_data(TensorData::new(gt, [n, 3, 3]), device);

        Ok(AlignSample { stack, gt_matrices })
    }

    /// Materialise one reference/frame pair (`frame_index` against frame 0),
    /// decoding only those two frames. `frame_index = 0` registers the reference
    /// against itself, which is a valid sample (identity ground truth).
    ///
    /// # Errors
    /// [`DataError::InvalidMetadata`] when `frame_index >= n_planes`, plus the
    /// decode and crop errors of [`Self::get`].
    pub fn get_pair<B: Backend>(
        &self,
        frame_index: usize,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<PairAlignSample<B>, DataError> {
        self.check_frame_index(frame_index)?;
        let reference = self.open_frame(&self.meta.stack[0])?;
        let frame = if frame_index == 0 {
            reference.clone()
        } else {
            self.open_frame(&self.meta.stack[frame_index])?
        };
        self.pair_from_parts(&reference, &frame, frame_index, device, crop)
    }

    /// [`Self::get_pair`] from frames that are already decoded (the cached path).
    ///
    /// # Errors
    /// As [`Self::get_pair`], plus [`DataError::InvalidMetadata`] when `images`
    /// holds a different number of frames than the scene.
    pub fn pair_from_images<B: Backend>(
        &self,
        images: &AlignImages,
        frame_index: usize,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<PairAlignSample<B>, DataError> {
        self.check_frame_index(frame_index)?;
        if images.planes() != self.meta.n_planes {
            return Err(DataError::InvalidMetadata {
                path: self.scene_dir.join("metadata.json"),
                reason: format!(
                    "decoded scene holds {} frames, the scene has {}",
                    images.planes(),
                    self.meta.n_planes
                ),
            });
        }
        self.pair_from_parts(
            &images.frames[0],
            &images.frames[frame_index],
            frame_index,
            device,
            crop,
        )
    }

    fn check_frame_index(&self, frame_index: usize) -> Result<(), DataError> {
        if frame_index >= self.meta.n_planes {
            return Err(DataError::InvalidMetadata {
                path: self.scene_dir.join("metadata.json"),
                reason: format!(
                    "frame_index {frame_index} out of range for n_planes={}",
                    self.meta.n_planes
                ),
            });
        }
        Ok(())
    }

    fn pair_from_parts<B: Backend>(
        &self,
        reference: &RgbImage,
        frame: &RgbImage,
        frame_index: usize,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<PairAlignSample<B>, DataError> {
        let window = self.window(crop)?;
        let (rows, cols) = (window.2, window.3);
        let lut = srgb_lut();
        let cut = |img: &RgbImage| {
            Tensor::<B, 3>::from_data(
                TensorData::new(crop_rgb_linear(img, window, &lut), [3, rows, cols]),
                device,
            )
        };
        let gt: Vec<f32> = self
            .gt_normalized(frame_index, crop)
            .into_iter()
            .flatten()
            .collect();
        Ok(PairAlignSample {
            reference: cut(reference),
            frame: cut(frame),
            gt_matrix: Tensor::<B, 2>::from_data(TensorData::new(gt, [3, 3]), device),
        })
    }
}

/// Scan `root` for sub-directories containing a `metadata.json` and load each as
/// an [`AlignSequence`], sorted by path.
///
/// # Errors
/// [`DataError::Io`] when `root` cannot be read, or any error of
/// [`AlignSequence::new`].
pub fn discover_align_scenes(root: &Path) -> Result<Vec<AlignSequence>, DataError> {
    let mut scenes = Vec::new();
    for dir in scene_dirs(root)? {
        scenes.push(AlignSequence::new(dir)?);
    }
    Ok(scenes)
}

/// The sub-directories of `root` that contain a `metadata.json`, sorted by path.
pub(super) fn scene_dirs(root: &Path) -> Result<Vec<PathBuf>, DataError> {
    let io = |source| DataError::Io {
        path: root.to_owned(),
        source,
    };
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(root).map_err(io)? {
        let path = entry.map_err(io)?.path();
        if path.is_dir() && path.join("metadata.json").exists() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// A dataset yielding one [`AlignSample`] per scene found under a root directory.
#[derive(Debug)]
pub struct AlignStackDataset<B: Backend> {
    sequences: Vec<AlignSequence>,
    device: Device<B>,
    /// Square window cut from every frame (`None`: whole frames). The ground truth
    /// follows the window, see [`AlignSequence::get`].
    pub crop: Option<CropParams>,
}

impl<B: Backend> AlignStackDataset<B> {
    /// Scan `root_dir` for scene directories and build the dataset.
    ///
    /// # Errors
    /// [`DataError::Io`] when `root_dir` cannot be read, or any error of
    /// [`AlignSequence::new`] for a malformed scene.
    pub fn from_root(
        root_dir: &Path,
        device: Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Self, DataError> {
        Ok(Self {
            sequences: discover_align_scenes(root_dir)?,
            device,
            crop,
        })
    }

    /// Number of scenes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sequences.len()
    }

    /// `true` when no scene was found.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Materialise the sample of scene `index`.
    ///
    /// # Errors
    /// As [`AlignSequence::get`].
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    pub fn get(&self, index: usize) -> Result<AlignSample<B>, DataError> {
        self.sequences[index].get(&self.device, self.crop)
    }
}

// ---------------------------------------------------------------------------
// Real (unlabelled) stacks
// ---------------------------------------------------------------------------

/// A real, unlabelled focus stack for the photometric fine-tune: a directory of
/// image files in stack order, no metadata and no ground truth.
///
/// Frames are the directory's `png`, `jpg`, `jpeg` and `tif` files (any case),
/// ordered naturally (`img2` before `img10`); the first is the reference. They are
/// decoded to 8 bits per channel (a 16-bit file is narrowed) and linearised after
/// cropping.
#[derive(Debug, Clone)]
pub struct RealStackScene {
    /// Root directory of the scene.
    pub scene_dir: PathBuf,
    /// Frame file paths in stack order.
    pub frames: Vec<PathBuf>,
}

/// Compare two file names so that digit runs order by value and everything else
/// case-insensitively (`img2` < `img10`).
fn natural_cmp(a: &str, b: &str) -> Ordering {
    fn runs(s: &str) -> Vec<(bool, String)> {
        let mut out: Vec<(bool, String)> = Vec::new();
        for c in s.chars() {
            let digit = c.is_ascii_digit();
            let c = c.to_ascii_lowercase();
            if let Some((d, run)) = out.last_mut()
                && *d == digit
            {
                run.push(c);
            } else {
                out.push((digit, c.to_string()));
            }
        }
        out
    }
    let (ra, rb) = (runs(a), runs(b));
    for ((da, sa), (db, sb)) in ra.iter().zip(&rb) {
        let ord = if *da && *db {
            let (ta, tb) = (sa.trim_start_matches('0'), sb.trim_start_matches('0'));
            ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb))
        } else {
            sa.cmp(sb)
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    ra.len().cmp(&rb.len()).then_with(|| a.cmp(b))
}

impl RealStackScene {
    /// File extensions accepted as frames (compared case-insensitively).
    const IMAGE_EXTENSIONS: [&'static str; 4] = ["png", "jpg", "jpeg", "tif"];

    /// Scan `scene_dir` for image files and treat them as one stack.
    ///
    /// # Errors
    /// [`DataError::Io`] when `scene_dir` cannot be read.
    pub fn new(scene_dir: PathBuf) -> Result<Self, DataError> {
        let io = |source| DataError::Io {
            path: scene_dir.clone(),
            source,
        };
        let mut frames = Vec::new();
        for entry in std::fs::read_dir(&scene_dir).map_err(io)? {
            let path = entry.map_err(io)?.path();
            let is_image = path.is_file()
                && path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| {
                        Self::IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
                    });
            if is_image {
                frames.push(path);
            }
        }
        frames.sort_by(|a, b| {
            let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
            natural_cmp(&name(a).unwrap_or_default(), &name(b).unwrap_or_default())
        });
        Ok(Self { scene_dir, frames })
    }

    /// Number of frames found.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` when no frame was found.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Size `(height, width)` of the first frame, read from its header only.
    ///
    /// # Errors
    /// [`DataError::InvalidMetadata`] for an empty scene and
    /// [`DataError::ImageDecode`] when the header cannot be read.
    pub fn frame_dims(&self) -> Result<(usize, usize), DataError> {
        let Some(first) = self.frames.first() else {
            return Err(DataError::InvalidMetadata {
                path: self.scene_dir.clone(),
                reason: "the scene holds no image files".to_owned(),
            });
        };
        let (w, h) = image::image_dimensions(first).map_err(|source| DataError::ImageDecode {
            path: first.clone(),
            source,
        })?;
        Ok((h as usize, w as usize))
    }

    /// Materialise the stack as `[S, 3, H, W]` linear RGB, cropping each frame as
    /// it is decoded so only the crop is ever held in `f32`.
    ///
    /// # Errors
    /// [`DataError::InvalidMetadata`] for an empty scene,
    /// [`DataError::ImageDecode`] / [`DataError::Io`] for a bad frame,
    /// [`DataError::ShapeMismatch`] when the frames differ in size, and
    /// [`DataError::CropTooLarge`] for a window that does not fit.
    pub fn get<B: Backend>(
        &self,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Tensor<B, 4>, DataError> {
        let (h, w) = self.frame_dims()?;
        let window = crop_window(h, w, crop)?;
        let (rows, cols) = (window.2, window.3);
        let lut = srgb_lut();

        let mut pixels = Vec::with_capacity(self.frames.len() * 3 * rows * cols);
        for path in &self.frames {
            let img = open_rgb(path)?;
            if (img.height() as usize, img.width() as usize) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![3, h, w],
                    got: vec![3, img.height() as usize, img.width() as usize],
                });
            }
            pixels.extend(crop_rgb_linear(&img, window, &lut));
        }
        Ok(Tensor::<B, 4>::from_data(
            TensorData::new(pixels, [self.frames.len(), 3, rows, cols]),
            device,
        ))
    }
}

/// Scan `root_dir` for real stacks: every sub-directory that holds at least one
/// recognised image file is a scene (there is no `metadata.json` gate). Sorted by
/// path.
///
/// # Errors
/// [`DataError::Io`] when `root_dir` cannot be read.
pub fn discover_real_stacks(root_dir: &Path) -> Result<Vec<RealStackScene>, DataError> {
    let io = |source| DataError::Io {
        path: root_dir.to_owned(),
        source,
    };
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(root_dir).map_err(io)? {
        let path = entry.map_err(io)?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    let mut scenes = Vec::new();
    for dir in dirs {
        let scene = RealStackScene::new(dir)?;
        if !scene.is_empty() {
            scenes.push(scene);
        }
    }
    Ok(scenes)
}

#[cfg(test)]
mod unit_tests {
    use super::natural_cmp;
    use std::cmp::Ordering;

    #[test]
    fn natural_order_compares_digit_runs_by_value() {
        assert_eq!(natural_cmp("img2.png", "img10.png"), Ordering::Less);
        assert_eq!(natural_cmp("img10.png", "img2.png"), Ordering::Greater);
        assert_ne!(natural_cmp("img_007.png", "img_7.png"), Ordering::Equal);
        assert_eq!(natural_cmp("img_007.png", "img_8.png"), Ordering::Less);
        assert_eq!(natural_cmp("a.png", "a.png"), Ordering::Equal);
        assert_eq!(natural_cmp("a1b.png", "a1c.png"), Ordering::Less);
    }
}
