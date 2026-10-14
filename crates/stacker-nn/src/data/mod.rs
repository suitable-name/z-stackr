//! Dataset loading, preprocessing, and rollout construction for focus-stacking.
//!
//! ## On-disk layout (per scene directory)
//!
//! ```text
//! scene/
//!   metadata.json          – scene descriptor (see [`SceneMeta`])
//!   focus_00.png           – input frames (sRGB PNG), in the order of `stack`
//!   focus_01.png
//!   …
//!   mask_00.png            – soft in-focus weight, 0..1 stored as 0..255, in
//!   mask_01.png              the order of `masks`
//!   …
//!   allfocus.png           – ground-truth all-in-focus composite (sRGB PNG)
//!   occlusion.png          – depth-edge mask (grayscale PNG, 255 = edge)
//! ```
//!
//! The file names above are what `scripts/02_blender_focus_stack.py` writes
//! (two-digit indices). The loader does **not** depend on them: every name is
//! read from `metadata.json` (`stack`, `masks`, `allfocus`, `occlusion`), so any
//! naming works as long as the metadata lists the files. `stack[i]` and
//! `masks[i]` describe the same focal plane `i`; `merge_order` is a permutation
//! of the plane indices giving the order the frames are folded in.
//!
//! ## Bit depth and mask semantics
//!
//! Every PNG is decoded to **8 bits** per channel (`image`'s `into_rgb8` /
//! `into_luma8`), so a 16-bit PNG is silently narrowed to 8 bits and
//! carries no extra precision into training; the generator writes 8-bit files
//! for that reason. A mask is a weight in 0..1 (255 = fully in focus), a colour
//! frame is sRGB, and the occlusion map is 0 or 255.
//!
//! ## Linear-light policy
//!
//! RGB PNGs are decoded as 8-bit sRGB and **linearised** before any tensor
//! operation via [`to_linear`].  Grayscale masks represent weights (not colour)
//! and are normalised to 0..1 without additional gamma conversion.
//! Every tensor produced here is `f32`, linear-light, values in 0..1.
//!
//! ## Rollout sequence
//!
//! For a scene with `n_planes` frames the rollout yields `n_planes - 1` steps:
//!
//! ```text
//!   step 0: target_0 = frame[merge_order[0]]   (the RAW first frame)
//!           conf_0   = 0                       (as at inference)
//!           source_1 = frame[merge_order[1]]
//!           gt_1     = prefix_composite(frames 0..=1)
//!
//!   step k: target_k = prefix_composite(frames 0..=k)   (= gt_{k-1}, teacher forcing)
//!           conf_k   = max mask over frames 0..=k
//!           source   = frame[merge_order[k+1]]
//!           gt_{k+1} = prefix_composite(frames 0..=k+1)
//! ```
//!
//! Step 0 starts from the raw first frame with confidence 0 because that is
//! exactly what inference feeds the first merge (`infer::fuse_stack`); an oracle
//! composite with the mask as confidence there would train the first merge on a
//! distribution it never sees.
//!
//! ## Decoding cost and memory
//!
//! A training step needs every plane of a scene, once. [`SceneImages::load`]
//! decodes the `2·n_planes + 2` PNGs of a scene exactly once (frames, masks,
//! all-focus, occlusion; in parallel with rayon) and keeps them as 8-bit images;
//! [`RolloutSequence::crop_steps`] then crops first and linearises only the crop
//! before building all `n_planes - 1` steps from it. [`SceneCache`] keeps decoded
//! scenes between optimiser steps under a byte budget with least-recently-used
//! eviction: the decoded pixel memory held by the cache never exceeds the budget,
//! and a scene larger than the budget is decoded per use but never retained.
//! Decoding per rollout step instead would cost `2k + 7` full-resolution PNGs per
//! step.
//!
//! ## Batch fusion
//!
//! The batch fusion network ([`crate::model::batch_merge::BatchMergeNet`]) sees the
//! whole stack at once. [`RolloutSequence::get_batch`] (and
//! [`RolloutSequence::batch_from_images`] for decoded, cached scenes) builds one
//! [`BatchSample`]: the frames and masks in merge order, the all-focus composite as
//! the target and the occlusion map, all cut to the same crop. It reads the same
//! scene layout as the pairwise strategy, so one rendered dataset trains both.
//!
//! ## Alignment
//!
//! The alignment strategies read `scripts/03_simulate_misalignment.py` scenes
//! (`metadata.json` with `alignment_gt`, no masks) and unlabelled real stacks for
//! the photometric fine-tune; see the [`align`] module for the layout and for the
//! crop-correct normalisation of the ground-truth matrices.

use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// The tests build tensors on the CPU `NdArray` backend, so they need that feature.
#[cfg(all(test, feature = "ndarray"))]
mod tests;

pub mod align;
pub use align::{
    AlignImages, AlignSample, AlignSceneCache, AlignSceneMeta, AlignSequence, AlignStackDataset,
    MIN_ALIGN_CROP, PairAlignSample, RealStackScene, crop_normalized_matrix, crop_pixel_matrix,
    discover_align_scenes, discover_real_stacks, normalized_matrix_to_pixel,
    pixel_matrix_to_normalized,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// All errors that can arise while loading or constructing a dataset sample.
#[derive(Debug, Error)]
pub enum DataError {
    /// An I/O error occurred while reading a file.
    #[error("I/O error reading {path}: {source}")]
    Io {
        /// The file or directory being read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The `image` crate could not decode a PNG.
    #[error("image decode error for {path}: {source}")]
    ImageDecode {
        /// The PNG that failed to decode.
        path: PathBuf,
        /// The decoder error.
        #[source]
        source: image::ImageError,
    },

    /// `metadata.json` could not be parsed.
    #[error("metadata parse error in {path}: {source}")]
    MetadataParse {
        /// The metadata file.
        path: PathBuf,
        /// The JSON error.
        #[source]
        source: serde_json::Error,
    },

    /// The metadata describes an inconsistent scene (wrong file counts, etc.).
    #[error("invalid metadata in {path}: {reason}")]
    InvalidMetadata {
        /// The metadata file.
        path: PathBuf,
        /// What is inconsistent.
        reason: String,
    },

    /// A tensor shape mismatch was detected.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// The shape the first image of the scene established.
        expected: Vec<usize>,
        /// The shape of the offending image.
        got: Vec<usize>,
    },

    /// Requested crop size is larger than the image.
    #[error("crop size {crop} exceeds image dimension {dim}")]
    CropTooLarge {
        /// The requested extent (size, or offset plus size).
        crop: usize,
        /// The image dimension it exceeds.
        dim: usize,
    },

    /// Requested crop size is below what the alignment networks accept.
    #[error("crop size {crop} is below the minimum of {min} pixels for alignment training")]
    CropTooSmall {
        /// The requested side length.
        crop: usize,
        /// The smallest accepted side length.
        min: usize,
    },
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Deserialised representation of a scene's `metadata.json`.
///
/// All filenames in the struct are relative to the scene directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMeta {
    /// Number of focal planes.
    pub n_planes: usize,
    /// Normalised depth fraction each plane is focused at (length `n_planes`).
    pub focus_fractions: Vec<f32>,
    /// PNG filenames of the input frames, in stack order (length `n_planes`).
    pub stack: Vec<String>,
    /// PNG filenames of soft in-focus weight maps (length `n_planes`).
    pub masks: Vec<String>,
    /// Filename of the ground-truth all-in-focus composite PNG.
    pub allfocus: String,
    /// Filename of the depth-edge / occlusion mask PNG.
    pub occlusion: String,
    /// Order in which frames should be merged (length `n_planes`).
    pub merge_order: Vec<usize>,
}

impl SceneMeta {
    /// Load and parse `metadata.json` from `scene_dir`.
    ///
    /// # Errors
    /// [`DataError::Io`] when the file cannot be read, [`DataError::MetadataParse`]
    /// when it is not valid JSON of the expected shape, and
    /// [`DataError::InvalidMetadata`] when the described scene is inconsistent.
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
        meta.validate(scene_dir)?;
        Ok(meta)
    }

    fn validate(&self, scene_dir: &Path) -> Result<(), DataError> {
        let meta_path = scene_dir.join("metadata.json");
        let n = self.n_planes;

        let check = |got: usize, field: &str| -> Result<(), DataError> {
            if got == n {
                Ok(())
            } else {
                Err(DataError::InvalidMetadata {
                    path: meta_path.clone(),
                    reason: format!("`{field}` has length {got}, expected n_planes={n}"),
                })
            }
        };

        check(self.focus_fractions.len(), "focus_fractions")?;
        check(self.stack.len(), "stack")?;
        check(self.masks.len(), "masks")?;
        check(self.merge_order.len(), "merge_order")?;

        if n < 2 {
            return Err(DataError::InvalidMetadata {
                path: meta_path,
                reason: "n_planes must be >= 2".to_string(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Linear-light helpers (public — shared with the rest of the crate)
// ---------------------------------------------------------------------------

/// Convert an sRGB-encoded value in 0..1 to linear light.
///
/// Applies the IEC 61966-2-1 piecewise transfer function.
#[inline]
#[must_use]
pub fn to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

// ---------------------------------------------------------------------------
// Image loading helpers (private)
// ---------------------------------------------------------------------------

/// Decode an RGB PNG to an 8-bit image (sRGB-encoded; linearised after cropping).
fn open_rgb(path: &Path) -> Result<RgbImage, DataError> {
    Ok(image::open(path)
        .map_err(|e| DataError::ImageDecode {
            path: path.to_owned(),
            source: e,
        })?
        .into_rgb8())
}

/// Decode a grayscale PNG to an 8-bit image (weights; no gamma conversion).
fn open_gray(path: &Path) -> Result<GrayImage, DataError> {
    Ok(image::open(path)
        .map_err(|e| DataError::ImageDecode {
            path: path.to_owned(),
            source: e,
        })?
        .into_luma8())
}

/// `[0, 255]` sRGB code -> linear-light f32, as a lookup table.
fn srgb_lut() -> [f32; 256] {
    let mut lut = [0f32; 256];
    for (code, slot) in (0..=u8::MAX).zip(lut.iter_mut()) {
        *slot = to_linear(f32::from(code) / 255.0);
    }
    lut
}

// ---------------------------------------------------------------------------
// Tensor helpers (private)
// ---------------------------------------------------------------------------

/// Build a rank-3 Burn tensor from a flat f32 buffer with shape `[c, h, w]`.
fn make_tensor<B: Backend>(
    data: Vec<f32>,
    c: usize,
    h: usize,
    w: usize,
    device: &Device<B>,
) -> Tensor<B, 3> {
    let td = TensorData::new(data, [c, h, w]);
    Tensor::<B, 3>::from_data(td, device)
}

// ---------------------------------------------------------------------------
// Core sample type
// ---------------------------------------------------------------------------

/// A single pairwise-merge training sample with all tensors on the chosen
/// Burn backend.
///
/// All tensors are `f32`, linear-light, values in 0..1.
#[derive(Debug)]
pub struct MergeSample<B: Backend> {
    /// Current composite — shape `[3, H, W]`, linear RGB. At step 0 this is the
    /// raw first frame (what inference starts from), afterwards the oracle prefix
    /// composite.
    pub target: Tensor<B, 3>,
    /// Per-pixel confidence of `target` — shape `[1, H, W]`. All zeros at step 0
    /// (as at inference), afterwards the per-pixel max mask value.
    pub target_conf: Tensor<B, 3>,
    /// Next frame to merge in — shape `[3, H, W]`, linear RGB.
    pub source: Tensor<B, 3>,
    /// Ground-truth composite after merging `source` — shape `[3, H, W]`.
    pub gt_merged: Tensor<B, 3>,
    /// Ground-truth confidence / in-focus coverage for `gt_merged` — shape
    /// `[1, H, W]`. Per-pixel max mask value over the frames composited so far
    /// (`0..=k+1`). Used to supervise the network's predicted confidence.
    pub gt_conf: Tensor<B, 3>,
    /// Depth-edge / occlusion mask — shape `[1, H, W]` (1 = depth edge).
    pub occlusion: Tensor<B, 3>,
}

/// One batch-fusion training sample: the whole stack of a scene.
///
/// All tensors are `f32`; the colour ones are linear light in 0..1, the masks are
/// weights in 0..1.
#[derive(Debug)]
pub struct BatchSample<B: Backend> {
    /// The focus stack in merge order, `[S, 3, H, W]`, linear RGB.
    pub stack: Tensor<B, 4>,
    /// Ground-truth composite of the full stack, `[3, H, W]`, linear RGB (see
    /// [`RolloutSequence::get_batch`]).
    pub gt_merged: Tensor<B, 3>,
    /// Soft in-focus weight of every frame, `[S, 1, H, W]`, in the same merge
    /// order and crop as `stack`. Supervises the network's per-frame gate.
    pub masks: Tensor<B, 4>,
    /// Depth-edge / occlusion mask, `[1, H, W]` (1 = depth edge), same crop.
    pub occlusion: Tensor<B, 3>,
}

// ---------------------------------------------------------------------------
// Prefix composite
// ---------------------------------------------------------------------------

/// Mask value above which the ground truth treats a pixel as in focus.
///
/// `exp(-0.5)`, the one-sigma contour of the Gaussian mask
/// `exp(-(depth - t)^2 / (2 sigma^2))` that `scripts/02_blender_focus_stack.py` writes
/// with `sigma` equal to the plane spacing.
///
/// A threshold of `0.1` would be the 2.15-sigma contour: with `sigma` equal to
/// the plane spacing it would copy the all-focus render into pixels up to about
/// two plane spacings away from every included plane, i.e. visibly defocused in
/// every included frame, so the supervision would ask for output sharper than any
/// input shows. One sigma keeps the "some included plane is within one spacing"
/// meaning; everything further out falls back to the least-defocused included
/// frame.
pub const ALLFOCUS_MASK_THRESHOLD: f32 = 0.606_530_66;

/// Build the ground-truth composite for the first `included` frames.
///
/// The `frames_linear` and `masks_linear` slices are **already re-indexed by
/// `merge_order`**: entry `i` corresponds to `merge_order[i]`, so the caller
/// only passes the `0..included` prefix.
///
/// Per-pixel rule:
/// * If any included mask value exceeds `threshold` → copy from `allfocus`.
/// * Otherwise → copy from the included frame whose mask is highest
///   (the "least defocused" frame).
///
/// Returns a flat CHW f32 buffer of shape `[3, H, W]`.
#[must_use]
pub fn prefix_composite(
    frames_linear: &[Vec<f32>],
    masks_linear: &[Vec<f32>],
    allfocus_linear: &[f32],
    included: usize,
    h: usize,
    w: usize,
    threshold: f32,
) -> Vec<f32> {
    debug_assert!(included >= 1);
    debug_assert!(included <= frames_linear.len());
    debug_assert_eq!(masks_linear.len(), frames_linear.len());

    let np = h * w;
    let mut out = vec![0f32; 3 * np];

    for px in 0..np {
        // argmax mask among included frames
        let (best, best_val) = (0..included).fold((0usize, 0f32), |(bi, bv), i| {
            let v = masks_linear[i][px];
            if v > bv { (i, v) } else { (bi, bv) }
        });

        if best_val > threshold {
            // Confident focus somewhere in the included set → use allfocus.
            out[px] = allfocus_linear[px];
            out[np + px] = allfocus_linear[np + px];
            out[2 * np + px] = allfocus_linear[2 * np + px];
        } else {
            // No confident focus → fall back to the least-defocused frame.
            let f = &frames_linear[best];
            out[px] = f[px];
            out[np + px] = f[np + px];
            out[2 * np + px] = f[2 * np + px];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Spatial crop helpers (private)
// ---------------------------------------------------------------------------

/// Parameters for a deterministic spatial crop.
#[derive(Debug, Clone, Copy)]
pub struct CropParams {
    /// Top-left row of the crop window.
    pub top: usize,
    /// Top-left column of the crop window.
    pub left: usize,
    /// Side length of the square crop (≤ both H and W).
    pub size: usize,
}

/// Validate `cp` against an `h × w` image and return `(top, left, rows, cols)`.
///
/// `None` selects the whole image.
const fn crop_window(
    h: usize,
    w: usize,
    crop: Option<CropParams>,
) -> Result<(usize, usize, usize, usize), DataError> {
    let Some(cp) = crop else {
        return Ok((0, 0, h, w));
    };
    if cp.size > h {
        return Err(DataError::CropTooLarge {
            crop: cp.size,
            dim: h,
        });
    }
    if cp.size > w {
        return Err(DataError::CropTooLarge {
            crop: cp.size,
            dim: w,
        });
    }
    if cp.top + cp.size > h {
        return Err(DataError::CropTooLarge {
            crop: cp.top + cp.size,
            dim: h,
        });
    }
    if cp.left + cp.size > w {
        return Err(DataError::CropTooLarge {
            crop: cp.left + cp.size,
            dim: w,
        });
    }
    Ok((cp.top, cp.left, cp.size, cp.size))
}

/// Crop an 8-bit RGB image to the window and return linear-light planar CHW f32.
fn crop_rgb_linear(
    img: &RgbImage,
    (top, left, rows, cols): (usize, usize, usize, usize),
    lut: &[f32; 256],
) -> Vec<f32> {
    let stride = img.width() as usize;
    let raw = img.as_raw();
    let np = rows * cols;
    let mut out = vec![0f32; 3 * np];
    for row in 0..rows {
        let src = ((top + row) * stride + left) * 3;
        let line = &raw[src..src + cols * 3];
        let dst = row * cols;
        for (col, px) in line.as_chunks::<3>().0.iter().enumerate() {
            out[dst + col] = lut[usize::from(px[0])];
            out[np + dst + col] = lut[usize::from(px[1])];
            out[2 * np + dst + col] = lut[usize::from(px[2])];
        }
    }
    out
}

/// Crop an 8-bit gray image to the window and return flat HW f32 in 0..1.
fn crop_gray(img: &GrayImage, (top, left, rows, cols): (usize, usize, usize, usize)) -> Vec<f32> {
    let stride = img.width() as usize;
    let raw = img.as_raw();
    let mut out = vec![0f32; rows * cols];
    for row in 0..rows {
        let src = (top + row) * stride + left;
        for (slot, &v) in out[row * cols..(row + 1) * cols]
            .iter_mut()
            .zip(&raw[src..src + cols])
        {
            *slot = f32::from(v) / 255.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Decoded scene (8-bit) and the bounded cache
// ---------------------------------------------------------------------------

/// The decoded planes of one scene, kept as 8-bit images.
///
/// Entry `i` of the frame and mask lists is the plane at merge-order position `i`
/// (`meta.merge_order[i]`), so step construction never needs the permutation
/// again. Created once per scene use by [`SceneImages::load`]; shared
/// through an [`Arc`] by [`SceneCache`].
#[derive(Debug)]
pub struct SceneImages {
    h: usize,
    w: usize,
    frames: Vec<RgbImage>,
    masks: Vec<GrayImage>,
    allfocus: RgbImage,
    occlusion: GrayImage,
    decoded_files: usize,
}

impl SceneImages {
    /// Decode the first `planes` merge-order planes (frames and masks, in
    /// parallel) plus the all-focus and occlusion images — `2·planes + 2` PNG
    /// decodes, each file exactly once.
    ///
    /// `planes` is clamped to `n_planes`. Training uses `n_planes`; a single
    /// [`RolloutSequence::get_step`] only needs `k + 2`.
    ///
    /// # Errors
    /// [`DataError::ImageDecode`] or [`DataError::ShapeMismatch`] when a frame,
    /// mask, all-focus or occlusion image is missing, cannot be decoded, or
    /// differs in size from the others.
    pub fn load(seq: &RolloutSequence, planes: usize) -> Result<Self, DataError> {
        let meta = &seq.meta;
        let sd = &seq.scene_dir;
        let planes = planes.min(meta.n_planes);

        let meta_path = sd.join("metadata.json");
        let plane_paths = |names: &[String]| -> Result<Vec<PathBuf>, DataError> {
            meta.merge_order[..planes]
                .iter()
                .map(|&fi| {
                    names
                        .get(fi)
                        .map(|n| sd.join(n))
                        .ok_or_else(|| DataError::InvalidMetadata {
                            path: meta_path.clone(),
                            reason: format!("merge_order entry {fi} is out of range"),
                        })
                })
                .collect()
        };
        let frame_paths = plane_paths(&meta.stack)?;
        let mask_paths = plane_paths(&meta.masks)?;

        let frames = frame_paths
            .par_iter()
            .map(|p| open_rgb(p))
            .collect::<Result<Vec<_>, _>>()?;
        let masks = mask_paths
            .par_iter()
            .map(|p| open_gray(p))
            .collect::<Result<Vec<_>, _>>()?;
        let allfocus = open_rgb(&sd.join(&meta.allfocus))?;
        let occlusion = open_gray(&sd.join(&meta.occlusion))?;

        let (h, w) = (allfocus.height() as usize, allfocus.width() as usize);
        for f in &frames {
            if (f.height() as usize, f.width() as usize) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![3, h, w],
                    got: vec![3, f.height() as usize, f.width() as usize],
                });
            }
        }
        for m in masks.iter().chain(std::iter::once(&occlusion)) {
            if (m.height() as usize, m.width() as usize) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![1, h, w],
                    got: vec![1, m.height() as usize, m.width() as usize],
                });
            }
        }

        Ok(Self {
            h,
            w,
            decoded_files: 2 * frames.len() + 2,
            frames,
            masks,
            allfocus,
            occlusion,
        })
    }

    /// Image size `(height, width)` in pixels.
    #[must_use]
    pub const fn dims(&self) -> (usize, usize) {
        (self.h, self.w)
    }

    /// Number of merge-order planes held.
    #[must_use]
    pub const fn planes(&self) -> usize {
        self.frames.len()
    }

    /// Number of PNG files decoded to build this value (`2·planes + 2`).
    #[must_use]
    pub const fn decoded_files(&self) -> usize {
        self.decoded_files
    }

    /// Bytes of decoded pixel data held: `(4·planes + 4)·H·W`.
    #[must_use]
    pub const fn byte_size(&self) -> usize {
        (4 * self.frames.len() + 4) * self.h * self.w
    }
}

/// The bookkeeping shared by [`SceneCache`] and [`AlignSceneCache`]: values keyed
/// by scene directory, least recently used first, retained only while their summed
/// byte sizes stay within the budget.
#[derive(Debug)]
struct ByteBudgetLru<T> {
    budget_bytes: usize,
    used_bytes: usize,
    /// `(key, value, bytes)`, least recently used first.
    entries: VecDeque<(PathBuf, Arc<T>, usize)>,
}

impl<T> ByteBudgetLru<T> {
    const fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            entries: VecDeque::new(),
        }
    }

    /// The value under `key`, marked most recently used.
    fn lookup(&mut self, key: &Path) -> Option<Arc<T>> {
        let pos = self.entries.iter().position(|(p, _, _)| p == key)?;
        let entry = self.entries.remove(pos)?;
        let value = Arc::clone(&entry.1);
        self.entries.push_back(entry);
        Some(value)
    }

    /// Keep `value` (of `bytes` bytes) under `key` if it fits the budget at all,
    /// evicting the least recently used entries to make room. A value larger than
    /// the whole budget is not retained.
    fn retain(&mut self, key: PathBuf, value: &Arc<T>, bytes: usize) {
        if bytes > self.budget_bytes {
            return;
        }
        while self.used_bytes + bytes > self.budget_bytes {
            let Some((_, _, evicted)) = self.entries.pop_front() else {
                break;
            };
            self.used_bytes -= evicted;
        }
        self.used_bytes += bytes;
        self.entries.push_back((key, Arc::clone(value), bytes));
    }

    const fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Least-recently-used cache of decoded scenes under a byte budget.
///
/// **Memory bound:** the pixel data the cache itself retains never exceeds
/// `budget_bytes` ([`cached_bytes`](Self::cached_bytes) `<=` budget). A scene whose
/// decoded size exceeds the budget is returned but not retained. Peak decoded
/// memory of a trainer is therefore `budget + one scene in flight`, plus the
/// linearised crop (`(4·planes + 4)·crop²·4` bytes). Each scene costs
/// `(4·planes + 4)·H·W` bytes (8-bit RGB frames, 8-bit masks, all-focus, occlusion),
/// e.g. 11 planes of 6000×4000 is about 1.1 GB.
///
/// With `budget_bytes = 0` nothing is retained and every use decodes the scene
/// once (`2·planes + 2` files), which is still `~(k+3)×` fewer decodes per
/// optimiser step than the per-rollout-step path it replaced.
#[derive(Debug)]
pub struct SceneCache {
    store: ByteBudgetLru<SceneImages>,
    decoded_files: usize,
}

impl SceneCache {
    /// A cache that retains at most `budget_bytes` of decoded pixel data.
    #[must_use]
    pub const fn new(budget_bytes: usize) -> Self {
        Self {
            store: ByteBudgetLru::new(budget_bytes),
            decoded_files: 0,
        }
    }

    /// The decoded scene for `seq`: a cache hit (no decoding) or a fresh
    /// [`SceneImages::load`] of all planes, retained if it fits the budget.
    ///
    /// # Errors
    /// Any [`DataError`] that [`SceneImages::load`] reports.
    pub fn get(&mut self, seq: &RolloutSequence) -> Result<Arc<SceneImages>, DataError> {
        if let Some(images) = self.store.lookup(&seq.scene_dir) {
            return Ok(images);
        }

        let images = Arc::new(SceneImages::load(seq, seq.meta.n_planes)?);
        self.decoded_files += images.decoded_files();
        self.store
            .retain(seq.scene_dir.clone(), &images, images.byte_size());
        Ok(images)
    }

    /// Total PNG files this cache has decoded so far (cache hits add nothing).
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

/// A scene cropped to the training window and linearised, ready to cut steps from.
struct LinearScene {
    h: usize,
    w: usize,
    /// Merge-order frames, linear CHW.
    frames: Vec<Vec<f32>>,
    /// Merge-order masks, HW in 0..1.
    masks: Vec<Vec<f32>>,
    allfocus: Vec<f32>,
    occlusion: Vec<f32>,
}

impl LinearScene {
    fn new(images: &SceneImages, crop: Option<CropParams>) -> Result<Self, DataError> {
        let window = crop_window(images.h, images.w, crop)?;
        let lut = srgb_lut();
        Ok(Self {
            h: window.2,
            w: window.3,
            frames: images
                .frames
                .iter()
                .map(|f| crop_rgb_linear(f, window, &lut))
                .collect(),
            masks: images.masks.iter().map(|m| crop_gray(m, window)).collect(),
            allfocus: crop_rgb_linear(&images.allfocus, window, &lut),
            occlusion: crop_gray(&images.occlusion, window),
        })
    }

    /// Materialise the whole stack as one batch sample.
    fn batch<B: Backend>(&self, device: &Device<B>) -> BatchSample<B> {
        let (h, w) = (self.h, self.w);
        let planes = self.frames.len();
        let stack = Tensor::<B, 4>::from_data(
            TensorData::new(self.frames.concat(), [planes, 3, h, w]),
            device,
        );
        let masks = Tensor::<B, 4>::from_data(
            TensorData::new(self.masks.concat(), [planes, 1, h, w]),
            device,
        );
        // The target is the pairwise strategy's final-step ground truth: the
        // all-focus render where some plane is in focus, the least-defocused plane
        // elsewhere (never asks for output sharper than every input shows).
        let gt = prefix_composite(
            &self.frames,
            &self.masks,
            &self.allfocus,
            planes,
            h,
            w,
            ALLFOCUS_MASK_THRESHOLD,
        );
        BatchSample {
            stack,
            gt_merged: make_tensor::<B>(gt, 3, h, w, device),
            masks,
            occlusion: make_tensor::<B>(self.occlusion.clone(), 1, h, w, device),
        }
    }

    /// Materialise step `k` (`k + 2 <= planes`).
    fn step<B: Backend>(&self, k: usize, device: &Device<B>) -> MergeSample<B> {
        let (h, w) = (self.h, self.w);
        let np = h * w;
        let n_target = k + 1;
        let n_gt = k + 2;
        let thr = ALLFOCUS_MASK_THRESHOLD;

        // Step 0 is what inference feeds the first merge: the raw first frame with
        // confidence 0. Later steps are teacher-forced from the oracle prefix.
        let (target, target_conf) = if k == 0 {
            (self.frames[0].clone(), vec![0f32; np])
        } else {
            let conf: Vec<f32> = (0..np)
                .map(|px| {
                    (0..n_target)
                        .map(|i| self.masks[i][px])
                        .fold(0f32, f32::max)
                })
                .collect();
            (
                prefix_composite(
                    &self.frames,
                    &self.masks,
                    &self.allfocus,
                    n_target,
                    h,
                    w,
                    thr,
                ),
                conf,
            )
        };
        let gt_merged =
            prefix_composite(&self.frames, &self.masks, &self.allfocus, n_gt, h, w, thr);
        let gt_conf: Vec<f32> = (0..np)
            .map(|px| (0..n_gt).map(|i| self.masks[i][px]).fold(0f32, f32::max))
            .collect();

        MergeSample {
            target: make_tensor::<B>(target, 3, h, w, device),
            target_conf: make_tensor::<B>(target_conf, 1, h, w, device),
            source: make_tensor::<B>(self.frames[k + 1].clone(), 3, h, w, device),
            gt_merged: make_tensor::<B>(gt_merged, 3, h, w, device),
            gt_conf: make_tensor::<B>(gt_conf, 1, h, w, device),
            occlusion: make_tensor::<B>(self.occlusion.clone(), 1, h, w, device),
        }
    }
}

// ---------------------------------------------------------------------------
// Rollout sequence
// ---------------------------------------------------------------------------

/// A lazy sequence of `n_planes - 1` pairwise-merge steps for one scene.
///
/// Tensors are materialised on demand via [`RolloutSequence::get_step`] to
/// keep memory usage bounded (important for large scenes).
#[derive(Debug, Clone)]
pub struct RolloutSequence {
    /// Root directory of the scene.
    pub scene_dir: PathBuf,
    /// Parsed scene metadata.
    pub meta: SceneMeta,
    /// Number of steps (`n_planes - 1`).
    pub n_steps: usize,
}

impl RolloutSequence {
    /// Build a rollout sequence by loading `metadata.json` from `scene_dir`.
    ///
    /// # Errors
    /// Any [`DataError`] that [`SceneMeta::load`] reports.
    pub fn new(scene_dir: PathBuf) -> Result<Self, DataError> {
        let meta = SceneMeta::load(&scene_dir)?;
        let n_steps = meta.n_planes - 1;
        Ok(Self {
            scene_dir,
            meta,
            n_steps,
        })
    }

    /// Materialise step `k` (0-indexed, `k < n_steps`) as Burn tensors.
    ///
    /// Tensors are placed on `device`.  Pass `crop = Some(…)` for patch-based
    /// training or `None` for full-image evaluation.
    ///
    /// Decodes only the `k + 2` planes this step needs (plus all-focus and
    /// occlusion); for whole-scene training use [`Self::crop_steps`].
    ///
    /// Fields returned:
    /// * `target`      – step 0: the raw first frame; `k > 0`:
    ///   `prefix_composite(0..=k)` → `[3, H, W]`
    /// * `target_conf` – step 0: zeros; `k > 0`: per-pixel max mask value over
    ///   `0..=k` → `[1, H, W]`
    /// * `source`      – `frame[merge_order[k+1]]` → `[3, H, W]`
    /// * `gt_merged`   – `prefix_composite(0..=k+1)` → `[3, H, W]`
    /// * `gt_conf`     – per-pixel max mask value over `0..=k+1` → `[1, H, W]`
    /// * `occlusion`   – scene occlusion mask → `[1, H, W]`
    ///
    /// # Errors
    /// Any [`DataError`] from decoding the planes, or [`DataError::CropTooLarge`]
    /// when `crop` exceeds the image.
    ///
    /// # Panics
    ///
    /// Panics if `k >= n_steps`.
    pub fn get_step<B: Backend>(
        &self,
        k: usize,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<MergeSample<B>, DataError> {
        assert!(
            k < self.n_steps,
            "step index {k} out of range (n_steps = {})",
            self.n_steps
        );

        // Only frames/masks 0..=k+1 are needed for this one step.
        let images = SceneImages::load(self, k + 2)?;
        Ok(LinearScene::new(&images, crop)?.step(k, device))
    }

    /// Materialise **all** `n_steps` steps of the scene from one decoded
    /// [`SceneImages`] — the training path.
    ///
    /// The images are cropped first and only the crop is linearised, so cost and
    /// memory scale with the crop (e.g. 256²), not the full frame. Combined with
    /// [`SceneCache`] each scene is decoded at most once per optimiser step (and
    /// not at all on a cache hit), instead of `2k + 7` PNGs per rollout step.
    ///
    /// Memory: `n_steps × 12 channels × crop² × 4 B` of tensors (about 3 MB per
    /// step at 256²) plus the linearised crop planes.
    ///
    /// # Errors
    /// [`DataError::InvalidMetadata`] when `images` holds fewer planes than the
    /// scene has, or [`DataError::CropTooLarge`] when `crop` exceeds the image.
    pub fn crop_steps<B: Backend>(
        &self,
        images: &SceneImages,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Vec<MergeSample<B>>, DataError> {
        if images.planes() < self.meta.n_planes {
            return Err(DataError::InvalidMetadata {
                path: self.scene_dir.join("metadata.json"),
                reason: format!(
                    "decoded scene holds {} planes, the scene has {}",
                    images.planes(),
                    self.meta.n_planes
                ),
            });
        }
        let linear = LinearScene::new(images, crop)?;
        Ok((0..self.n_steps)
            .map(|k| linear.step::<B>(k, device))
            .collect())
    }

    /// Materialise the whole stack as one [`BatchSample`] for the batch fusion
    /// network, decoding the scene first (`2·n_planes + 2` PNGs).
    ///
    /// The target is the pairwise strategy's final-step ground truth,
    /// `prefix_composite` over every plane: the all-focus render wherever some
    /// plane is in focus, the least-defocused plane elsewhere. Use
    /// [`Self::batch_from_images`] with a [`SceneCache`] to avoid decoding the
    /// scene again on every optimiser step.
    ///
    /// # Errors
    /// Any [`DataError`] from decoding the planes, or [`DataError::CropTooLarge`]
    /// when `crop` exceeds the image.
    pub fn get_batch<B: Backend>(
        &self,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<BatchSample<B>, DataError> {
        let images = SceneImages::load(self, self.meta.n_planes)?;
        self.batch_from_images(&images, device, crop)
    }

    /// [`Self::get_batch`] from an already decoded scene (the cached path). The
    /// images are cropped first and only the crop is linearised.
    ///
    /// # Errors
    /// [`DataError::InvalidMetadata`] when `images` holds fewer planes than the
    /// scene has, or [`DataError::CropTooLarge`] when `crop` exceeds the image.
    pub fn batch_from_images<B: Backend>(
        &self,
        images: &SceneImages,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<BatchSample<B>, DataError> {
        if images.planes() < self.meta.n_planes {
            return Err(DataError::InvalidMetadata {
                path: self.scene_dir.join("metadata.json"),
                reason: format!(
                    "decoded scene holds {} planes, the scene has {}",
                    images.planes(),
                    self.meta.n_planes
                ),
            });
        }
        Ok(LinearScene::new(images, crop)?.batch(device))
    }
}

/// Scan `root` for sub-directories containing a `metadata.json` and load each as a
/// [`RolloutSequence`] (the layout of `02_blender_focus_stack.py`), sorted by path.
///
/// # Errors
/// [`DataError::Io`] when `root` cannot be read, or any error of
/// [`RolloutSequence::new`] for a malformed scene.
pub fn discover_rollout_scenes(root: &Path) -> Result<Vec<RolloutSequence>, DataError> {
    align::scene_dirs(root)?
        .into_iter()
        .map(RolloutSequence::new)
        .collect()
}

/// A dataset yielding one [`BatchSample`] per scene found under a root directory.
#[derive(Debug)]
pub struct BatchStackDataset<B: Backend> {
    sequences: Vec<RolloutSequence>,
    device: Device<B>,
    /// Square window cut from every sample (`None`: whole frames). For training
    /// pass `Some(..)` and randomise `top`/`left` per epoch; for evaluation `None`.
    pub crop: Option<CropParams>,
}

impl<B: Backend> BatchStackDataset<B> {
    /// Scan `root_dir` for scene directories and build the dataset.
    ///
    /// # Errors
    /// [`DataError::Io`] when `root_dir` cannot be read, or any error of
    /// [`RolloutSequence::new`] for a malformed scene.
    pub fn from_root(
        root_dir: &Path,
        device: Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Self, DataError> {
        Ok(Self {
            sequences: discover_rollout_scenes(root_dir)?,
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

    /// Materialise the batch sample of scene `index`.
    ///
    /// # Errors
    /// As [`RolloutSequence::get_batch`].
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    pub fn get(&self, index: usize) -> Result<BatchSample<B>, DataError> {
        self.sequences[index].get_batch(&self.device, self.crop)
    }
}
