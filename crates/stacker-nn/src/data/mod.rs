//! Dataset loading, preprocessing, and sample construction for the crate's
//! four training strategies: pairwise fusion ([`FocusStackDataset`] /
//! [`RolloutSequence`]), batch fusion ([`BatchStackDataset`]), batch
//! alignment ([`AlignStackDataset`] / [`AlignSequence`]), and pairwise
//! alignment ([`PairAlignSample`], produced by [`AlignSequence::get_pair`] —
//! it reuses the exact same on-disk scenes/metadata as batch alignment, just
//! extracting one reference/frame pair at a time instead of materialising
//! the whole stack; see that method's docs and `docs/fusionalign-design.md`).
//!
//! ## On-disk layout — fusion scenes (pairwise + batch)
//!
//! ```text
//! scene/
//!   metadata.json          – scene descriptor (see [`SceneMeta`])
//!   frame_000.png          – input frames (sRGB PNG, 8-bit)
//!   frame_001.png
//!   …
//!   mask_000.png           – soft in-focus weight, grayscale 8- or 16-bit
//!                             (16-bit is downconverted to 8-bit on load)
//!   mask_001.png
//!   …
//!   allfocus.png           – ground-truth all-in-focus composite (sRGB PNG)
//!   occlusion.png          – depth-edge mask (grayscale PNG, 255 = edge)
//! ```
//!
//! This is the output of `scripts/02_blender_focus_stack.py`.
//!
//! ## On-disk layout — alignment scenes
//!
//! ```text
//! scene_unaligned/
//!   metadata.json          – scene descriptor (see [`AlignSceneMeta`]);
//!                             the stage-2 fields above PLUS `alignment_gt`
//!                             and `cropped_dims` (any additional
//!                             informational fields are ignored by serde)
//!   <stack entries>        – misaligned input frames (sRGB PNG, 8-bit)
//! ```
//!
//! This is the output of `scripts/03_simulate_misalignment.py` run on a
//! stage-2 scene. See [`AlignSceneMeta`]'s docs for the ground-truth matrix
//! convention.
//!
//! ## Linear-light policy
//!
//! RGB PNGs are decoded as 8-bit sRGB and **linearised** before any tensor
//! operation via [`to_linear`].  Grayscale masks represent weights (not colour)
//! and are normalised to 0..1 without additional gamma conversion.
//! Use [`from_linear`] to re-encode linear-light floats for visualisation.
//! Every tensor produced here is `f32`, linear-light, values in 0..1.
//!
//! ## Rollout sequence
//!
//! For a scene with `n_planes` frames the rollout yields `n_planes - 1` steps:
//!
//! ```text
//!   step 0: target_0 = prefix_composite(frames 0..=0)
//!           source_1 = frame[merge_order[1]]
//!           gt_1     = prefix_composite(frames 0..=1)
//!
//!   step k: target_k = gt_{k-1}   (output of previous step)
//!           source   = frame[merge_order[k+1]]
//!           gt_{k+1} = prefix_composite(frames 0..=k+1)
//! ```

use std::path::{Path, PathBuf};

use burn::{
    prelude::Backend,
    tensor::{Device, Tensor, TensorData},
};
use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// All errors that can arise while loading or constructing a dataset sample.
#[derive(Debug, Error)]
pub enum DataError {
    /// A required file was not found on disk.
    #[error("missing file: {0}")]
    MissingFile(PathBuf),

    /// An I/O error occurred while reading a file.
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The `image` crate could not decode a PNG.
    #[error("image decode error for {path}: {source}")]
    ImageDecode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    /// `metadata.json` could not be parsed.
    #[error("metadata parse error in {path}: {source}")]
    MetadataParse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// The metadata describes an inconsistent scene (wrong file counts, etc.).
    #[error("invalid metadata in {path}: {reason}")]
    InvalidMetadata { path: PathBuf, reason: String },

    /// A tensor shape mismatch was detected.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Requested crop size is larger than the image.
    #[error("crop size {crop} exceeds image dimension {dim}")]
    CropTooLarge { crop: usize, dim: usize },
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

/// Convert a linear-light value in 0..1 back to sRGB encoding.
#[inline]
#[must_use]
pub fn from_linear(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055_f32.mul_add(v.powf(1.0 / 2.4), -0.055)
    }
}

// ---------------------------------------------------------------------------
// Image loading helpers (private)
// ---------------------------------------------------------------------------

/// Load an RGB PNG and return CHW linear-light f32 data plus (H, W).
///
/// Layout: `data[c * H * W + y * W + x]` for channel `c`.
fn load_rgb_linear(path: &Path) -> Result<(Vec<f32>, usize, usize), DataError> {
    let img: RgbImage = image::open(path)
        .map_err(|e| DataError::ImageDecode {
            path: path.to_owned(),
            source: e,
        })?
        .into_rgb8();

    let w = img.width() as usize;
    let h = img.height() as usize;
    let np = h * w;
    let mut data = vec![0f32; 3 * np];

    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x as u32, y as u32);
            let base = y * w + x;
            data[base] = to_linear(f32::from(px[0]) / 255.0);
            data[np + base] = to_linear(f32::from(px[1]) / 255.0);
            data[2 * np + base] = to_linear(f32::from(px[2]) / 255.0);
        }
    }
    Ok((data, h, w))
}

/// Load a grayscale PNG and return flat HW f32 data plus (H, W).
///
/// Values are normalised to 0..1 without gamma conversion (masks are weights).
fn load_gray(path: &Path) -> Result<(Vec<f32>, usize, usize), DataError> {
    let img: GrayImage = image::open(path)
        .map_err(|e| DataError::ImageDecode {
            path: path.to_owned(),
            source: e,
        })?
        .into_luma8();

    let w = img.width() as usize;
    let h = img.height() as usize;
    let data: Vec<f32> = img.pixels().map(|p| f32::from(p[0]) / 255.0).collect();
    Ok((data, h, w))
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
    /// Current composite — shape `[3, H, W]`, linear RGB.
    pub target: Tensor<B, 3>,
    /// Per-pixel confidence of `target` — shape `[1, H, W]`.
    pub target_conf: Tensor<B, 3>,
    /// Next frame to merge in — shape `[3, H, W]`, linear RGB.
    pub source: Tensor<B, 3>,
    /// Ground-truth composite after merging `source` — shape `[3, H, W]`.
    pub gt_merged: Tensor<B, 3>,
    /// Ground-truth confidence / in-focus coverage for `gt_merged` — shape
    /// `[1, H, W]`.
    ///
    /// Per-pixel max mask value over the frames composited so far
    /// (`0..=k+1`). Used to supervise the network's predicted confidence.
    pub gt_conf: Tensor<B, 3>,
    /// Depth-edge / occlusion mask — shape `[1, H, W]` (1 = depth edge).
    pub occlusion: Tensor<B, 3>,
}

// ---------------------------------------------------------------------------
// Prefix composite
// ---------------------------------------------------------------------------

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

/// Crop a flat CHW buffer `[c, h, w]` to `[c, size, size]`.
fn crop_chw(
    data: &[f32],
    h: usize,
    w: usize,
    c: usize,
    cp: CropParams,
) -> Result<Vec<f32>, DataError> {
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
    let s = cp.size;
    let mut out = vec![0f32; c * s * s];
    for ci in 0..c {
        for row in 0..s {
            let src_row = (ci * h + cp.top + row) * w + cp.left;
            let dst_row = (ci * s + row) * s;
            out[dst_row..dst_row + s].copy_from_slice(&data[src_row..src_row + s]);
        }
    }
    Ok(out)
}

/// Crop a flat HW buffer `[h, w]` to `[size, size]`.
fn crop_hw(data: &[f32], h: usize, w: usize, cp: CropParams) -> Result<Vec<f32>, DataError> {
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
    let s = cp.size;
    let mut out = vec![0f32; s * s];
    for row in 0..s {
        let src = (cp.top + row) * w + cp.left;
        let dst = row * s;
        out[dst..dst + s].copy_from_slice(&data[src..src + s]);
    }
    Ok(out)
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
    /// Fields returned:
    /// * `target`      – `prefix_composite(0..=k)` → `[3, H, W]`
    /// * `target_conf` – per-pixel max mask value over `0..=k` → `[1, H, W]`
    /// * `source`      – `frame[merge_order[k+1]]` → `[3, H, W]`
    /// * `gt_merged`   – `prefix_composite(0..=k+1)` → `[3, H, W]`
    /// * `gt_conf`     – per-pixel max mask value over `0..=k+1` → `[1, H, W]`
    /// * `occlusion`   – scene occlusion mask → `[1, H, W]`
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

        let meta = &self.meta;
        let sd = &self.scene_dir;

        // Number of frames needed for target (k+1) and gt (k+2).
        let n_target = k + 1;
        let n_gt = k + 2;

        // Load frames and masks for indices merge_order[0..n_gt] only.
        let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_gt);
        let mut masks: Vec<Vec<f32>> = Vec::with_capacity(n_gt);
        let mut h = 0usize;
        let mut w = 0usize;

        for &fi in &meta.merge_order[..n_gt] {
            let fp = sd.join(&meta.stack[fi]);
            let (fdata, fh, fw) = load_rgb_linear(&fp)?;
            if h == 0 {
                h = fh;
                w = fw;
            } else if (fh, fw) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![3, h, w],
                    got: vec![3, fh, fw],
                });
            }
            frames.push(fdata);

            let mp = sd.join(&meta.masks[fi]);
            let (mdata, mh, mw) = load_gray(&mp)?;
            if (mh, mw) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![1, h, w],
                    got: vec![1, mh, mw],
                });
            }
            masks.push(mdata);
        }

        // Load allfocus and occlusion.
        let (af, afh, afw) = load_rgb_linear(&sd.join(&meta.allfocus))?;
        if (afh, afw) != (h, w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![3, h, w],
                got: vec![3, afh, afw],
            });
        }

        let (occ, occ_h, occ_w) = load_gray(&sd.join(&meta.occlusion))?;
        if (occ_h, occ_w) != (h, w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![1, h, w],
                got: vec![1, occ_h, occ_w],
            });
        }

        let thr = 0.1_f32;
        let np = h * w;

        // Prefix composite for target (frames 0..=k) and gt (frames 0..=k+1).
        let target_chw = prefix_composite(&frames, &masks, &af, n_target, h, w, thr);
        let gt_chw = prefix_composite(&frames, &masks, &af, n_gt, h, w, thr);

        // target_conf: per-pixel max mask value over the target set (0..=k).
        let conf_hw: Vec<f32> = (0..np)
            .map(|px| (0..n_target).map(|i| masks[i][px]).fold(0f32, f32::max))
            .collect();

        // gt_conf: per-pixel max mask value over the gt set (0..=k+1).
        let gt_conf_hw: Vec<f32> = (0..np)
            .map(|px| (0..n_gt).map(|i| masks[i][px]).fold(0f32, f32::max))
            .collect();

        // Source frame (re-load so we don't need to keep all frames in memory).
        let source_fi = meta.merge_order[k + 1];
        let (src_chw, sh, sw) = load_rgb_linear(&sd.join(&meta.stack[source_fi]))?;
        if (sh, sw) != (h, w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![3, h, w],
                got: vec![3, sh, sw],
            });
        }

        // Apply optional crop.
        let (out_h, out_w, tc, cc, sc, gc, gcc, oc) = match crop {
            Some(cp) => {
                let tc = crop_chw(&target_chw, h, w, 3, cp)?;
                let cc = crop_hw(&conf_hw, h, w, cp)?;
                let sc = crop_chw(&src_chw, h, w, 3, cp)?;
                let gc = crop_chw(&gt_chw, h, w, 3, cp)?;
                let gcc = crop_hw(&gt_conf_hw, h, w, cp)?;
                let oc = crop_hw(&occ, h, w, cp)?;
                (cp.size, cp.size, tc, cc, sc, gc, gcc, oc)
            }
            None => (h, w, target_chw, conf_hw, src_chw, gt_chw, gt_conf_hw, occ),
        };

        Ok(MergeSample {
            target: make_tensor::<B>(tc, 3, out_h, out_w, device),
            target_conf: make_tensor::<B>(cc, 1, out_h, out_w, device),
            source: make_tensor::<B>(sc, 3, out_h, out_w, device),
            gt_merged: make_tensor::<B>(gc, 3, out_h, out_w, device),
            gt_conf: make_tensor::<B>(gcc, 1, out_h, out_w, device),
            occlusion: make_tensor::<B>(oc, 1, out_h, out_w, device),
        })
    }
}

// ---------------------------------------------------------------------------
// FocusStackDataset
// ---------------------------------------------------------------------------

/// A Burn-compatible dataset over a collection of scene directories.
///
/// Scanning `root_dir` for subdirectories that contain `metadata.json`
/// builds a flat list of `(scene, step)` pairs — one per rollout step.
///
/// * For **training** pass `crop = Some(CropParams { size: 256, top: .., left: .. })`.
///   Randomise `top`/`left` per epoch externally (e.g. in a data-loader wrapper).
/// * For **evaluation** pass `crop = None` to get full-resolution tensors.
pub struct FocusStackDataset<B: Backend> {
    sequences: Vec<RolloutSequence>,
    /// Flat index: `items[i] = (sequence_idx, step_idx)`.
    items: Vec<(usize, usize)>,
    /// Burn device tensors are placed on.
    device: Device<B>,
    /// Spatial crop applied to every sample (`None` = full image).
    pub crop: Option<CropParams>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> FocusStackDataset<B> {
    /// Scan `root_dir` and construct the dataset.
    pub fn from_root(
        root_dir: &Path,
        device: Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Self, DataError> {
        let entries = std::fs::read_dir(root_dir).map_err(|e| DataError::Io {
            path: root_dir.to_owned(),
            source: e,
        })?;

        let mut sequences = Vec::new();
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("metadata.json").exists() {
                sequences.push(RolloutSequence::new(p)?);
            }
        }

        let items: Vec<(usize, usize)> = sequences
            .iter()
            .enumerate()
            .flat_map(|(si, seq)| (0..seq.n_steps).map(move |step| (si, step)))
            .collect();

        Ok(Self {
            sequences,
            items,
            device,
            crop,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Total number of `(scene, step)` samples.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the dataset is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Retrieve and materialise sample at `index`.
    pub fn get(&self, index: usize) -> Result<MergeSample<B>, DataError> {
        let (si, step) = self.items[index];
        self.sequences[si].get_step(step, &self.device, self.crop)
    }
}

// ---------------------------------------------------------------------------
// Batch processing additions
// ---------------------------------------------------------------------------

/// A single batched training sample with all tensors on the chosen backend.
///
/// All tensors are `f32`, linear-light (RGB channels) or unitless weights
/// (`masks`/`occlusion`), values in 0..1.
#[derive(Debug)]
pub struct BatchSample<B: Backend> {
    /// Full focus stack — shape `[S, 3, H, W]`, linear RGB.
    pub stack: Tensor<B, 4>,
    /// Ground-truth final composite — shape `[3, H, W]`, linear RGB.
    pub gt_merged: Tensor<B, 3>,
    /// Per-frame soft in-focus weight maps — shape `[S, 1, H, W]`.
    ///
    /// Re-indexed by `merge_order` exactly like `stack`'s frames (entry `i`
    /// corresponds to `merge_order[i]`), and cropped identically. Used by
    /// [`crate::loss::FocusBatchLoss`]'s gate-supervision term to build the
    /// target blending distribution.
    pub masks: Tensor<B, 4>,
    /// Depth-edge / occlusion mask — shape `[1, H, W]` (1 = depth edge).
    ///
    /// Cropped identically to `stack`/`masks`. Used by
    /// [`crate::loss::FocusBatchLoss`] to down-weight every term near depth
    /// edges, exactly like [`crate::loss::FocusFusionLoss`] does for the
    /// pairwise strategy.
    pub occlusion: Tensor<B, 3>,
}

impl RolloutSequence {
    /// Materialise the entire stack as a single batch tensor for the batch fusion model.
    ///
    /// Loads frames, per-frame masks, the ground-truth all-in-focus
    /// composite, and the scene's occlusion mask — the same set of inputs
    /// [`Self::get_step`] loads per-step, but for the WHOLE stack at once, so
    /// batch-fusion training can supervise the gate and down-weight depth
    /// edges the same way the pairwise strategy does.
    pub fn get_batch<B: Backend>(
        &self,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<BatchSample<B>, DataError> {
        let meta = &self.meta;
        let sd = &self.scene_dir;
        let n_planes = meta.n_planes;

        let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_planes);
        let mut masks: Vec<Vec<f32>> = Vec::with_capacity(n_planes);
        let mut h = 0usize;
        let mut w = 0usize;

        for &fi in &meta.merge_order {
            let fp = sd.join(&meta.stack[fi]);
            let (fdata, fh, fw) = load_rgb_linear(&fp)?;
            if h == 0 {
                h = fh;
                w = fw;
            } else if (fh, fw) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![3, h, w],
                    got: vec![3, fh, fw],
                });
            }
            frames.push(fdata);

            let mp = sd.join(&meta.masks[fi]);
            let (mdata, mh, mw) = load_gray(&mp)?;
            if (mh, mw) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![1, h, w],
                    got: vec![1, mh, mw],
                });
            }
            masks.push(mdata);
        }

        let (af, afh, afw) = load_rgb_linear(&sd.join(&meta.allfocus))?;
        if (afh, afw) != (h, w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![3, h, w],
                got: vec![3, afh, afw],
            });
        }

        let (occ, occ_h, occ_w) = load_gray(&sd.join(&meta.occlusion))?;
        if (occ_h, occ_w) != (h, w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![1, h, w],
                got: vec![1, occ_h, occ_w],
            });
        }

        let (out_h, out_w, frames_cropped, masks_cropped, af_cropped, occ_cropped) = match crop {
            Some(cp) => {
                let mut fc = Vec::with_capacity(n_planes);
                for fdata in frames {
                    fc.push(crop_chw(&fdata, h, w, 3, cp)?);
                }
                let mut mc = Vec::with_capacity(n_planes);
                for mdata in masks {
                    mc.push(crop_hw(&mdata, h, w, cp)?);
                }
                let ac = crop_chw(&af, h, w, 3, cp)?;
                let oc = crop_hw(&occ, h, w, cp)?;
                (cp.size, cp.size, fc, mc, ac, oc)
            }
            None => (h, w, frames, masks, af, occ),
        };

        // Flatten the frames_cropped into one big vector
        let mut stack_data = Vec::with_capacity(n_planes * 3 * out_h * out_w);
        for fc in frames_cropped {
            stack_data.extend_from_slice(&fc);
        }
        let stack_td = TensorData::new(stack_data, [n_planes, 3, out_h, out_w]);
        let stack = Tensor::<B, 4>::from_data(stack_td, device);

        let mut masks_data = Vec::with_capacity(n_planes * out_h * out_w);
        for mc in masks_cropped {
            masks_data.extend_from_slice(&mc);
        }
        let masks_td = TensorData::new(masks_data, [n_planes, 1, out_h, out_w]);
        let masks = Tensor::<B, 4>::from_data(masks_td, device);

        let gt_merged = make_tensor::<B>(af_cropped, 3, out_h, out_w, device);
        let occlusion = make_tensor::<B>(occ_cropped, 1, out_h, out_w, device);

        Ok(BatchSample {
            stack,
            gt_merged,
            masks,
            occlusion,
        })
    }
}

/// A Burn-compatible dataset yielding batched scene samples.
pub struct BatchStackDataset<B: Backend> {
    sequences: Vec<RolloutSequence>,
    device: Device<B>,
    /// Spatial crop applied to every sample (`None` = full image).
    ///
    /// Same semantics as [`FocusStackDataset::crop`]: for **training** pass
    /// `Some(CropParams { size, .. })` and randomise `top`/`left` per epoch
    /// externally; for **evaluation** pass `None` for full-resolution tensors.
    pub crop: Option<CropParams>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> BatchStackDataset<B> {
    /// Scan `root_dir` and construct the dataset.
    pub fn from_root(
        root_dir: &Path,
        device: Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Self, DataError> {
        let entries = std::fs::read_dir(root_dir).map_err(|e| DataError::Io {
            path: root_dir.to_owned(),
            source: e,
        })?;

        let mut sequences = Vec::new();
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("metadata.json").exists() {
                sequences.push(RolloutSequence::new(p)?);
            }
        }

        Ok(Self {
            sequences,
            device,
            crop,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Total number of scenes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Returns `true` if the dataset is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Retrieve and materialise the batched sample at `index`.
    pub fn get(&self, index: usize) -> Result<BatchSample<B>, DataError> {
        self.sequences[index].get_batch(&self.device, self.crop)
    }
}

// ---------------------------------------------------------------------------
// Alignment dataset
// ---------------------------------------------------------------------------
//
// On-disk layout (per scene directory, as emitted by
// `scripts/03_simulate_misalignment.py`):
//
//   scene_unaligned/
//     metadata.json     - the stage-2 SceneMeta fields PLUS:
//                            alignment_gt: one 3x3 row-major matrix per
//                                          `stack` entry
//                            crop_factor:   f32
//                            original_dims: [u32; 2]  (W, H before cropping)
//                            cropped_dims:  [u32; 2]  (w, h - every emitted
//                                                       image's actual size)
//     <stack[i]>         - misaligned frame i (sRGB PNG), size == cropped_dims
//
// Coordinate convention (see `scripts/03_simulate_misalignment.py`'s module
// docstring for the derivation): `alignment_gt[k]` is a PIXEL-space matrix,
// in the shared `cropped_dims` [w, h] space, mapping a reference-frame pixel
// coordinate to the corresponding coordinate in `stack[k]`'s emitted frame:
// `[x_k, y_k, 1]^T = M_k @ [x_ref, y_ref, 1]^T`.
//
// `AlignSample::gt_matrices` converts each `M_k` to the crate-wide NORMALIZED
// [-1, 1]^2 convention that `BatchAlignmentModel`/`BatchAlignmentLoss`
// operate in (see `crate::bridge::align_planar`'s docs for the exact
// pixel <-> normalized conjugation `M_n = N @ M_px @ N^-1`, where `N` maps
// pixel homogeneous coordinates to normalized ones). This mirrors
// `align_planar`'s conversion but in the opposite direction (pixel ->
// normalized here; normalized -> pixel there), using the SAME `N` for both
// the "domain" (reference) and "codomain" (frame k) spaces since they share
// `cropped_dims`.

/// Metadata for an alignment-training scene, as emitted by
/// `scripts/03_simulate_misalignment.py`.
///
/// Holds the stage-2 [`SceneMeta`] fields (frame filenames etc.) plus the
/// ground-truth alignment matrices and the crop bookkeeping needed to
/// interpret them.
///
/// Deserialises directly from the augmented `metadata.json`; fields the
/// alignment loader does not need (`masks`, `allfocus`, `occlusion`,
/// `merge_order`, `focus_fractions`, `crop_factor`, `original_dims`) are
/// simply not part of this struct and are ignored by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct AlignSceneMeta {
    /// Number of frames in the stack.
    pub n_planes: usize,
    /// PNG filenames of the (misaligned) input frames, in stack order.
    pub stack: Vec<String>,
    /// Ground-truth alignment matrices, one per `stack` entry, row-major
    /// `[[f32; 3]; 3]`, in the PIXEL-space convention described in the
    /// module docs above.
    pub alignment_gt: Vec<[[f32; 3]; 3]>,
    /// `[width, height]` shared by every emitted image in this scene
    /// (the space `alignment_gt`'s matrices operate in).
    pub cropped_dims: [u32; 2],
}

impl AlignSceneMeta {
    /// Load and parse `metadata.json` from `scene_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::Io`] if the file cannot be read,
    /// [`DataError::MetadataParse`] if it is not valid JSON matching this
    /// schema, or [`DataError::InvalidMetadata`] if `stack`/`alignment_gt`
    /// lengths disagree with `n_planes`.
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
        let n = self.n_planes;
        if self.stack.len() != n {
            return Err(DataError::InvalidMetadata {
                path: meta_path.to_owned(),
                reason: format!(
                    "`stack` has length {}, expected n_planes={n}",
                    self.stack.len()
                ),
            });
        }
        if self.alignment_gt.len() != n {
            return Err(DataError::InvalidMetadata {
                path: meta_path.to_owned(),
                reason: format!(
                    "`alignment_gt` has length {}, expected n_planes={n}",
                    self.alignment_gt.len()
                ),
            });
        }
        if n < 1 {
            return Err(DataError::InvalidMetadata {
                path: meta_path.to_owned(),
                reason: "n_planes must be >= 1".to_string(),
            });
        }
        Ok(())
    }
}

/// Convert a pixel-space `[width, height]`-space affine matrix to the
/// crate-wide normalized `[-1, 1]^2` convention.
///
/// This is the inverse direction of the conjugation
/// [`crate::bridge::align_planar`] applies (which goes normalized -> pixel);
/// see that function's docs for the derivation of `N`. Given pixel-space `M`,
/// the normalized-space matrix is `M_n = N @ M @ N^-1` where:
///
/// ```text
/// N = [ 2/(W-1)    0      -1 ]      N^-1 = [ (W-1)/2    0     (W-1)/2 ]
///     [   0     2/(H-1)   -1 ]             [   0     (H-1)/2  (H-1)/2 ]
///     [   0        0       1 ]             [   0        0        1   ]
/// ```
///
/// `w`/`h` must be `>= 2` (a single-pixel dimension has no well-defined
/// normalized extent); callers are expected to validate this via real scene
/// dimensions.
#[must_use]
fn pixel_matrix_to_normalized(m: [[f32; 3]; 3], w: usize, h: usize) -> [[f32; 3]; 3] {
    let sx = 2.0_f32 / (w as f32 - 1.0);
    let sy = 2.0_f32 / (h as f32 - 1.0);
    // n = N @ m  (apply N's scale+shift to m's rows 0 and 1; row 2 is [0,0,1]).
    let n00 = sx * m[0][0];
    let n01 = sx * m[0][1];
    let n02 = sx * m[0][2] - 1.0;
    let n10 = sy * m[1][0];
    let n11 = sy * m[1][1];
    let n12 = sy * m[1][2] - 1.0;

    // result = n @ N^-1  (apply N^-1's scale+shift to n's columns 0 and 1;
    // column 2 absorbs the translation contributed by N^-1's last column).
    let ix = (w as f32 - 1.0) / 2.0;
    let iy = (h as f32 - 1.0) / 2.0;
    [
        [n00 * ix, n01 * iy, n00 * ix + n01 * iy + n02],
        [n10 * ix, n11 * iy, n10 * ix + n11 * iy + n12],
        [0.0, 0.0, 1.0],
    ]
}

/// A single alignment-training sample with all tensors on the chosen
/// backend.
///
/// All matrices are in the crate-wide normalized `[-1, 1]^2` convention (see
/// the module docs above and [`crate::bridge::align_planar`]).
#[derive(Debug)]
pub struct AlignSample<B: Backend> {
    /// Full (misaligned) focus stack — shape `[S, 3, H, W]`, linear RGB.
    pub stack: Tensor<B, 4>,
    /// Ground-truth alignment matrices, normalized-space — shape `[S, 3, 3]`.
    pub gt_matrices: Tensor<B, 3>,
}

/// A lazy alignment-training scene: one focus stack plus its ground-truth
/// registration matrices, as emitted by `scripts/03_simulate_misalignment.py`.
#[derive(Debug, Clone)]
pub struct AlignSequence {
    /// Root directory of the scene.
    pub scene_dir: PathBuf,
    /// Parsed scene metadata.
    pub meta: AlignSceneMeta,
}

impl AlignSequence {
    /// Build an alignment sequence by loading `metadata.json` from `scene_dir`.
    pub fn new(scene_dir: PathBuf) -> Result<Self, DataError> {
        let meta = AlignSceneMeta::load(&scene_dir)?;
        Ok(Self { scene_dir, meta })
    }

    /// Materialise the whole scene as an [`AlignSample`].
    ///
    /// Tensors are placed on `device`. Pass `crop = Some(…)` for patch-based
    /// training or `None` for full-image evaluation.
    ///
    /// **The ground-truth matrices are always computed from the ORIGINAL
    /// (uncropped) frame dimensions recorded in `metadata.json`**, not the
    /// crop size — the normalized `[-1, 1]^2` convention is anchored to the
    /// full frame, so a training crop must not perturb it (a crop changes
    /// what the loaded image tensor covers, not the coordinate system the
    /// matrices are defined in).
    pub fn get<B: Backend>(
        &self,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<AlignSample<B>, DataError> {
        let meta = &self.meta;
        let sd = &self.scene_dir;
        let n_planes = meta.n_planes;
        let [full_w, full_h] = meta.cropped_dims;
        let (full_w, full_h) = (full_w as usize, full_h as usize);

        let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_planes);
        let mut h = 0usize;
        let mut w = 0usize;

        for fname in &meta.stack {
            let fp = sd.join(fname);
            let (fdata, fh, fw) = load_rgb_linear(&fp)?;
            if h == 0 {
                h = fh;
                w = fw;
            } else if (fh, fw) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![3, h, w],
                    got: vec![3, fh, fw],
                });
            }
            frames.push(fdata);
        }
        if (h, w) != (full_h, full_w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![3, full_h, full_w],
                got: vec![3, h, w],
            });
        }

        let (out_h, out_w, frames_cropped) = match crop {
            Some(cp) => {
                let mut fc = Vec::with_capacity(n_planes);
                for fdata in frames {
                    fc.push(crop_chw(&fdata, h, w, 3, cp)?);
                }
                (cp.size, cp.size, fc)
            }
            None => (h, w, frames),
        };

        let mut stack_data = Vec::with_capacity(n_planes * 3 * out_h * out_w);
        for fc in frames_cropped {
            stack_data.extend_from_slice(&fc);
        }
        let stack_td = TensorData::new(stack_data, [n_planes, 3, out_h, out_w]);
        let stack = Tensor::<B, 4>::from_data(stack_td, device);

        // Convert every ground-truth matrix to normalized space using the
        // FULL (uncropped) frame dimensions — see this method's docs.
        let mut mat_data = Vec::with_capacity(n_planes * 9);
        for m in &meta.alignment_gt {
            let n = pixel_matrix_to_normalized(*m, full_w, full_h);
            for row in n {
                mat_data.extend_from_slice(&row);
            }
        }
        let mat_td = TensorData::new(mat_data, [n_planes, 3, 3]);
        let gt_matrices = Tensor::<B, 3>::from_data(mat_td, device);

        Ok(AlignSample { stack, gt_matrices })
    }

    /// Materialise ONE reference/frame pair from this scene as a
    /// [`PairAlignSample`], for training [`crate::model::FusionAlignNet`]
    /// (the pairwise alignment architecture).
    ///
    /// `frame_index` selects which of `meta.stack`'s entries becomes
    /// `frame` (the reference is always `stack[0]`, matching
    /// [`crate::bridge::align_planar_pairwise`]'s convention that frame 0 of
    /// a stack is the alignment reference). Passing `frame_index = 0`
    /// registers the reference against itself — a legitimate training
    /// sample (see [`crate::model::FusionAlignNet`]'s docs on why
    /// near-identity, rather than construction-guaranteed exact identity,
    /// is what training is expected to produce for that case).
    ///
    /// Reuses [`AlignSceneMeta`]'s same PNG-loading, cropping, and
    /// pixel→normalized matrix conversion as [`Self::get`] — this is
    /// deliberately just a 2-frame slice of the exact same on-disk scene
    /// format, not a separate pairwise dataset layout, so a single
    /// `scripts/03_simulate_misalignment.py` run can train either
    /// architecture. The ground-truth matrix, like [`Self::get`]'s, is
    /// always computed from the ORIGINAL (uncropped) `cropped_dims`, not the
    /// crop size.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::InvalidMetadata`] if `frame_index >=
    /// meta.n_planes`, or any of [`Self::get`]'s I/O/shape errors.
    pub fn get_pair<B: Backend>(
        &self,
        frame_index: usize,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<PairAlignSample<B>, DataError> {
        let meta = &self.meta;
        if frame_index >= meta.n_planes {
            return Err(DataError::InvalidMetadata {
                path: self.scene_dir.join("metadata.json"),
                reason: format!(
                    "frame_index {frame_index} out of range for n_planes={}",
                    meta.n_planes
                ),
            });
        }

        let sd = &self.scene_dir;
        let [full_w, full_h] = meta.cropped_dims;
        let (full_w, full_h) = (full_w as usize, full_h as usize);

        let load_one = |fname: &str| -> Result<(Vec<f32>, usize, usize), DataError> {
            let fp = sd.join(fname);
            load_rgb_linear(&fp)
        };

        let (ref_data, ref_h, ref_w) = load_one(&meta.stack[0])?;
        let (frm_data, frm_h, frm_w) = load_one(&meta.stack[frame_index])?;
        if (ref_h, ref_w) != (full_h, full_w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![3, full_h, full_w],
                got: vec![3, ref_h, ref_w],
            });
        }
        if (frm_h, frm_w) != (full_h, full_w) {
            return Err(DataError::ShapeMismatch {
                expected: vec![3, full_h, full_w],
                got: vec![3, frm_h, frm_w],
            });
        }

        let (out_h, out_w, ref_data, frm_data) = match crop {
            Some(cp) => (
                cp.size,
                cp.size,
                crop_chw(&ref_data, full_h, full_w, 3, cp)?,
                crop_chw(&frm_data, full_h, full_w, 3, cp)?,
            ),
            None => (full_h, full_w, ref_data, frm_data),
        };

        let reference =
            Tensor::<B, 3>::from_data(TensorData::new(ref_data, [3, out_h, out_w]), device);
        let frame = Tensor::<B, 3>::from_data(TensorData::new(frm_data, [3, out_h, out_w]), device);

        // Same full-resolution normalisation as `Self::get` — the crop must
        // not perturb the coordinate system the ground-truth matrix lives in.
        let m_n = pixel_matrix_to_normalized(meta.alignment_gt[frame_index], full_w, full_h);
        let mut mat_data = Vec::with_capacity(9);
        for row in m_n {
            mat_data.extend_from_slice(&row);
        }
        let gt_matrix = Tensor::<B, 2>::from_data(TensorData::new(mat_data, [3, 3]), device);

        Ok(PairAlignSample {
            reference,
            frame,
            gt_matrix,
        })
    }
}

/// A single pairwise-alignment training sample: one reference/frame pair
/// plus the ground-truth registration matrix, unbatched (`N`-dimension-free,
/// same convention as [`MergeSample`]) — the pairwise analogue of
/// [`AlignSample`], for training [`crate::model::FusionAlignNet`].
///
/// All tensors are in the crate-wide normalized `[-1, 1]^2` convention (see
/// the module docs above and [`crate::bridge::align_planar_pairwise`]).
#[derive(Debug)]
pub struct PairAlignSample<B: Backend> {
    /// Reference frame — shape `[3, H, W]`, linear RGB.
    pub reference: Tensor<B, 3>,
    /// Moving frame to register against `reference` — shape `[3, H, W]`,
    /// linear RGB.
    pub frame: Tensor<B, 3>,
    /// Ground-truth alignment matrix, normalized-space — shape `[3, 3]`.
    pub gt_matrix: Tensor<B, 2>,
}

// ---------------------------------------------------------------------------
// Real (unlabelled) stacks — §5.2 photometric fine-tuning input
// ---------------------------------------------------------------------------

/// A real, unlabelled focus stack for the §5.2 photometric fine-tuning phase
/// (`docs/batchalign-v2-design.md`): "plain image-folder scenes — the loader
/// needs frames only, no metadata."
///
/// Unlike [`AlignSequence`] (synthetic training data with ground-truth
/// matrices), a [`RealStackScene`] is just a directory of frame images in
/// stack order, sorted by filename — no `metadata.json`, no ground truth.
/// Used only by [`crate::loss::photometric_gradient_loss`] via the model's
/// OWN predicted matrices, never by [`crate::loss::CornerAlignmentLoss`].
#[derive(Debug, Clone)]
pub struct RealStackScene {
    /// Root directory of the scene.
    pub scene_dir: PathBuf,
    /// Frame file paths, in stack order (sorted by filename).
    pub frames: Vec<PathBuf>,
}

impl RealStackScene {
    /// Common raster image extensions accepted when scanning a real-stack
    /// directory (case-insensitive).
    const IMAGE_EXTENSIONS: [&'static str; 4] = ["png", "jpg", "jpeg", "tif"];

    /// Scan `scene_dir` for image files (see [`Self::IMAGE_EXTENSIONS`]),
    /// sorted by filename, and treat them as one focus stack in that order.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::Io`] if `scene_dir` cannot be read.
    pub fn new(scene_dir: PathBuf) -> Result<Self, DataError> {
        let entries = std::fs::read_dir(&scene_dir).map_err(|e| DataError::Io {
            path: scene_dir.clone(),
            source: e,
        })?;
        let mut frames: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()).is_some_and(|ext| {
                    Self::IMAGE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
                })
            })
            .collect();
        frames.sort();
        Ok(Self { scene_dir, frames })
    }

    /// Number of frames found.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if no frames were found.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Materialise the whole stack as a plain `[S, 3, H, W]` tensor (no
    /// ground truth — see the type docs). Every frame must share the same
    /// dimensions; pass `crop = Some(...)` for patch-based fine-tuning.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::ImageDecode`]/[`DataError::Io`] on a bad frame,
    /// or [`DataError::ShapeMismatch`] if frame dimensions disagree.
    pub fn get<B: Backend>(
        &self,
        device: &Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Tensor<B, 4>, DataError> {
        let n = self.frames.len();
        let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut h = 0usize;
        let mut w = 0usize;
        for fp in &self.frames {
            let (fdata, fh, fw) = load_rgb_linear(fp)?;
            if h == 0 {
                h = fh;
                w = fw;
            } else if (fh, fw) != (h, w) {
                return Err(DataError::ShapeMismatch {
                    expected: vec![3, h, w],
                    got: vec![3, fh, fw],
                });
            }
            frames.push(fdata);
        }

        let (out_h, out_w, frames_cropped) = match crop {
            Some(cp) => {
                let mut fc = Vec::with_capacity(n);
                for fdata in frames {
                    fc.push(crop_chw(&fdata, h, w, 3, cp)?);
                }
                (cp.size, cp.size, fc)
            }
            None => (h, w, frames),
        };

        let mut stack_data = Vec::with_capacity(n * 3 * out_h * out_w);
        for fc in frames_cropped {
            stack_data.extend_from_slice(&fc);
        }
        let stack_td = TensorData::new(stack_data, [n, 3, out_h, out_w]);
        Ok(Tensor::<B, 4>::from_data(stack_td, device))
    }
}

/// Scan `root_dir` for sub-directories to use as [`RealStackScene`]s (§5.2
/// fine-tuning data): unlike [`AlignStackDataset`], a real-stack root has NO
/// `metadata.json` gate — every sub-directory containing at least one
/// recognised image file (see [`RealStackScene::IMAGE_EXTENSIONS`]) is
/// treated as one scene.
///
/// # Errors
///
/// Returns [`DataError::Io`] if `root_dir` cannot be read.
pub fn discover_real_stacks(root_dir: &Path) -> Result<Vec<RealStackScene>, DataError> {
    let entries = std::fs::read_dir(root_dir).map_err(|e| DataError::Io {
        path: root_dir.to_owned(),
        source: e,
    })?;
    let mut scenes = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let scene = RealStackScene::new(p)?;
            if !scene.is_empty() {
                scenes.push(scene);
            }
        }
    }
    scenes.sort_by(|a, b| a.scene_dir.cmp(&b.scene_dir));
    Ok(scenes)
}

/// A Burn-compatible dataset yielding [`AlignSample`]s, one per scene.
///
/// Scanning `root_dir` for subdirectories that contain a `metadata.json`
/// matching the [`AlignSceneMeta`] schema (i.e. the output of
/// `scripts/03_simulate_misalignment.py`) builds the scene list.
pub struct AlignStackDataset<B: Backend> {
    sequences: Vec<AlignSequence>,
    device: Device<B>,
    /// Spatial crop applied to every sample's frame tensors (`None` = full
    /// image).
    ///
    /// The ground-truth matrices are unaffected by this — see
    /// [`AlignSequence::get`]'s docs.
    pub crop: Option<CropParams>,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> AlignStackDataset<B> {
    /// Scan `root_dir` and construct the dataset.
    ///
    /// # Errors
    ///
    /// Returns [`DataError::Io`] if `root_dir` cannot be read, or any error
    /// from [`AlignSequence::new`] if a scene's `metadata.json` is malformed.
    pub fn from_root(
        root_dir: &Path,
        device: Device<B>,
        crop: Option<CropParams>,
    ) -> Result<Self, DataError> {
        let entries = std::fs::read_dir(root_dir).map_err(|e| DataError::Io {
            path: root_dir.to_owned(),
            source: e,
        })?;

        let mut sequences = Vec::new();
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("metadata.json").exists() {
                sequences.push(AlignSequence::new(p)?);
            }
        }

        Ok(Self {
            sequences,
            device,
            crop,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Total number of scenes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Returns `true` if the dataset is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Retrieve and materialise the sample at `index`.
    pub fn get(&self, index: usize) -> Result<AlignSample<B>, DataError> {
        self.sequences[index].get(&self.device, self.crop)
    }
}
