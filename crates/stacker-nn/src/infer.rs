//! Tiled inference utilities for large (8 K+) images.
//!
//! This module provides:
//! * [`TileConfig`] — parameters controlling tile size and overlap, plus
//!   [`TileConfig::for_model`] to derive a seam-safe overlap from a model's
//!   [`crate::traits::FusionModel::receptive_field`].
//! * [`fuse_stack`] / [`fuse_stack_with`] — fold a focus stack into a single
//!   all-in-focus image, generic over any [`crate::traits::FusionModel`]
//!   (not just the built-in [`FocusMergeNet`]). Dispatches on
//!   [`crate::traits::FusionModel::strategy`]: [`FusionStrategy::Pairwise`]
//!   models stream through [`merge_tiled`] frame-by-frame; [`FusionStrategy::Batch`]
//!   models go through [`fuse_batch_tiled`], which tiles the whole stack at
//!   once per tile (see that function's docs for its memory profile and
//!   progress-callback semantics, which differ from the pairwise path's).
//! * [`load_weights`] / [`load_weights_align`] / [`load_weights_fusion_align`]
//!   — deserialise a `.mpk` checkpoint into a [`FocusMergeNet`] /
//!   [`crate::model::BatchAlignNet`] / [`crate::model::FusionAlignNet`].

use std::f32::consts::PI;

use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{Tensor, TensorData},
};

use crate::{
    model::{FocusMergeNet, FocusMergeNetConfig},
    traits::{FusionModel, FusionStrategy},
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Tiling parameters for large-image inference.
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Square tile side length in pixels (model is run per tile).
    pub tile: usize,
    /// Apron / overlap (pixels) between adjacent tiles; blended with a
    /// raised-cosine window to suppress seams.
    pub overlap: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile: 512,
            overlap: 64,
        }
    }
}

impl TileConfig {
    /// Derive a seam-safe [`TileConfig`] for `model`, using the default tile
    /// size and an overlap taken from [`FusionModel::receptive_field`]
    /// (see that method's docs for why the overlap must be at least the
    /// receptive field).
    ///
    /// The overlap is clamped to at least 96 px — sized for the built-in
    /// `xxl` preset (see [`crate::runtime::recommended_tile`]) — so this
    /// helper never regresses tiling quality for [`FocusMergeNet`] callers,
    /// even though `FocusMergeNet::receptive_field` computes a slightly
    /// tighter value. A custom architecture with a genuinely larger
    /// receptive field gets a wider overlap than 96 automatically.
    #[must_use]
    pub fn for_model<B: Backend, M: FusionModel<B> + ?Sized>(model: &M) -> Self {
        Self {
            overlap: model.receptive_field().max(96),
            ..Self::default()
        }
    }
}

/// Errors that can arise during inference.
#[derive(Debug, thiserror::Error)]
pub enum InferError {
    /// The stack was empty; at least one frame is required.
    #[error("empty stack: need at least one frame")]
    EmptyStack,
    /// A frame's shape does not match frame 0.
    #[error("frame {index} shape {got:?} does not match frame 0 shape {expected:?}")]
    ShapeMismatch {
        /// Frame index that caused the mismatch.
        index: usize,
        /// Shape of frame 0.
        expected: Vec<usize>,
        /// Shape of the offending frame.
        got: Vec<usize>,
    },
    /// Weight loading failed.
    #[error("weight load error from {path}: {source}")]
    Record {
        /// Path to the checkpoint file.
        path: std::path::PathBuf,
        /// Underlying recorder error.
        #[source]
        source: burn::record::RecorderError,
    },
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Merge a focus stack into a single all-in-focus image by folding the network
/// over the stack.
///
/// The result starts as `frames[0]` (confidence 0), then each subsequent frame
/// is merged into the running composite via tiled inference.
///
/// `frames`: ordered stack, each `[3, H, W]`, f32 linear-light 0..1.
/// Returns the merged `[3, H, W]` composite.
///
/// Generic over any [`FusionModel`] — this is the extensibility
/// point third-party architectures plug into; see the trait's docs and the
/// crate docs' "Extending with your own architecture" section.
pub fn fuse_stack<B: Backend, M: FusionModel<B>>(
    model: &M,
    frames: &[Tensor<B, 3>],
    cfg: TileConfig,
    device: &B::Device,
) -> Result<Tensor<B, 3>, InferError> {
    fuse_stack_with(model, frames, cfg, device, |_, _| {})
}

/// Like [`fuse_stack`], but invokes `on_step` for progress reporting —
/// useful for a live preview. **The meaning of `on_step`'s arguments depends
/// on `model`'s [`FusionStrategy`]:**
///
/// * [`FusionStrategy::Pairwise`] — `on_step(frame_index, running_composite)`
///   is called once per input frame (including the initial frame at index 0,
///   before any merge), with the composite as folded so far. This is the
///   crate's original, finer-grained progress signal.
/// * [`FusionStrategy::Batch`] — [`fuse_batch_tiled`] calls
///   `on_step(tile_row_index, running_composite)` once per completed ROW of
///   tiles (not once per frame — a batch model has no per-frame partial
///   result, and once per individual tile would force an O(tiles·H·W)
///   full-image clone; see that function's docs). For an image that fits in
///   a single tile it is called exactly once, at completion.
///
/// In both cases the running composite is handed to the callback as an owned
/// `[3, H, W]` tensor that the callback may consume.
pub fn fuse_stack_with<B: Backend, M: FusionModel<B>, F>(
    model: &M,
    frames: &[Tensor<B, 3>],
    cfg: TileConfig,
    device: &B::Device,
    mut on_step: F,
) -> Result<Tensor<B, 3>, InferError>
where
    F: FnMut(usize, Tensor<B, 3>),
{
    if frames.is_empty() {
        return Err(InferError::EmptyStack);
    }

    let shape0 = frames[0].dims();

    // Validate all frames share frame 0's shape.
    for (index, frame) in frames.iter().enumerate().skip(1) {
        let got = frame.dims();
        if got != shape0 {
            return Err(InferError::ShapeMismatch {
                index,
                expected: shape0.to_vec(),
                got: got.to_vec(),
            });
        }
    }

    if model.strategy() == FusionStrategy::Batch {
        return Ok(fuse_batch_tiled(model, frames, cfg, device, on_step));
    }

    let [c, h, w] = shape0;

    // Start with frame 0 as the initial composite and zero confidence.
    let mut result = frames[0].clone().unsqueeze_dim::<4>(0); // [1, 3, H, W]
    let mut conf = Tensor::<B, 4>::zeros([1, 1, h, w], device); // [1, 1, H, W]
    on_step(0, result.clone().reshape([c, h, w]));

    for (index, frame) in frames.iter().enumerate().skip(1) {
        let source = frame.clone().unsqueeze_dim::<4>(0); // [1, 3, H, W]
        let (new_result, new_conf) = merge_tiled(model, result, conf, source, cfg, device);
        result = new_result;
        conf = new_conf;
        on_step(index, result.clone().reshape([c, h, w]));
    }

    // Squeeze batch dim: [1, 3, H, W] -> [3, H, W]
    Ok(result.reshape([c, h, w]))
}

/// Load trained weights from a `.mpk` checkpoint produced by `CompactRecorder`.
pub fn load_weights<B: Backend>(
    config: &FocusMergeNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<FocusMergeNet<B>, InferError> {
    let record = CompactRecorder::new()
        .load(path.to_path_buf(), device)
        .map_err(|source| InferError::Record {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(config.init(device).load_record(record))
}

/// Load trained batch-fusion weights (a [`crate::model::BatchMergeNet`]) from
/// a `.mpk` checkpoint produced by `CompactRecorder`.
///
/// Mirrors [`load_weights`] for the batch-fusion architecture — see that
/// function's docs and [`crate::discovery::ModelEntry::load_batch`], which is
/// the discovery-driven entry point most callers should prefer over calling
/// this directly.
pub fn load_weights_batch<B: Backend>(
    config: &crate::model::BatchMergeNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<crate::model::BatchMergeNet<B>, InferError> {
    let record = CompactRecorder::new()
        .load(path.to_path_buf(), device)
        .map_err(|source| InferError::Record {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(config.init(device).load_record(record))
}

/// Load trained alignment weights (a [`crate::model::BatchAlignNet`]) from a
/// `.mpk` checkpoint produced by `CompactRecorder`.
///
/// Mirrors [`load_weights`] for the alignment architecture — see that
/// function's docs and [`crate::discovery::ModelEntry::load_align`], which is
/// the discovery-driven entry point most callers should prefer over calling
/// this directly.
pub fn load_weights_align<B: Backend>(
    config: &crate::model::BatchAlignNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<crate::model::BatchAlignNet<B>, InferError> {
    let record = CompactRecorder::new()
        .load(path.to_path_buf(), device)
        .map_err(|source| InferError::Record {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(config.init(device).load_record(record))
}

/// Load trained pairwise-alignment weights (a
/// [`crate::model::FusionAlignNet`]) from a `.mpk` checkpoint produced by
/// `CompactRecorder`.
///
/// Mirrors [`load_weights_align`] for the pairwise alignment architecture —
/// see that function's docs and
/// [`crate::discovery::ModelEntry::load_fusion_align`], which is the
/// discovery-driven entry point most callers should prefer over calling this
/// directly.
pub fn load_weights_fusion_align<B: Backend>(
    config: &crate::model::FusionAlignNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<crate::model::FusionAlignNet<B>, InferError> {
    let record = CompactRecorder::new()
        .load(path.to_path_buf(), device)
        .map_err(|source| InferError::Record {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(config.init(device).load_record(record))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Run a single pairwise merge step with tiling.
///
/// Returns `(merged [1,3,H,W], conf [1,1,H,W])`.
fn merge_tiled<B: Backend, M: FusionModel<B>>(
    model: &M,
    target: Tensor<B, 4>,
    target_conf: Tensor<B, 4>,
    source: Tensor<B, 4>,
    cfg: TileConfig,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_n, _c, h, w] = target.dims();

    // Fast path: image fits in a single tile.
    if h <= cfg.tile && w <= cfg.tile {
        let out = model.merge(target, target_conf, source);
        return (out.merged, out.conf);
    }

    let tile = cfg.tile;
    let overlap = cfg.overlap.min(tile.saturating_sub(1));
    let step = tile.saturating_sub(overlap).max(1);

    // Generate tile start offsets covering 0..dim fully.
    let offsets_h = tile_offsets(h, tile, step);
    let offsets_w = tile_offsets(w, tile, step);

    // Accumulators for weighted blending.
    let mut merged_acc = Tensor::<B, 4>::zeros([1, 3, h, w], device);
    let mut conf_acc = Tensor::<B, 4>::zeros([1, 1, h, w], device);
    let mut weight_acc = Tensor::<B, 4>::zeros([1, 1, h, w], device);

    for &r in &offsets_h {
        let th = tile.min(h - r);
        let hann_row = hann_1d(th, device);

        for &c_off in &offsets_w {
            let tw = tile.min(w - c_off);
            let hann_col = hann_1d(tw, device);

            // Build the 2-D Hann window [1, 1, th, tw] via outer product.
            let win = outer_hann(&hann_row, &hann_col, th, tw, device);

            // Crop tensors to the tile region.
            let target_tile = target
                .clone()
                .slice([0..1, 0..3, r..r + th, c_off..c_off + tw]);
            let target_conf_tile =
                target_conf
                    .clone()
                    .slice([0..1, 0..1, r..r + th, c_off..c_off + tw]);
            let source_tile = source
                .clone()
                .slice([0..1, 0..3, r..r + th, c_off..c_off + tw]);

            // Run the model on this tile.
            let out = model.merge(target_tile, target_conf_tile, source_tile);

            // Weighted accumulation.
            let win3 = win.clone().expand([1, 3, th, tw]);
            let win1 = win.clone(); // already [1, 1, th, tw]

            // Read existing values, add contribution, write back.
            let cur_merged = merged_acc
                .clone()
                .slice([0..1, 0..3, r..r + th, c_off..c_off + tw]);
            merged_acc = merged_acc.slice_assign(
                [0..1, 0..3, r..r + th, c_off..c_off + tw],
                cur_merged.add(out.merged.mul(win3)),
            );

            let cur_conf = conf_acc
                .clone()
                .slice([0..1, 0..1, r..r + th, c_off..c_off + tw]);
            conf_acc = conf_acc.slice_assign(
                [0..1, 0..1, r..r + th, c_off..c_off + tw],
                cur_conf.add(out.conf.mul(win1.clone())),
            );

            let cur_weight = weight_acc
                .clone()
                .slice([0..1, 0..1, r..r + th, c_off..c_off + tw]);
            weight_acc = weight_acc.slice_assign(
                [0..1, 0..1, r..r + th, c_off..c_off + tw],
                cur_weight.add(win1),
            );
        }
    }

    // Normalise by accumulated weights.  Broadcasting: weight_acc [1,1,H,W]
    // broadcasts over the colour dimension of merged_acc [1,3,H,W].
    let merged_out = merged_acc.div(weight_acc.clone());
    let conf_out = conf_acc.div(weight_acc);

    (merged_out, conf_out)
}

/// Tiled whole-stack fusion for [`FusionStrategy::Batch`] models.
///
/// Unlike the pairwise path ([`merge_tiled`]), a batch model has no
/// incremental per-frame state to fold — [`FusionModel::fuse_batch`] always
/// consumes the entire stack at once. Tiling therefore slices EVERY frame to
/// the same spatial region and stacks just that region into a
/// `[1, S, 3, tile_h, tile_w]` tensor per tile, so at most one tile's worth of
/// the stack is resident on `device` at a time (VRAM scales with
/// `S · tile_h · tile_w`, not `S · H · W` — see the crate README's memory
/// note for the batch strategy). Frames themselves stay as the caller's
/// per-frame `Tensor<B, 3>` slices between tiles; only the current tile's
/// crop of each frame is concatenated into a batch tensor.
///
/// ## Progress semantics
///
/// `on_step(tile_row_index, running_composite)` fires once per completed ROW
/// of tiles (all tiles sharing a `[r, r+th)` vertical band), not once per
/// individual tile: computing a full-image blended preview after every
/// single tile would cost an extra O(`tiles · H · W`) of blend/normalise
/// work on top of the fusion itself, which is wasteful for a progress signal
/// alone. Once per row is a cheap middle ground — visible incremental
/// progress on large images without paying per-tile. For an image that fits
/// in a single tile, `on_step` fires exactly once, at completion, with row
/// index `0`.
// Always returns `Ok` today (the batch path has no failure mode of its own),
// but keeps `Result<_, InferError>` to match `fuse_stack_with`'s signature,
// which dispatches to both this and the pairwise path via a single `?`.
fn fuse_batch_tiled<B: Backend, M: FusionModel<B>, F>(
    model: &M,
    frames: &[Tensor<B, 3>],
    cfg: TileConfig,
    device: &B::Device,
    mut on_step: F,
) -> Tensor<B, 3>
where
    F: FnMut(usize, Tensor<B, 3>),
{
    let shape0 = frames[0].dims();
    let [c, h, w] = shape0;
    let s = frames.len();

    if h <= cfg.tile && w <= cfg.tile {
        let mut stack = Vec::with_capacity(s);
        for frame in frames {
            stack.push(frame.clone().unsqueeze_dim::<4>(0).unsqueeze_dim::<5>(0));
        }
        let stack_tensor = Tensor::cat(stack, 1);
        let merged = model.fuse_batch(stack_tensor);
        let out = merged.reshape([c, h, w]);
        on_step(0, out.clone());
        return out;
    }

    let tile = cfg.tile;
    let overlap = cfg.overlap.min(tile.saturating_sub(1));
    let step = tile.saturating_sub(overlap).max(1);

    let offsets_h = tile_offsets(h, tile, step);
    let offsets_w = tile_offsets(w, tile, step);

    let mut merged_acc = Tensor::<B, 4>::zeros([1, 3, h, w], device);
    let mut weight_acc = Tensor::<B, 4>::zeros([1, 1, h, w], device);

    for (row_idx, &r) in offsets_h.iter().enumerate() {
        let th = tile.min(h - r);
        let hann_row = hann_1d(th, device);

        for &c_off in &offsets_w {
            let tw = tile.min(w - c_off);
            let hann_col = hann_1d(tw, device);

            let win = outer_hann(&hann_row, &hann_col, th, tw, device);

            // Crop just this tile's region from each frame, THEN stack —
            // never materialises a full-resolution [1,S,3,H,W] tensor.
            let mut stack_tile = Vec::with_capacity(s);
            for frame in frames {
                let cropped = frame
                    .clone()
                    .slice([0..3, r..r + th, c_off..c_off + tw])
                    .unsqueeze_dim::<4>(0)
                    .unsqueeze_dim::<5>(0);
                stack_tile.push(cropped);
            }
            let stack_tile = Tensor::cat(stack_tile, 1);

            let merged_tile = model.fuse_batch(stack_tile);

            let win3 = win.clone().expand([1, 3, th, tw]);
            let win1 = win;

            let cur_merged = merged_acc
                .clone()
                .slice([0..1, 0..3, r..r + th, c_off..c_off + tw]);
            merged_acc = merged_acc.slice_assign(
                [0..1, 0..3, r..r + th, c_off..c_off + tw],
                cur_merged.add(merged_tile.mul(win3)),
            );

            let cur_weight = weight_acc
                .clone()
                .slice([0..1, 0..1, r..r + th, c_off..c_off + tw]);
            weight_acc = weight_acc.slice_assign(
                [0..1, 0..1, r..r + th, c_off..c_off + tw],
                cur_weight.add(win1),
            );
        }

        // One progress callback per completed tile ROW (see this function's
        // docs) — normalise the accumulator so far into a preview composite.
        let preview = merged_acc
            .clone()
            .div(weight_acc.clone().add_scalar(1e-6_f32));
        on_step(row_idx, preview.reshape([c, h, w]));
    }

    let merged_out = merged_acc.div(weight_acc);
    merged_out.reshape([c, h, w])
}

/// Compute the tile start offsets along one axis of length `dim`.
///
/// Steps by `step` starting at 0; always appends a final tile starting at
/// `dim - tile` to ensure the right/bottom border is fully covered.
fn tile_offsets(dim: usize, tile: usize, step: usize) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut pos = 0usize;
    while pos + tile <= dim {
        offsets.push(pos);
        pos += step;
    }
    // Always include a tile clamped to end at `dim`.
    let last = dim.saturating_sub(tile);
    if offsets.last().copied() != Some(last) {
        offsets.push(last);
    }
    // Edge case: dim < tile — just use offset 0 (the slice will be clamped).
    if offsets.is_empty() {
        offsets.push(0);
    }
    offsets
}

/// Build a 1-D Hann window of length `n` as a `[n]` tensor.
///
/// Formula: `w[i] = 0.5 - 0.5 * cos(2π*(i+0.5)/n)`, floored at `1e-3`.
fn hann_1d<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 1> {
    let n_f = n as f32;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let v = 0.5f32.mul_add(-(2.0 * PI * (i as f32 + 0.5) / n_f).cos(), 0.5);
            v.max(1e-3_f32)
        })
        .collect();
    Tensor::<B, 1>::from_data(TensorData::new(data, [n]), device)
}

/// Build a 2-D Hann window `[1, 1, th, tw]` as the outer product of two 1-D windows.
fn outer_hann<B: Backend>(
    row: &Tensor<B, 1>,
    col: &Tensor<B, 1>,
    th: usize,
    tw: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    // row: [th], col: [tw]
    // Reshape to [th, 1] and [1, tw] then multiply for [th, tw].
    let r = row.clone().reshape([th, 1]); // [th, 1]
    let c = col.clone().reshape([1, tw]); // [1, tw]
    let win2d = r.mul(c); // [th, tw]  (broadcast)
    // Add batch and channel dims: [1, 1, th, tw]
    win2d
        .unsqueeze_dim::<3>(0) // [1, th, tw]
        .unsqueeze_dim::<4>(0) // [1, 1, th, tw]
        .to_device(device)
}
