//! Tiled inference utilities for large (8 K+) images.
//!
//! This module provides:
//! * [`TileConfig`] — parameters controlling tile size and overlap.
//! * [`fuse_stack`] — fold a focus stack into a single all-in-focus image with
//!   any [`FusionModel`]; [`fuse_stream_with`] is the same over a lazy iterator of
//!   frames. A pairwise model ([`FusionStrategy::Pairwise`]) folds frame by frame
//!   (one tiled merge per step); a batch model ([`FusionStrategy::Batch`]) runs
//!   once per tile over the whole stack ([`fuse_batch_tiled`]).
//! * [`load_weights`] and friends — deserialise a `.mpk` checkpoint (any
//!   precision) into one of the four networks.

use std::f32::consts::PI;

use burn::{
    prelude::*,
    record::Recorder,
    tensor::{Tensor, TensorData},
};

use crate::{
    model::{
        BatchAlignNet, BatchAlignNetConfig, BatchMergeNet, BatchMergeNetConfig, FocusMergeNet,
        FocusMergeNetConfig, FusionAlignNet, FusionAlignNetConfig, ModelSize,
        receptive_field_radius,
    },
    traits::{FusionModel, FusionStrategy},
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Tiling parameters for large-image inference.
///
/// The model's output at a pixel depends only on the inputs within its
/// receptive-field radius `R` ([`receptive_field_radius`]; 16 px for `xs` up to
/// 100 px for `xxl`) because every normalisation layer is per-pixel
/// ([`crate::model::ChannelNorm`]). Tiles overlapping by `2·R` therefore give every
/// pixel at least one tile that sees its full context, and the raised-cosine blend
/// tapers each tile's truncated-context edge to zero weight. Build the config with
/// [`TileConfig::for_size`] / [`TileConfig::for_radius`] / [`TileConfig::for_model`]
/// rather than guessing an overlap.
///
/// For a batch model the memory per tile is `S · width · tile²` activations; use a
/// smaller `tile` than the 512 default for the `xl` / `xxl` presets and long
/// stacks (the overlap `2·R` stays the lower bound).
#[derive(Debug, Clone, Copy)]
pub struct TileConfig {
    /// Square tile side length in pixels (model is run per tile).
    pub tile: usize,
    /// Apron / overlap (pixels) between adjacent tiles; blended with a
    /// raised-cosine window to suppress seams. Should be at least `2·R`.
    pub overlap: usize,
}

impl TileConfig {
    /// Preferred tile side; grown only when the overlap would leave less than half
    /// a tile of new pixels per step.
    pub const PREFERRED_TILE: usize = 512;

    /// Tiling for a model with receptive-field radius `radius`: overlap `2·radius`,
    /// tile [`Self::PREFERRED_TILE`] (or `2·overlap` when that is larger, so the
    /// stride never drops below the overlap).
    #[must_use]
    pub const fn for_radius(radius: usize) -> Self {
        let overlap = 2 * radius;
        let tile = if 2 * overlap > Self::PREFERRED_TILE {
            2 * overlap
        } else {
            Self::PREFERRED_TILE
        };
        Self { tile, overlap }
    }

    /// Tiling for a [`ModelSize`] preset: `for_radius(receptive_field_radius(size))`.
    #[must_use]
    pub const fn for_size(size: ModelSize) -> Self {
        Self::for_radius(receptive_field_radius(size))
    }

    /// Tiling for a resident model: `for_radius(model.receptive_field())`, the exact
    /// radius of that network (not the preset's).
    #[must_use]
    pub fn for_model<B: Backend>(model: &impl FusionModel<B>) -> Self {
        Self::for_radius(model.receptive_field())
    }
}

impl Default for TileConfig {
    /// Safe for every preset: the overlap of the largest one (`xxl`, `R = 100`).
    fn default() -> Self {
        Self::for_size(ModelSize::Xxl)
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
    /// Weight loading failed. A checkpoint written before the tile-invariant
    /// normalisation change (`GroupNorm` -> per-pixel `ChannelNorm`) has different
    /// parameter names and fails here; such models must be retrained.
    #[error(
        "weight load error from {path}: {source} (a checkpoint trained before the per-pixel \
         normalisation change is incompatible and must be retrained)"
    )]
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

/// Merge a focus stack into a single all-in-focus image.
///
/// For a pairwise model the result starts as `frames[0]` (confidence 0) and each
/// subsequent frame is merged into the running composite via tiled inference; a
/// batch model fuses the whole stack tile by tile (see [`fuse_batch_tiled`]).
///
/// `frames`: ordered stack, each `[3, H, W]`, f32 linear-light 0..1.
/// Returns the merged `[3, H, W]` composite.
///
/// # Errors
/// [`InferError::EmptyStack`] for no frames and [`InferError::ShapeMismatch`]
/// when a frame's shape differs from frame 0's.
pub fn fuse_stack<B: Backend>(
    model: &impl FusionModel<B>,
    frames: &[Tensor<B, 3>],
    cfg: TileConfig,
    device: &B::Device,
) -> Result<Tensor<B, 3>, InferError> {
    fuse_stack_with(model, frames, cfg, device, |_, _| {})
}

/// Like [`fuse_stack`], but invokes `on_step` for progress reporting — useful for
/// a live preview. **The meaning of the arguments depends on the model's
/// [`FusionStrategy`]:**
///
/// * [`FusionStrategy::Pairwise`] — `on_step(frame_index, running_composite)` is
///   called once per input frame (including the initial frame at index 0, before
///   any merge), with the composite folded so far.
/// * [`FusionStrategy::Batch`] — `on_step(tile_row_index, running_composite)` is
///   called once per completed ROW of tiles, see [`fuse_batch_tiled`]. For an
///   image that fits in a single tile it is called exactly once, at completion.
///
/// In both cases the composite is handed over as an owned `[3, H, W]` tensor that
/// the callback may consume.
///
/// # Errors
/// As [`fuse_stack`].
pub fn fuse_stack_with<B: Backend, F>(
    model: &impl FusionModel<B>,
    frames: &[Tensor<B, 3>],
    cfg: TileConfig,
    device: &B::Device,
    on_step: F,
) -> Result<Tensor<B, 3>, InferError>
where
    F: FnMut(usize, Tensor<B, 3>),
{
    fuse_stream_with(model, frames.iter().cloned(), cfg, device, on_step)
}

/// The streaming form of [`fuse_stack_with`]: frames come from an iterator.
///
/// For a pairwise model each frame is taken only when it is folded in, so a caller
/// that produces the tensors lazily (for example by converting one planar frame at
/// a time) never holds more than the running composite plus the one frame in
/// flight. A batch model needs the whole stack, so its frames are collected first
/// (each is shape-checked as it arrives).
///
/// Frame 0 starts the composite, every later frame is checked against its shape as
/// it arrives (a mismatch is reported at that point, after the earlier pairwise
/// folds have run), and `on_step` is called as documented on [`fuse_stack_with`].
///
/// # Errors
/// [`InferError::EmptyStack`] when the iterator yields nothing and
/// [`InferError::ShapeMismatch`] for the first frame whose shape differs from
/// frame 0's.
pub fn fuse_stream_with<B, I, F>(
    model: &impl FusionModel<B>,
    frames: I,
    cfg: TileConfig,
    device: &B::Device,
    mut on_step: F,
) -> Result<Tensor<B, 3>, InferError>
where
    B: Backend,
    I: IntoIterator<Item = Tensor<B, 3>>,
    F: FnMut(usize, Tensor<B, 3>),
{
    let mut frames = frames.into_iter();
    let first = frames.next().ok_or(InferError::EmptyStack)?;
    let shape0 = first.dims();
    let [c, h, w] = shape0;

    if model.strategy() == FusionStrategy::Batch {
        let mut all = vec![first];
        for (offset, frame) in frames.enumerate() {
            check_shape(offset + 1, shape0, frame.dims())?;
            all.push(frame);
        }
        return fuse_batch_tiled(model, &all, cfg, device, on_step);
    }

    // Start with frame 0 as the initial composite and zero confidence.
    let mut result = first.unsqueeze_dim::<4>(0); // [1, 3, H, W]
    let mut conf = Tensor::<B, 4>::zeros([1, 1, h, w], device); // [1, 1, H, W]
    on_step(0, result.clone().reshape([c, h, w]));

    for (offset, frame) in frames.enumerate() {
        let index = offset + 1;
        check_shape(index, shape0, frame.dims())?;
        let source = frame.unsqueeze_dim::<4>(0); // [1, 3, H, W]
        let (new_result, new_conf) = merge_tiled(model, result, conf, source, cfg, device);
        result = new_result;
        conf = new_conf;
        on_step(index, result.clone().reshape([c, h, w]));
    }

    // Squeeze batch dim: [1, 3, H, W] -> [3, H, W]
    Ok(result.reshape([c, h, w]))
}

/// Tiled whole-stack fusion for [`FusionStrategy::Batch`] models.
///
/// Unlike the pairwise fold there is no per-frame state: the model consumes the
/// entire stack at once. Tiling therefore crops EVERY frame to the same region and
/// stacks just that region into a `[1, S, 3, tile_h, tile_w]` tensor per tile, so
/// at most one tile's worth of the stack is resident at a time (memory scales with
/// `S · tile_h · tile_w`, not `S · H · W`). The tile layout, overlap clamp and
/// raised-cosine blend are exactly those of the pairwise path, and the model's
/// per-pixel normalisation makes tiled output agree with the whole-image run away
/// from the blend bands. An image no larger than one tile is fused in a single call.
///
/// ## Progress
///
/// `on_step(tile_row_index, running_composite)` fires once per completed ROW of
/// tiles (all tiles sharing a vertical band), not once per tile: a full-image
/// blended preview after every tile would cost an extra O(`tiles · H · W`). For an
/// image that fits in one tile it fires exactly once, at completion, with row `0`.
///
/// # Errors
/// [`InferError::EmptyStack`] for no frames and [`InferError::ShapeMismatch`] when
/// a frame's shape differs from frame 0's.
pub fn fuse_batch_tiled<B, F>(
    model: &impl FusionModel<B>,
    frames: &[Tensor<B, 3>],
    cfg: TileConfig,
    device: &B::Device,
    mut on_step: F,
) -> Result<Tensor<B, 3>, InferError>
where
    B: Backend,
    F: FnMut(usize, Tensor<B, 3>),
{
    let first = frames.first().ok_or(InferError::EmptyStack)?;
    let shape0 = first.dims();
    for (index, frame) in frames.iter().enumerate().skip(1) {
        check_shape(index, shape0, frame.dims())?;
    }
    let [c, h, w] = shape0;
    let s = frames.len();

    // Fast path: the whole image fits in a single tile.
    if h <= cfg.tile && w <= cfg.tile {
        let stack = Tensor::cat(
            frames
                .iter()
                .map(|f| f.clone().unsqueeze_dim::<4>(0).unsqueeze_dim::<5>(0))
                .collect(),
            1,
        );
        let out = model.fuse_batch(stack).reshape([c, h, w]);
        on_step(0, out.clone());
        return Ok(out);
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

            // Crop just this tile's region from each frame, THEN stack — never
            // materialises a full-resolution [1, S, 3, H, W] tensor.
            let mut stack_tile = Vec::with_capacity(s);
            for frame in frames {
                stack_tile.push(
                    frame
                        .clone()
                        .slice([0..3, r..r + th, c_off..c_off + tw])
                        .unsqueeze_dim::<4>(0)
                        .unsqueeze_dim::<5>(0),
                );
            }
            let merged_tile = model.fuse_batch(Tensor::cat(stack_tile, 1));

            let win3 = win.clone().expand([1, 3, th, tw]);
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
                cur_weight.add(win),
            );
        }

        // One progress callback per completed tile ROW: normalise the
        // accumulator so far into a preview composite.
        let preview = merged_acc
            .clone()
            .div(weight_acc.clone().add_scalar(1e-6_f32));
        on_step(row_idx, preview.reshape([c, h, w]));
    }

    Ok(merged_acc.div(weight_acc).reshape([c, h, w]))
}

/// Recorder used to READ every `.mpk` checkpoint: named-msgpack with f32 settings.
///
/// A record is self-describing (every tensor stores its own dtype) and the
/// settings only choose the element type tensors are converted to on load. Reading
/// with the full-precision settings therefore loads both kinds of file this crate
/// writes without losing anything:
///
/// * `CompactRecorder` files (half precision: the `epoch` / `final` exports) are
///   widened f16 -> f32 exactly;
/// * `checkpoint::RollingRecorder` files (f32: the rolling `latest` model and the
///   optimiser state) are read bit-for-bit.
///
/// Reading a full-precision file with `CompactRecorder` would silently round it to
/// f16, which is why no reader uses it.
pub type WeightsReader = burn::record::NamedMpkFileRecorder<burn::record::FullPrecisionSettings>;

/// Load a module's record from `path` into `module` (which supplies the shape).
fn load_module<B: Backend, M: Module<B>>(
    module: M,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<M, InferError> {
    let record = WeightsReader::new()
        .load(path.to_path_buf(), device)
        .map_err(|source| InferError::Record {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(module.load_record(record))
}

/// Load trained pairwise-fusion weights from a `.mpk` checkpoint (half- or
/// full-precision; see [`WeightsReader`]).
///
/// `path` must be the complete file name including `.mpk`. Burn's file recorders
/// replace the last extension of the path they are given, so a stem that itself
/// contains a dot (`focusmerge_latest.model`) would be truncated; a path that
/// already ends in `.mpk` is left untouched.
///
/// # Errors
/// [`InferError::Record`] when the file is missing, unreadable, or was written by
/// an incompatible architecture.
pub fn load_weights<B: Backend>(
    config: &FocusMergeNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<FocusMergeNet<B>, InferError> {
    load_module(config.init(device), path, device)
}

/// Load trained batch-fusion weights ([`BatchMergeNet`]); see [`load_weights`].
///
/// # Errors
/// As [`load_weights`].
pub fn load_weights_batch<B: Backend>(
    config: &BatchMergeNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<BatchMergeNet<B>, InferError> {
    load_module(config.init(device), path, device)
}

/// Load trained whole-stack alignment weights ([`BatchAlignNet`]); see
/// [`load_weights`].
///
/// # Errors
/// As [`load_weights`].
pub fn load_weights_align<B: Backend>(
    config: &BatchAlignNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<BatchAlignNet<B>, InferError> {
    load_module(config.init(device), path, device)
}

/// Load trained pairwise-alignment weights ([`FusionAlignNet`]); see
/// [`load_weights`].
///
/// # Errors
/// As [`load_weights`].
pub fn load_weights_fusion_align<B: Backend>(
    config: &FusionAlignNetConfig,
    path: &std::path::Path,
    device: &B::Device,
) -> Result<FusionAlignNet<B>, InferError> {
    load_module(config.init(device), path, device)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// `Err(ShapeMismatch)` when `got` differs from frame 0's `expected` shape.
fn check_shape(index: usize, expected: [usize; 3], got: [usize; 3]) -> Result<(), InferError> {
    if got == expected {
        Ok(())
    } else {
        Err(InferError::ShapeMismatch {
            index,
            expected: expected.to_vec(),
            got: got.to_vec(),
        })
    }
}

/// Run a single pairwise merge step with tiling.
///
/// Returns `(merged [1,3,H,W], conf [1,1,H,W])`.
fn merge_tiled<B: Backend>(
    model: &impl FusionModel<B>,
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
            let win1 = win; // already [1, 1, th, tw]

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
#[expect(
    clippy::cast_precision_loss,
    reason = "tile sizes are a few hundred pixels, exact in f32"
)]
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
