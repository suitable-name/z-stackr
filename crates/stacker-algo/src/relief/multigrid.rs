//! The `Relief` multigrid engine: a confidence-weighted depth map by diffusion.
//!
//! The frame index acts as a depth coordinate (the stack is a focus sweep, so the
//! frames are ordered by depth). Where the focus measure is confident, the depth of
//! a pixel is the index of its sharpest frame and the pixel is *pinned* to it; where
//! it is not (flat, blurred or noisy areas), the depth is filled in by diffusion from
//! the pinned pixels. [`MultigridSolver`] solves that on a hierarchy of coarser
//! grids, [`pyramid_smooth`] smooths the result, and [`composite_depth`] turns the
//! continuous depth back into pixels by interpolating the two nearest frames.
//!
//! # Entry points
//!
//! * [`multigrid_depth`]: the whole-image depth from the per-pixel winning frame
//!   index and a confidence weight. The planes it reads are small (6 bytes per
//!   pixel), so a tiled run computes them per tile ([`update_argmax`],
//!   [`confidence_weights`]), solves **once** on the whole image and composites per
//!   tile with [`composite_depth`] on the [`crop_plane`] of the depth: the depth
//!   has no seams and does not depend on the tile size.
//! * [`composite_depth`]: the fused pixels for a stack (or a tile of it) and the
//!   matching depth.
//!
//! # Memory
//!
//! [`multigrid_depth`] needs about 28 bytes per pixel at its peak, on top of the
//! frames: the two input planes (2 + 4), the four solver planes of the padded
//! finest level (16) and the coarser levels (about a third more), the smoothing
//! pyramid being built only after the coarser solver levels are dropped. The result
//! is 4 bytes per pixel.
//!
//! # The algorithm, step by step (so that a GPU twin can reproduce it)
//!
//! All arithmetic is `f32`; every operation below is a pure function of the planes
//! it reads and every output sample is computed on its own, so the result does not
//! depend on the thread count. A sum is evaluated left to right in the order named.
//!
//! * **Padding.** The field is padded to the odd size `(2 * (w / 2) + 1) × (2 * (h /
//!   2) + 1)` (one column and/or row is added for an even side) by replicating the
//!   last column and row. Level 0 is the padded field.
//! * **Levels.** Level `l + 1` is `ceil(w_l / 2) × ceil(h_l / 2)`; levels are added
//!   while both sides of the last level are above 2. Every level has four planes:
//!   `a` the solution, `b` the weight in `0..=1`, `c` the target, `d` a scratch
//!   plane.
//! * **Initial state of level 0.** A sample whose weight is positive gets
//!   `c = target`, `b = 1`, `a = target`; every other sample has `a = b = c = 0`.
//! * **Cycle.** `cycle(l)` does nothing on the last level. On every other level it
//!   does three times: `restrict(l -> l + 1)`, `cycle(l + 1)`, `prolong(l + 1 -> l)`
//!   and then three `relax(l)` sweeps. `solve` is `cycle(0)`.
//! * **Restrict.** A coarse sample `(cx, cy)` covers the fine block with origin
//!   `(2 cx, 2 cy)`, clipped to the fine level (so 1, 2 or 4 samples). With `n`
//!   samples taken in the order (row 0: left, right; row 1: left, right):
//!   `a = sum(a) / n`, `b = sum(b) / n`, and `c = sum(b * c) / sum(b)` if `sum(b) > 0`
//!   else `0`.
//! * **Prolong.** Every fine sample `(x, y)` belongs to the coarse sample `(x / 2,
//!   y / 2)`. If that coarse sample has `b >= 1`, the fine sample is left alone.
//!   Otherwise `value = bilinear(coarse a)` and `a = b * c + (1 - b) * value` with
//!   the fine `b` and `c`. The bilinear value of an even/even fine sample is the
//!   coarse `a`; of an odd `x` the mean `(a + a_right) * 0.5`; of an odd `y` the
//!   mean `(a + a_down) * 0.5`; of an odd/odd sample `((a + a_right) + (a_down +
//!   a_down_right)) * 0.25`, where the neighbour indices are clamped to the coarse
//!   level.
//! * **Relax** (one Jacobi sweep: reads `a`, writes `d`, then `d` becomes `a`). For
//!   every sample, `nv = c` if `b > 0`, else the mean of `a` over the in-bounds 3×3
//!   neighbourhood (rows top to bottom, columns left to right, the sample itself
//!   included); `d = b * c + (1 - b) * nv`.
//! * **Smoothing.** [`pyramid_smooth`] of the padded finest `a` with
//!   `max_scale = smoothing_radius`, then the crop to `w × h`.
//!
//! Two deliberate deviations from the GitHub z-stackr algorithm, both bug fixes: the
//! restricted target of a 1×1 corner block is the target itself (GitHub halved it),
//! and the prolongation writes all four samples of a block (GitHub skipped the
//! lower-left sample, which then kept a stale value until the next relaxation).

#![allow(
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the solver spells its weighted means as plain multiply-add in a fixed order that a GPU twin reproduces; frame indices and counts are far below 2^24"
)]

use rayon::prelude::*;
use stacker_core::image::PlanarImage;

use super::pyramid::{pyramid_smooth, pyramid_smooth_owned};

/// Planes with fewer samples than this are processed on the calling thread (the
/// coarse levels are visited millions of times and are tiny).
const PARALLEL_MIN_SAMPLES: usize = 1 << 14;

/// How often a level restricts to the next one, solves it and prolongs back.
pub const VISITS_PER_LEVEL: usize = 3;

/// How many relaxation sweeps follow each prolongation.
pub const SWEEPS_PER_VISIT: usize = 3;

/// Run `row_fn(y, row)` for every `width`-wide row of `data`, in parallel when the
/// plane is large.
fn for_each_row<F>(data: &mut [f32], width: usize, row_fn: F)
where
    F: Fn(usize, &mut [f32]) + Sync + Send,
{
    if data.len() >= PARALLEL_MIN_SAMPLES {
        data.par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| row_fn(y, row));
    } else {
        data.chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| row_fn(y, row));
    }
}

/// Run `row_fn(y, row_a, row_b, row_c)` for every `width`-wide row of three
/// equally sized planes, in parallel when the planes are large.
fn for_each_row3<F>(
    first: &mut [f32],
    second: &mut [f32],
    third: &mut [f32],
    width: usize,
    row_fn: F,
) where
    F: Fn(usize, &mut [f32], &mut [f32], &mut [f32]) + Sync + Send,
{
    if first.len() >= PARALLEL_MIN_SAMPLES {
        first
            .par_chunks_mut(width)
            .zip(second.par_chunks_mut(width))
            .zip(third.par_chunks_mut(width))
            .enumerate()
            .for_each(|(y, ((row_a, row_b), row_c))| row_fn(y, row_a, row_b, row_c));
    } else {
        first
            .chunks_mut(width)
            .zip(second.chunks_mut(width))
            .zip(third.chunks_mut(width))
            .enumerate()
            .for_each(|(y, ((row_a, row_b), row_c))| row_fn(y, row_a, row_b, row_c));
    }
}

/// One row of a relaxation sweep: the new value of every sample of row `y`.
fn relax_row(a: &[f32], b: &[f32], c: &[f32], (w, h): (usize, usize), y: usize, row: &mut [f32]) {
    let y_min = y.saturating_sub(1);
    let y_max = (y + 1).min(h - 1);
    for (x, out) in row.iter_mut().enumerate() {
        let i = y * w + x;
        let neighbourhood = if b[i] > 0.0 {
            c[i]
        } else {
            let x_min = x.saturating_sub(1);
            let x_max = (x + 1).min(w - 1);
            let mut sum = 0.0_f32;
            let mut count = 0.0_f32;
            for ny in y_min..=y_max {
                for nx in x_min..=x_max {
                    sum += a[ny * w + nx];
                    count += 1.0;
                }
            }
            sum / count
        };
        *out = b[i] * c[i] + (1.0 - b[i]) * neighbourhood;
    }
}

/// A single level of the multigrid hierarchy: four planes of `width × height`.
#[derive(Debug, Clone)]
pub struct MultigridLayer {
    /// Width in samples.
    pub width: usize,
    /// Height in samples.
    pub height: usize,
    /// The solution.
    pub a: Vec<f32>,
    /// The weight (confidence) in `0.0..=1.0`; a positive weight pins the sample to
    /// its target.
    pub b: Vec<f32>,
    /// The target value.
    pub c: Vec<f32>,
    /// A scratch plane for [`Self::relax`] (its contents are not meaningful between
    /// calls).
    pub d: Vec<f32>,
}

impl MultigridLayer {
    /// A level of `width × height` samples, all planes zero.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let len = width * height;
        Self {
            width,
            height,
            a: vec![0.0; len],
            b: vec![0.0; len],
            c: vec![0.0; len],
            d: vec![0.0; len],
        }
    }

    /// Restrict `self` (the fine level) onto `coarse`: the mean of `a` and `b` over
    /// each 2×2 block and the `b`-weighted mean of `c` (see the module
    /// documentation for the exact order).
    ///
    /// # Panics
    /// Panics when `coarse` is not `ceil(width / 2) × ceil(height / 2)`.
    pub fn restrict_to(&self, coarse: &mut Self) {
        let (fw, fh) = (self.width, self.height);
        let cw = coarse.width;
        assert!(
            cw == fw.div_ceil(2) && coarse.height == fh.div_ceil(2),
            "the coarse level must be half the fine level, rounded up"
        );
        for_each_row3(
            &mut coarse.a,
            &mut coarse.b,
            &mut coarse.c,
            cw,
            |cy, row_a, row_b, row_c| {
                let fy0 = cy * 2;
                let rows = if fy0 + 1 < fh { 2 } else { 1 };
                for cx in 0..cw {
                    let fx0 = cx * 2;
                    let cols = if fx0 + 1 < fw { 2 } else { 1 };
                    let mut sum_a = 0.0_f32;
                    let mut sum_b = 0.0_f32;
                    let mut sum_bc = 0.0_f32;
                    let mut count = 0.0_f32;
                    for dy in 0..rows {
                        for dx in 0..cols {
                            let i = (fy0 + dy) * fw + fx0 + dx;
                            sum_a += self.a[i];
                            sum_b += self.b[i];
                            sum_bc += self.b[i] * self.c[i];
                            count += 1.0;
                        }
                    }
                    row_a[cx] = sum_a / count;
                    row_b[cx] = sum_b / count;
                    row_c[cx] = if sum_b > 0.0 { sum_bc / sum_b } else { 0.0 };
                }
            },
        );
    }

    /// Prolong `self` (the coarse level) onto `fine`: every fine sample whose coarse
    /// sample is not fully pinned (`b < 1`) takes the bilinear coarse value, blended
    /// with its own target by its own weight (see the module documentation).
    ///
    /// # Panics
    /// Panics when `self` is not `ceil(width / 2) × ceil(height / 2)` of `fine`.
    pub fn prolong_from(&self, fine: &mut Self) {
        let (cw, ch) = (self.width, self.height);
        let fw = fine.width;
        assert!(
            cw == fw.div_ceil(2) && ch == fine.height.div_ceil(2),
            "the coarse level must be half the fine level, rounded up"
        );
        let (fine_b, fine_c) = (&fine.b, &fine.c);
        for_each_row(&mut fine.a, fw, |y, row| {
            let cy = y / 2;
            let cy_next = (cy + 1).min(ch - 1);
            for (x, out) in row.iter_mut().enumerate() {
                let cx = x / 2;
                if self.b[cy * cw + cx] >= 1.0 {
                    continue;
                }
                let cx_next = (cx + 1).min(cw - 1);
                let here = self.a[cy * cw + cx];
                let value = match (x % 2, y % 2) {
                    (0, 0) => here,
                    (_, 0) => f32::midpoint(here, self.a[cy * cw + cx_next]),
                    (0, _) => f32::midpoint(here, self.a[cy_next * cw + cx]),
                    _ => {
                        ((here + self.a[cy * cw + cx_next])
                            + (self.a[cy_next * cw + cx] + self.a[cy_next * cw + cx_next]))
                            * 0.25
                    }
                };
                let i = y * fw + x;
                *out = fine_b[i] * fine_c[i] + (1.0 - fine_b[i]) * value;
            }
        });
    }

    /// One Jacobi relaxation sweep: every pinned sample (`b > 0`) is pulled to its
    /// target, every other sample becomes the mean of its in-bounds 3×3
    /// neighbourhood. Reads `a`, writes `d`, then swaps the two planes.
    pub fn relax(&mut self) {
        let dims = (self.width, self.height);
        {
            let (a, b, c) = (&self.a, &self.b, &self.c);
            for_each_row(&mut self.d, dims.0, |y, row| {
                relax_row(a, b, c, dims, y, row);
            });
        }
        std::mem::swap(&mut self.a, &mut self.d);
    }
}

/// The multigrid depth solver for one image.
#[derive(Debug, Clone)]
pub struct MultigridSolver {
    /// The hierarchy, finest (padded) level first; empty for an empty image.
    pub layers: Vec<MultigridLayer>,
    /// Width of the image in pixels (the unpadded size).
    pub width: usize,
    /// Height of the image in pixels (the unpadded size).
    pub height: usize,
}

impl MultigridSolver {
    /// A solver for the `width × height` field with the given `target` values and
    /// confidence `weight`s (a weight `> 0` pins the pixel to its target).
    ///
    /// Memory: the four planes of the padded finest level (16 bytes per pixel) plus
    /// the coarser levels (about a third more).
    ///
    /// # Panics
    /// Panics when a slice holds fewer than `width * height` samples.
    #[must_use]
    pub fn new(width: usize, height: usize, target: &[f32], weight: &[f32]) -> Self {
        assert!(
            target.len() >= width * height,
            "the target must hold width * height samples"
        );
        Self::build(width, height, weight, |i| target[i])
    }

    /// Like [`Self::new`], but the target of a pixel is the frame index `index[i]`
    /// (so no `f32` target plane is needed).
    ///
    /// # Panics
    /// Panics when a slice holds fewer than `width * height` samples.
    #[must_use]
    pub fn from_index(width: usize, height: usize, index: &[u16], weight: &[f32]) -> Self {
        assert!(
            index.len() >= width * height,
            "the index must hold width * height samples"
        );
        Self::build(width, height, weight, |i| f32::from(index[i]))
    }

    /// Pad, initialise the finest level and add the coarser ones.
    fn build<F>(width: usize, height: usize, weight: &[f32], target_at: F) -> Self
    where
        F: Fn(usize) -> f32 + Sync + Send,
    {
        assert!(
            weight.len() >= width * height,
            "the weight must hold width * height samples"
        );
        if width == 0 || height == 0 {
            return Self {
                layers: Vec::new(),
                width,
                height,
            };
        }
        let pad_w = (width / 2) * 2 + 1;
        let pad_h = (height / 2) * 2 + 1;

        let mut finest = MultigridLayer::new(pad_w, pad_h);
        for_each_row3(
            &mut finest.a,
            &mut finest.b,
            &mut finest.c,
            pad_w,
            |y, row_a, row_b, row_c| {
                let source_row = y.min(height - 1) * width;
                for x in 0..pad_w {
                    let source = source_row + x.min(width - 1);
                    if weight[source] > 0.0 {
                        let target = target_at(source);
                        row_a[x] = target;
                        row_b[x] = 1.0;
                        row_c[x] = target;
                    }
                }
            },
        );

        let mut layers = vec![finest];
        let (mut level_w, mut level_h) = (pad_w, pad_h);
        while level_w > 2 && level_h > 2 {
            level_w = level_w.div_ceil(2);
            level_h = level_h.div_ceil(2);
            layers.push(MultigridLayer::new(level_w, level_h));
        }
        Self {
            layers,
            width,
            height,
        }
    }

    /// Run the cycle once (`cycle(0)` of the module documentation). An image whose
    /// padded finest level already has a side of 2 or less (a single row or column)
    /// has no coarser level and is left as initialised.
    pub fn solve(&mut self) {
        if self.layers.len() > 1 {
            self.cycle(0);
        }
    }

    /// The recursive cycle on `level`; the coarsest level does nothing (its `a` is
    /// fully determined by the restriction).
    fn cycle(&mut self, level: usize) {
        if level + 1 >= self.layers.len() {
            return;
        }
        for _ in 0..VISITS_PER_LEVEL {
            {
                let (fine, coarse) = self.layers.split_at_mut(level + 1);
                fine[level].restrict_to(&mut coarse[0]);
            }
            self.cycle(level + 1);
            {
                let (fine, coarse) = self.layers.split_at_mut(level + 1);
                coarse[0].prolong_from(&mut fine[level]);
            }
            for _ in 0..SWEEPS_PER_VISIT {
                self.layers[level].relax();
            }
        }
    }

    /// The solved field cropped to `width × height` (no smoothing); empty for an
    /// empty image.
    #[must_use]
    pub fn get_solution(&self) -> Vec<f32> {
        self.layers.first().map_or_else(Vec::new, |finest| {
            crop(&finest.a, finest.width, self.width, self.height)
        })
    }

    /// The solved field after [`pyramid_smooth`] (with `max_scale`, the
    /// `relief_smoothing_radius`), cropped to `width × height`; empty for an empty
    /// image.
    #[must_use]
    pub fn get_smoothed_solution(&self, max_scale: usize) -> Vec<f32> {
        self.layers.first().map_or_else(Vec::new, |finest| {
            let smoothed = pyramid_smooth(&finest.a, finest.width, finest.height, max_scale);
            crop(&smoothed, finest.width, self.width, self.height)
        })
    }

    /// Like [`Self::get_smoothed_solution`], but consumes the solver and frees the
    /// coarse levels and the weight and target planes before smoothing, which lowers
    /// the peak memory.
    #[must_use]
    pub fn into_smoothed_solution(self, max_scale: usize) -> Vec<f32> {
        let (width, height) = (self.width, self.height);
        let mut layers = self.layers;
        layers.truncate(1);
        let Some(finest) = layers.pop() else {
            return Vec::new();
        };
        let MultigridLayer {
            width: padded_width,
            height: padded_height,
            a,
            ..
        } = finest;
        let smoothed = pyramid_smooth_owned(a, padded_width, padded_height, max_scale);
        crop_owned(smoothed, padded_width, width, height)
    }
}

/// The top-left `width × height` part of a `padded_width`-wide field.
fn crop(padded: &[f32], padded_width: usize, width: usize, height: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(width * height);
    for row in padded.chunks(padded_width).take(height) {
        out.extend_from_slice(&row[..width]);
    }
    out
}

/// [`crop`] in place.
fn crop_owned(mut padded: Vec<f32>, padded_width: usize, width: usize, height: usize) -> Vec<f32> {
    if padded_width != width {
        for y in 1..height {
            padded.copy_within(y * padded_width..y * padded_width + width, y * width);
        }
    }
    padded.truncate(width * height);
    padded
}

/// The whole-image depth map from the per-pixel winning frame `index` and a
/// confidence `weight`, tile-invariant by construction.
///
/// A pixel with a positive `weight` is pinned to the frame index `index[i]`; the
/// others are filled in by the multigrid diffusion ([`MultigridSolver`], three
/// cycles of three sweeps as described in the module documentation) and the result
/// is smoothed by [`pyramid_smooth`] with `max_scale = smoothing_radius`
/// (`relief_smoothing_radius`; a value of 1 or less skips the smoothing). The
/// values are fractional frame indices, in the range of the pinned indices (the
/// whole procedure averages) up to rounding; pass them to [`composite_depth`].
///
/// With no pinned pixel at all the depth is 0 everywhere. An empty image gives an
/// empty result. Memory: see the module documentation (about 28 bytes per pixel at
/// the peak). The result does not depend on the thread count.
///
/// # Panics
/// Panics when `index` or `weight` holds fewer than `width * height` samples.
#[must_use]
pub fn multigrid_depth(
    width: usize,
    height: usize,
    index: &[u16],
    weight: &[f32],
    smoothing_radius: usize,
) -> Vec<f32> {
    let mut solver = MultigridSolver::from_index(width, height, index, weight);
    solver.solve();
    solver.into_smoothed_solution(smoothing_radius)
}

/// Fold one frame's focus map into the running per-pixel maximum: where
/// `focus[i] > best[i]` (strictly, so the earlier frame wins a tie), `best[i]`
/// becomes `focus[i]` and `index[i]` becomes `frame`.
///
/// Start with `best` filled with `f32::NEG_INFINITY` and `index` zero, call this
/// for the frames in stack order, then pass `index` to [`multigrid_depth`] and
/// `best` to [`confidence_weights`]. A pixel whose focus values are all NaN keeps
/// index 0. The update works on any sub-range of pixels (a tile row, a tile).
///
/// # Panics
/// Panics when the three slices differ in length.
pub fn update_argmax(best: &mut [f32], index: &mut [u16], frame: u16, focus: &[f32]) {
    assert!(
        best.len() == index.len() && best.len() == focus.len(),
        "the planes must have one sample per pixel"
    );
    best.par_iter_mut()
        .zip(index.par_iter_mut())
        .zip(focus.par_iter())
        .for_each(|((best, index), &focus)| {
            if focus > *best {
                *best = focus;
                *index = frame;
            }
        });
}

/// The confidence weights for [`multigrid_depth`]: 1 where the best focus value
/// reaches the contrast `threshold` (`max_focus >= threshold`, the same test as
/// `mask_above`), else 0.
#[must_use]
pub fn confidence_weights(max_focus: &[f32], threshold: f32) -> Vec<f32> {
    max_focus
        .par_iter()
        .map(|&value| if value >= threshold { 1.0 } else { 0.0 })
        .collect()
}

/// Copy the `width × height` window at `(x0, y0)` out of a `src_width`-wide plane
/// (to take the depth of a tile out of the whole-image depth).
///
/// # Panics
/// Panics when the window reaches outside the plane.
#[must_use]
pub fn crop_plane(
    plane: &[f32],
    src_width: usize,
    (x0, y0): (usize, usize),
    (width, height): (usize, usize),
) -> Vec<f32> {
    let mut out = Vec::with_capacity(width * height);
    for y in y0..y0 + height {
        let start = y * src_width + x0;
        out.extend_from_slice(&plane[start..start + width]);
    }
    out
}

/// The fused image for a continuous `depth`.
///
/// Every pixel interpolates, on all three planes, between the frames
/// `floor(depth)` and `ceil(depth)` of `frames` with the weight
/// `depth - floor(depth)` (so an integral depth takes that frame as it is).
///
/// `depth` is clamped to `0..=frames.len() - 1` first (NaN becomes 0), and holds one
/// value per pixel of the frames. Frames of a tile and the [`crop_plane`] of the
/// whole-image depth give the tile of the whole-image result: the operation is
/// per pixel. No guided filter follows; the smoothing is in [`multigrid_depth`].
/// Returns an empty image for no frames.
///
/// # Panics
/// Panics when the frames differ in size or `depth` does not hold one sample per
/// pixel.
#[must_use]
pub fn composite_depth(frames: &[PlanarImage<f32>], depth: &[f32]) -> PlanarImage<f32> {
    let Some(first) = frames.first() else {
        return PlanarImage::new(0, 0);
    };
    let (width, height) = (first.width, first.height);
    let len = width * height;
    assert!(
        frames
            .iter()
            .all(|f| f.width == width && f.height == height),
        "all frames must have the same size"
    );
    assert_eq!(depth.len(), len, "one depth value per pixel");
    let last = (frames.len() - 1) as f32;

    let mut luma = vec![0.0_f32; len];
    let mut chroma_a = vec![0.0_f32; len];
    let mut chroma_b = vec![0.0_f32; len];
    luma.par_iter_mut()
        .zip(chroma_a.par_iter_mut())
        .zip(chroma_b.par_iter_mut())
        .zip(depth.par_iter())
        .enumerate()
        .for_each(|(i, (((out_l, out_a), out_b), &depth))| {
            let position = if depth.is_nan() {
                0.0
            } else {
                depth.clamp(0.0, last)
            };
            let lower = position.floor();
            let frac = position - lower;
            let low = &frames[lower as usize];
            let high = &frames[position.ceil() as usize];
            *out_l = low.luma[i] * (1.0 - frac) + high.luma[i] * frac;
            *out_a = low.chroma_a[i] * (1.0 - frac) + high.chroma_a[i] * frac;
            *out_b = low.chroma_b[i] * (1.0 - frac) + high.chroma_b[i] * frac;
        });

    PlanarImage::from_planes(width, height, luma, chroma_a, chroma_b)
        .expect("every plane has one sample per pixel")
}
