//! `Apex` fusion: combine one Laplacian pyramid per frame into a single pyramid.
//!
//! [`build_and_fuse_pyramids`] builds the pyramids in parallel and fuses them;
//! [`fuse_pyramids`] is the selection rule itself (strongest coefficient per
//! pixel and band, averaged base). [`ApexFold`] is the same fusion streamed one
//! frame at a time, bit-identical and with memory independent of the frame
//! count ([`apex_fold_bytes_per_pixel`]). [`apron_for_levels`] and
//! [`pyramid_grid`] give the tile apron and alignment a tiled run needs.

use crate::{
    apex::pyramid::{LaplacianPyramid, reconstruct_from_bands},
    strata::FrameSource,
};
use rayon::prelude::*;
use stacker_core::image::PlanarImage;

/// Fewest pyramid levels `Apex` fusion accepts: one detail band plus the
/// averaged base.  [`build_and_fuse_pyramids`] clamps smaller requests to it.
pub const MIN_PYRAMID_LEVELS: usize = 2;

/// Deepest pyramid the apron functions distinguish (the config allows 12).
const MAX_APRON_LEVELS: usize = 32;

/// The number of source pixels around a pixel that its `Apex` output depends
/// on, for a pyramid of `levels` levels.
///
/// One reduce step reads radius 2 at its own level, so level `k` of the
/// Gaussian pyramid depends on the source within `2·(2^k − 1)` pixels; the
/// reconstruction adds one expand per level (radius 2 at level `k`, i.e.
/// `2·2^k` source pixels). Both chains over `L` levels give
/// `4·(2^(L−1) − 1)`. The 3×3 neighbourhood energy of the grit suppression adds
/// one more pixel at level 0. Levels below [`MIN_PYRAMID_LEVELS`] are raised to
/// it, as [`build_and_fuse_pyramids`] does.
///
/// | levels | radius |
/// |---|---|
/// | 6 | 125 |
/// | 7 | 253 |
/// | 8 | 509 |
#[must_use]
pub const fn pyramid_dependence_radius(levels: usize) -> usize {
    let levels = if levels < MIN_PYRAMID_LEVELS {
        MIN_PYRAMID_LEVELS
    } else if levels > MAX_APRON_LEVELS {
        MAX_APRON_LEVELS
    } else {
        levels
    };
    4 * ((1_usize << (levels - 1)) - 1) + 1
}

/// The apron (pixels on every side of a tile) a tiled `Apex` run needs.
///
/// With this apron the tile interior equals the whole-image result: it is the
/// dependence radius of [`pyramid_dependence_radius`] rounded **up to a multiple
/// of `2^(L−1)`**. The multiple keeps the padded tile's origin on the global decimation grid
/// when the tile size is itself a multiple of `2^(L−1)` (the pyramid halves
/// from the tile's own top-left corner, so an origin off the grid would shift
/// every coarse level against the whole-image pyramid).
///
/// | levels | apron |
/// |---|---|
/// | 6 | 128 |
/// | 7 | 256 |
/// | 8 | 512 |
#[must_use]
pub const fn apron_for_levels(levels: usize) -> usize {
    let grid = pyramid_grid(levels);
    pyramid_dependence_radius(levels).div_ceil(grid) * grid
}

/// `2^(L−1)`: the spacing of the coarsest pyramid level in source pixels, and
/// the multiple that tile sizes and aprons must have for tiles to share the
/// whole image's decimation grid.
#[must_use]
pub const fn pyramid_grid(levels: usize) -> usize {
    let levels = if levels < MIN_PYRAMID_LEVELS {
        MIN_PYRAMID_LEVELS
    } else if levels > MAX_APRON_LEVELS {
        MAX_APRON_LEVELS
    } else {
        levels
    };
    1_usize << (levels - 1)
}

// ── Private blend helpers ────────────────────────────────────────────────────

/// Average all source pixels at `level_idx` into the output buffers.
#[expect(
    clippy::cast_precision_loss,
    reason = "the number of source frames is far below 2^24"
)]
fn blend_average(
    pyramids: &[LaplacianPyramid],
    level_idx: usize,
    out_luma: &mut [f32],
    out_chroma_a: &mut [f32],
    out_chroma_b: &mut [f32],
) {
    let n = pyramids.len() as f32;
    out_luma
        .par_iter_mut()
        .zip(out_chroma_a.par_iter_mut())
        .zip(out_chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((ol, oa), ob))| {
            let mut sl = 0.0_f32;
            let mut sa = 0.0_f32;
            let mut sb = 0.0_f32;
            for p in pyramids {
                let lvl = &p.levels[level_idx];
                sl += lvl.luma[i];
                sa += lvl.chroma_a[i];
                sb += lvl.chroma_b[i];
            }
            *ol = sl / n;
            *oa = sa / n;
            *ob = sb / n;
        });
}

/// Compute the single-pixel selection energy for a given source pyramid level
/// at pixel index `i`.
///
/// - `use_color = false`: energy = luma² only.
/// - `use_color = true`:  energy = luma² + `chroma_a²` + `chroma_b²`.
#[inline]
#[expect(
    clippy::suboptimal_flops,
    reason = "plain multiply-add keeps the selection energy identical on targets with and without FMA, so ties break the same way"
)]
fn pixel_energy(lvl: &stacker_core::image::PlanarImage<f32>, i: usize, use_color: bool) -> f32 {
    let luma = lvl.luma[i];
    let energy = luma * luma;
    if use_color {
        let ca = lvl.chroma_a[i];
        let cb = lvl.chroma_b[i];
        energy + ca * ca + cb * cb
    } else {
        energy
    }
}

/// Per-pixel energy selection for coarser Laplacian levels.
///
/// When `use_color` is `false` the metric is luma² only.  When `true` the
/// metric is luma² + `chroma_a²` + `chroma_b²`.
///
/// Regardless of the metric, all three channels are always copied from the
/// *same* winning source so they remain coherent and free of false colours.
fn blend_max_contrast(
    pyramids: &[LaplacianPyramid],
    level_idx: usize,
    use_color: bool,
    out_luma: &mut [f32],
    out_chroma_a: &mut [f32],
    out_chroma_b: &mut [f32],
) {
    out_luma
        .par_iter_mut()
        .zip(out_chroma_a.par_iter_mut())
        .zip(out_chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((ol, oa), ob))| {
            // Single pass: evaluate each source's energy exactly once and keep
            // the argmax. Strict `>`: on a tie the FIRST source in input order
            // wins, the same rule as Relief and the auto-cull (see `fuse_pyramids`).
            let mut winner = 0usize;
            let mut best_e = f32::NEG_INFINITY;
            for (k, p) in pyramids.iter().enumerate() {
                let e = pixel_energy(&p.levels[level_idx], i, use_color);
                if e > best_e {
                    best_e = e;
                    winner = k;
                }
            }
            let lvl = &pyramids[winner].levels[level_idx];
            *ol = lvl.luma[i];
            *oa = lvl.chroma_a[i];
            *ob = lvl.chroma_b[i];
        });
}

/// 3×3 neighbourhood energy selection for the finest Laplacian level.
///
/// Out-of-bounds neighbours are skipped (clamped boundary).
///
/// When `use_color` is `false` each neighbour contributes luma² to the
/// accumulated energy.  When `true` each neighbour contributes
/// luma² + `chroma_a²` + `chroma_b²`.
///
/// All three channels are always copied coherently from the winning source.
#[expect(
    clippy::too_many_arguments,
    reason = "pyramid set, level, size, metric flag and the three output planes; bundling them would only add a struct used once"
)]
fn blend_max_contrast_neighborhood(
    pyramids: &[LaplacianPyramid],
    level_idx: usize,
    width: usize,
    height: usize,
    use_color: bool,
    out_luma: &mut [f32],
    out_chroma_a: &mut [f32],
    out_chroma_b: &mut [f32],
) {
    let len = width * height;
    if len == 0 {
        return;
    }
    // Frame-outer, pixel-inner: each source's 3×3 energy plane is computed once
    // (with the same nine-term order as ever), folded into a running best and
    // winner, and dropped. Strict `>`: on a tie the FIRST source in input order
    // wins (see `fuse_pyramids`).
    let mut best = vec![f32::NEG_INFINITY; len];
    let mut winner = vec![0_u32; len];
    let mut pixel = vec![0.0_f32; len];
    let mut energy = vec![0.0_f32; len];
    for (k, p) in pyramids.iter().enumerate() {
        let k = u32::try_from(k).expect("fewer than 2^32 frames");
        grit_energy_plane(&p.levels[level_idx], use_color, &mut pixel, &mut energy);
        best.par_iter_mut()
            .zip(winner.par_iter_mut())
            .zip(energy.par_iter())
            .for_each(|((b, w), &e)| {
                if e > *b {
                    *b = e;
                    *w = k;
                }
            });
    }

    out_luma
        .par_chunks_mut(width)
        .zip(out_chroma_a.par_chunks_mut(width))
        .zip(out_chroma_b.par_chunks_mut(width))
        .enumerate()
        .for_each(|(row, ((rl, ra), rb))| {
            let base = row * width;
            for col in 0..width {
                let pi = base + col;
                let lvl = &pyramids[winner[pi] as usize].levels[level_idx];
                rl[col] = lvl.luma[pi];
                ra[col] = lvl.chroma_a[pi];
                rb[col] = lvl.chroma_b[pi];
            }
        });
}

/// The grit-suppression selection energy of one level: for every pixel the sum of
/// [`pixel_energy`] over its 3×3 neighbourhood (out-of-image neighbours skipped),
/// added row by row from the top-left neighbour to the bottom-right one starting
/// from `0.0`. This is the exact summation order of the per-pixel formulation, so
/// the sums (and therefore the exact ties) are bit-identical to it.
///
/// `pixel` is scratch for the per-pixel energies; both slices hold
/// `width * height` samples.
fn grit_energy_plane(lvl: &PlanarImage<f32>, use_color: bool, pixel: &mut [f32], out: &mut [f32]) {
    let (width, height) = (lvl.width, lvl.height);
    if width == 0 || height == 0 {
        return;
    }
    pixel
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, p)| *p = pixel_energy(lvl, i, use_color));
    let pixel: &[f32] = pixel;
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out_row)| {
            let first_row = row.saturating_sub(1);
            let last_row = (row + 1).min(height - 1);
            for (col, cell) in out_row.iter_mut().enumerate() {
                let first_col = col.saturating_sub(1);
                let last_col = (col + 1).min(width - 1);
                let mut e = 0.0_f32;
                for nr in first_row..=last_row {
                    let base = nr * width;
                    for nc in first_col..=last_col {
                        e += pixel[base + nc];
                    }
                }
                *cell = e;
            }
        });
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Fuse multiple Laplacian pyramids into a single one.
///
/// Selection semantics:
/// - **Base / low-frequency level** (last level): averaged across all sources.
///   Unaffected by `use_color` or `grit_suppression`.
/// - **Finest / level 0**: winner selection is controlled by `grit_suppression`:
///   - `grit_suppression = true` (default): the winner is the source with the
///     largest selection energy summed over the 3×3 neighbourhood of the pixel
///     (out-of-image neighbours are skipped).  This only changes *which source
///     wins*; the winner's own coefficients are copied unchanged, so nothing is
///     smoothed or filtered.  Its effect is that an isolated noisy coefficient
///     is less likely to win a pixel by itself.
///   - `grit_suppression = false`: the winner is chosen by the pixel's own
///     energy.
/// - **All other Laplacian levels**: the winner is chosen by the pixel's own
///   energy (the 3×3 sum is never used there).
///
/// The selection energy is controlled by `use_color`:
/// - `use_color = false` (default): energy = luma² only.  Chroma noise cannot
///   steer the winner.
/// - `use_color = true`: energy = luma² + `chroma_a²` + `chroma_b²`, i.e. the
///   chroma bands of the linear YCbCr pyramid add to the selection energy.
///   The channels are not blended independently, and the cost is a few extra
///   multiplies per pixel.
///
/// In every winning case all three channels (luma, `chroma_a`, `chroma_b`) are
/// taken from the *same* source so they remain coherent and false colours
/// are avoided.
///
/// **Tie rule:** when two or more sources have exactly the same selection energy
/// at a pixel, the *first* of them in `pyramids` order wins.  This is the same
/// rule as the `Relief` per-pixel selection and the auto-cull census (strict `>`
/// comparisons), so reordering the frames changes ties the same way in all
/// three.  Flat regions (every detail coefficient zero) therefore take their
/// detail coefficients and chroma from the first frame.
///
/// # Panics
/// Panics if `pyramids` is empty or the pyramids differ in level count or in
/// the size of any level.
#[must_use]
pub fn fuse_pyramids(
    pyramids: &[LaplacianPyramid],
    use_color: bool,
    grit_suppression: bool,
) -> LaplacianPyramid {
    assert!(
        !pyramids.is_empty(),
        "cannot fuse an empty array of pyramids"
    );

    let levels_count = pyramids[0].levels.len();
    for p in pyramids {
        assert_eq!(
            p.levels.len(),
            levels_count,
            "all pyramids must have the same number of levels"
        );
    }

    let mut fused_levels = Vec::with_capacity(levels_count);

    for level_idx in 0..levels_count {
        let width = pyramids[0].levels[level_idx].width;
        let height = pyramids[0].levels[level_idx].height;
        let len = width * height;

        for p in pyramids {
            assert_eq!(
                p.levels[level_idx].width, width,
                "pyramid levels must have matching widths"
            );
            assert_eq!(
                p.levels[level_idx].height, height,
                "pyramid levels must have matching heights"
            );
        }

        let mut fused_luma = vec![0.0_f32; len];
        let mut fused_chroma_a = vec![0.0_f32; len];
        let mut fused_chroma_b = vec![0.0_f32; len];

        if level_idx == levels_count - 1 {
            blend_average(
                pyramids,
                level_idx,
                &mut fused_luma,
                &mut fused_chroma_a,
                &mut fused_chroma_b,
            );
        } else if level_idx == 0 && grit_suppression {
            blend_max_contrast_neighborhood(
                pyramids,
                level_idx,
                width,
                height,
                use_color,
                &mut fused_luma,
                &mut fused_chroma_a,
                &mut fused_chroma_b,
            );
        } else {
            blend_max_contrast(
                pyramids,
                level_idx,
                use_color,
                &mut fused_luma,
                &mut fused_chroma_a,
                &mut fused_chroma_b,
            );
        }

        fused_levels.push(
            PlanarImage::from_planes(width, height, fused_luma, fused_chroma_a, fused_chroma_b)
                .expect("each fused plane holds width * height samples"),
        );
    }

    LaplacianPyramid {
        levels: fused_levels,
    }
}

/// A requested depth raised to [`MIN_PYRAMID_LEVELS`], with a warning when it was
/// raised.
fn clamp_requested_levels(max_levels: usize) -> usize {
    if max_levels < MIN_PYRAMID_LEVELS {
        tracing::warn!(
            requested = max_levels,
            used = MIN_PYRAMID_LEVELS,
            "pyramid_levels below {MIN_PYRAMID_LEVELS} would turn Apex into a plain mean; clamping"
        );
        MIN_PYRAMID_LEVELS
    } else {
        max_levels
    }
}

/// Build one `LaplacianPyramid` per image using rayon parallelism across
/// the image stack, then fuse them.
///
/// This is the primary entry point for parallel pyramid fusion.  Building
/// per-image pyramids is the heaviest part of the `Apex` pipeline; doing it
/// in parallel saturates all available cores without increasing peak memory
/// (all per-image pyramids must be live simultaneously during `fuse_pyramids`
/// regardless of whether they were built serially or in parallel).
///
/// `use_color` and `grit_suppression` are forwarded directly to
/// [`fuse_pyramids`]; see that function's documentation for their semantics.
///
/// A `max_levels` below [`MIN_PYRAMID_LEVELS`] is raised to it with a warning:
/// a one-level pyramid has only the base band, which is *averaged* across the
/// frames, so `Apex` would silently degrade to a plain mean of the stack.
///
/// # Panics
/// Panics if `images` is empty or the images differ in size.
#[must_use]
pub fn build_and_fuse_pyramids(
    images: &[PlanarImage<f32>],
    max_levels: usize,
    use_color: bool,
    grit_suppression: bool,
) -> LaplacianPyramid {
    assert!(!images.is_empty(), "cannot fuse an empty image stack");
    let max_levels = clamp_requested_levels(max_levels);
    let pyramids: Vec<LaplacianPyramid> = images
        .par_iter()
        .map(|img| LaplacianPyramid::build(img, max_levels))
        .collect();
    fuse_pyramids(&pyramids, use_color, grit_suppression)
}

// ── Streaming fold ───────────────────────────────────────────────────────────

/// Samples per parallel work item of the fold's per-pixel selection.
const FOLD_CHUNK: usize = 4096;

/// `Σ ⌈per_pixel / 4^k⌉` over the first `levels` pyramid levels: the bytes per
/// source pixel of a pyramid whose finest level costs `per_pixel` (a safe upper
/// bound, each level rounded up).
const fn pyramid_bytes(per_pixel: usize, levels: usize) -> usize {
    let mut total = 0;
    let mut k = 0;
    while k < levels {
        total += if 2 * k >= 16 {
            1
        } else {
            per_pixel.div_ceil(1 << (2 * k))
        };
        k += 1;
    }
    total
}

/// Bytes per source pixel that an [`ApexFold`] with `levels` pyramid levels holds
/// at its peak, independent of the number of frames.
///
/// The sum of: the accumulator pyramid (12 B per level-0 pixel, a quarter of that
/// per further level), the `best_energy` planes of every level but the residual
/// (4 B per pixel, likewise), the grit scratch of the finest level (two `f32`
/// planes, 8 B), the incoming frame's pyramid (as the accumulator), and 24 B for
/// the transients of building that pyramid (the working copy of the frame and the
/// expanded plane set). For 8 levels this is about 84 B per pixel.
///
/// The batch path ([`build_and_fuse_pyramids`]) holds every frame's pyramid at
/// once: the pipeline's estimate for it is `28 N + 66` B per pixel, so the fold is
/// lower for any stack of two or more frames and stays flat as `N` grows.
/// `levels` is clamped like the fold clamps it (at least [`MIN_PYRAMID_LEVELS`]).
#[must_use]
pub const fn apex_fold_bytes_per_pixel(levels: usize) -> usize {
    let levels = if levels < MIN_PYRAMID_LEVELS {
        MIN_PYRAMID_LEVELS
    } else if levels > MAX_APRON_LEVELS {
        MAX_APRON_LEVELS
    } else {
        levels
    };
    let accumulator = pyramid_bytes(12, levels);
    let best = pyramid_bytes(4, levels - 1);
    let grit_scratch = 8;
    let frame_pyramid = pyramid_bytes(12, levels);
    let build_transients = 24;
    accumulator + best + grit_scratch + frame_pyramid + build_transients
}

/// The streaming form of [`build_and_fuse_pyramids`]: frames are folded in one at
/// a time and only one accumulator pyramid is ever held.
///
/// Per pyramid level the fold keeps the winner's coefficients and a
/// `best_energy` plane (at the finest level with grit suppression, the winner's
/// own 3×3 energy sum); a later frame replaces a pixel only when its energy is
/// **strictly** larger, so the first frame wins ties, exactly as in the batch
/// rule. The residual (last) level is a running **sum** divided by the frame
/// count in [`Self::finish`], added in frame order, so the result equals the batch
/// fusion bit for bit (not a running mean, which would round differently).
///
/// ```
/// use stacker_algo::apex::fuse::{build_and_fuse_pyramids, ApexFold};
/// use stacker_core::image::PlanarImage;
///
/// let frames: Vec<PlanarImage<f32>> = (0..3)
///     .map(|k| PlanarImage::with_luma(8, 8, vec![k as f32; 64]).unwrap())
///     .collect();
/// let mut fold = ApexFold::new(3, false, true);
/// for frame in &frames {
///     fold.fold(frame);
/// }
/// let streamed = fold.finish();
/// let batch = build_and_fuse_pyramids(&frames, 3, false, true).reconstruct();
/// assert_eq!(streamed.luma, batch.luma);
/// ```
pub struct ApexFold {
    max_levels: usize,
    use_color: bool,
    grit_suppression: bool,
    state: Option<FoldState>,
}

/// The accumulator: allocated when the first frame arrives (its pyramid fixes
/// the level sizes).
struct FoldState {
    /// Per level the winner's coefficients; the last level is the running sum.
    values: Vec<PlanarImage<f32>>,
    /// Per level but the last: the winner's selection energy.
    best: Vec<Vec<f32>>,
    /// Frames folded so far.
    count: usize,
    /// Scratch of the finest level's grit energy (per-pixel energies, 3×3 sums);
    /// empty unless grit suppression is on.
    pixel: Vec<f32>,
    energy: Vec<f32>,
}

impl ApexFold {
    /// A fold for pyramids of at most `max_levels` levels, with the selection
    /// rules of [`fuse_pyramids`] (`use_color`, `grit_suppression`). A
    /// `max_levels` below [`MIN_PYRAMID_LEVELS`] is raised to it with a warning,
    /// as [`build_and_fuse_pyramids`] does.
    #[must_use]
    pub fn new(max_levels: usize, use_color: bool, grit_suppression: bool) -> Self {
        Self {
            max_levels: clamp_requested_levels(max_levels),
            use_color,
            grit_suppression,
            state: None,
        }
    }

    /// The number of frames folded so far.
    #[must_use]
    pub fn frames_folded(&self) -> usize {
        self.state.as_ref().map_or(0, |s| s.count)
    }

    /// Fold one frame into the accumulator. Frames must arrive in stack order
    /// (ties go to the earlier one) and all have the size of the first.
    ///
    /// # Panics
    /// Panics when `frame` differs in size from the first frame folded.
    pub fn fold(&mut self, frame: &PlanarImage<f32>) {
        let pyramid = LaplacianPyramid::build(frame, self.max_levels);
        match self.state {
            Some(ref mut state) => state.fold_in(&pyramid, self.use_color, self.grit_suppression),
            None => {
                self.state = Some(FoldState::start(
                    pyramid,
                    self.use_color,
                    self.grit_suppression,
                ));
            }
        }
    }

    /// Fold every frame of `frames` in index order, calling `on_frame(done, self)`
    /// after each (`done` frames folded so far), so a caller can report progress
    /// or take a [`Self::preview`].
    ///
    /// # Panics
    /// Panics when the frames differ in size.
    pub fn fold_source(
        &mut self,
        frames: &(impl FrameSource + ?Sized),
        mut on_frame: impl FnMut(usize, &Self),
    ) {
        for index in 0..frames.len() {
            let frame = frames.frame(index);
            self.fold(&frame);
            drop(frame);
            on_frame(index + 1, self);
        }
    }

    /// The image the accumulator would give now, as if the frames folded so far
    /// were the whole stack. It reads the accumulator in place (only the small
    /// residual level is copied), so it costs one reconstruction, not a clone of
    /// the pyramid. An empty fold gives a 0 × 0 image.
    #[must_use]
    pub fn preview(&self) -> PlanarImage<f32> {
        let Some(state) = self.state.as_ref() else {
            return PlanarImage::new(0, 0);
        };
        let Some((residual, bands)) = state.values.split_last() else {
            return PlanarImage::new(0, 0);
        };
        reconstruct_from_bands(bands, mean_of(residual, state.count))
    }

    /// The fused pyramid: the batch result of [`build_and_fuse_pyramids`] on the
    /// frames folded.
    ///
    /// # Panics
    /// Panics when no frame was folded.
    #[must_use]
    pub fn finish_pyramid(self) -> LaplacianPyramid {
        let state = self
            .state
            .expect("cannot fuse an empty image stack: fold at least one frame");
        let count = state.count;
        let mut levels = state.values;
        if let Some(residual) = levels.last_mut() {
            *residual = mean_of(residual, count);
        }
        LaplacianPyramid { levels }
    }

    /// The fused image: [`Self::finish_pyramid`] reconstructed.
    ///
    /// # Panics
    /// Panics when no frame was folded.
    #[must_use]
    pub fn finish(self) -> PlanarImage<f32> {
        self.finish_pyramid().reconstruct()
    }
}

/// `sum / count` per sample, the residual level's final form.
#[expect(
    clippy::cast_precision_loss,
    reason = "the number of source frames is far below 2^24"
)]
fn mean_of(sum: &PlanarImage<f32>, count: usize) -> PlanarImage<f32> {
    let n = count as f32;
    let divide = |plane: &[f32]| -> Vec<f32> { plane.par_iter().map(|&s| s / n).collect() };
    PlanarImage::from_planes(
        sum.width,
        sum.height,
        divide(&sum.luma),
        divide(&sum.chroma_a),
        divide(&sum.chroma_b),
    )
    .expect("each mean plane holds width * height samples")
}

impl FoldState {
    /// The accumulator after the first frame: that frame's pyramid is the
    /// starting winner everywhere (the batch rule's default winner is source 0).
    fn start(pyramid: LaplacianPyramid, use_color: bool, grit_suppression: bool) -> Self {
        let mut values = pyramid.levels;
        let last = values.len() - 1;
        let mut best: Vec<Vec<f32>> = Vec::with_capacity(last);
        let mut pixel: Vec<f32> = Vec::new();
        let mut energy: Vec<f32> = Vec::new();
        for (k, level) in values.iter_mut().enumerate() {
            if k == last {
                // Residual: the running sum starts from 0.0 + the first value.
                for plane in [&mut level.luma, &mut level.chroma_a, &mut level.chroma_b] {
                    plane.par_iter_mut().for_each(|v| *v += 0.0);
                }
            } else if k == 0 && grit_suppression {
                pixel = vec![0.0_f32; level.width * level.height];
                energy = vec![0.0_f32; level.width * level.height];
                grit_energy_plane(level, use_color, &mut pixel, &mut energy);
                best.push(energy.iter().map(|&e| initial_best(e)).collect());
            } else {
                best.push(
                    (0..level.width * level.height)
                        .into_par_iter()
                        .map(|i| initial_best(pixel_energy(level, i, use_color)))
                        .collect(),
                );
            }
        }
        Self {
            values,
            best,
            count: 1,
            pixel,
            energy,
        }
    }

    /// Fold one more frame's pyramid in.
    fn fold_in(&mut self, pyramid: &LaplacianPyramid, use_color: bool, grit_suppression: bool) {
        assert_eq!(
            pyramid.levels.len(),
            self.values.len(),
            "all frames must give pyramids with the same number of levels (same size)"
        );
        let last = self.values.len() - 1;
        for (k, (acc, src)) in self.values.iter_mut().zip(&pyramid.levels).enumerate() {
            assert!(
                acc.width == src.width && acc.height == src.height,
                "all frames must have the same size"
            );
            if k == last {
                for (sum, add) in [
                    (&mut acc.luma, &src.luma),
                    (&mut acc.chroma_a, &src.chroma_a),
                    (&mut acc.chroma_b, &src.chroma_b),
                ] {
                    sum.par_iter_mut()
                        .zip(add.par_iter())
                        .for_each(|(s, &v)| *s += v);
                }
            } else if k == 0 && grit_suppression {
                grit_energy_plane(src, use_color, &mut self.pixel, &mut self.energy);
                let energy = &self.energy;
                select_strongest(acc, &mut self.best[k], src, |i| energy[i]);
            } else {
                select_strongest(acc, &mut self.best[k], src, |i| {
                    pixel_energy(src, i, use_color)
                });
            }
        }
        self.count += 1;
    }
}

/// A first-frame energy as the starting `best_energy`: the batch rule starts from
/// `-inf` and a NaN energy never beats it, so NaN maps to `-inf` (a later frame
/// with any comparable energy then wins, as in batch).
const fn initial_best(energy: f32) -> f32 {
    if energy > f32::NEG_INFINITY {
        energy
    } else {
        f32::NEG_INFINITY
    }
}

/// Per pixel, replace the accumulated coefficients of `acc` by those of `src`
/// where `energy(i)` is strictly larger than `best[i]` (and update `best`).
fn select_strongest(
    acc: &mut PlanarImage<f32>,
    best: &mut [f32],
    src: &PlanarImage<f32>,
    energy: impl Fn(usize) -> f32 + Sync,
) {
    best.par_chunks_mut(FOLD_CHUNK)
        .zip(acc.luma.par_chunks_mut(FOLD_CHUNK))
        .zip(acc.chroma_a.par_chunks_mut(FOLD_CHUNK))
        .zip(acc.chroma_b.par_chunks_mut(FOLD_CHUNK))
        .zip(src.luma.par_chunks(FOLD_CHUNK))
        .zip(src.chroma_a.par_chunks(FOLD_CHUNK))
        .zip(src.chroma_b.par_chunks(FOLD_CHUNK))
        .enumerate()
        .for_each(|(chunk, ((((((b, ol), oa), ob), sl), sa), sb))| {
            let base = chunk * FOLD_CHUNK;
            for (j, best_j) in b.iter_mut().enumerate() {
                let e = energy(base + j);
                if e > *best_j {
                    *best_j = e;
                    ol[j] = sl[j];
                    oa[j] = sa[j];
                    ob[j] = sb[j];
                }
            }
        });
}

/// Fuse the frames of `frames` with the streaming [`ApexFold`].
///
/// The result of [`build_and_fuse_pyramids`] followed by a reconstruction, bit
/// for bit, but only one frame and one accumulator pyramid are alive at a time.
///
/// `on_frame(done, &fold)` is called after each frame (see
/// [`ApexFold::fold_source`]).
///
/// # Panics
/// Panics if `frames` is empty or the frames differ in size.
#[must_use]
pub fn fuse_streaming(
    frames: &(impl FrameSource + ?Sized),
    max_levels: usize,
    use_color: bool,
    grit_suppression: bool,
    on_frame: impl FnMut(usize, &ApexFold),
) -> PlanarImage<f32> {
    assert!(!frames.is_empty(), "cannot fuse an empty image stack");
    let mut fold = ApexFold::new(max_levels, use_color, grit_suppression);
    fold.fold_source(frames, on_frame);
    fold.finish()
}

#[cfg(test)]
mod apron_tests {
    use super::{apron_for_levels, pyramid_dependence_radius, pyramid_grid};

    #[test]
    fn apron_follows_the_pyramid_depth() {
        assert_eq!(pyramid_dependence_radius(6), 125);
        assert_eq!(pyramid_dependence_radius(7), 253);
        assert_eq!(pyramid_dependence_radius(8), 509);
        assert_eq!(apron_for_levels(6), 128);
        assert_eq!(apron_for_levels(7), 256);
        assert_eq!(apron_for_levels(8), 512);
        assert_eq!(apron_for_levels(4), 32);
        assert_eq!(apron_for_levels(2), 6);
    }

    #[test]
    fn apron_covers_the_radius_and_stays_on_the_grid() {
        for levels in 0..=14 {
            let apron = apron_for_levels(levels);
            assert!(
                apron >= pyramid_dependence_radius(levels),
                "levels {levels}"
            );
            assert_eq!(apron % pyramid_grid(levels), 0, "levels {levels}");
            assert!(
                apron < pyramid_dependence_radius(levels) + pyramid_grid(levels),
                "levels {levels}: rounded up by less than one grid step"
            );
        }
    }

    #[test]
    fn levels_below_the_minimum_use_the_minimum() {
        assert_eq!(apron_for_levels(0), apron_for_levels(2));
        assert_eq!(apron_for_levels(1), apron_for_levels(2));
    }
}
