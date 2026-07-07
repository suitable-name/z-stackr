use crate::apex::pyramid::LaplacianPyramid;
use rayon::prelude::*;
use stacker_core::image::PlanarImage;

// ── Private blend helpers ────────────────────────────────────────────────────

/// Average all source pixels at `level_idx` into the output buffers.
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
            // the argmax. `max_by` re-evaluated the energy of *both* operands on
            // every comparison (~2× the work); this evaluates each source once.
            // `>=` reproduces `Iterator::max_by`'s "last maximum wins" tie-break,
            // so the selection is identical to the previous implementation.
            let mut winner = 0usize;
            let mut best_e = f32::NEG_INFINITY;
            for (k, p) in pyramids.iter().enumerate() {
                let e = pixel_energy(&p.levels[level_idx], i, use_color);
                if e >= best_e {
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
#[allow(clippy::too_many_arguments)]
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
    out_luma
        .par_chunks_mut(width)
        .zip(out_chroma_a.par_chunks_mut(width))
        .zip(out_chroma_b.par_chunks_mut(width))
        .enumerate()
        .for_each(|(row, ((rl, ra), rb))| {
            for col in 0..width {
                // Evaluate each source's 3×3 neighbourhood energy exactly once,
                // then take the argmax. The previous `max_by` recomputed the
                // (9-pixel) neighbourhood energy for *both* operands on every
                // comparison — ~2× the work on the largest pyramid level. `>=`
                // reproduces `max_by`'s "last maximum wins" tie-break.
                let mut winner = 0usize;
                let mut best_e = f32::NEG_INFINITY;
                for (k, p) in pyramids.iter().enumerate() {
                    let lvl = &p.levels[level_idx];
                    let mut e = 0.0_f32;
                    for dr in -1_isize..=1 {
                        for dc in -1_isize..=1 {
                            let nr = row as isize + dr;
                            let nc = col as isize + dc;
                            if nr >= 0 && nr < height as isize && nc >= 0 && nc < width as isize {
                                let ni = nr as usize * width + nc as usize;
                                e += pixel_energy(lvl, ni, use_color);
                            }
                        }
                    }
                    if e >= best_e {
                        best_e = e;
                        winner = k;
                    }
                }

                let lvl = &pyramids[winner].levels[level_idx];
                let pi = row * width + col;
                rl[col] = lvl.luma[pi];
                ra[col] = lvl.chroma_a[pi];
                rb[col] = lvl.chroma_b[pi];
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
///   - `grit_suppression = true` (default): winner chosen by summing Laplacian
///     energy over a 3×3 neighborhood with clamped boundary.  Neighborhood
///     energy reduces single-pixel noise artefacts on the highest-frequency band.
///   - `grit_suppression = false`: winner chosen by per-pixel energy only.
/// - **All other Laplacian levels**: winner chosen by per-pixel energy.
///
/// The energy metric for winner selection is controlled by `use_color`:
/// - `use_color = false` (default): energy = luma² only.  Chroma noise cannot
///   steer the winner.
/// - `use_color = true`: energy = luma² + `chroma_a²` + `chroma_b²`.
///
/// In every winning case all three channels (luma, `chroma_a`, `chroma_b`) are
/// taken from the *same* source so they remain coherent and false colours
/// are avoided.
///
/// # GPU dispatch
/// When compiled with the `gpu` feature, this tries a `wgpu` compute-shader
/// port of the exact per-level semantics above first
/// ([`crate::apex::gpu::fuse_pyramids_gpu`]) and falls back to the CPU/rayon
/// implementation below on any failure (no adapter available, or any `wgpu`
/// call erroring) — never a panic. See `apex::gpu`'s module docs for the
/// tolerance-equal (not bit-equal) GPU/CPU parity guarantee and its tested
/// epsilon. [`ApexAccumulator`]'s incremental whole-image path (used by the
/// GUI's in-RAM mode) has its own, separate GPU-resident accelerator — see
/// [`fuse_pyramids_incremental_with_progress`] and
/// `crate::apex::gpu::accumulator`'s module docs.
pub fn fuse_pyramids(
    pyramids: &[LaplacianPyramid],
    use_color: bool,
    grit_suppression: bool,
) -> LaplacianPyramid {
    assert!(
        !pyramids.is_empty(),
        "cannot fuse an empty array of pyramids"
    );

    // GPU dispatch: try the GPU compute-shader path first (adapter acquired,
    // per-level semantics matching the CPU code below — see
    // `apex::gpu`'s module docs), and fall back to the CPU/rayon
    // implementation on ANY failure (no adapter, a `wgpu` call erroring,
    // etc.) — never a panic, and the CPU body below always stays reachable.
    // The chosen backend is logged once by `stacker_core::gpu::context`.
    #[cfg(feature = "gpu")]
    if let Some(fused) = crate::apex::gpu::fuse_pyramids_gpu(pyramids, use_color, grit_suppression)
    {
        return fused;
    }

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

        fused_levels.push(PlanarImage {
            width,
            height,
            luma: fused_luma,
            chroma_a: fused_chroma_a,
            chroma_b: fused_chroma_b,
        });
    }

    LaplacianPyramid {
        levels: fused_levels,
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
pub fn build_and_fuse_pyramids(
    images: &[PlanarImage<f32>],
    max_levels: usize,
    use_color: bool,
    grit_suppression: bool,
) -> LaplacianPyramid {
    assert!(!images.is_empty(), "cannot fuse an empty image stack");
    let pyramids: Vec<LaplacianPyramid> = images
        .par_iter()
        .map(|img| LaplacianPyramid::build(img, max_levels))
        .collect();
    fuse_pyramids(&pyramids, use_color, grit_suppression)
}

// ── Incremental whole-image accumulator ─────────────────────────────────────

/// Full-depth pyramid level count passed to [`LaplacianPyramid::build`].
///
/// A value of 32 always exceeds the maximum number of halvings possible for
/// any practical image (a 4 `GPixel` image only produces ~16 levels before the
/// early-break condition fires), so the pyramid depth is determined entirely
/// by the image dimensions rather than this constant.
const FULL_DEPTH: usize = 32;

/// `FULL_DEPTH`, re-exported at `pub(crate)` visibility for
/// `apex::gpu::accumulator::GpuApexAccumulator::new`, which needs the exact
/// same level budget the CPU accumulator uses so a mid-stack GPU-to-CPU
/// hand-off (see that module's docs) builds subsequent frames' pyramids to
/// a depth that matches the levels already accumulated on the GPU.
#[cfg(feature = "gpu")]
pub(crate) const FULL_DEPTH_PUB: usize = FULL_DEPTH;

/// Memory-bounded incremental `Apex` focus-stacking accumulator.
///
/// Rather than holding one pyramid per input frame (O(N) memory for N frames),
/// this type maintains a single *accumulator pyramid* that is updated in-place
/// as each new frame arrives.  Peak memory is therefore ~2 whole-image
/// pyramids at any one time: the long-lived accumulator plus the single
/// per-frame pyramid built inside [`blend`](Self::blend) — which is dropped
/// immediately after blending.
///
/// # Selection semantics
///
/// - **Residual (last) level**: running arithmetic mean across all blended
///   frames.  Each call to [`blend`](Self::blend) updates the accumulator
///   with the online formula `acc = (acc * count + src) / (count + 1)`.
/// - **Finest level (index 0) with `grit_suppression = true`**: the centre
///   pixel of the accumulator is replaced by the corresponding source pixel
///   when the source's 3×3-neighbourhood energy strictly exceeds the
///   accumulator's 3×3-neighbourhood energy (ties keep the accumulator, i.e.
///   the earliest frame wins).
/// - **All other Laplacian levels**: per-pixel energy selection.  The source
///   replaces the accumulator pixel when the source energy strictly exceeds
///   the accumulator energy (ties keep the accumulator).
///
/// Energy is luma² when `use_color = false`, or luma² + `chroma_a²` +
/// `chroma_b²` when `use_color = true`.  All three channels are always copied
/// coherently from the winning source to avoid false colours.
pub struct ApexAccumulator {
    levels: Vec<PlanarImage<f32>>,
    count: usize,
    build_levels: usize,
    use_color: bool,
    grit_suppression: bool,
}

impl ApexAccumulator {
    /// Create a new accumulator seeded with `first`.
    ///
    /// A full-depth Laplacian pyramid is built from `first` (using
    /// [`FULL_DEPTH`] as the level budget, so the actual depth is limited only
    /// by the image dimensions).  The resulting levels become the initial
    /// accumulator state; `count` is set to 1.
    ///
    /// Peak memory during construction: the source image plus two full-size
    /// pyramid allocations (one temporary inside [`LaplacianPyramid::build`],
    /// one stored as the accumulator).
    pub fn new(first: &PlanarImage<f32>, use_color: bool, grit_suppression: bool) -> Self {
        let p = LaplacianPyramid::build(first, FULL_DEPTH);
        Self {
            levels: p.levels,
            count: 1,
            build_levels: FULL_DEPTH,
            use_color,
            grit_suppression,
        }
    }

    /// Construct a CPU accumulator resuming from an already-in-progress
    /// state — the mid-stack hand-off from [`crate::apex::gpu::accumulator::GpuApexAccumulator`]
    /// when a GPU dispatch fails part-way through a stack (see that
    /// module's docs for the full rationale). `levels` is the exact
    /// GPU-accumulated state read back via `GpuApexAccumulator::read_back`,
    /// `count` is the number of frames already folded into it, and
    /// `build_levels`/`use_color`/`grit_suppression` must match the values
    /// the `GpuApexAccumulator` was constructed with so subsequent
    /// [`blend`](Self::blend) calls stay numerically consistent with the
    /// frames already accumulated on the GPU.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub(crate) const fn from_gpu_state(
        levels: Vec<PlanarImage<f32>>,
        count: usize,
        build_levels: usize,
        use_color: bool,
        grit_suppression: bool,
    ) -> Self {
        Self {
            levels,
            count,
            build_levels,
            use_color,
            grit_suppression,
        }
    }

    /// Blend one more frame into the accumulator.
    ///
    /// A full-depth pyramid is built from `img`, blended into the accumulator
    /// level-by-level, and then immediately dropped.  Peak memory during this
    /// call is ~2 whole-image pyramids (the accumulator plus the freshly built
    /// source pyramid).
    ///
    /// # Panics
    ///
    /// Panics if the new frame produces a pyramid whose level count or any
    /// level's dimensions differ from those stored in the accumulator.  In
    /// practice this cannot happen when all input frames share the same
    /// dimensions, which is a precondition of focus stacking.
    fn blend_finest_with_grit_suppression(
        use_color: bool,
        src: &PlanarImage<f32>,
        acc: &mut PlanarImage<f32>,
    ) {
        let width = acc.width;
        let height = acc.height;
        let acc_energy: Vec<f32> = (0..width * height)
            .map(|i| pixel_energy(acc, i, use_color))
            .collect();
        let src_energy: Vec<f32> = (0..width * height)
            .map(|i| pixel_energy(src, i, use_color))
            .collect();
        acc.luma
            .par_chunks_mut(width)
            .zip(acc.chroma_a.par_chunks_mut(width))
            .zip(acc.chroma_b.par_chunks_mut(width))
            .enumerate()
            .for_each(|(row, ((rl, ra), rb))| {
                for col in 0..width {
                    // Sum 3×3 neighbourhood energies for acc and src
                    // using the pre-computed snapshots.
                    let mut tgt_e = 0.0_f32;
                    let mut src_e = 0.0_f32;
                    for dr in -1_isize..=1 {
                        for dc in -1_isize..=1 {
                            let nr = row as isize + dr;
                            let nc = col as isize + dc;
                            if nr >= 0 && nr < height as isize && nc >= 0 && nc < width as isize {
                                let ni = nr as usize * width + nc as usize;
                                tgt_e += acc_energy[ni];
                                src_e += src_energy[ni];
                            }
                        }
                    }
                    // Strict >: ties keep the accumulator (earliest frame wins).
                    if src_e > tgt_e {
                        let pi = row * width + col;
                        rl[col] = src.luma[pi];
                        ra[col] = src.chroma_a[pi];
                        rb[col] = src.chroma_b[pi];
                    }
                }
            });
    }

    pub fn blend(&mut self, img: &PlanarImage<f32>) {
        let p = LaplacianPyramid::build(img, self.build_levels);

        let n = self.levels.len();
        assert_eq!(
            p.levels.len(),
            n,
            "source pyramid has {} levels but accumulator has {} levels — \
             all frames must have the same dimensions",
            p.levels.len(),
            n,
        );
        for idx in 0..n {
            assert_eq!(
                p.levels[idx].width, self.levels[idx].width,
                "source pyramid level {idx} width {} differs from accumulator width {} — \
                 all frames must have the same dimensions",
                p.levels[idx].width, self.levels[idx].width,
            );
            assert_eq!(
                p.levels[idx].height, self.levels[idx].height,
                "source pyramid level {idx} height {} differs from accumulator height {} — \
                 all frames must have the same dimensions",
                p.levels[idx].height, self.levels[idx].height,
            );
        }

        let count = self.count;
        let use_color = self.use_color;
        let grit_suppression = self.grit_suppression;

        for idx in 0..n {
            let src = &p.levels[idx];
            let acc = &mut self.levels[idx];

            if idx == n - 1 {
                // Residual (low-frequency) level: running arithmetic mean.
                // acc_new = (acc_old * count + src) / (count + 1)
                let count_f = count as f32;
                acc.luma
                    .par_iter_mut()
                    .zip(acc.chroma_a.par_iter_mut())
                    .zip(acc.chroma_b.par_iter_mut())
                    .zip(src.luma.par_iter())
                    .zip(src.chroma_a.par_iter())
                    .zip(src.chroma_b.par_iter())
                    .for_each(|(((((al, aa), ab), sl), sa), sb)| {
                        *al = (*al * count_f + sl) / (count_f + 1.0);
                        *aa = (*aa * count_f + sa) / (count_f + 1.0);
                        *ab = (*ab * count_f + sb) / (count_f + 1.0);
                    });
            } else if idx == 0 && grit_suppression {
                Self::blend_finest_with_grit_suppression(use_color, src, acc);
            } else {
                // All other Laplacian levels: per-pixel energy selection.
                // Pre-compute the accumulator's energy snapshot to avoid
                // borrowing acc both mutably (via par_iter_mut) and immutably
                // (via pixel_energy) inside the same closure.
                let len = acc.width * acc.height;
                let acc_energy: Vec<f32> =
                    (0..len).map(|i| pixel_energy(acc, i, use_color)).collect();
                // Strict >: ties keep the accumulator (earliest frame wins).
                acc.luma
                    .par_iter_mut()
                    .zip(acc.chroma_a.par_iter_mut())
                    .zip(acc.chroma_b.par_iter_mut())
                    .enumerate()
                    .for_each(|(i, ((al, aa), ab))| {
                        if pixel_energy(src, i, use_color) > acc_energy[i] {
                            *al = src.luma[i];
                            *aa = src.chroma_a[i];
                            *ab = src.chroma_b[i];
                        }
                    });
            }
        }

        self.count += 1;
    }

    /// Consume the accumulator and reconstruct the fused image.
    ///
    /// Wraps the stored levels in a [`LaplacianPyramid`] and calls
    /// [`reconstruct`](LaplacianPyramid::reconstruct).  The accumulator is
    /// consumed so no extra copy is needed.
    pub fn reconstruct(self) -> PlanarImage<f32> {
        LaplacianPyramid {
            levels: self.levels,
        }
        .reconstruct()
    }

    /// Non-consuming reconstruction of the current accumulator state.
    ///
    /// Clones the accumulator's levels (an O(image size) allocation and copy)
    /// before reconstructing, so the accumulator itself remains usable for
    /// further [`blend`](Self::blend) calls afterwards. Intended for periodic
    /// progress previews during [`fuse_pyramids_incremental_with_progress`],
    /// not for per-frame use on large stacks — prefer
    /// [`reconstruct`](Self::reconstruct) when the accumulator is no longer
    /// needed.
    #[must_use]
    pub fn reconstruct_preview(&self) -> PlanarImage<f32> {
        LaplacianPyramid {
            levels: self.levels.clone(),
        }
        .reconstruct()
    }
}

/// Build a fused image from a stack using the incremental whole-image
/// accumulator.
///
/// Peak memory is ~2 whole-image pyramids regardless of `images.len()`,
/// compared with O(N) for the batch [`build_and_fuse_pyramids`] approach.
///
/// `use_color` and `grit_suppression` have the same semantics as in
/// [`fuse_pyramids`].
///
/// # GPU dispatch
/// When compiled with the `gpu` feature, this tries the GPU-resident
/// [`crate::apex::gpu::accumulator::GpuApexAccumulator`] first — see that
/// module's docs for the fallback contract, including the mid-stack
/// GPU-to-CPU hand-off on a dispatch failure after some frames were already
/// accumulated on the GPU.
pub fn fuse_pyramids_incremental(
    images: &[PlanarImage<f32>],
    use_color: bool,
    grit_suppression: bool,
) -> PlanarImage<f32> {
    assert!(!images.is_empty(), "cannot fuse an empty image stack");

    #[cfg(feature = "gpu")]
    if let Some(result) =
        try_fuse_pyramids_incremental_gpu(images, use_color, grit_suppression, |_, _, _| {})
    {
        return result;
    }

    let mut acc = ApexAccumulator::new(&images[0], use_color, grit_suppression);
    for img in &images[1..] {
        acc.blend(img);
    }
    acc.reconstruct()
}

/// GPU-resident incremental accumulation, shared by [`fuse_pyramids_incremental`]
/// and [`fuse_pyramids_incremental_with_progress`].
///
/// Returns `None` — never panics — when no `wgpu` context is available at
/// all (the caller falls back to the plain CPU accumulator for the whole
/// stack in that case). Once a [`crate::apex::gpu::accumulator::GpuApexAccumulator`]
/// exists, a mid-stack `accumulate` failure is handled internally (read back
/// the exact GPU state so far, finish the remaining frames on a CPU
/// accumulator resumed from that state via [`ApexAccumulator::from_gpu_state`])
/// rather than surfaced as `None` — see `crate::apex::gpu::accumulator`'s
/// module docs for why that hand-off, not a full CPU restart, is the right
/// failure mode here.
#[cfg(feature = "gpu")]
fn try_fuse_pyramids_incremental_gpu<F>(
    images: &[PlanarImage<f32>],
    use_color: bool,
    grit_suppression: bool,
    mut on_progress: F,
) -> Option<PlanarImage<f32>>
where
    F: FnMut(usize, usize, Option<PlanarImage<f32>>),
{
    use crate::apex::gpu::accumulator::GpuApexAccumulator;

    let total = images.len();
    let mut gpu_acc = GpuApexAccumulator::new(&images[0], use_color, grit_suppression)?;
    // The CPU path (`fuse_pyramids_incremental_with_progress`) always sends a
    // preview with the very first progress call — it already holds the
    // seeded accumulator in RAM, so `reconstruct_preview` is effectively
    // free there. The GPU path must match that observable contract exactly
    // (per `ApexAccumulator`'s progress-callback semantics), so it pays for
    // one readback here rather than reporting `None` on frame 1: skipping it
    // is an observable behavioural difference from the CPU path, not merely
    // a perf trade-off, and callers (and this crate's own tests) rely on
    // every completed-call sequence carrying a preview on both frame 1 and
    // the final frame. A `read_back` failure here is handled exactly like
    // any other mid-stack GPU failure: hand off to the CPU from the state
    // already seeded (no frame has been lost, `resume_at = 1` re-blends
    // every remaining frame from the CPU accumulator built off that same
    // seed).
    let first_preview = match gpu_acc.read_back() {
        Some(levels) => Some(LaplacianPyramid { levels }.reconstruct()),
        None => {
            return finish_on_cpu_after_gpu_failure(&gpu_acc, images, 1, &mut on_progress);
        }
    };
    on_progress(1, total, first_preview);

    for (offset, img) in images[1..].iter().enumerate() {
        if gpu_acc.accumulate(img) {
            let completed = offset + 2;
            let want_preview = completed == total || completed % PREVIEW_STRIDE == 0;
            let preview = if want_preview {
                // A preview mid-run costs a full readback of every resident
                // level; `read_back` failing here is treated the same as any
                // other GPU failure — hand off to the CPU for the rest of
                // the stack rather than silently skipping this preview.
                match gpu_acc.read_back() {
                    Some(levels) => Some(LaplacianPyramid { levels }.reconstruct()),
                    None => {
                        return finish_on_cpu_after_gpu_failure(
                            &gpu_acc,
                            images,
                            offset + 2,
                            &mut on_progress,
                        );
                    }
                }
            } else {
                None
            };
            on_progress(completed, total, preview);
            continue;
        }

        // `accumulate` itself failed for this frame: hand off to the CPU
        // from the last successfully accumulated state (this frame,
        // `images[offset + 1]`, has NOT been folded in yet, so the CPU
        // accumulator must still blend it).
        return finish_on_cpu_after_gpu_failure(&gpu_acc, images, offset + 1, &mut on_progress);
    }

    // `total` here doubles as "every frame has already been folded in" — if
    // the final readback fails, `finish_on_cpu_after_gpu_failure`
    // (`resume_at == total`) skips its blend loop entirely (the `skip`
    // iterator is empty) and simply re-reads-back + reconstructs on the CPU
    // side, rather than needlessly restarting the whole stack from frame 0.
    // `read_back` (not the consuming `finish`) is used here so `gpu_acc` is
    // still available to pass to the hand-off helper if it fails.
    if let Some(levels) = gpu_acc.read_back() {
        return Some(LaplacianPyramid { levels }.reconstruct());
    }
    finish_on_cpu_after_gpu_failure(&gpu_acc, images, total, &mut on_progress)
}

/// Shared mid-stack GPU-failure hand-off: read back `gpu_acc`'s current
/// state, resume a CPU [`ApexAccumulator`] from it, blend every remaining
/// frame from `resume_at` (the index of the next frame that still needs
/// blending) onward, and report progress for each.
///
/// Returns `None` if even the readback fails (nothing left to resume from —
/// the caller's own CPU-from-scratch fallback then applies).
#[cfg(feature = "gpu")]
fn finish_on_cpu_after_gpu_failure<F>(
    gpu_acc: &crate::apex::gpu::accumulator::GpuApexAccumulator,
    images: &[PlanarImage<f32>],
    resume_at: usize,
    on_progress: &mut F,
) -> Option<PlanarImage<f32>>
where
    F: FnMut(usize, usize, Option<PlanarImage<f32>>),
{
    tracing::debug!(
        resume_at,
        total = images.len(),
        "gpu: incremental apex accumulator failed mid-stack, resuming on CPU from the last good state"
    );
    let levels = gpu_acc.read_back()?;
    let mut cpu_acc = ApexAccumulator::from_gpu_state(
        levels,
        gpu_acc.count(),
        gpu_acc.build_levels(),
        gpu_acc.use_color(),
        gpu_acc.grit_suppression(),
    );

    let total = images.len();
    for (i, img) in images.iter().enumerate().skip(resume_at) {
        cpu_acc.blend(img);
        let completed = i + 1;
        let want_preview = completed == total || completed % PREVIEW_STRIDE == 0;
        on_progress(
            completed,
            total,
            want_preview.then(|| cpu_acc.reconstruct_preview()),
        );
    }
    Some(cpu_acc.reconstruct())
}

/// Number of frames between reconstructed preview snapshots in
/// [`fuse_pyramids_incremental_with_progress`].
///
/// A full reconstruction costs roughly as much as a single
/// [`ApexAccumulator::blend`] call, so previewing every frame would ~double
/// runtime on large stacks. Progress (without a preview image) is still
/// reported on every frame.
const PREVIEW_STRIDE: usize = 4;

/// Progress-reporting variant of [`fuse_pyramids_incremental`].
///
/// `on_progress(completed, total, preview)` is called once per frame, with
/// `completed` counting `1..=total`. `preview` carries a freshly
/// reconstructed snapshot of the accumulator on the first frame, the last
/// frame, and every [`PREVIEW_STRIDE`]th frame in between; it is `None` on
/// the other calls so callers can cheaply report progress without paying for
/// a reconstruction on every frame.
///
/// # GPU dispatch
/// When compiled with the `gpu` feature, this tries the GPU-resident
/// [`crate::apex::gpu::accumulator::GpuApexAccumulator`] first — see that
/// module's docs for the fallback contract, including the mid-stack
/// GPU-to-CPU hand-off on a dispatch failure after some frames were already
/// accumulated on the GPU. `preview` snapshots on the GPU path cost a full
/// readback of every resident level (same relative cost as the CPU path's
/// own [`ApexAccumulator::reconstruct_preview`]), so they are taken on the
/// same [`PREVIEW_STRIDE`] schedule; the very first progress call (frame 1)
/// also carries a preview on the GPU path, matching the CPU path's own
/// frame-1 preview exactly (both read back/clone the freshly seeded
/// accumulator, so the observable progress-callback contract is identical
/// regardless of which backend is engaged).
pub fn fuse_pyramids_incremental_with_progress<F>(
    images: &[PlanarImage<f32>],
    use_color: bool,
    grit_suppression: bool,
    mut on_progress: F,
) -> PlanarImage<f32>
where
    F: FnMut(usize, usize, Option<PlanarImage<f32>>),
{
    assert!(!images.is_empty(), "cannot fuse an empty image stack");

    #[cfg(feature = "gpu")]
    if let Some(result) =
        try_fuse_pyramids_incremental_gpu(images, use_color, grit_suppression, &mut on_progress)
    {
        return result;
    }

    let total = images.len();
    let mut acc = ApexAccumulator::new(&images[0], use_color, grit_suppression);
    on_progress(1, total, Some(acc.reconstruct_preview()));

    for (offset, img) in images[1..].iter().enumerate() {
        acc.blend(img);
        let completed = offset + 2;
        let want_preview = completed == total || completed % PREVIEW_STRIDE == 0;
        on_progress(
            completed,
            total,
            want_preview.then(|| acc.reconstruct_preview()),
        );
    }

    acc.reconstruct()
}

// Tests moved to tests/apex_fuse.rs
