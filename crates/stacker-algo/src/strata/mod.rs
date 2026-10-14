//! Strata: guided-filter soft-blend fusion.
//!
//! [`apex`](crate::apex) takes the winning coefficient per pyramid band and
//! [`relief`](crate::relief) takes the winning frame per pixel. Strata makes a
//! *soft* decision: every frame contributes to every output pixel, weighted by
//! how sharp that frame is there, and the weights are refined by an edge-aware
//! guided filter so the transitions follow real image edges.
//!
//! # The algorithm
//! 1. **Saliency** (pass 1). For each frame the Laplacian magnitude of the luma,
//!    smoothed by [`saliency::BLUR_PASSES`] blur passes. A running argmax keeps
//!    the index of the sharpest frame per pixel (`u16`; a tie goes to the
//!    earlier frame).
//! 2. **Base and detail** (pass 2). Each plane of a frame is split into a base
//!    (a box filter of [`StrataParams::base_radius`]) and a detail
//!    (`frame - base`).
//! 3. **Weights.** The binary winner mask `P_i` (1 where frame `i` has the
//!    sharpest saliency) is refined with the frame's own luma as guidance, at two
//!    scales: wide and smooth for the base (`R_BIG`, `EPS_BIG`), tight for the
//!    detail (`R_SMALL`, `EPS_SMALL`). Both are clamped to `[0, 1]`, and the
//!    detail weight is raised to [`StrataParams::detail_focus`].
//! 4. **Accumulate and normalise.** Base and detail are averaged with their
//!    weights; where the weights of every frame vanish the plain mean of the
//!    layer is used. The result is `base + detail`.
//!
//! Weights come from the luma only and are applied identically to all three
//! planes, so the fusion cannot introduce color fringes. The output is not
//! clamped (the pipeline clamps when it converts to RGB, as for Apex).
//!
//! # Deep stacks and `detail_focus`
//! The guided-filter feathering leaks a small positive detail weight (typically
//! 0.01 to 0.05) to frames next to the winner. On a deep stack the leaks of
//! dozens of losers add up to the winner's weight and the detail layers of many
//! defocused frames dilute the one sharp one. Raising the clamped detail weight
//! to a power re-concentrates it: a winner at 1 stays 1, a leak of 0.05 becomes
//! `0.05^3 = 1.25e-4` at the default exponent 3. Only the detail weight is
//! raised; the softness of the base weight is what suppresses halos.
//! `detail_focus = 1` is the original soft behaviour.
//!
//! # Linear light
//! The guided-filter constants (`EPS_BIG = 0.3`, `EPS_SMALL = 1e-6`) were tuned
//! on gamma-encoded data. In linear light the luma is darker and its local
//! variance smaller. `EPS_BIG` is already far above any variance a `[0, 1]`
//! plane can have (at most 0.25), so the base weight is a smoothed box average
//! of the mask in either encoding. `EPS_SMALL` only has to stay below the
//! variance of real texture; the tests in `tests/strata_fuse.rs` check that a
//! dark, low-contrast sharp region is still taken from its sharp frame.
//!
//! # Memory
//! Both passes stream over the frames. Pass 2 keeps the accumulators (two
//! layers of seven planes), the argmax and **two** frames' base/mask planes (the
//! next frame is prepared on a second task while the current frame's weights
//! are computed). All of it is proportional to the image size and independent of
//! the frame count; [`strata_bytes_per_pixel`] gives the figure. The frames are
//! read through a [`FrameSource`], so a caller may supply them one at a time
//! (both passes ask for each frame once, in order).
//!
//! # Devices
//! The two heavy steps (saliency and the weight refinement) can run elsewhere
//! through [`StrataKernels`]; the default methods are the CPU implementation and
//! a device that cannot serve a call returns `None`.

pub mod saliency;

use crate::relief::guided::{box_filter, guided_filter_pair_luma};
use rayon::prelude::*;
use saliency::{SALIENCY_REACH_PX, compute_saliency, update_argmax};
use stacker_core::{fma::fma_f32, image::PlanarImage};
use std::borrow::Cow;

// ── Public parameters ─────────────────────────────────────────────────────

/// The user-tunable Strata parameters.
///
/// The guided-filter radii and epsilons and the saliency blur are fixed on
/// purpose: they interact, and exposing them invites settings that cannot work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrataParams {
    /// Box-filter radius (pixels) that separates the low-frequency base layer
    /// from the high-frequency detail layer. Default `31`; the settings clamp
    /// is `8..=64`.
    pub base_radius: usize,
    /// Exponent applied to the clamped detail weight ("Detail focus"). Default
    /// `3`; the settings clamp is `1..=5`. `1` leaves the weight alone (the
    /// softest result), `5` is close to winner-takes-all.
    pub detail_focus: u32,
}

impl Default for StrataParams {
    fn default() -> Self {
        Self {
            base_radius: 31,
            detail_focus: 3,
        }
    }
}

// ── Fixed algorithm constants ─────────────────────────────────────────────

/// Wide guided-filter radius of the base weight: a smooth handover so
/// brightness and color blend without seams.
const R_BIG: usize = 45;
/// Guided-filter epsilon of the base weight: strong smoothing, weak edges are
/// ignored.
const EPS_BIG: f32 = 0.3;
/// Tight guided-filter radius of the detail weight: a pixel-accurate handover
/// that prevents ghosted double detail.
const R_SMALL: usize = 7;
/// Guided-filter epsilon of the detail weight: follows every real edge.
const EPS_SMALL: f32 = 1e-6;
/// A layer whose summed weight is below this is treated as "no frame
/// contributed" and falls back to the plain mean of the layer.
const NORM_EPS: f32 = 1e-6;

/// The largest `base_radius` the settings accept (`8..=64`). Duplicated here
/// because `stacker-core` cannot depend on this crate; it keeps the apron
/// assertion below meaningful.
pub const MAX_BASE_RADIUS: usize = 64;

/// Pixels a tile needs around its interior so that a tiled Strata run sees
/// every pixel the whole-image run would read.
///
/// The widest chain is the weight branch: the saliency reads
/// [`SALIENCY_REACH_PX`] (11) pixels around a pixel, the winner mask it
/// produces is refined by the guided filter, which averages the window means of
/// its coefficients (two box filters of radius [`R_BIG`], so `2 * 45 = 90`
/// pixels). The base/detail split only reads `base_radius` (at most 64)
/// pixels. The apron is therefore `max(11 + 90, 64) = 101`.
pub const STRATA_APRON_PX: usize = {
    let weights = SALIENCY_REACH_PX + 2 * R_BIG;
    let split = MAX_BASE_RADIUS;
    if weights > split { weights } else { split }
};

const _: () = assert!(
    stacker_core::memory::APRON_PX >= STRATA_APRON_PX,
    "Strata's neighbourhood exceeds the shared tile apron"
);

// Working set per output pixel, excluding the input frames, in bytes.

/// Two layers (base, detail) of seven `f32` planes each: weighted sums of the
/// three planes, the weight sum, and the unweighted sums of the three planes
/// (for the plain-mean fallback).
const BYTES_ACCUMULATORS: usize = 2 * 7 * 4;
/// The `u16` winner index.
const BYTES_ARGMAX: usize = 2;
/// Two frames' prepared planes (three base planes and the mask), because the
/// next frame is prepared while the current one is refined.
const BYTES_PREPARED: usize = 2 * 4 * 4;
/// The two refined weight planes of the current frame.
const BYTES_WEIGHTS: usize = 2 * 4;
/// Scratch of the guided-filter pair: one `f64` summed-area table and about
/// eight `f32` planes (window means and coefficients).
const BYTES_FILTER_SCRATCH: usize = 8 + 8 * 4;
/// The materialised output (the same size is used for a live preview).
const BYTES_OUTPUT: usize = 3 * 4;

/// Peak working set of a Strata run in bytes per output pixel, excluding the
/// input frames themselves.
///
/// The sum of the accumulators (56), the winner index (2), two frames' prepared
/// planes (32), the refined weights (8), the guided-filter scratch (40) and the
/// output or preview snapshot (12): 150. Pass 1 (saliency) needs less. The
/// filter scratch is an upper estimate of the guided filter's temporaries.
#[must_use]
pub const fn strata_bytes_per_pixel() -> usize {
    BYTES_ACCUMULATORS
        + BYTES_ARGMAX
        + BYTES_PREPARED
        + BYTES_WEIGHTS
        + BYTES_FILTER_SCRATCH
        + BYTES_OUTPUT
}

/// `preview_every` value that disables previews entirely (no mid-run snapshot
/// and no final one).
pub const NO_PREVIEW: usize = usize::MAX;

// ── Frame access ──────────────────────────────────────────────────────────

/// Where Strata reads its frames from. The algorithm makes two passes and asks
/// for every frame once per pass, in index order, so an implementation may load
/// frames on demand and drop them again.
pub trait FrameSource {
    /// Number of frames.
    fn len(&self) -> usize;

    /// `true` when there are no frames.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Frame `index` (`< len()`). Every frame must have the same size.
    fn frame(&self, index: usize) -> Cow<'_, PlanarImage<f32>>;
}

impl FrameSource for [PlanarImage<f32>] {
    fn len(&self) -> usize {
        <[PlanarImage<f32>]>::len(self)
    }

    fn frame(&self, index: usize) -> Cow<'_, PlanarImage<f32>> {
        Cow::Borrowed(&self[index])
    }
}

/// A [`FrameSource`] built from a frame count and a loader closure.
pub struct FnFrames<F> {
    count: usize,
    load: F,
}

impl<F> FnFrames<F>
where
    F: Fn(usize) -> PlanarImage<f32>,
{
    /// A source of `count` frames, frame `i` being `load(i)`.
    #[must_use]
    pub const fn new(count: usize, load: F) -> Self {
        Self { count, load }
    }
}

impl<F> FrameSource for FnFrames<F>
where
    F: Fn(usize) -> PlanarImage<f32>,
{
    fn len(&self) -> usize {
        self.count
    }

    fn frame(&self, index: usize) -> Cow<'_, PlanarImage<f32>> {
        Cow::Owned((self.load)(index))
    }
}

// ── Device seam ───────────────────────────────────────────────────────────

/// Kernels Strata can run elsewhere (a GPU).
///
/// The default methods say "not served", so the caller uses the CPU
/// implementation; an implementation overrides what it can do. A result of the
/// wrong length is ignored and the CPU runs instead.
///
/// The trait requires `Send` because the next frame is prepared on another task
/// while the current frame's weights are computed through these kernels.
pub trait StrataKernels: Send {
    /// Per-pixel saliency of one luma plane (the output of
    /// [`saliency::compute_saliency`]). `None` = could not, the CPU is used.
    fn saliency(&mut self, luma: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
        let _ = (luma, width, height);
        None
    }

    /// Twin of [`guided_filter_pair_luma`]: the guided filter of `src` with
    /// `guidance` at two radius/epsilon pairs, returned as `(first, second)`.
    /// `None` = use the CPU.
    #[allow(clippy::too_many_arguments)]
    fn guided_pair(
        &mut self,
        guidance: &[f32],
        src: &[f32],
        width: usize,
        height: usize,
        radius_a: usize,
        eps_a: f32,
        radius_b: usize,
        eps_b: f32,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let _ = (
            guidance, src, width, height, radius_a, eps_a, radius_b, eps_b,
        );
        None
    }
}

/// The CPU implementation: every kernel is served by the default methods.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuStrata;

impl StrataKernels for CpuStrata {}

// ── Small helpers ─────────────────────────────────────────────────────────

/// Raises an already clamped detail weight to the integer power `exponent`
/// (the "Detail focus").
///
/// Repeated multiplication is used for `1..=5` and `powi` outside that range. `1` returns the weight unchanged; `0.0` and `1.0` are
/// fixed points of every exponent of at least 1.
#[must_use]
#[inline]
pub fn raise_detail_weight(w: f32, exponent: u32) -> f32 {
    match exponent {
        1 => w,
        2 => w * w,
        3 => w * w * w,
        4 => w * w * w * w,
        5 => w * w * w * w * w,
        _ => w.powi(exponent.cast_signed()),
    }
}

/// The fused value of one plane of one layer at a pixel: the weighted mean
/// `acc / norm`, or the plain mean `sum / n` when `norm` is below the
/// underflow guard (every frame's weight vanished there).
#[must_use]
#[inline]
pub fn resolve_layer(acc: f32, norm: f32, sum: f32, n: f32) -> f32 {
    if norm < NORM_EPS { sum / n } else { acc / norm }
}

/// Per-pixel running sums of one layer (base or detail):
/// `[acc_l, acc_a, acc_b, norm, sum_l, sum_a, sum_b]`.
struct Layer {
    cells: Vec<[f32; 7]>,
}

impl Layer {
    fn new(len: usize) -> Self {
        Self {
            cells: vec![[0.0; 7]; len],
        }
    }

    /// Adds one frame: weight plane `w` and the frame's three plane values at
    /// each pixel from `value`.
    fn add(&mut self, w: &[f32], value: impl Fn(usize) -> [f32; 3] + Sync) {
        self.cells
            .par_iter_mut()
            .zip(w.par_iter())
            .enumerate()
            .for_each(|(p, (cell, &wv))| {
                let [l, a, b] = value(p);
                cell[0] = fma_f32(wv, l, cell[0]);
                cell[1] = fma_f32(wv, a, cell[1]);
                cell[2] = fma_f32(wv, b, cell[2]);
                cell[3] += wv;
                cell[4] += l;
                cell[5] += a;
                cell[6] += b;
            });
    }

    /// The fused `[l, a, b]` of this layer at pixel `p`.
    fn resolve(&self, p: usize, n: f32) -> [f32; 3] {
        let c = &self.cells[p];
        [
            resolve_layer(c[0], c[3], c[4], n),
            resolve_layer(c[1], c[3], c[5], n),
            resolve_layer(c[2], c[3], c[6], n),
        ]
    }
}

/// Materialises the fused image (`base + detail`) from the running sums. The
/// final output and a live preview are the same computation.
#[allow(clippy::cast_precision_loss)]
fn snapshot(
    width: usize,
    height: usize,
    n: usize,
    base: &Layer,
    detail: &Layer,
) -> PlanarImage<f32> {
    let n_f = n as f32;
    let mut out = PlanarImage::new(width, height);
    out.luma
        .par_iter_mut()
        .zip(out.chroma_a.par_iter_mut())
        .zip(out.chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(p, ((ol, oa), ob))| {
            let b = base.resolve(p, n_f);
            let d = detail.resolve(p, n_f);
            *ol = b[0] + d[0];
            *oa = b[1] + d[1];
            *ob = b[2] + d[2];
        });
    out
}

/// The CPU-side preparation of one frame: its three base planes and the winner
/// mask `P_i` (materialised from the argmax so that only one mask is resident).
struct Prepared {
    base_l: Vec<f32>,
    base_a: Vec<f32>,
    base_b: Vec<f32>,
    mask: Vec<f32>,
}

fn prepare_frame(
    frame: &PlanarImage<f32>,
    argmax_idx: &[u16],
    base_radius: usize,
    index: usize,
) -> Prepared {
    let (w, h) = (frame.width, frame.height);
    let mask = argmax_idx
        .par_iter()
        .map(|&winner| {
            if usize::from(winner) == index {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    Prepared {
        base_l: box_filter(&frame.luma, w, h, base_radius),
        base_a: box_filter(&frame.chroma_a, w, h, base_radius),
        base_b: box_filter(&frame.chroma_b, w, h, base_radius),
        mask,
    }
}

// ── Entry points ──────────────────────────────────────────────────────────

/// Fuses `frames` with Strata on the CPU.
///
/// # Panics
/// Panics if `frames` is empty, holds more than 65 536 frames, or the frames
/// differ in size.
#[must_use]
pub fn fuse_strata(frames: &[PlanarImage<f32>], params: &StrataParams) -> PlanarImage<f32> {
    fuse_strata_with_progress(frames, params, || {})
}

/// Fuses `frames` with Strata, calling `on_tick` once per frame per pass: `2 *
/// frames.len()` times in all (`N` during the saliency pass, `N` during the
/// accumulation pass).
///
/// # Panics
/// Panics if `frames` is empty, holds more than 65 536 frames, or the frames
/// differ in size.
#[must_use]
pub fn fuse_strata_with_progress(
    frames: &[PlanarImage<f32>],
    params: &StrataParams,
    on_tick: impl FnMut(),
) -> PlanarImage<f32> {
    fuse_strata_on(
        &mut CpuStrata,
        frames,
        params,
        on_tick,
        NO_PREVIEW,
        |_: &PlanarImage<f32>| {},
    )
}

/// Fuses `frames` with Strata, with progress ticks and live previews.
///
/// `on_tick` is called as in [`fuse_strata_with_progress`]. `on_preview` gets a
/// snapshot of the composite during the second pass after every `preview_every`
/// frames (`0` is treated as `1`), never after the last frame of the pass; one
/// more call with the finished image follows, so the last preview equals the
/// returned image. [`NO_PREVIEW`] switches previews off (including the final
/// one). A snapshot costs one extra pass over the image.
///
/// # Panics
/// Panics if `frames` is empty, holds more than 65 536 frames, or the frames
/// differ in size.
#[must_use]
pub fn fuse_strata_with_preview(
    frames: &[PlanarImage<f32>],
    params: &StrataParams,
    on_tick: impl FnMut(),
    preview_every: usize,
    on_preview: impl FnMut(&PlanarImage<f32>),
) -> PlanarImage<f32> {
    fuse_strata_on(
        &mut CpuStrata,
        frames,
        params,
        on_tick,
        preview_every,
        on_preview,
    )
}

/// The Strata implementation behind every entry point.
///
/// `kernels` may serve the saliency and the weight refinement (see
/// [`StrataKernels`]); what it declines runs on the CPU. `frames` can be a
/// slice or any [`FrameSource`].
///
/// # Panics
/// Panics if `frames` is empty, holds more than 65 536 frames, or the frames
/// differ in size (or in size from the first frame).
#[must_use]
pub fn fuse_strata_on(
    kernels: &mut impl StrataKernels,
    frames: &(impl FrameSource + ?Sized),
    params: &StrataParams,
    mut on_tick: impl FnMut(),
    preview_every: usize,
    mut on_preview: impl FnMut(&PlanarImage<f32>),
) -> PlanarImage<f32> {
    let n = frames.len();
    assert!(n > 0, "cannot fuse an empty frame stack");
    assert!(
        n <= usize::from(u16::MAX) + 1,
        "Strata supports at most 65 536 frames"
    );
    let first = frames.frame(0);
    let (width, height) = (first.width, first.height);
    drop(first);
    let len = width * height;
    let previews_enabled = preview_every != NO_PREVIEW;
    let preview_every = preview_every.max(1);

    // ── Pass 1: saliency, running argmax ──
    let mut argmax_idx = vec![0_u16; len];
    {
        let mut argmax_val = vec![f32::NEG_INFINITY; len];
        for i in 0..n {
            let frame = frames.frame(i);
            check_size(&frame, width, height);
            let map = kernels
                .saliency(&frame.luma, width, height)
                .filter(|m| m.len() == len)
                .unwrap_or_else(|| compute_saliency(&frame.luma, width, height));
            update_argmax(&mut argmax_idx, &mut argmax_val, &map, i);
            on_tick();
        }
    }

    // ── Pass 2: split, refine the weights, accumulate ──
    let mut base = Layer::new(len);
    let mut detail = Layer::new(len);

    let mut current = frames.frame(0);
    check_size(&current, width, height);
    let mut prepared = prepare_frame(&current, &argmax_idx, params.base_radius, 0);

    for i in 0..n {
        let next = (i + 1 < n).then(|| frames.frame(i + 1));
        if let Some(next) = &next {
            check_size(next, width, height);
        }

        // The next frame's CPU preparation runs while this frame's weights are
        // computed (possibly on a device).
        let mut next_prepared = None;
        let (mut w_base, mut w_detail) = rayon::scope(|scope| {
            if let Some(next) = next.as_deref() {
                let slot = &mut next_prepared;
                let idx = &argmax_idx;
                let base_radius = params.base_radius;
                scope.spawn(move |_| {
                    *slot = Some(prepare_frame(next, idx, base_radius, i + 1));
                });
            }
            refine_weights(kernels, &current, &prepared.mask)
        });

        w_base.par_iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
        let focus = params.detail_focus;
        w_detail
            .par_iter_mut()
            .for_each(|v| *v = raise_detail_weight(v.clamp(0.0, 1.0), focus));

        {
            let frame: &PlanarImage<f32> = &current;
            let prep = &prepared;
            base.add(&w_base, |p| {
                [prep.base_l[p], prep.base_a[p], prep.base_b[p]]
            });
            detail.add(&w_detail, |p| {
                [
                    frame.luma[p] - prep.base_l[p],
                    frame.chroma_a[p] - prep.base_a[p],
                    frame.chroma_b[p] - prep.base_b[p],
                ]
            });
        }
        on_tick();

        if let Some(next) = next {
            current = next;
            prepared = next_prepared
                .take()
                .expect("the scope prepared the next frame");
            if previews_enabled && (i + 1) % preview_every == 0 {
                on_preview(&snapshot(width, height, n, &base, &detail));
            }
        }
    }

    let out = snapshot(width, height, n, &base, &detail);
    if previews_enabled {
        on_preview(&out);
    }
    out
}

/// The refined `(base weight, detail weight)` of the current frame, from the
/// device when it serves the call, else from the CPU guided filter. Guidance is
/// the frame's own luma; the source is its winner mask.
fn refine_weights(
    kernels: &mut impl StrataKernels,
    frame: &PlanarImage<f32>,
    mask: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (frame.width, frame.height);
    let len = w * h;
    kernels
        .guided_pair(&frame.luma, mask, w, h, R_BIG, EPS_BIG, R_SMALL, EPS_SMALL)
        .filter(|(a, b)| a.len() == len && b.len() == len)
        .unwrap_or_else(|| {
            guided_filter_pair_luma(&frame.luma, mask, w, h, R_BIG, EPS_BIG, R_SMALL, EPS_SMALL)
        })
}

fn check_size(frame: &PlanarImage<f32>, width: usize, height: usize) {
    assert_eq!(frame.width, width, "all frames must share one width");
    assert_eq!(frame.height, height, "all frames must share one height");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_layer_falls_back_to_the_plain_mean() {
        // Weighted mean when the weights are meaningful.
        assert!((resolve_layer(0.6, 2.0, 9.0, 3.0) - 0.3).abs() < 1e-7);
        // Plain mean (sum / n) when the weight sum underflows.
        assert!((resolve_layer(0.0, 0.0, 9.0, 3.0) - 3.0).abs() < 1e-7);
        assert!(resolve_layer(0.0, 5e-7, 6.0, 2.0).is_finite());
    }

    #[test]
    fn apron_matches_its_derivation() {
        assert_eq!(STRATA_APRON_PX, 101);
        const { assert!(STRATA_APRON_PX >= 2 * R_BIG) };
        const { assert!(STRATA_APRON_PX >= MAX_BASE_RADIUS) };
    }

    #[test]
    fn bytes_per_pixel_is_the_documented_sum() {
        assert_eq!(strata_bytes_per_pixel(), 150);
    }
}
