//! Strata — guided-filter soft-blend focus-stacking fusion engine.
//!
//! See `docs/strata-fusion-design.md` for the full derivation and rationale.
//! Summary: unlike `Apex` (hard per-band winner-take-all) and `Relief` (hard
//! per-pixel argmax), Strata makes a SOFT decision — every frame contributes
//! to every output pixel, weighted by how in-focus that frame is *there*,
//! with the weight maps refined by an edge-aware guided filter so weight
//! transitions hug real image edges instead of cutting across them. Two
//! refinement scales are used (smooth/wide for the low-frequency "base"
//! layer, tight/crisp for the high-frequency "detail" layer), which is what
//! suppresses halos at depth edges.
//!
//! All processing is luma-only for saliency/weights (§2's rationale: weights
//! answer "which frame is sharp here", a luminance question — applying one
//! weight set to all three planes guarantees no fusion-induced colour
//! fringing), applied identically to `luma`, `chroma_a`, `chroma_b`.
//!
//! # Deep-stack detail dilution (and the fix)
//! The Li-Kang-Hu formulation this module ports is a 2-frame-oriented
//! method; its guided-filter feathering was designed to soften a single
//! hard boundary between two winners. On deep stacks (dozens of frames)
//! that same feathering leaks a small positive `w_detail` (typically
//! 0.01-0.05) from the sigma~=2.4 saliency blur plus guided-filter
//! smoothing to every near-winner frame, not just the true local winner.
//! With `N~=50`, the ~49 losers' leaked weights sum to `O(1)` -
//! comparable to the winner's own weight near 1 - so `Sum(w_detail*D_i) /
//! Sum(w_detail)` mixes in dozens of defocused detail layers alongside the
//! one sharp one, visible as a global "watercolour" softness that Step 1's
//! exposed `base_radius` cannot touch (it only moves the base/detail split
//! point, not the detail-layer weights). See [`DETAIL_WEIGHT_EXPONENT`]'s
//! doc comment for the fix — now a runtime setting, [`StrataParams::detail_focus`]
//! ("Detail focus"), rather than a fixed constant.
//!
//! # GPU acceleration
//! With the `gpu` feature compiled in, the per-pixel Laplacian-magnitude
//! saliency pass ([`saliency::compute_saliency`]) tries a `wgpu`
//! compute-shader dispatch first, falling back transparently to the CPU
//! path on any failure — see [`gpu`]'s module docs for the exact scope and
//! fallback contract. The per-frame weight refinement below (Step 4's
//! [`guided_filter_pair`] call, producing `w_base`/`w_detail` together) is
//! GPU-accelerated transitively: `guided_filter_pair` tries a fused GPU
//! pipeline (`relief::gpu::guided_filter_pair_gpu`) internally — one upload
//! of this frame's `(luma, p_image)` pair, both radius chains dispatched
//! under a single held dispatch guard, two readbacks batched at the end
//! (see that function's doc comment for why this beats two independent
//! `guided_filter` calls). Step 1's direct [`box_filter`] call (the
//! base/detail split) is **always CPU** — it is not part of that fused
//! dispatch and is cheap enough (one box filter per frame per plane, not
//! six) that a separate GPU path was not worth adding; see
//! `relief::guided::box_filter`'s doc comment for why per-call GPU dispatch
//! was removed from `box_filter` entirely.

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]

use rayon::prelude::*;
use stacker_core::image::PlanarImage;

// GPU acceleration (`gpu` feature): only the per-pixel Laplacian-magnitude
// saliency pass (`saliency::compute_saliency`'s first step) is
// GPU-accelerated — see `strata::gpu`'s module docs for the fallback
// contract and scope, and `docs/gpu_acceleration_summary.md` for the full
// architecture writeup.
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod saliency;

use crate::relief::guided::{box_filter, guided_filter_pair};
use saliency::compute_saliency;

// ── Public parameters ────────────────────────────────────────────────────

/// User-tunable Strata parameters.
///
/// `base_radius` and `detail_focus` are the only two settings exposed — see
/// the fixed constants below for why the guided-filter radii/eps and blur
/// sigma are not settings (design doc §3: "they interact; exposing them
/// invites unfixable 'bad settings' reports").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrataParams {
    /// Box-filter radius (pixels) separating the low-frequency "base" layer
    /// from the high-frequency "detail" layer (design doc §2 Step 1).
    /// Default `31`; settings-clamped `8..=64` by
    /// `stacker_core::settings::StackingSettings::clamp_valid`.
    pub base_radius: usize,
    /// Detail-layer weight re-concentration exponent ("Detail focus"),
    /// applied as `w_detail <- w_detail^detail_focus` after the `[0, 1]`
    /// clamp (design doc §2 Step 4) and before accumulation — the runtime
    /// knob for what used to be the fixed [`DETAIL_WEIGHT_EXPONENT`]
    /// constant (gamma = 3). See [`raise_detail_weight`]'s doc comment for
    /// the deep-stack dilution rationale this generalises.
    ///
    /// Default `3` (the original fixed behaviour). Settings-clamped
    /// `1..=5` by `stacker_core::settings::StackingSettings::clamp_valid`:
    /// `1` is a no-op (the original 2-frame-oriented Li-Kang-Hu softness,
    /// best for flat/glossy subjects where the leaked-weight dilution this
    /// exponent fights barely matters), `5` is near winner-take-all
    /// (crispest depth edges, most detail retention on deep stacks, at the
    /// cost of the risk of a harder transition at ambiguous boundaries).
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

// ── Fixed algorithm constants ────────────────────────────────────────────

/// Wide guided-filter radius for the base-layer weight refinement: smooth,
/// wide handover so brightness/colour blends without visible seams.
const R_BIG: usize = 45;
/// Guided-filter epsilon for the base layer: strong smoothing, ignores weak
/// edges.
const EPS_BIG: f32 = 0.3;
/// Tight guided-filter radius for the detail-layer weight refinement:
/// pixel-accurate handover, prevents ghosted double detail.
const R_SMALL: usize = 7;
/// Guided-filter epsilon for the detail layer: hugs every real edge.
const EPS_SMALL: f32 = 1e-6;

/// Number of times the existing 5-tap separable blur
/// (`apex::pyramid::apply_gaussian_blur`) is applied to approximate the
/// design doc's target local-averaging `sigma ~= 2.5` (§2 Step 2).
///
/// That kernel (`[0.085, 0.25, 0.33, 0.25, 0.085]`) approximates a single
/// Gaussian pass with variance `2*(0.085*2^2 + 0.25*1^2) = 1.18` (`sigma ~=
/// 1.09`). Repeated convolution with the same kernel sums variances, so `k`
/// applications approximate `sigma = sqrt(k * 1.18)`; `k = 5` gives `sigma ~=
/// 2.43` — close to the design's target without introducing a second blur
/// kernel into the codebase (design doc §2 Step 2 explicitly permits reusing
/// the existing separable blur "with a small sigma").
const BLUR_PASSES: usize = 5;

/// Fraction-of-total guard below which a pixel's per-layer weight sum is
/// treated as "no frame meaningfully contributed" and that layer falls back
/// to a plain `1/N` average (design doc §2 Step 5's "flat wall" case).
const NORM_EPS: f32 = 1e-6;

/// Default detail-layer weight re-concentration exponent (gamma = 3) — the
/// value [`StrataParams::detail_focus`] defaults to, and the value this
/// module shipped with before the exponent became a runtime setting
/// ("Detail focus"). Applied as `w_detail <- w_detail^detail_focus` after
/// the `[0, 1]` clamp (design doc §2 Step 4) and before accumulation into
/// `acc_detail_*`/`norm_detail`.
///
/// *Why this exists — deep-stack detail dilution:* the module doc's "Deep-
/// stack detail dilution" section above spells out the failure this fixes.
/// The paper's guided-filter feathering was tuned for 2-6 frames, where a
/// handful of `0.01-0.05` leaked weights from non-winning frames are
/// negligible next to the winner's ~1. On deep stacks (dozens of frames)
/// those leaked weights sum to `O(1)` and dilute the winner's sharp detail
/// with many defocused ones, producing a global "watercolour" softness no
/// exposed setting can reach.
///
/// Raising `w_detail` to this power after the clamp re-concentrates that
/// competition:
/// * A winner at `w ~= 1` stays `1^n = 1` — untouched, for any `n`.
/// * A loser leaking `w ~= 0.05` collapses much faster than linearly (at
///   the default `n = 3`: `0.05^3 = 1.25e-4`, roughly 400x smaller,
///   effectively removed from the sum) — the higher `detail_focus` is set,
///   the more aggressively leaked weight is suppressed.
/// * Because the guided filter's edge-aware feathering still produces a
///   smooth ramp of intermediate weights through a real transition zone
///   (not a step function), raising that ramp to a power sharpens the
///   transition while it remains continuous — no new discontinuity is
///   introduced at depth edges.
///
/// Deliberately asymmetric: only `w_detail` is raised to this power.
/// `w_base`'s softness *is* the halo suppression (design doc §2's "why two
/// scales" — the base layer carries low-frequency brightness/colour, which
/// is insensitive to being averaged across several near-winner frames), so
/// re-concentrating it would reintroduce the seams the wide/smooth base
/// radius exists to avoid. This exponent is a deliberate deviation from the
/// original 2-frame-oriented Li-Kang-Hu formulation, not a literal port of
/// it — `detail_focus = 1` recovers that original (no-op) behaviour exactly.
///
/// A no-op for the exact `0.0`/`1.0` weights the small-stack behaviour
/// contracts (constant-stack identity, lowest-index ties, the flat-region
/// fallback, and the two-half-sharp-frames test) produce — `0^n = 0` and
/// `1^n = 1` for any `n >= 1` — so those tests are unaffected by this
/// change; only genuine fractional leakage from many-frame stacks is
/// suppressed.
const DETAIL_WEIGHT_EXPONENT: u32 = 3;

/// Applies the deep-stack detail-weight fix: raises an already-`[0, 1]`-
/// clamped detail weight to `exponent` (see [`StrataParams::detail_focus`]
/// / [`DETAIL_WEIGHT_EXPONENT`]'s doc comment), via repeated multiplication
/// rather than `powf` — a cheap exact integer power, no transcendental call
/// per pixel per frame in this hot loop. `exponent` is expected to be in
/// `1..=5` (the settings-clamped range of `detail_focus`); values outside
/// that range still compute correctly (the `_ =>` arm below falls back to
/// `powi`), they just fall outside the range the settings layer permits.
#[inline]
fn raise_detail_weight(w: f32, exponent: u32) -> f32 {
    match exponent {
        1 => w,
        2 => w * w,
        3 => w * w * w,
        4 => w * w * w * w,
        5 => w * w * w * w * w,
        _ => w.powi(exponent.cast_signed()),
    }
}

/// Worst-case `base_radius` accepted by
/// `stacker_core::settings::StackingSettings::clamp_valid` (`8..=64`).
/// Duplicated here as a literal — `stacker-core` cannot depend on
/// `stacker-algo` — purely so the static assertion below stays load-bearing;
/// update it if that clamp range ever changes.
const MAX_BASE_RADIUS_SETTING: usize = 64;

// Strata's widest tile-boundary neighbourhood read chains Step 1's box
// filter (`base_radius`, worst case 64) with Step 4's `r_big` guided filter
// (45): worst case 109 px, safely inside the shared `APRON_PX` (128) every
// tiled fetch already pads with (`stacker_core::memory`). If this assertion
// ever fires, either `APRON_PX` or the settings clamp changed without
// updating the other.
const _: () = assert!(
    stacker_core::memory::APRON_PX >= R_BIG + MAX_BASE_RADIUS_SETTING,
    "Strata's worst-case neighbourhood (r_big + max base_radius) exceeds APRON_PX"
);

// ── Normalisation fallback (Step 5) ──────────────────────────────────────

/// Resolve one pixel's fused value for one layer (base or detail): the
/// weighted mean `acc / norm`, or a plain `sum / n` fallback when `norm`
/// underflows [`NORM_EPS`] (design doc §2 Step 5 — "happens only where
/// every frame is featureless", e.g. a flat wall). The fallback avoids a
/// division blow-up / NaN rather than requiring callers to special-case it.
#[inline]
fn resolve_layer(acc: f32, norm: f32, sum: f32, n: f32) -> f32 {
    if norm < NORM_EPS { sum / n } else { acc / norm }
}

// ── Snapshot (shared by the final fuse and live previews) ───────────────

/// Materialise one fused `PlanarImage` from the current running
/// accumulator/norm/sum state — the same per-pixel `resolve_layer` combine
/// [`fuse_strata_with_preview`] uses for its final output, factored out so
/// a live preview snapshot mid-Pass-2 and the final result are pixel-for-
/// pixel the same computation. Parallelised across pixels (one rayon pass).
///
/// No explicit clamp on the fused planar output here: `Apex`'s whole-image
/// accumulator (`apex::fuse::ApexAccumulator::reconstruct`) does not clamp
/// its planar output either — the only clamp in the pipeline happens at
/// final RGB reconstruction (`stacker_pipeline::output::
/// planar_to_gamma_rgb`), which every fusion mode's output passes through
/// identically. Matching that precedent keeps Strata consistent with Apex
/// rather than introducing a clamp Apex itself doesn't have.
#[allow(clippy::too_many_arguments)]
fn snapshot_layer(
    width: usize,
    height: usize,
    n_f: f32,
    acc_base: (&[f32], &[f32], &[f32]),
    acc_detail: (&[f32], &[f32], &[f32]),
    norm: (&[f32], &[f32]),
    sum_base: (&[f32], &[f32], &[f32]),
    sum_detail: (&[f32], &[f32], &[f32]),
) -> PlanarImage<f32> {
    let (acc_base_l, acc_base_a, acc_base_b) = acc_base;
    let (acc_detail_l, acc_detail_a, acc_detail_b) = acc_detail;
    let (norm_base, norm_detail) = norm;
    let (sum_base_l, sum_base_a, sum_base_b) = sum_base;
    let (sum_detail_l, sum_detail_a, sum_detail_b) = sum_detail;

    let mut out = PlanarImage::new(width, height);
    out.luma
        .par_iter_mut()
        .zip(out.chroma_a.par_iter_mut())
        .zip(out.chroma_b.par_iter_mut())
        .enumerate()
        .for_each(|(p, ((ol, oa), ob))| {
            *ol = resolve_layer(acc_base_l[p], norm_base[p], sum_base_l[p], n_f)
                + resolve_layer(acc_detail_l[p], norm_detail[p], sum_detail_l[p], n_f);
            *oa = resolve_layer(acc_base_a[p], norm_base[p], sum_base_a[p], n_f)
                + resolve_layer(acc_detail_a[p], norm_detail[p], sum_detail_a[p], n_f);
            *ob = resolve_layer(acc_base_b[p], norm_base[p], sum_base_b[p], n_f)
                + resolve_layer(acc_detail_b[p], norm_detail[p], sum_detail_b[p], n_f);
        });
    out
}

// ── Public entry points ──────────────────────────────────────────────────

/// Fuse `frames` with Strata (see module docs). Convenience wrapper around
/// [`fuse_strata_with_progress`] with a no-op progress callback.
///
/// # Panics
/// Panics if `frames` is empty, or if any frame's dimensions differ from
/// `frames[0]`'s.
pub fn fuse_strata(frames: &[PlanarImage<f32>], params: &StrataParams) -> PlanarImage<f32> {
    fuse_strata_with_progress(frames, params, |_| {})
}

/// Fuse `frames` with Strata, invoking a progress callback along the way.
///
/// `on_tick(completed)` is called once per frame per pass — `2 *
/// frames.len()` ticks total: `1..=N` during the saliency pass, `N+1..=2N`
/// during the accumulation pass — for progress bars.
///
/// Thin delegation to [`fuse_strata_with_preview`] with previews disabled
/// (`preview_every = usize::MAX`, a no-op `on_preview`) so there is exactly
/// one implementation body for both entry points.
///
/// # Memory
///
/// Two streaming passes over `frames` (Pass 1: saliency then a running
/// argmax; Pass 2: decompose, refine weights, accumulate) hold at most
/// **two** frames' base/detail/`p_image` buffers at a time — see the
/// "CPU/GPU overlap" note below for why this is two, not the design doc's
/// original one. Six of the accumulator buffers below (an unweighted
/// running sum of the base/detail layers per plane) are not in the design
/// doc's literal buffer count; they exist purely so the [`NORM_EPS`]
/// normalisation-underflow fallback (§2 Step 5) never needs a third pass
/// over the frame stack. All of it remains proportional to image size,
/// independent of the frame count — the property the design's memory
/// discipline is actually protecting, and what lets this drop into the
/// tiled pipeline unchanged.
///
/// # CPU/GPU overlap across frames (post-launch optimisation)
///
/// Pass 2's per-frame body has two phases with very different resource
/// profiles: CPU-only prep (Step 1's base/detail box-filter split, plus the
/// `p_image` materialisation from the Pass-1 argmax) and a GPU-bound weight
/// refinement ([`guided_filter_pair`]'s fused dispatch). Run strictly
/// serially, per-frame GPU utilization sawtooths: CPU prep with the GPU
/// idle, then GPU dispatch with most CPU cores idle. Pass 2 now computes
/// frame `i+1`'s CPU prep on a `rayon::scope` task WHILE frame `i`'s
/// `guided_filter_pair` call is in flight on the main task, so the CPU prep
/// for the next frame overlaps the GPU-bound (or CPU-SAT-bound, in a
/// non-`gpu` build) work for the current one. This raises Pass 2's peak
/// resident frame count from one to two (frame `i`'s buffers, still needed
/// for this iteration's accumulation, plus frame `i+1`'s, being prepared
/// ahead) — still `O(1)` in the frame count `N`, not `O(N)`, so the tiled
/// pipeline's out-of-core memory contract is unaffected; only the small
/// constant changed. Frames are too large in practice to consider batching
/// more than one frame ahead or keeping multiple frames GPU-resident at
/// once (out of scope — see the module's task history for the explicit
/// "do not attempt cross-frame batching or multi-frame GPU residency"
/// boundary).
///
/// # Panics
/// Panics if `frames` is empty, or if any frame's dimensions differ from
/// `frames[0]`'s.
pub fn fuse_strata_with_progress<F>(
    frames: &[PlanarImage<f32>],
    params: &StrataParams,
    mut on_tick: F,
) -> PlanarImage<f32>
where
    F: FnMut(usize),
{
    fuse_strata_with_preview(frames, params, &mut on_tick, PREVIEW_DISABLED, |_| {})
}

/// Sentinel `preview_every` value [`fuse_strata_with_progress`] passes to
/// mean "previews disabled". [`fuse_strata_with_preview`] special-cases
/// exactly this value (`usize::MAX`, distinct from every real cadence and
/// from `0`, which that function treats as "preview every frame") to skip
/// every mid-stack preview *and* the otherwise-mandatory final-frame
/// preview, so the no-preview path never pays for the extra
/// `snapshot_layer` pass.
const PREVIEW_DISABLED: usize = usize::MAX;

/// Fuse `frames` with Strata, invoking both a per-frame progress callback
/// and a periodic live-preview callback during Pass 2.
///
/// `on_tick(completed)` behaves exactly as in [`fuse_strata_with_progress`]
/// (`2 * frames.len()` ticks total). `on_preview` is called during Pass 2
/// — never Pass 1, where there is no partial composite yet to show — at
/// most once every `preview_every` frames, and always once more after the
/// final frame regardless of where that falls relative to the cadence, so
/// callers always see a preview reflecting the finished stack. Passing
/// `preview_every = 0` is treated the same as `1` (a preview after every
/// frame) rather than dividing by zero.
///
/// Each preview snapshot is a full materialised `PlanarImage` — the same
/// `resolve_layer` normalisation the final output goes through, applied to
/// the accumulators as they stand after the frames folded in so far (see
/// [`snapshot_layer`]). This is for the GUI's live preview (wired by a
/// follow-up task — this function only provides the API); a snapshot costs
/// one extra parallel normalise pass over the whole image, which is
/// negligible next to a frame's guided-filter cost at the default cadence,
/// but callers driving this from a UI thread should choose `preview_every`
/// (and how the returned image is consumed) with their own responsiveness
/// budget in mind — this function does not throttle by wall-clock time,
/// only by frame count.
///
/// # Memory
/// See [`fuse_strata_with_progress`]'s "Memory"/"CPU/GPU overlap" sections
/// — unchanged here (this function's own Pass 2 body is the same
/// two-frame-resident, overlapped-prep loop); a preview snapshot is a
/// temporary extra output-sized buffer, freed before the next frame's
/// Pass-2 work begins.
///
/// # Panics
/// Panics if `frames` is empty, or if any frame's dimensions differ from
/// `frames[0]`'s.
///
/// CPU-side prep for one frame (Step 1's base/detail split plus the
/// lazily-materialised `p_image` — see design doc §2 Steps 1 & 3): pure
/// CPU work with no GPU dispatch, independent of every other frame's
/// prep given only `frames`/`argmax_idx` (both read-only here).
///
/// Overlap note (CPU/GPU overlap across frames, item 3 of the
/// per-frame-utilization-smoothing work): factoring this out lets frame
/// `i+1`'s prep run on a second rayon task WHILE frame `i`'s
/// `guided_filter_pair` GPU dispatch is in flight on the main task (see
/// the `rayon::scope` below) — keeping at most two frames' base/detail/
/// `p_image` buffers resident at once (frame `i`'s, still needed for the
/// accumulation step below, plus frame `i+1`'s, being prepared ahead),
/// one more than the design's original strictly-one-frame-resident
/// streaming discipline. This still bounds memory independent of stack
/// depth (`N`) — only ever 2 frames' worth, never N — so the tiled
/// pipeline's memory contract (`docs/strata-fusion-design.md` §2's
/// "Streaming form") is preserved; only the peak-memory *constant*
/// changed (1 frame -> 2), not its dependence on `N`.
#[allow(clippy::type_complexity)]
fn prep_frame(
    frame: &PlanarImage<f32>,
    argmax_idx: &[u16],
    width: usize,
    height: usize,
    base_radius: usize,
    i: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, PlanarImage<f32>) {
    let base_l = box_filter(&frame.luma, width, height, base_radius);
    let base_a = box_filter(&frame.chroma_a, width, height, base_radius);
    let base_b = box_filter(&frame.chroma_b, width, height, base_radius);

    // P_i materialised lazily from the Pass-1 argmax — never hold N
    // full P_i buffers at once (design doc §2 Step 3's memory note).
    // Rayonised: independent per pixel.
    let p_i: Vec<f32> = argmax_idx
        .par_iter()
        .map(|&idx| if idx as usize == i { 1.0 } else { 0.0 })
        .collect();
    let p_image = PlanarImage {
        width,
        height,
        luma: p_i,
        chroma_a: Vec::new(),
        chroma_b: Vec::new(),
    };
    (base_l, base_a, base_b, p_image)
}

pub fn fuse_strata_with_preview<F, P>(
    frames: &[PlanarImage<f32>],
    params: &StrataParams,
    mut on_tick: F,
    preview_every: usize,
    mut on_preview: P,
) -> PlanarImage<f32>
where
    F: FnMut(usize),
    P: FnMut(&PlanarImage<f32>),
{
    assert!(!frames.is_empty(), "cannot fuse an empty frame stack");
    let width = frames[0].width;
    let height = frames[0].height;
    let len = width * height;
    let n = frames.len();
    for f in frames {
        assert_eq!(f.width, width, "all frames must share one width");
        assert_eq!(f.height, height, "all frames must share one height");
    }
    let n_f = n as f32;
    let previews_enabled = preview_every != PREVIEW_DISABLED;
    let preview_every = preview_every.max(1);

    // ── Pass 1 (streaming): saliency -> running argmax ──────────────────
    //
    // Per-frame saliency computation is already internally parallel
    // (`compute_saliency`). The per-pixel running-max update below is
    // rayonised across pixels *within* one frame's update — never across
    // frames (the streaming memory discipline is deliberate: only one
    // frame's saliency buffer is ever resident — see the module doc's
    // "Memory" section on `fuse_strata_with_progress`).
    let mut argmax_idx = vec![0_u16; len];
    let mut argmax_val = vec![f32::NEG_INFINITY; len];
    for (i, frame) in frames.iter().enumerate() {
        let s = compute_saliency(&frame.luma, width, height);
        let idx = i as u16;
        argmax_idx
            .par_iter_mut()
            .zip(argmax_val.par_iter_mut())
            .zip(s.par_iter())
            .for_each(|((a_idx, a_val), &sv)| {
                if sv > *a_val {
                    *a_val = sv;
                    *a_idx = idx;
                }
            });
        on_tick(i + 1);
    }

    // ── Pass 2 (streaming): decompose, refine weights, accumulate ───────
    let mut acc_base_l = vec![0.0_f32; len];
    let mut acc_base_a = vec![0.0_f32; len];
    let mut acc_base_b = vec![0.0_f32; len];
    let mut acc_detail_l = vec![0.0_f32; len];
    let mut acc_detail_a = vec![0.0_f32; len];
    let mut acc_detail_b = vec![0.0_f32; len];
    let mut norm_base = vec![0.0_f32; len];
    let mut norm_detail = vec![0.0_f32; len];
    let mut sum_base_l = vec![0.0_f32; len];
    let mut sum_base_a = vec![0.0_f32; len];
    let mut sum_base_b = vec![0.0_f32; len];
    let mut sum_detail_l = vec![0.0_f32; len];
    let mut sum_detail_a = vec![0.0_f32; len];
    let mut sum_detail_b = vec![0.0_f32; len];

    // Prime the pipeline with frame 0's prep before the loop starts.
    let mut next_prep = Some(prep_frame(
        &frames[0],
        &argmax_idx,
        width,
        height,
        params.base_radius,
        0,
    ));

    for (i, frame) in frames.iter().enumerate() {
        // Take this iteration's already-prepared buffers (computed either
        // just above, priming the loop, or by the previous iteration's
        // overlapped `rayon::scope` below).
        let (base_l, base_a, base_b, p_image) = next_prep.take().expect(
            "next_prep is always populated by the previous iteration or the priming call above",
        );

        // Overlap frame `i+1`'s CPU prep with frame `i`'s GPU weight
        // refinement: `rayon::scope` runs the prep closure on a rayon
        // worker while the current thread calls the (blocking, GPU-bound)
        // `guided_filter_pair` — both complete before the scope returns,
        // but they run concurrently instead of serially, smoothing the
        // per-frame CPU/GPU utilization sawtooth two purely-sequential
        // phases would otherwise produce. The last frame has no successor
        // to prep, so its scope's prep closure is a no-op.
        let mut prepped_next = None;
        let (w_base_img, w_detail_img) = rayon::scope(|s| {
            if i + 1 < n {
                let argmax_idx_ref = &argmax_idx;
                let next_frame = &frames[i + 1];
                let base_radius = params.base_radius;
                let prepped_next_slot = &mut prepped_next;
                s.spawn(move |_| {
                    *prepped_next_slot = Some(prep_frame(
                        next_frame,
                        argmax_idx_ref,
                        width,
                        height,
                        base_radius,
                        i + 1,
                    ));
                });
            }
            // `guided_filter_pair` only ever reads `.luma` of both
            // `guidance` and `src` — the guide is this frame's own luma
            // (design doc §2 Step 4). Both scales share the same
            // `(frame, p_image)` pair, so computing them together (one GPU
            // upload / one set of shared CPU SATs instead of two
            // independent `guided_filter` calls) removes duplicated
            // per-frame work — see `guided_filter_pair`'s doc comment.
            guided_filter_pair(frame, &p_image, R_BIG, EPS_BIG, R_SMALL, EPS_SMALL)
        });
        next_prep = prepped_next;

        let mut w_base = w_base_img.luma;
        let mut w_detail = w_detail_img.luma;
        w_base.par_iter_mut().for_each(|v| {
            *v = v.clamp(0.0, 1.0);
        });
        w_detail.par_iter_mut().for_each(|v| {
            *v = v.clamp(0.0, 1.0);
            // Deep-stack detail-weight re-concentration ("Detail focus" —
            // see `DETAIL_WEIGHT_EXPONENT`'s doc comment): raise the
            // clamped detail weight to `params.detail_focus`, collapsing
            // leaked non-winner weight while leaving winners (w ~= 1) and
            // true zeros untouched.
            *v = raise_detail_weight(*v, params.detail_focus);
        });

        // Big 14-buffer accumulation loop, rayonised across pixels as two
        // independent per-buffer-group parallel passes (base-layer
        // buffers, detail-layer buffers) rather than one deeply-nested
        // 14-way zip — same per-pixel arithmetic, same order
        // (`acc[p] += w * value`), only the pixel loop's iteration is now
        // parallel, so results are unchanged; readability wins over
        // collapsing both groups into a single zip chain.
        acc_base_l
            .par_iter_mut()
            .zip(acc_base_a.par_iter_mut())
            .zip(acc_base_b.par_iter_mut())
            .zip(norm_base.par_iter_mut())
            .zip(sum_base_l.par_iter_mut())
            .zip(sum_base_a.par_iter_mut())
            .zip(sum_base_b.par_iter_mut())
            .enumerate()
            .for_each(|(p, ((((((abl, aba), abb), nb), sbl), sba), sbb))| {
                let wb = w_base[p];
                let bl = base_l[p];
                let ba = base_a[p];
                let bb = base_b[p];

                *abl += wb * bl;
                *aba += wb * ba;
                *abb += wb * bb;
                *nb += wb;
                *sbl += bl;
                *sba += ba;
                *sbb += bb;
            });

        acc_detail_l
            .par_iter_mut()
            .zip(acc_detail_a.par_iter_mut())
            .zip(acc_detail_b.par_iter_mut())
            .zip(norm_detail.par_iter_mut())
            .zip(sum_detail_l.par_iter_mut())
            .zip(sum_detail_a.par_iter_mut())
            .zip(sum_detail_b.par_iter_mut())
            .enumerate()
            .for_each(|(p, ((((((adl, ada), adb), nd), sdl), sda), sdb))| {
                let wd = w_detail[p];
                let dl = frame.luma[p] - base_l[p];
                let da = frame.chroma_a[p] - base_a[p];
                let db = frame.chroma_b[p] - base_b[p];

                *adl += wd * dl;
                *ada += wd * da;
                *adb += wd * db;
                *nd += wd;
                *sdl += dl;
                *sda += da;
                *sdb += db;
            });

        on_tick(n + i + 1);

        let is_last_frame = i + 1 == n;
        if previews_enabled && (is_last_frame || (i + 1) % preview_every == 0) {
            let snapshot = snapshot_layer(
                width,
                height,
                n_f,
                (&acc_base_l, &acc_base_a, &acc_base_b),
                (&acc_detail_l, &acc_detail_a, &acc_detail_b),
                (&norm_base, &norm_detail),
                (&sum_base_l, &sum_base_a, &sum_base_b),
                (&sum_detail_l, &sum_detail_a, &sum_detail_b),
            );
            on_preview(&snapshot);
        }
    }

    // ── Normalise + fuse (design doc §2 Step 5) ──────────────────────────
    snapshot_layer(
        width,
        height,
        n_f,
        (&acc_base_l, &acc_base_a, &acc_base_b),
        (&acc_detail_l, &acc_detail_a, &acc_detail_b),
        (&norm_base, &norm_detail),
        (&sum_base_l, &sum_base_a, &sum_base_b),
        (&sum_detail_l, &sum_detail_a, &sum_detail_b),
    )
}
