//! Per-frame brightness correction against the reference frame.
//!
//! Between two shots of a stack the exposure can drift (flicker, a changing
//! aperture, auto-exposure, a light source that warms up). The intensity
//! refinement in [`crate::refine`] is made exposure-invariant for *finding* the
//! transform, but the pixels still differ, which shows up as brightness steps in
//! a fused result. This module matches each aligned frame's luma statistics to
//! the reference's.
//!
//! # Model
//!
//! Every frame is stored in **linear light** (see [`stacker_core::color`]), where
//! an exposure change is a pure multiplication. The original z-stackr
//! implementation worked on gamma-encoded luma and fitted `y' = (s * y)^g`
//! there. That model cannot be copied over: a power law with an exponent up to 5
//! on linear data is a different (and, for exposure drift, wrong) family. This
//! port keeps the model family but moves it to the domain where the drift is
//! simple:
//!
//! * the correction is `y' = (scale * y)^gamma` on **linear** luma, so a pure
//!   exposure change is recovered exactly with `gamma = 1`;
//! * `gamma` is restricted to `[0.5, 2.0]` (the original allowed `[0, 5]`) so
//!   the fit can only absorb a mild tone-curve drift and never invent contrast;
//! * the **histogram is built over the sRGB-encoded luma** (4096 bins over
//!   `[0, 2]`) so the bins are fine in the shadows, where a linear histogram
//!   would put whole tonal ranges into one bin, while every bin keeps the *exact
//!   mean linear value* of its pixels and all statistics are evaluated on those
//!   linear means;
//! * the fit matches the mean and the standard deviation of the linear luma of
//!   the frame to those of the reference (relative errors, so dark and bright
//!   stacks are weighted alike) with a deterministic two-parameter Nelder-Mead
//!   search that starts at the identity;
//! * the chroma planes are multiplied by the same per-pixel ratio `y' / y`, which
//!   keeps the colour of a pixel when its linear RGB is scaled uniformly.
//!
//! The correction is skipped (the frame is left bit-for-bit unchanged) when the
//! statistics already agree, when the reference has no usable pixel, or when the
//! fit fails to improve on the identity.
//!
//! # Where it runs
//!
//! The correction is meant for **warped** frames. A warp leaves a border of
//! out-of-frame pixels; pass the frame's coverage mask
//! ([`correct_brightness_masked`]) so they do not enter the statistics. As a
//! fallback for callers without a mask, pixels with a luma of exactly zero (or
//! not finite) are never counted either, which is what a zero-filled border
//! looks like, and zero is a fixed point of the correction.

use rayon::prelude::*;
use stacker_core::{color::linear_to_srgb, fma::fma_f64, image::PlanarImage};

/// Number of histogram bins over the encoded luma range.
const NUM_BINS: usize = 4096;
/// [`NUM_BINS`] as a float (kept literal so no cast is needed).
const NUM_BINS_F32: f32 = 4096.0;
/// Encoded luma covered by the histogram: `[0, ENCODED_RANGE]`. Linear light
/// above 1 (over-range) is still binned up to an encoded value of 2 (linear
/// ~4.9) and clamped into the last bin beyond that.
const ENCODED_RANGE: f32 = 2.0;

/// Centre of the allowed gamma interval.
const GAMMA_CENTRE: f64 = 1.25;
/// Half width of the allowed gamma interval: gamma is in `[0.5, 2.0]`.
const GAMMA_HALF_RANGE: f64 = 0.75;
/// Smallest and largest scale the fit may return; a fit outside is rejected as
/// a failed match rather than applied.
const SCALE_LIMITS: (f64, f64) = (0.05, 20.0);

/// Nelder-Mead iteration cap of the fit.
const FIT_MAX_ITER: usize = 400;
/// Nelder-Mead stops when the objective spread of the simplex is below this.
const FIT_TOLERANCE: f64 = 1.0e-24;
/// Initial simplex step in the unconstrained fit parameters.
const FIT_STEP: f64 = 0.1;
/// An objective below this at the identity means the statistics already agree.
const IDENTITY_OBJECTIVE: f64 = 1.0e-20;

/// The statistics of the reference frame a [`correct_brightness`] call matches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrightnessTarget {
    /// Mean linear luma over the counted pixels.
    mean: f64,
    /// Sample standard deviation of the linear luma over the counted pixels.
    std_dev: f64,
    /// `false` when the reference had no usable pixel (all black, all masked
    /// out or not finite); corrections against such a target are skipped.
    valid: bool,
}

impl BrightnessTarget {
    /// The statistics of `reference`: pixels with a finite luma above zero.
    ///
    /// Use the reference as it is aligned against (the unwarped first frame or
    /// the rolling reference).
    #[must_use]
    pub fn from_reference(reference: &PlanarImage<f32>) -> Self {
        Self::from_reference_masked(reference, None)
    }

    /// Like [`Self::from_reference`], counting only pixels whose entry in
    /// `coverage` is `true`. A mask whose length differs from the pixel count is
    /// ignored.
    #[must_use]
    pub fn from_reference_masked(reference: &PlanarImage<f32>, coverage: Option<&[bool]>) -> Self {
        let bins = collect_bins(reference, coverage);
        if bins.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                valid: false,
            };
        }
        let (mean, std_dev) = moments(&bins, 1.0, 1.0);
        Self {
            mean,
            std_dev,
            valid: mean.is_finite() && std_dev.is_finite() && mean > 0.0,
        }
    }

    /// Mean linear luma of the reference.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Standard deviation of the linear luma of the reference.
    #[must_use]
    pub const fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// `true` when the reference had pixels to take statistics from.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.valid
    }
}

/// The fitted correction `y' = (scale * y)^gamma` of one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrightnessFit {
    /// Linear gain applied before the exponent.
    pub scale: f32,
    /// Exponent, in `[0.5, 2.0]`.
    pub gamma: f32,
}

impl BrightnessFit {
    /// The no-op correction.
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        gamma: 1.0,
    };

    /// `true` for [`Self::IDENTITY`] (nothing was changed).
    #[must_use]
    pub const fn is_identity(&self) -> bool {
        self.scale.to_bits() == 1.0_f32.to_bits() && self.gamma.to_bits() == 1.0_f32.to_bits()
    }
}

/// One histogram bin: how many pixels it holds and their mean linear luma.
#[derive(Clone, Copy)]
struct Bin {
    count: f64,
    mean: f64,
}

/// The histogram bin of a positive, finite linear luma value.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the encoded value is non-negative and clamped to the bin range before the cast"
)]
fn bin_of(luma: f32) -> usize {
    let encoded = linear_to_srgb(luma);
    let position = encoded / ENCODED_RANGE * NUM_BINS_F32;
    if position >= NUM_BINS_F32 {
        NUM_BINS - 1
    } else {
        position as usize
    }
}

/// Histogram of the counted pixels of `img`, as the non-empty bins in order.
///
/// A pixel counts when its luma is finite and above zero and (with a mask of
/// the right length) its coverage entry is `true`.
fn collect_bins(img: &PlanarImage<f32>, coverage: Option<&[bool]>) -> Vec<Bin> {
    let mask = coverage.filter(|mask| mask.len() == img.luma.len());
    let mut counts = vec![0_u32; NUM_BINS];
    let mut sums = vec![0.0_f64; NUM_BINS];
    for (index, &luma) in img.luma.iter().enumerate() {
        if !luma.is_finite() || luma <= 0.0 {
            continue;
        }
        if mask.is_some_and(|mask| !mask[index]) {
            continue;
        }
        let bin = bin_of(luma);
        counts[bin] += 1;
        sums[bin] += f64::from(luma);
    }
    counts
        .iter()
        .zip(&sums)
        .filter(|&(&count, _)| count > 0)
        .map(|(&count, &sum)| {
            let count = f64::from(count);
            Bin {
                count,
                mean: sum / count,
            }
        })
        .collect()
}

/// Mean and sample standard deviation of `(scale * v)^gamma` over the binned
/// pixels (every pixel of a bin is represented by the bin's mean linear value).
fn moments(bins: &[Bin], scale: f64, gamma: f64) -> (f64, f64) {
    let (mut total, mut sum, mut sum_sq) = (0.0_f64, 0.0_f64, 0.0_f64);
    for bin in bins {
        let value = (bin.mean * scale).powf(gamma);
        total += bin.count;
        sum = fma_f64(bin.count, value, sum);
        sum_sq = fma_f64(bin.count * value, value, sum_sq);
    }
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    let mean = sum / total;
    let variance = (sum_sq - sum * sum / total) / (total - 1.0).max(1.0);
    (mean, variance.max(0.0).sqrt())
}

/// Scale and gamma of the unconstrained fit parameters `[q, t]`:
/// `scale = q^2` (positive), `gamma = 1.25 + 0.75 sin t` (in `[0.5, 2.0]`).
fn parameters(params: [f64; 2]) -> (f64, f64) {
    (
        params[0] * params[0],
        GAMMA_HALF_RANGE.mul_add(params[1].sin(), GAMMA_CENTRE),
    )
}

/// Minimise a two-variable function with Nelder-Mead from `start`.
///
/// Deterministic: fixed coefficients, `total_cmp` ordering, no randomness.
fn nelder_mead(
    objective: &impl Fn([f64; 2]) -> f64,
    start: [f64; 2],
    step: f64,
    max_iter: usize,
    tolerance: f64,
) -> [f64; 2] {
    let second = [start[0] + step, start[1]];
    let third = [start[0], start[1] + step];
    let mut simplex = [
        (start, objective(start)),
        (second, objective(second)),
        (third, objective(third)),
    ];
    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        if simplex[2].1 - simplex[0].1 <= tolerance {
            break;
        }
        let centroid = [
            f64::midpoint(simplex[0].0[0], simplex[1].0[0]),
            f64::midpoint(simplex[0].0[1], simplex[1].0[1]),
        ];
        let worst = simplex[2].0;
        // The point `centroid + k * (centroid - worst)`.
        let along = |k: f64| {
            [
                k.mul_add(centroid[0] - worst[0], centroid[0]),
                k.mul_add(centroid[1] - worst[1], centroid[1]),
            ]
        };
        let reflected = along(1.0);
        let reflected_value = objective(reflected);
        if reflected_value < simplex[0].1 {
            let expanded = along(2.0);
            let expanded_value = objective(expanded);
            simplex[2] = if expanded_value < reflected_value {
                (expanded, expanded_value)
            } else {
                (reflected, reflected_value)
            };
        } else if reflected_value < simplex[1].1 {
            simplex[2] = (reflected, reflected_value);
        } else {
            let contracted = if reflected_value < simplex[2].1 {
                along(0.5)
            } else {
                along(-0.5)
            };
            let contracted_value = objective(contracted);
            if contracted_value < reflected_value.min(simplex[2].1) {
                simplex[2] = (contracted, contracted_value);
            } else {
                let best = simplex[0].0;
                for vertex in &mut simplex[1..] {
                    let moved = [
                        0.5_f64.mul_add(vertex.0[0] - best[0], best[0]),
                        0.5_f64.mul_add(vertex.0[1] - best[1], best[1]),
                    ];
                    *vertex = (moved, objective(moved));
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex[0].0
}

/// Fit the correction of a frame's `bins` to `target`; `None` when no
/// acceptable correction better than the identity exists.
fn fit(bins: &[Bin], target: &BrightnessTarget) -> Option<(f64, f64)> {
    let mean_scale = target.mean.max(1.0e-9);
    let std_scale = target.std_dev.max(0.01 * target.mean).max(1.0e-9);
    let objective = |params: [f64; 2]| {
        let (scale, gamma) = parameters(params);
        let (mean, std_dev) = moments(bins, scale, gamma);
        let d_mean = (mean - target.mean) / mean_scale;
        let d_std = (std_dev - target.std_dev) / std_scale;
        d_mean.mul_add(d_mean, d_std * d_std)
    };
    let start = [1.0, (-1.0_f64 / 3.0).asin()]; // scale 1, gamma 1
    let start_value = objective(start);
    if !start_value.is_finite() || start_value < IDENTITY_OBJECTIVE {
        return None;
    }
    let best = nelder_mead(&objective, start, FIT_STEP, FIT_MAX_ITER, FIT_TOLERANCE);
    let best_value = objective(best);
    let (scale, gamma) = parameters(best);
    let acceptable = best_value.is_finite()
        && best_value < start_value
        && scale.is_finite()
        && gamma.is_finite()
        && (SCALE_LIMITS.0..=SCALE_LIMITS.1).contains(&scale);
    acceptable.then_some((scale, gamma))
}

/// Match the brightness of `img` (a warped frame) to `target`, over every pixel
/// that is not zero. See the [module docs](self) for the model.
///
/// Returns the applied correction ([`BrightnessFit::IDENTITY`] when the frame
/// was left unchanged).
///
/// Non-finite and non-positive luma values are set to zero together with their
/// chroma (this also removes NaNs) -- but only when a correction is applied; a
/// skipped correction leaves the frame untouched.
pub fn correct_brightness(img: &mut PlanarImage<f32>, target: &BrightnessTarget) -> BrightnessFit {
    correct_brightness_masked(img, target, None)
}

/// Like [`correct_brightness`], taking the statistics only from pixels whose
/// entry in `coverage` is `true` (the frame's warp coverage mask, so the
/// out-of-frame border cannot bias the fit).
///
/// The correction itself is applied
/// to every pixel. A mask whose length differs from the pixel count is ignored.
pub fn correct_brightness_masked(
    img: &mut PlanarImage<f32>,
    target: &BrightnessTarget,
    coverage: Option<&[bool]>,
) -> BrightnessFit {
    if !target.valid {
        return BrightnessFit::IDENTITY;
    }
    let bins = collect_bins(img, coverage);
    if bins.is_empty() {
        return BrightnessFit::IDENTITY;
    }
    let Some((scale, gamma)) = fit(&bins, target) else {
        return BrightnessFit::IDENTITY;
    };
    apply(img, scale, gamma)
}

/// Apply `y' = (scale * y)^gamma` to the luma of `img` and scale its chroma by
/// the same per-pixel ratio.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the f64 fit is stored in the f32 planes"
)]
fn apply(img: &mut PlanarImage<f32>, scale: f64, gamma: f64) -> BrightnessFit {
    let scale = scale as f32;
    let gamma = gamma as f32;
    // Elementwise, so any split into chunks gives the identical result; the
    // chunks are processed in parallel (rayon runs them inline on wasm32).
    img.luma
        .par_chunks_mut(APPLY_CHUNK)
        .zip(img.chroma_a.par_chunks_mut(APPLY_CHUNK))
        .zip(img.chroma_b.par_chunks_mut(APPLY_CHUNK))
        .for_each(|((luma, chroma_a), chroma_b)| {
            apply_chunk(luma, chroma_a, chroma_b, scale, gamma);
        });
    BrightnessFit { scale, gamma }
}

/// Samples per parallel work item of [`apply`].
const APPLY_CHUNK: usize = 8192;

/// The serial per-pixel body of [`apply`] over equally long plane slices.
fn apply_chunk(
    luma: &mut [f32],
    chroma_a: &mut [f32],
    chroma_b: &mut [f32],
    scale: f32,
    gamma: f32,
) {
    for ((luma, chroma_a), chroma_b) in luma
        .iter_mut()
        .zip(chroma_a.iter_mut())
        .zip(chroma_b.iter_mut())
    {
        let y = *luma;
        if y.is_finite() && y > 0.0 {
            let out = (y * scale).powf(gamma);
            let ratio = if y > 1.0e-6 { out / y } else { 1.0 };
            *luma = out;
            *chroma_a *= ratio;
            *chroma_b *= ratio;
        } else {
            *luma = 0.0;
            *chroma_a = 0.0;
            *chroma_b = 0.0;
        }
    }
}
