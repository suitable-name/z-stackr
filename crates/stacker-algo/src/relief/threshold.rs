//! Contrast threshold for the `Relief` focus measure.
//!
//! Covers percentile selection, robust auto-detection (median + 1.5 · MAD,
//! [`auto_detect_threshold`]) and the resulting in-focus mask.
//!
//! Auto-detection used to be Otsu's method ([`otsu_threshold`], still available).
//! A focus map is heavy-tailed (a huge noise floor near zero and a long tail of
//! strong edges), which pulls Otsu's split to the top of the range and rejects
//! almost everything; the median + MAD rule finds the noise floor instead.
//! The threshold is a property of the *whole image*.  Callers that process the
//! frame in tiles must compute it once with [`compute_relief_threshold`] on the
//! whole-image map and then apply the resulting absolute
//! [`ReliefThreshold::value`] to every tile with [`mask_above`]; calling
//! [`generate_mask`] per tile would give every tile its own threshold and make
//! the mask jump at the tile borders.

#![expect(
    clippy::cast_precision_loss,
    reason = "histogram counts and pixel counts are far below 2^52, so the u64/usize to f64 conversions are exact"
)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the histogram bin and the order-statistic index are non-negative and clamped into range; the Otsu fraction is narrowed to f32 on purpose"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "the Otsu class sums are evaluated as plain multiply-add"
)]

use stacker_core::image::PlanarImage;

/// Number of histogram bins Otsu's method uses.
pub const OTSU_BINS: usize = 256;

/// The one `Relief` option [`generate_mask`] reads.
///
/// The full option set lives in `stacker_core::settings::StackingSettings`; this
/// struct carries just the part the stand-alone mask helper needs.
pub struct ReliefSettings {
    /// Fraction of pixels (0.0 ..= 1.0) that the percentile rule rejects.
    pub contrast_pct: f32,
}

/// An absolute threshold on a focus map together with the fraction it came
/// from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReliefThreshold {
    /// The rejected fraction actually used: the caller's `contrast_pct`, or
    /// the [`auto_detect_threshold`] fraction when auto-detection was requested.
    pub contrast_pct: f32,
    /// Pixels with `value >= threshold` are in focus.
    pub value: f32,
}

/// How many pixels [`auto_detect_threshold`] looks at at most.
pub const AUTO_DETECT_SAMPLES: usize = 100_000;

/// The multiple of the median absolute deviation that [`auto_detect_threshold`]
/// adds to the median to get the noise-floor limit.
pub const AUTO_DETECT_MAD_FACTOR: f32 = 1.5;

/// A median absolute deviation at or below this fraction of the median counts as
/// "no spread" in [`auto_detect_threshold`].
///
/// The guard is relative on purpose: a focus map is a residual *variance*, so its
/// scale follows the square of the pixel scale (and differs between gamma and
/// linear data by orders of magnitude); an absolute floor would misfire on a
/// dark or low-contrast stack.
const AUTO_DETECT_MIN_RELATIVE_MAD: f32 = 1e-6;

/// The samples [`auto_detect_threshold`] works on: all of `luma` up to
/// [`AUTO_DETECT_SAMPLES`] pixels, otherwise one pixel from every block of
/// `len / AUTO_DETECT_SAMPLES` consecutive pixels.
///
/// The pixel inside a block is chosen by a fixed hash of the block number, not at a
/// constant offset: a constant stride that divides the image width would look at a
/// few columns only. The choice is deterministic.
fn auto_detect_sample(luma: &[f32]) -> Vec<f32> {
    let step = (luma.len() / AUTO_DETECT_SAMPLES).max(1);
    if step == 1 {
        return luma.to_vec();
    }
    (0..luma.len() / step)
        .map(|block| {
            let hash = (block as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32;
            luma[block * step + hash as usize % step]
        })
        .collect()
}

/// The rejected fraction (`0.0..=1.0`) that auto-detection picks for the focus map
/// `luma`.
///
/// It is the fraction of pixels at or below `median + 1.5 · MAD`, where the
/// median and the median absolute deviation (MAD) are those of the **non-zero**
/// pixels.
///
/// The returned fraction counts *all* sampled pixels, zeros included, because it is
/// fed to [`percentile_threshold`], which ranks all pixels. It is therefore the
/// same kind of number as the manual `contrast_pct`, and the resulting threshold
/// value sits at the noise floor `median + 1.5 · MAD` of the non-zero values.
///
/// At most [`AUTO_DETECT_SAMPLES`] pixels are examined (see [`AUTO_DETECT_MAD_FACTOR`]
/// for the factor), picked deterministically, so the cost does not grow with the
/// image. The result is `0.0` (reject nothing) for an empty map, a map without
/// positive values and a map without spread (the MAD at or below `1e-6` of the
/// median).
#[must_use]
pub fn auto_detect_threshold(luma: &[f32]) -> f32 {
    let sample = auto_detect_sample(luma);
    let mut positive: Vec<f32> = sample.iter().copied().filter(|&v| v > 0.0).collect();
    if positive.is_empty() {
        return 0.0;
    }
    let mid = positive.len() / 2;
    let median = *positive.select_nth_unstable_by(mid, f32::total_cmp).1;
    let mut deviations: Vec<f32> = positive.iter().map(|&v| (v - median).abs()).collect();
    let mad = *deviations.select_nth_unstable_by(mid, f32::total_cmp).1;
    if mad <= median * AUTO_DETECT_MIN_RELATIVE_MAD {
        return 0.0;
    }

    let limit = median + AUTO_DETECT_MAD_FACTOR * mad;
    let at_or_below = sample.iter().filter(|&&v| v <= limit).count();
    (at_or_below as f32 / sample.len() as f32).clamp(0.0, 1.0)
}

/// Fraction of pixels in the Otsu background class of `luma` (the pixels at or
/// below the Otsu split), computed on a 256-bin histogram.
///
/// No longer used for auto-detection (see [`auto_detect_threshold`]): the split of
/// a heavy-tailed focus map lands near the top of the range.
///
/// The background class is histogram bins `0..=t` where `t` maximises the
/// between-class variance, and the returned fraction counts exactly those
/// bins (the scoring and the conversion use the same class definition).
/// Counts are `u64` and the sums `f64`, so frames above 2^24 pixels are exact.
/// A map without any spread (or an empty one) returns `0.0`.
#[must_use]
pub fn otsu_threshold(luma: &[f32]) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in luma {
        min = min.min(v);
        max = max.max(v);
    }
    if luma.is_empty() || max <= min {
        return 0.0;
    }

    let span = f64::from(max) - f64::from(min);
    let top = (OTSU_BINS - 1) as f64;
    let mut hist = [0_u64; OTSU_BINS];
    for &v in luma {
        let bin = ((f64::from(v) - f64::from(min)) / span * top) as usize;
        hist[bin.min(OTSU_BINS - 1)] += 1;
    }

    let total = luma.len() as u64;
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();

    let mut sum_b = 0.0_f64;
    let mut w_b = 0_u64;
    let mut var_max = 0.0_f64;
    let mut threshold_bin = 0;
    for (i, &count) in hist.iter().enumerate() {
        w_b += count;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += i as f64 * count as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum - sum_b) / w_f as f64;
        let var_between = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if var_between > var_max {
            var_max = var_between;
            threshold_bin = i;
        }
    }

    let background: u64 = hist[..=threshold_bin].iter().sum();
    ((background as f64 / total as f64) as f32).clamp(0.0, 1.0)
}

/// The absolute threshold that rejects the lowest `contrast_pct` fraction of
/// `luma`, with ties resolved towards the requested kept fraction.
///
/// The base value is the `round(len * contrast_pct)`-th order statistic `t`,
/// found in O(n).  `v >= t` keeps every pixel tied with `t`, which on a map
/// with a large plateau (flat backgrounds are exactly 0) can keep far more than
/// `1 - contrast_pct`, even everything.  So the count of pixels with `v >= t`
/// and the count with `v > t` are both compared with the wanted count
/// `len - k`, and the closer one wins (the inclusive rule on a tie).  With
/// `contrast_pct == 0` the threshold is the minimum and every pixel is kept.
///
/// The returned value is always applied as `v >= value`.  An empty map
/// returns `0.0`.
pub fn percentile_threshold(luma: &[f32], contrast_pct: f32) -> f32 {
    let len = luma.len();
    if len == 0 {
        return 0.0;
    }
    let pct = f64::from(contrast_pct).clamp(0.0, 1.0);
    // Rounded, so that an `f32` fraction such as 0.9 (really 0.89999998...)
    // still selects index 900 of 1000.
    let k = ((len as f64 * pct).round() as usize).min(len - 1);

    // Only the k-th order statistic is needed, so partition in O(n) rather
    // than a full O(n log n) sort.
    let mut buf = luma.to_vec();
    let (_, kth, _) = buf.select_nth_unstable_by(k, f32::total_cmp);
    let t = *kth;

    let wanted = len - k;
    let at_least = luma.iter().filter(|&&v| v >= t).count();
    let above = luma.iter().filter(|&&v| v > t).count();
    if above.abs_diff(wanted) < at_least.abs_diff(wanted) {
        t.next_up()
    } else {
        t
    }
}

/// Compute the threshold for a whole-image focus map: either from the
/// caller's `contrast_pct` or, with `auto_detect`, from [`auto_detect_threshold`]
/// (median + 1.5 · MAD).
///
/// Call this once per image and reuse [`ReliefThreshold::value`] for every tile.
#[must_use]
pub fn compute_relief_threshold(
    luma: &[f32],
    contrast_pct: f32,
    auto_detect: bool,
) -> ReliefThreshold {
    let contrast_pct = if auto_detect {
        auto_detect_threshold(luma)
    } else {
        contrast_pct
    };
    ReliefThreshold {
        contrast_pct,
        value: percentile_threshold(luma, contrast_pct),
    }
}

/// In-focus mask of `luma` for an absolute threshold: `value >= threshold`.
#[must_use]
pub fn mask_above(luma: &[f32], threshold: f32) -> Vec<bool> {
    luma.iter().map(|&v| v >= threshold).collect()
}

/// Mask for one focus map with its *own* percentile threshold.  Thin wrapper
/// over [`compute_relief_threshold`] and [`mask_above`]; use those directly when
/// the image is processed in tiles.
#[must_use]
pub fn generate_mask(focus_measure: &PlanarImage<f32>, settings: &ReliefSettings) -> Vec<bool> {
    let threshold = compute_relief_threshold(&focus_measure.luma, settings.contrast_pct, false);
    mask_above(&focus_measure.luma, threshold.value)
}
