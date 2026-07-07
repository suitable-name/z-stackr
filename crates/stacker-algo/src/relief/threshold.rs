use stacker_core::image::PlanarImage;

pub struct ReliefSettings {
    pub est_radius: usize,
    pub smooth_radius: usize,
    /// Fraction (`0.0..=1.0`) of the focus-measure population to exclude as
    /// "not sharp enough" before selecting the per-pixel source frame.
    ///
    /// `0.0` is documented to keep the invariant "every pixel passes" (an
    /// all-`true` mask) — see [`generate_mask`]'s doc comment. When `> 0.0`,
    /// the order statistic is computed over the **non-zero** population only
    /// (see [`generate_mask`]), so this is a percentile of the non-zero SML
    /// values, not of the whole image including its dense zero/near-zero
    /// plateau.
    ///
    /// Ignored when [`Self::absolute_threshold`] is `Some`.
    pub contrast_pct: f32,
    /// When set, use this **absolute focus-measure value** as the mask
    /// threshold (`v >= absolute_threshold`) instead of deriving a threshold
    /// from `contrast_pct`'s order statistic of this call's own data.
    ///
    /// This exists so a threshold VALUE resolved once from **global**
    /// (whole-image) statistics can be reused identically across every tile
    /// in the tiled `Relief` pipeline (see `stacker_pipeline::run_relief` and
    /// its per-tile-threshold-seam doc comment) — otherwise each tile's
    /// `generate_mask` call would derive its own tile-local percentile and
    /// produce a different threshold value per tile.
    ///
    /// `None` (the default/GUI in-RAM path) preserves the original
    /// tile-local / whole-image-local behaviour driven by `contrast_pct`.
    pub absolute_threshold: Option<f32>,
}

/// Build a boolean per-pixel mask selecting focus-measure values considered
/// "sharp enough" to drive per-pixel source-frame selection (as opposed to
/// falling back to the below-threshold averaging branch).
///
/// # Threshold resolution
///
/// - If `settings.absolute_threshold` is `Some(t)`, the mask is simply `v >=
///   t` for every pixel, and `contrast_pct` is ignored entirely (used by the
///   tiled pipeline to apply one globally-resolved threshold value uniformly
///   across tiles).
/// - Otherwise, the threshold is the `contrast_pct`-th order statistic of
///   this call's own `focus_measure` data:
///   - `contrast_pct == 0.0` **always** yields an all-`true` mask (documented
///     invariant relied on by the GUI in-RAM default path) — this includes
///     the degenerate all-zero-image case, since the 0th order statistic is
///     always `<=` every value regardless of which population it is drawn
///     from.
///   - `contrast_pct > 0.0` computes the order statistic over the **non-zero**
///     values only. The SML focus measure has a dense exact-zero / near-zero
///     plateau (out-of-focus background, and degenerate windows that write
///     exactly `0.0`), so nearby percentiles of the *whole* population
///     (including that plateau) collapse to nearly the same threshold value
///     and the mask barely changes between e.g. 20% and 30%. Restricting the
///     order statistic to the non-zero population makes `contrast_pct` a
///     percentile of the *meaningful* (in-focus-candidate) values, so
///     different slider positions produce visibly different thresholds.
///     If every value happens to be zero (all-zero image), there is no
///     non-zero population to sample from — the function falls back to an
///     all-`true` mask, matching the `contrast_pct == 0.0` invariant rather
///     than panicking or picking an arbitrary threshold.
pub fn generate_mask(focus_measure: &PlanarImage<f32>, settings: &ReliefSettings) -> Vec<bool> {
    if focus_measure.luma.is_empty() {
        return vec![];
    }

    let threshold = if let Some(t) = settings.absolute_threshold {
        t
    } else if settings.contrast_pct <= 0.0 {
        // Preserve the documented invariant: pct == 0.0 (or below) always
        // selects every pixel, regardless of the non-zero-population logic
        // below.
        return vec![true; focus_measure.luma.len()];
    } else {
        let mut buf: Vec<f32> = focus_measure
            .luma
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .collect();

        if buf.is_empty() {
            // All-zero image: no non-zero population to threshold against —
            // fall back to the all-true invariant instead of an arbitrary
            // (or panicking) selection.
            return vec![true; focus_measure.luma.len()];
        }

        let len = buf.len();
        let mut idx = (len as f32 * settings.contrast_pct) as usize;
        if idx >= len {
            idx = len - 1;
        }

        // Only the idx-th order statistic is needed, so partition in O(n)
        // rather than a full O(n log n) sort. `select_nth_unstable_by`
        // places the idx-th-smallest element at `idx` with all smaller
        // elements before it — the same value `sorted[idx]` would have held.
        let (_, kth, _) = buf.select_nth_unstable_by(idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        *kth
    };

    focus_measure.luma.iter().map(|&v| v >= threshold).collect()
}
