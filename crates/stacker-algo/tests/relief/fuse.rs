#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use stacker_algo::relief::fuse::auto_contrast_threshold;
use stacker_core::image::PlanarImage;

/// Synthetic heavy-tailed SML-like distribution: 95% of values clustered
/// near-zero (out-of-focus background, with a little noise), 5% spread over
/// a wide high range (in-focus edge pixels). The old linear-domain Otsu
/// collapsed to the extreme top bin (reported as auto-detect picking ~99%);
/// the MAD-based noise-floor statistic should land well below that while
/// still separating the two populations.
#[test]
fn test_auto_threshold_heavy_tailed_distribution_not_near_top() {
    let n_background = 9500;
    let n_signal = 500;
    let total = n_background + n_signal;

    let mut luma = Vec::with_capacity(total);

    // Background: ~0.001 with tiny noise, deterministic pseudo-random spread.
    for i in 0..n_background {
        let noise = ((i * 2_654_435_761) % 1000) as f32 / 1_000_000.0; // 0..0.001
        luma.push(0.001 + noise);
    }
    // Signal: spread across 0.1..1.0.
    for i in 0..n_signal {
        let frac = i as f32 / n_signal as f32; // 0..1
        luma.push(0.1 + frac * 0.9);
    }

    let img = PlanarImage {
        width: total,
        height: 1,
        luma,
        chroma_a: vec![0.0; total],
        chroma_b: vec![0.0; total],
    };

    let pct = auto_contrast_threshold(&img);

    assert!(
        pct <= 0.97,
        "auto-detected percentile should not sit near the top of the \
         distribution for a heavy-tailed focus measure, got {pct}"
    );
    // The two populations are well separated (background ~0.001..0.002,
    // signal 0.1..1.0): the noise-floor threshold should cover at least the
    // bulk of the background cluster without collapsing to ~0 or ~1.
    assert!(
        pct > 0.5,
        "threshold should separate at least the bulk of the near-zero \
         background population, got {pct}"
    );
    // None of the genuine signal population (top 5%) may fall below the
    // resolved threshold.
    assert!(
        pct <= 0.95,
        "threshold must not eat into the in-focus signal population, got {pct}"
    );
}

#[test]
fn test_auto_threshold_flat_image_returns_zero() {
    let img = PlanarImage {
        width: 4,
        height: 4,
        luma: vec![0.5; 16],
        chroma_a: vec![0.0; 16],
        chroma_b: vec![0.0; 16],
    };
    assert_eq!(auto_contrast_threshold(&img), 0.0);
}

#[test]
fn test_auto_threshold_all_zero_image_returns_zero() {
    let img = PlanarImage {
        width: 4,
        height: 4,
        luma: vec![0.0; 16],
        chroma_a: vec![0.0; 16],
        chroma_b: vec![0.0; 16],
    };
    assert_eq!(auto_contrast_threshold(&img), 0.0);
}
