#![allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]

use stacker_algo::relief::threshold::*;
use stacker_core::image::PlanarImage;

#[test]
fn test_noise_rejection_percentile() {
    let mut luma = vec![0.0; 100];
    // 90 pixels of noise (value 0.1), 10 pixels of signal (value 1.0)
    luma.iter_mut().take(90).for_each(|v| *v = 0.1);
    luma.iter_mut().skip(90).for_each(|v| *v = 1.0);

    let img = PlanarImage {
        width: 10,
        height: 10,
        luma,
        chroma_a: vec![0.0; 100],
        chroma_b: vec![0.0; 100],
    };

    let settings = ReliefSettings {
        est_radius: 1,
        smooth_radius: 2,
        contrast_pct: 0.80, // 80th percentile is 0.1
        absolute_threshold: None,
    };

    let mask1 = generate_mask(&img, &settings);
    // values >= 0.1 will be true (which is all of them since min is 0.1)
    assert!(mask1.iter().all(|&v| v));

    let settings_high = ReliefSettings {
        est_radius: 1,
        smooth_radius: 2,
        contrast_pct: 0.95, // 95th percentile is 1.0
        absolute_threshold: None,
    };

    let mask2 = generate_mask(&img, &settings_high);
    let true_count = mask2.iter().filter(|&&v| v).count();
    assert_eq!(true_count, 10); // Only the 10 signal pixels remain
}

/// `contrast_pct == 0.0` must always yield an all-true mask, regardless of
/// how many exact-zero values are present (the documented invariant the GUI
/// in-RAM default path relies on).
#[test]
fn test_pct_zero_always_all_true() {
    let mut luma = vec![0.0; 100];
    // Half zeros, half positive values.
    luma.iter_mut().skip(50).enumerate().for_each(|(i, v)| {
        *v = 0.01 + i as f32 * 0.01;
    });

    let img = PlanarImage {
        width: 10,
        height: 10,
        luma,
        chroma_a: vec![0.0; 100],
        chroma_b: vec![0.0; 100],
    };

    let settings = ReliefSettings {
        est_radius: 1,
        smooth_radius: 2,
        contrast_pct: 0.0,
        absolute_threshold: None,
    };
    let mask = generate_mask(&img, &settings);
    assert!(mask.iter().all(|&v| v), "pct=0.0 must select every pixel");

    // All-zero image edge case must also be all-true, not panic or produce
    // an arbitrary mask.
    let all_zero = PlanarImage {
        width: 10,
        height: 10,
        luma: vec![0.0; 100],
        chroma_a: vec![0.0; 100],
        chroma_b: vec![0.0; 100],
    };
    let mask_zero_img = generate_mask(&all_zero, &settings);
    assert!(
        mask_zero_img.iter().all(|&v| v),
        "all-zero image at pct=0.0 must select every pixel"
    );
}

/// Regression test for task 54a2: with a dense zero plateau (out-of-focus
/// background) plus a spread of positive values, different `contrast_pct`
/// settings (e.g. 20% vs 30%) must resolve to genuinely different threshold
/// values once the order statistic is restricted to the non-zero
/// population — previously both percentiles landed inside/near the zero
/// plateau and produced nearly identical masks.
#[test]
fn test_nonzero_population_percentile_distinguishes_pct_20_vs_30() {
    let n_zero = 500;
    let n_positive = 500;
    let total = n_zero + n_positive;

    let mut luma = vec![0.0; n_zero];
    // Positive values spread evenly across 0.01..1.0.
    for i in 0..n_positive {
        let frac = i as f32 / (n_positive - 1) as f32;
        luma.push(0.01 + frac * 0.99);
    }

    let img = PlanarImage {
        width: total,
        height: 1,
        luma,
        chroma_a: vec![0.0; total],
        chroma_b: vec![0.0; total],
    };

    let settings_20 = ReliefSettings {
        est_radius: 1,
        smooth_radius: 2,
        contrast_pct: 0.2,
        absolute_threshold: None,
    };
    let settings_30 = ReliefSettings {
        est_radius: 1,
        smooth_radius: 2,
        contrast_pct: 0.3,
        absolute_threshold: None,
    };

    let mask_20 = generate_mask(&img, &settings_20);
    let mask_30 = generate_mask(&img, &settings_30);

    let count_20 = mask_20.iter().filter(|&&v| v).count();
    let count_30 = mask_30.iter().filter(|&&v| v).count();

    assert_ne!(
        count_20, count_30,
        "20% and 30% contrast thresholds must select a different number of \
         pixels once computed over the non-zero population"
    );
}

/// `absolute_threshold`, when set, must be used directly (`v >= t`) and
/// bypass `contrast_pct` entirely.
#[test]
fn test_absolute_threshold_overrides_contrast_pct() {
    let luma = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let img = PlanarImage {
        width: 6,
        height: 1,
        luma,
        chroma_a: vec![0.0; 6],
        chroma_b: vec![0.0; 6],
    };

    let settings = ReliefSettings {
        est_radius: 1,
        smooth_radius: 2,
        // Would normally select almost everything at pct=0.01, but the
        // absolute threshold must win.
        contrast_pct: 0.01,
        absolute_threshold: Some(0.5),
    };

    let mask = generate_mask(&img, &settings);
    assert_eq!(mask, vec![false, false, false, true, true, true]);
}
