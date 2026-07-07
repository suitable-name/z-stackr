use crate::refine::nelder_mead_generic;
use stacker_core::image::PlanarImage;

/// Target statistics for per-frame brightness correction.
pub struct BrightnessTarget {
    pub target_mean: f64,
    pub target_std: f64,
}

// Gamma-space luma is [0, 2]. 10000 bins.
const NUM_BINS: usize = 10000;
const BIN_WIDTH: f64 = 2.0 / NUM_BINS as f64;

impl BrightnessTarget {
    pub fn new(frame: &PlanarImage<f32>) -> Self {
        let (mean, std) = compute_stats_from_histogram(frame);
        Self {
            target_mean: mean,
            target_std: std,
        }
    }
}

/// Bins a single non-negative, finite luma sample into `[0, NUM_BINS)`.
fn luma_to_bin(val: f32) -> usize {
    let bin = (f64::from(val) / BIN_WIDTH).floor() as isize;
    bin.clamp(0, NUM_BINS.cast_signed() - 1) as usize
}

fn compute_stats_from_histogram(frame: &PlanarImage<f32>) -> (f64, f64) {
    let mut hist = vec![0usize; NUM_BINS];
    let mut count = 0usize;
    for &val in &frame.luma {
        if !val.is_finite() || val < 0.0 {
            continue;
        }
        hist[luma_to_bin(val)] += 1;
        count += 1;
    }

    if count == 0 {
        return (0.0, 0.0);
    }

    let mut sum = 0.0;
    for (i, &h) in hist.iter().enumerate() {
        if h > 0 {
            let bin_center = (i as f64 + 0.5) * BIN_WIDTH;
            sum = (h as f64).mul_add(bin_center, sum);
        }
    }
    let mean = sum / count as f64;

    let mut sq_sum = 0.0;
    for (i, &h) in hist.iter().enumerate() {
        if h > 0 {
            let bin_center = (i as f64 + 0.5) * BIN_WIDTH;
            let diff = bin_center - mean;
            sq_sum = (h as f64 * diff).mul_add(diff, sq_sum);
        }
    }
    let std = (sq_sum / (count as f64 - 1.0).max(1.0)).max(0.0).sqrt();

    (mean, std)
}

/// Applies per-frame brightness/gamma correction.
pub fn apply_brightness_correction(frame: &mut PlanarImage<f32>, target: &BrightnessTarget) {
    let mut hist = vec![0usize; NUM_BINS];
    let mut count = 0usize;
    for &val in &frame.luma {
        if !val.is_finite() || val < 0.0 {
            continue;
        }
        hist[luma_to_bin(val)] += 1;
        count += 1;
    }

    if count == 0 {
        return;
    }

    let f = |p: &[f64]| {
        let scale = p[0] * p[0];
        let gamma = 2.5 * (p[1].sin() + 1.0);

        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for (i, &h) in hist.iter().enumerate() {
            if h > 0 {
                let n = h as f64;
                let bin_center = (i as f64 + 0.5) * BIN_WIDTH;
                let corrected = (bin_center * scale).powf(gamma);
                sum = n.mul_add(corrected, sum);
                sum_sq = (n * corrected).mul_add(corrected, sum_sq);
            }
        }

        let mean_c = sum / count as f64;
        let var_c = (sum_sq - sum * sum / count as f64) / (count as f64 - 1.0).max(1.0);
        let std_c = var_c.max(0.0).sqrt();

        let d_mean = mean_c - target.target_mean;
        let d_std = std_c - target.target_std;
        d_mean.mul_add(d_mean, d_std * d_std)
    };

    let p0 = 1.0;
    let p1 = (-0.6_f64).asin();
    let initial = [p0, p1];

    let best_p = nelder_mead_generic(&f, &initial, 0.1, 100, 1e-6);

    let best_scale = (best_p[0] * best_p[0]) as f32;
    let best_gamma = (2.5 * (best_p[1].sin() + 1.0)) as f32;

    for i in 0..frame.luma.len() {
        let y = frame.luma[i];
        if y > 0.0 && y.is_finite() {
            let y_out = (y * best_scale).powf(best_gamma);
            let ratio = if y > 1e-6 { y_out / y } else { 1.0 };
            frame.luma[i] = y_out;
            frame.chroma_a[i] *= ratio;
            frame.chroma_b[i] *= ratio;
        } else {
            frame.luma[i] = 0.0;
            frame.chroma_a[i] = 0.0;
            frame.chroma_b[i] = 0.0;
        }
    }
}
