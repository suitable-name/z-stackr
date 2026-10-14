use stacker_algo::relief::threshold::*;
use stacker_core::image::PlanarImage;

fn map_from(luma: Vec<f32>, width: usize, height: usize) -> PlanarImage<f32> {
    let len = width * height;
    assert_eq!(luma.len(), len);
    PlanarImage::with_luma(width, height, luma).expect("one sample per pixel")
}

const fn settings(contrast_pct: f32) -> ReliefSettings {
    ReliefSettings { contrast_pct }
}

#[test]
fn test_noise_rejection_percentile() {
    let mut luma = vec![0.0; 100];
    // 90 pixels of noise (value 0.1), 10 pixels of signal (value 1.0)
    luma.iter_mut().take(90).for_each(|v| *v = 0.1);
    luma.iter_mut().skip(90).for_each(|v| *v = 1.0);
    let img = map_from(luma, 10, 10);

    // The 80th percentile value is 0.1, which 90 % of the pixels share.  The
    // tie rule keeps the 10 signal pixels (closest to the wanted 20 %) rather
    // than every pixel.
    let mask1 = generate_mask(&img, &settings(0.80));
    assert_eq!(mask1.iter().filter(|&&v| v).count(), 10);
    assert!(mask1[90..].iter().all(|&v| v));

    // 95th percentile is 1.0: only the 10 signal pixels remain.
    let mask2 = generate_mask(&img, &settings(0.95));
    assert_eq!(mask2.iter().filter(|&&v| v).count(), 10);
}

/// FUS-4: a map that is 0 in 90 % of the pixels keeps exactly the other 10 %
/// at `contrast_pct = 0.5` (it used to keep all of them).
#[test]
fn test_flat_plateau_is_not_kept() {
    let mut luma = vec![0.0; 100];
    luma.iter_mut().skip(90).for_each(|v| *v = 1.0);
    let img = map_from(luma, 10, 10);

    let mask = generate_mask(&img, &settings(0.5));
    assert_eq!(mask.iter().filter(|&&v| v).count(), 10);
    assert!(mask[90..].iter().all(|&v| v));
}

/// `contrast_pct == 0` keeps everything, ties or not.
#[test]
fn test_zero_pct_keeps_everything() {
    let mut luma = vec![0.0; 100];
    luma.iter_mut().skip(90).for_each(|v| *v = 1.0);
    let mask = generate_mask(&map_from(luma, 10, 10), &settings(0.0));
    assert!(mask.iter().all(|&v| v));
}

/// A tie-free map keeps exactly `len - round(len * pct)` pixels.
#[test]
fn test_distinct_values_keep_the_requested_fraction() {
    let luma: Vec<f32> = (0..1000_u16).map(f32::from).collect();
    let img = map_from(luma, 40, 25);
    for (pct, kept) in [(0.0, 1000), (0.25, 750), (0.5, 500), (0.9, 100)] {
        let mask = generate_mask(&img, &settings(pct));
        assert_eq!(mask.iter().filter(|&&v| v).count(), kept, "pct {pct}");
    }
}

#[test]
fn test_empty_map() {
    let img = map_from(vec![], 0, 0);
    assert_eq!(generate_mask(&img, &settings(0.5)).len(), 0);
    assert!(otsu_threshold(&[]).abs() < f32::EPSILON);
}

/// FUS-4: one whole-image threshold applied to row-chunks (tiles) gives the
/// same mask as the whole-image call.
#[test]
fn test_threshold_computed_once_matches_whole_image_mask() {
    let (w, h) = (64, 64);
    let mut state = 0x2545_F491_u32;
    let luma: Vec<f32> = (0..w * h)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Tiles at different scales: the top half is 10x brighter.
            let scale = if i < w * h / 2 { 10.0 } else { 1.0 };
            f32::from(u16::try_from(state >> 16).unwrap()) / 65_536.0 * scale
        })
        .collect();
    let img = map_from(luma.clone(), w, h);

    let whole = generate_mask(&img, &settings(0.3));
    let threshold = compute_relief_threshold(&luma, 0.3, false);
    assert!((threshold.contrast_pct - 0.3).abs() < f32::EPSILON);

    let tiled: Vec<bool> = luma
        .chunks(w * 16)
        .flat_map(|tile| mask_above(tile, threshold.value))
        .collect();
    assert_eq!(whole, tiled);
}

/// FUS-4: Otsu's fraction counts the same bins it scored.  A 90/10 two-level
/// map splits at the lower level, so the background fraction is 0.9 (the old
/// conversion excluded the split bin and returned 0.0).
#[test]
fn test_otsu_fraction_includes_split_bin() {
    let mut luma = vec![0.1; 100];
    luma.iter_mut().skip(90).for_each(|v| *v = 1.0);
    let frac = otsu_threshold(&luma);
    assert!((frac - 0.9).abs() < 1e-6, "got {frac}");
}

#[test]
fn test_otsu_constant_map_is_zero() {
    assert!(otsu_threshold(&[0.5; 64]).abs() < f32::EPSILON);
}

/// A synthetic heavy-tailed focus map (95 % noise floor in `[0.001, 0.002)`, 5 %
/// edges spread over `[0.1, 1.0)`), the shape that pushes Otsu's split to the top
/// of the range.
fn heavy_tailed() -> Vec<f32> {
    let (n_background, n_signal) = (9500_u64, 500_u32);
    let mut luma = Vec::new();
    for i in 0..n_background {
        let noise = f32::from(u16::try_from(i * 2_654_435_761 % 1000).expect("below 1000"));
        luma.push(0.001 + noise / 1_000_000.0);
    }
    for i in 0..n_signal {
        let ramp = f32::from(u16::try_from(i).expect("below 500")) / 500.0;
        luma.push(ramp.mul_add(0.9, 0.1));
    }
    luma
}

/// D9: the median + 1.5 MAD rule lands between the noise floor and the edges where
/// Otsu would sit near the top.
#[test]
fn test_auto_detect_heavy_tailed_distribution_not_near_top() {
    let luma = heavy_tailed();
    let frac = auto_detect_threshold(&luma);
    assert!(
        frac > 0.5,
        "should cover the bulk of the noise floor, got {frac}"
    );
    assert!(
        frac <= 0.95,
        "must not eat into the edge population, got {frac}"
    );
}

#[test]
fn test_auto_detect_flat_or_empty_or_all_zero_map_returns_zero() {
    assert!(auto_detect_threshold(&[0.5; 16]).abs() < f32::EPSILON);
    assert!(auto_detect_threshold(&[0.0; 16]).abs() < f32::EPSILON);
    assert!(auto_detect_threshold(&[]).abs() < f32::EPSILON);
    assert!(auto_detect_threshold(&[-1.0, f32::NAN, 0.0]).abs() < f32::EPSILON);
}

/// The scale of a focus map follows the square of the pixel scale, so the "no
/// spread" guard is relative: the same map a million times smaller gives the same
/// fraction.
#[test]
fn test_auto_detect_is_scale_free() {
    let luma = heavy_tailed();
    let tiny: Vec<f32> = luma.iter().map(|&v| v * 1e-6).collect();
    let (a, b) = (auto_detect_threshold(&luma), auto_detect_threshold(&tiny));
    assert!((a - b).abs() < 5e-3, "{a} vs {b}");
    assert!(b > 0.5);
}

/// The fraction counts zeros too, because the percentile rule ranks all pixels: a
/// map with a large exact-zero plateau keeps the edge population and drops the
/// noise floor.
#[test]
fn test_auto_detect_fraction_feeds_the_all_pixel_percentile() {
    let mut luma = vec![0.0_f32; 5000];
    luma.extend(heavy_tailed()[..9500].iter().take(9000));
    luma.extend(heavy_tailed()[9500..].iter().copied());
    let signal = 500;

    let frac = auto_detect_threshold(&luma);
    assert!(frac > 0.5 && frac <= 0.95, "got {frac}");

    let auto = compute_relief_threshold(&luma, 0.0, true);
    assert!((auto.contrast_pct - frac).abs() < f32::EPSILON);
    let mask = mask_above(&luma, auto.value);
    let kept = mask.iter().filter(|&&v| v).count();
    let n = luma.len();
    assert!(
        mask[n - signal..].iter().all(|&v| v),
        "every edge pixel is kept"
    );
    assert!(
        kept < n / 2,
        "the noise floor is dropped, kept {kept} of {n}"
    );
    assert!(
        mask[..5000].iter().all(|&v| !v),
        "the zero plateau is dropped"
    );
}

/// Large maps are sampled (at most 100 000 pixels, deterministically). A stride
/// that divides the image width would look at a few columns only: here every
/// fourth column is a constant edge column of 0.1 and the rest is a noise floor, and
/// a constant stride of 4 would see nothing but the edge columns (no spread, so a
/// fraction of 0). The sample has to see both populations: about 75 % of it is the
/// noise floor.
#[test]
fn test_auto_detect_sampling_is_not_aliased_with_the_width() {
    let (w, h) = (2000_usize, 200_usize);
    let luma: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if x % 4 == 0 {
                0.1
            } else {
                let noise = u16::try_from((x * 31 + y * 17) % 1000).expect("below 1000");
                0.001 + f32::from(noise) / 1_000_000.0
            }
        })
        .collect();
    let frac = auto_detect_threshold(&luma);
    assert!(frac > 0.6 && frac < 0.9, "got {frac}");
    assert!(
        (auto_detect_threshold(&luma) - frac).abs() < f32::EPSILON,
        "the sampling is deterministic"
    );
}
