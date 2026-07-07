// Test-only pixel-coordinate/count -> f32 casts throughout this file are
// pedantic noise with no behavioural relevance to what these tests assert.
#![allow(clippy::cast_precision_loss)]

use stacker_algo::apex::fuse::{
    ApexAccumulator, fuse_pyramids_incremental, fuse_pyramids_incremental_with_progress,
};
use stacker_core::image::PlanarImage;

/// Build a small test image with a non-trivial luma pattern (sine wave).
fn make_sine_image(w: usize, h: usize, phase: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;

            let v = ((x as f32).mul_add(0.3, phase).sin()
                * phase.mul_add(1.1, y as f32 * 0.25).cos())
            .mul_add(0.5, 0.5)
            .clamp(0.0, 1.0);
            img.luma[i] = v;
            img.chroma_a[i] = (v - 0.5) * 0.3;
            img.chroma_b[i] = (0.5 - v) * 0.2;
        }
    }
    img
}

/// Max absolute difference between two images' luma channels.
fn max_abs_diff(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> f32 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    a.luma
        .iter()
        .zip(b.luma.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// RMS difference over a rectangular region of the luma channel.
fn region_rms(img: &PlanarImage<f32>, x0: usize, y0: usize, x1: usize, y1: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let v = img.luma[y * img.width + x];
            sum = v.mul_add(v, sum);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f32).sqrt()
    }
}

/// RMS difference between two images' luma channels in a rectangular region.
fn region_rms_diff(
    a: &PlanarImage<f32>,
    b: &PlanarImage<f32>,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) -> f32 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for y in y0..y1 {
        for x in x0..x1 {
            let i = y * a.width + x;
            let d = a.luma[i] - b.luma[i];
            sum = d.mul_add(d, sum);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f32).sqrt()
    }
}

/// 1. Round-trip single frame: a single-frame accumulator reconstructs
///    approximately to the original image (per-pixel abs diff < 1e-3).
#[test]
fn single_frame_roundtrip() {
    let img = make_sine_image(33, 17, 0.0);
    let result = ApexAccumulator::new(&img, false, false).reconstruct();
    let diff = max_abs_diff(&img, &result);
    assert!(
        diff < 1e-3,
        "round-trip max abs diff {diff} >= 1e-3 — pyramid reconstruct is not reversible"
    );
}

/// 2. Identical frames: blending N identical copies reconstructs
///    approximately to the original (residual mean of identical values is
///    unchanged; equal energies keep the accumulator at every level).
#[test]
fn identical_frames_roundtrip() {
    let img = make_sine_image(32, 32, 1.0);
    let images: Vec<PlanarImage<f32>> = (0..5).map(|_| img.clone()).collect();
    let result = fuse_pyramids_incremental(&images, false, false);
    let diff = max_abs_diff(&img, &result);
    assert!(diff < 1e-3, "identical-frames max abs diff {diff} >= 1e-3");
}

/// 3. Max-contrast selection: frame B has strictly higher-frequency detail
///    in the left half; frame A has it in the right half.  After fusing,
///    the RMS difference to B should be smaller than to A in the left half,
///    and vice-versa on the right.
#[test]
fn max_contrast_selection() {
    let w = 32usize;
    let h = 32usize;

    // Frame A: high-frequency detail on the RIGHT half (x >= w/2), flat left.
    let mut frame_a = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;

            if x >= w / 2 {
                // High-frequency detail: fine sinusoidal pattern.
                frame_a.luma[i] = (y as f32)
                    .mul_add(1.3, x as f32 * 1.5)
                    .sin()
                    .mul_add(0.5, 0.5)
                    .clamp(0.0, 1.0);
            } else {
                // Flat grey.
                frame_a.luma[i] = 0.5;
            }
        }
    }

    // Frame B: high-frequency detail on the LEFT half (x < w/2), flat right.
    let mut frame_b = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;

            if x < w / 2 {
                frame_b.luma[i] = (y as f32)
                    .mul_add(1.3, x as f32 * 1.5)
                    .sin()
                    .mul_add(0.5, 0.5)
                    .clamp(0.0, 1.0);
            } else {
                frame_b.luma[i] = 0.5;
            }
        }
    }

    let result = fuse_pyramids_incremental(&[frame_a.clone(), frame_b.clone()], false, false);

    // In the left half the result should be closer to B than to A.
    let left_rms_vs_a = region_rms_diff(&result, &frame_a, 0, 0, w / 2, h);
    let left_rms_vs_b = region_rms_diff(&result, &frame_b, 0, 0, w / 2, h);
    assert!(
        left_rms_vs_b < left_rms_vs_a,
        "left half: expected result closer to B (rms_vs_b={left_rms_vs_b:.4}) \
         than to A (rms_vs_a={left_rms_vs_a:.4})"
    );

    // In the right half the result should be closer to A than to B.
    let right_rms_vs_a = region_rms_diff(&result, &frame_a, w / 2, 0, w, h);
    let right_rms_vs_b = region_rms_diff(&result, &frame_b, w / 2, 0, w, h);
    assert!(
        right_rms_vs_a < right_rms_vs_b,
        "right half: expected result closer to A (rms_vs_a={right_rms_vs_a:.4}) \
         than to B (rms_vs_b={right_rms_vs_b:.4})"
    );
}

/// 4. First-wins tie-break: two identical frames → the accumulator equals
///    the single-frame pyramid at every level (no spurious changes from
///    ties, since strict > keeps the accumulator).
#[test]
fn first_wins_tiebreak() {
    let img = make_sine_image(16, 16, 2.5);
    // Single-frame reference.
    let single = ApexAccumulator::new(&img, false, false).reconstruct();
    // Two identical frames.
    let two = fuse_pyramids_incremental(&[img.clone(), img], false, false);
    let diff = max_abs_diff(&single, &two);
    assert!(
        diff < 1e-6,
        "first-wins tie-break: max abs diff {diff} >= 1e-6 — \
         the second identical frame mutated the accumulator"
    );
}

/// 5. Progress-reporting variant matches the plain incremental result and
///    invokes the callback exactly `images.len()` times, with `completed`
///    counting up from 1 and a preview image present on the first and
///    last calls.
#[test]
fn incremental_with_progress_matches_plain_and_reports_correctly() {
    let images: Vec<PlanarImage<f32>> = (0_u8..6)
        .map(|i| make_sine_image(20, 20, f32::from(i) * 0.4))
        .collect();

    let plain = fuse_pyramids_incremental(&images, false, false);

    let mut calls: Vec<(usize, usize, bool)> = Vec::new();
    let with_progress = fuse_pyramids_incremental_with_progress(
        &images,
        false,
        false,
        |completed, total, preview| {
            calls.push((completed, total, preview.is_some()));
        },
    );

    assert_eq!(calls.len(), images.len());
    for (expected, &(completed, total, _)) in (1..=images.len()).zip(calls.iter()) {
        assert_eq!(completed, expected);
        assert_eq!(total, images.len());
    }
    assert!(calls[0].2, "first call should carry a preview");
    assert!(calls.last().unwrap().2, "last call should carry a preview");

    let diff = max_abs_diff(&plain, &with_progress);
    assert!(
        diff < 1e-6,
        "progress-reporting variant diverged from plain incremental result: {diff}"
    );
}

// Suppress dead-code warnings for helpers only used in some tests.
#[allow(dead_code)]
fn _use_helpers(
    a: &PlanarImage<f32>,
    b: &PlanarImage<f32>,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
) {
    let _ = region_rms(a, x0, y0, x1, y1);
    let _ = region_rms_diff(a, b, x0, y0, x1, y1);
}
