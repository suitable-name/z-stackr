//! Behaviour of the Strata fusion: identities, which frame wins where, the
//! detail focus, and the deep-stack dilution regression.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use stacker_algo::{
    apex::pyramid::apply_gaussian_blur,
    strata::{
        STRATA_APRON_PX, StrataParams, fuse_strata, raise_detail_weight,
        saliency::{compute_argmax, laplacian_magnitude},
        strata_bytes_per_pixel,
    },
};
use stacker_core::image::PlanarImage;

fn const_image(w: usize, h: usize, l: f32, a: f32, b: f32) -> PlanarImage<f32> {
    PlanarImage {
        width: w,
        height: h,
        luma: vec![l; w * h],
        chroma_a: vec![a; w * h],
        chroma_b: vec![b; w * h],
    }
}

fn textured_image(w: usize, h: usize, phase: f32) -> PlanarImage<f32> {
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

/// A frame that is a checkerboard (`lo`/`hi`) on one half and flat (`flat`) on
/// the other.
fn half_sharp(
    w: usize,
    h: usize,
    sharp_left: bool,
    lo: f32,
    hi: f32,
    flat: f32,
) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sharp_here = if sharp_left { x < w / 2 } else { x >= w / 2 };
            img.luma[y * w + x] = if sharp_here {
                if (x + y) % 2 == 0 { hi } else { lo }
            } else {
                flat
            };
        }
    }
    img
}

fn mean_abs_laplacian(img: &PlanarImage<f32>, x0: usize, x1: usize) -> f32 {
    let lap = laplacian_magnitude(&img.luma, img.width, img.height);
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for y in 0..img.height {
        for x in x0..x1 {
            sum += lap[y * img.width + x];
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn max_diff_in_columns(a: &PlanarImage<f32>, b: &PlanarImage<f32>, x0: usize, x1: usize) -> f32 {
    let mut worst = 0.0_f32;
    for y in 0..a.height {
        for x in x0..x1 {
            let i = y * a.width + x;
            worst = worst.max((a.luma[i] - b.luma[i]).abs());
        }
    }
    worst
}

fn params(base_radius: usize) -> StrataParams {
    StrataParams {
        base_radius,
        ..StrataParams::default()
    }
}

#[test]
fn defaults_are_the_documented_ones() {
    let p = StrataParams::default();
    assert_eq!(p.base_radius, 31);
    assert_eq!(p.detail_focus, 3);
}

#[test]
fn constant_stack_is_identity() {
    let (l, a, b) = (0.42_f32, 0.1_f32, -0.05_f32);
    let img = const_image(24, 24, l, a, b);
    let frames: Vec<_> = (0..4).map(|_| img.clone()).collect();
    let fused = fuse_strata(&frames, &params(8));
    for &v in &fused.luma {
        assert!((v - l).abs() < 1e-5, "luma: got {v}, want {l}");
    }
    for &v in &fused.chroma_a {
        assert!((v - a).abs() < 1e-5, "chroma_a: got {v}, want {a}");
    }
    for &v in &fused.chroma_b {
        assert!((v - b).abs() < 1e-5, "chroma_b: got {v}, want {b}");
    }
}

#[test]
fn identical_frames_reproduce_the_frame() {
    let img = textured_image(32, 32, 0.7);
    let frames: Vec<_> = (0..5).map(|_| img.clone()).collect();
    let fused = fuse_strata(&frames, &params(6));
    for (plane_fused, plane_src) in [
        (&fused.luma, &img.luma),
        (&fused.chroma_a, &img.chroma_a),
        (&fused.chroma_b, &img.chroma_b),
    ] {
        let worst = plane_fused
            .iter()
            .zip(plane_src)
            .map(|(f, s)| (f - s).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-5, "identical frames: max abs diff {worst}");
    }
}

#[test]
fn single_frame_is_returned_unchanged() {
    let img = textured_image(20, 20, 0.2);
    let fused = fuse_strata(std::slice::from_ref(&img), &params(5));
    assert!(max_diff_in_columns(&fused, &img, 0, 20) < 1e-5);
}

#[test]
fn two_half_sharp_frames_are_sharper_than_the_flat_halves() {
    let (w, h) = (40, 40);
    let a = half_sharp(w, h, true, 0.1, 0.9, 0.5);
    let b = half_sharp(w, h, false, 0.1, 0.9, 0.5);
    let fused = fuse_strata(&[a.clone(), b.clone()], &params(6));
    assert!(mean_abs_laplacian(&fused, 0, w / 2) > mean_abs_laplacian(&b, 0, w / 2));
    assert!(mean_abs_laplacian(&fused, w / 2, w) > mean_abs_laplacian(&a, w / 2, w));
}

#[test]
fn each_half_is_taken_from_the_sharp_frame() {
    // 96 x 32: the winner changes at column 48; the weight refinement reaches
    // 14 px (two radius-7 boxes), so columns well away from 48 take their value
    // from the sharp frame alone. The base layers of the two frames differ by at
    // most one checker cell of a (clipped) box window.
    let (w, h) = (96, 32);
    let a = half_sharp(w, h, true, 0.1, 0.9, 0.5);
    let b = half_sharp(w, h, false, 0.1, 0.9, 0.5);
    let fused = fuse_strata(&[a.clone(), b.clone()], &params(6));
    let left = max_diff_in_columns(&fused, &a, 0, 28);
    let right = max_diff_in_columns(&fused, &b, 68, w);
    assert!(left < 0.02, "left interior differs from frame A by {left}");
    assert!(
        right < 0.02,
        "right interior differs from frame B by {right}"
    );
}

#[test]
fn frame_order_does_not_change_which_half_wins() {
    let (w, h) = (96, 32);
    let a = half_sharp(w, h, true, 0.1, 0.9, 0.5);
    let b = half_sharp(w, h, false, 0.1, 0.9, 0.5);
    let fused = fuse_strata(&[b.clone(), a.clone()], &params(6));
    assert!(max_diff_in_columns(&fused, &a, 0, 28) < 0.02);
    assert!(max_diff_in_columns(&fused, &b, 68, w) < 0.02);
}

#[test]
fn dark_low_contrast_detail_still_wins_in_linear_light() {
    // Linear-light values of a dim scene: a 0.02/0.08 checker on a 0.05 field.
    // The guided-filter constants were tuned on gamma-encoded data; the sharp
    // half must still come from its sharp frame here.
    let (w, h) = (96, 32);
    let a = half_sharp(w, h, true, 0.02, 0.08, 0.05);
    let b = half_sharp(w, h, false, 0.02, 0.08, 0.05);
    let fused = fuse_strata(&[a.clone(), b.clone()], &params(6));
    assert!(max_diff_in_columns(&fused, &a, 0, 28) < 0.005);
    assert!(max_diff_in_columns(&fused, &b, 68, w) < 0.005);
    let sharp_energy = mean_abs_laplacian(&a, 0, w / 2);
    assert!(
        mean_abs_laplacian(&fused, 0, w / 2) > 0.4 * sharp_energy,
        "the dark sharp half lost its detail"
    );
}

#[test]
fn exact_ties_go_to_the_first_frame() {
    let img = textured_image(20, 20, 1.3);
    let single = fuse_strata(std::slice::from_ref(&img), &params(5));
    let two = fuse_strata(&[img.clone(), img.clone()], &params(5));
    assert!(max_diff_in_columns(&single, &two, 0, 20) < 1e-5);

    let (argmax_idx, _) = compute_argmax(&[img.clone(), img], 20, 20);
    assert!(argmax_idx.iter().all(|&i| i == 0));
}

#[test]
fn flat_identical_stack_stays_finite_and_flat() {
    let frames: Vec<_> = (0..3)
        .map(|_| const_image(16, 16, 0.33, 0.0, 0.0))
        .collect();
    let fused = fuse_strata(&frames, &params(4));
    for &v in &fused.luma {
        assert!(v.is_finite());
        assert!((v - 0.33).abs() < 1e-4);
    }
}

#[test]
fn featureless_frames_of_different_value_fuse_to_the_first() {
    // With no texture anywhere the saliency is zero in every frame and the
    // first-wins tie rule makes frame 0 the winner everywhere; the result is
    // finite and equals that frame (not the mean).
    let frames = vec![
        const_image(16, 16, 0.2, 0.0, 0.0),
        const_image(16, 16, 0.4, 0.0, 0.0),
        const_image(16, 16, 0.9, 0.0, 0.0),
    ];
    let fused = fuse_strata(&frames, &params(4));
    for &v in &fused.luma {
        assert!(v.is_finite());
        assert!((v - 0.2).abs() < 1e-4, "got {v}");
    }
}

#[test]
fn detail_focus_one_leaves_the_weight_alone() {
    for &w in &[0.0_f32, 0.05, 0.3, 0.77, 1.0] {
        assert_eq!(raise_detail_weight(w, 1).to_bits(), w.to_bits());
        let cube = w * w * w;
        assert!((raise_detail_weight(w, 3) - cube).abs() < 1e-7);
        assert!((raise_detail_weight(w, 5) - w.powi(5)).abs() < 1e-7);
    }
    for exponent in 1..=6 {
        assert_eq!(
            raise_detail_weight(0.0, exponent).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            raise_detail_weight(1.0, exponent).to_bits(),
            1.0_f32.to_bits()
        );
    }
}

#[test]
fn detail_focus_does_not_change_exact_winner_regions() {
    // Far from the winner boundary the detail weights are exactly 0 or 1, which
    // every exponent leaves alone.
    let (w, h) = (96, 32);
    let frames = [
        half_sharp(w, h, true, 0.1, 0.9, 0.5),
        half_sharp(w, h, false, 0.1, 0.9, 0.5),
    ];
    let soft = fuse_strata(
        &frames,
        &StrataParams {
            base_radius: 6,
            detail_focus: 1,
        },
    );
    let hard = fuse_strata(
        &frames,
        &StrataParams {
            base_radius: 6,
            detail_focus: 5,
        },
    );
    assert!(max_diff_in_columns(&soft, &hard, 0, 28) < 1e-4);
    assert!(max_diff_in_columns(&soft, &hard, 68, w) < 1e-4);
}

#[test]
fn saliency_laplacian_of_a_spike() {
    let (w, h) = (9, 9);
    let mut luma = vec![0.0_f32; w * h];
    luma[4 * w + 4] = 1.0;
    let lap = laplacian_magnitude(&luma, w, h);
    assert!((lap[4 * w + 4] - 4.0).abs() < 1e-6);
    for (x, y) in [(3, 4), (5, 4), (4, 3), (4, 5)] {
        assert!((lap[y * w + x] - 1.0).abs() < 1e-6);
    }
    assert!(lap[0].abs() < 1e-6);
    let flat = laplacian_magnitude(&vec![0.3_f32; w * h], w, h);
    assert!(flat.iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn apron_and_memory_figures_are_consistent() {
    const { assert!(STRATA_APRON_PX <= stacker_core::memory::APRON_PX) };
    const { assert!(STRATA_APRON_PX >= 90 + 11) };
    assert!(strata_bytes_per_pixel() > 14 * 4);
}

#[test]
#[should_panic(expected = "empty")]
fn an_empty_stack_panics() {
    let _ = fuse_strata(&[], &StrataParams::default());
}

#[test]
#[should_panic(expected = "width")]
fn frames_of_different_width_panic() {
    let _ = fuse_strata(
        &[
            const_image(8, 8, 0.1, 0.0, 0.0),
            const_image(9, 8, 0.1, 0.0, 0.0),
        ],
        &params(2),
    );
}

// ── Deep-stack detail-dilution regression ─────────────────────────────────
//
// With dozens of frames the guided filter leaks a small positive detail weight
// to every near-winner frame; unchecked, those leaks dilute the one sharp
// detail layer. A synthetic stack of 12 frames, each sharp in its own
// horizontal band of one high-frequency texture and heavily blurred elsewhere,
// must keep a good part of the texture's detail energy and clearly beat the
// plain average. The thresholds sit between the values the soft (exponent 1)
// and focused (exponent 3) weights reached on the gamma-encoded reference
// implementation (0.29/3.5 against 0.50/6.0); they have not been re-measured
// in linear light.

const DEEP_FRAMES: usize = 12;

fn ground_truth(w: usize, h: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = ((x as f32) * 0.9).sin() * ((y as f32) * 0.8).cos();
            img.luma[y * w + x] = v.mul_add(0.5, 0.5).clamp(0.0, 1.0);
        }
    }
    img
}

fn heavily_blurred(img: &PlanarImage<f32>, passes: usize) -> PlanarImage<f32> {
    let mut out = img.clone();
    for _ in 0..passes {
        out = apply_gaussian_blur(&out);
    }
    out
}

fn band_sharp_frame(
    truth: &PlanarImage<f32>,
    blurred: &PlanarImage<f32>,
    band: usize,
    bands: usize,
) -> PlanarImage<f32> {
    let (w, h) = (truth.width, truth.height);
    let band_h = h / bands;
    let y0 = band * band_h;
    let y1 = if band + 1 == bands {
        h
    } else {
        (band + 1) * band_h
    };
    let mut out = blurred.clone();
    out.luma[y0 * w..y1 * w].copy_from_slice(&truth.luma[y0 * w..y1 * w]);
    out
}

fn rows_energy(img: &PlanarImage<f32>, y0: usize, y1: usize) -> f32 {
    let lap = laplacian_magnitude(&img.luma, img.width, img.height);
    let rows = &lap[y0 * img.width..y1 * img.width];
    rows.iter().sum::<f32>() / rows.len() as f32
}

/// Mean interior (away from the band edges) detail energy of `image`.
fn interior_energy(image: &PlanarImage<f32>) -> f32 {
    let band_h = image.height / DEEP_FRAMES;
    let margin = (band_h / 4).max(1);
    let mut total = 0.0_f32;
    for band in 0..DEEP_FRAMES {
        let y0 = band * band_h + margin;
        let y1 = ((band + 1) * band_h).saturating_sub(margin).max(y0 + 1);
        total += rows_energy(image, y0, y1);
    }
    total / DEEP_FRAMES as f32
}

fn deep_stack() -> (PlanarImage<f32>, Vec<PlanarImage<f32>>) {
    let (w, h) = (48, 48);
    let truth = ground_truth(w, h);
    let blurred = heavily_blurred(&truth, 12);
    let frames = (0..DEEP_FRAMES)
        .map(|band| band_sharp_frame(&truth, &blurred, band, DEEP_FRAMES))
        .collect();
    (truth, frames)
}

fn plain_average(frames: &[PlanarImage<f32>]) -> PlanarImage<f32> {
    let mut out = PlanarImage::new(frames[0].width, frames[0].height);
    let n = frames.len() as f32;
    for frame in frames {
        for (o, v) in out.luma.iter_mut().zip(&frame.luma) {
            *o += v / n;
        }
    }
    out
}

#[test]
fn deep_stack_keeps_its_detail_energy() {
    let (truth, frames) = deep_stack();
    let fused = fuse_strata(&frames, &params(8));
    let fused_energy = interior_energy(&fused);
    let truth_energy = interior_energy(&truth);
    let average_energy = interior_energy(&plain_average(&frames));
    assert!(
        fused_energy >= 0.35 * truth_energy,
        "fused {fused_energy} should keep 0.35x the truth's {truth_energy}"
    );
    assert!(
        fused_energy >= 4.0 * average_energy,
        "fused {fused_energy} should beat the plain average {average_energy} by 4x"
    );
}

#[test]
fn deep_stack_detail_focus_is_monotonic() {
    let (_, frames) = deep_stack();
    let energy = |detail_focus| {
        interior_energy(&fuse_strata(
            &frames,
            &StrataParams {
                base_radius: 8,
                detail_focus,
            },
        ))
    };
    let (e1, e3, e5) = (energy(1), energy(3), energy(5));
    assert!(
        e5 >= e3,
        "focus 5 ({e5}) should keep at least focus 3 ({e3})"
    );
    assert!(
        e3 >= e1,
        "focus 3 ({e3}) should keep at least focus 1 ({e1})"
    );
}
