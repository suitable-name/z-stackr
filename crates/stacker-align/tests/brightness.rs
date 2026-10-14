//! Tests of the per-frame brightness correction (linear light).

#![allow(
    clippy::cast_precision_loss,
    reason = "pixel indices in the fixtures are far below 2^24"
)]
#![allow(
    clippy::suboptimal_flops,
    clippy::many_single_char_names,
    reason = "plain fixture arithmetic; the serial reference formula mirrors the production one"
)]

use stacker_align::brightness::{
    BrightnessFit, BrightnessTarget, correct_brightness, correct_brightness_masked,
};
use stacker_core::image::PlanarImage;

const W: usize = 64;
const H: usize = 48;

/// A deterministic value in `[0, 1)` for pixel `index` (`SplitMix64` finaliser).
fn hash01(index: usize) -> f32 {
    let mut z = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let top = u16::try_from(z >> 48).unwrap_or(0);
    f32::from(top) / 65_536.0
}

/// A textured linear-light reference: luma in `[0.1, 0.9]` with a skewed
/// histogram, chroma proportional to the luma so the colour is constant.
fn reference() -> PlanarImage<f32> {
    let mut img = PlanarImage::<f32>::new(W, H);
    for i in 0..W * H {
        let u = hash01(i);
        let luma = 0.1 + 0.8 * u * u;
        img.luma[i] = luma;
        img.chroma_a[i] = 0.2 * luma;
        img.chroma_b[i] = -0.1 * luma;
    }
    img
}

/// The reference with its luma mapped through `f` (chroma scaled by the ratio).
fn mapped(reference: &PlanarImage<f32>, f: impl Fn(f32) -> f32) -> PlanarImage<f32> {
    let mut img = reference.clone();
    for i in 0..img.luma.len() {
        let y = reference.luma[i];
        let out = f(y);
        img.luma[i] = out;
        img.chroma_a[i] *= out / y;
        img.chroma_b[i] *= out / y;
    }
    img
}

/// Largest relative luma difference over `pixels` (all when `None`).
fn max_rel_err(a: &PlanarImage<f32>, b: &PlanarImage<f32>, keep: impl Fn(usize) -> bool) -> f32 {
    (0..a.luma.len())
        .filter(|&i| keep(i))
        .map(|i| (a.luma[i] - b.luma[i]).abs() / b.luma[i].abs().max(1.0e-6))
        .fold(0.0, f32::max)
}

#[test]
fn exposure_scaled_by_0_8_is_restored_within_one_percent() {
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let mut frame = mapped(&reference, |y| y * 0.8);
    let fit = correct_brightness(&mut frame, &target);
    assert!((fit.scale - 1.25).abs() < 0.0125, "scale {}", fit.scale);
    assert!((fit.gamma - 1.0).abs() < 0.02, "gamma {}", fit.gamma);
    let err = max_rel_err(&frame, &reference, |_| true);
    assert!(err < 0.01, "max relative luma error {err}");
    // Chroma follows the luma ratio, so the colour of a pixel is unchanged.
    for i in 0..W * H {
        let want = 0.2 * reference.luma[i];
        assert!(
            (frame.chroma_a[i] - want).abs() < 0.01 * want.abs().max(0.01),
            "chroma_a at {i}: {} vs {want}",
            frame.chroma_a[i]
        );
    }
}

#[test]
fn brighter_frame_is_darkened() {
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let mut frame = mapped(&reference, |y| (y * 1.15).min(1.0e3));
    let fit = correct_brightness(&mut frame, &target);
    assert!(fit.scale < 1.0, "scale {}", fit.scale);
    let err = max_rel_err(&frame, &reference, |_| true);
    assert!(err < 0.01, "max relative luma error {err}");
}

#[test]
fn gamma_shifted_frame_is_restored() {
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let mut frame = mapped(&reference, |y| y.powf(1.2));
    let fit = correct_brightness(&mut frame, &target);
    assert!(
        (fit.gamma - 1.0 / 1.2).abs() < 0.03,
        "fitted gamma {}",
        fit.gamma
    );
    let err = max_rel_err(&frame, &reference, |_| true);
    assert!(err < 0.02, "max relative luma error {err}");
}

#[test]
fn equal_frames_are_left_bit_identical() {
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let mut frame = reference.clone();
    let fit = correct_brightness(&mut frame, &target);
    assert!(fit.is_identity(), "{fit:?}");
    assert_eq!(fit, BrightnessFit::IDENTITY);
    assert_eq!(frame.luma, reference.luma);
    assert_eq!(frame.chroma_a, reference.chroma_a);
    assert_eq!(frame.chroma_b, reference.chroma_b);
}

#[test]
fn target_reports_mean_and_deviation() {
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    assert!(target.is_valid());
    let n = reference.luma.len() as f64;
    let mean = reference.luma.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    assert!((target.mean() - mean).abs() < 1.0e-6, "{}", target.mean());
    assert!(target.std_dev() > 0.05);
}

/// A bright, wrong border that the warp coverage mask excludes must not bias
/// the fit of the interior.
#[test]
fn masked_border_is_ignored() {
    let reference = reference();
    let border = 8;
    let inside = |i: usize| {
        let (x, y) = (i % W, i / W);
        x >= border && x < W - border && y >= border && y < H - border
    };
    // The target covers the same region, so the interior statistics match
    // exactly and the test measures only the effect of the border.
    let region: Vec<bool> = (0..W * H).map(inside).collect();
    let target = BrightnessTarget::from_reference_masked(&reference, Some(&region));
    let mut frame = mapped(&reference, |y| y * 0.8);
    for i in 0..W * H {
        if !inside(i) {
            frame.luma[i] = 0.99;
            frame.chroma_a[i] = 0.0;
            frame.chroma_b[i] = 0.0;
        }
    }
    let coverage: Vec<bool> = (0..W * H).map(inside).collect();
    let mut masked = frame.clone();
    let fit = correct_brightness_masked(&mut masked, &target, Some(&coverage));
    assert!((fit.scale - 1.25).abs() < 0.03, "scale {}", fit.scale);
    let err = max_rel_err(&masked, &reference, inside);
    assert!(err < 0.02, "interior error with mask {err}");

    // Without the mask the border pulls the fit away.
    let mut unmasked = frame;
    correct_brightness(&mut unmasked, &target);
    let err_unmasked = max_rel_err(&unmasked, &reference, inside);
    assert!(
        err_unmasked > err,
        "the mask should help: {err_unmasked} vs {err}"
    );
}

/// Zero-filled out-of-frame pixels (what a warp leaves) are not counted even
/// without a mask, and stay exactly zero.
#[test]
fn zero_border_does_not_bias_the_fit() {
    let reference = reference();
    let border = 8;
    let inside = |i: usize| {
        let (x, y) = (i % W, i / W);
        x >= border && x < W - border && y >= border && y < H - border
    };
    let region: Vec<bool> = (0..W * H).map(inside).collect();
    let target = BrightnessTarget::from_reference_masked(&reference, Some(&region));
    let mut frame = mapped(&reference, |y| y * 0.8);
    for i in 0..W * H {
        if !inside(i) {
            frame.luma[i] = 0.0;
            frame.chroma_a[i] = 0.0;
            frame.chroma_b[i] = 0.0;
        }
    }
    correct_brightness(&mut frame, &target);
    let err = max_rel_err(&frame, &reference, inside);
    assert!(err < 0.02, "interior error {err}");
    for i in (0..W * H).filter(|&i| !inside(i)) {
        assert_eq!(frame.luma[i].to_bits(), 0.0_f32.to_bits());
        assert_eq!(frame.chroma_a[i].to_bits(), 0.0_f32.to_bits());
    }
}

#[test]
fn black_frames_stay_finite_and_unchanged() {
    // Black frame against a real target: nothing to fit.
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let mut black = PlanarImage::<f32>::new(W, H);
    let fit = correct_brightness(&mut black, &target);
    assert!(fit.is_identity());
    assert!(black.luma.iter().all(|&v| v == 0.0));

    // Black reference: the target is invalid and every correction is skipped.
    let black_target = BrightnessTarget::from_reference(&PlanarImage::<f32>::new(W, H));
    assert!(!black_target.is_valid());
    let mut frame = reference.clone();
    let fit = correct_brightness(&mut frame, &black_target);
    assert!(fit.is_identity());
    assert_eq!(frame.luma, reference.luma);

    // Everything masked out behaves like a black reference.
    let none = vec![false; W * H];
    let masked_target = BrightnessTarget::from_reference_masked(&reference, Some(&none));
    assert!(!masked_target.is_valid());
}

#[test]
fn non_finite_pixels_are_cleaned_when_a_correction_applies() {
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let mut frame = mapped(&reference, |y| y * 0.8);
    frame.luma[5] = f32::NAN;
    frame.chroma_a[5] = f32::NAN;
    frame.luma[6] = f32::INFINITY;
    correct_brightness(&mut frame, &target);
    assert!(frame.luma.iter().all(|v| v.is_finite()));
    assert!(frame.chroma_a.iter().all(|v| v.is_finite()));
    assert!(frame.chroma_b.iter().all(|v| v.is_finite()));
    assert_eq!(frame.luma[5].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn correction_is_deterministic() {
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let run = || {
        let mut frame = mapped(&reference, |y| y * 0.7);
        let fit = correct_brightness(&mut frame, &target);
        (fit, frame.luma)
    };
    let (fit_a, luma_a) = run();
    let (fit_b, luma_b) = run();
    assert_eq!(fit_a, fit_b);
    assert_eq!(
        luma_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        luma_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// The parallel apply (several work chunks) equals the serial per-pixel formula
/// bit for bit, including the cleaned non-finite and zero pixels.
#[test]
fn parallel_apply_equals_serial_formula() {
    let (w, h) = (200, 100);
    let mut base = PlanarImage::<f32>::new(w, h);
    for i in 0..w * h {
        let luma = 0.05 + 0.9 * hash01(i);
        base.luma[i] = luma;
        base.chroma_a[i] = 0.3 * luma - 0.01;
        base.chroma_b[i] = -0.2 * luma + 0.02;
    }
    base.luma[3] = 0.0;
    base.luma[9000] = f32::NAN;
    base.luma[15_000] = 5.0e-7;
    let reference = reference();
    let target = BrightnessTarget::from_reference(&reference);
    let original = base.clone();
    let mut frame = base;
    let fit = correct_brightness(&mut frame, &target);
    assert!(!fit.is_identity(), "the fixture must trigger a correction");
    for i in 0..w * h {
        let y = original.luma[i];
        let (luma, a, b) = if y.is_finite() && y > 0.0 {
            let out = (y * fit.scale).powf(fit.gamma);
            let ratio = if y > 1.0e-6 { out / y } else { 1.0 };
            (
                out,
                original.chroma_a[i] * ratio,
                original.chroma_b[i] * ratio,
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        assert_eq!(frame.luma[i].to_bits(), luma.to_bits(), "luma {i}");
        assert_eq!(frame.chroma_a[i].to_bits(), a.to_bits(), "chroma_a {i}");
        assert_eq!(frame.chroma_b[i].to_bits(), b.to_bits(), "chroma_b {i}");
    }
}

/// A mask of the wrong length is ignored instead of panicking.
#[test]
fn mismatched_mask_is_ignored() {
    let reference = reference();
    let target = BrightnessTarget::from_reference_masked(&reference, Some(&[true; 3]));
    assert_eq!(target, BrightnessTarget::from_reference(&reference));
}
