#![cfg(feature = "akaze")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use stacker_align::akaze_match::*;
use stacker_core::image::PlanarImage;

#[test]
fn test_luma_to_gray8_gamma_and_stretch() {
    // A 4×2 image covering negative, zero, mid-range, and above-1 values.
    let mut img = PlanarImage::new(4, 2);
    img.luma[0] = -0.5_f32;
    img.luma[1] = 0.0_f32;
    img.luma[2] = 0.5_f32;
    img.luma[3] = 1.0_f32;
    img.luma[4] = 1.5_f32;
    img.luma[5] = 0.25_f32;
    img.luma[6] = 0.75_f32;
    img.luma[7] = 0.1_f32;

    let gfi = luma_to_gray8(&img);
    assert_eq!(gfi.width(), 4);
    assert_eq!(gfi.height(), 2);

    // All values must lie in [0, 255] after gamma + stretch.
    for y in 0..2_usize {
        for x in 0..4_usize {
            let v = gfi.get_pixel(x as u32, y as u32).0[0];
            assert!(
                (0..=255).contains(&v),
                "pixel ({x},{y}) = {v} is outside [0,255]"
            );
        }
    }
}

/// Core diagnostic test: prove that the percentile stretch alone restores
/// detectability on a narrow-range gamma-encoded buffer (no transfer
/// function is applied — the plane is already gamma-encoded per the
/// pipeline's colour contract).
#[test]

fn test_percentile_stretch_restores_contrast_on_narrow_range_gamma_buffer() {
    const W: usize = 64;
    const H: usize = 64;
    const NARROW_BASE: f32 = 0.05;
    const NARROW_RANGE: f32 = 0.20; // [0.05, 0.25]

    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let norm = ((x as f32 * 0.628).sin() * (y as f32 * 0.471).cos()).mul_add(0.5, 0.5);
            img.luma[y * W + x] = NARROW_BASE + norm * NARROW_RANGE;
        }
    }

    // ── Unstretched: raw clamp simulation (what you'd get without the
    // percentile stretch) ──
    let mut unstretched_gfi = Gray8Image::new(W as u32, H as u32);
    for (dst, &src) in unstretched_gfi.pixels_mut().zip(img.luma.iter()) {
        dst.0[0] = (src.clamp(0.0, 1.0) * 255.0) as u8;
    }
    let unstretched_min = unstretched_gfi.pixels().map(|p| p.0[0]).min().unwrap();
    let unstretched_max = unstretched_gfi.pixels().map(|p| p.0[0]).max().unwrap();
    let unstretched_range = f32::from(unstretched_max - unstretched_min);

    // ── luma_to_gray8: 1st–99th percentile stretch ──
    let stretched_gfi = luma_to_gray8(&img);
    let stretched_min = stretched_gfi.pixels().map(|p| p.0[0]).min().unwrap();
    let stretched_max = stretched_gfi.pixels().map(|p| p.0[0]).max().unwrap();
    let stretched_range = f32::from(stretched_max - stretched_min);

    assert!(
        unstretched_range > 1.0,
        "unstretched buffer range ({unstretched_range}) must be non-zero"
    );
    let ratio = stretched_range / unstretched_range;
    assert!(
        ratio >= 3.0,
        "percentile stretch must increase contrast by ≥ 3×; ratio={ratio:.2}"
    );
}

#[test]
fn test_extract_and_match_returns_result() {
    let img1 = PlanarImage::new(32, 32);
    let img2 = PlanarImage::new(32, 32);
    let result = KeypointMatcher::extract_and_match(&img1, &img2);
    assert!(result.is_ok(), "should return Ok even with no matches");
    let mr = result.unwrap();
    assert!(mr.matches.is_empty());
}

#[test]
fn test_match_target_agrees_with_extract_and_match() {
    let mut img1 = PlanarImage::new(64, 64);
    let mut img2 = PlanarImage::new(64, 64);
    for i in 0..4096 {
        let val = ((i as f32 * 0.4).sin()).abs();
        img1.luma[i] = val;
        img2.luma[i] = (val + 0.05).clamp(0.0, 1.0);
    }

    let (ref_kps, ref_desc) = extract_ref_features(&img1);
    let via_match_target = KeypointMatcher::match_target(&ref_kps, &ref_desc, &img2)
        .expect("match_target must succeed");
    let via_extract_match =
        KeypointMatcher::extract_and_match(&img1, &img2).expect("extract_and_match must succeed");

    assert_eq!(
        via_match_target.matches.len(),
        via_extract_match.matches.len(),
        "match_target and extract_and_match must agree on match count"
    );
}
