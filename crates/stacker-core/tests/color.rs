#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use stacker_core::color::*;

#[test]
fn test_srgb_decode_u8_table_exact() {
    let t = srgb_decode_u8_table();
    for (i, &val) in t.iter().enumerate() {
        let expected = srgb_to_linear(i as f32 / 255.0);
        assert!((val - expected).abs() < 1e-7, "u8 decode[{i}]");
    }
}

#[test]
fn test_srgb_decode_u16_table_exact() {
    let t = srgb_decode_u16_table();
    assert_eq!(t.len(), 65536);
    for &i in &[0u16, 1, 100, 12_345, 32_768, 65_534, 65_535] {
        let expected = srgb_to_linear(f32::from(i) / 65_535.0);
        assert!((t[i as usize] - expected).abs() < 1e-7, "u16 decode[{i}]");
    }
}

#[test]
fn test_encode_linear_to_srgb_u8_matches_scalar() {
    // 1003 values — deliberately not a multiple of the SIMD width so the
    // scalar tail path is exercised alongside the vector body.
    let src: Vec<f32> = (0..1003).map(|i| i as f32 / 1002.0).collect();
    let mut dst = vec![0u8; src.len()];
    encode_linear_to_srgb_u8(&src, &mut dst);
    for (i, &v) in src.iter().enumerate() {
        let expected = (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0).round() as u8;
        let diff = (i16::from(dst[i]) - i16::from(expected)).abs();
        assert!(
            diff <= 1,
            "encode[{i}] = {} (simd) vs {expected} (scalar)",
            dst[i]
        );
    }
}

#[test]
fn test_encode_clamps_out_of_range() {
    let src = [-1.0_f32, 0.0, 0.5, 1.0, 2.0];
    let mut dst = [0u8; 5];
    encode_linear_to_srgb_u8(&src, &mut dst);
    assert_eq!(dst[0], 0, "negative must clamp to black");
    assert_eq!(dst[4], 255, "values > 1 must clamp to white");
}

#[test]
fn test_srgb_to_linear() {
    assert_eq!(srgb_to_linear(0.0), 0.0);
    // f32 powf rounding: 1.055^2.4-style ops do not land on exactly 1.0.
    assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);

    // Test mid-range
    let v = 0.5;
    let linear = srgb_to_linear(v);
    assert!(linear > 0.0 && linear < 1.0);
    assert!((linear - 0.21404114).abs() < 1e-5);
}

#[test]
fn test_linear_to_srgb() {
    assert_eq!(linear_to_srgb(0.0), 0.0);
    // 1.055 * 1.0.powf(1/2.4) - 0.055 = 0.99999994 in f32, not exactly 1.0.
    assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);

    // Test mid-range
    let v = 0.21404114;
    let srgb = linear_to_srgb(v);
    assert!((srgb - 0.5).abs() < 1e-5);
}

#[test]
fn test_srgb_linear_roundtrip() {
    for i in 0..=100 {
        let v = i as f32 / 100.0;
        let linear = srgb_to_linear(v);
        let srgb = linear_to_srgb(linear);
        assert!((srgb - v).abs() < 1e-5);
    }
}

#[test]
fn test_rgb_to_oklab() {
    let rgb = [1.0, 0.0, 0.0];
    let oklab = rgb_to_oklab(rgb);

    // Red in OKLAB roughly: L: ~0.6, a: ~0.2, b: ~0.1
    assert!(oklab[0] > 0.0);
    assert!(oklab[1] > 0.0);

    // Black
    let black = [0.0, 0.0, 0.0];
    let black_oklab = rgb_to_oklab(black);
    assert_eq!(black_oklab[0], 0.0);
    assert_eq!(black_oklab[1], 0.0);
    assert_eq!(black_oklab[2], 0.0);

    // White
    let white = [1.0, 1.0, 1.0];
    let white_oklab = rgb_to_oklab(white);
    assert!((white_oklab[0] - 1.0).abs() < 1e-5);
    assert!(white_oklab[1].abs() < 1e-5);
    assert!(white_oklab[2].abs() < 1e-5);
}
