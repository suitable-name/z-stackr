#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the sweeps index small integer ranges and the quantised values are clamped to the code range first"
)]

use stacker_core::color::*;

/// Independent f64 evaluation of the IEC 61966-2-1 decode, so the tables are
/// compared with a high-precision computation and not with the `f32`
/// expression that built them.
fn srgb_to_linear_f64(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Independent f64 evaluation of the IEC 61966-2-1 encode.
fn linear_to_srgb_f64(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055_f64.mul_add(v.powf(1.0 / 2.4), -0.055)
    }
}

/// Largest acceptable distance between an `f32` table entry and the f64 value:
/// about eight `f32` ulps at 1.0, three orders of magnitude below one 8-bit code.
const TABLE_TOLERANCE: f64 = 1e-6;

#[test]
fn test_srgb_decode_u8_table_exact() {
    let t = srgb_decode_u8_table();
    assert_eq!(
        t[0].to_bits(),
        0.0_f32.to_bits(),
        "code 0 decodes to exactly 0"
    );
    assert!(
        (f64::from(t[255]) - 1.0).abs() < TABLE_TOLERANCE,
        "code 255 decodes to 1"
    );
    for (i, &val) in t.iter().enumerate() {
        let expected = srgb_to_linear_f64(i as f64 / 255.0);
        assert!(
            (f64::from(val) - expected).abs() < TABLE_TOLERANCE,
            "u8 decode[{i}] = {val}, f64 reference {expected}"
        );
    }
    assert!(
        t.windows(2).all(|w| w[0] < w[1]),
        "the decode table is strictly increasing"
    );
}

#[test]
fn test_srgb_decode_u16_table_exact() {
    let t = srgb_decode_u16_table();
    assert_eq!(t.len(), 65536);
    assert_eq!(
        t[0].to_bits(),
        0.0_f32.to_bits(),
        "code 0 decodes to exactly 0"
    );
    for (i, &val) in t.iter().enumerate() {
        let expected = srgb_to_linear_f64(i as f64 / 65_535.0);
        assert!(
            (f64::from(val) - expected).abs() < TABLE_TOLERANCE,
            "u16 decode[{i}] = {val}, f64 reference {expected}"
        );
    }
    // Every 257th 16-bit code is an 8-bit code replicated into both bytes, so the
    // two tables must agree there.
    let t8 = srgb_decode_u8_table();
    for (code, &val8) in t8.iter().enumerate() {
        assert!(
            (f64::from(t[code * 257]) - f64::from(val8)).abs() < TABLE_TOLERANCE,
            "u16[{}] must equal u8[{code}]",
            code * 257
        );
    }
}

/// The exact 8-bit code for `v` and whether `v` sits within 1e-4 of a rounding
/// boundary.
///
/// The reference is evaluated in f64 on the exact `f32` input. The row encoder
/// compares against thresholds that are themselves `f32` values, which can
/// misplace a sample by about 1.5e-5 of a code at most, so 1e-4 is the width in
/// which a one-code disagreement is legitimate; everywhere else the result must
/// be exact.
fn exact_u8(v: f32) -> (u8, bool) {
    let x = linear_to_srgb_f64(f64::from(v.clamp(0.0, 1.0))) * 255.0;
    (x.round() as u8, (x.fract() - 0.5).abs() < 1e-4)
}

/// Assert `got` equals the exact code unless `v` is on a boundary, where it may be
/// one code off.
fn assert_exact_u8(i: usize, v: f32, got: u8) {
    let (expected, on_boundary) = exact_u8(v);
    if on_boundary {
        assert!(
            (i16::from(got) - i16::from(expected)).abs() <= 1,
            "encode[{i}] v={v}: {got} vs {expected} (boundary)"
        );
    } else {
        assert_eq!(got, expected, "encode[{i}] v={v}");
    }
}

#[test]
fn test_encode_row_u8_matches_scalar() {
    // 1003 values — deliberately not a multiple of the SIMD width so the
    // scalar tail path is exercised alongside the vector body.
    let src: Vec<f32> = (0..1003).map(|i| i as f32 / 1002.0).collect();
    let mut dst = vec![0u8; src.len()];
    encode_row_u8(&src, &mut dst);
    for (i, &v) in src.iter().enumerate() {
        assert_exact_u8(i, v, dst[i]);
    }
}

/// CORE-1: white must reach the top code. `linear_to_srgb(1.0)` is
/// 0.99999994 in f32, so a truncating quantiser gives 254 / 65534.
#[test]
fn test_white_reaches_255_and_65535() {
    assert_eq!(linear_to_srgb_u8(1.0), 255);
    assert_eq!(linear_to_srgb_u16(1.0), 65_535);
    assert_eq!(linear_to_srgb_u8(0.0), 0);
    assert_eq!(linear_to_srgb_u16(0.0), 0);
    // Out of range clamps, NaN is black.
    assert_eq!(linear_to_srgb_u8(7.0), 255);
    assert_eq!(linear_to_srgb_u16(-3.0), 0);
    assert_eq!(linear_to_srgb_u8(f32::NAN), 0);
    assert_eq!(linear_to_srgb_u16(f32::NAN), 0);

    // Row encoders, with lengths that exercise both the vector body and the tail.
    for len in [1usize, 15, 16, 17, 37, 64] {
        let white = vec![1.0_f32; len];
        let mut out8 = vec![0u8; len];
        let mut out16 = vec![0u16; len];
        encode_row_u8(&white, &mut out8);
        encode_row_u16(&white, &mut out16);
        assert!(out8.iter().all(|&c| c == 255), "u8 white, len {len}");
        assert!(out16.iter().all(|&c| c == 65_535), "u16 white, len {len}");

        let nan = vec![f32::NAN; len];
        encode_row_u8(&nan, &mut out8);
        assert!(out8.iter().all(|&c| c == 0), "u8 NaN, len {len}");
    }
}

/// The scalar helpers round: a value half a code above a boundary lands on the
/// upper code, which truncation cannot do.
#[test]
fn test_scalar_quantisers_round() {
    for i in 0..=1000 {
        let v = i as f32 / 1000.0;
        let x8 = linear_to_srgb(v) * 255.0;
        assert_eq!(linear_to_srgb_u8(v), x8.round() as u8, "u8 at {v}");
        let x16 = linear_to_srgb(v) * 65_535.0;
        assert_eq!(linear_to_srgb_u16(v), x16.round() as u16, "u16 at {v}");
    }
}

/// CORE-5: the row encoder must never return a code below the exact rounded
/// code (a truncated table index lands one below it in about 1.6 % of the cells
/// near black).
#[test]
fn test_encode_row_never_below_exact_code() {
    // Fine sweep of the dark range where one table cell spans < 1 code,
    // then the full range. Lengths are not multiples of the SIMD width.
    let dark: Vec<f32> = (0..=100_000).map(|i| i as f32 * 1e-7).collect();
    let full: Vec<f32> = (0..=200_001).map(|i| i as f32 / 200_001.0).collect();
    for src in [dark, full] {
        let mut dst = vec![0u8; src.len()];
        encode_row_u8(&src, &mut dst);
        for (i, (&v, &got)) in src.iter().zip(&dst).enumerate() {
            assert_exact_u8(i, v, got);
            let (expected, on_boundary) = exact_u8(v);
            assert!(
                on_boundary || got >= expected,
                "encode[{i}] v={v}: {got} is below the exact code {expected}"
            );
        }
    }
}

/// CORE-5 regression point: v = 0.000152 is just above the code 0/1 boundary
/// (0.5/255 in sRGB, 0.00015176 linear) but its truncated table index (2) holds
/// code 0.
#[test]
fn test_encode_row_just_above_first_boundary() {
    let src = [0.000_152_f32, 0.000_151_f32, 2.0 / 16_383.0 + 1e-6];
    let mut dst = [0u8; 3];
    encode_row_u8(&src, &mut dst);
    assert_eq!(dst[0], 1, "just above the boundary must round up");
    assert_eq!(dst[1], 0, "just below the boundary must stay 0");
    assert_eq!(dst[2], exact_u8(src[2]).0);
}

#[test]
fn test_encode_row_u16_matches_scalar() {
    let src: Vec<f32> = (0..=1000).map(|i| i as f32 / 1000.0).collect();
    let mut dst = vec![0u16; src.len()];
    encode_row_u16(&src, &mut dst);
    for (&v, &got) in src.iter().zip(&dst) {
        assert_eq!(got, linear_to_srgb_u16(v));
    }
    assert_eq!(dst[0], 0);
    assert_eq!(dst[1000], 65_535);
}

#[test]
fn test_encode_clamps_out_of_range() {
    let src = [-1.0_f32, 0.0, 0.5, 1.0, 2.0];
    let mut dst = [0u8; 5];
    encode_row_u8(&src, &mut dst);
    assert_eq!(dst[0], 0, "negative must clamp to black");
    assert_eq!(dst[4], 255, "values > 1 must clamp to white");
}

#[test]
fn test_srgb_to_linear() {
    assert_eq!(srgb_to_linear(0.0).to_bits(), 0.0_f32.to_bits());
    // f32 powf rounding: 1.055^2.4-style ops do not land on exactly 1.0.
    assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);

    // Test mid-range
    let v = 0.5;
    let linear = srgb_to_linear(v);
    assert!(linear > 0.0 && linear < 1.0);
    assert!((linear - 0.214_041_14).abs() < 1e-5);
}

#[test]
fn test_linear_to_srgb() {
    assert_eq!(linear_to_srgb(0.0).to_bits(), 0.0_f32.to_bits());
    // 1.055 * 1.0.powf(1/2.4) - 0.055 = 0.99999994 in f32, not exactly 1.0.
    assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);

    // Test mid-range
    let v = 0.214_041_14;
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
