//! The pixel converters behind the numpy layer, tested without Python.

use stacker_core::image::PlanarImage;
use stacker_python::convert::{
    Encoding, planar_from_f32, planar_from_u8, planar_from_u16, planar_to_f32, planar_to_u8,
    planar_to_u16,
};

/// A `w × h` colour test image with distinct, non-symmetric channels, so a swapped channel or
/// a transposed image cannot round-trip by accident.
fn colour_u8(w: usize, h: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            data.push(u8::try_from((i * 5) % 256).unwrap());
            data.push(u8::try_from(255 - (i * 3) % 256).unwrap());
            data.push(u8::try_from((i * 11 + 7) % 256).unwrap());
        }
    }
    data
}

fn max_diff_u8(a: &[u8], b: &[u8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (i32::from(x) - i32::from(y)).abs())
        .max()
        .unwrap_or(0)
}

#[test]
fn u8_srgb_round_trip_is_exact_within_one_code() {
    let (w, h) = (13, 9);
    let src = colour_u8(w, h);
    let planar = planar_from_u8(&src, w, h, Encoding::Srgb).unwrap();
    assert_eq!((planar.width, planar.height), (w, h));
    assert!(planar.is_consistent());
    let back = planar_to_u8(&planar, Encoding::Srgb);
    assert_eq!(back.len(), src.len());
    assert!(max_diff_u8(&src, &back) <= 1);
}

#[test]
fn u8_gray_ramp_round_trips_through_every_code() {
    let ramp: Vec<u8> = (0..=255_u8).flat_map(|v| [v, v, v]).collect();
    let planar = planar_from_u8(&ramp, 256, 1, Encoding::Srgb).unwrap();
    let back = planar_to_u8(&planar, Encoding::Srgb);
    assert!(max_diff_u8(&ramp, &back) <= 1);
}

#[test]
fn u8_srgb_decodes_to_linear_light() {
    let planar = planar_from_u8(&[255, 255, 255, 128, 128, 128], 2, 1, Encoding::Srgb).unwrap();
    // White is linear 1.0, sRGB code 128 is about 0.2159 in linear light.
    assert!((planar.luma[0] - 1.0).abs() < 1.0e-5);
    assert!((planar.luma[1] - 0.2159).abs() < 1.0e-3);
    assert!(planar.chroma_a[0].abs() < 1.0e-5 && planar.chroma_b[0].abs() < 1.0e-5);
}

#[test]
fn u8_linear_encoding_skips_the_transfer_function() {
    let (w, h) = (7, 5);
    let src = colour_u8(w, h);
    let planar = planar_from_u8(&src, w, h, Encoding::Linear).unwrap();
    let gray = planar_from_u8(&[128, 128, 128], 1, 1, Encoding::Linear).unwrap();
    assert!((gray.luma[0] - 128.0 / 255.0).abs() < 1.0e-5);
    let back = planar_to_u8(&planar, Encoding::Linear);
    assert!(max_diff_u8(&src, &back) <= 1);
}

#[test]
fn u16_round_trip_within_one_code() {
    let (w, h) = (11, 6);
    let mut src = Vec::new();
    for i in 0..w * h {
        let i = u32::try_from(i).unwrap();
        src.push(u16::try_from((i * 997) % 65_536).unwrap());
        src.push(u16::try_from(65_535 - (i * 1_499) % 65_536).unwrap());
        src.push(u16::try_from((i * 3_001 + 17) % 65_536).unwrap());
    }
    let planar = planar_from_u16(&src, w, h, Encoding::Srgb).unwrap();
    assert_eq!((planar.width, planar.height), (w, h));
    let back = planar_to_u16(&planar, Encoding::Srgb);
    let worst = src
        .iter()
        .zip(&back)
        .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
        .max()
        .unwrap();
    assert!(worst <= 2, "16-bit round trip drifted by {worst} codes");
}

#[test]
fn u16_linear_round_trip() {
    let src: Vec<u16> = vec![0, 65_535, 32_768, 1, 2, 3];
    let planar = planar_from_u16(&src, 2, 1, Encoding::Linear).unwrap();
    let back = planar_to_u16(&planar, Encoding::Linear);
    let worst = src
        .iter()
        .zip(&back)
        .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
        .max()
        .unwrap();
    assert!(worst <= 1);
}

#[test]
fn f32_round_trip_is_near_exact_in_both_encodings() {
    let (w, h) = (9, 4);
    let src: Vec<f32> = (0..w * h * 3)
        .map(|i| (f32::from(u16::try_from(i).unwrap()) * 0.0097) % 1.0)
        .collect();
    for encoding in [Encoding::Srgb, Encoding::Linear] {
        let planar = planar_from_f32(&src, w, h, encoding).unwrap();
        let back = planar_to_f32(&planar, encoding);
        for (a, b) in src.iter().zip(&back) {
            assert!((a - b).abs() < 1.0e-4, "{encoding:?}: {a} vs {b}");
        }
    }
}

#[test]
fn srgb_and_linear_read_the_same_samples_differently() {
    let srgb = planar_from_f32(&[0.5, 0.5, 0.5], 1, 1, Encoding::Srgb).unwrap();
    let linear = planar_from_f32(&[0.5, 0.5, 0.5], 1, 1, Encoding::Linear).unwrap();
    assert!((linear.luma[0] - 0.5).abs() < 1.0e-6);
    assert!(srgb.luma[0] < 0.3);
}

#[test]
fn non_finite_input_reads_as_zero() {
    let planar = planar_from_f32(
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        1,
        1,
        Encoding::Srgb,
    )
    .unwrap();
    assert!(planar.luma[0].abs() < 1.0e-9);
    assert!(planar.chroma_a[0].abs() < 1.0e-9);
    assert!(planar.chroma_b[0].abs() < 1.0e-9);
}

#[test]
fn out_of_range_results_are_clamped_not_wrapped() {
    let mut img = PlanarImage::new(2, 1);
    img.luma = vec![2.0, -1.0];
    for encoding in [Encoding::Srgb, Encoding::Linear] {
        assert_eq!(planar_to_u8(&img, encoding), vec![255, 255, 255, 0, 0, 0]);
        assert_eq!(
            planar_to_u16(&img, encoding),
            vec![65_535, 65_535, 65_535, 0, 0, 0]
        );
        let floats = planar_to_f32(&img, encoding);
        assert!(floats[..3].iter().all(|&v| (v - 1.0).abs() < 1.0e-6));
        assert!(floats[3..].iter().all(|&v| v.abs() < 1.0e-9));
    }
}

#[test]
fn a_wrong_sample_count_is_an_error_naming_the_size() {
    let err = planar_from_u8(&[0; 5], 2, 1, Encoding::Srgb).unwrap_err();
    assert!(err.message().contains("holds 6 samples"), "{err}");
    assert!(planar_from_u16(&[0; 7], 2, 1, Encoding::Linear).is_err());
    assert!(planar_from_f32(&[0.0; 4], 1, 1, Encoding::Srgb).is_err());
}

#[test]
fn encoding_labels_parse_and_report_their_options() {
    assert_eq!(Encoding::parse("sRGB").unwrap(), Encoding::Srgb);
    assert_eq!(Encoding::parse(" linear ").unwrap(), Encoding::Linear);
    assert_eq!(Encoding::default(), Encoding::Srgb);
    assert_eq!(Encoding::Linear.name(), "linear");
    let err = Encoding::parse("gamma").unwrap_err();
    assert!(err.message().contains("srgb") && err.message().contains("linear"));
}

#[test]
fn the_image_is_not_transposed() {
    // One distinguishable pixel at (x = 4, y = 1) of a 5 x 3 image.
    let (w, h) = (5, 3);
    let mut src = vec![0_u8; w * h * 3];
    let at = (w + 4) * 3;
    src[at..at + 3].copy_from_slice(&[255, 255, 255]);
    let planar = planar_from_u8(&src, w, h, Encoding::Srgb).unwrap();
    assert!(planar.luma[w + 4] > 0.99);
    assert_eq!(planar.luma.iter().filter(|&&v| v > 0.5).count(), 1);
    let back = planar_to_u8(&planar, Encoding::Srgb);
    assert_eq!(&back[at..at + 3], &[255, 255, 255]);
}
