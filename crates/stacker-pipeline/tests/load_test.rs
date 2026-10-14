#![expect(
    clippy::suboptimal_flops,
    reason = "synthetic-image formulas read better without mul_add"
)]

//! `load::decode_frame`: one decoder, every preprocessing key.

use std::path::PathBuf;

use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use stacker_core::settings::PreprocessingSettings;
use stacker_pipeline::{
    PipelineError,
    load::{
        RAW_EXTENSIONS, RAW_FRAME_EXTENSIONS, RAW_SUPPORTED, apply_orientation, decode_frame,
        decode_frames, has_raw_extension, jpeg_exif_orientation, parse_crop_spec,
        planar_from_dynamic,
    },
};

fn dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stacker_pipeline_load_{tag}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A frame whose pixel at (x, y) encodes its position, so rotations and crops
/// are visible in the samples.
fn marked_png(path: &std::path::Path, w: u32, h: u32) {
    RgbImage::from_fn(w, h, |x, y| {
        Rgb([
            u8::try_from(x * 3).unwrap(),
            u8::try_from(y * 3).unwrap(),
            200,
        ])
    })
    .save(path)
    .unwrap();
}

#[test]
fn defaults_leave_the_frame_untouched_and_linearise_it() {
    let d = dir("defaults");
    let path = d.join("a.png");
    marked_png(&path, 20, 10);

    let planar = decode_frame(&path, &PreprocessingSettings::default()).unwrap();
    assert_eq!((planar.width, planar.height), (20, 10));
    assert!(planar.is_consistent());
    // Black-blue-ish pixel (0, 0): r = g = 0, b = 200 -> linear blue is
    // srgb_to_linear(200/255), and luma is 0.114 of it.
    let blue = stacker_core::color::srgb_to_linear(200.0 / 255.0);
    assert!(
        (planar.luma[0] - 0.114 * blue).abs() < 1e-6,
        "{}",
        planar.luma[0]
    );
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn rotation_swaps_the_axes_and_moves_the_samples() {
    let d = dir("rotation");
    let path = d.join("a.png");
    marked_png(&path, 20, 10);
    let plain = decode_frame(&path, &PreprocessingSettings::default()).unwrap();

    for (degrees, dims) in [(90, (10, 20)), (180, (20, 10)), (270, (10, 20))] {
        let pre = PreprocessingSettings {
            pre_rotation: degrees,
            ..PreprocessingSettings::default()
        };
        let rotated = decode_frame(&path, &pre).unwrap();
        assert_eq!((rotated.width, rotated.height), dims, "{degrees} deg");
    }

    // 90 degrees clockwise: the top-left pixel of the source ends up top-right.
    let pre = PreprocessingSettings {
        pre_rotation: 90,
        ..PreprocessingSettings::default()
    };
    let rotated = decode_frame(&path, &pre).unwrap();
    assert!((rotated.luma[rotated.width - 1] - plain.luma[0]).abs() < 1e-6);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn crop_spec_selects_the_region_and_a_bad_spec_is_skipped() {
    let d = dir("crop");
    let path = d.join("a.png");
    marked_png(&path, 40, 30);

    let centred = PreprocessingSettings {
        pre_crop_enabled: true,
        pre_crop_spec: "20,10".to_owned(),
        ..PreprocessingSettings::default()
    };
    let c = decode_frame(&path, &centred).unwrap();
    assert_eq!((c.width, c.height), (20, 10));

    // Explicit offset: pixel (0, 0) of the crop is pixel (6, 4) of the source,
    // whose red channel encodes x = 6.
    let offset = PreprocessingSettings {
        pre_crop_enabled: true,
        pre_crop_spec: "6,4,10,8".to_owned(),
        ..PreprocessingSettings::default()
    };
    let o = decode_frame(&path, &offset).unwrap();
    assert_eq!((o.width, o.height), (10, 8));
    let source = decode_frame(&path, &PreprocessingSettings::default()).unwrap();
    assert!((o.luma[0] - source.luma[4 * 40 + 6]).abs() < 1e-6);

    // Disabled crop and unusable specs leave the frame whole.
    for (enabled, spec) in [(false, "20,10"), (true, "nonsense"), (true, "0,0")] {
        let pre = PreprocessingSettings {
            pre_crop_enabled: enabled,
            pre_crop_spec: spec.to_owned(),
            ..PreprocessingSettings::default()
        };
        let whole = decode_frame(&path, &pre).unwrap();
        assert_eq!((whole.width, whole.height), (40, 30), "{enabled} {spec}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn resize_percent_scales_both_axes() {
    let d = dir("resize");
    let path = d.join("a.png");
    marked_png(&path, 40, 30);
    for (percent, dims) in [(100, (40, 30)), (50, (20, 15)), (10, (4, 3))] {
        let pre = PreprocessingSettings {
            pre_resize_percent: percent,
            ..PreprocessingSettings::default()
        };
        let r = decode_frame(&path, &pre).unwrap();
        assert_eq!((r.width, r.height), dims, "{percent} %");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn rotation_crop_and_resize_apply_in_that_order() {
    let d = dir("order");
    let path = d.join("a.png");
    marked_png(&path, 80, 40);
    let pre = PreprocessingSettings {
        pre_rotation: 90, // 80x40 -> 40x80
        pre_crop_enabled: true,
        pre_crop_spec: "40,40".to_owned(), // centred 40x40 of the rotated frame
        pre_resize_percent: 50,            // -> 20x20
        ..PreprocessingSettings::default()
    };
    let r = decode_frame(&path, &pre).unwrap();
    assert_eq!((r.width, r.height), (20, 20));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn sixteen_bit_precision_survives_decoding() {
    let d = dir("deep");
    let path = d.join("deep.png");
    let mut buf: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(2, 1);
    buf.put_pixel(0, 0, Rgb([1000, 1000, 1000]));
    buf.put_pixel(1, 0, Rgb([1010, 1010, 1010]));
    buf.save(&path).unwrap();
    let planar = decode_frame(&path, &PreprocessingSettings::default()).unwrap();
    assert!(
        planar.luma[1] > planar.luma[0],
        "two codes inside one 8-bit bucket must stay distinct"
    );
    let _ = std::fs::remove_dir_all(&d);
}

/// An APP1 EXIF segment with orientation `orientation`, inserted right after
/// the JPEG start marker.
fn jpeg_with_orientation(w: u32, h: u32, orientation: u16) -> Vec<u8> {
    let mut plain = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut plain, 90)
        .encode_image(&RgbImage::from_fn(w, h, |x, y| {
            Rgb([
                u8::try_from(x * 3).unwrap(),
                u8::try_from(y * 3).unwrap(),
                90,
            ])
        }))
        .unwrap();

    // TIFF block, little endian: header, one IFD0 entry (orientation, SHORT).
    let mut tiff = vec![b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0];
    tiff.extend_from_slice(&0x0112_u16.to_le_bytes());
    tiff.extend_from_slice(&3_u16.to_le_bytes());
    tiff.extend_from_slice(&1_u32.to_le_bytes());
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0, 0]);
    tiff.extend_from_slice(&0_u32.to_le_bytes());

    let mut segment = b"Exif\0\0".to_vec();
    segment.extend_from_slice(&tiff);
    let len = u16::try_from(segment.len() + 2).unwrap();

    let mut out = vec![0xFF, 0xD8, 0xFF, 0xE1];
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&segment);
    out.extend_from_slice(&plain[2..]);
    out
}

#[test]
fn exif_orientation_is_applied_unless_ignored() {
    let d = dir("exif");
    let path = d.join("tilted.jpg");
    std::fs::write(&path, jpeg_with_orientation(32, 16, 6)).unwrap();

    // Orientation 6 means "rotate 90 degrees clockwise to display".
    let upright = decode_frame(&path, &PreprocessingSettings::default()).unwrap();
    assert_eq!((upright.width, upright.height), (16, 32));

    let ignored = PreprocessingSettings {
        ignore_exif_orientation: true,
        ..PreprocessingSettings::default()
    };
    let raw = decode_frame(&path, &ignored).unwrap();
    assert_eq!((raw.width, raw.height), (32, 16));

    // Orientation 1 (upright) and a JPEG without EXIF are left alone.
    let upright_file = d.join("upright.jpg");
    std::fs::write(&upright_file, jpeg_with_orientation(32, 16, 1)).unwrap();
    let still = decode_frame(&upright_file, &PreprocessingSettings::default()).unwrap();
    assert_eq!((still.width, still.height), (32, 16));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_exif_reader_handles_both_byte_orders_and_garbage() {
    let little = jpeg_with_orientation(8, 8, 8);
    assert_eq!(jpeg_exif_orientation(&little), Some(8));

    // Big-endian TIFF block.
    let mut tiff = vec![b'M', b'M', 0, 42, 0, 0, 0, 8, 0, 1];
    tiff.extend_from_slice(&0x0112_u16.to_be_bytes());
    tiff.extend_from_slice(&3_u16.to_be_bytes());
    tiff.extend_from_slice(&1_u32.to_be_bytes());
    tiff.extend_from_slice(&3_u16.to_be_bytes());
    tiff.extend_from_slice(&[0, 0]);
    let mut seg = b"Exif\0\0".to_vec();
    seg.extend_from_slice(&tiff);
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
    jpeg.extend_from_slice(&u16::try_from(seg.len() + 2).unwrap().to_be_bytes());
    jpeg.extend_from_slice(&seg);
    assert_eq!(jpeg_exif_orientation(&jpeg), Some(3));

    // Not a JPEG, truncated, orientation out of range: no answer, no panic.
    assert_eq!(jpeg_exif_orientation(b"\x89PNG\r\n"), None);
    assert_eq!(jpeg_exif_orientation(&little[..12]), None);
    assert_eq!(jpeg_exif_orientation(&jpeg_with_orientation(8, 8, 9)), None);
    assert_eq!(jpeg_exif_orientation(&[]), None);
}

#[test]
fn every_orientation_maps_a_corner_where_the_exif_table_says() {
    // A 3x2 image with a unique value per pixel; the red channel is the index.
    let img = DynamicImage::ImageRgb8(RgbImage::from_fn(3, 2, |x, y| {
        Rgb([u8::try_from(y * 3 + x).unwrap(), 0, 0])
    }));
    let cell = |o: u8| -> Vec<u8> {
        apply_orientation(img.clone(), o)
            .to_rgb8()
            .pixels()
            .map(|p| p[0])
            .collect()
    };
    assert_eq!(cell(1), [0, 1, 2, 3, 4, 5]);
    assert_eq!(cell(2), [2, 1, 0, 5, 4, 3]); // mirrored horizontally
    assert_eq!(cell(3), [5, 4, 3, 2, 1, 0]); // rotated 180
    assert_eq!(cell(4), [3, 4, 5, 0, 1, 2]); // mirrored vertically
    assert_eq!(cell(5), [0, 3, 1, 4, 2, 5]); // transposed
    assert_eq!(cell(6), [3, 0, 4, 1, 5, 2]); // 90 clockwise
    assert_eq!(cell(7), [5, 2, 4, 1, 3, 0]); // anti-transposed
    assert_eq!(cell(8), [2, 5, 1, 4, 0, 3]); // 270 clockwise
}

#[test]
fn an_undecodable_file_is_a_decode_error_naming_it() {
    let d = dir("broken");
    let path = d.join("broken.png");
    std::fs::write(&path, b"this is not a png").unwrap();
    let err = decode_frame(&path, &PreprocessingSettings::default()).unwrap_err();
    assert!(matches!(err, PipelineError::Decode { .. }), "{err:?}");
    assert!(err.to_string().contains("broken.png"), "{err}");

    // `decode_frames` keeps the slots aligned with the paths.
    let good = d.join("good.png");
    marked_png(&good, 6, 4);
    let slots = decode_frames(&[good, path], &PreprocessingSettings::default());
    assert!(slots[0].is_ok());
    assert!(slots[1].is_err());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn crop_spec_parser_forms() {
    assert_eq!(parse_crop_spec("40,20", 100, 60), Some((30, 20, 40, 20)));
    assert_eq!(
        parse_crop_spec("10,5,200,200", 100, 60),
        Some((10, 5, 90, 55))
    );
    assert_eq!(parse_crop_spec("0,0", 100, 60), None);
    assert_eq!(parse_crop_spec("200,0,10,10", 100, 60), None);
    assert_eq!(parse_crop_spec("garbage", 100, 60), None);
}

/// The frame extensions a build adds for RAW: all of them with RAW support,
/// none without (a scan must not list files the build cannot decode).
#[test]
fn raw_frame_extensions_follow_the_build() {
    if RAW_SUPPORTED {
        assert_eq!(RAW_FRAME_EXTENSIONS, RAW_EXTENSIONS);
        assert!(RAW_FRAME_EXTENSIONS.contains(&"nef") && RAW_FRAME_EXTENSIONS.contains(&"dng"));
    } else {
        assert_eq!(RAW_FRAME_EXTENSIONS, &[] as &[&str]);
    }
    // Recognising a RAW name does not depend on the build.
    assert!(has_raw_extension(std::path::Path::new("IMG_1.CR3")));
    assert!(!has_raw_extension(std::path::Path::new("IMG_1.jpg")));
}

/// A RAW path never reaches the `image` crate: it is a decode error naming the
/// file, and without RAW support the message says how to turn RAW on.
#[test]
fn a_raw_file_that_cannot_be_decoded_is_a_decode_error() {
    let d = dir("raw");
    let path = d.join("shot.NEF");
    std::fs::write(&path, b"this is not a RAW file").unwrap();
    let err = decode_frame(&path, &PreprocessingSettings::default()).unwrap_err();
    assert!(matches!(err, PipelineError::Decode { .. }), "{err:?}");
    let message = err.to_string();
    assert!(message.contains("shot.NEF"), "{message}");
    if !RAW_SUPPORTED {
        assert!(message.contains("--features raw"), "{message}");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn planar_conversion_of_a_dynamic_image_matches_the_core_matrix() {
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([255, 128, 0])));
    let planar = planar_from_dynamic(&img);
    let lin = [1.0, stacker_core::color::srgb_to_linear(128.0 / 255.0), 0.0];
    let expected = stacker_core::color::linear_rgb_to_planar(lin);
    assert!((planar.luma[0] - expected[0]).abs() < 1e-6);
    assert!((planar.chroma_a[0] - expected[1]).abs() < 1e-6);
    assert!((planar.chroma_b[0] - expected[2]).abs() < 1e-6);
}
