//! `metadata`: EXIF extraction, orientation rewrite and splicing into JPEG and PNG.

use std::io::Cursor;

use image::{
    ColorType, ImageEncoder, Rgb, RgbImage,
    codecs::{jpeg::JpegEncoder, png::PngEncoder, tiff::TiffEncoder},
};
use stacker_core::metadata::{
    JPEG_EXIF_MAX_BYTES, MetadataError, copy_metadata, exif_orientation, extract_exif,
    extract_exif_from_bytes, inject_exif, inject_exif_into_bytes, set_exif_orientation,
};

/// A little-endian TIFF block with IFD0 = { `ImageDescription` "hi", Orientation 6 }.
fn blob(orientation: u8) -> Vec<u8> {
    vec![
        0x49,
        0x49,
        0x2A,
        0x00,
        0x08,
        0x00,
        0x00,
        0x00, // header, IFD0 at 8
        0x02,
        0x00, // two entries
        0x0E,
        0x01,
        0x02,
        0x00,
        0x03,
        0x00,
        0x00,
        0x00,
        b'h',
        b'i',
        0x00,
        0x00, // description
        0x12,
        0x01,
        0x03,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        orientation,
        0x00,
        0x00,
        0x00, // orient.
        0x00,
        0x00,
        0x00,
        0x00, // no next IFD
    ]
}

fn picture() -> RgbImage {
    RgbImage::from_fn(12, 8, |x, y| {
        Rgb([
            u8::try_from(x * 20).unwrap(),
            u8::try_from(y * 30).unwrap(),
            77,
        ])
    })
}

fn jpeg() -> Vec<u8> {
    let mut out = Vec::new();
    JpegEncoder::new_with_quality(&mut out, 90)
        .encode_image(&picture())
        .unwrap();
    out
}

fn png() -> Vec<u8> {
    let img = picture();
    let mut out = Vec::new();
    PngEncoder::new(&mut out)
        .write_image(img.as_raw(), img.width(), img.height(), ColorType::Rgb8)
        .unwrap();
    out
}

fn tiff() -> Vec<u8> {
    let img = picture();
    let mut out = Vec::new();
    TiffEncoder::new(Cursor::new(&mut out))
        .write_image(img.as_raw(), img.width(), img.height(), ColorType::Rgb8)
        .unwrap();
    out
}

fn description(blob: &[u8]) -> Option<String> {
    let exif = exif::Reader::new().read_raw(blob.to_vec()).ok()?;
    let field = exif.get_field(exif::Tag::ImageDescription, exif::In::PRIMARY)?;
    Some(field.display_value().to_string())
}

fn temp(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stacker_core_metadata_{tag}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn orientation_is_read_and_rewritten_in_place() {
    let mut b = blob(6);
    assert_eq!(exif_orientation(&b), Some(6));
    let before = b.clone();
    assert!(set_exif_orientation(&mut b, 1));
    assert_eq!(exif_orientation(&b), Some(1));
    // Only the two value bytes changed.
    let changed: Vec<usize> = (0..b.len()).filter(|&i| b[i] != before[i]).collect();
    assert_eq!(changed.len(), 1, "{changed:?}");
    assert_eq!(b[changed[0]], 1);
}

#[test]
fn orientation_works_in_big_endian_and_rejects_garbage() {
    // Big-endian: IFD0 = { Orientation 8 }.
    let mut be = vec![
        b'M', b'M', 0, 42, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1,
    ];
    be.extend_from_slice(&[0, 8, 0, 0, 0, 0, 0, 0]);
    assert_eq!(exif_orientation(&be), Some(8));
    assert!(set_exif_orientation(&mut be, 3));
    assert_eq!(exif_orientation(&be), Some(3));

    // No orientation entry, wrong magic, truncation, nothing at all.
    let mut no_orientation = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
    assert_eq!(exif_orientation(&no_orientation), None);
    assert!(!set_exif_orientation(&mut no_orientation, 1));
    let mut wrong = blob(6);
    wrong[2] = 43;
    assert_eq!(exif_orientation(&wrong), None);
    assert_eq!(exif_orientation(&blob(6)[..20]), None);
    assert_eq!(exif_orientation(&[]), None);
    assert!(!set_exif_orientation(&mut [], 1));
}

#[test]
fn jpeg_round_trip_keeps_the_pixels_and_the_fields() {
    let plain = jpeg();
    assert_eq!(extract_exif_from_bytes(&plain), None);

    let with = inject_exif_into_bytes(&plain, &blob(6)).unwrap();
    let back = extract_exif_from_bytes(&with).expect("EXIF in the output");
    assert_eq!(exif_orientation(&back), Some(6));
    assert_eq!(description(&back).as_deref(), Some("\"hi\""));
    assert_eq!(
        image::load_from_memory(&with).unwrap().to_rgb8(),
        image::load_from_memory(&plain).unwrap().to_rgb8(),
        "pixels untouched"
    );

    // A second injection replaces the block rather than adding one.
    let again = inject_exif_into_bytes(&with, &blob(1)).unwrap();
    let back = extract_exif_from_bytes(&again).unwrap();
    assert_eq!(exif_orientation(&back), Some(1));
}

#[test]
fn png_round_trip_keeps_the_pixels_and_the_fields() {
    let plain = png();
    let with = inject_exif_into_bytes(&plain, &blob(3)).unwrap();
    let back = extract_exif_from_bytes(&with).expect("EXIF in the output");
    assert_eq!(exif_orientation(&back), Some(3));
    assert_eq!(description(&back).as_deref(), Some("\"hi\""));
    assert_eq!(
        image::load_from_memory(&with).unwrap().to_rgb8(),
        picture(),
        "PNG is lossless"
    );
}

#[test]
fn tiff_and_garbage_outputs_are_refused() {
    let err = inject_exif_into_bytes(&tiff(), &blob(1)).unwrap_err();
    assert!(matches!(err, MetadataError::UnsupportedFormat(_)), "{err}");
    assert!(err.to_string().contains("JPEG and PNG"), "{err}");

    let err = inject_exif_into_bytes(b"not an image at all", &blob(1)).unwrap_err();
    assert!(matches!(err, MetadataError::UnsupportedFormat(_)), "{err}");
}

#[test]
fn a_blob_too_big_for_a_jpeg_segment_is_refused() {
    let big = vec![0_u8; JPEG_EXIF_MAX_BYTES + 1];
    let err = inject_exif_into_bytes(&jpeg(), &big).unwrap_err();
    assert!(matches!(err, MetadataError::TooLarge { .. }), "{err}");
}

#[test]
fn a_tiff_source_yields_a_small_block_of_descriptive_tags_only() {
    // A TIFF file: IFD0 = { ImageWidth 64, Make "hi", Orientation 8 } followed by
    // 8 KiB standing in for pixel data. Its EXIF must come out as a few bytes of
    // tags, not the file.
    let mut file = vec![
        0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // header, IFD0 at 8
        0x03, 0x00, // three entries
        0x00, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, // ImageWidth
        0x0F, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, b'h', b'i', 0x00, 0x00, // Make
        0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, // Orientation
        0x00, 0x00, 0x00, 0x00, // no next IFD
    ];
    file.resize(file.len() + 8192, 0);

    let extracted = extract_exif_from_bytes(&file).expect("EXIF of a TIFF source");
    assert!(extracted.len() < 256, "{} bytes", extracted.len());
    assert_eq!(exif_orientation(&extracted), Some(8));
    let exif = exif::Reader::new().read_raw(extracted).unwrap();
    assert!(exif.get_field(exif::Tag::Make, exif::In::PRIMARY).is_some());
    assert!(
        exif.get_field(exif::Tag::ImageWidth, exif::In::PRIMARY)
            .is_none(),
        "structural tags are not copied"
    );
}

#[test]
fn file_wrappers_copy_and_report_a_source_without_exif() {
    let d = temp("files");
    let source = d.join("source.jpg");
    std::fs::write(&source, inject_exif_into_bytes(&jpeg(), &blob(6)).unwrap()).unwrap();
    let target = d.join("target.png");
    std::fs::write(&target, png()).unwrap();

    assert!(copy_metadata(&source, &target).unwrap());
    let copied = extract_exif(&target).expect("EXIF in the target file");
    assert_eq!(exif_orientation(&copied), Some(6));

    // A source without EXIF, and a missing source, are "nothing to copy".
    let bare = d.join("bare.jpg");
    std::fs::write(&bare, jpeg()).unwrap();
    assert!(!copy_metadata(&bare, &target).unwrap());
    assert!(!copy_metadata(&d.join("missing.jpg"), &target).unwrap());
    assert_eq!(extract_exif(&d.join("missing.jpg")), None);

    // An unreadable output is an I/O error.
    let err = inject_exif(&d.join("missing_output.jpg"), &blob(1)).unwrap_err();
    assert!(matches!(err, MetadataError::Io(_)), "{err}");
    let _ = std::fs::remove_dir_all(&d);
}
