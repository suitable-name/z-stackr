//! `output`: planar linear light to integer samples and files.

use std::path::PathBuf;

use stacker_core::{
    image::PlanarImage,
    metadata::{exif_orientation, extract_exif, extract_exif_from_bytes},
    settings::{ImageSavingSettings, OutputFormat, PreprocessingSettings},
};
use stacker_pipeline::{
    crop::resize_planar,
    output::{
        EncodedImage, encode, jpeg_quality, source_exif, source_exif_from_bytes, supports_exif,
        wants_16_bit,
    },
};

fn saving(format: OutputFormat, bit_depth: u32) -> ImageSavingSettings {
    ImageSavingSettings {
        output_format: format,
        bit_depth,
        ..ImageSavingSettings::default()
    }
}

fn filled(w: usize, h: usize, luma: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    img.luma.fill(luma);
    img
}

fn dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("stacker_pipeline_out_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// CORE-1 at the shared encoder: linear white is full scale at both depths.
#[test]
fn white_encodes_to_full_scale() {
    let white = filled(5, 3, 1.0);
    let e8 = encode(&white, &saving(OutputFormat::Png, 8));
    assert!(e8.rgb8().unwrap().iter().all(|&s| s == 255));
    let e16 = encode(&white, &saving(OutputFormat::Png, 16));
    assert!(e16.rgb16().unwrap().iter().all(|&s| s == 65_535));
    // Black stays black, and overshoot clamps rather than wrapping.
    let black = encode(&filled(2, 2, 0.0), &saving(OutputFormat::Tiff, 8));
    assert!(black.rgb8().unwrap().iter().all(|&s| s == 0));
    let over = encode(&filled(2, 2, 1.7), &saving(OutputFormat::Tiff, 16));
    assert!(over.rgb16().unwrap().iter().all(|&s| s == 65_535));
    let under = encode(&filled(2, 2, -0.4), &saving(OutputFormat::Tiff, 16));
    assert!(under.rgb16().unwrap().iter().all(|&s| s == 0));
}

/// The bit depth key applies to PNG and TIFF; JPEG is always 8-bit.
#[test]
fn bit_depth_and_format_decide_the_buffer() {
    assert!(wants_16_bit(OutputFormat::Png, 16));
    assert!(wants_16_bit(OutputFormat::Tiff, 16));
    assert!(!wants_16_bit(OutputFormat::Png, 8));
    assert!(!wants_16_bit(OutputFormat::Jpeg, 16));

    let img = filled(2, 2, 0.5);
    assert!(encode(&img, &saving(OutputFormat::Tiff, 16)).is_16_bit());
    assert!(!encode(&img, &saving(OutputFormat::Tiff, 8)).is_16_bit());
    assert!(!encode(&img, &saving(OutputFormat::Jpeg, 16)).is_16_bit());
}

#[test]
fn jpeg_quality_is_clamped_to_its_range() {
    assert_eq!(jpeg_quality(0), 1);
    assert_eq!(jpeg_quality(95), 95);
    assert_eq!(jpeg_quality(500), 100);
}

/// Sixteen bits are 16 bits of the tone curve, not 8 bits widened.
#[test]
fn sixteen_bit_output_keeps_the_fine_steps() {
    let mut img = PlanarImage::<f32>::new(2, 1);
    // Two linear values that are the same 8-bit code.
    img.luma[0] = 0.2000;
    img.luma[1] = 0.2003;
    let e8 = encode(&img, &saving(OutputFormat::Png, 8));
    let e16 = encode(&img, &saving(OutputFormat::Png, 16));
    let s8 = e8.rgb8().unwrap();
    let s16 = e16.rgb16().unwrap();
    assert_eq!(s8[0], s8[3], "same 8-bit code");
    assert!(s16[3] > s16[0], "different 16-bit codes");
}

/// Pasting interiors at their offsets writes the right pixels and nothing else.
#[test]
fn paste_tile_addresses_the_destination_rectangle() {
    // A 6x6 padded tile whose luma encodes its position; the interior is the
    // 2x2 block at offset (3, 2).
    let mut tile = PlanarImage::<f32>::new(6, 6);
    for y in 0..6_u8 {
        for x in 0..6_u8 {
            tile.luma[usize::from(y) * 6 + usize::from(x)] = f32::from(y * 6 + x) / 40.0;
        }
    }
    let mut out = EncodedImage::zeroed(8, 5, OutputFormat::Png, true, 95);
    out.paste_tile(&tile, (3, 2), (4, 1, 2, 2));
    let samples = out.rgb16().unwrap();
    let at = |x: usize, y: usize| samples[(y * 8 + x) * 3];
    let expect = |tx: u8, ty: u8| -> u16 {
        encode(
            &filled(1, 1, f32::from(ty * 6 + tx) / 40.0),
            &saving(OutputFormat::Png, 16),
        )
        .rgb16()
        .unwrap()[0]
    };
    assert_eq!(at(4, 1), expect(3, 2));
    assert_eq!(at(5, 1), expect(4, 2));
    assert_eq!(at(4, 2), expect(3, 3));
    assert_eq!(at(5, 2), expect(4, 3));
    // Everything else is still zero.
    let nonzero = samples.iter().filter(|&&s| s != 0).count();
    assert_eq!(nonzero, 4 * 3, "four pixels, three channels each");
}

#[test]
fn crop_cuts_the_rectangle_out() {
    let mut img = PlanarImage::<f32>::new(6, 4);
    for (i, v) in img.luma.iter_mut().enumerate() {
        *v = f32::from(u8::try_from(i).unwrap()) / 30.0;
    }
    let full = encode(&img, &saving(OutputFormat::Png, 8));
    let cut = full.clone().crop((2, 1, 3, 2));
    assert_eq!((cut.width(), cut.height()), (3, 2));
    let (full_px, cut_px) = (full.rgb8().unwrap(), cut.rgb8().unwrap());
    for y in 0..2 {
        for x in 0..3 {
            for c in 0..3 {
                assert_eq!(
                    cut_px[(y * 3 + x) * 3 + c],
                    full_px[((y + 1) * 6 + x + 2) * 3 + c]
                );
            }
        }
    }
    // The whole frame returns the buffer unchanged.
    let same = full.clone().crop((0, 0, 6, 4));
    assert_eq!(same.rgb8(), full.rgb8());
}

/// What is written to disk reads back as the samples that were encoded.
#[test]
fn files_round_trip() {
    let d = dir("files");
    let mut img = PlanarImage::<f32>::new(16, 12);
    for (i, v) in img.luma.iter_mut().enumerate() {
        *v = f32::from(u8::try_from(i).unwrap()) / 200.0;
    }

    let e16 = encode(&img, &saving(OutputFormat::Tiff, 16));
    let want16 = e16.rgb16().unwrap().to_vec();
    let tiff = d.join("out.tiff");
    e16.save(&tiff).unwrap();
    let back = image::open(&tiff).unwrap().to_rgb16();
    assert_eq!(back.as_raw(), &want16);

    let e8 = encode(&img, &saving(OutputFormat::Png, 8));
    let want8 = e8.rgb8().unwrap().to_vec();
    let png = d.join("out.png");
    e8.save(&png).unwrap();
    assert_eq!(image::open(&png).unwrap().to_rgb8().as_raw(), &want8);

    // JPEG: lower quality, smaller file.
    let mut noisy = PlanarImage::<f32>::new(64, 64);
    for (i, v) in noisy.luma.iter_mut().enumerate() {
        *v = f32::from(u8::try_from((i * 37 + (i / 64) * 91) % 251).unwrap()) / 255.0;
    }
    let mut sizes = Vec::new();
    for q in [30, 95] {
        let s = ImageSavingSettings {
            jpeg_quality: q,
            ..saving(OutputFormat::Jpeg, 8)
        };
        let path = d.join(format!("q{q}.jpg"));
        encode(&noisy, &s).save(&path).unwrap();
        sizes.push(std::fs::metadata(&path).unwrap().len());
    }
    assert!(sizes[0] < sizes[1], "{sizes:?}");
    let _ = std::fs::remove_dir_all(&d);
}

// ── copy_metadata ─────────────────────────────────────────────────────────────

/// A JPEG whose EXIF block holds only the orientation tag, `orientation`.
fn source_jpeg(orientation: u16) -> Vec<u8> {
    let mut plain = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut plain, 90)
        .encode_image(&image::RgbImage::from_pixel(
            8,
            8,
            image::Rgb([90, 120, 30]),
        ))
        .unwrap();
    // Little-endian TIFF block: header, one IFD0 entry (orientation, SHORT).
    let mut tiff = vec![b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0];
    tiff.extend_from_slice(&0x0112_u16.to_le_bytes());
    tiff.extend_from_slice(&3_u16.to_le_bytes());
    tiff.extend_from_slice(&1_u32.to_le_bytes());
    tiff.extend_from_slice(&orientation.to_le_bytes());
    tiff.extend_from_slice(&[0, 0]);
    tiff.extend_from_slice(&0_u32.to_le_bytes());
    let mut segment = b"Exif\0\0".to_vec();
    segment.extend_from_slice(&tiff);
    let mut out = vec![0xFF, 0xD8, 0xFF, 0xE1];
    out.extend_from_slice(&u16::try_from(segment.len() + 2).unwrap().to_be_bytes());
    out.extend_from_slice(&segment);
    out.extend_from_slice(&plain[2..]);
    out
}

fn copying(format: OutputFormat) -> ImageSavingSettings {
    ImageSavingSettings {
        copy_metadata: true,
        ..saving(format, 8)
    }
}

#[test]
fn nothing_is_copied_unless_copy_metadata_is_on() {
    let d = dir("exif_off");
    let source = d.join("source.jpg");
    std::fs::write(&source, source_jpeg(6)).unwrap();
    let pre = PreprocessingSettings::default();

    let off = saving(OutputFormat::Jpeg, 8);
    assert!(!off.copy_metadata);
    assert_eq!(source_exif(&source, &pre, &off), None);
    assert_eq!(source_exif_from_bytes(&source_jpeg(6), &pre, &off), None);

    // On, but the source has no EXIF or does not exist.
    let bare = d.join("bare.png");
    image::RgbImage::new(4, 4).save(&bare).unwrap();
    assert_eq!(source_exif(&bare, &pre, &copying(OutputFormat::Jpeg)), None);
    assert_eq!(
        source_exif(&d.join("gone.jpg"), &pre, &copying(OutputFormat::Jpeg)),
        None
    );

    // An output written without `with_exif` carries none.
    let out = d.join("out.jpg");
    encode(&filled(8, 8, 0.4), &off).save(&out).unwrap();
    assert_eq!(extract_exif(&out), None);
    let _ = std::fs::remove_dir_all(&d);
}

/// The loader turns a JPEG upright, so its tag must become 1; an ignored
/// orientation leaves the pixels as they are, so the tag stays, unless the
/// pre-rotation turned them anyway.
#[test]
fn the_orientation_tag_follows_what_the_loader_did_to_the_pixels() {
    let d = dir("exif_orientation");
    let source = d.join("source.jpg");
    let bytes = source_jpeg(6);
    std::fs::write(&source, &bytes).unwrap();
    let on = copying(OutputFormat::Jpeg);

    let applied = PreprocessingSettings::default();
    let ignored = PreprocessingSettings {
        ignore_exif_orientation: true,
        ..PreprocessingSettings::default()
    };
    let ignored_and_turned = PreprocessingSettings {
        ignore_exif_orientation: true,
        pre_rotation: 90,
        ..PreprocessingSettings::default()
    };

    for (pre, expected) in [(&applied, 1), (&ignored, 6), (&ignored_and_turned, 1)] {
        let from_file = source_exif(&source, pre, &on).expect("EXIF of the source");
        let from_memory = source_exif_from_bytes(&bytes, pre, &on).expect("EXIF of the source");
        assert_eq!(exif_orientation(&from_file), Some(expected));
        assert_eq!(from_file, from_memory);
    }

    // An already upright source stays 1.
    let upright = source_jpeg(1);
    let blob = source_exif_from_bytes(&upright, &applied, &on).unwrap();
    assert_eq!(exif_orientation(&blob), Some(1));
    let _ = std::fs::remove_dir_all(&d);
}

/// JPEG and PNG outputs carry the block, in memory and on disk; a TIFF output is
/// written without it (and still written).
#[test]
fn exif_is_written_into_jpeg_and_png_but_a_tiff_is_written_without() {
    let d = dir("exif_outputs");
    let blob = source_exif_from_bytes(
        &source_jpeg(6),
        &PreprocessingSettings::default(),
        &copying(OutputFormat::Jpeg),
    )
    .unwrap();
    assert!(supports_exif(OutputFormat::Jpeg) && supports_exif(OutputFormat::Png));
    assert!(!supports_exif(OutputFormat::Tiff));

    let img = filled(16, 16, 0.3);
    for (format, name) in [
        (OutputFormat::Jpeg, "out.jpg"),
        (OutputFormat::Png, "out.png"),
    ] {
        let encoded = encode(&img, &saving(format, 8)).with_exif(Some(blob.clone()));
        assert_eq!(encoded.exif(), Some(blob.as_slice()));

        let bytes = encoded.to_bytes().unwrap();
        let in_memory = extract_exif_from_bytes(&bytes).expect("EXIF in the encoded bytes");
        assert_eq!(exif_orientation(&in_memory), Some(1), "{name}");

        let path = d.join(name);
        encoded.save(&path).unwrap();
        let on_disk = extract_exif(&path).expect("EXIF in the saved file");
        assert_eq!(exif_orientation(&on_disk), Some(1), "{name}");
        // The pixels are those of the same buffer without metadata.
        let plain = encode(&img, &saving(format, 8)).to_bytes().unwrap();
        assert_eq!(
            image::load_from_memory(&bytes).unwrap().to_rgb8(),
            image::load_from_memory(&plain).unwrap().to_rgb8(),
            "{name}"
        );
    }

    let tiff = encode(&img, &saving(OutputFormat::Tiff, 8));
    let plain_bytes = tiff.to_bytes().unwrap();
    let with = tiff.with_exif(Some(blob));
    assert_eq!(with.to_bytes().unwrap(), plain_bytes);
    let path = d.join("out.tif");
    with.save(&path).unwrap();
    assert_eq!(image::open(&path).unwrap().to_rgb8().dimensions(), (16, 16));
    // The TIFF encoder itself writes resolution tags, which `extract_exif`
    // rebuilds as a compact block; the point is that attaching EXIF changes
    // nothing: the block equals that of the same TIFF written without it,
    // and carries no orientation.
    let plain_path = d.join("plain.tif");
    std::fs::write(&plain_path, &plain_bytes).unwrap();
    assert_eq!(extract_exif(&path), extract_exif(&plain_path));
    let from_tiff = extract_exif(&path);
    assert_eq!(from_tiff.as_deref().and_then(exif_orientation), None);
    let _ = std::fs::remove_dir_all(&d);
}

/// A gently varying grey ramp, the case where resampling the encoded samples
/// and resampling the linear planes agree to the 8-bit code.
fn ramp(w: usize, h: usize) -> PlanarImage<f32> {
    let unit = |n: usize, of: usize| {
        f32::from(u8::try_from(n).unwrap()) / f32::from(u8::try_from(of).unwrap())
    };
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.luma[y * w + x] = 0.01f32.mul_add(unit(y, h), 0.06f32.mul_add(unit(x, w - 1), 0.4));
        }
    }
    img
}

/// `EncodedImage::resize` is the encoded-sample counterpart of `resize_planar`:
/// within one code of resizing the planes and encoding, down and up.
#[test]
fn resizing_an_encoded_image_matches_resizing_the_planes_and_encoding() {
    let source = ramp(12, 8);
    let settings = saving(OutputFormat::Png, 8);
    for (w, h) in [(9, 6), (20, 14), (12, 5), (7, 8)] {
        let expected = encode(&resize_planar(source.clone(), w, h), &settings);
        let got = encode(&source, &settings).resize(w, h);
        assert_eq!((got.width(), got.height()), (w, h));
        let (a, b) = (got.rgb8().unwrap(), expected.rgb8().unwrap());
        assert_eq!(a.len(), b.len());
        let worst = a.iter().zip(b).map(|(&p, &q)| p.abs_diff(q)).max().unwrap();
        assert!(worst <= 1, "{w}x{h}: off by {worst} codes");
    }
}

/// A flat 16-bit image stays flat (within one code) at any size, and a buffer of
/// the target size comes back as it is, EXIF block included.
#[test]
fn resizing_keeps_flat_sixteen_bit_images_and_the_exif_block() {
    let flat = encode(&filled(10, 6, 0.3), &saving(OutputFormat::Png, 16));
    let reference = flat.rgb16().unwrap()[0];
    let bigger = flat.clone().resize(25, 9);
    assert_eq!((bigger.width(), bigger.height()), (25, 9));
    assert!(bigger.is_16_bit());
    assert!(
        bigger
            .rgb16()
            .unwrap()
            .iter()
            .all(|&s| s.abs_diff(reference) <= 1)
    );

    let blob = vec![1_u8, 2, 3];
    let tagged = flat.with_exif(Some(blob.clone()));
    let same = tagged.clone().resize(10, 6);
    assert_eq!(same.rgb16(), tagged.rgb16());
    assert_eq!(same.exif(), Some(blob.as_slice()));
    let resized = tagged.resize(5, 3);
    assert_eq!(resized.exif(), Some(blob.as_slice()));
}

/// An empty target gives an empty buffer instead of a panic.
#[test]
fn resizing_to_or_from_nothing_gives_zeros() {
    let img = encode(&filled(4, 4, 0.5), &saving(OutputFormat::Png, 8));
    let none = img.resize(0, 0);
    assert_eq!((none.width(), none.height()), (0, 0));
    let empty = EncodedImage::zeroed(0, 0, OutputFormat::Png, false, 95);
    let grown = empty.resize(3, 2);
    assert_eq!(grown.rgb8().unwrap(), &[0_u8; 18]);
}
