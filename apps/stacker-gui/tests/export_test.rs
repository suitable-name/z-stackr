//! Save encodes from the `f32` result: real 16-bit output, retouch included.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use image::ColorType;
use stacker_algo::hybrid::retouch::RetouchSession;
use stacker_core::{
    color::{linear_to_srgb_u8, linear_to_srgb_u16},
    image::PlanarImage,
};
use stacker_gui::{
    export::{
        encode_planar_exif, exif_block, exif_note, planar_to_rgb8, planar_to_rgb16, save_file,
        save_planar, save_planar_exif,
    },
    retouch::{RetouchState, SourceChoice},
    settings::{ImageSavingSettings, OutputFormat},
};

const W: u16 = 64;
const H: u16 = 16;

/// A dark horizontal ramp (gray: chroma zero) whose levels do not fall on 8-bit codes.
fn ramp() -> PlanarImage<f32> {
    let mut image = PlanarImage::new(usize::from(W), usize::from(H));
    for y in 0..H {
        for x in 0..W {
            let i = usize::from(y) * usize::from(W) + usize::from(x);
            image.luma[i] = 0.002 + 0.25 * f32::from(x) / f32::from(W - 1);
        }
    }
    image
}

fn white() -> PlanarImage<f32> {
    let mut image = PlanarImage::new(usize::from(W), usize::from(H));
    image.luma.fill(1.0);
    image
}

/// The ramp with one full-opacity stroke of white over (32, 8).
fn retouched() -> PlanarImage<f32> {
    let base = Arc::new(ramp());
    let mut state = RetouchState::new();
    state.start_session(
        RetouchSession::new(Arc::clone(&base), base),
        PathBuf::from("stacker_result_Apex_1.png"),
    );
    state.refresh_sources(None, &[PathBuf::from("frame.png")]);
    let SourceChoice::Load(request) = state.choose_source(Some(0)) else {
        panic!("choosing the only frame should ask for it");
    };
    assert_eq!(state.finish_source(&request, white()), Ok(true));
    assert!(state.begin_stroke());
    state.queue_point(Some((32, 8)), 3.0, 1.0);
    assert!(state.end_stroke());
    state.composite_f32().expect("a session")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stacker_export_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the scratch directory");
    dir.join(name)
}

fn settings(format: OutputFormat, bit_depth: u32) -> ImageSavingSettings {
    ImageSavingSettings {
        output_format: format,
        bit_depth,
        ..ImageSavingSettings::default()
    }
}

fn color_of(path: &Path) -> ColorType {
    image::open(path).expect("the saved file decodes").color()
}

/// A bare TIFF structure holding one tag: Orientation = 6.
const EXIF_BLOB: [u8; 26] = [
    0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // header, first IFD at 8
    0x01, 0x00, // one entry
    0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, // Orientation
    0x00, 0x00, 0x00, 0x00, // no next IFD
];

/// A JPEG on disk that carries [`EXIF_BLOB`].
fn jpeg_with_exif(name: &str) -> PathBuf {
    let path = scratch(name);
    image::RgbImage::from_pixel(8, 8, image::Rgb([90, 120, 150]))
        .save(&path)
        .expect("write the source jpeg");
    stacker_core::metadata::inject_exif(&path, &EXIF_BLOB).expect("add the EXIF block");
    path
}

#[test]
fn copy_metadata_carries_the_first_frames_exif_into_a_jpeg() {
    let source = jpeg_with_exif("exif_source.jpg");
    let saving = ImageSavingSettings {
        output_format: OutputFormat::Jpeg,
        bit_depth: 8,
        copy_metadata: true,
        ..ImageSavingSettings::default()
    };
    let pre = stacker_gui::settings::PreprocessingSettings::default();
    let exif = exif_block(Some(&source), &pre, &saving, None);
    assert!(exif.is_some(), "the source frame has EXIF");

    let dest = scratch("exif_out.jpg");
    save_planar_exif(&ramp(), &dest, &saving, exif).expect("jpeg save");
    let bytes = std::fs::read(&dest).expect("read the result");
    assert!(
        stacker_core::metadata::extract_exif_from_bytes(&bytes).is_some(),
        "the saved JPEG carries the EXIF block"
    );

    // The browser's way (the frame's bytes) finds the same block.
    let source_bytes = std::fs::read(&source).expect("read the source");
    let read = |_: &Path| -> std::io::Result<Vec<u8>> { Ok(source_bytes.clone()) };
    assert!(exif_block(Some(&source), &pre, &saving, Some(&read)).is_some());
}

#[test]
fn copy_metadata_off_or_no_source_gives_no_block() {
    let source = jpeg_with_exif("exif_off.jpg");
    let pre = stacker_gui::settings::PreprocessingSettings::default();
    let off = ImageSavingSettings {
        copy_metadata: false,
        ..ImageSavingSettings::default()
    };
    assert!(exif_block(Some(&source), &pre, &off, None).is_none());
    let on = ImageSavingSettings {
        copy_metadata: true,
        ..ImageSavingSettings::default()
    };
    assert!(exif_block(None, &pre, &on, None).is_none());
    assert!(exif_block(Some(&scratch("missing.jpg")), &pre, &on, None).is_none());
}

#[test]
fn a_tiff_cannot_carry_metadata_and_the_note_says_so() {
    let on = ImageSavingSettings {
        output_format: OutputFormat::Tiff,
        copy_metadata: true,
        ..ImageSavingSettings::default()
    };
    assert!(exif_note(Path::new("out.tif"), &on).is_some());
    assert!(
        exif_note(Path::new("out.png"), &on).is_none(),
        "the extension wins"
    );
    let off = ImageSavingSettings {
        copy_metadata: false,
        ..on
    };
    assert!(exif_note(Path::new("out.tif"), &off).is_none());
    // The block is dropped, not an error.
    let (bytes, _) = encode_planar_exif(
        &ramp(),
        Path::new("out.tif"),
        &off,
        Some(EXIF_BLOB.to_vec()),
    )
    .expect("tiff encodes");
    assert_ne!(bytes.len(), 0);
}

#[test]
fn a_16_bit_save_of_a_retouched_result_reaches_full_scale() {
    let composite = retouched();
    let p16 = scratch("retouched16.png");
    let p8 = scratch("retouched8.png");
    let saved16 =
        save_planar(&composite, &p16, &settings(OutputFormat::Png, 16)).expect("16-bit save");
    let saved8 = save_planar(&composite, &p8, &settings(OutputFormat::Png, 8)).expect("8-bit save");
    assert!(saved16.bit16 && !saved8.bit16);
    assert_eq!(color_of(&p16), ColorType::Rgb16);
    assert_eq!(color_of(&p8), ColorType::Rgb8);

    let wide = image::open(&p16).expect("decodes").to_rgb16();
    let narrow = image::open(&p8).expect("decodes").to_rgb8();

    // The stroke is in the file, and white is full scale at both depths.
    assert_eq!(wide.get_pixel(32, 8).0, [65_535; 3]);
    assert_eq!(narrow.get_pixel(32, 8).0, [255; 3]);

    // Outside the stroke the ramp is there, dark.
    assert!(wide.get_pixel(2, 8).0[0] < 20_000);

    // The 16-bit file carries data the 8-bit path cannot: some samples are not a
    // widened 8-bit code (multiples of 257), which is all a re-encode of the 8-bit
    // preview PNG could ever produce.
    let off_grid = wide.pixels().filter(|p| p.0[0] % 257 != 0).count();
    assert!(off_grid > 0, "a real 16-bit save has sub-8-bit levels");

    // And the two depths agree to within one 8-bit step.
    for (a, b) in wide.pixels().zip(narrow.pixels()) {
        let widened = i32::from(b.0[0]) * 257;
        assert!((i32::from(a.0[0]) - widened).abs() <= 257, "{a:?} vs {b:?}");
    }

    let _ = std::fs::remove_file(p16);
    let _ = std::fs::remove_file(p8);
}

#[test]
fn the_save_contains_the_retouch_not_just_the_fused_image() {
    let composite = retouched();
    let plain = ramp();
    let with_stroke = planar_to_rgb16(&composite).expect("encodes");
    let without = planar_to_rgb16(&plain).expect("encodes");
    assert_eq!(with_stroke.get_pixel(32, 8).0, [65_535; 3]);
    assert_ne!(without.get_pixel(32, 8).0, [65_535; 3]);
    // Away from the stroke the two are identical.
    assert_eq!(with_stroke.get_pixel(60, 3), without.get_pixel(60, 3));
}

#[test]
fn a_16_bit_tiff_is_16_bit() {
    let composite = retouched();
    let path = scratch("retouched16.tiff");
    let saved =
        save_planar(&composite, &path, &settings(OutputFormat::Png, 16)).expect("tiff save");
    assert_eq!(
        saved.format,
        OutputFormat::Tiff,
        "the extension names the format"
    );
    assert!(saved.bit16);
    assert_eq!(color_of(&path), ColorType::Rgb16);
    let wide = image::open(&path).expect("decodes").to_rgb16();
    assert_eq!(wide.get_pixel(32, 8).0, [65_535; 3]);
    let _ = std::fs::remove_file(path);
}

#[test]
fn jpeg_is_always_8_bit() {
    let composite = retouched();
    let path = scratch("retouched.jpg");
    let saved =
        save_planar(&composite, &path, &settings(OutputFormat::Png, 16)).expect("jpeg save");
    assert_eq!(saved.format, OutputFormat::Jpeg);
    assert!(!saved.bit16);
    assert_eq!(color_of(&path), ColorType::Rgb8);
    let _ = std::fs::remove_file(path);
}

#[test]
fn the_encoders_round_to_the_core_codes() {
    let ramp = ramp();
    let rgb8 = planar_to_rgb8(&ramp).expect("encodes");
    let rgb16 = planar_to_rgb16(&ramp).expect("encodes");
    for x in 0..u32::from(W) {
        let luma = ramp.luma[x as usize];
        let wide = rgb16.get_pixel(x, 0).0;
        assert_eq!(wide, [linear_to_srgb_u16(luma); 3], "column {x}");
        let narrow = rgb8.get_pixel(x, 0).0[0];
        assert!(narrow.abs_diff(linear_to_srgb_u8(luma)) <= 1, "column {x}");
    }
}

#[test]
fn a_file_save_cannot_exceed_the_precision_of_the_file() {
    // The source-frame path: an 8-bit file saved as 16-bit is that file, widened.
    let source = scratch("source8.png");
    planar_to_rgb8(&ramp())
        .expect("encodes")
        .save(&source)
        .expect("writes the source");
    let dest = scratch("from_file16.png");
    let saved = save_file(&source, &dest, &settings(OutputFormat::Png, 16)).expect("re-encodes");
    assert!(saved.bit16);
    assert_eq!(color_of(&dest), ColorType::Rgb16);
    let wide = image::open(&dest).expect("decodes").to_rgb16();
    let off: Vec<u16> = wide
        .pixels()
        .map(|p| p.0[0])
        .filter(|v| v % 257 != 0)
        .take(5)
        .collect();
    assert!(
        off.is_empty(),
        "an 8-bit file stays on the 8-bit grid; off-grid samples: {off:?}"
    );
    let _ = std::fs::remove_file(source);
    let _ = std::fs::remove_file(dest);
}

#[test]
fn an_empty_image_is_an_error_not_a_panic() {
    let empty = PlanarImage::<f32>::new(0, 0);
    assert!(planar_to_rgb8(&empty).is_err());
    assert!(planar_to_rgb16(&empty).is_err());
    let path = scratch("empty.png");
    assert!(save_planar(&empty, &path, &settings(OutputFormat::Png, 16)).is_err());
}
