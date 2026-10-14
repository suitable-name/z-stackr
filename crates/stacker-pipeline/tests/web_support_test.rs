//! Bytes in, bytes out, names: what a caller without a file system needs.

use std::path::{Path, PathBuf};

use stacker_algo::strata::NO_PREVIEW;
use stacker_core::{
    image::PlanarImage,
    settings::{ImageSavingSettings, OutputFormat, PreprocessingSettings, StackingSettings},
};
use stacker_pipeline::{
    NeverCancel, NoProgress,
    compute::Compute,
    load::{decode_frame, decode_frame_bytes, preprocessed_dimensions},
    order::{natural_sort, natural_sort_names, select_frame_names, select_frames_by},
    output::encode,
    session::{BytesSource, fuse_relief, fuse_strata, load_frames, threshold_from_settings},
};

fn sample_frame(index: usize) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../sample")
        .join(format!("frame_{index:04}.jpg"))
}

fn same_planes(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> bool {
    let planes = |i: &PlanarImage<f32>| {
        (
            i.width,
            i.height,
            [i.luma.clone(), i.chroma_a.clone(), i.chroma_b.clone()],
        )
    };
    planes(a) == planes(b)
}

/// A small image with distinct values in every sample.
fn gradient(w: usize, h: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for i in 0..w * h {
        #[expect(clippy::cast_precision_loss, reason = "tiny test index")]
        let t = i as f32 / (w * h) as f32;
        img.luma[i] = t;
        img.chroma_a[i] = 0.05 * (t - 0.5);
        img.chroma_b[i] = 0.03 * (0.5 - t);
    }
    img
}

fn saving(format: OutputFormat, bit_depth: u32) -> ImageSavingSettings {
    ImageSavingSettings {
        output_format: format,
        bit_depth,
        ..ImageSavingSettings::default()
    }
}

#[test]
fn decoding_bytes_equals_decoding_the_file() {
    let path = sample_frame(0);
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("skipping: {} is missing", path.display());
        return;
    };
    for percent in [100, 25] {
        let pre = PreprocessingSettings {
            pre_resize_percent: percent,
            ..PreprocessingSettings::default()
        };
        let from_file = decode_frame(&path, &pre).unwrap();
        let from_bytes = decode_frame_bytes(&bytes, "frame_0000.jpg", &pre).unwrap();
        assert!(same_planes(&from_file, &from_bytes), "{percent} %");
    }
}

/// `Strata` and the multigrid `Relief` need nothing but the frames: encoded bytes
/// in, fused planes out, no file system and no thread of their own.
#[test]
fn strata_and_the_multigrid_run_from_bytes() {
    let items: Vec<(String, Vec<u8>)> = (0..2_usize)
        .map(|i| {
            let mut img = gradient(64, 48);
            for (k, v) in img.luma.iter_mut().enumerate() {
                if k % (i + 2) == 0 {
                    *v *= 0.5;
                }
            }
            let bytes = encode(&img, &saving(OutputFormat::Png, 8))
                .to_bytes()
                .unwrap();
            (format!("web:/f{i}.png"), bytes)
        })
        .collect();
    let loaded = load_frames(
        &mut BytesSource::new(items),
        &PreprocessingSettings::default(),
        &NeverCancel,
        &NoProgress,
    )
    .unwrap();
    let settings = StackingSettings {
        relief_use_multigrid: true,
        ..StackingSettings::default()
    };

    let strata = fuse_strata(&loaded.frames, &settings, &NoProgress, NO_PREVIEW, |_| {});
    assert_eq!((strata.width, strata.height), (64, 48));
    assert!(strata.luma.iter().all(|v| v.is_finite()));

    let relief = fuse_relief(&mut Compute::cpu(), &loaded.frames, &settings, |max| {
        Some(threshold_from_settings(max, &settings))
    })
    .unwrap();
    assert_eq!((relief.width, relief.height), (64, 48));
    assert!(relief.luma.iter().all(|v| v.is_finite()));
}

#[test]
fn undecodable_bytes_name_the_frame() {
    let error = decode_frame_bytes(
        b"not an image",
        "broken.jpg",
        &PreprocessingSettings::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("broken.jpg"), "{error}");
}

#[test]
fn preprocessed_dimensions_follow_the_preprocessing() {
    let pre = PreprocessingSettings::default();
    assert_eq!(preprocessed_dimensions(400, 300, &pre), (400, 300));
    let resized = PreprocessingSettings {
        pre_resize_percent: 50,
        ..PreprocessingSettings::default()
    };
    assert_eq!(preprocessed_dimensions(400, 300, &resized), (200, 150));
    let rotated = PreprocessingSettings {
        pre_rotation: 90,
        pre_crop_enabled: true,
        pre_crop_spec: "100,200".to_owned(),
        ..PreprocessingSettings::default()
    };
    assert_eq!(preprocessed_dimensions(400, 300, &rotated), (100, 200));
}

#[test]
fn encoded_bytes_decode_back_to_the_samples() {
    let planar = gradient(7, 5);
    for format in [OutputFormat::Png, OutputFormat::Tiff] {
        let e8 = encode(&planar, &saving(format, 8));
        let decoded = image::load_from_memory(&e8.to_bytes().unwrap()).unwrap();
        assert_eq!(
            decoded
                .as_rgb8()
                .expect("an 8-bit file")
                .as_raw()
                .as_slice(),
            e8.rgb8().unwrap(),
            "{format:?} 8 bit"
        );

        let e16 = encode(&planar, &saving(format, 16));
        let decoded = image::load_from_memory(&e16.to_bytes().unwrap()).unwrap();
        assert_eq!(
            decoded
                .as_rgb16()
                .expect("a 16-bit file")
                .as_raw()
                .as_slice(),
            e16.rgb16().unwrap(),
            "{format:?} 16 bit"
        );
    }
}

#[test]
fn a_jpeg_decodes_to_its_size() {
    let encoded = encode(&gradient(16, 8), &saving(OutputFormat::Jpeg, 16));
    assert!(!encoded.is_16_bit(), "JPEG is always 8-bit");
    assert_eq!(encoded.extension(), "jpg");
    let decoded = image::load_from_memory(&encoded.to_bytes().unwrap()).unwrap();
    assert_eq!(
        (
            decoded.as_rgb8().unwrap().width(),
            decoded.as_rgb8().unwrap().height()
        ),
        (16, 8)
    );
}

#[test]
fn to_bytes_is_what_save_writes() {
    let dir = std::env::temp_dir().join(format!("stacker_pipeline_bytes_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let planar = gradient(9, 6);
    for (format, depth) in [
        (OutputFormat::Png, 8),
        (OutputFormat::Png, 16),
        (OutputFormat::Tiff, 8),
        (OutputFormat::Tiff, 16),
        (OutputFormat::Jpeg, 8),
    ] {
        let encoded = encode(&planar, &saving(format, depth));
        let bytes = encoded.to_bytes().unwrap();
        let path = dir.join(format!("out_{depth}.{}", encoded.extension()));
        encoded.save(&path).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            bytes,
            "{format:?} {depth} bit"
        );
    }
}

#[test]
fn sorting_names_matches_sorting_paths() {
    let raw = [
        "frame_10.jpg",
        "frame_2.jpg",
        "Frame_1.jpg",
        "frame_01.jpg",
        "frame_1.jpg",
        "a.png",
        "frame_100.jpg",
    ];
    let mut names: Vec<String> = raw.iter().map(|s| (*s).to_owned()).collect();
    let mut paths: Vec<PathBuf> = raw.iter().map(PathBuf::from).collect();
    natural_sort_names(&mut names);
    natural_sort(&mut paths);
    let from_paths: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, from_paths);
}

#[test]
fn selecting_names_matches_selecting_paths() {
    let raw = [
        "a.jpg",
        "stacker_result_Apex_1.png",
        "b.jpg",
        "c.jpg",
        "d.jpg",
        "e.jpg",
    ];
    for reverse in [false, true] {
        for nth in [0, 1, 2, 3] {
            let names =
                select_frame_names(raw.iter().map(|s| (*s).to_owned()).collect(), reverse, nth);
            let paths = select_frames_by(raw.iter().map(PathBuf::from).collect(), reverse, nth);
            let expected: Vec<String> = paths
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            assert_eq!(names, expected, "reverse {reverse}, every {nth}");
            assert!(names.iter().all(|n| !n.contains("stacker_result_")));
        }
    }
}
