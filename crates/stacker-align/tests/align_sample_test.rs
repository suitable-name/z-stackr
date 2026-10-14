#![cfg(feature = "akaze")]
#![expect(
    clippy::many_single_char_names,
    reason = "conventional x/y/w/h names in the smoke test"
)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "downscaled sizes and 8-bit output are derived from clamped, in-range values"
)]

use image::{ColorType, DynamicImage, GenericImageView};
#[cfg(feature = "akaze")]
use stacker_align::{
    akaze_match::{KeypointMatcher, extract_ref_features},
    ransac::{AlignmentEstimator, AlignmentMode},
};
use stacker_align::{
    refine::{RefineOptions, refine_alignment},
    transform::warp_image,
};
use stacker_core::{
    color::{linear_rgb_to_planar, linear_to_srgb_u8, planar_to_linear_rgb, srgb_to_linear},
    image::PlanarImage,
};
use std::path::{Path, PathBuf};

/// Long-side cap (px) applied to every sample frame before processing.
///
/// `test_align_sample_images` is a *smoke* test — it asserts nothing about
/// alignment accuracy, only that the AKAZE → RANSAC → refine → warp pipeline
/// runs end-to-end and produces output files. AKAZE feature extraction on
/// full-resolution macro frames is the dominant cost (seconds per frame), so we
/// shrink each frame to keep the whole test in the sub-second range.
const MAX_TEST_DIM: u32 = 256;

/// Number of target frames (after the reference) the smoke test processes.
const MAX_TEST_TARGETS: usize = 2;

fn dynamic_to_planar(img: &DynamicImage) -> PlanarImage<f32> {
    let (w, h) = img.dimensions();
    let mut planar = PlanarImage::new(w as usize, h as usize);

    let is_16bit = matches!(
        img.color(),
        ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16
    );

    // The colour model is the core's: the same matrix every app and the neural
    // bridge use (`stacker_align` must not depend on the pipeline crate).
    if is_16bit {
        let rgb16 = img.to_rgb16();
        for (i, p) in rgb16.pixels().enumerate() {
            let r = srgb_to_linear(f32::from(p[0]) / 65_535.0);
            let g = srgb_to_linear(f32::from(p[1]) / 65_535.0);
            let b = srgb_to_linear(f32::from(p[2]) / 65_535.0);
            [planar.luma[i], planar.chroma_a[i], planar.chroma_b[i]] =
                linear_rgb_to_planar([r, g, b]);
        }
    } else {
        let rgb8 = img.to_rgb8();
        for (i, p) in rgb8.pixels().enumerate() {
            let r = srgb_to_linear(f32::from(p[0]) / 255.0);
            let g = srgb_to_linear(f32::from(p[1]) / 255.0);
            let b = srgb_to_linear(f32::from(p[2]) / 255.0);
            [planar.luma[i], planar.chroma_a[i], planar.chroma_b[i]] =
                linear_rgb_to_planar([r, g, b]);
        }
    }

    planar
}

/// Downscale a frame so its long side is at most [`MAX_TEST_DIM`] px. Frames
/// already small enough are returned unchanged.
fn downscale_for_test(img: DynamicImage) -> DynamicImage {
    let (w, h) = img.dimensions();
    let long = w.max(h);
    if long <= MAX_TEST_DIM {
        return img;
    }
    let scale = f64::from(MAX_TEST_DIM) / f64::from(long);
    let nw = (f64::from(w) * scale).round() as u32;
    let nh = (f64::from(h) * scale).round() as u32;
    img.resize(nw, nh, image::imageops::FilterType::Triangle)
}

fn collect_image_paths(dir: &Path) -> Vec<PathBuf> {
    // A missing `sample/` directory is an empty list (the test then skips with
    // a printed reason) rather than a panic.
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("frame_")
                && p.extension().is_some_and(|e| e == "jpg")
        })
        .collect();
    paths.sort();
    paths
}

#[test]
#[cfg(feature = "akaze")]
fn test_align_sample_images() {
    let sample_dir = Path::new("../../sample");
    let paths = collect_image_paths(sample_dir);
    if paths.is_empty() {
        eprintln!(
            "SKIPPED test_align_sample_images: no frame_*.jpg found in {}",
            sample_dir.display()
        );
        return;
    }

    let aligned_dir = std::env::temp_dir().join("z-stackr").join("sample_aligned");
    std::fs::create_dir_all(&aligned_dir).unwrap();

    let ref_dyn = downscale_for_test(image::open(&paths[0]).unwrap());
    let ref_frame = dynamic_to_planar(&ref_dyn);

    println!(
        "{:?}",
        aligned_dir.join(format!("aligned_frame_{:04}.jpg", 0))
    );

    ref_dyn
        .save(aligned_dir.join(format!("aligned_frame_{:04}.jpg", 0)))
        .unwrap();

    let (ref_kps, ref_desc) = extract_ref_features(&ref_frame);

    for (i, path) in paths.iter().enumerate().skip(1).take(MAX_TEST_TARGETS) {
        let dyn_img = downscale_for_test(image::open(path).unwrap());
        let frame = dynamic_to_planar(&dyn_img);

        let mr = KeypointMatcher::match_target(&ref_kps, &ref_desc, &frame).unwrap();

        let coarse_matrix = AlignmentEstimator::compute_matrix(
            &mr.matches[..],
            &mr.kps0,
            &mr.kps1,
            AlignmentMode::TranslationAndScale,
        )
        .unwrap_or_else(|_| stacker_align::Matrix3::identity());

        let final_matrix = refine_alignment(
            &ref_frame,
            &frame,
            &coarse_matrix,
            &RefineOptions {
                max_iterations: 100,
                ..RefineOptions::default()
            },
        )
        .unwrap_or(coarse_matrix);

        let warped_planar = warp_image(&frame, &final_matrix).unwrap_or(frame);

        let w = warped_planar.width as u32;
        let h = warped_planar.height as u32;
        let mut out_img = image::RgbImage::new(w, h);
        for (i_px, p) in out_img.pixels_mut().enumerate() {
            let rgb = planar_to_linear_rgb([
                warped_planar.luma[i_px],
                warped_planar.chroma_a[i_px],
                warped_planar.chroma_b[i_px],
            ]);
            // Clamps to [0, 1] and rounds, like every output encoder.
            p[0] = linear_to_srgb_u8(rgb[0]);
            p[1] = linear_to_srgb_u8(rgb[1]);
            p[2] = linear_to_srgb_u8(rgb[2]);
        }

        out_img
            .save(aligned_dir.join(format!("aligned_frame_{i:04}.jpg")))
            .unwrap();
    }
}
