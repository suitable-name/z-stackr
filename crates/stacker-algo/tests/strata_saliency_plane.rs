//! The single-plane blur path: `blur_plane` / `blur_plane_into` and Strata's
//! saliency smoothing equal blurring the luma of a three-plane image with
//! `apply_gaussian_blur`, bit for bit.

use stacker_algo::{
    apex::pyramid::{apply_gaussian_blur, blur_plane, blur_plane_into},
    strata::saliency::{BLUR_PASSES, compute_saliency, laplacian_magnitude, smooth_saliency},
};
use stacker_core::image::PlanarImage;

#[expect(
    clippy::cast_precision_loss,
    reason = "the 24-bit values and 2^24 are exact in f32"
)]
fn plane(w: usize, h: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761) | 1;
    (0..w * h)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1_u32 << 24) as f32
        })
        .collect()
}

fn three_plane(w: usize, h: usize) -> PlanarImage<f32> {
    PlanarImage::from_planes(w, h, plane(w, h, 1), plane(w, h, 2), plane(w, h, 3))
        .expect("planes hold w * h samples")
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const SIZES: [(usize, usize); 8] = [
    (1, 1),
    (2, 7),
    (5, 3),
    (16, 16),
    (17, 9),
    (33, 31),
    (64, 5),
    (70, 70),
];

#[test]
fn blur_plane_equals_the_image_blur_of_every_plane() {
    for (w, h) in SIZES {
        let img = three_plane(w, h);
        let blurred = apply_gaussian_blur(&img);
        assert_eq!(
            bits(&blur_plane(w, h, &img.luma)),
            bits(&blurred.luma),
            "{w}x{h}"
        );
        assert_eq!(
            bits(&blur_plane(w, h, &img.chroma_a)),
            bits(&blurred.chroma_a)
        );
        assert_eq!(
            bits(&blur_plane(w, h, &img.chroma_b)),
            bits(&blurred.chroma_b)
        );
    }
}

#[test]
fn blur_plane_into_overwrites_dirty_buffers() {
    for (w, h) in SIZES {
        let img = three_plane(w, h);
        let expected = blur_plane(w, h, &img.luma);
        let mut temp = vec![f32::NAN; w * h];
        let mut out = vec![123.0_f32; w * h];
        blur_plane_into(w, h, &img.luma, &mut temp, &mut out);
        assert_eq!(bits(&out), bits(&expected), "{w}x{h}");
    }
}

#[test]
fn blur_plane_of_an_empty_plane_is_empty() {
    assert_eq!(blur_plane(0, 5, &[]), vec![]);
    assert_eq!(blur_plane(5, 0, &[]), vec![]);
}

#[test]
fn smooth_saliency_equals_repeated_image_blur_of_the_luma() {
    for (w, h) in SIZES {
        let magnitude = plane(w, h, 9);
        let mut img =
            PlanarImage::with_luma(w, h, magnitude.clone()).expect("one sample per pixel");
        for _ in 0..BLUR_PASSES {
            img = apply_gaussian_blur(&img);
        }
        let got = smooth_saliency(magnitude, w, h);
        assert_eq!(bits(&got), bits(&img.luma), "{w}x{h}");
    }
}

#[test]
fn compute_saliency_is_the_laplacian_then_the_smoothing() {
    let (w, h) = (23, 19);
    let luma = plane(w, h, 4);
    let expected = smooth_saliency(laplacian_magnitude(&luma, w, h), w, h);
    assert_eq!(bits(&compute_saliency(&luma, w, h)), bits(&expected));
}

#[test]
fn smooth_saliency_of_an_empty_plane_is_empty() {
    assert_eq!(smooth_saliency(Vec::new(), 0, 0), vec![]);
}

#[test]
#[should_panic(expected = "width * height")]
fn smooth_saliency_rejects_a_wrong_length() {
    let _ = smooth_saliency(vec![0.0; 10], 4, 4);
}
