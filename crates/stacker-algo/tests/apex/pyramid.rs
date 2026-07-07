use stacker_algo::apex::pyramid::*;
use stacker_core::image::PlanarImage;

fn create_test_image(w: usize, h: usize) -> PlanarImage<f32> {
    let mut luma = vec![0.0_f32; w * h];
    let mut a = vec![0.0_f32; w * h];
    let mut b = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            luma[idx] = (x + y) as f32;
            a[idx] = (x as f32) * 0.5;
            b[idx] = (y as f32) * 0.5;
        }
    }
    PlanarImage {
        width: w,
        height: h,
        luma,
        chroma_a: a,
        chroma_b: b,
    }
}

#[test]
fn test_blur_propagation() {
    let img = create_test_image(16, 16);
    let blurred = apply_gaussian_blur(&img);
    assert_eq!(blurred.width, 16);
    assert_eq!(blurred.height, 16);
    assert!(blurred.luma[8 * 16 + 8].is_finite());
}

#[test]
fn test_reversibility() {
    let img = create_test_image(15, 13);
    let pyramid = LaplacianPyramid::build(&img, 4);
    let recon = pyramid.reconstruct();

    assert_eq!(recon.width, img.width);
    assert_eq!(recon.height, img.height);

    for i in 0..img.width * img.height {
        assert!((recon.luma[i] - img.luma[i]).abs() < 1e-3);
        assert!((recon.chroma_a[i] - img.chroma_a[i]).abs() < 1e-3);
        assert!((recon.chroma_b[i] - img.chroma_b[i]).abs() < 1e-3);
    }
}