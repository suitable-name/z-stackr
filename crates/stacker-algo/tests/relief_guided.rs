use stacker_algo::relief::guided::*;
use stacker_core::image::PlanarImage;

fn make_image(width: usize, height: usize, val: f32) -> PlanarImage<f32> {
    PlanarImage::with_luma(width, height, vec![val; width * height]).expect("one sample per pixel")
}

/// A constant image filtered with a constant guide is a fixed point of the
/// guided filter for every window size, including windows larger than the image
/// (the filter is built from clamped box means, so the radius must not matter).
/// This replaces a wall-clock ratio, which measured scheduler noise on a
/// constant image and whose first call paid the rayon start-up.
#[test]
fn test_constant_image_is_a_fixed_point_for_every_radius() {
    let (w, h) = (100, 100);
    let guidance = make_image(w, h, 0.5);
    let src = make_image(w, h, 0.5);

    for radius in [0_usize, 1, 2, 10, 49, 150] {
        let out = guided_filter(&guidance, &src, radius, 0.1);
        assert_eq!((out.width, out.height), (w, h));
        let worst = out
            .luma
            .iter()
            .map(|&v| (v - 0.5).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-6, "radius {radius}: worst deviation {worst}");
    }
}

/// FUS-2: the summed-area table must not lose precision on large frames.
///
/// At 2048 × 2048 the bottom-right prefix sum of a constant 0.7 image is about
/// 2.9e6, where the `f32` spacing is 0.25, so an `f32` table makes the radius-2
/// window mean wrong by up to ~1e-2 (and by ~0.19 at 24 MP).  A constant image
/// filtered with a constant guide must come back as that constant.
#[test]
fn test_large_constant_image_keeps_precision() {
    let (w, h) = (2048, 2048);
    let guidance = make_image(w, h, 0.7);
    let src = make_image(w, h, 0.7);

    let out = guided_filter(&guidance, &src, 2, 0.01);

    let worst = out
        .luma
        .iter()
        .map(|&v| (v - 0.7).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst < 1e-5, "worst deviation from 0.7 is {worst}");
}

/// The self-guided shortcut (the same slice as guidance and source skips two box
/// filters) gives exactly the bits of the general path fed an equal copy.
#[test]
fn test_self_guided_shortcut_is_bit_identical_to_the_general_path() {
    let (w, h) = (37, 29);
    let plane: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            f32::from(u8::try_from((x * 7 + y * 13) % 31).expect("below 31")) / 31.0
        })
        .collect();
    let copy = plane.clone();
    for radius in [0_usize, 1, 3, 50] {
        let shortcut = guided_filter_luma(&plane, &plane, w, h, radius, 0.01);
        let general = guided_filter_luma(&plane, &copy, w, h, radius, 0.01);
        assert_eq!(shortcut.len(), general.len());
        for (i, (a, b)) in shortcut.iter().zip(&general).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "radius {radius}, sample {i}");
        }
    }
}

#[test]
fn test_halo_suppression_reference() {
    let width = 10;
    let height = 10;
    let mut guidance = make_image(width, height, 0.0);
    let mut src = make_image(width, height, 0.0);

    for y in 0..height {
        for x in 5..width {
            guidance.luma[y * width + x] = 1.0;
            src.luma[y * width + x] = 1.0;
        }
    }

    let result = guided_filter(&guidance, &src, 2, 0.01);

    let val_left = result.luma[5 * width + 4];
    let val_right = result.luma[5 * width + 5];

    assert!(val_left < 0.1, "Edge blurred to the left");
    assert!(val_right > 0.9, "Edge blurred to the right");
}
