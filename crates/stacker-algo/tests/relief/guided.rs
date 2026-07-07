#![allow(clippy::cast_precision_loss)]

use stacker_algo::relief::guided::*;
use stacker_core::image::PlanarImage;
use std::time::Instant;

fn make_image(width: usize, height: usize, val: f32) -> PlanarImage<f32> {
    PlanarImage {
        width,
        height,
        luma: vec![val; width * height],
        chroma_a: vec![0.0; width * height],
        chroma_b: vec![0.0; width * height],
    }
}

#[test]
fn test_linear_time_complexity() {
    let w = 100;
    let h = 100;
    let guidance = make_image(w, h, 0.5);
    let src = make_image(w, h, 0.5);

    let start1 = Instant::now();
    let _ = guided_filter(&guidance, &src, 2, 0.1);
    let t1 = start1.elapsed().as_secs_f64();

    let start2 = Instant::now();
    let _ = guided_filter(&guidance, &src, 10, 0.1);
    let t2 = start2.elapsed().as_secs_f64();

    let ratio = t2 / t1.max(1e-9);
    assert!(
        ratio < 5.0,
        "Execution time should not scale quadratically with radius"
    );
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

/// Regression test for the f32 integral-image precision-loss bug (task 9384):
/// at a few megapixels the running SAT sum reaches ~1e6-1e7, where an f32
/// accumulator's ULP rivals the small per-window sums recovered by the
/// 4-corner difference, catastrophically cancelling and injecting full-frame
/// noise. A large constant-valued image must box-filter back to that exact
/// constant everywhere, including the bottom-right corner where the SAT
/// accumulation error is largest.
#[test]
fn test_box_filter_constant_field_large_image_precision() {
    let width = 3000;
    let height = 1300;
    let value = 0.734_f32;
    let img = make_image(width, height, value);

    // guided_filter(img, img, r, eps) with a constant guidance/source field
    // degenerates to q = mean_a * I + mean_b, which for a perfectly constant
    // field must reproduce the input value at every pixel. Use box_filter's
    // public sibling behaviour via guided_filter itself so the test also
    // exercises the full pipeline (mean_i, mean_p, var_i, a, b, mean_a, mean_b).
    let result = guided_filter(&img, &img, 4, 1e-4);

    let mut max_err = 0.0_f32;
    let mut max_err_idx = 0usize;
    for (i, &v) in result.luma.iter().enumerate() {
        let err = (v - value).abs();
        if err > max_err {
            max_err = err;
            max_err_idx = i;
        }
    }
    assert!(
        max_err < 1e-4,
        "constant field must survive box/guided filtering within 1e-4 \
         (max_err={max_err} at flat index {max_err_idx}, i.e. ({}, {}))",
        max_err_idx % width,
        max_err_idx / width,
    );

    // Explicitly check the bottom-right corner: the integral image
    // accumulates left-to-right, top-to-bottom, so any f32 cancellation
    // error grows monotonically toward the bottom-right and was the first
    // place the original bug's noise artifact became visible.
    let br = result.luma[(height - 1) * width + (width - 1)];
    assert!(
        (br - value).abs() < 1e-4,
        "bottom-right pixel should equal the constant field value within 1e-4, got {br}"
    );
}

/// A smooth gradient guided by itself should pass through the guided filter
/// close to unchanged (guided filter with guidance == src is a near-identity
/// operator away from strong edges), even at large image sizes where the SAT
/// precision bug would otherwise inject noise.
#[test]
fn test_guided_filter_smooth_gradient_self_guided_stays_close_to_input() {
    let width = 2048;
    let height = 1024;
    let mut img = make_image(width, height, 0.0);
    for y in 0..height {
        for x in 0..width {
            // Smooth horizontal gradient in 0..1.
            img.luma[y * width + x] = x as f32 / (width - 1) as f32;
        }
    }

    let result = guided_filter(&img, &img, 4, 1e-4);

    let mut max_err = 0.0_f32;
    for (out, inp) in result.luma.iter().zip(img.luma.iter()) {
        let err = (out - inp).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 1e-2,
        "self-guided filtering of a smooth gradient should stay close to the \
         input (max_err={max_err})"
    );
}
