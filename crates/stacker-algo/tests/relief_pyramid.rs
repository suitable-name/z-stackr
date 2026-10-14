#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

//! Tests of the Gaussian-pyramid smoothing behind the `Relief` multigrid engine.

use stacker_algo::relief::pyramid::{pyramid_smooth, pyramid_smooth_owned};

#[test]
fn max_scale_at_most_one_is_the_identity() {
    let field: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    assert_eq!(pyramid_smooth(&field, 8, 8, 0), field);
    assert_eq!(pyramid_smooth(&field, 8, 8, 1), field);
}

#[test]
fn an_empty_field_stays_empty() {
    assert_eq!(pyramid_smooth(&[], 0, 0, 4), vec![]);
    assert_eq!(pyramid_smooth(&[], 0, 5, 4), vec![]);
}

/// A hard step comes out with a graded transition band, while the far interior of
/// each side stays close to its flat value.
#[test]
fn a_step_is_smoothed() {
    let (w, h) = (32, 32);
    let mut field = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in w / 2..w {
            field[y * w + x] = 1.0;
        }
    }

    let smoothed = pyramid_smooth(&field, w, h, 4);

    let y = h / 2;
    assert!(
        smoothed[y * w + 2] < 0.1,
        "left interior should stay near 0"
    );
    assert!(
        smoothed[y * w + (w - 3)] > 0.9,
        "right interior should stay near 1"
    );
    let boundary = smoothed[y * w + w / 2];
    assert!(
        (0.05..0.95).contains(&boundary),
        "the boundary should be smoothed, got {boundary}"
    );
}

/// Every weighted average of a constant is that constant, at any size.
#[test]
fn a_uniform_field_is_a_fixed_point() {
    for (w, h) in [(16, 16), (7, 5), (1, 9), (2, 2), (33, 3), (5, 40)] {
        let field = vec![0.42_f32; w * h];
        for max_scale in [2, 4, 8, 50] {
            let smoothed = pyramid_smooth(&field, w, h, max_scale);
            assert_eq!(smoothed.len(), w * h);
            for (i, &v) in smoothed.iter().enumerate() {
                assert!(
                    (v - 0.42).abs() < 1e-4,
                    "{w}x{h}, scale {max_scale}, index {i}: {v}"
                );
            }
        }
    }
}

/// The kernel weights are positive, so the output stays inside the input range.
#[test]
fn the_output_stays_inside_the_input_range() {
    let (w, h) = (37, 21);
    let field: Vec<f32> = (0..w * h)
        .map(|i| ((((i * 7919) % 101) as f32) / 100.0).mul_add(4.0, -1.0))
        .collect();
    let (lo, hi) = (-1.0 - 1e-4, 3.0 + 1e-4);
    for max_scale in [2, 3, 8] {
        let smoothed = pyramid_smooth(&field, w, h, max_scale);
        assert!(
            smoothed.iter().all(|v| (lo..=hi).contains(v)),
            "scale {max_scale}"
        );
    }
}

/// A larger scale smooths more: the total variation of a noisy field falls.
#[test]
fn more_octaves_smooth_more() {
    let (w, h) = (64, 64);
    let field: Vec<f32> = (0..w * h)
        .map(|i| {
            if (i * 2_654_435_761_usize) % 7 < 3 {
                0.0
            } else {
                1.0
            }
        })
        .collect();
    let variation = |v: &[f32]| -> f32 { v.windows(2).map(|p| (p[1] - p[0]).abs()).sum() };
    let raw = variation(&field);
    let two = variation(&pyramid_smooth(&field, w, h, 2));
    let eight = variation(&pyramid_smooth(&field, w, h, 8));
    assert!(two < raw && eight < two, "{raw} > {two} > {eight} expected");
}

#[test]
fn the_owned_variant_matches_the_borrowed_one() {
    let (w, h) = (19, 13);
    let field: Vec<f32> = (0..w * h).map(|i| ((i * 31) % 17) as f32).collect();
    for max_scale in [0, 1, 2, 5, 16] {
        let borrowed = pyramid_smooth(&field, w, h, max_scale);
        let owned = pyramid_smooth_owned(field.clone(), w, h, max_scale);
        assert_eq!(borrowed, owned);
    }
}
