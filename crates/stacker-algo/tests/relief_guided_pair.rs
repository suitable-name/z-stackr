#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

//! Tests of the public box filter and of the guided-filter pair that shares the
//! summed-area tables between two `(radius, eps)` settings.

use stacker_algo::relief::guided::{box_filter, guided_filter_luma, guided_filter_pair_luma};

/// A deterministic plane of values in `[0, 1)`.
fn plane(width: usize, height: usize, seed: u32) -> Vec<f32> {
    let mut state = 0x9E37_79B9_u32 ^ seed.wrapping_mul(0x85EB_CA6B);
    (0..width * height)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).expect("16 bits")) / 65_536.0
        })
        .collect()
}

#[test]
fn box_filter_is_the_clipped_window_mean() {
    // 3x3 plane 1..=9: the centre window (radius 1) holds everything, a corner
    // window only four samples.
    let data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let out = box_filter(&data, 3, 3, 1);
    assert!((out[4] - 5.0).abs() < 1e-6, "centre");
    assert!(
        (out[0] - (1.0 + 2.0 + 4.0 + 5.0) / 4.0).abs() < 1e-6,
        "corner"
    );
    assert!(
        (out[1] - (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6.0).abs() < 1e-6,
        "edge"
    );

    // Radius 0 is the identity, a huge radius the global mean.
    assert_eq!(box_filter(&data, 3, 3, 0), data);
    for v in box_filter(&data, 3, 3, 100) {
        assert!((v - 5.0).abs() < 1e-6);
    }
}

#[test]
fn box_filter_of_an_empty_plane_is_empty() {
    assert_eq!(box_filter(&[], 0, 0, 3), vec![]);
    assert_eq!(box_filter(&[], 5, 0, 3), vec![]);
    assert_eq!(box_filter(&[], 0, 5, 3), vec![]);
}

/// The summed-area table is accumulated in `f64`: a large constant plane comes
/// back as that constant, including the bottom-right corner where an `f32` table
/// loses the most.
#[test]
fn box_filter_of_a_large_constant_plane_keeps_precision() {
    let (w, h) = (2048, 1536);
    let data = vec![0.734_f32; w * h];
    let out = box_filter(&data, w, h, 4);
    let worst = out
        .iter()
        .map(|&v| (v - 0.734).abs())
        .fold(0.0_f32, f32::max);
    assert!(worst < 1e-6, "worst deviation {worst}");
}

/// The parallel table build agrees with a naive window mean.
#[test]
fn box_filter_matches_a_naive_window_mean() {
    for (w, h) in [(1, 1), (1, 9), (9, 1), (13, 7), (40, 33)] {
        let data = plane(w, h, 3);
        for radius in [0_usize, 1, 2, 5, 60] {
            let fast = box_filter(&data, w, h, radius);
            for y in 0..h {
                for x in 0..w {
                    let (y0, y1) = (y.saturating_sub(radius), (y + radius).min(h - 1));
                    let (x0, x1) = (x.saturating_sub(radius), (x + radius).min(w - 1));
                    let mut sum = 0.0_f64;
                    for yy in y0..=y1 {
                        for xx in x0..=x1 {
                            sum += f64::from(data[yy * w + xx]);
                        }
                    }
                    let want = sum / ((y1 - y0 + 1) * (x1 - x0 + 1)) as f64;
                    assert!(
                        (f64::from(fast[y * w + x]) - want).abs() < 1e-6,
                        "{w}x{h} r{radius} ({x},{y})"
                    );
                }
            }
        }
    }
}

/// The pair equals two separate calls, to within 1e-6.
#[test]
fn the_pair_equals_two_separate_calls() {
    for (w, h) in [(1, 1), (5, 3), (37, 29), (64, 48)] {
        let guidance = plane(w, h, 1);
        let src = plane(w, h, 2);
        for ((radius_a, eps_a), (radius_b, eps_b)) in [
            ((45, 0.3_f32), (7, 1e-6_f32)),
            ((1, 0.01), (2, 0.1)),
            ((0, 0.5), (3, 0.5)),
        ] {
            let (pair_a, pair_b) =
                guided_filter_pair_luma(&guidance, &src, w, h, radius_a, eps_a, radius_b, eps_b);
            let one_a = guided_filter_luma(&guidance, &src, w, h, radius_a, eps_a);
            let one_b = guided_filter_luma(&guidance, &src, w, h, radius_b, eps_b);
            assert_eq!(pair_a.len(), w * h);
            assert_eq!(pair_b.len(), w * h);
            for i in 0..w * h {
                assert!((pair_a[i] - one_a[i]).abs() < 1e-6, "{w}x{h} a sample {i}");
                assert!((pair_b[i] - one_b[i]).abs() < 1e-6, "{w}x{h} b sample {i}");
            }
        }
    }
}

/// With the same slice as guidance and source the pair takes the self-guided
/// route and still equals the single-call results.
#[test]
fn the_self_guided_pair_equals_two_separate_calls() {
    let (w, h) = (41, 27);
    let image = plane(w, h, 9);
    let (pair_a, pair_b) = guided_filter_pair_luma(&image, &image, w, h, 6, 0.01, 2, 1e-4);
    let one_a = guided_filter_luma(&image, &image, w, h, 6, 0.01);
    let one_b = guided_filter_luma(&image, &image, w, h, 2, 1e-4);
    for i in 0..w * h {
        assert!((pair_a[i] - one_a[i]).abs() < 1e-6, "sample {i}");
        assert!((pair_b[i] - one_b[i]).abs() < 1e-6, "sample {i}");
    }

    // The same values in a separate allocation take the general route: equal bits.
    let copy = image.clone();
    let (general_a, general_b) = guided_filter_pair_luma(&image, &copy, w, h, 6, 0.01, 2, 1e-4);
    assert!(
        pair_a
            .iter()
            .zip(&general_a)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        pair_b
            .iter()
            .zip(&general_b)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
fn the_pair_of_empty_planes_is_empty() {
    let (a, b) = guided_filter_pair_luma(&[], &[], 0, 0, 3, 0.1, 5, 0.2);
    assert!(a.is_empty() && b.is_empty());
}

/// A constant plane is a fixed point of both chains. With `eps = 1e-6` the
/// variance of such a window would be a rounding residue (the clamp keeps it from
/// going negative and the quotient from blowing up).
#[test]
fn a_constant_plane_is_a_fixed_point_of_the_pair() {
    let (w, h) = (300, 200);
    let data = vec![0.7_f32; w * h];
    let (a, b) = guided_filter_pair_luma(&data, &data, w, h, 45, 0.3, 7, 1e-6);
    for (name, out) in [("a", &a), ("b", &b)] {
        let worst = out.iter().map(|&v| (v - 0.7).abs()).fold(0.0_f32, f32::max);
        assert!(worst < 1e-5, "chain {name}: worst deviation {worst}");
    }
}

/// An edge in the guidance keeps its sharpness in the tight chain and the wide
/// chain leaks more across it.
#[test]
fn the_tight_chain_keeps_an_edge_sharper_than_the_wide_one() {
    let (w, h) = (40, 20);
    let edge: Vec<f32> = (0..w * h)
        .map(|i| if i % w < w / 2 { 0.1 } else { 0.9 })
        .collect();
    let (wide, tight) = guided_filter_pair_luma(&edge, &edge, w, h, 12, 0.3, 2, 1e-6);
    let row = (h / 2) * w;
    let jump = |out: &[f32]| out[row + w / 2] - out[row + w / 2 - 1];
    assert!(
        jump(&tight) > jump(&wide),
        "{} vs {}",
        jump(&tight),
        jump(&wide)
    );
    assert!(jump(&tight) > 0.6);
}
