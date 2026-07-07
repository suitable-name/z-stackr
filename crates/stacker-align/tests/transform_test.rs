#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]

use stacker_align::{
    Matrix3,
    transform::{coverage_mask, intersect_coverage, largest_true_rectangle},
};

#[test]
fn test_coverage_mask_identity_is_all_true() {
    let m = Matrix3::<f32>::identity();
    let mask = coverage_mask(&m, 8, 6);
    assert_eq!(mask.len(), 48);
    assert!(mask.iter().all(|&v| v), "identity must cover every pixel");
}

#[test]
fn test_coverage_mask_translation_drops_border() {
    // Forward translation of +3 px in x: output column x maps back to x-3, so
    // the left 3 columns map outside the source and are invalid.
    let mut m = Matrix3::<f32>::identity();
    m[(0, 2)] = 3.0;
    let (w, h) = (10, 4);
    let mask = coverage_mask(&m, w, h);
    for y in 0..h {
        for x in 0..w {
            let valid = mask[y * w + x];
            // x - 3 must be within [0, w-1]  ->  x >= 3
            assert_eq!(valid, x >= 3, "pixel ({x},{y})");
        }
    }
}

#[test]
fn test_intersect_coverage_ands_masks() {
    let mut a = vec![true, true, true, false];
    let b = vec![true, false, true, true];
    intersect_coverage(&mut a, &b);
    assert_eq!(a, vec![true, false, true, false]);
}

#[test]
fn test_largest_true_rectangle_basic() {
    // 4×4 mask, all true except the rightmost column and bottom row ->
    // the largest all-true rectangle is the top-left 3×3 block at (0,0).
    let (w, h) = (4, 4);
    let mut mask = vec![true; w * h];
    for y in 0..h {
        mask[y * w + (w - 1)] = false; // last column
    }
    for x in 0..w {
        mask[(h - 1) * w + x] = false; // last row
    }
    let rect = largest_true_rectangle(&mask, w, h).expect("non-empty");
    assert_eq!(rect, (0, 0, 3, 3));
}

#[test]
fn test_largest_true_rectangle_all_false_is_none() {
    let mask = vec![false; 9];
    assert_eq!(largest_true_rectangle(&mask, 3, 3), None);
}

#[test]
fn test_crop_rect_from_two_translations() {
    // Reference (identity) + one frame shifted +2 in x and +1 in y. The common
    // valid rectangle excludes the 2 left columns and the top row.
    let w = 8;
    let h = 8;
    let mut common = vec![true; w * h]; // reference covers everything
    let mut m = Matrix3::<f32>::identity();
    m[(0, 2)] = 2.0;
    m[(1, 2)] = 1.0;
    intersect_coverage(&mut common, &coverage_mask(&m, w, h));
    let (x, y, rw, rh) = largest_true_rectangle(&common, w, h).expect("non-empty");
    assert_eq!((x, y), (2, 1), "crop origin must skip the warped-in border");
    assert_eq!((rw, rh), (w - 2, h - 1));
}
