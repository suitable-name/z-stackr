#![expect(
    clippy::cast_precision_loss,
    reason = "test images are a few dozen pixels wide, so index to f32 conversions are exact"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "x/y/w/h/c/s are the conventional names in the geometry fixtures"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "the fixtures replicate the warp's unfused mul+add order on purpose"
)]

use stacker_align::{
    Matrix3,
    transform::{coverage_mask, intersect_coverage, largest_true_rectangle, warp_image},
};
use stacker_core::image::PlanarImage;

#[test]
fn test_warp_image_identity() {
    let mut img = PlanarImage::<f32>::new(4, 4);
    for (i, v) in img.luma.iter_mut().enumerate() {
        *v = i as f32;
    }

    let matrix = Matrix3::identity();
    let warped = warp_image(&img, &matrix).unwrap();

    assert_eq!(warped.width, 4);
    assert_eq!(warped.height, 4);
    for i in 0..16 {
        assert_eq!(warped.luma[i], img.luma[i]);
    }
}

/// The SIMD body and the scalar tail of `warp_image` are specified to agree bit
/// for bit with the scalar bicubic sampler, whatever vector width the build
/// selected (4, 8 or 16 lanes). Widths 50 and 37 leave a scalar tail after the
/// vector body at every width (e.g. 48 + 2 and 32 + 5 at 16 lanes), 16 is whole
/// vectors, and 5 is no more than one vector with a tail, so the vector body and
/// the tail are both compared at each width.
#[test]
fn test_warp_simd_matches_scalar_reference() {
    for w in [50usize, 37, 16, 5] {
        assert_warp_matches_scalar_reference(w, 32);
    }
}

fn assert_warp_matches_scalar_reference(w: usize, h: usize) {
    use stacker_align::transform::bicubic_interpolate_channel;

    let mut img = PlanarImage::<f32>::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            // High-frequency content so any coordinate/weight error shows up.
            img.luma[idx] = ((x * 5 + y * 3) % 13) as f32 / 13.0;
            img.chroma_a[idx] = ((x + 2 * y) % 11) as f32 / 11.0 - 0.5;
            img.chroma_b[idx] = ((3 * x + y) % 17) as f32 / 17.0 - 0.5;
        }
    }

    // Non-trivial similarity transform: 4° rotation × 1.06 scale + translation.
    let ang = 4.0_f32.to_radians();
    let (cs, sn) = (ang.cos() * 1.06, ang.sin() * 1.06);
    let mut m = Matrix3::<f32>::identity();
    m[(0, 0)] = cs;
    m[(0, 1)] = -sn;
    m[(1, 0)] = sn;
    m[(1, 1)] = cs;
    m[(0, 2)] = 2.4;
    m[(1, 2)] = -1.3;

    let warped = warp_image(&img, &m).unwrap();

    // Independent scalar reference: replicate the warp's inverse-map (same op
    // order, including the reciprocal) and sample with the scalar bicubic.
    let inv = m.try_inverse().unwrap();
    let (m00, m01, m02) = (inv[(0, 0)], inv[(0, 1)], inv[(0, 2)]);
    let (m10, m11, m12) = (inv[(1, 0)], inv[(1, 1)], inv[(1, 2)]);
    let (m20, m21, m22) = (inv[(2, 0)], inv[(2, 1)], inv[(2, 2)]);
    for y in 0..h {
        let yf = y as f32;
        let c0 = m01 * yf + m02;
        let c1 = m11 * yf + m12;
        let c2 = m21 * yf + m22;
        for x in 0..w {
            let xf = x as f32;
            let inv_p2 = 1.0 / (m20 * xf + c2);
            let sx = (m00 * xf + c0) * inv_p2;
            let sy = (m10 * xf + c1) * inv_p2;
            let i = y * w + x;
            let rl = bicubic_interpolate_channel(&img.luma, w, h, sx, sy);
            let ra = bicubic_interpolate_channel(&img.chroma_a, w, h, sx, sy);
            let rb = bicubic_interpolate_channel(&img.chroma_b, w, h, sx, sy);
            assert_eq!(warped.luma[i], rl, "luma ({x},{y}) at width {w}");
            assert_eq!(warped.chroma_a[i], ra, "chroma_a ({x},{y}) at width {w}");
            assert_eq!(warped.chroma_b[i], rb, "chroma_b ({x},{y}) at width {w}");
        }
    }
}

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

/// ALN-14: `coverage_mask` and `warp_image` share one inverse-mapping routine,
/// so a pixel is covered exactly when the warp produced real (non-zero-fill)
/// data.  A constant-1 luma plane makes every covered pixel non-zero and every
/// zero-filled pixel exactly 0.
fn assert_mask_matches_zero_fill(m: &Matrix3<f32>, w: usize, h: usize) {
    let mut img = PlanarImage::<f32>::new(w, h);
    img.luma.fill(1.0);

    let warped = warp_image(&img, m).unwrap();
    let mask = coverage_mask(m, w, h);

    assert_eq!(mask.len(), w * h);
    let mut covered = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let sampled = warped.luma[i] != 0.0;
            assert_eq!(
                mask[i], sampled,
                "pixel ({x},{y}): mask {} but warp luma {}",
                mask[i], warped.luma[i]
            );
            covered += usize::from(mask[i]);
        }
    }
    // Guard against a vacuous pass: some pixels covered, some not.
    assert!(
        covered > 0 && covered < w * h,
        "covered {covered} of {}",
        w * h
    );
}

#[test]
fn test_coverage_mask_matches_warp_zero_fill_translation() {
    // Width 37 exercises both the SIMD body and the scalar tail.
    let (w, h) = (37, 29);
    let mut m = Matrix3::<f32>::identity();
    m[(0, 2)] = 3.0; // exact-integer shift: source coordinates land on the edge
    m[(1, 2)] = -2.5;
    assert_mask_matches_zero_fill(&m, w, h);

    let mut m2 = Matrix3::<f32>::identity();
    m2[(0, 2)] = -5.3;
    m2[(1, 2)] = 4.0;
    assert_mask_matches_zero_fill(&m2, w, h);
}

#[test]
fn test_coverage_mask_matches_warp_zero_fill_rotation() {
    let (w, h) = (37, 29);
    let ang = 7.0_f32.to_radians();
    let (c, s) = (ang.cos(), ang.sin());
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let mut m = Matrix3::<f32>::identity();
    m[(0, 0)] = c;
    m[(0, 1)] = -s;
    m[(1, 0)] = s;
    m[(1, 1)] = c;
    m[(0, 2)] = cx - c * cx + s * cy + 1.5;
    m[(1, 2)] = cy - s * cx - c * cy - 0.75;
    assert_mask_matches_zero_fill(&m, w, h);
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
