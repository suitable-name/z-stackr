#![expect(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "test images are at most 101 px wide; the reference blur mirrors the signed reflect-index arithmetic"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "the reference blur accumulates with plain multiply-add in kernel order, exactly as the production blur does"
)]

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
    PlanarImage::from_planes(w, h, luma, a, b).expect("one sample per pixel")
}

#[test]
fn test_blur_propagation() {
    let img = create_test_image(16, 16);
    let blurred = apply_gaussian_blur(&img);
    assert_eq!(blurred.width, 16);
    assert_eq!(blurred.height, 16);
    assert!(blurred.luma[8 * 16 + 8].is_finite());
}

/// Independent scalar reference separable blur (the pre-SIMD implementation),
/// used to pin the SIMD `blur_channel` to identical results.
fn reference_blur(width: usize, height: usize, data: &[f32]) -> Vec<f32> {
    let w2 = 0.25_f32 - 0.33_f32 / 2.0;
    let kernel = [w2, 0.25_f32, 0.33_f32, 0.25_f32, w2];
    let reflect = |i: isize, n: isize| -> usize {
        let mut k = i;
        if k < 0 {
            k = -k;
        } else if k >= n {
            k = 2 * n - 2 - k;
        }
        if k < 0 || k >= n {
            k = 0;
        }
        k as usize
    };
    let (w, h) = (width as isize, height as isize);
    let mut temp = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut s = 0.0_f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let nx = reflect(x as isize + ki as isize - 2, w);
                s += k * data[y * width + nx];
            }
            temp[y * width + x] = s;
        }
    }
    let mut out = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut s = 0.0_f32;
            for (ki, &k) in kernel.iter().enumerate() {
                let ny = reflect(y as isize + ki as isize - 2, h);
                s += k * temp[ny * width + x];
            }
            out[y * width + x] = s;
        }
    }
    out
}

/// The SIMD blur is documented as bit-identical to the scalar form (same taps,
/// same accumulation order, no fused multiply-add), so the comparison is exact:
/// the luma values reach 30, where a `1e-5` tolerance would still admit one ulp
/// of drift.
#[test]
fn test_simd_blur_matches_scalar_reference() {
    // Sizes spanning sub-SIMD-width, exact multiples, and ragged remainders, so
    // the boundary, vector body, and scalar-tail paths are all exercised.
    for &(w, h) in &[(1, 1), (3, 3), (4, 4), (16, 16), (17, 5), (33, 31), (70, 9)] {
        let img = create_test_image(w, h);
        let blurred = apply_gaussian_blur(&img);
        let reference = reference_blur(w, h, &img.luma);
        for (i, (&b, &r)) in blurred.luma.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                b.to_bits(),
                r.to_bits(),
                "SIMD blur differs from scalar reference at index {i} for {w}×{h}: {b} vs {r}"
            );
        }
    }
}

// ── PERF-4: decimating downsample and polyphase expand ───────────────────
//
// The production `downsample` blurs only the pixels it keeps and `expand`
// skips the zero taps.  Both are pinned to the literal definitions below
// (blur the whole image with `apply_gaussian_blur`, then decimate; zero-insert,
// blur, times four) and must agree bit for bit.  The images contain no exact
// zeros, so the one tolerated difference (the sign of an exact zero) cannot
// occur.

/// Deterministic pseudo-random plane in (-1, 1), never exactly zero.
fn noise_image(w: usize, h: usize, seed: u32) -> PlanarImage<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    let mut plane = |offset: f32| -> Vec<f32> {
        (0..w * h)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32).mul_add(1.998, -0.999) + offset
            })
            .collect()
    };
    let luma = plane(0.0);
    let chroma_a = plane(0.0017);
    let chroma_b = plane(-0.0031);
    PlanarImage::from_planes(w, h, luma, chroma_a, chroma_b).expect("one sample per pixel")
}

fn reference_downsample(img: &PlanarImage<f32>) -> PlanarImage<f32> {
    let blurred = apply_gaussian_blur(img);
    let new_w = img.width.div_ceil(2);
    let new_h = img.height.div_ceil(2);
    let pick = |plane: &[f32]| -> Vec<f32> {
        let mut out = Vec::with_capacity(new_w * new_h);
        for y in 0..new_h {
            for x in 0..new_w {
                out.push(plane[2 * y * img.width + 2 * x]);
            }
        }
        out
    };
    PlanarImage::from_planes(
        new_w,
        new_h,
        pick(&blurred.luma),
        pick(&blurred.chroma_a),
        pick(&blurred.chroma_b),
    )
    .expect("one sample per pixel")
}

fn reference_expand(img: &PlanarImage<f32>, w: usize, h: usize) -> PlanarImage<f32> {
    let scatter = |plane: &[f32]| -> Vec<f32> {
        let mut canvas = vec![0.0_f32; w * h];
        for y in 0..img.height {
            for x in 0..img.width {
                if 2 * x < w && 2 * y < h {
                    canvas[2 * y * w + 2 * x] = plane[y * img.width + x];
                }
            }
        }
        canvas
    };
    let canvas = PlanarImage::from_planes(
        w,
        h,
        scatter(&img.luma),
        scatter(&img.chroma_a),
        scatter(&img.chroma_b),
    )
    .expect("one sample per pixel");
    let mut blurred = apply_gaussian_blur(&canvas);
    for plane in [
        &mut blurred.luma,
        &mut blurred.chroma_a,
        &mut blurred.chroma_b,
    ] {
        for v in plane.iter_mut() {
            *v *= 4.0;
        }
    }
    blurred
}

fn assert_same_bits(got: &PlanarImage<f32>, want: &PlanarImage<f32>, what: &str) {
    assert_eq!((got.width, got.height), (want.width, want.height), "{what}");
    for (name, got_plane, want_plane) in [
        ("luma", &got.luma, &want.luma),
        ("chroma_a", &got.chroma_a, &want.chroma_a),
        ("chroma_b", &got.chroma_b, &want.chroma_b),
    ] {
        assert_eq!(got_plane.len(), want_plane.len(), "{what}: {name} length");
        for (idx, (lhs, rhs)) in got_plane.iter().zip(want_plane).enumerate() {
            assert_eq!(
                lhs.to_bits(),
                rhs.to_bits(),
                "{what}: {name}[{idx}] is {lhs} vs {rhs}"
            );
        }
    }
}

/// Shapes spanning one-pixel dimensions, widths below and above the vector width
/// (4, 8 or 16 lanes by target feature), odd and even sizes, and the
/// reflecting-boundary corner cases.
const SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (1, 7),
    (7, 1),
    (2, 2),
    (3, 3),
    (4, 5),
    (5, 4),
    (6, 6),
    (15, 9),
    (16, 16),
    (17, 5),
    (31, 33),
    (33, 31),
    (64, 48),
    (70, 9),
    (101, 67),
];

#[test]
fn test_downsample_is_bit_identical_to_blur_then_decimate() {
    for (seed, &(w, h)) in (1_u32..).zip(SHAPES) {
        let img = noise_image(w, h, seed);
        assert_same_bits(
            &downsample(&img),
            &reference_downsample(&img),
            &format!("downsample {w}x{h}"),
        );
    }
}

#[test]
fn test_expand_is_bit_identical_to_zero_insert_and_blur() {
    // The small image covers the canvas: ceil(w/2) x ceil(h/2), the size the
    // pyramid produces (the polyphase path).
    for (seed, &(w, h)) in (100_u32..).zip(SHAPES) {
        let small = noise_image(w.div_ceil(2), h.div_ceil(2), seed);
        assert_same_bits(
            &expand(&small, w, h),
            &reference_expand(&small, w, h),
            &format!("expand to {w}x{h}"),
        );
    }
}

#[test]
fn test_expand_handles_a_source_that_does_not_cover_or_overshoots_the_canvas() {
    // Smaller than the canvas needs (takes the literal definition) and larger
    // than it needs (surplus columns and rows are ignored).
    for &(sw, sh, w, h) in &[
        (3_usize, 3_usize, 20_usize, 14_usize),
        (10, 4, 20, 14),
        (4, 10, 20, 14),
        (20, 20, 9, 11),
        (12, 9, 9, 11),
    ] {
        let small = noise_image(sw, sh, 7);
        assert_same_bits(
            &expand(&small, w, h),
            &reference_expand(&small, w, h),
            &format!("expand {sw}x{sh} to {w}x{h}"),
        );
    }
}

#[test]
fn test_empty_canvas_does_not_panic() {
    let small = noise_image(3, 3, 1);
    let e = expand(&small, 0, 5);
    assert_eq!((e.width, e.height, e.luma.len()), (0, 5, 0));
    let d = downsample(&PlanarImage::new(0, 0));
    assert_eq!((d.width, d.height, d.luma.len()), (0, 0, 0));
}

/// The whole pyramid (which now runs on the decimating/polyphase forms)
/// reconstructs the input.
#[test]
fn test_pyramid_reconstructs_noise_for_ragged_sizes() {
    for &(w, h) in &[(17_usize, 13_usize), (40, 31), (64, 64)] {
        let img = noise_image(w, h, 5);
        let recon = LaplacianPyramid::build(&img, 4).reconstruct();
        assert_eq!((recon.width, recon.height), (w, h));
        for i in 0..w * h {
            assert!(
                (recon.luma[i] - img.luma[i]).abs() < 1e-3,
                "{w}x{h} luma {i}"
            );
            assert!((recon.chroma_a[i] - img.chroma_a[i]).abs() < 1e-3);
            assert!((recon.chroma_b[i] - img.chroma_b[i]).abs() < 1e-3);
        }
    }
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
