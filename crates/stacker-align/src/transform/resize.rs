use rayon::prelude::*;
use stacker_core::image::PlanarImage;

use crate::transform::warp::spline4x4_sample_clamped;

// ── Resampling (crop → original-canvas restretch) ────────────────────────────
//
// `resize_cropped_to_original` optionally stretches a common-coverage-cropped
// stack back to the original canvas resolution. The crop rectangle's aspect
// ratio can differ fractionally from the canvas (focus-breathing crops are
// rarely perfectly square-proportioned to the source), so this is a slight
// non-uniform (independent X/Y scale) resize rather than a uniform one.

/// Resize `src` to `(new_w, new_h)` using the 4-tap spline kernel with
/// edge-clamped borders ([`spline4x4_sample_clamped`]), the same
/// high-quality resampler [`warp_image_clamped`] uses for warping.
///
/// Used to optionally stretch a common-coverage-cropped stack back to the
/// original canvas resolution (see
/// `stacker_core::settings::StackingSettings::resize_cropped_to_original`).
/// Edge-clamped sampling means there is no border ringing or zero-fill, even
/// though the source and destination aspect ratios can differ slightly.
///
/// Returns a clone of `src` unchanged when `(new_w, new_h)` already matches
/// `(src.width, src.height)`, or when either requested dimension is `0`
/// (nothing sensible to resize to).
#[must_use]
pub fn resize_planar_clamped(
    src: &PlanarImage<f32>,
    new_w: usize,
    new_h: usize,
) -> PlanarImage<f32> {
    if new_w == 0 || new_h == 0 || (new_w == src.width && new_h == src.height) {
        return src.clone();
    }

    let (src_w, src_h) = (src.width, src.height);
    let luma_src = &src.luma;
    let src_chroma_a = &src.chroma_a;
    let chroma_b_src_data = &src.chroma_b;

    let scale_x = src_w as f32 / new_w as f32;
    let scale_y = src_h as f32 / new_h as f32;

    let mut dst = PlanarImage::new(new_w, new_h);
    dst.luma
        .par_chunks_mut(new_w)
        .zip(dst.chroma_a.par_chunks_mut(new_w))
        .zip(dst.chroma_b.par_chunks_mut(new_w))
        .enumerate()
        .for_each(|(y, ((row_l, row_a), row_b))| {
            // Pixel-centre mapping: dst pixel (x, y)'s centre maps to src
            // coordinate ((x + 0.5) * scale - 0.5), so a same-size resize
            // (scale == 1) is the identity sampling grid.
            let sy = (y as f32 + 0.5).mul_add(scale_y, -0.5);
            for x in 0..new_w {
                let sx = (x as f32 + 0.5).mul_add(scale_x, -0.5);
                row_l[x] = spline4x4_sample_clamped(luma_src, src_w, src_h, sx, sy);
                row_a[x] = spline4x4_sample_clamped(src_chroma_a, src_w, src_h, sx, sy);
                row_b[x] = spline4x4_sample_clamped(chroma_b_src_data, src_w, src_h, sx, sy);
            }
        });

    dst
}

#[cfg(test)]
mod resize_planar_clamped_tests {
    use super::resize_planar_clamped;
    use stacker_core::image::PlanarImage;

    fn constant_image(w: usize, h: usize, v: f32) -> PlanarImage<f32> {
        let mut img = PlanarImage::new(w, h);
        img.luma.fill(v);
        img.chroma_a.fill(v * 0.5);
        img.chroma_b.fill(-v * 0.25);
        img
    }

    #[test]
    fn same_size_is_identity_clone() {
        let src = constant_image(12, 9, 0.42);
        let out = resize_planar_clamped(&src, 12, 9);
        assert_eq!(out.width, 12);
        assert_eq!(out.height, 9);
        assert_eq!(out.luma, src.luma);
        assert_eq!(out.chroma_a, src.chroma_a);
        assert_eq!(out.chroma_b, src.chroma_b);
    }

    #[test]
    fn zero_dim_is_identity_clone() {
        let src = constant_image(8, 8, 0.1);
        let out = resize_planar_clamped(&src, 0, 20);
        assert_eq!(out.width, src.width);
        assert_eq!(out.height, src.height);
        let out2 = resize_planar_clamped(&src, 20, 0);
        assert_eq!(out2.width, src.width);
        assert_eq!(out2.height, src.height);
    }

    #[test]
    fn constant_image_stays_constant_after_upscale() {
        let src = constant_image(10, 10, 0.75);
        let out = resize_planar_clamped(&src, 40, 25);
        assert_eq!(out.width, 40);
        assert_eq!(out.height, 25);
        for &v in &out.luma {
            assert!((v - 0.75).abs() < 1.0e-4, "got {v}");
        }
        for &v in &out.chroma_a {
            assert!((v - 0.375).abs() < 1.0e-4, "got {v}");
        }
        for &v in &out.chroma_b {
            assert!((v - (-0.1875)).abs() < 1.0e-4, "got {v}");
        }
    }

    #[test]
    fn output_dims_correct_upscale_and_downscale() {
        let src = constant_image(16, 16, 0.3);
        let up = resize_planar_clamped(&src, 32, 20);
        assert_eq!((up.width, up.height), (32, 20));
        let down = resize_planar_clamped(&src, 8, 5);
        assert_eq!((down.width, down.height), (8, 5));
    }

    #[test]
    fn horizontal_ramp_stays_monotonic_after_upscale() {
        // Linear horizontal ramp: luma increases strictly left-to-right.
        let (w, h) = (10usize, 4usize);
        let mut src = PlanarImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                src.luma[y * w + x] = x as f32 / (w - 1) as f32;
            }
        }
        let out = resize_planar_clamped(&src, 40, 4);
        for y in 0..out.height {
            let row = &out.luma[y * out.width..(y + 1) * out.width];
            for pair in row.windows(2) {
                assert!(
                    pair[1] + 1.0e-4 >= pair[0],
                    "row {y} not monotonic non-decreasing: {pair:?}"
                );
            }
        }
    }
}
