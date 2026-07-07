use rayon::prelude::*;
use stacker_core::image::PlanarImage;

/// Minimum short-side size (pixels) for a pyramid level.
/// Levels whose short side would fall below this are not generated.
const PYRAMID_MIN_SIZE: usize = 32;

/// A single level of a luma pyramid: raw pixel data plus dimensions.
pub struct PyramidLevel {
    pub luma: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Downsample a luma plane by a factor of 2 using a separable 3-tap box
/// (1/4, 1/2, 1/4) blur + stride-2 decimation.
///
/// This is equivalent to one level of a Gaussian pyramid and is entirely
/// self-contained — no extra crate dependencies.
pub fn downsample_luma(src: &[f32], src_w: usize, src_h: usize) -> (Vec<f32>, usize, usize) {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);

    // Horizontal pass: blur along X into a `src_h × dst_w` intermediate.
    let mut h_blur = vec![0.0_f32; src_h * dst_w];
    h_blur
        .par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(y, out_row)| {
            let in_row = &src[y * src_w..(y + 1) * src_w];
            for (dx, out) in out_row.iter_mut().enumerate() {
                let sx = dx * 2;
                let left = if sx > 0 { in_row[sx - 1] } else { in_row[0] };
                let center = in_row[sx];
                let right = if sx + 1 < src_w {
                    in_row[sx + 1]
                } else {
                    in_row[src_w - 1]
                };
                *out = left.mul_add(0.25, center.mul_add(0.5, right * 0.25));
            }
        });

    // Vertical pass: blur along Y into the final `dst_h × dst_w` plane.
    let mut dst = vec![0.0_f32; dst_h * dst_w];
    dst.par_chunks_mut(dst_w)
        .enumerate()
        .for_each(|(dy, out_row)| {
            let sy = dy * 2;
            let top = if sy > 0 { sy - 1 } else { 0 };
            let bot = if sy + 1 < src_h { sy + 1 } else { src_h - 1 };
            let row_top = &h_blur[top * dst_w..(top + 1) * dst_w];
            let row_mid = &h_blur[sy * dst_w..(sy + 1) * dst_w];
            let row_bot = &h_blur[bot * dst_w..(bot + 1) * dst_w];
            for x in 0..dst_w {
                out_row[x] = row_top[x].mul_add(0.25, row_mid[x].mul_add(0.5, row_bot[x] * 0.25));
            }
        });

    (dst, dst_w, dst_h)
}

/// Build a luma pyramid for `img`.
///
/// The first level is the original full-resolution luma (cloned).  Each
/// subsequent level is half the size.  Generation stops when the short side
/// would drop below [`PYRAMID_MIN_SIZE`].  For very small images (short side
/// already < `2 * PYRAMID_MIN_SIZE`) only a single level is returned so the
/// pyramid still degrades gracefully.
///
/// # Panics
///
/// Will panic if the `levels` vector is empty, which should be impossible as it is primed with the base image.
pub fn build_luma_pyramid(img: &PlanarImage<f32>) -> Vec<PyramidLevel> {
    let mut levels: Vec<PyramidLevel> = Vec::new();

    // Level 0 — full resolution.
    levels.push(PyramidLevel {
        luma: img.luma.clone(),
        width: img.width,
        height: img.height,
    });

    // Additional levels: halve until the short side hits PYRAMID_MIN_SIZE.
    loop {
        let last = levels.last().expect("levels is non-empty");
        let short = last.width.min(last.height);
        // Stop if halving would go below the minimum.
        if short / 2 < PYRAMID_MIN_SIZE {
            break;
        }
        let (luma, w, h) = downsample_luma(&last.luma, last.width, last.height);
        levels.push(PyramidLevel {
            luma,
            width: w,
            height: h,
        });
    }

    // Reverse so index 0 = coarsest, last = finest.
    levels.reverse();
    levels
}

/// Repeatedly halve a luma plane (via [`downsample_luma`]) until its short
/// side is at most `max_short_side` pixels.
///
/// Used by the `pipeline::align_frame` post-refinement sanity gate to get a
/// cheap small-resolution copy of both the reference and source luma planes
/// before evaluating the registration RMS objective on them.  Returns the
/// (possibly unchanged) luma plane plus its final width/height.  Stops
/// early if the image is already small enough, and also stops if a further
/// halving would produce a zero-sized dimension.
pub fn downsample_luma_to_max_side(
    luma: &[f32],
    width: usize,
    height: usize,
    max_short_side: usize,
) -> (Vec<f32>, usize, usize) {
    let mut cur = luma.to_vec();
    let mut w = width;
    let mut h = height;
    while w.min(h) > max_short_side && w > 1 && h > 1 {
        let (next, nw, nh) = downsample_luma(&cur, w, h);
        cur = next;
        w = nw;
        h = nh;
    }
    (cur, w, h)
}
