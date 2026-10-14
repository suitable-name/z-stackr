//! Laplacian pyramids for the `Apex` fusion.
//!
//! The pieces are the 5-tap pyramid blur ([`apply_gaussian_blur`]), the
//! decimating [`downsample`], the polyphase [`expand`], and [`LaplacianPyramid`]
//! with its build and reconstruct.
//!
//! Every filter is plain multiply and add in kernel order (no fused
//! multiply-add), written once as a scalar expression and once as a `std::simd`
//! body that performs the same operations in the same order, so a pixel's value
//! does not depend on whether the vector body or the scalar tail computed it,
//! nor on the vector width the build selected (`BLUR_LANES`).

#![expect(
    clippy::suboptimal_flops,
    reason = "the blur is specified as plain multiply and add in kernel order (no FMA) so the SIMD and scalar paths stay bit-identical"
)]
#![expect(
    clippy::cast_possible_wrap,
    reason = "image dimensions and pixel indices never exceed isize::MAX; the signed values only feed the reflect-index arithmetic"
)]

use rayon::prelude::*;
use stacker_core::image::PlanarImage;
use std::simd::prelude::*;

/// SIMD width (in `f32` lanes) of the separable blur, selected from the
/// compile-time target features: 16 (one `zmm`) with `avx512f`, 8 (one `ymm`)
/// with `avx2`, otherwise 4 (one `xmm`, the SSE2 baseline and NEON).
///
/// `std::simd` has no runtime dispatch: the width is fixed when the crate is
/// compiled and follows `-C target-cpu` / `-C target-feature` (see the root
/// README's "CPU features"). The blur uses plain multiply and add, so it needs
/// no `fma` feature, and every operation is elementwise, so the width changes
/// speed only: widths 4, 8 and 16 produce identical pixels and the scalar tails
/// cover whatever does not fill a vector.
const BLUR_LANES: usize = if cfg!(target_feature = "avx512f") {
    16
} else if cfg!(target_feature = "avx2") {
    8
} else {
    4
};
type BlurSimd = Simd<f32, BLUR_LANES>;

/// Reflecting boundary index used by the blur kernel: mirrors out-of-range
/// coordinates (`-i`, `2n-2-i`) and clamps to 0 for `n < 3` (reflecting
/// boundary handling).
#[inline]
const fn reflect_index(i: isize, n: isize) -> usize {
    let mut k = i;
    if k < 0 {
        k = -k;
    } else if k >= n {
        k = 2 * n - 2 - k;
    }
    if k < 0 || k >= n {
        k = 0;
    }
    k.cast_unsigned()
}

/// A Laplacian pyramid of one image.
///
/// `levels[0]` is the finest Laplacian band (the image minus its blurred,
/// re-expanded half-resolution copy), each following level is half the size of
/// the one before, and the last level is the low-pass base (not a difference).
/// [`Self::reconstruct`] inverts [`Self::build`] up to `f32` rounding.
#[derive(Clone)]
pub struct LaplacianPyramid {
    /// The bands from finest to coarsest, then the base; see the type docs.
    pub levels: Vec<PlanarImage<f32>>,
}

/// 5-tap separable kernel: `[weight2, weight1, reductionFactor, weight1, weight2]`
/// where `reductionFactor = 0.33`, `weight1 = 0.25`, `weight2 = 0.25 − 0.33/2 = 0.085`.
/// The vector sums to 1.0 (already normalised).  This differs from the standard
/// Burt–Adelson kernel `[1,4,6,4,1]/16` (centre tap 0.375 vs 0.33).
const KERNEL_REDUCTION_FACTOR: f32 = 0.33;
const KERNEL_WEIGHT1: f32 = 0.25;
const KERNEL_WEIGHT2: f32 = 0.25 - KERNEL_REDUCTION_FACTOR / 2.0; // 0.085
/// Separable 5-tap kernel: `[KERNEL_WEIGHT2, KERNEL_WEIGHT1, KERNEL_REDUCTION_FACTOR, KERNEL_WEIGHT1, KERNEL_WEIGHT2]`.
const KERNEL: [f32; 5] = [
    KERNEL_WEIGHT2,
    KERNEL_WEIGHT1,
    KERNEL_REDUCTION_FACTOR,
    KERNEL_WEIGHT1,
    KERNEL_WEIGHT2,
];

/// Separable 5-tap Gaussian blur of one plane with the pyramid kernel.
///
/// This is the per-plane core of [`apply_gaussian_blur`]: blurring a plane
/// here gives exactly the bits [`apply_gaussian_blur`] gives that plane inside
/// an image. An empty plane (`width` or `height` zero) returns an empty vector.
/// To blur repeatedly without allocating, use [`blur_plane_into`].
///
/// # Panics
/// Panics when `data` holds fewer than `width * height` samples.
#[must_use]
pub fn blur_plane(width: usize, height: usize, data: &[f32]) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![];
    }
    let mut temp = vec![0.0_f32; width * height];
    let mut out = vec![0.0_f32; width * height];
    blur_plane_into(width, height, data, &mut temp, &mut out);
    out
}

/// [`blur_plane`] into caller-supplied buffers.
///
/// `temp` holds the horizontal pass and `out` receives the result; both must
/// have exactly `width * height` samples and may hold anything on entry (every
/// sample is overwritten). `data` must not alias them. Nothing is allocated, so
/// a caller that blurs many times (Strata's saliency smoothing) reuses three
/// planes instead of allocating two per pass.
///
/// The horizontal pass is parallelised across rows; the vertical pass is
/// parallelised across rows of the output (each output row only reads a
/// clamped-window of `temp` rows, so no data races).
///
/// Boundary handling: reflect (`nx = −nx`, `nx = 2W−2−nx`) with an extra
/// clamp to index 0 for very narrow images (`width < 3`).
///
/// # Panics
/// Panics when `data`, `temp` or `out` does not hold exactly `width * height`
/// samples.
// Cohesive separable-blur kernel (horizontal + vertical SIMD passes) kept in one
// function to preserve the documented bit-for-bit accumulation order.
#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the vm2/vm1/vp1/vp2 taps are named by offset; the two passes stay in one function to keep the documented accumulation order"
)]
pub fn blur_plane_into(
    width: usize,
    height: usize,
    data: &[f32],
    temp: &mut [f32],
    out: &mut [f32],
) {
    if width == 0 || height == 0 {
        return;
    }
    let len = width * height;
    assert!(
        data.len() >= len,
        "input plane is shorter than width * height"
    );
    assert_eq!(
        temp.len(),
        len,
        "temp plane must hold width * height samples"
    );
    assert_eq!(
        out.len(),
        len,
        "output plane must hold width * height samples"
    );
    let kernel = KERNEL;
    // Splatted kernel taps for the SIMD interior.
    let (sk0, sk1, sk2, sk3, sk4) = (
        BlurSimd::splat(kernel[0]),
        BlurSimd::splat(kernel[1]),
        BlurSimd::splat(kernel[2]),
        BlurSimd::splat(kernel[3]),
        BlurSimd::splat(kernel[4]),
    );
    let w = width as isize;
    let h = height as isize;

    // ── Horizontal pass (parallelised over rows) ──────────────────────────
    temp.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            let in_row = &data[y * width..y * width + width];

            // Scalar pixel with full reflect handling — identical to the original
            // accumulation order so boundary results are bit-for-bit unchanged.
            let scalar = |x: usize| -> f32 {
                let xi = x as isize;
                let mut s = kernel[0] * in_row[reflect_index(xi - 2, w)];
                s += kernel[1] * in_row[reflect_index(xi - 1, w)];
                s += kernel[2] * in_row[reflect_index(xi, w)];
                s += kernel[3] * in_row[reflect_index(xi + 1, w)];
                s += kernel[4] * in_row[reflect_index(xi + 2, w)];
                s
            };

            let lo = 2usize.min(width);
            let hi = width.saturating_sub(2); // interior is [2, hi): no reflection.

            // Left boundary.
            for (x, cell) in out_row.iter_mut().take(lo).enumerate() {
                *cell = scalar(x);
            }
            // Interior: SIMD body then scalar remainder.
            let mut x = lo.max(2);
            while x + BLUR_LANES <= hi {
                // x-2 >= 0 and x+2+LANES <= width, so all five loads stay in bounds.
                let vm2 = BlurSimd::from_slice(&in_row[x - 2..]);
                let vm1 = BlurSimd::from_slice(&in_row[x - 1..]);
                let v0 = BlurSimd::from_slice(&in_row[x..]);
                let vp1 = BlurSimd::from_slice(&in_row[x + 1..]);
                let vp2 = BlurSimd::from_slice(&in_row[x + 2..]);
                // Plain mul + add in the same order as `scalar` (no fused
                // multiply-add) → bit-identical to the scalar path, per lane.
                let mut acc = sk0 * vm2;
                acc += sk1 * vm1;
                acc += sk2 * v0;
                acc += sk3 * vp1;
                acc += sk4 * vp2;
                acc.copy_to_slice(&mut out_row[x..x + BLUR_LANES]);
                x += BLUR_LANES;
            }
            while x < hi {
                out_row[x] = scalar(x);
                x += 1;
            }
            // Right boundary.
            for (x, cell) in out_row.iter_mut().enumerate().skip(hi.max(lo)) {
                *cell = scalar(x);
            }
        });

    // ── Vertical pass (parallelised over output rows) ─────────────────────
    let temp: &[f32] = temp;
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            let yi = y as isize;
            if yi >= 2 && yi < h - 2 {
                // Interior row: the five source rows are contiguous, no reflection.
                let base = (y - 2) * width;
                let r_m2 = &temp[base..base + width];
                let r_m1 = &temp[base + width..base + 2 * width];
                let r_0 = &temp[base + 2 * width..base + 3 * width];
                let r_p1 = &temp[base + 3 * width..base + 4 * width];
                let r_p2 = &temp[base + 4 * width..base + 5 * width];

                let mut x = 0usize;
                while x + BLUR_LANES <= width {
                    let mut acc = sk0 * BlurSimd::from_slice(&r_m2[x..]);
                    acc += sk1 * BlurSimd::from_slice(&r_m1[x..]);
                    acc += sk2 * BlurSimd::from_slice(&r_0[x..]);
                    acc += sk3 * BlurSimd::from_slice(&r_p1[x..]);
                    acc += sk4 * BlurSimd::from_slice(&r_p2[x..]);
                    acc.copy_to_slice(&mut out_row[x..x + BLUR_LANES]);
                    x += BLUR_LANES;
                }
                for x in x..width {
                    let mut s = kernel[0] * r_m2[x];
                    s += kernel[1] * r_m1[x];
                    s += kernel[2] * r_0[x];
                    s += kernel[3] * r_p1[x];
                    s += kernel[4] * r_p2[x];
                    out_row[x] = s;
                }
            } else {
                // Boundary row: reflect the five source-row indices in y.
                let ny0 = reflect_index(yi - 2, h) * width;
                let ny1 = reflect_index(yi - 1, h) * width;
                let ny2 = reflect_index(yi, h) * width;
                let ny3 = reflect_index(yi + 1, h) * width;
                let ny4 = reflect_index(yi + 2, h) * width;
                for (x, cell) in out_row.iter_mut().enumerate() {
                    let mut s = kernel[0] * temp[ny0 + x];
                    s += kernel[1] * temp[ny1 + x];
                    s += kernel[2] * temp[ny2 + x];
                    s += kernel[3] * temp[ny3 + x];
                    s += kernel[4] * temp[ny4 + x];
                    *cell = s;
                }
            }
        });
}

/// Gaussian-blur every plane of `img`.
///
/// # Panics
/// Panics when a plane does not hold `width * height` samples (a
/// [`PlanarImage`] built through its constructors always does).
#[must_use]
pub fn apply_gaussian_blur(img: &PlanarImage<f32>) -> PlanarImage<f32> {
    PlanarImage::from_planes(
        img.width,
        img.height,
        blur_plane(img.width, img.height, &img.luma),
        blur_plane(img.width, img.height, &img.chroma_a),
        blur_plane(img.width, img.height, &img.chroma_b),
    )
    .expect("a blurred plane has one sample per pixel")
}

/// Combine up to five equally long rows with their kernel weights:
/// `out[x] = taps[0].0·taps[0].1[x] + taps[1].0·taps[1].1[x] + …` (then
/// `· post` when given), accumulated left to right with plain multiply and add
/// (no fused multiply-add), exactly like the blur kernel's vertical pass.
/// An empty tap list writes zeros.
fn combine_rows(out: &mut [f32], taps: &[(f32, &[f32])], post: Option<f32>) {
    let Some(((k0, r0), rest)) = taps.split_first() else {
        out.fill(0.0);
        return;
    };
    let width = out.len();
    let post_v = post.map(BlurSimd::splat);
    let mut x = 0usize;
    while x + BLUR_LANES <= width {
        let mut acc = BlurSimd::splat(*k0) * BlurSimd::from_slice(&r0[x..]);
        for (k, r) in rest {
            acc += BlurSimd::splat(*k) * BlurSimd::from_slice(&r[x..]);
        }
        if let Some(p) = post_v {
            acc *= p;
        }
        acc.copy_to_slice(&mut out[x..x + BLUR_LANES]);
        x += BLUR_LANES;
    }
    for x in x..width {
        let mut s = *k0 * r0[x];
        for (k, r) in rest {
            s += *k * r[x];
        }
        if let Some(p) = post {
            s *= p;
        }
        out[x] = s;
    }
}

/// Horizontal 5-tap blur evaluated only at the even columns `0, 2, 4, …` of
/// `in_row`; `out_row[i]` is the blur at column `2i`.  Same taps, same order and
/// same reflection as the horizontal pass of [`blur_plane_into`].
fn decimate_row(in_row: &[f32], out_row: &mut [f32]) {
    let w = in_row.len();
    let wi = w as isize;
    for (dx, out) in out_row.iter_mut().enumerate() {
        let sx = dx * 2;
        *out = if sx >= 2 && sx + 2 < w {
            // Interior: no reflection.
            let win = &in_row[sx - 2..=sx + 2];
            let mut s = KERNEL[0] * win[0];
            s += KERNEL[1] * win[1];
            s += KERNEL[2] * win[2];
            s += KERNEL[3] * win[3];
            s += KERNEL[4] * win[4];
            s
        } else {
            let xi = sx as isize;
            let mut s = KERNEL[0] * in_row[reflect_index(xi - 2, wi)];
            s += KERNEL[1] * in_row[reflect_index(xi - 1, wi)];
            s += KERNEL[2] * in_row[reflect_index(xi, wi)];
            s += KERNEL[3] * in_row[reflect_index(xi + 1, wi)];
            s += KERNEL[4] * in_row[reflect_index(xi + 2, wi)];
            s
        };
    }
}

/// Blur one plane and keep every second pixel in both directions, computing
/// only the pixels that are kept.
///
/// Bit-identical to [`blur_plane`] followed by taking pixels `(2x, 2y)`: each
/// kept pixel is the same 5 + 5 taps summed in the same order.  Only the even
/// columns are blurred horizontally (half the rows' work) and only the even rows
/// vertically (a quarter of the plane), and the intermediate plane has
/// `new_w` instead of `width` columns.
fn downsample_channel(width: usize, height: usize, data: &[f32]) -> Vec<f32> {
    let new_w = width.div_ceil(2);
    let new_h = height.div_ceil(2);
    if new_w == 0 || new_h == 0 {
        return vec![];
    }
    let h = height as isize;

    // Horizontal pass: all rows (the vertical taps reach odd rows), even columns.
    let mut temp = vec![0.0_f32; height * new_w];
    temp.par_chunks_mut(new_w)
        .enumerate()
        .for_each(|(y, out_row)| decimate_row(&data[y * width..(y + 1) * width], out_row));

    // Vertical pass: even rows only.
    let mut out = vec![0.0_f32; new_h * new_w];
    out.par_chunks_mut(new_w)
        .enumerate()
        .for_each(|(dy, out_row)| {
            let yi = (dy * 2) as isize;
            let taps: [(f32, &[f32]); 5] = std::array::from_fn(|t| {
                let r = reflect_index(yi + t as isize - 2, h);
                (KERNEL[t], &temp[r * new_w..(r + 1) * new_w])
            });
            combine_rows(out_row, &taps, None);
        });
    out
}

/// Gaussian-blur `img` and keep every second pixel in both directions.
///
/// The output is bit-identical to blurring the whole image with
/// [`apply_gaussian_blur`] and copying pixels `(2x, 2y)`, but only the kept
/// pixels are computed (see `downsample_channel`).
///
/// # Panics
/// Panics when a plane does not hold `width * height` samples (a
/// [`PlanarImage`] built through its constructors always does).
#[must_use]
pub fn downsample(img: &PlanarImage<f32>) -> PlanarImage<f32> {
    PlanarImage::from_planes(
        img.width.div_ceil(2),
        img.height.div_ceil(2),
        downsample_channel(img.width, img.height, &img.luma),
        downsample_channel(img.width, img.height, &img.chroma_a),
        downsample_channel(img.width, img.height, &img.chroma_b),
    )
    .expect("a downsampled plane has one sample per pixel")
}

/// `true` iff `expand_channel` can be used for an `sw × sh` image on a
/// `w × h` canvas: the small image must cover every even canvas position.
const fn expand_covers_canvas(sw: usize, sh: usize, w: usize, h: usize) -> bool {
    sw >= w.div_ceil(2) && sh >= h.div_ceil(2)
}

/// Horizontal pass of the polyphase expand for one source row.
///
/// Equals the 5-tap blur of the zero-inserted row (`src[i]` at even positions,
/// zeros between), skipping the taps that fall on the zeros: an even output sees
/// three non-zero taps, an odd output two.  Terms are added in the kernel's
/// order, so the result matches the blur of the zero-inserted row exactly (up to
/// the sign of an exact zero).
fn expand_row_h(src: &[f32], out: &mut [f32]) {
    let w = out.len();
    let wi = w as isize;
    // Boundary pixel: generic, with reflection.
    let boundary = |x: usize| -> f32 {
        let xi = x as isize;
        let mut acc: Option<f32> = None;
        for (t, &k) in KERNEL.iter().enumerate() {
            let c = reflect_index(xi + t as isize - 2, wi);
            if c.is_multiple_of(2) {
                let term = k * src[c / 2];
                acc = Some(acc.map_or(term, |s| s + term));
            }
        }
        acc.unwrap_or(0.0)
    };

    let lo = 2usize.min(w);
    let hi = w.saturating_sub(2).max(lo); // interior is [lo, hi): no reflection.
    for (x, cell) in out.iter_mut().enumerate() {
        *cell = if x < lo || x >= hi {
            boundary(x)
        } else if x.is_multiple_of(2) {
            let m = x / 2;
            let mut s = KERNEL[0] * src[m - 1];
            s += KERNEL[2] * src[m];
            s += KERNEL[4] * src[m + 1];
            s
        } else {
            let m = (x - 1) / 2;
            let mut s = KERNEL[1] * src[m];
            s += KERNEL[3] * src[m + 1];
            s
        };
    }
}

/// Placeholder for unused tap slots in `expand_channel`.
const NO_ROW: &[f32] = &[];

/// Expand one plane (`src`, rows of `src_width`) to a `width × height` canvas:
/// zero-insert, 5-tap blur, times 4 — without materialising the zero-inserted
/// canvas.
///
/// Requires [`expand_covers_canvas`].  The result equals the blur of the
/// zero-inserted canvas times 4, bit for bit (up to the sign of an exact zero):
/// only the `ceil(height/2)` even canvas rows are non-zero, so the horizontal
/// pass runs on those alone and on their non-zero taps, and the vertical pass
/// sums only the (two or three) non-zero rows per output row and applies the
/// `· 4` in the same pass.
fn expand_channel(src: &[f32], src_width: usize, width: usize, height: usize) -> Vec<f32> {
    let rows = height.div_ceil(2);
    let mut temp = vec![0.0_f32; rows * width];
    temp.par_chunks_mut(width)
        .enumerate()
        .for_each(|(half, out_row)| {
            expand_row_h(&src[half * src_width..(half + 1) * src_width], out_row);
        });

    let height_i = height as isize;
    let mut out = vec![0.0_f32; width * height];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            let yi = y as isize;
            let mut taps: [(f32, &[f32]); 5] = [(0.0, NO_ROW); 5];
            let mut count = 0usize;
            for (t, &k) in KERNEL.iter().enumerate() {
                let canvas_row = reflect_index(yi + t as isize - 2, height_i);
                if canvas_row.is_multiple_of(2) {
                    let half = canvas_row / 2;
                    taps[count] = (k, &temp[half * width..(half + 1) * width]);
                    count += 1;
                }
            }
            combine_rows(out_row, &taps[..count], Some(4.0));
        });
    out
}

/// Upsample `img` to `w × h`: insert zeros between pixels, blur with the pyramid
/// kernel and scale by 4.
///
/// When `img` covers the canvas (`img.width ≥ ceil(w/2)` and
/// `img.height ≥ ceil(h/2)`, which is every call the pyramid makes) this is the
/// polyphase form: bit-identical to the zero-insert-and-blur definition, but the
/// zero canvas is never built and the zero taps are never multiplied.  Any other
/// size takes the literal definition.
///
/// # Panics
/// Panics when a plane of `img` does not hold `width * height` samples (a
/// [`PlanarImage`] built through its constructors always does).
#[must_use]
pub fn expand(img: &PlanarImage<f32>, w: usize, h: usize) -> PlanarImage<f32> {
    if w == 0 || h == 0 {
        return PlanarImage::from_planes(w, h, vec![], vec![], vec![])
            .expect("an empty canvas has no samples");
    }
    if expand_covers_canvas(img.width, img.height, w, h) {
        return PlanarImage::from_planes(
            w,
            h,
            expand_channel(&img.luma, img.width, w, h),
            expand_channel(&img.chroma_a, img.width, w, h),
            expand_channel(&img.chroma_b, img.width, w, h),
        )
        .expect("each expanded plane holds w * h samples");
    }
    expand_by_definition(img, w, h)
}

/// [`expand`] as defined: zero-insert onto a `w × h` canvas, blur, times 4.
/// Used when the source does not cover the canvas.
fn expand_by_definition(img: &PlanarImage<f32>, w: usize, h: usize) -> PlanarImage<f32> {
    let mut exp_luma = vec![0.0_f32; w * h];
    let mut exp_a = vec![0.0_f32; w * h];
    let mut exp_b = vec![0.0_f32; w * h];

    // Scatter even-row/even-col pixels from the smaller image.
    // Parallelise over the source rows.
    // We need to write into non-overlapping strided positions of the output.
    // Using index arithmetic on a flat slice avoids aliasing.
    exp_luma
        .par_chunks_mut(w)
        .zip(exp_a.par_chunks_mut(w))
        .zip(exp_b.par_chunks_mut(w))
        .enumerate()
        .for_each(|(new_y, ((rl, ra), rb))| {
            // Only even destination rows carry data (odd rows stay zero).
            if new_y % 2 != 0 {
                return;
            }
            let y = new_y / 2;
            if y >= img.height {
                return;
            }
            for x in 0..img.width {
                let new_x = x * 2;
                if new_x >= w {
                    break;
                }
                let old_idx = y * img.width + x;
                rl[new_x] = img.luma[old_idx];
                ra[new_x] = img.chroma_a[old_idx];
                rb[new_x] = img.chroma_b[old_idx];
            }
        });

    let expanded = PlanarImage::from_planes(w, h, exp_luma, exp_a, exp_b)
        .expect("each canvas plane holds w * h samples");
    let mut blurred = apply_gaussian_blur(&expanded);

    // Scale the expanded band by 4 (compensating for the zero-insertion during
    // upsampling) *in place*, avoiding a second full-size set of buffers and an
    // extra parallel pass. `blurred` already has the target `w × h` dimensions.
    blurred
        .luma
        .par_iter_mut()
        .zip(blurred.chroma_a.par_iter_mut())
        .zip(blurred.chroma_b.par_iter_mut())
        .for_each(|((l, a), b)| {
            *l *= 4.0;
            *a *= 4.0;
            *b *= 4.0;
        });

    blurred
}

impl LaplacianPyramid {
    /// Build a pyramid of at most `max_levels` levels: Laplacian bands from the
    /// finest level down, then the low-pass base as the last level.
    ///
    /// # Panics
    /// Panics when a plane of `image` does not hold `width * height` samples (a
    /// [`PlanarImage`] built through its constructors always does).
    #[must_use]
    pub fn build(image: &PlanarImage<f32>, max_levels: usize) -> Self {
        if max_levels <= 1 {
            return Self {
                levels: vec![image.clone()],
            };
        }
        let mut current = image.clone();
        let mut levels = Vec::new();
        for _ in 0..max_levels - 1 {
            let next = downsample(&current);
            let expanded = expand(&next, current.width, current.height);

            // Laplacian difference — parallelise over all pixels.
            let len = current.width * current.height;
            let mut luma = vec![0.0_f32; len];
            let mut a = vec![0.0_f32; len];
            let mut b = vec![0.0_f32; len];

            luma.par_iter_mut()
                .zip(a.par_iter_mut())
                .zip(b.par_iter_mut())
                .zip(current.luma.par_iter())
                .zip(current.chroma_a.par_iter())
                .zip(current.chroma_b.par_iter())
                .zip(expanded.luma.par_iter())
                .zip(expanded.chroma_a.par_iter())
                .zip(expanded.chroma_b.par_iter())
                .for_each(|((((((((ol, oa), ob), cl), ca), cb), el), ea), eb)| {
                    *ol = cl - el;
                    *oa = ca - ea;
                    *ob = cb - eb;
                });

            levels.push(
                PlanarImage::from_planes(current.width, current.height, luma, a, b)
                    .expect("each Laplacian plane holds width * height samples"),
            );
            current = next;
            if current.width <= 2 || current.height <= 2 {
                break;
            }
        }
        levels.push(current);
        Self { levels }
    }

    /// Collapse the pyramid back into one image: expand the base and add each
    /// Laplacian band in turn.
    ///
    /// # Panics
    /// Panics when a level does not hold `width * height` samples per plane.
    #[must_use]
    pub fn reconstruct(&self) -> PlanarImage<f32> {
        let Some((last, bands)) = self.levels.split_last() else {
            return PlanarImage::new(0, 0);
        };
        reconstruct_from_bands(bands, last.clone())
    }
}

/// Collapse a pyramid given as its Laplacian `bands` (finest first) and its
/// low-pass `base`: expand the base and add each band in turn.
///
/// [`LaplacianPyramid::reconstruct`] is this function applied to its own levels;
/// a caller that holds the bands and the base separately (the streaming `Apex`
/// fold previewing its accumulator) uses it to avoid copying the bands.
///
/// # Panics
/// Panics when a band does not hold `width * height` samples per plane.
#[must_use]
pub fn reconstruct_from_bands(
    bands: &[PlanarImage<f32>],
    base: PlanarImage<f32>,
) -> PlanarImage<f32> {
    let mut current = base;
    for level in bands.iter().rev() {
        let expanded = expand(&current, level.width, level.height);

        let len = level.width * level.height;
        let mut luma = vec![0.0_f32; len];
        let mut a = vec![0.0_f32; len];
        let mut b = vec![0.0_f32; len];

        luma.par_iter_mut()
            .zip(a.par_iter_mut())
            .zip(b.par_iter_mut())
            .zip(level.luma.par_iter())
            .zip(level.chroma_a.par_iter())
            .zip(level.chroma_b.par_iter())
            .zip(expanded.luma.par_iter())
            .zip(expanded.chroma_a.par_iter())
            .zip(expanded.chroma_b.par_iter())
            .for_each(|((((((((ol, oa), ob), ll), la), lb), el), ea), eb)| {
                *ol = ll + el;
                *oa = la + ea;
                *ob = lb + eb;
            });

        current = PlanarImage::from_planes(level.width, level.height, luma, a, b)
            .expect("each reconstructed plane holds width * height samples");
    }
    current
}
