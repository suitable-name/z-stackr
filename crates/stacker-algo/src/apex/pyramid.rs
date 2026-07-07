use rayon::prelude::*;
use stacker_core::image::PlanarImage;
use std::simd::prelude::*;

/// SIMD width for the separable blur: 16 × f32 = 512 bits (one AVX-512 zmm on
/// zen4, two AVX2 ymm on zen3, scalar elsewhere).
const BLUR_LANES: usize = 16;
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
    k as usize
}

#[derive(Clone)]
pub struct LaplacianPyramid {
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

/// Separable 5-tap Gaussian blur using the pyramid kernel.
///
/// The horizontal pass is parallelised across rows; the vertical pass is
/// parallelised across rows of the output (each output row only reads a
/// clamped-window of `temp` rows, so no data races).
///
/// Boundary handling: reflect (`nx = −nx`, `nx = 2W−2−nx`) with an extra
/// clamp to index 0 for very narrow images (`width < 3`).
// Cohesive separable-blur kernel (horizontal + vertical SIMD passes) kept in one
// function to preserve the documented bit-for-bit accumulation order.
#[allow(clippy::too_many_lines)] // splitting would break the documented single-pass accumulation order
fn blur_channel(width: usize, height: usize, data: &[f32]) -> Vec<f32> {
    if width == 0 || height == 0 {
        return vec![];
    }
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
    let mut temp = vec![0.0_f32; width * height];
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
    let mut out = vec![0.0_f32; width * height];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            let yi = y as isize;
            if yi >= 2 && yi < h - 2 {
                // Interior row: the five source rows are contiguous, no reflection.
                let base = (y - 2) * width;
                let row_minus_2 = &temp[base..base + width];
                let row_minus_1 = &temp[base + width..base + 2 * width];
                let row_center = &temp[base + 2 * width..base + 3 * width];
                let row_plus_1 = &temp[base + 3 * width..base + 4 * width];
                let row_plus_2 = &temp[base + 4 * width..base + 5 * width];

                let mut x = 0usize;
                while x + BLUR_LANES <= width {
                    let mut acc = sk0 * BlurSimd::from_slice(&row_minus_2[x..]);
                    acc += sk1 * BlurSimd::from_slice(&row_minus_1[x..]);
                    acc += sk2 * BlurSimd::from_slice(&row_center[x..]);
                    acc += sk3 * BlurSimd::from_slice(&row_plus_1[x..]);
                    acc += sk4 * BlurSimd::from_slice(&row_plus_2[x..]);
                    acc.copy_to_slice(&mut out_row[x..x + BLUR_LANES]);
                    x += BLUR_LANES;
                }
                for x in x..width {
                    let mut s = kernel[0] * row_minus_2[x];
                    s += kernel[1] * row_minus_1[x];
                    s += kernel[2] * row_center[x];
                    s += kernel[3] * row_plus_1[x];
                    s += kernel[4] * row_plus_2[x];
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
    out
}

pub fn apply_gaussian_blur(img: &PlanarImage<f32>) -> PlanarImage<f32> {
    PlanarImage {
        width: img.width,
        height: img.height,
        luma: blur_channel(img.width, img.height, &img.luma),
        chroma_a: blur_channel(img.width, img.height, &img.chroma_a),
        chroma_b: blur_channel(img.width, img.height, &img.chroma_b),
    }
}

pub fn downsample(img: &PlanarImage<f32>) -> PlanarImage<f32> {
    let blurred = apply_gaussian_blur(img);
    let new_w = (img.width + 1) / 2;
    let new_h = (img.height + 1) / 2;

    let mut out_luma = vec![0.0_f32; new_w * new_h];
    let mut out_a = vec![0.0_f32; new_w * new_h];
    let mut out_b = vec![0.0_f32; new_w * new_h];

    // Parallelise over output rows.
    out_luma
        .par_chunks_mut(new_w)
        .zip(out_a.par_chunks_mut(new_w))
        .zip(out_b.par_chunks_mut(new_w))
        .enumerate()
        .for_each(|(y, ((rl, ra), rb))| {
            let old_y = y * 2;
            for x in 0..new_w {
                let old_x = x * 2;
                let old_idx = old_y * img.width + old_x;
                let new_idx = x; // offset within this row chunk
                rl[new_idx] = blurred.luma[old_idx];
                ra[new_idx] = blurred.chroma_a[old_idx];
                rb[new_idx] = blurred.chroma_b[old_idx];
            }
        });

    PlanarImage {
        width: new_w,
        height: new_h,
        luma: out_luma,
        chroma_a: out_a,
        chroma_b: out_b,
    }
}

pub fn expand(img: &PlanarImage<f32>, w: usize, h: usize) -> PlanarImage<f32> {
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

    let expanded = PlanarImage {
        width: w,
        height: h,
        luma: exp_luma,
        chroma_a: exp_a,
        chroma_b: exp_b,
    };
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
    pub fn build(image: &PlanarImage<f32>, max_levels: usize) -> Self {
        if max_levels <= 1 {
            return Self {
                levels: vec![PlanarImage {
                    width: image.width,
                    height: image.height,
                    luma: image.luma.clone(),
                    chroma_a: image.chroma_a.clone(),
                    chroma_b: image.chroma_b.clone(),
                }],
            };
        }
        let mut current = PlanarImage {
            width: image.width,
            height: image.height,
            luma: image.luma.clone(),
            chroma_a: image.chroma_a.clone(),
            chroma_b: image.chroma_b.clone(),
        };
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

            levels.push(PlanarImage {
                width: current.width,
                height: current.height,
                luma,
                chroma_a: a,
                chroma_b: b,
            });
            current = next;
            if current.width <= 2 || current.height <= 2 {
                break;
            }
        }
        levels.push(current);
        Self { levels }
    }

    pub fn reconstruct(&self) -> PlanarImage<f32> {
        if self.levels.is_empty() {
            return PlanarImage {
                width: 0,
                height: 0,
                luma: vec![],
                chroma_a: vec![],
                chroma_b: vec![],
            };
        }
        let last = self.levels.last().unwrap();
        let mut current = PlanarImage {
            width: last.width,
            height: last.height,
            luma: last.luma.clone(),
            chroma_a: last.chroma_a.clone(),
            chroma_b: last.chroma_b.clone(),
        };
        for i in (0..self.levels.len() - 1).rev() {
            let level = &self.levels[i];
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

            current = PlanarImage {
                width: level.width,
                height: level.height,
                luma,
                chroma_a: a,
                chroma_b: b,
            };
        }
        current
    }
}
