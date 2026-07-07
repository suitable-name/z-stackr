use crate::apex::pyramid::apply_gaussian_blur;
use rayon::prelude::*;
use stacker_core::image::PlanarImage;

const BLUR_PASSES: usize = 5;

/// The per-pixel 4-neighbour Laplacian-magnitude pass:
/// `h[y,x] = |up + down + left + right - 4*center|`, clamped boundary.
///
/// Pure-CPU/rayon reference implementation. See
/// [`compute_saliency`]'s docs for the GPU dispatch that tries to run this
/// exact computation on the GPU first (when the `gpu` feature is compiled
/// in and the runtime switch is on).
fn laplacian_magnitude_cpu(luma: &[f32], width: usize, height: usize) -> Vec<f32> {
    let len = width * height;
    let mut h = vec![0.0_f32; len];
    h.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let up_row = y.saturating_sub(1);
        let down_row = (y + 1).min(height.saturating_sub(1));
        for (x, out) in row.iter_mut().enumerate() {
            let left_col = x.saturating_sub(1);
            let right_col = (x + 1).min(width.saturating_sub(1));
            let center = luma[y * width + x];
            let up = luma[up_row * width + x];
            let down = luma[down_row * width + x];
            let left = luma[y * width + left_col];
            let right = luma[y * width + right_col];
            *out = (up + down + left + right - 4.0 * center).abs();
        }
    });
    h
}

/// Strata's saliency metric.
///
/// A per-pixel Laplacian-magnitude response, smoothed by [`BLUR_PASSES`]
/// applications of the existing 5-tap separable blur (see `strata::mod`'s
/// `BLUR_PASSES` doc comment for the sigma derivation).
///
/// # GPU dispatch
/// When compiled with the `gpu` feature, the Laplacian-magnitude pass tries
/// a `wgpu` compute-shader port first ([`crate::strata::gpu::laplacian_magnitude_gpu`])
/// and falls back to the CPU/rayon implementation on any failure (no
/// adapter, the runtime switch off, or any `wgpu` call erroring) — never a
/// panic. The five blur passes that follow always run on the CPU
/// regardless — see `strata::gpu`'s module docs for the scope rationale.
pub fn compute_saliency(luma: &[f32], width: usize, height: usize) -> Vec<f32> {
    let len = width * height;

    #[cfg(feature = "gpu")]
    let h = crate::strata::gpu::laplacian_magnitude_gpu(luma, width, height)
        .unwrap_or_else(|| laplacian_magnitude_cpu(luma, width, height));
    #[cfg(not(feature = "gpu"))]
    let h = laplacian_magnitude_cpu(luma, width, height);

    let mut img = PlanarImage {
        width,
        height,
        luma: h,
        chroma_a: vec![0.0; len],
        chroma_b: vec![0.0; len],
    };
    for _ in 0..BLUR_PASSES {
        img = apply_gaussian_blur(&img);
    }
    img.luma
}

pub fn compute_argmax(
    frames: &[PlanarImage<f32>],
    width: usize,
    height: usize,
) -> (Vec<u16>, Vec<f32>) {
    let len = width * height;
    let mut argmax_idx = vec![0_u16; len];
    let mut argmax_val = vec![f32::NEG_INFINITY; len];
    for (i, frame) in frames.iter().enumerate() {
        let s = compute_saliency(&frame.luma, width, height);
        let idx = i as u16;
        for p in 0..len {
            if s[p] > argmax_val[p] {
                argmax_val[p] = s[p];
                argmax_idx[p] = idx;
            }
        }
    }
    (argmax_idx, argmax_val)
}
