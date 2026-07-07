#![allow(clippy::similar_names)]

use nalgebra::Matrix3;
use rayon::prelude::*;
use stacker_core::{error::StackerError, image::PlanarImage};

fn validate_matrix(matrix: &Matrix3<f32>) -> Result<(), StackerError> {
    if !matrix.iter().all(|&v| v.is_finite()) {
        return Err(StackerError::MathError("non-finite matrix element".into()));
    }
    Ok(())
}

// ── Faithful 4-tap spline warp (edge-clamped border) ──────────────────────────
//
// Mirrors the reference engine's default interpolation: a 4-tap spline kernel
// with edge-clamped, weight-normalised border handling. Edge clamping keeps
// the warped border from biasing the registration objective and avoids a
// hard black seam in the output for the registration alignment mode.

/// 4-tap spline interpolation weights for fractional offset `t` in `[0, 1)`.
/// The four weights are a partition of unity (they sum to 1).
#[inline]
fn spline4(t: f32) -> [f32; 4] {
    [
        (-0.333_333_34_f32)
            .mul_add(t, 0.8)
            .mul_add(t, -0.466_666_67)
            * t,
        (t - 1.8).mul_add(t, -0.2).mul_add(t, 1.0),
        (1.2 - t).mul_add(t, 0.8) * t,
        0.333_333_34_f32.mul_add(t, -0.2).mul_add(t, -0.133_333_34) * t,
    ]
}

/// Sample `plane` at `(x, y)` with the 4-tap spline kernel and edge-clamped borders.
///
/// Out-of-range taps are clamped to the nearest valid pixel and the result is
/// renormalised by the weight sum, so no zero-fill border is introduced.
/// Returns `0.0` for non-finite coordinates.
pub fn spline4x4_sample_clamped(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if !x.is_finite() || !y.is_finite() {
        return 0.0;
    }
    let idx_x = x.floor() as isize;
    let idx_y = y.floor() as isize;
    let wx = spline4(x - x.floor());
    let wy = spline4(y - y.floor());
    let iw = width as isize;
    let ih = height as isize;

    // Fast path: all four taps in range (centre taps at idx-1..=idx+2).
    if idx_x >= 1 && idx_x <= iw - 3 && idx_y >= 1 && idx_y <= ih - 3 {
        let mut val = 0.0_f32;
        for (ky, &wyk) in wy.iter().enumerate() {
            let row = (idx_y - 1 + ky as isize) as usize * width;
            let mut row_sum = 0.0_f32;
            for (kx, &wxk) in wx.iter().enumerate() {
                let px = (idx_x - 1 + kx as isize) as usize;
                row_sum = wxk.mul_add(plane[row + px], row_sum);
            }
            val = wyk.mul_add(row_sum, val);
        }
        return val;
    }

    // Border path: clamp tap indices to the edge and renormalise by weight sum.
    let mut sum_w = 0.0_f32;
    let mut acc = 0.0_f32;
    for (ky, &wyk) in wy.iter().enumerate() {
        let cy = (idx_y - 1 + ky as isize).clamp(0, ih - 1) as usize;
        for (kx, &wxk) in wx.iter().enumerate() {
            let cx = (idx_x - 1 + kx as isize).clamp(0, iw - 1) as usize;
            let w = wxk * wyk;
            sum_w += w;
            acc = w.mul_add(plane[cy * width + cx], acc);
        }
    }
    if sum_w.abs() < 1.0e-12 {
        0.0
    } else {
        acc / sum_w
    }
}

/// Warp `src` by the forward mapping `matrix` (src → dst) using 4-tap spline
/// interpolation with edge-clamped borders — the production path.
///
/// All three planes are warped independently.
///
/// # GPU dispatch
/// When compiled with the `gpu` feature, this internally tries a `wgpu`
/// compute-shader port of the identical kernel first
/// ([`super::gpu::warp_image_clamped_gpu`]) and falls back to
/// [`warp_image_clamped_cpu`] whenever no GPU is available or the GPU path
/// fails for any reason — this function's public signature is unchanged
/// either way, so every production call site (alignment, pipeline, GUI)
/// benefits with zero call-site churn. GPU output is tolerance-equal (not
/// bit-equal) to the CPU path — see `super::gpu`'s module docs for the
/// tested epsilon. Without the `gpu` feature this is a thin, zero-overhead
/// wrapper around [`warp_image_clamped_cpu`].
///
/// # Errors
/// Returns [`StackerError::MathError`] when `matrix` is non-finite or
/// non-invertible.
pub fn warp_image_clamped(
    src: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> Result<PlanarImage<f32>, StackerError> {
    #[cfg(feature = "gpu")]
    {
        match super::gpu::warp_image_clamped_gpu(src, matrix) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {
                tracing::debug!("warp_image_clamped: no GPU context available, using CPU path");
            }
            Err(err) => return Err(err),
        }
    }
    warp_image_clamped_cpu(src, matrix)
}

/// Pure-CPU reference implementation of [`warp_image_clamped`]'s kernel.
///
/// Kept as a separately callable function (rather than inlined into
/// [`warp_image_clamped`]) so the GPU-vs-CPU parity tests
/// (`tests/gpu_warp_parity.rs`) can call it directly regardless of whether
/// the `gpu` feature is enabled, and so [`warp_image_clamped`]'s
/// try-GPU-else-CPU dispatch has a concrete CPU target to fall back to.
///
/// # Errors
/// Returns [`StackerError::MathError`] when `matrix` is non-finite or
/// non-invertible.
pub fn warp_image_clamped_cpu(
    src: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> Result<PlanarImage<f32>, StackerError> {
    validate_matrix(matrix)?;
    let m_inv = matrix
        .try_inverse()
        .ok_or_else(|| StackerError::MathError("non-invertible warp matrix".into()))?;

    let width = src.width;
    let height = src.height;
    let mut dst = PlanarImage::new(width, height);

    let luma_src = &src.luma;
    let ca_src = &src.chroma_a;
    let cb_src = &src.chroma_b;

    dst.luma
        .par_chunks_mut(width)
        .zip(dst.chroma_a.par_chunks_mut(width))
        .zip(dst.chroma_b.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, ((row_l, row_a), row_b))| {
            let m00 = m_inv[(0, 0)];
            let m01 = m_inv[(0, 1)];
            let m02 = m_inv[(0, 2)];
            let m10 = m_inv[(1, 0)];
            let m11 = m_inv[(1, 1)];
            let m12 = m_inv[(1, 2)];
            let m20 = m_inv[(2, 0)];
            let m21 = m_inv[(2, 1)];
            let m22 = m_inv[(2, 2)];
            let yf = y as f32;
            let c0 = m01.mul_add(yf, m02);
            let c1 = m11.mul_add(yf, m12);
            let c2 = m21.mul_add(yf, m22);

            for x in 0..width {
                let xf = x as f32;
                let inv = 1.0 / m20.mul_add(xf, c2);
                let src_x = m00.mul_add(xf, c0) * inv;
                let src_y = m10.mul_add(xf, c1) * inv;
                row_l[x] = spline4x4_sample_clamped(luma_src, width, height, src_x, src_y);
                row_a[x] = spline4x4_sample_clamped(ca_src, width, height, src_x, src_y);
                row_b[x] = spline4x4_sample_clamped(cb_src, width, height, src_x, src_y);
            }
        });

    Ok(dst)
}
