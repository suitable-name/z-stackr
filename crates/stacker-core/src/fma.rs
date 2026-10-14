//! Fused multiply-add helpers that stay fast on `wasm32`.
//!
//! `wasm32` has no fused multiply-add instruction (the relaxed-SIMD `fma` is
//! not available in stable browsers), so [`f32::mul_add`], [`f64::mul_add`] and
//! [`std::simd::StdFloat::mul_add`] compile to a software routine per
//! scalar or per lane there, which makes the alignment hot loops run orders of
//! magnitude slower in the browser. These helpers call `mul_add` on every target
//! except `wasm32`, where they use the unfused `a * b + c`.
//!
//! Native results are bit-identical to calling `mul_add` directly. On `wasm32`
//! the product is rounded before the addition, so browser results may differ
//! from the desktop in the last bits.

#![expect(
    clippy::inline_always,
    reason = "single-instruction wrappers used in the innermost loops; they must vanish into the caller"
)]

use std::simd::Simd;

/// `a * b + c` for `f32`: fused (one rounding) everywhere except `wasm32`,
/// where it is the unfused product and sum.
#[inline(always)]
#[must_use]
pub const fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(target_arch = "wasm32")]
    {
        // wasm32 has no hardware FMA; the software fallback is far slower than mul + add.
        a * b + c
    }
}

/// `a * b + c` for `f64`: fused (one rounding) everywhere except `wasm32`,
/// where it is the unfused product and sum.
#[inline(always)]
#[must_use]
pub const fn fma_f64(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(target_arch = "wasm32")]
    {
        // wasm32 has no hardware FMA; the software fallback is far slower than mul + add.
        a * b + c
    }
}

/// Lane-wise `a * b + c` for `Simd<f32, N>`: fused everywhere except `wasm32`,
/// where it is the unfused product and sum.
#[inline(always)]
#[must_use]
pub fn fma_simd<const N: usize>(a: Simd<f32, N>, b: Simd<f32, N>, c: Simd<f32, N>) -> Simd<f32, N> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::simd::StdFloat;
        a.mul_add(b, c)
    }
    #[cfg(target_arch = "wasm32")]
    {
        a * b + c
    }
}
