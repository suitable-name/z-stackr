#![allow(
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::must_use_candidate,
    clippy::suboptimal_flops,
    clippy::excessive_precision,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

pub fn rgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    let l_ok = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    let a_ok = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    let b_ok = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    [l_ok, a_ok, b_ok]
}

// ── Lookup-table + SIMD accelerated colour conversion ─────────────────────────
//
// The sRGB transfer function is a `powf`, which does not auto-vectorise and has
// no portable-SIMD intrinsic. We precompute it into lookup tables so the hot
// paths never call `powf` at runtime, and use `std::simd` portable vectors for
// the surrounding arithmetic — they lower to AVX2 (zen3) or AVX-512 (zen4)
// depending on `-C target-cpu` / `target-feature`.

use std::{simd::prelude::*, sync::OnceLock};

/// SIMD width: 16 × f32 = 512 bits — one zmm register on AVX-512 (zen4), a pair
/// of ymm registers on AVX2 (zen3). Both lower efficiently.
const LANES: usize = 16;

/// Linear→sRGB encode-table resolution (14-bit). The encode error is well under
/// one 8-bit output level even where the sRGB curve is steepest (near black).
const ENCODE_BITS: u32 = 14;
const ENCODE_LEN: usize = 1usize << ENCODE_BITS;

static DECODE_U8: OnceLock<[f32; 256]> = OnceLock::new();
static DECODE_U16: OnceLock<Vec<f32>> = OnceLock::new();
static ENCODE_U8: OnceLock<Vec<u8>> = OnceLock::new();

/// Exact `srgb_to_linear` for every 8-bit sRGB code value (`v / 255`).
///
/// Lets the load path decode integer pixels to linear light with a table lookup
/// instead of a per-pixel `powf`.
pub fn srgb_decode_u8_table() -> &'static [f32; 256] {
    DECODE_U8.get_or_init(|| {
        let mut t = [0.0_f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = srgb_to_linear(i as f32 / 255.0);
        }
        t
    })
}

/// Exact `srgb_to_linear` for every 16-bit sRGB code value (256 KiB table).
pub fn srgb_decode_u16_table() -> &'static [f32] {
    DECODE_U16
        .get_or_init(|| {
            (0..=u16::MAX)
                .map(|i| srgb_to_linear(f32::from(i) / 65_535.0))
                .collect()
        })
        .as_slice()
}

/// `linear_to_srgb` quantised to 8-bit output, indexed by a 14-bit linear value.
///
/// Used by [`encode_linear_to_srgb_u8`] and by callers that already have a
/// linear value and a quantising index (`idx = clamp(lin, 0, 1) * (len - 1)`).
pub fn srgb_encode_u8_table() -> &'static [u8] {
    ENCODE_U8
        .get_or_init(|| {
            (0..ENCODE_LEN)
                .map(|i| {
                    let lin = i as f32 / (ENCODE_LEN - 1) as f32;
                    (linear_to_srgb(lin) * 255.0).round().clamp(0.0, 255.0) as u8
                })
                .collect()
        })
        .as_slice()
}

/// Encode contiguous **linear-light** `f32` samples to 8-bit sRGB:
/// `dst[i] = round(linear_to_srgb(clamp(src[i], 0, 1)) * 255)`.
///
/// `std::simd` performs the clamp + table-index computation `LANES` lanes at a
/// time (AVX2 on zen3, AVX-512 on zen4); the transfer function itself is a
/// lookup, so no `powf` runs in the hot loop.
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
pub fn encode_linear_to_srgb_u8(src: &[f32], dst: &mut [u8]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "encode_linear_to_srgb_u8: src/dst length mismatch"
    );
    let table = srgb_encode_u8_table();
    let scale = Simd::<f32, LANES>::splat((ENCODE_LEN - 1) as f32);
    let lo = Simd::<f32, LANES>::splat(0.0);
    let hi = Simd::<f32, LANES>::splat(1.0);

    let mut src_chunks = src.chunks_exact(LANES);
    let mut dst_chunks = dst.chunks_exact_mut(LANES);
    for (s, d) in (&mut src_chunks).zip(&mut dst_chunks) {
        let v = Simd::<f32, LANES>::from_slice(s).simd_clamp(lo, hi);
        let idx = (v * scale).cast::<usize>();
        let bytes = Simd::<u8, LANES>::gather_or(table, idx, Simd::splat(0));
        bytes.copy_to_slice(d);
    }

    // Scalar tail (fewer than LANES pixels left).
    for (s, d) in src_chunks
        .remainder()
        .iter()
        .zip(dst_chunks.into_remainder())
    {
        let idx = (s.clamp(0.0, 1.0) * (ENCODE_LEN - 1) as f32) as usize;
        *d = table[idx];
    }
}
