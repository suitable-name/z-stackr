//! Colour model and sRGB transfer functions.
//!
//! Three layers:
//!
//! * the scalar transfer functions ([`srgb_to_linear`], [`linear_to_srgb`]) and
//!   their exact integer quantisers ([`linear_to_srgb_u8`], [`linear_to_srgb_u16`]);
//! * the luma/chroma model every plane of a [`PlanarImage`](crate::image::PlanarImage)
//!   is expressed in ([`LUMA_CHROMA_FORWARD`], [`LUMA_CHROMA_INVERSE`] and the two
//!   per-pixel conversions);
//! * lookup-table decoders and a `std::simd` row encoder ([`encode_row_u8`]) so the
//!   8-bit hot paths never call `powf` per pixel; the 16-bit encoder
//!   ([`encode_row_u16`]) classifies a sample by a table search and calls the
//!   scalar transfer function only inside a narrow guard zone around a code
//!   boundary, so it equals [`linear_to_srgb_u16`] exactly.
//!
//! The row encoder is a vector body plus a scalar tail that perform the same
//! operations in the same order, so a sample's code never depends on which of the
//! two handled it, nor on the vector width the build selected (see `LANES`).

/// IEC 61966-2-1 sRGB decode: gamma-encoded `v` in `[0, 1]` to linear light.
#[must_use]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// IEC 61966-2-1 sRGB encode: linear light `v` in `[0, 1]` to gamma-encoded.
#[must_use]
#[expect(
    clippy::suboptimal_flops,
    reason = "mul_add would change the rounding of the reference transfer function that every table is built from"
)]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

// ── Luma / chroma model ───────────────────────────────────────────────────────
//
// The whole pipeline stores a frame as three planes (`luma`, `chroma_a`,
// `chroma_b`) derived from **linear-light** RGB with the BT.601 / JFIF
// coefficients. These two matrices are the only definition of that model: the
// loaders, the output encoders, the preview, the retouch export and the neural
// bridge all go through them (directly or through the two functions below).

/// Linear RGB to `[Y, Cb, Cr]`: row `i` holds the weights of `R, G, B` for
/// output `i`. `Y` is in `[0, 1]`, `Cb` and `Cr` in `[-0.5, 0.5]`.
pub const LUMA_CHROMA_FORWARD: [[f32; 3]; 3] = [
    [0.299, 0.587, 0.114],
    [-0.168_736, -0.331_264, 0.5],
    [0.5, -0.418_688, -0.081_312],
];

/// `[Y, Cb, Cr]` to linear RGB: row `i` holds the weights of `Y, Cb, Cr` for
/// output `i` (the first column is 1 and the zero entries are structural).
pub const LUMA_CHROMA_INVERSE: [[f32; 3]; 3] = [
    [1.0, 0.0, 1.402],
    [1.0, -0.344_136, -0.714_136],
    [1.0, 1.772, 0.0],
];

/// Linear-light `[R, G, B]` to the pipeline's `[luma, chroma_a, chroma_b]`
/// ([`LUMA_CHROMA_FORWARD`]).
#[inline]
#[must_use]
#[expect(
    clippy::suboptimal_flops,
    reason = "plain multiply-add keeps the conversion bit-identical on targets without FMA and across every caller"
)]
pub fn linear_rgb_to_planar(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let m = &LUMA_CHROMA_FORWARD;
    [
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    ]
}

/// The pipeline's `[luma, chroma_a, chroma_b]` back to linear-light
/// `[R, G, B]` ([`LUMA_CHROMA_INVERSE`]). The result is **not** clamped: the
/// fused image can overshoot `[0, 1]`, and the sRGB encoders clamp.
#[inline]
#[must_use]
#[expect(
    clippy::suboptimal_flops,
    reason = "plain multiply-add keeps the conversion bit-identical on targets without FMA and across every caller"
)]
pub fn planar_to_linear_rgb(ycc: [f32; 3]) -> [f32; 3] {
    let [y, cb, cr] = ycc;
    let m = &LUMA_CHROMA_INVERSE;
    [
        y + m[0][2] * cr,
        y + m[1][1] * cb + m[1][2] * cr,
        y + m[2][1] * cb,
    ]
}

// ── Lookup-table + SIMD accelerated colour conversion ─────────────────────────
//
// The sRGB transfer function is a `powf`, which does not auto-vectorise and has
// no portable-SIMD intrinsic. The decoders and the 8-bit encoder read lookup
// tables instead, so they never call `powf` at runtime, and use `std::simd`
// portable vectors for the surrounding arithmetic (4, 8 or 16 lanes by target
// feature, see `LANES`). The 16-bit encoder is a scalar table search.

use std::{simd::prelude::*, sync::OnceLock};

/// SIMD width in `f32` lanes, selected from the compile-time target features:
/// 16 (one `zmm` register) with `avx512f`, 8 (one `ymm`) with `avx2`, otherwise
/// 4 (one `xmm`, the SSE2 baseline and NEON).
///
/// `std::simd` has no runtime dispatch: the vector type is fixed when the crate
/// is compiled, so the width follows `-C target-cpu` / `-C target-feature`. The
/// choice only affects speed. Every lane is processed by elementwise operations
/// (clamp, multiply, add, compare, gather), so the encoded bytes are identical
/// for every width and identical to the scalar tail; the tests compare the row
/// encoder with the scalar reference at lengths that leave a tail.
const LANES: usize = if cfg!(target_feature = "avx512f") {
    16
} else if cfg!(target_feature = "avx2") {
    8
} else {
    4
};

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
        for (code, slot) in (0..=u8::MAX).zip(t.iter_mut()) {
            *slot = srgb_to_linear(f32::from(code) / 255.0);
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

/// Largest index of the 14-bit encode table, as the `f32` scale used to index it.
const ENCODE_MAX_INDEX: f32 = 16_383.0;
const _: () = assert!(
    ENCODE_LEN - 1 == 16_383,
    "ENCODE_MAX_INDEX must track ENCODE_BITS"
);

/// Per-code decision thresholds of the 8-bit encoder.
///
/// `.0[c]` is the smallest linear value that encodes to code `c + 1`
/// (`srgb_to_linear((c + 0.5) / 255)`; `+inf` for `c == 255`), and `.1[c]` is
/// the smallest linear value that still encodes to code `c` (`-inf` for
/// `c == 0`). The 14-bit table is within one code of the exact answer, and a
/// single compare against these two thresholds makes it exact.
static ENCODE_THRESHOLDS: OnceLock<([f32; 256], [f32; 256])> = OnceLock::new();

fn encode_thresholds() -> &'static ([f32; 256], [f32; 256]) {
    ENCODE_THRESHOLDS.get_or_init(|| {
        let mut up = [f32::INFINITY; 256];
        let mut down = [f32::NEG_INFINITY; 256];
        for (c, slot) in (0..u8::MAX).zip(up.iter_mut()) {
            let t = srgb_to_linear((f32::from(c) + 0.5) / 255.0);
            *slot = t;
            down[usize::from(c) + 1] = t;
        }
        (up, down)
    })
}

/// Clamp to `[0, 1]`, mapping NaN to 0.
#[inline]
const fn unit_clamp(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

/// Exact `round(linear_to_srgb(clamp(v, 0, 1)) * 255)` as an 8-bit code.
///
/// Rounds (never truncates): `1.0` encodes to 255 and `0.0` to 0. NaN encodes
/// to 0. This is the scalar reference the row encoder [`encode_row_u8`] agrees
/// with; it costs one `powf`, so use the row encoder for whole rows.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the argument is clamped to [0, 1] and rounded, so the product lies in 0..=255"
)]
pub fn linear_to_srgb_u8(v: f32) -> u8 {
    (linear_to_srgb(unit_clamp(v)) * 255.0).round() as u8
}

/// Exact `round(linear_to_srgb(clamp(v, 0, 1)) * 65535)` as a 16-bit code.
///
/// Rounds (never truncates): `1.0` encodes to 65535 and `0.0` to 0. NaN
/// encodes to 0.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the argument is clamped to [0, 1] and rounded, so the product lies in 0..=65535"
)]
pub fn linear_to_srgb_u16(v: f32) -> u16 {
    (linear_to_srgb(unit_clamp(v)) * 65_535.0).round() as u16
}

/// `linear_to_srgb` quantised to 8-bit output, indexed by a 14-bit linear value.
///
/// Index it **rounded**: `idx = (clamp(lin, 0, 1) * (len - 1) + 0.5) as usize`.
/// Even then an entry is the code of the cell's own sample point, so it can be
/// one code away from the exact answer near black (cells are 0.2 codes wide
/// there); use [`encode_row_u8`] or [`linear_to_srgb_u8`] where exactness
/// matters.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the index is below 2^14, exact in f32, and the code is clamped to 0..=255 before the cast"
)]
pub fn srgb_encode_u8_table() -> &'static [u8] {
    ENCODE_U8
        .get_or_init(|| {
            (0..ENCODE_LEN)
                .map(|i| {
                    let lin = i as f32 / ENCODE_MAX_INDEX;
                    (linear_to_srgb(lin) * 255.0).round().clamp(0.0, 255.0) as u8
                })
                .collect()
        })
        .as_slice()
}

/// One lane of [`encode_row_u8`] in scalar form (the tail path). Operation for
/// operation the same as the vector body: rounded table index, then one compare
/// against each neighbouring code threshold.
#[expect(
    clippy::suboptimal_flops,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "operation-for-operation twin of the SIMD body (no FMA); the index lies in 0..=16383 and the corrected code in 0..=255"
)]
fn encode_one_u8(v: f32, table: &[u8], up: &[f32; 256], down: &[f32; 256]) -> u8 {
    let v = unit_clamp(v);
    let idx = (v * ENCODE_MAX_INDEX + 0.5) as usize;
    let code = table[idx];
    let c = usize::from(code);
    let adjusted = i32::from(code) + i32::from(v >= up[c]) - i32::from(v < down[c]);
    adjusted as u8
}

/// Encode a row of **linear-light** `f32` samples to 8-bit sRGB.
///
/// `out[i] = round(linear_to_srgb(clamp(linear[i], 0, 1)) * 255)`, exactly (it
/// agrees with [`linear_to_srgb_u8`] except possibly within an `f32` ulp of a
/// rounding boundary). White (`1.0`) is 255, black is 0, NaN is 0.
///
/// `std::simd` performs the clamp, the rounded 14-bit table lookup and the
/// exactness correction `LANES` lanes at a time, so no `powf` runs in the hot
/// loop. The table alone is up to one code
/// off near black; a compare against the two neighbouring code thresholds
/// fixes that.
///
/// # The byte gather stays a gather of bytes
/// The code table is read with `Simd<u8, _>::gather_or`. x86 has no 8-bit
/// hardware gather (AVX2 and AVX-512 gather 32- and 64-bit elements only), so
/// that one load is lowered to per-lane scalar loads on every target; the two
/// threshold gathers (`f32`) do use the hardware gather where the build has it.
/// A two-stage table (a `u32` code table) would turn the byte gather into a
/// hardware gather but quadruple the table from 16 KiB to 64 KiB, which no longer
/// fits beside the two 1 KiB threshold tables in a 32 KiB L1 data cache, so the
/// compact byte table is kept.
///
/// # Panics
/// Panics if `linear.len() != out.len()`.
pub fn encode_row_u8(linear: &[f32], out: &mut [u8]) {
    assert_eq!(
        linear.len(),
        out.len(),
        "encode_row_u8: src/dst length mismatch"
    );
    let table = srgb_encode_u8_table();
    let (up_t, down_t) = encode_thresholds();
    let scale = Simd::<f32, LANES>::splat(ENCODE_MAX_INDEX);
    let half = Simd::<f32, LANES>::splat(0.5);
    let lo = Simd::<f32, LANES>::splat(0.0);
    let hi = Simd::<f32, LANES>::splat(1.0);
    let pos_inf = Simd::<f32, LANES>::splat(f32::INFINITY);
    let neg_inf = Simd::<f32, LANES>::splat(f32::NEG_INFINITY);

    let (src_chunks, src_tail) = linear.as_chunks::<LANES>();
    let (dst_chunks, dst_tail) = out.as_chunks_mut::<LANES>();
    for (s, d) in src_chunks.iter().zip(dst_chunks.iter_mut()) {
        // simd_max/simd_min return the non-NaN operand, so NaN lands on 0.
        let v = Simd::<f32, LANES>::from_array(*s).simd_max(lo).simd_min(hi);
        let idx = (v * scale + half).cast::<usize>();
        let code = Simd::<u8, LANES>::gather_or(table, idx, Simd::splat(0));
        let code_idx = code.cast::<usize>();
        let up = Simd::<f32, LANES>::gather_or(up_t, code_idx, pos_inf);
        let down = Simd::<f32, LANES>::gather_or(down_t, code_idx, neg_inf);
        // `Mask::to_simd` gives -1 for true lanes: `- up_mask` adds one and
        // `+ down_mask` subtracts one.
        let adjusted = code.cast::<i32>() - v.simd_ge(up).to_simd() + v.simd_lt(down).to_simd();
        *d = adjusted.cast::<u8>().to_array();
    }

    // Scalar tail (fewer than LANES samples left).
    for (s, d) in src_tail.iter().zip(dst_tail.iter_mut()) {
        *d = encode_one_u8(*s, table, up_t, down_t);
    }
}

/// Encode a row of **linear-light** `f32` samples to 16-bit sRGB:
/// `out[i] = round(linear_to_srgb(clamp(linear[i], 0, 1)) * 65535)`.
///
/// White (`1.0`) is 65535, black is 0, NaN is 0. The result is **exactly**
/// [`linear_to_srgb_u16`] for every input (tables with a scalar fallback inside
/// a narrow guard zone around each code boundary), but almost
/// every sample is classified by a table search instead of a `powf`.
///
/// # Panics
/// Panics if `linear.len() != out.len()`.
pub fn encode_row_u16(linear: &[f32], out: &mut [u16]) {
    assert_eq!(
        linear.len(),
        out.len(),
        "encode_row_u16: src/dst length mismatch"
    );
    let tables = encode_u16_tables();
    for (s, d) in linear.iter().zip(out.iter_mut()) {
        *d = encode_one_u16(*s, tables);
    }
}

/// Number of decision edges of the 16-bit encoder: the boundaries between the
/// 65 536 codes.
const U16_EDGES: usize = 65_535;
/// Cells of the coarse index over `[0, 1]` (a power of two, so the cell of a
/// value is an exact multiply). One extra cell holds the value `1.0`.
const U16_CELLS: usize = 1 << 14;
/// [`U16_CELLS`] as a float (kept literal so no cast is needed).
const U16_CELLS_F32: f32 = 16_384.0;
/// Half width, in output codes, of the guard zone around every decision edge.
/// A sample inside a zone is evaluated with the exact scalar function. The
/// scalar `f32` evaluation differs from the real-number curve by well under
/// 0.02 code (a few `f32` roundings of a value below 1, scaled by 65535), so a
/// zone of 0.05 code on each side of the edge contains every sample whose
/// scalar code could differ from the table code.
const U16_GUARD: f64 = 0.05;
/// Largest linear value on the linear segment of [`linear_to_srgb`]; the
/// constant is the same literal the transfer function compares against.
const LINEAR_SEGMENT_END: f32 = 0.003_130_8;

/// Decision tables of the 16-bit encoder.
struct EncodeU16Tables {
    /// `lo[k]`: values at or above it are past edge `k` (the `k + 1` code
    /// boundary) minus the guard; `-inf` where that lies on the linear segment.
    lo: Vec<f32>,
    /// `hi[k]`: values at or above it are past edge `k` plus the guard.
    hi: Vec<f32>,
    /// `start_lo[i]`: number of `lo` entries at or below `i / U16_CELLS`.
    start_lo: Vec<u32>,
    /// Same for `hi`.
    start_hi: Vec<u32>,
}

static ENCODE_U16: OnceLock<EncodeU16Tables> = OnceLock::new();

/// The linear value whose scalar encoding is the (fractional) code `position`,
/// evaluated in `f64` with the same `f32` constants and exponent the scalar
/// transfer function uses.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the edge is rounded to the nearest f32 on purpose; the guard zone absorbs the rounding"
)]
fn u16_edge(position: f64) -> f32 {
    let exponent = f64::from(1.0_f32 / 2.4_f32);
    let scale = f64::from(1.055_f32);
    let offset = f64::from(0.055_f32);
    let encoded = position / 65_535.0;
    (((encoded + offset) / scale).powf(1.0 / exponent)) as f32
}

fn encode_u16_tables() -> &'static EncodeU16Tables {
    ENCODE_U16.get_or_init(|| {
        let edge_or_neg_inf = |position: f64| {
            let v = u16_edge(position);
            if v <= LINEAR_SEGMENT_END {
                f32::NEG_INFINITY
            } else {
                v
            }
        };
        let mut lo = Vec::with_capacity(U16_EDGES);
        let mut hi = Vec::with_capacity(U16_EDGES);
        for k in 0..U16_EDGES {
            #[expect(clippy::cast_precision_loss, reason = "k < 65535 is exact in f64")]
            let centre = k as f64 + 0.5;
            lo.push(edge_or_neg_inf(centre - U16_GUARD));
            hi.push(edge_or_neg_inf(centre + U16_GUARD));
        }
        let starts = |edges: &[f32]| -> Vec<u32> {
            (0..=U16_CELLS + 1)
                .map(|cell| {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "cell <= 2^14 + 1 is exact in f32"
                    )]
                    let bound = cell as f32 / U16_CELLS_F32;
                    // At most 65 535, fits u32.
                    u32::try_from(edges.partition_point(|&e| e <= bound)).unwrap_or(u32::MAX)
                })
                .collect()
        };
        let start_lo = starts(&lo);
        let start_hi = starts(&hi);
        EncodeU16Tables {
            lo,
            hi,
            start_lo,
            start_hi,
        }
    })
}

/// Number of entries of `edges` at or below `v`, searched only inside the span
/// the coarse cell `cell` allows.
#[inline]
fn count_edges_below(edges: &[f32], start: &[u32], cell: usize, v: f32) -> usize {
    let first = start[cell] as usize;
    let last = start[cell + 1] as usize;
    first + edges[first..last].partition_point(|&e| e <= v)
}

/// Exact `round(linear_to_srgb(clamp(v, 0, 1)) * 65535)` as a 16-bit code,
/// equal to [`linear_to_srgb_u16`] for every `f32` input.
///
/// On the linear segment the scalar expression is evaluated directly. Above it
/// the code is the number of decision edges at or below `v`, found by a short
/// search inside a coarse cell of the 16 K index. Every edge has a guard zone
/// of 0.05 code on both sides; a sample inside a zone (about one in ten) is
/// handed to the scalar function, so no sample can be misclassified by the
/// `f32` rounding of the table.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "v is clamped to [0, 1] so the cell index is in 0..=16384 and the code in 0..=65535"
)]
fn encode_one_u16(v: f32, tables: &EncodeU16Tables) -> u16 {
    let v = unit_clamp(v);
    if v <= LINEAR_SEGMENT_END {
        return (linear_to_srgb(v) * 65_535.0).round() as u16;
    }
    let cell = (v * U16_CELLS_F32) as usize;
    let below_lo = count_edges_below(&tables.lo, &tables.start_lo, cell, v);
    let below_hi = count_edges_below(&tables.hi, &tables.start_hi, cell, v);
    if below_lo == below_hi {
        below_lo as u16
    } else {
        linear_to_srgb_u16(v)
    }
}
