// GPU port of `transform::warp::spline4x4_sample_clamped` /
// `warp_image_clamped` (crates/stacker-align/src/transform/warp.rs) — the
// PRODUCTION edge-clamped 4-tap spline kernel every active alignment mode
// warps with. This deliberately does NOT implement the Catmull-Rom bicubic
// kernel from the retired `warp_image` (that shader has been removed from
// this crate's GPU path — see the module docs on why).
//
// Weights and edge-clamp behaviour mirror the CPU kernel exactly (same
// polynomial coefficients, same "clamp tap index to the nearest valid pixel,
// then renormalise by the summed weight" border rule) so GPU output is
// tolerance-equal to the CPU/SIMD path — see the Rust module docs for the
// tested epsilon.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba32float, write>;

struct Uniforms {
    m_inv: mat3x4<f32>, // WGSL uses mat3x4 for layout padding compatibility
    width: u32,
    height: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

/// 4-tap spline weights for fractional offset `t` in `[0, 1)` — identical
/// polynomial to `transform::warp::spline4` (a partition of unity: the four
/// weights always sum to 1).
fn spline4(t: f32) -> vec4<f32> {
    let w0 = ((-0.33333334 * t + 0.8) * t - 0.46666667) * t;
    let w1 = ((t - 1.8) * t - 0.2) * t + 1.0;
    let w2 = ((1.2 - t) * t + 0.8) * t;
    let w3 = ((0.33333334 * t - 0.2) * t - 0.13333334) * t;
    return vec4<f32>(w0, w1, w2, w3);
}

/// Edge-clamped 4x4-tap spline sample at `(x, y)` — the GPU equivalent of
/// `spline4x4_sample_clamped`. Always takes the "border" code path (clamp
/// every tap index to the valid range then renormalise), which is
/// numerically identical to the CPU fast path in-bounds because the four
/// spline weights are a partition of unity regardless of clamping.
fn sample_clamped(x: f32, y: f32) -> vec4<f32> {
    let w = i32(params.width);
    let h = i32(params.height);

    let idx_x = i32(floor(x));
    let idx_y = i32(floor(y));
    let wx = spline4(x - floor(x));
    let wy = spline4(y - floor(y));

    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum_w: f32 = 0.0;
    for (var ky: i32 = 0; ky < 4; ky++) {
        let cy = clamp(idx_y - 1 + ky, 0, h - 1);
        let wyk = wy[ky];
        for (var kx: i32 = 0; kx < 4; kx++) {
            let cx = clamp(idx_x - 1 + kx, 0, w - 1);
            let wxk = wx[kx];
            let weight = wxk * wyk;
            sum_w = sum_w + weight;
            let texel = textureLoad(src_texture, vec2<i32>(cx, cy), 0);
            acc = acc + texel * weight;
        }
    }

    if (abs(sum_w) < 1e-12) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return acc / sum_w;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= params.width || y >= params.height) { return; }

    let xf = f32(x);
    let yf = f32(y);
    // Index convention: the Rust side packs the ROWS of `m_inv` into the
    // three vec4 columns of this `mat3x4`, so `params.m_inv[i][j]` is the
    // matrix element `m_inv[(i, j)]` — row `i`, column `j`. The inverse map
    // is therefore `src = row_i · (x, y, 1)`, exactly like the CPU kernel's
    // `m00*x + m01*y + m02` per row (`transform::warp::warp_image_clamped_cpu`).
    let denom = params.m_inv[2][0] * xf + params.m_inv[2][1] * yf + params.m_inv[2][2];
    let inv_denom = 1.0 / denom;

    let src_x = (params.m_inv[0][0] * xf + params.m_inv[0][1] * yf + params.m_inv[0][2]) * inv_denom;
    let src_y = (params.m_inv[1][0] * xf + params.m_inv[1][1] * yf + params.m_inv[1][2]) * inv_denom;

    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    // Non-finite source coordinates (NaN from a near-singular matrix, or an
    // out-of-range infinity) sample as zero, matching
    // `spline4x4_sample_clamped`'s own non-finite guard. NaN comparisons are
    // always false (including `<`/`>`), so the innocuous-looking bounds
    // check below doubles as the NaN guard: a NaN `src_x`/`src_y` fails both
    // branches of `is_finite_coord` and falls through to the zero default.
    let is_finite_coord = (src_x > -3.4e38 && src_x < 3.4e38) && (src_y > -3.4e38 && src_y < 3.4e38);
    if (is_finite_coord) {
        color = sample_clamped(src_x, src_y);
    }
    textureStore(dst_texture, vec2<i32>(i32(x), i32(y)), color);
}
