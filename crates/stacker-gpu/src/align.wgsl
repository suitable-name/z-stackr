// Alignment on the GPU: the registration objective (warp the reference, moments
// of source - warped) and the final warp of a frame. One module, two entry
// points; each gets the bind-group layout of the bindings it uses.
//
// The samplers follow `stacker_align::transform` operation for operation:
// - `bicubic`: Catmull-Rom with fused multiply-adds, taps clamped to the frame,
//   only called for a coordinate inside [0, w-1] x [0, h-1];
// - `spline`: the four-tap C0 cubic, weights unfused, a fast path when every
//   tap is inside and a clamped, renormalised border path otherwise.
// The source coordinate is computed the way each CPU caller computes it: the
// objective as `inverse * (x, y, 1)` and a division, the warp per row
// (`RowMap`) with a reciprocal.

struct Params {
    // The inverse matrix, one row per vector (the fourth component unused).
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    w: u32,
    h: u32,
    // 0 = bicubic (outside pixels take no part / are black), 1 = clamped spline.
    kind: u32,
    // moments: the number of workgroups dispatched (grid stride).
    groups: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
// The image sampled: the reference luma (moments) or the frame's three planes
// (warp).
@group(0) @binding(1) var<storage, read> img: array<f32>;
// moments: the source luma.
@group(0) @binding(2) var<storage, read> src: array<f32>;
// moments: one (count, mean, m2, 0) per workgroup. warp: the three warped planes.
@group(0) @binding(3) var<storage, read_write> out_moments: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> out_image: array<f32>;
// warp: one bit per pixel, set when the source coordinate is inside the frame.
@group(0) @binding(5) var<storage, read_write> coverage: array<atomic<u32>>;

const GROUP_SIZE: u32 = 256u;
// Beyond 2^24 an f32 has no fractional part, so clamping there before the
// integer conversion changes no tap and keeps the conversion in range.
const COORD_LIMIT: f32 = 16777216.0;

fn finite(v: f32) -> bool {
    // f32::MAX written exactly: browsers' WGSL compilers reject the decimal
    // `3.4028235e38`, which rounds just above it.
    return abs(v) <= 0x1.fffffep+127f;
}

fn in_bounds(x: f32, y: f32) -> bool {
    return finite(x) && finite(y) && x >= 0.0 && y >= 0.0
        && x <= f32(p.w) - 1.0 && y <= f32(p.h) - 1.0;
}

fn cubic(t: f32) -> f32 {
    let a = abs(t);
    let a2 = a * a;
    let a3 = a2 * a;
    if (a <= 1.0) {
        return fma(2.5, -a2, fma(1.5, a3, 1.0));
    }
    if (a < 2.0) {
        return fma(4.0, -a, fma(2.5, a2, fma(-0.5, a3, 2.0)));
    }
    return 0.0;
}

// One row of a 4 x 4 sample: Σ wx[k] * img[row + tx[k]], fused, k in order.
fn row4(row: u32, tx: vec4<u32>, wx: vec4<f32>) -> f32 {
    var s = fma(wx.x, img[row + tx.x], 0.0);
    s = fma(wx.y, img[row + tx.y], s);
    s = fma(wx.z, img[row + tx.z], s);
    s = fma(wx.w, img[row + tx.w], s);
    return s;
}

// Catmull-Rom sample of the plane starting at `base`, for an in-bounds (x, y).
// The taps at floor - 1 ..= floor + 2 are clamped to the frame.
fn bicubic(base: u32, x: f32, y: f32) -> f32 {
    let fx = floor(x);
    let fy = floor(y);
    let dx = x - fx;
    let dy = y - fy;
    let ix = i32(fx);
    let iy = i32(fy);
    let wmax = i32(p.w) - 1;
    let hmax = i32(p.h) - 1;
    // cubic(d - i) for i = -1, 0, 1, 2.
    let wx = vec4<f32>(cubic(dx - -1.0), cubic(dx - 0.0), cubic(dx - 1.0), cubic(dx - 2.0));
    let wy = vec4<f32>(cubic(dy - -1.0), cubic(dy - 0.0), cubic(dy - 1.0), cubic(dy - 2.0));
    let tx = vec4<u32>(clamp(vec4<i32>(ix - 1, ix, ix + 1, ix + 2), vec4<i32>(0), vec4<i32>(wmax)));
    let ty = vec4<u32>(clamp(vec4<i32>(iy - 1, iy, iy + 1, iy + 2), vec4<i32>(0), vec4<i32>(hmax)));
    var sum = fma(wy.x, row4(base + ty.x * p.w, tx, wx), 0.0);
    sum = fma(wy.y, row4(base + ty.y * p.w, tx, wx), sum);
    sum = fma(wy.z, row4(base + ty.z * p.w, tx, wx), sum);
    sum = fma(wy.w, row4(base + ty.w * p.w, tx, wx), sum);
    return sum;
}

fn spline_weights(t: f32) -> vec4<f32> {
    return vec4<f32>(
        ((-0.33333334 * t + 0.8) * t - 0.46666667) * t,
        ((t - 1.8) * t - 0.2) * t + 1.0,
        ((1.2 - t) * t + 0.8) * t,
        ((0.33333334 * t - 0.2) * t - 0.13333334) * t,
    );
}

// One row of the clamped border path: adds the four weights wx * wy_k to
// `sum_w` and the weighted samples to `acc`, in tap order.
fn border_row(row: u32, tx: vec4<u32>, wx: vec4<f32>, wyk: f32, acc_in: vec2<f32>) -> vec2<f32> {
    // Unrolled: no run-time index into a vector (drivers differ on that).
    let w0 = wx.x * wyk;
    let w1 = wx.y * wyk;
    let w2 = wx.z * wyk;
    let w3 = wx.w * wyk;
    var sum_w = acc_in.x + w0;
    var acc = fma(w0, img[row + tx.x], acc_in.y);
    sum_w = sum_w + w1;
    acc = fma(w1, img[row + tx.y], acc);
    sum_w = sum_w + w2;
    acc = fma(w2, img[row + tx.z], acc);
    sum_w = sum_w + w3;
    acc = fma(w3, img[row + tx.w], acc);
    return vec2<f32>(sum_w, acc);
}

// The clamped spline sample of the plane starting at `base`; 0 for a
// non-finite coordinate.
fn spline(base: u32, x: f32, y: f32) -> f32 {
    if (!finite(x) || !finite(y)) {
        return 0.0;
    }
    let cx = clamp(x, -COORD_LIMIT, COORD_LIMIT);
    let cy = clamp(y, -COORD_LIMIT, COORD_LIMIT);
    let fx = floor(cx);
    let fy = floor(cy);
    let wx = spline_weights(cx - fx);
    let wy = spline_weights(cy - fy);
    let ix = i32(fx);
    let iy = i32(fy);
    let iw = i32(p.w);
    let ih = i32(p.h);
    if (ix >= 1 && ix <= iw - 3 && iy >= 1 && iy <= ih - 3) {
        let tx = vec4<u32>(vec4<i32>(ix - 1, ix, ix + 1, ix + 2));
        let r0 = base + u32(iy - 1) * p.w;
        var val = fma(wy.x, row4(r0, tx, wx), 0.0);
        val = fma(wy.y, row4(r0 + p.w, tx, wx), val);
        val = fma(wy.z, row4(r0 + 2u * p.w, tx, wx), val);
        val = fma(wy.w, row4(r0 + 3u * p.w, tx, wx), val);
        return val;
    }
    let tx = vec4<u32>(clamp(vec4<i32>(ix - 1, ix, ix + 1, ix + 2), vec4<i32>(0), vec4<i32>(iw - 1)));
    let ty = vec4<u32>(clamp(vec4<i32>(iy - 1, iy, iy + 1, iy + 2), vec4<i32>(0), vec4<i32>(ih - 1)));
    var acc = vec2<f32>(0.0, 0.0);
    acc = border_row(base + ty.x * p.w, tx, wx, wy.x, acc);
    acc = border_row(base + ty.y * p.w, tx, wx, wy.y, acc);
    acc = border_row(base + ty.z * p.w, tx, wx, wy.z, acc);
    acc = border_row(base + ty.w * p.w, tx, wx, wy.w, acc);
    if (abs(acc.x) < 1.0e-12) {
        return 0.0;
    }
    return acc.y / acc.x;
}

// ── Objective ────────────────────────────────────────────────────────────────

var<workgroup> wg_count: array<f32, 256>;
var<workgroup> wg_mean: array<f32, 256>;
var<workgroup> wg_m2: array<f32, 256>;

// The warped reference at output pixel (x, y), and whether the pixel takes
// part (the CPU marks the others NaN and leaves them out).
fn objective_sample(x: u32, y: u32) -> vec2<f32> {
    let fx = f32(x);
    let fy = f32(y);
    let q0 = p.m0.x * fx + p.m0.y * fy + p.m0.z;
    let q1 = p.m1.x * fx + p.m1.y * fy + p.m1.z;
    let q2 = p.m2.x * fx + p.m2.y * fy + p.m2.z;
    if (abs(q2) < 1.0e-8) {
        return vec2<f32>(0.0, 0.0);
    }
    let sx = q0 / q2;
    let sy = q1 / q2;
    if (p.kind == 0u) {
        if (!in_bounds(sx, sy)) {
            return vec2<f32>(0.0, 0.0);
        }
        return vec2<f32>(bicubic(0u, sx, sy), 1.0);
    }
    return vec2<f32>(spline(0u, sx, sy), 1.0);
}

// Moments of d = source - warped reference: every thread folds its pixels
// (grid stride) with Welford's update, the workgroup merges the threads'
// moments pairwise (Chan et al.), one (count, mean, m2) per workgroup.
@compute @workgroup_size(256, 1, 1)
fn moments(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let n = p.w * p.h;
    let stride = p.groups * GROUP_SIZE;
    var count = 0.0;
    var mean = 0.0;
    var m2 = 0.0;
    for (var i = group.x * GROUP_SIZE + lane; i < n; i = i + stride) {
        let s = objective_sample(i % p.w, i / p.w);
        if (s.y != 0.0) {
            let d = src[i] - s.x;
            count = count + 1.0;
            let delta = d - mean;
            mean = mean + delta / count;
            m2 = m2 + delta * (d - mean);
        }
    }
    wg_count[lane] = count;
    wg_mean[lane] = mean;
    wg_m2[lane] = m2;
    workgroupBarrier();
    for (var half = GROUP_SIZE / 2u; half > 0u; half = half / 2u) {
        if (lane < half) {
            let na = wg_count[lane];
            let nb = wg_count[lane + half];
            if (nb > 0.0) {
                if (na == 0.0) {
                    wg_count[lane] = nb;
                    wg_mean[lane] = wg_mean[lane + half];
                    wg_m2[lane] = wg_m2[lane + half];
                } else {
                    let total = na + nb;
                    let delta = wg_mean[lane + half] - wg_mean[lane];
                    wg_mean[lane] = wg_mean[lane] + delta * (nb / total);
                    wg_m2[lane] = wg_m2[lane] + wg_m2[lane + half]
                        + delta * delta * (na * nb / total);
                    wg_count[lane] = total;
                }
            }
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        out_moments[group.x] = vec4<f32>(wg_count[0], wg_mean[0], wg_m2[0], 0.0);
    }
}

// ── Final warp ───────────────────────────────────────────────────────────────

// Warp the three planes of `img` by the inverse matrix (output pixel -> source
// coordinate per row, as `RowMap`): bicubic with a black border, or the clamped
// spline. The coverage bit is the in-bounds test of `coverage_mask`.
@compute @workgroup_size(16, 16, 1)
fn warp(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let fx = f32(id.x);
    let fy = f32(id.y);
    let c0 = p.m0.y * fy + p.m0.z;
    let c1 = p.m1.y * fy + p.m1.z;
    let c2 = p.m2.y * fy + p.m2.z;
    let q0 = p.m0.x * fx + c0;
    let q1 = p.m1.x * fx + c1;
    let q2 = p.m2.x * fx + c2;
    let r = 1.0 / q2;
    let sx = q0 * r;
    let sy = q1 * r;
    let n = p.w * p.h;
    let i = id.y * p.w + id.x;
    let inside = in_bounds(sx, sy);
    if (inside) {
        atomicOr(&coverage[i / 32u], 1u << (i % 32u));
    }
    for (var plane = 0u; plane < 3u; plane = plane + 1u) {
        var v = 0.0;
        if (p.kind == 0u) {
            if (inside) {
                v = bicubic(plane * n, sx, sy);
            }
        } else {
            v = spline(plane * n, sx, sy);
        }
        out_image[plane * n + i] = v;
    }
}
