// Strata on the GPU: the saliency of a luma plane and the guided filter that
// refines the layer weights. Every entry point gets the bind-group layout of the
// bindings it uses. Planes are stored one after the other in a storage buffer:
// element = plane * (w * h) + y * w + x. Grids are 16 x 16 workgroups over (x, y).
//
// Saliency is `stacker_algo::strata::saliency::compute_saliency`: the magnitude of
// the 4-neighbour Laplacian (border replicated), then five passes of the 5-tap
// pyramid blur (reflecting border, accumulated left to right).
//
// The guided filter is `stacker_algo::relief::guided::guided_filter_pair_luma` for
// one (radius, eps) pair per chain: box means of g, p, g*g, g*p, then
// a = cov / (max(var, 0) + eps), b = mean_p - a * mean_g, box means of a and b,
// q = mean_a * g + mean_b. The CPU takes the window sums from f64 summed-area
// tables; here they are direct f32 sums over the window (two separable passes) of
// values shifted by a constant near their mean. Covariance and variance do not
// change under that shift, and it keeps the f32 sums small, which is what makes
// the variance (a difference of two nearly equal means) usable.

struct Params {
    w: u32,
    h: u32,
    // saliency: unused; guided: the window half-size.
    radius: u32,
    // guided: which plane of `outp` receives q.
    out_plane: u32,
    // guided: the regularisation.
    eps: f32,
    // guided: subtracted from the guidance and from the source.
    shift_g: f32,
    shift_p: f32,
    // blur: the kernel taps [k0, k1, k2, k1, k0].
    k0: f32,
    k1: f32,
    k2: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
// The inputs: plane 0 = guidance (or the luma), plane 1 = the source.
@group(0) @binding(1) var<storage, read> inp: array<f32>;
// Four planes of working space: the saliency and its blur temp, or the horizontal
// window sums.
@group(0) @binding(2) var<storage, read_write> hs: array<f32>;
// The guided filter coefficients a (plane 0) and b (plane 1).
@group(0) @binding(3) var<storage, read_write> ab: array<f32>;
// The results of the two chains (two planes).
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;

// ── Saliency ─────────────────────────────────────────────────────────────────

// |up + down + left + right - 4 * center| with the border replicated, into hs 0.
@compute @workgroup_size(16, 16, 1)
fn laplacian(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let x = id.x;
    let y = id.y;
    let left = max(x, 1u) - 1u;
    let right = min(x + 1u, p.w - 1u);
    let up = max(y, 1u) - 1u;
    let down = min(y + 1u, p.h - 1u);
    let center = inp[y * p.w + x];
    let u = inp[up * p.w + x];
    let d = inp[down * p.w + x];
    let l = inp[y * p.w + left];
    let r = inp[y * p.w + right];
    hs[y * p.w + x] = abs(fma(-4.0, center, u + d + l + r));
}

// The mirror of an out-of-range coordinate, as the CPU blur does: -i, 2n - 2 - i,
// and 0 when that is still out of range (images narrower than 3).
fn reflect_index(i: i32, n: i32) -> i32 {
    var k = i;
    if (k < 0) {
        k = -k;
    } else if (k >= n) {
        k = 2 * n - 2 - k;
    }
    if (k < 0 || k >= n) {
        k = 0;
    }
    return k;
}

// Horizontal blur of hs 0 into hs 1.
@compute @workgroup_size(16, 16, 1)
fn blur_h(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let n = i32(p.w);
    let x = i32(id.x);
    let row = id.y * p.w;
    var s = p.k0 * hs[row + u32(reflect_index(x - 2, n))];
    s = s + p.k1 * hs[row + u32(reflect_index(x - 1, n))];
    s = s + p.k2 * hs[row + u32(reflect_index(x, n))];
    s = s + p.k1 * hs[row + u32(reflect_index(x + 1, n))];
    s = s + p.k0 * hs[row + u32(reflect_index(x + 2, n))];
    hs[p.w * p.h + row + id.x] = s;
}

// Vertical blur of hs 1 back into hs 0.
@compute @workgroup_size(16, 16, 1)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let n = i32(p.h);
    let y = i32(id.y);
    let plane = p.w * p.h;
    var s = p.k0 * hs[plane + u32(reflect_index(y - 2, n)) * p.w + id.x];
    s = s + p.k1 * hs[plane + u32(reflect_index(y - 1, n)) * p.w + id.x];
    s = s + p.k2 * hs[plane + u32(reflect_index(y, n)) * p.w + id.x];
    s = s + p.k1 * hs[plane + u32(reflect_index(y + 1, n)) * p.w + id.x];
    s = s + p.k0 * hs[plane + u32(reflect_index(y + 2, n)) * p.w + id.x];
    hs[id.y * p.w + id.x] = s;
}

// ── Guided filter ────────────────────────────────────────────────────────────

// Horizontal window sums of g', p', g'^2 and g'*p' (g' = g - shift_g, p' = p -
// shift_p) over the clamped columns around (x, y), into hs 0..3.
@compute @workgroup_size(16, 16, 1)
fn box_h_moments(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let r = i32(p.radius);
    let x = i32(id.x);
    let n = p.w * p.h;
    let row = id.y * p.w;
    var s = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var c = max(x - r, 0); c <= min(x + r, i32(p.w) - 1); c = c + 1) {
        let g = inp[row + u32(c)] - p.shift_g;
        let q = inp[n + row + u32(c)] - p.shift_p;
        s = s + vec4<f32>(g, q, g * g, g * q);
    }
    let i = row + id.x;
    hs[i] = s.x;
    hs[n + i] = s.y;
    hs[2u * n + i] = s.z;
    hs[3u * n + i] = s.w;
}

// The vertical sums of hs 0..3 at (x, y) divided by the window size: the four
// window means.
fn means4(id: vec3<u32>) -> vec4<f32> {
    let r = i32(p.radius);
    let x = i32(id.x);
    let y = i32(id.y);
    let n = p.w * p.h;
    let y0 = max(y - r, 0);
    let y1 = min(y + r, i32(p.h) - 1);
    var s = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var row = y0; row <= y1; row = row + 1) {
        let j = u32(row) * p.w + id.x;
        s = s + vec4<f32>(hs[j], hs[n + j], hs[2u * n + j], hs[3u * n + j]);
    }
    let cols = min(x + r, i32(p.w) - 1) - max(x - r, 0) + 1;
    return s / f32(cols * (y1 - y0 + 1));
}

// The vertical sums of hs 0 and 1 at (x, y) divided by the window size.
fn means2(id: vec3<u32>) -> vec2<f32> {
    let r = i32(p.radius);
    let x = i32(id.x);
    let y = i32(id.y);
    let n = p.w * p.h;
    let y0 = max(y - r, 0);
    let y1 = min(y + r, i32(p.h) - 1);
    var s = vec2<f32>(0.0, 0.0);
    for (var row = y0; row <= y1; row = row + 1) {
        let j = u32(row) * p.w + id.x;
        s = s + vec2<f32>(hs[j], hs[n + j]);
    }
    let cols = min(x + r, i32(p.w) - 1) - max(x - r, 0) + 1;
    return s / f32(cols * (y1 - y0 + 1));
}

// a and b of the guided filter into ab 0 and 1. The means are those of the
// shifted values; the covariance and the variance are shift invariant, and b
// restores the shifts.
@compute @workgroup_size(16, 16, 1)
fn coeffs(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let m = means4(id);
    let cov = m.w - m.x * m.y;
    let variance = max(m.z - m.x * m.x, 0.0);
    let a = cov / (variance + p.eps);
    let i = id.y * p.w + id.x;
    ab[i] = a;
    ab[p.w * p.h + i] = (p.shift_p + m.y) - a * (p.shift_g + m.x);
}

// Horizontal window sums of a and b (ab 0 and 1) into hs 0 and 1.
@compute @workgroup_size(16, 16, 1)
fn box_h_ab(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let r = i32(p.radius);
    let x = i32(id.x);
    let n = p.w * p.h;
    let row = id.y * p.w;
    var s = vec2<f32>(0.0, 0.0);
    for (var c = max(x - r, 0); c <= min(x + r, i32(p.w) - 1); c = c + 1) {
        s = s + vec2<f32>(ab[row + u32(c)], ab[n + row + u32(c)]);
    }
    hs[row + id.x] = s.x;
    hs[n + row + id.x] = s.y;
}

// q = mean(a) * g + mean(b) into plane `out_plane` of the results.
@compute @workgroup_size(16, 16, 1)
fn blend(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let m = means2(id);
    let i = id.y * p.w + id.x;
    outp[p.out_plane * p.w * p.h + i] = m.x * inp[i] + m.y;
}
