// Apex on the GPU: Laplacian pyramid build, strongest-coefficient fold and
// reconstruction. One module, seven entry points, one bind-group layout.
//
// Every image is three planes (luma, chroma_a, chroma_b) one after the other in a
// storage buffer: element = plane * (w * h) + y * w + x. The dispatch grid is
// (x, y, plane) with 16 x 16 x 1 workgroups; the fold dispatches one plane.
//
// The arithmetic follows the CPU code (`stacker_algo::apex`) operation for
// operation: the same 5-tap kernel, the same reflecting border, the same
// polyphase expand, plain multiply and add in kernel order, strict greater-than
// for the winner so that the first frame wins ties.
//
// The three tap constants below are filled in from the Rust constants before the
// module is compiled, so the taps carry the exact bits the CPU uses.

const K_OUTER: f32 = KERNEL_OUTER_VALUE;
const K_INNER: f32 = KERNEL_INNER_VALUE;
const K_CENTRE: f32 = KERNEL_CENTRE_VALUE;

struct Params {
    // Image size the kernel works on (see each entry point for which one).
    w: u32,
    h: u32,
    // Size of the smaller image for the expand kernels.
    sw: u32,
    sh: u32,
    // expand_v: 0 = base - expanded (Laplacian band), 1 = base + expanded.
    // fold: bit 0 = 3 x 3 neighbourhood energy (grit suppression).
    mode: u32,
    // fold: chroma bands take part in the selection energy.
    use_color: u32,
    pad0: u32,
    pad1: u32,
}

struct Globals {
    // 1 for the first frame of a stack: it takes every pixel.
    first: u32,
    n_frames: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read_write> best: array<f32>;
@group(0) @binding(5) var<uniform> g: Globals;

// Reflecting border: -i and 2n - 2 - i, and 0 for a degenerate size.
fn reflect_idx(i: i32, n: i32) -> u32 {
    var k = i;
    if (k < 0) {
        k = -k;
    } else if (k >= n) {
        k = 2 * n - 2 - k;
    }
    if (k < 0 || k >= n) {
        k = 0;
    }
    return u32(k);
}

fn tap(t: i32) -> f32 {
    if (t == 0 || t == 4) {
        return K_OUTER;
    }
    if (t == 1 || t == 3) {
        return K_INNER;
    }
    return K_CENTRE;
}

// Horizontal blur evaluated at the even columns only.
// In: a, p.w x p.h. Out: o, ((p.w + 1) / 2) x p.h.
@compute @workgroup_size(16, 16, 1)
fn down_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let w2 = (p.w + 1u) / 2u;
    if (id.x >= w2 || id.y >= p.h) {
        return;
    }
    let n = i32(p.w);
    let xs = i32(2u * id.x);
    let row = id.z * p.w * p.h + id.y * p.w;
    var s = K_OUTER * a[row + reflect_idx(xs - 2, n)];
    s = s + K_INNER * a[row + reflect_idx(xs - 1, n)];
    s = s + K_CENTRE * a[row + reflect_idx(xs, n)];
    s = s + K_INNER * a[row + reflect_idx(xs + 1, n)];
    s = s + K_OUTER * a[row + reflect_idx(xs + 2, n)];
    o[id.z * w2 * p.h + id.y * w2 + id.x] = s;
}

// Vertical blur evaluated at the even rows only.
// In: a, p.w x p.h (p.w is the already halved width). Out: o, p.w x ((p.h + 1) / 2).
@compute @workgroup_size(16, 16, 1)
fn down_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let h2 = (p.h + 1u) / 2u;
    if (id.x >= p.w || id.y >= h2) {
        return;
    }
    let n = i32(p.h);
    let ys = i32(2u * id.y);
    let base = id.z * p.w * p.h;
    var s = K_OUTER * a[base + reflect_idx(ys - 2, n) * p.w + id.x];
    s = s + K_INNER * a[base + reflect_idx(ys - 1, n) * p.w + id.x];
    s = s + K_CENTRE * a[base + reflect_idx(ys, n) * p.w + id.x];
    s = s + K_INNER * a[base + reflect_idx(ys + 1, n) * p.w + id.x];
    s = s + K_OUTER * a[base + reflect_idx(ys + 2, n) * p.w + id.x];
    o[id.z * p.w * h2 + id.y * p.w + id.x] = s;
}

// Polyphase expand, horizontal pass: only the even canvas rows carry data, and
// only the taps that land on an even canvas column are non-zero.
// In: a, p.sw x p.sh (the small image). Out: o, p.w x ((p.h + 1) / 2).
@compute @workgroup_size(16, 16, 1)
fn expand_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let rows = (p.h + 1u) / 2u;
    if (id.x >= p.w || id.y >= rows) {
        return;
    }
    let n = i32(p.w);
    let xi = i32(id.x);
    let src = id.z * p.sw * p.sh + id.y * p.sw;
    var s = 0.0;
    var have = false;
    for (var t = 0; t < 5; t = t + 1) {
        let c = reflect_idx(xi + t - 2, n);
        if ((c & 1u) == 0u) {
            let term = tap(t) * a[src + c / 2u];
            if (have) {
                s = s + term;
            } else {
                s = term;
                have = true;
            }
        }
    }
    o[id.z * p.w * rows + id.y * p.w + id.x] = s;
}

// Polyphase expand, vertical pass, times 4, combined with the base image:
// o = a - expanded (mode 0) or a + expanded (mode 1).
// In: a, p.w x p.h (the base), b (the horizontal pass). Out: o, p.w x p.h.
@compute @workgroup_size(16, 16, 1)
fn expand_v(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let rows = (p.h + 1u) / 2u;
    let n = i32(p.h);
    let yi = i32(id.y);
    var s = 0.0;
    var have = false;
    for (var t = 0; t < 5; t = t + 1) {
        let r = reflect_idx(yi + t - 2, n);
        if ((r & 1u) == 0u) {
            let term = tap(t) * b[id.z * p.w * rows + (r / 2u) * p.w + id.x];
            if (have) {
                s = s + term;
            } else {
                s = term;
                have = true;
            }
        }
    }
    let up = s * 4.0;
    let idx = id.z * p.w * p.h + id.y * p.w + id.x;
    if (p.mode == 0u) {
        o[idx] = a[idx] - up;
    } else {
        o[idx] = a[idx] + up;
    }
}

// Selection energy of one pixel of the band in `a`.
fn energy_at(i: u32, n: u32) -> f32 {
    let l = a[i];
    var e = l * l;
    if (p.use_color != 0u) {
        let ca = a[n + i];
        let cb = a[2u * n + i];
        e = e + ca * ca + cb * cb;
    }
    return e;
}

// Fold one frame's band into the fused band: the pixel is taken when its energy
// is strictly greater than the best so far (or when this is the first frame), so
// the first frame wins ties. All three planes come from the same frame.
@compute @workgroup_size(16, 16, 1)
fn fold(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let n = p.w * p.h;
    let i = id.y * p.w + id.x;
    var e = 0.0;
    if ((p.mode & 1u) != 0u) {
        for (var dr = -1; dr <= 1; dr = dr + 1) {
            for (var dc = -1; dc <= 1; dc = dc + 1) {
                let nr = i32(id.y) + dr;
                let nc = i32(id.x) + dc;
                if (nr >= 0 && nr < i32(p.h) && nc >= 0 && nc < i32(p.w)) {
                    e = e + energy_at(u32(nr) * p.w + u32(nc), n);
                }
            }
        }
    } else {
        e = energy_at(i, n);
    }
    if (g.first != 0u || e > best[i]) {
        best[i] = e;
        o[i] = a[i];
        o[n + i] = a[n + i];
        o[2u * n + i] = a[2u * n + i];
    }
}

// Running sum of the base band, in frame order (the CPU sums in the same order).
@compute @workgroup_size(16, 16, 1)
fn accum(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let idx = id.z * p.w * p.h + id.y * p.w + id.x;
    if (g.first != 0u) {
        o[idx] = a[idx];
    } else {
        o[idx] = o[idx] + a[idx];
    }
}

// The averaged base: sum / number of frames.
@compute @workgroup_size(16, 16, 1)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let idx = id.z * p.w * p.h + id.y * p.w + id.x;
    o[idx] = o[idx] / f32(g.n_frames);
}
