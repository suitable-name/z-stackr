// Relief on the GPU: focus measure per frame folded into the best frame per
// pixel, selection against the threshold, guided-filter smoothing of the luma.
// Every entry point gets the bind-group layout of the bindings it uses.
//
// Images are planes one after the other in a storage buffer:
// element = plane * (w * h) + y * w + x. Grids are 16 x 16 workgroups over (x, y).
//
// The focus measure is that of `stacker_algo::relief::focus`: the residual
// variance of a least-squares plane over the window of half-size r around the
// pixel (clamped to the image), divided by n - 1, 0 for a window of fewer than
// two pixels or a single row or column. The CPU evaluates it from f64
// summed-area tables; here the window sums are two separable passes in f32,
// over coordinates relative to the pixel and values shifted by the frame's mean
// (the measure does not change under either shift, and both keep the sums
// small). The window geometry is exact integer arithmetic.
//
// The guided filter is `stacker_algo::relief::guided` in its self-guided form
// (guidance = input): box means of I and I^2, a = var / (var + eps),
// b = mean - a * mean, box means of a and b, q = mean_a * I + mean_b.

struct Params {
    w: u32,
    h: u32,
    // focus: the window half-size; guided: the smoothing radius.
    radius: u32,
    n_frames: u32,
    // 1 for the first frame of a stack: it takes every pixel.
    first: u32,
    // focus: subtracted from every luma value.
    shift: f32,
    // select: pixels whose best focus is >= this take the best frame.
    threshold: f32,
    // guided: the regularisation.
    eps: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
// The current frame: luma, chroma_a, chroma_b.
@group(0) @binding(1) var<storage, read> frame: array<f32>;
// Horizontal window sums (three planes).
@group(0) @binding(2) var<storage, read_write> tmp: array<f32>;
// The best focus so far (one plane).
@group(0) @binding(3) var<storage, read_write> best_focus: array<f32>;
// The three planes of the frame that has the best focus so far.
@group(0) @binding(4) var<storage, read_write> best_lab: array<f32>;
// The running sum of the frames (three planes); later a, b and the result.
@group(0) @binding(5) var<storage, read_write> sum_lab: array<f32>;
// The selected image (three planes).
@group(0) @binding(6) var<storage, read_write> selected: array<f32>;

// Σ_{i=0}^{k} i², exact for the radii the plan accepts (at most 100).
fn sum_sq_to(k: i32) -> i32 {
    return k * (k + 1) * (2 * k + 1) / 6;
}

// ── Focus measure and fold ───────────────────────────────────────────────────

// Horizontal pass: over the clamped columns of the window around (x, y), the
// sums of v, v^2 and (column - x) * v, with v = luma - shift.
@compute @workgroup_size(16, 16, 1)
fn focus_h(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let r = i32(p.radius);
    let x = i32(id.x);
    let x0 = max(x - r, 0);
    let x1 = min(x + r, i32(p.w) - 1);
    let row = id.y * p.w;
    var s0 = 0.0;
    var s2 = 0.0;
    var sx = 0.0;
    for (var c = x0; c <= x1; c = c + 1) {
        let v = frame[row + u32(c)] - p.shift;
        s0 = s0 + v;
        s2 = s2 + v * v;
        sx = sx + f32(c - x) * v;
    }
    let n = p.w * p.h;
    let i = row + id.x;
    tmp[i] = s0;
    tmp[n + i] = s2;
    tmp[2u * n + i] = sx;
}

// Vertical pass, the measure, and the fold of this frame into the best frame
// (taken when its focus is strictly greater, so the first frame wins ties) and
// into the running sum.
@compute @workgroup_size(16, 16, 1)
fn focus_v_fold(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let r = i32(p.radius);
    let x = i32(id.x);
    let y = i32(id.y);
    let n = p.w * p.h;
    let y0 = max(y - r, 0);
    let y1 = min(y + r, i32(p.h) - 1);
    var s0 = 0.0;
    var s2 = 0.0;
    var sx = 0.0;
    var sy = 0.0;
    for (var row = y0; row <= y1; row = row + 1) {
        let j = u32(row) * p.w + id.x;
        let h0 = tmp[j];
        s0 = s0 + h0;
        s2 = s2 + tmp[n + j];
        sx = sx + tmp[2u * n + j];
        sy = sy + f32(row - y) * h0;
    }

    // Window geometry relative to the pixel: columns a..=b, rows c..=d, each
    // range containing 0.
    let a = max(x - r, 0) - x;
    let b = min(x + r, i32(p.w) - 1) - x;
    let c = y0 - y;
    let d = y1 - y;
    let nx = b - a + 1;
    let ny = d - c + 1;
    let count = nx * ny;
    var focus = 0.0;
    // Σi and Σi² over a..=b (a <= 0 <= b), and the unnormalised variance of the
    // offsets times nx: nx Σi² - (Σi)².
    let sum_x = (b * (b + 1) - (-a) * (-a + 1)) / 2;
    let num_x = nx * (sum_sq_to(-a) + sum_sq_to(b)) - sum_x * sum_x;
    let sum_y = (d * (d + 1) - (-c) * (-c + 1)) / 2;
    let num_y = ny * (sum_sq_to(-c) + sum_sq_to(d)) - sum_y * sum_y;
    if (count >= 2 && num_x > 0 && num_y > 0) {
        let fn_ = f32(count);
        let var_x = f32(ny) * f32(num_x) / f32(nx);
        let var_y = f32(nx) * f32(num_y) / f32(ny);
        let covar_x = sx - (f32(sum_x) / f32(nx)) * s0;
        let covar_y = sy - (f32(sum_y) / f32(ny)) * s0;
        let residual = s2 - covar_x * covar_x / var_x - covar_y * covar_y / var_y;
        focus = max((residual - s0 * s0 / fn_) / (fn_ - 1.0), 0.0);
    }

    let i = id.y * p.w + id.x;
    if (p.first != 0u || focus > best_focus[i]) {
        best_focus[i] = focus;
        best_lab[i] = frame[i];
        best_lab[n + i] = frame[n + i];
        best_lab[2u * n + i] = frame[2u * n + i];
    }
    for (var k = 0u; k < 3u; k = k + 1u) {
        let e = k * n + i;
        if (p.first != 0u) {
            sum_lab[e] = frame[e];
        } else {
            sum_lab[e] = sum_lab[e] + frame[e];
        }
    }
}

// ── Selection ────────────────────────────────────────────────────────────────

// Above the threshold the sharpest frame, below it the mean of the frames.
@compute @workgroup_size(16, 16, 1)
fn select_frame(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let n = p.w * p.h;
    let i = id.y * p.w + id.x;
    let take_best = best_focus[i] >= p.threshold;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let e = k * n + i;
        if (take_best) {
            selected[e] = best_lab[e];
        } else {
            selected[e] = sum_lab[e] / f32(p.n_frames);
        }
    }
}

// ── Guided filter ────────────────────────────────────────────────────────────

// Horizontal box sums of I and I^2 (I = the selected luma) into tmp 0 and 1.
@compute @workgroup_size(16, 16, 1)
fn box_h_i(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let r = i32(p.radius);
    let x = i32(id.x);
    let row = id.y * p.w;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var c = max(x - r, 0); c <= min(x + r, i32(p.w) - 1); c = c + 1) {
        let v = selected[row + u32(c)];
        s1 = s1 + v;
        s2 = s2 + v * v;
    }
    let n = p.w * p.h;
    tmp[row + id.x] = s1;
    tmp[n + row + id.x] = s2;
}

// Horizontal box sums of a and b (sum_lab 0 and 1) into tmp 0 and 1.
@compute @workgroup_size(16, 16, 1)
fn box_h_ab(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let r = i32(p.radius);
    let x = i32(id.x);
    let row = id.y * p.w;
    let n = p.w * p.h;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var c = max(x - r, 0); c <= min(x + r, i32(p.w) - 1); c = c + 1) {
        s1 = s1 + sum_lab[row + u32(c)];
        s2 = s2 + sum_lab[n + row + u32(c)];
    }
    tmp[row + id.x] = s1;
    tmp[n + row + id.x] = s2;
}

// The vertical box sums of tmp 0 and 1 at (x, y), divided by the window size:
// the two box means.
fn box_means(id: vec3<u32>) -> vec2<f32> {
    let r = i32(p.radius);
    let x = i32(id.x);
    let y = i32(id.y);
    let n = p.w * p.h;
    let y0 = max(y - r, 0);
    let y1 = min(y + r, i32(p.h) - 1);
    var s1 = 0.0;
    var s2 = 0.0;
    for (var row = y0; row <= y1; row = row + 1) {
        let j = u32(row) * p.w + id.x;
        s1 = s1 + tmp[j];
        s2 = s2 + tmp[n + j];
    }
    let cols = min(x + r, i32(p.w) - 1) - max(x - r, 0) + 1;
    let count = f32(cols * (y1 - y0 + 1));
    return vec2<f32>(s1 / count, s2 / count);
}

// a and b of the guided filter into sum_lab 0 and 1.
@compute @workgroup_size(16, 16, 1)
fn guided_ab(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let m = box_means(id);
    let variance = m.y - m.x * m.x;
    let a = variance / (variance + p.eps);
    let i = id.y * p.w + id.x;
    sum_lab[i] = a;
    sum_lab[p.w * p.h + i] = m.x - a * m.x;
}

// The smoothed luma, q = mean(a) * I + mean(b), into sum_lab 2.
@compute @workgroup_size(16, 16, 1)
fn guided_q(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.w || id.y >= p.h) {
        return;
    }
    let m = box_means(id);
    let i = id.y * p.w + id.x;
    sum_lab[2u * p.w * p.h + i] = m.x * selected[i] + m.y;
}
