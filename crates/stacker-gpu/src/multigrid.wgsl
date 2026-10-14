// The Relief multigrid depth solver on the GPU: the twin of
// `stacker_algo::relief::multigrid` (see the module documentation there for the
// algorithm; the operations below follow it step by step, sums evaluated left to
// right in the order it names).
//
// All levels of the hierarchy live in one storage buffer `lv`. Level l starts at
// element `off_l` and holds four planes of `w_l * h_l` samples one after the
// other: plane 0 and plane 1 are `a` (the solution) and `d` (the scratch plane of
// the Jacobi sweep) in an order that flips with every sweep (the host tracks it
// and tells each operation which plane is `a`), plane 2 is `b` (the weight), plane
// 3 is `c` (the target).
//
// The coarse levels are tiny and a cycle visits them millions of times, so the
// levels below a size threshold are not dispatched operation by operation: one
// workgroup (`tail`) runs a whole cycle of them from a schedule the host
// prepared, with a barrier between operations.

struct Params {
    // The fine level of the operation: width, height, first element.
    fw: u32,
    fh: u32,
    foff: u32,
    // The coarse level (restrict, prolong): width, height, first element.
    cw: u32,
    ch: u32,
    coff: u32,
    // Which plane (0 or 1) holds `a` on the fine and the coarse level; the other
    // one is `d`.
    fa: u32,
    ca: u32,
    // init: the size of the unpadded image.
    w: u32,
    h: u32,
    // tail: the range of the schedule to run.
    sched_off: u32,
    sched_len: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> lv: array<f32>;
// The winning frame index of every pixel, two u16 per word, low half first.
@group(0) @binding(2) var<storage, read> idx: array<u32>;
// The confidence weight of every pixel.
@group(0) @binding(3) var<storage, read> wgt: array<f32>;
// The operations of a tail cycle: op | level << 2 | fine a plane << 10 |
// coarse a plane << 11, with op 0 = restrict, 1 = prolong, 2 = relax.
@group(0) @binding(4) var<storage, read> sched: array<u32>;
// Per level: width, height, first element, padding.
@group(0) @binding(5) var<storage, read> table: array<u32>;

// ── Operations on explicit geometry (shared by the dispatched and tail paths) ─

// restrict: the coarse sample (cx, cy) from its fine 2 x 2 block, clipped.
fn restrict_one(
    fw: u32, fh: u32, foff: u32,
    cw: u32, ch: u32, coff: u32,
    fa: u32, ca: u32,
    cx: u32, cy: u32,
) {
    let fn_ = fw * fh;
    let cn = cw * ch;
    let fy0 = cy * 2u;
    let fx0 = cx * 2u;
    var rows = 1u;
    if (fy0 + 1u < fh) {
        rows = 2u;
    }
    var cols = 1u;
    if (fx0 + 1u < fw) {
        cols = 2u;
    }
    var sum_a = 0.0;
    var sum_b = 0.0;
    var sum_bc = 0.0;
    var count = 0.0;
    for (var dy = 0u; dy < rows; dy = dy + 1u) {
        for (var dx = 0u; dx < cols; dx = dx + 1u) {
            let i = (fy0 + dy) * fw + fx0 + dx;
            let b = lv[foff + 2u * fn_ + i];
            sum_a = sum_a + lv[foff + fa * fn_ + i];
            sum_b = sum_b + b;
            sum_bc = sum_bc + b * lv[foff + 3u * fn_ + i];
            count = count + 1.0;
        }
    }
    let j = cy * cw + cx;
    lv[coff + ca * cn + j] = sum_a / count;
    lv[coff + 2u * cn + j] = sum_b / count;
    var target_c = 0.0;
    if (sum_b > 0.0) {
        target_c = sum_bc / sum_b;
    }
    lv[coff + 3u * cn + j] = target_c;
}

// prolong: the fine sample (x, y) from the bilinear coarse value, unless its
// coarse sample is fully pinned.
fn prolong_one(
    fw: u32, fh: u32, foff: u32,
    cw: u32, ch: u32, coff: u32,
    fa: u32, ca: u32,
    x: u32, y: u32,
) {
    let fn_ = fw * fh;
    let cn = cw * ch;
    let cx = x / 2u;
    let cy = y / 2u;
    if (lv[coff + 2u * cn + cy * cw + cx] >= 1.0) {
        return;
    }
    let cx_next = min(cx + 1u, cw - 1u);
    let cy_next = min(cy + 1u, ch - 1u);
    let here = lv[coff + ca * cn + cy * cw + cx];
    let odd_x = (x & 1u) == 1u;
    let odd_y = (y & 1u) == 1u;
    var value = here;
    if (odd_x && !odd_y) {
        value = (here + lv[coff + ca * cn + cy * cw + cx_next]) * 0.5;
    } else if (!odd_x && odd_y) {
        value = (here + lv[coff + ca * cn + cy_next * cw + cx]) * 0.5;
    } else if (odd_x && odd_y) {
        let right = lv[coff + ca * cn + cy * cw + cx_next];
        let down = lv[coff + ca * cn + cy_next * cw + cx];
        let down_right = lv[coff + ca * cn + cy_next * cw + cx_next];
        value = ((here + right) + (down + down_right)) * 0.25;
    }
    let i = y * fw + x;
    let b = lv[foff + 2u * fn_ + i];
    let c = lv[foff + 3u * fn_ + i];
    lv[foff + fa * fn_ + i] = b * c + (1.0 - b) * value;
}

// relax: one Jacobi sweep at (x, y), reading plane `fa` and writing the other.
fn relax_one(fw: u32, fh: u32, foff: u32, fa: u32, x: u32, y: u32) {
    let fn_ = fw * fh;
    let i = y * fw + x;
    let b = lv[foff + 2u * fn_ + i];
    let c = lv[foff + 3u * fn_ + i];
    var nv = c;
    if (!(b > 0.0)) {
        let y_min = max(y, 1u) - 1u;
        let y_max = min(y + 1u, fh - 1u);
        let x_min = max(x, 1u) - 1u;
        let x_max = min(x + 1u, fw - 1u);
        var sum = 0.0;
        var count = 0.0;
        for (var ny = y_min; ny <= y_max; ny = ny + 1u) {
            for (var nx = x_min; nx <= x_max; nx = nx + 1u) {
                sum = sum + lv[foff + fa * fn_ + ny * fw + nx];
                count = count + 1.0;
            }
        }
        nv = sum / count;
    }
    lv[foff + (1u - fa) * fn_ + i] = b * c + (1.0 - b) * nv;
}

// ── Dispatched operations ────────────────────────────────────────────────────

// Level 0: the padded field (the last column and row replicated), pinned samples
// get a = target, b = 1, c = target, all others zero.
@compute @workgroup_size(16, 16, 1)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.fw || id.y >= p.fh) {
        return;
    }
    let n = p.fw * p.fh;
    let source = min(id.y, p.h - 1u) * p.w + min(id.x, p.w - 1u);
    let i = id.y * p.fw + id.x;
    var target_v = 0.0;
    var pinned = 0.0;
    if (wgt[source] > 0.0) {
        let word = idx[source >> 1u];
        target_v = f32((word >> ((source & 1u) * 16u)) & 0xffffu);
        pinned = 1.0;
    }
    lv[p.foff + i] = target_v;
    lv[p.foff + 2u * n + i] = pinned;
    lv[p.foff + 3u * n + i] = target_v;
}

@compute @workgroup_size(16, 16, 1)
fn restrict_level(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.cw || id.y >= p.ch) {
        return;
    }
    restrict_one(p.fw, p.fh, p.foff, p.cw, p.ch, p.coff, p.fa, p.ca, id.x, id.y);
}

@compute @workgroup_size(16, 16, 1)
fn prolong_level(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.fw || id.y >= p.fh) {
        return;
    }
    prolong_one(p.fw, p.fh, p.foff, p.cw, p.ch, p.coff, p.fa, p.ca, id.x, id.y);
}

@compute @workgroup_size(16, 16, 1)
fn relax_level(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.fw || id.y >= p.fh) {
        return;
    }
    relax_one(p.fw, p.fh, p.foff, p.fa, id.x, id.y);
}

// ── The tail: a whole cycle of the small levels in one workgroup ─────────────

@compute @workgroup_size(256, 1, 1)
fn tail(@builtin(local_invocation_index) lid: u32) {
    for (var s = 0u; s < p.sched_len; s = s + 1u) {
        let word = sched[p.sched_off + s];
        let op = word & 3u;
        let level = (word >> 2u) & 255u;
        let fa = (word >> 10u) & 1u;
        let ca = (word >> 11u) & 1u;
        let fw = table[4u * level];
        let fh = table[4u * level + 1u];
        let foff = table[4u * level + 2u];
        let cw = table[4u * (level + 1u)];
        let ch = table[4u * (level + 1u) + 1u];
        let coff = table[4u * (level + 1u) + 2u];
        if (op == 0u) {
            let count = cw * ch;
            for (var i = lid; i < count; i = i + 256u) {
                restrict_one(fw, fh, foff, cw, ch, coff, fa, ca, i % cw, i / cw);
            }
        } else if (op == 1u) {
            let count = fw * fh;
            for (var i = lid; i < count; i = i + 256u) {
                prolong_one(fw, fh, foff, cw, ch, coff, fa, ca, i % fw, i / fw);
            }
        } else {
            let count = fw * fh;
            for (var i = lid; i < count; i = i + 256u) {
                relax_one(fw, fh, foff, fa, i % fw, i / fw);
            }
        }
        // Every thread has taken the same turns of this loop (its bound is a
        // uniform), so the barriers are in uniform control flow.
        storageBarrier();
        workgroupBarrier();
    }
}
