// GPU port of `relief::multigrid::MultigridLayer::relax`
// (crates/stacker-algo/src/relief/multigrid.rs): one Jacobi-style relaxation
// sweep over a single multigrid layer.
//
//   d[i] = b[i] > 0  ? c[i]
//                    : mean of a[] over the up-to-8 in-bounds 3x3 neighbours
//   a[i] = d[i]   (the CPU reference copies d back into a after the sweep)
//
// Matches the CPU reference's exact neighbour-count normalisation (dividing
// by however many of the up to 8 neighbours are actually in-bounds, not a
// fixed 8 — pixels on the layer's border therefore average fewer terms,
// exactly like the CPU loop's `count` accumulator).

@group(0) @binding(0) var<storage, read> a_in: array<f32>;
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
@group(0) @binding(2) var<storage, read> c_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> a_out: array<f32>;

struct Uniforms {
    width: u32,
    height: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let w = i32(params.width);
    let h = i32(params.height);
    if (x >= w || y >= h) { return; }

    let idx = u32(y * w + x);

    var result: f32;
    if (b_in[idx] > 0.0) {
        result = c_in[idx];
    } else {
        var sum: f32 = 0.0;
        var count: f32 = 0.0;
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nx = x + dx;
                let ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    sum = sum + a_in[u32(ny * w + nx)];
                    count = count + 1.0;
                }
            }
        }
        result = sum / count;
    }

    let blended = b_in[idx] * c_in[idx] + (1.0 - b_in[idx]) * result;
    a_out[idx] = blended;
}
