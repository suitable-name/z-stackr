// Small elementwise-arithmetic shader shared by the three non-box-filter
// steps of the fused guided-filter GPU pipeline
// (`relief::gpu::guided_filter_gpu`, crates/stacker-algo/src/relief/gpu/mod.rs):
// mirroring the elementwise loops in the CPU reference
// (`relief::guided::guided_filter`, crates/stacker-algo/src/relief/guided.rs)
// exactly, just in f32 instead of that function's `a`/`b` scratch buffers
// (which are themselves already f32 — only the box-filter SAT accumulation
// is f64 on the CPU side).
//
// Four textures are always bound (`in0`..`in3`) even though a given mode
// only reads some of them, and two storage textures are always bound
// (`out0`, `out1`) even though a given mode only writes one — this lets one
// shader module/pipeline/bind-group-layout serve every elementwise step of
// the fused pipeline instead of three near-identical shader files, at the
// cost of a few unused bindings per dispatch (the same "spare slot" pattern
// `fuse.wgsl` and `box_filter.wgsl` already use).
//
//   mode 0 (products)  — out0 = in0 * in1 (I * p), out1 = in0 * in0 (I * I).
//                        Used for `i_p`/`i_i` in the CPU reference.
//   mode 1 (coeffs)    — in0 = mean_I, in1 = mean_p, in2 = mean_Ip, in3 = mean_Ii.
//                        cov_Ip = mean_Ip - mean_I*mean_p
//                        var_I  = max(mean_Ii - mean_I*mean_I, 0.0)   <- CPU's clamp, preserved exactly
//                        out0 (a) = cov_Ip / (var_I + eps)
//                        out1 (b) = mean_p - a * mean_I
//                        Used for `a`/`b` in the CPU reference.
//   mode 2 (final)     — in0 = mean_a, in1 = mean_b, in2 = I (guidance luma).
//                        out0 (q) = mean_a * I + mean_b.
//                        Used for the final `q` in the CPU reference.

@group(0) @binding(0) var in0: texture_2d<f32>;
@group(0) @binding(1) var in1: texture_2d<f32>;
@group(0) @binding(2) var in2: texture_2d<f32>;
@group(0) @binding(3) var in3: texture_2d<f32>;
@group(0) @binding(4) var out0: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var out1: texture_storage_2d<rgba32float, write>;

struct Uniforms {
    width: u32,
    height: u32,
    // 0 = products (I*p, I*I), 1 = guided-filter a/b coefficients, 2 = final q.
    mode: u32,
    _padding: u32,
    eps: f32,
};
@group(0) @binding(6) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    if (x >= i32(params.width) || y >= i32(params.height)) { return; }
    let coords = vec2<i32>(x, y);

    if (params.mode == 0u) {
        let i_val = textureLoad(in0, coords, 0).r;
        let p_val = textureLoad(in1, coords, 0).r;
        textureStore(out0, coords, vec4<f32>(i_val * p_val, 0.0, 0.0, 1.0));
        textureStore(out1, coords, vec4<f32>(i_val * i_val, 0.0, 0.0, 1.0));
    } else if (params.mode == 1u) {
        let mean_i = textureLoad(in0, coords, 0).r;
        let mean_p = textureLoad(in1, coords, 0).r;
        let mean_ip = textureLoad(in2, coords, 0).r;
        let mean_ii = textureLoad(in3, coords, 0).r;

        let cov_ip = mean_ip - mean_i * mean_p;
        // Clamp to >= 0, exactly mirroring the CPU's `.max(0.0)` guard
        // against subtractive-cancellation noise for near-constant windows.
        let var_i = max(mean_ii - mean_i * mean_i, 0.0);

        let a = cov_ip / (var_i + params.eps);
        let b = mean_p - a * mean_i;
        textureStore(out0, coords, vec4<f32>(a, 0.0, 0.0, 1.0));
        textureStore(out1, coords, vec4<f32>(b, 0.0, 0.0, 1.0));
    } else {
        let mean_a = textureLoad(in0, coords, 0).r;
        let mean_b = textureLoad(in1, coords, 0).r;
        let guidance_i = textureLoad(in2, coords, 0).r;
        let q = mean_a * guidance_i + mean_b;
        textureStore(out0, coords, vec4<f32>(q, 0.0, 0.0, 1.0));
    }
}
