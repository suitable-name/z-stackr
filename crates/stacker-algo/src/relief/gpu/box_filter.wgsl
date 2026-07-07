// GPU port of `relief::guided::box_filter` (crates/stacker-algo/src/relief/guided.rs):
// a `(2*radius+1)^2` window mean with the CPU's exact edge semantics — the
// window is CLIPPED to the image bounds and normalised by the number of
// taps actually inside it (`count = (x_max-x_min+1)*(y_max-y_min+1)` in the
// CPU code), NOT edge-replicated with a constant divisor. Because the
// clipped count factorises as `count_x * count_y`, the two normalised
// separable passes below (each dividing by its own axis' valid-tap count)
// compose to exactly the CPU's normalisation.
//
// The CPU reference uses an f64 summed-area table for numerical stability
// (see that function's doc comment). The direct O(radius) *separable*
// two-pass window sum below (horizontal pass then vertical pass, each a
// plain per-pixel accumulation over at most `2*radius+1` taps) is a
// different accumulation order/precision — output is tolerance-equal (not
// bit-equal) to the CPU path, matching every other GPU module in this
// workspace, and f32 accumulation over a window this size (`radius` is
// settings-clamped well under 100) does not lose enough precision to fail
// the ~1e-3 max-abs-diff parity tolerance. `mode` selects horizontal (0)
// or vertical (1) so one shader module serves both passes.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba32float, write>;

struct Uniforms {
    width: u32,
    height: u32,
    radius: u32,
    // 0 = horizontal pass, 1 = vertical pass.
    mode: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    if (x >= i32(params.width) || y >= i32(params.height)) { return; }

    let r = i32(params.radius);
    var sum: f32 = 0.0;
    var count: f32 = 0.0;
    if (params.mode == 0u) {
        // Clipped window: out-of-bounds taps are skipped (and not counted),
        // matching the CPU SAT's x_min/x_max clamping + actual-count divide.
        for (var dx: i32 = -r; dx <= r; dx++) {
            let sx = x + dx;
            if (sx >= 0 && sx < i32(params.width)) {
                sum = sum + textureLoad(src_texture, vec2<i32>(sx, y), 0).r;
                count = count + 1.0;
            }
        }
    } else {
        for (var dy: i32 = -r; dy <= r; dy++) {
            let sy = y + dy;
            if (sy >= 0 && sy < i32(params.height)) {
                sum = sum + textureLoad(src_texture, vec2<i32>(x, sy), 0).r;
                count = count + 1.0;
            }
        }
    }
    textureStore(dst_texture, vec2<i32>(x, y), vec4<f32>(sum / count, 0.0, 0.0, 1.0));
}
