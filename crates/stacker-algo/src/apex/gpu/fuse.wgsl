// GPU port of `apex::fuse::fuse_pyramids`'s per-level blend semantics
// (crates/stacker-algo/src/apex/fuse.rs). Unlike the previous version of
// this shader (which applied 3x3-neighbourhood winner-take-all to EVERY
// level, which does not match the CPU reference), this shader is
// per-dispatch mode-selected from the Rust side so GPU output matches CPU
// semantics level-by-level:
//
//   mode 0 (average)               — base/last level: per-pixel mean across
//                                     all source layers.
//   mode 1 (per-pixel argmax)       — every level except the base and the
//                                     finest-with-grit-suppression level:
//                                     per-pixel energy argmax, ties = last
//                                     maximum wins (matches the CPU `>=`
//                                     accumulation order).
//   mode 2 (neighbourhood argmax)   — finest level (index 0) when
//                                     `grit_suppression` is enabled: 3x3
//                                     neighbourhood energy argmax, ties =
//                                     last maximum wins.
//
// Energy = luma^2 (+ chroma^2 * 2 when `use_color` is set), exactly matching
// `apex::fuse::pixel_energy`.

@group(0) @binding(0) var src_pyramids: texture_2d_array<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba32float, write>;

struct Uniforms {
    layer_count: u32,
    width: u32,
    height: u32,
    use_color: u32,
    // 0 = average, 1 = per-pixel argmax, 2 = neighbourhood (3x3) argmax.
    mode: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

fn pixel_energy(layer: u32, coords: vec2<i32>) -> f32 {
    let pixel = textureLoad(src_pyramids, coords, layer, 0);
    var energy = pixel.r * pixel.r; // luma squared
    if (params.use_color == 1u) {
        energy = energy + (pixel.g * pixel.g) + (pixel.b * pixel.b);
    }
    return energy;
}

fn blend_average(coords: vec2<i32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var k: u32 = 0u; k < params.layer_count; k++) {
        sum = sum + textureLoad(src_pyramids, coords, k, 0);
    }
    let n = f32(params.layer_count);
    // Alpha is not meaningful data (always written as 1.0 on upload); average
    // it too for simplicity, matching the CPU's non-existence of an alpha
    // channel (only luma/chroma_a/chroma_b are ever read back).
    return sum / n;
}

fn blend_argmax_per_pixel(coords: vec2<i32>) -> vec4<f32> {
    var best_energy: f32 = -1.0e30;
    var winner_layer: u32 = 0u;
    for (var k: u32 = 0u; k < params.layer_count; k++) {
        let e = pixel_energy(k, coords);
        // `>=`: last maximum wins, matching `apex::fuse::blend_max_contrast`.
        if (e >= best_energy) {
            best_energy = e;
            winner_layer = k;
        }
    }
    return textureLoad(src_pyramids, coords, winner_layer, 0);
}

fn blend_argmax_neighborhood(coords: vec2<i32>) -> vec4<f32> {
    let w = i32(params.width);
    let h = i32(params.height);
    var best_energy: f32 = -1.0e30;
    var winner_layer: u32 = 0u;
    for (var k: u32 = 0u; k < params.layer_count; k++) {
        var local_energy: f32 = 0.0;
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nx = clamp(coords.x + dx, 0, w - 1);
                let ny = clamp(coords.y + dy, 0, h - 1);
                local_energy = local_energy + pixel_energy(k, vec2<i32>(nx, ny));
            }
        }
        // `>=`: last maximum wins, matching
        // `apex::fuse::blend_max_contrast_neighborhood`.
        if (local_energy >= best_energy) {
            best_energy = local_energy;
            winner_layer = k;
        }
    }
    return textureLoad(src_pyramids, coords, winner_layer, 0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    if (x >= i32(params.width) || y >= i32(params.height)) { return; }
    let coords = vec2<i32>(x, y);

    var out_pixel: vec4<f32>;
    if (params.mode == 0u) {
        out_pixel = blend_average(coords);
    } else if (params.mode == 2u) {
        out_pixel = blend_argmax_neighborhood(coords);
    } else {
        out_pixel = blend_argmax_per_pixel(coords);
    }

    textureStore(dst_texture, coords, out_pixel);
}
