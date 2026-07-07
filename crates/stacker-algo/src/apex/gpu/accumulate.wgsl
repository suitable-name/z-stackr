// GPU port of `apex::fuse::ApexAccumulator::blend`'s per-level update rule
// (crates/stacker-algo/src/apex/fuse.rs). Unlike `fuse.wgsl` (which fuses N
// source layers into one output in a single dispatch), this shader updates
// a single RESIDENT accumulator texture with exactly one new frame per
// dispatch — the whole point of the GPU-resident incremental accumulator
// (`apex::gpu::accumulator`) is that the accumulator texture never leaves
// the GPU between frames, only the newly-uploaded per-frame level texture
// does.
//
//   mode 0 (average)               — residual/last level: running mean.
//                                     acc_new = (acc_old*count + src) / (count+1)
//   mode 1 (per-pixel argmax)      — every level except the residual and
//                                     the finest-with-grit-suppression
//                                     level: replace the accumulator pixel
//                                     with the source pixel when the
//                                     source's per-pixel energy STRICTLY
//                                     exceeds the accumulator's (ties keep
//                                     the accumulator — matches the CPU's
//                                     `src_e > tgt_e`, the OPPOSITE tie rule
//                                     from the batch `fuse.wgsl`'s `>=`,
//                                     since the CPU accumulator's own
//                                     "earliest frame wins" semantics differ
//                                     from the batch fuser's "last maximum
//                                     wins").
//   mode 2 (neighbourhood argmax)  — finest level (index 0) when
//                                     `grit_suppression` is enabled: same
//                                     rule but energy is a 3x3-neighbourhood
//                                     sum, strict `>` tie-break.
//
// Energy = luma^2 (+ chroma^2 * 2 when `use_color` is set), exactly matching
// `apex::fuse::pixel_energy`.

@group(0) @binding(0) var acc_texture: texture_2d<f32>;
@group(0) @binding(1) var src_texture: texture_2d<f32>;
@group(0) @binding(2) var dst_texture: texture_storage_2d<rgba32float, write>;

// No trailing padding field here, deliberately: WGSL requires a uniform
// address-space `array<T, N>`'s element stride to be rounded up to 16
// bytes (the std140-style rule), so an `array<u32, 3>` padding field would
// actually occupy 48 bytes (16 * 3), not the 12 the Rust `Uniforms` struct
// (`_padding: [u32; 3]`) assumes — and a `vec3<u32>` field has the same
// problem via its own 16-byte alignment. Rather than fight WGSL's uniform
// layout rules for a field whose only purpose is padding, this struct just
// omits it: `wgpu` only requires the bound buffer to be AT LEAST as large
// as the shader's struct (`min_binding_size`), never exactly equal, so the
// Rust-side struct staying 32 bytes while this one is naturally 20 bytes
// (5 plain `u32` fields, all align 4, no implicit padding) is fine — this
// is the same shape `apex::gpu::fuse.wgsl`'s `Uniforms` already uses
// successfully opposite its own padded Rust struct.
struct Uniforms {
    width: u32,
    height: u32,
    use_color: u32,
    // 0 = average, 1 = per-pixel argmax, 2 = neighbourhood (3x3) argmax.
    mode: u32,
    // Number of frames already blended into `acc_texture` BEFORE this
    // dispatch (mode 0 only).
    count: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

fn energy_of(pixel: vec4<f32>) -> f32 {
    var e = pixel.r * pixel.r;
    if (params.use_color == 1u) {
        e = e + (pixel.g * pixel.g) + (pixel.b * pixel.b);
    }
    return e;
}

fn acc_energy(coords: vec2<i32>) -> f32 {
    return energy_of(textureLoad(acc_texture, coords, 0));
}

fn src_energy(coords: vec2<i32>) -> f32 {
    return energy_of(textureLoad(src_texture, coords, 0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    if (x >= i32(params.width) || y >= i32(params.height)) { return; }
    let coords = vec2<i32>(x, y);

    let acc_pixel = textureLoad(acc_texture, coords, 0);
    let src_pixel = textureLoad(src_texture, coords, 0);

    var out_pixel: vec4<f32>;
    if (params.mode == 0u) {
        let n = f32(params.count);
        out_pixel = (acc_pixel * n + src_pixel) / (n + 1.0);
    } else if (params.mode == 2u) {
        let w = i32(params.width);
        let h = i32(params.height);
        var tgt_e: f32 = 0.0;
        var s_e: f32 = 0.0;
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nx = x + dx;
                let ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    let ncoords = vec2<i32>(nx, ny);
                    tgt_e = tgt_e + acc_energy(ncoords);
                    s_e = s_e + src_energy(ncoords);
                }
            }
        }
        // Strict >: ties keep the accumulator (earliest frame wins),
        // matching the CPU accumulator's tie rule exactly.
        out_pixel = select(acc_pixel, src_pixel, s_e > tgt_e);
    } else {
        let tgt_e = energy_of(acc_pixel);
        let s_e = energy_of(src_pixel);
        out_pixel = select(acc_pixel, src_pixel, s_e > tgt_e);
    }

    textureStore(dst_texture, coords, out_pixel);
}
