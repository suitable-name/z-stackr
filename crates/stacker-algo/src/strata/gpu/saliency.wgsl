// GPU port of the 4-neighbour Laplacian-magnitude step in
// `strata::saliency::compute_saliency` (crates/stacker-algo/src/strata/saliency.rs):
//
//   h[y,x] = |up + down + left + right - 4 * center|
//
// with a clamped (edge-replicated) boundary, matching the CPU reference's
// `saturating_sub`/`.min(width|height - 1)` clamp exactly. Only luma is read
// (Strata's saliency metric is luma-only — see the module docs' rationale).
//
// This shader computes ONLY that single per-pixel pass; the five
// fixed-kernel blur passes that follow it in `compute_saliency` stay on the
// CPU (`apex::pyramid::apply_gaussian_blur`, reused as-is) — porting a
// five-times-repeated small separable convolution would add another shader
// and another upload/readback round trip for comparatively little of the
// total saliency-stage cost, whereas the Laplacian pass below is the single
// per-pixel-neighbourhood computation the whole pipeline can skip entirely
// on the CPU once this shader has produced `h`.

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba32float, write>;

struct Uniforms {
    width: u32,
    height: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

fn get_luma(x: i32, y: i32) -> f32 {
    let px = clamp(x, 0, i32(params.width) - 1);
    let py = clamp(y, 0, i32(params.height) - 1);
    return textureLoad(src_texture, vec2<i32>(px, py), 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    if (x >= i32(params.width) || y >= i32(params.height)) { return; }

    let center = get_luma(x, y);
    let up = get_luma(x, y - 1);
    let down = get_luma(x, y + 1);
    let left = get_luma(x - 1, y);
    let right = get_luma(x + 1, y);

    let h = abs(up + down + left + right - 4.0 * center);
    textureStore(dst_texture, vec2<i32>(x, y), vec4<f32>(h, 0.0, 0.0, 1.0));
}
