//! Exactness tests of the fast kernels: each must equal the plain reference.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "test fixtures: small indices and hashed bit patterns"
)]

use stacker_core::{
    color::{encode_row_u16, linear_to_srgb_u16, srgb_to_linear},
    image::PlanarImage,
    memory::paste_interior,
};

/// `SplitMix64` step: a deterministic stream of 64-bit values.
const fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The per-pixel reference `paste_interior` replaced.
#[allow(clippy::too_many_arguments, reason = "mirrors paste_interior")]
fn paste_reference(
    output: &mut PlanarImage<f32>,
    tile: &PlanarImage<f32>,
    ix: usize,
    iy: usize,
    out_x: usize,
    out_y: usize,
    out_w: usize,
    out_h: usize,
) {
    for oy in 0..out_h {
        for ox in 0..out_w {
            let src = (iy + oy) * tile.width + ix + ox;
            let dst = (out_y + oy) * output.width + out_x + ox;
            output.luma[dst] = tile.luma[src];
            output.chroma_a[dst] = tile.chroma_a[src];
            output.chroma_b[dst] = tile.chroma_b[src];
        }
    }
}

fn random_image(w: usize, h: usize, state: &mut u64) -> PlanarImage<f32> {
    let mut img = PlanarImage::<f32>::new(w, h);
    for i in 0..w * h {
        img.luma[i] = (next(state) >> 40) as f32;
        img.chroma_a[i] = (next(state) >> 40) as f32 - 1.0e6;
        img.chroma_b[i] = (next(state) >> 40) as f32 * 0.5;
    }
    img
}

fn same_bits(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> bool {
    let eq = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
    };
    a.width == b.width
        && a.height == b.height
        && eq(&a.luma, &b.luma)
        && eq(&a.chroma_a, &b.chroma_a)
        && eq(&a.chroma_b, &b.chroma_b)
}

#[test]
fn paste_interior_equals_per_pixel_reference() {
    let mut state = 7;
    for round in 0..60 {
        let dest_cols = 1 + (next(&mut state) % 90) as usize;
        let dest_rows = 1 + (next(&mut state) % 70) as usize;
        let tile_w = 1 + (next(&mut state) % 90) as usize;
        let tile_h = 1 + (next(&mut state) % 70) as usize;
        // A rectangle that fits in both images, at random offsets (including
        // flush against the right and bottom edges and empty ones).
        let max_w = tile_w.min(dest_cols);
        let max_h = tile_h.min(dest_rows);
        let out_w = (next(&mut state) as usize) % (max_w + 1);
        let out_h = (next(&mut state) as usize) % (max_h + 1);
        let ix = (next(&mut state) as usize) % (tile_w - out_w + 1);
        let iy = (next(&mut state) as usize) % (tile_h - out_h + 1);
        let (out_x, out_y) = if round % 3 == 0 {
            (dest_cols - out_w, dest_rows - out_h)
        } else {
            (
                (next(&mut state) as usize) % (dest_cols - out_w + 1),
                (next(&mut state) as usize) % (dest_rows - out_h + 1),
            )
        };
        let tile = random_image(tile_w, tile_h, &mut state);
        let base = random_image(dest_cols, dest_rows, &mut state);
        let mut fast = base.clone();
        let mut slow = base;
        paste_interior(&mut fast, &tile, ix, iy, out_x, out_y, out_w, out_h);
        paste_reference(&mut slow, &tile, ix, iy, out_x, out_y, out_w, out_h);
        assert!(same_bits(&fast, &slow), "round {round}");
    }
}

#[test]
fn paste_interior_with_zero_width_changes_nothing() {
    let mut state = 3;
    let tile = random_image(8, 8, &mut state);
    let base = random_image(8, 8, &mut state);
    let mut out = base.clone();
    paste_interior(&mut out, &tile, 2, 2, 1, 1, 0, 5);
    assert!(same_bits(&out, &base));
}

fn check_inputs(inputs: &[f32]) {
    let mut fast = vec![0_u16; inputs.len()];
    encode_row_u16(inputs, &mut fast);
    for (i, (&v, &code)) in inputs.iter().zip(&fast).enumerate() {
        assert_eq!(
            code,
            linear_to_srgb_u16(v),
            "input #{i} = {v:e} (bits {:#010x})",
            v.to_bits()
        );
    }
}

/// Around every one of the 65 536 code boundaries (the decode of the code
/// midpoints) and on the f32 neighbours of each.
#[test]
fn encode_u16_equals_scalar_at_every_code_boundary() {
    let mut inputs = Vec::new();
    for code in 0..=u32::from(u16::MAX) {
        for half in [-0.5_f32, 0.5] {
            let encoded = (code as f32 + half) / 65_535.0;
            let centre = srgb_to_linear(encoded.clamp(0.0, 1.0));
            let bits = centre.to_bits();
            for delta in -40_i32..=40 {
                let near = f32::from_bits(bits.wrapping_add_signed(delta));
                inputs.push(near);
            }
        }
        // The code's own decode point and its immediate neighbours.
        let own = srgb_to_linear(code as f32 / 65_535.0);
        let bits = own.to_bits();
        for delta in -3_i32..=3 {
            inputs.push(f32::from_bits(bits.wrapping_add_signed(delta)));
        }
    }
    check_inputs(&inputs);
}

#[test]
fn encode_u16_equals_scalar_on_special_values() {
    let inputs = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        1.000_000_1,
        0.999_999_94,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        1.0e-45,
        0.003_130_8,
        0.003_130_809,
        0.003_130_791,
        0.5,
    ];
    check_inputs(&inputs);
}

#[test]
fn encode_u16_equals_scalar_on_dense_random_samples() {
    let mut state = 99;
    let mut inputs = Vec::with_capacity(3_000_000);
    // Uniform in [0, 1], including the dark end where cells span many codes.
    for _ in 0..1_000_000 {
        inputs.push((next(&mut state) >> 40) as f32 / 16_777_216.0);
    }
    // Log-uniform over the dark range.
    for _ in 0..500_000 {
        let u = (next(&mut state) >> 40) as f32 / 16_777_216.0;
        inputs.push(10.0_f32.powf(-6.0 * u));
    }
    // Arbitrary bit patterns (negatives, > 1, NaN, inf, subnormals).
    for _ in 0..1_500_000 {
        inputs.push(f32::from_bits(next(&mut state) as u32));
    }
    check_inputs(&inputs);
}

/// Rows of every length around the work-chunk sizes go through the same path.
#[test]
fn encode_row_u16_handles_odd_lengths() {
    let mut state = 5;
    for len in [0, 1, 2, 3, 7, 16, 17, 255, 1001] {
        let inputs: Vec<f32> = (0..len)
            .map(|_| (next(&mut state) >> 40) as f32 / 16_777_216.0)
            .collect();
        check_inputs(&inputs);
    }
}
