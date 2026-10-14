//! Tile invariance and receptive field of [`stacker_nn::model::FocusMergeNet`]
//! (review findings NN-2 and NN-5).
//!
//! The model's normalisation is per-pixel, so a pixel's output depends only on the
//! inputs within `R = 2·Σ dilations + 10` pixels. These tests pin that:
//!
//! * the radius table the README prints, from the single `receptive_field_radius`;
//! * a perturbation never reaches farther than `R` (measured on an `f64` backend,
//!   where unaffected pixels are bit-identical);
//! * running the model on two overlapping halves of an image reproduces the
//!   whole-image output away from the cut edges: within `1e-9` on `f64`
//!   (rounding only), and within `1e-4` on `f32` through the real tiled `fuse_stack`
//!   path (away from the Hann blend bands).
#![cfg(feature = "ndarray")]

use burn::{
    backend::NdArray,
    prelude::{Backend, Tensor},
    tensor::{Distribution, TensorData},
};
use stacker_nn::{
    infer::{TileConfig, fuse_stack},
    model::{ChannelNorm, FocusMergeNet, FocusMergeNetConfig, ModelSize, receptive_field_radius},
    runtime::recommended_tile,
};

type F32 = NdArray;
type F64 = NdArray<f64>;

fn rand4<B: Backend>(shape: [usize; 4], dev: &B::Device) -> Tensor<B, 4> {
    Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, 1.0), dev)
}

fn values<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().convert::<f64>().to_vec::<f64>().unwrap()
}

// ---------------------------------------------------------------------------
// NN-5: one receptive-field function
// ---------------------------------------------------------------------------

#[test]
fn receptive_field_table() {
    let expected = [
        (ModelSize::Xs, 16),
        (ModelSize::S, 24),
        (ModelSize::M, 40),
        (ModelSize::L, 46),
        (ModelSize::Xl, 72),
        (ModelSize::Xxl, 100),
    ];
    for (size, radius) in expected {
        assert_eq!(receptive_field_radius(size), radius, "{}", size.as_str());
        assert_eq!(
            FocusMergeNetConfig::from_size(size).receptive_field_radius(),
            radius,
            "config radius of {}",
            size.as_str()
        );
        // Every consumer derives its overlap from the same radius.
        let tile = TileConfig::for_size(size);
        assert_eq!(tile.overlap, 2 * radius);
        assert!(tile.tile >= 2 * tile.overlap, "stride must not collapse");
    }
    // The model-agnostic defaults are safe for the largest preset.
    assert_eq!(recommended_tile().overlap, 200);
    assert_eq!(TileConfig::default().overlap, 200);
    assert_eq!(recommended_tile().tile, TileConfig::default().tile);
}

/// Largest Chebyshev distance from `(cy, cx)` of a pixel whose value differs
/// between `a` and `b` (`[C, H, W]` flattened, `c` channels) by more than `eps`,
/// together with the largest difference at distances beyond `radius`.
fn extent(
    base: &[f64],
    moved: &[f64],
    (chans, rows, cols): (usize, usize, usize),
    (centre_row, centre_col): (usize, usize),
    eps: f64,
    radius: usize,
) -> (usize, f64) {
    let mut reach = 0;
    let mut leak = 0.0_f64;
    for ch in 0..chans {
        for row in 0..rows {
            for col in 0..cols {
                let idx = (ch * rows + row) * cols + col;
                let dist = row.abs_diff(centre_row).max(col.abs_diff(centre_col));
                let diff = (base[idx] - moved[idx]).abs();
                if diff > eps {
                    reach = reach.max(dist);
                }
                if dist > radius {
                    leak = leak.max(diff);
                }
            }
        }
    }
    (reach, leak)
}

/// Perturb one source pixel in the middle of a large-enough image and measure how
/// far the change reaches in the merged output and the confidence.
fn check_radius(config: &FocusMergeNetConfig, side: usize) {
    let dev = <F64 as Backend>::Device::default();
    let model: FocusMergeNet<F64> = config.init(&dev);
    let radius = config.receptive_field_radius();
    let centre = side / 2;
    assert!(centre > radius + 2, "image too small to see the full field");

    let target = rand4::<F64>([1, 3, side, side], &dev);
    let target_conf = rand4::<F64>([1, 1, side, side], &dev);
    let source = rand4::<F64>([1, 3, side, side], &dev);

    // +1.0 on one red source pixel (a bump in a copy of the source).
    let mut bump = vec![0.0_f64; 3 * side * side];
    bump[centre * side + centre] = 1.0;
    let bump = Tensor::<F64, 4>::from_data(TensorData::new(bump, [1, 3, side, side]), &dev);
    let perturbed = source.clone().add(bump);

    let base = model.forward(target.clone(), target_conf.clone(), source);
    let moved = model.forward(target, target_conf, perturbed);

    for (name, a, b, ch) in [
        ("merged", base.merged, moved.merged, 3),
        ("conf", base.conf, moved.conf, 1),
    ] {
        let (reach, leak) = extent(
            &values(a),
            &values(b),
            (ch, side, side),
            (centre, centre),
            1e-12,
            radius,
        );
        // Nothing beyond R: unaffected pixels are bit-identical.
        assert!(
            leak <= 1e-12,
            "{name}: change of {leak} beyond the stated radius {radius}"
        );
        assert!(reach <= radius, "{name}: reach {reach} > radius {radius}");
        // And the field is really used: the change travels at least a quarter of R
        // (the signal shrinks by roughly 5x per distance-growing layer, so the very
        // edge of the field is below float resolution, but a quarter of it is not).
        assert!(
            reach >= radius / 4,
            "{name}: reach {reach} is less than a quarter of the stated radius {radius}"
        );
    }
}

#[test]
fn perturbation_stays_inside_the_stated_radius_xs() {
    // xs: dilations 1, 2 -> R = 16.
    check_radius(&FocusMergeNetConfig::from_size(ModelSize::Xs), 65);
}

#[test]
fn perturbation_stays_inside_the_stated_radius_past_one_dilation_cycle() {
    // Depth 6 goes through 1, 2, 4, 8, 1, 2 -> R = 46 (a narrow model keeps it fast).
    let config = FocusMergeNetConfig::new().with_width(8).with_depth(6);
    assert_eq!(config.receptive_field_radius(), 46);
    check_radius(&config, 105);
}

// ---------------------------------------------------------------------------
// NN-2: per-pixel normalisation, tile-invariant model
// ---------------------------------------------------------------------------

#[test]
fn channel_norm_is_per_pixel_and_normalises_channels() {
    let dev = <F32 as Backend>::Device::default();
    let norm = ChannelNorm::<F32>::new(1, 8, &dev);
    // Spread out (std about 1.2) so the variance floor `eps` is irrelevant.
    let input = rand4::<F32>([1, 8, 5, 5], &dev).mul_scalar(4.0);
    let normed = norm.forward(input.clone());

    // Zero mean and unit variance over the channels of every pixel.
    let mean = values(normed.clone().mean_dim(1));
    assert!(mean.iter().all(|m| m.abs() < 1e-5), "mean {mean:?}");
    let var = values(normed.clone().mul(normed.clone()).mean_dim(1));
    assert!(var.iter().all(|v| (v - 1.0).abs() < 1e-3), "var {var:?}");

    // Changing one pixel changes only that pixel's output.
    let mut poke = vec![0.0_f32; 8 * 25];
    for ch in 0..8_u8 {
        poke[usize::from(ch) * 25 + 12] = 3.0 * (f32::from(ch) + 1.0); // pixel (2, 2)
    }
    let poke = Tensor::<F32, 4>::from_data(TensorData::new(poke, [1, 8, 5, 5]), &dev);
    let poked = norm.forward(input.add(poke));
    let (before, after) = (values(normed), values(poked));
    for ch in 0..8 {
        for px in 0..25 {
            let idx = ch * 25 + px;
            if px == 12 {
                continue;
            }
            assert!(
                (before[idx] - after[idx]).abs() < 1e-6,
                "pixel {px} (channel {ch}) changed when only pixel 12 was edited"
            );
        }
    }
}

fn slice_cols<B: Backend>(t: Tensor<B, 4>, cols: std::ops::Range<usize>) -> Tensor<B, 4> {
    let [n, c, h, _] = t.dims();
    t.slice([0..n, 0..c, 0..h, cols])
}

#[test]
fn model_on_overlapping_halves_matches_the_whole_image() {
    let dev = <F64 as Backend>::Device::default();
    let config = FocusMergeNetConfig::from_size(ModelSize::Xs);
    let radius = config.receptive_field_radius(); // 16
    let model: FocusMergeNet<F64> = config.init(&dev);

    // 24 x 96, cut into [0, 64) and [32, 96): overlap 2R = 32.
    let (h, w) = (24, 96);
    let target = rand4::<F64>([1, 3, h, w], &dev);
    let target_conf = rand4::<F64>([1, 1, h, w], &dev);
    let source = rand4::<F64>([1, 3, h, w], &dev);
    assert_eq!(2 * radius, 32);

    let whole = model.forward(target.clone(), target_conf.clone(), source.clone());
    let run = |cols: std::ops::Range<usize>| {
        model.forward(
            slice_cols(target.clone(), cols.clone()),
            slice_cols(target_conf.clone(), cols.clone()),
            slice_cols(source.clone(), cols),
        )
    };
    let left = run(0..64);
    let right = run(32..96);

    // Interior of each half = everything at least R from the cut edge.
    let tol = 1e-9;
    for (name, whole_t, left_t, right_t) in [
        ("merged", whole.merged, left.merged, right.merged),
        ("conf", whole.conf, left.conf, right.conf),
        ("alpha", whole.alpha, left.alpha, right.alpha),
    ] {
        let reference_left = values(slice_cols(whole_t.clone(), 0..64 - radius));
        let tiled_left = values(slice_cols(left_t, 0..64 - radius));
        let reference_right = values(slice_cols(whole_t, 32 + radius..96));
        let tiled_right = values(slice_cols(right_t, radius..64));
        for (r, t) in reference_left.iter().zip(&tiled_left) {
            assert!(
                (r - t).abs() < tol,
                "{name}: left half off by {}",
                (r - t).abs()
            );
        }
        for (r, t) in reference_right.iter().zip(&tiled_right) {
            assert!(
                (r - t).abs() < tol,
                "{name}: right half off by {}",
                (r - t).abs()
            );
        }
    }
}

#[test]
fn tiled_fuse_stack_matches_whole_image_away_from_the_blend_band() {
    let dev = <F32 as Backend>::Device::default();
    let config = FocusMergeNetConfig::from_size(ModelSize::Xs);
    let model: FocusMergeNet<F32> = config.init(&dev);
    let radius = config.receptive_field_radius();

    let (h, w) = (24, 96);
    let frames: Vec<Tensor<F32, 3>> = (0..2)
        .map(|_| Tensor::<F32, 3>::random([3, h, w], Distribution::Uniform(0.0, 1.0), &dev))
        .collect();

    // One tile larger than the image: the whole-image fast path.
    let whole = fuse_stack(
        &model,
        &frames,
        TileConfig {
            tile: usize::MAX,
            overlap: 0,
        },
        &dev,
    )
    .unwrap();
    // 64-px tiles at x = 0 and x = 32 with the documented overlap 2R = 32.
    let tiled = fuse_stack(
        &model,
        &frames,
        TileConfig {
            tile: 64,
            overlap: 2 * radius,
        },
        &dev,
    )
    .unwrap();

    let (whole, tiled) = (values(whole), values(tiled));
    // Columns [0, 32) come from the first tile alone, [64, 96) from the second
    // alone; both have at least R px of context inside their tile.
    let tol = 1e-4;
    let mut worst = 0.0_f64;
    for ch in 0..3 {
        for y in 0..h {
            for x in (0..32).chain(64..96) {
                let i = (ch * h + y) * w + x;
                worst = worst.max((whole[i] - tiled[i]).abs());
            }
        }
    }
    assert!(
        worst < tol,
        "tiled fold differs from the whole image by {worst}"
    );
}
