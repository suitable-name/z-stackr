//! Integration tests for [`stacker_nn::model::FocusMergeNet`] and friends.

use burn::{
    backend::{Autodiff, NdArray},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::Tensor,
    tensor::Distribution,
};
use stacker_nn::{
    model::{
        BatchAlignNet, BatchAlignNetConfig, BatchMergeNet, BatchMergeNetConfig, FocusMergeNet,
        FocusMergeNetConfig, FusionAlignNet, FusionAlignNetConfig, ModelSize,
    },
    traits::{BatchAlignmentModel, FusionModel, PairAlignmentModel},
};

type B = NdArray;

fn device() -> burn::prelude::Device<B> {
    burn::prelude::Device::<B>::default()
}

/// Build a default model on `NdArray`.
fn default_model() -> FocusMergeNet<B> {
    FocusMergeNetConfig::new().init(&device())
}

/// Random rank-4 tensor on `NdArray`.
fn rand_tensor(shape: [usize; 4]) -> Tensor<B, 4> {
    Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &device())
}

#[test]
fn forward_shape_square() {
    let model = default_model();
    let target = rand_tensor([1, 3, 16, 16]);
    let target_conf = rand_tensor([1, 1, 16, 16]);
    let source = rand_tensor([1, 3, 16, 16]);

    let out = model.forward(target, target_conf, source);

    assert_eq!(out.merged.dims(), [1, 3, 16, 16]);
    assert_eq!(out.conf.dims(), [1, 1, 16, 16]);
    assert_eq!(out.alpha.dims(), [1, 1, 16, 16]);
}

#[test]
fn forward_shape_non_square_odd() {
    // Verify Same padding preserves non-square / odd spatial dimensions.
    let model = default_model();
    let target = rand_tensor([1, 3, 15, 17]);
    let target_conf = rand_tensor([1, 1, 15, 17]);
    let source = rand_tensor([1, 3, 15, 17]);

    let out = model.forward(target, target_conf, source);

    assert_eq!(out.merged.dims(), [1, 3, 15, 17]);
    assert_eq!(out.conf.dims(), [1, 1, 15, 17]);
    assert_eq!(out.alpha.dims(), [1, 1, 15, 17]);
}

#[test]
fn alpha_and_conf_in_unit_range() {
    let model = default_model();
    let target = rand_tensor([1, 3, 16, 16]);
    let target_conf = rand_tensor([1, 1, 16, 16]);
    let source = rand_tensor([1, 3, 16, 16]);

    let out = model.forward(target, target_conf, source);

    let alpha_lo: f32 = out.alpha.clone().min().into_scalar();
    let alpha_hi: f32 = out.alpha.max().into_scalar();
    assert!(alpha_lo >= 0.0, "alpha min {alpha_lo} < 0");
    assert!(alpha_hi <= 1.0, "alpha max {alpha_hi} > 1");

    let conf_lo: f32 = out.conf.clone().min().into_scalar();
    let conf_hi: f32 = out.conf.max().into_scalar();
    assert!(conf_lo >= 0.0, "conf min {conf_lo} < 0");
    assert!(conf_hi <= 1.0, "conf max {conf_hi} > 1");
}

#[test]
fn backward_pass_no_panic() {
    // Use the autodiff backend to verify end-to-end differentiability.
    type Ab = Autodiff<NdArray>;

    let dev = burn::prelude::Device::<Ab>::default();
    let model: FocusMergeNet<Ab> = FocusMergeNetConfig::new().init(&dev);

    let rand4 =
        |shape: [usize; 4]| Tensor::<Ab, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &dev);

    let target = rand4([1, 3, 8, 8]);
    let target_conf = rand4([1, 1, 8, 8]);
    let source = rand4([1, 3, 8, 8]);

    let out = model.forward(target, target_conf, source);
    let loss = out.merged.mean();
    let scalar: f32 = loss.clone().into_scalar();
    assert!(scalar.is_finite(), "loss {scalar} is not finite");

    // backward() not panicking proves the graph is differentiable.
    let _grads = loss.backward();
}

#[test]
fn all_presets_forward_shape() {
    // Every preset must initialise and run while preserving spatial dims.
    let dev = device();
    for size in ModelSize::ALL {
        let model: FocusMergeNet<B> = FocusMergeNetConfig::from_size(size).init(&dev);
        let target = rand_tensor([1, 3, 16, 16]);
        let target_conf = rand_tensor([1, 1, 16, 16]);
        let source = rand_tensor([1, 3, 16, 16]);

        let out = model.forward(target, target_conf, source);
        assert_eq!(
            out.merged.dims(),
            [1, 3, 16, 16],
            "preset {} changed spatial dims",
            size.as_str()
        );
        assert_eq!(out.conf.dims(), [1, 1, 16, 16]);
    }
}

#[test]
fn model_size_parse_roundtrips() {
    for size in ModelSize::ALL {
        assert_eq!(ModelSize::parse(size.as_str()), Some(size));
        assert_eq!(ModelSize::parse(&size.as_str().to_uppercase()), Some(size));
    }
    assert_eq!(ModelSize::parse("nonsense"), None);
}

// ---------------------------------------------------------------------------
// BatchMergeNet
// ---------------------------------------------------------------------------

/// [`BatchMergeNet::forward`] on a `[1, S, 3, H, W]` stack must preserve
/// spatial dims, collapse the stack dim, and produce finite output.
#[test]
fn batch_merge_net_forward_shape() {
    let dev = device();
    let model: BatchMergeNet<B> = BatchMergeNetConfig::new().init(&dev);
    let stack = Tensor::<B, 5>::random([1, 3, 3, 16, 16], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.fuse_batch(stack);

    assert_eq!(out.dims(), [1, 3, 16, 16], "output shape mismatch");
    let data = out.into_data();
    for v in data.iter::<f32>() {
        assert!(v.is_finite(), "output contains non-finite value: {v}");
    }
}

// ---------------------------------------------------------------------------
// BatchAlignNet
// ---------------------------------------------------------------------------

/// Bottom row of every output matrix must be EXACTLY `[0, 0, 1]`
/// (constructed via `Tensor::zeros`/`Tensor::ones`, not a learned value),
/// and every output value must be finite — the basic reshape/concat
/// plumbing sanity check, independent of the `Linear` head's
/// initialisation scheme.
#[test]
fn batch_align_net_bottom_row_is_exact_identity_row() {
    let dev = device();
    let model: BatchAlignNet<B> = BatchAlignNetConfig::new().init(&dev);
    let stack = Tensor::<B, 5>::random([2, 3, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_batch(stack);
    assert_eq!(out.dims(), [2, 3, 3, 3], "output shape mismatch");

    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    let n = 2 * 3;
    for i in 0..n {
        let base = i * 9;
        let m = &data[base..base + 9];
        // Bottom row must be EXACTLY [0, 0, 1] (constructed, not learned).
        assert!(
            (m[6] - 0.0).abs() < 1e-8 && (m[7] - 0.0).abs() < 1e-8 && (m[8] - 1.0).abs() < 1e-8,
            "matrix {i} bottom row {:?} != [0, 0, 1]",
            &m[6..9]
        );
        for (k, &v) in m.iter().enumerate() {
            assert!(v.is_finite(), "matrix {i} entry {k} is not finite: {v}");
        }
    }
}

// ---------------------------------------------------------------------------
// BatchAlignNet v2 — design doc §3.6 "Phase 1 definition of done" tests.
// ---------------------------------------------------------------------------

/// A small config used by the v2 tests below — keeps them fast (tiny spatial
/// input) while still exercising the full x8-downsample encoder + head.
fn tiny_align_config() -> BatchAlignNetConfig {
    BatchAlignNetConfig::new()
        .with_width(8)
        .with_depth(1)
        .with_norm_groups(2)
        .with_corr_radius(2)
}

/// §3.6 test 1: `batch_align_v2_output_shape` — random `[1, 4, 3, 96, 96]`
/// in, `[1, 4, 3, 3]` out, all finite.
#[test]
fn batch_align_v2_output_shape() {
    let dev = device();
    let model: BatchAlignNet<B> = tiny_align_config().init(&dev);
    let stack = Tensor::<B, 5>::random([1, 4, 3, 96, 96], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_batch(stack);
    assert_eq!(out.dims(), [1, 4, 3, 3], "output shape mismatch");

    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    for (k, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "entry {k} is not finite: {v}");
    }
}

/// §3.6 test 2 (renamed from v1's `batch_align_net_is_exact_identity_at_init`):
/// `batch_align_v2_is_exact_identity_at_init` — zero-init of the head's final
/// `Linear` layer plus the §3.5 frame-0 normalisation means every predicted
/// matrix (not just frame 0's) must be EXACTLY the 3x3 identity to `1e-6` at
/// initialisation, for every frame S in the stack.
#[test]
fn batch_align_v2_is_exact_identity_at_init() {
    let dev = device();
    let model: BatchAlignNet<B> = tiny_align_config().init(&dev);
    let stack = Tensor::<B, 5>::random([2, 3, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_batch(stack);
    assert_eq!(out.dims(), [2, 3, 3, 3], "output shape mismatch");

    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    let identity: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let n = 2 * 3;
    for i in 0..n {
        let base = i * 9;
        let m = &data[base..base + 9];
        for (k, (&v, &want)) in m.iter().zip(identity.iter()).enumerate() {
            assert!(
                (v - want).abs() < 1e-6,
                "matrix {i} entry {k} = {v}, expected {want} (exact identity at init)"
            );
        }
    }
}

/// §3.6 test 3: `batch_align_v2_frame0_identity_always` — after one optimiser
/// step on random data (so weights are non-zero), frame 0's matrix is still
/// exact identity, because §3.5's frame-0 normalisation is structural
/// (`M_0' = M_0^-1 . M_0 = I` algebraically), not an artifact of zero-init.
#[test]
fn batch_align_v2_frame0_identity_always() {
    type Ab = Autodiff<NdArray>;
    let dev = burn::prelude::Device::<Ab>::default();
    let mut model: BatchAlignNet<Ab> = tiny_align_config().init(&dev);

    let stack = Tensor::<Ab, 5>::random([2, 3, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);
    let out = model.align_batch(stack.clone());
    // Any scalar loss suffices — we only need non-zero gradients so the
    // weights move away from their zero-init.
    let loss = out.sum();

    let mut optim = AdamWConfig::new().init();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    model = optim.step(1e-2, model, grads);

    // Re-run with the now-updated (non-zero) weights.
    let out2 = model.align_batch(stack);
    let data: Vec<f32> = out2.into_data().iter::<f32>().collect();
    let identity: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    // Frame 0 of both stacks (n=0,s=0 and n=1,s=0) must still be identity.
    for n_idx in 0..2 {
        let base = (n_idx * 3) * 9; // s = 0 is the first frame of stack n_idx
        let m = &data[base..base + 9];
        for (k, (&v, &want)) in m.iter().zip(identity.iter()).enumerate() {
            assert!(
                (v - want).abs() < 1e-4,
                "stack {n_idx} frame 0 entry {k} = {v}, expected {want} (must stay identity \
                 after an optimiser step)"
            );
        }
    }
}

/// §3.6 test 4: `local_correlation_peaks_at_true_shift` — build a random
/// `[1, 8, 16, 16]` feature map, shift it by `(+2, -1)` with the same
/// pad/slice technique the model uses, correlate with `r = 3`, and assert
/// the argmax channel at interior positions is the one encoding displacement
/// `(dy=2, dx=-1)`. Catches the easiest bug in this phase: mixing up
/// displacement sign or row-major channel order.
#[test]
// `r`/`c`/`h`/`w` (search radius / channels / height / width) mirror the
// design doc's `local_correlation` signature verbatim; `dy`/`dx` are the
// paired displacement components under test. All `usize -> i64` casts below
// operate on test-fixture sizes (<= 16), nowhere near `i64::MAX`, so wrap
// cannot occur in practice; the lint has no size context to know that.
// `wanted_delta_y_idx`/`wanted_delta_x_idx` intentionally share a name
// prefix — they are the paired y/x halves of the same expected-index
// computation immediately below.
#[allow(clippy::many_single_char_names, clippy::similar_names)]
fn local_correlation_peaks_at_true_shift() {
    use burn::tensor::TensorData;

    let dev = device();
    let r: usize = 3;
    let (c, h, w) = (8_usize, 16_usize, 16_usize);

    // Deterministic feature map with a SINGLE, SHARP, UNIQUE landmark per
    // channel (a one-hot "delta spike"): channel `ci` has value `1.0` at
    // exactly one pixel `(spike_y[ci], spike_x[ci])` and `0.0` everywhere
    // else. This is essential, not decoration — several earlier attempts
    // each had a disqualifying flaw, all variants of the same root cause
    // (the "aperture problem": smooth/low-frequency content has no sharp,
    // locally-unique structure for a correlation to lock onto, so the
    // argmax at a given pixel is dominated by noise/asymmetry rather than
    // the true global shift):
    //
    // * per-pixel i.i.d. "random" content has no spatial correlation at all;
    // * a periodic 2D sinusoid aliases across its own period;
    // * a chained pseudo-random "scramble" reshaped into 2D has no real 2D
    //   locality;
    // * multiple identically-shaped Gaussian bumps re-introduce aliasing
    //   between different bumps of the same shape;
    // * even a single very broad Gaussian (an attempted "smooth gradient")
    //   is still smooth enough, near any given pixel, that neighbouring
    //   displacements score almost as well as the true one — this was
    //   verified directly: a `debug_corr_delta_spike` scratch test (removed
    //   after diagnosis) confirmed `local_correlation` ITSELF is correct
    //   (a single spike's true displacement is found EXACTLY, with the
    //   correlation being identically zero everywhere else), proving every
    //   prior failure here was a test-fixture defect, not an implementation
    //   bug.
    //
    // A one-hot spike per channel has a perfectly sharp, globally unique
    // peak with zero ambiguity: at the spike's own reference location, the
    // correlation is *exactly* zero for every displacement except the one
    // pointing at the spike's shifted location, where it is *exactly*
    // `1/c` (mean over channels, only one of which is nonzero there).
    let spike_pos: [(usize, usize); 8] = [
        (8, 8),
        (6, 10),
        (10, 6),
        (7, 7),
        (9, 9),
        (6, 6),
        (10, 10),
        (8, 6),
    ];
    let mut data = vec![0f32; c * h * w];
    for (ci, &(sy, sx)) in spike_pos.iter().enumerate() {
        data[(ci * h + sy) * w + sx] = 1.0;
    }
    let f_ref = Tensor::<B, 4>::from_data(TensorData::new(data, [1, c, h, w]), &dev);

    // Shift the reference by (dy=2, dx=-1): content at f_ref[y,x] moves to
    // f_frm[y+2, x-1]. Each channel's spike therefore moves from
    // `(sy, sx)` to `(sy+2, sx-1)`.
    let dy: i64 = 2;
    let dx: i64 = -1;
    let mut shifted = vec![0f32; c * h * w];
    for (ci, &(sy, sx)) in spike_pos.iter().enumerate() {
        let ny = usize::try_from(i64::try_from(sy).unwrap() + dy).expect("stays in-bounds");
        let nx = usize::try_from(i64::try_from(sx).unwrap() + dx).expect("stays in-bounds");
        shifted[(ci * h + ny) * w + nx] = 1.0;
    }
    let f_frm = Tensor::<B, 4>::from_data(TensorData::new(shifted, [1, c, h, w]), &dev);

    // Exercise the model's actual (doc-hidden but `pub`) local_correlation
    // directly, so this test catches a real sign/order bug in the
    // implementation rather than a hand-duplicated reference.
    let corr = stacker_nn::model::local_correlation(&f_ref, f_frm, r, &dev);
    let corr_data: Vec<f32> = corr.into_data().iter::<f32>().collect();
    let n_disp = 2 * r + 1;
    let wanted_delta_y_idx = usize::try_from(dy + r as i64).expect("in-range by construction");
    let wanted_delta_x_idx = usize::try_from(dx + r as i64).expect("in-range by construction");
    let want_k = wanted_delta_y_idx * n_disp + wanted_delta_x_idx;

    // Check the argmax at every one of this channel's own spike positions
    // (each is an "interior" position by construction, since all spike
    // coordinates sit comfortably away from the canvas border relative to
    // `r`): the true displacement must be the (exact, unique) maximum.
    for &(y, x) in &spike_pos {
        let mut best_k = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for k in 0..(n_disp * n_disp) {
            let idx = (k * h + y) * w + x;
            let v = corr_data[idx];
            if v > best_v {
                best_v = v;
                best_k = k;
            }
        }
        assert_eq!(
            best_k,
            want_k,
            "at ({y},{x}): argmax channel {best_k} (val {best_v}) != expected {want_k} \
             (val {}) (expected displacement dy={dy}, dx={dx})",
            corr_data[(want_k * h + y) * w + x]
        );
    }
}

/// §3.6 test 5: `matrix_determinant_always_positive` — random inputs, assert
/// `det(A) > 0` for every output matrix (the §3.4 bounded parameterisation
/// guarantees `det(A) = sx * sy >= exp(-0.16) > 0`).
#[test]
fn matrix_determinant_always_positive() {
    let dev = device();
    let model: BatchAlignNet<B> = tiny_align_config().init(&dev);
    let stack = Tensor::<B, 5>::random([2, 4, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_batch(stack);
    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    let n = 2 * 4;
    for i in 0..n {
        let base = i * 9;
        let m = &data[base..base + 9];
        let det = m[1].mul_add(-m[3], m[0] * m[4]);
        assert!(det > 0.0, "matrix {i}: det(A) = {det} is not positive");
    }
}

// ---------------------------------------------------------------------------
// FusionAlignNet — the pairwise (streaming) sibling of BatchAlignNet.
// See docs/fusionalign-design.md for the architecture rationale; these tests
// mirror the "BatchAlignNet v2" block above, adapted to the reference/frame
// pair shape (`[N,3,3]`, no `S` dimension) instead of a whole-stack batch.
// ---------------------------------------------------------------------------

/// A small config used by the tests below — mirrors `tiny_align_config`
/// above, keeping these fast while still exercising the full x8-downsample
/// encoder + correlation + head.
fn tiny_fusion_align_config() -> FusionAlignNetConfig {
    FusionAlignNetConfig::new()
        .with_width(8)
        .with_depth(1)
        .with_norm_groups(2)
        .with_corr_radius(2)
}

/// Every [`ModelSize`] preset must construct and run without panicking,
/// preserving the `[N,3,3]` output shape and producing finite values.
#[test]
fn fusion_align_net_all_presets_construct_and_run() {
    let dev = device();
    for size in ModelSize::ALL {
        let model: FusionAlignNet<B> = FusionAlignNetConfig::from_size(size).init(&dev);
        let reference =
            Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);
        let frame = Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);

        let out = model.align_pair(reference, frame);
        assert_eq!(
            out.dims(),
            [2, 3, 3],
            "preset {} produced wrong output shape",
            size.as_str()
        );

        let data: Vec<f32> = out.into_data().iter::<f32>().collect();
        for (k, &v) in data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "preset {}: entry {k} is not finite: {v}",
                size.as_str()
            );
        }
    }
}

/// `align_pair`'s output shape contract: random `[1, 3, 96, 96]` reference +
/// frame in, `[1, 3, 3]` out, all finite — mirrors
/// `batch_align_v2_output_shape` for the pairwise architecture.
#[test]
fn fusion_align_output_shape() {
    let dev = device();
    let model: FusionAlignNet<B> = tiny_fusion_align_config().init(&dev);
    let reference = Tensor::<B, 4>::random([1, 3, 96, 96], Distribution::Uniform(0.0, 1.0), &dev);
    let frame = Tensor::<B, 4>::random([1, 3, 96, 96], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_pair(reference, frame);
    assert_eq!(out.dims(), [1, 3, 3], "output shape mismatch");

    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    for (k, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "entry {k} is not finite: {v}");
    }
}

/// Bottom row of every output matrix must be EXACTLY `[0, 0, 1]`
/// (constructed via `Tensor::zeros`/`Tensor::ones` inside
/// `bounded_affine_from_raw6`, not a learned value) — mirrors
/// `batch_align_net_bottom_row_is_exact_identity_row`.
#[test]
fn fusion_align_bottom_row_is_exact_identity_row() {
    let dev = device();
    let model: FusionAlignNet<B> = FusionAlignNetConfig::new().init(&dev);
    let reference = Tensor::<B, 4>::random([3, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);
    let frame = Tensor::<B, 4>::random([3, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_pair(reference, frame);
    assert_eq!(out.dims(), [3, 3, 3], "output shape mismatch");

    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    for i in 0..3 {
        let base = i * 9;
        let m = &data[base..base + 9];
        assert!(
            (m[6] - 0.0).abs() < 1e-8 && (m[7] - 0.0).abs() < 1e-8 && (m[8] - 1.0).abs() < 1e-8,
            "matrix {i} bottom row {:?} != [0, 0, 1]",
            &m[6..9]
        );
        for (k, &v) in m.iter().enumerate() {
            assert!(v.is_finite(), "matrix {i} entry {k} is not finite: {v}");
        }
    }
}

/// At initialisation (zero-init final `Linear` layer, per
/// `bounded_affine_from_raw6`'s docs), every predicted matrix must be
/// EXACTLY the 3x3 identity to `1e-6` — mirrors
/// `batch_align_v2_is_exact_identity_at_init`. Unlike `BatchAlignNet`,
/// `FusionAlignNet` has no reference-normalisation step at all (see
/// `FusionAlignNet`'s docs), so this identity comes ENTIRELY from the
/// zero-init head, for every frame passed in — there is no analogue of
/// `batch_align_v2_frame0_identity_always` because there is no frame-0
/// special case to stay invariant after a training step.
#[test]
fn fusion_align_is_exact_identity_at_init() {
    let dev = device();
    let model: FusionAlignNet<B> = tiny_fusion_align_config().init(&dev);
    let reference = Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);
    let frame = Tensor::<B, 4>::random([2, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_pair(reference, frame);
    assert_eq!(out.dims(), [2, 3, 3], "output shape mismatch");

    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    let identity: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for i in 0..2 {
        let base = i * 9;
        let m = &data[base..base + 9];
        for (k, (&v, &want)) in m.iter().zip(identity.iter()).enumerate() {
            assert!(
                (v - want).abs() < 1e-6,
                "matrix {i} entry {k} = {v}, expected {want} (exact identity at init)"
            );
        }
    }
}

/// `det(A) = sx * sy >= exp(-0.16) > 0` for every output matrix — mirrors
/// `matrix_determinant_always_positive` for the pairwise architecture (both
/// share the identical `bounded_affine_from_raw6` head).
#[test]
fn fusion_align_matrix_determinant_always_positive() {
    let dev = device();
    let model: FusionAlignNet<B> = tiny_fusion_align_config().init(&dev);
    let reference = Tensor::<B, 4>::random([4, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);
    let frame = Tensor::<B, 4>::random([4, 3, 32, 32], Distribution::Uniform(0.0, 1.0), &dev);

    let out = model.align_pair(reference, frame);
    let data: Vec<f32> = out.into_data().iter::<f32>().collect();
    for i in 0..4 {
        let base = i * 9;
        let m = &data[base..base + 9];
        let det = m[1].mul_add(-m[3], m[0] * m[4]);
        assert!(det > 0.0, "matrix {i}: det(A) = {det} is not positive");
    }
}
