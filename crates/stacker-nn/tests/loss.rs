//! Integration tests for [`stacker_nn::loss::FocusFusionLoss`].

use burn::{
    backend::{Autodiff, NdArray},
    prelude::Tensor,
    tensor::Distribution,
};
use stacker_nn::{
    loss::{
        BatchAlignmentLossConfig, FocusBatchLoss, FocusBatchLossConfig, FocusFusionLoss,
        FocusFusionLossConfig, PairAlignmentLossConfig,
    },
    model::{BatchMergeNet, BatchMergeNetConfig, FocusMergeNetConfig},
    traits::{
        BatchAlignmentLoss, BatchFusionLoss, MergeStep, PairAlignmentLoss, PairwiseFusionLoss,
    },
};

type B = NdArray;

fn device() -> burn::prelude::Device<B> {
    burn::prelude::Device::<B>::default()
}

fn rand4(shape: [usize; 4]) -> Tensor<B, 4> {
    Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &device())
}

fn default_loss() -> FocusFusionLoss {
    FocusFusionLossConfig::new().init()
}

/// Build a hand-constructed `MergeStep` from random tensors.
fn fake_pred(nb: usize, nh: usize, nw: usize) -> MergeStep<B> {
    MergeStep {
        merged: rand4([nb, 3, nh, nw]),
        conf: rand4([nb, 1, nh, nw]),
    }
}

#[test]
fn loss_forward_finite_scalars() {
    let loss = default_loss();
    let pred = fake_pred(1, 16, 16);
    let gt_merged = rand4([1, 3, 16, 16]);
    let gt_conf = rand4([1, 1, 16, 16]);
    let source = rand4([1, 3, 16, 16]);
    let occlusion = rand4([1, 1, 16, 16]);

    let total = loss.forward(&pred, &gt_merged, gt_conf, source, &occlusion);

    let check = |name: &str, val: f32| {
        assert!(val.is_finite(), "{name} is not finite: {val}");
        assert!(val >= 0.0, "{name} is negative: {val}");
    };
    check("total", total.into_scalar());
}

#[test]
fn loss_forward_finite_non_square() {
    // Ensure the loss handles non-square inputs (important for tiling).
    let loss = default_loss();
    let pred = fake_pred(2, 15, 17);
    let gt_merged = rand4([2, 3, 15, 17]);
    let gt_conf = rand4([2, 1, 15, 17]);
    let source = rand4([2, 3, 15, 17]);
    let occlusion = rand4([2, 1, 15, 17]);

    let total = loss.forward(&pred, &gt_merged, gt_conf, source, &occlusion);
    assert!(total.into_scalar().is_finite());
}

#[test]
fn loss_backward_end_to_end() {
    // Full autodiff smoke test: forward + backward through model + loss.
    type Ab = Autodiff<NdArray>;

    let dev = burn::prelude::Device::<Ab>::default();

    let rand4_ab =
        |shape: [usize; 4]| Tensor::<Ab, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &dev);

    let model = FocusMergeNetConfig::new().init::<Ab>(&dev);

    let target = rand4_ab([1, 3, 8, 8]);
    let target_conf = rand4_ab([1, 1, 8, 8]);
    let source = rand4_ab([1, 3, 8, 8]);

    let out = model.forward(target, target_conf, source.clone());
    let pred = MergeStep {
        merged: out.merged,
        conf: out.conf,
    };

    let loss = FocusFusionLossConfig::new().init();
    let gt_merged = rand4_ab([1, 3, 8, 8]);
    let gt_conf = rand4_ab([1, 1, 8, 8]);
    let occlusion = rand4_ab([1, 1, 8, 8]);

    let total = loss.forward(&pred, &gt_merged, gt_conf, source, &occlusion);

    let scalar: f32 = total.clone().into_scalar();
    assert!(scalar.is_finite(), "total loss {scalar} not finite");

    // backward() not panicking proves end-to-end differentiability.
    let _grads = total.backward();
}

// ---------------------------------------------------------------------------
// CornerAlignmentLoss
// ---------------------------------------------------------------------------

/// Build an `[N, S, 3, 3]` tensor by tiling one 3x3 row-major matrix over
/// every `(n, s)` slot.
fn tile_matrix(n: usize, s: usize, m: [f32; 9]) -> Tensor<B, 4> {
    let dev = device();
    let base = Tensor::<B, 1>::from_floats(m, &dev).reshape([1, 1, 3, 3]);
    base.expand([n, s, 3, 3])
}

/// Identity matrix, row-major.
const IDENTITY_3X3: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[test]
fn corner_alignment_loss_zero_when_matrices_match() {
    let loss = BatchAlignmentLossConfig::new().init();
    let m = tile_matrix(2, 3, IDENTITY_3X3);

    let total: f32 = loss.forward(m.clone(), m).into_scalar();
    assert!(
        total.abs() < 1e-6,
        "loss should be ~0 for identical matrices, got {total}"
    );
}

#[test]
fn corner_alignment_loss_increases_with_perturbation() {
    let loss = BatchAlignmentLossConfig::new().init();
    let gt = tile_matrix(1, 1, IDENTITY_3X3);

    // Small perturbation: slight translation in the top-right entry.
    let mut small_pert = IDENTITY_3X3;
    small_pert[2] = 0.05;
    let pred_small = tile_matrix(1, 1, small_pert);

    // Larger perturbation: bigger translation, same entry.
    let mut large_pert = IDENTITY_3X3;
    large_pert[2] = 0.3;
    let pred_large = tile_matrix(1, 1, large_pert);

    let loss_small: f32 = loss.forward(pred_small, gt.clone()).into_scalar();
    let loss_large: f32 = loss.forward(pred_large, gt).into_scalar();

    assert!(
        loss_small > 0.0,
        "small-perturbation loss should be > 0, got {loss_small}"
    );
    assert!(
        loss_large > 0.0,
        "large-perturbation loss should be > 0, got {loss_large}"
    );
    assert!(
        loss_small < loss_large,
        "loss should grow with perturbation magnitude: small {loss_small} >= large {loss_large}"
    );
}

// ---------------------------------------------------------------------------
// PairCornerAlignmentLoss — the `[N,3,3]` (no `S`) analogue of
// CornerAlignmentLoss, used to train FusionAlignNet.
// ---------------------------------------------------------------------------

/// Build an `[N, 3, 3]` tensor by tiling one 3x3 row-major matrix over every
/// `n` slot — the pairwise (no `S` dimension) analogue of `tile_matrix`.
fn tile_matrix_pair(n: usize, m: [f32; 9]) -> Tensor<B, 3> {
    let dev = device();
    let base = Tensor::<B, 1>::from_floats(m, &dev).reshape([1, 3, 3]);
    base.expand([n, 3, 3])
}

#[test]
fn pair_corner_alignment_loss_zero_when_matrices_match() {
    let loss = PairAlignmentLossConfig::new().init();
    let m = tile_matrix_pair(4, IDENTITY_3X3);

    let total: f32 = loss.forward(m.clone(), m).into_scalar();
    assert!(
        total.abs() < 1e-6,
        "loss should be ~0 for identical matrices, got {total}"
    );
}

/// The loss must decrease toward zero as the predicted matrix approaches the
/// ground truth — i.e. it is a well-behaved training signal, not just zero
/// at the exact match and otherwise flat/non-monotonic.
#[test]
fn pair_corner_alignment_loss_decreases_toward_zero_as_pred_approaches_gt() {
    let loss = PairAlignmentLossConfig::new().init();
    let gt = tile_matrix_pair(1, IDENTITY_3X3);

    let mut far_pert = IDENTITY_3X3;
    far_pert[2] = 0.3; // translation entry
    let mut near_pert = IDENTITY_3X3;
    near_pert[2] = 0.05;

    let loss_far: f32 = loss
        .forward(tile_matrix_pair(1, far_pert), gt.clone())
        .into_scalar();
    let loss_near: f32 = loss
        .forward(tile_matrix_pair(1, near_pert), gt.clone())
        .into_scalar();
    let loss_zero: f32 = loss.forward(gt.clone(), gt).into_scalar();

    assert!(
        loss_far > loss_near,
        "loss_far {loss_far} <= loss_near {loss_near}"
    );
    assert!(
        loss_near > loss_zero,
        "loss_near {loss_near} <= loss_zero {loss_zero}"
    );
    assert!(
        loss_zero.abs() < 1e-6,
        "loss at exact match should be ~0, got {loss_zero}"
    );
}

#[test]
fn pair_corner_alignment_loss_increases_with_perturbation() {
    let loss = PairAlignmentLossConfig::new().init();
    let gt = tile_matrix_pair(1, IDENTITY_3X3);

    let mut small_pert = IDENTITY_3X3;
    small_pert[2] = 0.05;
    let pred_small = tile_matrix_pair(1, small_pert);

    let mut large_pert = IDENTITY_3X3;
    large_pert[2] = 0.3;
    let pred_large = tile_matrix_pair(1, large_pert);

    let loss_small: f32 = loss.forward(pred_small, gt.clone()).into_scalar();
    let loss_large: f32 = loss.forward(pred_large, gt).into_scalar();

    assert!(
        loss_small > 0.0,
        "small-perturbation loss should be > 0, got {loss_small}"
    );
    assert!(
        loss_large > 0.0,
        "large-perturbation loss should be > 0, got {loss_large}"
    );
    assert!(
        loss_small < loss_large,
        "loss should grow with perturbation magnitude: small {loss_small} >= large {loss_large}"
    );
}

// ---------------------------------------------------------------------------
// FocusBatchLoss
// ---------------------------------------------------------------------------

fn default_batch_loss() -> FocusBatchLoss {
    FocusBatchLossConfig::new().init()
}

/// Random `[N, S, 1, H, W]` tensor in `0..1` (masks / alpha shape).
fn rand5_masks(n: usize, s: usize, h: usize, w: usize) -> Tensor<B, 5> {
    Tensor::<B, 5>::random([n, s, 1, h, w], Distribution::Uniform(0.0, 1.0), &device())
}

/// Random `[N, S, 3, H, W]` stack tensor in `0..1`.
fn rand_stack(n: usize, s: usize, h: usize, w: usize) -> Tensor<B, 5> {
    Tensor::<B, 5>::random([n, s, 3, h, w], Distribution::Uniform(0.0, 1.0), &device())
}

#[test]
fn charbonnier_batch_loss_near_zero_when_equal() {
    let loss = default_batch_loss();
    let pred = rand4([1, 3, 16, 16]);
    let gt = pred.clone();

    let total: f32 = BatchFusionLoss::forward(&loss, &pred, &gt).into_scalar();
    // Charbonnier's epsilon smoothing (default charbonnier_eps = 0.001) means
    // the reconstruction term floors at exactly `eps` (sqrt(0 + eps^2)), not
    // 0, when pred == gt; the gradient term is exactly 0 in that case. Use a
    // tolerance just above the default eps to allow for that floor.
    assert!(
        total < 2e-3,
        "loss should be near 0 (floored at charbonnier_eps) for identical inputs, got {total}"
    );
    assert!(total >= 0.0, "loss should be non-negative, got {total}");
}

#[test]
fn charbonnier_batch_loss_positive_when_different() {
    let loss = default_batch_loss();
    let pred = rand4([1, 3, 16, 16]);
    let gt = rand4([1, 3, 16, 16]);

    let total: f32 = BatchFusionLoss::forward(&loss, &pred, &gt).into_scalar();
    assert!(
        total > 2e-3,
        "loss should be clearly positive for different inputs, got {total}"
    );
}

// ---------------------------------------------------------------------------
// BatchMergeNet::forward_with_alpha
// ---------------------------------------------------------------------------

/// `alpha`'s shape must be `[N, S, 1, H, W]`, it must sum to 1 (±1e-4) over
/// the stack dimension at every pixel (it is a softmax), and `merged` must be
/// finite.
#[test]
fn batch_merge_forward_with_alpha_shapes_and_simplex() {
    let dev = device();
    let model: BatchMergeNet<B> = BatchMergeNetConfig::new().init(&dev);
    let stack = rand_stack(1, 4, 16, 16);

    let (merged, alpha) = model.forward_with_alpha(stack);

    assert_eq!(merged.dims(), [1, 3, 16, 16]);
    assert_eq!(alpha.dims(), [1, 4, 1, 16, 16]);

    let merged_data: Vec<f32> = merged.into_data().iter::<f32>().collect();
    for (i, v) in merged_data.iter().enumerate() {
        assert!(v.is_finite(), "merged[{i}] is not finite: {v}");
    }

    let sums = alpha.sum_dim(1); // [1, 1, 1, 16, 16]
    let sums_data: Vec<f32> = sums.into_data().iter::<f32>().collect();
    for (i, &s) in sums_data.iter().enumerate() {
        assert!(
            (s - 1.0).abs() < 1e-4,
            "alpha sum at pixel {i} = {s}, expected ~1.0"
        );
    }
}

// ---------------------------------------------------------------------------
// FocusBatchLoss::forward_full
// ---------------------------------------------------------------------------

/// Build a `[1, S, 1, H, W]` alpha tensor that exactly matches `masks`'
/// normalised distribution, given `masks: [1, S, 1, H, W]`.
fn alpha_matching_masks(masks: &Tensor<B, 5>) -> Tensor<B, 5> {
    let sum = masks.clone().sum_dim(1);
    masks.clone().div(sum.add_scalar(1e-6_f32))
}

#[test]
fn batch_loss_zero_ish_for_perfect_prediction() {
    let loss = default_batch_loss();
    let (h, w) = (16, 16);
    let stack = rand_stack(1, 3, h, w);
    // gt == the stack's first frame (arbitrary but self-consistent); pred == gt.
    let gt = stack
        .clone()
        .slice([0..1, 0..1, 0..3, 0..h, 0..w])
        .reshape([1, 3, h, w]);
    let pred = gt.clone();
    let occlusion = Tensor::<B, 4>::zeros([1, 1, h, w], &device());

    // masks concentrate fully on frame 0 everywhere, so the target alpha
    // distribution is a one-hot on frame 0; feed exactly that as `alpha`.
    let mut masks_data = vec![0f32; 3 * h * w];
    masks_data[..(h * w)].fill(1.0);
    let masks = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(masks_data, [3 * h * w]),
        &device(),
    )
    .reshape([1, 3, 1, h, w]);
    let alpha = alpha_matching_masks(&masks);

    let total: f32 = loss
        .forward_full(&pred, &alpha, &gt, &stack, &masks, &occlusion)
        .into_scalar();
    assert!(total.is_finite(), "loss is not finite: {total}");
    // Charbonnier floors at `charbonnier_eps` (default 0.001) and the
    // sharpness term can floor slightly above 0 depending on the random
    // stack's Laplacian content, so allow a small but tight tolerance well
    // below any genuinely-wrong-prediction loss.
    assert!(
        total < 0.05,
        "expected near-zero loss for a perfect prediction, got {total}"
    );
}

#[test]
fn batch_loss_sharpness_term_penalises_blur() {
    let loss = default_batch_loss();
    let (h, w) = (16, 16);

    // A stack with real high-frequency content (checkerboard-ish), so
    // blurring the prediction measurably reduces its sharpness relative to
    // the input frames.
    let mut data = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                data[c * h * w + y * w + x] = v;
            }
        }
    }
    let frame =
        Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(data, [3 * h * w]), &device())
            .reshape([1, 3, h, w]);
    let stack = frame.clone().unsqueeze_dim::<5>(1).repeat_dim(1, 2); // [1,2,3,h,w]
    let gt = frame.clone();
    let occlusion = Tensor::<B, 4>::zeros([1, 1, h, w], &device());
    let masks = Tensor::<B, 5>::ones([1, 2, 1, h, w], &device());
    let alpha = alpha_matching_masks(&masks);

    // Sharp prediction: identical to the checkerboard.
    let sharp_pred = frame;
    // Blurred prediction: flat mid-grey (zero high-frequency content).
    let blurred_pred = Tensor::<B, 4>::zeros([1, 3, h, w], &device()).add_scalar(0.5_f32);

    let sharp_loss: f32 = loss
        .forward_full(&sharp_pred, &alpha, &gt, &stack, &masks, &occlusion)
        .into_scalar();
    let blurred_loss: f32 = loss
        .forward_full(&blurred_pred, &alpha, &gt, &stack, &masks, &occlusion)
        .into_scalar();

    assert!(
        blurred_loss > sharp_loss,
        "blurred prediction loss ({blurred_loss}) should exceed sharp prediction loss ({sharp_loss})"
    );
}

#[test]
fn batch_loss_gate_supervision_prefers_correct_frame() {
    let loss = default_batch_loss();
    let (h, w) = (8, 8);

    // 2-frame stack; masks say frame 1 is in focus everywhere (frame 0 is not).
    let stack = rand_stack(1, 2, h, w);
    let gt = stack
        .clone()
        .slice([0..1, 1..2, 0..3, 0..h, 0..w])
        .reshape([1, 3, h, w]);
    let pred = gt.clone();
    let occlusion = Tensor::<B, 4>::zeros([1, 1, h, w], &device());

    let mut masks_data = vec![0f32; 2 * h * w];
    masks_data[(h * w)..].fill(1.0); // frame 1 = 1.0, frame 0 = 0.0
    let masks = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(masks_data, [2 * h * w]),
        &device(),
    )
    .reshape([1, 2, 1, h, w]);

    // alpha favouring frame 1 (matches target) vs. alpha favouring frame 0
    // (opposite of target).
    let mut alpha_correct_data = vec![0f32; 2 * h * w];
    alpha_correct_data[(h * w)..].fill(1.0);
    let alpha_correct = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(alpha_correct_data, [2 * h * w]),
        &device(),
    )
    .reshape([1, 2, 1, h, w]);

    let mut alpha_wrong_data = vec![0f32; 2 * h * w];
    alpha_wrong_data[..(h * w)].fill(1.0);
    let alpha_wrong = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(alpha_wrong_data, [2 * h * w]),
        &device(),
    )
    .reshape([1, 2, 1, h, w]);

    let loss_correct: f32 = loss
        .forward_full(&pred, &alpha_correct, &gt, &stack, &masks, &occlusion)
        .into_scalar();
    let loss_wrong: f32 = loss
        .forward_full(&pred, &alpha_wrong, &gt, &stack, &masks, &occlusion)
        .into_scalar();

    assert!(
        loss_correct < loss_wrong,
        "loss favouring the correct frame ({loss_correct}) should be less than favouring the \
         wrong frame ({loss_wrong})"
    );
}

#[test]
fn batch_loss_occlusion_downweights_edges() {
    let loss = default_batch_loss();
    let (h, w) = (8, 8);

    let stack = rand_stack(1, 2, h, w);
    let gt = rand4([1, 3, h, w]);
    let masks = rand5_masks(1, 2, h, w);
    let alpha = alpha_matching_masks(&masks);

    // A fixed, non-zero prediction error, injected either entirely inside
    // the occlusion region (top half) or entirely outside it (bottom half).
    let mut err_inside = vec![0f32; 3 * h * w];
    let mut err_outside = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let idx = c * h * w + y * w + x;
                if y < h / 2 {
                    err_inside[idx] = 0.5;
                } else {
                    err_outside[idx] = 0.5;
                }
            }
        }
    }
    let gt_data: Vec<f32> = gt.clone().into_data().iter::<f32>().collect();
    let pred_inside_data: Vec<f32> = gt_data
        .iter()
        .zip(&err_inside)
        .map(|(g, e)| g + e)
        .collect();
    let pred_outside_data: Vec<f32> = gt_data
        .iter()
        .zip(&err_outside)
        .map(|(g, e)| g + e)
        .collect();
    let pred_inside = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(pred_inside_data, [3 * h * w]),
        &device(),
    )
    .reshape([1, 3, h, w]);
    let pred_outside = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(pred_outside_data, [3 * h * w]),
        &device(),
    )
    .reshape([1, 3, h, w]);

    // Occlusion mask: top half = 1 (edge), bottom half = 0.
    let mut occ_data = vec![0f32; h * w];
    for y in 0..(h / 2) {
        for x in 0..w {
            occ_data[y * w + x] = 1.0;
        }
    }
    let occlusion =
        Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(occ_data, [h * w]), &device())
            .reshape([1, 1, h, w]);

    let loss_inside: f32 = loss
        .forward_full(&pred_inside, &alpha, &gt, &stack, &masks, &occlusion)
        .into_scalar();
    let loss_outside: f32 = loss
        .forward_full(&pred_outside, &alpha, &gt, &stack, &masks, &occlusion)
        .into_scalar();

    assert!(
        loss_inside < loss_outside,
        "identical-magnitude error inside the occlusion region ({loss_inside}) should score \
         less than the same error outside it ({loss_outside})"
    );
}
