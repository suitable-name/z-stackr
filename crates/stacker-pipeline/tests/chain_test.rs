#![expect(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    reason = "synthetic-image formulas read better without mul_add and with plain casts"
)]

//! `chain_alignment` (D7): every frame is refined against the raw previous
//! frame, the steps are composed into the cumulative matrix, and the coverage
//! mask is the cumulative one (not the last step's).

use stacker_align::{Matrix3, transform::warp_image};
use stacker_core::{image::PlanarImage, settings::AlignmentModeSetting};
use stacker_pipeline::{
    NeverCancel, NoProgress,
    align::{AlignedFrame, AlignmentSettings, FrameFlow, Refiner, align_dispatch, align_sequence},
};

const W: usize = 128;
const H: usize = 128;

fn texture(dx: f32, dy: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f32 - dx, y as f32 - dy);
            img.luma[y * W + x] = 0.5
                + 0.2 * (0.23 * fx).sin() * (0.19 * fy).cos()
                + 0.15 * (0.07 * (fx + fy)).sin()
                + 0.1 * (0.31 * fx - 0.27 * fy).cos()
                + 0.08 * (0.5 * fx).sin() * (0.43 * fy).sin();
        }
    }
    img
}

struct Delivered {
    matrix: Matrix3<f32>,
    coverage: Vec<bool>,
    luma: Vec<f32>,
    common: Vec<bool>,
}

struct VecFlow {
    frames: Vec<Option<PlanarImage<f32>>>,
    out: Vec<Delivered>,
}

impl VecFlow {
    fn new(frames: &[PlanarImage<f32>]) -> Self {
        Self {
            frames: frames.iter().cloned().map(Some).collect(),
            out: Vec::new(),
        }
    }
}

impl FrameFlow<String> for VecFlow {
    fn load(&mut self, index: usize) -> Result<PlanarImage<f32>, String> {
        self.frames[index]
            .take()
            .ok_or_else(|| format!("frame {index} requested twice"))
    }

    fn deliver(
        &mut self,
        _index: usize,
        aligned: AlignedFrame,
        common: &[bool],
    ) -> Result<(), String> {
        self.out.push(Delivered {
            matrix: aligned.matrix_applied,
            coverage: aligned.coverage,
            luma: aligned.frame.luma,
            common: common.to_vec(),
        });
        Ok(())
    }
}

fn run(frames: &[PlanarImage<f32>], settings: &AlignmentSettings) -> VecFlow {
    let mut flow = VecFlow::new(frames);
    align_sequence(
        &frames[0],
        frames.len(),
        settings,
        &NeverCancel,
        &NoProgress,
        &mut flow,
    )
    .expect("the run succeeds");
    flow
}

const fn chained(mode: AlignmentModeSetting) -> AlignmentSettings {
    AlignmentSettings::new(mode, false).with_chain(true)
}

/// Frames 0, 1, 2 shifted by 0, (3, 2) and (6, 4): the chain finds the 3-pixel
/// steps, and the cumulative matrix of frame 2 (the inverse of what was applied)
/// is the 6-pixel shift to within 0.05 px.
#[test]
fn a_chain_of_three_frames_reconstructs_the_cumulative_shift() {
    let frames = vec![texture(0.0, 0.0), texture(3.0, 2.0), texture(6.0, 4.0)];
    for mode in [
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Registration,
    ] {
        let flow = run(&frames, &chained(mode));
        assert_eq!(flow.out.len(), 2);

        let cumulative = |i: usize| {
            flow.out[i]
                .matrix
                .try_inverse()
                .expect("an invertible applied matrix")
        };
        let c1 = cumulative(0);
        let c2 = cumulative(1);
        assert!(
            (c1[(0, 2)] - 3.0).abs() < 0.05,
            "{mode:?}: C1 tx {}",
            c1[(0, 2)]
        );
        assert!(
            (c1[(1, 2)] - 2.0).abs() < 0.05,
            "{mode:?}: C1 ty {}",
            c1[(1, 2)]
        );
        assert!(
            (c2[(0, 2)] - 6.0).abs() < 0.05,
            "{mode:?}: C2 tx {}",
            c2[(0, 2)]
        );
        assert!(
            (c2[(1, 2)] - 4.0).abs() < 0.05,
            "{mode:?}: C2 ty {}",
            c2[(1, 2)]
        );
        assert!(
            (c2[(0, 0)] - 1.0).abs() < 0.01,
            "{mode:?}: scale {}",
            c2[(0, 0)]
        );
    }
}

/// The coverage of frame 2 is that of the cumulative 6-pixel shift: the columns
/// a 3-pixel step would still claim are already outside. The common mask of the
/// run is the intersection.
#[test]
fn the_chain_coverage_is_the_cumulative_one() {
    let frames = vec![texture(0.0, 0.0), texture(3.0, 2.0), texture(6.0, 4.0)];
    let flow = run(&frames, &chained(AlignmentModeSetting::Registration));

    let row = 10;
    let near_edge = row * W + (W - 5);
    // C2 maps (W - 5, row) to x = W + 1: outside. A step of 3 would give W - 2.
    assert!(
        !flow.out[1].coverage[near_edge],
        "the cumulative shift uncovers it"
    );
    assert!(
        flow.out[0].coverage[near_edge],
        "frame 1 is only 3 pixels off, so it is covered"
    );
    assert!(flow.out[1].coverage[(H / 2) * W + W / 2]);
    // The running mask only ever loses pixels, and includes frame 2's loss.
    assert!(!flow.out[1].common[near_edge]);
    assert!(flow.out[0].common[(H / 2) * W + W / 2]);
}

/// The first chain step has the reference itself as its reference, so a
/// two-frame stack aligns exactly as without the chain.
#[test]
fn a_two_frame_chain_is_the_fixed_reference_run() {
    let frames = vec![texture(0.0, 0.0), texture(3.0, 2.0)];
    for mode in [
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Registration,
        AlignmentModeSetting::Affine,
    ] {
        let fixed = run(&frames, &AlignmentSettings::new(mode, false));
        let chain = run(&frames, &chained(mode));
        assert_eq!(fixed.out[0].matrix, chain.out[0].matrix, "{mode:?}");
        assert_eq!(fixed.out[0].luma, chain.out[0].luma, "{mode:?}");
        assert_eq!(fixed.out[0].coverage, chain.out[0].coverage, "{mode:?}");
    }
}

/// A chain follows a drift that no single step has to bridge: 5 steps of 2
/// pixels reach 10 pixels, beyond the 5 % search window (6.4 px) of a direct
/// alignment to frame 0 from the identity, and the chain still lands on it.
#[test]
fn a_chain_follows_a_drift_longer_than_the_search_window() {
    let frames: Vec<PlanarImage<f32>> = (0..6).map(|i| texture(2.0 * i as f32, 0.0)).collect();
    let flow = run(&frames, &chained(AlignmentModeSetting::Translation));
    assert_eq!(flow.out.len(), 5);
    let last = flow.out[4]
        .matrix
        .try_inverse()
        .expect("an invertible applied matrix");
    assert!((last[(0, 2)] - 10.0).abs() < 0.25, "C5 tx {}", last[(0, 2)]);
    assert!(last[(1, 2)].abs() < 0.25, "C5 ty {}", last[(1, 2)]);

    // The aligned last frame matches the reference where both have data.
    let reference = &frames[0];
    let mut sum = 0.0_f32;
    let mut n = 0.0_f32;
    for y in 20..H - 20 {
        for x in 20..W - 30 {
            sum += (flow.out[4].luma[y * W + x] - reference.luma[y * W + x]).abs();
            n += 1.0;
        }
    }
    assert!(sum / n < 0.03, "mean error {}", sum / n);
}

/// `none` ignores the chain; so does a direct (network) dispatch.
#[test]
fn the_chain_changes_nothing_without_a_refinement() {
    let frames = vec![texture(0.0, 0.0), texture(3.0, 2.0)];
    let flow = run(&frames, &chained(AlignmentModeSetting::None));
    assert_eq!(flow.out[0].luma, frames[1].luma);
    assert!(flow.out[0].coverage.iter().all(|&c| c));
    assert_eq!(
        align_dispatch(AlignmentModeSetting::None, false).refiner,
        Refiner::Skip
    );
}

/// The unwarped source of a frame, to confirm the helper used above: warping
/// with the cumulative inverse is what the pipeline applied.
#[test]
fn the_applied_matrix_is_the_inverse_of_the_cumulative_one() {
    let frames = vec![texture(0.0, 0.0), texture(3.0, 2.0), texture(6.0, 4.0)];
    let flow = run(&frames, &chained(AlignmentModeSetting::Translation));
    let rewarped = warp_image(&frames[2], &flow.out[1].matrix).expect("an invertible warp");
    assert_eq!(rewarped.luma, flow.out[1].luma);
}
