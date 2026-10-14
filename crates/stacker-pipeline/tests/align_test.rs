#![expect(
    clippy::suboptimal_flops,
    reason = "synthetic-image formulas read better without mul_add"
)]

//! `align`: the mode table, `align_frame` on a synthetic shift, and the
//! sequence driver.

use std::cell::Cell;

use stacker_align::{Matrix3, Motion, refine::LumaPyramid};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, OptimizerSetting},
};
use stacker_pipeline::{
    Cancel, NeverCancel, NoProgress,
    align::{
        AlignedFrame, AlignmentSettings, Dispatch, Fallback, FrameFlow, Refiner, SequenceError,
        Warp, align_dispatch, align_frame, align_frame_with_pyramid, align_sequence, is_sane_seed,
    },
};

const SIZE: u16 = 128;

/// A smooth multi-scale texture, evaluated at `(x - dx, y - dy)`: the frame is
/// the reference's content moved by `(dx, dy)` pixels, with no resampling error.
fn texture(dx: f32, dy: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(usize::from(SIZE), usize::from(SIZE));
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (fx, fy) = (f32::from(x) - dx, f32::from(y) - dy);
            let v = 0.5
                + 0.2 * (0.23 * fx).sin() * (0.19 * fy).cos()
                + 0.15 * (0.07 * (fx + fy)).sin()
                + 0.1 * (0.31 * fx - 0.27 * fy).cos()
                + 0.08 * (0.5 * fx).sin() * (0.43 * fy).sin();
            img.luma[usize::from(y) * usize::from(SIZE) + usize::from(x)] = v;
        }
    }
    img
}

/// Mean absolute luma difference over the interior (a margin away from the
/// borders, where the warp has no data).
fn interior_error(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> f32 {
    let margin = 16;
    let mut sum = 0.0_f32;
    let mut n = 0_u32;
    for y in margin..a.height - margin {
        for x in margin..a.width - margin {
            sum += (a.luma[y * a.width + x] - b.luma[y * a.width + x]).abs();
            n += 1;
        }
    }
    sum / f32::from(u16::try_from(n).unwrap_or(u16::MAX)).max(1.0)
}

/// The classic three switches with the simplex optimiser (what the tests below
/// were written against), no brightness fit.
const fn settings(mode: AlignmentModeSetting, bounded: bool) -> AlignmentSettings {
    AlignmentSettings::new(mode, bounded)
}

// ── the mode table ────────────────────────────────────────────────────────────

/// FLOW-1 / GUI-5 / ALN-2 / D3: every `(mode, bounded)` pair maps to one
/// algorithm, the same for both apps. Ten rows (the neural mode is always the
/// bounded similarity refinement of the network's seed).
#[test]
fn the_dispatch_table_has_ten_rows() {
    use AlignmentModeSetting::{Affine, Neural, None, Registration, Translation};
    let motion = |motion, bounded| Refiner::Motion { motion, bounded };
    let rows = [
        (None, false, Refiner::Skip, Warp::None),
        (None, true, Refiner::Skip, Warp::None),
        (
            Translation,
            false,
            motion(Motion::Translation, false),
            Warp::Bicubic,
        ),
        (
            Translation,
            true,
            motion(Motion::Translation, true),
            Warp::Bicubic,
        ),
        (
            Registration,
            false,
            motion(Motion::Similarity, false),
            Warp::Bicubic,
        ),
        (
            Registration,
            true,
            motion(Motion::Similarity, true),
            Warp::Bicubic,
        ),
        (Affine, false, motion(Motion::Affine, false), Warp::Bicubic),
        (Affine, true, motion(Motion::Affine, true), Warp::Bicubic),
        (
            Neural,
            false,
            motion(Motion::Similarity, true),
            Warp::Bicubic,
        ),
        (
            Neural,
            true,
            motion(Motion::Similarity, true),
            Warp::Bicubic,
        ),
    ];
    for (mode, flag, refiner, warp) in rows {
        assert_eq!(
            align_dispatch(mode, flag),
            Dispatch { refiner, warp },
            "{mode:?} bounded={flag}"
        );
    }
}

#[test]
fn settings_convert_from_the_stacking_settings() {
    let s = stacker_core::settings::StackingSettings {
        alignment_mode: AlignmentModeSetting::Translation,
        bounded_search: true,
        akaze_seeding: true,
        optimizer: OptimizerSetting::LucasKanade,
        correct_brightness: false,
        chain_alignment: true,
        neural_refine_classically: false,
        crop_to_common_area: false,
        ..stacker_core::settings::StackingSettings::default()
    };
    let a = AlignmentSettings::from(&s);
    assert_eq!(a.mode, AlignmentModeSetting::Translation);
    assert!(a.bounded && a.akaze_seeding);
    assert_eq!(a.optimizer, OptimizerSetting::LucasKanade);
    assert!(!a.correct_brightness && a.chain_alignment && !a.neural_refine_classically);
    assert!(a.clamp_border, "no common-area crop keeps the border");
    assert!(a.align_model.is_none());
}

/// The application defaults are those of `StackingSettings`: brightness on,
/// Lucas-Kanade first, no chain, the border cropped; `new` is the old minimal
/// form (simplex, no brightness).
#[test]
fn the_defaults_follow_the_stacking_settings() {
    let d = AlignmentSettings::default();
    assert_eq!(d.mode, AlignmentModeSetting::Registration);
    assert_eq!(d.optimizer, OptimizerSetting::Auto);
    assert!(d.correct_brightness && !d.chain_alignment && !d.clamp_border);

    let n = AlignmentSettings::new(AlignmentModeSetting::Affine, true);
    assert_eq!(n.optimizer, OptimizerSetting::NelderMead);
    assert!(!n.correct_brightness && !n.chain_alignment && !n.akaze_seeding);
    assert!(n.bounded);
}

// ── align_frame ───────────────────────────────────────────────────────────────

fn assert_recovers_the_shift(
    aligned: &AlignedFrame,
    reference: &PlanarImage<f32>,
    shifted: &PlanarImage<f32>,
) {
    assert_eq!(aligned.fallback, None);
    let m = aligned.matrix_applied;
    assert!((m[(0, 2)].abs() - 3.0).abs() < 0.6, "tx {}", m[(0, 2)]);
    assert!((m[(1, 2)].abs() - 2.0).abs() < 0.6, "ty {}", m[(1, 2)]);
    // Aligning moves the shifted content back, so the signs are opposite.
    assert!(m[(0, 2)] < 0.0 && m[(1, 2)] < 0.0, "{m}");

    let before = interior_error(reference, shifted);
    let after = interior_error(reference, &aligned.frame);
    assert!(
        after < 0.25 * before,
        "the aligned frame must match the reference far better: {after} vs {before}"
    );
    // The border the warp could not fill is not covered; the centre is.
    let w = reference.width;
    assert!(
        !aligned.coverage[w * w - 1],
        "the far corner has no source data"
    );
    assert!(aligned.coverage[(w / 2) * w + w / 2]);
}

#[test]
fn translation_mode_recovers_a_shift_and_keeps_scale_and_rotation_fixed() {
    let reference = texture(0.0, 0.0);
    let shifted = texture(3.0, 2.0);
    for bounded in [false, true] {
        let aligned = align_frame(
            &reference,
            shifted.clone(),
            &settings(AlignmentModeSetting::Translation, bounded),
            None,
        );
        assert_recovers_the_shift(&aligned, &reference, &shifted);
        // D3: translation fits shift and one uniform scale (focus breathing), never
        // rotation, bounded or not. The content is a pure shift, so the scale it
        // finds stays at one; both axes carry the same scale.
        let m = aligned.matrix_applied;
        assert!((m[(0, 0)] - 1.0).abs() < 0.01, "bounded={bounded}: {m}");
        assert!((m[(1, 1)] - 1.0).abs() < 0.01, "bounded={bounded}: {m}");
        assert!(
            (m[(0, 0)] - m[(1, 1)]).abs() < 1e-4,
            "bounded={bounded}: {m}"
        );
        assert!(
            m[(0, 1)].abs() < 1e-3 && m[(1, 0)].abs() < 1e-3,
            "bounded={bounded}: {m}"
        );
    }
}

#[test]
fn affine_mode_recovers_the_same_shift() {
    let reference = texture(0.0, 0.0);
    let shifted = texture(3.0, 2.0);
    let aligned = align_frame(
        &reference,
        shifted.clone(),
        &settings(AlignmentModeSetting::Affine, false),
        None,
    );
    assert_recovers_the_shift(&aligned, &reference, &shifted);
}

#[test]
fn none_mode_returns_the_frame_untouched() {
    let reference = texture(0.0, 0.0);
    let shifted = texture(3.0, 2.0);
    let aligned = align_frame(
        &reference,
        shifted.clone(),
        &settings(AlignmentModeSetting::None, false),
        // Even a (sane) seed must not be applied.
        Some(translation(2.0, 2.0)),
    );
    assert_eq!(aligned.matrix_applied, Matrix3::identity());
    assert_eq!(aligned.frame.luma, shifted.luma);
    assert_eq!(aligned.fallback, None);
    assert!(aligned.coverage.iter().all(|&c| c));
    assert!((aligned.covered_fraction() - 1.0).abs() < 1e-12);
}

fn translation(x: f32, y: f32) -> Matrix3<f32> {
    let mut m = Matrix3::identity();
    m[(0, 2)] = x;
    m[(1, 2)] = y;
    m
}

/// A failed refinement is not an error: the frame falls back to the seed and is
/// flagged, and the matrix that is reported is the one that was applied.
#[test]
fn a_refinement_failure_falls_back_to_the_seed() {
    let reference = texture(0.0, 0.0);
    // A frame of another size makes every refiner reject the pair.
    let odd = PlanarImage::<f32>::new(64, 64);
    let aligned = align_frame(
        &reference,
        odd,
        &settings(AlignmentModeSetting::Affine, false),
        None,
    );
    assert_eq!(aligned.fallback, Some(Fallback::Seed));
    assert_eq!((aligned.frame.width, aligned.frame.height), (64, 64));
    assert_eq!(aligned.coverage.len(), 64 * 64);
}

#[test]
fn only_sane_seeds_are_used() {
    let (w, h) = (100, 80);
    assert!(is_sane_seed(&Matrix3::identity(), w, h));
    // GitHub's rule: a shift of up to a quarter of the frame is within the
    // refiners' search window (it was a whole frame before the port).
    assert!(is_sane_seed(&translation(25.0, -20.0), w, h));
    assert!(
        !is_sane_seed(&translation(26.0, 0.0), w, h),
        "shift beyond a quarter of the width"
    );
    assert!(
        !is_sane_seed(&translation(0.0, -21.0), w, h),
        "shift beyond a quarter of the height"
    );
    assert!(
        !is_sane_seed(&translation(500.0, 0.0), w, h),
        "shift beyond the frame"
    );
    let mut scaled = Matrix3::<f32>::identity();
    scaled[(0, 0)] = 3.0;
    assert!(!is_sane_seed(&scaled, w, h), "scale outside 0.5..2");
    let mut nan = Matrix3::<f32>::identity();
    nan[(1, 2)] = f32::NAN;
    assert!(!is_sane_seed(&nan, w, h));
}

// ── align_sequence ────────────────────────────────────────────────────────────

struct VecFlow {
    frames: Vec<Option<PlanarImage<f32>>>,
    delivered: Vec<usize>,
    masks: Vec<Vec<bool>>,
    out: Vec<Option<PlanarImage<f32>>>,
}

impl VecFlow {
    fn new(frames: Vec<PlanarImage<f32>>) -> Self {
        let n = frames.len();
        Self {
            frames: frames.into_iter().map(Some).collect(),
            delivered: Vec::new(),
            masks: Vec::new(),
            out: (0..n).map(|_| None).collect(),
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
        index: usize,
        aligned: AlignedFrame,
        common: &[bool],
    ) -> Result<(), String> {
        self.delivered.push(index);
        self.masks.push(common.to_vec());
        self.out[index] = Some(aligned.frame);
        Ok(())
    }
}

#[test]
fn the_driver_aligns_every_frame_and_intersects_the_coverage() {
    let reference = texture(0.0, 0.0);
    let frames = vec![
        reference.clone(),
        texture(3.0, 2.0),
        texture(-2.0, 3.0),
        texture(1.0, -3.0),
    ];
    let mut flow = VecFlow::new(frames.clone());
    let outcome = align_sequence(
        &reference,
        frames.len(),
        &settings(AlignmentModeSetting::Translation, false),
        &NeverCancel,
        &NoProgress,
        &mut flow,
    )
    .unwrap();

    assert_eq!(flow.delivered, [1, 2, 3], "frame 0 is the reference");
    assert_eq!(outcome.fallbacks, Vec::new());
    // The running mask only ever loses pixels.
    for pair in flow.masks.windows(2) {
        assert!(
            pair[0]
                .iter()
                .zip(&pair[1])
                .all(|(before, after)| *before || !*after)
        );
    }
    assert_eq!(
        outcome.coverage.mask(),
        flow.masks.last().unwrap().as_slice()
    );
    assert!(!outcome.coverage.mask()[0]);
    let centre = usize::from(SIZE) * usize::from(SIZE) / 2 + usize::from(SIZE) / 2;
    assert!(outcome.coverage.mask()[centre]);
    for (i, original) in frames.iter().enumerate().skip(1) {
        let aligned = flow.out[i].as_ref().unwrap();
        assert!(
            interior_error(&reference, aligned) < 0.5 * interior_error(&reference, original),
            "frame {i} improved"
        );
    }
}

/// PERF-1: the driver builds the reference's luma pyramid once, and the result
/// is bit for bit what the per-frame function (which builds its own) returns,
/// in every mode that reads the pyramid.
#[test]
fn the_driver_with_a_shared_reference_pyramid_matches_the_per_frame_result() {
    let reference = texture(0.0, 0.0);
    let frames = vec![reference.clone(), texture(3.0, 2.0)];
    for (mode, bounded) in [
        (AlignmentModeSetting::Translation, false),
        (AlignmentModeSetting::Translation, true),
        (AlignmentModeSetting::Affine, false),
        (AlignmentModeSetting::Affine, true),
    ] {
        let s = settings(mode, bounded);
        let mut flow = VecFlow::new(frames.clone());
        align_sequence(&reference, 2, &s, &NeverCancel, &NoProgress, &mut flow).unwrap();
        let from_driver = flow.out[1].take().unwrap();

        let per_frame = align_frame(&reference, frames[1].clone(), &s, None);
        let with_pyramid = align_frame_with_pyramid(
            &reference,
            &LumaPyramid::new(&reference),
            frames[1].clone(),
            &s,
            None,
        );
        assert_eq!(
            from_driver.luma, per_frame.frame.luma,
            "{mode:?} bounded={bounded}: driver vs per-frame"
        );
        assert_eq!(
            with_pyramid.frame.luma, per_frame.frame.luma,
            "{mode:?} bounded={bounded}: supplied pyramid vs own"
        );
        assert_eq!(with_pyramid.matrix_applied, per_frame.matrix_applied);
        assert_eq!(with_pyramid.coverage, per_frame.coverage);
    }
}

/// With `alignment_mode = none` the frames still flow through, untouched.
#[test]
fn the_driver_delivers_untouched_frames_in_none_mode() {
    let reference = texture(0.0, 0.0);
    let frames = vec![reference.clone(), texture(3.0, 2.0)];
    let mut flow = VecFlow::new(frames.clone());
    let outcome = align_sequence(
        &reference,
        2,
        &settings(AlignmentModeSetting::None, false),
        &NeverCancel,
        &NoProgress,
        &mut flow,
    )
    .unwrap();
    assert_eq!(flow.out[1].as_ref().unwrap().luma, frames[1].luma);
    assert!(outcome.coverage.mask().iter().all(|&c| c));
}

struct CancelAfter(Cell<usize>);

impl Cancel for CancelAfter {
    fn is_cancelled(&self) -> bool {
        let seen = self.0.get();
        self.0.set(seen + 1);
        seen >= 1
    }
}

#[test]
fn the_driver_stops_when_cancelled_and_reports_flow_errors() {
    let reference = texture(0.0, 0.0);
    let frames = vec![reference.clone(), texture(1.0, 1.0), texture(2.0, 2.0)];

    let mut flow = VecFlow::new(frames.clone());
    let result = align_sequence(
        &reference,
        3,
        &settings(AlignmentModeSetting::None, false),
        &CancelAfter(Cell::new(0)),
        &NoProgress,
        &mut flow,
    );
    assert!(matches!(result, Err(SequenceError::Cancelled)));
    assert_eq!(flow.delivered, [1], "stopped before the second frame");

    // A frame that cannot be loaded ends the run with the flow's own error.
    let mut flow = VecFlow::new(frames);
    flow.frames[2] = None;
    let result = align_sequence(
        &reference,
        3,
        &settings(AlignmentModeSetting::None, false),
        &NeverCancel,
        &NoProgress,
        &mut flow,
    );
    match result {
        Err(SequenceError::Flow(message)) => assert!(message.contains("frame 2"), "{message}"),
        other => panic!("expected a flow error, got {other:?}"),
    }
}

#[test]
fn a_single_frame_sequence_asks_for_nothing() {
    let reference = texture(0.0, 0.0);
    let mut flow = VecFlow::new(vec![reference.clone()]);
    let outcome = align_sequence(
        &reference,
        1,
        &settings(AlignmentModeSetting::Affine, false),
        &NeverCancel,
        &NoProgress,
        &mut flow,
    )
    .unwrap();
    assert_eq!(flow.delivered, Vec::<usize>::new());
    assert!(outcome.coverage.mask().iter().all(|&c| c));
}
