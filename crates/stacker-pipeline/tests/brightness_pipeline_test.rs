#![expect(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    reason = "synthetic-image formulas read better without mul_add and with plain casts"
)]

//! `correct_brightness` (D6) in the alignment pipeline: the aligned frames take
//! the reference's exposure (a pure gain in linear light), the fit looks only at
//! the pixels the warp covered, the reference and mode `none` stay untouched.

use stacker_core::{image::PlanarImage, settings::AlignmentModeSetting};
use stacker_pipeline::{
    NeverCancel, NoProgress,
    align::{AlignedFrame, AlignmentSettings, FrameFlow, align_frame, align_sequence},
};

const W: usize = 128;
const H: usize = 128;

fn texture(dx: f32, dy: f32, gain: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f32 - dx, y as f32 - dy);
            img.luma[y * W + x] = gain
                * (0.5
                    + 0.2 * (0.23 * fx).sin() * (0.19 * fy).cos()
                    + 0.15 * (0.07 * (fx + fy)).sin()
                    + 0.1 * (0.31 * fx - 0.27 * fy).cos()
                    + 0.08 * (0.5 * fx).sin() * (0.43 * fy).sin());
        }
    }
    img
}

/// Mean luma away from the borders (the warp leaves none there).
fn interior_mean(img: &PlanarImage<f32>) -> f32 {
    let margin = 24;
    let mut sum = 0.0_f32;
    let mut n = 0.0_f32;
    for y in margin..H - margin {
        for x in margin..W - margin {
            sum += img.luma[y * W + x];
            n += 1.0;
        }
    }
    sum / n
}

const fn settings(brightness: bool) -> AlignmentSettings {
    AlignmentSettings::new(AlignmentModeSetting::Translation, false).with_brightness(brightness)
}

/// A frame shot at 70 % of the reference's exposure comes out at the
/// reference's exposure with the fit on, and stays dark with it off.
#[test]
fn the_aligned_frame_takes_the_reference_exposure() {
    let reference = texture(0.0, 0.0, 1.0);
    let dark = texture(2.0, 1.0, 0.7);
    let target = interior_mean(&reference);

    let off = align_frame(&reference, dark.clone(), &settings(false), None);
    assert!(
        target - interior_mean(&off.frame) > 0.1,
        "without the fit the frame stays dark: {} vs {target}",
        interior_mean(&off.frame)
    );

    let on = align_frame(&reference, dark, &settings(true), None);
    assert_eq!(on.fallback, None);
    assert!(
        (interior_mean(&on.frame) - target).abs() < 0.01,
        "with the fit the exposure matches: {} vs {target}",
        interior_mean(&on.frame)
    );
}

/// A brighter frame is pulled down the same way.
#[test]
fn a_brighter_frame_is_pulled_down() {
    let reference = texture(0.0, 0.0, 1.0);
    let bright = texture(-2.0, 2.0, 1.25);
    let target = interior_mean(&reference);
    let on = align_frame(&reference, bright, &settings(true), None);
    assert!((interior_mean(&on.frame) - target).abs() < 0.015);
}

/// Mode `none` uses every frame as it is, brightness fit or not.
#[test]
fn mode_none_leaves_the_exposure_alone() {
    let reference = texture(0.0, 0.0, 1.0);
    let dark = texture(2.0, 1.0, 0.7);
    let none = AlignmentSettings::new(AlignmentModeSetting::None, false).with_brightness(true);
    let aligned = align_frame(&reference, dark.clone(), &none, None);
    assert_eq!(aligned.frame.luma, dark.luma);
}

/// The fit looks at the pixels the warp covered only: a frame that is shifted far
/// enough to lose a wide black border still matches the reference's exposure.
#[test]
fn the_black_border_does_not_bias_the_fit() {
    let reference = texture(0.0, 0.0, 1.0);
    let dark = texture(6.0, 5.0, 0.8);
    let on = align_frame(&reference, dark, &settings(true), None);
    assert!(
        (interior_mean(&on.frame) - interior_mean(&reference)).abs() < 0.015,
        "{}",
        interior_mean(&on.frame)
    );
}

struct Collect {
    frames: Vec<Option<PlanarImage<f32>>>,
    means: Vec<f32>,
}

impl FrameFlow<String> for Collect {
    fn load(&mut self, index: usize) -> Result<PlanarImage<f32>, String> {
        self.frames[index]
            .take()
            .ok_or_else(|| format!("frame {index} requested twice"))
    }

    fn deliver(
        &mut self,
        _index: usize,
        aligned: AlignedFrame,
        _common: &[bool],
    ) -> Result<(), String> {
        self.means.push(interior_mean(&aligned.frame));
        Ok(())
    }
}

/// A stack whose exposure drifts 10 % per frame is delivered at the reference's
/// exposure, frame after frame (the target is built once from the reference).
#[test]
fn an_exposure_ramp_is_flattened_across_the_stack() {
    let frames: Vec<PlanarImage<f32>> = (0..4)
        .map(|i| texture(1.5 * i as f32, -(i as f32), 1.0 - 0.1 * i as f32))
        .collect();
    let target = interior_mean(&frames[0]);
    let mut flow = Collect {
        frames: frames.iter().cloned().map(Some).collect(),
        means: Vec::new(),
    };
    align_sequence(
        &frames[0],
        frames.len(),
        &settings(true),
        &NeverCancel,
        &NoProgress,
        &mut flow,
    )
    .expect("the run succeeds");
    assert_eq!(flow.means.len(), 3);
    for (i, mean) in flow.means.iter().enumerate() {
        assert!(
            (mean - target).abs() < 0.015,
            "frame {}: {mean} vs {target}",
            i + 1
        );
    }
}
