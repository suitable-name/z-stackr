#![expect(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    reason = "synthetic-image formulas read better without mul_add and with plain casts"
)]

//! Neural alignment in the pipeline (D18): the network's matrices as a seed
//! source, the hybrid form (they seed the bounded classical refinement), the
//! direct form (they are applied as they are), and the fallbacks. The network
//! itself is not run here (no trained weights ship); the matrices are the
//! ground truth a perfect network would return.

use stacker_align::{Matrix3, transform::warp_image};
use stacker_core::{image::PlanarImage, settings::AlignmentModeSetting};
use stacker_pipeline::{
    NeverCancel, NoProgress,
    align::{
        AlignedFrame, AlignmentSettings, Fallback, FrameFlow, PrecomputedSeeds, align_sequence,
        align_sequence_seeded,
    },
    compute::Compute,
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

fn translation(x: f32, y: f32) -> Matrix3<f32> {
    let mut m = Matrix3::identity();
    m[(0, 2)] = x;
    m[(1, 2)] = y;
    m
}

struct Delivered {
    matrix: Matrix3<f32>,
    fallback: Option<Fallback>,
    luma: Vec<f32>,
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
        _common: &[bool],
    ) -> Result<(), String> {
        self.out.push(Delivered {
            matrix: aligned.matrix_applied,
            fallback: aligned.fallback,
            luma: aligned.frame.luma,
        });
        Ok(())
    }
}

/// Run the sequence with `seeds` standing in for the network.
fn run_with_seeds(
    frames: &[PlanarImage<f32>],
    settings: &AlignmentSettings,
    seeds: &PrecomputedSeeds,
) -> VecFlow {
    let mut flow = VecFlow::new(frames);
    align_sequence_seeded(
        &frames[0],
        frames.len(),
        settings,
        Some(seeds),
        &NeverCancel,
        &NoProgress,
        &mut flow,
        &mut Compute::cpu(),
    )
    .expect("the run succeeds");
    flow
}

fn stack() -> Vec<PlanarImage<f32>> {
    vec![texture(0.0, 0.0), texture(3.0, 2.0), texture(-4.0, 1.0)]
}

/// The matrices of the stack above, reference → frame, with frame 0 absent as
/// the pre-pass returns it.
fn truth() -> Vec<Option<Matrix3<f32>>> {
    vec![
        None,
        Some(translation(3.0, 2.0)),
        Some(translation(-4.0, 1.0)),
    ]
}

/// `neural_refine_classically = false`: the network's matrix is the answer. The
/// applied matrix is its inverse and the frame is the warp by it, bit for bit;
/// no refinement and no identity gate run.
#[test]
fn the_direct_form_applies_the_matrices_as_they_are() {
    let frames = stack();
    let settings =
        AlignmentSettings::new(AlignmentModeSetting::Neural, false).with_neural_refinement(false);
    let flow = run_with_seeds(&frames, &settings, &PrecomputedSeeds(truth()));
    assert_eq!(flow.out.len(), 2);
    for (i, delivered) in flow.out.iter().enumerate() {
        let seed = truth()[i + 1].expect("a seed");
        let applied = seed.try_inverse().expect("invertible");
        assert_eq!(delivered.fallback, None, "frame {}", i + 1);
        assert_eq!(delivered.matrix, applied, "frame {}", i + 1);
        let expected = warp_image(&frames[i + 1], &applied).expect("an invertible warp");
        assert_eq!(delivered.luma, expected.luma, "frame {}", i + 1);
    }
}

/// A direct run does not second-guess the network: a wild but finite matrix is
/// applied (the sanity filter is the classical seed's), a missing one is a
/// fallback to the identity.
#[test]
fn the_direct_form_flags_a_missing_matrix_and_trusts_a_finite_one() {
    let frames = stack();
    let settings =
        AlignmentSettings::new(AlignmentModeSetting::Neural, false).with_neural_refinement(false);
    let seeds = PrecomputedSeeds(vec![None, None, Some(translation(100.0, 0.0))]);
    let flow = run_with_seeds(&frames, &settings, &seeds);
    assert_eq!(flow.out[0].fallback, Some(Fallback::Seed));
    assert_eq!(flow.out[0].matrix, Matrix3::identity());
    assert_eq!(flow.out[1].fallback, None);
    assert!((flow.out[1].matrix[(0, 2)] + 100.0).abs() < 1e-3);
}

/// The hybrid form: a network matrix that is off by a pixel seeds the bounded
/// similarity refinement, which lands on the true shift.
#[test]
fn the_hybrid_form_refines_the_networks_matrices() {
    let frames = stack();
    let settings = AlignmentSettings::new(AlignmentModeSetting::Neural, false);
    assert!(settings.neural_refine_classically);
    let off = PrecomputedSeeds(vec![
        None,
        Some(translation(4.0, 2.5)),
        Some(translation(-3.2, 0.4)),
    ]);
    let flow = run_with_seeds(&frames, &settings, &off);
    for (i, (dx, dy)) in [(3.0, 2.0), (-4.0, 1.0)].into_iter().enumerate() {
        let m = flow.out[i].matrix;
        assert_eq!(flow.out[i].fallback, None);
        assert!(
            (m[(0, 2)] + dx).abs() < 0.2,
            "frame {}: tx {}",
            i + 1,
            m[(0, 2)]
        );
        assert!(
            (m[(1, 2)] + dy).abs() < 0.2,
            "frame {}: ty {}",
            i + 1,
            m[(1, 2)]
        );
    }
}

/// A network matrix beyond a quarter of the frame is not a sane seed: the
/// refinement starts from the identity instead and still finds the shift.
#[test]
fn the_hybrid_form_ignores_an_absurd_matrix() {
    let frames = stack();
    let settings = AlignmentSettings::new(AlignmentModeSetting::Neural, false);
    let absurd = PrecomputedSeeds(vec![
        None,
        Some(translation(100.0, 0.0)),
        Some(translation(0.0, -90.0)),
    ]);
    let flow = run_with_seeds(&frames, &settings, &absurd);
    let m = flow.out[0].matrix;
    assert!((m[(0, 2)] + 3.0).abs() < 0.5, "tx {}", m[(0, 2)]);
    assert!((m[(1, 2)] + 2.0).abs() < 0.5, "ty {}", m[(1, 2)]);
}

/// The neural mode without a model (or in a build without the network) aligns
/// with the registration mode, with a warning, rather than failing the run.
#[test]
fn neural_without_a_model_falls_back_to_registration() {
    let frames = stack();
    let settings = AlignmentSettings::new(AlignmentModeSetting::Neural, false);
    assert!(settings.align_model.is_none());
    let mut flow = VecFlow::new(&frames);
    align_sequence(
        &frames[0],
        frames.len(),
        &settings,
        &NeverCancel,
        &NoProgress,
        &mut flow,
    )
    .expect("the run succeeds");
    assert_eq!(flow.out.len(), 2);
    let m = flow.out[0].matrix;
    assert!((m[(0, 2)] + 3.0).abs() < 0.5, "tx {}", m[(0, 2)]);
    assert!((m[(1, 2)] + 2.0).abs() < 0.5, "ty {}", m[(1, 2)]);
}

#[cfg(feature = "nn")]
mod with_the_network {
    use std::path::PathBuf;

    use stacker_nn::{InferDevice, ModelEntry, ModelManifest, ModelSize};
    use stacker_pipeline::align::{AlignModel, NeuralError, neural_seeds};

    use super::*;

    fn missing_model() -> AlignModel {
        AlignModel::new(
            ModelEntry {
                name: "missing".to_owned(),
                weights_path: PathBuf::from("this/model/does/not/exist.mpk"),
                manifest: ModelManifest::from_size_align(ModelSize::Xs),
            },
            InferDevice::Cpu,
        )
    }

    /// A model whose weights cannot be loaded is a model error, reported before
    /// any frame is read.
    #[test]
    fn missing_weights_are_a_model_error() {
        let reference = texture(0.0, 0.0);
        let result = neural_seeds(
            &missing_model(),
            &reference,
            3,
            |_| None::<Result<PlanarImage<f32>, String>>,
            &NeverCancel,
        );
        assert!(matches!(result, Err(NeuralError::Model(_))), "{result:?}");
    }

    /// The run with an unloadable model aligns with the registration mode.
    #[test]
    fn an_unloadable_model_falls_back_to_registration() {
        let frames = stack();
        let settings = AlignmentSettings::new(AlignmentModeSetting::Neural, false)
            .with_model(Some(missing_model()));
        assert_eq!(settings, settings.clone());
        let mut flow = VecFlow::new(&frames);
        align_sequence(
            &frames[0],
            frames.len(),
            &settings,
            &NeverCancel,
            &NoProgress,
            &mut flow,
        )
        .expect("the run succeeds");
        let m = flow.out[0].matrix;
        assert!((m[(0, 2)] + 3.0).abs() < 0.5, "tx {}", m[(0, 2)]);
    }
}
