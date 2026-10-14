#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::many_single_char_names
)]

//! The coarse-biased Nelder-Mead level schedule with its full-resolution polish
//! (`refine::level_plan`): what the refiners recover with and without the
//! polish, and which levels, and how many evaluations of them, the objective
//! sees.
//!
//! The fixtures are analytic: the source is the reference texture evaluated at
//! the inverse-mapped coordinates, so no interpolation sits between the truth and
//! the objective's minimum.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use nalgebra::{Matrix3, Vector3};
use stacker_align::{
    device::{DeviceHandle, LevelDevice, Moments, ObjectiveDevice, ObjectiveSampler},
    refine::{
        DEFAULT_POLISH_ITERATIONS, LumaPyramid, RefineOptions, RegistrationParams, level_plan,
        params_to_matrix, refine_alignment_with_pyramid,
    },
};
use stacker_core::image::PlanarImage;

// ── fixtures ──────────────────────────────────────────────────────────────────

/// A smooth multi-scale texture in [0.07, 0.93]; the highest frequency is
/// well resolved by the pyramid levels that matter.
fn texture_at(x: f64, y: f64) -> f32 {
    (0.5 + 0.16 * (0.021 * x + 0.013 * y).sin()
        + 0.14 * (0.037 * x - 0.029 * y).cos()
        + 0.10 * (0.061 * x + 0.047 * y).sin()
        + 0.08 * (0.113 * x).sin() * (0.097 * y).cos()) as f32
}

fn image(w: usize, h: usize, f: impl Fn(f64, f64) -> f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.luma[y * w + x] = f(x as f64, y as f64);
        }
    }
    img
}

/// The reference content moved by `m` (reference to source): `source(m p) =
/// reference(p)`.
fn moved(w: usize, h: usize, m: &Matrix3<f32>) -> PlanarImage<f32> {
    let inv = m.cast::<f64>().try_inverse().expect("invertible matrix");
    image(w, h, |x, y| {
        let p = inv * Vector3::new(x, y, 1.0);
        texture_at(p.x, p.y)
    })
}

/// The largest displacement, over the corners and the centre, between the
/// points `a` and `b` map.
fn corner_error(a: &Matrix3<f32>, b: &Matrix3<f32>, w: usize, h: usize) -> f64 {
    let (a, b) = (a.cast::<f64>(), b.cast::<f64>());
    let (w, h) = (w as f64, h as f64);
    [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h), (w / 2.0, h / 2.0)]
        .iter()
        .map(|&(x, y)| {
            let v = Vector3::new(x, y, 1.0);
            let d = a * v - b * v;
            d.x.hypot(d.y)
        })
        .fold(0.0, f64::max)
}

const BIG: usize = 1600;

/// Reference, source, the reference's luma pyramid and the true matrix.
type Fixture = (
    PlanarImage<f32>,
    PlanarImage<f32>,
    LumaPyramid,
    Matrix3<f32>,
);

/// A 1600 x 1600 pair related by a sub-pixel similarity (shift of 0.63 and
/// -0.41 px, scale 1.0002, rotation 0.0002 rad: about a quarter pixel of corner
/// displacement each from scale and rotation), built once for the tests below.
fn big_fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let truth = params_to_matrix(
            &RegistrationParams {
                tx: 0.63 / BIG as f64,
                ty: -0.41 / BIG as f64,
                scale: 1.0002,
                rotate: 0.0002,
            },
            BIG,
            BIG,
        )
        .expect("finite parameters");
        let reference = image(BIG, BIG, texture_at);
        let source = moved(BIG, BIG, &truth);
        let pyramid = LumaPyramid::new(&reference);
        (reference, source, pyramid, truth)
    })
}

// ── accuracy of the scheduled search ──────────────────────────────────────────

/// The scheduled simplex plus the full-resolution polish recovers a known
/// sub-pixel similarity on a 1600 px image to within 0.05 px at the corners,
/// although full resolution only gets the short polish (the 800 px level is
/// large and ends the walk).
#[test]
fn scheduled_simplex_with_polish_recovers_a_subpixel_warp() {
    let (reference, source, pyramid, truth) = big_fixture();
    assert_eq!(pyramid.n_levels(), 6, "50, 100, 200, 400, 800, 1600 px");
    let opts = RefineOptions::default();
    assert_eq!(opts.polish_iterations, DEFAULT_POLISH_ITERATIONS);
    let refined =
        refine_alignment_with_pyramid(reference, pyramid, source, &Matrix3::identity(), &opts)
            .expect("refinement succeeds");
    let err = corner_error(&refined, truth, BIG, BIG);
    assert!(err < 0.05, "corner error {err} px with the polish");
}

/// Without the polish (`polish_iterations = 0`, GitHub's schedule exactly) the
/// result is only as good as the last coarse level: still inside a pixel, the
/// trade-off the polish exists for.
#[test]
fn schedule_without_polish_stays_inside_a_pixel() {
    let (reference, source, pyramid, truth) = big_fixture();
    let opts = RefineOptions {
        polish_iterations: 0,
        ..RefineOptions::default()
    };
    let refined =
        refine_alignment_with_pyramid(reference, pyramid, source, &Matrix3::identity(), &opts)
            .expect("refinement succeeds");
    let err = corner_error(&refined, truth, BIG, BIG);
    assert!(err < 1.0, "corner error {err} px without the polish");
    // The identity is worse than that (about 1.2 px at the corners), so the
    // search did real work.
    let identity = corner_error(&Matrix3::identity(), truth, BIG, BIG);
    assert!(err < identity, "refined {err} px vs identity {identity} px");
}

// ── what the objective sees ───────────────────────────────────────────────────

/// A device that counts the levels it is offered and the evaluations asked of
/// each, and answers none of them (the CPU takes over, so the result is the
/// plain CPU one).
#[derive(Clone, Default)]
struct Counting {
    levels: Arc<Mutex<Vec<(usize, usize)>>>,
    evals: Arc<Mutex<HashMap<(usize, usize), usize>>>,
}

struct CountingLevel {
    key: (usize, usize),
    evals: Arc<Mutex<HashMap<(usize, usize), usize>>>,
}

impl ObjectiveDevice for Counting {
    fn level(
        &self,
        _reference: &[f32],
        _source: &[f32],
        width: usize,
        height: usize,
        _sampler: ObjectiveSampler,
    ) -> Option<Box<dyn LevelDevice>> {
        self.levels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((width, height));
        Some(Box::new(CountingLevel {
            key: (width, height),
            evals: Arc::clone(&self.evals),
        }))
    }
}

impl LevelDevice for CountingLevel {
    fn moments(&mut self, _inverse: &Matrix3<f32>) -> Option<Moments> {
        *self
            .evals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(self.key)
            .or_default() += 1;
        None
    }
}

/// The levels offered to the objective are the plan's, in order, and the
/// full-resolution level is evaluated at most `(n + 1) + polish * (n + 2)` times
/// (the initial simplex plus the most evaluations an iteration of an
/// `n`-dimensional simplex can need), a small fraction of what 400 iterations
/// would be allowed.  Without the polish full resolution is never touched.
#[test]
fn objective_sees_the_planned_levels_and_a_bounded_full_resolution_count() {
    const W: usize = 160;
    const H: usize = 128;
    const N: usize = 4;
    let reference = image(W, H, texture_at);
    let truth = params_to_matrix(
        &RegistrationParams {
            tx: 2.3 / W as f64,
            ty: -1.4 / H as f64,
            scale: 1.004,
            rotate: 0.006,
        },
        W,
        H,
    )
    .expect("finite parameters");
    let source = moved(W, H, &truth);
    let pyramid = LumaPyramid::new(&reference);

    let sides: Vec<usize> = (0..pyramid.n_levels())
        .map(|i| {
            let (_, w, h) = pyramid.level_luma(i, &reference.luma);
            w.min(h)
        })
        .collect();

    for polish in [0_usize, 12, DEFAULT_POLISH_ITERATIONS] {
        let device = Counting::default();
        let opts = RefineOptions {
            polish_iterations: polish,
            device: Some(DeviceHandle(Arc::new(device.clone()))),
            ..RefineOptions::default()
        };
        refine_alignment_with_pyramid(&reference, &pyramid, &source, &Matrix3::identity(), &opts)
            .expect("refinement succeeds");

        let expected: Vec<usize> = level_plan(&sides, opts.max_iterations, polish)
            .iter()
            .map(|step| pyramid.level_luma(step.level, &reference.luma).1)
            .collect();
        let asked: Vec<usize> = device
            .levels
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|&(w, _)| w)
            .collect();
        assert_eq!(asked, expected, "polish {polish}: levels offered");

        let at_full = device
            .evals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(W, H))
            .copied()
            .unwrap_or(0);
        if polish == 0 {
            assert_eq!(at_full, 0, "no polish: full resolution is never evaluated");
            assert!(!asked.contains(&W), "no polish: full level never offered");
        } else {
            assert_eq!(asked.last(), Some(&W), "the polish is the last level");
            assert!(at_full > N, "at least the initial simplex: {at_full}");
            let bound = (N + 1) + polish * (N + 2);
            assert!(
                at_full <= bound,
                "polish {polish}: {at_full} full-resolution evaluations > {bound}"
            );
        }
    }
}
