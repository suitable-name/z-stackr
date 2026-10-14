#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]

//! The resumable refiners (`refine::search::AlignSearch`) against the
//! synchronous ones: with the same answers to the same questions the refined
//! matrix is the same bit for bit. Checked with the CPU objective (`tell(None)`
//! and `cpu_only`) and with a scripted device that declines a level and fails
//! every eleventh evaluation, for every refiner kind; the device sees the same
//! sequence of evaluations in both forms.

use std::sync::{Arc, Mutex, PoisonError};

use stacker_align::{
    Matrix3,
    device::{DeviceHandle, LevelDevice, Moments, ObjectiveDevice, ObjectiveSampler},
    refine::{
        BoundedRefineOptions, DEFAULT_POLISH_ITERATIONS, LumaPyramid, RefineOptions,
        RegistrationParams, params_to_matrix, refine_alignment_bounded_with_pyramid,
        refine_alignment_registration, refine_alignment_registration_with_pyramid,
        refine_alignment_with_pyramid,
        search::{AlignSearch, Ask, SearchResult},
    },
    transform::warp_image,
};
use stacker_core::{error::StackerError, image::PlanarImage};

const W: usize = 160;
const H: usize = 128;
/// Iteration cap: the schedule and the restarts are exercised, the run stays short.
const ITER: usize = 80;

/// A smooth multi-scale texture, evaluated at `(x - dx, y - dy)`.
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

/// The reference warped by a small similarity (scale, rotation, shift).
fn similar(
    reference: &PlanarImage<f32>,
    tx: f64,
    ty: f64,
    scale: f64,
    rotate: f64,
) -> PlanarImage<f32> {
    let p = RegistrationParams {
        tx: tx / W as f64,
        ty: ty / H as f64,
        scale,
        rotate,
    };
    let m = params_to_matrix(&p, W, H).expect("finite parameters");
    warp_image(reference, &m).expect("invertible matrix")
}

fn translation(x: f32, y: f32) -> Matrix3<f32> {
    let mut m = Matrix3::identity();
    m[(0, 2)] = x;
    m[(1, 2)] = y;
    m
}

fn bits(m: &Matrix3<f32>) -> [u32; 9] {
    std::array::from_fn(|i| m[(i / 3, i % 3)].to_bits())
}

/// Both outcomes are the same: the same matrix bit for bit, or the same error.
fn assert_same(label: &str, sync: &Result<Matrix3<f32>, StackerError>, got: &SearchResult) {
    match (sync, got) {
        (Ok(a), Ok(b)) => assert_eq!(bits(a), bits(b), "{label}: {a} vs {b}"),
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{label}"),
        (a, b) => panic!("{label}: {a:?} vs {b:?}"),
    }
}

/// Answer every question with "the CPU does it".
fn run_cpu(mut search: AlignSearch<'_>) -> SearchResult {
    loop {
        match search.ask() {
            Ask::Level { .. } => search.level_ready(false),
            Ask::Evaluate { .. } => search.tell(None),
            Ask::Done(result) => return result,
        }
    }
}

// ── a scripted device ─────────────────────────────────────────────────────────

/// Moments of a made-up objective with its minimum near a fixed warp; some
/// candidates cover too little of the frame (the penalty rule).
fn mock_moments(inv: &Matrix3<f32>, w: usize, h: usize) -> Moments {
    let n = (w * h) as u64;
    let tx = f64::from(inv[(0, 2)]) / w as f64;
    let ty = f64::from(inv[(1, 2)]) / h as f64;
    let cost = (tx - 0.012).powi(2)
        + (ty + 0.008).powi(2)
        + f64::from(inv[(0, 0)] - 1.01).powi(2)
        + f64::from(inv[(1, 1)] - 0.995).powi(2)
        + f64::from(inv[(0, 1)] - 0.004).powi(2)
        + f64::from(inv[(1, 0)] + 0.004).powi(2);
    let count = if tx.abs() > 0.2 { n / 3 } else { n };
    Moments {
        count,
        mean: 0.01,
        m2: (cost + 1.0e-6) * (count as f64 - 1.0),
    }
}

type Log = Arc<Mutex<Vec<[u32; 9]>>>;

/// Declines every level of width `decline_width`; the other levels fail every
/// eleventh evaluation. Logs every evaluation asked of it.
#[derive(Clone)]
struct Scripted {
    decline_width: usize,
    log: Log,
}

struct ScriptedLevel {
    w: usize,
    h: usize,
    calls: u32,
    log: Log,
}

impl Scripted {
    fn new(decline_width: usize) -> Self {
        Self {
            decline_width,
            log: Arc::default(),
        }
    }

    fn open(&self, w: usize, h: usize) -> Option<ScriptedLevel> {
        (w != self.decline_width).then(|| ScriptedLevel {
            w,
            h,
            calls: 0,
            log: Arc::clone(&self.log),
        })
    }

    fn evaluations(&self) -> Vec<[u32; 9]> {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl ObjectiveDevice for Scripted {
    fn level(
        &self,
        _reference: &[f32],
        _source: &[f32],
        width: usize,
        height: usize,
        _sampler: ObjectiveSampler,
    ) -> Option<Box<dyn LevelDevice>> {
        self.open(width, height)
            .map(|l| Box::new(l) as Box<dyn LevelDevice>)
    }
}

impl ScriptedLevel {
    fn answer(&mut self, inverse: &Matrix3<f32>) -> Option<Moments> {
        self.calls += 1;
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(bits(inverse));
        (!self.calls.is_multiple_of(11)).then(|| mock_moments(inverse, self.w, self.h))
    }
}

impl LevelDevice for ScriptedLevel {
    fn moments(&mut self, inverse: &Matrix3<f32>) -> Option<Moments> {
        self.answer(inverse)
    }
}

/// Drive a search against a scripted device the way `drive` does.
fn run_device(mut search: AlignSearch<'_>, device: &Scripted) -> SearchResult {
    let mut level: Option<ScriptedLevel> = None;
    loop {
        match search.ask() {
            Ask::Level { level: idx } => {
                let planes = search.level_planes(idx);
                level = device.open(planes.width, planes.height);
                search.level_ready(level.is_some());
            }
            Ask::Evaluate { inverse } => {
                let moments = level
                    .as_mut()
                    .expect("an evaluation is asked only on an accepted level")
                    .answer(&inverse);
                search.tell(moments);
            }
            Ask::Done(result) => return result,
        }
    }
}

// ── the cases ─────────────────────────────────────────────────────────────────

struct Case {
    name: &'static str,
    source: PlanarImage<f32>,
    initial: Matrix3<f32>,
}

fn cases(reference: &PlanarImage<f32>) -> Vec<Case> {
    vec![
        Case {
            name: "translation",
            source: texture(3.0, 2.0),
            initial: Matrix3::identity(),
        },
        Case {
            name: "small affine",
            source: similar(reference, 2.5, -1.5, 1.01, 0.012),
            initial: Matrix3::identity(),
        },
        Case {
            name: "near the band edge, seeded",
            source: texture(7.4, -6.9),
            initial: translation(1.0, 0.5),
        },
        Case {
            name: "unchanged",
            source: reference.clone(),
            initial: Matrix3::identity(),
        },
    ]
}

const fn unbounded_opts(similarity: bool, device: Option<DeviceHandle>) -> RefineOptions {
    RefineOptions {
        max_iterations: ITER,
        polish_iterations: DEFAULT_POLISH_ITERATIONS,
        allow_scale: similarity,
        allow_rotate: similarity,
        device,
    }
}

fn bounded_opts(similarity: bool, device: Option<DeviceHandle>) -> BoundedRefineOptions {
    BoundedRefineOptions {
        allow_scale: similarity,
        allow_rotation: similarity,
        max_iterations: ITER,
        device,
        ..BoundedRefineOptions::default()
    }
}

/// Only the X shift is searched: the golden-section branch.
fn single_dof_opts(device: Option<DeviceHandle>) -> BoundedRefineOptions {
    BoundedRefineOptions {
        allow_shift_y: false,
        allow_scale: false,
        allow_rotation: false,
        max_iterations: ITER,
        device,
        ..BoundedRefineOptions::default()
    }
}

#[test]
fn unbounded_search_matches_the_sync_refiner_on_the_cpu() {
    let reference = texture(0.0, 0.0);
    let pyramid = LumaPyramid::new(&reference);
    assert!(pyramid.n_levels() >= 3, "the test needs a deep pyramid");
    for case in cases(&reference) {
        for similarity in [false, true] {
            let opts = unbounded_opts(similarity, None);
            let sync = refine_alignment_with_pyramid(
                &reference,
                &pyramid,
                &case.source,
                &case.initial,
                &opts,
            );
            let label = format!("unbounded {} similarity={similarity}", case.name);
            let asked =
                AlignSearch::unbounded(&reference, &pyramid, &case.source, &case.initial, &opts);
            assert_same(&label, &sync, &run_cpu(asked));
            let cpu_only =
                AlignSearch::unbounded(&reference, &pyramid, &case.source, &case.initial, &opts)
                    .cpu_only();
            assert_same(&label, &sync, &run_cpu(cpu_only));
        }
    }
}

#[test]
fn bounded_search_matches_the_sync_refiner_on_the_cpu() {
    let reference = texture(0.0, 0.0);
    let pyramid = LumaPyramid::new(&reference);
    for case in cases(&reference) {
        let variants = [
            ("translation", bounded_opts(false, None)),
            ("similarity", bounded_opts(true, None)),
            ("single dof", single_dof_opts(None)),
        ];
        for (kind, opts) in variants {
            let sync = refine_alignment_bounded_with_pyramid(
                &reference,
                &pyramid,
                &case.source,
                &case.initial,
                &opts,
            );
            let label = format!("bounded {kind} {}", case.name);
            let asked =
                AlignSearch::bounded(&reference, &pyramid, &case.source, &case.initial, &opts);
            assert_same(&label, &sync, &run_cpu(asked));
            let cpu_only =
                AlignSearch::bounded(&reference, &pyramid, &case.source, &case.initial, &opts)
                    .cpu_only();
            assert_same(&label, &sync, &run_cpu(cpu_only));
        }
    }
}

#[test]
fn registration_search_matches_the_sync_refiner_on_the_cpu() {
    let reference = texture(0.0, 0.0);
    for case in cases(&reference) {
        let variants = [
            ("similarity", bounded_opts(true, None)),
            ("translation", bounded_opts(false, None)),
            ("single dof", single_dof_opts(None)),
        ];
        for (kind, opts) in variants {
            let sync =
                refine_alignment_registration(&reference, &case.source, &case.initial, &opts);
            let label = format!("registration {kind} {}", case.name);
            let asked = AlignSearch::registration(&reference, &case.source, &case.initial, &opts);
            assert_same(&label, &sync, &run_cpu(asked));
            let cpu_only =
                AlignSearch::registration(&reference, &case.source, &case.initial, &opts)
                    .cpu_only();
            assert_same(&label, &sync, &run_cpu(cpu_only));
        }
    }
}

/// The device declines the middle level and fails every eleventh evaluation of
/// the others: both forms ask it the same questions in the same order and end on
/// the same matrix.
#[test]
fn device_answers_give_the_same_result_and_the_same_evaluations() {
    let reference = texture(0.0, 0.0);
    let pyramid = LumaPyramid::new(&reference);
    for case in cases(&reference) {
        // Level widths of a 160-wide pyramid: 40, 80, 160. Decline the middle one.
        for decline in [80, 0] {
            // Unbounded, with the restarts of the coarsest level.
            let dev = Scripted::new(decline);
            let handle = DeviceHandle(Arc::new(dev.clone()));
            let opts = unbounded_opts(true, Some(handle));
            let sync = refine_alignment_with_pyramid(
                &reference,
                &pyramid,
                &case.source,
                &case.initial,
                &opts,
            );
            let sync_log = dev.evaluations();
            let dev = Scripted::new(decline);
            let asked = run_device(
                AlignSearch::unbounded(&reference, &pyramid, &case.source, &case.initial, &opts),
                &dev,
            );
            let label = format!("unbounded+device {} decline={decline}", case.name);
            assert_same(&label, &sync, &asked);
            assert!(!sync_log.is_empty(), "{label}: the device was never asked");
            assert_eq!(sync_log, dev.evaluations(), "{label}: evaluations");

            // Bounded: simplex and golden section.
            for single in [false, true] {
                let make = |h| {
                    if single {
                        single_dof_opts(Some(h))
                    } else {
                        bounded_opts(true, Some(h))
                    }
                };
                let dev = Scripted::new(decline);
                let opts = make(DeviceHandle(Arc::new(dev.clone())));
                let sync = refine_alignment_bounded_with_pyramid(
                    &reference,
                    &pyramid,
                    &case.source,
                    &case.initial,
                    &opts,
                );
                let sync_log = dev.evaluations();
                let dev = Scripted::new(decline);
                let asked = run_device(
                    AlignSearch::bounded(&reference, &pyramid, &case.source, &case.initial, &opts),
                    &dev,
                );
                let label = format!(
                    "bounded+device single={single} {} decline={decline}",
                    case.name
                );
                assert_same(&label, &sync, &asked);
                assert_eq!(sync_log, dev.evaluations(), "{label}: evaluations");
            }
        }

        // Registration (a pyramid user like the others): the full-resolution
        // level of the polish, accepted or declined.
        for decline in [W, 0] {
            for single in [false, true] {
                let dev = Scripted::new(decline);
                let handle = DeviceHandle(Arc::new(dev.clone()));
                let opts = if single {
                    single_dof_opts(Some(handle))
                } else {
                    bounded_opts(true, Some(handle))
                };
                let sync =
                    refine_alignment_registration(&reference, &case.source, &case.initial, &opts);
                let sync_log = dev.evaluations();
                let dev = Scripted::new(decline);
                let asked = run_device(
                    AlignSearch::registration(&reference, &case.source, &case.initial, &opts),
                    &dev,
                );
                let label = format!(
                    "registration+device single={single} {} decline={decline}",
                    case.name
                );
                assert_same(&label, &sync, &asked);
                assert_eq!(sync_log, dev.evaluations(), "{label}: evaluations");
            }
        }
    }
}

/// The registration refiner is a pyramid user: the form with a shared reference
/// pyramid is the plain form bit for bit, in both drivers.
#[test]
fn registration_with_a_shared_pyramid_is_the_plain_form() {
    let reference = texture(0.0, 0.0);
    let pyramid = LumaPyramid::new(&reference);
    for case in cases(&reference) {
        for (kind, opts) in [
            ("similarity", bounded_opts(true, None)),
            ("single dof", single_dof_opts(None)),
        ] {
            let plain =
                refine_alignment_registration(&reference, &case.source, &case.initial, &opts);
            let shared = refine_alignment_registration_with_pyramid(
                &reference,
                &pyramid,
                &case.source,
                &case.initial,
                &opts,
            );
            let label = format!("registration {kind} {}", case.name);
            assert_same(&label, &plain, &shared);
            let asked = AlignSearch::registration_with_pyramid(
                &reference,
                &pyramid,
                &case.source,
                &case.initial,
                &opts,
            );
            assert_same(&label, &plain, &run_cpu(asked));
        }
    }
}

/// The polish is part of the identity: with it switched off (GitHub's schedule)
/// and with a long one, the resumable searches still match the synchronous
/// refiners, for every refiner kind.
#[test]
fn polish_budgets_keep_the_search_identical() {
    let reference = texture(0.0, 0.0);
    let pyramid = LumaPyramid::new(&reference);
    for polish in [0_usize, 7, 90] {
        for case in cases(&reference) {
            let opts = RefineOptions {
                polish_iterations: polish,
                ..unbounded_opts(true, None)
            };
            let sync = refine_alignment_with_pyramid(
                &reference,
                &pyramid,
                &case.source,
                &case.initial,
                &opts,
            );
            let asked =
                AlignSearch::unbounded(&reference, &pyramid, &case.source, &case.initial, &opts);
            assert_same(
                &format!("unbounded polish={polish} {}", case.name),
                &sync,
                &run_cpu(asked),
            );

            for (kind, base) in [
                ("similarity", bounded_opts(true, None)),
                ("single dof", single_dof_opts(None)),
            ] {
                let opts = BoundedRefineOptions {
                    polish_iterations: polish,
                    ..base
                };
                let label = format!("bounded {kind} polish={polish} {}", case.name);
                let sync = refine_alignment_bounded_with_pyramid(
                    &reference,
                    &pyramid,
                    &case.source,
                    &case.initial,
                    &opts,
                );
                let asked =
                    AlignSearch::bounded(&reference, &pyramid, &case.source, &case.initial, &opts);
                assert_same(&label, &sync, &run_cpu(asked));

                let label = format!("registration {kind} polish={polish} {}", case.name);
                let sync = refine_alignment_registration_with_pyramid(
                    &reference,
                    &pyramid,
                    &case.source,
                    &case.initial,
                    &opts,
                );
                let asked = AlignSearch::registration_with_pyramid(
                    &reference,
                    &pyramid,
                    &case.source,
                    &case.initial,
                    &opts,
                );
                assert_same(&label, &sync, &run_cpu(asked));
            }
        }
    }
}

#[test]
fn invalid_inputs_end_with_the_sync_error() {
    let reference = texture(0.0, 0.0);
    let pyramid = LumaPyramid::new(&reference);
    let source = texture(1.0, 1.0);
    let mut nan = Matrix3::<f32>::identity();
    nan[(0, 2)] = f32::NAN;
    let small = PlanarImage::<f32>::new(64, 64);

    let opts = unbounded_opts(true, None);
    let bopts = bounded_opts(true, None);
    for (label, src, initial) in [
        ("non-finite initial", &source, nan),
        ("size mismatch", &small, Matrix3::identity()),
    ] {
        assert_same(
            label,
            &refine_alignment_with_pyramid(&reference, &pyramid, src, &initial, &opts),
            &run_cpu(AlignSearch::unbounded(
                &reference, &pyramid, src, &initial, &opts,
            )),
        );
        assert_same(
            label,
            &refine_alignment_bounded_with_pyramid(&reference, &pyramid, src, &initial, &bopts),
            &run_cpu(AlignSearch::bounded(
                &reference, &pyramid, src, &initial, &bopts,
            )),
        );
        assert_same(
            label,
            &refine_alignment_registration(&reference, src, &initial, &bopts),
            &run_cpu(AlignSearch::registration(&reference, src, &initial, &bopts)),
        );
    }

    // A pyramid of another image.
    let other = PlanarImage::<f32>::new(96, 96);
    let wrong = LumaPyramid::new(&other);
    assert_same(
        "wrong pyramid",
        &refine_alignment_with_pyramid(&reference, &wrong, &source, &Matrix3::identity(), &opts),
        &run_cpu(AlignSearch::unbounded(
            &reference,
            &wrong,
            &source,
            &Matrix3::identity(),
            &opts,
        )),
    );

    // Nothing to search: the initial matrix comes back.
    let none = BoundedRefineOptions {
        allow_shift_x: false,
        allow_shift_y: false,
        allow_scale: false,
        allow_rotation: false,
        ..BoundedRefineOptions::default()
    };
    let initial = translation(2.0, 1.0);
    assert_same(
        "no active dof",
        &refine_alignment_bounded_with_pyramid(&reference, &pyramid, &source, &initial, &none),
        &run_cpu(AlignSearch::bounded(
            &reference, &pyramid, &source, &initial, &none,
        )),
    );
}

#[test]
fn asking_twice_repeats_the_question_and_done_is_reported_once() {
    let reference = texture(0.0, 0.0);
    let pyramid = LumaPyramid::new(&reference);
    let source = texture(1.0, 0.5);
    let opts = bounded_opts(false, None);
    let mut search =
        AlignSearch::bounded(&reference, &pyramid, &source, &Matrix3::identity(), &opts);
    let Ask::Level { level } = search.ask() else {
        panic!("the first question is a level");
    };
    assert_eq!(level, 0);
    assert!(matches!(search.ask(), Ask::Level { level: 0 }));
    let planes = search.level_planes(0);
    assert_eq!(planes.reference.len(), planes.width * planes.height);
    assert_eq!(planes.source.len(), planes.width * planes.height);
    search.level_ready(true);
    let Ask::Evaluate { inverse } = search.ask() else {
        panic!("the level is accepted, so the objective is asked for");
    };
    let Ask::Evaluate { inverse: again } = search.ask() else {
        panic!("the question stands until it is answered");
    };
    assert_eq!(bits(&inverse), bits(&again));

    // Run to the end on the CPU; a second `ask` after `Done` is an error, not a panic.
    search.tell(None);
    let result = loop {
        match search.ask() {
            Ask::Level { .. } => search.level_ready(false),
            Ask::Evaluate { .. } => search.tell(None),
            Ask::Done(result) => break result,
        }
    };
    assert!(result.is_ok());
    assert!(matches!(search.ask(), Ask::Done(Err(_))));
}
