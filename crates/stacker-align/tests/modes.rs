#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

//! The motion models (`Motion`), the six-parameter model (`AffineParams`), the
//! optimiser dispatch (`refine_motion`) and the identity gate.
//!
//! Every fixture is analytic: the source is the reference texture evaluated at
//! the inverse-mapped coordinates, so no interpolation or border artefact sits
//! between the truth and the objective's minimum.

use nalgebra::{Matrix3, Vector3};
use stacker_align::{
    refine::{
        AffineParams, BoundedRefineOptions, Engine, LumaPyramid, Motion, MotionRefine, Optimizer,
        RegistrationParams, apply_identity_gate, matrix_to_affine_params, matrix_to_params,
        params_to_matrix, refine_alignment_bounded, refine_alignment_bounded_with_pyramid,
        refine_alignment_registration, refine_motion, refined_beats_identity,
        search::{AlignSearch, Ask, SearchResult},
    },
    refine_alignment_lk,
};
use stacker_core::{error::StackerError, image::PlanarImage};

const W: usize = 160;
const H: usize = 128;
/// Iteration cap for the ask/tell comparisons: the schedule is exercised, the
/// run stays short.
const ITER: usize = 60;

// ── fixtures ──────────────────────────────────────────────────────────────────

/// A smooth multi-directional texture in [0.06, 0.94].
fn texture_at(x: f64, y: f64) -> f32 {
    (0.5 + 0.18 * (0.21 * x + 0.13 * y).sin()
        + 0.18 * (0.17 * x - 0.19 * y).cos()
        + 0.08 * (0.07 * x).sin() * (0.05 * y).cos()) as f32
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

fn reference(w: usize, h: usize) -> PlanarImage<f32> {
    image(w, h, texture_at)
}

/// The reference content moved by `m` (reference to source): `source(m p) =
/// reference(p)`, evaluated analytically.
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

fn bits(m: &Matrix3<f32>) -> [u32; 9] {
    std::array::from_fn(|i| m[(i / 3, i % 3)].to_bits())
}

fn assert_same(label: &str, sync: &Result<Matrix3<f32>, StackerError>, got: &SearchResult) {
    match (sync, got) {
        (Ok(a), Ok(b)) => assert_eq!(bits(a), bits(b), "{label}: {a} vs {b}"),
        (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}"), "{label}"),
        (a, b) => panic!("{label}: {a:?} vs {b:?}"),
    }
}

/// Answer every question of an ask/tell search with "the CPU does it".
fn run_cpu(mut search: AlignSearch<'_>) -> SearchResult {
    loop {
        match search.ask() {
            Ask::Level { .. } => search.level_ready(false),
            Ask::Evaluate { .. } => search.tell(None),
            Ask::Done(result) => return result,
        }
    }
}

const fn known_affine() -> AffineParams {
    AffineParams {
        tx: 0.012,
        ty: -0.009,
        scale: 1.008,
        rotate: 0.012,
        aspect: 1.012,
        shear: 0.012,
    }
}

// ── the parameter model ───────────────────────────────────────────────────────

/// `params -> matrix -> params -> matrix` returns the matrix for a batch of
/// non-degenerate six-parameter sets: the decomposition is exact.
#[test]
fn affine_params_round_trip_is_exact() {
    let (w, h) = (200_usize, 150_usize);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    for case in 0..200 {
        let orig = AffineParams {
            tx: (next() - 0.5) * 0.2,
            ty: (next() - 0.5) * 0.2,
            scale: 0.8 + next() * 0.4,
            rotate: (next() - 0.5) * 20.0_f64.to_radians(),
            aspect: 0.9 + next() * 0.2,
            shear: (next() - 0.5) * 0.4,
        };
        let m1 = orig.to_matrix(w, h).expect("finite parameters");
        let recovered = matrix_to_affine_params(&m1, w, h);
        let m2 = recovered.to_matrix(w, h).expect("finite parameters");
        let diff = (m1 - m2).norm();
        assert!(
            diff < 1.0e-3,
            "case {case}: round trip changed the matrix by {diff}; {orig:?} -> {recovered:?}"
        );
        // The parameters themselves come back, not only the matrix.
        assert!(
            (recovered.aspect - orig.aspect).abs() < 1.0e-4
                && (recovered.shear - orig.shear).abs() < 1.0e-3
                && (recovered.scale - orig.scale).abs() < 1.0e-4,
            "case {case}: {orig:?} -> {recovered:?}"
        );
    }
}

/// With `aspect == 1`, `shear == 0` the six-parameter matrix is the
/// four-parameter one bit for bit (the searches rely on it).
#[test]
fn similarity_is_bit_identical_to_the_four_parameter_model() {
    for &(tx, ty, scale, rotate) in &[
        (0.0, 0.0, 1.0, 0.0),
        (0.05, -0.03, 1.02, 0.01),
        (-0.011, 0.0123, 0.97, -0.04),
        (0.3, 0.2, 1.3, 0.5),
    ] {
        let p = RegistrationParams {
            tx,
            ty,
            scale,
            rotate,
        };
        let four = params_to_matrix(&p, 200, 150).expect("finite");
        let six = AffineParams::from(p).to_matrix(200, 150).expect("finite");
        assert_eq!(bits(&four), bits(&six), "{p:?}");
    }
}

/// A non-finite parameter gives no matrix; a singular matrix decomposes to the
/// identity.
#[test]
fn affine_params_guard_non_finite_and_degenerate_input() {
    let mut p = AffineParams::identity();
    p.shear = f64::NAN;
    assert!(p.to_matrix(100, 100).is_none());
    p = AffineParams::identity();
    p.aspect = f64::INFINITY;
    assert!(p.to_matrix(100, 100).is_none());

    let mut singular = Matrix3::<f32>::identity();
    singular[(0, 0)] = 0.0;
    singular[(1, 0)] = 0.0;
    assert_eq!(
        matrix_to_affine_params(&singular, 100, 100),
        AffineParams::identity()
    );
    let mut rank_one = Matrix3::<f32>::identity();
    rank_one[(0, 1)] = 1.0;
    rank_one[(1, 0)] = 1.0;
    rank_one[(1, 1)] = 1.0;
    assert_eq!(
        matrix_to_affine_params(&rank_one, 100, 100),
        AffineParams::identity()
    );
    let mut nan = Matrix3::<f32>::identity();
    nan[(0, 2)] = f32::NAN;
    assert_eq!(
        matrix_to_affine_params(&nan, 100, 100),
        AffineParams::identity()
    );
}

/// The exact decomposition keeps what the similarity reduction throws away.
#[test]
fn decomposition_keeps_aspect_and_shear_while_the_reduction_drops_them() {
    let m = known_affine().to_matrix(W, H).expect("finite");
    let exact = matrix_to_affine_params(&m, W, H);
    assert!((exact.aspect - 1.012).abs() < 1.0e-5, "{exact:?}");
    assert!((exact.shear - 0.012).abs() < 1.0e-5, "{exact:?}");
    let reduced = params_to_matrix(&matrix_to_params(&m, W, H), W, H).expect("finite");
    assert!(
        (reduced - m).norm() > 0.1,
        "the similarity reduction must differ from the full matrix"
    );
}

#[test]
fn motion_flags_follow_the_dof_ladder() {
    assert!(!Motion::Shift.allows_scale());
    assert!(!Motion::Shift.allows_rotation());
    assert!(Motion::Translation.allows_scale());
    assert!(!Motion::Translation.allows_rotation());
    assert!(Motion::Similarity.allows_rotation());
    assert!(!Motion::Similarity.allows_aspect_shear());
    assert!(Motion::Affine.allows_aspect_shear());

    let default = BoundedRefineOptions::default();
    assert!(!default.uses_affine());
    assert!(!default.allow_aspect && !default.allow_shear);
    assert!((default.max_aspect - 10.0).abs() < f64::EPSILON);
    assert!((default.max_shear - 10.0).abs() < f64::EPSILON);

    let affine = Motion::Affine.bounded_options();
    assert!(affine.uses_affine() && affine.allow_scale && affine.allow_rotation);
    let translation = Motion::Translation.bounded_options();
    assert!(translation.allow_scale && !translation.allow_rotation && !translation.uses_affine());
    // The search windows are the crate's, not GitHub's 20 %.
    assert!((translation.max_shift_x - 5.0).abs() < f64::EPSILON);

    assert!(Motion::Affine.unbounded_options().is_none());
    let sim = Motion::Similarity.unbounded_options().expect("has a form");
    assert!(sim.allow_scale && sim.allow_rotate);
    let tr = Motion::Translation.unbounded_options().expect("has a form");
    assert!(tr.allow_scale && !tr.allow_rotate);

    // Affine is always bounded.
    assert!(MotionRefine::new(Motion::Affine).simplex_is_bounded());
    assert!(!MotionRefine::new(Motion::Similarity).simplex_is_bounded());
    assert!(
        MotionRefine::new(Motion::Affine)
            .unbounded_options()
            .is_none()
    );
    assert_eq!(Optimizer::default(), Optimizer::Auto);
}

// ── the simplex on the six-parameter model ────────────────────────────────────

/// The bounded simplex recovers a known six-parameter warp: the refined matrix
/// beats the identity by far and lands within a pixel.  (Six dimensions in a
/// logit space: the bound is deliberately looser than the one of Lucas-Kanade,
/// which is the accurate optimiser.)
#[test]
fn bounded_simplex_recovers_a_six_parameter_warp() {
    let truth = known_affine().to_matrix(W, H).expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth);
    let opts = Motion::Affine.bounded_options();
    let refined = refine_alignment_bounded(&reference, &source, &Matrix3::identity(), &opts)
        .expect("refinement succeeds");
    let err = corner_error(&refined, &truth, W, H);
    assert!(err < 1.0, "corner error {err} px");
    let rms = |m: &Matrix3<f32>| stacker_align::refine::lk_rms(&reference, &source, m);
    assert!(
        rms(&refined) < 0.3 * rms(&Matrix3::identity()),
        "refined {} vs identity {}",
        rms(&refined),
        rms(&Matrix3::identity())
    );
}

/// The six-parameter search lets the aspect move; the four-parameter one cannot.
#[test]
fn aspect_and_shear_are_only_searched_in_the_affine_model() {
    let truth = AffineParams {
        aspect: 1.03,
        shear: 0.02,
        ..AffineParams::identity()
    }
    .to_matrix(W, H)
    .expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth);

    let similarity = refine_alignment_bounded(
        &reference,
        &source,
        &Matrix3::identity(),
        &Motion::Similarity.bounded_options(),
    )
    .expect("refinement succeeds");
    let p = matrix_to_affine_params(&similarity, W, H);
    assert!(
        (p.aspect - 1.0).abs() < 1.0e-6 && p.shear.abs() < 1.0e-6,
        "a similarity search must leave aspect and shear alone: {p:?}"
    );

    let affine = refine_alignment_bounded(
        &reference,
        &source,
        &Matrix3::identity(),
        &Motion::Affine.bounded_options(),
    )
    .expect("refinement succeeds");
    let q = matrix_to_affine_params(&affine, W, H);
    assert!(
        (q.aspect - 1.0).abs() > 0.01 && q.shear.abs() > 0.005,
        "the affine search must move aspect and shear: {q:?}"
    );
    let rms = |m: &Matrix3<f32>| stacker_align::refine::lk_rms(&reference, &source, m);
    assert!(rms(&affine) < rms(&similarity));
}

/// The ask/tell machine runs the six-parameter search through the same
/// arithmetic as the synchronous refiners, for the pyramid and the single-pass
/// form and through `for_motion`.
#[test]
fn six_parameter_search_matches_the_sync_refiners_bit_for_bit() {
    let truth = known_affine().to_matrix(W, H).expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth);
    let pyramid = LumaPyramid::new(&reference);
    assert!(pyramid.n_levels() >= 3, "the test needs a deep pyramid");
    let seeds = [Matrix3::identity(), {
        let mut s = Matrix3::identity();
        s[(0, 2)] = 1.0;
        s
    }];
    for initial in seeds {
        let opts = BoundedRefineOptions {
            max_iterations: ITER,
            ..Motion::Affine.bounded_options()
        };
        let sync =
            refine_alignment_bounded_with_pyramid(&reference, &pyramid, &source, &initial, &opts);
        let asked = AlignSearch::bounded(&reference, &pyramid, &source, &initial, &opts);
        assert_same("affine pyramid", &sync, &run_cpu(asked));
        let cpu_only =
            AlignSearch::bounded(&reference, &pyramid, &source, &initial, &opts).cpu_only();
        assert_same("affine pyramid cpu_only", &sync, &run_cpu(cpu_only));

        let single = refine_alignment_registration(&reference, &source, &initial, &opts);
        let asked = AlignSearch::registration(&reference, &source, &initial, &opts);
        assert_same("affine single pass", &single, &run_cpu(asked));

        let cfg = MotionRefine {
            bounded: true,
            max_iterations: ITER,
            ..MotionRefine::new(Motion::Affine)
        };
        let via_cfg = refine_alignment_bounded_with_pyramid(
            &reference,
            &pyramid,
            &source,
            &initial,
            &cfg.bounded_options(),
        );
        let asked = AlignSearch::for_motion(&reference, &pyramid, &source, &initial, &cfg);
        assert_same("for_motion affine", &via_cfg, &run_cpu(asked));
    }
}

/// `for_motion` picks the unbounded search for a similarity and the bounded one
/// for the rest, like the synchronous dispatch.
#[test]
fn for_motion_selects_the_same_search_as_the_sync_dispatch() {
    let truth = RegistrationParams {
        tx: 0.01,
        ty: -0.006,
        scale: 1.006,
        rotate: 0.01,
    };
    let truth = params_to_matrix(&truth, W, H).expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth);
    let pyramid = LumaPyramid::new(&reference);
    for motion in [Motion::Shift, Motion::Translation, Motion::Similarity] {
        for bounded in [false, true] {
            let cfg = MotionRefine {
                bounded,
                max_iterations: ITER,
                optimizer: Optimizer::NelderMead,
                ..MotionRefine::new(motion)
            };
            let sync = refine_motion(&reference, &pyramid, &source, &Matrix3::identity(), &cfg)
                .map(|r| r.matrix);
            let asked = run_cpu(AlignSearch::for_motion(
                &reference,
                &pyramid,
                &source,
                &Matrix3::identity(),
                &cfg,
            ));
            assert_same(&format!("{motion:?} bounded={bounded}"), &sync, &asked);
        }
    }
}

// ── motion models through the dispatcher ──────────────────────────────────────

/// `Translation` fits the focus-breathing scale and nothing else.
#[test]
fn translation_motion_recovers_breathing_scale_and_leaves_rotation() {
    let truth = RegistrationParams {
        tx: 0.006,
        ty: -0.004,
        scale: 1.012,
        rotate: 0.0,
    };
    let truth_m = params_to_matrix(&truth, W, H).expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth_m);
    let pyramid = LumaPyramid::new(&reference);

    for (optimizer, bounded, tolerance) in [
        (Optimizer::LucasKanade, true, 0.002),
        (Optimizer::NelderMead, true, 0.01),
        (Optimizer::NelderMead, false, 0.01),
    ] {
        let cfg = MotionRefine {
            optimizer,
            bounded,
            max_iterations: 200,
            ..MotionRefine::new(Motion::Translation)
        };
        let refined = refine_motion(&reference, &pyramid, &source, &Matrix3::identity(), &cfg)
            .expect("refinement succeeds");
        let p = matrix_to_params(&refined.matrix, W, H);
        assert!(
            (p.scale - 1.012).abs() < tolerance,
            "{optimizer:?} bounded={bounded}: scale {} vs 1.012",
            p.scale
        );
        assert!(
            p.rotate.abs() < 1.0e-9,
            "{optimizer:?}: rotation must stay at the seed's 0, got {}",
            p.rotate
        );
    }
}

/// Strong focus breathing (12 % scale from the identity seed, as between the
/// first and last frame of a deep series) is recovered by every optimiser with
/// bounded search off. The narrow windows used to apply here too and capped the
/// scale at 1.05, which left the frames misaligned (echoed edges).
#[test]
fn default_search_recovers_strong_breathing() {
    let truth = RegistrationParams {
        tx: -0.01,
        ty: 0.008,
        scale: 1.12,
        rotate: 0.0,
    };
    let truth_m = params_to_matrix(&truth, W, H).expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth_m);
    let pyramid = LumaPyramid::new(&reference);

    for motion in [Motion::Translation, Motion::Similarity, Motion::Affine] {
        for optimizer in [Optimizer::Auto, Optimizer::LucasKanade, Optimizer::NelderMead] {
            let cfg = MotionRefine {
                optimizer,
                max_iterations: 200,
                ..MotionRefine::new(motion)
            };
            let refined = refine_motion(&reference, &pyramid, &source, &Matrix3::identity(), &cfg)
                .expect("refinement succeeds");
            let p = matrix_to_affine_params(&refined.matrix, W, H);
            assert!(
                (p.scale - 1.12).abs() < 0.01,
                "{motion:?} {optimizer:?}: scale {} vs 1.12",
                p.scale
            );
        }
    }

    // Bounded search keeps its narrow window: the scale stops at the seed + 5 %.
    let cfg = MotionRefine {
        optimizer: Optimizer::LucasKanade,
        bounded: true,
        ..MotionRefine::new(Motion::Similarity)
    };
    let refined = refine_motion(&reference, &pyramid, &source, &Matrix3::identity(), &cfg)
        .expect("refinement succeeds");
    let p = matrix_to_params(&refined.matrix, W, H);
    assert!(p.scale <= 1.05 + 1.0e-6, "bounded scale {} beyond 1.05", p.scale);
}

/// `Shift` keeps scale and rotation at the seed's, whatever the source does.
#[test]
fn shift_motion_pins_scale_and_rotation() {
    let truth_m = AffineParams {
        tx: 0.01,
        scale: 1.01,
        ..AffineParams::identity()
    }
    .to_matrix(W, H)
    .expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth_m);
    let pyramid = LumaPyramid::new(&reference);
    for optimizer in [Optimizer::LucasKanade, Optimizer::NelderMead] {
        let cfg = MotionRefine {
            optimizer,
            bounded: true,
            ..MotionRefine::new(Motion::Shift)
        };
        let refined = refine_motion(&reference, &pyramid, &source, &Matrix3::identity(), &cfg)
            .expect("refinement succeeds");
        let m = refined.matrix;
        assert!(
            (m[(0, 0)] - 1.0).abs() < 1.0e-6 && (m[(1, 1)] - 1.0).abs() < 1.0e-6,
            "{optimizer:?}: diagonal must stay 1, got {m}"
        );
        assert!(
            m[(0, 1)].abs() < 1.0e-6 && m[(1, 0)].abs() < 1.0e-6,
            "{optimizer:?}: off-diagonal must stay 0, got {m}"
        );
    }
}

/// The optimiser dispatch: `Auto` uses Lucas-Kanade when it improves, the forced
/// optimisers are what they say, and all find the shift.
#[test]
fn optimizers_dispatch_and_agree_on_a_similarity() {
    let truth = RegistrationParams {
        tx: 0.012,
        ty: -0.008,
        scale: 1.006,
        rotate: 0.01,
    };
    let truth_m = params_to_matrix(&truth, W, H).expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth_m);
    let pyramid = LumaPyramid::new(&reference);

    let run = |optimizer| {
        let cfg = MotionRefine {
            optimizer,
            bounded: true,
            ..MotionRefine::new(Motion::Similarity)
        };
        refine_motion(&reference, &pyramid, &source, &Matrix3::identity(), &cfg)
            .expect("refinement succeeds")
    };
    let auto = run(Optimizer::Auto);
    assert_eq!(auto.engine, Engine::LucasKanade);
    assert!(!auto.fell_back);
    let lk = run(Optimizer::LucasKanade);
    assert_eq!(lk.engine, Engine::LucasKanade);
    let nm = run(Optimizer::NelderMead);
    assert_eq!(nm.engine, Engine::NelderMead);
    assert!(!nm.fell_back);

    assert_eq!(
        bits(&auto.matrix),
        bits(&lk.matrix),
        "Auto is LK when LK improves"
    );
    assert!(corner_error(&lk.matrix, &truth_m, W, H) < 0.05);
    assert!(
        corner_error(&nm.matrix, &truth_m, W, H) < 0.3,
        "NM corner error {}",
        corner_error(&nm.matrix, &truth_m, W, H)
    );
    assert!(corner_error(&lk.matrix, &nm.matrix, W, H) < 0.35);
}

/// A mismatch between the frames is an error for every optimiser (and for `Auto`
/// after its fallback).
#[test]
fn every_optimizer_rejects_mismatched_frames() {
    let a = reference(64, 64);
    let b = reference(96, 64);
    let pyramid = LumaPyramid::new(&a);
    for optimizer in [
        Optimizer::Auto,
        Optimizer::LucasKanade,
        Optimizer::NelderMead,
    ] {
        let cfg = MotionRefine {
            optimizer,
            ..MotionRefine::new(Motion::Similarity)
        };
        let result = refine_motion(&a, &pyramid, &b, &Matrix3::identity(), &cfg);
        assert!(
            matches!(result, Err(StackerError::AlignmentFailed(_))),
            "{optimizer:?}: {result:?}"
        );
    }
}

/// The wrapper agrees with the free function it calls.
#[test]
fn lk_free_function_and_dispatcher_agree() {
    let truth = known_affine().to_matrix(W, H).expect("finite");
    let reference = reference(W, H);
    let source = moved(W, H, &truth);
    let pyramid = LumaPyramid::new(&reference);
    let cfg = MotionRefine {
        optimizer: Optimizer::LucasKanade,
        ..MotionRefine::new(Motion::Affine)
    };
    let via = refine_motion(&reference, &pyramid, &source, &Matrix3::identity(), &cfg)
        .expect("refinement succeeds");
    let direct = refine_alignment_lk(
        &reference,
        &source,
        &Matrix3::identity(),
        &Motion::Affine.bounded_options(),
    )
    .expect("refinement succeeds");
    assert_eq!(bits(&via.matrix), bits(&direct.matrix));
}

// ── the identity gate ─────────────────────────────────────────────────────────

/// A matrix that really aligns the frames is kept, at a size where the gate has
/// to halve the planes (more than 256 pixels on the short side).
#[test]
fn identity_gate_keeps_a_good_refinement() {
    for &(w, h) in &[(96_usize, 80_usize), (600, 520)] {
        let truth = RegistrationParams {
            tx: 2.0 / w as f64,
            ty: -1.0 / h as f64,
            scale: 1.0,
            rotate: 0.0,
        };
        let truth_m = params_to_matrix(&truth, w, h).expect("finite");
        let reference = reference(w, h);
        let source = moved(w, h, &truth_m);
        assert!(
            refined_beats_identity(&reference, &source, &truth_m),
            "{w}x{h}: a correct matrix must beat the identity"
        );
        let gated = apply_identity_gate(&reference, &source, &truth_m);
        assert_eq!(bits(&gated), bits(&truth_m), "{w}x{h}: the matrix is kept");
    }
}

/// A refinement that is worse than doing nothing falls back to the identity: a
/// wrong shift on frames that already agree, and a matrix that moves most of the
/// frame out of view.
#[test]
fn identity_gate_rejects_a_refinement_worse_than_identity() {
    let (w, h) = (96_usize, 80_usize);
    let reference = reference(w, h);
    let same = reference.clone();

    let mut wrong = Matrix3::<f32>::identity();
    wrong[(0, 2)] = 9.0;
    wrong[(1, 2)] = -6.0;
    assert!(!refined_beats_identity(&reference, &same, &wrong));
    let gated = apply_identity_gate(&reference, &same, &wrong);
    assert_eq!(
        bits(&gated),
        bits(&Matrix3::identity()),
        "identity returned"
    );

    let mut away = Matrix3::<f32>::identity();
    away[(0, 2)] = w as f32 * 0.8;
    assert!(!refined_beats_identity(&reference, &same, &away));
}

/// Equal scores keep the refined matrix (no margin), and unusable input is
/// reported as "do not keep".
#[test]
fn identity_gate_ties_and_bad_input() {
    let (w, h) = (96_usize, 80_usize);
    let reference = reference(w, h);
    // The identity against itself is a tie: kept.
    assert!(refined_beats_identity(
        &reference,
        &reference,
        &Matrix3::identity()
    ));
    // Mismatched sizes and non-finite matrices are never kept.
    let other = self::reference(w + 8, h);
    assert!(!refined_beats_identity(
        &reference,
        &other,
        &Matrix3::identity()
    ));
    let mut nan = Matrix3::<f32>::identity();
    nan[(1, 1)] = f32::NAN;
    assert!(!refined_beats_identity(&reference, &reference, &nan));
}
