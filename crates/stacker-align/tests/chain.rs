#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::float_cmp
)]

//! Chain (sequential) alignment: composing the per-step matrices and the coverage
//! of the result.
//!
//! The frames are an analytic texture drifting by 1.5 px per frame, so the truth
//! of every cumulative matrix is a pure shift of `1.5 * k` px.  Two ways of
//! running the chain are exercised (see `stacker_align::chain`):
//!
//! - **A**: the raw previous frame is the reference; the steps are composed;
//! - **B**: the previous frame *warped into frame 0's geometry* is the
//!   reference; the refiner's result is already the cumulative matrix.

use nalgebra::Matrix3;
use stacker_align::{
    chain::{chain_coverage, compose_chain},
    refine::{BoundedRefineOptions, Motion, refine_alignment_lk},
    transform::{coverage_mask, intersect_coverage, largest_true_rectangle},
};
use stacker_core::image::PlanarImage;

const W: usize = 128;
const H: usize = 96;
/// Drift per frame, in pixels.
const STEP: f64 = 1.5;

/// A smooth multi-directional texture, drifted by `dx` pixels in x.
fn texture_at(x: f64, y: f64) -> f32 {
    (0.5 + 0.18 * (0.21 * x + 0.13 * y).sin()
        + 0.18 * (0.17 * x - 0.19 * y).cos()
        + 0.08 * (0.07 * x).sin() * (0.05 * y).cos()) as f32
}

/// Frame `k` of the drifting sequence: `frame_k(q) = texture(q - (STEP k, 0))`,
/// so the content moves by `+STEP k` px and the cumulative matrix of frame `k`
/// (frame-0 coordinates to frame-`k` coordinates) is `T(STEP k, 0)`.
fn frame(k: usize) -> PlanarImage<f32> {
    let dx = STEP * k as f64;
    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            img.luma[y * W + x] = texture_at(x as f64 - dx, y as f64);
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

/// Shift-only Lucas-Kanade options (the drift is a pure shift).
fn shift_options() -> BoundedRefineOptions {
    Motion::Shift.bounded_options()
}

/// Scheme A: refine frame `k` against the raw frame `k - 1` and compose.
#[test]
fn three_frames_with_known_shifts_compose_to_the_total_shift() {
    let (f0, f1, f2) = (frame(0), frame(1), frame(2));
    let opts = shift_options();

    let r1 = refine_alignment_lk(&f0, &f1, &Matrix3::identity(), &opts)
        .expect("step 1")
        .matrix;
    let r2 = refine_alignment_lk(&f1, &f2, &Matrix3::identity(), &opts)
        .expect("step 2")
        .matrix;
    // Each step is the per-frame drift ...
    assert!(
        (f64::from(r1[(0, 2)]) - STEP).abs() < 0.05,
        "step 1 shift {}",
        r1[(0, 2)]
    );
    assert!(
        (f64::from(r2[(0, 2)]) - STEP).abs() < 0.05,
        "step 2 shift {}",
        r2[(0, 2)]
    );

    // ... and the cumulative matrices are the total drift.
    let c1 = compose_chain(&Matrix3::identity(), &r1);
    let c2 = compose_chain(&c1, &r2);
    assert!(
        (f64::from(c2[(0, 2)]) - 2.0 * STEP).abs() < 0.05,
        "composed shift {} px, expected {}",
        c2[(0, 2)],
        2.0 * STEP
    );
    assert!(
        f64::from(c2[(1, 2)]).abs() < 0.05,
        "composed y shift {}",
        c2[(1, 2)]
    );
    assert!((c2[(0, 0)] - 1.0).abs() < 1.0e-6 && (c2[(1, 1)] - 1.0).abs() < 1.0e-6);
}

/// Scheme B: the reference is the previous frame already in frame 0's geometry
/// (here the analytic stand-in for the warped frame, which has no undefined
/// border); seeded with the previous cumulative matrix the refiner returns the
/// cumulative matrix itself, and composing it again would apply the drift twice.
#[test]
fn a_warped_reference_yields_the_cumulative_matrix_directly() {
    let f0 = frame(0);
    let f2 = frame(2);
    let c1 = translation(STEP as f32, 0.0);

    let c2 = refine_alignment_lk(&f0, &f2, &c1, &shift_options())
        .expect("step 2")
        .matrix;
    assert!(
        (f64::from(c2[(0, 2)]) - 2.0 * STEP).abs() < 0.05,
        "cumulative shift {} px",
        c2[(0, 2)]
    );

    let doubled = compose_chain(&c1, &c2);
    assert!(
        (f64::from(doubled[(0, 2)]) - 2.0 * STEP).abs() > 1.0,
        "composing a cumulative matrix again must be visibly wrong, got {}",
        doubled[(0, 2)]
    );
}

/// The composition order, derived in the module documentation: `C_k = R_k C_{k-1}`
/// maps frame-0 coordinates to frame-`k` coordinates.
#[test]
fn composition_order_matches_the_derivation() {
    // Content scaled by 2 about the origin, then shifted by 10: the order matters.
    let mut prev = Matrix3::<f32>::identity();
    prev[(0, 0)] = 2.0;
    prev[(1, 1)] = 2.0;
    let step = translation(10.0, 0.0);
    let c = compose_chain(&prev, &step);
    // A frame-0 point p goes to prev * p (frame k-1) and on to step * (prev * p).
    let p = nalgebra::Vector3::new(3.0_f32, 4.0, 1.0);
    let expected = step * (prev * p);
    let got = c * p;
    assert!((expected - got).norm() < 1.0e-6, "{expected} vs {got}");
    assert!((got.x - 16.0).abs() < 1.0e-6 && (got.y - 8.0).abs() < 1.0e-6);
    // The other order would be a different matrix.
    assert!((prev * step - c).norm() > 1.0);
}

/// A long chain does not drift numerically: a hundred 1.5 px steps compose to
/// 150 px.
#[test]
fn a_long_chain_composes_without_rounding_drift() {
    let step = translation(1.5, 0.25);
    let mut c = Matrix3::<f32>::identity();
    for _ in 0..100 {
        c = compose_chain(&c, &step);
    }
    assert!((c[(0, 2)] - 150.0).abs() < 1.0e-3, "x {}", c[(0, 2)]);
    assert!((c[(1, 2)] - 25.0).abs() < 1.0e-3, "y {}", c[(1, 2)]);
}

/// The coverage of the last frame is the region valid under the *composed*
/// matrix, not under the last step.
#[test]
fn coverage_follows_the_composed_matrix() {
    let step = translation(1.5, 0.0);
    let composed = compose_chain(&step, &step);
    assert_eq!(composed[(0, 2)], 3.0);

    let mask = chain_coverage(&composed, W, H);
    assert_eq!(mask.len(), W * H);
    // Output pixel (x, y) receives data iff C (x, y) = (x + 3, y) is inside the
    // frame, [0, W - 1] x [0, H - 1].
    for y in 0..H {
        for x in 0..W {
            let valid = (x as f32 + 3.0) <= (W - 1) as f32;
            assert_eq!(mask[y * W + x], valid, "pixel ({x}, {y})");
        }
    }

    // The mask built from the last step alone (the GitHub z-stackr mask) claims
    // more area: the columns whose shifted position is past the right edge under
    // 3 px but not under 1.5 px.
    let step_only = chain_coverage(&step, W, H);
    let differing = mask.iter().zip(&step_only).filter(|(a, b)| a != b).count();
    assert!(differing > 0, "the step-only mask must differ");
    assert!(
        step_only[W - 3],
        "valid under the step alone (x + 1.5 <= 127)"
    );
    assert!(
        !mask[W - 3],
        "but not under the composed matrix (x + 3 > 127)"
    );

    // It equals the mask of the matrix the frame is warped with, C^-1.
    let applied = composed.try_inverse().expect("invertible");
    assert_eq!(mask, coverage_mask(&applied, W, H));
}

/// Folding the chain's coverage masks and cropping leaves exactly the region all
/// frames share.
#[test]
fn common_area_of_a_chain_is_the_intersection_of_the_cumulative_masks() {
    let mut common = vec![true; W * H];
    for k in 1..=3_u8 {
        let c = translation(STEP as f32 * f32::from(k), 0.0);
        intersect_coverage(&mut common, &chain_coverage(&c, W, H));
    }
    let (x, y, w, h) = largest_true_rectangle(&common, W, H).expect("a common rectangle remains");
    // The last frame (shift 4.5 px) limits the width: x + 4.5 <= 127  =>  x <= 122.
    assert_eq!((x, y, h), (0, 0, H));
    assert_eq!(w, 123);
}

/// A singular matrix covers nothing.
#[test]
fn a_singular_cumulative_matrix_covers_nothing() {
    let mut m = Matrix3::<f32>::identity();
    m[(0, 0)] = 0.0;
    let mask = chain_coverage(&m, 8, 8);
    assert_eq!(mask.len(), 64);
    assert!(mask.iter().all(|&v| !v));
}
