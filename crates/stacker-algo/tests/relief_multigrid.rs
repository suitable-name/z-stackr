#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

//! Tests of the `Relief` multigrid engine: the solver, the layer operators, the
//! whole-image depth and the compositing.

use stacker_algo::relief::multigrid::{
    MultigridLayer, MultigridSolver, composite_depth, confidence_weights, crop_plane,
    multigrid_depth, update_argmax,
};
use stacker_core::image::PlanarImage;

/// A deterministic pseudo-random generator (LCG), good enough for test data.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.0 >> 33).expect("31 bits fit")
    }
}

/// Targets and weights with the left column pinned to 0 and the right column to 1.
fn pinned_columns(w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut target = vec![0.0_f32; w * h];
    let mut weight = vec![0.0_f32; w * h];
    for y in 0..h {
        weight[y * w] = 1.0;
        target[y * w] = 0.0;
        weight[y * w + (w - 1)] = 1.0;
        target[y * w + (w - 1)] = 1.0;
    }
    (target, weight)
}

/// A constant target field with full weight everywhere solves back to that
/// constant: every pixel is pinned, and the restriction and prolongation keep
/// pinned values.
#[test]
fn constant_target_with_full_weight_solves_to_constant() {
    const CONST: f32 = 0.42;
    for (w, h) in [(10, 8), (1, 1), (2, 2), (3, 7), (9, 9), (16, 5), (33, 17)] {
        let target = vec![CONST; w * h];
        let weight = vec![1.0; w * h];

        let mut solver = MultigridSolver::new(w, h, &target, &weight);
        solver.solve();
        let solution = solver.get_solution();

        assert_eq!(solution.len(), w * h);
        for (i, &v) in solution.iter().enumerate() {
            assert!(v.is_finite(), "{w}x{h}: solution[{i}] is not finite: {v}");
            assert!(
                (v - CONST).abs() < 1e-4,
                "{w}x{h}: solution[{i}] = {v}, expected {CONST}"
            );
        }
    }
}

/// Pinned pixels keep their target after `solve()`; they are boundary conditions,
/// not free variables.
#[test]
fn constrained_pixels_keep_their_target_values() {
    let (w, h) = (12, 12);
    let (target, weight) = pinned_columns(w, h);

    let mut solver = MultigridSolver::new(w, h, &target, &weight);
    solver.solve();
    let solution = solver.get_solution();

    for y in 0..h {
        assert!(
            solution[y * w].abs() < 1e-6,
            "left pixel at row {y}: {}",
            solution[y * w]
        );
        assert!(
            (solution[y * w + (w - 1)] - 1.0).abs() < 1e-6,
            "right pixel at row {y}: {}",
            solution[y * w + (w - 1)]
        );
    }
}

/// An unconstrained region between a pinned 0 and a pinned 1 is filled with
/// intermediate values: finite, inside the bracket of the constraints, and a real
/// blend in the middle.
#[test]
fn unconstrained_hole_is_filled_with_intermediate_values() {
    let (w, h) = (16, 16);
    let (target, weight) = pinned_columns(w, h);

    let mut solver = MultigridSolver::new(w, h, &target, &weight);
    solver.solve();
    let solution = solver.get_solution();

    for (i, &v) in solution.iter().enumerate() {
        assert!(v.is_finite(), "solution[{i}] is not finite: {v}");
        assert!(
            (0.0..=1.0).contains(&v),
            "solution[{i}] = {v} outside the constraint bracket [0, 1]"
        );
    }
    let mid = solution[(h / 2) * w + w / 2];
    assert!(
        mid > 0.05 && mid < 0.95,
        "centre pixel {mid} is not a genuine interior blend"
    );
}

/// A ramp pinned at both ends fills in (close to) linearly: the Laplace solution
/// with these boundary values is linear in x.
#[test]
fn a_ramp_pinned_at_both_ends_fills_linearly() {
    let (w, h) = (33, 17);
    let (target, weight) = pinned_columns(w, h);

    let mut solver = MultigridSolver::new(w, h, &target, &weight);
    solver.solve();
    let solution = solver.get_solution();

    let mut worst = 0.0_f32;
    for y in 0..h {
        for x in 0..w {
            let expected = x as f32 / (w - 1) as f32;
            worst = worst.max((solution[y * w + x] - expected).abs());
        }
    }
    assert!(
        worst < 0.15,
        "worst deviation from the linear ramp is {worst}"
    );

    // And the three thirds of the middle row are ordered.
    let row = &solution[(h / 2) * w..(h / 2 + 1) * w];
    let third =
        |range: std::ops::Range<usize>| row[range.clone()].iter().sum::<f32>() / range.len() as f32;
    assert!(third(0..11) < third(11..22) && third(11..22) < third(22..33));
}

/// With no pinned pixel at all there is nothing to diffuse: the depth stays 0.
#[test]
fn unpinned_only_input_stays_zero() {
    for (w, h) in [(8, 8), (7, 5), (1, 4)] {
        let target = vec![3.0_f32; w * h];
        let weight = vec![0.0_f32; w * h];
        let mut solver = MultigridSolver::new(w, h, &target, &weight);
        solver.solve();
        let solution = solver.get_solution();
        assert_eq!(solution.len(), w * h);
        assert!(solution.iter().all(|&v| v == 0.0), "{w}x{h}");
    }
}

/// Odd and even sizes, thin images and a random pin pattern: the solution has the
/// right size, is finite, keeps every pinned value and stays inside the range of
/// the targets.
#[test]
fn every_size_keeps_pins_and_the_target_range() {
    let sizes = [
        (1, 1),
        (1, 7),
        (7, 1),
        (2, 2),
        (2, 5),
        (3, 3),
        (4, 6),
        (5, 4),
        (8, 8),
        (15, 9),
        (16, 16),
        (31, 33),
        (64, 3),
    ];
    for (w, h) in sizes {
        let mut rng = Lcg(0x1234_5678 + (w * 131 + h) as u64);
        let target: Vec<f32> = (0..w * h).map(|_| (rng.next() % 5) as f32).collect();
        let weight: Vec<f32> = (0..w * h)
            .map(|_| {
                if rng.next().is_multiple_of(4) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let mut solver = MultigridSolver::new(w, h, &target, &weight);
        solver.solve();
        let solution = solver.get_solution();
        assert_eq!(solution.len(), w * h, "{w}x{h}");

        let any_pinned = weight.iter().any(|&b| b > 0.0);
        for i in 0..w * h {
            let v = solution[i];
            assert!(v.is_finite(), "{w}x{h} sample {i}");
            if weight[i] > 0.0 {
                assert!(
                    (v - target[i]).abs() < 1e-6,
                    "{w}x{h}: pin {i} drifted to {v}"
                );
            } else if any_pinned {
                assert!(
                    (-1e-4..=4.0 + 1e-4).contains(&v),
                    "{w}x{h}: sample {i} = {v} outside the target range"
                );
            }
        }
    }
}

/// `from_index` is `new` with the index as an `f32` target.
#[test]
fn from_index_matches_new_with_float_targets() {
    let (w, h) = (21, 14);
    let mut rng = Lcg(77);
    let index: Vec<u16> = (0..w * h)
        .map(|_| u16::try_from(rng.next() % 9).expect("small"))
        .collect();
    let weight: Vec<f32> = (0..w * h)
        .map(|_| {
            if rng.next().is_multiple_of(5) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let target: Vec<f32> = index.iter().map(|&i| f32::from(i)).collect();

    let mut from_floats = MultigridSolver::new(w, h, &target, &weight);
    from_floats.solve();
    let mut from_index = MultigridSolver::from_index(w, h, &index, &weight);
    from_index.solve();

    let (left, right) = (from_floats.get_solution(), from_index.get_solution());
    assert!(
        left.iter()
            .zip(&right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

/// Restriction averages `a` and `b` and takes the `b`-weighted mean of `c`; a 1x1
/// corner block keeps its target (it was halved before the fix).
#[test]
fn restriction_means_and_a_corner_block_keeps_its_target() {
    // A 3x3 fine level restricts to 2x2; the bottom-right coarse sample covers the
    // single fine sample (2, 2).
    let mut fine = MultigridLayer::new(3, 3);
    fine.a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    // Pin (0,0) to 4 and (1,0) to 8, and the corner (2,2) to 8.
    fine.b[0] = 1.0;
    fine.c[0] = 4.0;
    fine.b[1] = 1.0;
    fine.c[1] = 8.0;
    fine.b[8] = 1.0;
    fine.c[8] = 8.0;

    let mut coarse = MultigridLayer::new(2, 2);
    fine.restrict_to(&mut coarse);

    // Block (0,0): samples 0, 1, 3, 4.
    assert!((coarse.a[0] - (1.0 + 2.0 + 4.0 + 5.0) / 4.0).abs() < 1e-6);
    assert!((coarse.b[0] - 0.5).abs() < 1e-6);
    assert!((coarse.c[0] - 6.0).abs() < 1e-6, "mean of 4 and 8 is 6");
    // Block (1,0): the column {2, 5} (a 1x2 block), no pins.
    assert!((coarse.a[1] - f32::midpoint(3.0, 6.0)).abs() < 1e-6);
    assert!(coarse.b[1].abs() < 1e-6 && coarse.c[1].abs() < 1e-6);
    // Block (0,1): the row {6, 7} (a 2x1 block), no pins.
    assert!((coarse.a[2] - f32::midpoint(7.0, 8.0)).abs() < 1e-6);
    // Block (1,1): the single sample 8, pinned to 8.
    assert!((coarse.a[3] - 9.0).abs() < 1e-6);
    assert!((coarse.b[3] - 1.0).abs() < 1e-6);
    assert!((coarse.c[3] - 8.0).abs() < 1e-6, "got {}", coarse.c[3]);
}

/// Prolongation interpolates bilinearly and writes every sample of a block,
/// including the one below the coarse sample.
#[test]
fn prolongation_is_bilinear_and_writes_all_four_samples() {
    let mut coarse = MultigridLayer::new(2, 2);
    coarse.a = vec![0.0, 2.0, 4.0, 6.0];
    let mut fine = MultigridLayer::new(3, 3);
    fine.a = vec![-1.0; 9];

    coarse.prolong_from(&mut fine);

    let expected = [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0];
    for (i, (&got, &want)) in fine.a.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "sample {i}: {got}, expected {want}"
        );
    }

    // A pinned fine sample takes its own target instead.
    fine.b[4] = 1.0;
    fine.c[4] = 10.0;
    coarse.prolong_from(&mut fine);
    assert!((fine.a[4] - 10.0).abs() < 1e-6);
    // A fully pinned coarse sample leaves its block alone.
    coarse.b[0] = 1.0;
    fine.a[0] = 123.0;
    coarse.prolong_from(&mut fine);
    assert!((fine.a[0] - 123.0).abs() < 1e-6);
}

/// One Jacobi sweep: free samples take the mean of their in-bounds 3x3
/// neighbourhood of the *old* values, pinned samples their target.
#[test]
fn relax_is_a_jacobi_sweep() {
    let mut layer = MultigridLayer::new(3, 3);
    layer.a[4] = 9.0;
    layer.relax();
    assert!((layer.a[4] - 1.0).abs() < 1e-6, "centre: mean of all nine");
    assert!((layer.a[0] - 2.25).abs() < 1e-6, "corner: mean of four");
    assert!((layer.a[1] - 1.5).abs() < 1e-6, "edge: mean of six");

    let mut pinned = MultigridLayer::new(3, 3);
    pinned.a[4] = 9.0;
    pinned.b[0] = 1.0;
    pinned.c[0] = 5.0;
    pinned.relax();
    assert!((pinned.a[0] - 5.0).abs() < 1e-6);
}

/// Strictly greater wins, so the earlier frame wins a tie; a NaN never wins.
#[test]
fn argmax_keeps_the_first_of_equal_values() {
    let mut best = vec![f32::NEG_INFINITY; 4];
    let mut index = vec![0_u16; 4];
    update_argmax(&mut best, &mut index, 0, &[1.0, 2.0, 3.0, f32::NAN]);
    update_argmax(&mut best, &mut index, 1, &[1.0, 5.0, 2.0, f32::NAN]);
    update_argmax(&mut best, &mut index, 2, &[0.5, 5.0, 4.0, 1.0]);
    assert_eq!(index, vec![0, 1, 2, 2]);
    assert_eq!(best, vec![1.0, 5.0, 4.0, 1.0]);
}

#[test]
fn confidence_weights_use_the_inclusive_threshold() {
    let weights = confidence_weights(&[0.0, 0.5, 0.6, f32::NAN, 2.0], 0.5);
    assert_eq!(weights, vec![0.0, 1.0, 1.0, 0.0, 1.0]);
}

/// Index and weight planes of an image where the left band is pinned to frame 1,
/// the right band to frame 3 and the middle is free.
fn banded_planes(w: usize, h: usize, band: usize) -> (Vec<u16>, Vec<f32>) {
    let mut index = vec![0_u16; w * h];
    let mut weight = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..band {
            index[y * w + x] = 1;
            weight[y * w + x] = 1.0;
            index[y * w + (w - 1 - x)] = 3;
            weight[y * w + (w - 1 - x)] = 1.0;
        }
    }
    (index, weight)
}

#[test]
fn depth_stays_inside_the_pinned_range_and_fills_the_gap() {
    let (w, h) = (40, 24);
    let (index, weight) = banded_planes(w, h, 12);

    // Radius 1 is the identity: the pins are exact.
    let raw = multigrid_depth(w, h, &index, &weight, 1);
    assert_eq!(raw.len(), w * h);
    for y in 0..h {
        for x in 0..12 {
            assert!((raw[y * w + x] - 1.0).abs() < 1e-6);
            assert!((raw[y * w + (w - 1 - x)] - 3.0).abs() < 1e-6);
        }
    }

    // Unpinned samples start at 0 and every step is an average, so the whole field
    // stays inside [0, 3] (the pins are 1 and 3).
    for radius in [1, 2, 4, 8] {
        let depth = multigrid_depth(w, h, &index, &weight, radius);
        for (i, &v) in depth.iter().enumerate() {
            assert!(
                (-1e-3..=3.0 + 1e-3).contains(&v),
                "radius {radius}: depth[{i}] = {v}"
            );
        }
        let centre = depth[(h / 2) * w + w / 2];
        assert!(
            centre > 1.3 && centre < 2.7,
            "radius {radius}: centre depth {centre} is not a blend of frames 1 and 3"
        );
    }
}

#[test]
fn depth_of_an_empty_or_unpinned_image() {
    assert_eq!(multigrid_depth(0, 0, &[], &[], 2), vec![]);
    assert_eq!(multigrid_depth(0, 5, &[], &[], 2), vec![]);
    let depth = multigrid_depth(6, 4, &[2; 24], &[0.0; 24], 2);
    assert_eq!(depth.len(), 24);
    assert!(depth.iter().all(|&v| v == 0.0));
}

/// Tile invariance: the depth is a function of the whole-image planes only, so the
/// thread count (and so the parallel row split) changes nothing, bit for bit; a run
/// is repeatable.
#[test]
fn depth_is_independent_of_the_thread_count() {
    let (w, h) = (200, 100);
    let mut rng = Lcg(2024);
    let index: Vec<u16> = (0..w * h)
        .map(|_| u16::try_from(rng.next() % 5).expect("small"))
        .collect();
    let weight: Vec<f32> = (0..w * h)
        .map(|_| {
            if rng.next().is_multiple_of(3) {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a one-thread pool");
    let single = pool.install(|| multigrid_depth(w, h, &index, &weight, 4));
    let multi = multigrid_depth(w, h, &index, &weight, 4);
    let again = multigrid_depth(w, h, &index, &weight, 4);

    assert_eq!(single.len(), w * h);
    assert!(
        single
            .iter()
            .zip(&multi)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        multi
            .iter()
            .zip(&again)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

/// A frame whose three planes are `k`, `2k` and `-k`.
fn constant_frame(w: usize, h: usize, k: f32) -> PlanarImage<f32> {
    PlanarImage::from_planes(w, h, vec![k; w * h], vec![2.0 * k; w * h], vec![-k; w * h])
        .expect("one sample per pixel")
}

#[test]
fn compositing_interpolates_the_two_nearest_frames() {
    let frames: Vec<PlanarImage<f32>> = (0..5).map(|k| constant_frame(4, 2, k as f32)).collect();
    let depth = [-2.0, 0.0, 0.25, 1.5, 3.999, 4.0, 9.0, f32::NAN];
    let fused = composite_depth(&frames, &depth);

    let want = [0.0, 0.0, 0.25, 1.5, 3.999, 4.0, 4.0, 0.0];
    for (i, &w) in want.iter().enumerate() {
        assert!(
            (fused.luma[i] - w).abs() < 1e-5,
            "luma {i}: {}",
            fused.luma[i]
        );
        assert!(
            2.0_f32.mul_add(-w, fused.chroma_a[i]).abs() < 1e-5,
            "chroma a {i}"
        );
        assert!((fused.chroma_b[i] + w).abs() < 1e-5, "chroma b {i}");
    }
}

#[test]
fn compositing_nothing_or_one_frame() {
    let empty = composite_depth(&[], &[]);
    assert_eq!((empty.width, empty.height), (0, 0));

    let single = [constant_frame(3, 3, 7.0)];
    let fused = composite_depth(&single, &[0.0, 1.0, 2.0, -1.0, 0.5, 9.0, 0.0, 0.0, 0.0]);
    assert!(fused.luma.iter().all(|&v| (v - 7.0).abs() < 1e-6));
}

/// A tile of the stack composited with the crop of the whole-image depth equals
/// the tile of the whole-image composite.
#[test]
fn a_tile_composite_equals_the_tile_of_the_whole_composite() {
    let (w, h) = (23, 17);
    let mut rng = Lcg(5);
    let frames: Vec<PlanarImage<f32>> = (0..4)
        .map(|_| {
            let plane = |rng: &mut Lcg| -> Vec<f32> {
                (0..w * h)
                    .map(|_| (rng.next() % 1000) as f32 / 1000.0)
                    .collect()
            };
            let (l, a, b) = (plane(&mut rng), plane(&mut rng), plane(&mut rng));
            PlanarImage::from_planes(w, h, l, a, b).expect("one sample per pixel")
        })
        .collect();
    let depth: Vec<f32> = (0..w * h)
        .map(|_| (rng.next() % 3001) as f32 / 1000.0)
        .collect();
    let whole = composite_depth(&frames, &depth);

    let (x0, y0, tw, th) = (5, 3, 11, 9);
    let tile_frames: Vec<PlanarImage<f32>> = frames
        .iter()
        .map(|f| {
            let crop = |plane: &[f32]| crop_plane(plane, w, (x0, y0), (tw, th));
            PlanarImage::from_planes(tw, th, crop(&f.luma), crop(&f.chroma_a), crop(&f.chroma_b))
                .expect("one sample per pixel")
        })
        .collect();
    let tile = composite_depth(&tile_frames, &crop_plane(&depth, w, (x0, y0), (tw, th)));

    assert_eq!(tile.luma, crop_plane(&whole.luma, w, (x0, y0), (tw, th)));
    assert_eq!(
        tile.chroma_a,
        crop_plane(&whole.chroma_a, w, (x0, y0), (tw, th))
    );
    assert_eq!(
        tile.chroma_b,
        crop_plane(&whole.chroma_b, w, (x0, y0), (tw, th))
    );
}

#[test]
fn crop_plane_copies_the_window() {
    let plane: Vec<f32> = (0..30).map(|v| v as f32).collect();
    // A 6x5 plane, the 3x2 window at (2, 1).
    assert_eq!(
        crop_plane(&plane, 6, (2, 1), (3, 2)),
        vec![8.0, 9.0, 10.0, 14.0, 15.0, 16.0]
    );
}
