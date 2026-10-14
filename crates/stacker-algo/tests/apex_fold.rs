//! The streaming Apex fold equals the batch fusion bit for bit, and the batch
//! fusion (with its precomputed grit energies) equals the old per-pixel
//! formulation, kept here as the reference.

use stacker_algo::{
    apex::{
        fuse::{
            ApexFold, MIN_PYRAMID_LEVELS, apex_fold_bytes_per_pixel, build_and_fuse_pyramids,
            fuse_pyramids, fuse_streaming,
        },
        pyramid::LaplacianPyramid,
    },
    strata::FnFrames,
};
use stacker_core::image::PlanarImage;

/// A small deterministic generator (xorshift), so the stacks are reproducible.
struct Rng(u64);

impl Rng {
    const fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "a 24-bit value and 2^24 are exact in f32"
    )]
    fn unit(&mut self) -> f32 {
        ((self.step() >> 40) as f32) / ((1_u64 << 24) as f32)
    }
}

/// A frame with all three planes random; `quantised` restricts the values to a
/// few levels so that exact energy ties are common.
fn random_frame(rng: &mut Rng, w: usize, h: usize, quantised: bool) -> PlanarImage<f32> {
    let plane = |rng: &mut Rng| -> Vec<f32> {
        (0..w * h)
            .map(|_| {
                if quantised {
                    f32::from(u8::try_from(rng.step() % 3).expect("below 3")) * 0.25
                } else {
                    // Exact: the unit value has 24 bits, doubling and the shift are exact.
                    rng.unit().mul_add(2.0, -1.0)
                }
            })
            .collect()
    };
    let luma = plane(rng);
    let chroma_a = plane(rng);
    let chroma_b = plane(rng);
    PlanarImage::from_planes(w, h, luma, chroma_a, chroma_b).expect("planes hold w * h samples")
}

fn stack(seed: u64, n: usize, w: usize, h: usize, quantised: bool) -> Vec<PlanarImage<f32>> {
    let mut rng = Rng(seed | 1);
    (0..n)
        .map(|_| random_frame(&mut rng, w, h, quantised))
        .collect()
}

fn assert_same_image(a: &PlanarImage<f32>, b: &PlanarImage<f32>, what: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{what}: size");
    for (name, pa, pb) in [
        ("luma", &a.luma, &b.luma),
        ("chroma_a", &a.chroma_a, &b.chroma_a),
        ("chroma_b", &a.chroma_b, &b.chroma_b),
    ] {
        for (i, (x, y)) in pa.iter().zip(pb).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: {name}[{i}]: {x} vs {y}");
        }
    }
}

fn assert_same_pyramid(a: &LaplacianPyramid, b: &LaplacianPyramid, what: &str) {
    assert_eq!(a.levels.len(), b.levels.len(), "{what}: level count");
    for (k, (la, lb)) in a.levels.iter().zip(&b.levels).enumerate() {
        assert_same_image(la, lb, &format!("{what} level {k}"));
    }
}

/// The per-pixel formulation `fuse_pyramids` used before the grit energies were
/// precomputed: every source's energy evaluated at every pixel (3×3 sums with
/// out-of-image neighbours skipped), argmax with a strict `>`, base averaged.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "the reference must round exactly like the unfused production arithmetic; \
              the casts are on tiny image dimensions"
)]
fn reference_fuse(pyramids: &[LaplacianPyramid], use_color: bool, grit: bool) -> LaplacianPyramid {
    let energy = |lvl: &PlanarImage<f32>, i: usize| -> f32 {
        let l = lvl.luma[i];
        let e = l * l;
        if use_color {
            let (a, b) = (lvl.chroma_a[i], lvl.chroma_b[i]);
            e + a * a + b * b
        } else {
            e
        }
    };
    let levels_count = pyramids[0].levels.len();
    let mut levels = Vec::new();
    for k in 0..levels_count {
        let (w, h) = (pyramids[0].levels[k].width, pyramids[0].levels[k].height);
        let (mut ol, mut oa, mut ob) = (
            vec![0.0_f32; w * h],
            vec![0.0_f32; w * h],
            vec![0.0_f32; w * h],
        );
        for row in 0..h {
            for col in 0..w {
                let i = row * w + col;
                if k == levels_count - 1 {
                    let n = pyramids.len() as f32;
                    let (mut sl, mut sa, mut sb) = (0.0_f32, 0.0_f32, 0.0_f32);
                    for p in pyramids {
                        sl += p.levels[k].luma[i];
                        sa += p.levels[k].chroma_a[i];
                        sb += p.levels[k].chroma_b[i];
                    }
                    ol[i] = sl / n;
                    oa[i] = sa / n;
                    ob[i] = sb / n;
                    continue;
                }
                let mut winner = 0;
                let mut best = f32::NEG_INFINITY;
                for (s, p) in pyramids.iter().enumerate() {
                    let lvl = &p.levels[k];
                    let e = if k == 0 && grit {
                        let mut e = 0.0_f32;
                        for dr in -1_isize..=1 {
                            for dc in -1_isize..=1 {
                                let nr = row as isize + dr;
                                let nc = col as isize + dc;
                                if nr >= 0 && nr < h as isize && nc >= 0 && nc < w as isize {
                                    e += energy(lvl, nr as usize * w + nc as usize);
                                }
                            }
                        }
                        e
                    } else {
                        energy(lvl, i)
                    };
                    if e > best {
                        best = e;
                        winner = s;
                    }
                }
                let lvl = &pyramids[winner].levels[k];
                ol[i] = lvl.luma[i];
                oa[i] = lvl.chroma_a[i];
                ob[i] = lvl.chroma_b[i];
            }
        }
        levels.push(PlanarImage::from_planes(w, h, ol, oa, ob).expect("one sample per pixel"));
    }
    LaplacianPyramid { levels }
}

fn fold_all(frames: &[PlanarImage<f32>], levels: usize, use_color: bool, grit: bool) -> ApexFold {
    let mut fold = ApexFold::new(levels, use_color, grit);
    for f in frames {
        fold.fold(f);
    }
    fold
}

/// Sizes (odd and even, square and not, small enough for the pyramid to run out
/// of levels), level counts, and the flag grid.
#[test]
fn batch_matches_the_per_pixel_reference() {
    for (si, &(w, h)) in [(1, 1), (3, 5), (8, 8), (17, 13), (33, 30)]
        .iter()
        .enumerate()
    {
        for &quantised in &[false, true] {
            let frames = stack(100 + si as u64, 4, w, h, quantised);
            for levels in [2, 3, 5] {
                let pyramids: Vec<_> = frames
                    .iter()
                    .map(|f| LaplacianPyramid::build(f, levels))
                    .collect();
                for use_color in [false, true] {
                    for grit in [false, true] {
                        let expected = reference_fuse(&pyramids, use_color, grit);
                        let got = fuse_pyramids(&pyramids, use_color, grit);
                        assert_same_pyramid(
                            &got,
                            &expected,
                            &format!("{w}x{h} q{quantised} L{levels} color{use_color} grit{grit}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fold_equals_batch_bit_for_bit() {
    for (si, &(w, h)) in [(2, 2), (7, 9), (16, 16), (31, 17), (40, 25)]
        .iter()
        .enumerate()
    {
        for &quantised in &[false, true] {
            let frames = stack(7 + si as u64, 5, w, h, quantised);
            for levels in 1..=8 {
                for use_color in [false, true] {
                    for grit in [false, true] {
                        let what =
                            format!("{w}x{h} q{quantised} L{levels} color{use_color} grit{grit}");
                        let batch = build_and_fuse_pyramids(&frames, levels, use_color, grit);
                        let folded = fold_all(&frames, levels, use_color, grit);
                        assert_eq!(folded.frames_folded(), frames.len());
                        let pyramid = folded.finish_pyramid();
                        assert_same_pyramid(&pyramid, &batch, &what);
                        let image = fold_all(&frames, levels, use_color, grit).finish();
                        assert_same_image(&image, &batch.reconstruct(), &what);
                    }
                }
            }
        }
    }
}

#[test]
fn fold_handles_exact_ties_like_batch() {
    // Identical frames: every energy ties, the first frame must win everywhere.
    let base = stack(3, 1, 12, 11, false).remove(0);
    let frames = vec![base.clone(), base.clone(), base];
    for grit in [false, true] {
        let batch = build_and_fuse_pyramids(&frames, 4, true, grit);
        let folded = fold_all(&frames, 4, true, grit).finish_pyramid();
        assert_same_pyramid(&folded, &batch, "identical frames");
    }
    // Constant frames: all detail is zero, so the chroma of the first frame wins.
    let flat = |v: f32, c: f32| {
        PlanarImage::from_planes(6, 6, vec![v; 36], vec![c; 36], vec![-c; 36]).expect("planes")
    };
    let frames = vec![flat(0.5, 0.1), flat(0.5, 0.9), flat(0.2, 0.3)];
    let batch = build_and_fuse_pyramids(&frames, 3, true, true);
    let folded = fold_all(&frames, 3, true, true).finish_pyramid();
    assert_same_pyramid(&folded, &batch, "flat frames");
}

#[test]
fn fold_with_nan_in_the_first_frame_matches_batch() {
    let mut frames = stack(21, 3, 10, 10, false);
    for i in [0, 13, 57, 99] {
        frames[0].luma[i] = f32::NAN;
    }
    for levels in [2, 4] {
        for grit in [false, true] {
            let pyramids: Vec<_> = frames
                .iter()
                .map(|f| LaplacianPyramid::build(f, levels))
                .collect();
            let reference = reference_fuse(&pyramids, false, grit);
            let folded = fold_all(&frames, levels, false, grit).finish_pyramid();
            // Bit patterns are compared, so NaN coefficients compare equal when
            // both sides copy them from the same source.
            assert_same_pyramid(&folded, &reference, "nan frame");
        }
    }
}

#[test]
fn a_single_frame_folds_to_itself_through_the_pyramid() {
    let frames = stack(5, 1, 9, 9, false);
    let batch = build_and_fuse_pyramids(&frames, 3, false, true);
    let folded = fold_all(&frames, 3, false, true).finish_pyramid();
    assert_same_pyramid(&folded, &batch, "one frame");
}

#[test]
fn preview_equals_the_batch_of_the_frames_so_far() {
    let frames = stack(11, 6, 19, 14, false);
    let mut fold = ApexFold::new(4, false, true);
    assert_eq!(fold.preview().width, 0, "an empty fold previews as 0 x 0");
    for (k, frame) in frames.iter().enumerate() {
        fold.fold(frame);
        let expected = build_and_fuse_pyramids(&frames[..=k], 4, false, true).reconstruct();
        assert_same_image(
            &fold.preview(),
            &expected,
            &format!("preview after {}", k + 1),
        );
    }
    // Previewing does not disturb the accumulator.
    let batch = build_and_fuse_pyramids(&frames, 4, false, true).reconstruct();
    assert_same_image(&fold.finish(), &batch, "finish after previews");
}

#[test]
fn streaming_driver_over_a_frame_source() {
    let frames = stack(13, 5, 21, 18, false);
    let expected = build_and_fuse_pyramids(&frames, 5, true, true).reconstruct();

    let from_slice = fuse_streaming(&frames[..], 5, true, true, |_, _| {});
    assert_same_image(&from_slice, &expected, "slice source");

    let mut seen = Vec::new();
    let source = FnFrames::new(frames.len(), |i| frames[i].clone());
    let from_fn = fuse_streaming(&source, 5, true, true, |done, fold| {
        seen.push(done);
        assert_eq!(fold.frames_folded(), done);
    });
    assert_same_image(&from_fn, &expected, "closure source");
    assert_eq!(seen, [1, 2, 3, 4, 5]);
}

#[test]
fn levels_below_the_minimum_are_clamped() {
    let frames = stack(17, 3, 12, 12, false);
    let batch = build_and_fuse_pyramids(&frames, 0, false, true);
    assert_eq!(batch.levels.len(), MIN_PYRAMID_LEVELS);
    let folded = fold_all(&frames, 0, false, true).finish_pyramid();
    assert_same_pyramid(&folded, &batch, "clamped depth");
}

#[test]
#[should_panic(expected = "same size")]
fn folding_a_frame_of_another_size_panics() {
    let mut fold = ApexFold::new(3, false, false);
    fold.fold(&stack(1, 1, 8, 8, false)[0]);
    fold.fold(&stack(2, 1, 16, 16, false)[0]);
}

#[test]
#[should_panic(expected = "empty")]
fn finishing_an_empty_fold_panics() {
    let _ = ApexFold::new(3, false, false).finish();
}

#[test]
fn memory_estimate_is_flat_in_the_frame_count_and_below_the_batch() {
    assert_eq!(apex_fold_bytes_per_pixel(8), 84);
    assert_eq!(apex_fold_bytes_per_pixel(0), apex_fold_bytes_per_pixel(2));
    let mut previous = 0;
    for levels in 2..=12 {
        let bytes = apex_fold_bytes_per_pixel(levels);
        assert!(bytes >= previous, "levels {levels}");
        previous = bytes;
        // The batch estimate for two frames is already higher.
        assert!(bytes <= 28 * 2 + 66 + 40, "levels {levels}: {bytes}");
    }
    assert!(apex_fold_bytes_per_pixel(8) < 28 * 5 + 66);
}
