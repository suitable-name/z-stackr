use stacker_algo::apex::{fuse::*, pyramid::LaplacianPyramid};
use stacker_core::image::PlanarImage;

fn create_test_image(w: usize, h: usize, val: f32) -> PlanarImage<f32> {
    PlanarImage::with_luma(w, h, vec![val; w * h]).expect("one sample per pixel")
}

#[test]
fn test_fuse_pyramids_basic() {
    let img1 = create_test_image(8, 8, 1.0);
    let img2 = create_test_image(8, 8, 2.0);

    let pyr1 = LaplacianPyramid::build(&img1, 3);
    let pyr2 = LaplacianPyramid::build(&img2, 3);

    let fused = fuse_pyramids(&[pyr1, pyr2], false, false);
    assert_eq!(fused.levels.len(), 3);
    assert_eq!(fused.levels[0].width, 8);
    assert_eq!(fused.levels[0].height, 8);
}

/// FUS-3: a requested depth of 0 or 1 used to give a single, averaged base
/// level, i.e. the per-pixel mean of the frames.  It is now clamped to
/// `MIN_PYRAMID_LEVELS` and picks the sharp detail.
#[test]
fn test_levels_below_two_are_clamped_not_a_mean() {
    let (w, h) = (16, 16);
    let mut sharp = create_test_image(w, h, 0.0);
    for (i, v) in sharp.luma.iter_mut().enumerate() {
        *v = if (i % w + i / w) % 2 == 0 { 1.0 } else { 0.0 };
    }
    let blurred = create_test_image(w, h, 0.5);

    let mean: Vec<f32> = sharp
        .luma
        .iter()
        .zip(&blurred.luma)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();

    for requested in [0, 1] {
        let fused =
            build_and_fuse_pyramids(&[sharp.clone(), blurred.clone()], requested, false, true);
        assert_eq!(fused.levels.len(), MIN_PYRAMID_LEVELS);

        let out = fused.reconstruct();
        let worst = out
            .luma
            .iter()
            .zip(&mean)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            worst > 0.1,
            "levels = {requested}: result is still the plain mean (max deviation {worst})"
        );
    }
}

/// The constant-image tests above only check shapes (every Laplacian band of a
/// constant image is zero). With a real texture the selection itself is checked:
/// the finest detail band of the fused pyramid is, bit for bit, the band of the
/// source that has detail, wherever it is the sharper one, whatever the input
/// order and with or without the 3x3 grit-suppression neighbourhood; and the base
/// band is the average of the two bases.
#[test]
fn test_fused_detail_band_comes_from_the_sharper_source() {
    let (w, h) = (16, 16);
    let mut sharp = create_test_image(w, h, 0.0);
    for (i, v) in sharp.luma.iter_mut().enumerate() {
        *v = if (i % w + i / w) % 2 == 0 { 1.0 } else { 0.0 };
    }
    let flat = create_test_image(w, h, 0.5);
    let pyr_sharp = LaplacianPyramid::build(&sharp, 3);
    let pyr_flat = LaplacianPyramid::build(&flat, 3);
    assert!(
        pyr_sharp.levels[0].luma.iter().all(|v| v.abs() > 0.1),
        "the checkerboard must have strong detail at every pixel of the finest band"
    );

    for grit_suppression in [false, true] {
        for flat_first in [true, false] {
            let pyramids = if flat_first {
                [pyr_flat.clone(), pyr_sharp.clone()]
            } else {
                [pyr_sharp.clone(), pyr_flat.clone()]
            };
            let fused = fuse_pyramids(&pyramids, false, grit_suppression);
            let context = format!("grit={grit_suppression} flat_first={flat_first}");

            for (i, (got, want)) in fused.levels[0]
                .luma
                .iter()
                .zip(&pyr_sharp.levels[0].luma)
                .enumerate()
            {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{context}: finest band [{i}]"
                );
            }

            // Base band: the plain average of the two bases (sum in input order).
            let base = fused.levels.len() - 1;
            for i in 0..fused.levels[base].luma.len() {
                let want = f32::midpoint(
                    pyramids[0].levels[base].luma[i],
                    pyramids[1].levels[base].luma[i],
                );
                assert_eq!(
                    fused.levels[base].luma[i].to_bits(),
                    want.to_bits(),
                    "{context}: base band [{i}]"
                );
            }
        }
    }
}

/// FUS-6: the documented tie rule is "the first source in input order wins",
/// the same as `Relief` and the auto-cull.  Two frames with the same luma have
/// exactly equal selection energy at every pixel (the chroma of one is the
/// negation of the other, which squares to the same value), so every detail
/// level, chroma included, must come from the first frame, whichever it is.
#[test]
fn test_ties_go_to_the_first_frame() {
    let (w, h) = (16, 16);
    let checker = |scale: f32| -> PlanarImage<f32> {
        let pattern: Vec<f32> = (0..w * h)
            .map(|i| if (i % w + i / w) % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let chroma_a: Vec<f32> = pattern.iter().map(|p| scale * 0.2 * p).collect();
        let chroma_b: Vec<f32> = pattern.iter().map(|p| scale * 0.1 * p).collect();
        PlanarImage::from_planes(w, h, pattern, chroma_a, chroma_b).expect("planes match size")
    };
    let pyr_pos = LaplacianPyramid::build(&checker(1.0), 3);
    let pyr_neg = LaplacianPyramid::build(&checker(-1.0), 3);
    assert!(
        pyr_pos.levels[0]
            .chroma_a
            .iter()
            .zip(&pyr_neg.levels[0].chroma_a)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the two sources must differ in chroma for the test to discriminate"
    );

    for use_color in [false, true] {
        for grit_suppression in [false, true] {
            for first_is_pos in [true, false] {
                let (first, second) = if first_is_pos {
                    (&pyr_pos, &pyr_neg)
                } else {
                    (&pyr_neg, &pyr_pos)
                };
                let fused = fuse_pyramids(
                    &[first.clone(), second.clone()],
                    use_color,
                    grit_suppression,
                );
                let context = format!(
                    "color={use_color} grit={grit_suppression} first_is_pos={first_is_pos}"
                );

                // Every detail level (all but the averaged base).
                for level in 0..fused.levels.len() - 1 {
                    let got = &fused.levels[level];
                    let want = &first.levels[level];
                    for i in 0..got.luma.len() {
                        assert_eq!(
                            got.chroma_a[i].to_bits(),
                            want.chroma_a[i].to_bits(),
                            "{context}: level {level} chroma_a [{i}]"
                        );
                        assert_eq!(
                            got.chroma_b[i].to_bits(),
                            want.chroma_b[i].to_bits(),
                            "{context}: level {level} chroma_b [{i}]"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_build_and_fuse_pyramids() {
    let img1 = create_test_image(8, 8, 1.0);
    let img2 = create_test_image(8, 8, 2.0);

    let fused = build_and_fuse_pyramids(&[img1, img2], 3, false, true);
    assert_eq!(fused.levels.len(), 3);
    assert_eq!(fused.levels[0].width, 8);
    assert_eq!(fused.levels[0].height, 8);
}
