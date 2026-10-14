#![cfg(feature = "akaze")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the test images are a few hundred pixels wide and the 8-bit buffers are quantised from clamped values"
)]

use stacker_align::akaze_match::*;
use stacker_core::image::PlanarImage;

/// A deterministic textured test image: a flat background plus sixty Gaussian
/// blobs of different size and brightness at pseudo-random positions. Blobs are
/// exactly what AKAZE's determinant-of-Hessian detector responds to, and the
/// image is far above the 64-px size gate of `extract_ref_features`.
fn blob_image(w: usize, h: usize, seed: u64) -> PlanarImage<f32> {
    let mut state = seed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f32 / (1_u64 << 31) as f32
    };
    let mut img = PlanarImage::new(w, h);
    img.luma.fill(0.2);
    for _ in 0..60 {
        let (cx, cy) = (next() * w as f32, next() * h as f32);
        let sigma = next().mul_add(3.0, 2.0);
        let amp = next().mul_add(0.5, 0.3);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                let bump = amp * (-dx.mul_add(dx, dy * dy) / (2.0 * sigma * sigma)).exp();
                img.luma[y * w + x] += bump;
            }
        }
    }
    for v in &mut img.luma {
        *v = v.clamp(0.0, 1.0);
    }
    img
}

#[test]
fn test_luma_to_gray8_gamma_and_stretch() {
    // A 4×2 image covering negative, zero, mid-range, and above-1 values.
    let mut img = PlanarImage::new(4, 2);
    let inputs = [-0.5_f32, 0.0, 0.5, 1.0, 1.5, 0.25, 0.75, 0.1];
    img.luma.copy_from_slice(&inputs);

    let gfi = luma_to_gray8(&img);
    assert_eq!(gfi.width(), 4);
    assert_eq!(gfi.height(), 2);
    let out: Vec<u8> = gfi.as_raw().clone();

    // The percentile stretch must reach both ends of the 8-bit range (a `u8`
    // is always within 0..=255, so range membership proves nothing).
    assert_eq!(*out.iter().min().unwrap(), 0, "black end of the stretch");
    assert_eq!(*out.iter().max().unwrap(), 255, "white end of the stretch");
    // Negative and zero both clamp to black; 1.0 and 1.5 both clamp to white.
    assert_eq!(out[0], out[1], "-0.5 and 0.0 clamp to the same code");
    assert_eq!(out[0], 0);
    assert_eq!(out[3], out[4], "1.0 and 1.5 clamp to the same code");
    assert_eq!(out[3], 255);
    // Gamma encoding and stretching are monotone: sorting by input sorts the codes.
    let mut order: Vec<usize> = (0..8).collect();
    order.sort_by(|&a, &b| inputs[a].total_cmp(&inputs[b]));
    for pair in order.windows(2) {
        assert!(
            out[pair[0]] <= out[pair[1]],
            "input {} -> {} must not exceed input {} -> {}",
            inputs[pair[0]],
            out[pair[0]],
            inputs[pair[1]],
            out[pair[1]]
        );
    }
}

/// Core diagnostic test: prove that gamma+normalise restores detectability.
#[test]
fn test_gamma_normalize_restores_contrast_on_low_linear_amplitude() {
    const W: usize = 64;
    const H: usize = 64;
    const LOW_LIN_BASE: f32 = 0.05;
    const LOW_LIN_RANGE: f32 = 0.20; // [0.05, 0.25]

    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let norm = ((x as f32 * 0.628).sin() * (y as f32 * 0.471).cos()).mul_add(0.5, 0.5);
            img.luma[y * W + x] = norm.mul_add(LOW_LIN_RANGE, LOW_LIN_BASE);
        }
    }

    // ── Old path: raw linear clamp simulation ──
    let mut old_gfi = Gray8Image::new(W as u32, H as u32);
    for (dst, &src) in old_gfi.pixels_mut().zip(img.luma.iter()) {
        dst.0[0] = (src.clamp(0.0, 1.0) * 255.0) as u8;
    }
    let old_min = old_gfi.pixels().map(|p| p.0[0]).min().unwrap();
    let old_max = old_gfi.pixels().map(|p| p.0[0]).max().unwrap();
    let old_range = f32::from(old_max - old_min);

    // ── New path: gamma + 1st–99th percentile stretch ──
    let new_gfi = luma_to_gray8(&img);
    let new_min = new_gfi.pixels().map(|p| p.0[0]).min().unwrap();
    let new_max = new_gfi.pixels().map(|p| p.0[0]).max().unwrap();
    let new_range = f32::from(new_max - new_min);

    assert!(
        old_range > 1.0,
        "old buffer range ({old_range}) must be non-zero"
    );
    let ratio = new_range / old_range;
    assert!(
        ratio >= 3.0,
        "gamma+normalize must increase contrast by ≥ 3×; ratio={ratio:.2}"
    );
}

/// An image below the 64-px gate never reaches the detector: no features, no
/// matches, and no error. (The detector itself is exercised by the textured-image
/// tests below.)
#[test]
fn test_images_below_the_size_gate_produce_no_features_or_matches() {
    let img1 = PlanarImage::new(32, 32);
    let img2 = PlanarImage::new(32, 32);
    let (kps, desc) = extract_ref_features(&img1);
    assert!(
        kps.is_empty() && desc.is_empty(),
        "below the 64-px gate the detector must not run"
    );
    let mr = KeypointMatcher::extract_and_match(&img1, &img2).expect("small images are fine");
    assert!(mr.matches.is_empty(), "no features, so no matches");
}

/// ALN-6: the ratio test needs a second candidate.  With one target descriptor
/// there is no second-best distance, and the old `<=` test against `u32::MAX`
/// accepted every query; with two identical target descriptors the match is
/// ambiguous (`0 <= 0` used to accept it).
#[test]
fn test_ratio_test_rejects_missing_or_tied_second_best() {
    let query = Descriptor::new([0b0101_0101_u8; 64]);
    let far = Descriptor::new([0b1010_1010_u8; 64]);

    let single = match_features(&[query], &[far], 0.86);
    assert!(
        single.is_empty(),
        "one target descriptor has no second-best: {single:?}"
    );

    let tied = match_features(&[query], &[query, query], 0.86);
    assert!(
        tied.is_empty(),
        "two identical nearest descriptors are ambiguous: {tied:?}"
    );

    // A clearly distinctive nearest neighbour is still accepted.
    let mut near = [0b0101_0101_u8; 64];
    near[0] = 0b0101_0100;
    let kept = match_features(&[query], &[Descriptor::new(near), far], 0.86);
    assert_eq!(
        kept,
        vec![Match {
            index_0: 0,
            index_1: 0
        }]
    );
}

/// On a textured 128-px image the detector finds features, and matching the image
/// with itself pairs each keypoint with the keypoint at the same position.
#[test]
fn test_textured_image_self_match_pairs_identical_positions() {
    let img = blob_image(128, 128, 0x5eed);
    let (kps, desc) = extract_ref_features(&img);
    assert!(
        kps.len() >= 4,
        "the detector must find blobs in a textured image, found {}",
        kps.len()
    );
    assert_eq!(kps.len(), desc.len(), "one descriptor per keypoint");

    // A descriptor is at distance 0 from itself, so the strict ratio test
    // (0 < r * d2) accepts every keypoint whose second-nearest descriptor differs.
    let matches = match_features(&desc, &desc, 0.86);
    assert_eq!(
        matches.len(),
        kps.len(),
        "self-matching must keep every keypoint"
    );
    let same_position = matches
        .iter()
        .filter(|m| kps[m.index_0].point == kps[m.index_1].point)
        .count();
    assert!(
        same_position * 10 >= matches.len() * 9,
        "{same_position} of {} self-matches pair a keypoint with itself",
        matches.len()
    );
}

/// `match_target` against pre-extracted reference features must produce the very
/// same match list as extracting both images afresh, on an image that actually
/// has features (so the equality is not `0 == 0`).
#[test]
fn test_match_target_agrees_with_extract_and_match() {
    let img1 = blob_image(128, 128, 0x5eed);
    let img2 = blob_image(128, 128, 0x5eed);

    let (ref_kps, ref_desc) = extract_ref_features(&img1);
    assert!(!ref_kps.is_empty(), "the reference must have features");
    let via_match_target = KeypointMatcher::match_target(&ref_kps, &ref_desc, &img2)
        .expect("match_target must succeed");
    let via_extract_match =
        KeypointMatcher::extract_and_match(&img1, &img2).expect("extract_and_match must succeed");

    assert!(
        !via_extract_match.matches.is_empty(),
        "identical textured images must match"
    );
    assert_eq!(
        via_match_target.matches, via_extract_match.matches,
        "match_target and extract_and_match must agree on every match"
    );
    assert_eq!(via_match_target.kps0.len(), ref_kps.len());
    assert_eq!(via_match_target.kps1.len(), via_extract_match.kps1.len());
}
