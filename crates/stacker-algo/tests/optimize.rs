use stacker_algo::optimize::*;
use stacker_core::image::PlanarImage;

/// Build a flat (near-zero focus score) `PlanarImage` of given dimensions.
fn flat_frame(width: usize, height: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    // Fill with a tiny constant so it is not zero but well below any
    // contrast floor.  Value chosen to be < CONTRAST_FLOOR_ABSOLUTE even
    // after the focus measure; the plane-fit variance of a flat field is 0.
    img.luma.fill(0.001);
    img
}

/// Build a frame with a high-contrast blob in a rectangle [bx..ex, by..ey].
/// Pixels inside the blob alternate between 0 and 1 so the focus measure
/// gives a strong response.  Outside pixels are set to 0.001 (flat).
fn detail_frame(
    width: usize,
    height: usize,
    bx: usize,
    ex: usize,
    by: usize,
    ey: usize,
) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if x >= bx && x < ex && y >= by && y < ey {
                // Checkerboard pattern — maximum local variance for the focus measure.
                img.luma[idx] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            } else {
                img.luma[idx] = 0.001;
            }
        }
    }
    img
}

/// Synthetic stack:
///
/// - Frame 0: high-contrast blob in the left half.
/// - Frame 1: high-contrast blob in the right half.
/// - Frame 2: completely flat (near-zero focus score everywhere).
/// - Frame 3: completely flat (near-zero focus score everywhere).
///
/// Expected outcome:
/// - Frames 0 and 1 are KEPT (they win a significant fraction of detail
///   pixels each).
/// - Frames 2 and 3 are DROPPED (they never win any detail pixel).
///
/// Only pixels above the contrast floor count. A metric over all pixels would
/// let frame 0 accumulate *all* the background/tie-break wins (the running best
/// starts at `best_idx` = 0, `max_val` = -1.0), inflating its win count past the
/// threshold even for genuinely flat frames, and flat frames 2 and 3 would also
/// appear to contribute (their score = 0 > -1.0 for background pixels not yet
/// won by frame 0). Excluding flat pixels from the census is what lets this
/// test pass.
#[test]
fn test_flat_frames_are_culled() {
    let w = 64_usize;
    let h = 64_usize;

    // Frame 0 has a detail blob on the left half.
    let frame0 = detail_frame(w, h, 0, w / 2, 0, h);
    // Frame 1 has a detail blob on the right half.
    let frame1 = detail_frame(w, h, w / 2, w, 0, h);
    // Frames 2 and 3 are completely flat.
    let frame2 = flat_frame(w, h);
    let frame3 = flat_frame(w, h);

    let images = vec![frame0, frame1, frame2, frame3];

    // Use a low threshold (0.5 %) so that each of the two detail frames
    // passes easily, but flat frames (0 detail wins) cannot.
    let result = optimize_stack(&images, 0.5);

    // Detail frames must be kept.
    assert!(
        result.kept_indices.contains(&0),
        "frame 0 (left-detail) must be kept; kept={:?}",
        result.kept_indices
    );
    assert!(
        result.kept_indices.contains(&1),
        "frame 1 (right-detail) must be kept; kept={:?}",
        result.kept_indices
    );

    // Flat frames must be dropped.
    assert!(
        !result.kept_indices.contains(&2),
        "frame 2 (flat) must be dropped; kept={:?}",
        result.kept_indices
    );
    assert!(
        !result.kept_indices.contains(&3),
        "frame 3 (flat) must be dropped; kept={:?}",
        result.kept_indices
    );

    // Recommended order must contain exactly the two kept frames.
    assert_eq!(
        result.recommended_order.len(),
        2,
        "recommended order should have 2 frames; got {:?}",
        result.recommended_order
    );
}

/// When ALL frames are blank (nothing to distinguish), we still get a
/// sane result: either all kept or all dropped, but no panic.
#[test]
fn test_all_flat_no_panic() {
    let images: Vec<PlanarImage<f32>> = (0..4).map(|_| flat_frame(16, 16)).collect();
    let result = optimize_stack(&images, 0.5);
    // With zero detail pixels every frame's win count is 0 and
    // 0 >= min_detail_pixels (also 0), so exactly all four frames are kept, and
    // the recommended order is a permutation of them.
    assert_eq!(result.kept_indices, vec![0, 1, 2, 3]);
    let mut order = result.recommended_order;
    order.sort_unstable();
    assert_eq!(order, vec![0, 1, 2, 3]);
}

/// `frame_count` frames of `band * textured.len()` columns and `height` rows.
/// The frame `textured[k]` carries the checkerboard in the `k`-th band of
/// `band` columns; every other frame is flat.
fn banded_stack(
    frame_count: usize,
    textured: &[usize],
    band: usize,
    height: usize,
) -> Vec<PlanarImage<f32>> {
    let width = band * textured.len();
    (0..frame_count)
        .map(|f| {
            textured.iter().position(|&t| t == f).map_or_else(
                || flat_frame(width, height),
                |k| detail_frame(width, height, k * band, (k + 1) * band, 0, height),
            )
        })
        .collect()
}

#[test]
fn test_cull_threshold_formula() {
    assert!((cull_threshold(0) - 2.0).abs() < 1e-6);
    assert!((cull_threshold(1) - 2.0).abs() < 1e-6);
    assert!((cull_threshold(25) - 2.0).abs() < 1e-6);
    assert!((cull_threshold(50) - 1.0).abs() < 1e-6);
    assert!((cull_threshold(100) - 0.5).abs() < 1e-6);
    // Never above the ceiling, never increasing with the frame count.
    for n in 1..300 {
        assert!(cull_threshold(n) <= MAX_CULL_THRESHOLD_PCT);
        assert!(cull_threshold(n + 1) <= cull_threshold(n));
    }
}

/// FUS-1: 60 frames, each sharp in its own 1/60 vertical band.  With the
/// absolute 2 % rule every frame won 1.67 % and all of them were dropped.
#[test]
fn test_deep_uniform_stack_keeps_every_frame() {
    let textured: Vec<usize> = (0..60).collect();
    let images = banded_stack(60, &textured, 8, 16);
    let result = optimize_stack(&images, 2.0);
    assert_eq!(
        result.kept_indices,
        textured,
        "all 60 frames must survive; got {} kept",
        result.kept_indices.len()
    );
    assert_eq!(result.recommended_order.len(), 60);
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
}

/// FUS-1: a small stack still drops exactly the blank frames.
#[test]
fn test_ten_frames_two_blank_drops_exactly_those() {
    let textured = [0, 1, 2, 4, 5, 6, 8, 9];
    let images = banded_stack(10, &textured, 10, 16);
    let result = optimize_stack(&images, 2.0);
    assert_eq!(result.kept_indices, textured.to_vec());
    assert!(!result.kept_indices.contains(&3));
    assert!(!result.kept_indices.contains(&7));
    let mut order = result.recommended_order;
    order.sort_unstable();
    assert_eq!(order, textured.to_vec());
}

/// FUS-1: an absurd caller threshold can no longer empty the list; the
/// frame-count ceiling applies.
#[test]
fn test_huge_threshold_never_empties_the_stack() {
    let textured: Vec<usize> = (0..12).collect();
    let images = banded_stack(12, &textured, 8, 16);
    let result = optimize_stack(&images, 99.0);
    assert_eq!(result.kept_indices.len(), 12);
}

/// A stack without any detail keeps everything and says so.
#[test]
fn test_no_detail_keeps_all_with_warning() {
    let images: Vec<PlanarImage<f32>> = (0..4).map(|_| flat_frame(16, 16)).collect();
    let result = optimize_stack(&images, 2.0);
    assert_eq!(result.kept_indices, vec![0, 1, 2, 3]);
    assert_eq!(result.warnings.len(), 1, "{:?}", result.warnings);
}

/// Single-frame stack: trivially kept, no nearest-neighbour ordering needed.
#[test]
fn test_single_frame_kept() {
    let frame = detail_frame(16, 16, 4, 12, 4, 12);
    let result = optimize_stack(&[frame], 2.0);
    assert_eq!(result.kept_indices, vec![0]);
    assert_eq!(result.recommended_order, vec![0]);
}
