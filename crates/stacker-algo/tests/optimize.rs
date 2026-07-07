use stacker_algo::optimize::*;
use stacker_core::image::PlanarImage;

/// Build a flat (near-zero SML) `PlanarImage` of given dimensions.
fn flat_frame(width: usize, height: usize) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(width, height);
    // Fill with a tiny constant so it is not zero but well below any
    // contrast floor.  Value chosen to be < CONTRAST_FLOOR_ABSOLUTE even
    // after the SML convolution; the SML of a flat field is 0.
    for v in &mut img.luma {
        *v = 0.001;
    }
    img
}

/// Build a frame with a high-contrast blob in a rectangle [bx..ex, by..ey].
/// Pixels inside the blob alternate between 0 and 1 so the SML kernel
/// measures a strong response.  Outside pixels are set to 0.001 (flat).
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
                // Checkerboard pattern — maximum gradient for the SML kernel.
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
/// - Frame 2: completely flat (near-zero SML everywhere).
/// - Frame 3: completely flat (near-zero SML everywhere).
///
/// Expected outcome:
/// - Frames 0 and 1 are KEPT (they win a significant fraction of detail
///   pixels each).
/// - Frames 2 and 3 are DROPPED (they never win any detail pixel).
///
/// Under the old all-pixels metric, frame 0 would accumulate *all* the
/// background/tie-break wins (initialised `best_idx` = 0, `max_val` = -1.0),
/// giving it an inflated win count that swamps the threshold even for
/// genuinely flat frames — this test would FAIL under that metric because
/// flat frames 2 and 3 would also appear to contribute via the same tie
/// (their SML = 0 > -1.0 for all background pixels not yet won by frame 0).
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
    // No panic is the primary assertion.  The kept set is secondary —
    // when there are zero detail pixels every frame's win count is 0,
    // and 0 >= min_detail_pixels (which is also 0) so all frames are kept.
    assert!(result.kept_indices.len() <= 4);
    assert!(result.recommended_order.len() <= 4);
}

/// Single-frame stack: trivially kept, no nearest-neighbour ordering needed.
#[test]
fn test_single_frame_kept() {
    let frame = detail_frame(16, 16, 4, 12, 4, 12);
    let result = optimize_stack(&[frame], 2.0);
    assert_eq!(result.kept_indices, vec![0]);
    assert_eq!(result.recommended_order, vec![0]);
}
