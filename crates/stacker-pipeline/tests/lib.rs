#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]
use stacker_pipeline::{TileClip, clip_tile_to_crop};

#[test]
fn tile_fully_inside_crop_is_unclipped() {
    // Crop covers the whole image; a 100x100 tile at (200, 200) is
    // entirely inside it, so the clip should equal the tile's own
    // interior, offset only by the padded-buffer interior offset.
    let clip = clip_tile_to_crop(200, 200, 100, 100, (16, 16), 0, 0, 1000, 1000);
    assert_eq!(
        clip,
        Some(TileClip {
            pad_off: (16, 16),
            dest_off: (200, 200),
            width: 100,
            height: 100,
        })
    );
}

#[test]
fn tile_fully_outside_crop_is_none() {
    // Crop is a small rect in the top-left; a tile far away doesn't
    // intersect it at all.
    let clip = clip_tile_to_crop(500, 500, 100, 100, (16, 16), 0, 0, 100, 100);
    assert_eq!(clip, None);
}

#[test]
fn tile_partially_overlapping_crop_is_clipped_correctly() {
    // Crop is x in [50, 150), y in [50, 150) (100x100). Tile interior is
    // x in [0, 100), y in [0, 100) with a 16px apron, so its padded
    // interior offset is (16, 16). The overlap is x in [50, 100),
    // y in [50, 100) — 50x50.
    let clip = clip_tile_to_crop(0, 0, 100, 100, (16, 16), 50, 50, 100, 100);
    assert_eq!(
        clip,
        Some(TileClip {
            // Overlap starts 50px into the tile's interior on both axes,
            // so padded offset is 16 + 50 = 66 on both axes.
            pad_off: (66, 66),
            // Overlap starts at (50, 50) in image space; crop origin is
            // (50, 50), so dest offset is (0, 0).
            dest_off: (0, 0),
            width: 50,
            height: 50,
        })
    );
}

#[test]
fn tile_touching_crop_edge_is_none() {
    // Tile ends exactly where the crop begins — zero-width/height
    // overlap, must be treated as no intersection (not a degenerate
    // zero-size paste).
    let clip = clip_tile_to_crop(0, 0, 50, 50, (16, 16), 50, 50, 100, 100);
    assert_eq!(clip, None);
}

#[test]
fn crop_fully_inside_one_tile() {
    // A small crop rectangle entirely inside a single large tile.
    let clip = clip_tile_to_crop(0, 0, 512, 512, (0, 0), 100, 100, 50, 50);
    assert_eq!(
        clip,
        Some(TileClip {
            pad_off: (100, 100),
            dest_off: (0, 0),
            width: 50,
            height: 50,
        })
    );
}
