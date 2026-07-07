use stacker_core::{image::PlanarImage, memory::*};
use std::collections::HashMap;

#[test]
fn test_tile_manager_round_trip() {
    let temp_dir = std::env::temp_dir().join(format!("stacker_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let mut manager = TileManager {
        temp_dir: temp_dir.clone(),
        tiles: HashMap::new(),
    };

    let coord = TileCoordinate {
        start_x: 0,
        start_y: 0,
        width: 2,
        height: 2,
    };

    let mut img = PlanarImage::new(2, 2);
    img.luma[0] = 1.0;
    img.luma[1] = 2.0;
    img.chroma_a[0] = 3.0;
    img.chroma_b[3] = 4.5;

    manager.commit_tile(0, &coord, img.clone()).unwrap();

    let fetched = manager.fetch_tile(0, &coord).unwrap();

    // f32 round-trip through le_bytes is exact — allow direct comparison.
    assert!((fetched.luma[0] - 1.0).abs() < f32::EPSILON);
    assert!((fetched.luma[1] - 2.0).abs() < f32::EPSILON);
    assert!((fetched.chroma_a[0] - 3.0).abs() < f32::EPSILON);
    assert!((fetched.chroma_b[3] - 4.5).abs() < f32::EPSILON);
    // Unset values round-trip as exactly 0.0.
    assert!((fetched.luma[2]).abs() < f32::EPSILON);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_fetch_missing_tile_returns_err() {
    let manager = TileManager {
        temp_dir: std::env::temp_dir(),
        tiles: HashMap::new(),
    };
    let coord = TileCoordinate {
        start_x: 0,
        start_y: 0,
        width: 4,
        height: 4,
    };
    let result = manager.fetch_tile(99, &coord);
    assert!(
        result.is_err(),
        "fetching a non-existent tile must return Err"
    );
}

/// Verify `enumerate_tiles` covers the full image with correct sizes.
#[test]
fn test_enumerate_tiles_coverage() {
    let (img_w, img_h, tile_size) = (100, 70, 32);
    let tiles = enumerate_tiles(img_w, img_h, tile_size);

    // Every pixel must be covered by exactly one tile.
    let mut hits = vec![0u8; img_w * img_h];
    for t in &tiles {
        for y in t.start_y..(t.start_y + t.height) {
            for x in t.start_x..(t.start_x + t.width) {
                hits[y * img_w + x] += 1;
            }
        }
    }
    assert!(
        hits.iter().all(|&h| h == 1),
        "each pixel must be covered exactly once"
    );
}

/// Verify apron boundary math: interior tiles, edge tiles, and corner tile.
///
/// Coordinates are derived from `APRON_PX` so the test stays correct regardless
/// of the apron value: the interior tile is placed far enough from every edge to
/// receive the full apron on all sides.
#[test]
fn test_apron_boundary_math() {
    // Generous margins so the interior tile has room for a full apron on all
    // sides and the image comfortably contains a right-edge tile.
    let tile = 32usize;
    let margin = APRON_PX + 36; // interior start offset (> APRON_PX)
    let img_w = margin + tile + APRON_PX + 64; // room for right apron + spare
    let img_h = margin + tile + APRON_PX + 64;

    // Interior tile — start >= APRON_PX on both axes, end+APRON_PX <= img bound.
    let interior = TileCoordinate {
        start_x: margin,
        start_y: margin,
        width: tile,
        height: tile,
    };
    let (px, py, pw, ph) = interior.padded_region(img_w, img_h);
    assert_eq!(px, margin - APRON_PX, "interior: left apron");
    assert_eq!(py, margin - APRON_PX, "interior: top apron");
    assert_eq!(
        pw,
        tile + 2 * APRON_PX,
        "interior: full width with both aprons"
    );
    assert_eq!(
        ph,
        tile + 2 * APRON_PX,
        "interior: full height with both aprons"
    );

    let (ix, iy) = interior.interior_offset_in_padded(img_w, img_h);
    assert_eq!(
        ix, APRON_PX,
        "interior: x offset in padded tile equals APRON_PX"
    );
    assert_eq!(
        iy, APRON_PX,
        "interior: y offset in padded tile equals APRON_PX"
    );

    // Top-left corner tile — both aprons clamp at (0, 0).
    let corner = TileCoordinate {
        start_x: 0,
        start_y: 0,
        width: 32,
        height: 32,
    };
    let (px2, py2, pw2, ph2) = corner.padded_region(img_w, img_h);
    assert_eq!(px2, 0, "corner: x clamped to 0");
    assert_eq!(py2, 0, "corner: y clamped to 0");
    assert_eq!(pw2, 32 + APRON_PX, "corner: only right apron (left clamps)");
    assert_eq!(ph2, 32 + APRON_PX, "corner: only bottom apron (top clamps)");

    let (ix2, iy2) = corner.interior_offset_in_padded(img_w, img_h);
    assert_eq!(ix2, 0, "corner: interior starts at x=0 in padded tile");
    assert_eq!(iy2, 0, "corner: interior starts at y=0 in padded tile");

    // Right-edge tile — its right edge sits exactly at `img_w`, so the right
    // apron clamps to the image boundary while the left apron is present.
    let re_w = 30usize;
    let right_edge = TileCoordinate {
        start_x: img_w - re_w,
        start_y: margin,
        width: re_w,
        height: tile,
    };
    let (px3, _py3, pw3, _ph3) = right_edge.padded_region(img_w, img_h);
    assert_eq!(
        px3,
        (img_w - re_w) - APRON_PX,
        "right-edge: left apron present"
    );
    assert_eq!(
        pw3,
        img_w - px3,
        "right-edge: width clamped to image right boundary"
    );

    // Exhaustive: no padded tile can exceed image bounds.
    for t in &enumerate_tiles(img_w, img_h, 32) {
        let (bx, by, bw, bh) = t.padded_region(img_w, img_h);
        assert!(bx + bw <= img_w, "padded tile x overflows image at {t:?}");
        assert!(by + bh <= img_h, "padded tile y overflows image at {t:?}");
    }
}

/// `extract_tile` must faithfully copy the requested region from the source.
#[test]
fn test_extract_tile_values() {
    let img_width = 8;
    let img_height = 8;
    let mut img = PlanarImage::new(img_width, img_height);
    for row in 0..img_height {
        for col in 0..img_width {
            let i = row * img_width + col;

            {
                img.luma[i] = (row * img_width + col) as f32;
            }
        }
    }

    // Crop a 4×3 tile starting at (2, 1).
    let tile = extract_tile(&img, 2, 1, 4, 3);
    assert_eq!(tile.width, 4);
    assert_eq!(tile.height, 3);
    // Pixel (0,0) of tile == pixel (2,1) of source.
    assert!((tile.luma[0] - img.luma[img_width + 2]).abs() < f32::EPSILON);
    // Pixel (3,2) of tile == pixel (5,3) of source.
    assert!((tile.luma[2 * 4 + 3] - img.luma[3 * img_width + 5]).abs() < f32::EPSILON);
}

/// `paste_interior` must write into the correct output region.
#[test]
fn test_paste_interior_correctness() {
    let mut output = PlanarImage::new(16, 16);
    // Padded tile: 8×8, all luma = 7.0.
    let padded = PlanarImage {
        width: 8,
        height: 8,
        luma: vec![7.0; 64],
        chroma_a: vec![2.0; 64],
        chroma_b: vec![3.0; 64],
    };
    // Paste interior (4×4) starting at offset (2,2) in padded, into (4,4) of output.
    paste_interior(&mut output, &padded, 2, 2, 4, 4, 4, 4);

    for oy in 0..4 {
        for ox in 0..4 {
            let i = (4 + oy) * 16 + (4 + ox);
            assert!((output.luma[i] - 7.0).abs() < f32::EPSILON);
            assert!((output.chroma_a[i] - 2.0).abs() < f32::EPSILON);
            assert!((output.chroma_b[i] - 3.0).abs() < f32::EPSILON);
        }
    }
    // Outside the pasted region must still be zero.
    assert!((output.luma[0]).abs() < f32::EPSILON);
}
