use stacker_core::{image::PlanarImage, memory::*};
use std::collections::HashMap;

#[test]
fn test_mmap_tile_manager_round_trip() {
    let temp_dir = std::env::temp_dir().join(format!("stacker_test_mmap_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let mut manager = MmapTileManager {
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
    let manager = MmapTileManager {
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
    let w = 8;
    let h = 8;
    let mut img = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            img.luma[i] = f32::from(u16::try_from(i).expect("an 8x8 image has 64 pixels"));
        }
    }

    // Crop a 4×3 tile starting at (2, 1).
    let tile = extract_tile(&img, 2, 1, 4, 3);
    assert_eq!(tile.width, 4);
    assert_eq!(tile.height, 3);
    // Pixel (0,0) of tile == pixel (2,1) of source.
    assert_eq!(tile.luma[0].to_bits(), img.luma[w + 2].to_bits());
    // Pixel (3,2) of tile == pixel (5,3) of source.
    assert_eq!(
        tile.luma[2 * 4 + 3].to_bits(),
        img.luma[3 * w + 5].to_bits()
    );
}

/// The row-copy `extract_tile` equals the per-pixel clamped gather it replaced,
/// for crops inside the image, overhanging the right and bottom edges, and
/// starting beyond them (where every sample is the replicated corner).
#[test]
fn test_extract_tile_matches_clamped_reference() {
    let (w, h) = (7_usize, 5_usize);
    let luma: Vec<f32> = (0..35_u16).map(f32::from).collect();
    let chroma_a = luma.iter().map(|v| v * 2.0).collect();
    let chroma_b = luma.iter().map(|v| -v).collect();
    let img = PlanarImage::from_planes(w, h, luma, chroma_a, chroma_b).unwrap();
    let crops = [
        (0, 0, 7, 5),
        (2, 1, 3, 2),
        (5, 3, 6, 4),
        (6, 4, 3, 3),
        (7, 0, 4, 2),
        (20, 20, 3, 3),
        (3, 0, 0, 4),
    ];
    for (cx, cy, cw, ch) in crops {
        let tile = extract_tile(&img, cx, cy, cw, ch);
        assert_eq!((tile.width, tile.height), (cw, ch));
        assert!(tile.is_consistent());
        for oy in 0..ch {
            for ox in 0..cw {
                let sx = (cx + ox).min(w - 1);
                let sy = (cy + oy).min(h - 1);
                let (si, di) = (sy * w + sx, oy * cw + ox);
                assert_eq!(tile.luma[di].to_bits(), img.luma[si].to_bits());
                assert_eq!(tile.chroma_a[di].to_bits(), img.chroma_a[si].to_bits());
                assert_eq!(tile.chroma_b[di].to_bits(), img.chroma_b[si].to_bits());
            }
        }
    }
}

/// `paste_interior` must write into the correct output region.
#[test]
fn test_paste_interior_correctness() {
    let mut output = PlanarImage::new(16, 16);
    // Padded tile: 8×8, all luma = 7.0.
    let padded =
        PlanarImage::from_planes(8, 8, vec![7.0; 64], vec![2.0; 64], vec![3.0; 64]).unwrap();
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

// ── CORE-2: the store validates what it is given ─────────────────────────────

fn scratch_store(tag: &str) -> (MmapTileManager, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("stacker_core_store_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    (MmapTileManager::new(dir.clone()), dir)
}

const fn coord(width: usize, height: usize) -> TileCoordinate {
    TileCoordinate {
        start_x: 0,
        start_y: 0,
        width,
        height,
    }
}

/// A tile whose size does not match its coordinate is rejected at commit, not
/// silently written and misread later.
#[test]
fn commit_rejects_a_tile_that_does_not_match_its_coordinate() {
    let (mut store, dir) = scratch_store("mismatch");
    let err = store
        .commit_tile(0, &coord(4, 4), PlanarImage::new(2, 2))
        .unwrap_err();
    assert!(err.to_string().contains("expected 16"), "{err}");
    assert!(
        store.is_empty(),
        "nothing may be recorded for a rejected tile"
    );

    // Planes of unequal length whose sum happens to match are rejected too.
    let skewed = PlanarImage {
        width: 2,
        height: 2,
        luma: vec![0.0; 6],
        chroma_a: vec![0.0; 4],
        chroma_b: vec![0.0; 2],
    };
    assert!(store.commit_tile(0, &coord(2, 2), skewed).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A missing tile surfaces as the typed error that names the coordinate.
#[test]
fn fetching_a_missing_tile_is_tile_not_found() {
    let (store, dir) = scratch_store("missing");
    let key = TileCoordinate {
        start_x: 7,
        start_y: 9,
        width: 2,
        height: 2,
    };
    match store.fetch_tile(3, &key) {
        Err(stacker_core::error::StackerError::TileNotFound(x, y)) => assert_eq!((x, y), (7, 9)),
        other => panic!("expected TileNotFound, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Committing a key twice replaces the tile; `clear` removes the files it wrote
/// and nothing else.
#[test]
fn recommit_replaces_and_clear_removes_only_its_own_files() {
    let (mut store, dir) = scratch_store("clear");
    let bystander = dir.join("not_a_tile.txt");
    std::fs::write(&bystander, b"keep me").unwrap();

    let mut first = PlanarImage::new(2, 2);
    first.luma[0] = 1.0;
    let mut second = PlanarImage::new(2, 2);
    second.luma[0] = 2.0;
    store.commit_tile(0, &coord(2, 2), first).unwrap();
    store.commit_tile(0, &coord(2, 2), second).unwrap();
    assert_eq!(store.len(), 1);
    assert!((store.fetch_tile(0, &coord(2, 2)).unwrap().luma[0] - 2.0).abs() < f32::EPSILON);

    store.clear().unwrap();
    assert!(store.is_empty());
    assert!(store.fetch_tile(0, &coord(2, 2)).is_err());
    assert!(bystander.exists(), "clear must not touch foreign files");
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The coordinate is the padded rectangle the caller stores: tiles keyed by
/// padded rectangles of different size coexist and round-trip.
#[test]
fn tiles_are_keyed_by_the_rectangle_the_caller_stores() {
    let (mut store, dir) = scratch_store("padded");
    let interior = TileCoordinate {
        start_x: 40,
        start_y: 40,
        width: 8,
        height: 8,
    };
    let (px, py, pw, ph) = interior.padded_region_with(100, 100, 6);
    assert_eq!((px, py, pw, ph), (34, 34, 20, 20));
    let padded = TileCoordinate {
        start_x: px,
        start_y: py,
        width: pw,
        height: ph,
    };
    let mut tile = PlanarImage::new(pw, ph);
    tile.luma[5] = 3.5;
    store.commit_tile(1, &padded, tile).unwrap();
    assert!((store.fetch_tile(1, &padded).unwrap().luma[5] - 3.5).abs() < f32::EPSILON);
    // The interior rectangle was never committed.
    assert!(store.fetch_tile(1, &interior).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// `padded_region_with` clamps at the image border and reports the interior
/// offset for any apron; the default apron is `APRON_PX`.
#[test]
fn padded_region_with_matches_the_default_for_apron_px() {
    let tile = TileCoordinate {
        start_x: 300,
        start_y: 10,
        width: 64,
        height: 64,
    };
    assert_eq!(
        tile.padded_region(1000, 500),
        tile.padded_region_with(1000, 500, APRON_PX)
    );
    assert_eq!(
        tile.interior_offset_in_padded(1000, 500),
        tile.interior_offset_in_padded_with(1000, 500, APRON_PX)
    );
    // Apron 4: every side grows by 4 (the tile is clear of the borders).
    assert_eq!(tile.padded_region_with(1000, 500, 4), (296, 6, 72, 72));
    assert_eq!(tile.interior_offset_in_padded_with(1000, 500, 4), (4, 4));
    assert_eq!(tile.padded_region_with(1000, 500, 0), (300, 10, 64, 64));
}
