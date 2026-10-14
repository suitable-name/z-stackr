//! `tiling` (apron and geometry from the settings) and `crop`.

use stacker_algo::{apex::fuse::apex_fold_bytes_per_pixel, strata::STRATA_APRON_PX};
use stacker_core::{
    image::PlanarImage,
    memory::{APRON_PX, MmapTileManager, TileProvider},
    settings::StackingSettings,
};
use stacker_pipeline::{
    crop::{CommonCoverage, common_rect},
    tiling::{
        Geometry, MULTIGRID_DEPTH_BYTES_PER_PIXEL, MULTIGRID_FINISH_BYTES_PER_PIXEL,
        MULTIGRID_VOTE_BYTES_PER_PIXEL, Method, apron_for, commit_frame,
        multigrid_whole_image_bytes, plan_tiles,
    },
};

fn settings_with(levels: u32, est: u32, smooth: u32) -> StackingSettings {
    StackingSettings {
        pyramid_levels: levels,
        relief_estimation_radius: est,
        relief_smoothing_radius: smooth,
        ..StackingSettings::default()
    }
}

/// The apron follows what is computed (CORE-3): the pyramid depth for `Apex`, the
/// two radii for `Relief`, the receptive field for the network.
#[test]
fn the_apron_is_derived_from_the_settings() {
    for (levels, apron) in [(2, 6), (4, 32), (6, 128), (7, 256), (8, 512)] {
        assert_eq!(
            apron_for(Method::Apex, &settings_with(levels, 5, 2)),
            apron,
            "{levels} levels"
        );
    }
    // The default settings: 8 levels -> 512, radii 5 and 2 -> 10.
    let defaults = StackingSettings::default();
    assert_eq!(apron_for(Method::Apex, &defaults), 512);
    assert_eq!(apron_for(Method::Relief, &defaults), 10);
    assert_eq!(apron_for(Method::Neural, &defaults), APRON_PX);
    // Never below 2: the census window must fit.
    assert_eq!(apron_for(Method::Relief, &settings_with(8, 0, 0)), 2);
    // The largest allowed radii.
    assert_eq!(apron_for(Method::Relief, &settings_with(8, 100, 50)), 201);
}

#[test]
fn apex_tile_sizes_stay_on_the_decimation_grid() {
    let s = settings_with(8, 5, 2);
    let g = Geometry::choose(512, Method::Apex, &s);
    assert_eq!((g.tile_size, g.apron), (512, 512));
    // 300 is rounded up to the next multiple of 2^7.
    assert_eq!(Geometry::choose(300, Method::Apex, &s).tile_size, 384);
    assert_eq!(Geometry::choose(64, Method::Apex, &s).tile_size, 128);
    // Other methods keep the requested size.
    assert_eq!(Geometry::choose(300, Method::Relief, &s).tile_size, 300);
    assert_eq!(Geometry::choose(300, Method::Neural, &s).tile_size, 300);
}

/// The overhead of the derived apron is stated, not hidden.
#[test]
fn the_geometry_reports_its_overhead() {
    let g = Geometry {
        tile_size: 512,
        apron: 512,
    };
    assert_eq!(g.padded_edge(), 1536);
    assert!((g.padding_factor() - 9.0).abs() < 1e-9);
    assert!((g.scratch_bytes_per_source_pixel() - 108.0).abs() < 1e-9);
    // One frame's padded tile while fusing: the tile (12 per padded pixel). The
    // Apex fold and the streamed Relief focus maps keep nothing per frame beyond
    // it (they were 28 and 16 when a pyramid or a focus map was kept per frame).
    assert_eq!(g.fusion_bytes_per_frame(Method::Apex), 12 * 1536 * 1536);
    assert_eq!(g.fusion_bytes_per_frame(Method::Relief), 12 * 1536 * 1536);
    assert_eq!(g.fusion_bytes_per_frame(Method::Neural), 12 * 1536 * 1536);
    // Their fixed working sets, per padded pixel: the fold (84 at 8 levels, taken
    // at the 12 levels the settings allow at most), the votes and the focus map
    // in flight.
    let fold = apex_fold_bytes_per_pixel(12);
    assert!(
        (apex_fold_bytes_per_pixel(8)..=100).contains(&fold),
        "{fold}"
    );
    assert_eq!(g.fusion_fixed_bytes(Method::Apex), fold * 1536 * 1536);
    assert_eq!(g.fusion_fixed_bytes(Method::Relief), 10 * 1536 * 1536);
    assert_eq!(g.fusion_fixed_bytes(Method::Neural), 0);
}

/// The read-ahead of the next tile holds the padded tile of every frame again,
/// except for `Strata`, which reads on demand.
#[test]
fn the_prefetch_holds_the_next_tiles_frames() {
    let g = Geometry {
        tile_size: 512,
        apron: 128,
    };
    let padded = g.padded_pixels();
    assert_eq!(g.prefetch_bytes(Method::Apex, 7), 7 * 12 * padded);
    assert_eq!(g.prefetch_bytes(Method::Relief, 7), 7 * 12 * padded);
    assert_eq!(g.prefetch_bytes(Method::Strata, 7), 0);
}

#[test]
fn the_grid_covers_the_image_and_pads_inside_it() {
    let (w, h) = (100, 100);
    let cells = plan_tiles(w, h, 32, 10);
    let mut hits = vec![0_u8; w * h];
    for cell in &cells {
        let i = &cell.interior;
        for y in i.start_y..i.start_y + i.height {
            for x in i.start_x..i.start_x + i.width {
                hits[y * w + x] += 1;
            }
        }
        let p = &cell.padded;
        assert!(p.start_x + p.width <= w && p.start_y + p.height <= h);
        // The interior sits at `interior_offset` inside the padded tile.
        assert_eq!(p.start_x + cell.interior_offset.0, i.start_x);
        assert_eq!(p.start_y + cell.interior_offset.1, i.start_y);
        assert!(cell.interior_offset.0 + i.width <= p.width);
    }
    assert!(
        hits.iter().all(|&n| n == 1),
        "every pixel in exactly one tile"
    );
    // An inner tile has the full apron on every side.
    let inner = cells
        .iter()
        .find(|c| c.interior.start_x == 32 && c.interior.start_y == 32)
        .unwrap();
    assert_eq!(inner.interior_offset, (10, 10));
    assert_eq!((inner.padded.width, inner.padded.height), (52, 52));
}

#[test]
fn committed_tiles_read_back_by_their_padded_key() {
    let dir = std::env::temp_dir().join(format!("stacker_pipeline_tiles_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut store = MmapTileManager::new(dir.clone());

    let (w, h) = (40, 30);
    let mut frame = PlanarImage::<f32>::new(w, h);
    for (i, v) in frame.luma.iter_mut().enumerate() {
        *v = f32::from(u16::try_from(i).unwrap());
    }
    let cells = plan_tiles(w, h, 16, 4);
    commit_frame(&mut store, 2, &frame, &cells).unwrap();
    assert_eq!(store.len(), cells.len());

    for cell in &cells {
        let tile = store.fetch_tile(2, &cell.padded).unwrap();
        let (ix, iy) = cell.interior_offset;
        // The first interior pixel is the source pixel at the interior origin.
        let want =
            f32::from(u16::try_from(cell.interior.start_y * w + cell.interior.start_x).unwrap());
        assert!((tile.luma[iy * tile.width + ix] - want).abs() < f32::EPSILON);
    }
    store.clear().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// `Strata`: the apron is the algorithm's own constant (101 px, below the shared
/// 128), whatever the settings, and its working set is a fixed per-pixel figure
/// besides the tiles.
#[test]
fn strata_has_its_own_apron_and_a_fixed_working_set() {
    let defaults = StackingSettings::default();
    assert_eq!(apron_for(Method::Strata, &defaults), STRATA_APRON_PX);
    assert_eq!(STRATA_APRON_PX, 101);
    const { assert!(STRATA_APRON_PX <= APRON_PX) };
    let wide = StackingSettings {
        strata_base_radius: 64,
        strata_detail_focus: 5,
        ..defaults.clone()
    };
    assert_eq!(apron_for(Method::Strata, &wide), STRATA_APRON_PX);

    // Tile sizes are kept as requested (no decimation grid).
    let g = Geometry::choose(300, Method::Strata, &defaults);
    assert_eq!((g.tile_size, g.apron), (300, 101));

    // 512-px tiles: the padded edge is 714.
    let g = Geometry::choose(512, Method::Strata, &defaults);
    assert_eq!(g.padded_edge(), 714);
    let padded = 714 * 714;
    // The tile frames are read on demand, two at a time, so nothing is held per
    // frame ...
    assert_eq!(g.fusion_bytes_per_frame(Method::Strata), 0);
    // ... and 150 bytes per padded pixel plus the two frames (24) whatever the
    // frame count.
    assert_eq!(g.fusion_fixed_bytes(Method::Strata), (150 + 24) * padded);
    assert_eq!(g.fusion_bytes(Method::Strata, 20), (150 + 24) * padded);
    assert_eq!(g.fusion_bytes(Method::Strata, 200), (150 + 24) * padded);
    // The neural mode has no fixed term.
    assert_eq!(g.fusion_fixed_bytes(Method::Neural), 0);
    assert_eq!(
        g.fusion_bytes(Method::Neural, 3),
        3 * g.fusion_bytes_per_frame(Method::Neural)
    );
    // The others are the tiles plus the fixed working set.
    for method in [Method::Apex, Method::Relief] {
        assert_eq!(
            g.fusion_bytes(method, 3),
            3 * g.fusion_bytes_per_frame(method) + g.fusion_fixed_bytes(method)
        );
    }
}

/// The whole-image planes of the multigrid `Relief` are in the estimate: 28 bytes
/// per pixel at the solve, plus the best-focus map that is still held.
#[test]
fn the_multigrid_adds_whole_image_bytes() {
    assert_eq!(MULTIGRID_VOTE_BYTES_PER_PIXEL, 6);
    assert_eq!(MULTIGRID_FINISH_BYTES_PER_PIXEL, 28);
    assert_eq!(MULTIGRID_DEPTH_BYTES_PER_PIXEL, 4);
    assert_eq!(multigrid_whole_image_bytes(1_000_000), 32_000_000);
    assert_eq!(multigrid_whole_image_bytes(0), 0);
}

// ── crop ──────────────────────────────────────────────────────────────────────

fn mask(w: usize, h: usize, valid: impl Fn(usize, usize) -> bool) -> Vec<bool> {
    (0..w * h).map(|i| valid(i % w, i / w)).collect()
}

#[test]
fn the_common_rectangle_is_the_intersection() {
    let a = mask(10, 8, |x, _| x >= 2);
    let b = mask(10, 8, |_, y| y < 6);
    let c = mask(10, 8, |x, _| x < 9);
    let rect = common_rect([a.as_slice(), b.as_slice(), c.as_slice()], 10, 8);
    assert_eq!(rect, Some((2, 0, 7, 6)));
    assert_eq!(common_rect(std::iter::empty::<&[bool]>(), 10, 8), None);
    let none = mask(10, 8, |_, _| false);
    assert_eq!(common_rect([none.as_slice()], 10, 8), None);
}

#[test]
fn a_usable_crop_is_taken_and_the_full_frame_needs_none() {
    let mut cov = CommonCoverage::new(10, 8);
    cov.fold(1, &mask(10, 8, |x, _| x >= 1));
    let decision = cov.crop(|i| format!("frame {i}"));
    assert!(!decision.refused);
    assert_eq!(decision.rect, (1, 0, 9, 8));
    assert_eq!(decision.needed(10, 8), Some((1, 0, 9, 8)));

    let full = CommonCoverage::new(10, 8).crop(|i| format!("frame {i}"));
    assert_eq!(full.rect, (0, 0, 10, 8));
    assert_eq!(full.needed(10, 8), None);
}

/// One wandering frame must not collapse the result: a common area below half
/// the frame is refused, and the frame that covers least is the one blamed.
#[test]
fn a_crop_below_half_the_frame_is_refused() {
    let mut cov = CommonCoverage::new(10, 8);
    cov.fold(1, &mask(10, 8, |x, _| x >= 2));
    cov.fold(2, &mask(10, 8, |x, _| x >= 7)); // only 3 columns survive
    assert_eq!(cov.worst(), Some((2, 24)));
    let decision = cov.crop(|i| format!("frame {i}"));
    assert!(decision.refused);
    assert_eq!(decision.rect, (0, 0, 10, 8), "the output is not cropped");
    assert_eq!(decision.needed(10, 8), None);

    let mut nothing = CommonCoverage::new(10, 8);
    nothing.fold(1, &mask(10, 8, |_, _| false));
    let decision = nothing.crop(|i| format!("frame {i}"));
    assert!(decision.refused);
    assert_eq!(decision.rect, (0, 0, 10, 8));
}
