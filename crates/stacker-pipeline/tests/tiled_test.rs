#![expect(
    clippy::suboptimal_flops,
    reason = "synthetic-image formulas read better without mul_add"
)]

//! `tiled`: the out-of-core tile loop. Tiled equals whole-image for every engine
//! (`Apex`, guided `Relief`, multigrid `Relief`, `Strata`), the callbacks are
//! honoured, the scratch directory never outlives a run, and the order and cull
//! decisions match the in-memory ones.

use std::{
    cell::{Cell, RefCell},
    path::{Path, PathBuf},
};

use image::{Rgb, RgbImage};
use stacker_core::{
    image::PlanarImage,
    memory::MmapTileManager,
    settings::{AlignmentModeSetting, OutputFormat, StackingSettings},
};
use stacker_pipeline::{
    Cancel, NeverCancel, NoProgress, PipelineError, Progress, Stage,
    compute::Compute,
    crop::CommonCoverage,
    fuse,
    output::{EncodedImage, encode},
    session::{PathSource, load_frames},
    tiled::{
        FusePlan, Hooks, MIN_TILE_SIZE, MemoryFrames, RunLog, ScratchDir, SilentLog, TileFrames,
        TileSink, TiledJob, TiledOutput, TiledPlan, crop_rect, effective_tile_size, frame_order,
        fuse_tiles, fuse_tiles_by, plan_run, run_tiled,
    },
    tiling::{Method, commit_frame},
};

// ── Fixtures ─────────────────────────────────────────────────────────────────────

/// A synthetic stack with detail at several scales and a different in-focus region
/// per frame (a Gaussian-weighted high-frequency texture), so the winner changes
/// across tile borders.
#[expect(
    clippy::cast_precision_loss,
    reason = "pixel coordinates and frame counts are small"
)]
fn multiscale_stack(width: usize, height: usize, count: usize) -> Vec<PlanarImage<f32>> {
    (0..count)
        .map(|k| {
            let mut img = PlanarImage::new(width, height);
            let cx = width as f32 * (k as f32 + 0.5) / count as f32;
            let cy = height as f32 * (0.5 + 0.2 * (k as f32 * 1.7).sin());
            let sigma = width as f32 / 6.0;
            for row in 0..height {
                for col in 0..width {
                    let (fx, fy) = (col as f32, row as f32);
                    let focus =
                        (-((fx - cx).powi(2) + (fy - cy).powi(2)) / (2.0 * sigma * sigma)).exp();
                    let fine = (0.9 * fx + 0.5 * fy).sin() * focus;
                    let mid = 0.5 * (0.11 * fx).sin() * (0.07 * fy).cos();
                    let coarse = 0.3 * (0.03 * fx + 0.02 * fy).cos();
                    let value = (0.5 + 0.35 * (fine + mid + coarse)).clamp(0.0, 1.0);
                    let idx = row * width + col;
                    img.luma[idx] = value;
                    img.chroma_a[idx] = (value - 0.5) * 0.2;
                    img.chroma_b[idx] = (0.5 - value) * 0.1;
                }
            }
            img
        })
        .collect()
}

fn max_plane_error(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> [f32; 3] {
    assert_eq!((a.width, a.height), (b.width, b.height));
    let worst = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .fold(0.0_f32, |acc, (p, q)| acc.max((p - q).abs()))
    };
    [
        worst(&a.luma, &b.luma),
        worst(&a.chroma_a, &b.chroma_a),
        worst(&a.chroma_b, &b.chroma_b),
    ]
}

fn assert_close(label: &str, tiled: &PlanarImage<f32>, whole: &PlanarImage<f32>, tol: f32) {
    let [luma, chroma_a, chroma_b] = max_plane_error(tiled, whole);
    assert!(
        luma <= tol,
        "{label}: luma deviates by {luma:.6} (tol {tol})"
    );
    assert!(
        chroma_a <= tol,
        "{label}: chroma_a deviates by {chroma_a:.6} (tol {tol})"
    );
    assert!(
        chroma_b <= tol,
        "{label}: chroma_b deviates by {chroma_b:.6} (tol {tol})"
    );
}

/// A fresh, empty directory of this test process.
fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stacker_tiled_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn entries(dir: &Path) -> usize {
    std::fs::read_dir(dir).unwrap().count()
}

/// Fuse `frames` (in the order `order` names) tile by tile in memory.
fn fuse_in_tiles(
    method: Method,
    frames: &[PlanarImage<f32>],
    order: &[usize],
    settings: &StackingSettings,
    tile: usize,
) -> PlanarImage<f32> {
    let size = (frames[0].width, frames[0].height);
    let plan = plan_run(method, size, tile, settings);
    let mut out = PlanarImage::new(size.0, size.1);
    fuse_tiles(
        &FusePlan::new(method, settings, &plan, order),
        &MemoryFrames(frames),
        &mut Compute::cpu(),
        &mut out,
        &Hooks::silent(),
    )
    .unwrap();
    out
}

fn in_order(frames: &[PlanarImage<f32>], order: &[usize]) -> Vec<PlanarImage<f32>> {
    order.iter().map(|&i| frames[i].clone()).collect()
}

// ── Tiled equals whole ──────────────────────────────────────────────────────────

/// **Tiled `Apex` equals whole-image `Apex`.** 128x128 with 32-px tiles and 4
/// pyramid levels (apron 32): every padded tile is a strict part of the image, so
/// the seams of a too-small apron would show.
#[test]
fn tiled_apex_matches_the_whole_image() {
    let settings = StackingSettings {
        pyramid_levels: 4,
        ..StackingSettings::default()
    };
    let frames = multiscale_stack(128, 128, 4);
    let plan = plan_run(Method::Apex, (128, 128), 32, &settings);
    assert_eq!((plan.geometry.tile_size, plan.geometry.apron), (32, 32));
    assert_eq!(plan.tiles.len(), 16);
    assert!(
        plan.tiles
            .iter()
            .all(|c| c.padded.width < 128 || c.padded.height < 128),
        "every padded tile must be a strict part of the image"
    );

    let whole = fuse::apex(&frames, &settings);
    let tiled = fuse_in_tiles(Method::Apex, &frames, &[0, 1, 2, 3], &settings, 32);
    assert_close("Apex", &tiled, &whole, 0.005);
}

/// **Tiled guided `Relief` equals whole-image `Relief`**, with the threshold decided
/// once from the whole-image focus map (percentile and auto-detect).
#[test]
fn tiled_relief_matches_the_whole_image() {
    let frames = multiscale_stack(96, 96, 3);
    for (contrast_pct, auto_detect) in [(0.3_f32, false), (0.0, true)] {
        let settings = StackingSettings {
            relief_contrast_pct: contrast_pct,
            relief_auto_detect: auto_detect,
            ..StackingSettings::default()
        };
        let plan = plan_run(Method::Relief, (96, 96), 32, &settings);
        assert_eq!((plan.geometry.tile_size, plan.geometry.apron), (32, 10));
        assert!(
            plan.tiles
                .iter()
                .all(|c| c.padded.width < 96 || c.padded.height < 96)
        );

        let whole = fuse::relief(&frames, &settings);
        let tiled = fuse_in_tiles(Method::Relief, &frames, &[0, 1, 2], &settings, 32);
        assert_close(
            &format!("Relief (pct {contrast_pct}, auto {auto_detect})"),
            &tiled,
            &whole,
            0.01,
        );
    }
}

/// **Tiled multigrid `Relief` equals the whole-image multigrid `Relief`.** The depth
/// is solved once on whole-image planes, so there are no seams: only the focus
/// maps' rounding may differ between a tile and the image.
#[test]
fn tiled_multigrid_relief_matches_the_whole_image() {
    let frames = multiscale_stack(96, 96, 4);
    for (contrast_pct, auto_detect) in [(0.3_f32, false), (0.0, true)] {
        let settings = StackingSettings {
            relief_use_multigrid: true,
            relief_contrast_pct: contrast_pct,
            relief_auto_detect: auto_detect,
            ..StackingSettings::default()
        };
        let whole = fuse::relief(&frames, &settings);
        assert_eq!(
            whole.luma,
            fuse::relief_multigrid(&frames, &settings).luma,
            "relief() takes the multigrid path with the flag"
        );
        for tile in [32, 40] {
            let tiled = fuse_in_tiles(Method::Relief, &frames, &[0, 1, 2, 3], &settings, tile);
            assert_close(
                &format!("multigrid (pct {contrast_pct}, auto {auto_detect}, tile {tile})"),
                &tiled,
                &whole,
                1e-5,
            );
        }
    }
}

/// **Tiled `Strata` equals whole-image `Strata`.** The apron is the algorithm's
/// 101 px: in a 288x224 image the corner tiles' padded regions end inside the
/// image, 101 px from their interiors.
#[test]
fn tiled_strata_matches_the_whole_image() {
    let settings = StackingSettings::default();
    let frames = multiscale_stack(288, 224, 3);
    let plan = plan_run(Method::Strata, (288, 224), 96, &settings);
    assert_eq!(plan.geometry.apron, 101);
    assert_eq!(plan.tiles.len(), 9);
    assert!(
        plan.tiles
            .iter()
            .any(|c| c.padded.width < 288 && c.padded.height < 224),
        "a padded tile must end inside the image"
    );

    let whole = fuse::strata(&frames, &settings);
    let tiled = fuse_in_tiles(Method::Strata, &frames, &[0, 1, 2], &settings, 96);
    assert_close("Strata", &tiled, &whole, 1e-3);
}

/// The fusion order maps to the frames of the store: fusing `[2, 0, 1]` is fusing
/// those frames in that order.
#[test]
fn the_fusion_order_selects_and_orders_the_stored_frames() {
    let frames = multiscale_stack(96, 96, 4);
    let settings = StackingSettings::default();
    for order in [vec![2, 0, 1], vec![3, 1]] {
        let whole = fuse::relief(&in_order(&frames, &order), &settings);
        let tiled = fuse_in_tiles(Method::Relief, &frames, &order, &settings, 32);
        assert_close(&format!("order {order:?}"), &tiled, &whole, 0.01);
    }
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..settings
    };
    let order = [2, 0, 3];
    let whole = fuse::relief(&in_order(&frames, &order), &multigrid);
    let tiled = fuse_in_tiles(Method::Relief, &frames, &order, &multigrid, 32);
    assert_close("multigrid order", &tiled, &whole, 1e-5);
}

// ── The per-tile loop ───────────────────────────────────────────────────────────

/// Without an engine: the loop cuts the tiles, passes them in order, and pastes the
/// interiors back, so a "fusion" that returns one frame reproduces that frame.
#[test]
fn the_tile_loop_reassembles_a_frame_exactly() {
    let frames = multiscale_stack(70, 50, 3);
    let plan = plan_run(Method::Relief, (70, 50), 24, &StackingSettings::default());
    let order = [2, 0];
    let mut out = PlanarImage::new(70, 50);
    let mut seen_counts = Vec::new();
    fuse_tiles_by(
        &FusePlan::new(Method::Relief, &StackingSettings::default(), &plan, &order),
        &MemoryFrames(&frames),
        &mut out,
        &Hooks::silent(),
        |tile, tile_frames| {
            seen_counts.push(tile_frames.len());
            assert_eq!(tile_frames[0].width, tile.padded.width);
            Ok(tile_frames[0].clone())
        },
    )
    .unwrap();
    assert!(seen_counts.iter().all(|&n| n == 2));
    assert_eq!(seen_counts.len(), plan.tiles.len());
    assert_eq!(out.luma, frames[2].luma);
    assert_eq!(out.chroma_a, frames[2].chroma_a);
    assert_eq!(out.chroma_b, frames[2].chroma_b);
}

#[test]
fn bad_requests_to_the_loop_are_errors() {
    let frames = multiscale_stack(48, 48, 2);
    let settings = StackingSettings::default();
    let plan = plan_run(Method::Relief, (48, 48), 24, &settings);
    let mut out = PlanarImage::new(48, 48);
    let hooks = Hooks::silent();

    // No frames to fuse.
    let none = fuse_tiles(
        &FusePlan::new(Method::Relief, &settings, &plan, &[]),
        &MemoryFrames(&frames),
        &mut Compute::cpu(),
        &mut out,
        &hooks,
    );
    assert!(matches!(none, Err(PipelineError::Invalid(_))));

    // A frame that is not in the store.
    let missing = fuse_tiles(
        &FusePlan::new(Method::Relief, &settings, &plan, &[0, 5]),
        &MemoryFrames(&frames),
        &mut Compute::cpu(),
        &mut out,
        &hooks,
    );
    assert!(matches!(missing, Err(PipelineError::Invalid(_))));
    assert!(MemoryFrames(&frames).tile(9, &plan.tiles[0]).is_err());

    // The neural mode has no engine here.
    let neural = fuse_tiles(
        &FusePlan::new(Method::Neural, &settings, &plan, &[0, 1]),
        &MemoryFrames(&frames),
        &mut Compute::cpu(),
        &mut out,
        &hooks,
    );
    assert!(matches!(neural, Err(PipelineError::Invalid(_))));
}

struct Recorder(RefCell<Vec<(Stage, f32)>>);

impl Progress for Recorder {
    fn stage(&self, stage: Stage, fraction: f32, _label: &str) {
        self.0.borrow_mut().push((stage, fraction));
    }
}

struct CancelAfter {
    polls: Cell<usize>,
    limit: usize,
}

impl Cancel for CancelAfter {
    fn is_cancelled(&self) -> bool {
        let n = self.polls.get();
        self.polls.set(n + 1);
        n >= self.limit
    }
}

/// One progress step per tile and pass: `Apex` and `Strata` make one pass,
/// `Relief` two (the threshold or the winner planes, then the fusion).
#[test]
fn progress_counts_tiles_per_pass_and_ends_at_one() {
    let frames = multiscale_stack(96, 96, 2);
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..StackingSettings::default()
    };
    let levels = StackingSettings {
        pyramid_levels: 3,
        ..StackingSettings::default()
    };
    let default = StackingSettings::default();
    let cases = [
        (Method::Apex, &levels, 9),
        (Method::Relief, &default, 18),
        (Method::Relief, &multigrid, 18),
        (Method::Strata, &default, 9),
    ];
    for (method, settings, expected) in cases {
        let plan = plan_run(method, (96, 96), 32, settings);
        assert_eq!(plan.tiles.len(), 9);
        let recorder = Recorder(RefCell::new(Vec::new()));
        let hooks = Hooks {
            progress: &recorder,
            cancel: &NeverCancel,
            log: &SilentLog,
        };
        let mut out = PlanarImage::new(96, 96);
        fuse_tiles(
            &FusePlan::new(method, settings, &plan, &[0, 1]),
            &MemoryFrames(&frames),
            &mut Compute::cpu(),
            &mut out,
            &hooks,
        )
        .unwrap();
        let seen = recorder.0.borrow();
        assert_eq!(seen.len(), expected, "{method:?}");
        assert!(seen.iter().all(|(stage, _)| *stage == Stage::Fuse));
        assert!(seen.windows(2).all(|pair| pair[0].1 <= pair[1].1));
        assert!((seen[seen.len() - 1].1 - 1.0).abs() < f32::EPSILON);
    }
}

/// Cancel is polled between tiles in every pass of every engine.
#[test]
fn a_cancelled_loop_stops_with_cancelled() {
    let frames = multiscale_stack(96, 96, 2);
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..StackingSettings::default()
    };
    // Apex at the default eight levels needs an apron wider than the image, so
    // it is one tile and would be polled only once; three levels give 3 x 3.
    let apex = StackingSettings {
        pyramid_levels: 3,
        ..StackingSettings::default()
    };
    for (method, settings) in [
        (Method::Apex, apex),
        (Method::Relief, StackingSettings::default()),
        (Method::Relief, multigrid),
        (Method::Strata, StackingSettings::default()),
    ] {
        let plan = plan_run(method, (96, 96), 32, &settings);
        assert!(
            plan.tiles.len() > 2,
            "{method:?}: {} tiles",
            plan.tiles.len()
        );
        let cancel = CancelAfter {
            polls: Cell::new(0),
            limit: 2,
        };
        let hooks = Hooks {
            progress: &NoProgress,
            cancel: &cancel,
            log: &SilentLog,
        };
        let mut out = PlanarImage::new(96, 96);
        let result = fuse_tiles(
            &FusePlan::new(method, &settings, &plan, &[0, 1]),
            &MemoryFrames(&frames),
            &mut Compute::cpu(),
            &mut out,
            &hooks,
        );
        assert!(
            matches!(result, Err(PipelineError::Cancelled)),
            "{method:?}: {result:?}"
        );
    }
}

// ── Sinks and plans ─────────────────────────────────────────────────────────────

/// The streamed tile output encodes exactly like the whole-image encoder (8 and 16
/// bits): only the addressing differs.
#[test]
fn the_encoded_sink_matches_the_whole_image_encoding() {
    const W: usize = 64;
    const H: usize = 64;
    let whole_fused = fuse::apex(&multiscale_stack(W, H, 4), &StackingSettings::default());
    for bit_depth in [8, 16] {
        let saving = stacker_core::settings::ImageSavingSettings {
            output_format: OutputFormat::Png,
            bit_depth,
            ..stacker_core::settings::ImageSavingSettings::default()
        };
        let reference = encode(&whole_fused, &saving);
        let mut streamed = EncodedImage::zeroed(W, H, OutputFormat::Png, bit_depth == 16, 95);
        for coord in stacker_core::memory::enumerate_tiles(W, H, 16) {
            streamed.paste(
                &whole_fused,
                (coord.start_x, coord.start_y),
                (coord.start_x, coord.start_y, coord.width, coord.height),
            );
        }
        assert_eq!(streamed.rgb8(), reference.rgb8(), "8-bit samples differ");
        assert_eq!(streamed.rgb16(), reference.rgb16(), "16-bit samples differ");
        assert!(reference.rgb8().is_some() || reference.rgb16().is_some());
    }
}

#[test]
fn the_plan_reports_disk_memory_and_padding() {
    let settings = StackingSettings::default();

    // One tile: the store holds each frame once, three planes.
    let single = plan_run(Method::Relief, (100, 80), 4096, &settings);
    assert_eq!(single.tiles.len(), 1);
    assert_eq!(single.geometry.tile_size, 100, "limited to the image");
    assert_eq!(single.scratch_bytes(5), 100 * 80 * 12 * 5);

    // The padded tiles overlap, so a grid stores more than the image.
    let grid: TiledPlan = plan_run(Method::Relief, (100, 100), 32, &settings);
    assert!(grid.scratch_bytes(1) > 100 * 100 * 12);

    // The multigrid adds its whole-image planes to the tile working set.
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..settings.clone()
    };
    let guided_ram = grid.fusion_ram_bytes(Method::Relief, 6, &settings);
    let multigrid_ram = grid.fusion_ram_bytes(Method::Relief, 6, &multigrid);
    assert_eq!(multigrid_ram - guided_ram, 32 * 100 * 100);
    // Strata's fixed working set is in the figure.
    assert!(
        grid.fusion_ram_bytes(Method::Strata, 6, &settings)
            > grid.fusion_ram_bytes(Method::Relief, 6, &settings)
    );

    // A deep pyramid has a huge apron: the plan says so, and only for Apex.
    let apex = plan_run(Method::Apex, (2000, 2000), 128, &settings);
    let warning = apex.padding_warning(Method::Apex).expect("a warning");
    assert!(warning.contains("fewer pyramid levels"), "{warning}");
    assert!(
        plan_run(Method::Relief, (2000, 2000), 512, &settings)
            .padding_warning(Method::Relief)
            .is_none()
    );
}

#[test]
fn the_tile_size_comes_from_the_request_or_the_setting() {
    let settings = StackingSettings {
        tile_size: 256,
        ..StackingSettings::default()
    };
    assert_eq!(effective_tile_size(0, &settings), 256);
    assert_eq!(effective_tile_size(128, &settings), 128);
    assert_eq!(effective_tile_size(0, &StackingSettings::default()), 0);
}

#[test]
fn the_scratch_directory_is_removed_on_drop_and_on_unwind() {
    let root = temp("scratch");
    let guard = ScratchDir::create(&root).unwrap();
    let path = guard.path().to_path_buf();
    std::fs::write(path.join("tile.bin"), b"data").unwrap();
    assert!(path.is_dir());
    drop(guard);
    assert!(!path.exists());

    let seen = RefCell::new(PathBuf::new());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let guard = ScratchDir::create(&root).unwrap();
        *seen.borrow_mut() = guard.path().to_path_buf();
        panic!("forced failure inside the guarded scope");
    }));
    assert!(result.is_err());
    assert!(!seen.borrow().as_os_str().is_empty());
    assert!(!seen.borrow().exists(), "the guard cleans up on unwind");
    // Two runs in one process get two directories.
    let a = ScratchDir::create(&root).unwrap();
    let b = ScratchDir::create(&root).unwrap();
    assert_ne!(a.path(), b.path());
    drop((a, b));
    assert_eq!(entries(&root), 0);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_crop_follows_crop_to_common_area() {
    let mut coverage = CommonCoverage::new(10, 8);
    let mask: Vec<bool> = (0..80).map(|i| i % 10 >= 2).collect();
    coverage.fold(1, &mask);
    let on = StackingSettings::default();
    assert!(on.crop_to_common_area, "cropping is the default");
    assert_eq!(
        crop_rect(&coverage, &on, (10, 8), |i| format!("f{i}")),
        Some((2, 0, 8, 8))
    );
    let off = StackingSettings {
        crop_to_common_area: false,
        ..on
    };
    assert_eq!(
        crop_rect(&coverage, &off, (10, 8), |i| format!("f{i}")),
        None
    );
    assert_eq!(
        crop_rect(
            &CommonCoverage::new(10, 8),
            &StackingSettings::default(),
            (10, 8),
            |i| format!("f{i}")
        ),
        None,
        "nothing to cut"
    );
}

// ── Order and cull from the store ───────────────────────────────────────────────

/// Frame `k` carries hash noise in the column band `bands[k]` (of `n_bands`) and a
/// smooth ramp elsewhere; `u16::MAX` is a frame with no texture.
fn banded(w: u16, h: u16, n_bands: u16, bands: &[u16]) -> Vec<PlanarImage<f32>> {
    let width = w / n_bands;
    bands
        .iter()
        .enumerate()
        .map(|(k, &band)| {
            let mut img = PlanarImage::new(usize::from(w), usize::from(h));
            let seed = u32::try_from(k + 1).unwrap();
            for y in 0..h {
                for x in 0..w {
                    let hash = (u32::from(x).wrapping_mul(73_856_093)
                        ^ u32::from(y).wrapping_mul(19_349_663)
                        ^ seed.wrapping_mul(83_492_791))
                        % 1000;
                    let noise = f32::from(u16::try_from(hash).unwrap()) / 1000.0 - 0.5;
                    let ramp = 0.4 + 0.2 * f32::from(x) / f32::from(w);
                    let i = usize::from(y) * usize::from(w) + usize::from(x);
                    img.luma[i] = ramp + if x / width == band { 0.3 * noise } else { 0.0 };
                }
            }
            img
        })
        .collect()
}

struct LogBook(RefCell<Vec<String>>);

impl RunLog for LogBook {
    fn info(&self, _message: &str) {}

    fn warn(&self, message: &str) {
        self.0.borrow_mut().push(message.to_owned());
    }
}

fn order_of(
    frames: &[PlanarImage<f32>],
    settings: &StackingSettings,
    tag: &str,
    book: &LogBook,
) -> Vec<usize> {
    let plan = plan_run(Method::Relief, (96, 64), 32, settings);
    let dir = temp(tag);
    let mut store = MmapTileManager::new(dir.clone());
    for (i, frame) in frames.iter().enumerate() {
        commit_frame(&mut store, i, frame, &plan.tiles).unwrap();
    }
    let hooks = Hooks {
        progress: &NoProgress,
        cancel: &NeverCancel,
        log: book,
    };
    let order = frame_order(settings, &plan, &store, frames.len(), &hooks).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    order
}

#[test]
fn the_order_from_the_store_follows_cull_sort_and_threshold() {
    // Four frames, each sharp in one quarter, scrambled: the census order is
    // frame 1 (left), 3, 2, 0 (right); a fifth frame has no texture at all.
    let frames = banded(96, 64, 4, &[3, 0, 2, 1, u16::MAX]);
    let settings = |auto_cull: bool, sort: bool, pct: f32| StackingSettings {
        auto_cull,
        sort_by_sharpness: sort,
        auto_cull_threshold_pct: pct,
        ..StackingSettings::default()
    };
    let book = LogBook(RefCell::new(Vec::new()));

    // Neither toggle: stack order, nothing dropped.
    assert_eq!(
        order_of(&frames, &settings(false, false, 0.0), "o1", &book),
        [0, 1, 2, 3, 4]
    );
    // The cull drops the textureless frame; survivors stay in stack order.
    assert_eq!(
        order_of(&frames, &settings(true, false, 0.0), "o2", &book),
        [0, 1, 2, 3]
    );
    // Sorting alone keeps every frame, in census order; the textureless frame
    // (centroid in the middle of the image) sits wherever the walk passes it.
    let sorted = order_of(&frames, &settings(false, true, 0.0), "o3", &book);
    assert_eq!(sorted.len(), 5);
    let textured: Vec<usize> = sorted.iter().copied().filter(|&i| i != 4).collect();
    assert_eq!(textured, [1, 3, 2, 0]);
    // Both: the survivors in census order.
    assert_eq!(
        order_of(&frames, &settings(true, true, 0.0), "o4", &book),
        [1, 3, 2, 0]
    );
    // A fixed threshold: 10 % behaves like the automatic rule here ...
    assert_eq!(
        order_of(&frames, &settings(true, false, 10.0), "o5", &book),
        [0, 1, 2, 3]
    );
    assert!(book.0.borrow().is_empty(), "no warning so far");
    // ... and 40 % would drop everything: all are kept and the log says so.
    assert_eq!(
        order_of(&frames, &settings(true, false, 40.0), "o6", &book),
        [0, 1, 2, 3, 4]
    );
    assert!(
        book.0
            .borrow()
            .iter()
            .any(|w| w.contains("would have dropped every frame")),
        "{:?}",
        book.0.borrow()
    );
}

// ── The whole run, from files ───────────────────────────────────────────────────

const RUN_W: u32 = 96;
const RUN_H: u32 = 80;

/// Write frame `k`: hash texture in column band `bands[k]` (of `n_bands`), a gentle
/// ramp elsewhere.
fn write_frames(dir: &Path, n_bands: u32, bands: &[u32]) -> Vec<PathBuf> {
    let width = RUN_W / n_bands;
    bands
        .iter()
        .enumerate()
        .map(|(k, &band)| {
            let seed = u32::try_from(k + 1).unwrap();
            let image = RgbImage::from_fn(RUN_W, RUN_H, |x, y| {
                let hash = x.wrapping_mul(73_856_093)
                    ^ y.wrapping_mul(19_349_663)
                    ^ seed.wrapping_mul(83_492_791);
                let ramp = 80 + x * 60 / RUN_W;
                let texture = if x / width == band { hash % 100 } else { 0 };
                let v = u8::try_from(ramp + texture).unwrap();
                Rgb([v, v, v])
            });
            let path = dir.join(format!("frame_{k:02}.png"));
            image.save(&path).unwrap();
            path
        })
        .collect()
}

fn plain_settings() -> StackingSettings {
    StackingSettings {
        alignment_mode: AlignmentModeSetting::None,
        correct_brightness: false,
        auto_cull: false,
        sort_by_sharpness: false,
        pyramid_levels: 3,
        ..StackingSettings::default()
    }
}

fn whole_fusion(
    method: Method,
    paths: &[PathBuf],
    settings: &StackingSettings,
) -> PlanarImage<f32> {
    let loaded = load_frames(
        &mut PathSource::new(paths.to_vec()),
        &settings.preprocessing,
        &NeverCancel,
        &NoProgress,
    )
    .unwrap();
    match method {
        Method::Apex => fuse::apex(&loaded.frames, settings),
        Method::Relief => fuse::relief(&loaded.frames, settings),
        Method::Strata => fuse::strata(&loaded.frames, settings),
        Method::Neural => unreachable!("no whole-image neural fusion here"),
    }
}

fn run_from_files(
    method: Method,
    paths: &[PathBuf],
    settings: &StackingSettings,
    root: &Path,
) -> Result<stacker_pipeline::tiled::TiledOutput<PlanarImage<f32>>, PipelineError> {
    run_tiled(
        &mut PathSource::new(paths.to_vec()),
        &TiledJob {
            method,
            settings,
            tile_size: MIN_TILE_SIZE,
            scratch_root: root,
        },
        &mut Compute::cpu(),
        PlanarImage::new,
        &Hooks::silent(),
    )
}

/// Decode, align (off), tile, fuse: the result of a run from files is the
/// whole-image fusion of the same decoded frames, for every engine, and the
/// scratch directory is gone afterwards.
#[test]
fn a_run_from_files_equals_the_whole_image_fusion() {
    let dir = temp("run_files");
    let root = temp("run_scratch");
    let paths = write_frames(&dir, 3, &[2, 0, 1]);
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..plain_settings()
    };
    for (method, settings, tol) in [
        (Method::Apex, plain_settings(), 0.005),
        (Method::Relief, plain_settings(), 0.01),
        (Method::Relief, multigrid, 1e-5),
        (Method::Strata, plain_settings(), 1e-3),
    ] {
        let out = run_from_files(method, &paths, &settings, &root).unwrap();
        assert_eq!(out.size, (96, 80));
        assert_eq!(out.order, [0, 1, 2]);
        assert_eq!(out.crop, None, "no alignment, nothing to crop");
        let whole = whole_fusion(method, &paths, &settings);
        assert_close(&format!("{method:?} from files"), &out.sink, &whole, tol);
        assert_eq!(entries(&root), 0, "{method:?} left scratch files behind");
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&root);
}

/// The cull and sort keys reach the run: the textureless frame is dropped and the
/// order is reported.
#[test]
fn a_run_reports_the_frames_it_fused() {
    let dir = temp("run_order");
    let root = temp("run_order_scratch");
    // Frame 3 has no texture (band 9 does not exist).
    let paths = write_frames(&dir, 3, &[2, 0, 1, 9]);
    let settings = StackingSettings {
        auto_cull: true,
        ..plain_settings()
    };
    let out = run_from_files(Method::Relief, &paths, &settings, &root).unwrap();
    assert_eq!(out.order, [0, 1, 2], "stack order of the survivors");

    let sorted = StackingSettings {
        sort_by_sharpness: true,
        ..settings
    };
    let out = run_from_files(Method::Relief, &paths, &sorted, &root).unwrap();
    assert_eq!(out.order, [1, 2, 0], "census order: left band first");
    assert_eq!(entries(&root), 0);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_failed_run_leaves_no_scratch_files() {
    let dir = temp("run_fail");
    let root = temp("run_fail_scratch");
    let mut paths = write_frames(&dir, 3, &[2, 0]);
    // A frame of another size ends the run, with the file named.
    let odd = dir.join("frame_odd.png");
    RgbImage::from_pixel(RUN_W - 8, RUN_H, Rgb([9, 9, 9]))
        .save(&odd)
        .unwrap();
    paths.push(odd);
    let error = run_from_files(Method::Relief, &paths, &plain_settings(), &root).unwrap_err();
    assert!(matches!(error, PipelineError::Invalid(_)), "{error:?}");
    let text = error.to_string();
    assert!(
        text.contains("frame_odd.png") && text.contains("same size"),
        "{text}"
    );
    assert_eq!(entries(&root), 0);

    // Cancelled before the first decode: nothing is created.
    let cancelled = run_tiled(
        &mut PathSource::new(paths[..2].to_vec()),
        &TiledJob {
            method: Method::Relief,
            settings: &plain_settings(),
            tile_size: MIN_TILE_SIZE,
            scratch_root: &root,
        },
        &mut Compute::cpu(),
        PlanarImage::<f32>::new,
        &Hooks {
            progress: &NoProgress,
            cancel: &CancelAfter {
                polls: Cell::new(0),
                limit: 0,
            },
            log: &SilentLog,
        },
    );
    assert!(matches!(cancelled, Err(PipelineError::Cancelled)));
    assert_eq!(entries(&root), 0);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_run_refuses_what_it_cannot_do_before_decoding_anything() {
    let root = temp("run_refuse");
    let settings = plain_settings();
    let paths = vec![PathBuf::from("does_not_exist.png")];
    let job = |method: Method, tile_size: usize| TiledJob {
        method,
        settings: &settings,
        tile_size,
        scratch_root: &root,
    };
    let run = |job: &TiledJob<'_>, paths: Vec<PathBuf>| {
        run_tiled(
            &mut PathSource::new(paths),
            job,
            &mut Compute::cpu(),
            PlanarImage::<f32>::new,
            &Hooks::silent(),
        )
        .err()
    };
    assert!(matches!(
        run(&job(Method::Relief, MIN_TILE_SIZE - 1), paths.clone()),
        Some(PipelineError::Invalid(_))
    ));
    assert!(matches!(
        run(&job(Method::Neural, MIN_TILE_SIZE), paths.clone()),
        Some(PipelineError::Invalid(_))
    ));
    assert!(matches!(
        run(&job(Method::Relief, MIN_TILE_SIZE), Vec::new()),
        Some(PipelineError::Invalid(_))
    ));
    // An unreadable reference frame is the decoder's error.
    assert!(matches!(
        run(&job(Method::Relief, MIN_TILE_SIZE), paths),
        Some(PipelineError::Decode { .. })
    ));
    assert_eq!(entries(&root), 0);
    let _ = std::fs::remove_dir_all(&root);
}

/// `finish` on an encoded sink crops and resizes back to the canvas like the
/// planar one does, up to the encoding of the resampled samples (one 8-bit code).
#[test]
fn an_encoded_output_is_cropped_and_resized_like_a_planar_one() {
    fn output<K>(sink: K, crop: Option<(usize, usize, usize, usize)>) -> TiledOutput<K> {
        TiledOutput {
            sink,
            size: (24, 16),
            order: vec![0],
            crop,
            coverage: CommonCoverage::new(24, 16),
        }
    }
    let (w, h) = (24_usize, 16_usize);
    let mut planar = PlanarImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (
                f32::from(u8::try_from(x).unwrap()) / 24.0,
                f32::from(u8::try_from(y).unwrap()) / 16.0,
            );
            planar.luma[y * w + x] = 0.4 + 0.06 * fx + 0.01 * fy;
        }
    }
    let saving = stacker_core::settings::ImageSavingSettings {
        output_format: OutputFormat::Png,
        bit_depth: 8,
        ..stacker_core::settings::ImageSavingSettings::default()
    };
    let crop = Some((2, 1, 18, 12));
    let on = StackingSettings {
        resize_cropped_to_original: true,
        ..StackingSettings::default()
    };
    let off = StackingSettings {
        resize_cropped_to_original: false,
        ..StackingSettings::default()
    };

    let expected = encode(&output(planar.clone(), crop).finish(&on), &saving);
    let got = output(encode(&planar, &saving), crop).finish(&on);
    assert_eq!((got.width(), got.height()), (w, h));
    let worst = got
        .rgb8()
        .unwrap()
        .iter()
        .zip(expected.rgb8().unwrap())
        .map(|(&a, &b)| a.abs_diff(b))
        .max()
        .unwrap();
    assert!(worst <= 1, "off by {worst} codes");

    // Without the setting the result is the plain crop; without a crop, the canvas.
    let cropped = output(encode(&planar, &saving), crop).finish(&off);
    assert_eq!((cropped.width(), cropped.height()), (18, 12));
    let whole = output(encode(&planar, &saving), None).finish(&on);
    assert_eq!((whole.width(), whole.height()), (w, h));
}
