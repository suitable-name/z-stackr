//! `fuse`: the shared `Apex` / `Relief` / `Strata` stages, the multigrid `Relief`
//! and the whole-image `Relief` threshold.

use stacker_algo::{
    relief::threshold::mask_above,
    strata::{CpuStrata, NO_PREVIEW},
};
use stacker_core::{image::PlanarImage, settings::StackingSettings};
use stacker_pipeline::{
    fuse::{
        DepthVotes, MAX_MULTIGRID_FRAMES, apex, focus_maps, multigrid_votes, relief,
        relief_multigrid, relief_threshold, relief_tile, select_and_smooth, strata, strata_on,
        strata_params,
    },
    tiling::plan_tiles,
};

/// Frame `k` carries deterministic noise texture in its own band of columns and
/// a smooth ramp elsewhere.
fn bands(w: u16, h: u16, n: u16) -> Vec<PlanarImage<f32>> {
    (0..n)
        .map(|k| {
            let mut img = PlanarImage::new(usize::from(w), usize::from(h));
            let band = f32::from(w) / f32::from(n);
            for y in 0..h {
                for x in 0..w {
                    let fx = f32::from(x);
                    let in_band = fx >= band * f32::from(k) && fx < band * f32::from(k + 1);
                    let hash = (u32::from(x).wrapping_mul(73_856_093)
                        ^ u32::from(y).wrapping_mul(19_349_663)
                        ^ u32::from(k + 1).wrapping_mul(83_492_791))
                        % 1000;
                    let noise = f32::from(u16::try_from(hash).unwrap()) / 1000.0 - 0.5;
                    let base = 0.4 + 0.2 * fx / f32::from(w);
                    let i = usize::from(y) * usize::from(w) + usize::from(x);
                    img.luma[i] = base + if in_band { 0.3 * noise } else { 0.0 };
                    img.chroma_a[i] = 0.05;
                    img.chroma_b[i] = -0.03;
                }
            }
            img
        })
        .collect()
}

/// FUS-4 (CLI half): the threshold of a tiled run is computed once, from the
/// whole-image focus map, and every tile is fused against that one value. The
/// map the CLI assembles from the tile interiors equals the whole-image map, so
/// the value it computes does too, for both the percentile rule and the
/// median + MAD auto-detection.
#[test]
fn the_threshold_from_tile_interiors_equals_the_whole_image_threshold() {
    let (w, h) = (96_u16, 64_u16);
    let frames = bands(w, h, 3);
    let (wu, hu) = (usize::from(w), usize::from(h));

    for (pct, auto) in [(0.0_f32, false), (0.3, false), (0.7, false), (0.0, true)] {
        let settings = StackingSettings {
            relief_contrast_pct: pct,
            relief_auto_detect: auto,
            ..StackingSettings::default()
        };
        let radius = settings.relief_estimation_radius as usize;
        let whole = focus_maps(&frames, radius).max;
        let want = relief_threshold(&whole.luma, &settings);

        // The CLI's pre-pass: per padded tile, focus maps; keep the interiors.
        let mut assembled = vec![f32::NEG_INFINITY; wu * hu];
        for cell in plan_tiles(wu, hu, 32, 10) {
            let p = &cell.padded;
            let tiles: Vec<PlanarImage<f32>> = frames
                .iter()
                .map(|f| {
                    stacker_core::memory::extract_tile(f, p.start_x, p.start_y, p.width, p.height)
                })
                .collect();
            let maps = focus_maps(&tiles, radius);
            let (ix, iy) = cell.interior_offset;
            let i = &cell.interior;
            for row in 0..i.height {
                let src = (iy + row) * maps.max.width + ix;
                let dst = (i.start_y + row) * wu + i.start_x;
                assembled[dst..dst + i.width].copy_from_slice(&maps.max.luma[src..src + i.width]);
            }
        }
        let got = relief_threshold(&assembled, &settings);

        assert!(
            (got.value - want.value).abs() <= 1e-4 * want.value.abs().max(1e-3),
            "pct {pct} auto {auto}: tiled {} vs whole {}",
            got.value,
            want.value
        );
        assert!((got.contrast_pct - want.contrast_pct).abs() < 1e-3);
    }
}

/// The percentile is a property of the whole image: tiles fused against the
/// whole-image value reject exactly the whole-image pixels, while a per-tile
/// threshold of the same fraction rejects a different set.
#[test]
fn one_threshold_for_every_tile_gives_one_mask() {
    let (w, h) = (96_u16, 64_u16);
    let frames = bands(w, h, 3);
    let settings = StackingSettings {
        relief_contrast_pct: 0.5,
        ..StackingSettings::default()
    };
    let maps = focus_maps(&frames, settings.relief_estimation_radius as usize);
    let value = relief_threshold(&maps.max.luma, &settings).value;
    let whole_mask = mask_above(&maps.max.luma, value);
    let kept = whole_mask.iter().filter(|&&k| k).count();
    let total = whole_mask.len();
    // About half of the pixels are kept (ties can move it a little).
    assert!(
        kept * 4 > total && kept * 4 < total * 3,
        "{kept} of {total}"
    );

    // A percentile evaluated per padded tile would pick its own absolute value
    // in (almost) every tile, so the mask would jump at the tile borders.
    let (wu, hu) = (usize::from(w), usize::from(h));
    let mut differing = 0_usize;
    for cell in plan_tiles(wu, hu, 32, 10) {
        let p = &cell.padded;
        let tiles: Vec<PlanarImage<f32>> = frames
            .iter()
            .map(|f| stacker_core::memory::extract_tile(f, p.start_x, p.start_y, p.width, p.height))
            .collect();
        let local = focus_maps(&tiles, settings.relief_estimation_radius as usize).max;
        let local_value = relief_threshold(&local.luma, &settings).value;
        if (local_value - value).abs() > 1e-4 * value.abs().max(1e-3) {
            differing += 1;
        }
    }
    assert!(
        differing > 0,
        "per-tile thresholds are not the whole-image one"
    );
}

/// A tile fused with the whole-image threshold equals the corresponding part of
/// the whole-image fusion, and `relief` is `relief_tile` with its own threshold.
#[test]
fn relief_equals_relief_tile_with_its_own_threshold() {
    let frames = bands(48, 40, 3);
    let settings = StackingSettings {
        relief_contrast_pct: 0.4,
        ..StackingSettings::default()
    };
    let maps = focus_maps(&frames, settings.relief_estimation_radius as usize);
    let value = relief_threshold(&maps.max.luma, &settings).value;
    let whole = relief(&frames, &settings);
    let via_tile = relief_tile(&frames, &settings, value);
    assert_eq!(whole.luma, via_tile.luma);
    assert_eq!(whole.chroma_a, via_tile.chroma_a);
    assert_eq!(whole.chroma_b, via_tile.chroma_b);
}

/// Where the mask is `true` the sharpest frame wins (earlier frame on ties),
/// elsewhere the mean of the frames; only the luma is smoothed.
#[test]
fn selection_follows_the_mask() {
    let mut a = PlanarImage::<f32>::new(4, 1);
    let mut b = PlanarImage::<f32>::new(4, 1);
    a.luma = vec![0.2, 0.2, 0.2, 0.2];
    b.luma = vec![0.8, 0.8, 0.8, 0.8];
    a.chroma_a = vec![0.1; 4];
    b.chroma_a = vec![0.3; 4];
    let sa = vec![1.0, 1.0, 0.0, 0.0];
    let mut sb = vec![0.0, 0.0, 1.0, 1.0];
    sb[0] = 1.0; // a tie at pixel 0: the earlier frame wins
    let mask = [true, true, true, false];

    let out = select_and_smooth(&[a, b], &[sa, sb], &mask, 0);
    // Radius 0: the guided filter is the identity on the luma it was given.
    assert!(
        (out.luma[0] - 0.2).abs() < 1e-5,
        "tie goes to frame 0: {}",
        out.luma[0]
    );
    assert!((out.luma[1] - 0.2).abs() < 1e-5);
    assert!((out.luma[2] - 0.8).abs() < 1e-5);
    assert!(
        (out.luma[3] - 0.5).abs() < 1e-5,
        "masked-out pixel is the mean"
    );
    assert!((out.chroma_a[1] - 0.1).abs() < 1e-6);
    assert!((out.chroma_a[2] - 0.3).abs() < 1e-6);
    assert!((out.chroma_a[3] - 0.2).abs() < 1e-6);
}

#[test]
fn apex_reads_its_keys_and_handles_an_empty_stack() {
    let frames = bands(64, 64, 3);
    let base = StackingSettings::default();
    let fused = apex(&frames, &base);
    assert_eq!((fused.width, fused.height), (64, 64));
    assert!(fused.is_consistent());

    let shallow = apex(
        &frames,
        &StackingSettings {
            pyramid_levels: 3,
            ..base.clone()
        },
    );
    assert_ne!(
        fused.luma, shallow.luma,
        "pyramid_levels reaches the fusion"
    );
    assert_eq!(apex(&[], &base).width, 0);
    assert_eq!(relief(&[], &base).width, 0);
}

// ── Strata ──────────────────────────────────────────────────────────────────

fn same_planes(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.luma == b.luma
        && a.chroma_a == b.chroma_a
        && a.chroma_b == b.chroma_b
}

/// `strata_base_radius` and `strata_detail_focus` are the only Strata keys; both
/// reach the result, and a hand-built value outside the settings' range is
/// clamped so the apron assumption holds.
#[test]
fn strata_reads_its_keys_and_clamps_them() {
    let frames = bands(96, 64, 3);
    let base = StackingSettings::default();
    let reference = strata(&frames, &base);
    assert_eq!((reference.width, reference.height), (96, 64));
    assert!(reference.is_consistent());
    assert!(reference.luma.iter().all(|v| v.is_finite()));

    let narrow = StackingSettings {
        strata_base_radius: 8,
        ..base.clone()
    };
    assert!(
        !same_planes(&reference, &strata(&frames, &narrow)),
        "strata_base_radius has no effect"
    );
    let soft = StackingSettings {
        strata_detail_focus: 1,
        ..base.clone()
    };
    assert!(
        !same_planes(&reference, &strata(&frames, &soft)),
        "strata_detail_focus has no effect"
    );

    let wild = StackingSettings {
        strata_base_radius: 1000,
        strata_detail_focus: 0,
        ..base.clone()
    };
    let params = strata_params(&wild);
    assert_eq!((params.base_radius, params.detail_focus), (64, 1));
    let tiny = StackingSettings {
        strata_base_radius: 1,
        strata_detail_focus: 99,
        ..base.clone()
    };
    let params = strata_params(&tiny);
    assert_eq!((params.base_radius, params.detail_focus), (8, 5));
    let defaults = strata_params(&base);
    assert_eq!((defaults.base_radius, defaults.detail_focus), (31, 3));

    assert_eq!(strata(&[], &base).width, 0);
}

/// `strata_on` is the entry the device seam plugs into: with the CPU kernels it is
/// `strata`, it counts two ticks per frame, and previews can be switched off.
#[test]
fn strata_on_with_the_cpu_kernels_is_strata() {
    let frames = bands(64, 48, 3);
    let settings = StackingSettings::default();
    let want = strata(&frames, &settings);

    let mut ticks = 0_usize;
    let mut previews = 0_usize;
    let got = strata_on(
        &mut CpuStrata,
        frames.as_slice(),
        &settings,
        || ticks += 1,
        NO_PREVIEW,
        |_: &PlanarImage<f32>| previews += 1,
    );
    assert!(same_planes(&want, &got));
    assert_eq!(ticks, 6, "two passes over three frames");
    assert_eq!(previews, 0);
}

/// Identical frames carry no focus information: the first one wins every pixel and
/// the result is that frame (the tie rule shared with `Apex` and `Relief`).
#[test]
fn strata_of_identical_frames_is_the_first_frame() {
    let frame = bands(48, 40, 1).remove(0);
    let frames = [frame.clone(), frame.clone(), frame.clone()];
    let fused = strata(&frames, &StackingSettings::default());
    let worst = fused
        .luma
        .iter()
        .zip(&frame.luma)
        .fold(0.0_f32, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(worst < 1e-4, "{worst}");
}

// ── Multigrid ───────────────────────────────────────────────────────────────

/// Votes assembled tile by tile from the interiors equal the whole-image votes,
/// and a tie goes to the earlier frame.
#[test]
fn depth_votes_from_tile_interiors_equal_the_whole_image_votes() {
    let (w, h) = (96_u16, 64_u16);
    let frames = bands(w, h, 4);
    let (wu, hu) = (usize::from(w), usize::from(h));
    let radius = StackingSettings::default().relief_estimation_radius as usize;
    let whole = multigrid_votes(&frames, radius);

    let mut assembled = DepthVotes::new(wu, hu);
    for cell in plan_tiles(wu, hu, 32, 10) {
        let p = &cell.padded;
        let tiles: Vec<PlanarImage<f32>> = frames
            .iter()
            .map(|f| stacker_core::memory::extract_tile(f, p.start_x, p.start_y, p.width, p.height))
            .collect();
        let local = multigrid_votes(&tiles, radius);
        let i = &cell.interior;
        assembled.merge_window(
            &local,
            cell.interior_offset,
            (i.start_x, i.start_y, i.width, i.height),
        );
    }
    assert_eq!(assembled.index(), whole.index());
    let worst = assembled
        .best()
        .iter()
        .zip(whole.best())
        .fold(0.0_f32, |acc, (a, b)| acc.max((a - b).abs()));
    assert!(worst < 1e-5, "{worst}");

    // A tie keeps the earlier frame; a strictly better value takes over.
    let mut votes = DepthVotes::new(2, 1);
    votes.fold_frame(0, &[1.0, 1.0]);
    votes.fold_frame(1, &[1.0, 2.0]);
    assert_eq!(votes.index(), [0, 1]);
    assert_eq!(votes.best(), [1.0, 2.0]);
}

#[test]
#[should_panic(expected = "MAX_MULTIGRID_FRAMES")]
fn the_frame_index_of_the_votes_is_a_u16() {
    assert_eq!(MAX_MULTIGRID_FRAMES, 65_536);
    DepthVotes::new(1, 1).fold_frame(MAX_MULTIGRID_FRAMES, &[0.0]);
}

/// With a threshold that keeps every pixel and no smoothing, every pixel is
/// pinned to its winning frame, so the multigrid result is the per-pixel winner.
#[test]
fn the_multigrid_with_every_pixel_pinned_is_the_per_pixel_winner() {
    let frames = bands(64, 48, 4);
    let settings = StackingSettings {
        relief_use_multigrid: true,
        relief_contrast_pct: 0.0,
        relief_smoothing_radius: 1,
        ..StackingSettings::default()
    };
    let fused = relief_multigrid(&frames, &settings);
    let maps = focus_maps(&frames, settings.relief_estimation_radius as usize);
    for i in 0..64 * 48 {
        let mut best = f32::NEG_INFINITY;
        let mut winner = 0;
        for (k, sml) in maps.smls.iter().enumerate() {
            if sml[i] > best {
                best = sml[i];
                winner = k;
            }
        }
        assert!(
            (fused.luma[i] - frames[winner].luma[i]).abs() < 1e-6,
            "pixel {i}: frame {winner}"
        );
        assert!((fused.chroma_a[i] - frames[winner].chroma_a[i]).abs() < 1e-6);
    }
}

/// `relief_use_multigrid` switches the engine of `relief`; the guided path is the
/// default and does not depend on the flag's code.
#[test]
fn relief_dispatches_on_the_multigrid_flag() {
    let frames = bands(64, 48, 3);
    let guided = StackingSettings {
        relief_contrast_pct: 0.4,
        ..StackingSettings::default()
    };
    let multigrid = StackingSettings {
        relief_use_multigrid: true,
        ..guided.clone()
    };
    assert!(
        !guided.relief_use_multigrid,
        "the guided filter is the default"
    );
    let a = relief(&frames, &guided);
    let b = relief(&frames, &multigrid);
    assert!(same_planes(&b, &relief_multigrid(&frames, &multigrid)));
    assert!(!same_planes(&a, &b), "the flag selects another engine");
    assert!(b.luma.iter().all(|v| v.is_finite()));
    assert_eq!(relief_multigrid(&[], &multigrid).width, 0);
}
