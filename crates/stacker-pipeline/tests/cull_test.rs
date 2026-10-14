#![expect(
    clippy::suboptimal_flops,
    reason = "synthetic-image formulas read better without mul_add"
)]

//! `cull`: the out-of-core census and the in-memory decision give the same
//! answer (FLOW-1), and the decision rules themselves.

use stacker_algo::optimize::{MAX_CULL_THRESHOLD_PCT, optimize_stack};
use stacker_core::{image::PlanarImage, memory::MmapTileManager, settings::StackingSettings};
use stacker_pipeline::{
    cull::{
        CullRule, census_from_frames, census_from_store, census_from_winners, order_from_census,
        plan, plan_frames, plan_frames_with, plan_with,
    },
    tiling::{commit_frame, plan_tiles},
};

/// Frame `k` of `n` is sharp (high-frequency texture) in a vertical band and
/// smooth elsewhere, so the frames win different parts of the image. Frames in
/// `blurred` are smooth everywhere (they should be culled).
fn stack(w: u16, h: u16, n: u16, blurred: &[u16]) -> Vec<PlanarImage<f32>> {
    (0..n)
        .map(|k| {
            let mut img = PlanarImage::new(usize::from(w), usize::from(h));
            let band = f32::from(w) / f32::from(n);
            for y in 0..h {
                for x in 0..w {
                    let (fx, fy) = (f32::from(x), f32::from(y));
                    let in_band = fx >= band * f32::from(k) && fx < band * f32::from(k + 1);
                    let smooth = 0.5 + 0.3 * (0.02 * fx + 0.03 * fy).sin();
                    // Deterministic per-frame noise: strong focus response at
                    // every pixel of the band.
                    let hash = (u32::from(x).wrapping_mul(73_856_093)
                        ^ u32::from(y).wrapping_mul(19_349_663)
                        ^ u32::from(k + 1).wrapping_mul(83_492_791))
                        % 1000;
                    let noise = f32::from(u16::try_from(hash).unwrap()) / 1000.0 - 0.5;
                    let texture = if in_band && !blurred.contains(&k) {
                        0.5 * noise
                    } else {
                        0.0
                    };
                    img.luma[usize::from(y) * usize::from(w) + usize::from(x)] = smooth + texture;
                }
            }
            img
        })
        .collect()
}

fn store_with(
    frames: &[PlanarImage<f32>],
    tile: usize,
    apron: usize,
) -> (
    MmapTileManager,
    Vec<stacker_pipeline::tiling::TilePlan>,
    std::path::PathBuf,
) {
    let dir = std::env::temp_dir().join(format!(
        "stacker_pipeline_cull_{tile}_{apron}_{}_{}",
        frames.len(),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut store = MmapTileManager::new(dir.clone());
    let (w, h) = (frames[0].width, frames[0].height);
    let cells = plan_tiles(w, h, tile, apron);
    for (i, frame) in frames.iter().enumerate() {
        commit_frame(&mut store, i, frame, &cells).unwrap();
    }
    (store, cells, dir)
}

/// The CLI's tile-by-tile census and the desktop app's in-memory
/// `optimize_stack` keep the same frames in the same order.
#[test]
fn the_out_of_core_census_decides_like_the_in_memory_one() {
    let frames = stack(96, 64, 6, &[2, 4]);
    let (w, h) = (96, 64);
    let (store, cells, dir) = store_with(&frames, 32, 10);

    let census = census_from_store(&store, &cells, frames.len(), w, h).unwrap();
    let out_of_core = plan(&census, w, h, MAX_CULL_THRESHOLD_PCT);

    let in_memory = plan_frames(&frames, MAX_CULL_THRESHOLD_PCT);
    let reference = optimize_stack(&frames, MAX_CULL_THRESHOLD_PCT);

    assert_eq!(in_memory.order, reference.recommended_order);
    assert_eq!(
        out_of_core.order, in_memory.order,
        "same frames, same order"
    );
    assert_eq!(out_of_core.dropped, in_memory.dropped);
    assert!(
        out_of_core.dropped.contains(&2) && out_of_core.dropped.contains(&4),
        "the blurred frames are culled: {:?}",
        out_of_core.dropped
    );
    assert_eq!(out_of_core.warnings, in_memory.warnings);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Many frames: the threshold scales with the frame count (`50 / N`), so an
/// evenly spread deep stack keeps every frame instead of being emptied by a flat
/// 2 % (FUS-1), and the census path applies the same cap as `optimize_stack`.
#[test]
fn deep_stacks_use_the_scaled_threshold_in_both_paths() {
    let frames = stack(120, 24, 60, &[]);
    let (w, h) = (120, 24);
    let (store, cells, dir) = store_with(&frames, 40, 6);

    let census = census_from_store(&store, &cells, frames.len(), w, h).unwrap();
    let out_of_core = plan(&census, w, h, MAX_CULL_THRESHOLD_PCT);
    let in_memory = plan_frames(&frames, MAX_CULL_THRESHOLD_PCT);

    assert!((out_of_core.effective_pct - 50.0 / 60.0).abs() < 1e-5);
    assert!((in_memory.effective_pct - out_of_core.effective_pct).abs() < 1e-6);
    assert_eq!(out_of_core.order, in_memory.order);
    assert_eq!(
        out_of_core.order.len(),
        60,
        "an even deep stack keeps all frames"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn nearest_neighbour_order_starts_at_the_leftmost_centroid() {
    // Three frames each win two pixels of a 6-pixel row; frame 2 wins the left.
    let best_sml = vec![1.0_f32; 6];
    let best_idx = vec![2_u32, 2, 0, 0, 1, 1];
    let census = census_from_winners(&best_sml, &best_idx, 6, 3);
    assert_eq!(census.win_counts, [2, 2, 2]);
    assert_eq!(census.detail_pixels, 6);
    assert_eq!(order_from_census(&census, 6, 1, 2.0), [2, 0, 1]);
}

#[test]
fn frames_below_the_threshold_are_culled_and_flat_pixels_ignored() {
    // 100 detail pixels: frame 0 wins 99, frame 1 wins 1; 50 flat pixels (below
    // the 1 % floor) do not count.
    let mut best_sml = vec![1.0_f32; 150];
    let mut best_idx = vec![0_u32; 150];
    best_idx[99] = 1;
    for v in &mut best_sml[100..] {
        *v = 0.001;
    }
    let census = census_from_winners(&best_sml, &best_idx, 150, 2);
    assert_eq!(census.detail_pixels, 100);
    assert_eq!(census.win_counts, [99, 1]);
    assert_eq!(order_from_census(&census, 150, 1, 5.0), [0]);
    assert_eq!(order_from_census(&census, 150, 1, 0.5).len(), 2);
}

/// No detail at all, or a stack that would be emptied: everything is kept, and
/// a warning says so.
#[test]
fn degenerate_censuses_keep_every_frame_with_a_warning() {
    let flat = census_from_winners(&[0.0; 8], &[0; 8], 8, 3);
    let decision = plan(&flat, 8, 1, MAX_CULL_THRESHOLD_PCT);
    assert_eq!(decision.order, [0, 1, 2]);
    assert_eq!(decision.dropped, Vec::<usize>::new());
    assert!(
        decision
            .warnings
            .iter()
            .any(|w| w.contains("No frame shows any in-focus detail")),
        "{:?}",
        decision.warnings
    );

    // A requested threshold of 90 % would empty the stack; it is capped at
    // `min(2, 50/N)` = 2 % for four frames, and each frame wins 25 %.
    let census = census_from_winners(&[1.0; 4], &[0, 1, 2, 3], 4, 4);
    let decision = plan(&census, 4, 1, 90.0);
    assert_eq!(decision.order.len(), 4);
    assert!((decision.effective_pct - 2.0).abs() < 1e-6);
}

// ── The cull rule (auto_cull_threshold_pct) ──────────────────────────────────

/// 0 keeps the frame-count-scaled rule; a positive value is a fixed percentage.
#[test]
fn the_rule_comes_from_auto_cull_threshold_pct() {
    let rule = |pct: f32| {
        CullRule::from_settings(&StackingSettings {
            auto_cull_threshold_pct: pct,
            ..StackingSettings::default()
        })
    };
    assert_eq!(rule(0.0), CullRule::Scaled(MAX_CULL_THRESHOLD_PCT));
    assert_eq!(rule(3.5), CullRule::Fixed(3.5));
    assert_eq!(rule(f32::NAN), CullRule::Scaled(MAX_CULL_THRESHOLD_PCT));

    // The scaled rule is capped by 50 / N; the fixed one is not.
    let scaled = CullRule::Scaled(2.0);
    assert!((scaled.effective_pct(10) - 2.0).abs() < 1e-6);
    assert!((scaled.effective_pct(100) - 0.5).abs() < 1e-6);
    assert!((CullRule::Fixed(7.0).effective_pct(100) - 7.0).abs() < 1e-6);
}

/// The census built from frames in memory decides like `optimize_stack`, and a
/// fixed percentage drops what the scaled rule keeps.
#[test]
fn a_fixed_percentage_replaces_the_scaled_rule() {
    let frames = stack(96, 64, 6, &[2, 4]);
    let (w, h) = (96, 64);

    // The scaled rule through the census equals optimize_stack's decision.
    let census = census_from_frames(&frames);
    let scaled = plan_with(&census, w, h, CullRule::Scaled(MAX_CULL_THRESHOLD_PCT));
    let reference = plan_frames(&frames, MAX_CULL_THRESHOLD_PCT);
    assert_eq!(scaled.order, reference.order);
    assert_eq!(scaled.dropped, reference.dropped);
    assert_eq!(
        plan_frames_with(&frames, CullRule::Scaled(MAX_CULL_THRESHOLD_PCT)).order,
        reference.order
    );

    // The four sharp frames win about a quarter of the detail each: 20 % keeps
    // them, 30 % would drop all of them, so everything is kept with a warning.
    let fixed = plan_frames_with(&frames, CullRule::Fixed(20.0));
    assert_eq!(fixed.dropped, [2, 4]);
    assert!((fixed.effective_pct - 20.0).abs() < 1e-6);
    let too_high = plan_frames_with(&frames, CullRule::Fixed(30.0));
    assert_eq!(too_high.dropped, Vec::<usize>::new());
    assert!(
        too_high
            .warnings
            .iter()
            .any(|w| w.contains("would have dropped every frame")),
        "{:?}",
        too_high.warnings
    );

    // Fixed(0) keeps every frame, and an empty stack is not a panic.
    assert_eq!(
        plan_frames_with(&frames, CullRule::Fixed(0.0)).order.len(),
        6
    );
    assert_eq!(
        plan_frames_with(&[], CullRule::Fixed(5.0)).order,
        [] as [usize; 0]
    );
    assert_eq!(
        plan_frames_with(&[], CullRule::Scaled(2.0)).order,
        [] as [usize; 0]
    );
}

/// On a deep, evenly spread stack the scaled rule keeps every frame and a fixed 2 %
/// does not: 60 frames cannot all win 2 % of the detail, so at least one is
/// dropped, or (when none reaches it) everything is kept with the warning. That is
/// the point of the key.
#[test]
fn a_fixed_percentage_is_not_scaled_by_the_frame_count() {
    let frames = stack(120, 24, 60, &[]);
    let scaled = plan_frames_with(&frames, CullRule::Scaled(MAX_CULL_THRESHOLD_PCT));
    assert_eq!(scaled.order.len(), 60);
    assert!((scaled.effective_pct - 50.0 / 60.0).abs() < 1e-5);

    let fixed = plan_frames_with(&frames, CullRule::Fixed(2.0));
    assert!((fixed.effective_pct - 2.0).abs() < 1e-6);
    let warned = fixed
        .warnings
        .iter()
        .any(|w| w.contains("would have dropped every frame"));
    assert!(
        !fixed.dropped.is_empty() || warned,
        "a flat 2 % cannot keep all 60 frames: {:?}",
        fixed.warnings
    );
}

/// Stack order or census order of the survivors.
#[test]
fn the_fusion_order_is_stack_order_unless_sorting_is_asked_for() {
    let frames = stack(96, 64, 6, &[2, 4]);
    let decision = plan_frames_with(&frames, CullRule::Fixed(20.0));
    let census_order = decision.fusion_order(true);
    assert_eq!(census_order, decision.order);
    assert_eq!(decision.fusion_order(false), [0, 1, 3, 5]);
    let mut sorted = census_order;
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 3, 5], "the same frames either way");
}
