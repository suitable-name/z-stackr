use std::path::PathBuf;

use stacker_core::image::PlanarImage;

use stacker_algo::optimize::optimize_stack;

use crate::{align_cache::compute_align_fingerprint, settings::StackingSettings};

// ── Sort/Cull cache ──────────────────────────────────────────────────────────

/// Cached result of a standalone Sort and/or Cull pass.
///
/// The Sort ("sort by sharpness") and Cull ("auto-cull") buttons score the
/// exact same **aligned, common-area-cropped, `stack_every_nth`-subsampled**
/// frame set that the Stack handler scores at its own "Auto-cull stage" (see
/// `on_request_stack`) — never the raw/unaligned loaded frames — so a button
/// press and a subsequent Stack run are guaranteed to agree. Both call
/// [`run_sort_cull`], the single shared implementation used by both the
/// buttons and the Stack handler's own "Auto-cull stage".
///
/// `order` and `culled_paths` are expressed as **file paths** rather than
/// indices: indices into the aligned/subsampled frame set do not survive a
/// later manual reorder/add/remove of `file_paths` (the fingerprint would
/// simply go stale and force a recompute in that case), but storing paths
/// lets the cache be applied to `file_paths` directly and cheaply once
/// validated.
pub struct SortCullCache {
    /// Fingerprint of the exact inputs and settings that produced this
    /// result — see [`compute_sort_cull_fingerprint`].
    pub fingerprint: u64,
    /// Full recommended order (kept frames only, sharpness-sorted when
    /// `ran_sort`; otherwise the original relative order of the kept
    /// frames), as absolute file paths.
    pub order: Vec<PathBuf>,
    /// Frames culled by this pass, as absolute file paths. Empty when
    /// `ran_cull` is `false`.
    pub culled: Vec<PathBuf>,
    /// Whether this cache entry reflects a Sort pass (order is
    /// sharpness-recommended rather than merely "kept, original order").
    pub ran_sort: bool,
    /// Whether this cache entry reflects a Cull pass (`culled` may be
    /// non-empty; frames not in `culled` survived the win-rate threshold).
    pub ran_cull: bool,
}

/// Change-detection fingerprint for the Sort/Cull stage.
///
/// Sort and Cull run on the *aligned* frame set, so this fingerprint is a
/// superset of [`compute_align_fingerprint`]: it starts from that same value
/// (covering input identity — path, size, mtime — plus every
/// alignment/preprocessing setting that changes pixel content) and folds in
/// the additional settings that affect Sort/Cull specifically:
///
/// * `stack_every_nth` — Sort/Cull score the post-subsampling frame set,
///   exactly like the Stack handler does (see `on_request_stack`'s
///   "Stack-every-Nth subsampling" section, which runs *before* the
///   "Auto-cull stage").
/// * `crop_to_common_area` / `resize_cropped_to_original` — both change the
///   pixel content handed to `optimize_stack` (crop) or the canvas the
///   result is later resized to (irrelevant to scoring itself, but included
///   for completeness/future-proofing since it travels with the crop
///   decision).
/// * `auto_cull_threshold_pct` — directly changes which frames Cull drops.
///
/// `sort_by_sharpness` and `auto_cull` (the Stack-run toggles) are
/// deliberately **not** hashed here. The Sort and Cull buttons are
/// independent commands that never read those toggles (see
/// `button_run_flags`), so folding them into this fingerprint would
/// invalidate a button-produced cache purely because the user flipped a
/// Stack setting that button press never consulted — exactly the coupling
/// this fingerprint must not reintroduce. Op *coverage* (whether a cache
/// actually ran the op a given run wants) is already handled correctly and
/// separately by [`decide_sort_cull`]'s `Fresh(ops)` vs `wants` comparison,
/// which was the toggles' only legitimate job in this hash. One accepted,
/// documented simplification falls out of this: a sort-only cache is still
/// invalidated by an `auto_cull_threshold_pct` change even though that
/// setting can't affect a sort-only result — kept in the hash anyway
/// because it's cheap, rare to hit in practice, and correct to over-
/// invalidate rather than risk under-invalidating a cache that *did* cull.
///
/// A cache entry whose fingerprint matches the current inputs/settings is
/// known-equivalent to what a fresh Sort/Cull pass would produce right now.
#[must_use]
pub fn compute_sort_cull_fingerprint(paths_work: &[PathBuf], settings: &StackingSettings) -> u64 {
    use std::hash::{Hash, Hasher};
    // Reuse the align fingerprint as a base so Sort/Cull automatically
    // invalidate on everything that already invalidates alignment (including
    // drag/manual reordering of `paths_work`, since `compute_align_fingerprint`
    // hashes the list in order).
    let base = compute_align_fingerprint(paths_work, settings);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    base.hash(&mut hasher);
    settings.stack_every_nth.hash(&mut hasher);
    settings.crop_to_common_area.hash(&mut hasher);
    settings.resize_cropped_to_original.hash(&mut hasher);
    settings.auto_cull_threshold_pct.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// Result of applying [`run_sort_cull`] to a frame set: the kept frames in
/// their final order, plus which original positions were culled.
pub struct SortCullOutcome {
    /// Indices into the input slice, in final order (sharpness-sorted when
    /// `sort_by_sharpness` was requested; otherwise original relative order
    /// of the kept frames).
    pub order: Vec<usize>,
    /// Indices into the input slice that were dropped by culling. Empty
    /// unless `auto_cull` was requested.
    pub culled: Vec<usize>,
}

/// The single shared Sort/Cull computation, extracted from the Stack
/// handler's "Auto-cull stage" so the Sort/Cull buttons and the Stack
/// pipeline can never diverge on the same inputs.
///
/// `planar_imgs` must already be the aligned, common-area-cropped,
/// `stack_every_nth`-subsampled frame set — i.e. exactly what
/// `on_request_stack` hands to `optimize_stack` today. This function does
/// not itself align, crop, or subsample; callers (button handlers and the
/// Stack handler) are responsible for producing that frame set identically
/// (see `prepare_aligned_frame_set`).
///
/// Behaviour:
/// * neither toggle set → identity order, nothing culled.
/// * `sort_by_sharpness` only → `optimize_stack`'s `recommended_order`
///   (threshold `0.0`, so nothing is culled).
/// * `auto_cull` only → `optimize_stack`'s `kept_indices` (original relative
///   order preserved).
/// * both → `recommended_order`, which is already computed only over the
///   surviving (post-threshold) frames.
#[must_use]
pub fn run_sort_cull(
    planar_imgs: &[PlanarImage<f32>],
    sort_by_sharpness: bool,
    auto_cull: bool,
    auto_cull_threshold_pct: f32,
) -> SortCullOutcome {
    if !sort_by_sharpness && !auto_cull {
        return SortCullOutcome {
            order: (0..planar_imgs.len()).collect(),
            culled: Vec::new(),
        };
    }
    let threshold = if auto_cull {
        auto_cull_threshold_pct
    } else {
        0.0
    };
    let opt = optimize_stack(planar_imgs, threshold);
    let order = if sort_by_sharpness {
        opt.recommended_order
    } else {
        opt.kept_indices
    };
    let culled: Vec<usize> = (0..planar_imgs.len())
        .filter(|i| !order.contains(i))
        .collect();
    SortCullOutcome { order, culled }
}

/// Which standalone button the user just pressed — see [`button_run_flags`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortCullOpPressed {
    /// The **Sort** button.
    Sort,
    /// The **Cull** button.
    Cull,
}

/// Decide which flags a Sort/Cull **button** press should run with.
///
/// The Sort and Cull buttons are independent commands: Sort always runs the
/// sharpness sort and never reads `settings.sort_by_sharpness`/`auto_cull`,
/// and Cull always runs the cull (at the threshold slider's current value)
/// and never reads those toggles either — that's the whole point of "fully
/// independent". The only question this function answers is whether the
/// *other* op should also be preserved, and the answer comes exclusively
/// from whether a still-fresh cache already covers it: pressing Sort after a
/// Cull (with nothing else having changed since) keeps the culled set
/// applied — re-sorting the survivors rather than silently discarding the
/// cull — and mirrored for pressing Cull after a Sort. A stale or absent
/// cache carries no such guarantee, so the pressed op runs alone.
///
/// Returns `(sort, cull)` — the exact flags to pass to [`run_sort_cull`].
#[must_use]
pub const fn button_run_flags(
    op_pressed: SortCullOpPressed,
    cache_state: SortCullCacheState,
) -> (bool, bool) {
    let ops = match cache_state {
        SortCullCacheState::Fresh(ops) => ops,
        SortCullCacheState::Absent | SortCullCacheState::Stale => SortCullCacheOps {
            ran_sort: false,
            ran_cull: false,
        },
    };
    match op_pressed {
        SortCullOpPressed::Sort => (true, ops.ran_cull),
        SortCullOpPressed::Cull => (ops.ran_sort, true),
    }
}

/// Cross-thread context for the "Sort/Cull cache is stale" confirmation popup.
///
/// Mirrors [`crate::stacking::ReliefPreviewContext`]'s established pattern
/// exactly: the Stack background thread parks on `reply_rx.recv()` after
/// publishing this into the shared `Arc<Mutex<Option<..>>>` slot and
/// flipping the dialog's `*-open` property from the UI thread via
/// `slint::invoke_from_event_loop`; the two dialog button callbacks
/// (`on_sort_cull_stale_run_again` / `on_sort_cull_stale_keep_as_is`) take
/// the context back out of the slot and send the user's choice down
/// `reply_tx`, which un-parks the background thread. This keeps the popup
/// off the UI thread's critical path (Slint keeps rendering/handling input
/// normally; only the background Stack thread blocks) — the same
/// non-blocking-UI property `ReliefPreviewContext` already relies on.
pub struct SortCullPromptContext {
    /// Sends the user's choice back to the parked Stack thread: `true` for
    /// "Run again" (recompute Sort/Cull during this Stack run), `false` for
    /// "Keep as is" (proceed with the current list order/selection,
    /// discarding the stale cache).
    pub reply_tx: std::sync::mpsc::Sender<bool>,
}

/// Three-way decision for whether pressing **Stack** should skip Sort/Cull
/// recomputation, run it inline as before, or pause and ask the user.
///
/// Kept as a small pure function (inputs = cache-validity facts, output =
/// the decision) precisely so it's unit-testable without spinning up any
/// background thread, file I/O, or Slint state — see `tests/sort_cull_test.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortCullDecision {
    /// No cache exists for either op that this Stack run needs — compute
    /// during the stack, exactly like the pre-cache behaviour.
    Compute,
    /// A cache exists and its fingerprint matches the current inputs and
    /// settings — apply its `order`/`culled` and skip recomputation.
    Skip,
    /// A cache exists but its fingerprint no longer matches (inputs or
    /// settings changed since Sort/Cull last ran) — surface the
    /// confirmation popup instead of silently picking a side.
    Ask,
}

/// Which Sort/Cull ops a cache entry actually ran, for the [`Fresh`] cache
/// state — see [`SortCullCacheState`].
///
/// [`Fresh`]: SortCullCacheState::Fresh
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortCullCacheOps {
    /// Whether the cached pass ran Sort (sharpness ordering).
    pub ran_sort: bool,
    /// Whether the cached pass ran Cull (win-rate threshold dropping).
    pub ran_cull: bool,
}

/// State of the current [`SortCullCache`] (if any) relative to the
/// fingerprint a `Stack` run just computed for its own
/// (aligned/cropped/subsampled) frame set — see
/// [`compute_sort_cull_fingerprint`].
///
/// Replaces a `(cache_present, fingerprint_matches)` bool pair with a single
/// value that cannot express the meaningless "fingerprint matches but no
/// cache exists" combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortCullCacheState {
    /// No cache exists for either op.
    Absent,
    /// A cache exists but its fingerprint no longer matches the current
    /// inputs/settings.
    Stale,
    /// A cache exists and its fingerprint matches the current inputs and
    /// settings.
    Fresh(SortCullCacheOps),
}

/// This Stack run's `sort_by_sharpness` / `auto_cull` toggles, i.e. which
/// op(s) it needs Sort/Cull to have covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortCullWants {
    /// Whether this Stack run has `sort_by_sharpness` enabled.
    pub sort: bool,
    /// Whether this Stack run has `auto_cull` enabled.
    pub cull: bool,
}

/// Decide [`SortCullDecision`] for a `Stack` press.
///
/// # Partial-cache rule
///
/// A cache entry's `ran_sort`/`ran_cull` reflect the **union** of every op
/// that has actually run since the cache was last replaced: the Sort and
/// Cull buttons are independent commands (see `button_run_flags`), but each
/// one merges its result into the existing fresh cache rather than
/// overwriting it outright, so e.g. pressing Sort then Cull leaves a single
/// cache entry with both `ran_sort` and `ran_cull` true. A stale or absent
/// cache is simply replaced by whichever single op just ran. If the current
/// Stack run wants an op the cache didn't run (e.g. only Sort was ever
/// cached but Stack also wants Cull), that is treated the same as "no usable
/// cache" for the *missing* op — this function returns
/// [`SortCullDecision::Compute`] rather than [`SortCullDecision::Skip`], so
/// the missing op still gets computed at stack time. This keeps the rule
/// simple and always correct (never silently skips work the cache didn't
/// actually do) at the cost of occasionally recomputing an op that was
/// already partially valid.
#[must_use]
pub const fn decide_sort_cull(cache: SortCullCacheState, wants: SortCullWants) -> SortCullDecision {
    let ops = match cache {
        SortCullCacheState::Absent => return SortCullDecision::Compute,
        SortCullCacheState::Stale => return SortCullDecision::Ask,
        SortCullCacheState::Fresh(ops) => ops,
    };
    // Partial-cache rule: every op this run wants must have actually been
    // run by the cached pass, or we fall back to computing fresh rather than
    // silently skipping an op the cache never performed.
    let sort_covered = !wants.sort || ops.ran_sort;
    let cull_covered = !wants.cull || ops.ran_cull;
    if sort_covered && cull_covered {
        SortCullDecision::Skip
    } else {
        SortCullDecision::Compute
    }
}

/// Apply a [`SortCullCache`] to `file_paths`.
///
/// Keeps only the paths present in `cache.order` (in that order). Paths in
/// `file_paths` that no longer exist in the cache's recorded set (e.g.
/// removed on disk between the button press and this Stack run — the
/// fingerprint match already rules out additions/removals/reorders, so this
/// is only a defensive fallback) are dropped silently rather than panicking
/// on a missing index.
///
/// Returns the new ordered path list to push into the GUI's source list, and
/// the culled paths (for marking) unchanged from the cache.
#[must_use]
pub fn apply_sort_cull_cache(cache: &SortCullCache) -> (Vec<PathBuf>, Vec<PathBuf>) {
    (cache.order.clone(), cache.culled.clone())
}
