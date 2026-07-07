#![allow(clippy::suboptimal_flops)] // colour-matrix coefficients; mul_add notation would harm readability

//! The single shared out-of-core, tiled focus-stacking pipeline used by both
//! `z-stackr-cli` and `z-stackr-gui`.
//!
//! This crate exists so the two apps can never silently diverge on the tiled
//! processing path: `z-stackr-cli` always runs through it, and
//! `z-stackr-gui` switches to it whenever `StackingSettings::tile_size > 0`
//! (its default in-RAM path — which supports Auto-Cull and full live
//! per-frame preview — is used when `tile_size == 0`). See [`run_pipeline`]
//! for the full memory model.

mod align;
mod cache;
mod fuse;
mod load;
pub mod output;

pub use fuse::FusionMode;
pub use load::collect_image_paths;

use image::{ImageBuffer, Rgb, RgbImage};
#[cfg(feature = "akaze")]
use rayon::prelude::*;
#[cfg(feature = "akaze")]
use stacker_align::akaze_match::extract_ref_features;
use stacker_core::{
    error::StackerError,
    memory::{APRON_PX, TileCoordinate, TileManager, TileProvider, enumerate_tiles, extract_tile},
};
use std::{collections::HashMap, path::PathBuf, time::Instant};

use align::align_frame;
#[cfg(feature = "akaze")]
use align::compute_akaze_coarse_seed;
#[cfg(feature = "nn")]
use fuse::{load_nn_model, nn_fuse_tile};
use fuse::{run_apex, run_relief, run_strata};
use load::dynamic_to_planar;
use output::{paste_tile_to_rgb8, paste_tile_to_rgb16, supports_16bit_output};
use stacker_algo::{
    apex::fuse::ApexAccumulator,
    relief::fuse::{auto_contrast_threshold, compute_relief_smls},
};

/// Frame paths and per-run parameters for [`run_pipeline`].
///
/// This is the single shared entry point both `z-stackr-cli` and
/// `z-stackr-gui` use for out-of-core tiled stacking. Keeping this parameter
/// struct (rather than each app's own CLI-arg or GUI-settings type) is what
/// lets the two apps call exactly the same code path instead of maintaining
/// separate copies that can silently drift apart.
#[derive(Debug, Clone)]
pub struct PipelineParams {
    /// Ordered list of frame file paths to stack. The CLI resolves this via
    /// [`collect_image_paths`] against its `--input-dir`; the GUI passes its
    /// own already-curated loaded-file list directly.
    pub paths: Vec<PathBuf>,
    /// Destination file for the stacked result. The format is inferred from
    /// the extension (`.png`/`.tif`/`.tiff` support 16-bit output; anything
    /// else, including `.jpg`/`.jpeg`, is written 8-bit).
    pub output_file: PathBuf,
    /// Fusion mode: `"apex"`, `"relief"`, or (with the `nn` feature) `"ai"`/`"nn"`.
    pub mode: String,
    /// Tile edge length in pixels for the out-of-core tiled fusion pass.
    /// Must be > 0 — the whole point of this pipeline is bounded per-tile
    /// memory, so a `0` tile size has no sensible interpretation here (unlike
    /// `StackingSettings::tile_size`'s GUI-only "0 = don't use this pipeline
    /// at all" sentinel, which the *caller* checks before ever constructing a
    /// `PipelineParams`).
    pub tile_size: usize,
    /// AI mode (`mode == "ai"`): name of the model to use (file stem in the
    /// models directory). `None` uses the first discovered model.
    pub model: Option<String>,
    /// AI mode: inference device, `"cpu"` or `"gpu"`. `None` uses the best
    /// available backend.
    pub device: Option<String>,
    /// Neural alignment mode (`settings.alignment_mode` set to
    /// [`stacker_core::settings::AlignmentModeSetting::Neural`], requires the
    /// `nn` feature): name of the alignment model to use (file stem in the
    /// models directory, filtered to `stacker_nn::ModelEntry::is_alignment`
    /// entries). `None` uses the first discovered alignment model. Mirrors
    /// `model` above, but kept as a separate field since a fusion checkpoint
    /// and an alignment checkpoint are never interchangeable — `mode == "ai"`
    /// and neural alignment mode can both be active in the same run (neural
    /// alignment followed by neural fusion), each picking its own model
    /// independently.
    pub align_model: Option<String>,
}

/// A progress event emitted by [`run_pipeline`].
///
/// Callers interpret this however suits their UI: the CLI renders
/// `indicatif` progress bars from it; the GUI updates its Slint
/// status/progress bar (and, unlike the CLI, has no per-tile preview image to
/// show — the tiled path never holds a full decoded result in memory, which
/// is the whole point of it).
#[derive(Debug, Clone, Copy)]
pub enum PipelineProgress {
    /// RAW-decode pre-pass: frame `current` (1-based) of `total` RAW frames
    /// has just been decoded and cached (see `cache::FrameCache`). Only
    /// emitted when at least one input path is a RAW file; standard
    /// (non-RAW) runs never see this variant, keeping their progress
    /// timeline byte-for-byte identical to before RAW support existed. Owns
    /// a small early slice of the overall "load" progress segment, ahead of
    /// [`PipelineProgress::AlignStart`].
    DecodeRaw { current: usize, total: usize },
    /// The alignment + tile-commit pre-pass is starting; `total` is the
    /// number of frames to process (including the reference frame).
    AlignStart { total: usize },
    /// Frame `current` (1-based) of `total` has just been aligned and
    /// committed.
    AlignFrame { current: usize, total: usize },
    /// The alignment pre-pass has finished.
    AlignDone,
    /// The tiled fusion pass is starting; `total` is the number of tiles.
    /// Not emitted in whole-image `Apex` mode, which fuses via a single
    /// incremental accumulator instead of per-tile I/O.
    FuseStart { total: usize },
    /// Tile `current` (1-based) of `total` has just been fused and written
    /// into the output buffer.
    FuseTile { current: usize, total: usize },
    /// The fusion pass has finished.
    FuseDone,
    /// Encoding and writing the final image to disk.
    Encoding,
}

/// Run the full focus-stacking pipeline in a **tiled, out-of-core** manner.
///
/// `on_progress` is called from the calling thread at each pipeline stage
/// boundary and per frame/tile — see [`PipelineProgress`]. Pass `|_| {}` to
/// ignore progress entirely.
///
/// ## Memory model
///
/// ### What is memory-bounded (scales with tile area, NOT total image area)
/// - **Fusion**: for each output tile we load one padded crop per input frame,
///   fuse them, then discard.  Peak RAM during fusion ≈ `n_frames × tile_area
///   × 3 channels × 4 bytes`.  For 100 frames at 512 px tiles that is ≈ 300 MB
///   regardless of image resolution.
///
/// ### What still loads the full frame (and why)
/// - **Alignment** (AKAZE + RANSAC): feature detection requires the full image
///   to find globally consistent correspondences.  Attempting to match on
///   individual tiles would produce incoherent, tile-local transforms with no
///   guaranteed cross-tile continuity.
///
///   *Mitigation*: each aligned frame is loaded once, warped, its tile crops
///   committed to the `TileManager`, and then **dropped immediately** before
///   the next frame is loaded.  Peak RAM during the sequential alignment pass
///   is a single aligned frame (≈ 3 × W × H × 4 bytes), not the entire stack.
///
///   *AKAZE seeding pre-pass* (only when `akaze_seeding = true` and the crate
///   is built with `--features akaze`): coarse seeds for every frame are
///   computed **in parallel** (via `rayon`) before the sequential pass above,
///   since each frame's match+RANSAC against the reference only depends on
///   that frame's own pixels, never on another frame's result. Each parallel
///   task loads, matches, and immediately drops one frame — peak RAM is
///   bounded by the thread-pool size (one frame per worker thread), not the
///   whole stack — at the cost of a second disk read per frame (once here,
///   once again in the sequential pass).
///
/// ### Output buffer
/// - The final `PlanarImage<f32>` accumulating tile results is W × H × 3 × 4
///   bytes (e.g. ≈ 50 MB for a 4096 × 4096 image).  This is unavoidable given
///   that tiles must be stitched into a coherent result before encoding to disk.
///
/// ## Apron strategy
///
/// Each input tile is fetched with an [`APRON_PX`]-pixel halo on every side
/// (clamped at image borders).  Fusing the oversized tile ensures that
/// neighbourhood operations (Laplacian pyramid levels, SML box-sums, guided
/// filter) see correct pixel context at all tile boundaries, eliminating seams.
/// Only the interior `(width × height)` region is copied into the output after
/// fusion.
///
/// ## `Relief` global-threshold pre-pass (double SML cost)
///
/// `Relief` mode additionally runs a pre-pass **before** the tiled fusion loop
/// (see `resolve_global_relief_threshold`) that computes the per-tile SML maps
/// a second time, purely to sample interior (apron-excluded) max-SML values
/// and resolve one global absolute contrast threshold shared by every tile.
/// This exists because resolving each tile's percentile/Otsu threshold from
/// its own local SML statistics would let neighbouring tiles' slightly
/// different distributions each pick a slightly different absolute
/// threshold — visible as tile-to-tile seams in the fused depth map wherever
/// the per-pixel mask flips differently across a tile boundary. This trades
/// roughly 2× the SML compute cost (SML is cheap relative to alignment and
/// tile I/O) for a seamless, globally-consistent threshold.
///
/// ## Common-coverage crop (`settings.crop_to_common_area`)
///
/// Focus breathing means a warped frame's edge-clamped border replicates
/// outer pixels into the band the warp doesn't actually cover; fusing that
/// smeared band bleeds it into the result. When `crop_to_common_area` is
/// `true` (default) and alignment actually ran (`n_frames > 1` and
/// `alignment_mode != None`), the accumulated `common_mask` is resolved to a
/// crop rectangle via `stacker_align::transform::resolve_common_crop`, which
/// also applies a rogue-frame guard: a misaligned/rogue frame that would
/// shrink the crop below 25% of the canvas area is rejected (logged as a
/// warning) and the pipeline falls back to the full canvas.
///
/// The two fusion families apply this crop differently:
/// - **Tiled (`Relief`/`AI`)**: the output buffer is allocated directly at the
///   cropped dimensions, and tiles whose interior rectangle doesn't
///   intersect the crop are skipped entirely (never fused, never fetched
///   from disk) — see `clip_tile_to_crop`. Intersecting tiles paste only
///   their overlap with the crop. The `Relief` global-threshold pre-pass
///   applies the same skip/clip so excluded pixels never influence the
///   global threshold statistics either.
/// - **Whole-image `Apex`**: the incremental accumulator is seeded from
///   frame 0 *before* any frame's coverage is known, so pre-fusion cropping
///   isn't possible without a second pass over the whole stack. This path
///   always fuses full-canvas and the crop (when active) is applied once at
///   the save step instead (step 6, via `crop_interleaved`).
///
/// `crop_to_common_area = false` always saves the full-canvas output for
/// both paths (edge-clamped warp data visible at the borders).
///
/// ## Restretch back to canvas resolution (`settings.resize_cropped_to_original`)
///
/// When `crop_to_common_area` cropped the output down, setting
/// `resize_cropped_to_original` additionally resamples the final cropped
/// buffer (both fusion families reach the same code path for this, at step
/// 6, after the tiled path's already-cropped buffer or `Apex`'s
/// `crop_interleaved` result is resolved) back up to the original
/// `img_w` × `img_h` canvas size via `image::imageops::resize` with
/// `FilterType::Lanczos3`, applied before the JPEG-encode branch so every
/// save path (16-bit TIFF/PNG, 8-bit TIFF/PNG, JPEG) gets the resized
/// buffer. Ignored when no crop shrank the buffer, or when
/// `crop_to_common_area = false`.
///
/// # Errors
/// Returns [`StackerError`] on I/O failures, unsupported mode strings, or
/// alignment failures propagated from image I/O.
///
/// # Panics
///
/// Panics if `params.paths` is empty after `settings`-driven filtering, or if
/// internal crop/reconstruct invariants the pipeline itself maintains are
/// violated (should never happen in practice).
#[allow(clippy::too_many_lines, clippy::unused_async)]
pub async fn run_pipeline<F>(
    params: &PipelineParams,
    settings: &stacker_core::settings::StackingSettings,
    mut on_progress: F,
) -> Result<(), StackerError>
where
    F: FnMut(PipelineProgress),
{
    // Engage/disengage the shared runtime GPU switch once, before any
    // stacking work starts, from the single entry point every interface
    // (CLI, GUI tiled path, Python `stack_files`/`batch_stack`) calls — see
    // `stacker_core::gpu::set_enabled`'s docs. A no-op in a default,
    // non-`gpu` build (the whole call disappears, since neither the
    // function nor `StackingSettings::use_gpu`'s value has any GPU code to
    // gate in that build).
    #[cfg(feature = "gpu")]
    stacker_core::gpu::set_enabled(settings.use_gpu);

    let mode: FusionMode = params.mode.parse()?;
    // Apex fuses whole-image via the incremental accumulator (peak memory
    // ~2 pyramids, independent of frame count); Relief/AI keep the tiled path.
    let apex_whole_image = matches!(mode, FusionMode::Apex);
    let apex_use_color = settings.use_all_color_channels;
    let apex_grit = settings.grit_suppression;

    if settings.auto_cull || settings.sort_by_sharpness {
        tracing::warn!(
            "auto_cull = true or sort_by_sharpness = true, but frame culling/sorting is not implemented in this out-of-core \
             pipeline (GUI in-RAM path only — see the `auto_cull` field doc comment in \
             stacker_core::settings). All frames will be stacked in loaded order."
        );
    }

    let t_total = Instant::now();
    let mut paths = params.paths.clone();
    if settings.preprocessing.sort_reverse {
        paths.reverse();
    }
    if settings.stack_every_nth > 1 {
        paths = paths
            .into_iter()
            .step_by(settings.stack_every_nth as usize)
            .collect();
    }
    let n_frames = paths.len();
    tracing::info!(n_frames, "found images");

    // ── 0. RAW decode-once pre-pass (only when at least one input is RAW) ───
    //
    // Standard (all-PNG/JPEG/TIFF) stacks skip this entirely — `frame_cache`
    // is built with an empty blob map and every `frame_cache.load(path)`
    // call below falls straight through to `stacker_core::io::load_frame`,
    // which is `image::open` under the hood for non-RAW extensions. Zero
    // added overhead for a non-RAW stack. See `cache`'s module docs for the
    // full rationale (bounded memory, why post-decode/pre-preprocessing
    // caching, why no staleness handling is needed).
    let temp_dir = std::env::temp_dir().join(format!(
        "stacker_tiles_{}_{}",
        std::process::id(),
        // cheap uniquifier in case of multiple concurrent runs
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos()),
    ));
    std::fs::create_dir_all(&temp_dir)?;

    let has_raw_input = paths.iter().any(|p| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(stacker_core::io::is_raw_extension)
    });
    let n_raw = if has_raw_input {
        paths
            .iter()
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(stacker_core::io::is_raw_extension)
            })
            .count()
    } else {
        0
    };
    let frame_cache = cache::FrameCache::build(&paths, &temp_dir, |current, total| {
        on_progress(PipelineProgress::DecodeRaw { current, total });
    })?;
    if has_raw_input {
        tracing::info!(n_raw, n_frames, "RAW decode-once pre-pass complete");
    }

    // ── 1. Load the reference frame to get image dimensions ─────────────────
    let t_load = Instant::now();
    let ref_dyn = frame_cache.load(&paths[0])?;
    let ref_dyn = stacker_core::preprocessing::preprocess_frame(ref_dyn, &settings.preprocessing);
    let ref_frame = dynamic_to_planar(&ref_dyn);
    let img_w = ref_frame.width;
    let img_h = ref_frame.height;

    let brightness_target = if settings.correct_brightness {
        Some(stacker_align::brightness::BrightnessTarget::new(&ref_frame))
    } else {
        None
    };

    drop(ref_dyn);
    tracing::info!(
        img_w,
        img_h,
        elapsed_ms = t_load.elapsed().as_millis(),
        "reference frame loaded"
    );

    // ── 2. Build tile grid ───────────────────────────────────────────────────
    let tile_size = params.tile_size;
    let tiles = enumerate_tiles(img_w, img_h, tile_size);
    let n_tiles = tiles.len();
    tracing::info!(
        "image {}×{}, tile_size={}, {} tiles, apron={} px",
        img_w,
        img_h,
        tile_size,
        n_tiles,
        APRON_PX
    );

    // ── 3. Alignment + tile-commit pre-pass ─────────────────────────────────
    //
    // For each frame: load → align → slice into padded tiles → commit to disk
    // → drop.  Only ONE aligned frame lives in RAM at a time.
    //
    // `temp_dir` was already created above (step 0) so the RAW decode-once
    // cache's blob files and this run's tile files share one ephemeral
    // per-run directory, cleaned up together at step 5.
    let mut store = TileManager {
        temp_dir: temp_dir.clone(),
        tiles: HashMap::new(),
    };

    tracing::info!("pre-pass: aligning and tiling {} frames", n_frames);
    let t_align = Instant::now();
    on_progress(PipelineProgress::AlignStart { total: n_frames });

    // Accumulator for the whole-image Apex path; `None` for Relief/AI.
    let mut apex_acc: Option<ApexAccumulator> = if apex_whole_image {
        Some(ApexAccumulator::new(&ref_frame, apex_use_color, apex_grit))
    } else {
        None
    };

    // Commit the reference frame (frame 0) tile-by-tile (tiled path only).
    if !apex_whole_image {
        for (ti, coord) in tiles.iter().enumerate() {
            let (px, py, pw, ph) = coord.padded_region(img_w, img_h);
            let padded = extract_tile(&ref_frame, px, py, pw, ph);
            // Store under a padded-tile coordinate so the fusion pass can fetch it.
            let padded_coord = TileCoordinate {
                start_x: px,
                start_y: py,
                width: pw,
                height: ph,
            };
            store.commit_tile(0, &padded_coord, padded)?;
            tracing::trace!("frame 0 tile {ti}/{n_tiles} committed");
        }
    }
    drop(ref_frame);
    on_progress(PipelineProgress::AlignFrame {
        current: 1,
        total: n_frames,
    });

    // Common-coverage mask: starts all-true (reference frame is identity → fully
    // valid).  Each warped non-reference frame ANDs its own coverage into this
    // mask so that after the loop it records which pixels are valid in every frame.
    let mut common_mask = vec![true; img_w * img_h];

    // Load and align frames 1..N.
    if n_frames > 1 {
        // Re-load reference once and extract its AKAZE features once — reused
        // for every target frame instead of re-extracting N times.
        let ref_for_align_dyn = frame_cache.load(&paths[0])?;
        let ref_for_align_dyn = stacker_core::preprocessing::preprocess_frame(
            ref_for_align_dyn,
            &settings.preprocessing,
        );
        let ref_for_align = dynamic_to_planar(&ref_for_align_dyn);
        // Reference AKAZE features are only needed when seeding is enabled.
        #[cfg(feature = "akaze")]
        let (ref_kps, ref_desc) = if settings.akaze_seeding {
            extract_ref_features(&ref_for_align)
        } else {
            (Vec::new(), Vec::new())
        };
        #[cfg(feature = "akaze")]
        tracing::debug!(ref_kps = ref_kps.len(), "reference keypoints extracted");

        // ── AKAZE pre-pass: coarse seeds for every frame, computed in parallel ──
        // Each frame's match+RANSAC against the fixed reference features above
        // only depends on that frame's own pixels, never on the sequential
        // warm-start chain below, so it can all run across every core up front
        // instead of being interleaved one-frame-at-a-time into the sequential
        // pass. Each task loads its own frame, matches, and drops the pixel
        // data immediately, keeping only the resulting `Matrix3`.
        #[cfg(feature = "akaze")]
        let coarse_seeds: Vec<Option<stacker_align::Matrix3<f32>>> = if settings.akaze_seeding {
            paths[1..]
                .par_iter()
                .enumerate()
                .map(|(offset, path)| {
                    let frame_idx = offset + 1;
                    let dyn_img = match frame_cache.load(path) {
                        Ok(img) => img,
                        Err(e) => {
                            tracing::warn!(
                                frame = frame_idx,
                                path = %path.display(),
                                error = %e,
                                "AKAZE pre-pass: failed to load frame; no coarse seed"
                            );
                            return None;
                        }
                    };
                    let dyn_img = stacker_core::preprocessing::preprocess_frame(
                        dyn_img,
                        &settings.preprocessing,
                    );
                    let frame = dynamic_to_planar(&dyn_img);
                    compute_akaze_coarse_seed(
                        frame_idx,
                        &frame,
                        &ref_kps,
                        &ref_desc,
                        settings.alignment_mode,
                    )
                })
                .collect()
        } else {
            Vec::new()
        };

        // ── Neural alignment pre-pass ────────────────────────────────────
        //
        // `AlignmentModeSetting::Neural` (requires the `nn` feature) is
        // dispatched entirely BEFORE the sequential per-frame loop below,
        // unlike the AKAZE coarse-seed pre-pass: the neural model needs the
        // whole stack (downscaled to 512px longest side internally by
        // `stacker_nn::bridge::align_planar`) resident at once to run one
        // batch inference call, rather than a per-frame independent
        // computation. See `align::compute_neural_alignment`'s docs for the
        // memory trade-off this makes against the pipeline's usual
        // one-frame-at-a-time bound.
        //
        // Neural mode has two behaviours, selected by
        // `settings.neural_refine_classically` (`docs/batchalign-v2-design.md`
        // §6, default `true`):
        //   * `true` (hybrid, default): the model's matrix is fed as a SEED
        //     into the same classical `align_frame` dispatch the AKAZE seed
        //     uses, per-frame in the sequential loop below — sanity-filtered
        //     (`is_sane_seed`) and gated against ending up worse than
        //     identity, exactly like an AKAZE seed. Passed as
        //     `AlignmentModeSetting::Registration` so it actually reaches
        //     `align_frame`'s classical-refinement branch (the `Neural` arm
        //     itself is a pass-through — see `stacker_align::pipeline::
        //     align_frame`'s docs).
        //   * `false`: DIRECT-MATRIX path — the model's output matrix is
        //     used as the frame's final transform with no classical
        //     refinement layered on top, for benchmarking the network in
        //     isolation.
        #[cfg(feature = "nn")]
        let neural_matrices: Option<Vec<stacker_align::Matrix3<f32>>> =
            if settings.alignment_mode == stacker_core::settings::AlignmentModeSetting::Neural {
                Some(align::compute_neural_alignment(
                    &paths,
                    &settings.preprocessing,
                    params.align_model.as_deref(),
                    params.device.as_deref(),
                    &frame_cache,
                )?)
            } else {
                None
            };

        // Sequential ("chain") alignment: each frame is registered to the
        // previously warped frame (a rolling reference) rather than to a fixed
        // frame 0, and warm-started from the previous frame's solved transform.
        // This tracks gradual focus-breathing across the stack. Skipped
        // entirely for Neural mode's per-frame matrix (the direct-matrix
        // path below uses `neural_matrices` instead), but the rolling-ref
        // bookkeeping is harmless dead state in that case.
        let mut rolling_ref = ref_for_align;
        let mut prev_matrix = stacker_align::Matrix3::identity();

        for (frame_idx, path) in paths.iter().enumerate().skip(1) {
            let t_frame_start = Instant::now();
            let dyn_img = frame_cache.load(path)?;
            let dyn_img =
                stacker_core::preprocessing::preprocess_frame(dyn_img, &settings.preprocessing);
            let frame = dynamic_to_planar(&dyn_img);
            drop(dyn_img);
            tracing::debug!(
                frame = frame_idx,
                elapsed_ms = t_frame_start.elapsed().as_millis(),
                "frame loaded"
            );

            // Use this frame's precomputed AKAZE coarse seed (from the
            // parallel pre-pass above) when available, falling back to the
            // previous frame's solved matrix — the same warm-start fallback
            // the pre-pass itself doesn't have access to, since it runs
            // outside the sequential chain.
            #[cfg(feature = "akaze")]
            let seed = coarse_seeds
                .get(frame_idx - 1)
                .copied()
                .flatten()
                .unwrap_or(prev_matrix);
            #[cfg(not(feature = "akaze"))]
            let seed = prev_matrix;

            #[cfg(feature = "nn")]
            let (final_matrix, aligned) = if let Some(matrices) = neural_matrices.as_ref() {
                let matrix = matrices.get(frame_idx).copied().unwrap_or_else(|| {
                    tracing::warn!(
                        frame = frame_idx,
                        "neural alignment produced no matrix for this frame index; falling back to identity"
                    );
                    stacker_align::Matrix3::identity()
                });
                if settings.neural_refine_classically {
                    // Hybrid path (§6, default): feed the neural matrix as a
                    // SEED into the same classical dispatch the AKAZE seed
                    // uses, against the rolling reference. `Registration`
                    // mode is used regardless of `settings.alignment_mode`
                    // (which is `Neural` here) so this actually reaches
                    // `align_frame`'s classical-refinement branch instead of
                    // its `Neural` pass-through arm.
                    align_frame(
                        frame_idx,
                        frame,
                        &rolling_ref,
                        matrix,
                        stacker_core::settings::AlignmentModeSetting::Registration,
                        settings.optimizer,
                        true, // always-bounded regardless of caller
                        brightness_target.as_ref(),
                    )?
                } else {
                    // Direct-matrix neural path: no classical refinement —
                    // the model's matrix is used as-is. Kept for
                    // benchmarking the network in isolation.
                    match stacker_align::transform::warp_image_clamped(&frame, &matrix) {
                        Ok(mut warped) => {
                            if let Some(target) = brightness_target.as_ref() {
                                stacker_align::brightness::apply_brightness_correction(
                                    &mut warped,
                                    target,
                                );
                            }
                            (matrix, warped)
                        }
                        Err(err) => {
                            tracing::warn!(
                                frame = frame_idx,
                                error = %err,
                                "neural-alignment warp failed; using the unwarped frame"
                            );
                            (stacker_align::Matrix3::identity(), frame)
                        }
                    }
                }
            } else {
                align_frame(
                    frame_idx,
                    frame,
                    &rolling_ref,
                    seed,
                    settings.alignment_mode,
                    settings.optimizer,
                    true, // always-bounded regardless of caller
                    brightness_target.as_ref(),
                )?
            };
            #[cfg(not(feature = "nn"))]
            let (final_matrix, aligned) = align_frame(
                frame_idx,
                frame,
                &rolling_ref,
                seed,
                settings.alignment_mode,
                settings.optimizer,
                true, // always-bounded regardless of caller
                brightness_target.as_ref(),
            )?;

            let t_tiles = Instant::now();
            if let Some(acc) = apex_acc.as_mut() {
                // Whole-image Apex: blend this aligned frame into the
                // accumulator and drop it; no tiles are committed.
                acc.blend(&aligned);
            } else {
                for coord in &tiles {
                    let (px, py, pw, ph) = coord.padded_region(img_w, img_h);
                    let padded = extract_tile(&aligned, px, py, pw, ph);
                    let padded_coord = TileCoordinate {
                        start_x: px,
                        start_y: py,
                        width: pw,
                        height: ph,
                    };
                    store.commit_tile(frame_idx, &padded_coord, padded)?;
                }
            }
            tracing::debug!(
                frame = frame_idx,
                elapsed_ms = t_tiles.elapsed().as_millis(),
                "frame fusion-prep done"
            );

            // Fold this frame's warp coverage into the common mask.
            stacker_align::transform::intersect_coverage(
                &mut common_mask,
                &stacker_align::transform::coverage_mask(&final_matrix, img_w, img_h),
            );

            tracing::debug!(
                frame = frame_idx,
                elapsed_ms = t_frame_start.elapsed().as_millis(),
                "frame total done"
            );

            // Roll the reference forward to this aligned frame and carry its
            // transform as the warm-start seed for the next frame.
            prev_matrix = final_matrix;
            rolling_ref = aligned;

            on_progress(PipelineProgress::AlignFrame {
                current: frame_idx + 1,
                total: n_frames,
            });
        }
    }
    on_progress(PipelineProgress::AlignDone);
    tracing::info!(
        elapsed_ms = t_align.elapsed().as_millis(),
        "alignment + tile-commit pre-pass complete"
    );

    // ── Resolve the common-coverage crop rect ───────────────────────────────
    //
    // Alignment must have actually run — a single frame, or `alignment_mode
    // == None`, leaves `common_mask` all-`true` (its identity seed), which
    // `resolve_common_crop` would (correctly) treat as "nothing to crop" —
    // but gating on `n_frames > 1 && alignment_mode != None` explicitly here
    // avoids even attempting the resolution (and its rogue-frame-guard log
    // line) when it can never do anything.
    let crop_rect: Option<(usize, usize, usize, usize)> = if settings.crop_to_common_area
        && n_frames > 1
        && settings.alignment_mode != stacker_core::settings::AlignmentModeSetting::None
    {
        let resolved = stacker_align::transform::resolve_common_crop(&common_mask, img_w, img_h);
        if resolved.is_none()
            && stacker_align::transform::largest_true_rectangle(&common_mask, img_w, img_h)
                .is_some()
        {
            tracing::warn!(
                "common-coverage crop rejected by the rogue-frame guard (covers < 25% of the \
                 canvas); falling back to full canvas"
            );
        }
        resolved
    } else {
        None
    };
    if let Some((cx, cy, cw, ch)) = crop_rect {
        tracing::info!(
            crop_x = cx,
            crop_y = cy,
            crop_w = cw,
            crop_h = ch,
            "cropping output to common-coverage rectangle"
        );
    } else {
        tracing::info!("no common-coverage crop applied; output covers the full canvas");
    }

    // ── 4. Tiled fusion pass ─────────────────────────────────────────────────
    //
    // For each output tile: load one padded crop per frame from disk, fuse,
    // then write the interior pixels DIRECTLY into the typed output buffer.
    //
    // ## Streaming memory profile
    //
    // The output buffer (`ImageBuffer<Rgb<u8/u16>>`) is allocated once here
    // at output bit-depth, not as a full f32 planar image.  Each fused tile
    // (`PlanarImage<f32>`) is converted and written into the output buffer
    // immediately, then dropped.  Peak RAM during this pass:
    //   - Output buffer:     W × H × (3 or 6) bytes  [unavoidable — final result]
    //   - Active tile:       tile_area × 3 × 4 bytes  [scales with tile, not image]
    //   - All input frames:  n_frames × tile_area × 3 × 4 bytes  [tile-sized]
    tracing::info!("fusion pass: {} tiles × {} frames", n_tiles, n_frames);
    let t_fuse = Instant::now();

    // Allocate the output image at the final bit depth.  This is the ONLY
    // full-resolution buffer; all subsequent writes go directly into it.
    //
    // Dimensions: the tiled (Relief/AI) path allocates directly at the CROPPED
    // dimensions when a crop is active, so tiles outside the crop are simply
    // never fused (see the `clip_tile_to_crop` skip below) instead of being
    // fused and then discarded at the save step. The whole-image Apex path
    // cannot do this — see the doc comment on the `apex_whole_image` branch
    // below — so it always allocates full-canvas and crops once at step 6.
    let (out_w, out_h) = if apex_whole_image {
        (img_w, img_h)
    } else {
        crop_rect.map_or((img_w, img_h), |(_, _, cw, ch)| (cw, ch))
    };

    let out_dims = (out_w as u32, out_h as u32);
    let use_16bit =
        supports_16bit_output(&params.output_file) && settings.image_saving.bit_depth == 16;

    // Allocate the typed output buffer at the correct bit depth.
    // Both variants are wrapped in Option so one loop body handles both paths.
    let mut out16: Option<ImageBuffer<Rgb<u16>, Vec<u16>>> = if use_16bit {
        Some(ImageBuffer::new(out_dims.0, out_dims.1))
    } else {
        None
    };
    let mut out8: Option<RgbImage> = if use_16bit {
        None
    } else {
        Some(RgbImage::new(out_dims.0, out_dims.1))
    };

    // Load the neural model once (AI mode only); the per-tile loop reuses it.
    #[cfg(feature = "nn")]
    let nn_model: Option<stacker_nn::LoadedModel> = if matches!(mode, FusionMode::Ai) {
        Some(load_nn_model(params)?)
    } else {
        None
    };

    if apex_whole_image {
        // Whole-image Apex path: reconstruct once from the accumulator and
        // write directly into the output buffer; no per-tile I/O needed.
        let fused_whole = apex_acc
            .take()
            .expect("Apex accumulator present in Apex mode")
            .reconstruct();
        if let Some(ref mut buf) = out16 {
            paste_tile_to_rgb16(buf, &fused_whole, (0, 0), 0, 0, img_w, img_h);
        } else if let Some(ref mut buf) = out8 {
            paste_tile_to_rgb8(buf, &fused_whole, (0, 0), 0, 0, img_w, img_h);
        }
        tracing::info!("whole-image Apex fusion complete");
    } else {
        // ── Relief global-threshold pre-pass ──────────────────────────────────
        //
        // `run_relief` (per-tile) used to resolve its contrast threshold
        // (percentile order-statistic, or Otsu) from TILE-LOCAL SML
        // statistics. Since every tile's SML distribution differs slightly,
        // each tile ended up with a different absolute threshold value, which
        // shows up as visible tile-to-tile seams in the fused output wherever
        // the mask flips differently across a tile boundary.
        //
        // To fix this we run the SML computation TWICE for Relief tiled runs:
        // once here, over every tile, to collect a bounded-memory sample of
        // interior (non-apron) max-SML values and resolve ONE global absolute
        // threshold from it; and once again inside the fusion loop below
        // (`run_relief`), which now receives that resolved value and skips its
        // own per-tile threshold resolution entirely. This doubles Relief SML
        // computation cost, but SML is cheap relative to alignment/I/O, and
        // global threshold consistency (no seams) is worth more than the
        // per-tile speed this throws away.
        let relief_abs_threshold = if matches!(mode, FusionMode::Relief) {
            resolve_global_relief_threshold(
                &store, &tiles, n_frames, img_w, img_h, settings, crop_rect,
            )?
        } else {
            None
        };

        // Tiles whose interior rectangle doesn't intersect the crop (when a
        // crop is active) are skipped entirely — they would only ever
        // contribute pixels that get discarded, so fusing them at all is
        // wasted compute. `FuseStart`'s `total` counts only the tiles that
        // will actually be fused, so progress reporting matches reality.
        let fuse_tiles: Vec<&TileCoordinate> = tiles
            .iter()
            .filter(|coord| {
                crop_rect.is_none_or(|(cx, cy, cw, ch)| {
                    clip_tile_to_crop(
                        coord.start_x,
                        coord.start_y,
                        coord.width,
                        coord.height,
                        (0, 0), // pad offset is irrelevant for the intersection test
                        cx,
                        cy,
                        cw,
                        ch,
                    )
                    .is_some()
                })
            })
            .collect();
        let n_fuse_tiles = fuse_tiles.len();
        if crop_rect.is_some() {
            tracing::info!(
                n_fuse_tiles,
                n_tiles,
                "crop active: {} of {} tiles intersect the crop and will be fused",
                n_fuse_tiles,
                n_tiles
            );
        }

        on_progress(PipelineProgress::FuseStart {
            total: n_fuse_tiles,
        });
        // Tiled path for Relief and AI: fetch per-tile crops from disk, fuse,
        // and write the interior directly into the output buffer.
        //
        // Async coordination wrapper — I/O is currently synchronous (std::fs
        // via TileManager), but the smol boundary is retained so the
        // fusion loop is executor-aware and yields between tiles.
        smol::block_on(async {
            for (ti, coord) in fuse_tiles.iter().enumerate() {
                let (px, py, pw, ph) = coord.padded_region(img_w, img_h);
                let padded_coord = TileCoordinate {
                    start_x: px,
                    start_y: py,
                    width: pw,
                    height: ph,
                };

                // Load all frames for this padded tile.
                let mut frame_tiles = Vec::with_capacity(n_frames);
                for frame_idx in 0..n_frames {
                    let ft = store.fetch_tile(frame_idx, &padded_coord)?;
                    frame_tiles.push(ft);
                }

                // Fuse the padded tile.
                let fused_padded = match mode {
                    FusionMode::Apex => run_apex(&frame_tiles, settings),
                    FusionMode::Relief => run_relief(&frame_tiles, settings, relief_abs_threshold),
                    FusionMode::Strata => run_strata(&frame_tiles, settings),
                    #[cfg(feature = "nn")]
                    FusionMode::Ai => nn_fuse_tile(
                        nn_model.as_ref().expect("nn model is loaded in AI mode"),
                        &frame_tiles,
                    )?,
                };
                drop(frame_tiles);

                // Write fused interior DIRECTLY into the output buffer — no
                // full f32 accumulator needed. When a crop is active, only
                // the intersection of this tile's interior with the crop
                // rectangle is pasted, at an offset adjusted for both the
                // padded-buffer interior offset and the crop's own origin
                // (see `clip_tile_to_crop`).
                let pad_off = coord.interior_offset_in_padded(img_w, img_h);
                let (paste_pad_off, paste_x, paste_y, paste_w, paste_h) = match crop_rect {
                    Some((cx, cy, cw, ch)) => {
                        let Some(clip) = clip_tile_to_crop(
                            coord.start_x,
                            coord.start_y,
                            coord.width,
                            coord.height,
                            pad_off,
                            cx,
                            cy,
                            cw,
                            ch,
                        ) else {
                            // Shouldn't happen — `fuse_tiles` was already
                            // filtered to intersecting tiles — but skip
                            // defensively rather than paste garbage.
                            tracing::trace!(
                                "tile {ti}/{n_fuse_tiles} has no crop overlap; skipped"
                            );
                            on_progress(PipelineProgress::FuseTile {
                                current: ti + 1,
                                total: n_fuse_tiles,
                            });
                            smol::future::yield_now().await;
                            continue;
                        };
                        (
                            clip.pad_off,
                            clip.dest_off.0,
                            clip.dest_off.1,
                            clip.width,
                            clip.height,
                        )
                    }
                    None => (
                        pad_off,
                        coord.start_x,
                        coord.start_y,
                        coord.width,
                        coord.height,
                    ),
                };
                if let Some(ref mut buf) = out16 {
                    paste_tile_to_rgb16(
                        buf,
                        &fused_padded,
                        paste_pad_off,
                        paste_x,
                        paste_y,
                        paste_w,
                        paste_h,
                    );
                } else if let Some(ref mut buf) = out8 {
                    paste_tile_to_rgb8(
                        buf,
                        &fused_padded,
                        paste_pad_off,
                        paste_x,
                        paste_y,
                        paste_w,
                        paste_h,
                    );
                }
                // fused_padded dropped here — tile memory reclaimed immediately.

                tracing::trace!("tile {ti}/{n_fuse_tiles} fused and written to output buffer");
                on_progress(PipelineProgress::FuseTile {
                    current: ti + 1,
                    total: n_fuse_tiles,
                });
                // Yield to the smol executor between tiles.
                smol::future::yield_now().await;
            }
            Ok::<(), StackerError>(())
        })?;
    }
    on_progress(PipelineProgress::FuseDone);

    tracing::info!(
        elapsed_ms = t_fuse.elapsed().as_millis(),
        "fusion pass complete"
    );

    // ── 5. Cleanup temp tiles ────────────────────────────────────────────────
    let _ = std::fs::remove_dir_all(&temp_dir);

    // ── 6. Crop to common valid area (Apex only) + save output ───────────────
    on_progress(PipelineProgress::Encoding);
    let t_encode = Instant::now();

    // The tiled (Relief/AI) path already allocated its output buffer at the
    // cropped dimensions (see the `out_dims` computation above) and skipped
    // fusing non-intersecting tiles entirely, so `out16`/`out8` are already
    // exactly `crop_rect`'s size there — nothing left to crop here.
    //
    // The whole-image Apex path is the one exception: its accumulator is
    // seeded from frame 0 before any frame's coverage mask is known, so
    // pre-fusion cropping isn't possible without a second full pass over the
    // stack. It always fuses full-canvas, and the crop (when active) is
    // applied here instead, via `crop_interleaved`, right before saving.
    //
    // `(cx, cy, cw, ch)` is the crop to apply to the buffer that was
    // actually allocated: for Apex that buffer is always full-canvas
    // (`img_w` × `img_h`), so `crop_rect` (in image-space coordinates) is
    // exactly right; for the tiled Relief/AI path the buffer was already
    // allocated at the cropped size (`out_w` × `out_h`, see the `out_dims`
    // computation above), so there is nothing left to crop — `(0, 0, out_w,
    // out_h)` always compares equal to the buffer's own full extent and
    // takes the uncropped `buf.save()` fast path below.
    let (cx, cy, cw, ch) = if apex_whole_image {
        crop_rect.unwrap_or((0, 0, img_w, img_h))
    } else {
        (0, 0, out_w, out_h)
    };
    // `buf_w` is the stride `crop_interleaved` must use — the width of the
    // buffer actually allocated above, which is `img_w` for Apex and
    // `out_w` for the tiled path (identical to `img_w` unless a crop was
    // applied there).
    let buf_w = if apex_whole_image { img_w } else { out_w };
    let buf_h = if apex_whole_image { img_h } else { out_h };
    if (cx, cy, cw, ch) == (0, 0, buf_w, buf_h) {
        tracing::info!(
            original_w = img_w,
            original_h = img_h,
            "saving full-canvas output"
        );
    } else {
        tracing::info!(
            original_w = img_w,
            original_h = img_h,
            crop_x = cx,
            crop_y = cy,
            crop_w = cw,
            crop_h = ch,
            "saving output cropped to the common-coverage rectangle"
        );
    }

    let output_ext = params
        .output_file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("?");
    let is_jpeg = matches!(output_ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg");

    // Optionally restretch a cropped result back up to the original canvas
    // resolution — see `resize_cropped_to_original`'s doc comment on
    // `StackingSettings` and the "Common-coverage crop" section above. Only
    // meaningful when a crop actually shrank the buffer below (img_w,
    // img_h); a no-op full-canvas save (or a same-size crop, which can't
    // happen given `resolve_common_crop`'s guard rails) leaves it alone.

    let resize_target: Option<(u32, u32)> = if settings.crop_to_common_area
        && settings.resize_cropped_to_original
    {
        Some((img_w as u32, img_h as u32)).filter(|&(rw, rh)| (rw, rh) != (cw as u32, ch as u32))
    } else {
        None
    };

    if let Some(buf) = out16 {
        let mut cropped_buf = if (cx, cy, cw, ch) == (0, 0, buf_w, buf_h) {
            buf
        } else {
            let cropped_pixels = crop_interleaved(buf.as_raw(), buf_w, 3, cx, cy, cw, ch);
            image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(
                cw as u32,
                ch as u32,
                cropped_pixels,
            )
            .expect("crop dimensions are always valid")
        };
        if let Some((rw, rh)) = resize_target {
            tracing::info!(
                crop_w = cropped_buf.width(),
                crop_h = cropped_buf.height(),
                original_w = rw,
                original_h = rh,
                "resizing cropped output back to original canvas resolution"
            );
            cropped_buf = image::imageops::resize(
                &cropped_buf,
                rw,
                rh,
                image::imageops::FilterType::Lanczos3,
            );
        }
        cropped_buf.save(&params.output_file)?;
    } else if let Some(buf) = out8 {
        let mut cropped = if (cx, cy, cw, ch) == (0, 0, buf_w, buf_h) {
            buf
        } else {
            let cropped_pixels = crop_interleaved(buf.as_raw(), buf_w, 3, cx, cy, cw, ch);
            image::RgbImage::from_raw(cw as u32, ch as u32, cropped_pixels)
                .expect("crop dimensions are always valid")
        };
        if let Some((rw, rh)) = resize_target {
            tracing::info!(
                crop_w = cropped.width(),
                crop_h = cropped.height(),
                original_w = rw,
                original_h = rh,
                "resizing cropped output back to original canvas resolution"
            );
            cropped =
                image::imageops::resize(&cropped, rw, rh, image::imageops::FilterType::Lanczos3);
        }
        if is_jpeg {
            let file = std::fs::File::create(&params.output_file)?;
            let mut writer = std::io::BufWriter::new(file);
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut writer,
                settings.image_saving.jpeg_quality.clamp(1, 100) as u8,
            );
            enc.encode(
                cropped.as_raw(),
                cropped.width(),
                cropped.height(),
                image::ColorType::Rgb8.into(),
            )?;
        } else {
            cropped.save(&params.output_file)?;
        }
    }
    tracing::info!(
        output_path = %params.output_file.display(),
        format = output_ext,
        encode_ms = t_encode.elapsed().as_millis(),
        total_ms  = t_total.elapsed().as_millis(),
        "pipeline complete"
    );

    if settings.image_saving.copy_metadata {
        match stacker_core::metadata::copy_metadata(&paths[0], &params.output_file) {
            Ok(true) => tracing::info!(
                source = %paths[0].display(),
                "copied EXIF metadata to output"
            ),
            Ok(false) => tracing::debug!(
                source = %paths[0].display(),
                "copy_metadata enabled, but source frame has no EXIF metadata; nothing to copy"
            ),
            Err(e) => tracing::warn!("copy_metadata enabled but failed: {e}"),
        }
    }

    Ok(())
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Resolve a single **global** absolute SML threshold for the `Relief` tiled
/// fusion pass, by sampling the interior (non-apron) max-SML values of every
/// tile before the fusion loop runs.
///
/// # Why interior-only sampling
///
/// Adjacent tiles' padded (apron-included) regions overlap by up to
/// `2 * APRON_PX`; sampling the padded region would double-count those
/// shared pixels and skew the resulting order statistic toward
/// apron-heavy areas. [`stacker_core::memory::TileCoordinate::interior_offset_in_padded`]
/// gives the top-left of the true (non-overlapping) interior region within
/// the padded tile, and `coord.width` / `coord.height` give its extent, so
/// indexing the padded max-SML buffer with that offset+extent yields exactly
/// the interior pixels — each image pixel contributes to the sample exactly
/// once across all tiles.
///
/// # Bounded memory
///
/// A deterministic decimation (`take every k-th interior value`) keeps the
/// total collected sample at or below ~4 million `f32` regardless of image
/// resolution — `k` is derived from `img_w * img_h` up front so the stride is
/// uniform across the whole image rather than being recomputed per tile
/// (which could otherwise bias the sample toward small tiles at image
/// edges).
///
/// Returns `None` when the resolved threshold would be a no-op (`pct == 0`,
/// matching the `generate_mask` all-`true` invariant) or when there is no
/// positive-valued sample to threshold against.
fn resolve_global_relief_threshold(
    store: &TileManager,
    tiles: &[TileCoordinate],
    n_frames: usize,
    img_w: usize,
    img_h: usize,
    settings: &stacker_core::settings::StackingSettings,
    crop_rect: Option<(usize, usize, usize, usize)>,
) -> Result<Option<f32>, StackerError> {
    const MAX_SAMPLE: usize = 4_000_000;

    let total_px = (img_w * img_h).max(1);
    // Stride so that, in expectation, sampling every pixel of the image once
    // would collect roughly MAX_SAMPLE values; always at least 1.
    let stride = (total_px / MAX_SAMPLE).max(1);

    let est_radius = settings.relief_estimation_radius as usize;

    let mut sample: Vec<f32> = Vec::new();
    for coord in tiles {
        // Skip tiles whose interior doesn't intersect the crop when a crop
        // is active — their pixels never reach the final output, so
        // including them in the global-threshold sample would let excluded
        // (and possibly smeared) pixels skew the statistic.
        let clip = crop_rect.map(|(cx, cy, cw, ch)| {
            clip_tile_to_crop(
                coord.start_x,
                coord.start_y,
                coord.width,
                coord.height,
                coord.interior_offset_in_padded(img_w, img_h),
                cx,
                cy,
                cw,
                ch,
            )
        });
        // `Some(None)` means a crop is active and this tile has no overlap.
        if matches!(clip, Some(None)) {
            continue;
        }

        let (px, py, pw, ph) = coord.padded_region(img_w, img_h);
        let padded_coord = TileCoordinate {
            start_x: px,
            start_y: py,
            width: pw,
            height: ph,
        };

        let mut frame_tiles = Vec::with_capacity(n_frames);
        for frame_idx in 0..n_frames {
            frame_tiles.push(store.fetch_tile(frame_idx, &padded_coord)?);
        }

        let (_, max_sml) = compute_relief_smls(&frame_tiles, est_radius);
        drop(frame_tiles);

        // Sample region within the padded max-SML buffer: either the full
        // interior (no crop, or `clip_rect` is `None` meaning "no crop
        // active"), or just the tile-vs-crop intersection (clip.pad_off /
        // width / height) so pixels outside the crop never enter the sample.
        let (sample_off_x, sample_off_y, sample_w, sample_h) = if let Some(Some(c)) = clip {
            (c.pad_off.0, c.pad_off.1, c.width, c.height)
        } else {
            let (ix, iy) = coord.interior_offset_in_padded(img_w, img_h);
            (ix, iy, coord.width, coord.height)
        };
        let padded_w = max_sml.width;

        let mut counter = 0usize;
        for row in 0..sample_h {
            let src_row = sample_off_y + row;
            let row_start = src_row * padded_w + sample_off_x;
            for col in 0..sample_w {
                if counter.is_multiple_of(stride) {
                    sample.push(max_sml.luma[row_start + col]);
                }
                counter += 1;
            }
        }
    }

    if sample.is_empty() {
        return Ok(None);
    }

    // Resolve ONE fraction (percentile) the same way the whole-image /
    // GUI in-RAM path would: either via log-domain Otsu on the sample, or
    // directly from `relief_contrast_pct`.
    let sample_img = stacker_core::image::PlanarImage {
        width: sample.len(),
        height: 1,
        luma: sample.clone(),
        chroma_a: vec![0.0; sample.len()],
        chroma_b: vec![0.0; sample.len()],
    };
    let pct = if settings.relief_auto_detect {
        auto_contrast_threshold(&sample_img)
    } else {
        settings.relief_contrast_pct
    };

    if pct <= 0.0 {
        // Mirrors `generate_mask`'s documented invariant: pct == 0 (or
        // below) always means "every pixel passes" — signal that via `None`
        // so `run_relief` falls back to its own (also all-true) resolution
        // instead of computing a spurious absolute value.
        tracing::info!(
            "Relief global threshold: contrast_pct <= 0, using all-true mask (no threshold)"
        );
        return Ok(None);
    }

    // Order statistic over the strictly-positive sample values only,
    // mirroring `generate_mask`'s non-zero-population semantics (FIX 3) so
    // the globally-resolved value has the same meaning as a tile-local one
    // would.
    let mut positive: Vec<f32> = sample.into_iter().filter(|&v| v > 0.0).collect();
    if positive.is_empty() {
        tracing::info!(
            "Relief global threshold: no positive SML sample values, using all-true mask"
        );
        return Ok(None);
    }

    let len = positive.len();
    let mut idx = (len as f32 * pct) as usize;
    if idx >= len {
        idx = len - 1;
    }
    let (_, kth, _) = positive.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = *kth;

    tracing::info!(
        threshold,
        pct,
        sample_len = len,
        "Relief global threshold resolved"
    );

    Ok(Some(threshold))
}

/// Result of clipping a tile's interior rectangle against the output crop
/// rectangle, in the coordinate systems each downstream consumer needs.
///
/// All fields describe the same physical pixels — the intersection of the
/// tile's interior region and the crop rectangle — expressed in three
/// different origins:
/// - `pad_off`: offset into the *padded* (apron-included) fused tile buffer.
/// - `dest_off`: offset into the cropped output buffer.
/// - `width` / `height`: the shared extent (identical in every coordinate
///   system — clipping never changes the size of the overlap, only where it
///   starts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileClip {
    pub pad_off: (usize, usize),
    pub dest_off: (usize, usize),
    pub width: usize,
    pub height: usize,
}

/// Clip a tile's interior rectangle `(tile_x, tile_y, tile_w, tile_h)`
/// against the output crop rectangle `(crop_x, crop_y, crop_w, crop_h)`.
///
/// (Image-space coordinates, i.e. `coord.start_{x,y}` / `coord.{width,height}`.)
/// Express the resulting overlap relative to the tile's *padded* buffer
/// (whose interior begins at `pad_off_in_tile` within that padded buffer —
/// i.e. `coord.interior_offset_in_padded(..)`) and relative to the cropped
/// output buffer's origin.
///
/// Returns `None` when the tile's interior does not intersect the crop
/// rectangle at all — callers should skip fusing/pasting that tile entirely.
///
/// This is a pure geometry function (no I/O, no image data) so the
/// tile-vs-crop intersection math can be unit-tested independently of the
/// rest of the tiled fusion pipeline.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn clip_tile_to_crop(
    tile_x: usize,
    tile_y: usize,
    tile_w: usize,
    tile_h: usize,
    pad_off_in_tile: (usize, usize),
    crop_x: usize,
    crop_y: usize,
    crop_w: usize,
    crop_h: usize,
) -> Option<TileClip> {
    // Intersection of [tile_x, tile_x+tile_w) with [crop_x, crop_x+crop_w),
    // and the equivalent for y — standard axis-aligned rectangle overlap.
    let ix0 = tile_x.max(crop_x);
    let iy0 = tile_y.max(crop_y);
    let ix1 = (tile_x + tile_w).min(crop_x + crop_w);
    let iy1 = (tile_y + tile_h).min(crop_y + crop_h);

    if ix0 >= ix1 || iy0 >= iy1 {
        return None;
    }

    let width = ix1 - ix0;
    let height = iy1 - iy0;

    // The overlap starts `(ix0 - tile_x, iy0 - tile_y)` pixels into the
    // tile's interior — add that delta to the interior's offset within the
    // padded buffer to land on the overlap's position in padded-buffer space.
    let (pad_base_x, pad_base_y) = pad_off_in_tile;
    let pad_off = (pad_base_x + (ix0 - tile_x), pad_base_y + (iy0 - tile_y));

    // The overlap starts at `(ix0, iy0)` in image space; subtracting the
    // crop's own origin gives its position in the cropped output buffer.
    let dest_off = (ix0 - crop_x, iy0 - crop_y);

    Some(TileClip {
        pad_off,
        dest_off,
        width,
        height,
    })
}

/// Crop an interleaved pixel buffer (stride = `full_w * channels`) to the
/// sub-rectangle `(cx, cy, cw, ch)`.  Returns a new `Vec<T>` of size
/// `cw * ch * channels`.
fn crop_interleaved<T: Copy>(
    src: &[T],
    full_w: usize,
    channels: usize,
    cx: usize,
    cy: usize,
    cw: usize,
    ch: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(cw * ch * channels);
    for row in 0..ch {
        let src_row = cy + row;
        let start = (src_row * full_w + cx) * channels;
        let end = start + cw * channels;
        out.extend_from_slice(&src[start..end]);
    }
    out
}
