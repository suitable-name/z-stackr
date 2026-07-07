# z-stackr-pipeline

The shared out-of-core, tiled focus-stacking pipeline for **z-stackr** (crate identifier `stacker_pipeline`).

This crate exists so `z-stackr-cli` and `z-stackr-gui` can never silently diverge on the tiled processing path: the CLI always runs through [`run_pipeline`], and the GUI switches to it whenever its `tile_size` setting is greater than `0`. It owns the full end-to-end orchestration; the algorithms themselves live in `z-stackr-align` and `z-stackr-algo`.

## What it does

0. **RAW decode-once cache** (`cache::FrameCache`, only when at least one input is a RAW file and the crate is built with the `raw` feature) — every RAW frame is demosaiced exactly once, one at a time (bounded memory), into a binary blob cache under the run's temp directory, before any of the steps below run. Every frame-load site below then reads from that cache first (falling through to `stacker_core::io::load_frame` for cache misses and every non-RAW frame), so a RAW frame is never demosaiced 2-3× just because the pipeline naturally reloads frames across its alignment stages. Standard (non-RAW) stacks skip this step entirely — zero added overhead. See `src/cache.rs`'s module docs for the full rationale, including why the cache stores frames *before* preprocessing (preprocessing settings are identical at every load site within one run, so caching post-decode keeps preprocessing semantics byte-for-byte unchanged) and why no cross-run staleness handling is needed (the cache is ephemeral, created and destroyed with each run's temp directory).
1. **Load & preprocess** — frames are loaded one at a time (never the whole stack) via the decode-once cache described above, run through the shared `stacker_core::preprocessing` transforms, and converted to the pipeline's gamma-encoded planar YCbCr representation.
2. **Alignment pre-pass** — sequential chain alignment via the shared `stacker_align::align_frame` dispatch (optional parallel AKAZE seeding with the `akaze` feature), with each aligned frame sliced into apron-padded tiles and committed to a file-backed tile store (`stacker_core::memory::TileManager`) before the next frame is loaded. Peak RAM during alignment is a single frame. When `settings.alignment_mode` is `Neural` (requires the `nn` feature, experimental), this step is replaced entirely: `align::compute_neural_alignment` loads the whole stack once and runs a single batch inference call through a trained `batchalign-v2` model (internally downscaled to 512px-longest-side by `stacker_nn::bridge::align_planar`), producing one direct-use matrix per frame with no classical refinement on top — see [`PipelineParams::align_model`]'s docs. This trades the usual one-frame-at-a-time memory bound for a single up-front pass over the whole (still full-resolution, pre-downscale) stack.
3. **Common-coverage crop** — the per-frame coverage masks are intersected and resolved (guard-railed) into the crop rectangle used by the fusion pass and the saved output (`crop_to_common_area`, optionally restretched to the original canvas via `resize_cropped_to_original`).
4. **Tiled fusion** — per output tile: fetch every frame's padded crop, fuse (`Apex` uses a whole-image incremental accumulator instead; `Relief` runs a global-threshold pre-pass first so tile-local statistics can't cause threshold seams), and write the interior directly into the output buffer. Peak RAM scales with tile area, not image area.
5. **Encode & save** — 8/16-bit PNG/TIFF/JPEG, optional EXIF copy-through.

## Optional GPU acceleration (`gpu` feature, experimental)

This crate's `gpu` feature is a pure forward: `gpu = ["z-stackr-algo/gpu", "z-stackr-align/gpu"]`. It engages an automatic, toggle-free `wgpu` compute-shader path for two kernels the tiled fusion loop above calls into — tiled/batch Apex fusion (step 4) and the production edge-clamped warp used during the alignment pre-pass (step 2) — with a transparent fallback to the existing CPU/SIMD implementations whenever no adapter is found or a GPU call fails. There is no pipeline setting for this: it is purely a compile-time/hardware-availability choice, and the CPU path remains the default and the reference implementation. Relief, Strata, and neural fusion/alignment are unaffected and always run on the CPU. See `crates/stacker-algo/README.md` / `crates/stacker-align/README.md`'s own GPU sections for the full scope and implementation status.

## Memory model

Fusion memory ≈ `n_frames × tile_area × 3 channels × 4 bytes` — for 100 frames at 512 px tiles roughly 300 MB regardless of sensor resolution. The only full-resolution allocation is the output buffer at its final bit depth. See [`run_pipeline`]'s documentation for the complete accounting, the apron strategy, and the Relief global-threshold pre-pass rationale.

## Usage

This crate is consumed through [`PipelineParams`] + [`run_pipeline`] with a `stacker_core::settings::StackingSettings`; progress is reported through the [`PipelineProgress`] callback enum. See `z-stackr-cli` for a complete, minimal integration.

```toml
[dependencies]
z-stackr-pipeline = { version = "1.0", features = ["akaze"] }
```

Features: `akaze` (feature-match seeding for alignment), `nn` (neural fusion mode, CPU), `nn-gpu` (adds the wgpu/Vulkan backend), `raw` (pure-Rust camera RAW decoding via `z-stackr-core/raw`, including CR3 — see that crate's README for the full format list), `gpu` (experimental automatic GPU acceleration for tiled Apex fusion + the production warp — see "Optional GPU acceleration" above; unrelated to `nn-gpu`, which is the neural-fusion inference backend).

> **Nightly required.** The workspace uses `#![feature(portable_simd)]`.
