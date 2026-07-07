# z-stackr-align

The precision alignment engine for **z-stackr** (crate identifier `stacker_align`).

`stacker-align` is responsible for registering misaligned macro images before they are fused. Because depth-of-field bracketed sequences involve changing the lens focal distance, subjects naturally experience magnification scaling (focus breathing) and minor lateral shifts, especially if shot handheld. This crate calculates and applies the exact homographic transformations necessary to bring all images into perfect subpixel alignment.

## Alignment Pipeline

Both `z-stackr-cli` and `z-stackr-gui` call through a **single shared dispatch function, `pipeline::align_frame`** — this is deliberate: alignment logic must never be duplicated or re-derived per app. `align_frame` takes a frame, a reference, a seed matrix, the selected `AlignmentModeSetting`, the selected `OptimizerSetting`, and an optional brightness target, and returns the solved matrix plus the warped, brightness-corrected frame.

The process is **intensity-based at its core** with an *optional* feature-matching seed:

### 1. Optional Coarse Seed (Feature Gated, off by default)
AKAZE feature matching is an **optional accelerator, never a requirement** — even when the `akaze` feature is compiled in, the consuming application decides per-run whether to use it. When enabled, the crate uses the [`akaze`](https://crates.io/crates/akaze) crate to extract keypoints and binary (M-LDB) descriptors, matches them between target and reference, and uses a robust **RANSAC** estimator to compute a coarse `Matrix3` seed. The RANSAC acceptance is deliberately strict: a winning model must be supported by **at least 20% of all candidate matches** (floored at `minimal sample + 2`, capped at 50) — a model corroborated only by its own minimal sample is rejected, because on defocused/low-texture macro frames most feature matches are wrong and a barely-supported model is typically an arbitrary large, wrong transform. Index sampling uses the LCG's high bits (low bits of a power-of-two-modulus LCG have degenerate periods). The mapping from `AlignmentModeSetting` to the AKAZE matcher's degrees of freedom is itself a shared function, `pipeline::akaze_mode_for_alignment` (`Affine`/`Registration` → affine seed; `Translation` → shift + uniform-scale seed, matching the refinement DOFs) — both apps call it so the seed's DOF selection can never drift from the actual alignment mode. For `Affine`, the seed's anisotropic-scale/shear estimate now feeds real refinement DOFs instead of being discarded, since `Affine` is the only mode that solves them; for `Registration` the same affine seed is still used, but the refinement stage only solves the similarity subset, so the seed's extra DOFs are simply ignored there.

Two safety nets guard the seed and the refined result:
- `pipeline::is_sane_seed` filters the seed **before** refinement: it must be all-finite, with similarity scales in `[0.5, 2.0]` and translations within **25% of the frame dimension** (the bounded optimiser searches ±20% around the seed, so a seed further off than that would make the true alignment unreachable). An insane seed is replaced by identity. When RANSAC itself errors, the *caller* falls back to the previous frame's solved matrix (chain warm start). A shear-heavy garbage seed failing the similarity-scale band check is expected to fall back to identity too — the seed sanity check is deliberately similarity-shaped even for `Affine` mode.
- A **post-refinement gate** compares the refined matrix against identity on a ≤256 px downsampled RMS objective and falls back to identity (with a warning log) if refinement made things worse — a bounded-but-wrong seed can otherwise converge on a confidently wrong transform.

Because feature detectors fail on glossy / low-texture subjects (e.g. crystals, polished metal), seeding is **off by default**. With seeding disabled the optimiser simply starts from the identity matrix (or, in sequential/chain alignment, the previous frame's solved matrix).

### 2. Subpixel Refinement — two optimisers, one objective

Every alignment mode (`Translation`, `Registration`, `Affine`) shares the same 6-DOF parameter model (`tx, ty, scale, rotate, aspect, shear`) and the same RMS-difference objective. Which **optimiser** solves it is selected independently, via `StackingSettings::optimizer` (`OptimizerSetting`):

- **`Auto`** (default) — try the pyramid Lucas-Kanade / Gauss-Newton optimiser (`refine::refine_alignment_lk`) first; if it errors or its final RMS does not improve on its own starting RMS, fall back to Nelder-Mead (`refine_alignment_registration`) from the same seed.
- **`LucasKanade`** — force Lucas-Kanade only. On error there is no Nelder-Mead fallback (only the fallback to the sanity-filtered seed every optimiser has).
- **`NelderMead`** — force the original bounded-simplex optimiser (described below), unconditionally.

Lucas-Kanade uses analytic image-gradient Jacobians and converges in far fewer iterations per pyramid level than Nelder-Mead's blind function sampling. See `refine::lucas_kanade`'s module doc for the implementation-level detail.

#### Nelder-Mead — `refine_alignment_registration`

It performs a coarse-to-fine multi-scale search over a Gaussian luma pyramid, minimising the Root-Mean-Square (RMS) intensity difference against the reference, with:
- a per-level iteration schedule (short-side thresholding, a fixed iteration count on the large early levels, a larger count near the target, and — critically — the **finest level is deliberately under-refined** rather than iterated to convergence);
- random restarts at the coarsest level to escape local minima;
- always-bounded, logit-space parameter mapping so the simplex can never drift into a degenerate transform;
- per-mode degree-of-freedom gating — a strict ladder, each mode a superset of the last: `Translation` solves shift X/Y **plus centre-anchored uniform scale** with rotation (and aspect/shear) held fixed — the scale DOF stays enabled because focus breathing (per-frame magnification as the focus distance changes) is present in essentially every macro focus stack, and a pure X/Y shift cannot represent it (the residual radial misalignment grows with distance from the centre until the fused result is unusable). `Registration` (**the default**) adds rotation: shift X/Y + uniform scale + rotation, a similarity transform (aspect/shear held fixed). `Affine` adds anisotropic scale and shear on top of `Registration`: a true 6-DOF affine solve — the only mode that can recover independent X/Y scale or shear, at the cost of being more expensive to converge and more prone to overfitting sensor noise on otherwise-similarity-only misalignment. `None` skips refinement entirely and uses the seed as the final matrix.

This schedule exists because refining the full-resolution level to convergence chases sensor noise and produces a visible comb/ghosting artifact in the fused output — under-refining the finest level on purpose is intentional.

Every mode warps with `transform::warp_image_clamped` — an **edge-clamped** Spline4x4 interpolation that samples the nearest in-bounds pixel rather than zero-filling past the border, avoiding black edges in the full-canvas output.

### 3. Neural alignment (`AlignmentModeSetting::Neural`, requires the `nn` feature) — experimental

`Neural` replaces AKAZE seeding + classical-only refinement with a trained `stacker-nn` alignment model, via `stacker_nn::runtime::LoadedAlignModel::align`. Two architectures are supported, selected automatically from the chosen model's manifest tag: `batchalign-v2` (`BatchAlignNet` — all frames' matrices in one batched whole-stack inference call, wrapping `stacker_nn::bridge::align_planar`) and `fusionalign-v1` (`FusionAlignNet` — one reference/frame pair per call, streamed, O(1) memory in stack size, wrapping `stacker_nn::bridge::align_planar_pairwise`). The model internally downscales every frame to a 512px-longest-side copy before inference — callers pass full-resolution frames and get back full-resolution PIXEL-space matrices; no manual downscaling or matrix decomposition is needed on the caller side, since the returned `Matrix3<f32>` is fed straight into the same `warp_image_clamped`/`warp_planar` warp step every other mode uses.

By default (`neural_refine_classically = true`) the neural matrix is used as a **seed** into the same classical intensity refinement every other mode runs (`pipeline::align_frame`, `Registration` DOFs) — the seed is sanity-filtered by `pipeline::is_sane_seed` and the post-refinement identity gate applies, so a poor neural estimate degrades gracefully to classical behaviour. Set `neural_refine_classically = false` to apply the network's matrix directly with no classical refinement (useful for benchmarking the network in isolation).

Status: **experimental**, coarse registration only. No pretrained alignment model ships with this repository — train one with `z-stackr-train --strategy align` (whole-stack `BatchAlignNet`) or `--strategy fusion-align` (streaming pairwise `FusionAlignNet`), using training data generated by `scripts/03_simulate_misalignment.py` (see that binary's module docs and the `stacker-nn` crate README for the full training data format). Both the GUI and CLI grey out / refuse the Neural option when no alignment-capable model (either architecture) is discovered in the `models/` directory.

### 4. Per-frame brightness correction (`brightness` module)
`stacker_align::brightness` implements per-frame brightness/gamma correction (on by default, wired via the `correct_brightness` setting): after a frame is warped, its luma is corrected toward a **target** built from the very first frame — a 10,000-bin histogram over gamma-space luma `[0, 2]` yields a target mean and (Bessel-corrected) standard deviation. Every other frame then solves a 2-parameter `(scale, gamma)` fit (`corrected = pow(luma * scale, gamma)`) via a small Nelder-Mead run, minimising squared error between its own corrected mean/std and the target's. The fit is applied to luma, and the resulting per-pixel `luma_out / luma_in` ratio is used to scale both chroma channels too, so brightness is corrected without shifting hue. `BrightnessTarget::new(&first_frame)` builds the target once; `align_frame` applies the correction automatically when a target is passed in.

## Optional GPU acceleration (`gpu` feature)

`transform::warp_image_clamped` — the production edge-clamped warp every active alignment mode uses — internally tries a `wgpu` compute-shader port of its own kernel (`transform::gpu::warp_image_clamped_gpu`) before falling back to the CPU/SIMD implementation (`transform::warp_image_clamped_cpu`). GPU is used only when all three hold: the `gpu` feature is compiled in, the runtime toggle is on, and a `wgpu` adapter is actually available. The runtime toggle is `StackingSettings::use_gpu` (default `true`; CLI `--no-gpu`, GUI switch, Python `use_gpu`), applied via `stacker_core::gpu::set_enabled` — when off, `transform::gpu::warp_image_clamped_gpu` returns `Ok(None)` immediately and the CPU path runs, the same as when no adapter is available. The public `warp_image_clamped` signature and every call site are unaffected either way. GPU output is tolerance-equal (not bit-equal) to the CPU path — see `transform::gpu`'s module docs for the tested epsilon (max-abs diff `< 1e-3`) and `tests/gpu_warp_parity.rs`. Off by default (`--features gpu`). See `docs/gpu_acceleration_summary.md` for the full scope and implementation status.

## Core Features

- **Shared dispatch, not per-app logic**: `pipeline::align_frame` and `pipeline::akaze_mode_for_alignment` are the only alignment entry points either app should call — see "Alignment Pipeline" above.
- **Two optimisers, one objective**: `OptimizerSetting::Auto` (default) tries the cheaper Lucas-Kanade / Gauss-Newton optimiser first, falling back to the Nelder-Mead bounded simplex on failure/regression; both can be forced unconditionally. See "Subpixel Refinement" above.
- **Multi-Scale Pyramids with Random Restarts** (Nelder-Mead): Solves large displacements quickly at coarse resolutions, escapes local minima via coarsest-level restarts, then refines — deliberately not to full convergence at the finest level (see above). Lucas-Kanade also runs coarse-to-fine over the same pyramid, but with an analytic-gradient Gauss-Newton solve instead of random restarts, needing far fewer iterations per level.
- **Robust Transformation**: Handles scaling (focus breathing), translation (handheld camera shake), rotation, and (in `Affine` mode) anisotropic scale + shear, via the alignment mode's DOF gating (`Translation` keeps scale enabled precisely because of focus breathing — see above).
- **Gamma-Space Contract**: The whole pipeline stores and compares **gamma-encoded (sRGB) BT.601 YCbCr** planes — no transfer function is applied on load (a deliberate reference-fidelity choice). The RMS similarity metric, the AKAZE detector buffer, and the warp all operate on those gamma-encoded values; the RMS objective is additionally mean-bias-corrected, so global brightness offsets do not bias the fit.
- **Warping**: `transform::warp_image_clamped` (edge-clamped Spline4x4, the production path), with an optional GPU-accelerated path (see "Optional GPU acceleration" above).
- **Common-Area Crop**: `coverage_mask` and `largest_true_rectangle` compute the rectangle valid in every aligned frame; `resolve_common_crop` wraps that with the guard rails an automatic crop needs — it returns `None` (meaning "don't crop") when there's nothing to crop (the rectangle already covers the whole canvas) or when a rogue/misaligned frame has collapsed the rectangle below 25% of the canvas area. Both `z-stackr-pipeline`'s tiled/whole-image fusion and the GUI's in-RAM Stack path call `resolve_common_crop` to crop the saved output (and exclude the smeared edge-replication band from fusion processing) whenever `StackingSettings::crop_to_common_area` is enabled (the default) — see each app's README for how the crop is applied. Setting `crop_to_common_area = false` always saves the full, uncropped canvas.
- **Per-Frame Brightness Correction**: see the `brightness` module above — on by default.

## Usage

```toml
[dependencies]
z-stackr-align = { version = "1.0", features = ["akaze"] }
```

```rust
use stacker_align::{pipeline::align_frame, brightness::BrightnessTarget, Matrix3};
use stacker_core::settings::{AlignmentModeSetting, OptimizerSetting};

// Build the brightness target once, from the first frame (or pass `None` to disable correction).
let brightness_target = Some(BrightnessTarget::new(&reference_frame));

// Align + warp one subsequent frame against the reference (or, in a chain, the
// previously-warped frame), warm-started from `seed` (e.g. the previous frame's
// solved matrix, or `Matrix3::identity()` for the first frame).
let (matrix, warped_frame) = align_frame(
    target_frame,
    &reference_frame,
    seed,
    AlignmentModeSetting::Registration,
    OptimizerSetting::Auto,
    /* bounded (currently always-bounded regardless) */ true,
    frame_index,
    brightness_target.as_ref(),
);
```
