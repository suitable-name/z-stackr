# z-stackr-algo

The focus fusion algorithm engine for **z-stackr** (crate identifier `stacker_algo`).

`stacker-algo` implements the core algorithms that take perfectly aligned, differently-focused images and selectively blends the sharpest pixels into a single, infinite-depth-of-field composite.

It relies heavily on mathematical operations, convolutions, and filtering, entirely isolated from file I/O or alignment concerns.

## Implemented Algorithms

This crate provides three distinct rendering methodologies, selectable at runtime:

### 1. Apex (Laplacian Pyramid Blending)
The `apex` module constructs a multi-band Laplacian pyramid for every source image. At each spatial frequency band, it selects the maximum absolute coefficient (the sharpest localized contrast) across all images. It then collapses the pyramid to synthesize a seamlessly blended image. (`apex` remains the internal module name; the method is known as "Apex" in user-facing displays.)
* **Advantages**: Highly resilient against stacking artifacts like halos, ghosting, or intersecting depth planes (e.g., overlapping insect bristles).
* **Trade-offs**: Can increase background noise and alter global contrast slightly.

### 2. Relief (Depth Map) — two selectable engines
The `relief` module identifies sharp regions using a **Sum-Modified Laplacian (SML)**-style contrast metric (in practice a detrended local-plane-fit residual variance) computed by `compute_relief_smls`. From there, `fuse.rs` offers **two fusion engines**, selected by the `relief_use_multigrid` setting (both remain available — this is not a replacement, per the project's CLI/GUI-parity design principle). (`relief` remains the internal module name; the method is known as "Relief" in user-facing displays.)

* **Guided-Filter engine (`fuse_relief_with_mask`, default)** — computes a per-pixel "winner-takes-all" source-frame pick (hard argmax of SML, with a contrast mask excluding low-confidence pixels, which fall back to a simple cross-frame mean), then smooths the resulting colour with a single edge-preserving **Guided Filter** pass (f64 integral images — an f32 summed-area table catastrophically cancels at full resolution).
* **Multigrid engine (`fuse_relief_multigrid`)** — seeds a per-pixel "known" frame-index field from the same hard-argmax-of-SML pick (confidence-weighted: `weight = 1.0` where above the contrast threshold, `0.0` elsewhere), then runs a confidence-weighted **geometric multigrid V-cycle** (`multigrid::MultigridSolver`, restrict → recurse-coarser → prolong → relax) to diffuse a *continuous* depth-index field into the low-confidence regions — rather than a single-pass smoothing of colour, this solves the depth field itself at multiple scales. The final image is a two-endpoint linear blend between the two frames bracketing each pixel's solved (fractional) depth index. A post-solve smoothing pass is applied to the solved index field before blending: `relief::pyramid::pyramid_smooth` (`MultigridSolver::get_smoothed_solution`) — a Gaussian pyramid built by repeated blur+decimate then collapsed coarsest-to-finest by repeated blur+interpolate, with `relief_smoothing_radius` controlling how many octaves the pyramid spans. See the doc comment on `relief::pyramid::expand_replace` for the reasoning behind the collapse step's combine mode.

Both engines share the same `ReliefSettings` (`est_radius`, `smooth_radius`, `contrast_pct`, `absolute_threshold`) and the same SML/contrast-mask front end — only the source-selection-and-blend back end differs.

**Contrast-mask semantics** (`threshold::generate_mask`): `contrast_pct == 0` always yields an all-`true` mask; `contrast_pct > 0` is a percentile of the **non-zero** SML population (the SML map has a dense exact-zero plateau from defocused background, which would otherwise make nearby percentiles indistinguishable), and exact-zero pixels are always masked out. `absolute_threshold: Some(v)` bypasses the percentile entirely — the tiled pipeline resolves ONE absolute value from global (all-tile) statistics in a pre-pass and passes it to every tile, eliminating tile-to-tile threshold seams. The auto-detect helper `fuse::auto_contrast_threshold` derives a suggested percentile from the noise floor (median + 1.5×MAD of the non-zero population); classic Otsu was abandoned because the heavy-tailed SML distribution collapses its between-class split to ~99%.

* **Advantages** (both engines): Maintains beautiful, low-noise backgrounds and accurate original contrast; the multigrid engine additionally gives multi-scale-consistent depth continuity rather than a single-pass smooth.
* **Trade-offs**: Struggles slightly on complex interlocking geometry compared to Apex.

### 3. Strata (Guided-Filter Fusion)
The `strata` module makes a **soft** decision instead of `Apex`'s/`Relief`'s hard per-band/per-pixel picks: every frame contributes to every output pixel, weighted by how in-focus that frame is *there*. A per-frame `|laplacian3x3|` saliency map (locally averaged) feeds a winner-take-all indicator per frame, which is then refined by the **existing Guided Filter** (`relief::guided::guided_filter`) at two scales — wide/smooth (`r=45, eps=0.3`) for the low-frequency "base" layer, tight/crisp (`r=7, eps=1e-6`) for the high-frequency "detail" layer (`base`/`detail` split by an existing box filter, `relief::guided::box_filter`, at a settings-exposed radius, default 31). The two scales are what suppress halos: a depth edge gets a gradual base-layer handover but a pixel-accurate detail-layer handover. 

**Deep-stack fix:** on stacks of dozens of frames, the detail-layer guided filter's edge-aware feathering leaks a small positive weight from the true per-pixel winner to every near-winner frame; with tens of losers those leaks sum to a magnitude comparable to the winner's own weight, diluting sharp detail into a "watercolour" softness that `base_radius` cannot reach. The detail weight (only the detail weight — the base weight is untouched, since its softness is the halo suppression) is now cubed after its `[0, 1]` clamp and before accumulation (`DETAIL_WEIGHT_EXPONENT = 3` in `strata::mod`), re-concentrating the per-pixel competition: a winner near `1` is unaffected, a `0.05` leak collapses to `~1.25e-4`. A no-op for the exact `0`/`1` weights small stacks produce.

* **Advantages**: A genuine third option between `Apex`'s dense-bristle intersecting detail and `Relief`'s smooth-surface depth continuity — soft edge-aware blending gives fewer halos at depth edges than either hard-decision engine.
* **Trade-offs**: Two guided-filter passes per frame (base + detail) make it the most compute-intensive of the three engines.
* **Live preview**: `strata::fuse_strata_with_preview(frames, params, on_tick, preview_every, on_preview)` adds a periodic snapshot callback (a fully materialised, normalised composite of the accumulators so far) during the accumulation pass, for GUI live-preview wiring; `fuse_strata_with_progress` delegates to it with previews disabled.

## Optional GPU acceleration (`gpu` feature)

Every fusion engine in this crate has a `wgpu` compute-shader path, tried automatically before falling back to the unmodified CPU/rayon implementation: the tiled/batch Apex fusion (`apex::gpu::fuse_pyramids_gpu`), the incremental whole-image `ApexAccumulator` (GPU-resident running state via `apex::gpu::accumulator::GpuApexAccumulator`, read back only at `finish()`), Strata's Laplacian-magnitude saliency pass (`strata::gpu`), and Relief's two hot loops — the fused guided filter (`relief::gpu::guided_filter_gpu`, one GPU round trip for the whole six-box-mean-step pipeline, tried by `relief::guided::guided_filter`) and the Multigrid Jacobi relaxation sweep (`relief::gpu::relax_gpu`). `relief::guided::box_filter` itself is always CPU/rayon, including Strata's direct call for its base/detail split. Whether the GPU is used at runtime is controlled by `StackingSettings::use_gpu` (default `true`; CLI `--no-gpu`, a GUI switch, and the Python `use_gpu` setting all map to `stacker_core::gpu::set_enabled`) — with the switch off, or no adapter present, or any GPU call failing, the CPU path runs exactly as it always has. GPU output is tolerance-equal (not bit-equal, max-abs diff `< 1e-3`) to the CPU path — see the runtime-gated parity suites (`tests/gpu_fuse_parity.rs`, `tests/gpu_apex_accumulator_parity.rs`, `tests/gpu_strata_parity.rs`, `tests/gpu_relief_parity.rs`) for scope and implementation status. Off by default at compile time (`--features gpu`); a default build has no `wgpu` dependency at all.

## Frame Optimisation — auto-cull & ordering (`optimize_stack`)

Before fusion the pipeline can run `optimize_stack`, which both **drops
low-value frames** and **re-orders the survivors**. It is driven by the same
Sum-Modified-Laplacian (SML) focus metric Relief uses, and decides the order as
follows:

1. **Per-frame focus maps.** An SML focus map is computed for every frame (in
   parallel across the stack).
2. **Scene-adaptive contrast floor.** The global maximum SML over *all* frames
   and pixels sets a floor of `max(1 % of global-max, 1e-6)`. Pixels below it
   are flat background / blown-out highlights and are excluded entirely — they
   count toward neither the numerator nor the denominator of a frame's score, so
   the scoring is over meaningful detail only.
3. **Per-pixel winner-take-all.** At each *detail* pixel (SML above the floor)
   the frame with the highest SML "wins" that pixel. Each frame accumulates a
   win count and the centroid (mean x/y) of the pixels it won.
4. **Cull.** A frame is dropped when its win count is below
   `min_contribution_pct` of the total detail-pixel count — a percentage of
   *detail* pixels, so it is insensitive to how much flat background a scene
   has. This removes frames that never contribute the sharpest version of any
   region (redundant near-duplicates, or frames focused on empty space) rather
   than "blurry" frames as such. The GUI's **Auto-Cull Frames** toggle uses a
   2 % threshold.
5. **Order — nearest-neighbour by in-focus centroid.** The kept frames are
   sequenced by a greedy nearest-neighbour walk over their win-centroids,
   starting from the left-most centroid (smallest x) and repeatedly hopping to
   the nearest unvisited centroid (squared-Euclidean distance). Because the
   in-focus region sweeps across the frame as focus moves through the subject,
   ordering by centroid proximity approximates the physical focus sweep — it
   keeps consecutive frames spatially coherent, which matters most for the
   recurrent AI-merge mode and gives Apex/Relief a tidy, monotonic progression.

`optimize_stack` returns both `kept_indices` (the culled subset in original order) and `recommended_order` (the culled subset sequenced by sharpness). The caller (e.g. the GUI) exposes these as independent settings: you can cull without re-ordering, re-order without culling (by setting the cull threshold to 0%), or both.

## Architecture

* **Tile-Friendly**: All three engines are designed to operate perfectly on `PlanarImage` tiles fetched by the `stacker-core` tile manager. They only require a fixed `APRON_PX` halo to guarantee artifact-free boundaries.
* **Gamma-Space `f32`**: All computations, from SML evaluation to pyramid decomposition, run on `f32` planes holding **gamma-encoded (sRGB) values** — the pipeline-wide reference-fidelity contract (no transfer function is applied on load). Precision-critical accumulations (SML summed-area tables, guided-filter integral images) use `f64` internally.
* **Rayon Parallelism**: Internally parallelizes intensive filtering loops (like SML and guided filtering) using `rayon`, including the guided filter's summed-area table construction (`relief::guided::box_filter`, two rayon passes: a per-row horizontal prefix sum, then a vertical accumulation parallelised across column chunks — bit-identical to the old sequential scan) and Strata's per-pixel Pass-1/Pass-2 loops.
* **SIMD**: The separable Laplacian-pyramid blur uses portable `std::simd` (AVX2 on zen3, AVX-512 on zen4, scalar fallback elsewhere) on top of the rayon parallelism.

## Usage
`stacker-algo` does not read files. It expects `PlanarImage` structs and returns `PlanarImage` structs.

```rust
use stacker_algo::{
    apex::fuse::build_and_fuse_pyramids,
    relief::guided::guided_filter
};

// ... feed it an array of pre-aligned PlanarImages
```