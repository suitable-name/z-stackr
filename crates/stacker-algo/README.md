# z-stackr-algo

The focus-fusion algorithms of **z-stackr** (crate identifier `stacker_algo`).

The crate takes aligned, differently focused images (`PlanarImage<f32>`, linear light) and combines the sharpest parts into one image. It does no file I/O and knows nothing about alignment.

## Implemented algorithms

### 1. Apex (Laplacian-pyramid blending), `apex`
`LaplacianPyramid::build` makes a multi-band Laplacian pyramid for every source image. At each detail band `fuse_pyramids` selects, per pixel, the source with the largest selection energy (the squared Laplacian coefficient) and copies that source's coefficients unchanged; the lowest band is the average over all sources. `reconstruct` collapses the fused pyramid into an image. `build_and_fuse_pyramids` builds the pyramids in parallel and fuses them; a request for fewer than 2 levels is raised to 2 with a warning.

Two options change only how the winner is chosen, never the values that are copied:
* **Grit suppression** (default on): at the finest detail band, the winner is the source with the largest energy summed over the 3x3 neighbourhood of the pixel instead of the pixel's own energy. Nothing is smoothed or filtered; an isolated noisy coefficient is simply less likely to win a pixel on its own.
* **Use all colour channels** (default off): the selection energy also includes the squared chroma coefficients (`luma^2 + chroma_a^2 + chroma_b^2`) instead of `luma^2` alone. The channels are not blended independently: luma and chroma are always copied together from the single winning source.

Ties (identical selection energy) go to the first frame in input order, the same rule as Relief and the auto-cull.

**`ApexFold`** is the same fusion streamed one frame at a time: `ApexFold::new(levels, use_color, grit)`, then `fold` (or `fold_source`, which pulls the frames from a `FrameSource`) for each frame, `preview` for the running fusion and `finish` for the result. It keeps a persistent best-energy plane and a strict `>` comparison, so it is bit-identical to the batch fusion, and its memory does not depend on the number of frames (`apex_fold_bytes_per_pixel(levels)`, 84 B per pixel at 8 levels, against a batch fusion that grows with the frame count). The grit energy (the 3x3 sum) of each source is computed once per frame. `fuse_streaming` is the one-call form with a per-frame callback.

* **Strengths**: keeps fine detail across depth intersections (for example overlapping bristles).
* **Trade-offs**: it can increase noise in out-of-focus background areas.

### 2. Relief (depth map with a plane-fit focus measure and a guided filter), `relief`
`relief::focus::compute_focus_map` scores every pixel with a **plane-fit residual-variance** focus measure: in a window around the pixel a plane is fitted to the luma by least squares, and the variance of what the plane does not explain is the sharpness score. A flat region and a linear ramp both score 0, and scores near the image border are biased low because the window is clipped there. The window sums come from summed-area tables kept in `f64` (a rolling band of them, so the memory stays small on large images).

The fusion (`stacker_pipeline::fuse::relief`) then:
1. takes, per pixel, the best focus score over all frames and thresholds it (`relief::threshold`): either a rejected fraction of pixels (`relief_contrast_pct`) or a robust automatic estimate (`relief_auto_detect`: the median plus 1.5 times the median absolute deviation of the map, `auto_detect_threshold`). A focus map is heavy-tailed, which pulls Otsu's method (`otsu_threshold`, still available) to the top of the range, so the rule looks for the noise floor instead;
2. where the best score is at or above the threshold, copies the pixel from the frame with the highest score there (ties go to the earlier frame); elsewhere it uses the mean of all frames;
3. smooths the luma plane of the result with an edge-preserving guided filter (`relief::guided`, guided by the luma itself, `eps` 0.01; the window variance is clamped at zero so rounding cannot make it negative). The chroma planes are not filtered. `guided_filter_pair_luma` evaluates two filters (two radii, two `eps`) over one shared set of tables, which Strata uses for its two weight scales.

This is a hard per-pixel selection followed by a filter on the result, not a blend with a continuous weight mask. The threshold is a property of the whole image: tiled callers compute it once from the whole-image focus map and pass the value to each tile.

* **Strengths**: smooth, low-noise backgrounds.
* **Trade-offs**: weaker than Apex on complex, interlocking geometry.

#### Multigrid depth (`relief::multigrid`)
`relief_use_multigrid` replaces the per-pixel argmax with a smooth depth surface. The frame index acts as a depth coordinate. Where the focus is confident (`confidence_weights(max_focus, threshold)`), the depth of a pixel is the index of its sharpest frame and the pixel is pinned to it; elsewhere the depth is filled in by diffusion from the pinned pixels. `MultigridSolver` solves that on a hierarchy of coarser grids (restrict, prolong, Jacobi relaxation), `pyramid_smooth` smooths the result with `relief_smoothing_radius` as the largest scale, and `composite_depth` turns the continuous depth back into pixels by interpolating linearly between the two nearest frames.

The planes the solver reads are small (a `u16` frame index and an `f32` weight per pixel), so a tiled run keeps per-tile running argmaxes (`update_argmax`, strict `>`, ties to the earlier frame), solves **once** on the whole image (`multigrid_depth`, about 28 B per pixel at its peak) and composites each tile from the `crop_plane` of the depth: the depth has no seams and does not depend on the tile size. The frames must be in capture order for the depth to mean anything, which is why the auto-cull keeps the stack order. The module documentation spells out every step of the cycle, in the order the CPU evaluates it, so a GPU twin (`stacker-gpu`'s `multigrid` module) can reproduce it. Two bugs of the GitHub implementation are fixed in the port (a corner block's target was halved; the prolongation skipped the lower-left sample of a block).

### 3. Strata (guided-filter soft blend), `strata`
Apex and Relief make a hard choice (a winning coefficient per band, a winning frame per pixel). Strata blends: every frame contributes to every pixel, weighted by how sharp it is there, and the weights follow the edges of the image.

`fuse_strata` makes two passes over the frames:
1. **Saliency**: the Laplacian magnitude of each frame's luma, smoothed by 5 passes of the pyramid's 5-tap blur (about sigma 2.4). A running argmax keeps the sharpest frame's index per pixel (`u16`; the first frame wins a tie).
2. **Accumulation**: each plane of a frame is split into a base (box filter of `StrataParams::base_radius`, default 31) and a detail (`frame - base`). The binary winner mask of the frame is refined by a guided filter guided by the frame's own luma, at two scales: wide and smooth for the base weight (radius 45, eps 0.3) and tight for the detail weight (radius 7, eps 1e-6). Both are clamped to `[0, 1]` and the detail weight is raised to `StrataParams::detail_focus` (default 3; `1` = no change). Base and detail are averaged with their weights (the plain mean of the layer where every weight vanishes) and added.

The weights come from the luma only and are applied to all three planes, so the fusion cannot make color fringes. The output is not clamped (the pipeline clamps, as for Apex).

* **Why `detail_focus`**: on deep stacks the guided filter leaks a small weight (0.01 to 0.05) to the frames next to the winner, and the leaks of dozens of frames add up to the winner's weight. Raising the clamped detail weight to a power keeps a winner at 1 and drives a 0.05 leak to 1e-4 (exponent 3).
* **Memory**: both passes stream; the working set is `strata_bytes_per_pixel()` (150 B per output pixel) whatever the frame count, and `STRATA_APRON_PX` (101 px: saliency reach 11 + the guided filter's two radius-45 boxes) is the tile apron. Frames are read through the `FrameSource` trait (a slice, or `FnFrames` for a loader), so a caller can supply them one at a time.
* **Devices**: `fuse_strata_on` takes a `StrataKernels` implementation that may serve the saliency and the guided-filter pair elsewhere (a GPU); a method that returns `None`, or a result of the wrong length, runs on the CPU (`CpuStrata`).
* **Progress and preview**: `fuse_strata_with_progress` ticks `2 * N` times; `fuse_strata_with_preview` also hands over a snapshot every `preview_every` frames and once with the finished image.
* **Constants in linear light**: the guided-filter constants were tuned on gamma-encoded data; `tests/strata_fuse.rs` checks that a dark, low-contrast sharp region is still taken from its sharp frame.

## Frame optimisation: auto-cull and ordering (`optimize_stack`)

Before fusion the apps can run `optimize_stack`, which drops frames that add almost nothing and re-orders the survivors. It uses the same plane-fit focus measure as Relief, with window radius 1:

1. **Per-frame focus maps** are computed in parallel, one plane per frame.
2. **Scene-adaptive contrast floor.** The global maximum focus score over all frames and pixels sets a floor of `max(1 % of that maximum, 1e-6)`. Pixels whose best score is not above the floor (flat background, blown highlights) count toward neither the numerator nor the denominator of a frame's score, so the scoring covers meaningful detail only.
3. **Per-pixel winner.** At each detail pixel the frame with the highest score wins it (the first frame on an exact tie). Each frame accumulates a win count and the centroid of the pixels it won.
4. **Cull.** A frame is dropped when its win count is below `effective %` of the detail-pixel count, where `effective % = min(requested %, cull_threshold(N))` and `cull_threshold(N) = min(2, 50 / N)` for `N` frames. The apps request 2 % unless `auto_cull_threshold_pct` sets a fixed share (0, the default, keeps this rule). For up to 25 frames that is the absolute 2 %; for deeper stacks it shrinks to half of the average per-frame share, so a deep stack keeps its frames (a flat 2 % would empty any evenly spread stack of 51 or more frames). If the cull would drop every frame, or the stack has no detail at all, all frames are kept and a note is added to `OptimizationResult::warnings`.
5. **Order.** The kept frames are sequenced by a greedy nearest-neighbour walk over their win centroids, starting from the frame whose centroid has the smallest x and repeatedly moving to the nearest unvisited centroid (squared Euclidean distance). A frame that won no pixel gets the image centre as its centroid.

`optimize_stack` returns `kept_indices` (ascending) and `recommended_order`. With auto-cull switched off the apps do not call it: no frame is dropped and the stack keeps its loaded order (natural file-name order, optionally reversed). The out-of-core CLI computes the same census tile by tile (`stacker_pipeline::cull`, with a test that it agrees with `optimize_stack`).

## Retouch (`hybrid::retouch`)
`RetouchSession` holds a base image (the fused result), a second image (a frame, or another fusion pass) and an alpha mask. `apply_brush` paints a hard-edged circular dab that raises alpha with `new = old + opacity * (1 - old)`; `render_composite` gives `base * (1 - alpha) + source * alpha`. `RetouchHistory` undoes and redoes whole strokes (`begin_stroke`, `extend_stroke`, `end_stroke`, `undo_stroke`, `redo_stroke`) by storing only the 128-value blocks of the mask that a stroke touched, so a stroke's memory follows the area it covers, not the image size. The GUI uses this for its retouch brush.

## Architecture

* **Tile-friendly**: both algorithms run on `PlanarImage` tiles fetched from the `stacker-core` tile store. A tiled `Apex` run needs an apron that depends on the pyramid depth, `apex::fuse::apron_for_levels`: 128, 256 or 512 px for 6, 7 or 8 levels (the dependence radius `4 * (2^(L-1) - 1) + 1`, rounded up to a multiple of `2^(L-1)`). The tile interior of such a run equals the whole-image result. A `Relief` tile needs `estimation radius + 2 * smoothing radius + 1` px. A `Strata` tile needs `STRATA_APRON_PX` (101 px).
* **Linear space**: all computation, from the focus measure to the pyramid, runs on linear `f32` planes.
* **Parallelism**: `rayon` parallelises the per-frame focus maps, the guided-filter window queries, the pyramid builds across frames and the fusion loops, and the first pass of the guided filter's summed-area-table builds.
* **SIMD**: the separable pyramid blur uses portable `std::simd`. The lane count is fixed at compile time from the target features (16 with `avx512f`, 8 with `avx2`, otherwise 4: one 128-bit operation on the default `x86-64` build) and does not change the result; the blur uses plain multiply and add, so it needs no `fma`. Build with `-C target-cpu=x86-64-v3` or newer for wider vectors (see the root README, "CPU features").

## Usage
`stacker-algo` does not read files. It takes `PlanarImage` values and returns `PlanarImage` values. Most callers use the whole-image stages in `stacker_pipeline::fuse` (`apex`, `relief`, `relief_tile`, `strata`, `relief_multigrid`), which read the settings keys.

```rust
use stacker_algo::apex::fuse::build_and_fuse_pyramids;
use stacker_core::image::PlanarImage;

// `frames` holds pre-aligned, equally sized images.
fn fuse_apex(frames: &[PlanarImage<f32>]) -> PlanarImage<f32> {
    // 8 levels, luma-only selection energy, grit suppression on.
    build_and_fuse_pyramids(frames, 8, false, true).reconstruct()
}
```
