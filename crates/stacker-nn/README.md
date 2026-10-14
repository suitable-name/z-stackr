# z-stackr-nn

The neural focus-merge engine for **z-stackr** (crate identifier `stacker_nn`), built on the
[Burn](https://burn.dev/) deep-learning framework. It provides the model, the
training pipeline, tiled inference, and the glue that lets a trained model act as
a stacking algorithm ("AI Model") alongside the classical Apex, Relief and Strata.

## Inference scheme

Rather than ingesting the whole stack at once, the network performs a **recurrent
pairwise merge**: the running composite starts as frame 0 and each subsequent
frame is merged into it.

```text
frame_0 ─► (target) ─┐
                     ├─► FocusMergeNet ─► composite_1 ─┐
frame_1 ─► (source) ─┘                                 ├─► FocusMergeNet ─► composite_2 ─► …
                                        frame_2 ───────┘
```

Each step takes the current composite (`target` RGB + a per-pixel confidence
channel) and a new `source` frame, and predicts an updated composite, an updated
confidence, and the soft selection map used to blend them. The confidence
channel is meant to make the recurrent accumulation less sensitive to frame order
(the training data from `scripts/02_blender_focus_stack.py` records a random fold
order per scene by default, so the network sees other orders than the focus
sweep; this has not been measured). The fold is streaming: `bridge::fuse_planar`
converts one planar frame to a tensor at a time, so memory holds the caller's
planar frames plus the running composite and one converted frame, not a second
copy of the whole stack. Inference is still all in memory; it is not out-of-core.

## `FocusMergeNet` architecture

Fully-convolutional and **size-preserving** (one model serves every resolution;
8K is handled by tiled inference, not a bigger model):

1. **Shared stem** — encodes each 3-channel frame; weight-shared between target
   and source so they live in a comparable feature space.
2. **Dilated residual context stack** — `depth` residual blocks whose dilations
   cycle `1 → 2 → 4 → 8`, growing the receptive field geometrically **without any
   pooling**, so tiles stay independent and seams blend cleanly.
3. **Sharpness branch** — a shared high-pass head producing a per-pixel sharpness
   estimate for the target and source.
4. **Gating head** — fuses features + sharpness + confidence into a soft
   selection map `α ∈ [0, 1]`.
5. **Pre-merge** — `merged = α·source + (1 − α)·target`.
6. **Residual refinement** — a small `tanh`-bounded residual that suppresses
   halos/seams.
7. **Confidence head** — predicts the updated confidence channel.

Every normalisation layer is a **per-pixel channel norm** (`model::ChannelNorm`: a
LayerNorm over the channel axis at each pixel, then a learned per-channel
gain/bias). No statistic involves a spatial extent, a batch or a tile, so a
pixel's output depends only on the input pixels inside its receptive field and
tiled inference agrees with whole-image inference away from the tile edges
(`tests/tiling.rs`). The earlier `GroupNorm` normalised over the whole `H × W`
extent of whatever tensor it was handed, so training crops, GUI tiles, CLI tiles
and the whole-image path all saw different statistics. Per-pixel statistics were
chosen over a fixed spatial window because a window of radius `r` would add `r`
to the receptive field at *every* norm layer (hundreds of pixels in total); the
price is that the absolute activation magnitude at a pixel is normalised away
(ConvNeXt makes the same trade), which only a retraining can evaluate.

### Size presets

| Preset | Channels | Context depth | `Σ dilations` | Receptive-field radius `R` | Tile overlap `2·R` |
|--------|---------:|--------------:|--------------:|---------------------------:|-------------------:|
| `xs`   | 16       | 2  | 3  | 16 px  | 32 px  |
| `s`    | 24       | 3  | 7  | 24 px  | 48 px  |
| `m`    | 32       | 4  | 15 | 40 px  | 80 px  |
| `l`    | 48       | 6  | 18 | 46 px  | 92 px  |
| `xl`   | 64       | 9  | 31 | 72 px  | 144 px |
| `xxl`  | 96       | 12 | 45 | 100 px | 200 px |

`R = 2·Σ dilations + 10` (stem 2, context blocks `2·d` each, sharpness conv 1,
gate 1 + 2, refinement 1 + 2 + 1), computed by the one function
`model::receptive_field_radius(size)` (or `FocusMergeNetConfig::receptive_field_radius()`
for a custom geometry). `TileConfig::for_size`/`for_radius`, `TileConfig::default()`
(largest preset), `runtime::recommended_tile()` (largest preset) and
`runtime::recommended_tile_for(&entry)` (the entry's own geometry) all derive their
overlap from it. With overlap `2·R` every pixel has at least one tile that sees its
full context, and the raised-cosine blend tapers a tile's truncated edge to zero.

Build a config with `FocusMergeNetConfig::from_size(ModelSize::L)`. A trained
model records its exact geometry in a sidecar manifest so it always reloads
correctly.

## Training loss

`FocusFusionLoss` combines four terms, **all** weighted by the scene's occlusion
mask (`w = 1 - (1 - occlusion_min_weight) * occlusion`):

* **Charbonnier** on RGB — robust reconstruction.
* **Multi-scale gradient L1** — preserves high frequency (the whole point); the
  weight map is pooled to each scale and each finite difference gets the mean
  weight of the two pixels it spans.
* **Sharpness retention** — penalises output blurrier than the sharpest evidence:
  the hinge is against the per-pixel maximum of the Laplacian sharpness over *all
  frames merged so far* (`loss::sharpness_map`, `forward_with_reference`), not just
  the current source.
* **Confidence supervision** — predicted confidence vs. the ground-truth in-focus
  coverage map.

The ground-truth composite copies the all-focus render wherever an included plane's
mask exceeds `exp(-0.5)` (one sigma of the Gaussian mask, `data::ALLFOCUS_MASK_THRESHOLD`);
elsewhere it falls back to the least-defocused included frame. (The old `0.1` was a
2.15-sigma contour: pixels visibly defocused in every included frame were supervised
with the all-focus image.)

The recurrent rollout trains on the pairwise accumulation with **scheduled
sampling** (periodically feeding the model's own detached prediction as the next
target) to fight exposure bias. Step 0 starts from the **raw first frame with
confidence 0**, exactly what inference feeds the first merge (it used to start from
an oracle composite with the mask as confidence).

`train::rollout_train` does `forward -> loss -> backward -> optimiser step` for each
step and drops that step's graph before the next, so peak graph memory is that of one
pairwise merge whatever the plane count (`train::rollout_loss`, which keeps every
step's graph until one backward, remains for evaluation and tests). The model is
updated after every step (`n_planes - 1` AdamW updates per scene, at the scene's
learning rate).

### Training data (`scripts/`)

A scene directory holds the frames, soft in-focus masks, the all-focus ground
truth, an occlusion map and a `metadata.json` that names every file; the loader
(`data`) reads all file names from the metadata, so none are hard-coded. The
two scripts in the repository's `scripts/` folder produce this layout from
ordinary photographs:

1. `scripts/01_estimate_depth.py` (normal Python; packages in
   `scripts/requirements.txt` plus Depth Anything 3) writes a `<name>_depth.npy`
   near/far map per photo.
2. `scripts/02_blender_focus_stack.py` runs inside Blender
   (`blender -b -P scripts/02_blender_focus_stack.py -- --img ... --depth ... --out scene_dir`),
   needs no packages beyond Blender's bundled numpy, and writes `focus_NN.png`,
   `mask_NN.png`, `allfocus.png`, `occlusion.png` and `metadata.json`. Frames are
   8-bit by default; masks and the occlusion map are always 8-bit (the loader
   decodes every PNG to 8 bits, so 16-bit data would only be narrowed). The
   displaced mesh resolves depth at `--subdiv` (default 400) per side, which
   smooths depth edges relative to the full-resolution masks.

The scripts are not exercised by any test. Run them on your own photographs and
point `z-stackr-train --data` at the folder that holds the scene directories.

The two alignment strategies read a different layout. `scripts/03_simulate_misalignment.py`
takes a stage-2 scene and writes a new scene directory with randomly shifted, rotated,
scaled and sheared frames (every frame, including the reference, warped and centre-cropped
so no border gives the shift away) and a `metadata.json` that adds the ground truth:

```text
scene_unaligned/
  metadata.json   n_planes, stack[], alignment_gt[][3][3], cropped_dims [w, h]
  <stack[i]>      misaligned frame i, sRGB PNG, every one cropped_dims in size
```

`alignment_gt[k]` is a pixel matrix in the `cropped_dims` grid that maps a reference
pixel to the matching pixel of frame `k`: `[x_k, y_k, 1]^T = M_k [x_ref, y_ref, 1]^T`.
That is the direction the classical refiners return, so a network's matrices can seed
them. Frame 0 is the reference and has the identity. The ranges of the simulated
misalignment have to stay inside the bounds of the network's head (shift 0.30
normalised, rotation 0.10 rad, log-scale 0.08, shear 0.05); widen both together.

The photometric fine-tune (`--real-data`) needs no metadata at all: one directory per
real stack holding `png`, `jpg`, `jpeg` or `tif` files, ordered naturally
(`img2` before `img10`), the first being the reference.

### Crop-correct alignment labels

The aligners work in normalised coordinates, `(x, y)` of an image of `w x h` pixels being
`(2x/(w-1) - 1, 2y/(h-1) - 1)`, so the image covers `[-1, 1]^2`, and a matrix is
`M_n = N M_px N^-1`. **`w x h` is the size of the image the network is shown**: the
downscaled thumbnail at inference and the training crop in the trainer.

A crop is a different image with its own pixel grid, so the label of a crop at offset
`o = (left, top)` is `N(size, size) . T(-o) . M_px . T(o) . N(size, size)^-1`
(`data::crop_normalized_matrix`). Normalising the full-frame matrix by the full frame
while training on 256 px crops (what the port started from) would make every label's
translation too small by `size / frame` and put every rotation and scale about the wrong
point; no inference-time image could reproduce that mapping. `tests/data_align.rs`
checks the label against real shifted pixels and against a rotation about the frame
centre. `data::pixel_matrix_to_normalized` / `normalized_matrix_to_pixel` are the plain
conversions for a whole image.

### Data loading and memory

`data::SceneImages::load` decodes the `2·n_planes + 2` PNGs of a scene once (in
parallel); `RolloutSequence::crop_steps` crops first, linearises only the crop and
builds every step from it. `data::SceneCache` keeps decoded scenes (8-bit,
`(4·n_planes + 4)·H·W` bytes each) between optimiser steps under a byte budget with
least-recently-used eviction: **the cache never retains more than `--cache-mb`
(default 1024 MiB)**, a scene larger than the budget is decoded per use and not
retained, and the peak is the budget plus one scene in flight plus the linearised
crop. `--cache-mb 0` keeps nothing (still one decode per scene per step, instead of
`2k + 7` per rollout step).

The other strategies use the same machinery: `RolloutSequence::batch_from_images`
builds the batch fusion sample from a cached `SceneImages`, and
`data::AlignSceneCache` (3 bytes per pixel and frame) caches alignment scenes under the
same budget rules. `fusion-align` trains on one (scene, frame) pair per step and
shuffles all pairs together, so give it a budget that holds the whole data set, or each
step decodes its scene again.

## Retraining required (old weights are incompatible)

The normalisation layers changed (`GroupNorm` -> `ChannelNorm`, parameters `scale` /
`shift` instead of `gamma` / `beta`), the presets now use `norm_groups = 1`, the
step-0 training target changed, the loss terms changed and each step now updates the
optimiser. A checkpoint from before this change **fails to load** (missing field
`scale`; the error says so) and must not be resumed. Before retraining, delete from
the `--out` directory (and from `models/` if copied there):

* `focusmerge_latest.model.mpk`, `focusmerge_latest.model.json`,
  `focusmerge_latest.optim.mpk`, `train_state.json` (the rolling checkpoint and
  optimiser state; do not use `--resume` across this change);
* every `focusmerge_epoch_NNN.mpk` / `.json` and `focusmerge_final.mpk` / `.json`
  pair, and any legacy `focusmerge_latest.mpk` / `.json`.

Then start a fresh run (no `--resume`). Hyper-parameters (`--lr`, `--epochs`) were
tuned for the old per-scene update; with `n_planes - 1` updates per scene the
effective number of optimiser steps grows accordingly, so consider a lower `--lr`
(e.g. 1e-4) for the first run.

## Feature / backend matrix

| Feature    | Backend                  | Use-case                         |
|------------|--------------------------|----------------------------------|
| `ndarray`  | `burn-ndarray` (CPU)     | default; tests, CPU inference     |
| `wgpu`     | `burn-wgpu` (Vulkan/MSL) | GPU inference and training        |
| `autodiff` | autodiff wrapper         | default; enables backprop         |

There is no CUDA feature. At least one of `ndarray` / `wgpu` must be on (the crate
refuses to compile otherwise). The integration tests that need a CPU backend are
gated on `ndarray`, so `--no-default-features --features wgpu,autodiff` builds and
lints cleanly too.

`burn::optim` (AdamW) and the file recorders are available once `burn/std` is on,
which every backend feature enables — so the trainer needs no special feature.
Several backends may be enabled at once (e.g. the GUI compiles `ndarray` + `wgpu`
to offer a runtime CPU/GPU choice); the scalar `SelectedBackend` alias resolves
by priority `wgpu > ndarray`, while runtime inference uses concrete backend types.

## Public API (app-facing)

The CLI/GUI never touch Burn directly:

* `discover_default_models()` / `discover_models(dir)` → `Vec<ModelEntry>` from a
  `models/` directory (`discover_default_fusion_models()` and
  `discover_default_alignment_models()` are the lists for the AI-model and the
  alignment-model pickers; the apps use these two) (`<stem>.mpk` + `<stem>.json`), best default first:
  `focusmerge_latest`, `focusmerge_final`, `focusmerge_epoch_NNN` newest first,
  then other names A–Z. The apps default to the first entry, so a trainer
  `--out models` directory yields the newest usable model. A bad manifest is
  skipped (and logged) on its own; `discover_models_report` also returns the
  skipped files and reasons.
* `available_devices()` / `gpu_available()` — what backends were compiled in.
* `LoadedModel::load(entry, device)` then `.fuse(frames, tile)` /
  `.fuse_with_progress(frames, tile, on_step)` — fuse a stack of
  `PlanarImage<f32>` (the pipeline's colour type), with an optional per-frame
  progress callback for live preview.
* `fuse_entry(entry, device, frames, tile)` — one-shot convenience.
* `LoadedAlignModel::load(entry, device)` then `.align(frames)` — the pixel matrices
  (reference to frame) a trained alignment model predicts for a stack; the
  pipeline's `neural` alignment mode seeds from them.

Lower-level building blocks are also public: `model`, `loss`, `infer`
(`fuse_stack` / tiling), `bridge` (planar↔tensor), `data` (dataset + rollout),
`train` (rollout loss + schedules).

## Training binary

`z-stackr-train` (built when the `autodiff` backend is on, which is the
default) is the optimiser loop: AdamW, cosine LR with warm-up, recurrent rollout
with scheduled sampling, crash-safe rolling checkpoints, and `--resume`. See the
project README's "Training the AI model" section for usage. For GPU training build
with `--no-default-features --features wgpu,autodiff`.

### Strategies (`--strategy`)

| `--strategy` | Network | Reads | Checkpoint stem | One "step" |
|---|---|---|---|---|
| `pairwise` (default) | `FocusMergeNet` | `02_blender_focus_stack.py` scenes | `focusmerge` | one scene, one update per merge |
| `batch` | `BatchMergeNet` | the same scenes | `batchmerge` | one scene, one update |
| `align` | `BatchAlignNet` | `03_simulate_misalignment.py` scenes | `batchalign` | one scene, one update |
| `fusion-align` | `FusionAlignNet` | the same scenes | `fusionalign` | one reference/frame pair, one update |

`pairwise` behaves exactly as before. All four share `--size`, `--epochs`, `--lr`
(cosine schedule with warm-up), `--crop`, `--seed`, `--ckpt-every`, `--cache-mb` and
`--resume [--fresh]`; the last three flags below belong to the aligners:

* `--corr-radius N` — search radius of the local correlation volume (in feature cells at
  the /8 scale; default 4, the preset's). It fixes the width of the head, so it is
  recorded in the manifest and a model loads with the radius it was trained with.
  A larger radius lets the network see larger displacements, but the head's bounds
  (see above) and the simulated ranges have to grow with it.
* `--real-data DIR` (`align` only) — directories of real, unlabelled stacks. In the last
  quarter of the epochs (rounded down, so none below four epochs) each step also draws
  one random real stack and adds `--photometric-weight W` (default 0.1) times the mean
  gradient-domain photometric loss of its frames, each warped by the matrix the network
  predicts. **This phase is unvalidated**: it has been checked for shapes and finiteness
  only, never shown to improve a trained model; leave it off until a baseline exists.
* `--ss-max` (scheduled sampling) is pairwise only; flags a strategy ignores are
  reported at start-up (`note: --corr-radius has no effect with --strategy batch`).

Every strategy writes its manifest with the architecture tag of its network, so
`discover_models` and the apps pick fusion and alignment models apart, and an
alignment checkpoint is never mistaken for a focus-merge one. The first run of the
aligners: the head is zero-initialised, so every predicted matrix starts as the
identity; the loss is exactly zero on an identity ground truth, which makes a quick
smoke test (`--epochs 1 --crop 64` on a few scenes) a meaningful one.

### Checkpoints (`checkpoint` and `train::strategy_checkpoint`)

Everything is written to `--out`, each file atomically (temp name + rename). The table
is the pairwise strategy (module `checkpoint`); every other strategy writes the same
files under its own stem (module `train::strategy_checkpoint`), with the state file named
`<stem>_train_state.json` so strategies sharing an `--out` directory never touch each
other's files:

| File | Contents | Precision |
|---|---|---|
| `focusmerge_latest.model.mpk` + `.json` | rolling model, refreshed every `--ckpt-every` steps and each epoch end | f32 |
| `focusmerge_latest.optim.mpk` | AdamW moments | f32 |
| `train_state.json` | resume point (epoch, epoch-start step, preset, schedule length) | n/a |
| `focusmerge_epoch_NNN.mpk` + `.json` | per-epoch export | f16 |
| `focusmerge_final.mpk` + `.json` | final export | f16 |

The other strategies' state file also records the strategy; resuming it under another
`--strategy` is refused. Pairwise keeps `train_state.json` and has no strategy field.

The model and optimiser stems differ in a dotted marker (`.model` / `.optim`)
because Burn's recorders replace the last extension of the path they are given;
the module always passes complete `….mpk` paths so the markers survive. One
reader (`infer::load_weights`, f32 settings) loads both precisions: f16 files
widen exactly and f32 files are untouched, whereas reading an f32 file with the
half-precision `CompactRecorder` would silently round it.

`--resume` requires `train_state.json`, the rolling model and the optimiser file;
a missing or corrupt one is an error, as is a different `--size` or a different
schedule length (`--epochs` x scene count). `--resume --fresh` resumes when it can
and otherwise starts from scratch. A mid-epoch checkpoint resumes at the start of
its epoch with the step counter rewound, so the cosine schedule still ends at its
last step. A stale `focusmerge_latest.mpk` from the old trainer holds no usable
weights (the optimiser state was written over it); discovery skips it, delete it.

## Evaluating an alignment model

`scripts/04_eval_alignment.py` judges a trained aligner on held-out simulated scenes: the
mean corner error in pixels (the four image corners moved by the predicted and by the
ground-truth matrix, averaged over the non-reference frames), raw and scaled to a 3000 px
wide image, for the network alone, the network plus classical refinement, and the
classical baseline. The inference side is the `stacker-nn-eval` binary:

```text
cargo build -p z-stackr-nn --release --bin stacker-nn-eval
python scripts/03_simulate_misalignment.py --input scene_dir --out held_out/scene_a   # per scene
python scripts/04_eval_alignment.py --scenes-dir held_out --model models/batchalign_final.mpk \
    --eval-binary target/release/stacker-nn-eval
```

The script writes one JSON request per scene to the binary's stdin
(`model_path`, `ref_index`, `output_dims`, `refine`, `frames`) and reads
`{"status": "ok", "matrices": [...]}` back; the matrices map reference pixels to frame
pixels in the frames' own grid, the convention of `alignment_gt`. `refine: true` seeds
`stacker_align::refine::refine_alignment_registration` (the classical registration
refiner, with its default bounded search of +-5 % around the seed) with the network's
matrices; without `model_path` the same refiner runs from the identity, which is the
classical baseline. The binary can also be driven directly:

```text
stacker-nn-eval --scenes-dir held_out --model models/batchalign_final.mpk --classical [--json]
```

prints the corner errors per scene and method and the mean, median and worst per method.
It is built by the default `eval` feature (which only adds the aligner as a dependency;
the apps depend on this crate without default features and never see it).

## Status

Implemented and covered by the crate's tests: model + presets, loss, tiled
inference, planar bridge, discovery, runtime dispatch, the training
primitives (`train`, `checkpoint`), the datasets of all four strategies, the
trainer's command line and the strategy checkpoints. The `z-stackr-train` binary's
optimiser loops and the `stacker-nn-eval` binary (`src/bin/`) have no automated test;
what they call is tested. No model has been trained with the batch or alignment
strategies, so nothing here claims an accuracy. The
`wgpu` test target is only compiled in CI (`--no-run`), never run. The classical
Apex/Relief remain the defaults; the AI mode is additive and opt-in behind the
`nn` feature in the apps. No pre-trained weights ship with the repo: train your
own and drop the `.mpk`/`.json` pair into `models/`.
