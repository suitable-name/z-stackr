# z-stackr-nn

The neural focus-merge and alignment engine for **z-stackr** (crate identifier `stacker_nn`), built on the
[Burn](https://burn.dev/) deep-learning framework. It provides four training/inference strategies, tiled
inference, and the glue that lets a trained model act as a stacking/alignment algorithm alongside the
classical Apex and Relief.

## Four strategies

| Strategy | Trait | Built-in model | Built-in loss | Manifest tag |
|---|---|---|---|---|
| Pairwise fusion | `FusionModel` (`merge`) | `FocusMergeNet` | `FocusFusionLoss` | `focusmerge-v1` |
| Batch fusion | `FusionModel` (`fuse_batch`) | `BatchMergeNet` (v2) | `FocusBatchLoss` | `batchmerge-v2` |
| Batch alignment | `BatchAlignmentModel` | `BatchAlignNet` (v2) | `CornerAlignmentLoss` | `batchalign-v2` |
| Pairwise alignment | `PairAlignmentModel` | `FusionAlignNet` | `PairCornerAlignmentLoss` | `fusionalign-v1` |

### Pairwise fusion

Rather than ingesting the whole stack at once, `FocusMergeNet` performs a **recurrent
pairwise merge**: the running composite starts as frame 0 and each subsequent frame is
merged into it.

```text
frame_0 ─► (target) ─┐
                     ├─► FocusMergeNet ─► composite_1 ─┐
frame_1 ─► (source) ─┘                                 ├─► FocusMergeNet ─► composite_2 ─► …
                                        frame_2 ───────┘
```

Each step takes the current composite (`target` RGB + a per-pixel confidence
channel) and a new `source` frame, and predicts an updated composite, an updated
confidence, and the soft selection map used to blend them. The confidence
channel makes the recurrent accumulation order-robust. This streaming fold keeps
memory bounded and dovetails with the out-of-core pipeline.

### Batch fusion

`BatchMergeNet` (v2) instead ingests the entire focus stack `[N, S, 3, H, W]` at
once: a shared encoder extracts per-frame features, a shared sharpness branch
(mirroring `FocusMergeNet`'s) estimates per-frame per-pixel sharpness, a gating
head + softmax over the stack dimension predicts per-frame per-pixel blending
weights from `[features ‖ sharpness]`, and a max-pooled-feature refinement head
cleans up the composite. Simpler to train (no recurrence, no truncated BPTT) at
the cost of memory. The sharpness branch and `FocusBatchLoss`'s gate/edge
supervision terms (see "Training losses" below) exist so the softmax gate has
a cheap, direct way to localise the in-focus frame right at a depth
discontinuity, keeping halo risk there low.

**Memory note:** VRAM scales linearly with stack size. Tiled inference
(`infer::fuse_batch_tiled`) keeps at most one TILE's worth of every frame resident
at a time (not the whole stack at full resolution), but that is still `S × tile_h ×
tile_w` per tile — a batch model with a large stack or a large tile needs
proportionally more VRAM than the pairwise strategy, which only ever holds the
composite-so-far plus one source frame.

### Batch alignment

`BatchAlignNet` (v2) predicts one affine registration matrix per frame instead of a
fused image — the step that runs BEFORE fusion when a stack has slight misalignment
(focus breathing, camera shift). Unlike a per-frame-isolated model, it explicitly
compares each frame's features against the reference frame's via a local
correlation volume before the head ever sees them — alignment is a comparison, and
the network needs to see both frames to perform one. It
downscales the input stack (longest side to ~512px) before inference and predicts
matrices in a normalized `[-1, 1]²` coordinate space; `bridge::align_planar`
converts the result back to pixel-space matrices for the caller's actual frame
resolution. See that function's docs for the exact conjugation. Training data
comes from `scripts/03_simulate_misalignment.py`, which perturbs an aligned
dataset (from stage 2) with synthetic tilt/shift/scale and records ground-truth
matrices.

### Pairwise alignment

`FusionAlignNet` is the streaming sibling of `BatchAlignNet`: instead of
ingesting the whole stack at once, `align_pair(reference, frame)` registers ONE
frame against the reference per call, so a caller loops once per frame with
**O(1) memory in stack size** — the same streaming/low-memory relationship
`FocusMergeNet` has to `BatchMergeNet`, applied to alignment instead of fusion.
Architecturally it is a two-input twin of `BatchAlignNet`: identical
shared-weight strided encoder, L2-normalised local-correlation comparison, and
bounded 6-number geometric head (reusing those private building blocks
directly, not reimplementing them) — the one deliberate difference is that
`FusionAlignNet` has no reference-normalisation step (`BatchAlignNet`'s exact
frame-0-identity trick is only cheap because frame 0 lives inside the same
batched call). `bridge::align_planar_pairwise` is its
bridge entry point, sharing the identical pixel ⇄ normalized coordinate
contract `align_planar` uses. `crate::runtime::LoadedAlignModel` dispatches
between the two alignment architectures transparently by manifest tag, so
CLI/GUI/Python callers never need their own branch. `FusionAlignNet` trains on
the SAME `03_simulate_misalignment.py` scene data as `BatchAlignNet`, just
sliced into one reference/frame pair per training step
(`AlignSequence::get_pair`) instead of a whole stack.

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

`GroupNorm` only (never `BatchNorm`), so statistics never depend on tile or batch
size.

`BatchMergeNet` shares the stem/context/sharpness-branch/GroupNorm design but is
NOT size-preserving in the same "no global pooling" sense as `FocusMergeNet` for
its gating (it operates per-pixel across the stack dimension via softmax, not a
global pool) — it stays fully convolutional and tileable. Its gate input is
`[features ‖ sharpness]` per frame, exactly mirroring how `FocusMergeNet`'s gate
sees `[feat_t, feat_s, sharp_t, sharp_s, target_conf]`.

`BatchAlignNet` (v2) is different: its encoder is a fully-convolutional STRIDED
(x8 downsample) encoder shared across frames, followed by an L2-normalised local
correlation volume between each frame's features and the reference frame's — the
explicit comparison v1 never performed. A match head then reduces the
correlation-plus-features input to a per-frame feature vector, and an MLP predicts
6 raw geometric parameters (translation, rotation, per-axis log-scale, shear),
each passed through a scaled `tanh` bound before being composed into the affine
matrix — this guarantees a well-conditioned, non-degenerate matrix (positive
determinant, no reflections) at every training step. The `Linear` output layer is
zero-initialised, so an untrained network starts by predicting exactly zero
deltas, which compose to exactly the identity transform. Finally every frame's
matrix is left-composed with the inverse of frame 0's matrix (a differentiable,
analytic 2x2 inverse), forcing frame 0 to be EXACTLY the identity by construction.

### Size presets

| Preset | Channels | Context depth |
|--------|---------:|--------------:|
| `xs`   | 16       | 2 |
| `s`    | 24       | 3 |
| `m`    | 32       | 4 |
| `l`    | 48       | 6 |
| `xl`   | 64       | 9 |
| `xxl`  | 96       | 12 |

Build a config with `FocusMergeNetConfig::from_size(ModelSize::L)` (or the
`BatchMergeNetConfig`/`BatchAlignNetConfig`/`FusionAlignNetConfig` equivalents). A
trained model records its exact geometry in a sidecar manifest so it always
reloads correctly. `BatchAlignNetConfig::from_size` uses its own,
wider-and-shallower `(width, depth)` table (`xs=(32,1)` .. `xxl=(160,3)`),
distinct from the fusion presets above — see that method's docs.
`FusionAlignNetConfig::from_size` reuses the EXACT same table as
`BatchAlignNetConfig`'s (the two architectures are structurally analogous, just
applied to a pair vs. a whole stack).

## Training losses

`FocusFusionLoss` (pairwise) combines four masked terms:

* **Charbonnier** on RGB — robust reconstruction.
* **Multi-scale gradient L1** — preserves high frequency (the whole point).
* **Sharpness retention** — penalises output blurrier than the sharpest source.
* **Confidence supervision** — predicted confidence vs. the ground-truth in-focus
  coverage map.

All terms are down-weighted near depth edges via the scene's occlusion mask.

`FocusBatchLoss` (batch fusion) is brought up to the same standard as
`FocusFusionLoss`, with four analogous terms:

* **Charbonnier** on RGB vs. ground truth — robust reconstruction.
* **Multi-scale gradient L1** vs. ground truth — preserves high frequency.
* **Sharpness retention** — penalises the output being blurrier than the
  SHARPEST input frame at each pixel (max, over the stack dimension, of each
  frame's own Laplacian high-pass measure) — the batch adaptation of the
  pairwise term, which instead compares against a single `source` frame.
* **Gate supervision** — the batch analog of pairwise confidence supervision:
  builds a target blending distribution `target_alpha = masks / sum_S(masks)`
  from the per-frame in-focus masks and supervises the predicted softmax gate
  (`alpha`, from `BatchMergeNet::forward_with_alpha`) against it with L1, only
  where `sum_S(masks)` clears a small threshold (elsewhere no input frame
  claims to be in focus, so the correct gate is genuinely ambiguous and the
  term is masked out).

All four terms are down-weighted near depth edges via the scene's occlusion
mask, using the exact same weighting formula and default as `FocusFusionLoss`.

`CornerAlignmentLoss` (batch alignment) projects the four normalized-space image
corners through both the predicted and ground-truth matrices and computes MSE
between the projected points — scale-aware, since a rotation/shear error is
penalised in proportion to how far it actually displaces a corner.
`PairCornerAlignmentLoss` (pairwise alignment) is the identical technique over a
single `[N,3,3]` matrix pair, with no `S` (stack) dimension.

The recurrent rollout (`train::rollout_loss`) trains the pairwise fusion strategy
with **scheduled sampling** (periodically feeding the model's own detached
prediction as the next target) to fight exposure bias. `train::batch_loss`,
`train::align_loss`, and `train::fusion_align_loss` are single-shot
forward+loss calls (no recurrence to schedule).

## Feature / backend matrix

| Feature    | Backend                  | Use-case                         |
|------------|--------------------------|----------------------------------|
| `ndarray`  | `burn-ndarray` (CPU)     | default; CI, tests, CPU inference |
| `wgpu`     | `burn-wgpu` (Vulkan/MSL) | GPU inference (AMD/Intel/NVIDIA)  |
| `cuda`     | `burn-cuda` (CUDA)       | NVIDIA training (A100)            |
| `autodiff` | autodiff wrapper         | default; enables backprop         |

`burn::optim` (AdamW) and the file recorders are available once `burn/std` is on,
which every backend feature enables — so the trainer needs no special feature.
Several backends may be enabled at once (e.g. the GUI compiles `ndarray` + `wgpu`
to offer a runtime CPU/GPU choice); the scalar `SelectedBackend` alias resolves
by priority `cuda > wgpu > ndarray`, while runtime inference uses concrete
backend types.

## Extending with your own architecture

The built-in models are one implementation each of three small extensibility
traits, [`FusionModel`](src/traits.rs) (fusion),
[`BatchAlignmentModel`](src/traits.rs) (batch alignment), and
[`PairAlignmentModel`](src/traits.rs) (pairwise alignment) — none of them is
hard-wired into the tiled inference, feathered blending, planar bridge, or
recurrent fold. A third party can implement the relevant trait for their own
[Burn](https://burn.dev/) `Module` and reuse all of that machinery:

```rust
pub enum FusionStrategy {
    Pairwise,
    Batch,
}

pub trait FusionModel<B: Backend> {
    fn strategy(&self) -> FusionStrategy;

    fn merge(
        &self,
        target: Tensor<B, 4>,
        target_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
    ) -> MergeStep<B> { unimplemented!() }

    fn fuse_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4> { unimplemented!() }

    fn receptive_field(&self) -> usize;
}

pub trait BatchAlignmentModel<B: Backend> {
    fn align_batch(&self, stack: Tensor<B, 5>) -> Tensor<B, 4>;
}

pub trait PairAlignmentModel<B: Backend> {
    fn align_pair(&self, reference: Tensor<B, 4>, frame: Tensor<B, 4>) -> Tensor<B, 3>;
}
```

* `strategy` declares how a `FusionModel` processes data (streaming pairwise, or all-at-once batch).
* `merge` runs one recurrent pairwise-merge step (for `Pairwise` models) — same shape contract as
  `FocusMergeNet::forward` (`[N,3,H,W]` linear-RGB `target`/`source`/`merged`,
  `[N,1,H,W]` `target_conf`/`conf` in `0..1`), returning a minimal
  `MergeStep { merged, conf }`.
* `fuse_batch` runs an all-at-once merge for `Batch` models. **Warning**: Batch models require entire stacks of images (or, per-tile, entire stacks of tile-sized crops) to be loaded into VRAM. Memory scales linearly with the stack size. Large models or large data amounts will use a lot of VRAM and may require extremely small tiles or a very powerful GPU!
* `receptive_field` reports how far spatial context propagates through your
  convolutions (in pixels). It drives the tile overlap the tiled inference
  must use to stay seam-free — use `TileConfig::for_model(&your_model)`
  instead of hard-coding an overlap.
* `align_batch` predicts one affine matrix per frame in the stack, in the
  normalized `[-1, 1]²` coordinate convention (see `bridge::align_planar`'s docs).
* `align_pair` predicts one affine matrix registering `frame` against
  `reference`, called once per frame (streaming, O(1) memory in stack size)
  instead of batched over the whole stack — same normalized `[-1, 1]²`
  convention as `align_batch` (see `bridge::align_planar_pairwise`'s docs).

A trivial illustrative fusion model — averages `target` and `source` with no spatial
mixing at all, so its receptive field is `0`:

```rust
use burn::prelude::*;
use stacker_nn::{FusionModel, FusionStrategy, traits::MergeStep};

struct AverageMerge;

impl<B: Backend> FusionModel<B> for AverageMerge {
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::Pairwise
    }

    fn merge(
        &self,
        target: Tensor<B, 4>,
        target_conf: Tensor<B, 4>,
        source: Tensor<B, 4>,
    ) -> MergeStep<B> {
        let merged = target.mul_scalar(0.5).add(source.mul_scalar(0.5));
        MergeStep { merged, conf: target_conf } // pass confidence through unchanged
    }

    fn receptive_field(&self) -> usize {
        0 // purely pointwise — no pixel depends on its neighbours
    }
}
```

Run it through the same tiled inference and planar bridge `FocusMergeNet`
uses:

```rust
let model = AverageMerge;
let cfg = stacker_nn::infer::TileConfig::for_model(&model);
let fused = stacker_nn::bridge::fuse_planar(&model, &frames, cfg, &device)?;
```

What stays specific to the built-in architectures: weight loading/discovery
(`discovery`, `runtime::LoadedModel`, `runtime::LoadedAlignModel`) is tied to their
`.mpk`/manifest formats — a custom architecture brings its own checkpoint format
and constructs itself directly. `discovery::ModelManifest` carries an
`architecture` tag (one of `"focusmerge-v1"`, `"batchmerge-v2"`, `"batchalign-v2"`,
`"fusionalign-v1"`, default `"focusmerge-v1"`) precisely so a foreign manifest
sitting in the same `models/` directory is skipped rather than misloaded, and so
the four built-in architectures never get cross-loaded into the wrong parameter
layout (each `ModelEntry::load*` method validates the tag before deserialising).
Training (`train::rollout_loss` / `train::batch_loss` / `train::align_loss` /
`train::fusion_align_loss`) is also left concrete over the built-in models — see
that module's doc comment for why — so a custom architecture trains with its own
loop and only needs `FusionModel` / `BatchAlignmentModel` / `PairAlignmentModel`
to reuse this crate's *inference*-side machinery afterwards.

## Public API (app-facing)

The CLI/GUI never touch Burn directly:

* `discover_default_models()` / `discover_models(dir)` → `Vec<ModelEntry>` from a
  `models/` directory (`<name>.mpk` + `<name>.json`), covering all four
  architectures. Use `ModelEntry::is_fusion()` / `ModelEntry::is_alignment()` to
  filter for a specific picker (or the finer-grained
  `ModelEntry::is_batch_alignment()` / `ModelEntry::is_pairwise_alignment()` when
  a caller specifically needs to know which alignment architecture it found).
* `available_devices()` / `gpu_available()` — what backends were compiled in.
* `LoadedModel::load(entry, device)` then `.fuse(frames, tile)` /
  `.fuse_with_progress(frames, tile, on_step)` — fuse a stack of
  `PlanarImage<f32>` (the pipeline's colour type) with either fusion architecture,
  with an optional progress callback for live preview. **Colour-space boundary:**
  the pipeline's planes are gamma-encoded YCbCr, while the network operates on —
  and was trained with — **linear RGB**; the `bridge` module owns that conversion
  in both directions (`data::to_linear` / `from_linear`, the exact functions the
  training loader uses), so the model never sees gamma-encoded values.
* `LoadedAlignModel::load(entry, device)` then `.align(frames)` → one pixel-space
  affine matrix per frame, regardless of which alignment architecture (batch or
  pairwise) the entry names — `LoadedAlignModel` dispatches internally.
* `fuse_entry(entry, device, frames, tile)` — one-shot convenience for fusion.

Lower-level building blocks are also public: `model`, `loss`, `infer`
(`fuse_stack` / tiling), `bridge` (planar↔tensor, `fuse_planar`/`align_planar`/
`align_planar_pairwise`), `data` (datasets + sample construction for all four
strategies), `train` (loss functions + schedules), `traits` (the `FusionModel` /
`BatchAlignmentModel` / `PairAlignmentModel` extensibility traits — see
"Extending with your own architecture" above).

## Training binary

`z-stackr-train` (built when the `autodiff` backend is on, which is the
default) is the optimiser loop: AdamW, cosine LR with warm-up, crash-safe rolling
checkpoints, and `--resume`, for all four strategies via `--strategy
pairwise|batch|align|fusion-align`. Pairwise fusion additionally gets scheduled
sampling. `fusion-align` trains on the SAME `--data` scenes as `align`, one
reference/frame pair per optimiser step. See the project README's "Training the
AI model" section for usage, and this crate's `src/bin/train.rs` module docs for
the exact checkpoint naming scheme (each strategy writes its own
`<prefix>_latest`/`<prefix>_epoch_NNN`/`<prefix>_final` stems so concurrent runs
don't clobber each other).

## Status

Implemented and tested (models + presets, losses, tiled inference, planar bridge,
alignment bridge, discovery, runtime dispatch, trainer) for all four strategies.
The classical Apex/Relief remain the defaults; the AI mode is additive and opt-in
behind the `nn` feature in the apps. No pre-trained weights ship with the repo —
train your own and drop the `.mpk`/`.json` pair into `models/`.

Tiled inference, blending, the planar bridge, and the recurrent fold are generic
over the `FusionModel` trait (alignment over `BatchAlignmentModel` /
`PairAlignmentModel`), so third-party architectures can reuse them without
modifying this crate — see "Extending with your own architecture" above. Weight
discovery/loading (`discovery`, `runtime`) and the differentiable training losses
(`train`) remain specific to the built-in models.
