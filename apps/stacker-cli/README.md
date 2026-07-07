# z-stackr-cli

The official command-line interface for **z-stackr** (package `z-stackr-cli`, binary `z-stackr`).

`z-stackr-cli` is a high-performance executable that orchestrates the entire focus stacking pipeline from end to end. Designed for automation, batch processing, and integration into existing photographic workflows, it provides direct access to the `z-stackr-align` and `z-stackr-algo` engines.

## Key Capabilities

* **Out-of-Core Processing**: Powered by `z-stackr-core`'s `TileManager`, the CLI streams image tiles from disk into memory, guaranteeing low and deterministic peak RAM usage. You can effortlessly stack hundreds of 50-Megapixel images without exceeding memory limits.
* **Format Flexibility**: Reads `.jpg`, `.png`, and `.tif` files. Fully preserves depth by supporting 16-bit ingestion and natively writing 16-bit output. With `--features raw`, also reads camera RAW files (CR2/CR3/NEF/ARW/DNG/RAF/ORF/RW2/PEF/…, decoded via pure-Rust `rawler` — see the project README's "RAW Image Support" section) with no change to the pure-Rust build chain.
* **Deterministic Alignment**: Runs the same coarse-to-fine, always-bounded subpixel refinement (shared with the GUI via `stacker_align::pipeline::align_frame` — never duplicated per app) against a selected reference frame. Alignment mode is configured via `settings.toml`'s `alignment_mode` as a strict DOF ladder — `translation` ⊂ `registration` (default) ⊂ `affine`: `translation` (shift + focus-breathing scale, rotation held fixed), `registration` (shift + uniform scale + rotation — a similarity transform; **the default**), or `affine` (everything `registration` solves plus separate X/Y scale and shear — a true 6-DOF affine solve), or `none` (skip alignment). All three active classical modes share whichever **optimiser** is selected — `refine_alignment_registration` (Nelder-Mead bounded simplex) or the newer `refine_alignment_lk` (pyramid Lucas-Kanade / Gauss-Newton) — differing only in which degrees of freedom are solved, and every refined result passes a post-refinement gate that falls back to identity if refinement scored worse than doing nothing.
* **Optimizer selection**: `settings.toml`'s `optimizer` field (`"auto"` default, `"lucaskanade"`, or `"neldermead"`) or the `--optimizer <auto|lk|nm>` CLI flag (which overrides the config value when given) chooses which optimiser solves the alignment above. `auto` tries Lucas-Kanade first and falls back to Nelder-Mead on error or RMS regression; `lk`/`nm` force one optimiser unconditionally, with no fallback. See the project README's "Advanced Alignment Engine" section.
* **Neural Alignment (`alignment_mode = "neural"`, requires `--features nn`, experimental)**: Replaces the classical pipeline entirely with a trained `batchalign-v2` model, run once over the whole (internally-downscaled) stack via `stacker_pipeline::align::compute_neural_alignment`. No classical refinement runs on top — the model's matrix is used directly. Select the alignment model with `--align-model` (separate from the fusion `--model` flag — the two are never interchangeable). Unlike the classical alignment pre-pass (which loads one frame at a time), this loads the whole stack once to run a single batch inference call, trading the pipeline's usual one-frame-at-a-time memory bound for a single-pass neural registration.
* **Per-Frame Brightness Correction**: `correct_brightness` (config-TOML, default `true`) applies per-frame brightness/gamma correction — see the project README's GUI Settings section for the algorithm; it applies identically in the CLI via the same shared `align_frame` call.
* **Common-Area Crop (default) / Full-Canvas Output**: Edge-clamped warping fills the frame edges from the last valid pixel rather than leaving a black border, but that replicated band doesn't reflect real scene data. `crop_to_common_area` (config-TOML, default `true`) crops the stacked output to the largest rectangle covered by every aligned frame — removing that smeared band from both the saved output and the fusion input (tiled fusion allocates the output buffer at the cropped size and skips tiles outside it entirely; whole-image Apex fuses full-canvas and crops once at save time) — and is skipped (falls back to full canvas, with a warning logged) if a rogue/misaligned frame would shrink the crop below 25% of the canvas area. Set `crop_to_common_area = false` to always save the full original canvas with no cropping. `resize_cropped_to_original` (config-TOML, default `false`) additionally resamples that cropped output back up to the original canvas resolution (Lanczos3, edge-clamped) instead of saving the smaller cropped size — a slight non-uniform stretch, since the crop rectangle's aspect ratio can differ fractionally from the canvas; it's ignored unless `crop_to_common_area = true` actually shrank the output.
* **Feature-Gated Pre-Alignment**: When compiled with `--features akaze`, the CLI performs a highly robust AKAZE-based feature matching pass and RANSAC transformation prior to subpixel refinement.
* **Optional AI ("neural") Stacking**: When compiled with `--features nn` (CPU) or `--features nn-gpu` (GPU), a third fusion mode `ai` runs a trained `FocusMergeNet` from the `models/` folder. Without these features the CLI has **zero ML dependencies** and the `ai` mode is rejected.
* **EXIF Metadata Copy**: When `image_saving.copy_metadata = true`, the raw EXIF blob from the first input frame is re-injected into the output via the shared `stacker_core::metadata` module — identical logic to the GUI. Works for JPEG/PNG output; TIFF output logs a warning and is skipped (see the project README's "Image Saving" section).

## Usage

```bash
z-stackr \
    --input-dir <PATH> \
    --output-file <PATH> \
    --mode <MODE> \
    --tile-size <SIZE> \
    [--log-file <PATH>]
```

### Arguments

* `--input-dir` (Required): Directory path containing your bracketed image sequence (`.jpg`/`.png`/`.tif`/`.tiff`, plus camera RAW formats in `--features raw` builds — see the project README's "RAW Image Support" section). Files are sorted alphabetically.
* `--output-file` (Required): Path and filename for the final stacked output. Using `.tif` or `.png` automatically preserves 16-bit depth (if supported). **In subfolder-batch mode** (see "Batch processing" below), this is instead interpreted as an output *directory* (created if it doesn't exist) — passing a path with a recognised image extension in that mode is a clear error rather than a confusingly-named file.
* `--mode` (Required): Focus fusion algorithm. Select `apex` for Laplacian Pyramids (ideal for intersecting subjects), `relief` for Guided Filter Depth Maps (ideal for smooth surfaces and low noise), `strata` for Guided-Filter soft-blend fusion (edge-aware blending of every frame at every pixel — fewer halos at depth edges than a hard per-pixel/per-band pick), or `ai` for the learned neural model (requires an `nn`/`nn-gpu` build).
* `--tile-size` (Required): Processing tile size. We recommend `512` for an optimal balance between memory usage and multi-threaded throughput.
* `--model` (AI mode): Name (file stem) of the fusion model in the `models/` folder to use. Defaults to the first fusion model found.
* `--device` (AI mode / neural alignment): `cpu` or `gpu`. Defaults to the best available backend in the build.
* `--align-model` (neural alignment, `alignment_mode = "neural"` in the config): Name (file stem) of the alignment model in the `models/` folder to use. Defaults to the first alignment model found. Independent of `--model` — never interchangeable with a fusion checkpoint.
* `--optimizer <auto|lk|nm>` (Optional): Overrides the config file's `optimizer` setting for this run. `auto` (default) tries pyramid Lucas-Kanade first, falling back to Nelder-Mead on error or RMS regression; `lk` forces Lucas-Kanade only (no fallback); `nm` forces the original Nelder-Mead bounded simplex only.
* `--log-file` (Optional): Path to write comprehensive structured execution traces and profiling statistics.
* `--config <FILE>` (Optional): Path to a TOML settings file. If the file doesn't exist yet, it's created with commented defaults and the run proceeds with those defaults; if it exists, its values override the built-in defaults. **Most stacking settings — alignment DOF gating nuances aside from `--mode`, `correct_brightness`, `relief_use_multigrid` and the other Relief tuning knobs, `auto_cull` (see caveat below), `auto_cull_threshold_pct` (GUI-only effect — see caveat below), `crop_to_common_area`, preprocessing (rotation/crop/resize/reverse-sort), and image-saving options — are config-TOML-only; there are no dedicated argv flags for them.** This is intentional, not an oversight: no `StackingSettings` field has a dedicated CLI flag, so adding flags for a couple of settings would be an inconsistency, not a fix.
  * `crop_to_common_area` (default `true`, `0.1`–`5.0` n/a — boolean): see "Common-Area Crop" above. The CLI honours this field directly (unlike `auto_cull`).
  * `resize_cropped_to_original` (default `false` — boolean): see "Common-Area Crop" above. Only has an effect when `crop_to_common_area = true` actually cropped the output.
  * `auto_cull_threshold_pct` (default `2.0`, range `0.1`–`5.0`): the minimum percentage of the scene's in-focus detail pixels a frame must win to survive Auto-Cull. Since the CLI doesn't implement `auto_cull` at all (see below), this field has no effect on `z-stackr-cli` runs — it exists in `StackingSettings` for the GUI and simply round-trips through CLI config files unused.
* `--monitor` (Optional flag): Watches `--input-dir` for new/changed files and re-runs the full stacking pipeline automatically on every change, in addition to an initial pass at startup. Useful for a tethered-capture workflow where frames land in the folder as they're shot. **Incompatible with subfolder-batch mode** (see "Batch processing" below) — watching N subfolders at once is future work; combining the two is a hard error.
* `--stacks <subfolders|single>` (Optional): Only relevant when `--input-dir` contains image-bearing subfolders (see "Batch processing" below). `subfolders` stacks each image-bearing direct subfolder of `--input-dir` independently, sequentially, one output file per subfolder; `single` stacks only the images directly inside `--input-dir`, ignoring any subfolders (today's behaviour). If `--input-dir` has no image-bearing subfolders, this flag has no effect. If it does and this flag is omitted, the CLI prompts interactively when stdin is a TTY, or exits with an error asking you to pass this flag when it isn't (e.g. in a script or CI job).

> **Models folder.** `ai` mode and neural alignment discover models in a `models/`
> directory next to the executable (and the current working directory). Each
> model is a `<name>.mpk` weights file plus a `<name>.json` architecture
> manifest, produced by `z-stackr-train` (see the project README). Fusion and
> alignment checkpoints can share the same folder — each of `--model` and
> `--align-model` is filtered to its own kind (`ModelEntry::is_fusion` /
> `ModelEntry::is_alignment`), so a fusion checkpoint never shows up as an
> alignment-model candidate and vice versa.

> **`auto_cull` and `sort_by_sharpness` are not implemented in the CLI.** The GUI loads and aligns the
> entire stack into RAM up front and only then compares every frame's
> contribution; the CLI is deliberately out-of-core (memory scales with tile
> size, not frame count) and streams one frame at a time, which the current
> culling and sorting algorithm can't do without abandoning that memory guarantee. If
> `auto_cull = true` or `sort_by_sharpness = true` is set in your config, the CLI logs a startup warning and
> stacks every frame anyway — it will not silently drop or reorder frames.

## Examples

Standard stack using Apex:
```bash
z-stackr --input-dir ~/Pictures/Macro/Stack01 --output-file ~/Pictures/Macro/Stack01_Final.tif --mode apex --tile-size 512
```

Stack using Relief for a continuous surface, saving trace logs:
```bash
z-stackr --input-dir ~/Pictures/Coins/Stack02 --output-file ~/Pictures/Coins/Final.png --mode relief --tile-size 1024 --log-file ~/stack_log.txt
```

Stack using Strata for soft, edge-aware blending:
```bash
z-stackr --input-dir ~/Pictures/Macro/Stack03 --output-file ~/Pictures/Macro/Stack03_Final.tif --mode strata --tile-size 512
```

## Batch processing

`--input-dir` can point at more than a flat folder of frames. When the CLI scans it and finds subfolders that themselves contain images, it needs to know which of two things you mean:

1. **Each subfolder is its own stack** — e.g. `Session01/` contains `rock_a/`, `rock_b/`, `rock_c/`, each holding one bracketed sequence, and you want three separate stacked outputs in one command.
2. **Only the images directly in the pointed-to folder are the target** — the subfolders are incidental (thumbnails, a `raw/` backup copy, whatever) and should be ignored, exactly like today.

The CLI never silently guesses. It classifies `--input-dir` into one of three shapes:

* **Images only** (no subfolder contains images): the plain case — one stack, one output file at `--output-file`.
* **Subfolders with images, no direct images**, or **mixed** (both direct images and image-bearing subfolders): the CLI asks. The prompt (and the equivalent non-interactive error) states the exact counts, e.g. *"'--input-dir' contains 3 subfolders with images and 12 images directly in the folder."*

A "subfolder with images" means a **direct child directory** whose own **direct children** include at least one recognised image (standard formats always; camera RAW extensions in `--features raw` builds — the same extension rule `--input-dir` itself is scanned with, so RAW support is picked up automatically with no separate list to maintain). This check is **not recursive beyond one level**: a subfolder's own subfolders are never inspected, and a subfolder with zero direct images is silently skipped even if something nested deeper inside it has images.

### Answering the question

* **Interactively**: if stdin is a terminal, the CLI prints the counts and prompts for `s` (each subfolder is its own stack) or `d` (only the direct images):
  ```bash
  z-stackr --input-dir ~/Pictures/Macro/Session01 --output-file ~/Pictures/Macro/Session01_out --mode apex --tile-size 512
  # '--input-dir' contains 3 subfolders with images and 0 images directly in the folder.
  # Stack each subfolder independently, or only the images directly in this folder?
  #   [s] each subfolder is its own stack
  #   [d] only the direct images in this folder
  # (pass --stacks subfolders / --stacks single to skip this prompt next time)
  # > s
  ```
* **Non-interactively** (scripts, CI, cron, no TTY on stdin): pass `--stacks subfolders` or `--stacks single` up front. `--stacks` also skips the prompt on a TTY, so it's the right choice for anything you intend to run unattended more than once.
  ```bash
  z-stackr --input-dir ~/Pictures/Macro/Session01 --output-file ~/Pictures/Macro/Session01_out --mode apex --tile-size 512 --stacks subfolders
  ```
  Omitting `--stacks` with no TTY available (and image-bearing subfolders present) is a hard error telling you to pass it — the CLI will not guess in a non-interactive context.

### Batch outputs

In subfolder-batch mode, `--output-file` is reinterpreted as an **output directory** (created automatically if it doesn't exist yet). Passing a path with a recognised image extension (e.g. `.tif` — something that looks like a single output *file*) is rejected with a clear error, since batch mode needs a directory to place multiple results into.

Each subfolder's output filename is derived exactly the way the GUI's Save dialog derives filenames from your `image_saving` settings (the same config-TOML `--config` loads — see "Configuration flow" below): the configured `filename_template` with `{name}` substituted by **the subfolder's own directory name**, plus the extension for the configured `output_format`. For example, with the default `filename_template = "{name}_stacked"` and `output_format = "tiff"`, a subfolder named `rock_a/` produces `rock_a_stacked.tiff` inside the output directory.

### Execution and failure handling

Subfolders are stacked **sequentially, never concurrently** — the pipeline itself is already internally parallel (multi-threaded tiled fusion), and running several stacks at once would multiply peak memory per concurrent stack, defeating the whole point of the out-of-core design. Each subfolder's progress bars and log lines are prefixed with a `=== [subfolder_name] ... ===` header so a long batch run's console output stays legible.

A failing subfolder (bad images, disk error, whatever) is logged and **does not abort the batch** — every remaining subfolder still runs. At the end, a summary table lists every subfolder's outcome:
```
Batch summary (3 stacks):
  [ok]     rock_a -> /home/user/Pictures/Macro/Session01_out/rock_a_stacked.tiff
  [FAILED] rock_b -> no images found in '.../rock_b'
  [ok]     rock_c -> /home/user/Pictures/Macro/Session01_out/rock_c_stacked.tiff
```
The process exits nonzero if any subfolder failed, so batch runs remain script/CI-friendly (`$?` reflects overall success) even though no single failure stops the run early.

`--monitor` is **incompatible with subfolder-batch mode** and is a hard error if combined with it — watching and re-stacking N subfolders live is future work, not implemented today.

### Configuration flow: GUI-configured, CLI-run

`--stacks` and batch discovery layer on top of the CLI's existing `--config` flow — the CLI loads the exact same TOML config format the GUI writes from its Settings panel. The intended workflow for a batch job is:

1. Open `z-stackr-gui`, dial in alignment mode, fusion algorithm, Relief/Apex tuning, preprocessing, and **Image Saving** (filename template, output format, bit depth) against one representative stack, then save the config from the GUI.
2. Point the CLI at that same config file and a parent folder full of per-subject subfolders:
   ```bash
   z-stackr --config ~/.config/z-stackr/settings.toml \
       --input-dir ~/Pictures/Macro/Session01 \
       --output-file ~/Pictures/Macro/Session01_out \
       --mode apex --tile-size 512 --stacks subfolders
   ```
   Every subfolder is stacked with the exact settings the GUI run validated, including the filename template used to name each result.

## Building

```bash
# Build standard version (binary: z-stackr)
cargo build --release -p z-stackr-cli

# Build with AKAZE feature matching (recommended for handheld stacks)
cargo build --release -p z-stackr-cli --features akaze

# Build with the AI ("neural") stacking mode — CPU inference
cargo build --release -p z-stackr-cli --features nn

# …or with GPU (wgpu/Vulkan) inference (implies `nn`)
cargo build --release -p z-stackr-cli --features nn-gpu

# Build with pure-Rust camera RAW support (CR2/CR3/NEF/ARW/DNG/RAF/ORF/RW2/PEF/…)
cargo build --release -p z-stackr-cli --features raw
```
