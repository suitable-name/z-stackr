# Changelog

All notable changes to **z-stackr** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Renamed

The fusion algorithms take the names of the GitHub release. Everything that wrote the
old names still reads.

- **PMax is now Apex, DMap is now Relief** in code, settings, CLI, GUI and docs. The
  GUI algorithm list reads "Relief (Depth Map)", "Apex (Laplacian)", "Strata (Guided
  Fusion)", "Both (Relief & Apex)", "All Three (Strata, Relief & Apex)" and, in `nn`
  builds, "AI Model (Neural)".
- **Legacy names still accepted when reading:** the settings keys `dmap_estimation_radius`,
  `dmap_smoothing_radius`, `dmap_contrast_pct`, `dmap_show_preview` and
  `dmap_auto_detect` (now `relief_*`); the CLI modes `pmax` and `dmap` (`both` is still a mode: Relief and Apex); the
  GUI combo box and project-file labels "PMax (Laplacian)", "DMap (Depth Map)" and
  "Both (DMap & PMax)"; the preferences hint id `dmap-preview`; the GitHub key
  `use_gpu` (read as `compute_device`; the CLI flag `--no-gpu`, in builds with a GPU
  feature, keeps the run on the CPU);
  `relief_engine = "multigrid" | "guided_filter"` (read as `relief_use_multigrid`).
  Saving always writes the new names.
- **The `{algo}` value of file-name templates changes** from `PMax` / `DMap` to
  `Apex` / `Relief` (and `Strata`). A template that contains `{algo}` names its files
  differently than before; the CLI and GUI exports of an earlier build are not
  overwritten.
- **Settings migration, `settings_version = 2`.** Settings files now carry
  `settings_version = 2`. A file without it is a legacy file and is migrated when
  read: `alignment_mode = "affine"` becomes `registration` (the same 4-parameter
  pyramid search; see "Changed"), the old single-pass `registration` becomes
  `registration` with `bounded_search = true`, `alignment_mode = "bounded"` becomes
  `registration` with `bounded_search = true`, and `translation` stays `translation`.
  Default results are unchanged because the default mode is still the same search.
  The settings now have 39 keys (it was 26).

### Added

- **Strata**, a third fusion engine: a guided-filter soft blend where every frame
  contributes to every pixel, weighted by its saliency, with the weights following
  the image's edges. Settings `strata_base_radius` (8 to 64, default 31) and
  `strata_detail_focus` (1 to 5, default 3). CLI `--mode strata` and `--mode all`
  (Strata, Relief and Apex), a Strata tab and an "All Three" algorithm in the GUI, a
  worker pass in the browser. Tile apron 101 px, memory independent of the frame
  count. GPU kernels for the saliency and the guided-filter pair in `stacker-gpu`.
- **Relief multigrid** (`relief_use_multigrid`, off by default): the frame index is
  solved as a smooth depth surface on whole-image planes, so the result does not
  depend on the tile size. Whole-image GPU kernel in `stacker-gpu`.
- **Alignment optimizer** (`optimizer = "auto" | "lucaskanade" | "neldermead"`): a
  Lucas-Kanade (Gauss-Newton) solver beside the Nelder-Mead simplex. `auto` runs
  Lucas-Kanade and falls back to the simplex when it fails or ends no better. CPU
  only; the browser build uses the simplex (`lucaskanade` and `auto` mean
  Nelder-Mead there). CLI `--optimizer auto|lk|nm` (the settings file also reads
  `lk` and `nm`).
- **Alignment models take the GitHub meanings:** `translation` = shift and uniform
  scale, `registration` = shift, scale and rotation, `affine` = the full 6-parameter
  model (adds aspect ratio and shear, always bounded).
- **Brightness correction** (`correct_brightness`, on by default): each aligned frame
  is matched to the reference in linear light (scale and a mild gamma fitted to the
  mean and standard deviation over the covered pixels).
- **Chain alignment** (`chain_alignment`, off by default): each frame is registered
  to its predecessor and the steps are composed; the coverage mask comes from the
  cumulative matrix.
- **Neural alignment** (`alignment_mode = "neural"`, `neural_refine_classically`):
  a trained network predicts matrices on thumbnails of at most 512 px; they seed the
  classical refinement or are applied directly. Native only (`nn` builds), CLI
  `--align-model`, GUI alignment-model picker. No trained model ships.
- **More neural models** in `stacker-nn`: `BatchMergeNet` (whole-stack fusion),
  `BatchAlignNet` and `FusionAlignNet` (alignment) with corner and photometric
  losses, a manifest `architecture` tag (manifests without one are `FocusMergeNet`),
  separate fusion and alignment model discovery, and the trainer options
  `--strategy pairwise|batch|align|fusion-align`, `--corr-radius`, `--real-data`
  and `--photometric-weight`. All networks use the per-pixel channel norm and work in
  linear light.
- **`stacker-nn-eval`**, the inference binary behind `scripts/04_eval_alignment.py`
  (JSON on stdin, or `--scenes-dir` directly). The script now also runs the classical
  refiner alone as its baseline.
- **Scripts** `03_simulate_misalignment.py` (unaligned, photometrically perturbed
  training scenes with ground-truth matrices) and `04_eval_alignment.py`.
- **RAW input** (`raw` feature, native only): CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2,
  PEF and other formats through `rawler`, developed with the camera white balance and
  color matrix.
- **EXIF copy** (`copy_metadata`, which used to be accepted and ignored): the EXIF
  block of the source frames is spliced into JPEG and PNG outputs; the orientation
  tag is set to upright when the pixels were rotated. TIFF output cannot carry it
  and a warning says so.
- **Python bindings** (`crates/stacker-python`, module `zstackr`, a maturin project
  outside the Cargo workspace) for the in-memory and the out-of-core paths.
- **`stacker_pipeline::tiled`**, the callback-driven out-of-core runner extracted from
  the CLI, now used by the CLI (including the batch mode `--stacks`), the GUI
  (`tile_size`) and the Python bindings. The CLI modes `both` and `all` write one
  file per algorithm, `<stem>_<algo>.<ext>`; `--stacks subfolders|single` chooses
  between stacking every image-bearing subfolder and only the top folder.
- **Frame selection:** `auto_cull_threshold_pct` (0 = the automatic rule),
  `sort_by_sharpness`, and standalone Sort and Cull actions in the GUI.
- **Crop options:** `crop_to_common_area` (on) and `resize_cropped_to_original` (off).
- **Retouch** keeps its strokes across project save and open: each stroke (source
  frame, points, radius, opacity) is stored in the project file and replayed on
  open. The undo history is not stored.
- **Build tasks:** `docs/vscode-tasks.json` (CPU-level matrix for the GUI, CLI and
  trainer on Windows and Linux; PGO chains, and BOLT on Linux) with `docs/build-tasks.md`,
  a PGO workload (`scripts/pgo-workload.sh` / `.ps1`, settings in `scripts/pgo/`) that
  covers Apex, Relief (with and without multigrid) and Strata with registration +
  Lucas-Kanade, registration + Nelder-Mead and affine alignment, and the
  `release-perf` cargo profile (release code, line tables and symbols kept).
- **Web app** (`apps/stacker-web`, `apps/stacker-web-worker`, `crates/stacker-web-core`):
  the desktop interface in the browser, built with Trunk. The stack runs in a Web
  Worker with a 2 GiB memory budget; a stack that does not fit is offered a reduced
  Pre-resize percentage. Files stay local, nothing is uploaded.
- **Browser alignment is GPU-only (WebGPU).** A browser without WebGPU offers to stack
  without alignment instead.
- **Async GPU APIs** in `stacker-gpu` and the ask/tell `AlignSearch` in `stacker-align`,
  bit-identical to the synchronous refiners.
- **`session` module** in `stacker-pipeline`, shared by the desktop runner and the web
  worker.
- **wasm32 FMA helpers** (`stacker_core::fma`): fused multiply-add on native targets,
  unfused `a * b + c` on wasm32, where `mul_add` is a software routine.
- **App icons** for the GUI and CLI executables, the window and the web favicon
  (`scripts/make-icons.py`).
- **UI harness for the desktop app** (`cargo test -p z-stackr-gui --features ui-harness
  --test ui`): the real window under slint-harness, with a geometry sweep at two
  sizes, a click-everything pass, a coverage gate over the `.slint` files and scenarios
  for opening frames, every stack engine, the dialogs, projects and settings, all in a
  throw-away profile with the native file dialogs answered from a queue. The scenarios
  run in parallel worker processes while cores are idle (`SLINT_HARNESS_JOBS=serial`
  for one process).

### Changed

Behaviour changes you will notice (read these first if you used an earlier build):

- **Default alignment mode is `registration`, which is the old `affine`.** The names
  moved to the GitHub meanings (see "Added"); an old file is migrated, so the same
  search runs. The mode that is now called `affine` is the new 6-parameter model.
- **Brightness correction is on by default.** Fused results of stacks with exposure
  drift differ from earlier builds. Set `correct_brightness = false` for the old
  behaviour.
- **Cull survivors fuse in stack order** unless `sort_by_sharpness` is on. Before, the
  frames kept by the auto-cull were fused in the census order (the nearest-neighbour
  walk). The multigrid depth needs capture order.
- **Nelder-Mead schedule and polish.** The simplex spends 150 iterations on each
  coarse pyramid level, 3 on the large levels near full resolution, and finishes with
  a short full-resolution polish (40 iterations). Alignment results change slightly,
  and are faster.
- **Identity gate.** A refined matrix that fits strictly worse than the identity on a
  reduced copy of the frames is replaced by the identity.
- **RANSAC inlier floor.** The seed fit (and the breathing estimate) must be agreed
  on by at least a fifth of the matches (at least the minimal sample plus 2, never
  more than 50 required).
- **Relief auto-detect uses the median plus 1.5 times the median absolute deviation**
  of the focus map instead of Otsu's method, which collapses on heavy-tailed maps. The
  percentile semantics of `relief_contrast_pct` are unchanged. The guided filter
  clamps its variance and builds its summed-area tables in parallel (a golden may
  move by one unit in the last place).
- The Settings dialog is organised in tabs (General, Apex, Relief, Strata, Input,
  Output).
- Debug builds no longer print panics at exit: the wgpu instance flags now come from the
  environment only.

Performance, with the output unchanged unless stated above:

- The alignment schedule above, plus Lucas-Kanade as the default optimiser path.
- A streaming Apex fold in `stacker-algo` (`ApexFold`): frames are folded in one at a
  time with a persistent best-energy plane, equal to the batch fusion, with memory that
  does not grow with the frame count; the grit energy of each source is precomputed.
- The Strata saliency blurs one plane with ping-pong scratch instead of three.
- Row-wise `paste_interior`; the 16-bit sRGB encoder uses an exact table plus a
  correction instead of one `powf` per sample; the brightness correction is applied in
  parallel.
- Frame decode and AKAZE seeding look ahead in the alignment (bounded, native only,
  in order); the CLI prefetches the next tile's frames while the current tile is fused.
- GPU kernels for Strata and the Relief multigrid solver, with CPU fallbacks and
  parity tests.

### Fixed

- **Alignment of series with strong focus breathing.** Lucas-Kanade (the first pass of
  the default `auto` optimizer) and the `affine` simplex searched the narrow bounded-search
  windows even with Bounded Search off, so no frame could get more than 5 % scale from the
  reference; on a series that breathes more (13.5 % on the sample frames) the late frames
  stayed misaligned and the result showed echoed edges. With Bounded Search off they now
  search GitHub's windows (shift and scale 20 %, rotation 20 degrees).
- The WGSL `finite()` helper uses the hex float literal, because browsers' WGSL
  compilers reject `3.4028235e38`.
- The multigrid solver of the GitHub release restricted its corners wrongly and
  prolonged into the wrong corner; both are corrected in the port.
- The GitHub chain alignment derived its coverage mask from the step's matrix, which
  under-estimated the shift and let a border of undefined pixels into the common area;
  the mask now comes from the cumulative matrix.
- The alignment-crop normalisation of the neural alignment code is fixed (tested with
  a known shift).

## [1.0.0] - 2026-10-01

First stable release. The project was renamed to **z-stackr** (Cargo packages
`z-stackr-*`; in-code crate identifiers remain `stacker_core`, `stacker_algo`, …;
binaries `z-stackr`, `z-stackr-gui`, `z-stackr-train`). The entry has two parts:
the behaviour changes of the review and fix campaign that closed on 2026-10-01
(read these first if you used an earlier build), and the features the release
introduced.

### Behaviour changes you will notice

- **Alignment direction.** The refined matrix maps reference coordinates to source
  coordinates, and both apps used to warp the frame with it instead of its
  inverse, so a frame shifted by `t` came out shifted by `2t` the other way. Both
  apps now warp with the inverse. Every earlier stack with a non-identity
  alignment was misregistered; results differ from earlier builds.
- **EXIF orientation is applied by default.** JPEG frames with an EXIF
  orientation tag now come out upright in the CLI and the GUI. The setting
  `ignore_exif_orientation` (off by default) used to do nothing; it now switches
  the rotation off.
- **Auto-cull threshold scales with the frame count.** A frame is dropped when it
  wins fewer than `min(2, 50 / N)` percent of the in-focus detail pixels (`N`
  frames). The flat 2 % rule emptied every evenly spread stack of more than 50
  frames. If every frame would be dropped, all are kept and a warning is shown.
  The CLI now honours `auto_cull` as well.
- **Keyboard shortcuts.** Ctrl+S saves the project and Ctrl+E opens the export
  dialog (the old direct image save is the export dialog's Save As). The full list
  is in `apps/stacker-gui/docs/shortcuts.md` and behind `?`.
- **Settings persist on every edit.** The GUI saves each change to `settings.toml`
  after a 600 ms pause (atomic write). Cancel, Esc and the dialog's cross go back
  to the settings as they were when the dialog opened (they used to keep every
  edit, since each one had already been applied); Reset to Defaults and Load
  Config take effect on Apply or the next edit. Integer sliders snap while dragging. A settings file that cannot be
  parsed is renamed `settings.toml.corrupt-<unix seconds>` and never overwritten.
- **The PMax apron follows the pyramid depth, and the disk cost with it.** A tiled
  run needs an apron of 128 px at 6 pyramid levels, 256 at 7 and 512 at 8 (the
  default); the old fixed 128 px was not enough at the default depth. The CLI's
  scratch disk is `12 * ((T + 2A) / T)^2` bytes per source pixel per frame: 108 at
  the PMax default with `--tile-size 512` (it was 27 with the old apron). The CLI
  logs the figure before it starts and warns when a stored tile is more than 4
  times its interior; fewer `pyramid_levels` or a larger `--tile-size` reduce it.
  `--tile-size` is now at least 64 and, for `pmax`, rounded up to a multiple of
  `2^(pyramid_levels - 1)`.
- **Retraining is required for the AI mode.** The model's GroupNorm layers are
  replaced by a per-pixel channel norm, so tiled inference matches whole-image
  inference away from the tile edges. Weights trained before this change fail to
  load. The trainer's rolling checkpoint is now `focusmerge_latest.model.mpk` plus
  `focusmerge_latest.optim.mpk` (the old file name was overwritten by the
  optimiser state, so `--resume` never worked). See `crates/stacker-nn/README.md`
  for the files to delete.
- **Output encoding rounds.** The 8-bit and 16-bit writers truncated after the
  sRGB curve, so a saved file could never reach 255 or 65535 and was slightly
  darker than the preview. Linear white is now exactly 255 / 65535 in both apps.
- **Natural file order.** `frame_2` now sorts before `frame_10` (the sort was by
  byte value).
- **Retouch is a tool, not a permanent panel.** The card under the image (and the
  split handle above it) is gone. The toolbar's Retouch button, the `R` key or
  Edit > Retouch Brush switches the tool on once there is a stacked result; a slim
  options row appears above the image while it is on, and Done or Esc leaves it.
  A left-drag on the image now paints only while the tool is on and pans otherwise
  (it used to paint whenever a result was shown). `[` and `]` change the brush
  size. The first-use explainer appears when the tool is first switched on instead
  of when the first result appears. `layout.toml` no longer stores a bottom-panel
  height; an old file loads and the key is dropped on the next save.
- **PMax, DMap and alignment can run on the GPU (opt-in).** A new setting
  `compute_device` (`cpu`, the default, or `gpu`) and a `gpu` feature on the CLI
  and the GUI (independent of `nn-gpu`). In a build with the feature, `gpu` runs
  the PMax and DMap fusions, the alignment objective and the final warp of each
  frame on the graphics card through wgpu, and falls back to the CPU, with a
  warning, when there is no usable adapter, a call needs more than the 3 GiB
  device budget, or the device fails. Results agree with the CPU within a small
  tolerance, not bit for bit; changing the device does not mark a finished
  result stale. The settings file now has 26 keys. The wgpu debug and
  validation layers are off unless `WGPU_DEBUG=1` / `WGPU_VALIDATION=1` asks for
  them; in debug builds they printed a panic per driver message at exit.
- **Keyboard and window fixes found by the UI flow tests.** After a frame was
  picked in the Retouch row and the tool was left, the next change to the canvas
  (New Project, Open) left the keyboard focus on nothing and every shortcut
  stopped working; leaving the tool, and any press on the image, now hand the keys
  back. Esc leaves the Retouch tool before it dismisses a message. The window's
  buttons, boxes and switches are always dark (they followed a light Windows
  theme); file and result names in the sidebar are left-aligned (they were
  centred); the Retouch row keeps each label next to its control and shows
  "Choose a frame…" until a frame is picked; the shortcut list no longer squashes
  its rows; a message no longer covers the zoom buttons.

### Added

- **AI ("neural") stacking mode** — a specialized, fully-convolutional pairwise
  merge network (`FocusMergeNet`) on the [Burn](https://burn.dev/) framework that
  folds a focus stack into one all-in-focus image. Includes:
  - `xs`–`xxl` capacity presets (dilated-residual context stack, per-pixel
    channel normalisation, size-preserving; receptive-field radius
    16/24/40/46/72/100 px for xs…xxl, tile overlap twice that).
  - A multi-term masked training loss (Charbonnier + multi-scale gradient +
    sharpness retention + confidence supervision, occlusion-weighted).
  - Tiled inference with apron-overlap and a feathered (raised-cosine) blend for
    arbitrarily large (8K+) images.
  - A planar↔tensor bridge, model discovery from a `models/` folder
    (`.mpk` weights + `.json` manifest; the default is `focusmerge_latest`, then
    `focusmerge_final`, then the newest epoch), and a runtime backend dispatch
    hiding Burn from the apps.
  - The `z-stackr-train` binary: hand-rolled AdamW loop, cosine LR with warm-up,
    recurrent rollout with scheduled sampling, atomic full-precision rolling
    checkpoints, `--resume` (restores model weights, optimiser moments, epoch and
    step; fails loudly on a missing or corrupt checkpoint unless `--fresh` is
    given), and `--cache-mb` for the decoded-scene cache.
  - Integration as a third stacking mode in both the CLI (`--mode ai`,
    `--model`, `--device`, only in `nn` builds) and the GUI (`AI Model (Neural)`
    with a model picker that greys out when no models are installed, and a CPU/GPU
    selector shown only in multi-backend builds).
  - All of the above is **opt-in** behind the `nn` / `nn-gpu` features; the
    applications build with **zero ML dependencies by default**. The `cuda`
    feature was removed (it never compiled).
- **Live preview**: during alignment the canvas shows each frame as it is
  registered; during AI stacking it shows the running composite sharpening as
  frames are folded in (throttled thumbnails of at most 360 px).
- **New crate `z-stackr-pipeline`** (`stacker_pipeline`): the stages both apps
  call: frame discovery and ordering, one decoder, alignment, auto-cull, the PMax
  and DMap fusion bodies, tile geometry, output encoding and the settings loader.
  The CLI and the GUI no longer carry their own copies; the same settings file
  gives the same pixels in both.
- **UI flow tests** (`apps/stacker-gui/src/ui_flow_tests`): the real window and
  its Rust wiring driven by clicks and keys on Slint's headless backend, with the
  config folder moved aside and file dialogs answered by the test. Twenty-four
  user journeys run with `cargo test -p z-stackr-gui`; the action matrix (every
  toolbar button, shortcut, menu item and sidebar action in nine app states,
  about 520 cells) runs with `UI_FLOW_MATRIX=1` and as its own CI job.
- **Accessible names** for every control: the dialogs' close crosses, the
  activity chip's cancel, the toast's close, the sidebar's file, result, remove
  and clear actions, the stale badge's Recompute, the empty canvas's project
  cards, the Show Aligned switch, the algorithm box and each setting's control.
  Screen readers can name them, and the tests use them.

### Changed

#### Core (`z-stackr-core`)
- Output quantisers round (see above); `encode_row_u8` is a SIMD table encoder
  that is exact, `encode_row_u16` evaluates the curve per sample.
- The settings loader is lenient per key: a wrongly typed value or an unknown
  enum string keeps that key's default and is reported; numeric keys are clamped
  to the ranges written in the default file; enum strings are case-insensitive.
- The tile store validates plane sizes at commit, reports `TileNotFound` instead
  of panicking, and documents that it never deletes its directory.
- `PlanarImage::from_planes`, `with_luma` and `is_consistent` state and check the
  plane-size invariant; the colour matrices live once in `color`.
- `init_tracing` is idempotent and reports an invalid filter.

#### Alignment (`z-stackr-align`)
- The direction convention is documented and tested (reference to source; warp
  with the inverse).
- Translation mode never fits scale or rotation, also with Bounded search; the
  bounded options have per-degree-of-freedom switches.
- The objective counts only pixels the warp can sample; a transform that samples
  less than half of the frame is penalised.
- The reference luma pyramid is built once per run, and the objective allocates
  nothing per evaluation.
- The four-point homography fit no longer panics; the registration kernel is
  documented as a four-tap C0 cubic kernel (not a spline or Catmull-Rom).

#### Fusion (`z-stackr-algo`)
- DMap: window sums in `f64` (no precision loss at 24 MP), a single-plane focus
  map computed with a rolling band of summed-area tables, a corrected Otsu split
  and tie rule, and a threshold that is a whole-image property.
- PMax: `pyramid_levels` below 2 are raised to 2 with a warning (a one-level
  pyramid was silently a mean); the apron follows the depth.
- Retouch: a stroke API (`begin_stroke`, `extend_stroke`, `end_stroke`) with
  block-diff history, so one drag is one undo step.

#### Command line (`z-stackr-cli`)
- All 26 settings keys are honoured (the CLI used to read 10); `alignment_mode =
  "bounded"` is an alias for `affine` with `bounded_search = true`.
- `--config` must exist and parse; `--write-default-config` creates a commented
  file; `--monitor` re-runs on new frames (errors are reported, not panics).
- Validation runs before the first frame is decoded (mode, tile size, output
  path, AI model).
- The temporary tile directory is removed on every exit path except a killed
  process.
- Wrongly typed settings values stop the run with the file name in the message.

#### Desktop app (`z-stackr-gui`)
- **Project files** `<name>.zstack.toml`: the source list, the whole settings and
  the retouch strokes (replayable), with New, Open, Open Recent (10 entries),
  Save and Save As; a changed-since-saved check by content, an unsaved-changes
  question before New, Open and closing, and a menu bar (File, Edit, View, Help).
- **Export dialog** (Ctrl+E): output folder, file-name template with the
  variables `{name} {algo} {date} {time} {timestamp} {n} {w} {h}`, format, bit
  depth and JPEG quality; existing files are numbered, never overwritten. Save
  encodes the newest result from the linear `f32` data with the retouch included,
  so a 16-bit file holds real 16-bit samples (it used to re-encode an 8-bit PNG).
- **Retouch from a chosen source frame**, with one undo step per stroke; it used
  to composite the result with itself and never saved.
- **Manual** (F1), keyboard-shortcut overlay (`?`), first-use explainers with
  "Don't show again", and drag-and-drop of image files, folders and project files.
- **Stale badges with Recompute** on the aligned frames and the newest result;
  **activity chips** with elapsed time, an estimate of the time left and a cancel
  button; one blocking job at a time, panics reported instead of hanging.
- A frame that fails to decode is skipped and named (it used to shift every later
  index); auto-cull marks dropped frames instead of rewriting the list.
- Wheel zoom about the pointer, pan, a true 1:1 view, resizable panels kept in
  `layout.toml`; the depth-map preview is a non-modal card; the log file moved to
  the data folder.
- Alignment mode `Registration` is not offered in the combo box (a settings file
  that says `registration` is kept and used).

#### Build, tests and documentation
- A pinned nightly (`rust-toolchain.toml`), `LICENSE-MIT` and `LICENSE-APACHE`
  with `license = "MIT OR Apache-2.0"` in every manifest, one workspace table for
  versions and dependencies, and a CI workflow (`.github/workflows/ci.yml`, not yet
  run on a hosted runner). Unused dependencies were removed; `akaze` 0.7.0 still
  pins `image` 0.23.14.
- No `allow` attributes remain; the pedantic and nursery clippy matrix is clean.
- The quality tests against the reference stack are `#[ignore]`d and write under
  `target/quality/`; tests no longer write into the working tree. The unreferenced
  `akaze.tar.gz` was deleted.
- README claims were rewritten against the code: the DMap focus measure is a
  plane-fit residual variance (not a Sum-Modified Laplacian), the PMax options
  only change how the winning frame is chosen, and the alignment README snippet
  compiles. See `claim_ledger_after.md` for the check.

### Notes
- No pre-trained AI weights ship with the repository: train your own with
  `z-stackr-train` and drop the `.mpk`/`.json` pair into `models/`. The classical
  PMax and DMap modes remain the defaults and require no ML build.
- `sample/result/zerene.jpg` is the output of a commercial product used only by
  the ignored quality tests; it stays until the owner has confirmed its licence.
- All crate versions are `1.0.0`.

## Earlier changes of this release

### Changed
- **Alignment is intensity-first and AKAZE-independent.** Intensity-based RMS
  refinement runs for every alignment mode; AKAZE feature matching is only ever an
  *optional seed*, gated behind the `akaze` feature **and** an off-by-default
  runtime toggle. In the GUI a seed that is not plausible (non-finite, scale
  outside 0.5 to 2, or a shift larger than the frame) is discarded and the
  refinement starts from the identity; the CLI and the GUI now share this check.
- Alignment modes are `Affine`, `Translation`, `Registration` and `None`
  (`Registration` is the single-resolution bounded registration); the redundant
  `Backward Mapping` alias was removed and `Bounded` became a standalone **Bounded
  search** toggle. An **AKAZE seeding** toggle (default off) was added. Both are
  exposed in the GUI, the config file (`bounded_search`, `akaze_seeding`), and the
  CLI pipeline.
- The out-of-core tile manager (de)serialises each planar channel as a single bulk
  `bytemuck` byte copy (native endianness) instead of per-element
  `to_le_bytes`/`from_le_bytes` loops.
