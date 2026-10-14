# z-stackr

![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)
![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)

**z-stackr** is a focus-stacking application written in Rust. It takes a sequence of macro or micro photographs taken at different focus distances, aligns them, and fuses them into one picture that is sharp from front to back. There is a command-line tool (`z-stackr`), a desktop app (`z-stackr-gui`) and a web app (`z-stackr-web`) that runs the same interface in the browser, plus Python bindings; all run the same pipeline stages from one shared crate.

> **Renamed algorithms.** The fusion engines are now called **Apex** (was *PMax*, Laplacian pyramid), **Relief** (was *DMap*, depth map) and **Strata** (new, guided-filter blend). Everything that worked before still reads: the old `dmap_*` settings keys, `--mode pmax|dmap|both`, the GUI and project labels "PMax (Laplacian)", "DMap (Depth Map)" and "Both (DMap & PMax)", and the older `use_gpu` key. Writing always uses the new names, and the `{algo}` value in file-name templates is now `Apex`, `Relief` or `Strata`. See the [CHANGELOG](CHANGELOG.md).

---

## Key features

* **Alignment.** A coarse-to-fine intensity-based refinement of the luma planes, with a choice of motion model and optimiser. Four models (`translation`, `registration`, `affine`, `none`; `neural` in `nn` builds) and the optimisers Lucas-Kanade and Nelder-Mead (`optimizer = "auto"` tries Lucas-Kanade first), plus the toggles **Bounded search**, **AKAZE seeding**, **Correct Brightness** and **Chain Alignment**. A refined matrix that fits worse than doing nothing is replaced by the identity (an identity gate). After warping, the result is cropped to the rectangle that every frame covers (**Crop to Common Area**); a common rectangle below half of the frame area is refused and the output stays uncropped. See "Alignment" below.
* **Optional AKAZE + RANSAC seed.** With the `akaze` feature compiled in (on by default in the GUI, off in the CLI), `akaze_seeding = true` seeds the refinement from feature matches. Seeding is off by default; a seed that is not plausible (non-finite, scale outside 0.5 to 2, or a shift larger than the frame) is discarded and the refinement starts from the identity. The RANSAC fit must be agreed on by at least a fifth of the matches (never asking for more than 50), so a lucky minimal sample cannot pass.
* **Linear-light arithmetic.** Frames are decoded to linear light (exact lookup tables for 8-bit and 16-bit sources), stored as three `f32` planes (luma and two chroma planes, BT.601 weights on linear RGB), fused, and encoded back to sRGB with correct rounding (linear white is exactly 255 or 65535).
* **Portable SIMD.** The warp, the pyramid blur and the 8-bit sRGB encoder use `std::simd`. The vector width is fixed when the code is compiled (16 lanes with `avx512f`, 8 with `avx2`, otherwise 4) and does not change the output: a plain build targets baseline `x86-64` (SSE2, 4 lanes), so wider vectors need the `RUSTFLAGS` described under "CPU features" below. The 16-bit sRGB encoder is scalar code: an exact table search classifies each sample and only a sample next to a code boundary evaluates the transfer function.
* **Out-of-core CLI and GUI.** The CLI (and the GUI with a nonzero `tile_size`) writes every aligned frame to a temporary directory as padded tiles and fuses tile by tile, through the shared `stacker_pipeline::tiled` module. The alignment pass holds whole frames (about 39 bytes per pixel, independent of the frame count); the fusion pass holds one padded tile of every frame; the scratch disk scales with the padded tile area (see "Memory and disk" below and `crates/stacker-pipeline/README.md`). The CLI can also stack several folders in one run (`--stacks subfolders`).
* **Four fusion engines.** **Apex** (Laplacian pyramid), **Relief** (plane-fit residual-variance focus measure, per-pixel selection and a guided filter, optionally with a multigrid depth solver), **Strata** (a guided-filter soft blend of every frame) and, in `nn` builds, an **AI model** (a learned pairwise-merge network). The GUI can also run combinations (**Both**: Relief and Apex; **All Three**: Strata, Relief and Apex) as passes of one job.
* **Neural stacking and neural alignment (opt-in).** `FocusMergeNet` on the [Burn](https://burn.dev/) framework folds the stack into one image, with `xs` to `xxl` capacity presets, tiled inference, a CPU (`ndarray`) backend and an optional GPU (`wgpu`) backend. A second family of networks (`BatchMergeNet`, `BatchAlignNet`, `FusionAlignNet`) merges or aligns a whole stack; the trained alignment network is the `neural` alignment mode (desktop and CLI only). The training binary is `z-stackr-train`, the alignment evaluation binary `stacker-nn-eval`. Everything neural is behind the `nn` feature; without it the apps have no Burn dependency. No trained model ships with the repository.
* **Brightness correction.** Exposure drift between frames (flicker, auto-exposure) shows up as brightness steps in a fused result. With **Correct Brightness** (on by default) every aligned frame is matched to the reference in linear light (a scale and a mild gamma fitted to the frame's mean and standard deviation over the pixels the warp covers), which leaves frames that already agree bit for bit unchanged.
* **RAW input and EXIF copy.** Camera RAW files (CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2, PEF and others) are developed and stacked like any frame in a build with the `raw` feature (native only; the camera white balance and color matrix are used as stored, no lens corrections). **Copy Metadata to Output** copies the EXIF block of the source frames into JPEG and PNG outputs (the orientation tag is set to upright when the pixels were rotated on load). TIFF output cannot carry it and says so.
* **Python bindings.** `crates/stacker-python` (a maturin project, kept outside the Cargo workspace) exposes the in-memory path (NumPy arrays) and the out-of-core file path to Python as the module `zstackr`; see "Python" below.
* **Apex, Relief, Strata and alignment on the GPU (opt-in).** In a build with `--features gpu`, `compute_device = "gpu"` runs the heavy stages on the graphics card through wgpu (Vulkan, Direct3D 12 or Metal): the whole Apex and Relief fusions (frames folded in one at a time on the device, only the result comes back), kernels for the Strata saliency and guided-filter pair and the Relief multigrid solver (the pipeline hands them to `Compute::strata_kernels` and `Compute::multigrid_depth`; see `crates/stacker-gpu/README.md`), the alignment objective the optimiser evaluates hundreds of times per frame, and the final warp of each frame. Decoding, auto-cull, export and the AI mode stay on the CPU. Lucas-Kanade is a CPU path. The CPU stays the reference and the fallback: with no usable adapter, a call larger than the device budget (3 GiB) or any device failure the run continues on the CPU and a warning says why. Results agree with the CPU within a small tolerance, not bit for bit. See `crates/stacker-gpu/README.md` for the measured speed (provisional figures). It is independent of `nn-gpu`, which is the AI mode's backend.
* **Live preview.** While the GUI aligns, the canvas shows each frame as it is registered; during AI stacking it shows the running composite after each folded frame. Live previews are thumbnails of at most 360 px on the long edge, offered at most every 500 ms; the finished result is shown at full size.
* **16-bit output.** PNG and TIFF are written with 16 bits per channel by default (`image_saving.bit_depth = 16`); JPEG is always 8-bit. The GUI encodes its export from the full-precision linear result, so a 16-bit file holds real 16-bit samples.
* **Desktop app.** Project files (`<name>.zstack.toml`) with an unsaved-changes guard and a recent list, a menu bar, drag-and-drop of images and folders, an export dialog with a file-name template, stale badges with Recompute, activity chips with an estimate of the time left and a cancel button, retouch from a chosen source frame with one undo step per stroke (the strokes are saved in the project and replayed when it is opened; the undo history itself is not saved), and a built-in manual (F1). See `apps/stacker-gui/README.md`.
* **One settings file.** The GUI and the CLI read and write the same TOML schema with the same 39 keys (27 top-level, 6 in `[preprocessing]`, 6 in `[image_saving]`), starting with `settings_version = 2`. A file written by an older build is migrated when it is read (see the CHANGELOG).
* **Performance.** The hot loops are tuned rather than traded for quality: the Nelder-Mead alignment spends its iterations on the coarse pyramid levels and finishes with a short full-resolution polish, Lucas-Kanade needs a few iterations per level, the Apex fold in `stacker-algo` takes frames one at a time with memory that does not grow with the frame count, and the Strata, Relief and alignment kernels have GPU forms. Speed also depends on how the binary is built: see "CPU features" and "Build tasks, PGO and BOLT" below. No timing figures are quoted here; the figures in the crate READMEs are provisional.

---

## Architecture

A Cargo workspace. Package names are `z-stackr-*`; the in-code crate identifiers remain `stacker_core`, `stacker_pipeline`, and so on.

### Applications (`apps/`)
* **`z-stackr-cli`**: the command-line tool for batch and headless use (binary `z-stackr`).
* **`z-stackr-gui`**: the desktop app built with Slint (binary `z-stackr-gui`).
* **`z-stackr-web`** (with its Web Worker `z-stackr-web-worker`): the same Slint interface in the browser, built with Trunk. The images stay on your machine; the stack runs in a worker with a 2 GiB memory budget (a stack that does not fit is offered a reduced Pre-resize percentage). Alignment runs on WebGPU only (Chrome or Edge 113+, Firefox 141+ on Windows, Safari 26); a browser without it can stack without alignment. Run it locally with `trunk serve` from `apps/stacker-web`; see [`apps/stacker-web/README.md`](apps/stacker-web/README.md).

### Crates (`crates/`)
* **`z-stackr-core`**: `PlanarImage`, linear/sRGB conversions, the settings schema (`StackingSettings`), the file-backed tile store, RAW decoding (`io`, `raw` feature) and EXIF copy-through (`metadata`).
* **`z-stackr-align`**: the alignment engine: motion models (shift, translation, similarity, affine), the Nelder-Mead and Lucas-Kanade optimisers, the identity gate, brightness correction, chain-alignment arithmetic, the warps, the coverage mask, and the optional AKAZE + RANSAC seed.
* **`z-stackr-algo`**: the fusion algorithms (Laplacian-pyramid Apex, plane-fit Relief with guided filter and multigrid solver, guided-filter Strata), frame optimisation (auto-cull and ordering) and the retouch session.
* **`z-stackr-pipeline`**: the pipeline stages both apps call: frame discovery and ordering, decoding (EXIF orientation, rotation, crop, resize, RAW), alignment, auto-cull, fusion, tile geometry, the out-of-core `tiled` runner, output encoding (with EXIF), and the settings loader.
* **`z-stackr-gpu`**: GPU kernels through wgpu (Apex, Relief, Strata, the multigrid solver, the alignment objective and warp), kept out of the CPU crates; the pipeline uses it only in builds with the `gpu` feature.
* **`z-stackr-web-core`**: the message protocol and memory-budget policy shared by the web page and its worker.
* **`z-stackr-nn`**: the neural engine on Burn: the `FocusMergeNet`, `BatchMergeNet`, `BatchAlignNet` and `FusionAlignNet` models, the training losses, tiled inference, model discovery, the `z-stackr-train` binary and the `stacker-nn-eval` binary.
* **`z-stackr-python`** (`crates/stacker-python`, outside the workspace): the Python bindings (`pip install z-stackr`, `import zstackr`).

> **AKAZE**: feature-based seeding uses the [`akaze`](https://crates.io/crates/akaze) crate (v0.7.0) from crates.io, behind the optional `akaze` feature.

---

## Installation

### Prerequisites
A Rust toolchain installed with [rustup](https://rustup.rs/). The workspace needs a **nightly** compiler (it uses `#![feature(portable_simd)]`). The repository commits a `rust-toolchain.toml` that pins `nightly-2026-09-08` with the `clippy` and `rustfmt` components; running `rustup toolchain install` inside the checkout (no arguments) installs it. `.github/workflows/ci.yml` uses the same pin.

### Building from source
```bash
git clone <repository-url> z-stackr
cd z-stackr

# CLI (binary: z-stackr)
cargo build --release -p z-stackr-cli

# CLI with AKAZE seeding available
cargo build --release -p z-stackr-cli --features akaze

# CLI with the AI ("neural") mode, CPU (ndarray) backend
cargo build --release -p z-stackr-cli --features nn

# ... or with the GPU (wgpu / Vulkan) backend as well (implies `nn`)
cargo build --release -p z-stackr-cli --features nn-gpu

# Desktop app (binary: z-stackr-gui; AKAZE is on by default)
cargo build --release -p z-stackr-gui

# Camera RAW input (native only)
cargo build --release -p z-stackr-cli --features raw
cargo build --release -p z-stackr-gui --features raw

# Apex, Relief, Strata and alignment on the GPU (wgpu); independent of `nn-gpu`, and the CPU stays the fallback
cargo build --release -p z-stackr-cli --features gpu
cargo build --release -p z-stackr-gui --features gpu

# The trainer and the alignment evaluation binary
cargo build --release -p z-stackr-nn --bin z-stackr-train --bin stacker-nn-eval
```
The executables are written to `target/release/`. `<repository-url>` is a placeholder: the repository URL is not set in the manifests. Features combine (`--features akaze,raw,gpu`).

#### CPU features (important for speed)

The SIMD code uses portable `std::simd`, which is fixed at **compile time**. The vector width follows the target features of the build: 16 `f32` lanes with `avx512f`, 8 with `avx2`, otherwise 4 (one 128-bit SSE2 operation). A plain `cargo build` targets baseline `x86-64`, i.e. SSE2 only, so it gets 4 lanes; and without the `fma` feature the warp's `mul_add` is a library call per lane. The 8-bit sRGB encoder reads its code table with a gather of bytes on every target (x86 has no 8-bit hardware gather, and a 32-bit table would need 64 KiB instead of 16 KiB). The output does not depend on the width or on FMA: results stay correct, only the speed of the vector code changes.

For a build that runs on any AVX2 + FMA CPU (Intel Haswell or AMD Zen 1 and newer), pass the `x86-64-v3` level:
```bash
# bash / zsh
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release -p z-stackr-cli
# PowerShell
$env:RUSTFLAGS = "-C target-cpu=x86-64-v3"; cargo build --release -p z-stackr-cli
```
`-C target-cpu=native` additionally enables AVX-512 where the CPU has it, but the binary then only runs on CPUs with the same features. A binary built for `x86-64-v3` does not run on older CPUs. Use the same `RUSTFLAGS` for `cargo clippy` and `cargo test`, otherwise cargo rebuilds everything when the flags change. No `.cargo/config.toml` is committed on purpose: the right CPU level depends on who runs the binary. `stacker_align::transform::simd_build_report()` returns a one-line description of what the build was compiled for, and `has_hardware_fma()` a boolean.

#### Build tasks, PGO and BOLT

`docs/vscode-tasks.json` is a set of VS Code tasks: the GUI, the CLI and the trainer for each CPU level (`generic`, `x86-64-v3`, `x86-64-v4`, `znver3`, `znver4`) on Windows and Linux, and profile-guided-optimisation chains (plus BOLT on Linux) for the CLI and the GUI. Copy it to `.vscode/tasks.json`; `docs/build-tasks.md` explains the tasks, the prerequisites (`llvm-profdata`, `llvm-bolt`) and the training workload (Apex, Relief with and without multigrid, and Strata on the frames in `sample/`, aligned with registration + Lucas-Kanade, registration + Nelder-Mead and the six-parameter affine model). They use the `release-perf` cargo profile: the release code with line tables and symbols kept. These builds take a long time and are run by hand; nothing in the repository starts them.

For release binaries of every CPU and GPU variant, `scripts/pgo-build.ps1` (Windows) and `scripts/pgo-bolt-build.sh` (Linux, PGO then BOLT) run the whole PGO chain per variant (`-Cpu scalar|avx2|avx512|native|all`, `-Gpu gpu|cpu|both`, `-Target cli|gui|all`) and write the binaries to `dist/`; see "Build scripts (PGO, every CPU and GPU variant)" in `docs/build-tasks.md`.

> **No ML by default.** Without `--features nn` (or `nn-gpu`) neither app pulls in Burn, and the AI option is absent from the GUI. The GPU backend is a separate opt-in on top of `nn`.

> **No GPU stack by default.** Without `--features gpu` neither app links wgpu for the stacking stages, and the Compute device box in the GUI lists the CPU only. A settings file that says `compute_device = "gpu"` still loads in such a build; the run uses the CPU and says so.

---

## Usage (CLI)

The CLI takes a directory of focus-bracketed images and writes one composite image.

```bash
cargo run --release -p z-stackr-cli -- \
    --input-dir ./sample/ \
    --output-file ./sample/result/stacked_output.tif \
    --mode apex \
    --tile-size 512
```

### Command-line arguments

* `--input-dir <DIR>` (required): the folder with the source frames (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, any letter case, and camera RAW files in a build with `--features raw`). Files are used in natural file-name order (`frame_2` before `frame_10`); the first one is the alignment reference. The output file and the log file are never read as frames.
* `--output-file <FILE>` (required): the result. The extension selects the format (`.png`, `.tif`/`.tiff`, `.jpg`/`.jpeg`); a name without an extension gets the one from `image_saving.output_format`. With `--mode both` or `--mode all` one file per algorithm is written, named `<stem>_<algo>.<ext>` (`result_apex.tif`, `result_relief.tif`, `result_strata.tif`); with `--stacks subfolders` this is the output directory. PNG and TIFF are written at 16 bits per channel unless `image_saving.bit_depth = 8`; JPEG is 8-bit. The path is checked before the first frame is decoded.
* `--mode <MODE>` (required): `apex`, `relief`, `strata`, `both` (Relief and Apex on the same aligned stack), `all` (Strata, Relief and Apex on the same aligned stack) or `ai` (any letter case; `nn` is an alias of `ai`). The old names `pmax` and `dmap` are accepted as aliases of `apex` and `relief` (with a warning). `ai` needs a build with `--features nn`.
* `--tile-size <SIZE>` (required): the tile size in pixels, at least 64 (512 is a good value). For `apex` it is rounded up to a multiple of `2^(pyramid_levels - 1)`. A tile larger than the image is one tile. Larger tiles lower the padding overhead on disk and in RAM.
* `--config <FILE>`: a TOML settings file (the format the GUI writes). The file must exist and parse; an error stops the run instead of silently using defaults. Without `--config` the built-in defaults are used. A file without `settings_version = 2` is migrated as described under "Alignment"; `alignment_mode = "bounded"` is accepted as an alias for `registration` with `bounded_search = true`, and the old `dmap_*` keys and `use_gpu` still read.
* `--optimizer <auto|lk|nm>` (any letter case): overrides the settings file's `optimizer` (`lk` is Lucas-Kanade, `nm` Nelder-Mead; see "Alignment"). The settings file itself spells the values `auto`, `lucaskanade` and `neldermead` (`lk` and `nm` read too).
* `--no-gpu`: only in builds with `--features gpu` or `nn-gpu`. Keeps the run on the CPU whatever `compute_device` says (the same as `compute_device = "cpu"`, and the neural models run on the CPU as well, like `--device cpu`).
* `--stacks <subfolders|single>`: what to do when `--input-dir` holds subfolders with images. `subfolders` stacks every such subfolder on its own, one after the other, into `--output-file` (a directory; the files are named by `image_saving.filename_template`); `single` stacks only the images directly inside `--input-dir`. Without the flag a terminal is asked; without a terminal the run stops and asks for the flag. Not combinable with `--monitor` in `subfolders` mode.
* `--write-default-config`: with `--config`, create a missing file with commented defaults first.
* `--monitor`: keep watching `--input-dir` and re-run the stack when a frame is added or changed.
* `--log-file <FILE>`: also write the structured log to this file.
* `--model <NAME>`, `--device <cpu|gpu>`: AI mode, and only in builds with `--features nn` or `nn-gpu` (other builds reject both flags). `--model` is the file stem of a fusion model in the `models/` folder (default: the first one in discovery order); `--device` defaults to the best backend in the build.
* `--align-model <NAME>`: the alignment model for `alignment_mode = "neural"` (same `nn` builds); the file stem of an alignment model in `models/`.

> **Models folder.** AI mode and neural alignment look for models in a `models/` directory next to the executable and then in the current working directory. Each model is a pair `<name>.mpk` (weights) + `<name>.json` (architecture manifest, including the network architecture; a manifest without one is a `FocusMergeNet`). Fusion models and alignment models are listed separately. If no model of the kind is found the option is unavailable.

### Memory and disk

Every frame is stored on disk as padded tiles during a CLI run. The tile directory (`stacker_tiles_*` below the system temp directory) takes `12 * ((T + 2A) / T)^2` bytes per source pixel per frame, where `T` is the tile size and `A` the apron. The apron follows what is computed:

| Method | Apron `A` |
|---|---|
| `apex` | 128 px at 6 pyramid levels, 256 at 7, 512 at 8 (the default) |
| `relief` | `estimation radius + 2 * smoothing radius + 1` (10 px at the defaults) |
| `strata` | 101 px (the saliency reach plus the guided filter's two radius-45 boxes; fixed) |
| `ai` | 128 px |

Relief with the multigrid solver (`relief_use_multigrid`) solves the depth on whole-image index and weight planes (about 8 bytes per pixel, plus the solver's own levels, about 28 bytes per pixel at its peak), so the result does not depend on the tile size and shows no seams.

At the `apex` default (8 levels, `T` = 512) that is 108 bytes per source pixel per frame; with 6 levels (`A` = 128) it is 27. The CLI logs the padding factor and the scratch bytes per source pixel before it starts, and warns when a stored tile is more than 4 times its interior. The directory is removed when the run ends, also after an error; a killed process cannot clean up. The per-pixel RAM of every stage is in `crates/stacker-pipeline/README.md`.

---

## Fusion modes

### 1. `apex` (Laplacian pyramid; was `pmax`)
A Laplacian-pyramid blend. At every detail band the source with the largest selection energy (the squared Laplacian coefficient) supplies the coefficient, and the lowest band is averaged.
* **Settings**: pyramid levels, *Use all color channels*, *Grit suppression* (see below).
* **Character**: keeps fine detail across depth intersections; it can increase apparent noise in out-of-focus background areas.

### 2. `relief` (depth map; was `dmap`)
A plane-fit residual-variance focus measure per frame, a per-pixel choice of the sharpest frame, and a guided filter on the result.
* **Multigrid** (`relief_use_multigrid`, off by default): instead of taking each pixel's own best frame, the frame index is solved as a smooth depth surface by a multigrid solver. Pixels with a confident focus pin the surface, the rest are interpolated, and the composite blends the two frames around the fractional depth. It suits subjects whose depth is smooth, such as a flat specimen tilted in the stack; it can smear a fine structure that jumps in depth.
* **Character**: smooth, low-noise backgrounds; weaker than `apex` on interlocking geometry.

### 3. `strata` (guided-filter blend; new)
A soft blend. Every frame contributes to every pixel, weighted by how sharp it is there, and the weights follow the edges of the image (a guided filter guided by the frame's own luma). Each frame is split into a coarse base layer and a detail layer; the base weights are wide and smooth, the detail weights tight and concentrated on the sharpest frame.
* **Settings**: *Base radius* (`strata_base_radius`, 8 to 64, default 31) and *Detail focus* (`strata_detail_focus`, 1 to 5, default 3).
* **Character**: no hard seams between the frames a pixel could come from, and no color fringes (the weights come from the luma and apply to all planes); softer than `apex` on very fine, high-contrast structure. Two passes over the frames, with memory that does not grow with the frame count.

### 4. `ai` (AI model, neural)
Folds the stack through a trained `FocusMergeNet`: the running composite starts as the first frame, and each following frame is merged into it by a learned soft selection plus a residual refinement that also tracks a confidence channel. Requires a trained model in `models/` and a build with `--features nn` (CPU) or `nn-gpu` (GPU). See "Training the AI model" below.

---

## Alignment

Every frame is registered to the reference (the first frame, or the sharpest with **Sort by Sharpness**) before fusion. The matrix a refiner returns maps reference coordinates to frame coordinates, and the frame is warped with its inverse. Pixels the warp cannot fill are masked out of the later steps.

### Modes

| `alignment_mode` | Degrees of freedom | Notes |
|---|---|---|
| `translation` | shift X and Y, one uniform scale | corrects focus breathing; no rotation |
| `registration` (default) | shift, uniform scale, rotation | the former `affine` |
| `affine` | shift, scale, rotation, aspect ratio, shear | always searched inside bounds |
| `none` | none | frames are used as loaded |
| `neural` | learned | `nn` builds, desktop and CLI only |

Lucas-Kanade and the `affine` simplex search inside windows centred on the seed: 20 % of the image size for each shift, 20 % for the scale and 20 degrees for the rotation (10 % for aspect and shear). With Bounded Search on, the windows narrow to 5 %, 5 % and 5 degrees.

### Optimizer

* **Nelder-Mead** minimises the RMS difference of the luma planes over a luma pyramid. It spends its iterations on the coarse levels (150 each, restarting the coarsest level from several deterministic starts), gives the large levels near full resolution only 3, and ends with a short full-resolution polish. `bounded_search` runs it in a logit-bounded space.
* **Lucas-Kanade** is a Gauss-Newton solver on the same pyramid and the same objective, clamped to the same search windows. It runs on the CPU only.
* **Auto** (`optimizer = "auto"`, default) runs Lucas-Kanade first and the simplex from the same seed when Lucas-Kanade errors or ends no better than it started.
* The browser build can only use the simplex, so `lucaskanade` and `auto` mean Nelder-Mead there.

### Safeguards

* **Identity gate.** After refinement the matrix is compared with the identity on a reduced copy (short side at most 256 px); if doing nothing fits strictly better, the frame keeps the identity. A bad seed or a frame that shares nothing with the reference therefore cannot be made worse.
* **Seed check.** An AKAZE seed that is not plausible is discarded; the RANSAC fit needs at least a fifth of the matches to agree.
* **Brightness correction** (`correct_brightness`, default on): fits `y' = (scale * y)^gamma` on the linear luma of each warped frame so that its mean and standard deviation match the reference's over the pixels the warp covers (gamma limited to 0.5 to 2); chroma follows the same per-pixel ratio. A frame that already agrees is left unchanged.
* **Chain alignment** (`chain_alignment`, default off): frame `k` is registered to the raw frame `k - 1` and the steps are composed, `C_k = R_k * C_(k-1)`. The coverage of a frame is derived from the cumulative matrix, so the border of undefined pixels is excluded from the common area.

### Neural alignment (`nn` builds)

With `alignment_mode = "neural"` the stack is shown to an alignment network as thumbnails of at most 512 px (streamed frame by frame, not the full-resolution stack) and it predicts a matrix per frame. With `neural_refine_classically` (default) those matrices only seed the classical refinement; without it they are used directly. A neural run is part of the stale fingerprint, so changing the model recomputes the alignment. No trained model ships with the repository; train one with `z-stackr-train --strategy align` and check it with `scripts/04_eval_alignment.py` (see "Training the AI model").

### Settings migration

Settings files carry `settings_version = 2`. A file without it was written by an older build and is migrated when it is read: `alignment_mode = "affine"` becomes `registration` (the same 4-parameter search), the old single-pass `registration` becomes `registration` with `bounded_search = true`, and `translation` stays `translation` (it now also fits a scale). The old `dmap_*` keys, `relief_engine` and `use_gpu` are renamed on reading. Saving always writes the current names.

---

## GUI settings overview

The settings dialog ("Stacking Settings", Edit > Settings, Ctrl+,) edits the stacking settings in six tabs: **General**, **Apex**, **Relief**, **Strata**, **Input** and **Output**. Hover a setting for a tooltip. Every edit is saved at once (debounced) to `settings.toml`, the same file format the CLI reads with `--config`. The **Algorithm** box, **Align**, **Show Aligned** and **Start Stacking** are in the main window, not in the dialog.

### General tab
* **Alignment Mode**: the registration model applied before fusion. The classical modes minimise an intensity (RMS) difference of the luma planes over a coarse-to-fine pyramid. See "Alignment" above for the details.
    * `Registration` (default): shift, uniform scale and rotation (a 4-parameter similarity transform). This is what the mode called `Affine` did before the rename, so default results are unchanged.
    * `Translation`: shift and one uniform scale (focus breathing), no rotation.
    * `Affine`: the full 6-parameter model: shift, scale, rotation, plus aspect ratio and shear. It is always searched inside bounds.
    * `None`: skips alignment: no seed, no refinement, no warp; the frames are used as loaded.
    * `Neural` (`nn` builds, desktop only): a trained alignment network predicts the matrices on small thumbnails. Two more rows appear: **Alignment Model** (which model from the `models/` folder; the choice applies to the session and is not stored in the settings file) and **Refine Neural Alignment Classically**.
* **Optimizer** (not offered in the browser build): `Auto` (Lucas-Kanade first, the Nelder-Mead simplex when it fails or ends up worse), `Lucas-Kanade` or `Nelder-Mead`. Lucas-Kanade is a CPU path; the browser build always uses the simplex.
* **Bounded Search** (toggle): keeps the optimiser inside a band around the starting estimate (a logit-space bounded simplex). The default half-widths are 5 % of the image size for each shift, 5 % for the scale and 5 degrees for the rotation (10 % for aspect and shear). Under `Translation` only the shifts and the scale are searched.
* **AKAZE Seeding** (toggle, off by default): seeds the optimiser from an AKAZE feature match before the intensity refinement. It helps on textured subjects with large shifts and has no effect unless the app was built with the `akaze` feature.
* **Correct Brightness** (toggle, on by default): matches every aligned frame's brightness to the reference.
* **Chain Alignment** (toggle, off by default): aligns each frame to its predecessor instead of the reference, which helps long stacks whose ends share little content; drift can add up.
* **Refine Neural Alignment Classically** (toggle, on by default; only with `Neural`): the network's matrices seed the classical refinement instead of being applied directly.
* **Auto-Cull Frames**: runs the frame-optimisation pass before fusion. It scores every frame with the plane-fit focus measure (window radius 1), then (a) drops frames that win fewer than `min(2, 50 / N)` percent of the scene's in-focus detail pixels (N is the frame count: 2 percent for stacks of up to 25 frames, half of the average share for deeper ones); if every frame would be dropped, all are kept and a warning is shown; and (b) can re-order the kept frames by a greedy nearest-neighbour walk over the centroids of the pixels each frame won. The surviving frames are fused in stack (capture) order unless **Sort by Sharpness** is on, because the multigrid depth needs the capture order. With auto-cull off, every frame is used in its loaded order. Dropped frames stay in the Source Files list, dimmed and marked `(culled)`. **Cull Threshold %** (`auto_cull_threshold_pct`, 0 to 50, shown as "Auto" at 0) replaces the automatic rule with a fixed share; 0 keeps the automatic rule. The **Sort** and **Cull** buttons of the Source Files list run these steps on their own. See `crates/stacker-algo/README.md`.
* **Sort by Sharpness** (off by default): orders the frames sharpest-first before alignment, so the sharpest frame becomes the reference.
* **Stack Every Nth Frame**: keeps every Nth frame of the list (1 keeps all).
* **Compute Device** (not offered in the browser build): `CPU` or `GPU` (`compute_device`); see "Key features" for what runs on the GPU.
* **Tile Size** (`tile_size`; the box lists "In Memory" and tile sizes, the key takes 0 or 64 to 8192; not offered in the browser build): the GUI's out-of-core mode. The command line has its own `--tile-size`, which wins.

### Main window
* **Align / Show Aligned**: **Align** runs an alignment-only pass; **Show Aligned** then shows the aligned, common-area-cropped version of the frame you click. A following **Start Stacking** reuses the aligned frames while the file list and the alignment settings are unchanged; otherwise a stale badge appears.
* **Algorithm**: `Relief (Depth Map)`, `Apex (Laplacian)`, `Strata (Guided Fusion)`, `Both (Relief & Apex)` (two passes of one job), `All Three (Strata, Relief & Apex)` (three passes, Strata first) and, in `nn` builds, `AI Model (Neural)`. Settings and project files that carry the old labels ("PMax (Laplacian)", "DMap (Depth Map)", "Both (DMap & PMax)") still load.

### Apex tab (Laplacian pyramid)
* **Pyramid levels** (2 to 12, default 8): the number of bands. More levels reach larger structures and cost time; in the CLI they also raise the apron (see "Memory and disk").
* **Use all color channels**: adds the chroma bands to the energy Apex uses to pick the sharpest frame (luma only by default); luma and chroma are still copied together from one frame.
* **Grit suppression** (default on): at Apex's finest band, picks each pixel's sharpest frame by the energy of its 3x3 neighbourhood instead of the pixel alone; nothing is smoothed.

### Relief tab (depth map)
* **Estimation radius** (0 to 100, default 5): the radius of the window of the plane-fit residual-variance focus measure. Larger values give a smoother map and less precise depth.
* **Smoothing radius** (0 to 50, default 2): the radius of the guided filter applied to the selected luma (the chroma planes are not filtered).
* **Contrast threshold %**: the fraction of pixels, ranked by the best focus score any frame reaches there, that are not allowed to choose a frame. Those pixels take the mean of all frames instead. It is evaluated once on the whole image.
* **Show preview during stacking**: pauses the Relief pass and shows a non-modal card over the canvas where you can adjust the threshold and see the mask.
* **Auto-detect optimum contrast**: computes the threshold from the median and the median absolute deviation of the whole-image focus map (the median plus 1.5 times the deviation), which stays usable on the heavy-tailed maps where Otsu's method collapses. The setting applies to every Relief run, with or without the preview; the row is greyed out in the dialog while the preview switch is off.
* **Multigrid Depth Solver** (off by default): solves the depth with the multigrid solver (see "Fusion modes"); at most 65,536 frames.

### Strata tab (guided fusion)
* **Base Radius** (8 to 64, default 31): the box radius that splits each frame into a base and a detail layer. A larger radius moves more of the picture into the smoothly weighted base.
* **Detail Focus** (1 to 5, default 3): raises the detail weight to this power. On deep stacks the guided filter leaks a small weight to the frames next to the winner and the leaks of dozens of frames add up; a higher value concentrates the detail on the sharpest frame, `1` leaves the weights as they are.

### AI model (neural), `nn` builds only
These controls sit in the sidebar under the **Algorithm** box, not in the dialog.
* **AI model**: which trained fusion model from the `models/` folder to use; the picker is greyed out when none is installed. (The alignment model is picked in the General tab, under `Neural` alignment.)
* **Device**: CPU or GPU; shown only when the build offers more than one backend (`nn-gpu`).
* **Live preview**: the canvas updates after each frame is folded in.

### Input tab (preprocessing)
Applied to every frame when it is decoded, in this order: EXIF orientation, pre-rotation, pre-crop, pre-resize.
* **Pre-rotation**: rotates by 0, 90, 180 or 270 degrees.
* **Pre-crop** and **Crop spec (w,h or x,y,w,h)**: crops to `w,h` (centred) or `x,y,w,h`. An invalid spec is skipped with a warning.
* **Pre-resize %** (10 to 100): downscales before processing (Lanczos3).
* **Reverse sort order**: reverses the file order (the reference frame is then the other end of the stack).
* **Ignore EXIF orientation**: off by default, so a JPEG with an EXIF orientation tag is rotated upright; switch it on to load the pixels as stored. Only JPEG files are checked for the tag.

### Output tab (image saving)
* **Crop to Common Area** (toggle, on by default) and **Resize Cropped to Original** (off): crop to the rectangle every frame covers and, optionally, scale the result back to the original canvas size (a slight non-uniform stretch).
* **Output format**: TIFF, PNG or JPEG.
* **Bit depth**: 16 or 8 bit for PNG and TIFF; JPEG is always 8-bit.
* **JPEG quality** (1 to 100).
* **Filename template**: the pattern for the export file name (without extension). Variables: `{name}`, `{algo}`, `{date}`, `{time}`, `{timestamp}`, `{n}`, `{w}`, `{h}` (see `apps/stacker-gui/README.md`; `{algo}` is `Apex`, `Relief` or `Strata`). The CLI replaces `{name}` only (the stem of the first frame, or the folder name with `--stacks subfolders`) and uses the template when `--output-file` is a directory.
* **Default output dir**: the folder the export dialog's Save writes to without asking (Save As always asks). In the CLI, a bare `--output-file` name is placed there.
* **Copy metadata to output**: copies the EXIF block of the source frames (camera, exposure, date, GPS, ...) into JPEG and PNG outputs without touching the pixels. When the pixels were rotated on load, the orientation tag is rewritten to "upright". TIFF output cannot carry EXIF here, and the app says so instead of dropping it silently.

---

## Training the AI model

The `z-stackr-train` binary trains one of four networks from scratch, chosen with `--strategy`:

| `--strategy` | Network | Task | Training data | Checkpoint stem |
|---|---|---|---|---|
| `pairwise` (default) | `FocusMergeNet` | recurrent pairwise fusion | stage 2 scenes | `focusmerge` |
| `batch` | `BatchMergeNet` | whole-stack fusion in one pass | stage 2 scenes | `batchmerge` |
| `align` | `BatchAlignNet` | whole-stack alignment | stage 3 scenes | `batchalign` |
| `fusion-align` | `FusionAlignNet` | alignment of one frame against the reference | stage 3 scenes | `fusionalign` |

Training data is a directory of scene sub-directories, each holding the focus stack, a ground-truth all-in-focus image, soft in-focus masks, an occlusion map and a `metadata.json` that names every file (the alignment strategies also read `alignment_gt`). Three scripts in `scripts/` produce this layout from ordinary photographs:

1. `scripts/01_estimate_depth.py` estimates a depth map per photo (Python packages in `scripts/requirements.txt`, plus Depth Anything 3, which is installed separately; see the docstring of the script).
2. `scripts/02_blender_focus_stack.py` runs inside Blender (`blender -b -P scripts/02_blender_focus_stack.py -- --img ... --depth ... --out <scene_dir>`) and renders the stack, masks, ground truth and `metadata.json`.
3. `scripts/03_simulate_misalignment.py` (`--input <scene_dir> --out <new_dir>`) turns a perfectly aligned stage 2 scene into an unaligned one: per-frame shift, rotation, anisotropic scale and shear, plus brightness, gamma, noise and blur differences, with the ground-truth matrices in `alignment_gt`. All frames are cropped by a margin larger than any displacement, so no synthetic border teaches the network a shortcut.

Everything the networks do runs in linear light.

```bash
# CPU (default ndarray autodiff backend)
cargo run --release -p z-stackr-nn --bin z-stackr-train -- \
    --data /path/to/scenes --out models --size l --epochs 60

# GPU (wgpu: Vulkan / Metal / DX12)
cargo run --release -p z-stackr-nn --no-default-features --features "wgpu,autodiff" \
    --bin z-stackr-train -- --data /path/to/scenes --out models --size xxl
```

Flags (defaults in brackets): `--data <dir>` (required), `--out <dir>` [`checkpoints`], `--strategy <pairwise|batch|align|fusion-align>` [`pairwise`], `--size <xs|s|m|l|xl|xxl>` [`m`], `--epochs` [40], `--lr` [3e-4], `--crop` (training patch size) [256], `--seed` [42], `--ss-max` (scheduled-sampling probability) [0.5], `--ckpt-every <N>` (steps between rolling checkpoints) [200], `--cache-mb <N>` (budget for decoded scenes kept between steps, `0` keeps nothing) [1024], `--resume` (continue from the rolling checkpoint) and `--fresh` (with `--resume`: start from scratch instead of failing when no usable checkpoint exists). For the alignment strategies: `--corr-radius <N>` (correlation search radius, recorded in the manifest), and for `align` only `--real-data <dir>` (plain folders of real, unlabelled stacks; in the last quarter of the epochs a photometric loss on them is added) with `--photometric-weight <F>` [0.1].

**Checkpoints.** Every `--ckpt-every` steps and at each epoch end the trainer writes `<stem>_latest.model.mpk` (full precision), `<stem>_latest.optim.mpk` (the AdamW moments) and a train-state file, each through a temporary file and a rename (`<stem>` is `focusmerge` for the default strategy; each strategy has its own, so they can share an `--out` directory). Per-epoch snapshots (`<stem>_epoch_NNN.mpk`) and `<stem>_final.mpk` are also written. `--resume` restores model weights, optimiser moments, epoch and step, and fails when `train_state.json` or a checkpoint file is missing or corrupt. Every exported model has a `.json` manifest so the apps can rebuild the architecture.

**Retraining is required after the normalisation change.** The model now uses a per-pixel channel norm instead of GroupNorm, so weights trained earlier fail to load; see `crates/stacker-nn/README.md` for the files to delete and the learning-rate advice.

To use a trained model, copy its `.mpk` + `.json` pair into the `models/` folder next to the binary. A fusion model (`pairwise`, `batch`) appears in the GUI's model picker and is selectable with `--model`; an alignment model (`align`, `fusion-align`) appears in the alignment-model picker and is selectable with `--align-model`. The manifest records the architecture; a manifest without one is read as a `FocusMergeNet`. Every network uses the per-pixel channel norm, so checkpoints trained with the older group norm do not load.

Capacity presets (channels x context depth): `xs` 16x2, `s` 24x3, `m` 32x4, `l` 48x6, `xl` 64x9, `xxl` 96x12.

**Evaluating an alignment model.** Build the evaluation binary and run the script on held-out stage 3 scenes:

```bash
cargo build --release -p z-stackr-nn --bin stacker-nn-eval
python scripts/04_eval_alignment.py --scenes-dir data/held_out --model models/batchalign_final.mpk \
    --eval-binary target/release/stacker-nn-eval --output reports/alignment_eval.json
```

For each scene it reports the mean corner error in pixels (also normalised to a 3000 px wide image) of the network alone, of the network plus classical refinement, and of the classical refiner alone, a paired significance test between the last two, the network's wall-clock time, and a pass or fail against the acceptance bars in the script.

---

## Python

`crates/stacker-python` builds the Python module `zstackr` (PyPI name `z-stackr`) with [maturin](https://www.maturin.rs/). It exposes the pipeline's two paths: an in-memory path on NumPy arrays (align and fuse a list of images) and the out-of-core file path (stack a folder of files through `stacker_pipeline::tiled`, the same code the CLI runs). Settings are the same keys as in the TOML file. It is a separate Cargo project with its own `[workspace]`, listed under `exclude` in the root `Cargo.toml`, so `cargo test --workspace` never links libpython.

```bash
pip install maturin
cd crates/stacker-python
maturin develop --release                   # or: maturin build --release --features gpu,raw
cd ../..
cargo test --manifest-path crates/stacker-python/Cargo.toml   # the pure-Rust parts, from the repository root
```

`gpu`, `raw` and `akaze` are opt-in features of the module, as for the apps. The neural modes are not available from Python yet: `alignment_mode = "neural"` falls back to `registration` with a warning. See `crates/stacker-python/README.md` for the API.

---

## Testing

The repository has unit tests, integration tests, doctests and end-to-end tests of the CLI. Run them on the pinned nightly from the workspace root: `cargo test --workspace --all-features --all-targets` runs everything except the doctests, which need `cargo test --workspace --all-features --doc`. The GPU tests print `SKIPPED` and pass on a machine without an adapter. The Python crate is outside the workspace and is tested with `--manifest-path` (see "Python"). To run one package at a time, `cargo test -p <package>` covers the library tests, the integration tests in `tests/` and the doctests, and does not build the Criterion bench (`harness = false`):

```bash
cargo test -p z-stackr-core
cargo test -p z-stackr-align                      # without the AKAZE / RANSAC tests
cargo test -p z-stackr-align --features akaze     # the AKAZE / RANSAC tests as well
cargo test -p z-stackr-algo
cargo test -p z-stackr-nn                         # default: ndarray + autodiff
cargo test -p z-stackr-nn --no-default-features --features wgpu,autodiff --no-run   # compile only: running needs a GPU
cargo test -p z-stackr-pipeline
cargo test -p z-stackr-pipeline --features akaze
cargo test -p z-stackr-cli
cargo test -p z-stackr-cli --features akaze
cargo test -p z-stackr-gui                        # default features (akaze)
cargo test -p z-stackr-gui --features nn
cargo test -p z-stackr-gui --no-default-features
```

The lint gate is the same matrix with `cargo clippy ... --tests` (plus `--benches` for `z-stackr-algo`); it must report zero warnings. `.github/workflows/ci.yml` defines both matrices; it has not run on a hosted runner yet.

**What the tests write and need**

* Tests write only into the system temp directory (unique per test and process, removed afterwards) or, for the quality metrics, under `target/quality/`; `cargo test` leaves the working tree unchanged.
* The end-to-end tests (`apps/stacker-cli/tests/e2e_test.rs`) need the committed `sample/` frames. When `sample/` is missing they fail with a message naming the directory; set `Z_STACKR_ALLOW_MISSING_SAMPLE=1` to turn that into an explicit `SKIPPED:` line on stderr.
* The quality tests against a reference stack (`test_quality_vs_zerene_*` in `apps/stacker-cli/src/pipeline/tests.rs`) are `#[ignore]`d because they take minutes; run them with `cargo test -p z-stackr-cli --features akaze -- --ignored --nocapture test_quality_vs_zerene`. Their SSIM gates (0.40, and 0.30 at low resolution) were set before the alignment-direction fix and have not been re-measured since.
* The reference image `sample/result/zerene.jpg` is the output of a commercial product and is used only by those quality tests. It stays in the tree until the owner has confirmed that its licence allows redistribution.
* Wall-clock speed-up checks are `#[ignore]`d too and print timings without asserting on them. The GPU backend tests are not run in CI.

---

## Contributing
Contributions, bug reports and feature requests are welcome. Open an issue or submit a pull request.

## License
This project is licensed under either of

* the Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE)), or
* the MIT license ([`LICENSE-MIT`](LICENSE-MIT)),

at your option. Every package declares `license = "MIT OR Apache-2.0"`.
