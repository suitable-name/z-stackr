# z-stackr-cli

The command-line interface of **z-stackr** (package `z-stackr-cli`, binary `z-stackr`).

`z-stackr-cli` runs the whole focus-stacking pipeline from a folder of frames to one output image. It is meant for automation, batch processing and headless machines. All per-frame and per-tile work is done by the shared `z-stackr-pipeline` crate (the desktop app calls the same stages); the CLI adds the tile loop, the scratch directory, the progress bars and the output-path rules.

## Key capabilities

* **Out-of-core fusion.** Each aligned frame is written to a temporary directory as padded tiles, and the fusion pass reads one padded tile of every frame at a time. The alignment pass holds whole frames (about 39 bytes per pixel, independent of the number of frames); the fusion pass holds about `N * 28` (Apex) or `N * 16` (Relief) bytes per padded tile pixel, Strata a fixed working set per padded tile pixel; the scratch disk takes `12 * ((T + 2A) / T)^2` bytes per source pixel per frame (`T` tile size, `A` apron). The per-pixel figures are in `crates/stacker-pipeline/README.md`. The loop is `stacker_pipeline::tiled`, shared with the desktop app and the Python bindings.
* **Three algorithms.** `apex` (Laplacian pyramid), `relief` (depth map, with the guided-filter or the multigrid solver) and `strata` (guided fusion). `both` (Relief and Apex) and `all` (Strata, Relief and Apex) align once and fuse the same tile store once per algorithm, one output file each.
* **Formats.** Reads `.jpg`/`.jpeg`, `.png` and `.tif`/`.tiff`, including 16-bit PNG and TIFF, and, in a build with `--features raw`, camera RAW files; writes PNG and TIFF at 16 bits per channel by default, JPEG at 8 bits.
* **Metadata.** With `image_saving.copy_metadata = true`, the EXIF block of the first frame is written into a PNG or JPEG result (orientation 1, because the frames are rotated upright). A TIFF result cannot carry it; one warning says so.
* **Alignment.** A coarse-to-fine subpixel intensity alignment of every frame to the first one. The mode comes from the settings file (`--config`): `registration` (the default: pyramid search for shift, scale and rotation), `translation` (shift and uniform scale), `affine` (the full six-parameter model with aspect and shear), `none` (no seed, no refinement, no warp) or, in an `nn` build, `neural` (a trained network predicts the matrices; see `--align-model`). `bounded_search = true` bounds the search; `optimizer` (or `--optimizer`) picks Lucas-Kanade, the Nelder-Mead simplex or both; `correct_brightness`, `chain_alignment` and `crop_to_common_area` are described in the settings file.
* **Auto-crop to the common area.** With `crop_to_common_area` (the default) the result is cropped to the rectangle covered by every aligned frame, removing the empty borders that warping leaves; a common rectangle below half of the frame area is refused and the output stays uncropped (a warning names the frame that covers least).
* **Batch.** With `--stacks subfolders`, every subfolder of `--input-dir` that holds frames is stacked on its own.
* **Optional AKAZE seed.** In a build with `--features akaze`, `akaze_seeding = true` in the settings file seeds the refinement from feature matches. The feature alone changes nothing at run time. Without the feature the setting is reported once and ignored.
* **Optional AI stacking and neural alignment.** In a build with `--features nn` (CPU) or `--features nn-gpu` (GPU), `--mode ai` runs a trained fusion model from the `models/` folder, and `alignment_mode = "neural"` an alignment model. Without these features the CLI has no ML dependency and rejects `--mode ai`, `--model`, `--device` and `--align-model`, and `alignment_mode = "neural"` stops the run before any work.

## Usage

```bash
z-stackr \
    --input-dir <PATH> \
    --output-file <PATH> \
    --mode <MODE> \
    --tile-size <SIZE> \
    [--optimizer <auto|lk|nm>] \
    [--stacks <subfolders|single>] \
    [--config <PATH> [--write-default-config]] \
    [--monitor] \
    [--log-file <PATH>] \
    [--no-gpu] \
    [--model <NAME>] [--device <cpu|gpu>] [--align-model <NAME>]
```

`--no-gpu` exists in builds with the `gpu` or `nn-gpu` feature; `--model`, `--device` and `--align-model` in builds with `nn`.

### Arguments

* `--input-dir` (required): the folder with the bracketed frames. Files are used in natural file-name order (`frame_2` before `frame_10`, case-insensitive); the first one is the alignment reference. The output file and the log file are never read as frames, even when they sit in this directory. See `--stacks` for a folder with subfolders.
* `--output-file` (required): the result. The extension (`.png`, `.tif`/`.tiff`, `.jpg`/`.jpeg`) selects the format; a name without an extension gets the extension of `image_saving.output_format`; any other extension is rejected before work starts. The path is checked (parent directory created, file writable) before the first frame is decoded. PNG and TIFF are written at 16 bits per channel unless `image_saving.bit_depth = 8`; JPEG is 8-bit at `image_saving.jpeg_quality`. With `--mode both` or `all` the file of each algorithm gets `_<algorithm>` before the extension (`result.png` becomes `result_relief.png` and `result_apex.png`); no file is written under the plain name. With `--stacks subfolders` this is the output directory. See "Settings file" for the `image_saving` rules.
* `--mode` (required): the fusion algorithm, in any letter case: `apex` (Laplacian pyramid), `relief` (depth map), `strata` (guided fusion), `both` (Relief and Apex), `all` (Strata, Relief and Apex) or `ai` (alias `nn`; the learned model, nn builds only). The old names `pmax` (now `apex`) and `dmap` (now `relief`) are still accepted and logged as renamed.
* `--tile-size` (required): the processing tile size in pixels, at least `64`; 512 is a good value. For `apex` it is rounded up to a multiple of `2^(pyramid_levels - 1)`. Every frame is stored on disk as padded tiles during the run, so the apron adds to the size: the padded tile is `T + 2A` on a side, with `A` = 128, 256 or 512 px for `apex` at 6, 7 or 8 pyramid levels (8 is the default), `estimation radius + 2 * smoothing radius + 1` for `relief`, about 100 for `strata` and 128 for `ai`. With `both` and `all` the tiles are stored once with the largest apron any of the algorithms needs. At `apex` defaults and `T` = 512 that is 108 bytes per source pixel per frame; the CLI logs the figure for the current settings before it starts and warns when a stored tile is more than 4 times its interior (a larger `--tile-size` or fewer `pyramid_levels` reduces it). The tile directory (`stacker_tiles_*`, below the system temp directory) is removed when the run ends, also after an error. A tile larger than the image is clamped to the image.
* `--optimizer` (optional): `auto` (Lucas-Kanade, the simplex as fallback), `lk` (Lucas-Kanade only) or `nm` (the Nelder-Mead simplex only), in any letter case. Overrides `optimizer` in the settings file for this run; the file is not written back.
* `--stacks` (optional): what to do when `--input-dir` has subfolders that hold frames. `subfolders` stacks every such subfolder on its own into `--output-file`, which is then a directory (created if needed; a path with an image extension is refused). Each file is named by `image_saving.filename_template` with `{name}` replaced by the subfolder name, plus the extension of `output_format`; the stacks run one after the other, a failing stack is reported and the others still run, a summary is printed, and the exit status is an error when any stack failed. `single` stacks only the images directly in `--input-dir`. Without the flag, a terminal is asked which is meant (`[s]` or `[d]`), and without a terminal the run stops and asks for the flag. The folder named by `--output-file` is never a stack, so the results can sit in the input folder. Only the direct subfolders are looked at, not their subfolders. `--monitor` and subfolder mode exclude each other.
* `--no-gpu` (optional; builds with `--features gpu` or `nn-gpu`): keep this run off the graphics card. The same as `compute_device = "cpu"` in the settings and `--device cpu` for the networks; it contradicts `--device gpu`.
* `--model` (AI mode; `nn`/`nn-gpu` builds only, other builds reject the flag): the file stem of a fusion model in the `models/` folder. Defaults to the first one in discovery order (`focusmerge_latest`, then `focusmerge_final`, then the highest `focusmerge_epoch_NNN`, then other names A to Z). Alignment models are never offered. The model is loaded before alignment starts.
* `--device` (AI mode and neural alignment; `nn`/`nn-gpu` builds only): `cpu` or `gpu`, in any letter case. Defaults to the best backend in the build.
* `--align-model` (neural alignment; `nn`/`nn-gpu` builds only): the file stem of an alignment model in the `models/` folder, used with `alignment_mode = "neural"`. Defaults to the first alignment model found; fusion models are never offered. Without `alignment_mode = "neural"` the flag is ignored with a warning. The network runs on thumbnails of the whole stack before the frames are aligned; with `neural_refine_classically` (the default) its matrices seed the classical refinement.
* `--config` (optional): the path of a TOML settings file (the format the GUI writes). The file must exist and parse; an error stops the run instead of silently using defaults. Without `--config` the built-in defaults are used; the CLI never reads the GUI's own settings file unless you pass it here. With `--write-default-config` a missing file is created with commented defaults first.
* `--monitor` (optional): keep watching `--input-dir`. The CLI stacks once, then re-runs the pipeline (after a 500 ms pause) whenever a frame is added or changed. A failing pass is reported and the watch continues.
* `--log-file` (optional): also write the structured log (traces and timings) to this file.

### Settings file

Every key of the settings file is read (39 keys, and `settings_version`); each reaches the pipeline except where noted. A file written before `settings_version` existed is migrated (its `affine` becomes `registration`), and the old `dmap_*` keys and `use_gpu` are still read.

* Alignment and fusion: `alignment_mode`, `bounded_search`, `akaze_seeding` (needs `--features akaze`), `optimizer`, `correct_brightness`, `chain_alignment`, `neural_refine_classically`, `pyramid_levels`, `use_all_color_channels`, `grit_suppression`, `relief_estimation_radius`, `relief_smoothing_radius`, `relief_contrast_pct`, `relief_auto_detect`, `relief_use_multigrid`, `strata_base_radius`, `strata_detail_focus`.
* `compute_device` (`cpu`, the default, or `gpu`): where the alignment and the Apex and Relief fusions run (Strata runs on the CPU in the CLI). `gpu` needs a build with `--features gpu`; the alignment objective, the final warp of each frame and the fusion of every tile then run on the graphics card, and the CPU takes over, with a logged warning, if there is no usable adapter or the device fails (for the rest of the run), or a call needs more than the 3 GiB device budget (for that call). The result agrees with the CPU within a small tolerance, not bit for bit. The AI mode ignores it. The log's alignment and fusion lines name the device in use.
* Frame set: `auto_cull` (frames that win too little of the in-focus detail pixels are dropped: `auto_cull_threshold_pct` percent, or with 0 the frame-count-scaled rule `min(2, 50 / N)`; if that would drop every frame, all are stacked), `sort_by_sharpness` (the kept frames are fused in census order instead of stack order; the multigrid solver needs stack order), `stack_every_nth`, `preprocessing.sort_reverse` (the file order is reversed first, then every N-th frame is kept), and `preprocessing.pre_rotation` / `pre_crop_enabled` + `pre_crop_spec` / `pre_resize_percent` (applied to every frame, after the EXIF orientation, in that order).
* `crop_to_common_area` (the default) crops the result to the area every frame covers. `resize_cropped_to_original` has no effect in the CLI: the out-of-core path keeps the cropped size.
* `preprocessing.ignore_exif_orientation`: by default the EXIF orientation of a JPEG is applied, so the frames come out upright; set the key to `true` to load the pixels as stored.
* Output: `image_saving.bit_depth`, `image_saving.jpeg_quality`, `image_saving.output_format` (the extension used when `--output-file` has none), `image_saving.filename_template` and `image_saving.default_output_dir` (a bare `--output-file` name is placed in the default directory; an `--output-file` that is an existing directory or ends in a path separator gets `filename_template`, where `{name}` is the stem of the first frame, plus the extension of `output_format`), and `image_saving.copy_metadata` (the first frame's EXIF into a PNG or JPEG result).
* Accepted but without effect: `relief_show_preview` (the contrast preview is a GUI feature) and `tile_size` (the CLI tiles through `--tile-size`, which always wins).

Wrongly typed values and unknown enum values stop the run with a message that names the file. Numeric values outside their range are clamped and logged. All frames must have the same size after preprocessing; a different size is an error naming the file.

> **Models folder.** `ai` mode and neural alignment look for models in a `models/` directory next to the executable and then in the current working directory. Each model is a `<name>.mpk` weights file plus a `<name>.json` architecture manifest, produced by `z-stackr-train` (see the project README).

## Examples

Standard stack using Apex:
```bash
z-stackr --input-dir ~/Pictures/Macro/Stack01 --output-file ~/Pictures/Macro/Stack01_Final.tif --mode apex --tile-size 512
```

Stack using Relief for a continuous surface, saving trace logs:
```bash
z-stackr --input-dir ~/Pictures/Coins/Stack02 --output-file ~/Pictures/Coins/Final.png --mode relief --tile-size 1024 --log-file ~/stack_log.txt
```

All three algorithms from one alignment (writes `Final_strata.png`, `Final_relief.png` and `Final_apex.png`):
```bash
z-stackr --input-dir ~/Pictures/Coins/Stack02 --output-file ~/Pictures/Coins/Final.png --mode all --tile-size 512
```

Every subfolder of a folder as a stack of its own (writes `Shoot/results/<subfolder>_stacked.png`):
```bash
z-stackr --input-dir ~/Pictures/Shoot --output-file ~/Pictures/Shoot/results --mode strata --tile-size 512 --stacks subfolders
```

Force the Nelder-Mead simplex for this run:
```bash
z-stackr --input-dir ./frames --output-file ./out.tif --mode apex --tile-size 512 --optimizer nm
```

Create a commented settings file with the defaults (this run uses the defaults), then edit `stack.toml` and pass it again:
```bash
z-stackr --input-dir ./frames --output-file ./out.tif --mode apex --tile-size 512 --config ./stack.toml --write-default-config
z-stackr --input-dir ./frames --output-file ./out.tif --mode apex --tile-size 512 --config ./stack.toml
```

## Building

```bash
# Standard build (binary: z-stackr)
cargo build --release -p z-stackr-cli

# With AKAZE feature matching available
cargo build --release -p z-stackr-cli --features akaze

# With the AI ("neural") stacking mode, CPU inference
cargo build --release -p z-stackr-cli --features nn

# ... or with GPU (wgpu/Vulkan) inference as well (implies `nn`)
cargo build --release -p z-stackr-cli --features nn-gpu

# Apex, Relief and alignment on the GPU (`compute_device = "gpu"` in the settings file); independent of `nn-gpu`
cargo build --release -p z-stackr-cli --features gpu

# With camera RAW files as input frames (native only)
cargo build --release -p z-stackr-cli --features raw
```

The SIMD code is compiled for the baseline CPU unless you pass `RUSTFLAGS`; see "CPU features" in the root README.
