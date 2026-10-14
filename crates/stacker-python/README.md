# z-stackr for Python

Python bindings for **z-stackr**, the focus-stacking engine. The Python import name is
**`zstackr`**; the PyPI name is `z-stackr`; the Cargo package is `z-stackr-python` (library
`stacker_python`).

The crate is a thin layer over the crates the command line tool and the desktop app use
(`z-stackr-pipeline`, `z-stackr-align`, `z-stackr-algo`, `z-stackr-core`). Nothing in it
reimplements a stage, so a script gets the pixels the applications produce for the same
settings.

Two ways in:

| | `stack_files`, `batch_stack` | `stack_arrays` |
|---|---|---|
| Input | image files | a list of numpy `[H, W, 3]` arrays |
| Memory | out of core with `tile_size > 0`: bounded by the tile, not by the stack | the whole stack, plus working buffers, in RAM |
| Output | a file (`.tif`, `.png`, `.jpg`) | an array of the input's dtype |
| `auto_cull`, `sort_by_sharpness` | honoured (in the out-of-core run from the focus census of the tile store) | honoured |
| AI fusion, neural alignment | not available from Python | not available from Python |

## Build

The crate is **not** a member of the main Cargo workspace (it has its own `[workspace]`, its
own `Cargo.lock` and `target` directory), so `cargo test --workspace` never links libpython.
Build it with [maturin](https://www.maturin.rs/), from this directory:

```bash
pip install maturin numpy pytest

maturin develop --release        # editable install into the current environment
maturin build --release          # a wheel in target/wheels
python -m pytest tests/test_basic.py
```

The toolchain is the one of the repository (`rust-toolchain.toml` at the root is picked up from
this directory). `pyproject.toml` builds with the features `python` and `extension-module`;
add the optional ones with `--features`:

| Feature | Effect |
|---|---|
| `python` | the PyO3 / numpy bindings (maturin enables it) |
| `extension-module` | link as an extension module (maturin enables it; leave it off for `cargo test`) |
| `gpu` | the GPU kernels (settings key `compute_device = "GPU"`); the CPU is the fallback |
| `raw` | camera RAW input for `stack_files`, `batch_stack` and `load_image` |
| `akaze` | AKAZE seeding of the alignment (`akaze_seeding`) |
| `nn` | reserved: Python does not select a neural model yet |

```bash
maturin build --release --features gpu,raw
```

### Checking the Rust side

```bash
cargo clippy --manifest-path crates/stacker-python/Cargo.toml --all-targets
cargo test   --manifest-path crates/stacker-python/Cargo.toml          # no Python needed
cargo test   --manifest-path crates/stacker-python/Cargo.toml --features python   # links libpython
```

Without the `python` feature the crate builds and tests its pure-Rust parts: the settings
mirror, the option strings, the pixel converters, the progress mapping and the stacking flows
(`tests/convert.rs`, `settings.rs`, `progress.rs`, `stack.rs`). With it, `tests/python_api.rs`
runs the numpy layer against a linked interpreter.

## Quick start

```python
import zstackr

settings = zstackr.PyStackingSettings()          # the defaults of the applications
params = zstackr.PyPipelineParams(
    paths=["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"],
    output_file="stacked.tif",                   # the extension picks the format
    mode="apex",
    tile_size=512,                               # 0 = hold the stack in RAM
)
zstackr.stack_files(params, settings, progress=lambda stage, cur, total: print(stage, cur / total))
```

```python
import numpy as np, zstackr

frames = [np.asarray(f) for f in captured_frames]    # uint8, uint16 or float32, [H, W, 3]
settings = zstackr.PyStackingSettings(alignment_mode="Translation", auto_cull=True)
stacked = zstackr.stack_arrays(frames, settings, mode="strata")   # same dtype as the input
```

A settings file written by the GUI works as it is:

```python
settings = zstackr.load_config("my_settings.toml")
results = zstackr.batch_stack("folders/", "stacked/", settings, mode="relief")
```

## Colour

The pipeline computes in **linear light**; files and arrays are sRGB-encoded.

* `stack_files`: the files are decoded like the CLI and the GUI decode them (exact sRGB
  tables, 16-bit files keep their precision) and the result is encoded with the exact
  rounding encoder; 16-bit PNG and TIFF keep 16 bits.
* `stack_arrays(..., encoding="srgb")` (default): `uint8` and `uint16` samples are sRGB codes,
  `float32` samples are sRGB-encoded values in nominally `0.0..=1.0`. They are converted to
  linear light, stacked, and encoded back, so **the result has the dtype and the encoding of
  the input**.
* `encoding="linear"`: the samples are taken as linear light and no transfer function is
  applied in either direction (scientific data, for example).

On the way in nothing is clamped (a non-finite `float32` sample reads as 0). On the way out
every sample is clamped to `0.0..=1.0` and rounded, never truncated.

## API

### Functions

**`stack_files(params, settings, progress=None) -> None`**
Stack `params.paths` into `params.output_file`. The format is the extension of the output file
(`.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`); bit depth, JPEG quality and `copy_metadata` come
from `settings.image_saving`. `stack_every_nth` and `preprocessing` apply; a file that is the
output file itself is left out. `copy_metadata` copies the EXIF block of the first frame into
a JPEG or PNG result (a TIFF cannot carry it and a warning says so). With
`resize_cropped_to_original` the out-of-core run keeps the cropped size.

**`batch_stack(input_dir, output_dir, settings, mode="apex", tile_size=512) -> list[tuple[str, bool, str]]`**
Stack every direct sub-folder of `input_dir` that holds images, one after the other. The file
of a folder is `image_saving.filename_template` with `{name}` replaced by the folder name, plus
the extension of `image_saving.output_format`. Returns `(folder, succeeded, path_or_message)`
per folder; a folder that fails does not stop the batch. One level of folders only, no monitor
mode, no prompt.

**`stack_arrays(frames, settings, mode="apex", encoding="srgb", progress=None) -> ndarray`**
Stack arrays that are already in memory. All frames share one shape and one dtype and must be
C-contiguous (`numpy.ascontiguousarray` fixes a view). The first frame is the alignment
reference. The settings of alignment, cull and fusion apply, as do `stack_every_nth` and
`preprocessing.sort_reverse`; the rotation, crop and resize of `preprocessing` do not (the
pixels are yours). The result is `[H', W', 3]`, smaller than the input when
`crop_to_common_area` cropped it.

**`load_config(path) -> PyStackingSettings`**, **`save_config(path, settings)`**
The settings file of the GUI and of the CLI's `--config`. Reading is as lenient as in the
applications: missing keys keep their default, the old spellings (`dmap_*`, `use_gpu`,
`relief_engine`, `alignment_mode = "bounded"`) are accepted, and a file from before settings
version 2 is migrated. Every value that had to be replaced or clamped, and every unknown key,
is raised as a `UserWarning`.

**`load_image(path) -> ndarray`**
One frame as a `uint16` `[H, W, 3]` array of sRGB codes (8-bit files are scaled to the 16-bit
range, the EXIF orientation of a JPEG is applied). RAW files need the `raw` feature.

`mode` is `"apex"`, `"relief"` or `"strata"` (the older `"pmax"` and `"dmap"` work).
`"ai"` is rejected with a `ValueError`.

### Classes

**`PyPipelineParams(paths, output_file, mode="apex", tile_size=512)`**
`tile_size` is the edge of the out-of-core tiles (at least 64). `0` falls back to
`settings.tile_size`; `0` there as well holds the whole stack in RAM.

**`PyStackingSettings(...)`**: every key of the settings file under its own name, all keyword
arguments optional. Enums are strings, accepted in any letter case:

| Key | Type | Default | Values / range |
|---|---|---|---|
| `settings_version` | int | `2` | leave it |
| `alignment_mode` | str | `"Registration"` | `Translation`, `Registration`, `Affine`, `None`, `Neural` (without a model: `Registration`, with a warning) |
| `optimizer` | str | `"Auto"` | `Auto`, `Lucas-Kanade`, `Nelder-Mead` |
| `bounded_search`, `akaze_seeding`, `chain_alignment` | bool | `False` | |
| `correct_brightness` | bool | `True` | |
| `neural_refine_classically` | bool | `True` | |
| `crop_to_common_area` | bool | `True` | |
| `resize_cropped_to_original` | bool | `False` | |
| `auto_cull` | bool | `True` | |
| `auto_cull_threshold_pct` | float | `0.0` | `0.0..=50.0` (`0.0` = automatic) |
| `sort_by_sharpness` | bool | `False` | |
| `stack_every_nth` | int | `1` | `>= 1` |
| `tile_size` | int | `0` | `0` or `64..=8192` |
| `pyramid_levels` | int | `8` | `2..=12` |
| `use_all_color_channels` | bool | `False` | |
| `grit_suppression` | bool | `True` | |
| `relief_estimation_radius` | int | `5` | `0..=100` |
| `relief_smoothing_radius` | int | `2` | `0..=50` |
| `relief_contrast_pct` | float | `0.0` | `0.0..=1.0` |
| `relief_show_preview` | bool | `False` | no effect from Python |
| `relief_auto_detect` | bool | `False` | |
| `relief_use_multigrid` | bool | `False` | |
| `strata_base_radius` | int | `31` | `8..=64` |
| `strata_detail_focus` | int | `3` | `1..=5` |
| `compute_device` | str | `"CPU"` | `CPU`, `GPU` |
| `preprocessing` | `PyPreprocessingSettings` | | |
| `image_saving` | `PyImageSavingSettings` | | |

Two views of the real keys are kept from earlier versions: `relief_engine`
(`"guided_filter"` or `"multigrid"`, for `relief_use_multigrid`) and `use_gpu` (a bool, for
`compute_device`); both are properties and constructor arguments.

A value outside its range or an unknown option string raises `ValueError` naming the key and
the range **when the settings are used** (`stack_*`, `batch_stack`, `save_config`). The
applications clamp instead, which suits a hand-edited file; code that passes `pyramid_levels=99`
has made a mistake.

**`PyPreprocessingSettings(pre_rotation=0, pre_crop_enabled=False, pre_crop_spec="", pre_resize_percent=100, sort_reverse=False, ignore_exif_orientation=False)`**

**`PyImageSavingSettings(output_format="TIFF", bit_depth=16, jpeg_quality=95, filename_template="{name}_stacked", default_output_dir="", copy_metadata=False)`**

## Progress, Ctrl-C, warnings

`progress(stage, current, total)` is called while a stack runs. `stage` is one of

| `stage` | what runs |
|---|---|
| `"load"` | decoding the frames (the in-RAM run of `stack_files`) |
| `"align"` | aligning the frames to the reference |
| `"cull"` | the focus census and the auto-cull |
| `"fuse"` | fusing (tile by tile out of core; a two-pass engine counts both passes) |
| `"save"` | encoding and writing the result |

`total` is always `1000` and `current` is the progress of the stage in thousandths, so
`current / total` is the fraction and `current == total` ends the stage. A call is made when
`current` changes. (The older contract with `decode_raw`, `align_frame`, `fuse_tile` and
counts of frames and tiles is gone: the pipeline reports fractions now.)

The compute runs with the GIL released; the GIL is taken for the length of one `progress`
call. A callback that raises has its exception printed to stderr and is ignored. **Ctrl-C**
stops a stack between frames and tiles with `KeyboardInterrupt` (on the main thread). Notes
from the pipeline (a frame whose alignment fell back, a refused crop, a GPU that was given
up, a frame that could not be read) arrive as `UserWarning`.

Errors: `ValueError` for a request that cannot be honoured (bad settings or option, no frames,
frames of different sizes, a tile size below 64, an output path without a known extension),
`RuntimeError` for failures (unreadable frame, write error), `KeyboardInterrupt` for Ctrl-C.

## Differences from the version on GitHub

* The pipeline is the one of this repository: linear-light colour, the alignment of
  `Registration` with the inverse warp, out-of-frame masking, first-wins ties, and the
  frame-count-scaled auto-cull; `stack_files` honours `auto_cull` and `sort_by_sharpness`.
* `stack_files` and `batch_stack` run `stacker_pipeline::tiled::run_tiled`; `tile_size = 0`
  holds the stack in RAM. `stack_arrays` runs the stages of `stacker_pipeline::session`.
* `PyStackingSettings` mirrors the 39 keys of this repository (`compute_device` instead of
  `use_gpu`, `relief_use_multigrid` next to `relief_engine`, plus `bounded_search`,
  `chain_alignment`, `crop_to_common_area`, ...).
* `PyPipelineParams` has no `model`, `device` or `align_model`, and `mode="ai"` is rejected.
* The progress stages and the colour behaviour (`encoding=`) are as described above.
