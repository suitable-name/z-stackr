# z-stackr-python

Python bindings for **z-stackr** (crate identifier `stacker_python`, Python import name **`zstackr`**), a
pure-Rust, out-of-core focus-stacking engine. This crate is a thin PyO3 binding layer: every function here
forwards directly into the same shared Rust crates the CLI (`z-stackr-cli`) and GUI (`z-stackr-gui`) call
(`z-stackr-pipeline`, `z-stackr-align`, `z-stackr-algo`, `z-stackr-core`) — nothing here is a parallel
reimplementation that could silently drift from the two shipped applications.

## What / why

Focus stacking blends a sequence of images captured at different focus depths into one fully-sharp
composite. z-stackr's Rust core is built around an **out-of-core tiled pipeline**: peak memory scales with a
configurable tile size, not with the number of frames or their resolution, so stacks of 100+ high-resolution
16-bit frames process without exhausting RAM. This binding exposes that engine to Python so it can be driven
from automation scripts, Jupyter notebooks, or microscopy/lab acquisition pipelines without shelling out to
the CLI binary or writing Rust.

Typical use cases:

- **Automated microscopy / macro photography pipelines** — trigger a stack the moment a capture sequence
  finishes, without a human running the CLI or GUI.
- **Batch processing** many per-subject folders (`batch_stack`) as part of a larger Python-orchestrated
  workflow (e.g. alongside image cataloguing, metadata extraction, or a ML training-data pipeline).
- **Live-acquisition buffers already in memory** — a numpy array captured straight from a camera SDK, stacked
  without ever touching disk (`stack_arrays`).
- **GUI-configured, script-executed workflows** — dial in stacking settings visually in `z-stackr-gui`, save
  a config file, then run headlessly from Python with the exact same settings (`load_config`).

## Installation

### Prerequisites

- **Rust nightly** — this crate (transitively, via `z-stackr-align`/`z-stackr-algo`) uses
  `#![feature(portable_simd)]`, so building from source requires a nightly toolchain
  (`rustup default nightly`, or prefix commands with `cargo +nightly …`).
- **[maturin](https://www.maturin.rs/)** (`pip install maturin`), the PyO3/Rust <-> Python packaging tool
  this crate's `pyproject.toml` is configured for.

### PyPI package name vs. Python import name

This is intentionally **not** a matched pair — call it out explicitly so it never causes a confusing
`ModuleNotFoundError`:

| | Name | Where it's set |
|---|---|---|
| PyPI / `pip install` name | `z-stackr` | `pyproject.toml` → `[project] name` |
| Python `import` name | `zstackr` | `pyproject.toml` → `[tool.maturin] module-name`, and `#[pymodule] fn zstackr` in `src/lib.rs` |
| Cargo package name | `z-stackr-python` | `Cargo.toml` → `[package] name` |
| Rust crate identifier (for `cargo test`, `use stacker_python::…`) | `stacker_python` | `Cargo.toml` → `[lib] name` |

```bash
pip install z-stackr
python -c "import zstackr; print(zstackr.__doc__)"
```

### Building from source (development)

```bash
cd crates/stacker-python

# Editable install into your current Python environment — fastest loop for development.
maturin develop --release

# Or build a wheel for distribution:
maturin build --release
```

### Feature flags

| Feature | Effect | Included in the published wheel / a plain `maturin build`? |
|---|---|---|
| `python` | Compiles the PyO3 bindings themselves (always required to build this crate at all) | Yes |
| `extension-module` | Links against Python as an extension module rather than embedding an interpreter (required for a real wheel; omit only for `cargo test`, which needs libpython linked directly) | Yes |
| `gpu` | GPU-accelerated tiled/batch Apex fusion, production warp, Strata saliency, and Relief guided-filter/multigrid — forwards to `z-stackr-pipeline/gpu` and `z-stackr-core/gpu`. Falls back to CPU automatically at run time when no compatible GPU/driver is found. | **Yes** — default since this crate's `[tool.maturin] features` list in the workspace-root `pyproject.toml` includes it |
| `raw` | Camera RAW input (CR2/CR3/NEF/ARW/DNG/RAF/ORF/RW2/PEF/…) — forwards to `z-stackr-pipeline/raw` | **Yes** — same as `gpu` above |
| `nn` | AI ("neural") fusion mode and neural alignment mode — forwards to `z-stackr-pipeline/nn` | **No** — opt-in |

`nn` is the one feature still opt-in, since it pulls in the [Burn](https://burn.dev/) ML framework — a
heavy dependency most installs don't need (see the root project README's "No ML by default" note). Enable
it explicitly when building a wheel that needs AI fusion/neural alignment:

```bash
maturin build --release --features nn
```

Note the Cargo-level default in `crates/stacker-python/Cargo.toml` (`default = []`) is empty — `gpu`/`raw`
being "on by default" is a property of the **published wheel's build configuration**
(`pyproject.toml`'s `[tool.maturin] features`), not of the crate itself when built as a plain Rust
dependency. Building this crate directly with Cargo (rather than through `maturin`/`pyproject.toml`) starts
from nothing enabled, same as any other feature-gated crate.

### `cargo test` vs. wheels

`cargo test --features python` runs this crate's Rust-side test suite directly against a linked
`libpython` (via the `auto-initialize` dev-dependency feature) — no `maturin` or wheel involved. This is
what CI runs. `maturin build`/`develop` (with `extension-module` added) is the separate step that produces
the actual importable `zstackr` Python module.

## Quickstart

```python
import zstackr

params = zstackr.PyPipelineParams(
    paths=["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"],
    output_file="stacked.tiff",
    mode="apex",
    tile_size=512,
)
settings = zstackr.PyStackingSettings()  # defaults: Registration alignment, Apex-friendly, crop-to-common-area on

zstackr.stack_files(params, settings)
print("done ->", params.output_file)
```

## The two paths: file (out-of-core) vs. array (in-RAM)

| | `stack_files` (+ `batch_stack`) | `stack_arrays` |
|---|---|---|
| Input | File paths on disk | Python list of numpy `[H, W, 3]` arrays already in memory |
| Memory model | Out-of-core, tiled — bounded by `tile_size`, independent of stack size/resolution | In-RAM — the whole stack (plus alignment/fusion working buffers) must already fit, since the caller already holds it |
| `auto_cull` / `sort_by_sharpness` | **Ignored** (logged warning), exactly like the CLI — the tiled pipeline never holds the full aligned stack at once | **Honoured** — this is the one path in this crate where they actually run |
| AI fusion / neural alignment | Supported (`nn` feature) | Not supported — use the file path for AI mode |
| When to use | Large stacks, many frames, high resolution, anything that might not fit in RAM | Stacks that already comfortably fit in memory (e.g. a live-acquisition buffer, an already-loaded microscopy capture) |

Use `stack_files` for anything where "decode every frame into memory at once" is a real risk of exhausting
RAM; use `stack_arrays` when you already hold the frames as numpy arrays and want `auto_cull`/
`sort_by_sharpness` to actually take effect.

## Full API reference

### Module functions

#### `stack_files(params: PyPipelineParams, settings: PyStackingSettings, progress: Optional[Callable] = None) -> None`

Runs the out-of-core, tiled pipeline (`z-stackr-pipeline::run_pipeline` — the same engine `z-stackr-cli`
always uses) and writes the result to `params.output_file`. Peak memory scales with `params.tile_size`, not
frame count or resolution.

- `progress`, if given, is called with `(stage: str, current: int, total: int)` at each pipeline stage
  boundary — see "Progress callback" below for the exact mapping.
- Raises `ValueError` if `settings.alignment_mode` / `settings.relief_engine` /
  `settings.image_saving.output_format` is not a recognised value.
- Raises `RuntimeError` if the pipeline itself fails (I/O error, unsupported `mode` string, alignment
  failure).
- `params.model` / `params.device` / `params.align_model` only take effect in an extension built with the
  `nn` feature — see "AI models" below.

#### `stack_arrays(frames: list[np.ndarray], settings: PyStackingSettings, mode: str = "apex") -> np.ndarray`

The in-RAM numpy counterpart — see "The two paths" above. `frames` must all share one shape and one dtype
(`uint8` `0..=255`, `uint16` `0..=65535`, or `float32` nominally `0.0..=1.0`). `mode` is `"apex"` (default),
`"relief"`, or `"strata"` (guided-filter soft-blend fusion; not `"ai"` — AI fusion needs `stack_files`/
`batch_stack`). Returns an array of the **same
dtype** as the input, shape `[H', W', 3]` (`H'`/`W'` may shrink if `crop_to_common_area` cropped the result
and `resize_cropped_to_original` wasn't set to restretch it back).

Raises `ValueError` for: an empty `frames` list; an invalid `mode`; an unsupported dtype; a non-contiguous
or wrongly-shaped array; mismatched shapes/dtypes across `frames`; or an invalid settings enum string.

See "The in-RAM stacking flow" below for the exact step-by-step algorithm and which settings fields are
honoured at each step.

#### `batch_stack(input_dir: str, output_dir: str, settings: PyStackingSettings, mode: str = "apex", tile_size: int = 512) -> list[tuple[str, bool, str]]`

Stacks every image-bearing direct subfolder of `input_dir` into its own output file inside `output_dir`,
using `z-stackr-pipeline::collect_image_paths` (the identical shared helper the CLI uses) to discover each
subfolder's frames — so this never drifts onto a hand-maintained extension list, and automatically respects
the `raw` feature gate.

Each subfolder's output filename is derived exactly like the CLI's own subfolder-batch mode
(`apps/stacker-cli/src/batch.rs`'s `resolve_batch_output_path`): `settings.image_saving.filename_template`
with `{name}` substituted by the subfolder's own directory name, plus the extension for
`settings.image_saving.output_format`.

Returns a list of `(subfolder_name, succeeded, output_path_or_message)` triples, one per discovered
subfolder, in processing order. A failing subfolder is recorded in this list (message in the third element,
`succeeded=False`) rather than aborting the whole batch, mirroring the CLI's own "a failing subfolder must
not abort the batch" behaviour.

**Honest differences from the CLI**: this function does not reimplement the CLI's interactive/`--stacks`
disambiguation prompt, its mixed-direct-images-and-subfolders handling, or `--monitor` mode — it only walks
`input_dir`'s direct subfolders that themselves directly contain images (non-recursive beyond one level,
matching the CLI's own discovery rule).

#### `load_config(path: str) -> PyStackingSettings`

Loads a `StackingSettings` TOML file — the **exact same format** `z-stackr-cli`'s `--config` flag and the
GUI's config dialog read/write. Missing fields fall back to their documented defaults, exactly like the
CLI's own loader (including the same `clamp_valid()` calls on every sub-struct after deserialising). See
"Config interop" below.

Raises `RuntimeError` if the file can't be read, `ValueError` if it isn't valid TOML / fails to parse (or
contains an invalid enum string).

#### `save_config(path: str, settings: PyStackingSettings) -> None`

Writes `settings` to a TOML file using the identical serde the CLI/GUI use, so the result is a drop-in
`--config` for the CLI or a loadable config for the GUI.

#### `load_image(path: str) -> np.ndarray`

Loads a single frame via `stacker_core::io::load_frame` (the same loader the pipeline itself uses) and
returns a `[H, W, 3]` `numpy.uint16` array, preserving bit depth regardless of the source file's own depth
(8-bit sources are scaled up to the full `u16` range by the underlying `image` crate conversion). Supports
RAW input only in a `raw`-feature build; otherwise raises `RuntimeError` with the same "rebuild with
`--features raw`" message `stacker_core::io::LoadError::RawSupportDisabled` carries.

### Classes

#### `PyPipelineParams`

| Field | Type | Default | CLI equivalent |
|---|---|---|---|
| `paths` | `list[str]` | — (required) | positional input files (via `--input-dir` + directory scan) |
| `output_file` | `str` | — (required) | `--output-file` |
| `mode` | `str` | `"apex"` | `--mode` |
| `tile_size` | `int` | `512` | `--tile-size` |
| `model` | `Optional[str]` | `None` | `--model` (AI mode, `nn` feature) |
| `device` | `Optional[str]` | `None` | `--device` (AI mode / neural alignment, `nn` feature) |
| `align_model` | `Optional[str]` | `None` | `--align-model` (neural alignment, `nn` feature) |

#### `PyStackingSettings`

A 1:1 field mirror of `stacker_core::settings::StackingSettings` — the same struct the GUI's Settings panel
edits and the CLI's `--config` TOML deserialises into. See the root project README's "GUI Settings Overview"
section for the narrative description of each setting; this table cross-references it to the Python field
name.

| Field | Type | Default | GUI label / notes |
|---|---|---|---|
| `alignment_mode` | `str` | `"Registration"` | Alignment Mode: `"Affine"` \| `"Translation"` \| `"Registration"` \| `"None"` \| (`nn` builds) `"Neural"` |
| `optimizer` | `str` | `"auto"` | Optimizer: `"auto"` (Lucas-Kanade, falling back to Nelder-Mead on error/RMS regression) \| `"lucaskanade"` \| `"neldermead"` |
| `akaze_seeding` | `bool` | `False` | AKAZE seeding |
| `neural_refine_classically` | `bool` | `True` | (Neural alignment hybrid mode) |
| `correct_brightness` | `bool` | `True` | Correct brightness |
| `auto_cull` | `bool` | `True` | Auto-Cull Frames — **only honoured by `stack_arrays`**, see "Limitations" |
| `sort_by_sharpness` | `bool` | `True` | Sort by sharpness — same caveat as `auto_cull` |
| `tile_size` | `int` | `0` | (GUI-only sentinel; not meaningful from this binding — carried for config round-tripping) |
| `stack_every_nth` | `int` | `1` | Stack every Nth frame |
| `pyramid_levels` | `int` | `8` | Apex: Pyramid levels |
| `use_all_color_channels` | `bool` | `False` | Apex: Use all color channels |
| `grit_suppression` | `bool` | `True` | Apex: Grit suppression |
| `relief_estimation_radius` | `int` | `5` | Relief: Estimation radius |
| `relief_smoothing_radius` | `int` | `2` | Relief: Smoothing radius |
| `relief_contrast_pct` | `float` | `0.0` | Relief: Contrast threshold % |
| `relief_show_preview` | `bool` | `False` | Relief: Show preview during stacking (no interactive effect from Python — carried for config round-tripping) |
| `relief_auto_detect` | `bool` | `False` | Relief: Auto-detect optimum contrast |
| `relief_engine` | `str` | `"guided_filter"` | Relief engine: `"guided_filter"` \| `"multigrid"` (mirrors the core `relief_use_multigrid: bool` field as a symbolic string) |
| `strata_base_radius` | `int` | `31` | Strata: Base radius (`8..=64`) |
| `strata_detail_focus` | `int` | `3` | Strata: Detail focus (`1..=5`) — higher = crisper depth edges / more detail retention on deep stacks, lower = smoother with fewer artifacts on flat/glossy subjects. Raises `ValueError` (naming the `1..=5` range) if out of range. |
| `crop_to_common_area` | `bool` | `True` | Crop to common area |
| `resize_cropped_to_original` | `bool` | `False` | Restretch to original size |
| `auto_cull_threshold_pct` | `float` | `2.0` | Cull threshold (%) |
| `preprocessing` | `PyPreprocessingSettings` | see below | Preprocessing section |
| `image_saving` | `PyImageSavingSettings` | see below | Image Saving section |

#### `PyPreprocessingSettings`

| Field | Type | Default | GUI label |
|---|---|---|---|
| `pre_rotation` | `int` | `0` | Pre-rotation (`0`/`90`/`180`/`270`) |
| `pre_crop_enabled` | `bool` | `False` | Pre-crop |
| `pre_crop_spec` | `str` | `""` | Crop spec (`"w,h"` or `"x,y,w,h"`) |
| `pre_resize_percent` | `int` | `100` | Pre-resize % (`10..=100`) |
| `sort_reverse` | `bool` | `False` | Reverse sort order |
| `ignore_exif_orientation` | `bool` | `False` | Ignore EXIF orientation |

#### `PyImageSavingSettings`

| Field | Type | Default | GUI label |
|---|---|---|---|
| `output_format` | `str` | `"TIFF"` | Output format: `"TIFF"` \| `"PNG"` \| `"JPEG"` |
| `bit_depth` | `int` | `16` | Bit depth (`8`/`16`) |
| `jpeg_quality` | `int` | `95` | JPEG quality (`1..=100`) |
| `filename_template` | `str` | `"{name}_stacked"` | Filename template |
| `default_output_dir` | `str` | `""` | Default output dir (unused by `stack_files`, which always takes an explicit path) |
| `copy_metadata` | `bool` | `False` | Copy metadata to output |

### Enum validation

`alignment_mode`, `optimizer`, `relief_engine`, and `image_saving.output_format` are plain Python strings
rather than a bound Rust enum type, for ergonomics. An unrecognised value raises `ValueError` **listing the
valid options** at the point the settings are converted (i.e. when passed into `stack_files`/`stack_arrays`/
`save_config`/`batch_stack`) — it is never silently substituted with a default:

```python
settings = zstackr.PyStackingSettings(alignment_mode="Sideways")
zstackr.stack_files(params, settings)
# ValueError: invalid alignment_mode "Sideways"; valid options are: Affine, Translation, Registration, None
```

## Progress callback

Pass a callable to `stack_files`'s `progress` parameter to receive `(stage: str, current: int, total: int)`
updates as the pipeline runs:

```python
def on_progress(stage, current, total):
    print(f"{stage}: {current}/{total}")

zstackr.stack_files(params, settings, progress=on_progress)
```

Mapping (from `stacker_pipeline::PipelineProgress`):

| `stage` | `current` | `total` | Meaning |
|---|---|---|---|
| `"decode_raw"` | 1-based RAW frame index | RAW frame count | RAW decode-once pre-pass (only emitted when at least one input is RAW) |
| `"align_start"` | `0` | frame count | Alignment pre-pass starting |
| `"align_frame"` | 1-based frame index | frame count | One frame aligned + committed |
| `"align_done"` | `0` | `0` | Alignment pre-pass finished |
| `"fuse_start"` | `0` | tile count | Tiled fusion pass starting (not emitted in whole-image Apex mode) |
| `"fuse_tile"` | 1-based tile index | tile count | One tile fused + written |
| `"fuse_done"` | `0` | `0` | Fusion pass finished |
| `"encoding"` | `0` | `0` | Encoding/writing the final image |

**GIL behaviour**: the whole pipeline computation runs with the GIL released (so other Python threads —
e.g. a UI event loop — keep running during a multi-minute stack); the GIL is re-acquired only for the
duration of each individual `progress(...)` call. **If `progress` raises**, the exception is printed to
stderr and ignored — a misbehaving callback can never abort an otherwise-successful stack.

## Config interop with the GUI/CLI

Because `load_config`/`save_config` use the identical TOML serde `StackingSettings` derives, you can:

1. Open `z-stackr-gui`, load your frames, dial in alignment/fusion/preprocessing settings visually, and use
   its config-save dialog to write a `.toml` file.
2. In Python: `settings = zstackr.load_config("my_config.toml")`.
3. Run `zstackr.stack_files(params, settings)` (or `batch_stack`) with those exact settings, headlessly, as
   part of a larger automation script.

The reverse also works: build/tune a `PyStackingSettings` from Python, `zstackr.save_config(...)`, and open
the resulting file in the GUI or point the CLI's `--config` at it.

## AI models (`nn` feature)

`params.model` (fusion) / `params.align_model` (neural alignment) select a trained model by name (file
stem) from a `models/` directory next to your Python process's working directory (or next to the compiled
extension — see the root README's "Models folder" note). Each model is a `<name>.mpk` (weights) +
`<name>.json` (architecture manifest) pair, produced by the `z-stackr-train` binary — see the root README's
"Training the AI model" section.

This only has any effect in an extension built with `--features nn` (`maturin build --features nn`);
without it, `mode="ai"` / `alignment_mode="Neural"` fail at run time with a clear error rather than silently
falling back to a classical mode. `stack_arrays` does not support AI fusion at all (see "The two paths"
above) — use `stack_files`/`batch_stack`.

## GPU acceleration (`gpu` feature)

`gpu` is the other default feature baked into the published wheel (see "Feature flags" above): compute-heavy
stages of Apex/Relief/Strata fusion (plus production warp) run on the GPU via
[`wgpu`](https://wgpu.rs/)/Vulkan when a compatible GPU and driver are available. There is no separate
Python-side switch to enable it — it's simply used automatically. If no compatible GPU/driver is found at
run time, both `stack_files` and `stack_arrays` fall back to the CPU path transparently; there is no error or
warning in that case, since CPU-only is a perfectly normal, supported configuration.

Building from source with a trimmed feature set that excludes `gpu` (e.g. `maturin build --features
python,extension-module`, bypassing `pyproject.toml`'s default list) simply runs CPU-only, same as any build
predating this feature's addition.

## RAW support (`raw` feature)

The published wheel (`pip install z-stackr`) already accepts camera RAW files (`.cr2`, `.cr3`, `.nef`,
`.arw`, `.dng`, `.raf`, `.orf`, `.rw2`, `.pef`, and more — see `stacker_core::io::RAW_EXTENSIONS`) in
`stack_files`, `batch_stack`, and `load_image` — `raw` is one of the default features baked into the
published build (see "Feature flags" above). Decoding is pure-Rust (`rawler`, the dnglab decoding engine, no
C/C++ dependency). Building from source with a trimmed feature set that excludes it will instead raise
`RuntimeError` with a clear "rebuild with `--features raw`" message on any RAW input.

**CR3 is supported** — `rawler` has a native decoder for Canon's ISO-BMFF-based CR3 container, unlike the
previous `rawloader`-based implementation which could only parse the older TIFF-based CR2 format. See
`stacker_core::io`'s module docs for the full list of honest limitations (no lens corrections, no highlight
recovery, camera "as-shot" white balance only).

## Batch processing

See `batch_stack` above for the full contract. The intended headless workflow: configure once in the GUI,
save a config, then run `batch_stack(input_dir, output_dir, load_config("my_config.toml"))` over a folder of
per-subject subfolders — mirroring the CLI's `--stacks subfolders` batch mode.

## The in-RAM stacking flow (`stack_arrays`)

`stack_arrays` mirrors the GUI's in-RAM Stack pipeline step for step:

1. **Convert** every input array to the internal `PlanarImage<f32>` representation (dtype-dispatched: u8 /
   0.0-1.0, u16 / 0.0-1.0, f32 passed through), via bulk slice operations (never per-pixel Python calls).
2. **Align** sequentially — frame 0 is the reference; each later frame aligns against the *previously
   warped* frame (a rolling reference), warm-started from the previous frame's solved matrix. Honours
   `alignment_mode`. When `correct_brightness` is set, a brightness target from frame 0 is applied to every
   warped frame.
3. **Coverage + crop** — each frame's coverage mask is intersected into a running common mask; when
   `crop_to_common_area` is set, the common-coverage rectangle is resolved (with the same 25%-of-canvas
   rogue-frame guard the file pipeline uses) and every frame is cropped to it.
4. **`stack_every_nth`** subsampling of the aligned+cropped frame set.
5. **Sort/cull** — when `auto_cull` and/or `sort_by_sharpness` are set, `optimize_stack` reorders and/or
   drops frames. This is the one path in the crate where these settings actually take effect.
6. **Fuse** — `apex` (honouring `use_all_color_channels`, `grit_suppression`, `pyramid_levels`), `relief`
   (honouring `relief_engine`, `relief_estimation_radius`, `relief_smoothing_radius`, `relief_contrast_pct`,
   `relief_auto_detect`), or `strata` (honouring `strata_base_radius`, `strata_detail_focus`), selected by
   the `mode` parameter.
7. **Restretch** — when `crop_to_common_area && resize_cropped_to_original` and a crop actually shrank the
   canvas, resample the fused result back up to the original pre-crop resolution.
8. **Convert back** to the same numpy dtype the input arrays used.

The whole compute-heavy body (steps 1-8 minus the initial Python-array conversion) runs with the GIL
released.

## Limitations

- **`auto_cull` / `sort_by_sharpness` only affect `stack_arrays`.** `stack_files` (and `batch_stack`, which
  is built on it) use the tiled out-of-core pipeline, which never holds the full aligned stack in memory at
  once and therefore cannot compare frames against each other the way culling/sorting needs to — exactly the
  same limitation the CLI has always had. A startup warning is logged (visible via Rust `tracing`, not
  raised as a Python warning) if you leave these enabled with `stack_files`.
- **8-bit vs. 16-bit precision.** The internal pipeline always computes in `f32`. Round-tripping through
  `uint8` arrays quantises to 256 levels per channel on both ends (input and output); use `uint16` or
  `float32` arrays with `stack_arrays`, or a 16-bit-capable output extension (`.tiff`/`.png`) with
  `stack_files`, to preserve full precision.
- **GIL.** Both `stack_files` and `stack_arrays` release the GIL for their compute-heavy bodies, so other
  Python threads are not blocked during a long stack — but numpy array *construction/conversion* at the
  boundaries (before `.detach()`/after re-acquiring) does hold the GIL, as it must (numpy arrays are Python
  objects).
- **AI fusion is file-path-only.** `stack_arrays` does not support `mode="ai"` — use `stack_files`/
  `batch_stack` for neural fusion.

## Building wheels

```bash
cd crates/stacker-python

# Local wheel for the current platform — already includes gpu + raw (see
# "Feature flags" above), since maturin builds from the workspace-root
# pyproject.toml's feature list, not this crate's own (empty) Cargo default:
maturin build --release

# Also enable AI fusion / neural alignment (the one remaining opt-in feature):
maturin build --release --features nn

# Editable install for development (rebuilds automatically picked up by a
# re-import, no separate `pip install` step):
maturin develop --release
```

The produced wheel installs as `z-stackr` (PyPI name) but is imported as `import zstackr` — see
"Installation" above.
