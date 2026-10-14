# z-stackr-core

The shared data types and services of **z-stackr** (crate identifier `stacker_core`).

Every other crate (`z-stackr-align`, `z-stackr-algo`, `z-stackr-pipeline`, `z-stackr-nn`, `z-stackr-cli`, `z-stackr-gui`) depends on it. It defines the image type, the colour model, the settings schema, the file-backed tile store, the error type and the tracing set-up. The crate needs nightly Rust (`#![feature(portable_simd)]`).

## Components

### `image::PlanarImage`
All processing works on planar images: three contiguous, equally sized `f32` planes (`luma`, `chroma_a`, `chroma_b`, row-major with stride `width`) instead of interleaved RGB. For a photograph the planes are linear-light BT.601 `Y` in `[0, 1]` and `Cb`/`Cr` in `[-0.5, 0.5]`, computed from linear RGB. The same struct also carries focus maps, masks and Laplacian bands, for which the planes mean whatever the producer documents.

`PlanarImage::new`, `from_planes` and `with_luma` establish the invariant that every plane holds exactly `width * height` samples (the last two return an error otherwise); `is_consistent` checks it.

### `color`
Focus stacking is done in linear light, because blending gamma-encoded values sums mid-tones incorrectly.

* `srgb_to_linear` / `linear_to_srgb` are the IEC 61966-2-1 transfer functions. `LUMA_CHROMA_FORWARD` / `LUMA_CHROMA_INVERSE` and `linear_rgb_to_planar` / `planar_to_linear_rgb` are the only definition of the luma/chroma model.
* **Decoding** uses exact lookup tables: `srgb_decode_u8_table()` (256 entries) and `srgb_decode_u16_table()` (65 536 entries), so no `powf` runs per decoded sample.
* **8-bit encoding**: `encode_row_u8` turns a row of linear samples into sRGB codes with `std::simd` (a 14-bit table lookup plus a compare against the neighbouring code thresholds), so no `powf` runs per sample. The result equals `round(linear_to_srgb(clamp(v, 0, 1)) * 255)`: linear white is exactly 255, black 0, NaN 0. The vector width follows the compile-time target features (16 lanes with `avx512f`, 8 with `avx2`, otherwise 4); the output does not depend on it.
* **16-bit encoding**: `encode_row_u16` is scalar and exactly equal to `linear_to_srgb_u16` for every input: tables classify almost every sample by a search, and only a sample inside a narrow guard zone around a code boundary falls back to the transfer function (`powf`). Callers run rows in parallel. White is exactly 65 535.
* `linear_to_srgb_u8` / `linear_to_srgb_u16` are the scalar quantisers (one `powf` each) that the row encoders agree with.

### `io`: camera RAW input
RAW files are recognised by extension in every build (`is_raw_extension`, `has_raw_extension`, the list `RAW_EXTENSIONS`: CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2, PEF and others), so file pickers and directory scans agree whether or not RAW can be decoded. `decode_raw_file` needs the `raw` cargo feature and a native target (it is cfg-gated off wasm32); without either it returns `RawError::Disabled` with a message that says how to enable RAW, and `RAW_SUPPORTED` says which case a build is. With the feature on it runs `rawler` (the decoding engine of dnglab): demosaic, the camera white balance stored in the file, the file's color matrix to sRGB, the sRGB curve; the result is a 16-bit sRGB image that the loader linearises with the exact table like any 16-bit file. No lens corrections, no highlight recovery, no white-balance choice, and the EXIF orientation is not applied to RAW pixels.

### `metadata`: EXIF copy-through
`extract_exif(_from_bytes)` reads the EXIF block of a JPEG, PNG, HEIF, WebP or TIFF-based source (for a TIFF-based file the blob is rebuilt from the descriptive tags only), `inject_exif(_into_bytes)` splices it into an encoded JPEG or PNG without decoding a pixel (TIFF output is not supported and returns `UnsupportedFormat`), `set_exif_orientation` rewrites the orientation tag in place (the pipeline sets it to 1 when it rotated the pixels), and `copy_metadata` does the whole copy for files. The `*_bytes` forms work in the browser build. Backs the `image_saving.copy_metadata` setting.

### `memory`: the out-of-core tile store
`MmapTileManager` stores tiles in a directory and reads them back on demand, so a stack larger than RAM can be fused tile by tile. Each tile is one file holding its three planes; each plane is written and read as one bulk `bytemuck` byte copy in native endianness (the files are scratch for one run on one machine). Reads and writes are plain blocking `std::fs` calls. True `mmap(2)` is not used: the access pattern is write-once/read-once and `PlanarImage` owns its buffers, so a memory map would still be copied into them. (The `Mmap` prefix in the name is historical.)

* A tile is keyed by `(frame index, rectangle)`. The store does not interpret the rectangle; the CLI keys its tiles by the padded rectangle (interior plus apron, clamped to the image).
* `commit_tile` rejects a tile whose planes do not hold exactly `width * height` samples, and `fetch_tile` returns `TileNotFound` for a tile that was never committed.
* The store never deletes its directory by itself. The owner creates and removes it (`MmapTileManager::clear` removes just the files the manager wrote); the CLI does this with a guard that also runs after an error.
* `enumerate_tiles`, `extract_tile` and `paste_interior` cut an image into tiles, copy a padded region out, and splice a fused interior back. `APRON_PX` (128) is the apron of the neural mode and the fallback for callers that derive none; the apron of `Apex` depends on the pyramid depth (128, 256 or 512 px for 6, 7 or 8 levels) and is computed in `stacker_algo::apex::fuse::apron_for_levels`.

Peak memory of a run therefore depends on the stage: see `crates/stacker-pipeline/README.md` for the per-pixel figures.

### `settings`
`StackingSettings` is the user-facing option set shared by both apps: 39 keys (27 top-level, 6 in `[preprocessing]`, 6 in `[image_saving]`) with their defaults, starting with `settings_version` (`SETTINGS_VERSION` = 2). `DEFAULT_CONFIG_TOML` is the commented default file. The enums are `AlignmentModeSetting` (`translation`, `registration` (default), `affine`, `none`, `neural`) and `OptimizerSetting` (`auto`, `lucaskanade`, `neldermead`); the algorithm choice (Apex, Relief, Strata, the combinations, AI) lives in the apps.

* `StackingSettings::from_toml_str` reads a file leniently: only a TOML syntax error fails; a wrongly typed value or an unknown enum string keeps that key's default, an unknown key is ignored, and each case adds one line to the returned warnings. Enum strings are case-insensitive (`output_format` also accepts `tif` and `jpg`).
* **Legacy files.** `apply_legacy_aliases` renames the old `dmap_*` keys to `relief_*`, `use_gpu` to `compute_device`, `relief_engine` to `relief_use_multigrid` and `alignment_mode = "bounded"` to `registration` + `bounded_search = true`; the new name wins when both are present. `migrate_legacy` (also run by `sanitized()`) handles a file without `settings_version`: `affine` becomes `registration`, the old `registration` becomes `registration` + `bounded_search = true`, `translation` stays; the version is then 2. Saving always writes the current names.
* `sanitized()` clamps the numeric keys to their ranges and reports each change: `stack_every_nth` at least 1, `pyramid_levels` 2 to 12, `relief_estimation_radius` 0 to 100, `relief_smoothing_radius` 0 to 50, `relief_contrast_pct` 0.0 to 1.0, `strata_base_radius` 8 to 64, `strata_detail_focus` 1 to 5, `auto_cull_threshold_pct` 0.0 to 50.0, `tile_size` 0 or 64 to 8192, `pre_rotation` one of 0/90/180/270, `pre_resize_percent` 10 to 100, `bit_depth` 8 or 16, `jpeg_quality` 1 to 100. A `settings_version` newer than the build knows is reported.

### `error` and `telemetry`
`StackerError` is the workspace error type. `telemetry::init_tracing(filter)` installs a stdout subscriber and is idempotent (a second call returns `Ok` and keeps the first subscriber); `init_tracing_to_file` also appends to a log file and returns an error if a subscriber is already set. An invalid filter string is an error in both. `RUST_LOG`, when set and valid, overrides the filter argument.

## Usage
`z-stackr-core` is a library crate.

```toml
[dependencies]
z-stackr-core = { path = "../stacker-core" }
```
