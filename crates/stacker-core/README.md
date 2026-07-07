# z-stackr-core

The fundamental data structures and memory management layer for **z-stackr**.

`z-stackr-core` (crate identifier `stacker_core`) provides the foundational primitives that all other crates (`z-stackr-align`, `z-stackr-algo`, `z-stackr-nn`, `z-stackr-cli`) depend on. It defines the core image representations, handles precision color space conversions, and implements the project's powerful out-of-core memory management architecture.

## Core Components

### `PlanarImage`
Instead of using interlaced RGB bytes (e.g., `RGBARGBARGBA`), all processing in z-stackr operates on planar representations where each channel (Luma, Chroma A, Chroma B) is a distinct, contiguous `f32` slice.
This dramatically increases CPU cache coherency during heavy math operations like convolution and pyramid generation, and allows the fusion algorithms to easily operate on a single channel (Luma) where applicable.

### Colour model & the `color::` module
The pipeline-wide contract is **gamma-space**: loaders normalise raw sRGB-encoded samples to `[0, 1]` with **no transfer function applied**, store them as BT.601-style planar YCbCr, and all alignment metrics and fusion math operate directly on those gamma-encoded values. This is a deliberate reference-fidelity choice — it reproduces the established behaviour of classic stacking tools bit-for-bit rather than re-deriving it in linear light. (The one boundary that genuinely converts is the neural mode: `z-stackr-nn`'s bridge linearises at the model boundary because its network is trained on linearised data.)

The `color::` module provides exact sRGB⇄linear transfer functions plus lookup-table and `std::simd` accelerated encode/decode helpers (no per-pixel `powf` on hot paths). These are **utility routines for library consumers** — the shipped gamma-space pipeline itself does not route pixels through them.

### Shared Frame Preprocessing (`preprocessing::preprocess_frame`)
Both apps call this single shared function to apply pre-stacking transforms — rotation (0/90/180/270°), an optional crop (`"w,h"` centred or `"x,y,w,h"` explicit), and an optional percentage resize — in that order, driven by the shared `settings::PreprocessingSettings` struct. At default settings every step is a no-op, so the pipeline output is byte-identical to not preprocessing at all. Living here (rather than duplicated per app) is deliberate: this is exactly the kind of decision that must never be re-implemented separately in the CLI and GUI.

### Shared EXIF Metadata Copy-Through (`metadata::copy_metadata`)
Both apps call this to implement the `image_saving.copy_metadata` setting: [`extract_exif`] reads the raw TIFF-structured EXIF blob from the first source frame via `kamadak-exif` (container-format-agnostic — JPEG/TIFF/PNG/HEIF/WebP sources all work), and [`inject_exif`] splices that exact byte payload into the *output* file via `img-parts`, at the container level, with no pixel decode/re-encode. Only JPEG and PNG outputs are supported — TIFF output is not, since neither dependency exposes a hook to write EXIF into a freshly encoded baseline TIFF, and callers get an explicit `MetadataError::UnsupportedFormat` rather than a silent no-op.

### Frame Loading & Optional RAW Support (`io::load_frame`)
`io::load_frame` is the single shared entry point every app/crate uses to decode a frame from disk: standard formats (`.png`/`.jpg`/`.tif`/etc.) route straight through `image::open` — identical to, and with zero overhead over, calling `image::open` directly. Camera RAW files (`io::is_raw_extension`, backed by the canonical `io::RAW_EXTENSIONS` list — CR2/CR3/NEF/ARW/DNG/RAF/ORF/RW2/PEF and more) are decoded via the optional `raw` feature (pure-Rust `rawler`, the dnglab decoding engine: container/sensor decode plus its own default demosaic/white-balance/gamma develop pipeline to 16-bit RGB) — without that feature, a RAW extension returns a clear "rebuild with `--features raw`" error instead of a confusing decode failure. **CR3 is supported** (`rawler` has a native ISO-BMFF decoder for Canon's newer container, unlike the previous `rawloader`-based implementation). No lens corrections are applied to RAW frames; serious colour-critical work may prefer an external TIFF conversion. This module owns zero new dependencies unless `raw` is enabled, preserving the workspace's pure-Rust (no C/C++) build chain and its PGO/BOLT release profiles either way.

### Out-of-Core Processing (`memory::TileManager`)
Memory is the largest constraint in focus stacking. Stacking 100 images of 50 Megapixels in 32-bit float requires dozens of gigabytes of RAM.
`TileManager` streams the work tile-by-tile: aligned frames are written to temporary tile files on disk (plain blocking `std::fs`, each planar channel (de)serialised as one bulk `bytemuck` byte copy in native endianness), and fusion reads back only the tiles it needs. True `mmap(2)` is intentionally avoided — the access pattern is write-once/read-once and `PlanarImage` owns its buffers, so a map would just be copied into them anyway.
The application can "check out" a tile, operate on it entirely in RAM, and write it back out. Peak memory is therefore bounded by the tile size and frame count rather than the full image resolution, so deep stacks of very large images stay within a predictable footprint.

## Usage
`stacker-core` is a library crate and cannot be run independently.

```toml
[dependencies]
z-stackr-core = "1.0"
```
