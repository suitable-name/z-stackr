//! Per-run decode-once cache for RAW frames.
//!
//! # Why this exists
//!
//! The tiled pipeline in `lib.rs` reloads each frame at several independent
//! sites (the reference-frame load, the alignment reference reload, the
//! parallel AKAZE coarse-seed pre-pass, and the sequential per-frame
//! alignment loop), plus `align::compute_neural_alignment` loads the whole
//! stack again for neural alignment. For standard (non-RAW) formats
//! `image::open` is cheap enough that re-decoding the same file 2-3 times
//! per run isn't worth restructuring around. RAW decoding is a different
//! story: demosaicing a 45+ megapixel Bayer frame is expensive, and paying
//! that cost 2-3× per frame would make RAW support impractical.
//!
//! # What is cached, and why (post-decode, pre-preprocessing)
//!
//! Every frame-load site in `lib.rs` and `align.rs` follows the exact same
//! two-step shape: `image::open(path)` (or, now, [`FrameCache::load`]),
//! immediately followed by `stacker_core::preprocessing::preprocess_frame`
//! with the *same* `settings.preprocessing` value for the whole run (there
//! is only one `StackingSettings` per [`crate::run_pipeline`] call — no site
//! ever applies different preprocessing to the same frame within one run).
//! Since preprocessing (rotate/crop/resize, all no-ops at default settings)
//! is cheap relative to RAW demosaic, and reusing a POST-preprocessing cache
//! would require this module to know about crop/rotate/resize semantics it
//! has no business knowing about, the cache stores the frame exactly as
//! decoded — before preprocessing — and every call site still runs
//! `preprocess_frame` itself afterwards, exactly as it does today. This
//! keeps preprocessing semantics byte-for-byte identical to the pre-cache
//! code path while still eliminating every redundant RAW demosaic.
//!
//! # Bounded memory
//!
//! The pre-pass ([`FrameCache::build`]) decodes RAW frames **one at a
//! time** — never two at once — immediately serialising each to a binary
//! blob on disk before moving to the next frame, mirroring
//! `stacker_core::memory::TileManager`'s write-once/read-once file-per-
//! unit strategy (and its `bytemuck` bulk-copy style) rather than holding
//! decoded pixel data in RAM across frames.
//!
//! # Ephemeral, run-scoped cache — no staleness handling
//!
//! Blob files live under the pipeline's existing per-run `temp_dir` (the
//! same directory `TileManager` uses for tile files) and are deleted
//! along with it at the end of [`crate::run_pipeline`]. There is
//! deliberately no cache-invalidation/staleness logic (no mtime check, no
//! content hash): the cache's entire lifetime is bounded by one
//! `run_pipeline` call, the source file is never modified during a run, and
//! a fresh `temp_dir` (uniquified by PID + timestamp, see `lib.rs`) is
//! created per run, so there is no scenario in which a stale blob could ever
//! be read back.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use stacker_core::error::StackerError;

/// Magic bytes identifying a frame-cache blob file (`"ZSFB"` — z-stackr
/// frame blob), guarding against reading a truncated or foreign file as if
/// it were a valid blob.
const BLOB_MAGIC: u32 = 0x5A53_4642;
/// Blob format version. Bump if the header/payload layout ever changes.
const BLOB_VERSION: u32 = 1;
/// Fixed header size in bytes: magic (u32) + version (u32) + width (u32) + height (u32).
const HEADER_LEN: usize = 16;

/// Serialise a decoded RAW frame's 16-bit interleaved RGB pixel data to a
/// binary blob: a small fixed header (magic, version, width, height) followed
/// by the raw `u16` samples, written via a single `bytemuck` bulk copy —
/// mirroring `TileManager::commit_tile`'s style exactly.
///
/// `rgb16` must have exactly `width * height * 3` elements (one interleaved
/// RGB16 triple per pixel); this invariant is upheld by
/// [`FrameCache::build`], the sole caller.
fn write_blob(path: &Path, width: u32, height: u32, rgb16: &[u16]) -> Result<(), StackerError> {
    let mut buf = Vec::with_capacity(HEADER_LEN + rgb16.len() * 2);
    buf.extend_from_slice(&BLOB_MAGIC.to_ne_bytes());
    buf.extend_from_slice(&BLOB_VERSION.to_ne_bytes());
    buf.extend_from_slice(&width.to_ne_bytes());
    buf.extend_from_slice(&height.to_ne_bytes());
    buf.extend_from_slice(bytemuck::cast_slice::<u16, u8>(rgb16));
    std::fs::write(path, &buf)?;
    Ok(())
}

/// Deserialise a blob written by [`write_blob`] back into
/// `(width, height, rgb16)`, validating the magic/version header and the
/// payload length before the bulk `bytemuck` copy.
fn read_blob(path: &Path) -> Result<(u32, u32, Vec<u16>), StackerError> {
    let data = std::fs::read(path)?;
    if data.len() < HEADER_LEN {
        return Err(StackerError::MathError(format!(
            "frame cache blob {} is truncated (only {} bytes, need at least {HEADER_LEN})",
            path.display(),
            data.len()
        )));
    }
    let magic = u32::from_ne_bytes(data[0..4].try_into().expect("slice is 4 bytes"));
    let version = u32::from_ne_bytes(data[4..8].try_into().expect("slice is 4 bytes"));
    if magic != BLOB_MAGIC {
        return Err(StackerError::MathError(format!(
            "frame cache blob {} has bad magic (this is an ephemeral per-run cache — it should \
             never be read back across runs)",
            path.display()
        )));
    }
    if version != BLOB_VERSION {
        return Err(StackerError::MathError(format!(
            "frame cache blob {} has unsupported version {version} (expected {BLOB_VERSION})",
            path.display()
        )));
    }
    let width = u32::from_ne_bytes(data[8..12].try_into().expect("slice is 4 bytes"));
    let height = u32::from_ne_bytes(data[12..16].try_into().expect("slice is 4 bytes"));

    let n_pixels = width as usize * height as usize;
    let expected_payload = n_pixels * 3 * 2; // 3 channels x 2 bytes per u16
    let payload = &data[HEADER_LEN..];
    if payload.len() != expected_payload {
        return Err(StackerError::MathError(format!(
            "frame cache blob {} payload is {} bytes, expected {expected_payload} for {width}x{height}",
            path.display(),
            payload.len()
        )));
    }

    let mut rgb16 = vec![0u16; n_pixels * 3];
    bytemuck::cast_slice_mut::<u16, u8>(&mut rgb16).copy_from_slice(payload);
    Ok((width, height, rgb16))
}

/// A per-run, ephemeral decode-once cache for RAW frames.
///
/// Built once via [`FrameCache::build`] before the alignment pre-pass, then
/// consulted by every frame-load site via [`FrameCache::load`] for the rest
/// of the run. Standard (non-RAW) formats are never cached — they pass
/// straight through to `stacker_core::io::load_frame` at every call,
/// exactly as `image::open` always did, so this cache adds zero overhead to
/// the all-standard-formats path.
pub struct FrameCache {
    /// Path -> blob file, for every RAW frame that was successfully
    /// pre-decoded. Frames absent from this map (including every non-RAW
    /// frame) fall through to a direct [`stacker_core::io::load_frame`] call
    /// in [`FrameCache::load`].
    blobs: HashMap<PathBuf, PathBuf>,
}

impl FrameCache {
    /// Build the cache: decode every RAW path in `paths` **one at a time**
    /// (never holding two decodes in RAM at once) and persist each as a blob
    /// under `temp_dir`. Non-RAW paths are skipped entirely — they are
    /// never decoded here, keeping this pre-pass a no-op cost-wise for an
    /// all-standard-formats stack.
    ///
    /// Progress is reported through `on_progress`, called once per RAW frame
    /// (`current` is 1-based) — callers map this into their own progress
    /// model's early "load" slice; see [`crate::PipelineProgress::DecodeRaw`].
    ///
    /// A RAW frame that fails to decode is logged as a warning and skipped
    /// (left out of `blobs`); the caller's subsequent [`FrameCache::load`]
    /// call for that path will then attempt (and likely also fail) the
    /// direct decode path, surfacing the same error through the normal
    /// `?`-propagated error path instead of this pre-pass silently eating
    /// a fatal error.
    ///
    /// # Errors
    ///
    /// Returns [`StackerError`] only for I/O failures writing blob files —
    /// per-frame RAW *decode* failures are logged and skipped, not
    /// propagated (see above).
    pub fn build<F>(
        paths: &[PathBuf],
        temp_dir: &Path,
        mut on_progress: F,
    ) -> Result<Self, StackerError>
    where
        F: FnMut(usize, usize),
    {
        let raw_paths: Vec<(usize, &PathBuf)> = paths
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(stacker_core::io::is_raw_extension)
            })
            .collect();

        let total = raw_paths.len();
        let mut blobs = HashMap::with_capacity(total);

        for (i, (frame_idx, path)) in raw_paths.into_iter().enumerate() {
            on_progress(i + 1, total);

            let dyn_img = match stacker_core::io::load_frame(path) {
                Ok(img) => img,
                Err(e) => {
                    tracing::warn!(
                        frame = frame_idx,
                        path = %path.display(),
                        "RAW decode pre-pass: failed to decode frame (will retry, and fail loudly, \
                         at its actual load site): {e}"
                    );
                    continue;
                }
            };

            // Normalise to interleaved RGB16 for the blob — `load_frame`'s
            // RAW path always returns `DynamicImage::ImageRgb16` today, but
            // routing through `to_rgb16()` keeps this robust even if a future
            // `rawler` version (or a differently-shaped RAW result, e.g. a
            // monochrome camera's `ImageLuma16`) ever returns a different
            // `DynamicImage` variant.
            let rgb16_buf = dyn_img.to_rgb16();
            let (w, h) = (rgb16_buf.width(), rgb16_buf.height());
            drop(dyn_img);

            let blob_path = temp_dir.join(format!("rawcache_{frame_idx}.bin"));
            write_blob(&blob_path, w, h, rgb16_buf.as_raw())?;
            drop(rgb16_buf);

            blobs.insert(path.clone(), blob_path);
        }

        Ok(Self { blobs })
    }

    /// Load `path`'s decoded (pre-preprocessing) image: from the cached blob
    /// if [`FrameCache::build`] successfully pre-decoded it, otherwise via a
    /// direct `stacker_core::io::load_frame` call (the `image::open` path
    /// for every non-RAW frame, and the fallback/error path for a RAW frame
    /// that failed during the pre-pass).
    ///
    /// # Errors
    ///
    /// Returns [`StackerError`] if the file can't be decoded (RAW or
    /// otherwise), or if a cached blob is somehow corrupt (should not happen
    /// in practice — see the module docs on why staleness isn't a concern).
    pub fn load(&self, path: &Path) -> Result<image::DynamicImage, StackerError> {
        if let Some(blob_path) = self.blobs.get(path) {
            let (w, h, rgb16) = read_blob(blob_path)?;
            let buf = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(w, h, rgb16)
                .ok_or_else(|| {
                    StackerError::MathError(format!(
                        "cached blob for {} had a pixel-count/dimension mismatch",
                        path.display()
                    ))
                })?;
            return Ok(image::DynamicImage::ImageRgb16(buf));
        }
        Ok(stacker_core::io::load_frame(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_round_trip_tiny_synthetic_frame() {
        let dir = std::env::temp_dir().join(format!(
            "z_stackr_cache_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("create temp test dir");
        let path = dir.join("frame.bin");

        // 2x2 synthetic RGB16 frame with distinct per-channel values so a
        // transposition/stride bug would be caught.
        let width = 2u32;
        let height = 2u32;
        let rgb16: Vec<u16> = vec![
            100, 200, 300, // pixel (0,0)
            400, 500, 600, // pixel (1,0)
            700, 800, 900, // pixel (0,1)
            1000, 1100, 1200, // pixel (1,1)
        ];

        write_blob(&path, width, height, &rgb16).expect("write blob");
        let (rw, rh, rdata) = read_blob(&path).expect("read blob back");

        assert_eq!(rw, width);
        assert_eq!(rh, height);
        assert_eq!(rdata, rgb16);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_read_blob_rejects_bad_magic() {
        let dir = std::env::temp_dir().join(format!(
            "z_stackr_cache_test_badmagic_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp test dir");
        let path = dir.join("bad.bin");
        std::fs::write(&path, [0u8; 20]).expect("write garbage");

        let err = read_blob(&path).unwrap_err();
        assert!(matches!(err, StackerError::MathError(_)));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_read_blob_rejects_truncated_file() {
        let dir =
            std::env::temp_dir().join(format!("z_stackr_cache_test_trunc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp test dir");
        let path = dir.join("short.bin");
        std::fs::write(&path, [0u8; 4]).expect("write too-short file");

        let err = read_blob(&path).unwrap_err();
        assert!(matches!(err, StackerError::MathError(_)));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_frame_cache_build_is_empty_for_no_raw_paths() {
        let dir = std::env::temp_dir().join(format!(
            "z_stackr_cache_test_norawpaths_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp test dir");

        let paths = vec![PathBuf::from("a.png"), PathBuf::from("b.jpg")];
        let mut calls = 0usize;
        let cache = FrameCache::build(&paths, &dir, |_, _| calls += 1).expect("build cache");
        assert_eq!(
            calls, 0,
            "no RAW paths means the progress callback never fires"
        );
        assert!(cache.blobs.is_empty());

        let _ = std::fs::remove_dir(&dir);
    }
}
