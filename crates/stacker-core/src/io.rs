//! Shared frame-loading abstraction.
//!
//! Routes standard image formats through the `image` crate and (optionally)
//! camera RAW formats through `rawler`, so `z-stackr-cli`, `z-stackr-gui`,
//! and `z-stackr-pipeline` all decode frames identically instead of each
//! re-implementing extension sniffing and RAW dispatch.
//!
//! # RAW support (`raw` feature)
//!
//! Without the `raw` feature this crate has zero RAW dependencies: RAW
//! extensions are still *recognised* by [`is_raw_extension`] (so callers can
//! filter file pickers/directory scans consistently either way), but
//! [`load_frame`] returns a clear [`LoadError::RawSupportDisabled`] instead
//! of attempting to decode.
//!
//! With the `raw` feature, [`load_frame`] decodes RAW frames via `rawler`
//! (the actively-maintained dnglab decoding engine): `rawler::decode_file`
//! parses the container and Bayer/X-Trans/etc. sensor data, then
//! `rawler::imgop::develop::RawDevelop`'s default pipeline (rescale to
//! `[0, 1]`, demosaic, active-area crop, camera white balance, colour-matrix
//! calibration to sRGB, default-crop, sRGB gamma) produces a demosaiced,
//! white-balanced, gamma-encoded sRGB image, returned as
//! `DynamicImage::ImageRgb16` — the same shape the old `rawloader` +
//! `imagepipe` pipeline produced.
//!
//! ## Honest limitations
//!
//! * **No lens corrections** (distortion, vignetting, chromatic aberration)
//!   are applied — `RawDevelop`'s default pipeline only does demosaic +
//!   camera white balance + colour-matrix calibration + gamma. Serious
//!   colour-critical work may prefer converting RAW files to TIFF externally
//!   (e.g. with `darktable` or Adobe Camera Raw) first, applying whatever
//!   corrections/colour grading is needed there, and feeding the TIFFs to
//!   z-stackr instead.
//! * **No highlight recovery / advanced white balance tuning** — the camera
//!   white balance embedded in the RAW file is used as-is; there is no
//!   "as shot" vs. "auto" vs. custom white-balance picker.
//! * **EXIF orientation is not applied to RAW pixel data** — exactly like
//!   every other format `load_frame` handles: `rawler::decode_file` records
//!   the file's `Orientation` tag but `RawDevelop`'s pipeline never rotates
//!   pixels by it, matching the `image` crate's own `image::open` (which
//!   likewise never auto-rotates). This means `ignore_exif_orientation`
//!   continues to behave as a documented no-op for every format, RAW
//!   included — see `preprocessing::preprocess_frame`'s module docs.
//!
//! Supported RAW extensions (from `rawler`'s own decoder registry): Canon
//! `.cr2` and `.cr3`, Nikon `.nef`, Sony `.arw`, Adobe `.dng`, Fujifilm
//! `.raf`, Olympus `.orf`, Panasonic `.rw2`, Pentax `.pef`, and a handful of
//! other manufacturer formats `rawler` recognises regardless of extension —
//! see [`RAW_EXTENSIONS`] for the exact list this crate filters directory
//! scans and file dialogs by.

use std::path::Path;

use image::DynamicImage;

/// Canonical list of recognised RAW file extensions (lower-case, no dot).
///
/// This is the single source of truth for "is this filename a RAW frame" —
/// [`is_raw_extension`], `stacker_pipeline::load::collect_image_paths`, and
/// the GUI's file-open dialog filter all read from this list so they can
/// never drift apart. Sourced from `rawler`'s supported decoder registry
/// (container formats it can parse); this is an extension allow-list, not a
/// guarantee that every file with one of these extensions will decode
/// (`rawler` may still reject an unsupported camera model's dialect).
///
/// **Includes `cr3`**: unlike the previous `rawloader`-based decoder, which
/// could only parse Canon's older TIFF-based CR2 container, `rawler` has a
/// native CR3 (ISO-BMFF-based) decoder — see the module docs.
pub const RAW_EXTENSIONS: &[&str] = &[
    "cr2", "cr3", "nef", "arw", "dng", "raf", "orf", "rw2", "pef", "srw", "3fr", "erf", "kdc",
    "mrw", "nrw", "raw", "rwl", "sr2", "srf", "x3f",
];

/// Returns `true` if `ext` (case-insensitive, no leading dot) is a
/// recognised RAW file extension.
///
/// Uses [`RAW_EXTENSIONS`] as the single source of truth so directory scans
/// and file-dialog filters can never drift apart from what [`load_frame`]
/// actually knows how to route.
#[must_use]
pub fn is_raw_extension(ext: &str) -> bool {
    let lower = ext.to_ascii_lowercase();
    RAW_EXTENSIONS.contains(&lower.as_str())
}

/// Errors from [`load_frame`].
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// A standard (non-RAW) format failed to decode via the `image` crate.
    #[error("failed to decode {path}: {source}")]
    Image {
        path: std::path::PathBuf,
        #[source]
        source: image::ImageError,
    },
    /// The file's extension is a recognised RAW format, but this binary was
    /// built without the `raw` feature.
    #[error(
        "'{path}' looks like a RAW file (.{ext}), but this build was compiled without RAW \
         support — rebuild with `--features raw`, or convert the file to TIFF/DNG first"
    )]
    RawSupportDisabled {
        path: std::path::PathBuf,
        ext: String,
    },
    /// RAW decoding itself failed (feature `raw` only).
    #[cfg(feature = "raw")]
    #[error("failed to decode RAW file {path}: {message}")]
    RawDecode {
        path: std::path::PathBuf,
        message: String,
    },
}

/// Load an image frame from `path`, routing by file extension.
///
/// - Standard formats (anything not in [`RAW_EXTENSIONS`]) are decoded via
///   `image::open` directly, with zero added overhead or dependencies.
/// - RAW formats are decoded via `rawler` when this crate is built with the
///   `raw` feature (16-bit RGB output, camera white balance applied, default
///   demosaic — see the module docs for the exact pipeline and its honest
///   limitations); without the feature this returns
///   [`LoadError::RawSupportDisabled`] with a clear remediation message.
///
/// # Errors
///
/// Returns [`LoadError`] if the file cannot be read or decoded, or (in
/// non-`raw` builds) if `path` has a RAW extension.
pub fn load_frame(path: &Path) -> Result<DynamicImage, LoadError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    if is_raw_extension(&ext) {
        return load_raw_frame(path, &ext);
    }

    image::open(path).map_err(|source| LoadError::Image {
        path: path.to_path_buf(),
        source,
    })
}

/// RAW-specific decode path, split out so the `#[cfg(feature = "raw")]`
/// gate lives in exactly one place rather than smeared across
/// [`load_frame`].
#[cfg(feature = "raw")]
fn load_raw_frame(path: &Path, _ext: &str) -> Result<DynamicImage, LoadError> {
    // `rawler::decode_file` parses the container (CR2/CR3/NEF/ARW/DNG/…)
    // and returns the raw sensor data plus all metadata needed to develop it
    // (white balance coefficients, colour matrix, active/crop areas, the
    // EXIF `Orientation` tag, etc.) — this is the `rawloader::decode_file`
    // equivalent in the old pipeline.
    let rawimage = rawler::decode_file(path).map_err(|source| LoadError::RawDecode {
        path: path.to_path_buf(),
        message: source.to_string(),
    })?;

    // `RawDevelop::default()`'s step list (Rescale, Demosaic,
    // CropActiveArea, WhiteBalance, Calibrate, CropDefault, SRgb) is
    // `rawler`'s equivalent of `imagepipe::Pipeline`'s default op chain:
    // rescale black/white levels to `[0, 1]`, demosaic the CFA pattern,
    // crop to the sensor's active area, apply the camera's embedded white
    // balance, calibrate camera RGB to sRGB via the file's colour matrix,
    // crop to the recommended output area, then apply the sRGB gamma
    // curve. Like the old pipeline, this never rotates pixels according to
    // the file's `Orientation` tag (see the module docs) — orientation
    // handling stays a no-op regardless of `ignore_exif_orientation`,
    // exactly matching every other format's behaviour.
    let intermediate = rawler::imgop::develop::RawDevelop::default()
        .develop_intermediate(&rawimage)
        .map_err(|source| LoadError::RawDecode {
            path: path.to_path_buf(),
            message: source.to_string(),
        })?;

    // 16-bit sRGB output. `None` cache: each RAW frame in this pipeline is
    // decoded exactly once per run (see `z-stackr-pipeline`'s decode-once
    // cache), so there is nothing to usefully cache across calls here.
    // `to_dynamic_image` returns `ImageRgb16` for the normal three-colour
    // case (the vast majority of cameras, after `Calibrate` folds any
    // four-colour CFA down to three channels); a genuinely monochrome
    // camera instead yields `ImageLuma16`, which downstream code already
    // handles via `DynamicImage::to_rgb16()`.
    intermediate
        .to_dynamic_image()
        .ok_or_else(|| LoadError::RawDecode {
            path: path.to_path_buf(),
            message: "decoded RAW pixel buffer size did not match its reported dimensions"
                .to_owned(),
        })
}

#[cfg(not(feature = "raw"))]
fn load_raw_frame(path: &Path, ext: &str) -> Result<DynamicImage, LoadError> {
    Err(LoadError::RawSupportDisabled {
        path: path.to_path_buf(),
        ext: ext.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_raw_extension_recognises_known_formats() {
        for ext in ["cr2", "NEF", "Arw", "dng", "raf", "orf", "rw2", "pef"] {
            assert!(
                is_raw_extension(ext),
                "expected '{ext}' to be recognised as RAW"
            );
        }
    }

    #[test]
    fn test_is_raw_extension_recognises_cr3() {
        // CR3 is now supported — `rawler` has a native ISO-BMFF decoder for
        // Canon's newer container, unlike the old `rawloader`-based path
        // which could only parse the TIFF-based CR2 format. See the module
        // docs.
        assert!(is_raw_extension("cr3"));
        assert!(is_raw_extension("CR3"));
    }

    #[test]
    fn test_is_raw_extension_rejects_standard_formats() {
        for ext in ["png", "jpg", "jpeg", "tif", "tiff", "PNG", "", "txt"] {
            assert!(
                !is_raw_extension(ext),
                "expected '{ext}' to NOT be recognised as RAW"
            );
        }
    }

    #[test]
    fn test_load_frame_raw_without_feature_gives_clear_error() {
        // This test only exercises the reachable branch in a default
        // (non-`raw`) build: a RAW extension without the feature must fail
        // with a clear, actionable message rather than a generic I/O error
        // or (worse) silently misinterpreting the RAW bytes as some other
        // format. The file need not even exist — the extension check runs
        // first, exactly like `metadata::inject_exif`'s unsupported-format
        // check.
        #[cfg(not(feature = "raw"))]
        {
            let path = Path::new("nonexistent_test_frame.nef");
            let err = load_frame(path).unwrap_err();
            assert!(matches!(err, LoadError::RawSupportDisabled { .. }));
            let msg = err.to_string();
            assert!(msg.contains("--features raw"));
            assert!(msg.contains("nef") || msg.contains(".nef"));
        }
    }

    #[test]
    fn test_load_frame_standard_format_missing_file_is_image_error() {
        let path = Path::new("this_file_really_does_not_exist_z_stackr_test.png");
        let err = load_frame(path).unwrap_err();
        assert!(matches!(err, LoadError::Image { .. }));
    }
}
