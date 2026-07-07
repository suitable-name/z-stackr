//! Shared EXIF-metadata copy-through: read the raw EXIF blob from a source
//! frame and re-inject it into a fused output file, without decoding or
//! re-encoding pixel data.
//!
//! This backs the `image_saving.copy_metadata` setting and is used
//! identically by `z-stackr-cli` and `z-stackr-gui` — the CLI/GUI-parity
//! principle applied throughout this crate: any metadata-handling decision
//! lives here once, not duplicated per app.
//!
//! # How it works
//!
//! [`extract_exif`] uses `kamadak-exif`'s container sniffing to pull the raw
//! TIFF-structured EXIF payload out of the source frame (JPEG, TIFF, PNG,
//! HEIF or WebP — whatever the source happens to be). [`inject_exif`] then
//! uses `img-parts` to splice that exact byte payload into the *output*
//! container at the byte level, with no pixel decode/re-encode involved.
//!
//! # RAW sources (`raw` feature)
//!
//! [`copy_metadata`]/[`extract_exif`] are always called with the *original*
//! source file path, never the decoded/cached blob — so for a RAW input
//! frame, `kamadak-exif` is handed the actual `.cr2`/`.nef`/`.dng`/etc. file.
//! Since CR2, NEF, and DNG are themselves TIFF-based containers,
//! `kamadak-exif`'s TIFF reader can often parse their EXIF IFD directly with
//! no code changes here. Formats it can't parse (or a container it doesn't
//! recognise at all) fall through [`extract_exif`]'s existing `None` return
//! path exactly like any other unparseable source — a graceful no-op, not an
//! error — so RAW sources need no special-casing in this module.
//!
//! # Supported output containers
//!
//! Re-injection only supports JPEG and PNG outputs, because `img-parts` only
//! understands those two (plus WebP, which this pipeline never writes).
//! **TIFF output is not supported**: neither `img-parts` nor the `image`
//! crate's baseline TIFF encoder expose a hook to write a custom EXIF IFD
//! into a freshly encoded TIFF. [`copy_metadata`] returns
//! [`MetadataError::UnsupportedFormat`] for `.tif`/`.tiff` outputs (and any
//! other unrecognised extension) rather than silently doing nothing, so
//! callers can log a clear, honest warning instead of overclaiming.

use std::{
    fs,
    io::BufReader,
    path::{Path, PathBuf},
};

use img_parts::{DynImage, ImageEXIF};

/// Errors that can occur while copying EXIF metadata into an output file.
///
/// These are all non-fatal from the pipeline's point of view: by the time
/// `copy_metadata` runs, the fused image has already been written
/// successfully. Callers should log and continue rather than fail the run.
#[derive(Debug, thiserror::Error)]
pub enum MetadataError {
    /// The output file could not be read back or written.
    #[error("I/O error while injecting metadata into {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// The output file's bytes could not be parsed as a supported container.
    #[error("failed to parse {path} as a JPEG/PNG container for metadata injection: {message}")]
    Parse { path: PathBuf, message: String },
    /// The output format doesn't support EXIF re-injection (e.g. TIFF).
    #[error(
        "copy_metadata is enabled, but output format \"{0}\" does not support EXIF injection \
         (only JPEG and PNG outputs are supported) — metadata was not copied"
    )]
    UnsupportedFormat(String),
}

/// Reads the raw EXIF blob (bare TIFF-structured bytes, no container
/// prefix) from `source_path`, if present.
///
/// Uses `kamadak-exif`'s container sniffing, so this works regardless of
/// whether the source frame is a JPEG, TIFF, PNG, HEIF, or WebP — we only
/// need the exact original byte payload to re-inject later, not any
/// Rust-level decoded fields.
///
/// Returns `None` if the file has no EXIF data, can't be opened, or can't
/// be parsed. A source frame with no/corrupt EXIF is not itself an error
/// condition for the caller — metadata copying just becomes a no-op.
#[must_use]
pub fn extract_exif(source_path: &Path) -> Option<Vec<u8>> {
    let file = fs::File::open(source_path).ok()?;
    let exif = exif::Reader::new()
        .read_from_container(&mut BufReader::new(file))
        .ok()?;
    Some(exif.buf().to_vec())
}

/// Injects `exif_blob` (bare TIFF-structured bytes, as returned by
/// [`extract_exif`]) into the JPEG/PNG file at `output_path`, in place.
///
/// This edits the container at the byte level — the pixel data is never
/// decoded or re-encoded. Any pre-existing EXIF in the output is replaced.
///
/// # Errors
///
/// Returns [`MetadataError::UnsupportedFormat`] if `output_path`'s
/// extension isn't `.jpg`/`.jpeg`/`.png`, [`MetadataError::Parse`] if the
/// file's bytes can't be parsed as that container format, and
/// [`MetadataError::Io`] on read/write failure.
pub fn inject_exif(output_path: &Path, exif_blob: &[u8]) -> Result<(), MetadataError> {
    let ext = output_path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(ext.as_deref(), Some("jpg" | "jpeg" | "png")) {
        return Err(MetadataError::UnsupportedFormat(
            ext.unwrap_or_else(|| "<no extension>".to_owned()),
        ));
    }

    let bytes = fs::read(output_path).map_err(|source| MetadataError::Io {
        path: output_path.to_path_buf(),
        source,
    })?;

    let mut image = DynImage::from_bytes(bytes.into())
        .map_err(|source| MetadataError::Parse {
            path: output_path.to_path_buf(),
            message: source.to_string(),
        })?
        .ok_or_else(|| MetadataError::UnsupportedFormat("<unrecognised container>".to_owned()))?;

    image.set_exif(Some(exif_blob.to_vec().into()));

    let file = fs::File::create(output_path).map_err(|source| MetadataError::Io {
        path: output_path.to_path_buf(),
        source,
    })?;
    image
        .encoder()
        .write_to(file)
        .map_err(|source| MetadataError::Io {
            path: output_path.to_path_buf(),
            source,
        })?;
    Ok(())
}

/// Convenience wrapper combining [`extract_exif`] + [`inject_exif`]: copies
/// EXIF metadata from `source_path` (the first frame of the input stack)
/// into `output_path` (the fused result).
///
/// Returns:
/// - `Ok(true)` — metadata was found in the source and copied.
/// - `Ok(false)` — the source had no EXIF data; nothing to copy (not an
///   error).
/// - `Err(_)` — the source had EXIF data, but the output format doesn't
///   support injection, or the output file couldn't be read/written.
///
/// Callers (CLI/GUI) should treat `Err` as non-fatal: log a warning and
/// continue, since the fused image itself was already written successfully
/// by the time metadata copying runs.
///
/// # Errors
///
/// See [`inject_exif`].
pub fn copy_metadata(source_path: &Path, output_path: &Path) -> Result<bool, MetadataError> {
    let Some(exif_blob) = extract_exif(source_path) else {
        return Ok(false);
    };
    inject_exif(output_path, &exif_blob)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufWriter;

    /// A minimal, hand-built, valid little-endian TIFF/EXIF IFD containing a
    /// single `ImageDescription` ASCII field ("hi"). Used as a deterministic
    /// EXIF payload for round-trip tests, without depending on any binary
    /// fixture files.
    fn synthetic_exif_blob() -> Vec<u8> {
        vec![
            // TIFF header: "II" (little-endian) + magic 42 + offset to IFD0 (8)
            0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // IFD0: 1 entry
            0x01, 0x00, // Entry: tag=0x010E (ImageDescription), type=2 (ASCII), count=3
            0x0E, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00,
            // value (inlined, ASCII count <= 4): "hi\0" + pad
            b'h', b'i', 0x00, 0x00, // next IFD offset = 0 (none)
            0x00, 0x00, 0x00, 0x00,
        ]
    }

    fn write_tiny_jpeg(path: &Path) {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([120u8, 60, 200]));
        let file = fs::File::create(path).expect("create tmp jpeg");
        let mut writer = BufWriter::new(file);
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, 90);
        enc.encode(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .expect("encode tmp jpeg");
    }

    fn write_tiny_png(path: &Path) {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([10u8, 200, 30]));
        img.save(path).expect("save tmp png");
    }

    fn read_back_description(path: &Path) -> String {
        let file = fs::File::open(path).expect("reopen output");
        let exif = exif::Reader::new()
            .read_from_container(&mut BufReader::new(file))
            .expect("parse injected exif");
        let field = exif
            .get_field(exif::Tag::ImageDescription, exif::In::PRIMARY)
            .expect("ImageDescription field present after injection");
        field.display_value().to_string()
    }

    #[test]
    fn test_inject_exif_jpeg_round_trip() {
        let path = std::env::temp_dir().join(format!(
            "z_stackr_metadata_test_{}_jpeg.jpg",
            std::process::id()
        ));
        write_tiny_jpeg(&path);

        inject_exif(&path, &synthetic_exif_blob()).expect("inject exif into jpeg");
        assert_eq!(read_back_description(&path), "\"hi\"");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_inject_exif_png_round_trip() {
        let path = std::env::temp_dir().join(format!(
            "z_stackr_metadata_test_{}_png.png",
            std::process::id()
        ));
        write_tiny_png(&path);

        inject_exif(&path, &synthetic_exif_blob()).expect("inject exif into png");
        assert_eq!(read_back_description(&path), "\"hi\"");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_inject_exif_rejects_unsupported_tiff_extension() {
        let path = std::env::temp_dir().join(format!(
            "z_stackr_metadata_test_{}_unsupported.tif",
            std::process::id()
        ));
        // File need not even exist: the extension check runs first.
        let err = inject_exif(&path, &synthetic_exif_blob()).unwrap_err();
        assert!(matches!(err, MetadataError::UnsupportedFormat(ref ext) if ext == "tif"));
    }

    #[test]
    fn test_copy_metadata_no_source_exif_is_not_an_error() {
        // A source path that doesn't exist / has no parseable EXIF should
        // make `copy_metadata` a clean no-op, not a hard failure.
        let missing_source = std::env::temp_dir().join(format!(
            "z_stackr_metadata_test_{}_missing_source.jpg",
            std::process::id()
        ));
        let out_path = std::env::temp_dir().join(format!(
            "z_stackr_metadata_test_{}_copy_out.png",
            std::process::id()
        ));
        write_tiny_png(&out_path);

        let copied = copy_metadata(&missing_source, &out_path).expect("no-op, not an error");
        assert!(!copied);

        let _ = fs::remove_file(&out_path);
    }

    #[test]
    fn test_extract_exif_round_trips_through_inject() {
        let source_path = std::env::temp_dir().join(format!(
            "z_stackr_metadata_test_{}_source.jpg",
            std::process::id()
        ));
        write_tiny_jpeg(&source_path);
        inject_exif(&source_path, &synthetic_exif_blob()).expect("seed source exif");

        let extracted = extract_exif(&source_path).expect("extract seeded exif");

        let out_path = std::env::temp_dir().join(format!(
            "z_stackr_metadata_test_{}_dest.png",
            std::process::id()
        ));
        write_tiny_png(&out_path);
        inject_exif(&out_path, &extracted).expect("inject extracted exif");
        assert_eq!(read_back_description(&out_path), "\"hi\"");

        let _ = fs::remove_file(&source_path);
        let _ = fs::remove_file(&out_path);
    }
}
