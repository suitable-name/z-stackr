//! EXIF copy-through from a source frame into an encoded output file.
//!
//! The EXIF block of the source is read and spliced into the output, without
//! decoding or re-encoding any pixel.
//!
//! This backs the `image_saving.copy_metadata` setting. Everything works on
//! byte slices (`*_bytes` functions, usable in the browser build) with thin
//! file wrappers on top.
//!
//! # How it works
//!
//! * [`extract_exif_from_bytes`] / [`extract_exif`] find the EXIF block of a JPEG,
//!   PNG, HEIF, WebP or TIFF-based file with `kamadak-exif` and return it as a
//!   bare TIFF-structured blob. For a JPEG, PNG, HEIF or WebP source this is the
//!   block as stored. A TIFF-based source (TIFF, DNG, NEF, CR2, ...) keeps its
//!   EXIF inside the image file itself, so the blob is **rebuilt**: only the
//!   descriptive tags (camera, exposure, date, GPS, ...) are written, never the
//!   structural ones (strip offsets, sizes, compression, MakerNote, ...), and the
//!   blob stays a few kilobytes.
//! * [`inject_exif_into_bytes`] / [`inject_exif`] splice a blob into an encoded
//!   JPEG or PNG with `img-parts`, replacing any EXIF block already there.
//!
//! # Supported outputs
//!
//! JPEG and PNG. **TIFF output is not supported**: neither `img-parts` nor the
//! `image` crate's TIFF encoder can write an EXIF block into a TIFF, so
//! [`inject_exif_into_bytes`] returns [`MetadataError::UnsupportedFormat`] and
//! callers log a warning.
//!
//! # Orientation
//!
//! The pipeline rotates pixels on load when the EXIF orientation is applied, so
//! the copied block must say "upright": [`set_exif_orientation`] rewrites tag
//! `0x0112` in place.

use std::{
    fmt, fs,
    io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom},
    path::Path,
};

use img_parts::{DynImage, ImageEXIF};

/// The largest EXIF blob a JPEG can carry: an APP1 segment holds at most
/// 65 533 bytes after its length field, 6 of them the `Exif\0\0` header.
pub const JPEG_EXIF_MAX_BYTES: usize = 65_527;

/// Why EXIF could not be copied into an output.
///
/// None of these is fatal for a run: the fused image is already encoded when
/// the metadata is copied, so callers log the error and carry on.
#[derive(Debug)]
pub enum MetadataError {
    /// The output file could not be read or written.
    Io(std::io::Error),
    /// The output bytes are not a well-formed container of the supported kind.
    Parse(String),
    /// The output container cannot carry EXIF (TIFF, or an unknown format).
    UnsupportedFormat(String),
    /// The blob does not fit the output container.
    TooLarge {
        /// Size of the blob.
        bytes: usize,
        /// The most the container takes.
        limit: usize,
    },
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error while copying metadata: {e}"),
            Self::Parse(message) => write!(
                f,
                "cannot parse the output as a JPEG or PNG container for metadata copying: \
                 {message}"
            ),
            Self::UnsupportedFormat(kind) => write!(
                f,
                "copy_metadata is on, but the output format ({kind}) cannot carry EXIF; only \
                 JPEG and PNG outputs can, so the metadata was not copied"
            ),
            Self::TooLarge { bytes, limit } => write!(
                f,
                "the EXIF block ({bytes} bytes) is larger than the output container takes \
                 ({limit} bytes); the metadata was not copied"
            ),
        }
    }
}

impl std::error::Error for MetadataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for MetadataError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Reading ───────────────────────────────────────────────────────────────────

/// `true` when `head` starts with a TIFF byte-order mark and the magic 42.
fn is_tiff_header(head: &[u8]) -> bool {
    matches!(
        head.get(..4),
        Some([b'I', b'I', 42, 0] | [b'M', b'M', 0, 42])
    )
}

/// The EXIF blob of an encoded image held in memory, or `None` when there is
/// none (or it cannot be parsed). See the module documentation for what the
/// blob holds for a TIFF-based source.
#[must_use]
pub fn extract_exif_from_bytes(bytes: &[u8]) -> Option<Vec<u8>> {
    read_blob(&mut Cursor::new(bytes), is_tiff_header(bytes))
}

/// The EXIF blob of the file at `path`, or `None` when the file cannot be read
/// or holds no EXIF. A frame without metadata is not an error: copying simply
/// does nothing.
#[must_use]
pub fn extract_exif(path: &Path) -> Option<Vec<u8>> {
    let mut reader = BufReader::new(fs::File::open(path).ok()?);
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic).ok()?;
    reader.seek(SeekFrom::Start(0)).ok()?;
    read_blob(&mut reader, is_tiff_header(&magic))
}

/// Parse the container behind `reader` and return its EXIF blob.
fn read_blob<R: BufRead + Seek>(reader: &mut R, tiff_source: bool) -> Option<Vec<u8>> {
    let exif = exif::Reader::new().read_from_container(reader).ok()?;
    if tiff_source {
        // `buf()` would be the whole image file: rebuild a compact block.
        rebuild_compact(&exif)
    } else {
        Some(exif.buf().to_vec())
    }
}

/// Write the descriptive fields of the primary image of a TIFF-based source
/// into a fresh, small EXIF block of the same byte order.
fn rebuild_compact(exif: &exif::Exif) -> Option<Vec<u8>> {
    let fields: Vec<&exif::Field> = exif
        .fields()
        .filter(|f| f.ifd_num == exif::In::PRIMARY && keep_field(f))
        .collect();
    if fields.is_empty() {
        return None;
    }
    let mut writer = exif::experimental::Writer::new();
    for field in fields {
        writer.push_field(field);
    }
    let mut out = Cursor::new(Vec::new());
    writer.write(&mut out, exif.little_endian()).ok()?;
    Some(out.into_inner())
}

/// Tags of the TIFF (IFD0) context worth copying: they describe the camera and
/// the shot, not the image data of the source file.
const KEPT_TIFF_TAGS: &[u16] = &[
    0x010E, // ImageDescription
    0x010F, // Make
    0x0110, // Model
    0x0112, // Orientation
    0x011A, // XResolution
    0x011B, // YResolution
    0x0128, // ResolutionUnit
    0x0131, // Software
    0x0132, // DateTime
    0x013B, // Artist
    0x013C, // HostComputer
    0x8298, // Copyright
];

/// Tags of the Exif context that are dropped: the `MakerNote` (its inner offsets
/// do not survive a rewrite), the pixel dimensions (stale after any crop or
/// resize) and the pointer to the Interoperability IFD.
const DROPPED_EXIF_TAGS: &[u16] = &[0x927C, 0xA002, 0xA003, 0xA005];

/// Whether `field` goes into a rebuilt block.
fn keep_field(field: &exif::Field) -> bool {
    if matches!(field.value, exif::Value::Unknown(..)) {
        return false;
    }
    let number = field.tag.number();
    match field.tag.context() {
        exif::Context::Tiff => KEPT_TIFF_TAGS.contains(&number),
        exif::Context::Exif => !DROPPED_EXIF_TAGS.contains(&number),
        exif::Context::Gps => true,
        _ => false,
    }
}

// ── Orientation ───────────────────────────────────────────────────────────────

fn read_u16(blob: &[u8], at: usize, little: bool) -> Option<u16> {
    let b = blob.get(at..at + 2)?;
    Some(if little {
        u16::from_le_bytes([b[0], b[1]])
    } else {
        u16::from_be_bytes([b[0], b[1]])
    })
}

fn read_u32(blob: &[u8], at: usize, little: bool) -> Option<u32> {
    let b = blob.get(at..at + 4)?;
    Some(if little {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    } else {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    })
}

/// Where the orientation value sits in a TIFF-structured blob: the offset of its
/// two bytes and whether the blob is little-endian. `None` when IFD0 has no
/// orientation entry of type SHORT with a count of one.
fn orientation_slot(blob: &[u8]) -> Option<(usize, bool)> {
    let little = match blob.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    if read_u16(blob, 2, little)? != 42 {
        return None;
    }
    let ifd = usize::try_from(read_u32(blob, 4, little)?).ok()?;
    let entries = usize::from(read_u16(blob, ifd, little)?);
    for i in 0..entries {
        let entry = ifd + 2 + 12 * i;
        if read_u16(blob, entry, little)? == 0x0112 {
            let is_one_short =
                read_u16(blob, entry + 2, little)? == 3 && read_u32(blob, entry + 4, little)? == 1;
            return is_one_short.then_some((entry + 8, little));
        }
    }
    None
}

/// The orientation tag (`0x0112`) of IFD0 in a TIFF-structured EXIF blob, or
/// `None` when the blob has none.
#[must_use]
pub fn exif_orientation(blob: &[u8]) -> Option<u16> {
    let (at, little) = orientation_slot(blob)?;
    read_u16(blob, at, little)
}

/// Set the orientation tag of IFD0 in place and return `true`; `false` (blob
/// untouched) when the blob has no such tag.
///
/// Only the two value bytes change, so every offset in the blob stays valid.
pub fn set_exif_orientation(blob: &mut [u8], orientation: u16) -> bool {
    let Some((at, little)) = orientation_slot(blob) else {
        return false;
    };
    let bytes = if little {
        orientation.to_le_bytes()
    } else {
        orientation.to_be_bytes()
    };
    blob[at..at + 2].copy_from_slice(&bytes);
    true
}

// ── Writing ───────────────────────────────────────────────────────────────────

/// Splice `exif_blob` (a bare TIFF-structured block, as the `extract` functions
/// return it) into the encoded JPEG or PNG `container` and return the new file
/// contents.
///
/// The pixel data is not touched; an EXIF block already present is
/// replaced. The container is recognised from its leading bytes.
///
/// # Errors
/// [`MetadataError::UnsupportedFormat`] for a TIFF or any other container,
/// [`MetadataError::Parse`] for a malformed JPEG or PNG,
/// [`MetadataError::TooLarge`] when a JPEG cannot hold the blob,
/// [`MetadataError::Io`] when writing fails.
pub fn inject_exif_into_bytes(
    container: &[u8],
    exif_blob: &[u8],
) -> Result<Vec<u8>, MetadataError> {
    let mut image = DynImage::from_bytes(container.to_vec().into())
        .map_err(|e| MetadataError::Parse(e.to_string()))?
        .ok_or_else(|| MetadataError::UnsupportedFormat(sniff_name(container).to_owned()))?;
    if matches!(image, DynImage::Jpeg(_)) && exif_blob.len() > JPEG_EXIF_MAX_BYTES {
        return Err(MetadataError::TooLarge {
            bytes: exif_blob.len(),
            limit: JPEG_EXIF_MAX_BYTES,
        });
    }
    image.set_exif(Some(exif_blob.to_vec().into()));
    let mut out = Vec::with_capacity(container.len() + exif_blob.len() + 16);
    image.encoder().write_to(&mut out)?;
    Ok(out)
}

/// The name of the container `bytes` starts like, for messages.
fn sniff_name(bytes: &[u8]) -> &'static str {
    if is_tiff_header(bytes) {
        "TIFF"
    } else {
        "unrecognised container"
    }
}

/// [`inject_exif_into_bytes`] on the file at `output_path`, in place.
///
/// # Errors
/// See [`inject_exif_into_bytes`]; [`MetadataError::Io`] also when the file
/// cannot be read.
pub fn inject_exif(output_path: &Path, exif_blob: &[u8]) -> Result<(), MetadataError> {
    let bytes = fs::read(output_path)?;
    let spliced = inject_exif_into_bytes(&bytes, exif_blob)?;
    fs::write(output_path, spliced)?;
    Ok(())
}

/// Copy the EXIF of `source_path` into the JPEG or PNG at `output_path`
/// unchanged ([`extract_exif`] then [`inject_exif`]).
///
/// Returns `Ok(true)` when metadata was copied and `Ok(false)` when the source
/// has none. The pipeline uses the two halves directly, because it must fix the
/// orientation tag in between (see the module documentation).
///
/// # Errors
/// See [`inject_exif`].
pub fn copy_metadata(source_path: &Path, output_path: &Path) -> Result<bool, MetadataError> {
    let Some(blob) = extract_exif(source_path) else {
        return Ok(false);
    };
    inject_exif(output_path, &blob)?;
    Ok(true)
}
