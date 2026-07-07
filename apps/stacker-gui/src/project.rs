//! Picture-set project files (`.zsproj`): save/load a stacking session.
//!
//! Covers source file references (optionally with embedded bytes), the
//! fused output(s) (optionally embedded), the stacking settings (optional),
//! and a best-effort snapshot of any active retouch session (optionally
//! including its full undo/redo history).
//!
//! This module is deliberately free of any `Mutex`/`RetouchState` awareness:
//! callers (see `callbacks::project`) lock, extract, and clone whatever
//! plain data they need *before* calling into here, so no lock is ever held
//! across the file I/O/compression work below.
//!
//! ## On-disk layout
//!
//! ```text
//! [8 bytes]  magic "ZSPROJ01"
//! [8 bytes]  u64 LE: length of the compressed manifest section
//! [N bytes]  zstd-compressed, bincode-next-serialized `ProjectManifest`
//! [rest]     raw, UNCOMPRESSED blob section — concatenated embedded file
//!            bytes, referenced by `BlobRef { offset, len }` inside the
//!            manifest.
//! ```
//!
//! The blob section is deliberately left uncompressed: embedded sources/
//! outputs are already-compressed image formats (PNG/JPEG/TIFF/RAW), so a
//! second compression pass over them wastes CPU for near-zero size benefit.
//! Only the manifest itself (settings, paths, retouch alpha/history floats —
//! all genuinely compressible) goes through zstd.
//!
//! GUI-only by design (per this feature's current scope) — the CLI does not
//! link this module.

use std::{
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use stacker_core::settings::StackingSettings;

/// File magic — the first 8 bytes of every `.zsproj` file.
pub const MAGIC: &[u8; 8] = b"ZSPROJ01";

/// Bumped whenever `ProjectManifest`'s shape changes in a way that isn't
/// safely forward-compatible.
///
/// `load_project` refuses anything with a `format_version` newer than this
/// — an older file is still accepted as-is (serde's `#[serde(default)]`-
/// style field evolution handles adding new optional fields over time
/// without a version bump at all).
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobRef {
    pub offset: u64,
    pub len: u64,
    /// Lowercase extension without the dot (e.g. "png", "jpg", "cr2") —
    /// needed to know how to decode the blob back into an image; the
    /// container format itself doesn't care, it's just bytes.
    pub ext: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceEntry {
    pub original_path: PathBuf,
    pub embedded: Option<BlobRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputEntry {
    pub original_path: Option<PathBuf>,
    pub algorithm: String,
    pub embedded: Option<BlobRef>,
}

/// The full undo/redo stack + cursor for a retouch session.
///
/// As returned by `RetouchHistory::export_full`. Kept as a separate,
/// always-optional field from `RetouchSnapshot::alpha` because it can be
/// considerably larger: up to `RetouchHistory::MAX_DEPTH` (50)
/// full-resolution alpha masks, versus just the one current-state mask
/// that's saved unconditionally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistorySnapshot {
    pub stack: Vec<Vec<f32>>,
    pub cursor: Option<usize>,
}

/// Best-effort snapshot of an active retouch session at save time.
///
/// Restored on load only if `base_output_idx`/`src_output_idx` (indices
/// into `ProjectManifest::outputs`) resolve to outputs that are actually
/// available (embedded, or reloadable from their `original_path`) and end
/// up with matching dimensions — otherwise it's silently dropped on load
/// rather than failing the whole project open. See [`Self::build`]'s doc
/// comment for the one case this precisely round-trips (single-algorithm,
/// self-blend sessions) versus the conservative fallback for a rewired
/// (e.g. "All Three" donor-swapped) session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetouchSnapshot {
    pub base_output_idx: usize,
    pub src_output_idx: usize,
    pub width: usize,
    pub height: usize,
    pub alpha: Vec<f32>,
    /// Only present when the user opted in via
    /// `SaveOptions::include_rebrush_history` — see [`HistorySnapshot`]'s
    /// doc comment for why this is kept separate from `alpha`.
    pub history: Option<HistorySnapshot>,
}

impl RetouchSnapshot {
    /// Build a snapshot from already-unlocked, plain session data.
    ///
    /// `result_path` is the active session's owning result path (i.e.
    /// `RetouchState::result_path`); this only produces a snapshot when it
    /// matches one of `outputs` by path — a session re-wired to a
    /// different donor (e.g. via "Use as Retouch Source", or the "All
    /// Three" apex/relief pairing) still gets its alpha mask saved this
    /// way, but on load falls back to re-blending against that same output
    /// for both `base` and `src`, a conservative, safe default rather than
    /// a guess about which donor was actually in play at save time.
    ///
    /// Returns `None` when there's nothing to anchor the snapshot to (no
    /// session, or its result path isn't among `outputs`).
    #[must_use]
    pub fn build(
        base_dimensions: (usize, usize),
        alpha: Vec<f32>,
        result_path: Option<&Path>,
        outputs: &[(PathBuf, String)],
        history: Option<HistorySnapshot>,
    ) -> Option<Self> {
        let base_output_idx = outputs
            .iter()
            .position(|(p, _)| Some(p.as_path()) == result_path)?;
        Some(Self {
            base_output_idx,
            src_output_idx: base_output_idx,
            width: base_dimensions.0,
            height: base_dimensions.1,
            alpha,
            history,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub format_version: u32,
    pub settings: Option<StackingSettings>,
    pub sources: Vec<SourceEntry>,
    pub outputs: Vec<OutputEntry>,
    pub retouch: Option<RetouchSnapshot>,
}

/// What to include when saving — every axis is independently optional.
///
/// Per the save dialog's checkboxes. `include_rebrush_history` is only
/// ever offered in the UI when there's actually a painted-on retouch
/// session to save history for; see `callbacks::project` for that check.
// Four independent yes/no save-dialog toggles are the clearest model for
// this data (same precedent as `StackingSettings`) — a bitflags/enum
// encoding would just obscure the same four independent choices.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy)]
pub struct SaveOptions {
    pub include_settings: bool,
    pub embed_sources: bool,
    pub embed_outputs: bool,
    pub include_rebrush_history: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("not a z-stackr project file (bad magic)")]
    BadMagic,
    #[error(
        "project file is from a newer, incompatible format version \
         ({found}; this build understands up to {supported})"
    )]
    UnsupportedVersion { found: u32, supported: u32 },
    #[error("failed to encode project manifest: {0}")]
    Encode(String),
    #[error("failed to decode project manifest: {0}")]
    Decode(String),
    #[error("failed to compress project manifest: {0}")]
    Compress(String),
}

/// Save a picture-set project to `path`.
///
/// `sources` are the currently-loaded source file paths; `outputs` are the
/// currently-listed result paths paired with the algorithm label that
/// produced them. `settings` is included only when
/// `opts.include_settings`; source/output bytes are embedded only when the
/// matching `opts` flag is set — paths are always recorded either way, so a
/// reference-only project still round-trips as long as the original files
/// haven't moved. `retouch_snapshot` (built via [`RetouchSnapshot::build`]
/// by the caller, from an already-unlocked session) is stored as-is,
/// regardless of `opts` — the caller is responsible for having honoured
/// `opts.include_rebrush_history` when deciding whether to populate its
/// `history` field.
///
/// # Errors
///
/// Returns [`ProjectError::Io`] if `path` (or any embedded source/output
/// file) can't be written/read, or [`ProjectError::Encode`]/
/// [`ProjectError::Compress`] if serialization or compression fails.
pub fn save_project(
    path: &Path,
    sources: &[PathBuf],
    outputs: &[(PathBuf, String)],
    settings: &StackingSettings,
    retouch_snapshot: Option<RetouchSnapshot>,
    opts: SaveOptions,
) -> Result<(), ProjectError> {
    let mut blob = Vec::new();

    let source_entries: Vec<SourceEntry> = sources
        .iter()
        .map(|p| SourceEntry {
            original_path: p.clone(),
            embedded: opts
                .embed_sources
                .then(|| embed_file(p, &mut blob))
                .flatten(),
        })
        .collect();

    let output_entries: Vec<OutputEntry> = outputs
        .iter()
        .map(|(p, algo)| OutputEntry {
            original_path: Some(p.clone()),
            algorithm: algo.clone(),
            embedded: opts
                .embed_outputs
                .then(|| embed_file(p, &mut blob))
                .flatten(),
        })
        .collect();

    let manifest = ProjectManifest {
        format_version: FORMAT_VERSION,
        settings: opts.include_settings.then(|| settings.clone()),
        sources: source_entries,
        outputs: output_entries,
        retouch: retouch_snapshot,
    };

    let encoded = bincode_next::serde::encode_to_vec(&manifest, bincode_next::config::standard())
        .map_err(|e| ProjectError::Encode(e.to_string()))?;
    let compressed =
        zstd::encode_all(&encoded[..], 19).map_err(|e| ProjectError::Compress(e.to_string()))?;

    let mut file = fs::File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&(compressed.len() as u64).to_le_bytes())?;
    file.write_all(&compressed)?;
    file.write_all(&blob)?;
    Ok(())
}

/// Embed `path`'s raw bytes into `blob`, appending them and returning the
/// resulting `BlobRef`. Returns `None` (and leaves `blob` untouched) if the
/// file can't be read — a failed embed degrades to reference-only for that
/// one entry instead of failing the whole save.
fn embed_file(path: &Path, blob: &mut Vec<u8>) -> Option<BlobRef> {
    let bytes = fs::read(path).ok()?;
    let offset = blob.len() as u64;
    let len = bytes.len() as u64;
    blob.extend_from_slice(&bytes);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    Some(BlobRef { offset, len, ext })
}

/// A loaded project, ready for the GUI to apply back onto its own state.
pub struct LoadedProject {
    pub manifest: ProjectManifest,
    /// The raw blob section, sliced per-entry via each `BlobRef` — callers
    /// index into this with [`blob_slice`] rather than this module
    /// re-parsing image formats itself (that's `image_utils`'s job).
    pub blob: Vec<u8>,
}

/// Load a picture-set project from `path`.
///
/// # Errors
///
/// Returns [`ProjectError::Io`] on any read failure,
/// [`ProjectError::BadMagic`] if `path` isn't a `.zsproj` file,
/// [`ProjectError::UnsupportedVersion`] if it was written by a newer,
/// incompatible version of this format, or
/// [`ProjectError::Compress`]/[`ProjectError::Decode`] if decompression or
/// deserialization of the manifest fails.
pub fn load_project(path: &Path) -> Result<LoadedProject, ProjectError> {
    let mut file = fs::File::open(path)?;

    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(ProjectError::BadMagic);
    }

    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let manifest_len = u64::from_le_bytes(len_buf) as usize;

    let mut compressed = vec![0u8; manifest_len];
    file.read_exact(&mut compressed)?;

    let mut blob = Vec::new();
    file.read_to_end(&mut blob)?;

    let decoded_manifest_bytes =
        zstd::decode_all(&compressed[..]).map_err(|e| ProjectError::Compress(e.to_string()))?;
    let (manifest, _): (ProjectManifest, usize) = bincode_next::serde::decode_from_slice(
        &decoded_manifest_bytes,
        bincode_next::config::standard(),
    )
    .map_err(|e| ProjectError::Decode(e.to_string()))?;

    if manifest.format_version > FORMAT_VERSION {
        return Err(ProjectError::UnsupportedVersion {
            found: manifest.format_version,
            supported: FORMAT_VERSION,
        });
    }

    Ok(LoadedProject { manifest, blob })
}

/// Slice `blob_ref`'s bytes out of `blob`.
#[must_use]
pub fn blob_slice<'a>(blob: &'a [u8], blob_ref: &BlobRef) -> &'a [u8] {
    let start = blob_ref.offset as usize;
    let end = start + blob_ref.len as usize;
    &blob[start..end]
}
