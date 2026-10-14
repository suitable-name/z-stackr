//! Discovery of trained models from a directory.
//!
//! Each model is a pair of files sharing a stem:
//!
//! ```text
//!   <stem>.mpk    — the weights (half precision `CompactRecorder` for the
//!                   `epoch` / `final` exports, full precision for the rolling
//!                   `latest` checkpoint; both are read by `infer::load_weights`)
//!   <stem>.json   — a [`ModelManifest`] describing the architecture
//! ```
//!
//! The manifest records the [`ModelSize`] preset plus the exact geometry, so a
//! model trained today still loads correctly even if the preset definitions
//! change later. A `.mpk` without a sibling manifest is ignored (its
//! architecture is unknown and cannot be reconstructed safely) — this is also
//! what keeps the optimiser checkpoint (`focusmerge_latest.optim.mpk`) out of
//! the picker.
//!
//! ## Architecture tags
//!
//! The manifest's `architecture` field names the network family, so four
//! families share one directory without being mis-loaded into each other:
//!
//! | tag | [`Architecture`] | network | role | file stems |
//! |---|---|---|---|---|
//! | `focusmerge-v1` | `FocusMerge` | [`crate::model::FocusMergeNet`] | fusion, pairwise | `focusmerge_*` |
//! | `batchmerge-v2` | `BatchMerge` | [`crate::model::BatchMergeNet`] | fusion, whole stack | `batchmerge_*` |
//! | `batchalign-v2` | `BatchAlign` | [`crate::model::BatchAlignNet`] | alignment, whole stack | `batchalign_*` |
//! | `fusionalign-v1` | `FusionAlign` | [`crate::model::FusionAlignNet`] | alignment, streaming | `fusionalign_*` |
//!
//! A manifest **without** an `architecture` field is `focusmerge-v1`, so every
//! manifest written before the field existed keeps loading. The bare family names
//! (`focusmerge`, `batchmerge`, `batchalign`, `fusionalign`) are accepted as
//! aliases of the tagged ones. A manifest with any other tag (a foreign or
//! future architecture) is **skipped and reported** by [`discover_models_report`]
//! rather than loaded into the wrong network, and every `load_*` method of
//! [`ModelEntry`] checks the tag again and fails with
//! [`DiscoveryError::ArchitectureMismatch`]. Fusion models and alignment models
//! are listed together; filter with [`ModelEntry::is_fusion`] /
//! [`ModelEntry::is_alignment`] for a picker.
//!
//! ## Names and ordering
//!
//! The trainer's rolling checkpoint is stored as `<family>_latest.model.mpk` (the
//! `.model` marker keeps it apart from the optimiser file, see `checkpoint`); the
//! entry name drops that marker and reads `<family>_latest`.
//!
//! Entries are ordered newest-usable first, so the apps' "default to the first
//! entry" behaviour picks the best model: `<family>_latest`, then
//! `<family>_final`, then `<family>_epoch_NNN` from the highest epoch down, then
//! any other name alphabetically; between families at the same rank the order is
//! focusmerge, batchmerge, batchalign, fusionalign. All fusion models come before
//! all alignment models, so a "first entry" default is never an aligner. A
//! manifest that cannot be read, parsed or validated is skipped on its own and
//! reported; it never hides the other models.

use std::{
    cmp::Reverse,
    path::{Path, PathBuf},
};

use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

use crate::{
    infer::{
        InferError, load_weights, load_weights_align, load_weights_batch, load_weights_fusion_align,
    },
    model::{
        BatchAlignNet, BatchAlignNetConfig, BatchMergeNet, BatchMergeNetConfig, FocusMergeNet,
        FocusMergeNetConfig, FusionAlignNet, FusionAlignNetConfig, ModelSize,
        align_common::check_align_geometry,
    },
};

/// Architecture tag of the pairwise fusion network
/// ([`crate::model::FocusMergeNet`]); also the default of a manifest without one.
pub const FOCUSMERGE_V1: &str = "focusmerge-v1";
/// Architecture tag of the whole-stack fusion network
/// ([`crate::model::BatchMergeNet`]).
pub const BATCHMERGE_V2: &str = "batchmerge-v2";
/// Architecture tag of the whole-stack alignment network
/// ([`crate::model::BatchAlignNet`]).
pub const BATCHALIGN_V2: &str = "batchalign-v2";
/// Architecture tag of the pairwise alignment network
/// ([`crate::model::FusionAlignNet`]).
pub const FUSIONALIGN_V1: &str = "fusionalign-v1";

/// `serde(default)` value for [`ModelManifest::architecture`].
fn default_architecture() -> String {
    FOCUSMERGE_V1.to_owned()
}

/// One of the four built-in network families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// [`crate::model::FocusMergeNet`] — pairwise fusion.
    FocusMerge,
    /// [`crate::model::BatchMergeNet`] — whole-stack fusion.
    BatchMerge,
    /// [`crate::model::BatchAlignNet`] — whole-stack alignment.
    BatchAlign,
    /// [`crate::model::FusionAlignNet`] — streaming pairwise alignment.
    FusionAlign,
}

impl Architecture {
    /// All families, in listing order.
    pub const ALL: [Self; 4] = [
        Self::FocusMerge,
        Self::BatchMerge,
        Self::BatchAlign,
        Self::FusionAlign,
    ];

    /// The manifest tag (`"focusmerge-v1"`, …).
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::FocusMerge => FOCUSMERGE_V1,
            Self::BatchMerge => BATCHMERGE_V2,
            Self::BatchAlign => BATCHALIGN_V2,
            Self::FusionAlign => FUSIONALIGN_V1,
        }
    }

    /// The bare family name, which is also the checkpoint file-stem prefix
    /// (`"focusmerge"`, `"batchmerge"`, `"batchalign"`, `"fusionalign"`).
    #[must_use]
    pub const fn family(self) -> &'static str {
        match self {
            Self::FocusMerge => "focusmerge",
            Self::BatchMerge => "batchmerge",
            Self::BatchAlign => "batchalign",
            Self::FusionAlign => "fusionalign",
        }
    }

    /// Parse a manifest tag or bare family name, case-insensitively.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|a| s == a.tag() || s == a.family())
    }

    /// Whether this family fuses images (as opposed to predicting alignments).
    #[must_use]
    pub const fn is_fusion(self) -> bool {
        matches!(self, Self::FocusMerge | Self::BatchMerge)
    }

    /// Whether this family predicts alignment matrices.
    #[must_use]
    pub const fn is_alignment(self) -> bool {
        !self.is_fusion()
    }
}

/// Errors while discovering or loading models.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    /// Filesystem error while scanning a directory or reading a manifest.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// A manifest file could not be parsed.
    #[error("manifest parse error in {path}: {source}")]
    Manifest {
        /// Manifest path.
        path: PathBuf,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
    /// A `load_*` method was called on an entry whose manifest names a different
    /// architecture than the one that method loads (for example `load_align` on a
    /// `focusmerge-v1` entry).
    #[error("architecture mismatch loading {path}: expected {expected}, found {found}")]
    ArchitectureMismatch {
        /// Architecture tag the calling `load_*` method requires.
        expected: &'static str,
        /// Architecture tag actually present in the manifest.
        found: String,
        /// Path to the `.mpk` weights file that was NOT loaded.
        path: PathBuf,
    },
    /// Loading the weights into the model failed.
    #[error(transparent)]
    Load(#[from] InferError),
    /// A manifest describes a geometry the network cannot be built with.
    #[error("invalid model geometry: {0}")]
    InvalidManifest(String),
}

/// Sidecar manifest describing a trained model's architecture.
///
/// `size` is sufficient to reconstruct the config via the matching
/// `*Config::from_size`; the optional fields override individual dimensions for
/// custom-trained geometries and make the manifest fully self-describing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Network family: a tag from the module docs' table. Defaults to
    /// [`FOCUSMERGE_V1`] so manifests written before the field existed keep
    /// loading unchanged. Use [`Self::architecture_kind`] to interpret it.
    #[serde(default = "default_architecture")]
    pub architecture: String,
    /// Capacity preset the model was trained at.
    pub size: ModelSize,
    /// Exact feature width (overrides the preset if present).
    #[serde(default)]
    pub width: Option<usize>,
    /// Exact context depth (overrides the preset if present). For the alignment
    /// networks: residual blocks per encoder stage.
    #[serde(default)]
    pub depth: Option<usize>,
    /// Exact channel-group count of the per-pixel normalisation (overrides the
    /// preset if present).
    #[serde(default)]
    pub norm_groups: Option<usize>,
    /// Residual-refinement scale (overrides the preset if present; unused by the
    /// alignment networks, which have no refinement head).
    #[serde(default)]
    pub refine_scale: Option<f64>,
    /// Correlation search radius in feature cells at the `/8` scale (overrides
    /// the preset if present; used by the alignment networks only).
    #[serde(default)]
    pub corr_radius: Option<usize>,
    /// Free-form human description (optional).
    #[serde(default)]
    pub description: String,
}

/// Check the width / group-count / scale part shared by the fusion geometries.
fn check_fusion_geometry(
    width: usize,
    norm_groups: usize,
    refine_scale: f64,
) -> Result<(), DiscoveryError> {
    if width == 0 || norm_groups == 0 {
        return Err(DiscoveryError::InvalidManifest(format!(
            "width ({width}) and norm_groups ({norm_groups}) must be positive"
        )));
    }
    if !width.is_multiple_of(norm_groups) {
        return Err(DiscoveryError::InvalidManifest(format!(
            "width {width} is not divisible by norm_groups {norm_groups}"
        )));
    }
    if !refine_scale.is_finite() {
        return Err(DiscoveryError::InvalidManifest(format!(
            "refine_scale {refine_scale} is not finite"
        )));
    }
    Ok(())
}

impl ModelManifest {
    /// Build a [`FOCUSMERGE_V1`] manifest from a preset, recording the exact
    /// geometry.
    #[must_use]
    pub fn from_size(size: ModelSize) -> Self {
        let cfg = FocusMergeNetConfig::from_size(size);
        Self {
            architecture: FOCUSMERGE_V1.to_owned(),
            size,
            width: Some(cfg.width),
            depth: Some(cfg.depth),
            norm_groups: Some(cfg.norm_groups),
            refine_scale: Some(cfg.refine_scale),
            corr_radius: None,
            description: String::new(),
        }
    }

    /// Build a [`BATCHMERGE_V2`] manifest from a preset, recording the exact
    /// geometry.
    #[must_use]
    pub fn from_size_batch(size: ModelSize) -> Self {
        let cfg = BatchMergeNetConfig::from_size(size);
        Self {
            architecture: BATCHMERGE_V2.to_owned(),
            size,
            width: Some(cfg.width),
            depth: Some(cfg.depth),
            norm_groups: Some(cfg.norm_groups),
            refine_scale: Some(cfg.refine_scale),
            corr_radius: None,
            description: String::new(),
        }
    }

    /// Build a [`BATCHALIGN_V2`] manifest from a preset, recording the exact
    /// geometry including the correlation radius. `refine_scale` is `None`.
    #[must_use]
    pub fn from_size_align(size: ModelSize) -> Self {
        let cfg = BatchAlignNetConfig::from_size(size);
        Self {
            architecture: BATCHALIGN_V2.to_owned(),
            size,
            width: Some(cfg.width),
            depth: Some(cfg.depth),
            norm_groups: Some(cfg.norm_groups),
            refine_scale: None,
            corr_radius: Some(cfg.corr_radius),
            description: String::new(),
        }
    }

    /// Build a [`FUSIONALIGN_V1`] manifest from a preset; mirrors
    /// [`Self::from_size_align`].
    #[must_use]
    pub fn from_size_fusion_align(size: ModelSize) -> Self {
        let cfg = FusionAlignNetConfig::from_size(size);
        Self {
            architecture: FUSIONALIGN_V1.to_owned(),
            size,
            width: Some(cfg.width),
            depth: Some(cfg.depth),
            norm_groups: Some(cfg.norm_groups),
            refine_scale: None,
            corr_radius: Some(cfg.corr_radius),
            description: String::new(),
        }
    }

    /// Build the manifest of `arch` for a preset (dispatches to the matching
    /// `from_size*`).
    #[must_use]
    pub fn from_size_for(arch: Architecture, size: ModelSize) -> Self {
        match arch {
            Architecture::FocusMerge => Self::from_size(size),
            Architecture::BatchMerge => Self::from_size_batch(size),
            Architecture::BatchAlign => Self::from_size_align(size),
            Architecture::FusionAlign => Self::from_size_fusion_align(size),
        }
    }

    /// The network family named by [`Self::architecture`], or `None` for a foreign
    /// or unrecognised tag.
    #[must_use]
    pub fn architecture_kind(&self) -> Option<Architecture> {
        Architecture::parse(&self.architecture)
    }

    /// Reconstruct the [`FocusMergeNetConfig`] this model was built with.
    ///
    /// Fails (instead of panicking later inside `ChannelNorm::new`) when the
    /// geometry is unusable: zero width or group count, a width that the group
    /// count does not divide, or a non-finite refine scale.
    ///
    /// # Errors
    /// [`DiscoveryError::InvalidManifest`] for any of those unusable geometries.
    pub fn to_config(&self) -> Result<FocusMergeNetConfig, DiscoveryError> {
        let mut cfg = FocusMergeNetConfig::from_size(self.size);
        if let Some(w) = self.width {
            cfg = cfg.with_width(w);
        }
        if let Some(d) = self.depth {
            cfg = cfg.with_depth(d);
        }
        if let Some(g) = self.norm_groups {
            cfg = cfg.with_norm_groups(g);
        }
        if let Some(s) = self.refine_scale {
            cfg = cfg.with_refine_scale(s);
        }
        check_fusion_geometry(cfg.width, cfg.norm_groups, cfg.refine_scale)?;
        Ok(cfg)
    }

    /// Reconstruct the [`BatchMergeNetConfig`] this model was built with.
    ///
    /// # Errors
    /// [`DiscoveryError::InvalidManifest`] as for [`Self::to_config`].
    pub fn to_batch_config(&self) -> Result<BatchMergeNetConfig, DiscoveryError> {
        let mut cfg = BatchMergeNetConfig::from_size(self.size);
        if let Some(w) = self.width {
            cfg = cfg.with_width(w);
        }
        if let Some(d) = self.depth {
            cfg = cfg.with_depth(d);
        }
        if let Some(g) = self.norm_groups {
            cfg = cfg.with_norm_groups(g);
        }
        if let Some(s) = self.refine_scale {
            cfg = cfg.with_refine_scale(s);
        }
        check_fusion_geometry(cfg.width, cfg.norm_groups, cfg.refine_scale)?;
        Ok(cfg)
    }

    /// Reconstruct the [`BatchAlignNetConfig`] this model was built with.
    ///
    /// # Errors
    /// [`DiscoveryError::InvalidManifest`] when the geometry cannot be built: zero,
    /// odd or non-divisible width / group count, or a correlation radius above
    /// [`crate::model::align_common::MAX_CORR_RADIUS`].
    pub fn to_align_config(&self) -> Result<BatchAlignNetConfig, DiscoveryError> {
        let mut cfg = BatchAlignNetConfig::from_size(self.size);
        if let Some(w) = self.width {
            cfg = cfg.with_width(w);
        }
        if let Some(d) = self.depth {
            cfg = cfg.with_depth(d);
        }
        if let Some(g) = self.norm_groups {
            cfg = cfg.with_norm_groups(g);
        }
        if let Some(r) = self.corr_radius {
            cfg = cfg.with_corr_radius(r);
        }
        check_align_geometry(cfg.width, cfg.norm_groups, cfg.corr_radius)
            .map_err(DiscoveryError::InvalidManifest)?;
        Ok(cfg)
    }

    /// Reconstruct the [`FusionAlignNetConfig`] this model was built with.
    ///
    /// # Errors
    /// As [`Self::to_align_config`].
    pub fn to_fusion_align_config(&self) -> Result<FusionAlignNetConfig, DiscoveryError> {
        let mut cfg = FusionAlignNetConfig::from_size(self.size);
        if let Some(w) = self.width {
            cfg = cfg.with_width(w);
        }
        if let Some(d) = self.depth {
            cfg = cfg.with_depth(d);
        }
        if let Some(g) = self.norm_groups {
            cfg = cfg.with_norm_groups(g);
        }
        if let Some(r) = self.corr_radius {
            cfg = cfg.with_corr_radius(r);
        }
        check_align_geometry(cfg.width, cfg.norm_groups, cfg.corr_radius)
            .map_err(DiscoveryError::InvalidManifest)?;
        Ok(cfg)
    }

    /// Validate the geometry against the family the manifest names.
    ///
    /// # Errors
    /// [`DiscoveryError::InvalidManifest`] for an unknown tag or an unusable
    /// geometry.
    pub fn validate(&self) -> Result<Architecture, DiscoveryError> {
        let arch = self.architecture_kind().ok_or_else(|| {
            DiscoveryError::InvalidManifest(format!(
                "unknown architecture tag `{}` (expected one of {FOCUSMERGE_V1}, \
                 {BATCHMERGE_V2}, {BATCHALIGN_V2}, {FUSIONALIGN_V1})",
                self.architecture
            ))
        })?;
        match arch {
            Architecture::FocusMerge => {
                self.to_config()?;
            }
            Architecture::BatchMerge => {
                self.to_batch_config()?;
            }
            Architecture::BatchAlign => {
                self.to_align_config()?;
            }
            Architecture::FusionAlign => {
                self.to_fusion_align_config()?;
            }
        }
        Ok(arch)
    }
}

/// A discovered, loadable model.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Display name (the file stem, minus the trainer's `.model` marker).
    pub name: String,
    /// Path to the `.mpk` weights file.
    pub weights_path: PathBuf,
    /// Parsed architecture manifest.
    pub manifest: ModelManifest,
}

impl ModelEntry {
    /// The network family of this entry, or `None` for a foreign tag (discovery
    /// never lists such an entry, but a hand-built one can carry one).
    #[must_use]
    pub fn architecture(&self) -> Option<Architecture> {
        self.manifest.architecture_kind()
    }

    /// Whether this entry is a fusion model ([`FOCUSMERGE_V1`] or
    /// [`BATCHMERGE_V2`]) that [`crate::runtime::LoadedModel`] can run. Use it to
    /// filter a fusion-model picker.
    #[must_use]
    pub fn is_fusion(&self) -> bool {
        self.architecture().is_some_and(Architecture::is_fusion)
    }

    /// Whether this entry is an alignment model ([`BATCHALIGN_V2`] or
    /// [`FUSIONALIGN_V1`]) that [`crate::runtime::LoadedAlignModel`] can run. Use
    /// it to filter an alignment-model picker.
    #[must_use]
    pub fn is_alignment(&self) -> bool {
        self.architecture().is_some_and(Architecture::is_alignment)
    }

    /// Whether this entry is specifically a [`BATCHALIGN_V2`] model.
    #[must_use]
    pub fn is_batch_alignment(&self) -> bool {
        self.architecture() == Some(Architecture::BatchAlign)
    }

    /// Whether this entry is specifically a [`FUSIONALIGN_V1`] model.
    #[must_use]
    pub fn is_pairwise_alignment(&self) -> bool {
        self.architecture() == Some(Architecture::FusionAlign)
    }

    /// `Ok(())` when the entry's tag is `expected`.
    fn require(&self, expected: Architecture) -> Result<(), DiscoveryError> {
        if self.architecture() == Some(expected) {
            Ok(())
        } else {
            Err(DiscoveryError::ArchitectureMismatch {
                expected: expected.tag(),
                found: self.manifest.architecture.clone(),
                path: self.weights_path.clone(),
            })
        }
    }

    /// Load this model's weights as a [`FocusMergeNet`] (pairwise fusion).
    ///
    /// # Errors
    /// [`DiscoveryError::ArchitectureMismatch`] when the entry is not
    /// [`FOCUSMERGE_V1`], [`DiscoveryError::InvalidManifest`] when the manifest
    /// geometry is unusable, or the weight-loading error when the `.mpk` cannot be
    /// read or does not match the architecture.
    pub fn load<B: Backend>(&self, device: &B::Device) -> Result<FocusMergeNet<B>, DiscoveryError> {
        self.require(Architecture::FocusMerge)?;
        Ok(load_weights(
            &self.manifest.to_config()?,
            &self.weights_path,
            device,
        )?)
    }

    /// Load this model's weights as a [`BatchMergeNet`] (whole-stack fusion).
    ///
    /// # Errors
    /// As [`Self::load`], for [`BATCHMERGE_V2`].
    pub fn load_batch<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BatchMergeNet<B>, DiscoveryError> {
        self.require(Architecture::BatchMerge)?;
        Ok(load_weights_batch(
            &self.manifest.to_batch_config()?,
            &self.weights_path,
            device,
        )?)
    }

    /// Load this model's weights as a [`BatchAlignNet`] (whole-stack alignment).
    ///
    /// # Errors
    /// As [`Self::load`], for [`BATCHALIGN_V2`].
    pub fn load_align<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BatchAlignNet<B>, DiscoveryError> {
        self.require(Architecture::BatchAlign)?;
        Ok(load_weights_align(
            &self.manifest.to_align_config()?,
            &self.weights_path,
            device,
        )?)
    }

    /// Load this model's weights as a [`FusionAlignNet`] (streaming alignment).
    ///
    /// # Errors
    /// As [`Self::load`], for [`FUSIONALIGN_V1`].
    pub fn load_fusion_align<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<FusionAlignNet<B>, DiscoveryError> {
        self.require(Architecture::FusionAlign)?;
        Ok(load_weights_fusion_align(
            &self.manifest.to_fusion_align_config()?,
            &self.weights_path,
            device,
        )?)
    }
}

/// A `.mpk` that discovery skipped, and why.
#[derive(Debug, Clone)]
pub struct SkippedModel {
    /// The weights file whose manifest was unusable.
    pub weights_path: PathBuf,
    /// Human-readable reason (unreadable, unparsable, foreign architecture or
    /// invalid manifest).
    pub reason: String,
}

/// Result of scanning one directory: the usable models plus what was skipped.
#[derive(Debug, Clone, Default)]
pub struct DiscoveryReport {
    /// Usable models, best default first (see the module docs for the order).
    pub models: Vec<ModelEntry>,
    /// Models skipped because their manifest was bad or named a foreign
    /// architecture.
    pub skipped: Vec<SkippedModel>,
}

/// Stem the pre-fix trainer used for its rolling checkpoint. The optimiser state
/// was written over it, so it is never a loadable model.
const LEGACY_LATEST_STEM: &str = "focusmerge_latest";

/// Entry name for a weights stem: the trainer's `.model` marker is dropped.
fn entry_name(stem: &str) -> &str {
    stem.strip_suffix(".model").unwrap_or(stem)
}

/// Ordering key: fusion models before alignment models (so a "first entry"
/// default is a fusion model), then newest-usable first (`latest`, `final`, epochs
/// descending, then everything else alphabetically); at equal rank the families
/// keep their [`Architecture::ALL`] order.
fn preference(name: &str) -> (u8, Reverse<u64>, usize, String) {
    for (index, arch) in Architecture::ALL.into_iter().enumerate() {
        let Some(rest) = name
            .strip_prefix(arch.family())
            .and_then(|r| r.strip_prefix('_'))
        else {
            continue;
        };
        let ranked = match rest {
            "latest" => Some((0, 0)),
            "final" => Some((1, 0)),
            _ => rest
                .strip_prefix("epoch_")
                .and_then(|n| n.parse::<u64>().ok())
                .map(|n| (2, n)),
        };
        if let Some((rank, epoch)) = ranked {
            return (rank, Reverse(epoch), index, name.to_owned());
        }
    }
    (3, Reverse(0), 0, name.to_owned())
}

/// Sort `entries` so the best default model comes first (fusion models, by the
/// entry's manifest rather than its name, before alignment models).
fn sort_by_preference(entries: &mut [ModelEntry]) {
    entries.sort_by_cached_key(|e| (e.is_alignment(), preference(&e.name)));
}

/// Read and validate the manifest next to `stem`; the error is the skip reason.
fn read_manifest(dir: &Path, stem: &str) -> Result<ModelManifest, String> {
    let manifest_path = dir.join(format!("{stem}.json"));
    let bytes = std::fs::read(&manifest_path)
        .map_err(|e| format!("cannot read {}: {e}", manifest_path.display()))?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes)
        .map_err(|e| format!("cannot parse {}: {e}", manifest_path.display()))?;
    manifest
        .validate()
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    Ok(manifest)
}

/// Scan a single directory for `(.mpk, .json)` model pairs, reporting what was
/// skipped.
///
/// A non-existent directory yields an empty report (not an error) so callers can
/// simply grey out the model picker when nothing is installed. `.mpk` files
/// without a sibling `.json` manifest are ignored silently; a manifest that is
/// unreadable, unparsable, names a foreign architecture or describes an
/// impossible geometry is skipped and listed in [`DiscoveryReport::skipped`] — it
/// never aborts the scan. All four built-in families are returned in one list;
/// filter with [`ModelEntry::is_fusion`] / [`ModelEntry::is_alignment`].
///
/// # Errors
/// [`DiscoveryError::Io`] when `dir` exists but cannot be listed.
pub fn discover_models_report(dir: &Path) -> Result<DiscoveryReport, DiscoveryError> {
    let mut report = DiscoveryReport::default();
    if !dir.exists() {
        return Ok(report);
    }

    let read = std::fs::read_dir(dir).map_err(|source| DiscoveryError::Io {
        path: dir.to_path_buf(),
        source,
    })?;

    for item in read {
        let Ok(item) = item else {
            continue; // one unreadable directory entry must not hide the rest.
        };
        let weights_path = item.path();
        if weights_path.extension().and_then(|e| e.to_str()) != Some("mpk") {
            continue;
        }
        let Some(stem) = weights_path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if !dir.join(format!("{stem}.json")).exists() {
            continue; // unknown architecture (e.g. the optimiser state) — skip.
        }
        if stem == LEGACY_LATEST_STEM {
            report.skipped.push(SkippedModel {
                weights_path,
                reason: "legacy rolling checkpoint: the old trainer overwrote it with the \
                         optimiser state, so it holds no usable weights (delete it; the \
                         trainer now writes focusmerge_latest.model.mpk)"
                    .to_owned(),
            });
            continue;
        }
        match read_manifest(dir, stem) {
            Ok(manifest) => report.models.push(ModelEntry {
                name: entry_name(stem).to_owned(),
                weights_path,
                manifest,
            }),
            Err(reason) => report.skipped.push(SkippedModel {
                weights_path,
                reason,
            }),
        }
    }

    sort_by_preference(&mut report.models);
    report
        .skipped
        .sort_by(|a, b| a.weights_path.cmp(&b.weights_path));
    Ok(report)
}

/// Scan a single directory for `(.mpk, .json)` model pairs, best default first.
///
/// Same scan as [`discover_models_report`]; skipped models are logged to stderr
/// (one line each) and otherwise dropped.
///
/// # Errors
/// As [`discover_models_report`].
pub fn discover_models(dir: &Path) -> Result<Vec<ModelEntry>, DiscoveryError> {
    let report = discover_models_report(dir)?;
    for skipped in &report.skipped {
        eprintln!(
            "model discovery: skipping {}: {}",
            skipped.weights_path.display(),
            skipped.reason
        );
    }
    Ok(report.models)
}

/// The directories searched by [`discover_default_models`], in priority order:
/// `models/` next to the executable, then `models/` in the working directory.
#[must_use]
pub fn default_model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.join("models"));
    }
    dirs.push(PathBuf::from("models"));
    dirs
}

/// Best-effort discovery across [`default_model_dirs`].
///
/// Directories that fail to read are skipped; results are de-duplicated by name
/// (earlier directories win) and ordered best-default-first (see the module
/// docs). Intended for UI population where a hard error is undesirable — an
/// empty result simply greys out the picker.
#[must_use]
pub fn discover_default_models() -> Vec<ModelEntry> {
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<ModelEntry> = Vec::new();
    for dir in default_model_dirs() {
        let Ok(found) = discover_models(&dir) else {
            continue;
        };
        for entry in found {
            if seen.insert(entry.name.clone()) {
                out.push(entry);
            }
        }
    }
    sort_by_preference(&mut out);
    out
}

/// [`discover_default_models`] restricted to the fusion families
/// ([`ModelEntry::is_fusion`]) — the list for an "AI model" picker.
#[must_use]
pub fn discover_default_fusion_models() -> Vec<ModelEntry> {
    let mut models = discover_default_models();
    models.retain(ModelEntry::is_fusion);
    models
}

/// [`discover_default_models`] restricted to the alignment families
/// ([`ModelEntry::is_alignment`]) — the list for an "alignment model" picker.
#[must_use]
pub fn discover_default_alignment_models() -> Vec<ModelEntry> {
    let mut models = discover_default_models();
    models.retain(ModelEntry::is_alignment);
    models
}
