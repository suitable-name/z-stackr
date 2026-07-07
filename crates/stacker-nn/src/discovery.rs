//! Discovery of trained models from a directory.
//!
//! Each model is a pair of files sharing a stem:
//!
//! ```text
//!   <name>.mpk    — the weights (written by `CompactRecorder`)
//!   <name>.json   — a [`ModelManifest`] describing the architecture
//! ```
//!
//! The manifest records the [`ModelSize`] preset plus the exact geometry, so a
//! model trained today still loads correctly even if the preset definitions
//! change later. A `.mpk` without a sibling manifest is ignored (its
//! architecture is unknown and cannot be reconstructed safely).
//!
//! ## Architecture tags
//!
//! [`ModelManifest::architecture`] names which of this crate's FOUR built-in
//! network families the checkpoint holds:
//!
//! | Tag | Constant | Model | Loader |
//! |---|---|---|---|
//! | `"focusmerge-v1"` | [`FOCUSMERGE_V1`] | [`crate::model::FocusMergeNet`] (pairwise fusion) | [`ModelEntry::load`] |
//! | `"batchmerge-v2"` | [`BATCHMERGE_V2`] | [`crate::model::BatchMergeNet`] (batch fusion) | [`ModelEntry::load_batch`] |
//! | `"batchalign-v2"` | [`BATCHALIGN_V2`] | [`crate::model::BatchAlignNet`] (batch alignment, v2 architecture) | [`ModelEntry::load_align`] |
//! | `"fusionalign-v1"` | [`FUSIONALIGN_V1`] | [`crate::model::FusionAlignNet`] (pairwise alignment) | [`ModelEntry::load_fusion_align`] |
//!
//! [`discover_models`] accepts all four tags into one flat [`ModelEntry`]
//! list (so a UI can show every trained model in a directory at once); use
//! [`ModelEntry::is_fusion`] to filter for a fusion-model picker, or
//! [`ModelEntry::is_alignment`] for EITHER alignment architecture (batch or
//! pairwise), plus the finer-grained [`ModelEntry::is_batch_alignment`] /
//! [`ModelEntry::is_pairwise_alignment`] when a caller specifically needs to
//! know which alignment *kind* it found (e.g. the tiled pipeline dispatching
//! a whole-stack [`ModelEntry::load_align`] call vs. a streaming
//! per-frame [`ModelEntry::load_fusion_align`] loop — see
//! `crate::runtime::LoadedAlignModel`'s docs).
//!
//! **Each `load_*` method validates `manifest.architecture` against the tag
//! it expects and returns [`DiscoveryError::ArchitectureMismatch`] on a
//! mismatch** — e.g. calling `load_align` on a `focusmerge-v1` entry is a
//! caller bug (wrong picker wired to wrong loader) and must fail loudly
//! rather than attempt to deserialise weights into the wrong parameter
//! layout (which panics deep inside the record loader with a confusing
//! error, or worse, silently succeeds into garbage if the layouts happen to
//! overlap in shape).
//!
//! ## Forward compatibility with third-party architectures
//!
//! [`ModelManifest`] carries an `architecture` tag (defaulted to
//! [`FOCUSMERGE_V1`] via `serde(default)`, so existing manifests without the
//! field keep working unchanged). [`discover_models`] skips any manifest
//! whose `architecture` is not one of the four built-in tags above — a third
//! party implementing [`crate::traits::FusionModel`],
//! [`crate::traits::BatchAlignmentModel`], or
//! [`crate::traits::PairAlignmentModel`] with a different network is free to
//! drop their own `<name>.mpk`-equivalent + manifest-equivalent files
//! alongside the built-in ones without this crate misinterpreting the
//! foreign checkpoint as a built-in architecture and failing to load it (or
//! worse, loading it into the wrong parameter layout). Discovery, loading,
//! and runtime dispatch for those architectures is up to the third party —
//! see the crate docs' "Extending with your own architecture" section.

use std::path::{Path, PathBuf};

use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

use crate::{
    infer::{InferError, load_weights},
    model::{
        BatchAlignNet, BatchMergeNet, BatchMergeNetConfig, FocusMergeNet, FocusMergeNetConfig,
        FusionAlignNet, ModelSize,
    },
};

/// Architecture tag identifying the built-in [`crate::model::FocusMergeNet`]
/// (pairwise fusion) geometry. See the module docs' "Architecture tags" table.
pub const FOCUSMERGE_V1: &str = "focusmerge-v1";
/// Architecture tag identifying the built-in [`crate::model::BatchMergeNet`]
/// (batch fusion) geometry. See the module docs' "Architecture tags" table.
///
/// The version suffix in the tag is part of the architecture identifier; a
/// manifest with any unrecognised tag is skipped by [`discover_models`] like
/// any other foreign tag.
pub const BATCHMERGE_V2: &str = "batchmerge-v2";
/// Architecture tag identifying the built-in [`crate::model::BatchAlignNet`]
/// (alignment, v2 architecture — see `docs/batchalign-v2-design.md`) geometry.
/// See the module docs' "Architecture tags" table.
///
/// The version suffix in the tag is part of the architecture identifier; a
/// manifest with any unrecognised tag is skipped by [`discover_models`] like
/// any other foreign tag.
pub const BATCHALIGN_V2: &str = "batchalign-v2";
/// Architecture tag identifying the built-in [`crate::model::FusionAlignNet`]
/// (pairwise alignment — see `docs/fusionalign-design.md`) geometry. This
/// identifies the pairwise alignment family. See the module docs' "Architecture
/// tags" table.
pub const FUSIONALIGN_V1: &str = "fusionalign-v1";

/// `serde(default)` value for [`ModelManifest::architecture`] — see
/// [`FOCUSMERGE_V1`].
fn default_architecture() -> String {
    FOCUSMERGE_V1.to_owned()
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
    /// A `load_*` method was called on a [`ModelEntry`] whose manifest names
    /// a different architecture than the one that method loads (e.g.
    /// `load_align` on a `focusmerge-v1` entry). See the module docs'
    /// "Architecture tags" section.
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
}

/// Sidecar manifest describing a trained model's architecture.
///
/// `size` is sufficient to reconstruct the config via
/// [`FocusMergeNetConfig::from_size`] (or the `BatchMergeNetConfig`/
/// `BatchAlignNetConfig` equivalents); the optional fields override
/// individual dimensions for custom-trained geometries and make the manifest
/// fully self-describing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Architecture tag identifying which network family this manifest
    /// describes — one of [`FOCUSMERGE_V1`], [`BATCHMERGE_V2`],
    /// [`BATCHALIGN_V2`], or a third-party value.
    ///
    /// Defaults to [`FOCUSMERGE_V1`] via `serde(default)` so every manifest
    /// written before this field existed keeps loading unchanged.
    /// [`discover_models`] skips manifests whose tag is none of the three
    /// built-in values, and each [`ModelEntry::load`]/`load_batch`/`load_align`
    /// method further validates the tag matches what IT loads (see the
    /// module docs).
    #[serde(default = "default_architecture")]
    pub architecture: String,
    /// Capacity preset the model was trained at.
    pub size: ModelSize,
    /// Exact feature width (overrides the preset if present).
    #[serde(default)]
    pub width: Option<usize>,
    /// Exact context depth (overrides the preset if present).
    #[serde(default)]
    pub depth: Option<usize>,
    /// Exact `GroupNorm` group count (overrides the preset if present).
    #[serde(default)]
    pub norm_groups: Option<usize>,
    /// Residual-refinement scale (overrides the preset if present; unused by
    /// the alignment architecture, which has no refinement head).
    #[serde(default)]
    pub refine_scale: Option<f64>,
    /// Correlation search radius, in feature cells at the /8 encoder scale
    /// (overrides the preset if present; unused by the fusion architectures).
    /// See [`crate::model::BatchAlignNetConfig::corr_radius`]'s docs for the
    /// displacement-range/training-data coupling. `serde(default)` so
    /// manifests written before v2 existed (and fusion manifests, which
    /// never set this) keep deserialising unchanged.
    #[serde(default)]
    pub corr_radius: Option<usize>,
    /// Free-form human description (optional).
    #[serde(default)]
    pub description: String,
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
    /// geometry. `refine_scale` is left `None` — the alignment architecture
    /// has no refinement head. `corr_radius` is recorded from the preset's
    /// config so the manifest round-trips the exact correlation search
    /// radius the checkpoint was trained with.
    #[must_use]
    pub fn from_size_align(size: ModelSize) -> Self {
        let cfg = crate::model::BatchAlignNetConfig::from_size(size);
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

    /// Build a [`FUSIONALIGN_V1`] manifest from a preset, recording the exact
    /// geometry. Mirrors [`Self::from_size_align`] exactly — the pairwise
    /// alignment architecture reuses the identical
    /// `width`/`depth`/`norm_groups`/`corr_radius` knobs as `BatchAlignNet`,
    /// just applied to a two-input pair rather than a whole stack, so no new
    /// manifest fields are needed. `refine_scale` is left `None` for the same
    /// reason as `from_size_align` (no refinement head).
    #[must_use]
    pub fn from_size_fusion_align(size: ModelSize) -> Self {
        let cfg = crate::model::FusionAlignNetConfig::from_size(size);
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

    /// Reconstruct the [`FocusMergeNetConfig`] this model was built with.
    #[must_use]
    pub fn to_config(&self) -> FocusMergeNetConfig {
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
        cfg
    }

    /// Reconstruct the [`BatchMergeNetConfig`] this model was built with.
    #[must_use]
    pub fn to_batch_config(&self) -> BatchMergeNetConfig {
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
        cfg
    }

    /// Reconstruct the [`crate::model::BatchAlignNetConfig`] this model was built with.
    #[must_use]
    pub fn to_align_config(&self) -> crate::model::BatchAlignNetConfig {
        let mut cfg = crate::model::BatchAlignNetConfig::from_size(self.size);
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
        cfg
    }

    /// Reconstruct the [`crate::model::FusionAlignNetConfig`] this model was
    /// built with. Mirrors [`Self::to_align_config`] exactly.
    #[must_use]
    pub fn to_fusion_align_config(&self) -> crate::model::FusionAlignNetConfig {
        let mut cfg = crate::model::FusionAlignNetConfig::from_size(self.size);
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
        cfg
    }
}

/// A discovered, loadable model.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Display name (the file stem).
    pub name: String,
    /// Path to the `.mpk` weights file.
    pub weights_path: PathBuf,
    /// Parsed architecture manifest.
    pub manifest: ModelManifest,
}

impl ModelEntry {
    /// Returns `true` if this entry's [`ModelManifest::architecture`] is a
    /// fusion architecture ([`FOCUSMERGE_V1`] or [`BATCHMERGE_V2`]).
    ///
    /// I.e. it implements [`crate::traits::FusionModel`] and can be run
    /// through [`crate::runtime::LoadedModel`]. Use to filter
    /// [`discover_models`]'s output for a fusion-model picker.
    #[must_use]
    pub fn is_fusion(&self) -> bool {
        self.manifest.architecture == FOCUSMERGE_V1 || self.manifest.architecture == BATCHMERGE_V2
    }

    /// Returns `true` if this entry's [`ModelManifest::architecture`] is
    /// [`BATCHALIGN_V2`] or [`FUSIONALIGN_V1`] — i.e. EITHER alignment
    /// architecture family.
    ///
    /// Both implement an alignment trait ([`crate::traits::BatchAlignmentModel`]
    /// or [`crate::traits::PairAlignmentModel`]) and can be run through
    /// [`crate::runtime::LoadedAlignModel`], which returns the same
    /// `Vec<nalgebra::Matrix3<f32>>` regardless of which kind is loaded. Use
    /// this to filter [`discover_models`]'s output for a picker that just
    /// wants "any alignment model, don't care which architecture" (e.g. the
    /// GUI's/CLI's alignment-model dropdown, which resolves a plain name
    /// string against [`crate::runtime::LoadedAlignModel::load`] and doesn't
    /// need to know the kind up front).
    ///
    /// If a caller specifically needs to know WHICH alignment kind an entry
    /// is (e.g. the tiled pipeline choosing between a whole-stack
    /// [`Self::load_align`] call and a streaming per-frame
    /// [`Self::load_fusion_align`] loop), use [`Self::is_batch_alignment`] /
    /// [`Self::is_pairwise_alignment`] instead.
    #[must_use]
    pub fn is_alignment(&self) -> bool {
        self.manifest.architecture == BATCHALIGN_V2 || self.manifest.architecture == FUSIONALIGN_V1
    }

    /// Returns `true` if this entry's [`ModelManifest::architecture`] is
    /// specifically [`BATCHALIGN_V2`] (whole-stack batch alignment).
    ///
    /// Narrower than [`Self::is_alignment`] — use this when the caller's
    /// dispatch genuinely differs between the batch and pairwise alignment
    /// kinds (see [`crate::runtime::LoadedAlignModel`]'s docs for the
    /// batch-vs-pairwise dispatch this enables).
    #[must_use]
    pub fn is_batch_alignment(&self) -> bool {
        self.manifest.architecture == BATCHALIGN_V2
    }

    /// Returns `true` if this entry's [`ModelManifest::architecture`] is
    /// specifically [`FUSIONALIGN_V1`] (streaming pairwise alignment).
    ///
    /// See [`Self::is_batch_alignment`]'s docs for when to prefer this over
    /// the broader [`Self::is_alignment`].
    #[must_use]
    pub fn is_pairwise_alignment(&self) -> bool {
        self.manifest.architecture == FUSIONALIGN_V1
    }

    /// Load this model's weights as a [`FocusMergeNet`] (pairwise fusion).
    ///
    /// # Errors
    ///
    /// Returns [`DiscoveryError::ArchitectureMismatch`] if
    /// `self.manifest.architecture` is not [`FOCUSMERGE_V1`], or
    /// [`DiscoveryError::Load`] if the `.mpk` fails to deserialise.
    pub fn load<B: Backend>(&self, device: &B::Device) -> Result<FocusMergeNet<B>, DiscoveryError> {
        if self.manifest.architecture != FOCUSMERGE_V1 {
            return Err(DiscoveryError::ArchitectureMismatch {
                expected: FOCUSMERGE_V1,
                found: self.manifest.architecture.clone(),
                path: self.weights_path.clone(),
            });
        }
        Ok(load_weights(
            &self.manifest.to_config(),
            &self.weights_path,
            device,
        )?)
    }

    /// Load this model's weights as a [`BatchMergeNet`] (batch fusion).
    ///
    /// # Errors
    ///
    /// Returns [`DiscoveryError::ArchitectureMismatch`] if
    /// `self.manifest.architecture` is not [`BATCHMERGE_V2`], or
    /// [`DiscoveryError::Load`] if the `.mpk` fails to deserialise.
    pub fn load_batch<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BatchMergeNet<B>, DiscoveryError> {
        if self.manifest.architecture != BATCHMERGE_V2 {
            return Err(DiscoveryError::ArchitectureMismatch {
                expected: BATCHMERGE_V2,
                found: self.manifest.architecture.clone(),
                path: self.weights_path.clone(),
            });
        }
        Ok(crate::infer::load_weights_batch(
            &self.manifest.to_batch_config(),
            &self.weights_path,
            device,
        )?)
    }

    /// Load this model's weights as a [`BatchAlignNet`] (alignment).
    ///
    /// # Errors
    ///
    /// Returns [`DiscoveryError::ArchitectureMismatch`] if
    /// `self.manifest.architecture` is not [`BATCHALIGN_V2`], or
    /// [`DiscoveryError::Load`] if the `.mpk` fails to deserialise.
    pub fn load_align<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BatchAlignNet<B>, DiscoveryError> {
        if self.manifest.architecture != BATCHALIGN_V2 {
            return Err(DiscoveryError::ArchitectureMismatch {
                expected: BATCHALIGN_V2,
                found: self.manifest.architecture.clone(),
                path: self.weights_path.clone(),
            });
        }
        Ok(crate::infer::load_weights_align(
            &self.manifest.to_align_config(),
            &self.weights_path,
            device,
        )?)
    }

    /// Load this model's weights as a [`FusionAlignNet`] (pairwise alignment).
    ///
    /// # Errors
    ///
    /// Returns [`DiscoveryError::ArchitectureMismatch`] if
    /// `self.manifest.architecture` is not [`FUSIONALIGN_V1`], or
    /// [`DiscoveryError::Load`] if the `.mpk` fails to deserialise.
    pub fn load_fusion_align<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<FusionAlignNet<B>, DiscoveryError> {
        if self.manifest.architecture != FUSIONALIGN_V1 {
            return Err(DiscoveryError::ArchitectureMismatch {
                expected: FUSIONALIGN_V1,
                found: self.manifest.architecture.clone(),
                path: self.weights_path.clone(),
            });
        }
        Ok(crate::infer::load_weights_fusion_align(
            &self.manifest.to_fusion_align_config(),
            &self.weights_path,
            device,
        )?)
    }
}

/// Scan a single directory for `(.mpk, .json)` model pairs.
///
/// A non-existent directory yields an empty list (not an error) so callers can
/// simply grey out the model picker when nothing is installed. `.mpk` files
/// without a sibling `.json` manifest are skipped. Accepts all four built-in
/// architecture tags ([`FOCUSMERGE_V1`], [`BATCHMERGE_V2`], [`BATCHALIGN_V2`],
/// [`FUSIONALIGN_V1`]) into one flat list — use [`ModelEntry::is_fusion`] /
/// [`ModelEntry::is_alignment`] to filter for a specific kind of picker (or
/// [`ModelEntry::is_batch_alignment`] / [`ModelEntry::is_pairwise_alignment`]
/// for the finer-grained alignment split).
pub fn discover_models(dir: &Path) -> Result<Vec<ModelEntry>, DiscoveryError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let read = std::fs::read_dir(dir).map_err(|source| DiscoveryError::Io {
        path: dir.to_path_buf(),
        source,
    })?;

    let mut entries = Vec::new();
    for item in read {
        let item = item.map_err(|source| DiscoveryError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let weights_path = item.path();
        if weights_path.extension().and_then(|e| e.to_str()) != Some("mpk") {
            continue;
        }
        let Some(stem) = weights_path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let manifest_path = dir.join(format!("{stem}.json"));
        if !manifest_path.exists() {
            continue; // unknown architecture — skip.
        }
        let bytes = std::fs::read(&manifest_path).map_err(|source| DiscoveryError::Io {
            path: manifest_path.clone(),
            source,
        })?;
        let manifest: ModelManifest =
            serde_json::from_slice(&bytes).map_err(|source| DiscoveryError::Manifest {
                path: manifest_path.clone(),
                source,
            })?;

        // Unknown-architecture manifests cannot be reconstructed into a
        // built-in config safely — skip rather than risk loading mismatched
        // weights into the wrong architecture. See `ModelManifest::architecture`'s
        // docs / the module docs' "Architecture tags" table.
        if manifest.architecture != FOCUSMERGE_V1
            && manifest.architecture != BATCHMERGE_V2
            && manifest.architecture != BATCHALIGN_V2
            && manifest.architecture != FUSIONALIGN_V1
        {
            tracing::debug!(
                name = stem,
                architecture = manifest.architecture.as_str(),
                "skipping model with unknown architecture"
            );
            continue;
        }

        entries.push(ModelEntry {
            name: stem.to_owned(),
            weights_path,
            manifest,
        });
    }

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
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
/// (earlier directories win) and sorted. Intended for UI population where a
/// hard error is undesirable — an empty result simply greys out the picker.
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
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}
