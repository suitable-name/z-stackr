//! The UI-thread application state.
//!
//! [`AppState`] lives in an `Rc<RefCell<..>>` inside the GUI's `Ctx`, so only the
//! UI thread can reach it. Workers never hold a borrow: they compute off-thread
//! and hand a finished value to the event loop (a [`crate::runner`] event), which
//! stores it here with one short `borrow_mut`.
//!
//! The aligned frames themselves are not kept here: the job runner holds them
//! (they are large, and in the browser they live in the worker). The state keeps
//! the record of what the Align pass produced ([`AlignedCache`]).

use std::{path::PathBuf, sync::Arc};

use stacker_algo::hybrid::retouch::RetouchSession;
use stacker_core::image::PlanarImage;

use crate::{
    retouch::RetouchState,
    settings::StackingSettings,
    stale::{
        Fingerprint, StaleReason, fingerprint_alignment_with, fingerprint_paths, fingerprint_stack,
        stale_reason,
    },
};

/// A common-coverage crop rectangle `(x, y, width, height)`.
pub type Crop = (usize, usize, usize, usize);

/// What a stacked result remembers about how it was made, parallel to
/// [`AppState::result_paths`] (see [`AppState::result_meta`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResultMeta {
    /// The inputs the result was built from; `None` when unknown (never stale).
    pub fingerprint: Option<Fingerprint>,
    /// The pass that produced it (`Apex`, `Relief`, `AI`).
    pub label: String,
    /// The settings it was built with, in one line.
    pub settings_summary: String,
}

/// What a stack run over some files can take from the last Align pass: the files
/// in the run's order and the common-area crop of that pass. The job runner holds
/// the frames; the run asks it to reuse them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReusedFrames {
    /// The files, in the run's order.
    pub paths: Vec<PathBuf>,
    /// The common-area crop of the cache the frames came from.
    pub crop: Option<Crop>,
}

/// Record of the standalone Align pass.
///
/// `paths` are the files that were aligned. The frames are held by the job
/// runner ([`crate::runner::JobRunner`]) and looked up by path, so list edits,
/// `sort_reverse`, skipped frames and filtered result files can never shift which
/// frame a sidebar click shows. `crop` is the largest rectangle valid in every
/// frame (`None` when no crop is needed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedCache {
    /// The files that were aligned.
    pub paths: Vec<PathBuf>,
    /// The rectangle valid in every frame.
    pub crop: Option<Crop>,
}

impl AlignedCache {
    /// The cache seen as the aligned frames for `wanted`, in that order, with the
    /// cache's crop. `None` when `wanted` is empty or the cache lacks any of the
    /// files.
    #[must_use]
    pub fn covering(&self, wanted: &[PathBuf]) -> Option<ReusedFrames> {
        if wanted.is_empty() || !wanted.iter().all(|path| self.paths.contains(path)) {
            return None;
        }
        Some(ReusedFrames {
            paths: wanted.to_vec(),
            crop: self.crop,
        })
    }
}

/// Everything the callbacks mutate, owned by the UI thread.
#[derive(Default)]
pub struct AppState {
    /// The source list shown in the sidebar, in display order.
    pub file_paths: Vec<PathBuf>,
    /// The stacked results of this session.
    pub result_paths: Vec<PathBuf>,
    /// The file currently on the canvas (a source or a result).
    pub viewed_path: Option<PathBuf>,
    /// Sidebar index of the source on the canvas, for the Show-Aligned toggle.
    pub view_idx: Option<usize>,
    /// What each entry of [`Self::result_paths`] was built from (parallel to it;
    /// [`Self::push_result`] keeps the two in step, a missing entry reads as
    /// "unknown, never stale").
    pub result_meta: Vec<ResultMeta>,
    /// The last Align pass, if any.
    pub aligned: Option<AlignedCache>,
    /// The alignment fingerprint of [`Self::aligned`]: the sources and settings
    /// the Align pass started with. `None` with a cache present means unknown,
    /// which counts as stale.
    pub aligned_fp: Option<Fingerprint>,
    /// The neural alignment model the session currently has picked (a name, not a
    /// setting; empty without one). It belongs to the alignment fingerprint in the
    /// neural mode.
    pub align_model: String,
    /// The source files the last auto-cull dropped. They stay in
    /// [`Self::file_paths`] (the list is the user's); the sidebar dims them.
    pub culled: Vec<PathBuf>,
    /// The retouch session of the newest result.
    pub retouch: RetouchState,
    /// The newest fused result in linear light, kept so Save can encode at full
    /// precision (16-bit output, retouch included) instead of re-reading the
    /// 8-bit preview PNG. `None` until a stack pass has finished.
    pub fused: Option<Arc<PlanarImage<f32>>>,
}

impl AppState {
    /// Append the files that are not listed yet; returns how many were added.
    pub fn add_paths(&mut self, files: impl IntoIterator<Item = PathBuf>) -> usize {
        let mut added = 0;
        for file in files {
            if !self.file_paths.contains(&file) {
                self.file_paths.push(file);
                added += 1;
            }
        }
        added
    }

    /// Remove the source at `idx`, keeping `view_idx` pointing at the same file.
    pub fn remove_path(&mut self, idx: usize) -> Option<PathBuf> {
        if idx >= self.file_paths.len() {
            return None;
        }
        let removed = self.file_paths.remove(idx);
        self.view_idx = match self.view_idx {
            Some(v) if v == idx => None,
            Some(v) if v > idx => Some(v - 1),
            other => other,
        };
        Some(removed)
    }

    /// Empty the source list.
    pub fn clear_paths(&mut self) {
        self.file_paths.clear();
        self.culled.clear();
        self.view_idx = None;
    }

    /// Replace the source list wholesale (monitor passes). The auto-cull marks of
    /// the old list do not carry over.
    pub fn replace_paths(&mut self, files: Vec<PathBuf>) {
        self.file_paths = files;
        self.culled.clear();
        self.view_idx = None;
    }

    /// Record the files the last auto-cull dropped (empty clears the marks).
    pub fn set_culled(&mut self, culled: Vec<PathBuf>) {
        self.culled = culled;
    }

    /// One flag per entry of [`Self::file_paths`]: `true` when the last auto-cull
    /// dropped that file.
    #[must_use]
    pub fn culled_flags(&self) -> Vec<bool> {
        self.file_paths
            .iter()
            .map(|path| !self.culled.is_empty() && self.culled.contains(path))
            .collect()
    }

    /// Store an Align result together with the fingerprint it was built from.
    pub fn set_aligned(&mut self, cache: AlignedCache, fingerprint: Fingerprint) {
        self.aligned = Some(cache);
        self.aligned_fp = Some(fingerprint);
    }

    /// Why the aligned cache no longer matches `sources` (the fingerprint of the
    /// current file list) and `settings`; `None` when there is no cache or it is
    /// current.
    #[must_use]
    pub fn aligned_staleness(
        &self,
        sources: &Fingerprint,
        settings: &StackingSettings,
    ) -> Option<StaleReason> {
        self.aligned.as_ref()?;
        let Some(built) = self.aligned_fp.as_ref() else {
            return Some(StaleReason::Both);
        };
        stale_reason(
            built,
            &fingerprint_alignment_with(sources, settings, &self.align_model),
        )
    }

    /// `true` when there is an aligned cache and it was built from the current
    /// file list (order, sizes, modified times) and alignment settings.
    #[must_use]
    pub fn aligned_is_current(&self, settings: &StackingSettings) -> bool {
        self.aligned.is_some()
            && self
                .aligned_staleness(&fingerprint_paths(&self.file_paths), settings)
                .is_none()
    }

    /// Why result `idx` no longer matches the current inputs; `None` when it is
    /// current, out of range, or carries no record of its inputs.
    #[must_use]
    pub fn result_staleness(
        &self,
        idx: usize,
        sources: &Fingerprint,
        settings: &StackingSettings,
    ) -> Option<StaleReason> {
        let built = self.result_meta.get(idx)?.fingerprint.as_ref()?;
        if idx >= self.result_paths.len() {
            return None;
        }
        stale_reason(built, &fingerprint_stack(sources, settings))
    }

    /// `true` when result `idx` exists and was built from the current file list
    /// and settings (a result with no record counts as current).
    #[must_use]
    pub fn result_is_current(&self, idx: usize, settings: &StackingSettings) -> bool {
        idx < self.result_paths.len()
            && self
                .result_staleness(idx, &fingerprint_paths(&self.file_paths), settings)
                .is_none()
    }

    /// The aligned frames a stack run over `wanted` can reuse: the cache's record
    /// when it is current for `sources` and `settings` and holds every file of
    /// `wanted`, else `None` (the run aligns afresh).
    #[must_use]
    pub fn reusable_frames(
        &self,
        sources: &Fingerprint,
        settings: &StackingSettings,
        wanted: &[PathBuf],
    ) -> Option<ReusedFrames> {
        let cache = self.aligned.as_ref()?;
        if self.aligned_staleness(sources, settings).is_some() {
            return None;
        }
        cache.covering(wanted)
    }

    /// Mark the source at `idx` as the one on the canvas and return its path.
    /// `view_idx` is recorded even when `idx` is out of range, so a stale click
    /// cannot leave the toggle pointing at the previous frame.
    pub fn select_source(&mut self, idx: usize) -> Option<PathBuf> {
        self.view_idx = Some(idx);
        let path = self.file_paths.get(idx).cloned()?;
        self.viewed_path = Some(path.clone());
        Some(path)
    }

    /// Mark the result at `idx` as the one on the canvas and return its path.
    pub fn select_result(&mut self, idx: usize) -> Option<PathBuf> {
        let path = self.result_paths.get(idx).cloned()?;
        self.viewed_path = Some(path.clone());
        Some(path)
    }

    /// List a finished result and put it on the canvas.
    pub fn push_result(&mut self, path: PathBuf) {
        self.push_result_with(path, ResultMeta::default());
    }

    /// [`Self::push_result`] with the record of how the result was made.
    pub fn push_result_with(&mut self, path: PathBuf, meta: ResultMeta) {
        // Keep the parallel vector aligned even if a caller pushed a bare path.
        self.result_meta
            .resize_with(self.result_paths.len(), ResultMeta::default);
        self.result_paths.push(path.clone());
        self.result_meta.push(meta);
        self.viewed_path = Some(path);
    }

    /// `true` when the canvas shows the newest result, the one the retouch
    /// session belongs to (the only image the brush may paint on).
    #[must_use]
    pub fn viewing_retouchable(&self) -> bool {
        self.retouch.is_active_for(self.viewed_path.as_deref())
    }

    /// Why a brush stroke cannot start now (see
    /// [`RetouchState::blocker`]), or `None` when painting is possible.
    #[must_use]
    pub fn retouch_blocker(&self) -> Option<&'static str> {
        self.retouch.blocker(self.viewed_path.as_deref())
    }

    /// Rebuild the retouch source picker's list from the Align cache and the
    /// source list; `true` when it changed.
    pub fn refresh_retouch_sources(&mut self) -> bool {
        self.retouch
            .refresh_sources(self.aligned.as_ref(), &self.file_paths)
    }

    /// Make the retouch session match the newest fused result.
    ///
    /// When [`Self::fused`] holds a result that the session is not built on (no
    /// session yet, or one over an older pass), a fresh session over it starts,
    /// with the newest result file as its image. Returns `true` when it did.
    /// A session that already sits on the fused result is left alone, and so is
    /// `fused` when it is `None`: a session started elsewhere fills it.
    pub fn ensure_retouch_session(&mut self) -> bool {
        let Some(fused) = self.fused.as_ref() else {
            if let Some(session) = self.retouch.session.as_ref() {
                self.fused = Some(Arc::clone(&session.base));
            }
            return false;
        };
        let up_to_date = self
            .retouch
            .session
            .as_ref()
            .is_some_and(|session| Arc::ptr_eq(&session.base, fused));
        if up_to_date {
            return false;
        }
        let Some(result) = self.result_paths.last().cloned() else {
            return false;
        };
        let session = RetouchSession::new(Arc::clone(fused), Arc::clone(fused));
        self.retouch.start_session(session, result);
        true
    }

    /// What Save writes for the image on the canvas: the full-precision composite
    /// of the newest result (retouch included) when that is what is shown, else
    /// the viewed file as it is. `None` when nothing is shown.
    #[must_use]
    pub fn save_target(&self) -> Option<SaveTarget> {
        let viewed = self.viewed_path.as_ref()?;
        if let Some(session) = self.retouch.export_session(Some(viewed.as_path())) {
            return Some(SaveTarget::Result(session));
        }
        if let Some(fused) = self.fused.as_ref()
            && self.result_paths.last() == Some(viewed)
        {
            let session = RetouchSession::new(Arc::clone(fused), Arc::clone(fused));
            return Some(SaveTarget::Result(session));
        }
        Some(SaveTarget::File(viewed.clone()))
    }
}

/// What Save encodes (see [`AppState::save_target`]).
pub enum SaveTarget {
    /// The newest result: composite the session at `f32` and encode that.
    Result(RetouchSession),
    /// Any other image (a source frame, an older result): re-encode the file.
    File(PathBuf),
}
