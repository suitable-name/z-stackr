use rayon::prelude::*;
use stacker_core::image::PlanarImage;
use std::sync::Arc;

// ── RetouchSession ────────────────────────────────────────────────────────────

/// An interactive alpha-compositing session.
///
/// `base` is the primary fused result; `src` is the secondary frame (or a
/// different fusion pass) that the user paints in via the brush.  `alpha` is
/// a single-channel mask stored in `luma`; chroma channels are unused.
pub struct RetouchSession {
    pub base: Arc<PlanarImage<f32>>,
    pub src: Arc<PlanarImage<f32>>,
    pub alpha: PlanarImage<f32>,
}

impl RetouchSession {
    /// Create a new session with a zeroed alpha mask.
    pub fn new(base: Arc<PlanarImage<f32>>, src: Arc<PlanarImage<f32>>) -> Self {
        let (w, h) = (base.width, base.height);
        Self {
            base,
            src,
            alpha: PlanarImage::new(w, h),
        }
    }

    /// Paint a soft circular brush stroke centred at `(x, y)`.
    ///
    /// Within the circle `alpha` is raised towards `opacity` using the
    /// standard "paint-over" formula: `new = old + opacity * (1 − old)`.
    /// This is idempotent (repeated strokes converge to 1.0) and ensures the
    /// result stays in `[0, 1]`.
    pub fn apply_brush(&mut self, x: usize, y: usize, radius: f32, opacity: f32) {
        let width = self.alpha.width;
        let height = self.alpha.height;
        let r_sq = radius * radius;

        let r_int = radius.ceil() as usize;
        let start_x = x.saturating_sub(r_int);
        let end_x = (x + r_int).min(width.saturating_sub(1));
        let start_y = y.saturating_sub(r_int);
        let end_y = (y + r_int).min(height.saturating_sub(1));

        for cy in start_y..=end_y {
            for cx in start_x..=end_x {
                let dx = cx as f32 - x as f32;

                let dy = cy as f32 - y as f32;
                if dx * dx + dy * dy <= r_sq {
                    let idx = cy * width + cx;
                    let current = self.alpha.luma[idx];
                    self.alpha.luma[idx] = (current + opacity * (1.0 - current)).clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Swap the donor (`src`) image without touching `base` or the current
    /// `alpha` mask/history.
    ///
    /// Lets the brush heal from a different image than the one the session
    /// was created with — either another completed fusion pass's result, or
    /// a raw (aligned, pre-fusion) source frame — without losing in-progress
    /// brush strokes or undo/redo state. `new_src` must have the exact same
    /// `width`/`height` as `base`; callers are responsible for cropping/
    /// resizing beforehand (see `stacker_align::transform::resize_planar_clamped`)
    /// since this type has no alignment/resampling logic of its own.
    ///
    /// # Panics
    ///
    /// Does not panic itself, but a subsequent [`Self::render_composite`] or
    /// [`Self::apply_brush`] call will panic/index out of bounds if
    /// `new_src`'s dimensions don't match `base`'s — this is a caller
    /// contract, not something this method can safely enforce without a
    /// `Result` return that every call site would then have to handle.
    pub fn set_src(&mut self, new_src: Arc<PlanarImage<f32>>) {
        self.src = new_src;
    }

    /// Alpha-composite `base` and `src` using the current mask.
    ///
    /// Output pixel = `base × (1 − alpha) + src × alpha` for each channel.
    pub fn render_composite(&self) -> PlanarImage<f32> {
        let width = self.base.width;
        let height = self.base.height;
        let len = width * height;

        let mut luma = vec![0.0_f32; len];
        let mut chroma_a = vec![0.0_f32; len];
        let mut chroma_b = vec![0.0_f32; len];

        luma.par_iter_mut().enumerate().for_each(|(i, out)| {
            let a = self.alpha.luma[i];
            *out = self.base.luma[i] * (1.0 - a) + self.src.luma[i] * a;
        });
        chroma_a.par_iter_mut().enumerate().for_each(|(i, out)| {
            let a = self.alpha.luma[i];
            *out = self.base.chroma_a[i] * (1.0 - a) + self.src.chroma_a[i] * a;
        });
        chroma_b.par_iter_mut().enumerate().for_each(|(i, out)| {
            let a = self.alpha.luma[i];
            *out = self.base.chroma_b[i] * (1.0 - a) + self.src.chroma_b[i] * a;
        });

        PlanarImage {
            width,
            height,
            luma,
            chroma_a,
            chroma_b,
        }
    }

    /// Return a snapshot of the current alpha mask (used by `RetouchHistory`).
    pub fn snapshot_alpha(&self) -> Vec<f32> {
        self.alpha.luma.clone()
    }

    /// Restore the alpha mask from a previously taken snapshot.
    pub fn restore_alpha(&mut self, snapshot: Vec<f32>) {
        self.alpha.luma = snapshot;
    }
}

// ── RetouchHistory ────────────────────────────────────────────────────────────

/// Undo/redo stack for retouch brush strokes.
///
/// ## Design: snapshot-per-stroke
///
/// After every completed brush stroke the caller must call
/// [`RetouchHistory::push`] with the new alpha mask.  Each entry in the
/// history is a full copy of the alpha channel (a `Vec<f32>` of `width ×
/// height` floats — roughly 4 MB for a 32 MP image, negligible for smaller
/// previews).
///
/// This approach is simple to reason about and correct: undo/redo always
/// restores an exact state rather than replaying inverse operations.
///
/// ## Memory bound
///
/// The history is capped at [`RetouchHistory::MAX_DEPTH`] entries.  When the
/// cap is hit the oldest entry is discarded (FIFO eviction at the bottom of the
/// stack).
///
/// ## Redo-truncation
///
/// Calling `push` after one or more undos **discards the redo branch** — the
/// canonical behaviour of a linear undo stack (identical to most editors).
///
/// ## Indices
///
/// ```text
/// stack:    [s0, s1, s2, s3]
/// cursor:                 ^--- points to the currently applied snapshot
/// ```
///
/// `undo` decrements the cursor (restores `s2`).\
/// `redo` increments it (restores `s3`).
/// `push` appends after the cursor and truncates anything that was ahead of it.
pub struct RetouchHistory {
    /// Circular-bounded list of alpha-mask snapshots.
    stack: Vec<Vec<f32>>,
    /// Index into `stack` of the *currently applied* state.
    /// `None` means the session has not yet recorded any stroke.
    cursor: Option<usize>,
}

impl RetouchHistory {
    /// Maximum number of undo steps retained.
    pub const MAX_DEPTH: usize = 50;

    /// Create an empty history.
    pub const fn new() -> Self {
        Self {
            stack: Vec::new(),
            cursor: None,
        }
    }

    /// Record `snapshot` as the new head of the history.
    ///
    /// Any states that were ahead of the current cursor (i.e. available via
    /// `redo`) are discarded.  If the resulting stack would exceed
    /// [`Self::MAX_DEPTH`] the oldest entry is evicted.
    pub fn push(&mut self, snapshot: Vec<f32>) {
        // Truncate redo branch.
        if let Some(c) = self.cursor {
            self.stack.truncate(c + 1);
        } else {
            self.stack.clear();
        }

        self.stack.push(snapshot);

        // Evict oldest if over capacity.
        if self.stack.len() > Self::MAX_DEPTH {
            self.stack.remove(0);
        }

        self.cursor = Some(self.stack.len() - 1);
    }

    /// Move the cursor one step back and return the snapshot to restore.
    ///
    /// Returns `None` if there is no earlier state (already at the oldest).
    pub fn undo(&mut self) -> Option<&Vec<f32>> {
        let c = self.cursor?;
        if c == 0 {
            return None;
        }
        self.cursor = Some(c - 1);
        self.stack.get(c - 1)
    }

    /// Move the cursor one step forward and return the snapshot to restore.
    ///
    /// Returns `None` if there is no later state (already at the newest).
    pub fn redo(&mut self) -> Option<&Vec<f32>> {
        let c = self.cursor?;
        let next = c + 1;
        if next >= self.stack.len() {
            return None;
        }
        self.cursor = Some(next);
        self.stack.get(next)
    }

    /// `true` if `undo` would succeed.
    pub fn can_undo(&self) -> bool {
        self.cursor.is_some_and(|c| c > 0)
    }

    /// `true` if `redo` would succeed.
    pub fn can_redo(&self) -> bool {
        self.cursor.is_some_and(|c| c + 1 < self.stack.len())
    }

    /// Number of entries currently in the stack.
    pub const fn len(&self) -> usize {
        self.stack.len()
    }

    /// `true` when the stack has no entries.
    pub const fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Export the entire undo/redo stack and cursor position, for callers
    /// that need to persist the *full* history (every intermediate stroke,
    /// not just the currently-applied state — contrast with
    /// [`RetouchSession::snapshot_alpha`]). Used by project-file save when
    /// the user opts in to preserving rebrush history across a save/reload.
    #[must_use]
    pub fn export_full(&self) -> (Vec<Vec<f32>>, Option<usize>) {
        (self.stack.clone(), self.cursor)
    }

    /// Restore a full stack + cursor previously produced by
    /// [`Self::export_full`]. Replaces the current history outright — this
    /// is meant for loading a freshly-opened project, not for merging with
    /// in-progress state.
    pub fn restore_full(&mut self, stack: Vec<Vec<f32>>, cursor: Option<usize>) {
        self.stack = stack;
        self.cursor = cursor;
    }
}

impl Default for RetouchHistory {
    fn default() -> Self {
        Self::new()
    }
}
