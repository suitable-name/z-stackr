use std::path::PathBuf;

use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_algo::hybrid::retouch::{RetouchHistory, RetouchSession};

use crate::image_utils::planar_to_rgba_buffer_with_overlay;

// ── Shared retouch state ──────────────────────────────────────────────────────

/// All mutable retouch state, shared between callbacks via `Arc<Mutex<>>`.
pub struct RetouchState {
    pub session: Option<RetouchSession>,
    pub history: RetouchHistory,
    pub result_path: Option<PathBuf>,
    /// "Show painted area" toggle: when set, the composite is re-rendered
    /// with a display-only tint over the brushed region (see
    /// [`crate::image_utils::planar_to_rgba_buffer_with_overlay`]). Purely a
    /// preview aid — never affects the saved/committed image, which is
    /// always re-encoded from `session.render_composite()` directly.
    pub show_painted: bool,
}

impl Default for RetouchState {
    fn default() -> Self {
        Self::new()
    }
}

impl RetouchState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            session: None,
            history: RetouchHistory::new(),
            result_path: None,
            show_painted: false,
        }
    }

    /// Render the current session's composite, tinting the painted region
    /// when `show_painted` is enabled.
    fn render_display_buffer(
        session: &RetouchSession,
        show_painted: bool,
    ) -> SharedPixelBuffer<Rgba8Pixel> {
        let composite = session.render_composite();
        let alpha = show_painted.then_some(session.alpha.luma.as_slice());
        planar_to_rgba_buffer_with_overlay(&composite, alpha)
    }

    pub fn apply_and_snapshot(
        &mut self,
        px: usize,
        py: usize,
        radius: f32,
        opacity: f32,
    ) -> Option<(SharedPixelBuffer<Rgba8Pixel>, bool, bool)> {
        let (snapshot, composite_buf) = {
            let session = self.session.as_mut()?;
            session.apply_brush(px, py, radius, opacity);
            let snapshot = session.snapshot_alpha();
            let composite_buf = Self::render_display_buffer(session, self.show_painted);
            (snapshot, composite_buf)
        };
        self.history.push(snapshot);
        let can_undo = self.history.can_undo();
        let can_redo = self.history.can_redo();
        Some((composite_buf, can_undo, can_redo))
    }

    pub fn apply_undo(&mut self) -> Option<(SharedPixelBuffer<Rgba8Pixel>, bool, bool)> {
        // Check the session *before* mutating the history cursor: if there is
        // no active session, `?`-returning here leaves `history` untouched so
        // a later real undo isn't desynced by a cursor move that was never
        // actually applied to any session.
        self.session.as_ref()?;
        let snap = self.history.undo()?.clone();
        let can_undo = self.history.can_undo();
        let can_redo = self.history.can_redo();
        let session = self.session.as_mut()?;
        session.restore_alpha(snap);
        let composite_buf = Self::render_display_buffer(session, self.show_painted);
        Some((composite_buf, can_undo, can_redo))
    }

    pub fn apply_redo(&mut self) -> Option<(SharedPixelBuffer<Rgba8Pixel>, bool, bool)> {
        // See `apply_undo`: verify the session exists before moving the
        // history cursor forward, so a `None` session can't silently desync
        // `history`'s cursor from what's actually been applied.
        self.session.as_ref()?;
        let snap = self.history.redo()?.clone();
        let can_undo = self.history.can_undo();
        let can_redo = self.history.can_redo();
        let session = self.session.as_mut()?;
        session.restore_alpha(snap);
        let composite_buf = Self::render_display_buffer(session, self.show_painted);
        Some((composite_buf, can_undo, can_redo))
    }

    /// Re-render the current composite using the given `show_painted` value,
    /// without mutating any brush/history state. Used when the "Show painted
    /// area" toggle flips, so the overlay appears/disappears live.
    pub fn set_show_painted(
        &mut self,
        show_painted: bool,
    ) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
        self.show_painted = show_painted;
        let session = self.session.as_ref()?;
        Some(Self::render_display_buffer(session, self.show_painted))
    }
}
