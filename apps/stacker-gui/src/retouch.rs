//! The retouch state: one session over the newest result, its undo history, the
//! stroke in progress and the choice of the source frame to paint from.
//!
//! # Model
//!
//! A [`RetouchSession`] holds two layers and an alpha mask: the **base** (the fused
//! result, in `f32`) and the **source** (an aligned frame the user picked), and
//! `alpha` says per pixel how much of the source shows through. Painting raises
//! `alpha`; the composite is `base * (1 - alpha) + source * alpha`.
//!
//! # Strokes and history
//!
//! One pointer drag is one undo step: [`RetouchState::begin_stroke`]
//! on press, [`RetouchState::queue_point`] for every move, [`RetouchState::end_stroke`]
//! on release. The pointer events only append to a queue; the GUI drains it on a
//! 16 ms timer ([`RetouchState::drain_pending`]), so a burst of `moved` events
//! costs one composite, not one per event. The history is the algo crate's
//! block-diff history: a step stores only the 128-value blocks of the
//! alpha plane the stroke touched, never a full plane.
//!
//! # Changing the source frame
//!
//! The history records **alpha only**. Switching the source layer
//! ([`RetouchState::choose_source`]) therefore keeps every stroke and its undo
//! steps, and re-sources them: the painted regions now show the newly chosen
//! frame. That is what lets the user try several frames over one set of strokes,
//! and it is why undo and redo never need the source.
//!
//! The module is headless (no window), so `tests/retouch_test.rs` covers it.

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_algo::hybrid::retouch::{RetouchHistory, RetouchSession};
use stacker_core::image::PlanarImage;

use crate::{
    frames::{decode_frames, working_paths},
    image_utils::planar_to_rgba_buffer,
    retouch_preview::{PreviewFeed, Rect},
    settings::PreprocessingSettings,
    state::AlignedCache,
    viewport_map::{Frame, View},
};

/// A freshly rendered composite plus the undo and redo availability that goes
/// with it: everything the UI needs after one retouch operation.
pub type RetouchFrame = (SharedPixelBuffer<Rgba8Pixel>, bool, bool);

/// Text of the undo button's label while a stroke can be undone.
pub const UNDO_LABEL: &str = "Undo: brush stroke";
/// Text of the redo button's label while a stroke can be redone.
pub const REDO_LABEL: &str = "Redo: brush stroke";

/// Why painting is not possible right now: shown as the hint under the Retouch
/// options and as the message when the user presses on the canvas anyway.
pub const BLOCKED_NO_RESULT: &str = "Stack the images first: there is no result to retouch yet.";
/// See [`BLOCKED_NO_RESULT`].
pub const BLOCKED_NOT_NEWEST: &str = "Select the newest result in the results list to retouch it.";
/// See [`BLOCKED_NO_RESULT`].
pub const BLOCKED_NO_SOURCE: &str = "Choose a frame in the Paint from box first.";

/// Brush dabs along a dragged segment are at most this fraction of the radius
/// apart, so a fast drag leaves a continuous stroke instead of a dotted line.
const DAB_SPACING: f32 = 0.5;

/// Why a source frame could not become the source layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceError {
    /// There is no retouch session.
    NoSession,
    /// The frame is not the size of the result.
    SizeMismatch {
        /// The result's size `(width, height)`.
        expected: (usize, usize),
        /// The frame's size.
        found: (usize, usize),
    },
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSession => write!(f, "there is no result to retouch"),
            Self::SizeMismatch { expected, found } => write!(
                f,
                "the frame is {}x{} px but the result is {}x{} px (run Align, then Stack, so \
                 both use the same crop)",
                found.0, found.1, expected.0, expected.1
            ),
        }
    }
}

impl std::error::Error for SourceError {}

/// A pending change of the source layer, handed back by
/// [`RetouchState::choose_source`] when the frame still has to be fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRequest {
    /// Identifies this request; a newer choice makes it stale.
    pub generation: u64,
    /// The chosen position in [`RetouchState::sources`].
    pub index: usize,
    /// The selection before this request, restored if the load fails.
    pub previous: Option<usize>,
    /// The frame's file.
    pub path: PathBuf,
}

/// The outcome of [`RetouchState::choose_source`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceChoice {
    /// Nothing to do: same selection, no session, or an index out of range.
    Unchanged,
    /// The source layer was cleared (it is the base again); repaint.
    Cleared,
    /// The frame must be fetched, then given to [`RetouchState::finish_source`].
    Load(SourceRequest),
}

/// One brush dab waiting for the next drain.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Dab {
    x: u32,
    y: u32,
    radius: f32,
    opacity: f32,
}

/// A queued pointer event of the open stroke.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Queued {
    /// The pointer is at this raster pixel.
    Dab(Dab),
    /// The pointer left the image: the next dab must not connect to the last.
    Outside,
}

/// The point that stands for "the pointer left the image" inside
/// [`Stroke::points`]: the line is broken there, the next point starts afresh.
pub const STROKE_BREAK: (u32, u32) = (u32::MAX, u32::MAX);

/// One committed brush drag, recorded so it can be replayed onto a fresh session
/// (the project file stores these).
///
/// Strokes touch only the alpha plane, so replaying them over the same result
/// reproduces the mask exactly, whatever the source layer is. `radius` and
/// `opacity` are those of the stroke's first dab.
#[derive(Debug, Clone, PartialEq)]
pub struct Stroke {
    /// The frame the stroke was painted from (informative: the mask does not
    /// depend on it); `None` when none was chosen.
    pub source: Option<PathBuf>,
    /// The raster pixels the pointer visited, in order; [`STROKE_BREAK`] marks
    /// where it left the image.
    pub points: Vec<(u32, u32)>,
    /// Brush radius in raster pixels.
    pub radius: f32,
    /// Brush opacity in `0..=1`.
    pub opacity: f32,
}

/// The open stroke: where the last dab landed (`None` after the pointer left)
/// and what has been recorded of it so far.
#[derive(Debug, Clone)]
struct StrokeTrack {
    last: Option<(u32, u32)>,
    record: Stroke,
}

impl StrokeTrack {
    const fn new(source: Option<PathBuf>) -> Self {
        Self {
            last: None,
            record: Stroke {
                source,
                points: Vec::new(),
                radius: 0.0,
                opacity: 1.0,
            },
        }
    }

    /// Remember a queued pointer event for the project's stroke log.
    fn note(&mut self, entry: Queued) {
        match entry {
            Queued::Dab(dab) => {
                if self.record.points.is_empty() {
                    self.record.radius = dab.radius;
                    self.record.opacity = dab.opacity;
                }
                self.record.points.push((dab.x, dab.y));
            }
            Queued::Outside => {
                // A leading or repeated break paints nothing: leave it out.
                if self
                    .record
                    .points
                    .last()
                    .is_some_and(|&point| point != STROKE_BREAK)
                {
                    self.record.points.push(STROKE_BREAK);
                }
            }
        }
    }

    /// The finished record: a trailing break carries no information.
    fn finish(mut self) -> Stroke {
        if self.record.points.last() == Some(&STROKE_BREAK) {
            self.record.points.pop();
        }
        self.record
    }
}

/// All mutable retouch state. Owned by [`crate::state::AppState`]; only the UI
/// thread touches it (workers hand a finished result over instead).
pub struct RetouchState {
    /// The session over the newest result.
    pub session: Option<RetouchSession>,
    /// The undo history (block diffs of the alpha plane).
    pub history: RetouchHistory,
    /// The temporary file of the result the session belongs to.
    pub result_path: Option<PathBuf>,
    /// The frames the source picker lists, in picker order.
    sources: Vec<PathBuf>,
    /// The selected position in `sources`; `None` means no source chosen.
    source_index: Option<usize>,
    /// Bumped whenever the picker's list or selection changes meaning, so a late
    /// frame load can tell it is stale.
    generation: u64,
    /// The open stroke, if a drag is in progress.
    stroke: Option<StrokeTrack>,
    /// Pointer events not yet painted.
    queue: Vec<Queued>,
    /// The committed strokes, oldest first: what the project file stores.
    /// Follows the undo history (undo moves the newest to `undone`).
    log: Vec<Stroke>,
    /// The strokes undone and still redoable, newest undo last.
    undone: Vec<Stroke>,
    /// What the preview worker composites from (see [`crate::retouch_preview`]).
    feed: Arc<PreviewFeed>,
    /// The area painted since the feed last saw the alpha plane.
    dirty: Option<Rect>,
    /// The feed's mirror is out of date as a whole (new session, undo, redo, a
    /// different source layer) and must be replaced, not patched.
    resync: bool,
}

impl RetouchState {
    /// A state with no session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            session: None,
            history: RetouchHistory::new(),
            result_path: None,
            sources: Vec::new(),
            source_index: None,
            generation: 0,
            stroke: None,
            queue: Vec::new(),
            log: Vec::new(),
            undone: Vec::new(),
            feed: Arc::new(PreviewFeed::new()),
            dirty: None,
            resync: false,
        }
    }

    /// Replace the session with a fresh one over `session` and forget the old
    /// undo history and source choice (a new result starts a new retouch).
    pub fn start_session(&mut self, session: RetouchSession, result_path: PathBuf) {
        self.session = Some(session);
        self.history = RetouchHistory::new();
        self.result_path = Some(result_path);
        self.source_index = None;
        self.generation = self.generation.wrapping_add(1);
        self.stroke = None;
        self.queue.clear();
        self.log.clear();
        self.undone.clear();
        self.dirty = None;
        self.resync = true;
    }

    /// Bring the preview feed up to date with the session (copying only what the
    /// brush changed since the last call) and return it for the worker to take
    /// work from. Cheap: no composite and no encode happen here.
    pub fn flush_preview(&mut self) -> Arc<PreviewFeed> {
        match self.session.as_ref() {
            None => {
                self.feed.reset(None);
                self.dirty = None;
                self.resync = false;
            }
            Some(session) if self.resync => {
                self.feed.reset(Some(session));
                self.dirty = None;
                self.resync = false;
            }
            Some(session) => {
                if let Some(rect) = self.dirty.take() {
                    self.feed.patch(session, rect);
                }
            }
        }
        Arc::clone(&self.feed)
    }

    /// The committed strokes of this session, oldest first (undone ones left
    /// out): what [`Self::replay`] reproduces on a fresh session.
    #[must_use]
    pub fn strokes(&self) -> &[Stroke] {
        &self.log
    }

    /// Paint `strokes` onto the session, one undo step each, as if the user had
    /// drawn them. Returns how many changed the mask (a stroke that paints
    /// nothing records no step). The log then holds exactly `strokes` for every
    /// stroke that was applied, so a replayed session saves back unchanged.
    pub fn replay(&mut self, strokes: &[Stroke]) -> usize {
        let mut applied = 0;
        for stroke in strokes {
            if !self.begin_stroke() {
                break;
            }
            for &point in &stroke.points {
                let raster = (point != STROKE_BREAK).then_some(point);
                self.queue_point(raster, stroke.radius, stroke.opacity);
            }
            if self.end_stroke() {
                applied += 1;
                // Keep the stored points verbatim (queueing merges repeats).
                if let Some(last) = self.log.last_mut() {
                    last.clone_from(stroke);
                }
            }
        }
        applied
    }

    /// The result's size as `(width, height)`.
    #[must_use]
    pub fn dims(&self) -> Option<(u32, u32)> {
        let alpha = &self.session.as_ref()?.alpha;
        Some((
            u32::try_from(alpha.width).ok()?,
            u32::try_from(alpha.height).ok()?,
        ))
    }

    /// `true` when `viewed` is the result this session belongs to.
    #[must_use]
    pub fn is_active_for(&self, viewed: Option<&Path>) -> bool {
        self.session.is_some() && viewed.is_some() && self.result_path.as_deref() == viewed
    }

    /// Why a stroke cannot start now, or `None` when painting is possible.
    #[must_use]
    pub fn blocker(&self, viewed: Option<&Path>) -> Option<&'static str> {
        if self.session.is_none() {
            Some(BLOCKED_NO_RESULT)
        } else if !self.is_active_for(viewed) {
            Some(BLOCKED_NOT_NEWEST)
        } else if self.source_index.is_none() {
            Some(BLOCKED_NO_SOURCE)
        } else {
            None
        }
    }

    // ── Pointer mapping ──────────────────────────────────────────────────────

    /// The raster pixel under canvas position `(x, y)`, or `None` when the pointer
    /// is outside the displayed image (or there is no session).
    #[must_use]
    pub fn map_pointer(
        &self,
        canvas: (f32, f32),
        view: View,
        x: f32,
        y: f32,
    ) -> Option<(u32, u32)> {
        let (w, h) = self.dims()?;
        Frame::new(canvas.0, canvas.1, w, h).to_raster(view, x, y)
    }

    // ── Strokes ──────────────────────────────────────────────────────────────

    /// `true` while a drag is in progress.
    #[must_use]
    pub const fn stroke_open(&self) -> bool {
        self.stroke.is_some()
    }

    /// Open a stroke (pointer press). A stroke still open from an earlier press
    /// is closed first. Returns `false` when there is no session.
    pub fn begin_stroke(&mut self) -> bool {
        if self.session.is_none() {
            return false;
        }
        if self.stroke.is_some() {
            self.end_stroke();
        }
        self.history.begin_stroke();
        let source = self.source_index.and_then(|i| self.sources.get(i)).cloned();
        self.stroke = Some(StrokeTrack::new(source));
        true
    }

    /// Queue the pointer position of a press or move: `raster` is the pixel under
    /// the pointer, `None` when it is outside the image (which paints nothing and
    /// breaks the line to the next point). Ignored without an open stroke.
    pub fn queue_point(&mut self, raster: Option<(u32, u32)>, radius: f32, opacity: f32) {
        if self.stroke.is_none() {
            return;
        }
        let entry = match raster {
            Some((x, y)) => Queued::Dab(Dab {
                x,
                y,
                radius: radius.max(0.0),
                opacity: opacity.clamp(0.0, 1.0),
            }),
            None => Queued::Outside,
        };
        // Merge repeats: a pointer that reports the same pixel twice paints once.
        if self.queue.last() == Some(&entry) {
            return;
        }
        if let Some(track) = self.stroke.as_mut() {
            track.note(entry);
        }
        self.queue.push(entry);
    }

    /// Paint everything queued into the open stroke; returns the number of dabs
    /// painted (interpolated ones included). Zero means the composite is unchanged.
    pub fn drain_pending(&mut self) -> usize {
        let Self {
            session,
            history,
            stroke,
            queue,
            dirty,
            ..
        } = self;
        let (Some(session), Some(track)) = (session.as_mut(), stroke.as_mut()) else {
            queue.clear();
            return 0;
        };
        let size = (session.alpha.width, session.alpha.height);
        let mut painted = 0;
        for entry in queue.drain(..) {
            match entry {
                Queued::Outside => track.last = None,
                Queued::Dab(dab) => {
                    if track.last == Some((dab.x, dab.y)) {
                        continue;
                    }
                    painted += paint_segment(history, session, track.last, dab);
                    let touched =
                        Rect::around_segment(track.last, (dab.x, dab.y), dab.radius, size);
                    if let Some(rect) = touched {
                        *dirty = Some(dirty.map_or(rect, |pending| pending.union(rect)));
                    }
                    track.last = Some((dab.x, dab.y));
                }
            }
        }
        painted
    }

    /// Close the open stroke (pointer release): paints what is still queued and
    /// records the whole drag as one undo step. Returns `true` when a step was
    /// recorded (a stroke that painted nothing records none).
    pub fn end_stroke(&mut self) -> bool {
        self.drain_pending();
        let track = self.stroke.take();
        let recorded = match self.session.as_ref() {
            Some(session) => self.history.end_stroke(session),
            None => false,
        };
        if recorded && let Some(track) = track {
            self.log.push(track.finish());
            // A new step ends the redo branch, in the history and in the log.
            self.undone.clear();
        }
        recorded
    }

    /// Abort the open stroke and restore everything it painted.
    pub fn cancel_stroke(&mut self) {
        self.queue.clear();
        self.stroke = None;
        if let Some(session) = self.session.as_mut() {
            self.history.cancel_stroke(session);
        }
        // The restore touched blocks the dirty area does not describe.
        self.resync = true;
    }

    // ── Undo and redo ────────────────────────────────────────────────────────

    /// Undo the last stroke; `false` when there is nothing to undo or a stroke is
    /// open.
    pub fn undo(&mut self) -> bool {
        let undone = match self.session.as_mut() {
            Some(session) => self.history.undo_stroke(session),
            None => false,
        };
        if undone && let Some(stroke) = self.log.pop() {
            self.undone.push(stroke);
        }
        // The history swapped blocks the dirty area does not describe.
        self.resync |= undone;
        undone
    }

    /// Redo the stroke most recently undone.
    pub fn redo(&mut self) -> bool {
        let redone = match self.session.as_mut() {
            Some(session) => self.history.redo_stroke(session),
            None => false,
        };
        if redone && let Some(stroke) = self.undone.pop() {
            self.log.push(stroke);
        }
        self.resync |= redone;
        redone
    }

    /// `true` if [`Self::undo`] would succeed.
    #[must_use]
    pub const fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// `true` if [`Self::redo`] would succeed.
    #[must_use]
    pub const fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// What Undo would undo, for the button's label; empty when it cannot.
    #[must_use]
    pub const fn undo_label(&self) -> &'static str {
        if self.history.can_undo() {
            UNDO_LABEL
        } else {
            ""
        }
    }

    /// What Redo would redo, for the button's label; empty when it cannot.
    #[must_use]
    pub const fn redo_label(&self) -> &'static str {
        if self.history.can_redo() {
            REDO_LABEL
        } else {
            ""
        }
    }

    // ── Source frame ─────────────────────────────────────────────────────────

    /// The frames the picker lists, in picker order.
    #[must_use]
    pub fn sources(&self) -> &[PathBuf] {
        &self.sources
    }

    /// The picker's labels: the 1-based position and the file name.
    #[must_use]
    pub fn source_labels(&self) -> Vec<String> {
        self.sources
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let name = path.file_name().map_or_else(
                    || path.display().to_string(),
                    |n| n.to_string_lossy().into_owned(),
                );
                format!("{}. {name}", i + 1)
            })
            .collect()
    }

    /// The selected position in [`Self::sources`].
    #[must_use]
    pub const fn source_index(&self) -> Option<usize> {
        self.source_index
    }

    /// Rebuild the picker's list: the Align pass's frames when it ran, else the
    /// source list (result files left out). The selection follows its file; if the
    /// file is gone the source layer is cleared. Returns `true` when the list
    /// changed.
    pub fn refresh_sources(
        &mut self,
        aligned: Option<&AlignedCache>,
        file_paths: &[PathBuf],
    ) -> bool {
        let list = match aligned {
            Some(cache) if !cache.paths.is_empty() => cache.paths.clone(),
            _ => working_paths(file_paths, false, 1),
        };
        if list == self.sources {
            return false;
        }
        let selected = self.source_index.and_then(|i| self.sources.get(i)).cloned();
        self.sources = list;
        self.generation = self.generation.wrapping_add(1);
        self.source_index = selected
            .as_ref()
            .and_then(|path| self.sources.iter().position(|p| p == path));
        if selected.is_some() && self.source_index.is_none() {
            self.clear_source_layer();
        }
        true
    }

    fn clear_source_layer(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.src = Arc::clone(&session.base);
        }
        self.resync = true;
    }

    /// Select the source frame at `index` (`None` clears it).
    ///
    /// Clearing happens at once. A frame is only named here ([`SourceChoice::Load`]):
    /// the caller fetches it (from the Align cache, or by decoding the file) and
    /// hands it to [`Self::finish_source`]. The painted strokes and the undo
    /// history stay (see the module docs).
    pub fn choose_source(&mut self, index: Option<usize>) -> SourceChoice {
        if self.session.is_none() || index == self.source_index {
            return SourceChoice::Unchanged;
        }
        let previous = self.source_index;
        match index {
            None => {
                self.generation = self.generation.wrapping_add(1);
                self.source_index = None;
                self.clear_source_layer();
                SourceChoice::Cleared
            }
            Some(i) => {
                let Some(path) = self.sources.get(i).cloned() else {
                    return SourceChoice::Unchanged;
                };
                self.generation = self.generation.wrapping_add(1);
                self.source_index = Some(i);
                SourceChoice::Load(SourceRequest {
                    generation: self.generation,
                    index: i,
                    previous,
                    path,
                })
            }
        }
    }

    /// Make `frame` the source layer for `request`. `Ok(false)` means a newer
    /// choice superseded the request and the frame was dropped.
    ///
    /// # Errors
    ///
    /// [`SourceError`] when the frame cannot serve as a layer; the caller then
    /// calls [`Self::fail_source`].
    pub fn finish_source(
        &mut self,
        request: &SourceRequest,
        frame: PlanarImage<f32>,
    ) -> Result<bool, SourceError> {
        if request.generation != self.generation {
            return Ok(false);
        }
        let session = self.session.as_mut().ok_or(SourceError::NoSession)?;
        let expected = (session.base.width, session.base.height);
        let found = (frame.width, frame.height);
        if expected != found {
            return Err(SourceError::SizeMismatch { expected, found });
        }
        session.src = Arc::new(frame);
        self.resync = true;
        Ok(true)
    }

    /// Give up on `request`: the selection returns to what it was. `false` when a
    /// newer choice had already superseded the request.
    pub const fn fail_source(&mut self, request: &SourceRequest) -> bool {
        if request.generation != self.generation {
            return false;
        }
        self.source_index = request.previous;
        self.generation = self.generation.wrapping_add(1);
        true
    }

    // ── Composite and export ─────────────────────────────────────────────────

    /// The composite as an 8-bit preview with the undo/redo availability.
    #[must_use]
    pub fn render_frame(&self) -> Option<RetouchFrame> {
        let session = self.session.as_ref()?;
        let buffer = planar_to_rgba_buffer(&session.render_composite());
        Some((buffer, self.history.can_undo(), self.history.can_redo()))
    }

    /// The composite at full `f32` precision: what Save encodes, retouch included.
    #[must_use]
    pub fn composite_f32(&self) -> Option<PlanarImage<f32>> {
        self.session.as_ref().map(RetouchSession::render_composite)
    }

    /// An owned copy of the session (layers shared, alpha copied) that a worker
    /// thread can composite while the UI keeps painting.
    #[must_use]
    pub fn snapshot_session(&self) -> Option<RetouchSession> {
        let session = self.session.as_ref()?;
        Some(RetouchSession {
            base: Arc::clone(&session.base),
            src: Arc::clone(&session.src),
            alpha: session.alpha.clone(),
        })
    }

    /// [`Self::snapshot_session`] when `viewed` is the result the session belongs
    /// to, else `None` (the viewed file is then saved as a file).
    #[must_use]
    pub fn export_session(&self, viewed: Option<&Path>) -> Option<RetouchSession> {
        if self.is_active_for(viewed) {
            self.snapshot_session()
        } else {
            None
        }
    }
}

impl Default for RetouchState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode the source frame in `path` the way the Stack pass does (including the
/// pre-processing), for a source picked when no Align pass supplied it.
///
/// # Errors
///
/// The decoder's message.
pub fn decode_source(path: &Path, pre: &PreprocessingSettings) -> Result<PlanarImage<f32>, String> {
    let mut set = decode_frames(vec![path.to_path_buf()], pre).map_err(|e| e.to_string())?;
    set.frames
        .pop()
        .ok_or_else(|| "the image could not be read".to_owned())
}

/// Paint the dab at `dab`, and the dabs between the previous one (`from`) and it,
/// into the open stroke. Returns the number of dabs painted.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "raster coordinates are far below 2^24; interpolated points lie on the segment"
)]
fn paint_segment(
    history: &mut RetouchHistory,
    session: &mut RetouchSession,
    from: Option<(u32, u32)>,
    dab: Dab,
) -> usize {
    let mut painted = 0;
    if let Some((from_x, from_y)) = from {
        let (fx, fy) = (from_x as f32, from_y as f32);
        let (dx, dy) = (dab.x as f32 - fx, dab.y as f32 - fy);
        let distance = dx.hypot(dy);
        let step = (dab.radius * DAB_SPACING).max(1.0);
        let between = (distance / step).floor() as u32;
        for i in 1..=between {
            let t = i as f32 * step / distance;
            let x = (fx + dx * t).round() as usize;
            let y = (fy + dy * t).round() as usize;
            history.extend_stroke(session, x, y, dab.radius, dab.opacity);
            painted += 1;
        }
    }
    history.extend_stroke(
        session,
        dab.x as usize,
        dab.y as usize,
        dab.radius,
        dab.opacity,
    );
    painted + 1
}
