//! Interactive retouching of a fused result.
//!
//! A brush paints an alpha mask that blends a second image over the fused base
//! ([`RetouchSession`]), and [`RetouchHistory`] undoes and redoes the strokes by
//! saving only the mask blocks each stroke touched.

use rayon::prelude::*;
use stacker_core::image::PlanarImage;
use std::{collections::BTreeMap, sync::Arc};

/// Granularity (in `f32` values) of the dirty regions the history stores.
///
/// A stroke saves every block of this many consecutive alpha values it touches,
/// once.  A brush row of `n` pixels therefore touches at most `n / 128 + 2`
/// blocks, so a stroke's memory is proportional to the area it covers, not to
/// the size of the image.
const BLOCK: usize = 128;

// ── RetouchSession ────────────────────────────────────────────────────────────

/// An interactive alpha-compositing session.
///
/// `base` is the primary fused result; `src` is the secondary frame (or a
/// different fusion pass) that the user paints in via the brush.  `alpha` is
/// a single-channel mask stored in `luma`; chroma channels are unused.
pub struct RetouchSession {
    /// The image shown where the mask is 0.
    pub base: Arc<PlanarImage<f32>>,
    /// The image painted in where the mask is 1.
    pub src: Arc<PlanarImage<f32>>,
    /// The brush mask, one value in `[0, 1]` per pixel in the `luma` plane.
    pub alpha: PlanarImage<f32>,
}

/// Inclusive pixel rectangle `(start_x, end_x, start_y, end_y)` the brush at
/// `(x, y)` with `radius` can touch, clipped to the plane; `None` when it lies
/// entirely outside or the plane is empty.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the radius is a non-negative brush size in pixels; a NaN or negative one saturates to 0"
)]
fn brush_bounds(
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: f32,
) -> Option<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 {
        return None;
    }
    let r_int = radius.ceil() as usize;
    let start_x = x.saturating_sub(r_int);
    let end_x = x.saturating_add(r_int).min(width - 1);
    let start_y = y.saturating_sub(r_int);
    let end_y = y.saturating_add(r_int).min(height - 1);
    (start_x <= end_x && start_y <= end_y).then_some((start_x, end_x, start_y, end_y))
}

impl RetouchSession {
    /// Create a new session with a zeroed alpha mask.
    #[must_use]
    pub fn new(base: Arc<PlanarImage<f32>>, src: Arc<PlanarImage<f32>>) -> Self {
        let (w, h) = (base.width, base.height);
        Self {
            base,
            src,
            alpha: PlanarImage::new(w, h),
        }
    }

    /// Paint a hard-edged circular brush dab centred at `(x, y)`.
    ///
    /// Every pixel whose centre lies within `radius` of `(x, y)` is treated
    /// alike (there is no falloff towards the rim).  Within the disc `alpha`
    /// is raised towards 1 using the standard "paint-over" formula:
    /// `new = old + opacity * (1 − old)`.  The operation is *not* idempotent:
    /// each dab moves `alpha` further, so repeated dabs converge monotonically
    /// to 1.0 (for `opacity` in `(0, 1]`), and the result stays in `[0, 1]`.
    #[expect(
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        reason = "pixel coordinates are far below 2^24; plain multiply-add keeps the disc test and the paint-over formula identical across targets"
    )]
    pub fn apply_brush(&mut self, x: usize, y: usize, radius: f32, opacity: f32) {
        let width = self.alpha.width;
        let height = self.alpha.height;
        let r_sq = radius * radius;

        let Some((start_x, end_x, start_y, end_y)) = brush_bounds(width, height, x, y, radius)
        else {
            return;
        };

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

    /// Alpha-composite `base` and `src` using the current mask.
    ///
    /// Output pixel = `base × (1 − alpha) + src × alpha` for each channel.
    ///
    /// # Panics
    /// Panics when `base`, `src` or the alpha mask is smaller than the base
    /// image (a [`PlanarImage`] built through its constructors never is).
    #[must_use]
    #[expect(
        clippy::suboptimal_flops,
        reason = "the composite is specified as base × (1 − alpha) + src × alpha in plain multiply-add"
    )]
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

        PlanarImage::from_planes(width, height, luma, chroma_a, chroma_b)
            .expect("each plane holds width * height samples")
    }

    /// Return a copy of the whole alpha mask: the legacy full-snapshot interface
    /// that pairs with [`RetouchHistory::push`]. The desktop app uses the stroke
    /// API instead, which is proportional to the painted area, not the image.
    #[must_use]
    pub fn snapshot_alpha(&self) -> Vec<f32> {
        self.alpha.luma.clone()
    }

    /// Restore the alpha mask from a previously taken snapshot.
    pub fn restore_alpha(&mut self, snapshot: Vec<f32>) {
        self.alpha.luma = snapshot;
    }
}

// ── RetouchHistory ────────────────────────────────────────────────────────────

/// The saved values of the blocks one undo step touched.  Applying a diff
/// *swaps* its contents with the plane, so the same diff undoes and redoes.
struct StrokeDiff {
    /// `(block index, values)`, ascending by block index.  Block `b` covers
    /// plane indices `b * BLOCK .. min((b + 1) * BLOCK, plane.len())`.
    blocks: Vec<(usize, Vec<f32>)>,
}

impl StrokeDiff {
    fn swap_with(&mut self, plane: &mut [f32]) {
        for (block, data) in &mut self.blocks {
            let start = *block * BLOCK;
            if let Some(dst) = plane.get_mut(start..start + data.len()) {
                dst.swap_with_slice(data);
            }
        }
    }

    fn stored_values(&self) -> usize {
        self.blocks.iter().map(|(_, d)| d.len()).sum()
    }
}

/// Undo/redo stack for retouch brush strokes.
///
/// ## Design: dirty-region diffs, one step per stroke
///
/// One pointer drag is one undo step:
///
/// 1. [`begin_stroke`](Self::begin_stroke) on pointer press,
/// 2. [`extend_stroke`](Self::extend_stroke) for every brush dab while
///    dragging (it saves the not yet saved [`BLOCK`]-sized pieces of the alpha
///    plane the dab touches and then paints),
/// 3. [`end_stroke`](Self::end_stroke) on release, which commits the saved
///    pieces as one undo step (or discards the stroke when it changed nothing).
///
/// [`undo_stroke`](Self::undo_stroke) and [`redo_stroke`](Self::redo_stroke)
/// then swap those pieces back into the session's alpha plane.
///
/// Because the pre-stroke values are what is saved, the pristine state is
/// always recoverable: the first stroke of a session can be undone.
///
/// ## Memory
///
/// A step stores only the blocks its stroke touched.  A stroke covering
/// 100 × 100 pixels of a 2000 × 2000 plane stores roughly 25 000 values
/// instead of the 4 000 000 of a full snapshot (a full snapshot of a 32 MP
/// image is 128 MB, so 50 of them would be 6.4 GB).  The worst case is a
/// stroke dragged across the whole image, which saves the blocks it crossed
/// once.  The history is capped at [`RetouchHistory::MAX_DEPTH`] steps; when
/// the cap is hit the oldest step is discarded.
///
/// ## Redo-truncation
///
/// Committing a stroke after one or more undos **discards the redo branch**,
/// the canonical behaviour of a linear undo stack.
///
/// ## Legacy full-snapshot API
///
/// [`push`](Self::push), [`undo`](Self::undo) and [`redo`](Self::redo) keep the
/// earlier interface that exchanges whole alpha snapshots.  They are a wrapper
/// over the same diff store (a private mirror of the current plane, allocated on
/// the first `push`, is diffed against each pushed snapshot; the state before
/// the first `push` is taken to be the all-zero pristine mask).  Use either the
/// stroke API or the snapshot API on one history, not both.
pub struct RetouchHistory {
    /// Undo steps, oldest first.
    entries: Vec<StrokeDiff>,
    /// Number of steps currently applied (`0..=entries.len()`).
    cursor: usize,
    /// Blocks saved by the stroke in progress, keyed by block index.
    pending: Option<BTreeMap<usize, Vec<f32>>>,
    /// Current alpha plane, kept only by the legacy snapshot API.
    mirror: Option<Vec<f32>>,
}

impl RetouchHistory {
    /// Maximum number of undo steps retained.
    pub const MAX_DEPTH: usize = 50;

    /// Create an empty history.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            cursor: 0,
            pending: None,
            mirror: None,
        }
    }

    fn commit(&mut self, blocks: Vec<(usize, Vec<f32>)>) {
        self.entries.truncate(self.cursor);
        self.entries.push(StrokeDiff { blocks });
        if self.entries.len() > Self::MAX_DEPTH {
            self.entries.remove(0);
        }
        self.cursor = self.entries.len();
    }

    // ── Stroke API ───────────────────────────────────────────────────────────

    /// Start a stroke (pointer press).  Harmless if one is already open.
    pub fn begin_stroke(&mut self) {
        if self.pending.is_none() {
            self.pending = Some(BTreeMap::new());
        }
    }

    /// Paint one brush dab into `session` as part of the open stroke (opening
    /// one if needed), saving the plane blocks it touches first.
    pub fn extend_stroke(
        &mut self,
        session: &mut RetouchSession,
        x: usize,
        y: usize,
        radius: f32,
        opacity: f32,
    ) {
        let pending = self.pending.get_or_insert_with(BTreeMap::new);
        let width = session.alpha.width;
        let height = session.alpha.height;
        if let Some((start_x, end_x, start_y, end_y)) = brush_bounds(width, height, x, y, radius) {
            let plane = &session.alpha.luma;
            for cy in start_y..=end_y {
                let first = (cy * width + start_x) / BLOCK;
                let last = (cy * width + end_x) / BLOCK;
                for block in first..=last {
                    pending.entry(block).or_insert_with(|| {
                        let start = block * BLOCK;
                        plane[start..(start + BLOCK).min(plane.len())].to_vec()
                    });
                }
            }
        }
        session.apply_brush(x, y, radius, opacity);
    }

    /// Finish the open stroke (pointer release) and record it as one undo step.
    ///
    /// Blocks whose values did not actually change are dropped; a stroke that
    /// changed nothing records no step and leaves the redo branch alone.
    /// Returns `true` when a step was recorded.
    pub fn end_stroke(&mut self, session: &RetouchSession) -> bool {
        let Some(pending) = self.pending.take() else {
            return false;
        };
        let plane = &session.alpha.luma;
        let blocks: Vec<(usize, Vec<f32>)> = pending
            .into_iter()
            .filter(|(block, saved)| {
                let start = block * BLOCK;
                plane
                    .get(start..start + saved.len())
                    .is_some_and(|now| now != saved.as_slice())
            })
            .collect();
        if blocks.is_empty() {
            return false;
        }
        self.commit(blocks);
        true
    }

    /// Abort the open stroke, restoring everything it painted.
    pub fn cancel_stroke(&mut self, session: &mut RetouchSession) {
        if let Some(pending) = self.pending.take() {
            let mut diff = StrokeDiff {
                blocks: pending.into_iter().collect(),
            };
            diff.swap_with(&mut session.alpha.luma);
        }
    }

    /// Undo the last stroke in `session`.  Returns `false` when there is
    /// nothing to undo or a stroke is still open.
    pub fn undo_stroke(&mut self, session: &mut RetouchSession) -> bool {
        if !self.can_undo() {
            return false;
        }
        self.cursor -= 1;
        self.entries[self.cursor].swap_with(&mut session.alpha.luma);
        true
    }

    /// Redo the stroke most recently undone in `session`.  Returns `false`
    /// when there is nothing to redo or a stroke is still open.
    pub fn redo_stroke(&mut self, session: &mut RetouchSession) -> bool {
        if !self.can_redo() {
            return false;
        }
        self.entries[self.cursor].swap_with(&mut session.alpha.luma);
        self.cursor += 1;
        true
    }

    // ── Legacy full-snapshot API ─────────────────────────────────────────────

    /// Record `snapshot` (the alpha mask after a stroke) as a new step.
    ///
    /// Only the blocks that differ from the previous state are kept.  Any
    /// states ahead of the cursor (available via `redo`) are discarded; if the
    /// stack would exceed [`Self::MAX_DEPTH`] the oldest step is evicted.  The
    /// state before the first `push` is the all-zero pristine mask, so the
    /// first stroke is undoable.
    pub fn push(&mut self, snapshot: Vec<f32>) {
        let previous = match self.mirror.take() {
            Some(m) if m.len() == snapshot.len() => m,
            _ => {
                self.entries.clear();
                self.cursor = 0;
                vec![0.0; snapshot.len()]
            }
        };
        let blocks: Vec<(usize, Vec<f32>)> = previous
            .chunks(BLOCK)
            .zip(snapshot.chunks(BLOCK))
            .enumerate()
            .filter(|(_, (old, new))| old != new)
            .map(|(block, (old, _))| (block, old.to_vec()))
            .collect();
        self.commit(blocks);
        self.mirror = Some(snapshot);
    }

    /// Move one step back and return the full alpha mask to restore.
    ///
    /// Returns `None` if there is no earlier state; the pristine (all-zero)
    /// mask is returned when the first step is undone.
    pub fn undo(&mut self) -> Option<&Vec<f32>> {
        if self.cursor == 0 {
            return None;
        }
        let mirror = self.mirror.as_mut()?;
        self.cursor -= 1;
        self.entries[self.cursor].swap_with(mirror);
        self.mirror.as_ref()
    }

    /// Move one step forward and return the full alpha mask to restore.
    ///
    /// Returns `None` if there is no later state (already at the newest).
    pub fn redo(&mut self) -> Option<&Vec<f32>> {
        if self.cursor >= self.entries.len() {
            return None;
        }
        let mirror = self.mirror.as_mut()?;
        self.entries[self.cursor].swap_with(mirror);
        self.cursor += 1;
        self.mirror.as_ref()
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    /// `true` if an undo would succeed (and no stroke is open).
    #[must_use]
    pub const fn can_undo(&self) -> bool {
        self.pending.is_none() && self.cursor > 0
    }

    /// `true` if a redo would succeed (and no stroke is open).
    #[must_use]
    pub const fn can_redo(&self) -> bool {
        self.pending.is_none() && self.cursor < self.entries.len()
    }

    /// Number of undo steps currently stored.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no undo step is stored.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of `f32` values held by the stored steps and the open stroke
    /// (excluding the legacy API's mirror plane); 4 bytes each.
    pub fn stored_values(&self) -> usize {
        let steps: usize = self.entries.iter().map(StrokeDiff::stored_values).sum();
        let open: usize = self
            .pending
            .as_ref()
            .map_or(0, |p| p.values().map(Vec::len).sum());
        steps + open
    }
}

impl Default for RetouchHistory {
    fn default() -> Self {
        Self::new()
    }
}
