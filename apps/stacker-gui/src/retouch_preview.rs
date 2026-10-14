//! The retouch preview without the UI thread.
//!
//! Compositing the base and source layers and encoding the result as 8-bit sRGB
//! touches every pixel of the result. Done on the UI thread once per 16 ms brush
//! tick that stalls the window on a large image. This module splits the job in two:
//!
//! * the UI thread keeps a [`PreviewFeed`]: a mirror of the alpha plane plus the
//!   two layers (shared, never copied). After a tick it copies only the rows of
//!   the rectangle the brush touched ([`PreviewFeed::patch`]); after an undo, a
//!   redo or a source change it replaces the mirror ([`PreviewFeed::reset`]);
//! * a worker thread asks the feed for work ([`PreviewFeed::take`]) and gives it to
//!   its [`Compositor`], which keeps the last 8-bit frame and re-composites only
//!   the dirty rectangle into it.
//!
//! Work that piles up while the worker is busy is merged, not queued: the dirty
//! rectangles are united, so the worker always catches up in one step and never
//! shows an out-of-date mix of strokes. A reset replaces everything pending.
//!
//! Everything here is headless (no window), so `tests/retouch_preview_test.rs`
//! covers it.

use std::sync::{Arc, Mutex};

use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_algo::hybrid::retouch::RetouchSession;
use stacker_core::{image::PlanarImage, memory::extract_tile};

use crate::{image_utils::planar_to_rgba_buffer, sync::lock};

/// A pixel rectangle: columns `x0..x1`, rows `y0..y1` (the upper bounds are
/// exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// First column.
    pub x0: usize,
    /// First row.
    pub y0: usize,
    /// One past the last column.
    pub x1: usize,
    /// One past the last row.
    pub y1: usize,
}

/// A `u32` pixel coordinate as an index.
fn index(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

impl Rect {
    /// The number of columns.
    #[must_use]
    pub const fn width(self) -> usize {
        self.x1 - self.x0
    }

    /// The number of rows.
    #[must_use]
    pub const fn height(self) -> usize {
        self.y1 - self.y0
    }

    /// The smallest rectangle holding both.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    /// This rectangle cut to a `width` x `height` frame; `None` when nothing of it
    /// is left.
    #[must_use]
    pub fn clipped(self, width: usize, height: usize) -> Option<Self> {
        let clipped = Self {
            x1: self.x1.min(width),
            y1: self.y1.min(height),
            ..self
        };
        (clipped.x0 < clipped.x1 && clipped.y0 < clipped.y1).then_some(clipped)
    }

    /// The area a brush of `radius` can change when it paints at `to`, and on the
    /// way there from `from` (the previous dab of the stroke), in a frame of
    /// `size` = `(width, height)`. The padding is generous (a pixel more than the
    /// radius), never short. `None` when the area lies outside the frame.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the radius is a brush size in pixels; a negative or NaN one saturates to 0"
    )]
    pub fn around_segment(
        from: Option<(u32, u32)>,
        to: (u32, u32),
        radius: f32,
        size: (usize, usize),
    ) -> Option<Self> {
        let pad = (radius.ceil() as usize).saturating_add(1);
        let (from_x, from_y) = from.unwrap_or(to);
        let (left, right) = (from_x.min(to.0), from_x.max(to.0));
        let (top, bottom) = (from_y.min(to.1), from_y.max(to.1));
        Self {
            x0: index(left).saturating_sub(pad),
            y0: index(top).saturating_sub(pad),
            x1: index(right).saturating_add(pad).saturating_add(1),
            y1: index(bottom).saturating_add(pad).saturating_add(1),
        }
        .clipped(size.0, size.1)
    }
}

/// The two layers and a copy of the alpha plane.
struct Layers {
    base: Arc<PlanarImage<f32>>,
    src: Arc<PlanarImage<f32>>,
    /// The alpha plane (`width * height` values), or in a patch only the rows of
    /// the patch rectangle, packed.
    alpha: Vec<f32>,
    /// The size of the whole frame.
    width: usize,
    height: usize,
}

impl Layers {
    /// Everything: the layers are shared, the whole alpha plane is copied.
    fn copy_all(&self) -> Self {
        Self {
            base: Arc::clone(&self.base),
            src: Arc::clone(&self.src),
            alpha: self.alpha.clone(),
            width: self.width,
            height: self.height,
        }
    }

    /// The layers (shared) and a copy of the alpha rows of `rect`, packed.
    fn copy_rows(&self, rect: Rect) -> Self {
        let mut rows = Vec::with_capacity(rect.width() * rect.height());
        for y in rect.y0..rect.y1 {
            let start = y * self.width + rect.x0;
            rows.extend_from_slice(&self.alpha[start..start + rect.width()]);
        }
        Self {
            base: Arc::clone(&self.base),
            src: Arc::clone(&self.src),
            alpha: rows,
            width: self.width,
            height: self.height,
        }
    }
}

/// A [`RetouchSession`] over `base` and `src` whose alpha plane is `alpha`. The
/// composite reads only the first plane of the mask, so the other two stay empty.
const fn session_over(
    base: Arc<PlanarImage<f32>>,
    src: Arc<PlanarImage<f32>>,
    alpha: Vec<f32>,
    (width, height): (usize, usize),
) -> RetouchSession {
    RetouchSession {
        base,
        src,
        alpha: PlanarImage {
            width,
            height,
            luma: alpha,
            chroma_a: Vec::new(),
            chroma_b: Vec::new(),
        },
    }
}

struct Inner {
    layers: Option<Layers>,
    /// What the mirror changed in since the worker last took work.
    dirty: Option<Rect>,
    /// The worker must composite the whole frame.
    full: bool,
}

enum Job {
    Full(Layers),
    Patch { rect: Rect, layers: Layers },
}

/// One unit of work for the [`Compositor`], handed out by [`PreviewFeed::take`].
pub struct Work(Job);

/// The session's layers and alpha plane as the preview worker sees them. Shared
/// between the UI thread (writer) and the worker (reader).
pub struct PreviewFeed {
    inner: Mutex<Inner>,
}

impl PreviewFeed {
    /// A feed with no session: [`Self::take`] has nothing to hand out.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                layers: None,
                dirty: None,
                full: false,
            }),
        }
    }

    /// Replace the mirror with `session` (`None`: no session). The next work is a
    /// full composite, and everything that was pending is dropped.
    pub fn reset(&self, session: Option<&RetouchSession>) {
        let mut inner = lock(&self.inner);
        inner.dirty = None;
        inner.full = session.is_some();
        inner.layers = session.map(|session| Layers {
            base: Arc::clone(&session.base),
            src: Arc::clone(&session.src),
            alpha: session.alpha.luma.clone(),
            width: session.alpha.width,
            height: session.alpha.height,
        });
    }

    /// Copy the rows of `rect` from the session's alpha plane into the mirror and
    /// remember that the worker has to redo that area. Ignored when the session's
    /// size is not the mirror's (a [`Self::reset`] is due first).
    pub fn patch(&self, session: &RetouchSession, rect: Rect) {
        let mut inner = lock(&self.inner);
        let Some(layers) = inner.layers.as_mut() else {
            return;
        };
        let (width, height) = (layers.width, layers.height);
        if session.alpha.width != width || session.alpha.height != height {
            return;
        }
        let Some(rect) = rect.clipped(width, height) else {
            return;
        };
        for y in rect.y0..rect.y1 {
            let start = y * width + rect.x0;
            let end = start + rect.width();
            layers.alpha[start..end].copy_from_slice(&session.alpha.luma[start..end]);
        }
        inner.dirty = Some(inner.dirty.map_or(rect, |pending| pending.union(rect)));
    }

    /// The next work for the compositor: the whole frame after a reset, else the
    /// united dirty rectangle; `None` when nothing changed since the last call.
    #[must_use]
    pub fn take(&self) -> Option<Work> {
        let mut inner = lock(&self.inner);
        inner.layers.as_ref()?;
        if std::mem::take(&mut inner.full) {
            inner.dirty = None;
            return inner
                .layers
                .as_ref()
                .map(|layers| Work(Job::Full(layers.copy_all())));
        }
        let rect = inner.dirty.take()?;
        inner.layers.as_ref().map(|layers| {
            Work(Job::Patch {
                rect,
                layers: layers.copy_rows(rect),
            })
        })
    }
}

impl Default for PreviewFeed {
    fn default() -> Self {
        Self::new()
    }
}

/// Turns [`Work`] into 8-bit frames; owned by the worker thread. It keeps the last
/// frame, so a patch rewrites only its rectangle of it.
#[derive(Default)]
pub struct Compositor {
    frame: Option<SharedPixelBuffer<Rgba8Pixel>>,
}

impl Compositor {
    /// Carry out `work` and return the frame to show. `None` when a patch has no
    /// frame to patch (the size changed without a full composite in between),
    /// which a reset prevents.
    pub fn apply(&mut self, work: Work) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
        match work.0 {
            Job::Full(layers) => {
                let size = (layers.width, layers.height);
                let session = session_over(layers.base, layers.src, layers.alpha, size);
                let frame = planar_to_rgba_buffer(&session.render_composite());
                self.frame = Some(frame.clone());
                Some(frame)
            }
            Job::Patch { rect, layers } => self.patch(rect, layers),
        }
    }

    fn patch(&mut self, rect: Rect, layers: Layers) -> Option<SharedPixelBuffer<Rgba8Pixel>> {
        let frame = self.frame.as_mut()?;
        let stride = usize::try_from(frame.width()).ok()?;
        if stride != layers.width || usize::try_from(frame.height()).ok()? != layers.height {
            return None;
        }
        let (width, height) = (rect.width(), rect.height());
        if width == 0 || height == 0 {
            return None;
        }
        let tile = session_over(
            Arc::new(extract_tile(&layers.base, rect.x0, rect.y0, width, height)),
            Arc::new(extract_tile(&layers.src, rect.x0, rect.y0, width, height)),
            layers.alpha,
            (width, height),
        );
        let tile = planar_to_rgba_buffer(&tile.render_composite());
        let pixels = frame.make_mut_slice();
        for (row, tile_row) in tile.as_slice().chunks_exact(width).enumerate() {
            let start = (rect.y0 + row) * stride + rect.x0;
            pixels[start..start + width].copy_from_slice(tile_row);
        }
        Some(frame.clone())
    }
}
