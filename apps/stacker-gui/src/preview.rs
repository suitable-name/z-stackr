//! Display-side helpers for previews: everything a worker needs to put a small,
//! cheap picture in front of the user without stalling either side.
//!
//! * [`PreviewThrottle`] rate-limits live previews (about 2 per second) so a pass
//!   that finishes a unit of work every few milliseconds does not build a
//!   thumbnail for each.
//! * [`downsample_to_long_edge`] / [`downsample_into`] box-average a linear-light
//!   planar frame down to a thumbnail and encode it with the shared SIMD row
//!   encoder; the full frame is never converted.
//! * [`FrameRing`] hands the finished thumbnails to Slint through three rotating
//!   buffers, so the target is unshared when it is rewritten (Slint copies a
//!   shared buffer the moment it is written).
//! * [`Serial`] marks requests so a result that was superseded while it ran is
//!   dropped.
//! * [`FocusThreshold`] and [`tint_rejected`] are the `Relief` preview's maths: an
//!   exact contrast threshold as an index lookup, and the mask overlay.
//! * [`decode_file`] decodes an image file into a display buffer (for a worker
//!   thread; the UI thread never decodes); [`decode_bytes`] does the same for a
//!   file the host holds in memory.
//!
//! No Slint window type appears here, only `SharedPixelBuffer`, so all of it is
//! unit-testable.

use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use rayon::slice::ParallelSliceMut;
use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_core::{
    color::{encode_row_u8, planar_to_linear_rgb},
    image::PlanarImage,
};
use web_time::Instant;

/// The long edge of a live preview thumbnail, in pixels. Large enough to read the
/// composition, small enough that a pass over it costs a bounded fraction of the
/// work being previewed.
pub const THUMB_LONG_EDGE: u32 = 360;

/// The shortest time between two live preview thumbnails.
pub const THUMB_MIN_INTERVAL: Duration = Duration::from_millis(500);

/// A region of a frame: `(x, y, width, height)` in pixels.
pub type Region = (usize, usize, usize, usize);

// ── Throttle ─────────────────────────────────────────────────────────────────

/// A source of "now" for [`PreviewThrottle`], abstracted so a test can drive the
/// throttle with pre-scheduled instants instead of racing the real clock.
pub trait Clock {
    /// The current instant.
    fn now(&mut self) -> Instant;
}

/// The wall clock: [`Instant::now`].
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&mut self) -> Instant {
        Instant::now()
    }
}

/// Lets a preview through at most once per interval; the first call always
/// passes. One instance lives for the whole pass, not per unit of work.
#[derive(Debug)]
pub struct PreviewThrottle<C: Clock = SystemClock> {
    last: Option<Instant>,
    interval: Duration,
    clock: C,
}

impl PreviewThrottle {
    /// A throttle on the wall clock with the standard [`THUMB_MIN_INTERVAL`].
    #[must_use]
    pub const fn new() -> Self {
        Self::with_clock(SystemClock, THUMB_MIN_INTERVAL)
    }
}

impl Default for PreviewThrottle {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Clock> PreviewThrottle<C> {
    /// A throttle reading `clock`, letting one preview through per `interval`.
    #[must_use]
    pub const fn with_clock(clock: C, interval: Duration) -> Self {
        Self {
            last: None,
            interval,
            clock,
        }
    }

    /// `true` when a preview may be built now (and starts a new interval);
    /// `false` while the previous one is still too recent.
    pub fn ready(&mut self) -> bool {
        let now = self.clock.now();
        if self
            .last
            .is_some_and(|last| now.duration_since(last) < self.interval)
        {
            return false;
        }
        self.last = Some(now);
        true
    }

    /// Forget the last preview, so the next [`Self::ready`] passes.
    pub const fn reset(&mut self) {
        self.last = None;
    }
}

// ── Request serial ───────────────────────────────────────────────────────────

/// A counter that numbers requests so only the newest result is used.
///
/// The requester calls [`Self::begin`] per request and sends the token with it;
/// the worker checks [`Self::is_current`] before it starts and again before it
/// publishes, and drops the result when a newer request has begun meanwhile.
#[derive(Debug, Default)]
pub struct Serial(AtomicU64);

impl Serial {
    /// A counter that has issued no token yet.
    #[must_use]
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Start a request: the token of the newest request from now on.
    pub fn begin(&self) -> u64 {
        self.0.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Whether `token` is still the newest request's token.
    #[must_use]
    pub fn is_current(&self, token: u64) -> bool {
        self.0.load(Ordering::Acquire) == token
    }
}

// ── Rotating buffers ─────────────────────────────────────────────────────────

/// How many frame buffers take turns: the one on screen, the one a pending
/// redraw still holds, and the one being written.
const RING_LEN: usize = 3;

/// Hands finished RGBA8 frames to Slint, rewriting a rotating set of buffers in
/// place instead of allocating one per frame.
///
/// The returned [`SharedPixelBuffer`] shares the ring buffer's storage (a
/// reference-count bump, not a copy). By the time the rotation returns to a
/// buffer the UI has normally replaced it, so the write finds it unshared. If the
/// UI still holds it, Slint copies on write: slower, but the held frame is never
/// altered. A buffer whose size no longer fits is replaced.
#[derive(Default)]
pub struct FrameRing {
    buffers: [Option<SharedPixelBuffer<Rgba8Pixel>>; RING_LEN],
    next: usize,
}

impl FrameRing {
    /// An empty ring; nothing is allocated until the first frame.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffers: [const { None }; RING_LEN],
            next: 0,
        }
    }

    /// Let `fill` write the next `width` x `height` frame straight into the
    /// buffer that is handed out. The buffer still holds the frame from its last
    /// use until `fill` overwrites it.
    pub fn fill_with(
        &mut self,
        width: u32,
        height: u32,
        fill: impl FnOnce(&mut [Rgba8Pixel]),
    ) -> SharedPixelBuffer<Rgba8Pixel> {
        let index = self.next;
        self.next = (index + 1) % RING_LEN;
        let slot = &mut self.buffers[index];
        if !slot
            .as_ref()
            .is_some_and(|buffer| buffer.width() == width && buffer.height() == height)
        {
            *slot = None;
        }
        let buffer = slot.get_or_insert_with(|| SharedPixelBuffer::new(width, height));
        fill(buffer.make_mut_slice());
        buffer.clone()
    }
}

// ── Downsampling ─────────────────────────────────────────────────────────────

/// `value` as a `u32`, saturating.
fn to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// The size of a `width` x `height` image scaled so its long edge is at most
/// `max_edge` (never upscaled; each side is at least 1 once scaled).
#[must_use]
pub fn thumbnail_size(width: u32, height: u32, max_edge: u32) -> (u32, u32) {
    let max_edge = max_edge.max(1);
    let long = width.max(height);
    if long <= max_edge {
        return (width, height);
    }
    let scale = |side: u32| {
        let rounded =
            (u64::from(side) * u64::from(max_edge) + u64::from(long) / 2) / u64::from(long);
        u32::try_from(rounded).unwrap_or(max_edge).max(1)
    };
    (scale(width), scale(height))
}

/// `region` (or the whole frame) clipped to a `width` x `height` frame.
fn clamp_region(width: usize, height: usize, region: Option<Region>) -> Region {
    let (x, y, w, h) = region.unwrap_or((0, 0, width, height));
    let x = x.min(width);
    let y = y.min(height);
    (x, y, w.min(width - x), h.min(height - y))
}

/// Box-average the `N` planes (row stride `stride`) over `region` into
/// `dst_w` x `dst_h` buckets: every source pixel lands in exactly one bucket and
/// each bucket holds the mean of its pixels. `dst_w`/`dst_h` must not exceed the
/// region's size (no upscaling), which makes every bucket non-empty.
fn box_average<const N: usize>(
    planes: [&[f32]; N],
    stride: usize,
    region: Region,
    (dst_w, dst_h): (usize, usize),
) -> [Vec<f32>; N] {
    let (rx, ry, rw, rh) = region;
    let cols: Vec<usize> = (0..rw).map(|x| x * dst_w / rw).collect();
    let mut sums: [Vec<f32>; N] = std::array::from_fn(|_| vec![0.0; dst_w * dst_h]);
    let mut counts = vec![0.0_f32; dst_w * dst_h];
    for y in 0..rh {
        let dst_row = (y * dst_h / rh) * dst_w;
        let src_row = (ry + y) * stride + rx;
        for &tx in &cols {
            counts[dst_row + tx] += 1.0;
        }
        for (plane, sum) in planes.iter().zip(sums.iter_mut()) {
            let source = &plane[src_row..src_row + rw];
            for (&value, &tx) in source.iter().zip(&cols) {
                sum[dst_row + tx] += value;
            }
        }
    }
    for sum in &mut sums {
        for (value, &count) in sum.iter_mut().zip(&counts) {
            *value /= count.max(1.0);
        }
    }
    sums
}

/// Box-downsample one plane (`width` x `height`, row-major) so its long edge is
/// at most `max_edge`. Returns the thumbnail size and the averaged values; the
/// size equals [`thumbnail_size`] and the frame thumbnails'.
#[must_use]
pub fn downsample_plane(
    plane: &[f32],
    width: usize,
    height: usize,
    max_edge: u32,
) -> (u32, u32, Vec<f32>) {
    if width == 0 || height == 0 || plane.len() < width * height {
        return (0, 0, Vec::new());
    }
    let (tw, th) = thumbnail_size(to_u32(width), to_u32(height), max_edge);
    let [values] = box_average(
        [plane],
        width,
        (0, 0, width, height),
        (tw as usize, th as usize),
    );
    (tw, th, values)
}

/// Encode a `width`-wide linear YCbCr thumbnail (the three planes hold one
/// value per pixel) into `out` as sRGB RGBA8 with the SIMD row encoder.
fn encode_thumbnail(planes: &[Vec<f32>; 3], width: usize, out: &mut [Rgba8Pixel]) {
    let mut linear = vec![0.0_f32; width * 3];
    let mut encoded = vec![0_u8; width * 3];
    for (row, px_row) in out.chunks_exact_mut(width).enumerate() {
        let base = row * width;
        for (x, lin) in linear.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            *lin = planar_to_linear_rgb([
                planes[0][base + x],
                planes[1][base + x],
                planes[2][base + x],
            ]);
        }
        encode_row_u8(&linear, &mut encoded);
        for (px, rgb) in px_row.iter_mut().zip(encoded.as_chunks::<3>().0) {
            *px = Rgba8Pixel::new(rgb[0], rgb[1], rgb[2], 255);
        }
    }
}

/// Downsample `region` (or the whole frame) of the linear-light `img` to a
/// thumbnail with a long edge of at most `max_edge`, written into the next
/// buffer of `ring`. An empty region gives a 1x1 black frame.
#[must_use]
pub fn downsample_into(
    ring: &mut FrameRing,
    img: &PlanarImage<f32>,
    region: Option<Region>,
    max_edge: u32,
) -> SharedPixelBuffer<Rgba8Pixel> {
    let region = clamp_region(img.width, img.height, region);
    let (_, _, rw, rh) = region;
    let plane_len = img.width * img.height;
    if rw == 0
        || rh == 0
        || img.luma.len() < plane_len
        || img.chroma_a.len() < plane_len
        || img.chroma_b.len() < plane_len
    {
        return ring.fill_with(1, 1, |pixels| pixels.fill(Rgba8Pixel::new(0, 0, 0, 255)));
    }
    let (tw, th) = thumbnail_size(to_u32(rw), to_u32(rh), max_edge);
    let planes = box_average(
        [&img.luma, &img.chroma_a, &img.chroma_b],
        img.width,
        region,
        (tw as usize, th as usize),
    );
    ring.fill_with(tw, th, |pixels| {
        encode_thumbnail(&planes, tw as usize, pixels);
    })
}

/// Downsample the whole frame `img` so its long edge is at most `max_edge`
/// (never upscaled), encoded as sRGB RGBA8.
#[must_use]
pub fn downsample_to_long_edge(
    img: &PlanarImage<f32>,
    max_edge: u32,
) -> SharedPixelBuffer<Rgba8Pixel> {
    downsample_into(&mut FrameRing::new(), img, None, max_edge)
}

// ── Live preview feed ────────────────────────────────────────────────────────

/// What a pass uses to show its work in progress: a throttle, a rotating buffer
/// set and the function that puts a finished buffer on the canvas.
///
/// [`Self::offer`] is called with every intermediate frame (an aligned frame, an
/// AI running composite). Most calls return at once; about twice a second one
/// builds a [`THUMB_LONG_EDGE`] thumbnail and hands it to `submit`. The full
/// frame is never converted, so the cost per call is a clock read.
pub struct LivePreview<S, C = SystemClock>
where
    S: FnMut(SharedPixelBuffer<Rgba8Pixel>),
    C: Clock,
{
    throttle: PreviewThrottle<C>,
    ring: FrameRing,
    submit: S,
}

impl<S: FnMut(SharedPixelBuffer<Rgba8Pixel>)> LivePreview<S> {
    /// A feed on the wall clock that hands thumbnails to `submit`.
    #[must_use]
    pub const fn new(submit: S) -> Self {
        Self::with_throttle(submit, PreviewThrottle::new())
    }
}

impl<S, C> LivePreview<S, C>
where
    S: FnMut(SharedPixelBuffer<Rgba8Pixel>),
    C: Clock,
{
    /// A feed limited by `throttle` (a test passes a fake clock).
    #[must_use]
    pub const fn with_throttle(submit: S, throttle: PreviewThrottle<C>) -> Self {
        Self {
            throttle,
            ring: FrameRing::new(),
            submit,
        }
    }

    /// Offer the intermediate frame `frame`; `true` when a thumbnail was sent.
    pub fn offer(&mut self, frame: &PlanarImage<f32>) -> bool {
        self.offer_region(frame, None)
    }

    /// Offer the `region` of `frame` (for example the common-coverage crop of an
    /// aligned frame); `true` when a thumbnail was sent.
    pub fn offer_region(&mut self, frame: &PlanarImage<f32>, region: Option<Region>) -> bool {
        if !self.throttle.ready() {
            return false;
        }
        self.send(frame, region);
        true
    }

    /// Send a thumbnail of `region` of `frame` regardless of the throttle (the
    /// last frame of a pass, so the preview ends on the final state).
    pub fn offer_now(&mut self, frame: &PlanarImage<f32>, region: Option<Region>) {
        self.throttle.reset();
        let _ = self.throttle.ready();
        self.send(frame, region);
    }

    fn send(&mut self, frame: &PlanarImage<f32>, region: Option<Region>) {
        let thumbnail = downsample_into(&mut self.ring, frame, region, THUMB_LONG_EDGE);
        (self.submit)(thumbnail);
    }
}

// ── Relief preview maths ───────────────────────────────────────────────────────

/// A focus map sorted once, so the contrast threshold for any rejected fraction
/// is an index lookup instead of a fresh `select_nth` over the whole frame.
///
/// [`Self::threshold`] returns exactly what
/// `stacker_algo::relief::threshold::percentile_threshold` returns for the same
/// values, including its tie handling.
#[derive(Debug, Clone)]
pub struct FocusThreshold {
    sorted: Vec<f32>,
}

impl FocusThreshold {
    /// Sort a copy of `values` (in parallel).
    #[must_use]
    pub fn new(values: &[f32]) -> Self {
        let mut sorted = values.to_vec();
        sorted.par_sort_unstable_by(f32::total_cmp);
        Self { sorted }
    }

    /// How many values were sorted.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.sorted.len()
    }

    /// Whether no value was sorted.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// The absolute threshold that rejects the lowest `contrast_pct` fraction
    /// (applied as `value >= threshold`); `0.0` for an empty map.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the rank is a rounded fraction of a pixel count, clamped to the map"
    )]
    pub fn threshold(&self, contrast_pct: f32) -> f32 {
        let len = self.sorted.len();
        if len == 0 {
            return 0.0;
        }
        let pct = f64::from(contrast_pct).clamp(0.0, 1.0);
        let k = ((len as f64 * pct).round() as usize).min(len - 1);
        let t = self.sorted[k];
        let wanted = len - k;
        let at_least = len - self.sorted.partition_point(|&v| v < t);
        let above = len - self.sorted.partition_point(|&v| v <= t);
        if above.abs_diff(wanted) < at_least.abs_diff(wanted) {
            t.next_up()
        } else {
            t
        }
    }
}

/// Paint the `Relief` rejection overlay: pixels whose focus value is below
/// `threshold` get a red tint, the rest are copied from `base`.
pub fn tint_rejected(base: &[Rgba8Pixel], focus: &[f32], threshold: f32, out: &mut [Rgba8Pixel]) {
    for ((dst, src), &value) in out.iter_mut().zip(base).zip(focus) {
        *dst = if value < threshold {
            Rgba8Pixel::new(
                src.r.saturating_add(100),
                src.g.saturating_sub(50),
                src.b.saturating_sub(50),
                src.a,
            )
        } else {
            *src
        };
    }
}

// ── Decoding ─────────────────────────────────────────────────────────────────

/// Decode the image file `path` into a display buffer. Meant for a worker
/// thread: the UI thread never decodes.
///
/// # Errors
///
/// The decoder's message when the file cannot be read or is not an image.
pub fn decode_file(path: &Path) -> Result<SharedPixelBuffer<Rgba8Pixel>, String> {
    let rgba = image::open(path).map_err(|e| e.to_string())?.into_rgba8();
    Ok(rgba_to_buffer(&rgba))
}

/// Decode the bytes of an image file (JPEG, PNG, TIFF, ...) into a display
/// buffer: [`decode_file`] for a file the host holds in memory (the browser).
///
/// # Errors
///
/// The decoder's message when the bytes are not an image.
pub fn decode_bytes(bytes: &[u8]) -> Result<SharedPixelBuffer<Rgba8Pixel>, String> {
    let rgba = image::load_from_memory(bytes)
        .map_err(|e| e.to_string())?
        .into_rgba8();
    Ok(rgba_to_buffer(&rgba))
}

/// `rgba` as a Slint pixel buffer.
fn rgba_to_buffer(rgba: &image::RgbaImage) -> SharedPixelBuffer<Rgba8Pixel> {
    let (width, height) = rgba.dimensions();
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(width, height);
    for (px, src) in buffer
        .make_mut_slice()
        .iter_mut()
        .zip(rgba.as_raw().as_chunks::<4>().0)
    {
        *px = Rgba8Pixel::new(src[0], src[1], src[2], src[3]);
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::{FrameRing, Serial, clamp_region, thumbnail_size};

    #[test]
    fn thumbnail_size_keeps_the_aspect_and_never_upscales() {
        assert_eq!(thumbnail_size(3600, 2400, 360), (360, 240));
        assert_eq!(thumbnail_size(2400, 3600, 360), (240, 360));
        assert_eq!(thumbnail_size(100, 50, 360), (100, 50));
        assert_eq!(thumbnail_size(100_000, 1, 360), (360, 1));
    }

    #[test]
    fn a_region_is_clipped_to_the_frame() {
        assert_eq!(clamp_region(10, 8, None), (0, 0, 10, 8));
        assert_eq!(clamp_region(10, 8, Some((6, 6, 9, 9))), (6, 6, 4, 2));
        assert_eq!(clamp_region(10, 8, Some((20, 20, 9, 9))), (10, 8, 0, 0));
    }

    #[test]
    fn serial_tokens_supersede_each_other() {
        let serial = Serial::new();
        let first = serial.begin();
        assert!(serial.is_current(first));
        let second = serial.begin();
        assert!(!serial.is_current(first));
        assert!(serial.is_current(second));
    }

    #[test]
    fn the_ring_rotates_through_three_buffers_and_reuses_released_ones() {
        let mut ring = FrameRing::new();
        let first = ring.fill_with(4, 4, |_| {}).as_slice().as_ptr();
        drop(ring.fill_with(4, 4, |_| {}));
        drop(ring.fill_with(4, 4, |_| {}));
        let again = ring.fill_with(4, 4, |_| {});
        assert_eq!(again.as_slice().as_ptr(), first);
    }

    #[test]
    fn the_ring_replaces_a_buffer_of_the_wrong_size() {
        let mut ring = FrameRing::new();
        for _ in 0..3 {
            drop(ring.fill_with(4, 4, |_| {}));
        }
        let resized = ring.fill_with(2, 3, |_| {});
        assert_eq!((resized.width(), resized.height()), (2, 3));
    }
}
