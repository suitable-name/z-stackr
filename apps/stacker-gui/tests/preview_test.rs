//! The preview path: the redraw gate, the throttle, the downsampler, the
//! last-wins worker and the decoder that serves list clicks.

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use slint::{Rgba8Pixel, SharedPixelBuffer};
use stacker_core::image::PlanarImage;
use stacker_gui::{
    latest_worker::LatestWorker,
    preview::{
        Clock, FocusThreshold, FrameRing, LivePreview, PreviewThrottle, Serial, THUMB_LONG_EDGE,
        THUMB_MIN_INTERVAL, decode_file, downsample_into, downsample_to_long_edge, tint_rejected,
    },
    redraw_gate::RedrawGate,
    stall_guard::stall_guard,
};

/// Long enough for a loaded machine, short enough that a hang fails the test.
const WAIT: Duration = Duration::from_secs(10);

// ── Helpers ─────────────────────────────────────────────────────────────────

/// A `width` x `height` frame with the same linear `luma` everywhere and no
/// chroma: a grey whose sRGB code is easy to predict (0.0 is 0, 1.0 is 255).
fn grey(width: usize, height: usize, luma: f32) -> PlanarImage<f32> {
    let mut image = PlanarImage::new(width, height);
    image.luma.fill(luma);
    image
}

fn pixels_of(buffer: &SharedPixelBuffer<Rgba8Pixel>) -> &[Rgba8Pixel] {
    buffer.as_slice()
}

/// A clock that returns pre-scheduled instants, one per call.
struct FakeClock {
    times: std::vec::IntoIter<Instant>,
}

impl FakeClock {
    fn new(times: Vec<Instant>) -> Self {
        Self {
            times: times.into_iter(),
        }
    }
}

impl Clock for FakeClock {
    fn now(&mut self) -> Instant {
        self.times
            .next()
            .expect("the fake clock ran out of instants")
    }
}

fn temp_png(name: &str, width: u32, height: u32, rgba: [u8; 4]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "z_stackr_preview_test_{}_{name}.png",
        std::process::id()
    ));
    image::RgbaImage::from_pixel(width, height, image::Rgba(rgba))
        .save(&path)
        .expect("the temp PNG is written");
    path
}

// ── Redraw gate ─────────────────────────────────────────────────────────────

#[test]
fn a_burst_of_submits_queues_one_closure_and_it_sees_the_latest_payload() {
    let gate: RedrawGate<u32> = RedrawGate::new();
    let queued = (0..200).filter(|&i| gate.submit(i).is_some()).count();
    assert_eq!(
        queued, 1,
        "only the first submit of a burst queues a closure"
    );
    assert_eq!(gate.current_generation(), 200, "every submit is counted");
    assert_eq!(
        gate.take(),
        Some(199),
        "the closure shows the newest payload"
    );
    assert!(!gate.is_pending());
    assert!(gate.submit(200).is_some(), "the next burst queues again");
}

#[test]
fn submits_from_several_threads_still_queue_exactly_one_closure() {
    let gate = Arc::new(RedrawGate::<usize>::new());
    let queued = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let gate = Arc::clone(&gate);
            let queued = Arc::clone(&queued);
            thread::spawn(move || {
                for i in 0..250 {
                    if gate.submit(t * 1000 + i).is_some() {
                        queued.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();
    for handle in threads {
        handle.join().expect("a producer thread finished");
    }
    assert_eq!(queued.load(Ordering::SeqCst), 1);
    assert_eq!(gate.current_generation(), 1000);
    assert!(gate.take().is_some());
}

// ── Throttle ────────────────────────────────────────────────────────────────

#[test]
fn the_throttle_passes_the_first_call_then_waits_out_the_interval() {
    let base = Instant::now();
    let mut throttle = PreviewThrottle::with_clock(
        FakeClock::new(vec![
            base,
            base + THUMB_MIN_INTERVAL / 2,
            base + Duration::from_millis(499),
            base + THUMB_MIN_INTERVAL,
            base + THUMB_MIN_INTERVAL + THUMB_MIN_INTERVAL / 2,
        ]),
        THUMB_MIN_INTERVAL,
    );
    assert!(throttle.ready(), "the first call always passes");
    assert!(!throttle.ready(), "half an interval later: withheld");
    assert!(!throttle.ready(), "a millisecond short: withheld");
    assert!(throttle.ready(), "a full interval later: passes");
    assert!(!throttle.ready(), "half an interval after that: withheld");
}

#[test]
fn a_live_preview_sends_a_thumbnail_per_interval_not_per_frame() {
    let base = Instant::now();
    let throttle = PreviewThrottle::with_clock(
        FakeClock::new(vec![
            base,
            base + Duration::from_millis(10),
            base + Duration::from_millis(20),
            base + THUMB_MIN_INTERVAL + Duration::from_millis(30),
        ]),
        THUMB_MIN_INTERVAL,
    );
    let sent: Arc<Mutex<Vec<(u32, u32)>>> = Arc::default();
    let sink = Arc::clone(&sent);
    let mut live = LivePreview::with_throttle(
        move |buffer: SharedPixelBuffer<Rgba8Pixel>| {
            sink.lock()
                .expect("the sink is not poisoned")
                .push((buffer.width(), buffer.height()));
        },
        throttle,
    );
    // A 1200 x 600 frame: the thumbnail is 360 x 180, never the full frame.
    let frame = grey(1200, 600, 0.5);
    assert!(live.offer(&frame));
    assert!(!live.offer(&frame));
    assert!(!live.offer(&frame));
    assert!(live.offer(&frame));
    let sent = sent.lock().expect("the sink is not poisoned");
    assert_eq!(*sent, vec![(360, 180), (360, 180)]);
}

// ── Downsampling ────────────────────────────────────────────────────────────

#[test]
fn the_thumbnail_has_the_requested_long_edge_and_keeps_the_aspect() {
    let wide = downsample_to_long_edge(&grey(1000, 500, 0.5), THUMB_LONG_EDGE);
    assert_eq!((wide.width(), wide.height()), (360, 180));
    let tall = downsample_to_long_edge(&grey(500, 1000, 0.5), THUMB_LONG_EDGE);
    assert_eq!((tall.width(), tall.height()), (180, 360));
}

#[test]
fn a_small_frame_is_not_upscaled() {
    let small = downsample_to_long_edge(&grey(40, 30, 0.5), THUMB_LONG_EDGE);
    assert_eq!((small.width(), small.height()), (40, 30));
}

#[test]
fn a_white_image_encodes_to_255_and_a_black_one_to_0() {
    let white = downsample_to_long_edge(&grey(900, 600, 1.0), THUMB_LONG_EDGE);
    assert!(
        pixels_of(&white)
            .iter()
            .all(|p| (p.r, p.g, p.b, p.a) == (255, 255, 255, 255)),
        "white must encode to 255 in every channel, opaque"
    );
    let black = downsample_to_long_edge(&grey(900, 600, 0.0), THUMB_LONG_EDGE);
    assert!(
        pixels_of(&black)
            .iter()
            .all(|p| (p.r, p.g, p.b, p.a) == (0, 0, 0, 255))
    );
}

#[test]
fn buckets_average_in_linear_light_before_encoding() {
    // Left half black, right half white, reduced to 2 x 1: one bucket each.
    let mut frame = grey(8, 2, 0.0);
    for row in 0..2 {
        for x in 4..8 {
            frame.luma[row * 8 + x] = 1.0;
        }
    }
    let thumb = downsample_to_long_edge(&frame, 2);
    assert_eq!((thumb.width(), thumb.height()), (2, 1));
    let px = pixels_of(&thumb);
    assert_eq!(px[0].r, 0);
    assert_eq!(px[1].r, 255);
}

#[test]
fn a_region_is_downsampled_without_copying_the_frame_out() {
    // Right half white: the crop of the right half is all white.
    let mut frame = grey(100, 40, 0.0);
    for row in 0..40 {
        for x in 50..100 {
            frame.luma[row * 100 + x] = 1.0;
        }
    }
    let mut ring = FrameRing::new();
    let crop = downsample_into(&mut ring, &frame, Some((50, 0, 50, 40)), THUMB_LONG_EDGE);
    assert_eq!((crop.width(), crop.height()), (50, 40));
    assert!(pixels_of(&crop).iter().all(|p| p.r == 255));
    // An empty region degrades to a 1 x 1 frame instead of panicking.
    let empty = downsample_into(&mut ring, &frame, Some((10, 10, 0, 0)), THUMB_LONG_EDGE);
    assert_eq!((empty.width(), empty.height()), (1, 1));
}

#[test]
fn the_ring_never_alters_a_frame_the_caller_still_holds() {
    let mut ring = FrameRing::new();
    let held: Vec<_> = (1..=7_u8)
        .map(|shade| {
            ring.fill_with(2, 1, |px| {
                px.fill(Rgba8Pixel::new(shade, shade, shade, 255));
            })
        })
        .collect();
    for (shade, frame) in (1..=7_u8).zip(&held) {
        assert!(
            pixels_of(frame).iter().all(|p| p.r == shade),
            "frame {shade}"
        );
    }
}

// ── Last-wins worker ────────────────────────────────────────────────────────

#[test]
fn the_worker_handles_only_the_newest_waiting_request() {
    let (seen_tx, seen_rx) = mpsc::channel::<u32>();
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let worker = LatestWorker::spawn("test-last-wins", move |n: u32| {
        let _ = seen_tx.send(n);
        if n == 0 {
            let _ = started_tx.send(());
            let _ = release_rx.recv();
        }
    });
    assert!(worker.submit(0));
    started_rx.recv_timeout(WAIT).expect("request 0 started");
    for n in 1..=50 {
        assert!(worker.submit(n));
    }
    release_tx.send(()).expect("the handler waits");
    assert_eq!(seen_rx.recv_timeout(WAIT), Ok(0));
    assert_eq!(seen_rx.recv_timeout(WAIT), Ok(50));
    assert_eq!(
        seen_rx.recv_timeout(Duration::from_millis(200)),
        Err(mpsc::RecvTimeoutError::Timeout),
        "the 49 requests in between were replaced, not queued"
    );
}

#[test]
fn the_worker_survives_a_panicking_request() {
    let (seen_tx, seen_rx) = mpsc::channel::<u32>();
    let (hook_tx, hook_rx) = mpsc::channel::<()>();
    let worker = LatestWorker::spawn_with_panic_hook(
        "test-panic",
        move |n: u32| {
            assert_ne!(n, 0, "the first request panics on purpose");
            let _ = seen_tx.send(n);
        },
        move || {
            let _ = hook_tx.send(());
        },
    );
    assert!(worker.submit(0));
    hook_rx.recv_timeout(WAIT).expect("the panic hook ran");
    assert!(worker.submit(1));
    assert_eq!(seen_rx.recv_timeout(WAIT), Ok(1));
}

// ── The list-click decoder ──────────────────────────────────────────────────

/// The shape of `gui::files`' viewer: one decoder thread, a `Serial` per click,
/// a stale decode dropped before it is published.
#[test]
fn the_decoder_publishes_the_newest_click_and_drops_the_superseded_one() {
    let files: Vec<PathBuf> = (0..3_u8)
        .map(|i| temp_png(&format!("click{i}"), 4, 4, [i, i, i, 255]))
        .collect();
    let serial = Arc::new(Serial::new());
    let (published_tx, published_rx) = mpsc::channel::<u8>();
    let (decoded_tx, decoded_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();

    let worker_serial = Arc::clone(&serial);
    let worker = LatestWorker::spawn("test-decoder", move |(token, path): (u64, PathBuf)| {
        if !worker_serial.is_current(token) {
            return;
        }
        let decoded = decode_file(&path).expect("the temp PNG decodes");
        if token == 1 {
            // Hold the first decode "in flight" while newer clicks arrive.
            let _ = decoded_tx.send(());
            let _ = release_rx.recv();
        }
        if !worker_serial.is_current(token) {
            return;
        }
        let _ = published_tx.send(decoded.as_slice()[0].r);
    });

    assert!(worker.submit((serial.begin(), files[0].clone())));
    decoded_rx
        .recv_timeout(WAIT)
        .expect("the first decode is in flight");
    assert!(worker.submit((serial.begin(), files[1].clone())));
    assert!(worker.submit((serial.begin(), files[2].clone())));
    release_tx.send(()).expect("the first decode waits");

    assert_eq!(
        published_rx.recv_timeout(WAIT),
        Ok(2),
        "only the newest click is published"
    );
    assert_eq!(
        published_rx.recv_timeout(Duration::from_millis(200)),
        Err(mpsc::RecvTimeoutError::Timeout),
        "the superseded in-flight decode and the replaced click publish nothing"
    );
    for file in files {
        let _ = std::fs::remove_file(file);
    }
}

#[test]
fn decode_file_returns_the_pixels_and_reports_unreadable_files() {
    let path = temp_png("decode", 3, 2, [10, 20, 30, 255]);
    let buffer = decode_file(&path).expect("the PNG decodes");
    assert_eq!((buffer.width(), buffer.height()), (3, 2));
    assert!(
        pixels_of(&buffer)
            .iter()
            .all(|p| (p.r, p.g, p.b, p.a) == (10, 20, 30, 255))
    );
    let _ = std::fs::remove_file(&path);

    let missing = std::env::temp_dir().join("z_stackr_preview_test_missing.png");
    assert!(decode_file(&missing).is_err());
}

// ── Relief preview maths ──────────────────────────────────────────────────────

#[test]
fn the_sorted_threshold_equals_the_algorithm_crates_percentile() {
    // A plateau of zeros (flat background), ties, and a ramp: the cases the tie
    // rule exists for.
    let mut values: Vec<f32> = vec![0.0; 400];
    values.extend((0..300_i16).map(|i| f32::from(i / 3)));
    values.extend(std::iter::repeat_n(7.5, 100));
    values.extend((0..200_i16).map(|i| f32::from(i) * 0.25));
    let sorted = FocusThreshold::new(&values);
    assert_eq!(sorted.len(), values.len());
    for step in 0..=100_u8 {
        let pct = f32::from(step) / 100.0;
        let expected = stacker_algo::relief::threshold::percentile_threshold(&values, pct);
        assert_eq!(
            sorted.threshold(pct).to_bits(),
            expected.to_bits(),
            "contrast_pct {pct}"
        );
    }
    assert_eq!(
        FocusThreshold::new(&[]).threshold(0.5).to_bits(),
        0.0_f32.to_bits()
    );
}

#[test]
fn the_overlay_tints_only_the_rejected_pixels() {
    let base = vec![Rgba8Pixel::new(100, 100, 100, 255); 3];
    let focus = [0.1, 0.5, 0.9];
    let mut out = vec![Rgba8Pixel::default(); 3];
    tint_rejected(&base, &focus, 0.5, &mut out);
    assert_eq!(
        (out[0].r, out[0].g, out[0].b, out[0].a),
        (200, 50, 50, 255),
        "below the threshold: red tint"
    );
    assert_eq!(out[1], base[1], "at the threshold: kept");
    assert_eq!(out[2], base[2], "above the threshold: kept");
}

// ── Stall guard ─────────────────────────────────────────────────────────────

#[test]
fn the_stall_guard_is_transparent() {
    let mut order = Vec::new();
    let value = stall_guard("test", || {
        order.push("body");
        7
    });
    assert_eq!(value, 7);
    assert_eq!(order, ["body"]);
}
