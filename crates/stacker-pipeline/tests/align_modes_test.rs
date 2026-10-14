#![expect(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    reason = "synthetic-image formulas read better without mul_add and with plain casts"
)]

//! The alignment modes and optimisers, the identity gate, the seed look-ahead
//! (same results with one thread and with many, loads in order, one seed per
//! frame), the overlapped driver, the new knobs against the asynchronous driver,
//! and the crop helpers of `crop`.

use std::{
    future::{Future, ready},
    pin::pin,
    sync::{Mutex, PoisonError},
    task::{Context, Poll, Waker},
};

use stacker_align::{
    Matrix3,
    device::{AsyncLevelDevice, Moments, ObjectiveSampler},
    transform::warp_image,
};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, OptimizerSetting},
};
use stacker_pipeline::{
    Cancel, NeverCancel, NoProgress,
    align::{
        AlignedFrame, AlignmentSettings, AsyncAlignBackend, Fallback, FrameFlow, SeedSource,
        SequenceError, SequenceOutcome, align_frame, align_sequence, align_sequence_async,
        align_sequence_overlapped, align_sequence_seeded, lookahead_extra_bytes, lookahead_width,
    },
    compute::Compute,
    crop::{CommonCoverage, resize_planar},
};

const W: usize = 128;
const H: usize = 128;

/// A smooth multi-scale texture, evaluated at `(x - dx, y - dy)`: the frame is
/// the reference's content moved by `(dx, dy)` pixels, with no resampling error.
fn texture(dx: f32, dy: f32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f32 - dx, y as f32 - dy);
            let i = y * W + x;
            img.luma[i] = 0.5
                + 0.2 * (0.23 * fx).sin() * (0.19 * fy).cos()
                + 0.15 * (0.07 * (fx + fy)).sin()
                + 0.1 * (0.31 * fx - 0.27 * fy).cos()
                + 0.08 * (0.5 * fx).sin() * (0.43 * fy).sin();
            img.chroma_a[i] = 0.1 * (0.11 * fx).sin();
            img.chroma_b[i] = 0.1 * (0.13 * fy).cos();
        }
    }
    img
}

/// Mean absolute luma difference away from the borders.
fn interior_error(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> f32 {
    let margin = 20;
    let mut sum = 0.0_f32;
    let mut n = 0.0_f32;
    for y in margin..a.height - margin {
        for x in margin..a.width - margin {
            sum += (a.luma[y * a.width + x] - b.luma[y * a.width + x]).abs();
            n += 1.0;
        }
    }
    sum / n.max(1.0)
}

fn translation(x: f32, y: f32) -> Matrix3<f32> {
    let mut m = Matrix3::identity();
    m[(0, 2)] = x;
    m[(1, 2)] = y;
    m
}

fn bits(m: &Matrix3<f32>) -> [u32; 9] {
    std::array::from_fn(|i| m[(i / 3, i % 3)].to_bits())
}

fn plane_bits(plane: &[f32]) -> Vec<u32> {
    plane.iter().map(|v| v.to_bits()).collect()
}

// ── the flows ─────────────────────────────────────────────────────────────────

struct Delivered {
    index: usize,
    matrix: [u32; 9],
    fallback: Option<Fallback>,
    coverage: Vec<bool>,
    common: Vec<bool>,
    luma: Vec<u32>,
    chroma_a: Vec<u32>,
}

struct VecFlow {
    frames: Vec<Option<PlanarImage<f32>>>,
    loaded: Vec<usize>,
    out: Vec<Delivered>,
}

impl VecFlow {
    fn new(frames: &[PlanarImage<f32>]) -> Self {
        Self {
            frames: frames.iter().cloned().map(Some).collect(),
            loaded: Vec::new(),
            out: Vec::new(),
        }
    }
}

impl FrameFlow<String> for VecFlow {
    fn load(&mut self, index: usize) -> Result<PlanarImage<f32>, String> {
        self.loaded.push(index);
        self.frames[index]
            .take()
            .ok_or_else(|| format!("frame {index} requested twice"))
    }

    fn deliver(
        &mut self,
        index: usize,
        aligned: AlignedFrame,
        common: &[bool],
    ) -> Result<(), String> {
        self.out.push(Delivered {
            index,
            matrix: bits(&aligned.matrix_applied),
            fallback: aligned.fallback,
            luma: plane_bits(&aligned.frame.luma),
            chroma_a: plane_bits(&aligned.frame.chroma_a),
            coverage: aligned.coverage,
            common: common.to_vec(),
        });
        Ok(())
    }
}

fn assert_same(label: &str, a: &[Delivered], b: &[Delivered]) {
    assert_eq!(a.len(), b.len(), "{label}: frame count");
    for (x, y) in a.iter().zip(b) {
        let at = format!("{label}, frame {}", x.index);
        assert_eq!(x.index, y.index, "{at}");
        assert_eq!(x.matrix, y.matrix, "{at}: matrix");
        assert_eq!(x.fallback, y.fallback, "{at}: fallback");
        assert_eq!(x.coverage, y.coverage, "{at}: coverage");
        assert_eq!(x.common, y.common, "{at}: common coverage");
        assert_eq!(x.luma, y.luma, "{at}: luma");
        assert_eq!(x.chroma_a, y.chroma_a, "{at}: chroma");
    }
}

// ── optimisers and motion models ──────────────────────────────────────────────

/// Every optimiser, in the translation and the registration mode, finds the
/// shift of a pure-shift frame.
#[test]
fn every_optimizer_recovers_a_shift() {
    let reference = texture(0.0, 0.0);
    let shifted = texture(3.0, 2.0);
    for mode in [
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Registration,
        AlignmentModeSetting::Affine,
    ] {
        for optimizer in [
            OptimizerSetting::Auto,
            OptimizerSetting::LucasKanade,
            OptimizerSetting::NelderMead,
        ] {
            let settings = AlignmentSettings::new(mode, false).with_optimizer(optimizer);
            let aligned = align_frame(&reference, shifted.clone(), &settings, None);
            let at = format!("{mode:?} {optimizer:?}");
            assert_eq!(aligned.fallback, None, "{at}");
            let m = aligned.matrix_applied;
            assert!((m[(0, 2)] + 3.0).abs() < 0.5, "{at}: tx {}", m[(0, 2)]);
            assert!((m[(1, 2)] + 2.0).abs() < 0.5, "{at}: ty {}", m[(1, 2)]);
            assert!(
                interior_error(&reference, &aligned.frame)
                    < 0.25 * interior_error(&reference, &shifted),
                "{at}: the aligned frame must match the reference far better"
            );
        }
    }
}

/// D3: `affine` fits aspect and shear as well; a source with both is matched
/// better by it than by the four-parameter `registration`.
#[test]
fn affine_mode_fits_aspect_and_shear_that_registration_cannot() {
    let reference = texture(0.0, 0.0);
    let mut b = Matrix3::<f32>::identity();
    b[(0, 0)] = 1.01;
    b[(0, 1)] = 0.02;
    b[(0, 2)] = 1.5;
    b[(1, 1)] = 0.985;
    b[(1, 2)] = -1.0;
    // `source(B·p) = reference(p)`.
    let source = warp_image(&reference, &b).expect("an invertible warp");

    let error_of = |mode| {
        let settings = AlignmentSettings::new(mode, false);
        let aligned = align_frame(&reference, source.clone(), &settings, None);
        interior_error(&reference, &aligned.frame)
    };
    let affine = error_of(AlignmentModeSetting::Affine);
    let registration = error_of(AlignmentModeSetting::Registration);
    assert!(
        affine < 0.7 * registration,
        "affine {affine} must beat registration {registration}"
    );
}

/// Frames that already agree stay where they are in every mode and with every
/// optimiser (the identity gate keeps an equal score, and nothing wanders).
#[test]
fn identical_frames_stay_put() {
    let reference = texture(0.0, 0.0);
    for mode in [
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Registration,
        AlignmentModeSetting::Affine,
    ] {
        for optimizer in [
            OptimizerSetting::Auto,
            OptimizerSetting::LucasKanade,
            OptimizerSetting::NelderMead,
        ] {
            let settings = AlignmentSettings::new(mode, true).with_optimizer(optimizer);
            let aligned = align_frame(&reference, reference.clone(), &settings, None);
            let m = aligned.matrix_applied;
            let at = format!("{mode:?} {optimizer:?}: {m}");
            assert!(m[(0, 2)].abs() < 0.05 && m[(1, 2)].abs() < 0.05, "{at}");
            assert!((m[(0, 0)] - 1.0).abs() < 2e-3, "{at}");
            assert!((m[(1, 1)] - 1.0).abs() < 2e-3, "{at}");
        }
    }
}

/// Without the border crop the warp is the edge-clamped spline: the corner the
/// shift uncovers holds replicated data instead of black, and the coverage mask
/// is the same.
#[test]
fn a_clamped_border_replicates_instead_of_going_black() {
    let reference = texture(0.0, 0.0);
    let shifted = texture(3.0, 2.0);
    let zero = AlignmentSettings::new(AlignmentModeSetting::Translation, false);
    let clamped = zero.clone().with_clamped_border(true);
    let black = align_frame(&reference, shifted.clone(), &zero, None);
    let replicated = align_frame(&reference, shifted, &clamped, None);

    let corner = W * H - 1;
    assert!(!black.coverage[corner] && !replicated.coverage[corner]);
    assert!(black.frame.luma[corner].abs() < 1e-6, "zero border");
    assert!(replicated.frame.luma[corner] > 0.05, "replicated border");
}

// ── look-ahead ────────────────────────────────────────────────────────────────

/// Seeds the true shift of frame `i`, `(1.5·i, −i)`, and records which frames it
/// was asked about.
struct TrueShifts {
    asked: Mutex<Vec<usize>>,
}

impl SeedSource for TrueShifts {
    fn seed(&self, index: usize, _frame: &PlanarImage<f32>) -> Option<Matrix3<f32>> {
        self.asked
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(index);
        Some(translation(1.5 * index as f32, -(index as f32)))
    }
}

fn shifted_stack(count: usize) -> Vec<PlanarImage<f32>> {
    (0..count)
        .map(|i| texture(1.5 * i as f32, -(i as f32)))
        .collect()
}

fn run_seeded(
    threads: usize,
    frames: &[PlanarImage<f32>],
    settings: &AlignmentSettings,
) -> (Vec<Delivered>, Vec<usize>, Vec<usize>) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a thread pool");
    let seeds = TrueShifts {
        asked: Mutex::new(Vec::new()),
    };
    let mut flow = VecFlow::new(frames);
    pool.install(|| {
        align_sequence_seeded(
            &frames[0],
            frames.len(),
            settings,
            Some(&seeds),
            &NeverCancel,
            &NoProgress,
            &mut flow,
            &mut Compute::cpu(),
        )
    })
    .expect("the run succeeds");
    let mut asked = seeds
        .asked
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    asked.sort_unstable();
    (flow.out, flow.loaded, asked)
}

/// The look-ahead seeds several frames at once, yet the frames are loaded once
/// each and in order, every frame is seeded exactly once, and the aligned frames
/// are the same bits with one thread and with four.
#[test]
fn the_seed_look_ahead_does_not_change_the_result() {
    let frames = shifted_stack(6);
    let settings =
        AlignmentSettings::new(AlignmentModeSetting::Registration, false).with_brightness(true);
    let (one, loaded_one, asked_one) = run_seeded(1, &frames, &settings);
    let (many, loaded_many, asked_many) = run_seeded(4, &frames, &settings);

    assert_same("1 thread vs 4 threads", &one, &many);
    let expected: Vec<usize> = (1..frames.len()).collect();
    assert_eq!(loaded_one, expected, "loads in order (1 thread)");
    assert_eq!(loaded_many, expected, "loads in order (4 threads)");
    assert_eq!(asked_one, expected, "one seed per frame (1 thread)");
    assert_eq!(asked_many, expected, "one seed per frame (4 threads)");

    // And the seeds were used: the frames line up with the reference.
    for (i, delivered) in many.iter().enumerate() {
        assert_eq!(delivered.fallback, None, "frame {}", i + 1);
        assert!(delivered.coverage[W * H / 2 + W / 2]);
    }
}

/// A frame that cannot be loaded ends the run with the flow's error, but only
/// after every earlier frame was delivered, however many frames the look-ahead
/// had already loaded.
#[test]
fn a_load_error_surfaces_in_order_even_with_a_look_ahead() {
    let frames = shifted_stack(6);
    let settings = AlignmentSettings::new(AlignmentModeSetting::Registration, false);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("a thread pool");
    let seeds = TrueShifts {
        asked: Mutex::new(Vec::new()),
    };
    let mut flow = VecFlow::new(&frames);
    flow.frames[3] = None;
    let result = pool.install(|| {
        align_sequence_seeded(
            &frames[0],
            frames.len(),
            &settings,
            Some(&seeds),
            &NeverCancel,
            &NoProgress,
            &mut flow,
            &mut Compute::cpu(),
        )
    });
    match result {
        Err(SequenceError::Flow(message)) => assert!(message.contains("frame 3"), "{message}"),
        other => panic!("expected the flow error, got {other:?}"),
    }
    let delivered: Vec<usize> = flow.out.iter().map(|d| d.index).collect();
    assert_eq!(delivered, [1, 2], "the frames before the failure arrive");
    assert!(
        !flow.loaded.contains(&4),
        "no frame is requested after the failing one: {:?}",
        flow.loaded
    );
}

/// The overlapped driver (the next frame is decoded on a second thread) gives
/// the serial driver's bits, in every mode.
#[test]
fn the_overlapped_driver_matches_the_serial_one() {
    let frames = shifted_stack(5);
    for mode in [
        AlignmentModeSetting::None,
        AlignmentModeSetting::Translation,
        AlignmentModeSetting::Affine,
        AlignmentModeSetting::Registration,
    ] {
        let settings = AlignmentSettings::new(mode, false).with_brightness(true);

        let mut serial = VecFlow::new(&frames);
        let reference_run = align_sequence(
            &frames[0],
            frames.len(),
            &settings,
            &NeverCancel,
            &NoProgress,
            &mut serial,
        )
        .expect("the serial run succeeds");

        let mut overlapped = VecFlow::new(&frames);
        let overlapped_run = align_sequence_overlapped(
            &frames[0],
            frames.len(),
            &settings,
            &NeverCancel,
            &NoProgress,
            &mut overlapped,
            &mut Compute::cpu(),
        )
        .expect("the overlapped run succeeds");

        assert_same(&format!("{mode:?}"), &serial.out, &overlapped.out);
        assert_eq!(overlapped.loaded, serial.loaded, "{mode:?}: loads");
        assert_eq!(
            reference_run.coverage.mask(),
            overlapped_run.coverage.mask(),
            "{mode:?}: final mask"
        );
        assert_eq!(reference_run.fallbacks, overlapped_run.fallbacks);
    }
}

struct CancelAfter(std::sync::atomic::AtomicUsize);

impl Cancel for CancelAfter {
    fn is_cancelled(&self) -> bool {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 1
    }
}

/// Cancelling the overlapped run stops it (and the decoding thread) cleanly.
#[test]
fn cancelling_the_overlapped_driver_stops_it() {
    let frames = shifted_stack(5);
    let settings = AlignmentSettings::new(AlignmentModeSetting::None, false);
    let mut flow = VecFlow::new(&frames);
    let result = align_sequence_overlapped(
        &frames[0],
        frames.len(),
        &settings,
        &CancelAfter(std::sync::atomic::AtomicUsize::new(0)),
        &NoProgress,
        &mut flow,
        &mut Compute::cpu(),
    );
    assert!(matches!(result, Err(SequenceError::Cancelled)));
    assert_eq!(flow.out.len(), 1, "stopped before the second frame");
}

#[test]
fn the_look_ahead_memory_is_bounded() {
    assert!((1..=stacker_pipeline::align::MAX_LOOKAHEAD).contains(&lookahead_width()));
    let (w, h) = (6000, 4000);
    let serial = lookahead_extra_bytes(w, h, false);
    let overlapped = lookahead_extra_bytes(w, h, true);
    assert_eq!(
        overlapped - serial,
        12 * w * h,
        "one more frame in the decoder"
    );
    assert_eq!(serial, (lookahead_width() - 1) * 12 * w * h);
}

// ── the new knobs against the asynchronous driver ─────────────────────────────

/// A device that takes nothing: every level and every warp is the CPU's.
struct NullBackend;

impl AsyncLevelDevice for NullBackend {
    fn level(
        &mut self,
        _reference: &[f32],
        _source: &[f32],
        _width: usize,
        _height: usize,
        _sampler: ObjectiveSampler,
    ) -> impl Future<Output = bool> {
        ready(false)
    }

    fn moments(&mut self, _inverse: &Matrix3<f32>) -> impl Future<Output = Option<Moments>> {
        ready(None)
    }
}

impl AsyncAlignBackend for NullBackend {
    fn warp(
        &mut self,
        _frame: &PlanarImage<f32>,
        _apply: &Matrix3<f32>,
        _edge_clamped: bool,
    ) -> impl Future<Output = Option<(PlanarImage<f32>, Vec<bool>)>> {
        ready(None)
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

/// Brightness, the identity gate, the chain and the clamped border behave the
/// same in the browser's driver: same bits as the synchronous run with the
/// simplex.
#[test]
fn the_asynchronous_driver_agrees_on_brightness_gate_chain_and_border() {
    let frames = shifted_stack(4);
    let darker: Vec<PlanarImage<f32>> = frames
        .iter()
        .enumerate()
        .map(|(i, frame)| {
            let mut frame = frame.clone();
            let gain = 1.0 - 0.08 * i as f32;
            frame.luma.iter_mut().for_each(|v| *v *= gain);
            frame
        })
        .collect();
    let base = |mode| AlignmentSettings::new(mode, false);
    let cases = [
        base(AlignmentModeSetting::Registration).with_brightness(true),
        base(AlignmentModeSetting::Translation).with_chain(true),
        base(AlignmentModeSetting::Affine)
            .with_chain(true)
            .with_brightness(true),
        base(AlignmentModeSetting::Registration)
            .with_brightness(true)
            .with_clamped_border(true),
        base(AlignmentModeSetting::Registration).with_optimizer(OptimizerSetting::Auto),
    ];
    for (n, settings) in cases.iter().enumerate() {
        let mut sync_flow = VecFlow::new(&darker);
        // `Auto` runs Lucas-Kanade on the CPU, which the browser never does: the
        // simplex is the common ground.
        let sync_settings = settings
            .clone()
            .with_optimizer(OptimizerSetting::NelderMead);
        let sync_outcome: SequenceOutcome = align_sequence(
            &darker[0],
            darker.len(),
            &sync_settings,
            &NeverCancel,
            &NoProgress,
            &mut sync_flow,
        )
        .expect("the synchronous run succeeds");

        let mut async_flow = VecFlow::new(&darker);
        let async_outcome = block_on(align_sequence_async(
            &darker[0],
            darker.len(),
            settings,
            &NeverCancel,
            &NoProgress,
            &mut async_flow,
            &mut NullBackend,
            true,
        ))
        .unwrap_or_else(|_| panic!("case {n}: the asynchronous run fails"));

        assert_same(&format!("case {n}"), &sync_flow.out, &async_flow.out);
        assert_eq!(
            sync_outcome.coverage.mask(),
            async_outcome.coverage.mask(),
            "case {n}: final mask"
        );
        assert_eq!(sync_outcome.fallbacks, async_outcome.fallbacks, "case {n}");
    }
}

// ── the crop helpers ──────────────────────────────────────────────────────────

#[test]
fn the_crop_can_be_switched_off() {
    let mut coverage = CommonCoverage::new(10, 8);
    let mut mask = vec![true; 80];
    for y in 0..8 {
        mask[y * 10] = false;
        mask[y * 10 + 1] = false;
    }
    coverage.fold(1, &mask);

    let on = coverage.crop_when(true, |i| format!("frame {i}"));
    assert_eq!(on.rect, (2, 0, 8, 8));
    assert!(!on.refused);
    assert_eq!(on.needed(10, 8), Some((2, 0, 8, 8)));

    let off = coverage.crop_when(false, |i| format!("frame {i}"));
    assert_eq!(off.rect, (0, 0, 10, 8));
    assert!(!off.refused);
    assert_eq!(off.needed(10, 8), None);
}

#[test]
fn resizing_back_keeps_constants_and_ramps() {
    // Constants stay constant (the kernel's weights sum to one).
    let mut flat = PlanarImage::<f32>::new(7, 5);
    flat.luma.fill(0.6);
    flat.chroma_a.fill(-0.2);
    let big = resize_planar(flat, 20, 11);
    assert_eq!((big.width, big.height), (20, 11));
    assert!(big.luma.iter().all(|v| (v - 0.6).abs() < 1e-5));
    assert!(big.chroma_a.iter().all(|v| (v + 0.2).abs() < 1e-5));
    assert!(big.chroma_b.iter().all(|v| v.abs() < 1e-6));

    // A ramp in x stays a ramp: output pixel `o` samples `(o + ½)·src/dst − ½`.
    let (sw, dw) = (16, 40);
    let mut ramp = PlanarImage::<f32>::new(sw, 4);
    for y in 0..4 {
        for x in 0..sw {
            ramp.luma[y * sw + x] = x as f32;
        }
    }
    let up = resize_planar(ramp, dw, 4);
    for (o, &value) in up.luma.iter().enumerate().take(dw - 4).skip(4) {
        let expected = (o as f32 + 0.5) * (sw as f32 / dw as f32) - 0.5;
        assert!(
            (value - expected).abs() < 1e-3,
            "o={o}: {value} vs {expected}"
        );
    }

    // The target size is returned as it is; empty input gives zeros.
    let same = PlanarImage::<f32>::new(3, 3);
    assert_eq!(resize_planar(same, 3, 3).luma.len(), 9);
    let empty = resize_planar(PlanarImage::<f32>::new(0, 0), 4, 4);
    assert_eq!((empty.width, empty.height, empty.luma.len()), (4, 4, 16));
}
