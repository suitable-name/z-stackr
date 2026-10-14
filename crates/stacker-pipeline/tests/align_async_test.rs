#![allow(
    clippy::suboptimal_flops,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

//! `align_sequence_async` against `align_sequence`: a backend that answers like
//! the CPU (or like a scripted device that the synchronous run is given as its
//! `Compute` backend) gives the same aligned frames, coverage and fallbacks bit
//! for bit; and with `cpu_allowed = false` a refusing device ends the run with a
//! clear error.

use std::{
    future::Future,
    pin::pin,
    sync::{Arc, Mutex, PoisonError},
    task::{Context, Poll, Waker},
};

use stacker_align::{
    Matrix3,
    device::{
        AsyncLevelDevice, DeviceHandle, LevelDevice, Moments, ObjectiveDevice, ObjectiveSampler,
    },
    transform::{coverage_mask, warp_image, warp_image_clamped},
};
use stacker_core::{
    image::PlanarImage,
    settings::{AlignmentModeSetting, StackingSettings},
};
use stacker_pipeline::{
    Cancel, NeverCancel, NoProgress,
    align::{
        AlignedFrame, AlignmentSettings, AsyncAlignBackend, AsyncSequenceError, Fallback,
        FrameFlow, SequenceOutcome, align_sequence, align_sequence_async, align_sequence_on,
    },
    compute::{Backend, BackendError, Compute},
};

const W: usize = 160;
const H: usize = 128;

/// A smooth multi-scale texture, evaluated at `(x - dx, y - dy)`.
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

/// The reference first, then three shifted frames.
fn stack() -> Vec<PlanarImage<f32>> {
    vec![
        texture(0.0, 0.0),
        texture(2.0, 1.5),
        texture(-3.0, 2.0),
        texture(5.0, -4.0),
    ]
}

/// The simplex optimiser (the only one the asynchronous driver has), no
/// brightness fit: the minimal form, so the scripted device sees the same
/// evaluations on both sides. `align_modes_test` covers the new knobs.
const fn settings(mode: AlignmentModeSetting, bounded: bool) -> AlignmentSettings {
    AlignmentSettings::new(mode, bounded)
}

const MODES: [(AlignmentModeSetting, bool); 7] = [
    (AlignmentModeSetting::None, false),
    (AlignmentModeSetting::Translation, false),
    (AlignmentModeSetting::Translation, true),
    (AlignmentModeSetting::Affine, false),
    (AlignmentModeSetting::Affine, true),
    (AlignmentModeSetting::Registration, false),
    (AlignmentModeSetting::Registration, true),
];

// ── the flow and the comparison ───────────────────────────────────────────────

struct Delivered {
    index: usize,
    frame: PlanarImage<f32>,
    coverage: Vec<bool>,
    matrix: [u32; 9],
    fallback: Option<Fallback>,
    common: Vec<bool>,
}

struct Flow {
    frames: Vec<Option<PlanarImage<f32>>>,
    out: Vec<Delivered>,
}

impl Flow {
    fn new(frames: &[PlanarImage<f32>]) -> Self {
        Self {
            frames: frames.iter().cloned().map(Some).collect(),
            out: Vec::new(),
        }
    }
}

impl FrameFlow<String> for Flow {
    fn load(&mut self, index: usize) -> Result<PlanarImage<f32>, String> {
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
        let m = aligned.matrix_applied;
        self.out.push(Delivered {
            index,
            matrix: std::array::from_fn(|i| m[(i / 3, i % 3)].to_bits()),
            coverage: aligned.coverage,
            fallback: aligned.fallback,
            frame: aligned.frame,
            common: common.to_vec(),
        });
        Ok(())
    }
}

fn plane_bits(plane: &[f32]) -> Vec<u32> {
    plane.iter().map(|v| v.to_bits()).collect()
}

type Run = (Vec<Delivered>, SequenceOutcome);

fn assert_same_run(label: &str, sync: &Run, asynchronous: &Run) {
    assert_eq!(sync.0.len(), asynchronous.0.len(), "{label}: frame count");
    for (a, b) in sync.0.iter().zip(&asynchronous.0) {
        let at = format!("{label}, frame {}", a.index);
        assert_eq!(a.index, b.index, "{at}");
        assert_eq!(a.matrix, b.matrix, "{at}: matrix");
        assert_eq!(a.fallback, b.fallback, "{at}: fallback");
        assert_eq!(a.coverage, b.coverage, "{at}: coverage");
        assert_eq!(a.common, b.common, "{at}: common coverage");
        assert_eq!(
            plane_bits(&a.frame.luma),
            plane_bits(&b.frame.luma),
            "{at}: luma"
        );
        assert_eq!(
            plane_bits(&a.frame.chroma_a),
            plane_bits(&b.frame.chroma_a),
            "{at}: chroma a"
        );
        assert_eq!(
            plane_bits(&a.frame.chroma_b),
            plane_bits(&b.frame.chroma_b),
            "{at}: chroma b"
        );
    }
    assert_eq!(sync.1.fallbacks, asynchronous.1.fallbacks, "{label}");
    assert_eq!(
        sync.1.coverage.mask(),
        asynchronous.1.coverage.mask(),
        "{label}: final mask"
    );
    assert_eq!(
        sync.1.coverage.worst(),
        asynchronous.1.coverage.worst(),
        "{label}: worst frame"
    );
    assert_eq!(
        format!("{:?}", sync.1.coverage.rect()),
        format!("{:?}", asynchronous.1.coverage.rect()),
        "{label}: crop rectangle"
    );
}

fn run_sync(
    frames: &[PlanarImage<f32>],
    settings: &AlignmentSettings,
    compute: Option<&mut Compute>,
) -> Run {
    let mut flow = Flow::new(frames);
    let outcome = match compute {
        Some(compute) => align_sequence_on(
            &frames[0],
            frames.len(),
            settings,
            &NeverCancel,
            &NoProgress,
            &mut flow,
            compute,
        ),
        None => align_sequence(
            &frames[0],
            frames.len(),
            settings,
            &NeverCancel,
            &NoProgress,
            &mut flow,
        ),
    }
    .expect("the synchronous run succeeds");
    (flow.out, outcome)
}

fn run_async(
    frames: &[PlanarImage<f32>],
    settings: &AlignmentSettings,
    backend: &mut AsyncMock,
    cpu_allowed: bool,
    cancel: &dyn Cancel,
) -> Result<Run, AsyncSequenceError<String>> {
    let mut flow = Flow::new(frames);
    let outcome = block_on(align_sequence_async(
        &frames[0],
        frames.len(),
        settings,
        cancel,
        &NoProgress,
        &mut flow,
        backend,
        cpu_allowed,
    ))?;
    Ok((flow.out, outcome))
}

/// Poll a future to completion on this thread; the mocks never return `Pending`.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

// ── a scripted device, for both forms ─────────────────────────────────────────

/// Moments of a made-up objective with its minimum near a fixed warp.
fn mock_moments(inv: &Matrix3<f32>, w: usize, h: usize) -> Moments {
    let n = (w * h) as u64;
    let tx = f64::from(inv[(0, 2)]) / w as f64;
    let ty = f64::from(inv[(1, 2)]) / h as f64;
    let cost = (tx - 0.012).powi(2)
        + (ty + 0.008).powi(2)
        + f64::from(inv[(0, 0)] - 1.01).powi(2)
        + f64::from(inv[(1, 1)] - 0.995).powi(2);
    Moments {
        count: n,
        mean: 0.01,
        m2: (cost + 1.0e-6) * (n as f64 - 1.0),
    }
}

type Log = Arc<Mutex<Vec<[u32; 9]>>>;

/// What the device does: accept levels (all but those of width `decline_width`),
/// fail every `fail_every`-th evaluation of a level (0 = never).
#[derive(Clone)]
struct Script {
    accept: bool,
    decline_width: usize,
    fail_every: u32,
    log: Log,
}

struct ScriptLevel {
    w: usize,
    h: usize,
    calls: u32,
    fail_every: u32,
    log: Log,
}

impl Script {
    fn new(accept: bool, decline_width: usize, fail_every: u32) -> Self {
        Self {
            accept,
            decline_width,
            fail_every,
            log: Arc::default(),
        }
    }

    fn open(&self, w: usize, h: usize) -> Option<ScriptLevel> {
        (self.accept && w != self.decline_width).then(|| ScriptLevel {
            w,
            h,
            calls: 0,
            fail_every: self.fail_every,
            log: Arc::clone(&self.log),
        })
    }

    fn evaluations(&self) -> Vec<[u32; 9]> {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl ScriptLevel {
    fn answer(&mut self, inverse: &Matrix3<f32>) -> Option<Moments> {
        self.calls += 1;
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(std::array::from_fn(|i| inverse[(i / 3, i % 3)].to_bits()));
        let fails = self.fail_every != 0 && self.calls.is_multiple_of(self.fail_every);
        (!fails).then(|| mock_moments(inverse, self.w, self.h))
    }
}

impl ObjectiveDevice for Script {
    fn level(
        &self,
        _reference: &[f32],
        _source: &[f32],
        width: usize,
        height: usize,
        _sampler: ObjectiveSampler,
    ) -> Option<Box<dyn LevelDevice>> {
        self.open(width, height)
            .map(|l| Box::new(l) as Box<dyn LevelDevice>)
    }
}

impl LevelDevice for ScriptLevel {
    fn moments(&mut self, inverse: &Matrix3<f32>) -> Option<Moments> {
        self.answer(inverse)
    }
}

/// The synchronous side: the scripted device as a `Compute` backend. It has no
/// warp kernel, so `Compute` warps on the CPU.
struct SyncMock {
    script: Script,
}

impl Backend for SyncMock {
    fn name(&self) -> String {
        "scripted".to_owned()
    }

    fn apex(
        &mut self,
        _images: &[PlanarImage<f32>],
        _settings: &StackingSettings,
    ) -> Result<PlanarImage<f32>, BackendError> {
        Err(BackendError::unsupported("Apex"))
    }

    fn alignment_device(&mut self) -> Option<DeviceHandle> {
        Some(DeviceHandle(Arc::new(self.script.clone())))
    }
}

/// The asynchronous side: the same script, and a CPU warp (or none).
struct AsyncMock {
    script: Script,
    level: Option<ScriptLevel>,
    warp_ok: bool,
}

impl AsyncMock {
    const fn new(script: Script, warp_ok: bool) -> Self {
        Self {
            script,
            level: None,
            warp_ok,
        }
    }
}

impl AsyncLevelDevice for AsyncMock {
    fn level(
        &mut self,
        _reference: &[f32],
        _source: &[f32],
        width: usize,
        height: usize,
        _sampler: ObjectiveSampler,
    ) -> impl Future<Output = bool> {
        self.level = self.script.open(width, height);
        std::future::ready(self.level.is_some())
    }

    fn moments(&mut self, inverse: &Matrix3<f32>) -> impl Future<Output = Option<Moments>> {
        std::future::ready(self.level.as_mut().and_then(|level| level.answer(inverse)))
    }
}

impl AsyncAlignBackend for AsyncMock {
    fn warp(
        &mut self,
        frame: &PlanarImage<f32>,
        apply: &Matrix3<f32>,
        edge_clamped: bool,
    ) -> impl Future<Output = Option<(PlanarImage<f32>, Vec<bool>)>> {
        std::future::ready(self.warped(frame, apply, edge_clamped))
    }
}

impl AsyncMock {
    /// The CPU warp and its coverage mask, or `None` when the device is set to fail.
    fn warped(
        &self,
        frame: &PlanarImage<f32>,
        apply: &Matrix3<f32>,
        edge_clamped: bool,
    ) -> Option<(PlanarImage<f32>, Vec<bool>)> {
        if !self.warp_ok {
            return None;
        }
        let warped = if edge_clamped {
            warp_image_clamped(frame, apply)
        } else {
            warp_image(frame, apply)
        }
        .ok()?;
        Some((warped, coverage_mask(apply, frame.width, frame.height)))
    }
}

// ── the tests ─────────────────────────────────────────────────────────────────

/// A device that takes nothing: every level and every warp is the CPU's, so the
/// asynchronous run is the plain synchronous one.
#[test]
fn async_run_on_the_cpu_matches_align_sequence_in_every_mode() {
    let frames = stack();
    for (mode, bounded) in MODES {
        let s = settings(mode, bounded);
        let sync = run_sync(&frames, &s, None);
        for warp_ok in [true, false] {
            let mut backend = AsyncMock::new(Script::new(false, usize::MAX, 0), warp_ok);
            let got = run_async(&frames, &s, &mut backend, true, &NeverCancel)
                .unwrap_or_else(|_| panic!("{mode:?} bounded={bounded}: the run fails"));
            assert_same_run(
                &format!("{mode:?} bounded={bounded} warp_ok={warp_ok}"),
                &sync,
                &got,
            );
        }
    }
}

/// A scripted device that declines the middle level and fails every eleventh
/// evaluation, given to the synchronous run as its backend and to the
/// asynchronous run as its device: same frames, same evaluations.
#[test]
fn async_run_with_a_scripted_device_matches_the_sync_run_on_the_same_device() {
    let frames = stack();
    for (mode, bounded) in MODES {
        let s = settings(mode, bounded);

        let script = Script::new(true, W / 2, 11);
        let mut compute = Compute::with_backend(Box::new(SyncMock {
            script: script.clone(),
        }));
        let sync = run_sync(&frames, &s, Some(&mut compute));
        assert!(
            compute.uses_gpu(),
            "{mode:?}: the scripted backend must stay in use"
        );
        let sync_log = script.evaluations();

        let script = Script::new(true, W / 2, 11);
        let mut backend = AsyncMock::new(script.clone(), true);
        let got = run_async(&frames, &s, &mut backend, true, &NeverCancel)
            .unwrap_or_else(|_| panic!("{mode:?} bounded={bounded}: the run fails"));
        let label = format!("{mode:?} bounded={bounded}");
        assert_same_run(&label, &sync, &got);
        assert_eq!(sync_log, script.evaluations(), "{label}: evaluations");
        if mode != AlignmentModeSetting::None {
            assert!(!sync_log.is_empty(), "{label}: the device was never asked");
        }
    }
}

/// The browser rule: a device that cannot do the work is an error, not a CPU run.
#[test]
fn without_the_cpu_a_refusing_device_ends_the_run_with_a_clear_error() {
    let frames = stack();
    let s = settings(AlignmentModeSetting::Affine, true);
    let cases: [(&str, Script, bool); 3] = [
        ("level declined", Script::new(false, usize::MAX, 0), true),
        ("evaluation failed", Script::new(true, usize::MAX, 1), true),
        ("warp failed", Script::new(true, usize::MAX, 0), false),
    ];
    for (label, script, warp_ok) in cases {
        let mut backend = AsyncMock::new(script, warp_ok);
        match run_async(&frames, &s, &mut backend, false, &NeverCancel) {
            Err(AsyncSequenceError::Device(error)) => {
                let text = error.to_string();
                assert!(text.contains("GPU"), "{label}: {text}");
            }
            Err(AsyncSequenceError::Cancelled | AsyncSequenceError::Flow(_)) => {
                panic!("{label}: wrong error")
            }
            Ok(_) => panic!("{label}: the run must fail"),
        }
    }
}

/// A device that works carries a GPU-only run through, and a run that does not
/// align at all never asks the device.
#[test]
fn without_the_cpu_a_working_device_succeeds_and_none_mode_needs_no_device() {
    let frames = stack();
    let s = settings(AlignmentModeSetting::Translation, true);
    let mut backend = AsyncMock::new(Script::new(true, usize::MAX, 0), true);
    let run = run_async(&frames, &s, &mut backend, false, &NeverCancel)
        .unwrap_or_else(|_| panic!("a working device must carry the run"));
    assert_eq!(run.0.len(), frames.len() - 1);
    assert_eq!(run.1.fallbacks, []);

    let s = settings(AlignmentModeSetting::None, false);
    let mut refusing = AsyncMock::new(Script::new(false, usize::MAX, 1), false);
    let run = run_async(&frames, &s, &mut refusing, false, &NeverCancel)
        .unwrap_or_else(|_| panic!("mode none never touches the device"));
    assert_eq!(run.0.len(), frames.len() - 1);
    assert_eq!(refusing.script.evaluations(), [[0_u32; 9]; 0]);
}

struct CancelNow;

impl Cancel for CancelNow {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn a_cancelled_run_stops_before_the_first_frame() {
    let frames = stack();
    let s = settings(AlignmentModeSetting::Translation, false);
    let mut backend = AsyncMock::new(Script::new(true, usize::MAX, 0), true);
    let result = run_async(&frames, &s, &mut backend, false, &CancelNow);
    assert!(matches!(result, Err(AsyncSequenceError::Cancelled)));
}
