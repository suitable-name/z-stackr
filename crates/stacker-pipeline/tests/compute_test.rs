#![expect(
    clippy::suboptimal_flops,
    reason = "synthetic-image formulas read better without mul_add"
)]

//! The fallback rules of `compute::Compute`, tested with backends that behave (or
//! misbehave) on command, so no GPU is needed; plus, in a `gpu` build, the real
//! device with faults injected into it.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use stacker_algo::{relief::multigrid::multigrid_depth, strata::StrataKernels};
use stacker_align::{
    Matrix3,
    transform::{coverage_mask, warp_image},
};
use stacker_core::{
    image::PlanarImage,
    settings::{ComputeDevice, StackingSettings},
};
use stacker_pipeline::{
    compute::{Backend, BackendError, Compute},
    fuse::{apex, relief, relief_threshold, relief_tile},
};

fn textured(seed: u32) -> PlanarImage<f32> {
    let mut img = PlanarImage::new(40, 32);
    for i in 0..40 * 32_u32 {
        let h = (i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) % 1000;
        let n = f32::from(u16::try_from(h).unwrap()) / 1000.0 - 0.5;
        let idx = usize::try_from(i).unwrap();
        img.luma[idx] = 0.5 + 0.3 * n;
        img.chroma_a[idx] = 0.05 * n;
    }
    img
}

fn frames() -> Vec<PlanarImage<f32>> {
    vec![textured(1), textured(2), textured(3)]
}

fn same(a: &PlanarImage<f32>, b: &PlanarImage<f32>) -> bool {
    a.luma == b.luma && a.chroma_a == b.chroma_a && a.chroma_b == b.chroma_b
}

/// What a scripted backend does when asked to fuse.
#[derive(Clone)]
enum Behaviour {
    Succeed(PlanarImage<f32>),
    DoesNotFit,
    Fault,
    Panic,
    NonFinite,
}

struct Scripted {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
}

impl Backend for Scripted {
    fn name(&self) -> String {
        "scripted GPU".to_owned()
    }

    fn apex(
        &mut self,
        images: &[PlanarImage<f32>],
        _settings: &StackingSettings,
    ) -> Result<PlanarImage<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match &self.behaviour {
            Behaviour::Succeed(image) => Ok(image.clone()),
            Behaviour::DoesNotFit => Err(BackendError::DoesNotFit("too big".to_owned())),
            Behaviour::Fault => Err(BackendError::Fault("device lost".to_owned())),
            Behaviour::Panic => panic!("shader exploded"),
            Behaviour::NonFinite => {
                let mut bad = images[0].clone();
                bad.luma[7] = f32::NAN;
                Ok(bad)
            }
        }
    }

    fn relief_fold(
        &mut self,
        images: &[PlanarImage<f32>],
        _estimation_radius: usize,
    ) -> Result<Vec<f32>, BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let n = images[0].width * images[0].height;
        match &self.behaviour {
            Behaviour::Succeed(_) => Ok(vec![1.0; n]),
            Behaviour::DoesNotFit => Err(BackendError::DoesNotFit("too big".to_owned())),
            Behaviour::Fault => Err(BackendError::Fault("device lost".to_owned())),
            Behaviour::Panic => panic!("shader exploded"),
            Behaviour::NonFinite => Ok(vec![f32::NAN; n]),
        }
    }

    fn relief_finish(
        &mut self,
        _threshold: f32,
        _smoothing_radius: usize,
    ) -> Result<PlanarImage<f32>, BackendError> {
        match &self.behaviour {
            Behaviour::Succeed(image) => Ok(image.clone()),
            _ => Err(BackendError::Fault("finish failed".to_owned())),
        }
    }

    fn warp(
        &mut self,
        frame: &PlanarImage<f32>,
        _inverse: &Matrix3<f32>,
        _clamped: bool,
    ) -> Result<(PlanarImage<f32>, Vec<bool>), BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let n = frame.width * frame.height;
        match &self.behaviour {
            Behaviour::Succeed(image) => Ok((image.clone(), vec![true; n])),
            Behaviour::DoesNotFit => Err(BackendError::DoesNotFit("too big".to_owned())),
            Behaviour::Fault => Err(BackendError::Fault("device lost".to_owned())),
            Behaviour::Panic => panic!("shader exploded"),
            Behaviour::NonFinite => {
                let mut bad = frame.clone();
                bad.chroma_b[3] = f32::INFINITY;
                Ok((bad, vec![true; n]))
            }
        }
    }

    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
}

fn scripted(behaviour: Behaviour) -> (Compute, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let resets = Arc::new(AtomicUsize::new(0));
    let compute = Compute::with_backend(Box::new(Scripted {
        behaviour,
        calls: Arc::clone(&calls),
        resets: Arc::clone(&resets),
    }));
    (compute, calls, resets)
}

#[test]
fn a_working_backend_supplies_the_result() {
    let settings = StackingSettings::default();
    let marker = textured(99);
    let (mut compute, calls, _) = scripted(Behaviour::Succeed(marker.clone()));
    assert!(compute.uses_gpu());
    assert_eq!(compute.describe(), "scripted GPU");
    let out = compute.apex(&frames(), &settings);
    assert!(same(&out, &marker), "the backend's image is returned as is");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(compute.take_notice().is_none());
}

#[test]
fn a_stack_that_does_not_fit_goes_to_the_cpu_and_the_backend_stays() {
    let settings = StackingSettings::default();
    let (mut compute, calls, resets) = scripted(Behaviour::DoesNotFit);
    let expected = apex(&frames(), &settings);
    for _ in 0..2 {
        let out = compute.apex(&frames(), &settings);
        assert!(same(&out, &expected), "the CPU result");
    }
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "tried again: smaller calls may fit"
    );
    assert_eq!(resets.load(Ordering::SeqCst), 0);
    assert!(compute.uses_gpu());
    assert!(compute.take_notice().is_none(), "not a failure, no warning");
}

#[test]
fn a_fault_gives_up_the_backend_for_the_rest_of_the_run() {
    let settings = StackingSettings::default();
    let (mut compute, calls, resets) = scripted(Behaviour::Fault);
    let expected = apex(&frames(), &settings);

    let first = compute.apex(&frames(), &settings);
    assert!(same(&first, &expected));
    let notice = compute.take_notice().expect("a notice says what happened");
    assert!(notice.contains("device lost"), "{notice}");
    assert!(
        compute.take_notice().is_none(),
        "the notice is delivered once"
    );

    let second = compute.apex(&frames(), &settings);
    assert!(same(&second, &expected));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the backend is not asked again"
    );
    assert_eq!(resets.load(Ordering::SeqCst), 1, "it was reset once");
    assert!(!compute.uses_gpu());
    assert_eq!(compute.describe(), "CPU");
    assert!(compute.take_notice().is_none());
}

#[test]
fn a_panic_in_the_backend_is_caught_and_handled_like_a_fault() {
    let settings = StackingSettings::default();
    let (mut compute, calls, _) = scripted(Behaviour::Panic);
    let expected = apex(&frames(), &settings);
    let out = compute.apex(&frames(), &settings);
    assert!(same(&out, &expected));
    let notice = compute.take_notice().expect("a notice");
    assert!(notice.contains("shader exploded"), "{notice}");
    assert!(!compute.uses_gpu());
    let _ = compute.apex(&frames(), &settings);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn a_result_with_non_finite_pixels_is_not_trusted() {
    let settings = StackingSettings::default();
    let (mut compute, _, _) = scripted(Behaviour::NonFinite);
    let expected = apex(&frames(), &settings);
    let out = compute.apex(&frames(), &settings);
    assert!(same(&out, &expected), "the CPU result replaces the bad one");
    let notice = compute.take_notice().expect("a notice");
    assert!(notice.contains("non-finite"), "{notice}");
    assert!(!compute.uses_gpu());
}

#[test]
fn an_empty_stack_never_reaches_the_backend() {
    let settings = StackingSettings::default();
    let (mut compute, calls, _) = scripted(Behaviour::Fault);
    let out = compute.apex(&[], &settings);
    assert_eq!((out.width, out.height), (0, 0));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(compute.uses_gpu(), "nothing went wrong");
}

#[test]
fn the_cpu_setting_is_the_cpu_reference() {
    let settings = StackingSettings::default();
    assert_eq!(settings.compute_device, ComputeDevice::Cpu);
    let mut compute = Compute::for_settings(&settings);
    assert!(!compute.uses_gpu());
    assert_eq!(compute.describe(), "CPU");
    assert!(
        compute.take_notice().is_none(),
        "asked for the CPU, got the CPU"
    );
    let out = compute.apex(&frames(), &settings);
    assert!(same(&out, &apex(&frames(), &settings)));
}

/// The `Relief` with the settings' threshold, through `Compute::relief`.
fn relief_through(compute: &mut Compute, settings: &StackingSettings) -> Option<PlanarImage<f32>> {
    compute.relief(&frames(), settings, |map| {
        Some(relief_threshold(&map.luma, settings).value)
    })
}

/// Declines every stage but the two `Relief` steps; the finish faults.
struct FoldOnly;

impl Backend for FoldOnly {
    fn name(&self) -> String {
        "fold only".to_owned()
    }

    fn apex(
        &mut self,
        _images: &[PlanarImage<f32>],
        _settings: &StackingSettings,
    ) -> Result<PlanarImage<f32>, BackendError> {
        Err(BackendError::unsupported("Apex"))
    }

    fn relief_fold(
        &mut self,
        images: &[PlanarImage<f32>],
        _radius: usize,
    ) -> Result<Vec<f32>, BackendError> {
        Ok(vec![0.5; images[0].width * images[0].height])
    }

    fn relief_finish(&mut self, _t: f32, _s: usize) -> Result<PlanarImage<f32>, BackendError> {
        Err(BackendError::Fault("finish lost the device".to_owned()))
    }
}

/// Reports an alignment-device fault once when built with `true`.
struct AlignmentFault(bool);

impl Backend for AlignmentFault {
    fn name(&self) -> String {
        "alignment fault".to_owned()
    }

    fn apex(
        &mut self,
        _images: &[PlanarImage<f32>],
        _settings: &StackingSettings,
    ) -> Result<PlanarImage<f32>, BackendError> {
        Err(BackendError::unsupported("Apex"))
    }

    fn take_alignment_fault(&mut self) -> Option<String> {
        std::mem::take(&mut self.0).then(|| "objective lost the device".to_owned())
    }
}

#[test]
fn relief_falls_back_like_apex() {
    let settings = StackingSettings::default();
    let expected = relief(&frames(), &settings);

    // Not fitting: the CPU result, the backend stays, no notice.
    let (mut compute, calls, _) = scripted(Behaviour::DoesNotFit);
    let out = relief_through(&mut compute, &settings).expect("not cancelled");
    assert!(same(&out, &expected));
    let tile = compute.relief_tile(&frames(), &settings, 0.0);
    assert!(same(&tile, &relief_tile(&frames(), &settings, 0.0)));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(compute.uses_gpu() && compute.take_notice().is_none());

    // A fault, a panic, a non-finite focus map: the CPU result, the GPU given up.
    for behaviour in [Behaviour::Fault, Behaviour::Panic, Behaviour::NonFinite] {
        let (mut compute, _, resets) = scripted(behaviour);
        let out = relief_through(&mut compute, &settings).expect("not cancelled");
        assert!(same(&out, &expected));
        assert!(!compute.uses_gpu());
        assert!(compute.take_notice().is_some());
        assert_eq!(resets.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn relief_asks_for_the_threshold_once_and_passes_a_cancel_on() {
    let settings = StackingSettings::default();
    for behaviour in [Behaviour::Succeed(textured(7)), Behaviour::Fault] {
        let (mut compute, _, _) = scripted(behaviour);
        let mut asked = 0;
        let out = compute.relief(&frames(), &settings, |map| {
            asked += 1;
            assert_eq!(map.luma.len(), 40 * 32);
            None
        });
        assert!(out.is_none(), "a cancel is passed on");
        assert_eq!(asked, 1);
    }
    // The backend's result is used when it delivers.
    let marker = textured(7);
    let (mut compute, _, _) = scripted(Behaviour::Succeed(marker.clone()));
    let out = relief_through(&mut compute, &settings).expect("not cancelled");
    assert!(same(&out, &marker));
    assert!(same(
        &compute.relief_tile(&frames(), &settings, 0.3),
        &marker
    ));
}

#[test]
fn a_finish_that_fails_after_the_threshold_was_chosen_keeps_that_threshold() {
    // The threshold chosen from the device focus map is reused by the CPU; the
    // user is not asked again.
    let settings = StackingSettings::default();
    let mut compute = Compute::with_backend(Box::new(FoldOnly));
    let mut asked = 0;
    let out = compute
        .relief(&frames(), &settings, |_| {
            asked += 1;
            Some(0.25)
        })
        .expect("not cancelled");
    assert_eq!(asked, 1);
    assert!(same(&out, &relief_tile(&frames(), &settings, 0.25)));
    assert!(!compute.uses_gpu());
}

#[test]
fn the_warp_falls_back_to_the_cpu_warp_and_mask() {
    let frame = textured(4);
    let matrix = Matrix3::new(1.0, 0.0, 2.5, 0.0, 1.0, -1.25, 0.0, 0.0, 1.0);
    let cpu = warp_image(&frame, &matrix).expect("warp");
    let mask = coverage_mask(&matrix, frame.width, frame.height);
    for (behaviour, keeps_gpu) in [
        (Behaviour::DoesNotFit, true),
        (Behaviour::Fault, false),
        (Behaviour::Panic, false),
        (Behaviour::NonFinite, false),
    ] {
        let (mut compute, _, _) = scripted(behaviour);
        let (out, coverage) = compute.warp(&frame, &matrix, false).expect("warp");
        assert!(same(&out, &cpu));
        assert_eq!(coverage, mask);
        assert_eq!(compute.uses_gpu(), keeps_gpu);
    }
    // A singular matrix is the CPU warp's error, and never reaches the backend.
    let (mut compute, calls, _) = scripted(Behaviour::Fault);
    assert!(compute.warp(&frame, &Matrix3::zeros(), false).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(compute.uses_gpu());
}

#[test]
fn an_alignment_device_fault_gives_the_backend_up() {
    let mut compute = Compute::with_backend(Box::new(AlignmentFault(true)));
    assert!(compute.alignment_device().is_none(), "the default has none");
    compute.check_alignment_device();
    assert!(!compute.uses_gpu());
    let notice = compute.take_notice().expect("a notice");
    assert!(notice.contains("objective lost the device"), "{notice}");

    let mut healthy = Compute::with_backend(Box::new(AlignmentFault(false)));
    healthy.check_alignment_device();
    assert!(healthy.uses_gpu());
}

/// Pins for the multigrid tests: about one pixel in nine is pinned to a frame
/// index below 40, from a fixed pseudo-random sequence.
fn pins(width: usize, height: usize) -> (Vec<u16>, Vec<f32>) {
    let mut state = 0x2468_ACE1_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let index = (0..width * height)
        .map(|_| u16::try_from(next() % 40).expect("below 40"))
        .collect();
    let weight = (0..width * height)
        .map(|_| if next().is_multiple_of(9) { 1.0 } else { 0.0 })
        .collect();
    (index, weight)
}

/// A backend that serves only the multigrid solver, as scripted.
struct MultigridScript(Behaviour);

impl Backend for MultigridScript {
    fn name(&self) -> String {
        "multigrid script".to_owned()
    }

    fn apex(
        &mut self,
        _images: &[PlanarImage<f32>],
        _settings: &StackingSettings,
    ) -> Result<PlanarImage<f32>, BackendError> {
        Err(BackendError::unsupported("Apex"))
    }

    fn multigrid_depth(
        &mut self,
        width: usize,
        height: usize,
        _index: &[u16],
        _weight: &[f32],
        _smoothing_radius: usize,
    ) -> Result<Vec<f32>, BackendError> {
        match &self.0 {
            Behaviour::Succeed(_) => Ok(vec![3.0; width * height]),
            Behaviour::DoesNotFit => Err(BackendError::DoesNotFit("too big".to_owned())),
            Behaviour::Fault => Err(BackendError::Fault("device lost".to_owned())),
            Behaviour::Panic => panic!("shader exploded"),
            Behaviour::NonFinite => Ok(vec![f32::NAN; width * height]),
        }
    }
}

#[test]
fn the_multigrid_falls_back_like_the_other_stages() {
    let (w, h) = (31, 22);
    let (index, weight) = pins(w, h);
    let expected = multigrid_depth(w, h, &index, &weight, 3);

    // A backend that delivers: its depth, no notice.
    let mut compute =
        Compute::with_backend(Box::new(MultigridScript(Behaviour::Succeed(textured(1)))));
    let depth = compute.multigrid_depth(w, h, &index, &weight, 3);
    assert_eq!(depth, vec![3.0; w * h]);
    assert!(compute.uses_gpu() && compute.take_notice().is_none());

    // Not fitting: the CPU depth, the backend stays, no warning.
    let mut compute = Compute::with_backend(Box::new(MultigridScript(Behaviour::DoesNotFit)));
    assert_eq!(compute.multigrid_depth(w, h, &index, &weight, 3), expected);
    assert!(compute.uses_gpu() && compute.take_notice().is_none());

    // A fault, a panic, non-finite values: the CPU depth, the GPU given up.
    for behaviour in [Behaviour::Fault, Behaviour::Panic, Behaviour::NonFinite] {
        let mut compute = Compute::with_backend(Box::new(MultigridScript(behaviour)));
        assert_eq!(compute.multigrid_depth(w, h, &index, &weight, 3), expected);
        assert!(!compute.uses_gpu());
        assert!(compute.take_notice().is_some());
    }

    // A backend without the stage declines it; so does the CPU handle.
    let mut compute = Compute::with_backend(Box::new(AlignmentFault(false)));
    assert_eq!(compute.multigrid_depth(w, h, &index, &weight, 3), expected);
    assert!(compute.uses_gpu(), "declining is not a fault");
    let mut cpu = Compute::cpu();
    assert_eq!(cpu.multigrid_depth(w, h, &index, &weight, 3), expected);
    assert_eq!(cpu.multigrid_depth(0, 0, &[], &[], 3), [] as [f32; 0]);
}

/// Serves `Strata`'s saliency with a marker value and reports a fault once when
/// built with a message.
struct StrataScript(Option<String>);

struct MarkerKernels;

impl StrataKernels for MarkerKernels {
    fn saliency(&mut self, _luma: &[f32], width: usize, height: usize) -> Option<Vec<f32>> {
        Some(vec![2.0; width * height])
    }
}

impl Backend for StrataScript {
    fn name(&self) -> String {
        "strata script".to_owned()
    }

    fn apex(
        &mut self,
        _images: &[PlanarImage<f32>],
        _settings: &StackingSettings,
    ) -> Result<PlanarImage<f32>, BackendError> {
        Err(BackendError::unsupported("Apex"))
    }

    fn strata_kernels(&mut self) -> Option<Box<dyn StrataKernels>> {
        Some(Box::new(MarkerKernels))
    }

    fn take_strata_fault(&mut self) -> Option<String> {
        self.0.take()
    }
}

#[test]
fn the_strata_kernels_come_from_the_backend_or_are_the_cpu_ones() {
    let luma = vec![0.5_f32; 6 * 4];

    // The CPU handle and a backend without kernels hand out the CPU kernels,
    // which serve nothing (the caller then runs the CPU code).
    for mut compute in [
        Compute::cpu(),
        Compute::with_backend(Box::new(AlignmentFault(false))),
    ] {
        let mut kernels = compute.strata_kernels();
        assert!(kernels.saliency(&luma, 6, 4).is_none());
        assert!(
            kernels
                .guided_pair(&luma, &luma, 6, 4, 2, 0.1, 1, 0.1)
                .is_none()
        );
        compute.check_strata_kernels();
        assert!(compute.take_notice().is_none());
    }

    // A backend's kernels are used, and stay usable by another thread.
    let mut compute = Compute::with_backend(Box::new(StrataScript(None)));
    let mut kernels = compute.strata_kernels();
    let served = std::thread::spawn(move || kernels.saliency(&luma, 6, 4))
        .join()
        .expect("the kernels run on another thread");
    assert_eq!(served, Some(vec![2.0; 24]));
    compute.check_strata_kernels();
    assert!(compute.uses_gpu(), "no fault, no change");
}

#[test]
fn a_strata_kernel_fault_gives_the_backend_up() {
    let mut compute = Compute::with_backend(Box::new(StrataScript(Some(
        "saliency lost the device".to_owned(),
    ))));
    let _kernels = compute.strata_kernels();
    compute.check_strata_kernels();
    assert!(!compute.uses_gpu());
    let notice = compute.take_notice().expect("a notice");
    assert!(notice.contains("saliency lost the device"), "{notice}");
    // With the GPU given up the handle hands out the CPU kernels again.
    assert!(compute.strata_kernels().saliency(&[0.0; 4], 2, 2).is_none());
}

#[cfg(not(feature = "gpu"))]
#[test]
fn asking_for_the_gpu_in_a_build_without_it_uses_the_cpu_and_says_so() {
    let settings = StackingSettings {
        compute_device: ComputeDevice::Gpu,
        ..StackingSettings::default()
    };
    let mut compute = Compute::for_settings(&settings);
    assert!(!compute.uses_gpu());
    let notice = compute.take_notice().expect("a notice");
    assert!(notice.contains("no GPU support"), "{notice}");
    let out = compute.apex(&frames(), &settings);
    assert!(same(&out, &apex(&frames(), &settings)));
}

#[cfg(feature = "gpu")]
mod real_device {
    use std::sync::{Mutex, PoisonError};

    use stacker_algo::{
        relief::guided::guided_filter_pair_luma, strata::saliency::compute_saliency,
    };
    use stacker_align::refine::LumaPyramid;
    use stacker_core::settings::AlignmentModeSetting;
    use stacker_gpu::Failure;
    use stacker_pipeline::{
        align::{AlignmentSettings, align_frame_with_compute, align_frame_with_pyramid},
        compute::shared_gpu_context,
    };

    use super::*;

    /// The tests share one process-wide device, and an armed failure must fire
    /// inside the test that armed it, so they run one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn gpu_settings() -> StackingSettings {
        StackingSettings {
            compute_device: ComputeDevice::Gpu,
            pyramid_levels: 4,
            ..StackingSettings::default()
        }
    }

    /// A GPU compute, or `None` (with a `SKIPPED` line) on a machine without one.
    fn gpu_compute(settings: &StackingSettings) -> Option<Compute> {
        let mut compute = Compute::for_settings(settings);
        if compute.uses_gpu() {
            Some(compute)
        } else {
            eprintln!("SKIPPED: {:?}", compute.take_notice());
            None
        }
    }

    #[test]
    fn the_gpu_result_agrees_with_the_cpu() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let Some(mut compute) = gpu_compute(&settings) else {
            return;
        };
        assert!(
            compute.describe().starts_with("GPU"),
            "{}",
            compute.describe()
        );
        let frames = frames();
        let gpu = compute.apex(&frames, &settings);
        assert!(compute.uses_gpu(), "no fault on a healthy device");
        let cpu = apex(&frames, &settings);
        let worst = gpu
            .luma
            .iter()
            .zip(&cpu.luma)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-3, "largest luma difference {worst}");
    }

    #[test]
    fn an_injected_device_failure_falls_back_to_the_exact_cpu_result() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let frames = frames();
        let expected = apex(&frames, &settings);

        for at in [Failure::Plan, Failure::Fold, Failure::ReadBack] {
            let Some(mut compute) = gpu_compute(&settings) else {
                return;
            };
            // Fetched per round: a fault clears the shared device, and the next
            // round opens a new one.
            let ctx = shared_gpu_context().expect("the device is open");
            ctx.inject_failure(at);
            let out = compute.apex(&frames, &settings);
            assert!(same(&out, &expected), "{at:?}: the CPU result");
            let notice = compute.take_notice().expect("a notice");
            assert!(notice.contains("injected"), "{at:?}: {notice}");
            assert!(!compute.uses_gpu(), "{at:?}: gave up the GPU");
            // The next run reopens the device and works again.
            let mut again = gpu_compute(&settings).expect("the device comes back");
            let _ = again.apex(&frames, &settings);
            assert!(again.uses_gpu(), "{at:?}: a fresh run is on the GPU again");
        }
    }

    #[test]
    fn a_stack_over_the_memory_budget_is_fused_on_the_cpu_without_giving_up_the_gpu() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        // 24 MP needs about 2 GiB of device memory, so 37 MP is over the 3 GiB
        // budget. Two frames of zeros keep the CPU side cheap enough.
        let settings = gpu_settings();
        let Some(mut compute) = gpu_compute(&settings) else {
            return;
        };
        let big = PlanarImage::<f32>::new(6_800, 5_400);
        let out = compute.apex(&[big.clone(), big], &settings);
        assert_eq!((out.width, out.height), (6_800, 5_400));
        assert!(compute.uses_gpu(), "not a fault");
        assert!(compute.take_notice().is_none());
    }

    /// A textured frame large enough that the finest pyramid levels go to the
    /// device, shifted by `(dx, dy)`.
    fn big_frame(dx: f32, dy: f32) -> PlanarImage<f32> {
        let (w, h) = (640, 400);
        let mut img = PlanarImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                #[expect(clippy::cast_precision_loss, reason = "small coordinates")]
                let (fx, fy) = (x as f32 - dx, y as f32 - dy);
                let i = y * w + x;
                img.luma[i] = 0.4
                    + 0.2 * (0.21 * fx).sin() * (0.17 * fy).cos()
                    + 0.1 * (0.05 * fx + 0.03 * fy).sin();
                img.chroma_a[i] = 0.05 * (0.09 * fx).cos();
            }
        }
        img
    }

    const fn alignment(mode: AlignmentModeSetting, bounded: bool) -> AlignmentSettings {
        AlignmentSettings::new(mode, bounded)
    }

    #[test]
    fn relief_on_the_gpu_agrees_with_the_cpu() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = StackingSettings {
            relief_contrast_pct: 0.3,
            ..gpu_settings()
        };
        let Some(mut compute) = gpu_compute(&settings) else {
            return;
        };
        let frames = frames();
        let gpu = compute
            .relief(&frames, &settings, |map| {
                Some(relief_threshold(&map.luma, &settings).value)
            })
            .expect("not cancelled");
        assert!(compute.uses_gpu(), "no fault on a healthy device");
        let cpu = relief(&frames, &settings);
        let worst = gpu
            .luma
            .iter()
            .zip(&cpu.luma)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-3, "largest luma difference {worst}");
        let tile = compute.relief_tile(&frames, &settings, 0.0);
        assert!(compute.uses_gpu());
        let cpu_tile = relief_tile(&frames, &settings, 0.0);
        let worst = tile
            .luma
            .iter()
            .zip(&cpu_tile.luma)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-3, "tile: largest luma difference {worst}");
    }

    #[test]
    fn alignment_on_the_gpu_finds_the_cpu_shift() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let reference = big_frame(0.0, 0.0);
        let moved = big_frame(2.3, -1.4);
        let pyramid = LumaPyramid::new(&reference);
        for (mode, bounded) in [
            (AlignmentModeSetting::Translation, false),
            (AlignmentModeSetting::Affine, true),
            (AlignmentModeSetting::Registration, false),
        ] {
            let Some(mut compute) = gpu_compute(&settings) else {
                return;
            };
            let a = alignment(mode, bounded);
            let cpu = align_frame_with_pyramid(&reference, &pyramid, moved.clone(), &a, None);
            let gpu = align_frame_with_compute(
                &reference,
                &pyramid,
                moved.clone(),
                &a,
                None,
                &mut compute,
            );
            assert!(compute.uses_gpu(), "{mode:?}: no fault on a healthy device");
            let (c, g) = (cpu.matrix_applied, gpu.matrix_applied);
            eprintln!(
                "{mode:?} bounded={bounded}: cpu ({}, {}) gpu ({}, {})",
                c[(0, 2)],
                c[(1, 2)],
                g[(0, 2)],
                g[(1, 2)]
            );
            for (r, col) in [(0, 2), (1, 2)] {
                assert!(
                    (c[(r, col)] - g[(r, col)]).abs() < 0.02,
                    "{mode:?}: shift differs: {c:?} vs {g:?}"
                );
            }
            for (r, col) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                assert!(
                    (c[(r, col)] - g[(r, col)]).abs() < 1e-4,
                    "{mode:?}: linear part differs: {c:?} vs {g:?}"
                );
            }
            let covered = gpu.coverage.iter().filter(|&&v| v).count();
            let cpu_covered = cpu.coverage.iter().filter(|&&v| v).count();
            assert!(
                covered.abs_diff(cpu_covered) <= 2 * 640,
                "{covered} vs {cpu_covered}"
            );
        }
    }

    #[test]
    fn an_alignment_fault_falls_back_to_the_exact_cpu_result() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let reference = big_frame(0.0, 0.0);
        let moved = big_frame(1.7, 0.6);
        let pyramid = LumaPyramid::new(&reference);
        let a = alignment(AlignmentModeSetting::Translation, false);
        let cpu = align_frame_with_pyramid(&reference, &pyramid, moved.clone(), &a, None);
        for at in [Failure::Plan, Failure::Fold, Failure::ReadBack] {
            let Some(mut compute) = gpu_compute(&settings) else {
                return;
            };
            let ctx = shared_gpu_context().expect("the device is open");
            // The first device call of the refinement fails.
            ctx.inject_failure(at);
            let out = align_frame_with_compute(
                &reference,
                &pyramid,
                moved.clone(),
                &a,
                None,
                &mut compute,
            );
            assert_eq!(out.matrix_applied, cpu.matrix_applied, "{at:?}");
            assert!(same(&out.frame, &cpu.frame), "{at:?}: the CPU warp");
            assert_eq!(out.coverage, cpu.coverage);
            assert!(!compute.uses_gpu(), "{at:?}: gave up the GPU");
            let notice = compute.take_notice().expect("a notice");
            assert!(notice.contains("injected"), "{at:?}: {notice}");
        }
    }

    /// A luma plane with edges and noise, in `0.05..0.55`.
    fn luma_plane(width: usize, height: usize) -> Vec<f32> {
        let mut state = 0x9E37_79B9_u32;
        (0..width * height)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = f32::from(u16::try_from(state % 1000).expect("below 1000")) / 1000.0;
                let (x, y) = (i % width, i / width);
                let edge = if (x / 17 + y / 13) % 2 == 0 { 0.3 } else { 0.0 };
                0.05 + edge + 0.2 * noise
            })
            .collect()
    }

    /// A soft mask in `0..=1` like the one Strata refines: blocks with noise.
    fn mask_plane(width: usize, height: usize) -> Vec<f32> {
        luma_plane(width, height)
            .iter()
            .map(|v| ((v - 0.05) * 2.0).clamp(0.0, 1.0))
            .collect()
    }

    #[test]
    fn the_strata_kernels_on_the_gpu_agree_with_the_cpu() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let Some(mut compute) = gpu_compute(&settings) else {
            return;
        };
        let mut kernels = compute.strata_kernels();
        for (w, h) in [(130_usize, 90_usize), (61, 47), (200, 33)] {
            let luma = luma_plane(w, h);
            let mask = mask_plane(w, h);

            let gpu = kernels.saliency(&luma, w, h).expect("the device serves it");
            let cpu = compute_saliency(&luma, w, h);
            let scale = cpu.iter().copied().fold(0.0_f32, f32::max);
            let worst = gpu
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("{w}x{h}: saliency worst {worst:.2e} of {scale:.2e}");
            assert!(
                worst <= 1e-5 * scale,
                "{w}x{h}: saliency differs by {worst}"
            );

            let (gpu_big, gpu_small) = kernels
                .guided_pair(&luma, &mask, w, h, 45, 0.3, 7, 1e-6)
                .expect("the device serves it");
            let (cpu_big, cpu_small) =
                guided_filter_pair_luma(&luma, &mask, w, h, 45, 0.3, 7, 1e-6);
            for (name, gpu, cpu) in [
                ("big", &gpu_big, &cpu_big),
                ("small", &gpu_small, &cpu_small),
            ] {
                let worst = gpu
                    .iter()
                    .zip(cpu)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                #[expect(clippy::cast_precision_loss, reason = "small counts")]
                let mean = gpu
                    .iter()
                    .zip(cpu)
                    .map(|(a, b)| f64::from((a - b).abs()))
                    .sum::<f64>()
                    / (w * h) as f64;
                eprintln!("{w}x{h}: guided {name} worst {worst:.2e} mean {mean:.2e}");
                assert!(worst < 1e-3, "{w}x{h} {name}: worst {worst}");
                assert!(mean < 1e-5, "{w}x{h} {name}: mean {mean}");
            }
        }
        // A frame the kernels were not built for is a new plan, not a fault.
        compute.check_strata_kernels();
        assert!(compute.uses_gpu(), "no fault on a healthy device");
    }

    #[test]
    fn a_strata_device_failure_gives_the_gpu_up_and_the_caller_runs_the_cpu() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let (w, h) = (64, 48);
        let luma = luma_plane(w, h);
        for at in [Failure::Plan, Failure::Fold, Failure::ReadBack] {
            let Some(mut compute) = gpu_compute(&settings) else {
                return;
            };
            shared_gpu_context()
                .expect("the device is open")
                .inject_failure(at);
            let mut kernels = compute.strata_kernels();
            assert!(kernels.saliency(&luma, w, h).is_none(), "{at:?}");
            // Dead from then on, even though the failure was armed once.
            assert!(
                kernels.saliency(&luma, w, h).is_none(),
                "{at:?}: stays dead"
            );
            compute.check_strata_kernels();
            assert!(!compute.uses_gpu(), "{at:?}: gave up the GPU");
            let notice = compute.take_notice().expect("a notice");
            assert!(notice.contains("injected"), "{at:?}: {notice}");
        }
    }

    #[test]
    fn the_multigrid_on_the_gpu_agrees_with_the_cpu() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let Some(mut compute) = gpu_compute(&settings) else {
            return;
        };
        // 1x1 and 1x7 have no coarse level, 64x48 is solved entirely by the tail
        // workgroup, 200x150 by dispatches on the finest level and the tail below.
        for (w, h) in [
            (1_usize, 1_usize),
            (1, 7),
            (5, 5),
            (64, 48),
            (97, 61),
            (200, 150),
        ] {
            let (index, weight) = pins(w, h);
            let cpu = multigrid_depth(w, h, &index, &weight, 3);
            let gpu = compute.multigrid_depth(w, h, &index, &weight, 3);
            assert!(compute.uses_gpu(), "{w}x{h}: no fault on a healthy device");
            assert_eq!(gpu.len(), w * h);
            let worst = gpu
                .iter()
                .zip(&cpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            #[expect(clippy::cast_precision_loss, reason = "small counts")]
            let mean = gpu
                .iter()
                .zip(&cpu)
                .map(|(a, b)| f64::from((a - b).abs()))
                .sum::<f64>()
                / (w * h) as f64;
            eprintln!("{w}x{h}: multigrid worst {worst:.2e} mean {mean:.2e}");
            assert!(worst < 1e-3, "{w}x{h}: worst {worst}");
            assert!(mean < 1e-4, "{w}x{h}: mean {mean}");
        }
        // A plan repeats itself: the same pins give the same depth.
        let (index, weight) = pins(97, 61);
        let first = compute.multigrid_depth(97, 61, &index, &weight, 3);
        let second = compute.multigrid_depth(97, 61, &index, &weight, 3);
        assert_eq!(first, second);
    }

    #[test]
    fn a_multigrid_device_failure_falls_back_to_the_exact_cpu_result() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let (w, h) = (97, 61);
        let (index, weight) = pins(w, h);
        let expected = multigrid_depth(w, h, &index, &weight, 3);
        for at in [Failure::Plan, Failure::Fold, Failure::ReadBack] {
            let Some(mut compute) = gpu_compute(&settings) else {
                return;
            };
            shared_gpu_context()
                .expect("the device is open")
                .inject_failure(at);
            let out = compute.multigrid_depth(w, h, &index, &weight, 3);
            assert_eq!(out, expected, "{at:?}: the CPU result");
            assert!(!compute.uses_gpu(), "{at:?}: gave up the GPU");
            let notice = compute.take_notice().expect("a notice");
            assert!(notice.contains("injected"), "{at:?}: {notice}");
        }
    }

    #[test]
    fn a_relief_device_failure_falls_back_to_the_exact_cpu_result() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let settings = gpu_settings();
        let frames = frames();
        let expected = relief(&frames, &settings);
        for at in [Failure::Plan, Failure::Fold, Failure::ReadBack] {
            let Some(mut compute) = gpu_compute(&settings) else {
                return;
            };
            shared_gpu_context()
                .expect("the device is open")
                .inject_failure(at);
            let out = compute
                .relief(&frames, &settings, |map| {
                    Some(relief_threshold(&map.luma, &settings).value)
                })
                .expect("not cancelled");
            assert!(same(&out, &expected), "{at:?}: the CPU result");
            assert!(!compute.uses_gpu(), "{at:?}");
        }
    }
}
