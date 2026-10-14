//! Aligning frames onto the reference: one implementation of
//! seed → refine → gate → invert → warp → brightness → coverage mask, and the
//! drivers that run it over a whole stack.
//!
//! # Direction
//!
//! The refiners return `M` mapping **reference coordinates to source
//! coordinates** (`source(M·p) ≈ reference(p)`), while
//! `warp_image(src, A)` produces `dst(p) = src(A⁻¹·p)`. Bringing a frame onto
//! the reference therefore warps with `A = M⁻¹`. [`AlignedFrame::matrix_applied`]
//! is that `A` (or the identity when the frame was left unwarped), so a coverage
//! mask built from it describes the pixels that were actually committed.
//!
//! # Mode table
//!
//! [`align_dispatch`] is the single place where an `alignment_mode` and the
//! `bounded_search` flag become an algorithm; the `optimizer` setting then picks
//! the optimiser inside it ([`AlignmentSettings::optimizer`]).
//!
//! | Mode | Motion model | Degrees of freedom | Warp |
//! |---|---|---|---|
//! | `none` | – | none | none (frame untouched, no brightness fit) |
//! | `translation` | `Motion::Translation` | shift, uniform scale | bicubic, zero border |
//! | `registration` | `Motion::Similarity` | shift, scale, rotation | bicubic, zero border |
//! | `affine` | `Motion::Affine` | shift, scale, rotation, aspect, shear (always bounded) | bicubic, zero border |
//! | `neural` | network matrix, then `Motion::Similarity` bounded (`neural_refine_classically`) or the matrix as it is | per the network | bicubic, zero border |
//!
//! With `crop_to_common_area = false` ([`AlignmentSettings::clamp_border`]) every
//! warp is the edge-clamped spline instead, so the uncropped border shows
//! replicated data rather than black.
//!
//! # Per frame
//!
//! 1. **Seed**: the AKAZE (or network, or previous-step) estimate when it is
//!    [sane](is_sane_seed), otherwise the identity.
//! 2. **Refine**: [`stacker_align::refine_motion`] (Lucas-Kanade first with
//!    `optimizer = auto`, the Nelder-Mead simplex otherwise). A failed refinement
//!    continues from the seed ([`Fallback::Seed`]).
//! 3. **Identity gate**: [`stacker_align::apply_identity_gate`] replaces a result
//!    that fits worse than doing nothing by the identity.
//! 4. **Warp** with the inverse; a matrix that cannot be inverted or a warp that
//!    fails returns the frame **unwarped with the identity**
//!    ([`Fallback::Unaligned`]). Neither fallback is an error.
//! 5. **Brightness** (`correct_brightness`): the frame's exposure is matched to
//!    the reference over the pixels the warp covered.
//!
//! # Chain alignment
//!
//! With `chain_alignment` every frame is refined against the **raw previous
//! frame** (scheme A of [`stacker_align::chain`]), warm-started with the previous
//! step; the cumulative matrix `C_k = step · C_{k-1}` warps the frame, and its
//! coverage is [`stacker_align::chain::chain_coverage`]. Costs one extra frame
//! (and its luma pyramid) of memory.
//!
//! # Device
//!
//! The `_with_compute` / `_on` variants take a [`Compute`]: with a GPU backend
//! the simplex evaluates its objective on the device (for the pyramid levels
//! large enough to gain from it) and the final warp runs there, with the CPU
//! behind both (see [`crate::compute`]). Lucas-Kanade is a CPU path. The plain
//! functions are the CPU path.
//!
//! # Look-ahead (native only)
//!
//! [`align_sequence_on`] loads up to [`lookahead_width`] frames at a time and
//! seeds them in parallel (the AKAZE detector is single-threaded, so the seeding
//! is what scales with the cores); the frames are then aligned strictly in order,
//! so the results do not depend on the thread count. [`align_sequence_overlapped`]
//! additionally decodes the next frame on a second thread while the current one
//! is aligned. See [`lookahead_extra_bytes`] for the memory this costs.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    Mutex, PoisonError,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::{collections::VecDeque, fmt, future::Future};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use stacker_align::{
    Matrix3, Motion, MotionRefine, Optimizer, apply_identity_gate,
    brightness::{BrightnessTarget, correct_brightness_masked},
    chain::{chain_coverage, compose_chain},
    device::{AsyncLevelDevice, DeviceRefused, drive},
    refine::{LumaPyramid, reduce_to_similarity, search::AlignSearch},
    refine_motion,
    transform::{coverage_mask, warp_image, warp_image_clamped},
};
use stacker_core::{
    error::StackerError,
    image::PlanarImage,
    settings::{AlignmentModeSetting, OptimizerSetting, StackingSettings},
};

use crate::{
    Cancel, PipelineError, Progress, Stage, compute::Compute, crop::CommonCoverage, fraction_of,
};

// ── Settings ──────────────────────────────────────────────────────────────────

/// A neural alignment model (`alignment_mode = "neural"`), chosen by the app.
///
/// Without the `nn` feature (and on wasm32) there is no way to build one: the
/// type exists so [`AlignmentSettings`] has the same shape in every build.
#[derive(Clone)]
pub struct AlignModel {
    #[cfg(all(feature = "nn", not(target_arch = "wasm32")))]
    entry: stacker_nn::ModelEntry,
    #[cfg(all(feature = "nn", not(target_arch = "wasm32")))]
    device: stacker_nn::InferDevice,
    #[cfg(not(all(feature = "nn", not(target_arch = "wasm32"))))]
    _unconstructible: (),
}

#[cfg(all(feature = "nn", not(target_arch = "wasm32")))]
impl AlignModel {
    /// The alignment model `entry` (see `stacker_nn::discover_default_alignment_models`)
    /// run on `device`.
    #[must_use]
    pub const fn new(entry: stacker_nn::ModelEntry, device: stacker_nn::InferDevice) -> Self {
        Self { entry, device }
    }

    /// The discovered model.
    #[must_use]
    pub const fn entry(&self) -> &stacker_nn::ModelEntry {
        &self.entry
    }

    /// The inference device.
    #[must_use]
    pub const fn device(&self) -> stacker_nn::InferDevice {
        self.device
    }
}

impl AlignModel {
    /// A string that changes whenever the model or its device does (empty without
    /// the `nn` feature), for a cache key such as the GUI's stale-alignment
    /// fingerprint.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "the native build formats the model's name and path"
    )]
    pub fn fingerprint(&self) -> String {
        #[cfg(all(feature = "nn", not(target_arch = "wasm32")))]
        {
            format!(
                "{}|{}|{}",
                self.entry.name,
                self.entry.weights_path.display(),
                self.device.label()
            )
        }
        #[cfg(not(all(feature = "nn", not(target_arch = "wasm32"))))]
        {
            String::new()
        }
    }
}

impl fmt::Debug for AlignModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignModel")
            .field("fingerprint", &self.fingerprint())
            .finish()
    }
}

impl PartialEq for AlignModel {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint() == other.fingerprint()
    }
}

/// The settings that decide how a frame is aligned.
#[derive(Debug, Clone, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per alignment setting of the settings file, which has the same flags"
)]
pub struct AlignmentSettings {
    /// `alignment_mode`.
    pub mode: AlignmentModeSetting,
    /// `bounded_search`: the simplex searches in the logit-bounded space
    /// (`affine` is always bounded).
    pub bounded: bool,
    /// `akaze_seeding`: seed the refinement from feature matches (only with the
    /// `akaze` feature; ignored with a warning otherwise).
    pub akaze_seeding: bool,
    /// `optimizer`: Lucas-Kanade, Nelder-Mead or both (LK first). The
    /// asynchronous (browser) driver always runs the simplex.
    pub optimizer: OptimizerSetting,
    /// `correct_brightness`: match every aligned frame's exposure to the
    /// reference.
    pub correct_brightness: bool,
    /// `chain_alignment`: align every frame to the raw previous one.
    pub chain_alignment: bool,
    /// `neural_refine_classically`: with `alignment_mode = "neural"`, the
    /// network's matrices seed the classical refinement (`true`) or are applied
    /// as they are (`false`).
    pub neural_refine_classically: bool,
    /// Warp with the edge-clamped spline (replicated border) instead of the
    /// bicubic zero border. `From<&StackingSettings>` sets it when
    /// `crop_to_common_area` is `false`, because then the border stays in the
    /// result.
    pub clamp_border: bool,
    /// The network of `alignment_mode = "neural"`. Without it (or without the
    /// `nn` feature) the neural mode falls back to `registration` with a
    /// warning; the app should refuse the combination before the run.
    pub align_model: Option<AlignModel>,
}

impl AlignmentSettings {
    /// The three classic switches and nothing else: the simplex optimiser, no
    /// brightness fit, no chain, no network, bicubic zero-border warp. This is
    /// what the struct meant before it grew; [`Default`] and
    /// `From<&StackingSettings>` are the application's defaults.
    #[must_use]
    pub const fn new(mode: AlignmentModeSetting, bounded: bool) -> Self {
        Self {
            mode,
            bounded,
            akaze_seeding: false,
            optimizer: OptimizerSetting::NelderMead,
            correct_brightness: false,
            chain_alignment: false,
            neural_refine_classically: true,
            clamp_border: false,
            align_model: None,
        }
    }

    /// With `akaze_seeding` set to `on`.
    #[must_use]
    pub const fn with_akaze(mut self, on: bool) -> Self {
        self.akaze_seeding = on;
        self
    }

    /// With the optimiser `optimizer`.
    #[must_use]
    pub const fn with_optimizer(mut self, optimizer: OptimizerSetting) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// With `correct_brightness` set to `on`.
    #[must_use]
    pub const fn with_brightness(mut self, on: bool) -> Self {
        self.correct_brightness = on;
        self
    }

    /// With `chain_alignment` set to `on`.
    #[must_use]
    pub const fn with_chain(mut self, on: bool) -> Self {
        self.chain_alignment = on;
        self
    }

    /// With `neural_refine_classically` set to `on`.
    #[must_use]
    pub const fn with_neural_refinement(mut self, on: bool) -> Self {
        self.neural_refine_classically = on;
        self
    }

    /// With the edge-clamped warp switched `on` or off.
    #[must_use]
    pub const fn with_clamped_border(mut self, on: bool) -> Self {
        self.clamp_border = on;
        self
    }

    /// With the neural model `model`.
    #[must_use]
    #[allow(
        clippy::missing_const_for_fn,
        reason = "replacing the model drops the old one, which a const fn cannot do on native"
    )]
    pub fn with_model(mut self, model: Option<AlignModel>) -> Self {
        self.align_model = model;
        self
    }
}

impl From<&StackingSettings> for AlignmentSettings {
    fn from(s: &StackingSettings) -> Self {
        Self {
            mode: s.alignment_mode,
            bounded: s.bounded_search,
            akaze_seeding: s.akaze_seeding,
            optimizer: s.optimizer,
            correct_brightness: s.correct_brightness,
            chain_alignment: s.chain_alignment,
            neural_refine_classically: s.neural_refine_classically,
            clamp_border: !s.crop_to_common_area,
            align_model: None,
        }
    }
}

impl Default for AlignmentSettings {
    /// The application defaults: those of [`StackingSettings::default`].
    fn default() -> Self {
        Self::from(&StackingSettings::default())
    }
}

/// How a mode refines the seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refiner {
    /// No refinement, no warp.
    Skip,
    /// [`stacker_align::refine_motion`] with `motion`. `bounded` is the
    /// `bounded_search` flag (the `affine` motion is bounded whatever it says).
    Motion {
        /// The motion model.
        motion: Motion,
        /// Whether the simplex searches in the logit-bounded space.
        bounded: bool,
    },
    /// The seed is the answer (a network's matrix applied without refinement).
    Direct,
}

/// How the refined matrix is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warp {
    /// The frame is left as it is.
    None,
    /// Bicubic sampling, black border where the source has no data.
    Bicubic,
    /// 4-tap spline sampling with edge-clamped borders.
    EdgeClamped,
}

/// The algorithm choice for one `(mode, bounded)` pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dispatch {
    /// How the seed is refined.
    pub refiner: Refiner,
    /// How the refined matrix is applied to the frame.
    pub warp: Warp,
}

const fn motion_dispatch(motion: Motion, bounded: bool) -> Dispatch {
    Dispatch {
        refiner: Refiner::Motion { motion, bounded },
        warp: Warp::Bicubic,
    }
}

/// The mode table of the module documentation.
///
/// `neural` is the hybrid form
/// (the network's matrix seeds a bounded similarity refinement, as in GitHub
/// z-stackr); [`AlignmentSettings`] turns that into [`Refiner::Direct`] when
/// `neural_refine_classically` is off.
#[must_use]
pub const fn align_dispatch(mode: AlignmentModeSetting, bounded: bool) -> Dispatch {
    match mode {
        AlignmentModeSetting::None => Dispatch {
            refiner: Refiner::Skip,
            warp: Warp::None,
        },
        AlignmentModeSetting::Translation => motion_dispatch(Motion::Translation, bounded),
        AlignmentModeSetting::Registration => motion_dispatch(Motion::Similarity, bounded),
        AlignmentModeSetting::Affine => motion_dispatch(Motion::Affine, bounded),
        AlignmentModeSetting::Neural => motion_dispatch(Motion::Similarity, true),
    }
}

/// [`align_dispatch`] for `settings`, which also knows about the neural mode's
/// direct form.
fn dispatch_for(settings: &AlignmentSettings) -> Dispatch {
    if settings.mode == AlignmentModeSetting::Neural && !settings.neural_refine_classically {
        return Dispatch {
            refiner: Refiner::Direct,
            warp: Warp::Bicubic,
        };
    }
    align_dispatch(settings.mode, settings.bounded)
}

/// The warp the frame really gets: the edge-clamped spline when
/// [`AlignmentSettings::clamp_border`] asks for it.
const fn effective_warp(dispatch: Dispatch, settings: &AlignmentSettings) -> Warp {
    match dispatch.warp {
        Warp::Bicubic if settings.clamp_border => Warp::EdgeClamped,
        other => other,
    }
}

/// `true` for the refiners that walk the reference's luma pyramid.
const fn uses_pyramid(refiner: Refiner) -> bool {
    matches!(refiner, Refiner::Motion { .. })
}

const fn optimizer_of(setting: OptimizerSetting) -> Optimizer {
    match setting {
        OptimizerSetting::Auto => Optimizer::Auto,
        OptimizerSetting::LucasKanade => Optimizer::LucasKanade,
        OptimizerSetting::NelderMead => Optimizer::NelderMead,
    }
}

/// What a frame's alignment fell back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fallback {
    /// The refinement failed; the frame was aligned from the starting estimate.
    Seed,
    /// The frame could not be warped; it was kept as loaded.
    Unaligned,
}

/// One aligned frame.
#[derive(Debug)]
pub struct AlignedFrame {
    /// The matrix that was applied (`warp_image(frame, matrix_applied)`), the
    /// identity when the frame was not warped.
    pub matrix_applied: Matrix3<f32>,
    /// The warped frame (the input itself when it was not warped).
    pub frame: PlanarImage<f32>,
    /// Pixels of `frame` that received real data (`width × height`,
    /// row-major); all `true` for an unwarped frame.
    pub coverage: Vec<bool>,
    /// Set when the frame did not go the normal way.
    pub fallback: Option<Fallback>,
}

impl AlignedFrame {
    /// Fraction of the frame's pixels that are covered, `0.0..=1.0`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel counts are far below 2^52"
    )]
    pub fn covered_fraction(&self) -> f64 {
        if self.coverage.is_empty() {
            return 1.0;
        }
        let covered = self.coverage.iter().filter(|&&c| c).count();
        covered as f64 / self.coverage.len() as f64
    }
}

// ── One frame ─────────────────────────────────────────────────────────────────

/// What the per-frame core needs to know about the reference.
struct Ctx<'a> {
    /// The image the frame is refined against (the reference, or in chain mode
    /// the raw previous frame).
    reference: &'a PlanarImage<f32>,
    /// `reference`'s luma pyramid, when the caller has one.
    pyramid: Option<&'a LumaPyramid>,
    /// The exposure of the reference, when the brightness fit is on.
    brightness: Option<&'a BrightnessTarget>,
    /// `Some(C_{k-1})` in chain mode: the cumulative matrix of `reference`.
    previous: Option<Matrix3<f32>>,
}

/// The outcome of one frame, with the matrices a chain needs.
struct Core {
    aligned: AlignedFrame,
    /// The reference → source matrix the frame was warped with the inverse of
    /// (`C_k` in chain mode).
    cumulative: Matrix3<f32>,
    /// The refiner's matrix for this frame against `Ctx::reference`.
    step: Matrix3<f32>,
}

/// The refinement of one frame (reference → source).
struct Step {
    matrix: Matrix3<f32>,
    fallback: Option<Fallback>,
}

/// Align `frame` onto `reference`.
///
/// `seed` is the starting estimate in the refiners' convention (reference →
/// source), usually from a [`Seeder`]; it is used only when it is
/// [sane](is_sane_seed), otherwise the optimiser starts from the identity.
///
/// `frame` is taken by value so an unwarped frame comes back without a copy.
///
/// The reference's luma pyramid and brightness target are built here for this
/// one call; a loop over a stack should use [`align_frame_with_pyramid`] (or
/// [`align_sequence`], which builds them once per run). A single frame has no
/// predecessor, so `chain_alignment` has no effect here.
#[must_use]
pub fn align_frame(
    reference: &PlanarImage<f32>,
    frame: PlanarImage<f32>,
    settings: &AlignmentSettings,
    seed: Option<Matrix3<f32>>,
) -> AlignedFrame {
    align_frame_impl(reference, None, frame, settings, seed, &mut Compute::cpu())
}

/// [`align_frame`] with the reference's luma pyramid supplied by the caller
/// (`LumaPyramid::new(reference)`, built once per run). The result is bit for bit
/// the one [`align_frame`] returns.
#[must_use]
pub fn align_frame_with_pyramid(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    frame: PlanarImage<f32>,
    settings: &AlignmentSettings,
    seed: Option<Matrix3<f32>>,
) -> AlignedFrame {
    align_frame_impl(
        reference,
        Some(ref_pyramid),
        frame,
        settings,
        seed,
        &mut Compute::cpu(),
    )
}

/// [`align_frame_with_pyramid`] on `compute`: the objective and the warp run on
/// its device when it has one (the result then agrees with the CPU path within
/// the refiners' sub-pixel tolerance, not bit for bit).
#[must_use]
pub fn align_frame_with_compute(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    frame: PlanarImage<f32>,
    settings: &AlignmentSettings,
    seed: Option<Matrix3<f32>>,
    compute: &mut Compute,
) -> AlignedFrame {
    align_frame_impl(reference, Some(ref_pyramid), frame, settings, seed, compute)
}

fn align_frame_impl(
    reference: &PlanarImage<f32>,
    ref_pyramid: Option<&LumaPyramid>,
    frame: PlanarImage<f32>,
    settings: &AlignmentSettings,
    seed: Option<Matrix3<f32>>,
    compute: &mut Compute,
) -> AlignedFrame {
    let dispatch = dispatch_for(settings);
    let brightness = brightness_target(reference, settings, dispatch);
    let ctx = Ctx {
        reference,
        pyramid: ref_pyramid,
        brightness: brightness.as_ref(),
        previous: None,
    };
    align_core(&ctx, frame, settings, dispatch, seed, compute).aligned
}

/// The exposure of `reference`, when the brightness fit is on and frames are
/// aligned at all.
fn brightness_target(
    reference: &PlanarImage<f32>,
    settings: &AlignmentSettings,
    dispatch: Dispatch,
) -> Option<BrightnessTarget> {
    (settings.correct_brightness && dispatch.refiner != Refiner::Skip)
        .then(|| BrightnessTarget::from_reference(reference))
}

/// Match `warped`'s exposure to the target over the pixels `coverage` marks.
fn apply_brightness(
    warped: &mut PlanarImage<f32>,
    target: Option<&BrightnessTarget>,
    coverage: &[bool],
) {
    if let Some(target) = target {
        let fit = correct_brightness_masked(warped, target, Some(coverage));
        tracing::debug!(scale = fit.scale, gamma = fit.gamma, "brightness corrected");
    }
}

/// The seed the refiner starts from: the caller's when it is usable, the
/// identity otherwise. A network's direct matrix only has to be finite; every
/// other seed has to be [sane](is_sane_seed).
fn prepare_seed(
    seed: Option<Matrix3<f32>>,
    refiner: Refiner,
    width: usize,
    height: usize,
) -> Option<Matrix3<f32>> {
    seed.filter(|m| match refiner {
        Refiner::Direct => m.iter().all(|v| v.is_finite()),
        _ => is_sane_seed(m, width, height),
    })
}

/// The matrix a failed refinement continues from. A six-parameter model keeps
/// the seed's aspect and shear; the others are reduced to a similarity.
fn seed_fallback(motion: Motion, seed: &Matrix3<f32>, width: usize, height: usize) -> Matrix3<f32> {
    if motion == Motion::Affine {
        *seed
    } else {
        reduce_to_similarity(seed, width, height)
    }
}

/// `matrix`, or the identity when it fits worse than doing nothing.
fn gated(
    reference: &PlanarImage<f32>,
    frame: &PlanarImage<f32>,
    matrix: &Matrix3<f32>,
) -> Matrix3<f32> {
    let kept = apply_identity_gate(reference, frame, matrix);
    if kept != *matrix {
        tracing::warn!("the refined alignment fits worse than none — using the identity");
    }
    kept
}

/// The cumulative matrix: the step itself, or `step · C_{k-1}` in chain mode.
fn cumulative_of(previous: Option<Matrix3<f32>>, step: &Matrix3<f32>) -> Matrix3<f32> {
    previous.map_or(*step, |previous| compose_chain(&previous, step))
}

/// `frame` untouched, with the identity and full coverage (mode `none`).
fn untouched(frame: PlanarImage<f32>) -> AlignedFrame {
    AlignedFrame {
        matrix_applied: Matrix3::identity(),
        coverage: vec![true; frame.width * frame.height],
        frame,
        fallback: None,
    }
}

/// `frame` untouched, with the identity and full coverage, marked as a fallback.
fn unaligned(frame: PlanarImage<f32>) -> AlignedFrame {
    AlignedFrame {
        fallback: Some(Fallback::Unaligned),
        ..untouched(frame)
    }
}

/// A [`Core`] for a frame that was left as it is.
fn left_alone(aligned: AlignedFrame) -> Core {
    Core {
        aligned,
        cumulative: Matrix3::identity(),
        step: Matrix3::identity(),
    }
}

/// Finish a warped frame: chain coverage, brightness, the [`AlignedFrame`].
fn warped_core(
    ctx: &Ctx<'_>,
    mut warped: PlanarImage<f32>,
    mut coverage: Vec<bool>,
    apply: Matrix3<f32>,
    matrices: (Matrix3<f32>, Matrix3<f32>),
    fallback: Option<Fallback>,
) -> Core {
    let (cumulative, step) = matrices;
    if ctx.previous.is_some() {
        coverage = chain_coverage(&cumulative, warped.width, warped.height);
    }
    apply_brightness(&mut warped, ctx.brightness, &coverage);
    Core {
        aligned: AlignedFrame {
            matrix_applied: apply,
            frame: warped,
            coverage,
            fallback,
        },
        cumulative,
        step,
    }
}

/// The refinement of `frame` against `reference` for `refiner`, on the CPU
/// (or with the simplex's objective on `compute`'s device).
fn refine_sync(
    reference: &PlanarImage<f32>,
    pyramid: Option<&LumaPyramid>,
    frame: &PlanarImage<f32>,
    settings: &AlignmentSettings,
    refiner: Refiner,
    seed: &Matrix3<f32>,
    compute: &mut Compute,
) -> Step {
    let Refiner::Motion { motion, bounded } = refiner else {
        return Step {
            matrix: *seed,
            fallback: None,
        };
    };
    let (width, height) = (frame.width, frame.height);
    let Some(pyramid) = pyramid else {
        tracing::warn!("no luma pyramid for the reference — using the seed matrix");
        return Step {
            matrix: seed_fallback(motion, seed, width, height),
            fallback: Some(Fallback::Seed),
        };
    };
    let cfg = MotionRefine {
        optimizer: optimizer_of(settings.optimizer),
        bounded,
        device: compute.alignment_device(),
        ..MotionRefine::new(motion)
    };
    let result = refine_motion(reference, pyramid, frame, seed, &cfg);
    // A device that failed during the refinement was replaced by the CPU for the
    // rest of it; the backend is given up for the rest of the run.
    compute.check_alignment_device();

    let (matrix, fallback) = match result {
        Ok(fit) => (fit.matrix, None),
        Err(error) => {
            tracing::warn!(error = ?error, "alignment refinement failed — using the seed matrix");
            (
                seed_fallback(motion, seed, width, height),
                Some(Fallback::Seed),
            )
        }
    };
    Step {
        matrix: gated(reference, frame, &matrix),
        fallback,
    }
}

fn align_core(
    ctx: &Ctx<'_>,
    frame: PlanarImage<f32>,
    settings: &AlignmentSettings,
    dispatch: Dispatch,
    seed: Option<Matrix3<f32>>,
    compute: &mut Compute,
) -> Core {
    if dispatch.refiner == Refiner::Skip {
        return left_alone(untouched(frame));
    }
    let (width, height) = (frame.width, frame.height);

    // A hint that is not finite and sane would send the optimiser chasing a wild
    // transform; the identity start is the robust default.
    let seed = prepare_seed(seed, dispatch.refiner, width, height);
    let missing_direct = dispatch.refiner == Refiner::Direct && seed.is_none();
    let seed = seed.unwrap_or_else(Matrix3::identity);

    // A pyramid supplied by the caller is used as it is; otherwise one is built
    // for this call, and only for the refiners that read it.
    let own_pyramid;
    let pyramid = match (ctx.pyramid, uses_pyramid(dispatch.refiner)) {
        (Some(pyramid), _) => Some(pyramid),
        (None, true) => {
            own_pyramid = LumaPyramid::new(ctx.reference);
            Some(&own_pyramid)
        }
        (None, false) => None,
    };

    let mut step = refine_sync(
        ctx.reference,
        pyramid,
        &frame,
        settings,
        dispatch.refiner,
        &seed,
        compute,
    );
    if missing_direct {
        step.fallback = Some(Fallback::Seed);
    }

    let cumulative = cumulative_of(ctx.previous, &step.matrix);
    let Some(apply) = cumulative.try_inverse() else {
        tracing::warn!("refined matrix not invertible — using the unwarped frame");
        return left_alone(unaligned(frame));
    };
    let clamped = effective_warp(dispatch, settings) == Warp::EdgeClamped;
    match compute.warp(&frame, &apply, clamped) {
        Ok((warped, coverage)) => warped_core(
            ctx,
            warped,
            coverage,
            apply,
            (cumulative, step.matrix),
            step.fallback,
        ),
        Err(error) => {
            tracing::warn!(error = ?error, "warp failed — using the unwarped frame");
            left_alone(unaligned(frame))
        }
    }
}

/// `true` if `m` is a plausible alignment seed.
///
/// A plausible seed is all-finite, with a similarity scale within `0.5..=2` and a
/// translation of at most a quarter of the frame (GitHub z-stackr's rule: a
/// larger seed puts the optimum out of reach of the refiners' search window).
/// Garbage feature-match estimates (common on glossy or featureless subjects)
/// are rejected so the optimiser falls back to the identity.
#[must_use]
pub fn is_sane_seed(m: &Matrix3<f32>, width: usize, height: usize) -> bool {
    if !m.iter().all(|v| v.is_finite()) {
        return false;
    }
    let sx = m[(0, 0)].hypot(m[(1, 0)]);
    let sy = m[(0, 1)].hypot(m[(1, 1)]);
    if !(0.5..=2.0).contains(&sx) || !(0.5..=2.0).contains(&sy) {
        return false;
    }
    #[expect(clippy::cast_precision_loss, reason = "image edges are far below 2^24")]
    let (w, h) = (width as f32, height as f32);
    m[(0, 2)].abs() <= 0.25 * w && m[(1, 2)].abs() <= 0.25 * h
}

// ── Seeding ───────────────────────────────────────────────────────────────────

/// Where the starting estimates of a run come from. `Sync` because the
/// look-ahead of [`align_sequence_on`] asks for several frames at once.
pub trait SeedSource: Sync {
    /// The starting estimate (reference → source) for frame `index` (`frame` is
    /// that frame), if there is one. Must depend on nothing but `index` and the
    /// pixels, so the result does not depend on the order or the thread.
    fn seed(&self, index: usize, frame: &PlanarImage<f32>) -> Option<Matrix3<f32>>;
}

/// Seeds computed beforehand, one per frame index (the network's matrices).
#[derive(Debug, Clone, PartialEq)]
pub struct PrecomputedSeeds(pub Vec<Option<Matrix3<f32>>>);

impl SeedSource for PrecomputedSeeds {
    fn seed(&self, index: usize, _frame: &PlanarImage<f32>) -> Option<Matrix3<f32>> {
        self.0.get(index).copied().flatten()
    }
}

/// Computes a starting estimate for each frame from AKAZE feature matches
/// against the reference. The reference's features are extracted once.
///
/// The model follows the alignment mode (shift and scale for `translation`,
/// `none` and `neural`, affine otherwise: GitHub z-stackr's table). A seeding
/// failure (too few matches, a degenerate model, an estimate that is not
/// [sane](is_sane_seed)) yields `None` for that frame, never an error: the
/// optimiser then starts from the identity.
///
/// Without the `akaze` feature there is nothing to seed with:
/// [`Self::for_reference`] returns `None`.
pub struct Seeder {
    #[cfg(feature = "akaze")]
    keypoints: Vec<stacker_align::akaze_match::KeyPoint>,
    #[cfg(feature = "akaze")]
    descriptors: Vec<stacker_align::akaze_match::Descriptor>,
    #[cfg(feature = "akaze")]
    model: stacker_align::ransac::AlignmentMode,
}

#[cfg(feature = "akaze")]
impl Seeder {
    /// A seeder for `reference`, or `None` when seeding is switched off or
    /// alignment is skipped altogether.
    #[must_use]
    pub fn for_reference(
        reference: &PlanarImage<f32>,
        settings: &AlignmentSettings,
    ) -> Option<Self> {
        use stacker_align::{akaze_match::extract_ref_features, ransac::AlignmentMode};

        if !settings.akaze_seeding || settings.mode == AlignmentModeSetting::None {
            return None;
        }
        let (keypoints, descriptors) = extract_ref_features(reference);
        tracing::debug!(ref_kps = keypoints.len(), "reference keypoints extracted");
        let model = match settings.mode {
            AlignmentModeSetting::Translation | AlignmentModeSetting::Neural => {
                AlignmentMode::TranslationAndScale
            }
            _ => AlignmentMode::Affine,
        };
        Some(Self {
            keypoints,
            descriptors,
            model,
        })
    }

    /// The starting estimate for `frame` (reference → source), if the matches
    /// support one and it is [sane](is_sane_seed).
    #[must_use]
    pub fn seed(&self, frame: &PlanarImage<f32>) -> Option<Matrix3<f32>> {
        use stacker_align::{akaze_match::KeypointMatcher, ransac::AlignmentEstimator};

        let matched = match KeypointMatcher::match_target(&self.keypoints, &self.descriptors, frame)
        {
            Ok(matched) => matched,
            Err(error) => {
                tracing::warn!(error = ?error, "AKAZE matching failed — no seed");
                return None;
            }
        };
        tracing::debug!(
            target_kps = matched.kps1.len(),
            matched = matched.matches.len(),
            "AKAZE matching done"
        );
        let matrix = match AlignmentEstimator::compute_matrix(
            &matched.matches,
            &matched.kps0,
            &matched.kps1,
            self.model,
        ) {
            Ok(matrix) => matrix,
            Err(error) => {
                tracing::warn!(error = ?error, "RANSAC failed — seeding from the identity");
                return None;
            }
        };
        if is_sane_seed(&matrix, frame.width, frame.height) {
            Some(matrix)
        } else {
            tracing::warn!("AKAZE estimate outside the search window — seeding from the identity");
            None
        }
    }
}

#[cfg(not(feature = "akaze"))]
impl Seeder {
    /// Always `None`: this build has no feature matching. A request for AKAZE
    /// seeding is reported once and otherwise ignored.
    #[must_use]
    pub fn for_reference(
        _reference: &PlanarImage<f32>,
        settings: &AlignmentSettings,
    ) -> Option<Self> {
        if settings.akaze_seeding && settings.mode != AlignmentModeSetting::None {
            tracing::warn!(
                "akaze_seeding is set but this build has no AKAZE support; \
                 aligning from the identity"
            );
        }
        None
    }

    /// Never reached (there is no seeder without the `akaze` feature).
    #[must_use]
    pub const fn seed(&self, _frame: &PlanarImage<f32>) -> Option<Matrix3<f32>> {
        None
    }
}

impl SeedSource for Seeder {
    fn seed(&self, _index: usize, frame: &PlanarImage<f32>) -> Option<Matrix3<f32>> {
        Self::seed(self, frame)
    }
}

// ── Neural pre-pass ───────────────────────────────────────────────────────────

/// Why the neural pre-pass produced no matrices.
#[derive(Debug)]
pub enum NeuralError<E> {
    /// This build or this flow cannot run the network (no `nn` feature, wasm32,
    /// or a [`FrameFlow`] that cannot be read twice); the message says which.
    Unsupported(String),
    /// Loading the model or running it failed.
    Model(String),
    /// Reading a frame failed.
    Flow(E),
    /// The [`Cancel`] asked to stop.
    Cancelled,
}

/// Run the alignment network over the stack.
///
/// Every frame is read once through
/// `reload` (`reload(index)` for `1..frame_count`; `None` means "cannot be read
/// again"), reduced to a thumbnail of at most 512 px on the long side
/// (`stacker_nn::bridge::FrameThumbs`) and dropped, so only the thumbnails are
/// resident. The matrices come back as full-resolution pixel matrices mapping
/// reference coordinates to frame coordinates (index 0 is `None`).
///
/// # Errors
/// [`NeuralError::Unsupported`] without the `nn` feature or when `reload` gives
/// `None`, [`NeuralError::Model`] when the model cannot be loaded or run,
/// [`NeuralError::Flow`] when a frame cannot be read, [`NeuralError::Cancelled`].
pub fn neural_seeds<E, L>(
    model: &AlignModel,
    reference: &PlanarImage<f32>,
    frame_count: usize,
    mut reload: L,
    cancel: &dyn Cancel,
) -> Result<PrecomputedSeeds, NeuralError<E>>
where
    L: FnMut(usize) -> Option<Result<PlanarImage<f32>, E>>,
{
    #[cfg(all(feature = "nn", not(target_arch = "wasm32")))]
    {
        use stacker_nn::{LoadedAlignModel, bridge::FrameThumbs};

        let to_model_error = |why: &dyn fmt::Display| NeuralError::Model(why.to_string());
        let loaded = LoadedAlignModel::load(model.entry(), model.device())
            .map_err(|error| to_model_error(&error))?;
        let mut thumbs = FrameThumbs::new(reference.width, reference.height)
            .map_err(|error| to_model_error(&error))?;
        thumbs
            .push(reference)
            .map_err(|error| to_model_error(&error))?;
        for index in 1..frame_count {
            if cancel.is_cancelled() {
                return Err(NeuralError::Cancelled);
            }
            let frame = match reload(index) {
                Some(Ok(frame)) => frame,
                Some(Err(error)) => return Err(NeuralError::Flow(error)),
                None => {
                    return Err(NeuralError::Unsupported(
                        "the frame source cannot be read a second time".to_owned(),
                    ));
                }
            };
            thumbs
                .push(&frame)
                .map_err(|error| to_model_error(&error))?;
        }
        let matrices = loaded
            .align_thumbs(&thumbs)
            .map_err(|error| to_model_error(&error))?;
        if matrices.len() != frame_count {
            return Err(NeuralError::Model(format!(
                "the network returned {} matrices for {frame_count} frames",
                matrices.len()
            )));
        }
        Ok(PrecomputedSeeds(
            matrices
                .into_iter()
                .enumerate()
                .map(|(index, m)| (index > 0).then(|| Matrix3::from_fn(|r, c| m[r][c])))
                .collect(),
        ))
    }
    #[cfg(not(all(feature = "nn", not(target_arch = "wasm32"))))]
    {
        let _ = (model, reference, frame_count, &mut reload, cancel);
        Err(NeuralError::Unsupported(
            "this build has no neural alignment (feature `nn`, native only)".to_owned(),
        ))
    }
}

// ── The synchronous driver ────────────────────────────────────────────────────

/// Where the frames of a sequence come from and go to. The CLI streams them
/// from disk into its tile store; the GUI moves them in and out of an in-memory
/// list.
pub trait FrameFlow<E> {
    /// The frame `index` (1..count; frame 0 is the reference and is not
    /// requested). Called once per frame, in increasing order.
    ///
    /// # Errors
    /// Whatever the source fails with (decoding, a size mismatch).
    fn load(&mut self, index: usize) -> Result<PlanarImage<f32>, E>;

    /// Receive the aligned frame `index` together with the intersection of the
    /// coverage of frames `1..=index`.
    ///
    /// # Errors
    /// Whatever the sink fails with (writing a tile).
    fn deliver(&mut self, index: usize, aligned: AlignedFrame, common: &[bool]) -> Result<(), E>;

    /// The frame `index` again, **without consuming it**, for the neural
    /// pre-pass that must see the whole stack before the first frame is aligned.
    /// `None` (the default) means the flow cannot: a disk flow reads the file
    /// again, an in-memory flow clones the frame it still holds. Not called
    /// unless `alignment_mode = "neural"`.
    ///
    /// # Errors
    /// As [`Self::load`].
    fn reload(&mut self, _index: usize) -> Option<Result<PlanarImage<f32>, E>> {
        None
    }
}

/// Why [`align_sequence`] stopped early.
#[derive(Debug)]
pub enum SequenceError<E> {
    /// The [`Cancel`] asked to stop; the delivered frames are partly aligned.
    Cancelled,
    /// The [`FrameFlow`] failed.
    Flow(E),
}

/// What [`align_sequence`] learned about the stack.
#[derive(Debug)]
pub struct SequenceOutcome {
    /// The intersection of every frame's coverage, and the worst frame.
    pub coverage: CommonCoverage,
    /// The frames that fell back, in order.
    pub fallbacks: Vec<(usize, Fallback)>,
}

/// The most frames [`align_sequence_on`] loads and seeds at once, whatever the
/// size of the thread pool: an upper bound on the memory the look-ahead may take.
pub const MAX_LOOKAHEAD: usize = 8;

/// How many frames the look-ahead holds at once: the rayon pool size, at most
/// [`MAX_LOOKAHEAD`]. One on wasm32.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn lookahead_width() -> usize {
    rayon::current_num_threads().clamp(1, MAX_LOOKAHEAD)
}

/// How many frames the look-ahead holds at once (one on wasm32).
#[cfg(target_arch = "wasm32")]
#[must_use]
pub const fn lookahead_width() -> usize {
    1
}

/// Bytes the look-ahead holds beyond the serial driver's one frame in flight.
///
/// For a `width × height` stack that is `lookahead_width() − 1` frames waiting in the
/// window, plus the frame the second thread of [`align_sequence_overlapped`] is
/// decoding (`overlapped`). A frame is three `f32` planes (12 B/px). It applies
/// only with a [`SeedSource`] (AKAZE or network seeds); without one the window is
/// one frame wide. The seeding scratch of each concurrent worker (the AKAZE
/// pyramids) comes on top and is not included.
#[must_use]
pub fn lookahead_extra_bytes(width: usize, height: usize, overlapped: bool) -> usize {
    let frames = lookahead_width() - 1 + usize::from(overlapped);
    frames * 12 * width * height
}

/// Align frames `1..frame_count` onto `reference`, one at a time.
///
/// For every frame: poll `cancel`, `flow.load`, seed (with the `akaze` feature
/// and `akaze_seeding`, or from the network in neural mode),
/// [`align_frame`]'s steps, fold the coverage, `flow.deliver`, report
/// [`Stage::Align`] progress. The frames are aligned in order and the result does
/// not depend on the thread count. Without a seed source only one frame is in
/// flight; with one, up to [`lookahead_width`] frames are loaded and seeded in
/// parallel at a time ([`lookahead_extra_bytes`]).
///
/// With `alignment_mode = none` the frames are still loaded and delivered
/// (untouched), so the caller's bookkeeping is the same in every mode.
///
/// # Errors
/// [`SequenceError::Cancelled`] when `cancel` fires between frames,
/// [`SequenceError::Flow`] when `flow` fails.
pub fn align_sequence<E, F: FrameFlow<E>>(
    reference: &PlanarImage<f32>,
    frame_count: usize,
    settings: &AlignmentSettings,
    cancel: &dyn Cancel,
    progress: &dyn Progress,
    flow: &mut F,
) -> Result<SequenceOutcome, SequenceError<E>> {
    align_sequence_on(
        reference,
        frame_count,
        settings,
        cancel,
        progress,
        flow,
        &mut Compute::cpu(),
    )
}

/// [`align_sequence`] on `compute` (see [`align_frame_with_compute`]).
///
/// # Errors
/// As [`align_sequence`].
pub fn align_sequence_on<E, F: FrameFlow<E>>(
    reference: &PlanarImage<f32>,
    frame_count: usize,
    settings: &AlignmentSettings,
    cancel: &dyn Cancel,
    progress: &dyn Progress,
    flow: &mut F,
    compute: &mut Compute,
) -> Result<SequenceOutcome, SequenceError<E>> {
    if frame_count < 2 {
        return Ok(empty_outcome(reference));
    }
    let prepared = prepare::<E, F>(reference, frame_count, settings, flow, cancel)?;
    run_sequence(
        reference,
        frame_count,
        &prepared.settings,
        prepared.seeds(),
        cancel,
        progress,
        &mut Inline(flow),
        compute,
    )
}

/// [`align_sequence_on`] with the seeds supplied by the caller.
///
/// They are not built from the settings; the look-ahead and everything else are the same.
/// `settings.mode` is used as it is (a `neural` mode with `seeds` is the
/// network's hybrid or direct form; there is no model run here).
///
/// # Errors
/// As [`align_sequence`].
#[expect(
    clippy::too_many_arguments,
    reason = "align_sequence_on's seven inputs and the seed source"
)]
pub fn align_sequence_seeded<E, F: FrameFlow<E>>(
    reference: &PlanarImage<f32>,
    frame_count: usize,
    settings: &AlignmentSettings,
    seeds: Option<&dyn SeedSource>,
    cancel: &dyn Cancel,
    progress: &dyn Progress,
    flow: &mut F,
    compute: &mut Compute,
) -> Result<SequenceOutcome, SequenceError<E>> {
    run_sequence(
        reference,
        frame_count,
        settings,
        seeds,
        cancel,
        progress,
        &mut Inline(flow),
        compute,
    )
}

/// [`align_sequence_on`] with the next frame decoded on a second thread.
///
/// The decode runs while the current frame is aligned. Same results, bit for bit; `flow` is shared
/// between the two threads behind a lock (`load` on the second, `deliver` on
/// this one), so both calls must be safe to make from different threads.
/// Exactly one more frame is resident than in [`align_sequence_on`]
/// ([`lookahead_extra_bytes`]). Native only.
///
/// # Errors
/// As [`align_sequence`].
#[cfg(not(target_arch = "wasm32"))]
pub fn align_sequence_overlapped<E: Send, F: FrameFlow<E> + Send>(
    reference: &PlanarImage<f32>,
    frame_count: usize,
    settings: &AlignmentSettings,
    cancel: &dyn Cancel,
    progress: &dyn Progress,
    flow: &mut F,
    compute: &mut Compute,
) -> Result<SequenceOutcome, SequenceError<E>> {
    if frame_count < 2 {
        return Ok(empty_outcome(reference));
    }
    let prepared = prepare::<E, F>(reference, frame_count, settings, flow, cancel)?;
    let shared = Mutex::new(flow);
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::sync_channel::<Result<PlanarImage<f32>, E>>(0);
        let (loader, stopper) = (&shared, &stop);
        scope.spawn(move || {
            for index in 1..frame_count {
                if stopper.load(Ordering::Relaxed) {
                    break;
                }
                let read = loader
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .load(index);
                let failed = read.is_err();
                if sender.send(read).is_err() || failed {
                    break;
                }
            }
        });
        let mut pipe = Overlapped {
            receiver,
            flow: &shared,
        };
        let result = run_sequence(
            reference,
            frame_count,
            &prepared.settings,
            prepared.seeds(),
            cancel,
            progress,
            &mut pipe,
            compute,
        );
        stop.store(true, Ordering::Relaxed);
        // Closing the channel ends a loader that is blocked handing a frame over.
        drop(pipe);
        result
    })
}

fn empty_outcome(reference: &PlanarImage<f32>) -> SequenceOutcome {
    SequenceOutcome {
        coverage: CommonCoverage::new(reference.width, reference.height),
        fallbacks: Vec::new(),
    }
}

/// The settings a run really uses and its seed source.
struct Prepared {
    settings: AlignmentSettings,
    akaze: Option<Seeder>,
    neural: Option<PrecomputedSeeds>,
}

impl Prepared {
    fn seeds(&self) -> Option<&dyn SeedSource> {
        match (&self.neural, &self.akaze) {
            (Some(neural), _) => Some(neural as &dyn SeedSource),
            (None, Some(akaze)) => Some(akaze as &dyn SeedSource),
            (None, None) => None,
        }
    }
}

/// Build the seed source: the network's matrices in neural mode (a mode that
/// cannot run falls back to `registration` with a warning), the AKAZE seeder
/// otherwise.
fn prepare<E, F: FrameFlow<E>>(
    reference: &PlanarImage<f32>,
    frame_count: usize,
    settings: &AlignmentSettings,
    flow: &mut F,
    cancel: &dyn Cancel,
) -> Result<Prepared, SequenceError<E>> {
    let mut effective = settings.clone();
    let mut neural = None;
    if settings.mode == AlignmentModeSetting::Neural {
        let outcome = settings.align_model.as_ref().map(|model| {
            neural_seeds(
                model,
                reference,
                frame_count,
                |index| flow.reload(index),
                cancel,
            )
        });
        match outcome {
            Some(Ok(seeds)) => neural = Some(seeds),
            Some(Err(NeuralError::Cancelled)) => return Err(SequenceError::Cancelled),
            Some(Err(NeuralError::Flow(error))) => return Err(SequenceError::Flow(error)),
            Some(Err(NeuralError::Unsupported(why) | NeuralError::Model(why))) => {
                tracing::warn!(
                    "neural alignment unavailable ({why}); aligning with the registration mode"
                );
                effective.mode = AlignmentModeSetting::Registration;
            }
            None => {
                tracing::warn!(
                    "neural alignment needs a model and none was chosen; \
                     aligning with the registration mode"
                );
                effective.mode = AlignmentModeSetting::Registration;
            }
        }
    }
    let akaze = if neural.is_none() {
        Seeder::for_reference(reference, &effective)
    } else {
        None
    };
    Ok(Prepared {
        settings: effective,
        akaze,
        neural,
    })
}

/// The two ends of the driver's loop: frames in (in increasing order) and
/// aligned frames out.
trait Frames<E> {
    /// The next frame, `None` when the supply ended without one.
    fn next(&mut self, index: usize) -> Option<Result<PlanarImage<f32>, E>>;

    fn deliver(&mut self, index: usize, aligned: AlignedFrame, common: &[bool]) -> Result<(), E>;
}

/// The caller's [`FrameFlow`], used on this thread.
struct Inline<'a, F>(&'a mut F);

impl<E, F: FrameFlow<E>> Frames<E> for Inline<'_, F> {
    fn next(&mut self, index: usize) -> Option<Result<PlanarImage<f32>, E>> {
        Some(self.0.load(index))
    }

    fn deliver(&mut self, index: usize, aligned: AlignedFrame, common: &[bool]) -> Result<(), E> {
        self.0.deliver(index, aligned, common)
    }
}

/// Frames decoded on another thread and handed over one at a time.
#[cfg(not(target_arch = "wasm32"))]
struct Overlapped<'a, 'f, E, F> {
    receiver: mpsc::Receiver<Result<PlanarImage<f32>, E>>,
    flow: &'a Mutex<&'f mut F>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<E, F: FrameFlow<E>> Frames<E> for Overlapped<'_, '_, E, F> {
    fn next(&mut self, _index: usize) -> Option<Result<PlanarImage<f32>, E>> {
        self.receiver.recv().ok()
    }

    fn deliver(&mut self, index: usize, aligned: AlignedFrame, common: &[bool]) -> Result<(), E> {
        self.flow
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .deliver(index, aligned, common)
    }
}

/// A loaded frame waiting for its turn, with its seed.
struct Slot<E> {
    frame: Result<PlanarImage<f32>, E>,
    seed: Option<Matrix3<f32>>,
}

/// The state of a chain: the raw previous frame and what is known about it.
struct ChainState {
    /// The raw previous frame; `None` while the previous frame is the reference.
    raw: Option<PlanarImage<f32>>,
    /// `raw`'s luma pyramid.
    pyramid: Option<LumaPyramid>,
    /// `C_{k-1}`.
    cumulative: Matrix3<f32>,
    /// The previous step, the warm start of the next one.
    last_step: Matrix3<f32>,
}

impl ChainState {
    fn new() -> Self {
        Self {
            raw: None,
            pyramid: None,
            cumulative: Matrix3::identity(),
            last_step: Matrix3::identity(),
        }
    }

    fn context<'a>(
        &'a self,
        reference: &'a PlanarImage<f32>,
        reference_pyramid: Option<&'a LumaPyramid>,
        brightness: Option<&'a BrightnessTarget>,
    ) -> Ctx<'a> {
        Ctx {
            reference: self.raw.as_ref().unwrap_or(reference),
            pyramid: if self.raw.is_some() {
                self.pyramid.as_ref()
            } else {
                reference_pyramid
            },
            brightness,
            previous: Some(self.cumulative),
        }
    }

    /// The warm start of the next step: the feature estimate (frame 0 → frame
    /// `k`) carried to the previous frame, `S_k · C_{k-1}⁻¹`, or else the
    /// previous step.
    fn step_seed(&self, estimate: Option<Matrix3<f32>>) -> Matrix3<f32> {
        if let Some(estimate) = estimate
            && let Some(inverse) = self.cumulative.try_inverse()
        {
            return estimate * inverse;
        }
        self.last_step
    }

    /// Make the frame just aligned (`raw`, if it is kept) the next reference,
    /// unless it could not be warped.
    fn advance(&mut self, raw: Option<PlanarImage<f32>>, core: &Core) {
        if core.aligned.fallback == Some(Fallback::Unaligned) {
            return;
        }
        self.cumulative = core.cumulative;
        self.last_step = core.step;
        self.pyramid = raw.as_ref().map(LumaPyramid::new);
        self.raw = raw;
    }
}

/// `true` when the chain applies to `dispatch`: asked for, and the frames are
/// refined at all (not skipped, not a network's direct matrices).
fn chain_enabled(settings: &AlignmentSettings, dispatch: Dispatch) -> bool {
    if !settings.chain_alignment {
        return false;
    }
    if matches!(dispatch.refiner, Refiner::Motion { .. }) {
        return true;
    }
    tracing::warn!("chain_alignment has no effect without a refinement; aligning to the reference");
    false
}

/// Load `width` frames starting at `*next` (stopping after the first that fails)
/// and seed them in parallel.
fn refill<E, P: Frames<E>>(
    pipe: &mut P,
    seeds: Option<&dyn SeedSource>,
    next: &mut usize,
    frame_count: usize,
    width: usize,
    window: &mut VecDeque<Slot<E>>,
) {
    let first = *next;
    let mut loaded: Vec<Result<PlanarImage<f32>, E>> = Vec::new();
    while loaded.len() < width && *next < frame_count {
        let Some(result) = pipe.next(*next) else {
            break;
        };
        let failed = result.is_err();
        loaded.push(result);
        *next += 1;
        if failed {
            break;
        }
    }
    let found: Vec<Option<Matrix3<f32>>> = seeds.map_or_else(
        || vec![None; loaded.len()],
        |source| {
            let frames: Vec<Option<&PlanarImage<f32>>> =
                loaded.iter().map(|result| result.as_ref().ok()).collect();
            frames
                .par_iter()
                .enumerate()
                .map(|(offset, frame)| {
                    let frame: Option<&PlanarImage<f32>> = *frame;
                    frame.and_then(|frame| source.seed(first + offset, frame))
                })
                .collect()
        },
    );
    window.extend(
        loaded
            .into_iter()
            .zip(found)
            .map(|(frame, seed)| Slot { frame, seed }),
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "the stack, the settings, the seeds, the two hooks, the frame supply and the device"
)]
fn run_sequence<E, P: Frames<E>>(
    reference: &PlanarImage<f32>,
    frame_count: usize,
    settings: &AlignmentSettings,
    seeds: Option<&dyn SeedSource>,
    cancel: &dyn Cancel,
    progress: &dyn Progress,
    pipe: &mut P,
    compute: &mut Compute,
) -> Result<SequenceOutcome, SequenceError<E>> {
    let mut coverage = CommonCoverage::new(reference.width, reference.height);
    let mut fallbacks = Vec::new();
    if frame_count < 2 {
        return Ok(SequenceOutcome {
            coverage,
            fallbacks,
        });
    }
    let dispatch = dispatch_for(settings);
    // The reference does not change, so its luma pyramid and exposure are built
    // once for the whole run (PERF-1) and only when the mode needs them.
    let ref_pyramid = uses_pyramid(dispatch.refiner).then(|| LumaPyramid::new(reference));
    let brightness = brightness_target(reference, settings, dispatch);
    let mut chain = chain_enabled(settings, dispatch).then(ChainState::new);
    let width = if seeds.is_some() && dispatch.refiner != Refiner::Skip {
        lookahead_width()
    } else {
        1
    };
    let last = frame_count - 1;
    let mut window: VecDeque<Slot<E>> = VecDeque::new();
    let mut next_load = 1;

    for index in 1..frame_count {
        if cancel.is_cancelled() {
            return Err(SequenceError::Cancelled);
        }
        if window.is_empty() {
            refill(pipe, seeds, &mut next_load, frame_count, width, &mut window);
        }
        let Some(slot) = window.pop_front() else {
            break;
        };
        let frame = slot.frame.map_err(SequenceError::Flow)?;

        let aligned = if let Some(chain) = chain.as_mut() {
            // The raw frame is the next reference; the last frame has no next.
            let keep = (index < last).then(|| frame.clone());
            let step_seed = Some(chain.step_seed(slot.seed));
            let ctx = chain.context(reference, ref_pyramid.as_ref(), brightness.as_ref());
            let core = align_core(&ctx, frame, settings, dispatch, step_seed, compute);
            chain.advance(keep, &core);
            core.aligned
        } else {
            let ctx = Ctx {
                reference,
                pyramid: ref_pyramid.as_ref(),
                brightness: brightness.as_ref(),
                previous: None,
            };
            align_core(&ctx, frame, settings, dispatch, slot.seed, compute).aligned
        };

        if let Some(kind) = aligned.fallback {
            fallbacks.push((index, kind));
        }
        coverage.fold(index, &aligned.coverage);
        tracing::debug!(
            frame = index,
            scale_x = aligned.matrix_applied[(0, 0)],
            scale_y = aligned.matrix_applied[(1, 1)],
            tx = aligned.matrix_applied[(0, 2)],
            ty = aligned.matrix_applied[(1, 2)],
            "frame aligned"
        );
        pipe.deliver(index, aligned, coverage.mask())
            .map_err(SequenceError::Flow)?;
        progress.stage(
            Stage::Align,
            fraction_of(index, last),
            &format!("Aligning… frame {index}/{last}"),
        );
    }
    Ok(SequenceOutcome {
        coverage,
        fallbacks,
    })
}

// ── The asynchronous driver (the browser) ─────────────────────────────────────
//
// A browser GPU answers asynchronously, so the refiners run as resumable
// searches (`stacker_align::refine::search`) and the warp is awaited. The
// functions below mirror `align_core` / `run_sequence` step for step (seed,
// refiner choice from `dispatch_for`, the identity gate, fallbacks, the warp with
// its edge-clamping choice, brightness, the coverage mask, `CommonCoverage`, the
// chain), so that a backend that answers like the CPU gives the sync results
// bit for bit. They take no AKAZE seed and always run the Nelder-Mead simplex
// (the only optimiser the search machine has; `lucaskanade` means `auto`
// there, and both mean the simplex): the web build has no feature matching or
// network, and a request for either is noted and ignored.

/// The device side of the asynchronous alignment: the objective
/// ([`AsyncLevelDevice`]) and the final warp.
pub trait AsyncAlignBackend: AsyncLevelDevice {
    /// Warp `frame` by `apply` (what [`Compute::warp`] takes as its matrix:
    /// `dst(p) = frame(apply⁻¹ · p)`), with the bicubic sampler and a black
    /// border, or with the edge-clamped spline (`edge_clamped`); return the
    /// warped frame and its coverage mask. `None` when the device failed.
    fn warp(
        &mut self,
        frame: &PlanarImage<f32>,
        apply: &Matrix3<f32>,
        edge_clamped: bool,
    ) -> impl Future<Output = Option<(PlanarImage<f32>, Vec<bool>)>>;
}

/// Why [`align_sequence_async`] stopped early.
#[derive(Debug)]
pub enum AsyncSequenceError<E> {
    /// The [`Cancel`] asked to stop; the delivered frames are partly aligned.
    Cancelled,
    /// The [`FrameFlow`] failed.
    Flow(E),
    /// The device failed and the CPU was not allowed to take over; the message
    /// says so.
    Device(PipelineError),
}

/// The error for a device that failed while the CPU was not allowed to take over.
fn gpu_failed(what: &str) -> PipelineError {
    PipelineError::Core(StackerError::AlignmentFailed(format!(
        "Alignment runs on the GPU only here, and the GPU failed: {what}."
    )))
}

impl From<DeviceRefused> for PipelineError {
    fn from(refused: DeviceRefused) -> Self {
        gpu_failed(&refused.what)
    }
}

/// `true` when every sample of every plane is finite.
fn all_finite(image: &PlanarImage<f32>) -> bool {
    [&image.luma, &image.chroma_a, &image.chroma_b]
        .into_iter()
        .all(|plane| plane.iter().all(|v| v.is_finite()))
}

/// [`align_frame_with_compute`] on an asynchronous backend.
///
/// With `cpu_allowed` a declined level, a failed evaluation or a failed warp is
/// done on the CPU (as the synchronous path does behind a GPU); without it any of
/// them is an error. `ref_pyramid` is built here when it is `None` and the
/// refiner needs one. The optimiser is always the simplex.
///
/// # Errors
/// [`PipelineError`] when the device failed and `cpu_allowed` is `false`.
pub async fn align_frame_async<B: AsyncAlignBackend>(
    reference: &PlanarImage<f32>,
    ref_pyramid: Option<&LumaPyramid>,
    frame: PlanarImage<f32>,
    settings: &AlignmentSettings,
    seed: Option<Matrix3<f32>>,
    backend: &mut B,
    cpu_allowed: bool,
) -> Result<AlignedFrame, PipelineError> {
    let dispatch = dispatch_for(settings);
    let brightness = brightness_target(reference, settings, dispatch);
    let ctx = Ctx {
        reference,
        pyramid: ref_pyramid,
        brightness: brightness.as_ref(),
        previous: None,
    };
    align_core_async(&ctx, frame, settings, dispatch, seed, backend, cpu_allowed)
        .await
        .map(|core| core.aligned)
}

async fn align_core_async<B: AsyncAlignBackend>(
    ctx: &Ctx<'_>,
    frame: PlanarImage<f32>,
    settings: &AlignmentSettings,
    dispatch: Dispatch,
    seed: Option<Matrix3<f32>>,
    backend: &mut B,
    cpu_allowed: bool,
) -> Result<Core, PipelineError> {
    if dispatch.refiner == Refiner::Skip {
        return Ok(left_alone(untouched(frame)));
    }
    let (width, height) = (frame.width, frame.height);

    let seed = prepare_seed(seed, dispatch.refiner, width, height);
    let missing_direct = dispatch.refiner == Refiner::Direct && seed.is_none();
    let seed = seed.unwrap_or_else(Matrix3::identity);

    let own_pyramid;
    let pyramid = match (ctx.pyramid, uses_pyramid(dispatch.refiner)) {
        (Some(pyramid), _) => Some(pyramid),
        (None, true) => {
            own_pyramid = LumaPyramid::new(ctx.reference);
            Some(&own_pyramid)
        }
        (None, false) => None,
    };

    let mut fallback = missing_direct.then_some(Fallback::Seed);
    let step = match (dispatch.refiner, pyramid) {
        (Refiner::Motion { motion, bounded }, Some(pyramid)) => {
            let cfg = MotionRefine {
                bounded,
                ..MotionRefine::new(motion)
            };
            let search = AlignSearch::for_motion(ctx.reference, pyramid, &frame, &seed, &cfg);
            let refined = drive(search, backend, cpu_allowed).await?;
            let matrix = match refined {
                Ok(matrix) => matrix,
                Err(error) => {
                    tracing::warn!(
                        error = ?error,
                        "alignment refinement failed — using the seed matrix"
                    );
                    fallback = Some(Fallback::Seed);
                    seed_fallback(motion, &seed, width, height)
                }
            };
            gated(ctx.reference, &frame, &matrix)
        }
        _ => seed,
    };

    let cumulative = cumulative_of(ctx.previous, &step);
    let Some(apply_matrix) = cumulative.try_inverse() else {
        tracing::warn!("refined matrix not invertible — using the unwarped frame");
        return Ok(left_alone(unaligned(frame)));
    };
    let edge_clamped = effective_warp(dispatch, settings) == Warp::EdgeClamped;

    // `Compute::warp` asks the device only for a finite, invertible matrix.
    let device_can_warp =
        apply_matrix.iter().all(|v| v.is_finite()) && apply_matrix.try_inverse().is_some();
    if device_can_warp {
        let done = backend
            .warp(&frame, &apply_matrix, edge_clamped)
            .await
            .filter(|(image, _)| all_finite(image));
        if let Some((warped, coverage)) = done {
            return Ok(warped_core(
                ctx,
                warped,
                coverage,
                apply_matrix,
                (cumulative, step),
                fallback,
            ));
        }
        if !cpu_allowed {
            return Err(gpu_failed("the warp did not complete"));
        }
    }
    let warped = if edge_clamped {
        warp_image_clamped(&frame, &apply_matrix)
    } else {
        warp_image(&frame, &apply_matrix)
    };
    match warped {
        Ok(warped) => Ok(warped_core(
            ctx,
            warped,
            coverage_mask(&apply_matrix, width, height),
            apply_matrix,
            (cumulative, step),
            fallback,
        )),
        Err(error) => {
            tracing::warn!(error = ?error, "warp failed — using the unwarped frame");
            Ok(left_alone(unaligned(frame)))
        }
    }
}

/// [`align_sequence_on`] on an asynchronous backend.
///
/// The same loop (cancel poll, load, align, fold the coverage, deliver,
/// progress), with the refiners as resumable searches and the warp awaited.
/// AKAZE seeding and the neural mode are not available here (a neural request
/// aligns with `registration`); the frames are aligned from the identity. There is
/// no look-ahead: the browser is single-threaded.
///
/// # Errors
/// [`AsyncSequenceError::Cancelled`] when `cancel` fires between frames,
/// [`AsyncSequenceError::Flow`] when `flow` fails,
/// [`AsyncSequenceError::Device`] when the device failed and `cpu_allowed` is
/// `false`.
#[expect(
    clippy::future_not_send,
    reason = "the backend and the callbacks need not be Send: the web worker is single-threaded"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the synchronous driver's eight inputs: the stack, the hooks, the backend and the CPU rule"
)]
pub async fn align_sequence_async<E, F: FrameFlow<E>, B: AsyncAlignBackend>(
    reference: &PlanarImage<f32>,
    frame_count: usize,
    settings: &AlignmentSettings,
    cancel: &dyn Cancel,
    progress: &dyn Progress,
    flow: &mut F,
    backend: &mut B,
    cpu_allowed: bool,
) -> Result<SequenceOutcome, AsyncSequenceError<E>> {
    let mut coverage = CommonCoverage::new(reference.width, reference.height);
    let mut fallbacks = Vec::new();
    if frame_count < 2 {
        return Ok(SequenceOutcome {
            coverage,
            fallbacks,
        });
    }
    if settings.akaze_seeding && settings.mode != AlignmentModeSetting::None {
        tracing::warn!(
            "akaze_seeding is set but the asynchronous aligner has no feature matching; \
             aligning from the identity"
        );
    }
    let mut effective = settings.clone();
    if settings.mode == AlignmentModeSetting::Neural {
        tracing::warn!(
            "neural alignment is not available in the asynchronous aligner; \
             aligning with the registration mode"
        );
        effective.mode = AlignmentModeSetting::Registration;
    }
    let settings = &effective;
    let dispatch = dispatch_for(settings);
    let ref_pyramid = uses_pyramid(dispatch.refiner).then(|| LumaPyramid::new(reference));
    let brightness = brightness_target(reference, settings, dispatch);
    let mut chain = chain_enabled(settings, dispatch).then(ChainState::new);
    let last = frame_count - 1;

    for index in 1..frame_count {
        if cancel.is_cancelled() {
            return Err(AsyncSequenceError::Cancelled);
        }
        let frame = flow.load(index).map_err(AsyncSequenceError::Flow)?;
        let aligned = if let Some(chain) = chain.as_mut() {
            let keep = (index < last).then(|| frame.clone());
            let step_seed = Some(chain.step_seed(None));
            let ctx = chain.context(reference, ref_pyramid.as_ref(), brightness.as_ref());
            let core = align_core_async(
                &ctx,
                frame,
                settings,
                dispatch,
                step_seed,
                backend,
                cpu_allowed,
            )
            .await
            .map_err(AsyncSequenceError::Device)?;
            chain.advance(keep, &core);
            core.aligned
        } else {
            let ctx = Ctx {
                reference,
                pyramid: ref_pyramid.as_ref(),
                brightness: brightness.as_ref(),
                previous: None,
            };
            align_core_async(&ctx, frame, settings, dispatch, None, backend, cpu_allowed)
                .await
                .map_err(AsyncSequenceError::Device)?
                .aligned
        };
        if let Some(kind) = aligned.fallback {
            fallbacks.push((index, kind));
        }
        coverage.fold(index, &aligned.coverage);
        tracing::debug!(
            frame = index,
            scale_x = aligned.matrix_applied[(0, 0)],
            scale_y = aligned.matrix_applied[(1, 1)],
            tx = aligned.matrix_applied[(0, 2)],
            ty = aligned.matrix_applied[(1, 2)],
            "frame aligned"
        );
        flow.deliver(index, aligned, coverage.mask())
            .map_err(AsyncSequenceError::Flow)?;
        progress.stage(
            Stage::Align,
            fraction_of(index, last),
            &format!("Aligning… frame {index}/{last}"),
        );
    }
    Ok(SequenceOutcome {
        coverage,
        fallbacks,
    })
}
