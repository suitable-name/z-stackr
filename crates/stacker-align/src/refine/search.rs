//! Resumable ("ask/tell") forms of the refiners.
//!
//! The synchronous refiners ([`refine_alignment_with_pyramid`](super::refine_alignment_with_pyramid),
//! [`refine_alignment_bounded_with_pyramid`](super::refine_alignment_bounded_with_pyramid),
//! [`refine_alignment_registration`](super::refine_alignment_registration))
//! call the objective as a closure and so cannot wait for it. A browser GPU
//! answers asynchronously, so [`AlignSearch`] runs the *same* search as an
//! explicit state machine: it asks for one thing at a time ([`Ask`]), the caller
//! answers ([`AlignSearch::level_ready`], [`AlignSearch::tell`]), and the search
//! carries on from where it stopped.
//!
//! # Identity with the synchronous refiners
//!
//! The optimiser loops are the same loops, cut at every objective evaluation:
//! the Nelder-Mead simplex (`NelderMead`) and the golden-section search
//! (`Golden`) perform exactly the arithmetic of `nelder_mead_generic` and
//! `golden_section_min`, in the same order; the level schedule (the very same
//! [`level_plan`](super::level_plan), polish included), the polish's simplex
//! steps and bracket, the ten restarts of the coarsest level (and their
//! jitter, drawn from the same fixed-seed generator in the same order), the
//! objective's penalty and coverage rules (`prepare` / `finish` split
//! `LevelObjective::eval` at the device call) and the final checks are copied
//! over. Given the same
//! answers for the same questions the result is therefore bit for bit the
//! synchronous one; the tests in `tests/ask_tell.rs` pin that, with the CPU
//! objective and with a device that declines levels and fails evaluations.
//!
//! The synchronous functions are untouched; this module only adds a second
//! driver for the same code.

use std::cell::RefCell;

use nalgebra::Matrix3;
use stacker_core::{error::StackerError, image::PlanarImage};

use super::{
    AffineParams, BOUNDED_STEP, BoundedRefineOptions, Flavor, GOLDEN_BRACKET, GOLDEN_TOL,
    JitterRng, LevelObjective, LevelStep, LevelView, LumaPyramid, NM_STEP_ROTATE, NM_STEP_SCALE,
    NM_STEP_TX, NM_STEP_TY, NM_TOL, PENALTY, RESTARTS, RefineOptions, RegistrationParams, Schedule,
    SearchBounds, StepKind, Warper, check_pyramid, matrix_to_params, motion::MotionRefine,
    params_to_matrix, polish_steps_4d, rms_of, seed_params,
};
use crate::device::{Moments, ObjectiveSampler};

/// What a finished search returns: the refined matrix, or the error the
/// synchronous refiner would have returned.
pub type SearchResult = Result<Matrix3<f32>, StackerError>;

/// What an [`AlignSearch`] needs next.
#[derive(Debug)]
pub enum Ask {
    /// Prepare pyramid level `level` (0 = coarsest) on the device: upload the
    /// planes [`AlignSearch::level_planes`] gives and answer with
    /// [`AlignSearch::level_ready`] (`false` = the device declines the level, so
    /// the search evaluates it on the CPU).
    Level {
        /// Index into the pyramid levels.
        level: usize,
    },
    /// Evaluate the candidate whose inverse matrix (in the level's pixel
    /// coordinates, exactly what `LevelDevice::moments` receives) is `inverse`.
    /// Answer with [`AlignSearch::tell`]: `Some(moments)`, or `None` to have this
    /// one evaluated on the CPU by the search itself.
    Evaluate {
        /// Output pixel `p` samples the reference at `inverse * p`.
        inverse: Matrix3<f32>,
    },
    /// Finished: what the synchronous refiner returns.
    Done(SearchResult),
}

/// The planes of one pyramid level, to upload for [`Ask::Level`].
#[derive(Debug, Clone, Copy)]
pub struct LevelPlanes<'a> {
    /// Reference luma, `width * height` samples.
    pub reference: &'a [f32],
    /// Source luma, `width * height` samples.
    pub source: &'a [f32],
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// The sampler the objective warps the reference with.
    pub sampler: ObjectiveSampler,
}

// ── the minimisers as state machines ─────────────────────────────────────────

/// Reflection, expansion, contraction and shrink coefficients of
/// `nelder_mead_generic`.
const ALPHA: f64 = 1.0;
const GAMMA: f64 = 2.0;
const RHO: f64 = 0.5;
const SIGMA: f64 = 0.5;

/// One step of a minimiser: the next point whose value it needs, or its result.
enum MStep {
    Eval(Vec<f64>),
    Done(Vec<f64>),
}

#[derive(Clone, Copy)]
enum NmState {
    /// Waiting for the value of `simplex[i]`.
    Init(usize),
    Reflect,
    Expand,
    Contract,
    /// Waiting for the value of the shrunk `simplex[i]`.
    Shrink(usize),
}

/// `nelder_mead_generic` as a state machine: every objective call is a return
/// to the caller, everything else is the same code in the same order.
struct NelderMead {
    n: usize,
    max_iter: usize,
    tol: f64,
    simplex: Vec<Vec<f64>>,
    fvals: Vec<f64>,
    /// Completed iterations (the `for _ in 0..max_iter` counter).
    iter: usize,
    state: NmState,
    centroid: Vec<f64>,
    xr: Vec<f64>,
    fr: f64,
    /// The expansion or contraction point under evaluation.
    xt: Vec<f64>,
    outside: bool,
}

impl NelderMead {
    fn new(x0: &[f64], step: &[f64], max_iter: usize, tol: f64) -> Self {
        let n = x0.len();
        debug_assert_eq!(step.len(), n, "one initial step per dimension");
        let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        simplex.push(x0.to_vec());
        for (i, &s) in step.iter().enumerate() {
            let mut v = x0.to_vec();
            v[i] += s;
            simplex.push(v);
        }
        Self {
            n,
            max_iter,
            tol,
            simplex,
            fvals: Vec::with_capacity(n + 1),
            iter: 0,
            state: NmState::Init(0),
            centroid: Vec::new(),
            xr: Vec::new(),
            fr: 0.0,
            xt: Vec::new(),
            outside: false,
        }
    }

    /// `None` starts the search; `Some(value)` answers the last `Eval`.
    fn next(&mut self, value: Option<f64>) -> MStep {
        let Some(v) = value else {
            return MStep::Eval(self.simplex[0].clone());
        };
        let n = self.n;
        match self.state {
            NmState::Init(i) => {
                self.fvals.push(v);
                if i < n {
                    self.state = NmState::Init(i + 1);
                    return MStep::Eval(self.simplex[i + 1].clone());
                }
                self.begin_iteration()
            }
            NmState::Reflect => {
                self.fr = v;
                if v < self.fvals[0] {
                    self.xt = (0..n)
                        .map(|j| self.centroid[j] + GAMMA * (self.xr[j] - self.centroid[j]))
                        .collect();
                    self.state = NmState::Expand;
                    MStep::Eval(self.xt.clone())
                } else if v < self.fvals[n - 1] {
                    self.simplex[n] = std::mem::take(&mut self.xr);
                    self.fvals[n] = v;
                    self.next_iteration()
                } else {
                    // Outside contraction when the reflected point beats the
                    // worst vertex, inside contraction otherwise.
                    self.outside = v < self.fvals[n];
                    self.xt = if self.outside {
                        (0..n)
                            .map(|j| self.centroid[j] + RHO * (self.xr[j] - self.centroid[j]))
                            .collect()
                    } else {
                        (0..n)
                            .map(|j| {
                                self.centroid[j] - RHO * (self.centroid[j] - self.simplex[n][j])
                            })
                            .collect()
                    };
                    self.state = NmState::Contract;
                    MStep::Eval(self.xt.clone())
                }
            }
            NmState::Expand => {
                if v < self.fr {
                    self.simplex[n] = std::mem::take(&mut self.xt);
                    self.fvals[n] = v;
                } else {
                    self.simplex[n] = std::mem::take(&mut self.xr);
                    self.fvals[n] = self.fr;
                }
                self.next_iteration()
            }
            NmState::Contract => {
                let accepted = if self.outside {
                    v <= self.fr
                } else {
                    v < self.fvals[n]
                };
                if accepted {
                    self.simplex[n] = std::mem::take(&mut self.xt);
                    self.fvals[n] = v;
                    self.next_iteration()
                } else {
                    self.shrink_vertex(1);
                    self.state = NmState::Shrink(1);
                    MStep::Eval(self.simplex[1].clone())
                }
            }
            NmState::Shrink(i) => {
                self.fvals[i] = v;
                if i < n {
                    self.shrink_vertex(i + 1);
                    self.state = NmState::Shrink(i + 1);
                    return MStep::Eval(self.simplex[i + 1].clone());
                }
                self.next_iteration()
            }
        }
    }

    /// Move vertex `i` towards the best vertex (`simplex[0]`, which a shrink
    /// never changes).
    fn shrink_vertex(&mut self, i: usize) {
        let best = self.simplex[0].clone();
        for j in 0..self.n {
            self.simplex[i][j] = best[j] + SIGMA * (self.simplex[i][j] - best[j]);
        }
    }

    fn next_iteration(&mut self) -> MStep {
        self.iter += 1;
        self.begin_iteration()
    }

    /// The top of the loop body: order the vertices, test convergence, reflect.
    fn begin_iteration(&mut self) -> MStep {
        let n = self.n;
        if self.iter >= self.max_iter {
            return self.finish();
        }
        let mut order: Vec<usize> = (0..=n).collect();
        let fvals = &self.fvals;
        order.sort_unstable_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.simplex = order.iter().map(|&i| self.simplex[i].clone()).collect();
        self.fvals = order.iter().map(|&i| self.fvals[i]).collect();

        let f_spread = self.fvals[n] - self.fvals[0];
        if f_spread < self.tol && f_spread >= 0.0 {
            return self.finish();
        }

        let mut centroid = vec![0.0_f64; n];
        for v in &self.simplex[..n] {
            for (cj, vj) in centroid.iter_mut().zip(v.iter()) {
                *cj += vj;
            }
        }
        for cj in &mut centroid {
            *cj /= n as f64;
        }
        self.xr = (0..n)
            .map(|j| centroid[j] + ALPHA * (centroid[j] - self.simplex[n][j]))
            .collect();
        self.centroid = centroid;
        self.state = NmState::Reflect;
        MStep::Eval(self.xr.clone())
    }

    /// After the loop: the best vertex.
    fn finish(&self) -> MStep {
        let n = self.n;
        let mut order: Vec<usize> = (0..=n).collect();
        let fvals = &self.fvals;
        order.sort_unstable_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        MStep::Done(self.simplex[order[0]].clone())
    }
}

#[derive(Clone, Copy)]
enum GsState {
    /// Waiting for `f(x1)` of the initial probes.
    First,
    /// Waiting for `f(x2)` of the initial probes.
    Second,
    /// Waiting for the new `x1` (the interval lost its right end).
    NewLeft,
    /// Waiting for the new `x2` (the interval lost its left end).
    NewRight,
}

/// `golden_section_min` as a state machine.
struct Golden {
    a: f64,
    b: f64,
    x1: f64,
    x2: f64,
    f1: f64,
    f2: f64,
    iter: usize,
    state: GsState,
}

impl Golden {
    const R: f64 = 0.618_033_988_749_895;
    const C: f64 = 1.0 - Self::R;

    fn new(a: f64, b: f64) -> Self {
        Self {
            a,
            b,
            x1: a + Self::C * (b - a),
            x2: a + Self::R * (b - a),
            f1: 0.0,
            f2: 0.0,
            iter: 0,
            state: GsState::First,
        }
    }

    fn next(&mut self, value: Option<f64>) -> MStep {
        let Some(v) = value else {
            return MStep::Eval(vec![self.x1]);
        };
        match self.state {
            GsState::First => {
                self.f1 = v;
                self.state = GsState::Second;
                MStep::Eval(vec![self.x2])
            }
            GsState::Second | GsState::NewRight => {
                self.f2 = v;
                self.advance()
            }
            GsState::NewLeft => {
                self.f1 = v;
                self.advance()
            }
        }
    }

    /// The top of the loop body (`for _ in 0..200`).
    fn advance(&mut self) -> MStep {
        if self.iter >= 200 || (self.b - self.a).abs() < GOLDEN_TOL {
            return MStep::Done(vec![if self.f1 < self.f2 { self.x1 } else { self.x2 }]);
        }
        self.iter += 1;
        if self.f1 < self.f2 {
            self.b = self.x2;
            self.x2 = self.x1;
            self.f2 = self.f1;
            self.x1 = self.a + Self::C * (self.b - self.a);
            self.state = GsState::NewLeft;
            MStep::Eval(vec![self.x1])
        } else {
            self.a = self.x1;
            self.x1 = self.x2;
            self.f1 = self.f2;
            self.x2 = self.a + Self::R * (self.b - self.a);
            self.state = GsState::NewRight;
            MStep::Eval(vec![self.x2])
        }
    }
}

enum Minimizer {
    Simplex(NelderMead),
    Golden(Golden),
}

impl Minimizer {
    fn next(&mut self, value: Option<f64>) -> MStep {
        match self {
            Self::Simplex(m) => m.next(value),
            Self::Golden(m) => m.next(value),
        }
    }
}

// ── the objective, split at the device call ──────────────────────────────────

/// `LevelObjective::eval` up to the device call.
enum Prepared {
    /// The objective has its value without any evaluation (penalty, empty level).
    Value(f64),
    /// The warp whose moments the objective needs.
    Inverse(Matrix3<f32>),
}

fn prepare(params_vec: &[f64; 4], width: usize, height: usize) -> Prepared {
    let p = RegistrationParams::from_vec(params_vec);
    if !p.is_finite() {
        return Prepared::Value(PENALTY);
    }
    let Some(fwd) = params_to_matrix(&p, width, height) else {
        return Prepared::Value(PENALTY);
    };
    prepare_forward(&fwd, width, height)
}

/// [`prepare`] for the six-parameter model.
fn prepare_affine(p: &AffineParams, width: usize, height: usize) -> Prepared {
    p.to_matrix(width, height)
        .map_or(Prepared::Value(PENALTY), |fwd| {
            prepare_forward(&fwd, width, height)
        })
}

/// [`prepare`] after the forward matrix is built: the inverse and the empty
/// level rule.
fn prepare_forward(fwd: &Matrix3<f32>, width: usize, height: usize) -> Prepared {
    let Some(inv) = fwd.try_inverse() else {
        return Prepared::Value(PENALTY);
    };
    if !inv.iter().all(|v| v.is_finite()) {
        return Prepared::Value(PENALTY);
    }
    if width * height == 0 {
        return Prepared::Value(0.0);
    }
    Prepared::Inverse(inv)
}

// ── the search ───────────────────────────────────────────────────────────────

/// What the caller has to supply next.
#[derive(Clone, Copy)]
enum Outstanding {
    Idle,
    Level(usize),
    Evaluate(Matrix3<f32>),
    Finished,
}

/// What the search logic reacts to.
#[derive(Clone, Copy)]
enum Input {
    Begin,
    LevelReady,
    Value(f64),
}

/// What the search logic needs from the objective.
enum Need {
    Level(usize),
    /// The objective at these four parameters.
    Params([f64; 4]),
    /// The objective at these six parameters (the bounded searches).
    Affine(AffineParams),
    /// The objective's penalty (non-finite search coordinates), no evaluation.
    Penalty,
    Done(SearchResult),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// A minimiser is running; the next value is one of its points.
    Minimize,
    /// The next value is the objective at the end of a coarsest-level restart.
    Restart,
    Other,
}

enum Kind {
    /// Nothing to do (the search finished at construction).
    Empty,
    /// The unbounded 4-DOF pyramid search.
    Unbounded {
        step: [f64; 4],
        /// The parameters carried from level to level.
        best: [f64; 4],
    },
    /// The bounded logit-space search (bounded and registration).
    Bounded {
        bounds: SearchBounds,
        coords: Vec<f64>,
    },
}

/// The multistart of the coarsest level, as the synchronous refiners run it.
struct Restarts {
    /// The restarts run (the one under way is `attempt`), `0..RESTARTS`.
    attempt: u32,
    /// A multistart is under way.
    active: bool,
    /// The parameters carried into the coarsest level (the seed).
    base: Vec<f64>,
    /// The best objective over the restarts so far.
    best_obj: f64,
    /// The parameters that reached it (the seed until one does).
    best: Vec<f64>,
    /// The minimiser's result awaiting its objective value.
    res: Vec<f64>,
}

impl Restarts {
    const fn idle() -> Self {
        Self {
            attempt: 0,
            active: false,
            base: Vec::new(),
            best_obj: f64::INFINITY,
            best: Vec::new(),
            res: Vec::new(),
        }
    }

    /// A multistart from the carried parameters `base`.
    fn begin(base: Vec<f64>) -> Self {
        Self {
            attempt: 0,
            active: true,
            best: base.clone(),
            base,
            best_obj: f64::INFINITY,
            res: Vec::new(),
        }
    }
}

/// A refinement that can wait for its objective: the resumable form of
/// [`refine_alignment_with_pyramid`](super::refine_alignment_with_pyramid),
/// [`refine_alignment_bounded_with_pyramid`](super::refine_alignment_bounded_with_pyramid)
/// and [`refine_alignment_registration`](super::refine_alignment_registration).
///
/// Drive it with [`Self::ask`]:
///
/// - [`Ask::Level`]: upload the level, then [`Self::level_ready`];
/// - [`Ask::Evaluate`]: evaluate, then [`Self::tell`];
/// - [`Ask::Done`]: the result.
///
/// `device.rs`' `drive` is the loop for an `AsyncLevelDevice`. Without a device
/// ([`Self::cpu_only`]) `ask` runs straight to [`Ask::Done`].
///
/// The `device` field of the options is ignored: the caller *is* the device.
pub struct AlignSearch<'a> {
    reference: &'a PlanarImage<f32>,
    /// The caller's reference pyramid, or `None` when the search owns one
    /// (`ref_owned`) or has none (it finished at construction).
    ref_pyramid: Option<&'a LumaPyramid>,
    /// The reference pyramid of a search that built its own
    /// ([`Self::registration`]).
    ref_owned: Option<LumaPyramid>,
    source: &'a PlanarImage<f32>,
    src_pyramid: Option<LumaPyramid>,
    sampler: ObjectiveSampler,
    kind: Kind,
    /// `"refine_alignment"`, `"refine_alignment_bounded"` or
    /// `"refine_alignment_registration"`, for the error messages.
    who: &'static str,
    /// The registration refiner (its errors do not mention the NaN guard).
    registration: bool,
    /// The level schedule ([`level_plan`](super::level_plan)).
    plan: Vec<LevelStep>,
    /// Expected residual in pixels before the polish (see [`Schedule`]).
    residual_px: f64,
    // Position in the search.
    /// Index of the current step in `plan`.
    step_idx: usize,
    /// Pyramid level of the current step.
    level: usize,
    budget: usize,
    phase: Phase,
    minimizer: Option<Minimizer>,
    restarts: Restarts,
    /// Jitter of the bounded restarts.
    rng: JitterRng,
    // Ask/tell state.
    use_device: bool,
    /// The device took the current level.
    device_level: bool,
    input: Input,
    outstanding: Outstanding,
    result: Option<SearchResult>,
    /// The CPU warp buffer, kept across evaluations of a level.
    scratch: Vec<f32>,
}

/// The checks every refiner makes before it starts, with the same messages.
fn validate(
    who: &str,
    reference: &PlanarImage<f32>,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
) -> Result<(), StackerError> {
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(StackerError::MathError(format!(
            "{who}: initial matrix contains non-finite values"
        )));
    }
    if reference.width != source.width || reference.height != source.height {
        return Err(StackerError::AlignmentFailed(format!(
            "{who}: reference and source dimensions do not match"
        )));
    }
    Ok(())
}

impl<'a> AlignSearch<'a> {
    /// A search that has nothing to run: it ends with `result` at the first `ask`.
    const fn finished(
        reference: &'a PlanarImage<f32>,
        source: &'a PlanarImage<f32>,
        who: &'static str,
        result: SearchResult,
    ) -> Self {
        Self {
            reference,
            ref_pyramid: None,
            ref_owned: None,
            source,
            src_pyramid: None,
            sampler: ObjectiveSampler::Bicubic,
            kind: Kind::Empty,
            who,
            registration: false,
            plan: Vec::new(),
            residual_px: 0.0,
            step_idx: 0,
            level: 0,
            budget: 0,
            phase: Phase::Other,
            minimizer: None,
            restarts: Restarts::idle(),
            rng: JitterRng::new(),
            use_device: true,
            device_level: false,
            input: Input::Begin,
            outstanding: Outstanding::Finished,
            result: Some(result),
            scratch: Vec::new(),
        }
    }

    /// The resumable [`refine_alignment_with_pyramid`](super::refine_alignment_with_pyramid).
    /// `ref_pyramid` must have been built from `reference`; an invalid input
    /// ends the search at once with the error the synchronous function returns.
    #[must_use]
    pub fn unbounded(
        reference: &'a PlanarImage<f32>,
        ref_pyramid: &'a LumaPyramid,
        source: &'a PlanarImage<f32>,
        initial: &Matrix3<f32>,
        opts: &RefineOptions,
    ) -> Self {
        let who = "refine_alignment";
        if let Err(e) = validate(who, reference, source, initial)
            .and_then(|()| check_pyramid(who, reference, ref_pyramid))
        {
            return Self::finished(reference, source, who, Err(e));
        }
        let p0 = matrix_to_params(initial, source.width, source.height);
        let step = [
            NM_STEP_TX,
            NM_STEP_TY,
            if opts.allow_scale { NM_STEP_SCALE } else { 0.0 },
            if opts.allow_rotate {
                NM_STEP_ROTATE
            } else {
                0.0
            },
        ];
        let best = p0.to_vec();
        let schedule = Schedule::new(ref_pyramid, opts.max_iterations, opts.polish_iterations);
        let level = schedule.steps.first().map_or(0, |s| s.level);
        Self {
            reference,
            ref_pyramid: Some(ref_pyramid),
            source,
            src_pyramid: Some(LumaPyramid::new(source)),
            sampler: ObjectiveSampler::Bicubic,
            kind: Kind::Unbounded { step, best },
            who,
            plan: schedule.steps,
            residual_px: schedule.residual_px,
            level,
            ..Self::finished(reference, source, who, Ok(*initial))
        }
        .started()
    }

    /// The resumable
    /// [`refine_alignment_bounded_with_pyramid`](super::refine_alignment_bounded_with_pyramid).
    #[must_use]
    pub fn bounded(
        reference: &'a PlanarImage<f32>,
        ref_pyramid: &'a LumaPyramid,
        source: &'a PlanarImage<f32>,
        initial: &Matrix3<f32>,
        opts: &BoundedRefineOptions,
    ) -> Self {
        Self::bounded_levels(
            Flavor::Bounded,
            reference,
            Some(ref_pyramid),
            source,
            initial,
            opts,
        )
    }

    /// The resumable
    /// [`refine_alignment_registration_with_pyramid`](super::refine_alignment_registration_with_pyramid):
    /// the bounded pyramid search with the edge-clamped spline sampler.
    /// `ref_pyramid` must have been built from `reference`.
    #[must_use]
    pub fn registration_with_pyramid(
        reference: &'a PlanarImage<f32>,
        ref_pyramid: &'a LumaPyramid,
        source: &'a PlanarImage<f32>,
        initial: &Matrix3<f32>,
        opts: &BoundedRefineOptions,
    ) -> Self {
        Self::bounded_levels(
            Flavor::Registration,
            reference,
            Some(ref_pyramid),
            source,
            initial,
            opts,
        )
    }

    /// The resumable
    /// [`refine_alignment_registration`](super::refine_alignment_registration):
    /// [`Self::registration_with_pyramid`] with a reference pyramid the search
    /// builds itself (a caller with a stack should build it once and use the
    /// other form).
    #[must_use]
    pub fn registration(
        reference: &'a PlanarImage<f32>,
        source: &'a PlanarImage<f32>,
        initial: &Matrix3<f32>,
        opts: &BoundedRefineOptions,
    ) -> Self {
        Self::bounded_levels(Flavor::Registration, reference, None, source, initial, opts)
    }

    /// The bounded search of `flavor`. Without a `ref_pyramid` the search builds
    /// the reference pyramid itself (only once the inputs are valid and there is
    /// something to search).
    fn bounded_levels(
        flavor: Flavor,
        reference: &'a PlanarImage<f32>,
        ref_pyramid: Option<&'a LumaPyramid>,
        source: &'a PlanarImage<f32>,
        initial: &Matrix3<f32>,
        opts: &BoundedRefineOptions,
    ) -> Self {
        let who = flavor.who();
        if let Err(e) = validate(who, reference, source, initial).and_then(|()| {
            ref_pyramid.map_or(Ok(()), |pyramid| check_pyramid(who, reference, pyramid))
        }) {
            return Self::finished(reference, source, who, Err(e));
        }
        let p0 = seed_params(initial, source.width, source.height, opts);
        let bounds = SearchBounds::new(&p0, opts);
        if bounds.n() == 0 {
            return Self::finished(reference, source, who, Ok(*initial));
        }
        let coords = bounds.to_search_space(&p0);
        let ref_owned = ref_pyramid.is_none().then(|| LumaPyramid::new(reference));
        let schedule = Schedule::new(
            ref_pyramid
                .or(ref_owned.as_ref())
                .unwrap_or_else(|| unreachable!("a pyramid is borrowed or built above")),
            opts.max_iterations,
            opts.polish_iterations,
        );
        let level = schedule.steps.first().map_or(0, |s| s.level);
        Self {
            reference,
            ref_pyramid,
            ref_owned,
            source,
            src_pyramid: Some(LumaPyramid::new(source)),
            sampler: flavor.sampler(),
            kind: Kind::Bounded { bounds, coords },
            who,
            registration: flavor == Flavor::Registration,
            plan: schedule.steps,
            residual_px: schedule.residual_px,
            level,
            ..Self::finished(reference, source, who, Ok(*initial))
        }
        .started()
    }

    /// The resumable simplex of a [`MotionRefine`]: the unbounded pyramid search
    /// ([`Self::unbounded`]) or the bounded one ([`Self::bounded`]), whichever
    /// [`refine_motion`](super::refine_motion) would run for the same
    /// configuration.
    ///
    /// The ask/tell machine is Nelder-Mead only, so the optimiser of `cfg` is
    /// ignored: on the web `LucasKanade` and `Auto` mean the simplex.  `cfg.device`
    /// is ignored too (the caller of the search is the device).
    #[must_use]
    pub fn for_motion(
        reference: &'a PlanarImage<f32>,
        ref_pyramid: &'a LumaPyramid,
        source: &'a PlanarImage<f32>,
        initial: &Matrix3<f32>,
        cfg: &MotionRefine,
    ) -> Self {
        cfg.unbounded_options().map_or_else(
            || {
                Self::bounded(
                    reference,
                    ref_pyramid,
                    source,
                    initial,
                    &cfg.bounded_options(),
                )
            },
            |opts| Self::unbounded(reference, ref_pyramid, source, initial, &opts),
        )
    }

    /// A search that has been set up and is ready to run (not finished).
    fn started(mut self) -> Self {
        self.outstanding = Outstanding::Idle;
        self.result = None;
        self
    }

    /// Never ask for a device: every level and every evaluation runs on the CPU
    /// inside the search, as in the synchronous refiner without a device, and
    /// [`Self::ask`] goes straight to [`Ask::Done`].
    #[must_use]
    pub const fn cpu_only(mut self) -> Self {
        self.use_device = false;
        self
    }

    /// The reference pyramid, the caller's or the search's own.
    fn ref_pyr(&self) -> Option<&LumaPyramid> {
        self.ref_pyramid.or(self.ref_owned.as_ref())
    }

    /// The two views of pyramid level `idx` (0 = coarsest).
    fn views(&self, idx: usize) -> (LevelView<'_>, LevelView<'_>) {
        match (self.ref_pyr(), &self.src_pyramid) {
            (Some(rp), Some(sp)) => (
                rp.level(idx, &self.reference.luma),
                sp.level(idx, &self.source.luma),
            ),
            _ => (
                LevelView {
                    luma: &self.reference.luma,
                    width: self.reference.width,
                    height: self.reference.height,
                },
                LevelView {
                    luma: &self.source.luma,
                    width: self.source.width,
                    height: self.source.height,
                },
            ),
        }
    }

    /// The planes of level `level`, for an [`Ask::Level`].
    #[must_use]
    pub fn level_planes(&self, level: usize) -> LevelPlanes<'_> {
        let (r, s) = self.views(level);
        LevelPlanes {
            reference: r.luma,
            source: s.luma,
            width: r.width,
            height: r.height,
            sampler: self.sampler,
        }
    }

    /// What the search needs next. Calling it again without answering repeats the
    /// question; after [`Ask::Done`] it reports an error.
    pub fn ask(&mut self) -> Ask {
        loop {
            match self.outstanding {
                Outstanding::Level(level) => return Ask::Level { level },
                Outstanding::Evaluate(inverse) => return Ask::Evaluate { inverse },
                Outstanding::Finished => {
                    return Ask::Done(self.result.take().unwrap_or_else(|| {
                        Err(StackerError::AlignmentFailed(format!(
                            "{}: the search result was already taken",
                            self.who
                        )))
                    }));
                }
                Outstanding::Idle => {}
            }
            match self.resume(self.input) {
                Need::Level(level) => {
                    if self.use_device {
                        self.outstanding = Outstanding::Level(level);
                    } else {
                        self.device_level = false;
                        self.input = Input::LevelReady;
                    }
                }
                Need::Penalty => self.input = Input::Value(PENALTY),
                Need::Params(params) => {
                    let (w, h) = self.level_dims();
                    self.evaluate(&prepare(&params, w, h));
                }
                Need::Affine(params) => {
                    let (w, h) = self.level_dims();
                    self.evaluate(&prepare_affine(&params, w, h));
                }
                Need::Done(result) => {
                    self.result = Some(result);
                    self.outstanding = Outstanding::Finished;
                }
            }
        }
    }

    /// Width and height of the current level.
    fn level_dims(&self) -> (usize, usize) {
        let (r, _) = self.views(self.level);
        (r.width, r.height)
    }

    /// Turn a prepared objective request into the next ask (a device
    /// evaluation) or straight into the objective value (penalty, or no device
    /// on this level).
    fn evaluate(&mut self, prepared: &Prepared) {
        match prepared {
            Prepared::Value(v) => self.input = Input::Value(*v),
            Prepared::Inverse(inverse) => {
                if self.device_level {
                    self.outstanding = Outstanding::Evaluate(*inverse);
                } else {
                    let v = self.finish(inverse, None);
                    self.input = Input::Value(v);
                }
            }
        }
    }

    /// Answer an [`Ask::Level`]: `accepted` is whether the device took the level
    /// (`false` = the search evaluates this level on the CPU).
    ///
    /// # Panics
    /// When the last [`Self::ask`] was not an [`Ask::Level`].
    pub fn level_ready(&mut self, accepted: bool) {
        assert!(
            matches!(self.outstanding, Outstanding::Level(_)),
            "level_ready without an Ask::Level"
        );
        self.device_level = accepted;
        self.input = Input::LevelReady;
        self.outstanding = Outstanding::Idle;
    }

    /// Answer an [`Ask::Evaluate`]: the moments of the warp, or `None` to have the
    /// search evaluate this candidate on the CPU.
    ///
    /// # Panics
    /// When the last [`Self::ask`] was not an [`Ask::Evaluate`].
    pub fn tell(&mut self, moments: Option<Moments>) {
        let Outstanding::Evaluate(inverse) = self.outstanding else {
            panic!("tell without an Ask::Evaluate");
        };
        let v = self.finish(&inverse, moments);
        self.input = Input::Value(v);
        self.outstanding = Outstanding::Idle;
    }

    /// `LevelObjective::eval` after the device call: the RMS (from the moments,
    /// or from the CPU warp) and the coverage rule.
    fn finish(&mut self, inverse: &Matrix3<f32>, moments: Option<Moments>) -> f64 {
        let n = {
            let (r, _) = self.views(self.level);
            r.width * r.height
        };
        let (rms, covered) = moments.map_or_else(|| self.cpu_rms(inverse), rms_of);
        // A transform that samples less than half of the frame is not a candidate.
        if covered * 2 < n {
            return PENALTY;
        }
        rms
    }

    /// Warp and RMS of the current level on the CPU, through the very code the
    /// synchronous objective uses.
    fn cpu_rms(&mut self, inverse: &Matrix3<f32>) -> (f64, usize) {
        let scratch = std::mem::take(&mut self.scratch);
        let (ref_lvl, src_lvl) = self.views(self.level);
        let objective = LevelObjective {
            warper: Warper {
                ref_luma: ref_lvl.luma,
                width: ref_lvl.width,
                height: ref_lvl.height,
                sampler: self.sampler,
            },
            src_luma: src_lvl.luma,
            scratch: RefCell::new(scratch),
            device: None,
        };
        let out = objective.cpu_rms(inverse);
        let scratch = objective.scratch.into_inner();
        self.scratch = scratch;
        out
    }

    // ── the level schedule ───────────────────────────────────────────────────

    fn resume(&mut self, input: Input) -> Need {
        match input {
            Input::Begin => Need::Level(self.level),
            Input::LevelReady => self.start_level(),
            Input::Value(v) => self.on_value(v),
        }
    }

    /// The device has answered for `self.level`: start its minimisation, in the
    /// role the schedule gives the step ([`StepKind`]).
    fn start_level(&mut self) -> Need {
        let Some(planned) = self.plan.get(self.step_idx).copied() else {
            return Need::Done(Err(self.empty_error()));
        };
        self.budget = planned.iters;
        let polish = planned.kind == StepKind::Polish;
        let (full_w, full_h) = (self.source.width, self.source.height);
        let residual = self.residual_px;
        let minimizer = match &self.kind {
            Kind::Unbounded { step, best } => {
                if planned.kind == StepKind::Coarsest {
                    self.restarts = Restarts::begin(best.to_vec());
                    return self.start_restart();
                }
                let steps = if polish {
                    polish_steps_4d(step, residual, full_w, full_h)
                } else {
                    *step
                };
                Minimizer::Simplex(NelderMead::new(best, &steps, self.budget, NM_TOL))
            }
            Kind::Bounded { bounds, coords } => {
                if bounds.n() == 1 {
                    let (lo, hi) = if polish {
                        bounds.polish_bracket(coords, residual, full_w, full_h)
                    } else {
                        (-GOLDEN_BRACKET, GOLDEN_BRACKET)
                    };
                    Minimizer::Golden(Golden::new(lo, hi))
                } else if planned.kind == StepKind::Coarsest {
                    self.restarts = Restarts::begin(coords.clone());
                    return self.start_restart();
                } else {
                    let steps = if polish {
                        bounds.polish_steps(coords, residual, full_w, full_h)
                    } else {
                        vec![BOUNDED_STEP; coords.len()]
                    };
                    Minimizer::Simplex(NelderMead::new(coords, &steps, self.budget, NM_TOL))
                }
            }
            Kind::Empty => return Need::Done(Err(self.empty_error())),
        };
        self.minimizer = Some(minimizer);
        self.pump(None)
    }

    fn empty_error(&self) -> StackerError {
        StackerError::AlignmentFailed(format!("{}: the search has no work", self.who))
    }

    /// Coarsest level: start restart number `attempt` from the incoming
    /// parameters (the first start is the seed; the others are the deterministic
    /// offsets of the unbounded search, or the generator's jitter on every logit
    /// coordinate of the bounded one).
    fn start_restart(&mut self) -> Need {
        let attempt = self.restarts.attempt;
        let mut start = self.restarts.base.clone();
        // The generator is touched exactly where the synchronous `multistart`
        // touches it: once per coordinate, from the second start on.
        if attempt > 0 && matches!(self.kind, Kind::Bounded { .. }) {
            for c in &mut start {
                *c += self.rng.jitter();
            }
        }
        let steps: Vec<f64> = match &self.kind {
            Kind::Unbounded { step, .. } => {
                if attempt > 0 {
                    let factor = f64::from(attempt) * 0.001;
                    start[0] += factor * if attempt.is_multiple_of(2) { 1.0 } else { -1.0 };
                    start[1] += factor * if attempt.is_multiple_of(3) { 1.0 } else { -1.0 };
                }
                step.to_vec()
            }
            Kind::Bounded { .. } => vec![BOUNDED_STEP; start.len()],
            Kind::Empty => return Need::Done(Err(self.empty_error())),
        };
        self.minimizer = Some(Minimizer::Simplex(NelderMead::new(
            &start,
            &steps,
            self.budget,
            NM_TOL,
        )));
        self.pump(None)
    }

    /// Feed a value to the running minimiser (or start it) and turn its answer
    /// into a need.
    fn pump(&mut self, value: Option<f64>) -> Need {
        let Some(minimizer) = self.minimizer.as_mut() else {
            return Need::Done(Err(self.empty_error()));
        };
        match minimizer.next(value) {
            MStep::Eval(point) => {
                self.phase = Phase::Minimize;
                self.point_need(&point)
            }
            MStep::Done(result) => {
                self.minimizer = None;
                self.minimized(&result)
            }
        }
    }

    /// The objective request for a minimiser point.
    fn point_need(&self, point: &[f64]) -> Need {
        match &self.kind {
            Kind::Unbounded { .. } => Need::Params([point[0], point[1], point[2], point[3]]),
            Kind::Bounded { bounds, .. } => {
                // `objective_bounded`: non-finite coordinates cost the penalty.
                if !point.iter().all(|c| c.is_finite()) {
                    return Need::Penalty;
                }
                Need::Affine(bounds.to_param_space(point))
            }
            Kind::Empty => Need::Penalty,
        }
    }

    /// A level's minimiser has finished with `result`.
    fn minimized(&mut self, result: &[f64]) -> Need {
        if self.restarts.active {
            // The restart's own objective value decides whether it wins.
            self.restarts.res = result.to_vec();
            self.phase = Phase::Restart;
            return self.point_need(result);
        }
        if let Err(e) = self.set_carried(result) {
            return Need::Done(Err(e));
        }
        self.finish_level()
    }

    /// Store `params` as the parameters carried to the next level.
    fn set_carried(&mut self, params: &[f64]) -> Result<(), StackerError> {
        match &mut self.kind {
            Kind::Unbounded { best, .. } => {
                *best = [params[0], params[1], params[2], params[3]];
            }
            Kind::Bounded { coords, .. } => *coords = params.to_vec(),
            Kind::Empty => return Err(self.empty_error()),
        }
        Ok(())
    }

    fn on_value(&mut self, v: f64) -> Need {
        if self.phase == Phase::Restart {
            let r = &mut self.restarts;
            if v < r.best_obj {
                r.best_obj = v;
                r.best = std::mem::take(&mut r.res);
            }
            r.attempt += 1;
            if r.attempt < RESTARTS {
                return self.start_restart();
            }
            let best = std::mem::take(&mut r.best);
            self.restarts = Restarts::idle();
            self.phase = Phase::Other;
            if let Err(e) = self.set_carried(&best) {
                return Need::Done(Err(e));
            }
            return self.finish_level();
        }
        self.pump(Some(v))
    }

    fn finish_level(&mut self) -> Need {
        self.step_idx += 1;
        self.phase = Phase::Other;
        if let Some(next) = self.plan.get(self.step_idx) {
            self.level = next.level;
            return Need::Level(self.level);
        }
        Need::Done(self.finalize())
    }

    /// The checks and the matrix after the last level.
    fn finalize(&self) -> SearchResult {
        let (w, h) = (self.source.width, self.source.height);
        let (best_p, nan_note) = match &self.kind {
            Kind::Unbounded { best, .. } => {
                (AffineParams::from(RegistrationParams::from_vec(best)), true)
            }
            Kind::Bounded { bounds, coords } => {
                (bounds.to_param_space(coords), !self.is_registration())
            }
            Kind::Empty => return Err(self.empty_error()),
        };
        if !best_p.is_finite() {
            return Err(StackerError::AlignmentFailed(if nan_note {
                format!(
                    "{}: optimizer produced non-finite parameters (NaN encountered)",
                    self.who
                )
            } else {
                format!("{}: optimizer produced non-finite parameters", self.who)
            }));
        }
        best_p.to_matrix(w, h).ok_or_else(|| {
            StackerError::AlignmentFailed(format!(
                "{}: refined parameters produced a degenerate matrix",
                self.who
            ))
        })
    }

    const fn is_registration(&self) -> bool {
        self.registration
    }
}
