//! Motion models and optimiser selection.
//!
//! One entry point, [`refine_motion`], that the pipeline maps its alignment
//! settings onto: *which motion* ([`Motion`]), *which optimiser*
//! ([`Optimizer`]), whether the simplex is logit-bounded, the iteration cap and
//! an optional GPU objective device.
//!
//! ## Motion models
//!
//! | [`Motion`]     | degrees of freedom                                   |
//! |----------------|------------------------------------------------------|
//! | `Shift`        | shift X, shift Y                                     |
//! | `Translation`  | shift X, shift Y, uniform scale (focus breathing)    |
//! | `Similarity`   | shift, uniform scale, rotation (4 DOF)               |
//! | `Affine`       | similarity + aspect + shear (6 DOF)                  |
//!
//! `Similarity` is what the pipeline's `registration` mode runs (the coarse to
//! fine pyramid search that was called `affine` before the GitHub meanings were
//! adopted); `Affine` is the new full six-parameter solve.  The windows of
//! Lucas-Kanade and of the bounded simplex depend on `bounded`: with bounded
//! search on they are the crate's narrow ones (shift and scale ±5 %, rotation
//! ±5°), with it off GitHub's wide ones (shift and scale ±20 %, rotation ±20°,
//! [`BoundedRefineOptions::wide`]); aspect and shear are ±10 % either way.
//! The narrow windows on the default path capped every frame's scale at 5 %
//! from the seed, so a series with stronger focus breathing came out with
//! echoed edges.
//!
//! ## Optimisers
//!
//! - [`Optimizer::NelderMead`]: the simplex.  Unbounded (`bounded = false`,
//!   [`refine_alignment_with_pyramid`]) or logit-bounded (`bounded = true`,
//!   [`refine_alignment_bounded_with_pyramid`]); `Affine` is always bounded, an
//!   unbounded six-parameter simplex wanders into degenerate matrices.  The only
//!   optimiser that uses `device` and the only one the ask/tell machine
//!   ([`super::search::AlignSearch`]) runs.
//! - [`Optimizer::LucasKanade`]: [`refine_alignment_lk_with_pyramid`], CPU only.
//!   A failure is returned as the error; there is no simplex fallback.
//! - [`Optimizer::Auto`]: Lucas-Kanade first; if it errors, or its final
//!   objective is not finite or is worse than its starting one, the simplex runs
//!   from the same seed.  This is GitHub's rule.
//!
//! With a GPU `device` the simplex objective runs there (when it accepts the
//! level) while Lucas-Kanade still runs on the CPU; so `Auto` and `LucasKanade`
//! use the CPU for their LK pass even with a device configured, and the
//! simplex fallback and `NelderMead` use the device.  In the browser (ask/tell)
//! every optimiser is the simplex: build the search with
//! [`AlignSearch::for_motion`](super::search::AlignSearch::for_motion).

use nalgebra::Matrix3;
use stacker_core::{error::StackerError, image::PlanarImage};

use super::{
    BoundedRefineOptions, LumaPyramid, NM_MAX_ITER, RefineOptions,
    lk::refine_alignment_lk_with_pyramid, refine_alignment_bounded_with_pyramid,
    refine_alignment_with_pyramid,
};
use crate::device::DeviceHandle;

/// The motion model a refinement may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Motion {
    /// Shift only (X and Y); scale and rotation stay at the seed's.  For tests
    /// and seeding; no pipeline mode uses it.
    Shift,
    /// Shift and a uniform scale about the image centre (focus breathing);
    /// rotation stays at the seed's.
    Translation,
    /// Shift, uniform scale and rotation: four degrees of freedom.
    Similarity,
    /// Similarity plus aspect ratio and shear: six degrees of freedom.
    Affine,
}

impl Motion {
    /// Whether the uniform scale is searched.
    #[must_use]
    pub const fn allows_scale(self) -> bool {
        !matches!(self, Self::Shift)
    }

    /// Whether the rotation is searched.
    #[must_use]
    pub const fn allows_rotation(self) -> bool {
        matches!(self, Self::Similarity | Self::Affine)
    }

    /// Whether the aspect ratio and the shear are searched (the six-parameter
    /// model).
    #[must_use]
    pub const fn allows_aspect_shear(self) -> bool {
        matches!(self, Self::Affine)
    }

    /// The bounded options of this motion: the DOF switches set, the crate's
    /// default windows, no device.
    #[must_use]
    pub fn bounded_options(self) -> BoundedRefineOptions {
        BoundedRefineOptions {
            allow_scale: self.allows_scale(),
            allow_rotation: self.allows_rotation(),
            allow_aspect: self.allows_aspect_shear(),
            allow_shear: self.allows_aspect_shear(),
            ..BoundedRefineOptions::default()
        }
    }

    /// The options of the unbounded simplex ([`refine_alignment_with_pyramid`]),
    /// or `None` for `Affine`, which has no unbounded form.
    #[must_use]
    pub fn unbounded_options(self) -> Option<RefineOptions> {
        (!self.allows_aspect_shear()).then(|| RefineOptions {
            allow_scale: self.allows_scale(),
            allow_rotate: self.allows_rotation(),
            ..RefineOptions::default()
        })
    }
}

/// Which optimiser minimises the objective.  See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Optimizer {
    /// Lucas-Kanade, with the simplex as the fallback (GitHub's rule).
    #[default]
    Auto,
    /// Lucas-Kanade only (CPU).
    LucasKanade,
    /// The Nelder-Mead simplex only.
    NelderMead,
}

/// What a [`refine_motion`] call is asked to do.
#[derive(Debug, Clone)]
pub struct MotionRefine {
    /// The motion model.
    pub motion: Motion,
    /// The optimiser.
    pub optimizer: Optimizer,
    /// Bounded search: the simplex searches in the logit-bounded space
    /// ([`refine_alignment_bounded_with_pyramid`]) instead of the unbounded one,
    /// and Lucas-Kanade and the bounded simplex use the narrow windows. Off,
    /// they use the wide windows ([`BoundedRefineOptions::wide`]); `Affine`'s
    /// simplex is bounded either way.
    pub bounded: bool,
    /// Iteration cap of every level of the simplex's schedule
    /// ([`level_plan`](super::level_plan)); the full-resolution polish keeps its
    /// default budget ([`DEFAULT_POLISH_ITERATIONS`](super::DEFAULT_POLISH_ITERATIONS)).
    pub max_iterations: usize,
    /// GPU objective for the simplex; Lucas-Kanade ignores it.
    pub device: Option<DeviceHandle>,
}

impl MotionRefine {
    /// The defaults for `motion`: automatic optimiser, unbounded simplex, the
    /// crate's iteration cap, no device.
    #[must_use]
    pub const fn new(motion: Motion) -> Self {
        Self {
            motion,
            optimizer: Optimizer::Auto,
            bounded: false,
            max_iterations: NM_MAX_ITER,
            device: None,
        }
    }

    /// `true` when the simplex has to be the logit-bounded one: `bounded`, or a
    /// motion without an unbounded form.
    #[must_use]
    pub const fn simplex_is_bounded(&self) -> bool {
        self.bounded || self.motion.allows_aspect_shear()
    }

    /// The [`BoundedRefineOptions`] of the bounded simplex and of
    /// Lucas-Kanade: the motion's degrees of freedom, the narrow windows with
    /// bounded search on and the wide ones without.
    #[must_use]
    pub fn bounded_options(&self) -> BoundedRefineOptions {
        let windows = if self.bounded {
            BoundedRefineOptions::default()
        } else {
            BoundedRefineOptions::wide()
        };
        BoundedRefineOptions {
            max_iterations: self.max_iterations,
            device: self.device.clone(),
            max_shift_x: windows.max_shift_x,
            max_shift_y: windows.max_shift_y,
            max_scale: windows.max_scale,
            max_rotation_deg: windows.max_rotation_deg,
            ..self.motion.bounded_options()
        }
    }

    /// The [`RefineOptions`] of the unbounded simplex, `None` when the simplex is
    /// bounded.
    #[must_use]
    pub fn unbounded_options(&self) -> Option<RefineOptions> {
        if self.simplex_is_bounded() {
            return None;
        }
        self.motion.unbounded_options().map(|o| RefineOptions {
            max_iterations: self.max_iterations,
            device: self.device.clone(),
            ..o
        })
    }
}

impl Default for MotionRefine {
    fn default() -> Self {
        Self::new(Motion::Similarity)
    }
}

/// Which optimiser produced a [`Refined`] matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Lucas-Kanade.
    LucasKanade,
    /// The Nelder-Mead simplex.
    NelderMead,
}

/// The outcome of [`refine_motion`].
#[derive(Debug, Clone, Copy)]
pub struct Refined {
    /// The refined matrix (reference coordinates to source coordinates).
    pub matrix: Matrix3<f32>,
    /// The optimiser that produced it.
    pub engine: Engine,
    /// `true` when `Optimizer::Auto` ran Lucas-Kanade, rejected its result (an
    /// error, or no improvement) and let the simplex produce the matrix.
    pub fell_back: bool,
}

/// Run the simplex of `cfg` from `initial`.
fn simplex(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    cfg: &MotionRefine,
) -> Result<Matrix3<f32>, StackerError> {
    cfg.unbounded_options().map_or_else(
        || {
            refine_alignment_bounded_with_pyramid(
                reference,
                ref_pyramid,
                source,
                initial,
                &cfg.bounded_options(),
            )
        },
        |opts| refine_alignment_with_pyramid(reference, ref_pyramid, source, initial, &opts),
    )
}

/// Refine `initial` (reference to source) for the motion and optimiser of `cfg`.
///
/// `ref_pyramid` must have been built from `reference` (once per run).  With
/// [`Optimizer::Auto`] a Lucas-Kanade result that errors, is not finite or is no
/// better than its start is replaced by the simplex from the same seed
/// ([`Refined::fell_back`]).
///
/// # Errors
/// [`StackerError::MathError`] for a non-finite `initial`,
/// [`StackerError::AlignmentFailed`] for mismatched sizes, a pyramid of another
/// image or a failed optimisation (for `Auto` only when the simplex fails too).
pub fn refine_motion(
    reference: &PlanarImage<f32>,
    ref_pyramid: &LumaPyramid,
    source: &PlanarImage<f32>,
    initial: &Matrix3<f32>,
    cfg: &MotionRefine,
) -> Result<Refined, StackerError> {
    let nelder_mead = |fell_back: bool| -> Result<Refined, StackerError> {
        simplex(reference, ref_pyramid, source, initial, cfg).map(|matrix| Refined {
            matrix,
            engine: Engine::NelderMead,
            fell_back,
        })
    };
    let lucas_kanade = || {
        refine_alignment_lk_with_pyramid(
            reference,
            ref_pyramid,
            source,
            initial,
            &cfg.bounded_options(),
        )
    };
    match cfg.optimizer {
        Optimizer::NelderMead => nelder_mead(false),
        Optimizer::LucasKanade => lucas_kanade().map(|r| Refined {
            matrix: r.matrix,
            engine: Engine::LucasKanade,
            fell_back: false,
        }),
        Optimizer::Auto => match lucas_kanade() {
            Ok(r) if r.final_rms.is_finite() && r.final_rms <= r.starting_rms => Ok(Refined {
                matrix: r.matrix,
                engine: Engine::LucasKanade,
                fell_back: false,
            }),
            // An error, or a result that did not improve on its own start.
            Ok(_) | Err(_) => nelder_mead(true),
        },
    }
}
