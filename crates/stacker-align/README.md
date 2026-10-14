# z-stackr-align

The alignment engine of **z-stackr** (crate identifier `stacker_align`).

Focus-bracketed sequences change in magnification (focus breathing) and shift sideways from frame to frame, especially when shot hand-held. This crate estimates, for each frame, a transform that registers it to the reference frame, and applies it. The transform is chosen from a **motion model** ([`Motion`]): shift only, shift and uniform scale, the four-parameter **similarity** (shift in x and y, uniform scale, rotation) or the six-parameter **affine** map (adds aspect ratio and shear). No model is a full homography.

## Alignment pipeline

The alignment is intensity-based at its core, with an optional feature-matching seed.

### 1. Optional seed (feature `akaze`, off by default at run time)
With the `akaze` feature compiled in, the consuming app decides per run whether to use the seed. The crate then extracts AKAZE keypoints and binary descriptors with the [`akaze`](https://crates.io/crates/akaze) crate, matches them between the reference and the frame, and fits a model with RANSAC (`ransac::AlignmentEstimator::compute_matrix`: translation-only or a 6-parameter affine model, 500 iterations). The sampling uses a fixed-seed `SplitMix64` generator, so the same matches always give the same matrix. A fit is accepted only when at least `min(clamp(ceil(n / 5), min_sample + 2, 50), n)` of the `n` matches agree with it (the same floor guards the focus-breathing estimate).

Feature detectors fail on glossy or low-texture subjects, so seeding is off by default. Without a seed, the optimiser starts from the identity matrix. A seed that is not plausible is discarded by the caller (`stacker_pipeline::align::is_sane_seed`).

### 2. Subpixel refinement (always runs)
Starting from the seed (or the identity), an optimiser minimises the RMS difference between the source luma plane and the warped reference luma plane, counting only pixels the warp can sample (a transform that samples less than half of the frame is penalised). The search is coarse-to-fine over a luma pyramid (`LumaPyramid`; the coarsest level is 32 to 63 px on its short side). `refine_motion` is the one entry point the pipeline maps its settings onto:

| `Motion` | Degrees of freedom | Pipeline mode |
|---|---|---|
| `Shift` | shift X, shift Y | (tests and seeding) |
| `Translation` | shift X, shift Y, uniform scale | `translation` |
| `Similarity` | shift, uniform scale, rotation | `registration` (the default) |
| `Affine` | similarity + aspect ratio + shear | `affine` |

The `Optimizer` selects how the objective is minimised:

* **`NelderMead`**: the simplex, in the unbounded form (`refine_alignment`) or the logit-bounded form (`refine_alignment_bounded`). `Affine` is always bounded, because an unbounded six-parameter simplex wanders into degenerate matrices. It is the only optimiser the GPU objective and the ask/tell `search::AlignSearch` (the browser build) serve.
* **`LucasKanade`**: Gauss-Newton on the same pyramid and objective (`refine_alignment_lk`), clamped to the same windows. CPU only; a failure is returned as an error.
* **`Auto`** (default): Lucas-Kanade first; if it errors, or its final objective is not finite or is worse than its start, the simplex runs from the same seed (`Refined::fell_back` says so).

#### The simplex schedule
The Nelder-Mead refiners walk the pyramid with one coarse-biased schedule (`level_plan`, shared by the synchronous refiners and `AlignSearch`): 150 iterations on every level up to the large ones, 3 on a level whose short side exceeds 512 px, none on the finest level, restarting the coarsest level from ten deterministic starts (the seed plus nine fixed offsets of `0.001 * j` on tx and ty), then a short full-resolution **polish** (`DEFAULT_POLISH_ITERATIONS`, 40) from a simplex scaled to the expected residual. Each level is capped by `max_iterations`.

#### Direction convention
The refiners return a matrix `M` that maps **reference coordinates to source coordinates**: at the optimum `source(M * p) ~ reference(p)`. `transform::warp_image(src, A)` produces `dst(p) = src(A^-1 * p)`. To bring a frame onto the reference, warp it with the **inverse**, `A = M^-1`; passing `M` itself displaces the frame by twice its offset. The RANSAC seed uses the same direction.

#### Bounded mode (`refine_alignment_bounded`)
The same objective, but each active degree of freedom (rotation, scale, shift x, shift y, and for `Affine` aspect and shear) is mapped into a logit-transformed search space with an explicit interval around the starting estimate (`BoundedRefineOptions`). Because the inverse map is a sigmoid, the parameters cannot leave their intervals, so the simplex cannot drift into a degenerate transform. The default half-widths are 5 percent of the image size for each shift, 5 percent for the scale and 5 degrees for the rotation (10 percent for aspect and shear). `allow_shift_x`, `allow_shift_y`, `allow_scale`, `allow_rotation`, `allow_aspect` and `allow_shear` switch degrees of freedom off (a switched-off one is pinned to the value in the seed); a shift-only mode must pass `allow_scale: false` and `allow_rotation: false`, because the defaults enable them. With a single active degree of freedom a plain golden-section search (a bracketing search without parabolic steps, so not Brent's method) replaces the simplex. The unbounded and the bounded paths share one Nelder-Mead implementation.

#### Registration objective (`refine_alignment_registration`)
The bounded pyramid search with a different sampler: the objective samples the reference with a four-tap, edge-clamped **C0 cubic kernel**. This kernel is not Catmull-Rom: its weights sum to 1 and it interpolates at the knots, but it is only continuous across integer coordinates, so the objective has a kink at integer shifts. The kernel is kept as it is because recorded registration results depend on it. The matching warp is `transform::warp_image_clamped`. It used to be a single-resolution search; it now uses the same level schedule as the other refiners (`refine_alignment_registration_with_pyramid` takes a reference pyramid built once).

#### Determinism
The RANSAC sampling is deterministic, and the RMS objective adds its per-row partial sums serially in row order, so the objective does not depend on the thread count or on rayon's scheduling.

### 3. Identity gate (`apply_identity_gate`)
A bounded search is centred on its seed, so a plausible but wrong seed can still end on a transform that fits *worse* than doing nothing. After the refinement the gate halves both luma planes until the short side is at most 256 px and compares the objective of the refined matrix with that of the identity; if the identity is strictly better, it replaces the matrix (equal scores keep the refined one). `refined_beats_identity` is the predicate. The gate has no GPU form and no state.

### 4. Brightness correction (`brightness`)
Exposure drift between shots shows up as brightness steps in a fused result. `BrightnessTarget::from_reference` records the reference's luma statistics; `correct_brightness` and `correct_brightness_masked` (with the warp's coverage mask) fit `y' = (scale * y)^gamma` on **linear** luma so that the frame's mean and standard deviation match the reference's, with `gamma` limited to `[0.5, 2.0]`, and apply the same per-pixel ratio to the chroma planes. A frame that already agrees, or a fit that does not improve on the identity, leaves the frame bit for bit unchanged.

### 5. Chain alignment (`chain`)
Instead of aligning every frame to frame 0, a chain aligns frame `k` to frame `k - 1`. `compose_chain(&C_prev, &step)` composes the steps (`C_k = R_k * C_(k-1)`, in `f64`), and `chain_coverage(&C_k, w, h)` gives the coverage of a frame in frame-0 coordinates from the cumulative matrix, not the step's, so a border of undefined pixels never enters the common area. The module documentation compares the two ways to run the loop.

## Core features

- **Multi-level pyramids** (`LumaPyramid`): large displacements are solved at coarse resolution before the full-resolution polish. A stack aligns N-1 frames to one reference, so the `*_with_pyramid` forms of every refiner take a reference pyramid built once.
- **Linear-light metrics**: the objective runs on the linear-light luma plane, so it has no sRGB gamma weighting.
- **Warping**: `warp_image` resamples with bicubic (Catmull-Rom) interpolation and a zero border, with a `std::simd` inner loop; `warp_image_clamped` is the edge-clamped C0-cubic variant. The SIMD width is fixed when the crate is compiled from the target features (16 lanes with `avx512f`, 8 with `avx2`, otherwise 4, one SSE2 operation on the default `x86-64` baseline) and does not change the result; fused multiply-adds go through `stacker_core::fma`, which is an instruction on native targets that have `fma` and an unfused multiply and add on wasm32. `-C target-cpu=x86-64-v3` or a newer level selects the wider vectors (see the root README, "CPU features"). `transform::simd_build_report()` says what the build was compiled for.
- **Common-area crop**: `coverage_mask` marks the pixels a warp fills with real data, `intersect_coverage` combines masks, and `largest_true_rectangle` finds the rectangle valid in every frame.
- **GPU objective**: `device::DeviceHandle` is the hook through which the simplex objective can run on a GPU (`stacker-gpu` implements it); the crate itself has no GPU dependency.

## Usage

```toml
[dependencies]
z-stackr-align = { path = "../stacker-align", features = ["akaze"] }
```

Refine one frame and warp it onto the reference:

```rust
use stacker_align::{
    Matrix3, RefineOptions, refine_alignment,
    transform::{coverage_mask, warp_image},
};
use stacker_core::{error::StackerError, image::PlanarImage};

fn align_one(
    reference: &PlanarImage<f32>,
    frame: &PlanarImage<f32>,
) -> Result<(PlanarImage<f32>, Vec<bool>), StackerError> {
    // `m` maps reference coordinates to frame coordinates.
    let m = refine_alignment(
        reference,
        frame,
        &Matrix3::identity(),
        &RefineOptions::default(),
    )?;
    // Bring the frame onto the reference: warp with the inverse.
    let apply = m
        .try_inverse()
        .ok_or_else(|| StackerError::MathError("the refined matrix is not invertible".into()))?;
    let aligned = warp_image(frame, &apply)?;
    let covered = coverage_mask(&apply, frame.width, frame.height);
    Ok((aligned, covered))
}
```

Align a whole stack with a chosen motion model and optimiser, the reference pyramid built once and the identity gate applied:

```rust
use stacker_align::{
    Matrix3, Motion, MotionRefine, Optimizer, apply_identity_gate, refine::LumaPyramid,
    refine_motion, transform::warp_image,
};
use stacker_core::{error::StackerError, image::PlanarImage};

fn align_stack(
    reference: &PlanarImage<f32>,
    frames: &[PlanarImage<f32>],
) -> Result<Vec<PlanarImage<f32>>, StackerError> {
    let pyramid = LumaPyramid::new(reference);
    // Shift, scale and rotation, Lucas-Kanade with the simplex as the fallback.
    let cfg = MotionRefine {
        optimizer: Optimizer::Auto,
        bounded: true,
        ..MotionRefine::new(Motion::Similarity)
    };
    frames
        .iter()
        .map(|frame| -> Result<PlanarImage<f32>, StackerError> {
            let refined = refine_motion(reference, &pyramid, frame, &Matrix3::identity(), &cfg)?;
            let matrix = apply_identity_gate(reference, frame, &refined.matrix);
            let apply = matrix.try_inverse().ok_or_else(|| {
                StackerError::MathError("the refined matrix is not invertible".into())
            })?;
            warp_image(frame, &apply)
        })
        .collect()
}
```

The apps do not call these pieces one by one. `stacker_pipeline::align::align_frame(reference, frame, &AlignmentSettings::from(&settings), seed)` performs the whole step (mode dispatch, refinement, the identity gate, the inversion, the warp, the coverage mask, brightness correction, and the fallbacks when a refinement or a warp fails) and `align_sequence` drives it over a stack, including the chain mode and the neural seeds; use those unless you need a single piece.
