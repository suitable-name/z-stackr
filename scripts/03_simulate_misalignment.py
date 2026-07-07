#!/usr/bin/env python3
"""
03_simulate_misalignment.py — Stage 3 of focus-stack training-data generation.

Takes a perfectly aligned focus stack (from 02_blender_focus_stack.py) and
generates a new dataset with simulated misalignment (focus breathing,
anisotropic scale, translation, rotation, shear) for training
`z-stackr-nn`'s alignment network (`BatchAlignNet`). To avoid black/replicated
borders ever appearing anywhere in the training set, every frame (including
the reference) is warped and then centre-cropped by a fixed safety margin
that is provably larger than the maximum possible per-frame displacement —
see `compute_crop_margin()` / design-doc §4.1. Per-frame photometric and
noise augmentation (design-doc §4.3) is applied on top, so the alignment
network also learns robustness to brightness/gamma/noise/blur differences
between frames, not just geometry.

Usage:
    python3 scripts/03_simulate_misalignment.py \
        --input dataset/scene \
        --out dataset/scene_unaligned \
        --max-shift-min 0.01 \
        --max-shift-max 0.05 \
        --max-angle 2.0 \
        --max-log-scale 0.03 \
        --max-shear 0.01 \
        --seed 0

    # Smoke-test with all frames = copies of stack_files[0] instead of the
    # real per-plane defocus renders (fast, no real optical variation):
    python3 scripts/03_simulate_misalignment.py --input ... --out ... --flat

    # Run the standalone self-check (synthetic data, no dataset required):
    python3 scripts/03_simulate_misalignment.py --self-test

Ground-truth matrix convention (`metadata.json["alignment_gt"]`)
------------------------------------------------------------------
`alignment_gt` is a list of 3x3 row-major matrices, one per `stack` entry,
each mapping **homogeneous PIXEL coordinates of the reference (cropped)
frame** to **homogeneous PIXEL coordinates of that stack entry's unaligned,
cropped output frame** — both in the OUTPUT `[w, h]` space recorded as
`metadata.json["cropped_dims"]` (every emitted image, including `allfocus`,
`masks`, and `occlusion`, shares this size). Concretely, for frame k:

    [x_k, y_k, 1]^T = M_k @ [x_ref, y_ref, 1]^T

where `(x_ref, y_ref)` is a pixel location in the reference frame's cropped
output image and `(x_k, y_k)` is the corresponding pixel location in
`stack[k]`'s emitted, misaligned, cropped image. To resample frame k back
onto the reference grid (the correction a trained alignment network is used
for), warp frame k by `M_k` in `cv2.warpAffine`'s forward-mapping convention
(the same one `warp_image` below uses: `dst(x) = src(A @ x)`), i.e. pass
`M_k` directly, NOT its inverse.

Frame index 0 is always the reference: it receives NO random perturbation
(`A_0 = identity` before cropping), so its post-crop `alignment_gt` entry is
exactly the identity matrix and it is never photometrically augmented — see
`process_dataset()` and §4.2/§4.3 below for the rationale.

`z-stackr-nn`'s Rust loader (`stacker_nn::data::AlignSample`) converts these
PIXEL-space matrices to the crate-wide NORMALIZED `[-1, 1]^2` convention (see
`stacker_nn::bridge::align_planar`'s docs for the exact conjugation) using
the CROPPED `[w, h]` from `metadata.json["cropped_dims"]` — this script
intentionally stays in pixel space since that's the natural space for
`cv2.warpAffine`/manual affine construction, and the Rust side needs no
change to consume the crop-adjusted matrices produced here.

Crop-margin logic (design doc §4.1 — "kill the border shortcut")
------------------------------------------------------------------
Every random per-frame affine transform can push image content around by at
most a bounded amount (translation + rotation + scale + shear, all clamped
by the CLI ranges below). If we only warped and never cropped, pixels near
the border of the warped output would sometimes come from
`cv2.BORDER_CONSTANT`-filled (or, with a different border mode, clamped/
replicated) regions rather than genuine scene content — a trivially
learnable shortcut that has nothing to do with real alignment. To prevent
this, we:

  1. Compute a single `margin_px` (see `compute_crop_margin()`) that is
     provably larger than the worst-case displacement any sampled transform
     can produce at the image boundary.
  2. Warp each frame at the ORIGINAL resolution (so warping never runs out
     of source pixels near the middle of the frame, where the final crop
     will land).
  3. Centre-crop every single emitted image (all stack frames, reference,
     `allfocus`, `masks`, `occlusion`) by exactly `margin_px` on every side.
     Because the crop is identical for all frames, no image in the final
     dataset ever exposes a synthetic border pixel.
  4. Re-express every ground-truth matrix for the cropped coordinate frame
     via `crop_matrix()` (a similarity conjugation by the crop's pure
     translation), so `alignment_gt` is correct in the FINAL, cropped pixel
     space that ships in the dataset.

This replaces the old `--crop-factor` fractional-crop behavior; see the
deprecation note on `--crop-factor` below.

Photometric augmentation (design doc §4.3)
------------------------------------------------------------------
After warping and cropping, every NON-reference frame independently receives:
  - brightness scale ~ U[0.9, 1.1]
  - gamma ~ U[0.9, 1.1]
  - additive Gaussian noise, sigma ~ U[0, 0.01] in normalized [0, 1] domain
  - with probability 0.3, extra Gaussian blur, sigma ~ U[0, 1.0] px

DELIBERATE IMPLEMENTER DECISION: the reference frame (index 0) is NOT
photometrically augmented, even though design-doc §4.3's text says "per
frame" without explicitly excluding the reference. We exclude it so the
reference stays a clean, noise-free anchor — the network's job is to learn
robustness to photometric differences BETWEEN a (clean-ish) reference and
the frames being aligned to it, which is what a real focus-stack aligner
sees (the base/reference frame is typically the one everything else is
registered against, and adding independent noise to it too would just make
the learning target noisier without adding realism, since real capture
noise is per-shot and already implicitly present in the original renders).
If you want reference augmentation too, flip `_AUGMENT_REFERENCE` below.

DELIBERATE IMPLEMENTER DECISION (bit-depth / noise domain): the upstream
renders may be 8-bit or 16-bit (`cv2.IMREAD_UNCHANGED` is used to read
whatever `02_blender_focus_stack.py` produced). To keep the noise sigma
meaningful regardless of source bit depth, every image is converted to a
`float64` array normalized to [0, 1] (dividing by 255 or 65535 as detected
from dtype) before brightness/gamma/noise/blur are applied, and converted
back to the original dtype (with clipping) before being written out. Gaussian
noise sigma=0.01 therefore means "1% of full-scale", consistently, regardless
of whether the source is 8-bit or 16-bit.

Transform ranges (design doc §4.4)
------------------------------------------------------------------
  - translation: per-axis, magnitude drawn from U[max_shift_min, max_shift_max]
    (default min=0.01, max=0.05) as a fraction of the corresponding
    dimension, with an independently randomized sign per axis — i.e. we
    explicitly avoid sampling near-zero shifts by drawing the MAGNITUDE from
    U[min, max] and then giving it a random sign, rather than drawing
    directly from U[-max, max] (which would put a lot of mass near zero and
    under-exercise the "genuinely misaligned by at least ~1%" regime the
    network needs to learn to correct).
  - rotation: uniform in [-max_angle, max_angle] degrees, default max_angle
    = 2.0 deg (~0.035 rad).
  - log-scale: INDEPENDENT per-axis, uniform in [-max_log_scale,
    max_log_scale], default 0.03, applied as sx = exp(u_x), sy = exp(u_y).
    This is a change from the old script's single isotropic `s` fed into
    `cv2.getRotationMatrix2D` (which only supports uniform scale) — genuine
    anisotropic scale now requires building the affine matrix by hand; see
    `build_random_affine()`.
  - shear: NEW. uniform in [-max_shear, max_shear], default 0.01, applied as
    an off-diagonal shear term in the linear part of the affine matrix.

These ranges are deliberately kept well inside the Rust model's tanh output
bounds (from docs/batchalign-v2-design.md §3.4: tx/ty <= 0.30 normalized,
rotation <= 0.10 rad, log-scale <= 0.08, shear <= 0.05) so that every
training sample is representable by the model without saturating its
output activations:
    rotation:  2 deg ~= 0.035 rad  <  0.10 rad   (~2.9x headroom)
    log-scale: 0.03               <  0.08        (~2.7x headroom)
    shear:     0.01               <  0.05        (~5x headroom)
    translation: 5% pixel-fraction of dimension is used here as a reasonable
                 proxy for the normalized-unit bound of 0.30; it is well
                 inside it.
If you change any of `--max-angle` / `--max-log-scale` / `--max-shear` /
the shift range, cross-check against docs/batchalign-v2-design.md §3.4
so the sampled ground truth stays representable by the model's tanh heads.

Self-test (design doc §4.1 — "add a self-check")
------------------------------------------------------------------
`self_test()` builds a tiny synthetic image and a handful of known random
affine matrices entirely in memory (no dataset required), warps a probe
point grid with the UNCROPPED matrix, crops, and verifies that
`crop_matrix()`'s cropped matrix reproduces the same mapped locations to
within 0.1 px. This runs automatically (fast, ~milliseconds) at the start
of every invocation of this script as a cheap sanity check, and can also be
run standalone, verbosely, via `--self-test` (exits 0 on success, 1 on
failure) so it is CI-able without any dataset present.
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Design-doc §4.3: whether the reference frame (index 0) also receives
# photometric augmentation. Default False — see module docstring for the
# rationale ("reference stays a clean anchor").
# ---------------------------------------------------------------------------
_AUGMENT_REFERENCE = False

# ---------------------------------------------------------------------------
# Design-doc §4.4 cross-reference: these bounds must stay well inside the
# Rust model's tanh output bounds documented in
# docs/batchalign-v2-design.md §3.4 (tx/ty: 0.30 normalized, rotation:
# 0.10 rad, log-scale: 0.08, shear: 0.05). See module docstring for the
# margin-of-safety arithmetic. If you touch these defaults, re-check that
# doc so the two stay consistent.
# ---------------------------------------------------------------------------
DEFAULT_MAX_SHIFT_MIN = 0.01
DEFAULT_MAX_SHIFT_MAX = 0.05
DEFAULT_MAX_ANGLE_DEG = 2.0
DEFAULT_MAX_LOG_SCALE = 0.03
DEFAULT_MAX_SHEAR = 0.01


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stage 3: simulate misalignment + photometric augmentation "
                     "for BatchAlignNet training data."
    )
    ap.add_argument("--input", required=False, default=None,
                     help="Input dataset directory (from stage 02). Not required with --self-test.")
    ap.add_argument("--out", required=False, default=None,
                     help="Output directory for the unaligned dataset. Not required with --self-test.")
    ap.add_argument("--crop-factor", type=float, default=None,
                     help="DEPRECATED / no-op. The margin-based crop (design-doc §4.1) is now "
                          "mandatory and fully determined by the transform ranges below; there is "
                          "no longer a free-standing crop fraction to choose. This flag is accepted "
                          "for backward-compatible CLI parsing only and, if passed, prints a "
                          "deprecation warning and is otherwise ignored.")
    ap.add_argument("--max-shift-min", type=float, default=DEFAULT_MAX_SHIFT_MIN,
                     help="Minimum per-axis translation magnitude, as a fraction of that "
                          "dimension (default 0.01 = 1%%). Sign is randomized independently.")
    ap.add_argument("--max-shift-max", type=float, default=DEFAULT_MAX_SHIFT_MAX,
                     help="Maximum per-axis translation magnitude, as a fraction of that "
                          "dimension (default 0.05 = 5%%). Sign is randomized independently.")
    ap.add_argument("--max-angle", type=float, default=DEFAULT_MAX_ANGLE_DEG,
                     help="Maximum absolute rotation angle in degrees (default 2.0).")
    ap.add_argument("--max-log-scale", type=float, default=DEFAULT_MAX_LOG_SCALE,
                     help="Maximum absolute per-axis log-scale (default 0.03); "
                          "sx = exp(U[-max,max]), sy = exp(U[-max,max]) independently.")
    ap.add_argument("--max-shear", type=float, default=DEFAULT_MAX_SHEAR,
                     help="Maximum absolute shear coefficient (default 0.01).")
    ap.add_argument("--flat", action="store_true",
                     help="Legacy smoke-test mode: every emitted 'frame' is a warped/augmented "
                          "copy of stack_files[0] (the sharpest/first Blender render) instead of "
                          "each frame's genuine distinct defocus render. Useful for a fast pipeline "
                          "smoke test where optical realism doesn't matter. Default: off, i.e. the "
                          "real per-plane defocus stack from stage 02 is used.")
    ap.add_argument("--no-augment", action="store_true",
                     help="Disable photometric/noise augmentation (§4.3) entirely; only geometric "
                          "misalignment is applied. Useful for isolating geometry-only regressions.")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible perturbations "
                                                             "(covers both geometric and photometric draws).")
    ap.add_argument("--self-test", action="store_true",
                     help="Run the standalone geometric self-check (§4.1) on synthetic in-memory "
                          "data only, print the result, and exit (0 = pass, 1 = fail). No --input/"
                          "--out required.")
    args = ap.parse_args()

    if args.self_test:
        return args

    if not args.input or not args.out:
        ap.error("--input and --out are required unless --self-test is given")

    return args


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def compute_crop_margin(w, h, max_shift_max, max_angle_deg, safety_px=8):
    """
    Computes the fixed centre-crop margin (in pixels, applied identically on
    all four sides of every emitted image) that guarantees no synthetic
    border pixel from `cv2.warpAffine`'s fill (or, in principle, clamped
    edge content) can survive into the final cropped dataset.

    Design-doc §4.1 formula:
        margin_px = ceil(max_translation_frac * max(W, H))
                  + ceil(|sin(max_rot)| * max(W, H) / 2)
                  + safety_px

    The first term bounds the worst-case translation displacement; the
    second bounds the worst-case displacement a rotation by `max_angle_deg`
    can introduce at the edge of the frame (a point at radius max(W,H)/2 from
    the centre moves by roughly `radius * sin(theta)` under pure rotation);
    `safety_px` is a small fixed pad absorbing scale/shear's (much smaller,
    already-bounded) contribution and general numerical slack.

    Returns an int margin, identical for x and y (a single scalar, since the
    crop is applied symmetrically on both axes for simplicity and because
    `max(W, H)` is already used as the conservative dimension in the
    translation term).
    """
    dim = max(w, h)
    max_rot_rad = math.radians(max_angle_deg)
    margin = (
        math.ceil(max_shift_max * dim)
        + math.ceil(abs(math.sin(max_rot_rad)) * dim / 2.0)
        + safety_px
    )
    return int(margin)


def build_random_affine(rng, w, h, max_shift_min, max_shift_max, max_angle_deg,
                         max_log_scale, max_shear, identity=False):
    """
    Builds a random affine matrix `A_k` (3x3, homogeneous, row-major) mapping
    the reference's UNCROPPED pixel grid `[0, w] x [0, h]` (centred at
    (w/2, h/2)) onto the corresponding location in the same-sized uncropped
    frame k, before any centre-cropping is applied.

    If `identity` is True, returns the exact 3x3 identity matrix and draws
    no random numbers (used for the reference frame, index 0 — design-doc
    §4.2: the reference receives no perturbation).

    Otherwise, independently samples (design-doc §4.4):
      - tx, ty:      magnitude ~ U[max_shift_min, max_shift_max] * {w, h}
                     respectively, sign randomized independently per axis
                     (avoids under-sampling near-zero shifts).
      - angle:       ~ U[-max_angle_deg, max_angle_deg] degrees.
      - sx, sy:      independent per-axis log-scale, exp(U[-max_log_scale,
                     max_log_scale]) each (genuine anisotropic scale — this
                     is why we can't just use `cv2.getRotationMatrix2D`,
                     which only supports a single isotropic scale factor).
      - shear:       ~ U[-max_shear, max_shear], applied as an off-diagonal
                     term in the linear part before rotation is applied.

    The linear part is composed as `R @ Shear @ Scale` (rotation applied
    last, around the crop centre), and translation is added in absolute
    pixels around that same centre:

        A_k = [ R @ Shear @ Scale   |  t ]
              [        0    0      |  1 ]

    where `t` recentres the transform on (cx, cy) and then applies (tx, ty).
    """
    cx, cy = w / 2.0, h / 2.0

    if identity:
        return np.eye(3, dtype=np.float64)

    # --- translation: magnitude in [min, max], random sign, per axis ---
    def _signed_magnitude(lo, hi):
        mag = rng.uniform(lo, hi)
        sign = rng.choice([-1.0, 1.0])
        return mag * sign

    tx_frac = _signed_magnitude(max_shift_min, max_shift_max)
    ty_frac = _signed_magnitude(max_shift_min, max_shift_max)
    tx = tx_frac * w
    ty = ty_frac * h

    # --- rotation ---
    angle_deg = rng.uniform(-max_angle_deg, max_angle_deg)
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    # --- independent per-axis log-scale (anisotropic) ---
    log_sx = rng.uniform(-max_log_scale, max_log_scale)
    log_sy = rng.uniform(-max_log_scale, max_log_scale)
    sx, sy = math.exp(log_sx), math.exp(log_sy)
    Scale = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)

    # --- shear (single off-diagonal coefficient on the x-row) ---
    shear = rng.uniform(-max_shear, max_shear)
    Shear = np.array([[1.0, shear], [0.0, 1.0]], dtype=np.float64)

    linear = R @ Shear @ Scale

    # Compose the full affine so that it rotates/scales/shears about the
    # centre (cx, cy) and then translates by (tx, ty):
    #   x' = linear @ (x - c) + c + t
    A2x3 = np.zeros((2, 3), dtype=np.float64)
    A2x3[:2, :2] = linear
    A2x3[:, 2] = np.array([cx, cy]) - linear @ np.array([cx, cy]) + np.array([tx, ty])

    A_k = np.vstack([A2x3, [0.0, 0.0, 1.0]])
    return A_k


def crop_matrix(M_px, mx, my):
    """
    Re-expresses a pixel-space matrix `M_px` (valid on the UNCROPPED image)
    for the coordinate frame of an image that has been centre-cropped by
    `(mx, my)` pixels (i.e. the crop `[my : H - my, mx : W - mx]`, identical
    convention on both the domain and codomain sides since every emitted
    image in this pipeline uses the SAME crop).

    Cropping by `(mx, my)` is the pixel-space translation
        C = [[1, 0, -mx],
             [0, 1, -my],
             [0, 0,  1 ]]
    (a cropped-image pixel coordinate equals the uncropped coordinate minus
    the crop offset). Since both the reference and frame-k coordinate
    systems are cropped identically, the cropped-space ground-truth matrix
    is the similarity conjugation:

        M_px_cropped = C @ M_px @ C^-1

    Design-doc §4.1 requires exactly this composition; see `self_test()`
    for a numeric verification that it reproduces the correct point
    mappings.
    """
    C = np.array([
        [1.0, 0.0, -mx],
        [0.0, 1.0, -my],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    C_inv = np.array([
        [1.0, 0.0, mx],
        [0.0, 1.0, my],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return C @ M_px @ C_inv


def warp_image(img, A, out_shape):
    """
    Warps `img` using the affine transform `A` (3x3 homogeneous, forward
    convention) to produce an image of size `out_shape = (w, h)`.

    Forward-mapping convention: `dst(x) = src(A @ x)` for output pixel `x`
    (implemented via `cv2.warpAffine`'s inverse-map argument, `A^-1`, since
    OpenCV's `M` parameter is itself the dst->src inverse map).
    """
    A_inv = np.linalg.inv(A)
    M = A_inv[:2, :]
    w, h = out_shape
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def center_crop(img, margin):
    """
    Centre-crops `img` by `margin` pixels on every side. `img` may be 2D
    (single channel / mask) or 3D (H, W, C). Returns a view/copy of shape
    `(H - 2*margin, W - 2*margin[, C])`.
    """
    h, w = img.shape[:2]
    if 2 * margin >= h or 2 * margin >= w:
        raise ValueError(
            f"Crop margin {margin}px is too large for image of size {w}x{h}; "
            "reduce the transform ranges or check compute_crop_margin()."
        )
    return img[margin: h - margin, margin: w - margin, ...]


# ---------------------------------------------------------------------------
# Photometric augmentation (design-doc §4.3)
# ---------------------------------------------------------------------------

def _to_unit_float(img):
    """
    Converts `img` (uint8 or uint16, as produced by cv2.imread(...,
    IMREAD_UNCHANGED)) to a float64 array normalized to [0, 1], and returns
    (unit_float_img, original_dtype, original_max_value) so the inverse
    conversion can round-trip exactly for unmodified pixels.
    """
    if img.dtype == np.uint8:
        max_val = 255.0
    elif img.dtype == np.uint16:
        max_val = 65535.0
    else:
        # Fallback: assume already-float or unusual dtype is already in a
        # sensible range; treat its own max as 1.0 equivalent won't be
        # correct in general, so we conservatively assume [0, 1] already.
        max_val = 1.0
    return img.astype(np.float64) / max_val, img.dtype, max_val


def _from_unit_float(img_unit, dtype, max_val):
    """Inverse of `_to_unit_float`: clips to [0, 1] then rescales/casts back."""
    clipped = np.clip(img_unit, 0.0, 1.0)
    return (clipped * max_val).astype(dtype)


def apply_photometric_augmentation(img, rng, max_noise_sigma=0.01, blur_prob=0.3, max_blur_sigma=1.0):
    """
    Applies independent per-frame photometric + noise augmentation
    (design-doc §4.3) to `img` (uint8 or uint16, any number of channels):

      1. brightness scale ~ U[0.9, 1.1]      (multiplicative, in [0,1] domain)
      2. gamma ~ U[0.9, 1.1]                  (img ** gamma, in [0,1] domain)
      3. additive Gaussian noise, sigma ~ U[0, max_noise_sigma], in the same
         normalized [0, 1] linear-ish domain (see module docstring for the
         exact bit-depth-independence rationale).
      4. with probability `blur_prob`, extra Gaussian blur with sigma ~
         U[0, max_blur_sigma] pixels, simulating residual motion blur /
         focus error.

    Returns an array of the same dtype and shape as `img`. Deterministic
    given `rng`'s state (caller controls seeding/derivation for
    reproducibility).
    """
    unit, dtype, max_val = _to_unit_float(img)

    brightness = rng.uniform(0.9, 1.1)
    gamma = rng.uniform(0.9, 1.1)
    noise_sigma = rng.uniform(0.0, max_noise_sigma)

    out = unit * brightness
    out = np.clip(out, 0.0, 1.0) ** gamma

    if noise_sigma > 0.0:
        noise = rng.normal(loc=0.0, scale=noise_sigma, size=out.shape)
        out = out + noise

    out = np.clip(out, 0.0, 1.0)

    if rng.uniform(0.0, 1.0) < blur_prob:
        blur_sigma = rng.uniform(0.0, max_blur_sigma)
        if blur_sigma > 1e-6:
            # ksize=0 lets OpenCV derive an appropriate kernel size from sigma.
            out = cv2.GaussianBlur(out, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    return _from_unit_float(out, dtype, max_val)


# ---------------------------------------------------------------------------
# Self-test (design-doc §4.1 mandatory self-check)
# ---------------------------------------------------------------------------

def self_test(verbose=True):
    """
    Standalone geometric self-check, runnable without any real dataset.

    Builds a synthetic probe point grid over a fake uncropped frame of size
    (W, H) = (200, 160), generates a handful of random affine matrices via
    `build_random_affine()` (including one identity, mimicking the
    reference), computes each matrix's cropped counterpart via
    `crop_matrix()` given a synthetic `compute_crop_margin()`-derived
    margin, and verifies:

        M_px_cropped @ [x_ref_cropped, y_ref_cropped, 1]
            == (M_px @ [x_ref_uncropped, y_ref_uncropped, 1]) - (mx, my)

    i.e. mapping a point in cropped reference coordinates with the cropped
    matrix must agree (within 0.1 px) with mapping the corresponding
    uncropped point with the uncropped matrix and then subtracting the crop
    offset.

    Returns True if all checks pass (prints a summary if `verbose`), False
    otherwise (prints the first failure).
    """
    rng = np.random.default_rng(12345)
    w, h = 200, 160
    max_shift_min, max_shift_max = DEFAULT_MAX_SHIFT_MIN, DEFAULT_MAX_SHIFT_MAX
    max_angle_deg = DEFAULT_MAX_ANGLE_DEG
    max_log_scale = DEFAULT_MAX_LOG_SCALE
    max_shear = DEFAULT_MAX_SHEAR

    margin = compute_crop_margin(w, h, max_shift_max, max_angle_deg)
    if 2 * margin >= min(w, h):
        if verbose:
            print(f"[self-test] FAIL: margin {margin}px too large for probe image {w}x{h}")
        return False

    # A small grid of probe points, all safely inside the CROPPED region.
    xs = np.linspace(margin + 1, w - margin - 1, 5)
    ys = np.linspace(margin + 1, h - margin - 1, 5)
    grid = np.array([[x, y, 1.0] for x in xs for y in ys], dtype=np.float64)  # (N, 3)

    test_cases = [("identity/reference", True)] + [(f"random_{i}", False) for i in range(5)]

    all_ok = True
    max_err_seen = 0.0
    for name, is_identity in test_cases:
        M_px = build_random_affine(
            rng, w, h, max_shift_min, max_shift_max, max_angle_deg,
            max_log_scale, max_shear, identity=is_identity,
        )
        M_px_cropped = crop_matrix(M_px, margin, margin)

        # Uncropped-space mapping, then shift into cropped coordinates.
        mapped_uncropped = (M_px @ grid.T).T  # (N, 3), homogeneous
        mapped_uncropped_xy = mapped_uncropped[:, :2] / mapped_uncropped[:, 2:3]
        expected_cropped_xy = mapped_uncropped_xy - np.array([margin, margin])

        # Cropped-space mapping using the cropped matrix directly, starting
        # from the reference points expressed in CROPPED coordinates.
        grid_cropped = grid.copy()
        grid_cropped[:, 0] -= margin
        grid_cropped[:, 1] -= margin
        mapped_cropped = (M_px_cropped @ grid_cropped.T).T
        mapped_cropped_xy = mapped_cropped[:, :2] / mapped_cropped[:, 2:3]

        err = np.abs(mapped_cropped_xy - expected_cropped_xy)
        max_err = float(err.max())
        max_err_seen = max(max_err_seen, max_err)

        ok = max_err <= 0.1
        all_ok = all_ok and ok
        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"[self-test] {name}: max |err| = {max_err:.6f} px -> {status}")
        if not ok:
            if verbose:
                print(f"[self-test] FAIL: matrix mismatch exceeds 0.1px tolerance for case '{name}'")
            return False

    if verbose:
        print(f"[self-test] all {len(test_cases)} cases passed (worst-case error "
              f"{max_err_seen:.6f} px <= 0.1 px tolerance)")
    return all_ok


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_dataset(args):
    """
    Runs the full stage-3 pipeline: reads stage-2's metadata.json, computes
    the shared crop margin, generates per-frame random affine transforms
    (identity for the reference), warps + crops + photometrically augments
    every stack frame, centre-crops the shared reference-space images
    (allfocus/masks/occlusion), and writes the new metadata.json with
    crop-adjusted `alignment_gt`.
    """
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Separate child RNG for photometric augmentation so that adding/removing
    # augmentation calls never perturbs the geometric transform sequence (and
    # vice versa), while both remain fully determined by --seed.
    photo_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

    if args.crop_factor is not None:
        print("[WARNING] --crop-factor is deprecated and ignored: the margin-based "
              "crop from design-doc §4.1 (compute_crop_margin) is now mandatory and "
              "fully determined by --max-shift-max/--max-angle. Remove this flag from "
              "your invocation; it will be rejected in a future version.")

    meta_path = os.path.join(args.input, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    stack_files = meta.get("stack", [])
    if not stack_files:
        sys.exit("No stack files found in metadata.json")

    first_img_path = os.path.join(args.input, stack_files[0])
    img0 = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
    if img0 is None:
        sys.exit(f"Failed to read {first_img_path}")

    H, W = img0.shape[:2]

    margin = compute_crop_margin(W, H, args.max_shift_max, args.max_angle)
    w = W - 2 * margin
    h = H - 2 * margin
    if w <= 0 or h <= 0:
        sys.exit(
            f"Computed crop margin ({margin}px per side) leaves no image "
            f"({W}x{H} -> {w}x{h}); reduce --max-shift-max/--max-angle."
        )
    print(f"[info] original dims {W}x{H}, crop margin {margin}px per side, "
          f"final cropped dims {w}x{h}")

    alignment_gt = []

    for idx, fname in enumerate(stack_files):
        is_reference = (idx == 0)

        # §4.2: real per-frame defocus stack by default; --flat reuses
        # stack_files[0] for every frame as a fast, optically-unrealistic
        # smoke test.
        src_fname = stack_files[0] if args.flat else fname
        img_path = os.path.join(args.input, src_fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f"Failed to read {img_path}")

        # §4.2: the reference (idx 0) gets NO random perturbation.
        A_k_uncropped = build_random_affine(
            rng, W, H,
            args.max_shift_min, args.max_shift_max,
            args.max_angle, args.max_log_scale, args.max_shear,
            identity=is_reference,
        )

        warped_full = warp_image(img, A_k_uncropped, (W, H))
        out_img = center_crop(warped_full, margin)

        # §4.3: photometric augmentation, skipped for the reference frame
        # (see _AUGMENT_REFERENCE / module docstring for the rationale).
        if not args.no_augment and (_AUGMENT_REFERENCE or not is_reference):
            out_img = apply_photometric_augmentation(out_img, photo_rng)

        out_path = os.path.join(args.out, fname)
        cv2.imwrite(out_path, out_img)

        # §4.1: re-express the ground-truth matrix for the cropped output
        # space. Since the reference is ALSO cropped identically, and A_k is
        # already defined as reference-uncropped -> frame-k-uncropped, the
        # crop conjugation with the SAME (margin, margin) offset on both
        # sides of the mapping correctly yields reference-cropped ->
        # frame-k-cropped.
        M_k_cropped = crop_matrix(A_k_uncropped, margin, margin)
        alignment_gt.append(M_k_cropped.tolist())

        tag = "reference (identity)" if is_reference else "warped+augmented"
        print(f"Processed {fname} [{tag}]" + (" (--flat: source=stack[0])" if args.flat else ""))

    # Reference-space images (allfocus, masks, occlusion) only need the
    # shared centre-crop applied — they already live in the reference's
    # (uncropped) pixel grid, since stage 02 emits them aligned to it.
    other_files = [meta.get("allfocus"), meta.get("occlusion")] + meta.get("masks", [])
    for fname in other_files:
        if not fname:
            continue
        img_path = os.path.join(args.input, fname)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                out_img = center_crop(img, margin)
                out_path = os.path.join(args.out, fname)
                cv2.imwrite(out_path, out_img)
                print(f"Centrally cropped {fname}")

    meta["alignment_gt"] = alignment_gt
    meta["original_dims"] = [W, H]
    meta["cropped_dims"] = [w, h]
    meta["crop_margin_px"] = margin
    meta["transform_ranges"] = {
        "max_shift_min_frac": args.max_shift_min,
        "max_shift_max_frac": args.max_shift_max,
        "max_angle_deg": args.max_angle,
        "max_log_scale": args.max_log_scale,
        "max_shear": args.max_shear,
    }
    meta["flat_mode"] = bool(args.flat)
    meta["photometric_augmentation"] = (not args.no_augment)
    # crop_factor intentionally omitted from output metadata: superseded by
    # crop_margin_px / cropped_dims (see --crop-factor deprecation notice).

    out_meta_path = os.path.join(args.out, "metadata.json")
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done! Unaligned dataset saved to {args.out}")


if __name__ == "__main__":
    args = parse_args()

    if args.self_test:
        ok = self_test(verbose=True)
        sys.exit(0 if ok else 1)

    # Cheap automatic sanity check on every real invocation too (design-doc
    # §4.1: "runs automatically ... using synthetic data so it needs no
    # dataset"). Kept silent on success, loud on failure, and fast (a few ms).
    if not self_test(verbose=False):
        sys.exit(
            "[FATAL] Internal geometric self-check failed (crop_matrix/build_random_affine "
            "convention mismatch) — refusing to generate a dataset with potentially incorrect "
            "ground truth. Run with --self-test for a verbose report."
        )

    process_dataset(args)
