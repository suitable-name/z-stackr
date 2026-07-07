#!/usr/bin/env python3
"""
scripts/04_eval_alignment.py
=============================

Evaluation harness for the z-stackr alignment network, implementing design-doc
Section 7 ("Evaluation gate for dropping the 'experimental' label from
batchalign-v2").

WHAT THIS SCRIPT DOES
----------------------
For every held-out synthetic scene directory under ``--scenes-dir`` (each
produced by ``scripts/03_simulate_misalignment.py`` and never used in
training), this script computes the **mean corner error in pixels** between
predicted and ground-truth alignment matrices for up to three methods:

    (a) "neural"        -- the trained alignment network run alone
    (b) "neural_refine" -- the network's output refined by the classical
                            (AKAZE/RANSAC/intensity) refinement pass
    (c) "classical"      -- classical AKAZE-seeded alignment alone, used as
                            the baseline the neural methods are judged against

It then aggregates per-scene numbers into a report (mean / median / worst
scene, both in raw full-resolution pixels and normalized to a 3000px-wide
reference image), runs a paired statistical comparison between (b) and (c),
measures neural wall-clock time on one representative scene, and prints a
PASS/FAIL verdict against each of the three design-doc §7 acceptance
bullets. The same data is optionally written out as JSON via ``--output``.

GROUND-TRUTH CONVENTION (copied verbatim from scripts/03_simulate_misalignment.py
-- this is load-bearing, read it carefully before touching any matrix math
below)
------------------------------------------------------------------------------
``alignment_gt[k]`` is a 3x3 matrix ``M_k`` such that

    [x_k, y_k, 1]^T = M_k @ [x_ref, y_ref, 1]^T

i.e. it maps a PIXEL coordinate in the reference (frame 0) to the
corresponding PIXEL coordinate in frame k, both expressed in the shared
``cropped_dims`` pixel space that every emitted PNG in the scene directory
uses. Frame 0's own ``alignment_gt[0]`` is the identity matrix.

Predicted matrices (from the neural/classical methods below) are expected in
the exact same convention: ``pred[k]`` maps reference-frame pixel coordinates
to frame-k pixel coordinates, in ``cropped_dims`` space. This lets predicted
and ground-truth matrices be compared directly without any change of basis.

WHY THIS SCRIPT SHELLS OUT (dependency on an eval-inference binary)
---------------------------------------------------------------------
There is no Python binding for the Rust ``z-stackr-nn`` / ``z-stackr-align``
crates. The only way to invoke the trained model (and the classical
refinement machinery that sits on top of it) from Python is to shell out to
a compiled Rust CLI binary.

``stacker-nn-train`` (the existing training binary) does not expose an
inference-only mode, so this script assumes a **dedicated eval-inference
binary** which, as of this writing, DOES NOT YET EXIST and must be built by
whoever runs this evaluation. This is a documented TODO / dependency, not a
bug in this script:

    TODO(z-stackr-nn): build a small Rust binary (suggested name:
    "stacker-nn-eval", suggested location: crates/z-stackr-nn/examples/ or a
    new crates/z-stackr-nn-eval/ bin target) that implements the JSON I/O
    contract documented in ``run_eval_binary()`` below.

Because that binary does not exist yet, this script is designed to fail
*gracefully* and *per-column* if it is missing or errors: the "neural alone"
and "neural + refine" columns are computed independently and either one can
come back as "unavailable" (with a clear, actionable stderr message) without
crashing the rest of the comparison.

CLASSICAL BASELINE: DOCUMENTED AS TODO, NOT IMPLEMENTED
---------------------------------------------------------
The design doc explicitly allows the classical-baseline column to be either
(a) shelled out to ``z-stackr-cli`` with some invented flag convention to
recover solved matrices, or (b) marked "not implemented" with documented
reasoning, provided the other two columns (neural alone, neural + refine)
are fully implemented.

This script takes choice (b). Rationale, spelled out here and repeated as an
inline comment at the call site in ``run_classical_baseline()``:

    ``z-stackr-cli`` (the ``z-stackr`` command) aligns and stacks a folder of
    images end-to-end, but has no flag today that dumps the *solved
    per-frame alignment matrices* as machine-readable output -- it only
    produces a final fused image. Inventing an "inferred flag convention"
    against a CLI surface that does not actually support this would mean
    guessing at (and hard-coding around) behavior that doesn't exist, which
    is more likely to silently rot into a misleading result than to save
    real work. A future patch that adds a `--dump-alignment-json` (or
    similar) mode to z-stackr-cli / z-stackr-align is the correct fix; at
    that point ``run_classical_baseline()`` below is the single function
    that needs to change.

The classical column therefore always reports ``None`` / "N/A -- classical
baseline not implemented" in this version of the script, but is fully wired
into the aggregation, statistics, and verdict logic so that plugging in a
real implementation later requires no structural changes elsewhere.

CORNER-ERROR METRIC
--------------------
For each held-out scene:
  1. Read ``cropped_dims = (w, h)`` from metadata.json. All emitted images in
     the scene (including frame 0, the reference) share this size.
  2. Corners are defined in the CROPPED image's pixel space (the space
     ``alignment_gt`` lives in): ``[(0,0), (w-1,0), (0,h-1), (w-1,h-1)]``.
  3. For every non-reference frame k, transform each corner with the
     predicted matrix and with ``alignment_gt[k]``, and take the Euclidean
     distance between the two transformed points.
  4. Average the 4 per-corner distances -> one number per frame.
  5. Average over all non-reference frames in the scene -> one number per
     scene per method. This is the scene's "raw" corner error, reported in
     full-resolution pixels of the scene's actual ``cropped_dims`` width.

NORMALIZATION TO 3000px WIDTH
-------------------------------
The acceptance criteria states the target as "mean corner error < 8 px at
3000 px image width". Held-out scenes may not all be rendered at exactly
3000px wide, so this script reports BOTH:
  - ``raw_px``: the corner error as measured in the scene's actual
    ``cropped_dims`` pixel space (no scaling).
  - ``normalized_3000px``: ``raw_px * 3000 / actual_width`` -- i.e. the error
    rescaled proportionally as if the scene had been 3000px wide, which is
    the natural reading of "at 3000px image width" as a normalization
    convention for comparing scenes of different resolutions on equal
    footing. This assumes alignment error scales roughly linearly with
    image scale, which holds for similarity/affine misalignments of the
    kind scripts/03_simulate_misalignment.py generates.
The acceptance verdict in §7 is evaluated against the ``normalized_3000px``
numbers, since that is the unit the acceptance bar itself is stated in.

WALL-CLOCK MEASUREMENT
------------------------
The design doc's bar is "neural seed pass for a 40-frame stack at 512px <=
2s on the reference CPU machine, single batch inference, measure don't
assume". This script measures wall-clock time for ONE representative
held-out scene's neural-alone shell-out call (chosen as the scene with the
most frames, to be a reasonably representative/stressful sample) and
reports:
  - the raw measured seconds for that scene's actual (frame_count, width)
  - a naively-normalized estimate scaled to "40 frames @ 512px", using a
    linear-in-(frame_count * pixel_count) proxy:

        est_40f_512px = measured_seconds
                         * (40 / actual_frame_count)
                         * (512*512) / (actual_width*actual_height)

    This is a rough proxy, not a physical model -- real inference cost may
    scale sub-linearly (batching overhead amortized) or super-linearly
    (memory pressure) in either dimension. It is reported ALONGSIDE the raw
    number, clearly labeled as an estimate, precisely so nobody mistakes it
    for a real measurement at that shape. If a scene actually matching
    (40 frames, 512px) is found among the held-out set, its raw measurement
    is used directly instead of the scaled estimate, and this is noted in
    the report.

STATISTICAL COMPARISON (neural+refine vs classical-alone)
-------------------------------------------------------------
The design doc requires "neural + refine: statistically indistinguishable
from classical-alone on synthetic (paired comparison)". This script performs
a paired comparison of per-scene ``normalized_3000px`` corner errors between
the "neural_refine" and "classical" columns:
  - If ``scipy`` is importable, a paired Wilcoxon signed-rank test is used
    (non-parametric, robust to the non-normal error distributions typical of
    alignment metrics), falling back to a paired t-test if the Wilcoxon test
    cannot be computed (e.g. all differences are zero).
  - If ``scipy`` is NOT importable, this script falls back to a simple sign-
    based summary (documented, not a real p-value): it reports the mean and
    median paired difference and the fraction of scenes where neural+refine
    beat classical, and explicitly labels this fallback as such rather than
    fabricating a p-value.
  - If either column has no data (e.g. classical is the documented TODO
    stub, or the neural binary is unavailable), the comparison is skipped
    and the report says so explicitly, rather than crashing.

DEPENDENCIES
--------------
Only ``numpy`` + Python stdlib are required. ``scipy.stats`` is optional and
guarded by a try/except; see STATISTICAL COMPARISON above for the documented
fallback when it's absent.

USAGE
-------
    python scripts/04_eval_alignment.py \
        --scenes-dir data/held_out_scenes \
        --model models/batchalign_v2.onnx \
        --eval-binary target/release/stacker-nn-eval \
        --output reports/alignment_eval.json \
        --verbose
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import scipy.stats as _scipy_stats

    _HAVE_SCIPY = True
except ImportError:
    _scipy_stats = None
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Acceptance-criteria constants (design doc §7)
# ---------------------------------------------------------------------------

NORMALIZED_WIDTH_PX = 3000  # "at 3000 px image width" reference resolution
NEURAL_MEAN_ERROR_BAR_PX = 8.0  # neural alone: mean corner error < 8px
NEURAL_MAX_SCENE_ERROR_BAR_PX = 25.0  # neural alone: no scene above 25px
WALLCLOCK_REF_FRAMES = 40
WALLCLOCK_REF_WIDTH_PX = 512
WALLCLOCK_REF_HEIGHT_PX = 512
WALLCLOCK_BAR_SECONDS = 2.0
STAT_INDISTINGUISHABLE_ALPHA = 0.05  # p >= alpha => "statistically indistinguishable"


# ---------------------------------------------------------------------------
# Scene discovery and metadata loading
# ---------------------------------------------------------------------------


class SceneLoadError(Exception):
    """Raised when a scene directory is malformed or missing required data."""


def discover_scenes(scenes_dir: Path) -> list[Path]:
    """Find held-out scene directories under ``scenes_dir``.

    Uses the same discovery convention as the Rust side: any immediate
    subdirectory of ``scenes_dir`` that contains a ``metadata.json`` file is
    treated as a scene. Non-directories and directories without
    metadata.json are silently skipped (they are not scenes, e.g. stray
    files or a README).

    Returns scene directories sorted by name for reproducible ordering.
    """
    if not scenes_dir.is_dir():
        raise SceneLoadError(
            f"--scenes-dir {scenes_dir!s} does not exist or is not a directory. "
            "Expected a directory containing one subdirectory per held-out scene, "
            "each with a metadata.json (as produced by scripts/03_simulate_misalignment.py)."
        )

    scenes = sorted(
        child
        for child in scenes_dir.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file()
    )

    if not scenes:
        raise SceneLoadError(
            f"No scene subdirectories with a metadata.json were found under {scenes_dir!s}. "
            "Nothing to evaluate. Check that --scenes-dir points at the parent of your "
            "held-out scene directories, not at a single scene directory itself."
        )

    return scenes


def load_scene_metadata(scene_dir: Path) -> dict[str, Any]:
    """Load and lightly validate a scene's metadata.json.

    Checks presence of the fields this script depends on
    (`stack`, `alignment_gt`, `cropped_dims`) and that their shapes are
    mutually consistent. Raises SceneLoadError with an actionable message on
    any problem, rather than letting a KeyError/IndexError propagate.
    """
    meta_path = scene_dir / "metadata.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise SceneLoadError(f"{meta_path!s} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise SceneLoadError(f"Could not read {meta_path!s}: {exc}") from exc

    for required_key in ("stack", "alignment_gt", "cropped_dims"):
        if required_key not in meta:
            raise SceneLoadError(
                f"{meta_path!s} is missing required key '{required_key}'. "
                "This scene was likely produced by an incompatible version of "
                "scripts/03_simulate_misalignment.py."
            )

    stack = meta["stack"]
    alignment_gt = meta["alignment_gt"]
    cropped_dims = meta["cropped_dims"]

    if len(stack) != len(alignment_gt):
        raise SceneLoadError(
            f"{meta_path!s}: len(stack)={len(stack)} != len(alignment_gt)={len(alignment_gt)}. "
            "Metadata is internally inconsistent; refusing to guess a correspondence."
        )

    if len(stack) < 2:
        raise SceneLoadError(
            f"{meta_path!s}: stack has fewer than 2 frames ({len(stack)}); "
            "there must be at least a reference frame and one non-reference frame "
            "to compute a corner error."
        )

    if not (isinstance(cropped_dims, (list, tuple)) and len(cropped_dims) == 2):
        raise SceneLoadError(
            f"{meta_path!s}: cropped_dims must be a 2-element [w, h] list, got {cropped_dims!r}."
        )

    for idx, mat in enumerate(alignment_gt):
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (3, 3):
            raise SceneLoadError(
                f"{meta_path!s}: alignment_gt[{idx}] has shape {arr.shape}, expected (3, 3)."
            )

    return meta


# ---------------------------------------------------------------------------
# Corner-error geometry
# ---------------------------------------------------------------------------


def cropped_corners(width: int, height: int) -> np.ndarray:
    """Return the four corners of a ``width`` x ``height`` image as homogeneous
    pixel coordinates, shape (4, 3).

    Corner order: top-left, top-right, bottom-left, bottom-right, using the
    convention `[(0,0), (w-1,0), (0,h-1), (w-1,h-1)]` specified in the design
    doc (i.e. corners are the outermost valid pixel centers, not the
    (w, h) outer edge).
    """
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [width - 1.0, 0.0, 1.0],
            [0.0, height - 1.0, 1.0],
            [width - 1.0, height - 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def apply_matrix(matrix: np.ndarray, points_homogeneous: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to an (N, 3) array of homogeneous pixel coordinates,
    returning an (N, 2) array of resulting (x, y) pixel coordinates.

    Follows the ground-truth convention exactly: `p_k = M_k @ p_ref`, i.e.
    matrix-vector multiplication with the point as a column vector. Points
    are de-homogenized by dividing through by the resulting w component; a
    ValueError is raised if any w component is (numerically) zero, since
    that indicates a degenerate matrix rather than a valid alignment.
    """
    transformed = points_homogeneous @ matrix.T  # (N,3) @ (3,3)^T -> (N,3)
    w = transformed[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError(
            "Degenerate transform encountered: homogeneous w-component is (near) zero. "
            "This indicates an invalid alignment matrix, not a valid perspective transform."
        )
    return transformed[:, :2] / w[:, np.newaxis]


def scene_corner_error(
    predicted_matrices: list[Optional[np.ndarray]],
    ground_truth_matrices: list[np.ndarray],
    cropped_dims: tuple[int, int],
    ref_index: int = 0,
) -> Optional[float]:
    """Compute one scene's mean corner error (raw pixels, at cropped_dims
    resolution) for a single method.

    ``predicted_matrices`` and ``ground_truth_matrices`` must be parallel
    lists indexed the same way as metadata's `stack`/`alignment_gt`. Entries
    in `predicted_matrices` may be None if that frame's prediction is
    missing/failed; such frames are skipped (and this is the caller's
    responsibility to report, not silently hide -- see
    ``compute_method_scene_errors``).

    Returns None if there are no usable (non-reference, non-None-prediction)
    frames to average over, so the caller can distinguish "zero error" from
    "no data".
    """
    width, height = cropped_dims
    corners = cropped_corners(width, height)

    per_frame_errors: list[float] = []
    for k, (pred, gt) in enumerate(zip(predicted_matrices, ground_truth_matrices)):
        if k == ref_index:
            continue  # frame 0 (reference) is identity by definition, not scored
        if pred is None:
            continue  # missing prediction for this frame; excluded, not zero-filled

        pred_pts = apply_matrix(pred, corners)
        gt_pts = apply_matrix(gt, corners)
        per_corner_dist = np.linalg.norm(pred_pts - gt_pts, axis=1)  # (4,)
        per_frame_errors.append(float(np.mean(per_corner_dist)))

    if not per_frame_errors:
        return None
    return float(np.mean(per_frame_errors))


# ---------------------------------------------------------------------------
# Shelling out to the neural eval-inference binary
# ---------------------------------------------------------------------------


class EvalBinaryUnavailable(Exception):
    """Raised when the eval-inference binary is missing, fails, or returns
    malformed output. Callers catch this per-column so one broken method
    doesn't take down the whole comparison."""


def run_eval_binary(
    eval_binary: str,
    model_path: str,
    scene_dir: Path,
    meta: dict[str, Any],
    refine: bool,
    verbose: bool = False,
) -> tuple[list[Optional[np.ndarray]], float]:
    """Invoke the (assumed) dedicated Rust eval-inference binary for one scene.

    JSON I/O CONTRACT (invented here; this is the authoritative spec the
    binary must implement -- there is no existing binary as of this writing,
    see the module docstring's TODO):

    The script writes a JSON *request* to the binary's stdin:

        {
          "model_path": "<value of --model>",
          "ref_index": 0,
          "output_dims": [width, height],       // == metadata's cropped_dims
          "refine": true | false,                 // apply classical refinement pass?
          "frames": [
            "<absolute path to focus_00.png>",
            "<absolute path to focus_01.png>",
            ...
          ]
        }

    `frames` is given in the same order as metadata's `stack` field, and
    `frames[ref_index]` is the reference frame. The binary is expected to
    predict, for every frame, the 3x3 matrix mapping reference-frame pixel
    coordinates to that frame's pixel coordinates (same convention as
    `alignment_gt`, see module docstring). When `refine` is true, the binary
    should run its classical refinement pass on top of the network's raw
    output before returning; when false, it should return the network's raw
    output unmodified.

    The binary is expected to write a JSON *response* to stdout:

        {
          "status": "ok",
          "matrices": [
            [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]],  // frame 0 (identity)
            ...                                                     // one per input frame
          ]
        }

    or, on failure to align a specific frame (e.g. the network diverged):

        {
          "status": "ok",
          "matrices": [ ... , null, ... ]   // null for frames it could not align
        }

    or, on a fatal/global failure:

        {
          "status": "error",
          "message": "<human-readable reason>"
        }

    Any non-zero exit code, timeout, malformed JSON, or `"status": "error"`
    response raises ``EvalBinaryUnavailable`` with an actionable message;
    it is the caller's job to catch this per-column and report
    "unavailable" rather than crash the whole script.

    Returns (matrices, wall_clock_seconds) where `matrices` is a list
    parallel to `meta["stack"]` (entries may be None for frames the binary
    could not align) and `wall_clock_seconds` is the measured wall-clock
    time of the subprocess call (used for the §7 wall-clock acceptance
    check on the non-refine, "neural alone" call).
    """
    frame_paths = [str((scene_dir / name).resolve()) for name in meta["stack"]]
    request = {
        "model_path": model_path,
        "ref_index": 0,
        "output_dims": list(meta["cropped_dims"]),
        "refine": refine,
        "frames": frame_paths,
    }
    request_json = json.dumps(request)

    if verbose:
        mode = "neural+refine" if refine else "neural alone"
        print(
            f"    [eval-binary] invoking '{eval_binary}' ({mode}) for scene "
            f"{scene_dir.name} ({len(frame_paths)} frames)...",
            file=sys.stderr,
        )

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [eval_binary],
            input=request_json,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except FileNotFoundError as exc:
        raise EvalBinaryUnavailable(
            f"Eval-inference binary '{eval_binary}' was not found on disk / in PATH. "
            "This binary is expected to be a dedicated Rust CLI (suggested name "
            "'stacker-nn-eval') that does NOT yet exist in this repository as of "
            "this script's authoring -- see the TODO in this script's module "
            "docstring for the JSON I/O contract it must implement. Build it, or "
            "pass --eval-binary pointing at wherever you've built it."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise EvalBinaryUnavailable(
            f"Eval-inference binary '{eval_binary}' did not finish within "
            f"{exc.timeout}s for scene {scene_dir.name}. Treating this column as "
            "unavailable for this scene rather than hanging indefinitely."
        ) from exc
    wall_clock = time.perf_counter() - start

    if completed.returncode != 0:
        raise EvalBinaryUnavailable(
            f"Eval-inference binary '{eval_binary}' exited with code "
            f"{completed.returncode} for scene {scene_dir.name}.\n"
            f"stderr:\n{completed.stderr.strip()[:2000]}"
        )

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EvalBinaryUnavailable(
            f"Eval-inference binary '{eval_binary}' produced non-JSON stdout for "
            f"scene {scene_dir.name}: {exc}. First 500 chars of stdout:\n"
            f"{completed.stdout[:500]!r}"
        ) from exc

    if response.get("status") != "ok":
        message = response.get("message", "<no message provided>")
        raise EvalBinaryUnavailable(
            f"Eval-inference binary '{eval_binary}' reported an error for scene "
            f"{scene_dir.name}: {message}"
        )

    raw_matrices = response.get("matrices")
    if not isinstance(raw_matrices, list) or len(raw_matrices) != len(frame_paths):
        raise EvalBinaryUnavailable(
            f"Eval-inference binary '{eval_binary}' returned "
            f"{len(raw_matrices) if isinstance(raw_matrices, list) else 'non-list'} "
            f"matrices for scene {scene_dir.name}, expected {len(frame_paths)} "
            "(one per input frame, matching metadata's 'stack' order)."
        )

    matrices: list[Optional[np.ndarray]] = []
    for idx, mat in enumerate(raw_matrices):
        if mat is None:
            matrices.append(None)
            continue
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (3, 3):
            raise EvalBinaryUnavailable(
                f"Eval-inference binary '{eval_binary}' returned a matrix of shape "
                f"{arr.shape} for frame index {idx} in scene {scene_dir.name}, expected (3, 3)."
            )
        matrices.append(arr)

    return matrices, wall_clock


# ---------------------------------------------------------------------------
# Classical baseline (TODO stub -- see module docstring for rationale)
# ---------------------------------------------------------------------------


def run_classical_baseline(
    scene_dir: Path, meta: dict[str, Any], verbose: bool = False
) -> Optional[list[Optional[np.ndarray]]]:
    """Compute (or fail to compute) the classical AKAZE-seeded baseline
    alignment for one scene.

    NOT IMPLEMENTED. This is a deliberate, documented choice (see the
    "CLASSICAL BASELINE" section of this script's module docstring for full
    rationale): ``z-stackr-cli`` has no existing flag that dumps solved
    per-frame alignment matrices as machine-readable JSON, only a final
    fused image. Inventing an ad hoc flag/parsing convention against a CLI
    surface that doesn't support this would be guesswork dressed up as an
    implementation, so this function always returns None (meaning: "no data
    for this scene"), and the rest of the script treats the classical
    column as globally unavailable, reporting "N/A -- classical baseline
    not implemented" wherever it would otherwise show a number.

    TODO(z-stackr-align / z-stackr-cli): add a `--dump-alignment-json <path>`
    (or similar) flag to z-stackr-cli that writes out the solved per-frame
    3x3 matrices (in the same [x_k,y_k,1]^T = M_k @ [x_ref,y_ref,1]^T
    convention documented in this script's module docstring) alongside the
    normal fused-image output. Once that exists, this function should shell
    out to it the same way ``run_eval_binary`` shells out to the neural
    binary, and parse the dumped JSON directly -- no format invention
    needed at that point.
    """
    if verbose:
        print(
            f"    [classical] skipping scene {scene_dir.name}: classical baseline "
            "is not implemented in this script (see run_classical_baseline docstring).",
            file=sys.stderr,
        )
    return None


# ---------------------------------------------------------------------------
# Per-scene, per-method orchestration
# ---------------------------------------------------------------------------


def gt_matrices_from_meta(meta: dict[str, Any]) -> list[np.ndarray]:
    """Extract ground-truth matrices from metadata as a list of (3,3) arrays."""
    return [np.asarray(m, dtype=np.float64) for m in meta["alignment_gt"]]


def normalize_to_3000px(raw_px_error: float, actual_width: int) -> float:
    """Rescale a raw corner-error (in pixels, measured at `actual_width`) to
    the equivalent error at a 3000px-wide reference image, per the
    "NORMALIZATION TO 3000px WIDTH" section of the module docstring."""
    return raw_px_error * (NORMALIZED_WIDTH_PX / float(actual_width))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def paired_comparison(
    values_a: list[float], values_b: list[float], label_a: str, label_b: str
) -> dict[str, Any]:
    """Run a paired statistical comparison between two equal-length lists of
    per-scene errors (`values_a`, `values_b`), returning a dict describing
    the result. See "STATISTICAL COMPARISON" in the module docstring for the
    scipy / no-scipy fallback behavior.
    """
    if len(values_a) != len(values_b):
        return {
            "available": False,
            "reason": (
                f"Paired comparison requires equal-length samples; got "
                f"{len(values_a)} for {label_a} and {len(values_b)} for {label_b}."
            ),
        }
    if len(values_a) < 2:
        return {
            "available": False,
            "reason": f"Need at least 2 paired scenes to compare; got {len(values_a)}.",
        }

    diffs = [a - b for a, b in zip(values_a, values_b)]

    if _HAVE_SCIPY:
        try:
            if any(d != 0 for d in diffs):
                stat, p_value = _scipy_stats.wilcoxon(values_a, values_b)
                test_name = "Wilcoxon signed-rank (paired)"
            else:
                # All differences are exactly zero; Wilcoxon is undefined here.
                stat, p_value = float("nan"), 1.0
                test_name = "degenerate (all paired differences are zero)"
        except ValueError:
            # scipy raises ValueError e.g. when the sample is too small/degenerate
            # for Wilcoxon; fall back to a paired t-test instead of crashing.
            stat, p_value = _scipy_stats.ttest_rel(values_a, values_b)
            test_name = "paired t-test (Wilcoxon fallback)"

        indistinguishable = bool(p_value >= STAT_INDISTINGUISHABLE_ALPHA)
        return {
            "available": True,
            "method": test_name,
            "statistic": float(stat),
            "p_value": float(p_value),
            "alpha": STAT_INDISTINGUISHABLE_ALPHA,
            "statistically_indistinguishable": indistinguishable,
            "interpretation": (
                f"{label_b} vs {label_a}: p={p_value:.4g} "
                f"({'>=' if indistinguishable else '<'} alpha={STAT_INDISTINGUISHABLE_ALPHA}) "
                f"=> {'statistically indistinguishable' if indistinguishable else 'statistically different'} "
                f"at the {STAT_INDISTINGUISHABLE_ALPHA} significance level, per {test_name}."
            ),
        }

    # --- scipy not installed: documented non-statistical fallback ---
    mean_diff = statistics.mean(diffs)
    median_diff = statistics.median(diffs)
    frac_a_better = sum(1 for d in diffs if d < 0) / len(diffs)
    return {
        "available": True,
        "method": "sign-based summary (scipy not installed -- no p-value computed)",
        "mean_paired_difference": mean_diff,
        "median_paired_difference": median_diff,
        "fraction_scenes_a_better": frac_a_better,
        "statistic": None,
        "p_value": None,
        "statistically_indistinguishable": None,
        "interpretation": (
            "scipy is not installed, so no formal p-value is available. As a rough "
            f"substitute: mean paired difference ({label_a} - {label_b}) = {mean_diff:.3f}px, "
            f"median = {median_diff:.3f}px, and {label_a} had lower error than {label_b} in "
            f"{frac_a_better:.0%} of scenes. Install scipy for a real paired significance test "
            "(Wilcoxon signed-rank) before relying on this for the §7 acceptance decision."
        ),
    }


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


def summarize_column(per_scene_raw: dict[str, Optional[float]], actual_widths: dict[str, int]) -> dict[str, Any]:
    """Summarize one method's per-scene raw corner errors into mean/median/
    worst, in both raw and 3000px-normalized units.

    ``per_scene_raw`` maps scene name -> raw px error (or None if
    unavailable for that scene). ``actual_widths`` maps scene name ->
    cropped_dims width, used for normalization.

    Scenes with None are excluded from the summary but counted separately
    as `scenes_missing`, so a partially-failed column is still reported
    honestly rather than silently averaging over fewer scenes.
    """
    usable = {name: err for name, err in per_scene_raw.items() if err is not None}
    missing = [name for name, err in per_scene_raw.items() if err is None]

    if not usable:
        return {
            "available": False,
            "scenes_evaluated": 0,
            "scenes_missing": missing,
        }

    raw_values = list(usable.values())
    normalized_values = [normalize_to_3000px(usable[name], actual_widths[name]) for name in usable]

    worst_scene_name = max(usable, key=lambda name: normalize_to_3000px(usable[name], actual_widths[name]))

    return {
        "available": True,
        "scenes_evaluated": len(usable),
        "scenes_missing": missing,
        "raw_px": {
            "mean": statistics.mean(raw_values),
            "median": statistics.median(raw_values),
            "worst": max(raw_values),
            "worst_scene": worst_scene_name,
        },
        "normalized_3000px": {
            "mean": statistics.mean(normalized_values),
            "median": statistics.median(normalized_values),
            "worst": max(normalized_values),
            "worst_scene": worst_scene_name,
        },
        "per_scene_raw_px": dict(usable),
        "per_scene_normalized_3000px": {
            name: normalize_to_3000px(usable[name], actual_widths[name]) for name in usable
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------


def evaluate(
    scenes_dir: Path,
    model_path: str,
    eval_binary: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full evaluation over all discovered scenes and return a
    structured results dict ready for reporting / JSON serialization."""
    scene_dirs = discover_scenes(scenes_dir)
    if verbose:
        print(f"Discovered {len(scene_dirs)} held-out scene(s) under {scenes_dir!s}", file=sys.stderr)

    per_scene_neural: dict[str, Optional[float]] = {}
    per_scene_refine: dict[str, Optional[float]] = {}
    per_scene_classical: dict[str, Optional[float]] = {}
    actual_widths: dict[str, int] = {}

    neural_errors: list[str] = []
    refine_errors: list[str] = []

    wallclock_record: Optional[dict[str, Any]] = None
    # Prefer a scene that actually matches the reference shape (40 frames);
    # otherwise pick the scene with the most frames as the most representative
    # / stressful sample, per the module docstring's wall-clock methodology.
    wallclock_scene_dir = max(
        scene_dirs, key=lambda d: len(json.loads((d / "metadata.json").read_text())["stack"])
    )

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        try:
            meta = load_scene_metadata(scene_dir)
        except SceneLoadError as exc:
            print(f"WARNING: skipping scene '{scene_name}': {exc}", file=sys.stderr)
            continue

        cropped_dims = tuple(meta["cropped_dims"])
        actual_widths[scene_name] = cropped_dims[0]
        gt_matrices = gt_matrices_from_meta(meta)

        # --- (a) neural alone ---
        try:
            neural_matrices, wall_clock = run_eval_binary(
                eval_binary, model_path, scene_dir, meta, refine=False, verbose=verbose
            )
            per_scene_neural[scene_name] = scene_corner_error(neural_matrices, gt_matrices, cropped_dims)
            if scene_dir == wallclock_scene_dir:
                wallclock_record = {
                    "scene": scene_name,
                    "frame_count": len(meta["stack"]),
                    "width": cropped_dims[0],
                    "height": cropped_dims[1],
                    "measured_seconds": wall_clock,
                }
        except EvalBinaryUnavailable as exc:
            per_scene_neural[scene_name] = None
            neural_errors.append(f"{scene_name}: {exc}")

        # --- (b) neural + classical refine ---
        try:
            refine_matrices, _ = run_eval_binary(
                eval_binary, model_path, scene_dir, meta, refine=True, verbose=verbose
            )
            per_scene_refine[scene_name] = scene_corner_error(refine_matrices, gt_matrices, cropped_dims)
        except EvalBinaryUnavailable as exc:
            per_scene_refine[scene_name] = None
            refine_errors.append(f"{scene_name}: {exc}")

        # --- (c) classical alone (AKAZE seed) -- documented TODO stub ---
        # See run_classical_baseline() docstring and the module docstring's
        # "CLASSICAL BASELINE" section for why this is intentionally not
        # implemented: z-stackr-cli has no matrix-dumping mode to shell out
        # to today, and inventing one would be undocumented guesswork.
        classical_matrices = run_classical_baseline(scene_dir, meta, verbose=verbose)
        if classical_matrices is not None:
            per_scene_classical[scene_name] = scene_corner_error(classical_matrices, gt_matrices, cropped_dims)
        else:
            per_scene_classical[scene_name] = None

    neural_summary = summarize_column(per_scene_neural, actual_widths)
    refine_summary = summarize_column(per_scene_refine, actual_widths)
    classical_summary = summarize_column(per_scene_classical, actual_widths)
    # classical is a documented global stub, not a per-scene failure; label distinctly.
    classical_summary["not_implemented"] = True
    classical_summary["not_implemented_reason"] = (
        "z-stackr-cli has no flag today to dump solved per-frame alignment matrices "
        "as machine-readable output; see run_classical_baseline() docstring for the "
        "documented TODO."
    )

    # --- statistical comparison: neural+refine vs classical ---
    common_scenes = [
        name
        for name in per_scene_refine
        if per_scene_refine[name] is not None and per_scene_classical.get(name) is not None
    ]
    if not common_scenes:
        stat_comparison = {
            "available": False,
            "reason": (
                "Cannot compare neural+refine against classical-alone: classical baseline "
                "is not implemented (see run_classical_baseline docstring), so there is no "
                "classical data to pair against."
                if classical_summary["scenes_evaluated"] == 0
                else "No scenes have valid results for both neural+refine and classical."
            ),
        }
    else:
        refine_vals = [normalize_to_3000px(per_scene_refine[n], actual_widths[n]) for n in common_scenes]
        classical_vals = [normalize_to_3000px(per_scene_classical[n], actual_widths[n]) for n in common_scenes]
        stat_comparison = paired_comparison(refine_vals, classical_vals, "neural_refine", "classical")
        stat_comparison["scenes_compared"] = len(common_scenes)

    # --- wall-clock ---
    wallclock_report: dict[str, Any]
    if wallclock_record is None:
        wallclock_report = {
            "available": False,
            "reason": "Neural-alone inference did not succeed on any scene; cannot measure wall-clock.",
        }
    else:
        matches_reference_shape = (
            wallclock_record["frame_count"] == WALLCLOCK_REF_FRAMES
            and wallclock_record["width"] == WALLCLOCK_REF_WIDTH_PX
            and wallclock_record["height"] == WALLCLOCK_REF_HEIGHT_PX
        )
        scale_factor = (WALLCLOCK_REF_FRAMES / wallclock_record["frame_count"]) * (
            (WALLCLOCK_REF_WIDTH_PX * WALLCLOCK_REF_HEIGHT_PX)
            / (wallclock_record["width"] * wallclock_record["height"])
        )
        estimated_seconds = wallclock_record["measured_seconds"] * scale_factor
        wallclock_report = {
            "available": True,
            "scene": wallclock_record["scene"],
            "measured_frame_count": wallclock_record["frame_count"],
            "measured_width": wallclock_record["width"],
            "measured_height": wallclock_record["height"],
            "measured_seconds": wallclock_record["measured_seconds"],
            "matches_reference_shape_exactly": matches_reference_shape,
            "estimated_seconds_at_40f_512px": (
                wallclock_record["measured_seconds"] if matches_reference_shape else estimated_seconds
            ),
            "estimate_is_raw_measurement": matches_reference_shape,
            "scaling_caveat": (
                "Measured directly at the 40-frame/512px reference shape; no scaling applied."
                if matches_reference_shape
                else (
                    "Scaled from the measured shape using a linear-in-(frame_count * pixel_count) "
                    "proxy; this is a rough estimate, not a real measurement at 40f/512px. See "
                    "the module docstring's WALL-CLOCK MEASUREMENT section."
                )
            ),
        }

    return {
        "scenes_dir": str(scenes_dir),
        "model_path": model_path,
        "eval_binary": eval_binary,
        "scene_count": len(scene_dirs),
        "columns": {
            "neural": neural_summary,
            "neural_refine": refine_summary,
            "classical": classical_summary,
        },
        "column_errors": {
            "neural": neural_errors,
            "neural_refine": refine_errors,
        },
        "statistical_comparison_refine_vs_classical": stat_comparison,
        "wallclock": wallclock_report,
        "acceptance": build_verdict(neural_summary, stat_comparison, wallclock_report, classical_summary),
    }


def build_verdict(
    neural_summary: dict[str, Any],
    stat_comparison: dict[str, Any],
    wallclock_report: dict[str, Any],
    classical_summary: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the three design-doc §7 acceptance bullets and produce a
    PASS/FAIL/N/A verdict for each, plus an overall verdict."""
    verdicts: dict[str, Any] = {}

    # Bullet 1: neural alone mean corner error < 8px @ 3000px width, no scene > 25px
    if not neural_summary.get("available"):
        verdicts["neural_alone"] = {
            "verdict": "FAIL",
            "detail": "Neural-alone column produced no usable data (eval binary unavailable/errored on every scene).",
        }
    else:
        mean_ok = neural_summary["normalized_3000px"]["mean"] < NEURAL_MEAN_ERROR_BAR_PX
        worst_ok = neural_summary["normalized_3000px"]["worst"] <= NEURAL_MAX_SCENE_ERROR_BAR_PX
        verdicts["neural_alone"] = {
            "verdict": "PASS" if (mean_ok and worst_ok) else "FAIL",
            "mean_normalized_3000px": neural_summary["normalized_3000px"]["mean"],
            "mean_bar_px": NEURAL_MEAN_ERROR_BAR_PX,
            "mean_ok": mean_ok,
            "worst_normalized_3000px": neural_summary["normalized_3000px"]["worst"],
            "worst_bar_px": NEURAL_MAX_SCENE_ERROR_BAR_PX,
            "worst_ok": worst_ok,
            "worst_scene": neural_summary["normalized_3000px"]["worst_scene"],
        }

    # Bullet 2a: neural+refine statistically indistinguishable from classical-alone
    if classical_summary.get("not_implemented"):
        verdicts["neural_refine_vs_classical"] = {
            "verdict": "N/A",
            "detail": "N/A -- classical baseline not implemented, see classical column notes.",
        }
    elif not stat_comparison.get("available"):
        verdicts["neural_refine_vs_classical"] = {
            "verdict": "FAIL",
            "detail": stat_comparison.get("reason", "Statistical comparison unavailable."),
        }
    elif stat_comparison.get("statistically_indistinguishable") is None:
        verdicts["neural_refine_vs_classical"] = {
            "verdict": "INCONCLUSIVE",
            "detail": "scipy not installed; only a non-statistical sign-based summary is available. "
            + stat_comparison.get("interpretation", ""),
        }
    else:
        verdicts["neural_refine_vs_classical"] = {
            "verdict": "PASS" if stat_comparison["statistically_indistinguishable"] else "FAIL",
            "detail": stat_comparison.get("interpretation", ""),
        }

    # Bullet 2b: no regression on real-stack smoke set -- inherently outside
    # the scope of a synthetic-scene script; always reported N/A here.
    verdicts["real_stack_smoke_test"] = {
        "verdict": "N/A",
        "detail": (
            "N/A -- requires a small real-stack smoke set with visual check and downstream "
            "fused-image sharpness comparison; out of scope for this synthetic-scene script. "
            "Run separately per design-doc §7."
        ),
    }

    # Bullet 3: wall-clock <= 2s for 40 frames @ 512px
    if not wallclock_report.get("available"):
        verdicts["wallclock"] = {
            "verdict": "FAIL",
            "detail": wallclock_report.get("reason", "Wall-clock measurement unavailable."),
        }
    else:
        est = wallclock_report["estimated_seconds_at_40f_512px"]
        ok = est <= WALLCLOCK_BAR_SECONDS
        verdicts["wallclock"] = {
            "verdict": "PASS" if ok else "FAIL",
            "estimated_seconds_at_40f_512px": est,
            "bar_seconds": WALLCLOCK_BAR_SECONDS,
            "is_raw_measurement": wallclock_report["estimate_is_raw_measurement"],
            "caveat": wallclock_report["scaling_caveat"],
        }

    relevant_for_overall = [
        verdicts["neural_alone"]["verdict"],
        verdicts["neural_refine_vs_classical"]["verdict"],
        verdicts["wallclock"]["verdict"],
    ]
    if "FAIL" in relevant_for_overall:
        overall = "FAIL"
    elif "INCONCLUSIVE" in relevant_for_overall:
        overall = "INCONCLUSIVE"
    elif "N/A" in relevant_for_overall:
        overall = "INCOMPLETE (classical baseline not implemented; see per-bullet detail)"
    else:
        overall = "PASS"
    verdicts["overall"] = overall

    return verdicts


# ---------------------------------------------------------------------------
# Reporting (stdout + JSON)
# ---------------------------------------------------------------------------


def format_stat(value: Optional[float]) -> str:
    """Format an optional float for display, using 'N/A' when None."""
    return "N/A" if value is None else f"{value:6.2f}"


def print_report(results: dict[str, Any]) -> None:
    """Print a human-readable summary table of the evaluation results."""
    print("=" * 78)
    print("z-stackr alignment evaluation (design doc §7)")
    print("=" * 78)
    print(f"Scenes directory : {results['scenes_dir']}")
    print(f"Scenes evaluated : {results['scene_count']}")
    print(f"Model            : {results['model_path']}")
    print(f"Eval binary      : {results['eval_binary']}")
    print()

    header = (
        f"{'method':<16} {'n':>4} {'mean_raw':>10} {'median_raw':>11} {'worst_raw':>10} "
        f"{'mean_3000':>10} {'median_3000':>12} {'worst_3000':>11}"
    )
    print(header)
    print("-" * len(header))
    for key, label in (("neural", "neural"), ("neural_refine", "neural+refine"), ("classical", "classical")):
        col = results["columns"][key]
        if col.get("not_implemented"):
            print(f"{label:<16} {'--':>4}  N/A -- classical baseline not implemented (see column notes)")
            continue
        if not col.get("available"):
            print(
                f"{label:<16} {'--':>4}  UNAVAILABLE "
                f"({len(col.get('scenes_missing', []))} scene(s) failed; see --verbose / column_errors)"
            )
            continue
        n = col["scenes_evaluated"]
        r, z = col["raw_px"], col["normalized_3000px"]
        print(
            f"{label:<16} {n:>4} {format_stat(r['mean']):>10} {format_stat(r['median']):>11} "
            f"{format_stat(r['worst']):>10} {format_stat(z['mean']):>10} {format_stat(z['median']):>12} "
            f"{format_stat(z['worst']):>11}"
        )
        if col.get("scenes_missing"):
            print(
                f"{'':<16}   ({len(col['scenes_missing'])} scene(s) missing/unavailable: "
                f"{', '.join(col['scenes_missing'])})"
            )

    print()
    print("Statistical comparison: neural+refine vs classical-alone")
    print("-" * 78)
    stat = results["statistical_comparison_refine_vs_classical"]
    if not stat.get("available"):
        print(f"  N/A -- {stat.get('reason')}")
    else:
        print(f"  {stat['interpretation']}")

    print()
    print("Wall-clock (neural alone)")
    print("-" * 78)
    wc = results["wallclock"]
    if not wc.get("available"):
        print(f"  N/A -- {wc.get('reason')}")
    else:
        print(
            f"  Measured: {wc['measured_seconds']:.3f}s for scene '{wc['scene']}' "
            f"({wc['measured_frame_count']} frames @ {wc['measured_width']}x{wc['measured_height']})"
        )
        print(
            f"  Estimated at {WALLCLOCK_REF_FRAMES}f/{WALLCLOCK_REF_WIDTH_PX}px: "
            f"{wc['estimated_seconds_at_40f_512px']:.3f}s "
            f"({'raw measurement, shape matched exactly' if wc['is_raw_measurement'] else 'scaled estimate'}) "
            f"vs bar of {WALLCLOCK_BAR_SECONDS}s"
        )
        print(f"  Caveat: {wc['scaling_caveat']}")

    print()
    print("Acceptance verdict (design doc §7, 'drop experimental label from batchalign-v2')")
    print("-" * 78)
    acc = results["acceptance"]
    na = acc["neural_alone"]
    print(
        f"  1. Neural alone (<{NEURAL_MEAN_ERROR_BAR_PX}px mean, <={NEURAL_MAX_SCENE_ERROR_BAR_PX}px "
        f"worst @ 3000px): {na['verdict']}"
    )
    if "mean_normalized_3000px" in na:
        print(
            f"       mean={na['mean_normalized_3000px']:.2f}px worst={na['worst_normalized_3000px']:.2f}px "
            f"(scene: {na['worst_scene']})"
        )
    else:
        print(f"       {na.get('detail')}")

    nrc = acc["neural_refine_vs_classical"]
    print(f"  2a. Neural+refine vs classical (paired, statistically indistinguishable): {nrc['verdict']}")
    print(f"       {nrc.get('detail')}")

    smoke = acc["real_stack_smoke_test"]
    print(f"  2b. Real-stack smoke test (visual + sharpness, no regression): {smoke['verdict']}")
    print(f"       {smoke.get('detail')}")

    wcv = acc["wallclock"]
    print(f"  3. Wall-clock (<= {WALLCLOCK_BAR_SECONDS}s for {WALLCLOCK_REF_FRAMES}f @ {WALLCLOCK_REF_WIDTH_PX}px): {wcv['verdict']}")
    if "estimated_seconds_at_40f_512px" in wcv:
        print(f"       estimate={wcv['estimated_seconds_at_40f_512px']:.3f}s ({wcv['caveat']})")
    else:
        print(f"       {wcv.get('detail')}")

    print()
    print(f"  OVERALL: {acc['overall']}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate z-stackr's alignment network against held-out synthetic scenes "
            "per design doc §7. Requires a dedicated Rust eval-inference binary that "
            "does not yet exist -- see this script's module docstring for the JSON I/O "
            "contract it must implement."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        required=True,
        help="Directory containing held-out scene subdirectories (each with a metadata.json), "
        "as produced by scripts/03_simulate_misalignment.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path (or name) of the trained alignment model to pass through to the eval binary.",
    )
    parser.add_argument(
        "--eval-binary",
        type=str,
        default="target/release/stacker-nn-eval",
        help=(
            "Path to the dedicated Rust eval-inference binary implementing this script's JSON "
            "I/O contract (see module docstring). NOTE: this binary does not exist in the "
            "repository as of this script's authoring and must be built separately; the default "
            "path assumes a cargo release build at that conventional location."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full structured results as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and per-scene diagnostic messages to stderr while running.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Script entry point. Returns a process exit code (0 = ran successfully
    and overall verdict is PASS; 1 = ran successfully but verdict is not
    PASS; 2 = could not run at all, e.g. bad --scenes-dir)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        results = evaluate(
            scenes_dir=args.scenes_dir,
            model_path=args.model,
            eval_binary=args.eval_binary,
            verbose=args.verbose,
        )
    except SceneLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_report(results)

    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, sort_keys=False)
            if args.verbose:
                print(f"\nWrote full results JSON to {args.output!s}", file=sys.stderr)
        except OSError as exc:
            print(f"WARNING: could not write --output {args.output!s}: {exc}", file=sys.stderr)

    return 0 if results["acceptance"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
