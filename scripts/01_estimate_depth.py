#!/usr/bin/env python3
"""
01_estimate_depth.py — Stage 1 of focus-stack training-data generation.

Upgraded from MiDaS to **Depth Anything 3** (DA3), the current state-of-the-art
monocular depth model (ByteDance, Nov 2025). Takes ordinary all-in-focus photos
and produces a per-image *relative* depth map normalised to [0,1] with
0 = NEAR (closest to camera) and 1 = FAR.

    python 01_estimate_depth.py --input ./photos --output ./depth

Output per <name>.<ext>  (contract UNCHANGED — stage 2 is unaffected):
    <output>/<name>_depth.npy   float32 [0,1], 0=near 1=far, at ORIGINAL resolution
    <output>/<name>_depth.png   16-bit preview

Why DA3, and how it differs from MiDaS
--------------------------------------
  * MiDaS / Depth-Anything-V2 emit *inverse* depth (disparity): large = NEAR.
    DA3's monocular model `DA3MONO-LARGE` predicts **depth directly**
    (small = near, large = far) with much better geometric accuracy and sharper
    depth edges. Sharp edges matter here because stage 2 derives the per-frame
    in-focus masks and the occlusion map from this depth — crisper edges give
    cleaner focus labels. Because it is direct depth, we do NOT invert before
    the near->far normalisation (use --invert for disparity-style models).
  * The map is robustly normalised (1/99 percentile clip) to a stable 0..1
    near->far ordering. Absolute scale is irrelevant — stage 2 re-normalises.
  * A single photo cannot reveal occluded background, so stage 2 still emits a
    depth-discontinuity ("occlusion") map; the training loss down-weights those
    pixels, where the synthetic defocus is necessarily wrong.

One-time setup
--------------
    git clone https://github.com/ByteDance-Seed/Depth-Anything-3
    cd Depth-Anything-3
    pip install xformers "torch>=2" torchvision
    pip install -e .
  Checkpoints download from Hugging Face on first run. Behind a firewall:
    export HF_ENDPOINT=https://hf-mirror.com

Model choices (--model)
-----------------------
    depth-anything/DA3MONO-LARGE     relative monocular depth   (default, best fit)
    depth-anything/DA3METRIC-LARGE   metric monocular depth     (fine; scale normalised away)
    depth-anything/DA3-LARGE         any-view foundation model  (heavier)
    depth-anything/DA3-BASE          smaller / faster, lower quality
    depth-anything/DA3-SMALL         smallest / fastest
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

try:
    import torch
    import cv2
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing dependency: {e}. Try: pip install torch opencv-python")

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:  # pragma: no cover
    sys.exit(
        "Depth Anything 3 is not installed. See the setup notes in this file's "
        "docstring:\n  git clone https://github.com/ByteDance-Seed/Depth-Anything-3\n"
        "  cd Depth-Anything-3 && pip install xformers 'torch>=2' torchvision && pip install -e ."
    )

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def to_near_far(depth: np.ndarray, clip_pct: float, invert: bool) -> np.ndarray:
    """DA3 direct depth -> robust [0,1] where 0=near, 1=far.

    DA3MONO predicts depth directly (small=near, large=far), so no inversion is
    needed. `--invert` is provided for disparity/inverse-depth models (MiDaS, DA2)
    where large=near.
    """
    d = depth.astype(np.float32)
    if invert:
        d = d.max() - d
    lo = np.percentile(d, clip_pct)            # robust min/max ignores outliers
    hi = np.percentile(d, 100.0 - clip_pct)
    if hi - lo < 1e-8:
        return np.zeros_like(d)
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3 monocular depth for focus-stack synthesis.")
    ap.add_argument("--input", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--model", default="depth-anything/DA3MONO-LARGE",
                    help="Hugging Face id or local dir of a DA3 checkpoint")
    ap.add_argument("--clip-pct", type=float, default=1.0,
                    help="robust normalisation percentile (clip this %% at each end)")
    ap.add_argument("--invert", action="store_true",
                    help="invert before normalising (for disparity / inverse-depth models)")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[depth] device={device} model={args.model}")
    if device.type == "cpu":
        print("[depth] WARNING: running on CPU — DA3 is slow without a GPU.")

    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f"no images found in {args.input}")

    for i, path in enumerate(images, 1):
        # Read only to recover the ORIGINAL resolution: the depth map must match
        # the frames stage 2 renders (which render at the source resolution).
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[depth] !! skip unreadable {path.name}")
            continue
        h0, w0 = bgr.shape[:2]

        # Monocular: pass ONE image per call so any-view checkpoints never fuse
        # multiple views into a single (wrong) prediction.
        pred = model.inference([str(path)])
        depth = np.asarray(pred.depth[0], dtype=np.float32)        # [H, W] direct depth

        # DA3 may run at a downscaled internal resolution — resize back to source.
        if depth.shape[:2] != (h0, w0):
            depth = cv2.resize(depth, (w0, h0), interpolation=cv2.INTER_CUBIC)

        near_far = to_near_far(depth, args.clip_pct, args.invert)

        stem = path.stem
        np.save(args.output / f"{stem}_depth.npy", near_far)
        cv2.imwrite(str(args.output / f"{stem}_depth.png"),
                    (near_far * 65535.0).clip(0, 65535).astype(np.uint16))
        print(f"[depth] ({i}/{len(images)}) {path.name} -> {stem}_depth.npy  [{w0}x{h0}]")

    print(f"[depth] done. {len(images)} maps in {args.output}")


if __name__ == "__main__":
    main()