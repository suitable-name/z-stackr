#!/usr/bin/env python3
"""
02_blender_focus_stack.py — Stage 2 of focus-stack training-data generation.

Run INSIDE Blender (background). Given one photo + its depth map (stage 1), it
builds a depth-displaced, photo-textured plane and emits everything the PAIRWISE
merge network needs to be trained with prefix-composite supervision:

  focus_00.png .. focus_10.png   the defocus STACK (focus swept in 10% z-steps)
  allfocus.png                   ground-truth all-in-focus (DoF disabled)
  mask_00.png .. mask_10.png     per-frame soft IN-FOCUS weight (0..1) per pixel
  occlusion.png                  depth-discontinuity map (down-weight loss here)
  metadata.json                  focus schedule, fstop, geometry, merge order

Usage (one invocation PER photo):
    blender -b -P 02_blender_focus_stack.py -- \
        --img    photos/scene.jpg \
        --depth  depth/scene_depth.npy \
        --out    dataset/scene \
        --focus-step-pct 10 --fstop 2.8 --depth-scale 2.0 --cam-dist 10 \
        --gpu                 # optional: render on the GPU (OptiX/CUDA/HIP/Metal/oneAPI)

Then point the trainer at the parent folder:
    z-stackr-train --data dataset --out models --size m --epochs 60

Blender version: tested against the Blender 5.x Python API; also runs on 4.x.
The API calls used here are stable across both. On 5.x this script additionally
uses the OpenImageDenoise denoiser and supports optional GPU rendering (--gpu).

What changed vs the original (to raise training-data quality)
------------------------------------------------------------
  * COLOR FIDELITY (important): the view transform is forced to **Standard**.
    Blender 4.x AND 5.x both default to the AgX view transform, which tone-maps
    the render; the photo is emitted unlit, so only under Standard does the
    rendered `allfocus.png` reproduce the source colours pixel-for-pixel. Under
    AgX the ground-truth target would carry a colour/contrast shift. (On 5.x,
    keep the scene working space at its Linear Rec.709 default — do NOT switch
    to ACEScg — or wide-gamut conversion will shift colours too.) This is the
    single most important fix for target correctness.
  * OCCLUSION BAND: depth-edge defocus is wrong in a *band* around the edge,
    not a single pixel line, so the occlusion mask is dilated (`--occl-dilate`)
    to actually cover the region the loss should ignore.
  * CLEANER DEFOCUS: Cycles denoising (OpenImageDenoise) is enabled and the
    default sample count is raised, so out-of-focus regions aren't noisy (noise
    would otherwise be learned as a texture).
  * 8-bit sRGB output by default to match the Rust data loader (which reads
    8-bit); use --color-depth 16 if you ever switch the loader to 16-bit.

How the masks define the training targets
------------------------------------------
Focus plane i is focused at depth fraction t_i (0=near .. 1=far). A pixel with
normalised depth d is "in focus" in plane i with weight exp(-(d - t_i)^2 / 2σ^2).
A dataloader builds the prefix composite of frames {0..k} per pixel as:
    if max_i<=k weight_i  is high -> take allfocus value (some frame has it sharp)
    else                         -> take the least-defocused included frame
giving exact (target, source) -> merged supervision for every recurrent step.

LIMITATION (unchanged): single-view depth cannot reveal occluded background, so
defocus near strong depth edges is wrong. occlusion.png marks those pixels so the
loss can ignore them; for top quality, render true 3-D scenes instead.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import bpy
except ImportError:
    sys.exit("Run inside Blender:  blender -b -P 02_blender_focus_stack.py -- ...")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--depth", required=True, help=".npy or image, 0=near 1=far")
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus-step-pct", type=float, default=10.0,
                    help="focus-plane spacing in %% of depth range (10 => 11 planes 0..100)")
    ap.add_argument("--fstop", type=float, default=2.8, help="smaller = more blur")
    ap.add_argument("--depth-scale", type=float, default=2.0, help="world-units of relief")
    ap.add_argument("--cam-dist", type=float, default=10.0)
    ap.add_argument("--subdiv", type=int, default=400)
    ap.add_argument("--samples", type=int, default=128,
                    help="Cycles samples per frame (raised from 64; denoised)")
    ap.add_argument("--edge-thresh", type=float, default=0.04,
                    help="|grad(depth)| above this is flagged as occlusion edge")
    ap.add_argument("--occl-dilate", type=int, default=3,
                    help="dilate the occlusion mask by N px to cover the defocus-error band")
    ap.add_argument("--color-depth", choices=["8", "16"], default="8",
                    help="PNG bit depth for rendered frames (8 matches the Rust loader)")
    ap.add_argument("--no-denoise", action="store_true",
                    help="disable Cycles denoising (on by default)")
    ap.add_argument("--gpu", action="store_true",
                    help="render on the GPU (best-effort: OptiX/CUDA/HIP/Metal/oneAPI)")
    return ap.parse_args(argv)


# ── depth helpers (numpy; Blender bundles numpy) ────────────────────────────────

def load_depth(depth_path: str) -> np.ndarray:
    if depth_path.endswith(".npy"):
        d = np.load(depth_path).astype(np.float32)
    else:
        import cv2
        raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        d = raw if raw.ndim == 2 else raw[..., 0]
    d = (d - d.min()) / max(d.max() - d.min(), 1e-8)
    return d  # 0=near, 1=far


def depth_to_displace_png(depth: np.ndarray, out_dir: str) -> str:
    import cv2
    png = os.path.join(out_dir, "_depth_displace.png")
    cv2.imwrite(png, (depth * 65535).astype(np.uint16))
    return png


def write_png_gray(arr01: np.ndarray, path: str) -> None:
    import cv2
    cv2.imwrite(path, (np.clip(arr01, 0, 1) * 65535).astype(np.uint16))


def emit_masks_and_meta(depth: np.ndarray, focus_t: list, args, out_dir: str) -> None:
    """Per-plane soft in-focus weights + occlusion map + metadata.json."""
    # σ ties the in-focus band to the plane spacing so adjacent planes overlap.
    step = (focus_t[1] - focus_t[0]) if len(focus_t) > 1 else 1.0
    sigma = max(step, 1e-3)
    mask_files = []
    for i, t in enumerate(focus_t):
        w = np.exp(-((depth - t) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        name = f"mask_{i:02d}.png"
        write_png_gray(w, os.path.join(out_dir, name))
        mask_files.append(name)

    # Depth-discontinuity / occlusion map from depth gradient magnitude.
    gy, gx = np.gradient(depth.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    occ = (grad > args.edge_thresh).astype(np.float32)
    # Defocus is wrong in a *band* around each edge, not a 1-px line — dilate so
    # the loss actually ignores the affected region.
    if args.occl_dilate > 0:
        import cv2
        k = np.ones((2 * args.occl_dilate + 1, 2 * args.occl_dilate + 1), np.uint8)
        occ = cv2.dilate(occ, k, iterations=1)
    write_png_gray(occ, os.path.join(out_dir, "occlusion.png"))

    meta = {
        "image": os.path.basename(args.img),
        "n_planes": len(focus_t),
        "focus_step_pct": args.focus_step_pct,
        "focus_fractions": [round(float(t), 4) for t in focus_t],   # 0..1, depth a plane focuses on
        "focus_distances": [round(args.cam_dist - args.depth_scale + float(t) * args.depth_scale, 4)
                            for t in focus_t],
        "in_focus_sigma": round(float(sigma), 4),
        "fstop": args.fstop,
        "depth_scale": args.depth_scale,
        "cam_dist": args.cam_dist,
        "occl_dilate_px": args.occl_dilate,
        "stack": [f"focus_{i:02d}.png" for i in range(len(focus_t))],
        "masks": mask_files,
        "allfocus": "allfocus.png",
        "occlusion": "occlusion.png",
        "merge_order": list(range(len(focus_t))),
        "note": "prefix composite of {0..k}: allfocus where any included mask>thr, "
                "else least-defocused included frame (argmin |depth - focus_fraction|).",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ── Blender scene ──────────────────────────────────────────────────────────────

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_plane(img_path: str, depth_png: str, subdiv: int, depth_scale: float):
    img = bpy.data.images.load(img_path)
    img.colorspace_settings.name = "sRGB"        # photo is sRGB-encoded
    w, h = img.size
    aspect = w / h
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdiv, y_subdivisions=subdiv, size=2.0)
    plane = bpy.context.active_object
    plane.scale = (aspect, 1.0, 1.0)

    depth_img = bpy.data.images.load(depth_png)
    depth_img.colorspace_settings.name = "Non-Color"
    tex = bpy.data.textures.new("depth_tex", type="IMAGE")
    tex.image = depth_img
    mod = plane.modifiers.new("displace", type="DISPLACE")
    mod.texture = tex
    mod.texture_coords = "UV"
    mod.mid_level = 1.0          # depth=1 (far) flat; depth=0 (near) raised
    mod.strength = -depth_scale  # near pixels move toward the camera

    mat = bpy.data.materials.new("photo")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    texnode = nt.nodes.new("ShaderNodeTexImage")
    texnode.image = img
    nt.links.new(texnode.outputs["Color"], emis.inputs["Color"])
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    plane.data.materials.append(mat)
    return plane, aspect


def setup_camera(aspect: float, cam_dist: float, fstop: float):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0 * max(aspect, 1.0)
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (0.0, 0.0, cam_dist)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam


def enable_gpu() -> bool:
    """Best-effort: enable the first available Cycles GPU backend.

    Tries each vendor backend in turn (NVIDIA OptiX/CUDA, AMD HIP, Apple Metal,
    Intel oneAPI). Safe to call on any machine — on failure it leaves Cycles on
    the CPU and returns False. Stable across the Blender 4.x / 5.x Python API.
    """
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue  # backend not compiled into this build
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type != "CPU"  # GPUs on, CPU off
                bpy.context.scene.cycles.device = "GPU"
                print(f"[blender] GPU rendering via {backend}: "
                      f"{', '.join(d.name for d in gpus)}")
                return True
        print("[blender] no GPU backend available — rendering on CPU")
    except Exception as e:  # never let GPU setup break the render
        print(f"[blender] GPU setup failed ({e}) — rendering on CPU")
    return False


def configure_render(img_path: str, args) -> None:
    src = bpy.data.images.load(img_path)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    # Denoise so out-of-focus areas are clean (noise would be learned as texture).
    scene.cycles.use_denoising = not args.no_denoise
    if not args.no_denoise:
        # OpenImageDenoise gives clean, deterministic results across machines
        # (independent of the GPU's hardware denoiser). Best-effort on 4.x/5.x.
        try:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except (TypeError, AttributeError):
            pass
    scene.render.resolution_x = src.size[0]
    scene.render.resolution_y = src.size[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = args.color_depth

    # CRITICAL: emit the photo faithfully. The unlit emission of an sRGB image
    # under the "Standard" view transform reproduces the source colours exactly.
    # Blender 4.x and 5.x both default to AgX, which would tone-map the target.
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"


def render_to(path: str) -> None:
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    depth = load_depth(args.depth)
    displace_png = depth_to_displace_png(depth, args.out)

    # Focus schedule: planes every focus_step_pct, 0..100 inclusive (10% => 11 planes).
    step = args.focus_step_pct / 100.0
    focus_t = list(np.round(np.arange(0.0, 1.0 + 1e-6, step), 6))
    if focus_t[-1] < 1.0:
        focus_t.append(1.0)

    reset_scene()
    _, aspect = build_plane(args.img, displace_png, args.subdiv, args.depth_scale)
    cam = setup_camera(aspect, args.cam_dist, args.fstop)
    configure_render(args.img, args)
    if args.gpu:
        enable_gpu()

    near, far = args.cam_dist - args.depth_scale, args.cam_dist
    for i, t in enumerate(focus_t):
        cam.data.dof.focus_distance = near + t * (far - near)
        render_to(os.path.join(args.out, f"focus_{i:02d}.png"))
        print(f"[blender] plane {i+1}/{len(focus_t)} focus_frac={t:.2f}")

    cam.data.dof.use_dof = False
    render_to(os.path.join(args.out, "allfocus.png"))

    # Labels for the pairwise/prefix-composite training format.
    emit_masks_and_meta(depth, focus_t, args, args.out)
    print(f"[blender] done -> {args.out}  ({len(focus_t)} planes + masks + metadata.json)")


if __name__ == "__main__":
    main()
