# z-stackr-gui

A graphical user interface for **z-stackr** (package `z-stackr-gui`, binary `z-stackr-gui`), built with the [Slint](https://slint-ui.com/) UI toolkit.

`z-stackr-gui` provides an interactive, visual alternative to the `z-stackr` command-line tool. It allows photographers to select image sequences, configure processing parameters, preview alignments, and execute focus fusion — all without needing a terminal.

## Features
- **Visual Image Selection**: Interactively load your sequence of macro images.
- **Parameter Configuration**: Easily adjust out-of-core tile size, output paths, and switch between `Apex`, `Relief`, `Both`, and (in `nn` builds) `AI Model (Neural)` fusion.
- **Align & Preview**: Run an alignment-only pass via the **Align** button, then flip the **Show Aligned** switch to preview each frame's warped, common-area-cropped result before committing to a full stack.
- **Sort & Cull (standalone, independent of the Settings toggles)**: Next to **Align**, the **Sort** and **Cull** buttons run the sharpness-ordering and auto-cull passes on their own — **Sort** reorders the source-file list by sharpness, **Cull** dims/marks culled frames in place using the current Cull-threshold slider value — against the same aligned/cropped/subsampled frame set the Stack pipeline scores internally. Each button is a fully independent command: it never reads the **Sort by sharpness** / **Auto-Cull Frames** Settings toggles, so pressing **Sort** always sorts and pressing **Cull** always culls no matter how those toggles are set. Re-pressing one after the other (with nothing else changed) preserves the other's result — e.g. Cull after Sort keeps the sharpness order and culls from it. Pressing **Stack** afterwards reuses whichever ops the cached result covers instead of recomputing them, as long as the inputs/settings haven't changed since; if they have, Stack pauses with a popup asking whether to re-run Sort/Cull or keep the current list as-is.
- **Live Preview & Progress**: While alignment runs (Align button *or* the Stack pipeline's alignment stage), the canvas shows each frame as it is registered. Apex fusion refreshes the running composite on throttled checkpoints; Relief fusion reports per-frame focus-measure progress and previews the raw depth-index map before the solve; AI stacking shows the running composite getting sharper as frames are folded in. A two-bar progress display shows monotonic total progress plus per-step progress.
- **Image Navigation**: The preview auto-fits newly loaded images ("Fit" zoom), and supports mouse-wheel scrolling, drag-to-pan (whenever the retouch brush is not active), and explicit Fit / 100% / ± zoom controls.
- **AI Model Mode** *(`nn` builds)*: Choose `AI Model (Neural)` to fuse with a trained model from the `models/` folder. A **model picker** lists installed fusion models and is **greyed out when none are found**; a **device selector** (CPU/GPU) appears only when the binary was built with more than one backend (`nn-gpu`).
- **Neural Alignment Mode** *(`nn` builds, experimental)*: Choose `Neural` in the Alignment Mode dropdown to register frames with a trained `batchalign-v2` model instead of the classical AKAZE/intensity pipeline — a **separate alignment-model picker** (filtered to alignment-capable checkpoints, never the fusion picker's list) appears and is greyed out when no alignment models are found. No classical refinement runs on top of the model's output; see the project README's "Neural alignment" section for the full contract.
- **Per-Frame Brightness Correction & Dual-Engine Relief**: Exposes per-frame brightness correction (on by default) and, for Relief fusion, a choice between the default Guided-Filter engine and a Multigrid depth-solver engine — see the project README's "GUI Settings Overview" for the full settings reference.
- **Retouching (always-on-top popup, brush, undo/redo, painted-area overlay)**: fix small errors in a finished stack by copying pixels in from one of your original frames — right-click a result to open the retouch popup, which stays on top of the main window while you work. See "Retouching a stack result" below for the full walkthrough.
- **EXIF Metadata Copy on Save**: When "Copy metadata to output" is enabled, saving re-injects the raw EXIF blob from the first source frame into the JPEG/PNG output (byte-level, via the shared `stacker_core::metadata` module — identical logic to the CLI). Not supported for TIFF output.
- **RAW Image Support** *(`raw` builds)*: The "Open files" dialog and the folder-monitor scan additionally accept camera RAW files (CR2/CR3/NEF/ARW/DNG/RAF/ORF/RW2/PEF/…, decoded via pure-Rust `rawler`) when built with `--features raw`; without it, RAW extensions never appear in the file picker. See the project README's "RAW Image Support" section for the full format list and pipeline limitations (no lens corrections, default demosaic only).
- **Real-time Log Output**: Integrated tracing hooks to display stacking progress directly in the application window.
- **Asynchronous Execution**: The UI remains highly responsive while the core engines perform heavy I/O and alignment workloads in the background on the `smol` executor.

## Retouching a stack result

Focus stacking blends many photos of the same subject, each sharp in a different spot, into one image that is sharp everywhere. Occasionally the blend picks the wrong spot in a small area — you'll see a halo, a doubled edge, or a smeared patch where the algorithm couldn't tell which frame was actually in focus. Usually, though, at least one of your *original* input frames does have that exact area in sharp focus. The retouch brush lets you manually copy the good pixels from that one frame into the result, right where the blend went wrong — like a local "undo the algorithm's mistake, just here" tool.

All the brush UI (size, opacity, undo/redo, the "show painted area" toggle, and the brush-cursor outline) lives in its own **always-on-top retouch popup**, separate from the main window, so you can keep painting while comparing against the main window underneath. Donor-frame selection stays in the main window.

Workflow:

1. **Right-click the result you want to fix**, in the "Results" list on the left, and choose **Retouch…**. This opens the retouch popup showing the result's current composite. The popup is a single reusable window — right-clicking again (the same result, or a different one) reuses it instead of opening a new one each time; closing it (titlebar close, Alt+F4, etc.) just hides it, so your in-progress strokes are preserved if you reopen it later.
2. **Pick a source (donor) frame.** Back in the main window's file list on the left, click the input frame you believe has the problem area in focus. This is the frame the brush will copy *from*.
3. **Turn on "Show Aligned" and check the frame.** Alignment warps and crops every frame onto a common canvas so they all line up pixel-for-pixel with the result — this only works correctly if you've already run **Align** or **Stack** first. With "Show Aligned" on, the main window's canvas shows this frame in its aligned form; zoom in on the problem area and confirm it really is sharp there. If it isn't, pick a different frame.
4. **Drag over the popup's image to paint.** The brush is always active in the popup — there's no separate "Show Aligned" gate to flip there, unlike the old in-window canvas. There is no separate "apply" step either — painting *is* the fix, applied live, pixel by pixel, as you drag. What you see while dragging is exactly what gets saved.

A few things that aren't obvious the first time:

- **Opacity accumulates, it doesn't overwrite.** Each pass of the brush pushes the painted area further toward the donor frame rather than jumping straight to it (repeated strokes converge towards fully replaced, they don't overshoot). Since the brush is a hard-edged circle with no feathered falloff, this is also how you feather a fix by hand: a few light, slightly-overlapping passes at low opacity blend the edges of the correction in more gently than one pass at high opacity, which leaves a harder-edged patch.
- **Undo/redo is the correction mechanism**, not a safety net you'll rarely touch — since there's no separate apply step, "I painted too much" or "wrong frame entirely" is meant to be fixed by undoing strokes (or the whole area) and trying again, not by manually painting a correction back. Undo/redo live in the popup, next to the brush controls.
- **"Show painted area"** (next to the brush controls in the popup) tints the area you've brushed in magenta so you can see exactly where the mask currently covers — useful for checking you got full coverage, or that you didn't paint further than intended. It's a preview overlay only: it never changes the image that gets saved.
- **The brush outline** shown at your cursor in the popup draws the exact circle that will be affected, at the current brush size, scaled to match the popup's own zoom level.
- **Changing the donor frame or "Show Aligned" in the main window takes effect immediately** for the next brush stroke, and refreshes the popup's displayed image on the spot if a retouch session is active — you don't need to close and reopen the popup.

## Usage

You can launch the GUI using Cargo from the workspace root:

```bash
# Standard build
cargo run --release -p z-stackr-gui

# Build with AKAZE feature matching enabled (recommended)
cargo run --release -p z-stackr-gui --features akaze

# Build with the AI ("neural") stacking mode — CPU inference
cargo run --release -p z-stackr-gui --features nn

# …or with GPU (wgpu/Vulkan) inference, which also enables the device selector
cargo run --release -p z-stackr-gui --features nn-gpu

# Build with pure-Rust camera RAW support (CR2/CR3/NEF/ARW/DNG/RAF/ORF/RW2/PEF/…)
cargo run --release -p z-stackr-gui --features raw
```

> Without `--features nn`/`nn-gpu` the GUI builds with no ML dependencies and the
> AI option does not appear in the algorithm list. Trained models live in a
> `models/` folder next to the executable (`<name>.mpk` + `<name>.json`).

Once launched, use the interface to point to your image sequence directory, choose a destination file, and start the processing pipeline.

## System Requirements
Because `z-stackr-gui` uses Slint, it relies on your system's native windowing environment (e.g., Wayland, X11 on Linux; DWM on Windows; Cocoa on macOS). Ensure you have standard graphics drivers and libraries installed.
