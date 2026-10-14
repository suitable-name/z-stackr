# z-stackr manual

z-stackr combines a series of photographs taken at different focus distances into one picture that is sharp from front to back. This is called focus stacking. This page explains the workflow, the choices you can make, and where things are kept. Press Esc or Return to close it.

## Workflow

The short version:

- Add your photographs with File > Add Images (Ctrl+O), or drop the files or their folder onto the window.
- Pick a fusion method in the Algorithm box on the left.
- Optional: press Align to align the frames first and check them with Show Aligned.
- Press Start Stacking.
- Look at the result, retouch it if needed, then File > Export (Ctrl+E).

The frames are processed in natural file-name order, so frame_2 comes before frame_10. If your rail runs from the back to the front, switch on Reverse sort order in the settings (Input tab). A frame that cannot be read is skipped and named in a message.

When a stack finishes, the result appears under Results in the left panel. The picture on the canvas is an 8-bit preview; Export writes the newest result from the full-precision data.

While a job runs, a chip in the status strip shows its name, how far it is, the time it has taken and an estimate of the time left. The cross on the chip cancels the job. Only one such job runs at a time; if you start another, the program says which job is still running and asks you to cancel it or wait.

## Viewing the image

Turn the mouse wheel to zoom about the pointer. Hold Space and drag with the left button, or drag with the middle button, to move the image. The Fit and 100 % buttons below the canvas, and the keys 0 and 1, show the whole image or one image pixel per screen pixel. Drag the thin bars between the panels to resize them; the sizes are remembered.

## Alignment modes

Hand-held and focus-bracketed series change in scale and position from frame to frame, so the frames are aligned before they are fused. Choose the mode under Settings > General > Alignment Mode:

- Registration solves shift, scale and rotation (four degrees of freedom). This is the default and the right choice for most series.
- Translation solves the shift and one uniform scale, with no rotation. Use it on a tripod or a macro rail, where the only change between frames is the focus breathing.
- Affine solves the full six degrees of freedom: shift, scale and rotation plus a stretch (aspect) and a shear. Use it when a lens or rail changes the proportions from frame to frame. It always searches within the bounded band described below.
- None skips alignment. Use it when the frames are already registered.
- Neural predicts each frame's transform with a trained alignment model. It appears only in builds made with the AI feature, never in the browser version. The Alignment Model box (greyed out when no alignment model is installed) picks the model for this session. Refine Neural Alignment Classically, on by default, uses the model's answer as the starting point for the classical optimiser; switched off, the model's transforms are applied as they are.

A settings file from an older version is read like this: its old Affine mode becomes Registration, and its old Registration mode becomes Registration with Bounded Search switched on, so a project keeps the result it had.

The classical modes search from coarse to fine on a luma pyramid. The Optimizer box chooses the search: Auto tries Lucas-Kanade (fast, needs a good starting point) and falls back to Nelder-Mead from the same start when Lucas-Kanade fails or ends worse; Lucas-Kanade and Nelder-Mead force one of the two. The browser version always uses Nelder-Mead and does not show this box.

Four switches tune the aligner:

- Bounded Search keeps the optimiser within a small band around its starting estimate (5 percent of the image size for each shift, 5 percent for the scale, 5 degrees for the rotation), which avoids degenerate results.
- AKAZE Seeding first matches image features to find a good start; it helps on textured subjects with large shifts, and is better switched off on glossy or featureless ones. Seeding needs a build with AKAZE support, which the standard build has.
- Correct Brightness (on by default) matches the brightness of every aligned frame to the reference, measured in linear light, so a flicker or an auto-exposure step does not leak into the stack. Switch it off for a series shot at deliberately different exposures.
- Chain Alignment aligns every frame to the frame before it instead of to the single reference frame, then chains the results. It helps long stacks whose end frames share little with the reference, at the price of a little accumulated drift.

The aligned frames are cropped to the area every frame covers. If that area would be smaller than half of a frame, the result is not cropped and a message names the frame that covers least. Two switches on the Output tab control this: Crop to Common Area (on by default; off keeps the full canvas, with the nearest edge pixels filling the border) and Resize Cropped to Original (off by default; stretches the cropped result back to the original size, which changes the proportions very slightly).

The Align button runs only the alignment pass. Show Aligned (toolbar, View menu) then displays the aligned version of the frame you click. A following Start Stacking reuses the aligned frames as long as the source list and the alignment settings are unchanged. If they have changed, a badge reading Stale appears in the left panel with a Recompute button.

## Fusion methods

The Algorithm box offers these methods:

- Relief (depth map) rates the sharpness around every pixel in every frame and takes each pixel from the sharpest frame, then smooths the result with a guided filter. It gives smooth, low-noise backgrounds. Its settings are the estimation radius (the size of the window that rates sharpness; larger is smoother but less precise), the smoothing radius (the guided filter), the contrast threshold (the share of the pixels with the weakest sharpness, which take the average of all frames instead) and the Multigrid Depth Solver (see below).
- Apex (Laplacian pyramid) blends the frames band by band and picks the strongest detail in each band. It copes well with fine, overlapping structures such as hair. Its settings are the number of pyramid levels, Use all color channels (also lets the colour bands vote on which frame is sharpest; the colour is still taken from that one frame) and Grit suppression (judges the finest band by a pixel and its eight neighbours, so an isolated bright speck is less likely to win; nothing is smoothed).
- Strata (guided fusion) splits every frame into a smooth base layer and a detail layer. Each layer is weighted by how sharp the frame is at that pixel, and the weights are refined with the frame's own brightness as a guide, so the transitions follow the edges in the picture. Its settings are on the Strata tab: Base Radius (8 to 64 pixels, the size of the box blur that makes the base layer) and Detail Focus (1 to 5; how strongly the detail follows the sharpest frame).
- Both (Relief & Apex) runs those two methods one after the other and lists both results.
- All Three (Strata, Relief & Apex) runs all three and lists the three results.
- AI Model (Neural) appears only in builds made with the AI feature. Pick the model, which is greyed out when none is installed, and the device when more than one is offered.

The Multigrid Depth Solver (Relief tab, off by default) replaces the "sharpest frame per pixel" choice by a depth map solved over the whole image with a multigrid method. The depth map is smoother, the depths between two frames are blended from both, and the result does not depend on the tile size. It needs a little more memory and time and handles up to 65,536 frames.

Stack every Nth frame skips frames to get a quick preview. Set it to 1 for the final run.

Compute device (on the General tab of the settings) chooses where the alignment and the Apex, Relief and Strata fusions run. CPU is the reference and the default. GPU appears only in a build made with GPU support; it uses the graphics card and goes back to the CPU, with a warning, when there is no usable card, the frames are too large for it, or the card fails. The result agrees with the CPU within a small tolerance, not bit for bit: an aligned frame can sit a fraction of a pixel away from where the CPU would put it when two frames have little in common. The AI mode does not use it; it has its own device choice.

### The depth-map preview

With Show preview during stacking switched on, Relief pauses and opens a small card over the image. The rest of the window stays usable. Move the contrast threshold and watch the mask change, or press Auto-detect to let the program choose a value (the median of the sharpness values plus 1.5 times their median absolute deviation, which copes with the long tail of a sharpness map). Apply continues with the chosen value; Cancel stops the run.

## Auto-cull, Sort and Cull

Switch on Auto-Cull Frames in the settings (General tab) to leave out frames that add almost nothing. By default a frame is dropped when it wins less than half of the average share of the sharp detail, but never more than 2 percent of it; this automatic rule scales with the number of frames. Deep stacks keep their frames this way and the fusion gets faster. The Cull Threshold % slider replaces the automatic rule with a fixed share, 0.1 to 50 percent of the sharp detail a frame must win to be kept; at 0 (shown as Auto) the automatic rule applies.

Dropped frames stay in the Source Files list, dimmed and marked as culled, so the list is still yours. If every frame would be dropped, all are kept and a warning is shown. Culling is decided again on every run.

The kept frames are fused in capture order, which the Relief depth solver needs. Switch on Sort by Sharpness (General tab) to order the frames sharpest-first by the same census instead; the sharpest frame then becomes the alignment reference.

Two buttons above the Source Files list, and two items in the Edit menu, do the same on demand without running a stack:

- Sort (Edit > Sort Frames by Sharpness) rates the frames and reorders the list sharpest-first.
- Cull (Edit > Cull Frames) rates the frames and drops the ones below the cull threshold, using the Cull Threshold % setting. The dropped frames are marked as culled in the list.

Both are unavailable while a job runs or the live monitor is on, and both need at least one loaded frame.

## Large stacks: tile size

Tile Size (General tab, desktop only) chooses how a stack is held while it is fused. In Memory, the default, keeps all frames in RAM and is the fastest. A tile size of 512, 1024, 2048 or 4096 pixels processes the picture in square tiles that are kept in a scratch folder, so a stack that does not fit in memory can still be fused; smaller tiles need less memory and more time. The browser version always works in memory. The command-line version has its own tile size option, which takes precedence over this setting in a shared settings file.

## Retouch

Retouch is a tool. Once there is a stacked result, press the Retouch button in the toolbar, or the R key, or choose Edit > Retouch Brush. A row with the brush settings appears above the image and stays until you press Done or Esc, or press R again. While the tool is off, the image is only looked at and a left-drag pans it.

With the tool on, the brush paints pixels from one source frame onto the newest result:

- Choose the frame in the Paint from box. Changing the frame later re-sources the strokes you have already painted. If you ran Align, the aligned frames are offered; otherwise the original files. A frame whose size differs from the result cannot be used, and a message says so.
- Size sets the brush radius in image pixels and Opacity how strongly it paints. The keys [ and ] make the brush smaller and larger. The ring on the canvas shows the size at the current zoom.
- Paint with the left mouse button. Hold Space and drag, or use the middle button, to pan without leaving the tool.
- Undo and Redo (Ctrl+Z, Ctrl+Y, or the buttons in the row) step through the strokes. A stroke is one drag.

The brush works only on the newest result. A line under the row says why when it is blocked, for example when an older result or a source frame is on the canvas. Strokes are part of the image: Export writes the result with the retouching included.

## Export

File > Export (Ctrl+E, or the Export button in the toolbar) opens the export dialog:

- Output folder: the folder the file is written to. Change picks another one; Ask each time clears it, and the program then asks for the name in a save dialog.
- File name template: the name of the file without its extension. The variables are name (the project name once the project has a file, otherwise the first source file), algo (the method that made the result), date and time (both in UTC), timestamp (Unix seconds), n (the number of source frames), w and h (the image size). Write them in braces, for example name_algo_date with braces around each word. The dialog shows the resulting name as you type. A word in braces that is not a variable is kept as typed, and the dialog warns you.
- Format: TIFF, PNG or JPEG. PNG and TIFF can be 16-bit or 8-bit; JPEG is always 8-bit and has a quality setting.

When the file goes into the output folder, an existing file is never overwritten: the program appends a number to the name instead, for example name (2). Characters that file systems do not allow are replaced by an underscore, and names such as CON or NUL that Windows reserves are changed.

Save As lets you choose the name in a save dialog instead; the dialog itself asks before it replaces a file. What is written depends on what is on the canvas: the newest result is encoded from the full-precision data with the retouching included, so a 16-bit file holds real 16-bit values; anything else (a source frame, an older result) is re-encoded from its file, and the message says so. With Copy Metadata on, the EXIF block of the first source frame is copied into a newest-result file saved as JPEG or PNG (the orientation tag is set upright because the app already turned the pixels); TIFF files carry no metadata, and the message says so.

## Live monitor

Live Monitor asks for a folder and watches it. Every time new frames arrive, for example from a camera that saves to a shared folder, the program stacks the folder again with Apex and shows the new result. Press Stop Monitor to end it.

## Projects

A project file remembers your work so you can continue later. It holds the list of source files in their order, all the stacking settings, the retouch strokes you painted, and an output name if the file has one. File > Save Project (Ctrl+S) writes it, Save Project As (Ctrl+Shift+S) writes a copy, and Open Project (Ctrl+Shift+O) reads one. Open Recent lists the last ten projects. New Project (Ctrl+N) starts again. Project files are named like stack.zstack.toml and are suggested next to the first source image. They do not contain the photographs themselves, only their paths, kept relative to the project file where they lie below its folder.

Strokes are not pixels: when you open a project, its strokes are painted again onto the next stack result, so stack once after opening before you expect to see them.

A star in front of the name in the title bar means that there are unsaved changes. The program compares the content with what was last saved, so undoing a change removes the star again. New, Open, Open Recent and closing the window ask whether to save, discard or cancel first.

## Drag and drop

Drop image files, or a folder of images (its files, not its subfolders), on the window to add them to the list, exactly as Add Images does. Dropping a project file opens it, after the question about unsaved changes. While a dialog is open the drop is refused and a message says so.

## Settings file

The settings dialog (Edit > Settings, Ctrl+,) edits the stacking settings in six tabs: General (alignment, frame selection, compute device, tile size), Apex, Relief, Strata, Input (rotation, crop, resize, frame order) and Output (the crop of the aligned result, file format, name and metadata). Every change takes effect at once and is saved a moment later in a file called settings.toml in the program's configuration folder: on Windows below %APPDATA%\z-stackr\config, on Linux in ~/.config/z-stackr, on macOS in ~/Library/Application Support/z-stackr. The command-line version of z-stackr reads the same format when you pass the file with its config option, and every setting reaches both programs. Load Config and Save Config in the dialog exchange settings with other files; Reset to Defaults and Load Config change the dialog and take effect when you press Apply or edit the next setting. Cancel, Esc and the cross in the corner go back to the settings as they were when you opened the dialog.

Next to it the program keeps layout.toml (the panel sizes), prefs.toml (the hints you switched off with Don't show again) and recent.toml (the recent projects). Delete prefs.toml to see the hints again. A settings file that cannot be read is renamed with a corrupt suffix instead of being overwritten, and the defaults are used.

The run log is called stacker.log and lives in the program's data folder; its path is shown in a message when the program starts.

## Keyboard shortcuts

Press ? or choose Help > Keyboard Shortcuts to see the list. The file docs/shortcuts.md in the source tree holds the same table. Shortcuts are ignored while a dialog is open; Esc closes the dialog. The depth-map preview is not a dialog, so the shortcuts keep working while it is open.

## Help and about

Help > Manual (F1) shows this page. Help > About shows the version and the licence.
