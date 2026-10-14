# z-stackr keyboard shortcuts

The same list is shown in the app: press `?` or click the `?` button in the
toolbar. On macOS, Cmd replaces Ctrl.

Shortcuts are ignored while a dialog is open (settings, export, manual, about,
a confirmation, a first-use explainer, this list); only Esc is handled there, and
it closes the dialog. The `Relief` preview is a non-modal card over the canvas, not
a dialog: the shortcuts keep working while it is open.

The table below is generated from `src/shortcuts_table.rs` (the single source
of truth). Do not edit it by hand: change `SHORTCUTS`, then paste the output of
`markdown_table()` between the markers. `tests/shortcuts_test.rs` fails when the
two differ.

<!-- SHORTCUTS:BEGIN -->
| Keys | Action | Where |
| --- | --- | --- |
| Ctrl+N | Start a new project | Global |
| Ctrl+Shift+O | Open a project | Global |
| Ctrl+O | Add images | Global |
| Ctrl+S | Save the project | Global |
| Ctrl+Shift+S | Save the project under a new name | Global |
| Ctrl+E | Export the image (opens the export dialog) | Global |
| Ctrl+, | Open the settings | Global |
| Ctrl+Z | Undo the last retouch stroke | Global |
| Ctrl+Y | Redo the last undone stroke (also Ctrl+Shift+Z) | Global |
| F1 | Show the manual | Global |
| ? | Show this list of shortcuts | Global |
| Esc | Close the open dialog or overlay, leave the Retouch tool, or dismiss the message | Global |
| + | Zoom in (also =) | Image view |
| - | Zoom out | Image view |
| 0 | Fit the image to the window | Image view |
| 1 | Show the image at 100 % | Image view |
| Space | Hold to pan with a left-button drag while the Retouch tool is on (middle button pans too) | Image view |
| R | Turn the Retouch tool on or off (needs a stacked result) | Image view |
| [ | Smaller brush (Retouch tool on) | Image view |
| ] | Larger brush (Retouch tool on) | Image view |
<!-- SHORTCUTS:END -->
