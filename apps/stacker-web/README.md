# z-stackr web

The z-stackr desktop interface in the browser. It is the same window (`apps/stacker-gui`'s
Slint UI and Rust wiring) drawn on a canvas; the stacking pipeline runs in a Web Worker so
the page stays responsive. Nothing is uploaded: the images never leave the tab.

## Run it

```text
rustup target add wasm32-unknown-unknown
cargo install --locked trunk
cd apps/stacker-web
trunk serve
```

Then open <http://127.0.0.1:8768>. `trunk serve` builds with the optimised
`stacker-web-dev` profile (debug wasm is far too slow to stack with) and rebuilds when
the page, the worker, the protocol crate, the pipeline or the GUI library change.

A browser with WebAssembly SIMD and WebGL2 is needed (Chrome or Edge 91, Firefox 89,
Safari 16.4 or newer); alignment additionally needs WebGPU (Chrome or Edge 113, Firefox 141
on Windows, Safari 26 or newer), see below.

## Release

```text
cd apps/stacker-web
trunk build --release
```

`dist/` is a plain static site: any static host works, and no special headers are needed
(no COOP/COEP, no `SharedArrayBuffer`). The page loads no third-party script.

## What differs from the desktop app

* **No Live Monitor, no AI, no Neural alignment, no RAW.** A browser cannot watch a folder,
  and the AI fusion, the Neural alignment mode and the RAW decoder need the native build.
  Fusion (Apex, Relief with the guided filter or the multigrid depth map, Strata), the
  auto-cull and decoding run on the CPU in one worker, single-threaded. Camera RAW files
  are not accepted; develop them to JPEG, PNG or TIFF first. Out-of-core (tiled) stacking
  is a desktop and CLI feature: the worker holds every frame in memory.
* **Engines.** The algorithm list offers Relief (Depth Map), Apex (Laplacian), Strata
  (Guided Fusion), and the combinations of them; "AI Model (Neural)" is desktop only.
  Old project files that name PMax or DMap open as Apex and Relief.
* **Optimizer.** The browser's GPU alignment refines with the Nelder-Mead search; the
  desktop's Lucas-Kanade optimizer has no asynchronous form, so Auto and Lucas-Kanade both
  run Nelder-Mead here. Brightness correction and chained alignment work as on the
  desktop.
* **Alignment runs on the GPU only.** The browser aligns on WebGPU, in the worker; there
  is no CPU alignment here (it took about a minute per frame). WebGPU is in Chrome and
  Edge 113 or newer, Firefox 141 or newer on Windows, and Safari 26. Without it, Start
  Stacking with an alignment mode other than None asks "Stack without alignment?" and the
  Align button says the browser has no WebGPU; alignment mode None needs no GPU. The status
  line shows the adapter once the worker is up. A GPU failure during alignment fails that
  run with a message (no CPU fallback). Results differ from the desktop's by rounding, not
  by method.
* **Files are downloads and uploads.** Add Images, drag and drop and Open Project read
  the chosen files into the page; Save, Export and Save Config download a file. There is
  no output folder, no recent-projects list and no folder picking.
* **Same file name, same frame.** The page identifies a file by its name: adding a second
  file with a name that is already loaded replaces the first.
* **Projects.** Save downloads `<name>.zstack.toml`. To open one, pick the project file
  (or drop it); the app then asks for its images, which it matches by file name.
* **Metadata.** With Copy Metadata on, the first frame's EXIF block is written into the
  result files (JPEG and PNG; TIFF cannot carry it).
* **Settings** are kept in the browser's local storage (`z-stackr.settings`,
  `z-stackr.prefs`, `z-stackr.layout`).
* **Memory.** A browser tab fails on memory long before the 4 GiB a wasm32 module can
  address, so the app estimates a run's peak before it starts and budgets 2 GiB. When a
  stack does not fit it offers to stack at the largest Pre-resize percentage that does
  (a multiple of 5) for that run, or to cancel; the status line says the run was reduced.
  The desktop app and the CLI stack the same images at full size.
* **Cancel** stops the worker and starts a new one, so the aligned frames are lost and the
  next run aligns again.
* **Shortcuts.** Ctrl+N belongs to the browser (a new window), so New Project is Alt+N
  here (the shortcut list shows the keys in effect).

## Crates

| Crate | Path | Role |
| --- | --- | --- |
| `z-stackr-web` (this one) | `apps/stacker-web` | The page: entry point, the browser `Platform` (virtual files, pickers, downloads, local storage) and the `JobRunner` that talks to the worker. |
| `z-stackr-web-worker` | `apps/stacker-web-worker` | The Web Worker: decodes a message, runs the pipeline (`stacker_pipeline::session`), posts the reply. |
| `z-stackr-web-core` | `crates/stacker-web-core` | The postcard message protocol between the two and the memory budget policy; plain library, tested natively. |

The interface and the wiring are `apps/stacker-gui`'s library (`stacker_gui`); this crate
only supplies what a browser does differently, through the `platform` and `runner` seams.
