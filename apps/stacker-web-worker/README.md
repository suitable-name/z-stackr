# z-stackr-web-worker

The Web Worker of the z-stackr browser app. Trunk builds it as a worker
(`data-type="worker"`) next to `apps/stacker-web`; the page starts it with
`new Worker(...)`. On a native build the crate is empty.

It runs the same in-memory stack sequence as the desktop app
(`stacker_pipeline::session`): decode, align, cull, reorder, fuse, crop, on one
thread (no AKAZE seeding). Alignment (each pyramid level's objective and the
final warp) runs on WebGPU through `z-stackr-gpu`, awaited; there is no CPU
alignment in the browser. Decoding, the cull and fusion run on the CPU. The
messages are those of `z-stackr-web-core`.

## Async structure

At start-up the worker opens WebGPU (`GpuContext::new_async`) and then posts
`Ready { gpu_adapter }` (`None`: no WebGPU, the reason is in the console). A Run,
an Align, a FetchSource and a FetchPreview are each a task on the event loop
(`spawn_local`). A second Run or Align while one is in flight is answered with
`Failed` ("busy"); `ReliefThreshold` still resumes a parked run. A Run or Align
that aligns while there is no WebGPU fails with the GPU-only message; alignment
mode None needs no GPU. Unlike the desktop backend the web backend evaluates
every pyramid level on the GPU, however small (there is no CPU to hand it to).

## State between messages

- the encoded files of the last Run or Align (kept by name for `FetchPreview`
  and for a Run with `reuse_aligned`)
- the aligned frames: after an Align all decoded frames; after a Run the frames
  that entered the fusion, in fusion order (the names culled by earlier runs are
  remembered so a reusing Run still reports them)
- a `Relief` run parked at its contrast preview, waiting for `ReliefThreshold`

The worker cannot read a message while it computes, so cancelling means the page
terminates it and starts a fresh one; the cache is then gone.

## What the page decides

The page sends the frames already selected (every N-th, reversed, natural order)
and the effective settings as TOML. A `pre_resize_override` replaces the
Pre-resize percentage for one run. An Align ends with `AlignDone` only; a Run
ends with `Done`, `Failed` or `Cancelled`.

## Memory

The worker holds the encoded files plus 12 B/px per decoded frame; a run's peak is
about 850 B/px for `Apex`, about 510 B/px for `Relief` (36 B/px more with the multigrid
depth map) and 12 N + 150 B/px for `Strata` over N frames (see `z-stackr-web-core`'s
budget). The page checks the run against a 2 GiB budget before it sends the files.

## Passes and limits

`Apex`, `Relief` (guided filter, or the multigrid depth map when `relief_use_multigrid`
is on) and `Strata` all run on the CPU. A message that asks for the `Neural` alignment
mode fails with a message: the browser has no neural runtime. The effective settings
force the Nelder-Mead optimizer (the asynchronous GPU path has no Lucas-Kanade), no
AKAZE seeding and no tiling.
