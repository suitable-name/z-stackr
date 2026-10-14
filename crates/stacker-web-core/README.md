# z-stackr-web-core

The contract between the z-stackr browser page (`apps/stacker-web`) and its Web
Worker (`apps/stacker-web-worker`). A plain library, tested natively; it knows
nothing about the browser.

## Messages

`protocol::ToWorker` and `protocol::FromWorker` are serde enums; `encode` and
`decode` turn them into postcard bytes, which travel as a transferred
`ArrayBuffer`. `PROTOCOL_VERSION` (3) is sent in `Hello` and `Ready`. New variants
are appended at the end of an enum (version 3 appended `PassKind::Strata`).

Flow of a run:

1. The page sends `Run { job_id, settings_toml, passes, frames, reuse_aligned,
   pre_resize_override }`.
2. The worker answers with `Progress` (stage, fraction, label), `Warning`, live
   `AlignedFrame` pictures, then one `Result` per pass (fused planes, already
   cropped) and `Done`. `Failed` ends a run that could not finish.
3. With the `Relief` preview on, the worker posts `ReliefPreview` (panel-size focus
   map and first frame, the auto-detected fraction, and a 1001-entry threshold table) and
   waits for `ReliefThreshold`; `None` ends the run with `Cancelled`.
4. `Align` fills the aligned-frame cache and ends with `AlignDone`.
5. `FetchSource` returns an aligned, cropped frame (`Source` / `SourceFailed`);
   `FetchPreview` returns a thumbnail of an original file (`Preview` /
   `PreviewFailed`); `ForgetAligned` drops everything the worker holds.

A run is cancelled by terminating the worker; there is no cancel message.

## Memory budget

`WORKER_BUDGET_BYTES` is 2 GiB. `plan_budget` adds the summed size of the encoded
files and a 64 MiB margin to `session::estimate_peak_bytes` and, when the run does
not fit, asks `session::fit_pre_resize` for the largest Pre-resize percentage (a
multiple of 5) that does. Per-pixel figures (28 frames): about 850 B/px for
`Apex`, about 510 B/px for `Relief` (36 B/px more with the multigrid depth map), and
12 N + 150 B/px for `Strata`; frames alone are 12 B/px each.
`dims_from_bytes` reads the size of an image from its header.

## Helpers

`thumb` holds the picture helpers both sides share: box-averaged thumbnails in
linear light, plane downsampling, and the `Relief` threshold table (identical to
`percentile_threshold` at each of the 1001 fractions).
