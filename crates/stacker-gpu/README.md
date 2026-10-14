# z-stackr-gpu

GPU kernels for z-stackr through [wgpu](https://wgpu.rs), kept out of the CPU
crates. The crate identifier is `stacker_gpu`.

**Status: Apex, Relief, Strata kernels, the Relief multigrid solver and
alignment.** `z-stackr-pipeline` uses this crate in
builds with its `gpu` feature (`z-stackr-cli --features gpu`,
`z-stackr-gui --features gpu`), through `stacker_pipeline::compute::Compute`, when
the settings say `compute_device = "gpu"`. Decoding, colour conversion,
auto-cull, export and the AI mode (which has its own `nn-gpu` path) stay on the
CPU. The CPU code in `stacker-algo` and `stacker-align` stays the reference and the
fallback.

## What it does

| Module | Plan | Mirrors |
|---|---|---|
| `apex` | `ApexPlan`: a Laplacian pyramid per frame, the strongest coefficient per pixel and band (the first frame wins ties), the optional 3×3 grit window on the finest band, the averaged base, the reconstruction | `stacker_algo::apex` |
| `relief` | `ReliefPlan`: the plane-fit focus measure per frame folded into the best frame per pixel (the first frame wins ties) and a running sum; the maximum focus map read back for the threshold; then the selection (best frame where the focus is `>=` the threshold, the mean elsewhere) and the self-guided filter on the luma | `stacker_algo::relief`, `stacker_pipeline::fuse` |
| `strata` | `StrataPlan`: the saliency of a luma plane (4-neighbour Laplacian magnitude, five passes of the pyramid blur) and the guided-filter pair that refines the layer weights (direct `f32` window sums over mean-shifted values where the CPU reads `f64` summed-area tables) | `stacker_algo::strata::saliency`, `stacker_algo::relief::guided::guided_filter_pair_luma` |
| `multigrid` | `MultigridPlan`: the pins are uploaded once, the whole cycle (restrict, prolong, Jacobi sweeps, in the CPU's order) runs on the device and the padded finest level is read back once; levels below 8192 samples run in one workgroup per cycle from a host-built schedule, the larger ones dispatch per operation; the final `pyramid_smooth` stays on the CPU | `stacker_algo::relief::multigrid` |
| `align` | `ObjectivePlan`: one pyramid level of the registration objective; each evaluation warps the reference by a candidate matrix (bicubic, outside pixels left out, or the clamped spline) and returns the moments of `source - warped` | `stacker_align::refine` |
| `align` | `WarpPlan`: the final warp of a frame (bicubic with a black border, or the clamped spline) and its coverage mask from the same dispatch | `stacker_align::transform` |

Frames are folded in one at a time, so the memory of the fusion plans does not
depend on the number of frames. `apex::plan_bytes`, `relief::plan_bytes`,
`align::objective_bytes` and `align::warp_bytes` give what a plan needs without
touching the device (a 24 MP frame needs about 2 GiB for Apex, 1.5 GiB for Relief,
0.9 GiB for the warp).

An objective evaluation is one dispatch and a read-back of one small vector per
workgroup (each thread folds its pixels with Welford's update, the workgroup
merges them pairwise, the host merges the workgroups in `f64` in a fixed order),
so the optimiser's hundreds of evaluations per level move no image.

`GpuContext::new(None)` opens the best adapter: a discrete GPU before an
integrated one, Vulkan before Direct3D 12 and Metal, a software adapter only when
nothing else exists, and never an OpenGL adapter (it lacks the compute features
the kernels need). `GpuContext::adapters` lists them all, and an index picks one.

## In the browser

For `wasm32-unknown-unknown` the crate builds with wgpu's WebGPU backend and offers
only the alignment kernels, through an async API that never blocks, so it works
inside a Web Worker:

- `GpuContext::new_async().await` asks the browser for a high-performance WebGPU
  adapter and opens a device with the adapter's buffer limits (`NoAdapter` when the
  browser has no WebGPU);
- `ObjectivePlan::moments_async` and `WarpPlan::warp_async` run the same dispatch as
  `moments` and `warp` (one shared encode step) and await the read-back through a
  oneshot channel the map callback completes. On native targets they also work
  (the device is polled once after the request), and the tests check them bit for
  bit against the blocking calls.

`apex`, `relief`, `strata`, `multigrid`, `GpuContext::new`, `GpuContext::adapters`,
`ObjectivePlan::moments` and `WarpPlan::warp` are native only; fusion (Apex, Relief, Strata
and the Relief multigrid solver) stays on the CPU in the browser, where Strata and the
multigrid run in the worker's CPU path. Lucas-Kanade (the first
step of the `auto` optimiser) is a CPU path in every build: only the Nelder-Mead
objective runs here, so a GPU run with `optimizer = "neldermead"` is the one that
exercises the alignment kernels fully. Both
align kernels fit WebGPU's defaults: at most 3 storage buffers per stage (limit 8),
256 invocations per workgroup (limit 256) and 3 KiB of workgroup memory (limit
16 KiB). The WebGPU device is never polled by the host; validation errors reach
the context's error sink asynchronously and are reported by the next read-back.

## When things go wrong

wgpu reports device errors (validation, out of memory, device lost) by panicking.
The context collects them instead; a plan checks after building, after each frame
or evaluation and after reading back, and returns `GpuError::Device`. A caller
such as `Compute` treats any error as a failure of the GPU path and carries on
with the CPU. `GpuContext::inject_failure` (hidden from the docs) makes the next
plan, fold (or evaluation, or warp) or read-back fail once, to test exactly that.

## Agreement with the CPU

Within a tolerance, not bit for bit: a driver may fuse a multiply and an add, the
division and the reciprocal are not correctly rounded on every GPU, and the Relief
window sums are `f32` on the device where the CPU uses `f64` tables (on values
shifted by the frame's mean, which keeps them small). Measured by the tests:

- Apex: largest difference about 3e-7 on small stacks;
- Relief: focus map within 5e-7 of its range; a winner can flip where two frames'
  focus values are within rounding of each other (on 8 real 8 MP frames, 0.016 %
  of the pixels differ by more than 1e-3, the mean difference is 2e-6);
- Strata saliency: within 1e-5 of the map's range (the parity test asserts it);
  guided-filter pair: worst sample below 1e-3 and mean below 1e-5 for masks in
  `0..=1` (expected about 1e-5 worst; the tests print the measured values);
- multigrid: the padded solution within 1e-3 frame indices at worst and 1e-4 on
  average (the operations and their order are the CPU's; only fused multiply-adds
  differ). **Not yet timed:** whether the single-workgroup tail beats the CPU
  solve has to be measured (`MultigridPlan::dispatches` says how many dispatches
  a solve records; `multigrid::TAIL_SAMPLES` is the knob);
- objective: the same pixel count and an RMS within 1e-7 relative;
- warp: within 3e-6, the same coverage mask.

An alignment on the device therefore takes a slightly different path through the
simplex. Where the objective has one clear minimum the result is the CPU's to
about 0.001 px; on a pair whose objective is flat (two frames of a focus stack
far apart in focus), both land on points of equal quality that can be some tenths
of a pixel apart (the CPU's own modes differ by as much there).

The tests print `SKIPPED` and pass on a machine without a GPU adapter. The WGSL is
also parsed and validated by naga in unit tests, which need no GPU.

## Writing kernels

naga accepts more than the drivers compile. Two rules from this crate's shaders:
no function returns an `array`, and no function-local `array` is indexed in a
loop: the AMD Vulkan driver crashed and FXC (Direct3D 12) refused such code that
naga validated. Use `vec4` and unrolled taps.

## Testing and timing

```bash
cargo test -p z-stackr-gpu
cargo run --release -p z-stackr-gpu --example apex_bench -- --width 1536 --height 1536 --frames 50 --levels 8 --reps 3 --check
cargo run --release -p z-stackr-gpu --example relief_phases -- 3840 2160 8
cargo run --release -p z-stackr-pipeline --features gpu --example gpu_profile -- sample 8
```

`apex_bench` prints the CPU time, the GPU time end to end, and the GPU time split
into compute, upload and download. `relief_phases` splits the Relief into fold,
focus read-back and finish. `gpu_profile` compares the CPU and the GPU on real
frames for one objective evaluation per pyramid level, the alignment of one frame
per mode, the final warp and the Relief.

On an integrated Radeon with shared memory (3840×2160 frames). **Provisional:**
measured while other work ran on the machine; to be re-measured.

| Stage | CPU | GPU |
|---|---|---|
| one objective evaluation, full resolution | 88 ms | 2.3 ms |
| alignment of one frame, affine | 12.8 s | 0.9 to 1.3 s |
| alignment of one frame, affine bounded | 22 to 30 s | 0.8 to 1.5 s |
| alignment of one frame, registration | 8 to 9 s | 0.5 to 0.8 s |
| final warp | 250 ms | 110 to 130 ms |
| Relief of 8 frames | 1.0 s | about 0.5 s (fold 37 ms per frame) |

The upload (about 5 GB/s on shared memory) bounds the warp and the Relief; the
alignment gains most because the frames go up once per pyramid level and every
evaluation after that stays on the device. A discrete GPU on PCIe should do
better; run the examples to see.
