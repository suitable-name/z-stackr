# Build tasks for VS Code

`docs/vscode-tasks.json` is a ready-made set of VS Code tasks for building z-stackr for a
specific CPU and for profile-guided optimisation (PGO) and BOLT. It is not installed by default,
because the right CPU level depends on who runs the binary and `.vscode/` is not committed.

## Installing it

Copy the file to `.vscode/tasks.json` in the repository root:

```bash
mkdir -p .vscode
cp docs/vscode-tasks.json .vscode/tasks.json
```

```powershell
New-Item -ItemType Directory -Force .vscode | Out-Null
Copy-Item docs\vscode-tasks.json .vscode\tasks.json
```

If you already have a `.vscode/tasks.json`, merge the `tasks` array into it. The file is JSON
with comments, which is what VS Code reads. Run a task with **Terminal > Run Task...** (or
`Ctrl+Shift+B` for the default build group) and pick it by label. Nothing in the file runs by
itself, and the tasks never start a release build unless you pick one.

The project needs the nightly toolchain pinned by `rust-toolchain.toml`; `rustup toolchain
install` inside the checkout installs it.

## The build matrix

53 tasks. The first 30 build one program for one CPU level and one operating system:

| Program | Package | Binary |
|---|---|---|
| GUI | `z-stackr-gui` | `z-stackr-gui` |
| CLI | `z-stackr-cli` | `z-stackr` |
| Trainer | `z-stackr-nn` (`--bin z-stackr-train`) | `z-stackr-train` |

| Variant | `-C target-cpu` | SIMD lanes | Runs on |
|---|---|---|---|
| `generic` | (none: baseline `x86-64`) | 4 (SSE2) | any x86-64 CPU |
| `x86-64-v3` | `x86-64-v3` | 8 (AVX2 + FMA) | Intel Haswell / AMD Zen 1 and newer |
| `x86-64-v4` | `x86-64-v4` | 16 (AVX-512) | CPUs with AVX-512 |
| `znver3` | `znver3` | 8 | AMD Zen 3 and newer (tuned for Zen 3) |
| `znver4` | `znver4` | 16 | AMD Zen 4 and newer (tuned for Zen 4) |

The vector width is fixed when the code is compiled and the output does not depend on it, so a
wider variant only changes the speed. A binary built for a newer level than the CPU has stops
with an illegal-instruction error; if in doubt use `x86-64-v3`. Each variant builds into
`target/<variant>`, so switching never rebuilds another variant's cache. Add features by
appending them to the command, for example `--features akaze,raw,gpu` (see the root README).

A further task builds `stacker-nn-eval`, the binary `scripts/04_eval_alignment.py` calls.

## PGO and BOLT

Profile-guided optimisation builds the program once with counters, runs a representative
workload, and rebuilds it with that profile. The tasks come in numbered chains; run the steps of
one chain in order:

1. build the instrumented binary,
2. run the workload,
3. merge the profile data with `llvm-profdata`,
4. build the optimised binary.

On Linux, BOLT then reorders the machine code of the PGO binary (steps 5 to 7). Windows
executables (PE) are not supported by BOLT, so the Windows chains stop at step 4.

Chains provided: CLI and GUI for `znver4` on Linux (with BOLT), CLI and GUI for `znver3` on
Windows. Those CPU names are examples: edit `target-cpu` and the directory names in all steps of
a chain to match your machine.

All steps use the `release-perf` profile (the root `Cargo.toml`): the same code as `release`,
with line tables and symbols kept, which PGO matching and BOLT need. The target directory is
`target/pgo`, apart from the normal builds; the profile data lives in `target/pgo-data`.

**Prerequisites.** `rustup component add llvm-tools`, with its `bin` directory (under the
toolchain's `lib/rustlib/<host>`) on `PATH` so that `llvm-profdata` is found; its LLVM major
version must match `rustc -vV`. BOLT needs `llvm-bolt` of the same version.

**The workload.** The CLI chains run `scripts/pgo-workload.sh` (Linux) or
`scripts/pgo-workload.ps1` (Windows) on the 28 frames in `sample/`. It covers every engine and
every alignment path that matters for speed:

| Run | Engine | Alignment (settings file in `scripts/pgo/`) |
|---|---|---|
| `apex_lk` | Apex | registration + Lucas-Kanade (`registration-lk.toml`) |
| `relief_lk` | Relief | registration + Lucas-Kanade |
| `relief_multigrid` | Relief with the multigrid solver | registration, automatic optimiser (`relief-multigrid.toml`) |
| `strata_lk` | Strata | registration + Lucas-Kanade |
| `apex_nm` | Apex | registration + bounded Nelder-Mead (`registration-nm.toml`) |
| `relief_affine` | Relief | six-parameter affine, automatic optimiser (`affine-auto.toml`) |

Brightness correction is on in all of them (the default). The GUI workload is interactive: the
task starts the instrumented GUI; load a stack, align it, fuse with Apex, Relief and Strata,
then quit (the profile is written on exit).

**Checking that the profile applied.** Both cargo phases of a chain run the same command and
differ only in `RUSTFLAGS`. If they did not (a different `-p`, `--features` or `--target`), the
crate hashes in the symbol names differ and LLVM silently skips every function of the profile.
Step 4 passes `-pgo-warn-missing-function`: a flood of `no profile data available for function`
warnings that name z-stackr functions means the profile did not match. Do not edit the sources
between steps 1 and 4.

## Build scripts (PGO, every CPU and GPU variant)

For release binaries of every CPU and GPU variant, two scripts run the whole PGO chain
(instrumented build, training, merge, check, optimised build) per combination; the VS Code
tasks above remain for single builds and experiments.

```powershell
pwsh -File scripts/pgo-build.ps1                 # Windows
```

```bash
scripts/pgo-bolt-build.sh                        # Linux: PGO, then BOLT on the CLI
```

| Axis | Windows | Linux | Values |
|---|---|---|---|
| CPU | `-Cpu` | `--cpu` | `scalar` (`x86-64-v2`), `avx2` (`x86-64-v3`), `avx512` (`x86-64-v4`), `native`, `all` (default: scalar, avx2 and avx512 when the host has it) |
| GPU | `-Gpu` | `--gpu` | `gpu` (Cargo feature `gpu`), `cpu` (none), `both` (default) |
| Target | `-Target` | `--target` | `cli`, `gui`, `all` (default; one combined build) |

Other switches: `-SkipTrain` / `--skip-train` (reuse the profile in `pgo-profile/`), `-OutDir` /
`--out-dir` (default `dist`), and on Linux `--no-bolt`. `akaze` and `raw` are always built; `nn`,
`nn-gpu` and `ui-harness` never are. An explicit AVX-512 request on a CPU without AVX-512 is
refused, and `all` drops it. The SIMD width is fixed at compile time (no runtime dispatch), so
every CPU tier is trained once, under its own build.

**Where the binaries land.** In `dist/`, as `z-stackr-win-<cpu>-<gpu>.exe` and
`z-stackr-gui-win-<cpu>-<gpu>.exe` on Windows, `z-stackr-linux-<cpu>-<gpu>` and
`z-stackr-gui-linux-<cpu>-<gpu>` on Linux, plus `z-stackr-linux-<cpu>-<gpu>-bolt` (the BOLT
result for the CLI). Merged profiles are kept in `pgo-profile/`, build logs in
`target/pgo-all-<cpu>-<gpu>/build.log` (`pgo-cli-...` for `--target cli`).

**What is trained.** `-Target all` builds `z-stackr-cli` and `z-stackr-gui` in one cargo
invocation, so the shared library crates get one crate hash and the one profile applies to both
apps. The profile comes from `scripts/pgo-workload.*` run with the instrumented CLI (the six
runs above, all with `--stacks single`; in the `gpu` tier also with `compute_device = "gpu"`,
and without a usable adapter the pipeline falls back to the CPU with a warning, which leaves the
GPU code untrained). The GUI is not driven, so GUI-only code (the Slint wiring) stays cold.
After the optimised build the scripts compare the crate-hash tokens of `stacker_pipeline` and
`stacker_algo` in the profile with the built rlibs and scan `build.log` for "hash mismatch".

**Tools.** The `llvm-tools` component of the pinned toolchain (`rustup component add llvm-tools
--toolchain nightly-2026-09-08`; its LLVM must match `rustc -vV`), and on Linux `llvm-bolt` and
`merge-fdata` of the same LLVM version (or `--no-bolt`).

Builds and training runs are the owner's to run and time; nothing in the repository starts
them. Speed figures from them are only comparable on a quiet machine.
