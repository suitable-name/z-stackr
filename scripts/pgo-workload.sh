#!/usr/bin/env bash
# PGO / BOLT training workload for the z-stackr CLI (Linux, macOS, Git Bash).
#
# Usage (from the repository root):
#     scripts/pgo-workload.sh [--gpu] <path-to-instrumented-z-stackr> [input-dir] [output-dir]
#
# Runs the stack in `sample/` (28 frames) through every hot path the release build is
# tuned for, so the profile covers them all:
#
#   Apex   (Laplacian pyramid)        with registration + Lucas-Kanade, brightness on
#   Relief (plane-fit depth map)      with registration + Lucas-Kanade
#   Relief with the multigrid solver
#   Strata (guided-filter blend)      with registration + Lucas-Kanade
#   Apex   with registration + Nelder-Mead (bounded)
#   Relief with the six-parameter affine model (auto optimiser, identity gate)
#
# Every run passes `--stacks single`: `sample/` holds a `result/` subfolder with images, and
# without the flag the CLI stops and asks which of `--stacks subfolders` / `--stacks single`
# is meant.
#
# --gpu (for an instrumented binary built with the `gpu` feature) repeats the same six runs
# with `compute_device = "gpu"`, so the wgpu paths (alignment objective, Apex, Relief,
# Strata kernels) are trained as well. The CPU runs stay in: they train the Lucas-Kanade
# optimiser and the CPU stages, which the GPU runs do not reach. The GPU settings are the
# files under scripts/pgo/ plus one `compute_device = "gpu"` line, written next to the
# output (no second set of files to keep in step). On a machine without a usable adapter the
# pipeline falls back to the CPU and logs a warning, so the run still works, but then the GPU
# code stays untrained; that is logged here too. Without --gpu the settings keep the default
# `compute_device = "cpu"`, which is what a CPU-only build wants (that build has no `--no-gpu`
# flag, so none is passed).
#
# The input, the tile size and the settings files under scripts/pgo/ are fixed on purpose: a
# profile is only comparable between builds when the workload does not change. Output goes to
# target/pgo-data/workload/ and is not kept.
set -euo pipefail

gpu=0
args=()
for a in "$@"; do
    case "$a" in
        --gpu) gpu=1 ;;
        *) args+=("$a") ;;
    esac
done
set -- ${args[@]+"${args[@]}"}

bin="${1:?usage: scripts/pgo-workload.sh [--gpu] <z-stackr binary> [input-dir] [output-dir]}"
input="${2:-sample}"
out="${3:-target/pgo-data/workload}"
cfg="scripts/pgo"
tile=512

mkdir -p "$out"
[ "$gpu" -eq 1 ] && mkdir -p "$out/gpu-configs"

run() {
    local name="$1" mode="$2" config="$3"
    echo "== PGO workload: $name ($mode, $config)"
    "$bin" --input-dir "$input" --output-file "$out/$name.tif" \
        --mode "$mode" --tile-size "$tile" --stacks single --config "$cfg/$config"
}

# The same run with `compute_device = "gpu"`: the settings file plus one line.
run_gpu() {
    local name="$1" mode="$2" config="$3"
    local gpu_file="$out/gpu-configs/$config"
    { cat "$cfg/$config"; printf '\ncompute_device = "gpu"\n'; } > "$gpu_file"
    echo "== PGO workload: ${name}_gpu ($mode, $config + compute_device = gpu; no usable adapter -> CPU fallback with a warning)"
    "$bin" --input-dir "$input" --output-file "$out/${name}_gpu.tif" \
        --mode "$mode" --tile-size "$tile" --stacks single --config "$gpu_file"
}

run apex_lk apex registration-lk.toml
run relief_lk relief registration-lk.toml
run relief_multigrid relief relief-multigrid.toml
run strata_lk strata registration-lk.toml
run apex_nm apex registration-nm.toml
run relief_affine relief affine-auto.toml

if [ "$gpu" -eq 1 ]; then
    run_gpu apex_lk apex registration-lk.toml
    run_gpu relief_lk relief registration-lk.toml
    run_gpu relief_multigrid relief relief-multigrid.toml
    run_gpu strata_lk strata registration-lk.toml
    run_gpu apex_nm apex registration-nm.toml
    run_gpu relief_affine relief affine-auto.toml
fi
echo "== PGO workload finished"
