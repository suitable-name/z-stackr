#!/usr/bin/env bash
# PGO / BOLT training workload for the z-stackr CLI (Linux, macOS, Git Bash).
#
# Usage (from the repository root):
#     scripts/pgo-workload.sh <path-to-instrumented-z-stackr> [input-dir] [output-dir]
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
# The input, the tile size and the settings files under scripts/pgo/ are fixed on purpose: a
# profile is only comparable between builds when the workload does not change. Output goes to
# target/pgo-data/workload/ and is not kept.
set -euo pipefail

bin="${1:?usage: scripts/pgo-workload.sh <z-stackr binary> [input-dir] [output-dir]}"
input="${2:-sample}"
out="${3:-target/pgo-data/workload}"
cfg="scripts/pgo"
tile=512

mkdir -p "$out"

run() {
    local name="$1" mode="$2" config="$3"
    echo "== PGO workload: $name ($mode, $config)"
    "$bin" --input-dir "$input" --output-file "$out/$name.tif" \
        --mode "$mode" --tile-size "$tile" --config "$cfg/$config"
}

run apex_lk apex registration-lk.toml
run relief_lk relief registration-lk.toml
run relief_multigrid relief relief-multigrid.toml
run strata_lk strata registration-lk.toml
run apex_nm apex registration-nm.toml
run relief_affine relief affine-auto.toml
echo "== PGO workload finished"
