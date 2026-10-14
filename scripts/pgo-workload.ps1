# PGO training workload for the z-stackr CLI (Windows, PowerShell 7+ or Windows PowerShell).
#
# Usage (from the repository root):
#     pwsh -File scripts/pgo-workload.ps1 -Binary <path-to-instrumented-z-stackr.exe> [-Gpu]
#
# Runs the stack in `sample/` (28 frames) through every hot path the release build is tuned
# for, so the profile covers them all:
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
# -Gpu (for an instrumented binary built with the `gpu` feature) repeats the same six runs
# with `compute_device = "gpu"`, so the wgpu paths (alignment objective, Apex, Relief,
# Strata kernels) are trained as well. The CPU runs above stay in: they train the Lucas-Kanade
# optimiser and the CPU stages, which the GPU runs do not reach. The GPU settings are the
# files under scripts/pgo/ plus one `compute_device = "gpu"` line, written next to the
# output (no second set of files to keep in step). On a machine without a usable adapter the
# pipeline falls back to the CPU and logs a warning, so the run still works, but then the GPU
# code stays untrained; that is logged here too. Without -Gpu the settings keep the default
# `compute_device = "cpu"`, which is what a CPU-only build wants (that build has no `--no-gpu`
# flag, so none is passed).
#
# The input, the tile size and the settings files under scripts/pgo/ are fixed on purpose: a
# profile is only comparable between builds when the workload does not change. Output goes to
# target/pgo-data/workload/ and is not kept. Same steps as scripts/pgo-workload.sh.
param(
    [Parameter(Mandatory = $true)] [string] $Binary,
    [string] $InputDir = "sample",
    [string] $OutDir = "target/pgo-data/workload",
    [switch] $Gpu
)

$ErrorActionPreference = "Stop"
$cfg = "scripts/pgo"
$tile = 512

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$gpuCfg = Join-Path $OutDir "gpu-configs"
if ($Gpu) { New-Item -ItemType Directory -Force -Path $gpuCfg | Out-Null }

function Invoke-Run([string] $Name, [string] $Mode, [string] $Config) {
    Write-Host "== PGO workload: $Name ($Mode, $Config)"
    & $Binary --input-dir $InputDir --output-file "$OutDir/$Name.tif" `
        --mode $Mode --tile-size $tile --stacks single --config "$cfg/$Config"
    if ($LASTEXITCODE -ne 0) { throw "z-stackr exited with code $LASTEXITCODE ($Name)" }
}

# The same run with `compute_device = "gpu"`: the settings file plus one line.
function Invoke-GpuRun([string] $Name, [string] $Mode, [string] $Config) {
    $gpuFile = Join-Path $gpuCfg $Config
    $text = (Get-Content -Raw -LiteralPath (Join-Path $cfg $Config)).TrimEnd()
    # UTF-8 without a BOM on both PowerShell editions (a BOM would trip the TOML parser).
    $full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($gpuFile)
    [System.IO.File]::WriteAllText($full, $text + "`ncompute_device = `"gpu`"`n", (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "== PGO workload: ${Name}_gpu ($Mode, $Config + compute_device = gpu; no usable adapter -> CPU fallback with a warning)"
    & $Binary --input-dir $InputDir --output-file "$OutDir/${Name}_gpu.tif" `
        --mode $Mode --tile-size $tile --stacks single --config $gpuFile
    if ($LASTEXITCODE -ne 0) { throw "z-stackr exited with code $LASTEXITCODE (${Name}_gpu)" }
}

$runs = @(
    @("apex_lk", "apex", "registration-lk.toml"),
    @("relief_lk", "relief", "registration-lk.toml"),
    @("relief_multigrid", "relief", "relief-multigrid.toml"),
    @("strata_lk", "strata", "registration-lk.toml"),
    @("apex_nm", "apex", "registration-nm.toml"),
    @("relief_affine", "relief", "affine-auto.toml")
)

foreach ($r in $runs) { Invoke-Run $r[0] $r[1] $r[2] }
if ($Gpu) {
    foreach ($r in $runs) { Invoke-GpuRun $r[0] $r[1] $r[2] }
}
Write-Host "== PGO workload finished"
