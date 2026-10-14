# PGO training workload for the z-stackr CLI (Windows, PowerShell 7+ or Windows PowerShell).
#
# Usage (from the repository root):
#     pwsh -File scripts/pgo-workload.ps1 -Binary <path-to-instrumented-z-stackr.exe>
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
# The input, the tile size and the settings files under scripts/pgo/ are fixed on purpose: a
# profile is only comparable between builds when the workload does not change. Output goes to
# target/pgo-data/workload/ and is not kept. Same steps as scripts/pgo-workload.sh.
param(
    [Parameter(Mandatory = $true)] [string] $Binary,
    [string] $InputDir = "sample",
    [string] $OutDir = "target/pgo-data/workload"
)

$ErrorActionPreference = "Stop"
$cfg = "scripts/pgo"
$tile = 512

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Invoke-Run([string] $Name, [string] $Mode, [string] $Config) {
    Write-Host "== PGO workload: $Name ($Mode, $Config)"
    & $Binary --input-dir $InputDir --output-file "$OutDir/$Name.tif" `
        --mode $Mode --tile-size $tile --config "$cfg/$Config"
    if ($LASTEXITCODE -ne 0) { throw "z-stackr exited with code $LASTEXITCODE ($Name)" }
}

Invoke-Run "apex_lk" "apex" "registration-lk.toml"
Invoke-Run "relief_lk" "relief" "registration-lk.toml"
Invoke-Run "relief_multigrid" "relief" "relief-multigrid.toml"
Invoke-Run "strata_lk" "strata" "registration-lk.toml"
Invoke-Run "apex_nm" "apex" "registration-nm.toml"
Invoke-Run "relief_affine" "relief" "affine-auto.toml"
Write-Host "== PGO workload finished"
