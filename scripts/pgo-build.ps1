<#
.SYNOPSIS
  Profile-guided-optimization (PGO) release builds of z-stackr (the CLI `z-stackr` and the
  desktop app `z-stackr-gui`) on Windows, over CPU and GPU axes.

    CPU:  avx512 -> -C target-cpu=x86-64-v4 (AVX-512; Zen 4 / Skylake-X or newer). Opt-in:
                    this host's native CPU support is probed before an avx512 build is
                    attempted (see Test-Avx512Supported below); an explicit `-Cpu avx512`
                    on an unsupported host refuses with a clear error instead of crashing
                    mid-training (0xC000001D, illegal instruction), and the default
                    `-Cpu all` silently drops avx512 from the tier list on such a host.
          avx2   -> -C target-cpu=x86-64-v3 (AVX2+FMA+BMI baseline; Haswell/Zen or newer)
          scalar -> -C target-cpu=x86-64-v2 (SSE4.2 baseline; standard non-AVX fallback)
          native -> -C target-cpu=native (host CPU instruction set)
          all    -> avx512 (when the host has it), avx2 and scalar
    GPU:  gpu    -> Cargo feature `gpu` on both apps (wgpu kernels, CPU fallback when there
                    is no usable adapter); the training workload also runs with
                    `compute_device = "gpu"`
          cpu    -> no gpu feature at all (pure CPU binaries)
          both   -> gpu, then cpu (default)

  The z-stackr SIMD lane width is fixed at compile time (`portable_simd`, sized by the
  build's target features); there is NO runtime SIMD dispatch. So each CPU tier is trained
  ONCE, under its own build. That is why the two-run tier training Indicatrix's
  scripts/pgo-build.ps1 does (one run with the host's native dispatch, one capped) is absent
  here: a build only ever executes its own kernels.

  sccache is switched off (RUSTC_WRAPPER / CARGO_BUILD_RUSTC_WRAPPER empty) for the whole
  combination, instrumented and -Cprofile-use builds alike: a cache hit would serve objects
  built without -Cprofile-generate (empty profiles) or stale profile-use objects.

  Toolchain: the repository pins `nightly-2026-09-08` (rust-toolchain.toml). The script runs
  from the repository root, so every `cargo` / `rustc` call below uses that toolchain, and
  llvm-profdata / llvm-nm are taken from THAT toolchain's `llvm-tools` component.

.DESCRIPTION
  `-Target all` (the default) is ONE combined build per combination, not a CLI run plus a GUI
  run: a single instrumented `cargo build` over `z-stackr-cli` and `z-stackr-gui` together
  (features = `akaze` and `raw` on both, plus `gpu` on both for the gpu tier), the training
  workload runs once, ONE merged profile, and ONE optimised build that emits BOTH binaries.
  Why: one package set means feature unification compiles the shared library crates
  (z-stackr-pipeline, z-stackr-algo, z-stackr-align, z-stackr-core and, with gpu,
  z-stackr-gpu) once, with one crate hash shared by both apps, so a single profile applies
  to both -- and the work is done once instead of twice. Instrumented and optimised builds
  use the same package set, features, target, target-dir and RUSTFLAGS apart from the profile
  flag; if they differed, every profiled crate would get a different hash, profile-use would
  find nothing to apply and the binaries would be optimised as if untrained.

  `-Target gui` builds the SAME combined package set (the GUI has no scriptable workload, so
  its profile can only come through the shared library crates, which the CLI workload trains,
  and the two apps must be in one build for the hashes to match); it shares target directory
  and profile with `-Target all` and copies only the GUI binary. `-Target cli` builds the CLI
  alone, in its own target directory and profile subdirectory (its package set, and therefore
  its crate hashes, differ from the combined build).

  Per combination, in its own target directory (target\pgo-all-<cpu>-<gpu>, or
  target\pgo-cli-<cpu>-<gpu> for -Target cli):
    1. one `cargo build` instrumented with -C profile-generate over the selected packages
       (--bins).
    2. run the instrumented z-stackr.exe through scripts\pgo-workload.ps1 (Apex, Relief with
       and without multigrid, Strata and Apex-NM/affine alignment on the 28 frames in
       sample\, every run with `--stacks single`); in the gpu tier with -Gpu, which repeats
       the runs with `compute_device = "gpu"`. Without a usable adapter the pipeline falls
       back to the CPU: the run still works, a warning is printed, and the GPU code stays
       untrained in that profile.
    3. merge the .profraw files with llvm-profdata.
    4. rebuild the SAME packages, target, target-dir and feature list as step 1, with
       -C profile-use=<merged profile> instead of -C profile-generate (and
       -C llvm-args=-pgo-warn-missing-function, so a totally-unapplied profile shows up as a
       flood of "no profile data available" warnings in build.log instead of silently
       compiling as if untrained).
    5. verify the profile actually applied (crate-hash check of `stacker_pipeline` and
       `stacker_algo` against the merged .profdata, plus a build.log scan for
       "hash mismatch" / "no profile data available" -- see Test-ProfileHashApplied and
       Test-ProfileWarnings).
    6. copy the executables to <OutDir> as z-stackr-win-<cpu>-<gpu>.exe and
       z-stackr-gui-win-<cpu>-<gpu>.exe.

  Training coverage is a known limitation. The GUI is NOT driven: nothing here runs the
  instrumented desktop app, so the GUI-only code (the Slint wiring, event loops, dialogs,
  the retouch and preview code of z-stackr-gui itself) has zero profile counts and is
  optimised as cold code. The GUI binary benefits only through the shared library crates the
  CLI workload exercises (pipeline, algo, align, core, gpu). The workload does not exercise
  the AI mode (`nn`; never built here), the monitor mode, or the RAW decoders (the sample
  frames are JPEG).

.PARAMETER Cpu
  avx512, avx2, scalar, native, or all (default all; see the CPU axis note above -- avx512
  is dropped automatically from `all` on a host without AVX-512 support, and refused
  outright if requested explicitly on such a host).
.PARAMETER Gpu
  gpu, cpu, or both (default both).
.PARAMETER Target
  cli, gui, or all (default all); see .DESCRIPTION for what each builds and shares.
.PARAMETER OutDir
  Where the renamed executables are copied (default <repo>\dist).
.PARAMETER SkipTrain
  Reuse an existing profile from the pgo-profile directory
  (windows-<cpu>-<gpu>.profdata; pgo-profile\cli\ for -Target cli).
.PARAMETER MaxMissingFunctionWarnings
  Fallback threshold for the post-build "no profile data available for function" warning
  count (default 400000; see Test-ProfileWarnings) -- only consulted when
  Test-ProfileHashApplied could not verify the profile; once the crate-hash check has
  confirmed it applied, the count is reported and never fatal. The count is large by design:
  the GUI, Slint and every untrained dependency report their functions. This default is NOT
  calibrated against a known-good run; tighten it once a real run has established the
  baseline count.
#>
[CmdletBinding()]
param(
    [ValidateSet('avx512', 'avx2', 'scalar', 'native', 'all')]
    [string]$Cpu = 'all',
    [ValidateSet('gpu', 'cpu', 'both')]
    [string]$Gpu = 'both',
    [ValidateSet('cli', 'gui', 'all')]
    [string]$Target = 'all',
    [string]$OutDir = '',
    [switch]$SkipTrain,
    [int]$MaxMissingFunctionWarnings = 400000
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$hostTarget = 'x86_64-pc-windows-msvc'
if (-not $OutDir) { $OutDir = Join-Path $root 'dist' }
# Resolve a relative -OutDir against the caller's directory, before the cd below.
$OutDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir)
$pgoProfileDir = Join-Path $root 'pgo-profile'
$workloadScript = Join-Path $PSScriptRoot 'pgo-workload.ps1'
# Unit separator: how CARGO_ENCODED_RUSTFLAGS joins individual rustc arguments, so a repo
# path containing a space can't split a flag in two the way space-joined RUSTFLAGS would.
$unitSep = [char]0x1F

# Everything below (rustc, cargo, rustup) must resolve the repository's pinned toolchain.
Set-Location $root

# `-Target all` and `-Target gui` are ONE combined build (see .DESCRIPTION): their own target
# directory prefix keeps them from sharing rlibs/logs with a `cli`-only run, so the crate-hash
# check can never pick up another package set's newest rlib.
$dirPrefix = if ($Target -eq 'cli') { 'pgo-cli' } else { 'pgo-all' }
# Package set: identical for the instrumented and the optimised build.
$pkgList = if ($Target -eq 'cli') { @('z-stackr-cli') } else { @('z-stackr-cli', 'z-stackr-gui') }
$binList = switch ($Target) {
    'cli' { @('z-stackr') }
    'gui' { @('z-stackr-gui') }
    default { @('z-stackr', 'z-stackr-gui') }
}

# --- llvm-profdata / llvm-nm from the pinned toolchain ------------------------------
$sysroot = (& rustc --print sysroot).Trim()
$llvmBin = Join-Path $sysroot "lib\rustlib\$hostTarget\bin"
$profdata = Join-Path $llvmBin 'llvm-profdata.exe'
$llvmNm = Join-Path $llvmBin 'llvm-nm.exe'
if (-not (Test-Path $profdata) -or -not (Test-Path $llvmNm)) {
    if (Get-Command rustup -ErrorAction SilentlyContinue) {
        Write-Host "llvm-tools not in the sysroot; trying 'rustup component add llvm-tools' (harmless if this is not a rustup toolchain)..."
        try { & rustup component add llvm-tools 2>&1 | Out-Null } catch {}
    }
    if (-not (Test-Path $profdata)) {
        $fromPath = Get-Command llvm-profdata -ErrorAction SilentlyContinue
        if ($fromPath) { $profdata = $fromPath.Source } else { throw "llvm-profdata not found at $profdata nor in PATH. Install the pinned toolchain's llvm-tools: 'rustup component add llvm-tools --toolchain nightly-2026-09-08' (the version in rust-toolchain.toml). Its LLVM must match this rustc's." }
    }
    if (-not (Test-Path $llvmNm)) {
        $fromPath = Get-Command llvm-nm -ErrorAction SilentlyContinue
        if ($fromPath) { $llvmNm = $fromPath.Source } else { throw "llvm-nm not found at $llvmNm nor in PATH (same fix as for llvm-profdata: 'rustup component add llvm-tools --toolchain nightly-2026-09-08')." }
    }
}

# --- AVX-512 host support probe ------------------------------------------------------
# `-Cpu avx512` (x86-64-v4) crashes a training run with 0xC000001D (illegal instruction)
# on any host whose CPU lacks AVX-512 -- there is no graceful runtime fallback for an
# instrumented binary built with a target-cpu baseline it can't execute. Probe once,
# up front, instead of discovering this mid-run.
function Test-Avx512Supported {
    $cfg = & rustc --print cfg -C target-cpu=native 2>$null
    return [bool]($cfg | Select-String -SimpleMatch 'target_feature="avx512f"')
}

$avx512Supported = $null
function Get-Avx512Supported {
    if ($null -eq $script:avx512Supported) { $script:avx512Supported = Test-Avx512Supported }
    return $script:avx512Supported
}

if ($Cpu -eq 'avx512' -and -not (Get-Avx512Supported)) {
    throw "-Cpu avx512 requested, but this host's native CPU has no AVX-512 support " +
    "(rustc --print cfg -C target-cpu=native has no target_feature=`"avx512f`"). " +
    "An instrumented avx512 binary would crash immediately with 0xC000001D (illegal " +
    "instruction) instead of training. Use -Cpu avx2 or -Cpu scalar, or run this on " +
    "AVX-512-capable hardware."
}

$cpuList = if ($Cpu -eq 'all') { @('avx512', 'avx2', 'scalar') } else { @($Cpu) }
if ($Cpu -eq 'all' -and $cpuList -contains 'avx512' -and -not (Get-Avx512Supported)) {
    Write-Warning ("Dropping avx512 from -Cpu all: this host's native CPU has no AVX-512 " +
        "support (see Test-Avx512Supported). Building avx2 and scalar only. Pass -Cpu avx512 " +
        "explicitly on AVX-512-capable hardware to opt back in.")
    $cpuList = @($cpuList | Where-Object { $_ -ne 'avx512' })
}
$gpuList = if ($Gpu -eq 'both') { @('gpu', 'cpu') } else { @($Gpu) }

# One feature list, used VERBATIM by both the instrumented (profile-generate) and the
# optimised (profile-use) cargo invocation, so both produce the same crate hashes.
# `akaze` and `raw` are always on (the desktop release builds read camera RAW and seed the
# aligner with AKAZE features); `gpu` follows the -Gpu axis. Never `nn` / `nn-gpu` (no
# trained models ship) and never `ui-harness`. Package-qualified so one `--features` string
# serves both packages of the combined build.
function Get-PgoFeatures([string[]]$Packages, [string]$GpuName) {
    $features = @()
    foreach ($pkg in $Packages) {
        $features += "$pkg/akaze"
        $features += "$pkg/raw"
        if ($GpuName -eq 'gpu') { $features += "$pkg/gpu" }
    }
    return ($features -join ',')
}

# The rustc flags both builds share (only the profile flag differs between them). v0 symbol
# mangling puts a per-crate hash token (`Cs<hash>_<len><name>`) into every symbol, which the
# crate-hash check below compares; naming it explicitly keeps that check independent of the
# toolchain's default mangling.
function Get-BaseFlags([string]$TargetCpu) {
    return @('-C', "target-cpu=$TargetCpu", '-C', 'symbol-mangling-version=v0')
}

# Runs the training workload (scripts\pgo-workload.ps1) with the instrumented CLI. In the
# gpu tier the workload also runs with `compute_device = "gpu"`. `$env:LLVM_PROFILE_FILE`
# must already point into the profile directory.
function Invoke-Workload([string]$Exe, [string]$GpuName) {
    $wl = @{ Binary = $Exe }
    if ($GpuName -eq 'gpu') {
        Write-Host "   gpu tier: the workload also runs with compute_device = gpu (no usable adapter -> the pipeline falls back to the CPU with a warning, and the GPU code stays untrained)"
        $wl.Gpu = $true
    }
    & $workloadScript @wl
}

# --- Profile-applied verification: the merged profile must cover the crate hash the
#     optimised build just produced, i.e. the instrumented and optimised builds must
#     produce the same crate hash or profile-use silently finds nothing to apply.
#     Best-effort: a parsing miss prints a warning and returns rather than failing the whole
#     build over a diagnostic script's own regex, but a CONFIRMED mismatch between the two
#     crate-hash tokens is a hard failure.
# Returns $true when the match was CONFIRMED for every checked crate, $false when any could
# not be verified; throws on a confirmed mismatch in either.
function Test-ProfileHashApplied([string]$ProfdataFile, [string]$Tdir, [string]$Name) {
    # Several candidate functions per crate: public, non-generic, trained by the workload.
    # The first one found in BOTH the profile and the rlib is compared.
    $pipelineOk = Test-CrateHashToken -ProfdataFile $ProfdataFile -Tdir $Tdir -Name $Name `
        -FuncNames @('align_frame_with_compute', 'plan_tiles', 'commit_frame', 'apron_for') `
        -PackageDir 'z-stackr-pipeline' -RlibStem 'stacker_pipeline'
    $algoOk = Test-CrateHashToken -ProfdataFile $ProfdataFile -Tdir $Tdir -Name $Name `
        -FuncNames @('build_and_fuse_pyramids', 'compute_focus_map', 'fuse_strata_on', 'apply_gaussian_blur') `
        -PackageDir 'z-stackr-algo' -RlibStem 'stacker_algo'
    return ($pipelineOk -and $algoOk)
}

# Compares one crate's hash token between the merged profile and the newest optimised rlib
# of that crate. `$PackageDir` is the package's directory name under the cargo build-dir
# layout, `$RlibStem` the rlib's crate-name stem (underscores).
function Test-CrateHashToken([string]$ProfdataFile, [string]$Tdir, [string]$Name, [string[]]$FuncNames, [string]$PackageDir, [string]$RlibStem) {
    $tokenPattern = "Cs[0-9a-zA-Z]+_$($RlibStem.Length)$RlibStem"
    $allFunctions = @(& $profdata show --all-functions $ProfdataFile 2>&1)

    # The compiled library is `lib<stem>-<hash>.rlib` in one of two layouts: the new cargo
    # build-dir layout (`build\<package>\<hash>\out\`, current nightly) or the classic one
    # (`deps\`). More than one may exist; take the newest across both -- the optimised
    # build that just finished.
    $releaseRoot = Join-Path $Tdir "$hostTarget\release"
    $rlib = @(
        Get-ChildItem -Path (Join-Path $releaseRoot "build\$PackageDir\*\out\lib$RlibStem-*.rlib") -ErrorAction SilentlyContinue
        Get-ChildItem -Path (Join-Path $releaseRoot "deps\lib$RlibStem-*.rlib") -ErrorAction SilentlyContinue
    ) | Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $rlib) {
        Write-Warning "[$Name] no lib$RlibStem-*.rlib found under $releaseRoot\build\$PackageDir\*\out or $releaseRoot\deps -- cannot verify the $PackageDir crate-hash match."
        return $false
    }
    $nmLines = @(& $llvmNm $rlib.FullName 2>&1)

    foreach ($func in $FuncNames) {
        # Every manifest line that names the function, not just the first: a merged profile
        # can hold the same symbol under more than one crate hash, so every distinct token
        # is collected and the rlib's token must be among them.
        $profLines = @($allFunctions | Select-String -SimpleMatch $func)
        if ($profLines.Count -eq 0) { continue }
        $nmLine = $nmLines | Select-String -SimpleMatch $func | Select-Object -First 1
        if (-not $nmLine) { continue }
        $profTokens = @($profLines | ForEach-Object { [regex]::Match($_.Line, $tokenPattern).Value } |
            Where-Object { $_ } | Sort-Object -Unique)
        $nmToken = [regex]::Match($nmLine.Line, $tokenPattern).Value
        if ($profTokens.Count -eq 0 -or -not $nmToken) { continue }

        if ($profTokens -notcontains $nmToken) {
            throw "[$Name] profile did NOT apply to ${PackageDir}: the merged profile's crate-hash token ($($profTokens -join ',')) differs from the optimised build's rlib ($nmToken) for '$func'. The instrumented and optimised builds resolved '$func' to two different compiled crates (their package set, target, target-dir, feature list or flags differ), so profile-use silently found nothing to apply."
        }
        Write-Host "  [$Name] crate-hash check OK ($PackageDir): $func -> $nmToken"
        return $true
    }
    Write-Warning "[$Name] none of '$($FuncNames -join "', '")' was found with a '$tokenPattern' token in both $ProfdataFile and $($rlib.FullName) -- cannot verify the $PackageDir crate-hash match (training may not have reached it, or the symbol text differs)."
    return $false
}

# Scans an optimised build's log for the two warning classes -pgo-warn-missing-function and
# profile staleness produce. See -MaxMissingFunctionWarnings' doc comment for why "no profile
# data available" is thresholded rather than required to be zero, and why "hash mismatch" is
# not.
function Test-ProfileWarnings([string]$LogFile, [string]$Name, [int]$MaxMissing, [bool]$HashVerified) {
    if (-not (Test-Path $LogFile)) { return }
    $lines = Get-Content -LiteralPath $LogFile
    $hashMismatch = @($lines | Select-String -SimpleMatch 'hash mismatch')
    $noProfile = @($lines | Select-String -SimpleMatch 'no profile data available')

    if ($hashMismatch.Count -gt 0) {
        $sample = ($hashMismatch | Select-Object -First 5 | ForEach-Object { $_.Line }) -join "`n"
        throw "[$Name] $($hashMismatch.Count) 'hash mismatch' warning(s) in $LogFile. With identical --target/--target-dir/package-set/--features between the instrumented and optimised builds and no source edits between the two steps, this should be zero -- a function's compiled code differs from what was profiled. First few:`n$sample"
    }
    if ($HashVerified) {
        # The crate-hash check already proved the profile applied; the count is only code
        # the training run never executes (the whole GUI, Slint, untrained dependencies).
        Write-Host "  [$Name] $($noProfile.Count) function(s) without profile data (untrained code; the crate-hash check confirmed the profile applied)"
        return
    }
    if ($noProfile.Count -gt $MaxMissing) {
        throw "[$Name] $($noProfile.Count) 'no profile data available' warning(s) in $LogFile, over the configured threshold ($MaxMissing, -MaxMissingFunctionWarnings), and the crate-hash check could not verify the profile. Expected in bulk for the GUI and Slint code that the CLI workload never runs -- but without the crate-hash confirmation a count this high may mean the profile silently stopped applying. Check the crate-hash warning printed just above."
    }
    Write-Host "  [$Name] $($noProfile.Count) function(s) without profile data (crate-hash check could not verify; under the $MaxMissing threshold)"
}

# Only the combined build keeps the un-prefixed profile path; a cli-only build's package set
# (and crate hashes) differ, so it gets its own subdirectory and never mixes with it.
function Get-SharedProfilePath([string]$Name) {
    if ($Target -eq 'cli') { return Join-Path (Join-Path $pgoProfileDir 'cli') "windows-$Name.profdata" }
    return Join-Path $pgoProfileDir "windows-$Name.profdata"
}

function Build-PgoCombination {
    param([string]$CpuName, [string]$TargetCpu, [string]$GpuName)

    $name = "$CpuName-$GpuName"
    $tdir = Join-Path $root "target\$dirPrefix-$name"
    $pdir = Join-Path $tdir 'profiles'
    $localMerged = Join-Path $tdir 'merged.profdata'
    $sharedMerged = Get-SharedProfilePath $name
    $baseFlags = Get-BaseFlags $TargetCpu
    $logfile = Join-Path $tdir 'build.log'
    New-Item -ItemType Directory -Force $tdir | Out-Null
    $null > $logfile # Truncate/create log file, so Test-ProfileWarnings has something to scan

    # Same feature string, same package set for BOTH cargo invocations below.
    $features = Get-PgoFeatures $pkgList $GpuName
    $pkgArgs = $pkgList | ForEach-Object { '-p', $_ }

    $savedEncodedRustflags = $env:CARGO_ENCODED_RUSTFLAGS
    $savedRustflags = $env:RUSTFLAGS
    $savedProfileFile = $env:LLVM_PROFILE_FILE
    $savedWrapper = $env:RUSTC_WRAPPER
    $savedCargoWrapper = $env:CARGO_BUILD_RUSTC_WRAPPER

    function Invoke-Checked {
        param([string]$Description, [scriptblock]$Command, [string]$LogFile)
        Write-Host "==> $Description" -ForegroundColor Cyan
        "==> $Description" | Out-File -FilePath $LogFile -Append -Encoding utf8
        & $Command *>&1 | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0) { throw "$Description failed (exit $LASTEXITCODE)" }
    }

    try {
        Push-Location $root
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
        # sccache OFF for every cargo call of this combination, i.e. the instrumented build
        # AND the final -Cprofile-use build. An empty env value overrides any configured
        # `rustc-wrapper` (cargo treats an empty wrapper as none). With the wrapper on, an
        # instrumented crate can be served from the cache as an object compiled WITHOUT
        # -Cprofile-generate -> the training run writes empty profiles, and a profile-use
        # build can be served stale objects keyed on the profdata PATH.
        $env:RUSTC_WRAPPER = ''
        $env:CARGO_BUILD_RUSTC_WRAPPER = ''

        if (-not $SkipTrain -or -not (Test-Path $sharedMerged)) {
            if (Test-Path $pdir) { Remove-Item -Recurse -Force $pdir }
            New-Item -ItemType Directory -Force $pdir | Out-Null

            $env:CARGO_ENCODED_RUSTFLAGS = (@($baseFlags) + @('-C', "profile-generate=$pdir")) -join $unitSep
            Invoke-Checked "[$name] instrumented build (packages: $($pkgList -join ', '); features: $features)" {
                cargo build --release --target $hostTarget --target-dir $tdir @pkgArgs --bins --features $features
            } $logfile

            $env:LLVM_PROFILE_FILE = Join-Path $pdir 'train-%p-%m.profraw'
            $exe = Join-Path $tdir "$hostTarget\release\z-stackr.exe"
            Invoke-Checked "[$name] training run (scripts\pgo-workload.ps1)" {
                Invoke-Workload $exe $GpuName
            } $logfile

            $raw = @(Get-ChildItem -Path $pdir -Filter '*.profraw')
            if ($raw.Count -eq 0) { throw "no .profraw files were written to $pdir" }
            Invoke-Checked "[$name] merging $($raw.Count) profile(s)" {
                & $profdata merge -o $localMerged $pdir
            } $logfile

            $sharedMergedDir = Split-Path -Parent $sharedMerged
            if (-not (Test-Path $sharedMergedDir)) { New-Item -ItemType Directory -Force $sharedMergedDir | Out-Null }
            Copy-Item -Force $localMerged $sharedMerged
            $activeProfile = $sharedMerged
        }
        else {
            Write-Host "[$name] reusing shared profile $sharedMerged"
            $activeProfile = $sharedMerged
        }

        Remove-Item Env:LLVM_PROFILE_FILE -ErrorAction SilentlyContinue
        # Same package/target/target-dir/features as the instrumented build above,
        # byte-for-byte, so both builds produce the same crate hashes.
        # `-pgo-warn-missing-function` surfaces every function profile-use can't find data
        # for, in $logfile, for Test-ProfileWarnings to scan below.
        $env:CARGO_ENCODED_RUSTFLAGS = (@($baseFlags) + @('-C', "profile-use=$activeProfile", '-C', 'llvm-args=-pgo-warn-missing-function')) -join $unitSep

        Invoke-Checked "[$name] PGO release build (packages: $($pkgList -join ', '); features: $features)" {
            cargo build --release --target $hostTarget --target-dir $tdir @pkgArgs --bins --features $features
        } $logfile

        # Crate-hash check FIRST: it is the real proof the profile applied; the
        # missing-function count is only fatal when it could not verify.
        $verified = Test-ProfileHashApplied -ProfdataFile $activeProfile -Tdir $tdir -Name $name
        Test-ProfileWarnings -LogFile $logfile -Name $name -MaxMissing $MaxMissingFunctionWarnings -HashVerified ([bool]$verified)

        if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
        $releaseDir = Join-Path $tdir "$hostTarget\release"
        foreach ($bin in $binList) {
            $src = Join-Path $releaseDir "$bin.exe"
            if (-not (Test-Path $src)) { throw "expected binary missing: $src" }
            $dst = Join-Path $OutDir "$bin-win-$name.exe"
            Copy-Item -Force $src $dst
            Write-Host ("   {0}  ({1:N1} MB)" -f $dst, ((Get-Item $dst).Length / 1MB)) -ForegroundColor Green
        }
        Write-Host ""
    }
    finally {
        Pop-Location
        if ($null -ne $savedEncodedRustflags) { $env:CARGO_ENCODED_RUSTFLAGS = $savedEncodedRustflags } else { Remove-Item Env:CARGO_ENCODED_RUSTFLAGS -ErrorAction SilentlyContinue }
        if ($null -ne $savedRustflags) { $env:RUSTFLAGS = $savedRustflags } else { Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue }
        if ($null -ne $savedProfileFile) { $env:LLVM_PROFILE_FILE = $savedProfileFile } else { Remove-Item Env:LLVM_PROFILE_FILE -ErrorAction SilentlyContinue }
        if ($null -ne $savedWrapper) { $env:RUSTC_WRAPPER = $savedWrapper } else { Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue }
        if ($null -ne $savedCargoWrapper) { $env:CARGO_BUILD_RUSTC_WRAPPER = $savedCargoWrapper } else { Remove-Item Env:CARGO_BUILD_RUSTC_WRAPPER -ErrorAction SilentlyContinue }
    }
}

foreach ($c in $cpuList) {
    $targetCpu = switch ($c) {
        'avx512' { 'x86-64-v4' }
        'avx2' { 'x86-64-v3' }
        'scalar' { 'x86-64-v2' }
        'native' { 'native' }
    }
    foreach ($g in $gpuList) {
        Build-PgoCombination -CpuName $c -TargetCpu $targetCpu -GpuName $g
    }
}

Write-Host "`nAll requested PGO builds are in $OutDir" -ForegroundColor Green
