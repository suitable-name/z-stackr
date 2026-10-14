#!/usr/bin/env bash
#
# Profile-guided optimization (PGO), then BOLT, for the z-stackr CLI (`z-stackr`) and the
# desktop app (`z-stackr-gui`) on Linux, over CPU and GPU axes. Windows has no BOLT: use
# scripts/pgo-build.ps1 there.
#
# Axes:
#   --cpu avx512   -C target-cpu=x86-64-v4 (AVX-512). Opt-in: the host's native CPU support
#                  is probed first (`rustc --print cfg -C target-cpu=native`); an explicit
#                  `--cpu avx512` on a host without it refuses with a clear error (an
#                  instrumented binary would die with SIGILL mid-training), and `--cpu all`
#                  silently drops it.
#         avx2     -C target-cpu=x86-64-v3 (AVX2+FMA+BMI; Haswell/Zen or newer)
#         scalar   -C target-cpu=x86-64-v2 (SSE4.2 baseline)
#         native   -C target-cpu=native
#         all      avx512 (when the host has it), avx2 and scalar (default)
#   --gpu gpu      Cargo feature `gpu` on both apps; the training workload also runs with
#                  `compute_device = "gpu"` (no usable adapter -> the pipeline falls back
#                  to the CPU with a warning, the run still works, the GPU code stays untrained)
#         cpu      no gpu feature (pure CPU binaries)
#         both     gpu, then cpu (default)
#   --target cli   the CLI alone (own target dir and profile)
#            gui   the GUI, built in the combined package set (see below); only its binary is copied
#            all   both apps (default)
#
# z-stackr has NO runtime SIMD dispatch: its `portable_simd` lane width is fixed at compile
# time by the build's target features. Each CPU tier is therefore trained ONCE, under its
# own build; that is why the two-run tier training of Indicatrix's pgo-bolt-build.sh (one run
# with the host's native dispatch, one capped to the tier) is absent here: a build only ever
# executes its own kernels.
#
# Package/feature parity (the instrumented and optimised builds must produce the same crate
# hash, or the profile silently does not apply): both builds select IDENTICAL packages,
# features, target, target-dir and RUSTFLAGS -- ONLY the profile flag differs. `--target all`
# is ONE combined build: a single instrumented `cargo build` over `z-stackr-cli` and
# `z-stackr-gui` together (features `akaze` and `raw` on both, plus `gpu` for the gpu tier;
# never `nn`, `nn-gpu` or `ui-harness`), ONE workload run, ONE merged profile and ONE
# optimised build that emits both binaries. One package set means feature unification
# compiles the shared library crates (z-stackr-pipeline, -algo, -align, -core and, with gpu,
# -gpu) once with one crate hash shared by both apps, so the single profile applies to both.
# Had the two apps been built separately, or the instrumented step selected another package
# set than the optimised one, every crate's `-C metadata` (and so every mangled symbol)
# would differ and profile-use would find nothing to apply. `--target gui` uses the same
# combined package set (the GUI has no scriptable workload: its profile comes only through
# the shared library crates the CLI workload trains) and shares target dir and profile with
# `--target all`; `--target cli` is a separate, CLI-only package set with its own profile
# under pgo-profile/cli/.
#
# Training (per combination, in target/pgo-<all|cli>-<cpu>-<gpu>):
#   1. instrumented `cargo build` (-C profile-generate), --bins, both apps;
#   2. scripts/pgo-workload.sh with the instrumented `z-stackr`: Apex, Relief with and
#      without multigrid, Strata, registration LK/NM and affine alignment on the 28 frames
#      in sample/, every run with `--stacks single` (`--gpu` repeats them with
#      compute_device = "gpu" in the gpu tier);
#   3. llvm-profdata merge;
#   4. the optimised build (-C profile-use, -C llvm-args=-pgo-warn-missing-function);
#   5. checks: crate-hash tokens of `stacker_pipeline` and `stacker_algo` in the merged
#      profile against the optimised rlibs, and a build.log scan for "hash mismatch" /
#      "no profile data available";
#   6. the binaries are copied to the output dir as z-stackr-linux-<cpu>-<gpu> and
#      z-stackr-gui-linux-<cpu>-<gpu>.
#
# The GUI is NOT driven interactively. In the combined build the GUI binary shares the
# trained library crates, but GUI-only code (the Slint wiring, event loops, dialogs, the
# retouch/preview code of z-stackr-gui itself) has no profile counts and stays cold. The
# workload also skips the AI mode (`nn`; never built here), the monitor mode and the RAW
# decoders (the sample frames are JPEG).
#
# BOLT (default; `--no-bolt` switches it off): after the PGO build, the CLI binary is
# instrumented with llvm-bolt, the same workload runs again, and the binary is rewritten with
# the merged BOLT profile. The BOLT result is copied as z-stackr-linux-<cpu>-<gpu>-bolt next
# to the plain PGO binary. BOLT applies to the CLI only: the GUI is interactive with no
# scriptable workload, and a startup-only profile would lay its code out worse than none. To
# let BOLT rewrite the binary, both builds link it unstripped with relocations kept
# (`-C link-arg=-Wl,--emit-relocs`, frame pointers on) -- the same flags in the instrumented
# and the optimised build, so they stay comparable -- and the copies are stripped afterwards.
#
# Toolchain: the repository pins nightly-2026-09-08 (rust-toolchain.toml). The script cds to
# the repository root, so cargo/rustc are the pinned toolchain; llvm-profdata and llvm-nm
# must come from ITS llvm-tools component and match rustc's LLVM version:
#   rustup component add llvm-tools --toolchain nightly-2026-09-08
# BOLT needs llvm-bolt and merge-fdata (LLVM's bolt package) in PATH, same LLVM major.
#
# Usage (from anywhere):
#   scripts/pgo-bolt-build.sh                        # all CPU tiers x both GPU tiers, both apps, PGO + BOLT
#   scripts/pgo-bolt-build.sh --cpu avx2 --gpu gpu   # one combination
#   scripts/pgo-bolt-build.sh --target cli           # the CLI only
#   scripts/pgo-bolt-build.sh --no-bolt              # PGO only
#   scripts/pgo-bolt-build.sh --skip-train           # reuse pgo-profile/linux-<cpu>-<gpu>.profdata
#   scripts/pgo-bolt-build.sh --out-dir <dir>        # default <repo>/dist
#
set -euo pipefail

# Disable sccache/cache wrappers: a cache hit would serve objects built without
# -Cprofile-generate (empty profiles) or stale profile-use objects.
export RUSTC_WRAPPER=""
export CARGO_BUILD_RUSTC_WRAPPER=""
export RUSTC_WORKSPACE_WRAPPER=""

# ---------------------------------------------------------------------------------
# Argument defaults & parsing
# ---------------------------------------------------------------------------------
CPU_ARG=all
GPU_ARG=both
TARGET_ARG=all
OUT_DIR=""
SKIP_TRAIN=0
USE_BOLT=1
# "no profile data available" fires in bulk, and expectedly, for the GUI, Slint and every
# dependency the CLI workload never runs. This count is only a FALLBACK gate, used when
# test_profile_hash_applied could not verify the profile (a parsing miss); once the
# crate-hash check has proven the profile applied, the count is reported and never fatal.
# The default is NOT calibrated against a known-good run: tighten it after a real run.
MAX_MISSING_FUNCTION_WARNINGS=400000
# Set to 1 by test_profile_hash_applied when it CONFIRMED the merged profile's crate hashes
# match the optimised rlibs; reset to 0 before every combination's checks.
PROFILE_HASH_VERIFIED=0

die() { printf '%s\n' "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)          CPU_ARG="${2:?--cpu needs a value}"; shift 2 ;;
        --gpu)          GPU_ARG="${2:?--gpu needs a value}"; shift 2 ;;
        --target)       TARGET_ARG="${2:?--target needs a value}"; shift 2 ;;
        --out-dir)      OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --skip-train)   SKIP_TRAIN=1; shift ;;
        --no-bolt)      USE_BOLT=0; shift ;;
        --max-missing-function-warnings) MAX_MISSING_FUNCTION_WARNINGS="${2:?--max-missing-function-warnings needs a value}"; shift 2 ;;
        -h|--help)      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0 ;;
        *)              die "unknown argument: $1 (try --help)" ;;
    esac
done

case "$CPU_ARG" in
    all)                       CPU_LIST=(avx512 avx2 scalar) ;;
    avx512|avx2|scalar|native) CPU_LIST=("$CPU_ARG") ;;
    *)                         die "--cpu must be avx512, avx2, scalar, native, or all" ;;
esac

case "$GPU_ARG" in
    both)    GPU_LIST=(gpu cpu) ;;
    gpu|cpu) GPU_LIST=("$GPU_ARG") ;;
    *)       die "--gpu must be gpu, cpu, or both" ;;
esac

case "$TARGET_ARG" in
    all|cli|gui) ;;
    *)           die "--target must be cli, gui, or all" ;;
esac

# ---------------------------------------------------------------------------------
# Repository root and toolchain
# ---------------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Everything below must resolve the repository's pinned toolchain.
cd "$ROOT"
[ -n "$OUT_DIR" ] || OUT_DIR="$ROOT/dist"
case "$OUT_DIR" in /*) ;; *) OUT_DIR="$PWD/$OUT_DIR" ;; esac
PGO_PROFILE_DIR="$ROOT/pgo-profile"
# Unit separator: how CARGO_ENCODED_RUSTFLAGS joins individual rustc arguments, so a repo
# path containing a space can't split a flag in two the way space-joined RUSTFLAGS would.
US=$'\x1f'

HOST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
case "$HOST_TARGET" in
    *linux*) ;;
    *) die "this script trains and builds for Linux hosts (this host is $HOST_TARGET). Use scripts/pgo-build.ps1 on Windows." ;;
esac
SYSROOT="$(rustc --print sysroot)"

# `all` and `gui` are ONE combined build (see the header); `cli` is its own package set.
if [ "$TARGET_ARG" = cli ]; then
    DIR_PREFIX=pgo-cli
    PKG_LIST=(z-stackr-cli)
    BIN_LIST=(z-stackr)
else
    DIR_PREFIX=pgo-all
    PKG_LIST=(z-stackr-cli z-stackr-gui)
    if [ "$TARGET_ARG" = gui ]; then BIN_LIST=(z-stackr-gui); else BIN_LIST=(z-stackr z-stackr-gui); fi
fi

# --- AVX-512 host support probe --------------------------------------------------
# avx512 (x86-64-v4) SIGILLs immediately on a host CPU without AVX-512, so probe once, up
# front, instead of discovering it mid-training-run.
avx512_supported() {
    rustc --print cfg -C target-cpu=native 2>/dev/null | grep -q 'target_feature="avx512f"'
}

if [[ " ${CPU_LIST[*]} " =~ " avx512 " ]] && ! avx512_supported; then
    if [ "$CPU_ARG" = avx512 ]; then
        die "--cpu avx512 requested, but this host's native CPU has no AVX-512 support (rustc --print cfg -C target-cpu=native has no target_feature=\"avx512f\"). An instrumented avx512 binary would crash immediately (SIGILL) instead of training. Use --cpu avx2 or --cpu scalar, or run this on AVX-512-capable hardware."
    fi
    printf 'warning: dropping avx512 from --cpu all: this host'"'"'s native CPU has no AVX-512 support. Building avx2 and scalar only. Pass --cpu avx512 explicitly on AVX-512-capable hardware to opt back in.\n' >&2
    filtered=()
    for cpu in "${CPU_LIST[@]}"; do
        [ "$cpu" = avx512 ] && continue
        filtered+=("$cpu")
    done
    CPU_LIST=("${filtered[@]}")
fi

# llvm-profdata and llvm-nm must be the SAME LLVM major.minor as rustc. The llvm-tools in a
# sysroot are installed separately from rustc and are not updated with it; a stale llvm-nm
# cannot read the bitcode inside an LTO build's rlibs, and an older llvm-profdata writes a
# profile the compiler rejects.
#
# Candidates, first matching version wins:
#   1. $PGO_LLVM_BIN (an explicit directory);
#   2. the sysroot's rustlib (the llvm-tools component of the pinned toolchain);
#   3. /usr/lib/llvm/<major>/bin (Gentoo's slotted LLVM);
#   4. PATH.
llvm_major_minor() { grep -oE 'LLVM version:? [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -n1; }
RUSTC_LLVM="$(rustc -vV | llvm_major_minor || true)"
find_llvm_tool() {
    local name="$1" candidate version first_found=""
    local -a candidates=()
    [ -n "${PGO_LLVM_BIN:-}" ] && candidates+=("$PGO_LLVM_BIN/$name")
    candidates+=("$SYSROOT/lib/rustlib/$HOST_TARGET/bin/$name")
    [ -n "$RUSTC_LLVM" ] && candidates+=("/usr/lib/llvm/${RUSTC_LLVM%%.*}/bin/$name")
    command -v "$name" >/dev/null 2>&1 && candidates+=("$(command -v "$name")")
    for candidate in "${candidates[@]}"; do
        [ -x "$candidate" ] || continue
        [ -z "$first_found" ] && first_found="$candidate"
        version="$("$candidate" --version 2>&1 | llvm_major_minor || true)"
        if [ -z "$RUSTC_LLVM" ] || [ "$version" = "$RUSTC_LLVM" ]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    if [ -n "$first_found" ]; then
        die "no $name matching rustc's LLVM $RUSTC_LLVM found (first candidate $first_found is LLVM $("$first_found" --version 2>&1 | llvm_major_minor || true)). Reinstall the pinned toolchain's llvm-tools (rustup component add llvm-tools --toolchain nightly-2026-09-08), install LLVM $RUSTC_LLVM, or point PGO_LLVM_BIN at a directory holding a matching $name."
    fi
    die "$name not found. Install the pinned toolchain's llvm-tools: rustup component add llvm-tools --toolchain nightly-2026-09-08 (or set PGO_LLVM_BIN, or put a matching LLVM in PATH)."
}
PROFDATA="$(find_llvm_tool llvm-profdata)"
LLVM_NM="$(find_llvm_tool llvm-nm)"

# BOLT validation
if [ "$USE_BOLT" -eq 1 ]; then
    if ! command -v llvm-bolt >/dev/null 2>&1 || ! command -v merge-fdata >/dev/null 2>&1; then
        die "BOLT is on (the default) but llvm-bolt or merge-fdata was not found in PATH. Install LLVM's bolt package (same LLVM major as rustc: $RUSTC_LLVM), or pass --no-bolt for PGO only."
    fi
fi

step() { printf '\n==> %s\n' "$*"; }

# The full --features string for a gpu axis value: `akaze` and `raw` always (desktop release
# builds read camera RAW and seed the aligner with AKAZE features), `gpu` per the axis.
# Package-qualified, so one string serves both packages. Never nn/nn-gpu/ui-harness. Used
# VERBATIM by the instrumented and the optimised build -- see the parity note above.
pgo_features() {
    local gpu="$1" pkg
    local -a feats=()
    for pkg in "${PKG_LIST[@]}"; do
        feats+=("$pkg/akaze" "$pkg/raw")
        [ "$gpu" = gpu ] && feats+=("$pkg/gpu")
    done
    local IFS=,
    printf '%s' "${feats[*]}"
}

# Scans an optimised build's log for the two warning classes -pgo-warn-missing-function and
# profile staleness produce: "hash mismatch" is always fatal; the "no profile data available"
# count is only thresholded when the crate-hash check could not verify the profile.
test_profile_warnings() {
    local logfile="$1" label="$2"
    [ -f "$logfile" ] || return 0

    local hash_mismatch_count no_profile_count
    hash_mismatch_count="$(grep -c 'hash mismatch' "$logfile" || true)"
    no_profile_count="$(grep -c 'no profile data available' "$logfile" || true)"

    if [ "${hash_mismatch_count:-0}" -gt 0 ]; then
        printf '[%s] %s "hash mismatch" warning(s) in %s. With identical --target/--target-dir/package-set/--features/RUSTFLAGS between the instrumented and optimised builds and no source edits between the two steps, this should be zero -- a function'"'"'s compiled code differs from what was profiled. First few:\n' "$label" "$hash_mismatch_count" "$logfile" >&2
        grep -m5 'hash mismatch' "$logfile" >&2 || true
        die "[$label] profile verification failed (hash mismatch) -- see above"
    fi
    if [ "${PROFILE_HASH_VERIFIED:-0}" -eq 1 ]; then
        printf '  [%s] %s function(s) without profile data (untrained code; the crate-hash check confirmed the profile applied)\n' "$label" "${no_profile_count:-0}"
        return 0
    fi
    if [ "${no_profile_count:-0}" -gt "$MAX_MISSING_FUNCTION_WARNINGS" ]; then
        die "[$label] $no_profile_count 'no profile data available' warning(s) in $logfile, over the threshold ($MAX_MISSING_FUNCTION_WARNINGS, --max-missing-function-warnings), and the crate-hash check could not verify the profile. Expected in bulk for the GUI and Slint code the CLI workload never runs -- but without the crate-hash confirmation a count this high may mean the profile silently stopped applying."
    fi
    printf '  [%s] %s function(s) without profile data (crate-hash check could not verify; under the %s threshold)\n' "$label" "${no_profile_count:-0}" "$MAX_MISSING_FUNCTION_WARNINGS"
}

# Verifies the merged profile's crate-hash tokens match the tokens actually baked into the
# optimised build's rlibs for `stacker_pipeline` and `stacker_algo`, i.e. the profile
# actually applies to them. Best-effort: a parsing miss prints a warning and leaves
# PROFILE_HASH_VERIFIED unset rather than failing the whole build over a diagnostic
# script's own pattern; a CONFIRMED mismatch in either crate is a hard failure.
test_profile_hash_applied() {
    local profdata="$1" bdir="$2" label="$3"
    local pipeline_rc=0 algo_rc=0
    # The merged profile's function list is large; dump it once for both crates.
    "$PROFDATA" show --all-functions "$profdata" > "$bdir/profile-functions.txt" 2>&1 || true
    test_crate_hash_token "$bdir" "$label" z-stackr-pipeline stacker_pipeline \
        align_frame_with_compute plan_tiles commit_frame apron_for || pipeline_rc=$?
    test_crate_hash_token "$bdir" "$label" z-stackr-algo stacker_algo \
        build_and_fuse_pyramids compute_focus_map fuse_strata_on apply_gaussian_blur || algo_rc=$?
    if [ "$pipeline_rc" -eq 0 ] && [ "$algo_rc" -eq 0 ]; then
        PROFILE_HASH_VERIFIED=1
    fi
}

# Compares one crate's hash token between the merged profile and the newest optimised rlib
# of that crate. Returns 0 when CONFIRMED equal, 2 when it could not be verified (a warning
# is printed), and dies on a confirmed mismatch. `pkg_dir` is the package's directory name
# under the cargo build-dir layout, `rlib_stem` the rlib's crate-name stem (underscores).
# The remaining arguments are candidate function names (public, non-generic, trained by the
# workload); the first one found in BOTH the profile and the rlib is compared.
test_crate_hash_token() {
    local bdir="$1" label="$2" pkg_dir="$3" rlib_stem="$4"
    shift 4
    local pattern="Cs[0-9a-zA-Z]+_${#rlib_stem}${rlib_stem}"

    # The compiled library is `lib<stem>-<hash>.rlib` in one of two layouts: the new cargo
    # build-dir layout (`build/<package>/<hash>/out/`, current nightly) or the classic one
    # (`deps/`). More than one may exist; take the newest across both layouts -- the
    # optimised build that just finished.
    local rel="$bdir/$HOST_TARGET/release"
    local -a rlib_candidates
    shopt -s nullglob
    rlib_candidates=("$rel/build/$pkg_dir/"*"/out/lib$rlib_stem-"*.rlib "$rel/deps/lib$rlib_stem-"*.rlib)
    shopt -u nullglob
    local rlib=""
    if [ "${#rlib_candidates[@]}" -gt 0 ]; then
        rlib="$(ls -t "${rlib_candidates[@]}" 2>/dev/null | head -n1 || true)"
    fi
    if [ -z "$rlib" ]; then
        printf 'warning: [%s] no lib%s-*.rlib found under %s/build/%s/*/out or %s/deps -- cannot verify the %s crate-hash match.\n' "$label" "$rlib_stem" "$rel" "$pkg_dir" "$rel" "$pkg_dir" >&2
        return 2
    fi
    "$LLVM_NM" "$rlib" > "$bdir/nm-$rlib_stem.txt" 2>&1 || true

    local func prof_lines prof_tokens nm_line nm_token
    for func in "$@"; do
        # Every profile line that names the function, not just the first: a merged profile
        # can hold the same symbol under more than one crate hash; the profile applies when
        # ANY copy carries the optimised rlib's token.
        prof_lines="$(grep -F -- "$func" "$bdir/profile-functions.txt" || true)"
        [ -n "$prof_lines" ] || continue
        nm_line="$(grep -m1 -F -- "$func" "$bdir/nm-$rlib_stem.txt" || true)"
        [ -n "$nm_line" ] || continue
        prof_tokens="$(printf '%s\n' "$prof_lines" | grep -oE "$pattern" | sort -u || true)"
        nm_token="$(printf '%s' "$nm_line" | grep -oE "$pattern" | head -n1 || true)"
        if [ -z "$prof_tokens" ] || [ -z "$nm_token" ]; then continue; fi
        if ! printf '%s\n' "$prof_tokens" | grep -qxF "$nm_token"; then
            die "[$label] profile did NOT apply to $pkg_dir: none of the merged profile's crate-hash tokens for '$func' ($(printf '%s' "$prof_tokens" | tr '\n' ' ')) matches the optimised build's rlib ($nm_token). The instrumented and optimised builds must use identical --target, --target-dir, package set, --features and RUSTFLAGS (apart from the profile flag); a crate-hash token mismatch means the profile did not apply."
        fi
        printf '  [%s] crate-hash check OK (%s): %s -> %s (profile holds %s token(s))\n' "$label" "$pkg_dir" "$func" "$nm_token" "$(printf '%s\n' "$prof_tokens" | grep -c .)"
        return 0
    done
    printf 'warning: [%s] none of "%s" was found with a "%s" token in both the merged profile and %s -- cannot verify the %s crate-hash match (training may not have reached it, or the symbol text differs).\n' "$label" "$*" "$pattern" "$rlib" "$pkg_dir" >&2
    return 2
}

# ---------------------------------------------------------------------------------
# BOLT: instrument the PGO CLI binary, rerun the workload, rewrite. CLI only.
# ---------------------------------------------------------------------------------
bolt_binary() {
    local name="$1" release_dir="$2" bdir="$3" gpu="$4"
    local bin=z-stackr
    local src="$release_dir/$bin"
    local work="$bdir/bolt"
    rm -rf "$work"
    mkdir -p "$work"
    local -a wl_args=()
    [ "$gpu" = gpu ] && wl_args+=(--gpu)

    step "[linux-$name][bolt] Instrumenting $bin"
    llvm-bolt "$src" -instrument --instrumentation-file="$work/prof.fdata" \
        --instrumentation-file-append-pid -o "$work/$bin.inst"

    step "[linux-$name][bolt] Running the workload with the instrumented binary"
    env -u LLVM_PROFILE_FILE \
        bash "$ROOT/scripts/pgo-workload.sh" ${wl_args[@]+"${wl_args[@]}"} "$work/$bin.inst" sample "$work/workload"

    merge-fdata "$work"/prof.fdata* > "$work/merged.fdata"

    step "[linux-$name][bolt] Rewriting binary"
    llvm-bolt "$src" -data="$work/merged.fdata" -o "$work/$bin.bolt" \
        -reorder-blocks=ext-tsp -reorder-functions=hfsort+ \
        -split-functions -split-all-cold -icf=1 -dyno-stats
    # The build kept the symbols for BOLT; strip them now, as a plain release would.
    strip --strip-all "$work/$bin.bolt"

    cp -f "$work/$bin.bolt" "$OUT_DIR/$bin-linux-$name-bolt"
    printf '    %s (BOLT, %s)\n' "$OUT_DIR/$bin-linux-$name-bolt" "$(du -h "$OUT_DIR/$bin-linux-$name-bolt" | cut -f1)"
}

# ---------------------------------------------------------------------------------
# Combination worker
# ---------------------------------------------------------------------------------
build_combination() {
    local cpu="$1" gpu="$2"
    local name="$cpu-$gpu"
    local label="$TARGET_ARG-linux-$name"

    local target_cpu
    case "$cpu" in
        avx512) target_cpu=x86-64-v4 ;;
        avx2)   target_cpu=x86-64-v3 ;;
        scalar) target_cpu=x86-64-v2 ;;
        native) target_cpu=native ;;
    esac

    local bdir="$ROOT/target/$DIR_PREFIX-$name"
    local rawdir="$bdir/profiles"
    local local_merged="$bdir/merged.profdata"
    local shared_merged
    if [ "$TARGET_ARG" = cli ]; then
        shared_merged="$PGO_PROFILE_DIR/cli/linux-$name.profdata"
    else
        shared_merged="$PGO_PROFILE_DIR/linux-$name.profdata"
    fi
    local logfile="$bdir/build.log"
    mkdir -p "$bdir"
    : > "$logfile"

    # ONE feature list and ONE package set, used verbatim by both cargo invocations below.
    local features
    features="$(pgo_features "$gpu")"
    local -a pkg_args=()
    local p
    for p in "${PKG_LIST[@]}"; do pkg_args+=(-p "$p"); done

    # The flags both builds share (only the profile flag differs). v0 symbol mangling puts a
    # per-crate hash token (`Cs<hash>_<len><name>`) into every symbol, which the crate-hash
    # check compares; naming it explicitly keeps that check independent of the toolchain's
    # default mangling. BOLT needs the relocations (--emit-relocs) and the symbols, so a
    # build BOLT will rewrite is linked unstripped (GNU ld refuses --strip-all together with
    # --emit-relocs) and the copies are stripped afterwards; the same flags go into the
    # instrumented build so the two stay identical apart from the profile flag.
    local base_flags="-C${US}target-cpu=$target_cpu${US}-C${US}symbol-mangling-version=v0"
    local -a strip_env=()
    if [ "$USE_BOLT" -eq 1 ]; then
        base_flags="$base_flags${US}-C${US}force-frame-pointers=yes${US}-C${US}link-arg=-Wl,--emit-relocs"
        strip_env=(CARGO_PROFILE_RELEASE_STRIP=none)
    fi

    local active_profile=""
    if [ "$SKIP_TRAIN" -eq 1 ] && [ -f "$shared_merged" ]; then
        step "[$label][train] Reusing existing shared profile: $shared_merged"
        active_profile="$shared_merged"
    else
        rm -rf "$rawdir"
        mkdir -p "$rawdir"

        step "[$label][train] Instrumented build (packages: ${PKG_LIST[*]}; features: $features)"
        env ${strip_env[@]+"${strip_env[@]}"} \
        CARGO_ENCODED_RUSTFLAGS="$base_flags${US}-C${US}profile-generate=$rawdir" \
        cargo build --release --target "$HOST_TARGET" --target-dir "$bdir" \
            "${pkg_args[@]}" --bins --features "$features" 2>&1 | tee -a "$logfile"

        local -a wl_args=()
        if [ "$gpu" = gpu ]; then
            wl_args+=(--gpu)
            step "[$label][train] gpu tier: the workload also runs with compute_device = gpu (no usable adapter -> the pipeline falls back to the CPU with a warning, and the GPU code stays untrained)"
        fi
        step "[$label][train] Running scripts/pgo-workload.sh"
        env LLVM_PROFILE_FILE="$rawdir/train-%p-%m.profraw" \
            bash "$ROOT/scripts/pgo-workload.sh" ${wl_args[@]+"${wl_args[@]}"} \
            "$bdir/$HOST_TARGET/release/z-stackr" sample "$bdir/workload" 2>&1 | tee -a "$logfile"

        shopt -s nullglob
        local -a prof_files=("$rawdir"/*.profraw)
        shopt -u nullglob
        [ "${#prof_files[@]}" -gt 0 ] || die "[$label][train] No .profraw files generated in $rawdir"

        step "[$label][merge] Merging ${#prof_files[@]} profile(s) with $PROFDATA"
        "$PROFDATA" merge -o "$local_merged" "${prof_files[@]}"
        [ -s "$local_merged" ] || die "[$label][merge] llvm-profdata failed or output is empty: $local_merged"

        mkdir -p "$(dirname "$shared_merged")"
        cp -f "$local_merged" "$shared_merged"
        active_profile="$shared_merged"
    fi

    # Optimised release build -- SAME packages/target/target-dir/features as the instrumented
    # build above; only the profile flag differs (profile-use instead of profile-generate,
    # plus -pgo-warn-missing-function so a totally-unapplied profile shows up as a flood of
    # warnings in $logfile instead of compiling silently as if untrained).
    step "[$label][build] PGO release build (packages: ${PKG_LIST[*]}; features: $features)"
    env ${strip_env[@]+"${strip_env[@]}"} \
    CARGO_ENCODED_RUSTFLAGS="$base_flags${US}-C${US}profile-use=$active_profile${US}-C${US}llvm-args=-pgo-warn-missing-function" \
    cargo build --release --target "$HOST_TARGET" --target-dir "$bdir" \
        "${pkg_args[@]}" --bins --features "$features" 2>&1 | tee -a "$logfile"

    # Crate-hash check FIRST: it is the real proof the profile applied, and
    # test_profile_warnings only treats the missing-function count as fatal when this could
    # not verify it.
    PROFILE_HASH_VERIFIED=0
    test_profile_hash_applied "$active_profile" "$bdir" "$label"
    test_profile_warnings "$logfile" "$label"

    local release_dir="$bdir/$HOST_TARGET/release"
    local bin src dst
    for bin in "${BIN_LIST[@]}"; do
        src="$release_dir/$bin"
        [ -f "$src" ] || die "[$label][build] Expected binary missing: $src"
        dst="$OUT_DIR/$bin-linux-$name"
        cp -f "$src" "$dst"
        # A BOLT-ready build is linked unstripped; ship the plain copies stripped.
        [ "$USE_BOLT" -eq 1 ] && strip --strip-all "$dst"
        printf '    %s (%s)\n' "$dst" "$(du -h "$dst" | cut -f1)"
    done

    # BOLT: CLI only (see the header). `--target gui` has no CLI binary to rewrite.
    if [ "$USE_BOLT" -eq 1 ]; then
        if [ "$TARGET_ARG" = gui ]; then
            step "[$label][bolt] Skipping BOLT for z-stackr-gui (interactive, no scriptable workload)"
        else
            bolt_binary "$name" "$release_dir" "$bdir" "$gpu"
            if [ "$TARGET_ARG" = all ]; then
                step "[$label][bolt] z-stackr-gui is PGO only (interactive, no scriptable workload)"
            fi
        fi
    fi
}

# ---------------------------------------------------------------------------------
# Execution entry point
# ---------------------------------------------------------------------------------
printf 'Host:         %s\n' "$HOST_TARGET"
printf 'CPU tiers:    %s\n' "${CPU_LIST[*]}"
printf 'GPU tiers:    %s\n' "${GPU_LIST[*]}"
printf 'Target:       %s (packages: %s)\n' "$TARGET_ARG" "${PKG_LIST[*]}"
printf 'BOLT:         %s\n' "$([ "$USE_BOLT" -eq 1 ] && echo 'enabled (z-stackr CLI only)' || echo 'disabled')"
printf 'Output Dir:   %s\n' "$OUT_DIR"

mkdir -p "$OUT_DIR" "$PGO_PROFILE_DIR"

for cpu in "${CPU_LIST[@]}"; do
    for gpu in "${GPU_LIST[@]}"; do
        build_combination "$cpu" "$gpu"
    done
done

printf '\nAll requested artifacts generated in %s:\n' "$OUT_DIR"
ls -lh "$OUT_DIR"
