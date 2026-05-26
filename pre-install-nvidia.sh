#!/usr/bin/env bash
# ============================================================================
# pre-install-nvidia.sh
#
#   Pre-flight validation for the NVIDIA stack install on a clean Ubuntu
#   24.04 server with 8x B300 + 4th-gen NVSwitch + ConnectX-7/8 + DOCA-OFED
#   pre-installed by the vendor. Run BEFORE install-nvidia.sh.
#
#   This script does NOT install anything. It only inspects the system and
#   reports whether install-nvidia.sh is safe to run.
#
#   Severity tiers (matches pre-install-check.sh):
#     RED    — blocks install. exit 1 if any RED fails.
#     YELLOW — warns but continues.
#     GREEN  — inventory only.
#
#   Usage:
#     sudo bash pre-install-nvidia.sh [--bundle PATH] [--ignore=R20,Y03] [--json]
#
#   Exit codes:
#     0  ready to install NVIDIA stack
#     1  one or more RED checks failed
#     2  bundle missing or invalid
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_LOG="${REPORT_LOG:-/tmp/preinstall-nvidia-report-$TS.log}"

# ── CLI ─────────────────────────────────────────────────────────────────────
BUNDLE_PATH=""
IGNORE_LIST=""
JSON_OUT=0
while (( $# > 0 )); do
    case "$1" in
        --bundle)    BUNDLE_PATH="$2"; shift 2 ;;
        --bundle=*)  BUNDLE_PATH="${1#*=}"; shift ;;
        --ignore)    IGNORE_LIST="$2"; shift 2 ;;
        --ignore=*)  IGNORE_LIST="${1#*=}"; shift ;;
        --json)      JSON_OUT=1; shift ;;
        -h|--help)
            sed -n '2,23p' "$0"; exit 0 ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

if [[ -z "$BUNDLE_PATH" ]]; then
    for cand in "$SCRIPT_DIR/nvidia-airgap-bundle-ubuntu24.04.bin" \
                "$PWD/nvidia-airgap-bundle-ubuntu24.04.bin"; do
        [[ -f "$cand" ]] && { BUNDLE_PATH="$cand"; break; }
    done
fi

# ── result tracking ─────────────────────────────────────────────────────────
RESULTS=()
RED_FAILS=0
YEL_WARNS=0
GREEN_INFO=0

c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_yel=$'\033[1;33m'
c_cyan=$'\033[1;36m'; c_mag=$'\033[1;35m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
if ! [[ -t 1 ]] || (( JSON_OUT )); then
    c_green=""; c_red=""; c_yel=""; c_cyan=""; c_mag=""; c_dim=""; c_off=""
fi

_ignored() { local id="$1"; [[ ",$IGNORE_LIST," == *",$id,"* ]]; }

_section() {
    (( JSON_OUT )) || printf '\n%s── %s ──%s\n' "$c_mag" "$*" "$c_off"
    printf '\n── %s ──\n' "$*" >> "$REPORT_LOG"
}

record() {
    local id="$1" sev="$2" name="$3" status="$4" detail="${5:-}"
    if _ignored "$id" && [[ "$status" != "PASS" && "$status" != "INFO" ]]; then
        status="SKIP"; detail="ignored via --ignore ($detail)"
    fi
    RESULTS+=( "$id|$sev|$name|$status|$detail" )
    local tag color
    case "$status" in
        PASS) tag="[ PASS ]"; color="$c_green" ;;
        FAIL) tag="[ FAIL ]"; color="$c_red"   ;;
        WARN) tag="[ WARN ]"; color="$c_yel"   ;;
        INFO) tag="[ INFO ]"; color="$c_cyan"  ;;
        SKIP) tag="[ SKIP ]"; color="$c_dim"   ;;
        *)    tag="[ ?    ]"; color=""         ;;
    esac
    case "$sev:$status" in
        R:FAIL) RED_FAILS=$((RED_FAILS+1)) ;;
        Y:WARN) YEL_WARNS=$((YEL_WARNS+1)) ;;
        G:INFO) GREEN_INFO=$((GREEN_INFO+1)) ;;
    esac
    if (( ! JSON_OUT )); then
        printf '  %s%s%s %-5s %-42s %s\n' "$color" "$tag" "$c_off" "$id" "$name" "$detail"
    fi
    printf '  %s %-5s %-42s %s\n' "$tag" "$id" "$name" "$detail" >> "$REPORT_LOG"
}
red_pass()   { record "$1" R "$2" PASS "${3:-}"; }
red_fail()   { record "$1" R "$2" FAIL "${3:-}"; }
yel_pass()   { record "$1" Y "$2" PASS "${3:-}"; }
yel_warn()   { record "$1" Y "$2" WARN "${3:-}"; }
green_info() { record "$1" G "$2" INFO "${3:-}"; }

# ── start ───────────────────────────────────────────────────────────────────
{
    echo "pre-install-nvidia.sh report"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host:    $(hostname -f 2>/dev/null || hostname)"
    echo "user:    $(id -un) (uid=$EUID)"
    echo "bundle:  ${BUNDLE_PATH:-<auto-detect failed>}"
    echo "ignored: ${IGNORE_LIST:-<none>}"
} > "$REPORT_LOG"

if (( ! JSON_OUT )); then
    printf '%s\n' "════════════════════════════════════════════════════════════════"
    printf '%s  pre-install-nvidia.sh — NVIDIA stack readiness (R580 LTS + B300)%s\n' "$c_mag" "$c_off"
    printf '%s\n'  "════════════════════════════════════════════════════════════════"
    printf '  report: %s\n' "$REPORT_LOG"
fi

# ============================================================================
# N1. OS / KERNEL
# ============================================================================
_section "N1. OS / kernel"

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]]; then
        red_pass N01 "OS Ubuntu 24.04" "$PRETTY_NAME"
    else
        red_fail N01 "OS Ubuntu 24.04" "found ${PRETTY_NAME:-?} — NVIDIA bundle is .deb-only"
    fi
else
    red_fail N01 "OS Ubuntu 24.04" "/etc/os-release missing"
fi

KERNEL_VER="$(uname -r)"
if [[ -d "/lib/modules/$KERNEL_VER" ]]; then
    red_pass N02 "kernel modules tree" "$KERNEL_VER"
else
    red_fail N02 "kernel modules tree" "/lib/modules/$KERNEL_VER missing — DKMS/peermem will fail"
fi

# N02B — pending reboot from a prior apt run. If a base-OS update (libc6 /
# systemd / kernel) was installed without rebooting first, install-nvidia.sh
# would land its driver against the OLD running kernel; on next reboot the
# new kernel boots without the matching nvidia.ko and the box comes up
# without GPUs. Canonical silent-fail seen on DGX Spark / B200 after DOCA
# DKMS upgrades — peermem reports "Invalid argument" on mlx5_core binding.
# Hard refuse; never let install-nvidia.sh run with a pending reboot.
if [[ -f /run/reboot-required || -f /var/run/reboot-required ]]; then
    pending=$( { cat /run/reboot-required.pkgs /var/run/reboot-required.pkgs 2>/dev/null | sort -u | tr '\n' ' '; } | sed 's/[[:space:]]*$//' )
    red_fail N02B "no pending reboot" "/run/reboot-required is set${pending:+ (pkgs: $pending)} — REBOOT before install-nvidia.sh"
else
    red_pass N02B "no pending reboot" "ok"
fi

# N02C — running kernel == latest installed linux-image-* package. Catches
# the case where apt upgraded linux-image-* but the box has not rebooted
# into it yet. Without this gate, install-nvidia.sh's apt install of
# nvidia-driver-580-open would pull modules for the latest installed kernel,
# not the running one, and the .ko would refuse to load on next boot.
latest_kimg=$(dpkg-query -W -f='${Package} ${Version}\n' 'linux-image-*' 2>/dev/null \
    | awk '$1 ~ /^linux-image-[0-9]/ { sub(/^linux-image-/, "", $1); print $1 }' \
    | sort -V | tail -1)
if [[ -z "$latest_kimg" ]]; then
    # No versioned linux-image-N.M.* packages found at all — exceptional,
    # but stay strict. We need a concrete kernel to compare against.
    red_fail N02C "running kernel is latest installed" "no linux-image-N.M.* packages found — cannot verify"
elif [[ "$latest_kimg" == "$KERNEL_VER" ]]; then
    red_pass N02C "running kernel is latest installed" "$KERNEL_VER"
else
    red_fail N02C "running kernel is latest installed" "running=$KERNEL_VER latest_installed=$latest_kimg — REBOOT then re-run preflight"
fi

if dpkg-query -W -f='${db:Status-Abbrev}\n' "linux-headers-$KERNEL_VER" 2>/dev/null \
        | grep -q '^ii'; then
    red_pass N03 "linux-headers installed" "linux-headers-$KERNEL_VER"
else
    red_fail N03 "linux-headers installed" "missing linux-headers-$KERNEL_VER — peermem build will fail"
fi

# ============================================================================
# N2. SECURE BOOT
# ============================================================================
_section "N2. Secure Boot / DKMS"

if command -v mokutil >/dev/null 2>&1; then
    sb_state=$(mokutil --sb-state 2>/dev/null | head -1)
    case "$sb_state" in
        *enabled*)
            # Strict: with Secure Boot enabled, nvidia-driver-*-open's prebuilt
            # .ko fails to load without MOK enrollment, which is an interactive
            # setup we cannot script. Refuse rather than risk a half-installed
            # driver state on reboot.
            red_fail N04 "Secure Boot disabled" "ENABLED — open kmod needs MOK enrollment ($sb_state). Disable in firmware before re-running." ;;
        *disabled*)
            red_pass N04 "Secure Boot disabled" "disabled — no MOK enrollment needed" ;;
        *)  red_fail N04 "Secure Boot disabled" "state unknown: $sb_state — cannot verify" ;;
    esac
else
    # Strict: without mokutil we cannot prove Secure Boot is off; refuse.
    red_fail N04 "Secure Boot disabled" "mokutil not installed — cannot determine SB state; install mokutil or boot a known-SB-off OS"
fi

# ============================================================================
# N3. GPU HARDWARE
# ============================================================================
_section "N3. GPU hardware"

if command -v lspci >/dev/null 2>&1; then
    gpu_lines=$(lspci -d 10de: -nn 2>/dev/null | grep -iE '3D controller|VGA' || true)
    gpu_count=$(printf '%s\n' "$gpu_lines" | grep -c . || true)
    if (( gpu_count > 0 )); then
        green_info N05 "NVIDIA GPUs detected" "$gpu_count device(s)"
        if (( gpu_count == 8 )); then
            red_pass N06 "GPU count == 8 (HGX/DGX B300)" "$gpu_count"
        else
            # Strict: this installer targets an 8x B300 box. A different GPU
            # count means either wrong server or PCIe enumeration failure —
            # either way the rest of the install (FM topology, NVLSM ports,
            # tensor split flags in 16-ops) is calibrated to N=8.
            red_fail N06 "GPU count == 8 (HGX/DGX B300)" "expected 8 for HGX/DGX B300, found $gpu_count"
        fi
        # Blackwell device-id heuristic (B100/B200/B300 use a range of 2BXX/2DXX ids).
        if printf '%s\n' "$gpu_lines" | grep -qiE '2[bd][0-9a-f]{2}'; then
            green_info N07 "Blackwell device-id" "matched pattern (2BXX/2DXX)"
        else
            # Strict: if the GPUs aren't Blackwell, R580+sm_103 selections are
            # wrong end-to-end (driver branch, CUDA arch list, NCCL pin).
            red_fail N07 "Blackwell device-id" "no 2BXX/2DXX device-id matched — these are NOT Blackwell GPUs; this bundle is for B300"
        fi
    else
        red_fail N05 "NVIDIA GPUs detected" "no 10de: devices found by lspci"
    fi
else
    red_fail N05 "lspci available" "install pciutils"
fi

# ============================================================================
# N4. NVSWITCH / NVLINK FABRIC
# ============================================================================
_section "N4. NVSwitch fabric"

# NVSwitch and NVLink controllers also show under 10de:; class 0680 (system
# peripheral) or 0c0X.
nvsw_count=$(lspci -d 10de: 2>/dev/null | grep -ciE 'NVSwitch|NVLink' || true)
if (( nvsw_count > 0 )); then
    green_info N08 "NVSwitch / NVLink devices" "$nvsw_count entry(ies) — fabric manager required"
else
    # Strict: HGX B300 must expose NVSwitch via lspci. Absence means either
    # the NVSwitch is firmware-broken or this is the wrong server SKU.
    # FabricManager will fail to initialize without it.
    red_fail N08 "NVSwitch / NVLink devices" "lspci shows no NVSwitch/NVLink — wrong server or NVSwitch fw issue; FabricManager will fail"
fi

# ============================================================================
# N5. DOCA-OFED (vendor pre-installed expected)
# ============================================================================
_section "N5. DOCA-OFED (vendor pre-installed)"

if command -v ofed_info >/dev/null 2>&1; then
    ofed_short=$(ofed_info -s 2>/dev/null | head -1 | tr -d ':' )
    ofed_n=$(ofed_info -n 2>/dev/null | head -1)
    if [[ -n "$ofed_short" ]]; then
        red_pass N09 "OFED installed" "$ofed_short"
        # DOCA 3.2+ ships OFED kernel 25.10+; older OFED may not have stable
        # nvidia-peermem ABI with R580.
        ofed_major=$(echo "$ofed_n" | awk -F. '{print $1}')
        if [[ -n "$ofed_major" ]] && (( ofed_major >= 25 )); then
            red_pass N10 "OFED >= 25.10 (DOCA 3.2+)" "$ofed_n"
        else
            # Strict: nvidia-peermem on R580 needs the DOCA 3.2+ mlx5_core ABI;
            # older MOFED produces the "Invalid argument" failure at module
            # bind time, which silently disables GPUDirect RDMA across the
            # fabric. Refuse rather than ship a half-working RDMA path.
            red_fail N10 "OFED >= 25.10 (DOCA 3.2+)" "found $ofed_n — peermem ABI unstable; escalate to vendor for DOCA 3.2+"
        fi
    else
        red_fail N09 "OFED installed" "ofed_info present but reports no version"
    fi
else
    red_fail N09 "OFED installed" "ofed_info missing — install DOCA-OFED first"
fi

doca_host_ver=$(dpkg-query -W -f='${Version}' doca-host 2>/dev/null || true)
if [[ -n "$doca_host_ver" ]]; then
    green_info N11 "doca-host package" "$doca_host_ver"
else
    yel_warn N11 "doca-host package" "not present — DOCA may be installed via a different metapackage"
fi

mlx5_loaded=$(lsmod 2>/dev/null | awk '$1 == "mlx5_core" {print $1}' )
if [[ -n "$mlx5_loaded" ]]; then
    green_info N12 "mlx5_core loaded" "kernel module active"
else
    # If OFED is installed (N09 passed) but mlx5_core isn't loaded, peermem
    # has nothing to bind to on the next boot — GPUDirect RDMA will silently
    # fail. Caller can bypass with --ignore=N12B if they don't need RDMA.
    if command -v ofed_info >/dev/null 2>&1; then
        red_fail N12B "mlx5_core loaded (DOCA expected)" "OFED installed but mlx5_core not loaded; ConnectX RDMA inactive and nvidia-peermem will have no NIC to bind"
    else
        yel_warn N12 "mlx5_core loaded" "module not loaded; ConnectX RDMA inactive"
    fi
fi

# nvidia-peermem requires the RDMA peer-memory API from the active ib_core.
# Ubuntu inbox RDMA can load mlx5_core successfully but still lack these
# exports, producing "Unknown symbol ib_register_peer_memory_client" later.
if grep -qw 'ib_register_peer_memory_client' /proc/kallsyms 2>/dev/null \
        && grep -qw 'ib_unregister_peer_memory_client' /proc/kallsyms 2>/dev/null; then
    red_pass N12C "RDMA peer-memory symbols" "ib_register_peer_memory_client exports present"
else
    red_fail N12C "RDMA peer-memory symbols" "missing ib_register_peer_memory_client/ib_unregister_peer_memory_client in active ib_core; nvidia-peermem will fail with Unknown symbol. Install/repair DOCA-OFED first, then install/rebuild NVIDIA."
fi

# ============================================================================
# N6. NO CONFLICTING NVIDIA INSTALL ALREADY
# ============================================================================
_section "N6. No conflicting NVIDIA install"

if command -v nvidia-smi >/dev/null 2>&1; then
    drv_ver=$(modinfo -F version nvidia 2>/dev/null || true)
    if [[ -n "$drv_ver" ]]; then
        # Strict: do not reinstall on top of an existing driver. Mixed-version
        # leftovers (different driver branch, abandoned DKMS modules, stale
        # hold marks) are a documented cause of FM "system not initialized"
        # after reboot. Operator must purge before re-running.
        red_fail N13 "no existing NVIDIA driver" "found loaded driver $drv_ver — purge with: apt-get purge 'nvidia-*' 'cuda-*' 'libnvidia-*'; apt-get autoremove; then reboot before re-running"
    else
        # nvidia-smi binary exists but no kmod loaded — almost always means
        # a prior install was partial (driver pkg installed, reboot skipped).
        # Refuse: clean it up first.
        red_fail N13 "no existing NVIDIA driver" "nvidia-smi present but kmod not loaded — leftover from a partial prior install; purge first"
    fi
else
    red_pass N13 "no existing NVIDIA driver" "nvidia-smi absent (clean slate)"
fi

# nouveau must not be loaded at install time, or driver install will fail.
NOUVEAU_LOADED=0
if lsmod 2>/dev/null | awk '$1 == "nouveau" {found=1} END {exit !found}'; then
    NOUVEAU_LOADED=1
    red_fail N14 "nouveau not loaded" "nouveau is loaded; run install-nvidia.sh once to stage blacklist, reboot, then rerun"
else
    red_pass N14 "nouveau not loaded" "ok"
fi

# Stale NVIDIA debs left from a previous attempt?
stale_nv=$(dpkg -l 2>/dev/null \
    | awk '$1 == "ii" && ($2 ~ /^nvidia-/ || $2 ~ /^cuda-/ || $2 ~ /^libnvidia/) {print $2}' \
    | grep -vE '^doca-|^libnvhws|^dpa-|^flexio-|^ibarr|^mft|^libnvidia-utils-' || true)
if [[ -n "$stale_nv" ]]; then
    # Strict: any installed nvidia-/cuda-/libnvidia- package (outside the
    # DOCA-shipped allowlist) implies an incomplete prior install. Refuse
    # rather than let apt's dependency resolver pull mismatched versions.
    red_fail N15 "no stale NVIDIA packages" "found: $(echo "$stale_nv" | tr '\n' ' ') — purge before re-running"
else
    red_pass N15 "no stale NVIDIA packages" "clean (DOCA-shipped libnvhws/mft/flexio ignored)"
fi

# ============================================================================
# N7. DISK SPACE
# ============================================================================
_section "N7. Disk space"

free_var=$(df -BG /var 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
free_root=$(df -BG / 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
if [[ -n "$free_var" ]] && (( free_var >= 10 )); then
    red_pass N16 "/var has >= 10G free" "${free_var}G"
else
    red_fail N16 "/var has >= 10G free" "${free_var:-?}G — driver+CUDA+NCCL needs ~5G installed, +buffer"
fi
green_info N17 "/ free space" "${free_root:-?}G"

# ============================================================================
# N8. APT / REPOS
# ============================================================================
_section "N8. apt / repos"

if command -v apt-get >/dev/null 2>&1; then
    red_pass N18 "apt available" "$(apt-get --version 2>&1 | head -1)"
else
    red_fail N18 "apt available" "apt-get missing — Debian-derived OS required"
fi

# Ensure no existing nvidia/cuda apt repo that would conflict with our file://
existing_nv_lists=$(ls /etc/apt/sources.list.d/ 2>/dev/null | grep -iE 'nvidia|cuda' || true)
if [[ -n "$existing_nv_lists" ]]; then
    # Strict: a leftover nvidia/cuda sources.list.d entry on an airgapped box
    # is a dpkg foot-gun — apt may try to reach an unreachable host and slow
    # every install, OR (worse) silently pull mismatched versions. Refuse.
    red_fail N19 "no existing NVIDIA apt repos" "$(echo "$existing_nv_lists" | tr '\n' ' ') — remove these before re-running"
else
    red_pass N19 "no existing NVIDIA apt repos" "none — clean"
fi

# ============================================================================
# N9. BUNDLE — presence + outer SHA256 + inner contract
# ============================================================================
_section "N9. Bundle file"

if [[ -z "$BUNDLE_PATH" ]]; then
    red_fail N20 "bundle file present" "set --bundle PATH or place next to script"
elif [[ ! -f "$BUNDLE_PATH" ]]; then
    red_fail N20 "bundle file present" "$BUNDLE_PATH not found"
else
    bundle_size=$(du -h "$BUNDLE_PATH" | cut -f1)
    red_pass N20 "bundle file present" "$BUNDLE_PATH ($bundle_size)"
    if [[ -f "${BUNDLE_PATH}.sha256" ]]; then
        if ( cd "$(dirname "$BUNDLE_PATH")" && sha256sum -c --status "$(basename "$BUNDLE_PATH").sha256" ); then
            red_pass N21 "bundle sha256 matches" "ok"
        else
            red_fail N21 "bundle sha256 matches" "MISMATCH — re-transfer"
        fi
    else
        # Strict: a missing .sha256 sidecar means we cannot prove the bundle
        # arrived intact. Bricks are cheaper to avoid than to debug.
        red_fail N21 "bundle sha256 matches" "no .sha256 sidecar at ${BUNDLE_PATH}.sha256 — re-transfer bundle with its sidecar"
    fi

    # ── Inner bundle contract (peek inside the tarball — no extraction) ────
    # Mirrors what pre-install-check.sh does for the userland bundle. Catches
    # mis-built bundles (wrong OS, wrong variant, missing helpers, missing
    # Packages index) BEFORE install-nvidia.sh extracts.
    target_env=""
    if target_env=$(tar -xzOf "$BUNDLE_PATH" --wildcards '*meta/target.env' 2>/dev/null | head -200); then
        # N22 — variant guard.
        if printf '%s\n' "$target_env" | grep -q '^BUNDLE_VARIANT=nvidia-stack$'; then
            red_pass N22 "BUNDLE_VARIANT=nvidia-stack" "ok"
        else
            variant=$(printf '%s\n' "$target_env" | grep '^BUNDLE_VARIANT=' | cut -d= -f2)
            red_fail N22 "BUNDLE_VARIANT=nvidia-stack" "found: '${variant:-<unset>}' — wrong bundle (the userland 'prepped' bundle goes through pre-install-check.sh, not this preflight)"
        fi

        # N23 — OS version matches target (we already know N01 said target is 24.04).
        bundle_os=$(printf '%s\n' "$target_env" | grep '^BUNDLE_OS_VERSION=' | cut -d= -f2)
        bundle_target_os=$(printf '%s\n' "$target_env" | grep '^BUNDLE_TARGET_OS=' | cut -d= -f2)
        target_os="${VERSION_ID:-?}"
        if [[ "$bundle_target_os" == "$target_os" || "$bundle_os" == "$target_os" ]]; then
            red_pass N23 "bundle target OS matches" "${bundle_target_os:-$bundle_os} == $target_os"
        else
            red_fail N23 "bundle target OS matches" "bundle says target=${bundle_target_os:-?}, gather host=${bundle_os:-?}, but this server is $target_os — .deb compat will likely break"
        fi

        # N24 — bundle arch matches target.
        bundle_arch=$(printf '%s\n' "$target_env" | grep '^BUNDLE_ARCH=' | cut -d= -f2)
        target_arch=$(dpkg --print-architecture 2>/dev/null || echo unknown)
        if [[ "$bundle_arch" == "$target_arch" ]]; then
            red_pass N24 "bundle arch matches" "$bundle_arch"
        else
            red_fail N24 "bundle arch matches" "bundle=$bundle_arch target=$target_arch"
        fi

        # N25 — CUDA major.minor recorded (sanity check; the actual nvcc check
        # happens in pre-install-check.sh after install).
        bundle_cuda=$(printf '%s\n' "$target_env" | grep '^BUNDLE_CUDA=' | cut -d= -f2)
        if [[ "$bundle_cuda" == "13.0" ]]; then
            green_info N25 "bundle CUDA = 13.0" "$bundle_cuda"
        elif [[ -n "$bundle_cuda" ]]; then
            # Strict: the cu130 PyTorch/PyG/vLLM/NCCL pin chain is calibrated
            # to CUDA 13.0 specifically. A different toolkit means wheelhouse
            # / NCCL ABI mismatches everywhere.
            red_fail N25 "bundle CUDA = 13.0" "bundle has CUDA=$bundle_cuda (expected 13.0)"
        else
            red_fail N25 "bundle CUDA = 13.0" "BUNDLE_CUDA not recorded in meta/target.env — rebuild bundle with current gather-nvidia.sh"
        fi

        # Inventory: driver, FM, NCCL versions recorded by gather time.
        for key in BUNDLE_DRIVER_VERSION BUNDLE_FABRICMANAGER_VERSION BUNDLE_NCCL_VERSION BUNDLE_DATE BUNDLE_GATHER_HOST; do
            line=$(printf '%s\n' "$target_env" | grep "^${key}=" | head -1)
            [[ -n "$line" ]] && green_info "G-${key}" "$key" "${line#*=}"
        done
    else
        red_fail N22 "BUNDLE_VARIANT=nvidia-stack" "could not read meta/target.env from bundle — corrupt or wrong format"
    fi

    # N26 — debs/Packages index present in the tarball.
    if tar -tzf "$BUNDLE_PATH" --wildcards '*debs/Packages' >/dev/null 2>&1; then
        red_pass N26 "debs/Packages index present" "ok"
    else
        red_fail N26 "debs/Packages index present" "missing — install-nvidia.sh will need dpkg-scanpackages on the target"
    fi

    # N27 — helper scripts bundled inside.
    helpers_missing=()
    for helper in install-nvidia.sh pre-install-nvidia.sh test-nvidia.sh; do
        if ! tar -tzf "$BUNDLE_PATH" --wildcards "*${helper}" >/dev/null 2>&1; then
            helpers_missing+=( "$helper" )
        fi
    done
    if (( ${#helpers_missing[@]} == 0 )); then
        red_pass N27 "helper scripts bundled" "install-nvidia.sh + pre-install-nvidia.sh + test-nvidia.sh"
    else
        red_fail N27 "helper scripts bundled" "missing: ${helpers_missing[*]}"
    fi

    # N28 — inner SHA256SUMS exists. Actual hash verification happens at
    # install time after extraction (see install-nvidia.sh step 1b).
    if tar -tzf "$BUNDLE_PATH" --wildcards '*meta/SHA256SUMS' >/dev/null 2>&1; then
        red_pass N28 "meta/SHA256SUMS present" "ok (install-nvidia.sh will verify after extract)"
    else
        # Strict: without inner SHA256SUMS, install-nvidia.sh's per-file
        # integrity check is bypassed. On an airgapped install, dropped /
        # corrupted .debs surface as cryptic apt failures during driver
        # install — we want them caught NOW.
        red_fail N28 "meta/SHA256SUMS present" "missing — rebuild bundle with current gather-nvidia.sh (writes meta/SHA256SUMS post-gather)"
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================
TOTAL_CHECKS=${#RESULTS[@]}

if (( JSON_OUT )); then
    printf '{\n  "report": "%s",\n  "checks": [\n' "$REPORT_LOG"
    first=1
    for r in "${RESULTS[@]}"; do
        IFS='|' read -r id sev name status detail <<<"$r"
        (( first )) || printf ',\n'; first=0
        printf '    {"id": "%s", "sev": "%s", "name": "%s", "status": "%s", "detail": "%s"}' \
            "$id" "$sev" "${name//\"/\\\"}" "$status" "${detail//\"/\\\"}"
    done
    printf '\n  ],\n  "red_fails": %d, "yel_warns": %d, "green_info": %d, "total": %d\n}\n' \
        "$RED_FAILS" "$YEL_WARNS" "$GREEN_INFO" "$TOTAL_CHECKS"
else
    printf '\n%s════════════════════════════════════════════════════════════════%s\n' "$c_mag" "$c_off"
    printf '  SUMMARY  '
    printf '%s%d PASS-RED%s  ' "$c_green" "$((TOTAL_CHECKS - RED_FAILS - YEL_WARNS - GREEN_INFO))" "$c_off"
    printf '%s%d FAIL%s  '     "$c_red"   "$RED_FAILS"  "$c_off"
    printf '%s%d WARN%s  '     "$c_yel"   "$YEL_WARNS"  "$c_off"
    printf '%s%d INFO%s\n'     "$c_cyan"  "$GREEN_INFO" "$c_off"
    printf '%s════════════════════════════════════════════════════════════════%s\n' "$c_mag" "$c_off"
    printf '  Report: %s\n\n' "$REPORT_LOG"
fi

if (( RED_FAILS > 0 )); then
    (( JSON_OUT )) || printf '%s%d RED check(s) FAILED. Fix before running install-nvidia.sh.%s\n' "$c_red" "$RED_FAILS" "$c_off"
    if (( ! JSON_OUT && NOUVEAU_LOADED )); then
        printf '%sN14 recovery:%s sudo bash install-nvidia.sh  # stages blacklist, then stops for reboot\n' "$c_yel" "$c_off"
    fi
    exit 1
fi
(( JSON_OUT )) || printf '%sReady. Run: sudo bash install-nvidia.sh%s\n' "$c_green" "$c_off"
exit 0
