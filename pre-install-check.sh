#!/usr/bin/env bash
# ============================================================================
# pre-install-check.sh
#   Pre-flight validation for the airgapped Ubuntu 24.04 userland install on a
#   server where install-nvidia.sh has already installed R580 LTS driver +
#   CUDA 13.0 + FabricManager + NVLSM (and the box has been rebooted). Run
#   AFTER test-nvidia.sh passes, BEFORE install-all.sh.
#
#   Severity tiers (kubeadm pattern):
#     RED    — blocks install. exit 1 if any RED fails.
#     YELLOW — warns but continues. installer still runs.
#     GREEN  — inventory recorded to the report log only.
#
#   Usage:
#     sudo bash pre-install-check.sh [--bundle PATH] [--ignore=R20,Y03] [--json]
#
#   Exit codes:
#     0  ready to install
#     1  one or more RED checks failed
#     2  bundle missing or invalid
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_LOG="${REPORT_LOG:-/tmp/preinstall-report-$TS.log}"

# ── CLI parsing ─────────────────────────────────────────────────────────────
BUNDLE_PATH=""
IGNORE_LIST=""
JSON_OUT=0
FORCE=0
while (( $# > 0 )); do
    case "$1" in
        --bundle)    BUNDLE_PATH="$2"; shift 2 ;;
        --bundle=*)  BUNDLE_PATH="${1#*=}"; shift ;;
        --ignore)    IGNORE_LIST="$2"; shift 2 ;;
        --ignore=*)  IGNORE_LIST="${1#*=}"; shift ;;
        --json)      JSON_OUT=1; shift ;;
        --force)     FORCE=1; shift ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0 ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Auto-detect bundle if not specified
if [[ -z "$BUNDLE_PATH" ]]; then
    for cand in "$SCRIPT_DIR/all-airgap-bundle-ubuntu24.04.bin" \
                "$PWD/all-airgap-bundle-ubuntu24.04.bin"; do
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

# Strip color when output isn't a TTY (e.g. piped to file)
if ! [[ -t 1 ]] || (( JSON_OUT )); then
    c_green=""; c_red=""; c_yel=""; c_cyan=""; c_mag=""; c_dim=""; c_off=""
fi

_ignored() {
    local id="$1"
    [[ ",$IGNORE_LIST," == *",$id,"* ]]
}

_section() {
    (( JSON_OUT )) || printf '\n%s── %s ──%s\n' "$c_mag" "$*" "$c_off"
    printf '\n── %s ──\n' "$*" >> "$REPORT_LOG"
}

# record <id> <severity:R|Y|G> <name> <status:PASS|FAIL|WARN|INFO> <detail>
record() {
    local id="$1" sev="$2" name="$3" status="$4" detail="${5:-}"
    if _ignored "$id" && [[ "$status" != "PASS" && "$status" != "INFO" ]]; then
        status="SKIP"
        detail="ignored via --ignore  ($detail)"
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
        printf '  %s%s%s %-4s %-42s %s\n' "$color" "$tag" "$c_off" "$id" "$name" "$detail"
    fi
    printf '  %s %-4s %-42s %s\n' "$tag" "$id" "$name" "$detail" >> "$REPORT_LOG"
}

# Convenience helpers
red_pass()  { record "$1" R "$2" PASS "${3:-}"; }
red_fail()  { record "$1" R "$2" FAIL "${3:-}"; }
yel_pass()  { record "$1" Y "$2" PASS "${3:-}"; }
yel_warn()  { record "$1" Y "$2" WARN "${3:-}"; }
green_info(){ record "$1" G "$2" INFO "${3:-}"; }

# parse driver version like 595.58.03 -> integer for comparison
ver_to_int() {
    local v="$1"
    local maj min pat
    IFS='.' read -r maj min pat <<<"$v"
    printf '%d' $(( 10#${maj:-0}*1000000 + 10#${min:-0}*1000 + 10#${pat:-0} ))
}

# ── start ───────────────────────────────────────────────────────────────────
{
    echo "pre-install-check.sh report"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host:    $(hostname -f 2>/dev/null || hostname)"
    echo "user:    $(id -un) (uid=$EUID)"
    echo "bundle:  ${BUNDLE_PATH:-<auto-detect failed>}"
    echo "ignored: ${IGNORE_LIST:-<none>}"
} > "$REPORT_LOG"

if (( ! JSON_OUT )); then
    printf '%s\n' "════════════════════════════════════════════════════════════════"
    printf '%s  pre-install-check.sh — airgap GPU server readiness%s\n' "$c_mag" "$c_off"
    printf '%s\n'  "════════════════════════════════════════════════════════════════"
    printf '  report: %s\n' "$REPORT_LOG"
fi

# ============================================================================
# 1. OS / KERNEL / ARCH
# ============================================================================
_section "1. OS / kernel / architecture"

# R01 Ubuntu 24.04
if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]]; then
        red_pass R01 "Ubuntu 24.04 (noble)" "$PRETTY_NAME"
    else
        red_fail R01 "Ubuntu 24.04 (noble)" "found: ${ID:-?} ${VERSION_ID:-?}"
    fi
else
    red_fail R01 "Ubuntu 24.04 (noble)" "/etc/os-release missing"
fi

# R02 x86_64
arch=$(uname -m)
if [[ "$arch" == "x86_64" ]]; then
    red_pass R02 "x86_64 architecture" "$arch"
else
    red_fail R02 "x86_64 architecture" "found: $arch"
fi

# R03 Kernel >= 6.8
kver=$(uname -r)
kmaj=$(printf '%s' "$kver" | awk -F. '{print $1}')
kmin=$(printf '%s' "$kver" | awk -F. '{print $2}')
if (( 10#${kmaj:-0} > 6 )) || ( (( 10#${kmaj:-0} == 6 )) && (( 10#${kmin:-0} >= 8 )) ); then
    red_pass R03 "Kernel >= 6.8" "$kver"
else
    red_fail R03 "Kernel >= 6.8" "$kver (R580+ requires 6.8+)"
fi

# R04 glibc >= 2.28 (PyTorch manylinux_2_28)
glibc_ver=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [[ -n "$glibc_ver" ]]; then
    # `sort -V -C` returns 0 iff the input is already version-sorted ascending.
    # Feed "2.28\n<found>\n" so a pass means 2.28 <= found.
    if printf '2.28\n%s\n' "$glibc_ver" | sort -V -C; then
        red_pass R04 "glibc >= 2.28" "$glibc_ver"
    else
        red_fail R04 "glibc >= 2.28" "$glibc_ver (PyTorch wheels require 2.28+)"
    fi
else
    red_fail R04 "glibc >= 2.28" "could not detect glibc"
fi

# R05 systemd >= 252
if command -v systemctl >/dev/null 2>&1; then
    sd_ver=$(systemctl --version 2>/dev/null | head -1 | awk '{print $2}')
    if [[ "$sd_ver" =~ ^[0-9]+$ ]] && (( sd_ver >= 252 )); then
        red_pass R05 "systemd >= 252" "$sd_ver"
    else
        red_fail R05 "systemd >= 252" "found: ${sd_ver:-?}"
    fi
else
    red_fail R05 "systemd >= 252" "systemctl not available"
fi

# R17 /run/reboot-required — system needs a reboot from a pending update.
# (Pull this up early — if set, nothing else matters.)
if [[ -f /run/reboot-required ]]; then
    detail="reboot required"
    [[ -f /run/reboot-required.pkgs ]] && detail+=" by: $(tr '\n' ' ' </run/reboot-required.pkgs)"
    red_fail R17 "no /run/reboot-required" "$detail — reboot first, then re-run"
else
    red_pass R17 "no /run/reboot-required" "system has no pending reboot"
fi

# ============================================================================
# 2. NVIDIA STACK (installed by install-nvidia.sh)
# ============================================================================
_section "2. NVIDIA driver & CUDA toolkit"

# R06 nvidia-smi runs
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        red_pass R06 "nvidia-smi runs" "OK"
    else
        red_fail R06 "nvidia-smi runs" "nvidia-smi present but failing — driver kmod not loaded, reboot?"
    fi
else
    red_fail R06 "nvidia-smi runs" "command not found — install-nvidia.sh not run yet?"
fi

# R07 Driver version: R580 LTS baseline (580.159.04 at time of writing).
# R580 LTS branch is supported until 2028-06. Older 580.x is acceptable as
# long as it meets the CUDA 13.0 minimum.
DRIVER_VER=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
fi
if [[ -n "$DRIVER_VER" ]]; then
    drv_int=$(ver_to_int "$DRIVER_VER")
    min_int=$(ver_to_int "580.65.06")        # CUDA 13.0 minimum
    base_int=$(ver_to_int "580.159.04")      # R580 LTS baseline
    if (( drv_int < min_int )); then
        red_fail R07 "driver >= 580.65.06 (CUDA 13.0 min)" "$DRIVER_VER too old"
    elif (( drv_int < base_int )); then
        red_pass R07 "driver >= 580.65.06 (CUDA 13.0 min)" "$DRIVER_VER (OK; R580 LTS baseline is 580.159.04)"
    else
        red_pass R07 "driver >= 580.65.06 (CUDA 13.0 min)" "$DRIVER_VER (R580 LTS baseline)"
    fi
else
    red_fail R07 "driver >= 580.65.06 (CUDA 13.0 min)" "could not query driver version"
fi

# R08 8 GPUs detected
GPU_COUNT=0
if nvidia-smi -L >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
fi
if (( GPU_COUNT == 8 )); then
    red_pass R08 "8 GPUs detected" "$GPU_COUNT"
elif (( GPU_COUNT > 0 )); then
    red_fail R08 "8 GPUs detected" "found $GPU_COUNT (expected 8 for full B300 HGX)"
else
    red_fail R08 "8 GPUs detected" "nvidia-smi -L returned nothing"
fi

# R09 All 8 GPUs are B300 (288 GB VRAM)
if (( GPU_COUNT > 0 )); then
    gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u)
    name_count=$(printf '%s\n' "$gpu_names" | grep -c .)
    if (( name_count == 1 )) && printf '%s' "$gpu_names" | grep -qi 'B300'; then
        red_pass R09 "all GPUs are B300" "$(printf '%s' "$gpu_names" | tr -d '\n')"
    elif (( name_count == 1 )); then
        red_fail R09 "all GPUs are B300" "$(printf '%s' "$gpu_names" | tr -d '\n') (expected B300)"
    else
        red_fail R09 "all GPUs are B300" "mixed SKUs: $(printf '%s' "$gpu_names" | tr '\n' ',' | sed 's/,$//')"
    fi
    # VRAM total per-GPU sanity (B300 = 288 GB; older B300 SKU could be 192 GB)
    mem_min=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk '{print int($1/1024)}' | sort -n | head -1)
    if [[ -n "$mem_min" ]] && (( mem_min >= 280 )); then
        yel_pass Y20 "B300 VRAM = 288 GB per GPU" "min observed: ${mem_min} GB"
    elif [[ -n "$mem_min" ]] && (( mem_min >= 180 )); then
        yel_warn Y20 "B300 VRAM = 288 GB per GPU" "${mem_min} GB observed — looks like 192 GB SKU"
    elif [[ -n "$mem_min" ]]; then
        yel_warn Y20 "B300 VRAM = 288 GB per GPU" "${mem_min} GB observed — below 192 GB SKU"
    fi
else
    red_fail R09 "all GPUs are B300" "no GPUs to check"
fi

# R10 nvcc 13.0.x — install-nvidia.sh installs nvcc at /usr/local/cuda/bin/nvcc but
# does NOT add it to a global PATH (no /etc/profile.d snippet). So a user running
# this script without sudo, or with a minimal PATH, won't see nvcc via `command -v`.
# Fall back to the standard CUDA install paths, matching test-nvidia.sh:217-220.
NVCC_BIN=""
NVCC_VER=""
for c in nvcc /usr/local/cuda/bin/nvcc "/usr/local/cuda-13.0/bin/nvcc"; do
    if command -v "$c" >/dev/null 2>&1; then NVCC_BIN=$(command -v "$c"); break; fi
    [[ -x "$c" ]] && { NVCC_BIN="$c"; break; }
done
if [[ -n "$NVCC_BIN" ]]; then
    NVCC_VER=$("$NVCC_BIN" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
    if [[ "$NVCC_VER" == 13.0* ]]; then
        red_pass R10 "nvcc on PATH, 13.0.x" "$NVCC_VER ($NVCC_BIN)"
    else
        red_fail R10 "nvcc on PATH, 13.0.x" "found: ${NVCC_VER:-?} at $NVCC_BIN (need 13.0)"
    fi
else
    red_fail R10 "nvcc on PATH, 13.0.x" "nvcc not found on PATH or under /usr/local/cuda* — install-nvidia.sh did not install cuda-nvcc-13-0 (or /etc/profile.d/cuda.sh missing)?"
fi

# R12 cuBLAS / cuDNN headers (need to BUILD extensions; runtime is shipped by wheels)
HDR_OK=0
for h in /usr/local/cuda/include/cublas_v2.h /usr/include/cublas_v2.h; do
    [[ -f "$h" ]] && { HDR_OK=1; break; }
done
if (( HDR_OK )); then
    red_pass R12 "cuBLAS headers present" "$h"
else
    red_fail R12 "cuBLAS headers present" "cublas_v2.h not found — libcublas-dev-13-0 / cuda-cudart-dev-13-0 missing?"
fi

# R13 CUDA smoke test: compile + run a 10-line vector-add against sm_103
if [[ -n "$NVCC_BIN" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    smoke_dir=$(mktemp -d)
    cat > "$smoke_dir/vadd.cu" <<'CUDA_EOF'
#include <cstdio>
__global__ void add(int *a) { a[threadIdx.x] += threadIdx.x; }
int main() {
    int *d, h[32];
    for (int i = 0; i < 32; ++i) h[i] = i;
    cudaMalloc(&d, sizeof(h));
    cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice);
    add<<<1, 32>>>(d);
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    cudaFree(d);
    int err = (h[31] == 62) ? 0 : 1;
    printf(err ? "FAIL h[31]=%d\n" : "OK h[31]=%d\n", h[31]);
    return err;
}
CUDA_EOF
    # Try sm_103 first (B300), fall back to sm_100 (B200/B300 PTX-compat)
    smoke_out=""
    smoke_rc=1
    for arch_flag in "-arch=sm_103" "-arch=sm_100"; do
        if (cd "$smoke_dir" && "$NVCC_BIN" $arch_flag -o vadd vadd.cu) >/dev/null 2>&1; then
            smoke_out=$("$smoke_dir/vadd" 2>&1); smoke_rc=$?
            if (( smoke_rc == 0 )); then
                red_pass R13 "CUDA smoke test (vector-add)" "$arch_flag :: $smoke_out"
                break
            fi
        fi
    done
    if (( smoke_rc != 0 )); then
        red_fail R13 "CUDA smoke test (vector-add)" "compile or run failed: ${smoke_out:-no output}"
    fi
    rm -rf "$smoke_dir"
else
    red_fail R13 "CUDA smoke test (vector-add)" "nvcc or nvidia-smi missing — covered by R06/R10"
fi

# R14 nvidia-fabricmanager.service active
if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
    fm_state=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo unknown)
    if [[ "$fm_state" == "active" ]]; then
        red_pass R14 "nvidia-fabricmanager.service" "active"
    else
        red_fail R14 "nvidia-fabricmanager.service" "$fm_state — multi-GPU fabric may be degraded"
    fi
else
    red_fail R14 "nvidia-fabricmanager.service" "unit not registered — install-nvidia.sh did not install nvidia-fabricmanager-580?"
fi

# R15 nvidia-nvlsm.service active (B300 NVSwitch requirement)
if systemctl list-unit-files nvidia-nvlsm.service >/dev/null 2>&1; then
    nvlsm_state=$(systemctl is-active nvidia-nvlsm 2>/dev/null || echo unknown)
    if [[ "$nvlsm_state" == "active" ]]; then
        red_pass R15 "nvidia-nvlsm.service" "active"
    else
        red_fail R15 "nvidia-nvlsm.service" "$nvlsm_state — required for B300 NVSwitch"
    fi
else
    # install-nvidia.sh: the nvidia-fabricmanager unit spawns the NVLSM daemon
    # as a child process — no separate nvidia-nvlsm.service unit by default.
    yel_warn Y19 "nvidia-nvlsm.service" "unit not registered (folded into nvidia-fabricmanager — verify nvlsm pid in test-nvidia.sh)"
    red_pass R15 "nvidia-nvlsm.service" "not separately registered; covered by FabricManager"
fi

# R16 Fabric State = Completed for all GPUs
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    fab_states=$(nvidia-smi -q 2>/dev/null | awk '
        /^[[:space:]]+(GPU[[:space:]]+)?Fabric[[:space:]]*$/ {
            in_fabric = 1; captured = 0; next
        }
        in_fabric && !captured && /^[[:space:]]+State[[:space:]]*:/ {
            v = $0; sub(/.*:[[:space:]]*/, "", v); sub(/[[:space:]]+$/, "", v)
            print v
            captured = 1; in_fabric = 0; next
        }
        in_fabric && /^[[:space:]]+[A-Z][A-Za-z0-9 ]*[[:space:]]*$/ {
            in_fabric = 0
        }
        /^[[:space:]]+Fabric[[:space:]]+State[[:space:]]*:/ {
            v = $0; sub(/.*:[[:space:]]*/, "", v); sub(/[[:space:]]+$/, "", v)
            print v
        }
    ')
    fab_total=$(printf '%s\n' "$fab_states" | grep -c . || true)
    fab_ok=$(printf '%s\n' "$fab_states" | grep -cE '^(Completed|Success)$' || true)
    if (( fab_total == 0 )); then
        red_fail R16 "fabric.state = Completed" "Fabric State not reported; expected Completed on 8x B300"
    elif (( GPU_COUNT > 0 && fab_total != GPU_COUNT )); then
        red_fail R16 "fabric.state = Completed" "matched $fab_total Fabric entries for $GPU_COUNT GPU(s) ($fab_ok Completed/Success)"
    elif (( fab_ok == fab_total )); then
        red_pass R16 "fabric.state = Completed" "$fab_ok/$fab_total Completed/Success"
    else
        red_fail R16 "fabric.state = Completed" "$fab_ok/$fab_total Completed/Success - common cause of CUDA Error 802"
    fi
else
    red_fail R16 "fabric.state = Completed" "nvidia-smi missing"
fi

# ============================================================================
# 3. AUTH / DISK
# ============================================================================
_section "3. Authentication & disk space"

# R18 sudo
if (( EUID == 0 )); then
    red_pass R18 "sudo available" "running as root"
elif sudo -n true 2>/dev/null; then
    red_pass R18 "sudo available" "sudo -n works"
else
    red_fail R18 "sudo available" "no cached sudo; run 'sudo -v' first or run this script via sudo"
fi

# R19 /scratch ≥ 60 GB free
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
if [[ -d "$SCRATCH_ROOT" ]]; then
    if [[ -w "$SCRATCH_ROOT" ]] || (( EUID == 0 )); then
        free_gb=$(df -BG --output=avail "$SCRATCH_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
        if [[ "$free_gb" =~ ^[0-9]+$ ]] && (( free_gb >= 60 )); then
            red_pass R19 "$SCRATCH_ROOT >= 60 GB free" "${free_gb} GB"
        else
            red_fail R19 "$SCRATCH_ROOT >= 60 GB free" "${free_gb:-?} GB (need 60+ for 3 venvs + llama.cpp)"
        fi
    else
        red_fail R19 "$SCRATCH_ROOT writable" "exists but not writable by $(id -un); run installer with sudo"
    fi
else
    red_fail R19 "$SCRATCH_ROOT exists" "directory does not exist — installer will need to mkdir; ensure mount is configured"
fi

# R20 /var ≥ 5 GB
free_gb=$(df -BG --output=avail /var 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ "$free_gb" =~ ^[0-9]+$ ]] && (( free_gb >= 5 )); then
    red_pass R20 "/var >= 5 GB free" "${free_gb} GB"
else
    red_fail R20 "/var >= 5 GB free" "${free_gb:-?} GB (apt cache)"
fi

# R21 /tmp ≥ 12 GB
free_gb=$(df -BG --output=avail /tmp 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ "$free_gb" =~ ^[0-9]+$ ]] && (( free_gb >= 12 )); then
    red_pass R21 "/tmp >= 12 GB free" "${free_gb} GB"
else
    red_fail R21 "/tmp >= 12 GB free" "${free_gb:-?} GB (bundle extraction)"
fi

# ============================================================================
# 4. BUNDLE
# ============================================================================
_section "4. Bundle integrity"

BUNDLE_VARIANT_OK=0
BUNDLE_DIRS_OK=0

if [[ -z "$BUNDLE_PATH" ]]; then
    red_fail R22 "bundle present" "no bundle specified or found — pass --bundle PATH"
elif [[ -d "$BUNDLE_PATH" ]]; then
    # ── Bundle is an already-extracted directory (e.g. install-all.sh resume) ──
    red_pass R22 "bundle present" "$BUNDLE_PATH (extracted directory)"

    yel_warn Y18 "bundle SHA256" "skipped — bundle is an extracted directory, not a .bin file"

    if [[ -f "$BUNDLE_PATH/meta/target.env" ]]; then
        target_env=$(cat "$BUNDLE_PATH/meta/target.env")
        if printf '%s' "$target_env" | grep -q '^BUNDLE_VARIANT=prepped$'; then
            red_pass R22-variant "BUNDLE_VARIANT=prepped" "ok"
            BUNDLE_VARIANT_OK=1
        else
            variant=$(printf '%s' "$target_env" | grep '^BUNDLE_VARIANT=' | cut -d= -f2)
            red_fail R22-variant "BUNDLE_VARIANT=prepped" "found: '${variant:-<unset>}' — wrong bundle (bare-metal bundles refuse to install here)"
        fi
        for key in BUNDLE_OS_VERSION BUNDLE_PYTHON BUNDLE_DATE BUNDLE_LLAMA_REF BUNDLE_DRIVER_BASELINE; do
            line=$(printf '%s' "$target_env" | grep "^${key}=" | head -1)
            [[ -n "$line" ]] && green_info "G-${key}" "$key" "${line#*=}"
        done
    else
        red_fail R22-variant "BUNDLE_VARIANT=prepped" "$BUNDLE_PATH/meta/target.env missing"
    fi

    missing_dirs=()
    for sub in debs wheels/inference wheels/training wheels/jupyter wheels/llamacpp apps src; do
        [[ -d "$BUNDLE_PATH/$sub" ]] || missing_dirs+=( "$sub/" )
    done
    if (( ${#missing_dirs[@]} == 0 )); then
        red_pass R23 "bundle subdirs present" "debs/ wheels/* apps/ src/"
        BUNDLE_DIRS_OK=1
    else
        red_fail R23 "bundle subdirs present" "missing: ${missing_dirs[*]}"
    fi

    if [[ -f "$BUNDLE_PATH/debs/Packages" && -f "$BUNDLE_PATH/meta/apt-packages.txt" ]]; then
        red_pass R23-index "bundle apt indexes present" "debs/Packages + meta/apt-packages.txt"
    else
        red_fail R23-index "bundle apt indexes present" "missing debs/Packages or meta/apt-packages.txt"
    fi
elif [[ -f "$BUNDLE_PATH" ]]; then
    # ── Bundle is a .bin file (initial gather → install path) ──
    bundle_size=$(du -BM "$BUNDLE_PATH" 2>/dev/null | cut -f1)
    red_pass R22 "bundle present" "$BUNDLE_PATH ($bundle_size)"

    # SHA256 check (sidecar file)
    if [[ -f "${BUNDLE_PATH}.sha256" ]]; then
        expected=$(awk '{print $1}' "${BUNDLE_PATH}.sha256")
        actual=$(sha256sum "$BUNDLE_PATH" | awk '{print $1}')
        if [[ "$expected" == "$actual" ]]; then
            yel_pass Y18 "bundle SHA256 matches sidecar" "${expected:0:16}…"
        else
            yel_warn Y18 "bundle SHA256 matches sidecar" "MISMATCH expected=${expected:0:16}… actual=${actual:0:16}…"
        fi
    else
        yel_warn Y18 "bundle SHA256 matches sidecar" "no .sha256 sidecar at ${BUNDLE_PATH}.sha256"
    fi

    # Peek the bundle's target.env without full extraction (it's a gzipped tarball)
    target_env=""
    if target_env=$(tar -xzOf "$BUNDLE_PATH" --wildcards '*meta/target.env' 2>/dev/null | head -200); then
        if printf '%s' "$target_env" | grep -q '^BUNDLE_VARIANT=prepped$'; then
            red_pass R22-variant "BUNDLE_VARIANT=prepped" "ok"
            BUNDLE_VARIANT_OK=1
        else
            variant=$(printf '%s' "$target_env" | grep '^BUNDLE_VARIANT=' | cut -d= -f2)
            red_fail R22-variant "BUNDLE_VARIANT=prepped" "found: '${variant:-<unset>}' — wrong bundle (bare-metal bundles refuse to install here)"
        fi
        # Record gather metadata as GREEN
        for key in BUNDLE_OS_VERSION BUNDLE_PYTHON BUNDLE_DATE BUNDLE_LLAMA_REF BUNDLE_DRIVER_BASELINE; do
            line=$(printf '%s' "$target_env" | grep "^${key}=" | head -1)
            [[ -n "$line" ]] && green_info "G-${key}" "$key" "${line#*=}"
        done
    else
        red_fail R22-variant "BUNDLE_VARIANT=prepped" "could not read meta/target.env from bundle"
    fi

    # Required subdirs present in the archive.
    # The previous implementation read the first 2000 entries of `tar -tzf`,
    # but a 12+ GB bundle that packs debs/ first has tens of thousands of
    # entries before wheels/* / apps/ / src/ appear — every later subdir
    # was reported missing. Single-pass scan: awk records which markers
    # have been seen and exits as soon as all 7 are present, sending SIGPIPE
    # back to tar so the gzip stream isn't read further than necessary.
    flags=$(tar -tzf "$BUNDLE_PATH" 2>/dev/null | awk '
        {
            if (index($0, "debs/"))             d=1
            if (index($0, "wheels/inference/")) wi=1
            if (index($0, "wheels/training/"))  wt=1
            if (index($0, "wheels/jupyter/"))   wj=1
            if (index($0, "wheels/llamacpp/"))  wl=1
            if (index($0, "apps/"))             a=1
            if (index($0, "src/"))              s=1
            if (d && wi && wt && wj && wl && a && s) exit
        }
        END { printf "%d %d %d %d %d %d %d\n", d+0, wi+0, wt+0, wj+0, wl+0, a+0, s+0 }
    ')
    read -r f_d f_wi f_wt f_wj f_wl f_a f_s <<<"$flags"
    missing_dirs=()
    (( f_d ))  || missing_dirs+=( "debs/" )
    (( f_wi )) || missing_dirs+=( "wheels/inference/" )
    (( f_wt )) || missing_dirs+=( "wheels/training/" )
    (( f_wj )) || missing_dirs+=( "wheels/jupyter/" )
    (( f_wl )) || missing_dirs+=( "wheels/llamacpp/" )
    (( f_a ))  || missing_dirs+=( "apps/" )
    (( f_s ))  || missing_dirs+=( "src/" )
    if (( ${#missing_dirs[@]} == 0 )); then
        red_pass R23 "bundle subdirs present" "debs/ wheels/* apps/ src/"
        BUNDLE_DIRS_OK=1
    else
        red_fail R23 "bundle subdirs present" "missing: ${missing_dirs[*]}"
    fi

    if tar -tzf "$BUNDLE_PATH" --wildcards '*debs/Packages' '*meta/apt-packages.txt' >/dev/null 2>&1; then
        red_pass R23-index "bundle apt indexes present" "debs/Packages + meta/apt-packages.txt"
    else
        red_fail R23-index "bundle apt indexes present" "missing debs/Packages or meta/apt-packages.txt"
    fi
else
    red_fail R22 "bundle present" "$BUNDLE_PATH is neither a file nor a directory"
fi

# R23-baseos — STRICT base-OS version gate.
#
# Refuse if the userland bundle's debs/Packages index advertises a libc6 /
# systemd / dbus / kernel / firmware / microcode version newer than what's
# installed on this target. Touching any of those AFTER install-nvidia.sh
# is the canonical brick path on B300 (peermem ABI break, FM "system not
# initialized", unbootable kernel without nvidia.ko).
#
# Mirrors the runtime check in install-all.d/04-apt-plan.sh, but surfaces
# the failure HERE so the operator sees it before install-all.sh even runs.
BASE_OS_DANGER='libc6 libc6-dev systemd systemd-sysv dbus dbus-daemon linux-firmware microcode intel-microcode amd64-microcode'
KERNEL_GLOB_PREFIXES='linux-image- linux-headers-'

# Read debs/Packages from either the extracted dir or the .bin tarball.
PKG_INDEX_CONTENT=""
if [[ -d "$BUNDLE_PATH" && -f "$BUNDLE_PATH/debs/Packages" ]]; then
    PKG_INDEX_CONTENT=$(cat "$BUNDLE_PATH/debs/Packages" 2>/dev/null)
elif [[ -f "$BUNDLE_PATH" ]]; then
    PKG_INDEX_CONTENT=$(tar -xzOf "$BUNDLE_PATH" --wildcards '*debs/Packages' 2>/dev/null)
fi

if [[ -z "$PKG_INDEX_CONTENT" ]]; then
    # We already failed R23-index above if Packages is missing; surface this
    # as a separate RED so the base-OS check isn't silently skipped.
    red_fail R23-baseos "bundle does not upgrade base OS" "debs/Packages unreadable — cannot prove no base-OS upgrade pending"
else
    # Build "pkg version" pairs from the apt Packages index.
    bundle_pairs=$(printf '%s\n' "$PKG_INDEX_CONTENT" \
        | awk '
            /^Package: / { pkg=$2; next }
            /^Version: / && pkg != "" { print pkg, $2; pkg="" }
        ')

    danger_hits=""
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        bp=${line%% *}; bv=${line##* }

        # Either an explicit base-OS name OR matches a kernel/header prefix.
        is_danger=0
        for d in $BASE_OS_DANGER; do
            [[ "$bp" == "$d" ]] && { is_danger=1; break; }
        done
        if (( ! is_danger )); then
            for p in $KERNEL_GLOB_PREFIXES; do
                [[ "$bp" == ${p}* ]] && { is_danger=1; break; }
            done
        fi
        (( is_danger )) || continue

        # Compare with installed version (if any). If not installed at all,
        # apt may pull it as a NEW dep — also a base-OS change, so flag it.
        installed_ver=$(dpkg-query -W -f='${Version}' "$bp" 2>/dev/null || true)
        if [[ -z "$installed_ver" ]]; then
            danger_hits+=$'\n'"  $bp: bundle=$bv installed=<none, would be NEW install>"
        else
            # sort -V -C succeeds if input is in ascending order.
            # "installed\nbundle\n" sorted means installed <= bundle.
            if printf '%s\n%s\n' "$installed_ver" "$bv" | sort -V -C 2>/dev/null; then
                if [[ "$installed_ver" != "$bv" ]]; then
                    danger_hits+=$'\n'"  $bp: bundle=$bv > installed=$installed_ver"
                fi
            fi
        fi
    done <<<"$bundle_pairs"

    if [[ -z "$danger_hits" ]]; then
        red_pass R23-baseos "bundle does not upgrade base OS" "libc6/systemd/dbus/kernel/firmware at or above bundle versions"
    else
        red_fail R23-baseos "bundle does not upgrade base OS" "would upgrade:$danger_hits — escalate to vendor for baseline match, then re-run from pre-install-nvidia.sh"
    fi
fi

# R24 No conflicting existing installs
existing=()
for prefix in "/scratch/llm_inference" "/scratch/general_training" "/scratch/jupyter" "/scratch/llama.cpp"; do
    [[ -e "$prefix" ]] && existing+=( "$prefix" )
done
if (( ${#existing[@]} == 0 )); then
    red_pass R24 "no existing install at /scratch" "clean"
elif (( FORCE )); then
    yel_warn Y17 "no existing install at /scratch" "existing dirs (--force will overwrite): ${existing[*]}"
    red_pass R24 "no existing install at /scratch" "existing dirs found but --force given"
else
    red_fail R24 "no existing install at /scratch" "would clobber: ${existing[*]}  (pass --force to allow)"
fi

# ============================================================================
# 5. YELLOW — environmental warnings
# ============================================================================
_section "5. Environmental warnings"

# Y01 ECC mode
if nvidia-smi -L >/dev/null 2>&1; then
    ecc=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | sort -u | xargs)
    case "$ecc" in
        "Enabled")   yel_pass Y01 "ECC mode" "Enabled (all GPUs)" ;;
        "Disabled")  yel_warn Y01 "ECC mode" "Disabled — fine for inference; ops decision" ;;
        *)           yel_warn Y01 "ECC mode" "mixed/unknown: $ecc" ;;
    esac
fi

# Y02 Persistence Mode
if nvidia-smi -L >/dev/null 2>&1; then
    pm_off=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | grep -c -i 'disabled' || true)
    if (( pm_off == 0 )); then
        yel_pass Y02 "Persistence Mode on all GPUs" "all Enabled"
    else
        yel_warn Y02 "Persistence Mode on all GPUs" "$pm_off GPU(s) Disabled — run: sudo nvidia-smi -pm 1"
    fi
fi

# Y03 NVLink lanes
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nv_status=$(nvidia-smi nvlink --status 2>/dev/null || true)
    if [[ -n "$nv_status" ]]; then
        nv_down=$(printf '%s\n' "$nv_status" | grep -ciE 'inactive|<inactive>|disabled' || true)
        nv_up=$(printf '%s\n' "$nv_status" | grep -c 'GB/s' || true)
        if (( nv_down == 0 )) && (( nv_up > 0 )); then
            yel_pass Y03 "NVLink lanes all active" "$nv_up active"
        else
            yel_warn Y03 "NVLink lanes all active" "$nv_down inactive (fabric degraded)"
        fi
    fi
fi

# Y04 topology NV18 mesh
if nvidia-smi -L >/dev/null 2>&1; then
    topo=$(nvidia-smi topo -m 2>/dev/null || true)
    if printf '%s' "$topo" | grep -q 'NV18'; then
        yel_pass Y04 "NV18 mesh (NVSwitch)" "present"
    elif [[ -n "$topo" ]]; then
        # B300 sometimes shows NV9/NV6 — still NVSwitch, just different counts
        yel_warn Y04 "NV18 mesh (NVSwitch)" "unexpected topology — review nvidia-smi topo -m"
    fi
fi

# Y05 internet connectivity — should be absent on airgap (but be quick about it)
if timeout 3 curl -fsS -o /dev/null https://download.pytorch.org/ 2>/dev/null; then
    yel_warn Y05 "no internet (airgap)" "reachable — not actually airgapped?"
else
    yel_pass Y05 "no internet (airgap)" "no route to PyTorch CDN (as expected)"
fi

# Y06 no proxy env
if [[ -n "${http_proxy:-}${HTTP_PROXY:-}${https_proxy:-}${HTTPS_PROXY:-}" ]] || \
   grep -RIq -e 'Proxy' /etc/apt/apt.conf.d/ 2>/dev/null; then
    yel_warn Y06 "no proxy configured" "proxy detected — could divert apt to internet"
else
    yel_pass Y06 "no proxy configured" "clean"
fi

# Y07 NVIDIA apt source presence (informational)
if [[ -f /etc/apt/sources.list.d/00-nvidia-bundle.list ]] \
   && grep -qs 'file:///var/tmp/airgap-nvidia-debs' /etc/apt/sources.list.d/00-nvidia-bundle.list; then
    yel_pass Y07 "NVIDIA bundle apt source present" "file:///var/tmp/airgap-nvidia-debs"
elif compgen -G '/etc/apt/sources.list.d/cuda*.list' >/dev/null \
   || grep -Rqs 'developer.download.nvidia.com' /etc/apt/sources.list.d/ 2>/dev/null; then
    yel_warn Y07 "NVIDIA online apt source present" "target should use install-nvidia.sh's file:// bundle source"
else
    yel_warn Y07 "NVIDIA bundle apt source present" "not found - install-nvidia.sh registers /etc/apt/sources.list.d/00-nvidia-bundle.list"
fi

# Y08 hostname resolves
hn=$(hostname 2>/dev/null)
if [[ -n "$hn" ]] && getent hosts "$hn" >/dev/null 2>&1; then
    yel_pass Y08 "hostname resolves" "$hn"
else
    yel_warn Y08 "hostname resolves" "$hn not in /etc/hosts — sshd login can be slow"
fi

# Y09 not running as root
if (( EUID == 0 )) && [[ "${SUDO_USER:-}" != "root" && -n "${SUDO_USER:-}" ]]; then
    yel_pass Y09 "running via sudo from real user" "user=$SUDO_USER"
elif (( EUID == 0 )); then
    yel_warn Y09 "running via sudo from real user" "EUID=0 with no SUDO_USER — \$HOME may be /root"
else
    yel_pass Y09 "running via sudo from real user" "user=$(id -un)"
fi

# Y10 ulimit -n
nofile=$(ulimit -n 2>/dev/null || echo 0)
if (( nofile >= 1048576 )); then
    yel_pass Y10 "ulimit -n >= 1048576" "$nofile"
else
    yel_warn Y10 "ulimit -n >= 1048576" "$nofile — installer will set /etc/security/limits.d/"
fi

# Y11 ulimit -l unlimited
memlock=$(ulimit -l 2>/dev/null || echo 0)
if [[ "$memlock" == "unlimited" ]] || ( [[ "$memlock" =~ ^[0-9]+$ ]] && (( memlock >= 1048576 )) ); then
    yel_pass Y11 "ulimit -l unlimited" "$memlock"
else
    yel_warn Y11 "ulimit -l unlimited" "$memlock — installer will set limits.d"
fi

# Y12 THP = madvise
if [[ -r /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    thp=$(awk '{for(i=1;i<=NF;i++) if($i ~ /\[.*\]/) print $i}' /sys/kernel/mm/transparent_hugepage/enabled)
    if [[ "$thp" == "[madvise]" ]]; then
        yel_pass Y12 "THP = madvise" "[madvise]"
    else
        yel_warn Y12 "THP = madvise" "current: $thp"
    fi
fi

# Y13 Secure Boot
if command -v mokutil >/dev/null 2>&1; then
    sb=$(mokutil --sb-state 2>/dev/null | head -1)
    case "$sb" in
        *disabled*) yel_pass Y13 "Secure Boot status" "disabled" ;;
        *enabled*)  yel_warn Y13 "Secure Boot status" "enabled — open kmod needs MOK enrollment (install-nvidia.sh does not configure this)" ;;
        *)          yel_warn Y13 "Secure Boot status" "${sb:-unknown}" ;;
    esac
fi

# Y14 NUMA / GPU topology
if command -v numactl >/dev/null 2>&1; then
    numa_nodes=$(numactl --hardware 2>/dev/null | awk '/available:/ {print $2}')
    yel_pass Y14 "NUMA topology" "${numa_nodes:-?} node(s)"
fi

# Y15 NTP
if systemctl is-active --quiet chrony 2>/dev/null \
    || systemctl is-active --quiet systemd-timesyncd 2>/dev/null \
    || systemctl is-active --quiet ntp 2>/dev/null; then
    yel_pass Y15 "time sync active" "ntp/chrony/timesyncd running"
else
    yel_warn Y15 "time sync active" "no time daemon running — wall clock may drift"
fi

# Y16 writability of system dirs (only meaningful when root)
unwritable=()
for d in /opt /etc/profile.d /usr/local/bin; do
    if [[ ! -w "$d" ]] && (( EUID != 0 )); then unwritable+=( "$d" ); fi
done
if (( ${#unwritable[@]} == 0 )); then
    yel_pass Y16 "system dirs writable" "/opt /etc/profile.d /usr/local/bin"
else
    yel_warn Y16 "system dirs writable" "not writable: ${unwritable[*]} (installer runs with sudo)"
fi

# ============================================================================
# 6. GREEN — inventory (recorded to log only, not stdout)
# ============================================================================
_section "6. Inventory (recorded to report log)"

{
    echo "── lscpu ──"
    lscpu 2>/dev/null | head -25
    echo
    echo "── memory ──"
    free -h 2>/dev/null
    echo
    echo "── block devices ──"
    lsblk 2>/dev/null | head -30
    echo
    echo "── kernel cmdline ──"
    cat /proc/cmdline 2>/dev/null
    echo
    echo "── GPU inventory ──"
    nvidia-smi --query-gpu=index,name,serial,uuid,vbios_version,driver_version,memory.total --format=csv 2>/dev/null
    echo
    echo "── fabricmanager / nvcc versions ──"
    dpkg-query -W -f='${Package}\t${Version}\n' 'nvidia-fabricmanager*' 'nvidia-nvlsm*' 2>/dev/null | head -20
    nvcc --version 2>/dev/null
} >> "$REPORT_LOG" 2>&1

green_info G01 "GPU inventory" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) x${GPU_COUNT}"
green_info G02 "Driver version" "${DRIVER_VER:-?}"
green_info G03 "CUDA toolkit" "${NVCC_VER:-?}"
green_info G04 "Kernel" "$kver"
green_info G05 "glibc" "${glibc_ver:-?}"

# ============================================================================
# SUMMARY
# ============================================================================
if (( JSON_OUT )); then
    printf '{\n  "started":"%s","host":"%s","report":"%s",\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname -f 2>/dev/null || hostname)" "$REPORT_LOG"
    printf '  "red_fails":%d,"yellow_warns":%d,"green_info":%d,\n' \
        "$RED_FAILS" "$YEL_WARNS" "$GREEN_INFO"
    printf '  "results":[\n'
    first=1
    for row in "${RESULTS[@]}"; do
        IFS='|' read -r id sev name status detail <<< "$row"
        detail=${detail//\\/\\\\}; detail=${detail//\"/\\\"}
        name=${name//\\/\\\\};     name=${name//\"/\\\"}
        (( first )) || printf ',\n'
        printf '    {"id":"%s","sev":"%s","name":"%s","status":"%s","detail":"%s"}' \
            "$id" "$sev" "$name" "$status" "$detail"
        first=0
    done
    printf '\n  ]\n}\n'
else
    printf '\n%s════════════════════════════════════════════════════════════════%s\n' "$c_mag" "$c_off"
    printf '%s  SUMMARY%s\n' "$c_mag" "$c_off"
    printf '%s════════════════════════════════════════════════════════════════%s\n' "$c_mag" "$c_off"
    printf '  %sRED  fails  %s : %d\n' "$c_red"   "$c_off" "$RED_FAILS"
    printf '  %sYELLOW warns%s : %d\n' "$c_yel"   "$c_off" "$YEL_WARNS"
    printf '  %sGREEN info  %s : %d\n' "$c_green" "$c_off" "$GREEN_INFO"
    printf '  report: %s\n\n' "$REPORT_LOG"

    if (( RED_FAILS > 0 )); then
        printf '%sFAILED RED CHECKS:%s\n' "$c_red" "$c_off"
        for row in "${RESULTS[@]}"; do
            IFS='|' read -r id sev name status detail <<< "$row"
            if [[ "$sev" == "R" && "$status" == "FAIL" ]]; then
                printf '  %-4s %-42s %s\n' "$id" "$name" "$detail"
            fi
        done
        printf '\n%sNot ready to install. Resolve the items above and re-run.%s\n' "$c_red" "$c_off"
    else
        if (( YEL_WARNS > 0 )); then
            printf '%sReady to install (with %d warning(s) — review the report).%s\n' "$c_yel" "$YEL_WARNS" "$c_off"
        else
            printf '%sReady to install.%s\n' "$c_green" "$c_off"
        fi
    fi
fi

# Exit codes per the contract documented at top.
#   0 = ready to install
#   1 = one or more RED checks failed
#   2 = bundle missing or invalid (so callers can offer to re-stage)
if [[ -z "$BUNDLE_PATH" ]] && (( ! FORCE )); then
    exit 2
fi
if [[ -n "$BUNDLE_PATH" && ! -f "$BUNDLE_PATH" && ! -d "$BUNDLE_PATH" ]] && (( ! FORCE )); then
    exit 2
fi
if (( BUNDLE_VARIANT_OK == 0 || BUNDLE_DIRS_OK == 0 )) && (( ! FORCE )); then
    [[ -n "$BUNDLE_PATH" ]] && exit 2
fi
(( RED_FAILS == 0 ))
