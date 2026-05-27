#!/usr/bin/env bash
# ============================================================================
# test-nvidia.sh
#
#   Post-install verification for install-nvidia.sh.
#   Run AFTER reboot, BEFORE pre-install-check.sh + install-all.sh.
#
#   Verifies:
#     - nvidia.ko loaded, driver version matches expected branch
#     - 8x GPUs visible, Blackwell device-id, ECC + persistence states
#     - nvidia-fabricmanager active, NVLSM daemon up
#     - All GPUs report Fabric State = Completed
#     - CUDA minimal toolkit installed (nvcc/cudart/cccl/cublas/nvjitlink),
#       /etc/profile.d/cuda.sh + /etc/ld.so.conf.d/cuda-system.conf present,
#       nvcc reports CUDA 13.x
#     - libnccl2 installed with +cuda13.0 suffix (or SKIPPED - default)
#     - DCGM status if installed (optional)
#     - nvidia-peermem module loaded (multi-node RDMA)
#
#   Usage:
#     bash test-nvidia.sh                    # human-readable
#     bash test-nvidia.sh --json             # machine-readable
#
#   Exit code: 0 if every required check passed, 1 otherwise.
# ============================================================================
set -uo pipefail

DRIVER_BRANCH="${DRIVER_BRANCH:-580}"
CUDA_MAJOR="${CUDA_MAJOR:-13}"
CUDA_MINOR="${CUDA_MINOR:-0}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
SKIP_DCGM="${SKIP_DCGM:-0}"
# Mirrors gather-nvidia.sh's NCCL_MIN_VER — when libnccl2 is installed on the
# host (SKIP_NCCL=0 path), refuse < 2.27.7. 2.26.x and 2.27.0-2.27.6 deadlock
# at AllReduce on TP>1 with Blackwell Ultra (NCCL release notes for 2.27.7).
NCCL_MIN_VER="${NCCL_MIN_VER:-2.27.7}"

JSON_OUT=0
while (( $# > 0 )); do
    case "$1" in
        --json)    JSON_OUT=1; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

RESULTS=()
PASS_COUNT=0
FAIL_COUNT=0
MISS_COUNT=0
SKIP_COUNT=0

c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_yel=$'\033[1;33m'
c_cyan=$'\033[1;36m'; c_mag=$'\033[1;35m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
(( JSON_OUT )) && { c_green=""; c_red=""; c_yel=""; c_cyan=""; c_mag=""; c_dim=""; c_off=""; }

step() { (( JSON_OUT )) || printf '\n%s== %s ==%s\n' "$c_mag" "$*" "$c_off"; }
record() {
    local name="$1" status="$2" detail="${3:-}"
    RESULTS+=( "$name|$status|$detail" )
    case "$status" in
        PASS)    PASS_COUNT=$((PASS_COUNT+1)); (( JSON_OUT )) || printf '  %s[ PASS ]%s %-40s %s\n' "$c_green" "$c_off" "$name" "$detail" ;;
        FAIL)    FAIL_COUNT=$((FAIL_COUNT+1)); (( JSON_OUT )) || printf '  %s[ FAIL ]%s %-40s %s\n' "$c_red"   "$c_off" "$name" "$detail" ;;
        MISSING) MISS_COUNT=$((MISS_COUNT+1)); (( JSON_OUT )) || printf '  %s[MISSING]%s %-40s %s\n' "$c_yel"   "$c_off" "$name" "$detail" ;;
        SKIP)    SKIP_COUNT=$((SKIP_COUNT+1)); (( JSON_OUT )) || printf '  %s[ SKIP ]%s %-40s %s\n' "$c_dim"   "$c_off" "$name" "$detail" ;;
    esac
}

check_dpkg() {
    local pkg="$1" state
    state=$(dpkg-query -W -f='${db:Status-Abbrev}|${Version}\n' "$pkg" 2>/dev/null || true)
    if [[ -z "$state" ]]; then
        record "dpkg: $pkg" MISSING "not installed"; return
    fi
    local abbrev="${state%%|*}" ver="${state##*|}"
    # Accept both "ii" (install/installed) and "hi" (hold/installed) — install-nvidia.sh
    # deliberately `apt-mark hold`s every NVIDIA package, so `hi` is the expected end state.
    local trimmed="${abbrev%% *}"
    case "$trimmed" in
        ii|hi) record "dpkg: $pkg" PASS "$ver" ;;
        *)     record "dpkg: $pkg" FAIL "state=$abbrev ver=$ver" ;;
    esac
}

check_dpkg_optional() {
    local pkg="$1" state
    state=$(dpkg-query -W -f='${db:Status-Abbrev}|${Version}\n' "$pkg" 2>/dev/null || true)
    if [[ -z "$state" ]]; then
        record "dpkg: $pkg" SKIP "optional package not installed"; return
    fi
    local abbrev="${state%%|*}" ver="${state##*|}"
    local trimmed="${abbrev%% *}"
    case "$trimmed" in
        ii|hi) record "dpkg: $pkg" PASS "$ver" ;;
        *)     record "dpkg: $pkg" FAIL "state=$abbrev ver=$ver" ;;
    esac
}

check_service() {
    local svc="$1" required="${2:-1}"
    if ! command -v systemctl >/dev/null 2>&1; then
        record "service: $svc" SKIP "systemctl not available"; return
    fi
    if ! systemctl cat "$svc" >/dev/null 2>&1; then
        if (( required )); then
            record "service: $svc" MISSING "unit not registered"
        else
            record "service: $svc" SKIP "optional unit not present"
        fi
        return
    fi
    local active enabled
    active=$(systemctl is-active "$svc" 2>/dev/null || true)
    enabled=$(systemctl is-enabled "$svc" 2>/dev/null || true)
    if [[ "$active" == "active" ]]; then
        record "service: $svc" PASS "active, $enabled"
    else
        if (( required )); then
            record "service: $svc" FAIL "state=$active enabled=$enabled"
        else
            record "service: $svc" SKIP "state=$active (non-fatal)"
        fi
    fi
}

# ============================================================================
# 1. Driver kmod + nvidia-smi
# ============================================================================
step "1. Driver kernel module"

if command -v nvidia-smi >/dev/null 2>&1; then
    record "binary: nvidia-smi" PASS "$(command -v nvidia-smi)"
else
    record "binary: nvidia-smi" MISSING "not on PATH"
fi

KMOD_VER=$(modinfo -F version nvidia 2>/dev/null || true)
if [[ -n "$KMOD_VER" ]]; then
    if [[ "$KMOD_VER" == ${DRIVER_BRANCH}.* ]]; then
        record "kmod: nvidia version" PASS "$KMOD_VER (matches R${DRIVER_BRANCH})"
    else
        record "kmod: nvidia version" FAIL "$KMOD_VER (expected R${DRIVER_BRANCH}.x.y)"
    fi
else
    record "kmod: nvidia version" FAIL "modinfo nvidia returned nothing — kmod not loaded"
fi

if lsmod 2>/dev/null | awk '$1 == "nvidia" {found=1} END {exit !found}'; then
    record "kmod: nvidia loaded" PASS "in lsmod"
else
    record "kmod: nvidia loaded" FAIL "not in lsmod — driver not loaded, reboot?"
fi

# Confirm the loaded nvidia.ko actually lives under /lib/modules/$(uname -r).
# Stale kmod can persist in memory if the kernel was upgraded after the
# driver was installed — modinfo will still report a version, but the .ko
# file on disk is in a different /lib/modules/<kver> tree and on next boot
# the driver won't load. Catches the kernel-bumped-after-driver-install case.
RUNNING_KERNEL_FOR_TEST="$(uname -r)"
NV_KO_PATH=$(modinfo -F filename nvidia 2>/dev/null || true)
if [[ -z "$NV_KO_PATH" ]]; then
    record "kmod: nvidia .ko path" FAIL "modinfo -F filename nvidia returned nothing"
elif [[ "$NV_KO_PATH" == "/lib/modules/${RUNNING_KERNEL_FOR_TEST}"* ]]; then
    record "kmod: nvidia .ko path" PASS "$NV_KO_PATH"
else
    record "kmod: nvidia .ko path" FAIL "$NV_KO_PATH not under /lib/modules/${RUNNING_KERNEL_FOR_TEST} — kernel was upgraded after driver install; reboot will not load nvidia"
fi

# ============================================================================
# 2. GPU enumeration
# ============================================================================
step "2. GPU enumeration"

if command -v nvidia-smi >/dev/null 2>&1; then
    smi_out=$(nvidia-smi -L 2>&1) || true
    gpu_count=$(printf '%s\n' "$smi_out" | grep -c '^GPU ' || true)
    if (( gpu_count == EXPECTED_GPUS )); then
        record "GPU count" PASS "$gpu_count (expected $EXPECTED_GPUS)"
    elif (( gpu_count > 0 )); then
        record "GPU count" FAIL "$gpu_count (expected $EXPECTED_GPUS)"
    else
        record "GPU count" FAIL "nvidia-smi -L returned no GPUs"
    fi

    if printf '%s\n' "$smi_out" | grep -qiE 'B300|B200|Blackwell'; then
        record "GPU model" PASS "$(printf '%s\n' "$smi_out" | head -1 | cut -d: -f2- | xargs)"
    else
        record "GPU model" FAIL "no Blackwell match in nvidia-smi -L output"
    fi
else
    record "GPU count" SKIP "nvidia-smi missing"
    record "GPU model" SKIP "nvidia-smi missing"
fi

# ============================================================================
# 3. Fabric Manager + NVLSM + fabric state
# ============================================================================
step "3. NVSwitch fabric"

check_service "nvidia-fabricmanager" 1
# R580 Fabric Manager on NVL5+ starts NVLSM through
# nvidia-fabricmanager-start.sh. A legacy custom nvidia-nvlsm.service from an
# older installer revision races that wrapper, so the authoritative check is
# the nvlsm process below, not a standalone systemd unit.
if systemctl cat nvidia-nvlsm.service >/dev/null 2>&1; then
    if [[ -f /etc/systemd/system/nvidia-nvlsm.service ]] \
            && grep -q 'Installed by install-nvidia.sh' /etc/systemd/system/nvidia-nvlsm.service; then
        record "service: nvidia-nvlsm" FAIL "legacy custom unit present — remove it and let nvidia-fabricmanager own NVLSM"
    else
        check_service "nvidia-nvlsm" 0
    fi
else
    record "service: nvidia-nvlsm" SKIP "Fabric Manager wrapper owns NVLSM on this build"
fi
check_service "nvidia-persistenced" 1

if (( EXPECTED_GPUS > 1 )) && [[ "${ALLOW_NO_FABRIC:-0}" != "1" ]]; then
    if lsmod 2>/dev/null | awk '$1 == "ib_umad" {f=1} END {exit !f}'; then
        record "kmod: ib_umad" PASS "loaded"
    else
        record "kmod: ib_umad" FAIL "not loaded — NVLSM/OpenSM cannot use UMAD"
    fi
    IB_UMAD_CONF=/etc/modules-load.d/ib-umad.conf
    if [[ -r "$IB_UMAD_CONF" ]] && grep -qE '^[[:space:]]*ib_umad[[:space:]]*$' "$IB_UMAD_CONF"; then
        record "config: ib_umad autoload" PASS "$IB_UMAD_CONF"
    else
        record "config: ib_umad autoload" FAIL "$IB_UMAD_CONF missing or empty — Fabric Manager may fail after reboot"
    fi
fi

if command -v nv-fabricmanager >/dev/null 2>&1; then
    fm_ver=$(nv-fabricmanager -v 2>&1 | head -1)
    record "fm: nv-fabricmanager -v" PASS "$fm_ver"
else
    record "fm: nv-fabricmanager -v" MISSING "binary not on PATH"
fi

# Check that every GPU reports Fabric State = Completed.
# nvidia-smi -q emits one of two layouts depending on driver version:
#   (a) "Fabric" section header line, then an indented "State : Completed" subline
#   (b) "Fabric State : Completed" on a single line
# It may ALSO emit a "Fabric Manager <something>" / nested "State" line per GPU,
# which the old `/Fabric/ ... /State/` awk caught greedily and double-counted.
# Match only the real fabric.state by anchoring on either layout.
if command -v nvidia-smi >/dev/null 2>&1; then
    fab_q=$(nvidia-smi -q 2>/dev/null | awk '
        # (a) Section header: indented "Fabric" (optionally "GPU Fabric"), no colon
        /^[[:space:]]+(GPU[[:space:]]+)?Fabric[[:space:]]*$/ {
            in_fabric = 1; captured = 0; next
        }
        # First "State : value" line inside that section
        in_fabric && !captured && /^[[:space:]]+State[[:space:]]*:/ {
            v = $0; sub(/.*:[[:space:]]*/, "", v); sub(/[[:space:]]+$/, "", v)
            print v
            captured = 1; in_fabric = 0; next
        }
        # New section header (no colon) ends the Fabric block without a match
        in_fabric && /^[[:space:]]+[A-Z][A-Za-z0-9 ]*[[:space:]]*$/ {
            in_fabric = 0
        }
        # (b) Single-line "Fabric State : value"
        /^[[:space:]]+Fabric[[:space:]]+State[[:space:]]*:/ {
            v = $0; sub(/.*:[[:space:]]*/, "", v); sub(/[[:space:]]+$/, "", v)
            print v
        }
    ')
    total=$(printf '%s\n' "$fab_q" | grep -c . || true)
    ok=$(printf '%s\n' "$fab_q" | grep -c '^Completed$' || true)
    if (( total == 0 )); then
        # On a single-GPU dev box (EXPECTED_GPUS=1) the Fabric stanza is
        # legitimately absent — no NVSwitch, no FabricManager work to do.
        # On a multi-GPU NVSwitch box (B300 HGX, EXPECTED_GPUS>=2) the
        # absence means FabricManager didn't initialize: that's a hard
        # failure, not a "feature unavailable". Override the FAIL only with
        # an explicit ALLOW_NO_FABRIC=1 for unusual recovery scenarios.
        if (( EXPECTED_GPUS <= 1 )) || [[ "${ALLOW_NO_FABRIC:-0}" == "1" ]]; then
            record "fabric state (per GPU)" SKIP "no Fabric stanza in nvidia-smi -q (single-GPU or ALLOW_NO_FABRIC=1)"
        else
            record "fabric state (per GPU)" FAIL "no Fabric stanza on a $EXPECTED_GPUS-GPU box — FabricManager did not initialize NVSwitch. Set ALLOW_NO_FABRIC=1 only for explicit recovery."
        fi
    elif (( total != EXPECTED_GPUS )); then
        # Parser sanity check: if we matched a count != GPU count, surface that
        # instead of a misleading "rest pending" — the awk pattern is the bug.
        record "fabric state (per GPU)" FAIL "matched $total entries for $EXPECTED_GPUS GPUs ($ok Completed) — parser/driver mismatch"
    elif (( ok == total )); then
        record "fabric state (per GPU)" PASS "$ok/$total Completed"
    else
        record "fabric state (per GPU)" FAIL "$ok/$total Completed (rest pending or failed)"
    fi
else
    record "fabric state (per GPU)" SKIP "nvidia-smi missing"
fi

# Also check the NVLSM child process is alive (FM unit owns it on most builds).
if (( EXPECTED_GPUS <= 1 )) || [[ "${ALLOW_NO_FABRIC:-0}" == "1" ]]; then
    record "nvlsm process" SKIP "no NVSwitch fabric required for this run"
elif pgrep -x nvlsm >/dev/null 2>&1; then
    record "nvlsm process" PASS "running"
else
    record "nvlsm process" FAIL "no nvlsm process — NVSwitch routing tables not configured"
fi

# ============================================================================
# 4. peermem (for multi-node RDMA / GPUDirect)
# ============================================================================
step "4. nvidia-peermem"

if lsmod 2>/dev/null | awk '$1 == "nvidia_peermem" {f=1} END {exit !f}'; then
    record "kmod: nvidia_peermem" PASS "loaded — GPUDirect RDMA ready"
else
    # Strict: peermem absence is the canary for the DOCA/driver ordering bug
    # (forum 370357 — "Invalid argument" on mlx5_core bind) or missing RDMA
    # peer-memory exports from ib_core.
    if ! grep -qw 'ib_register_peer_memory_client' /proc/kallsyms 2>/dev/null \
            || ! grep -qw 'ib_unregister_peer_memory_client' /proc/kallsyms 2>/dev/null; then
        record "kmod: nvidia_peermem" FAIL "not loaded — active ib_core lacks peer-memory symbols; repair DOCA-OFED then rebuild/reinstall NVIDIA DKMS"
    else
        record "kmod: nvidia_peermem" FAIL "not loaded — check dmesg for nvidia_peermem/mlx5_core bind errors"
    fi
fi
# nvidia-peermem.ko ships transitively with nvidia-driver-*-open; install-nvidia.sh
# writes /etc/modules-load.d/nvidia-peermem.conf for boot-time autoload. There is no
# separate nvidia-peermem-loader package in this bundle — verify the autoload config.
PEERMEM_CONF=/etc/modules-load.d/nvidia-peermem.conf
if [[ -r "$PEERMEM_CONF" ]] && grep -qE '^[[:space:]]*nvidia-peermem[[:space:]]*$' "$PEERMEM_CONF"; then
    record "config: nvidia-peermem autoload" PASS "$PEERMEM_CONF"
else
    record "config: nvidia-peermem autoload" FAIL "$PEERMEM_CONF missing or empty — peermem may not reload on reboot"
fi

# ============================================================================
# 5. CUDA toolkit
# ============================================================================
step "5. CUDA toolkit ${CUDA_MAJOR}.${CUDA_MINOR}"

# install-nvidia.sh deliberately avoids the cuda-toolkit-${MAJOR}-${MINOR}
# metapackage (it pulls cuFFT/cuSPARSE/NPP we don't need) and installs the
# minimal set directly. Check each piece individually so MISSING calls out
# the specific package that failed instead of a vague "toolkit missing".
check_dpkg "cuda-nvcc-${CUDA_MAJOR}-${CUDA_MINOR}"
check_dpkg "cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}"
check_dpkg "cuda-cudart-dev-${CUDA_MAJOR}-${CUDA_MINOR}"
check_dpkg "cuda-cccl-${CUDA_MAJOR}-${CUDA_MINOR}"
check_dpkg "libcublas-${CUDA_MAJOR}-${CUDA_MINOR}"
check_dpkg "libcublas-dev-${CUDA_MAJOR}-${CUDA_MINOR}"
check_dpkg "libnvjitlink-${CUDA_MAJOR}-${CUDA_MINOR}"

# CUDA env wiring written by install-nvidia.sh step 5b.
if [[ -r /etc/profile.d/cuda.sh ]] && grep -q '/usr/local/cuda/bin' /etc/profile.d/cuda.sh 2>/dev/null; then
    record "config: /etc/profile.d/cuda.sh" PASS "nvcc on login PATH"
else
    record "config: /etc/profile.d/cuda.sh" FAIL "missing or empty — nvcc won't be on PATH for login shells"
fi
if [[ -r /etc/ld.so.conf.d/cuda-system.conf ]] && ldconfig -p 2>/dev/null | grep -q 'libcudart.so'; then
    record "ld.so cache: libcudart" PASS "$(ldconfig -p | grep -m1 'libcudart.so' | awk '{print $NF}')"
else
    record "ld.so cache: libcudart" FAIL "libcudart.so not in ldconfig cache — non-venv binaries (llama-cli) will fail to load"
fi

NVCC_BIN=""
for c in nvcc /usr/local/cuda/bin/nvcc "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin/nvcc"; do
    if command -v "$c" >/dev/null 2>&1; then NVCC_BIN=$(command -v "$c"); break; fi
    [[ -x "$c" ]] && { NVCC_BIN="$c"; break; }
done
if [[ -n "$NVCC_BIN" ]]; then
    nvcc_ver=$("$NVCC_BIN" --version 2>&1 | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    if [[ "$nvcc_ver" == "${CUDA_MAJOR}.${CUDA_MINOR}" || "$nvcc_ver" == "${CUDA_MAJOR}.${CUDA_MINOR%.*}" ]]; then
        record "nvcc release" PASS "$nvcc_ver via $NVCC_BIN"
    else
        record "nvcc release" FAIL "found $nvcc_ver (expected ${CUDA_MAJOR}.${CUDA_MINOR})"
    fi
else
    record "nvcc release" MISSING "nvcc not on PATH or /usr/local/cuda*"
fi

# ============================================================================
# 6. NCCL (host)
# ============================================================================
step "6. NCCL"

nccl_state=$(dpkg-query -W -f='${db:Status-Abbrev}|${Version}\n' libnccl2 2>/dev/null || true)
if [[ -n "$nccl_state" ]]; then
    nccl_ver="${nccl_state##*|}"
    if [[ "$nccl_ver" == *"+cuda${CUDA_MAJOR}.${CUDA_MINOR}" ]]; then
        # Suffix matches; now enforce the B300-stable minimum version.
        # Version layout is "X.Y.Z-N+cuda13.0" — strip everything from the
        # first '-' onward to get the bare X.Y.Z for sort -V.
        nccl_base="${nccl_ver%%-*}"
        if printf '%s\n%s\n' "$NCCL_MIN_VER" "$nccl_base" | sort -V -C 2>/dev/null; then
            record "dpkg: libnccl2" PASS "$nccl_ver (>= ${NCCL_MIN_VER})"
        else
            record "dpkg: libnccl2" FAIL "$nccl_ver below NCCL_MIN_VER=${NCCL_MIN_VER} — TP>1 AllReduce deadlocks on B300"
        fi
    else
        record "dpkg: libnccl2" FAIL "$nccl_ver (expected +cuda${CUDA_MAJOR}.${CUDA_MINOR})"
    fi
    check_dpkg "libnccl-dev"
else
    record "dpkg: libnccl2" SKIP "not installed (SKIP_NCCL was set, or PyTorch wheel only)"
fi

# ============================================================================
# 7. DCGM
# ============================================================================
step "7. DCGM"

if [[ "$SKIP_DCGM" == "1" ]]; then
    record "dpkg: datacenter-gpu-manager-4-cuda${CUDA_MAJOR}" SKIP "SKIP_DCGM=1"
    record "service: nvidia-dcgm" SKIP "SKIP_DCGM=1"
else
    check_dpkg_optional "datacenter-gpu-manager-4-cuda${CUDA_MAJOR}"
    check_service "nvidia-dcgm" 0

    if command -v dcgmi >/dev/null 2>&1; then
        dcgm_ver=$(dcgmi --version 2>&1 | head -1)
        record "dcgm: dcgmi --version" PASS "$dcgm_ver"
    else
        record "dcgm: dcgmi --version" SKIP "optional binary not on PATH"
    fi
fi

# ============================================================================
# 8. Persistence + ECC
# ============================================================================
step "8. Persistence + ECC"

if command -v nvidia-smi >/dev/null 2>&1; then
    pmode=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
    if [[ -n "$pmode" ]]; then
        if [[ "$pmode" == "Enabled" ]]; then
            record "persistence mode" PASS "Enabled on all GPUs"
        else
            record "persistence mode" FAIL "$pmode — run: nvidia-smi -pm 1"
        fi
    fi
    ecc=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
    record "ecc mode" PASS "${ecc:-?}"
fi

# ============================================================================
# 9. Production-grade GPU sanity (HGX B300)
#    Catches subtle misconfigs that pass earlier checks but degrade workloads:
#    - PCIe link not at full Gen5 x16 (firmware/topology issue)
#    - NVLink lanes count mismatch (broken fabric, but Fabric State still
#      reports Completed because peer-init succeeded on the remaining lanes)
#    - MIG mode accidentally enabled (incompatible with LLM tensor-parallel)
#    - Confidential Computing mode left on (limits perf + GPU sharing)
#    - Duplicate GPU UUIDs (very rare; happens after firmware reflash without
#      proper provisioning — breaks UUID-based GPU pinning)
#    - Kernel module version != installed driver package version (kmod from a
#      prior install still loaded; reboot needed)
# ============================================================================
step "9. Production sanity (HGX B300)"

# nvidia-smi presence already covered by step 1. Skip the rest if absent.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then

    # (a) Kernel module version == installed driver package version.
    # If they disagree, the running kmod is from a previous install and a
    # reboot is mandatory — fabricmanager will refuse to talk to it.
    pkg_drv_ver=$(dpkg-query -W -f='${Version}' "nvidia-driver-${DRIVER_BRANCH}-open" 2>/dev/null | sed 's/-.*//')
    if [[ -n "$pkg_drv_ver" && -n "$KMOD_VER" ]]; then
        if [[ "$KMOD_VER" == "${pkg_drv_ver}"* ]] || [[ "$pkg_drv_ver" == "${KMOD_VER}"* ]]; then
            record "kmod vs package version" PASS "kmod=$KMOD_VER  pkg=$pkg_drv_ver"
        else
            record "kmod vs package version" FAIL "kmod=$KMOD_VER pkg=$pkg_drv_ver — reboot required"
        fi
    fi

    # (b) PCIe link generation + width per GPU. Require >= Gen5 x16 on B300.
    # nvidia-smi reports `gen` as the negotiated PCIe gen number, `width` as
    # the negotiated lane count. B300 boards advertise Gen6 capability, so a
    # strict `gen == 5` check would FAIL on a healthy board. We require
    # >= Gen5 to flag genuine downgrades (BIOS misconfig dropping to Gen4 or
    # a slot reseat issue dropping to x8) without rejecting Gen6 links.
    pcie_data=$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null)
    if [[ -n "$pcie_data" ]]; then
        bad_pcie=$(printf '%s\n' "$pcie_data" | awk -F',' '
            { gsub(/ /, "", $1); gsub(/ /, "", $2)
              if ($1+0 < 5 || $2 != "16") n++ }
            END { print n+0 }')
        total_pcie=$(printf '%s\n' "$pcie_data" | grep -c .)
        if (( bad_pcie == 0 )); then
            sample=$(printf '%s\n' "$pcie_data" | awk -F',' 'NR==1 {gsub(/ /, ""); print $1"x"$2}')
            record "PCIe Gen5+ x16 (per GPU)" PASS "all $total_pcie GPU(s) at >=Gen5 x16 (e.g. Gen${sample})"
        else
            sample=$(printf '%s\n' "$pcie_data" | awk -F',' 'NR==1 {gsub(/ /, ""); print $1"x"$2}')
            record "PCIe Gen5+ x16 (per GPU)" FAIL "$bad_pcie/$total_pcie GPU(s) below Gen5 x16 (e.g. Gen${sample})"
        fi
    fi

    # (c) NVLink lane count per GPU. B300 = 18 NVLink5 lanes per GPU.
    # `nvidia-smi nvlink --status` prints one line per lane "GPU 0: NVLink 0: ..."
    # An "inactive" lane (cable issue, port fault) reduces aggregate fabric BW
    # without failing Fabric State.
    if nvidia-smi nvlink --status >/dev/null 2>&1; then
        nv_status=$(nvidia-smi nvlink --status 2>/dev/null)
        per_gpu_counts=$(printf '%s\n' "$nv_status" | awk '
            /^GPU [0-9]+:/ { gpu = $2; sub(/:/, "", gpu); next }
            /Link [0-9]+:.*GB\/s/ && !/Disabled|Inactive/ { active[gpu]++ }
            END { for (g in active) print active[g] }
        ')
        gpu_count_with_nvl=$(printf '%s\n' "$per_gpu_counts" | grep -c . || true)
        expected_lanes="${EXPECTED_NVLINK_LANES:-18}"
        if (( gpu_count_with_nvl == 0 )); then
            # Strict: HGX B300 must have NVLink lanes per GPU. Empty output
            # from `nvlink --status` means fabric init failed silently — Fabric
            # State may report Completed via a degraded fallback path while
            # all 18 lanes per GPU are actually down.
            record "NVLink lanes per GPU" FAIL "no NVLink data in nvlink --status — fabric did not enumerate links; check dmesg for NVLink/NVSwitch errors"
        else
            mismatched=$(printf '%s\n' "$per_gpu_counts" \
                | awk -v want="$expected_lanes" '$1 != want { n++ } END { print n+0 }')
            min_lanes=$(printf '%s\n' "$per_gpu_counts" | sort -n | head -1)
            if (( mismatched == 0 )); then
                record "NVLink lanes per GPU" PASS "$gpu_count_with_nvl GPU(s) × $expected_lanes active lanes"
            else
                record "NVLink lanes per GPU" FAIL "$mismatched/$gpu_count_with_nvl GPU(s) below $expected_lanes lanes (min observed: $min_lanes). Set EXPECTED_NVLINK_LANES=N if your B300 SKU advertises a different count."
            fi
        fi
    fi

    # (d) MIG mode. HGX B300 LLM workloads don't use MIG — accidental enable
    # would break tensor-parallel partitioning of the model.
    mig_modes=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
    if [[ -z "$mig_modes" ]] || [[ "$mig_modes" == "N/A" ]]; then
        record "MIG mode (must be disabled)" PASS "N/A — MIG not supported or not exposed (ok)"
    elif [[ "$mig_modes" == "Disabled" ]]; then
        record "MIG mode (must be disabled)" PASS "Disabled on all GPUs"
    else
        record "MIG mode (must be disabled)" FAIL "$mig_modes — run: sudo nvidia-smi -i <id> -mig 0"
    fi

    # (e) Confidential Computing mode. On B300, CC has runtime cost and
    # restricts multi-process GPU sharing; for normal LLM training/inference
    # we want it OFF. Query may exit non-zero on older drivers; treat as SKIP.
    if cc_out=$(nvidia-smi conf-compute -f 2>&1); then
        # R580 emits "CC status: OFF" (no "mode" word, no "protected mode"
        # prefix). The legacy patterns covered older drivers; add the
        # CC-status form so OFF is recognized as PASS rather than falling
        # through to the "could not parse" FAIL branch.
        if printf '%s' "$cc_out" | grep -qiE 'mode.*off|protected mode\s*:\s*off|cc status\s*:\s*off|(^|[^a-z])disabled([^a-z]|$)'; then
            record "Confidential Computing OFF" PASS "$(printf '%s' "$cc_out" | head -1 | tr -d '\r')"
        elif printf '%s' "$cc_out" | grep -qiE 'mode.*on|protected mode\s*:\s*on|cc status\s*:\s*on|(^|[^a-z])enabled([^a-z]|$)'; then
            record "Confidential Computing OFF" FAIL "$(printf '%s' "$cc_out" | head -1 | tr -d '\r') — disable: nvidia-smi conf-compute -srs 0"
        else
            # Strict: unparseable output means we cannot prove CC is off.
            # Treat as FAIL so the operator confirms manually with
            # `nvidia-smi conf-compute -f` before proceeding.
            record "Confidential Computing OFF" FAIL "could not parse: $(printf '%s' "$cc_out" | head -1) — verify manually then rerun"
        fi
    else
        # Driver claims conf-compute not supported. On R580+B300 it IS
        # supported; absence indicates a driver/kmod mismatch worth catching.
        record "Confidential Computing OFF" FAIL "nvidia-smi conf-compute returned non-zero — driver may not match this GPU SKU"
    fi

    # (f) GPU UUIDs all unique. Duplicates are rare but they break any tool
    # that maps GPUs by UUID (Kubernetes device plugin, MIG instances,
    # dcgmi, custom orchestrators).
    uuids=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
    if [[ -n "$uuids" ]]; then
        total_uuid=$(printf '%s\n' "$uuids" | grep -c .)
        unique_uuid=$(printf '%s\n' "$uuids" | sort -u | grep -c .)
        if (( total_uuid == unique_uuid )); then
            record "GPU UUIDs unique" PASS "$unique_uuid/$total_uuid distinct"
        else
            record "GPU UUIDs unique" FAIL "$unique_uuid distinct, $total_uuid total — duplicate UUIDs detected"
        fi
    fi

else
    record "production sanity" SKIP "nvidia-smi not usable; covered by earlier checks"
fi

# ============================================================================
# Summary
# ============================================================================
TOTAL=${#RESULTS[@]}

if (( JSON_OUT )); then
    printf '{\n  "checks": [\n'
    first=1
    for r in "${RESULTS[@]}"; do
        IFS='|' read -r name status detail <<<"$r"
        (( first )) || printf ',\n'; first=0
        printf '    {"name": "%s", "status": "%s", "detail": "%s"}' \
            "${name//\"/\\\"}" "$status" "${detail//\"/\\\"}"
    done
    printf '\n  ],\n  "pass": %d, "fail": %d, "missing": %d, "skip": %d, "total": %d\n}\n' \
        "$PASS_COUNT" "$FAIL_COUNT" "$MISS_COUNT" "$SKIP_COUNT" "$TOTAL"
else
    printf '\n%s════════════════════════════════════════════════════════════════%s\n' "$c_mag" "$c_off"
    printf '  SUMMARY  '
    printf '%s%d PASS%s  '   "$c_green" "$PASS_COUNT" "$c_off"
    printf '%s%d FAIL%s  '   "$c_red"   "$FAIL_COUNT" "$c_off"
    printf '%s%d MISS%s  '   "$c_yel"   "$MISS_COUNT" "$c_off"
    printf '%s%d SKIP%s\n'   "$c_dim"   "$SKIP_COUNT" "$c_off"
    printf '%s════════════════════════════════════════════════════════════════%s\n' "$c_mag" "$c_off"
fi

if (( FAIL_COUNT > 0 || MISS_COUNT > 0 )); then
    (( JSON_OUT )) || printf '%sNVIDIA stack NOT ready. Fix failures before running install-all.sh.%s\n' "$c_red" "$c_off"
    exit 1
fi
(( JSON_OUT )) || printf '%sNVIDIA stack ready. Next: sudo bash pre-install-check.sh%s\n' "$c_green" "$c_off"
exit 0
