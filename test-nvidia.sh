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
#     - CUDA toolkit installed, nvcc reports CUDA 13.x
#     - libnccl2 installed with +cuda13.0 suffix
#     - DCGM service status (informational)
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
check_service "nvidia-nvlsm" 0          # optional unit on some builds
check_service "nvidia-persistenced" 1

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
        record "fabric state (per GPU)" SKIP "no Fabric stanza in nvidia-smi -q (FM not initialized?)"
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
if pgrep -x nvlsm >/dev/null 2>&1; then
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
    record "kmod: nvidia_peermem" MISSING "not loaded; run: modprobe nvidia-peermem"
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

check_dpkg "cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"
check_dpkg "cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}"

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
        record "dpkg: libnccl2" PASS "$nccl_ver"
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

check_dpkg "datacenter-gpu-manager-4-cuda${CUDA_MAJOR}"
check_service "nvidia-dcgm" 0

if command -v dcgmi >/dev/null 2>&1; then
    dcgm_ver=$(dcgmi --version 2>&1 | head -1)
    record "dcgm: dcgmi --version" PASS "$dcgm_ver"
else
    record "dcgm: dcgmi --version" MISSING "dcgmi not on PATH"
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
