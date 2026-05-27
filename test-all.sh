#!/usr/bin/env bash
# ============================================================================
# test-all.sh  (manufacturer-prepped variant)
#
#   Post-install verification for the prepped 8x B300 Ubuntu 24.04 server.
#   Adapted from Ubuntu_offline_setup/test-all.sh; drops driver/DKMS install
#   verification (assumes vendor handled that — pre-install-check.sh already
#   validated it) and focuses on what install-all.sh just did: app installs,
#   Python venvs, llama.cpp, and system tuning.
#
#   Usage:
#     bash test-all.sh                    # human-readable, run every phase
#     bash test-all.sh --json             # machine-readable
#     bash test-all.sh --phase userland   # only the userland-apt sections
#     bash test-all.sh --phase apps       # only browsers + GUI tooling
#     bash test-all.sh --phase desktop    # only xrdp/xfce4
#     bash test-all.sh --phase venvs      # only Python venvs + llama.cpp
#     bash test-all.sh --phase tuning     # only sysctl/limits/ops tooling
#     bash test-all.sh --phase nvidia     # only the vendor NVIDIA stack
#     bash test-all.sh --phase all        # default — every phase
#
#   The --phase flag exists so the multi-reboot install flow can verify just
#   the slice the operator wants after a given reboot. The mapping mirrors
#   the install phases in install-all.sh (see CLAUDE.md).
#
#   Exit code: 0 if every check passed, 1 if anything is MISSING or FAILED.
# ============================================================================
set -uo pipefail

# Mirror install-all.sh defaults
PYTHON_VER="${PYTHON_VER:-3.12}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
INFERENCE_PREFIX="${INFERENCE_PREFIX:-$SCRATCH_ROOT/llm_inference}"
TRAINING_PREFIX="${TRAINING_PREFIX:-$SCRATCH_ROOT/general_training}"
JUPYTER_PREFIX="${JUPYTER_PREFIX:-$SCRATCH_ROOT/jupyter}"
LLAMA_PREFIX="${LLAMA_PREFIX:-$SCRATCH_ROOT/llama.cpp}"

# CLI
JSON_OUT=0
PHASE="${PHASE:-all}"
while (( $# > 0 )); do
    case "$1" in
        --json)            JSON_OUT=1; shift ;;
        --phase)           PHASE="$2"; shift 2 ;;
        --phase=*)         PHASE="${1#*=}"; shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$PHASE" in
    all|nvidia|userland|apps|desktop|venvs|tuning) ;;
    *) printf 'unknown --phase: %s (expected: all, nvidia, userland, apps, desktop, venvs, tuning)\n' "$PHASE" >&2; exit 2 ;;
esac

# phase_runs <name> — true if PHASE=all OR PHASE matches <name>.
phase_runs() {
    [[ "$PHASE" == "all" || "$PHASE" == "$1" ]]
}

# Result tracking
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
        PASS)    PASS_COUNT=$((PASS_COUNT+1)); (( JSON_OUT )) || printf '  %s[ PASS ]%s %-36s %s\n' "$c_green" "$c_off" "$name" "$detail" ;;
        FAIL)    FAIL_COUNT=$((FAIL_COUNT+1)); (( JSON_OUT )) || printf '  %s[ FAIL ]%s %-36s %s\n' "$c_red"   "$c_off" "$name" "$detail" ;;
        MISSING) MISS_COUNT=$((MISS_COUNT+1)); (( JSON_OUT )) || printf '  %s[MISSING]%s %-36s %s\n' "$c_yel"   "$c_off" "$name" "$detail" ;;
        SKIP)    SKIP_COUNT=$((SKIP_COUNT+1)); (( JSON_OUT )) || printf '  %s[ SKIP ]%s %-36s %s\n' "$c_dim"   "$c_off" "$name" "$detail" ;;
    esac
}

resolve_bin() {
    local c
    for c in "$@"; do
        [[ -x "$c" ]] && { echo "$c"; return 0; }
        command -v "$c" >/dev/null 2>&1 && { command -v "$c"; return 0; }
    done
    return 1
}

check_cmd() {
    local name="$1" flag="$2"; shift 2
    local bin out rc
    if ! bin=$(resolve_bin "$@"); then
        record "$name" MISSING "none of: $*"
        return
    fi
    out=$("$bin" $flag 2>&1); rc=$?
    out=$(printf '%s' "$out" | head -n 1 | tr -d '\r')
    if (( rc == 0 )); then
        record "$name" PASS "$bin :: $out"
    else
        record "$name" FAIL "$bin exit=$rc :: $out"
    fi
}

check_gui() {
    local name="$1"; shift
    local bin out rc
    if ! bin=$(resolve_bin "$@"); then
        record "$name" MISSING "none of: $*"
        return
    fi
    out=$("$bin" --version 2>&1); rc=$?
    if (( rc != 0 )); then
        local missing_libs
        missing_libs=$(ldd "$bin" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')
        if [[ -n "$missing_libs" ]]; then
            record "$name" FAIL "missing libs: ${missing_libs% }"
        else
            local err
            err=$(printf '%s' "$out" | head -n 2 | tr '\n' ' ' | tr -d '\r')
            record "$name" FAIL "exit=$rc :: ${err:-no output}"
        fi
        return
    fi
    if printf '%s' "$out" | grep -qiE 'CanCreateUserNamespace|user namespace|sandbox.*EACCES|clone\(\) failure'; then
        record "$name" FAIL "sandbox/userns blocked — sysctl kernel.apparmor_restrict_unprivileged_userns=0"
        return
    fi
    record "$name" PASS "$bin :: $(printf '%s' "$out" | head -n 1 | tr -d '\r')"
}

check_dpkg() {
    local pkg="$1" state
    state=$(dpkg-query -W -f='${db:Status-Abbrev}|${Version}\n' "$pkg" 2>/dev/null || true)
    if [[ -z "$state" ]]; then
        record "dpkg: $pkg" MISSING "not installed"; return
    fi
    local abbrev="${state%%|*}" ver="${state##*|}"
    if [[ "$abbrev" == "ii "* || "$abbrev" == "ii" ]]; then
        record "dpkg: $pkg" PASS "$ver"
    else
        record "dpkg: $pkg" FAIL "state=$abbrev ver=$ver"
    fi
}

check_venv() {
    local name="$1" prefix="$2" script="$3"
    local py="$prefix/venv/bin/python"
    if [[ ! -x "$py" ]]; then
        record "venv: $name" MISSING "no python at $py"; return
    fi
    local out rc
    out=$("$py" -c "$script" 2>&1); rc=$?
    out=$(printf '%s' "$out" | tr '\n' ' ' | tr -d '\r')
    if (( rc == 0 )); then
        record "venv: $name" PASS "$out"
    else
        record "venv: $name" FAIL "$out"
    fi
}

check_service() {
    local svc="$1"
    if ! command -v systemctl >/dev/null 2>&1; then
        record "service: $svc" SKIP "systemctl not available"; return
    fi
    if ! systemctl cat "$svc" >/dev/null 2>&1; then
        record "service: $svc" MISSING "unit not registered"; return
    fi
    local state
    state=$(systemctl is-active "$svc" 2>/dev/null || true)
    case "$state" in
        active)            record "service: $svc" PASS "active" ;;
        inactive|failed)   record "service: $svc" FAIL "$state" ;;
        *)                 record "service: $svc" FAIL "${state:-unknown}" ;;
    esac
}

check_port() {
    local name="$1" port="$2" listener=""
    if command -v ss >/dev/null 2>&1; then
        listener=$(ss -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $4; exit}')
    elif command -v netstat >/dev/null 2>&1; then
        listener=$(netstat -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $4; exit}')
    else
        record "port: $name ($port)" SKIP "no ss/netstat"; return
    fi
    if [[ -n "$listener" ]]; then
        record "port: $name ($port)" PASS "listening on $listener"
    else
        record "port: $name ($port)" FAIL "nothing listening"
    fi
}

# ============================================================================
# 1) TOOLCHAIN  (phase: userland)
# ============================================================================
if phase_runs userland; then
step "Toolchain"
check_cmd "gcc"               "--version" gcc
check_cmd "g++"               "--version" g++
check_cmd "make"              "--version" make
check_cmd "cmake"             "--version" cmake
check_cmd "ninja"             "--version" ninja
check_cmd "git"               "--version" git
check_cmd "python${PYTHON_VER}" "--version" "python${PYTHON_VER}" python3
check_cmd "pip"               "--version" pip3 pip
fi

# ============================================================================
# 2) NVIDIA / CUDA (vendor-installed — verify presence, not install)  (phase: nvidia)
# ============================================================================
if phase_runs nvidia; then
step "NVIDIA / CUDA (vendor-installed)"

check_cmd "nvidia-smi" ""          nvidia-smi
check_cmd "nvcc"       "--version" nvcc /usr/local/cuda/bin/nvcc

# Driver version
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    record "driver version" PASS "$DRV"

    # Expect 8 B300 GPUs
    n=$(nvidia-smi -L | wc -l | tr -d ' ')
    if (( n == 8 )); then
        record "GPU count" PASS "$n"
    else
        record "GPU count" FAIL "found $n (expected 8 for B300 HGX)"
    fi
    names=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)
    if (( $(printf '%s\n' "$names" | grep -c .) == 1 )) && printf '%s' "$names" | grep -qi 'B300'; then
        record "GPU SKU" PASS "$(printf '%s' "$names" | tr -d '\n')"
    else
        record "GPU SKU" FAIL "$(printf '%s' "$names" | tr -d '\n')"
    fi

    # Fabric state: count the real per-GPU Fabric State entries. A loose grep
    # can pass when one GPU says Completed while another is still pending.
    fab=$(nvidia-smi -q 2>/dev/null | awk '
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
    fab_total=$(printf '%s\n' "$fab" | grep -c . || true)
    fab_ok=$(printf '%s\n' "$fab" | grep -cE '^(Completed|Success)$' || true)
    if (( fab_total == 0 )); then
        record "fabric.state" FAIL "not reported - common cause of CUDA Error 802"
    elif (( fab_total != n )); then
        record "fabric.state" FAIL "matched $fab_total Fabric entries for $n GPU(s) ($fab_ok Completed/Success)"
    elif (( fab_ok == fab_total )); then
        record "fabric.state" PASS "$fab_ok/$fab_total Completed/Success"
    else
        record "fabric.state" FAIL "$fab_ok/$fab_total Completed/Success - common cause of CUDA Error 802"
    fi
fi

# CUDA runtime libs
if ldconfig -p 2>/dev/null | grep -q 'libcudart.so'; then
    record "cuda runtime libs" PASS "$(ldconfig -p | grep -m1 'libcudart.so' | awk '{print $NF}')"
else
    record "cuda runtime libs" MISSING "libcudart.so not in ldconfig cache"
fi

# Multi-GPU services (vendor-managed — just check they're active)
check_service "nvidia-fabricmanager"
check_service "nvidia-persistenced"
if systemctl list-unit-files nvidia-nvlsm.service >/dev/null 2>&1; then
    check_service "nvidia-nvlsm"
fi

# Persistence mode
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    pm_off=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | grep -c -i 'disabled' || true)
    if (( pm_off == 0 )); then
        record "GPU persistence mode" PASS "all GPUs Enabled"
    else
        record "GPU persistence mode" FAIL "$pm_off GPU(s) Disabled — run: sudo nvidia-smi -pm 1"
    fi
fi

# NVLink fabric health
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    nv_status=$(nvidia-smi nvlink --status 2>/dev/null || true)
    if [[ -n "$nv_status" ]]; then
        nv_down=$(printf '%s\n' "$nv_status" | grep -ciE 'inactive|<inactive>|disabled' || true)
        nv_up=$(printf '%s\n' "$nv_status" | grep -c 'GB/s' || true)
        if (( nv_down == 0 )) && (( nv_up > 0 )); then
            record "NVLink fabric" PASS "$nv_up active lane(s), 0 inactive"
        elif (( nv_down > 0 )); then
            record "NVLink fabric" FAIL "$nv_down inactive lane(s) — fabric degraded"
        else
            record "NVLink fabric" SKIP "no NVLink data"
        fi
    fi
fi

# NUMA tooling
check_cmd "numactl" "--show" numactl
check_cmd "nvtop"   "--version" nvtop
fi   # phase: nvidia

# ============================================================================
# 2b) System tuning artifacts + ops helpers  (phase: tuning)
# ============================================================================
if phase_runs tuning; then
step "System tuning + operational helpers"
if [[ -f /etc/security/limits.d/99-llm-multigpu.conf ]]; then
    record "limits.d" PASS "/etc/security/limits.d/99-llm-multigpu.conf"
else
    record "limits.d" MISSING "/etc/security/limits.d/99-llm-multigpu.conf"
fi
if [[ -f /etc/sysctl.d/99-llm-multigpu.conf ]]; then
    record "sysctl.d" PASS "/etc/sysctl.d/99-llm-multigpu.conf"
else
    record "sysctl.d" MISSING "/etc/sysctl.d/99-llm-multigpu.conf"
fi
thp_state=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true)
if [[ "$thp_state" == *"[madvise]"* ]]; then
    record "THP = madvise" PASS "$thp_state"
elif [[ -n "$thp_state" ]]; then
    record "THP = madvise" FAIL "current: $thp_state — disable-thp-defrag.service did not apply"
else
    record "THP = madvise" MISSING "/sys/kernel/mm/transparent_hugepage/enabled unreadable"
fi

# Helper scripts
check_cmd "gpu-health-check"      "" gpu-health-check      /usr/local/bin/gpu-health-check
check_cmd "llama-server-multigpu" "" llama-server-multigpu /usr/local/bin/llama-server-multigpu
check_cmd "llama-model-preload"   "" llama-model-preload   /usr/local/bin/llama-model-preload
if systemctl cat llama-server@.service >/dev/null 2>&1; then
    record "llama-server@.service" PASS "systemd template installed"
else
    record "llama-server@.service" MISSING "systemd template not registered"
fi
fi   # phase: tuning

# ============================================================================
# 3) GUI LAUNCH CAPABILITY  (phase: apps)
# ============================================================================
if phase_runs apps; then
step "GUI launch capability"

if [[ -n "${DISPLAY:-}" ]]; then
    record "display: \$DISPLAY"         PASS "$DISPLAY"
elif [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    record "display: \$WAYLAND_DISPLAY" PASS "$WAYLAND_DISPLAY"
else
    record "display: \$DISPLAY"         FAIL "no DISPLAY — GUI apps will silently exit when launched from SSH"
fi

if [[ -r /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
    val=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo "?")
    if [[ "$val" == "0" ]]; then
        record "apparmor userns" PASS "unrestricted (0)"
    else
        record "apparmor userns" FAIL "restricted ($val) — blocks Chrome/VS Code"
    fi
fi

for sb in /opt/google/chrome/chrome-sandbox /usr/share/code/chrome-sandbox; do
    [[ -f "$sb" ]] || continue
    perms=$(stat -c '%a %U' "$sb" 2>/dev/null)
    if [[ "$perms" == "4755 root" ]]; then
        record "chrome-sandbox SUID" PASS "$sb ($perms)"
    else
        record "chrome-sandbox SUID" FAIL "$sb has $perms (expected 4755 root)"
    fi
done

fi   # phase: apps  (closes section 3)

# ============================================================================
# 4) BROWSERS / GUI APPS  (phase: apps)
# ============================================================================
if phase_runs apps; then
step "Browsers & GUI apps"
check_gui "VS Code"        code /usr/bin/code /usr/share/code/code
check_gui "Google Chrome"  google-chrome-stable google-chrome /opt/google/chrome/google-chrome
check_gui "Firefox"        firefox /opt/firefox/firefox /usr/local/bin/firefox
fi   # phase: apps

# ============================================================================
# 5) DEV CLIs  (phase: apps)
# ============================================================================
if phase_runs apps; then
step "Developer CLIs"
check_cmd "Node.js"  "--version" node /opt/nodejs/bin/node /usr/local/bin/node
check_cmd "npm"      "--version" npm  /opt/nodejs/bin/npm  /usr/local/bin/npm
check_cmd "Bun"      "--version" bun  /usr/local/bin/bun
check_cmd "Opencode" "--version" opencode /usr/local/bin/opencode
fi   # phase: apps

# ============================================================================
# 6) LLAMA.CPP  (phase: venvs)
# ============================================================================
if phase_runs venvs; then
step "llama.cpp"
check_cmd "llama-cli"    "--version" "$LLAMA_PREFIX/build/bin/llama-cli" llama-cli
check_cmd "llama-server" "--version" "$LLAMA_PREFIX/build/bin/llama-server" llama-server
fi   # phase: venvs

# ============================================================================
# 7) PYTHON VENVS  (phase: venvs)
# ============================================================================
if phase_runs venvs; then
step "Python venvs"
# Inference venv runs on the SAME torch +cu130 as training (see CLAUDE.md);
# probe torch + the RAG/FastAPI stack. vLLM is intentionally absent.
check_venv "inference (RAG + GPU torch)" "$INFERENCE_PREFIX" \
    'import sys, importlib
mods = (
    "torch", "fastapi", "uvicorn", "uvloop", "watchfiles", "httptools",
    "bcrypt", "cryptography", "jose", "langchain", "sentence_transformers",
    "transformers", "tiktoken",
)
out = []
for m in mods:
    try:
        out.append(f"{m}={getattr(importlib.import_module(m), \"__version__\", \"?\")}")
    except Exception as e:
        out.append(f"{m}=IMPORT_FAIL({e.__class__.__name__})")
try:
    import torch
    out.append(f"cuda={torch.cuda.is_available()}")
except Exception:
    out.append("cuda=PROBE_FAIL")
print(f"py{sys.version_info.major}.{sys.version_info.minor} " + " ".join(out))'

check_venv "training"  "$TRAINING_PREFIX" \
    'import torch, torch_geometric; print(f"torch={torch.__version__} pyg={torch_geometric.__version__} cuda={torch.cuda.is_available()}")'

check_venv "jupyter"   "$JUPYTER_PREFIX" \
    'import jupyterlab, notebook; print(f"jupyterlab={jupyterlab.__version__} notebook={notebook.__version__}")'
fi   # phase: venvs

# ============================================================================
# 8) REMOTE DESKTOP (if installed)  (phase: desktop)
# ============================================================================
if phase_runs desktop; then
step "Remote desktop"
if dpkg-query -W -f='${db:Status-Abbrev}' xrdp 2>/dev/null | grep -q '^ii'; then
    check_dpkg "xrdp"
    check_dpkg "xfce4"
    check_service "xrdp"
    check_port "RDP" 3389
else
    record "xrdp" SKIP "not installed (INSTALL_DESKTOP=0 during install)"
fi
fi   # phase: desktop

# ============================================================================
# 9) BROKEN DPKG STATE  (phase: tuning)
# ============================================================================
if phase_runs tuning; then
step "Broken dpkg state"
if command -v dpkg >/dev/null 2>&1; then
    broken=$(dpkg -l 2>/dev/null | awk '/^.[HUF]/ {print $2}')
    bcount=$(printf '%s\n' "$broken" | grep -c '\S' || true)
    if (( bcount == 0 )); then
        record "dpkg health" PASS "no broken packages"
    else
        record "dpkg health" FAIL "$bcount broken: $(echo $broken | tr '\n' ' ' | cut -c1-120)"
    fi
fi
fi   # phase: tuning

# ============================================================================
# 10) /run/reboot-required  (phase: tuning)
# ============================================================================
if phase_runs tuning; then
step "Reboot state"
if [[ -f /run/reboot-required ]]; then
    pkgs=$(tr '\n' ' ' </run/reboot-required.pkgs 2>/dev/null || echo "")
    record "reboot pending" FAIL "/run/reboot-required is set${pkgs:+ (pkgs: $pkgs)}"
else
    record "reboot pending" PASS "no pending reboot"
fi
fi   # phase: tuning

# ============================================================================
# SUMMARY
# ============================================================================
if (( JSON_OUT )); then
    printf '{\n  "pass":%d,"fail":%d,"missing":%d,"skip":%d,\n  "results":[\n' \
        "$PASS_COUNT" "$FAIL_COUNT" "$MISS_COUNT" "$SKIP_COUNT"
    first=1
    for row in "${RESULTS[@]}"; do
        IFS='|' read -r n s d <<< "$row"
        d=${d//\\/\\\\}; d=${d//\"/\\\"}
        n=${n//\\/\\\\}; n=${n//\"/\\\"}
        (( first )) || printf ',\n'
        printf '    {"name":"%s","status":"%s","detail":"%s"}' "$n" "$s" "$d"
        first=0
    done
    printf '\n  ]\n}\n'
else
    step "Summary"
    printf '  %sPASS%s    : %d\n' "$c_green" "$c_off" "$PASS_COUNT"
    printf '  %sFAIL%s    : %d\n' "$c_red"   "$c_off" "$FAIL_COUNT"
    printf '  %sMISSING%s : %d\n' "$c_yel"   "$c_off" "$MISS_COUNT"
    printf '  %sSKIP%s    : %d\n' "$c_dim"   "$c_off" "$SKIP_COUNT"

    if (( FAIL_COUNT + MISS_COUNT > 0 )); then
        printf '\n%sThings that need attention:%s\n' "$c_yel" "$c_off"
        for row in "${RESULTS[@]}"; do
            IFS='|' read -r n s d <<< "$row"
            [[ "$s" == "FAIL" || "$s" == "MISSING" ]] && printf '  [%s] %-36s %s\n' "$s" "$n" "$d"
        done
        printf '\n%sHints:%s\n' "$c_cyan" "$c_off"
        printf '  GUI app silent in SSH? Three likely causes:\n'
        printf '    1. No DISPLAY — launch via xrdp/xfce4 session or ssh -X.\n'
        printf '    2. apparmor unprivileged userns restricted — sysctl kernel.apparmor_restrict_unprivileged_userns=0\n'
        printf '    3. Missing shared libs — apt-get install the names ldd reports.\n'
        printf '  busBW < 1200 GB/s on 8x B300? Check nvidia-fabricmanager status + nvidia-smi nvlink --status\n'
        printf '  Inference venv import errors? It now ships torch +cu130 — verify with: %s/venv/bin/pip list | grep -iE "^(torch|vllm)\\b" (expect torch present, vllm absent)\n' "$INFERENCE_PREFIX"
        printf '  Broken dpkg? sudo apt-get -f install\n'
    fi
fi

(( FAIL_COUNT + MISS_COUNT == 0 ))
