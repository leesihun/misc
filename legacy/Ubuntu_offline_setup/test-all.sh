#!/usr/bin/env bash
# ============================================================================
# test-all.sh
#   Verifies that every component installed by install-all.sh actually works.
#   For each item: checks whether the binary/service/venv is reachable, runs
#   a smoke command, and — for GUI apps — reports missing shared libraries
#   (the typical cause of "installed but won't launch" on a headless airgap
#   server, e.g. VS Code / Chrome needing libgtk-3, libnss3, libgbm, etc.).
#
# Usage:
#   bash test-all.sh                       # run all checks
#   bash test-all.sh --json                # machine-readable summary
#   INFERENCE_PREFIX=/opt/llm_inference bash test-all.sh
#
# Exit code: 0 if every check passed, 1 if anything is MISSING or FAILED.
# ============================================================================
set -uo pipefail

# -------- configurable (mirror install-all.sh defaults) ---------------------
PYTHON_VER="${PYTHON_VER:-3.12}"
INFERENCE_PREFIX="${INFERENCE_PREFIX:-$HOME/llm_inference}"
TRAINING_PREFIX="${TRAINING_PREFIX:-$HOME/general_training}"
JUPYTER_PREFIX="${JUPYTER_PREFIX:-$HOME/jupyter}"
LLAMA_PREFIX="${LLAMA_PREFIX:-$HOME/llama.cpp}"
# ---------------------------------------------------------------------------

JSON_OUT=0
[[ "${1:-}" == "--json" ]] && JSON_OUT=1

# Track results: name|status|detail   status in {PASS,FAIL,MISSING,SKIP}
RESULTS=()
PASS_COUNT=0
FAIL_COUNT=0
MISS_COUNT=0
SKIP_COUNT=0

c_green=$'\033[1;32m'; c_red=$'\033[1;31m'; c_yel=$'\033[1;33m'
c_cyan=$'\033[1;36m'; c_mag=$'\033[1;35m'; c_dim=$'\033[2m'; c_off=$'\033[0m'

step()   { (( JSON_OUT )) || printf '\n%s== %s ==%s\n' "$c_mag" "$*" "$c_off"; }
record() {
    # record <name> <status> <detail>
    local name="$1" status="$2" detail="${3:-}"
    RESULTS+=( "$name|$status|$detail" )
    case "$status" in
        PASS)    PASS_COUNT=$((PASS_COUNT+1)); (( JSON_OUT )) || printf '  %s[ PASS ]%s %-32s %s\n'    "$c_green" "$c_off" "$name" "$detail" ;;
        FAIL)    FAIL_COUNT=$((FAIL_COUNT+1)); (( JSON_OUT )) || printf '  %s[ FAIL ]%s %-32s %s\n'    "$c_red"   "$c_off" "$name" "$detail" ;;
        MISSING) MISS_COUNT=$((MISS_COUNT+1)); (( JSON_OUT )) || printf '  %s[MISSING]%s %-32s %s\n'   "$c_yel"   "$c_off" "$name" "$detail" ;;
        SKIP)    SKIP_COUNT=$((SKIP_COUNT+1)); (( JSON_OUT )) || printf '  %s[ SKIP ]%s %-32s %s\n'    "$c_dim"   "$c_off" "$name" "$detail" ;;
    esac
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# resolve_bin <candidate1> [candidate2 ...] -> echoes first found, or empty
resolve_bin() {
    local c
    for c in "$@"; do
        if [[ -x "$c" ]]; then echo "$c"; return 0; fi
        if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
    done
    return 1
}

# check_cmd <display-name> <version-flag> <candidate-path...>
#   Runs <bin> <version-flag>; PASS on exit 0, FAIL on non-zero, MISSING if
#   none of the candidates resolve. Captures first line of output as detail.
check_cmd() {
    local name="$1" flag="$2"; shift 2
    local bin
    if ! bin=$(resolve_bin "$@"); then
        record "$name" MISSING "none of: $*"
        return
    fi
    local out rc
    out=$("$bin" $flag 2>&1); rc=$?
    out=$(printf '%s' "$out" | head -n 1 | tr -d '\r')
    if (( rc == 0 )); then
        record "$name" PASS "$bin :: $out"
    else
        record "$name" FAIL "$bin exit=$rc :: $out"
    fi
}

# check_gui <display-name> <candidate-path...>
#   Three-layer check:
#     1) --version (catches missing .so deps via ldd on fail)
#     2) scan --version output for sandbox / userns errors (Firefox/Chrome leak
#        EACCES from CanCreateUserNamespace here even when exit code is 0)
#     3) optional headless launch (only if HEADLESS_GUI_TEST=1) to confirm the
#        GUI can actually start — this is what "code" does in a real terminal.
check_gui() {
    local name="$1"; shift
    local bin
    if ! bin=$(resolve_bin "$@"); then
        record "$name" MISSING "none of: $*"
        return
    fi
    local out rc
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
    # exit==0 but check for sandbox/userns trouble in stderr
    if printf '%s' "$out" | grep -qiE 'CanCreateUserNamespace|user namespace|sandbox.*EACCES|clone\(\) failure'; then
        record "$name" FAIL "sandbox/userns blocked — likely AppArmor restrict_unprivileged_userns=1 (see Diagnostics)"
        return
    fi
    record "$name" PASS "$bin :: $(printf '%s' "$out" | head -n 1 | tr -d '\r')"
}

# check_dpkg <pkg-name>  -> PASS if installed (ii), MISSING otherwise
check_dpkg() {
    local pkg="$1"
    local state
    state=$(dpkg-query -W -f='${db:Status-Abbrev}|${Version}\n' "$pkg" 2>/dev/null || true)
    if [[ -z "$state" ]]; then
        record "dpkg: $pkg" MISSING "not installed"
        return
    fi
    local abbrev="${state%%|*}" ver="${state##*|}"
    if [[ "$abbrev" == "ii "* || "$abbrev" == "ii" ]]; then
        record "dpkg: $pkg" PASS "$ver"
    else
        record "dpkg: $pkg" FAIL "state=$abbrev ver=$ver"
    fi
}

# check_venv <name> <prefix> <python-import-test>
check_venv() {
    local name="$1" prefix="$2" script="$3"
    local py="$prefix/venv/bin/python"
    if [[ ! -x "$py" ]]; then
        record "venv: $name" MISSING "no python at $py"
        return
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

# check_service <name>  -> systemd active / inactive / missing
#   `systemctl cat` exits 0 iff the unit exists — works under SysV-init aliases,
#   masked units, and units that list-unit-files doesn't surface. Avoids the
#   false MISSING that the awk-on-list-unit-files approach gives for xrdp.
check_service() {
    local svc="$1"
    if ! command -v systemctl >/dev/null 2>&1; then
        record "service: $svc" SKIP "systemctl not available"
        return
    fi
    if ! systemctl cat "$svc" >/dev/null 2>&1; then
        record "service: $svc" MISSING "unit not registered with systemd"
        return
    fi
    local state
    state=$(systemctl is-active "$svc" 2>/dev/null || true)
    case "$state" in
        active)            record "service: $svc" PASS    "active" ;;
        inactive|failed)   record "service: $svc" FAIL    "$state" ;;
        *)                 record "service: $svc" FAIL    "${state:-unknown}" ;;
    esac
}

# check_port <name> <port>  -> PASS if anything is listening
check_port() {
    local name="$1" port="$2"
    local listener=""
    if command -v ss >/dev/null 2>&1; then
        listener=$(ss -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $4; exit}')
    elif command -v netstat >/dev/null 2>&1; then
        listener=$(netstat -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $4; exit}')
    else
        record "port: $name ($port)" SKIP "no ss/netstat"
        return
    fi
    if [[ -n "$listener" ]]; then
        record "port: $name ($port)" PASS "listening on $listener"
    else
        record "port: $name ($port)" FAIL "nothing listening"
    fi
}

# ===========================================================================
# 1) TOOLCHAIN
# ===========================================================================
step "Toolchain"
check_cmd "gcc"        "--version" gcc
check_cmd "g++"        "--version" g++
check_cmd "make"       "--version" make
check_cmd "cmake"      "--version" cmake
check_cmd "git"        "--version" git
check_cmd "python${PYTHON_VER}" "--version" "python${PYTHON_VER}" python3
check_cmd "pip"        "--version" pip3 pip

# ===========================================================================
# 2) NVIDIA / CUDA
# ===========================================================================
step "NVIDIA / CUDA"
check_cmd "nvidia-smi" ""          nvidia-smi
check_cmd "nvcc"       "--version" nvcc /usr/local/cuda/bin/nvcc
# Probe a couple of CUDA runtime libs to catch a half-installed toolkit
if ldconfig -p 2>/dev/null | grep -q 'libcudart.so'; then
    record "cuda runtime libs" PASS "$(ldconfig -p | grep -m1 'libcudart.so' | awk '{print $NF}')"
else
    record "cuda runtime libs" MISSING "libcudart.so not in ldconfig cache"
fi
# NCCL — required for multi-GPU collectives over NVLink/NVSwitch (incl. NVLS
# SHARP on B100/B200/B300). Without it, llama.cpp's multi-GPU mode falls back
# to a slow per-pair P2P copy.
_nccl_hdr=""; _nccl_so=""
for _h in /usr/include/nccl.h /usr/local/cuda/include/nccl.h; do
    [[ -f "$_h" ]] && { _nccl_hdr="$_h"; break; }
done
for _l in /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so \
          /usr/local/cuda/lib64/libnccl.so.2; do
    [[ -e "$_l" ]] && { _nccl_so="$_l"; break; }
done
if [[ -n "$_nccl_hdr" && -n "$_nccl_so" ]]; then
    _nccl_ver=$(awk '/NCCL_MAJOR/ {maj=$3} /NCCL_MINOR/ {min=$3} /NCCL_PATCH/ {pat=$3} END {if(maj!="") print maj"."min"."pat}' "$_nccl_hdr" 2>/dev/null)
    record "NCCL" PASS "${_nccl_ver:-installed} ($_nccl_so)"
elif [[ -n "$_nccl_so" ]]; then
    record "NCCL" FAIL "runtime present but header missing — install libnccl-dev so llama.cpp can link against it"
else
    record "NCCL" MISSING "libnccl2/libnccl-dev not installed — multi-GPU collectives will be slow"
fi

# ── Multi-GPU services & fabric health ──────────────────────────────────────
# nvidia-fabricmanager: hard-required on any NVSwitch box. Without it, NCCL
# falls back from NVLink (~700 GB/s) to PCIe (~25 GB/s) — silently.
_gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
    _gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
fi
if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
    _fm_state=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo unknown)
    if [[ "$_fm_state" == "active" ]]; then
        record "nvidia-fabricmanager" PASS "active"
    elif (( _gpu_count > 1 )); then
        record "nvidia-fabricmanager" FAIL "NOT active on $_gpu_count-GPU box — NCCL will fall back to PCIe (~20x slowdown); check 'journalctl -u nvidia-fabricmanager -n 80'"
    else
        record "nvidia-fabricmanager" SKIP "service inactive but only $_gpu_count GPU(s) — fabric not strictly required"
    fi
fi

# nvidia-persistenced: keeps driver loaded, eliminates 5-10s cold CUDA tax.
if systemctl list-unit-files nvidia-persistenced.service >/dev/null 2>&1; then
    _ps_state=$(systemctl is-active nvidia-persistenced 2>/dev/null || echo unknown)
    if [[ "$_ps_state" == "active" ]]; then
        record "nvidia-persistenced" PASS "active"
    else
        record "nvidia-persistenced" FAIL "inactive — every cold CUDA call pays a 5-10s driver-reinit tax"
    fi
fi

# DCGM: telemetry for XID/NVLink/thermal events.
if systemctl list-unit-files nvidia-dcgm.service >/dev/null 2>&1; then
    _dcgm_state=$(systemctl is-active nvidia-dcgm 2>/dev/null || echo unknown)
    if [[ "$_dcgm_state" == "active" ]]; then
        record "nvidia-dcgm" PASS "active (dcgmi diag -r 1 to run health check)"
    else
        record "nvidia-dcgm" FAIL "inactive — no XID/NVLink/thermal telemetry"
    fi
fi

# Persistence mode (separate from the persistenced daemon — this is the
# per-GPU PM flag set by `nvidia-smi -pm 1`).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    _pm_off=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | grep -c -i 'disabled' || true)
    if (( _pm_off == 0 )); then
        record "GPU persistence mode" PASS "all GPUs in persistence mode"
    else
        record "GPU persistence mode" FAIL "$_pm_off GPU(s) NOT in persistence mode — run: sudo nvidia-smi -pm 1"
    fi
fi

# NVLink fabric health: any inactive lane = bandwidth degraded proportionally.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    _nv_status=$(nvidia-smi nvlink --status 2>/dev/null || true)
    if [[ -n "$_nv_status" ]]; then
        _nv_down=$(printf '%s\n' "$_nv_status" | grep -ciE 'inactive|<inactive>|disabled' || true)
        _nv_up=$(printf '%s\n' "$_nv_status" | grep -c 'GB/s' || true)
        if (( _nv_down == 0 )) && (( _nv_up > 0 )); then
            record "NVLink fabric" PASS "$_nv_up active lane(s), 0 inactive"
        elif (( _nv_down > 0 )); then
            record "NVLink fabric" FAIL "$_nv_down inactive NVLink lane(s) — fabric degraded"
        else
            record "NVLink fabric" SKIP "no NVLink data (single-GPU or no NVSwitch)"
        fi
    fi
fi

# nccl-tests presence — the only reliable way to verify NVLink is actually
# carrying NCCL traffic at line rate. The test isn't run here (it's slow);
# we just confirm the binary is available.
check_cmd "nccl-tests: all_reduce_perf" "--help" all_reduce_perf /usr/local/bin/all_reduce_perf

# NUMA tooling
check_cmd "numactl" "--show"   numactl
check_cmd "lstopo"  "--version" lstopo lstopo-no-graphics
check_cmd "nvtop"   "--version" nvtop

# NCCL env defaults — confirms /etc/profile.d/nccl-multigpu.sh was deployed
# (NVLS is OFF by default in NCCL; the file enables it for all login shells).
if [[ -f /etc/profile.d/nccl-multigpu.sh ]]; then
    if grep -q 'NCCL_NVLS_ENABLE=1' /etc/profile.d/nccl-multigpu.sh; then
        record "NCCL env defaults" PASS "/etc/profile.d/nccl-multigpu.sh (NVLS enabled)"
    else
        record "NCCL env defaults" FAIL "file present but NCCL_NVLS_ENABLE=1 not set"
    fi
else
    record "NCCL env defaults" MISSING "/etc/profile.d/nccl-multigpu.sh not installed"
fi

# System limits — required for multi-process inference / training without
# hitting EMFILE or mlock failures mid-run.
if [[ -f /etc/security/limits.d/99-llm-multigpu.conf ]]; then
    record "limits.d" PASS "/etc/security/limits.d/99-llm-multigpu.conf installed"
else
    record "limits.d" MISSING "/etc/security/limits.d/99-llm-multigpu.conf not installed"
fi

# LLM inference helpers
check_cmd "gpu-health-check"      "" gpu-health-check      /usr/local/bin/gpu-health-check
check_cmd "llama-server-multigpu" "" llama-server-multigpu /usr/local/bin/llama-server-multigpu
check_cmd "llama-model-preload"   "" llama-model-preload   /usr/local/bin/llama-model-preload
if systemctl list-unit-files 'llama-server@.service' >/dev/null 2>&1 && \
   systemctl list-unit-files 'llama-server@.service' 2>/dev/null | grep -q llama-server; then
    record "llama-server@.service" PASS "systemd template installed (configure /etc/llama-server/<x>.env)"
else
    record "llama-server@.service" MISSING "systemd template not installed"
fi

# ===========================================================================
# 3) GUI LAUNCH CAPABILITY (why "code" / "google-chrome" do nothing in SSH)
# ===========================================================================
step "GUI launch capability"

# 3a) Display server present in this shell?
if [[ -n "${DISPLAY:-}" ]]; then
    record "display: \$DISPLAY"         PASS "$DISPLAY"
elif [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    record "display: \$WAYLAND_DISPLAY" PASS "$WAYLAND_DISPLAY"
else
    record "display: \$DISPLAY"         FAIL "no DISPLAY/WAYLAND_DISPLAY — GUI apps will silently exit. Launch from xrdp/XFCE session, not raw SSH."
fi

# 3b) Ubuntu 24.04 AppArmor unprivileged userns restriction.
#     This is the #1 reason Chrome and VS Code "do nothing" — Electron's
#     sandbox calls clone(CLONE_NEWUSER) which AppArmor blocks since 24.04.
if [[ -r /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
    val=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo "?")
    if [[ "$val" == "0" ]]; then
        record "apparmor userns"            PASS "unrestricted (0)"
    else
        record "apparmor userns"            FAIL "restricted ($val) — blocks Chrome/VS Code sandbox. Fix: sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0"
    fi
else
    record "apparmor userns"                SKIP "kernel does not expose this sysctl"
fi

# 3c) chrome-sandbox SUID bit (needed when running with sandbox enabled)
for sb in /opt/google/chrome/chrome-sandbox /usr/share/code/chrome-sandbox; do
    [[ -f "$sb" ]] || continue
    perms=$(stat -c '%a %U' "$sb" 2>/dev/null)
    if [[ "$perms" == "4755 root" ]]; then
        record "chrome-sandbox SUID"        PASS "$sb ($perms)"
    else
        record "chrome-sandbox SUID"        FAIL "$sb has $perms (expected 4755 root). Fix: sudo chown root:root $sb && sudo chmod 4755 $sb"
    fi
done

# ===========================================================================
# 4) BROWSERS / GUI APPS
# ===========================================================================
step "Browsers & GUI apps"
check_gui "VS Code"        code /usr/bin/code /usr/share/code/code /snap/bin/code
check_gui "Google Chrome"  google-chrome-stable google-chrome /opt/google/chrome/google-chrome /opt/google/chrome/chrome
check_gui "Firefox"        firefox /opt/firefox/firefox /usr/local/bin/firefox

# ===========================================================================
# 4) DEV CLI TOOLS
# ===========================================================================
step "Developer CLIs"
check_cmd "Node.js"  "--version" node /opt/nodejs/bin/node /usr/local/bin/node
check_cmd "npm"      "--version" npm  /opt/nodejs/bin/npm  /usr/local/bin/npm
check_cmd "npx"      "--version" npx  /opt/nodejs/bin/npx  /usr/local/bin/npx
check_cmd "Bun"      "--version" bun  /usr/local/bin/bun
check_cmd "Opencode" "--version" opencode /usr/local/bin/opencode

# ===========================================================================
# 5) LLAMA.CPP
# ===========================================================================
step "llama.cpp"
check_cmd "llama-cli"    "--version" "$LLAMA_PREFIX/build/bin/llama-cli" llama-cli
check_cmd "llama-server" "--version" "$LLAMA_PREFIX/build/bin/llama-server" llama-server

# ===========================================================================
# 6) PYTHON VENVS
# ===========================================================================
step "Python venvs"
check_venv "inference" "$INFERENCE_PREFIX" \
    'import sys, torch; print(f"py{sys.version_info.major}.{sys.version_info.minor} torch={torch.__version__} cuda={torch.cuda.is_available()}")'
check_venv "training"  "$TRAINING_PREFIX" \
    'import torch, torch_geometric; print(f"torch={torch.__version__} pyg={torch_geometric.__version__} cuda={torch.cuda.is_available()}")'
check_venv "jupyter"   "$JUPYTER_PREFIX" \
    'import jupyterlab, notebook; print(f"jupyterlab={jupyterlab.__version__} notebook={notebook.__version__}")'

# Bonus: vLLM (only meaningful in inference venv)
if [[ -x "$INFERENCE_PREFIX/venv/bin/python" ]]; then
    check_venv "inference: vLLM" "$INFERENCE_PREFIX" \
        'import vllm; print(f"vllm={vllm.__version__}")'
fi

# ===========================================================================
# 7) K3S / KUBERNETES
# ===========================================================================
step "K3s / Kubernetes"
check_cmd "k3s"     "--version" k3s     /usr/local/bin/k3s
check_cmd "kubectl" "version --client=true --output=yaml" kubectl /usr/local/bin/kubectl
check_cmd "helm"    "version --short"   helm    /usr/local/bin/helm
check_service "k3s"
check_service "k3s-agent"
# Cluster reachability (only if kubectl + kubeconfig are present)
if command -v kubectl >/dev/null 2>&1 && { [[ -f "$HOME/.kube/config" ]] || [[ -f /etc/rancher/k3s/k3s.yaml ]]; }; then
    export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
    [[ -f "$KUBECONFIG" ]] || export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    nodes=$(kubectl get nodes --no-headers 2>&1)
    nrc=$?
    if (( nrc == 0 )); then
        ready=$(echo "$nodes" | awk '$2=="Ready"' | wc -l)
        total=$(echo "$nodes" | wc -l)
        record "k8s cluster" PASS "$ready/$total nodes Ready"
    else
        record "k8s cluster" FAIL "$(echo "$nodes" | head -n 1)"
    fi
else
    record "k8s cluster" SKIP "no kubeconfig"
fi

# ===========================================================================
# 8) REMOTE DESKTOP (xrdp + XFCE4)
# ===========================================================================
step "Remote desktop"
check_dpkg "xrdp"
check_dpkg "xfce4"
check_service "xrdp"
check_port "RDP" 3389

# ===========================================================================
# 9) BROKEN DPKG STATE (catches half-installed deps that often break GUI apps)
# ===========================================================================
step "Broken dpkg state"
if command -v dpkg >/dev/null 2>&1; then
    broken=$(dpkg -l 2>/dev/null | awk '/^.[HUF]/ {print $2}')
    bcount=$(printf '%s\n' "$broken" | grep -c '\S' || true)
    if (( bcount == 0 )); then
        record "dpkg health" PASS "no broken packages"
    else
        record "dpkg health" FAIL "$bcount broken: $(echo $broken | tr '\n' ' ' | cut -c1-120)..."
    fi
else
    record "dpkg health" SKIP "dpkg not available"
fi

# ===========================================================================
# SUMMARY
# ===========================================================================
if (( JSON_OUT )); then
    printf '{\n  "pass": %d, "fail": %d, "missing": %d, "skip": %d,\n  "results": [\n' \
        "$PASS_COUNT" "$FAIL_COUNT" "$MISS_COUNT" "$SKIP_COUNT"
    first=1
    for row in "${RESULTS[@]}"; do
        IFS='|' read -r n s d <<< "$row"
        # naive JSON escape: backslash + quote
        d=${d//\\/\\\\}; d=${d//\"/\\\"}
        n=${n//\\/\\\\}; n=${n//\"/\\\"}
        (( first )) || printf ',\n'
        printf '    {"name": "%s", "status": "%s", "detail": "%s"}' "$n" "$s" "$d"
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
            [[ "$s" == "FAIL" || "$s" == "MISSING" ]] && printf '  [%s] %-32s %s\n' "$s" "$n" "$d"
        done
        printf '\n%sHints:%s\n' "$c_cyan" "$c_off"
        printf '  GUI app "does nothing" in SSH? Three likely causes (in order):\n'
        printf '    1. No DISPLAY — you are in raw SSH; launch from the xrdp/XFCE session,\n'
        printf '       or "ssh -X" with X11 forwarding.\n'
        printf '    2. AppArmor blocks unprivileged userns (default on Ubuntu 24.04):\n'
        printf '       sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0\n'
        printf '       (persist via: echo kernel.apparmor_restrict_unprivileged_userns=0 |\n'
        printf '        sudo tee /etc/sysctl.d/60-apparmor-userns.conf)\n'
        printf '    3. Missing shared libs — apt-get install the names ldd reports,\n'
        printf '       e.g. libnss3, libgbm1, libgtk-3-0, libasound2t64, libxshmfence1.\n'
        printf '  Broken dpkg state? Try:  sudo apt-get -f install\n'
    fi
fi

(( FAIL_COUNT + MISS_COUNT == 0 ))
