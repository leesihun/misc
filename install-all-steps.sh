#!/usr/bin/env bash
# ============================================================================
# install-all-steps.sh
#
#   Consolidated library — sourced by install-all.sh. Contains:
#     1. Shared helpers (formerly install-all.d/00-common.sh)
#     2. The ordered list of step names (ALL_STEPS array)
#     3. One function per step (step_01_preflight … step_17_final_status),
#        formerly the 17 individual install-all.d/NN-name.sh scripts.
#
#   The split-file install-all.d/ layout was collapsed into this single file
#   so the airgap operator only has to transfer ONE script to the target
#   alongside install-all.sh (FTP one-by-one transfer ergonomics).
#
#   Each step function:
#     - is invoked by install-all.sh in a SUBSHELL `( step_NN_name )` so its
#       init_step exec-redirect, ERR/EXIT traps, and any die/checkpoint_reboot
#       exit 75 are scoped to that subshell.
#     - starts with `init_step "NN-name"` and ends with `mark_step_ok` (and,
#       for phase-boundary steps 06/08/09/15/17, `checkpoint_reboot`).
#     - inherits `set -Eeuo pipefail` from the launcher; do not re-set it.
#     - does NOT call `require_root` — install-all.sh re-execs under sudo
#       once, before sourcing this file.
#
#   To run a single step interactively for debugging:
#     sudo bash install-all.sh --run 14
# ============================================================================

# Guard against double-source.
[[ -n "${_INSTALL_ALL_STEPS_LOADED:-}" ]] && return 0
_INSTALL_ALL_STEPS_LOADED=1

# ============================================================================
# SECTION 1 — Shared helpers (formerly install-all.d/00-common.sh)
# ============================================================================

# Force noninteractive apt for every child process. lightdm (and a few other
# debconf-using packages) will hang a batch install if a prompt slips through;
# -y alone does NOT suppress debconf questions.
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

# ── Paths ───────────────────────────────────────────────────────────────────
INSTALL_ALL_STEPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$INSTALL_ALL_STEPS_DIR"

# Allow caller to inherit RUN_ID (so multiple step scripts in one launcher
# invocation share one log directory). If unset, generate now.
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

STATE_DIR="${INSTALL_ALL_STATE_DIR:-/var/lib/install-all}"
LOG_ROOT="${INSTALL_ALL_LOG_ROOT:-/var/log/install-all}"
STEPS_DIR="$STATE_DIR/steps"
RUN_LOG_DIR="$LOG_ROOT/$RUN_ID"
DIAG_DIR="${INSTALL_ALL_DIAG_DIR:-$REPO_ROOT}"
ACC_DIR="$STATE_DIR/run-$RUN_ID"  # accumulator dir for warnings/errors across steps

# Resume marker from the pre-existing two-stage apt path (kept for back-compat).
RESUME_MARKER="${RESUME_MARKER:-/var/lib/install-all-prepped/stage1.done}"

# ── Default env knobs (mirror install-all.sh defaults) ─────────────────────
BUNDLE_DIR="${BUNDLE_DIR:-$REPO_ROOT}"
BUNDLE_BIN="${BUNDLE_BIN:-}"
PYTHON_VER="${PYTHON_VER:-3.12}"
PYTHON_BIN="${PYTHON_BIN:-python${PYTHON_VER}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
INFERENCE_PREFIX="${INFERENCE_PREFIX:-$SCRATCH_ROOT/llm_inference}"
TRAINING_PREFIX="${TRAINING_PREFIX:-$SCRATCH_ROOT/general_training}"
LLAMA_PREFIX="${LLAMA_PREFIX:-$SCRATCH_ROOT/llama.cpp}"
JUPYTER_PREFIX="${JUPYTER_PREFIX:-$SCRATCH_ROOT/jupyter}"

INSTALL_INFERENCE="${INSTALL_INFERENCE:-1}"
INSTALL_TRAINING="${INSTALL_TRAINING:-1}"
INSTALL_JUPYTER="${INSTALL_JUPYTER:-1}"
INSTALL_LLAMA="${INSTALL_LLAMA:-1}"
INSTALL_DESKTOP="${INSTALL_DESKTOP:-1}"
# INSTALL_VLLM is intentionally NOT exposed. The inference venv is a CPU-only
# RAG/FastAPI/langchain stack — no torch, no vLLM. See step_11_venv_inference.

# B300 = sm_103 (Blackwell Ultra). 100 covers B200 in mixed fleets.
# Use -real to strip PTX — host is fixed-hardware (B200/B300 only).
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-100-real;103-real}"
BUILD_BLAS="${BUILD_BLAS:-1}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

APT_REPO_DIR="${APT_REPO_DIR:-/var/tmp/airgap-bundle-debs}"

# ── Pre-step state, set/overridden by init_step ─────────────────────────────
STEP_NAME=""
STEP_LOG=""

# ── Color codes (off if not a TTY) ──────────────────────────────────────────
if [[ -t 1 ]]; then
    _C_LOG=$'\033[1;32m'; _C_WARN=$'\033[1;33m'; _C_ERR=$'\033[1;31m'
    _C_STEP=$'\033[1;35m'; _C_INFO=$'\033[1;36m'; _C_OFF=$'\033[0m'
else
    _C_LOG=""; _C_WARN=""; _C_ERR=""; _C_STEP=""; _C_INFO=""; _C_OFF=""
fi

# ── Step state directories ──────────────────────────────────────────────────
_ensure_state_dirs() {
    install -d -m 0755 "$STATE_DIR" "$STEPS_DIR" "$LOG_ROOT" "$RUN_LOG_DIR" "$ACC_DIR" 2>/dev/null || true
}
_ensure_state_dirs

# ── Print helpers ───────────────────────────────────────────────────────────
log()  { printf '%s[install]%s %s\n'        "$_C_LOG"  "$_C_OFF" "$*"; }
warn() {
    printf '%s[install:WARN]%s %s\n' "$_C_WARN" "$_C_OFF" "$*" >&2
    [[ -d "$ACC_DIR" ]] && printf '%s\n' "${STEP_NAME:-?}: $*" >> "$ACC_DIR/warnings.log" 2>/dev/null || true
}
die()  {
    printf '%s[install:ERROR]%s %s\n' "$_C_ERR"  "$_C_OFF" "$*" >&2
    [[ -d "$ACC_DIR" ]] && printf '%s\n' "${STEP_NAME:-?}: $*" >> "$ACC_DIR/errors.log" 2>/dev/null || true
    exit 1
}
step() { printf '\n%s== %s ==%s\n'           "$_C_STEP" "$*"      "$_C_OFF"; }

# ── Step lifecycle ──────────────────────────────────────────────────────────
init_step() {
    STEP_NAME="$1"
    STEP_LOG="$RUN_LOG_DIR/${STEP_NAME}.log"
    _ensure_state_dirs

    # Redirect stdout+stderr through tee so console AND log capture. Scoped
    # to the current subshell — when the launcher invokes a step as
    # `( step_NN_name )`, exec's redirect lasts only for that subshell.
    if [[ -d "$RUN_LOG_DIR" ]]; then
        exec > >(tee -a "$STEP_LOG") 2>&1
    fi

    printf '%s[step]%s %s starting (run %s) — log %s\n' \
        "$_C_INFO" "$_C_OFF" "$STEP_NAME" "$RUN_ID" "$STEP_LOG"

    rm -f "$STEPS_DIR/${STEP_NAME}.failed" 2>/dev/null || true

    trap '_step_on_err $?' ERR
    trap '_step_on_exit $?' EXIT
}

_step_on_err() {
    local rc="${1:-1}" cmd="${BASH_COMMAND:-?}"
    printf '%s[step:ERR]%s %s failed at: %s (rc=%s)\n' \
        "$_C_ERR" "$_C_OFF" "$STEP_NAME" "$cmd" "$rc" >&2
    [[ -d "$ACC_DIR" ]] && printf '%s\n' "${STEP_NAME}: ${cmd} (rc=${rc})" >> "$ACC_DIR/errors.log" 2>/dev/null || true
    return "$rc"
}

_step_on_exit() {
    local rc="${1:-0}"
    # rc=75 is checkpoint_reboot's distinguished exit code; mark_step_ok ran
    # immediately before it, so .ok exists. Writing .failed here too would
    # leave BOTH markers on disk and confuse external tools that grep for
    # .failed. Treat 75 as a successful exit for marker purposes.
    if (( rc != 0 && rc != 75 )) && [[ -d "$STEPS_DIR" ]]; then
        : > "$STEPS_DIR/${STEP_NAME}.failed" 2>/dev/null || true
    fi
    return "$rc"
}

mark_step_ok() {
    [[ -d "$STEPS_DIR" ]] || _ensure_state_dirs
    : > "$STEPS_DIR/${STEP_NAME}.ok" 2>/dev/null || true
    rm -f "$STEPS_DIR/${STEP_NAME}.failed" 2>/dev/null || true
    printf '%s[step]%s %s ok\n' "$_C_LOG" "$_C_OFF" "$STEP_NAME"
}

# ── Reboot checkpoint ───────────────────────────────────────────────────────
# Force a deliberate reboot break after a phase that ran successfully. Pattern:
#
#   mark_step_ok       # write the .ok marker FIRST so re-run skips this step
#   checkpoint_reboot "<one-line rationale>"
#
# The launcher catches exit 75 and prints a banner directing the operator to
# reboot. Re-running install-all.sh after reboot skips this .ok step and
# resumes at the next pending one. Set SKIP_CHECKPOINTS=1 to bypass (NOT
# recommended on B300 — see CLAUDE.md "Many reboots are good").
checkpoint_reboot() {
    local reason="${1:-phase boundary}"
    if [[ "${SKIP_CHECKPOINTS:-0}" == "1" ]]; then
        log "SKIP_CHECKPOINTS=1 — bypassing reboot break ($reason)"
        return 0
    fi
    printf '\n%s================================================================%s\n' "$_C_WARN" "$_C_OFF"
    printf '%s  REBOOT CHECKPOINT — %s%s\n' "$_C_WARN" "$STEP_NAME" "$_C_OFF"
    printf '%s================================================================%s\n' "$_C_WARN" "$_C_OFF"
    printf '  Reason : %s\n' "$reason"
    if [[ -f /run/reboot-required ]]; then
        printf '  /run/reboot-required is SET\n'
        if [[ -r /run/reboot-required.pkgs ]]; then
            printf '  Triggered by: %s\n' "$(tr '\n' ' ' < /run/reboot-required.pkgs)"
        fi
    fi
    printf '\n  Next:\n'
    printf '    sudo reboot\n'
    printf '    # after reboot, verify and resume:\n'
    printf '    sudo bash test-nvidia.sh\n'
    printf '    sudo bash install-all.sh   # picks up at next pending step\n\n'
    exit 75
}

# Step skip helper: invoked by the launcher only.
step_is_done() {
    local name="$1"
    [[ -f "$STEPS_DIR/${name}.ok" ]]
}

# ── Root check (idempotent re-exec under sudo) ──────────────────────────────
require_root() {
    if [[ $EUID -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            log "Re-exec under sudo (RUN_ID=$RUN_ID propagated)"
            exec sudo -E env RUN_ID="$RUN_ID" bash "$0" "$@"
        fi
        die "Must run as root."
    fi
}

# ── Bundle discovery + metadata sourcing ────────────────────────────────────
locate_bundle() {
    if [[ -d "$BUNDLE_DIR/debs" && -d "$BUNDLE_DIR/apps" ]]; then
        return 0
    fi
    if [[ -z "$BUNDLE_BIN" ]]; then
        shopt -s nullglob
        local _bins=( "$REPO_ROOT"/all-airgap-bundle-ubuntu*.bin )
        (( ${#_bins[@]} > 0 )) && BUNDLE_BIN="${_bins[0]}"
        shopt -u nullglob
    fi
    [[ -f "$BUNDLE_BIN" ]] || die "No extracted bundle and no .bin found. Set BUNDLE_DIR or BUNDLE_BIN."

    if [[ -f "${BUNDLE_BIN}.sha256" ]]; then
        log "Verifying bundle SHA256 against sidecar"
        ( cd "$(dirname "$BUNDLE_BIN")" && sha256sum -c "$(basename "$BUNDLE_BIN").sha256" ) \
            || die "Bundle SHA256 mismatch — $BUNDLE_BIN is corrupt or doesn't match sidecar."
    else
        warn "No .sha256 sidecar at ${BUNDLE_BIN}.sha256 — skipping integrity verification."
    fi

    log "Extracting $BUNDLE_BIN -> $REPO_ROOT"
    tar -xf "$BUNDLE_BIN" -C "$REPO_ROOT" || die "Failed to extract $BUNDLE_BIN"

    shopt -s nullglob
    local cand
    for cand in "$REPO_ROOT"/*/; do
        if [[ -d "${cand}debs" && -d "${cand}apps" ]]; then
            BUNDLE_DIR="${cand%/}"; break
        fi
    done
    shopt -u nullglob
    [[ -d "$BUNDLE_DIR/debs" && -d "$BUNDLE_DIR/apps" ]] \
        || die "Bundle extracted but no debs/+apps/ found under $REPO_ROOT"
    log "Using bundle: $BUNDLE_DIR"
}

source_bundle_metadata() {
    if [[ -f "$BUNDLE_DIR/meta/target.env" ]]; then
        # shellcheck disable=SC1091
        source "$BUNDLE_DIR/meta/target.env"
    fi
    local nv_env=""
    for cand in \
        /var/tmp/GPU_server_downloads_nvidia/meta/target.env \
        "$BUNDLE_DIR/../GPU_server_downloads_nvidia/meta/target.env"; do
        [[ -f "$cand" ]] && { nv_env="$cand"; break; }
    done
    if [[ -n "$nv_env" ]]; then
        local _bundle_cuda
        _bundle_cuda=$(awk -F= '/^BUNDLE_CUDA=/ {print $2; exit}' "$nv_env" 2>/dev/null)
        if [[ "$_bundle_cuda" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
            export CUDA_MAJOR="${BASH_REMATCH[1]}"
            export CUDA_MINOR="${BASH_REMATCH[2]}"
        fi
    fi
    export CUDA_MAJOR="${CUDA_MAJOR:-13}"
    export CUDA_MINOR="${CUDA_MINOR:-0}"
}

# ── Target user identity ────────────────────────────────────────────────────
detect_target_user() {
    TARGET_USER="${SUDO_USER:-$USER}"
    TARGET_GROUP="$(id -gn "$TARGET_USER" 2>/dev/null || echo "$TARGET_USER")"
    export TARGET_USER TARGET_GROUP
}

_as_user() {
    if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
        sudo -u "$SUDO_USER" "$@"
    else
        "$@"
    fi
}

# ── apt helpers ─────────────────────────────────────────────────────────────
_pkg_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

# t64 transition mapping (Ubuntu 24.04). Some Chrome/VS Code deps name the
# pre-t64 package; this maps the old name to its t64-suffixed equivalent.
_pkg_satisfied() {
    local pkg="$1" alt=""
    _pkg_installed "$pkg" && return 0
    case "$pkg" in
        libglib2.0-0)       alt="libglib2.0-0t64" ;;
        libatk1.0-0)        alt="libatk1.0-0t64" ;;
        libatk-bridge2.0-0) alt="libatk-bridge2.0-0t64" ;;
        libcups2)           alt="libcups2t64" ;;
        libgtk-3-0)         alt="libgtk-3-0t64" ;;
        libasound2)         alt="libasound2t64" ;;
    esac
    [[ -n "$alt" ]] && _pkg_installed "$alt"
}

_normalize_pkg_name() {
    case "$1" in
        libglib2.0-0)       printf '%s\n' libglib2.0-0t64 ;;
        libatk1.0-0)        printf '%s\n' libatk1.0-0t64 ;;
        libatk-bridge2.0-0) printf '%s\n' libatk-bridge2.0-0t64 ;;
        libcups2)           printf '%s\n' libcups2t64 ;;
        libgtk-3-0)         printf '%s\n' libgtk-3-0t64 ;;
        libasound2)         printf '%s\n' libasound2t64 ;;
        *)                  printf '%s\n' "$1" ;;
    esac
}

_apt_install_strict() {
    log "  apt-get install ${*:1:4}..."
    apt-get install -y --no-install-recommends --allow-downgrades "$@" \
        || apt-get install -y --allow-downgrades "$@" \
        || die "apt-get install failed for required packages: $*"
}

_apt_install() {
    log "  apt-get install ${*:1:4}..."
    if apt-get install -y --no-install-recommends --allow-downgrades "$@" 2>/dev/null; then
        return 0
    fi
    if apt-get install -y --allow-downgrades "$@" 2>/dev/null; then
        return 0
    fi
    warn "Batch install failed; retrying packages individually and skipping ones with unmet deps."
    local pkg failed=()
    for pkg in "$@"; do
        if ! apt-get install -y --no-install-recommends --allow-downgrades "$pkg" >/dev/null 2>&1; then
            if ! apt-get install -y --allow-downgrades "$pkg" >/dev/null 2>&1; then
                failed+=( "$pkg" )
            fi
        fi
    done
    if (( ${#failed[@]} > 0 )); then
        warn "Skipped (unsatisfiable in this bundle's local apt repo): ${failed[*]}"
    fi
}

# ── Wheelhouse helpers ──────────────────────────────────────────────────────
_wheelhouse_has_packages() {
    local d="$1"
    [[ -d "$d" ]] || return 1
    compgen -G "$d/*.whl"     >/dev/null \
        || compgen -G "$d/*.tar.gz" >/dev/null \
        || compgen -G "$d/*.tgz"    >/dev/null \
        || compgen -G "$d/*.zip"    >/dev/null
}

generate_wheelhouse_requirements() {
    local wheels_root="$BUNDLE_DIR/wheels" wheelhouse req archive base name version stem
    [[ -d "$wheels_root" ]] || { warn "wheels/ directory missing; no wheelhouse manifests generated."; return 0; }

    shopt -s nullglob
    for wheelhouse in "$wheels_root"/*; do
        [[ -d "$wheelhouse" ]] || continue
        req="$wheelhouse/requirements.txt"
        : > "$req"
        for archive in "$wheelhouse"/*.whl "$wheelhouse"/*.tar.gz "$wheelhouse"/*.tgz "$wheelhouse"/*.zip; do
            [[ -f "$archive" ]] || continue
            base="$(basename "$archive")"
            case "$base" in
                *.whl)
                    name="${base%%-*}"
                    stem="${base#*-}"
                    version="${stem%%-*}"
                    ;;
                *.tar.gz)
                    stem="${base%.tar.gz}"
                    name="${stem%-*}"
                    version="${stem##*-}"
                    ;;
                *.tgz)
                    stem="${base%.tgz}"
                    name="${stem%-*}"
                    version="${stem##*-}"
                    ;;
                *.zip)
                    stem="${base%.zip}"
                    name="${stem%-*}"
                    version="${stem##*-}"
                    ;;
            esac
            [[ -n "${name:-}" && -n "${version:-}" && "$name" != "$version" ]] || continue
            name="${name//_/-}"
            printf '%s==%s\n' "$name" "$version" >> "$req"
        done
        sort -u "$req" -o "$req"
        if [[ -s "$req" ]]; then
            log "Generated $(wc -l < "$req" | tr -d ' ') entries in ${req#$BUNDLE_DIR/}"
        else
            rm -f "$req"
            warn "${wheelhouse#$BUNDLE_DIR/} has no package archives; requirements.txt not generated."
        fi
    done
    shopt -u nullglob
}

# ── Misc ────────────────────────────────────────────────────────────────────
_ver_int() {
    local v="$1" maj min pat
    IFS='.' read -r maj min pat <<<"$v"
    printf '%d' $(( 10#${maj:-0}*1000000 + 10#${min:-0}*1000 + 10#${pat:-0} ))
}


# ============================================================================
# SECTION 2 — Step list (ordered)
# ============================================================================
ALL_STEPS=(
    01-preflight
    02-scratch
    03-apt-repo
    04-apt-plan
    05-reboot-trigger-packages
    06-apt-userland
    07-app-debs
    08-tarball-apps
    09-desktop-xrdp
    10-wheelhouse-manifests
    11-venv-inference
    12-venv-training
    13-venv-jupyter
    14-llamacpp-build
    15-system-tuning
    16-operational-tooling
    17-final-status
)


# ============================================================================
# SECTION 3 — Step functions
# ============================================================================

# ────────────────────────────────────────────────────────────────────────────
# STEP 01: preflight
#   Re-exec under sudo, locate + verify + extract the userland bundle, apply
#   variant guard, run pre-install-check.sh.
# ────────────────────────────────────────────────────────────────────────────
step_01_preflight() {
    init_step "01-preflight"

    local SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
    local FORCE="${FORCE:-0}"

    step "1. Bundle locate + extract"
    locate_bundle
    source_bundle_metadata
    detect_target_user
    log "Bundle: $BUNDLE_DIR  (variant=${BUNDLE_VARIANT:-?}, target=ubuntu ${BUNDLE_OS_VERSION:-?}, built=${BUNDLE_DATE:-?})"
    log "CUDA   : ${CUDA_MAJOR}.${CUDA_MINOR}"
    log "Identity: $TARGET_USER:$TARGET_GROUP (uid=$EUID)"

    step "2. Variant guard"
    if [[ "${BUNDLE_VARIANT:-}" != "prepped" ]]; then
        if (( FORCE )); then
            warn "Bundle variant is '${BUNDLE_VARIANT:-<unset>}' (expected 'prepped'); proceeding due to --force."
        else
            die "Bundle variant is '${BUNDLE_VARIANT:-<unset>}', expected 'prepped'. This bundle is for a bare-metal server; install Ubuntu_offline_setup/install-all.sh instead. Set FORCE=1 to override."
        fi
    fi
    . /etc/os-release
    [[ "${ID}" == "${BUNDLE_OS_ID:-}" && "${VERSION_ID}" == "${BUNDLE_OS_VERSION:-}" ]] \
        || warn "OS mismatch: target ${ID}/${VERSION_ID} vs bundle ${BUNDLE_OS_ID:-?}/${BUNDLE_OS_VERSION:-?}"

    step "3. Pre-flight gate (pre-install-check.sh)"
    if (( SKIP_PREFLIGHT )); then
        warn "Skipping pre-install-check.sh per SKIP_PREFLIGHT=1"
    else
        local PREFLIGHT=""
        for cand in "$REPO_ROOT/pre-install-check.sh" "$BUNDLE_DIR/pre-install-check.sh"; do
            [[ -r "$cand" ]] && { PREFLIGHT="$cand"; break; }
        done
        if [[ -n "$PREFLIGHT" ]]; then
            log "Running $PREFLIGHT"
            local preflight_args=( --bundle "${BUNDLE_BIN:-$BUNDLE_DIR}" )
            (( FORCE )) && preflight_args+=( --force )
            if bash "$PREFLIGHT" "${preflight_args[@]}"; then
                log "Pre-flight passed."
            else
                local rc=$?
                if (( FORCE )); then
                    warn "Pre-flight failed (rc=$rc) but FORCE=1 was given; proceeding anyway."
                else
                    die "Pre-flight failed (rc=$rc). Fix the RED findings above, or re-run with FORCE=1 / SKIP_PREFLIGHT=1."
                fi
            fi
        else
            warn "pre-install-check.sh not found. Proceeding without pre-flight gate — RECOMMEND running it first."
        fi
    fi

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 02: scratch
#   Create $SCRATCH_ROOT (default /scratch) and chown to target user.
# ────────────────────────────────────────────────────────────────────────────
step_02_scratch() {
    init_step "02-scratch"
    detect_target_user

    step "Scratch directory: $SCRATCH_ROOT"
    if [[ ! -d "$SCRATCH_ROOT" ]]; then
        mkdir -p "$SCRATCH_ROOT" || die "Could not create $SCRATCH_ROOT"
    fi
    chown "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT" || warn "chown $SCRATCH_ROOT failed."
    chmod 0775 "$SCRATCH_ROOT" 2>/dev/null || true
    log "$SCRATCH_ROOT ready (owner $TARGET_USER:$TARGET_GROUP)"

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 03: apt-repo
#   Set up the local file:// apt repo from the bundle's debs/, place
#   defensive holds on system libraries the NVIDIA driver/CUDA stack links
#   against, and refresh apt indexes.
# ────────────────────────────────────────────────────────────────────────────
step_03_apt_repo() {
    init_step "03-apt-repo"
    locate_bundle
    source_bundle_metadata

    step "1. System-lib holds (protect CUDA runtime link target)"
    # Hold libstdc++6/libgcc-s1/libgomp1/libc6 so the userland install can't
    # silently DOWNGRADE these via --allow-downgrades when the bundle ships older
    # versions than what install-nvidia.sh's CUDA toolkit pulled in — that would
    # break nvidia-smi / NVML at runtime.
    local sys_pkgs
    sys_pkgs=$({ dpkg-query -W -f='${Package} ${Status}\n' \
        'libstdc++6' 'libgcc-s1' 'libgomp1' 'libc6' \
        2>/dev/null || true; } \
        | awk '$2 == "install" && $3 == "ok" && $4 == "installed" {print $1}' \
        | sort -u)
    if [[ -n "$sys_pkgs" ]]; then
        log "Placing apt-mark hold on $(printf '%s\n' "$sys_pkgs" | wc -l) system runtime-lib packages"
        # shellcheck disable=SC2086
        apt-mark hold $sys_pkgs >/dev/null 2>&1 || warn "apt-mark hold reported errors (non-fatal)."
        install -d -m 0755 "$STATE_DIR"
        printf '%s\n' "$sys_pkgs" > "$STATE_DIR/system-libs-held.txt"
    else
        warn "No system runtime libs matched the hold patterns — unusual."
    fi

    step "2. Local apt repo: $APT_REPO_DIR"
    rm -rf "$APT_REPO_DIR"
    mkdir -p "$APT_REPO_DIR"
    cp -r "$BUNDLE_DIR/debs/." "$APT_REPO_DIR/"
    if command -v dpkg-scanpackages >/dev/null 2>&1; then
        ( cd "$APT_REPO_DIR" && dpkg-scanpackages . /dev/null > Packages 2>/dev/null )
        gzip -9c "$APT_REPO_DIR/Packages" > "$APT_REPO_DIR/Packages.gz"
    elif [[ -f "$APT_REPO_DIR/Packages" ]]; then
        log "Using bundled debs/Packages index (dpkg-scanpackages not installed yet)"
        [[ -f "$APT_REPO_DIR/Packages.gz" ]] || gzip -9c "$APT_REPO_DIR/Packages" > "$APT_REPO_DIR/Packages.gz"
    else
        die "Local apt repo has no Packages index and dpkg-scanpackages is unavailable. Rebuild the bundle with gather-all.sh."
    fi

    step "3. Register sources.list.d entry"
    tee /etc/apt/sources.list.d/00-bundle.list > /dev/null <<EOF
# Installed by step_03_apt_repo (install-all-steps.sh) — local airgap bundle apt repo.
deb [trusted=yes] file://$APT_REPO_DIR ./
EOF

    step "4. apt-get update"
    apt-get update -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 \
        || warn "apt-get update reported errors (vendor's NVIDIA repo may be unreachable on airgap — OK)."

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 04: apt-plan
#   STRICT base-OS gate. Runs an apt dry-run to confirm the userland bundle
#   would NOT upgrade libc6 / systemd / dbus / kernel / firmware / microcode
#   on the target. Any such trigger is a HARD FAIL (no FORCE escape).
# ────────────────────────────────────────────────────────────────────────────
step_04_apt_plan() {
    init_step "04-apt-plan"
    locate_bundle
    source_bundle_metadata

    local APT_PKGS_FILE="$BUNDLE_DIR/meta/apt-packages.txt"
    [[ -f "$APT_PKGS_FILE" ]] || die "$APT_PKGS_FILE not found in bundle."

    local REQUESTED_PKGS=()
    local pkg
    while IFS= read -r pkg; do
        [[ -n "$pkg" && "$pkg" != \#* ]] || continue
        REQUESTED_PKGS+=( "$(_normalize_pkg_name "$pkg")" )
    done < "$APT_PKGS_FILE"
    (( ${#REQUESTED_PKGS[@]} > 0 )) || die "$APT_PKGS_FILE did not contain any installable package names."

    install -d -m 0755 "$STATE_DIR"
    printf '%s\n' "${REQUESTED_PKGS[@]}" > "$STATE_DIR/apt-requested.txt"
    log "Requested packages: ${#REQUESTED_PKGS[@]} (see $STATE_DIR/apt-requested.txt)"

    # Anything in this set, if upgraded post-NVIDIA, can desync the driver/kernel
    # ABI or restart PID-1-adjacent daemons. We refuse ANY of these.
    local BASE_OS_DANGER_REGEX='^(libc6|libc6-dev|systemd|systemd-sysv|dbus|dbus-daemon|linux-image-.*|linux-headers-.*|linux-firmware|microcode|intel-microcode|amd64-microcode)$'

    if [[ -f "$RESUME_MARKER" ]]; then
        log "Resume marker present ($RESUME_MARKER) — base-OS work assumed complete from a prior run."
        : > "$STATE_DIR/apt-proposed.txt"
        : > "$STATE_DIR/apt-reboot-triggers.txt"
        mark_step_ok
        return 0
    fi

    step "apt dry-run (strict base-OS gate)"
    # NOTE: do NOT set a `trap '... rm SIMULATE_OUT' EXIT` here — init_step
    # already installed an EXIT trap (_step_on_exit). Clean up the tempfile
    # inline at each exit path instead.
    local SIMULATE_OUT
    SIMULATE_OUT=$(mktemp)
    if ! apt-get install -s -y --no-install-recommends --allow-downgrades \
            -o APT::Get::Show-Versions=1 \
            "${REQUESTED_PKGS[@]}" 2>&1 | tee "$SIMULATE_OUT" > /dev/null; then
        tail -40 "$SIMULATE_OUT" >&2 || true
        local sim_copy="$STATE_DIR/apt-dry-run-fail.log"
        cp -f "$SIMULATE_OUT" "$sim_copy" 2>/dev/null || true
        rm -f "$SIMULATE_OUT"
        die "apt dry-run failed — dependency resolution broken in this bundle. Last 40 lines above; full output saved to $sim_copy. Rebuild gather-all.sh."
    fi

    local PROPOSED_PKGS
    PROPOSED_PKGS=$(grep -E '^(Inst|Conf) ' "$SIMULATE_OUT" | awk '{print $2}' | sort -u)
    printf '%s\n' "$PROPOSED_PKGS" > "$STATE_DIR/apt-proposed.txt"
    rm -f "$SIMULATE_OUT"

    local BASE_OS_HITS
    BASE_OS_HITS=$(printf '%s\n' "$PROPOSED_PKGS" | grep -E "$BASE_OS_DANGER_REGEX" || true)
    if [[ -n "$BASE_OS_HITS" ]]; then
        printf '\n\033[1;31m================================================================\033[0m\n'
        printf '\033[1;31m  BASE-OS UPGRADE DETECTED — INSTALL REFUSED (no FORCE escape)\033[0m\n'
        printf '\033[1;31m================================================================\033[0m\n'
        printf '  The userland bundle would upgrade the following base-OS packages:\n'
        printf '    %s\n' $BASE_OS_HITS
        printf '\n  These touch the libc/systemd/kernel/firmware ABI surface that\n'
        printf '  the already-installed NVIDIA driver (R%s) was matched against\n' "${DRIVER_BRANCH:-580}"
        printf '  in install-nvidia.sh. Upgrading them now is the canonical brick\n'
        printf '  path on B300 (peermem ABI break, FM "system not initialized",\n'
        printf '  unbootable kernel without nvidia.ko).\n'
        printf '\n  Required recovery (in this order):\n'
        printf '    1. Coordinate with the server vendor to update the BASE OS to\n'
        printf '       the level matching this bundle (or rebuild the userland\n'
        printf '       bundle on a gather host that matches the target baseline).\n'
        printf '    2. Reboot into the updated baseline.\n'
        printf '    3. Re-run the FULL sequence from pre-install-nvidia.sh.\n'
        printf '\n  (Trigger list saved to %s for reference.)\n\n' "$STATE_DIR/apt-base-os-hits.txt"
        printf '%s\n' "$BASE_OS_HITS" > "$STATE_DIR/apt-base-os-hits.txt"
        die "Refusing to upgrade base-OS packages on top of an already-installed NVIDIA driver."
    fi

    log "No base-OS upgrades pending — clean apt plan."
    : > "$STATE_DIR/apt-reboot-triggers.txt"

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 05: reboot-trigger-packages  (STRICT NO-OP ASSERTER)
#   Step 04 hard-refuses any base-OS upgrade. By the time this step runs the
#   trigger file MUST be empty. If it isn't, refuse.
# ────────────────────────────────────────────────────────────────────────────
step_05_reboot_trigger_packages() {
    init_step "05-reboot-trigger-packages"

    local TRIGGER_FILE="$STATE_DIR/apt-reboot-triggers.txt"

    if [[ -f "$RESUME_MARKER" ]]; then
        log "Resume marker present — Stage 1 already handled by a prior run."
        mark_step_ok
        return 0
    fi

    if [[ -s "$TRIGGER_FILE" ]]; then
        local triggers
        triggers=$(tr '\n' ' ' < "$TRIGGER_FILE" | sed 's/[[:space:]]*$//')
        die "Refusing to perform base-OS install. Step 04 should have refused first; trigger file '$TRIGGER_FILE' is non-empty: $triggers. Delete the trigger file ONLY after confirming step 04 is the current revision."
    fi

    log "No base-OS triggers — step 05 is a no-op (clean path)."
    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 06: apt-userland
#   apt install of toolchain, python3.12-venv/dev, needrestart, CLI tools,
#   GUI runtime libs, scientific libs, and (optionally) xfce4 + xrdp.
#   Ends with checkpoint_reboot — verify nvidia survives apt churn.
# ────────────────────────────────────────────────────────────────────────────
step_06_apt_userland() {
    init_step "06-apt-userland"

    step "1. Toolchain"
    _apt_install_strict build-essential cmake ninja-build pkg-config git ccache curl wget ca-certificates unzip xz-utils

    step "2. python${PYTHON_VER}-venv + dev"
    _apt_install_strict "python${PYTHON_VER}-venv" "python${PYTHON_VER}-dev" python3-pip

    step "3. needrestart"
    _apt_install needrestart

    step "4. CLI utilities + monitoring"
    _apt_install gedit vim nano htop btop nvtop iotop tmux screen \
        net-tools iproute2 dnsutils mtr-tiny traceroute \
        jq tree ncdu zip pigz zstd rsync \
        numactl hwloc-nox

    step "5. GUI runtime libs (Chrome/VS Code deps)"
    _apt_install \
        libglib2.0-0t64 libatk1.0-0t64 libatk-bridge2.0-0t64 \
        libcairo2 libcups2t64 libdbus-1-3 libdrm2 libexpat1 \
        libfontconfig1 fonts-liberation libgbm1 libgtk-3-0t64 \
        libnspr4 libnss3 libpango-1.0-0 libsecret-1-0 \
        libasound2t64 libx11-6 libx11-xcb1 libxcb1 \
        libxcomposite1 libxcursor1 libxdamage1 libxext6 \
        libxfixes3 libxi6 libxkbcommon0 libxkbfile1 \
        libxrandr2 libxrender1 libxss1 libxtst6 xdg-utils

    step "6. Scientific native libs (h5py/openblas) + libssl-dev for llama.cpp OpenSSL"
    _apt_install libopenblas-dev libopenblas0 libgomp1 libhdf5-dev libssl-dev libffi-dev libcurl4-openssl-dev

    if [[ "$INSTALL_DESKTOP" == "1" ]]; then
        if command -v lspci >/dev/null 2>&1; then
            if ! lspci 2>/dev/null | grep -qiE '(VGA compatible controller|Display controller)'; then
                warn "No VGA/display controller detected via lspci. lightdm will still be installed (INSTALL_DESKTOP=1)."
                warn "  - lightdm.service may stall graphical.target boot by ~90s on headless hosts."
                warn "  - sshd is on multi-user.target so SSH still works, but boot is slower."
                warn "  - Re-run with INSTALL_DESKTOP=0 if you only need SSH access."
            fi
        fi
        step "7. XFCE4 + xrdp + policykit"
        _apt_install \
            xfce4 xfce4-goodies xfce4-terminal xfce4-screenshooter xfce4-taskmanager xfce4-notifyd \
            lightdm lightdm-gtk-greeter \
            xrdp xorgxrdp ssl-cert \
            policykit-1-gnome \
            dbus-x11 x11-xserver-utils x11-utils xauth xinit xterm \
            file-roller evince ristretto \
            xclip dconf-editor \
            fonts-dejavu-core fonts-noto-core fonts-noto-color-emoji \
            adwaita-icon-theme gnome-themes-extra \
            p7zip-full bash-completion
    fi

    mark_step_ok
    checkpoint_reboot "userland apt install completed; reboot to confirm nvidia.ko still loads and graphical.target doesn't stall boot"
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 07: app-debs
#   Install VS Code and Google Chrome from the bundled .debs. Reload AppArmor
#   profiles. Allow unprivileged user namespaces (Chrome/VS Code Electron sandbox).
# ────────────────────────────────────────────────────────────────────────────
step_07_app_debs() {
    init_step "07-app-debs"
    locate_bundle

    step "1. VS Code"
    if [[ -f "$BUNDLE_DIR/apps/vscode.deb" ]]; then
        log "Installing VS Code (apt install ./)"
        apt-get install -y "$BUNDLE_DIR/apps/vscode.deb" || warn "VS Code install failed."
        command -v code >/dev/null && log "VS Code: $(code --version 2>/dev/null | head -1)" \
            || warn "VS Code installed but 'code' not on PATH."
    else
        warn "apps/vscode.deb not found; skipping."
    fi

    step "2. Google Chrome"
    if [[ -f "$BUNDLE_DIR/apps/chrome.deb" ]]; then
        log "Installing Google Chrome (apt install ./)"
        apt-get install -y "$BUNDLE_DIR/apps/chrome.deb" || warn "Chrome install failed."
        command -v google-chrome-stable >/dev/null \
            && log "Chrome: $(google-chrome-stable --version 2>/dev/null)" \
            || warn "Chrome installed but binary not in PATH."
    else
        warn "apps/chrome.deb not found; skipping."
    fi

    step "3. AppArmor profiles"
    if command -v aa-status >/dev/null 2>&1 && [[ -d /etc/apparmor.d ]]; then
        log "Reloading AppArmor profiles (registers Chrome/VS Code profiles)"
        systemctl reload apparmor 2>/dev/null || apparmor_parser -r /etc/apparmor.d/ 2>/dev/null || true
    fi

    step "4. Allow unprivileged user namespaces (Chrome/VS Code Electron sandbox)"
    if [[ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
        log "Disabling apparmor_restrict_unprivileged_userns"
        tee /etc/sysctl.d/60-apparmor-userns.conf > /dev/null <<'SYSCTL'
# Allow unprivileged user namespaces — Chrome/VS Code/Firefox sandbox.
# Set by step_07_app_debs (install-all-steps.sh) on Ubuntu 24.04+.
kernel.apparmor_restrict_unprivileged_userns = 0
SYSCTL
        sysctl --system >/dev/null 2>&1 \
            || sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 \
            || warn "Could not apply apparmor userns sysctl."
    fi

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 08: tarball-apps
#   Extract Firefox / Node.js / Bun / Opencode from the bundled tarballs.
#   Ends with checkpoint_reboot — needrestart restarted sshd/dbus/polkit etc.
# ────────────────────────────────────────────────────────────────────────────
step_08_tarball_apps() {
    init_step "08-tarball-apps"
    locate_bundle

    step "1. Firefox"
    local FF_TARBALL=""
    local c
    for c in firefox.tar.xz firefox.tar.bz2; do
        [[ -f "$BUNDLE_DIR/apps/$c" ]] && { FF_TARBALL="$BUNDLE_DIR/apps/$c"; break; }
    done
    if [[ -n "$FF_TARBALL" ]]; then
        local FF_VER
        FF_VER=$(cat "$BUNDLE_DIR/apps/firefox.version" 2>/dev/null || echo unknown)
        log "Installing Firefox $FF_VER to /opt/firefox"
        mkdir -p /opt/firefox
        local _ff_magic _flag
        _ff_magic=$(head -c 6 "$FF_TARBALL" | od -An -tx1 | tr -d ' \n')
        case "$_ff_magic" in
            fd377a585a00*) _flag="-xJf" ;;   # xz
            425a68*)       _flag="-xjf" ;;   # bz2
            1f8b*)         _flag="-xzf" ;;   # gz
            *)             _flag="" ;;
        esac
        if [[ -n "$_flag" ]] && tar "$_flag" "$FF_TARBALL" -C /opt/firefox --strip-components=1; then
            ln -sf /opt/firefox/firefox /usr/local/bin/firefox
            log "Firefox: $(/opt/firefox/firefox --version 2>/dev/null || echo OK)"
            tee /usr/share/applications/firefox-manual.desktop > /dev/null <<'EOF'
[Desktop Entry]
Name=Firefox
Comment=Web Browser
Exec=/opt/firefox/firefox %u
Icon=/opt/firefox/browser/chrome/icons/default/default128.png
Terminal=false
Type=Application
Categories=Network;WebBrowser;
MimeType=text/html;text/xml;application/xhtml+xml;x-scheme-handler/http;x-scheme-handler/https;
EOF
        else
            warn "Firefox extraction failed (magic=$_ff_magic)."
        fi
    else
        warn "apps/firefox.tar.{xz,bz2} not found; skipping."
    fi

    step "2. Node.js"
    if [[ -f "$BUNDLE_DIR/apps/nodejs.tar.xz" ]]; then
        local NODE_VER bin
        NODE_VER=$(cat "$BUNDLE_DIR/apps/nodejs.version" 2>/dev/null || echo unknown)
        log "Installing Node.js v$NODE_VER to /opt/nodejs"
        rm -rf /opt/nodejs
        mkdir -p /opt/nodejs
        if tar -xJf "$BUNDLE_DIR/apps/nodejs.tar.xz" -C /opt/nodejs --strip-components=1; then
            for bin in node npm npx; do
                ln -sf "/opt/nodejs/bin/$bin" "/usr/local/bin/$bin" || warn "Could not symlink $bin."
            done
            log "Node.js: $(node --version 2>/dev/null)  npm: $(npm --version 2>/dev/null)"
        else
            warn "Node.js extraction failed."
        fi
    else
        warn "apps/nodejs.tar.xz not found; skipping."
    fi

    step "3. Bun"
    if [[ -f "$BUNDLE_DIR/apps/bun-linux-x64.zip" ]]; then
        local BUN_TAG TMP_BUN
        BUN_TAG=$(cat "$BUNDLE_DIR/apps/bun.version" 2>/dev/null || echo unknown)
        log "Installing Bun $BUN_TAG"
        TMP_BUN=$(mktemp -d)
        if unzip -q "$BUNDLE_DIR/apps/bun-linux-x64.zip" -d "$TMP_BUN" \
            && [[ -x "$TMP_BUN/bun-linux-x64/bun" ]] \
            && install -m 0755 "$TMP_BUN/bun-linux-x64/bun" /usr/local/bin/bun; then
            ln -sf /usr/local/bin/bun /usr/local/bin/bunx
            log "Bun: $(bun --version 2>/dev/null)"
        else
            warn "Bun extraction/install failed."
        fi
        rm -rf "$TMP_BUN"
    else
        warn "apps/bun-linux-x64.zip not found; skipping."
    fi

    step "4. Opencode"
    if [[ -f "$BUNDLE_DIR/apps/opencode" ]]; then
        local OC_VER
        OC_VER=$(cat "$BUNDLE_DIR/apps/opencode.version" 2>/dev/null || echo unknown)
        log "Installing Opencode $OC_VER -> /usr/local/bin/opencode"
        install -m 0755 "$BUNDLE_DIR/apps/opencode" /usr/local/bin/opencode \
            && log "Opencode installed" \
            || warn "Opencode install failed."
    elif [[ -f "$BUNDLE_DIR/apps/opencode.MISSING" ]]; then
        warn "Opencode was not downloaded during gather. Place binary at /usr/local/bin/opencode manually."
    else
        warn "apps/opencode not found; skipping."
    fi

    step "5. needrestart -r a (auto-restart daemons holding old libs)"
    if command -v needrestart >/dev/null 2>&1; then
        NEEDRESTART_MODE=a needrestart -r a 2>&1 | tail -50 || warn "needrestart returned non-zero."
    else
        warn "needrestart not installed; skipping (libs may be stale until reboot)."
    fi

    mark_step_ok
    checkpoint_reboot "app debs + tarballs installed + needrestart fired; reboot to confirm daemons come up clean from boot, sshd accepts logins, nvidia.ko still loads"
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 09: desktop-xrdp
#   Configure xrdp to launch xfce4, set up polkit shutdown rules, open UFW
#   port 3389 if active. No-op if INSTALL_DESKTOP=0.
# ────────────────────────────────────────────────────────────────────────────
step_09_desktop_xrdp() {
    init_step "09-desktop-xrdp"
    detect_target_user

    if [[ "$INSTALL_DESKTOP" != "1" ]] || ! command -v xrdp >/dev/null 2>&1; then
        log "INSTALL_DESKTOP=$INSTALL_DESKTOP or xrdp not installed; skipping desktop config."
        mark_step_ok
        return 0
    fi

    step "1. xrdp startwm.sh"
    tee /etc/xrdp/startwm.sh > /dev/null <<'XRDPEOF'
#!/bin/sh
if [ -r /etc/default/locale ]; then
    . /etc/default/locale
    export LANG LANGUAGE
fi
# Mitigate xrdp #3248 (polkit prompts): autostart polkit-gnome agent in session.
if [ -x /usr/libexec/polkit-gnome-authentication-agent-1 ]; then
    /usr/libexec/polkit-gnome-authentication-agent-1 &
elif [ -x /usr/lib/policykit-1-gnome/polkit-gnome-authentication-agent-1 ]; then
    /usr/lib/policykit-1-gnome/polkit-gnome-authentication-agent-1 &
fi
exec startxfce4
XRDPEOF
    chmod +x /etc/xrdp/startwm.sh

    step "2. xrdp TLS cert group + service"
    adduser xrdp ssl-cert 2>/dev/null || true
    systemctl enable xrdp 2>/dev/null || true
    systemctl restart xrdp 2>/dev/null || warn "xrdp restart failed — run 'systemctl start xrdp' after reboot."
    log "xrdp listening on port 3389"

    step "3. Default xfce4 session for current + new users"
    echo "xfce4-session" | tee /etc/skel/.xsession > /dev/null
    if [[ -n "${SUDO_USER:-}" ]]; then
        su - "$SUDO_USER" -c "echo xfce4-session > ~/.xsession" 2>/dev/null || true
    fi

    step "4. polkit shutdown rule for sudo group"
    if [[ -d /usr/share/polkit-1/rules.d ]]; then
        tee /usr/share/polkit-1/rules.d/49-xfce-shutdown.rules > /dev/null <<'POLKIT'
polkit.addRule(function(action, subject) {
    if ((action.id == "org.freedesktop.login1.power-off" ||
         action.id == "org.freedesktop.login1.reboot") &&
        subject.isInGroup("sudo")) {
        return polkit.Result.YES;
    }
});
POLKIT
    fi

    step "5. UFW port 3389"
    if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
        ufw allow 3389/tcp 2>/dev/null && log "UFW: port 3389/tcp opened" || true
    fi

    log "Desktop setup complete. Connect via RDP to port 3389."
    mark_step_ok
    checkpoint_reboot "desktop + xrdp configured; reboot to confirm lightdm doesn't stall boot and xrdp starts on port 3389"
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 10: wheelhouse-manifests
#   Generate per-wheelhouse requirements.txt files. Pure file scan; airgap-safe.
# ────────────────────────────────────────────────────────────────────────────
step_10_wheelhouse_manifests() {
    init_step "10-wheelhouse-manifests"
    locate_bundle

    step "Wheelhouse manifests"
    generate_wheelhouse_requirements

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 11: venv-inference  (CPU-only RAG/FastAPI; no torch, no vLLM)
#   See CLAUDE.md "Inference venv is CPU-only" invariant. GPU inference runs
#   via llama.cpp's HTTP server (step 14).
# ────────────────────────────────────────────────────────────────────────────
step_11_venv_inference() {
    init_step "11-venv-inference"
    locate_bundle
    detect_target_user

    if [[ "$INSTALL_INFERENCE" != "1" ]]; then
        log "INSTALL_INFERENCE=0; skipping."
        mark_step_ok
        return 0
    fi

    local WHEELS_DIR="$BUNDLE_DIR/wheels/inference"
    if ! _wheelhouse_has_packages "$WHEELS_DIR"; then
        warn "wheels/inference/ empty; skipping."
        mark_step_ok
        return 0
    fi

    step "1. Create venv at $INFERENCE_PREFIX/venv"
    mkdir -p "$INFERENCE_PREFIX"
    chown "$TARGET_USER:$TARGET_GROUP" "$INFERENCE_PREFIX"
    _as_user "$PYTHON_BIN" -m venv "$INFERENCE_PREFIX/venv" || die "Could not create inference venv."

    local _PIP="$INFERENCE_PREFIX/venv/bin/pip"

    step "2. Bootstrap pip / wheel / setuptools"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
        || warn "Bootstrap pip install failed."

    step "3. Project requirements"
    local rf
    for rf in "$BUNDLE_DIR/requirements/llm_api.txt" "$BUNDLE_DIR/requirements/llm_api_full.txt"; do
        [[ -f "$rf" ]] || continue
        log "  Installing from $(basename "$rf")"
        _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
    done

    step "4. Core inference / RAG packages (CPU-only)"
    # NO torch / torchvision / torchaudio / vllm here — those caused multi-GPU
    # NCCL ABI skew when this venv coexisted with the training venv (#15525,
    # #20862, #28283). Inference workloads route through llama.cpp's HTTP server.
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
        sentence-transformers faiss-cpu rank-bm25 \
        transformers tokenizers safetensors huggingface-hub tiktoken \
        langchain langchain-core langchain-community langchain-ollama \
        langgraph langgraph-checkpoint langgraph-prebuilt langsmith \
        ollama tavily-python \
        fastapi uvicorn pydantic pydantic-settings sse-starlette \
        httpx httpx-sse aiohttp aiofiles websockets \
        passlib python-jose \
        PyMuPDF pypdf python-docx python-pptx openpyxl \
        pandas numpy Pillow python-dotenv python-multipart \
        jupyter_client ipykernel filelock tqdm rich 2>/dev/null || true

    step "5. Smoke test (CPU-only)"
    _as_user "$INFERENCE_PREFIX/venv/bin/python" - <<'PY' || warn "Inference smoke test failed."
import importlib
for mod in ("fastapi", "langchain", "sentence_transformers", "transformers", "tiktoken"):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {mod} {ver}")
    except Exception as e:
        print(f"  {mod}: import failed: {e}")
PY

    log "Inference venv ready (CPU-only RAG/FastAPI): $INFERENCE_PREFIX/venv"
    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 12: venv-training
#   Training venv with torch (cu130), PyG + extensions, scientific stack.
# ────────────────────────────────────────────────────────────────────────────
step_12_venv_training() {
    init_step "12-venv-training"
    locate_bundle
    detect_target_user

    if [[ "$INSTALL_TRAINING" != "1" ]]; then
        log "INSTALL_TRAINING=0; skipping."
        mark_step_ok
        return 0
    fi

    local WHEELS_DIR="$BUNDLE_DIR/wheels/training"
    if ! _wheelhouse_has_packages "$WHEELS_DIR"; then
        warn "wheels/training/ empty; skipping."
        mark_step_ok
        return 0
    fi

    step "1. Create venv at $TRAINING_PREFIX/venv"
    mkdir -p "$TRAINING_PREFIX"
    chown "$TARGET_USER:$TARGET_GROUP" "$TRAINING_PREFIX"
    _as_user "$PYTHON_BIN" -m venv "$TRAINING_PREFIX/venv" || die "Could not create training venv."

    local _PIP="$TRAINING_PREFIX/venv/bin/pip"

    step "2. Bootstrap pip / wheel / setuptools"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
        || warn "Bootstrap pip install failed."

    step "3. PyTorch (cu130) + PyG"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio \
        || warn "torch install failed."
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch-geometric \
        || warn "torch-geometric install failed."

    local pkg
    for pkg in pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv; do
        if _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" "$pkg" 2>/dev/null; then
            log "  $pkg: OK"
        else
            warn "  $pkg: not available in wheelhouse (expected for torch_spline_conv on cu130)"
        fi
    done

    step "4. Project requirements"
    local rf
    for rf in "$BUNDLE_DIR/requirements/meshgraphnets.txt" \
              "$BUNDLE_DIR/requirements/simulgen.txt" \
              "$BUNDLE_DIR/requirements/pemtron.txt" \
              "$BUNDLE_DIR/requirements/pemtron_transfer.txt"; do
        [[ -f "$rf" ]] || continue
        log "  Installing from $(basename "$rf")"
        _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
    done

    step "5. Core training/scientific stack"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
        numpy scipy h5py pandas tqdm matplotlib seaborn Pillow \
        scikit-learn scikit-image statsmodels networkx sympy \
        torchinfo tensorboard \
        opencv-python imageio librosa audiomentations soxr natsort \
        reportlab paramiko smbprotocol 2>/dev/null || true

    step "6. Smoke test: torch + PyG"
    _as_user "$TRAINING_PREFIX/venv/bin/python" - <<'PY' || warn "Training smoke test failed."
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
try:
    import torch_geometric
    print(f"  torch_geometric {torch_geometric.__version__}")
except Exception as e:
    print(f"  torch_geometric import failed: {e}")
if torch.cuda.is_available():
    print(f"  Device 0: {torch.cuda.get_device_name(0)}")
    print(f"  Device count: {torch.cuda.device_count()}")
PY

    log "Training venv ready: $TRAINING_PREFIX/venv"
    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 13: venv-jupyter
#   JupyterLab + data-science venv. Registers an ipykernel and drops a
#   start-jupyter.sh convenience launcher.
# ────────────────────────────────────────────────────────────────────────────
step_13_venv_jupyter() {
    init_step "13-venv-jupyter"
    locate_bundle
    detect_target_user

    if [[ "$INSTALL_JUPYTER" != "1" ]]; then
        log "INSTALL_JUPYTER=0; skipping."
        mark_step_ok
        return 0
    fi

    local WHEELS_DIR="$BUNDLE_DIR/wheels/jupyter"
    if ! _wheelhouse_has_packages "$WHEELS_DIR"; then
        warn "wheels/jupyter/ empty; skipping."
        mark_step_ok
        return 0
    fi

    step "1. Create venv at $JUPYTER_PREFIX/venv"
    mkdir -p "$JUPYTER_PREFIX"
    chown "$TARGET_USER:$TARGET_GROUP" "$JUPYTER_PREFIX"
    _as_user "$PYTHON_BIN" -m venv "$JUPYTER_PREFIX/venv" || die "Could not create jupyter venv."

    local _PIP="$JUPYTER_PREFIX/venv/bin/pip"

    step "2. Bootstrap pip / wheel / setuptools"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
        || warn "Bootstrap pip install failed."

    step "3. JupyterLab + data-science"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
        jupyterlab notebook ipykernel ipywidgets jupyter-server \
        pandas polars numpy scipy matplotlib seaborn plotly \
        scikit-learn statsmodels tqdm rich requests aiohttp \
        black ruff mypy pytest ipdb \
        || warn "Some Jupyter packages failed."

    step "4. Register kernel"
    _as_user "$JUPYTER_PREFIX/venv/bin/python" -m ipykernel install \
        --user --name "airgap-py${PYTHON_VER}" \
        --display-name "Python ${PYTHON_VER} (airgap)" 2>/dev/null || true

    step "5. Convenience launcher"
    if [[ -n "${SUDO_USER:-}" ]]; then
        local _SUDO_HOME
        _SUDO_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        cat > "$_SUDO_HOME/start-jupyter.sh" <<JEOF
#!/usr/bin/env bash
source "$JUPYTER_PREFIX/venv/bin/activate"
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser "\$@"
JEOF
        chmod +x "$_SUDO_HOME/start-jupyter.sh"
        chown "$TARGET_USER:$TARGET_GROUP" "$_SUDO_HOME/start-jupyter.sh"
        log "Jupyter launcher: $_SUDO_HOME/start-jupyter.sh"
    fi

    log "Jupyter venv ready: $JUPYTER_PREFIX/venv"
    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 14: llamacpp-build
#   Build llama.cpp from the bundled source against the system CUDA toolkit
#   installed by install-nvidia.sh.
#
#   Canonical cmake flags (2026-05 baseline):
#     -DGGML_CUDA=ON
#     -DCMAKE_CUDA_ARCHITECTURES=100-real;103-real   (B200=sm_100, B300=sm_103)
#     -DLLAMA_OPENSSL=ON                             (LLAMA_CURL deprecated #18922)
#     -DLLAMA_BUILD_UI=OFF
# ────────────────────────────────────────────────────────────────────────────
step_14_llamacpp_build() {
    init_step "14-llamacpp-build"
    locate_bundle
    source_bundle_metadata
    detect_target_user

    if [[ "$INSTALL_LLAMA" != "1" ]] || [[ ! -f "$BUNDLE_DIR/src/llama.cpp.tar.gz" ]]; then
        log "INSTALL_LLAMA=$INSTALL_LLAMA or src/llama.cpp.tar.gz missing; skipping."
        mark_step_ok
        return 0
    fi

    step "1. Extract llama.cpp source -> $LLAMA_PREFIX"
    rm -rf "$LLAMA_PREFIX"
    mkdir -p "$LLAMA_PREFIX"
    chown "$TARGET_USER:$TARGET_GROUP" "$LLAMA_PREFIX"
    _as_user tar -xzf "$BUNDLE_DIR/src/llama.cpp.tar.gz" -C "$LLAMA_PREFIX" --strip-components=1 \
        || die "llama.cpp source extraction failed."

    step "2. Locate nvcc"
    local NVCC_PATH=""
    local c
    for c in /usr/local/cuda/bin/nvcc "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin/nvcc"; do
        [[ -x "$c" ]] && { NVCC_PATH="$c"; break; }
    done
    [[ -n "$NVCC_PATH" ]] || die "nvcc not found under /usr/local/cuda* — install-nvidia.sh did not run?"
    log "nvcc: $NVCC_PATH"

    step "3. cmake configure"
    local CMAKE_ARGS=(
        -S "$LLAMA_PREFIX"
        -B "$LLAMA_PREFIX/build"
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_CUDA=ON
        -DGGML_NATIVE=OFF
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_LIST"
        -DCMAKE_CUDA_COMPILER="$NVCC_PATH"
        -DCUDAToolkit_ROOT=/usr/local/cuda
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=ON
        -DLLAMA_BUILD_SERVER=ON
        -DLLAMA_BUILD_UI=OFF
        -DLLAMA_OPENSSL=ON
    )

    log "cmake -S $LLAMA_PREFIX -B $LLAMA_PREFIX/build (arch=$CUDA_ARCH_LIST, jobs=$JOBS, nvcc=$NVCC_PATH)"
    _as_user cmake "${CMAKE_ARGS[@]}" \
        || die "cmake configure failed."

    step "4. cmake --build (parallel jobs=$JOBS)"
    _as_user cmake --build "$LLAMA_PREFIX/build" --config Release -j"$JOBS" \
        || die "llama.cpp build failed."

    log "llama.cpp built: $LLAMA_PREFIX/build/bin/llama-server"

    step "5. Python venv for convert_hf_to_gguf.py etc. (optional)"
    local LLAMA_WHEELS="$BUNDLE_DIR/wheels/llamacpp"
    if _wheelhouse_has_packages "$LLAMA_WHEELS"; then
        log "Creating llama.cpp Python venv at $LLAMA_PREFIX/venv"
        if _as_user "$PYTHON_BIN" -m venv "$LLAMA_PREFIX/venv"; then
            local _PIP="$LLAMA_PREFIX/venv/bin/pip"
            _as_user "$_PIP" install --no-index --find-links="$LLAMA_WHEELS" --upgrade pip wheel setuptools 2>/dev/null || true
            shopt -s nullglob
            local rf
            for rf in "$LLAMA_PREFIX"/requirements.txt "$LLAMA_PREFIX"/requirements/*.txt; do
                [[ -f "$rf" ]] || continue
                _as_user "$_PIP" install --no-index --find-links="$LLAMA_WHEELS" -r "$rf" 2>/dev/null || true
            done
            shopt -u nullglob
        fi
    else
        log "wheels/llamacpp/ empty — skipping llama.cpp utility venv"
    fi

    step "6. Smoke test"
    if "$LLAMA_PREFIX/build/bin/llama-cli" --version 2>&1 | head -3; then
        log "llama.cpp OK"
    else
        warn "llama-cli --version failed."
    fi

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 15: system-tuning
#   sysctl + Transparent Huge Pages + pam_limits tuning for multi-GPU LLM
#   workloads.
#
#   CUDA env wiring (/etc/profile.d/cuda.sh, /etc/ld.so.conf.d/cuda-system.conf)
#   lives in install-nvidia.sh — that's nvidia infrastructure, not userland.
# ────────────────────────────────────────────────────────────────────────────
step_15_system_tuning() {
    init_step "15-system-tuning"

    step "1. sysctl (vm.overcommit, vm.swappiness, vm.max_map_count, net buffers)"
    tee /etc/sysctl.d/99-llm-multigpu.conf > /dev/null <<'SYSCTL'
# Installed by step_15_system_tuning (install-all-steps.sh) — multi-GPU LLM workload tuning.
vm.overcommit_memory=1
vm.swappiness=0
vm.max_map_count=1048576
net.core.rmem_max=268435456
net.core.wmem_max=268435456
SYSCTL
    sysctl --system >/dev/null 2>&1 || warn "sysctl --system reload failed."
    log "sysctl applied"

    step "2. Transparent hugepages = madvise"
    tee /etc/systemd/system/disable-thp-defrag.service > /dev/null <<'UNIT'
[Unit]
Description=Set transparent_hugepage to madvise for LLM workloads
After=sysinit.target local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo madvise > /sys/kernel/mm/transparent_hugepage/enabled; echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag"
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload 2>/dev/null || true
    systemctl enable --now disable-thp-defrag.service >/dev/null 2>&1 \
        && log "THP = madvise" \
        || warn "Could not enable disable-thp-defrag.service."

    step "3. pam_limits (nofile, nproc, memlock, stack)"
    tee /etc/security/limits.d/99-llm-multigpu.conf > /dev/null <<'LIMITS'
*  soft  nofile   1048576
*  hard  nofile   1048576
*  soft  nproc    524288
*  hard  nproc    524288
*  soft  memlock  unlimited
*  hard  memlock  unlimited
*  soft  stack    65536
*  hard  stack    65536
LIMITS
    chmod 0644 /etc/security/limits.d/99-llm-multigpu.conf

    # ====================================================================
    # DO NOT WRITE /etc/systemd/system.conf.d/*.conf OR /etc/systemd/user.conf.d/*.conf
    # ====================================================================
    # These files set DefaultLimit*= for PID 1 and every systemd-managed service.
    # A typo here is unrecoverable without console access — there is no SSH route
    # back if you crash sshd at boot.
    #
    # Specific traps:
    #   1. UNIT MISMATCH. pam_limits and systemd use DIFFERENT units for the
    #      same resource:
    #        limits.conf  stack    65536   → 65536 KB  = 64 MB   (fine)
    #        systemd      LimitSTACK=65536 → 65536 B   = 64 KB   (catastrophic)
    #      LimitMEMLOCK/LimitDATA/LimitAS/LimitFSIZE all share the bytes-vs-KB trap.
    #   2. BLAST RADIUS. Per-service Limit*= affects ONE service. DefaultLimit*=
    #      in /etc/systemd/system.conf.d/ affects EVERY service (sshd, dbus,
    #      NetworkManager, polkit, fabricmanager).
    #   3. systemctl daemon-reexec APPLIES IT NOW. PID 1 re-execs with the new
    #      defaults immediately, so a bad value is live the moment the file is
    #      written.
    #
    # Right pattern: put Limit*= in the unit file of the service that needs it.
    # See /etc/systemd/system/llama-server@.service in step 16.
    if [[ -f /etc/systemd/system.conf.d/99-llm-multigpu.conf ]]; then
        rm -f /etc/systemd/system.conf.d/99-llm-multigpu.conf
        systemctl daemon-reexec 2>/dev/null || true
        log "Removed stale /etc/systemd/system.conf.d/99-llm-multigpu.conf (had broken DefaultLimitSTACK)"
    fi
    log "System limits applied (pam_limits only; per-service limits live in unit files)"

    mark_step_ok
    checkpoint_reboot "sysctl + THP + pam_limits staged; reboot to confirm clean boot with new tuning before installing ops tooling"
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 16: operational-tooling
#   Install helper scripts (gpu-health-check, llama-server-multigpu,
#   llama-model-preload) and the llama-server@.service systemd template.
# ────────────────────────────────────────────────────────────────────────────
step_16_operational_tooling() {
    init_step "16-operational-tooling"

    step "1. gpu-health-check"
    tee /usr/local/bin/gpu-health-check > /dev/null <<'HEALTH'
#!/usr/bin/env bash
# gpu-health-check — verify multi-GPU fabric is healthy.
# Installed by step_16_operational_tooling (install-all-steps.sh).
set -uo pipefail
rc=0
say()  { printf '\033[1;36m[health]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  PASS\033[0m  %s\n' "$*"; }
bad()  { printf '\033[1;31m  FAIL\033[0m  %s\n' "$*"; rc=1; }
warn() { printf '\033[1;33m  WARN\033[0m  %s\n' "$*"; }

say "1. GPU visibility"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    n=$(nvidia-smi -L | wc -l | tr -d ' ')
    ok "$n GPU(s) visible"
else
    bad "nvidia-smi cannot list GPUs"; exit 1
fi

say "2. nvidia-fabricmanager"
if (( n > 1 )); then
    state=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo unknown)
    [[ "$state" == "active" ]] && ok "active" \
        || bad "NOT active on $n-GPU NVSwitch box; fabric may be degraded"
fi

say "3. NVLink lane health"
status=$(nvidia-smi nvlink --status 2>/dev/null || true)
if [[ -n "$status" ]]; then
    down=$(printf '%s\n' "$status" | grep -ciE 'inactive|<inactive>|disabled' || true)
    up=$(printf '%s\n' "$status" | grep -c 'GB/s' || true)
    if (( down == 0 )) && (( up > 0 )); then ok "$up active lane(s), 0 inactive"
    elif (( down > 0 )); then bad "$down inactive NVLink lane(s)"
    else warn "no NVLink data reported"
    fi
fi

say "4. fabric.state per GPU"
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
total=$(printf '%s\n' "$fab" | grep -c . || true)
ok_n=$(printf '%s\n' "$fab" | grep -cE '^(Completed|Success)$' || true)
if (( total == 0 )); then
    if (( n > 1 )); then
        bad "no Fabric stanza in nvidia-smi -q on $n-GPU box; FM did not initialize NVSwitch"
    else
        warn "no Fabric stanza in nvidia-smi -q (single-GPU box)"
    fi
elif (( total != n )); then
    bad "matched $total Fabric entries for $n GPU(s); parser/driver mismatch"
elif (( ok_n == total )); then
    ok "fabric.state Completed on $ok_n/$total GPU(s)"
else
    bad "fabric.state $ok_n/$total Completed -- common cause of CUDA Error 802"
fi

echo
[[ $rc -eq 0 ]] && say "ALL CHECKS PASSED" || say "ONE OR MORE CHECKS FAILED"
exit $rc
HEALTH
    chmod 0755 /usr/local/bin/gpu-health-check
    log "Installed: gpu-health-check"

    step "2. llama-server-multigpu"
    tee /usr/local/bin/llama-server-multigpu > /dev/null <<LLAMAWRAP
#!/usr/bin/env bash
# llama-server-multigpu — NUMA + tensor-split wrapper for llama.cpp server.
# Installed by step_16_operational_tooling (install-all-steps.sh).
set -euo pipefail
LLAMA_BIN="\${LLAMA_BIN:-$LLAMA_PREFIX/build/bin/llama-server}"
[[ -x "\$LLAMA_BIN" ]] || { echo "llama-server not found at \$LLAMA_BIN" >&2; exit 1; }
N=\${LLAMA_N_GPUS:-\$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
extra_args=()
if (( N > 1 )) && ! printf '%s\n' "\$@" | grep -q -- '--tensor-split'; then
    split=\$(yes 1 | head -n "\$N" | paste -sd ',')
    extra_args+=( --tensor-split "\$split" )
    mode="\${LLAMA_SPLIT_MODE:-row}"
    if [[ "\$mode" != "none" ]] && ! printf '%s\n' "\$@" | grep -q -- '--split-mode'; then
        extra_args+=( --split-mode "\$mode" )
    fi
fi
prefix=()
if [[ "\${LLAMA_NO_NUMA:-0}" != "1" ]] && command -v numactl >/dev/null && command -v nvidia-smi >/dev/null; then
    pci=\$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    pci_short="\${pci#????}"
    node_file="/sys/bus/pci/devices/\${pci_short,,}/numa_node"
    if [[ -f "\$node_file" ]]; then
        node=\$(cat "\$node_file" 2>/dev/null || echo -1)
        [[ "\$node" -ge 0 ]] && prefix=(numactl --cpunodebind="\$node" --membind="\$node")
    fi
fi
echo "[llama-server-multigpu] \$N GPU(s); extra: \${extra_args[*]}; numa: \${prefix[*]:-none}" >&2
exec "\${prefix[@]}" "\$LLAMA_BIN" "\${extra_args[@]}" "\$@"
LLAMAWRAP
    chmod 0755 /usr/local/bin/llama-server-multigpu
    log "Installed: llama-server-multigpu"

    step "3. llama-model-preload"
    tee /usr/local/bin/llama-model-preload > /dev/null <<'PRELOAD'
#!/usr/bin/env bash
# llama-model-preload — pre-mmap GGUF model into page cache.
# Installed by step_16_operational_tooling (install-all-steps.sh).
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <model.gguf> [more.gguf ...]" >&2; exit 1; }
for f in "$@"; do
    [[ -r "$f" ]] || { echo "[preload] skip: $f (not readable)" >&2; continue; }
    sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
    sz_gb=$(awk -v s="$sz" 'BEGIN {printf "%.1f", s/(1024*1024*1024)}')
    printf '[preload] reading %s (%s GiB) -> page cache\n' "$f" "$sz_gb" >&2
    cat "$f" > /dev/null
done
echo "[preload] done; subsequent mmap reads will hit page cache" >&2
PRELOAD
    chmod 0755 /usr/local/bin/llama-model-preload
    log "Installed: llama-model-preload"

    step "4. llama-server@.service template"
    mkdir -p /etc/llama-server
    if [[ ! -f /etc/llama-server/example.env ]]; then
        tee /etc/llama-server/example.env > /dev/null <<'EXAMPLE'
# /etc/llama-server/<instance>.env
# Copy to <name>.env and customize, then: systemctl enable --now llama-server@<name>
MODEL=/opt/models/MODEL.gguf
HOST=0.0.0.0
PORT=8080
NGL=999
CTX=8192
EXTRA=
EXAMPLE
    fi
    tee /etc/systemd/system/llama-server@.service > /dev/null <<'UNIT'
[Unit]
Description=llama.cpp server (instance %i)
After=network-online.target nvidia-fabricmanager.service nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=exec
EnvironmentFile=/etc/llama-server/%i.env
ExecStartPre=/usr/local/bin/llama-model-preload ${MODEL}
ExecStart=/usr/local/bin/llama-server-multigpu \
    --model ${MODEL} \
    --host ${HOST} \
    --port ${PORT} \
    --n-gpu-layers ${NGL} \
    --ctx-size ${CTX} \
    ${EXTRA}
Restart=on-failure
RestartSec=10
LimitNOFILE=1048576
LimitMEMLOCK=infinity
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload 2>/dev/null || true
    log "Installed: llama-server@.service (configure /etc/llama-server/<name>.env)"

    mark_step_ok
}

# ────────────────────────────────────────────────────────────────────────────
# STEP 17: final-status
#   Clear the resume marker, chown -R $SCRATCH_ROOT, print final diagnostics.
#   Always fires checkpoint_reboot for the final cold-boot sanity reboot.
# ────────────────────────────────────────────────────────────────────────────
step_17_final_status() {
    init_step "17-final-status"
    detect_target_user

    step "1. Clear resume marker"
    rm -f "$RESUME_MARKER"

    step "2. chown -R $SCRATCH_ROOT"
    chown -R "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT" 2>/dev/null || true

    step "3. Final status"
    local REBOOT_REQUIRED=0
    if [[ -f /run/reboot-required ]]; then
        REBOOT_REQUIRED=1
        log "/run/reboot-required is set (post-install)."
        [[ -f /run/reboot-required.pkgs ]] && log "  triggered by: $(tr '\n' ' ' </run/reboot-required.pkgs)"
    fi

    printf '\n'
    printf '%s\n' "════════════════════════════════════════════════════════════════"
    printf '%s\n' "  INSTALL STEPS COMPLETED (per .ok markers under $STEPS_DIR)"
    printf '%s\n' "════════════════════════════════════════════════════════════════"
    printf '  Inference venv : %s/venv  (CPU-only RAG/FastAPI; no torch, no vLLM)\n' "$INFERENCE_PREFIX"
    printf '  Training venv  : %s/venv\n' "$TRAINING_PREFIX"
    printf '  Jupyter venv   : %s/venv\n' "$JUPYTER_PREFIX"
    printf '  llama-server   : %s/build/bin/llama-server\n' "$LLAMA_PREFIX"
    printf '\n'
    printf '  Logs dir       : %s\n' "$RUN_LOG_DIR"
    printf '  Run id         : %s\n' "$RUN_ID"
    printf '\n'

    printf 'Next steps:\n'
    printf '  bash test-nvidia.sh                            # nvidia stack still healthy?\n'
    printf '  bash test-all.sh                               # full userland verify\n'
    printf '  gpu-health-check                               # quick fabric sanity\n'
    printf '  %s/build/bin/llama-server --help                # serve a GGUF model\n' "$LLAMA_PREFIX"
    if [[ "$INSTALL_DESKTOP" == "1" ]]; then
        printf '  rdp connect — tcp 3389                         # remote desktop (xfce4)\n'
    fi
    printf '\n'

    mark_step_ok

    if (( REBOOT_REQUIRED )); then
        checkpoint_reboot "final phase complete AND /run/reboot-required is set — reboot then run test-nvidia.sh + test-all.sh"
    else
        checkpoint_reboot "final phase complete — reboot to confirm sysctl/THP/limits/ops units persist from cold boot, then run test-nvidia.sh + test-all.sh"
    fi
}
