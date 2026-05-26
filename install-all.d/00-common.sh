#!/usr/bin/env bash
# ============================================================================
# install-all.d/00-common.sh
#
#   Shared helpers + state for the install-all.d/NN-name.sh step scripts.
#   This file is **sourced**, never executed. Every step script does:
#
#       SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#       source "$SCRIPT_DIR/00-common.sh"
#       init_step "NN-name"
#       # ... step body ...
#       mark_step_ok
#
#   Provides:
#     - log/warn/die/step printing helpers (color, tee to per-step log)
#     - INSTALL_WARNINGS / INSTALL_ERRORS accumulators (file-backed so the
#       launcher can read them across step processes)
#     - Step status markers (/var/lib/install-all/steps/NN-name.{ok,failed})
#     - Per-step + per-run log file paths
#     - Bundle discovery + metadata sourcing (incl. CUDA_MAJOR/MINOR from the
#       nvidia bundle's meta/target.env when present)
#     - apt helpers (_apt_install, _apt_install_strict, _pkg_satisfied,
#       _normalize_pkg_name for the t64 transition)
#     - _wheelhouse_has_packages / generate_wheelhouse_requirements
#     - _as_user (drop privileges to $SUDO_USER for venv operations)
#     - _ver_int
#     - Trap registration for ERR / EXIT
#
#   Airgap mandate: no helper in this file may fetch from the internet.
# ============================================================================

# Guard against double-source.
[[ -n "${_INSTALL_ALL_COMMON_LOADED:-}" ]] && return 0
_INSTALL_ALL_COMMON_LOADED=1

# Force noninteractive apt for every child process. lightdm (and a few other
# debconf-using packages) will hang a batch install if a prompt slips through;
# -y alone does NOT suppress debconf questions.
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

# ── Paths ───────────────────────────────────────────────────────────────────
INSTALL_ALL_D_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$INSTALL_ALL_D_DIR/.." && pwd)"

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
# INSTALL_VLLM is intentionally NOT exposed. The inference venv is now a
# CPU-only RAG/FastAPI/langchain stack — no torch, no vLLM. See
# 11-venv-inference.sh for the current scope.

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
    # Best-effort: when running as root these exist; when sourced from a
    # non-root context (e.g. --list), state dirs may not yet exist.
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
    # Usage: init_step "NN-name"
    STEP_NAME="$1"
    STEP_LOG="$RUN_LOG_DIR/${STEP_NAME}.log"
    _ensure_state_dirs

    # Redirect stdout+stderr through tee so console AND log capture.
    # Each step gets its own log; the launcher tees its own banner separately.
    if [[ -d "$RUN_LOG_DIR" ]]; then
        exec > >(tee -a "$STEP_LOG") 2>&1
    fi

    printf '%s[step]%s %s starting (run %s) — log %s\n' \
        "$_C_INFO" "$_C_OFF" "$STEP_NAME" "$RUN_ID" "$STEP_LOG"

    # Remove any prior .failed marker; .ok marker is removed only when the step
    # actually runs (re-runs without --force/--rerun should skip via the
    # launcher, not delete an existing .ok here).
    rm -f "$STEPS_DIR/${STEP_NAME}.failed" 2>/dev/null || true

    # On error/exit, mark .failed if mark_step_ok wasn't called. Trap matches
    # install-all.sh's _on_err / _on_exit semantics.
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
    if (( rc != 0 )) && [[ -d "$STEPS_DIR" ]]; then
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
    # Sets BUNDLE_DIR (extracted layout with debs/ + apps/). Idempotent.
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
    # Userland bundle metadata.
    if [[ -f "$BUNDLE_DIR/meta/target.env" ]]; then
        # shellcheck disable=SC1091
        source "$BUNDLE_DIR/meta/target.env"
    fi
    # NVIDIA bundle metadata, if present at a known location. Sets CUDA_MAJOR,
    # CUDA_MINOR via BUNDLE_CUDA. We deliberately don't fail if it's missing
    # since the userland bundle is installable even on hand-installed CUDA.
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
    # Final fallback for CUDA_MAJOR/MINOR — matches the CLAUDE.md invariant.
    export CUDA_MAJOR="${CUDA_MAJOR:-13}"
    export CUDA_MINOR="${CUDA_MINOR:-0}"
}

# ── Target user identity ────────────────────────────────────────────────────
detect_target_user() {
    TARGET_USER="${SUDO_USER:-$USER}"
    TARGET_GROUP="$(id -gn "$TARGET_USER" 2>/dev/null || echo "$TARGET_USER")"
    export TARGET_USER TARGET_GROUP
}

# Run an arbitrary command as $TARGET_USER (drops privileges from root).
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
