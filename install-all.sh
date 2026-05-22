#!/usr/bin/env bash
# ============================================================================
# install-all.sh  (userland variant)
#
#   Run on the airgapped Ubuntu 24.04 server AFTER install-nvidia.sh has
#   installed the NVIDIA stack (R580 LTS driver + CUDA 13.0 + FabricManager
#   + NVLSM + NCCL) and the box has been rebooted. Place this next to
#   all-airgap-bundle-ubuntu24.04.bin (+ pre-install-check.sh) and run as root.
#
#   Full sequence on the target:
#     1. sudo bash pre-install-nvidia.sh   # NVIDIA bundle readiness
#     2. sudo bash install-nvidia.sh       # driver + FM + NVLSM + CUDA + NCCL
#     3. sudo reboot                       # mandatory ??load nvidia.ko
#     4. sudo bash test-nvidia.sh          # fabric Completed?
#     5. sudo bash pre-install-check.sh    # userland readiness (this preflight)
#     6. sudo bash install-all.sh          # ??THIS script (userland)
#     7. sudo bash test-all.sh             # post-install verification
#
#   This installer does NOT touch the NVIDIA stack ??no driver install, no
#   kernel module rebuild, no FabricManager setup. install-nvidia.sh and its
#   apt-mark hold already protect those. This script only:
#     - apt-installs ~25 userland packages from the bundle's local repo
#     - installs VS Code / Chrome / Firefox / Node.js / Bun / Opencode
#     - creates three Python venvs (inference / training / jupyter)
#     - builds llama.cpp against the vendor's CUDA toolkit
#     - configures xfce4 + xrdp for remote desktop (optional)
#     - installs operational tooling (gpu-health-check, llama-server@.service)
#     - applies kernel/limits tuning for B300 NVSwitch workloads
#
#   Conditional two-stage: an apt --simulate dry-run checks whether the install
#   would upgrade libc6/systemd/dbus/linux-image-* (the only packages on
#   Ubuntu 24.04 that write /run/reboot-required). If so, those triggering
#   packages install first, a resume marker is written, and the script exits
#   asking for a reboot. After reboot, re-running install-all.sh picks up at
#   the userland phase. Otherwise: single-pass, ~10 min, no reboot.
#
#   Usage:
#     sudo bash install-all.sh                       # full single-pass
#     sudo bash install-all.sh --skip-preflight     # skip pre-install-check.sh
#     INSTALL_DESKTOP=0 sudo bash install-all.sh    # SSH only, no xfce4/xrdp
#     INSTALL_JUPYTER=0 sudo bash install-all.sh    # skip Jupyter venv
#
#   Resume after a triggered reboot is automatic ??just re-run install-all.sh.
# ============================================================================
set -Eeuo pipefail

# Force noninteractive apt for every child process. Some packages (notably
# lightdm, which prompts to pick a default display manager via debconf) will
# silently hang an otherwise-batch install if a prompt slips through. -y alone
# does NOT suppress debconf questions.
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export NEEDRESTART_SUSPEND=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
INSTALL_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
INSTALL_LOG="${INSTALL_LOG:-$SCRIPT_DIR/install-all-$RUN_STAMP.log}"
DIAG_LOG="${DIAG_LOG:-$SCRIPT_DIR/install-diagnostics-$RUN_STAMP.log}"

# Conditional-reboot state marker. If present at startup we're resuming
# after a reboot triggered by Stage 1.
RESUME_MARKER="${RESUME_MARKER:-/var/lib/install-all-prepped/stage1.done}"
STAGE_MARKER_DIR="$(dirname "$RESUME_MARKER")"

# ?? CLI ?????????????????????????????????????????????????????????????????????
SKIP_PREFLIGHT=0
FORCE=0
while (( $# > 0 )); do
    case "$1" in
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# ?? Defaults ????????????????????????????????????????????????????????????????
BUNDLE_DIR="${BUNDLE_DIR:-$SCRIPT_DIR}"
BUNDLE_BIN="${BUNDLE_BIN:-}"
PYTHON_VER="${PYTHON_VER:-3.12}"
PYTHON_BIN="${PYTHON_BIN:-python${PYTHON_VER}}"

# Install prefixes default to /scratch (shared, no $HOME dependency).
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

# B300 = sm_103 (Blackwell Ultra). 100 covers B200 if mixed fleet.
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-100;103}"
BUILD_BLAS="${BUILD_BLAS:-1}"
JOBS="${JOBS:-$(nproc)}"

APT_REPO_DIR="${APT_REPO_DIR:-/var/tmp/airgap-bundle-debs}"

INSTALL_WARNINGS=()
INSTALL_ERRORS=()
INSTALL_STATUS="running"
FINAL_DIAGNOSTICS_PRINTED=0

# ?? Transcript ??????????????????????????????????????????????????????????????
start_transcript_log() {
    local log_dir
    log_dir="$(dirname "$INSTALL_LOG")"
    if ! mkdir -p "$log_dir" 2>/dev/null || ! touch "$INSTALL_LOG" 2>/dev/null; then
        INSTALL_LOG="/tmp/install-all-$RUN_STAMP.log"
        DIAG_LOG="/tmp/install-diagnostics-$RUN_STAMP.log"
        touch "$INSTALL_LOG" || { echo "cannot create $INSTALL_LOG" >&2; exit 1; }
    fi
    exec > >(tee -a "$INSTALL_LOG") 2>&1
    printf '[install] Full transcript log: %s\n' "$INSTALL_LOG"
    printf '[install] Diagnostics summary: %s\n' "$DIAG_LOG"
}
start_transcript_log

# ?? Helpers ?????????????????????????????????????????????????????????????????
log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { INSTALL_WARNINGS+=( "$*" ); printf '\033[1;33m[install:WARN]\033[0m %s\n' "$*"; }
die()  { INSTALL_ERRORS+=( "$*" ); printf '\033[1;31m[install:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;35m== %s ==\033[0m\n' "$*"; }

print_final_diagnostics() {
    local rc="${1:-0}" finished overall cause diag_dir diag_target
    (( FINAL_DIAGNOSTICS_PRINTED )) && return 0
    FINAL_DIAGNOSTICS_PRINTED=1

    finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "$INSTALL_STATUS" == "stage1-reboot-required" ]]; then
        overall="NEEDS REBOOT"
        cause="Stage 1 installed reboot-triggering packages. Reboot, then re-run install-all.sh."
    elif (( rc != 0 || ${#INSTALL_ERRORS[@]} > 0 )); then
        overall="FAILED"
        cause="A required installer step failed. Read the transcript near the first ERROR line."
    elif (( ${#INSTALL_WARNINGS[@]} > 0 )); then
        overall="NEEDS ATTENTION"
        cause="Install reached the end, but warning-only optional steps need review."
    elif [[ -f /run/reboot-required ]]; then
        overall="OK - REBOOT RECOMMENDED"
        cause="/run/reboot-required is set. Reboot before final production validation."
    else
        overall="OK"
        cause="No installer errors, no warnings, and no reboot-required flag was detected."
    fi

    diag_dir="$(dirname "$DIAG_LOG")"
    diag_target="$DIAG_LOG"
    if ! mkdir -p "$diag_dir" 2>/dev/null || ! touch "$diag_target" 2>/dev/null; then
        diag_target="/tmp/install-diagnostics-$RUN_STAMP.log"
    fi

    {
        printf 'install-all.sh diagnostics\n'
        printf '%s\n' '=========================='
        printf 'Overall : %s\n' "$overall"
        printf 'Cause   : %s\n' "$cause"
        printf 'Exit rc : %s\n' "$rc"
        printf 'Started : %s\n' "$INSTALL_STARTED_AT"
        printf 'Finished: %s\n' "$finished"
        printf 'Status  : %s\n' "$INSTALL_STATUS"
        printf 'Log     : %s\n' "$INSTALL_LOG"
        printf 'Diag    : %s\n' "$diag_target"
        printf '\n'
        printf 'Install targets\n'
        printf '%s\n' '---------------'
        printf 'Inference venv : %s/venv\n' "$INFERENCE_PREFIX"
        printf 'Training venv  : %s/venv\n' "$TRAINING_PREFIX"
        printf 'Jupyter venv   : %s/venv\n' "$JUPYTER_PREFIX"
        printf 'llama-server   : %s/build/bin/llama-server\n' "$LLAMA_PREFIX"
        printf '\n'
        printf 'Warnings (%d)\n' "${#INSTALL_WARNINGS[@]}"
        printf '%s\n' '------------'
        if (( ${#INSTALL_WARNINGS[@]} == 0 )); then
            printf '[OK] No warnings recorded.\n'
        else
            local w
            for w in "${INSTALL_WARNINGS[@]}"; do printf '[WARN] %s\n' "$w"; done
        fi
        printf '\n'
        printf 'Errors (%d)\n' "${#INSTALL_ERRORS[@]}"
        printf '%s\n' '----------'
        if (( ${#INSTALL_ERRORS[@]} == 0 )); then
            printf '[OK] No errors recorded.\n'
        else
            local e
            for e in "${INSTALL_ERRORS[@]}"; do printf '[BAD] %s\n' "$e"; done
        fi
        printf '\n'
        printf 'Follow-up commands\n'
        printf '%s\n' '------------------'
        if [[ "$INSTALL_STATUS" == "stage1-reboot-required" ]]; then
            printf '1. sudo reboot\n'
            printf '2. sudo bash %s\n' "$0"
        else
            printf '1. bash test-all.sh\n'
            printf '2. gpu-health-check\n'
            if [[ -f /run/reboot-required ]]; then
                printf '3. sudo reboot\n'
            fi
        fi
    } | tee "$diag_target" || true
}

_on_exit() {
    local rc=$?
    print_final_diagnostics "$rc" || true
    return "$rc"
}

_on_err() {
    local rc=$?
    local cmd="${BASH_COMMAND:-unknown command}"
    INSTALL_ERRORS+=( "command failed (exit $rc): $cmd" )
    return "$rc"
}
trap _on_err ERR
trap _on_exit EXIT

_pkg_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

# t64 transition mapping (Ubuntu 24.04). Some Chrome/VS Code deps name the
# pre-t64 package; this maps the old name to its t64-suffixed equivalent.
_pkg_satisfied() {
    local pkg="$1" alt=""
    _pkg_installed "$pkg" && return 0
    case "$pkg" in
        libglib2.0-0)      alt="libglib2.0-0t64" ;;
        libatk1.0-0)       alt="libatk1.0-0t64" ;;
        libatk-bridge2.0-0) alt="libatk-bridge2.0-0t64" ;;
        libcups2)          alt="libcups2t64" ;;
        libgtk-3-0)        alt="libgtk-3-0t64" ;;
        libasound2)        alt="libasound2t64" ;;
    esac
    [[ -n "$alt" ]] && _pkg_installed "$alt"
}

_wheelhouse_has_packages() {
    local d="$1"
    [[ -d "$d" ]] || return 1
    compgen -G "$d/*.whl" >/dev/null \
        || compgen -G "$d/*.tar.gz" >/dev/null \
        || compgen -G "$d/*.tgz" >/dev/null \
        || compgen -G "$d/*.zip" >/dev/null
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

# Helper: integer encoding of x.y.z for version comparisons.
_ver_int() {
    local v="$1" maj min pat
    IFS='.' read -r maj min pat <<<"$v"
    printf '%d' $(( 10#${maj:-0}*1000000 + 10#${min:-0}*1000 + 10#${pat:-0} ))
}

# ============================================================================
# 0. ROOT / BUNDLE / RESUME DETECTION
# ============================================================================

if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        log "Not running as root; re-exec'ing under sudo."
        exec sudo -E bash "$0" "$@"
    fi
    die "Must run as root."
fi

# Auto-extract bundle if needed.
if [[ ! -d "$BUNDLE_DIR/debs" || ! -d "$BUNDLE_DIR/apps" ]]; then
    if [[ -z "$BUNDLE_BIN" ]]; then
        shopt -s nullglob
        _bins=( "$SCRIPT_DIR"/all-airgap-bundle-ubuntu*.bin )
        (( ${#_bins[@]} > 0 )) && BUNDLE_BIN="${_bins[0]}"
        shopt -u nullglob
    fi
    [[ -f "$BUNDLE_BIN" ]] || die "No extracted bundle and no .bin found. Set BUNDLE_DIR or BUNDLE_BIN."

    # Verify bundle SHA256 against sidecar before extracting ??if the file is
    # corrupt, fail fast rather than spending 10 minutes on a broken install.
    if [[ -f "${BUNDLE_BIN}.sha256" ]]; then
        log "Verifying bundle SHA256 against sidecar"
        ( cd "$(dirname "$BUNDLE_BIN")" && sha256sum -c "$(basename "$BUNDLE_BIN").sha256" ) \
            || die "Bundle SHA256 mismatch ??$BUNDLE_BIN is corrupt or doesn't match sidecar."
    else
        warn "No .sha256 sidecar at ${BUNDLE_BIN}.sha256 ??skipping integrity verification."
    fi

    log "Extracting $BUNDLE_BIN -> $SCRIPT_DIR"
    tar -xf "$BUNDLE_BIN" -C "$SCRIPT_DIR" || die "Failed to extract $BUNDLE_BIN"

    shopt -s nullglob
    for cand in "$SCRIPT_DIR"/*/; do
        if [[ -d "${cand}debs" && -d "${cand}apps" ]]; then
            BUNDLE_DIR="${cand%/}"
            break
        fi
    done
    shopt -u nullglob
    [[ -d "$BUNDLE_DIR/debs" && -d "$BUNDLE_DIR/apps" ]] \
        || die "Bundle extracted but no debs/+apps/ found under $SCRIPT_DIR"
    log "Using bundle: $BUNDLE_DIR"
fi

# Read bundle metadata + variant guard.
if [[ -f "$BUNDLE_DIR/meta/target.env" ]]; then
    # shellcheck disable=SC1091
    source "$BUNDLE_DIR/meta/target.env"
    log "Bundle built: ${BUNDLE_DATE:-?}  (variant=${BUNDLE_VARIANT:-?}, target=ubuntu ${BUNDLE_OS_VERSION:-?})"

    if [[ "${BUNDLE_VARIANT:-}" != "prepped" ]]; then
        if (( FORCE )); then
            warn "Bundle variant is '${BUNDLE_VARIANT:-<unset>}' (expected 'prepped'); proceeding due to --force."
        else
            die "Bundle variant is '${BUNDLE_VARIANT:-<unset>}', expected 'prepped'. This bundle is for a bare-metal server; install Ubuntu_offline_setup/install-all.sh instead. Use --force to override."
        fi
    fi

    . /etc/os-release
    [[ "$ID" == "${BUNDLE_OS_ID:-}" && "$VERSION_ID" == "${BUNDLE_OS_VERSION:-}" ]] \
        || warn "OS mismatch: target $ID/$VERSION_ID vs bundle ${BUNDLE_OS_ID:-?}/${BUNDLE_OS_VERSION:-?}"
else
    warn "$BUNDLE_DIR/meta/target.env missing; skipping variant guard."
fi

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_GROUP="$(id -gn "$TARGET_USER" 2>/dev/null || echo "$TARGET_USER")"
log "Install identity: $TARGET_USER:$TARGET_GROUP (running uid=$EUID)"

# Stage marker dir
mkdir -p "$STAGE_MARKER_DIR"

# ============================================================================
# 1. PRE-FLIGHT GATE
# ============================================================================
step "1. Pre-flight gate"

if (( SKIP_PREFLIGHT )); then
    warn "Skipping pre-install-check.sh per --skip-preflight"
elif [[ -r "$SCRIPT_DIR/pre-install-check.sh" || -r "$BUNDLE_DIR/pre-install-check.sh" ]]; then
    PREFLIGHT="$SCRIPT_DIR/pre-install-check.sh"
    [[ -r "$PREFLIGHT" ]] || PREFLIGHT="$BUNDLE_DIR/pre-install-check.sh"
    log "Running $PREFLIGHT"
    preflight_args=( --bundle "${BUNDLE_BIN:-$BUNDLE_DIR}" )
    (( FORCE )) && preflight_args+=( --force )
    if bash "$PREFLIGHT" "${preflight_args[@]}"; then
        log "Pre-flight passed."
    else
        rc=$?
        if (( FORCE )); then
            warn "Pre-flight failed (rc=$rc) but --force was given; proceeding anyway."
        else
            die "Pre-flight failed (rc=$rc). Fix the RED findings above, or re-run with --force/--skip-preflight."
        fi
    fi
else
    warn "pre-install-check.sh not found. Proceeding without pre-flight gate ??RECOMMEND running it first."
fi

# ============================================================================
# 2. SCRATCH ROOT
# ============================================================================
step "2. Scratch directory"

if [[ ! -d "$SCRATCH_ROOT" ]]; then
    mkdir -p "$SCRATCH_ROOT" || die "Could not create $SCRATCH_ROOT"
fi
chown "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT" || warn "chown $SCRATCH_ROOT failed."
chmod 0775 "$SCRATCH_ROOT" 2>/dev/null || true
log "$SCRATCH_ROOT ready (owner $TARGET_USER:$TARGET_GROUP)"

# ============================================================================
# 3. APT ??defensive holds, pinning, local repo setup
# ============================================================================
step "3. APT setup: holds, pinning, local repo"

# Hold every nvidia-*/cuda-*/libnvidia-* package the vendor installed so apt
# can never accidentally upgrade or remove them while we install our userland.
# These holds are intentionally persistent; install-nvidia.sh already treats
# them as part of the NVIDIA stack contract.
#
# Also hold the system libraries the NVIDIA driver + CUDA stack link against
# (libstdc++6, libgcc-s1, libgomp1, libc6). Without these holds, the userland
# install can silently DOWNGRADE these libs via --allow-downgrades when the
# bundle ships older versions than what the vendor's CUDA toolkit pulled in,
# which breaks nvidia-smi / NVML at runtime.
#
# dpkg-query exits non-zero if ANY of the glob patterns matches zero installed
# packages (e.g. libcudnn* when cuDNN comes from PyTorch wheels instead of the
# system). With `set -Eeuo pipefail` that aborts the script before useful work,
# so swallow dpkg-query's exit code locally — we only care about its output.
nvidia_pkgs=$({ dpkg-query -W -f='${Package} ${Status}\n' \
    'nvidia-*' 'cuda-*' 'libnvidia-*' 'libcudart*' 'libcublas*' 'libcudnn*' \
    'libcurand*' 'libcufft*' 'libcusparse*' 'libcusolver*' 'libnpp*' \
    'libstdc++6' 'libgcc-s1' 'libgomp1' 'libc6' \
    2>/dev/null || true; } \
    | awk '$2 == "install" && $3 == "ok" && $4 == "installed" {print $1}' \
    | sort -u)
if [[ -n "$nvidia_pkgs" ]]; then
    log "Placing apt-mark hold on $(printf '%s\n' "$nvidia_pkgs" | wc -l) NVIDIA/CUDA + runtime-lib packages"
    # shellcheck disable=SC2086
    apt-mark hold $nvidia_pkgs >/dev/null 2>&1 || warn "apt-mark hold reported errors (non-fatal)."
    # Stash list for audit/debug. Do not unhold these automatically.
    printf '%s\n' "$nvidia_pkgs" > "$STAGE_MARKER_DIR/nvidia-held.txt"
fi

# APT pinning: keep NVIDIA-published packages flowing from origin=NVIDIA only.
log "Installing APT pinning: /etc/apt/preferences.d/99-nvidia-prefer-origin"
tee /etc/apt/preferences.d/99-nvidia-prefer-origin > /dev/null <<'PIN'
# Installed by install-all.sh. Prevents the local file:// bundle repo from
# providing nvidia-*/cuda-* packages (vendor's online repo always wins).
Package: nvidia-* cuda-* libnvidia-* libcudart* libcublas* libcudnn* libcurand* libcufft* libcusparse* libcusolver* libnpp*
Pin: origin "developer.download.nvidia.com"
Pin-Priority: 1001
PIN

# Local apt repo from bundle's debs/. Use file:// scheme; apt prefers higher
# pin-priority sources for upgrades.
log "Setting up local apt repo: $APT_REPO_DIR"
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

tee /etc/apt/sources.list.d/00-bundle.list > /dev/null <<EOF
# Installed by install-all.sh ??local airgap bundle apt repo.
deb [trusted=yes] file://$APT_REPO_DIR ./
EOF

log "Running apt-get update"
apt-get update -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 \
    || warn "apt-get update reported errors (vendor's NVIDIA repo may be unreachable on airgap ??OK)."

# ============================================================================
# 4. APT DRY-RUN GATE ??decide single-pass vs conditional two-stage
# ============================================================================
step "4. APT dry-run gate (decide reboot-required path)"

# Read the package list that gather-all.sh stamped into the bundle.
APT_PKGS_FILE="$BUNDLE_DIR/meta/apt-packages.txt"
[[ -f "$APT_PKGS_FILE" ]] || die "$APT_PKGS_FILE not found in bundle."

# Build the install list, mapping pre-t64 names to Ubuntu 24.04 package names
# so older bundles do not ask apt for unresolvable virtual names.
REQUESTED_PKGS=()
while IFS= read -r pkg; do
    [[ -n "$pkg" && "$pkg" != \#* ]] || continue
    REQUESTED_PKGS+=( "$(_normalize_pkg_name "$pkg")" )
done < "$APT_PKGS_FILE"
(( ${#REQUESTED_PKGS[@]} > 0 )) || die "$APT_PKGS_FILE did not contain any installable package names."

# Reboot-triggering packages: only these write /run/reboot-required on Ubuntu 24.04.
#   USERLAND set: libc6/systemd/dbus — safe to upgrade and reboot through. These
#     don't touch the kernel, so NVIDIA DKMS modules continue to load fine on
#     the next boot.
#   DKMS-DANGER set: linux-image-*/linux-headers-*/linux-firmware/microcode —
#     these require NVIDIA DKMS to rebuild nvidia.ko against the new kernel.
#     In an airgapped bundle that's not guaranteed to succeed. If apt wants to
#     pull any of these, we ABORT and ask the user to handle it manually rather
#     than silently boot into a kernel without NVIDIA modules.
REBOOT_TRIGGERS_REGEX='^(libc6|libc6-dev|systemd|systemd-sysv|dbus|dbus-daemon)$'
DKMS_DANGER_REGEX='^(linux-image-.*|linux-headers-.*|linux-firmware|microcode|intel-microcode|amd64-microcode)$'

if [[ -f "$RESUME_MARKER" ]]; then
    log "Resume marker found: $RESUME_MARKER (resuming after Stage 1 reboot)"
    REBOOT_TRIGGERS_NEEDED=0
else
    log "Running apt dry-run to detect kernel/libc/systemd/dbus upgrades..."
    SIMULATE_OUT=$(mktemp)
    # The dry-run's only job here is to surface reboot-triggering upgrades.
    # If apt reports unmet deps for unrelated packages (e.g. gedit pulling in
    # libenchant-2-2 whose hunspell/aspell deps aren't in the airgap bundle),
    # that's a problem for those specific packages — handled per-package by
    # _apt_install below — not a reason to abort the whole install. Capture
    # whatever output apt produces and grep for reboot triggers regardless.
    if ! apt-get install -s -y --no-install-recommends --allow-downgrades \
            -o APT::Get::Show-Versions=1 \
            "${REQUESTED_PKGS[@]}" 2>&1 | tee "$SIMULATE_OUT" > /dev/null; then
        warn "apt dry-run reported unmet dependencies (see last 20 lines below); will retry per-package during install."
        tail -20 "$SIMULATE_OUT" >&2 || true
    fi

    # apt -s lines look like:
    #   Inst libc6 [2.38-3ubuntu1] (2.39-0ubuntu8.6 Ubuntu:24.04) []
    PROPOSED_PKGS=$(grep -E '^(Inst|Conf) ' "$SIMULATE_OUT" \
        | awk '{print $2}' \
        | sort -u)

    # Refuse to proceed if apt wants to pull in a new kernel/headers/firmware.
    # install-nvidia.sh holds nvidia-* but not linux-image-*, so an unguarded
    # upgrade would boot into a kernel without nvidia.ko. Force the user to
    # decide: either accept the DKMS rebuild risk by holding the kernel
    # themselves and re-running, or skip the trigger.
    DKMS_DANGER_HITS=$(printf '%s\n' "$PROPOSED_PKGS" | grep -E "$DKMS_DANGER_REGEX" || true)
    if [[ -n "$DKMS_DANGER_HITS" ]] && (( FORCE == 0 )); then
        printf '\n\033[1;31m================================================================\033[0m\n'
        printf '\033[1;31m  KERNEL / FIRMWARE / MICROCODE UPGRADE DETECTED\033[0m\n'
        printf '\033[1;31m================================================================\033[0m\n'
        printf '  Apt wants to upgrade the following packages:\n'
        printf '    %s\n' $DKMS_DANGER_HITS
        printf '\n  These would replace the running kernel that NVIDIA DKMS built\n'
        printf '  nvidia.ko against. On reboot the new kernel would load WITHOUT\n'
        printf '  NVIDIA modules; nvidia-fabricmanager.service would fail.\n'
        printf '\n  To proceed safely, choose one of:\n'
        printf '    A) Hold the kernel and re-run:\n'
        printf '         sudo apt-mark hold %s\n' $DKMS_DANGER_HITS
        printf '         sudo bash %s\n' "$0"
        printf '    B) Accept the DKMS rebuild risk (you must verify nvidia.ko\n'
        printf '       rebuilds against the new kernel before rebooting):\n'
        printf '         sudo bash %s --force\n' "$0"
        printf '\n'
        die "Refusing to upgrade kernel/firmware while NVIDIA driver is held."
    elif [[ -n "$DKMS_DANGER_HITS" ]]; then
        warn "--force set; allowing kernel/firmware upgrade. Verify DKMS rebuilds nvidia.ko before reboot."
        # Fold DKMS-danger entries into the Stage 1 install set since the user opted in.
        REBOOT_TRIGGERS_REGEX='^(libc6|libc6-dev|systemd|systemd-sysv|dbus|dbus-daemon|linux-image-.*|linux-headers-.*|linux-firmware|microcode|intel-microcode|amd64-microcode)$'
    fi

    REBOOT_TRIGGER_HITS=$(printf '%s\n' "$PROPOSED_PKGS" \
        | grep -E "$REBOOT_TRIGGERS_REGEX" \
        || true)

    if [[ -n "$REBOOT_TRIGGER_HITS" ]]; then
        REBOOT_TRIGGERS_NEEDED=1
        log "Apt dry-run found reboot-triggering upgrades:"
        printf '    %s\n' $REBOOT_TRIGGER_HITS
    else
        REBOOT_TRIGGERS_NEEDED=0
        log "Apt dry-run: no kernel/libc/systemd/dbus upgrades. Single-pass install."
    fi
    rm -f "$SIMULATE_OUT"
fi

# ?? Stage 1: install only the reboot-triggering packages, then reboot ???????
if (( REBOOT_TRIGGERS_NEEDED == 1 )) && [[ ! -f "$RESUME_MARKER" ]]; then
    step "4a. Stage 1: install reboot-triggering packages"
    log "Installing only the libc6/systemd/dbus/kernel upgrades to clear /run/reboot-required"

    # shellcheck disable=SC2086
    apt-get install -y --no-install-recommends --allow-downgrades $REBOOT_TRIGGER_HITS \
        || die "Stage 1 install failed; aborting before reboot."

    # Run needrestart -r a in case the upgrade didn't actually require a reboot
    # after dependency resolution (sometimes apt's simulation is conservative).
    if command -v needrestart >/dev/null 2>&1; then
        needrestart -r a -q 2>&1 | tail -30 || true
    fi

    # If reboot-required wasn't actually set after install (e.g. only libc6
    # patch that needrestart handled cleanly), we can continue to Stage 2.
    if [[ ! -f /run/reboot-required ]]; then
        log "Stage 1 complete; no actual reboot-required flag set. Continuing in same run."
        touch "$RESUME_MARKER"
    else
        # Write the marker and request a reboot.
        touch "$RESUME_MARKER"
        INSTALL_STATUS="stage1-reboot-required"
        cat <<EOM

==============================================================================
  Stage 1 complete. The system upgraded packages that require a reboot:
  $(cat /run/reboot-required.pkgs 2>/dev/null | tr '\n' ' ')

  ACTION REQUIRED:
    1. sudo reboot
    2. After reboot, re-run: sudo bash $0

  install-all.sh will detect the resume marker at $RESUME_MARKER and continue
  with the userland install (apps, venvs, llama.cpp, etc.).
==============================================================================

EOM
        exit 0
    fi
fi

# ============================================================================
# 5. APT INSTALL ??userland packages
# ============================================================================
step "5. APT install: userland packages"

# Install in a specific order to keep the dependency graph clean:
# 1) toolchain (build-essential, cmake, ninja, pkg-config, git)
# 2) python3.12-venv + python3.12-dev (required before any venv creation)
# 3) needrestart (so the daemon-restart step actually has the tool)
# 4) CLI utilities (htop, btop, nvtop, tmux, jq, ...)
# 5) GUI runtime libs (libgtk, libnss3, ...) ??for Chrome/VS Code
# 6) Desktop (xfce4 + xrdp + policykit) last, biggest dep tree

# Strict install: die if anything fails. Use ONLY for packages without which
# the install cannot meaningfully continue (compiler toolchain, python venv).
_apt_install_strict() {
    log "  apt-get install ${*:1:4}..."
    apt-get install -y --no-install-recommends --allow-downgrades "$@" \
        || apt-get install -y --allow-downgrades "$@" \
        || die "apt-get install failed for required packages: $*"
}

# Best-effort install: try the batch, fall back to per-package, warn on any
# packages that can't be satisfied from the bundle's local repo. Salvages the
# rest of the batch when one nice-to-have (e.g. gedit) has missing transitive
# deps that aren't in the airgap bundle.
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

log "Installing toolchain"
_apt_install_strict build-essential cmake ninja-build pkg-config git ccache curl wget ca-certificates unzip xz-utils

log "Installing python${PYTHON_VER}-venv + dev"
_apt_install_strict "python${PYTHON_VER}-venv" "python${PYTHON_VER}-dev" python3-pip

log "Installing needrestart"
_apt_install needrestart

log "Installing CLI utilities + monitoring"
_apt_install gedit vim nano htop btop nvtop iotop tmux screen \
    net-tools iproute2 dnsutils mtr-tiny traceroute \
    jq tree ncdu zip pigz zstd rsync \
    numactl hwloc-nox

log "Installing GUI runtime libs (Chrome/VS Code deps)"
_apt_install \
    libglib2.0-0t64 libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libcairo2 libcups2t64 libdbus-1-3 libdrm2 libexpat1 \
    libfontconfig1 fonts-liberation libgbm1 libgtk-3-0t64 \
    libnspr4 libnss3 libpango-1.0-0 libsecret-1-0 \
    libasound2t64 libx11-6 libx11-xcb1 libxcb1 \
    libxcomposite1 libxcursor1 libxdamage1 libxext6 \
    libxfixes3 libxi6 libxkbcommon0 libxkbfile1 \
    libxrandr2 libxrender1 libxss1 libxtst6 xdg-utils

log "Installing scientific native libs (h5py/openblas)"
_apt_install libopenblas-dev libopenblas0 libgomp1 libhdf5-dev libssl-dev libffi-dev libcurl4-openssl-dev

if [[ "$INSTALL_DESKTOP" == "1" ]]; then
    # Headless detection: an NVIDIA Blackwell GPU shows up as a 3D controller,
    # NOT a "VGA compatible controller" or "Display controller", so lspci's VGA
    # check is a reasonable proxy for "this box has a real display". The user
    # may still want xrdp on a headless box (remote desktop with no monitor),
    # so we WARN rather than refuse. They can set INSTALL_DESKTOP=0 to skip.
    if command -v lspci >/dev/null 2>&1; then
        if ! lspci 2>/dev/null | grep -qiE '(VGA compatible controller|Display controller)'; then
            warn "No VGA/display controller detected via lspci. lightdm will still be installed (INSTALL_DESKTOP=1)."
            warn "  - lightdm.service may stall graphical.target boot by ~90s on headless hosts."
            warn "  - sshd is on multi-user.target so SSH still works, but boot is slower."
            warn "  - Re-run with INSTALL_DESKTOP=0 if you only need SSH access."
        fi
    fi
    log "Installing XFCE4 + xrdp + policykit"
    # DEBIAN_FRONTEND=noninteractive is exported at the top of this script so
    # lightdm's debconf prompt for default DM doesn't hang the install.
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

# ============================================================================
# 6. APP .DEBS ??VS Code + Chrome (via `apt install ./`)
# ============================================================================
step "6. App .debs (VS Code, Chrome)"

if [[ -f "$BUNDLE_DIR/apps/vscode.deb" ]]; then
    log "Installing VS Code (apt install ./)"
    # Use `apt install ./file.deb` (not raw dpkg) so apt resolves t64 deps.
    apt-get install -y "$BUNDLE_DIR/apps/vscode.deb" || warn "VS Code install failed."
    command -v code >/dev/null && log "VS Code: $(code --version 2>/dev/null | head -1)" \
        || warn "VS Code installed but 'code' not on PATH."
else
    warn "apps/vscode.deb not found; skipping."
fi

if [[ -f "$BUNDLE_DIR/apps/chrome.deb" ]]; then
    log "Installing Google Chrome (apt install ./)"
    apt-get install -y "$BUNDLE_DIR/apps/chrome.deb" || warn "Chrome install failed."
    command -v google-chrome-stable >/dev/null \
        && log "Chrome: $(google-chrome-stable --version 2>/dev/null)" \
        || warn "Chrome installed but binary not in PATH."
else
    warn "apps/chrome.deb not found; skipping."
fi

# AppArmor reload ??VS Code and Chrome ship profiles to /etc/apparmor.d/.
if command -v aa-status >/dev/null 2>&1 && [[ -d /etc/apparmor.d ]]; then
    log "Reloading AppArmor profiles (registers Chrome/VS Code profiles)"
    systemctl reload apparmor 2>/dev/null || apparmor_parser -r /etc/apparmor.d/ 2>/dev/null || true
fi

# Ubuntu 24.04 hardening: allow unprivileged user namespaces so Chrome/VS Code
# Electron sandboxes work. Without this, both apps silently fail to launch.
if [[ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
    log "Disabling apparmor_restrict_unprivileged_userns (Chrome/VS Code sandbox)"
    tee /etc/sysctl.d/60-apparmor-userns.conf > /dev/null <<'SYSCTL'
# Allow unprivileged user namespaces ??Chrome/VS Code/Firefox sandbox.
# Set by install-all.sh on Ubuntu 24.04+.
kernel.apparmor_restrict_unprivileged_userns = 0
SYSCTL
    sysctl --system >/dev/null 2>&1 \
        || sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 \
        || warn "Could not apply apparmor userns sysctl."
fi

# ============================================================================
# 7. TARBALL APPS ??Firefox, Node.js, Bun, Opencode
# ============================================================================
step "7. Tarball apps (Firefox, Node.js, Bun, Opencode)"

# ?? Firefox ?????????????????????????????????????????????????????????????????
FF_TARBALL=""
for c in firefox.tar.xz firefox.tar.bz2; do
    [[ -f "$BUNDLE_DIR/apps/$c" ]] && { FF_TARBALL="$BUNDLE_DIR/apps/$c"; break; }
done
if [[ -n "$FF_TARBALL" ]]; then
    FF_VER=$(cat "$BUNDLE_DIR/apps/firefox.version" 2>/dev/null || echo unknown)
    log "Installing Firefox $FF_VER to /opt/firefox"
    mkdir -p /opt/firefox
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
        # Desktop entry
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

# ?? Node.js + npm ???????????????????????????????????????????????????????????
if [[ -f "$BUNDLE_DIR/apps/nodejs.tar.xz" ]]; then
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

# ?? Bun ?????????????????????????????????????????????????????????????????????
if [[ -f "$BUNDLE_DIR/apps/bun-linux-x64.zip" ]]; then
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

# ?? Opencode ????????????????????????????????????????????????????????????????
if [[ -f "$BUNDLE_DIR/apps/opencode" ]]; then
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

# ============================================================================
# 8. needrestart -r a ??restart any daemons holding old libs
# ============================================================================
step "8. needrestart -r a (auto-restart daemons holding old libs)"

if command -v needrestart >/dev/null 2>&1; then
    NEEDRESTART_MODE=a needrestart -r a 2>&1 | tail -50 || warn "needrestart returned non-zero."
else
    warn "needrestart not installed; skipping (libs may be stale until reboot)."
fi

# ============================================================================
# 9. XRDP + XFCE4 CONFIG
# ============================================================================
step "9. xrdp + XFCE4 configuration"

if [[ "$INSTALL_DESKTOP" == "1" ]] && command -v xrdp >/dev/null 2>&1; then
    log "Configuring xrdp to launch XFCE4"
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

    # Allow xrdp to read TLS cert for NLA/encryption
    adduser xrdp ssl-cert 2>/dev/null || true

    systemctl enable xrdp 2>/dev/null || true
    systemctl restart xrdp 2>/dev/null || warn "xrdp restart failed ??run 'systemctl start xrdp' after reboot."
    log "xrdp listening on port 3389"

    # Default XFCE4 session for current user and new users
    echo "xfce4-session" | tee /etc/skel/.xsession > /dev/null
    if [[ -n "${SUDO_USER:-}" ]]; then
        su - "$SUDO_USER" -c "echo xfce4-session > ~/.xsession" 2>/dev/null || true
    fi

    # Polkit rule: allow sudo group to power-off/reboot from XFCE
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

    # Open UFW if active
    if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
        ufw allow 3389/tcp 2>/dev/null && log "UFW: port 3389/tcp opened" || true
    fi

    log "Desktop setup complete. Connect via RDP to port 3389."
else
    log "INSTALL_DESKTOP=0 or xrdp not installed; skipping desktop config."
fi

# ============================================================================
# 10. PYTHON VENV: Inference
# ============================================================================
step "10. Wheelhouse manifests"
generate_wheelhouse_requirements

step "10. Python venv: LLM Inference"

# Run venv creation as the real user (TARGET_USER) so the venv ownership is
# correct. We're root for apt; switch back for venv operations under $SCRATCH_ROOT.
_as_user() {
    if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
        sudo -u "$SUDO_USER" "$@"
    else
        "$@"
    fi
}

if [[ "$INSTALL_INFERENCE" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/inference"
    if _wheelhouse_has_packages "$WHEELS_DIR"; then
        log "Creating inference venv at $INFERENCE_PREFIX/venv"
        mkdir -p "$INFERENCE_PREFIX"
        chown "$TARGET_USER:$TARGET_GROUP" "$INFERENCE_PREFIX"

        if _as_user "$PYTHON_BIN" -m venv "$INFERENCE_PREFIX/venv"; then
            _PIP="$INFERENCE_PREFIX/venv/bin/pip"

            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
                || warn "Bootstrap pip install failed."

            log "Installing PyTorch (cu130)"
            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio \
                || warn "torch install failed."

            log "Installing vLLM (pinned to cu130 backend)"
            # vLLM picks its torch backend from the install index; with --no-index
            # and the cu130 wheels in find-links, it resolves correctly.
            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" vllm \
                || warn "vLLM install failed."

            for rf in "$BUNDLE_DIR/requirements/llm_api.txt" "$BUNDLE_DIR/requirements/llm_api_full.txt"; do
                [[ -f "$rf" ]] || continue
                log "  Installing from $(basename "$rf")"
                _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
            done

            log "Installing core inference / RAG packages"
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

            log "Smoke test: torch + vllm"
            _as_user "$INFERENCE_PREFIX/venv/bin/python" - <<'PY' || warn "Inference smoke test failed."
import torch
print(f"  torch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device count:   {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
try:
    import vllm
    print(f"  vllm  {vllm.__version__}")
except Exception as e:
    print(f"  vllm import failed: {e}")
PY

            # Pre-warm sm_103 PTX-JIT cache (B300 not in PyTorch 2.11 cubin list).
            if _as_user "$INFERENCE_PREFIX/venv/bin/python" -c \
                "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda').sum().item()" \
                2>/dev/null; then
                log "PTX-JIT cache pre-warmed for sm_103"
            else
                warn "PTX-JIT pre-warm skipped (CUDA not initialized yet?)"
            fi

            log "Inference venv ready: $INFERENCE_PREFIX/venv"
        else
            warn "Could not create inference venv."
        fi
    else
        warn "wheels/inference/ empty; skipping."
    fi
else
    log "INSTALL_INFERENCE=0; skipping."
fi

# ============================================================================
# 11. PYTHON VENV: Training
# ============================================================================
step "11. Python venv: General Training"

if [[ "$INSTALL_TRAINING" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/training"
    if _wheelhouse_has_packages "$WHEELS_DIR"; then
        log "Creating training venv at $TRAINING_PREFIX/venv"
        mkdir -p "$TRAINING_PREFIX"
        chown "$TARGET_USER:$TARGET_GROUP" "$TRAINING_PREFIX"

        if _as_user "$PYTHON_BIN" -m venv "$TRAINING_PREFIX/venv"; then
            _PIP="$TRAINING_PREFIX/venv/bin/pip"

            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
                || warn "Bootstrap pip install failed."

            log "Installing PyTorch (cu130) + PyG"
            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio \
                || warn "torch install failed."
            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch-geometric \
                || warn "torch-geometric install failed."

            for pkg in pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv; do
                if _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" "$pkg" 2>/dev/null; then
                    log "  $pkg: OK"
                else
                    warn "  $pkg: not available in wheelhouse (expected for torch_spline_conv on cu130)"
                fi
            done

            for rf in "$BUNDLE_DIR/requirements/meshgraphnets.txt" \
                      "$BUNDLE_DIR/requirements/simulgen.txt" \
                      "$BUNDLE_DIR/requirements/pemtron.txt" \
                      "$BUNDLE_DIR/requirements/pemtron_transfer.txt"; do
                [[ -f "$rf" ]] || continue
                log "  Installing from $(basename "$rf")"
                _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
            done

            log "Installing core training/scientific stack"
            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
                numpy scipy h5py pandas tqdm matplotlib seaborn Pillow \
                scikit-learn scikit-image statsmodels networkx sympy \
                torchinfo tensorboard \
                opencv-python imageio librosa audiomentations soxr natsort \
                reportlab paramiko smbprotocol 2>/dev/null || true

            log "Smoke test: torch + PyG"
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
        else
            warn "Could not create training venv."
        fi
    else
        warn "wheels/training/ empty; skipping."
    fi
else
    log "INSTALL_TRAINING=0; skipping."
fi

# ============================================================================
# 12. PYTHON VENV: Jupyter
# ============================================================================
step "12. Python venv: JupyterLab + data science"

if [[ "$INSTALL_JUPYTER" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/jupyter"
    if _wheelhouse_has_packages "$WHEELS_DIR"; then
        log "Creating jupyter venv at $JUPYTER_PREFIX/venv"
        mkdir -p "$JUPYTER_PREFIX"
        chown "$TARGET_USER:$TARGET_GROUP" "$JUPYTER_PREFIX"

        if _as_user "$PYTHON_BIN" -m venv "$JUPYTER_PREFIX/venv"; then
            _PIP="$JUPYTER_PREFIX/venv/bin/pip"

            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
                || warn "Bootstrap pip install failed."

            _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
                jupyterlab notebook ipykernel ipywidgets jupyter-server \
                pandas polars numpy scipy matplotlib seaborn plotly \
                scikit-learn statsmodels tqdm rich requests aiohttp \
                black ruff mypy pytest ipdb \
                || warn "Some Jupyter packages failed."

            # Register kernel
            _as_user "$JUPYTER_PREFIX/venv/bin/python" -m ipykernel install \
                --user --name "airgap-py${PYTHON_VER}" \
                --display-name "Python ${PYTHON_VER} (airgap)" 2>/dev/null || true

            # Convenience launcher in the user's home
            if [[ -n "${SUDO_USER:-}" ]]; then
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
        else
            warn "Could not create jupyter venv."
        fi
    else
        warn "wheels/jupyter/ empty; skipping."
    fi
else
    log "INSTALL_JUPYTER=0; skipping."
fi

# ============================================================================
# 13. LLAMA.CPP ??build against vendor's CUDA 13.0
# ============================================================================
step "13. llama.cpp build (sm_${CUDA_ARCH_LIST//;/+sm_})"

if [[ "$INSTALL_LLAMA" == "1" && -f "$BUNDLE_DIR/src/llama.cpp.tar.gz" ]]; then
    log "Extracting llama.cpp -> $LLAMA_PREFIX"
    rm -rf "$LLAMA_PREFIX"
    mkdir -p "$LLAMA_PREFIX"
    chown "$TARGET_USER:$TARGET_GROUP" "$LLAMA_PREFIX"
    if _as_user tar -xzf "$BUNDLE_DIR/src/llama.cpp.tar.gz" -C "$LLAMA_PREFIX" --strip-components=1; then
        # install-nvidia.sh puts nvcc at /usr/local/cuda/bin/nvcc but does NOT
        # add it to a system-wide PATH for non-login subshells (we ship
        # /etc/profile.d/cuda.sh further down for interactive login shells).
        # cmake's CUDA detection uses PATH or CMAKE_CUDA_COMPILER, so without
        # the explicit -D below the build aborts with
        #   "No CMAKE_CUDA_COMPILER could be found".
        NVCC_PATH=""
        for c in /usr/local/cuda/bin/nvcc "/usr/local/cuda-${CUDA_MAJOR:-13}.${CUDA_MINOR:-0}/bin/nvcc"; do
            [[ -x "$c" ]] && { NVCC_PATH="$c"; break; }
        done
        [[ -n "$NVCC_PATH" ]] || warn "nvcc not found under /usr/local/cuda*; CUDA build will fail."

        CMAKE_ARGS=(
            -S "$LLAMA_PREFIX"
            -B "$LLAMA_PREFIX/build"
            -DCMAKE_BUILD_TYPE=Release
            -DGGML_NATIVE=ON
            -DGGML_CUDA=ON
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_LIST"
            -DLLAMA_CURL=ON
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_EXAMPLES=ON
            -DLLAMA_BUILD_SERVER=ON
            # Disable UI target ??it tries to fetch JS/CSS from huggingface.co
            # at build time, which times out on airgapped boxes.
            -DLLAMA_BUILD_UI=OFF
        )
        [[ "$BUILD_BLAS" == "1" ]] && CMAKE_ARGS+=( -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS )
        [[ -n "$NVCC_PATH" ]] && CMAKE_ARGS+=(
            -DCMAKE_CUDA_COMPILER="$NVCC_PATH"
            -DCUDAToolkit_ROOT=/usr/local/cuda
        )

        log "Configuring + building llama.cpp (cmake, -j$JOBS, ${CUDA_ARCH_LIST}, nvcc=$NVCC_PATH)"
        if _as_user cmake "${CMAKE_ARGS[@]}" \
            && _as_user cmake --build "$LLAMA_PREFIX/build" --config Release -j"$JOBS"; then
            log "llama.cpp built: $LLAMA_PREFIX/build/bin/llama-server"

            # Python venv for convert_hf_to_gguf.py etc.
            LLAMA_WHEELS="$BUNDLE_DIR/wheels/llamacpp"
            if _wheelhouse_has_packages "$LLAMA_WHEELS"; then
                log "Creating llama.cpp Python venv"
                if _as_user "$PYTHON_BIN" -m venv "$LLAMA_PREFIX/venv"; then
                    _PIP="$LLAMA_PREFIX/venv/bin/pip"
                    _as_user "$_PIP" install --no-index --find-links="$LLAMA_WHEELS" --upgrade pip wheel setuptools 2>/dev/null || true
                    shopt -s nullglob
                    for rf in "$LLAMA_PREFIX"/requirements.txt "$LLAMA_PREFIX"/requirements/*.txt; do
                        [[ -f "$rf" ]] || continue
                        _as_user "$_PIP" install --no-index --find-links="$LLAMA_WHEELS" -r "$rf" 2>/dev/null || true
                    done
                    shopt -u nullglob
                fi
            fi

            # Smoke test
            "$LLAMA_PREFIX/build/bin/llama-cli" --version 2>&1 | head -3 \
                && log "llama.cpp OK" \
                || warn "llama-cli --version failed."
        else
            warn "llama.cpp build failed."
        fi
    else
        warn "llama.cpp source extraction failed."
    fi
else
    log "INSTALL_LLAMA=0 or src/llama.cpp.tar.gz missing; skipping."
fi

# ============================================================================
# 14. SYSTEM TUNING ??sysctl + THP + limits
# ============================================================================
step "14. System tuning (sysctl + THP + ulimits)"

tee /etc/sysctl.d/99-llm-multigpu.conf > /dev/null <<'SYSCTL'
# Installed by install-all.sh ??multi-GPU LLM workload tuning.
vm.overcommit_memory=1
vm.swappiness=0
vm.max_map_count=1048576
net.core.rmem_max=268435456
net.core.wmem_max=268435456
SYSCTL
sysctl --system >/dev/null 2>&1 || warn "sysctl --system reload failed."
log "sysctl applied"

# Transparent hugepages = madvise (vLLM/torch opts in explicitly).
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

# System limits ??multi-process inference hits EMFILE/memlock defaults fast.
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

# =============================================================================
# DO NOT WRITE /etc/systemd/system.conf.d/*.conf OR /etc/systemd/user.conf.d/*.conf
# =============================================================================
# These files set DefaultLimit*= for PID 1 and every systemd-managed service.
# A typo here is unrecoverable without console access — there is no SSH route
# back if you crash sshd at boot.
#
# Specific traps:
#   1. UNIT MISMATCH. pam_limits (`/etc/security/limits.d/`) and systemd use
#      DIFFERENT units for the same resource:
#        limits.conf  stack    65536   → 65536 KB  = 64 MB   (fine)
#        systemd      LimitSTACK=65536 → 65536 B   = 64 KB   (catastrophic)
#      LimitMEMLOCK/LimitDATA/LimitAS/LimitFSIZE all share the bytes-vs-KB trap.
#      LimitNOFILE/LimitNPROC are counts in both — those happen to agree.
#   2. BLAST RADIUS. Per-service Limit*= in a unit file affects ONE service.
#      DefaultLimit*= in /etc/systemd/system.conf.d/ affects EVERY service,
#      including ones you didn't author (sshd, dbus, NetworkManager, polkit,
#      fabricmanager). Even a "harmless" change cascades.
#   3. systemctl daemon-reexec APPLIES IT NOW. PID 1 re-execs with the new
#      defaults immediately, so a bad value is live the moment the file is
#      written. There is no "test before commit" path.
#
# Right pattern: put Limit*= in the unit file of the service that needs it.
# See /etc/systemd/system/llama-server@.service below for the canonical example.
#
# Drop any stale config from a prior buggy install-all.sh run that set
# DefaultLimitSTACK=65536 (= 64 KB), which crashes every systemd service.
if [[ -f /etc/systemd/system.conf.d/99-llm-multigpu.conf ]]; then
    rm -f /etc/systemd/system.conf.d/99-llm-multigpu.conf
    systemctl daemon-reexec 2>/dev/null || true
    log "Removed stale /etc/systemd/system.conf.d/99-llm-multigpu.conf (had broken DefaultLimitSTACK)"
fi
log "System limits applied (pam_limits only; per-service limits live in unit files)"

# Put CUDA on the PATH for every login shell. install-nvidia.sh installs nvcc
# under /usr/local/cuda/bin/ but does not advertise it; without this file every
# non-root login (and every CI/cron job) sees "nvcc: command not found".
tee /etc/profile.d/cuda.sh > /dev/null <<'CUDA_PATH'
# Installed by install-all.sh -- CUDA toolkit on PATH for login shells.
if [ -d /usr/local/cuda/bin ]; then
    case ":$PATH:" in
        *:/usr/local/cuda/bin:*) : ;;
        *) export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} ;;
    esac
fi
if [ -d /usr/local/cuda/lib64 ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *:/usr/local/cuda/lib64:*) : ;;
        *) export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ;;
    esac
fi
CUDA_PATH
chmod 0644 /etc/profile.d/cuda.sh
log "Wrote /etc/profile.d/cuda.sh (nvcc on PATH for login shells)"

# ============================================================================
# 15. OPERATIONAL TOOLING ??helper scripts + systemd template
# ============================================================================
step "15. Operational tooling (gpu-health-check, llama-server-multigpu, systemd)"

# ?? /usr/local/bin/gpu-health-check ?????????????????????????????????????????
tee /usr/local/bin/gpu-health-check > /dev/null <<'HEALTH'
#!/usr/bin/env bash
# gpu-health-check ??verify multi-GPU fabric is healthy. Installed by install-all.sh.
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
# Anchor on the Fabric section header (or the single-line "Fabric State :")
# and capture exactly one State value per GPU. The old grep-based check
# matched anything containing "Fabric"+"State" and passed on a single hit,
# so 7 Pending + 1 Completed could be reported as OK.
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
    warn "no Fabric stanza in nvidia-smi -q (single-GPU box or FM not initialized)"
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

# ?? /usr/local/bin/llama-server-multigpu ????????????????????????????????????
tee /usr/local/bin/llama-server-multigpu > /dev/null <<LLAMAWRAP
#!/usr/bin/env bash
# llama-server-multigpu ??NUMA + tensor-split wrapper for llama.cpp server.
# Installed by install-all.sh.
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

# ?? /usr/local/bin/llama-model-preload ??????????????????????????????????????
tee /usr/local/bin/llama-model-preload > /dev/null <<'PRELOAD'
#!/usr/bin/env bash
# llama-model-preload ??pre-mmap GGUF model into page cache.
# Installed by install-all.sh.
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

# ?? llama-server@.service template ??????????????????????????????????????????
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

# ============================================================================
# 16. CLEAR RESUME MARKER + FINAL STATUS
# ============================================================================
step "16. Final status"

# Clear the resume marker ??install is complete.
rm -f "$RESUME_MARKER"

# Final ownership sweep for /scratch
chown -R "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT" 2>/dev/null || true

INSTALL_STATUS="complete"
REBOOT_RECOMMENDED=0
if [[ -f /run/reboot-required ]]; then
    REBOOT_RECOMMENDED=1
    log "/run/reboot-required is set (post-install)."
    [[ -f /run/reboot-required.pkgs ]] && log "  triggered by: $(tr '\n' ' ' </run/reboot-required.pkgs)"
fi

printf '\n'
printf '%s\n' "════════════════════════════════════════════════════════════════"
printf '%s\n' "  INSTALL COMPLETE"
printf '%s\n' "════════════════════════════════════════════════════════════════"
printf '  Started : %s\n' "$INSTALL_STARTED_AT"
printf '  Finished: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '  Log     : %s\n' "$INSTALL_LOG"
printf '\n'
printf '  Inference venv : %s/venv\n' "$INFERENCE_PREFIX"
printf '  Training venv  : %s/venv\n' "$TRAINING_PREFIX"
printf '  Jupyter venv   : %s/venv\n' "$JUPYTER_PREFIX"
printf '  llama-server   : %s/build/bin/llama-server\n' "$LLAMA_PREFIX"
printf '\n'
printf '  Warnings : %d\n' "${#INSTALL_WARNINGS[@]}"
printf '  Errors   : %d\n' "${#INSTALL_ERRORS[@]}"
printf '\n'

if (( ${#INSTALL_WARNINGS[@]} > 0 )); then
    printf 'Warnings raised during install:\n'
    for w in "${INSTALL_WARNINGS[@]}"; do printf '  - %s\n' "$w"; done
    printf '\n'
fi

printf 'Next steps:\n'
printf '  bash test-all.sh                              # verify everything\n'
printf '  gpu-health-check                              # quick fabric sanity\n'
printf '  source %s/venv/bin/activate                    # use inference venv\n' "$INFERENCE_PREFIX"
printf '  %s/build/bin/llama-server --help               # serve a GGUF model\n' "$LLAMA_PREFIX"
if [[ "$INSTALL_DESKTOP" == "1" ]]; then
    printf '  rdp connect ??tcp 3389                         # remote desktop (xfce4)\n'
fi
printf '\n'

if (( REBOOT_RECOMMENDED )); then
    printf '\033[1;33mAdvisory: /run/reboot-required is set. Reboot recommended:\033[0m\n'
    printf '  sudo reboot\n\n'
else
    printf '\033[1;32mNo reboot required. Review diagnostics below before declaring production-ready.\033[0m\n\n'
fi

print_final_diagnostics 0
