#!/usr/bin/env bash
# ============================================================================
# install-nvidia.sh
#
#   Run on the AIR-GAPPED Ubuntu 24.04 server BEFORE install-all.sh.
#   Place this script next to nvidia-airgap-bundle-ubuntu24.04.bin and run:
#
#     sudo bash pre-install-nvidia.sh   # readiness gate (no changes)
#     sudo bash install-nvidia.sh       # this script — installs everything
#     sudo reboot                       # MANDATORY — loads nvidia.ko + FM/NVLSM
#     sudo bash test-nvidia.sh          # verify NVSwitch fabric Completed
#     # then proceed with: pre-install-check.sh → install-all.sh
#
#   What it installs (R580 LTS):
#     - nvidia-driver-580-open + cuda-drivers-580 + fabricmanager/NSCQ
#     - nvidia-modprobe (nvidia-persistenced + nvidia-peermem.ko ride in
#       transitively with the driver pkg; no separate apt names exist)
#     - /etc/modules-load.d/nvidia-peermem.conf for boot-time peermem load
#     - cuda-toolkit-13-0 (+ cudart, cudart-dev, compat)
#     - libnccl2 / libnccl-dev pinned to +cuda13.0
#     - datacenter-gpu-manager-4-cuda13
#     - nvidia-driver-pinning-580 (apt unattended-upgrade guard)
#
#   What it does NOT touch:
#     - DOCA-OFED (vendor pre-installed; pre-install-nvidia.sh verified)
#     - Userland packages (gather-all.sh / install-all.sh handle that)
#     - GPU mode (MIG / ECC / clocks / power) — operator policy
#
#   Optional overrides:
#     BUNDLE_DIR=/path/to/extracted bash install-nvidia.sh
#     SKIP_REBOOT_PROMPT=1 bash install-nvidia.sh
#     SKIP_NCCL=1 bash install-nvidia.sh
#     SKIP_DCGM=1 bash install-nvidia.sh
# ============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
INSTALL_LOG="${INSTALL_LOG:-$SCRIPT_DIR/install-nvidia-$RUN_STAMP.log}"

BUNDLE_DIR="${BUNDLE_DIR:-}"
BUNDLE_BIN="${BUNDLE_BIN:-}"
DRIVER_BRANCH="${DRIVER_BRANCH:-580}"
CUDA_MAJOR="${CUDA_MAJOR:-13}"
CUDA_MINOR="${CUDA_MINOR:-0}"
APT_REPO_DIR="${APT_REPO_DIR:-/var/tmp/airgap-nvidia-debs}"
APT_LIST_FILE="/etc/apt/sources.list.d/00-nvidia-bundle.list"
APT_PIN_FILE="/etc/apt/preferences.d/99-nvidia-prefer-bundle"
HELD_PKGS_FILE="${HELD_PKGS_FILE:-/var/lib/install-nvidia/nvidia-held.txt}"
SKIP_REBOOT_PROMPT="${SKIP_REBOOT_PROMPT:-0}"
SKIP_NCCL="${SKIP_NCCL:-0}"
SKIP_DCGM="${SKIP_DCGM:-0}"

# ============================================================================
# Helpers
# ============================================================================
log()  { printf '\033[1;36m[install-nv]\033[0m %s\n' "$*" | tee -a "$INSTALL_LOG"; }
warn() { printf '\033[1;33m[install-nv:WARN]\033[0m %s\n' "$*" | tee -a "$INSTALL_LOG"; }
die()  { printf '\033[1;31m[install-nv:ERROR]\033[0m %s\n' "$*" | tee -a "$INSTALL_LOG" >&2; exit 1; }
step() { printf '\n\033[1;35m══ %s ══\033[0m\n' "$*" | tee -a "$INSTALL_LOG"; }

mkdir -p "$(dirname "$INSTALL_LOG")"
: > "$INSTALL_LOG"
log "install-nvidia.sh started at $RUN_STAMP"
log "log: $INSTALL_LOG"

if [[ $EUID -ne 0 ]]; then
    log "Re-executing with sudo"
    exec sudo -E env \
        BUNDLE_DIR="$BUNDLE_DIR" BUNDLE_BIN="$BUNDLE_BIN" \
        SKIP_REBOOT_PROMPT="$SKIP_REBOOT_PROMPT" \
        SKIP_NCCL="$SKIP_NCCL" SKIP_DCGM="$SKIP_DCGM" \
        bash "$0" "$@"
fi

. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] \
    || die "Target must be Ubuntu 24.04 (found ${PRETTY_NAME:-?})."

# ============================================================================
# 1) Bundle: locate + extract
# ============================================================================
step "1. Bundle: locate + extract"

if [[ -z "$BUNDLE_DIR" ]]; then
    if [[ -z "$BUNDLE_BIN" ]]; then
        for cand in "$SCRIPT_DIR/nvidia-airgap-bundle-ubuntu24.04.bin" \
                    "$PWD/nvidia-airgap-bundle-ubuntu24.04.bin"; do
            [[ -f "$cand" ]] && { BUNDLE_BIN="$cand"; break; }
        done
    fi
    [[ -f "$BUNDLE_BIN" ]] || die "Bundle not found. Set BUNDLE_BIN=path/to/bundle.bin"

    if [[ -f "${BUNDLE_BIN}.sha256" ]]; then
        log "Verifying bundle SHA256"
        ( cd "$(dirname "$BUNDLE_BIN")" \
            && sha256sum -c --status "$(basename "$BUNDLE_BIN").sha256" ) \
            || die "Bundle SHA256 mismatch — re-transfer."
        log "SHA256 ok"
    else
        warn "No .sha256 sidecar — skipping integrity check."
    fi

    EXTRACT_PARENT="${EXTRACT_PARENT:-/var/tmp}"
    BUNDLE_DIR="$EXTRACT_PARENT/GPU_server_downloads_nvidia"
    log "Extracting bundle to $BUNDLE_DIR"
    rm -rf "$BUNDLE_DIR"
    mkdir -p "$EXTRACT_PARENT"
    tar -xzf "$BUNDLE_BIN" -C "$EXTRACT_PARENT"
    [[ -d "$BUNDLE_DIR/debs" ]] \
        || die "Extract did not produce expected layout (missing debs/)."
fi

DEBS_DIR="$BUNDLE_DIR/debs"
[[ -d "$DEBS_DIR" ]] || die "No debs/ in $BUNDLE_DIR"
log "Bundle: $BUNDLE_DIR  ($(ls "$DEBS_DIR" | wc -l) files, $(du -sh "$DEBS_DIR" | cut -f1))"

# Validate bundle metadata
if [[ -f "$BUNDLE_DIR/meta/target.env" ]]; then
    set -a; . "$BUNDLE_DIR/meta/target.env"; set +a
    [[ "${BUNDLE_VARIANT:-}" == "nvidia-stack" ]] \
        || die "Bundle variant '$BUNDLE_VARIANT' — expected 'nvidia-stack'. Wrong bundle?"
    log "Bundle: ${BUNDLE_VARIANT}  driver=${BUNDLE_DRIVER_VERSION:-?}  cuda=${BUNDLE_CUDA:-?}  nccl=${BUNDLE_NCCL_VERSION:-?}"
fi

_pkg_with_version() {
    local pkg="$1" ver="${2:-}"
    if [[ -n "$ver" && "$ver" != "skipped" ]]; then
        printf '%s=%s\n' "$pkg" "$ver"
    else
        printf '%s\n' "$pkg"
    fi
}

DRIVER_PACKAGE="${BUNDLE_DRIVER_PACKAGE:-nvidia-driver-${DRIVER_BRANCH}-open}"
DRIVER_PACKAGE_VERSION="${BUNDLE_DRIVER_PACKAGE_VERSION:-}"
CUDA_DRIVERS_PACKAGE="${BUNDLE_CUDA_DRIVERS_PACKAGE:-cuda-drivers-${DRIVER_BRANCH}}"
CUDA_DRIVERS_VERSION="${BUNDLE_CUDA_DRIVERS_VERSION:-}"
CUDA_DRIVERS_FM_PACKAGE="${BUNDLE_CUDA_DRIVERS_FABRICMANAGER_PACKAGE:-cuda-drivers-fabricmanager-${DRIVER_BRANCH}}"
CUDA_DRIVERS_FM_VERSION="${BUNDLE_CUDA_DRIVERS_FABRICMANAGER_VERSION:-}"
[[ "$CUDA_DRIVERS_FM_PACKAGE" == "skipped" ]] && CUDA_DRIVERS_FM_PACKAGE=""
[[ "$CUDA_DRIVERS_FM_VERSION" == "skipped" ]] && CUDA_DRIVERS_FM_VERSION=""
FM_PACKAGE="${BUNDLE_FABRICMANAGER_PACKAGE:-nvidia-fabricmanager}"
FM_VERSION="${BUNDLE_FABRICMANAGER_VERSION:-}"
NSCQ_PACKAGE="${BUNDLE_NSCQ_PACKAGE:-libnvidia-nscq}"
NSCQ_VERSION="${BUNDLE_NSCQ_VERSION:-}"
CUDA_COMPAT_PACKAGE="${BUNDLE_CUDA_COMPAT_PACKAGE:-cuda-compat-${CUDA_MAJOR}-${CUDA_MINOR}}"
CUDA_COMPAT_VERSION="${BUNDLE_CUDA_COMPAT_VERSION:-}"

# ============================================================================
# 2) Local apt repo + pin
# ============================================================================
step "2. Local apt repo + pin"

rm -rf "$APT_REPO_DIR"
mkdir -p "$APT_REPO_DIR"
cp "$DEBS_DIR"/*.deb "$APT_REPO_DIR/" 2>/dev/null || die "No .debs to copy."
if [[ -f "$DEBS_DIR/Packages" ]]; then
    cp "$DEBS_DIR/Packages" "$APT_REPO_DIR/"
    [[ -f "$DEBS_DIR/Packages.gz" ]] && cp "$DEBS_DIR/Packages.gz" "$APT_REPO_DIR/"
else
    log "Regenerating Packages index"
    command -v dpkg-scanpackages >/dev/null \
        || die "dpkg-scanpackages missing on target. Ensure dpkg-dev is in userland bundle."
    ( cd "$APT_REPO_DIR" && dpkg-scanpackages . /dev/null > Packages \
        && gzip -9c Packages > Packages.gz )
fi

_repo_has_version() {
    local pkg="$1" ver="$2"
    [[ -n "$pkg" && -n "$ver" && "$ver" != "skipped" ]] || return 1
    awk -v want_pkg="$pkg" -v want_ver="$ver" '
        /^Package: / { pkg=$2; next }
        /^Version: / && pkg == want_pkg && $2 == want_ver { found=1 }
        END { exit !found }
    ' "$APT_REPO_DIR/Packages"
}

_repo_versions_csv() {
    local pkg="$1"
    awk -v want_pkg="$pkg" '
        /^Package: / { pkg=$2; next }
        /^Version: / && pkg == want_pkg { if (!seen[$2]++) print $2 }
    ' "$APT_REPO_DIR/Packages" | paste -sd ', ' -
}

_resolve_repo_package_for_version() {
    local label="$1" current_pkg="$2" ver="$3"; shift 3
    local cand versions

    if [[ -z "$ver" || "$ver" == "skipped" ]]; then
        printf '%s\n' "$current_pkg"
        return 0
    fi

    if [[ -n "$current_pkg" && "$current_pkg" != "skipped" ]] \
        && _repo_has_version "$current_pkg" "$ver"; then
        printf '%s\n' "$current_pkg"
        return 0
    fi

    for cand in "$@"; do
        [[ -n "$cand" && "$cand" != "skipped" ]] || continue
        if _repo_has_version "$cand" "$ver"; then
            printf '\033[1;36m[install-nv]\033[0m Resolved %s package %s -> %s for version %s\n' \
                "$label" "$current_pkg" "$cand" "$ver" | tee -a "$INSTALL_LOG" >&2
            printf '%s\n' "$cand"
            return 0
        fi
    done

    versions="$(_repo_versions_csv "$current_pkg")"
    [[ -n "$versions" ]] || versions="none"
    die "Bundle metadata requests $label '$current_pkg=$ver', but debs/Packages does not contain it (available for $current_pkg: $versions). Rebuild the NVIDIA bundle with the fixed gather-nvidia.sh."
}

DRIVER_PACKAGE="$(_resolve_repo_package_for_version "driver" "$DRIVER_PACKAGE" "$DRIVER_PACKAGE_VERSION" \
    "nvidia-driver-${DRIVER_BRANCH}-open" "nvidia-driver-${DRIVER_BRANCH}" "nvidia-driver")"
CUDA_DRIVERS_PACKAGE="$(_resolve_repo_package_for_version "CUDA driver meta" "$CUDA_DRIVERS_PACKAGE" "$CUDA_DRIVERS_VERSION" \
    "cuda-drivers-${DRIVER_BRANCH}" "cuda-drivers")"
FM_PACKAGE="$(_resolve_repo_package_for_version "fabricmanager" "$FM_PACKAGE" "$FM_VERSION" \
    "nvidia-fabricmanager-${DRIVER_BRANCH}" "nvidia-fabricmanager")"
NSCQ_PACKAGE="$(_resolve_repo_package_for_version "NSCQ" "$NSCQ_PACKAGE" "$NSCQ_VERSION" \
    "libnvidia-nscq-${DRIVER_BRANCH}" "libnvidia-nscq")"
CUDA_COMPAT_PACKAGE="$(_resolve_repo_package_for_version "CUDA compat" "$CUDA_COMPAT_PACKAGE" "$CUDA_COMPAT_VERSION" \
    "cuda-compat-${CUDA_MAJOR}-${CUDA_MINOR}")"

if [[ -n "$CUDA_DRIVERS_FM_VERSION" ]]; then
    if _repo_has_version "$CUDA_DRIVERS_FM_PACKAGE" "$CUDA_DRIVERS_FM_VERSION"; then
        :
    elif _repo_has_version "cuda-drivers-fabricmanager" "$CUDA_DRIVERS_FM_VERSION"; then
        log "Resolved fabricmanager meta package '$CUDA_DRIVERS_FM_PACKAGE' -> 'cuda-drivers-fabricmanager'"
        CUDA_DRIVERS_FM_PACKAGE="cuda-drivers-fabricmanager"
    else
        log "No actual fabricmanager meta package for $CUDA_DRIVERS_FM_VERSION in debs/Packages; using $FM_PACKAGE=$FM_VERSION."
        CUDA_DRIVERS_FM_PACKAGE=""
        CUDA_DRIVERS_FM_VERSION=""
    fi
fi

cat > "$APT_LIST_FILE" <<EOF
deb [trusted=yes] file://$APT_REPO_DIR ./
EOF
log "Wrote $APT_LIST_FILE"

# Pin nvidia/cuda/libnvidia/libnccl packages to our file:// repo so a later
# install-all.sh cannot accidentally bump them via a different source.
cat > "$APT_PIN_FILE" <<EOF
Package: nvidia-* cuda-* libnvidia-* libnvjit* libnvfat* libnccl* nvlsm datacenter-gpu-manager-*
Pin: origin ""
Pin-Priority: 1001
EOF
log "Wrote $APT_PIN_FILE"

log "Refreshing apt indexes (file:// only)"
apt-get update -o Dir::Etc::sourcelist="$APT_LIST_FILE" \
    -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0" \
    >>"$INSTALL_LOG" 2>&1 || true
apt-get update >>"$INSTALL_LOG" 2>&1 || warn "Full apt update had errors (ok if airgapped)."

# ============================================================================
# 3) Pre-driver: nouveau blacklist
# ============================================================================
step "3. Pre-driver: nouveau blacklist"

NOUVEAU_BL=/etc/modprobe.d/blacklist-nouveau-nvidia.conf
if ! [[ -f "$NOUVEAU_BL" ]]; then
    cat > "$NOUVEAU_BL" <<'EOF'
# Installed by install-nvidia.sh — required for NVIDIA driver to load.
blacklist nouveau
options nouveau modeset=0
EOF
    log "Wrote $NOUVEAU_BL"
    if command -v update-initramfs >/dev/null; then
        update-initramfs -u >>"$INSTALL_LOG" 2>&1 \
            || warn "update-initramfs failed; nouveau may still load on next boot."
    fi
else
    log "$NOUVEAU_BL already present"
fi

# ============================================================================
# 4) Install NVIDIA core stack (driver + FM + NVLSM + NSCQ + NVSDM)
#    ONE transaction — version mismatch between these = FM abort.
# ============================================================================
if lsmod 2>/dev/null | awk '$1 == "nouveau" {found=1} END {exit !found}'; then
    warn "nouveau is currently loaded. The blacklist is staged, but it only takes effect after reboot."
    printf '\n\033[1;33m================================================================\033[0m\n'
    printf '\033[1;33m  REBOOT REQUIRED BEFORE DRIVER INSTALL\033[0m\n'
    printf '\033[1;33m================================================================\033[0m\n'
    printf '  Reboot now, then rerun:\n'
    printf '    sudo bash install-nvidia.sh\n\n'
    printf '  Verify nouveau is gone with:\n'
    printf '    lsmod | grep "^nouveau" || echo "nouveau not loaded"\n\n'
    exit 10
fi

step "4. NVIDIA core stack (single apt transaction)"

CORE_PKGS=(
    "nvidia-driver-pinning-${DRIVER_BRANCH}"
    "$(_pkg_with_version "$CUDA_DRIVERS_PACKAGE" "$CUDA_DRIVERS_VERSION")"
    "$(_pkg_with_version "$DRIVER_PACKAGE" "$DRIVER_PACKAGE_VERSION")"
    "$(_pkg_with_version "$FM_PACKAGE" "$FM_VERSION")"
    "nvlsm"
    "$(_pkg_with_version "$NSCQ_PACKAGE" "$NSCQ_VERSION")"
    "nvidia-modprobe"
)
if [[ -n "$CUDA_DRIVERS_FM_PACKAGE" ]]; then
    CORE_PKGS+=( "$(_pkg_with_version "$CUDA_DRIVERS_FM_PACKAGE" "$CUDA_DRIVERS_FM_VERSION")" )
fi
# Persistence (nvidia-persistenced) and peermem (nvidia-peermem.ko) come in
# transitively with the selected open driver package — no separate apt
# packages exist for them. peermem auto-load is handled in step 4b below.
# libnvsdm has no -580 variant in NVIDIA's repo (gap between R575 versioned
# and R595 unversioned). For NVSwitch-only setups it isn't needed.

log "Installing: ${CORE_PKGS[*]}"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --allow-downgrades \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    "${CORE_PKGS[@]}" 2>&1 | tee -a "$INSTALL_LOG" \
    || die "Core NVIDIA install failed. Check $INSTALL_LOG."

# ============================================================================
# 4b) Auto-load nvidia-peermem on boot (ships with nvidia-driver-*-open)
#     Needed for GPUDirect RDMA via DOCA-OFED. The .ko is in the driver
#     package; just declare it for systemd-modules-load.service.
# ============================================================================
PEERMEM_LOAD=/etc/modules-load.d/nvidia-peermem.conf
if ! [[ -f "$PEERMEM_LOAD" ]]; then
    printf 'nvidia-peermem\n' > "$PEERMEM_LOAD"
    log "Wrote $PEERMEM_LOAD"
else
    log "$PEERMEM_LOAD already present"
fi

# ============================================================================
# 5) CUDA toolkit (separate — does NOT pull driver again)
# ============================================================================
step "5. CUDA toolkit ${CUDA_MAJOR}.${CUDA_MINOR}"

CUDA_PKGS=(
    "cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"
    "cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}"
    "cuda-cudart-dev-${CUDA_MAJOR}-${CUDA_MINOR}"
    "$(_pkg_with_version "$CUDA_COMPAT_PACKAGE" "$CUDA_COMPAT_VERSION")"
)
log "Installing: ${CUDA_PKGS[*]}"
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades "${CUDA_PKGS[@]}" 2>&1 \
    | tee -a "$INSTALL_LOG" \
    || die "CUDA toolkit install failed."

# ============================================================================
# 6) NCCL (host install, strict +cuda13.0 pin)
# ============================================================================
if [[ "$SKIP_NCCL" != "1" ]]; then
    step "6. NCCL (+cuda${CUDA_MAJOR}.${CUDA_MINOR})"
    NCCL_VER_FROM_BUNDLE="${BUNDLE_NCCL_VERSION:-}"
    if [[ -z "$NCCL_VER_FROM_BUNDLE" || "$NCCL_VER_FROM_BUNDLE" == "skipped" ]]; then
        # Try to resolve at install time from the file:// repo.
        NCCL_VER_FROM_BUNDLE=$(apt-cache madison libnccl2 2>/dev/null \
            | awk -F'|' -v sfx="+cuda${CUDA_MAJOR}.${CUDA_MINOR}" \
                '{ gsub(/^ +| +$/, "", $2); if (index($2, sfx) > 0) { print $2; exit } }')
    fi
    [[ -n "$NCCL_VER_FROM_BUNDLE" ]] \
        || die "Cannot resolve NCCL version with +cuda${CUDA_MAJOR}.${CUDA_MINOR}. Re-gather bundle or set SKIP_NCCL=1."
    log "Installing libnccl2=$NCCL_VER_FROM_BUNDLE + libnccl-dev"
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "libnccl2=$NCCL_VER_FROM_BUNDLE" "libnccl-dev=$NCCL_VER_FROM_BUNDLE" 2>&1 \
        | tee -a "$INSTALL_LOG" \
        || die "NCCL install failed."
else
    step "6. NCCL"
    log "SKIP_NCCL=1 — skipped (PyTorch wheels carry their own nvidia-nccl-cu13)"
fi

# ============================================================================
# 7) DCGM
# ============================================================================
if [[ "$SKIP_DCGM" != "1" ]]; then
    step "7. DCGM"
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "datacenter-gpu-manager-4-cuda${CUDA_MAJOR}" 2>&1 \
        | tee -a "$INSTALL_LOG" \
        || warn "DCGM install failed (non-fatal); skipping."
else
    step "7. DCGM"
    log "SKIP_DCGM=1 — skipped"
fi

# ============================================================================
# 8) apt-mark hold (prevent future apt operations from bumping versions)
# ============================================================================
step "8. apt-mark hold"

mkdir -p "$(dirname "$HELD_PKGS_FILE")"
HELD_PATTERNS=(
    'nvidia-driver-'
    'nvidia-dkms-'
    'nvidia-kernel-'
    'nvidia-fabricmanager-'
    'nvidia-persistenced'
    'libnvidia-'
    'cuda-drivers'
    'cuda-toolkit-'
    'cuda-cudart-'
    'cuda-compat-'
    'libnccl'
    'nvlsm'
    'nvidia-modprobe'
    'datacenter-gpu-manager-'
)

INSTALLED_NV=$(dpkg-query -W -f='${Package}\n' 2>/dev/null \
    | awk -v pats="${HELD_PATTERNS[*]}" '
        BEGIN { n = split(pats, P, " ") }
        { for (i=1; i<=n; i++) if (index($0, P[i]) == 1) { print; next } }' \
    | sort -u)

if [[ -n "$INSTALLED_NV" ]]; then
    printf '%s\n' "$INSTALLED_NV" > "$HELD_PKGS_FILE"
    log "Holding $(printf '%s\n' "$INSTALLED_NV" | wc -l) NVIDIA packages"
    # shellcheck disable=SC2086
    apt-mark hold $INSTALLED_NV >>"$INSTALL_LOG" 2>&1 \
        || warn "apt-mark hold reported errors; check log."
else
    warn "No NVIDIA packages matched hold patterns — install may have failed silently."
fi

# ============================================================================
# 9) Services: enable on boot (DO NOT start now — nvidia.ko not yet loaded)
# ============================================================================
step "9. Services: enable on boot"

# nvidia-fabricmanager unit also spawns NVLSM daemon as a child — no separate
# unit required for nvlsm on a single host. Some NVIDIA builds ship
# nvidia-nvlsm.service; enable it too if present (idempotent).
for svc in nvidia-fabricmanager nvidia-persistenced nvidia-dcgm nvidia-nvlsm; do
    if systemctl cat "$svc" >/dev/null 2>&1; then
        systemctl enable "$svc" >>"$INSTALL_LOG" 2>&1 \
            && log "enabled: $svc" \
            || warn "enable failed: $svc"
    else
        log "skip (unit not present): $svc"
    fi
done

# ============================================================================
# 10) Summary + reboot prompt
# ============================================================================
step "10. Done — reboot required"

INSTALLED_DRV_VER=$(dpkg-query -W -f='${Version}\n' "$DRIVER_PACKAGE" 2>/dev/null || echo "?")
INSTALLED_FM_VER=$(dpkg-query -W -f='${Version}\n' "$FM_PACKAGE" 2>/dev/null || echo "?")
INSTALLED_NVLSM_VER=$(dpkg-query -W -f='${Version}\n' "nvlsm" 2>/dev/null || echo "?")
INSTALLED_CUDA_VER=$(dpkg-query -W -f='${Version}\n' "cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}" 2>/dev/null || echo "?")
INSTALLED_NCCL_VER=$(dpkg-query -W -f='${Version}\n' libnccl2 2>/dev/null || echo "skipped")

log ""
log "  driver  : $INSTALLED_DRV_VER"
log "  fm      : $INSTALLED_FM_VER"
log "  nvlsm   : $INSTALLED_NVLSM_VER"
log "  cuda    : $INSTALLED_CUDA_VER"
log "  nccl    : $INSTALLED_NCCL_VER"
log "  log     : $INSTALL_LOG"
log "  held    : $HELD_PKGS_FILE ($(wc -l < "$HELD_PKGS_FILE" 2>/dev/null || echo 0) pkgs)"
log ""

printf '\n\033[1;33m================================================================\033[0m\n'
printf '\033[1;33m  REBOOT REQUIRED\033[0m\n'
printf '\033[1;33m================================================================\033[0m\n'
printf '  The nvidia kernel module will load on next boot.\n'
printf '  Fabric Manager + NVLSM will then initialize the NVSwitch fabric.\n'
printf '\n'
printf '  After reboot:\n'
printf '    sudo bash test-nvidia.sh           # verify fabric Completed\n'
printf '    sudo bash pre-install-check.sh     # then proceed with userland\n'
printf '    sudo bash install-all.sh\n\n'

if [[ "$SKIP_REBOOT_PROMPT" != "1" ]]; then
    read -r -p "Reboot now? [y/N] " ans
    case "${ans:-N}" in
        y|Y|yes|YES) log "Rebooting..."; sleep 2; systemctl reboot ;;
        *)           log "Skipping reboot. Remember to reboot manually." ;;
    esac
fi

exit 0
