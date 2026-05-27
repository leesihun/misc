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
#     - /etc/modules-load.d/ib-umad.conf for Fabric Manager / NVLSM UMAD access
#     - /etc/modules-load.d/nvidia-peermem.conf for boot-time peermem load
#     - CUDA toolkit MINIMAL subset: cuda-nvcc-13-0, cuda-cudart-13-0,
#       cuda-cudart-dev-13-0, cuda-cccl-13-0, libcublas-13-0, libcublas-dev-13-0,
#       libnvjitlink-13-0, cuda-compat-13-0
#       (we DELIBERATELY avoid cuda-toolkit-13-0 — see step 5 comments)
#     - /etc/profile.d/cuda.sh (nvcc on PATH for login shells)
#     - /etc/ld.so.conf.d/cuda-system.conf (system CUDA libs for non-venv bins)
#     - Optional host libnccl2 / libnccl-dev pinned to +cuda13.0
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
#     SKIP_NCCL=0 bash install-nvidia.sh
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
# SKIP_NCCL default is 1 — system libnccl is a multi-GPU ABI-skew foot-gun
# (see step 6 docstring below). Override with SKIP_NCCL=0 only when you have a
# non-Python consumer that must dlopen libnccl.so.2 from /usr. Setting the
# default ONCE here (not at the step 6 site, where `${VAR:-1}` would be a
# no-op because line 56 already gave it a non-null value) so the policy is
# consistent across the sudo re-exec at line 77 and every later reference.
SKIP_NCCL="${SKIP_NCCL:-1}"
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
    # Explicit env whitelist — sudo's env_reset drops everything else even
    # with -E, so we forward every knob this script reads. Missing one here
    # caused SKIP_INNER_SHA256/EXTRACT_PARENT/etc. to silently revert to
    # default values after sudo, defeating the override.
    exec sudo -E env \
        BUNDLE_DIR="$BUNDLE_DIR" \
        BUNDLE_BIN="$BUNDLE_BIN" \
        EXTRACT_PARENT="${EXTRACT_PARENT:-}" \
        APT_REPO_DIR="$APT_REPO_DIR" \
        INSTALL_LOG="$INSTALL_LOG" \
        HELD_PKGS_FILE="$HELD_PKGS_FILE" \
        DRIVER_BRANCH="$DRIVER_BRANCH" \
        CUDA_MAJOR="$CUDA_MAJOR" \
        CUDA_MINOR="$CUDA_MINOR" \
        SKIP_REBOOT_PROMPT="$SKIP_REBOOT_PROMPT" \
        SKIP_NCCL="$SKIP_NCCL" \
        SKIP_DCGM="$SKIP_DCGM" \
        SKIP_INNER_SHA256="${SKIP_INNER_SHA256:-0}" \
        bash "$0" "$@"
fi

. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] \
    || die "Target must be Ubuntu 24.04 (found ${PRETTY_NAME:-?})."

# Cache the running kernel release once. This script can run for several
# minutes, and using `$(uname -r)` inline in every path expression both
# clutters the code and (very unlikely but possible) introduces inconsistency
# if the kernel were swapped mid-run. The 9b verification block alone uses
# this value 4+ times.
RUNNING_KERNEL="$(uname -r)"
readonly RUNNING_KERNEL

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

# ============================================================================
# 1b. Inner SHA256SUMS — verify the extracted bundle, even when BUNDLE_DIR
#     is supplied directly (skipping the .bin path means the outer sidecar
#     check at step 1 never ran).
#
# meta/SHA256SUMS was generated by gather-nvidia.sh AFTER all .deb downloads
# completed, so a mismatch here means either:
#   - someone modified the staging tree after gather
#   - the transfer truncated/corrupted a deb
#   - this is an older bundle without SHA256SUMS (warn-only)
# Override with SKIP_INNER_SHA256=1 only for explicit forensic re-runs.
# ============================================================================
SKIP_INNER_SHA256="${SKIP_INNER_SHA256:-0}"
INNER_SUMS="$BUNDLE_DIR/meta/SHA256SUMS"
if [[ "$SKIP_INNER_SHA256" == "1" ]]; then
    warn "SKIP_INNER_SHA256=1 — skipping integrity check on extracted bundle."
elif [[ ! -f "$INNER_SUMS" ]]; then
    warn "$INNER_SUMS missing — older bundle? Skipping inner integrity check."
else
    log "Verifying meta/SHA256SUMS ($(wc -l < "$INNER_SUMS" | tr -d ' ') entries)"
    if ( cd "$BUNDLE_DIR" && sha256sum -c --quiet "meta/SHA256SUMS" ) \
        >> "$INSTALL_LOG" 2>&1; then
        log "Inner SHA256SUMS verified"
    else
        die "meta/SHA256SUMS mismatch — bundle is corrupt or was modified after gather. See $INSTALL_LOG for the failing entries."
    fi
fi

# Validate bundle metadata. This is the contract between gather-nvidia.sh and
# install-nvidia.sh; without it the installer would fall back to unpinned apt
# candidates and could recreate the driver/userspace version drift this split
# bundle is meant to prevent.
TARGET_ENV="$BUNDLE_DIR/meta/target.env"
[[ -f "$TARGET_ENV" ]] || die "$TARGET_ENV missing — rebuild the NVIDIA bundle with gather-nvidia.sh."
set -a; . "$TARGET_ENV"; set +a
[[ "${BUNDLE_VARIANT:-}" == "nvidia-stack" ]] \
    || die "Bundle variant '${BUNDLE_VARIANT:-<unset>}' — expected 'nvidia-stack'. Wrong bundle?"

_missing_meta=()
for key in \
    BUNDLE_DRIVER_BRANCH \
    BUNDLE_DRIVER_VERSION \
    BUNDLE_DRIVER_PACKAGE \
    BUNDLE_DRIVER_PACKAGE_VERSION \
    BUNDLE_CUDA_DRIVERS_PACKAGE \
    BUNDLE_CUDA_DRIVERS_VERSION \
    BUNDLE_FABRICMANAGER_PACKAGE \
    BUNDLE_FABRICMANAGER_VERSION \
    BUNDLE_NSCQ_PACKAGE \
    BUNDLE_NSCQ_VERSION \
    BUNDLE_CUDA_COMPAT_PACKAGE \
    BUNDLE_CUDA_COMPAT_VERSION \
    BUNDLE_CUDA; do
    val="${!key:-}"
    [[ -n "$val" && "$val" != "skipped" ]] || _missing_meta+=( "$key" )
done
if (( ${#_missing_meta[@]} > 0 )); then
    die "Bundle metadata missing required NVIDIA pin(s): ${_missing_meta[*]}. Rebuild with gather-nvidia.sh."
fi
DRIVER_BRANCH="${BUNDLE_DRIVER_BRANCH#R}"
if [[ "${BUNDLE_CUDA:-}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    CUDA_MAJOR="${BASH_REMATCH[1]}"
    CUDA_MINOR="${BASH_REMATCH[2]}"
else
    die "Bundle metadata has invalid BUNDLE_CUDA='${BUNDLE_CUDA:-<unset>}'"
fi
log "Bundle: ${BUNDLE_VARIANT}  driver=${BUNDLE_DRIVER_VERSION:-?}  cuda=${BUNDLE_CUDA:-?}  nccl=${BUNDLE_NCCL_VERSION:-skipped}"

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

_repo_first_version_with_suffix() {
    local pkg="$1" suffix="$2"
    awk -v want_pkg="$pkg" -v want_suffix="$suffix" '
        /^Package: / { pkg=$2; next }
        /^Version: / && pkg == want_pkg && index($2, want_suffix) > 0 { print $2; exit }
    ' "$APT_REPO_DIR/Packages"
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

# Pin nvidia/cuda/libnvidia packages to our file:// repo so a later
# install-all.sh cannot accidentally bump them via a different source.
# libnccl* is included only as a guard for the explicit SKIP_NCCL=0 path.
cat > "$APT_PIN_FILE" <<EOF
Package: nvidia-* cuda-* libnvidia-* libnvjit* libnvfat* libnccl* nvlsm datacenter-gpu-manager-*
Pin: origin ""
Pin-Priority: 1001
EOF
log "Wrote $APT_PIN_FILE"

log "Refreshing apt indexes (file:// bundle only — hermetic)"
# Use ONLY the bundle's sources.list.d entry for every nvidia transaction.
# The pin at /etc/apt/preferences.d/99-nvidia-prefer-bundle would normally
# protect us, but on a target with a stale or accidentally-online apt list,
# transitive deps could still arrive from a non-bundle origin. Restricting
# the sources list at install time makes that impossible.
#
# APT_SRC_OPTS is reused below by every apt-get install call for nvidia
# packages, so changes here propagate consistently.
APT_SRC_OPTS=(
    -o "Dir::Etc::sourcelist=$APT_LIST_FILE"
    -o "Dir::Etc::sourceparts=-"
    -o "APT::Get::List-Cleanup=0"
)
apt-get update "${APT_SRC_OPTS[@]}" >>"$INSTALL_LOG" 2>&1 \
    || die "apt update from bundle file:// repo failed. Check $INSTALL_LOG."

# ============================================================================
# 3) Pre-driver: nouveau blacklist
# ============================================================================
step "3. Pre-driver: nouveau blacklist"

NOUVEAU_BL=/etc/modprobe.d/blacklist-nouveau-nvidia.conf
_nouveau_file_existed=0
if [[ -f "$NOUVEAU_BL" ]]; then
    _nouveau_file_existed=1
    log "$NOUVEAU_BL already present (overwriting to ensure exact content)"
fi
cat > "$NOUVEAU_BL" <<'EOF'
# Installed by install-nvidia.sh — required for NVIDIA driver to load.
blacklist nouveau
options nouveau modeset=0
EOF
chmod 0644 "$NOUVEAU_BL"
(( _nouveau_file_existed )) || log "Wrote $NOUVEAU_BL"

# Always re-run update-initramfs. It's idempotent and cheap, and skipping it
# on a re-run where the file existed but a prior initramfs build failed (or
# was never run) would leave nouveau bundled in the initramfs even though
# /etc/modprobe.d says blacklist. Cost: ~5–15 s; benefit: a guaranteed
# nouveau-free initramfs on every install attempt.
if command -v update-initramfs >/dev/null; then
    log "Regenerating initramfs to apply nouveau blacklist"
    update-initramfs -u >>"$INSTALL_LOG" 2>&1 \
        || warn "update-initramfs failed; nouveau may still load on next boot. See $INSTALL_LOG."
else
    warn "update-initramfs not found — initramfs may still contain nouveau on next boot."
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
    "${APT_SRC_OPTS[@]}" \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    "${CORE_PKGS[@]}" 2>&1 | tee -a "$INSTALL_LOG" \
    || die "Core NVIDIA install failed. Check $INSTALL_LOG."

# ============================================================================
# 4b) Auto-load fabric/RDMA helper modules on boot
#
#     ib_umad is mandatory for NVLSM/OpenSM and Fabric Manager on NVL5+
#     systems. Without it, nvlsm fails with:
#       can't read /sys/class/infiniband_mad/abi_version
#     and Fabric Manager refuses to start.
#
#     nvidia-peermem ships with nvidia-driver-*-open and is needed for
#     GPUDirect RDMA via DOCA-OFED. The softdep asks modprobe to load it after
#     mlx5_core has registered its peer-memory interface.
# ============================================================================
IB_UMAD_LOAD=/etc/modules-load.d/ib-umad.conf
if ! [[ -f "$IB_UMAD_LOAD" ]] || ! grep -qE '^[[:space:]]*ib_umad[[:space:]]*$' "$IB_UMAD_LOAD"; then
    printf 'ib_umad\n' > "$IB_UMAD_LOAD"
    log "Wrote $IB_UMAD_LOAD"
else
    log "$IB_UMAD_LOAD already present"
fi

PEERMEM_LOAD=/etc/modules-load.d/nvidia-peermem.conf
if ! [[ -f "$PEERMEM_LOAD" ]]; then
    printf 'nvidia-peermem\n' > "$PEERMEM_LOAD"
    log "Wrote $PEERMEM_LOAD"
else
    log "$PEERMEM_LOAD already present"
fi

PEERMEM_SOFTDEP=/etc/modprobe.d/nvidia-peermem-softdep.conf
if ! [[ -f "$PEERMEM_SOFTDEP" ]] \
        || ! grep -qE '^[[:space:]]*softdep[[:space:]]+mlx5_core[[:space:]]+post:[[:space:]]+nvidia-peermem' "$PEERMEM_SOFTDEP"; then
    cat > "$PEERMEM_SOFTDEP" <<'EOF'
# Installed by install-nvidia.sh — ensure nvidia-peermem loads AFTER the
# DOCA-OFED mlx5_core driver registers its peer-memory interface. The
# /etc/modules-load.d/nvidia-peermem.conf entry handles the first-boot
# autoload, but does NOT enforce ordering against mlx5_core. modprobe
# softdep is a non-binding hint with zero downside — if mlx5_core is
# already up, modprobe is a no-op; if not, peermem retries after it.
softdep mlx5_core post: nvidia-peermem
EOF
    chmod 0644 "$PEERMEM_SOFTDEP"
    log "Wrote $PEERMEM_SOFTDEP"
else
    log "$PEERMEM_SOFTDEP already present"
fi

# ============================================================================
# 4c) Fabric Manager / NVLSM ownership
#
#   R580 Fabric Manager on NVL5+ starts NVLSM through
#   nvidia-fabricmanager-start.sh, passing the selected HCA GUID, pid file, and
#   /usr/share/nvidia/nvlsm/nvlsm.conf. A separate service generated by older
#   versions of this script races that wrapper and produces:
#       Perhaps another instance of OpenSM is already running
#
#   Clean up only units/drop-ins that carry this installer's marker. Do not
#   touch a future vendor-provided nvidia-nvlsm.service.
# ============================================================================
step "4c. Fabric Manager owns NVLSM"

NVLSM_UNIT_PATH=/etc/systemd/system/nvidia-nvlsm.service
FM_DROPIN_DIR=/etc/systemd/system/nvidia-fabricmanager.service.d
FM_DROPIN_FILE="$FM_DROPIN_DIR/10-after-nvlsm.conf"
LEGACY_FM_CLEAN=0
LEGACY_BACKUP_DIR=/var/backups/nvidia-airgap-installer

if [[ -f "$NVLSM_UNIT_PATH" ]] && grep -q 'Installed by install-nvidia.sh' "$NVLSM_UNIT_PATH"; then
    mkdir -p "$LEGACY_BACKUP_DIR"
    systemctl disable --now nvidia-nvlsm >>"$INSTALL_LOG" 2>&1 || true
    backup="$LEGACY_BACKUP_DIR/nvidia-nvlsm.service.$(date -u +%Y%m%dT%H%M%SZ).bak"
    mv "$NVLSM_UNIT_PATH" "$backup" \
        && log "Moved legacy custom nvidia-nvlsm.service to $backup" \
        || warn "Could not move legacy $NVLSM_UNIT_PATH; Fabric Manager may race NVLSM."
    LEGACY_FM_CLEAN=1
fi

if [[ -f "$FM_DROPIN_FILE" ]] && grep -q 'Installed by install-nvidia.sh' "$FM_DROPIN_FILE"; then
    mkdir -p "$LEGACY_BACKUP_DIR"
    backup="$LEGACY_BACKUP_DIR/10-after-nvlsm.conf.$(date -u +%Y%m%dT%H%M%SZ).bak"
    mv "$FM_DROPIN_FILE" "$backup" \
        && log "Moved legacy Fabric Manager drop-in to $backup" \
        || warn "Could not move legacy $FM_DROPIN_FILE; Fabric Manager may still pull nvidia-nvlsm."
    rmdir "$FM_DROPIN_DIR" 2>/dev/null || true
    LEGACY_FM_CLEAN=1
fi

if (( LEGACY_FM_CLEAN )); then
    systemctl daemon-reload >>"$INSTALL_LOG" 2>&1 \
        || warn "systemctl daemon-reload failed after legacy NVLSM cleanup."
else
    log "No legacy installer-created nvidia-nvlsm service/drop-in found"
fi

# ============================================================================
# 5) CUDA toolkit — MINIMAL: only what llama.cpp build needs.
#
# We deliberately AVOID cuda-toolkit-${MAJOR}-${MINOR} (the ~3GB meta-package
# pulling cuFFT, cuSOLVER, cuSPARSE, NPP, nvJPEG, nvJitLink, samples, etc.).
# Rationale: PyTorch/vLLM/training/jupyter venvs ship their own CUDA runtime
# via pip's nvidia-*-cu13 packages (libcudart, libcublas, libcudnn, libnccl).
# Installing the full system toolkit means /usr/local/cuda/lib64 ends up in
# the linker search and silently shadows venv-bundled libs (esp. libnccl),
# producing the classic multi-GPU "vLLM hangs at first all_reduce" symptom.
#
# Only llama.cpp builds against the system toolkit; it needs nvcc + cudart
# + cublas + thrust headers. Everything else stays venv-local.
# ============================================================================
step "5. CUDA toolkit (minimal) ${CUDA_MAJOR}.${CUDA_MINOR}"

CUDA_PKGS=(
    "cuda-nvcc-${CUDA_MAJOR}-${CUDA_MINOR}"            # nvcc compiler
    "cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}"          # libcudart.so runtime
    "cuda-cudart-dev-${CUDA_MAJOR}-${CUDA_MINOR}"      # headers + static
    "cuda-cccl-${CUDA_MAJOR}-${CUDA_MINOR}"            # Thrust/CUB headers
    "libcublas-${CUDA_MAJOR}-${CUDA_MINOR}"            # libcublas.so + libcublasLt.so
    "libcublas-dev-${CUDA_MAJOR}-${CUDA_MINOR}"        # cublas headers
    "libnvjitlink-${CUDA_MAJOR}-${CUDA_MINOR}"         # JIT-link (cublasLt loads it)
    "$(_pkg_with_version "$CUDA_COMPAT_PACKAGE" "$CUDA_COMPAT_VERSION")"
)
log "Installing (minimal toolkit, ~600MB instead of ~3GB): ${CUDA_PKGS[*]}"
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
    "${APT_SRC_OPTS[@]}" "${CUDA_PKGS[@]}" 2>&1 \
    | tee -a "$INSTALL_LOG" \
    || die "CUDA toolkit install failed."

# ============================================================================
# 5b) CUDA environment wiring — PATH for login shells + ld.so.cache entry.
#
# These two files belong to the NVIDIA stack (not userland) because they
# expose the system CUDA toolkit we just installed. Moved here from
# install-all.sh so the nvidia/userland boundary stays clean: install-all.sh
# can be re-run, downgraded, or split into phases without affecting nvcc/cuda
# discovery.
#
# We intentionally do NOT touch LD_LIBRARY_PATH here. Prepending
# /usr/local/cuda/lib64 globally would override the RUNPATH baked into PyTorch
# wheels and silently replace their bundled libnccl / libcudnn with the system
# copies, producing multi-GPU CUDA/NCCL ABI skew at process start.
# PyTorch wheels resolve their own CUDA libs via $ORIGIN RUNPATH; leave them
# alone. ld.so.cache is searched AFTER RUNPATH, so the entry below only
# affects binaries with no RUNPATH (llama-server, llama-cli).
# ============================================================================
step "5b. CUDA env wiring (/etc/profile.d/cuda.sh, /etc/ld.so.conf.d/cuda-system.conf)"

PROFILE_CUDA=/etc/profile.d/cuda.sh
tee "$PROFILE_CUDA" > /dev/null <<'CUDA_PATH'
# Installed by install-nvidia.sh -- nvcc on PATH for login shells.
# Intentionally does NOT modify LD_LIBRARY_PATH; venv-bundled CUDA libs win
# via RUNPATH, and system libs (libcudart/libcublas) are reachable through
# /etc/ld.so.conf.d/cuda-system.conf below.
if [ -d /usr/local/cuda/bin ]; then
    case ":$PATH:" in
        *:/usr/local/cuda/bin:*) : ;;
        *) export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} ;;
    esac
fi
CUDA_PATH
chmod 0644 "$PROFILE_CUDA"
log "Wrote $PROFILE_CUDA (PATH only — LD_LIBRARY_PATH intentionally NOT set)"

LDSO_CONF=/etc/ld.so.conf.d/cuda-system.conf
if [[ -d /usr/local/cuda/lib64 ]]; then
    tee "$LDSO_CONF" > /dev/null <<'EOF'
# Installed by install-nvidia.sh -- expose system CUDA libs for non-venv binaries
# (llama-cli, llama-server). Searched AFTER RUNPATH so venv-bundled libs win.
/usr/local/cuda/lib64
EOF
    chmod 0644 "$LDSO_CONF"
    ldconfig 2>/dev/null || warn "ldconfig failed; system CUDA libs may not be discoverable until reboot."
    log "Wrote $LDSO_CONF and ran ldconfig"
else
    warn "/usr/local/cuda/lib64 missing — CUDA toolkit install may have failed; skipping ld.so.conf.d entry."
fi

# ============================================================================
# 6) NCCL — DEFAULT SKIPPED.
#
# System libnccl is a known foot-gun: when /usr/local/cuda/lib64 ends up in
# the dynamic-linker search path, system libnccl.so.2 silently overrides the
# NCCL version PyTorch wheels bundle (nvidia-nccl-cu13). That ABI skew
# produces multi-GPU hangs at the first all_reduce. Letting each venv carry
# its own matched NCCL eliminates this entire failure class.
#
# Set SKIP_NCCL=0 only if you have a non-Python consumer that must dlopen
# libnccl.so.2 from /usr (rare; almost no one outside MPI test harnesses).
# Default is set ONCE at the top of this script (see SKIP_NCCL near line 56).
# ============================================================================
if [[ "$SKIP_NCCL" != "1" ]]; then
    step "6. NCCL (+cuda${CUDA_MAJOR}.${CUDA_MINOR})"
    warn "SKIP_NCCL=0: installing system libnccl. Verify your venvs do NOT also bundle NCCL, or expect multi-GPU ABI skew."
    NCCL_VER_FROM_BUNDLE="${BUNDLE_NCCL_VERSION:-}"
    if [[ -z "$NCCL_VER_FROM_BUNDLE" || "$NCCL_VER_FROM_BUNDLE" == "skipped" ]]; then
        # Resolve only from the extracted bundle Packages index. Do not consult
        # global apt metadata here; this target is airgapped and may have stale
        # lists from a prior online install.
        NCCL_VER_FROM_BUNDLE="$(_repo_first_version_with_suffix "libnccl2" "+cuda${CUDA_MAJOR}.${CUDA_MINOR}")"
    fi
    [[ -n "$NCCL_VER_FROM_BUNDLE" ]] \
        || die "Cannot resolve NCCL version with +cuda${CUDA_MAJOR}.${CUDA_MINOR} inside the bundle. Re-gather with SKIP_NCCL=0 or run install-nvidia.sh with SKIP_NCCL=1."
    _repo_has_version "libnccl2" "$NCCL_VER_FROM_BUNDLE" \
        || die "Bundle metadata requests libnccl2=$NCCL_VER_FROM_BUNDLE but debs/Packages does not contain it."
    _repo_has_version "libnccl-dev" "$NCCL_VER_FROM_BUNDLE" \
        || die "Bundle metadata requests libnccl-dev=$NCCL_VER_FROM_BUNDLE but debs/Packages does not contain it."
    log "Installing libnccl2=$NCCL_VER_FROM_BUNDLE + libnccl-dev"
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        --allow-downgrades \
        "${APT_SRC_OPTS[@]}" \
        "libnccl2=$NCCL_VER_FROM_BUNDLE" "libnccl-dev=$NCCL_VER_FROM_BUNDLE" 2>&1 \
        | tee -a "$INSTALL_LOG" \
        || die "NCCL install failed."
else
    step "6. NCCL"
    log "SKIP_NCCL=1 (default) — host NCCL skipped; PyTorch/vLLM venvs use their bundled nvidia-nccl-cu13."
fi

# ============================================================================
# 7) DCGM
# ============================================================================
if [[ "$SKIP_DCGM" != "1" ]]; then
    step "7. DCGM"
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "${APT_SRC_OPTS[@]}" \
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
    # 'nvidia-fabricmanager' with NO trailing hyphen — index() is a strict
    # prefix match, and the package NVIDIA actually ships for R580 is the
    # unsuffixed `nvidia-fabricmanager` (not `nvidia-fabricmanager-580`).
    # The trailing-hyphen form would only match a `-580` variant and would
    # SILENTLY skip holding the FM package that's actually installed,
    # allowing a later userland apt run to upgrade/downgrade it. Without the
    # hyphen, this still matches `nvidia-fabricmanager-580` if it ever exists.
    'nvidia-fabricmanager'
    'nvidia-persistenced'
    'libnvidia-'
    'cuda-drivers'
    'cuda-nvcc-'
    'cuda-cudart-'
    'cuda-cccl-'
    'cuda-compat-'
    'libcublas-'
    'libnvjitlink-'
    # Present only when SKIP_NCCL=0 was used; harmless when absent.
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

# On this R580/NVL5+ stack, nvidia-fabricmanager-start.sh owns the NVLSM
# daemon. Do not enable a separate nvidia-nvlsm.service; it races Fabric
# Manager's wrapper.
for svc in nvidia-fabricmanager nvidia-persistenced nvidia-dcgm; do
    if systemctl cat "$svc" >/dev/null 2>&1; then
        systemctl enable "$svc" >>"$INSTALL_LOG" 2>&1 \
            && log "enabled: $svc" \
            || warn "enable failed: $svc"
    else
        log "skip (unit not present): $svc"
    fi
done

# ============================================================================
# 9b) Pre-reboot verification — fail fast on common silent install failures
#
# The driver kernel module doesn't load until reboot, so we can't smoke-test
# the GPU here. But several artifacts CAN be checked right now, and catching
# them pre-reboot is much cheaper than discovering after reboot that
# nvidia.ko was never written, peermem isn't in initramfs, or the FM unit
# file vanished.
# ============================================================================
step "9b. Pre-reboot verification"

VERIFY_RC=0
_verify_fail() { VERIFY_RC=1; warn "verify FAIL: $*"; }
_verify_ok()   { log  "verify ok  : $*"; }

# (a) Every CORE_PKG must be installed (state ii or hi).
for pkg_spec in "${CORE_PKGS[@]}"; do
    pkg="${pkg_spec%%=*}"
    state=$(dpkg-query -W -f='${db:Status-Abbrev}' "$pkg" 2>/dev/null || true)
    case "${state%% *}" in
        ii|hi) _verify_ok "dpkg $pkg ($(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null))" ;;
        *)     _verify_fail "dpkg $pkg state='$state' — expected ii/hi" ;;
    esac
done

# (b) nvidia.ko present in the running kernel's module tree. For
# nvidia-driver-*-open the .ko is bundled in the .deb (no DKMS build).
KMOD_PATHS=(
    "/lib/modules/${RUNNING_KERNEL}/kernel/drivers/video/nvidia.ko"
    "/lib/modules/${RUNNING_KERNEL}/kernel/drivers/video/nvidia.ko.zst"
    "/lib/modules/${RUNNING_KERNEL}/kernel/drivers/video/nvidia.ko.xz"
    "/lib/modules/${RUNNING_KERNEL}/updates/dkms/nvidia.ko"
    "/lib/modules/${RUNNING_KERNEL}/updates/dkms/nvidia.ko.zst"
)
_kmod_found=""
for kmod in "${KMOD_PATHS[@]}"; do
    [[ -f "$kmod" ]] && { _kmod_found="$kmod"; break; }
done
if [[ -n "$_kmod_found" ]]; then
    _verify_ok "kmod nvidia.ko present at $_kmod_found"
else
    # Fallback: any path under /lib/modules/${RUNNING_KERNEL} containing "nvidia.ko"
    _kmod_alt=$(find "/lib/modules/${RUNNING_KERNEL}" -name 'nvidia.ko*' 2>/dev/null | head -1)
    if [[ -n "$_kmod_alt" ]]; then
        _verify_ok "kmod nvidia.ko present at $_kmod_alt"
    else
        _verify_fail "nvidia.ko not found under /lib/modules/${RUNNING_KERNEL} — driver will not load on reboot"
    fi
fi

# (c) initramfs no longer references nouveau as a module to load. lsinitramfs
# may not be present on minimal installs — fall back to a grep of the binary.
INITRAMFS="/boot/initrd.img-${RUNNING_KERNEL}"
if [[ -f "$INITRAMFS" ]]; then
    if command -v lsinitramfs >/dev/null 2>&1; then
        if lsinitramfs "$INITRAMFS" 2>/dev/null | grep -qE '/nouveau\.ko'; then
            warn "initramfs still contains nouveau.ko at $INITRAMFS — blacklist will prevent load but module is bundled. Acceptable (modprobe.d blacklist wins) but verify on next boot."
        fi
        _verify_ok "initramfs present at $INITRAMFS"
    else
        _verify_ok "initramfs present at $INITRAMFS (lsinitramfs not available for content check)"
    fi
else
    warn "initramfs $INITRAMFS missing — likely a kernel mismatch or update-initramfs failed"
fi

# (d) modules-load.d entry + config files we wrote.
# Use explicit if/else rather than A && B || C — the latter would incorrectly
# fire C if B (the "ok" path) ever returned non-zero, masking a real error.
for f in "$NOUVEAU_BL" "$IB_UMAD_LOAD" "$PEERMEM_LOAD" "$PEERMEM_SOFTDEP" "$PROFILE_CUDA" "$LDSO_CONF" "$APT_LIST_FILE" "$APT_PIN_FILE"; do
    if [[ -f "$f" ]]; then
        _verify_ok "config $f"
    else
        _verify_fail "config $f missing"
    fi
done

if [[ -f "$NVLSM_UNIT_PATH" ]] && grep -q 'Installed by install-nvidia.sh' "$NVLSM_UNIT_PATH"; then
    _verify_fail "legacy custom $NVLSM_UNIT_PATH still present"
else
    _verify_ok "no legacy custom nvidia-nvlsm.service conflict"
fi
if [[ -f "$FM_DROPIN_FILE" ]] && grep -q 'Installed by install-nvidia.sh' "$FM_DROPIN_FILE"; then
    _verify_fail "legacy custom $FM_DROPIN_FILE still present"
else
    _verify_ok "no legacy Fabric Manager nvidia-nvlsm drop-in conflict"
fi

if grep -qw 'ib_register_peer_memory_client' /proc/kallsyms 2>/dev/null \
        && grep -qw 'ib_unregister_peer_memory_client' /proc/kallsyms 2>/dev/null; then
    _verify_ok "RDMA peer-memory symbols present for nvidia-peermem"
else
    _verify_fail "RDMA peer-memory symbols missing; nvidia-peermem will fail with Unknown symbol. Repair DOCA-OFED before accepting this install."
fi

# (e) Critical services enabled (not necessarily active — they need reboot).
# Fabric Manager owns NVLSM on this R580/NVL5+ package set.
for svc in nvidia-fabricmanager nvidia-persistenced; do
    if systemctl is-enabled "$svc" >/dev/null 2>&1; then
        _verify_ok "service $svc enabled"
    else
        _verify_fail "service $svc NOT enabled — fabric will not init on next boot"
    fi
done

# (f) apt-mark hold actually applied.
held_count=$(dpkg --get-selections 2>/dev/null | awk '$2 == "hold" {n++} END {print n+0}')
if (( held_count > 0 )); then
    _verify_ok "apt-mark hold applied to $held_count package(s)"
else
    _verify_fail "no apt-mark holds in effect — userland install can upgrade nvidia stack"
fi

# (g) nvcc on disk (PATH won't work yet — /etc/profile.d/cuda.sh only loads on next login)
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    _verify_ok "nvcc at /usr/local/cuda/bin/nvcc ($(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1))"
else
    _verify_fail "/usr/local/cuda/bin/nvcc missing or not executable"
fi

if (( VERIFY_RC != 0 )); then
    warn ""
    warn "One or more pre-reboot verifications FAILED. Reboot will NOT magically fix these."
    warn "Inspect $INSTALL_LOG, fix the missing artifacts, then re-run install-nvidia.sh."
    warn ""
    # We don't die() — we want the summary block below to still print so the
    # operator sees the full picture. But the exit code at the bottom reflects
    # the failure.
fi

# ============================================================================
# 10) Summary + reboot prompt
# ============================================================================
step "10. Done — reboot required"

INSTALLED_DRV_VER=$(dpkg-query -W -f='${Version}\n' "$DRIVER_PACKAGE" 2>/dev/null || echo "?")
INSTALLED_FM_VER=$(dpkg-query -W -f='${Version}\n' "$FM_PACKAGE" 2>/dev/null || echo "?")
INSTALLED_NVLSM_VER=$(dpkg-query -W -f='${Version}\n' "nvlsm" 2>/dev/null || echo "?")
# Query cuda-nvcc-* (actually installed by step 5) rather than the
# cuda-toolkit-* metapackage we deliberately avoid. The metapackage check
# would always return "?" and mislead operators reading the summary.
INSTALLED_CUDA_VER=$(dpkg-query -W -f='${Version}\n' "cuda-nvcc-${CUDA_MAJOR}-${CUDA_MINOR}" 2>/dev/null || echo "?")
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

if (( VERIFY_RC != 0 )); then
    warn "Skipping reboot prompt because pre-reboot verification failed (step 9b)."
    warn "Fix the issues above before rebooting; rebooting now will likely produce a non-functional driver."
    exit "$VERIFY_RC"
fi

if [[ "$SKIP_REBOOT_PROMPT" != "1" ]]; then
    read -r -p "Reboot now? [y/N] " ans
    case "${ans:-N}" in
        y|Y|yes|YES) log "Rebooting..."; sleep 2; systemctl reboot ;;
        *)           log "Skipping reboot. Remember to reboot manually." ;;
    esac
fi

exit 0
