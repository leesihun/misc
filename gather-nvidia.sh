#!/usr/bin/env bash
# ============================================================================
# gather-nvidia.sh
#
#   Run on an internet-connected WSL Ubuntu 24.04 machine. Builds the NVIDIA
#   stack airgap bundle for an Ubuntu 24.04 server with 8x B300 GPUs +
#   4th-gen NVSwitch + ConnectX-7/8 + DOCA-OFED pre-installed.
#
#   This bundle is SEPARATE from gather-all.sh — install-nvidia.sh runs
#   FIRST on the target, then a reboot, then the existing install-all.sh
#   from the userland bundle takes over.
#
#   What this bundles (R580 LTS, 580.159.04+ at time of writing):
#     - nvidia-driver-580-open + open kmod stack (incl. nvidia-peermem.ko,
#       nvidia-persistenced — both ride transitively, no separate apt pkg)
#     - cuda-drivers-580 plus the matching fabricmanager/NSCQ packages
#     - nvlsm (4th-gen NVSwitch — runs as a child of nvidia-fabricmanager)
#     - nvidia-modprobe
#     - cuda-toolkit-13-0 (toolkit only — NOT cuda/cuda-13-0 metapkgs that
#       drag the driver back in)
#     - libnccl2 / libnccl-dev pinned to +cuda13.0 suffix
#     - datacenter-gpu-manager-4-cuda13 (DCGM 4.3.x+)
#     - nvidia-driver-pinning-580 (apt unattended-upgrade guard)
#     - Full transitive .deb closure via apt-rdepends
#
#   What this does NOT bundle:
#     - DOCA-OFED — vendor pre-installed it (verified by pre-install-nvidia.sh)
#     - Userland (xfce4, python, vscode, etc.) — that's gather-all.sh
#     - cuda/cuda-13-0/nvidia-open metapackages that pull the driver in again
#
#   Output:
#     - ~/GPU_server_downloads_nvidia/                          (staging)
#     - ~/nvidia-airgap-bundle-ubuntu24.04.bin                  (single tar.gz)
#     - ~/nvidia-airgap-bundle-ubuntu24.04.bin.sha256           (sidecar)
#     - ~/install-nvidia.sh, ~/pre-install-nvidia.sh, ~/test-nvidia.sh
#
#   Usage:
#     bash gather-nvidia.sh                          # latest 580.x.y
#     DRIVER_BRANCH=580 bash gather-nvidia.sh        # explicit branch
#     PIN_DRIVER_VER=580.159.04 bash gather-nvidia.sh # pin a specific version
#     SKIP_NCCL=1 bash gather-nvidia.sh              # rely on PyTorch wheel NCCL
# ============================================================================
set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

OUT_DIR="${OUT_DIR:-$HOME/GPU_server_downloads_nvidia}"
APT_LOCK_TIMEOUT="${APT_LOCK_TIMEOUT:-900}"
APT_DOWNLOAD_RETRIES="${APT_DOWNLOAD_RETRIES:-5}"
APT_STAGE_CACHE_DIR="$OUT_DIR/.apt-cache"
APT_STAGE_ARCHIVES_DIR="$APT_STAGE_CACHE_DIR/archives"

DRIVER_BRANCH="${DRIVER_BRANCH:-580}"          # R580 LTS until 2028-06
DRIVER_FLAVOR="${DRIVER_FLAVOR:-open}"         # Blackwell requires open kernel modules
PIN_DRIVER_VER="${PIN_DRIVER_VER:-}"            # empty = latest in repo
SKIP_NCCL="${SKIP_NCCL:-0}"
CUDA_MAJOR="${CUDA_MAJOR:-13}"
CUDA_MINOR="${CUDA_MINOR:-0}"
NCCL_CUDA_SUFFIX="${NCCL_CUDA_SUFFIX:-cuda${CUDA_MAJOR}.${CUDA_MINOR}}"

# Target OS must match gather host for .deb compatibility.
TARGET_OS_VERSION="${TARGET_OS_VERSION:-$(. /etc/os-release && echo "$VERSION_ID")}"
TARGET_ARCH="${TARGET_ARCH:-$(dpkg --print-architecture)}"   # amd64

# Upstream NVIDIA CUDA apt repo for Ubuntu 24.04 x86_64.
NVIDIA_REPO_URL="${NVIDIA_REPO_URL:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${TARGET_OS_VERSION//./}/x86_64}"
NVIDIA_REPO_LIST="/etc/apt/sources.list.d/cuda-ubuntu${TARGET_OS_VERSION//./}-x86_64.list"
NVIDIA_KEYRING_DEB_URL="${NVIDIA_KEYRING_DEB_URL:-${NVIDIA_REPO_URL}/cuda-keyring_1.1-1_all.deb}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Helpers
# ============================================================================
log()  { printf '\033[1;36m[gather-nv]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gather-nv:WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[gather-nv:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;35m══ %s ══\033[0m\n' "$*"; }

_apt_get_wait() {
    sudo apt-get -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" "$@"
}

_apt_stage_prepare() {
    mkdir -p "$APT_STAGE_ARCHIVES_DIR/partial" "$OUT_DIR/debs" "$OUT_DIR/meta"
}

_apt_get_stage() {
    _apt_stage_prepare
    sudo apt-get \
        -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" \
        -o "Dir::Cache=${APT_STAGE_CACHE_DIR}" \
        -o "Dir::Cache::archives=${APT_STAGE_ARCHIVES_DIR}" \
        "$@"
}

_apt_download_stage() {
    local pkg="$1" attempt delay=10 tmp_err="$OUT_DIR/meta/.apt-download.err"
    _apt_stage_prepare
    for ((attempt=1; attempt<=APT_DOWNLOAD_RETRIES; attempt++)); do
        if (
            cd "$OUT_DIR/debs"
            sudo apt-get \
                -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" \
                -o "Dir::Cache=${APT_STAGE_CACHE_DIR}" \
                -o "Dir::Cache::archives=${APT_STAGE_ARCHIVES_DIR}" \
                download "$pkg" >/dev/null 2>"$tmp_err"
        ); then
            rm -f "$tmp_err"
            return 0
        fi
        if grep -qiE 'Could not get lock|Unable to lock|held by process|Could not open lock' "$tmp_err" \
            && (( attempt < APT_DOWNLOAD_RETRIES )); then
            warn "apt lock while downloading $pkg; retrying in ${delay}s (${attempt}/${APT_DOWNLOAD_RETRIES})"
            sleep "$delay"
            delay=$((delay * 2))
            continue
        fi
        {
            printf '=== %s ===\n' "$pkg"
            cat "$tmp_err"
        } >> "$OUT_DIR/meta/apt-download-errors.log"
        rm -f "$tmp_err"
        return 1
    done
}

[[ $EUID -eq 0 ]] && die "Do NOT run as root. Script will invoke sudo as needed."
command -v sudo  >/dev/null || die "sudo is required."
command -v curl  >/dev/null || die "curl is required. Run: sudo apt-get install curl"

sudo -v
_sudo_keepalive() { while true; do sudo -n true 2>/dev/null || true; sleep 60; done; }
_sudo_keepalive & _sudo_keepalive_pid=$!
trap 'kill "$_sudo_keepalive_pid" 2>/dev/null || true' EXIT

. /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Need Ubuntu gather host (found: ${ID:-?})."
if [[ "${VERSION_ID:-}" != "$TARGET_OS_VERSION" ]]; then
    warn "Gather host is Ubuntu $VERSION_ID but bundle targets $TARGET_OS_VERSION."
    warn "Press Ctrl-C now if wrong; continuing in 10s..."; sleep 10
fi

# ============================================================================
# Local prerequisites
# ============================================================================
step "Local prerequisites"
_need=()
command -v dpkg              >/dev/null || _need+=( dpkg )
command -v dpkg-scanpackages >/dev/null || _need+=( dpkg-dev )
command -v apt-rdepends      >/dev/null || _need+=( apt-rdepends )
command -v gpg               >/dev/null || _need+=( gnupg )
if (( ${#_need[@]} > 0 )); then
    log "Installing local prerequisites: ${_need[*]}"
    _apt_get_wait update -qq
    _apt_get_wait install -y "${_need[@]}" || true
fi
command -v dpkg-scanpackages >/dev/null || die "dpkg-scanpackages unavailable (dpkg-dev)."
command -v apt-rdepends      >/dev/null || die "apt-rdepends unavailable."

# ============================================================================
# Register NVIDIA CUDA apt repo on this gather host (idempotent)
# ============================================================================
step "Register NVIDIA CUDA apt repo"
if [[ ! -f "$NVIDIA_REPO_LIST" ]]; then
    log "Installing cuda-keyring from $NVIDIA_KEYRING_DEB_URL"
    _tmp=$(mktemp --suffix=.deb)
    curl -fL --retry 3 -o "$_tmp" "$NVIDIA_KEYRING_DEB_URL" \
        || die "Failed to download cuda-keyring."
    _apt_get_wait install -y "$_tmp"
    rm -f "$_tmp"
else
    log "NVIDIA repo already registered at $NVIDIA_REPO_LIST"
fi
log "Refreshing apt indexes (NVIDIA repo included)"
_apt_get_wait update \
    -o Acquire::http::Timeout=60 \
    -o Acquire::https::Timeout=60 \
    -o Acquire::Retries=2 \
    || warn "apt-get update had errors; continuing."

# ============================================================================
# Resolve concrete versions
# ============================================================================
step "Resolve concrete versions"

_apt_candidate() {
    # NOTE: awk must drain stdin (no early `exit`) — apt-cache policy prints
    # many lines, and an early awk exit triggers SIGPIPE on apt-cache, which
    # under `set -euo pipefail` aborts the script silently inside $(...).
    apt-cache policy "$1" 2>/dev/null \
        | awk '/Candidate:/ && !s {print $2; s=1}'
}

_apt_version_for_upstream() {
    local pkg="$1" upstream="$2"
    apt-cache madison "$pkg" 2>/dev/null \
        | awk -F'|' -v upstream="$upstream" '
            {
                gsub(/^ +| +$/, "", $2)
                suffix = substr($2, length(upstream) + 1, 1)
                if (!s && index($2, upstream) == 1 && (suffix == "" || suffix == "-" || suffix == "+")) { print $2; s=1 }
            }'
}

_apt_versions_csv() {
    apt-cache madison "$1" 2>/dev/null \
        | awk -F'|' '{ gsub(/^ +| +$/, "", $2); if ($2 && !seen[$2]++) print $2 }' \
        | paste -sd ', ' -
}

CUDA_DRIVERS_PKG="cuda-drivers-${DRIVER_BRANCH}"
CUDA_COMPAT_PKG="cuda-compat-${CUDA_MAJOR}-${CUDA_MINOR}"

# Driver version pin. If user did not pin, take the cuda-drivers branch version
# resolves to today — that's the metapackage that ties driver+FM together.
if [[ -z "$PIN_DRIVER_VER" ]]; then
    _cand=$(_apt_candidate "$CUDA_DRIVERS_PKG")
    [[ -n "$_cand" && "$_cand" != "(none)" ]] \
        || die "$CUDA_DRIVERS_PKG has no candidate. Check apt sources."
    # The version string is "580.159.03-0ubuntu0.24.04.1" etc.
    # Strip the apt revision to get the upstream driver version.
    PIN_DRIVER_VER="${_cand%%-*}"
fi

_resolve_pkg_for_upstream() {
    local label="$1" upstream="$2"; shift 2
    local pkg ver tried=()
    for pkg in "$@"; do
        [[ -n "$pkg" ]] || continue
        tried+=( "$pkg" )
        ver=$(_apt_version_for_upstream "$pkg" "$upstream")
        if [[ -n "$ver" ]]; then
            printf '%s\t%s\n' "$pkg" "$ver"
            return 0
        fi
    done
    die "$label has no version for $upstream. Tried: ${tried[*]}"
}

IFS=$'\t' read -r DRIVER_PKG DRIVER_PKG_VER < <(
    _resolve_pkg_for_upstream "driver package" "$PIN_DRIVER_VER" \
        "nvidia-driver-${DRIVER_BRANCH}-${DRIVER_FLAVOR}" \
        "nvidia-driver-${DRIVER_BRANCH}-open" \
        "nvidia-driver-${DRIVER_BRANCH}"
)
CUDA_DRIVERS_VER=$(_apt_version_for_upstream "$CUDA_DRIVERS_PKG" "$PIN_DRIVER_VER")
FM_META_PKG=""
FM_META_VER=""
if _fm_meta=$(_resolve_pkg_for_upstream "fabricmanager meta package" "$PIN_DRIVER_VER" \
        "cuda-drivers-fabricmanager-${DRIVER_BRANCH}" \
        "cuda-drivers-fabricmanager" 2>/dev/null); then
    IFS=$'\t' read -r FM_META_PKG FM_META_VER <<<"$_fm_meta"
else
    log "Fabricmanager meta package is virtual for R${DRIVER_BRANCH}; bundling actual package instead."
fi
IFS=$'\t' read -r FM_PKG FM_VER < <(
    _resolve_pkg_for_upstream "fabricmanager package" "$PIN_DRIVER_VER" \
        "nvidia-fabricmanager-${DRIVER_BRANCH}" \
        "nvidia-fabricmanager"
)
IFS=$'\t' read -r NSCQ_PKG NSCQ_VER < <(
    _resolve_pkg_for_upstream "NSCQ package" "$PIN_DRIVER_VER" \
        "libnvidia-nscq-${DRIVER_BRANCH}" \
        "libnvidia-nscq"
)
CUDA_COMPAT_VER=$(_apt_version_for_upstream "$CUDA_COMPAT_PKG" "$PIN_DRIVER_VER")

[[ -n "$DRIVER_PKG_VER" ]] \
    || die "$DRIVER_PKG has no version for $PIN_DRIVER_VER. Available: $(_apt_versions_csv "$DRIVER_PKG")"
[[ -n "$CUDA_DRIVERS_VER" ]] \
    || die "$CUDA_DRIVERS_PKG has no version for $PIN_DRIVER_VER. Available: $(_apt_versions_csv "$CUDA_DRIVERS_PKG")"
[[ -n "$FM_VER" ]] \
    || die "$FM_PKG has no version for $PIN_DRIVER_VER. Available: $(_apt_versions_csv "$FM_PKG")"
[[ -n "$NSCQ_VER" ]] \
    || die "$NSCQ_PKG has no version for $PIN_DRIVER_VER. Available: $(_apt_versions_csv "$NSCQ_PKG")"
[[ -n "$CUDA_COMPAT_VER" ]] \
    || die "$CUDA_COMPAT_PKG has no version for $PIN_DRIVER_VER. Available: $(_apt_versions_csv "$CUDA_COMPAT_PKG")"

log "Driver branch  : R${DRIVER_BRANCH}"
log "Driver version : ${PIN_DRIVER_VER}"
log "Driver package : ${DRIVER_PKG}=${DRIVER_PKG_VER}"
log "Fabric Manager : ${FM_PKG}=${FM_VER}"
[[ -n "$FM_META_PKG" ]] && log "FM meta        : ${FM_META_PKG}=${FM_META_VER}"

# Pick NCCL version that matches +cuda13.0 suffix.
NCCL_VER=""
if [[ "$SKIP_NCCL" != "1" ]]; then
    # Same SIGPIPE/pipefail caveat as _apt_candidate — drain stdin instead of
    # exiting early on first match.
    NCCL_VER=$(apt-cache madison libnccl2 2>/dev/null \
        | awk -F'|' -v sfx="+${NCCL_CUDA_SUFFIX}" '
            !s { gsub(/^ +| +$/, "", $2); if (index($2, sfx) > 0) { print $2; s=1 } }')
    if [[ -z "$NCCL_VER" ]]; then
        warn "No libnccl2 with +${NCCL_CUDA_SUFFIX} found. Set SKIP_NCCL=1 to ignore."
        die  "Refusing to bundle a mismatched NCCL — would break CUDA ${CUDA_MAJOR}.${CUDA_MINOR}."
    fi
    log "NCCL version   : ${NCCL_VER}  (matches +${NCCL_CUDA_SUFFIX})"
else
    log "NCCL           : SKIPPED (SKIP_NCCL=1)"
fi

# ============================================================================
# Output tree
# ============================================================================
step "Output tree"
log "Output directory: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{debs,meta}

cat > "$OUT_DIR/meta/target.env" <<EOF
BUNDLE_VARIANT=nvidia-stack
BUNDLE_OS_ID=$ID
BUNDLE_OS_VERSION=$VERSION_ID
BUNDLE_TARGET_OS=$TARGET_OS_VERSION
BUNDLE_ARCH=$TARGET_ARCH
BUNDLE_DRIVER_BRANCH=R${DRIVER_BRANCH}
BUNDLE_DRIVER_VERSION=${PIN_DRIVER_VER}
BUNDLE_DRIVER_FLAVOR=${DRIVER_FLAVOR}
BUNDLE_DRIVER_PACKAGE=${DRIVER_PKG}
BUNDLE_DRIVER_PACKAGE_VERSION=${DRIVER_PKG_VER}
BUNDLE_CUDA_DRIVERS_PACKAGE=${CUDA_DRIVERS_PKG}
BUNDLE_CUDA_DRIVERS_VERSION=${CUDA_DRIVERS_VER}
BUNDLE_CUDA_DRIVERS_FABRICMANAGER_PACKAGE=${FM_META_PKG:-skipped}
BUNDLE_CUDA_DRIVERS_FABRICMANAGER_VERSION=${FM_META_VER:-skipped}
BUNDLE_FABRICMANAGER_PACKAGE=${FM_PKG}
BUNDLE_FABRICMANAGER_VERSION=${FM_VER}
BUNDLE_NSCQ_PACKAGE=${NSCQ_PKG}
BUNDLE_NSCQ_VERSION=${NSCQ_VER}
BUNDLE_CUDA_COMPAT_PACKAGE=${CUDA_COMPAT_PKG}
BUNDLE_CUDA_COMPAT_VERSION=${CUDA_COMPAT_VER}
BUNDLE_CUDA=${CUDA_MAJOR}.${CUDA_MINOR}
BUNDLE_NCCL_VERSION=${NCCL_VER:-skipped}
BUNDLE_NCCL_CUDA_SUFFIX=${NCCL_CUDA_SUFFIX}
BUNDLE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUNDLE_GATHER_HOST=$(hostname -f 2>/dev/null || hostname)
BUNDLE_GATHER_USER=$(id -un)
EOF

# ============================================================================
# Package list
# ============================================================================
step "Package list"

NV_PKGS=(
    # Driver — open kmod (mandatory for Blackwell). Pulls in
    # nvidia-persistenced + nvidia-peermem.ko transitively, so no separate
    # "persistence" or "peermem" package is needed (and none exists in the
    # NVIDIA apt repo — install-nvidia.sh writes /etc/modules-load.d/ for
    # nvidia-peermem and enables nvidia-persistenced.service).
    "${DRIVER_PKG}=${DRIVER_PKG_VER}"
    "nvidia-driver-pinning-${DRIVER_BRANCH}"     # apt unattended-upgrade guard
    "nvidia-modprobe"

    # Driver-aligned NVSwitch / NVLink5 stack — version locked to driver via
    # the cuda-drivers-* metapackages that NVIDIA ships.
    # NOTE: libnvsdm has no -580 variant (NVIDIA shipped 565/570/575 then
    # switched to unversioned starting 595). It's for QM/IB switch telemetry,
    # not NVSwitch fabric — fabricmanager + libnvidia-nscq own the NVSwitch
    # side. Drop libnvsdm here; add unversioned `libnvsdm` later if needed.
    "${CUDA_DRIVERS_PKG}=${CUDA_DRIVERS_VER}"
    "${FM_PKG}=${FM_VER}"
    "${NSCQ_PKG}=${NSCQ_VER}"
    "nvlsm"                                       # 4th-gen NVSwitch subnet mgr

    # Monitoring
    "datacenter-gpu-manager-4-cuda${CUDA_MAJOR}"

    # CUDA toolkit (NOT cuda / cuda-13-0 metapkg — those pull driver again)
    "cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"
    "cuda-cudart-${CUDA_MAJOR}-${CUDA_MINOR}"
    "cuda-cudart-dev-${CUDA_MAJOR}-${CUDA_MINOR}"
    "${CUDA_COMPAT_PKG}=${CUDA_COMPAT_VER}"       # forward-compat, optional but cheap
)

if [[ -n "$FM_META_PKG" ]]; then
    NV_PKGS+=( "${FM_META_PKG}=${FM_META_VER}" )
fi

if [[ "$SKIP_NCCL" != "1" ]]; then
    NV_PKGS+=( "libnccl2=${NCCL_VER}" "libnccl-dev=${NCCL_VER}" )
fi

printf '%s\n' "${NV_PKGS[@]}" > "$OUT_DIR/meta/nvidia-packages.txt"
log "Top-level packages: ${#NV_PKGS[@]}"

# ============================================================================
# Dependency closure + download
# ============================================================================
step "Dependency closure"

# apt-rdepends can choke on virtual deps; mirror gather-all.sh's approach.
_apt_has_candidate() {
    apt-cache policy "$1" 2>/dev/null \
        | awk '/Candidate:/ && $2 != "(none)" {found=1} END {exit !found}'
}
_apt_virtual_providers() {
    apt-cache showpkg "$1" 2>/dev/null \
        | awk '/^Reverse Provides:/ {p=1; next} p && NF {print $1}' \
        | sort -u
}
_apt_resolve() {
    local pkg="$1" prov
    _apt_has_candidate "$pkg" && { echo "$pkg"; return 0; }
    while IFS= read -r prov; do
        [[ -n "$prov" ]] || continue
        _apt_has_candidate "$prov" && { echo "$prov"; return 0; }
    done < <(_apt_virtual_providers "$pkg")
    return 1
}

# Strip the version pin "=ver" from package names for apt-rdepends.
_pkg_names=()
for p in "${NV_PKGS[@]}"; do _pkg_names+=( "${p%%=*}" ); done

log "Running apt-rdepends on ${#_pkg_names[@]} top-level packages"
apt-rdepends "${_pkg_names[@]}" 2>/dev/null \
    | awk '/^[A-Za-z0-9][A-Za-z0-9+.-]*(:[A-Za-z0-9]+)?$/ {print $1}' \
    | sed 's/:.*//' \
    | sort -u > "$OUT_DIR/meta/closure.txt"

log "Closure: $(wc -l < "$OUT_DIR/meta/closure.txt") packages"

: > "$OUT_DIR/meta/closure-download.txt"
: > "$OUT_DIR/meta/closure-virtual.txt"
: > "$OUT_DIR/meta/closure-unresolved.txt"
: > "$OUT_DIR/meta/closure-failed.txt"

while IFS= read -r pkg; do
    [[ -n "$pkg" ]] || continue
    if resolved=$(_apt_resolve "$pkg"); then
        printf '%s\n' "$resolved" >> "$OUT_DIR/meta/closure-download.txt"
        [[ "$resolved" != "$pkg" ]] && \
            printf '%s -> %s\n' "$pkg" "$resolved" >> "$OUT_DIR/meta/closure-virtual.txt"
    else
        printf '%s\n' "$pkg" >> "$OUT_DIR/meta/closure-unresolved.txt"
    fi
done < "$OUT_DIR/meta/closure.txt"

sort -u "$OUT_DIR/meta/closure-download.txt" -o "$OUT_DIR/meta/closure-download.txt"

step "Download (apt-get download)"

# Use --download-only first to get version-pinned packages into apt cache,
# then explicit apt-get download for the closure (which won't honor pins).
_apt_get_stage clean
log "Pre-fetching pinned versions via apt-get install --download-only"
_apt_get_stage install -y --download-only --reinstall --allow-downgrades "${NV_PKGS[@]}" \
    || die "apt-get --download-only failed for pinned packages."

log "Downloading ${#NV_PKGS[@]} top-level (pinned) packages into staging"
shopt -s nullglob
debs=("$APT_STAGE_ARCHIVES_DIR"/*.deb)
if (( ${#debs[@]} > 0 )); then
    sudo cp "${debs[@]}" "$OUT_DIR/debs/"
fi
shopt -u nullglob

TXN_VERSIONS_FILE="$OUT_DIR/meta/install-transaction-versions.tsv"
: > "$TXN_VERSIONS_FILE"
for deb in "$OUT_DIR"/debs/*.deb; do
    [[ -f "$deb" ]] || continue
    pkg=$(dpkg-deb -f "$deb" Package 2>/dev/null || true)
    ver=$(dpkg-deb -f "$deb" Version 2>/dev/null || true)
    [[ -n "$pkg" && -n "$ver" ]] && printf '%s\t%s\n' "$pkg" "$ver" >> "$TXN_VERSIONS_FILE"
done
sort -u "$TXN_VERSIONS_FILE" -o "$TXN_VERSIONS_FILE"
log "Pinned install transaction packages: $(wc -l < "$TXN_VERSIONS_FILE")"

log "Downloading closure (transitive deps)"
while IFS= read -r pkg; do
    [[ -n "$pkg" ]] || continue
    _apt_download_stage "$pkg" \
        || printf '%s\n' "$pkg" >> "$OUT_DIR/meta/closure-failed.txt"
done < "$OUT_DIR/meta/closure-download.txt"

sudo chown -R "$(id -u):$(id -g)" "$OUT_DIR/debs"
sudo rm -rf "$APT_STAGE_CACHE_DIR"

# ============================================================================
# Sanity: every top-level package must be present in debs/
# ============================================================================
step "Sanity check"
_missing=()
for p in "${NV_PKGS[@]}"; do
    name="${p%%=*}"
    if ! compgen -G "$OUT_DIR/debs/${name}_*" >/dev/null; then
        _missing+=( "$name" )
    fi
done
if (( ${#_missing[@]} > 0 )); then
    warn "Top-level packages missing from debs/: ${_missing[*]}"
    warn "Check meta/closure-failed.txt and meta/closure-unresolved.txt"
    die "Refusing to package incomplete NVIDIA bundle."
fi
log "All ${#NV_PKGS[@]} top-level packages present in debs/."

# ============================================================================
# Dedupe (prefer the versions chosen by the pinned apt transaction)
# ============================================================================
step "Dedupe debs"
shopt -s nullglob
declare -A _preferred_ver _keep_path _newest_ver _newest_path
while IFS=$'\t' read -r pkg ver; do
    [[ -n "$pkg" && -n "$ver" ]] || continue
    _preferred_ver[$pkg]="$ver"
done < "$TXN_VERSIONS_FILE"
for f in "$OUT_DIR"/debs/*.deb; do
    pkg=$(dpkg-deb -f "$f" Package 2>/dev/null || true)
    ver=$(dpkg-deb -f "$f" Version 2>/dev/null || true)
    if [[ -z "$pkg" || -z "$ver" ]]; then
        warn "Removing unreadable deb metadata: $f"
        rm -f "$f"
        continue
    fi
    preferred="${_preferred_ver[$pkg]:-}"
    if [[ -n "$preferred" ]]; then
        if [[ "$ver" != "$preferred" ]]; then
            rm -f "$f"
            continue
        fi
        if [[ -z "${_keep_path[$pkg]:-}" ]]; then
            _keep_path[$pkg]="$f"
        else
            rm -f "$f"
        fi
        continue
    fi
    prev_ver="${_newest_ver[$pkg]:-}"
    if [[ -z "$prev_ver" ]]; then
        _newest_ver[$pkg]="$ver"; _newest_path[$pkg]="$f"
    elif dpkg --compare-versions "$ver" gt "$prev_ver" 2>/dev/null; then
        rm -f "${_newest_path[$pkg]}"
        _newest_ver[$pkg]="$ver"; _newest_path[$pkg]="$f"
    else
        rm -f "$f"
    fi
done
unset _preferred_ver _keep_path _newest_ver _newest_path
shopt -u nullglob

step "Pinned package sanity"
_deb_has_pkg_version() {
    local want_pkg="$1" want_ver="${2:-}" deb pkg ver
    for deb in "$OUT_DIR"/debs/*.deb; do
        [[ -f "$deb" ]] || continue
        pkg=$(dpkg-deb -f "$deb" Package 2>/dev/null || true)
        [[ "$pkg" == "$want_pkg" ]] || continue
        if [[ -z "$want_ver" ]]; then
            return 0
        fi
        ver=$(dpkg-deb -f "$deb" Version 2>/dev/null || true)
        [[ "$ver" == "$want_ver" ]] && return 0
    done
    return 1
}

_missing=()
for p in "${NV_PKGS[@]}"; do
    name="${p%%=*}"
    ver=""
    [[ "$p" == *=* ]] && ver="${p#*=}"
    if ! _deb_has_pkg_version "$name" "$ver"; then
        if [[ -n "$ver" ]]; then
            _missing+=( "$name=$ver" )
        else
            _missing+=( "$name" )
        fi
    fi
done
if (( ${#_missing[@]} > 0 )); then
    warn "Top-level packages missing after dedupe: ${_missing[*]}"
    die "Refusing to package incomplete or internally inconsistent NVIDIA bundle."
fi
log "All ${#NV_PKGS[@]} top-level packages and pinned versions survived dedupe."

# ============================================================================
# Local apt repo metadata
# ============================================================================
step "Local apt repo metadata"
bundle_debs=( "$OUT_DIR"/debs/*.deb )
(( ${#bundle_debs[@]} > 0 )) || die "No .deb files in $OUT_DIR/debs."
( cd "$OUT_DIR/debs" && dpkg-scanpackages . /dev/null > Packages )
gzip -9c "$OUT_DIR/debs/Packages" > "$OUT_DIR/debs/Packages.gz"
log "Repo: $(ls "$OUT_DIR/debs" | wc -l) files ($(du -sh "$OUT_DIR/debs" | cut -f1))"

# ============================================================================
# Bundle + checksums
# ============================================================================
step "Bundle + checksums"

# Copy helpers into the bundle so install-nvidia.sh has its siblings.
for helper in install-nvidia.sh pre-install-nvidia.sh test-nvidia.sh; do
    if [[ -f "$SCRIPT_DIR/$helper" ]]; then
        cp "$SCRIPT_DIR/$helper" "$OUT_DIR/$helper"
        chmod +x "$OUT_DIR/$helper"
        log "Bundled helper: $helper"
    else
        warn "Helper not found at $SCRIPT_DIR/$helper — bundle will lack this file."
    fi
done

log "Generating SHA256 manifest"
(
    cd "$OUT_DIR"
    find install-nvidia.sh pre-install-nvidia.sh test-nvidia.sh debs meta \
        -type f \
        ! -path 'meta/SHA256SUMS' \
        -print0 2>/dev/null \
        | sort -z \
        | xargs -0 sha256sum > meta/SHA256SUMS
)

BUNDLE_PARENT="$(dirname "$OUT_DIR")"
BUNDLE_BIN="$BUNDLE_PARENT/nvidia-airgap-bundle-ubuntu${TARGET_OS_VERSION}.bin"
log "Packing bundle -> $BUNDLE_BIN"
rm -f "$BUNDLE_BIN" "${BUNDLE_BIN}.sha256"
tar -czf "$BUNDLE_BIN" -C "$BUNDLE_PARENT" "$(basename "$OUT_DIR")"

log "Generating bundle SHA256 sidecar"
( cd "$BUNDLE_PARENT" && sha256sum "$(basename "$BUNDLE_BIN")" > "$(basename "$BUNDLE_BIN").sha256" )

for helper in install-nvidia.sh pre-install-nvidia.sh test-nvidia.sh; do
    [[ -f "$SCRIPT_DIR/$helper" ]] || continue
    cp "$SCRIPT_DIR/$helper" "$BUNDLE_PARENT/$helper"
    chmod +x "$BUNDLE_PARENT/$helper"
done

log "Done."
printf '\n'
printf '  Bundle    : %s (%s)\n' "$BUNDLE_BIN" "$(du -sh "$BUNDLE_BIN" | cut -f1)"
printf '  SHA256    : %s\n' "${BUNDLE_BIN}.sha256"
printf '  Pre-flight: %s\n' "$BUNDLE_PARENT/pre-install-nvidia.sh"
printf '  Installer : %s\n' "$BUNDLE_PARENT/install-nvidia.sh"
printf '  Verifier  : %s\n' "$BUNDLE_PARENT/test-nvidia.sh"
printf '  Staging   : %s\n' "$OUT_DIR"
printf '\n'
printf 'Transfer to airgapped server:\n'
printf '  scp "%s" "%s" \\\n         "%s" "%s" "%s" user@SERVER:~\n' \
    "$BUNDLE_BIN" "${BUNDLE_BIN}.sha256" \
    "$BUNDLE_PARENT/pre-install-nvidia.sh" \
    "$BUNDLE_PARENT/install-nvidia.sh" \
    "$BUNDLE_PARENT/test-nvidia.sh"
printf '  ssh user@SERVER\n'
printf '  sudo bash pre-install-nvidia.sh   # readiness gate\n'
printf '  sudo bash install-nvidia.sh       # installs driver+FM+NVLSM+CUDA+NCCL\n'
printf '  sudo reboot                       # required: load nvidia.ko + start FM\n'
printf '  sudo bash test-nvidia.sh          # verify NVSwitch fabric Completed\n'
printf '\n'
printf 'Driver baseline: R%s / %s\n' "$DRIVER_BRANCH" "$PIN_DRIVER_VER"
if [[ -n "$NCCL_VER" ]]; then
    printf 'NCCL          : %s\n' "$NCCL_VER"
fi
printf '\n'
printf 'Next: run gather-all.sh for the userland bundle (separate transfer).\n'
