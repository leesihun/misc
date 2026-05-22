#!/usr/bin/env bash
# ============================================================================
# gather-all.sh  (userland variant)
#
#   Run on an internet-connected WSL Ubuntu 24.04 machine. Builds the
#   USERLAND airgap bundle for an Ubuntu 24.04 server with 8x B300 GPUs
#   where install-nvidia.sh (from gather-nvidia.sh's bundle) has already
#   put R580 LTS driver + CUDA 13.0 + FM + NVLSM in place.
#
#   This is the second of two bundles:
#     1. gather-nvidia.sh ??nvidia-airgap-bundle  (driver+CUDA+FM+NCCL)
#     2. gather-all.sh    ??all-airgap-bundle     (THIS ??userland)
#
#   What this bundles:
#     - Userland apt packages (~25 + GUI runtime libs + xfce4/xrdp): build
#       tools, htop/nvtop/tmux, network utils, python3.12-venv/dev, etc.
#     - VS Code, Chrome, Firefox, Node.js LTS, Bun, Opencode
#     - Python wheels (cu130): inference (vLLM), training (PyG + Huni projects),
#       jupyter (data science), llama.cpp utility scripts
#     - llama.cpp source (built on target against the NVIDIA bundle's nvcc 13.0)
#
#   What this does NOT bundle (lives in the nvidia-airgap-bundle):
#     - NVIDIA driver / open kernel modules
#     - CUDA MINIMAL toolkit set: cuda-nvcc-13-0, cuda-cudart-13-0,
#       cuda-cudart-dev-13-0, cuda-cccl-13-0, libcublas-13-0, libcublas-dev-13-0,
#       libnvjitlink-13-0, cuda-compat-13-0  (we deliberately AVOID the
#       cuda-toolkit-13-0 metapkg — see gather-nvidia.sh for the rationale)
#     - nvidia-fabricmanager / nvlsm / libnvidia-nscq
#     - libnccl2 + libnccl-dev (gathered, but SKIP_NCCL=1 by default at
#       install time — system NCCL is opt-in)
#     - DCGM
#     - K3s, Helm, kubectl, container images (out of scope)
#
#   Output:
#     - ~/GPU_server_downloads/                          (staging tree)
#     - ~/all-airgap-bundle-ubuntu24.04.bin               (single bundle file)
#     - ~/all-airgap-bundle-ubuntu24.04.bin.sha256        (sidecar)
#     - ~/install-all.sh, ~/pre-install-check.sh, ~/test-all.sh  (helpers)
#
#   Usage:
#     bash gather-all.sh                  # default settings
#     INSTALL_DESKTOP=0 bash gather-all.sh # headless: no xfce4/xrdp
#     INCLUDE_JUPYTER=0 bash gather-all.sh # skip JupyterLab wheels
#
#   Bundle marker:
#     meta/target.env carries BUNDLE_VARIANT=prepped so install-all.sh refuses
#     to install a bare-metal bundle on a prepped server (and vice versa).
# ============================================================================
set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

OUT_DIR="${OUT_DIR:-$HOME/GPU_server_downloads}"
APT_LOCK_TIMEOUT="${APT_LOCK_TIMEOUT:-900}"
APT_DOWNLOAD_RETRIES="${APT_DOWNLOAD_RETRIES:-5}"
APT_STAGE_CACHE_DIR="$OUT_DIR/.apt-cache"
APT_STAGE_ARCHIVES_DIR="$APT_STAGE_CACHE_DIR/archives"

# Python
PYTHON_VER="${PYTHON_VER:-3.12}"
PYTHON_BIN="${PYTHON_BIN:-python${PYTHON_VER}}"

# PyTorch + CUDA 13.0 (cu130 wheels). Index confirmed: torch 2.11.0 for py3.12.
# PyG also publishes cu130 wheels for torch 2.11.0:
#   https://data.pyg.org/whl/torch-2.11.0+cu130.html
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu130}"
TORCH_VER_INFERENCE="${TORCH_VER_INFERENCE:-2.11.0}"
TORCH_VER_TRAINING="${TORCH_VER_TRAINING:-2.11.0}"

# App URLs
VSCODE_URL="${VSCODE_URL:-https://update.code.visualstudio.com/latest/linux-deb-x64/stable}"
CHROME_URL="${CHROME_URL:-https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb}"
FIREFOX_VER="${FIREFOX_VER:-latest}"
FIREFOX_LANG="${FIREFOX_LANG:-en-US}"

# Opencode (https://github.com/sst/opencode) ??self-contained binary
OPENCODE_VER="${OPENCODE_VER:-latest}"

# Node.js LTS major
NODE_LTS_MAJOR="${NODE_LTS_MAJOR:-22}"

# Bun
BUN_VER="${BUN_VER:-latest}"

# vLLM ??empty means "latest from PyPI". The PyPI default occasionally pulls
# cu129 wheels; the installer pins --torch-backend=cu130 at install time.
VLLM_VER="${VLLM_VER:-}"

# llama.cpp source (built on target against vendor's nvcc)
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"

# Desktop environment: xfce4 + xrdp (the user confirmed they want this).
# Set INSTALL_DESKTOP=0 for SSH-only headless install.
INSTALL_DESKTOP="${INSTALL_DESKTOP:-1}"

# Jupyter + data science wheels
INCLUDE_JUPYTER="${INCLUDE_JUPYTER:-1}"

# Target metadata. The NVIDIA bundle (gather-nvidia.sh) installs R580 LTS;
# this string is stamped into meta/target.env for pre-install-check.sh to
# cross-reference. R580 LTS is supported until 2028-06.
DRIVER_BASELINE="${DRIVER_BASELINE:-R580.159.04}"   # informational

# Target OS ??gather host should match for .deb compatibility.
TARGET_OS_VERSION="${TARGET_OS_VERSION:-$(. /etc/os-release && echo "$VERSION_ID")}"

# ?? Script location ?????????????????????????????????????????????????????????
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ?? Huni root: works from Windows path on WSL (/mnt/c/...) or native Linux ??
_find_huni_dir() {
    local candidates=(
        "$SCRIPT_DIR/.."
        "$SCRIPT_DIR/../.."
        "/mnt/c/Users/Lee/Desktop/Huni"
        "/mnt/c/Users/${USER}/Desktop/Huni"
        "$HOME/Huni"
        "$HOME/Desktop/Huni"
    )
    for d in "${candidates[@]}"; do
        if [[ -d "$d/LLM_API_fast" || -d "$d/MeshGraphNets - variational" ]]; then
            realpath "$d" 2>/dev/null || echo "$d"
            return 0
        fi
    done
}
HUNI_DIR="${HUNI_DIR:-$(_find_huni_dir)}"

# Per-project requirements ??auto-detected.
LLMAPI_REQ="${LLMAPI_REQ:-}"
MGN_REQ="${MGN_REQ:-}"
SIMULGEN_REQ="${SIMULGEN_REQ:-}"
PEMTRON_REQ="${PEMTRON_REQ:-}"
PEMTRON_TRANSFER_REQ="${PEMTRON_TRANSFER_REQ:-}"
LLMAPI_FULL_REQ="${LLMAPI_FULL_REQ:-}"
ALL_PROJECTS_REQ="${ALL_PROJECTS_REQ:-}"

# ============================================================================
# Helpers
# ============================================================================

log()  { printf '\033[1;36m[gather]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gather:WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[gather:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;35m?먥븧 %s ?먥븧\033[0m\n' "$*"; }

[[ $EUID -eq 0 ]] && die "Do NOT run as root. Script will invoke sudo as needed."
command -v sudo >/dev/null || die "sudo is required."
command -v curl >/dev/null || die "curl is required. Run: sudo apt-get install curl"
command -v git  >/dev/null || die "git is required.  Run: sudo apt-get install git"

_apt_get_wait() {
    sudo apt-get -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" "$@"
}

_apt_stage_prepare() {
    mkdir -p "$APT_STAGE_ARCHIVES_DIR/partial" "$OUT_DIR/debs" "$OUT_DIR/meta"
}

# APT::Sandbox::User=root keeps apt running as root for the actual download
# instead of dropping to the `_apt` user. Required because our cache lives
# under $OUT_DIR (typically /home/$USER/...) which `_apt` cannot traverse
# — without this, every package emits:
#   W: Download is performed unsandboxed as root as file '...' couldn't be
#      accessed by user '_apt'. - pkgAcquire::Run (13: Permission denied)
# Safe here: gather host is a trusted operator-controlled machine.
_apt_get_stage() {
    _apt_stage_prepare
    sudo apt-get \
        -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" \
        -o "Dir::Cache=${APT_STAGE_CACHE_DIR}" \
        -o "Dir::Cache::archives=${APT_STAGE_ARCHIVES_DIR}" \
        -o "APT::Sandbox::User=root" \
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
                -o "APT::Sandbox::User=root" \
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

# Keep sudo cache warm for long downloads.
sudo -v
_sudo_keepalive() {
    while true; do
        sudo -n true 2>/dev/null || true
        sleep 60
    done
}
_sudo_keepalive &
_sudo_keepalive_pid=$!
trap 'kill "$_sudo_keepalive_pid" 2>/dev/null || true' EXIT

# Sanity: bundle is .deb-flavored, so gather host must match target.
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
    die "This script requires an Ubuntu gather host (found: ${ID:-?}). .deb compat depends on it."
fi
if [[ "${VERSION_ID:-}" != "$TARGET_OS_VERSION" ]]; then
    warn "Gather host is Ubuntu $VERSION_ID but bundle targets Ubuntu $TARGET_OS_VERSION."
    warn "The .deb files downloaded here may not satisfy dependencies on the target."
    warn "Press Ctrl-C now if this is wrong, or wait 10s to continue..."
    sleep 10
fi

# Auto-detect requirements files.
_try_req() {
    local var="$1"; shift
    if [[ -z "${!var}" ]]; then
        for f in "$@"; do
            if [[ -f "$f" ]]; then
                printf -v "$var" '%s' "$f"
                break
            fi
        done
    fi
    if [[ -n "${!var}" ]]; then
        log "$var -> ${!var}"
    else
        warn "$var not found (checked: $*)"
    fi
}

[[ -n "$HUNI_DIR" ]] && log "Huni project root: $HUNI_DIR" \
    || warn "Huni project root not found ??set HUNI_DIR=/path/to/Huni for project requirements."

_try_req LLMAPI_REQ \
    "${HUNI_DIR:-}/LLM_API_fast/requirements.txt" \
    "$SCRIPT_DIR/../LLM_API_fast/requirements.txt" \
    "$HOME/LLM_API_fast/requirements.txt"

_try_req MGN_REQ \
    "${HUNI_DIR:-}/MeshGraphNets - variational/requirements.txt" \
    "$SCRIPT_DIR/../MeshGraphNets - variational/requirements.txt"

_try_req SIMULGEN_REQ \
    "${HUNI_DIR:-}/SimulGenVAE/requirements.txt" \
    "$SCRIPT_DIR/../SimulGenVAE/requirements.txt"

_try_req PEMTRON_REQ \
    "${HUNI_DIR:-}/PEMTRON_warpage/requirements.txt" \
    "$SCRIPT_DIR/../PEMTRON_warpage/requirements.txt"

_try_req PEMTRON_TRANSFER_REQ \
    "${HUNI_DIR:-}/PEMTRON_warpage/data_autotransfer/requirements.txt" \
    "$SCRIPT_DIR/../PEMTRON_warpage/data_autotransfer/requirements.txt"

_try_req LLMAPI_FULL_REQ \
    "${HUNI_DIR:-}/temp/LLM_API/requirements.txt" \
    "$SCRIPT_DIR/../temp/LLM_API/requirements.txt"

_try_req ALL_PROJECTS_REQ \
    "$SCRIPT_DIR/requirements-all-projects.txt" \
    "${HUNI_DIR:-}/misc/requirements-all-projects.txt"

# ?? Ensure local prerequisites for running THIS script ??????????????????????
step "Local prerequisites"
_need=()
command -v pip3              >/dev/null || _need+=( python3-pip )
"$PYTHON_BIN" -c '' 2>/dev/null         || _need+=( "python${PYTHON_VER}" )
"$PYTHON_BIN" -m venv --help &>/dev/null || _need+=( "python${PYTHON_VER}-venv" python3-venv )
command -v dpkg              >/dev/null || _need+=( dpkg )
command -v dpkg-scanpackages >/dev/null || _need+=( dpkg-dev )
command -v apt-rdepends      >/dev/null || _need+=( apt-rdepends )

if (( ${#_need[@]} > 0 )); then
    log "Installing local prerequisites: ${_need[*]}"
    _apt_get_wait update -qq
    _apt_get_wait install -y "${_need[@]}" || true
fi

"$PYTHON_BIN" -m venv --help &>/dev/null \
    || die "$PYTHON_BIN -m venv unavailable. Install python${PYTHON_VER}-venv."
command -v dpkg-scanpackages >/dev/null || die "dpkg-scanpackages unavailable. Install dpkg-dev."
command -v apt-rdepends      >/dev/null || die "apt-rdepends unavailable. Install apt-rdepends."

# ?? Setup output tree ???????????????????????????????????????????????????????
log "Output directory: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{debs,apps,src,wheels/inference,wheels/training,wheels/llamacpp,wheels/jupyter,requirements,meta}

cat > "$OUT_DIR/meta/target.env" <<EOF
BUNDLE_VARIANT=prepped
BUNDLE_OS_ID=$ID
BUNDLE_OS_VERSION=$VERSION_ID
BUNDLE_TARGET_OS=$TARGET_OS_VERSION
BUNDLE_ARCH=$(dpkg --print-architecture)
BUNDLE_PYTHON=$PYTHON_VER
BUNDLE_TORCH_INFERENCE=$TORCH_VER_INFERENCE
BUNDLE_TORCH_TRAINING=$TORCH_VER_TRAINING
BUNDLE_TORCH_CUDA=$TORCH_CUDA_TAG
BUNDLE_VLLM_VER=${VLLM_VER:-latest}
BUNDLE_NODE_LTS_MAJOR=$NODE_LTS_MAJOR
BUNDLE_INSTALL_DESKTOP=$INSTALL_DESKTOP
BUNDLE_INCLUDE_JUPYTER=$INCLUDE_JUPYTER
BUNDLE_DRIVER_BASELINE=$DRIVER_BASELINE
BUNDLE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUNDLE_GATHER_HOST=$(hostname -f 2>/dev/null || hostname)
BUNDLE_GATHER_USER=$(id -un)
EOF
log "Host: $ID $VERSION_ID / $(dpkg --print-architecture)   (target: ubuntu $TARGET_OS_VERSION)"

# ============================================================================
# 1) APT PACKAGES ??userland only. NO driver / CUDA / fabricmanager.
# ============================================================================
step "APT packages (userland only)"

APT_PKGS=(
    # Python 3.12 ecosystem (target may have system python but not the dev/venv)
    "python${PYTHON_VER}"
    "python${PYTHON_VER}-venv"
    "python${PYTHON_VER}-dev"
    python3-pip

    # Build toolchain (llama.cpp, native wheel builds)
    build-essential
    cmake
    ninja-build
    pkg-config
    ccache
    git
    curl
    wget
    ca-certificates
    unzip
    xz-utils                  # extract Node.js .tar.xz

    # Native libs for ML wheels with C-extensions
    libopenblas-dev
    libopenblas0
    libgomp1
    libhdf5-dev               # h5py native build
    libssl-dev
    libffi-dev
    libcurl4-openssl-dev      # llama-server LLAMA_CURL=ON

    # Editors
    gedit
    vim
    nano

    # gedit pulls libenchant-2-2 (spell-check). libenchant-2-2's deps are
    # listed as alternatives (hunspell-en-us | hunspell-dictionary | ...),
    # and apt-rdepends does not always follow alternatives — without these
    # explicit entries the airgap repo is missing them and `apt install gedit`
    # fails with "you have held broken packages" on the target box.
    hunspell-en-us
    libhunspell-1.7-0
    libaspell15

    # Monitoring
    htop
    btop
    nvtop
    iotop

    # Terminal multiplexer
    tmux
    screen

    # Networking diagnostics
    net-tools
    iproute2
    dnsutils
    mtr-tiny
    traceroute

    # Utilities
    jq
    tree
    ncdu
    zip
    pigz
    zstd
    rsync

    # Daemon-restart helper (replaces reboots after lib upgrades on 24.04)
    needrestart

    # NUMA / topology ??useful on multi-socket B300 boxes for binding inference
    # workers to the NUMA node nearest each GPU.
    numactl
    hwloc-nox

    # GUI runtime libs required by VS Code and Chrome .debs.
    # Without these the .debs install but the apps silently fail to launch.
    libglib2.0-0t64
    libatk1.0-0t64
    libatk-bridge2.0-0t64
    libcairo2
    libcups2t64
    libdbus-1-3
    libdrm2
    libexpat1
    libfontconfig1
    fonts-liberation
    libgbm1
    libgtk-3-0t64
    libnspr4
    libnss3
    libpango-1.0-0
    libsecret-1-0
    libasound2t64             # 24.04 explicit t64 (libasound2 is unresolvable virtual)
    libx11-6
    libx11-xcb1
    libxcb1
    libxcomposite1
    libxcursor1
    libxdamage1
    libxext6
    libxfixes3
    libxi6
    libxkbcommon0
    libxkbfile1
    libxrandr2
    libxrender1
    libxss1
    libxtst6
    xdg-utils
)

if [[ "$INSTALL_DESKTOP" == "1" ]]; then
    APT_PKGS+=(
        # XFCE4 desktop
        xfce4
        xfce4-goodies
        xfce4-terminal
        xfce4-screenshooter
        xfce4-taskmanager
        xfce4-notifyd

        # Display manager
        lightdm
        lightdm-gtk-greeter

        # Remote desktop (xrdp on port 3389)
        xrdp
        xorgxrdp
        ssl-cert                # xrdp TLS certificate group

        # Polkit (mitigates xrdp issue #3248 ??open as of May 2026)
        policykit-1-gnome

        # X11 / session infrastructure
        dbus-x11
        x11-xserver-utils
        x11-utils
        xauth
        xinit
        xterm

        # File manager helpers (used by xfce4-goodies but be explicit)
        file-roller
        evince
        ristretto

        # Misc desktop utilities
        xclip
        dconf-editor

        # Fonts
        fonts-dejavu-core
        fonts-noto-core
        fonts-noto-color-emoji

        # Themes
        adwaita-icon-theme
        gnome-themes-extra

        # Archive
        p7zip-full

        # Shell QoL
        bash-completion
    )
    log "Desktop packages added: xfce4 + xrdp + policykit + fonts"
fi

_normalized_apt_pkgs=()
for pkg in "${APT_PKGS[@]}"; do
    case "$pkg" in
        libglib2.0-0)       pkg=libglib2.0-0t64 ;;
        libatk1.0-0)        pkg=libatk1.0-0t64 ;;
        libatk-bridge2.0-0) pkg=libatk-bridge2.0-0t64 ;;
        libcups2)           pkg=libcups2t64 ;;
        libgtk-3-0)         pkg=libgtk-3-0t64 ;;
        libasound2)         pkg=libasound2t64 ;;
    esac
    _normalized_apt_pkgs+=( "$pkg" )
done
APT_PKGS=( "${_normalized_apt_pkgs[@]}" )
unset _normalized_apt_pkgs

printf '%s\n' "${APT_PKGS[@]}" > "$OUT_DIR/meta/apt-packages.txt"

log "Refreshing apt indexes"
_apt_get_wait update \
    -o Acquire::http::Timeout=60 \
    -o Acquire::https::Timeout=60 \
    -o Acquire::Retries=2 \
    || warn "apt-get update had errors; continuing."

log "Cleaning local apt cache"
_apt_get_stage clean

log "Downloading ${#APT_PKGS[@]} apt package groups (+ transitive deps)"
_apt_get_stage install -y --download-only --reinstall "${APT_PKGS[@]}"

# ?? Dependency closure (resolve virtual deps that --download-only misses) ???
_apt_has_candidate() {
    apt-cache policy "$1" 2>/dev/null \
        | awk '/Candidate:/ && $2 != "(none)" {found=1} END {exit !found}'
}
_apt_virtual_providers() {
    apt-cache showpkg "$1" 2>/dev/null \
        | awk '/^Reverse Provides:/ {providers=1; next} providers && NF {print $1}' \
        | sort -u
}
_apt_preferred_provider() {
    case "$1" in
        awk)                            echo mawk ;;
        dbus-session-bus)               echo dbus-user-session ;;
        dbus-system-bus)                echo dbus ;;
        default-logind|logind|elogind)  echo libpam-systemd ;;
        libtinfo5)                      echo libtinfo6 ;;
        mime-support)                   echo media-types ;;
    esac
}
_apt_resolve_download_name() {
    local pkg="$1" provider
    _apt_has_candidate "$pkg" && { echo "$pkg"; return 0; }
    while IFS= read -r provider; do
        [[ -n "$provider" ]] || continue
        _apt_has_candidate "$provider" && { echo "$provider"; return 0; }
    done < <(_apt_preferred_provider "$pkg")
    while IFS= read -r provider; do
        [[ -n "$provider" ]] || continue
        _apt_has_candidate "$provider" && { echo "$provider"; return 0; }
    done < <(_apt_virtual_providers "$pkg")
    return 1
}

log "Computing apt dependency closure"
apt-rdepends "${APT_PKGS[@]}" 2>/dev/null \
    | awk '/^[A-Za-z0-9][A-Za-z0-9+.-]*(:[A-Za-z0-9]+)?$/ {print $1}' \
    | sed 's/:.*//' \
    | sort -u > "$OUT_DIR/meta/apt-closure.txt"

: > "$OUT_DIR/meta/apt-closure-download.txt"
: > "$OUT_DIR/meta/apt-closure-virtual.txt"
: > "$OUT_DIR/meta/apt-closure-unresolved.txt"
: > "$OUT_DIR/meta/apt-closure-download-failed.txt"

while IFS= read -r pkg; do
    [[ -n "$pkg" ]] || continue
    if resolved=$(_apt_resolve_download_name "$pkg"); then
        printf '%s\n' "$resolved" >> "$OUT_DIR/meta/apt-closure-download.txt"
        [[ "$resolved" != "$pkg" ]] && \
            printf '%s -> %s\n' "$pkg" "$resolved" >> "$OUT_DIR/meta/apt-closure-virtual.txt"
    else
        printf '%s\n' "$pkg" >> "$OUT_DIR/meta/apt-closure-unresolved.txt"
    fi
done < "$OUT_DIR/meta/apt-closure.txt"

sort -u "$OUT_DIR/meta/apt-closure-download.txt" -o "$OUT_DIR/meta/apt-closure-download.txt"

while IFS= read -r pkg; do
    [[ -n "$pkg" ]] || continue
    _apt_download_stage "$pkg" \
        || printf '%s\n' "$pkg" >> "$OUT_DIR/meta/apt-closure-download-failed.txt"
done < "$OUT_DIR/meta/apt-closure-download.txt"

# Copy what --download-only already cached
shopt -s nullglob
debs=("$APT_STAGE_ARCHIVES_DIR"/*.deb)
if (( ${#debs[@]} > 0 )); then
    sudo cp "${debs[@]}" "$OUT_DIR/debs/"
fi
sudo chown -R "$(id -u):$(id -g)" "$OUT_DIR/debs"
sudo rm -rf "$APT_STAGE_CACHE_DIR"

# ?? Critical-package presence check ?????????????????????????????????????????
_required_globs=(
    "python${PYTHON_VER}-venv_*"
    "python${PYTHON_VER}-dev_*"
    "build-essential_*"
    "cmake_*"
    "needrestart_*"
    "nvtop_*"
)
if [[ "$INSTALL_DESKTOP" == "1" ]]; then
    _required_globs+=(
        "xfce4_*"
        "xrdp_*"
        "policykit-1-gnome_*"
    )
fi

_missing=()
for g in "${_required_globs[@]}"; do
    if ! compgen -G "$OUT_DIR/debs/$g" >/dev/null; then
        _missing+=("$g")
    fi
done
if (( ${#_missing[@]} > 0 )); then
    warn "Critical packages missing from debs/ (${#_missing[@]}): ${_missing[*]}"
    warn "Cross-reference meta/apt-closure-download-failed.txt and re-run apt-get update."
    die "Refusing to package an incomplete bundle."
fi
shopt -u nullglob

# ?? Prune debs that conflict at install time ????????????????????????????????
_prune_globs=(
    "libcurl4-gnutls-dev_*.deb"        # conflicts with libcurl4-openssl-dev
    "pulseaudio_*.deb"                  # conflicts with pipewire-audio (xfce4 default)
    "pulseaudio-module-bluetooth_*.deb"
    "systemd-standalone-sysusers_*.deb" # conflicts with systemd-sysusers
    # GNOME flashback pulls libgnome-panel3 we don't bundle
    "gnome-flashback*.deb"
    # ispell/aspell/hunspell chain ??dictionaries-common post-install requires
    # ienglish-common which we don't bundle. Not needed on a server.
    "iamerican_*.deb" "ibritish_*.deb" "ispell_*.deb"
    "dictionaries-common_*.deb" "wamerican_*.deb" "wbritish_*.deb"
    "aspell_*.deb" "aspell-en_*.deb" "hunspell_*.deb" "hunspell-en*_*.deb"
    "libaspell*_*.deb" "libhunspell*_*.deb"
    # plymouth-label needs fonts-ubuntu which we don't bundle
    "plymouth-label_*.deb"
    # Any cuda-* / nvidia-* / libnvidia-* that snuck in ??we DO NOT touch the
    # vendor's NVIDIA stack from this bundle.
    "cuda-*.deb"
    "nvidia-*.deb"
    "libnvidia-*.deb"
    "libcudart*.deb" "libcublas*.deb" "libcudnn*.deb" "libcusparse*.deb"
    "libcurand*.deb" "libcufft*.deb" "libcusolver*.deb" "libnpp*.deb"
)
shopt -s nullglob
for pat in "${_prune_globs[@]}"; do
    for f in "$OUT_DIR"/debs/$pat; do
        [[ -e "$f" ]] || continue
        log "Pruning conflicting/redundant deb: $(basename "$f")"
        rm -f "$f"
    done
done

# ?? Dedupe: keep newest version of each package ?????????????????????????????
declare -A _newest_ver _newest_path
for f in "$OUT_DIR"/debs/*.deb; do
    base=$(basename "$f")
    pkg="${base%%_*}"
    rest="${base#${pkg}_}"
    ver="${rest%_*}"
    prev_ver="${_newest_ver[$pkg]:-}"
    if [[ -z "$prev_ver" ]]; then
        _newest_ver[$pkg]="$ver"
        _newest_path[$pkg]="$f"
    elif dpkg --compare-versions "$ver" gt "$prev_ver" 2>/dev/null; then
        rm -f "${_newest_path[$pkg]}"
        _newest_ver[$pkg]="$ver"
        _newest_path[$pkg]="$f"
    else
        rm -f "$f"
    fi
done
unset _newest_ver _newest_path
shopt -u nullglob

# ?? Local apt repo metadata ?????????????????????????????????????????????????
bundle_debs=( "$OUT_DIR"/debs/*.deb )
(( ${#bundle_debs[@]} > 0 )) || die "No .deb files were downloaded from apt."
log "Generating local apt repository metadata"
( cd "$OUT_DIR/debs" && dpkg-scanpackages . /dev/null > Packages )
gzip -9c "$OUT_DIR/debs/Packages" > "$OUT_DIR/debs/Packages.gz"
log "APT: $(ls "$OUT_DIR/debs" | wc -l) debs ($(du -sh "$OUT_DIR/debs" | cut -f1))"

# ============================================================================
# 2) GUI APPS: VS Code, Chrome, Firefox
# ============================================================================
step "GUI applications"

log "Downloading VS Code (.deb)"
curl -L --retry 3 --progress-bar -o "$OUT_DIR/apps/vscode.deb" "$VSCODE_URL"
log "VS Code: $(du -sh "$OUT_DIR/apps/vscode.deb" | cut -f1)"

log "Downloading Google Chrome (.deb)"
curl -L --retry 3 --progress-bar -o "$OUT_DIR/apps/chrome.deb" "$CHROME_URL"
log "Chrome: $(du -sh "$OUT_DIR/apps/chrome.deb" | cut -f1)"

# Firefox: tarball avoids the Snap dependency.
log "Resolving Firefox version"
if [[ "$FIREFOX_VER" == "latest" ]]; then
    FIREFOX_VER=$(curl -sI "https://download.mozilla.org/?product=firefox-latest-ssl&os=linux64&lang=${FIREFOX_LANG}" \
        | grep -i '^location:' \
        | grep -oP 'releases/\K[^/]+' \
        | head -1)
    [[ -n "$FIREFOX_VER" ]] || die "Could not resolve latest Firefox version."
fi
log "Downloading Firefox $FIREFOX_VER"
_ff_base_url="https://releases.mozilla.org/pub/firefox/releases/${FIREFOX_VER}/linux-x86_64/${FIREFOX_LANG}"
_ff_format=""
for _ff_ext in tar.xz tar.bz2; do
    rm -f "$OUT_DIR/apps/firefox.${_ff_ext}"
    if curl -fL --retry 3 --progress-bar -o "$OUT_DIR/apps/firefox.${_ff_ext}" \
            "${_ff_base_url}/firefox-${FIREFOX_VER}.${_ff_ext}"; then
        _ff_format="$_ff_ext"
        break
    fi
    rm -f "$OUT_DIR/apps/firefox.${_ff_ext}"
done
[[ -n "$_ff_format" ]] || die "Could not download Firefox $FIREFOX_VER (tried .tar.xz and .tar.bz2)."
echo "$FIREFOX_VER" > "$OUT_DIR/apps/firefox.version"
log "Firefox: $(du -sh "$OUT_DIR/apps/firefox.${_ff_format}" | cut -f1) (.${_ff_format})"

# ============================================================================
# 3) OPENCODE CLI
# ============================================================================
step "Opencode CLI"

log "Resolving Opencode release ($OPENCODE_VER)"
if [[ "$OPENCODE_VER" == "latest" ]]; then
    OPENCODE_TAG=$(curl -sL "https://api.github.com/repos/sst/opencode/releases/latest" \
        | grep '"tag_name"' | grep -oP '(?<="tag_name": ")[^"]+')
    if [[ -z "$OPENCODE_TAG" ]]; then
        warn "Could not resolve latest Opencode release (rate limited?). Skipping."
        OPENCODE_TAG=""
    fi
else
    OPENCODE_TAG="$OPENCODE_VER"
fi

if [[ -n "$OPENCODE_TAG" ]]; then
    log "Opencode tag: $OPENCODE_TAG"
    _oc_base="https://github.com/sst/opencode/releases/download/${OPENCODE_TAG}"
    _oc_downloaded=0
    for asset in "opencode-linux-x64.tar.gz" "opencode-linux-x64-musl.tar.gz"; do
        _url="${_oc_base}/${asset}"
        _tmp="$OUT_DIR/apps/_opencode_tmp.tar.gz"
        if curl -fsSL --retry 2 -o "$_tmp" "$_url" 2>/dev/null; then
            tar -xzf "$_tmp" -C "$OUT_DIR/apps/" 2>/dev/null || true
            if [[ -f "$OUT_DIR/apps/opencode" ]]; then
                chmod +x "$OUT_DIR/apps/opencode"
                rm -f "$_tmp"
                _oc_downloaded=1
                log "Opencode: $asset extracted ($(du -sh "$OUT_DIR/apps/opencode" | cut -f1))"
                break
            fi
            rm -f "$_tmp"
        fi
    done
    if (( ! _oc_downloaded )); then
        warn "Could not download Opencode. Manually place at: $OUT_DIR/apps/opencode"
        echo "PLACEHOLDER" > "$OUT_DIR/apps/opencode.MISSING"
    fi
    echo "$OPENCODE_TAG" > "$OUT_DIR/apps/opencode.version"
fi

# ============================================================================
# 4) NODE.JS LTS + npm (bundled with the tarball)
# ============================================================================
step "Node.js LTS"

log "Resolving Node.js v${NODE_LTS_MAJOR} LTS"
NODE_VER=$(curl -sL "https://nodejs.org/dist/latest-v${NODE_LTS_MAJOR}.x/" \
    | grep -oP "node-v\K[\d.]+(?=-linux-x64\.tar\.xz)" \
    | head -1)
[[ -n "$NODE_VER" ]] || die "Could not resolve Node.js v${NODE_LTS_MAJOR} LTS."
log "Downloading Node.js v${NODE_VER}"
curl -L --retry 3 --progress-bar \
    -o "$OUT_DIR/apps/nodejs.tar.xz" \
    "https://nodejs.org/dist/v${NODE_VER}/node-v${NODE_VER}-linux-x64.tar.xz"
echo "$NODE_VER" > "$OUT_DIR/apps/nodejs.version"
log "Node.js: $(du -sh "$OUT_DIR/apps/nodejs.tar.xz" | cut -f1)"

# ============================================================================
# 5) BUN
# ============================================================================
step "Bun"

log "Resolving Bun release ($BUN_VER)"
if [[ "$BUN_VER" == "latest" ]]; then
    BUN_TAG=$(curl -s "https://api.github.com/repos/oven-sh/bun/releases/latest" \
        | grep '"tag_name"' | grep -oP '(?<="tag_name": ")[^"]+')
    [[ -n "$BUN_TAG" ]] || die "Could not resolve latest Bun release."
else
    BUN_TAG="$BUN_VER"
fi
log "Bun tag: $BUN_TAG"
curl -L --retry 3 --progress-bar \
    -o "$OUT_DIR/apps/bun-linux-x64.zip" \
    "https://github.com/oven-sh/bun/releases/download/${BUN_TAG}/bun-linux-x64.zip"
echo "$BUN_TAG" > "$OUT_DIR/apps/bun.version"
log "Bun: $(du -sh "$OUT_DIR/apps/bun-linux-x64.zip" | cut -f1)"

# ============================================================================
# 6) PYTHON WHEELS ??LLM Inference (vLLM, FastAPI, LLM_API_fast)
# ============================================================================
step "Python wheels: Inference (torch ${TORCH_VER_INFERENCE}+${TORCH_CUDA_TAG})"

VENV_INF="$(mktemp -d)/venv"
"$PYTHON_BIN" -m venv "$VENV_INF"
# shellcheck disable=SC1091
source "$VENV_INF/bin/activate"
pip install --upgrade pip wheel setuptools
pip download --dest "$OUT_DIR/wheels/inference" pip wheel setuptools

log "Downloading torch==${TORCH_VER_INFERENCE}+${TORCH_CUDA_TAG} + torchvision + torchaudio"
pip download --dest "$OUT_DIR/wheels/inference" \
    --index-url "$TORCH_INDEX" \
    "torch==${TORCH_VER_INFERENCE}" torchvision torchaudio \
    || die "Failed to download torch==${TORCH_VER_INFERENCE} from $TORCH_INDEX"

_vllm_pkg="vllm"; [[ -n "$VLLM_VER" ]] && _vllm_pkg="vllm==${VLLM_VER}"
log "Downloading $_vllm_pkg (large, multi-GB)"
# Force resolution against the cu130 index so vLLM picks the cu130 PyTorch
# variant. Installer pins --torch-backend=cu130 too.
pip download --dest "$OUT_DIR/wheels/inference" \
    --index-url "$TORCH_INDEX" \
    --extra-index-url https://pypi.org/simple \
    "$_vllm_pkg" \
    || warn "vLLM download failed; check network."

# Same exclusion list used for training requirements at line ~931. Drop:
#   - Windows-only (pyreadline3)
#   - PyPI-deprecated/renamed (langchain-classic, xlwt)
#   - host-only tools that aren't useful in the airgap venv (aider-chat, pyinstaller, pip-system-certs)
#   - sdists that need a CUDA toolchain to build at install time (llama-cpp-python)
#     ^ llama.cpp itself is built from source on the target, so this Python
#       binding isn't needed; if a user wants it, build it manually after
#       sourcing /etc/profile.d/cuda.sh.
_INF_REQ_EXCLUDE_RE='^\s*#|^\s*$|^torch$|^torchvision$|^torchaudio$|pyreadline3|langchain-classic|xlwt|aider-chat|pyinstaller|llama-cpp-python|pip-system-certs'

if [[ -n "$LLMAPI_REQ" && -f "$LLMAPI_REQ" ]]; then
    cp "$LLMAPI_REQ" "$OUT_DIR/requirements/llm_api.txt"
    grep -vE "$_INF_REQ_EXCLUDE_RE" "$OUT_DIR/requirements/llm_api.txt" \
        | pip download --dest "$OUT_DIR/wheels/inference" -r /dev/stdin \
        || warn "Some LLM_API_fast packages failed."
fi
if [[ -n "$LLMAPI_FULL_REQ" && -f "$LLMAPI_FULL_REQ" ]]; then
    cp "$LLMAPI_FULL_REQ" "$OUT_DIR/requirements/llm_api_full.txt"
    grep -vE "$_INF_REQ_EXCLUDE_RE" "$OUT_DIR/requirements/llm_api_full.txt" \
        | pip download --dest "$OUT_DIR/wheels/inference" -r /dev/stdin \
        || warn "Some LLM_API_full packages failed."
fi

log "Downloading core inference / RAG wheels"
pip download --dest "$OUT_DIR/wheels/inference" \
    sentence-transformers faiss-cpu rank-bm25 \
    transformers tokenizers safetensors huggingface-hub tiktoken \
    langchain langchain-core langchain-community langchain-ollama \
    langgraph langgraph-checkpoint langgraph-prebuilt langsmith \
    ollama tavily-python \
    fastapi "uvicorn[standard]" pydantic pydantic-settings sse-starlette \
    httpx httpx-sse aiohttp aiofiles websockets \
    "passlib[bcrypt]" "python-jose[cryptography]" \
    PyMuPDF pypdf python-docx python-pptx openpyxl \
    pandas numpy Pillow python-dotenv python-multipart \
    jupyter_client ipykernel filelock tqdm rich \
    || warn "Some inference packages failed."

deactivate
rm -rf "$(dirname "$VENV_INF")"
log "Inference wheels: $(ls "$OUT_DIR/wheels/inference" | wc -l) files ($(du -sh "$OUT_DIR/wheels/inference" | cut -f1))"

# ============================================================================
# 7) PYTHON WHEELS ??General Training (PyG + Huni projects)
# ============================================================================
step "Python wheels: Training (torch ${TORCH_VER_TRAINING}+${TORCH_CUDA_TAG})"

PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VER_TRAINING}+${TORCH_CUDA_TAG}.html"
log "PyG index: $PYG_INDEX"

VENV_TR="$(mktemp -d)/venv"
"$PYTHON_BIN" -m venv "$VENV_TR"
# shellcheck disable=SC1091
source "$VENV_TR/bin/activate"
pip install --upgrade pip wheel setuptools
pip download --dest "$OUT_DIR/wheels/training" pip wheel setuptools

log "Downloading torch==${TORCH_VER_TRAINING}+${TORCH_CUDA_TAG} + torchvision + torchaudio"
pip download --dest "$OUT_DIR/wheels/training" --index-url "$TORCH_INDEX" \
    "torch==${TORCH_VER_TRAINING}" torchvision torchaudio \
    || die "Failed to download torch==${TORCH_VER_TRAINING} from $TORCH_INDEX"

log "Downloading torch-geometric"
pip download --dest "$OUT_DIR/wheels/training" torch-geometric

log "Downloading PyG extensions (pyg_lib, scatter, sparse, cluster)"
# Note: torch_spline_conv is NOT published on PyG cu130 index. If MeshGraphNets
# uses SplineConv, fall back to CPU op at runtime or build from source on target.
pip download --dest "$OUT_DIR/wheels/training" \
    --find-links "$PYG_INDEX" \
    pyg_lib torch-scatter torch-sparse torch-cluster \
    || warn "PyG extensions partial ??check $PYG_INDEX"

# Try torch_spline_conv but don't fail if missing
pip download --dest "$OUT_DIR/wheels/training" \
    --find-links "$PYG_INDEX" \
    torch-spline-conv 2>/dev/null \
    || warn "torch_spline_conv not in cu130 index ??install-time fallback if needed."

# Project requirements (per-Huni-project)
[[ -n "$MGN_REQ"              && -f "$MGN_REQ"              ]] && cp "$MGN_REQ"              "$OUT_DIR/requirements/meshgraphnets.txt"
[[ -n "$SIMULGEN_REQ"         && -f "$SIMULGEN_REQ"         ]] && cp "$SIMULGEN_REQ"         "$OUT_DIR/requirements/simulgen.txt"
[[ -n "$PEMTRON_REQ"          && -f "$PEMTRON_REQ"          ]] && cp "$PEMTRON_REQ"          "$OUT_DIR/requirements/pemtron.txt"
[[ -n "$PEMTRON_TRANSFER_REQ" && -f "$PEMTRON_TRANSFER_REQ" ]] && cp "$PEMTRON_TRANSFER_REQ" "$OUT_DIR/requirements/pemtron_transfer.txt"
[[ -n "$ALL_PROJECTS_REQ"     && -f "$ALL_PROJECTS_REQ"     ]] && cp "$ALL_PROJECTS_REQ"     "$OUT_DIR/requirements/all_projects.txt"

for rf in \
    "$OUT_DIR/requirements/meshgraphnets.txt" \
    "$OUT_DIR/requirements/simulgen.txt" \
    "$OUT_DIR/requirements/pemtron.txt" \
    "$OUT_DIR/requirements/pemtron_transfer.txt"; do
    [[ -f "$rf" ]] || continue
    log "  Downloading from $(basename "$rf")"
    # Drop torch/torchvision/torchaudio (already downloaded) + Windows-only or
    # unavailable packages (pyreadline3, langchain-classic, xlwt, aider-chat,
    # pyinstaller, llama-cpp-python builds against CUDA on target).
    grep -vE '^\s*#|^\s*$|^torch$|^torchvision$|^torchaudio$|pyreadline3|langchain-classic|xlwt|aider-chat|pyinstaller|llama-cpp-python|pip-system-certs' "$rf" \
        | pip download --dest "$OUT_DIR/wheels/training" -r /dev/stdin \
        || warn "Some packages from $(basename "$rf") failed."
done

log "Downloading core training/scientific wheels"
pip download --dest "$OUT_DIR/wheels/training" \
    numpy scipy h5py pandas tqdm matplotlib seaborn Pillow \
    scikit-learn scikit-image statsmodels networkx sympy \
    torchinfo tensorboard pytorch-warmup \
    opencv-python imageio librosa audiomentations soxr natsort \
    reportlab paramiko smbprotocol \
    || warn "Some training packages failed."

deactivate
rm -rf "$(dirname "$VENV_TR")"
log "Training wheels: $(ls "$OUT_DIR/wheels/training" | wc -l) files ($(du -sh "$OUT_DIR/wheels/training" | cut -f1))"

# ============================================================================
# 8) LLAMA.CPP ??source + Python utility wheels
# ============================================================================
step "llama.cpp source + utility wheels"

log "Cloning $LLAMA_REPO @ $LLAMA_REF"
git clone --recurse-submodules "$LLAMA_REPO" "$OUT_DIR/src/llama.cpp"
git -C "$OUT_DIR/src/llama.cpp" checkout "$LLAMA_REF"
git -C "$OUT_DIR/src/llama.cpp" submodule update --init --recursive
LLAMA_COMMIT=$(git -C "$OUT_DIR/src/llama.cpp" rev-parse HEAD)
{
    echo "BUNDLE_LLAMA_REF=$LLAMA_REF"
    echo "BUNDLE_LLAMA_COMMIT=$LLAMA_COMMIT"
} >> "$OUT_DIR/meta/target.env"
log "llama.cpp at commit $LLAMA_COMMIT"

log "Archiving source tree"
tar --exclude='.git' -C "$OUT_DIR/src" -czf "$OUT_DIR/src/llama.cpp.tar.gz" llama.cpp
rm -rf "$OUT_DIR/src/llama.cpp"

# Download wheels for convert_hf_to_gguf.py and other Python utilities.
log "Downloading llama.cpp utility wheels"
REQ_DIR="$(mktemp -d)"
tar -xzf "$OUT_DIR/src/llama.cpp.tar.gz" -C "$REQ_DIR"
REQ_ROOT="$REQ_DIR/llama.cpp"

LLAMA_REQ_FILES=()
if [[ -f "$REQ_ROOT/requirements.txt" ]]; then
    LLAMA_REQ_FILES+=( "$REQ_ROOT/requirements.txt" )
fi
if [[ -d "$REQ_ROOT/requirements" ]]; then
    while IFS= read -r f; do LLAMA_REQ_FILES+=( "$f" ); done \
        < <(find "$REQ_ROOT/requirements" -maxdepth 1 -name '*.txt')
fi

if (( ${#LLAMA_REQ_FILES[@]} > 0 )); then
    VENV_LLAMA="$REQ_DIR/venv"
    "$PYTHON_BIN" -m venv "$VENV_LLAMA"
    # shellcheck disable=SC1091
    source "$VENV_LLAMA/bin/activate"
    pip install --upgrade pip wheel setuptools
    pip download --dest "$OUT_DIR/wheels/llamacpp" pip wheel setuptools
    for rf in "${LLAMA_REQ_FILES[@]}"; do
        pip download --dest "$OUT_DIR/wheels/llamacpp" -r "$rf" \
            || warn "pip download failed for ${rf##*/}."
    done
    mkdir -p "$OUT_DIR/meta/requirements/llamacpp"
    cp "${LLAMA_REQ_FILES[@]}" "$OUT_DIR/meta/requirements/llamacpp/" 2>/dev/null || true
    deactivate
else
    warn "No llama.cpp requirements files found; skipping wheel download."
fi
rm -rf "$REQ_DIR"
log "llama.cpp: source archived, $(ls "$OUT_DIR/wheels/llamacpp" 2>/dev/null | wc -l) wheels"

# ============================================================================
# 9) PYTHON WHEELS ??Jupyter + data science
# ============================================================================
step "Python wheels: Jupyter + data science"

if [[ "$INCLUDE_JUPYTER" == "1" ]]; then
    VENV_J="$(mktemp -d)/venv"
    "$PYTHON_BIN" -m venv "$VENV_J"
    # shellcheck disable=SC1091
    source "$VENV_J/bin/activate"
    pip install --upgrade pip wheel setuptools
    pip download --dest "$OUT_DIR/wheels/jupyter" pip wheel setuptools

    log "Downloading JupyterLab + data science wheels"
    pip download --dest "$OUT_DIR/wheels/jupyter" \
        jupyterlab notebook ipykernel ipywidgets \
        jupyter-server jupyter-collaboration \
        pandas polars numpy scipy \
        matplotlib seaborn plotly \
        scikit-learn statsmodels \
        tqdm rich requests aiohttp \
        black ruff mypy pytest ipdb \
        || warn "Some Jupyter packages failed."

    deactivate
    rm -rf "$(dirname "$VENV_J")"
    log "Jupyter wheels: $(ls "$OUT_DIR/wheels/jupyter" | wc -l) files ($(du -sh "$OUT_DIR/wheels/jupyter" | cut -f1))"
else
    log "INCLUDE_JUPYTER=0; skipping Jupyter wheels."
fi

# ============================================================================
# 11) CHECKSUMS + BUNDLE
# ============================================================================
step "Checksums and bundle"

# Copy installer helpers into the bundle so install-all.sh has pre-install-check.sh available.
for helper in install-all.sh pre-install-check.sh test-all.sh; do
    if [[ -f "$SCRIPT_DIR/$helper" ]]; then
        cp "$SCRIPT_DIR/$helper" "$OUT_DIR/$helper"
        chmod +x "$OUT_DIR/$helper"
        log "Bundled helper: $helper"
    else
        warn "Helper not found at $SCRIPT_DIR/$helper ??bundle will lack this file."
    fi
done

# Bundle the install-all.d/ step scripts. The launcher install-all.sh sources
# 00-common.sh from this dir and invokes 01-..17- in order; without these the
# launcher exits immediately on the target.
if [[ -d "$SCRIPT_DIR/install-all.d" ]]; then
    rm -rf "$OUT_DIR/install-all.d"
    cp -r "$SCRIPT_DIR/install-all.d" "$OUT_DIR/install-all.d"
    find "$OUT_DIR/install-all.d" -name '*.sh' -exec chmod +x {} +
    log "Bundled helper dir: install-all.d/ ($(ls "$OUT_DIR/install-all.d" | wc -l) files)"
else
    die "install-all.d/ not found at $SCRIPT_DIR — cannot ship a bundle without the step scripts."
fi

log "Generating SHA256 manifest (excluding meta/SHA256SUMS itself)"
(
    cd "$OUT_DIR"
    find install-all.sh install-all.d pre-install-check.sh test-all.sh debs apps wheels requirements src meta \
        -type f \
        ! -path 'meta/SHA256SUMS' \
        -print0 2>/dev/null \
        | sort -z \
        | xargs -0 sha256sum > meta/SHA256SUMS
)

BUNDLE_PARENT="$(dirname "$OUT_DIR")"
BUNDLE_BIN="$BUNDLE_PARENT/all-airgap-bundle-ubuntu${TARGET_OS_VERSION}.bin"
log "Packing bundle -> $BUNDLE_BIN (this can take several minutes)"
rm -f "$BUNDLE_BIN" "${BUNDLE_BIN}.sha256"
tar -czf "$BUNDLE_BIN" -C "$BUNDLE_PARENT" "$(basename "$OUT_DIR")"

log "Generating bundle SHA256 sidecar"
( cd "$BUNDLE_PARENT" && sha256sum "$(basename "$BUNDLE_BIN")" > "$(basename "$BUNDLE_BIN").sha256" )

# Copy installer helpers next to the bundle so the user only needs to
# transfer the bundle + .sha256 + these scripts (the bundle ALSO carries
# them, but having them sibling lets users invoke pre-install-check.sh
# without first extracting).
for helper in install-all.sh pre-install-check.sh test-all.sh; do
    [[ -f "$SCRIPT_DIR/$helper" ]] || continue
    cp "$SCRIPT_DIR/$helper" "$BUNDLE_PARENT/$helper"
    chmod +x "$BUNDLE_PARENT/$helper"
done
if [[ -d "$SCRIPT_DIR/install-all.d" ]]; then
    rm -rf "$BUNDLE_PARENT/install-all.d"
    cp -r "$SCRIPT_DIR/install-all.d" "$BUNDLE_PARENT/install-all.d"
    find "$BUNDLE_PARENT/install-all.d" -name '*.sh' -exec chmod +x {} +
fi

log "Done."
printf '\n'
printf '  Bundle    : %s (%s)\n' "$BUNDLE_BIN" "$(du -sh "$BUNDLE_BIN" | cut -f1)"
printf '  SHA256    : %s\n' "${BUNDLE_BIN}.sha256"
printf '  Pre-flight: %s\n' "$BUNDLE_PARENT/pre-install-check.sh"
printf '  Installer : %s\n' "$BUNDLE_PARENT/install-all.sh"
printf '  Verifier  : %s\n' "$BUNDLE_PARENT/test-all.sh"
printf '  Staging   : %s\n' "$OUT_DIR"
printf '\n'
printf 'Transfer to airgapped server:\n'
printf '  scp -r "%s" "%s" \\\n         "%s" "%s" "%s" "%s/install-all.d" user@SERVER:~\n' \
    "$BUNDLE_BIN" "${BUNDLE_BIN}.sha256" \
    "$BUNDLE_PARENT/pre-install-check.sh" \
    "$BUNDLE_PARENT/install-all.sh" \
    "$BUNDLE_PARENT/test-all.sh" \
    "$BUNDLE_PARENT"
printf '  ssh user@SERVER\n'
printf '  sudo bash pre-install-check.sh   # readiness gate\n'
printf '  sudo bash install-all.sh         # auto-extracts the bundle, runs install-all.d/01-..17-\n'
printf '  sudo bash install-all.sh --list  # any time: see per-step status\n'
printf '  sudo bash install-all.sh --rerun 14   # re-run a specific failed step\n'
printf '  sudo bash test-all.sh            # post-install verification\n'
printf '\n'
printf 'Bundle variant: prepped (vendor pre-installed driver + CUDA 13.0)\n'
printf 'Driver baseline: %s\n' "$DRIVER_BASELINE"
