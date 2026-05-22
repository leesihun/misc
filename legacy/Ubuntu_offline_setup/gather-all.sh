#!/usr/bin/env bash
# ============================================================================
# gather-all.sh
#   Run on an internet-connected WSL Ubuntu 24.04 machine.
#   Downloads all packages needed for the air-gapped Ubuntu 24.04 + CUDA 13.0
#   server with 8x B300 GPUs.
#
#   Produces: ~/GPU_server_downloads/  +  all-airgap-bundle-ubuntu<OS>.bin
#   Transfer the bundle + installer to the air-gapped server, then run install-all.sh.
#
#   NOTE: WSL distro MUST match target OS (22.04 or 24.04) for .deb compat.
#
# Usage:
#   bash gather-all.sh                         # default settings
#   INCLUDE_K3S=1 bash gather-all.sh           # include K3s + container images
#   INCLUDE_NVIDIA_DRIVER=0 bash gather-all.sh # skip NVIDIA driver debs
# ============================================================================
set -euo pipefail

# ============================================================================
# CONFIGURATION — edit these before running if needed
# ============================================================================

OUT_DIR="${OUT_DIR:-$HOME/GPU_server_downloads}"

# Python
PYTHON_VER="${PYTHON_VER:-3.12}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

# PyTorch + CUDA 13.0
# cu130 index confirmed: torch 2.9/2.10/2.11 for Python 3.12 (manylinux_2_28_x86_64)
# PyG also confirms cu130 support for torch 2.11: https://data.pyg.org/whl/torch-2.11.0+cu130.html
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu130}"
# The vLLM and training venvs are isolated, but both can use the current
# PyTorch cu130 release line.
TORCH_VER_INFERENCE="${TORCH_VER_INFERENCE:-2.11.0}"
TORCH_VER_TRAINING="${TORCH_VER_TRAINING:-2.11.0}"     # latest — best B300/H200 kernel support

# App URLs
VSCODE_URL="https://update.code.visualstudio.com/latest/linux-deb-x64/stable"
CHROME_URL="https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
# Firefox: set to a version number (e.g. "128.0") or "latest" to auto-resolve
FIREFOX_VER="${FIREFOX_VER:-latest}"
FIREFOX_LANG="${FIREFOX_LANG:-en-US}"

# Opencode (https://github.com/sst/opencode)
# "latest" or a specific tag like "v0.3.0"
OPENCODE_VER="${OPENCODE_VER:-latest}"

# Node.js LTS (ships npm). LTS major version — 22 is "Jod" LTS as of 2025.
NODE_LTS_MAJOR="${NODE_LTS_MAJOR:-22}"

# Bun (https://github.com/oven-sh/bun)
BUN_VER="${BUN_VER:-latest}"  # "latest" or a specific tag like "bun-v1.2.0"

# vLLM
# Empty means latest stable from PyPI. The latest vLLM release may not publish
# a separate "+cu130" GitHub wheel even though it supports CUDA 13.x via the
# PyTorch/cu13 dependency stack. For an older explicit cu130 wheel, run:
#   VLLM_VER=0.19.0 VLLM_CUDA_WHEEL=1 REQUIRE_VLLM_CUDA_WHEEL=1 bash gather-all.sh
VLLM_VER="${VLLM_VER:-}"
VLLM_CUDA_WHEEL="${VLLM_CUDA_WHEEL:-0}"
REQUIRE_VLLM_CUDA_WHEEL="${REQUIRE_VLLM_CUDA_WHEEL:-0}"

# llama.cpp — "master" always tracks the latest commit
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"

# nccl-tests — NVIDIA's canonical multi-GPU bandwidth benchmark suite.
# Bundled as source; built on the target where CUDA+NCCL are present.
# After install: /usr/local/bin/all_reduce_perf -b 8 -e 8G -f 2 -g 8
# On 8x B300 with NVSwitch, expect >700 GB/s busBW; <100 GB/s indicates
# NVLink/FabricManager isn't being used and traffic is falling back to PCIe.
INCLUDE_NCCL_TESTS="${INCLUDE_NCCL_TESTS:-1}"
NCCL_TESTS_REPO="${NCCL_TESTS_REPO:-https://github.com/NVIDIA/nccl-tests.git}"
NCCL_TESTS_REF="${NCCL_TESTS_REF:-master}"

# CUDA toolkit (needed to BUILD llama.cpp with CUDA on the target).
INCLUDE_CUDA_TOOLKIT="${INCLUDE_CUDA_TOOLKIT:-1}"
CUDA_META_PKG="${CUDA_META_PKG:-cuda-toolkit-13-0}"

# NVIDIA GPU driver (needed at RUNTIME for GPU workloads).
# B300/Blackwell + CUDA 13.0 should use the R580 data-center branch unless you
# intentionally override this.
# IMPORTANT: every package name MUST be branch-suffixed. The unversioned
# meta-packages (nvidia-open, nvidia-fabricmanager, libnvidia-nscq) resolve
# to whatever branch apt thinks is newest on the gather host, which then
# pulls a mismatched 595.* (or 600.*) driver into a bundle nominally pinned
# to 580 and breaks the install with file conflicts on the target.
INCLUDE_NVIDIA_DRIVER="${INCLUDE_NVIDIA_DRIVER:-1}"
NVIDIA_DRIVER_BRANCH="${NVIDIA_DRIVER_BRANCH:-580}"
# nvidia-fabricmanager and libnvidia-nscq are data-center packages that use
# server-specific deps (nvidia-kernel-common-<branch>-server) not available on
# desktop gather hosts. Keep them unversioned so apt resolves them against the
# server repo once nvidia-driver-pinning-<branch> is in effect on the target.
NVIDIA_DRIVER_PKGS="${NVIDIA_DRIVER_PKGS:-nvidia-driver-pinning-${NVIDIA_DRIVER_BRANCH} nvidia-driver-${NVIDIA_DRIVER_BRANCH}-open nvidia-utils-${NVIDIA_DRIVER_BRANCH} nvidia-fabricmanager libnvidia-nscq}"
INCLUDE_NVIDIA_CONTAINER_TOOLKIT="${INCLUDE_NVIDIA_CONTAINER_TOOLKIT:-1}"
CONFIGURE_NVIDIA_APT_REPO="${CONFIGURE_NVIDIA_APT_REPO:-1}"

# Desktop environment: XFCE4 + xrdp for remote-desktop access
# Set INSTALL_DESKTOP=0 to skip if running headless-only.
INSTALL_DESKTOP="${INSTALL_DESKTOP:-1}"

# Jupyter + data science wheels (notebook/exploratory work on each node)
INCLUDE_JUPYTER="${INCLUDE_JUPYTER:-1}"

# K3s cluster orchestration — set INCLUDE_K3S=1 to bundle everything needed
INCLUDE_K3S="${INCLUDE_K3S:-0}"
K3S_VER="${K3S_VER:-v1.31.4+k3s1}"
HELM_VER="${HELM_VER:-v3.16.3}"
KUBECTL_VER="${KUBECTL_VER:-v1.31.4}"
GPU_OPERATOR_CHART_VER="${GPU_OPERATOR_CHART_VER:-v25.3.2}"
KUBE_PROM_STACK_CHART_VER="${KUBE_PROM_STACK_CHART_VER:-66.3.1}"
LOKI_STACK_CHART_VER="${LOKI_STACK_CHART_VER:-2.10.2}"
KUBERAY_CHART_VER="${KUBERAY_CHART_VER:-1.2.2}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-registry:2.8.3}"
INCLUDE_K3S_EXAMPLE_IMAGES="${INCLUDE_K3S_EXAMPLE_IMAGES:-0}"
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-}"
RAY_IMAGE_TAG="${RAY_IMAGE_TAG:-}"
PYTORCH_IMAGE_TAG="${PYTORCH_IMAGE_TAG:-2.11.0-cuda13.0-cudnn9-runtime}"

# Target OS version (auto-detected from WSL; used in the bundle filename so
# Ubuntu 22.04 and 24.04 bundles don't overwrite each other)
TARGET_OS_VERSION="${TARGET_OS_VERSION:-$(. /etc/os-release && echo "$VERSION_ID")}"
CUDA_REPO_DIST="${CUDA_REPO_DIST:-ubuntu${TARGET_OS_VERSION//./}}"
case "$(dpkg --print-architecture 2>/dev/null || echo amd64)" in
    amd64) CUDA_REPO_ARCH="${CUDA_REPO_ARCH:-x86_64}" ;;
    arm64) CUDA_REPO_ARCH="${CUDA_REPO_ARCH:-sbsa}" ;;
    *)     CUDA_REPO_ARCH="${CUDA_REPO_ARCH:-x86_64}" ;;
esac

# Requirements files — all auto-detected; override via env vars if needed.
# ── Script location ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Huni root: works from Windows path on WSL (/mnt/c/...) or native Linux ──
_find_huni_dir() {
    local candidates=(
        "$SCRIPT_DIR/.."
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

# Per-project requirements (override with full path if auto-detect fails)
LLMAPI_REQ="${LLMAPI_REQ:-}"
MGN_REQ="${MGN_REQ:-}"
SIMULGEN_REQ="${SIMULGEN_REQ:-}"
PEMTRON_REQ="${PEMTRON_REQ:-}"
PEMTRON_TRANSFER_REQ="${PEMTRON_TRANSFER_REQ:-}"
LLMAPI_FULL_REQ="${LLMAPI_FULL_REQ:-}"    # temp/LLM_API full requirements
ALL_PROJECTS_REQ="${ALL_PROJECTS_REQ:-}"  # misc/requirements-all-projects.txt

# ============================================================================

log()  { printf '\033[1;36m[gather]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gather:WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[gather:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;35m══ %s ══\033[0m\n' "$*"; }

[[ $EUID -eq 0 ]] && die "Do NOT run as root. Script will invoke sudo as needed."
command -v sudo  >/dev/null || die "sudo is required."
command -v curl  >/dev/null || die "curl is required. Run: sudo apt-get install curl"
command -v git   >/dev/null || die "git is required.  Run: sudo apt-get install git"

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

# ── Auto-detect requirements files ──────────────────────────────────────────
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
    || warn "Huni project root not found — set HUNI_DIR=/path/to/Huni to fix auto-detection."

_try_req LLMAPI_REQ \
    "${HUNI_DIR}/LLM_API_fast/requirements.txt" \
    "$SCRIPT_DIR/../LLM_API_fast/requirements.txt" \
    "$HOME/LLM_API_fast/requirements.txt"

_try_req MGN_REQ \
    "${HUNI_DIR}/MeshGraphNets - variational/requirements.txt" \
    "$SCRIPT_DIR/../MeshGraphNets - variational/requirements.txt" \
    "$HOME/MeshGraphNets/requirements.txt"

_try_req SIMULGEN_REQ \
    "${HUNI_DIR}/SimulGenVAE/requirements.txt" \
    "$SCRIPT_DIR/../SimulGenVAE/requirements.txt"

_try_req PEMTRON_REQ \
    "${HUNI_DIR}/PEMTRON_warpage/requirements.txt" \
    "$SCRIPT_DIR/../PEMTRON_warpage/requirements.txt"

_try_req PEMTRON_TRANSFER_REQ \
    "${HUNI_DIR}/PEMTRON_warpage/data_autotransfer/requirements.txt" \
    "$SCRIPT_DIR/../PEMTRON_warpage/data_autotransfer/requirements.txt"

_try_req LLMAPI_FULL_REQ \
    "${HUNI_DIR}/temp/LLM_API/requirements.txt" \
    "$SCRIPT_DIR/../temp/LLM_API/requirements.txt"

_try_req ALL_PROJECTS_REQ \
    "$SCRIPT_DIR/requirements-all-projects.txt" \
    "${HUNI_DIR}/misc/requirements-all-projects.txt"

# ── Ensure local prerequisites for running THIS script ──────────────────────
step "Local prerequisites"
_need=()
command -v pip3          >/dev/null || _need+=( python3-pip )
"$PYTHON_BIN" -c '' 2>/dev/null    || _need+=( "python${PYTHON_VER}" )
"$PYTHON_BIN" -m venv --help &>/dev/null || _need+=( "python${PYTHON_VER}-venv" python3-venv )
command -v dpkg          >/dev/null || _need+=( dpkg )
command -v dpkg-scanpackages >/dev/null || _need+=( dpkg-dev )
command -v apt-rdepends  >/dev/null || _need+=( apt-rdepends )
if (( ${#_need[@]} > 0 )); then
    log "Installing local prerequisites: ${_need[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${_need[@]}" || true
fi
"$PYTHON_BIN" -m venv --help &>/dev/null \
    || die "$PYTHON_BIN -m venv unavailable. Install python${PYTHON_VER}-venv."
command -v dpkg-scanpackages >/dev/null \
    || die "dpkg-scanpackages unavailable. Install dpkg-dev."
command -v apt-rdepends >/dev/null \
    || die "apt-rdepends unavailable. Install apt-rdepends."

# ── Setup output tree ────────────────────────────────────────────────────────
log "Output directory: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{debs,apps,src,wheels/inference,wheels/training,wheels/llamacpp,wheels/jupyter,requirements,meta}
mkdir -p "$OUT_DIR"/k3s/{bin,airgap-images,images,charts,manifests,systemd,meta}

# Needed on a fresh internet-connected Ubuntu gather host before downloading
# CUDA Toolkit, data-center driver, Fabric Manager, or container toolkit debs.
if [[ "$CONFIGURE_NVIDIA_APT_REPO" == "1" && ( "$INCLUDE_CUDA_TOOLKIT" == "1" || "$INCLUDE_NVIDIA_DRIVER" == "1" || "$INCLUDE_NVIDIA_CONTAINER_TOOLKIT" == "1" ) ]]; then
    step "NVIDIA CUDA apt repository"
    CUDA_REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_DIST}/${CUDA_REPO_ARCH}"
    KEYRING_DEB="$OUT_DIR/meta/cuda-keyring_1.1-1_all.deb"
    PIN_FILE="$OUT_DIR/meta/cuda-${CUDA_REPO_DIST}.pin"

    log "Repository: $CUDA_REPO_BASE"
    curl -fsSL --retry 3 -o "$KEYRING_DEB" "$CUDA_REPO_BASE/cuda-keyring_1.1-1_all.deb" \
        || die "Could not download cuda-keyring from $CUDA_REPO_BASE"
    cp "$KEYRING_DEB" "$OUT_DIR/debs/"
    sudo dpkg -i "$KEYRING_DEB"

    if curl -fsSL --retry 3 -o "$PIN_FILE" "$CUDA_REPO_BASE/cuda-${CUDA_REPO_DIST}.pin"; then
        sudo install -m 0644 "$PIN_FILE" /etc/apt/preferences.d/cuda-repository-pin-600
        log "CUDA repository pin installed."
    else
        warn "CUDA repository pin not found; continuing with cuda-keyring only."
    fi

    if [[ "$INCLUDE_NVIDIA_DRIVER" == "1" ]]; then
        log "Installing NVIDIA driver pinning package for branch ${NVIDIA_DRIVER_BRANCH}"
        sudo apt-get update -qq
        sudo apt-get install -y "nvidia-driver-pinning-${NVIDIA_DRIVER_BRANCH}" \
            || warn "Driver pinning package was not installed; driver branch selection may drift."
    fi
fi

# ── NVIDIA Container Toolkit apt repo ─────────────────────────────────────
# The CUDA repo above does NOT host the container toolkit packages — those
# live in a separate repo at nvidia.github.io. If we don't add it here, apt
# can't resolve nvidia-container-toolkit / libnvidia-container1 / etc., and
# they end up missing from the bundle (silent failure: install on the target
# then complains "Unable to locate package nvidia-container-toolkit").
if [[ "$CONFIGURE_NVIDIA_APT_REPO" == "1" && "$INCLUDE_NVIDIA_CONTAINER_TOOLKIT" == "1" ]]; then
    step "NVIDIA Container Toolkit apt repository"
    NVCT_KEY="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    NVCT_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
    NVCT_DIST="${NVCT_DIST:-stable/deb}"

    if [[ ! -f "$NVCT_KEY" ]]; then
        log "Fetching NVIDIA container toolkit GPG key"
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o "$NVCT_KEY" \
            || die "Could not fetch NVIDIA container toolkit GPG key"
    fi
    log "Configuring repo: https://nvidia.github.io/libnvidia-container/${NVCT_DIST}"
    curl -fsSL "https://nvidia.github.io/libnvidia-container/${NVCT_DIST}/nvidia-container-toolkit.list" \
        | sed "s#deb https://#deb [signed-by=${NVCT_KEY}] https://#g" \
        | sudo tee "$NVCT_LIST" >/dev/null \
        || die "Could not fetch NVIDIA container toolkit apt list"
    sudo apt-get update -qq \
        || warn "apt-get update reported errors; container toolkit packages may be unavailable."

    # Stage the keyring + list into the bundle so the install can install the
    # packages on the target (apt won't trust the bundled repo without the key).
    sudo install -m 0644 "$NVCT_KEY"  "$OUT_DIR/meta/nvidia-container-toolkit-keyring.gpg"
    sudo install -m 0644 "$NVCT_LIST" "$OUT_DIR/meta/nvidia-container-toolkit.list"
fi

source /etc/os-release
cat > "$OUT_DIR/meta/target.env" <<EOF
BUNDLE_OS_ID=$ID
BUNDLE_OS_VERSION=$VERSION_ID
BUNDLE_ARCH=$(dpkg --print-architecture)
BUNDLE_PYTHON=$PYTHON_VER
BUNDLE_TORCH_INFERENCE=$TORCH_VER_INFERENCE
BUNDLE_TORCH_TRAINING=$TORCH_VER_TRAINING
BUNDLE_TORCH_CUDA=$TORCH_CUDA_TAG
BUNDLE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUNDLE_INCLUDE_K3S=$INCLUDE_K3S
BUNDLE_K3S_VER=$K3S_VER
BUNDLE_TARGET_OS=$TARGET_OS_VERSION
BUNDLE_INCLUDE_NVIDIA_DRIVER=$INCLUDE_NVIDIA_DRIVER
BUNDLE_NVIDIA_DRIVER_BRANCH=$NVIDIA_DRIVER_BRANCH
BUNDLE_INCLUDE_CUDA_TOOLKIT=$INCLUDE_CUDA_TOOLKIT
BUNDLE_CUDA_REPO_DIST=$CUDA_REPO_DIST
BUNDLE_CUDA_REPO_ARCH=$CUDA_REPO_ARCH
EOF
log "Host: $ID $VERSION_ID / $(dpkg --print-architecture)"

# ============================================================================
# 1) APT PACKAGES
# ============================================================================
step "APT packages"

APT_PKGS=(
    # Python 3.12 ecosystem
    python3.12
    python3.12-venv
    python3.12-dev
    python3-pip
    python3-setuptools
    python3-wheel

    # Build toolchain (for any packages needing native compilation)
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

    # Numeric / ML native libs
    libopenblas-dev
    libopenblas0
    libgomp1
    libhdf5-dev          # h5py native build
    libssl-dev
    libffi-dev
    libcurl4-openssl-dev # llama-server curl support (LLAMA_CURL=ON)

    # System utilities
    htop
    tmux
    vim
    tree
    rsync
    mokutil                  # Secure Boot state check for NVIDIA DKMS modules
    skopeo                   # OCI image copy — needed on server to push images to airgap registry

    # ── Multi-GPU / NUMA tooling ─────────────────────────────────────────
    # numactl: pin inference workers to the NUMA node nearest their GPU.
    # On a 2-socket 8x B300 box, GPUs split across NUMA nodes; wrong binding
    # costs ~30% throughput on memory-bound workloads.
    # hwloc-nox: lstopo for visualizing CPU/NUMA/GPU/PCIe topology (no X deps).
    # nvtop: TUI like htop but for GPUs — essential for monitoring long training
    # runs and catching one straggler GPU dragging down a collective.
    numactl
    hwloc-nox
    nvtop

    # Runtime libs for ML
    libgl1
    libgles2

    # GUI runtime libs — required by VS Code and Chrome .deb packages.
    # Without these, dpkg -i will succeed but the apps won't launch.
    xz-utils                 # needed to extract Node.js .tar.xz
    libglib2.0-0
    libatk1.0-0
    libatk-bridge2.0-0
    libcairo2
    libcups2                 # Ubuntu 24.04: virtual → libcups2t64
    libdbus-1-3
    libdrm2
    libexpat1
    libfontconfig1
    fonts-liberation
    libgbm1
    libgtk-3-0               # Ubuntu 24.04: virtual → libgtk-3-0t64
    libnspr4
    libnss3
    libpango-1.0-0
    libsecret-1-0
    libasound2t64            # Ubuntu 24.04 explicit t64 variant (libasound2 is unresolvable virtual)
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
        # ── XFCE4 desktop ──────────────────────────────────────────────────────
        xfce4
        xfce4-goodies           # panel plugins, power manager, etc.
        xfce4-terminal
        xfce4-screenshooter
        xfce4-taskmanager
        xfce4-notifyd

        # ── Display manager ────────────────────────────────────────────────────
        lightdm
        lightdm-gtk-greeter
        lightdm-gtk-greeter-settings

        # ── Remote desktop (RDP on port 3389) ─────────────────────────────────
        xrdp
        xorgxrdp
        ssl-cert                # xrdp TLS certificate group

        # ── Text editors ───────────────────────────────────────────────────────
        gedit
        mousepad                # lightweight XFCE-native editor

        # ── File manager & viewers ─────────────────────────────────────────────
        thunar                  # XFCE file manager (also in xfce4-goodies)
        file-roller             # archive manager
        ristretto               # image viewer
        evince                  # PDF/document viewer

        # ── Misc desktop utilities ─────────────────────────────────────────────
        galculator              # calculator
        xclip                   # clipboard CLI tool
        xdotool                 # X11 automation
        dconf-editor            # settings editor

        # ── X11 / session infrastructure ──────────────────────────────────────
        dbus-x11
        x11-xserver-utils       # xrandr, xhost, etc.
        x11-utils               # xlsfonts, xwininfo, etc.
        xauth
        xinit
        xterm                   # minimal fallback terminal

        # ── Network manager GUI ────────────────────────────────────────────────
        network-manager-gnome

        # ── Fonts ──────────────────────────────────────────────────────────────
        fonts-dejavu-core
        fonts-noto-core
        fonts-noto-color-emoji

        # ── Themes & icons ─────────────────────────────────────────────────────
        adwaita-icon-theme
        gnome-themes-extra
        gtk2-engines-pixbuf

        # ── Archive / compression ──────────────────────────────────────────────
        p7zip-full

        # ── Shell quality-of-life ──────────────────────────────────────────────
        bash-completion
    )
    log "Desktop packages added: XFCE4, xrdp, gedit, and utilities"
fi

if [[ "$INCLUDE_NVIDIA_DRIVER" == "1" ]]; then
    # Kernel/header packages are needed so DKMS can build on the air-gapped
    # host after rebooting into the bundled generic Ubuntu kernel.
    APT_PKGS+=( dkms linux-generic linux-headers-generic )
    read -r -a _nvidia_driver_pkgs <<< "$NVIDIA_DRIVER_PKGS"
    APT_PKGS+=( "${_nvidia_driver_pkgs[@]}" )
    log "NVIDIA driver enabled: ${_nvidia_driver_pkgs[*]}"
fi

if [[ "$INCLUDE_NVIDIA_CONTAINER_TOOLKIT" == "1" ]]; then
    APT_PKGS+=( nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container1 libnvidia-container-tools )
    log "NVIDIA container toolkit enabled."
    # Pre-check the gather host can actually resolve these. If apt-cache has
    # no candidate, the container-toolkit repo isn't configured (see the
    # 'NVIDIA Container Toolkit apt repository' step above) — fail loud now
    # rather than ship a bundle that's silently missing these packages.
    for _ctk in nvidia-container-toolkit libnvidia-container1; do
        if ! apt-cache policy "$_ctk" 2>/dev/null | awk '/Candidate:/ && $2 != "(none)" {found=1} END {exit !found}'; then
            die "apt has no candidate for $_ctk — the nvidia container toolkit repo isn't configured on this gather host. Set CONFIGURE_NVIDIA_APT_REPO=1 (default) or add the repo manually."
        fi
    done
fi

if [[ "$INCLUDE_CUDA_TOOLKIT" == "1" ]]; then
    # NCCL is the NVIDIA collective-comms lib that backs multi-GPU NVLink/NVSwitch
    # traffic (incl. NVLS SHARP on B100/B200/B300). Without libnccl-dev present at
    # llama.cpp build time, ggml-cuda's `find_package(NCCL)` fails and the build
    # falls back to a slow per-pair P2P copy on multi-GPU inference.
    #
    # datacenter-gpu-manager (DCGM): NVIDIA's fleet telemetry / health daemon.
    # Tracks XID errors, NVLink lane errors, thermal throttling. Essential for
    # any multi-GPU box you actually want to operate (vs. just demo).
    APT_PKGS+=( "$CUDA_META_PKG" libnccl2 libnccl-dev datacenter-gpu-manager )
    log "CUDA toolkit enabled: $CUDA_META_PKG (+ libnccl2/libnccl-dev + datacenter-gpu-manager)"
fi

printf '%s\n' "${APT_PKGS[@]}" > "$OUT_DIR/meta/apt-packages.txt"

log "Refreshing apt indexes (60 s timeout per source to avoid slow NVIDIA repo hangs)"
sudo apt-get update \
    -o Acquire::http::Timeout=60 \
    -o Acquire::https::Timeout=60 \
    -o Acquire::Retries=2 \
    || warn "apt-get update had errors (non-critical repos may be unreachable); continuing."

log "Cleaning local apt cache to isolate downloads"
sudo apt-get clean

log "Downloading ${#APT_PKGS[@]} apt package groups (+ transitive deps)"
sudo apt-get install -y --download-only --reinstall "${APT_PKGS[@]}"

_apt_has_candidate() {
    local pkg="$1"
    apt-cache policy "$pkg" 2>/dev/null \
        | awk '/Candidate:/ && $2 != "(none)" {found=1} END {exit !found}'
}

_apt_installed_ok() {
    local pkg="$1"
    dpkg-query -W -f='${Status}\n' "$pkg" 2>/dev/null \
        | grep -qx 'install ok installed'
}

_apt_virtual_providers() {
    local pkg="$1"
    apt-cache showpkg "$pkg" 2>/dev/null \
        | awk '
            /^Reverse Provides:/ {providers=1; next}
            providers && NF {print $1}
        ' \
        | sort -u
}

_apt_preferred_provider() {
    local pkg="$1"
    case "$pkg" in
        aspell-dictionary) printf '%s\n' aspell-en ;;
        awk) printf '%s\n' mawk ;;
        container-network-stack) printf '%s\n' containernetworking-plugins ;;
        dbus-session-bus|default-dbus-session-bus) printf '%s\n' dbus-user-session ;;
        dbus-system-bus|default-dbus-system-bus) printf '%s\n' dbus ;;
        default-logind|elogind|logind) printf '%s\n' libpam-systemd ;;
        hunspell-dictionary|myspell-dictionary) printf '%s\n' hunspell-en-us ;;
        ispell-dictionary) printf '%s\n' iamerican ;;
        libtinfo5) printf '%s\n' libtinfo6 ;;
        mime-support) printf '%s\n' media-types ;;
        openjdk-7-jre) printf '%s\n' default-jre ;;
    esac
}

_apt_resolve_download_name() {
    local pkg="$1" provider
    if _apt_has_candidate "$pkg"; then
        printf '%s\n' "$pkg"
        return 0
    fi

    while IFS= read -r provider; do
        [[ -n "$provider" ]] || continue
        if _apt_has_candidate "$provider"; then
            printf '%s\n' "$provider"
            return 0
        fi
    done < <(_apt_preferred_provider "$pkg")

    while IFS= read -r provider; do
        [[ -n "$provider" ]] || continue
        if _apt_installed_ok "$provider" && _apt_has_candidate "$provider"; then
            printf '%s\n' "$provider"
            return 0
        fi
    done < <(_apt_virtual_providers "$pkg")

    while IFS= read -r provider; do
        [[ -n "$provider" ]] || continue
        if _apt_has_candidate "$provider"; then
            printf '%s\n' "$provider"
            return 0
        fi
    done < <(_apt_virtual_providers "$pkg")

    return 1
}

log "Downloading full apt dependency closure (including packages already installed on gather host)"
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
    if resolved_pkg="$(_apt_resolve_download_name "$pkg")"; then
        printf '%s\n' "$resolved_pkg" >> "$OUT_DIR/meta/apt-closure-download.txt"
        if [[ "$resolved_pkg" != "$pkg" ]]; then
            printf '%s -> %s\n' "$pkg" "$resolved_pkg" >> "$OUT_DIR/meta/apt-closure-virtual.txt"
        fi
    else
        printf '%s\n' "$pkg" >> "$OUT_DIR/meta/apt-closure-unresolved.txt"
    fi
done < "$OUT_DIR/meta/apt-closure.txt"

sort -u "$OUT_DIR/meta/apt-closure-download.txt" -o "$OUT_DIR/meta/apt-closure-download.txt"
sort -u "$OUT_DIR/meta/apt-closure-virtual.txt" -o "$OUT_DIR/meta/apt-closure-virtual.txt"
sort -u "$OUT_DIR/meta/apt-closure-unresolved.txt" -o "$OUT_DIR/meta/apt-closure-unresolved.txt"

while IFS= read -r pkg; do
    [[ -n "$pkg" ]] || continue
    ( cd "$OUT_DIR/debs" && apt-get download "$pkg" >/dev/null 2>&1 ) \
        || printf '%s\n' "$pkg" >> "$OUT_DIR/meta/apt-closure-download-failed.txt"
done < "$OUT_DIR/meta/apt-closure-download.txt"

virtual_count=$(grep -c . "$OUT_DIR/meta/apt-closure-virtual.txt" 2>/dev/null || true)
unresolved_count=$(grep -c . "$OUT_DIR/meta/apt-closure-unresolved.txt" 2>/dev/null || true)
failed_count=$(grep -c . "$OUT_DIR/meta/apt-closure-download-failed.txt" 2>/dev/null || true)
if (( virtual_count > 0 )); then
    log "Resolved $virtual_count virtual apt dependencies; see meta/apt-closure-virtual.txt"
fi
if (( unresolved_count > 0 )); then
    warn "$unresolved_count virtual/ABI apt dependency names had no downloadable provider; see meta/apt-closure-unresolved.txt"
fi
if (( failed_count > 0 )); then
    warn "$failed_count real apt dependencies failed to download; see meta/apt-closure-download-failed.txt"
fi

shopt -s nullglob
debs=(/var/cache/apt/archives/*.deb)
if (( ${#debs[@]} > 0 )); then
    sudo cp "${debs[@]}" "$OUT_DIR/debs/"
fi
sudo chown -R "$(id -u):$(id -g)" "$OUT_DIR/debs"

# ── Critical-package presence check ──────────────────────────────────────────
# Before declaring the bundle good, verify the packages that the install
# script absolutely requires actually exist in debs/. The historical failure
# mode: a missing apt repo (e.g. nvidia container toolkit) means gather
# "succeeds" silently, then install on the target dies with "Unable to
# locate package nvidia-container-toolkit". Fail loudly at gather time.
_required_globs=()
[[ "$INCLUDE_CUDA_TOOLKIT" == "1" ]] && _required_globs+=(
    "cuda-toolkit-*"
    "cuda-nvcc-*"
    "libnccl2_*"
    "libnccl-dev_*"
)
[[ "$INCLUDE_NVIDIA_DRIVER" == "1" ]] && _required_globs+=(
    "nvidia-driver-${NVIDIA_DRIVER_BRANCH}*"
    "nvidia-dkms-${NVIDIA_DRIVER_BRANCH}*"
    "libnvidia-compute-${NVIDIA_DRIVER_BRANCH}*"
    "nvidia-fabricmanager*"
)
[[ "$INCLUDE_NVIDIA_CONTAINER_TOOLKIT" == "1" ]] && _required_globs+=(
    "nvidia-container-toolkit_*"
    "libnvidia-container1_*"
    "libnvidia-container-tools_*"
)
_missing_required=()
for _g in "${_required_globs[@]}"; do
    if ! compgen -G "$OUT_DIR/debs/$_g" >/dev/null 2>&1; then
        _missing_required+=("$_g")
    fi
done
if (( ${#_missing_required[@]} > 0 )); then
    warn "Critical packages missing from bundle (${#_missing_required[@]}): ${_missing_required[*]}"
    warn "These were listed in APT_PKGS but didn't download. Most common cause:"
    warn "  - apt repository for the package wasn't configured on the gather host"
    warn "  - apt-get update failed before the download loop ran"
    warn "Cross-reference with meta/apt-closure-download-failed.txt"
    die "Refusing to package an incomplete bundle. Fix the gather host and re-run."
fi
shopt -u nullglob

# ── Prune known-conflicting debs that the apt closure pulls in transitively ──
# These pairs conflict at dpkg-install time on the target. Keep the one we
# actually want (matching APT_PKGS above) and delete the other.
_prune_globs=(
    # libcurl4-gnutls-dev conflicts with libcurl4-openssl-dev (in APT_PKGS).
    "libcurl4-gnutls-dev_*.deb"
    # pulseaudio (main package) and its bluetooth module both conflict with
    # pipewire-audio, which Ubuntu 24.04 XFCE desktops install by default.
    "pulseaudio_*.deb"
    "pulseaudio-module-bluetooth_*.deb"
    # systemd-standalone-sysusers conflicts with systemd-sysusers, which is
    # provided by the systemd package already on the target.
    "systemd-standalone-sysusers_*.deb"
    # Old CUDA 11.x / 12.x cudart debs collide with cuda-toolkit-config-common
    # 13.x on /etc/ld.so.conf.d/000_cuda.conf when CUDA_META_PKG=cuda-toolkit-13-0.
    "cuda-cudart-11-*_*.deb"
    "cuda-cudart-12-*_*.deb"
    # ── Mellanox DOCA / MLNX_OFED ─────────────────────────────────────────
    # On gather hosts with Mellanox repos enabled, apt-rdepends pulls in
    # doca-* packages that depend on ibverbs-providers 2601.0+. The target
    # has stock Ubuntu ibverbs-providers 50.0, which makes every doca-*
    # install fail unresolvably. Install MLNX_OFED separately on the target.
    "doca-*.deb"
    "ibverbs-utils_*.deb"
    "libibverbs-dev_*.deb"
    "perftest_*.deb"
    "rdma-core_*.deb"
    "infiniband-diags_*.deb"
    # ── GNOME flashback ──────────────────────────────────────────────────
    # Pulls nautilus / libgnome-panel3 which we don't include in the
    # closure (we only ship XFCE). Removing the parent stops apt from
    # trying to install it.
    "gnome-flashback*.deb"
    # ── ispell / aspell / hunspell spell-check chain ─────────────────────
    # dictionaries-common's post-install needs an ispell dictionary; we
    # don't bundle ienglish-common, so the post-install fails. aspell-en
    # and hunspell-en-us also Depends: dictionaries-common, so they fail
    # the same way if dictionaries-common isn't installable. None of
    # these are needed on a server, so drop the whole chain.
    "iamerican_*.deb"
    "ibritish_*.deb"
    "ispell_*.deb"
    "dictionaries-common_*.deb"
    "wamerican_*.deb"
    "wbritish_*.deb"
    "aspell_*.deb"
    "aspell-en_*.deb"
    "hunspell_*.deb"
    "hunspell-en-us_*.deb"
    "hunspell-en-*.deb"
    "libaspell*_*.deb"
    "libhunspell*_*.deb"
    # ── plymouth-label ───────────────────────────────────────────────────
    # Needs fonts-ubuntu, which isn't in our closure. Not needed.
    "plymouth-label_*.deb"
)
# If pinned to a specific NVIDIA driver branch, drop any *other* branch debs
# that snuck in via dependency expansion.
if [[ "$INCLUDE_NVIDIA_DRIVER" == "1" && -n "${NVIDIA_DRIVER_BRANCH:-}" ]]; then
    for f in "$OUT_DIR"/debs/{nvidia,libnvidia}-*-[0-9]*_*.deb; do
        [[ -e "$f" ]] || continue
        base=$(basename "$f")
        if [[ "$base" =~ -([0-9]+)(-open)?_ ]]; then
            br="${BASH_REMATCH[1]}"
            if [[ "$br" != "$NVIDIA_DRIVER_BRANCH" ]]; then
                log "Pruning off-branch NVIDIA deb: $base (branch $br, want $NVIDIA_DRIVER_BRANCH)"
                rm -f "$f"
            fi
        fi
    done
    # Drop unversioned driver meta-debs that pull in a newer branch on install.
    # nvidia-fabricmanager and libnvidia-nscq are intentionally left unversioned
    # (kept in the bundle) — they're data-center packages with server-side deps.
    for unv in nvidia-open nvidia-driver-open nvidia-driver nvidia-firmware \
               nvidia-kernel-common nvidia-kernel-source-open nvidia-dkms-open \
               libnvidia-cfg1 libnvidia-compute libnvidia-decode libnvidia-encode \
               libnvidia-fbc1 libnvidia-gl libnvidia-gpucomp; do
        for f in "$OUT_DIR"/debs/"${unv}"_*.deb; do
            [[ -e "$f" ]] || continue
            log "Pruning unversioned NVIDIA meta deb: $(basename "$f")"
            rm -f "$f"
        done
    done
    # Final-sweep: any remaining nvidia-*/libnvidia-* deb whose version's leading
    # integer doesn't match NVIDIA_DRIVER_BRANCH is off-branch — remove it.
    # Exceptions: fabricmanager and nscq are intentionally kept unversioned.
    for f in "$OUT_DIR"/debs/{nvidia,libnvidia}-*.deb; do
        [[ -e "$f" ]] || continue
        base=$(basename "$f")
        case "$base" in
            nvidia-fabricmanager_*|libnvidia-nscq_*) continue ;;
        esac
        # Version string is between first _ and second _ in the filename.
        if [[ "$base" =~ _([0-9]+)\. ]]; then
            ver_branch="${BASH_REMATCH[1]}"
            if [[ "$ver_branch" != "$NVIDIA_DRIVER_BRANCH" ]]; then
                log "Pruning NVIDIA deb with off-branch version ($ver_branch≠$NVIDIA_DRIVER_BRANCH): $base"
                rm -f "$f"
            fi
        fi
    done
fi
for pat in "${_prune_globs[@]}"; do
    for f in "$OUT_DIR"/debs/$pat; do
        [[ -e "$f" ]] || continue
        log "Pruning conflicting deb: $(basename "$f")"
        rm -f "$f"
    done
done

# ── Dedupe: when /var/cache/apt/archives contained multiple versions of the
#    same package (e.g. vim-runtime 7.12 AND 7.13 because the gather host did
#    an apt update mid-run), keep only the newest. Without this the target's
#    dpkg pass installs the older one first and the bundle's matching newer
#    counterpart fails with Breaks ("vim-runtime breaks vim-tiny").
declare -A _newest_ver _newest_path
shopt -s nullglob
for f in "$OUT_DIR"/debs/*.deb; do
    base=$(basename "$f")
    # Filename format: <pkg>_<version>_<arch>.deb. Splitting on '_' is safe
    # because debian package names don't contain underscores.
    pkg="${base%%_*}"
    rest="${base#${pkg}_}"
    ver="${rest%_*}"
    prev_ver="${_newest_ver[$pkg]:-}"
    if [[ -z "$prev_ver" ]]; then
        _newest_ver[$pkg]="$ver"
        _newest_path[$pkg]="$f"
    elif dpkg --compare-versions "$ver" gt "$prev_ver" 2>/dev/null; then
        log "Dedupe: drop older $pkg $prev_ver, keep $ver"
        rm -f "${_newest_path[$pkg]}"
        _newest_ver[$pkg]="$ver"
        _newest_path[$pkg]="$f"
    else
        log "Dedupe: drop older $pkg $ver, keep $prev_ver"
        rm -f "$f"
    fi
done
shopt -u nullglob
unset _newest_ver _newest_path

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

# ── VS Code ──────────────────────────────────────────────────────────────────
log "Downloading VS Code (.deb)"
curl -L --retry 3 --progress-bar -o "$OUT_DIR/apps/vscode.deb" "$VSCODE_URL"
log "VS Code: $(du -sh "$OUT_DIR/apps/vscode.deb" | cut -f1)"

# ── Google Chrome ─────────────────────────────────────────────────────────────
log "Downloading Google Chrome (.deb)"
curl -L --retry 3 --progress-bar -o "$OUT_DIR/apps/chrome.deb" "$CHROME_URL"
log "Chrome: $(du -sh "$OUT_DIR/apps/chrome.deb" | cut -f1)"

# ── Firefox (binary tarball from Mozilla — avoids snap dependency) ────────────
log "Resolving Firefox version"
if [[ "$FIREFOX_VER" == "latest" ]]; then
    FIREFOX_VER=$(curl -sI "https://download.mozilla.org/?product=firefox-latest-ssl&os=linux64&lang=${FIREFOX_LANG}" \
        | grep -i '^location:' \
        | grep -oP 'releases/\K[^/]+' \
        | head -1)
    [[ -n "$FIREFOX_VER" ]] || die "Could not resolve latest Firefox version."
fi
log "Downloading Firefox $FIREFOX_VER"
# Mozilla replaced the linux .tar.bz2 tarball with .tar.xz starting in Firefox 135
# (early 2025). Try xz first; fall back to bz2 only for older versions. Without
# -f, curl writes the 404 page to the output file and the installer later fails
# trying to bunzip-extract HTML — which is exactly what install-all.log shows.
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
[[ -n "$_ff_format" ]] || die "Could not download Firefox $FIREFOX_VER from Mozilla CDN (tried .tar.xz and .tar.bz2)."
# Drop any stale alternate-format copy from a previous gather run.
case "$_ff_format" in
    tar.xz)  rm -f "$OUT_DIR/apps/firefox.tar.bz2" ;;
    tar.bz2) rm -f "$OUT_DIR/apps/firefox.tar.xz" ;;
esac
echo "$FIREFOX_VER" > "$OUT_DIR/apps/firefox.version"
log "Firefox: $(du -sh "$OUT_DIR/apps/firefox.${_ff_format}" | cut -f1) (.${_ff_format})"

# ============================================================================
# 3) OPENCODE CLI
# ============================================================================
step "Opencode CLI"

log "Resolving Opencode release ($OPENCODE_VER)"
if [[ "$OPENCODE_VER" == "latest" ]]; then
    # GitHub repo moved — must follow redirect with -L
    OPENCODE_TAG=$(curl -sL "https://api.github.com/repos/sst/opencode/releases/latest" \
        | grep '"tag_name"' | grep -oP '(?<="tag_name": ")[^"]+')
    if [[ -z "$OPENCODE_TAG" ]]; then
        warn "Could not resolve latest Opencode release (GitHub API rate limit?). Skipping."
        OPENCODE_TAG=""
    fi
else
    OPENCODE_TAG="$OPENCODE_VER"
fi

if [[ -n "$OPENCODE_TAG" ]]; then
    log "Opencode tag: $OPENCODE_TAG"
    _oc_base="https://github.com/sst/opencode/releases/download/${OPENCODE_TAG}"
    # Linux x64 CLI is shipped as a tarball; prefer glibc build, fall back to musl
    _oc_downloaded=0
    for asset in "opencode-linux-x64.tar.gz" "opencode-linux-x64-musl.tar.gz"; do
        _url="${_oc_base}/${asset}"
        _tmp="$OUT_DIR/apps/_opencode_tmp.tar.gz"
        if curl -fsSL --retry 2 -o "$_tmp" "$_url" 2>/dev/null; then
            tar -xzf "$_tmp" -C "$OUT_DIR/apps/" 2>/dev/null || true
            # Binary is named 'opencode' inside the tarball
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
        warn "Could not download Opencode binary. Place at: $OUT_DIR/apps/opencode"
        echo "PLACEHOLDER" > "$OUT_DIR/apps/opencode.MISSING"
    fi
    echo "$OPENCODE_TAG" > "$OUT_DIR/apps/opencode.version"
else
    echo "SKIPPED" > "$OUT_DIR/apps/opencode.MISSING"
fi

# ============================================================================
# 4) NODE.JS + NPM
# ============================================================================
step "Node.js LTS + npm"

log "Resolving Node.js v${NODE_LTS_MAJOR} LTS version"
NODE_VER=$(curl -sL "https://nodejs.org/dist/latest-v${NODE_LTS_MAJOR}.x/" \
    | grep -oP "node-v\K[\d.]+(?=-linux-x64\.tar\.xz)" \
    | head -1)
[[ -n "$NODE_VER" ]] || die "Could not resolve Node.js v${NODE_LTS_MAJOR} LTS version."
log "Downloading Node.js v${NODE_VER} (includes npm)"
NODE_URL="https://nodejs.org/dist/v${NODE_VER}/node-v${NODE_VER}-linux-x64.tar.xz"
curl -L --retry 3 --progress-bar -o "$OUT_DIR/apps/nodejs.tar.xz" "$NODE_URL"
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
# 6) PYTHON WHEELS — LLM Inference
#    torch 2.11.0+cu130 — current stable vLLM dependency line
#    Includes: vLLM, LLM_API_fast, RAG, llama.cpp Python utils
# ============================================================================
step "Python wheels: LLM Inference (torch ${TORCH_VER_INFERENCE}+${TORCH_CUDA_TAG})"

VENV_INF="$(mktemp -d)/venv"
"$PYTHON_BIN" -m venv "$VENV_INF"
# shellcheck disable=SC1091
source "$VENV_INF/bin/activate"
pip install --upgrade pip wheel setuptools
pip download --dest "$OUT_DIR/wheels/inference" pip wheel setuptools

log "Downloading torch==${TORCH_VER_INFERENCE}+${TORCH_CUDA_TAG} for inference venv"
if ! pip download --dest "$OUT_DIR/wheels/inference" --index-url "$TORCH_INDEX" "torch==${TORCH_VER_INFERENCE}" torchvision torchaudio; then
    die "Failed to download torch==${TORCH_VER_INFERENCE} from $TORCH_INDEX"
fi

_vllm_pkg="vllm"; [[ -n "$VLLM_VER" ]] && _vllm_pkg="vllm==${VLLM_VER}"
if [[ "$VLLM_CUDA_WHEEL" == "1" && "$TORCH_CUDA_TAG" =~ ^cu[0-9]+$ ]]; then
    _vllm_version="${VLLM_VER#v}"
    if [[ -z "$_vllm_version" ]]; then
        _vllm_tag=$(curl -fsSL --retry 3 "https://api.github.com/repos/vllm-project/vllm/releases/latest" \
            | grep '"tag_name"' | grep -oP '(?<="tag_name": ")[^"]+' || true)
        _vllm_version="${_vllm_tag#v}"
    fi
    [[ -n "$_vllm_version" ]] || die "Could not resolve latest vLLM release."

    _vllm_cuda="${TORCH_CUDA_TAG#cu}"
    _vllm_arch="$(uname -m)"
    _vllm_url="https://github.com/vllm-project/vllm/releases/download/v${_vllm_version}/vllm-${_vllm_version}+cu${_vllm_cuda}-cp38-abi3-manylinux_2_35_${_vllm_arch}.whl"
    log "Downloading vLLM ${_vllm_version}+cu${_vllm_cuda} release wheel (>5 GB)"
    if ! pip download \
        --dest "$OUT_DIR/wheels/inference" \
        --extra-index-url "$TORCH_INDEX" \
        "$_vllm_url"; then
        if [[ "$REQUIRE_VLLM_CUDA_WHEEL" == "1" ]]; then
            die "Failed to download CUDA-specific vLLM wheel: $_vllm_url"
        fi
        warn "CUDA-specific vLLM wheel failed; falling back to $_vllm_pkg from PyPI."
        pip download \
            --dest "$OUT_DIR/wheels/inference" \
            --extra-index-url "$TORCH_INDEX" \
            "$_vllm_pkg" \
            || warn "vLLM download failed — check network."
    fi
else
    log "Downloading $_vllm_pkg (>5 GB)"
    pip download \
        --dest "$OUT_DIR/wheels/inference" \
        --extra-index-url "$TORCH_INDEX" \
        "$_vllm_pkg" \
        || warn "vLLM download failed — check network."
fi

if [[ -n "$LLMAPI_REQ" && -f "$LLMAPI_REQ" ]]; then
    cp "$LLMAPI_REQ" "$OUT_DIR/requirements/llm_api.txt"
    pip download --dest "$OUT_DIR/wheels/inference" -r "$OUT_DIR/requirements/llm_api.txt" \
        || warn "Some LLM_API packages failed; continuing."
fi
if [[ -n "$LLMAPI_FULL_REQ" && -f "$LLMAPI_FULL_REQ" ]]; then
    cp "$LLMAPI_FULL_REQ" "$OUT_DIR/requirements/llm_api_full.txt"
    pip download --dest "$OUT_DIR/wheels/inference" -r "$OUT_DIR/requirements/llm_api_full.txt" \
        || warn "Some LLM_API_full packages failed; continuing."
fi

log "Downloading core inference/RAG packages"
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
    || warn "Some inference packages failed; continuing."

deactivate
rm -rf "$(dirname "$VENV_INF")"
log "Inference wheels: $(ls "$OUT_DIR/wheels/inference" | wc -l) files ($(du -sh "$OUT_DIR/wheels/inference" | cut -f1))"

# ============================================================================
# 7) PYTHON WHEELS — General Training
#    torch 2.11.0+cu130 — latest, best B300/H200/L40s support
#    Includes: PyG, MeshGraphNets, SimulGenVAE, PEMTRON
# ============================================================================
step "Python wheels: General Training (torch ${TORCH_VER_TRAINING}+${TORCH_CUDA_TAG})"

PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VER_TRAINING}+${TORCH_CUDA_TAG}.html"
log "Torch index  : $TORCH_INDEX"
log "Torch version: $TORCH_VER_TRAINING"
log "PyG index    : $PYG_INDEX"

VENV_TRAIN="$(mktemp -d)/venv"
"$PYTHON_BIN" -m venv "$VENV_TRAIN"
# shellcheck disable=SC1091
source "$VENV_TRAIN/bin/activate"
pip install --upgrade pip wheel setuptools
pip download --dest "$OUT_DIR/wheels/training" pip wheel setuptools

log "Downloading torch==${TORCH_VER_TRAINING}+${TORCH_CUDA_TAG} + torchvision + torchaudio"
if ! pip download --dest "$OUT_DIR/wheels/training" --index-url "$TORCH_INDEX" "torch==${TORCH_VER_TRAINING}" torchvision torchaudio; then
    die "Failed to download torch==${TORCH_VER_TRAINING} from $TORCH_INDEX"
fi

log "Downloading torch-geometric"
pip download --dest "$OUT_DIR/wheels/training" torch-geometric

log "Downloading PyG extensions (pyg_lib, scatter, sparse, cluster)"
pip download \
    --dest "$OUT_DIR/wheels/training" \
    --find-links "$PYG_INDEX" \
    pyg_lib torch-scatter torch-sparse torch-cluster \
    || die "PyG extensions failed — check $PYG_INDEX"

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
    grep -vE '^\s*#|^\s*$|^torch|^torchvision|^torchaudio|pyreadline3|langchain-classic|xlwt|aider-chat|pyinstaller|llama-cpp-python|pip-system-certs' "$rf" \
        | pip download --dest "$OUT_DIR/wheels/training" -r /dev/stdin \
        || warn "Some packages from $(basename "$rf") failed; continuing."
done

log "Downloading core training/scientific stack"
pip download --dest "$OUT_DIR/wheels/training" \
    numpy scipy h5py pandas tqdm matplotlib seaborn Pillow pyvista \
    scikit-learn scikit-image statsmodels networkx sympy \
    torchinfo tensorboard pytorch-warmup \
    opencv-python imageio librosa audiomentations soxr natsort \
    reportlab PyQt5 paramiko smbprotocol \
    || warn "Some training packages failed; continuing."

deactivate
rm -rf "$(dirname "$VENV_TRAIN")"
log "Training wheels: $(ls "$OUT_DIR/wheels/training" | wc -l) files ($(du -sh "$OUT_DIR/wheels/training" | cut -f1))"

# ============================================================================
# 8) LLAMA.CPP — source + Python wheels
# ============================================================================
step "llama.cpp"

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

log "Downloading Python wheels for llama.cpp convert/utility scripts"
REQ_DIR="$(mktemp -d)"
tar -xzf "$OUT_DIR/src/llama.cpp.tar.gz" -C "$REQ_DIR"
REQ_ROOT="$REQ_DIR/llama.cpp"

LLAMA_REQ_FILES=()
[[ -f "$REQ_ROOT/requirements.txt" ]] && LLAMA_REQ_FILES+=( "$REQ_ROOT/requirements.txt" )
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
            || warn "pip download failed for ${rf##*/}; continuing."
    done
    mkdir -p "$OUT_DIR/meta/requirements/llamacpp"
    cp "${LLAMA_REQ_FILES[@]}" "$OUT_DIR/meta/requirements/llamacpp/" 2>/dev/null || true
    deactivate
else
    warn "No requirements files found in llama.cpp source; skipping wheel download."
fi
rm -rf "$REQ_DIR"
log "llama.cpp: source archived, $(ls "$OUT_DIR/wheels/llamacpp" 2>/dev/null | wc -l) wheels"

# ============================================================================
# 8.5) NCCL-TESTS — source archive (built on target)
# ============================================================================
if [[ "$INCLUDE_NCCL_TESTS" == "1" ]]; then
    step "nccl-tests"
    log "Cloning $NCCL_TESTS_REPO @ $NCCL_TESTS_REF"
    if git clone "$NCCL_TESTS_REPO" "$OUT_DIR/src/nccl-tests" 2>/dev/null; then
        git -C "$OUT_DIR/src/nccl-tests" checkout "$NCCL_TESTS_REF" 2>/dev/null || true
        NCCL_TESTS_COMMIT=$(git -C "$OUT_DIR/src/nccl-tests" rev-parse HEAD 2>/dev/null || echo unknown)
        log "nccl-tests at commit $NCCL_TESTS_COMMIT"
        tar --exclude='.git' -C "$OUT_DIR/src" -czf "$OUT_DIR/src/nccl-tests.tar.gz" nccl-tests
        rm -rf "$OUT_DIR/src/nccl-tests"
        echo "BUNDLE_NCCL_TESTS_COMMIT=$NCCL_TESTS_COMMIT" >> "$OUT_DIR/meta/target.env"
        log "nccl-tests source archived"
    else
        warn "Could not clone nccl-tests; multi-GPU bandwidth benchmarks won't be bundled."
    fi
fi


# ============================================================================
# 11) PYTHON WHEELS — Jupyter + data science
# ============================================================================
step "Python wheels: Jupyter + data science"

if [[ "$INCLUDE_JUPYTER" == "1" ]]; then
    VENV_JUPYTER="$(mktemp -d)/venv"
    "$PYTHON_BIN" -m venv "$VENV_JUPYTER"
    # shellcheck disable=SC1091
    source "$VENV_JUPYTER/bin/activate"
    pip install --upgrade pip wheel setuptools
    pip download --dest "$OUT_DIR/wheels/jupyter" pip wheel setuptools

    log "Downloading Jupyter + data science wheels — this may take several minutes"
    pip download --dest "$OUT_DIR/wheels/jupyter" \
        jupyterlab \
        notebook \
        ipykernel \
        ipywidgets \
        jupyter-server \
        jupyter-collaboration \
        pandas \
        polars \
        numpy \
        scipy \
        matplotlib \
        seaborn \
        plotly \
        scikit-learn \
        statsmodels \
        tqdm \
        rich \
        requests \
        aiohttp \
        black \
        ruff \
        mypy \
        pytest \
        ipdb \
        || warn "Some Jupyter packages failed to download; continuing."

    deactivate
    rm -rf "$(dirname "$VENV_JUPYTER")"
    log "Jupyter wheels: $(ls "$OUT_DIR/wheels/jupyter" | wc -l) files ($(du -sh "$OUT_DIR/wheels/jupyter" | cut -f1))"
else
    log "INCLUDE_JUPYTER=0; skipping Jupyter wheels."
fi

# ============================================================================
# 12) K3s BINARIES + AIRGAP IMAGES
# ============================================================================
step "K3s binaries + airgap images"

if [[ "$INCLUDE_K3S" == "1" ]]; then
    K3S_URL_BASE="https://github.com/k3s-io/k3s/releases/download/${K3S_VER}"

    log "Downloading k3s ${K3S_VER} binary"
    curl -L --retry 3 --progress-bar \
        -o "$OUT_DIR/k3s/bin/k3s" \
        "${K3S_URL_BASE}/k3s"
    chmod +x "$OUT_DIR/k3s/bin/k3s"

    log "Downloading k3s airgap images (amd64)"
    if curl -fsSL --retry 2 -I "${K3S_URL_BASE}/k3s-airgap-images-amd64.tar.zst" >/dev/null 2>&1; then
        curl -L --retry 3 --progress-bar \
            -o "$OUT_DIR/k3s/airgap-images/k3s-airgap-images-amd64.tar.zst" \
            "${K3S_URL_BASE}/k3s-airgap-images-amd64.tar.zst"
    else
        curl -L --retry 3 --progress-bar \
            -o "$OUT_DIR/k3s/airgap-images/k3s-airgap-images-amd64.tar.gz" \
            "${K3S_URL_BASE}/k3s-airgap-images-amd64.tar.gz"
    fi

    log "Downloading k3s install script (get.k3s.io)"
    curl -sfL --retry 3 \
        -o "$OUT_DIR/k3s/bin/k3s-install.sh" \
        "https://get.k3s.io"
    chmod +x "$OUT_DIR/k3s/bin/k3s-install.sh"

    log "Downloading Helm ${HELM_VER}"
    curl -L --retry 3 --progress-bar \
        -o /tmp/helm.tar.gz \
        "https://get.helm.sh/helm-${HELM_VER}-linux-amd64.tar.gz"
    tar -xzf /tmp/helm.tar.gz -C /tmp linux-amd64/helm
    mv /tmp/linux-amd64/helm "$OUT_DIR/k3s/bin/helm"
    chmod +x "$OUT_DIR/k3s/bin/helm"
    rm -f /tmp/helm.tar.gz

    log "Downloading kubectl v${KUBECTL_VER}"
    curl -L --retry 3 --progress-bar \
        -o "$OUT_DIR/k3s/bin/kubectl" \
        "https://dl.k8s.io/release/v${KUBECTL_VER}/bin/linux/amd64/kubectl"
    chmod +x "$OUT_DIR/k3s/bin/kubectl"

    cat > "$OUT_DIR/k3s/meta/versions.env" <<EOF
K3S_VER=${K3S_VER}
HELM_VER=${HELM_VER}
KUBECTL_VER=${KUBECTL_VER}
GPU_OPERATOR_CHART_VER=${GPU_OPERATOR_CHART_VER}
KUBE_PROM_STACK_CHART_VER=${KUBE_PROM_STACK_CHART_VER}
LOKI_STACK_CHART_VER=${LOKI_STACK_CHART_VER}
KUBERAY_CHART_VER=${KUBERAY_CHART_VER}
REGISTRY_IMAGE=${REGISTRY_IMAGE}
VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG}
RAY_IMAGE_TAG=${RAY_IMAGE_TAG}
PYTORCH_IMAGE_TAG=${PYTORCH_IMAGE_TAG}
INCLUDE_K3S_EXAMPLE_IMAGES=${INCLUDE_K3S_EXAMPLE_IMAGES}
EOF

    log "K3s binaries ready: $(ls "$OUT_DIR/k3s/bin" | tr '\n' ' ')"
else
    log "INCLUDE_K3S=0; skipping K3s binaries."
fi

# ============================================================================
# 13) HELM CHARTS
# ============================================================================
step "Helm charts"

if [[ "$INCLUDE_K3S" == "1" ]]; then
    HELM_BIN="$OUT_DIR/k3s/bin/helm"
    [[ -x "$HELM_BIN" ]] || die "Helm binary not found in k3s/bin/ — section 12 must succeed first."

    log "Adding Helm repos"
    "$HELM_BIN" repo add nvidia               https://helm.ngc.nvidia.com/nvidia              2>/dev/null || true
    "$HELM_BIN" repo add prometheus-community  https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    "$HELM_BIN" repo add grafana              https://grafana.github.io/helm-charts            2>/dev/null || true
    "$HELM_BIN" repo add kuberay              https://ray-project.github.io/kuberay-helm/      2>/dev/null || true
    "$HELM_BIN" repo update

    log "Pulling GPU Operator chart ${GPU_OPERATOR_CHART_VER}"
    "$HELM_BIN" pull nvidia/gpu-operator \
        --version "$GPU_OPERATOR_CHART_VER" \
        -d "$OUT_DIR/k3s/charts/"

    log "Pulling kube-prometheus-stack chart ${KUBE_PROM_STACK_CHART_VER}"
    "$HELM_BIN" pull prometheus-community/kube-prometheus-stack \
        --version "$KUBE_PROM_STACK_CHART_VER" \
        -d "$OUT_DIR/k3s/charts/"

    log "Pulling loki-stack chart ${LOKI_STACK_CHART_VER}"
    "$HELM_BIN" pull grafana/loki-stack \
        --version "$LOKI_STACK_CHART_VER" \
        -d "$OUT_DIR/k3s/charts/"

    log "Pulling kuberay-operator chart ${KUBERAY_CHART_VER}"
    "$HELM_BIN" pull kuberay/kuberay-operator \
        --version "$KUBERAY_CHART_VER" \
        -d "$OUT_DIR/k3s/charts/"

    log "Charts: $(ls "$OUT_DIR/k3s/charts/")"

    # Build an image manifest by rendering each chart and extracting image refs.
    # These images will be pre-pulled and pushed to the airgap registry on install.
    IMAGE_MANIFEST="$OUT_DIR/k3s/meta/images-manifest.txt"
    : > "$IMAGE_MANIFEST"
    for chart_tgz in "$OUT_DIR/k3s/charts/"*.tgz; do
        log "  Scanning images in $(basename "$chart_tgz")"
        "$HELM_BIN" template tmp-scan "$chart_tgz" 2>/dev/null \
            | grep -oP '(?<=image: )["\x27]?\K[^\s"\x27]+' \
            | grep '\.' \
            >> "$IMAGE_MANIFEST" || true
    done

    # Standalone images required by the local registry and optional example
    # workloads. K3s itself is CUDA-agnostic; do not seed CUDA-mismatched
    # workload images by default.
    cat >> "$IMAGE_MANIFEST" <<EOF
docker.io/library/registry:2.8.3
EOF

    if [[ "$INCLUDE_K3S_EXAMPLE_IMAGES" == "1" ]]; then
        [[ -n "$VLLM_IMAGE_TAG" ]] && echo "docker.io/vllm/vllm-openai:${VLLM_IMAGE_TAG}" >> "$IMAGE_MANIFEST"
        [[ -n "$RAY_IMAGE_TAG" ]] && echo "docker.io/rayproject/ray:${RAY_IMAGE_TAG}" >> "$IMAGE_MANIFEST"
        [[ -n "$PYTORCH_IMAGE_TAG" ]] && echo "docker.io/pytorch/pytorch:${PYTORCH_IMAGE_TAG}" >> "$IMAGE_MANIFEST"
    fi

    sort -u "$IMAGE_MANIFEST" -o "$IMAGE_MANIFEST"
    log "Image manifest: $(wc -l < "$IMAGE_MANIFEST") unique images -> $IMAGE_MANIFEST"
else
    log "INCLUDE_K3S=0; skipping Helm charts."
fi

# ============================================================================
# 14) CONTAINER IMAGES (via skopeo)
# ============================================================================
step "Container images"

if [[ "$INCLUDE_K3S" == "1" ]]; then
    IMAGE_MANIFEST="$OUT_DIR/k3s/meta/images-manifest.txt"
    [[ -f "$IMAGE_MANIFEST" ]] || die "images-manifest.txt not found — section 13 must succeed first."

    if ! command -v skopeo >/dev/null 2>&1; then
        log "skopeo not found — attempting install on WSL"
        sudo apt-get install -y skopeo 2>/dev/null \
            || warn "Could not install skopeo automatically. Run: sudo apt-get install skopeo"
    fi

    if command -v skopeo >/dev/null 2>&1; then
        TOTAL=$(wc -l < "$IMAGE_MANIFEST")
        COUNT=0
        FAILED=0
        while IFS= read -r img; do
            [[ -z "$img" ]] && continue
            COUNT=$(( COUNT + 1 ))
            # Filesystem-safe filename
            safe="${img//\//_}"; safe="${safe//:/_}"; safe="${safe//@/_}"
            out="$OUT_DIR/k3s/images/${safe}.tar"
            if [[ -f "$out" ]]; then
                log "[$COUNT/$TOTAL] Cached: $img"
                continue
            fi
            log "[$COUNT/$TOTAL] Saving $img"
            if skopeo copy \
                --override-os linux \
                --override-arch amd64 \
                "docker://${img}" \
                "docker-archive:${out}" 2>/dev/null; then
                :
            else
                warn "  Failed: $img (may need auth, or not yet published)"
                FAILED=$(( FAILED + 1 ))
            fi
        done < "$IMAGE_MANIFEST"

        ( cd "$OUT_DIR/k3s" && find images -name '*.tar' -print0 \
            | xargs -0 sha256sum 2>/dev/null > meta/SHA256SUMS-images )

        log "Images: $COUNT processed, $FAILED failed ($(du -sh "$OUT_DIR/k3s/images" | cut -f1))"
    else
        warn "skopeo unavailable — container images not pre-pulled."
        warn "Install skopeo and re-run: INCLUDE_K3S=1 bash gather-all.sh"
    fi
else
    log "INCLUDE_K3S=0; skipping container images."
fi

# ============================================================================
# 15) K3s MANIFESTS + TEMPLATES
# ============================================================================
step "K3s manifests and templates"

if [[ "$INCLUDE_K3S" == "1" ]]; then
    # Copy k3s-manifests tree if it exists alongside gather-all.sh
    if [[ -d "$SCRIPT_DIR/k3s-manifests" ]]; then
        cp -r "$SCRIPT_DIR/k3s-manifests/." "$OUT_DIR/k3s/manifests/"
        log "k3s-manifests/ copied: $(find "$OUT_DIR/k3s/manifests" -type f | wc -l) files"
    else
        warn "k3s-manifests/ not found next to gather-all.sh — manifests not bundled."
        warn "Create $SCRIPT_DIR/k3s-manifests/ (see plan docs) or apply manifests manually."
    fi

    # Copy docs tree (agent runbook lives here)
    if [[ -d "$SCRIPT_DIR/docs" ]]; then
        mkdir -p "$OUT_DIR/docs"
        cp -r "$SCRIPT_DIR/docs/." "$OUT_DIR/docs/"
        log "docs/ copied: $(find "$OUT_DIR/docs" -type f | wc -l) files"
    else
        warn "docs/ not found next to gather-all.sh — agent runbook not bundled."
    fi

    # Embed registries.yaml template (install-all.sh fills in REGISTRY_HOST)
    cat > "$OUT_DIR/k3s/manifests/registries.yaml.tmpl" <<'REGTMPL'
# K3s containerd registry mirror config.
# install-all.sh replaces REGISTRY_HOST with the actual server-1 IP.
# Deploy to every node at: /etc/rancher/k3s/registries.yaml
mirrors:
  docker.io:
    endpoint:
      - "http://REGISTRY_HOST:5000"
  quay.io:
    endpoint:
      - "http://REGISTRY_HOST:5000"
  nvcr.io:
    endpoint:
      - "http://REGISTRY_HOST:5000"
  gcr.io:
    endpoint:
      - "http://REGISTRY_HOST:5000"
  ghcr.io:
    endpoint:
      - "http://REGISTRY_HOST:5000"
  registry.k8s.io:
    endpoint:
      - "http://REGISTRY_HOST:5000"
REGTMPL

    # Embed the airgap-registry Kubernetes manifest (deployed as a pod on server 1)
    cat > "$OUT_DIR/k3s/manifests/airgap-registry.yaml" <<'REGPOD'
apiVersion: v1
kind: Namespace
metadata:
  name: registry
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: airgap-registry
  namespace: registry
  labels:
    app: airgap-registry
spec:
  replicas: 1
  selector:
    matchLabels:
      app: airgap-registry
  template:
    metadata:
      labels:
        app: airgap-registry
    spec:
      nodeSelector:
        node-role.kubernetes.io/control-plane: "true"
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          operator: Exists
          effect: NoSchedule
      hostNetwork: true
      containers:
        - name: registry
          image: docker.io/library/registry:2.8.3
          imagePullPolicy: Never
          ports:
            - containerPort: 5000
              hostPort: 5000
          env:
            - name: REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY
              value: /var/lib/registry
          volumeMounts:
            - name: registry-data
              mountPath: /var/lib/registry
      volumes:
        - name: registry-data
          hostPath:
            path: /var/lib/registry
            type: DirectoryOrCreate
REGPOD

    log "Manifests and templates written to $OUT_DIR/k3s/manifests/"
else
    log "INCLUDE_K3S=0; skipping K3s manifests."
fi

# ============================================================================
# 10) CHECKSUMS + BUNDLE
# ============================================================================
step "Checksums and bundle"

log "Copying install-all.sh into bundle"
cp "$(dirname "${BASH_SOURCE[0]}")/install-all.sh" "$OUT_DIR/install-all.sh"
chmod +x "$OUT_DIR/install-all.sh"

log "Generating SHA256 manifest"
(
    cd "$OUT_DIR"
    checksum_roots=(install-all.sh debs apps wheels requirements meta)
    [[ -d k3s ]] && checksum_roots+=(k3s)
    [[ -d docs ]] && checksum_roots+=(docs)
    find "${checksum_roots[@]}" -type f \
        ! -path 'meta/SHA256SUMS' \
        ! -name 'SHA256SUMS-images' \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum > meta/SHA256SUMS
)

BUNDLE_BIN="$(dirname "$OUT_DIR")/all-airgap-bundle-ubuntu${TARGET_OS_VERSION}.bin"
log "Packing bundle -> $BUNDLE_BIN (this may take a while for the torch/image tars)"
rm -f "$BUNDLE_BIN"
tar -czf "$BUNDLE_BIN" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"

# Copy install-all.sh next to the bundle — user only needs to transfer two files
BUNDLE_PARENT="$(dirname "$OUT_DIR")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/install-all.sh" ]]; then
    cp "$SCRIPT_DIR/install-all.sh" "$BUNDLE_PARENT/install-all.sh"
    chmod +x "$BUNDLE_PARENT/install-all.sh"
    log "Companion installer: $BUNDLE_PARENT/install-all.sh"
fi

log "Done."
printf '\n'
printf '  Bundle    : %s (%s)\n' "$BUNDLE_BIN" "$(du -sh "$BUNDLE_BIN" | cut -f1)"
printf '  Installer : %s\n' "$BUNDLE_PARENT/install-all.sh"
printf '  Staging   : %s\n' "$OUT_DIR"
printf '\n'
printf 'Transfer to air-gapped server:\n'
printf '  scp "%s" "%s" user@SERVER:~\n' "$BUNDLE_BIN" "$BUNDLE_PARENT/install-all.sh"
printf '  ssh user@SERVER\n'
printf '  sudo bash install-all.sh   # auto-extracts the bundle\n'
printf '\n'
printf 'All components (Python envs, vLLM, llama.cpp, apps) are in the single bundle.\n'
