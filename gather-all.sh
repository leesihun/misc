#!/usr/bin/env bash
# ============================================================================
# gather-all.sh
#   Run on an internet-connected WSL Ubuntu 24.04 machine.
#   Downloads all packages needed for the air-gapped Ubuntu 24.04 + CUDA 13.1
#   server with 8x B300 GPUs.
#
#   Produces: ~/GPU_server_downloads/  +  all-airgap-bundle-ubuntu<OS>.tar.gz
#   Transfer the tarball to the air-gapped server, extract, run install-all.sh.
#
#   NOTE: WSL distro MUST match target OS (22.04 or 24.04) for .deb compat.
#
# Usage:
#   bash gather-all.sh                         # default settings
#   INCLUDE_K3S=1 bash gather-all.sh           # include K3s + container images
#   INCLUDE_NVIDIA_DRIVER=1 bash gather-all.sh # include NVIDIA driver debs
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
# Two torch versions — venvs are isolated so each uses the right one.
TORCH_VER_INFERENCE="${TORCH_VER_INFERENCE:-2.10.0}"   # vLLM 0.19.x pins this; do not bump
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
# Leave VLLM_VER empty for latest, or pin (e.g. "0.9.1")
VLLM_VER="${VLLM_VER:-}"

# llama.cpp — "master" always tracks the latest commit
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"

# CUDA toolkit (needed to BUILD llama.cpp with CUDA on the target).
# Requires the NVIDIA apt repo to be configured on the WSL machine first:
#   https://developer.nvidia.com/cuda-downloads (select Linux → Ubuntu → 24.04 → deb(network))
# Default OFF — enable with INCLUDE_CUDA_TOOLKIT=1
INCLUDE_CUDA_TOOLKIT="${INCLUDE_CUDA_TOOLKIT:-0}"
CUDA_META_PKG="${CUDA_META_PKG:-cuda-toolkit-13-0}"

# NVIDIA GPU driver (needed at RUNTIME for GPU workloads).
# Requires the NVIDIA apt repo to be configured on the WSL machine first.
# Default OFF — enable with INCLUDE_NVIDIA_DRIVER=1
# For B300 (Blackwell) GPUs you need driver >= 570.
INCLUDE_NVIDIA_DRIVER="${INCLUDE_NVIDIA_DRIVER:-0}"
NVIDIA_DRIVER_VER="${NVIDIA_DRIVER_VER:-570}"

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
VLLM_IMAGE_TAG="${VLLM_IMAGE_TAG:-v0.6.6.post1}"
RAY_IMAGE_TAG="${RAY_IMAGE_TAG:-2.40.0-py312-cu128}"

# Target OS version (auto-detected from WSL; used in the bundle filename so
# Ubuntu 22.04 and 24.04 bundles don't overwrite each other)
TARGET_OS_VERSION="${TARGET_OS_VERSION:-$(. /etc/os-release && echo "$VERSION_ID")}"

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
if (( ${#_need[@]} > 0 )); then
    log "Installing local prerequisites: ${_need[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${_need[@]}" || true
fi
"$PYTHON_BIN" -m venv --help &>/dev/null \
    || die "$PYTHON_BIN -m venv unavailable. Install python${PYTHON_VER}-venv."

# ── Setup output tree ────────────────────────────────────────────────────────
log "Output directory: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{debs,apps,src,wheels/inference,wheels/training,wheels/llamacpp,wheels/jupyter,requirements,meta}
mkdir -p "$OUT_DIR"/k3s/{bin,airgap-images,images,charts,manifests,systemd,meta}

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
    skopeo                   # OCI image copy — needed on server to push images to airgap registry

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
    APT_PKGS+=( "nvidia-driver-${NVIDIA_DRIVER_VER}" "nvidia-utils-${NVIDIA_DRIVER_VER}" )
    log "NVIDIA driver enabled: nvidia-driver-${NVIDIA_DRIVER_VER} (NVIDIA apt repo must be configured in WSL)"
fi

if [[ "$INCLUDE_CUDA_TOOLKIT" == "1" ]]; then
    APT_PKGS+=( "$CUDA_META_PKG" )
    log "CUDA toolkit enabled: $CUDA_META_PKG (NVIDIA apt repo must be configured in WSL)"
fi

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

shopt -s nullglob
debs=(/var/cache/apt/archives/*.deb)
(( ${#debs[@]} > 0 )) || die "No .deb files were downloaded from apt."
sudo cp "${debs[@]}" "$OUT_DIR/debs/"
sudo chown -R "$(id -u):$(id -g)" "$OUT_DIR/debs"
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
FIREFOX_URL="https://releases.mozilla.org/pub/firefox/releases/${FIREFOX_VER}/linux-x86_64/${FIREFOX_LANG}/firefox-${FIREFOX_VER}.tar.bz2"
curl -L --retry 3 --progress-bar -o "$OUT_DIR/apps/firefox.tar.bz2" "$FIREFOX_URL"
echo "$FIREFOX_VER" > "$OUT_DIR/apps/firefox.version"
log "Firefox: $(du -sh "$OUT_DIR/apps/firefox.tar.bz2" | cut -f1)"

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
#    torch 2.10.0+cu130 — pinned by vLLM 0.19.x
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
pip download \
    --dest "$OUT_DIR/wheels/inference" \
    --index-url "$TORCH_INDEX" \
    "torch==${TORCH_VER_INFERENCE}" torchvision \
    || die "Failed to download torch==${TORCH_VER_INFERENCE} from $TORCH_INDEX"

_vllm_pkg="vllm"; [[ -n "$VLLM_VER" ]] && _vllm_pkg="vllm==${VLLM_VER}"
log "Downloading $_vllm_pkg (>5 GB)"
pip download \
    --dest "$OUT_DIR/wheels/inference" \
    --extra-index-url "$TORCH_INDEX" \
    "$_vllm_pkg" \
    || warn "vLLM download failed — check network."

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
pip download \
    --dest "$OUT_DIR/wheels/training" \
    --index-url "$TORCH_INDEX" \
    "torch==${TORCH_VER_TRAINING}" torchvision torchaudio \
    || die "Failed to download torch==${TORCH_VER_TRAINING} from $TORCH_INDEX"

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

    # Standalone images referenced in example manifests but not in charts
    cat >> "$IMAGE_MANIFEST" <<EOF
docker.io/library/registry:2.8.3
docker.io/vllm/vllm-openai:${VLLM_IMAGE_TAG}
docker.io/rayproject/ray:${RAY_IMAGE_TAG}
docker.io/pytorch/pytorch:2.6.0-cuda12.8-cudnn9-runtime
EOF

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
( cd "$OUT_DIR" && find debs apps wheels requirements meta -type f -print0 2>/dev/null \
    | xargs -0 sha256sum > meta/SHA256SUMS )
# Append k3s and docs checksums (skips SHA256SUMS-images which is its own file)
[[ -d "$OUT_DIR/k3s" ]] && \
    ( cd "$OUT_DIR" && find k3s -type f -not -name 'SHA256SUMS-images' -print0 \
        | xargs -0 sha256sum >> meta/SHA256SUMS )
[[ -d "$OUT_DIR/docs" ]] && \
    ( cd "$OUT_DIR" && find docs -type f -print0 \
        | xargs -0 sha256sum >> meta/SHA256SUMS )

BUNDLE_TGZ="$(dirname "$OUT_DIR")/all-airgap-bundle-ubuntu${TARGET_OS_VERSION}.tar.gz"
log "Packing bundle -> $BUNDLE_TGZ (this may take a while for the torch/image tars)"
tar -czf "$BUNDLE_TGZ" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"

log "Done."
printf '\n'
printf '  Bundle  : %s (%s)\n' "$BUNDLE_TGZ" "$(du -sh "$BUNDLE_TGZ" | cut -f1)"
printf '  Staging : %s\n' "$OUT_DIR"
printf '\n'
printf 'Transfer to air-gapped server:\n'
printf '  scp "%s" user@SERVER:~\n' "$BUNDLE_TGZ"
printf '  ssh user@SERVER\n'
printf '  tar -xzf all-airgap-bundle-ubuntu%s.tar.gz\n' "${TARGET_OS_VERSION}"
printf '  cd GPU_server_downloads && sudo bash install-all.sh\n'
printf '\n'
printf 'All components (Python envs, vLLM, llama.cpp, apps) are in the single bundle.\n'
