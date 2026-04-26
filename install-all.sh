#!/usr/bin/env bash
# ============================================================================
# install-all.sh
#   Run on the AIR-GAPPED Ubuntu 24.04 server after extracting the bundle
#   created by gather-all.sh.
#
# Usage:
#   tar -xzf all-airgap-bundle-ubuntu<OS>.tar.gz
#   cd GPU_server_downloads
#   sudo bash install-all.sh              # system-wide apps (VS Code, Chrome, etc.)
#   bash install-all.sh                   # works too (will sudo internally)
#
# Optional overrides:
#   INSTALL_INFERENCE=0  bash install-all.sh   # skip LLM inference venv (vLLM/LLM_API_fast/RAG)
#   INSTALL_TRAINING=0   bash install-all.sh   # skip general training venv (PyG/MeshGraphNets)
#   INSTALL_JUPYTER=0    bash install-all.sh   # skip JupyterLab venv
#   INSTALL_DESKTOP=0    bash install-all.sh   # skip XFCE4/xrdp configuration
#   INFERENCE_PREFIX=/opt/llm_inference bash install-all.sh
#   TRAINING_PREFIX=/opt/general_training bash install-all.sh
#   INSTALL_K3S=1 K3S_ROLE=server bash install-all.sh   # bootstrap K3s control plane
#   INSTALL_K3S=1 K3S_ROLE=agent K3S_SERVER_IP=10.0.0.101 K3S_TOKEN_FILE=/tmp/k3s-join-token bash install-all.sh
# ============================================================================
set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_VER="${PYTHON_VER:-3.12}"
PYTHON_BIN="${PYTHON_BIN:-python${PYTHON_VER}}"
INFERENCE_PREFIX="${INFERENCE_PREFIX:-$HOME/llm_inference}"
TRAINING_PREFIX="${TRAINING_PREFIX:-$HOME/general_training}"
LLAMA_PREFIX="${LLAMA_PREFIX:-$HOME/llama.cpp}"
INSTALL_INFERENCE="${INSTALL_INFERENCE:-1}"
INSTALL_TRAINING="${INSTALL_TRAINING:-1}"
INSTALL_LLAMA="${INSTALL_LLAMA:-1}"
INSTALL_DESKTOP="${INSTALL_DESKTOP:-1}"
BUILD_CUDA="${BUILD_CUDA:-1}"
BUILD_BLAS="${BUILD_BLAS:-1}"
JOBS="${JOBS:-$(nproc)}"
VERIFY_CHECKSUMS="${VERIFY_CHECKSUMS:-1}"
INSTALL_EXTRA="${INSTALL_EXTRA:-1}"
EXTRA_PREFIX="${EXTRA_PREFIX:-$HOME/extra}"
INSTALL_JUPYTER="${INSTALL_JUPYTER:-1}"
JUPYTER_PREFIX="${JUPYTER_PREFIX:-$HOME/jupyter}"
INSTALL_K3S="${INSTALL_K3S:-0}"
K3S_ROLE="${K3S_ROLE:-none}"           # server | agent | none
K3S_SERVER_IP="${K3S_SERVER_IP:-}"     # required when K3S_ROLE=agent
K3S_TOKEN_FILE="${K3S_TOKEN_FILE:-}"   # path to join-token file
K3S_REGISTRY_PORT="${K3S_REGISTRY_PORT:-5000}"
GPU_OPERATOR_DRIVER_ENABLED="${GPU_OPERATOR_DRIVER_ENABLED:-false}"
INSTALL_KUBERAY="${INSTALL_KUBERAY:-1}"

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install:WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;35m══ %s ══\033[0m\n' "$*"; }

[[ -d "$BUNDLE_DIR/debs" && -d "$BUNDLE_DIR/apps" ]] \
    || die "Bundle not found under $BUNDLE_DIR (expected debs/ and apps/ subdirs)"

# ============================================================================
# 0) Sanity checks
# ============================================================================
step "Sanity checks"

if [[ -f "$BUNDLE_DIR/meta/target.env" ]]; then
    # shellcheck disable=SC1091
    source "$BUNDLE_DIR/meta/target.env"
    source /etc/os-release
    HERE_ARCH=$(dpkg --print-architecture 2>/dev/null || uname -m)
    log "Bundle built on : $BUNDLE_OS_ID $BUNDLE_OS_VERSION / $BUNDLE_ARCH / py$BUNDLE_PYTHON"
    log "This host       : $ID $VERSION_ID / $HERE_ARCH"
    [[ "$ID" == "$BUNDLE_OS_ID" && "$VERSION_ID" == "$BUNDLE_OS_VERSION" ]] \
        || warn "OS mismatch — .deb installation may fail. Abort with Ctrl-C if unsure."
    [[ "$HERE_ARCH" == "$BUNDLE_ARCH" ]] \
        || die "Architecture mismatch ($HERE_ARCH vs $BUNDLE_ARCH). Bundle is incompatible."
fi

if [[ "$VERIFY_CHECKSUMS" == "1" && -f "$BUNDLE_DIR/meta/SHA256SUMS" ]]; then
    log "Verifying SHA256 checksums"
    ( cd "$BUNDLE_DIR" && sha256sum --quiet -c meta/SHA256SUMS ) \
        || die "Checksum verification failed — bundle may be corrupted."
fi

# ============================================================================
# 1) APT PACKAGES
# ============================================================================
step "APT packages"

shopt -s nullglob
all_debs=( "$BUNDLE_DIR"/debs/*.deb )
(( ${#all_debs[@]} > 0 )) || die "No .deb files found in $BUNDLE_DIR/debs/"
log "Installing ${#all_debs[@]} .deb packages (multi-pass for dependency ordering)"

for pass in 1 2 3; do
    log "dpkg pass $pass"
    sudo dpkg -i --force-depends "${all_debs[@]}" 2>&1 \
        | grep -v '^\(Reading\|Selecting\|Preparing\|Unpacking\|Setting up\|Processing\)' || true
    broken=$(dpkg -l | awk '/^.[HUF]/ {print $2}' | wc -l)
    (( broken == 0 )) && break
    log "  $broken packages in broken state, retrying..."
done

# Fix up remaining dependency issues using only bundled debs
sudo apt-get -f install -y --no-download --ignore-missing \
    -o Dir::Cache::archives="$BUNDLE_DIR/debs" 2>/dev/null || true

broken_final=$(dpkg -l | awk '/^.[HUF]/ {print $2}' | wc -l)
(( broken_final == 0 )) || warn "$broken_final packages still broken. Check: dpkg -l | grep -E '^..H|^..U|^..F'"

for bin in python3 pip3; do
    command -v "$bin" >/dev/null || warn "Expected '$bin' not found after apt install."
done

# Check python3.12 specifically
if command -v "python${PYTHON_VER}" >/dev/null; then
    log "Python $PYTHON_VER OK: $(python${PYTHON_VER} --version)"
else
    warn "python${PYTHON_VER} not found after install. You may need to run: sudo apt install python${PYTHON_VER}"
fi

# ============================================================================
# 2) VS CODE
# ============================================================================
step "VS Code"

if [[ -f "$BUNDLE_DIR/apps/vscode.deb" ]]; then
    log "Installing VS Code"
    sudo dpkg -i "$BUNDLE_DIR/apps/vscode.deb" || \
        sudo apt-get -f install -y --no-download 2>/dev/null || true
    command -v code >/dev/null && log "VS Code: $(code --version | head -1)" \
        || warn "VS Code installed but 'code' not in PATH (may need re-login or GUI launch)."
else
    warn "apps/vscode.deb not found; skipping."
fi

# ============================================================================
# 3) GOOGLE CHROME
# ============================================================================
step "Google Chrome"

if [[ -f "$BUNDLE_DIR/apps/chrome.deb" ]]; then
    log "Installing Google Chrome"
    sudo dpkg -i "$BUNDLE_DIR/apps/chrome.deb" || \
        sudo apt-get -f install -y --no-download 2>/dev/null || true
    command -v google-chrome-stable >/dev/null \
        && log "Chrome: $(google-chrome-stable --version)" \
        || warn "Chrome installed but binary not found in PATH."
else
    warn "apps/chrome.deb not found; skipping."
fi

# ============================================================================
# 4) FIREFOX
# ============================================================================
step "Firefox"

if [[ -f "$BUNDLE_DIR/apps/firefox.tar.bz2" ]]; then
    FIREFOX_VER_FILE="$BUNDLE_DIR/apps/firefox.version"
    FF_VER=$(cat "$FIREFOX_VER_FILE" 2>/dev/null || echo "unknown")
    log "Installing Firefox $FF_VER to /opt/firefox"
    sudo mkdir -p /opt/firefox
    sudo tar -xjf "$BUNDLE_DIR/apps/firefox.tar.bz2" -C /opt/firefox --strip-components=1
    # Create launcher symlink
    sudo ln -sf /opt/firefox/firefox /usr/local/bin/firefox
    log "Firefox: $(/opt/firefox/firefox --version 2>/dev/null || echo 'installed')"

    # Create desktop entry
    sudo tee /usr/share/applications/firefox-manual.desktop > /dev/null <<'EOF'
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
    log "Firefox desktop entry created."
else
    warn "apps/firefox.tar.bz2 not found; skipping."
fi

# ============================================================================
# 5) OPENCODE
# ============================================================================
step "Opencode"

if [[ -f "$BUNDLE_DIR/apps/opencode" ]]; then
    OC_VER=$(cat "$BUNDLE_DIR/apps/opencode.version" 2>/dev/null || echo "unknown")
    log "Installing Opencode $OC_VER -> /usr/local/bin/opencode"
    sudo install -m 0755 "$BUNDLE_DIR/apps/opencode" /usr/local/bin/opencode
    opencode --version 2>/dev/null && log "Opencode: OK" || warn "opencode --version failed (may be OK if flag differs)."
elif [[ -f "$BUNDLE_DIR/apps/opencode.MISSING" ]]; then
    warn "Opencode binary was not downloaded automatically during gather."
    warn "Download manually from https://github.com/sst/opencode/releases"
    warn "and copy to /usr/local/bin/opencode"
else
    warn "apps/opencode not found; skipping."
fi

# ============================================================================
# 6) NODE.JS + NPM
# ============================================================================
step "Node.js + npm"

if [[ -f "$BUNDLE_DIR/apps/nodejs.tar.xz" ]]; then
    NODE_VER=$(cat "$BUNDLE_DIR/apps/nodejs.version" 2>/dev/null || echo "unknown")
    log "Installing Node.js v${NODE_VER} to /opt/nodejs"
    sudo mkdir -p /opt/nodejs
    sudo tar -xJf "$BUNDLE_DIR/apps/nodejs.tar.xz" -C /opt/nodejs --strip-components=1
    sudo ln -sf /opt/nodejs/bin/node /usr/local/bin/node
    sudo ln -sf /opt/nodejs/bin/npm  /usr/local/bin/npm
    sudo ln -sf /opt/nodejs/bin/npx  /usr/local/bin/npx
    log "Node.js: $(node --version)  npm: $(npm --version)"
else
    warn "apps/nodejs.tar.xz not found; skipping Node.js."
fi

# ============================================================================
# 7) BUN
# ============================================================================
step "Bun"

if [[ -f "$BUNDLE_DIR/apps/bun-linux-x64.zip" ]]; then
    BUN_TAG=$(cat "$BUNDLE_DIR/apps/bun.version" 2>/dev/null || echo "unknown")
    log "Installing Bun $BUN_TAG"
    command -v unzip >/dev/null || sudo apt-get install -y unzip 2>/dev/null || true
    TMP_BUN=$(mktemp -d)
    unzip -q "$BUNDLE_DIR/apps/bun-linux-x64.zip" -d "$TMP_BUN"
    sudo install -m 0755 "$TMP_BUN"/bun-linux-x64/bun /usr/local/bin/bun
    sudo ln -sf /usr/local/bin/bun /usr/local/bin/bunx
    rm -rf "$TMP_BUN"
    log "Bun: $(bun --version)"
else
    warn "apps/bun-linux-x64.zip not found; skipping Bun."
fi

# ============================================================================
# ============================================================================
# 8) PYTHON VENV: LLM Inference
#    torch 2.10.0+cu130 | vLLM + LLM_API_fast + RAG + llama.cpp Python utils
# ============================================================================
step "Python venv: LLM Inference"

if [[ "$INSTALL_INFERENCE" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/inference"
    LLAMA_WHEELS="$BUNDLE_DIR/wheels/llamacpp"

    if [[ -d "$WHEELS_DIR" && -n "$(ls -A "$WHEELS_DIR" 2>/dev/null)" ]]; then
        log "Creating LLM Inference venv at $INFERENCE_PREFIX/venv"
        mkdir -p "$INFERENCE_PREFIX"
        "$PYTHON_BIN" -m venv "$INFERENCE_PREFIX/venv"
        # shellcheck disable=SC1091
        source "$INFERENCE_PREFIX/venv/bin/activate"

        pip install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools

        log "Installing PyTorch (inference)"
        pip install --no-index --find-links="$WHEELS_DIR" torch torchvision             || die "torch install failed — check wheels/inference/"

        log "Installing vLLM"
        pip install --no-index --find-links="$WHEELS_DIR" vllm             || warn "vLLM install failed."

        for rf in             "$BUNDLE_DIR/requirements/llm_api.txt"             "$BUNDLE_DIR/requirements/llm_api_full.txt"; do
            [[ -f "$rf" ]] || continue
            log "  Installing from $(basename "$rf")"
            pip install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
        done

        if [[ -d "$LLAMA_WHEELS" && -n "$(ls -A "$LLAMA_WHEELS" 2>/dev/null)" ]]; then
            for rf in "$LLAMA_PREFIX"/requirements.txt "$LLAMA_PREFIX"/requirements/*.txt; do
                [[ -f "$rf" ]] || continue
                pip install --no-index                     --find-links="$WHEELS_DIR" --find-links="$LLAMA_WHEELS"                     -r "$rf" 2>/dev/null || true
            done
        fi

        log "Installing core inference/RAG packages"
        pip install --no-index --find-links="$WHEELS_DIR"             sentence-transformers faiss-cpu rank-bm25             transformers tokenizers safetensors huggingface-hub             langchain langchain-core langchain-community langchain-ollama             langgraph langgraph-checkpoint langgraph-prebuilt langsmith             ollama tavily-python             fastapi uvicorn pydantic sse-starlette httpx aiohttp             passlib python-jose PyMuPDF python-docx pandas numpy Pillow             jupyter_client ipykernel filelock tqdm rich             2>/dev/null || true

        deactivate

        log "Smoke test: torch + vllm"
        "$INFERENCE_PREFIX/venv/bin/python" -c "
import torch, vllm
print(f"  torch {torch.__version__}")
print(f"  vllm  {vllm.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
" || warn "Inference smoke test failed — check venv and CUDA driver."

        log "LLM Inference venv ready: $INFERENCE_PREFIX/venv"
        log "Activate: source $INFERENCE_PREFIX/venv/bin/activate"
    else
        warn "wheels/inference/ empty or missing; skipping."
    fi
else
    log "INSTALL_INFERENCE=0; skipping."
fi

# 10) LLAMA.CPP — build from source
# ============================================================================
step "llama.cpp (build from source)"

if [[ "$INSTALL_LLAMA" == "1" ]]; then
    if [[ -f "$BUNDLE_DIR/src/llama.cpp.tar.gz" ]]; then
        log "Extracting llama.cpp -> $LLAMA_PREFIX"
        mkdir -p "$LLAMA_PREFIX"
        tar -xzf "$BUNDLE_DIR/src/llama.cpp.tar.gz" -C "$LLAMA_PREFIX" --strip-components=1

        # Install CUDA keyring first if bundled, so dpkg accepts CUDA-signed packages
        shopt -s nullglob
        keyring_debs=( "$BUNDLE_DIR"/debs/cuda-keyring*.deb )
        if (( ${#keyring_debs[@]} > 0 )); then
            log "Installing cuda-keyring: ${keyring_debs[*]##*/}"
            sudo dpkg -i "${keyring_debs[@]}"
        fi

        CMAKE_ARGS=(
            -S "$LLAMA_PREFIX"
            -B "$LLAMA_PREFIX/build"
            -DCMAKE_BUILD_TYPE=Release
            -DGGML_NATIVE=ON
            -DLLAMA_CURL=ON
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_EXAMPLES=ON
            -DLLAMA_BUILD_SERVER=ON
        )
        [[ "$BUILD_BLAS" == "1" ]] && CMAKE_ARGS+=( -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS )
        if [[ "$BUILD_CUDA" == "1" ]]; then
            if command -v nvcc >/dev/null; then
                CMAKE_ARGS+=( -DGGML_CUDA=ON )
                log "CUDA build enabled: $(nvcc --version | grep release)"
            else
                warn "BUILD_CUDA=1 but nvcc not found; building without CUDA."
            fi
        fi

        log "Configuring (cmake)"
        cmake "${CMAKE_ARGS[@]}"
        log "Building with -j${JOBS}"
        cmake --build "$LLAMA_PREFIX/build" --config Release -j"$JOBS"

        # Python venv for convert_hf_to_gguf.py and friends
        LLAMA_WHEELS="$BUNDLE_DIR/wheels/llamacpp"
        if [[ -d "$LLAMA_WHEELS" && -n "$(ls -A "$LLAMA_WHEELS" 2>/dev/null)" ]]; then
            log "Creating llama.cpp Python venv at $LLAMA_PREFIX/venv"
            "$PYTHON_BIN" -m venv "$LLAMA_PREFIX/venv"
            # shellcheck disable=SC1091
            source "$LLAMA_PREFIX/venv/bin/activate"
            pip install --no-index --find-links="$LLAMA_WHEELS" --upgrade pip wheel setuptools
            shopt -s nullglob
            for rf in "$LLAMA_PREFIX"/requirements.txt "$LLAMA_PREFIX"/requirements/*.txt; do
                [[ -f "$rf" ]] || continue
                pip install --no-index --find-links="$LLAMA_WHEELS" -r "$rf" \
                    || warn "Some llama.cpp packages failed for ${rf##*/}."
            done
            deactivate
        fi

        log "Smoke test: llama-cli --version"
        "$LLAMA_PREFIX/build/bin/llama-cli" --version \
            && log "llama.cpp OK" || warn "llama-cli smoke test failed — check build output."
    else
        warn "src/llama.cpp.tar.gz not found in bundle; skipping llama.cpp build."
    fi
else
    log "INSTALL_LLAMA=0; skipping."
fi

# ============================================================================
# ============================================================================
# 9) PYTHON VENV: General Training
#    torch 2.11.0+cu130 | PyG + MeshGraphNets + SimulGenVAE + PEMTRON
# ============================================================================
step "Python venv: General Training"

if [[ "$INSTALL_TRAINING" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/training"

    if [[ -d "$WHEELS_DIR" && -n "$(ls -A "$WHEELS_DIR" 2>/dev/null)" ]]; then
        log "Creating General Training venv at $TRAINING_PREFIX/venv"
        mkdir -p "$TRAINING_PREFIX"
        "$PYTHON_BIN" -m venv "$TRAINING_PREFIX/venv"
        # shellcheck disable=SC1091
        source "$TRAINING_PREFIX/venv/bin/activate"

        pip install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools

        log "Installing PyTorch (training)"
        pip install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio             || die "torch install failed — check wheels/training/"

        log "Installing torch-geometric"
        pip install --no-index --find-links="$WHEELS_DIR" torch-geometric             || warn "torch-geometric failed."

        log "Installing PyG extensions"
        for pkg in pyg_lib torch-scatter torch-sparse torch-cluster; do
            pip install --no-index --find-links="$WHEELS_DIR" "$pkg" 2>/dev/null                 && log "  $pkg: OK" || warn "  $pkg not found."
        done

        for rf in             "$BUNDLE_DIR/requirements/meshgraphnets.txt"             "$BUNDLE_DIR/requirements/simulgen.txt"             "$BUNDLE_DIR/requirements/pemtron.txt"             "$BUNDLE_DIR/requirements/pemtron_transfer.txt"; do
            [[ -f "$rf" ]] || continue
            log "  Installing from $(basename "$rf")"
            pip install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
        done

        log "Installing core training/scientific stack"
        pip install --no-index --find-links="$WHEELS_DIR"             numpy scipy h5py pandas tqdm matplotlib seaborn Pillow pyvista             scikit-learn scikit-image statsmodels networkx sympy             torchinfo tensorboard opencv-python imageio             librosa audiomentations soxr natsort reportlab             2>/dev/null || true

        deactivate

        log "Smoke test: torch + PyG"
        "$TRAINING_PREFIX/venv/bin/python" -c "
import torch
from torch_geometric.data import Data
print(f"  torch {torch.__version__}")
print(f"  torch-geometric OK")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device 0: {torch.cuda.get_device_name(0)}")
" || warn "Training smoke test failed — check venv and CUDA driver."

        log "General Training venv ready: $TRAINING_PREFIX/venv"
        log "Activate: source $TRAINING_PREFIX/venv/bin/activate"
    else
        warn "wheels/training/ empty or missing; skipping."
    fi
else
    log "INSTALL_TRAINING=0; skipping."
fi

# 12) PYTHON VENV: Jupyter + data science
# ============================================================================
step "Python venv: Jupyter + data science"

if [[ "$INSTALL_JUPYTER" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/jupyter"

    if [[ -d "$WHEELS_DIR" && -n "$(ls -A "$WHEELS_DIR" 2>/dev/null)" ]]; then
        log "Creating Jupyter venv at $JUPYTER_PREFIX/venv"
        mkdir -p "$JUPYTER_PREFIX"
        "$PYTHON_BIN" -m venv "$JUPYTER_PREFIX/venv"
        # shellcheck disable=SC1091
        source "$JUPYTER_PREFIX/venv/bin/activate"

        pip install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools

        pip install --no-index --find-links="$WHEELS_DIR" \
            jupyterlab notebook ipykernel ipywidgets jupyter-server \
            pandas polars numpy scipy matplotlib seaborn plotly \
            scikit-learn statsmodels tqdm rich requests aiohttp \
            black ruff mypy pytest ipdb \
            || warn "Some Jupyter packages failed; check output above."

        # Register the kernel so it appears in JupyterLab
        "$JUPYTER_PREFIX/venv/bin/python" -m ipykernel install \
            --user --name "airgap-py${PYTHON_VER}" --display-name "Python ${PYTHON_VER} (airgap)" \
            2>/dev/null || true

        deactivate

        # Drop a convenience launcher
        cat > "$HOME/start-jupyter.sh" <<JEOF
#!/usr/bin/env bash
source "$JUPYTER_PREFIX/venv/bin/activate"
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser "\$@"
JEOF
        chmod +x "$HOME/start-jupyter.sh"
        log "Jupyter venv ready at $JUPYTER_PREFIX/venv"
        log "Start with: bash ~/start-jupyter.sh  (or run in tmux for persistence)"
    else
        warn "wheels/jupyter/ empty or missing; skipping Jupyter venv."
    fi
else
    log "INSTALL_JUPYTER=0; skipping."
fi

# ============================================================================
# 13) DESKTOP ENVIRONMENT — XFCE4 + xrdp
# ============================================================================
step "Desktop environment (XFCE4 + xrdp)"

if [[ "$INSTALL_DESKTOP" == "1" ]]; then

    # ── xrdp: configure XFCE4 as the default session ─────────────────────────
    if command -v xrdp >/dev/null 2>&1; then
        log "Configuring xrdp to launch XFCE4 session"
        sudo bash -c 'cat > /etc/xrdp/startwm.sh <<'"'"'XRDPEOF'"'"'
#!/bin/sh
# Set locale
if [ -r /etc/default/locale ]; then
    . /etc/default/locale
    export LANG LANGUAGE
fi
exec startxfce4
XRDPEOF'
        sudo chmod +x /etc/xrdp/startwm.sh

        # Allow xrdp to read the TLS certificate (needed for NLA / encryption)
        sudo adduser xrdp ssl-cert 2>/dev/null || true

        # Enable and start xrdp
        sudo systemctl enable xrdp 2>/dev/null || true
        sudo systemctl restart xrdp 2>/dev/null \
            || sudo service xrdp restart 2>/dev/null \
            || warn "Could not restart xrdp — run 'sudo systemctl start xrdp' after reboot."
        log "xrdp listening on port 3389"

        # Open RDP port in ufw if the firewall is active
        if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
            sudo ufw allow 3389/tcp
            log "UFW: port 3389/tcp opened for RDP"
        fi
    else
        warn "xrdp not found — was INSTALL_DESKTOP=1 set during gather-all.sh?"
    fi

    # ── Default XFCE4 session for current user and new users ─────────────────
    echo "xfce4-session" | sudo tee /etc/skel/.xsession > /dev/null
    echo "xfce4-session" > "$HOME/.xsession"
    log "XFCE4 session set for: $USER (and skeleton for new users)"

    # ── lightdm: enable only when a physical display is attached ─────────────
    # On a headless GPU server xrdp provides the display; lightdm is not needed
    # and would fail to start. Uncomment the line below if a local display is used.
    # sudo systemctl enable lightdm 2>/dev/null || true

    # ── polkit rule so normal users can reboot/shutdown from XFCE ────────────
    if [[ -d /usr/share/polkit-1/rules.d ]]; then
        sudo tee /usr/share/polkit-1/rules.d/49-xfce-shutdown.rules > /dev/null <<'POLKIT'
polkit.addRule(function(action, subject) {
    if ((action.id == "org.freedesktop.login1.power-off" ||
         action.id == "org.freedesktop.login1.reboot") &&
        subject.isInGroup("sudo")) {
        return polkit.Result.YES;
    }
});
POLKIT
        log "polkit rule installed (sudo group can power-off/reboot from XFCE)"
    fi

    log "Desktop setup complete."
    log "Connect via RDP to port 3389 with your Linux username/password."
else
    log "INSTALL_DESKTOP=0; skipping desktop configuration."
fi

# ============================================================================
# 14) K3s — Common setup (runs for both server and agent)
# ============================================================================
step "K3s: common setup"

if [[ "$INSTALL_K3S" == "1" ]]; then
    K3S_DIR="$BUNDLE_DIR/k3s"
    [[ -d "$K3S_DIR" ]] || die "k3s/ not found in bundle. Re-run gather-all.sh with INCLUDE_K3S=1."

    # Load pinned versions from bundle metadata
    [[ -f "$K3S_DIR/meta/versions.env" ]] && source "$K3S_DIR/meta/versions.env"

    # Check GPUs (informational — GPU Operator handles the driver plugin)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        log "GPU check: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPU(s) visible"
    else
        warn "nvidia-smi not working. Install the NVIDIA driver before relying on GPU workloads."
    fi

    # Install K3s, helm, kubectl binaries
    log "Installing k3s, helm, kubectl -> /usr/local/bin/"
    sudo install -m 0755 "$K3S_DIR/bin/k3s"     /usr/local/bin/k3s
    sudo install -m 0755 "$K3S_DIR/bin/helm"    /usr/local/bin/helm
    sudo install -m 0755 "$K3S_DIR/bin/kubectl" /usr/local/bin/kubectl

    # Stage K3s airgap image archives (K3s loads these on first start)
    log "Staging K3s airgap images to /var/lib/rancher/k3s/agent/images/"
    sudo mkdir -p /var/lib/rancher/k3s/agent/images
    shopt -s nullglob
    for img_file in "$K3S_DIR/airgap-images/"*; do
        sudo cp "$img_file" /var/lib/rancher/k3s/agent/images/
        log "  Staged: $(basename "$img_file")"
    done

    # Write registries.yaml (mirrors all registries to the airgap registry)
    log "Writing /etc/rancher/k3s/registries.yaml"
    sudo mkdir -p /etc/rancher/k3s
    REGISTRY_HOST="${K3S_SERVER_IP:-127.0.0.1}"
    [[ "$K3S_ROLE" == "server" ]] && REGISTRY_HOST="127.0.0.1"
    sudo sed "s/REGISTRY_HOST/${REGISTRY_HOST}/g" \
        "$K3S_DIR/manifests/registries.yaml.tmpl" \
        | sudo tee /etc/rancher/k3s/registries.yaml > /dev/null
    log "registries.yaml -> mirrors all registries to ${REGISTRY_HOST}:${K3S_REGISTRY_PORT}"

    log "K3s common setup complete."
else
    log "INSTALL_K3S=0; skipping K3s."
fi

# ============================================================================
# 15) K3s — Server install
# ============================================================================
step "K3s: server install"

if [[ "$INSTALL_K3S" == "1" && "$K3S_ROLE" == "server" ]]; then
    K3S_DIR="$BUNDLE_DIR/k3s"
    [[ -f "$K3S_DIR/meta/versions.env" ]] && source "$K3S_DIR/meta/versions.env"

    SERVER_IP=$(hostname -I | awk '{print $1}')
    log "Server IP: $SERVER_IP"

    # Generate or load join token
    if [[ -n "$K3S_TOKEN_FILE" && -f "$K3S_TOKEN_FILE" ]]; then
        K3S_INIT_TOKEN=$(cat "$K3S_TOKEN_FILE")
    else
        K3S_INIT_TOKEN=$(openssl rand -hex 32)
    fi
    sudo mkdir -p /var/lib/rancher/k3s/server
    printf '%s' "$K3S_INIT_TOKEN" | sudo tee /var/lib/rancher/k3s/server/node-token >/dev/null
    sudo chmod 0600 /var/lib/rancher/k3s/server/node-token
    log "Join token written to /var/lib/rancher/k3s/server/node-token"

    # Install and start K3s server
    log "Running K3s server install (takes ~1-2 min)"
    INSTALL_K3S_SKIP_DOWNLOAD=true \
    K3S_TOKEN="$K3S_INIT_TOKEN" \
    INSTALL_K3S_EXEC="server --tls-san=${SERVER_IP} --write-kubeconfig-mode=644" \
        bash "$K3S_DIR/bin/k3s-install.sh"

    # Wait for server node to be Ready
    export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    log "Waiting for K3s node to be Ready..."
    for i in $(seq 1 36); do
        kubectl get nodes 2>/dev/null | grep -q " Ready " && break
        sleep 5
    done
    kubectl get nodes 2>/dev/null | grep -q " Ready " \
        || warn "Node not Ready after 3 min — check: kubectl get nodes; journalctl -u k3s"

    # Copy kubeconfig for the current user with server IP (not 127.0.0.1)
    mkdir -p "$HOME/.kube"
    sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
    sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
    sed -i "s|127.0.0.1|${SERVER_IP}|g" "$HOME/.kube/config"
    log "kubeconfig: $HOME/.kube/config"

    # Import registry:2 image into k3s containerd so it can run as a pod
    log "Importing registry image into k3s containerd"
    shopt -s nullglob
    imported=0
    for reg_tar in \
        "$K3S_DIR/images/docker.io_library_registry_2.8.3.tar" \
        "$K3S_DIR/images/"*registry*.tar; do
        [[ -f "$reg_tar" ]] || continue
        sudo /usr/local/bin/k3s ctr images import "$reg_tar" \
            && imported=1 && break
    done
    (( imported )) || warn "registry:2 image tar not found in k3s/images/ — pod may fail to start."

    # Deploy registry as a K8s pod on the control-plane node
    log "Deploying airgap-registry pod"
    kubectl apply -f "$K3S_DIR/manifests/airgap-registry.yaml" \
        || die "Failed to apply airgap-registry.yaml"
    log "Waiting for registry pod to be Ready..."
    kubectl wait --for=condition=Ready pod \
        -l app=airgap-registry \
        -n registry \
        --timeout=120s \
        || warn "Registry pod not Ready — check: kubectl get pods -n registry"

    # Push all pre-pulled images into the airgap registry
    if command -v skopeo >/dev/null 2>&1 && [[ -f "$K3S_DIR/meta/images-manifest.txt" ]]; then
        log "Pushing pre-pulled images to localhost:${K3S_REGISTRY_PORT}..."
        PUSH_FAILED=0
        while IFS= read -r img; do
            [[ -z "$img" ]] && continue
            safe="${img//\//_}"; safe="${safe//:/_}"; safe="${safe//@/_}"
            tar_file="$K3S_DIR/images/${safe}.tar"
            [[ -f "$tar_file" ]] || continue
            # Strip registry hostname — containerd mirror maps registry.host/path → local/path
            dest_path="${img#*/}"
            log "  Push: $img -> localhost:${K3S_REGISTRY_PORT}/${dest_path}"
            skopeo copy \
                --dest-tls-verify=false \
                "docker-archive:${tar_file}" \
                "docker://localhost:${K3S_REGISTRY_PORT}/${dest_path}" 2>/dev/null \
                || { warn "  Failed: $img"; PUSH_FAILED=$(( PUSH_FAILED + 1 )); }
        done < "$K3S_DIR/meta/images-manifest.txt"
        log "Image push complete ($PUSH_FAILED failed)."
    else
        warn "skopeo not found or images manifest missing — push images to the registry manually."
    fi

    # Install Helm stacks
    log "Installing Helm stacks (this may take 5-10 min per chart)"
    HELM=/usr/local/bin/helm
    CHARTS="$K3S_DIR/charts"
    VALUES="$K3S_DIR/manifests/values"

    _helm_install() {
        local release="$1" chart_glob="$2" ns="$3"; shift 3
        kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true
        # shellcheck disable=SC2206
        local chart_files=( $chart_glob )
        [[ -f "${chart_files[0]}" ]] || { warn "Chart not found: $chart_glob"; return 1; }
        "$HELM" upgrade --install "$release" "${chart_files[0]}" \
            --namespace "$ns" \
            --wait --timeout 10m \
            "$@" \
            || warn "helm install $release failed — check: helm status $release -n $ns"
    }

    # GPU Operator: driver.enabled=false because we install the driver via apt
    GPU_OP_ARGS=(--set driver.enabled="${GPU_OPERATOR_DRIVER_ENABLED}" --set migManager.enabled=false)
    [[ -f "$VALUES/gpu-operator.yaml" ]] && GPU_OP_ARGS+=(--values "$VALUES/gpu-operator.yaml")
    _helm_install gpu-operator "$CHARTS/gpu-operator-"*.tgz gpu-operator "${GPU_OP_ARGS[@]}"

    # kube-prometheus-stack (Prometheus + Grafana + AlertManager + node-exporter + DCGM)
    PROM_ARGS=()
    [[ -f "$VALUES/kube-prometheus-stack.yaml" ]] && PROM_ARGS+=(--values "$VALUES/kube-prometheus-stack.yaml")
    _helm_install kube-prometheus-stack "$CHARTS/kube-prometheus-stack-"*.tgz monitoring "${PROM_ARGS[@]}"

    # Loki + Promtail (centralized logs)
    LOKI_ARGS=()
    [[ -f "$VALUES/loki-stack.yaml" ]] && LOKI_ARGS+=(--values "$VALUES/loki-stack.yaml")
    _helm_install loki-stack "$CHARTS/loki-stack-"*.tgz monitoring "${LOKI_ARGS[@]}"

    # KubeRay operator (for cross-node 700GB+ model inference)
    if [[ "$INSTALL_KUBERAY" == "1" ]]; then
        KUBERAY_ARGS=()
        [[ -f "$VALUES/kuberay-operator.yaml" ]] && KUBERAY_ARGS+=(--values "$VALUES/kuberay-operator.yaml")
        _helm_install kuberay-operator "$CHARTS/kuberay-operator-"*.tgz kuberay-system "${KUBERAY_ARGS[@]}"
    fi

    # Apply healer manifests if present
    if [[ -d "$K3S_DIR/manifests/healer" ]]; then
        kubectl apply -f "$K3S_DIR/manifests/healer/" \
            || warn "Healer manifests failed to apply."
        log "Healer pod manifests applied."
    fi

    printf '\n'
    printf '\033[1;32m[install]\033[0m K3s SERVER ready.\n'
    printf '  Server IP  : %s\n' "$SERVER_IP"
    printf '  Join token : /var/lib/rancher/k3s/server/node-token\n'
    printf '  kubeconfig : %s/.kube/config\n' "$HOME"
    printf '  Grafana    : http://%s:30030  (if kube-prometheus-stack NodePort configured)\n' "$SERVER_IP"
    printf '\n'
    printf 'Distribute join token to agents:\n'
    printf '  scp /var/lib/rancher/k3s/server/node-token user@AGENT:/tmp/k3s-join-token\n'
    printf '\n'
    printf 'On each agent:\n'
    printf '  sudo INSTALL_K3S=1 K3S_ROLE=agent K3S_SERVER_IP=%s \\\n' "$SERVER_IP"
    printf '       K3S_TOKEN_FILE=/tmp/k3s-join-token bash install-all.sh\n'
    printf '\n'

elif [[ "$INSTALL_K3S" == "1" && "$K3S_ROLE" != "agent" ]]; then
    log "INSTALL_K3S=1 but K3S_ROLE=$K3S_ROLE — set K3S_ROLE=server or K3S_ROLE=agent."
fi

# ============================================================================
# 16) K3s — Agent install
# ============================================================================
step "K3s: agent install"

if [[ "$INSTALL_K3S" == "1" && "$K3S_ROLE" == "agent" ]]; then
    [[ -n "$K3S_SERVER_IP" ]] \
        || die "K3S_ROLE=agent requires K3S_SERVER_IP. Set it before running."
    [[ -n "$K3S_TOKEN_FILE" && -f "$K3S_TOKEN_FILE" ]] \
        || die "K3S_ROLE=agent requires K3S_TOKEN_FILE pointing to a readable file. scp the token from server 1 first."

    K3S_JOIN_TOKEN=$(cat "$K3S_TOKEN_FILE")
    K3S_DIR="$BUNDLE_DIR/k3s"

    log "Joining K3s cluster at https://${K3S_SERVER_IP}:6443"
    INSTALL_K3S_SKIP_DOWNLOAD=true \
    K3S_URL="https://${K3S_SERVER_IP}:6443" \
    K3S_TOKEN="$K3S_JOIN_TOKEN" \
    INSTALL_K3S_EXEC="agent" \
        bash "$K3S_DIR/bin/k3s-install.sh"

    log "Waiting for k3s-agent service to be active..."
    for i in $(seq 1 24); do
        systemctl is-active k3s-agent >/dev/null 2>&1 && break
        sleep 5
    done
    systemctl is-active k3s-agent >/dev/null 2>&1 \
        && log "k3s-agent is running." \
        || warn "k3s-agent not active — check: journalctl -u k3s-agent -f"

    printf '\n'
    printf '\033[1;32m[install]\033[0m K3s AGENT joined cluster.\n'
    printf '  Server : https://%s:6443\n' "$K3S_SERVER_IP"
    printf '\n'
    printf 'Verify from server 1:\n'
    printf '  kubectl get nodes -o wide\n'
    printf '\n'
fi

# ============================================================================
# Summary
# ============================================================================
step "Installation complete"
cat <<EOF

  APT packages  : installed from $BUNDLE_DIR/debs/
  VS Code       : $(command -v code 2>/dev/null && code --version 2>/dev/null | head -1 || echo "installed (check GUI)")
  Chrome        : $(command -v google-chrome-stable 2>/dev/null && google-chrome-stable --version 2>/dev/null || echo "installed (check GUI)")
  Firefox       : $(/opt/firefox/firefox --version 2>/dev/null || echo "installed at /opt/firefox")
  Opencode      : $(command -v opencode 2>/dev/null && (opencode --version 2>/dev/null || echo "installed") || echo "not installed")
  Node.js       : $(command -v node 2>/dev/null && node --version || echo "not installed")
  npm           : $(command -v npm  2>/dev/null && npm  --version || echo "not installed")
  Bun           : $(command -v bun  2>/dev/null && bun  --version || echo "not installed")
  Desktop (xrdp) : port 3389 — connect with any RDP client (use Linux username/password)
  llama.cpp          : $LLAMA_PREFIX/build/bin/llama-cli (server: llama-server)
  LLM Inference venv : $INFERENCE_PREFIX/venv   (vLLM + LLM_API_fast + RAG, torch 2.10.0+cu130)
  General Training   : $TRAINING_PREFIX/venv    (PyG + MeshGraphNets + SimulGenVAE, torch 2.11.0+cu130)
  Jupyter venv       : $JUPYTER_PREFIX/venv  (start: bash ~/start-jupyter.sh)
  K3s role           : ${K3S_ROLE}  $(command -v kubectl >/dev/null 2>&1 && kubectl get nodes 2>/dev/null | tail -n +2 || true)

Activate venvs:
  source $INFERENCE_PREFIX/venv/bin/activate    # vLLM / LLM_API_fast / RAG
  source $TRAINING_PREFIX/venv/bin/activate     # PyG / MeshGraphNets / SimulGenVAE
  source $JUPYTER_PREFIX/venv/bin/activate      # JupyterLab

Run llama-server:
  $LLAMA_PREFIX/build/bin/llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080

Run vLLM server:
  source $INFERENCE_PREFIX/venv/bin/activate
  python -m vllm.entrypoints.openai.api_server --model /path/to/model --host 0.0.0.0 --port 8000

Run JupyterLab:
  bash ~/start-jupyter.sh   # (run inside tmux for persistence)

Remote desktop:
  Windows: mstsc → server_ip:3389 (or use Remmina on Linux)
  CUDA / NVIDIA driver: re-run with INCLUDE_NVIDIA_DRIVER=1 / INCLUDE_CUDA_TOOLKIT=1 after
    configuring the NVIDIA apt repo on your WSL machine, then re-gather and re-install.

EOF
