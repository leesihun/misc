#!/usr/bin/env bash
# ============================================================================
# gather-llamacpp.sh
#   Run on an internet-connected WSL/Ubuntu machine.
#   Produces: $OUT_DIR and llamacpp-airgap-bundle.tar.gz
#   Copy the tarball to the airgapped server, then run install-llamacpp.sh.
#
# Target: Ubuntu 22.04 / x86_64 (hardcoded in target.env).
#   The gather machine SHOULD also be Ubuntu 22.04 — .debs downloaded on
#   another release (e.g. 24.04) will fail dpkg dependency checks on 22.04.
#   CUDA toolkit defaults to 13.0; override with CUDA_META_PKG=cuda-toolkit-13-X.
# ============================================================================
set -euo pipefail

# -------- configurable ------------------------------------------------------
OUT_DIR="${OUT_DIR:-$HOME/llamacpp-airgap}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"          # tag/branch/commit
INCLUDE_CUDA="${INCLUDE_CUDA:-1}"         # 1 = also bundle CUDA toolkit debs
INCLUDE_PYTHON="${INCLUDE_PYTHON:-0}"     # 1 = also bundle Python wheels for convert_hf_to_gguf.py
CUDA_META_PKG="${CUDA_META_PKG:-cuda-toolkit-13-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# ---------------------------------------------------------------------------

log() { printf '\033[1;36m[gather]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[gather:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] && die "Do NOT run as root. Script will invoke sudo as needed."
command -v sudo >/dev/null || die "sudo is required."

# Ensure local prerequisites are installed on this (internet-connected) machine.
# These are needed to RUN the gather script itself, not just bundled for the target.
_need_install=()
command -v git    >/dev/null || _need_install+=( git )
command -v curl   >/dev/null || _need_install+=( curl )
if [[ "$INCLUDE_PYTHON" == "1" ]]; then
    "$PYTHON_BIN" -m venv --help &>/dev/null || _need_install+=( python3-venv )
fi
if (( ${#_need_install[@]} > 0 )); then
    log "Installing missing local prerequisites: ${_need_install[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${_need_install[@]}" || true
fi
if [[ "$INCLUDE_PYTHON" == "1" ]]; then
    "$PYTHON_BIN" -m venv --help &>/dev/null \
        || die "python3-venv still unavailable. Run: sudo apt install python3.$(python3 -c 'import sys;print(sys.version_info.minor)')-venv"
fi

log "Output directory: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{debs,src,meta}
[[ "$INCLUDE_PYTHON" == "1" ]] && mkdir -p "$OUT_DIR/wheels"

# Record target info so install script can sanity-check
# Target is Ubuntu 22.04 regardless of the gather machine's OS
GATHER_PY=$($PYTHON_BIN -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "n/a")
cat > "$OUT_DIR/meta/target.env" <<EOF
BUNDLE_OS_ID=ubuntu
BUNDLE_OS_VERSION=22.04
BUNDLE_ARCH=$(dpkg --print-architecture)
BUNDLE_PYTHON=$GATHER_PY
BUNDLE_LLAMA_REF=$LLAMA_REF
BUNDLE_INCLUDE_CUDA=$INCLUDE_CUDA
BUNDLE_INCLUDE_PYTHON=$INCLUDE_PYTHON
BUNDLE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
source /etc/os-release
log "Gather host: $ID $VERSION_ID / $(dpkg --print-architecture) / py$GATHER_PY"
log "Target:      ubuntu 22.04 / $(dpkg --print-architecture)"
if [[ "$INCLUDE_PYTHON" == "1" ]]; then
    TARGET_PY="3.10"   # Ubuntu 22.04 system Python
    if [[ "$GATHER_PY" != "$TARGET_PY" ]]; then
        cat <<EOF
[gather:WARN] Python mismatch: gather is $GATHER_PY, target is $TARGET_PY.
              C-extension wheels are tagged cp${GATHER_PY//./} and will NOT
              install on the target's cp${TARGET_PY//./}.
              Press Ctrl-C to abort, or wait 10s to continue anyway...
EOF
        sleep 10
    fi
else
    log "Python wheels: skipped (INCLUDE_PYTHON=0). C++ binaries don't need Python at runtime."
fi

# ---------------------------------------------------------------------------
# 1) APT packages (download only, with all transitive dependencies)
# ---------------------------------------------------------------------------
BASE_PKGS=(
    # Core build toolchain
    build-essential
    gcc-12-base          # explicitly included: many pkgs pin exact version; target may lag
    gcc-11
    g++-11
    libc6                # explicitly included: commonly patched, target may lag
    libc6-dev
    libgcc-11-dev
    libgomp1
    libatomic1
    libexpat1            # explicitly included: commonly patched
    zlib1g
    zlib1g-dev
    cmake
    git
    ccache
    pkg-config
    # Network (for llama-server --hf-repo; no-op offline but headers needed at build)
    curl
    ca-certificates
    libcurl4-openssl-dev
    # BLAS — include the real providers of the virtual deps, not just the meta-packages
    libopenblas-dev
    libopenblas0
    libopenblas-pthread-dev
    libopenblas0-pthread
    # Python (only needed if INCLUDE_PYTHON=1, but bundling avoids missing-dep noise)
    python3
    python3-pip
    python3-venv
    python3-dev
    python3-wheel
    python3-setuptools
)
if [[ "$INCLUDE_CUDA" == "1" ]]; then
    BASE_PKGS+=( "$CUDA_META_PKG" )
    log "CUDA bundle enabled: $CUDA_META_PKG (NVIDIA apt repo must already be configured)"
fi

log "Refreshing apt indexes"
sudo apt-get update

log "Cleaning local apt cache to isolate downloads"
sudo apt-get clean

log "Downloading .debs (install --download-only resolves transitive deps)"
# --reinstall forces debs to be (re)downloaded even if already installed on WSL.
sudo apt-get install -y --download-only --reinstall "${BASE_PKGS[@]}"

log "Copying debs from /var/cache/apt/archives/ -> $OUT_DIR/debs/"
shopt -s nullglob
debs=(/var/cache/apt/archives/*.deb)
(( ${#debs[@]} > 0 )) || die "No .deb files were downloaded."

# Packages that pull in unresolvable GUI/desktop deps on a headless server.
# They have nothing to do with llama.cpp and will leave broken-package state
# on targets that lack libwebkit2gtk, libgtk, etc.
EXCLUDE_PKGS=( open-code opencode )

copied=0; skipped=0
for deb in "${debs[@]}"; do
    pkg=$(dpkg-deb -f "$deb" Package 2>/dev/null || true)
    skip=0
    for ex in "${EXCLUDE_PKGS[@]}"; do
        [[ "$pkg" == "$ex" ]] && skip=1 && break
    done
    if (( skip )); then
        log "  Skipping unrelated package: $pkg ($(basename "$deb"))"
        (( skipped++ )) || true
    else
        sudo cp "$deb" "$OUT_DIR/debs/"
        (( copied++ )) || true
    fi
done
sudo chown -R "$(id -u):$(id -g)" "$OUT_DIR/debs"
log "Collected $copied .deb files ($(du -sh "$OUT_DIR/debs" | cut -f1)); skipped $skipped"

# ---------------------------------------------------------------------------
# 2) llama.cpp source (with submodules, at the requested ref)
# ---------------------------------------------------------------------------
log "Cloning $LLAMA_REPO @ $LLAMA_REF"
git clone --recurse-submodules "$LLAMA_REPO" "$OUT_DIR/src/llama.cpp"
git -C "$OUT_DIR/src/llama.cpp" checkout "$LLAMA_REF"
git -C "$OUT_DIR/src/llama.cpp" submodule update --init --recursive
LLAMA_COMMIT=$(git -C "$OUT_DIR/src/llama.cpp" rev-parse HEAD)
echo "BUNDLE_LLAMA_COMMIT=$LLAMA_COMMIT" >> "$OUT_DIR/meta/target.env"
log "llama.cpp at commit $LLAMA_COMMIT"

# Tarball the source so it is easy to move and preserves file modes
log "Archiving source tree"
tar --exclude='.git' -C "$OUT_DIR/src" -czf "$OUT_DIR/src/llama.cpp.tar.gz" llama.cpp
rm -rf "$OUT_DIR/src/llama.cpp"

# ---------------------------------------------------------------------------
# 3) Python wheels for llama.cpp's convert/utility scripts (optional)
#    Skipped by default — only needed if you plan to run convert_hf_to_gguf.py
#    on the target. Pre-converted .gguf models don't need Python at all.
# ---------------------------------------------------------------------------
if [[ "$INCLUDE_PYTHON" == "1" ]]; then
    log "Extracting requirements from source archive"
    REQ_DIR="$(mktemp -d)"
    tar -xzf "$OUT_DIR/src/llama.cpp.tar.gz" -C "$REQ_DIR"
    REQ_ROOT="$REQ_DIR/llama.cpp"

    REQ_FILES=()
    [[ -f "$REQ_ROOT/requirements.txt" ]] && REQ_FILES+=( "$REQ_ROOT/requirements.txt" )
    if [[ -d "$REQ_ROOT/requirements" ]]; then
        while IFS= read -r f; do REQ_FILES+=( "$f" ); done < <(find "$REQ_ROOT/requirements" -maxdepth 1 -name '*.txt')
    fi
    (( ${#REQ_FILES[@]} > 0 )) || log "WARN: no requirements files found; skipping wheel download"

    if (( ${#REQ_FILES[@]} > 0 )); then
        log "Downloading wheels for: ${REQ_FILES[*]##*/}"
        # Upgrade pip in a temp venv so we get modern resolver + wheel support
        "$PYTHON_BIN" -m venv "$REQ_DIR/venv"
        # shellcheck disable=SC1091
        source "$REQ_DIR/venv/bin/activate"
        pip install --upgrade pip wheel setuptools
        # Also stash pip/wheel/setuptools themselves so airgap can bootstrap cleanly
        pip download --dest "$OUT_DIR/wheels" pip wheel setuptools
        for rf in "${REQ_FILES[@]}"; do
            pip download --dest "$OUT_DIR/wheels" -r "$rf" || die "pip download failed for $rf"
        done
        deactivate
        log "Wheel cache: $(ls "$OUT_DIR/wheels" | wc -l) files ($(du -sh "$OUT_DIR/wheels" | cut -f1))"
        # Copy requirements into the bundle for offline use
        mkdir -p "$OUT_DIR/meta/requirements"
        cp "${REQ_FILES[@]}" "$OUT_DIR/meta/requirements/" 2>/dev/null || true
        [[ -d "$REQ_ROOT/requirements" ]] && cp -r "$REQ_ROOT/requirements/." "$OUT_DIR/meta/requirements/"
    fi
    rm -rf "$REQ_DIR"
else
    log "Skipping Python wheels (INCLUDE_PYTHON=0)"
    rmdir "$OUT_DIR/wheels" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 4) Checksums + final tarball + companion installer
# ---------------------------------------------------------------------------
log "Generating SHA256 manifest"
SHA_DIRS=( debs src meta )
[[ -d "$OUT_DIR/wheels" ]] && SHA_DIRS+=( wheels )
( cd "$OUT_DIR" && find "${SHA_DIRS[@]}" -type f -print0 | xargs -0 sha256sum > meta/SHA256SUMS )

BUNDLE_PARENT="$(dirname "$OUT_DIR")"
BUNDLE_NAME="$(basename "$OUT_DIR")"
BUNDLE_BIN="$BUNDLE_PARENT/llamacpp-airgap-bundle.bin"
log "Packing bundle -> $BUNDLE_BIN"
rm -f "$BUNDLE_BIN"
tar -czf "$BUNDLE_BIN" -C "$BUNDLE_PARENT" "$BUNDLE_NAME"

# Copy the installer next to the bundle so the user only has to transfer two files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/install-llamacpp.sh" ]]; then
    cp "$SCRIPT_DIR/install-llamacpp.sh" "$BUNDLE_PARENT/install-llamacpp.sh"
    chmod +x "$BUNDLE_PARENT/install-llamacpp.sh"
    log "Companion installer: $BUNDLE_PARENT/install-llamacpp.sh"
fi

log "Done."
printf '\n  Bundle    : %s (%s)\n' "$BUNDLE_BIN" "$(du -sh "$BUNDLE_BIN" | cut -f1)"
printf '  Installer : %s\n' "$BUNDLE_PARENT/install-llamacpp.sh"
printf '  Staging   : %s\n\n' "$OUT_DIR"
printf 'Next:\n'
printf '  scp "%s" "%s" user@airgapped:~\n' "$BUNDLE_BIN" "$BUNDLE_PARENT/install-llamacpp.sh"
printf '  ssh user@airgapped\n'
printf '  bash install-llamacpp.sh    # auto-extracts the bundle\n'
