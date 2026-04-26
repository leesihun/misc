#!/usr/bin/env bash
# ============================================================================
# gather-llamacpp.sh
#   Run on an internet-connected WSL/Ubuntu machine.
#   Produces: $OUT_DIR and llamacpp-airgap-bundle.tar.gz
#   Copy the tarball to the airgapped server, then run install-llamacpp.sh.
#
# Assumptions:
#   - WSL and target server are the SAME distro + major version (e.g. both
#     Ubuntu 22.04) and SAME architecture (x86_64). Mismatches will break
#     .deb installation on the target.
# ============================================================================
set -euo pipefail

# -------- configurable ------------------------------------------------------
OUT_DIR="${OUT_DIR:-$HOME/llamacpp-airgap}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_REF="${LLAMA_REF:-master}"          # tag/branch/commit
INCLUDE_CUDA="${INCLUDE_CUDA:-1}"         # 1 = also bundle CUDA toolkit debs
CUDA_META_PKG="${CUDA_META_PKG:-cuda-toolkit-13-1}"
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
"$PYTHON_BIN" -m venv --help &>/dev/null || _need_install+=( python3.12-venv python3-venv )
if (( ${#_need_install[@]} > 0 )); then
    log "Installing missing local prerequisites: ${_need_install[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${_need_install[@]}" || true
fi
"$PYTHON_BIN" -m venv --help &>/dev/null \
    || die "python3-venv still unavailable. Run: sudo apt install python3.$(python3 -c 'import sys;print(sys.version_info.minor)')-venv"

log "Output directory: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"/{debs,src,wheels,meta}

# Record target info so install script can sanity-check
source /etc/os-release
cat > "$OUT_DIR/meta/target.env" <<EOF
BUNDLE_OS_ID=$ID
BUNDLE_OS_VERSION=$VERSION_ID
BUNDLE_ARCH=$(dpkg --print-architecture)
BUNDLE_PYTHON=$($PYTHON_BIN -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
BUNDLE_LLAMA_REF=$LLAMA_REF
BUNDLE_INCLUDE_CUDA=$INCLUDE_CUDA
BUNDLE_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
log "Host: $ID $VERSION_ID / $(dpkg --print-architecture) / py$($PYTHON_BIN -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# ---------------------------------------------------------------------------
# 1) APT packages (download only, with all transitive dependencies)
# ---------------------------------------------------------------------------
BASE_PKGS=(
    build-essential
    cmake
    git
    ccache
    pkg-config
    curl
    ca-certificates
    libcurl4-openssl-dev
    libopenblas-dev
    libopenblas0
    libgomp1
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
sudo cp "${debs[@]}" "$OUT_DIR/debs/"
sudo chown -R "$(id -u):$(id -g)" "$OUT_DIR/debs"
log "Collected $(ls "$OUT_DIR/debs" | wc -l) .deb files ($(du -sh "$OUT_DIR/debs" | cut -f1))"

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
# 3) Python wheels for llama.cpp's convert/utility scripts
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 4) Checksums + final tarball
# ---------------------------------------------------------------------------
log "Generating SHA256 manifest"
( cd "$OUT_DIR" && find debs src wheels meta -type f -print0 | xargs -0 sha256sum > meta/SHA256SUMS )

BUNDLE_TGZ="$(dirname "$OUT_DIR")/llamacpp-airgap-bundle.tar.gz"
log "Packing bundle -> $BUNDLE_TGZ"
tar -czf "$BUNDLE_TGZ" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"

log "Done."
printf '\n  Bundle : %s (%s)\n' "$BUNDLE_TGZ" "$(du -sh "$BUNDLE_TGZ" | cut -f1)"
printf '  Staging: %s\n\n' "$OUT_DIR"
printf 'Next: scp "%s" user@airgapped:~ ; then run install-llamacpp.sh\n' "$BUNDLE_TGZ"
