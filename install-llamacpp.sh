#!/usr/bin/env bash
# ============================================================================
# install-llamacpp.sh
#   Run on the AIRGAPPED Linux server, after extracting the bundle created
#   by gather-llamacpp.sh.
#
# Usage:
#   tar -xzf llamacpp-airgap-bundle.tar.gz
#   cd llamacpp-airgap
#   ./install-llamacpp.sh                  # CPU build (default)
#   BUILD_CUDA=1 ./install-llamacpp.sh     # CUDA build (needs CUDA debs bundled)
#   PREFIX=/opt/llama.cpp ./install-llamacpp.sh
# ============================================================================
set -euo pipefail

# -------- configurable ------------------------------------------------------
BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PREFIX="${PREFIX:-$HOME/llama.cpp}"
BUILD_CUDA="${BUILD_CUDA:-1}"
BUILD_BLAS="${BUILD_BLAS:-1}"
VERIFY_CHECKSUMS="${VERIFY_CHECKSUMS:-1}"
JOBS="${JOBS:-$(nproc)}"
# ---------------------------------------------------------------------------

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install:WARN]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[install:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "$BUNDLE_DIR/debs" && -d "$BUNDLE_DIR/src" ]] \
    || die "Bundle not found under $BUNDLE_DIR (expected debs/ and src/ subdirs)"

# ---------------------------------------------------------------------------
# 0) Environment sanity check
# ---------------------------------------------------------------------------
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
        || die "Checksum verification failed — bundle is corrupted."
fi

# ---------------------------------------------------------------------------
# 1) Install .deb packages
# ---------------------------------------------------------------------------
log "Installing .deb packages from $BUNDLE_DIR/debs"

# cuda-keyring must be installed first — it registers NVIDIA's GPG key which
# dpkg needs before it will accept any CUDA-signed packages. It lands in debs/
# as a transitive dep of cuda-toolkit pulled in by apt during gather.
shopt -s nullglob
keyring_debs=( "$BUNDLE_DIR"/debs/cuda-keyring*.deb )
if (( ${#keyring_debs[@]} > 0 )); then
    log "Installing cuda-keyring first: ${keyring_debs[*]##*/}"
    sudo dpkg -i "${keyring_debs[@]}"
fi

# dpkg -i on a large CUDA bundle fails due to dependency ordering — packages
# reference each other and a single pass leaves some unconfigured. We retry
# until nothing changes (converges in 2-3 passes for typical CUDA bundles).
all_debs=( "$BUNDLE_DIR"/debs/*.deb )
(( ${#all_debs[@]} > 0 )) || die "No .deb files found in $BUNDLE_DIR/debs/"
log "Installing ${#all_debs[@]} .deb packages (may take several passes)"
for pass in 1 2 3; do
    log "dpkg pass $pass"
    sudo dpkg -i --force-depends "${all_debs[@]}" 2>&1 | grep -v '^(Reading\|Selecting\|Preparing\|Unpacking\|Setting up\|Processing)' || true
    # Check if anything is still broken
    broken=$(dpkg -l | awk '/^.H|^.U|^.F/ {print $2}' | wc -l)
    (( broken == 0 )) && break
    log "$broken packages still in broken state, retrying..."
done
# Final fix-up for any remaining dependency issues using only bundled debs
sudo apt-get -f install -y --no-download --ignore-missing \
    -o Dir::Cache::archives="$BUNDLE_DIR/debs" 2>/dev/null || true
broken_final=$(dpkg -l | awk '/^.H|^.U|^.F/ {print $2}' | wc -l)
(( broken_final == 0 )) || die "Package installation failed: $broken_final broken packages remain. Run 'dpkg -l | grep -E \"^.H|^.U|^.F\"' for details."

# Sanity: core build tools must exist now
for bin in gcc g++ make cmake git python3 pip3; do
    command -v "$bin" >/dev/null || die "Required tool '$bin' not found after install."
done
log "Toolchain OK: $(gcc --version | head -1), $(cmake --version | head -1)"

# ---------------------------------------------------------------------------
# 2) Extract llama.cpp source
# ---------------------------------------------------------------------------
log "Extracting llama.cpp -> $PREFIX"
mkdir -p "$PREFIX"
tar -xzf "$BUNDLE_DIR/src/llama.cpp.tar.gz" -C "$PREFIX" --strip-components=1

# ---------------------------------------------------------------------------
# 3) Configure & build
# ---------------------------------------------------------------------------
CMAKE_ARGS=(
    -S "$PREFIX"
    -B "$PREFIX/build"
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_NATIVE=ON            # optimize for THIS machine's CPU
    -DLLAMA_CURL=ON             # llama-server can pull HF models (no-op offline)
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_BUILD_SERVER=ON
)
[[ "$BUILD_BLAS" == "1" ]] && CMAKE_ARGS+=( -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS )
if [[ "$BUILD_CUDA" == "1" ]]; then
    command -v nvcc >/dev/null || die "BUILD_CUDA=1 but nvcc not found. CUDA debs missing from bundle?"
    CMAKE_ARGS+=( -DGGML_CUDA=ON )
    log "CUDA build enabled ($(nvcc --version | grep release))"
fi

log "Configuring (cmake)"
cmake "${CMAKE_ARGS[@]}"

log "Building with -j$JOBS"
cmake --build "$PREFIX/build" --config Release -j"$JOBS"

# ---------------------------------------------------------------------------
# 4) Python venv for convert_hf_to_gguf.py and friends
# ---------------------------------------------------------------------------
if [[ -d "$BUNDLE_DIR/wheels" && -n "$(ls -A "$BUNDLE_DIR/wheels" 2>/dev/null)" ]]; then
    log "Creating Python venv at $PREFIX/venv"
    python3 -m venv "$PREFIX/venv"
    # shellcheck disable=SC1091
    source "$PREFIX/venv/bin/activate"

    # Bootstrap pip/wheel/setuptools from bundled wheels (no index)
    pip install --no-index --find-links="$BUNDLE_DIR/wheels" --upgrade pip wheel setuptools

    shopt -s nullglob
    req_root_files=( "$PREFIX"/requirements.txt )
    req_sub_files=( "$PREFIX"/requirements/*.txt )
    for rf in "${req_root_files[@]}" "${req_sub_files[@]}"; do
        [[ -f "$rf" ]] || continue
        log "pip install -r $rf"
        pip install --no-index --find-links="$BUNDLE_DIR/wheels" -r "$rf" \
            || warn "Some packages in $rf failed to install; continuing."
    done
    deactivate
else
    warn "No wheels bundled; skipping Python env setup."
fi

# ---------------------------------------------------------------------------
# 5) Smoke test
# ---------------------------------------------------------------------------
log "Smoke test: $PREFIX/build/bin/llama-cli --version"
"$PREFIX/build/bin/llama-cli" --version || warn "llama-cli did not run cleanly."

cat <<EOF

\033[1;32m[install] Done.\033[0m

  Source      : $PREFIX
  Binaries    : $PREFIX/build/bin
  Python venv : $PREFIX/venv (if wheels were bundled)

Add to PATH:
    export PATH="$PREFIX/build/bin:\$PATH"

Run the server:
    llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080

Convert a HF model to GGUF (needs venv):
    source $PREFIX/venv/bin/activate
    python $PREFIX/convert_hf_to_gguf.py /path/to/hf-model --outfile model.gguf

EOF
