#!/usr/bin/env bash
# ============================================================================
# install-all.d/14-llamacpp-build.sh
#
#   Build llama.cpp from the bundled source against the system CUDA toolkit
#   installed by install-nvidia.sh.
#
#   Canonical cmake flags (updated for llama.cpp master, 2026-05):
#     -DGGML_CUDA=ON               (was LLAMA_CUBLAS / LLAMA_CUDA — both deprecated)
#     -DCMAKE_CUDA_ARCHITECTURES=100-real;103-real
#                                  (B200=sm_100, B300=sm_103; -real strips PTX
#                                   since hardware is fixed)
#     -DLLAMA_OPENSSL=ON           (replaces deprecated LLAMA_CURL after issue
#                                   #18922; libssl-dev is in the userland bundle)
#     -DLLAMA_BUILD_UI=OFF         (headless server — no embedded web UI)
#
#   NCCL: -DGGML_CUDA_NCCL is auto-detected. install-nvidia.sh defaults
#   SKIP_NCCL=1 (no system libnccl), so the build will NOT link NCCL.
#   Tensor-parallel (--split-mode row) will log a perf warning at runtime;
#   pipeline-parallel (--split-mode layer, default) is unaffected. This is
#   intentional — avoids ABI skew with PyTorch/vLLM venv-bundled NCCL.
#
#   Directly runnable: sudo bash install-all.d/14-llamacpp-build.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "14-llamacpp-build"
locate_bundle
source_bundle_metadata
detect_target_user

if [[ "$INSTALL_LLAMA" != "1" ]] || [[ ! -f "$BUNDLE_DIR/src/llama.cpp.tar.gz" ]]; then
    log "INSTALL_LLAMA=$INSTALL_LLAMA or src/llama.cpp.tar.gz missing; skipping."
    mark_step_ok
    exit 0
fi

step "1. Extract llama.cpp source -> $LLAMA_PREFIX"
rm -rf "$LLAMA_PREFIX"
mkdir -p "$LLAMA_PREFIX"
chown "$TARGET_USER:$TARGET_GROUP" "$LLAMA_PREFIX"
_as_user tar -xzf "$BUNDLE_DIR/src/llama.cpp.tar.gz" -C "$LLAMA_PREFIX" --strip-components=1 \
    || die "llama.cpp source extraction failed."

step "2. Locate nvcc"
NVCC_PATH=""
for c in /usr/local/cuda/bin/nvcc "/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin/nvcc"; do
    [[ -x "$c" ]] && { NVCC_PATH="$c"; break; }
done
[[ -n "$NVCC_PATH" ]] || die "nvcc not found under /usr/local/cuda* — install-nvidia.sh did not run?"
log "nvcc: $NVCC_PATH"

step "3. cmake configure"
CMAKE_ARGS=(
    -S "$LLAMA_PREFIX"
    -B "$LLAMA_PREFIX/build"
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CUDA=ON
    -DGGML_NATIVE=OFF
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_LIST"
    -DCMAKE_CUDA_COMPILER="$NVCC_PATH"
    -DCUDAToolkit_ROOT=/usr/local/cuda
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_UI=OFF
    -DLLAMA_OPENSSL=ON
)

log "cmake -S $LLAMA_PREFIX -B $LLAMA_PREFIX/build (arch=$CUDA_ARCH_LIST, jobs=$JOBS, nvcc=$NVCC_PATH)"
_as_user cmake "${CMAKE_ARGS[@]}" \
    || die "cmake configure failed."

step "4. cmake --build (parallel jobs=$JOBS)"
_as_user cmake --build "$LLAMA_PREFIX/build" --config Release -j"$JOBS" \
    || die "llama.cpp build failed."

log "llama.cpp built: $LLAMA_PREFIX/build/bin/llama-server"

step "5. Python venv for convert_hf_to_gguf.py etc. (optional)"
LLAMA_WHEELS="$BUNDLE_DIR/wheels/llamacpp"
if _wheelhouse_has_packages "$LLAMA_WHEELS"; then
    log "Creating llama.cpp Python venv at $LLAMA_PREFIX/venv"
    if _as_user "$PYTHON_BIN" -m venv "$LLAMA_PREFIX/venv"; then
        _PIP="$LLAMA_PREFIX/venv/bin/pip"
        _as_user "$_PIP" install --no-index --find-links="$LLAMA_WHEELS" --upgrade pip wheel setuptools 2>/dev/null || true
        shopt -s nullglob
        for rf in "$LLAMA_PREFIX"/requirements.txt "$LLAMA_PREFIX"/requirements/*.txt; do
            [[ -f "$rf" ]] || continue
            _as_user "$_PIP" install --no-index --find-links="$LLAMA_WHEELS" -r "$rf" 2>/dev/null || true
        done
        shopt -u nullglob
    fi
else
    log "wheels/llamacpp/ empty — skipping llama.cpp utility venv"
fi

step "6. Smoke test"
# Use explicit if/else rather than A && B || C — the latter would incorrectly
# fire the "failed" warn if the `log` helper itself ever returned non-zero.
if "$LLAMA_PREFIX/build/bin/llama-cli" --version 2>&1 | head -3; then
    log "llama.cpp OK"
else
    warn "llama-cli --version failed."
fi

mark_step_ok
