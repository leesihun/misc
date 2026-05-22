#!/usr/bin/env bash
# ============================================================================
# install-all.d/11-venv-inference.sh
#
#   LLM inference venv at $INFERENCE_PREFIX/venv. Installs torch (cu130),
#   optional vLLM, FastAPI + RAG stack from the bundled wheelhouse.
#
#   Directly runnable: sudo bash install-all.d/11-venv-inference.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "11-venv-inference"
locate_bundle
detect_target_user

if [[ "$INSTALL_INFERENCE" != "1" ]]; then
    log "INSTALL_INFERENCE=0; skipping."
    mark_step_ok
    exit 0
fi

WHEELS_DIR="$BUNDLE_DIR/wheels/inference"
if ! _wheelhouse_has_packages "$WHEELS_DIR"; then
    warn "wheels/inference/ empty; skipping."
    mark_step_ok
    exit 0
fi

step "1. Create venv at $INFERENCE_PREFIX/venv"
mkdir -p "$INFERENCE_PREFIX"
chown "$TARGET_USER:$TARGET_GROUP" "$INFERENCE_PREFIX"
_as_user "$PYTHON_BIN" -m venv "$INFERENCE_PREFIX/venv" || die "Could not create inference venv."

_PIP="$INFERENCE_PREFIX/venv/bin/pip"

step "2. Bootstrap pip / wheel / setuptools"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
    || warn "Bootstrap pip install failed."

step "3. PyTorch (cu130)"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio \
    || warn "torch install failed."

if [[ "$INSTALL_VLLM" == "1" ]]; then
    step "4. vLLM (pinned to cu130 backend)"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" vllm \
        || warn "vLLM install failed."
else
    log "INSTALL_VLLM=0 (default) — skipping vLLM. Re-run with INSTALL_VLLM=1 once stack is validated."
fi

step "5. Project requirements"
for rf in "$BUNDLE_DIR/requirements/llm_api.txt" "$BUNDLE_DIR/requirements/llm_api_full.txt"; do
    [[ -f "$rf" ]] || continue
    log "  Installing from $(basename "$rf")"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
done

step "6. Core inference / RAG packages"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
    sentence-transformers faiss-cpu rank-bm25 \
    transformers tokenizers safetensors huggingface-hub tiktoken \
    langchain langchain-core langchain-community langchain-ollama \
    langgraph langgraph-checkpoint langgraph-prebuilt langsmith \
    ollama tavily-python \
    fastapi uvicorn pydantic pydantic-settings sse-starlette \
    httpx httpx-sse aiohttp aiofiles websockets \
    passlib python-jose \
    PyMuPDF pypdf python-docx python-pptx openpyxl \
    pandas numpy Pillow python-dotenv python-multipart \
    jupyter_client ipykernel filelock tqdm rich 2>/dev/null || true

step "7. Smoke test"
INSTALL_VLLM="$INSTALL_VLLM" _as_user env INSTALL_VLLM="$INSTALL_VLLM" \
    "$INFERENCE_PREFIX/venv/bin/python" - <<'PY' || warn "Inference smoke test failed."
import os
import torch
print(f"  torch {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device count:   {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if os.environ.get("INSTALL_VLLM") == "1":
    try:
        import vllm
        print(f"  vllm  {vllm.__version__}")
    except Exception as e:
        print(f"  vllm import failed: {e}")
else:
    print("  vllm: skipped (INSTALL_VLLM=0)")
PY

step "8. Pre-warm sm_103 PTX-JIT cache (B300 cubins not in PyTorch 2.11)"
if _as_user "$INFERENCE_PREFIX/venv/bin/python" -c \
    "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda').sum().item()" \
    2>/dev/null; then
    log "PTX-JIT cache pre-warmed for sm_103"
else
    warn "PTX-JIT pre-warm skipped (CUDA not initialized yet?)"
fi

log "Inference venv ready: $INFERENCE_PREFIX/venv"
mark_step_ok
