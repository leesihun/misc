#!/usr/bin/env bash
# ============================================================================
# install-all.d/11-venv-inference.sh
#
#   CPU-only inference / RAG / FastAPI venv at $INFERENCE_PREFIX/venv.
#
#   NO PyTorch (cu130), NO vLLM — those used to live here but caused
#   GPU-NCCL-ABI footguns alongside the training/llama.cpp stacks. Inference
#   workloads on this box go through llama.cpp's HTTP server (built in
#   step 14); this venv exists to host the FastAPI / langchain / RAG /
#   embedding-loader glue around it.
#
#   If you later need a GPU-resident inference engine, build it in its own
#   venv on top of the training stack (step 12) — do NOT add torch back here.
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

step "3. Project requirements"
for rf in "$BUNDLE_DIR/requirements/llm_api.txt" "$BUNDLE_DIR/requirements/llm_api_full.txt"; do
    [[ -f "$rf" ]] || continue
    log "  Installing from $(basename "$rf")"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
done

step "4. Core inference / RAG packages (CPU-only)"
# NO torch / torchvision / torchaudio / vllm here — those caused multi-GPU
# NCCL ABI skew when this venv coexisted with the training venv (#15525,
# #20862, #28283). Inference workloads route through llama.cpp's HTTP server.
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

step "5. Smoke test (CPU-only)"
_as_user "$INFERENCE_PREFIX/venv/bin/python" - <<'PY' || warn "Inference smoke test failed."
import importlib
for mod in ("fastapi", "langchain", "sentence_transformers", "transformers", "tiktoken"):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {mod} {ver}")
    except Exception as e:
        print(f"  {mod}: import failed: {e}")
PY

log "Inference venv ready (CPU-only RAG/FastAPI): $INFERENCE_PREFIX/venv"
mark_step_ok
