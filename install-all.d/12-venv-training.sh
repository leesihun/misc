#!/usr/bin/env bash
# ============================================================================
# install-all.d/12-venv-training.sh
#
#   General training venv at $TRAINING_PREFIX/venv. Installs torch (cu130),
#   PyG + extensions, and per-project requirements (MeshGraphNets, SimulGen,
#   PEMTRON) from the bundled wheelhouse.
#
#   Directly runnable: sudo bash install-all.d/12-venv-training.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "12-venv-training"
locate_bundle
detect_target_user

if [[ "$INSTALL_TRAINING" != "1" ]]; then
    log "INSTALL_TRAINING=0; skipping."
    mark_step_ok
    exit 0
fi

WHEELS_DIR="$BUNDLE_DIR/wheels/training"
if ! _wheelhouse_has_packages "$WHEELS_DIR"; then
    warn "wheels/training/ empty; skipping."
    mark_step_ok
    exit 0
fi

step "1. Create venv at $TRAINING_PREFIX/venv"
mkdir -p "$TRAINING_PREFIX"
chown "$TARGET_USER:$TARGET_GROUP" "$TRAINING_PREFIX"
_as_user "$PYTHON_BIN" -m venv "$TRAINING_PREFIX/venv" || die "Could not create training venv."

_PIP="$TRAINING_PREFIX/venv/bin/pip"

step "2. Bootstrap pip / wheel / setuptools"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
    || warn "Bootstrap pip install failed."

step "3. PyTorch (cu130) + PyG"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio \
    || warn "torch install failed."
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" torch-geometric \
    || warn "torch-geometric install failed."

for pkg in pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv; do
    if _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" "$pkg" 2>/dev/null; then
        log "  $pkg: OK"
    else
        warn "  $pkg: not available in wheelhouse (expected for torch_spline_conv on cu130)"
    fi
done

step "4. Project requirements"
for rf in "$BUNDLE_DIR/requirements/meshgraphnets.txt" \
          "$BUNDLE_DIR/requirements/simulgen.txt" \
          "$BUNDLE_DIR/requirements/pemtron.txt" \
          "$BUNDLE_DIR/requirements/pemtron_transfer.txt"; do
    [[ -f "$rf" ]] || continue
    log "  Installing from $(basename "$rf")"
    _as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
done

step "5. Core training/scientific stack"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
    numpy scipy h5py pandas tqdm matplotlib seaborn Pillow \
    scikit-learn scikit-image statsmodels networkx sympy \
    torchinfo tensorboard \
    opencv-python imageio librosa audiomentations soxr natsort \
    reportlab paramiko smbprotocol 2>/dev/null || true

step "6. Smoke test: torch + PyG"
_as_user "$TRAINING_PREFIX/venv/bin/python" - <<'PY' || warn "Training smoke test failed."
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
try:
    import torch_geometric
    print(f"  torch_geometric {torch_geometric.__version__}")
except Exception as e:
    print(f"  torch_geometric import failed: {e}")
if torch.cuda.is_available():
    print(f"  Device 0: {torch.cuda.get_device_name(0)}")
    print(f"  Device count: {torch.cuda.device_count()}")
PY

log "Training venv ready: $TRAINING_PREFIX/venv"
mark_step_ok
