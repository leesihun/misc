#!/usr/bin/env bash
# ============================================================================
# install-all.d/13-venv-jupyter.sh
#
#   JupyterLab + data-science venv at $JUPYTER_PREFIX/venv. Registers an
#   ipykernel and drops a start-jupyter.sh convenience launcher.
#
#   Directly runnable: sudo bash install-all.d/13-venv-jupyter.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "13-venv-jupyter"
locate_bundle
detect_target_user

if [[ "$INSTALL_JUPYTER" != "1" ]]; then
    log "INSTALL_JUPYTER=0; skipping."
    mark_step_ok
    exit 0
fi

WHEELS_DIR="$BUNDLE_DIR/wheels/jupyter"
if ! _wheelhouse_has_packages "$WHEELS_DIR"; then
    warn "wheels/jupyter/ empty; skipping."
    mark_step_ok
    exit 0
fi

step "1. Create venv at $JUPYTER_PREFIX/venv"
mkdir -p "$JUPYTER_PREFIX"
chown "$TARGET_USER:$TARGET_GROUP" "$JUPYTER_PREFIX"
_as_user "$PYTHON_BIN" -m venv "$JUPYTER_PREFIX/venv" || die "Could not create jupyter venv."

_PIP="$JUPYTER_PREFIX/venv/bin/pip"

step "2. Bootstrap pip / wheel / setuptools"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
    || warn "Bootstrap pip install failed."

step "3. JupyterLab + data-science"
_as_user "$_PIP" install --no-index --find-links="$WHEELS_DIR" \
    jupyterlab notebook ipykernel ipywidgets jupyter-server \
    pandas polars numpy scipy matplotlib seaborn plotly \
    scikit-learn statsmodels tqdm rich requests aiohttp \
    black ruff mypy pytest ipdb \
    || warn "Some Jupyter packages failed."

step "4. Register kernel"
_as_user "$JUPYTER_PREFIX/venv/bin/python" -m ipykernel install \
    --user --name "airgap-py${PYTHON_VER}" \
    --display-name "Python ${PYTHON_VER} (airgap)" 2>/dev/null || true

step "5. Convenience launcher"
if [[ -n "${SUDO_USER:-}" ]]; then
    _SUDO_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    cat > "$_SUDO_HOME/start-jupyter.sh" <<JEOF
#!/usr/bin/env bash
source "$JUPYTER_PREFIX/venv/bin/activate"
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser "\$@"
JEOF
    chmod +x "$_SUDO_HOME/start-jupyter.sh"
    chown "$TARGET_USER:$TARGET_GROUP" "$_SUDO_HOME/start-jupyter.sh"
    log "Jupyter launcher: $_SUDO_HOME/start-jupyter.sh"
fi

log "Jupyter venv ready: $JUPYTER_PREFIX/venv"
mark_step_ok
