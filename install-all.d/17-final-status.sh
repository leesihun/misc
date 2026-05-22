#!/usr/bin/env bash
# ============================================================================
# install-all.d/17-final-status.sh
#
#   Clear the resume marker, chown -R $SCRATCH_ROOT, print final diagnostics.
#   The launcher (install-all.sh) prints an aggregate summary that subsumes
#   this — this step exists so a manual `--run 17` produces the same banner.
#
#   Directly runnable: sudo bash install-all.d/17-final-status.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "17-final-status"
detect_target_user

step "1. Clear resume marker"
rm -f "$RESUME_MARKER"

step "2. chown -R $SCRATCH_ROOT"
chown -R "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT" 2>/dev/null || true

step "3. Final status"
REBOOT_RECOMMENDED=0
if [[ -f /run/reboot-required ]]; then
    REBOOT_RECOMMENDED=1
    log "/run/reboot-required is set (post-install)."
    [[ -f /run/reboot-required.pkgs ]] && log "  triggered by: $(tr '\n' ' ' </run/reboot-required.pkgs)"
fi

printf '\n'
printf '%s\n' "════════════════════════════════════════════════════════════════"
printf '%s\n' "  INSTALL STEPS COMPLETED (per .ok markers under $STEPS_DIR)"
printf '%s\n' "════════════════════════════════════════════════════════════════"
printf '  Inference venv : %s/venv\n' "$INFERENCE_PREFIX"
printf '  Training venv  : %s/venv\n' "$TRAINING_PREFIX"
printf '  Jupyter venv   : %s/venv\n' "$JUPYTER_PREFIX"
printf '  llama-server   : %s/build/bin/llama-server\n' "$LLAMA_PREFIX"
printf '\n'
printf '  Logs dir       : %s\n' "$RUN_LOG_DIR"
printf '  Run id         : %s\n' "$RUN_ID"
printf '\n'

printf 'Next steps:\n'
printf '  bash test-all.sh                              # verify everything\n'
printf '  gpu-health-check                              # quick fabric sanity\n'
printf '  source %s/venv/bin/activate                    # use inference venv\n' "$INFERENCE_PREFIX"
printf '  %s/build/bin/llama-server --help               # serve a GGUF model\n' "$LLAMA_PREFIX"
if [[ "$INSTALL_DESKTOP" == "1" ]]; then
    printf '  rdp connect — tcp 3389                         # remote desktop (xfce4)\n'
fi
printf '\n'

if (( REBOOT_RECOMMENDED )); then
    printf '\033[1;33mAdvisory: /run/reboot-required is set. Reboot recommended:\033[0m\n'
    printf '  sudo reboot\n\n'
fi

mark_step_ok
