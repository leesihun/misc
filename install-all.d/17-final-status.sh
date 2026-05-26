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
REBOOT_REQUIRED=0
if [[ -f /run/reboot-required ]]; then
    REBOOT_REQUIRED=1
    log "/run/reboot-required is set (post-install)."
    [[ -f /run/reboot-required.pkgs ]] && log "  triggered by: $(tr '\n' ' ' </run/reboot-required.pkgs)"
fi

printf '\n'
printf '%s\n' "════════════════════════════════════════════════════════════════"
printf '%s\n' "  INSTALL STEPS COMPLETED (per .ok markers under $STEPS_DIR)"
printf '%s\n' "════════════════════════════════════════════════════════════════"
printf '  Inference venv : %s/venv  (CPU-only RAG/FastAPI; no torch, no vLLM)\n' "$INFERENCE_PREFIX"
printf '  Training venv  : %s/venv\n' "$TRAINING_PREFIX"
printf '  Jupyter venv   : %s/venv\n' "$JUPYTER_PREFIX"
printf '  llama-server   : %s/build/bin/llama-server\n' "$LLAMA_PREFIX"
printf '\n'
printf '  Logs dir       : %s\n' "$RUN_LOG_DIR"
printf '  Run id         : %s\n' "$RUN_ID"
printf '\n'

printf 'Next steps:\n'
printf '  bash test-nvidia.sh                            # nvidia stack still healthy?\n'
printf '  bash test-all.sh                               # full userland verify\n'
printf '  gpu-health-check                               # quick fabric sanity\n'
printf '  %s/build/bin/llama-server --help                # serve a GGUF model\n' "$LLAMA_PREFIX"
if [[ "$INSTALL_DESKTOP" == "1" ]]; then
    printf '  rdp connect — tcp 3389                         # remote desktop (xfce4)\n'
fi
printf '\n'

mark_step_ok

# Final reboot checkpoint. Strict: if any prior step set /run/reboot-required
# (e.g. via a kernel module update or a daemon refresh), exit 75 so the
# launcher loud-banners "REBOOT REQUIRED" — operator reboots, then runs
# test-nvidia.sh + test-all.sh to confirm everything survived the boot cycle.
# Even without /run/reboot-required we want one final post-tuning reboot
# checkpoint so the operator can prove sysctl + THP + pam_limits + ops units
# persist correctly from cold boot before declaring the install complete.
if (( REBOOT_REQUIRED )); then
    checkpoint_reboot "final phase complete AND /run/reboot-required is set — reboot then run test-nvidia.sh + test-all.sh"
else
    checkpoint_reboot "final phase complete — reboot to confirm sysctl/THP/limits/ops units persist from cold boot, then run test-nvidia.sh + test-all.sh"
fi
