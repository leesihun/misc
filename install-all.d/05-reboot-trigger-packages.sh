#!/usr/bin/env bash
# ============================================================================
# install-all.d/05-reboot-trigger-packages.sh
#
#   STRICT NO-OP ASSERTER.
#
#   Step 04 hard-refuses any base-OS upgrade (libc6/systemd/dbus/kernel/
#   firmware/microcode) on top of an already-installed NVIDIA driver. By the
#   time this step runs, the trigger file MUST be empty. If it isn't, an
#   earlier rev of step 04 wrote it before the strict gate landed (or someone
#   manually edited it). Either way, refuse rather than perform a base-OS
#   apt install — that's the canonical brick path on B300.
#
#   Kept as a separate step (not folded into 04) for two reasons:
#     - Back-compat with the existing $RESUME_MARKER mechanism in install-all.sh
#     - Explicit step boundary for operators reading the install log
#
#   Directly runnable: sudo bash install-all.d/05-reboot-trigger-packages.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "05-reboot-trigger-packages"

TRIGGER_FILE="$STATE_DIR/apt-reboot-triggers.txt"

# Resume marker present means a prior installer run already handled Stage 1;
# this step is a no-op on the resumed run.
if [[ -f "$RESUME_MARKER" ]]; then
    log "Resume marker present — Stage 1 already handled by a prior run."
    mark_step_ok
    exit 0
fi

# Strict: if step 04 emitted any triggers, we refuse. The current 04 should
# never produce them (it dies first), so a non-empty file here means a stale
# state directory or a hand-edited trigger list.
if [[ -s "$TRIGGER_FILE" ]]; then
    triggers=$(tr '\n' ' ' < "$TRIGGER_FILE" | sed 's/[[:space:]]*$//')
    die "Refusing to perform base-OS install. Step 04 should have refused first; trigger file '$TRIGGER_FILE' is non-empty: $triggers. Delete the trigger file ONLY after confirming step 04 is the current revision."
fi

log "No base-OS triggers — step 05 is a no-op (clean path)."
mark_step_ok
