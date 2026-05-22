#!/usr/bin/env bash
# ============================================================================
# install-all.d/05-reboot-trigger-packages.sh
#
#   Conditional Stage 1: install only the reboot-triggering packages
#   (libc6/systemd/dbus/optionally kernel) detected in step 04, then write
#   the resume marker and ask for a reboot.
#
#   Idempotent — if step 04 produced an empty trigger list, this step does
#   nothing and just records ok.
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

if [[ -f "$RESUME_MARKER" ]]; then
    log "Resume marker present — Stage 1 has already run."
    mark_step_ok
    exit 0
fi

if [[ ! -s "$TRIGGER_FILE" ]]; then
    log "No reboot triggers (see step 04). Skipping Stage 1."
    mark_step_ok
    exit 0
fi

# Read triggers; multiple packages whitespace-separated.
REBOOT_TRIGGER_HITS=$(tr '\n' ' ' < "$TRIGGER_FILE")

step "Stage 1: install reboot-triggering packages"
log "Installing: $REBOOT_TRIGGER_HITS"
# shellcheck disable=SC2086
apt-get install -y --no-install-recommends --allow-downgrades $REBOOT_TRIGGER_HITS \
    || die "Stage 1 install failed; aborting before reboot."

# Run needrestart -r a in case the upgrade didn't actually require a reboot
# after dependency resolution (sometimes apt's simulation is conservative).
if command -v needrestart >/dev/null 2>&1; then
    needrestart -r a -q 2>&1 | tail -30 || true
fi

install -d -m 0755 "$(dirname "$RESUME_MARKER")"
touch "$RESUME_MARKER"

if [[ ! -f /run/reboot-required ]]; then
    log "Stage 1 complete; /run/reboot-required not set. Continuing in same run."
    mark_step_ok
    exit 0
fi

cat <<EOM

==============================================================================
  Stage 1 complete. The system upgraded packages that require a reboot:
  $(cat /run/reboot-required.pkgs 2>/dev/null | tr '\n' ' ')

  ACTION REQUIRED:
    1. sudo reboot
    2. After reboot, re-run: sudo bash install-all.sh
       (the resume marker at $RESUME_MARKER is now set; steps 03–05
        will short-circuit and the install picks up at step 06.)
==============================================================================

EOM

mark_step_ok
# Exit with a distinguished code so the launcher can render the reboot prompt.
exit 75
