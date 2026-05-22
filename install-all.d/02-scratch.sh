#!/usr/bin/env bash
# ============================================================================
# install-all.d/02-scratch.sh
#
#   Create $SCRATCH_ROOT (default /scratch) and chown to target user.
#
#   Directly runnable: sudo bash install-all.d/02-scratch.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "02-scratch"
detect_target_user

step "Scratch directory: $SCRATCH_ROOT"
if [[ ! -d "$SCRATCH_ROOT" ]]; then
    mkdir -p "$SCRATCH_ROOT" || die "Could not create $SCRATCH_ROOT"
fi
chown "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT" || warn "chown $SCRATCH_ROOT failed."
chmod 0775 "$SCRATCH_ROOT" 2>/dev/null || true
log "$SCRATCH_ROOT ready (owner $TARGET_USER:$TARGET_GROUP)"

mark_step_ok
