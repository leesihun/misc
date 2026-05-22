#!/usr/bin/env bash
# ============================================================================
# install-all.d/01-preflight.sh
#
#   Re-exec under sudo, locate + verify + extract the userland bundle,
#   apply variant guard, run pre-install-check.sh.
#
#   Directly runnable:
#     sudo bash install-all.d/01-preflight.sh
#     SKIP_PREFLIGHT=1 sudo bash install-all.d/01-preflight.sh
#     FORCE=1 sudo bash install-all.d/01-preflight.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "01-preflight"

SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
FORCE="${FORCE:-0}"

step "1. Bundle locate + extract"
locate_bundle
source_bundle_metadata
detect_target_user
log "Bundle: $BUNDLE_DIR  (variant=${BUNDLE_VARIANT:-?}, target=ubuntu ${BUNDLE_OS_VERSION:-?}, built=${BUNDLE_DATE:-?})"
log "CUDA   : ${CUDA_MAJOR}.${CUDA_MINOR}"
log "Identity: $TARGET_USER:$TARGET_GROUP (uid=$EUID)"

step "2. Variant guard"
if [[ "${BUNDLE_VARIANT:-}" != "prepped" ]]; then
    if (( FORCE )); then
        warn "Bundle variant is '${BUNDLE_VARIANT:-<unset>}' (expected 'prepped'); proceeding due to --force."
    else
        die "Bundle variant is '${BUNDLE_VARIANT:-<unset>}', expected 'prepped'. This bundle is for a bare-metal server; install Ubuntu_offline_setup/install-all.sh instead. Set FORCE=1 to override."
    fi
fi
. /etc/os-release
[[ "${ID}" == "${BUNDLE_OS_ID:-}" && "${VERSION_ID}" == "${BUNDLE_OS_VERSION:-}" ]] \
    || warn "OS mismatch: target ${ID}/${VERSION_ID} vs bundle ${BUNDLE_OS_ID:-?}/${BUNDLE_OS_VERSION:-?}"

step "3. Pre-flight gate (pre-install-check.sh)"
if (( SKIP_PREFLIGHT )); then
    warn "Skipping pre-install-check.sh per SKIP_PREFLIGHT=1"
else
    PREFLIGHT=""
    for cand in "$REPO_ROOT/pre-install-check.sh" "$BUNDLE_DIR/pre-install-check.sh"; do
        [[ -r "$cand" ]] && { PREFLIGHT="$cand"; break; }
    done
    if [[ -n "$PREFLIGHT" ]]; then
        log "Running $PREFLIGHT"
        preflight_args=( --bundle "${BUNDLE_BIN:-$BUNDLE_DIR}" )
        (( FORCE )) && preflight_args+=( --force )
        if bash "$PREFLIGHT" "${preflight_args[@]}"; then
            log "Pre-flight passed."
        else
            rc=$?
            if (( FORCE )); then
                warn "Pre-flight failed (rc=$rc) but FORCE=1 was given; proceeding anyway."
            else
                die "Pre-flight failed (rc=$rc). Fix the RED findings above, or re-run with FORCE=1 / SKIP_PREFLIGHT=1."
            fi
        fi
    else
        warn "pre-install-check.sh not found. Proceeding without pre-flight gate — RECOMMEND running it first."
    fi
fi

mark_step_ok
