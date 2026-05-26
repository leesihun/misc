#!/usr/bin/env bash
# ============================================================================
# install-all.sh  (userland launcher)
#
#   Drives the install-all.d/NN-name.sh step scripts in numeric order.
#   Each step is independently runnable (just invoke `bash install-all.d/NN-...sh`);
#   this launcher exists for the common "do everything from a clean target"
#   path plus partial re-runs after a failed step.
#
#   Full target sequence (unchanged from the previous monolithic installer):
#     1. sudo bash pre-install-nvidia.sh        # NVIDIA bundle readiness
#     2. sudo bash install-nvidia.sh            # driver + FM + NVLSM + CUDA
#     3. sudo reboot                            # mandatory — loads nvidia.ko
#     4. sudo bash test-nvidia.sh               # fabric Completed?
#     5. sudo bash pre-install-check.sh         # userland readiness (gated by step 01)
#     6. sudo bash install-all.sh               # ← THIS launcher
#     7. sudo bash test-all.sh                  # post-install verification
#
#   Usage:
#     sudo bash install-all.sh                  # run every pending step
#     sudo bash install-all.sh --list           # show step status (.ok/.failed/pending)
#     sudo bash install-all.sh --run 14         # run one step (also accepts 14-llamacpp-build)
#     sudo bash install-all.sh --from 11        # run steps 11..17
#     sudo bash install-all.sh --rerun 14       # delete the .ok marker, then run 14
#     sudo bash install-all.sh --force          # ignore .ok markers, re-run everything
#     sudo bash install-all.sh --skip-preflight # skip step 01's pre-install-check.sh
#
#   Env knobs (all honored by every step script):
#     BUNDLE_DIR, BUNDLE_BIN, PYTHON_VER, SCRATCH_ROOT
#     INSTALL_INFERENCE, INSTALL_TRAINING, INSTALL_JUPYTER, INSTALL_LLAMA
#     INSTALL_DESKTOP  (inference venv is CPU-only — no INSTALL_VLLM knob)
#     CUDA_ARCH_LIST (default 100-real;103-real for B200+B300)
#     BUILD_BLAS, JOBS, FORCE, SKIP_PREFLIGHT
#     SKIP_CHECKPOINTS=1 to bypass the per-phase reboot breaks (NOT recommended)
#
#   Per-step state:
#     /var/lib/install-all/steps/NN-name.{ok,failed}
#     /var/log/install-all/<run-id>/NN-name.log
# ============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS_DIR_REPO="$SCRIPT_DIR/install-all.d"
[[ -d "$STEPS_DIR_REPO" ]] \
    || { echo "[install] install-all.d/ not found next to $0" >&2; exit 1; }

# Generate (and export) RUN_ID so every child step shares one log dir.
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
# shellcheck disable=SC1091
source "$STEPS_DIR_REPO/00-common.sh"

# ── CLI ─────────────────────────────────────────────────────────────────────
MODE="all"
ONE_STEP=""
FROM_STEP=""
RERUN_LIST=""
FORCE_ALL=0
SKIP_PREFLIGHT_ARG=0

usage() {
    sed -n '2,32p' "$0"
}

while (( $# > 0 )); do
    case "$1" in
        --list)            MODE="list"; shift ;;
        --run)             MODE="one"; ONE_STEP="$2"; shift 2 ;;
        --run=*)           MODE="one"; ONE_STEP="${1#*=}"; shift ;;
        --from)            MODE="from"; FROM_STEP="$2"; shift 2 ;;
        --from=*)          MODE="from"; FROM_STEP="${1#*=}"; shift ;;
        --rerun)           RERUN_LIST="${RERUN_LIST:+$RERUN_LIST,}$2"; shift 2 ;;
        --rerun=*)         RERUN_LIST="${RERUN_LIST:+$RERUN_LIST,}${1#*=}"; shift ;;
        --force)           FORCE_ALL=1; shift ;;
        --skip-preflight)  SKIP_PREFLIGHT_ARG=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) printf 'unknown arg: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

(( SKIP_PREFLIGHT_ARG )) && export SKIP_PREFLIGHT=1
(( FORCE_ALL ))         && export FORCE=1

# ── Step enumeration ────────────────────────────────────────────────────────
discover_steps() {
    # Returns step basenames in numeric order (00-common excluded).
    shopt -s nullglob
    for f in "$STEPS_DIR_REPO"/[0-9][0-9]-*.sh; do
        [[ "$(basename "$f")" == "00-common.sh" ]] && continue
        basename "$f" .sh
    done
    shopt -u nullglob
}

STEP_LIST=()
while IFS= read -r s; do STEP_LIST+=( "$s" ); done < <(discover_steps)
(( ${#STEP_LIST[@]} > 0 )) || die "No step scripts found in $STEPS_DIR_REPO"

# Resolve a user-supplied step token ("14" or "14-llamacpp-build") to a
# full step name; fail if none/multiple match.
resolve_step() {
    local needle="$1" hit
    for s in "${STEP_LIST[@]}"; do
        if [[ "$s" == "$needle" || "$s" == "${needle}-"* ]]; then
            echo "$s"; return 0
        fi
    done
    return 1
}

# ── Modes ───────────────────────────────────────────────────────────────────
print_step_status() {
    local s status
    printf '\n%-30s %-10s %s\n' "STEP" "STATUS" "LOG"
    printf '%-30s %-10s %s\n' "----" "------" "---"
    for s in "${STEP_LIST[@]}"; do
        if [[ -f "$STEPS_DIR/${s}.ok" ]];     then status="ok"
        elif [[ -f "$STEPS_DIR/${s}.failed" ]]; then status="FAILED"
        else                                       status="pending"
        fi
        printf '%-30s %-10s %s\n' "$s" "$status" "$RUN_LOG_DIR/${s}.log"
    done
    printf '\nLatest run-id: %s\n' "$RUN_ID"
    printf 'State dir    : %s\n' "$STATE_DIR"
    printf 'Log dir      : %s\n\n' "$LOG_ROOT"
}

if [[ "$MODE" == "list" ]]; then
    print_step_status
    exit 0
fi

# Validate step names BEFORE requiring root so a misspelling fails fast
# without prompting for sudo.
if [[ "$MODE" == "one" ]]; then
    resolve_step "$ONE_STEP" >/dev/null \
        || die "Unknown step: $ONE_STEP (run with --list to see step names)"
fi
if [[ "$MODE" == "from" ]]; then
    resolve_step "$FROM_STEP" >/dev/null \
        || die "Unknown step: $FROM_STEP (run with --list to see step names)"
fi
if [[ -n "$RERUN_LIST" ]]; then
    IFS=',' read -ra _rerun_check <<<"$RERUN_LIST"
    for tok in "${_rerun_check[@]}"; do
        resolve_step "$tok" >/dev/null \
            || die "Unknown step (--rerun): $tok (run with --list to see step names)"
    done
fi

# All other modes need root.
require_root "$@"

# Apply --rerun: delete .ok markers for those steps.
if [[ -n "$RERUN_LIST" ]]; then
    IFS=',' read -ra _rerun <<<"$RERUN_LIST"
    for tok in "${_rerun[@]}"; do
        if step=$(resolve_step "$tok"); then
            log "Re-running $step (deleting .ok marker)"
            rm -f "$STEPS_DIR/${step}.ok" "$STEPS_DIR/${step}.failed"
        else
            die "Unknown step: $tok (run with --list to see step names)"
        fi
    done
fi

# Build the actual to-run list.
TO_RUN=()
case "$MODE" in
    all)
        for s in "${STEP_LIST[@]}"; do TO_RUN+=( "$s" ); done ;;
    one)
        if step=$(resolve_step "$ONE_STEP"); then
            TO_RUN=( "$step" )
            # Honor force-via-single-step: --run NN always re-runs that step.
            rm -f "$STEPS_DIR/${step}.ok" "$STEPS_DIR/${step}.failed"
        else
            die "Unknown step: $ONE_STEP (run with --list to see step names)"
        fi ;;
    from)
        if step=$(resolve_step "$FROM_STEP"); then
            local_started=0
            for s in "${STEP_LIST[@]}"; do
                if [[ "$s" == "$step" ]]; then local_started=1; fi
                (( local_started )) && TO_RUN+=( "$s" )
            done
        else
            die "Unknown step: $FROM_STEP (run with --list to see step names)"
        fi ;;
esac

# Honor resume marker: skip 03-05 if Stage 1 already ran.
RESUMING=0
if [[ -f "$RESUME_MARKER" ]]; then
    RESUMING=1
    log "Resume marker present ($RESUME_MARKER) — skipping steps 03-05."
fi

# ── Drive each step ─────────────────────────────────────────────────────────
log "Launcher: ${MODE}  steps=${#TO_RUN[@]}  run-id=$RUN_ID"
log "Logs    : $RUN_LOG_DIR/"
log "State   : $STEPS_DIR/"

OVERALL_RC=0
SKIPPED=()
EXECUTED=()
FAILED=()
REBOOT_REQUESTED=0
REBOOT_STEP=""

for step in "${TO_RUN[@]}"; do
    # Resume-marker skip for 03/04/05.
    if (( RESUMING )); then
        case "$step" in
            03-*|04-*|05-*)
                log "Skipping $step (resume after Stage 1 reboot)."
                SKIPPED+=( "$step" )
                continue
                ;;
        esac
    fi

    # Skip already-ok steps unless --force or --rerun cleared the marker.
    if (( FORCE_ALL == 0 )) && [[ -f "$STEPS_DIR/${step}.ok" ]]; then
        log "Skipping $step (already ok — use --force or --rerun to repeat)."
        SKIPPED+=( "$step" )
        continue
    fi

    log "─── running $step ───"
    rc=0
    bash "$STEPS_DIR_REPO/${step}.sh" || rc=$?

    if (( rc == 0 )); then
        EXECUTED+=( "$step" )
    elif (( rc == 75 )); then
        # Distinguished code: any step that called checkpoint_reboot() in
        # 00-common.sh, or step 05's legacy Stage-1 reboot, or step 17's
        # /run/reboot-required gate. Always treated the same way — the step
        # already wrote its .ok marker, so re-running install-all.sh after
        # the reboot will resume at the next pending step.
        log "Step $step requested a reboot (exit 75)."
        EXECUTED+=( "$step" )
        REBOOT_REQUESTED=1
        REBOOT_STEP="$step"
        break
    else
        FAILED+=( "$step" )
        OVERALL_RC=$rc
        log "Step $step FAILED (rc=$rc). Stopping. Inspect $RUN_LOG_DIR/${step}.log"
        break
    fi
done

# ── Aggregate diagnostics ───────────────────────────────────────────────────
printf '\n'
printf '%s\n' "════════════════════════════════════════════════════════════════"
if (( REBOOT_REQUESTED )); then
    printf '  REBOOT REQUIRED — %s reached a phase boundary\n' "$REBOOT_STEP"
elif (( ${#FAILED[@]} > 0 )); then
    printf '%s\n' "  INSTALL FAILED"
else
    printf '%s\n' "  INSTALL COMPLETE"
fi
printf '%s\n' "════════════════════════════════════════════════════════════════"
printf '  Run id    : %s\n' "$RUN_ID"
printf '  Log dir   : %s\n' "$RUN_LOG_DIR"
printf '  Executed  : %s\n' "${EXECUTED[*]:-<none>}"
printf '  Skipped   : %s\n' "${SKIPPED[*]:-<none>}"
printf '  Failed    : %s\n' "${FAILED[*]:-<none>}"

# Aggregate warnings/errors recorded by step scripts.
if [[ -s "$ACC_DIR/warnings.log" ]]; then
    printf '\nWarnings:\n'
    sed 's/^/  - /' "$ACC_DIR/warnings.log"
fi
if [[ -s "$ACC_DIR/errors.log" ]]; then
    printf '\nErrors:\n'
    sed 's/^/  - /' "$ACC_DIR/errors.log"
fi

printf '\nNext steps:\n'
if (( REBOOT_REQUESTED )); then
    printf '  1. sudo reboot\n'
    printf '  2. sudo bash test-nvidia.sh                   # confirm nvidia stack alive\n'
    printf '  3. sudo bash %s                          # resume at next pending step\n' "$0"
elif (( ${#FAILED[@]} > 0 )); then
    printf '  Inspect the log for the failed step, then re-run with --rerun NN.\n'
    printf '  Example: sudo bash %s --rerun %s\n' "$0" "${FAILED[0]}"
else
    printf '  bash test-nvidia.sh                           # nvidia stack still healthy?\n'
    printf '  bash test-all.sh                              # verify everything\n'
    printf '  gpu-health-check                              # quick fabric sanity\n'
    printf '  sudo bash %s --list                       # step status\n' "$0"
fi
printf '\n'

exit "$OVERALL_RC"
