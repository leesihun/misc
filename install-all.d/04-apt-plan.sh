#!/usr/bin/env bash
# ============================================================================
# install-all.d/04-apt-plan.sh
#
#   STRICT base-OS gate. Runs an apt dry-run to confirm the userland bundle
#   would NOT upgrade libc6 / systemd / dbus / kernel / firmware / microcode
#   on the target. Any such trigger is a HARD FAIL (no FORCE escape) — those
#   packages must already be at the right baseline before install-nvidia.sh
#   ran. Touching them post-driver is the canonical brick path on B300.
#
#   For the all-clear case, writes empty trigger lists for back-compat with
#   step 05 (which now also strict-asserts they're empty).
#
#   Directly runnable: sudo bash install-all.d/04-apt-plan.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "04-apt-plan"
locate_bundle
source_bundle_metadata

APT_PKGS_FILE="$BUNDLE_DIR/meta/apt-packages.txt"
[[ -f "$APT_PKGS_FILE" ]] || die "$APT_PKGS_FILE not found in bundle."

# Build the install list, mapping pre-t64 names to Ubuntu 24.04 names.
REQUESTED_PKGS=()
while IFS= read -r pkg; do
    [[ -n "$pkg" && "$pkg" != \#* ]] || continue
    REQUESTED_PKGS+=( "$(_normalize_pkg_name "$pkg")" )
done < "$APT_PKGS_FILE"
(( ${#REQUESTED_PKGS[@]} > 0 )) || die "$APT_PKGS_FILE did not contain any installable package names."

install -d -m 0755 "$STATE_DIR"
printf '%s\n' "${REQUESTED_PKGS[@]}" > "$STATE_DIR/apt-requested.txt"
log "Requested packages: ${#REQUESTED_PKGS[@]} (see $STATE_DIR/apt-requested.txt)"

# Anything in this set, if upgraded post-NVIDIA, can desync the driver/kernel
# ABI or restart PID-1-adjacent daemons. We refuse ANY of these.
BASE_OS_DANGER_REGEX='^(libc6|libc6-dev|systemd|systemd-sysv|dbus|dbus-daemon|linux-image-.*|linux-headers-.*|linux-firmware|microcode|intel-microcode|amd64-microcode)$'

# Honor the legacy resume marker — if a prior installer wrote it, Stage 1
# already happened in some earlier run; just clear the trigger file and pass.
if [[ -f "$RESUME_MARKER" ]]; then
    log "Resume marker present ($RESUME_MARKER) — base-OS work assumed complete from a prior run."
    : > "$STATE_DIR/apt-proposed.txt"
    : > "$STATE_DIR/apt-reboot-triggers.txt"
    mark_step_ok
    exit 0
fi

step "apt dry-run (strict base-OS gate)"
# NOTE: do NOT set a `trap '... rm SIMULATE_OUT' EXIT` here — init_step in
# 00-common.sh already installed an EXIT trap (`_step_on_exit`) that writes
# the .failed marker on non-zero exit. A second `trap EXIT` would silently
# replace it. Clean up the tempfile inline at each exit path instead.
SIMULATE_OUT=$(mktemp)
if ! apt-get install -s -y --no-install-recommends --allow-downgrades \
        -o APT::Get::Show-Versions=1 \
        "${REQUESTED_PKGS[@]}" 2>&1 | tee "$SIMULATE_OUT" > /dev/null; then
    # Strict: an apt simulation that exits non-zero means dependency
    # resolution failed BEFORE we even got to the trigger check. Don't
    # paper over this with per-package retries later; surface it now.
    tail -40 "$SIMULATE_OUT" >&2 || true
    sim_copy="$STATE_DIR/apt-dry-run-fail.log"
    cp -f "$SIMULATE_OUT" "$sim_copy" 2>/dev/null || true
    rm -f "$SIMULATE_OUT"
    die "apt dry-run failed — dependency resolution broken in this bundle. Last 40 lines above; full output saved to $sim_copy. Rebuild gather-all.sh."
fi

PROPOSED_PKGS=$(grep -E '^(Inst|Conf) ' "$SIMULATE_OUT" | awk '{print $2}' | sort -u)
printf '%s\n' "$PROPOSED_PKGS" > "$STATE_DIR/apt-proposed.txt"
rm -f "$SIMULATE_OUT"

BASE_OS_HITS=$(printf '%s\n' "$PROPOSED_PKGS" | grep -E "$BASE_OS_DANGER_REGEX" || true)
if [[ -n "$BASE_OS_HITS" ]]; then
    printf '\n\033[1;31m================================================================\033[0m\n'
    printf '\033[1;31m  BASE-OS UPGRADE DETECTED — INSTALL REFUSED (no FORCE escape)\033[0m\n'
    printf '\033[1;31m================================================================\033[0m\n'
    printf '  The userland bundle would upgrade the following base-OS packages:\n'
    printf '    %s\n' $BASE_OS_HITS
    printf '\n  These touch the libc/systemd/kernel/firmware ABI surface that\n'
    printf '  the already-installed NVIDIA driver (R%s) was matched against\n' "${DRIVER_BRANCH:-580}"
    printf '  in install-nvidia.sh. Upgrading them now is the canonical brick\n'
    printf '  path on B300 (peermem ABI break, FM "system not initialized",\n'
    printf '  unbootable kernel without nvidia.ko).\n'
    printf '\n  Required recovery (in this order):\n'
    printf '    1. Coordinate with the server vendor to update the BASE OS to\n'
    printf '       the level matching this bundle (or rebuild the userland\n'
    printf '       bundle on a gather host that matches the target baseline).\n'
    printf '    2. Reboot into the updated baseline.\n'
    printf '    3. Re-run the FULL sequence from pre-install-nvidia.sh.\n'
    printf '\n  (Trigger list saved to %s for reference.)\n\n' "$STATE_DIR/apt-base-os-hits.txt"
    printf '%s\n' "$BASE_OS_HITS" > "$STATE_DIR/apt-base-os-hits.txt"
    die "Refusing to upgrade base-OS packages on top of an already-installed NVIDIA driver."
fi

# Clean path — confirm the legacy trigger files are empty so step 05 sees
# nothing to do.
log "No base-OS upgrades pending — clean apt plan."
: > "$STATE_DIR/apt-reboot-triggers.txt"

mark_step_ok
