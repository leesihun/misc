#!/usr/bin/env bash
# ============================================================================
# install-all.d/04-apt-plan.sh
#
#   apt dry-run gate — surface kernel/libc/systemd/dbus upgrades the bundle
#   would pull in. Writes the proposed reboot-trigger package list to
#   $STATE_DIR/apt-proposed.txt for step 05 to consume, and refuses to proceed
#   if apt would upgrade the kernel (without FORCE=1).
#
#   Directly runnable: sudo bash install-all.d/04-apt-plan.sh
#                      FORCE=1 sudo bash install-all.d/04-apt-plan.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "04-apt-plan"
locate_bundle
source_bundle_metadata

FORCE="${FORCE:-0}"

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

REBOOT_TRIGGERS_REGEX='^(libc6|libc6-dev|systemd|systemd-sysv|dbus|dbus-daemon)$'
DKMS_DANGER_REGEX='^(linux-image-.*|linux-headers-.*|linux-firmware|microcode|intel-microcode|amd64-microcode)$'

# If Stage 1 already ran, skip the dry-run cycle.
if [[ -f "$RESUME_MARKER" ]]; then
    log "Resume marker found ($RESUME_MARKER) — Stage 1 already ran. No reboot triggers needed."
    : > "$STATE_DIR/apt-proposed.txt"
    : > "$STATE_DIR/apt-reboot-triggers.txt"
    mark_step_ok
    exit 0
fi

step "apt dry-run (detect reboot-triggering upgrades)"
SIMULATE_OUT=$(mktemp)
if ! apt-get install -s -y --no-install-recommends --allow-downgrades \
        -o APT::Get::Show-Versions=1 \
        "${REQUESTED_PKGS[@]}" 2>&1 | tee "$SIMULATE_OUT" > /dev/null; then
    warn "apt dry-run reported unmet dependencies (will retry per-package during install)."
    tail -20 "$SIMULATE_OUT" >&2 || true
fi

PROPOSED_PKGS=$(grep -E '^(Inst|Conf) ' "$SIMULATE_OUT" | awk '{print $2}' | sort -u)
printf '%s\n' "$PROPOSED_PKGS" > "$STATE_DIR/apt-proposed.txt"

DKMS_DANGER_HITS=$(printf '%s\n' "$PROPOSED_PKGS" | grep -E "$DKMS_DANGER_REGEX" || true)
if [[ -n "$DKMS_DANGER_HITS" ]] && (( FORCE == 0 )); then
    printf '\n\033[1;31m================================================================\033[0m\n'
    printf '\033[1;31m  KERNEL / FIRMWARE / MICROCODE UPGRADE DETECTED\033[0m\n'
    printf '\033[1;31m================================================================\033[0m\n'
    printf '  Apt wants to upgrade the following packages:\n'
    printf '    %s\n' $DKMS_DANGER_HITS
    printf '\n  These would replace the running kernel that NVIDIA DKMS built\n'
    printf '  nvidia.ko against. On reboot the new kernel would load WITHOUT\n'
    printf '  NVIDIA modules; nvidia-fabricmanager.service would fail.\n'
    printf '\n  To proceed safely, choose one of:\n'
    printf '    A) Hold the kernel and re-run:\n'
    printf '         sudo apt-mark hold %s\n' $DKMS_DANGER_HITS
    printf '         sudo bash install-all.sh\n'
    printf '    B) Accept the DKMS rebuild risk (verify nvidia.ko rebuilds\n'
    printf '       against the new kernel before rebooting):\n'
    printf '         sudo FORCE=1 bash install-all.sh\n\n'
    die "Refusing to upgrade kernel/firmware while NVIDIA driver is held."
elif [[ -n "$DKMS_DANGER_HITS" ]]; then
    warn "FORCE=1 set; allowing kernel/firmware upgrade. Verify DKMS rebuilds nvidia.ko before reboot."
    # Fold DKMS-danger entries into the Stage 1 install set since the user opted in.
    REBOOT_TRIGGERS_REGEX='^(libc6|libc6-dev|systemd|systemd-sysv|dbus|dbus-daemon|linux-image-.*|linux-headers-.*|linux-firmware|microcode|intel-microcode|amd64-microcode)$'
fi

REBOOT_TRIGGER_HITS=$(printf '%s\n' "$PROPOSED_PKGS" | grep -E "$REBOOT_TRIGGERS_REGEX" || true)
if [[ -n "$REBOOT_TRIGGER_HITS" ]]; then
    log "Reboot-triggering upgrades detected:"
    printf '    %s\n' $REBOOT_TRIGGER_HITS
    printf '%s\n' "$REBOOT_TRIGGER_HITS" > "$STATE_DIR/apt-reboot-triggers.txt"
else
    log "No kernel/libc/systemd/dbus upgrades — single-pass install."
    : > "$STATE_DIR/apt-reboot-triggers.txt"
fi
rm -f "$SIMULATE_OUT"

mark_step_ok
