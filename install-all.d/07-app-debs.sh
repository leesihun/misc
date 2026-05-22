#!/usr/bin/env bash
# ============================================================================
# install-all.d/07-app-debs.sh
#
#   Install VS Code and Google Chrome from the bundled .debs. Reload AppArmor
#   profiles. Allow unprivileged user namespaces (Chrome/VS Code Electron
#   sandbox).
#
#   Directly runnable: sudo bash install-all.d/07-app-debs.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "07-app-debs"
locate_bundle

step "1. VS Code"
if [[ -f "$BUNDLE_DIR/apps/vscode.deb" ]]; then
    log "Installing VS Code (apt install ./)"
    apt-get install -y "$BUNDLE_DIR/apps/vscode.deb" || warn "VS Code install failed."
    command -v code >/dev/null && log "VS Code: $(code --version 2>/dev/null | head -1)" \
        || warn "VS Code installed but 'code' not on PATH."
else
    warn "apps/vscode.deb not found; skipping."
fi

step "2. Google Chrome"
if [[ -f "$BUNDLE_DIR/apps/chrome.deb" ]]; then
    log "Installing Google Chrome (apt install ./)"
    apt-get install -y "$BUNDLE_DIR/apps/chrome.deb" || warn "Chrome install failed."
    command -v google-chrome-stable >/dev/null \
        && log "Chrome: $(google-chrome-stable --version 2>/dev/null)" \
        || warn "Chrome installed but binary not in PATH."
else
    warn "apps/chrome.deb not found; skipping."
fi

step "3. AppArmor profiles"
if command -v aa-status >/dev/null 2>&1 && [[ -d /etc/apparmor.d ]]; then
    log "Reloading AppArmor profiles (registers Chrome/VS Code profiles)"
    systemctl reload apparmor 2>/dev/null || apparmor_parser -r /etc/apparmor.d/ 2>/dev/null || true
fi

step "4. Allow unprivileged user namespaces (Chrome/VS Code Electron sandbox)"
if [[ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
    log "Disabling apparmor_restrict_unprivileged_userns"
    tee /etc/sysctl.d/60-apparmor-userns.conf > /dev/null <<'SYSCTL'
# Allow unprivileged user namespaces — Chrome/VS Code/Firefox sandbox.
# Set by install-all.d/07-app-debs.sh on Ubuntu 24.04+.
kernel.apparmor_restrict_unprivileged_userns = 0
SYSCTL
    sysctl --system >/dev/null 2>&1 \
        || sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 \
        || warn "Could not apply apparmor userns sysctl."
fi

mark_step_ok
