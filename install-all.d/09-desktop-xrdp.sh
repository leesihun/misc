#!/usr/bin/env bash
# ============================================================================
# install-all.d/09-desktop-xrdp.sh
#
#   Configure xrdp to launch xfce4, set up polkit shutdown rules, open UFW
#   port 3389 if active. No-op if INSTALL_DESKTOP=0.
#
#   Directly runnable: sudo bash install-all.d/09-desktop-xrdp.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "09-desktop-xrdp"
detect_target_user

if [[ "$INSTALL_DESKTOP" != "1" ]] || ! command -v xrdp >/dev/null 2>&1; then
    log "INSTALL_DESKTOP=$INSTALL_DESKTOP or xrdp not installed; skipping desktop config."
    mark_step_ok
    exit 0
fi

step "1. xrdp startwm.sh"
tee /etc/xrdp/startwm.sh > /dev/null <<'XRDPEOF'
#!/bin/sh
if [ -r /etc/default/locale ]; then
    . /etc/default/locale
    export LANG LANGUAGE
fi
# Mitigate xrdp #3248 (polkit prompts): autostart polkit-gnome agent in session.
if [ -x /usr/libexec/polkit-gnome-authentication-agent-1 ]; then
    /usr/libexec/polkit-gnome-authentication-agent-1 &
elif [ -x /usr/lib/policykit-1-gnome/polkit-gnome-authentication-agent-1 ]; then
    /usr/lib/policykit-1-gnome/polkit-gnome-authentication-agent-1 &
fi
exec startxfce4
XRDPEOF
chmod +x /etc/xrdp/startwm.sh

step "2. xrdp TLS cert group + service"
adduser xrdp ssl-cert 2>/dev/null || true
systemctl enable xrdp 2>/dev/null || true
systemctl restart xrdp 2>/dev/null || warn "xrdp restart failed — run 'systemctl start xrdp' after reboot."
log "xrdp listening on port 3389"

step "3. Default xfce4 session for current + new users"
echo "xfce4-session" | tee /etc/skel/.xsession > /dev/null
if [[ -n "${SUDO_USER:-}" ]]; then
    su - "$SUDO_USER" -c "echo xfce4-session > ~/.xsession" 2>/dev/null || true
fi

step "4. polkit shutdown rule for sudo group"
if [[ -d /usr/share/polkit-1/rules.d ]]; then
    tee /usr/share/polkit-1/rules.d/49-xfce-shutdown.rules > /dev/null <<'POLKIT'
polkit.addRule(function(action, subject) {
    if ((action.id == "org.freedesktop.login1.power-off" ||
         action.id == "org.freedesktop.login1.reboot") &&
        subject.isInGroup("sudo")) {
        return polkit.Result.YES;
    }
});
POLKIT
fi

step "5. UFW port 3389"
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
    ufw allow 3389/tcp 2>/dev/null && log "UFW: port 3389/tcp opened" || true
fi

log "Desktop setup complete. Connect via RDP to port 3389."
mark_step_ok
# Reboot checkpoint — lightdm/xrdp/polkit config can hang graphical.target
# boot on a headless host (lightdm waits ~90s for an X server that never
# materializes). Reboot now to confirm boot reaches multi-user.target,
# sshd accepts logins, and xrdp listens on 3389 from a cold start.
checkpoint_reboot "desktop + xrdp configured; reboot to confirm lightdm doesn't stall boot and xrdp starts on port 3389"
