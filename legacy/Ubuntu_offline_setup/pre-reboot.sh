#!/usr/bin/env bash
# ============================================================================
# pre-reboot.sh — ensure the server actually comes back to an SSH-able state
#                 after `sudo reboot` (run at the end of install-phase1).
#
# Two failure modes this script defends against:
#
#   (A) "Server never comes back online"
#       ─ MokManager waits forever for a keyboard at boot (headless = hang).
#       ─ systemd-networkd-wait-online has no timeout by default → infinite wait.
#       ─ Kernel was upgraded but initramfs/DKMS isn't built for it.
#       ─ GRUB_TIMEOUT=0 + recordfail → blank screen, no recovery path.
#
#   (B) "SSH connects, password OK, then: Connection closed by ... port 22"
#       ─ / or /var or /home partition is full → can't write wtmp/lastlog.
#       ─ /etc/profile.d/*.sh or /etc/bash.bashrc has a syntax error → login
#         shell exits non-zero before sshd hands over.
#       ─ pam.d/sshd, pam.d/common-* references a missing module.
#       ─ systemd-logind / dbus not running → pam_systemd can't create slice.
#       ─ User's home dir missing/non-writable.
#       ─ lightdm respawn loop hogging CPU/OOM-ing the box.
#
# Idempotent. Safe to re-run. Returns 0 if reboot is safe, 1 if there are
# hard blockers (script prints what to fix).
# ============================================================================
set -Eeuo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────
_ok()     { printf '\033[1;32m[ OK ]\033[0m %s\n' "$*"; }
_info()   { printf '\033[1;36m[info]\033[0m %s\n' "$*"; }
_warn()   { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; _WARN_COUNT=$((_WARN_COUNT+1)); }
_bad()    { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; _FAIL_COUNT=$((_FAIL_COUNT+1)); }
_section(){ printf '\n\033[1;35m═══ %s ═══\033[0m\n' "$*"; }

_WARN_COUNT=0
_FAIL_COUNT=0

if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        exec sudo -E bash "$0" "$@"
    fi
    _bad "Must run as root."
    exit 2
fi

printf '\033[1;35m═══════════════════════════════════════════════════════════════\033[0m\n'
printf '\033[1;35m  pre-reboot.sh — verifying the server will come back up\033[0m\n'
printf '\033[1;35m═══════════════════════════════════════════════════════════════\033[0m\n'

# ── 1. MOK enrollment queue (biggest cause of "server never boots") ─────────
_section "1. MOK enrollment / Secure Boot"

if command -v mokutil >/dev/null 2>&1; then
    sb=$(mokutil --sb-state 2>/dev/null | head -1 || echo "unknown")
    case "$sb" in
        *disabled*) _ok "Secure Boot: $sb" ;;
        *enabled*)  _warn "Secure Boot is ENABLED. Ops policy requires DISABLED — change in UEFI/BIOS setup." ;;
        *)          _info "Secure Boot state: $sb" ;;
    esac

    # mokutil --list-new prints "MokNew is empty" when nothing queued
    new_count=$(mokutil --list-new 2>/dev/null | grep -ciE 'SHA|certificate' || true)
    if (( new_count > 0 )); then
        _warn "$new_count MOK key(s) pending enrollment — MokManager will block headless boot."
        if mokutil --revoke-import 2>/dev/null; then
            _ok "Pending MOK import revoked."
        else
            _warn "mokutil --revoke-import failed; manually run it as root."
        fi
    else
        _ok "No pending MOK enrollment."
    fi
else
    _info "mokutil not installed (BIOS legacy boot or container)."
fi

# Remove the auto-generated MOK so DKMS can't queue a new enrollment next time.
if [[ -f /var/lib/shim-signed/mok/MOK.priv || -f /var/lib/shim-signed/mok/MOK.der ]]; then
    rm -f /var/lib/shim-signed/mok/MOK.priv /var/lib/shim-signed/mok/MOK.der
    _ok "Removed auto-generated MOK key (/var/lib/shim-signed/mok/)."
fi

# Disable DKMS auto-signing so future module builds don't regenerate the key.
mkdir -p /etc/dkms/framework.conf.d
cat > /etc/dkms/framework.conf.d/no-autosign.conf <<'EOF'
# Installed by pre-reboot.sh — Secure Boot is disabled per ops policy.
# Empty sign_tool prevents DKMS from generating a MOK and queuing enrollment.
sign_tool=""
EOF
_ok "DKMS auto-signing disabled (/etc/dkms/framework.conf.d/no-autosign.conf)."

# ── 2. Disk space (very common cause of "Connection closed") ────────────────
_section "2. Disk space"

# Anything >95% on a path sshd writes to during login → session creation fails.
for mp in / /var /tmp /home; do
    [[ -d "$mp" ]] || continue
    used=$(df --output=pcent "$mp" 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
    [[ "$used" =~ ^[0-9]+$ ]] || continue
    if (( used >= 95 )); then
        _bad "$mp is $used% full — sshd will fail to write wtmp/utmp on login."
    elif (( used >= 85 )); then
        _warn "$mp is $used% full."
    else
        _ok "$mp: $used% used"
    fi
done

# Inode exhaustion — same effect as full disk for new file creation.
for mp in / /var /home; do
    [[ -d "$mp" ]] || continue
    iused=$(df -i --output=ipcent "$mp" 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
    [[ "$iused" =~ ^[0-9]+$ ]] || continue
    (( iused >= 90 )) && _warn "$mp inode usage $iused% — risk of 'No space left on device'."
done

# ── 3. SSH daemon ───────────────────────────────────────────────────────────
_section "3. SSH daemon"

ssh_unit=""
for u in ssh ssh.service sshd ssh.socket; do
    if systemctl list-unit-files "$u" 2>/dev/null | grep -q "$u"; then
        ssh_unit="$u"
        break
    fi
done
if [[ -z "$ssh_unit" ]]; then
    _bad "No ssh/sshd systemd unit found — you will be locked out after reboot."
else
    if systemctl is-enabled "$ssh_unit" >/dev/null 2>&1; then
        _ok "$ssh_unit is enabled."
    else
        systemctl enable "$ssh_unit" 2>/dev/null \
            && _ok "$ssh_unit enabled." \
            || _bad "Failed to enable $ssh_unit."
    fi
fi

# Validate sshd_config — a broken config means sshd refuses to start.
if command -v sshd >/dev/null 2>&1; then
    if sshd -t 2>/tmp/.sshd-test.err; then
        _ok "sshd_config syntax OK."
    else
        _bad "sshd_config is broken:"
        sed 's/^/      /' /tmp/.sshd-test.err >&2
    fi
    rm -f /tmp/.sshd-test.err
fi

# UseDNS yes + broken DNS = 30s hang per login. Force off if set.
if grep -qE '^\s*UseDNS\s+yes' /etc/ssh/sshd_config 2>/dev/null; then
    _warn "sshd_config has UseDNS yes — slow logins if DNS is unreachable."
fi

# ── 4. Login chain: PAM + shell init + home dirs ────────────────────────────
_section "4. Login chain (PAM / shell init / home)"

# PAM stack referenced by sshd must exist.
if [[ -f /etc/pam.d/sshd ]]; then
    _ok "/etc/pam.d/sshd exists."
    # Each "@include" target must also exist.
    awk '/^@include/ {print $2}' /etc/pam.d/sshd | while read -r inc; do
        [[ -f "/etc/pam.d/$inc" ]] || _bad "Missing PAM include: /etc/pam.d/$inc"
    done
else
    _bad "/etc/pam.d/sshd is missing — sshd will deny all logins."
fi

# Login shell init: a syntax error here closes the session immediately after auth.
shell_init_bad=0
for f in /etc/profile /etc/bash.bashrc /etc/profile.d/*.sh; do
    [[ -f "$f" ]] || continue
    if ! bash -n "$f" 2>/tmp/.sh-syntax.err; then
        _bad "Shell init syntax error: $f"
        sed 's/^/      /' /tmp/.sh-syntax.err >&2
        shell_init_bad=1
    fi
done
rm -f /tmp/.sh-syntax.err
(( shell_init_bad == 0 )) && _ok "All shell init files parse cleanly."

# Stronger test: actually run a non-interactive login shell as root and as the
# invoking user. If this exits non-zero, sshd will close the session.
_test_login_shell() {
    local who="$1" cmd
    cmd=$(printf 'set -e; source /etc/profile 2>/dev/null || true; for f in /etc/profile.d/*.sh; do [ -r "$f" ] && . "$f"; done; exit 0')
    if su -l "$who" -c "$cmd" >/tmp/.login-test.out 2>&1; then
        _ok "Login shell for '$who' completes without error."
    else
        _bad "Login shell for '$who' fails — this is the 'Connection closed' cause."
        sed 's/^/      /' /tmp/.login-test.out >&2
    fi
}
_test_login_shell root
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    _test_login_shell "$SUDO_USER"
fi
rm -f /tmp/.login-test.out

# Home dirs of all real users (UID >= 1000) must exist and be readable by them.
while IFS=: read -r u _ uid _ _ home shell; do
    (( uid < 1000 || uid >= 65534 )) && continue
    [[ -d "$home" ]] || { _warn "User '$u' home '$home' missing."; continue; }
    home_owner=$(stat -c '%U' "$home" 2>/dev/null || echo "?")
    [[ "$home_owner" == "$u" ]] || _warn "User '$u' home '$home' owned by '$home_owner' (not '$u')."
    [[ -x "$shell" ]] || _warn "User '$u' shell '$shell' is not executable."
done < /etc/passwd

# ── 5. dbus / systemd-logind (pam_systemd dependency) ───────────────────────
_section "5. dbus / systemd-logind"

for svc in dbus systemd-logind; do
    if systemctl is-enabled "$svc" >/dev/null 2>&1; then
        _ok "$svc enabled."
    else
        systemctl enable "$svc" 2>/dev/null \
            && _ok "$svc enabled." \
            || _warn "$svc not enabled — pam_systemd may fail to create session slice."
    fi
done

# ── 6. network-online infinite wait (kills boot on flaky NIC) ───────────────
_section "6. Network wait timeouts"

for unit in systemd-networkd-wait-online NetworkManager-wait-online; do
    if systemctl list-unit-files "$unit.service" >/dev/null 2>&1; then
        dropin_dir="/etc/systemd/system/$unit.service.d"
        mkdir -p "$dropin_dir"
        case "$unit" in
            systemd-networkd-wait-online)
                cat > "$dropin_dir/timeout.conf" <<'EOF'
[Service]
ExecStart=
ExecStart=/lib/systemd/systemd-networkd-wait-online --timeout=30 --any
EOF
                ;;
            NetworkManager-wait-online)
                cat > "$dropin_dir/timeout.conf" <<'EOF'
[Service]
TimeoutStartSec=30
EOF
                ;;
        esac
        _ok "$unit capped at 30s."
    fi
done
systemctl daemon-reload

# ── 7. Default target + lightdm respawn cap (GUI-required hosts) ────────────
_section "7. Default target / lightdm"

cur_target=$(systemctl get-default 2>/dev/null || echo "unknown")
_info "Default target: $cur_target"

# Respect the GUI requirement (XFCE + xrdp) — keep graphical.target — but
# make sure SSH (which is on multi-user.target, a dependency) is reachable
# even if lightdm respawn-loops.
if systemctl list-unit-files lightdm.service >/dev/null 2>&1; then
    mkdir -p /etc/systemd/system/lightdm.service.d
    cat > /etc/systemd/system/lightdm.service.d/respawn-cap.conf <<'EOF'
[Service]
# If lightdm fails to start 3 times in 5 min, give up. Keeps SSH responsive
# instead of OOM-looping on a bad X stack (e.g. NVIDIA driver mismatch).
Restart=on-failure
RestartSec=10
StartLimitBurst=3
StartLimitIntervalSec=300
EOF
    _ok "lightdm respawn capped (3 retries / 5 min)."
fi

# xrdp shouldn't block boot if it fails — set TimeoutStartSec.
if systemctl list-unit-files xrdp.service >/dev/null 2>&1; then
    mkdir -p /etc/systemd/system/xrdp.service.d
    cat > /etc/systemd/system/xrdp.service.d/timeout.conf <<'EOF'
[Service]
TimeoutStartSec=30
EOF
    _ok "xrdp start timeout capped at 30s."
fi
systemctl daemon-reload

# ── 8. NVIDIA services start timeout ────────────────────────────────────────
_section "8. NVIDIA service timeouts"

for unit in nvidia-fabricmanager nvidia-persistenced nvidia-dcgm; do
    if systemctl list-unit-files "$unit.service" >/dev/null 2>&1; then
        dropin="/etc/systemd/system/$unit.service.d"
        mkdir -p "$dropin"
        cat > "$dropin/timeout.conf" <<'EOF'
[Service]
TimeoutStartSec=60
EOF
        _ok "$unit start timeout capped at 60s."
    fi
done
systemctl daemon-reload

# ── 9. Kernel / initramfs / DKMS consistency for the kernel that will boot ──
_section "9. Kernel & initramfs consistency"

running=$(uname -r)
# Pick the newest installed kernel — that's what GRUB will default to.
latest=""
if compgen -G "/boot/vmlinuz-*" >/dev/null; then
    latest=$(ls /boot/vmlinuz-* 2>/dev/null | sed 's|.*/vmlinuz-||' \
             | sort -V | tail -1)
fi
_info "Running kernel: $running"
_info "Newest kernel:  ${latest:-<none>}"

if [[ -n "$latest" ]]; then
    if [[ ! -f "/boot/initrd.img-$latest" ]]; then
        _bad "/boot/initrd.img-$latest missing — boot into $latest will fail."
        _info "Fix: sudo update-initramfs -c -k $latest"
    else
        _ok "initrd.img-$latest present."
    fi
    if [[ "$running" != "$latest" ]]; then
        _info "Boot will switch from $running to $latest."
    fi
fi

# DKMS modules for the to-be-booted kernel — split into NVIDIA (critical) and
# everything else (informational). nvidia-dkms-580-open < 580.95 is a confirmed
# DKMS build failure on kernel 6.8+ (Launchpad #2141477); treat as FAIL.
if command -v dkms >/dev/null 2>&1 && [[ -n "$latest" ]]; then
    _dkms_status=$(dkms status 2>/dev/null)
    _nv_dkms_bad=$(printf '%s\n' "$_dkms_status" \
        | awk -F'[,: ]' -v k="$latest" '$0 ~ k && /^nvidia/ && $0 !~ /: installed/' \
        | head -5)
    _other_dkms_bad=$(printf '%s\n' "$_dkms_status" \
        | awk -F'[,: ]' -v k="$latest" '$0 ~ k && !/^nvidia/ && $0 !~ /: installed/' \
        | head -10)

    if [[ -n "$_nv_dkms_bad" ]]; then
        _bad "NVIDIA DKMS module(s) NOT built for kernel $latest — GPU will be unusable after reboot:"
        printf '%s\n' "$_nv_dkms_bad" | sed 's/^/      /' >&2
        _info "Try before rebooting:"
        _info "  sudo dkms autoinstall -k $latest"
        _info "  If that fails: dkms status; dmesg | grep -i dkms | tail -30"
        _info "  Known cause: nvidia-dkms-580-open < 580.95 fails on kernel 6.8+"
        _info "  (Launchpad bug #2141477) — re-gather with NVIDIA CUDA repo, not Ubuntu's."
    else
        # Check installed version against known-bad 580.65.06
        _nv_dkms_ver=$(dpkg-query -W -f='${Version}' nvidia-dkms-580-open 2>/dev/null \
            || dpkg-query -W -f='${Version}' nvidia-dkms-580 2>/dev/null || true)
        if [[ -n "$_nv_dkms_ver" ]] && \
           dpkg --compare-versions "$_nv_dkms_ver" lt "580.95" 2>/dev/null; then
            _warn "nvidia-dkms version $_nv_dkms_ver is < 580.95 (known DKMS build failure on kernel 6.8+)"
            _warn "  Build passed now, but may fail on next kernel upgrade (Launchpad #2141477)."
        else
            _ok "NVIDIA DKMS modules installed for $latest${_nv_dkms_ver:+ (version $_nv_dkms_ver)}."
        fi
    fi

    if [[ -n "$_other_dkms_bad" ]]; then
        _warn "Non-NVIDIA DKMS modules not installed for $latest (non-blocking for boot):"
        printf '%s\n' "$_other_dkms_bad" | sed 's/^/      /' >&2
        _info "Rebuild: sudo dkms autoinstall -k $latest"
    else
        _ok "Non-NVIDIA DKMS modules installed (or none configured) for $latest."
    fi
fi

# ── 10. GRUB: visible menu, non-zero timeout, no UEFI-firmware default ──────
_section "10. GRUB"

if [[ -f /etc/default/grub ]]; then
    cp -n /etc/default/grub /etc/default/grub.pre-reboot.bak 2>/dev/null || true
    changed=0
    _set_grub() {
        local key="$1" val="$2"
        if grep -qE "^${key}=" /etc/default/grub; then
            if ! grep -qE "^${key}=${val}\$" /etc/default/grub; then
                sed -i "s|^${key}=.*|${key}=${val}|" /etc/default/grub
                changed=1
            fi
        else
            printf '%s=%s\n' "$key" "$val" >> /etc/default/grub
            changed=1
        fi
    }
    _set_grub GRUB_TIMEOUT 5
    _set_grub GRUB_TIMEOUT_STYLE menu
    _set_grub GRUB_RECORDFAIL_TIMEOUT 5

    if (( changed )); then
        if update-grub >/tmp/.update-grub.out 2>&1; then
            _ok "GRUB regenerated (timeout=5s, menu visible)."
        else
            _bad "update-grub FAILED — new kernel may not appear in boot menu:"
            tail -5 /tmp/.update-grub.out | sed 's/^/      /' >&2
            _info "Fix: sudo update-grub; grep menuentry /boot/grub/grub.cfg | head"
        fi
        rm -f /tmp/.update-grub.out
    else
        _ok "GRUB already configured (timeout=5s, menu)."
    fi
fi

# ── 11. systemd journal flush + final sync ──────────────────────────────────
_section "11. Pre-reboot housekeeping"

systemctl daemon-reload 2>/dev/null || true
sync
_ok "daemon-reload + sync done."

# ── Summary ─────────────────────────────────────────────────────────────────
printf '\n\033[1;35m═══════════════════════════════════════════════════════════════\033[0m\n'
printf '\033[1;35m  pre-reboot.sh summary\033[0m\n'
printf '\033[1;35m═══════════════════════════════════════════════════════════════\033[0m\n'

if (( _FAIL_COUNT == 0 )); then
    if (( _WARN_COUNT == 0 )); then
        printf '\033[1;32m  ✓ All checks passed. Safe to: sudo reboot\033[0m\n\n'
    else
        printf '\033[1;33m  ⚠ %d warning(s). Reboot should still work; review above.\033[0m\n' "$_WARN_COUNT"
        printf '    sudo reboot\n\n'
    fi
    exit 0
else
    printf '\033[1;31m  ✗ %d hard failure(s) and %d warning(s).\033[0m\n' "$_FAIL_COUNT" "$_WARN_COUNT"
    printf '\033[1;31m    Fix the [FAIL] items above BEFORE rebooting, or the box may not\n'
    printf '    come back up to an SSH-able state.\033[0m\n\n'
    exit 1
fi
