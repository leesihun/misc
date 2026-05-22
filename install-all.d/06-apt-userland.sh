#!/usr/bin/env bash
# ============================================================================
# install-all.d/06-apt-userland.sh
#
#   apt install of toolchain, python3.12-venv/dev, needrestart, CLI tools,
#   GUI runtime libs, scientific libs, and (optionally) xfce4 + xrdp.
#
#   Directly runnable: sudo bash install-all.d/06-apt-userland.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "06-apt-userland"

step "1. Toolchain"
_apt_install_strict build-essential cmake ninja-build pkg-config git ccache curl wget ca-certificates unzip xz-utils

step "2. python${PYTHON_VER}-venv + dev"
_apt_install_strict "python${PYTHON_VER}-venv" "python${PYTHON_VER}-dev" python3-pip

step "3. needrestart"
_apt_install needrestart

step "4. CLI utilities + monitoring"
_apt_install gedit vim nano htop btop nvtop iotop tmux screen \
    net-tools iproute2 dnsutils mtr-tiny traceroute \
    jq tree ncdu zip pigz zstd rsync \
    numactl hwloc-nox

step "5. GUI runtime libs (Chrome/VS Code deps)"
_apt_install \
    libglib2.0-0t64 libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libcairo2 libcups2t64 libdbus-1-3 libdrm2 libexpat1 \
    libfontconfig1 fonts-liberation libgbm1 libgtk-3-0t64 \
    libnspr4 libnss3 libpango-1.0-0 libsecret-1-0 \
    libasound2t64 libx11-6 libx11-xcb1 libxcb1 \
    libxcomposite1 libxcursor1 libxdamage1 libxext6 \
    libxfixes3 libxi6 libxkbcommon0 libxkbfile1 \
    libxrandr2 libxrender1 libxss1 libxtst6 xdg-utils

step "6. Scientific native libs (h5py/openblas) + libssl-dev for llama.cpp OpenSSL"
_apt_install libopenblas-dev libopenblas0 libgomp1 libhdf5-dev libssl-dev libffi-dev libcurl4-openssl-dev

if [[ "$INSTALL_DESKTOP" == "1" ]]; then
    if command -v lspci >/dev/null 2>&1; then
        if ! lspci 2>/dev/null | grep -qiE '(VGA compatible controller|Display controller)'; then
            warn "No VGA/display controller detected via lspci. lightdm will still be installed (INSTALL_DESKTOP=1)."
            warn "  - lightdm.service may stall graphical.target boot by ~90s on headless hosts."
            warn "  - sshd is on multi-user.target so SSH still works, but boot is slower."
            warn "  - Re-run with INSTALL_DESKTOP=0 if you only need SSH access."
        fi
    fi
    step "7. XFCE4 + xrdp + policykit"
    _apt_install \
        xfce4 xfce4-goodies xfce4-terminal xfce4-screenshooter xfce4-taskmanager xfce4-notifyd \
        lightdm lightdm-gtk-greeter \
        xrdp xorgxrdp ssl-cert \
        policykit-1-gnome \
        dbus-x11 x11-xserver-utils x11-utils xauth xinit xterm \
        file-roller evince ristretto \
        xclip dconf-editor \
        fonts-dejavu-core fonts-noto-core fonts-noto-color-emoji \
        adwaita-icon-theme gnome-themes-extra \
        p7zip-full bash-completion
fi

mark_step_ok
