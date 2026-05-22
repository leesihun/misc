#!/usr/bin/env bash
# ============================================================================
# install-phase1.sh — DRIVER + SYSTEM phase
#
# RUN THIS FIRST on the airgapped server.
#
# What it does:
#   1. Auto-extracts the airgap bundle (if needed)
#   2. Purges any existing CUDA toolkit
#   3. Installs all bundled .deb packages (~1200): CUDA 13.0 toolkit,
#      NVIDIA driver 580, NCCL, DCGM, system libs, fonts, GUI runtime deps
#   4. Builds the NVIDIA DKMS kernel modules for the running kernel
#   5. Sets up CUDA env (/etc/profile.d/cuda.sh, /etc/bash.bashrc patch)
#   6. Enables nvidia-fabricmanager / nvidia-persistenced / nvidia-dcgm services
#      (most will FAIL to start now — that's expected; the new kmod isn't
#      loaded yet — they will auto-start on reboot)
#
# After this finishes (~20-40 min):
#
#   sudo reboot                       # load the new driver kmod
#   nvidia-smi                        # verify: 8x B300 visible
#   bash install-phase2.sh            # finish the install
#
# DO NOT skip the reboot. Phase 2 hard-fails if the new driver isn't loaded.
# ============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/install-all.sh"

if [[ ! -f "$INSTALLER" ]]; then
    printf '\033[1;31m[install-phase1] install-all.sh not found next to %s\033[0m\n' "$0" >&2
    printf 'Both install-phase1.sh and install-all.sh must live in the same directory.\n' >&2
    exit 2
fi

printf '\033[1;35m═══ install-phase1.sh: drivers + system packages ═══\033[0m\n'
printf '   After this finishes:  sudo reboot  &&  bash install-phase2.sh\n\n'

exec env PHASE=1 bash "$INSTALLER" "$@"
