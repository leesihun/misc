#!/usr/bin/env bash
# ============================================================================
# install-phase2.sh — APPS + VENVS + LLAMA.CPP phase
#
# RUN THIS AFTER install-phase1.sh AND a reboot.
#
# Pre-flight: hard-fails if the new NVIDIA driver isn't loaded (i.e. you
# forgot to reboot, or DKMS failed). This is intentional — running phase 2
# without a live driver reproduces the exact "system not yet initialized"
# bugs that the split-install is meant to avoid.
#
# What it does:
#   1. Re-applies CUDA env (idempotent) and starts FM/persistenced/DCGM
#      (these succeed now that the driver kmod is loaded)
#   2. Installs VS Code, Chrome, Firefox, Node.js, Bun, Opencode
#   3. Creates LLM inference venv (vLLM, FastAPI, RAG, torch 2.11.0+cu130)
#   4. Builds llama.cpp with CUDA + NCCL + NVLS for B300 (sm_90;100)
#   5. Builds nccl-tests (all_reduce_perf, etc.) for fabric bandwidth checks
#   6. Configures nvidia-container-toolkit runtime for docker/containerd
#   7. Applies kernel tuning (sysctl, transparent hugepages = madvise)
#   8. Drops NCCL env defaults (NVLS enabled) + system limits
#      (nofile=1M, memlock=unlimited)
#   9. Installs GPU/LLM helper scripts:
#        gpu-health-check, llama-server-multigpu, llama-model-preload
#  10. Installs llama-server@.service systemd template
#  11. Creates general training venv (torch+CUDA, PyG, MeshGraphNets, etc.)
#  12. Creates Jupyter venv
#  13. Runs post-install verification + diagnostics
#
# After this finishes (~15-25 min):
#
#   gpu-health-check                  # fabric verification (FM, NVLink, NCCL bw)
#   bash test-all.sh                  # full system test
# ============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/install-all.sh"

red()    { printf '\033[1;31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
magenta(){ printf '\033[1;35m%s\033[0m\n' "$*"; }

if [[ ! -f "$INSTALLER" ]]; then
    red "[install-phase2] install-all.sh not found next to $0"
    printf 'Both install-phase2.sh and install-all.sh must live in the same directory.\n' >&2
    exit 2
fi

# ── Pre-flight: was the system actually rebooted with a working driver? ────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    red "[install-phase2] FATAL: nvidia-smi not found."
    cat >&2 <<'NONVSMI'

The NVIDIA driver isn't installed. Either:
  - install-phase1.sh hasn't been run yet, or
  - install-phase1.sh aborted before installing the driver packages.

Run install-phase1.sh first, then reboot, then retry install-phase2.sh.

NONVSMI
    exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
    red "[install-phase2] FATAL: nvidia-smi cannot list GPUs."
    cat >&2 <<'NOREBOOT'

The NVIDIA driver is installed but the kernel module is not loaded /
not functional. Most likely cause:

  ► YOU FORGOT TO REBOOT after install-phase1.sh.

Other causes (check in this order):
  - DKMS failed to build the kmod for the running kernel:
        sudo dkms status
        sudo dkms autoinstall
  - Secure Boot is enabled and the kmod isn't signed:
        mokutil --sb-state
  - A kernel update is pending; the system is running an old kernel that
    the new driver doesn't have modules for:
        uname -r
        ls /lib/modules/$(uname -r)/updates/dkms/nvidia*.ko

Fix the underlying issue, then re-run install-phase2.sh.

NOREBOOT
    exit 1
fi

# ── Pre-flight: warn if FabricManager still isn't active on a multi-GPU box.
# Phase 2 will try to start it; surface up-front whether it's already up.
_gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if (( _gpu_count > 1 )); then
    if systemctl is-active nvidia-fabricmanager >/dev/null 2>&1; then
        magenta "[install-phase2] nvidia-fabricmanager already active across $_gpu_count GPU(s)"
    else
        yellow "[install-phase2] NOTE: nvidia-fabricmanager not active yet on $_gpu_count-GPU box."
        yellow "                Phase 2 will try to start it. If it still fails, check:"
        yellow "                    sudo journalctl -u nvidia-fabricmanager -n 80"
    fi
fi

magenta "═══ install-phase2.sh: apps + venvs + llama.cpp ═══"
printf '   Driver: %s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
printf '   GPUs:   %d × %s\n' "$_gpu_count" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
printf '\n'

exec env PHASE=2 bash "$INSTALLER" "$@"
