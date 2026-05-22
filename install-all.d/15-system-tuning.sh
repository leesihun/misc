#!/usr/bin/env bash
# ============================================================================
# install-all.d/15-system-tuning.sh
#
#   sysctl + Transparent Huge Pages + pam_limits tuning for multi-GPU LLM
#   workloads. Note: CUDA env wiring (/etc/profile.d/cuda.sh,
#   /etc/ld.so.conf.d/cuda-system.conf) lives in install-nvidia.sh — those
#   are nvidia infrastructure, not userland tuning.
#
#   Directly runnable: sudo bash install-all.d/15-system-tuning.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "15-system-tuning"

step "1. sysctl (vm.overcommit, vm.swappiness, vm.max_map_count, net buffers)"
tee /etc/sysctl.d/99-llm-multigpu.conf > /dev/null <<'SYSCTL'
# Installed by install-all.d/15-system-tuning.sh — multi-GPU LLM workload tuning.
vm.overcommit_memory=1
vm.swappiness=0
vm.max_map_count=1048576
net.core.rmem_max=268435456
net.core.wmem_max=268435456
SYSCTL
sysctl --system >/dev/null 2>&1 || warn "sysctl --system reload failed."
log "sysctl applied"

step "2. Transparent hugepages = madvise"
tee /etc/systemd/system/disable-thp-defrag.service > /dev/null <<'UNIT'
[Unit]
Description=Set transparent_hugepage to madvise for LLM workloads
After=sysinit.target local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo madvise > /sys/kernel/mm/transparent_hugepage/enabled; echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag"
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload 2>/dev/null || true
systemctl enable --now disable-thp-defrag.service >/dev/null 2>&1 \
    && log "THP = madvise" \
    || warn "Could not enable disable-thp-defrag.service."

step "3. pam_limits (nofile, nproc, memlock, stack)"
tee /etc/security/limits.d/99-llm-multigpu.conf > /dev/null <<'LIMITS'
*  soft  nofile   1048576
*  hard  nofile   1048576
*  soft  nproc    524288
*  hard  nproc    524288
*  soft  memlock  unlimited
*  hard  memlock  unlimited
*  soft  stack    65536
*  hard  stack    65536
LIMITS
chmod 0644 /etc/security/limits.d/99-llm-multigpu.conf

# ============================================================================
# DO NOT WRITE /etc/systemd/system.conf.d/*.conf OR /etc/systemd/user.conf.d/*.conf
# ============================================================================
# These files set DefaultLimit*= for PID 1 and every systemd-managed service.
# A typo here is unrecoverable without console access — there is no SSH route
# back if you crash sshd at boot.
#
# Specific traps:
#   1. UNIT MISMATCH. pam_limits (`/etc/security/limits.d/`) and systemd use
#      DIFFERENT units for the same resource:
#        limits.conf  stack    65536   → 65536 KB  = 64 MB   (fine)
#        systemd      LimitSTACK=65536 → 65536 B   = 64 KB   (catastrophic)
#      LimitMEMLOCK/LimitDATA/LimitAS/LimitFSIZE all share the bytes-vs-KB trap.
#      LimitNOFILE/LimitNPROC are counts in both — those happen to agree.
#   2. BLAST RADIUS. Per-service Limit*= in a unit file affects ONE service.
#      DefaultLimit*= in /etc/systemd/system.conf.d/ affects EVERY service,
#      including ones you didn't author (sshd, dbus, NetworkManager, polkit,
#      fabricmanager). Even a "harmless" change cascades.
#   3. systemctl daemon-reexec APPLIES IT NOW. PID 1 re-execs with the new
#      defaults immediately, so a bad value is live the moment the file is
#      written. There is no "test before commit" path.
#
# Right pattern: put Limit*= in the unit file of the service that needs it.
# See /etc/systemd/system/llama-server@.service in step 16 for the canonical
# example.
#
# Drop any stale config from a prior buggy install-all.sh run that set
# DefaultLimitSTACK=65536 (= 64 KB), which crashes every systemd service.
if [[ -f /etc/systemd/system.conf.d/99-llm-multigpu.conf ]]; then
    rm -f /etc/systemd/system.conf.d/99-llm-multigpu.conf
    systemctl daemon-reexec 2>/dev/null || true
    log "Removed stale /etc/systemd/system.conf.d/99-llm-multigpu.conf (had broken DefaultLimitSTACK)"
fi
log "System limits applied (pam_limits only; per-service limits live in unit files)"

mark_step_ok
