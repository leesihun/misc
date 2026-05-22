#!/usr/bin/env bash
# ============================================================================
# install-all.d/16-operational-tooling.sh
#
#   Install helper scripts (gpu-health-check, llama-server-multigpu,
#   llama-model-preload) and the llama-server@.service systemd template.
#
#   Directly runnable: sudo bash install-all.d/16-operational-tooling.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "16-operational-tooling"

step "1. gpu-health-check"
tee /usr/local/bin/gpu-health-check > /dev/null <<'HEALTH'
#!/usr/bin/env bash
# gpu-health-check — verify multi-GPU fabric is healthy.
# Installed by install-all.d/16-operational-tooling.sh.
set -uo pipefail
rc=0
say()  { printf '\033[1;36m[health]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  PASS\033[0m  %s\n' "$*"; }
bad()  { printf '\033[1;31m  FAIL\033[0m  %s\n' "$*"; rc=1; }
warn() { printf '\033[1;33m  WARN\033[0m  %s\n' "$*"; }

say "1. GPU visibility"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    n=$(nvidia-smi -L | wc -l | tr -d ' ')
    ok "$n GPU(s) visible"
else
    bad "nvidia-smi cannot list GPUs"; exit 1
fi

say "2. nvidia-fabricmanager"
if (( n > 1 )); then
    state=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo unknown)
    [[ "$state" == "active" ]] && ok "active" \
        || bad "NOT active on $n-GPU NVSwitch box; fabric may be degraded"
fi

say "3. NVLink lane health"
status=$(nvidia-smi nvlink --status 2>/dev/null || true)
if [[ -n "$status" ]]; then
    down=$(printf '%s\n' "$status" | grep -ciE 'inactive|<inactive>|disabled' || true)
    up=$(printf '%s\n' "$status" | grep -c 'GB/s' || true)
    if (( down == 0 )) && (( up > 0 )); then ok "$up active lane(s), 0 inactive"
    elif (( down > 0 )); then bad "$down inactive NVLink lane(s)"
    else warn "no NVLink data reported"
    fi
fi

say "4. fabric.state per GPU"
fab=$(nvidia-smi -q 2>/dev/null | awk '
    /^[[:space:]]+(GPU[[:space:]]+)?Fabric[[:space:]]*$/ {
        in_fabric = 1; captured = 0; next
    }
    in_fabric && !captured && /^[[:space:]]+State[[:space:]]*:/ {
        v = $0; sub(/.*:[[:space:]]*/, "", v); sub(/[[:space:]]+$/, "", v)
        print v
        captured = 1; in_fabric = 0; next
    }
    in_fabric && /^[[:space:]]+[A-Z][A-Za-z0-9 ]*[[:space:]]*$/ {
        in_fabric = 0
    }
    /^[[:space:]]+Fabric[[:space:]]+State[[:space:]]*:/ {
        v = $0; sub(/.*:[[:space:]]*/, "", v); sub(/[[:space:]]+$/, "", v)
        print v
    }
')
total=$(printf '%s\n' "$fab" | grep -c . || true)
ok_n=$(printf '%s\n' "$fab" | grep -cE '^(Completed|Success)$' || true)
if (( total == 0 )); then
    if (( n > 1 )); then
        bad "no Fabric stanza in nvidia-smi -q on $n-GPU box; FM did not initialize NVSwitch"
    else
        warn "no Fabric stanza in nvidia-smi -q (single-GPU box)"
    fi
elif (( total != n )); then
    bad "matched $total Fabric entries for $n GPU(s); parser/driver mismatch"
elif (( ok_n == total )); then
    ok "fabric.state Completed on $ok_n/$total GPU(s)"
else
    bad "fabric.state $ok_n/$total Completed -- common cause of CUDA Error 802"
fi

echo
[[ $rc -eq 0 ]] && say "ALL CHECKS PASSED" || say "ONE OR MORE CHECKS FAILED"
exit $rc
HEALTH
chmod 0755 /usr/local/bin/gpu-health-check
log "Installed: gpu-health-check"

step "2. llama-server-multigpu"
tee /usr/local/bin/llama-server-multigpu > /dev/null <<LLAMAWRAP
#!/usr/bin/env bash
# llama-server-multigpu — NUMA + tensor-split wrapper for llama.cpp server.
# Installed by install-all.d/16-operational-tooling.sh.
set -euo pipefail
LLAMA_BIN="\${LLAMA_BIN:-$LLAMA_PREFIX/build/bin/llama-server}"
[[ -x "\$LLAMA_BIN" ]] || { echo "llama-server not found at \$LLAMA_BIN" >&2; exit 1; }
N=\${LLAMA_N_GPUS:-\$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
extra_args=()
if (( N > 1 )) && ! printf '%s\n' "\$@" | grep -q -- '--tensor-split'; then
    split=\$(yes 1 | head -n "\$N" | paste -sd ',')
    extra_args+=( --tensor-split "\$split" )
    mode="\${LLAMA_SPLIT_MODE:-row}"
    if [[ "\$mode" != "none" ]] && ! printf '%s\n' "\$@" | grep -q -- '--split-mode'; then
        extra_args+=( --split-mode "\$mode" )
    fi
fi
prefix=()
if [[ "\${LLAMA_NO_NUMA:-0}" != "1" ]] && command -v numactl >/dev/null && command -v nvidia-smi >/dev/null; then
    pci=\$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    pci_short="\${pci#????}"
    node_file="/sys/bus/pci/devices/\${pci_short,,}/numa_node"
    if [[ -f "\$node_file" ]]; then
        node=\$(cat "\$node_file" 2>/dev/null || echo -1)
        [[ "\$node" -ge 0 ]] && prefix=(numactl --cpunodebind="\$node" --membind="\$node")
    fi
fi
echo "[llama-server-multigpu] \$N GPU(s); extra: \${extra_args[*]}; numa: \${prefix[*]:-none}" >&2
exec "\${prefix[@]}" "\$LLAMA_BIN" "\${extra_args[@]}" "\$@"
LLAMAWRAP
chmod 0755 /usr/local/bin/llama-server-multigpu
log "Installed: llama-server-multigpu"

step "3. llama-model-preload"
tee /usr/local/bin/llama-model-preload > /dev/null <<'PRELOAD'
#!/usr/bin/env bash
# llama-model-preload — pre-mmap GGUF model into page cache.
# Installed by install-all.d/16-operational-tooling.sh.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <model.gguf> [more.gguf ...]" >&2; exit 1; }
for f in "$@"; do
    [[ -r "$f" ]] || { echo "[preload] skip: $f (not readable)" >&2; continue; }
    sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
    sz_gb=$(awk -v s="$sz" 'BEGIN {printf "%.1f", s/(1024*1024*1024)}')
    printf '[preload] reading %s (%s GiB) -> page cache\n' "$f" "$sz_gb" >&2
    cat "$f" > /dev/null
done
echo "[preload] done; subsequent mmap reads will hit page cache" >&2
PRELOAD
chmod 0755 /usr/local/bin/llama-model-preload
log "Installed: llama-model-preload"

step "4. llama-server@.service template"
mkdir -p /etc/llama-server
if [[ ! -f /etc/llama-server/example.env ]]; then
    tee /etc/llama-server/example.env > /dev/null <<'EXAMPLE'
# /etc/llama-server/<instance>.env
# Copy to <name>.env and customize, then: systemctl enable --now llama-server@<name>
MODEL=/opt/models/MODEL.gguf
HOST=0.0.0.0
PORT=8080
NGL=999
CTX=8192
EXTRA=
EXAMPLE
fi
tee /etc/systemd/system/llama-server@.service > /dev/null <<'UNIT'
[Unit]
Description=llama.cpp server (instance %i)
After=network-online.target nvidia-fabricmanager.service nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=exec
EnvironmentFile=/etc/llama-server/%i.env
ExecStartPre=/usr/local/bin/llama-model-preload ${MODEL}
ExecStart=/usr/local/bin/llama-server-multigpu \
    --model ${MODEL} \
    --host ${HOST} \
    --port ${PORT} \
    --n-gpu-layers ${NGL} \
    --ctx-size ${CTX} \
    ${EXTRA}
Restart=on-failure
RestartSec=10
LimitNOFILE=1048576
LimitMEMLOCK=infinity
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload 2>/dev/null || true
log "Installed: llama-server@.service (configure /etc/llama-server/<name>.env)"

mark_step_ok
