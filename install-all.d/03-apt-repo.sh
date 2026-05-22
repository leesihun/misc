#!/usr/bin/env bash
# ============================================================================
# install-all.d/03-apt-repo.sh
#
#   Set up the local file:// apt repo from the bundle's debs/, place
#   defensive holds on system libraries the NVIDIA driver/CUDA stack links
#   against, and refresh apt indexes.
#
#   We DO NOT:
#     - apt-mark hold nvidia-*/cuda-*/libnvidia-* — install-nvidia.sh already
#       did that and the marks persist. Duplicating here was redundant.
#     - Pin nvidia-* to developer.download.nvidia.com — that origin is
#       unreachable on airgap, and install-nvidia.sh's
#       /etc/apt/preferences.d/99-nvidia-prefer-bundle already pins those
#       packages to the file:// nvidia bundle repo.
#
#   Directly runnable: sudo bash install-all.d/03-apt-repo.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "03-apt-repo"
locate_bundle
source_bundle_metadata

step "1. System-lib holds (protect CUDA runtime link target)"
# Hold libstdc++6/libgcc-s1/libgomp1/libc6 so the userland install can't
# silently DOWNGRADE these via --allow-downgrades when the bundle ships older
# versions than what install-nvidia.sh's CUDA toolkit pulled in — that would
# break nvidia-smi / NVML at runtime.
#
# dpkg-query exits non-zero if any glob matches zero packages. Swallow it.
sys_pkgs=$({ dpkg-query -W -f='${Package} ${Status}\n' \
    'libstdc++6' 'libgcc-s1' 'libgomp1' 'libc6' \
    2>/dev/null || true; } \
    | awk '$2 == "install" && $3 == "ok" && $4 == "installed" {print $1}' \
    | sort -u)
if [[ -n "$sys_pkgs" ]]; then
    log "Placing apt-mark hold on $(printf '%s\n' "$sys_pkgs" | wc -l) system runtime-lib packages"
    # shellcheck disable=SC2086
    apt-mark hold $sys_pkgs >/dev/null 2>&1 || warn "apt-mark hold reported errors (non-fatal)."
    install -d -m 0755 "$STATE_DIR"
    printf '%s\n' "$sys_pkgs" > "$STATE_DIR/system-libs-held.txt"
else
    warn "No system runtime libs matched the hold patterns — unusual."
fi

step "2. Local apt repo: $APT_REPO_DIR"
rm -rf "$APT_REPO_DIR"
mkdir -p "$APT_REPO_DIR"
cp -r "$BUNDLE_DIR/debs/." "$APT_REPO_DIR/"
if command -v dpkg-scanpackages >/dev/null 2>&1; then
    ( cd "$APT_REPO_DIR" && dpkg-scanpackages . /dev/null > Packages 2>/dev/null )
    gzip -9c "$APT_REPO_DIR/Packages" > "$APT_REPO_DIR/Packages.gz"
elif [[ -f "$APT_REPO_DIR/Packages" ]]; then
    log "Using bundled debs/Packages index (dpkg-scanpackages not installed yet)"
    [[ -f "$APT_REPO_DIR/Packages.gz" ]] || gzip -9c "$APT_REPO_DIR/Packages" > "$APT_REPO_DIR/Packages.gz"
else
    die "Local apt repo has no Packages index and dpkg-scanpackages is unavailable. Rebuild the bundle with gather-all.sh."
fi

step "3. Register sources.list.d entry"
tee /etc/apt/sources.list.d/00-bundle.list > /dev/null <<EOF
# Installed by install-all.d/03-apt-repo.sh — local airgap bundle apt repo.
deb [trusted=yes] file://$APT_REPO_DIR ./
EOF

step "4. apt-get update"
apt-get update -o Acquire::http::Timeout=10 -o Acquire::https::Timeout=10 \
    || warn "apt-get update reported errors (vendor's NVIDIA repo may be unreachable on airgap — OK)."

mark_step_ok
