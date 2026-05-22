#!/usr/bin/env bash
# ============================================================================
# install-all.d/10-wheelhouse-manifests.sh
#
#   Generate per-wheelhouse requirements.txt files from the bundled .whl /
#   .tar.gz / .tgz / .zip archives. Pure file scan; airgap-safe.
#
#   Directly runnable: sudo bash install-all.d/10-wheelhouse-manifests.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "10-wheelhouse-manifests"
locate_bundle

step "Wheelhouse manifests"
generate_wheelhouse_requirements

mark_step_ok
