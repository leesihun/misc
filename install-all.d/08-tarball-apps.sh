#!/usr/bin/env bash
# ============================================================================
# install-all.d/08-tarball-apps.sh
#
#   Extract Firefox / Node.js / Bun / Opencode from the bundled tarballs.
#
#   Directly runnable: sudo bash install-all.d/08-tarball-apps.sh
# ============================================================================
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00-common.sh"

require_root "$@"
init_step "08-tarball-apps"
locate_bundle

step "1. Firefox"
FF_TARBALL=""
for c in firefox.tar.xz firefox.tar.bz2; do
    [[ -f "$BUNDLE_DIR/apps/$c" ]] && { FF_TARBALL="$BUNDLE_DIR/apps/$c"; break; }
done
if [[ -n "$FF_TARBALL" ]]; then
    FF_VER=$(cat "$BUNDLE_DIR/apps/firefox.version" 2>/dev/null || echo unknown)
    log "Installing Firefox $FF_VER to /opt/firefox"
    mkdir -p /opt/firefox
    _ff_magic=$(head -c 6 "$FF_TARBALL" | od -An -tx1 | tr -d ' \n')
    case "$_ff_magic" in
        fd377a585a00*) _flag="-xJf" ;;   # xz
        425a68*)       _flag="-xjf" ;;   # bz2
        1f8b*)         _flag="-xzf" ;;   # gz
        *)             _flag="" ;;
    esac
    if [[ -n "$_flag" ]] && tar "$_flag" "$FF_TARBALL" -C /opt/firefox --strip-components=1; then
        ln -sf /opt/firefox/firefox /usr/local/bin/firefox
        log "Firefox: $(/opt/firefox/firefox --version 2>/dev/null || echo OK)"
        tee /usr/share/applications/firefox-manual.desktop > /dev/null <<'EOF'
[Desktop Entry]
Name=Firefox
Comment=Web Browser
Exec=/opt/firefox/firefox %u
Icon=/opt/firefox/browser/chrome/icons/default/default128.png
Terminal=false
Type=Application
Categories=Network;WebBrowser;
MimeType=text/html;text/xml;application/xhtml+xml;x-scheme-handler/http;x-scheme-handler/https;
EOF
    else
        warn "Firefox extraction failed (magic=$_ff_magic)."
    fi
else
    warn "apps/firefox.tar.{xz,bz2} not found; skipping."
fi

step "2. Node.js"
if [[ -f "$BUNDLE_DIR/apps/nodejs.tar.xz" ]]; then
    NODE_VER=$(cat "$BUNDLE_DIR/apps/nodejs.version" 2>/dev/null || echo unknown)
    log "Installing Node.js v$NODE_VER to /opt/nodejs"
    rm -rf /opt/nodejs
    mkdir -p /opt/nodejs
    if tar -xJf "$BUNDLE_DIR/apps/nodejs.tar.xz" -C /opt/nodejs --strip-components=1; then
        for bin in node npm npx; do
            ln -sf "/opt/nodejs/bin/$bin" "/usr/local/bin/$bin" || warn "Could not symlink $bin."
        done
        log "Node.js: $(node --version 2>/dev/null)  npm: $(npm --version 2>/dev/null)"
    else
        warn "Node.js extraction failed."
    fi
else
    warn "apps/nodejs.tar.xz not found; skipping."
fi

step "3. Bun"
if [[ -f "$BUNDLE_DIR/apps/bun-linux-x64.zip" ]]; then
    BUN_TAG=$(cat "$BUNDLE_DIR/apps/bun.version" 2>/dev/null || echo unknown)
    log "Installing Bun $BUN_TAG"
    TMP_BUN=$(mktemp -d)
    if unzip -q "$BUNDLE_DIR/apps/bun-linux-x64.zip" -d "$TMP_BUN" \
        && [[ -x "$TMP_BUN/bun-linux-x64/bun" ]] \
        && install -m 0755 "$TMP_BUN/bun-linux-x64/bun" /usr/local/bin/bun; then
        ln -sf /usr/local/bin/bun /usr/local/bin/bunx
        log "Bun: $(bun --version 2>/dev/null)"
    else
        warn "Bun extraction/install failed."
    fi
    rm -rf "$TMP_BUN"
else
    warn "apps/bun-linux-x64.zip not found; skipping."
fi

step "4. Opencode"
if [[ -f "$BUNDLE_DIR/apps/opencode" ]]; then
    OC_VER=$(cat "$BUNDLE_DIR/apps/opencode.version" 2>/dev/null || echo unknown)
    log "Installing Opencode $OC_VER -> /usr/local/bin/opencode"
    install -m 0755 "$BUNDLE_DIR/apps/opencode" /usr/local/bin/opencode \
        && log "Opencode installed" \
        || warn "Opencode install failed."
elif [[ -f "$BUNDLE_DIR/apps/opencode.MISSING" ]]; then
    warn "Opencode was not downloaded during gather. Place binary at /usr/local/bin/opencode manually."
else
    warn "apps/opencode not found; skipping."
fi

step "5. needrestart -r a (auto-restart daemons holding old libs)"
if command -v needrestart >/dev/null 2>&1; then
    NEEDRESTART_MODE=a needrestart -r a 2>&1 | tail -50 || warn "needrestart returned non-zero."
else
    warn "needrestart not installed; skipping (libs may be stale until reboot)."
fi

mark_step_ok
