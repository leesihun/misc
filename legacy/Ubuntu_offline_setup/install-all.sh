#!/usr/bin/env bash
# ============================================================================
# install-all.sh
#   Run on the AIR-GAPPED Ubuntu 24.04 server.
#   Place this script next to all-airgap-bundle-ubuntu24.04.bin and run it;
#   it will auto-extract the bundle and proceed with installation.
#
# Usage:
#   sudo bash install-all.sh              # auto-extracts bundle, installs everything
#   bash install-all.sh                   # works too (will sudo internally)
#
# Optional overrides:
#   INSTALL_INFERENCE=0  bash install-all.sh   # skip LLM inference venv (vLLM/LLM_API_fast/RAG)
#   INSTALL_TRAINING=0   bash install-all.sh   # skip general training venv (PyG/MeshGraphNets)
#   INSTALL_JUPYTER=0    bash install-all.sh   # skip JupyterLab venv
#   INSTALL_DESKTOP=0    bash install-all.sh   # skip XFCE4/xrdp configuration
#   INFERENCE_PREFIX=/opt/llm_inference bash install-all.sh
#   TRAINING_PREFIX=/opt/general_training bash install-all.sh
#   INSTALL_K3S=1 K3S_ROLE=server bash install-all.sh   # bootstrap K3s control plane
#   INSTALL_K3S=1 K3S_ROLE=agent K3S_SERVER_IP=10.0.0.101 K3S_TOKEN_FILE=/tmp/k3s-join-token bash install-all.sh
#   BUNDLE_DIR=/path/to/extracted bash install-all.sh   # skip auto-extract
#
# Two-pass install (recommended on a fresh box where the NVIDIA driver isn't
# loaded yet — without this, FabricManager can't init NVSwitch and torch
# fails with "system not yet initialized"):
#   PHASE=1 bash install-all.sh    # apt + CUDA + driver; exits before apps
#   sudo reboot
#   PHASE=2 bash install-all.sh    # services + apps + venvs + llama.cpp
# Default PHASE=all keeps the old single-pass behavior.
# ============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
INSTALL_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DIAG_LOG_USER_SET=0
[[ -n "${DIAG_LOG:-}" ]] && DIAG_LOG_USER_SET=1
INSTALL_LOG="${INSTALL_LOG:-$SCRIPT_DIR/install-all-$RUN_STAMP.log}"
DIAG_LOG="${DIAG_LOG:-$SCRIPT_DIR/install-diagnostics-$RUN_STAMP.log}"
BUNDLE_DIR="${BUNDLE_DIR:-$SCRIPT_DIR}"
BUNDLE_BIN="${BUNDLE_BIN:-}"   # auto-detected below if not set
PYTHON_VER="${PYTHON_VER:-3.12}"
PYTHON_BIN="${PYTHON_BIN:-python${PYTHON_VER}}"
# Install prefixes default to /scratch so output is shared and survives a
# sudo-vs-user run (no $HOME dependency). Override per-prefix at the env if
# /scratch isn't suitable.
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"
INFERENCE_PREFIX="${INFERENCE_PREFIX:-$SCRATCH_ROOT/llm_inference}"
TRAINING_PREFIX="${TRAINING_PREFIX:-$SCRATCH_ROOT/general_training}"
LLAMA_PREFIX="${LLAMA_PREFIX:-$SCRATCH_ROOT/llama.cpp}"
INSTALL_INFERENCE="${INSTALL_INFERENCE:-1}"
INSTALL_TRAINING="${INSTALL_TRAINING:-1}"
INSTALL_LLAMA="${INSTALL_LLAMA:-1}"
INSTALL_DESKTOP="${INSTALL_DESKTOP:-1}"
BUILD_CUDA="${BUILD_CUDA:-1}"
BUILD_BLAS="${BUILD_BLAS:-1}"

# Two-pass install mode. The NVIDIA kernel module isn't loaded until reboot
# after a fresh driver install, which means FabricManager can't initialize
# NVSwitch, torch can't init CUDA ("system not yet initialized"), and llama.cpp
# can't run kernels at install time. Splitting the run avoids this:
#
#   PHASE=1 bash install-all.sh   # apt, CUDA toolkit, driver, env config; STOPS
#   sudo reboot                   # driver kmod loads on boot
#   PHASE=2 bash install-all.sh   # services start, apps, venvs, llama.cpp build
#
# PHASE=all (the default) runs both back-to-back like before — fine for hosts
# that already have a matching nvidia driver loaded.
PHASE="${PHASE:-all}"
case "$PHASE" in
    1|2|all) ;;
    *) printf '[install:ERROR] PHASE=%s invalid; must be 1, 2, or all\n' "$PHASE" >&2; exit 2 ;;
esac

# Predicate helpers to gate sections by phase.
# _phase_runs_apt: true unless PHASE=2 (apt install is the slow step we skip).
# _phase_runs_apps: true unless PHASE=1 (we stop before apps).
_phase_runs_apt()  { [[ "$PHASE" != "2" ]]; }
_phase_runs_apps() { [[ "$PHASE" != "1" ]]; }
JOBS="${JOBS:-$(nproc)}"
VERIFY_CHECKSUMS="${VERIFY_CHECKSUMS:-1}"
INSTALL_EXTRA="${INSTALL_EXTRA:-1}"
EXTRA_PREFIX="${EXTRA_PREFIX:-$SCRATCH_ROOT/extra}"
INSTALL_JUPYTER="${INSTALL_JUPYTER:-1}"
JUPYTER_PREFIX="${JUPYTER_PREFIX:-$SCRATCH_ROOT/jupyter}"
INSTALL_K3S="${INSTALL_K3S:-0}"
K3S_ROLE="${K3S_ROLE:-none}"           # server | agent | none
K3S_SERVER_IP="${K3S_SERVER_IP:-}"     # required when K3S_ROLE=agent
K3S_TOKEN_FILE="${K3S_TOKEN_FILE:-}"   # path to join-token file
K3S_REGISTRY_PORT="${K3S_REGISTRY_PORT:-5000}"
GPU_OPERATOR_DRIVER_ENABLED="${GPU_OPERATOR_DRIVER_ENABLED:-false}"
INSTALL_KUBERAY="${INSTALL_KUBERAY:-1}"
APT_REPO_DIR="${APT_REPO_DIR:-/var/tmp/airgap-bundle-debs}"
ALLOW_DPKG_FALLBACK="${ALLOW_DPKG_FALLBACK:-1}"

INSTALL_WARNINGS=()
INSTALL_ERRORS=()
FINAL_DIAGNOSTICS_PRINTED=0
WHEEL_REQS_GENERATED=0

start_transcript_log() {
    local log_dir
    log_dir="$(dirname "$INSTALL_LOG")"

    if ! mkdir -p "$log_dir" 2>/dev/null || ! touch "$INSTALL_LOG" 2>/dev/null; then
        printf '[install:WARN] Cannot write log under %s; using /tmp instead.\n' "$log_dir" >&2
        INSTALL_LOG="/tmp/install-all-$RUN_STAMP.log"
        if [[ "$DIAG_LOG_USER_SET" != "1" ]]; then
            DIAG_LOG="/tmp/install-diagnostics-$RUN_STAMP.log"
        fi
        touch "$INSTALL_LOG" 2>/dev/null \
            || { printf '[install:ERROR] Cannot create transcript log: %s\n' "$INSTALL_LOG" >&2; exit 1; }
    fi

    exec > >(tee -a "$INSTALL_LOG") 2>&1
    printf '[install] Full transcript log: %s\n' "$INSTALL_LOG"
    printf '[install] Diagnostics summary: %s\n' "$DIAG_LOG"
}

start_transcript_log

log()  { printf '\033[1;32m[install]\033[0m %s\n' "$*"; }
warn() { INSTALL_WARNINGS+=( "$*" ); printf '\033[1;33m[install:WARN]\033[0m %s\n' "$*"; }
die()  { INSTALL_ERRORS+=( "$*" ); printf '\033[1;31m[install:ERROR]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;35m== %s ==\033[0m\n' "$*"; }

_on_err() {
    local rc=$?
    local cmd="${BASH_COMMAND:-unknown command}"
    INSTALL_ERRORS+=( "command failed with exit $rc: $cmd" )
    return "$rc"
}
trap _on_err ERR

_is_service_active() {
    local service_name="$1"
    systemctl is-active --quiet "$service_name" 2>/dev/null \
        || service "$service_name" status >/dev/null 2>&1
}

_pkg_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"
}

_pkg_satisfied() {
    local pkg="$1" alt=""
    _pkg_installed "$pkg" && return 0
    case "$pkg" in
        libglib2.0-0) alt="libglib2.0-0t64" ;;
        libatk1.0-0) alt="libatk1.0-0t64" ;;
        libatk-bridge2.0-0) alt="libatk-bridge2.0-0t64" ;;
        libcups2) alt="libcups2t64" ;;
        libgtk-3-0) alt="libgtk-3-0t64" ;;
        libasound2) alt="libasound2t64" ;;
    esac
    [[ -n "$alt" ]] && _pkg_installed "$alt"
}

_wheelhouse_has_packages() {
    local wheel_dir="$1"
    [[ -d "$wheel_dir" ]] || return 1
    compgen -G "$wheel_dir/*.whl" >/dev/null \
        || compgen -G "$wheel_dir/*.tar.gz" >/dev/null \
        || compgen -G "$wheel_dir/*.tgz" >/dev/null \
        || compgen -G "$wheel_dir/*.zip" >/dev/null
}

_venv_module_check() {
    local py="$1"
    shift
    "$py" - "$@" <<'PY'
import importlib.util
import sys

modules = sys.argv[1:]
missing = [m for m in modules if importlib.util.find_spec(m) is None]
if missing:
    print("missing: " + ", ".join(missing))
    sys.exit(1)
print("modules present: " + ", ".join(modules))
PY
}

generate_wheelhouse_requirements() {
    local wheels_root="${BUNDLE_DIR:-}/wheels"
    local generator_py=""

    [[ -d "$wheels_root" ]] || return 0
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        generator_py="$PYTHON_BIN"
    elif command -v python3 >/dev/null 2>&1; then
        generator_py="python3"
    else
        warn "Could not generate wheels/*/requirements.txt because python3 is not available yet."
        return 0
    fi

    log "Generating requirements.txt for wheel directories under $wheels_root"
    if "$generator_py" - "$wheels_root" <<'PYREQS'
from email.parser import Parser
from pathlib import Path
import re
import sys
import tarfile
import zipfile

wheels_root = Path(sys.argv[1])
archive_suffixes = (".whl", ".tar.gz", ".tgz", ".zip")

def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def version_key(version):
    parts = re.split(r"([0-9]+)", version)
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts)

def parse_metadata_text(text):
    msg = Parser().parsestr(text)
    name = msg.get("Name")
    version = msg.get("Version")
    if name and version:
        return name.strip(), version.strip()
    return None

def read_wheel_metadata(path):
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".dist-info/METADATA"):
                return parse_metadata_text(zf.read(name).decode("utf-8", "replace"))
    return None

def read_zip_sdist_metadata(path):
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("/PKG-INFO") or name == "PKG-INFO":
                return parse_metadata_text(zf.read(name).decode("utf-8", "replace"))
    return None

def read_tar_sdist_metadata(path):
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            if member.name.endswith("/PKG-INFO") or member.name == "PKG-INFO":
                fh = tf.extractfile(member)
                if fh:
                    return parse_metadata_text(fh.read().decode("utf-8", "replace"))
    return None

def parse_wheel_filename(path):
    stem = path.name[:-4]
    parts = stem.split("-")
    if len(parts) >= 5:
        return parts[0].replace("_", "-"), parts[1]
    return None

def parse_archive_filename(path):
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".zip"):
        if name.endswith(suffix):
            stem = name[:-len(suffix)]
            break
    else:
        return None
    match = re.match(r"(.+)-([0-9][A-Za-z0-9_.!+~-]*)$", stem)
    if match:
        return match.group(1).replace("_", "-"), match.group(2)
    return None

def package_from_archive(path):
    try:
        if path.suffix == ".whl":
            found = read_wheel_metadata(path)
            if found:
                return found
            return parse_wheel_filename(path)
        if path.name.endswith((".tar.gz", ".tgz")):
            found = read_tar_sdist_metadata(path)
            if found:
                return found
            return parse_archive_filename(path)
        if path.suffix == ".zip":
            found = read_zip_sdist_metadata(path)
            if found:
                return found
            return parse_archive_filename(path)
    except Exception as exc:
        print(f"warning: {path}: {exc}", file=sys.stderr)
        if path.suffix == ".whl":
            return parse_wheel_filename(path)
        return parse_archive_filename(path)
    return None

for wheel_dir in sorted(p for p in wheels_root.iterdir() if p.is_dir()):
    archives = sorted(
        p for p in wheel_dir.iterdir()
        if p.is_file() and p.name.endswith(archive_suffixes)
    )
    packages = {}
    unreadable = []

    for archive in archives:
        parsed = package_from_archive(archive)
        if not parsed:
            unreadable.append(archive.name)
            continue
        name, version = parsed
        key = normalize_name(name)
        display, versions = packages.setdefault(key, (name, set()))
        versions.add(version)

    out_path = wheel_dir / "requirements.txt"
    lines = [
        "# Generated by install-all.sh from the package archives in this directory.",
        "# Offline install example:",
        "#   python -m pip install --no-index --find-links=. -r requirements.txt",
        "",
    ]

    for key in sorted(packages):
        display, versions = packages[key]
        ordered_versions = sorted(versions, key=version_key)
        if len(ordered_versions) > 1:
            lines.append(
                f"# WARNING: multiple versions found for {display}: "
                + ", ".join(ordered_versions)
            )
        lines.append(f"{display}=={ordered_versions[-1]}")

    if unreadable:
        lines.append("")
        lines.append("# Archives that could not be converted into requirement pins:")
        lines.extend(f"#   {name}" for name in unreadable)

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"generated: {out_path} ({len(packages)} packages, {len(archives)} archives)")
PYREQS
    then
        WHEEL_REQS_GENERATED=1
    else
        warn "Could not generate one or more wheels/*/requirements.txt files."
    fi
}

print_final_diagnostics() {
    local exit_code="${1:-0}"
    [[ "$FINAL_DIAGNOSTICS_PRINTED" == "1" ]] && return 0
    FINAL_DIAGNOSTICS_PRINTED=1

    set +e
    local ok_count=0 warn_count=0 bad_count=0 skip_count=0
    local -a _bad_recap=()
    local -a _warn_recap=()
    local FAILURE_REPORT="${DIAG_LOG%.log}-failures.txt"

    _report_line() {
        local status="$1" area="$2" detail="$3" cause="${4:-}" check="${5:-}"
        case "$status" in
            OK) ok_count=$((ok_count + 1)) ;;
            WARN) warn_count=$((warn_count + 1)); _warn_recap+=( "$area|$detail|$cause|$check" ) ;;
            BAD) bad_count=$((bad_count + 1)); _bad_recap+=( "$area|$detail|$cause|$check" ) ;;
            SKIP) skip_count=$((skip_count + 1)) ;;
        esac
        printf '[%s] %s: %s\n' "$status" "$area" "$detail"
        [[ -n "$cause" ]] && printf '      cause: %s\n' "$cause"
        [[ -n "$check" ]] && printf '      check: %s\n' "$check"
        return 0
    }

    _check_venv_modules() {
        local label="$1" py="$2" modules="$3" why="$4"
        if [[ ! -x "$py" ]]; then
            _report_line BAD "$label" "venv python is missing: $py" \
                "$why did not create a usable virtualenv, or the prefix was changed." \
                "ls -l ${py%/bin/python}; inspect the installer output above this report"
            return
        fi
        local result
        # shellcheck disable=SC2086
        result="$(_venv_module_check "$py" $modules 2>&1)"
        if [[ "$?" == "0" ]]; then
            _report_line OK "$label" "$result" "" "$py -m pip check"
        else
            _report_line BAD "$label" "$result" \
                "One or more bundled wheels were missing, incompatible, or skipped during pip install." \
                "$py -m pip check; $py -m pip list"
        fi
    }

    _check_pip_check() {
        local label="$1" py="$2"
        [[ -x "$py" ]] || return
        local result
        result="$("$py" -m pip check 2>&1)"
        if [[ "$?" == "0" ]]; then
            _report_line OK "$label pip check" "$result"
        else
            _report_line WARN "$label pip check" "$result" \
                "Installed Python packages have dependency conflicts." \
                "$py -m pip check"
        fi
    }

    _check_torch_cuda() {
        local label="$1" py="$2"
        [[ -x "$py" ]] || return 0
        local result rc
        result="$("$py" - <<'PY' 2>&1
import sys
try:
    import torch
except Exception as e:
    print(f"torch import failed: {type(e).__name__}: {e}")
    sys.exit(1)
try:
    avail = torch.cuda.is_available()
    nd = torch.cuda.device_count() if avail else 0
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} cuda_available={avail} device_count={nd}")
    sys.exit(0 if avail else 2)
except Exception as e:
    print(f"torch CUDA probe failed: {type(e).__name__}: {e}")
    sys.exit(3)
PY
)"
        rc=$?
        case "$rc" in
            0) _report_line OK "$label torch+CUDA" "$result" ;;
            2) _report_line WARN "$label torch+CUDA" "$result" \
                "torch is installed but cannot see CUDA: NVIDIA driver not loaded, kernel module missing, or a CPU-only torch wheel was installed." \
                "nvidia-smi; $py -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())'" ;;
            *) _report_line BAD "$label torch+CUDA" "$result" \
                "torch import or CUDA probe raised an exception inside the venv." \
                "$py -m pip show torch; $py -m pip check" ;;
        esac
    }

    _check_disk_space() {
        local mp avail_kb avail_gb seen=""
        for mp in / /usr/local /var "$HOME"; do
            [[ -d "$mp" ]] || continue
            # Skip dupes when /usr/local is on / etc.
            local dev
            dev=$(df --output=source "$mp" 2>/dev/null | tail -n 1)
            [[ -z "$dev" ]] && continue
            case "$seen" in *"|$dev|"*) continue;; esac
            seen="${seen}|$dev|"
            avail_kb=$(df --output=avail "$mp" 2>/dev/null | tail -n 1 | tr -d ' ')
            [[ -z "$avail_kb" ]] && continue
            avail_gb=$(( avail_kb / 1024 / 1024 ))
            if (( avail_gb < 5 )); then
                _report_line BAD "Disk space $mp" "only ${avail_gb}G free on $dev" \
                    "Insufficient disk space; CUDA install (~6G), wheel installs, and DKMS builds will fail." \
                    "df -h $mp; sudo du -xh --max-depth=1 $mp 2>/dev/null | sort -h | tail"
            elif (( avail_gb < 20 )); then
                _report_line WARN "Disk space $mp" "${avail_gb}G free on $dev (low)" \
                    "Low disk space risks DKMS rebuild and image-pull failures." \
                    "df -h $mp"
            else
                _report_line OK "Disk space $mp" "${avail_gb}G free on $dev"
            fi
        done
    }

    _check_kernel_state() {
        local running_kver installed_kver
        running_kver=$(uname -r 2>/dev/null)
        installed_kver=$(dpkg-query -W -f='${Version}\n' linux-image-generic 2>/dev/null | head -n 1)
        if [[ -z "$running_kver" ]]; then
            _report_line WARN "Kernel state" "uname -r returned empty"
            return
        fi
        # Find the newest installed linux-image-* package version
        local newest_image
        newest_image=$(dpkg -l 'linux-image-[0-9]*' 2>/dev/null \
            | awk '/^ii/ {print $2}' \
            | sed -nE 's/^linux-image-(.+)$/\1/p' \
            | sort -V | tail -n 1)
        if [[ -n "$newest_image" && "$newest_image" != "$running_kver" ]]; then
            _report_line WARN "Kernel state" "running=$running_kver, newest installed=$newest_image" \
                "A newer kernel is installed but not yet booted. NVIDIA + DKMS modules built against $newest_image will not load until reboot." \
                "sudo reboot; after reboot: uname -r && nvidia-smi"
        else
            _report_line OK "Kernel state" "running kernel $running_kver (matches newest installed)"
        fi
    }

    _check_dkms_detail() {
        if ! command -v dkms >/dev/null 2>&1; then
            _report_line SKIP "DKMS detail" "dkms not installed"
            return
        fi
        local -a dkms_lines
        mapfile -t dkms_lines < <(dkms status 2>/dev/null | sed '/^$/d')
        if (( ${#dkms_lines[@]} == 0 )); then
            _report_line OK "DKMS detail" "no DKMS modules registered"
            return
        fi
        local installed=0 not_installed=0 line state
        local -a broken_lines=()
        for line in "${dkms_lines[@]}"; do
            # Format varies by dkms version:
            #   "name, ver, kver, arch: installed"     (old)
            #   "name/ver, kver, arch: installed"      (new)
            state="${line##*: }"
            case "$state" in
                installed) installed=$((installed + 1)) ;;
                *) not_installed=$((not_installed + 1)); broken_lines+=("$line") ;;
            esac
        done
        if (( not_installed == 0 )); then
            _report_line OK "DKMS detail" "$installed module(s) installed against running kernel"
        else
            _report_line WARN "DKMS detail" "$installed installed, $not_installed NOT installed" \
                "Some DKMS modules failed to build/install. Common causes: Mellanox OFED (iser/isert/mlnx-nfsrdma) vs newer kernel headers, or NVIDIA module not yet built post-install." \
                "sudo dkms autoinstall; sudo dkms status; sudo dmesg | grep -iE 'dkms|module'"
            local bad
            for bad in "${broken_lines[@]}"; do
                printf '      DKMS: %s\n' "$bad"
            done
        fi
    }

    _check_secure_boot() {
        if ! command -v mokutil >/dev/null 2>&1; then
            _report_line SKIP "Secure Boot" "mokutil not installed (likely BIOS boot or container)"
            return
        fi
        local sb_state
        sb_state=$(mokutil --sb-state 2>/dev/null | head -n 1)
        case "$sb_state" in
            *enabled*)
                _report_line WARN "Secure Boot" "$sb_state" \
                    "Secure Boot is enabled. Unsigned NVIDIA DKMS modules will refuse to load — nvidia-smi will fail even after a clean install." \
                    "sudo mokutil --list-enrolled; sudo update-secureboot-policy --enroll-key" ;;
            *disabled*)
                _report_line OK "Secure Boot" "$sb_state" ;;
            *)
                _report_line SKIP "Secure Boot" "mokutil returned: ${sb_state:-empty}" ;;
        esac
    }

    _check_nvidia_kmod_loaded() {
        local lsmod_nv
        lsmod_nv=$(lsmod 2>/dev/null | awk '/^nvidia(_uvm|_drm|_modeset|_peermem)?[[:space:]]/ {print $1}')
        if [[ -z "$lsmod_nv" ]]; then
            if command -v nvidia-smi >/dev/null 2>&1; then
                _report_line BAD "NVIDIA kmod" "no nvidia* kernel modules are currently loaded" \
                    "Userspace driver is installed but the kernel module did not build/load. Usually fixed by reboot, DKMS rebuild, or Secure Boot enrollment." \
                    "sudo dmesg | grep -iE 'nvidia|nvrm' | tail -n 40; sudo modprobe nvidia; nvidia-smi" ;
            else
                _report_line SKIP "NVIDIA kmod" "no NVIDIA driver bundled"
            fi
            return
        fi
        local nv_count loaded_ver
        nv_count=$(printf '%s\n' "$lsmod_nv" | wc -l | tr -d ' ')
        loaded_ver=$(awk '/Kernel Module/ {for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+\.[0-9]+/) {print $i; exit}}' /proc/driver/nvidia/version 2>/dev/null)
        if [[ -z "$loaded_ver" ]]; then
            loaded_ver=$(modinfo nvidia 2>/dev/null | awk '/^version:/ {print $2; exit}')
        fi
        _report_line OK "NVIDIA kmod" "$nv_count module(s) loaded; driver=${loaded_ver:-unknown}"
        # Cross-check loaded driver branch vs installed userspace branch
        if [[ -n "$loaded_ver" ]]; then
            local loaded_branch installed_branch
            loaded_branch="${loaded_ver%%.*}"
            installed_branch=$(dpkg -l 'nvidia-utils-*' 2>/dev/null | awk '/^ii/ {print $2}' | sed -nE 's/^nvidia-utils-([0-9]+).*/\1/p' | sort -u | head -n 1)
            if [[ -n "$installed_branch" && "$loaded_branch" != "$installed_branch" ]]; then
                _report_line WARN "NVIDIA driver branch" "loaded=$loaded_branch, userspace=$installed_branch" \
                    "Kernel module and userspace driver are from different branches. nvidia-smi will report 'Driver/library version mismatch'. Reboot to load the new kernel module." \
                    "sudo reboot; or: sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia"
            fi
        fi
    }

    _check_nvidia_container_toolkit() {
        if command -v nvidia-ctk >/dev/null 2>&1; then
            local v
            v=$(nvidia-ctk --version 2>/dev/null | head -n 1)
            _report_line OK "NVIDIA Container Toolkit" "${v:-installed}"
            return
        fi
        if [[ -f "${BUNDLE_DIR:-}/meta/apt-packages.txt" ]] && grep -q '^nvidia-container-toolkit' "$BUNDLE_DIR/meta/apt-packages.txt" 2>/dev/null; then
            _report_line BAD "NVIDIA Container Toolkit" "nvidia-ctk command not found" \
                "Container toolkit was in the bundle manifest but did not install." \
                "dpkg -l 'nvidia-container*'; command -v nvidia-ctk"
        else
            _report_line SKIP "NVIDIA Container Toolkit" "not in bundle"
        fi
    }

    _check_cuda_runtime_detail() {
        # Resolve /usr/local/cuda
        if [[ -L /usr/local/cuda ]]; then
            local tgt
            tgt=$(readlink -f /usr/local/cuda 2>/dev/null)
            _report_line OK "CUDA symlink" "/usr/local/cuda → $tgt"
        elif [[ -d /usr/local/cuda ]]; then
            _report_line WARN "CUDA symlink" "/usr/local/cuda is a directory, not a symlink"
        else
            _report_line WARN "CUDA symlink" "/usr/local/cuda missing" \
                "Builds that reference /usr/local/cuda will fail to find headers/libs." \
                "sudo ln -sfn /usr/local/cuda-13.0 /usr/local/cuda"
        fi
        # profile.d shell hook — accept either /etc/profile.d/cuda.sh (what the
        # current install writes) OR /etc/profile.d/cuda-*.sh (legacy versioned
        # dropin). The install explicitly removes legacy dropins to avoid double-
        # sourcing, so the glob alone would false-WARN.
        local _cuda_profile=""
        if [[ -f /etc/profile.d/cuda.sh ]]; then
            _cuda_profile="/etc/profile.d/cuda.sh"
        elif compgen -G "/etc/profile.d/cuda-*.sh" >/dev/null; then
            _cuda_profile=$(ls /etc/profile.d/cuda-*.sh 2>/dev/null | tr '\n' ' ')
        fi
        if [[ -n "$_cuda_profile" ]]; then
            _report_line OK "CUDA profile.d" "$_cuda_profile"
        else
            _report_line WARN "CUDA profile.d" "/etc/profile.d/cuda.sh missing" \
                "New shells will not have CUDA on PATH automatically." \
                "echo 'export PATH=/usr/local/cuda-13.0/bin:\$PATH' | sudo tee /etc/profile.d/cuda.sh"
        fi
        # ldconfig cache
        if command -v ldconfig >/dev/null 2>&1; then
            local cuda_libs
            cuda_libs=$(ldconfig -p 2>/dev/null | grep -cE 'libcudart|libcublas|libcudnn')
            if (( cuda_libs > 0 )); then
                _report_line OK "CUDA ldconfig" "$cuda_libs CUDA libs in linker cache (libcudart/cublas/cudnn-class)"
            else
                _report_line WARN "CUDA ldconfig" "no CUDA libs found in ld.so.cache" \
                    "Runtime linking against libcudart/libcublas will fail." \
                    "sudo ldconfig; ldconfig -p | grep -E 'libcudart|libcublas'"
            fi
        fi
    }

    _check_apt_source_cleanup() {
        if [[ -f /etc/apt/sources.list.d/airgap-bundle.list ]]; then
            _report_line WARN "APT source cleanup" "/etc/apt/sources.list.d/airgap-bundle.list still present" \
                "Bundle apt source was not removed. Future 'apt update' will warn if the staged repo path is missing." \
                "sudo rm /etc/apt/sources.list.d/airgap-bundle.list && sudo rm -rf $APT_REPO_DIR"
        else
            _report_line OK "APT source cleanup" "no leftover airgap-bundle.list"
        fi
    }

    _check_install_prefixes() {
        local -a prefixes=()
        [[ "$INSTALL_INFERENCE" == "1" ]] && prefixes+=("$INFERENCE_PREFIX|Inference")
        [[ "$INSTALL_TRAINING"  == "1" ]] && prefixes+=("$TRAINING_PREFIX|Training")
        [[ "$INSTALL_JUPYTER"   == "1" ]] && prefixes+=("$JUPYTER_PREFIX|Jupyter")
        [[ "$INSTALL_LLAMA"     == "1" ]] && prefixes+=("$LLAMA_PREFIX|llama.cpp")
        local entry path label sz
        for entry in "${prefixes[@]}"; do
            path="${entry%%|*}"
            label="${entry##*|}"
            if [[ -d "$path" ]]; then
                sz=$(du -sh "$path" 2>/dev/null | cut -f1)
                _report_line OK "$label prefix" "$path (${sz:-?})"
            else
                _report_line BAD "$label prefix" "$path does not exist" \
                    "Installation step did not create the directory. Look above for the relevant step's error." \
                    "ls -lah $(dirname $path 2>/dev/null); grep -nE 'Python venv|llama.cpp' $INSTALL_LOG | tail -n 20"
            fi
        done
    }

    _check_service_inventory() {
        local svc state
        local -a services_to_check=(
            nvidia-fabricmanager
            nvidia-persistenced
            nvidia-dcgm
            xrdp
            xrdp-sesman
            k3s
            k3s-agent
            docker
            containerd
            ssh
        )
        local found=0
        for svc in "${services_to_check[@]}"; do
            if systemctl list-unit-files "${svc}.service" 2>/dev/null | grep -q "${svc}\.service"; then
                found=$((found + 1))
                if _is_service_active "$svc"; then
                    _report_line OK "service:$svc" "active"
                else
                    state=$(systemctl is-active "$svc" 2>/dev/null || echo unknown)
                    _report_line WARN "service:$svc" "installed but $state" \
                        "Service unit exists but is not active. May need start, reboot, or has failed." \
                        "sudo systemctl status $svc; journalctl -u $svc -n 80"
                fi
            fi
        done
        if (( found == 0 )); then
            _report_line SKIP "Service inventory" "none of the tracked services are installed"
        fi
    }

    _extract_install_errors() {
        [[ -f "$INSTALL_LOG" ]] || return 0
        local -a err_lines
        # Common error patterns from apt/dpkg/install steps. We want recent ones.
        mapfile -t err_lines < <(grep -nE \
            '^(E:|W: GPG error|dpkg: error|dpkg-deb: error|dpkg: warning: files list file|Errors were encountered|FATAL:|fatal error:|Sub-process .* returned an error|Setting up .* failed)' \
            "$INSTALL_LOG" 2>/dev/null | tail -n 40)
        if (( ${#err_lines[@]} == 0 )); then
            _report_line OK "Install log errors" "no E:/dpkg-error/FATAL lines in transcript"
        else
            _report_line WARN "Install log errors" "${#err_lines[@]} error-like line(s) in transcript (showing tail)" \
                "These were surfaced during install. Cross-reference with BAD lines above to determine impact." \
                "grep -nE 'E:|dpkg: error|FATAL' $INSTALL_LOG | less"
            local line
            for line in "${err_lines[@]}"; do
                # Truncate over-long lines
                if (( ${#line} > 240 )); then
                    line="${line:0:237}..."
                fi
                printf '      %s\n' "$line"
            done
        fi
    }

    _check_nvidia_driver_branch_consistency() {
        # Detect if more than one nvidia driver branch is installed on the host
        local -a branches
        mapfile -t branches < <(dpkg -l 2>/dev/null \
            | awk '/^ii / {pkg=$2; sub(/:.*$/, "", pkg);
                if (pkg ~ /^(nvidia-(driver|headless|utils|dkms|firmware|kernel-common)|libnvidia-(cfg1|compute|decode|encode|fbc1|gl|gpucomp))-[0-9]+/) {
                    if (match(pkg, /-[0-9]+(-open)?$/)) {
                        tag = substr(pkg, RSTART+1, RLENGTH-1)
                        sub(/-open$/, "", tag)
                        print tag
                    }
                }
            }' | sort -u)
        if (( ${#branches[@]} == 0 )); then
            _report_line SKIP "NVIDIA driver branch consistency" "no branched driver packages installed"
        elif (( ${#branches[@]} == 1 )); then
            _report_line OK "NVIDIA driver branch consistency" "single branch installed: ${branches[0]}"
        else
            _report_line BAD "NVIDIA driver branch consistency" "multiple branches installed: ${branches[*]}" \
                "Co-installed driver branches always cause runtime conflicts (file overlaps, module-load failures, libnvidia-*.so version skew)." \
                "for b in ${branches[*]}; do echo --- \$b ---; dpkg -l '*-'\$b 'lib*-'\$b 2>/dev/null | awk '/^ii/{print \$2}'; done; sudo apt-get remove --purge '*-595' (e.g.)"
        fi
    }

    {
        printf '\n'
        printf '================ INSTALL DIAGNOSTICS ================\n'
        printf 'Started UTC : %s\n' "$INSTALL_STARTED_AT"
        printf 'Finished UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'Exit code   : %s\n' "$exit_code"
        printf 'Script      : %s\n' "${BASH_SOURCE[0]}"
        printf 'Bundle dir  : %s\n' "${BUNDLE_DIR:-unknown}"
        printf 'Transcript  : %s\n' "$INSTALL_LOG"
        printf 'Diagnostics : %s\n' "$DIAG_LOG"
        printf '\n'

        if (( exit_code == 0 )); then
            _report_line OK "Script exit" "installer reached the final diagnostics section"
        elif (( ${#INSTALL_ERRORS[@]} > 0 )); then
            _report_line BAD "Script exit" "installer exited with code $exit_code" \
                "${INSTALL_ERRORS[*]}" \
                "rerun the command after fixing the error above"
        else
            _report_line BAD "Script exit" "installer exited with code $exit_code" \
                "A command failed under set -e before it was classified by the script." \
                "grep -nE 'ERROR|BAD|failed|not found|Unable|E: ' $INSTALL_LOG | tail -n 80"
        fi

        if [[ -d "${BUNDLE_DIR:-}/debs" && -d "${BUNDLE_DIR:-}/apps" ]]; then
            _report_line OK "Bundle layout" "found debs/ and apps/"
        else
            _report_line BAD "Bundle layout" "missing debs/ or apps/ under ${BUNDLE_DIR:-unknown}" \
                "The bundle was not extracted, BUNDLE_DIR points at the wrong location, or the .bin is incomplete." \
                "ls -lah ${BUNDLE_DIR:-.}; tar -tf all-airgap-bundle-ubuntu*.bin | head"
        fi

        if [[ -f "${BUNDLE_DIR:-}/meta/target.env" ]]; then
            # shellcheck disable=SC1091
            source "$BUNDLE_DIR/meta/target.env"
            source /etc/os-release
            local here_arch
            here_arch=$(dpkg --print-architecture 2>/dev/null || uname -m)
            if [[ "$ID" == "${BUNDLE_OS_ID:-}" && "$VERSION_ID" == "${BUNDLE_OS_VERSION:-}" && "$here_arch" == "${BUNDLE_ARCH:-}" ]]; then
                _report_line OK "Host compatibility" "$ID $VERSION_ID / $here_arch matches bundle metadata"
            elif [[ "$here_arch" != "${BUNDLE_ARCH:-}" ]]; then
                _report_line BAD "Host compatibility" "$ID $VERSION_ID / $here_arch does not match bundle arch ${BUNDLE_ARCH:-unknown}" \
                    "The bundle was built for a different CPU architecture." \
                    "cat $BUNDLE_DIR/meta/target.env; dpkg --print-architecture"
            else
                _report_line WARN "Host compatibility" "$ID $VERSION_ID differs from bundle ${BUNDLE_OS_ID:-unknown} ${BUNDLE_OS_VERSION:-unknown}" \
                    ".deb dependency versions can mismatch across Ubuntu releases." \
                    "cat /etc/os-release; cat $BUNDLE_DIR/meta/target.env"
            fi
        else
            _report_line WARN "Host compatibility" "meta/target.env is missing" \
                "Bundle metadata was not staged, so OS/version checks could not be repeated." \
                "ls -lah $BUNDLE_DIR/meta"
        fi

        _check_disk_space
        _check_kernel_state

        if [[ "$VERIFY_CHECKSUMS" == "1" && -f "${BUNDLE_DIR:-}/meta/SHA256SUMS" ]]; then
            _report_line OK "Bundle checksums" "startup SHA256 verification completed before installation continued" \
                "" "cd $BUNDLE_DIR && sha256sum --quiet -c meta/SHA256SUMS"
        elif [[ "$VERIFY_CHECKSUMS" == "0" ]]; then
            _report_line SKIP "Bundle checksums" "VERIFY_CHECKSUMS=0"
        else
            _report_line WARN "Bundle checksums" "meta/SHA256SUMS not found" \
                "Corruption cannot be checked from this installer run." \
                "find $BUNDLE_DIR/meta -maxdepth 1 -type f -print"
        fi

        if [[ -d "${BUNDLE_DIR:-}/wheels" ]]; then
            local wheel_dir wheel_name req_path req_count wheel_dir_count=0
            shopt -s nullglob
            for wheel_dir in "$BUNDLE_DIR"/wheels/*/; do
                [[ -d "$wheel_dir" ]] || continue
                wheel_dir_count=$((wheel_dir_count + 1))
                wheel_dir="${wheel_dir%/}"
                wheel_name="${wheel_dir##*/}"
                req_path="$wheel_dir/requirements.txt"
                if [[ -f "$req_path" ]]; then
                    req_count=$(grep -Ec '^[A-Za-z0-9_.-]+==' "$req_path" 2>/dev/null || true)
                    _report_line OK "Wheel requirements: $wheel_name" "$req_count package pin(s) written to $req_path"
                else
                    _report_line WARN "Wheel requirements: $wheel_name" "requirements.txt missing" \
                        "The wheelhouse manifest generator did not run or could not write to this directory." \
                        "ls -lah $wheel_dir; touch $wheel_dir/requirements.txt"
                fi
            done
            if (( wheel_dir_count == 0 )); then
                _report_line WARN "Wheel requirements" "no subdirectories found under $BUNDLE_DIR/wheels"
            fi
        else
            _report_line WARN "Wheel requirements" "$BUNDLE_DIR/wheels is missing" \
                "Python wheelhouses were not bundled." \
                "find $BUNDLE_DIR -maxdepth 2 -type d -name wheels -o -path '*/wheels/*'"
        fi

        mapfile -t _broken_pkgs < <(dpkg -l 2>/dev/null | awk '/^.[HUF]/ {print $2}')
        if (( ${#_broken_pkgs[@]} == 0 )); then
            _report_line OK "dpkg state" "no half-installed or unconfigured packages detected"
        else
            _report_line BAD "dpkg state" "${#_broken_pkgs[@]} broken package(s): ${_broken_pkgs[*]}" \
                "The local apt install or dpkg fallback did not finish cleanly." \
                "dpkg -l | grep -E '^..H|^..U|^..F'; sudo apt-get -f install -y --no-download"
        fi

        if [[ -f "${BUNDLE_DIR:-}/meta/apt-packages.txt" ]]; then
            local total_pkgs=0 pkg
            local -a missing_pkgs=()
            while IFS= read -r pkg || [[ -n "$pkg" ]]; do
                [[ -z "$pkg" || "$pkg" =~ ^[[:space:]]*# ]] && continue
                total_pkgs=$((total_pkgs + 1))
                _pkg_satisfied "$pkg" || missing_pkgs+=( "$pkg" )
            done < "$BUNDLE_DIR/meta/apt-packages.txt"
            if (( ${#missing_pkgs[@]} == 0 )); then
                _report_line OK "APT package list" "$total_pkgs listed package(s) are installed"
            else
                _report_line BAD "APT package list" "${#missing_pkgs[@]}/$total_pkgs listed package(s) missing: ${missing_pkgs[*]:0:30}" \
                    "The bundled local apt repo failed, dependencies were unavailable, or packages were removed later." \
                    "grep -n '^' $BUNDLE_DIR/meta/apt-packages.txt; dpkg-query -W <package>"
            fi
        else
            _report_line WARN "APT package list" "meta/apt-packages.txt is missing" \
                "The installer cannot verify the intended APT package set." \
                "ls -lah $BUNDLE_DIR/meta"
        fi

        _check_apt_source_cleanup

        if command -v "python${PYTHON_VER}" >/dev/null 2>&1; then
            _report_line OK "Python ${PYTHON_VER}" "$(python${PYTHON_VER} --version 2>&1)"
        else
            _report_line BAD "Python ${PYTHON_VER}" "python${PYTHON_VER} not found" \
                "The Python APT packages were not installed, or PATH does not expose the interpreter." \
                "command -v python${PYTHON_VER}; dpkg -l 'python${PYTHON_VER}*'"
        fi

        if [[ -d /usr/local/cuda-13.0 ]]; then
            _report_line OK "CUDA toolkit" "/usr/local/cuda-13.0 exists"
        else
            _report_line BAD "CUDA toolkit" "/usr/local/cuda-13.0 missing" \
                "CUDA toolkit packages did not install, or a later cleanup removed them." \
                "dpkg -l | grep -E 'cuda-toolkit|cuda-compiler|cuda-cudart'"
        fi
        _check_cuda_runtime_detail
        if command -v nvcc >/dev/null 2>&1; then
            _report_line OK "nvcc" "$(nvcc --version 2>/dev/null | grep release | sed 's/^ *//')"
        elif [[ "$BUILD_CUDA" == "1" ]]; then
            _report_line BAD "nvcc" "nvcc not found while BUILD_CUDA=1" \
                "CUDA compiler packages did not install or /usr/local/cuda/bin is not on PATH." \
                "source /etc/profile.d/cuda-13-0.sh; command -v nvcc"
        else
            _report_line SKIP "nvcc" "BUILD_CUDA=0"
        fi
        local _nccl_hdr="" _nccl_so=""
        for _h in /usr/include/nccl.h /usr/local/cuda/include/nccl.h; do
            [[ -f "$_h" ]] && { _nccl_hdr="$_h"; break; }
        done
        for _l in /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so \
                  /usr/local/cuda/lib64/libnccl.so.2; do
            [[ -e "$_l" ]] && { _nccl_so="$_l"; break; }
        done
        if [[ -n "$_nccl_hdr" && -n "$_nccl_so" ]]; then
            _nccl_ver=$(awk '/NCCL_MAJOR/ {maj=$3} /NCCL_MINOR/ {min=$3} /NCCL_PATCH/ {pat=$3} END {if(maj!="") print maj"."min"."pat}' "$_nccl_hdr" 2>/dev/null)
            _report_line OK "NCCL" "${_nccl_ver:-installed} (lib: $_nccl_so)"
        elif [[ -n "$_nccl_so" && -z "$_nccl_hdr" ]]; then
            _report_line WARN "NCCL" "runtime present at $_nccl_so but libnccl-dev (header) missing — llama.cpp build won't link NCCL" \
                "libnccl2 installed but libnccl-dev wasn't bundled or installed." \
                "dpkg -l libnccl2 libnccl-dev; ls /usr/include/nccl.h"
        elif [[ "$BUILD_CUDA" == "1" ]]; then
            _report_line BAD "NCCL" "libnccl2/libnccl-dev not installed — multi-GPU collectives will fall back to slow per-pair P2P" \
                "Bundle does not contain NCCL debs (re-run gather-all.sh) or apt install failed." \
                "dpkg -l libnccl2 libnccl-dev; ls $BUNDLE_DIR/debs/libnccl*"
        else
            _report_line SKIP "NCCL" "BUILD_CUDA=0"
        fi
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            local _gpu_count _gpu_names
            _gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
            _gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | tr '\n' '|' | sed 's/|$//')
            _report_line OK "NVIDIA driver" "$_gpu_count GPU(s) visible: ${_gpu_names:-?}"
        else
            _report_line BAD "NVIDIA driver" "nvidia-smi cannot see GPUs" \
                "Driver/kernel modules are not ready, a reboot is needed, Secure Boot blocked DKMS, or no GPU is present. See NVIDIA kmod / DKMS / Secure Boot checks below." \
                "nvidia-smi; mokutil --sb-state; dmesg | grep -i nvidia | tail -n 40"
        fi
        _check_nvidia_kmod_loaded
        _check_nvidia_driver_branch_consistency
        _check_dkms_detail
        _check_secure_boot
        if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
            if _is_service_active nvidia-fabricmanager; then
                _report_line OK "Fabric Manager" "nvidia-fabricmanager service is active"
            else
                # Multi-GPU NVSwitch boxes (>1 GPU visible) hard-require FM. Without
                # it, NCCL silently falls back from NVLink to PCIe. Mark BAD, not WARN.
                local _gpu_n=0
                if command -v nvidia-smi >/dev/null 2>&1; then
                    _gpu_n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
                fi
                if (( _gpu_n > 1 )); then
                    _report_line BAD "Fabric Manager" "nvidia-fabricmanager NOT active on $_gpu_n-GPU NVSwitch box — multi-GPU collectives will fall back to PCIe (silent ~20x slowdown)" \
                        "Most common cause: driver kmod not loaded — reboot needed after install. Other: NVSwitch hardware unreachable." \
                        "sudo systemctl status nvidia-fabricmanager; sudo journalctl -u nvidia-fabricmanager -n 80"
                else
                    _report_line WARN "Fabric Manager" "nvidia-fabricmanager service is installed but not active" \
                        "Service failed to start or needs a reboot after driver installation." \
                        "sudo systemctl status nvidia-fabricmanager"
                fi
            fi
        fi
        # nvidia-persistenced: required for cold-start CUDA latency on inference.
        if systemctl list-unit-files nvidia-persistenced.service >/dev/null 2>&1; then
            if _is_service_active nvidia-persistenced; then
                _report_line OK "Persistence daemon" "nvidia-persistenced active"
            else
                _report_line WARN "Persistence daemon" "nvidia-persistenced installed but not active — every cold CUDA call pays a 5-10s driver-reinit tax" \
                    "Service failed to start. Often resolves after reboot." \
                    "sudo systemctl status nvidia-persistenced"
            fi
        fi
        # DCGM: telemetry for XID/NVLink/thermal events.
        if systemctl list-unit-files nvidia-dcgm.service >/dev/null 2>&1; then
            if _is_service_active nvidia-dcgm; then
                _report_line OK "DCGM" "nvidia-dcgm active (use 'dcgmi discovery -l' / 'dcgmi diag -r 1')"
            else
                _report_line WARN "DCGM" "nvidia-dcgm installed but not active — no XID/NVLink/thermal telemetry" \
                    "Service failed to start." \
                    "sudo systemctl status nvidia-dcgm; journalctl -u nvidia-dcgm -n 40"
            fi
        fi
        # NVLink fabric health: on B300 SXM6, every GPU should have 18 active
        # NVLink-5 lanes. Any inactive lane means a degraded fabric path and
        # NCCL bandwidth dropping proportionally.
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            local _link_status _down_count
            _link_status=$(nvidia-smi nvlink --status 2>/dev/null || true)
            if [[ -n "$_link_status" ]]; then
                _down_count=$(printf '%s\n' "$_link_status" | grep -ciE 'inactive|<inactive>|disabled' || true)
                if (( _down_count == 0 )); then
                    local _active
                    _active=$(printf '%s\n' "$_link_status" | grep -c 'GB/s' || true)
                    _report_line OK "NVLink fabric" "$_active active NVLink(s); no inactive lanes"
                else
                    _report_line BAD "NVLink fabric" "$_down_count inactive NVLink lane(s) — multi-GPU bandwidth is degraded" \
                        "Hardware fault, FabricManager not initialized, or NVSwitch not present. Check FM status and dmesg for NVSwitch errors." \
                        "nvidia-smi nvlink --status; nvidia-smi topo -m; dmesg | grep -i nvlink"
                fi
            fi
        fi
        # nccl-tests: bundled benchmark binary indicates a usable multi-GPU
        # verification path. Don't run the benchmark in the installer (too long),
        # just confirm presence and tell the user the command.
        if command -v all_reduce_perf >/dev/null 2>&1; then
            _report_line OK "nccl-tests" "all_reduce_perf installed (run: all_reduce_perf -b 8 -e 8G -f 2 -g \$(nvidia-smi -L | wc -l))"
        elif [[ -f "${BUNDLE_DIR:-}/src/nccl-tests.tar.gz" ]]; then
            _report_line BAD "nccl-tests" "source bundled but binaries not installed — build failed during install" \
                "Re-run install-all.sh and watch the 'nccl-tests' step output." \
                "ls ~/nccl-tests/build; make -C ~/nccl-tests CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr MPI=0"
        else
            _report_line SKIP "nccl-tests" "not bundled (re-run gather-all.sh)"
        fi
        _check_nvidia_container_toolkit

        if [[ -f "${BUNDLE_DIR:-}/apps/vscode.deb" ]]; then
            command -v code >/dev/null 2>&1 \
                && _report_line OK "VS Code" "$(code --version 2>/dev/null | head -1)" \
                || _report_line WARN "VS Code" "code command not found" "VS Code may be installed but PATH/desktop integration needs re-login." "command -v code; dpkg -l code"
        else
            _report_line SKIP "VS Code" "apps/vscode.deb not present in bundle"
        fi
        if [[ -f "${BUNDLE_DIR:-}/apps/chrome.deb" ]]; then
            command -v google-chrome-stable >/dev/null 2>&1 \
                && _report_line OK "Chrome" "$(google-chrome-stable --version 2>/dev/null)" \
                || _report_line WARN "Chrome" "google-chrome-stable command not found" "Chrome .deb install failed or PATH is incomplete." "dpkg -l google-chrome-stable"
        else
            _report_line SKIP "Chrome" "apps/chrome.deb not present in bundle"
        fi
        if [[ -x /opt/firefox/firefox ]]; then
            _report_line OK "Firefox" "$(/opt/firefox/firefox --version 2>/dev/null || echo installed)"
        elif [[ -f "${BUNDLE_DIR:-}/apps/firefox.tar.xz" || -f "${BUNDLE_DIR:-}/apps/firefox.tar.bz2" ]]; then
            _report_line BAD "Firefox" "/opt/firefox/firefox missing" \
                "Firefox tarball extraction failed or /opt/firefox was removed (Mozilla now ships .tar.xz; old bundles may have saved an HTML error page as firefox.tar.bz2)." \
                "ls -lah /opt/firefox; ls -lah $BUNDLE_DIR/apps/firefox.tar.*; head -c 200 $BUNDLE_DIR/apps/firefox.tar.*"
        else
            _report_line SKIP "Firefox" "apps/firefox.tar.{xz,bz2} not present in bundle"
        fi
        command -v node >/dev/null 2>&1 \
            && _report_line OK "Node.js" "$(node --version 2>/dev/null)" \
            || _report_line BAD "Node.js" "node command not found" "Node.js tarball was missing, extraction failed, or symlinks were not created." "ls -lah /opt/nodejs/bin; command -v node"
        command -v npm >/dev/null 2>&1 \
            && _report_line OK "npm" "$(npm --version 2>/dev/null)" \
            || _report_line BAD "npm" "npm command not found" "Node.js install did not expose npm." "ls -lah /opt/nodejs/bin/npm /usr/local/bin/npm"
        if [[ -f "${BUNDLE_DIR:-}/apps/bun-linux-x64.zip" ]]; then
            command -v bun >/dev/null 2>&1 \
                && _report_line OK "Bun" "$(bun --version 2>/dev/null)" \
                || _report_line BAD "Bun" "bun command not found" "Bun zip extraction failed or /usr/local/bin symlink is missing." "command -v bun; ls -lah /usr/local/bin/bun"
        else
            _report_line SKIP "Bun" "apps/bun-linux-x64.zip not present in bundle"
        fi
        if [[ -f "${BUNDLE_DIR:-}/apps/opencode" ]]; then
            command -v opencode >/dev/null 2>&1 \
                && _report_line OK "Opencode" "opencode command exists" \
                || _report_line BAD "Opencode" "opencode command not found" "Bundled binary was not installed to /usr/local/bin." "ls -lah /usr/local/bin/opencode"
        else
            _report_line SKIP "Opencode" "apps/opencode not present in bundle"
        fi

        if [[ "$INSTALL_INFERENCE" == "1" ]]; then
            _check_venv_modules "LLM inference venv" "$INFERENCE_PREFIX/venv/bin/python" "torch vllm fastapi transformers" "The inference install"
            _check_pip_check "LLM inference venv" "$INFERENCE_PREFIX/venv/bin/python"
            _check_torch_cuda "LLM inference venv" "$INFERENCE_PREFIX/venv/bin/python"
        else
            _report_line SKIP "LLM inference venv" "INSTALL_INFERENCE=0"
        fi
        if [[ "$INSTALL_TRAINING" == "1" ]]; then
            _check_venv_modules "General training venv" "$TRAINING_PREFIX/venv/bin/python" "torch torch_geometric numpy h5py scipy" "The training install"
            _check_pip_check "General training venv" "$TRAINING_PREFIX/venv/bin/python"
            _check_torch_cuda "General training venv" "$TRAINING_PREFIX/venv/bin/python"
        else
            _report_line SKIP "General training venv" "INSTALL_TRAINING=0"
        fi
        if [[ "$INSTALL_JUPYTER" == "1" ]]; then
            _check_venv_modules "Jupyter venv" "$JUPYTER_PREFIX/venv/bin/python" "jupyterlab notebook ipykernel pandas" "The Jupyter install"
            [[ -x "$HOME/start-jupyter.sh" ]] \
                && _report_line OK "Jupyter launcher" "$HOME/start-jupyter.sh exists" \
                || _report_line WARN "Jupyter launcher" "$HOME/start-jupyter.sh missing" "Jupyter venv install was skipped or failed before launcher creation." "ls -lah $HOME/start-jupyter.sh"
        else
            _report_line SKIP "Jupyter venv" "INSTALL_JUPYTER=0"
        fi

        if [[ "$INSTALL_LLAMA" == "1" ]]; then
            if [[ -x "$LLAMA_PREFIX/build/bin/llama-cli" ]]; then
                local llama_version
                llama_version="$("$LLAMA_PREFIX/build/bin/llama-cli" --version 2>&1 | head -1)"
                _report_line OK "llama.cpp" "${llama_version:-llama-cli exists}"
            else
                _report_line BAD "llama.cpp" "llama-cli missing at $LLAMA_PREFIX/build/bin/llama-cli" \
                    "Source tarball was missing, cmake failed, build failed, or CUDA/nvcc setup blocked the build." \
                    "ls -lah $LLAMA_PREFIX/build/bin; cmake --build $LLAMA_PREFIX/build --config Release"
            fi
        else
            _report_line SKIP "llama.cpp" "INSTALL_LLAMA=0"
        fi

        if [[ "$INSTALL_DESKTOP" == "1" ]]; then
            command -v startxfce4 >/dev/null 2>&1 \
                && _report_line OK "XFCE4" "startxfce4 command exists" \
                || _report_line BAD "XFCE4" "startxfce4 command missing" "xfce4 packages did not install." "dpkg -l 'xfce4*'; command -v startxfce4"
            if command -v xrdp >/dev/null 2>&1; then
                if _is_service_active xrdp; then
                    _report_line OK "xrdp" "service is active on the host"
                else
                    _report_line WARN "xrdp" "xrdp command exists but service is not active" \
                        "systemd/service startup failed, or the host needs reboot/login manager cleanup." \
                        "sudo systemctl status xrdp || sudo service xrdp status"
                fi
                grep -q 'startxfce4' /etc/xrdp/startwm.sh 2>/dev/null \
                    && _report_line OK "xrdp session" "/etc/xrdp/startwm.sh launches XFCE4" \
                    || _report_line BAD "xrdp session" "startwm.sh does not launch XFCE4" "The desktop configuration block did not write the expected file." "sudo sed -n '1,80p' /etc/xrdp/startwm.sh"
            else
                _report_line BAD "xrdp" "xrdp command missing" "xrdp packages did not install." "dpkg -l xrdp xorgxrdp"
            fi
        else
            _report_line SKIP "Desktop" "INSTALL_DESKTOP=0"
        fi

        if [[ "$INSTALL_K3S" == "1" ]]; then
            if [[ "$K3S_ROLE" == "server" ]]; then
                _is_service_active k3s \
                    && _report_line OK "K3s server" "k3s service is active" \
                    || _report_line BAD "K3s server" "k3s service is not active" "K3s install failed or airgap images/configuration prevented startup." "sudo systemctl status k3s; journalctl -u k3s -n 200"
                if command -v kubectl >/dev/null 2>&1 && kubectl get nodes 2>/dev/null | grep -q " Ready "; then
                    _report_line OK "K3s nodes" "$(kubectl get nodes --no-headers 2>/dev/null | wc -l) node(s) Ready"
                else
                    _report_line WARN "K3s nodes" "no Ready node found from kubectl" "API server is not ready, kubeconfig is wrong, or node startup is still in progress." "kubectl get nodes -o wide"
                fi
            elif [[ "$K3S_ROLE" == "agent" ]]; then
                _is_service_active k3s-agent \
                    && _report_line OK "K3s agent" "k3s-agent service is active" \
                    || _report_line BAD "K3s agent" "k3s-agent service is not active" "Agent could not reach server, token was wrong, or service startup failed." "sudo systemctl status k3s-agent; journalctl -u k3s-agent -n 200"
            else
                _report_line BAD "K3s role" "INSTALL_K3S=1 but K3S_ROLE=$K3S_ROLE" \
                    "K3S_ROLE must be server or agent." \
                    "INSTALL_K3S=1 K3S_ROLE=server bash install-all.sh"
            fi
        else
            _report_line SKIP "K3s" "INSTALL_K3S=0"
        fi

        _check_install_prefixes
        _check_service_inventory
        _extract_install_errors

        if (( ${#INSTALL_WARNINGS[@]} == 0 )); then
            _report_line OK "Installer warnings" "no warnings were raised during the run"
        else
            _report_line WARN "Installer warnings" "${#INSTALL_WARNINGS[@]} warning(s) were raised during the run" \
                "These are the warnings emitted while installing; some may be acceptable skips." \
                "read this diagnostics file and the terminal output above it"
            local idx=1 warning
            for warning in "${INSTALL_WARNINGS[@]}"; do
                printf '      warning %02d: %s\n' "$idx" "$warning"
                idx=$((idx + 1))
            done
        fi

        # ── WHAT FAILED — grouped recap so the user doesn't have to scroll ──
        if (( ${#_bad_recap[@]} > 0 )); then
            printf '\n'
            printf '################ WHAT FAILED (BAD) ################\n'
            printf 'These items did NOT install correctly. Address them in order:\n\n'
            local _idx=1 _entry _area _detail _cause _check
            for _entry in "${_bad_recap[@]}"; do
                IFS='|' read -r _area _detail _cause _check <<< "$_entry"
                printf '  %2d. [%s]\n' "$_idx" "$_area"
                printf '      what : %s\n' "$_detail"
                [[ -n "$_cause" ]] && printf '      why  : %s\n' "$_cause"
                [[ -n "$_check" ]] && printf '      fix  : %s\n' "$_check"
                printf '\n'
                _idx=$((_idx + 1))
            done
            # Persist a machine-/human-readable failures file
            {
                printf '# Install failures captured at %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                printf '# Transcript: %s\n#\n' "$INSTALL_LOG"
                _idx=1
                for _entry in "${_bad_recap[@]}"; do
                    IFS='|' read -r _area _detail _cause _check <<< "$_entry"
                    printf '[%d] AREA: %s\n    DETAIL: %s\n    CAUSE: %s\n    CHECK: %s\n\n' \
                        "$_idx" "$_area" "$_detail" "$_cause" "$_check"
                    _idx=$((_idx + 1))
                done
            } > "$FAILURE_REPORT" 2>/dev/null
        fi
        if (( ${#_warn_recap[@]} > 0 )); then
            printf '\n'
            printf '############### WHAT NEEDS REVIEW (WARN) ###############\n'
            printf 'These items installed but have caveats:\n\n'
            local _widx=1 _wentry _warea _wdetail _wcause _wcheck
            for _wentry in "${_warn_recap[@]}"; do
                IFS='|' read -r _warea _wdetail _wcause _wcheck <<< "$_wentry"
                printf '  %2d. [%s] %s\n' "$_widx" "$_warea" "$_wdetail"
                [[ -n "$_wcause" ]] && printf '      why  : %s\n' "$_wcause"
                [[ -n "$_wcheck" ]] && printf '      fix  : %s\n' "$_wcheck"
                _widx=$((_widx + 1))
            done
            printf '\n'
        fi

        printf '\n'
        printf 'Result counts: OK=%d WARN=%d BAD=%d SKIP=%d\n' "$ok_count" "$warn_count" "$bad_count" "$skip_count"
        if (( bad_count > 0 || exit_code != 0 )); then
            printf 'Overall: NEEDS ATTENTION  (%d failed item(s) listed above)\n' "$bad_count"
        elif (( warn_count > 0 )); then
            printf 'Overall: INSTALLED WITH WARNINGS  (%d warning(s) listed above)\n' "$warn_count"
        else
            printf 'Overall: INSTALLED OK  (all %d check(s) passed)\n' "$ok_count"
        fi
        if (( bad_count > 0 )); then
            printf 'Failures file: %s\n' "$FAILURE_REPORT"
        fi
        printf '\n'
        printf 'Useful next checks:\n'
        printf '  tail -n 200 %s\n' "$INSTALL_LOG"
        printf '  grep -nE '\''ERROR|BAD|failed|not found|Unable|E: '\'' %s | tail -n 80\n' "$INSTALL_LOG"
        printf '  nvidia-smi && nvcc --version\n'
        printf '  source %s/venv/bin/activate && python -m pip check\n' "$INFERENCE_PREFIX"
        printf '  source %s/venv/bin/activate && python -m pip check\n' "$TRAINING_PREFIX"
        printf '  %s/build/bin/llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080\n' "$LLAMA_PREFIX"
        printf '  bash %s/start-jupyter.sh\n' "$HOME"
        printf '  sudo systemctl status xrdp\n'
        printf '=====================================================\n'
        printf 'Full transcript saved to: %s\n' "$INSTALL_LOG"
        printf 'Diagnostics summary saved to: %s\n' "$DIAG_LOG"
        printf '\n'
    } | tee "$DIAG_LOG" || true
}

_on_exit() {
    local rc=$?
    trap - EXIT
    print_final_diagnostics "$rc" || true
    exit "$rc"
}
trap _on_exit EXIT

# ---------------------------------------------------------------------------
# Auto-extract bundle if not already extracted
# ---------------------------------------------------------------------------
if [[ ! -d "$BUNDLE_DIR/debs" || ! -d "$BUNDLE_DIR/apps" ]]; then
    # Auto-detect the .bin bundle next to the script if not explicitly set
    if [[ -z "$BUNDLE_BIN" ]]; then
        shopt -s nullglob
        _bins=( "$SCRIPT_DIR"/all-airgap-bundle-ubuntu*.bin )
        (( ${#_bins[@]} > 0 )) && BUNDLE_BIN="${_bins[0]}"
    fi
    [[ -f "$BUNDLE_BIN" ]] \
        || die "No extracted bundle and no .bin file found next to installer. Set BUNDLE_DIR or BUNDLE_BIN."
    log "Extracting $BUNDLE_BIN -> $SCRIPT_DIR"
    tar -xf "$BUNDLE_BIN" -C "$SCRIPT_DIR" \
        || die "Failed to extract $BUNDLE_BIN"
    shopt -s nullglob
    for cand in "$SCRIPT_DIR"/*/; do
        if [[ -d "${cand}debs" && -d "${cand}apps" ]]; then
            BUNDLE_DIR="${cand%/}"
            break
        fi
    done
    [[ -d "$BUNDLE_DIR/debs" && -d "$BUNDLE_DIR/apps" ]] \
        || die "Extracted bundle but could not find debs/ + apps/ under $SCRIPT_DIR"
    log "Using bundle: $BUNDLE_DIR"
fi

[[ -d "$BUNDLE_DIR/debs" && -d "$BUNDLE_DIR/apps" ]] \
    || die "Bundle not found under $BUNDLE_DIR (expected debs/ and apps/ subdirs)"

generate_wheelhouse_requirements

# ============================================================================
# 0) Sanity checks
# ============================================================================
step "Sanity checks"

# Determine the real human user (handles both `bash install-all.sh` and
# `sudo bash install-all.sh`). All scratch-owned artifacts will use this
# identity so they're usable without sudo afterwards.
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_GROUP="$(id -gn "$TARGET_USER" 2>/dev/null || echo "$TARGET_USER")"
log "Install identity: $TARGET_USER:$TARGET_GROUP (running uid=$EUID)"

# Ensure SCRATCH_ROOT exists and is writable by TARGET_USER BEFORE the
# venv/llama.cpp steps try mkdir into it. If we're root, do it directly;
# if we're a normal user, escalate via sudo.
if [[ ! -d "$SCRATCH_ROOT" ]]; then
    log "Creating $SCRATCH_ROOT"
    sudo mkdir -p "$SCRATCH_ROOT" || die "Could not create $SCRATCH_ROOT"
fi
# chown each time so the script self-heals if something else flipped ownership.
sudo chown "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT" \
    || warn "Could not chown $SCRATCH_ROOT to $TARGET_USER:$TARGET_GROUP."
sudo chmod 0775 "$SCRATCH_ROOT" 2>/dev/null || true
log "$SCRATCH_ROOT ready (owner $TARGET_USER:$TARGET_GROUP)"


if [[ -f "$BUNDLE_DIR/meta/target.env" ]]; then
    # shellcheck disable=SC1091
    source "$BUNDLE_DIR/meta/target.env"
    source /etc/os-release
    HERE_ARCH=$(dpkg --print-architecture 2>/dev/null || uname -m)
    log "Bundle built on : $BUNDLE_OS_ID $BUNDLE_OS_VERSION / $BUNDLE_ARCH / py$BUNDLE_PYTHON"
    log "This host       : $ID $VERSION_ID / $HERE_ARCH"
    [[ "$ID" == "$BUNDLE_OS_ID" && "$VERSION_ID" == "$BUNDLE_OS_VERSION" ]] \
        || warn "OS mismatch — .deb installation may fail. Abort with Ctrl-C if unsure."
    [[ "$HERE_ARCH" == "$BUNDLE_ARCH" ]] \
        || die "Architecture mismatch ($HERE_ARCH vs $BUNDLE_ARCH). Bundle is incompatible."
fi

if [[ "$VERIFY_CHECKSUMS" == "1" && -f "$BUNDLE_DIR/meta/SHA256SUMS" ]]; then
    log "Verifying SHA256 checksums"
    ( cd "$BUNDLE_DIR" && sha256sum --quiet -c meta/SHA256SUMS ) \
        || die "Checksum verification failed — bundle may be corrupted."
fi

# ============================================================================
# 0.5) PURGE EXISTING CUDA
#   Permanently removes any CUDA toolkit already on the system before
#   installing the bundled version, preventing version-conflict failures.
#   The NVIDIA kernel driver is NOT touched.
#   Skipped in PHASE=2: CUDA was already purged + reinstalled in PHASE=1, and
#   re-purging now would remove the live install.
# ============================================================================
if _phase_runs_apt; then
step "Purging existing CUDA toolkit"

# Collect all installed cuda toolkit packages (any version). Also catch
# cuda-toolkit-config-common, which owns /etc/ld.so.conf.d/000_cuda.conf and
# will otherwise collide with cuda-cudart-* debs from older CUDA branches.
mapfile -t _cuda_pkgs < <(dpkg -l \
    | awk '/^ii/ && (/cuda/ || $2 == "cuda-toolkit-config-common") && !/nvidia-driver|nvidia-utils|nvidia-kernel|libnvidia-|nvidia-fabricmanager|nvidia-persistenced/ {print $2}')

if (( ${#_cuda_pkgs[@]} > 0 )); then
    log "Removing ${#_cuda_pkgs[@]} existing CUDA packages: ${_cuda_pkgs[*]}"
    sudo apt-get remove --purge -y "${_cuda_pkgs[@]}" 2>/dev/null || true
    # Keep the running kernel and its headers from being autoremoved — apt
    # marks them auto-installed when linux-generic was the explicit install.
    sudo apt-mark manual \
        "linux-image-$(uname -r)" "linux-headers-$(uname -r)" \
        linux-generic linux-headers-generic linux-image-generic 2>/dev/null || true
    sudo apt-get autoremove --purge -y 2>/dev/null || true
else
    log "No existing CUDA toolkit packages found."
fi

# Clean up /usr/local/cuda* directories left by any prior manual installs
for cuda_dir in /usr/local/cuda-*/; do
    [[ -e "$cuda_dir" ]] || continue
    log "Removing leftover directory: $cuda_dir"
    sudo rm -rf "$cuda_dir"
done
# Remove the /usr/local/cuda symlink — will be re-created by the new install
sudo rm -f /usr/local/cuda
# /etc/ld.so.conf.d/000_cuda.conf is shipped by cuda-toolkit-config-common and
# also by ancient cuda-cudart-11-* debs. Drop it so the new package can write it.
sudo rm -f /etc/ld.so.conf.d/000_cuda.conf

log "CUDA purge complete."

# Detect installed NVIDIA driver branch and warn if the bundle ships a
# different branch — installing two driver branches on top of each other
# is the source of most "conflicting packages" dpkg failures.
_host_nv_branch=""
# dpkg -l may show package names with :arch suffix (e.g. libnvidia-compute-580:amd64).
# Strip that before matching so the branch number regex anchors correctly.
_host_nv_branch=$(dpkg -l 2>/dev/null \
    | awk '/^ii / {
        pkg = $2; sub(/:.*$/, "", pkg)
        if (pkg ~ /^(nvidia-(driver|headless|utils|dkms|firmware|kernel-common)|libnvidia-(cfg1|compute|decode|encode|fbc1|gl|gpucomp))-[0-9]+/) {
            if (match(pkg, /-[0-9]+(-open)?$/)) {
                tag = substr(pkg, RSTART+1, RLENGTH-1)
                sub(/-open$/, "", tag)
                print tag
            }
        }
    }' | sort -u | head -n 1)
if [[ -n "$_host_nv_branch" ]]; then
    log "Existing NVIDIA driver branch on host: $_host_nv_branch"
    shopt -s nullglob
    _bundle_nv_branches=$(printf '%s\n' "$BUNDLE_DIR"/debs/nvidia-{driver,headless,utils,dkms,firmware,kernel-common,kernel-source}-[0-9]*_*.deb \
        | sed -nE 's@.*/(nvidia-(driver|headless|utils|dkms|firmware|kernel-common|kernel-source))-([0-9]+)(-open)?_.*@\3@p' \
        | sort -u)
    shopt -u nullglob
    if [[ -n "$_bundle_nv_branches" ]] && ! grep -qx "$_host_nv_branch" <<<"$_bundle_nv_branches"; then
        warn "NVIDIA driver branch mismatch: host has $_host_nv_branch, bundle has $(echo "$_bundle_nv_branches" | tr '\n' ' ')"
        warn "Old branch packages will block the new ones. Purging nvidia-*-$_host_nv_branch first."
        mapfile -t _old_nv_pkgs < <(dpkg -l 2>/dev/null \
            | awk -v b="$_host_nv_branch" '/^ii / {
                pkg = $2; sub(/:.*$/, "", pkg)
                if (pkg ~ ("(nvidia|libnvidia)-.*-" b "(-open)?$")) print pkg
            }')
        if (( ${#_old_nv_pkgs[@]} > 0 )); then
            log "Removing ${#_old_nv_pkgs[@]} old NVIDIA packages: ${_old_nv_pkgs[*]}"
            sudo apt-get remove --purge -y "${_old_nv_pkgs[@]}" 2>/dev/null || true
            sudo apt-mark manual \
                "linux-image-$(uname -r)" "linux-headers-$(uname -r)" \
                linux-generic linux-headers-generic linux-image-generic 2>/dev/null || true
            sudo apt-get autoremove --purge -y 2>/dev/null || true
        fi
    fi
fi

# ── Stale-leftover NVIDIA package purge (matches host & bundle branch) ────
# Catches the failure mode where dpkg rejects bundle debs with:
#   "libnvidia-cfg1:amd64 conflicts with libnvidia-cfg1
#    libnvidia-cfg1-580:amd64 provides libnvidia-cfg1 and is to be installed"
# This happens when EITHER:
#   (a) The host has a partial-install of the *same* branch with a different
#       deb version than the bundle ships (e.g. host has 580.107, bundle has
#       580.159) — the version skew confuses dpkg's "Provides:" handling.
#   (b) Leftover unversioned libnvidia-* / nvidia-* packages from a prior
#       driver still own the unversioned virtuals that the bundle's
#       branch-suffixed packages also want to provide.
# The branch-mismatch purge above misses both cases because it only fires
# when the host and bundle branches differ.
#
# Detect the bundle's exact deb version for the matching branch (any branch
# that's also in $_bundle_nv_branches), then enumerate host packages that:
#   - are branch-suffixed for that branch AND have a different version, OR
#   - are unversioned NVIDIA libs / utils that "provide" the same virtuals.
# Purge with --force-depends so dependent packages come along; the bundle
# will reinstall them at consistent versions in the next apt step.
shopt -s nullglob
_bundle_nv_versions=""
for _br in $_bundle_nv_branches; do
    for _f in "$BUNDLE_DIR"/debs/libnvidia-compute-${_br}_*.deb \
              "$BUNDLE_DIR"/debs/nvidia-utils-${_br}_*.deb; do
        [[ -e "$_f" ]] || continue
        _v=$(basename "$_f" | sed -nE 's/^[^_]+_([^_]+)_.*\.deb$/\1/p')
        [[ -n "$_v" ]] && _bundle_nv_versions+="${_br}:${_v} "
        break
    done
done
shopt -u nullglob

mapfile -t _stale_nv_pkgs < <(dpkg -l 2>/dev/null \
    | awk -v bvs="$_bundle_nv_versions" '
        BEGIN {
            n = split(bvs, arr, " ")
            for (i = 1; i <= n; i++) {
                if (arr[i] == "") continue
                split(arr[i], kv, ":")
                want_ver[kv[1]] = kv[2]
            }
        }
        /^ii / {
            pkg = $2; sub(/:.*$/, "", pkg); ver = $3
            # (a) branch-suffixed package whose version mismatches the bundle
            for (br in want_ver) {
                pat = "^(nvidia|libnvidia)-.*-" br "(-open)?$"
                if (pkg ~ pat && ver != want_ver[br]) {
                    print pkg; next
                }
            }
            # (b) unversioned NVIDIA library / util leftover from a prior
            # branch that still owns the virtual the new branch wants.
            # libnvidia-cfg1, nvidia-dkms-kernel, nvidia-kernel-common,
            # nvidia-kernel-source, nvidia-driver-binary, xserver-xorg-video-nvidia
            # were the 5 packages that caused dpkg failures in phase1.log —
            # they were unversioned leftovers that "provide" the same virtuals
            # as the new branch-suffixed packages.
            if (pkg ~ /^libnvidia-(cfg1|compute|decode|encode|fbc1|gl|gpucomp|nscq|opencl|nvvm|extra)$/) {
                print pkg; next
            }
            if (pkg ~ /^nvidia-(compute-utils|prime|settings|utils|firmware|kernel-common|kernel-source|dkms-kernel|driver-binary)$/) {
                print pkg; next
            }
            if (pkg == "xserver-xorg-video-nvidia") {
                print pkg; next
            }
        }')

if (( ${#_stale_nv_pkgs[@]} > 0 )); then
    warn "Stale/unversioned NVIDIA packages detected — these cause 'conflicts with libnvidia-*' dpkg errors:"
    printf '       %s\n' "${_stale_nv_pkgs[@]}" >&2
    log "Purging ${#_stale_nv_pkgs[@]} stale NVIDIA package(s) before bundle install"
    sudo apt-get remove --purge -y "${_stale_nv_pkgs[@]}" 2>/dev/null \
        || sudo dpkg --remove --force-depends "${_stale_nv_pkgs[@]}" 2>/dev/null \
        || true
    sudo apt-mark manual \
        "linux-image-$(uname -r)" "linux-headers-$(uname -r)" \
        linux-generic linux-headers-generic linux-image-generic 2>/dev/null || true
    sudo apt-get autoremove --purge -y 2>/dev/null || true
fi
else
    log "PHASE=2: skipping CUDA purge (already done in PHASE=1)"
fi  # _phase_runs_apt

# ============================================================================
# 0.6) PRE-PURGE PACKAGES THAT CONFLICT WITH BUNDLE VERSIONS
#   vim-runtime >= 2:9.1 carries a Breaks: vim-tiny (< 2:9.1) declaration.
#   Since vim-tiny is part of the Ubuntu base install and our bundle ships
#   a newer vim, dpkg refuses to unpack vim-runtime without this purge.
#   Skipped in PHASE=2 (already done in PHASE=1).
# ============================================================================
if _phase_runs_apt; then
step "Pre-purging packages that conflict with bundle versions"
_pre_purge_pkgs=()
# Static list: packages known to cause Breaks/conflict failures during dpkg
# pass 1 when the bundle ships a newer version than the host (vim family) or
# the bundle's metapackage layout differs from what's already on the host
# (unbranched nvidia drivers; spell-check post-install that needs ispell deps
# we deliberately don't bundle).
for _pkg in vim vim-tiny vim-common vim-runtime \
            nvidia-open nvidia-driver-open \
            libnvidia-egl-wayland21 \
            dictionaries-common iamerican ispell ienglish-common \
            aspell aspell-en hunspell hunspell-en-us \
            ibritish wamerican wbritish \
            gnome-flashback gnome-flashback-common; do
    if dpkg -s "$_pkg" >/dev/null 2>&1; then
        _pre_purge_pkgs+=("$_pkg")
    fi
done
# Off-branch NVIDIA packages: if the host carries any nvidia-*-<N>* whose <N>
# is not NVIDIA_DRIVER_BRANCH, purge it so the bundle's branched packages can
# install without "would break <off-branch pkg>" errors.
if [[ -n "${NVIDIA_DRIVER_BRANCH:-}" ]]; then
    while IFS= read -r _ob; do
        [[ -n "$_ob" ]] && _pre_purge_pkgs+=("$_ob")
    done < <(dpkg -l 2>/dev/null \
        | awk -v b="$NVIDIA_DRIVER_BRANCH" '/^ii / {
            pkg=$2; sub(/:.*$/, "", pkg)
            if (match(pkg, /^(nvidia|libnvidia)-.*-([0-9]+)(-open)?$/, m)) {
                if (m[2] != b) print pkg
            }
        }')
fi
if (( ${#_pre_purge_pkgs[@]} > 0 )); then
    log "Pre-purging: ${_pre_purge_pkgs[*]}"
    sudo apt-get remove --purge -y "${_pre_purge_pkgs[@]}" 2>/dev/null \
        || sudo dpkg --remove --force-depends "${_pre_purge_pkgs[@]}" 2>/dev/null \
        || true
else
    log "Nothing to pre-purge."
fi
else
    log "PHASE=2: skipping pre-purge (already done in PHASE=1)"
fi  # _phase_runs_apt

# ============================================================================
# 0.7) BUNDLE CLEANUP
#   The captured bundle may contain debs that can't possibly install on this
#   host: Mellanox DOCA packages pinning ibverbs-providers 2601 against the
#   stock Ubuntu 50.0, unbranched nvidia metapackages from a different driver
#   branch, GNOME flashback bits whose deps weren't in the closure, and
#   spell-check chains whose post-install scripts need ispell support files
#   we don't ship. We also frequently see two micro-versions of the same
#   package in the bundle (e.g. vim-runtime 7.12 AND 7.13) when the gather
#   host did an apt update mid-run.
#
#   Prune the unfixable debs, dedupe to the newest version per package, and
#   regenerate the Packages metadata so the bundled apt repo is consistent.
#   Skipped in PHASE=2 (already done in PHASE=1; re-pruning is a no-op anyway).
# ============================================================================
if _phase_runs_apt; then
step "Bundle cleanup (pruning conflict-causing debs and deduping versions)"

_BUNDLE_CLEANED=0
if [[ -d "$BUNDLE_DIR/debs" ]] && sudo test -w "$BUNDLE_DIR/debs"; then
    _BUNDLE_CLEANED=1
elif [[ -d "$BUNDLE_DIR/debs" ]] && sudo chmod u+w "$BUNDLE_DIR/debs" 2>/dev/null; then
    _BUNDLE_CLEANED=1
fi

if (( _BUNDLE_CLEANED )); then
    _bundle_prune() {
        local desc="$1"; shift
        local glob count=0
        shopt -s nullglob
        for glob in "$@"; do
            for f in "$BUNDLE_DIR"/debs/$glob; do
                [[ -e "$f" ]] || continue
                sudo rm -f "$f"
                count=$((count + 1))
            done
        done
        shopt -u nullglob
        (( count > 0 )) && log "Pruned $count $desc deb(s)"
        return 0
    }

    # ── Mellanox DOCA / MLNX_OFED: depends on ibverbs-providers 2601.0+,
    #    which conflicts with Ubuntu's stock 50.0 already installed. The
    #    gather host had the DOCA repo enabled but the bundle only has the
    #    stock ibverbs. Drop the DOCA packages; install MLNX_OFED separately.
    _bundle_prune "Mellanox DOCA (ibverbs conflict)" \
        "doca-*.deb" "ibverbs-utils_*.deb" "libibverbs-dev_*.deb" \
        "perftest_*.deb" "rdma-core_*.deb" "infiniband-diags_*.deb"

    # ── Unversioned NVIDIA metapackages: resolve to whatever branch apt picks
    #    last (often a newer one), which contradicts the explicit branch pin.
    _bundle_prune "unversioned NVIDIA meta" \
        "nvidia-driver-open_*.deb" "nvidia-open_*.deb" \
        "nvidia-driver_*.deb" "nvidia-dkms-open_*.deb" \
        "nvidia-kernel-source-open_*.deb" "nvidia-kernel-common_*.deb"

    # ── Off-branch nvidia-*/libnvidia-* debs: keep only those matching the
    #    bundle's explicit driver branch (auto-detected from
    #    nvidia-driver-pinning-<N>_*.deb, or from $NVIDIA_DRIVER_BRANCH).
    _bundle_branch="${NVIDIA_DRIVER_BRANCH:-}"
    if [[ -z "$_bundle_branch" ]]; then
        shopt -s nullglob
        for _pin in "$BUNDLE_DIR"/debs/nvidia-driver-pinning-[0-9]*_*.deb; do
            [[ -e "$_pin" ]] || continue
            if [[ "$(basename "$_pin")" =~ ^nvidia-driver-pinning-([0-9]+)_ ]]; then
                _bundle_branch="${BASH_REMATCH[1]}"
                break
            fi
        done
        shopt -u nullglob
    fi
    if [[ -n "$_bundle_branch" ]]; then
        log "NVIDIA driver branch in bundle: $_bundle_branch (off-branch debs will be pruned)"
        shopt -s nullglob
        _off_branch_count=0
        for f in "$BUNDLE_DIR"/debs/nvidia-*.deb "$BUNDLE_DIR"/debs/libnvidia-*.deb; do
            [[ -e "$f" ]] || continue
            base=$(basename "$f")
            case "$base" in
                nvidia-fabricmanager_*|libnvidia-nscq_*|nvidia-container-*|libnvidia-container*) continue ;;
            esac
            # Branch number is the leading integer of the .deb version field,
            # which appears right after the first underscore: name_NNN.MM.PP-...
            if [[ "$base" =~ _([0-9]+)\. ]]; then
                if [[ "${BASH_REMATCH[1]}" != "$_bundle_branch" ]]; then
                    sudo rm -f "$f"
                    _off_branch_count=$((_off_branch_count + 1))
                fi
            fi
        done
        shopt -u nullglob
        (( _off_branch_count > 0 )) && log "Pruned $_off_branch_count off-branch NVIDIA deb(s)"
    fi

    # ── Packages with deps that aren't in the bundle. Removing the parent
    #    stops apt from trying to install them. None of these are critical
    #    for a GPU compute / dev server.
    _bundle_prune "gnome-flashback (missing nautilus/libgnome-panel3)" \
        "gnome-flashback*.deb"
    _bundle_prune "spell-check chain (missing ispell/ienglish-common)" \
        "iamerican_*.deb" "ispell_*.deb" "dictionaries-common_*.deb" \
        "ibritish_*.deb" "wamerican_*.deb" "wbritish_*.deb" \
        "aspell_*.deb" "aspell-en_*.deb" \
        "hunspell_*.deb" "hunspell-en-us_*.deb" "hunspell-en-*.deb" \
        "libaspell*_*.deb" "libhunspell*_*.deb"
    _bundle_prune "plymouth-label (missing fonts-ubuntu)" \
        "plymouth-label_*.deb"

    # ── systemd family version anchoring ────────────────────────────────
    #    Ubuntu ships systemd as a tight family where every -resolved /
    #    -sysv / libnss-systemd / libsystemd-shared deb has a strict
    #    "Depends: systemd (= <exact>)". When the gather host did an apt
    #    update mid-run (security micro-version bump), the bundle ends up
    #    with mixed micro-versions, and any version that doesn't match the
    #    bundle's anchor systemd will be unresolvable on the target.
    #
    #    Pick the version of the bundle's systemd_*.deb as the anchor; if
    #    systemd_*.deb isn't bundled, fall back to the host's installed
    #    systemd version. Then drop every systemd-family deb whose version
    #    doesn't match the anchor.
    _systemd_family=(
        systemd systemd-sysv systemd-resolved systemd-timesyncd
        systemd-container systemd-cryptsetup systemd-oomd systemd-coredump
        systemd-journal-remote systemd-userdbd systemd-boot systemd-repart
        libsystemd0 libsystemd-shared libpam-systemd libnss-systemd
        libudev1 udev
    )
    _anchor_ver=""
    shopt -s nullglob
    for _sf in "$BUNDLE_DIR"/debs/systemd_*_*.deb; do
        [[ -e "$_sf" ]] || continue
        _base=$(basename "$_sf")
        _rest="${_base#systemd_}"
        _anchor_ver="${_rest%_*}"
        break
    done
    shopt -u nullglob
    if [[ -z "$_anchor_ver" ]] && command -v dpkg-query >/dev/null 2>&1; then
        _anchor_ver=$(dpkg-query -W -f='${Version}' systemd 2>/dev/null || true)
    fi
    if [[ -n "$_anchor_ver" ]]; then
        log "systemd anchor version: $_anchor_ver (other versions in family will be pruned)"
        _systemd_mismatch=0
        shopt -s nullglob
        for _pkg in "${_systemd_family[@]}"; do
            for _sf in "$BUNDLE_DIR"/debs/"${_pkg}"_*_*.deb; do
                [[ -e "$_sf" ]] || continue
                _base=$(basename "$_sf")
                _rest="${_base#${_pkg}_}"
                _ver="${_rest%_*}"
                if [[ "$_ver" != "$_anchor_ver" ]]; then
                    sudo rm -f "$_sf"
                    _systemd_mismatch=$((_systemd_mismatch + 1))
                fi
            done
        done
        shopt -u nullglob
        (( _systemd_mismatch > 0 )) && log "Pruned $_systemd_mismatch systemd-family deb(s) that did not match anchor $_anchor_ver"
        # Verify critical family members survived the prune. libpam-systemd is
        # essential — its absence causes "Connection closed by port 22" after
        # login because PAM session open fails silently.
        shopt -s nullglob
        for _critical_sf in systemd libpam-systemd libsystemd0 libudev1; do
            _sf_debs=( "$BUNDLE_DIR/debs/${_critical_sf}_"*_*.deb )
            if (( ${#_sf_debs[@]} == 0 )); then
                warn "CRITICAL: ${_critical_sf} deb MISSING from bundle after systemd family prune!"
                warn "  Anchor was $_anchor_ver. If the host's ${_critical_sf} differs,"
                warn "  PAM/login will be broken after install. Re-gather the bundle on a"
                warn "  host with a consistent systemd version, or manually copy the matching deb."
            fi
        done
        shopt -u nullglob
    fi

    # ── Dedupe: when the bundle has multiple versions of the same package,
    #    keep only the newest (per dpkg --compare-versions).
    declare -A _newest_ver _newest_path
    shopt -s nullglob
    for f in "$BUNDLE_DIR"/debs/*.deb; do
        base=$(basename "$f")
        # Filename format: <pkg>_<version>_<arch>.deb. Splitting on '_' is
        # safe because debian package names don't contain underscores.
        pkg="${base%%_*}"
        rest="${base#${pkg}_}"
        ver="${rest%_*}"
        prev_ver="${_newest_ver[$pkg]:-}"
        if [[ -z "$prev_ver" ]]; then
            _newest_ver[$pkg]="$ver"
            _newest_path[$pkg]="$f"
        elif dpkg --compare-versions "$ver" gt "$prev_ver" 2>/dev/null; then
            sudo rm -f "${_newest_path[$pkg]}"
            log "Dedupe: dropped older $pkg $prev_ver (kept $ver)"
            _newest_ver[$pkg]="$ver"
            _newest_path[$pkg]="$f"
        else
            sudo rm -f "$f"
            log "Dedupe: dropped older $pkg $ver (kept $prev_ver)"
        fi
    done
    shopt -u nullglob
    unset _newest_ver _newest_path

    # ── Regenerate Packages metadata so the apt repo reflects the prune.
    if command -v dpkg-scanpackages >/dev/null 2>&1; then
        log "Regenerating bundled apt repo metadata"
        if ( cd "$BUNDLE_DIR/debs" && sudo dpkg-scanpackages . /dev/null > Packages ) 2>/dev/null; then
            sudo bash -c "gzip -9c '$BUNDLE_DIR/debs/Packages' > '$BUNDLE_DIR/debs/Packages.gz'" \
                || warn "Could not regenerate Packages.gz; apt will fall back to plain Packages."
        else
            warn "dpkg-scanpackages failed; the bundle's Packages index may be stale."
        fi
    else
        warn "dpkg-scanpackages not available; Packages index left as-is (may include pruned debs)."
    fi
else
    warn "Bundle debs/ is not writable; skipping bundle cleanup. Re-run as root or chmod u+w \"$BUNDLE_DIR/debs\"."
fi
else
    log "PHASE=2: skipping bundle cleanup (already done in PHASE=1)"
fi  # _phase_runs_apt

# ============================================================================
# 1) APT PACKAGES
#   The biggest step. Skipped in PHASE=2 — all packages were installed in
#   PHASE=1; re-running would be a waste of time. (Re-running is harmless if
#   you genuinely want to: dpkg would no-op on already-installed packages.)
# ============================================================================
if _phase_runs_apt; then
step "APT packages"

shopt -s nullglob
all_debs=( "$BUNDLE_DIR"/debs/*.deb )
(( ${#all_debs[@]} > 0 )) || die "No .deb files found in $BUNDLE_DIR/debs/"
APT_OK=0
if [[ -f "$BUNDLE_DIR/debs/Packages" || -f "$BUNDLE_DIR/debs/Packages.gz" ]]; then
    log "Installing from bundled local apt repository"
    # apt drops privileges to the _apt sandbox user when reading Packages files.
    # Stage a public local repo path so private home-directory permissions do
    # not block Packages access.
    case "$APT_REPO_DIR" in
        /tmp/airgap-bundle-debs|/var/tmp/airgap-bundle-debs|*/airgap-bundle-debs) ;;
        *) die "APT_REPO_DIR must end with airgap-bundle-debs for safe cleanup: $APT_REPO_DIR" ;;
    esac
    # Set o+x on each ancestor of BUNDLE_DIR (traversal only — no read access).
    sudo rm -rf "$APT_REPO_DIR"
    sudo mkdir -p "$APT_REPO_DIR"
    if sudo cp -al "$BUNDLE_DIR/debs/." "$APT_REPO_DIR/" 2>/dev/null; then
        log "Staged local apt repo with hardlinks: $APT_REPO_DIR"
    else
        warn "Hardlink staging failed; copying local apt repo to $APT_REPO_DIR"
        sudo cp -a "$BUNDLE_DIR/debs/." "$APT_REPO_DIR/"
    fi
    sudo chmod -R a+rX "$APT_REPO_DIR" 2>/dev/null || true
    printf 'deb [trusted=yes] file:%s ./\n' "$APT_REPO_DIR" \
        | sudo tee /etc/apt/sources.list.d/airgap-bundle.list > /dev/null
    # Acquire::Languages=none skips Translation-en lookups (we don't ship them in
    # the bundle so they always 404 and clutter the log with Err lines).
    sudo apt-get update \
        -o Dir::Etc::sourcelist="sources.list.d/airgap-bundle.list" \
        -o Dir::Etc::sourceparts="-" \
        -o APT::Get::List-Cleanup="0" \
        -o Acquire::Languages="none"

    mapfile -t apt_pkgs < <(grep -vE '^\s*#|^\s*$' "$BUNDLE_DIR/meta/apt-packages.txt" 2>/dev/null || true)
    if (( ${#apt_pkgs[@]} > 0 )); then
        # --force-overwrite tolerates shared-directory entries across CUDA
        # debs (libcusolver-dev-13-0 et al. all claim /usr/local/cuda-13.0/lib64).
        if sudo apt-get install -y --allow-downgrades --no-install-recommends \
            -o Dir::Etc::sourcelist="sources.list.d/airgap-bundle.list" \
            -o Dir::Etc::sourceparts="-" \
            -o APT::Get::List-Cleanup="0" \
            -o Acquire::Languages="none" \
            -o Dir::Cache::archives="$APT_REPO_DIR" \
            -o Dpkg::Options::="--force-overwrite" \
            -o Dpkg::Options::="--force-breaks" \
            -o Dpkg::Options::="--force-depends-version" \
            -o Dpkg::Options::="--force-confold" \
            "${apt_pkgs[@]}"; then
            APT_OK=1
        else
            warn "Local apt install failed; falling back to dpkg multi-pass."
        fi
    else
        warn "meta/apt-packages.txt missing; falling back to dpkg multi-pass."
    fi
fi

if [[ "$APT_OK" != "1" ]]; then
    if [[ "$ALLOW_DPKG_FALLBACK" != "1" ]]; then
        die "Local apt install failed; refusing dpkg fallback by default because force-installing every bundled .deb can leave broken packages. Fix the apt error above or rerun with ALLOW_DPKG_FALLBACK=1."
    fi
    warn "ALLOW_DPKG_FALLBACK=1 set; force-installing all bundled .debs."
    log "Installing ${#all_debs[@]} .deb packages (multi-pass for dependency ordering)"
    for pass in 1 2 3; do
        log "dpkg pass $pass"
        # --force-overwrite is required for the CUDA debs: each cuda-*-13-0
        # package claims /usr/local/cuda-13.0/{lib64,include}, so dpkg refuses
        # the second one without it.
        # --force-breaks tolerates Breaks constraints from the captured debs.
        # Real-world case: bundle ships vim-runtime 7.13 but the gather host
        # cached vim-tiny 7.12. Without --force-breaks dpkg refuses vim-runtime
        # ("would break vim-tiny") and leaves the package in a broken state.
        sudo dpkg -i \
            --force-depends --force-depends-version \
            --force-overwrite --force-breaks --force-confold "${all_debs[@]}" 2>&1 \
            | grep -v '^\(Reading\|Selecting\|Preparing\|Unpacking\|Setting up\|Processing\)' || true
        broken=$(dpkg -l | awk '/^.[HUF]/ {print $2}' | wc -l)
        (( broken == 0 )) && break
        log "  $broken packages in broken state, retrying..."
    done

    # Fix up remaining dependency issues using only bundled debs.
    # Dry-run first: apt-get -f install can remove packages to resolve deps
    # even with --no-download. Surface any such removal before it happens.
    _finstall_dry=$(sudo apt-get -f install --dry-run --no-download \
        -o Dir::Cache::archives="$APT_REPO_DIR" 2>/dev/null || true)
    if printf '%s\n' "$_finstall_dry" | grep -qE '^Remv '; then
        warn "apt-get -f install would REMOVE the following package(s) — check carefully:"
        printf '%s\n' "$_finstall_dry" | grep '^Remv' | sed 's/^/    /' >&2
    fi
    sudo apt-get -f install -y --no-download \
        -o Dir::Cache::archives="$APT_REPO_DIR" \
        -o Dpkg::Options::="--force-overwrite" 2>/dev/null || true
fi

# Belt-and-suspenders: regardless of which path succeeded above, try once more
# to drive dpkg out of any half-configured state, then run apt-get -f install
# off the local airgap repo. This recovers from postinst-script failures
# (e.g. dictionaries-common) that left a package in 'U' (unpacked) state
# even though the apt resolve itself succeeded.
broken_initial=$(dpkg -l | awk '/^.[HUF]/ {print $2}' | wc -l)
if (( broken_initial > 0 )); then
    log "Post-install repair: $broken_initial package(s) still in non-installed state; running configure + -f install"
    sudo dpkg --configure --pending 2>/dev/null || true
    if [[ -f /etc/apt/sources.list.d/airgap-bundle.list ]]; then
        # Dry-run first to detect if apt wants to REMOVE packages to fix deps.
        _finstall_dry2=$(sudo apt-get -f install --dry-run --no-download --allow-downgrades \
            -o Dir::Etc::sourcelist="sources.list.d/airgap-bundle.list" \
            -o Dir::Etc::sourceparts="-" \
            -o APT::Get::List-Cleanup="0" \
            -o Acquire::Languages="none" \
            -o Dir::Cache::archives="$APT_REPO_DIR" 2>/dev/null || true)
        if printf '%s\n' "$_finstall_dry2" | grep -qE '^Remv '; then
            warn "apt-get -f install (repair) would REMOVE the following — proceeding anyway but check carefully:"
            printf '%s\n' "$_finstall_dry2" | grep '^Remv' | sed 's/^/    /' >&2
        fi
        sudo apt-get -f install -y --no-download --allow-downgrades \
            -o Dir::Etc::sourcelist="sources.list.d/airgap-bundle.list" \
            -o Dir::Etc::sourceparts="-" \
            -o APT::Get::List-Cleanup="0" \
            -o Acquire::Languages="none" \
            -o Dir::Cache::archives="$APT_REPO_DIR" \
            -o Dpkg::Options::="--force-overwrite" \
            -o Dpkg::Options::="--force-breaks" \
            -o Dpkg::Options::="--force-depends-version" \
            -o Dpkg::Options::="--force-confold" 2>/dev/null || true
    else
        sudo apt-get -f install -y --no-download 2>/dev/null || true
    fi
    # Final resort: mark packages whose postinst failed as installed anyway so
    # dpkg stops blocking subsequent installs. We only do this for known-safe
    # offenders whose postinst depends on optional data files (dictionaries-common
    # ispell symlink). Packages with a real failure stay broken and surface in
    # the diagnostics block.
    for _broken in $(dpkg -l | awk '/^.[HUF]/ {print $2}'); do
        case "$_broken" in
            dictionaries-common|iamerican|ispell|ienglish-common| \
            aspell|aspell-en|aspell-*|hunspell|hunspell-en-us|hunspell-en-*| \
            ibritish|wamerican|wbritish)
                log "Force-removing post-install-failed cosmetic package: $_broken"
                sudo dpkg --remove --force-remove-reinstreq --force-depends "$_broken" 2>/dev/null || true
                ;;
        esac
    done
fi

broken_final=$(dpkg -l | awk '/^.[HUF]/ {print $2}' | wc -l)
(( broken_final == 0 )) || warn "$broken_final packages still broken. Check: dpkg -l | grep -E '^..H|^..U|^..F'"

# ── Pin NVIDIA packages at installed versions ────────────────────────────────
# FabricManager requires an EXACT version match with the driver. Unattended-
# upgrades or a stray `apt upgrade` that bumps only one side silently breaks
# multi-GPU NVLink fabric. Hold everything at the installed version.
if _phase_runs_apt; then
    mapfile -t _nv_installed_hold < <(dpkg -l 2>/dev/null \
        | awk '/^ii / {pkg=$2; sub(/:.*$/,"",pkg)
                       if (pkg ~ /^(nvidia|libnvidia)-/) print pkg}')
    if (( ${#_nv_installed_hold[@]} > 0 )); then
        sudo apt-mark hold "${_nv_installed_hold[@]}" 2>/dev/null || true
        log "Held ${#_nv_installed_hold[@]} NVIDIA package(s) at installed version (prevents FM/driver version skew)"
    fi
    # Also blacklist from unattended-upgrades as a belt-and-suspenders measure.
    sudo mkdir -p /etc/apt/apt.conf.d
    sudo tee /etc/apt/apt.conf.d/99-nvidia-no-auto-upgrade > /dev/null <<'NVAPTCONF'
// Prevent unattended-upgrades from touching NVIDIA packages.
// nvidia-fabricmanager requires an exact version match with the kernel driver;
// auto-upgrading one without the other silently breaks multi-GPU NVLink fabric.
Unattended-Upgrade::Package-Blacklist {
    "nvidia-";
    "libnvidia-";
    "nvidia-fabricmanager";
    "nvidia-persistenced";
};
NVAPTCONF
    log "Disabled unattended-upgrades for NVIDIA packages (/etc/apt/apt.conf.d/99-nvidia-no-auto-upgrade)"
fi

# Clean up the airgap apt source so future `apt update` runs don't try to
# refresh a path that may have been wiped. We only remove it if the install
# finished without dpkg state still broken — otherwise a follow-up rerun may
# need the source to retry. The diagnostic check at the end will OK this.
if (( broken_final == 0 )) && [[ -f /etc/apt/sources.list.d/airgap-bundle.list ]]; then
    log "Removing airgap apt source (install complete)"
    sudo rm -f /etc/apt/sources.list.d/airgap-bundle.list
    sudo rm -rf "$APT_REPO_DIR"
fi

# Report DKMS modules that failed to build. These are almost always pre-existing
# kernel/OFED modules (e.g. iser, isert, mlnx-nfsrdma) that can't compile against
# the newly installed kernel headers — not caused by this installer.
if command -v dkms &>/dev/null; then
    mapfile -t _dkms_bad < <(dkms status 2>/dev/null \
        | awk -F': ' '$2 != "installed" && NF==2 {print $0}' || true)
    for _dm in "${_dkms_bad[@]}"; do
        warn "DKMS: $_dm  (pre-existing module; not installed by this script)"
    done
    if (( ${#_dkms_bad[@]} > 0 )); then
        warn "DKMS build failures above are typically Mellanox OFED/kernel version mismatches on the host — rebuild manually: sudo dkms autoinstall"
    fi
fi

for bin in python3 pip3; do
    command -v "$bin" >/dev/null || warn "Expected '$bin' not found after apt install."
done

# Check python3.12 specifically
if command -v "python${PYTHON_VER}" >/dev/null; then
    log "Python $PYTHON_VER OK: $(python${PYTHON_VER} --version)"
else
    warn "python${PYTHON_VER} not found after install. You may need to run: sudo apt install python${PYTHON_VER}"
fi

if [[ "$WHEEL_REQS_GENERATED" != "1" ]]; then
    generate_wheelhouse_requirements
fi
else
    log "PHASE=2: skipping APT install (already done in PHASE=1)"
fi  # _phase_runs_apt

# ============================================================================
# 1.5) CUDA / NVIDIA post-install
#   Always run — in PHASE=1 it sets up env (no driver kmod loaded yet, so
#   service starts mostly fail and that's OK), in PHASE=2 it re-applies env
#   (idempotent) and the service starts actually succeed now that the driver
#   is loaded post-reboot.
# ============================================================================
step "CUDA / NVIDIA post-install"

# Auto-detect the CUDA root. NVIDIA's cuda-toolkit-13-0 deb may land at
# /usr/local/cuda-13.0 OR /usr/local/cuda-13 depending on the exact package
# revision, and CUDA 13.x ships libraries under targets/x86_64-linux/lib/
# rather than the legacy lib64/. Probe both layouts so the env is configured
# correctly regardless of which one the apt install produced.
_CUDA_ROOT=""
for _cand in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
    [[ -d "$_cand" && -x "$_cand/bin/nvcc" ]] || continue
    _CUDA_ROOT=$(readlink -f "$_cand")
    break
done

if [[ -n "$_CUDA_ROOT" ]]; then
    # Find a lib directory that actually contains libcudart.so*. CUDA 13 uses
    # targets/x86_64-linux/lib; older CUDA used lib64.
    _CUDA_LIB=""
    for _lc in \
        "$_CUDA_ROOT/targets/$(uname -m)-linux/lib" \
        "$_CUDA_ROOT/lib64" \
        "$_CUDA_ROOT/lib"; do
        if compgen -G "$_lc/libcudart.so*" >/dev/null; then
            _CUDA_LIB="$_lc"; break
        fi
    done

    sudo ln -sfn "$_CUDA_ROOT" /usr/local/cuda
    sudo tee /etc/profile.d/cuda.sh > /dev/null <<EOF
export PATH=$_CUDA_ROOT/bin\${PATH:+:\${PATH}}
${_CUDA_LIB:+export LD_LIBRARY_PATH=$_CUDA_LIB\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}}
EOF
    sudo chmod 0644 /etc/profile.d/cuda.sh
    # Syntax-check the generated script — a bad _CUDA_ROOT expansion would break
    # every SSH session on next login (profile.d sourced unconditionally).
    if ! sudo bash -n /etc/profile.d/cuda.sh 2>/dev/null; then
        warn "cuda.sh has syntax errors! Removing to prevent broken login shells."
        sudo rm -f /etc/profile.d/cuda.sh
    fi
    # Drop any stale per-version drop-ins from a previous run that pinned the
    # wrong path (e.g. /usr/local/cuda-13.0/lib64 when the install is actually
    # cuda-13). Otherwise both files get sourced and PATH gets noisy.
    sudo rm -f /etc/profile.d/cuda-13-0.sh /etc/ld.so.conf.d/cuda-13-0.conf

    # /etc/profile.d/*.sh is sourced ONLY by login shells. Non-login interactive
    # bash (SSH "bash" subshells, VS Code terminals, child shells) reads
    # /etc/bash.bashrc instead — so nvcc / LD_LIBRARY_PATH would be unset there.
    # Inject an idempotent source line into /etc/bash.bashrc.
    if [[ -f /etc/bash.bashrc ]] && ! grep -q '/etc/profile.d/cuda.sh' /etc/bash.bashrc; then
        sudo tee -a /etc/bash.bashrc > /dev/null <<'BASHRC'

# Source CUDA env in non-login interactive shells (added by install-all.sh)
[ -f /etc/profile.d/cuda.sh ] && . /etc/profile.d/cuda.sh
BASHRC
        log "Patched /etc/bash.bashrc to source CUDA env in non-login shells."
    fi

    # Belt-and-suspenders: symlink the most-used CUDA binaries into /usr/local/bin
    # so they work in ANY environment (cron jobs, systemd units, scripts run with
    # `bash -c`) regardless of whether the shell sourced profile.d.
    for _bin in nvcc cuobjdump nvdisasm cuda-gdb compute-sanitizer nvprof; do
        if [[ -x "$_CUDA_ROOT/bin/$_bin" ]]; then
            sudo ln -sf "$_CUDA_ROOT/bin/$_bin" "/usr/local/bin/$_bin" 2>/dev/null || true
        fi
    done
    log "Symlinked CUDA binaries (nvcc, etc.) into /usr/local/bin/"

    if [[ -n "$_CUDA_LIB" ]]; then
        echo "$_CUDA_LIB" | sudo tee /etc/ld.so.conf.d/cuda.conf > /dev/null
        sudo ldconfig || true
        export LD_LIBRARY_PATH="$_CUDA_LIB${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        warn "CUDA libs not found under $_CUDA_ROOT (tried targets/.../lib, lib64, lib); LD_LIBRARY_PATH not set."
    fi
    export PATH="$_CUDA_ROOT/bin${PATH:+:${PATH}}"
    log "CUDA environment configured: root=$_CUDA_ROOT lib=${_CUDA_LIB:-<none>}"
else
    warn "No CUDA install found under /usr/local/cuda*; CUDA toolkit may not have installed."
fi

if command -v nvcc >/dev/null 2>&1; then
    log "nvcc: $(nvcc --version | grep release | sed 's/^ *//')"
else
    warn "nvcc not found; llama.cpp CUDA build will fall back to CPU unless toolkit install is fixed."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L >/dev/null 2>&1 \
        && log "nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPU(s) visible" \
        || warn "nvidia-smi exists but no GPU is visible yet; reboot may be required after driver install."
else
    warn "nvidia-smi not found; NVIDIA driver packages may be missing."
fi

if command -v mokutil >/dev/null 2>&1 && mokutil --sb-state 2>/dev/null | grep -qi enabled; then
    warn "Secure Boot is enabled; NVIDIA DKMS modules may need MOK enrollment/signing before GPUs work."
fi

# ── Fabric Manager ──────────────────────────────────────────────────────────
# Without FabricManager running, NVSwitch is uninitialized and multi-GPU NCCL
# falls back from NVLink (~700 GB/s) to PCIe (~25 GB/s) — silently. The driver
# kernel module must be loaded before FM can talk to the switch hardware. On a
# fresh install (before reboot into the new kernel module) the start will fail;
# the enable persists across reboot so it auto-starts once kmod is loaded.
if systemctl list-unit-files nvidia-fabricmanager.service >/dev/null 2>&1; then
    sudo systemctl enable nvidia-fabricmanager 2>/dev/null || true
    if lsmod 2>/dev/null | grep -q '^nvidia '; then
        if sudo systemctl restart nvidia-fabricmanager 2>/dev/null; then
            log "nvidia-fabricmanager: started"
        else
            warn "nvidia-fabricmanager failed to start. Recent journal:"
            sudo journalctl -u nvidia-fabricmanager -n 30 --no-pager 2>&1 | sed 's/^/    /' >&2 || true
            warn "Service is enabled and will retry after reboot. If it still fails, check NVSwitch fabric init in dmesg."
        fi
    else
        log "nvidia driver kmod not loaded yet; nvidia-fabricmanager enabled but not started (will auto-start after reboot)."
    fi
fi

# ── nvidia-persistenced ─────────────────────────────────────────────────────
# Keeps the driver loaded across idle periods. Without it, every cold CUDA call
# pays a 5-10s driver-reinit tax (terrible for inference latency). The daemon
# also keeps NVSwitch state warm, so it pairs with FabricManager.
if systemctl list-unit-files nvidia-persistenced.service >/dev/null 2>&1; then
    sudo systemctl enable nvidia-persistenced 2>/dev/null || true
    sudo systemctl restart nvidia-persistenced 2>/dev/null \
        && log "nvidia-persistenced: started" \
        || warn "Could not start nvidia-persistenced; check 'systemctl status nvidia-persistenced'."
fi

# Also set persistence mode at runtime via nvidia-smi (belt-and-suspenders;
# persistenced does this on supported drivers but on some branches the daemon
# only watches for new clients without proactively enabling PM mode).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    sudo nvidia-smi -pm 1 >/dev/null 2>&1 \
        && log "GPU persistence mode: enabled (nvidia-smi -pm 1)" \
        || warn "Could not enable GPU persistence mode."
fi

# ── DCGM (Data Center GPU Manager) ──────────────────────────────────────────
# NVIDIA's telemetry daemon. Tracks XID events, NVLink errors, thermal
# throttling, ECC errors. Essential for catching silent multi-GPU degradation.
if systemctl list-unit-files nvidia-dcgm.service >/dev/null 2>&1; then
    sudo systemctl enable nvidia-dcgm 2>/dev/null || true
    sudo systemctl restart nvidia-dcgm 2>/dev/null \
        && log "nvidia-dcgm: started" \
        || warn "Could not start nvidia-dcgm; check 'systemctl status nvidia-dcgm'."
fi

# ── PHASE 1 stop point ─────────────────────────────────────────────────────
# Everything past this point assumes the NVIDIA driver kmod is loaded and
# NVSwitch is initialized. On a fresh driver install that requires a reboot.
if [[ "$PHASE" == "1" ]]; then
    step "PHASE=1 complete — running pre-reboot hardening"

    # Pre-reboot hardening: locks down everything that can prevent the box from
    # coming back to an SSH-able state after the reboot — MOK enrollment queue,
    # network-online infinite wait, broken shell init that closes sshd sessions
    # ("Connection closed by ... port 22" right after password), kernel/initramfs
    # consistency, GRUB visibility, lightdm respawn loops, etc.
    _pre_reboot_sh="$SCRIPT_DIR/pre-reboot.sh"
    _pre_reboot_rc=0
    if [[ -f "$_pre_reboot_sh" ]]; then
        sudo bash "$_pre_reboot_sh" || _pre_reboot_rc=$?
    else
        warn "$_pre_reboot_sh not found — SKIPPING pre-reboot hardening."
        warn "Box may not come back up to a working SSH state after reboot."
        _pre_reboot_rc=2
    fi

    printf '\n'
    if (( _pre_reboot_rc == 0 )); then
        printf '\033[1;33m  ╔══════════════════════════════════════════════════════════════╗\033[0m\n'
        printf '\033[1;33m  ║  PHASE 1 finished. Driver + CUDA + system services installed. ║\033[0m\n'
        printf '\033[1;33m  ║  Pre-reboot hardening passed.                                ║\033[0m\n'
        printf '\033[1;33m  ║                                                              ║\033[0m\n'
        printf '\033[1;33m  ║  Next steps:                                                 ║\033[0m\n'
        printf '\033[1;33m  ║    1. sudo reboot                                            ║\033[0m\n'
        printf '\033[1;33m  ║    2. nvidia-smi  (verify driver loaded, 8 GPUs visible)     ║\033[0m\n'
        printf '\033[1;33m  ║    3. PHASE=2 bash %-40s ║\033[0m\n' "$(basename "${BASH_SOURCE[0]}")"
        printf '\033[1;33m  ║                                                              ║\033[0m\n'
        printf '\033[1;33m  ║  PHASE=2 will skip apt install (already done) and run:       ║\033[0m\n'
        printf '\033[1;33m  ║    apps (vscode, chrome, etc), llama.cpp build,              ║\033[0m\n'
        printf '\033[1;33m  ║    inference + training venvs, nccl-tests, diagnostics       ║\033[0m\n'
        printf '\033[1;33m  ╚══════════════════════════════════════════════════════════════╝\033[0m\n'
    else
        printf '\033[1;31m  ╔══════════════════════════════════════════════════════════════╗\033[0m\n'
        printf '\033[1;31m  ║  PHASE 1 finished, BUT pre-reboot hardening FAILED.          ║\033[0m\n'
        printf '\033[1;31m  ║  DO NOT REBOOT until the [FAIL] items above are fixed —      ║\033[0m\n'
        printf '\033[1;31m  ║  the box may not come back up to a usable SSH session.       ║\033[0m\n'
        printf '\033[1;31m  ║                                                              ║\033[0m\n'
        printf '\033[1;31m  ║  After fixing, re-verify with:                               ║\033[0m\n'
        printf '\033[1;31m  ║    sudo bash %-46s ║\033[0m\n' "$(basename "$_pre_reboot_sh")"
        printf '\033[1;31m  ║  Then: sudo reboot && PHASE=2 bash %-25s ║\033[0m\n' "$(basename "${BASH_SOURCE[0]}")"
        printf '\033[1;31m  ╚══════════════════════════════════════════════════════════════╝\033[0m\n'
    fi
    printf '\n'
    exit "$_pre_reboot_rc"
fi

# ============================================================================
# 2) VS CODE
# ============================================================================
step "VS Code"

if [[ -f "$BUNDLE_DIR/apps/vscode.deb" ]]; then
    log "Installing VS Code"
    sudo dpkg -i "$BUNDLE_DIR/apps/vscode.deb" || \
        sudo apt-get -f install -y --no-download 2>/dev/null || true
    command -v code >/dev/null && log "VS Code: $(code --version | head -1)" \
        || warn "VS Code installed but 'code' not in PATH (may need re-login or GUI launch)."
else
    warn "apps/vscode.deb not found; skipping."
fi

# ============================================================================
# 3) GOOGLE CHROME
# ============================================================================
step "Google Chrome"

if [[ -f "$BUNDLE_DIR/apps/chrome.deb" ]]; then
    log "Installing Google Chrome"
    sudo dpkg -i "$BUNDLE_DIR/apps/chrome.deb" || \
        sudo apt-get -f install -y --no-download 2>/dev/null || true
    command -v google-chrome-stable >/dev/null \
        && log "Chrome: $(google-chrome-stable --version)" \
        || warn "Chrome installed but binary not found in PATH."
else
    warn "apps/chrome.deb not found; skipping."
fi

# ============================================================================
# 4) FIREFOX
#   Mozilla switched the linux tarball format from .tar.bz2 to .tar.xz starting
#   with Firefox 135 (early 2025). Accept either layout, and detect the actual
#   compression from magic bytes — a bundle gathered against the bz2 URL after
#   the switch will save an HTTP error page as firefox.tar.bz2 and bz2 will
#   refuse to open it. Treat extraction failure as a warning so the remaining
#   install steps still run.
# ============================================================================
step "Firefox"

FF_TARBALL=""
for _ff_cand in firefox.tar.xz firefox.tar.bz2; do
    if [[ -f "$BUNDLE_DIR/apps/$_ff_cand" ]]; then
        FF_TARBALL="$BUNDLE_DIR/apps/$_ff_cand"
        break
    fi
done

if [[ -n "$FF_TARBALL" ]]; then
    FF_VER=$(cat "$BUNDLE_DIR/apps/firefox.version" 2>/dev/null || echo "unknown")
    log "Installing Firefox $FF_VER to /opt/firefox (source: ${FF_TARBALL##*/})"
    if sudo mkdir -p /opt/firefox; then
        _ff_magic=$(head -c 6 "$FF_TARBALL" 2>/dev/null | od -An -tx1 | tr -d ' \n')
        case "$_ff_magic" in
            fd377a585a00*) _ff_tar_flag="-xJf" ; _ff_fmt="xz" ;;   # FD 37 7A 58 5A 00
            425a68*)       _ff_tar_flag="-xjf" ; _ff_fmt="bz2" ;;  # 'BZh'
            1f8b*)         _ff_tar_flag="-xzf" ; _ff_fmt="gzip" ;; # 1F 8B (rare)
            *)             _ff_tar_flag=""    ; _ff_fmt="unknown" ;;
        esac
    else
        warn "Could not create /opt/firefox; skipping Firefox."
        _ff_magic=""
        _ff_tar_flag=""
        _ff_fmt="unknown"
    fi
    if [[ -z "$_ff_tar_flag" ]]; then
        warn "Firefox archive has unrecognized format (magic=${_ff_magic:-empty}); skipping. Re-run gather-all.sh to refresh it."
    elif sudo tar "$_ff_tar_flag" "$FF_TARBALL" -C /opt/firefox --strip-components=1; then
        sudo ln -sf /opt/firefox/firefox /usr/local/bin/firefox
        log "Firefox: $(/opt/firefox/firefox --version 2>/dev/null || echo 'installed') (${_ff_fmt})"
        sudo tee /usr/share/applications/firefox-manual.desktop > /dev/null <<'EOF'
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
        log "Firefox desktop entry created."
    else
        warn "Firefox extraction failed (${_ff_fmt} archive at $FF_TARBALL); skipping. Bundle may be stale — re-run gather-all.sh."
    fi
else
    warn "apps/firefox.tar.{xz,bz2} not found; skipping."
fi

# ============================================================================
# 4.5) GUI SANDBOX HARDENING (Ubuntu 24.04+)
#   Ubuntu 24.04 ships kernel.apparmor_restrict_unprivileged_userns=1, which
#   blocks clone(CLONE_NEWUSER) for unconfined binaries. Chrome, VS Code
#   (Electron) and Firefox all use this for their sandbox and silently exit
#   without it — the visible symptom is "typed `code` in terminal, nothing
#   happens." Firefox at least surfaces it as:
#     Sandbox: CanCreateUserNamespace() clone() failure: EACCES
#   Persist the unrestricted setting so it survives reboot.
# ============================================================================
step "GUI sandbox hardening (apparmor userns)"

if [[ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
    if sudo tee /etc/sysctl.d/60-apparmor-userns.conf > /dev/null <<'SYSCTL'
# Allow unprivileged user namespaces so Chrome/VS Code/Firefox sandboxes work.
# Set by install-all.sh — required for Electron apps on Ubuntu 24.04+.
kernel.apparmor_restrict_unprivileged_userns = 0
SYSCTL
    then
        sudo sysctl --system >/dev/null 2>&1 \
            || sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 >/dev/null 2>&1 \
            || warn "Could not apply apparmor userns sysctl; reboot or run: sudo sysctl --system"
        log "AppArmor unprivileged userns restriction disabled (Chrome/Code/Firefox sandbox)."
    else
        warn "Could not write /etc/sysctl.d/60-apparmor-userns.conf; Chrome/VS Code may fail to launch."
    fi
else
    log "Kernel does not expose apparmor_restrict_unprivileged_userns; skipping (likely pre-24.04)."
fi

# ============================================================================
# 5) OPENCODE
# ============================================================================
step "Opencode"

if [[ -f "$BUNDLE_DIR/apps/opencode" ]]; then
    OC_VER=$(cat "$BUNDLE_DIR/apps/opencode.version" 2>/dev/null || echo "unknown")
    log "Installing Opencode $OC_VER -> /usr/local/bin/opencode"
    if sudo install -m 0755 "$BUNDLE_DIR/apps/opencode" /usr/local/bin/opencode; then
        opencode --version 2>/dev/null && log "Opencode: OK" || warn "opencode --version failed (may be OK if flag differs)."
    else
        warn "Opencode install failed; continuing."
    fi
elif [[ -f "$BUNDLE_DIR/apps/opencode.MISSING" ]]; then
    warn "Opencode binary was not downloaded automatically during gather."
    warn "Download manually from https://github.com/sst/opencode/releases"
    warn "and copy to /usr/local/bin/opencode"
else
    warn "apps/opencode not found; skipping."
fi

# ============================================================================
# 6) NODE.JS + NPM
# ============================================================================
step "Node.js + npm"

if [[ -f "$BUNDLE_DIR/apps/nodejs.tar.xz" ]]; then
    NODE_VER=$(cat "$BUNDLE_DIR/apps/nodejs.version" 2>/dev/null || echo "unknown")
    log "Installing Node.js v${NODE_VER} to /opt/nodejs"
    if sudo tar -tf "$BUNDLE_DIR/apps/nodejs.tar.xz" >/dev/null 2>&1; then
        sudo rm -rf /opt/nodejs
        sudo mkdir -p /opt/nodejs
        if sudo tar -xJf "$BUNDLE_DIR/apps/nodejs.tar.xz" -C /opt/nodejs --strip-components=1; then
            sudo ln -sf /opt/nodejs/bin/node /usr/local/bin/node || warn "Could not create node symlink."
            sudo ln -sf /opt/nodejs/bin/npm  /usr/local/bin/npm  || warn "Could not create npm symlink."
            sudo ln -sf /opt/nodejs/bin/npx  /usr/local/bin/npx  || warn "Could not create npx symlink."
            if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
                log "Node.js: $(node --version)  npm: $(npm --version)"
            else
                warn "Node.js archive extracted but node/npm is not available on PATH."
            fi
        else
            warn "Node.js extraction failed; continuing."
        fi
    else
        warn "apps/nodejs.tar.xz is not a valid tarball; skipping Node.js."
    fi
else
    warn "apps/nodejs.tar.xz not found; skipping Node.js."
fi

# ============================================================================
# 7) BUN
# ============================================================================
step "Bun"

if [[ -f "$BUNDLE_DIR/apps/bun-linux-x64.zip" ]]; then
    BUN_TAG=$(cat "$BUNDLE_DIR/apps/bun.version" 2>/dev/null || echo "unknown")
    log "Installing Bun $BUN_TAG"
    command -v unzip >/dev/null || sudo apt-get install -y unzip 2>/dev/null || true
    if command -v unzip >/dev/null 2>&1 && unzip -tq "$BUNDLE_DIR/apps/bun-linux-x64.zip" >/dev/null 2>&1; then
        TMP_BUN=$(mktemp -d)
        if unzip -q "$BUNDLE_DIR/apps/bun-linux-x64.zip" -d "$TMP_BUN" \
            && [[ -x "$TMP_BUN"/bun-linux-x64/bun ]] \
            && sudo install -m 0755 "$TMP_BUN"/bun-linux-x64/bun /usr/local/bin/bun; then
            sudo ln -sf /usr/local/bin/bun /usr/local/bin/bunx || warn "Could not create bunx symlink."
            log "Bun: $(bun --version)"
        else
            warn "Bun extraction/install failed; continuing."
        fi
        rm -rf "$TMP_BUN"
    else
        warn "Bun zip is invalid or unzip is unavailable; skipping Bun."
    fi
else
    warn "apps/bun-linux-x64.zip not found; skipping Bun."
fi

# ============================================================================
# ============================================================================
# 8) PYTHON VENV: LLM Inference
#    torch 2.11.0+cu130 | vLLM + LLM_API_fast + RAG + llama.cpp Python utils
# ============================================================================
step "Python venv: LLM Inference"

if [[ "$INSTALL_INFERENCE" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/inference"
    LLAMA_WHEELS="$BUNDLE_DIR/wheels/llamacpp"

    if _wheelhouse_has_packages "$WHEELS_DIR"; then
        log "Creating LLM Inference venv at $INFERENCE_PREFIX/venv"
        mkdir -p "$INFERENCE_PREFIX"
        if "$PYTHON_BIN" -m venv "$INFERENCE_PREFIX/venv"; then
        # shellcheck disable=SC1091
        source "$INFERENCE_PREFIX/venv/bin/activate"

        pip install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
            || warn "Inference venv bootstrap packages failed."

        log "Installing PyTorch (inference)"
        pip install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio \
            || warn "torch install failed; inference venv may be incomplete."

        log "Installing vLLM"
        pip install --no-index --find-links="$WHEELS_DIR" vllm             || warn "vLLM install failed."

        for rf in             "$BUNDLE_DIR/requirements/llm_api.txt"             "$BUNDLE_DIR/requirements/llm_api_full.txt"; do
            [[ -f "$rf" ]] || continue
            log "  Installing from $(basename "$rf")"
            pip install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
        done

        # Do NOT install llama.cpp's Python requirements here.
        #
        # llama.cpp ships requirements files (requirements-convert_hf_to_gguf.txt,
        # etc.) that pin `torch~=2.6.0`. Installing them into THIS venv made pip
        # downgrade torch from 2.11.0+cu130 to 2.6.0+cpu (the only 2.6 wheel in
        # the llamacpp wheelhouse is CPU-only), silently breaking vLLM and all
        # CUDA inference. The convert/utility scripts that need torch~=2.6 live
        # in the dedicated llama.cpp venv created later at $LLAMA_PREFIX/venv —
        # use that venv to run convert_hf_to_gguf.py and friends.
        #
        # If you actually need the `gguf` module in this venv to read/write GGUF
        # files from RAG/inference code, install it standalone:
        #     pip install --no-index --find-links=$LLAMA_WHEELS --no-deps gguf

        log "Installing core inference/RAG packages"
        pip install --no-index --find-links="$WHEELS_DIR"             sentence-transformers faiss-cpu rank-bm25             transformers tokenizers safetensors huggingface-hub             langchain langchain-core langchain-community langchain-ollama             langgraph langgraph-checkpoint langgraph-prebuilt langsmith             ollama tavily-python             fastapi uvicorn pydantic sse-starlette httpx aiohttp             passlib python-jose PyMuPDF python-docx pandas numpy Pillow             jupyter_client ipykernel filelock tqdm rich             2>/dev/null || true

        deactivate

        log "Smoke test: torch + vllm"
        "$INFERENCE_PREFIX/venv/bin/python" <<'PYSMOKE' || warn "Inference smoke test failed — check venv and CUDA driver."
import torch
import vllm
print(f"  torch {torch.__version__}")
print(f"  vllm  {vllm.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
PYSMOKE
        "$INFERENCE_PREFIX/venv/bin/python" -m pip check \
            || warn "Inference venv has dependency conflicts."

        log "LLM Inference venv ready: $INFERENCE_PREFIX/venv"
        log "Activate: source $INFERENCE_PREFIX/venv/bin/activate"
        else
            warn "Could not create LLM Inference venv; skipping inference environment."
        fi
    else
        warn "wheels/inference/ empty or missing; skipping."
    fi
else
    log "INSTALL_INFERENCE=0; skipping."
fi

# 10) LLAMA.CPP — build from source
# ============================================================================
step "llama.cpp (build from source)"

if [[ "$INSTALL_LLAMA" == "1" ]]; then
    if [[ -f "$BUNDLE_DIR/src/llama.cpp.tar.gz" ]]; then
        log "Extracting llama.cpp -> $LLAMA_PREFIX"
        mkdir -p "$LLAMA_PREFIX"
        if tar -xzf "$BUNDLE_DIR/src/llama.cpp.tar.gz" -C "$LLAMA_PREFIX" --strip-components=1; then

        # Install CUDA keyring first if bundled, so dpkg accepts CUDA-signed packages
        shopt -s nullglob
        keyring_debs=( "$BUNDLE_DIR"/debs/cuda-keyring*.deb )
        if (( ${#keyring_debs[@]} > 0 )); then
            log "Installing cuda-keyring: ${keyring_debs[*]##*/}"
            sudo dpkg -i "${keyring_debs[@]}" || warn "cuda-keyring install failed; continuing with existing apt trust state."
        fi

        CMAKE_ARGS=(
            -S "$LLAMA_PREFIX"
            -B "$LLAMA_PREFIX/build"
            -DCMAKE_BUILD_TYPE=Release
            -DGGML_NATIVE=ON
            -DLLAMA_CURL=ON
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_EXAMPLES=ON
            -DLLAMA_BUILD_SERVER=ON
            # llama.cpp's UI target tries to download a JS/CSS bundle from
            # huggingface.co at build time (scripts/ui-download.cmake). That
            # times out on an airgapped server and causes the *entire* build
            # to fail with "Error 2" after llama-cli/llama-server have already
            # been linked. We don't need the embedded web UI for llama-server.
            -DLLAMA_BUILD_UI=OFF
        )
        [[ "$BUILD_BLAS" == "1" ]] && CMAKE_ARGS+=( -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS )
        if [[ "$BUILD_CUDA" == "1" ]]; then
            if command -v nvcc >/dev/null; then
                CMAKE_ARGS+=( -DGGML_CUDA=ON )

                # Pin CUDA arches explicitly. CMAKE_CUDA_ARCHITECTURES=native
                # (llama.cpp's default with GGML_NATIVE=ON) shells out to nvcc
                # to probe the local GPU at configure time. On a freshly
                # installed driver where nvidia-fabricmanager hasn't started
                # yet (or before reboot) the probe returns "No CUDA devices
                # found" and CMake silently builds the CUDA backend with an
                # empty arch list — kernels never launch at runtime.
                # Defaults: sm_90 (H100/H200) + sm_100 (B100/B200/B300).
                # Override with CUDA_ARCH_LIST=... if you target other GPUs.
                CMAKE_ARGS+=( -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_LIST:-90;100}" )
                log "CUDA arches: ${CUDA_ARCH_LIST:-90;100}"

                # NCCL hints: libnccl-dev (bundled by gather-all.sh) puts the
                # header in /usr/include and the .so symlink under the multiarch
                # libdir. llama.cpp's FindNCCL only checks a handful of paths
                # and misses libnccl-dev's Debian layout, so we hand it the
                # exact files. Without these, multi-GPU collectives fall back
                # to a slow per-pair P2P copy.
                _nccl_inc=""; _nccl_lib=""
                for _h in /usr/include/nccl.h /usr/local/cuda/include/nccl.h; do
                    [[ -f "$_h" ]] && { _nccl_inc="$(dirname "$_h")"; break; }
                done
                for _l in /usr/lib/x86_64-linux-gnu/libnccl.so \
                          /usr/lib/$(uname -m)-linux-gnu/libnccl.so \
                          /usr/local/cuda/lib64/libnccl.so; do
                    [[ -e "$_l" ]] && { _nccl_lib="$_l"; break; }
                done
                if [[ -n "$_nccl_inc" && -n "$_nccl_lib" ]]; then
                    CMAKE_ARGS+=(
                        -DNCCL_INCLUDE_DIR="$_nccl_inc"
                        -DNCCL_LIBRARY="$_nccl_lib"
                    )
                    log "NCCL: $_nccl_lib (hdr: $_nccl_inc)"
                else
                    warn "NCCL not found (libnccl-dev missing?); multi-GPU performance will be suboptimal."
                fi

                log "CUDA build enabled: $(nvcc --version | grep release)"
            else
                warn "BUILD_CUDA=1 but nvcc not found; building without CUDA."
            fi
        fi

        log "Configuring/building llama.cpp (cmake, -j${JOBS})"
        if cmake "${CMAKE_ARGS[@]}" && cmake --build "$LLAMA_PREFIX/build" --config Release -j"$JOBS"; then

        # Python venv for convert_hf_to_gguf.py and friends
        LLAMA_WHEELS="$BUNDLE_DIR/wheels/llamacpp"
        if _wheelhouse_has_packages "$LLAMA_WHEELS"; then
            log "Creating llama.cpp Python venv at $LLAMA_PREFIX/venv"
            if "$PYTHON_BIN" -m venv "$LLAMA_PREFIX/venv"; then
            # shellcheck disable=SC1091
            source "$LLAMA_PREFIX/venv/bin/activate"
            pip install --no-index --find-links="$LLAMA_WHEELS" --upgrade pip wheel setuptools \
                || warn "llama.cpp venv bootstrap packages failed."
            shopt -s nullglob
            for rf in "$LLAMA_PREFIX"/requirements.txt "$LLAMA_PREFIX"/requirements/*.txt; do
                [[ -f "$rf" ]] || continue
                pip install --no-index --find-links="$LLAMA_WHEELS" -r "$rf" \
                    || warn "Some llama.cpp packages failed for ${rf##*/}."
            done
            deactivate
            else
                warn "Could not create llama.cpp Python venv; continuing without it."
            fi
        fi

        log "Smoke test: llama-cli --version"
        "$LLAMA_PREFIX/build/bin/llama-cli" --version \
            && log "llama.cpp OK" || warn "llama-cli smoke test failed — check build output."
        else
            warn "llama.cpp configure/build failed; continuing."
        fi
        else
            warn "llama.cpp source extraction failed; continuing."
        fi
    else
        warn "src/llama.cpp.tar.gz not found in bundle; skipping llama.cpp build."
    fi
else
    log "INSTALL_LLAMA=0; skipping."
fi

# ============================================================================
# 8.5) NCCL-TESTS — build NVIDIA's multi-GPU bandwidth benchmark suite
#   Without this, you can't verify NVLink/NVSwitch is actually being used.
#   `all_reduce_perf -b 8 -e 8G -f 2 -g 8` on 8x B300 should report busBW
#   in the hundreds of GB/s; anything under ~100 GB/s means traffic is
#   silently falling back to PCIe (typically: FabricManager not running).
# ============================================================================
if [[ -f "$BUNDLE_DIR/src/nccl-tests.tar.gz" ]] && [[ "$BUILD_CUDA" == "1" ]] && command -v nvcc >/dev/null 2>&1; then
    step "nccl-tests (build from source)"
    NCCL_TESTS_PREFIX="${NCCL_TESTS_PREFIX:-$HOME/nccl-tests}"
    rm -rf "$NCCL_TESTS_PREFIX"
    mkdir -p "$NCCL_TESTS_PREFIX"
    if tar -xzf "$BUNDLE_DIR/src/nccl-tests.tar.gz" -C "$NCCL_TESTS_PREFIX" --strip-components=1; then
        _nccl_home=""
        for _h in /usr/include/nccl.h /usr/local/cuda/include/nccl.h; do
            [[ -f "$_h" ]] && { _nccl_home="$(dirname "$(dirname "$_h")")"; break; }
        done
        _cuda_home="${_CUDA_ROOT:-/usr/local/cuda}"
        log "Building nccl-tests (CUDA_HOME=$_cuda_home, NCCL_HOME=${_nccl_home:-/usr}, MPI=0)"
        if make -C "$NCCL_TESTS_PREFIX" -j"$JOBS" \
                CUDA_HOME="$_cuda_home" \
                NCCL_HOME="${_nccl_home:-/usr}" \
                MPI=0 >/dev/null 2>&1; then
            # Install benchmark binaries to /usr/local/bin
            for _bin in all_reduce_perf all_gather_perf reduce_scatter_perf broadcast_perf \
                        reduce_perf alltoall_perf sendrecv_perf scatter_perf gather_perf; do
                if [[ -x "$NCCL_TESTS_PREFIX/build/$_bin" ]]; then
                    sudo install -m 0755 "$NCCL_TESTS_PREFIX/build/$_bin" "/usr/local/bin/$_bin"
                fi
            done
            log "nccl-tests installed: /usr/local/bin/all_reduce_perf (and friends)"
            log "Verify multi-GPU bandwidth: all_reduce_perf -b 8 -e 8G -f 2 -g \$(nvidia-smi -L | wc -l)"
        else
            warn "nccl-tests build failed; multi-GPU bandwidth verification won't be available."
        fi
    else
        warn "nccl-tests source extraction failed."
    fi
fi

# ============================================================================
# 8.6) NVIDIA CONTAINER TOOLKIT — runtime configuration
#   The toolkit packages install nvidia-ctk, but until you run `nvidia-ctk
#   runtime configure`, neither docker nor containerd know how to expose GPUs
#   to containers. Idempotent — safe to re-run.
# ============================================================================
if command -v nvidia-ctk >/dev/null 2>&1; then
    step "NVIDIA Container Toolkit runtime config"
    if command -v docker >/dev/null 2>&1 || [[ -S /var/run/docker.sock ]]; then
        sudo nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 \
            && log "docker runtime configured for GPU passthrough" \
            || warn "nvidia-ctk runtime configure --runtime=docker failed."
        sudo systemctl restart docker 2>/dev/null || true
    fi
    if command -v containerd >/dev/null 2>&1 || [[ -S /run/containerd/containerd.sock ]]; then
        sudo nvidia-ctk runtime configure --runtime=containerd >/dev/null 2>&1 \
            && log "containerd runtime configured for GPU passthrough" \
            || warn "nvidia-ctk runtime configure --runtime=containerd failed."
        sudo systemctl restart containerd 2>/dev/null || true
    fi
fi

# ============================================================================
# 8.7) KERNEL TUNING — sysctl + transparent hugepages for high-throughput
#   inference. Conservative defaults; safe on any LLM workload host.
# ============================================================================
step "Kernel tuning (sysctl + THP)"
sudo tee /etc/sysctl.d/99-llm-multigpu.conf > /dev/null <<'SYSCTL'
# Installed by install-all.sh — tuning for multi-GPU LLM inference / training.

# Allow allocator overcommit. vLLM's KV cache + page allocator assumes the
# kernel won't OOM-kill on legitimate large mmap reservations.
vm.overcommit_memory=1

# Don't swap inference workers under memory pressure — prefer the OOM killer
# (kernel will pick the offender) over thrashing model weights to disk.
vm.swappiness=0

# Increase mmap count — torch + vLLM open many shared memory segments per
# rank, and the default 65530 is easy to hit on 8-GPU setups.
vm.max_map_count=1048576

# Reduce kernel network receive backpressure for large model artifacts /
# checkpoints landing from internal storage.
net.core.rmem_max=268435456
net.core.wmem_max=268435456
SYSCTL
sudo sysctl --system >/dev/null 2>&1 || warn "sysctl --system reload failed."
log "sysctl applied: /etc/sysctl.d/99-llm-multigpu.conf"

# Transparent hugepages = madvise. Default 'always' triggers spurious khugepaged
# scans on the multi-hundred-GB working sets of LLM inference; 'never' loses
# the perf win. 'madvise' lets vLLM/torch opt in explicitly.
sudo tee /etc/systemd/system/disable-thp-defrag.service > /dev/null <<'UNIT'
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
sudo systemctl daemon-reload 2>/dev/null || true
sudo systemctl enable --now disable-thp-defrag.service >/dev/null 2>&1 \
    && log "transparent hugepages: madvise (via disable-thp-defrag.service)" \
    || warn "Could not enable disable-thp-defrag.service."

# ============================================================================
# 8.8) NCCL ENVIRONMENT DEFAULTS  +  SYSTEM LIMITS
#   NCCL_NVLS_ENABLE=1 is the single highest-impact knob on B300/NVSwitch.
#   NVLink SHARP does in-fabric reductions for allreduce — ~30% bandwidth win
#   vs. plain Ring/Tree. NCCL ships it OFF by default; users discover this the
#   hard way.
# ============================================================================
step "NCCL env defaults + system limits"

sudo tee /etc/profile.d/nccl-multigpu.sh > /dev/null <<'NCCLENV'
# Installed by install-all.sh — NCCL defaults for 8x B300 SXM6 NVSwitch.
# Override per-job by exporting the variable before launching the workload.

# Enable NVLink SHARP (NVLS) in-fabric reductions. The B300/NVSwitch killer
# feature for allreduce. NCCL 2.28+; harmless on hardware that doesn't
# support it (gets ignored).
export NCCL_NVLS_ENABLE=1

# Restrict P2P to NVLink-class transports. On a fully NVLinked NVSwitch box
# this prevents NCCL from ever silently downgrading to PCIe P2P or shared
# memory. Every GPU pair on this hardware reaches every other via NVLink.
export NCCL_P2P_LEVEL=NVL

# WARN level: quiet on success, loud on real problems (topology fallback,
# truncated buffers, comm init failure). Override with NCCL_DEBUG=INFO when
# you actually want the full init dump.
: "${NCCL_DEBUG:=WARN}"
export NCCL_DEBUG

# Parallel kernel launches across streams — already the default in NCCL
# 2.20+, but explicit is better for reproducibility.
export NCCL_LAUNCH_MODE=PARALLEL
NCCLENV
sudo chmod 0644 /etc/profile.d/nccl-multigpu.sh
log "NCCL env defaults: /etc/profile.d/nccl-multigpu.sh (NVLS enabled)"

# Limits — without these, multi-process inference / training easily hits
# EMFILE (default nofile=1024) or 'cannot allocate memory' (default memlock
# is 64 KB) and crashes mid-run with messages that look like NCCL bugs.
sudo tee /etc/security/limits.d/99-llm-multigpu.conf > /dev/null <<'LIMITS'
# Installed by install-all.sh — required ceilings for multi-GPU LLM workloads.
*  soft  nofile   1048576
*  hard  nofile   1048576
*  soft  nproc    524288
*  hard  nproc    524288
*  soft  memlock  unlimited
*  hard  memlock  unlimited
*  soft  stack    65536
*  hard  stack    65536
LIMITS
sudo chmod 0644 /etc/security/limits.d/99-llm-multigpu.conf

# Mirror limits into systemd defaults so services started without a PAM
# session (system units) also get them. /etc/security/limits.d/ only applies
# to PAM-mediated logins; systemd services use systemd.exec limits.
sudo mkdir -p /etc/systemd/system.conf.d
sudo tee /etc/systemd/system.conf.d/99-llm-multigpu.conf > /dev/null <<'SYSDLIM'
# Installed by install-all.sh — match /etc/security/limits.d/99-llm-multigpu.conf
# for systemd-started services (which don't go through PAM).
[Manager]
DefaultLimitNOFILE=1048576
DefaultLimitMEMLOCK=infinity
DefaultLimitNPROC=524288
DefaultLimitSTACK=65536
SYSDLIM
sudo systemctl daemon-reexec 2>/dev/null || true
log "System limits: /etc/security/limits.d/99-llm-multigpu.conf (+ systemd mirror)"

# ============================================================================
# 8.9) GPU + LLM-INFERENCE OPERATIONAL TOOLING
#   Helper scripts + a systemd template for serving llama.cpp models.
# ============================================================================
step "GPU + LLM-inference operational tooling"

# ── /usr/local/bin/gpu-health-check ────────────────────────────────────────
# One command that verifies the fabric is actually working: FabricManager
# active, NVLink lanes up, NCCL bandwidth at line rate, DCGM healthy.
# Run before any large training job or after any hardware change.
sudo tee /usr/local/bin/gpu-health-check > /dev/null <<'HEALTH'
#!/usr/bin/env bash
# gpu-health-check — verify multi-GPU fabric is healthy.
# Installed by install-all.sh.
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
    bad "nvidia-smi cannot list GPUs"
    exit 1
fi

say "2. nvidia-fabricmanager"
if (( n > 1 )); then
    state=$(systemctl is-active nvidia-fabricmanager 2>/dev/null || echo unknown)
    if [[ "$state" == "active" ]]; then ok "active"
    else bad "NOT active on $n-GPU NVSwitch box (NCCL will fall back to PCIe)"
         echo "       sudo journalctl -u nvidia-fabricmanager -n 40 --no-pager"
    fi
else
    warn "skipped — only $n GPU"
fi

say "3. nvidia-persistenced"
state=$(systemctl is-active nvidia-persistenced 2>/dev/null || echo unknown)
[[ "$state" == "active" ]] && ok "active" || bad "NOT active (cold CUDA latency penalty)"

say "4. NVLink lane health"
status=$(nvidia-smi nvlink --status 2>/dev/null || true)
if [[ -n "$status" ]]; then
    down=$(printf '%s\n' "$status" | grep -ciE 'inactive|<inactive>|disabled' || true)
    up=$(printf '%s\n' "$status" | grep -c 'GB/s' || true)
    if (( down == 0 )) && (( up > 0 )); then
        ok "$up active lane(s), 0 inactive"
    elif (( down > 0 )); then
        bad "$down inactive NVLink lane(s) — fabric degraded"
    else
        warn "no NVLink data reported"
    fi
fi

say "5. DCGM quick diagnostic (dcgmi diag -r 1)"
if command -v dcgmi >/dev/null 2>&1; then
    if dcgmi diag -r 1 2>&1 | tail -n 40 | grep -qE '"Pass"|PASS|Successful'; then
        ok "dcgmi diag -r 1 passed"
    else
        warn "dcgmi diag -r 1 reported issues — re-run manually for detail: dcgmi diag -r 2"
    fi
else
    warn "dcgmi not installed — skipped"
fi

say "6. NCCL all_reduce bandwidth (8 MiB → 512 MiB)"
if command -v all_reduce_perf >/dev/null && (( n > 1 )); then
    out=$(all_reduce_perf -b 8M -e 512M -f 2 -g "$n" 2>&1) || { bad "all_reduce_perf failed"; printf '%s\n' "$out" | tail -n 20; }
    if [[ -n "${out:-}" ]]; then
        # Get max busBW from the table (last column on data rows)
        bw=$(printf '%s\n' "$out" | awk '/^ *[0-9]+ +[0-9]+/ {print $(NF)}' | sort -gr | head -1)
        if [[ -n "$bw" ]]; then
            # Expect >100 GB/s on any NVLink-connected box; >400 GB/s on NVSwitch B300
            awk -v b="$bw" -v g="$n" 'BEGIN {
                if (b+0 >= 400) print "  PASS  busBW " b " GB/s (good for " g "x NVLink/NVSwitch)";
                else if (b+0 >= 100) print "  WARN  busBW " b " GB/s — below NVSwitch expectation (~400+ GB/s)";
                else { print "  FAIL  busBW " b " GB/s — fabric likely on PCIe"; exit 1; }
            }' || rc=1
        fi
    fi
elif (( n < 2 )); then
    warn "skipped — only $n GPU"
else
    warn "all_reduce_perf not installed — skipped"
fi

echo
[[ $rc -eq 0 ]] && say "ALL CHECKS PASSED" || say "ONE OR MORE CHECKS FAILED (see above)"
exit $rc
HEALTH
sudo chmod 0755 /usr/local/bin/gpu-health-check
log "Installed: gpu-health-check (run with: gpu-health-check)"

# ── /usr/local/bin/llama-server-multigpu ──────────────────────────────────
# NUMA + tensor-split aware launcher for llama-server. Computes a balanced
# --tensor-split across visible GPUs, sets --split-mode row (uses NCCL),
# and binds CPU threads to the NUMA node of GPU 0. Extra args pass through.
sudo tee /usr/local/bin/llama-server-multigpu > /dev/null <<'LLAMAWRAP'
#!/usr/bin/env bash
# llama-server-multigpu — NUMA + tensor-split wrapper for llama.cpp server.
# Installed by install-all.sh.
#
# Usage:
#   llama-server-multigpu --model /path/to/model.gguf --port 8080 [llama-server-args...]
# Env overrides:
#   LLAMA_BIN        path to llama-server binary (default: ~/llama.cpp/build/bin/llama-server)
#   LLAMA_N_GPUS     how many GPUs to use (default: all visible)
#   LLAMA_NO_NUMA=1  disable NUMA pinning
#   LLAMA_SPLIT_MODE row|layer|none (default: row when N>1, else none)
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-${HOME}/llama.cpp/build/bin/llama-server}"
[[ -x "$LLAMA_BIN" ]] || { echo "llama-server not found at $LLAMA_BIN; set LLAMA_BIN=..." >&2; exit 1; }

N=${LLAMA_N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
[[ "$N" -ge 1 ]] || { echo "no GPUs visible (LLAMA_N_GPUS=$N)" >&2; exit 1; }

extra_args=()

# --tensor-split: only add if caller didn't already provide one.
if (( N > 1 )) && ! printf '%s\n' "$@" | grep -q -- '--tensor-split'; then
    split=$(yes 1 | head -n "$N" | paste -sd ',')
    extra_args+=( --tensor-split "$split" )
    mode="${LLAMA_SPLIT_MODE:-row}"
    if [[ "$mode" != "none" ]] && ! printf '%s\n' "$@" | grep -q -- '--split-mode'; then
        extra_args+=( --split-mode "$mode" )
    fi
fi

# NUMA pinning to GPU 0's node.
prefix=()
if [[ "${LLAMA_NO_NUMA:-0}" != "1" ]] && command -v numactl >/dev/null && command -v nvidia-smi >/dev/null; then
    pci=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    # nvidia-smi gives 00000000:01:00.0; sysfs uses 0000:01:00.0
    pci_short="${pci#????}"
    node_file="/sys/bus/pci/devices/${pci_short,,}/numa_node"
    if [[ -f "$node_file" ]]; then
        node=$(cat "$node_file" 2>/dev/null || echo -1)
        if [[ -n "$node" && "$node" -ge 0 ]]; then
            prefix=(numactl --cpunodebind="$node" --membind="$node")
        fi
    fi
fi

echo "[llama-server-multigpu] $N GPU(s); extra args: ${extra_args[*]}; numa: ${prefix[*]:-none}" >&2
exec "${prefix[@]}" "$LLAMA_BIN" "${extra_args[@]}" "$@"
LLAMAWRAP
sudo chmod 0755 /usr/local/bin/llama-server-multigpu
log "Installed: llama-server-multigpu (run with: llama-server-multigpu --model X.gguf)"

# ── /usr/local/bin/llama-model-preload ─────────────────────────────────────
# Force a GGUF model into page cache before serving, so the first inference
# request doesn't pay the disk-read tax. Reads from a path or directory.
sudo tee /usr/local/bin/llama-model-preload > /dev/null <<'PRELOAD'
#!/usr/bin/env bash
# llama-model-preload — pre-mmap GGUF model into page cache.
# Installed by install-all.sh.
# Usage: llama-model-preload /path/to/model.gguf [more.gguf ...]
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <model.gguf> [more.gguf ...]" >&2; exit 1; }
for f in "$@"; do
    if [[ ! -r "$f" ]]; then
        echo "[preload] skip: $f (not readable)" >&2
        continue
    fi
    sz=$(stat -c '%s' "$f" 2>/dev/null || echo 0)
    sz_gb=$(awk -v s="$sz" 'BEGIN {printf "%.1f", s/(1024*1024*1024)}')
    printf '[preload] reading %s (%s GiB) → page cache\n' "$f" "$sz_gb" >&2
    # cat is the simplest POSIX way; vmtouch would be nicer but isn't in apt.
    cat "$f" > /dev/null
done
echo "[preload] done; subsequent mmap reads will hit page cache" >&2
PRELOAD
sudo chmod 0755 /usr/local/bin/llama-model-preload
log "Installed: llama-model-preload (preload models into page cache)"

# ── /etc/systemd/system/llama-server@.service ──────────────────────────────
# Instanced systemd unit for serving llama.cpp models. The instance name
# names a per-model env file under /etc/llama-server/.
#
# Usage:
#   1. echo 'MODEL=/opt/models/llama3-70b.gguf' | sudo tee /etc/llama-server/llama3.env
#   2. sudo systemctl enable --now llama-server@llama3
sudo mkdir -p /etc/llama-server
if [[ ! -f /etc/llama-server/example.env ]]; then
    sudo tee /etc/llama-server/example.env > /dev/null <<'EXAMPLE'
# /etc/llama-server/<instance>.env
# Copy to <name>.env and customize, then: systemctl enable --now llama-server@<name>
MODEL=/opt/models/MODEL.gguf
HOST=0.0.0.0
PORT=8080
# Layers offloaded to GPU. 999 = all (typical).
NGL=999
# Context window. Defaults to model's training context.
CTX=8192
# Extra args appended verbatim to llama-server.
EXTRA=
EXAMPLE
fi
sudo tee /etc/systemd/system/llama-server@.service > /dev/null <<'UNIT'
# Templated llama.cpp server unit.
# Configure: /etc/llama-server/<instance>.env
# Start:     systemctl enable --now llama-server@<instance>
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
# Capture all output to journal so 'journalctl -u llama-server@<x>' works.
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload 2>/dev/null || true
log "Installed: llama-server@.service template (config dir: /etc/llama-server/)"
log "  Try: cp /etc/llama-server/example.env /etc/llama-server/mymodel.env; \$EDITOR it; systemctl enable --now llama-server@mymodel"

# ============================================================================
# ============================================================================
# 9) PYTHON VENV: General Training
#    torch 2.11.0+cu130 | PyG + MeshGraphNets + SimulGenVAE + PEMTRON
# ============================================================================
step "Python venv: General Training"

if [[ "$INSTALL_TRAINING" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/training"

    if _wheelhouse_has_packages "$WHEELS_DIR"; then
        log "Creating General Training venv at $TRAINING_PREFIX/venv"
        mkdir -p "$TRAINING_PREFIX"
        if "$PYTHON_BIN" -m venv "$TRAINING_PREFIX/venv"; then
        # shellcheck disable=SC1091
        source "$TRAINING_PREFIX/venv/bin/activate"

        pip install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
            || warn "Training venv bootstrap packages failed."

        log "Installing PyTorch (training)"
        pip install --no-index --find-links="$WHEELS_DIR" torch torchvision torchaudio \
            || warn "torch install failed; training venv may be incomplete."

        log "Installing torch-geometric"
        pip install --no-index --find-links="$WHEELS_DIR" torch-geometric             || warn "torch-geometric failed."

        log "Installing PyG extensions"
        for pkg in pyg_lib torch-scatter torch-sparse torch-cluster; do
            pip install --no-index --find-links="$WHEELS_DIR" "$pkg" 2>/dev/null                 && log "  $pkg: OK" || warn "  $pkg not found."
        done

        for rf in             "$BUNDLE_DIR/requirements/meshgraphnets.txt"             "$BUNDLE_DIR/requirements/simulgen.txt"             "$BUNDLE_DIR/requirements/pemtron.txt"             "$BUNDLE_DIR/requirements/pemtron_transfer.txt"; do
            [[ -f "$rf" ]] || continue
            log "  Installing from $(basename "$rf")"
            pip install --no-index --find-links="$WHEELS_DIR" -r "$rf" 2>/dev/null || true
        done

        log "Installing core training/scientific stack"
        pip install --no-index --find-links="$WHEELS_DIR"             numpy scipy h5py pandas tqdm matplotlib seaborn Pillow pyvista             scikit-learn scikit-image statsmodels networkx sympy             torchinfo tensorboard opencv-python imageio             librosa audiomentations soxr natsort reportlab             2>/dev/null || true

        deactivate

        log "Smoke test: torch + PyG"
        "$TRAINING_PREFIX/venv/bin/python" <<'PYSMOKE' || warn "Training smoke test failed — check venv and CUDA driver."
import torch
from torch_geometric.data import Data
print(f"  torch {torch.__version__}")
print(f"  torch-geometric OK")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device 0: {torch.cuda.get_device_name(0)}")
PYSMOKE
        "$TRAINING_PREFIX/venv/bin/python" -m pip check \
            || warn "Training venv has dependency conflicts."

        log "General Training venv ready: $TRAINING_PREFIX/venv"
        log "Activate: source $TRAINING_PREFIX/venv/bin/activate"
        else
            warn "Could not create General Training venv; skipping training environment."
        fi
    else
        warn "wheels/training/ empty or missing; skipping."
    fi
else
    log "INSTALL_TRAINING=0; skipping."
fi

# 12) PYTHON VENV: Jupyter + data science
# ============================================================================
step "Python venv: Jupyter + data science"

if [[ "$INSTALL_JUPYTER" == "1" ]]; then
    WHEELS_DIR="$BUNDLE_DIR/wheels/jupyter"

    if _wheelhouse_has_packages "$WHEELS_DIR"; then
        log "Creating Jupyter venv at $JUPYTER_PREFIX/venv"
        mkdir -p "$JUPYTER_PREFIX"
        if "$PYTHON_BIN" -m venv "$JUPYTER_PREFIX/venv"; then
        # shellcheck disable=SC1091
        source "$JUPYTER_PREFIX/venv/bin/activate"

        pip install --no-index --find-links="$WHEELS_DIR" --upgrade pip wheel setuptools \
            || warn "Jupyter venv bootstrap packages failed."

        pip install --no-index --find-links="$WHEELS_DIR" \
            jupyterlab notebook ipykernel ipywidgets jupyter-server \
            pandas polars numpy scipy matplotlib seaborn plotly \
            scikit-learn statsmodels tqdm rich requests aiohttp \
            black ruff mypy pytest ipdb \
            || warn "Some Jupyter packages failed; check output above."

        # Register the kernel so it appears in JupyterLab
        "$JUPYTER_PREFIX/venv/bin/python" -m ipykernel install \
            --user --name "airgap-py${PYTHON_VER}" --display-name "Python ${PYTHON_VER} (airgap)" \
            2>/dev/null || true

        deactivate

        # Drop a convenience launcher
        cat > "$HOME/start-jupyter.sh" <<JEOF
#!/usr/bin/env bash
source "$JUPYTER_PREFIX/venv/bin/activate"
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser "\$@"
JEOF
        chmod +x "$HOME/start-jupyter.sh"
        log "Jupyter venv ready at $JUPYTER_PREFIX/venv"
        log "Start with: bash ~/start-jupyter.sh  (or run in tmux for persistence)"
        else
            warn "Could not create Jupyter venv; skipping Jupyter environment."
        fi
    else
        warn "wheels/jupyter/ empty or missing; skipping Jupyter venv."
    fi
else
    log "INSTALL_JUPYTER=0; skipping."
fi

# ============================================================================
# 13) DESKTOP ENVIRONMENT — XFCE4 + xrdp
# ============================================================================
step "Desktop environment (XFCE4 + xrdp)"

if [[ "$INSTALL_DESKTOP" == "1" ]]; then

    # ── xrdp: configure XFCE4 as the default session ─────────────────────────
    if command -v xrdp >/dev/null 2>&1; then
        log "Configuring xrdp to launch XFCE4 session"
        if sudo bash -c 'cat > /etc/xrdp/startwm.sh <<'"'"'XRDPEOF'"'"'
#!/bin/sh
# Set locale
if [ -r /etc/default/locale ]; then
    . /etc/default/locale
    export LANG LANGUAGE
fi
exec startxfce4
XRDPEOF'
        then
            sudo chmod +x /etc/xrdp/startwm.sh || warn "Could not chmod /etc/xrdp/startwm.sh."
        else
            warn "Could not write /etc/xrdp/startwm.sh; continuing."
        fi

        # Allow xrdp to read the TLS certificate (needed for NLA / encryption)
        sudo adduser xrdp ssl-cert 2>/dev/null || true

        # Enable and start xrdp
        sudo systemctl enable xrdp 2>/dev/null || true
        sudo systemctl restart xrdp 2>/dev/null \
            || sudo service xrdp restart 2>/dev/null \
            || warn "Could not restart xrdp — run 'sudo systemctl start xrdp' after reboot."
        log "xrdp listening on port 3389"

        # Open RDP port in ufw if the firewall is active
        if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
            sudo ufw allow 3389/tcp \
                && log "UFW: port 3389/tcp opened for RDP" \
                || warn "Could not open UFW port 3389/tcp."
        fi
    else
        warn "xrdp not found — was INSTALL_DESKTOP=1 set during gather-all.sh?"
    fi

    # ── Default XFCE4 session for current user and new users ─────────────────
    echo "xfce4-session" | sudo tee /etc/skel/.xsession > /dev/null \
        || warn "Could not write /etc/skel/.xsession."
    echo "xfce4-session" > "$HOME/.xsession" \
        || warn "Could not write $HOME/.xsession."
    log "XFCE4 session set for: $USER (and skeleton for new users)"

    # ── lightdm: enable only when a physical display is attached ─────────────
    # On a headless GPU server xrdp provides the display; lightdm is not needed
    # and would fail to start. Uncomment the line below if a local display is used.
    # sudo systemctl enable lightdm 2>/dev/null || true

    # ── polkit rule so normal users can reboot/shutdown from XFCE ────────────
    if [[ -d /usr/share/polkit-1/rules.d ]]; then
        sudo tee /usr/share/polkit-1/rules.d/49-xfce-shutdown.rules > /dev/null <<'POLKIT' || warn "Could not write XFCE polkit shutdown rule."
polkit.addRule(function(action, subject) {
    if ((action.id == "org.freedesktop.login1.power-off" ||
         action.id == "org.freedesktop.login1.reboot") &&
        subject.isInGroup("sudo")) {
        return polkit.Result.YES;
    }
});
POLKIT
        log "polkit rule installed (sudo group can power-off/reboot from XFCE)"
    fi

    log "Desktop setup complete."
    log "Connect via RDP to port 3389 with your Linux username/password."
else
    log "INSTALL_DESKTOP=0; skipping desktop configuration."
fi

# ============================================================================
# 14) K3s — Common setup (runs for both server and agent)
# ============================================================================
step "K3s: common setup"

if [[ "$INSTALL_K3S" == "1" ]]; then
    K3S_DIR="$BUNDLE_DIR/k3s"
    [[ -d "$K3S_DIR" ]] || die "k3s/ not found in bundle. Re-run gather-all.sh with INCLUDE_K3S=1."

    # Load pinned versions from bundle metadata
    [[ -f "$K3S_DIR/meta/versions.env" ]] && source "$K3S_DIR/meta/versions.env"

    # Check GPUs (informational — GPU Operator handles the driver plugin)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        log "GPU check: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPU(s) visible"
    else
        warn "nvidia-smi not working. Install the NVIDIA driver before relying on GPU workloads."
    fi

    # Install K3s, helm, kubectl binaries
    log "Installing k3s, helm, kubectl -> /usr/local/bin/"
    sudo install -m 0755 "$K3S_DIR/bin/k3s"     /usr/local/bin/k3s
    sudo install -m 0755 "$K3S_DIR/bin/helm"    /usr/local/bin/helm
    sudo install -m 0755 "$K3S_DIR/bin/kubectl" /usr/local/bin/kubectl

    # Stage K3s airgap image archives (K3s loads these on first start)
    log "Staging K3s airgap images to /var/lib/rancher/k3s/agent/images/"
    sudo mkdir -p /var/lib/rancher/k3s/agent/images
    shopt -s nullglob
    for img_file in "$K3S_DIR/airgap-images/"*; do
        sudo cp "$img_file" /var/lib/rancher/k3s/agent/images/
        log "  Staged: $(basename "$img_file")"
    done

    # Write registries.yaml (mirrors all registries to the airgap registry)
    log "Writing /etc/rancher/k3s/registries.yaml"
    sudo mkdir -p /etc/rancher/k3s
    REGISTRY_HOST="${K3S_SERVER_IP:-127.0.0.1}"
    [[ "$K3S_ROLE" == "server" ]] && REGISTRY_HOST="127.0.0.1"
    sudo sed "s/REGISTRY_HOST/${REGISTRY_HOST}/g" \
        "$K3S_DIR/manifests/registries.yaml.tmpl" \
        | sudo tee /etc/rancher/k3s/registries.yaml > /dev/null
    log "registries.yaml -> mirrors all registries to ${REGISTRY_HOST}:${K3S_REGISTRY_PORT}"

    log "K3s common setup complete."
else
    log "INSTALL_K3S=0; skipping K3s."
fi

# ============================================================================
# 15) K3s — Server install
# ============================================================================
step "K3s: server install"

if [[ "$INSTALL_K3S" == "1" && "$K3S_ROLE" == "server" ]]; then
    K3S_DIR="$BUNDLE_DIR/k3s"
    [[ -f "$K3S_DIR/meta/versions.env" ]] && source "$K3S_DIR/meta/versions.env"

    SERVER_IP=$(hostname -I | awk '{print $1}')
    log "Server IP: $SERVER_IP"

    # Generate or load join token
    if [[ -n "$K3S_TOKEN_FILE" && -f "$K3S_TOKEN_FILE" ]]; then
        K3S_INIT_TOKEN=$(cat "$K3S_TOKEN_FILE")
    else
        K3S_INIT_TOKEN=$(openssl rand -hex 32)
    fi
    sudo mkdir -p /var/lib/rancher/k3s/server
    printf '%s' "$K3S_INIT_TOKEN" | sudo tee /var/lib/rancher/k3s/server/node-token >/dev/null
    sudo chmod 0600 /var/lib/rancher/k3s/server/node-token
    log "Join token written to /var/lib/rancher/k3s/server/node-token"

    # Install and start K3s server
    log "Running K3s server install (takes ~1-2 min)"
    INSTALL_K3S_SKIP_DOWNLOAD=true \
    K3S_TOKEN="$K3S_INIT_TOKEN" \
    INSTALL_K3S_EXEC="server --tls-san=${SERVER_IP} --write-kubeconfig-mode=644" \
        bash "$K3S_DIR/bin/k3s-install.sh"

    # Wait for server node to be Ready
    export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    log "Waiting for K3s node to be Ready..."
    for i in $(seq 1 36); do
        kubectl get nodes 2>/dev/null | grep -q " Ready " && break
        sleep 5
    done
    kubectl get nodes 2>/dev/null | grep -q " Ready " \
        || warn "Node not Ready after 3 min — check: kubectl get nodes; journalctl -u k3s"

    # Copy kubeconfig for the current user with server IP (not 127.0.0.1)
    mkdir -p "$HOME/.kube"
    sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
    sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
    sed -i "s|127.0.0.1|${SERVER_IP}|g" "$HOME/.kube/config"
    log "kubeconfig: $HOME/.kube/config"

    # Import registry:2 image into k3s containerd so it can run as a pod
    log "Importing registry image into k3s containerd"
    shopt -s nullglob
    imported=0
    for reg_tar in \
        "$K3S_DIR/images/docker.io_library_registry_2.8.3.tar" \
        "$K3S_DIR/images/"*registry*.tar; do
        [[ -f "$reg_tar" ]] || continue
        sudo /usr/local/bin/k3s ctr images import "$reg_tar" \
            && imported=1 && break
    done
    (( imported )) || warn "registry:2 image tar not found in k3s/images/ — pod may fail to start."

    # Deploy registry as a K8s pod on the control-plane node
    log "Deploying airgap-registry pod"
    kubectl apply -f "$K3S_DIR/manifests/airgap-registry.yaml" \
        || die "Failed to apply airgap-registry.yaml"
    log "Waiting for registry pod to be Ready..."
    kubectl wait --for=condition=Ready pod \
        -l app=airgap-registry \
        -n registry \
        --timeout=120s \
        || warn "Registry pod not Ready — check: kubectl get pods -n registry"

    # Push all pre-pulled images into the airgap registry
    if command -v skopeo >/dev/null 2>&1 && [[ -f "$K3S_DIR/meta/images-manifest.txt" ]]; then
        log "Pushing pre-pulled images to localhost:${K3S_REGISTRY_PORT}..."
        PUSH_FAILED=0
        while IFS= read -r img; do
            [[ -z "$img" ]] && continue
            safe="${img//\//_}"; safe="${safe//:/_}"; safe="${safe//@/_}"
            tar_file="$K3S_DIR/images/${safe}.tar"
            [[ -f "$tar_file" ]] || continue
            # Strip registry hostname — containerd mirror maps registry.host/path → local/path
            dest_path="${img#*/}"
            log "  Push: $img -> localhost:${K3S_REGISTRY_PORT}/${dest_path}"
            skopeo copy \
                --dest-tls-verify=false \
                "docker-archive:${tar_file}" \
                "docker://localhost:${K3S_REGISTRY_PORT}/${dest_path}" 2>/dev/null \
                || { warn "  Failed: $img"; PUSH_FAILED=$(( PUSH_FAILED + 1 )); }
        done < "$K3S_DIR/meta/images-manifest.txt"
        log "Image push complete ($PUSH_FAILED failed)."
    else
        warn "skopeo not found or images manifest missing — push images to the registry manually."
    fi

    # Install Helm stacks
    log "Installing Helm stacks (this may take 5-10 min per chart)"
    HELM=/usr/local/bin/helm
    CHARTS="$K3S_DIR/charts"
    VALUES="$K3S_DIR/manifests/values"

    _helm_install() {
        local release="$1" chart_glob="$2" ns="$3"; shift 3
        kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true
        # shellcheck disable=SC2206
        local chart_files=( $chart_glob )
        [[ -f "${chart_files[0]}" ]] || { warn "Chart not found: $chart_glob"; return 1; }
        "$HELM" upgrade --install "$release" "${chart_files[0]}" \
            --namespace "$ns" \
            --wait --timeout 10m \
            "$@" \
            || warn "helm install $release failed — check: helm status $release -n $ns"
    }

    # GPU Operator: driver.enabled=false because we install the driver via apt
    GPU_OP_ARGS=(--set driver.enabled="${GPU_OPERATOR_DRIVER_ENABLED}" --set migManager.enabled=false)
    [[ -f "$VALUES/gpu-operator.yaml" ]] && GPU_OP_ARGS+=(--values "$VALUES/gpu-operator.yaml")
    _helm_install gpu-operator "$CHARTS/gpu-operator-"*.tgz gpu-operator "${GPU_OP_ARGS[@]}"

    # kube-prometheus-stack (Prometheus + Grafana + AlertManager + node-exporter + DCGM)
    PROM_ARGS=()
    [[ -f "$VALUES/kube-prometheus-stack.yaml" ]] && PROM_ARGS+=(--values "$VALUES/kube-prometheus-stack.yaml")
    _helm_install kube-prometheus-stack "$CHARTS/kube-prometheus-stack-"*.tgz monitoring "${PROM_ARGS[@]}"

    # Loki + Promtail (centralized logs)
    LOKI_ARGS=()
    [[ -f "$VALUES/loki-stack.yaml" ]] && LOKI_ARGS+=(--values "$VALUES/loki-stack.yaml")
    _helm_install loki-stack "$CHARTS/loki-stack-"*.tgz monitoring "${LOKI_ARGS[@]}"

    # KubeRay operator (for cross-node 700GB+ model inference)
    if [[ "$INSTALL_KUBERAY" == "1" ]]; then
        KUBERAY_ARGS=()
        [[ -f "$VALUES/kuberay-operator.yaml" ]] && KUBERAY_ARGS+=(--values "$VALUES/kuberay-operator.yaml")
        _helm_install kuberay-operator "$CHARTS/kuberay-operator-"*.tgz kuberay-system "${KUBERAY_ARGS[@]}"
    fi

    # Apply healer manifests if present
    if [[ -d "$K3S_DIR/manifests/healer" ]]; then
        kubectl apply -f "$K3S_DIR/manifests/healer/" \
            || warn "Healer manifests failed to apply."
        log "Healer pod manifests applied."
    fi

    printf '\n'
    printf '\033[1;32m[install]\033[0m K3s SERVER ready.\n'
    printf '  Server IP  : %s\n' "$SERVER_IP"
    printf '  Join token : /var/lib/rancher/k3s/server/node-token\n'
    printf '  kubeconfig : %s/.kube/config\n' "$HOME"
    printf '  Grafana    : http://%s:30030  (if kube-prometheus-stack NodePort configured)\n' "$SERVER_IP"
    printf '\n'
    printf 'Distribute join token to agents:\n'
    printf '  scp /var/lib/rancher/k3s/server/node-token user@AGENT:/tmp/k3s-join-token\n'
    printf '\n'
    printf 'On each agent:\n'
    printf '  sudo INSTALL_K3S=1 K3S_ROLE=agent K3S_SERVER_IP=%s \\\n' "$SERVER_IP"
    printf '       K3S_TOKEN_FILE=/tmp/k3s-join-token bash install-all.sh\n'
    printf '\n'

elif [[ "$INSTALL_K3S" == "1" && "$K3S_ROLE" != "agent" ]]; then
    log "INSTALL_K3S=1 but K3S_ROLE=$K3S_ROLE — set K3S_ROLE=server or K3S_ROLE=agent."
fi

# ============================================================================
# 16) K3s — Agent install
# ============================================================================
step "K3s: agent install"

if [[ "$INSTALL_K3S" == "1" && "$K3S_ROLE" == "agent" ]]; then
    [[ -n "$K3S_SERVER_IP" ]] \
        || die "K3S_ROLE=agent requires K3S_SERVER_IP. Set it before running."
    [[ -n "$K3S_TOKEN_FILE" && -f "$K3S_TOKEN_FILE" ]] \
        || die "K3S_ROLE=agent requires K3S_TOKEN_FILE pointing to a readable file. scp the token from server 1 first."

    K3S_JOIN_TOKEN=$(cat "$K3S_TOKEN_FILE")
    K3S_DIR="$BUNDLE_DIR/k3s"

    log "Joining K3s cluster at https://${K3S_SERVER_IP}:6443"
    INSTALL_K3S_SKIP_DOWNLOAD=true \
    K3S_URL="https://${K3S_SERVER_IP}:6443" \
    K3S_TOKEN="$K3S_JOIN_TOKEN" \
    INSTALL_K3S_EXEC="agent" \
        bash "$K3S_DIR/bin/k3s-install.sh"

    log "Waiting for k3s-agent service to be active..."
    for i in $(seq 1 24); do
        systemctl is-active k3s-agent >/dev/null 2>&1 && break
        sleep 5
    done
    systemctl is-active k3s-agent >/dev/null 2>&1 \
        && log "k3s-agent is running." \
        || warn "k3s-agent not active — check: journalctl -u k3s-agent -f"

    printf '\n'
    printf '\033[1;32m[install]\033[0m K3s AGENT joined cluster.\n'
    printf '  Server : https://%s:6443\n' "$K3S_SERVER_IP"
    printf '\n'
    printf 'Verify from server 1:\n'
    printf '  kubectl get nodes -o wide\n'
    printf '\n'
fi

# ============================================================================
# 17) Final ownership pass
#   When the installer is launched via `sudo bash install-all.sh`, everything
#   created under SCRATCH_ROOT (venvs, llama.cpp build, wheel installs) is
#   root-owned. Recursively chown back to TARGET_USER so the human can run
#   `source .../venv/bin/activate` and `pip install` afterwards without sudo.
# ============================================================================
step "Fixing $SCRATCH_ROOT ownership"

if [[ -d "$SCRATCH_ROOT" ]]; then
    if sudo chown -R "$TARGET_USER:$TARGET_GROUP" "$SCRATCH_ROOT"; then
        log "$SCRATCH_ROOT chown -R -> $TARGET_USER:$TARGET_GROUP"
    else
        warn "Could not chown $SCRATCH_ROOT to $TARGET_USER:$TARGET_GROUP."
    fi
    # Make readable/executable for everyone so other users can also activate
    # venvs and run llama-server. Write stays restricted to the owner.
    sudo chmod -R go+rX "$SCRATCH_ROOT" 2>/dev/null \
        || warn "Could not relax permissions on $SCRATCH_ROOT for other users."
else
    log "$SCRATCH_ROOT does not exist; skipping chown."
fi

# Convenience symlinks so any user has llama-cli/llama-server on PATH.
if [[ -x "$LLAMA_PREFIX/build/bin/llama-cli" ]]; then
    sudo ln -sf "$LLAMA_PREFIX/build/bin/llama-cli"    /usr/local/bin/llama-cli    2>/dev/null || true
    sudo ln -sf "$LLAMA_PREFIX/build/bin/llama-server" /usr/local/bin/llama-server 2>/dev/null || true
    log "Symlinked llama-cli/llama-server into /usr/local/bin/"
fi

# ============================================================================
# 18) Post-install verification (inlined from test-all.sh)
#   Runs every smoke check after the install finishes so the diagnostics block
#   below has accurate PASS/FAIL state. Records its own pass/fail counts but
#   never aborts the script — failures here just surface in the summary.
# ============================================================================
step "Post-install verification"

_V_PASS=0; _V_FAIL=0; _V_MISS=0; _V_SKIP=0
_v_record() {
    # _v_record <status> <name> <detail>
    local status="$1" name="$2" detail="${3:-}"
    case "$status" in
        PASS)    _V_PASS=$((_V_PASS+1)); printf '  \033[1;32m[ PASS ]\033[0m %-32s %s\n' "$name" "$detail" ;;
        FAIL)    _V_FAIL=$((_V_FAIL+1)); printf '  \033[1;31m[ FAIL ]\033[0m %-32s %s\n' "$name" "$detail" ;;
        MISSING) _V_MISS=$((_V_MISS+1)); printf '  \033[1;33m[MISSING]\033[0m %-32s %s\n' "$name" "$detail" ;;
        SKIP)    _V_SKIP=$((_V_SKIP+1)); printf '  \033[2m[ SKIP ]\033[0m %-32s %s\n' "$name" "$detail" ;;
    esac
}
_v_resolve() {
    local c
    for c in "$@"; do
        if [[ -x "$c" ]]; then echo "$c"; return 0; fi
        if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
    done
    return 1
}
_v_cmd() {
    # _v_cmd <name> <flag> <candidate...>
    local name="$1" flag="$2"; shift 2
    local bin out rc
    if ! bin=$(_v_resolve "$@"); then
        _v_record MISSING "$name" "none of: $*"; return
    fi
    out=$("$bin" $flag 2>&1); rc=$?
    out=$(printf '%s' "$out" | head -n 1 | tr -d '\r')
    if (( rc == 0 )); then
        _v_record PASS "$name" "$bin :: $out"
    else
        _v_record FAIL "$name" "$bin exit=$rc :: $out"
    fi
}
_v_gui() {
    # GUI check: --version + scan stderr for sandbox/userns errors + ldd
    local name="$1"; shift
    local bin out rc missing_libs
    if ! bin=$(_v_resolve "$@"); then
        _v_record MISSING "$name" "none of: $*"; return
    fi
    out=$("$bin" --version 2>&1); rc=$?
    if (( rc != 0 )); then
        missing_libs=$(ldd "$bin" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')
        if [[ -n "$missing_libs" ]]; then
            _v_record FAIL "$name" "missing libs: ${missing_libs% }"
        else
            _v_record FAIL "$name" "exit=$rc :: $(printf '%s' "$out" | head -n 2 | tr '\n' ' ' | tr -d '\r')"
        fi
        return
    fi
    if printf '%s' "$out" | grep -qiE 'CanCreateUserNamespace|user namespace|sandbox.*EACCES|clone\(\) failure'; then
        _v_record FAIL "$name" "sandbox/userns blocked — set kernel.apparmor_restrict_unprivileged_userns=0"
        return
    fi
    _v_record PASS "$name" "$bin :: $(printf '%s' "$out" | head -n 1 | tr -d '\r')"
}
_v_venv() {
    local name="$1" prefix="$2" script="$3"
    local py="$prefix/venv/bin/python" out rc
    if [[ ! -x "$py" ]]; then
        _v_record MISSING "venv: $name" "no python at $py"; return
    fi
    out=$("$py" -c "$script" 2>&1); rc=$?
    out=$(printf '%s' "$out" | tr '\n' ' ' | tr -d '\r')
    if (( rc == 0 )); then _v_record PASS "venv: $name" "$out"
    else                  _v_record FAIL "venv: $name" "$out"; fi
}
_v_service() {
    local svc="$1" state
    command -v systemctl >/dev/null 2>&1 || { _v_record SKIP "service: $svc" "no systemctl"; return; }
    systemctl cat "$svc" >/dev/null 2>&1 || { _v_record MISSING "service: $svc" "unit not registered"; return; }
    state=$(systemctl is-active "$svc" 2>/dev/null || true)
    case "$state" in
        active)           _v_record PASS "service: $svc" "active" ;;
        inactive|failed)  _v_record FAIL "service: $svc" "$state" ;;
        *)                _v_record FAIL "service: $svc" "${state:-unknown}" ;;
    esac
}
_v_port() {
    local name="$1" port="$2" listener=""
    if command -v ss >/dev/null 2>&1; then
        listener=$(ss -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $4; exit}')
    elif command -v netstat >/dev/null 2>&1; then
        listener=$(netstat -ltn 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $4; exit}')
    else
        _v_record SKIP "port: $name ($port)" "no ss/netstat"; return
    fi
    if [[ -n "$listener" ]]; then _v_record PASS "port: $name ($port)" "listening on $listener"
    else                          _v_record FAIL "port: $name ($port)" "nothing listening"; fi
}

printf '\n\033[1;36m-- Toolchain --\033[0m\n'
for b in gcc g++ make cmake git; do _v_cmd "$b" "--version" "$b"; done
_v_cmd "python${PYTHON_VER}" "--version" "python${PYTHON_VER}" python3
_v_cmd "pip" "--version" pip3 pip

printf '\n\033[1;36m-- NVIDIA / CUDA --\033[0m\n'
_v_cmd "nvidia-smi" "" nvidia-smi
_v_cmd "nvcc"       "--version" nvcc /usr/local/cuda/bin/nvcc
if ldconfig -p 2>/dev/null | grep -q 'libcudart.so'; then
    _v_record PASS "cuda runtime libs" "$(ldconfig -p | grep -m1 'libcudart.so' | awk '{print $NF}')"
else
    _v_record MISSING "cuda runtime libs" "libcudart.so not in ldconfig"
fi
# NCCL: required for multi-GPU NVLink/NVSwitch collectives (incl. NVLS on B300).
_nccl_h=""; _nccl_l=""
for _f in /usr/include/nccl.h /usr/local/cuda/include/nccl.h; do
    [[ -f "$_f" ]] && { _nccl_h="$_f"; break; }
done
for _f in /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so \
          /usr/local/cuda/lib64/libnccl.so.2; do
    [[ -e "$_f" ]] && { _nccl_l="$_f"; break; }
done
if [[ -n "$_nccl_h" && -n "$_nccl_l" ]]; then
    _nccl_v=$(awk '/NCCL_MAJOR/ {maj=$3} /NCCL_MINOR/ {min=$3} /NCCL_PATCH/ {pat=$3} END {if(maj!="") print maj"."min"."pat}' "$_nccl_h" 2>/dev/null)
    _v_record PASS "NCCL" "${_nccl_v:-installed} ($_nccl_l)"
elif [[ -n "$_nccl_l" ]]; then
    _v_record FAIL "NCCL" "runtime present but libnccl-dev (header) missing — llama.cpp won't link"
else
    _v_record MISSING "NCCL" "libnccl2/libnccl-dev not installed — multi-GPU will be slow"
fi
# Multi-GPU services
_v_service "nvidia-fabricmanager"
_v_service "nvidia-persistenced"
_v_service "nvidia-dcgm"
_v_cmd "all_reduce_perf" "" all_reduce_perf /usr/local/bin/all_reduce_perf
_v_cmd "numactl"         "--show" numactl
# NVLink lane health
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    _nv_status=$(nvidia-smi nvlink --status 2>/dev/null || true)
    if [[ -n "$_nv_status" ]]; then
        _nv_down=$(printf '%s\n' "$_nv_status" | grep -ciE 'inactive|<inactive>|disabled' || true)
        _nv_up=$(printf '%s\n' "$_nv_status" | grep -c 'GB/s' || true)
        if (( _nv_down == 0 )) && (( _nv_up > 0 )); then
            _v_record PASS "NVLink fabric" "$_nv_up link(s) active, 0 inactive"
        elif (( _nv_down > 0 )); then
            _v_record FAIL "NVLink fabric" "$_nv_down inactive NVLink lane(s) — degraded multi-GPU bandwidth"
        else
            _v_record SKIP "NVLink fabric" "no NVLink links reported (single-GPU or no NVSwitch)"
        fi
    fi
fi

printf '\n\033[1;36m-- GUI launch capability --\033[0m\n'
if [[ -r /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
    _v=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns)
    if [[ "$_v" == "0" ]]; then _v_record PASS "apparmor userns" "unrestricted (0)"
    else _v_record FAIL "apparmor userns" "restricted ($_v) — Chrome/Code will not launch"; fi
fi
for sb in /opt/google/chrome/chrome-sandbox /usr/share/code/chrome-sandbox; do
    [[ -f "$sb" ]] || continue
    p=$(stat -c '%a %U' "$sb" 2>/dev/null)
    if [[ "$p" == "4755 root" ]]; then _v_record PASS "chrome-sandbox SUID" "$sb ($p)"
    else _v_record FAIL "chrome-sandbox SUID" "$sb has $p (expected 4755 root)"; fi
done

printf '\n\033[1;36m-- Browsers & GUI apps --\033[0m\n'
_v_gui "VS Code"       code /usr/bin/code /usr/share/code/code
_v_gui "Google Chrome" google-chrome-stable google-chrome /opt/google/chrome/google-chrome
_v_gui "Firefox"       firefox /opt/firefox/firefox /usr/local/bin/firefox

printf '\n\033[1;36m-- Developer CLIs --\033[0m\n'
_v_cmd "Node.js"  "--version" node /opt/nodejs/bin/node /usr/local/bin/node
_v_cmd "npm"      "--version" npm  /opt/nodejs/bin/npm  /usr/local/bin/npm
_v_cmd "npx"      "--version" npx  /opt/nodejs/bin/npx  /usr/local/bin/npx
_v_cmd "Bun"      "--version" bun  /usr/local/bin/bun
_v_cmd "Opencode" "--version" opencode /usr/local/bin/opencode

printf '\n\033[1;36m-- llama.cpp --\033[0m\n'
_v_cmd "llama-cli"    "--version" "$LLAMA_PREFIX/build/bin/llama-cli"    llama-cli
_v_cmd "llama-server" "--version" "$LLAMA_PREFIX/build/bin/llama-server" llama-server

printf '\n\033[1;36m-- Python venvs --\033[0m\n'
_v_venv "inference" "$INFERENCE_PREFIX" \
    'import sys, torch; print(f"py{sys.version_info.major}.{sys.version_info.minor} torch={torch.__version__} cuda={torch.cuda.is_available()}")'
_v_venv "training"  "$TRAINING_PREFIX" \
    'import torch, torch_geometric; print(f"torch={torch.__version__} pyg={torch_geometric.__version__} cuda={torch.cuda.is_available()}")'
_v_venv "jupyter"   "$JUPYTER_PREFIX" \
    'import jupyterlab, notebook; print(f"jupyterlab={jupyterlab.__version__} notebook={notebook.__version__}")'
if [[ -x "$INFERENCE_PREFIX/venv/bin/python" ]]; then
    _v_venv "inference: vLLM" "$INFERENCE_PREFIX" 'import vllm; print(f"vllm={vllm.__version__}")'
fi

printf '\n\033[1;36m-- K3s / Kubernetes --\033[0m\n'
_v_cmd "k3s"     "--version" k3s     /usr/local/bin/k3s
_v_cmd "kubectl" "version --client=true --output=yaml" kubectl /usr/local/bin/kubectl
_v_cmd "helm"    "version --short"   helm    /usr/local/bin/helm
_v_service "k3s"
_v_service "k3s-agent"

printf '\n\033[1;36m-- Remote desktop --\033[0m\n'
_v_service "xrdp"
_v_port "RDP" 3389

printf '\n\033[1;36m-- Broken dpkg state --\033[0m\n'
if command -v dpkg >/dev/null 2>&1; then
    _broken=$(dpkg -l 2>/dev/null | awk '/^.[HUF]/ {print $2}')
    _bcount=$(printf '%s\n' "$_broken" | grep -c '\S' || true)
    if (( _bcount == 0 )); then _v_record PASS "dpkg health" "no broken packages"
    else _v_record FAIL "dpkg health" "$_bcount broken: $(echo $_broken | tr '\n' ' ' | cut -c1-120)..."
    fi
fi

printf '\n\033[1;35m== Verification summary ==\033[0m\n'
printf '  \033[1;32mPASS\033[0m    : %d\n' "$_V_PASS"
printf '  \033[1;31mFAIL\033[0m    : %d\n' "$_V_FAIL"
printf '  \033[1;33mMISSING\033[0m : %d\n' "$_V_MISS"
printf '  \033[2mSKIP\033[0m    : %d\n' "$_V_SKIP"
(( _V_FAIL + _V_MISS > 0 )) && warn "Post-install verification: $_V_FAIL failed, $_V_MISS missing (see above)."

# ============================================================================
# Summary
# ============================================================================
step "Installation diagnostics"
print_final_diagnostics 0
exit 0
: <<'LEGACY_SUMMARY_EOF'

  APT packages  : installed from $BUNDLE_DIR/debs/
  VS Code       : $(command -v code 2>/dev/null && code --version 2>/dev/null | head -1 || echo "installed (check GUI)")
  Chrome        : $(command -v google-chrome-stable 2>/dev/null && google-chrome-stable --version 2>/dev/null || echo "installed (check GUI)")
  Firefox       : $(/opt/firefox/firefox --version 2>/dev/null || echo "installed at /opt/firefox")
  Opencode      : $(command -v opencode 2>/dev/null && (opencode --version 2>/dev/null || echo "installed") || echo "not installed")
  Node.js       : $(command -v node 2>/dev/null && node --version || echo "not installed")
  npm           : $(command -v npm  2>/dev/null && npm  --version || echo "not installed")
  Bun           : $(command -v bun  2>/dev/null && bun  --version || echo "not installed")
  Desktop (xrdp) : port 3389 — connect with any RDP client (use Linux username/password)
  llama.cpp          : $LLAMA_PREFIX/build/bin/llama-cli (server: llama-server)
  LLM Inference venv : $INFERENCE_PREFIX/venv   (vLLM + LLM_API_fast + RAG, torch 2.11.0+cu130)
  General Training   : $TRAINING_PREFIX/venv    (PyG + MeshGraphNets + SimulGenVAE, torch 2.11.0+cu130)
  Jupyter venv       : $JUPYTER_PREFIX/venv  (start: bash ~/start-jupyter.sh)
  K3s role           : ${K3S_ROLE}  $(command -v kubectl >/dev/null 2>&1 && kubectl get nodes 2>/dev/null | tail -n +2 || true)

Activate venvs:
  source $INFERENCE_PREFIX/venv/bin/activate    # vLLM / LLM_API_fast / RAG
  source $TRAINING_PREFIX/venv/bin/activate     # PyG / MeshGraphNets / SimulGenVAE
  source $JUPYTER_PREFIX/venv/bin/activate      # JupyterLab

Run llama-server:
  $LLAMA_PREFIX/build/bin/llama-server -m /path/to/model.gguf --host 0.0.0.0 --port 8080

Run vLLM server:
  source $INFERENCE_PREFIX/venv/bin/activate
  python -m vllm.entrypoints.openai.api_server --model /path/to/model --host 0.0.0.0 --port 8000

Run JupyterLab:
  bash ~/start-jupyter.sh   # (run inside tmux for persistence)

Remote desktop:
  Windows: mstsc → server_ip:3389 (or use Remmina on Linux)
  CUDA / NVIDIA driver: reboot after driver/kernel packages install, then verify with:
    nvidia-smi && nvcc --version

LEGACY_SUMMARY_EOF
