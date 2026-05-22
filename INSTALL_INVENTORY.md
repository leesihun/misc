# Complete Install Inventory ??Two-Bundle Setup (Ubuntu 24.04 + 8x B300)

Two independent airgap bundles, run in sequence on the target.

**Bundle 1 ??NVIDIA stack** (R580 LTS driver + CUDA 13.0 + FM + NVLSM + NCCL + DCGM)
- `gather-nvidia.sh` (WSL ??`nvidia-airgap-bundle-ubuntu24.04.bin`)
- `pre-install-nvidia.sh` (target preflight)
- `install-nvidia.sh` (target install ??**reboot required after**)
- `test-nvidia.sh` (target verify)

**Bundle 2 ??Userland** (apt packages, apps, Python venvs, llama.cpp, xfce4/xrdp)
- `gather-all.sh` (WSL ??`all-airgap-bundle-ubuntu24.04.bin`)
- `pre-install-check.sh` (target preflight)
- `install-all.sh` (target install)
- `test-all.sh` (target verify)

**Sequence on target (clean Ubuntu 24.04 with DOCA-OFED 3.2+ pre-installed by vendor):**
1. `sudo bash pre-install-nvidia.sh`
2. `sudo bash install-nvidia.sh`
3. `sudo reboot`  ??mandatory
4. `sudo bash test-nvidia.sh`
5. `sudo bash pre-install-check.sh`
6. `sudo bash install-all.sh`
7. `sudo bash test-all.sh`

Sections 1??1 document the **userland bundle**.
Section 0 (immediately below) documents the **NVIDIA bundle**.
Section A covers what is still **vendor / hardware responsibility**.

All paths are on the target machine unless noted.

---

## 0. NVIDIA stack bundle (R580 LTS + CUDA 13.0)

### 0.1 APT packages (from NVIDIA CUDA repo `ubuntu2404/x86_64`)

**Driver ??R580 LTS** (supported until 2028-06; baseline 580.159.04 at gather time):
- `nvidia-driver-580-open` ??open kernel modules (mandatory for Blackwell)
- `nvidia-driver-pinning-580` ??apt unattended-upgrade guard
- `nvidia-modprobe`
- `cuda-drivers-580` ??metapackage that ties FM to driver version

**NVSwitch / NVLink5** (B300 = 4th-gen NVSwitch):
- `cuda-drivers-fabricmanager-580`
- `nvidia-fabricmanager-580`
- `nvlsm` ??NVLink Subnet Manager (runs as child of nvidia-fabricmanager)
- `libnvidia-nscq-580`
- `libnvsdm-580`

**Persistence + RDMA**:
- `nv-persistence-mode`
- `nvidia-peermem-loader` ??requires DOCA-OFED already installed

**CUDA toolkit 13.0** (explicit list ??avoids `cuda` / `cuda-13-0` metapkg that drags driver in again):
- `cuda-toolkit-13-0`
- `cuda-cudart-13-0`
- `cuda-cudart-dev-13-0`
- `cuda-compat-13-0`

**NCCL** (strict `+cuda13.0` suffix ??guards against accidental `+cuda13.2`):
- `libnccl2`  (e.g. `2.28.9-1+cuda13.0`, resolved at gather time)
- `libnccl-dev` (matching version)

**Monitoring**:
- `datacenter-gpu-manager-4-cuda13` ??DCGM 4.3.x+

Plus the full transitive .deb closure via `apt-rdepends`. Bundle size: **~2?? GB**.

### 0.2 Local apt repo + pins

| Path | What |
|------|------|
| `/var/tmp/airgap-nvidia-debs/` | Local `file://` apt repo (NVIDIA bundle's debs/) |
| `/etc/apt/sources.list.d/00-nvidia-bundle.list` | `deb [trusted=yes] file:///var/tmp/airgap-nvidia-debs ./` |
| `/etc/apt/preferences.d/99-nvidia-prefer-bundle` | Priority 1001 for nvidia-*/cuda-*/libnvidia-*/libnvsdm-*/libnccl*/nvlsm/nv-persistence-mode/datacenter-gpu-manager-* |
| `apt-mark hold` | Applied to every installed nvidia-*/cuda-*/libnvidia-*/libnvsdm-*/libnccl*/nvlsm/nv-persistence-mode/nvidia-peermem-loader/datacenter-gpu-manager-* package |
| `/var/lib/install-nvidia/nvidia-held.txt` | Manifest of held packages |

### 0.3 systemd units enabled

| Unit | Purpose |
|------|---------|
| `nvidia-fabricmanager.service` | NVSwitch routing config + spawns NVLSM daemon as child process |
| `nvidia-persistenced.service` | Persistence mode (no init cost between job runs) |
| `nvidia-dcgm.service` | DCGM telemetry / health monitoring |
| `nvidia-nvlsm.service` | Only enabled if NVIDIA build ships it as a separate unit |

### 0.4 Kernel / modules

| Path | What |
|------|------|
| `/etc/modprobe.d/blacklist-nouveau-nvidia.conf` | Blacklists nouveau (mandatory) |
| initramfs | `update-initramfs -u` re-run after nouveau blacklist |

### 0.5 Logs (transient ??kept for debug)

| Path | What |
|------|------|
| `$SCRIPT_DIR/install-nvidia-<timestamp>.log` | Full stdout/stderr transcript |
| `/tmp/preinstall-nvidia-report-<timestamp>.log` | pre-install-nvidia.sh report |

### 0.6 Reboot

install-nvidia.sh exits asking for a reboot. `nvidia.ko` loads on next boot; FM/NVLSM initialize the NVSwitch fabric automatically via the enabled systemd units. Run `test-nvidia.sh` after the reboot to verify all 8 GPUs report Fabric State = Completed.

### 0.7 What install-nvidia.sh does NOT touch

- **DOCA-OFED** ??vendor pre-installed; pre-install-nvidia.sh verifies `ofed_info -s` reports `OFED-internal-25.10+` (DOCA 3.2+)
- **NVSwitch / NIC firmware** ??vendor
- **GPU mode** (MIG / ECC / clocks / power limit / compute mode) ??operator policy (see Section B.1)
- **Userland** ??separate install-all.sh

---

## 1. APT packages (from bundle's `debs/` via `apt install`)

### Python 3.12 ecosystem
- python3.12-venv
- python3.12-dev
- python3-pip

### Build toolchain
- build-essential (gcc, g++, make, libc6-dev)
- cmake
- ninja-build
- pkg-config
- ccache
- git
- curl
- wget
- ca-certificates
- unzip
- xz-utils

### Scientific native libs (for Python wheel native builds)
- libopenblas-dev
- libopenblas0
- libgomp1
- libhdf5-dev
- libssl-dev
- libffi-dev
- libcurl4-openssl-dev

### Editors
- gedit
- vim
- nano

### Monitoring
- htop
- btop
- nvtop
- iotop

### Terminal multiplexer
- tmux
- screen

### Networking diagnostics
- net-tools
- iproute2
- dnsutils
- mtr-tiny
- traceroute

### Utilities
- jq
- tree
- ncdu
- zip
- pigz
- zstd
- rsync

### Daemon-restart helper (replaces reboots for lib upgrades)
- needrestart

### NUMA / hardware topology
- numactl
- hwloc-nox

### GUI runtime libs (Chrome / VS Code Electron deps; t64-renamed on 24.04)
- libglib2.0-0 ??libglib2.0-0t64
- libatk1.0-0 ??libatk1.0-0t64
- libatk-bridge2.0-0 ??libatk-bridge2.0-0t64
- libcairo2
- libcups2 ??libcups2t64
- libdbus-1-3
- libdrm2
- libexpat1
- libfontconfig1
- fonts-liberation
- libgbm1
- libgtk-3-0 ??libgtk-3-0t64
- libnspr4
- libnss3
- libpango-1.0-0
- libsecret-1-0
- libasound2t64
- libx11-6, libx11-xcb1, libxcb1
- libxcomposite1, libxcursor1, libxdamage1, libxext6
- libxfixes3, libxi6, libxkbcommon0, libxkbfile1
- libxrandr2, libxrender1, libxss1, libxtst6
- xdg-utils

### Desktop environment (only if `INSTALL_DESKTOP=1`, default)
- xfce4
- xfce4-goodies
- xfce4-terminal
- xfce4-screenshooter
- xfce4-taskmanager
- xfce4-notifyd
- lightdm
- lightdm-gtk-greeter
- xrdp
- xorgxrdp
- ssl-cert
- policykit-1-gnome (mitigates xrdp #3248 auth-prompt issue)
- dbus-x11
- x11-xserver-utils
- x11-utils
- xauth
- xinit
- xterm
- file-roller
- evince
- ristretto
- xclip
- dconf-editor
- fonts-dejavu-core
- fonts-noto-core
- fonts-noto-color-emoji
- adwaita-icon-theme
- gnome-themes-extra
- p7zip-full
- bash-completion

Plus the full transitive dependency closure (typically ~150 additional packages
resolved by `apt-rdepends` during gather, deduped to newest version per package).

---

## 2. .deb apps (via `apt install ./<file>.deb` ??resolves t64 deps)
- **Visual Studio Code** (`apps/vscode.deb`) ??`/usr/bin/code`
- **Google Chrome** (`apps/chrome.deb`) ??`/opt/google/chrome/`, `/usr/bin/google-chrome-stable`

After install: `systemctl reload apparmor` to register Chrome/VS Code AppArmor profiles.

---

## 3. Tarball / binary apps
| App | Source | Target |
|-----|--------|--------|
| Firefox | `apps/firefox.tar.xz` | `/opt/firefox/`, symlink `/usr/local/bin/firefox`, `.desktop` entry |
| Node.js v22 LTS | `apps/nodejs.tar.xz` | `/opt/nodejs/`, symlinks `/usr/local/bin/{node,npm,npx}` |
| Bun | `apps/bun-linux-x64.zip` | `/usr/local/bin/bun`, symlink `/usr/local/bin/bunx` |
| Opencode | `apps/opencode` | `/usr/local/bin/opencode` |

---

## 4. Python venvs (under `/scratch/` by default ??`SCRATCH_ROOT` override available)

### `/scratch/llm_inference/venv` (INSTALL_INFERENCE=1)
**Bootstrap:** pip, wheel, setuptools

**PyTorch stack (cu130):**
- torch==2.11.0+cu130
- torchvision
- torchaudio

**vLLM:**
- vllm (latest, resolved against cu130 wheel index)

**Transformers / NLP:**
- transformers, tokenizers, safetensors, huggingface-hub, tiktoken
- sentence-transformers, faiss-cpu, rank-bm25

**LangChain / LangGraph:**
- langchain, langchain-core, langchain-community, langchain-ollama
- langgraph, langgraph-checkpoint, langgraph-prebuilt, langsmith
- ollama, tavily-python

**Web / API:**
- fastapi, uvicorn, pydantic, pydantic-settings, sse-starlette
- httpx, httpx-sse, aiohttp, aiofiles, websockets

**Auth:**
- passlib, python-jose

**Document parsing:**
- PyMuPDF, pypdf, python-docx, python-pptx, openpyxl

**Numerics / utilities:**
- pandas, numpy, Pillow, python-dotenv, python-multipart
- jupyter_client, ipykernel, filelock, tqdm, rich

**Per-project (auto-detected at gather):**
- Anything in `LLM_API_fast/requirements.txt`
- Anything in `temp/LLM_API/requirements.txt`

**Install-time PTX-JIT warmup:** B300 (sm_103) is not in PyTorch 2.11 cubin list,
so a `torch.zeros(1, device='cuda').sum()` is run once to pre-populate the
JIT cache from compute_100.

---

### `/scratch/general_training/venv` (INSTALL_TRAINING=1)
**Bootstrap:** pip, wheel, setuptools

**PyTorch stack (cu130):**
- torch==2.11.0+cu130
- torchvision
- torchaudio

**PyG ecosystem (from `data.pyg.org/whl/torch-2.11.0+cu130.html`):**
- torch-geometric
- pyg_lib
- torch-scatter
- torch-sparse
- torch-cluster
- torch-spline-conv (best-effort ??may not be published on cu130 index)

**Scientific:**
- numpy, scipy, h5py, pandas, tqdm
- matplotlib, seaborn, Pillow
- scikit-learn, scikit-image, statsmodels, networkx, sympy

**Training utilities:**
- torchinfo, tensorboard

**Vision / audio:**
- opencv-python, imageio, librosa, audiomentations, soxr, natsort

**Other:**
- reportlab, paramiko, smbprotocol

**Per-project (auto-detected at gather):**
- MeshGraphNets ??variational/requirements.txt
- SimulGenVAE/requirements.txt
- PEMTRON_warpage/requirements.txt
- PEMTRON_warpage/data_autotransfer/requirements.txt

---

### `/scratch/jupyter/venv` (INSTALL_JUPYTER=1)
- jupyterlab, notebook, ipykernel, ipywidgets, jupyter-server
- pandas, polars, numpy, scipy
- matplotlib, seaborn, plotly
- scikit-learn, statsmodels
- tqdm, rich, requests, aiohttp
- black, ruff, mypy, pytest, ipdb

**Side effects:**
- IPython kernel registered as `airgap-py3.12`
- `~/start-jupyter.sh` launcher created in the real user's home

---

### `/scratch/llama.cpp/venv` (INSTALL_LLAMA=1)
For `convert_hf_to_gguf.py` and friends. Contents come from llama.cpp's own
`requirements*.txt`, typically:
- torch (older version, separate from cu130 stack ??that's why it's isolated)
- gguf, safetensors, transformers, protobuf, sentencepiece

---

## 5. Source-built binaries

### llama.cpp (INSTALL_LLAMA=1) ??under `/scratch/llama.cpp/build/bin/`
Built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100;103 -DGGML_BLAS=ON -DLLAMA_CURL=ON -DLLAMA_BUILD_UI=OFF` against the vendor's nvcc 13.0:
- llama-cli
- llama-server
- llama-quantize, llama-bench, llama-perplexity, llama-embedding
- llama-tokenize, llama-batched, llama-batched-bench
- llama-export-lora, llama-gguf-split, llama-imatrix
- llama-lookahead, llama-lookup, llama-parallel, llama-passkey
- llama-retrieval, llama-save-load-state, llama-simple, llama-speculative
- llama-tts, llama-vdot, llama-q8dot, llama-eval-callback
- (full llama.cpp examples set)

> **Note:** llama.cpp is no longer linked against system NCCL. Multi-GPU inference falls back
> to per-pair P2P over NVLink instead of NCCL collectives ??works on B300 NVSwitch but
> doesn't get NVLS in-fabric reductions. vLLM is unaffected (PyTorch cu130 wheels ship
> their own `nvidia-nccl-cu13` Python package).

---

## 6. Helper scripts (`/usr/local/bin/`, mode 0755)
- `gpu-health-check` ??fabric / NVLink / fabric.state sanity check
- `llama-server-multigpu` ??NUMA + tensor-split wrapper for `llama-server`
- `llama-model-preload` ??pre-mmap GGUF model into page cache

---

## 7. systemd units
| Path | Purpose |
|------|---------|
| `/etc/systemd/system/llama-server@.service` | Instanced llama-server template. Per-instance config: `/etc/llama-server/<name>.env` |
| `/etc/systemd/system/disable-thp-defrag.service` | Sets `transparent_hugepage=madvise` + `defrag=defer+madvise` on boot. Enabled. |
| (existing) `xrdp.service` | Enabled + restarted if `INSTALL_DESKTOP=1` |

---

## 8. Config files written

### APT
| Path | What |
|------|------|
| `/etc/apt/preferences.d/99-nvidia-prefer-origin` | Pins nvidia-*/cuda-*/libnvidia-* to `origin "developer.download.nvidia.com"` priority 1001 |
| `/etc/apt/sources.list.d/00-bundle.list` | `deb [trusted=yes] file:///var/tmp/airgap-bundle-debs ./` |
| `/var/tmp/airgap-bundle-debs/` | Local file:// apt repo (copy of bundle's debs/). Uses bundled `Packages` if `dpkg-scanpackages` is not installed yet. |
| `apt-mark hold` | Applied to every installed nvidia-*/cuda-*/libnvidia-*/libcudart*/libcublas*/libcudnn*/libcurand*/libcufft*/libcusparse*/libcusolver*/libnpp* package; holds remain in place after install. |

### Kernel / sysctl
| Path | Settings |
|------|----------|
| `/etc/sysctl.d/99-llm-multigpu.conf` | `vm.overcommit_memory=1`, `vm.swappiness=0`, `vm.max_map_count=1048576`, `net.core.rmem_max=268435456`, `net.core.wmem_max=268435456` |
| `/etc/sysctl.d/60-apparmor-userns.conf` | `kernel.apparmor_restrict_unprivileged_userns=0` (Chrome/VS Code sandbox on Ubuntu 24.04) |

### System limits
| Path | Settings |
|------|----------|
| `/etc/security/limits.d/99-llm-multigpu.conf` | `nofile=1048576`, `nproc=524288`, `memlock=unlimited`, `stack=65536` (both soft + hard, for `*`) |
| `/etc/systemd/system.conf.d/99-llm-multigpu.conf` | Mirrors above into systemd `DefaultLimit*` (services started outside PAM) |

### Desktop (if INSTALL_DESKTOP=1)
| Path | What |
|------|------|
| `/etc/xrdp/startwm.sh` | Launches `startxfce4` and autostarts polkit-gnome auth agent (fixes xrdp #3248) |
| `/etc/skel/.xsession` | `xfce4-session` (default for new users) |
| `~/.xsession` (real user) | `xfce4-session` |
| `/usr/share/polkit-1/rules.d/49-xfce-shutdown.rules` | Allows `sudo` group to power-off/reboot from XFCE |
| `/usr/share/applications/firefox-manual.desktop` | Firefox desktop entry |
| User added to `ssl-cert` group | `adduser xrdp ssl-cert` |
| UFW rule (if active) | `ufw allow 3389/tcp` |

### llama-server config
| Path | What |
|------|------|
| `/etc/llama-server/` | Directory (per-instance env files go here) |
| `/etc/llama-server/example.env` | Template: `MODEL`, `HOST`, `PORT`, `NGL`, `CTX`, `EXTRA` |

### State (transient)
| Path | What |
|------|------|
| `/var/lib/install-all-prepped/stage1.done` | Resume marker, only present during conditional 2-stage reboot path. Deleted at end of install. |
| `/var/lib/install-all-prepped/nvidia-held.txt` | Audit list of NVIDIA packages held by install-all.sh; not used to unhold automatically |

### Logs (transient ??kept for debug)
| Path | What |
|------|------|
| `$SCRIPT_DIR/install-all-<timestamp>.log` | Full stdout/stderr transcript of the install |
| `$SCRIPT_DIR/install-diagnostics-<timestamp>.log` | End-of-run diagnostics summary with overall status, causes, warnings, errors, and follow-up commands |
| `/tmp/preinstall-report-<timestamp>.log` | pre-install-check.sh report |

---

## 9. User-home additions (real user, not root)
- `~/start-jupyter.sh` (mode 0755) ??`source jupyter venv && jupyter lab --ip=0.0.0.0 --port=8888`
- `~/.xsession` ??`xfce4-session` (if INSTALL_DESKTOP=1)

---

## 10. Side effects from `apt install`
- `systemctl reload apparmor` after Chrome/VS Code .debs (registers their AppArmor profiles)
- `needrestart -r a` after all apt operations (restarts daemons holding old libs)
- `sysctl --system` after writing /etc/sysctl.d/* files
- `systemctl daemon-reload` after writing systemd units
- `systemctl daemon-reexec` after writing /etc/systemd/system.conf.d/*

---

## 11. Total footprint (estimates)

### NVIDIA bundle (Section 0)
| Component | Disk |
|-----------|------|
| NVIDIA driver + open kmod | ~600 MB |
| FabricManager + NVLSM + NSCQ + NVSDM | ~150 MB |
| CUDA toolkit 13.0 (toolkit + cudart) | ~3.5 GB |
| NCCL 2.x +cuda13.0 | ~250 MB |
| DCGM 4.x | ~200 MB |
| Transitive .deb deps | ~500 MB |
| **NVIDIA bundle subtotal** | **~5 GB** installed (~2?? GB compressed bundle) |

### Userland bundle (Sections 1??0)
| Component | Disk |
|-----------|------|
| APT packages (post-install) | ~3?? GB |
| VS Code | ~400 MB |
| Chrome | ~250 MB |
| Firefox | ~250 MB |
| Node.js | ~120 MB |
| Bun | ~100 MB |
| Opencode | ~80 MB |
| Inference venv (torch + vLLM dominate) | ~7 GB |
| Training venv | ~5 GB |
| Jupyter venv | ~1 GB |
| llama.cpp build artifacts | ~3 GB |
| **Userland subtotal** | **~20 GB** installed (~4?? GB compressed bundle) |

**Combined target footprint: ~25 GB.** Two bundles to transfer (~6?? GB total).

---

# WHAT IS NOT TOUCHED

## A. Vendor / hardware (still vendor's responsibility)

### Hardware / firmware
- Server BIOS / UEFI firmware
- IPMI / BMC management interface
- Disk RAID configuration
- BIOS boot order / Secure Boot setup (pre-flight reports state, doesn't change it)
- Network interface configuration (netplan / NetworkManager)
- **NVSwitch / NVLink firmware** ??must match driver branch or FM init aborts
- **ConnectX-7 / ConnectX-8 HCA firmware**

### DOCA-OFED
- DOCA-OFED + OFED kernel modules (DOCA 3.2+ / OFED 25.10+ for R580 compatibility)
- `mlx5_core` / `mlx5_ib` / `rdma-core` / `ibverbs` / `opensm`
- Verified by `pre-install-nvidia.sh` (checks N09?밡12); install-nvidia.sh does NOT touch DOCA

### Container / orchestration (explicitly excluded per plan)
- Docker / containerd / podman ??NOT installed
- nvidia-container-toolkit ??NOT installed
- K3s / Helm / kubectl ??NOT installed
- Kubernetes manifests, charts, container images ??NOT bundled

### Moved OUT of "vendor's responsibility" (now handled by install-nvidia.sh ??see Section 0)
- ~~NVIDIA driver~~ ??`nvidia-driver-580-open` (R580 LTS, 580.159.04 baseline)
- ~~cuda-toolkit-13-0 + cuda-* + libcudart/libcublas/...~~ ??bundled
- ~~nvidia-fabricmanager / nvlsm / libnvidia-nscq / libnvsdm~~ ??bundled
- ~~nvidia-persistenced / nvidia-dcgm services~~ ??enabled
- ~~libnccl2~~ ??bundled (`+cuda13.0` strict pin)
- ~~Kernel modules / DKMS~~ ??open kmod, no DKMS path needed (Secure Boot off)
- ~~nouveau blacklist~~ ??`/etc/modprobe.d/blacklist-nouveau-nvidia.conf`
- ~~Fabric state Completed~~ ??verified by `test-nvidia.sh`

---

## B. Gaps the scripts deliberately leave to the operator

These are things a typical 8x B300 production server might still need ??they're not installed because they're either out of scope or vendor/site-specific.

### 1. GPU runtime configuration (vendor or operator decision)
- **Persistence mode** ??pre-flight WARNS if disabled, doesn't run `nvidia-smi -pm 1`. Vendor usually sets this; if not, run manually after first boot.
- **ECC mode** ??pre-flight reports state only. Toggle with `nvidia-smi -e {0|1}` (requires reboot).
- **GPU clock locks** ??`nvidia-smi -lgc` / `-lmc` not applied. Use if you want deterministic perf.
- **Power limit** ??`nvidia-smi -pl` not applied. B300 default TDP is 1000W; site may want lower for thermal headroom.
- **Compute mode** ??left at DEFAULT. For dedicated inference, EXCLUSIVE_PROCESS prevents accidental shared-GPU contention.
- **MIG configuration** ??B300 supports MIG; install-all.sh leaves all GPUs in non-MIG mode. Use `nvidia-smi mig` if you want partitioned GPUs.
- **GPU reset capability** ??not configured. Use `nvidia-smi --gpu-reset` ad-hoc.

### 2. System / OS configuration
- **SSH server config** ??vendor's responsibility. install-all.sh does not touch `/etc/ssh/sshd_config`.
- **Firewall (ufw / nftables)** ??only opens port 3389 (xrdp) IF ufw is already active. No baseline rules added.
- **Hostname / /etc/hosts** ??not modified. Pre-flight warns if hostname doesn't resolve locally (slow sshd logins).
- **Time zone (`timedatectl set-timezone`)** ??not set. Verify with `timedatectl status`.
- **Locale (`localectl set-locale`)** ??not set. xrdp's `startwm.sh` sources `/etc/default/locale` if present.
- **DNS resolver config (`/etc/resolv.conf`, systemd-resolved)** ??not touched.
- **NTP / chrony / systemd-timesyncd** ??pre-flight checks one is active; doesn't install or configure. On airgap, you likely want a local NTP server.
- **Network configuration (`netplan`)** ??vendor's responsibility.
- **User accounts / groups** ??only `adduser xrdp ssl-cert` (for xrdp TLS cert access). No real-user accounts created; assumes operator handles user provisioning.
- **`/etc/sudoers`** ??not modified.
- **`/etc/security/access.conf`** ??not modified.

### 3. Log / cron / housekeeping
- **logrotate config** ??no rotation policy for `install-all-*.log` files; clean them up manually.
- **journald** ??no persistence policy applied. Default is /run/log/journal (in-memory) on Ubuntu. For long-running boxes you may want `Storage=persistent` in `/etc/systemd/journald.conf`.
- **cron / systemd timers** ??none added.
- **`unattended-upgrades`** ??NOT configured (and shouldn't be on airgap).

### 4. Model / data workflow
- **`/opt/models` directory** ??referenced in `llama-server@.service` example.env but NOT created. Create manually before using the systemd template.
- **HuggingFace cache (`HF_HOME`, `TRANSFORMERS_CACHE`)** ??not set. Defaults to `~/.cache/huggingface/`. For shared model storage, point all users at `/scratch/hf-cache` via `/etc/profile.d/`.
- **`huggingface-cli login` / `HF_TOKEN`** ??must be done manually per user. The `huggingface-hub` CLI ships with the inference venv.
- **vLLM cache (`VLLM_CACHE_ROOT`)** ??defaults to `~/.cache/vllm/`.
- **Global `pip.conf`** ??not created. The venvs install offline from `--find-links=` so no index config needed; if you add wheels later, you'll want a `~/.pip/pip.conf` with `--no-index` defaults.

### 5. Monitoring / observability
- **Prometheus `node_exporter`** ??NOT installed.
- **DCGM exporter** ??NOT installed (DCGM itself is vendor's choice).
- **Grafana / Loki / Telegraf / Fluentd** ??NOT installed.
- **GPU benchmark baseline** ??install-all.sh does NOT install nccl-tests or any bandwidth tool. To verify NVSwitch is healthy, install `nccl-tests` manually from NVIDIA (`https://github.com/NVIDIA/nccl-tests`) ??it builds against `nvidia-nccl-cu13` from the PyTorch wheel, or against any libnccl the vendor already has on disk.
- **System-wide NCCL tuning** ??`/etc/profile.d/nccl-multigpu.sh` is NO LONGER written. PyTorch / vLLM use NCCL's defaults. If you want `NCCL_NVLS_ENABLE=1` for in-fabric reductions on B300 NVSwitch, set it in your job submission script or `~/.bashrc`.

### 6. Multi-node / HPC
- **InfiniBand / DOCA-OFED stack** ??vendor-installed (DOCA 3.2+ / OFED 25.10+ on the verified box); pre-install-nvidia.sh verifies. `nvidia-peermem-loader` (in NVIDIA bundle) hooks `nvidia-peermem.ko` into mlx5 for GPUDirect RDMA.
- **NCCL** ??host `libnccl2` IS bundled by install-nvidia.sh with `+cuda13.0` strict pin. No system-wide env vars set; PyTorch wheels also carry `nvidia-nccl-cu13` independently.
- **SLURM / PBS / LSF** ??NOT bundled.
- **Passwordless SSH between nodes** ??NOT configured.
- **NFS / Lustre / BeeGFS client** ??NOT bundled.
- **MPI (OpenMPI / MPICH)** ??NOT bundled.

### 7. Storage
- **Mount points for shared / scratch storage** ??only `/scratch` is touched (chown'd to real user). If your site has additional NVMe / NFS mounts, configure them via `/etc/fstab`.
- **ZFS / btrfs / LVM** ??vendor's responsibility.
- **Snapshot / backup** ??`rsync` is installed; no `borg` / `restic` / `bup`.
- **smartmontools** ??NOT installed.

### 8. Security hardening
- **`auditd`** ??NOT installed.
- **`fail2ban`** ??NOT installed.
- **AppArmor profile authoring** ??only the Chrome/VS Code profiles shipped by their .debs are reloaded.
- **SELinux** ??Ubuntu uses AppArmor; nothing to do.
- **Secure Boot key enrollment** ??pre-flight reports state only; warns if enabled (DKMS modules would need vendor signing).
- **TLS certs for any services** ??only xrdp uses the ssl-cert auto-generated cert. No custom CA setup.

### 9. Documentation on the server itself
- No `README.md` / `INSTALLED.md` placed under `/etc/` or `/scratch/` describing what's installed.
- The install transcript is the only on-server record; cross-reference with this file (`misc/INSTALL_INVENTORY.md`) from the gather host.

### 10. Reboot policy
- Advisory reboot recommendation at end of install only if `/run/reboot-required` is set.
- No automatic reboot. Operator decides when to reboot.

---

## C. Manual post-install checklist (recommended)

After `install-all.sh` finishes cleanly, the operator should:

1. `bash test-all.sh` ??verify everything passes
2. `gpu-health-check` ??confirm fabric is healthy (FabricManager + NVLink + fabric.state)
3. `nvidia-smi -pm 1` ??set persistence mode if pre-flight Y02 warned
4. `mkdir -p /opt/models && chown $USER /opt/models` ??model storage dir
5. `timedatectl set-timezone <region>` ??set the wall clock
6. Verify `/etc/hosts` resolves the hostname locally (`getent hosts $(hostname)`)
7. Decide on GPU clock policy: `nvidia-smi -lgc` (lock for deterministic perf), `-pl` (power limit), MIG mode
8. If running services as system units, ensure they restart after reboot: `systemctl is-enabled xrdp llama-server@<name> disable-thp-defrag`
9. (Optional) For bandwidth verification, build nccl-tests manually from `https://github.com/NVIDIA/nccl-tests` against the system NCCL installed by install-nvidia.sh (`libnccl2 +cuda13.0`) ??needed only if you suspect fabric degradation.
10. Set up `HF_HOME` and any other cache env vars in `/etc/profile.d/`
11. For maximum vLLM performance on B300 NVSwitch, export `NCCL_NVLS_ENABLE=1 NCCL_P2P_LEVEL=NVL` in your job script (these are no longer set globally)
12. If you want monitoring: install node_exporter and DCGM exporter (out of scope here)
13. Reboot once before declaring production-ready (ensures the THP service, sysctl, and any deferred kmod loads come up cleanly from cold start)

---

*Generated alongside the scripts in this directory. Update when any of
`gather-all.sh`, `install-all.sh`, or `pre-install-check.sh` changes.*
