# Complete Install Inventory ??Two-Bundle Setup (Ubuntu 24.04 + 8x B300)

Two independent airgap bundles, run in sequence on the target.

**Bundle 1 ??NVIDIA stack** (R580 LTS driver + CUDA 13.0 + FM + NVLSM + optional host NCCL + DCGM)
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

The userland phase deliberately fires a `checkpoint_reboot` at the end of
steps 06, 08, 09 (if desktop), 15, and 17 — each checkpoint exits 75, the
launcher prints a "REBOOT REQUIRED" banner, the operator reboots, and a
re-run of `install-all.sh` resumes at the next pending step via the .ok
marker mechanism. This is intentional: bricking the box is more expensive
than rebooting it 4-5 times.

```
1.  sudo bash pre-install-nvidia.sh                   # strict preflight (no /etc writes)
2.  sudo bash install-nvidia.sh                       # driver + FM + NVLSM + CUDA-min
3.  sudo reboot                                       # MANDATORY — loads nvidia.ko
4.  sudo bash test-nvidia.sh                          # fabric Completed? peermem loaded? kmod from running kernel?
5.  sudo bash pre-install-check.sh                    # strict base-OS gate vs userland bundle
6.  sudo bash install-all.sh                          # runs 01-06, exits 75 after apt userland
7.  sudo reboot                                       # confirm nvidia survived apt churn
8.  sudo bash test-nvidia.sh && bash test-all.sh --phase userland
9.  sudo bash install-all.sh                          # resumes 07-08, exits 75 after needrestart
10. sudo reboot                                       # confirm sshd/dbus/polkit came up clean
11. sudo bash test-all.sh --phase apps
12. sudo bash install-all.sh                          # resumes 09 only (if INSTALL_DESKTOP=1)
13. sudo reboot                                       # confirm lightdm doesn't stall graphical.target
14. sudo bash test-all.sh --phase desktop
15. sudo bash install-all.sh                          # resumes 10-15, exits 75 after sysctl
16. sudo reboot                                       # confirm sysctl/THP/limits persist from cold boot
17. sudo bash install-all.sh                          # resumes 16-17, exits 75 at final-status
18. sudo reboot                                       # final cold-boot sanity
19. sudo bash test-nvidia.sh && bash test-all.sh      # full verify
20. gpu-health-check
```

**Total mandatory reboots: 5** (one per failure surface). Set `SKIP_CHECKPOINTS=1`
to bypass the userland reboots — NOT recommended on a box that has bricked
in prior bring-up attempts.

Sections 1??1 document the **userland bundle**.
Section 0 (immediately below) documents the **NVIDIA bundle**.
Section A covers what is still **vendor / hardware responsibility**.

All paths are on the target machine unless noted.

---

## Airgap mapping vs upstream

The target machine is airgapped. Every install-time fetch the upstream flows
assume is replaced with a bundled equivalent. **No script in this repo may
fetch from the internet on the target.**

| Upstream (online flow) | This repo (airgap) |
|------------------------|--------------------|
| `apt update` against `developer.download.nvidia.com` | `file:///var/tmp/airgap-nvidia-debs/` registered as `/etc/apt/sources.list.d/00-nvidia-bundle.list`; package versions resolved against the bundle's `meta/target.env` |
| `apt update` against `archive.ubuntu.com` | `file:///var/tmp/airgap-bundle-debs/` registered as `/etc/apt/sources.list.d/00-bundle.list` |
| `pip install … --index-url https://download.pytorch.org/whl/cu130` | `pip install --no-index --find-links="$BUNDLE_DIR/wheels/inference"` (or `wheels/training` / `wheels/jupyter` / `wheels/llamacpp`) |
| `pip install … --find-links https://data.pyg.org/whl/torch-2.11.0+cu130.html` | Same wheelhouse — PyG wheels are pre-downloaded into `wheels/training/` |
| `git clone https://github.com/ggml-org/llama.cpp` | `tar -xzf $BUNDLE_DIR/src/llama.cpp.tar.gz` — source archived at gather time, commit pinned in `meta/target.env` (`BUNDLE_LLAMA_COMMIT`) |
| `curl https://developer.download.nvidia.com/.../cuda-keyring_1.1-1_all.deb` | Pinned in the NVIDIA bundle; install-nvidia.sh registers the file:// repo, never reaches the internet |
| `pip install vllm` (PyPI) | **No longer bundled.** The inference venv is CPU-only; GPU inference goes through llama.cpp's HTTP server (step 14). |
| `nodejs.org` LTS tarball | Pre-staged at `$BUNDLE_DIR/apps/nodejs.tar.xz` |
| `releases.mozilla.org` Firefox tarball | Pre-staged at `$BUNDLE_DIR/apps/firefox.tar.xz` |
| `update.code.visualstudio.com/.../linux-deb-x64/stable` | Pre-staged at `$BUNDLE_DIR/apps/vscode.deb` |
| `dl.google.com/.../google-chrome-stable_current_amd64.deb` | Pre-staged at `$BUNDLE_DIR/apps/chrome.deb` |
| `github.com/oven-sh/bun/releases/...` zip | Pre-staged at `$BUNDLE_DIR/apps/bun-linux-x64.zip` |
| `github.com/sst/opencode/releases/...` tar.gz | Pre-staged binary at `$BUNDLE_DIR/apps/opencode` |

Gather scripts (`gather-nvidia.sh`, `gather-all.sh`) run on an internet-connected
WSL Ubuntu 24.04 host. They fetch everything above into the bundle. The
gather phase is the ONLY internet-dependent step; the target phase is fully
offline.

---

## 0. NVIDIA stack bundle (R580 LTS + CUDA 13.0)

### 0.1 APT packages (from NVIDIA CUDA repo `ubuntu2404/x86_64`)

**Driver ??R580 LTS** (supported until 2028-06; baseline 580.159.04 at gather time):
- `nvidia-driver-580-open` ??open kernel modules (mandatory for Blackwell)
- `nvidia-driver-pinning-580` ??apt unattended-upgrade guard
- `nvidia-modprobe`
- `cuda-drivers-580` ??metapackage that ties FM to driver version

**NVSwitch / NVLink5** (B300 = 4th-gen NVSwitch):
- `cuda-drivers-fabricmanager-580` (only when NVIDIA repo exposes the meta — auto-skipped on R580)
- `nvidia-fabricmanager-580`
- `nvlsm` ??NVLink Subnet Manager (runs as child of nvidia-fabricmanager)
- `libnvidia-nscq-580`

**Persistence + RDMA**:
- `nvidia-persistenced` (rides transitively with `nvidia-driver-580-open`; no separate apt name)
- `nvidia-peermem.ko` (ships in the open driver package; autoloaded via `/etc/modules-load.d/nvidia-peermem.conf`)

**CUDA toolkit 13.0 — MINIMAL subset** (we DELIBERATELY avoid `cuda-toolkit-13-0` — that ~3 GB metapackage pulls cuFFT/cuSPARSE/NPP/nvJPEG which PyTorch/vLLM venvs already ship via pip nvidia-*-cu13 wheels; installing the system metapackage causes ABI skew):
- `cuda-nvcc-13-0` — nvcc compiler
- `cuda-cudart-13-0` — `libcudart.so` runtime
- `cuda-cudart-dev-13-0` — headers + static lib
- `cuda-cccl-13-0` — Thrust / CUB headers (llama.cpp build needs this)
- `libcublas-13-0` — `libcublas.so` + `libcublasLt.so`
- `libcublas-dev-13-0` — cuBLAS headers
- `libnvjitlink-13-0` — JIT-link (cuBLASLt dlopens this)
- `cuda-compat-13-0` — forward-compat shim, optional but cheap

**Host NCCL** (optional; strict `+cuda13.0` suffix ??guards against accidental `+cuda13.2`):
- `libnccl2`  (e.g. `2.28.9-1+cuda13.0`, resolved at gather time only when `SKIP_NCCL=0`)
- `libnccl-dev` (matching version; only when `SKIP_NCCL=0`)

**Monitoring**:
- `datacenter-gpu-manager-4-cuda13` ??DCGM 4.3.x+

Plus the full transitive .deb closure via `apt-rdepends`. Bundle size: **~2?? GB**.

### 0.2 Local apt repo + pins

| Path | What |
|------|------|
| `/var/tmp/airgap-nvidia-debs/` | Local `file://` apt repo (NVIDIA bundle's debs/) |
| `/etc/apt/sources.list.d/00-nvidia-bundle.list` | `deb [trusted=yes] file:///var/tmp/airgap-nvidia-debs ./` |
| `/etc/apt/preferences.d/99-nvidia-prefer-bundle` | Priority 1001 for `nvidia-*` / `cuda-*` / `libnvidia-*` / `libnvjit*` / `libnvfat*` / `libnccl*` / `nvlsm` / `datacenter-gpu-manager-*` (libnccl matters only for explicit `SKIP_NCCL=0`) |
| `apt-mark hold` | Applied to every installed `nvidia-driver-*`, `nvidia-fabricmanager-*`, `nvidia-persistenced`, `libnvidia-*`, `cuda-drivers`, `cuda-nvcc-*`, `cuda-cudart-*`, `cuda-cccl-*`, `cuda-compat-*`, `libcublas-*`, `libnvjitlink-*`, `libnccl*`, `nvlsm`, `nvidia-modprobe`, `datacenter-gpu-manager-*` package. Userland installer (install-all.d/03-apt-repo.sh) only adds holds on the system runtime libs (libstdc++6/libgcc-s1/libgomp1/libc6). |
| `/var/lib/install-nvidia/nvidia-held.txt` | Manifest of NVIDIA packages held |
| `/etc/profile.d/cuda.sh` | nvcc on PATH for login shells (written by install-nvidia.sh step 5b; does NOT modify LD_LIBRARY_PATH on purpose) |
| `/etc/ld.so.conf.d/cuda-system.conf` | `/usr/local/cuda/lib64` added to ld.so cache so non-RPATH binaries (llama-cli / llama-server) find `libcudart.so.13` (written by install-nvidia.sh step 5b; searched AFTER RUNPATH so PyTorch/vLLM venv-bundled CUDA libs still win) |

### 0.3 systemd units enabled

| Unit | Purpose |
|------|---------|
| `nvidia-fabricmanager.service` | NVSwitch routing config + spawns NVLSM daemon as child process |
| `nvidia-persistenced.service` | Persistence mode (no init cost between job runs) |
| `nvidia-dcgm.service` | DCGM telemetry / health monitoring |
| `nvidia-nvlsm.service` | NVLink Subnet Manager. **NVLSM is MANDATORY on B300** (4th-gen NVSwitch needs it for SHARP). On most R580 builds it ships as a separate unit and `install-nvidia.sh` enables it; on a few it spawns as a child of `nvidia-fabricmanager` instead, in which case `pgrep -x nvlsm` is the authoritative liveness check (used by `test-nvidia.sh`). Either way the process MUST be running. |

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

## 0.8 Userland install steps (`install-all.d/`)

`install-all.sh` is now a thin launcher (~250 lines). The real work lives in
`install-all.d/NN-name.sh` — 17 standalone, directly-runnable step scripts
plus one shared helpers file. Each step writes its own log
(`/var/log/install-all/<RUN_ID>/NN-name.log`) and status marker
(`/var/lib/install-all/steps/NN-name.{ok,failed}`).

| # | Script | Concern |
|---|--------|---------|
| 00 | `install-all.d/00-common.sh` | Sourced by every step. Helpers (log/warn/die/step), step lifecycle (`init_step`/`mark_step_ok`), traps, apt helpers (`_apt_install`, `_pkg_satisfied`, `_normalize_pkg_name` t64 mapping), wheelhouse helpers, `_as_user` (drop to SUDO_USER), bundle metadata sourcing (incl. nvidia bundle's `meta/target.env` → CUDA_MAJOR/MINOR), env-knob defaults. Not executed. |
| 01 | `01-preflight.sh` | Re-exec under sudo, bundle locate + SHA256 + extract, variant guard (`BUNDLE_VARIANT=prepped`), runs `pre-install-check.sh`. |
| 02 | `02-scratch.sh` | `$SCRATCH_ROOT` directory creation + chown. |
| 03 | `03-apt-repo.sh` | System-lib holds, local file:// apt repo, `apt-get update`. **No NVIDIA pin** (install-nvidia.sh's pin already covers nvidia packages and points at the bundle); **no nvidia-* hold** (install-nvidia.sh already holds them). |
| 04 | `04-apt-plan.sh` | `apt -s` dry-run; refuses if a kernel/firmware/microcode upgrade is detected (unless `FORCE=1`); writes proposed/triggers files for step 05. |
| 05 | `05-reboot-trigger-packages.sh` | If step 04 found libc6/systemd/dbus upgrades, installs them only, writes `/var/lib/install-all-prepped/stage1.done`, exits 75 (launcher renders reboot prompt). |
| 06 | `06-apt-userland.sh` | Toolchain, `python${PYTHON_VER}-venv/dev`, needrestart, CLI tools, GUI runtime libs, scientific libs (incl. libssl-dev for llama.cpp OpenSSL HTTP), optional xfce4+xrdp+lightdm. |
| 07 | `07-app-debs.sh` | VS Code + Chrome `.deb` install via `apt install ./`, AppArmor reload, `kernel.apparmor_restrict_unprivileged_userns=0` sysctl. |
| 08 | `08-tarball-apps.sh` | Firefox, Node.js, Bun, Opencode (tarballs). `needrestart -r a`. |
| 09 | `09-desktop-xrdp.sh` | xrdp startwm → `startxfce4`, polkit shutdown rules, UFW 3389. Honors `INSTALL_DESKTOP`. |
| 10 | `10-wheelhouse-manifests.sh` | `generate_wheelhouse_requirements` — emits per-wheelhouse `requirements.txt`. |
| 11 | `11-venv-inference.sh` | **CPU-only** inference venv at `$INFERENCE_PREFIX/venv` — FastAPI + langchain + sentence-transformers + RAG stack. **No torch, no vLLM** (both removed; multi-GPU NCCL ABI footgun). Honors `INSTALL_INFERENCE`. GPU inference is the llama.cpp HTTP server (step 14). |
| 12 | `12-venv-training.sh` | Training venv — torch (cu130), torch-geometric + extensions (pyg_lib, scatter, sparse, cluster; torch_spline_conv unavailable on cu130), SciPy stack. |
| 13 | `13-venv-jupyter.sh` | JupyterLab venv, ipykernel registration, `~/start-jupyter.sh` launcher. |
| 14 | `14-llamacpp-build.sh` | llama.cpp build. cmake flags: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100-real;103-real -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_UI=OFF`. Drops deprecated `LLAMA_CURL`. NCCL link-time auto-disabled because `SKIP_NCCL=1` is the install-nvidia.sh default (intentional). |
| 15 | `15-system-tuning.sh` | sysctl (vm.overcommit, swappiness, max_map_count, net buffers), THP=madvise via `disable-thp-defrag.service`, pam_limits. Refuses to write `/etc/systemd/system.conf.d/*` (see notes in step 15 about unit mismatch). |
| 16 | `16-operational-tooling.sh` | `/usr/local/bin/gpu-health-check`, `/usr/local/bin/llama-server-multigpu`, `/usr/local/bin/llama-model-preload`, `/etc/systemd/system/llama-server@.service`, `/etc/llama-server/example.env`. |
| 17 | `17-final-status.sh` | Clears resume marker, `chown -R $SCRATCH_ROOT`, prints final banner. (Launcher also prints aggregate summary across the run.) |

The launcher (`install-all.sh`) supports:
- `--list` — show step status for the current `RUN_ID`
- `--run NN` (or `--run NN-name`) — run one step, clearing its marker first
- `--from NN` — run steps NN..17
- `--rerun NN[,NN]` — delete `.ok` for these steps, then run them in default order
- `--force` — ignore all `.ok` markers
- `--skip-preflight` — skip step 01's `pre-install-check.sh`

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

**This venv is CPU-only.** torch (cu130) and vLLM used to live here but were
removed — their coexistence with the training venv caused multi-GPU NCCL ABI
skew (vLLM #15525, #20862, #28283). GPU inference is the llama.cpp HTTP
server (step 14); this venv hosts the FastAPI / langchain / RAG glue around
it.

**Bootstrap:** pip, wheel, setuptools

**(historical, REMOVED — kept for archaeology) PyTorch + vLLM stack:**
- ~~torch==2.11.0+cu130~~
- ~~torchvision~~
- ~~torchaudio~~
- ~~vllm (latest, resolved against cu130 wheel index)~~

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
| `/etc/apt/sources.list.d/00-bundle.list` | `deb [trusted=yes] file:///var/tmp/airgap-bundle-debs ./` (written by step 03) |
| `/var/tmp/airgap-bundle-debs/` | Local file:// apt repo (copy of bundle's debs/). Uses bundled `Packages` if `dpkg-scanpackages` is not installed yet. |
| `apt-mark hold` | Step 03 adds holds ONLY on system runtime libs (`libstdc++6`, `libgcc-s1`, `libgomp1`, `libc6`) to prevent the userland install from downgrading the CUDA runtime's link target. The nvidia-*/cuda-*/libnvidia-* holds are owned by install-nvidia.sh; this script no longer duplicates them. |
| `/var/lib/install-all/system-libs-held.txt` | Audit list of system libs held by step 03 |

(The previous `/etc/apt/preferences.d/99-nvidia-prefer-origin` pin to `developer.download.nvidia.com` was removed: that origin is unreachable on airgap, and install-nvidia.sh's `99-nvidia-prefer-bundle` file:// pin already covers the nvidia/cuda package set.)

### Kernel / sysctl
| Path | Settings |
|------|----------|
| `/etc/sysctl.d/99-llm-multigpu.conf` | `vm.overcommit_memory=1`, `vm.swappiness=0`, `vm.max_map_count=1048576`, `net.core.rmem_max=268435456`, `net.core.wmem_max=268435456` |
| `/etc/sysctl.d/60-apparmor-userns.conf` | `kernel.apparmor_restrict_unprivileged_userns=0` (Chrome/VS Code sandbox on Ubuntu 24.04) |

### System limits
| Path | Settings |
|------|----------|
| `/etc/security/limits.d/99-llm-multigpu.conf` | `nofile=1048576`, `nproc=524288`, `memlock=unlimited`, `stack=65536` (both soft + hard, for `*`). pam_limits-only — step 15 will REMOVE any stale `/etc/systemd/system.conf.d/99-llm-multigpu.conf` from prior installs because DefaultLimit* there is unit-mismatched (bytes vs KB on LimitSTACK/MEMLOCK) and crashes systemd services. Per-service Limit*= belongs in unit files; see `llama-server@.service`. |

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

### State (transient + persistent)
| Path | What |
|------|------|
| `/var/lib/install-all/steps/NN-name.ok` | Step success marker. Written by `mark_step_ok` in each step script; the launcher (`install-all.sh`) consults these to skip already-completed steps on re-run. |
| `/var/lib/install-all/steps/NN-name.failed` | Step failure marker. Written by the EXIT trap in `00-common.sh` when a step exits non-zero. Step 17 / launcher report the first failed step by name. |
| `/var/lib/install-all/apt-requested.txt` | Step 04 dumps the apt-get install list (after t64 normalization) |
| `/var/lib/install-all/apt-proposed.txt` | Step 04 dumps every `(Inst|Conf)` line from `apt -s` simulation |
| `/var/lib/install-all/apt-reboot-triggers.txt` | Step 04 dumps reboot-triggering packages (libc6/systemd/dbus, +DKMS-danger if FORCE=1); step 05 reads this to decide whether to run Stage 1 |
| `/var/lib/install-all/system-libs-held.txt` | Step 03 audit list of system runtime libs held (libstdc++6/libgcc-s1/libgomp1/libc6) |
| `/var/lib/install-all/run-<RUN_ID>/warnings.log` | Per-run accumulator of step `warn` calls |
| `/var/lib/install-all/run-<RUN_ID>/errors.log` | Per-run accumulator of step `die` / ERR-trap calls |
| `/var/lib/install-all-prepped/stage1.done` | Resume marker, only present during conditional 2-stage reboot path. Steps 03/04/05 short-circuit when present; deleted by step 17 at end of install. |

### Logs (transient ??kept for debug)
| Path | What |
|------|------|
| `/var/log/install-all/<RUN_ID>/NN-name.log` | Per-step transcript (stdout+stderr via tee in `init_step`). One subdirectory per launcher invocation. |
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
| Optional host NCCL 2.x +cuda13.0 | ~250 MB only when `SKIP_NCCL=0` |
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
| Inference venv (CPU-only RAG/FastAPI) | ~600 MB |
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
- ~~CUDA toolkit~~ ??**minimal subset** (cuda-nvcc / cuda-cudart / cuda-cudart-dev / cuda-cccl / libcublas / libcublas-dev / libnvjitlink / cuda-compat — all `-13-0`). The `cuda-toolkit-13-0` metapackage is DELIBERATELY skipped (see Section 0.1).
- ~~nvidia-fabricmanager / nvlsm / libnvidia-nscq~~ ??bundled. (`libnvsdm` has no R580 variant in NVIDIA's apt repo and isn't needed for NVSwitch fabric — fabricmanager + libnvidia-nscq own that side.)
- ~~nvidia-persistenced / nvidia-dcgm services~~ ??enabled. `nvidia-persistenced` rides transitively with `nvidia-driver-580-open` (no separate apt name).
- ~~libnccl2~~ ??optional host package with `+cuda13.0` strict pin. **SKIP_NCCL=1 by default** at gather and install time. System NCCL is opt-in (`SKIP_NCCL=0` in both scripts); the default avoids ABI skew with PyTorch/vLLM venv-bundled NCCL.
- ~~Kernel modules / DKMS~~ ??open kmod, no DKMS path needed (Secure Boot off)
- ~~nouveau blacklist~~ ??`/etc/modprobe.d/blacklist-nouveau-nvidia.conf`
- ~~nvidia-peermem.ko~~ ??ships inside `nvidia-driver-580-open` and autoloads via `/etc/modules-load.d/nvidia-peermem.conf` written by install-nvidia.sh step 4b. (There is no separate `nvidia-peermem-loader` apt package.)
- ~~CUDA env wiring~~ ??`/etc/profile.d/cuda.sh` (PATH only) and `/etc/ld.so.conf.d/cuda-system.conf` (`/usr/local/cuda/lib64` in ld.so.cache) — written by install-nvidia.sh step 5b.
- ~~Fabric state Completed~~ ??verified by `test-nvidia.sh` (now FAILs on a multi-GPU box if the Fabric stanza is absent; SKIP only on single-GPU dev hosts or via `ALLOW_NO_FABRIC=1`).

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
- ~~**vLLM cache (`VLLM_CACHE_ROOT`)**~~ ??not applicable; vLLM is not installed.
- **Global `pip.conf`** ??not created. The venvs install offline from `--find-links=` so no index config needed; if you add wheels later, you'll want a `~/.pip/pip.conf` with `--no-index` defaults.

### 5. Monitoring / observability
- **Prometheus `node_exporter`** ??NOT installed.
- **DCGM exporter** ??NOT installed (DCGM itself is installed by the NVIDIA bundle when available, but install-nvidia.sh treats it as non-fatal/optional).
- **Grafana / Loki / Telegraf / Fluentd** ??NOT installed.
- **GPU benchmark baseline** ??install-all.sh does NOT install nccl-tests or any bandwidth tool. To verify NVSwitch is healthy, install `nccl-tests` manually from NVIDIA (`https://github.com/NVIDIA/nccl-tests`) ??it builds against `nvidia-nccl-cu13` from the PyTorch wheel, or against any libnccl the vendor already has on disk.
- **System-wide NCCL tuning** ??`/etc/profile.d/nccl-multigpu.sh` is NO LONGER written. PyTorch / vLLM use NCCL's defaults. If you want `NCCL_NVLS_ENABLE=1` for in-fabric reductions on B300 NVSwitch, set it in your job submission script or `~/.bashrc`.

### 6. Multi-node / HPC
- **InfiniBand / DOCA-OFED stack** ??vendor-installed (DOCA 3.2+ / OFED 25.10+ on the verified box); pre-install-nvidia.sh verifies. `nvidia-peermem.ko` (ships transitively with `nvidia-driver-580-open` — no separate `nvidia-peermem-loader` package) is autoloaded via `/etc/modules-load.d/nvidia-peermem.conf` and hooks into mlx5 for GPUDirect RDMA.
- **NCCL** ??host `libnccl2` is optional (`SKIP_NCCL=0` in both gather/install) with `+cuda13.0` strict pin. No system-wide env vars set; PyTorch wheels carry `nvidia-nccl-cu13` independently.
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
9. (Optional) For bandwidth verification, build nccl-tests manually from `https://github.com/NVIDIA/nccl-tests` against PyTorch's bundled `nvidia-nccl-cu13`, or against host `libnccl2 +cuda13.0` only if you installed it with `SKIP_NCCL=0`.
10. Set up `HF_HOME` and any other cache env vars in `/etc/profile.d/`
11. For multi-GPU training jobs that USE NCCL (training venv), export `NCCL_NVLS_ENABLE=1 NCCL_P2P_LEVEL=NVL` in your job script (these are no longer set globally). For llama.cpp / inference workloads NCCL is not on the path.
12. If you want monitoring: install node_exporter and DCGM exporter (out of scope here)
13. Reboot once before declaring production-ready (ensures the THP service, sysctl, and any deferred kmod loads come up cleanly from cold start)

---

*Generated alongside the scripts in this directory. Update when any of
`gather-all.sh`, `install-all.sh`, or `pre-install-check.sh` changes.*
